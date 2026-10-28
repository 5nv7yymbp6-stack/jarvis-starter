"""Memory 2.0 retrieval and ranking (Phase 4): given a query and the persistent memories, which are relevant, valid,
current, non-redundant and safe to put in front of Jarvis?

PURE and OFFLINE. No file, no socket, no model, no embedding service, no wall clock (the caller passes `now`), no
state, no randomness. It never changes what it is handed (items are frozen, everything is copied) and the same
inputs give the same Result, in any input order. Its only Jarvis imports are the pure core (backtalk.memory) and
the pure gate screens (backtalk.memory_gate). Nothing in the running system imports it yet.

    retrieve(query, items, *, now, context_items=(), limit=4, max_chars=8000,
             include_history=False, providers=()) -> Result

THE PIPELINE (a fixed order; the first stage that rejects an item decides, and its code is recorded):
    INTAKE          malformed item, duplicate id                              -> malformed, duplicate_id, id_conflict
    RESOLUTION      Phase 1 resolve(items, now) is the ONLY definition of "current":
                      current mode (default) keeps only resolve().active;
                      history mode (include_history=True, explicit) keeps active + superseded + ended items;
                      archived, not-yet-valid, shadowed, tied (conflict) and unresolvable items never come back
                      in either mode                    -> archived, not_yet_valid, shadowed, conflict, unresolvable ...
    SAFETY          never-expose, credential-like, injection-like, live Situation state, untrusted instruction,
                    unusable evidence. Applied AFTER resolution, so a hidden current fact never lets an older one
                    take its place                                           -> never_expose, credential_like ...
    DEDUPE          same key + type + value + validity period: the lowest id stands          -> duplicate
    RELEVANCE       the hybrid score below (lexical overlap, semantic score, key match); an item with none of the
                    three scores 0, and anything under MIN_RELEVANCE is dropped             -> not_relevant
    ADJUSTMENT      authority and recency, bounded multipliers on relevance (never a replacement for it)
    ORDER           by score; history mode puts each logical key's items in time order, newest first
    EVIDENCE DEDUPE the same value from the same words, under different keys: the better-ranked one   -> duplicate_evidence
    CONTEXT         a memory the supplied context already says is left out                  -> already_in_context
    BUDGET          at most `limit` results and `max_chars` characters; an item that does not fit is SKIPPED,
                    never truncated                                              -> over_limit, over_budget

SCORING (every number is in [0, 1], rounded to 6 places; the whole model is these lines):
    lexical   = sum over distinct query words (stopwords removed) of the best field weight the word hits, / word count
                field weights: attribute 1.0, value 1.0, evidence 0.6, subject 0.5. Exact words only: no stemming.
                A subject-only word counts only when another word matched elsewhere (the subject is on every memory:
                it can disambiguate a match, never create one).
    semantic  = the score a provider gave the memory id (0 when there is no provider or no score)
    key       = max(built-in key match, provider key hint). Built-in: every word of attribute AND subject in the query
                = 1.0 ("user.residence", "user residence"); every word of the attribute alone = 0.7
    relevance = rel  = 1 - (1 - lexical) * (1 - semantic)                                   (noisy-OR)
                relk = 1 - (1 - rel) * (1 - KEY_WEIGHT * key)                  (KEY_WEIGHT 0.7: a strong boost)
    score     = relk * (0.85 + 0.10 * authority + 0.05 * recency)
                authority: stated 1, inferred 0.   recency (current mode only): 1 / (1 + age_days / 365) from valid_from,
                0 in history mode. So authority moves a score by at most 10% and recency by at most 5% OF its relevance:
                a stated memory that is irrelevant stays irrelevant; with equal relevance, stated wins.
    ties break on the newer valid_from, then id (authority is already inside the score).

TIME. Validity at `now` is a HARD filter (valid_from <= now < valid_to, Phase 1). Only after that does recency matter,
and it is measured from valid_from (when the fact became true), never from `updated` (when the record was edited) or
`created` (when the record was made). Editing a memory does not make its fact newer.

PERSISTENT MEMORY ONLY. Live Situation state (mic, device, session, running task, now-playing ...) is not memory; an
item that names it is refused here even if a caller supplies it. There is no bridge from Situation to this module.

CANDIDATE PROVIDERS. Where semantic scores and key hints COME FROM is separate from what SHOULD be returned. A provider
is any callable `provider(query, docs) -> ProviderSignals`; it sees only items that already passed the filters (so an
embedding service never sees a never-expose memory), can only SCORE ids it was shown (it cannot introduce a memory),
and a provider that raises or returns junk is ignored and noted. No provider is required. EmbeddingProvider is an
optional adapter over any embed function; it scores a memory only when its stored vector matches the memory's current
content, so a stale index is detected, reported and not trusted.

DIAGNOSTICS carry ids and bounded codes only: never a value, never evidence, never a model's reasoning.
"""
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backtalk import memory as M
from backtalk import memory_gate as G

DEFAULT_LIMIT = 4
DEFAULT_MAX_CHARS = 8000
LIMIT_CAP = 20
MAX_CHARS_CAP = 20000
QUERY_MAX = 2000

W_ATTRIBUTE, W_VALUE, W_EVIDENCE, W_SUBJECT = 1.0, 1.0, 0.6, 0.5
KEY_WEIGHT = 0.7
KEY_ATTRIBUTE_STRENGTH = 0.7
MIN_RELEVANCE = 0.10
SCORE_BASE, AUTHORITY_WEIGHT, RECENCY_WEIGHT = 0.85, 0.10, 0.05
RECENCY_SCALE_DAYS = 365.0

REASONS = ("key_exact", "key_attribute", "key_provider", "lexical", "semantic", "stated", "inferred", "current",
           "history_superseded", "history_ended")
EXCLUSION_CODES = (
    "malformed", "duplicate_id", "id_conflict", "unresolvable", "archived", "not_yet_valid", "ended", "superseded",
    "shadowed", "duplicate", "conflict", "never_expose", "credential_like", "injection_like", "situation_state",
    "untrusted_instruction", "invalid_evidence", "not_relevant", "duplicate_evidence", "already_in_context",
    "over_limit", "over_budget")

STOPWORDS = G.FILLER_WORDS | frozenset({
    "where", "when", "who", "whom", "which", "why", "does", "of", "in", "on", "for", "at", "by", "with", "about",
    "from", "tell", "that", "this", "there", "be", "am", "was", "were", "been", "have", "has", "had", "his", "her",
    "their", "our", "your", "as", "if", "than", "then", "not", "now", "currently", "current", "still", "today",
    "before", "previously", "formerly", "ago", "m", "re", "ve", "ll", "d", "t"})


# ---- the results ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    key: str
    source_kind: str
    source_ref: str
    confidence: str
    sensitivity: str
    status: str
    valid_from: str
    valid_to: object
    created: str
    updated: str


@dataclass(frozen=True)
class Hit:
    id: str
    item: object                    # the memory.MemoryItem, unchanged
    evidence: str                   # the evidence stored with it, whole (never cut)
    rank: int                       # 1-based
    score: float                    # final, in [0, 1]
    relevance: float                # before the authority / recency adjustment
    lexical: float
    semantic: float
    key_match: float
    current: bool                   # true at `now` by Phase 1 resolve; False only for a history-mode result
    reasons: tuple                  # bounded codes from REASONS
    provenance: Provenance
    chars: int                      # what it costs against max_chars


@dataclass(frozen=True)
class Result:
    hits: tuple
    excluded: tuple                 # (memory id or "#index", code), sorted; includes every item not returned
    notes: tuple                    # provider codes: "provider0:error", "provider0:stale_index:2" ...
    mode: str                       # "current" | "history"
    query: str                      # the normalized query actually used
    now: str
    chars_used: int

    def to_dict(self) -> dict:
        return {"mode": self.mode, "query": self.query, "now": self.now, "chars_used": self.chars_used,
                "notes": list(self.notes), "excluded": [list(e) for e in self.excluded],
                "hits": [{"id": h.id, "rank": h.rank, "score": h.score, "relevance": h.relevance,
                          "lexical": h.lexical, "semantic": h.semantic, "key_match": h.key_match,
                          "current": h.current, "reasons": list(h.reasons), "chars": h.chars,
                          "evidence": h.evidence, "item": M.to_dict(h.item)} for h in self.hits]}


@dataclass(frozen=True)
class Doc:
    """What a provider is shown about one memory that passed every filter."""
    id: str
    key: str
    value: str
    evidence: str


@dataclass(frozen=True)
class ProviderSignals:
    semantic: object = None         # {memory id: score in [0, 1]}
    keys: object = None             # {logical key like "user.residence": strength in [0, 1]}
    stale: tuple = ()               # ids the provider could not trust (reported, not scored)


# ---- text ----------------------------------------------------------------------------------------

def _tokens(text) -> list:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(text)).casefold())


def normalize_query(text) -> str:
    """Lowercase, Unicode-normalized, punctuation and runs of whitespace reduced to single spaces. Words are never
    stemmed, dropped or rewritten, so the question asked is the question searched. ValueError if not a bounded str."""
    if not isinstance(text, str):
        raise ValueError("query must be a string")
    if len(text) > QUERY_MAX:
        raise ValueError("query is too long")
    return " ".join(_tokens(text))


def _contains_run(haystack, needle) -> bool:
    """Does `needle` (a non-empty list of words) appear in `haystack` as one unbroken run, in order?"""
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack)))


# ---- time ----------------------------------------------------------------------------------------

def _to_utc(at):
    if isinstance(at, datetime):
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("time must be timezone-aware")
        return at.astimezone(timezone.utc)
    t = M.parse_ts(at)
    if t is None:
        raise ValueError("time must be YYYY-MM-DDTHH:MM:SSZ or an aware datetime")
    return t


# ---- intake --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    id: str
    item: object
    evidence: str
    key: str


def _label(raw, index) -> str:
    """An id that is safe to print: only a well-formed memory id is ever shown, anything else is its position."""
    ident = getattr(raw, "id", None)
    if ident is None:
        ident = getattr(getattr(raw, "item", None), "id", None)
    return ident if isinstance(ident, str) and M._ID_RE.match(ident) else f"#{index}"


def _unwrap(raw):
    """(MemoryItem, evidence) for a MemoryItem or a stored memory (anything with .item and .evidence), else None."""
    if isinstance(raw, M.MemoryItem):
        return raw, ""
    item, ev = getattr(raw, "item", None), getattr(raw, "evidence", None)
    if isinstance(item, M.MemoryItem) and isinstance(ev, str):
        return item, ev
    return None


def _intake(items, excluded):
    """Valid entries, one per id. A repeated identical memory is kept once; one id with two different contents is
    ambiguous and neither is kept."""
    found = {}
    for index, raw in enumerate(items):
        pair = _unwrap(raw)
        if pair is None or M.validate(pair[0]):
            excluded.append((_label(raw, index), "malformed"))
            continue
        found.setdefault(pair[0].id, []).append(pair)
    entries = []
    for ident in sorted(found):
        group = found[ident]
        if any(p != group[0] for p in group):
            excluded.extend((ident, "id_conflict") for _ in group)
            continue
        excluded.extend((ident, "duplicate_id") for _ in group[1:])
        entries.append(_Entry(ident, group[0][0], group[0][1], M.key_of(group[0][0])))
    return entries


# ---- safety --------------------------------------------------------------------------------------

def _unsafe(e):
    """A code if this entry must never be returned, whoever passed it in; else None."""
    it = e.item
    if it.sensitivity == "never-expose":
        return "never_expose"
    hit = G.screen_item(it, e.evidence)
    if hit:
        return hit[0]
    if it.subject in G.TEMPORARY_NAMES or it.attribute in G.TEMPORARY_NAMES or G._TEMPORARY_PHRASES.search(it.value):
        return "situation_state"
    if it.type == "instruction" and (it.source.kind != "user" or it.confidence != "stated"):
        return "untrusted_instruction"
    if e.evidence and not G.evidence_usable(e.evidence):
        return "invalid_evidence"
    return None


# ---- providers -----------------------------------------------------------------------------------

def _clean_scores(scores, allowed) -> dict:
    out = {}
    if not hasattr(scores, "items"):
        return out
    for k, v in scores.items():
        if k in allowed and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[k] = min(1.0, max(0.0, float(v)))
    return out


def _gather(providers, query, entries, notes):
    """(semantic by id, key hints by key) merged across providers by the highest score. A provider only scores what it
    is shown; a provider that raises or answers with anything but ProviderSignals contributes nothing."""
    docs = tuple(Doc(e.id, e.key, e.item.value, e.evidence) for e in sorted(entries, key=lambda e: e.id))
    ids, keys = {e.id for e in entries}, {e.key for e in entries}
    semantic, hints = {}, {}
    for n, provider in enumerate(providers):
        try:
            sig = provider(query, docs)
        except Exception:                                    # a provider's failure is never allowed to hide memory
            notes.append(f"provider{n}:error")
            continue
        if not isinstance(sig, ProviderSignals):
            notes.append(f"provider{n}:invalid")
            continue
        for k, v in _clean_scores(sig.semantic, ids).items():
            semantic[k] = max(semantic.get(k, 0.0), v)
        for k, v in _clean_scores(sig.keys, keys).items():
            hints[k] = max(hints.get(k, 0.0), v)
        stale = tuple(sig.stale) if isinstance(sig.stale, (tuple, list)) else ()
        if stale:
            notes.append(f"provider{n}:stale_index:{len(stale)}")
    return semantic, hints


def doc_fingerprint(doc) -> str:
    """What a stored vector must have been made from: the key, value and evidence, kept apart by repr's quoting so
    text cannot slide from one field into the next (value 'x y' + evidence 'z' is not value 'x' + evidence 'y z')."""
    return M.fingerprint(repr((doc.key, doc.value, doc.evidence)))


def _cosine(a, b) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingProvider:
    """OPTIONAL adapter: semantic scores from any `embed(text) -> sequence of numbers` and a prebuilt
    {memory id: (fingerprint, vector)} index (see build_vectors). It scores a memory only when the vector was made
    from the memory's CURRENT content; a missing or stale vector is reported in `stale` and not trusted, which is how
    an out-of-date index is kept from silently steering retrieval. No network is touched here: `embed` is the caller's."""

    def __init__(self, embed, vectors):
        self._embed, self._vectors = embed, dict(vectors)

    def __call__(self, query, docs) -> ProviderSignals:
        qv = [float(x) for x in self._embed(query)]
        semantic, stale = {}, []
        for d in docs:
            rec = self._vectors.get(d.id)
            if rec is None or rec[0] != doc_fingerprint(d):
                stale.append(d.id)
                continue
            s = _cosine(qv, [float(x) for x in rec[1]])
            if s > 0:
                semantic[d.id] = min(1.0, s)
        return ProviderSignals(semantic=semantic, stale=tuple(stale))


def build_vectors(embed, docs) -> dict:
    """The index EmbeddingProvider reads: {id: (fingerprint of the content it was made from, vector)}."""
    return {d.id: (doc_fingerprint(d), tuple(float(x) for x in embed(f"{d.key} {d.value} {d.evidence}")))
            for d in docs}


# ---- scoring -------------------------------------------------------------------------------------

def _round(x) -> float:
    return round(x, 6)


def _lexical(words, e) -> float:
    """Weighted word coverage. A word that matches only the SUBJECT counts only when another word matched elsewhere:
    the subject ("user") is on every memory, so it may disambiguate a match but can never create one."""
    if not words:
        return 0.0
    it = e.item
    fields = ((W_ATTRIBUTE, set(_tokens(it.attribute))), (W_VALUE, set(_tokens(it.value))),
              (W_EVIDENCE, set(_tokens(e.evidence))))
    subject = set(_tokens(it.subject))
    total = sum(max((w for w, toks in fields if word in toks), default=0.0) for word in words)
    if total > 0:
        total += sum(W_SUBJECT for word in words
                     if word in subject and not any(word in toks for _, toks in fields))
    return total / len(words)


def _key_match(qset, e) -> float:
    attr = set(_tokens(e.item.attribute))
    if not attr or not attr <= qset:
        return 0.0
    return 1.0 if set(_tokens(e.item.subject)) <= qset else KEY_ATTRIBUTE_STRENGTH


def _auth(item) -> int:
    return 1 if item.confidence == "stated" else 0


def _chars(e) -> int:
    return len(e.key) + len(e.item.value) + len(e.evidence)


# ---- context -------------------------------------------------------------------------------------

def _context_index(context_items):
    ids, triples, texts = set(), set(), []
    for c in context_items:
        if isinstance(c, str):
            texts.append((c.casefold(), _tokens(c)))
            continue
        pair = _unwrap(c)
        if pair is None:
            raise ValueError("context items must be text or memory items")
        ids.add(pair[0].id)
        triples.add((M.key_of(pair[0]), pair[0].type, M.normalize_value(pair[0].value)))
    return ids, triples, texts


def _in_context(e, ctx) -> bool:
    """Does the context already say this memory? By id or by (key, type, value) for memory items; for text, the whole
    value must appear as a run of words AND a word that anchors it to this memory (its subject, attribute or evidence)
    must appear too, so "visited San Jose" does not hide a memory about a different fact that happens to share a word."""
    ids, triples, texts = ctx
    it = e.item
    if e.id in ids or (e.key, it.type, M.normalize_value(it.value)) in triples:
        return True
    value = _tokens(it.value)
    by_words = not all(w in STOPWORDS for w in value)            # a value of only stopwords ("no") cannot be matched by words
    anchors = (set(_tokens(it.subject)) | set(_tokens(it.attribute)) | set(_tokens(e.evidence))) \
        - set(value) - STOPWORDS
    return any(e.id in raw or (by_words and _contains_run(toks, value) and anchors & set(toks)) for raw, toks in texts)


# ---- retrieval -----------------------------------------------------------------------------------

def _count(name, value, cap):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= cap:
        raise ValueError(f"{name} must be an integer from 0 to {cap}")
    return value


def retrieve(query, items, *, now, context_items=(), limit=DEFAULT_LIMIT, max_chars=DEFAULT_MAX_CHARS,
             include_history=False, providers=()) -> Result:
    """The memories to put in front of Jarvis for `query`, as of `now`. See the module docstring for the pipeline.

    `items`: MemoryItems or stored memories (anything with .item and .evidence, like the repository's stored record). Pass the
    WHOLE repository, not a pre-filtered slice: an item whose `supersedes` link points at an id that is missing is
    unresolvable and left out. `context_items`: text and/or memory items already in the prompt. `providers`: a
    callable or a sequence of callables (see ProviderSignals); none is needed. Arguments that are not what they say
    (a non-bool include_history, a limit out of range, a query that is not text, a `now` that is not a time) are a
    ValueError, never a guess. Never raises for any item. Nothing passed in is changed."""
    t = _to_utc(now)
    if include_history is not True and include_history is not False:
        raise ValueError("include_history must be True or False")
    limit, max_chars = _count("limit", limit, LIMIT_CAP), _count("max_chars", max_chars, MAX_CHARS_CAP)
    qn = normalize_query(query)
    if callable(providers):
        providers = (providers,)
    providers = tuple(providers)
    ctx = _context_index(tuple(context_items))
    excluded, notes = [], []

    # INTAKE, then RESOLUTION: Phase 1 decides what is true at `now`; this module adds no second opinion.
    entries = _intake(tuple(items), excluded)
    res = M.resolve([e.item for e in entries], t)
    unresolved = {i for i, _ in res.rejected}
    active = {x.id for x in res.active}
    conflicted = {x.id for c in res.conflicts for x in c.candidates}
    why = {x.id: r for x, r in res.history}
    eligible = []
    for e in entries:
        if e.id in unresolved:
            excluded.append((e.id, "unresolvable"))
        elif e.id in conflicted:
            excluded.append((e.id, "conflict"))
        elif e.id in active:
            eligible.append(e)
        elif include_history and why[e.id] in ("superseded", "ended"):
            eligible.append(e)
        else:
            excluded.append((e.id, why[e.id]))

    # SAFETY, then DEDUPE (the lowest id of identical key + type + value + period stands)
    safe, seen = [], set()
    for e in eligible:
        code = _unsafe(e)
        if code:
            excluded.append((e.id, code))
            continue
        sig = (e.key, e.item.type, M.normalize_value(e.item.value), e.item.valid_from, e.item.valid_to)
        if sig in seen:
            excluded.append((e.id, "duplicate"))
        else:
            seen.add(sig)
            safe.append(e)

    # CANDIDATES AND RELEVANCE
    words = [w for w in dict.fromkeys(_tokens(qn)) if w not in STOPWORDS]
    qset = set(_tokens(qn))
    semantic, hints = _gather(providers, qn, safe, notes)
    scored = []
    for e in safe:
        lex, sem = _lexical(words, e), semantic.get(e.id, 0.0)
        built_in, hinted = _key_match(qset, e), hints.get(e.key, 0.0)
        key = max(built_in, hinted)
        rel = 1 - (1 - lex) * (1 - sem)
        relk = _round(1 - (1 - rel) * (1 - KEY_WEIGHT * key))
        if relk < MIN_RELEVANCE:
            excluded.append((e.id, "not_relevant"))
            continue
        age = (t - M.parse_ts(e.item.valid_from)) / timedelta(days=1)          # >= 0: resolution kept only valid_from <= now
        recency = 0.0 if include_history else 1.0 / (1.0 + age / RECENCY_SCALE_DAYS)
        score = _round(relk * (SCORE_BASE + AUTHORITY_WEIGHT * _auth(e.item) + RECENCY_WEIGHT * recency))
        reasons = []
        if built_in > 0:
            reasons.append("key_exact" if built_in == 1.0 else "key_attribute")
        if hinted > 0:
            reasons.append("key_provider")
        if lex > 0:
            reasons.append("lexical")
        if sem > 0:
            reasons.append("semantic")
        reasons.append(e.item.confidence)
        reasons.append("current" if e.id in active else "history_" + why[e.id])
        scored.append((e, _round(lex), _round(sem), _round(key), relk, score, tuple(reasons)))

    # ORDER: by score (which already carries authority and, in current mode, recency), then the newer fact, then id
    scored.sort(key=lambda s: (-s[5], -M.parse_ts(s[0].item.valid_from).timestamp(), s[0].id))
    if include_history:                       # each key's items keep their true time order, in the slots they hold
        slots = {}
        for pos, s in enumerate(scored):
            slots.setdefault(s[0].key, []).append(pos)
        ordered = list(scored)
        for positions in slots.values():
            chrono = sorted((scored[p] for p in positions),
                            key=lambda s: (M.parse_ts(s[0].item.valid_from), M.parse_ts(s[0].item.created), s[0].id),
                            reverse=True)
            for p, s in zip(positions, chrono):
                ordered[p] = s
        scored = ordered

    # EVIDENCE DEDUPE, CONTEXT, BUDGET
    hits, said, used = [], set(), 0
    for s in scored:
        e = s[0]
        words_said = (M.normalize_value(e.item.value), M.normalize_value(e.evidence))
        if e.evidence.strip() and words_said in said:
            excluded.append((e.id, "duplicate_evidence"))
            continue
        if _in_context(e, ctx):
            excluded.append((e.id, "already_in_context"))
        elif len(hits) >= limit:
            excluded.append((e.id, "over_limit"))
        elif _chars(e) > max_chars - used:
            excluded.append((e.id, "over_budget"))
        else:
            used += _chars(e)
            said.add(words_said)
            it = e.item
            prov = Provenance(e.key, it.source.kind, it.source.ref, it.confidence, it.sensitivity, it.status,
                              it.valid_from, it.valid_to, it.created, it.updated)
            hits.append(Hit(e.id, it, e.evidence, len(hits) + 1, s[5], s[4], s[1], s[2], s[3], e.id in active,
                            s[6], prov, _chars(e)))
    return Result(hits=tuple(hits), excluded=tuple(sorted(excluded)), notes=tuple(notes),
                  mode="history" if include_history else "current", query=qn,
                  now=t.strftime("%Y-%m-%dT%H:%M:%SZ"), chars_used=used)
