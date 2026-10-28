"""Memory 2.0 candidate extraction (Phase 6): what durable personal facts does Jarvis have EVIDENCE to propose, and
which of them need the user's confirmation? READ-ONLY and PROPOSE-ONLY.

PURE. No file, no socket, no model, no clock (the caller passes `generated_at`), no state. It imports only the pure
core (backtalk.memory) and the pure gate (backtalk.memory_gate). It never touches the store, the index or the vault:
the scanner (backtalk/evals/memory_extraction/scan.py) reads files and hands this module plain data, and writes the
review pack. Nothing in the running system imports it.

    classify_turn(entry)            one transcript line -> Turn (typed | voice) or an exclusion code
    build_candidates(turns, notes)  Turns + vault notes -> merged Candidates, each dry-run through the real gate
    known_conflicts(notes)          the seven Phase 5 conflicts, with the competing evidence side by side
    review_pack(...)                the deterministic, machine-readable pack (private values redacted)

WHO SAID IT (the provenance rules, approved in Phase 5):
    typed     a `claude-desktop` turn with no tool result, not meta, origin absent or `human`: the user's own words.
              -> source kind `user`, confidence `stated`.
    voice     an `sdk-py` turn (the voice line): speech-to-text, speaker NOT verified (his mother has been answered).
              Weak evidence. It becomes `user`/`stated` ONLY when independently corroborated: the same fact typed, or
              the same fact in voice turns of two different sessions on two different days. Otherwise the candidate is
              source kind `session`, confidence `inferred`, and needs confirmation.
    note      a vault note. Jarvis wrote it, so it is never the user's own words: source kind `note`, confidence
              `inferred`, always needs confirmation. A note is NEVER promoted to `stated`.
    Everything else (assistant text, tool results, injected recall/situation/board blocks, hooks, slash commands,
    pasted content, sub-agent prompts, scripted `sdk-cli` runs) is not evidence at all and is dropped.

TIME. A transcript turn's own timestamp is the only genuine instant (it becomes `valid_from`, meaning "known true
at this instant"; when the fact STARTED is unknown). A note carries a date at best, so a note candidate has NO
timestamp (`timestamp: null`); the gate dry-run uses `generated_at` as a labelled placeholder, and the real
`valid_from` would be the user's confirmation instant. No precision is manufactured.

SAFETY. The user's words are screened (credentials, injection) BEFORE anything is extracted; a message that trips the
screen is dropped whole and only counted, by category. Third-party contact details and relatives are never candidates.
Private facts (home address, money, health, family mentions) are redacted from the public pack. Evidence is a verbatim
substring of the cleaned source, at most 300 characters, never paraphrased or cut. The gate's rules stay the authority.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from backtalk import memory as M
from backtalk import memory_gate as G
from backtalk import memory_keys as K

CHANNEL_TYPED, CHANNEL_VOICE, CHANNEL_NOTE = "typed", "voice", "note"
PACK_FORMAT = "memory-review-pack/2"
VALUE_MAX = 200
REF_LIMIT = 5

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z\Z")

# ---- turns ---------------------------------------------------------------------------------------

_SYSTEM_TAGS = ("local-command-caveat", "command-name", "local-command-stdout", "local-command-stderr", "task-notification",
                "scheduled-task", "bash-input", "bash-stdout", "bash-stderr", "command-message", "command-args")
_INJECTED_PREFIXES = ("[From the vault", "[Holo board", "[Situation", "[Your previous response")
_BLOCK_RES = (re.compile(r"<pasted_content\b[^>]*>.*?</pasted_content\b[^>]*>", re.S),
              re.compile(r"<system-reminder>.*?</system-reminder>", re.S))
_NOISE_LINE = re.compile(r"(?m)^[ \t]*\[(?:Image:|Request interrupted)[^\n]*\][ \t]*$")


@dataclass(frozen=True)
class Turn:
    channel: str            # typed | voice
    text: str               # cleaned: only what the person said
    session: str
    uuid: str
    timestamp: object       # normalized `YYYY-MM-DDTHH:MM:SSZ`, or None when the line has no usable time
    file: str               # transcript file name (a stable reference, never a path to copy from)
    line: int               # 1-based line in that file
    text_sha: str           # sha256 of the cleaned text


def sha(text) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def norm_ts(value):
    """`YYYY-MM-DDTHH:MM:SSZ` for a strict ISO UTC timestamp (fractions dropped), else None. Never guesses."""
    if not isinstance(value, str):
        return None
    m = _TS.match(value)
    if not m:
        return None
    out = m.group(1) + "Z"
    return out if M.parse_ts(out) else None


def screen_text(text):
    """('credential_like', categories) | ('injection_like', ()) | None. Categories only, never the text."""
    cats = G.screen_secrets(text)
    if cats:
        return "credential_like", cats
    if isinstance(text, str) and G._INJECTION.search(text):
        return "injection_like", ()
    return None


def strip_injected(text) -> str:
    """Remove the blocks the voice line puts in FRONT of what he said (vault recall, board, situation, prior reply).
    A block ends at the first `\\n]\\n\\n` (else `]\\n\\n`); a block whose end cannot be found drops the whole message."""
    t = text
    while t.startswith(_INJECTED_PREFIXES):
        for end in ("\n]\n\n", "]\n\n"):
            i = t.find(end)
            if i != -1:
                t = t[i + len(end):]
                break
        else:
            return ""
    return t


def clean_text(text) -> str:
    t = text if isinstance(text, str) else ""
    for rx in _BLOCK_RES:
        t = rx.sub("", t)
    t = strip_injected(t.lstrip("\n"))
    return _NOISE_LINE.sub("", t).strip()


def classify_turn(entry, *, file="", line=0):
    """(Turn | None, exclusion code | None) for one parsed transcript line. Never raises for any shape."""
    if not isinstance(entry, dict) or entry.get("type") != "user":
        return None, "not_user"
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return None, "malformed"
    if entry.get("isMeta"):
        return None, "meta"
    if entry.get("isSidechain"):
        return None, "sidechain"
    if entry.get("isCompactSummary") or entry.get("isVisibleInTranscriptOnly"):
        return None, "compact_summary"
    if entry.get("toolUseResult") is not None or entry.get("sourceToolUseID"):
        return None, "tool_result"
    origin = entry.get("origin")
    if origin is not None and not (isinstance(origin, dict) and origin.get("kind") == "human"):
        return None, "origin_not_human"
    ep = entry.get("entrypoint")
    if ep == "claude-desktop":
        channel = CHANNEL_TYPED
    elif ep == "sdk-py":
        channel = CHANNEL_VOICE
    elif ep == "sdk-cli" and isinstance(origin, dict):
        channel = CHANNEL_TYPED
    else:
        return None, "scripted_or_unknown_entrypoint"
    content = msg.get("content")
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
        if any(b.get("type") == "tool_result" for b in blocks):
            return None, "tool_result"
        raw = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str))
    else:
        return None, "no_text"
    stripped = raw.lstrip()
    if stripped.startswith("<") and re.match(r"<(?:%s)\b" % "|".join(_SYSTEM_TAGS), stripped):
        return None, "system_tag"
    text = clean_text(raw)
    if not text:
        return None, "empty_after_clean"
    hit = screen_text(text)
    if hit:
        return None, "secret_like:" + ",".join(hit[1]) if hit[0] == "credential_like" else "injection_like"
    uuid, session = entry.get("uuid"), entry.get("sessionId")
    if not isinstance(uuid, str) or not isinstance(session, str) or not uuid or not session:
        return None, "malformed"
    return Turn(channel, text, session, uuid, norm_ts(entry.get("timestamp")), file, line, sha(text)), None


# ---- sentences, values, keys ---------------------------------------------------------------------

def sentences(text) -> list:
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln:
            out.extend(p for p in re.split(r"(?<=[.!?])\s+", ln) if p)
    return out


def norm(value) -> str:
    return M.normalize_value(value)


def slug(text, limit=3) -> str:
    ascii_ = unicodedata.normalize("NFKD", str(text)).replace("'", "").encode("ascii", "ignore").decode()
    words = [w for w in re.findall(r"[a-z0-9]+", ascii_.lower()) if w not in _SLUG_SKIP]
    return "_".join(words[:limit])[:40]


_SLUG_SKIP = frozenset({"a", "an", "the", "to", "of", "and", "or", "in", "on", "at", "my", "me", "i", "it", "is", "be", "that", "this"})
_TRIM = re.compile(r"\s+(?:because|but|so|which|since|although|though|until|and then|cause)\b.*$|\s+[—–-]\s+.*$|\s*\(.*$", re.I | re.S)
_BAD_START = re.compile(r"(?i)^(?:it|that|this|these|those|how|what|when|where|why|the way|to be|being|if|whether|the|a|an|your|you|my|him|her|them|his|their|some|any|all|one|not)\b")
_TEMPORARY_WORD = re.compile(r"(?i)\b(?:now|today|tonight|here|right now|at the moment)\b")


def trim_value(value):
    v = _TRIM.sub("", str(value)).strip(" \t,;:.-—'\"*_`")
    return v if 3 <= len(v) <= VALUE_MAX and re.search(r"[A-Za-z0-9]{2}", v) else None


_PHONE = re.compile(r"(?<!\d)(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_STREET = re.compile(r"\b\d{3,6}\s+(?:[NSEW]{1,2}\s+)?[A-Z0-9][\w]*(?:\s+\w+)?\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ter|Terrace|Way|Blvd|Ln|Lane|Ct)\b")
_RELATIVE = re.compile(r"(?i)\b(?:mom|mother|dad|father|uncle|aunt|brother|sister|wife|girlfriend|boyfriend|cousin|son|daughter|grandma|grandpa|friend|his|her|their)\b")
_FINANCE = re.compile(r"(?i)\$\s?\d|\b\d[\d,]*\s?(?:/|per\s)\s?(?:week|wk|month|mo|hour|hr)\b|\b(?:salary|income|savings|debt|paycheck|rent|mortgage|credit score|net worth)\b")
_HEALTH = re.compile(r"(?i)\b(?:hypno\w*|therapy|therapist|medication|diagnos\w*|anxiety|depress\w*|adhd|doctor|surgery)\b")


def contact_or_third_party(attribute, value, evidence):
    """Exclusion code if the VALUE is another person's contact detail or about a relative, else None. (A relative merely
    MENTIONED in the evidence only makes the candidate private; see sensitivity.)"""
    if _PHONE.search(value) or _EMAIL.search(value) or _PHONE.search(evidence) or _EMAIL.search(evidence):
        return "contact_detail"
    if attribute not in ("residence", "address") and (_STREET.search(value) or _STREET.search(evidence)):
        return "contact_detail"
    if _RELATIVE.search(value):
        return "third_party_value"
    return None


def sensitivity_of(attribute, value, evidence) -> str:
    """normal | private. (never-expose material is dropped before it can be a candidate.)"""
    if attribute in ("residence", "address") or _STREET.search(value) or _STREET.search(evidence):
        return "private"
    text = value + " " + evidence
    if _FINANCE.search(text) or _HEALTH.search(text) or _RELATIVE.search(text):
        return "private"
    return "normal"


# ---- the rules -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    name: str
    type: str
    attribute: object       # a fixed name, or "slug:<prefix>" meaning prefix + slug(value)
    regex: object
    channels: tuple         # which channels may use it


def _rx(p):
    return re.compile(p, re.I)


_ALL = (CHANNEL_TYPED, CHANNEL_VOICE)
_PREFER = _rx(r"\bI (?:really |always |much )?prefer (?P<v>[^.;!?\n]+)")
FIRST_PERSON = (
    Rule("residence", "fact", "residence",
         _rx(r"\b(?:I (?:live|stay|reside)|I(?:'m| am) (?:living|staying|residing)) (?:at|in|on|near) (?P<v>[^.;!?\n]+)"), _ALL),
    Rule("address", "fact", "residence", _rx(r"\bmy (?:home |current )?address is (?P<v>[^.;!?\n]+)"), _ALL),
    Rule("occupation", "identity", "occupation",
         _rx(r"\bI (?P<v>(?:wholesale|flip) (?:real estate|houses|properties|homes))\b"), _ALL),
    Rule("language", "fact", "language", _rx(r"\bI (?:speak|prefer speaking) (?P<v>[A-Za-z][A-Za-z ,]{2,40})"), _ALL),
    Rule("preference", "preference", "slug:pref_", _PREFER, _ALL),
    Rule("dislike", "preference", "slug:dislike_", _rx(r"\bI (?:don't like|do not like|dislike) (?P<v>[^.;!?\n]+)"), (CHANNEL_TYPED,)),
    Rule("instruction", "instruction", "slug:rule_",
         _rx(r"^(?:from now on|going forward|always|never)\b[,: ]+(?P<v>[^.;!?\n]{8,160})"), (CHANNEL_TYPED,)),
    Rule("goal", "goal", "slug:goal_",
         _rx(r"\bmy (?:main |long[- ]term )?(?:goal|plan|dream|target) (?:is|was) (?:to )?(?P<v>[^.;!?\n]+)"), _ALL),
    Rule("address_as", "preference", "address_as", re.compile(r"\b[Cc]all me (?P<v>[A-Z][a-z]{2,12})(?=[.,;!?]|$)"), (CHANNEL_TYPED,)),
    Rule("vehicle", "fact", "vehicle", _rx(r"\bI (?:drive|own) (?:a|an|the) (?P<v>[^.;!?\n]{3,50})"), _ALL),
    Rule("business_start", "fact", "business_start",
         _rx(r"\b(?:I started|started) (?:my )?(?:wholesaling|wholesale|real estate)[^.;!?\n]{0,30} in "
             r"(?P<v>(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})"), _ALL),
)


@dataclass(frozen=True)
class Raw:
    """One match, before screening and merging."""
    type: str
    attribute: str
    value: str
    evidence: str
    channel: str
    origin: str             # transcript | vault
    ref: str
    where: str              # transcript file or note path
    line: int
    text_sha: str
    timestamp: object
    session: str
    date: object            # YYYY-MM-DD of the turn (for the two-day corroboration rule), else None
    uuid: str = ""
    template_risk: bool = False     # from a section that may be shipped template defaults, not his own words
    date_hint: object = None        # a date visible in the evidence or the note's name: shown, never used as authority


def _attr(rule, value):
    if isinstance(rule.attribute, str) and rule.attribute.startswith("slug:"):
        s = slug(value)
        return rule.attribute[5:] + s if s else None
    return rule.attribute


def match_sentences(text, channel):
    """[(rule, value, sentence)] for first-person statements in what a person said. Questions are never statements."""
    out = []
    for sent in sentences(text):
        if "?" in sent or len(sent) > G.EVIDENCE_MAX:
            continue
        for rule in FIRST_PERSON:
            if channel not in rule.channels:
                continue
            m = rule.regex.search(sent)
            if not m:
                continue
            v = trim_value(m.group("v"))
            if not v or (rule.name in ("preference", "dislike", "instruction", "goal") and (_BAD_START.match(v) or _TEMPORARY_WORD.search(v))):
                continue
            if rule.name == "language" and not v[0].isupper():
                continue
            if rule.name in ("residence", "address") and not re.search(r"[A-Z0-9]", v):
                continue
            if rule.name == "instruction" and re.search(r"[`\"“”]|https?://", sent):
                continue
            out.append((rule, v, sent))
    return out


def raws_from_turn(turn, exclusions) -> list:
    out = []
    # short and stable: file prefix + line + the turn's own time + a fingerprint of the cleaned text. (A full UUID would
    # itself look like an opaque credential to the gate's screen, so the full ids travel in `session`/`uuid` instead.)
    ref = f"transcript:{turn.file[:8]}#L{turn.line}@{turn.timestamp or 'untimed'}/{turn.text_sha[:8]}"
    for rule, v, sent in match_sentences(turn.text, turn.channel):
        attr = _attr(rule, v)
        if not attr or not M._NAME_RE.match(attr):
            exclusions["no_usable_key"] += 1
            continue
        out.append(Raw(rule.type, attr, v, sent, turn.channel, "transcript", ref, turn.file, turn.line, turn.text_sha,
                       turn.timestamp, turn.session, turn.timestamp[:10] if turn.timestamp else None, turn.uuid, False,
                       turn.timestamp[:10] if turn.timestamp else None))
    return out


# ---- vault notes ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    path: str               # path relative to the vault
    text: str
    sha: str                # sha256 of the whole note at scan time


_H2 = re.compile(r"(?m)^##\s+(.*?)\s*$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_LEAD = re.compile(r"^\*\*(?P<lead>.+?)\*\*")
_DAILY_RX = re.compile(r"^01 - Daily Notes/.+/\d{4}-\d{2}-\d{2}\.md\Z")
# (type, attribute, line regex, path regex): what each eligible note is allowed to yield, by explicit rule
_THIRD = (
    ("fact", "residence", re.compile(r"\bHome address recorded:\s*(?P<v>[^—\n]+)"), _DAILY_RX),
    ("fact", "residence", re.compile(r"\bhe lives at (?P<v>[^.\n]+)"), re.compile(r"^03 - Markets/Kansas City \(Missouri\)\.md\Z")),
    ("fact", "business_start", re.compile(r"\(started (?P<v>[A-Z][a-z]{2,8}\.? \d{4})\)"), _DAILY_RX),
    ("fact", "business_start", re.compile(r"\bstarted (?P<v>February 2026)\b"), re.compile(r"^04 - Wholesaling/04 - Wholesaling\.md\Z")),
    ("identity", "occupation", re.compile(r"\bYousif is running (?P<v>a [^(.—]+)"), _DAILY_RX),
    ("goal", "operating_focus", re.compile(r"Chosen market: \*\*(?P<v>[^*]+)\*\*"), re.compile(r"^04 - Wholesaling/04 - Wholesaling\.md\Z")),
    ("goal", "new_ventures_priority",
     re.compile(r"Decided 2026-09-20[^:\n]*: (?P<v>Shopify is the priority, AI Agency is parked)"),
     re.compile(r"^10 - New Ventures/10 - New Ventures\.md\Z")),
)
_NOTE_SECTIONS = {"Who I Am": "who", "What I Want": "want", "My Preferences for Working with AI": "prefs"}
def _note_raw(note, atype, attr, value, evidence, lineno, template_risk=False) -> Raw:
    return Raw(atype, attr, value, evidence, CHANNEL_NOTE, "vault", f"vault:{note.path}#L{lineno}", note.path, lineno,
               sha(evidence), None, "", None, "", template_risk, _date_hint(evidence, note.path))


def _prefixed(prefix, value, limit=3):
    s = slug(value, limit)
    return prefix + s if s and M._NAME_RE.match(prefix + s) else None


def _index_raws(note, exclusions) -> list:
    """VAULT-INDEX.md: Who I Am (first person), What I Want (goals) and the bold lead of each working-style bullet."""
    out = []
    section = None
    for i, line in enumerate(note.text.split("\n"), 1):
        h = _H2.match(line)
        b = _BULLET.match(line)
        if h:
            section = _NOTE_SECTIONS.get(h.group(1))
        elif section == "prefs" and b and _LEAD.match(b.group(1)):
            text = _LEAD.match(b.group(1)).group("lead").strip()
            v = text.rstrip(".")
            first = text.startswith("I ")
            attr = _prefixed("pref_" if first else "rule_", v)
            said = _PREFER.search(text) if first else None
            if said and trim_value(said.group("v")):             # "I prefer X": the SAME key a spoken "I prefer X" gets
                v, attr = trim_value(said.group("v")), _prefixed("pref_", said.group("v"))
            if attr and len(text) <= G.EVIDENCE_MAX:
                out.append(_note_raw(note, "preference" if first else "instruction", attr, v, text, i, template_risk=True))
            else:
                exclusions["no_usable_key"] += 1
        elif section == "want" and b:
            text = b.group(1).strip()
            v = trim_value(text.rstrip("."))
            attr = _prefixed("goal_", v, 4) if v else None
            if attr:                                  # (a value is at most VALUE_MAX characters, well inside the evidence limit)
                out.append(_note_raw(note, "goal", attr, v, text, i))
            else:
                exclusions["no_usable_key"] += 1
        elif section == "who":
            for rule, v, sent in match_sentences(line, CHANNEL_TYPED):
                attr = _attr(rule, v)
                if rule.name == "occupation" and attr:
                    out.append(_note_raw(note, rule.type, attr, v, sent, i))
    return out


def _rule_raws(note, exclusions) -> list:
    """Every other eligible note: only its own explicit rules, and in a daily note only inside `### Profile Updates`."""
    out = []
    daily = bool(_DAILY_RX.match(note.path))
    in_profile = False
    for i, line in enumerate(note.text.split("\n"), 1):
        if daily and line.startswith("### "):
            in_profile = line.strip() == "### Profile Updates"
        elif not daily or in_profile:
            for atype, attr, rx, path_rx in _THIRD:
                m = rx.search(line) if path_rx.match(note.path) else None
                v = trim_value(m.group("v")) if m else None
                if not v:
                    continue
                sent = next((x for x in sentences(line) if m.group(0) in x), line.strip())
                if len(sent) <= G.EVIDENCE_MAX:
                    out.append(_note_raw(note, atype, attr, v, sent, i))
                else:
                    exclusions["evidence_too_long"] += 1
    return out


# ---- approved inbox statements ---------------------------------------------------------------------------------------------------
# The nightly capture proposes statements into `00 - Inbox/Memory Candidates.md`; ticking one is Yousif's approval to PROMOTE it, and
# `promote.py` appends the ticked line, verbatim, to this APPEND-ONLY note (its line numbers never shift, so reviewed candidates cannot go
# stale). It is read here as one more NOTE source, which means every such candidate is note-authored: it requires his affirmation in the
# interactive review and still passes the gate, the secret screen and all four commit steps. Nothing in this file writes a memory.
INBOX_APPROVED = "00 - Inbox/Approved Memory Statements.md"
APPROVED_LINE = re.compile(r"^- \[x\] \*\*(?P<cls>[A-Z_]+)\*\* \((?P<meta>[^)]*)\) (?P<text>.+?) <!-- cand:(?P<tag>[0-9a-f]{12}) -->\s*$")
INBOX_CLASS = {"PREFERENCE": ("preference", "pref_"), "COMMITMENT": ("goal", "goal_"), "PROJECT": ("goal", "goal_"),
               "LESSON": ("instruction", "rule_")}      # IMPORTANT_FACT has no generic key: a fact needs a registry key, so it is not promoted


def _approved_raws(note, exclusions) -> list:
    out = []
    for i, line in enumerate(note.text.split("\n"), 1):
        if not line.startswith("- [x] "):
            continue
        m = APPROVED_LINE.match(line)
        if not m:
            exclusions["approved_line_malformed"] += 1
            continue
        mapping = INBOX_CLASS.get(m.group("cls"))
        if mapping is None:
            exclusions["approved_class_needs_registry_key"] += 1
            continue
        text = m.group("text").strip()
        v = trim_value(text.split(" \u2014 said: ", 1)[0].strip().rstrip("."))
        attr = _prefixed(mapping[1], v, 4) if v else None
        if not attr:
            exclusions["no_usable_key"] += 1
        elif len(text) > G.EVIDENCE_MAX:
            exclusions["evidence_too_long"] += 1
        else:
            out.append(_note_raw(note, mapping[0], attr, v, text, i))
    return out


def raws_from_note(note, exclusions) -> list:
    if note.path == "VAULT-INDEX.md":
        return _index_raws(note, exclusions)
    if note.path == INBOX_APPROVED:
        return _approved_raws(note, exclusions)
    return _rule_raws(note, exclusions)


# ---- candidates ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    id: str
    type: str
    attribute: str
    value: str
    evidence: str
    source_kind: str            # user | session | note
    source_ref: str
    origin: str                 # transcript | vault
    channel: str                # typed | voice | note
    where: str                  # transcript file name or vault path
    line: int
    text_sha: str
    timestamp: object           # a genuine instant, or None
    timestamp_basis: str
    confidence: str             # stated | inferred
    sensitivity: str            # normal | private
    corroboration: object       # None | "typed" | "voice_independent"
    occurrences: int
    other_refs: tuple
    session: str
    uuid: str
    decision: str               # the real gate's dry-run decision
    reason: str
    details: tuple
    if_affirmed_decision: str   # what the gate would say if the user affirmed it (source user, stated)
    if_affirmed_reason: str
    requires_confirmation: bool
    confirmation_reasons: tuple
    conflict_ids: tuple = ()
    template_risk: bool = False
    source_date: object = None

    @property
    def key(self) -> str:
        return "user." + self.attribute


def candidate_id(attribute, value) -> str:
    return "cand_" + sha(f"{attribute}\x1f{norm(value)}")[:12]


def screen_raw(raw, exclusions, reference="") -> bool:
    """Keep a match only if it is safe and well formed. Counts every drop by reason, never the text. `reference` is the
    normalized text of the rule documents: a typed INSTRUCTION that is a quote of one is not a new rule from the user.
    (Only instructions: a plain fact he said that a note later copied is still his statement, and merges with the note.)"""
    hit = screen_text(raw.evidence + " " + raw.value)
    if hit:
        exclusions["secret_like:" + ",".join(hit[1]) if hit[0] == "credential_like" else "injection_like"] += 1
        return False
    if raw.value not in raw.evidence or not G.evidence_usable(raw.evidence):
        exclusions["unusable_evidence"] += 1
        return False
    if raw.type == "instruction" and raw.origin == "transcript" and len(norm(raw.evidence)) >= 20 \
            and norm(raw.evidence) in reference:
        exclusions["quoted_from_vault"] += 1
        return False
    why = contact_or_third_party(raw.attribute, raw.value, raw.evidence)
    if why:
        exclusions[why] += 1
        return False
    return True


def _decide(cand_fields, generated_at, affirmed):
    f = cand_fields
    kind, conf, ref = f["source_kind"], f["confidence"], f["source_ref"]
    if affirmed:
        kind, conf, ref = "user", "stated", "review-affirmation:" + f["id"]
    item = M.MemoryItem(id=M.derive_id("user", f["attribute"], f["value"], generated_at), type=f["type"], subject="user",
                        attribute=f["attribute"], value=f["value"], created=generated_at, updated=generated_at,
                        source=M.Source(kind, ref), confidence=conf, valid_from=f["timestamp"] or generated_at,
                        status="active", sensitivity=f["sensitivity"])
    d = G.evaluate(G.Proposal(item, f["evidence"]), (), generated_at)
    return d.decision, d.reason, tuple(d.details)


def build_candidates(turns, notes, *, generated_at, reference_notes=()):
    """(candidates, exclusions). Every candidate has been dry-run through the real gate; nothing is applied."""
    if M.parse_ts(generated_at) is None:
        raise ValueError("generated_at must be YYYY-MM-DDTHH:MM:SSZ")
    exclusions = Counter()
    raws = []
    for t in turns:
        raws.extend(raws_from_turn(t, exclusions))
    for n in notes:
        raws.extend(raws_from_note(n, exclusions))
    reference = norm(" ".join(n.text for n in list(notes) + list(reference_notes)))
    raws = [r for r in raws if screen_raw(r, exclusions, reference)]
    groups = {}
    for r in raws:
        groups.setdefault((r.attribute, norm(r.value)), []).append(r)
    out = []
    for (attr, nv), rs in groups.items():
        rs.sort(key=lambda r: (r.timestamp or "9999", r.ref))
        typed = [r for r in rs if r.channel == CHANNEL_TYPED]
        voice = [r for r in rs if r.channel == CHANNEL_VOICE]
        notes_ = [r for r in rs if r.channel == CHANNEL_NOTE]
        corroboration = None
        if typed:
            primary, kind, conf, corroboration = typed[0], "user", "stated", "typed"
        elif voice:
            primary = voice[0]
            if len({r.session for r in voice}) >= 2 and len({r.date for r in voice if r.date}) >= 2:
                kind, conf, corroboration = "user", "stated", "voice_independent"
            else:
                kind, conf = "session", "inferred"
        else:
            primary, kind, conf = notes_[0], "note", "inferred"
        sens = sensitivity_of(attr, primary.value, primary.evidence)
        cid = candidate_id(attr, primary.value)
        basis = ("transcript_turn" if primary.timestamp else "none_available") if primary.origin == "transcript" else "none_available"
        fields = dict(id=cid, type=primary.type, attribute=attr, value=primary.value, evidence=primary.evidence,
                      source_kind=kind, source_ref=primary.ref, confidence=conf, sensitivity=sens,
                      timestamp=primary.timestamp)
        decision, reason, details = _decide(fields, generated_at, False)
        aff_decision, aff_reason, _ = _decide(fields, generated_at, True)
        reasons = []
        if decision != G.ACCEPT:
            reasons.append("gate:" + reason)
        if sens != "normal":
            reasons.append("private")
        if primary.type == "instruction":
            reasons.append("instruction")
        if kind != "user":
            reasons.append("voice_uncorroborated" if kind == "session" else "note_authored_by_assistant")
        if primary.timestamp is None:
            reasons.append("timestamp_from_confirmation")
        out.append(Candidate(
            id=cid, type=primary.type, attribute=attr, value=primary.value, evidence=primary.evidence, source_kind=kind,
            source_ref=primary.ref, origin=primary.origin, channel=primary.channel, where=primary.where, line=primary.line,
            text_sha=primary.text_sha, timestamp=primary.timestamp, timestamp_basis=basis, confidence=conf, sensitivity=sens,
            corroboration=corroboration, occurrences=len(rs), other_refs=tuple(sorted({r.ref for r in rs} - {primary.ref}))[:REF_LIMIT], session=primary.session, uuid=primary.uuid,
            decision=decision, reason=reason, details=details, if_affirmed_decision=aff_decision,
            if_affirmed_reason=aff_reason, requires_confirmation=bool(reasons), confirmation_reasons=tuple(reasons),
            template_risk=primary.template_risk, source_date=primary.date_hint))
    out.sort(key=lambda c: (c.attribute, c.id))
    return mark_conflicts(out), exclusions


def mark_conflicts(cands):
    """Candidates that claim different values for one logical key point at each other. Nothing is resolved."""
    by_key = {}
    for c in cands:
        by_key.setdefault(c.attribute, []).append(c)
    ids = {}
    for key, cs in by_key.items():
        if len(cs) > 1:
            for c in cs:
                ids[c.id] = tuple(sorted(o.id for o in cs if o.id != c.id))
    out = []
    for c in cands:
        if c.id in ids:
            reasons = c.confirmation_reasons + ("conflict",)
            out.append(replace(c, conflict_ids=ids[c.id], requires_confirmation=True, confirmation_reasons=reasons))
        else:
            out.append(c)
    return out


# ---- the seven known conflicts -------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    key: str
    sides: tuple            # (label, note path, regex)


_P = lambda label, path, rx: (label, path, re.compile(rx))
KNOWN_PROBES = (
    Probe("user.operating_focus", (
        _P("columbus_chosen", "04 - Wholesaling/04 - Wholesaling.md", r"Chosen market: \*\*Columbus, Ohio\*\*"),
        _P("columbus_open_task", "Active Priorities.md", r"Wholesaling restart: pull probate leads in Columbus"),
        _P("kansas_city_open_task", "Active Priorities.md", r"KNOCK THE DOORS"),
        _P("kansas_city_channel", "03 - Markets/Kansas City (Missouri).md", r"Lead channel in use: city code violations"))),
    Probe("user.business_start", (
        _P("started_feb_2026_profile", "01 - Daily Notes/09 - September 2026/2026-09-14.md", r"started Feb 2026"),
        _P("started_february_2026", "04 - Wholesaling/04 - Wholesaling.md", r"started February 2026"),
        _P("nine_months_in", "04 - Wholesaling/04 - Wholesaling.md", r"nine months in"),
        _P("nine_months_priorities", "Active Priorities.md", r"Root cause of 9 months with no closed deals"))),
    Probe("user.priority", (
        _P("cash_buyers_current_priority", "VAULT-INDEX.md", r"Active — this is the current priority"),
        _P("cash_buyers_not_the_focus", "Active Priorities.md", r"not the focus right now"))),
    Probe("user.instruction_confirm_edits", (
        _P("double_confirm_locked", "BOOT-CONFIG.md", r"Double-confirm before any source-code edit"),
        _P("stop_asking_permission", "Active Priorities.md", r"stop asking permission for routine build choices"))),
    Probe("user.door_knock_status", (
        _P("zero_contacted", "Active Priorities.md", r"zero contacted"),
        _P("four_calls_made", "01 - Daily Notes/09 - September 2026/2026-09-30.md", r"made 4 calls off yesterday's Kansas City list"))),
    Probe("user.fund_count", (
        _P("nine_funds", "VAULT-INDEX.md", r"Nine funds, one per market"),
        _P("three_land_funds", "02 - Wholesaling/Buy Box.md", r"three land-only funds"),
        _P("not_original_nine", "03 - Markets/Markets.md", r"not part of the original 9-fund list"))),
    Probe("project.voice_lock", (
        _P("enroll_voice_lock_open", "Active Priorities.md", r"Enroll and confirm voice-lock"),
        _P("superseded_by_speaker_id", "Active Priorities.md", r"superseded by \[\[Speaker ID Evaluation"))),
)
_DATE_HINT = re.compile(r"20\d{2}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Conflict:
    key: str
    kind: str               # known | generic
    status: str             # detected (2+ sides found) | partial | not_found
    sides: tuple            # dicts: label, path, line, line_sha, note_sha, quote, date_hint
    needs_review: bool = True


def _date_hint(line, path):
    """A date that appears in the competing line or in the note's own file name: shown as evidence, never as authority."""
    m = _DATE_HINT.search(line) or _DATE_HINT.search(path)
    return m.group(0) if m else None


def known_conflicts(notes) -> list:
    """Each Phase 5 conflict with every competing statement found, side by side. Authority and chronology are shown as
    evidence only: `needs_review` is always True, nothing is resolved here."""
    by_path = {n.path: n for n in notes}
    out = []
    for probe in KNOWN_PROBES:
        sides = []
        for label, path, rx in probe.sides:
            note = by_path.get(path)
            if note is None:
                continue
            for i, line in enumerate(note.text.split("\n"), 1):
                m = rx.search(line)
                if m:
                    quote = m.group(0)
                    sides.append({"label": label, "path": path, "line": i, "line_sha": sha(line), "note_sha": note.sha,
                                  "quote": quote if not screen_text(quote) else "[withheld]",
                                  "date_hint": _date_hint(line, path)})
                    break
        status = "detected" if len(sides) >= 2 else ("partial" if sides else "not_found")
        out.append(Conflict(probe.key, "known", status, tuple(sides)))
    return out


def generic_conflicts(cands) -> list:
    out = []
    by_key = {}
    for c in cands:
        by_key.setdefault(c.attribute, []).append(c)
    for key in sorted(by_key):
        cs = by_key[key]
        if len(cs) > 1:
            sides = tuple({"label": c.id, "path": c.where, "line": c.line, "line_sha": c.text_sha, "note_sha": None,
                           "quote": None if c.sensitivity != "normal" else c.value, "date_hint": (c.timestamp or "")[:10] or None}
                          for c in cs)
            out.append(Conflict("user." + key, "generic", "detected", sides))
    return out


# ---- the review pack -----------------------------------------------------------------------------

def evidence_tier(c) -> str:
    """How far the evidence can be trusted, as a fixed label: typed words, corroborated speech, speech, or a note."""
    if c.channel == CHANNEL_TYPED:
        return "typed_user_turn"
    if c.channel == CHANNEL_VOICE:
        return "voice_corroborated" if c.corroboration == "voice_independent" else "voice_uncorroborated"
    return "note_template_default_risk" if c.template_risk else "note_authored"


def review_class(c) -> str:
    """What the review may do with a candidate in this phase. Only `offer` is shown for a decision."""
    if c.sensitivity != "normal":
        return "private_blocked"
    if c.type == "instruction":
        return "deferred_instruction"
    if c.template_risk:
        return "deferred_template"
    if c.if_affirmed_decision != G.ACCEPT:
        return "gate_blocked"
    return "offer"


def conflict_id(key, kind) -> str:
    return "conf_" + sha(f"{kind}\x1f{key}")[:8]


def conflict_scope(key) -> str:
    """`memory` when the conflict is over a registered personal key a memory could hold, else `document_drift`."""
    return "memory" if key.startswith("user.") and K.kind_of(key[5:]) != K.UNKNOWN else "document_drift"


def cand_dict(c, *, redact, groups=()) -> dict:
    d = {"id": c.id, "type": c.type, "subject": "user", "attribute": c.attribute, "key": c.key,
         "value": c.value, "evidence": c.evidence, "source_kind": c.source_kind, "source_ref": c.source_ref,
         "origin": c.origin, "channel": c.channel, "where": c.where, "line": c.line, "text_sha": c.text_sha,
         "timestamp": c.timestamp, "timestamp_basis": c.timestamp_basis, "confidence": c.confidence,
         "sensitivity": c.sensitivity, "corroboration": c.corroboration, "occurrences": c.occurrences,
         "session": c.session, "uuid": c.uuid, "other_refs": list(c.other_refs), "gate": {"decision": c.decision, "reason": c.reason, "details": list(c.details)},
         "if_affirmed": {"decision": c.if_affirmed_decision, "reason": c.if_affirmed_reason},
         "requires_confirmation": c.requires_confirmation, "confirmation_reasons": list(c.confirmation_reasons),
         "conflict_ids": list(c.conflict_ids), "redacted": False, "template_risk": c.template_risk,
         "source_date": c.source_date, "evidence_tier": evidence_tier(c), "review_class": review_class(c),
         "registry": K.describe(c.attribute), "conflict_groups": list(groups)}
    if redact and c.sensitivity != "normal":
        d.update(value=None, evidence=None, redacted=True, value_fingerprint=M.fingerprint(c.value),
                 value_length=len(c.value))
    return d


def conflict_dict(k, cands=()) -> dict:
    attr = k.key[5:] if k.key.startswith("user.") else None
    return {"id": conflict_id(k.key, k.kind), "key": k.key, "kind": k.kind, "status": k.status, "needs_review": k.needs_review,
            "scope": conflict_scope(k.key), "candidates": sorted(c.id for c in cands if c.attribute == attr),
            "sides": [dict(s) for s in k.sides]}


def review_pack(cands, conflicts, meta, *, redact=True) -> dict:
    dist = Counter(c.decision for c in cands)
    groups = {}
    for k in conflicts:
        if k.key.startswith("user."):
            groups.setdefault(k.key[5:], []).append(conflict_id(k.key, k.kind))
    return {"format": PACK_FORMAT, "meta": dict(meta),
            "summary": {"candidates": len(cands), "gate_decisions": dict(sorted(dist.items())),
                        "requires_confirmation": sum(c.requires_confirmation for c in cands),
                        "by_confidence": dict(sorted(Counter(c.confidence for c in cands).items())),
                        "by_review_class": dict(sorted(Counter(review_class(c) for c in cands).items())),
                        "by_source_kind": dict(sorted(Counter(c.source_kind for c in cands).items())),
                        "by_sensitivity": dict(sorted(Counter(c.sensitivity for c in cands).items())),
                        "if_affirmed": dict(sorted(Counter(c.if_affirmed_decision for c in cands).items()))},
            "candidates": [cand_dict(c, redact=redact, groups=sorted(groups.get(c.attribute, ()))) for c in cands],
            "conflicts": [conflict_dict(k, cands) for k in conflicts],
            "registry_version": K.REGISTRY_VERSION}


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def private_companion(cands) -> dict:
    """The private values that the public pack redacts, for the person reviewing. Same ids, nothing else."""
    return {"format": "memory-review-private/1",
            "warning": "PRIVATE. Personal facts for the reviewer only. Do not copy into the vault, Git, logs or Memory 2.0.",
            "candidates": [cand_dict(c, redact=False) for c in cands if c.sensitivity != "normal"]}


def report(pack) -> str:
    s, m = pack["summary"], pack["meta"]
    L = ["# Memory 2.0 candidate review pack", "",
         f"Generated {m.get('generated_at')}. READ-ONLY dry run: nothing was written to the vault, the store or the index.", "",
         "## Summary", "", f"- candidates: {s['candidates']}", f"- gate dry-run: {s['gate_decisions']}",
         f"- requiring confirmation: {s['requires_confirmation']}", f"- by confidence: {s['by_confidence']}",
         f"- by source kind: {s['by_source_kind']}", f"- by sensitivity: {s['by_sensitivity']}",
         f"- decision if the user affirmed each one: {s['if_affirmed']}", "", "## Conflicts (never resolved here)", ""]
    for k in pack["conflicts"]:
        L.append(f"### {k['key']}: {k['status']} ({k['kind']})")
        for sd in k["sides"]:
            q = sd.get("quote")
            L.append(f"- {sd['label']}: `{sd['path']}` line {sd['line']}" + (f" ({sd['date_hint']})" if sd.get("date_hint") else "")
                     + (f": \"{q}\"" if q else ": [private]"))
        L.append("")
    L += ["## Candidates", "", "| id | key | value | source | conf | sens | gate | confirm |", "|---|---|---|---|---|---|---|---|"]
    for c in pack["candidates"]:
        v = "[PRIVATE: see private companion]" if c["redacted"] else (c["value"] or "").replace("|", "/")[:70]
        L.append(f"| {c['id']} | {c['key']} | {v} | {c['source_kind']}/{c['channel']} | {c['confidence']} | "
                 f"{c['sensitivity']} | {c['gate']['decision']} {c['gate']['reason']} | "
                 f"{'yes' if c['requires_confirmation'] else 'no'} |")
    L += ["", "## Exclusions (counts only)", ""] + [f"- {k}: {v}" for k, v in sorted(pack["meta"].get("exclusions", {}).items())]
    return "\n".join(L) + "\n"
