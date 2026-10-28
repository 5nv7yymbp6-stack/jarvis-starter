"""The Memory 2.0 write gate (Phase 2): the model PROPOSES a memory, this module DECIDES.

PURE. It reads no file, writes no file, calls no model, no embedding, no network, reads no clock (the caller
passes `now`), keeps no state, and changes nothing it is given: the proposal and the existing items are only
read. Its only Jarvis import is the pure core, backtalk.memory. Nothing in the running system imports it yet.

    evaluate(proposal, existing, now) -> Decision

    PROPOSE -> VALIDATE -> SCREEN -> DEDUPE -> CONFLICT CHECK -> DECISION

DECISIONS (nothing else exists):
    ACCEPT                 safe and valid; the Decision lists the ids it supersedes, for the future writer
    REJECT                 refused, with a reason code; nothing may be stored
    REQUIRE_CONFIRMATION   valid and placeable, but the user must say yes first (private memories)
    CONFLICT               equal authority, equally recent, different values: no guess is made

The model's proposal is NOT authoritative. The model declares what it believes (stated or inferred, which
memory it updates); every one of those claims is checked, and none is ever upgraded: an inference stays an
inference. The first failing check decides, in a fixed order, so the same inputs always give the same Decision.

What a Decision carries (and never carries): reason codes, ids, the bounded evidence and the declared
confidence and sensitivity, as plain data (to_dict) for a later ledger. It never carries a secret (secret-like
content is detected FIRST, and then not even the evidence is kept) and never any reasoning of the model.

SOURCE KINDS (memory.SOURCE_KINDS) carry meaning here: "user" is the user's own words; "session", "note",
"dream" and "import" are everything else. Only "user" can make a memory STATED or an INSTRUCTION, which is
how an instruction found in a web page, a note or a tool result is kept out of the user's memory.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from backtalk import memory as M

ACCEPT, REJECT, REQUIRE_CONFIRMATION, CONFLICT = "ACCEPT", "REJECT", "REQUIRE_CONFIRMATION", "CONFLICT"
DECISIONS = (ACCEPT, REJECT, REQUIRE_CONFIRMATION, CONFLICT)

EVIDENCE_MAX = 300             # characters kept; more is refused, never cut
EVIDENCE_MIN = 8               # shorter is not evidence of anything
INFERRED_EVIDENCE_MIN_CHARS = 24
INFERRED_EVIDENCE_MIN_WORDS = 5
INFERRED_EVIDENCE_MIN_WORDS_SIGNIFICANT = 3     # words of 3+ letters: "a b c d e f" is not a basis for anything
_ID_OK = re.compile(r"^mem_[0-9a-f]{12}\Z")

# ---- screening vocabularies --------------------------------------------------------------------

# Credential-like material. Patterns follow the families shellscan's log redactions already cover
# (URL credentials, bearer/authorization, provider and cloud keys, JWTs, long opaque blobs), plus the ways a
# person SAYS a secret out loud ("my password is ...") and the shapes of card and social-security numbers.
_CREDENTIAL_PATTERNS = (
    ("credential:url", re.compile(r"://[^/\s:@]+:[^/\s@]+@")),
    ("credential:bearer", re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("credential:header", re.compile(r"(?i)\b(?:authorization|x-api-key|x-auth-token|set-cookie|cookie)\s*:\s*\S{6,}")),
    ("credential:assignment", re.compile(
        r"(?i)\b(?:password|passwd|passcode|passphrase|pass phrase|api[ _-]?key|secret|token|access[ _-]?key|"
        r"client[ _-]?secret|auth[ _-]?token|private[ _-]?key|pin|ssn|social security number)\b\s*"
        r"(?:is|was|=|:)\s*[\"'`]?\S{4,}")),
    ("credential:provider-key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}")),
    ("credential:github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("credential:slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("credential:aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("credential:jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*")),
    ("credential:private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("credential:card-number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("credential:ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credential:opaque-blob", re.compile(r"\b[A-Za-z0-9+_=-]{32,}\b")),
)

# Text that tries to steer the assistant instead of describing the user.
_INJECTION = re.compile(
    r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)\b"
    r"|\bsystem prompt\b|\byou are now\b|\bnew instructions\s*:|<\s*/?\s*system|\[\s*system\b|<<")

# Attributes and subjects that name live state: the Situation layer's job, never long-term memory.
TEMPORARY_NAMES = frozenset({
    "mic", "mic-mode", "mic_mode", "microphone", "ptt", "push-to-talk", "audio-device", "device", "device-state",
    "current-device", "current-task", "active-task", "task-state", "current-window", "frontmost-app", "window",
    "conversation", "conversation-state", "session", "session-state", "turn", "current-turn", "last-utterance",
    "last-command", "last-action", "last-run", "current-location", "now-playing", "volume", "situation",
    "clipboard", "screen", "cursor"})
_TEMPORARY_PHRASES = re.compile(
    r"(?i)\b(?:right now|at the moment|just now|this (?:turn|session|conversation|moment)|for this (?:turn|session|"
    r"conversation)|for now)\b")

# Words that carry no durable content on their own.
FILLER_WORDS = frozenset({
    "hi", "hello", "hey", "yo", "howdy", "thanks", "thank", "you", "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "sure", "please", "good", "morning", "afternoon", "evening", "night", "bye", "goodbye", "hmm", "uh", "um", "well",
    "so", "cool", "nice", "great", "right", "alright", "how", "are", "is", "it", "going", "what", "up", "the", "a",
    "an", "and", "or", "to", "i", "me", "my", "we", "do", "did", "can", "could", "would", "will", "jarvis", "boss",
    "sir", "true", "false", "none", "unknown", "n", "s"})


# ---- the proposal and the decision --------------------------------------------------------------

@dataclass(frozen=True)
class Proposal:
    """What the model wants remembered. `item.confidence` and `item.source` are the model's own declaration."""
    item: object                       # a memory.MemoryItem, status "active", no supersedes (the gate decides those)
    evidence: str                      # the user's own words (stated) or what the inference rests on (inferred)
    target_ids: tuple = ()             # ids of memories the model believes this updates; checked, never trusted


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str
    memory_id: object = None           # the proposed item's id, when it has a usable one
    evidence: str = ""                 # as proposed (bounded); empty whenever secret-like or never-expose
    confidence: object = None
    sensitivity: object = None
    duplicate_id: object = None
    supersedes: tuple = ()             # ids the new item replaces; the writer ends them at `effective_from`
    effective_from: object = None      # the proposed item's valid_from, when it supersedes something
    conflict: tuple = ()               # ids of the equally authoritative items it ties with
    details: tuple = ()                # further codes only (never values)

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "memory_id": self.memory_id,
                "evidence": self.evidence, "confidence": self.confidence, "sensitivity": self.sensitivity,
                "duplicate_id": self.duplicate_id, "supersedes": list(self.supersedes),
                "effective_from": self.effective_from, "conflict": list(self.conflict),
                "details": list(self.details)}


@dataclass(frozen=True)
class Confirmation:
    """A recorded yes for ONE private proposal, bound to its exact content by `proposal_digest`. It can change only the
    private REQUIRE_CONFIRMATION outcome; it never overrides a REJECT or CONFLICT. It is plain data and proves nothing
    about WHO said yes: the store derives it internally, only after its own authorization checks have passed."""
    item_digest: str
    kind: str = "private"


def proposal_digest(item, evidence, links) -> str:
    """sha256 over the exact memory this proposal would store: every item field, the supersession links the gate computed,
    and the evidence."""
    body = {"v": "confirm/1", "item": M.to_dict(replace(item, supersedes=tuple(sorted(links)))), "evidence": evidence}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_proposal(data):
    """(Proposal, ()) for a plain mapping {item, evidence, target_ids?}, else (None, error codes). Strict, like
    memory.parse: nothing is coerced and an unknown field is an error."""
    if not hasattr(data, "keys") or not hasattr(data, "get"):
        return None, ("invalid:not_a_mapping",)
    keys = list(data.keys())
    errors = [f"unknown_field:{k}" for k in sorted(map(str, keys)) if k not in ("item", "evidence", "target_ids")]
    errors += [f"missing:{k}" for k in ("item", "evidence") if k not in keys]
    if "evidence" in keys and not isinstance(data["evidence"], str):
        errors.append("invalid:evidence")
    targets = data.get("target_ids", ())
    if isinstance(targets, (list, tuple)) and all(isinstance(t, str) for t in targets):
        targets = tuple(targets)
    else:
        errors.append("invalid:target_ids")
        targets = ()
    item = None
    if "item" in keys:
        item, item_errors = M.parse(data["item"])
        errors += [f"item:{e}" for e in item_errors]
    if errors:
        return None, tuple(dict.fromkeys(errors))
    return Proposal(item=item, evidence=data["evidence"], target_ids=targets), ()


# ---- pure helpers --------------------------------------------------------------------------------

def screen_secrets(text) -> tuple:
    """Categories of credential-like material in `text` (never the material itself), in a fixed order."""
    if not isinstance(text, str):
        return ()
    return tuple(cat for cat, pat in _CREDENTIAL_PATTERNS if pat.search(text))


def screen_item(item, evidence):
    """('credential_like', categories) or ('injection_like', ()) when the item or its evidence carries
    credential-like or assistant-steering text, else None. Categories only, never the text. Pure: the same
    screen the gate runs first, exposed so a storage layer can refuse independently of the gate."""
    texts = [evidence] + [getattr(item, f, None) for f in ("value", "subject", "attribute")]
    src = getattr(item, "source", None)
    texts.append(src.ref if isinstance(src, M.Source) else "")
    found = tuple(dict.fromkeys(c for tx in texts for c in screen_secrets(tx)))
    if found:
        return "credential_like", found
    if any(isinstance(tx, str) and _INJECTION.search(tx) for tx in texts):
        return "injection_like", ()
    return None


def evidence_usable(evidence) -> bool:
    """Is this bounded, single-line, non-blank evidence text (the shape the gate and the store both require)?"""
    return isinstance(evidence, str) and bool(evidence.strip()) and not _bad_evidence(evidence)


def _tokens(text) -> list:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(text)).casefold())


def _significant(token) -> bool:
    return len(token) >= 3 or any(c.isdigit() for c in token)


def _to_utc(at):
    if isinstance(at, datetime):
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("time must be timezone-aware")
        return at.astimezone(timezone.utc)
    t = M.parse_ts(at)
    if t is None:
        raise ValueError("time must be YYYY-MM-DDTHH:MM:SSZ or an aware datetime")
    return t


def _stamp(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _bad_evidence(ev) -> bool:
    return len(ev) > EVIDENCE_MAX or any(ord(c) < 32 or ord(c) == 127 for c in ev)


def _order(it):
    return M.parse_ts(it.valid_from), M.parse_ts(it.created)


def _auth(it):
    return {"stated": 2, "inferred": 1}[it.confidence]


# ---- the gate ------------------------------------------------------------------------------------

def evaluate(proposal, existing, now, confirmation=None) -> Decision:
    """The Decision for `proposal` against the `existing` items, as of `now` (a strict UTC timestamp string or an
    aware datetime). A `now` that is not a time is a ValueError. Never raises for any proposal or any item.
    `confirmation` (a Confirmation, default None = unchanged behavior) is looked at in one place only: a private
    proposal that would REQUIRE_CONFIRMATION is ACCEPTED when the confirmation's digest matches this exact proposal."""
    t_now = _to_utc(now)
    existing = tuple(existing)
    if not (isinstance(proposal, Proposal) and isinstance(proposal.item, M.MemoryItem)
            and isinstance(proposal.evidence, str) and isinstance(proposal.target_ids, tuple)
            and all(isinstance(x, str) for x in proposal.target_ids)):
        return Decision(REJECT, "malformed_proposal")
    it, ev = proposal.item, proposal.evidence

    def out(decision, reason, *, keep_evidence=True, **kw):
        ident = it.id if isinstance(it.id, str) and _ID_OK.match(it.id) else None
        return Decision(decision, reason, memory_id=ident, evidence=ev if keep_evidence else "",
                        confidence=it.confidence if it.confidence in M.CONFIDENCES else None,
                        sensitivity=it.sensitivity if it.sensitivity in M.SENSITIVITIES else None, **kw)

    # SCREEN FIRST: nothing that looks like a secret is judged further, and none of it is echoed back.
    hit = screen_item(it, ev)
    if hit:
        return out(REJECT, hit[0], keep_evidence=False, details=hit[1])

    # VALIDATE
    if it.type not in M.TYPES:
        return out(REJECT, "unsupported_type")
    if not isinstance(it.source, M.Source) or not isinstance(it.source.ref, str) or not it.source.ref.strip() \
            or it.source.kind not in M.SOURCE_KINDS:
        return out(REJECT, "missing_source")
    if not ev.strip():
        return out(REJECT, "missing_evidence")
    errs = M.validate(it)
    if errs:
        return out(REJECT, "invalid_item", details=errs)
    if it.status != "active" or it.supersedes:
        return out(REJECT, "proposal_not_new")
    if _bad_evidence(ev):
        return out(REJECT, "invalid_evidence")
    if it.sensitivity == "never-expose":
        return out(REJECT, "never_expose_not_storable", keep_evidence=False)

    # SOURCE AND AUTHORITY: only the user's own words can be stated, and only a stated user instruction is one
    if it.confidence == "stated" and it.source.kind != "user":
        return out(REJECT, "stated_requires_user_source")
    if it.type == "instruction":
        if it.source.kind != "user":
            return out(REJECT, "external_instruction")
        if it.confidence != "stated":
            return out(REJECT, "instruction_requires_user_statement")

    # LOW VALUE AND LIVE STATE
    if it.subject in TEMPORARY_NAMES or it.attribute in TEMPORARY_NAMES or _TEMPORARY_PHRASES.search(it.value):
        return out(REJECT, "temporary_context")
    ev_tokens, value_tokens = set(_tokens(ev)), set(_tokens(it.value))
    if len(ev.strip()) < EVIDENCE_MIN or not (ev_tokens - FILLER_WORDS) or not (value_tokens - FILLER_WORDS):
        return out(REJECT, "low_value")
    if it.confidence == "stated":
        needed = {tok for tok in value_tokens if _significant(tok)}
        if not needed or not needed <= ev_tokens:
            return out(REJECT, "evidence_does_not_support_value")
    elif (len(ev.strip()) < INFERRED_EVIDENCE_MIN_CHARS or len(ev.split()) < INFERRED_EVIDENCE_MIN_WORDS
          or sum(1 for tok in _tokens(ev) if _significant(tok)) < INFERRED_EVIDENCE_MIN_WORDS_SIGNIFICANT):
        return out(REJECT, "weak_evidence")

    # DEDUPE AND CONFLICT, as of the later of now and the moment the proposed fact starts
    at = max(t_now, M.parse_ts(it.valid_from))
    at_s = _stamp(at)
    by_id = {x.id: x for x in existing if isinstance(x, M.MemoryItem) and not M.validate(x)}
    if any(getattr(x, "id", None) == it.id for x in existing):
        return out(REJECT, "id_exists")
    for tid in proposal.target_ids:
        old = by_id.get(tid)
        if old is None or M.key_of(old) != M.key_of(it):
            return out(REJECT, "invalid_target", details=(f"target:{tid}",))
    now_res = M.resolve(list(by_id.values()), at_s)            # what the core itself calls current at `at`
    key = M.key_of(it)
    current = [x for x in now_res.active if M.key_of(x) == key] + \
              [x for c in now_res.conflicts if c.key == key for x in c.candidates]
    for x in sorted(current, key=lambda x: x.id):
        if x.id != it.id and M.classify_pair(it, x) == M.EXACT_DUPLICATE and _auth(x) >= _auth(it):
            return out(REJECT, "duplicate", duplicate_id=x.id)
    links = tuple(sorted(x.id for x in current if x.id != it.id
                         and (_auth(it) > _auth(x) or (_auth(it) == _auth(x) and _order(it) > _order(x)))))
    trial = replace(it, supersedes=links)
    res = M.resolve(list(by_id.values()) + [trial], at_s)
    if any(r[0] == it.id for r in res.rejected):                # unreachable by construction; kept as a guard
        return out(REJECT, "invalid_supersession", details=next(r[1] for r in res.rejected if r[0] == it.id))
    for c in res.conflicts:
        if c.key == key and any(x.id == it.id for x in c.candidates):
            return out(CONFLICT, "conflict", conflict=tuple(x.id for x in c.candidates if x.id != it.id))
    winner = next((x for x in res.active if M.key_of(x) == key), None)
    if winner is None or winner.id != it.id:
        why = next((r for x, r in res.history if x.id == it.id), "superseded")
        if why == "shadowed":
            return out(REJECT, "inferred_cannot_override_stated",
                       details=tuple(x.id for x in res.active if M.key_of(x) == key))
        if why == "duplicate":
            return out(REJECT, "duplicate", duplicate_id=winner.id if winner else None)
        return out(REJECT, "stale_proposal", details=tuple(x.id for x in res.active if M.key_of(x) == key))
    for tid in proposal.target_ids:
        if tid not in links:
            return out(REJECT, "target_not_superseded", details=(f"target:{tid}",))

    # DECISION
    extra = {"supersedes": links, "effective_from": it.valid_from} if links else {}
    if it.sensitivity == "private":
        if isinstance(confirmation, Confirmation) and confirmation.kind == "private" \
                and confirmation.item_digest == proposal_digest(it, ev, links):
            return out(ACCEPT, "accepted_confirmed", **extra)
        return out(REQUIRE_CONFIRMATION, "private_requires_confirmation", **extra)
    return out(ACCEPT, "accepted_supersedes" if links else "accepted", **extra)


# Known limit (Phase 2): a proposal must be true at `max(now, its valid_from)`. A fact that has already ended
# (valid_to in the past) is refused as stale_proposal; recording history is a later phase.
