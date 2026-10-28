"""The Memory 2.0 commit PLANNING layer (Phase 8.2b): turn one reviewed ledger decision into a validated, digest-bound `Plan`,
and check a typed answer against it. NOTHING HERE WRITES. No store write, no receipt, no review ledger entry, no file, no
network, no clock read (the caller passes `clock`), no terminal. Nothing live imports it.

    plan(...) -> Plan             every check below, in order; any failure is a CommitError and no Plan exists
    render_preview(plan)          the lines a person would read, built only from the Plan
    challenge_token / expected_answer / check_answer
    compare_plans(a, b)           what differs between two plans (used by a later re-plan)

THE CHECKS, in order (each fails closed with a code; `Plan.checks` records the ones performed):
    inputs, replay (the ledger replays cleanly against this pack), committable (the candidate's final record is a lock or
    edit), age (at most 14 days, inclusive; a record from the future is refused), provenance and rederive (injected
    callables: the source still says what it said and the candidate still derives the same), build (the exact item that
    was reviewed), store (a verified-consistent read), gate (the real gate against that read), digests, display.

WHAT THIS MODULE IS NOT. `Plan.checks`, `verify_plan_integrity`, the display digest and the token are accidental-error and
regression defenses only. They do not prove that any check actually ran, and they do not prove that a person approved
anything. They do not protect against code running as the same user: such code can build a Plan whose fields and unkeyed
hashes are all consistent, forge a token and an answer, and edit any file. What is enforced independently of any Plan is
what the store itself checks at installation (the gate against the live items, the exact item and operation an
authorization commits to, the live state digest, the store's own clock window, the nonce).

STORE READS. The store has no atomic snapshot yet, so `read_consistent` reads the state digest, then the items, then the
digest again and accepts only if the two digests agree AND the items hash (byte for byte, with the store's own formula) to
that digest. This is a verified-consistent read, not an atomic snapshot.

SOURCES. A source check proves the text only at the moment of that read; sources are not locked.

ACCEPTANCE CRITERIA CARRIED INTO PHASE 8.2d (hard requirements; nothing here implements them):
    1. Immediately before the real store installation, the plan must be REBUILT from scratch by a fresh `plan()` call (fresh
       ledger read, pack bytes, sources, store read and trusted clock) and compared with the plan the person confirmed using
       `compare_plans`. Any difference, including "plan_inconsistent" or "clock_regressed", aborts before any receipt
       intent is appended and before any store call.
    2. A stale Plan must never authorize a write. A Plan, a token or an answer is never itself an authorization, however
       recently it was confirmed; the store's own install-time checks remain the final authority.
    3. `check_answer` puts no upper bound on how old a Plan is when its challenge is issued, and it reads no clock: the fresh
       re-plan in (1) is the only bound on staleness, and the issued and answered instants must come from the same
       trusted clock as the re-plan.
"""
import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone

from backtalk import memory as M
from backtalk import memory_extract as X
from backtalk import memory_gate as G
from backtalk import memory_keys as K
from backtalk import memory_review as R
from backtalk import memory_store as S

LOCK_MAX_AGE_DAYS = 14
CHALLENGE_SECONDS = 120
TOKEN_HEX = 8
READ_ATTEMPTS = 3
CONTEXT_VERSION = "commit-ctx/1"
TOKEN_VERSION = "commit-token/2"
DISPLAY_VERSION = "display/1"
REQUIRED_CHECKS = ("inputs", "replay", "committable", "age", "provenance", "rederive", "build", "store", "gate", "digests",
                   "display")
KIND_COMMIT, KIND_PRIVATE = "commit", "commit-private"
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"^[0-9a-f]{32}\Z")
_CODE = re.compile(r"^[a-z0-9_:\-]{1,60}\Z")
_STAMP = "%Y-%m-%dT%H:%M:%SZ"
_MISSING = object()
NOTICE = ("The confirmation token below is an accidental-error check only, not authentication; code running as you can "
          "compute it.")
INFO_HEADING = "informational, not part of the confirmation:"

# Every Plan field is bound by something named here, or is informational with a written reason. Nothing else exists: the
# module cannot be imported if a field is added without a reviewed entry.
BINDER_VOCABULARY = frozenset({"item_digest", "state_digest", "context_digest", "display_digest", "token",
                               "integrity:record", "integrity:item", "integrity:gate", "integrity:state", "integrity:age",
                               "integrity:required_checks", "integrity:display", "integrity:links"})
BINDINGS = {
    "candidate": ("context_digest", "integrity:record"),
    "run": ("context_digest", "item_digest"),
    "pack_sha": ("context_digest",),
    "registry_version": ("context_digest", "integrity:record"),
    "lock_hash": ("context_digest", "integrity:record"),
    "lock_record_json": ("integrity:record",),
    "operation": ("item_digest", "integrity:gate", "display_digest", "token"),
    "kind": ("token", "display_digest", "integrity:gate"),
    "item": ("item_digest", "integrity:item", "display_digest"),
    "evidence": ("item_digest", "integrity:item", "display_digest"),
    "links": ("item_digest", "integrity:links", "display_digest"),
    "decision": ("integrity:gate", "display_digest"),
    "store_items": ("integrity:state",),
    "superseded": ("integrity:links", "display_digest"),
    "state_digest": ("token", "integrity:state"),
    "item_digest": ("token", "integrity:item"),
    "context_digest": ("token", "integrity:record"),
    "display_digest": ("token", "integrity:display"),
    "planned_at": ("integrity:gate", "integrity:age"),
    "age_seconds": ("integrity:age",),
    "max_age_seconds": ("integrity:age",),
    "checks": ("integrity:required_checks",),
}
INFORMATIONAL = {
    "rederive_seconds": "measured wall time; it cannot be recomputed, and its budget is enforced when the plan is built",
}


class CommitError(Exception):
    """A refusal with a fixed code. Details are codes, ids and field names only, never a value."""

    def __init__(self, code, details=()):
        super().__init__(code + (": " + ", ".join(map(str, details)) if details else ""))
        self.code, self.details = code, tuple(details)


@dataclass(frozen=True, repr=False)
class OldItem:
    """A memory this commit would replace, captured from the accepted store read (never fetched again)."""
    id: str
    key: str
    value: str
    valid_from: str
    confidence: str
    sensitivity: str
    status: str
    file_sha8: str

    def __repr__(self):
        return f"OldItem(id={self.id}, file_sha8={self.file_sha8})"        # never the value


OLD_ITEM_BINDINGS = {"id": "links", "key": "display", "value": "display", "valid_from": "display", "confidence": "display",
                     "sensitivity": "display", "status": "display", "file_sha8": "display"}


@dataclass(frozen=True, repr=False)
class Plan:
    candidate: str
    run: str
    pack_sha: str
    registry_version: int
    lock_hash: str
    lock_record_json: str
    operation: str
    kind: str
    item: object
    evidence: str
    links: tuple
    decision: object
    store_items: tuple
    superseded: tuple
    state_digest: str
    item_digest: str
    context_digest: str
    display_digest: str
    planned_at: str
    age_seconds: int
    max_age_seconds: int
    rederive_seconds: float
    checks: tuple

    def __repr__(self):
        # ids and digest prefixes only: a Plan holds values in memory and must never print them
        return (f"Plan(candidate={self.candidate}, operation={self.operation}, kind={self.kind}, "
                f"item={getattr(self.item, 'id', None)}, item_digest={str(self.item_digest)[:8]}, "
                f"state_digest={str(self.state_digest)[:8]})")


@dataclass(frozen=True, repr=False)
class Preview:
    bound: tuple            # the lines the display digest covers
    info: tuple             # informational lines, outside the confirmation

    def __repr__(self):
        return f"Preview(bound=<{len(self.bound)} lines>, info=<{len(self.info)} lines>)"        # never the lines


def _check_bindings():
    names = {f.name for f in fields(Plan)}
    bound, info = set(BINDINGS), set(INFORMATIONAL)
    if bound & info or names != bound | info:
        raise RuntimeError("Plan fields without a reviewed binding or informational entry: "
                           + ", ".join(sorted(names ^ (bound | info))) + (" (overlap)" if bound & info else ""))
    for name, binders in BINDINGS.items():
        if not binders or not set(binders) <= BINDER_VOCABULARY:
            raise RuntimeError(f"unknown binder for {name}")
    if not all(isinstance(why, str) and why for why in INFORMATIONAL.values()):
        raise RuntimeError("an informational field needs a written reason")


_check_bindings()


# ---- small pure helpers --------------------------------------------------------------------------

def _sha(text) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_hex64(v) -> bool:
    return isinstance(v, str) and bool(_HEX64.match(v))


def safe_text(s) -> str:
    """A quoted rendering of `s` that cannot hide anything: quotes and backslashes are escaped, and every control, format,
    private-use, surrogate, unassigned, line/paragraph-separator and non-ASCII space character is written as \\uXXXX."""
    if not isinstance(s, str):
        raise CommitError("display_failed", ("not_text",))
    out = ['"']
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif cat[0] == "C" or cat in ("Zl", "Zp") or (cat == "Zs" and ch != " "):
            cp = ord(ch)
            out.append("\\u%04x" % cp if cp <= 0xFFFF else "\\U%08x" % cp)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _safe_code(value) -> str:
    """A store or check reason, only if it is a short code; anything else becomes "failed" (never echoed)."""
    return value if isinstance(value, str) and _CODE.match(value) else "failed"


def digest_of(items) -> str:
    """The state digest of a set of stored memories, computed from the records alone. It must equal
    `store.state_digest()` for the same store: each file is `serialize(item, evidence)` as UTF-8, named `<id>.md`, hashed,
    and the (name, sha256) list is hashed by `memory_store.manifest_digest`, the one authoritative implementation."""
    entries, seen = [], set()
    for st in items:
        if not isinstance(st, S.Stored):
            raise CommitError("store_unreadable", ("record_type",))
        if st.item.id in seen:
            raise CommitError("store_unreadable", ("duplicate_id",))
        seen.add(st.item.id)
        data = None
        try:
            data = S.serialize(st.item, st.evidence).encode("utf-8")
        except Exception:
            data = None
        if data is None:
            raise CommitError("store_unreadable", ("serialize",))
        entries.append([st.item.id + ".md", hashlib.sha256(data).hexdigest()])
    digest = None
    try:
        digest = S.manifest_digest(entries)
    except ValueError:
        digest = None
    if digest is None:
        raise CommitError("store_unreadable", ("manifest",))
    return digest


def read_consistent(store, attempts=READ_ATTEMPTS):
    """(items, state_digest) from a verified-consistent read of `store`, or CommitError. Not an atomic snapshot: it is
    accepted only if the digest read before and after the items agree AND the items hash to that digest."""
    if not (callable(getattr(store, "state_digest", None)) and callable(getattr(store, "list_items", None))):
        raise CommitError("bad_input", ("store",))
    last = "digest_of"
    for _ in range(attempts):
        failure = None
        try:
            d1 = store.state_digest()
            items = store.list_items()
            d2 = store.state_digest()
        except S.StoreError as e:
            failure = _safe_code(e.reason)
        except (OSError, UnicodeError):
            failure = "io_error"
        except Exception:
            failure = "failed"
        if failure is not None:                       # raised outside the handler: no exception context is retained
            raise CommitError("store_unreadable", (failure,))
        if not (_is_hex64(d1) and _is_hex64(d2) and isinstance(items, (tuple, list))):
            raise CommitError("store_unreadable", ("shape",))
        if d1 != d2:
            last = "state_digest_changed"
            continue
        if digest_of(items) != d1:
            last = "digest_of"
            continue
        return tuple(items), d1
    raise CommitError("store_unstable", (last,))


def _now(clock) -> str:
    """The trusted instant as a strict UTC stamp, truncated (never rounded) to the second."""
    raised = False
    try:
        v = clock()
    except Exception:
        raised = True
    if raised:
        raise CommitError("bad_clock", ("raised",))
    if isinstance(v, datetime):
        if v.tzinfo is None or v.utcoffset() is None:
            raise CommitError("bad_clock", ("naive",))
        v = v.astimezone(timezone.utc).strftime(_STAMP)
    if not isinstance(v, str) or M.parse_ts(v) is None:
        raise CommitError("bad_clock", ("format",))
    return v


def _stamp_dt(s):
    dt = M.parse_ts(s)
    if dt is None:
        raise CommitError("bad_timestamp")
    return dt


def _context_digest(run, pack_sha, candidate, lock_hash, registry_version) -> str:
    return _sha(R.canonical_json({"v": CONTEXT_VERSION, "run": run, "pack_sha": pack_sha, "candidate": candidate,
                                  "lock_hash": lock_hash, "registry_version": registry_version}))


def _old_item(stored) -> OldItem:
    it = stored.item
    return OldItem(it.id, M.key_of(it), it.value, it.valid_from, it.confidence, it.sensitivity, it.status,
                   hashlib.sha256(S.serialize(it, stored.evidence).encode("utf-8")).hexdigest()[:8])


def _expected_kind(item) -> str:
    return KIND_PRIVATE if item.sensitivity == "private" else KIND_COMMIT


# ---- the display ---------------------------------------------------------------------------------

def _need(missing, label, value, typ):
    if value is None or not isinstance(value, typ) or (isinstance(value, str) and not value):
        missing.append(label)


def _display_lines(plan) -> tuple:
    """The bound lines, in a fixed order, from the Plan's own fields. Raises display_incomplete rather than skip a line."""
    missing = []
    _need(missing, "operation", plan.operation, str)
    _need(missing, "kind", plan.kind, str)
    _need(missing, "item", plan.item, M.MemoryItem)
    _need(missing, "evidence", plan.evidence, str)
    _need(missing, "decision", plan.decision, G.Decision)
    _need(missing, "links", plan.links, tuple)
    _need(missing, "superseded", plan.superseded, tuple)
    for label in ("state_digest", "item_digest", "context_digest"):
        _need(missing, label, getattr(plan, label), str)
    _need(missing, "lock_record_json", plan.lock_record_json, str)
    if missing:
        raise CommitError("display_incomplete", tuple(missing))
    failed = False
    try:
        rec = json.loads(plan.lock_record_json)
        origin, adopted = rec["origin"], rec["adoption"]["at"]
        it = plan.item
        lines = [
            f"operation    {plan.operation}",
            f"kind         {plan.kind}",
            f"item id      {it.id}",
            f"key          {it.attribute} ({K.describe(it.attribute)['kind']})",
            f"subject      {it.subject}",
            f"type         {it.type}",
            f"value        {safe_text(it.value)}",
            f"evidence     {safe_text(plan.evidence)}",
            f"confidence   {it.confidence}",
            f"status       {it.status}",
            f"sensitivity  {it.sensitivity}",
            f"valid_from   {it.valid_from}",
            f"valid_to     {it.valid_to}",
            f"created      {it.created}",
            f"updated      {it.updated}",
            f"adopted_at   {adopted}",
            f"source       {it.source.kind} {safe_text(it.source.ref)}",
            f"origin       {origin['kind']} {safe_text(origin['where'])} line {origin['line']} sha {origin['text_sha'][:16]}",
        ]
        if plan.superseded:
            for o in plan.superseded:
                lines.append(f"replaces     {o.id} {o.key} {safe_text(o.value)} valid_from {o.valid_from} {o.status} "
                             f"{o.confidence} {o.sensitivity} sha8:{o.file_sha8}")
        else:
            lines.append("replaces     nothing")
        lines += [
            f"gate         {plan.decision.decision} {plan.decision.reason}",
            f"store        {plan.state_digest[:16]}",
            f"item digest  {plan.item_digest[:16]}",
            f"context      {plan.context_digest[:16]}",
            NOTICE,
        ]
    except CommitError:
        raise
    except Exception:
        failed = True
    if failed:
        raise CommitError("display_failed")
    return tuple(lines)


def _display_digest(lines) -> str:
    return _sha(R.canonical_json({"v": DISPLAY_VERSION, "lines": list(lines)}))


def _display_checked(plan):
    """(lines, digest), rendered twice: rendering must be deterministic."""
    first = _display_lines(plan)
    if first != _display_lines(plan):
        raise CommitError("display_unstable")
    return first, _display_digest(first)


# ---- integrity: a Plan agrees with itself --------------------------------------------------------

def verify_plan_integrity(plan) -> list:
    """Names of the invariants a Plan violates (empty = it agrees with itself). It recomputes everything derivable from the
    Plan's own fields. IT IS NOT A SECURITY CHECK: it never sees the ledger, the pack or the sources, so it cannot replace
    the replay, provenance and re-derivation checks, and a Plan built by code that recomputes every hash passes it."""
    if not isinstance(plan, Plan):
        return ["bad_plan"]
    bad = []

    def flag(name):
        if name not in bad:
            bad.append(name)
    try:
        if plan.checks != REQUIRED_CHECKS:
            flag("checks")
        rec = json.loads(plan.lock_record_json)
        if not isinstance(rec, dict) or rec.get("action") not in ("lock", "edit") or rec.get("candidate") != plan.candidate \
                or rec.get("registry_version") != plan.registry_version:
            flag("record")
        if rec.get("hash") != R.record_hash(rec) or rec.get("hash") != plan.lock_hash:
            flag("lock_hash")
        it, ev = plan.item, plan.evidence
        at = rec["adoption"]["at"]
        want_ref = R.review_ref(plan.run, plan.candidate, at)
        if not (isinstance(it, M.MemoryItem) and it.subject == "user" and it.confidence == "stated" and it.status == "active"
                and it.valid_to is None and it.created == at and it.updated == at and it.attribute == rec["key"]
                and it.type == rec["type"] and it.sensitivity == rec["sensitivity"] and it.valid_from == rec["valid_from"]
                and it.source == M.Source("user", want_ref) and it.id == M.derive_id("user", rec["key"], it.value, at)
                and M.fingerprint(it.value) == rec["value_fp"] and len(it.value) == rec["value_len"]
                and X.sha(ev) == rec["evidence_sha"]
                and (rec["value"] is None or (rec["value"] == it.value and rec["evidence"] == ev))):
            flag("item_vs_record")
        if it.supersedes != ():
            flag("item_supersedes")
        links = plan.links
        if S.item_digest(it, ev, links, plan.operation) != plan.item_digest:
            flag("item_digest")
        if _context_digest(plan.run, plan.pack_sha, plan.candidate, plan.lock_hash, plan.registry_version) != plan.context_digest:
            flag("context_digest")
        try:
            if _display_checked(plan)[1] != plan.display_digest:
                flag("display")
        except CommitError:
            flag("display")
        if plan.kind != _expected_kind(it):
            flag("kind")
        if plan.operation != ("apply" if links else "put"):
            flag("operation_links")
        d = plan.decision
        by_id = {s.item.id: s for s in plan.store_items}
        if not (links == d.supersedes == tuple(sorted(links)) == tuple(sorted(o.id for o in plan.superseded))
                and all(i in by_id and by_id[i].item.status == "active" for i in links)
                and tuple(sorted(plan.superseded, key=lambda o: o.id)) == plan.superseded
                and plan.superseded == tuple(_old_item(by_id[i]) for i in links)):
            flag("links")
        if G.evaluate(G.Proposal(it, ev), [s.item for s in plan.store_items], plan.planned_at) != d:
            flag("gate")
        row_ok = (d.decision, d.reason) == ((G.REQUIRE_CONFIRMATION, "private_requires_confirmation")
                                            if it.sensitivity == "private"
                                            else (G.ACCEPT, "accepted_supersedes" if links else "accepted"))
        if not row_ok:
            flag("gate_table")
        if not (d.memory_id == it.id and d.sensitivity == it.sensitivity and d.confidence == "stated" and d.evidence == ev
                and d.effective_from == (it.valid_from if links else None) and d.duplicate_id is None
                and d.conflict == () and d.details == ()):
            flag("decision_fields")
        if digest_of(plan.store_items) != plan.state_digest:
            flag("state")
        age = int((_stamp_dt(plan.planned_at) - _stamp_dt(rec["at"])).total_seconds())
        if age != plan.age_seconds or not 0 <= age <= plan.max_age_seconds:
            flag("age")
    except Exception:
        flag("integrity_error")
    return bad


def _require_consistent(plan):
    problems = verify_plan_integrity(plan)
    if problems:
        raise CommitError("plan_inconsistent", tuple(problems))


# ---- the plan ------------------------------------------------------------------------------------

def _call(name, fn, *args):
    failed = False
    try:
        return fn(*args)
    except CommitError:
        raise
    except Exception:
        failed = True
    raise CommitError("check_failed", (name,))            # outside the handler: the original exception is not retained


# The codes the source check (the provenance verifier) can return. Anything else is refused but NEVER echoed: a callable's own
# text could carry source material.
KNOWN_SOURCE_CODES = frozenset({"source_missing", "line_out_of_range", "evidence_not_in_line", "hash_mismatch", "ref_mismatch",
                                "turn_no_longer_eligible", "id_mismatch", "evidence_not_in_turn", "timestamp_mismatch",
                                "channel_mismatch"})


def _codes(result, code):
    """Validate an injected check's list-of-codes result: a list of short code strings, or a refusal. A non-empty clean list
    is also a refusal (the source no longer says what it said); unrecognized codes are reported only as "unrecognized_code"."""
    if not isinstance(result, list) or not all(isinstance(c, str) and _CODE.match(c) for c in result):
        raise CommitError(code, ("result_invalid",))
    if result:
        raise CommitError(code, tuple(c if c in KNOWN_SOURCE_CODES else "unrecognized_code" for c in result[:8]))


_REDERIVE_SAME = ("origin", "where", "line", "source_kind", "source_ref", "text_sha", "channel", "evidence_tier", "uuid",
                  "session", "timestamp", "confidence", "corroboration", "type", "attribute", "review_class", "registry")


def _norm(v):
    return tuple(sorted(v)) if isinstance(v, list) else v


def _rederive_mismatches(view, rec, fresh) -> list:
    bad = []
    if fresh.get("id", _MISSING) != view["id"]:
        bad.append("id")
    for k in _REDERIVE_SAME:
        if _norm(fresh.get(k, _MISSING)) != _norm(view.get(k, _MISSING)):
            bad.append(k)
    if _norm(fresh.get("conflict_groups", _MISSING)) != _norm(view.get("conflict_groups", _MISSING)):
        bad.append("conflict_groups")
    if fresh.get("review_class") != "offer":
        bad.append("review_class")
    org = rec["origin"]
    if (fresh.get("source_kind"), fresh.get("source_ref"), fresh.get("where"), fresh.get("line"), fresh.get("text_sha"),
            fresh.get("channel"), fresh.get("evidence_tier")) != (org["kind"], org["ref"], org["where"], org["line"],
                                                                  org["text_sha"], org["channel"], org["tier"]):
        bad.append("record_origin")
    if fresh.get("type") != rec["type"]:
        bad.append("type")
    if fresh.get("attribute") != rec["key_proposed"]:
        bad.append("attribute")
    if rec["valid_from_basis"] == "transcript_turn" and fresh.get("timestamp") != rec["valid_from"]:
        bad.append("valid_from")
    fs = fresh.get("sensitivity")
    if fs not in ("normal", "private") or (fs == "private" and rec["sensitivity"] == "normal"):
        bad.append("sensitivity")
    if view.get("redacted"):
        if fresh.get("redacted") is not True or fresh.get("value_fingerprint") != view.get("value_fingerprint"):
            bad.append("value")
    elif rec["action"] == "lock":
        for k in ("value", "evidence"):
            if not (fresh.get(k, _MISSING) == view.get(k) == rec[k]):
                bad.append(k)
    ev = fresh.get("evidence")
    if rec["action"] == "edit" and (X.sha(ev) if isinstance(ev, str) else None) != org.get("evidence_sha"):
        bad.append("origin_evidence")
    return sorted(set(bad))


def _valid_budget(v) -> bool:
    """A real, finite, positive number of seconds (bool, NaN, infinities and numbers too large for a float are not)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v) and v > 0
    except OverflowError:
        return False


def _bad_inputs(pack, pack_sha, records, private, candidate, store, provenance, rederive, clock, monotonic, preview_budget_s,
                max_age_days) -> tuple:
    """Names of the inputs that are unusable, checked BEFORE any injected check or store call is made."""
    bad = []
    if not isinstance(pack, R.Pack):
        bad.append("pack")
    if not _is_hex64(pack_sha):
        bad.append("pack_sha")
    if not (isinstance(records, (list, tuple)) and records and all(isinstance(r, dict) for r in records)):
        bad.append("records")
    if not (private is None or isinstance(private, dict)):
        bad.append("private")
    if not (isinstance(candidate, str) and isinstance(pack, R.Pack) and candidate in pack.views):
        bad.append("candidate")
    if not (callable(getattr(store, "state_digest", None)) and callable(getattr(store, "list_items", None))):
        bad.append("store")
    for name, fn in (("provenance", provenance), ("rederive", rederive), ("clock", clock), ("monotonic", monotonic)):
        if not callable(fn):
            bad.append(name)
    if not _valid_budget(preview_budget_s):
        bad.append("preview_budget_s")
    if not (isinstance(max_age_days, int) and not isinstance(max_age_days, bool) and max_age_days > 0):
        bad.append("max_age_days")
    return tuple(bad)


def plan(*, pack, pack_sha, records, private, candidate, store, provenance, rederive, clock, monotonic, preview_budget_s,
         max_age_days=LOCK_MAX_AGE_DAYS) -> Plan:
    """The validated Plan for one reviewed candidate, or a CommitError. Every callable is required (no defaults)."""
    done = []
    # inputs
    bad = _bad_inputs(pack, pack_sha, records, private, candidate, store, provenance, rederive, clock, monotonic,
                      preview_budget_s, max_age_days)
    if bad:
        raise CommitError("bad_input", bad)
    planned_at = _now(clock)                                              # the plan's one trusted instant
    done.append("inputs")
    # replay
    copy_records, copy_private = copy.deepcopy(list(records)), copy.deepcopy(private)
    problems = _call("replay", R.replay_validate, pack, copy_records, pack_sha, copy_private)
    if not isinstance(problems, list) or problems:
        raise CommitError("ledger_replay_invalid", tuple(problems[:5]) if isinstance(problems, list) else ("result_invalid",))
    if copy_records[0].get("registry_version") != K.REGISTRY_VERSION:
        raise CommitError("ledger_replay_invalid", ("registry_version",))
    done.append("replay")
    # committable
    state = R.State(copy_records)
    rec = state.final.get(candidate)
    if rec is None or rec.get("action") not in ("lock", "edit") or copy_records[rec["seq"]] is not rec \
            or rec["hash"] != R.record_hash(rec):
        raise CommitError("not_committable", (state.status(candidate),))
    run = copy_records[0]["run"]
    done.append("committable")
    # age
    age = int((_stamp_dt(planned_at) - _stamp_dt(rec["at"])).total_seconds())
    max_age = max_age_days * 86400
    if age < 0:
        raise CommitError("lock_from_future")
    if age > max_age:
        raise CommitError("lock_too_old")
    done.append("age")
    # provenance
    view = pack.views[candidate]
    _codes(_call("provenance", provenance, copy.deepcopy(view)), "source_stale")
    done.append("provenance")
    # rederive
    t0 = _call("monotonic", monotonic)
    fresh = _call("rederive", rederive, copy.deepcopy(view))
    t1 = _call("monotonic", monotonic)
    if not (isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and not isinstance(t0, bool)
            and not isinstance(t1, bool) and t1 >= t0):
        raise CommitError("bad_clock", ("monotonic",))
    elapsed = float(t1 - t0)
    if elapsed > preview_budget_s:
        raise CommitError("rederivation_too_slow")
    if fresh is None:
        raise CommitError("not_rederived")
    if not isinstance(fresh, dict):
        raise CommitError("not_rederived", ("result_invalid",))
    mism, comparison_failed = [], False
    try:
        mism = _rederive_mismatches(view, rec, fresh)
    except Exception:
        comparison_failed = True                          # e.g. mixed-type lists in a malformed result
    if comparison_failed:
        raise CommitError("rederive_mismatch", ("result_invalid",))
    if mism:
        raise CommitError("rederive_mismatch", tuple(mism))
    done.append("rederive")
    # build
    if rec["value"] is None:
        side = (copy_private or {}).get(rec["seq"])
        if not (isinstance(side, dict) and isinstance(side.get("value"), str) and isinstance(side.get("evidence"), str)):
            raise CommitError("item_invalid", ("private_missing",))
        value, evidence = side["value"], side["evidence"]
    else:
        value, evidence = rec["value"], rec["evidence"]
    bad = [n for n, good in (("value_fp", M.fingerprint(value) == rec["value_fp"]), ("value_len", len(value) == rec["value_len"]),
                             ("evidence_sha", X.sha(evidence) == rec["evidence_sha"])) if not good]
    if bad:
        raise CommitError("item_invalid", tuple(bad))
    if rec["registry_version"] != K.REGISTRY_VERSION:
        raise CommitError("ledger_replay_invalid", ("registry_version",))
    item, evidence = R.build_item(run, candidate, type=rec["type"], key=rec["key"], value=value, evidence=evidence,
                                  sensitivity=rec["sensitivity"], at=rec["adoption"]["at"], valid_from=rec["valid_from"])
    errs = M.validate(item) + tuple(K.validate_key(rec["key"]))
    if errs:
        raise CommitError("item_invalid", errs)
    hit = X.screen_text(value + " " + evidence) or X.contact_or_third_party(rec["key"], value, evidence)
    if hit:
        raise CommitError("screen_refused", (hit[0] if isinstance(hit, tuple) else hit,))
    if X.sensitivity_of(rec["key"], value, evidence) == "private" and rec["sensitivity"] != "private":
        raise CommitError("sensitivity_changed")
    done.append("build")
    # store
    items, state_digest = read_consistent(store)
    done.append("store")
    # gate
    decision = G.evaluate(G.Proposal(item, evidence), [s.item for s in items], planned_at)
    if decision.decision not in (G.ACCEPT, G.REQUIRE_CONFIRMATION):
        raise CommitError("gate_refused", (decision.decision, decision.reason))
    links = tuple(decision.supersedes)
    by_id = {s.item.id: s for s in items}
    if any(i not in by_id for i in links):
        raise CommitError("display_incomplete", ("replaces",))
    try:
        superseded, old_failed = tuple(_old_item(by_id[i]) for i in links), False
    except Exception:
        superseded, old_failed = (), True
    if old_failed:
        raise CommitError("store_unreadable", ("serialize",))
    operation = "apply" if links else "put"
    kind = _expected_kind(item)
    done.append("gate")
    # digests
    digest_error = None
    try:
        item_digest = S.item_digest(item, evidence, links, operation)
    except S.StoreError as e:
        digest_error = _safe_code(e.reason)
    if digest_error is not None:
        raise CommitError("item_invalid", (digest_error,))
    context_digest = _context_digest(run, pack_sha, candidate, rec["hash"], rec["registry_version"])
    done.append("digests")
    # display
    provisional = Plan(candidate, run, pack_sha, rec["registry_version"], rec["hash"], R.canonical_json(rec), operation, kind,
                       item, evidence, links, decision, tuple(items), superseded, state_digest, item_digest, context_digest,
                       "", planned_at, age, max_age, elapsed, REQUIRED_CHECKS[:-1])
    _, display_digest = _display_checked(provisional)
    done.append("display")
    result = replace(provisional, display_digest=display_digest, checks=tuple(done))
    _require_consistent(result)
    return result


# ---- preview, token, answer, comparison ----------------------------------------------------------

def render_preview(plan) -> Preview:
    _require_consistent(plan)
    lines, _ = _display_checked(plan)
    info = (INFO_HEADING, f"lock age     {plan.age_seconds} s (limit {plan.max_age_seconds} s)",
            f"planned at   {plan.planned_at}", f"re-derivation took {plan.rederive_seconds:.3f} s")
    return Preview(lines, info)


def _token_digest(operation, kind, item_digest, state_digest, context_digest, display_digest, nonce) -> str:
    body = {"v": TOKEN_VERSION, "operation": operation, "kind": kind, "item_digest": item_digest,
            "state_digest": state_digest, "context_digest": context_digest, "display_digest": display_digest, "nonce": nonce}
    return _sha(R.canonical_json(body))[:TOKEN_HEX]


def _token(plan, nonce) -> str:
    _require_consistent(plan)
    if not (isinstance(nonce, str) and _HEX32.match(nonce)):
        raise CommitError("bad_input", ("nonce",))
    return _token_digest(plan.operation, plan.kind, plan.item_digest, plan.state_digest, plan.context_digest,
                         plan.display_digest, nonce)


def challenge_token(plan, nonce) -> str:
    return _token(plan, nonce)


def expected_answer(plan, nonce) -> str:
    _require_consistent(plan)                              # before any Plan field is read
    return f"{plan.kind} {_token(plan, nonce)}"


def check_answer(text, plan, nonce, issued_at, answered_at) -> str:
    """`answered_at` (the confirmation instant) if `text` is exactly the expected answer, given within the window; else a
    CommitError. One attempt: there is no retry here."""
    _require_consistent(plan)
    want = expected_answer(plan, nonce)
    issued, answered, planned = _stamp_dt(issued_at), _stamp_dt(answered_at), _stamp_dt(plan.planned_at)
    if not planned <= issued <= answered:
        raise CommitError("clock_regressed")
    if (answered - issued).total_seconds() > CHALLENGE_SECONDS:
        raise CommitError("challenge_expired")
    if not isinstance(text, str):
        raise CommitError("challenge_ambiguous", ("not_text",))
    if text == want:
        return answered_at
    word = text.split(" ", 1)[0]
    if word in (KIND_COMMIT, KIND_PRIVATE) and word != plan.kind:
        raise CommitError("challenge_ambiguous", ("wrong_kind",))
    raise CommitError("challenge_declined")


_COMPARED = ("operation", "kind", "candidate", "run", "pack_sha", "registry_version", "lock_hash", "memory_id", "sensitivity",
             "valid_from", "links", "decision", "reason", "item_digest", "state_digest", "context_digest", "display")


def _compare_view(p) -> dict:
    return {"operation": p.operation, "kind": p.kind, "candidate": p.candidate, "run": p.run, "pack_sha": p.pack_sha,
            "registry_version": p.registry_version, "lock_hash": p.lock_hash, "memory_id": p.item.id,
            "sensitivity": p.item.sensitivity, "valid_from": p.item.valid_from, "links": p.links,
            "decision": p.decision.decision, "reason": p.decision.reason, "item_digest": p.item_digest,
            "state_digest": p.state_digest, "context_digest": p.context_digest, "display": p.display_digest}


def compare_plans(a, b) -> list:
    """Names of the fields that differ between two plans (empty = identical). A non-Plan is ["bad_plan"], an inconsistent
    Plan is ["plan_inconsistent"], and a second plan made at an earlier instant adds "clock_regressed". Age, the plan
    instants, the re-derivation time and the nonce are deliberately not compared."""
    if not (isinstance(a, Plan) and isinstance(b, Plan)):
        return ["bad_plan"]
    if verify_plan_integrity(a) or verify_plan_integrity(b):
        return ["plan_inconsistent"]
    va, vb = _compare_view(a), _compare_view(b)
    out = [k for k in _COMPARED if va[k] != vb[k]]
    if _stamp_dt(b.planned_at) < _stamp_dt(a.planned_at):
        out.append("clock_regressed")
    return out
