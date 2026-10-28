"""Memory 2.0, Phase 8.2c-2: the commit receipt log, the audit and reconciliation proof. A LIBRARY: nothing here is called
by Jarvis, and it is exercised only against temporary directories.

WHAT A RECEIPT IS. A write-ahead claim recorded before and after one store install. It is not evidence that the install
happened: the evidence is the store's files, which `classify_open_intent` compares with what the intent PREDICTED. So a
crash between "intent" and "done" is resolved from the store, never from the log alone.

THE LOG. `commits.jsonl` (mode 0600, in a 0700 folder the library never creates), one canonical-JSON record per line, hash
chained by the review ledger's own `new_record`/`verify_chain`. Records: open, intent, done, aborted, anomaly. Ids, digests,
nonce hashes and codes only: no memory values and no evidence text. Sidecar `commits.lock` is the lock (never the data
file); `commits.blocked` is an evidence marker, never the thing a block depends on.

STATE CHAIN. Every closing record carries the store's `state_digest`; every intent carries `state_digest_before`, which must
equal the previous closing digest. "The log explains the store" means exactly that check passing on every startup. A
condition that blocks is RECOMPUTED from the live store and the log each time, so deleting a marker never lifts it.

APPEND RULES (`ReceiptLog.append`): take the sidecar lock; check the folder and file (owner, mode, type, link count);
read the whole file once and require it canonical, chained and a valid state machine; refuse if the evidence marker
exists; build the next record from the validated tail and validate the resulting log; write in a loop until every byte is
out; fsync the file (and the folder on creation). Any failure after the write begins POISONS the object (all later appends
refused) and raises `receipts_append_ambiguous`. Nothing is ever truncated, repaired or removed.

PROOF (`classify_open_intent`), never from a digest alone. COMMITTED: the complete file map equals `manifest_after`, the
digest equals `state_digest_after`, and the nonce rows are `nonces_before` less prune-eligible rows plus exactly the one new
row. NOT_WRITTEN: the file map equals `manifest_before`, the digest equals `state_digest_before`, and the nonce rows equal
`nonces_before`. Anything else is INCONSISTENT with named codes. The store prunes rows older than (its clock - 24 h); its
clock is within the authorization window of `confirmed_at`, so a row older than `confirmed_at - 24 h` MUST be gone, a row at
or after `confirmed_at - 24 h + window` MUST remain, and the band between may be either.

RETRY (amendment). The id of a memory is a function of the reviewed decision, so a retry after a clean abort reuses it. A new
intent for an id is permitted only when EVERY earlier intent for it was closed by `aborted` with the code
`crashed_before_write` and none had an anomaly recorded against it, and at most `MAX_INTENTS_PER_ID` (3, the first attempt
included) may exist; the fourth is `RETRY_LIMIT`. Any other history (done, a different abort code, any anomaly) is
`DUPLICATE_INTENT` and needs a new decision. Row 6 also requires the live nonce rows to equal the rows as of the last closing
record (`closing_snapshot_digest`), so a nonce-only change cannot pass startup.

INTENTS HAVE ONE DOOR. `ReceiptLog.append` refuses an `intent`. `ReceiptLog.append_intent` does, inside the receipts lock and
in this order: lock, read and validate the log, take a fresh store snapshot, `check_new_intent`, write and fsync.
THE LIMIT, stated plainly: the receipts lock does not exclude non-cooperating store writers or direct file edits, and the
interval between the snapshot and the write is NOT atomic against them. Cooperating committers are excluded by the single
open intent. A stale intent can never authorize a write: the store's own Authorization state check (under its exclusive
lock) and the post-install proof remain the barriers, and a change they cannot prevent is detected and blocks.

AUDIT / STARTUP (`startup_check`, `audit`): READ-ONLY and fail-closed; the rows of the approved startup matrix. `audit`
returns facts, inferences and ONE proposed record (`done`/`aborted`/`anomaly`) that it never appends: appending is a
person's typed act in the later execute layer.

LIMITS, stated and pinned by tests. The hashes are unkeyed: code running as the same user can forge a fully consistent log
and store. A consistent rollback of store and log together is a valid earlier state. Deleting trailing records that changed
no store state (`aborted`, `anomaly`) is undetectable and loses audit history only. `fsync` on macOS is not a full flush.
"""
import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import timedelta

from backtalk import memory as M
from backtalk import memory_review as R
from backtalk import memory_store as S

FORMAT = "memory-commit-receipts/1"
RECEIPTS, LOCK, BLOCKED = "commits.jsonl", "commits.lock", "commits.blocked"
MAX_BYTES = 8 << 20                               # a receipts file this large is not one
NONCE_KEEP = 24 * 3600                            # the store's nonce retention (pinned equal to the store's by a test)
_CHUNK = 1 << 16
MAX_INTENTS_PER_ID = 3                            # intents ever allowed for one memory id, the first attempt included
RETRYABLE_ABORT_CODES = frozenset({"crashed_before_write"})
_HEX64, _HEX32 = re.compile(r"^[0-9a-f]{64}\Z"), re.compile(r"^[0-9a-f]{32}\Z")
_MEM = re.compile(r"^mem_[0-9a-f]{12}\Z")
_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}\Z")
_IFMT, _IFREG, _IFDIR = 0o170000, 0o100000, 0o040000

ENVELOPE = frozenset({"seq", "prev", "at", "hash", "action"})
SPECS = {
    "open": frozenset({"format", "baseline_state_digest", "baseline_snapshot_digest"}),
    "intent": frozenset({"candidate", "lock_hash", "context_digest", "operation", "memory_id", "links", "item_digest",
                         "nonce_hash", "confirmed_at", "apply_now", "state_digest_before", "snapshot_digest_before",
                         "manifest_before", "nonces_before", "manifest_after", "state_digest_after"}),
    "done": frozenset({"intent_seq", "state_digest", "snapshot_digest", "recovered"}),
    "aborted": frozenset({"intent_seq", "code", "state_digest"}),
    "anomaly": frozenset({"intent_seq", "codes", "state_digest"}),
}


class ReceiptError(Exception):
    """A refusal. `reason` is a short code; `details` are codes and ids only, never any content."""

    def __init__(self, reason, details=()):
        super().__init__(reason + (": " + ", ".join(map(str, details)) if details else ""))
        self.reason, self.details = reason, tuple(details)


class _IOOps:
    """Every OS call the receipt log makes, in one place so a test can make any of them short or fail."""
    open = staticmethod(os.open)
    close = staticmethod(os.close)
    fstat = staticmethod(os.fstat)
    lstat = staticmethod(os.lstat)
    read = staticmethod(os.read)
    write = staticmethod(os.write)
    fsync = staticmethod(os.fsync)
    flock = staticmethod(fcntl.flock)


_IO = _IOOps()


# ---- records: builders (fields only; the envelope is added by the log) ---------------------------------------------

def open_fields(snapshot):
    if not (isinstance(snapshot, S.Snapshot) and snapshot.ok):
        raise ReceiptError("snapshot_not_ok", (getattr(snapshot, "reason", "not_a_snapshot"),))
    return {"action": "open", "format": FORMAT, "baseline_state_digest": snapshot.state_digest,
            "baseline_snapshot_digest": snapshot.snapshot_digest}


def predict_after(manifest_before, stored_items, item, evidence, links, apply_now):
    """The file map the store will hold after installing `item` (superseding `links`) at `apply_now`: [[name, sha], ...].
    It reuses the store's own `serialize` and its construction of an ended record, so the bytes are the store's bytes."""
    try:
        before = dict(S.normalized_entries(manifest_before))
        by_id = {s.item.id: s for s in stored_items}
        new = replace(item, supersedes=tuple(links))
        if item.id + ".md" in before:
            raise ReceiptError("prediction_failed", ("memory_id_exists",))
        after = dict(before)
        after[item.id + ".md"] = hashlib.sha256(S.serialize(new, evidence).encode()).hexdigest()
        for oid in links:
            old = by_id.get(oid)
            if old is None or oid + ".md" not in before:
                raise ReceiptError("prediction_failed", ("link_not_stored",))
            ended = replace(old.item, status="superseded", updated=apply_now, valid_to=new.valid_from)
            after[oid + ".md"] = hashlib.sha256(S.serialize(ended, old.evidence).encode()).hexdigest()
        return S.normalized_entries(after.items())
    except S.StoreError as e:
        raise ReceiptError("prediction_failed", (e.reason,)) from None


def intent_fields(snapshot, plan, *, nonce, confirmed_at, apply_now):
    """The `intent` fields for one planned install, from a successful snapshot and a plan (read by attribute). The nonce
    is hashed exactly as the store hashes it for its log. Nothing here touches a file."""
    if not (isinstance(snapshot, S.Snapshot) and snapshot.ok):
        raise ReceiptError("snapshot_not_ok", (getattr(snapshot, "reason", "not_a_snapshot"),))
    links = [str(x) for x in plan.decision.supersedes]
    before = S.normalized_entries(snapshot.files)
    after = predict_after(before, snapshot.items, plan.item, plan.evidence, links, apply_now)
    return {"action": "intent", "candidate": plan.candidate, "lock_hash": plan.lock_hash,
            "context_digest": plan.context_digest, "operation": plan.operation, "memory_id": plan.item.id,
            "links": links, "item_digest": plan.item_digest,
            "nonce_hash": hashlib.sha256(nonce.encode()).hexdigest()[:32], "confirmed_at": confirmed_at,
            "apply_now": apply_now, "state_digest_before": snapshot.state_digest,
            "snapshot_digest_before": snapshot.snapshot_digest, "manifest_before": before,
            "nonces_before": S.normalized_nonce_rows(snapshot.nonces), "manifest_after": after,
            "state_digest_after": S.manifest_digest(after)}


def done_fields(intent_seq, snapshot, recovered=False):
    return {"action": "done", "intent_seq": intent_seq, "state_digest": snapshot.state_digest,
            "snapshot_digest": snapshot.snapshot_digest, "recovered": bool(recovered)}


def aborted_fields(intent_seq, code, state_digest):
    return {"action": "aborted", "intent_seq": intent_seq, "code": code, "state_digest": state_digest}


def anomaly_fields(intent_seq, codes, state_digest):
    return {"action": "anomaly", "intent_seq": intent_seq, "codes": list(codes), "state_digest": state_digest}


# ---- validation: fields, then the whole log as a state machine ------------------------------------------------------

def _is(pattern, value):
    return isinstance(value, str) and bool(pattern.match(value))


def _ts(value):
    return isinstance(value, str) and M.parse_ts(value) is not None


def _canonical_pairs(value):
    try:
        ok = isinstance(value, list) and S.normalized_entries(value) == value
    except (TypeError, ValueError):
        ok = False
    return ok


def _canonical_rows(value):
    try:
        ok = isinstance(value, list) and S.normalized_nonce_rows(value) == value
    except (TypeError, ValueError):
        ok = False
    return ok


def _seq(value):
    return type(value) is int and value >= 0


def field_problem(rec):
    """None for a record whose fields are exactly right for its action, else a stable code. Envelope fields are the
    chain's business; this checks shape, types and formats."""
    if not isinstance(rec, dict):
        return "FIELDS:not_a_record"
    action = rec.get("action")
    if action not in SPECS:
        return "FORMAT_UNKNOWN:action"
    if set(rec) != ENVELOPE | SPECS[action]:
        return f"FIELDS:{action}:keys"
    if not (type(rec["seq"]) is int and _is(_HEX64, rec["prev"]) and _ts(rec["at"]) and _is(_HEX64, rec["hash"])):
        return f"FIELDS:{action}:envelope"
    if action == "open" and rec["format"] != FORMAT:
        return "FORMAT_UNKNOWN:format"
    return None if _CHECKS[action](rec) else f"FIELDS:{action}:values"


def _open_ok(rec):
    return _is(_HEX64, rec["baseline_state_digest"]) and _is(_HEX64, rec["baseline_snapshot_digest"])


def _intent_ok(rec):
    return (_is(_CODE, rec["candidate"]) and all(_is(_HEX64, rec[k]) for k in (
                "lock_hash", "context_digest", "item_digest", "state_digest_before", "snapshot_digest_before",
                "state_digest_after"))
            and rec["operation"] in ("put", "apply") and _is(_MEM, rec["memory_id"])
            and isinstance(rec["links"], list) and all(_is(_MEM, x) for x in rec["links"])
            and len(set(rec["links"])) == len(rec["links"]) and rec["memory_id"] not in rec["links"]
            and _is(_HEX32, rec["nonce_hash"]) and _ts(rec["confirmed_at"]) and _ts(rec["apply_now"])
            and _canonical_pairs(rec["manifest_before"]) and _canonical_pairs(rec["manifest_after"])
            and _canonical_rows(rec["nonces_before"]))


def _done_ok(rec):
    return (_seq(rec["intent_seq"]) and _is(_HEX64, rec["state_digest"]) and _is(_HEX64, rec["snapshot_digest"])
            and type(rec["recovered"]) is bool)


def _aborted_ok(rec):
    return _seq(rec["intent_seq"]) and _is(_CODE, rec["code"]) and _is(_HEX64, rec["state_digest"])


def _anomaly_ok(rec):
    return ((rec["intent_seq"] is None or _seq(rec["intent_seq"])) and isinstance(rec["codes"], list)
            and 1 <= len(rec["codes"]) <= 16 and all(_is(_CODE, c) for c in rec["codes"])
            and (rec["state_digest"] is None or _is(_HEX64, rec["state_digest"])))


_CHECKS = {"open": _open_ok, "intent": _intent_ok, "done": _done_ok, "aborted": _aborted_ok, "anomaly": _anomaly_ok}


def intent_problem(rec):
    """None when an intent agrees with itself: its digests match its manifests and nonce rows, and the two manifests
    differ only at the new memory and the ones it supersedes. An edited or malformed intent cannot prove anything."""
    before, after = dict(rec["manifest_before"]), dict(rec["manifest_after"])
    if S.manifest_digest(rec["manifest_before"]) != rec["state_digest_before"]:
        return "digest_before"
    if S.manifest_digest(rec["manifest_after"]) != rec["state_digest_after"]:
        return "digest_after"
    if S.snapshot_digest(rec["manifest_before"], rec["nonces_before"]) != rec["snapshot_digest_before"]:
        return "snapshot_before"
    if rec["operation"] == "put" and rec["links"]:
        return "put_with_links"
    new_name = rec["memory_id"] + ".md"
    if new_name in before or new_name not in after or set(before) - set(after):
        return "new_file"
    if any(link + ".md" not in before for link in rec["links"]):
        return "link_not_stored"
    changed = {n for n in after if before.get(n) != after[n]}
    if changed != {new_name} | {link + ".md" for link in rec["links"]}:
        return "changed_set"
    if rec["nonce_hash"] in {row[0] for row in rec["nonces_before"]}:
        return "nonce_already_logged"
    return None


@dataclass(frozen=True)
class LogState:
    closing_digest: str            # the store state the log says it ended in
    open_intent: object            # the one intent with no done/aborted yet, or None
    open_anomalous: bool           # an anomaly was recorded against it
    blocked: bool                  # a global anomaly: nothing may follow
    nonce_hashes: frozenset        # the nonce hash of every intent ever recorded
    last_hash: str
    count: int
    closing_snapshot_digest: str   # the snapshot digest (files AND nonce rows) as of the last closing record
    undone_nonce_hashes: frozenset  # nonce hashes of intents that are open or aborted: no live row may carry one


def retry_refusal(prior):
    """None when a new intent for a memory id may follow the intents already recorded for it, else a stable code.
    `prior` is a list of {"status": "done"|"aborted" (absent while open), "code": abort code, "anomalous": bool}. A retry is
    only ever allowed after clean aborts; any other history needs a new decision, and attempts are bounded."""
    for p in prior:
        if p.get("status") != "aborted" or p.get("anomalous") or p.get("code") not in RETRYABLE_ABORT_CODES:
            return "DUPLICATE_INTENT"
    if len(prior) >= MAX_INTENTS_PER_ID:
        return "RETRY_LIMIT"
    return None


def validate_records(records):
    """(problems, LogState or None). `problems` is empty for a log that is intact and a valid state machine."""
    if not isinstance(records, list) or not records:
        return ["EMPTY"], None
    if not all(isinstance(r, dict) for r in records):
        return ["FIELDS:not_a_record"], None
    chain = R.verify_chain(records)
    if chain:
        return [f"CHAIN:{p}" for p in chain], None
    closing, closing_snapshot, open_intent, anomalous, blocked = None, None, None, False, False
    nonces, problems = set(), []
    history, entry_of, hash_of = {}, {}, {}
    for i, rec in enumerate(records):
        bad = field_problem(rec)
        if bad:
            return [bad], None
        action = rec["action"]
        if i > 0 and action == "open":
            return ["SECOND_OPEN"], None
        if blocked:
            return ["AFTER_ANOMALY"], None
        if action == "open":
            closing, closing_snapshot = rec["baseline_state_digest"], rec["baseline_snapshot_digest"]
        elif action == "intent":
            if open_intent is not None:
                return ["INTENT_OPEN"], None
            if rec["state_digest_before"] != closing:
                return ["STATE_GAP"], None
            inconsistent = intent_problem(rec)
            if inconsistent:
                return [f"INTENT_INCONSISTENT:{inconsistent}"], None
            if rec["nonce_hash"] in nonces:
                return ["DUPLICATE_NONCE"], None
            refusal = retry_refusal(history.get(rec["memory_id"], []))
            if refusal:
                return [refusal], None
            nonces.add(rec["nonce_hash"])
            entry = {}                                           # no status yet: it is open
            history.setdefault(rec["memory_id"], []).append(entry)
            entry_of[rec["seq"]], hash_of[rec["seq"]] = entry, rec["nonce_hash"]
            open_intent, anomalous = rec, False
        elif action in ("done", "aborted"):
            if open_intent is None or rec["intent_seq"] != open_intent["seq"]:
                prior = [r for r in records[:i] if r["action"] == "intent" and r["seq"] == rec["intent_seq"]]
                return ["DOUBLE_CLOSE" if prior else "CLOSE_WITHOUT_INTENT"], None
            want = open_intent["state_digest_after"] if action == "done" else open_intent["state_digest_before"]
            if rec["state_digest"] != want:
                return ["CLOSE_DIGEST_MISMATCH"], None
            entry = entry_of[open_intent["seq"]]
            entry["status"] = action
            entry["code"] = rec["code"] if action == "aborted" else None
            closing_snapshot = rec["snapshot_digest"] if action == "done" else open_intent["snapshot_digest_before"]
            closing, open_intent, anomalous = rec["state_digest"], None, False
        else:  # anomaly
            if rec["intent_seq"] is None:
                blocked = True
            elif open_intent is None or rec["intent_seq"] != open_intent["seq"]:
                return ["CLOSE_WITHOUT_INTENT"], None
            else:
                anomalous = True
                entry_of[rec["intent_seq"]]["anomalous"] = True
    undone = frozenset(hash_of[seq] for seq, entry in entry_of.items() if entry.get("status") != "done")
    return problems, LogState(closing, open_intent, anomalous, blocked, frozenset(nonces), records[-1]["hash"],
                              len(records), closing_snapshot, undone)


def check_new_intent(state, snapshot, fields):
    """None when a new intent is admissible against the validated log state and a FRESH successful snapshot, else a stable
    code. Pure. Compares the intent with the live store exactly: nothing is trusted because the intent says so."""
    if not isinstance(fields, dict) or fields.get("action") != "intent":
        return "bad_fields"
    if not (isinstance(snapshot, S.Snapshot) and snapshot.ok):
        return "snapshot_not_ok"
    if state is None or state.blocked:
        return "log_blocked"
    if state.open_intent is not None:
        return "intent_open"
    live_rows = {row[0] for row in snapshot.nonces}
    live_files = dict(snapshot.files)
    if fields["memory_id"] + ".md" in live_files:
        return "memory_file_present"
    if snapshot.state_digest != state.closing_digest:
        return "state_gap"
    if live_rows & state.undone_nonce_hashes:
        return "nonce_row_bound_to_undone_intent"
    if snapshot.snapshot_digest != state.closing_snapshot_digest:
        return "nonce_rows_changed"
    if fields["state_digest_before"] != snapshot.state_digest:
        return "state_stale"
    if fields["manifest_before"] != S.normalized_entries(snapshot.files):
        return "manifest_stale"
    if fields["snapshot_digest_before"] != snapshot.snapshot_digest:
        return "snapshot_stale"
    if fields["nonces_before"] != S.normalized_nonce_rows(snapshot.nonces):
        return "nonces_stale"
    if fields["nonce_hash"] in state.nonce_hashes or fields["nonce_hash"] in live_rows:
        return "nonce_reused"
    return None


# ---- reading the file ----------------------------------------------------------------------------------------------

def _kind(st):
    return st.st_mode & _IFMT


def _folder_ok(directory):
    """True for an existing folder of ours that nobody else can read, write or enter."""
    try:
        st = _IO.lstat(directory)
    except OSError:
        ok = False
    else:
        ok = _kind(st) == _IFDIR and st.st_uid == os.getuid() and not st.st_mode & 0o077
    return ok


def _marker_present(directory):
    """True when the evidence marker exists OR cannot be shown absent: an unreadable marker counts as present."""
    try:
        _IO.lstat(os.path.join(directory, BLOCKED))
    except FileNotFoundError:
        present = False
    except OSError:
        present = True
    else:
        present = True
    return present


def _file_ok(st):
    return _kind(st) == _IFREG and st.st_uid == os.getuid() and st.st_mode & 0o777 == 0o600 and st.st_nlink == 1


def _read_records(path):
    """The parsed records of an existing receipts file; ReceiptError otherwise. One open, one read loop, bounded."""
    try:
        fd = _IO.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        raise ReceiptError("receipts_missing") from None
    except OSError:
        raise ReceiptError("receipts_unavailable") from None
    try:
        st = _IO.fstat(fd)
        if not _file_ok(st) or st.st_size > MAX_BYTES:
            raise ReceiptError("receipts_unavailable")
        chunks, total = [], 0
        for _ in range(MAX_BYTES // _CHUNK + 2):
            chunk = _IO.read(fd, _CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise ReceiptError("receipts_unavailable")
            chunks.append(chunk)
        else:
            raise ReceiptError("receipts_unavailable")
    finally:
        _IO.close(fd)
    try:
        text = b"".join(chunks).decode()
    except UnicodeDecodeError:
        raise ReceiptError("receipts_corrupt", ("utf8",)) from None
    if not text.endswith("\n"):
        raise ReceiptError("receipts_corrupt", ("unterminated",))
    records = []
    for n, line in enumerate(text[:-1].split("\n")):
        try:
            obj = json.loads(line)
        except ValueError:
            raise ReceiptError("receipts_corrupt", (f"json:{n}",)) from None
        if not isinstance(obj, dict) or R.canonical_json(obj) != line:
            raise ReceiptError("receipts_corrupt", (f"not_canonical:{n}",))
        records.append(obj)
    return records


# ---- the log -------------------------------------------------------------------------------------------------------

class ReceiptLog:
    """Appends validated records to `<directory>/commits.jsonl`. The directory must already exist (0700, owned by the
    user): the library never creates it. A failure after a write began poisons the object for good."""

    def __init__(self, directory):
        self.directory = directory
        self.path = os.path.join(directory, RECEIPTS)
        self.lock_path = os.path.join(directory, LOCK)
        self.blocked_path = os.path.join(directory, BLOCKED)
        self.poisoned = False

    def records(self):
        """The validated records, or ReceiptError. Read-only."""
        records = _read_records(self.path)
        problems, _ = validate_records(records)
        if problems:
            raise ReceiptError("receipts_corrupt", tuple(problems))
        return records

    def create(self, snapshot, at):
        """First use only: the file must not exist and the store must be empty (an existing store is never baselined)."""
        fields = open_fields(snapshot)
        if snapshot.files or snapshot.nonces:
            raise ReceiptError("baseline_refused")
        return self._write(fields, at, None, creating=True)

    def append(self, fields, at, *, expect_prev=None):
        """Append a done, aborted or anomaly record. An `intent` has its own door: `append_intent`."""
        if not isinstance(fields, dict) or fields.get("action") in (None, "open"):
            raise ReceiptError("bad_fields")
        if fields.get("action") == "intent":
            raise ReceiptError("use_append_intent")
        return self._write(fields, at, expect_prev, creating=False)

    def append_intent(self, fields, at, store, *, expect_prev=None):
        """The ONLY way to append an intent. Inside the receipts lock, in this order: read and validate the log, take a
        fresh snapshot of `store`, `check_new_intent`, write and fsync. The receipts lock does not exclude a writer that
        does not take it; see the module docstring for what the check does and does not guarantee."""
        if not isinstance(fields, dict) or fields.get("action") != "intent":
            raise ReceiptError("bad_fields")
        if not callable(getattr(store, "snapshot", None)):
            raise ReceiptError("bad_fields", ("store",))
        return self._write(fields, at, expect_prev, creating=False, store=store)

    # -- internals --

    def _write(self, fields, at, expect_prev, creating, store=None):
        if self.poisoned:
            raise ReceiptError("receipts_poisoned")
        if not _folder_ok(self.directory):
            raise ReceiptError("receipts_unavailable")
        if _marker_present(self.directory):
            raise ReceiptError("receipts_blocked")
        lock_fd = self._lock(creating)
        try:
            if creating:
                if os.path.lexists(self.path):
                    raise ReceiptError("receipts_exists")
                records = []
            else:
                records = self.records()
                if _marker_present(self.directory):
                    raise ReceiptError("receipts_blocked")
            if expect_prev is not None and (not records or records[-1]["hash"] != expect_prev):
                raise ReceiptError("receipts_changed")
            try:
                rec = R.new_record(records, at, fields)
            except (ValueError, TypeError):
                raise ReceiptError("bad_fields", ("record",)) from None
            problems, _ = validate_records(records + [rec])
            if problems:
                raise ReceiptError("record_refused", tuple(problems))
            if fields.get("action") == "intent":
                if store is None:
                    raise ReceiptError("use_append_intent")
                refusal = check_new_intent(validate_records(records)[1], store.snapshot(), fields)
                if refusal:
                    raise ReceiptError("intent_refused", (refusal,))
            data = (R.canonical_json(rec) + "\n").encode()
            self._put(data, creating)
            return rec
        finally:
            _IO.close(lock_fd)

    def _lock(self, creating):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_CREAT if creating else 0)
        try:
            fd = _IO.open(self.lock_path, flags, 0o600)
        except FileNotFoundError:
            raise ReceiptError("receipts_lock_missing") from None
        except OSError:
            raise ReceiptError("receipts_unavailable") from None
        try:
            if not _file_ok(_IO.fstat(fd)):
                raise ReceiptError("receipts_unavailable")
            _IO.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _IO.close(fd)
            raise ReceiptError("receipts_busy") from None
        except BaseException:
            _IO.close(fd)
            raise
        return fd

    def _put(self, data, creating):
        """The write itself. Anything that goes wrong from here on is ambiguous: the bytes may or may not be durable."""
        try:
            flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_CREAT | os.O_EXCL if creating else os.O_APPEND)
            fd = _IO.open(self.path, flags, 0o600)
            try:
                view = memoryview(data)
                for _ in range(len(data) + 1):                     # bounded: every pass must make progress
                    if not view:
                        break
                    wrote = _IO.write(fd, view)
                    if not 0 < wrote <= len(view):
                        raise OSError
                    view = view[wrote:]
                else:
                    raise OSError
                _IO.fsync(fd)
            finally:
                _IO.close(fd)
            if creating:
                dfd = _IO.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    _IO.fsync(dfd)
                finally:
                    _IO.close(dfd)
        except BaseException as e:
            self.poisoned = True
            self._mark_blocked("append_ambiguous")
            if isinstance(e, Exception):
                raise ReceiptError("receipts_append_ambiguous") from None
            raise

    def _mark_blocked(self, reason):
        """Best effort, never raises: evidence for a person, not something a block depends on."""
        try:
            fd = _IO.open(self.blocked_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            try:
                _IO.write(fd, (R.canonical_json({"format": FORMAT, "reason": reason}) + "\n").encode())
                _IO.fsync(fd)
            finally:
                _IO.close(fd)
        except BaseException:
            pass


# ---- the proof -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Proof:
    kind: str                      # "COMMITTED" | "NOT_WRITTEN" | "INCONSISTENT"
    codes: tuple                   # empty for the first two


def classify_open_intent(intent, snapshot):
    """What the store's actual state says about one open intent. Needs a successful snapshot; never from a digest alone."""
    if not (isinstance(snapshot, S.Snapshot) and snapshot.ok):
        return Proof("INCONSISTENT", ("SNAPSHOT_NOT_OK",))
    live = dict(snapshot.files)
    before, after = dict(intent["manifest_before"]), dict(intent["manifest_after"])
    rows_now = set(snapshot.nonces)
    rows_before = {tuple(r) for r in intent["nonces_before"]}
    new_row = (intent["nonce_hash"], intent["memory_id"], intent["confirmed_at"])

    def file_codes(expected):
        codes = set()
        for name in sorted(set(live) | set(expected)):
            if name not in expected:
                codes.add("UNRECEIPTED_FILE")
            elif name not in live:
                codes.add("MISSING_FILE")
            elif live[name] != expected[name]:
                codes.add("MODIFIED_FILE")
        if S.manifest_digest(snapshot.files) != (intent["state_digest_after"] if expected is after
                                                 else intent["state_digest_before"]):
            codes.add("DIGEST_MISMATCH")
        return codes

    def nonce_codes_before():
        return set() if rows_now == rows_before else {"NONCE_ROWS_CHANGED"}

    def nonce_codes_after():
        codes = set()
        if rows_now - rows_before != {new_row}:
            codes.add("NONCE_ROW_MISMATCH")
        c = M.parse_ts(intent["confirmed_at"])
        low = c - timedelta(seconds=NONCE_KEEP)
        high = low + timedelta(seconds=S.AUTH_WINDOW)
        for row in rows_before:
            at = M.parse_ts(row[2])
            if row in rows_now and at < low:
                codes.add("NONCE_PRUNE_VIOLATION")
            if row not in rows_now and at >= high:
                codes.add("NONCE_PRUNE_VIOLATION")
        return codes

    after_codes = file_codes(after) | nonce_codes_after()
    if not after_codes:
        return Proof("COMMITTED", ())
    before_codes = file_codes(before) | nonce_codes_before()
    if not before_codes:
        return Proof("NOT_WRITTEN", ())
    return Proof("INCONSISTENT", tuple(sorted(after_codes | before_codes)))


# ---- startup check and audit (one implementation, read-only) -------------------------------------------------------

@dataclass(frozen=True)
class Assessment:
    row: int                       # the startup-matrix row that decided
    code: str                      # stable reason
    proceed: bool                  # True only for first use (row 3) and a clean log (row 6)
    first_use: bool
    details: tuple
    state: object                  # LogState or None
    snapshot: object               # the Snapshot used, or None
    proof: object                  # Proof for an open intent, else None
    facts: tuple
    inferences: tuple
    proposed: object               # fields of the ONE record a later writer MAY append, or None


def _blocked(row, code, details=(), facts=(), state=None, snapshot=None):
    return Assessment(row, code, False, False, tuple(details), state, snapshot, None, tuple(facts), (), None)


def assess(directory, store):
    """READ-ONLY. Marker, folder, file, chain and state machine, snapshot, comparison, classification, in the approved
    order. Anything not explicitly proceed is a block. Never creates, repairs, appends or removes anything."""
    if _marker_present(directory):
        return _blocked(1, "receipts_blocked")
    if not _folder_ok(directory):
        return _blocked(2, "receipts_unavailable")
    path = os.path.join(directory, RECEIPTS)
    try:
        _IO.lstat(path)
        exists = True
    except FileNotFoundError:
        exists = False
    except OSError:
        return _blocked(2, "receipts_unavailable")
    lock_fd = None
    try:
        if exists:
            try:
                lock_fd = _IO.open(os.path.join(directory, LOCK), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except FileNotFoundError:
                return _blocked(2, "receipts_lock_missing")
            except OSError:
                return _blocked(2, "receipts_unavailable")
            if not _file_ok(_IO.fstat(lock_fd)):
                return _blocked(2, "receipts_unavailable")
            try:
                _IO.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return _blocked(2, "receipts_busy")
        snap = store.snapshot()
        if not exists:
            if not snap.ok:
                return _blocked(5, "snapshot:" + snap.reason, snapshot=snap)
            if not snap.files and not snap.nonces:
                return Assessment(3, "first_use", True, True, (), None, snap, None,
                                  ("no receipts file", "the store is empty"), (), None)
            return _blocked(3, "receipts_missing", facts=(f"{len(snap.files)} memory file(s) and no receipts",),
                            snapshot=snap)
        try:
            records = _read_records(path)
        except ReceiptError as e:
            return _blocked(2 if e.reason == "receipts_unavailable" else 4, e.reason, e.details, snapshot=snap)
        problems, state = validate_records(records)
        if problems:
            return _blocked(4, "receipts_corrupt", problems, snapshot=snap)
        if not snap.ok:
            return _blocked(5, "snapshot:" + snap.reason, state=state, snapshot=snap)
        if state.blocked:
            return _blocked(8, "anomaly_recorded", state=state, snapshot=snap)
        live = snap.state_digest
        intent = state.open_intent
        if intent is None:
            if live != state.closing_digest:
                return _blocked(8, "state_gap", ("live digest differs from the log's last closing digest",),
                                state=state, snapshot=snap)
            stray = sorted(r[0] for r in snap.nonces if r[0] not in state.nonce_hashes)
            if stray:
                return _blocked(9, "unreceipted_commit", stray, state=state, snapshot=snap)
            if snap.snapshot_digest != state.closing_snapshot_digest:
                return _blocked(9, "nonce_rows_changed", ("the nonce rows differ from the last closing record",),
                                state=state, snapshot=snap)
            return Assessment(6, "clean", True, False, (), state, snap, None,
                              ("the log's last closing digest equals the live store",), (), None)
        if live not in (intent["state_digest_before"], intent["state_digest_after"]):
            return _blocked(8, "state_gap", ("an open intent and a store that is neither its before nor its after",),
                            state=state, snapshot=snap)
        proof = classify_open_intent(intent, snap)
        facts = (f"open intent seq {intent['seq']} for {intent['memory_id']}",
                 f"live digest equals the intent's {'after' if live == intent['state_digest_after'] else 'before'} digest")
        if proof.kind == "COMMITTED":
            inferences = ("consistent with a crash after the store commit and before the done record",)
            proposed = done_fields(intent["seq"], snap, recovered=True)
        elif proof.kind == "NOT_WRITTEN":
            inferences = ("consistent with a crash before the store wrote anything",)
            proposed = aborted_fields(intent["seq"], "crashed_before_write", live)
        else:
            inferences = ("the store matches neither side of the intent",)
            proposed = anomaly_fields(intent["seq"], proof.codes, live)
        return Assessment(7, "reconcile_required", False, False, proof.codes, state, snap, proof, facts, inferences,
                          proposed)
    finally:
        if lock_fd is not None:
            _IO.close(lock_fd)


@dataclass(frozen=True)
class Startup:
    proceed: bool
    row: int
    code: str
    first_use: bool
    details: tuple


def startup_check(directory, store):
    """Whether a commit may begin: True only for first use (row 3) or a clean log (row 6). Read-only."""
    a = assess(directory, store)
    return Startup(a.proceed, a.row, a.code, a.first_use, a.details)


@dataclass(frozen=True)
class AuditReport:
    verdict: str                   # "CLEAN" | "RECONCILE_REQUIRED" | "BLOCKED"
    row: int
    code: str
    facts: tuple                   # observations
    inferences: tuple              # marked hypotheses, never evidence
    details: tuple
    proof: object
    proposed_record: object        # fields of the one record a later writer MAY append; this module never appends it


def audit(directory, store):
    """READ-ONLY. The verdict, the facts, the inferences and at most one proposed record. It repairs nothing."""
    a = assess(directory, store)
    if a.proceed:
        verdict = "CLEAN"
    elif a.row == 7 and a.proof is not None and a.proof.kind != "INCONSISTENT":
        verdict = "RECONCILE_REQUIRED"
    else:
        verdict = "BLOCKED"
    return AuditReport(verdict, a.row, a.code, a.facts, a.inferences, a.details, a.proof, a.proposed)
