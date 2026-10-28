"""Memory 2.0, Tier 2 evidence and pilot gate (D1, D2, D6, D7).

WHAT THIS IS. The file-backed half of the second confirmation: an append-only, hash-chained EVIDENCE LOG kept under the
review root beside the receipts, the manually created PILOT APPROVAL file, and a `Tier2Gate` that the committer calls
through dependency injection. The confirmer itself (the phone side) is injected too, so this module never talks to a
network, a Keychain or a terminal.

WHAT THIS IS NOT. It is not authentication, and it does not defend against same-account code that bypasses the committer,
calls the store, or edits the vault, the evidence log or the approval file directly. The chain detects accidental damage
and naive edits, not a writer who recomputes it. The receipt schema is closed and is untouched: this log is a separate
file correlated to receipt intents by the nonce hash.

THE FILES. `tier2-evidence.jsonl` (mode 0600, one JSON record per line, each carrying `prev` and `hash`) and, created by the
person and never by this code, `pilot-approval.json` (mode 0600, exactly `candidate`, `lock_hash`, `approved_at`).

LIMITS (named constants, tested): at most `MAX_SENDS` confirmations sent per (candidate, lock hash) since the approval,
at least `MIN_SPACING_S` seconds between sends, approvals older than `APPROVAL_MAX_AGE_S` rejected, and at most `PILOT_MAX_COMMITS` successful
commits in total (it was one for the first pilot; 10 since 2026-10-19).
"""
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone

EVIDENCE_FILE = "tier2-evidence.jsonl"
APPROVAL_FILE = "pilot-approval.json"
MAX_SENDS = 3
MIN_SPACING_S = 30
APPROVAL_MAX_AGE_S = 24 * 3600
PILOT_MAX_COMMITS = 10          # was 1 for the first pilot; raised to 10 on Yousif's approval, 2026-10-19 (each commit still needs its own approval file, typed challenge and phone reply)
ALLOWED_KINDS = ("commit",)
VERSION = "evidence/1"
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_GENESIS = "0" * 64
_EVENTS = ("tier2_sent", "tier2_confirmed")
_BIND = ("candidate", "lock_hash", "operation", "item_digest", "state_digest", "context_digest", "display_digest",
         "nonce_hash")


class EvidenceError(Exception):
    def __init__(self, reason, details=()):
        super().__init__(reason)
        self.reason, self.details = reason, tuple(details)


class _M:
    """The one thing needed from the memory core: the strict timestamp parse (kept local so this module imports no
    Memory 2.0 layer, and in particular not the receipts)."""

    @staticmethod
    def parse_ts(s):
        if not isinstance(s, str) or not _TS.match(s):
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


M = _M


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(rec) -> str:
    return hashlib.sha256(_canon({k: v for k, v in rec.items() if k != "hash"}).encode()).hexdigest()


def nonce_hash(nonce) -> str:
    """Exactly how the receipts hash the nonce into an intent, so the two logs can be correlated."""
    return hashlib.sha256(nonce.encode()).hexdigest()[:32]


class EvidenceLog:
    def __init__(self, directory):
        self.path = os.path.join(directory, EVIDENCE_FILE)

    def records(self):
        """The validated records, or EvidenceError. A missing file is an empty log. Read-only."""
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return []
        except OSError as e:
            raise EvidenceError("evidence_unreadable", (errno.errorcode.get(e.errno, "oserror"),)) from None
        with os.fdopen(fd, "rb") as f:
            raw = f.read()
        return self._validate(raw)

    @staticmethod
    def _validate(raw):
        out, prev = [], _GENESIS
        if raw and not raw.endswith(b"\n"):
            raise EvidenceError("evidence_corrupt", ("unterminated_line",))
        for n, line in enumerate(raw.split(b"\n")[:-1] if raw else [], 1):
            try:
                rec = json.loads(line.decode("utf-8"))
            except ValueError:
                raise EvidenceError("evidence_corrupt", (f"line_{n}_unreadable",)) from None
            if not isinstance(rec, dict) or rec.get("v") != VERSION or rec.get("seq") != n:
                raise EvidenceError("evidence_corrupt", (f"line_{n}_shape",))
            if rec.get("prev") != prev or rec.get("hash") != _hash(rec):
                raise EvidenceError("evidence_corrupt", (f"line_{n}_chain",))
            if rec.get("event") not in _EVENTS or M.parse_ts(rec.get("ts")) is None:
                raise EvidenceError("evidence_corrupt", (f"line_{n}_fields",))
            out.append(rec)
            prev = rec["hash"]
        return out

    def append(self, event, fields, ts):
        """Append one record under an exclusive lock on the file, fsync, return it. Any failure is EvidenceError."""
        if event not in _EVENTS or M.parse_ts(ts) is None:
            raise EvidenceError("bad_record")
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        except OSError as e:
            raise EvidenceError("evidence_unwritable", (errno.errorcode.get(e.errno, "oserror"),)) from None
        try:
            with os.fdopen(fd, "r+b") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                mode = stat.S_IMODE(os.fstat(f.fileno()).st_mode)
                if mode & 0o077:
                    raise EvidenceError("evidence_permissions", (oct(mode),))
                f.seek(0)
                recs = self._validate(f.read())
                rec = {"v": VERSION, "seq": len(recs) + 1, "prev": recs[-1]["hash"] if recs else _GENESIS, "ts": ts,
                       "event": event}
                rec.update(fields)
                rec["hash"] = _hash(rec)
                f.write((_canon(rec) + "\n").encode())
                f.flush()
                os.fsync(f.fileno())
                return rec
        except EvidenceError:
            raise
        except OSError as e:
            raise EvidenceError("evidence_unwritable", (errno.errorcode.get(e.errno, "oserror"),)) from None

    def diagnose(self):
        """READ-ONLY report on the log, for the person who has to deal with a refusal. Never writes, never repairs.
        {"ok", "records", "last_hash", "problem", "bad_line", "torn_tail"}: `records` counts the valid prefix."""
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return {"ok": True, "records": 0, "last_hash": None, "problem": None, "bad_line": None, "torn_tail": False}
        except OSError:
            return {"ok": False, "records": 0, "last_hash": None, "problem": "evidence_unreadable", "bad_line": None,
                    "torn_tail": False}
        torn = bool(raw) and not raw.endswith(b"\n")
        complete = raw.split(b"\n")[:-1]
        good, last = 0, None
        for i in range(len(complete)):
            try:
                recs = self._validate(b"\n".join(complete[:i + 1]) + b"\n")
            except EvidenceError as e:
                return {"ok": False, "records": good, "last_hash": last, "problem": e.details[0] if e.details else e.reason,
                        "bad_line": i + 1, "torn_tail": torn}
            good, last = len(recs), recs[-1]["hash"]
        return {"ok": not torn, "records": good, "last_hash": last, "problem": "unterminated_line" if torn else None,
                "bad_line": good + 1 if torn else None, "torn_tail": torn}

    def sent_since(self, candidate, lock_hash, since):
        """The `tier2_sent` records for this (candidate, lock hash) stamped at or after `since`."""
        t0 = M.parse_ts(since)
        return [r for r in self.records() if r["event"] == "tier2_sent" and r.get("candidate") == candidate
                and r.get("lock_hash") == lock_hash and M.parse_ts(r["ts"]) >= t0]


QUARANTINE_PREFIX = EVIDENCE_FILE + ".quarantined"


def read_approval(path, *, uid=None):
    """The approval dict, or EvidenceError. Never creates, repairs or chmods it."""
    return read_approval_ex(path, uid=uid)[0]


def read_approval_ex(path, *, uid=None):
    """(approval dict, its mtime in epoch seconds)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)         # one descriptor: checked and read are the same file
    except FileNotFoundError:
        raise EvidenceError("approval_missing") from None
    except OSError as e:
        raise EvidenceError("approval_not_a_file" if e.errno == errno.ELOOP else "approval_unreadable") from None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise EvidenceError("approval_not_a_file")
        if stat.S_IMODE(st.st_mode) != 0o600:
            raise EvidenceError("approval_permissions", (oct(stat.S_IMODE(st.st_mode)),))
        if st.st_uid != (os.getuid() if uid is None else uid):
            raise EvidenceError("approval_owner")
        try:
            data = json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            raise EvidenceError("approval_unreadable") from None
    if not (isinstance(data, dict) and set(data) == {"candidate", "lock_hash", "approved_at"}
            and all(isinstance(v, str) for v in data.values())):
        raise EvidenceError("approval_shape")
    if M.parse_ts(data["approved_at"]) is None or not data["candidate"] or not data["lock_hash"]:
        raise EvidenceError("approval_shape")
    return data, st.st_mtime


class Tier2Gate:
    """What `memory_execute.commit` calls. `confirmer` needs `preflight()` and `confirm(plan, nonce, lines)`; both
    return an object with `ok`, `code`, `update_id`, `reply_date`, `sent_date`."""

    def __init__(self, confirmer, receipts_dir, clock, receipt_records, *, approval_path=None,
                 allowed_kinds=ALLOWED_KINDS, uid=None):
        """`receipt_records()` returns the validated receipt records ([] when there is no log yet) or raises
        EvidenceError; it is injected so this module never imports the receipts."""
        self.confirmer, self.dir, self.clock, self.receipt_records = confirmer, receipts_dir, clock, receipt_records
        self.log = EvidenceLog(receipts_dir)
        self.approval_path = approval_path or os.path.join(receipts_dir, APPROVAL_FILE)
        self.allowed_kinds, self.uid = tuple(allowed_kinds), uid
        self._approval = None

    def __repr__(self):
        return "Tier2Gate(<redacted>)"

    def _now(self):
        value = self.clock()
        if not (isinstance(value, str) and M.parse_ts(value) is not None):
            raise EvidenceError("bad_clock")
        return value

    # step 1: configuration, before anything sensitive is read or shown
    def prepare(self):
        if self.confirmer is None:
            return "not_configured"
        try:
            pre = self.confirmer.preflight()
        except Exception:
            return "preflight_error"
        if not pre.ok:
            return pre.code if pre.code in ("webhook_configured", "network") else "preflight_error"
        try:
            self.log.records()
            self._approval, mtime = read_approval_ex(self.approval_path, uid=self.uid)
            # A person who set a corrupt log aside (rename it to `tier2-evidence.jsonl.quarantined*`) must also have made a
            # NEW approval afterwards: the old one's send counts lived in the file that was set aside.
            for name in os.listdir(self.dir):
                if name.startswith(QUARANTINE_PREFIX) and os.stat(os.path.join(self.dir, name)).st_mtime >= mtime:
                    self._approval = None
                    return "approval_predates_quarantine"
        except EvidenceError as e:
            return e.reason
        except OSError:
            return "evidence_unreadable"
        return None

    # step 2: the plan against the approval, the kinds and the limits, before the person is asked anything
    def check_plan(self, plan):
        if self._approval is None:
            return "not_prepared"
        if plan.kind not in self.allowed_kinds:
            return "kind_not_allowed"
        ap = self._approval
        if plan.candidate != ap["candidate"]:
            return "approval_candidate_mismatch"
        if plan.lock_hash != ap["lock_hash"]:
            return "approval_lock_mismatch"
        try:
            now = M.parse_ts(self._now())
            age = (now - M.parse_ts(ap["approved_at"])).total_seconds()
            if age < 0:
                return "approval_in_future"
            if age > APPROVAL_MAX_AGE_S:
                return "approval_expired"
            records = self.receipt_records()
            intents = {r["seq"]: r for r in records if r["action"] == "intent"}
            done = [r for r in records if r["action"] == "done"]
            if any(intents.get(d["intent_seq"], {}).get("candidate") == ap["candidate"] for d in done):
                return "approval_consumed"
            if len(done) >= PILOT_MAX_COMMITS:
                return "pilot_limit_reached"
            return self._send_limit(ap, now)
        except EvidenceError as e:
            return e.reason

    def _send_limit(self, ap, now):
        sent = self.log.sent_since(ap["candidate"], ap["lock_hash"], ap["approved_at"])
        if len(sent) >= MAX_SENDS:
            return "send_limit_reached"
        if sent and (now - M.parse_ts(sent[-1]["ts"])).total_seconds() < MIN_SPACING_S:
            return "send_too_soon"
        return None

    # step 4: send, wait, and durably record. Returns an object with ok and code.
    def confirm(self, plan, nonce, lines):
        class _R:
            def __init__(self, ok, code, update_id=None):
                self.ok, self.code, self.update_id = ok, code, update_id
        try:
            if self._approval is None:
                return _R(False, "not_prepared")
            now = self._now()
            code = self._send_limit(self._approval, M.parse_ts(now))
            if code:
                return _R(False, code)
            bound = {"candidate": plan.candidate, "lock_hash": plan.lock_hash, "operation": plan.operation,
                     "item_digest": plan.item_digest, "state_digest": plan.state_digest,
                     "context_digest": plan.context_digest, "display_digest": plan.display_digest,
                     "nonce_hash": nonce_hash(nonce)}
            self.log.append("tier2_sent", bound, now)             # counted BEFORE anything is sent
        except EvidenceError as e:
            return _R(False, e.reason)
        res = self.confirmer.confirm(plan, nonce, lines)
        if not res.ok:
            return _R(False, res.code)
        try:
            self.log.append("tier2_confirmed", dict(bound, update_id=res.update_id, sent_date=res.sent_date,
                                                    reply_date=res.reply_date), self._now())
        except EvidenceError as e:
            return _R(False, "evidence_failed:" + e.reason)       # a confirmation that was not recorded is not used
        return _R(True, "confirmed", res.update_id)
