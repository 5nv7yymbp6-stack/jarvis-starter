"""Persistent scheduler (roadmap Phase F, item 1). PURE CORE: no speech, no model, no network, no wall clock of its own.

It decides WHAT is due and remembers what already ran; it never speaks or acts. A caller (the proactive loop) asks
`due(now)`, hands each Fire to the announcement path, then calls `complete` or `fail`. Nothing here is wired into the
voice line by itself.

STORAGE follows the ledger's convention: an append-only, fsync'd JSONL file (`scheduler.jsonl`) whose lines are the only
state; the in-memory tasks are a fold over them. A record is written BEFORE it is acted on (a fire is durable before the
caller runs it), so a crash never loses or silently repeats a run. A torn LAST line (a crash mid-append, no newline) is
cut off (nothing ever acted on an append that did not return); any other unreadable line makes the file fail closed (`SchedulerCorrupt`): it is never repaired or truncated.
A second process on the same directory is refused (`SchedulerLocked`).

SEMANTICS (deterministic; every number in `now` terms, epoch seconds):
  kinds     once (due_at), every (interval_s from start_at), daily (local "HH:MM" in a named timezone)
  statuses  scheduled -> running -> scheduled (recurring) | done | failed | cancelled | missed (once, too late)
  due(now)  a task is eligible when status is scheduled and (retry_at or next) <= now.
            ONCE, first attempt: late by more than grace_s -> recorded `missed` and returned ONCE as a Fire(missed=True)
            so the caller can still tell the user; it never runs. Otherwise fired.
            RECURRING: any number of missed slots collapse into ONE fire (`coalesced` counts the skipped ones) and `next`
            jumps to the first slot strictly after `now`; a run still in flight is never overlapped.
  fail      bounded retries: attempt n waits min(BACKOFF_BASE_S * 2**(n-1), BACKOFF_MAX_S); after max_retries retries the task
            is `failed` (once) or moves on to its next slot with the failure kept (recurring). A retry ignores grace_s.
  recover   a run found `running` at startup has an unknown outcome: it is treated as one failed attempt (bounded retry),
            never as success and never as "still running". At-least-once: the announcer must dedupe on `Fire.fire_key`
            (stable across every attempt of one slot; run_id changes with each attempt).
  cancel    the task never fires again; a run in flight is flagged and its later complete/fail is ignored.
  dedupe    `key` makes `add_*` idempotent: the same key returns the existing live task instead of a second one.
"""
import fcntl
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 3600.0
DEFAULT_GRACE_S = 6 * 3600.0         # a closed lid or a sleep must not cost the user a reminder
DEFAULT_RETRIES = 3
TITLE_MAX = 200
KINDS = ("once", "every", "daily")
TERMINAL = ("done", "failed", "cancelled", "missed")
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)\Z")
MIN_INTERVAL_S = 1.0
MAX_INTERVAL_S = 366 * 86400.0
MIN_EPOCH, MAX_EPOCH = 946684800.0, 4102444800.0      # years 2000 to 2100


class SchedulerError(Exception):
    pass


class SchedulerCorrupt(SchedulerError):
    pass


class SchedulerLocked(SchedulerError):
    pass


@dataclass(frozen=True)
class Fire:
    """One claimed run. `run_id` is the idempotency key for whatever the caller does with it."""
    task_id: str
    run_id: str
    title: str
    payload: dict
    kind: str
    scheduled_for: float
    late_s: float
    coalesced: int
    attempt: int
    fire_key: str = ""             # stable across every attempt of one slot: what an announcer must dedupe on
    slot: float = 0.0
    missed: bool = False           # a one-time task found too late to run: report it, never run it


@dataclass
class Task:
    id: str
    key: object
    kind: str
    title: str
    payload: dict
    spec: dict
    status: str = "scheduled"
    next: float = 0.0
    retry_at: object = None
    grace_s: float = DEFAULT_GRACE_S
    max_retries: int = DEFAULT_RETRIES
    attempts: int = 0
    runs: int = 0
    failures: int = 0
    run_id: object = None
    cancel_requested: bool = False
    last_run: object = None
    last_status: object = None
    last_error: object = None
    created: float = 0.0
    seq: int = 0
    slot: object = None
    history: list = field(default_factory=list)


def _local_slot(hhmm, tz, after):
    """The first epoch time strictly after `after` that reads `hhmm` on the wall clock of timezone `tz`."""
    m = _HHMM.match(hhmm)
    h, mi = int(m.group(1)), int(m.group(2))
    z = ZoneInfo(tz)
    base = datetime.fromtimestamp(after, z)
    day = base.date()
    for _ in range(3):
        cand = datetime(day.year, day.month, day.day, h, mi, tzinfo=z)
        if cand.timestamp() > after:
            return cand.timestamp()
        day += timedelta(days=1)
    raise SchedulerError("no next daily slot")


def _next_after(task_spec, kind, after):
    if kind == "every":
        iv, start = task_spec["interval_s"], task_spec["start_at"]
        if after < start:
            return start
        n = int((after - start) // iv) + 1
        return start + n * iv
    if kind == "daily":
        return _local_slot(task_spec["hhmm"], task_spec["tz"], after)
    raise SchedulerError("once has no next slot")


def _backoff(attempt):
    return min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)


class Scheduler:
    def __init__(self, directory, *, clock=time.time, fsync=True, lock=True, readonly=False):
        self.dir = directory
        self._clock, self._fsync = clock, fsync
        self._readonly, self._dead = readonly, False
        self.errors = []
        if not readonly:
            os.makedirs(directory, exist_ok=True)
        else:
            lock = False
        self.path = os.path.join(directory, "scheduler.jsonl")
        self._lockfh = None
        if lock:
            fh = open(os.path.join(directory, "scheduler.lock"), "a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                raise SchedulerLocked("another process holds the scheduler") from None
            self._lockfh = fh
        self._tasks = {}
        self._seq = 0
        self._fh = None
        try:
            self._load()
            self._fh = None if readonly else open(self.path, "ab")
        except BaseException:                         # never leave the lock held by an object that failed to build
            self.close()
            raise

    @classmethod
    def read_only(cls, directory):
        """A view of the tasks for a second process (the CLI's `list`): no lock, no write handle, never repairs."""
        return cls(directory, lock=False, readonly=True)

    # -- storage ---------------------------------------------------------------

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            raw = fh.read()
        lines = raw.split(b"\n")
        torn = lines[-1] != b""                       # no trailing newline: the last append may be partial
        body = lines[:-1]
        tail = lines[-1] if torn else None
        for i, line in enumerate(body):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict) or rec.get("t") != "sched":
                    raise ValueError
                self._apply(rec)
            except (ValueError, KeyError, TypeError):
                raise SchedulerCorrupt(f"unreadable record at line {i + 1}") from None
        if tail:
            # A torn final line is an append that never returned, so nothing ever acted on it: it is cut off (the ONLY
            # truncation this module ever does), unless it is a complete record that merely lacks its newline. A line that
            # parses but is not a valid record is corruption, not a tear.
            rec, ok = None, False
            try:
                rec = json.loads(tail)
                ok = isinstance(rec, dict) and rec.get("t") == "sched"
            except ValueError:
                ok = False
            if ok:
                try:
                    self._apply(rec)
                except (KeyError, TypeError, ValueError):
                    raise SchedulerCorrupt("an invalid last record") from None
            if not self._readonly:
                with open(self.path, "r+b") as f:
                    if ok:
                        f.seek(0, os.SEEK_END)
                        f.write(b"\n")
                    else:
                        f.truncate(len(raw) - len(tail))

    def _append(self, rec):
        if self._readonly:
            raise SchedulerError("read-only view")
        if self._dead:
            raise SchedulerError("the scheduler stopped after a write failure")
        rec = {"t": "sched", **rec}
        data = (json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        pos = self._fh.tell()
        try:
            self._fh.write(data)
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
        except OSError as e:
            # Memory has NOT moved: the fold below never ran. Put the file back as it was and stop for good (fail closed),
            # because a half-written or unsynced record means the disk and this process may disagree.
            self._dead = True
            try:
                self._fh.truncate(pos)
                self._fh.flush()
            except Exception:
                pass
            raise SchedulerError(f"write failed: {e.__class__.__name__}") from None
        self._apply(rec)
        return rec

    def close(self):
        try:
            if getattr(self, "_fh", None) is not None:
                self._fh.close()
        finally:
            if self._lockfh:
                self._lockfh.close()
                self._lockfh = None

    # -- the fold --------------------------------------------------------------

    def _apply(self, r):
        op = r["op"]
        if op == "add":
            self._seq += 1
            self._tasks[r["id"]] = Task(id=r["id"], key=r.get("key"), kind=r["kind"], title=r["title"],
                                        payload=r.get("payload") or {}, spec=r["spec"], next=r["next"],
                                        grace_s=r["grace_s"], max_retries=r["max_retries"], created=r["at"],
                                        seq=self._seq)
            return
        t = self._tasks[r["id"]]
        t.history.append((op, r["at"]))
        del t.history[:-20]
        if op == "fire":
            t.status, t.run_id, t.last_run = "running", r["run_id"], r["at"]
            t.slot = r.get("slot", r["scheduled_for"])
            t.runs += 1
            t.attempts = r["attempt"]
            t.retry_at = None
            if t.kind != "once":
                t.next = r["next"]
        elif op == "missed":
            t.status, t.last_status, t.last_error = "missed", "missed", f"missed by {r['late_s']:.0f}s"
        elif op == "done":
            if r["run_id"] == t.run_id and t.status == "running":
                t.last_status, t.last_error, t.attempts, t.run_id = "done", None, 0, None
                t.status = "cancelled" if t.cancel_requested else ("done" if t.kind == "once" else "scheduled")
        elif op == "fail":
            if r["run_id"] == t.run_id and t.status == "running":
                t.last_status, t.last_error, t.run_id = "failed", r["error"], None
                if t.cancel_requested:
                    t.status = "cancelled"
                elif r.get("retry_at") is not None:
                    t.status, t.retry_at = "scheduled", r["retry_at"]
                else:
                    t.failures += 1
                    t.attempts = 0
                    t.status = "failed" if t.kind == "once" else "scheduled"
        elif op == "reanchor":
            t.next = r["next"]
            if "start_at" in r:
                t.spec = {**t.spec, "start_at": r["start_at"]}
        elif op == "cancel":
            t.cancel_requested = True
            if t.status != "running":
                t.status = "cancelled"
            t.retry_at = None
        else:
            raise KeyError(op)

    # -- adding ----------------------------------------------------------------

    @staticmethod
    def _norm_key(key):
        return None if key is None else f"{type(key).__name__}:{key}"

    @staticmethod
    def _epoch(x, name):
        if (not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
                or not MIN_EPOCH <= x <= MAX_EPOCH):
            raise SchedulerError(f"{name} must be a real time between 2000 and 2100")
        return float(x)

    def has_key(self, key) -> bool:
        k = self._norm_key(key)
        return k is not None and any(t.key == k for t in self._tasks.values())

    def _add(self, kind, title, spec, next_at, key, grace_s, max_retries, payload):
        title = " ".join(str(title).split())[:TITLE_MAX]
        if not title:
            raise SchedulerError("a task needs a title")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 10:
            raise SchedulerError("max_retries must be 0 to 10")
        if (not isinstance(grace_s, (int, float)) or isinstance(grace_s, bool) or not math.isfinite(grace_s)
                or not 0 <= grace_s <= 30 * 86400):
            raise SchedulerError("grace_s must be 0 to 30 days")
        nkey = self._norm_key(key)
        if nkey is not None:
            for t in self._tasks.values():
                if t.key == nkey and t.status not in TERMINAL:
                    if (t.kind, t.title, t.spec, t.next) == (kind, title, spec, float(next_at)):
                        return t                      # the identical request again: idempotent
                    raise SchedulerError("that key is already in use by a different live task")
        tid = "sch_" + os.urandom(6).hex()
        self._append({"op": "add", "id": tid, "at": self._clock(), "key": nkey, "kind": kind, "title": title,
                      "spec": spec, "next": float(next_at), "grace_s": float(grace_s), "max_retries": max_retries,
                      "payload": dict(payload or {})})
        return self._tasks[tid]

    def add_once(self, title, due_at, *, key=None, grace_s=DEFAULT_GRACE_S, max_retries=DEFAULT_RETRIES, payload=None):
        due_at = self._epoch(due_at, "due_at")
        return self._add("once", title, {}, due_at, key, grace_s, max_retries, payload)

    def add_every(self, title, interval_s, *, start_at=None, key=None, max_retries=DEFAULT_RETRIES, payload=None):
        if (not isinstance(interval_s, (int, float)) or isinstance(interval_s, bool) or not math.isfinite(interval_s)
                or not MIN_INTERVAL_S <= interval_s <= MAX_INTERVAL_S):
            raise SchedulerError(f"interval_s must be {MIN_INTERVAL_S:.0f} second to a year")
        start = self._clock() + interval_s if start_at is None else self._epoch(start_at, "start_at")
        return self._add("every", title, {"interval_s": float(interval_s), "start_at": start}, start, key,
                         DEFAULT_GRACE_S, max_retries, payload)

    def add_daily(self, title, hhmm, *, tz="America/Chicago", key=None, max_retries=DEFAULT_RETRIES, payload=None):
        if not isinstance(hhmm, str) or not _HHMM.match(hhmm):
            raise SchedulerError("hhmm must look like 08:30")
        try:
            ZoneInfo(tz)
        except Exception:
            raise SchedulerError("unknown timezone") from None
        first = _local_slot(hhmm, tz, self._clock())
        return self._add("daily", title, {"hhmm": hhmm, "tz": tz}, first, key, DEFAULT_GRACE_S, max_retries, payload)

    def cancel(self, task_id) -> bool:
        t = self._tasks.get(task_id)
        if t is None or t.status in TERMINAL or t.cancel_requested:
            return False
        self._append({"op": "cancel", "id": task_id, "at": self._clock()})
        return True

    # -- claiming and finishing ------------------------------------------------

    def due(self, now=None):
        """Claim every eligible task. Each returned Fire is already durable; the caller must call complete or fail.
        A one-time task found too late is returned once with `missed=True` (it is not claimed and never runs). One task
        failing to be claimed never costs the others their claim: whatever was claimed is always returned."""
        now = self._clock() if now is None else now
        fires = []
        for t in sorted(self._tasks.values(), key=lambda x: (x.next, x.seq)):
            if t.status != "scheduled" or t.cancel_requested:
                continue
            try:
                got = self._claim(t, now)
            except SchedulerError as e:
                self.errors.append(str(e))
                if self._dead:
                    break
                continue
            except Exception as e:                    # one bad task never starves the rest
                self.errors.append(f"{t.id}: {e.__class__.__name__}")
                continue
            if got is not None:
                fires.append(got)
        del self.errors[:-50]
        return fires

    def _claim(self, t, now):
        retry = t.retry_at is not None
        # a recurring task whose next slot is absurdly far ahead (the system clock was set back): re-anchor it
        if t.kind != "once" and not retry:
            bound = t.spec["interval_s"] if t.kind == "every" else 2 * 86400.0
            if t.next - now > bound:
                if t.kind == "every":                 # rebase the grid on the corrected clock
                    nxt = now + t.spec["interval_s"]
                    self._append({"op": "reanchor", "id": t.id, "at": now, "next": nxt, "start_at": nxt})
                else:
                    self._append({"op": "reanchor", "id": t.id, "at": now, "next": _next_after(t.spec, t.kind, now)})
        at = t.retry_at if retry else t.next
        if at > now:
            return None
        late = now - at
        if t.kind == "once" and not retry and late > t.grace_s:
            self._append({"op": "missed", "id": t.id, "at": now, "late_s": late})
            return Fire(t.id, f"{t.id}:missed", t.title, dict(t.payload), t.kind, at, late, 0, 0,
                        f"{t.id}@{at}", at, True)
        coalesced, nxt = 0, None
        if t.kind != "once":
            if retry:
                nxt = t.next                          # a retry of an earlier slot: `next` was already advanced
            else:
                nxt = _next_after(t.spec, t.kind, now)
                first = t.next
                if t.kind == "every":                 # arithmetic: an outage of any length stays O(1)
                    coalesced = max(0, int(round((nxt - first) / t.spec["interval_s"])) - 1)
                else:                                 # daily: one step per skipped day
                    probe = first
                    while True:
                        probe = _next_after(t.spec, t.kind, probe)
                        if probe >= nxt:
                            break
                        coalesced += 1
        attempt = (t.attempts + 1) if retry else 1
        slot = t.slot if retry and t.slot is not None else at
        run_id = f"{t.id}:{t.runs + 1}"
        self._append({"op": "fire", "id": t.id, "at": now, "run_id": run_id, "attempt": attempt,
                      "scheduled_for": at, "slot": slot, "late_s": late, "coalesced": coalesced, "next": nxt})
        return Fire(t.id, run_id, t.title, dict(t.payload), t.kind, at, late, coalesced, attempt, f"{t.id}@{slot}", slot)

    def _task_for_run(self, run_id):
        for t in self._tasks.values():
            if t.run_id == run_id:
                return t
        return None

    def complete(self, run_id) -> bool:
        t = self._task_for_run(run_id)
        if t is None or t.status != "running":
            return False
        self._append({"op": "done", "id": t.id, "at": self._clock(), "run_id": run_id})
        return True

    def fail(self, run_id, error="failed") -> bool:
        t = self._task_for_run(run_id)
        if t is None or t.status != "running":
            return False
        error = " ".join(str(error).split())[:200] or "failed"
        retry_at = None
        if not t.cancel_requested and t.attempts <= t.max_retries:
            retry_at = self._clock() + _backoff(t.attempts)
        self._append({"op": "fail", "id": t.id, "at": self._clock(), "run_id": run_id, "error": error,
                      "retry_at": retry_at})
        return True

    def recover(self):
        """At startup: every run left `running` had an unknown outcome. Record each as one failed attempt. Returns ids."""
        out = []
        for t in list(self._tasks.values()):
            if t.status == "running":
                if self.fail(t.run_id, "interrupted by a restart; outcome unknown"):
                    out.append(t.id)
        return out

    # -- reading ---------------------------------------------------------------

    def get(self, task_id):
        return self._tasks.get(task_id)

    def tasks(self, status=None):
        return [t for t in self._tasks.values() if status is None or t.status == status]

    def next_due_at(self):
        """The earliest time something becomes eligible, or None. For sleeping until the next tick."""
        ats = [t.retry_at if t.retry_at is not None else t.next for t in self._tasks.values()
               if t.status == "scheduled" and not t.cancel_requested]
        return min(ats) if ats else None
