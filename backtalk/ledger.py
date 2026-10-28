"""The Action Ledger: a durable, append-only record of turns, tool actions and
background tasks, and of the epochs (client sessions) they lived in.

WHY THIS EXISTS. Before this, nothing in Backtalk recorded what an action did
after the SDK let it through. Tool results never reached code, and a
background task that died with its session left no trace. The ledger is the
one place that says what actually happened, and the only source the speech
policy may build a spoken line about an action from.

EVENT SOURCED. Every change is one JSON line appended to ledger.jsonl and
fsync'd; the in-memory state is a fold over those lines, so replaying the file
after a crash gives back exactly what was known. Nothing is ever rewritten.

WHAT IT DELIBERATELY DOES NOT DO.
  * There is no EXECUTED state. The SDK only tells us an action was
    requested (tool_use) and returned (tool result). Whether the
    effect happened is a separate question, answered by verify.py.
  * A tool's own success message never becomes VERIFIED (see set_verification).
  * Message bodies are never stored: only sizes, hashes and short redacted
    previews.

CRASH DETECTION. Each epoch writes OPEN, then CLOSING (with a reason), then
CLOSED_CLEAN after a successful disconnect. An epoch that is still OPEN or
CLOSING when the next process starts is a crash, and reconcile() says so.
"""
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid as _uuid
from pathlib import Path

# ---- states ---------------------------------------------------------------

EPOCH_OPEN, EPOCH_CLOSING = "OPEN", "CLOSING"
EPOCH_CLOSED_CLEAN, EPOCH_CRASHED = "CLOSED_CLEAN", "CRASHED"

# turns (foreground queries)
T_CREATED, T_SENT, T_CLAIMED, T_RUNNING = "CREATED", "SENT", "CLAIMED", "RUNNING"
T_RESULT, T_DONE = "RESULT", "DONE"
T_FAILED, T_INTERRUPTED, T_ABORTED = "FAILED", "INTERRUPTED", "ABORTED"
TURN_TERMINAL = {T_DONE, T_FAILED, T_INTERRUPTED, T_ABORTED}

# actions (one tool call). There is no EXECUTED: see the module docstring.
# ALLOWED and DENIED are no longer written by anything; they stay so an older
# ledger.jsonl still replays.
A_REQUESTED, A_ALLOWED, A_DENIED = "REQUESTED", "ALLOWED", "DENIED"
A_OK, A_ERROR = "RETURNED_OK", "RETURNED_ERROR"
A_LOST, A_ABORTED = "RESULT_LOST", "ABORTED"
ACTION_TERMINAL = {A_DENIED, A_OK, A_ERROR, A_LOST, A_ABORTED}

# background tasks
K_LAUNCHED, K_RUNNING = "LAUNCHED", "RUNNING"
K_COMPLETED, K_FAILED = "COMPLETED", "FAILED"
K_STOPPED_USER = "STOPPED_BY_USER"
K_KILLED_INTERRUPT = "KILLED_BY_INTERRUPT"
K_STOPPED_UNATTRIBUTED = "STOPPED_UNATTRIBUTED"   # stopped, and we recorded no intent
K_LOST_REBUILD = "LOST_ON_REBUILD"
K_UNKNOWN_CRASH = "UNKNOWN_AFTER_CRASH"
K_ORPHAN_RUNNING = "ORPHAN_RUNNING"
K_DEAD_UNKNOWN = "DEAD_OUTCOME_UNKNOWN"
TASK_TERMINAL = {K_COMPLETED, K_FAILED, K_STOPPED_USER, K_KILLED_INTERRUPT,
                 K_STOPPED_UNATTRIBUTED, K_LOST_REBUILD, K_UNKNOWN_CRASH,
                 K_ORPHAN_RUNNING, K_DEAD_UNKNOWN}
TASK_LIVE = {K_LAUNCHED, K_RUNNING}

# task kinds
KIND_BG_AGENT = "BG_AGENT"                # Agent(run_in_background=true)
KIND_SUBAGENT = "SUBAGENT_INTERNAL"       # owned by a subagent
KIND_FG_TOOL = "FG_TOOL_TASK"             # a foreground tool promoted to a task

# verification
V_NOT_REQUIRED, V_PENDING = "NOT_REQUIRED", "PENDING"
V_VERIFIED, V_UNVERIFIED, V_FAILED = "VERIFIED", "UNVERIFIED", "VERIFY_FAILED"
LEVEL_NONE, LEVEL_SELF, LEVEL_READBACK = "NONE", "SELF_REPORT", "READ_BACK"
# A claim that matches the raw tool result the router captured. It proves the
# claim is faithful to the evidence, NOT that the tool did the right thing.
LEVEL_TOOL_EVIDENCE = "TOOL_EVIDENCE"

_TURN_NEXT = {
    T_CREATED: {T_SENT, T_FAILED, T_ABORTED},
    T_SENT: {T_CLAIMED, T_FAILED, T_ABORTED},
    T_CLAIMED: {T_RUNNING, T_RESULT, T_FAILED, T_INTERRUPTED, T_ABORTED},
    T_RUNNING: {T_RESULT, T_FAILED, T_INTERRUPTED, T_ABORTED},
    T_RESULT: {T_DONE, T_FAILED},
}
_ACTION_NEXT = {
    A_REQUESTED: {A_ALLOWED, A_DENIED, A_OK, A_ERROR, A_LOST, A_ABORTED},
    A_ALLOWED: {A_OK, A_ERROR, A_LOST, A_ABORTED},
}
_TASK_NEXT = {s: TASK_TERMINAL | {K_RUNNING} for s in (K_LAUNCHED, K_RUNNING)}
_TASK_NEXT[K_LAUNCHED] = TASK_TERMINAL | {K_RUNNING}
_TASK_NEXT[K_RUNNING] = TASK_TERMINAL
# a crash verdict may later be refined by independent process evidence, and
# only by that
_TASK_NEXT[K_UNKNOWN_CRASH] = {K_ORPHAN_RUNNING, K_DEAD_UNKNOWN}
_EPOCH_NEXT = {
    EPOCH_OPEN: {EPOCH_CLOSING, EPOCH_CLOSED_CLEAN, EPOCH_CRASHED},
    EPOCH_CLOSING: {EPOCH_CLOSED_CLEAN, EPOCH_CRASHED},
}
_KIND_TABLES = {"turn": _TURN_NEXT, "action": _ACTION_NEXT,
                "task": _TASK_NEXT, "epoch": _EPOCH_NEXT}
_INITIAL = {"turn": T_CREATED, "action": A_REQUESTED, "task": K_LAUNCHED,
            "epoch": EPOCH_OPEN}


_MAX_DELIVERIES = 20          # entries kept per row


def default_dir() -> Path:
    env = os.environ.get("BACKTALK_LEDGER_DIR")
    if env:
        return Path(env)
    from backtalk.config import CFG, REPO
    return Path(os.path.expanduser(
        (CFG.get("ledger") or {}).get("dir") or (REPO / "ledger")))


def redact(text, limit: int = 200) -> str:
    """A short, single-line, secret-scrubbed preview. Never raises."""
    try:
        s = str(text or "")[:2000]
        try:
            from backtalk.shellscan import _PREVIEW_REDACTIONS
            for pat, rep in _PREVIEW_REDACTIONS:
                s = pat.sub(rep, s)
        except Exception:
            pass
        s = " ".join(s.split())
        return s if len(s) <= limit else s[:limit] + "..."
    except Exception:
        return ""


def digest(content) -> dict:
    """Size + hash of a tool result. The body itself is never stored."""
    try:
        blob = content if isinstance(content, str) else json.dumps(
            content, sort_keys=True, default=str)
        raw = blob.encode("utf-8", "replace")
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()[:16]}
    except Exception:
        return {"bytes": -1, "sha256": ""}


def process_alive(pid: int, lstart: str) -> bool | None:
    """True/False if the exact process (pid AND start time) is/is not there;
    None when it cannot be checked. Read-only: it never touches a process."""
    try:
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "lstart="],
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    got = " ".join(out.stdout.split())
    if not got:
        return False
    return got == " ".join(str(lstart).split())


class Ledger:
    def __init__(self, directory=None, *, fsync: bool = True, clock=time.time):
        self.dir = Path(directory) if directory else default_dir()
        self.path = self.dir / "ledger.jsonl"
        self._fsync = fsync
        self._clock = clock
        self._lock = threading.RLock()
        self._seq = 0
        self.epochs: dict = {}
        self.turns: dict = {}
        self.actions: dict = {}          # keyed by action_id
        self._by_tool_use: dict = {}     # tool_use_id -> action_id
        self.tasks: dict = {}
        self.notes: list = []
        self.corrupt_lines = 0
        self._dirty_tail = False
        self.write_failures = 0          # disk errors swallowed by _append (audit F-12); memory stays truthful, the file may not
        self.last_write_error = None
        self.current_epoch: str | None = None
        self._fh = None
        self.dir.mkdir(parents=True, exist_ok=True)
        self._replay()

    # ---- persistence -------------------------------------------------------

    def _replay(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.corrupt_lines += 1     # a torn last line after a crash
                    continue
                self._seq = max(self._seq, int(rec.get("seq") or 0))
                self._apply(rec)

    def _append(self, rec: dict):
        """Write the record, then fold it into memory. A DISK error (full, permission, gone) no longer raises into the router and no longer
        leaves memory behind the truth (audit F-12): the record is still applied, the failure is counted in `write_failures` and named in
        `last_write_error`, and the next append reopens the file. The cost is that a crash then loses what was never persisted."""
        with self._lock:
            self._seq += 1
            rec["seq"] = self._seq
            rec["ts"] = round(self._clock(), 3)
            try:
                if self._fh is None:
                    self._fh = open(self.path, "a", encoding="utf-8")
                try:
                    line = json.dumps(rec, default=str, sort_keys=True)
                except (TypeError, ValueError):
                    line = json.dumps({k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                                       for k, v in rec.items()}, sort_keys=True)
                if self._dirty_tail:
                    line = "\n" + line              # a failed write may have left half a line: never glue the next record onto it
                    self._dirty_tail = False
                self._fh.write(line + "\n")
                self._fh.flush()
                if self._fsync:
                    os.fsync(self._fh.fileno())
            except (OSError, ValueError) as e:
                self.write_failures += 1
                self.last_write_error = e.__class__.__name__
                self._dirty_tail = True
                print(f"[ledger] WRITE FAILED ({self.last_write_error}); {self.write_failures} so far", flush=True)
                try:
                    if self._fh is not None:
                        self._fh.close()
                except OSError:
                    pass
                self._fh = None
            self._apply(rec)

    def close(self):
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    # ---- the fold ----------------------------------------------------------

    def _table(self, kind):
        return {"epoch": self.epochs, "turn": self.turns,
                "action": self.actions, "task": self.tasks}[kind]

    def _apply(self, rec: dict):
        kind = rec.get("t")
        if kind == "note":
            self.notes.append(rec)
            if len(self.notes) > 2000:
                del self.notes[:500]
            return
        if kind == "index":                      # tool_use_id -> action_id
            self._by_tool_use[rec["tool_use_id"]] = rec["id"]
            return
        table = self._table(kind) if kind in _KIND_TABLES else None
        if table is None:
            return
        row = table.setdefault(rec["id"], {"id": rec["id"], "history": []})
        for k, v in rec.items():
            if k in ("t", "id", "seq", "ts", "to"):
                continue
            row[k] = v
        if "to" in rec:
            row["state"] = rec["to"]
            row["history"].append([rec["ts"], rec["to"], rec.get("cause", "")])
        row.setdefault("state", _INITIAL[kind])
        if kind == "epoch" and rec.get("to") == EPOCH_OPEN:
            self.current_epoch = rec["id"]

    def _transition(self, kind, ident, to, **fields) -> bool:
        """Apply a state change if the table allows it. A refused change is
        itself recorded, never silently dropped."""
        with self._lock:
            row = self._table(kind).get(ident)
            frm = row.get("state") if row else None
            if frm is not None:
                if frm == to:
                    if fields:
                        self._append({"t": kind, "id": ident, **fields})
                    return True
                if to not in _KIND_TABLES[kind].get(frm, set()):
                    self._append({"t": "note", "note": "refused_transition",
                                  "kind": kind, "id": ident, "from": frm, "to": to})
                    return False
            self._append({"t": kind, "id": ident, "to": to, **fields})
            return True

    def note(self, note: str, **fields):
        self._append({"t": "note", "note": note, **fields})

    # ---- epochs ------------------------------------------------------------

    def epoch_open(self, session_id=None, cli_pid=None, reason="start") -> str:
        eid = "E" + _uuid.uuid4().hex[:8]
        self._append({"t": "epoch", "id": eid, "to": EPOCH_OPEN,
                      "session_id": session_id, "cli_pid": cli_pid,
                      "cause": reason, "host_pid": os.getpid()})
        return eid

    def epoch_set(self, epoch, **fields):
        self._append({"t": "epoch", "id": epoch, **fields})

    def epoch_closing(self, epoch, reason):
        return self._transition("epoch", epoch, EPOCH_CLOSING, cause=reason,
                                close_reason=reason)

    def epoch_closed_clean(self, epoch):
        return self._transition("epoch", epoch, EPOCH_CLOSED_CLEAN)

    # ---- turns -------------------------------------------------------------

    def turn_new(self, epoch, kind, uuid, intent="", silent=False) -> str:
        tid = "T" + _uuid.uuid4().hex[:8]
        self._append({"t": "turn", "id": tid, "to": T_CREATED, "epoch": epoch,
                      "kind": kind, "uuid": uuid, "intent": redact(intent),
                      "silent": silent})
        return tid

    def turn_state(self, tid, state, **fields) -> bool:
        return self._transition("turn", tid, state, **fields)

    # ---- actions -----------------------------------------------------------

    def action_seen(self, tool_use_id, epoch, tool, *, turn_id=None,
                    parent_tool_use_id=None, intent="",
                    run_in_background=None) -> str:
        """Idempotent: the hook and the stream both reference the same
        tool_use_id, and whichever comes first creates the row."""
        with self._lock:
            aid = self._by_tool_use.get(tool_use_id) if tool_use_id else None
            if aid is None:
                aid = "A" + _uuid.uuid4().hex[:8]
                if tool_use_id:
                    self._append({"t": "index", "tool_use_id": tool_use_id,
                                  "id": aid})
                self._append({"t": "action", "id": aid, "to": A_REQUESTED,
                              "epoch": epoch, "tool": tool,
                              "tool_use_id": tool_use_id, "turn_id": turn_id,
                              "parent_tool_use_id": parent_tool_use_id,
                              "intent": redact(intent),
                              "run_in_background": run_in_background,
                              "verification": {"state": V_PENDING,
                                               "level": LEVEL_NONE,
                                               "evidence": "", "verifier": ""},
                              "spoken": None, "user_visible": None})
            else:
                upd = {}
                row = self.actions[aid]
                if turn_id and not row.get("turn_id"):
                    upd["turn_id"] = turn_id
                # The hook can create the row BEFORE the stream's tool_use is
                # seen (the two are not ordered). The row then has no parent and
                # no intent; without this a subagent's call is orphaned and
                # child_actions() cannot find it.
                if parent_tool_use_id and not row.get("parent_tool_use_id"):
                    upd["parent_tool_use_id"] = parent_tool_use_id
                if intent and not row.get("intent"):
                    upd["intent"] = redact(intent)
                if run_in_background is not None and row.get("run_in_background") is None:
                    upd["run_in_background"] = run_in_background
                if upd:
                    self._append({"t": "action", "id": aid, **upd})
            return aid

    def action_id_for(self, tool_use_id):
        return self._by_tool_use.get(tool_use_id)

    def action_result(self, tool_use_id, *, is_error, content=None, evidence=None):
        """`evidence` is the bounded structured fact from evidence.py (counts
        and a short redacted sample, never the body)."""
        aid = self._by_tool_use.get(tool_use_id)
        if aid is None:
            return False
        row = self.actions[aid]
        if row.get("state") == A_DENIED:
            self._append({"t": "action", "id": aid, "returned_at": round(self._clock(), 3),
                          "result": digest(content)})
            return True
        to = A_ERROR if is_error else A_OK
        extra = {"evidence": evidence} if evidence else {}
        return self._transition("action", aid, to, is_error=bool(is_error),
                                returned_at=round(self._clock(), 3),
                                result=digest(content), **extra)

    def child_actions(self, tool_use_id, depth=3):
        """Actions made by the agent that `tool_use_id` launched, found through
        parent_tool_use_id, and by agents those launched, to `depth` levels."""
        out, frontier, seen = [], {tool_use_id}, set()
        for _ in range(depth):
            nxt = set()
            for a in list(self.actions.values()):
                p = a.get("parent_tool_use_id")
                if p in frontier and a["id"] not in seen:
                    seen.add(a["id"])
                    out.append(a)
                    if a.get("tool_use_id"):
                        nxt.add(a["tool_use_id"])
            if not nxt:
                break
            frontier = nxt
        return out

    def action_set(self, aid, **fields):
        """Fields on an action row that are not a state change (the action's twin of task_set)."""
        if aid in self.actions:
            self._append({"t": "action", "id": aid, **fields})

    def action_state(self, aid, state, **fields) -> bool:
        return self._transition("action", aid, state, **fields)

    def set_verification(self, aid, state, level, evidence="", verifier=""):
        """The only door to VERIFIED, and it only opens for READ_BACK. A tool
        reporting its own success is UNVERIFIED with that as the evidence."""
        if state == V_VERIFIED and level != LEVEL_READBACK:
            state = V_UNVERIFIED
            evidence = ("tool self-report only; not independently confirmed. "
                        + str(evidence))[:300]
        if aid not in self.actions:
            return False
        astate = self.actions[aid].get("state")
        if state == V_VERIFIED and astate != A_OK:
            # audit F-12: nothing is VERIFIED unless the action itself returned OK; an errored, denied, lost or still-pending action
            # keeps whatever the caller found, but never the word "verified"
            state = V_UNVERIFIED
            evidence = (f"the action's state is {astate}, not {A_OK}, so it cannot be verified. " + str(evidence))[:300]
        self._append({"t": "action", "id": aid, "verification": {
            "state": state, "level": level,
            "evidence": redact(evidence, 300), "verifier": verifier,
            "verified_at": round(self._clock(), 3)}})
        return True

    def set_task_verification(self, task_id, state, level, evidence="", verifier="", **extra):
        """What a task's REPORTED result was checked against. VERIFIED opens
        only for READ_BACK or TOOL_EVIDENCE: a model's own summary can never
        verify itself."""
        if state == V_VERIFIED and level not in (LEVEL_READBACK, LEVEL_TOOL_EVIDENCE):
            state = V_UNVERIFIED
            evidence = ("no independent evidence behind this. " + str(evidence))[:300]
        if task_id not in self.tasks:
            return False
        self._append({"t": "task", "id": task_id, "result_verification": {
            "state": state, "level": level, "evidence": redact(evidence, 300),
            "verifier": verifier, "verified_at": round(self._clock(), 3), **extra}})
        return True

    def now(self) -> float:
        """The ledger's wall clock: the one every record's `ts` comes from."""
        return round(self._clock(), 3)

    @staticmethod
    def delivery_fingerprint(kind, row) -> str:
        """What a hand-off was ABOUT: the outcome state and the verification state right now. Not an
        instruction to say anything, and never evidence that anything worked."""
        if kind == "task":
            v = (row.get("result_verification") or {}).get("state")
        elif kind == "action":
            v = (row.get("verification") or {}).get("state")
        else:
            v = None
        return f"{row.get('state')}|{v or '-'}"

    def has_delivery(self, kind, ident, *, outcome=None, fingerprint=None) -> bool:
        """Was this row's outcome handed off? By outcome alone, or by the exact outcome+verification
        fingerprint (default: its fingerprint now)."""
        row = self._table(kind).get(ident)
        if row is None:
            return False
        if outcome is not None:
            return any(d.get("outcome") == outcome for d in row.get("deliveries") or [])
        fp = fingerprint or self.delivery_fingerprint(kind, row)
        return any(f"{d.get('outcome')}|{d.get('verification') or '-'}" == fp
                   for d in row.get("deliveries") or [])

    def mark_spoken(self, kind, ident, line_id, surface="spoken"):
        """Record that an outcome was HANDED OFF: a line queued to the mouth ("spoken") or a note taken for
        the model's next turn ("model_note"). It means handed off, not heard, and it says nothing about
        whether the action worked. `spoken` / `user_visible` keep their meaning; `deliveries` adds one durable
        entry per (surface, outcome, verification), so marking the same hand-off again changes nothing."""
        if ident not in self._table(kind):
            return False
        with self._lock:
            row = self._table(kind)[ident]
            fp = self.delivery_fingerprint(kind, row)
            deliveries = list(row.get("deliveries") or [])
            if any(d.get("kind") == surface and f"{d.get('outcome')}|{d.get('verification') or '-'}" == fp
                   for d in deliveries):
                return True
            at = round(self._clock(), 3)
            outcome, _, verification = fp.partition("|")
            deliveries.append({"at": at, "kind": surface, "line_id": line_id, "epoch": self.current_epoch,
                               "outcome": row.get("state"),
                               "verification": None if verification == "-" else verification})
            self._append({"t": kind, "id": ident,
                          "spoken": {"at": at, "line_id": line_id} if surface == "spoken" else
                          row.get("spoken"),
                          "user_visible": {"at": at, "surface": surface},
                          "deliveries": deliveries[-_MAX_DELIVERIES:]})
        return True

    # ---- tasks -------------------------------------------------------------

    def task_seen(self, task_id, epoch, *, kind, tool_use_id=None,
                  description="", parent_task_id=None, turn_id=None,
                  announce=False, protected=False) -> bool:
        with self._lock:
            if task_id in self.tasks:
                return False
            self._append({"t": "task", "id": task_id, "to": K_LAUNCHED,
                          "epoch": epoch, "task_kind": kind,
                          "tool_use_id": tool_use_id,
                          "description": redact(description, 120),
                          "parent_task_id": parent_task_id,
                          "launched_in_turn": turn_id, "announce": announce,
                          "protected": protected, "cancel_intent": None,
                          "process_snapshot": [], "process_evidence": "UNCHECKED",
                          "outcome_known": False, "spoken": None,
                          "user_visible": None})
            return True

    def task_state(self, task_id, state, **fields) -> bool:
        if state in TASK_TERMINAL and "outcome_known" not in fields:
            fields["outcome_known"] = state in (K_COMPLETED, K_FAILED,
                                                K_STOPPED_USER, K_KILLED_INTERRUPT,
                                                K_STOPPED_UNATTRIBUTED)
        return self._transition("task", task_id, state, **fields)

    def task_set(self, task_id, **fields):
        if task_id in self.tasks:
            self._append({"t": "task", "id": task_id, **fields})

    def task_cancel_intent(self, task_id, reason):
        """Written BEFORE the cancelling call: on the wire stop_task and
        interrupt() both arrive as killed/stopped, so the cause has to come
        from what we recorded we were about to do."""
        self.task_set(task_id, cancel_intent=reason)

    def live_tasks(self, kind=None, epoch=None):
        return [t for t in self.tasks.values()
                if t.get("state") in TASK_LIVE
                and (kind is None or t.get("task_kind") == kind)
                and (epoch is None or t.get("epoch") == epoch)]

    # ---- rebuild / crash ---------------------------------------------------

    def close_epoch_tasks(self, epoch, state, reason, **fields):
        """Every live task of an epoch becomes `state`. Used on a clean
        rebuild (LOST_ON_REBUILD) once the disconnect has returned."""
        n = 0
        for t in list(self.live_tasks(epoch=epoch)):
            if self.task_state(t["id"], state, cause=reason, terminal_at=round(self._clock(), 3), **fields):
                n += 1
        return n

    def lose_epoch_actions(self, epoch, cause) -> int:
        """An epoch is closing: its actions that never got a result (REQUESTED, ALLOWED) become RESULT_LOST,
        the same state the next start's reconcile gives them, so nothing of a closed epoch stays in flight.
        Verification is settled too, because no verifier can run for them any more: a lost action is
        UNVERIFIED (never VERIFIED), a denied one that never saw its result is NOT_REQUIRED. Only this epoch's
        rows are touched, only rows still waiting are changed, and a second call changes nothing. Returns the
        number of actions newly lost."""
        n = 0
        with self._lock:
            for a in list(self.actions.values()):
                if a.get("epoch") != epoch:
                    continue
                if a.get("state") in (A_REQUESTED, A_ALLOWED):
                    if self.action_state(a["id"], A_LOST, cause=cause):
                        n += 1
                if (a.get("verification") or {}).get("state") != V_PENDING:
                    continue
                if a.get("state") == A_LOST:
                    self.set_verification(a["id"], V_UNVERIFIED, LEVEL_NONE,
                                          "the result was lost when its epoch closed; nothing to check", "epoch-close")
                elif a.get("state") == A_DENIED:
                    self.set_verification(a["id"], V_NOT_REQUIRED, LEVEL_NONE,
                                          "denied; nothing ran", "epoch-close")
        return n

    def reconcile(self, proc_check=None) -> dict:
        """Run ONCE at startup, before opening the new epoch.

        An epoch still OPEN or CLOSING is a crash: nothing wrote CLOSED_CLEAN.
        Its live tasks are UNKNOWN_AFTER_CRASH, never STOPPED: after a hard
        kill the session may report a task stopped while its process is still
        running. `proc_check(pid, lstart) -> bool|None` may refine that, using
        ONLY process identities Backtalk recorded for its own tasks."""
        out = {"crashed_epochs": 0, "tasks_unknown": 0, "tasks_lost": 0,
               "actions_lost": 0, "turns_aborted": 0, "refined_alive": 0,
               "refined_dead": 0}
        with self._lock:
            for eid, ep in list(self.epochs.items()):
                st = ep.get("state")
                if st in (EPOCH_OPEN, EPOCH_CLOSING):
                    if self._transition("epoch", eid, EPOCH_CRASHED,
                                        cause="no CLOSED_CLEAN marker at startup"):
                        out["crashed_epochs"] += 1
                    crashed = True
                elif st == EPOCH_CLOSED_CLEAN:
                    crashed = False
                else:
                    continue
                for t in list(self.tasks.values()):
                    if t.get("epoch") != eid or t.get("state") not in TASK_LIVE:
                        continue
                    if crashed:
                        self.task_state(t["id"], K_UNKNOWN_CRASH,
                                        cause="epoch crashed",
                                        terminal_at=round(self._clock(), 3))
                        out["tasks_unknown"] += 1
                        self._refine(t["id"], proc_check, out)
                    else:
                        self.task_state(t["id"], K_LOST_REBUILD,
                                        cause="live task in a cleanly closed epoch",
                                        terminal_at=round(self._clock(), 3))
                        out["tasks_lost"] += 1
                for a in list(self.actions.values()):
                    if a.get("epoch") == eid and a.get("state") in (A_REQUESTED, A_ALLOWED):
                        self.action_state(a["id"], A_LOST, cause="epoch ended before a result")
                        out["actions_lost"] += 1
                for tr in list(self.turns.values()):
                    if tr.get("epoch") == eid and tr.get("state") not in TURN_TERMINAL:
                        self.turn_state(tr["id"], T_ABORTED, cause="epoch ended")
                        out["turns_aborted"] += 1
        return out

    def _refine(self, task_id, proc_check, out):
        snap = self.tasks[task_id].get("process_snapshot") or []
        if not proc_check or not snap:
            return
        verdicts = []
        for p in snap:
            try:
                verdicts.append(proc_check(p.get("pid"), p.get("lstart")))
            except Exception:
                verdicts.append(None)
        if any(v is True for v in verdicts):
            self.task_state(task_id, K_ORPHAN_RUNNING, process_evidence="ALIVE",
                            cause="recorded process still running after crash")
            out["refined_alive"] += 1
        elif verdicts and all(v is False for v in verdicts):
            self.task_state(task_id, K_DEAD_UNKNOWN, process_evidence="DEAD",
                            cause="recorded process gone; outcome still unknown")
            out["refined_dead"] += 1
