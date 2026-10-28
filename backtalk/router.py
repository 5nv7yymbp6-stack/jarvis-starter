"""The message router: one continuous reader for the Claude SDK stream, and the
rules that decide who owns every message on it.

THE BUG THIS REPLACES. Backtalk used to read the stream only while a turn was
open (`receive_response()`, which stops at the first ResultMessage) and pair a
query with "whatever result arrives next". A background task finishing while
nobody was reading left an unsolicited run sitting in the pipe, and the next
question was answered with THAT run's result, then every later question with
the previous one's answer (the "France/Italy" desync, reproduced in scratch on
2026-09-30). Nothing here pairs by arrival order.

WHAT THE SPIKES ESTABLISHED (scratchpad/spikes, 2026-09-30) AND THIS FILE
RELIES ON, AND ONLY THIS:
  * A user message sent with our own `uuid` (and the CLI flag
    replay-user-messages) is echoed back as a UserMessage carrying that uuid,
    before any of that run's content. That echo, and nothing else, claims a run.
    `init` is NOT a claim (an unsolicited run can start in the same instant).
  * Runs never interleave; a query sent during another run waits its turn.
  * ResultMessage carries no request id. An unsolicited run (a background
    report) ends in a ResultMessage with origin.kind == "task-notification";
    foreground results have no origin.
  * After a rebuild/resume each lost task yields a synthetic TaskNotification
    (status "stopped") and an empty unsolicited run (num_turns 0, result "").
  * interrupt() ends the run with an aborted ResultMessage AND kills every
    background task; stop_task(id) kills one task and leaves the run alone.
  * A task's end may arrive as TaskUpdated, TaskNotification, or both.
  * Slash commands answer BEFORE their echo, so a command handle may claim a
    run retroactively (and only a command handle).

Anything not covered by those rules goes to a bounded quarantine. It is never
handed to a turn and never reaches the mouth.
"""
import asyncio
import collections
import hashlib
import re
import subprocess
import time
import uuid as _uuid
from dataclasses import dataclass, field

from backtalk import evidence as E
from backtalk import ledger as L
from backtalk import taskstate as TS
from backtalk import verify as V
from backtalk.vlog import log

# ---- small helpers ----------------------------------------------------------

_TASK_ID_RE = re.compile(r"<task-id>\s*([^<\s]+)\s*</task-id>")
TERMINAL_WIRE = {"completed", "failed", "stopped", "killed"}


class RouterClosed(RuntimeError):
    """The epoch this router belongs to is closed or lost."""


class TurnFailed(RuntimeError):
    """A foreground turn ended in FAILED. .reason is a short code."""

    def __init__(self, reason, turn_id=None):
        super().__init__(reason)
        self.reason = reason
        self.turn_id = turn_id             # lets a handler look the turn up in the ledger (Phase C recovery decision)


def _origin_kind(m):
    o = getattr(m, "origin", None)
    return o.get("kind") if isinstance(o, dict) else None


def _brief(m):
    """Type and ids only. Message content is never put in a quarantine
    record or a log line."""
    d = {"type": type(m).__name__}
    for k in ("subtype", "task_id", "tool_use_id", "parent_tool_use_id", "uuid"):
        v = getattr(m, k, None)
        if v:
            d[k] = str(v)[:40]
    return d


def _text_of(m):
    out = []
    for b in getattr(m, "content", None) or []:
        t = getattr(b, "text", None)
        if t:
            out.append(t)
    return " ".join(out)


# ---- quarantine -------------------------------------------------------------

class Quarantine:
    """Bounded holding pen for anything the rules do not own."""

    def __init__(self, maxlen: int = 200):
        self.ring = collections.deque(maxlen=maxlen)
        self.counts = collections.Counter()

    def add(self, reason, msg=None, epoch=None):
        self.counts[reason] += 1
        rec = {"t": time.time(), "reason": reason, "epoch": epoch}
        if msg is not None:
            rec.update(_brief(msg))
        self.ring.append(rec)
        log(f"[router] quarantine reason={reason} epoch={epoch} "
            f"{ {k: v for k, v in rec.items() if k in ('type', 'subtype', 'task_id')} }")


# ---- foreground turn handle -------------------------------------------------

class TurnHandle:
    """One foreground query. ask_stream() consumes .events(); nothing else
    ever feeds it, and only the run its uuid claimed can."""

    def __init__(self, router, kind, uuid, turn_id, text_intent, silent=False):
        self.router = router
        self.epoch = router.epoch
        self.kind = kind                  # foreground | command
        self.uuid = uuid
        self.turn_id = turn_id
        self.intent = text_intent
        self.silent = silent
        self.state = L.T_SENT
        self.q: asyncio.Queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.result = None
        self.fail_reason = None
        self.contaminated_by: list = []
        self.interrupt_requested = False
        self.abandoned = False
        self.sent_at = time.monotonic()
        self.claimed_at = None
        self.last_progress = time.monotonic()

    @property
    def terminal(self):
        return self.state in L.TURN_TERMINAL

    def _push(self, tag, payload):
        self.last_progress = time.monotonic()
        self.q.put_nowait((tag, payload))

    def finish(self, state, reason=None):
        """Move to a terminal state exactly once and wake the consumer."""
        if self.terminal:
            return False
        if state == L.T_DONE and self.state != L.T_RESULT:
            self.router.ledger.turn_state(self.turn_id, L.T_RESULT)
        self.state = state
        self.fail_reason = reason
        self.router.ledger.turn_state(self.turn_id, state,
                                      **({"cause": reason} if reason else {}))
        self.q.put_nowait(("end", state))
        self.done.set()
        return True

    async def events(self):
        """Yield ('msg'|'result', message) until the turn ends. Raises
        TurnFailed for FAILED; returns quietly for DONE / INTERRUPTED /
        ABORTED (the caller that cancelled already knows)."""
        while True:
            tag, payload = await self.q.get()
            if tag == "end":
                if payload == L.T_FAILED:
                    raise TurnFailed(self.fail_reason or "failed", self.turn_id)
                return
            yield tag, payload


# ---- runs -------------------------------------------------------------------

UNSOLICITED, STALE = "UNSOLICITED", "STALE"

# Stale-echo interrupts are rate limited. An abandoned query's run still starts
# when its echo arrives, and _interrupt_stale() asks the CLI to stop it. When
# several abandoned queries are queued, their echoes arrive within milliseconds
# and each one used to send its own interrupt; two interrupt control requests
# a few ms apart killed the CLI (AbortError -> stream_lost) or interrupted the
# live query, reproduced with the real CLI on 2026-10-05 (spacing of 1.2 s was
# safe, 0.6 s was not). So a stale interrupt is NOT sent when another went out
# less than this many seconds ago. It is never delayed or queued: a delayed
# interrupt lands on whatever run is current by then, the live one included.
STALE_INTERRUPT_MIN_GAP_S = 1.2


# ---- passive SDK/CLI observability (2026-10-06) -------------------------------
# Observability only: nothing here changes routing, retries, timing or any decision.
# A first-token wait of 9 to 27 s was traced to the span between the CLI accepting a
# message and the first stream event, and the router used to drop every non-init
# system message and every rate-limit event, so a retry/backoff or a throttle would
# have been invisible. These helpers turn those messages into ONE compact log line
# built only from an allowlist of keys whose values are numbers, booleans or short
# lowercase enum-like words. Prompt text, transcripts, credentials, headers, tool or
# path lists, and free-text error bodies are never read into the line: values of any
# other key are dropped and only the key NAMES are listed, so an unmodelled retry
# field is discoverable without logging its value.
_SDK_VALUE_KEYS = frozenset({
    "status", "attempt", "max_retries", "max_attempts", "retry_attempt", "retry_count",
    "retry_delay_ms", "retry_delay_s", "retry_delay", "retry_after", "retry_after_ms",
    "delay_ms", "backoff_ms", "wait_ms", "error_status", "status_code", "http_status",
    "api_error_status", "error_type", "error_name", "error_code", "retryable",
    "is_retryable", "estimated_tokens", "estimated_tokens_delta", "reason", "category",
    "kind"})
_SDK_RETRY_HINTS = ("retry", "backoff", "attempt", "delay", "overload", "rate", "limit",
                    "status_code", "error_status", "http_status", "wait")
_SDK_WORD = re.compile(r"[a-z][a-z_.:-]{0,39}\Z")          # enum-like only (no digits, no capitals, short); anything else is text
_SDK_NAME = re.compile(r"[A-Za-z0-9_]{1,24}\Z")
_SDK_THINKING_LOG_GAP_S = 1.0       # thinking_tokens arrive in bursts: log about one a second


def _sdk_scalar(v):
    """A loggable form of v, or None: bool/int/float as numbers, a short enum-like
    lowercase word as is; everything else (text, lists, dicts) is not logged."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, str) and _SDK_WORD.match(v) and not v.startswith("sk-"):
        return v
    return None


def describe_sdk_system(m):
    """({"subtype", "session", "fields": {k: v}, "keys": [...], "retry": "present" |
    "none_exposed"}) for a non-init SystemMessage, or None for init. Never raises."""
    try:
        sub = getattr(m, "subtype", None)
        data = getattr(m, "data", None)
        data = data if isinstance(data, dict) else {}
        if sub == "init":
            return None
        fields, names, retry = {}, [], False
        for k, v in data.items():
            if not isinstance(k, str) or not _SDK_NAME.match(k):
                continue
            kl = k.lower()
            if any(h in kl for h in _SDK_RETRY_HINTS):
                retry = True
            if k in _SDK_VALUE_KEYS:
                val = _sdk_scalar(v)
                fields[k] = val if val is not None else "<text>"
            elif k not in ("subtype", "type", "uuid", "session_id"):
                names.append(k)
        sid = data.get("session_id")
        return {"subtype": sub if isinstance(sub, str) and _SDK_NAME.match(sub) else "?",
                "session": sid[:8] if isinstance(sid, str) and re.fullmatch(r"[0-9a-f-]{8,}", sid) else None,
                "fields": fields, "keys": sorted(names)[:14],
                "retry": "present" if retry else "none_exposed"}
    except Exception:
        return None


def describe_sdk_rate_limit(m):
    """The modelled, non-secret fields of a RateLimitEvent as a dict, or None. Never raises."""
    try:
        info = getattr(m, "rate_limit_info", None)
        if info is None:
            return None
        out = {}
        for k in ("status", "rate_limit_type", "utilization", "resets_at", "overage_status",
                  "overage_resets_at", "overage_disabled_reason"):
            v = _sdk_scalar(getattr(info, k, None))
            if v is not None:
                out[k] = v
        raw = getattr(info, "raw", None)
        keys = sorted(k for k in raw if isinstance(k, str) and _SDK_NAME.match(k))[:14] \
            if isinstance(raw, dict) else []
        return {"fields": out, "keys": keys}
    except Exception:
        return None


@dataclass
class Run:
    """One init..Result window. It has exactly one owner: a TurnHandle, or
    UNSOLICITED (a background report), or STALE (the echo of a dead handle).
    owner None means "not known yet": content is held, never routed."""
    owner: object = None
    saw_init: bool = False
    buffer: list = field(default_factory=list)      # AssistantMessages held pre-claim
    report_text: list = field(default_factory=list)
    task_ids: list = field(default_factory=list)    # background tasks whose notice this run contains
    held_stream_events: int = 0
    started_at: float = field(default_factory=time.monotonic)


@dataclass
class PendingReport:
    """A background/unsolicited run's outcome. The text is MODEL-AUTHORED and
    kept only for the record: it must never be spoken (speech_policy builds
    its own line from the ledger)."""
    at: float
    task_ids: list
    text: str
    synthetic: bool
    epoch: str


# ---- process evidence (read-only) -----------------------------------------

def proc_tree(root_pid):
    """(pid, lstart, cmd-hash) for every descendant of root_pid. Read-only."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,lstart=,command="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows[pid] = (ppid, " ".join(parts[2:7]), parts[7])
    kids = collections.defaultdict(list)
    for pid, (ppid, _, _) in rows.items():
        kids[ppid].append(pid)
    found, stack = [], [int(root_pid)]
    while stack:
        for k in kids.get(stack.pop(), []):
            found.append({"pid": k, "lstart": rows[k][1],
                          "cmd": hashlib.sha256(rows[k][2].encode()).hexdigest()[:10]})
            stack.append(k)
    return found


# ---- tasks ------------------------------------------------------------------

class TaskRegistry:
    """Background tasks, keyed by task_id (durable across a resume). tool_use_id
    is only a secondary, session-local key. State lives in the ledger."""

    def __init__(self, ledger, quarantine, *, protect_background=True):
        self.ledger = ledger
        self.quarantine = quarantine
        self.protect_background = protect_background
        self.listeners: list = []

    def _kind_for(self, m, action):
        """(kind, basis). The AUTHORITY is the task's own start message:
        `is_backgrounded` in the task_started payload. Captured against the
        live CLI on 2026-09-30:

            "launch a background agent" (Agent input has NO run_in_background)
                                       -> is_backgrounded true,  local_agent
            explicit foreground Agent  -> is_backgrounded false, local_agent
            foreground Bash task       -> is_backgrounded false, local_bash

        The Agent's own `run_in_background` input is NOT reliable: the CLI ran
        a task in the background with that flag unset, and the router, which
        trusted the flag, recorded it as a foreground tool task (live,
        2026-09-30 21:05). The flag and `task_type` are fallbacks only, for a
        CLI that does not send is_backgrounded."""
        data = getattr(m, "data", None) or {}
        flag = bool(action and action.get("run_in_background"))
        wire = data.get("is_backgrounded")
        if data.get("owned_by_subagent"):
            return L.KIND_SUBAGENT, "owned_by_subagent"
        if wire is True:
            return L.KIND_BG_AGENT, "wire:is_backgrounded"
        if wire is False:
            if flag:
                # the two disagree: protect the work (treat it as background)
                # and leave a record of the disagreement
                self.ledger.note("task_kind_conflict", task_id=getattr(m, "task_id", None),
                                 wire_is_backgrounded=False, input_run_in_background=True)
                return L.KIND_BG_AGENT, "conflict:input_flag_true_wire_false"
            return L.KIND_FG_TOOL, "wire:is_backgrounded=false"
        # no is_backgrounded on the wire (older CLI): fall back
        if flag:
            return L.KIND_BG_AGENT, "fallback:input_run_in_background"
        if action is None and getattr(m, "task_type", None) == "local_agent":
            return L.KIND_BG_AGENT, "fallback:task_type_without_action"
        return L.KIND_FG_TOOL, "fallback:default"

    def on_started(self, m, epoch, turn_id=None):
        tid = getattr(m, "task_id", None)
        if not tid:
            return self.quarantine.add("task_started_without_id", m, epoch)
        if tid in self.ledger.tasks:
            return
        tuid = getattr(m, "tool_use_id", None)
        aid = self.ledger.action_id_for(tuid) if tuid else None
        action = self.ledger.actions.get(aid) if aid else None
        kind, basis = self._kind_for(m, action)
        self.ledger.task_seen(
            tid, epoch, kind=kind, tool_use_id=tuid,
            description=getattr(m, "description", "") or "", turn_id=turn_id,
            protected=bool(self.protect_background and kind == L.KIND_BG_AGENT))
        # the evidence the classification rested on, so it can be audited
        self.ledger.task_set(
            tid, kind_basis=basis, wire_task_type=getattr(m, "task_type", None),
            wire_is_backgrounded=(getattr(m, "data", None) or {}).get("is_backgrounded"))
        self.ledger.task_state(tid, L.K_RUNNING)
        if action is not None and kind == L.KIND_BG_AGENT:
            self.ledger.action_state(aid, action["state"], task_id=tid)

    def on_progress(self, m, epoch):
        tid = getattr(m, "task_id", None)
        row = self.ledger.tasks.get(tid)
        if row is None or row.get("epoch") != epoch:
            return self.quarantine.add("progress_for_unknown_task", m, epoch)
        self.ledger.task_set(tid, last_progress_at=round(time.time(), 3))

    def _map_terminal(self, row, wire):
        if wire == "completed":
            return L.K_COMPLETED
        if wire == "failed":
            return L.K_FAILED
        intent = row.get("cancel_intent")
        if intent in ("USER_STOP", "STOP_ALL"):
            return L.K_STOPPED_USER
        if intent:                                # INTERRUPT, BARGE_IN..., REBUILD
            return L.K_KILLED_INTERRUPT
        return L.K_STOPPED_UNATTRIBUTED

    def _terminal(self, m, epoch, wire, summary=""):
        tid = getattr(m, "task_id", None)
        row = self.ledger.tasks.get(tid)
        if row is None:
            self.quarantine.add("terminal_for_unknown_task", m, epoch)
            return "unknown"
        if row.get("epoch") != epoch:
            # A task from a closed session, reported by a new one. This is the
            # CLI's synthetic "no completion record" notice after a resume. It
            # is evidence, low weight, and it never changes the state we
            # recorded ourselves: after a crash the process may still be running.
            self.ledger.task_set(tid, cli_reported=wire,
                                 cli_reported_at=round(time.time(), 3),
                                 cli_reported_source="synthetic_after_resume")
            self.ledger.note("synthetic_lost_task", task_id=tid, wire=wire,
                             recorded_state=row.get("state"), epoch=epoch)
            return "synthetic"
        if row.get("state") in L.TASK_TERMINAL:
            # The first terminal outcome stands. A later report that CONTRADICTS it is recorded once as
            # evidence (never as a state change); one that agrees is just a duplicate.
            if row.get("state") != self._map_terminal(row, wire):
                seen = list(row.get("conflicting_terminal") or [])
                if wire not in [c.get("wire") for c in seen]:
                    seen.append({"wire": wire, "at": round(time.time(), 3)})
                    self.ledger.task_set(tid, conflicting_terminal=seen[-5:])
                    self.ledger.note("conflicting_terminal_status", task_id=tid,
                                     recorded=row.get("state"), reported=wire, epoch=epoch)
            if summary and not row.get("summary"):
                self.ledger.task_set(tid, summary=L.redact(summary, 160))
                V.check_task_summary(self.ledger, tid, summary)
            return "duplicate"
        new = self._map_terminal(row, wire)
        self.ledger.task_state(tid, new, cause=f"wire:{wire}",
                               terminal_at=round(time.time(), 3),
                               summary=L.redact(summary, 160))
        # BEFORE the listeners run: whatever they build must see the check
        V.check_task_summary(self.ledger, tid, summary)
        if row.get("task_kind") == L.KIND_BG_AGENT:
            for fn in list(self.listeners):
                try:
                    fn(self.ledger.tasks[tid])
                except Exception as e:
                    log(f"[router] task listener failed: {e!r}")
        return "ok"

    def on_updated(self, m, epoch):
        status = getattr(m, "status", None) or (getattr(m, "patch", None) or {}).get("status")
        if status in TERMINAL_WIRE:
            return self._terminal(m, epoch, status)
        if status == "running":
            row = self.ledger.tasks.get(getattr(m, "task_id", None))
            if row and row.get("epoch") == epoch and row.get("state") == L.K_LAUNCHED:
                self.ledger.task_state(row["id"], L.K_RUNNING)
        return "ok"

    def on_notification(self, m, epoch):
        wire = getattr(m, "status", None)
        if wire not in TERMINAL_WIRE:
            return self.quarantine.add("odd_task_notification_status", m, epoch)
        return self._terminal(m, epoch, wire, getattr(m, "summary", "") or "")


# ---- the epoch router -------------------------------------------------------

class EpochRouter:
    """Everything for ONE client session. It is the only caller of that
    client's receive_messages(). A new session gets a new EpochRouter; nothing
    is shared with the old one except the ledger and the quarantine, and every
    key includes the epoch, so an old session's message cannot land on a new
    turn."""

    def __init__(self, client, ledger, *, epoch, quarantine=None, tasks=None,
                 echo_timeout_s=20.0, protect_background=True,
                 barge_escalate_s=20.0, proc=True, cli_pid=None):
        self.client = client
        self.ledger = ledger
        self.epoch = epoch
        self.quarantine = quarantine or Quarantine()
        self.tasks = tasks or TaskRegistry(ledger, self.quarantine,
                                           protect_background=protect_background)
        self.echo_timeout_s = echo_timeout_s
        self.protect_background = protect_background
        self.barge_escalate_s = barge_escalate_s
        self.cli_pid = cli_pid
        self.use_proc = bool(proc and cli_pid)
        self.handles: dict = {}                       # uuid -> TurnHandle
        self.run: Run | None = None
        self.reports = collections.deque(maxlen=100)
        self.stats = collections.Counter()
        self.state = "OPEN"                           # OPEN | CLOSING | CLOSED | LOST
        self.lost_cause = None
        self.last_run_end = time.monotonic()
        self.on_lost = None                           # callable(cause)
        self.action_listeners: list = []              # callable(action_row, tool_input)
        self._inputs = collections.OrderedDict()      # tool_use_id -> input (memory only)
        self._pump_task = None
        self._watch_task = None
        self._escalations: set = set()
        # When a stale-echo interrupt was last actually sent (monotonic). Lives
        # on the epoch's router, so a new epoch always starts with no history.
        self._last_stale_interrupt: float | None = None
        # Observability only (see describe_sdk_system): when the latest query was handed to
        # the SDK, so an SDK event can say how long after the submit it arrived.
        self._last_submit: tuple | None = None
        self._last_thinking_log = 0.0
        self._baseline: set = set()
        self._snapped: set = set()

    # ---- lifecycle ---------------------------------------------------------

    async def start(self):
        self._pump_task = asyncio.create_task(self._pump(), name="stream-pump")
        self._watch_task = asyncio.create_task(self._watchdog(), name="echo-watchdog")

    async def _pump(self):
        try:
            async for m in self.client.receive_messages():
                try:
                    self.dispatch(m)
                except Exception as e:                # a router bug must not stop the reader
                    self.stats["router_error"] += 1
                    log(f"[router] dispatch error {type(e).__name__}: {e}")
                    self.quarantine.add("router_error", m, self.epoch)
            self._lost(EOFError("stream ended"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._lost(e)

    async def _watchdog(self):
        try:
            while True:
                await asyncio.sleep(0.5)
                self.check_handles()
        except asyncio.CancelledError:
            raise

    def check_handles(self, now=None):
        """A SENT handle whose run never starts is FAILED(no_echo). It is never
        completed by 'the next result'."""
        now = now if now is not None else time.monotonic()
        if self.run is not None or self.state != "OPEN":
            return
        for h in list(self.handles.values()):
            if h.state == L.T_SENT and now - max(h.sent_at, self.last_run_end) > self.echo_timeout_s:
                self.quarantine.add("no_echo", None, self.epoch)
                h.finish(L.T_FAILED, "no_echo")

    def _lost(self, cause):
        if self.state in ("CLOSING", "CLOSED", "LOST"):
            return
        self.state = "LOST"
        self.lost_cause = cause
        text = str(cause)
        reason = ("oversize" if "maximum buffer size" in text else "stream_lost")
        log(f"[router] stream lost epoch={self.epoch} reason={reason}: {text[:120]}")
        for h in list(self.handles.values()):
            h.finish(L.T_FAILED, reason)
        for a in list(self.ledger.actions.values()):
            if a.get("epoch") == self.epoch and a.get("state") in (L.A_REQUESTED, L.A_ALLOWED):
                self.ledger.action_state(a["id"], L.A_LOST, cause=reason)
        if self.on_lost:
            try:
                self.on_lost(cause)
            except Exception as e:
                log(f"[router] on_lost failed: {e!r}")

    @property
    def usable(self):
        return self.state == "OPEN"

    async def close(self, reason, disconnect=None, timeout=15.0, proc_check=None):
        """Clean close. Order matters: mark CLOSING, abort handles, disconnect,
        THEN judge the tasks (their processes are verified gone or not), THEN
        write CLOSED_CLEAN. If the disconnect fails, the epoch is a crash and
        the tasks are UNKNOWN_AFTER_CRASH, never LOST."""
        if self.state == "CLOSED":
            return True
        prev, self.state = self.state, "CLOSING"
        self.ledger.epoch_closing(self.epoch, reason)
        for h in list(self.handles.values()):
            h.finish(L.T_ABORTED, f"epoch_closing:{reason}")
        for t in self.ledger.live_tasks(epoch=self.epoch):
            self.ledger.task_cancel_intent(t["id"], "REBUILD")
        if self._watch_task:
            self._watch_task.cancel()
        ok = True
        if disconnect is not None:
            try:
                await asyncio.wait_for(disconnect(), timeout)
            except Exception as e:
                ok = False
                self.ledger.note("disconnect_failed", epoch=self.epoch,
                                 error=type(e).__name__)
        if self._pump_task:
            self._pump_task.cancel()
            # bounded: a reader that will not die must not hang a rebuild or
            # a shutdown
            try:
                await asyncio.wait({self._pump_task}, timeout=3.0)
            except BaseException:
                pass
        # Nothing more can arrive for this epoch: whatever never got a result is lost, not left in flight.
        self.ledger.lose_epoch_actions(self.epoch, f"epoch closed before a result ({reason})")
        check = proc_check or L.process_alive
        for t in list(self.ledger.live_tasks(epoch=self.epoch)):
            snap = t.get("process_snapshot") or []
            verdicts = []
            for p in snap:
                try:
                    verdicts.append(check(p.get("pid"), p.get("lstart")))
                except Exception:
                    verdicts.append(None)
            if not ok:
                self.ledger.task_state(t["id"], L.K_UNKNOWN_CRASH,
                                       cause=f"disconnect failed during {reason}")
            elif any(v is True for v in verdicts):
                self.ledger.task_state(t["id"], L.K_ORPHAN_RUNNING, process_evidence="ALIVE",
                                       cause=f"still running after clean disconnect ({reason})")
            else:
                ev = "DEAD" if verdicts and all(v is False for v in verdicts) else "UNCHECKED"
                self.ledger.task_state(t["id"], L.K_LOST_REBUILD, process_evidence=ev,
                                       cause=reason, terminal_at=round(time.time(), 3))
        if ok:
            self.ledger.epoch_closed_clean(self.epoch)
        else:
            self.ledger._transition("epoch", self.epoch, L.EPOCH_CRASHED,
                                    cause="disconnect failed")
        self.state = "CLOSED"
        return ok

    # ---- sending -----------------------------------------------------------

    async def send(self, text, *, kind="foreground", intent=None, silent=False):
        if self.state != "OPEN":
            raise RouterClosed(f"epoch {self.epoch} is {self.state}")
        u = str(_uuid.uuid4())
        tid = self.ledger.turn_new(self.epoch, kind, u, intent or text, silent)
        h = TurnHandle(self, kind, u, tid, self.ledger.turns[tid]["intent"], silent)
        self.ledger.turn_state(tid, L.T_SENT)
        self.handles[u] = h
        self._trim_handles()

        async def gen():
            yield {"type": "user", "message": {"role": "user", "content": text},
                   "parent_tool_use_id": None, "uuid": u}
        # request-start boundary, immediately before the SDK is handed the query (length only, never the text)
        self._last_submit = (time.monotonic(), u[:8])
        try:
            log(f"[req] submit turn={tid} uuid={u[:8]} epoch={self.epoch} kind={kind} chars={len(text)}")
        except Exception:
            pass                        # observability must never be able to stop a query
        try:
            await self.client.query(gen())
        except Exception as e:
            h.finish(L.T_FAILED, f"send_failed:{type(e).__name__}")
            raise
        h.sent_at = time.monotonic()
        return h

    def _trim_handles(self, keep=200):
        if len(self.handles) <= keep:
            return
        dead = [u for u, h in self.handles.items() if h.terminal]
        for u in dead[:len(self.handles) - keep]:
            del self.handles[u]

    def active_handle(self):
        for h in self.handles.values():
            if h.state in (L.T_CLAIMED, L.T_RUNNING):
                return h
        return None

    # ---- the classifier ----------------------------------------------------

    def dispatch(self, m):
        """Classify one message and route it. Synchronous on purpose: order is
        the arrival order and nothing here awaits."""
        if self.state in ("CLOSED", "LOST"):
            self.quarantine.add("message_after_epoch_end", m, self.epoch)
            return
        name = type(m).__name__
        self.stats[name] += 1
        if name == "StreamEvent":
            return self._on_stream(m)
        if name == "AssistantMessage":
            return self._on_assistant(m)
        if name == "UserMessage":
            return self._on_user(m)
        if name == "ResultMessage":
            return self._on_result(m)
        if name == "TaskStartedMessage":
            return self.tasks.on_started(m, self.epoch, self._owner_turn_id())
        if name == "TaskProgressMessage":
            self.tasks.on_progress(m, self.epoch)
            return self._maybe_snapshot(m)
        if name == "TaskUpdatedMessage":
            return self.tasks.on_updated(m, self.epoch)
        if name == "TaskNotificationMessage":
            return self.tasks.on_notification(m, self.epoch)
        if name == "ConversationResetMessage":
            self.ledger.note("conversation_reset", epoch=self.epoch)
            return
        if name == "SystemMessage":
            if getattr(m, "subtype", "") == "init":
                self._ensure_run().saw_init = True      # a boundary hint, never a claim
            else:
                self._observe_system(m)
            return
        if name == "RateLimitEvent":
            self._observe_rate_limit(m)
            return
        self.quarantine.add("unknown_message_type", m, self.epoch)

    def _ensure_run(self) -> Run:
        if self.run is None:
            self.run = Run()
        return self.run

    # ---- passive observability: log only, never decide ----------------------------

    def _since_submit(self):
        ls = self._last_submit
        return f" since_submit={time.monotonic() - ls[0]:.2f}s submit={ls[1]}" if ls else ""

    def _observe_system(self, m):
        try:
            d = describe_sdk_system(m)
            if d is None:
                return
            now = time.monotonic()
            if d["subtype"] == "thinking_tokens":
                if now - self._last_thinking_log < _SDK_THINKING_LOG_GAP_S:
                    return
                self._last_thinking_log = now
            line = (f"[sdk] system subtype={d['subtype']} epoch={self.epoch}"
                    f"{(' session=' + d['session']) if d['session'] else ''}{self._since_submit()}")
            for k, v in d["fields"].items():
                line += f" {k}={v}"
            line += f" retry_metadata={d['retry']}"
            if d["keys"]:
                line += " other_keys=" + ",".join(d["keys"])
            log(line)
        except Exception:
            pass

    def _observe_rate_limit(self, m):
        try:
            d = describe_sdk_rate_limit(m)
            if d is None:
                return
            line = f"[sdk] rate_limit epoch={self.epoch}{self._since_submit()}"
            for k, v in d["fields"].items():
                line += f" {k}={v}"
            if d["keys"]:
                line += " raw_keys=" + ",".join(d["keys"])
            log(line)
        except Exception:
            pass

    def _log_claim(self, h, merged=False):
        try:
            sa = getattr(h, "sent_at", None)
            log(f"[req] echo turn={h.turn_id} uuid={h.uuid[:8]} epoch={self.epoch}"
                f"{(' since_submit=%.2fs' % (time.monotonic() - sa)) if sa else ''}"
                f"{' merged_into_run' if merged else ''}")
        except Exception:
            pass

    def _owner_turn_id(self):
        r = self.run
        return r.owner.turn_id if r is not None and isinstance(r.owner, TurnHandle) else None

    def _owner_handle(self):
        r = self.run
        return r.owner if r is not None and isinstance(r.owner, TurnHandle) else None

    # -- content ---------------------------------------------------------------

    def _on_stream(self, m):
        if getattr(m, "parent_tool_use_id", None):
            self.stats["subagent_stream_event"] += 1
            return
        run = self._ensure_run()
        if isinstance(run.owner, TurnHandle):
            if run.owner.state == L.T_CLAIMED:
                run.owner.state = L.T_RUNNING
                self.ledger.turn_state(run.owner.turn_id, L.T_RUNNING)
            run.owner._push("msg", m)
        else:
            run.held_stream_events += 1                # unowned: counted, never routed

    def _on_assistant(self, m):
        parent = getattr(m, "parent_tool_use_id", None)
        h = self._owner_handle()
        for b in getattr(m, "content", None) or []:
            if type(b).__name__ == "ToolUseBlock":
                inp = getattr(b, "input", None) or {}
                self.ledger.action_seen(
                    b.id, self.epoch, b.name,
                    turn_id=(h.turn_id if h else None),
                    parent_tool_use_id=parent,
                    intent=(h.intent if h else ""),
                    run_in_background=(inp.get("run_in_background")
                                       if isinstance(inp, dict) else None))
                self._inputs[b.id] = inp
                while len(self._inputs) > 300:
                    self._inputs.popitem(last=False)
        if parent:
            self.stats["subagent_message"] += 1        # belongs to a task, not a turn
            return
        run = self._ensure_run()
        if isinstance(run.owner, TurnHandle):
            if run.owner.state == L.T_CLAIMED:
                run.owner.state = L.T_RUNNING
                self.ledger.turn_state(run.owner.turn_id, L.T_RUNNING)
            run.owner._push("msg", m)
        elif run.owner in (UNSOLICITED, STALE):
            run.report_text.append(_text_of(m)[:2000])
        else:
            run.buffer.append(m)                       # held until the run's owner is known
            del run.buffer[:-50]

    def _on_user(self, m):
        parent = getattr(m, "parent_tool_use_id", None)
        content = getattr(m, "content", None)
        blocks = content if isinstance(content, list) else []
        # tool results: the ledger's business, whoever owns the run
        got_result = False
        for b in blocks:
            if type(b).__name__ == "ToolResultBlock":
                got_result = True
                self._tool_result(b)
        if got_result:
            return
        text = content if isinstance(content, str) else _text_of(m)
        u = getattr(m, "uuid", None)
        if not parent and u and u in self.handles:
            return self._echo(self.handles[u], m)
        if text.lstrip().startswith("<task-notification>") or _origin_kind(m) == "task-notification":
            return self._injected_notification(m, text)
        if text.startswith("[Request interrupted"):
            self.stats["interrupt_marker"] += 1
            return
        if parent:
            self.stats["subagent_user_message"] += 1
            return
        self.quarantine.add("unmatched_user_message", m, self.epoch)

    def _late_tool_result(self, aid, row, is_err):
        """A result for an action that already has its outcome. THE FIRST OUTCOME STANDS: nothing is
        overwritten (state, result, evidence), no verifier or listener runs again, and a result that
        CONTRADICTS the recorded outcome is recorded once as evidence on the row and in a note. Returns
        True if the result was handled here, i.e. it was not the action's first."""
        state = row.get("state")
        if state == L.A_DENIED:
            if (row.get("verification") or {}).get("state") != L.V_PENDING:
                return True                                       # its (error) result was already taken
            return False
        if state not in (L.A_OK, L.A_ERROR, L.A_LOST, L.A_ABORTED):
            return False
        self.stats["late_tool_result"] += 1
        reported = L.A_ERROR if is_err else L.A_OK
        if reported != state and not row.get("conflicting_result"):
            self.ledger.action_set(aid, conflicting_result={"recorded": state, "reported": reported,
                                                            "at": round(time.time(), 3), "epoch": self.epoch})
            self.ledger.note("conflicting_terminal_result", action_id=aid, recorded=state,
                             reported=reported, epoch=self.epoch)
        return True

    def _tool_result(self, b):
        tuid = getattr(b, "tool_use_id", None)
        is_err = bool(getattr(b, "is_error", False))
        content = getattr(b, "content", None)
        early = self.ledger.action_id_for(tuid)
        if early and self._late_tool_result(early, self.ledger.actions[early], is_err):
            return
        # the bounded structured fact, for EVERY tool (read-only ones included):
        # a claim about a result can only be checked against what came back
        ev = None
        row0 = self.ledger.actions.get(self.ledger.action_id_for(tuid) or "")
        if row0 is not None and row0.get("state") != L.A_DENIED:
            try:
                ev = E.extract_evidence(row0.get("tool"), self._inputs.get(tuid), content, is_err)
            except Exception as e:                 # evidence is a bonus: never cost the result
                log(f"[router] evidence extraction failed: {type(e).__name__}")
        self.ledger.action_result(tuid, is_error=is_err, content=content, evidence=ev)
        aid = self.ledger.action_id_for(tuid)
        if aid:
            if is_err:
                # Phase C: classify the failure ONCE, from the transient error text, and keep only the code (the ledger never
                # stores bodies). Recovery decisions are made from this later (taskstate.recover); nothing retries here.
                try:
                    self.ledger.action_set(aid, failure_class=TS.classify_tool_error(
                        self.ledger.actions[aid].get("tool"), content))
                except Exception as e:
                    log(f"[router] failure classification failed: {type(e).__name__}")
            row = self.ledger.actions[aid]
            self._recheck_owning_tasks(row)
            for fn in list(self.action_listeners):
                try:
                    fn(row, self._inputs.get(tuid))
                except Exception as e:
                    log(f"[router] action listener failed: {e!r}")

    def _recheck_owning_tasks(self, action_row):
        """A tool result landed. If it belongs to a task that has ALREADY
        finished and could not be checked for want of this evidence, check it
        again. verify_task_result never touches a recorded VERIFIED/FAILED
        verdict, so this can only move a task out of "could not check"."""
        try:
            p, depth = action_row.get("parent_tool_use_id"), 0
            while p and depth < 4:
                for t in self.ledger.tasks.values():
                    if t.get("tool_use_id") == p and t.get("state") in L.TASK_TERMINAL \
                            and t.get("claims"):
                        V.verify_task_result(self.ledger, t["id"])
                up = self.ledger.actions.get(self.ledger.action_id_for(p) or "")
                p, depth = (up.get("parent_tool_use_id") if up else None), depth + 1
        except Exception as e:
            log(f"[router] task recheck failed: {type(e).__name__}")

    def _injected_notification(self, m, text):
        """The CLI folded a task notification into a live run as a user turn.
        If a foreground run is live, its answer may mention it: flag it."""
        ids = _TASK_ID_RE.findall(text or "")
        run = self._ensure_run()
        h = self._owner_handle()
        if h is not None:
            h.contaminated_by.extend(ids or ["?"])
            self.ledger.note("foreground_contaminated", turn_id=h.turn_id, task_ids=ids)
            self.stats["contaminated_foreground"] += 1
        else:
            # No foreground turn owns this run yet. Remember WHICH tasks it
            # contains: a query that is delivered into the run afterwards takes
            # it over (see _echo) and must inherit the contamination, or its
            # answer would be spoken with no hold just because the notice came
            # before the turn did.
            run.task_ids.extend(ids or ["?"])
            if run.owner is None:
                run.owner = UNSOLICITED
                run.report_text.extend(_text_of(x) for x in run.buffer)
                run.buffer.clear()

    def _echo(self, h: TurnHandle, m):
        run = self._ensure_run()
        if h.state != L.T_SENT:
            if h.terminal:
                # the echo of a handle that is already dead: its run is stale
                self.quarantine.add("stale_echo", m, self.epoch)
                if run.owner is None:
                    run.owner = STALE
                    run.buffer.clear()
                    if not self.live_bg():
                        # its request was abandoned, but the CLI is about to
                        # run it anyway: stop it (with live background tasks
                        # this would kill them, so then it just runs silently)
                        asyncio.create_task(self._interrupt_stale())
            else:
                self.stats["duplicate_echo"] += 1
            return
        if run.owner is not None:
            # The CLI can deliver a queued user message INTO a run that is
            # still working (seen live 2026-09-30: an abandoned turn was mid
            # tool call, the next query's echo arrived inside that run, and the
            # run ended with ONE result). The new query then never gets a run
            # of its own. So when the run's owner is no longer wanted (an
            # abandoned or interrupted turn, a background report, a stale run)
            # the new turn takes the run over; what comes after its echo is its
            # answer. A run still being consumed by a live turn is NOT taken:
            # that would hand one turn's answer to another.
            prev = run.owner
            takeable = (prev in (UNSOLICITED, STALE)
                        or (isinstance(prev, TurnHandle)
                            and (prev.abandoned or prev.interrupt_requested or prev.terminal)))
            if not takeable:
                self.quarantine.add("echo_inside_owned_run", m, self.epoch)
                return
            if isinstance(prev, TurnHandle) and not prev.terminal:
                prev.finish(L.T_ABORTED, "superseded_by_next_turn")
            # what the run already held about background tasks is now part of
            # THIS turn's answer: flag it so the answer is held, not streamed
            inherited = list(run.task_ids) + (list(prev.contaminated_by)
                                              if isinstance(prev, TurnHandle) else [])
            if inherited:
                h.contaminated_by.extend(inherited)
                self.ledger.note("foreground_contaminated", turn_id=h.turn_id,
                                 task_ids=inherited, how="merged_into_run")
                self.stats["contaminated_foreground"] += 1
            run.task_ids = []
            run.owner = h
            run.buffer = []
            run.report_text = []
            h.state = L.T_CLAIMED
            h.claimed_at = time.monotonic()
            self.ledger.turn_state(h.turn_id, L.T_CLAIMED, merged_into_run=True)
            self._log_claim(h, merged=True)
            self.ledger.note("turn_merged_into_run", turn_id=h.turn_id,
                             previous=(prev.turn_id if isinstance(prev, TurnHandle) else prev))
            self.stats["merged_turn"] += 1
            return
        pre = run.buffer
        run.owner = h
        run.buffer = []
        h.state = L.T_CLAIMED
        h.claimed_at = time.monotonic()
        self.ledger.turn_state(h.turn_id, L.T_CLAIMED)
        self._log_claim(h)
        if pre:
            if h.kind == "command":                    # commands answer before they echo
                for x in pre:
                    h._push("msg", x)
            else:
                self.quarantine.add("content_before_echo", pre[0], self.epoch)

    # -- results ---------------------------------------------------------------

    def _on_result(self, m):
        run = self.run if self.run is not None else Run()
        kind = _origin_kind(m)
        unsolicited = kind not in (None, "human")
        owner = run.owner
        end_run = True
        try:
            if unsolicited:
                if isinstance(owner, TurnHandle):
                    # not something we have seen: do not let it end a
                    # foreground turn
                    self.quarantine.add("task_result_inside_foreground_run", m, self.epoch)
                    end_run = False
                    return
                synthetic = (getattr(m, "num_turns", None) == 0
                             and not (getattr(m, "result", "") or "").strip())
                text = " ".join(run.report_text + [_text_of(x) for x in run.buffer])
                self.reports.append(PendingReport(
                    time.time(), [], (text or (getattr(m, "result", "") or ""))[:2000],
                    synthetic, self.epoch))
                self.stats["synthetic_report" if synthetic else "unsolicited_report"] += 1
                if synthetic:
                    self.ledger.note("synthetic_task_run", epoch=self.epoch)
                return
            if isinstance(owner, TurnHandle):
                self._finish_foreground(owner, m)
                return
            if owner == STALE:
                self.stats["stale_run_ended"] += 1
                return
            # a foreground-shaped result that no echo claimed. This is the
            # France/Italy case: never pair it with whoever is waiting.
            # A model-initiated wake-up (a "checking back in ten minutes" turn that
            # fires on its own) lands here too. It is still never spoken or paired,
            # but its existence and size are no longer silent: one record-only
            # report and one ledger note, so the loss is visible (2026-10-15 22:01).
            text = " ".join(run.report_text + [_text_of(x) for x in run.buffer]).strip() \
                or (getattr(m, "result", "") or "").strip()
            if text:
                self.reports.append(PendingReport(time.time(), [], text[:2000], False, self.epoch))
                self.stats["unowned_report"] += 1
                self.ledger.note("unowned_result_text", epoch=self.epoch, chars=len(text))
            self.quarantine.add("unowned_result", m, self.epoch)
        finally:
            if end_run:
                self.run = None
                self.last_run_end = time.monotonic()

    async def _interrupt_stale(self):
        """Stop the run an abandoned query started, at most once per
        STALE_INTERRUPT_MIN_GAP_S. Called only from _echo() for a dead handle's
        echo, which has already quarantined it: skipping the interrupt here
        changes nothing about that. Never touches the live handle (router
        interrupt() is a different path and is not limited by this)."""
        now = time.monotonic()
        last = self._last_stale_interrupt
        if last is not None and now - last < STALE_INTERRUPT_MIN_GAP_S:
            self.stats["stale_interrupt_suppressed"] += 1
            log(f"[router] stale interrupt suppressed: another went out "
                f"{now - last:.2f}s ago")
            return
        self._last_stale_interrupt = now      # set before the await: a second echo in the same tick sees it
        try:
            await asyncio.wait_for(self.client.interrupt(), 2)
        except Exception:
            pass

    def _finish_foreground(self, h: TurnHandle, m):
        h.result = m
        term = getattr(m, "terminal_reason", None) or ""
        if h.state in (L.T_CLAIMED,):
            h.state = L.T_RUNNING
            self.ledger.turn_state(h.turn_id, L.T_RUNNING)
        if term.startswith("aborted") or h.interrupt_requested:
            h._push("result", m)
            h.finish(L.T_INTERRUPTED, term or "interrupt")
            return
        if getattr(m, "is_error", False) or getattr(m, "subtype", "success") != "success":
            h._push("result", m)
            h.finish(L.T_FAILED, term or getattr(m, "subtype", "error"))
            return
        self.ledger.turn_state(h.turn_id, L.T_RESULT,
                               session_id=getattr(m, "session_id", None))
        h._push("result", m)
        h.finish(L.T_DONE)

    # ---- process evidence (approximate, read-only) -------------------------

    def _maybe_snapshot(self, m):
        tid = getattr(m, "task_id", None)
        row = self.ledger.tasks.get(tid)
        if (not self.use_proc or row is None or tid in self._snapped
                or row.get("task_kind") != L.KIND_BG_AGENT):
            return
        self._snapped.add(tid)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def go():
            tree = await loop.run_in_executor(None, proc_tree, self.cli_pid)
            if tree is None:
                return
            snap = [p for p in tree if (p["pid"], p["lstart"]) not in self._baseline]
            if snap:
                self.ledger.task_set(tid, process_snapshot=snap[:20],
                                     process_snapshot_confidence="approximate")
        asyncio.create_task(go())

    async def capture_baseline(self):
        if not self.use_proc:
            return
        try:
            loop = asyncio.get_running_loop()
            tree = await loop.run_in_executor(None, proc_tree, self.cli_pid)
            self._baseline = {(p["pid"], p["lstart"]) for p in (tree or [])}
        except Exception:
            self._baseline = set()

    # ---- cancellation ------------------------------------------------------

    def live_bg(self):
        return self.ledger.live_tasks(kind=L.KIND_BG_AGENT, epoch=self.epoch)

    async def stop_task(self, task_id, reason="USER_STOP", timeout=10.0):
        """Scoped stop: one task, the foreground run untouched (Spike 5)."""
        row = self.ledger.tasks.get(task_id)
        if row is None or row.get("state") not in L.TASK_LIVE or row.get("epoch") != self.epoch:
            return False
        self.ledger.task_cancel_intent(task_id, reason)     # intent BEFORE effect
        await asyncio.wait_for(self.client.stop_task(task_id), timeout)
        return True

    async def interrupt(self, *, reason="INTERRUPT", force=False, wait=8.0):
        """Cancel the foreground run. interrupt() also kills EVERY background
        task (Spike 3/5), so with live protected tasks the run is muted and left
        to finish unless forced or the escalation timer fires. Returns one of
        none | aborted | protected | timeout."""
        h = self.active_handle()
        if h is None:
            # a SENT-but-unclaimed handle is abandoned; its run, if it ever
            # starts, is recognised as stale by its echo
            for x in self.handles.values():
                if x.state == L.T_SENT:
                    x.finish(L.T_ABORTED, "abandoned_before_claim")
            return "none"
        live = self.live_bg()
        if live and self.protect_background and not force and any(t.get("protected") for t in live):
            h.abandoned = True
            self._schedule_escalation(h)
            self.ledger.note("barge_in_protected", turn_id=h.turn_id,
                             tasks=[t["id"] for t in live])
            return "protected"
        for t in live:
            self.ledger.task_cancel_intent(t["id"], reason)
        h.interrupt_requested = True
        try:
            await asyncio.wait_for(self.client.interrupt(), 2)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.done.wait(), wait)
        except asyncio.TimeoutError:
            return "timeout"
        return "aborted"

    def _schedule_escalation(self, h):
        async def esc():
            try:
                await asyncio.sleep(self.barge_escalate_s)
                if not h.terminal:
                    self.ledger.note("barge_in_escalated", turn_id=h.turn_id)
                    await self.interrupt(reason="BARGE_IN_ESCALATION", force=True)
            except asyncio.CancelledError:
                pass
            finally:
                self._escalations.discard(task)
        task = asyncio.create_task(esc())
        self._escalations.add(task)

    async def stop_all(self, reason="STOP_ALL"):
        """Stop every live background task by name, then the foreground run.
        Order: intent -> stop_task each -> interrupt."""
        stopped = 0
        for t in list(self.live_bg()):
            try:
                if await self.stop_task(t["id"], reason):
                    stopped += 1
            except Exception as e:
                self.ledger.note("stop_task_failed", task_id=t["id"], error=type(e).__name__)
        res = await self.interrupt(reason=reason, force=True)
        return stopped, res

    def pending_reports(self):
        return list(self.reports)


# ---- epoch manager ----------------------------------------------------------

class EpochManager:
    """Opens epochs and tears them down. Reconciliation of the previous
    process's ledger happens exactly once, before the first epoch opens."""

    def __init__(self, ledger, quarantine=None, *, protect_background=True,
                 echo_timeout_s=20.0, barge_escalate_s=20.0, proc=True):
        self.ledger = ledger
        self.quarantine = quarantine or Quarantine()
        self.tasks = TaskRegistry(ledger, self.quarantine,
                                  protect_background=protect_background)
        self.kw = dict(protect_background=protect_background,
                       echo_timeout_s=echo_timeout_s,
                       barge_escalate_s=barge_escalate_s, proc=proc)
        self.current: EpochRouter | None = None
        self.reconciled = None

    def reconcile_once(self, proc_check=None):
        if self.reconciled is None:
            self.reconciled = self.ledger.reconcile(proc_check or L.process_alive)
            if any(self.reconciled.values()):
                log(f"[ledger] reconciled previous run: {self.reconciled}")
        return self.reconciled

    async def open(self, client, *, session_id=None, reason="start", cli_pid=None):
        self.reconcile_once()
        eid = self.ledger.epoch_open(session_id=session_id, cli_pid=cli_pid, reason=reason)
        r = EpochRouter(client, self.ledger, epoch=eid, quarantine=self.quarantine,
                        tasks=self.tasks, cli_pid=cli_pid, **self.kw)
        self.current = r
        await r.capture_baseline()
        await r.start()
        return r
