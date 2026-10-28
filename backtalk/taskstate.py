"""Typed task state, failure classes and recovery decisions (roadmap Phase C, sections 11 and 12). PURE: it reads ledger rows and
returns values; it writes nothing, speaks nothing and keeps no state.

The ledger already types every turn, action and background task with enforced transition tables (event-sourced, persistent). This
module does NOT replace that. It adds what was missing on top of it:

  1 ONE typed view over all three row kinds, in the roadmap's vocabulary
        PLANNED RUNNING WAITING_RESULT VERIFYING SUCCEEDED FAILED PARTIAL CANCELLED UNKNOWN
    (WAITING_PERMISSION is in the roadmap but nothing produces it: the spoken permission gate was removed on 2026-10-08.)
    Execution and verification stay separate fields: SUCCEEDED means the tool/agent finished, and `claim_allowed` says whether a
    result may be REPORTED as fact (only VERIFIED or NOT_REQUIRED). Nothing unverified becomes a confident success.
  2 FAILURE CLASSES (roadmap section 12): RETRYABLE USER_ACTION_REQUIRED PERMISSION_REQUIRED TOOL_ERROR INVALID_INPUT
    PARTIAL_SUCCESS UNKNOWN. A tool error is classified ONCE, at the moment the router sees the result, from the transient error
    text; only the short class code is stored (the ledger never keeps bodies).
  3 A RECOVERY DECISION per failure: RETRY_ONCE | CHECK_FIRST | ASK_USER | REPORT | NONE. Fail closed: a retry is only ever
    recommended when NOTHING with side effects could have run (a turn that never started, or only read-only tools), exactly once,
    and never after the same cause already repeated. Anything that might have changed the world is CHECK_FIRST: find out what
    happened before doing anything again. This module only DECIDES; no code here retries anything.
  4 CANCELLATION semantics as data: CANCELLED carries `cancel_cause` (user | interrupt | unattributed | aborted | stopped) so a
    stop the user asked for is never confused with work lost to a barge-in.

Every state a ledger can hold must map; tests/test_taskstate.py fails when a new ledger state is added without a mapping.
"""
import re
from dataclasses import dataclass

from backtalk import ledger as L

# ---- statuses (roadmap section 11) ----------------------------------------------------------------------------
PLANNED, RUNNING, WAITING_RESULT, VERIFYING = "PLANNED", "RUNNING", "WAITING_RESULT", "VERIFYING"
SUCCEEDED, FAILED, PARTIAL, CANCELLED, UNKNOWN = "SUCCEEDED", "FAILED", "PARTIAL", "CANCELLED", "UNKNOWN"
WAITING_PERMISSION = "WAITING_PERMISSION"          # never produced: the permission gate no longer exists
STATUSES = (PLANNED, WAITING_PERMISSION, RUNNING, WAITING_RESULT, VERIFYING, SUCCEEDED, FAILED, PARTIAL, CANCELLED, UNKNOWN)
LIVE = (PLANNED, RUNNING, WAITING_RESULT, VERIFYING)

# ---- failure classes (roadmap section 12) ---------------------------------------------------------------------
RETRYABLE, USER_ACTION_REQUIRED, PERMISSION_REQUIRED = "RETRYABLE", "USER_ACTION_REQUIRED", "PERMISSION_REQUIRED"
TOOL_ERROR, INVALID_INPUT, PARTIAL_SUCCESS, FC_UNKNOWN = "TOOL_ERROR", "INVALID_INPUT", "PARTIAL_SUCCESS", "UNKNOWN"
FAILURE_CLASSES = (RETRYABLE, USER_ACTION_REQUIRED, PERMISSION_REQUIRED, TOOL_ERROR, INVALID_INPUT, PARTIAL_SUCCESS, FC_UNKNOWN)

# ---- recovery decisions -----------------------------------------------------------------------------------------
RETRY_ONCE, CHECK_FIRST, ASK_USER, REPORT, NONE = "RETRY_ONCE", "CHECK_FIRST", "ASK_USER", "REPORT", "NONE"
RECOVERY_ACTIONS = (RETRY_ONCE, CHECK_FIRST, ASK_USER, REPORT, NONE)
MAX_SAME_CAUSE = 1             # a retry is recommended only when the SAME cause has not already failed just before

# tools that only read. Anything not listed here is assumed to have side effects (fail closed).
READ_ONLY_TOOLS = frozenset({"Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch", "WebSearch", "ToolSearch",
                             "TaskOutput", "BashOutput", "ListMcpResourcesTool", "ReadMcpResourceTool"})

# turn failure causes (what router.TurnHandle.finish records) and what they mean for "did anything run?"
_NOTHING_RAN = ("no_echo", "send_failed")          # the query never reached the model
_TURN_CLASS = {
    "no_echo": RETRYABLE,
    "send_failed": RETRYABLE,
    "api_error": RETRYABLE,
    "stream_lost": RETRYABLE,
    "oversize": USER_ACTION_REQUIRED,
    "max_turns": PARTIAL_SUCCESS,
    "error_max_turns": PARTIAL_SUCCESS,
}

_ERR_USER = re.compile(r"sign[- ]?in\b|please log ?in|log ?in (required|failed|again)|you must log ?in|authenticat|credential|"
                       r"captcha|verify (that )?you|session expired|quota|billing|subscription|out of usage|"
                       r"password (is )?required|a password is required")
_ERR_PERMISSION = re.compile(r"permission denied|operation not permitted|\beacces\b|\beperm\b|(not |un)authori[sz]ed|forbidden|"
                             r"\b40[13]\b|requires approval|not allowed|access denied")
_ERR_RETRY = re.compile(r"timed? ?out|timeout|etimedout|temporar|try again|rate[- ]?limit|\b429\b|\b50\d\b|overloaded|"
                        r"econn(reset|refused)|connection (reset|refused|error|closed)|\bnetwork (error|unreachable|is unreachable|"
                        r"is down|down)|service unavailable|\bunavailable\b|service busy|resource busy")
_ERR_INPUT = re.compile(r"no such file|not found|does not exist|\benoent\b|invalid|malformed|cannot parse|syntax error|"
                        r"unknown (option|flag|argument|command)|usage:|missing (required|argument)|is a directory|"
                        r"not a directory|unterminated|unexpected token")


@dataclass(frozen=True)
class Recovery:
    action: str
    reason: str                    # a code, never text
    safe_to_retry: bool = False


@dataclass(frozen=True)
class TaskView:
    kind: str                      # "turn" | "action" | "task"
    id: str
    status: str
    execution: str                 # RUNNING | SUCCESS | FAILURE | CANCELLED | UNKNOWN
    verification: object           # the ledger's verification state, or None
    claim_allowed: bool            # may a result be REPORTED as fact? only VERIFIED or NOT_REQUIRED
    failure_class: object = None
    cause: str = ""
    cancel_cause: object = None
    maybe_running: bool = False    # outcome unknown and the process may still be alive (ORPHAN_RUNNING)
    started_at: object = None
    ended_at: object = None
    epoch: object = None
    parent: object = None          # the turn an action or task belongs to
    tool: object = None
    task_kind: object = None
    side_effects: object = None    # True / False / None (unknown)
    reported: bool = False         # an outcome was handed to the user or the model
    recovery: Recovery = Recovery(NONE, "not_failed")


# ---- pure helpers ---------------------------------------------------------------------------------------------

def _last_cause(row) -> str:
    h = row.get("history") or []
    for ent in reversed(h):
        if len(ent) >= 3 and ent[2]:
            return str(ent[2])[:120]
    return str(row.get("cause") or "")[:120]


def _times(row):
    h = row.get("history") or []
    return (h[0][0] if h else None), (h[-1][0] if h else None)


def is_read_only(tool, tier=None) -> bool:
    """True only for a tool NAMED in READ_ONLY_TOOLS. An unknown tool is not read-only. `tier` is accepted and IGNORED: it is a
    legacy field from the retired permission classifier (35 of 60 Bash rows carry tier 0), not evidence that nothing changed."""
    return tool in READ_ONLY_TOOLS


def classify_tool_error(tool, content) -> str:
    """The failure class of a tool error, from its text (a string, or a list of {"text": ...} blocks). The text is read here
    and dropped; the caller stores only the returned code. Precedence: user action, permission, retryable, invalid input."""
    try:
        if isinstance(content, (list, tuple)):
            text = " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
        else:
            text = "" if content is None else str(content)
        low = text[:4000].lower()
    except Exception:
        return TOOL_ERROR
    if _ERR_USER.search(low):
        return USER_ACTION_REQUIRED
    if _ERR_PERMISSION.search(low):
        return PERMISSION_REQUIRED
    if _ERR_RETRY.search(low):
        return RETRYABLE
    if _ERR_INPUT.search(low):
        return INVALID_INPUT
    return TOOL_ERROR


def classify_turn_cause(cause) -> str:
    c = str(cause or "")
    for key in _NOTHING_RAN:
        if c == key or c.startswith(key + ":"):
            return _TURN_CLASS[key]
    return _TURN_CLASS.get(c, FC_UNKNOWN)


def nothing_ran(cause) -> bool:
    c = str(cause or "")
    return any(c == k or c.startswith(k + ":") for k in _NOTHING_RAN)


def turn_side_effects(led, turn_id):
    """Could anything with side effects have run in this turn? FAIL CLOSED: True when ANY of these holds
      - a recorded action of the turn is not known to be read-only;
      - a descendant action (a subagent's call, linked by parent_tool_use_id to one of the turn's actions) is not read-only;
      - a background task was launched in the turn (it may still be running, or have run);
      - an action with NO turn link, in the same epoch and inside the turn's lifetime, is not read-only (the router sets
        turn_id None whenever the handle is gone, so an unlinked action may belong to this turn).
    The ledger records every tool call as an action row, so none of the above means no tool ran."""
    row = led.turns.get(turn_id) or {}
    started, ended = _times(row)
    epoch = row.get("epoch")
    mine, uses = [], set()
    for a in led.actions.values():
        if a.get("turn_id") == turn_id:
            mine.append(a)
            if a.get("tool_use_id"):
                uses.add(a["tool_use_id"])
    if any(not is_read_only(a.get("tool")) for a in mine):
        return True
    if any(t.get("launched_in_turn") == turn_id for t in led.tasks.values()):
        return True
    grew = True
    while grew:                                                    # descendants of the turn's actions, to any depth
        grew = False
        for a in led.actions.values():
            if a.get("parent_tool_use_id") in uses and a.get("tool_use_id") not in uses and a.get("tool_use_id"):
                uses.add(a["tool_use_id"])
                grew = True
                if not is_read_only(a.get("tool")):
                    return True
            elif a.get("parent_tool_use_id") in uses and not a.get("tool_use_id") and not is_read_only(a.get("tool")):
                return True
    lo = (started or 0) - 5
    hi = (ended or float("inf")) + 5
    for a in led.actions.values():
        if a.get("turn_id") is None and a.get("epoch") == epoch and not is_read_only(a.get("tool")):
            at = (a.get("history") or [[None]])[0][0]
            if at is None or lo <= at <= hi:
                return True
    return False


def recover(status, failure_class, *, side_effects, recent_same_cause=0, retried=False) -> Recovery:
    """What to do about a failure. Only FAILED, PARTIAL and UNKNOWN statuses need one. `side_effects` is True, False or None
    (unknown, treated as True)."""
    if status not in (FAILED, PARTIAL, UNKNOWN):
        return Recovery(NONE, "not_failed")
    risky = side_effects is not False
    if status == UNKNOWN:
        return Recovery(CHECK_FIRST, "outcome_unknown")                  # never retry on a guess: find out what happened first
    fc = failure_class or FC_UNKNOWN
    if fc == RETRYABLE:
        if retried or recent_same_cause >= MAX_SAME_CAUSE:
            return Recovery(ASK_USER, "keeps_failing")                   # not transient any more: stop and say so
        if risky:
            return Recovery(CHECK_FIRST, "retryable_but_may_have_run")
        return Recovery(RETRY_ONCE, "retryable_nothing_ran", True)
    if fc in (USER_ACTION_REQUIRED, PERMISSION_REQUIRED, INVALID_INPUT):
        return Recovery(ASK_USER, fc.lower())
    if fc == PARTIAL_SUCCESS:
        return Recovery(ASK_USER, "partial_continue")
    if fc == TOOL_ERROR:
        return Recovery(REPORT, "tool_error")
    return Recovery(CHECK_FIRST if risky else REPORT, "unclassified")


# ---- the views ------------------------------------------------------------------------------------------------

def _reported(row) -> bool:
    return bool(row.get("deliveries") or row.get("user_visible"))


STREAK_GAP_S = 600.0           # failures further apart than this are separate incidents


def _cause_key(row):
    c = _last_cause(row)
    return (classify_turn_cause(c), c.split(":", 1)[0])


def same_cause_streak(led, tid) -> int:
    """How many turns IMMEDIATELY before this one failed with the same cause (same class and same cause family, so
    send_failed:OSError and send_failed:TypeError count as one), with no gap over STREAK_GAP_S. Barge-in (INTERRUPTED/ABORTED)
    turns are skipped, anything else ends the streak. Walks backwards from the end of the ledger without copying it."""
    me = led.turns.get(tid)
    if me is None:
        return 0
    key, n, last_t = _cause_key(me), 0, _times(me)[0]
    seen = False
    for k, r in reversed(led.turns.items()):
        if not seen:
            seen = k == tid
            continue
        if r.get("state") in (L.T_INTERRUPTED, L.T_ABORTED):
            continue
        if r.get("state") != L.T_FAILED or _cause_key(r) != key:
            break
        t = _times(r)[1]
        if last_t is not None and t is not None and abs(last_t - t) > STREAK_GAP_S:
            break
        n += 1
        last_t = _times(r)[0]
    return n


def turn_view(led, tid, recent_same_cause=None) -> TaskView:
    row = led.turns[tid]
    if recent_same_cause is None:
        recent_same_cause = same_cause_streak(led, tid) if row.get("state") == L.T_FAILED else 0
    st, cause = row.get("state"), _last_cause(row)
    started, ended = _times(row)
    status, execution, fc, cancel = {
        L.T_CREATED: (PLANNED, "RUNNING", None, None),
        L.T_SENT: (WAITING_RESULT, "RUNNING", None, None),
        L.T_CLAIMED: (RUNNING, "RUNNING", None, None),
        L.T_RUNNING: (RUNNING, "RUNNING", None, None),
        L.T_RESULT: (RUNNING, "RUNNING", None, None),
        L.T_DONE: (SUCCEEDED, "SUCCESS", None, None),
        L.T_FAILED: (FAILED, "FAILURE", classify_turn_cause(cause), None),
        L.T_INTERRUPTED: (CANCELLED, "CANCELLED", None, "interrupt"),
        L.T_ABORTED: (CANCELLED, "CANCELLED", None, "aborted"),
    }[st]
    side = None
    if status == FAILED:
        # stream_lost can lose a tool_use before its action row exists: unknown, never "nothing ran"
        side = False if nothing_ran(cause) else (None if str(cause) == "stream_lost" else turn_side_effects(led, tid))
    return TaskView("turn", tid, status, execution, None, False, fc, cause, cancel, False, started, ended,        # a turn has no verification: claim_allowed is for results of actions and tasks
                    row.get("epoch"), None, None, row.get("kind"), side, _reported(row),
                    recover(status, fc, side_effects=side, recent_same_cause=recent_same_cause))


def action_view(led, aid) -> TaskView:
    row = led.actions[aid]
    st, cause = row.get("state"), _last_cause(row)
    v = (row.get("verification") or {}).get("state")
    started, ended = _times(row)
    side = not is_read_only(row.get("tool"), row.get("tier"))
    fc, cancel, claim = None, None, False
    if st in (L.A_REQUESTED, L.A_ALLOWED):
        status, execution = WAITING_RESULT, "RUNNING"
    elif st == L.A_OK:
        execution = "SUCCESS"
        if v == L.V_PENDING:
            status = VERIFYING
        elif v == L.V_FAILED:                              # it reported success and the check says it did not land
            status, execution, fc = FAILED, "FAILURE", FC_UNKNOWN      # reported ok, did not land: what happened is unknown
            cause = cause or "verification_failed"
        else:
            status = SUCCEEDED
            claim = v in (L.V_VERIFIED, L.V_NOT_REQUIRED)
    elif st == L.A_ERROR:
        status, execution = FAILED, "FAILURE"
        fc = row.get("failure_class") if row.get("failure_class") in FAILURE_CLASSES else TOOL_ERROR
    elif st == L.A_LOST:
        status, execution = UNKNOWN, "UNKNOWN"
    elif st == L.A_ABORTED:
        status, execution, cancel = CANCELLED, "CANCELLED", "aborted"
    elif st == L.A_DENIED:                                 # legacy rows from the retired permission gate
        status, execution, fc = FAILED, "FAILURE", PERMISSION_REQUIRED
    else:
        raise KeyError(st)
    return TaskView("action", aid, status, execution, v, claim, fc, cause, cancel, False, started, ended, row.get("epoch"),
                    row.get("turn_id"), row.get("tool"), None, side, _reported(row),
                    recover(status, fc, side_effects=side))


_TASK_MAP = {
    L.K_LAUNCHED: (RUNNING, "RUNNING", None, None, False),
    L.K_RUNNING: (RUNNING, "RUNNING", None, None, False),
    L.K_FAILED: (FAILED, "FAILURE", TOOL_ERROR, None, False),
    L.K_STOPPED_USER: (CANCELLED, "CANCELLED", None, "user", False),
    L.K_KILLED_INTERRUPT: (CANCELLED, "CANCELLED", None, "interrupt", False),
    L.K_STOPPED_UNATTRIBUTED: (CANCELLED, "CANCELLED", None, "unattributed", False),
    L.K_LOST_REBUILD: (UNKNOWN, "UNKNOWN", None, None, False),
    L.K_UNKNOWN_CRASH: (UNKNOWN, "UNKNOWN", None, None, False),
    L.K_DEAD_UNKNOWN: (UNKNOWN, "UNKNOWN", None, None, False),
    L.K_ORPHAN_RUNNING: (UNKNOWN, "UNKNOWN", None, None, True),
}


def task_view(led, tid) -> TaskView:
    row = led.tasks[tid]
    st, cause = row.get("state"), _last_cause(row)
    started, ended = _times(row)
    v = (row.get("result_verification") or {}).get("state")
    claim = False
    if st == L.K_COMPLETED:
        if v == L.V_PENDING:
            status, execution = VERIFYING, "SUCCESS"
        elif v == L.V_FAILED:                              # finished, but its report contradicts its own tool evidence
            status, execution, fc = PARTIAL, "SUCCESS", PARTIAL_SUCCESS
            return _task(row, tid, status, execution, v, False, fc, cause, None, False, started, ended,
                         recover(status, fc, side_effects=True))
        else:
            status, execution = SUCCEEDED, "SUCCESS"
            claim = v in (L.V_VERIFIED, L.V_NOT_REQUIRED)
        return _task(row, tid, status, execution, v, claim, None, cause, None, False, started, ended, Recovery(NONE, "not_failed"))
    status, execution, fc, cancel, maybe = _TASK_MAP[st]
    return _task(row, tid, status, execution, v, False, fc, cause, cancel, maybe, started, ended,
                 recover(status, fc, side_effects=True))        # a background agent is assumed to have side effects


def _task(row, tid, status, execution, v, claim, fc, cause, cancel, maybe, started, ended, rec):
    return TaskView("task", tid, status, execution, v, claim, fc, cause, cancel, maybe, started, ended, row.get("epoch"),
                    row.get("launched_in_turn"), None, row.get("task_kind"), None, _reported(row), rec)


def view_of(led, kind, ident) -> TaskView:
    return {"turn": turn_view, "action": action_view, "task": task_view}[kind](led, ident)


def recent(led, n=20, *, kinds=("turn", "action", "task"), statuses=None):
    """The newest `n` views across the chosen kinds, newest first. Reads only."""
    out = []
    tables = {"turn": led.turns, "action": led.actions, "task": led.tasks}
    for kind in kinds:
        rows = list(tables[kind].items())[-max(n * 3, 50):]
        for ident, _ in rows:
            try:
                v = view_of(led, kind, ident)
            except (KeyError, TypeError, AttributeError, IndexError, ValueError):
                continue
            if statuses is None or v.status in statuses:
                out.append(v)
    out.sort(key=lambda v: (v.ended_at or v.started_at or 0), reverse=True)
    return out[:n]


def explain(v: TaskView) -> str:
    """One code-built line for logs and the CLI: ids, codes and times, never message text."""
    bits = [f"{v.kind} {v.id} {v.status}"]
    if v.cause:
        bits.append(f"cause={v.cause}")
    if v.cancel_cause:
        bits.append(f"cancel={v.cancel_cause}")
    if v.failure_class:
        bits.append(f"class={v.failure_class}")
    if v.recovery.action != NONE:
        bits.append(f"recovery={v.recovery.action}({v.recovery.reason})")
    if v.status == SUCCEEDED and v.kind != "turn":
        bits.append("claim_ok" if v.claim_allowed else "claim_not_allowed")
    if v.maybe_running:
        bits.append("maybe_still_running")
    return " ".join(bits)
