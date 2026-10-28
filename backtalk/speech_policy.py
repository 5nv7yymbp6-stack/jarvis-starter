"""Which events may reach the mouth, and what the mouth is told.

The router and the ledger know what happened; this module decides what is
SAID. It is a separate, small, pure module on purpose: the rules that keep a
background report, a synthetic lost-task notice or a stale result out of the
user's ears must be readable in one screen and testable without a voice.

RULES
  * NEVER spoken, whatever the context: synthetic lost-task reports, stale
    results, quarantined messages, and any model-authored text about
    background work.
  * Background outcomes are spoken only as a line BUILT HERE from ledger
    fields. The one piece of task-supplied text allowed in (a short result
    summary) is stripped of anything that could be read as a stage direction
    or markup, and is spoken ONLY when it contains no number. A number in a
    model's summary is a claim about a tool result: it is spoken only as the
    task's own recorded evidence confirmed it (verify.py), and a number the
    evidence contradicts is replaced by the tool's number, never repeated.
  * A background outcome never interrupts: it waits for a quiet mouth, no
    active foreground turn, and an active conversation. Anything not announced
    is handed to the next foreground turn as a short factual note instead.
  * What a FOREGROUND MODEL is handed about a finished background task is built
    here from the ledger too (take_notes), and carries the task's verification
    state. The agent's own report never reaches the foreground model directly:
    brain.py withholds the CLI's raw task notification (see is_task_notification),
    so a number the evidence does not support cannot enter its context as a fact.
  * Unknown kinds are dropped (fail closed).
  * A normal foreground answer is the model's own words and is spoken as it
    always was.
"""
import re
from dataclasses import dataclass, field

from backtalk import evidence as E
from backtalk import ledger as L

SPEAK, DEFER, DROP = "SPEAK", "DEFER", "DROP"

# The startup scan for outcomes whose hand-off was never recorded (F-09). Bounded on every side: only outcomes
# from the last SCAN_WINDOW_S seconds by their durable ledger timestamp (inclusive: age <= the window), only
# the newest SCAN_MAX_ROWS rows of each table are looked at, and at most SCAN_LIMIT are queued.
SCAN_WINDOW_S = 24 * 3600.0
SCAN_MAX_ROWS = 2000
SCAN_LIMIT = 10

# candidate kinds
FG_ANSWER = "FG_ANSWER"
FG_ERROR = "FG_ERROR"
BG_COMPLETION = "BG_COMPLETION"          # a background task reached a terminal state
BG_STOP_CONFIRM = "BG_STOP_CONFIRM"      # the user stopped one; say so once
ACTION_BAD_NEWS = "ACTION_BAD_NEWS"      # a consequential action failed / was lost
MODEL_REPORT = "MODEL_REPORT"            # model-authored text about background work
SYNTHETIC_REPORT = "SYNTHETIC_REPORT"    # the CLI's "no completion record" run
STALE = "STALE"
QUARANTINE = "QUARANTINE"

NEVER = {MODEL_REPORT, SYNTHETIC_REPORT, STALE, QUARANTINE}

_UNSAFE = re.compile(r"[<>\x00-\x1f\x7f`]")


@dataclass
class Candidate:
    kind: str
    provenance: str = "code"              # "code" | "model"
    source_id: str = ""                   # task_id / action_id / turn_id
    text: str = ""                        # only for FG_ANSWER / FG_ERROR
    announce: bool = False


@dataclass
class Context:
    fg_active: bool = False               # a foreground turn is in flight
    mouth_busy: bool = False
    conversation_active: bool = True      # addressee gate is ACTIVE / PTT
    zipped: bool = False


def sanitize(text, limit=80) -> str:
    """Task-supplied words, made safe to read aloud and to hand to a model:
    no <<stage directions>>, no markup, no control characters, short."""
    s = _UNSAFE.sub(" ", str(text or ""))
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit].rstrip() + "..."


def _vstate(task):
    return (task.get("result_verification") or {}).get("state")


def _name(task) -> str:
    return sanitize(task.get("description"), 60) or "a background task"


_NOTICE_PREFIX = "<task-notification>"
_NOTICE_ID_RE = re.compile(r"<task-id>\s*([^<\s]+)\s*</task-id>")


def is_task_notification(prompt) -> bool:
    """True for the CLI's own `<task-notification>` prompt: it carries the
    background agent's raw final text in a <result> block. Cannot raise."""
    try:
        return str(prompt or "").lstrip().startswith(_NOTICE_PREFIX)
    except Exception:
        return False


def task_notice_ids(prompt) -> list:
    try:
        return _NOTICE_ID_RE.findall(str(prompt or ""))
    except Exception:
        return []


_NOTICE_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.S)


def task_notice_result(prompt) -> str:
    """The agent's final report from a task notification, or ''. Cannot raise."""
    try:
        m = _NOTICE_RESULT_RE.search(str(prompt or ""))
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def _report_path(task):
    """Where brain.py saved this task's withheld report, if it did."""
    p = task.get("report_path")
    if p:
        return p
    try:
        from pathlib import Path
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task.get("id") or task.get("task_id") or ""))[:80]
        if safe:
            f = L.default_dir() / "reports" / f"{safe}.md"
            if f.exists():
                return str(f)
    except Exception:
        pass
    return None


def report_pointer(task) -> str:
    """For the foreground MODEL only (never spoken): where the full report is and
    what to do with it, so a finished agent's findings always reach Yousif."""
    if task.get("state") != L.K_COMPLETED:
        return ""
    p = _report_path(task)
    if not p:
        return ""
    return (f" Its full report is saved at {p}. Read that file now and tell him what it found, "
            "plainly and briefly; any number in it is the agent's claim, not checked, so say so if it matters.")


def verification_tag(task) -> str:
    """What the ledger says about a finished task's reported numbers, in words a
    model cannot mistake for a fact. Nothing for a task that needs no check."""
    if task.get("state") != L.K_COMPLETED:
        return ""
    state = (task.get("result_verification") or {}).get("state")
    if state == L.V_VERIFIED:
        return " (Ledger: VERIFIED against its own tool output.)"
    if state == L.V_FAILED:
        return " (Ledger: VERIFY_FAILED. Only the tool's number is a fact; the report's numbers are not.)"
    if state == L.V_UNVERIFIED:
        return " (Ledger: UNVERIFIED. None of the numbers in its report is a fact.)"
    return ""


def _numbers_phrase(nums) -> str:
    nums = [str(int(x)) for x in (nums or [])[:3]]
    return " and ".join(nums)


def _result_clause(task) -> str:
    """What may be said about a finished task's reported result. Built from
    the verification the router recorded; a model's number never gets here
    unless its own tool output confirmed it."""
    rv = task.get("result_verification") or {}
    state = rv.get("state")
    summ = sanitize(task.get("summary"), 80)
    if state == L.V_VERIFIED:
        said = []
        for c in (rv.get("matched") or [])[:2]:
            try:
                said.append(f"{int(c['n'])} {sanitize(c.get('noun'), 40)}".strip())
            except (KeyError, TypeError, ValueError):
                continue
        return (" It reported " + " and ".join(said) + ".") if said else ""
    if state == L.V_FAILED:
        got = _numbers_phrase(rv.get("facts"))
        return (" The count in its report didn't match what its own tool returned, "
                f"which was {got}." if got else
                " The count in its report didn't match what its own tool returned.")
    if state == L.V_UNVERIFIED:
        return " I've got its report."
    if not summ:
        return ""
    if E.extract_claims(task.get("summary")):          # no verification on record
        return " I've got its report."
    return f" It reported: {summ}" + ("" if summ.endswith((".", "!", "?")) else ".")


def task_line(task) -> str:
    """The sentence for a background task's outcome, built from the ledger."""
    n = _name(task)
    st = task.get("state")
    if st == L.K_COMPLETED:
        return f"Background task {n} finished." + _result_clause(task)
    if st == L.K_FAILED:
        return f"Background task {n} failed."
    if st == L.K_STOPPED_USER:
        return f"Stopped {n}."
    if st == L.K_KILLED_INTERRUPT:
        return f"Background task {n} was stopped when I was interrupted."
    if st == L.K_STOPPED_UNATTRIBUTED:
        return f"Background task {n} stopped, and I don't know why."
    if st == L.K_LOST_REBUILD:
        return f"Background task {n} was lost when I restarted my session."
    if st == L.K_ORPHAN_RUNNING:
        return f"After a crash, a process for {n} may still be running."
    if st in (L.K_UNKNOWN_CRASH, L.K_DEAD_UNKNOWN):
        return f"After a crash I can't tell whether {n} finished."
    return f"Background task {n} changed state."


def action_line(action) -> str | None:
    """Bad news about a consequential action, or None if there is none."""
    st = action.get("state")
    v = (action.get("verification") or {}).get("state")
    if st == L.A_LOST:
        return "I lost the result of an action, so I can't say whether it happened."
    if v == L.V_FAILED:
        return "An action reported success, but my check says it did not land."
    return None


FAILURE_LINES = {
    "oversize": "That result was too big for me to read, so I lost the thread. Say it again?",
    "stream_lost": "I lost my connection to the session mid-turn. Say it again?",
    "no_echo": "That one never started on my side. Say it again?",
    "max_turns": "I ran out of steps on that one.",
    # Phase C: causes that used to fall through to the generic apology. 17 of the 18 failed turns on record were api_error.
    "api_error": "I couldn't reach my brain just then, so that one didn't run. Say it again, and if it keeps happening, "
                 "check the sign-in and the network.",
    "send_failed": "That one never left my side, so nothing ran. Say it again?",
}


def failure_line(reason) -> str:
    r = str(reason)
    if r.startswith("send_failed:"):
        r = "send_failed"
    return FAILURE_LINES.get(r, "Something broke on my end, and that turn died. Say it again?")


def _tail(table, n):
    """The newest `n` rows of a ledger table (rows are kept in creation order), as (id, row)."""
    items = []
    for item in reversed(table.items()):
        items.append(item)
        if len(items) >= n:
            break
    return reversed(items)


class SpeechPolicy:
    def __init__(self, ledger=None, *, max_deferred=50):
        self.ledger = ledger
        self.max_deferred = max_deferred
        self.deferred: list = []          # Candidates waiting for the next turn
        self._surfaced: dict = {}         # task_id -> verification state last handed to the model
        self._bad_news: set = set()       # (action_id, line) already queued: report each once
        self.proactive_lines: list = []   # (epoch seconds, line) Jarvis said aloud on its own, awaiting the next turn's note
        self.last_scan: dict = {}

    # ---- the decision ------------------------------------------------------

    def decide(self, c: Candidate, ctx: Context | None = None) -> str:
        ctx = ctx or Context()
        if c.kind in NEVER:
            return DROP
        if c.provenance == "model" and c.kind not in (FG_ANSWER, FG_ERROR):
            return DROP                   # model words about background work: never
        if c.kind in (FG_ANSWER, FG_ERROR):
            return SPEAK
        if c.kind == BG_STOP_CONFIRM:
            return SPEAK
        if c.kind == BG_COMPLETION:
            if c.announce and not (ctx.fg_active or ctx.mouth_busy
                                   or ctx.zipped or not ctx.conversation_active):
                return SPEAK
            return DEFER
        if c.kind == ACTION_BAD_NEWS:
            return DEFER
        return DROP                       # unknown: fail closed

    # ---- background outcomes -----------------------------------------------

    def on_task_terminal(self, task_row, ctx: Context | None = None):
        """Called when a background task finishes. Returns the line to speak
        now, or None (it was deferred for the next turn's note, or dropped)."""
        c = Candidate(BG_COMPLETION, "code", task_row["id"],
                      announce=bool(task_row.get("announce")))
        d = self.decide(c, ctx)
        if d == SPEAK:
            return task_line(task_row)
        if d == DEFER:
            self._defer(c)
        return None

    def on_action_bad_news(self, action_row):
        line = action_line(action_row)
        if line and (action_row["id"], line) not in self._bad_news:
            self._bad_news.add((action_row["id"], line))
            self._defer(Candidate(ACTION_BAD_NEWS, "code", action_row["id"]))

    # ---- outcomes whose hand-off was never recorded (F-09) -----------------------

    @staticmethod
    def _outcome_at(row, verified_at):
        """When this row reached its present outcome, from durable ledger timestamps only: the latest
        transition into the current state, or a later verdict. None if the ledger holds neither."""
        ts = [h[0] for h in row.get("history") or [] if len(h) >= 2 and h[1] == row.get("state")]
        if verified_at:
            ts.append(verified_at)
        return max(ts) if ts else None

    def scan_undelivered(self, now=None, window_s=SCAN_WINDOW_S, max_rows=SCAN_MAX_ROWS, limit=SCAN_LIMIT) -> int:
        """Run after the ledger is reconciled and an epoch is open. First rebuild the in-memory dedupe
        state from the durable delivery records, then queue (into `deferred`, the same channel a live
        outcome uses) every eligible outcome of the window that no record shows as handed off. Returns how
        many were queued. A background task is eligible in any terminal state but a user stop; an action
        when it has bad news to report. A change in a delivered task's VERIFICATION is not a new outcome
        here: the existing update note covers it, once, through the restored `_surfaced`."""
        if self.ledger is None:
            return 0
        led = self.ledger
        now = led.now() if now is None else now
        cutoff = now - window_s
        examined, found = 0, []
        queued_ids = {(c.kind, c.source_id) for c in self.deferred}
        for tid, row in reversed(list(_tail(led.tasks, max_rows))):
            examined += 1
            ds = row.get("deliveries") or []
            if ds and (ds[-1].get("at") or 0) >= cutoff and row.get("task_kind") == L.KIND_BG_AGENT:
                self._surfaced.setdefault(tid, ds[-1].get("verification"))
            if (row.get("task_kind") != L.KIND_BG_AGENT or row.get("state") not in L.TASK_TERMINAL
                    or row.get("state") == L.K_STOPPED_USER or led.has_delivery("task", tid, outcome=row.get("state"))
                    or (not ds and row.get("user_visible"))          # handed off before deliveries were recorded
                    or (BG_COMPLETION, tid) in queued_ids):
                continue
            at = self._outcome_at(row, (row.get("result_verification") or {}).get("verified_at"))
            if at is not None and now - at <= window_s:
                found.append((at, Candidate(BG_COMPLETION, "code", tid)))
        for aid, row in reversed(list(_tail(led.actions, max_rows))):
            examined += 1
            line = action_line(row)
            if not line:
                continue
            if led.has_delivery("action", aid) or (not row.get("deliveries") and row.get("user_visible")):
                self._bad_news.add((aid, line))
                continue
            if (aid, line) in self._bad_news or (ACTION_BAD_NEWS, aid) in queued_ids:
                continue
            at = self._outcome_at(row, (row.get("verification") or {}).get("verified_at"))
            if at is not None and now - at <= window_s:
                found.append((at, Candidate(ACTION_BAD_NEWS, "code", aid)))
                self._bad_news.add((aid, line))
        found.sort(key=lambda x: x[0], reverse=True)
        keep = sorted(found[:limit], key=lambda x: x[0])
        for _, c in keep:
            self._defer(c)
        self.last_scan = {"examined": examined, "eligible": len(found), "queued": len(keep)}
        return len(keep)

    # ---- the proactive speaker (backtalk/proactive_loop.py) --------------------------------------------

    def discard(self, kind, source_id) -> bool:
        """Drop the deferred candidate for (kind, source_id) once it has been spoken. True if one was there."""
        for i, c in enumerate(self.deferred):
            if c.kind == kind and c.source_id == source_id:
                del self.deferred[i]
                return True
        return False

    def note_proactive(self, line, at, keep=3):
        """Remember a line Jarvis said aloud WITHOUT being asked, so the next foreground turn can tell the model it
        happened ("ok thanks" is otherwise an answer to nothing). Bounded; the line is built by code."""
        self.proactive_lines.append((at, sanitize(line, 240)))
        del self.proactive_lines[:-keep]

    def _defer(self, c):
        if len(self.deferred) >= self.max_deferred:
            self.deferred.pop(0)
        self.deferred.append(c)

    def take_notes(self, limit=3):
        """Short factual lines for the next foreground turn, built from the
        ledger. Taking them is what 'surfaces' them; each is marked once."""
        lines, keep = [], []
        for c in self.deferred:
            if len(lines) >= limit:
                keep.append(c)
                continue
            line = None
            if self.ledger is not None:
                if c.kind == BG_COMPLETION and c.source_id in self.ledger.tasks:
                    row = self.ledger.tasks[c.source_id]
                    line = task_line(row) + verification_tag(row) + report_pointer(row)
                    self._surfaced[c.source_id] = _vstate(row)
                    self.ledger.mark_spoken("task", c.source_id, "next_turn_note", "model_note")
                elif c.kind == ACTION_BAD_NEWS and c.source_id in self.ledger.actions:
                    line = action_line(self.ledger.actions[c.source_id])
                    self.ledger.mark_spoken("action", c.source_id, "next_turn_note", "model_note")
            if line:
                lines.append(line)
        self.deferred = keep
        while self.proactive_lines and len(lines) < limit:
            at, spoken = self.proactive_lines.pop(0)
            lines.append(f"You just said this aloud on your own, unprompted: {spoken}")
        # A task whose check moved on AFTER its note was handed over (evidence or
        # a summary that arrived late): say so once, in the same evidence-derived
        # words, so the model is never left holding the earlier, weaker state.
        if self.ledger is not None:
            for tid, was in list(self._surfaced.items()):
                row = self.ledger.tasks.get(tid)
                now = _vstate(row) if row else was
                if len(lines) >= limit or now == was or now in (None, L.V_NOT_REQUIRED, L.V_PENDING):
                    continue
                lines.append(f"Update on background task {_name(row)}:" + _result_clause(row)
                             + verification_tag(row) + report_pointer(row))
                self._surfaced[tid] = now
                self.ledger.mark_spoken("task", tid, "next_turn_update", "model_note")
        return lines

    def note_prefix(self, limit=3) -> str:
        """The bracketed note prepended to the next foreground turn (the same
        pattern as the holo-board note). Empty when there is nothing."""
        lines = self.take_notes(limit)
        if not lines:
            return ""
        return "[System note, from the task ledger, not from the user: " + " ".join(lines) + "]"
