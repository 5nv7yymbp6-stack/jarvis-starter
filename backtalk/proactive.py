"""Proactive decision engine (roadmap Phase F, items 2, 3 and 5). PURE: no speech, no model, no file, no network, no
wall clock (`now` is always passed in). It decides WHAT SHOULD HAPPEN to an event and remembers what it already
decided; the loop in main.py is the only thing that talks, and it talks only through the existing sink.

    Engine(policy).offer(event)            put an event in the bounded, deduplicated queue
    Engine.tick(ctx, now) -> [Decision]    re-evaluate the queue; at most one NOTIFY/ACT per tick; WAIT stays queued
    Engine.delivered(key, now)             the caller handed the announcement to the mouth: dedupe + rate accounting
    Engine.requeue(key, now)               the mouth flushed it before it was heard: put it back, bounded attempts

OUTCOMES (every decision carries a reason code and ids only; never message text):
    IGNORE  drop it: engine disabled, expired, or a duplicate of something already handled
    QUEUE   not worth interrupting for; it is left to the existing next-turn note path and leaves this queue
    WAIT    wanted, but not now (the user is speaking, Jarvis is busy, zipped, quiet hours); stays queued
    NOTIFY  say it now, through the one authorized speech path
    ACT     perform an allowlisted, NON-consequential action (default allowlist: empty, so ACT never happens by default)

RULES, in this order (the first that applies decides):
    1 engine disabled -> IGNORE            2 expired -> IGNORE (never for reminders / missed reminders: the user asked for those)
    3 duplicate key still remembered -> IGNORE
    4 an event that asks for an action: only an action in policy.act_allow that is not in CONSEQUENTIAL becomes ACT;
      anything else is NOT performed (it is announced instead, with reason `act_not_authorized`) - fail closed
    5 low severity -> QUEUE (never spoken unprompted)
    6 the USER is speaking (key held, capture in progress, or speech within quiet_after_user_s) -> WAIT, for EVERY severity
    7 brain not ready or just started/woken (settling) -> WAIT; zipped -> WAIT for every severity; on a likely CALL (the front app
      is a call app, from backtalk/presence.py) -> WAIT, reminders included, unless urgent AND policy.urgent_on_call; nobody at the
      machine (locked, display off, away) -> WAIT for anything that is not a user-requested reminder or urgent (a reminder you asked
      for still speaks: idle is not away); anything that is not a user-requested reminder also needs an active conversation (the
      existing speech rule); quiet hours -> WAIT unless urgent
    8 Jarvis busy (mouth busy or a foreground turn in flight) -> WAIT, unless severity is urgent AND
      policy.urgent == "speak_over_jarvis" (explicit opt-in; the default is "off")
    9 per-kind hourly cap (not for reminders or urgent) -> QUEUE
    10 otherwise -> NOTIFY
Urgent never means "over the user": rule 6 has no exception.
"""
import collections
import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

ACT, NOTIFY, QUEUE, WAIT, IGNORE = "ACT", "NOTIFY", "QUEUE", "WAIT", "IGNORE"
OUTCOMES = (ACT, NOTIFY, QUEUE, WAIT, IGNORE)
LOW, NORMAL, URGENT = "low", "normal", "urgent"
SEVERITIES = (LOW, NORMAL, URGENT)
KINDS = ("reminder", "bg_completion", "bg_failure", "action_bad_news", "overdue")
CONSEQUENTIAL = frozenset({"send", "spend", "purchase", "pay", "delete", "remove", "publish", "post", "email",
                           "message", "call", "transfer", "external_setting", "install", "deploy"})
USER_REQUESTED = ("reminder", "overdue")       # things the user asked for: never dropped as stale, never gated on a conversation
KEY_MAX = 160
QUEUE_MAX = 100
MAX_REQUEUES = 2
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)\Z")


@dataclass(frozen=True)
class Event:
    kind: str
    key: str                              # idempotency key: the same real-world thing always has the same key
    severity: str = NORMAL
    at: float = 0.0                       # when it happened (epoch seconds)
    expires_at: object = None
    source_id: str = ""                   # task id / reminder run id: ids only, never text
    relevance: float = 1.0                # 0..1 against the user's active goals (the caller supplies it)
    action: object = None                 # an action kind the event asks to be performed, or None
    verified: object = None               # evidence state of the thing being announced: VERIFIED | UNVERIFIED | ...


@dataclass(frozen=True)
class Context:
    user_speaking: bool = False           # key held, or a capture in progress
    since_user_speech_s: object = None    # seconds since the user last spoke, or None
    mouth_busy: bool = False
    fg_active: bool = False
    zipped: bool = False
    on_call: bool = False                 # presence: the frontmost app is a call app (a likely call, never a fact)
    user_away: bool = False               # presence: locked, display asleep or idle for minutes
    camera_busy: bool = False             # presence: some camera is running for some app (a browser video call, or the Bear Hands board)
    screen_busy: bool = False             # presence: the frontmost window is full-screen (a movie, a presentation)
    conversation_active: bool = True      # the addressee gate is ACTIVE (always true in push-to-talk)
    brain_ready: bool = True              # no rebuild in flight, a router exists
    settling: bool = False                # just started or just woke from sleep: devices are not ready yet


@dataclass(frozen=True)
class Policy:
    enabled: bool = False                 # DEFAULT OFF
    urgent: str = "off"                   # "off" | "speak_over_jarvis"
    act_allow: tuple = ()                 # action kinds that may be performed automatically; default none
    dedupe_ttl_s: float = 24 * 3600.0
    min_relevance: float = 0.2
    quiet_after_user_s: float = 2.0
    max_per_hour: int = 6                 # per kind, not for reminders or urgent
    urgent_on_call: bool = False          # explicit opt-in: an urgent line may speak during a likely call (default: it waits too)
    quiet_hours: object = None            # ("23:00", "07:00") or None
    tz: str = "America/Chicago"
    max_per_tick: int = 1

    @staticmethod
    def from_config(cfg):
        """Policy from a config dict (CFG.get("proactive")). Anything malformed means the safe value; only literal
        true enables."""
        try:
            c = cfg if isinstance(cfg, dict) else {}
            qh = c.get("quiet_hours")
            if not (isinstance(qh, (list, tuple)) and len(qh) == 2 and all(isinstance(x, str) and _HHMM.match(x) for x in qh)):
                qh = None
            allow = tuple(str(a).lower() for a in (c.get("act_allow") or ()) if isinstance(a, str))
            num = lambda k, d, lo, hi: (c[k] if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)
                                        and lo <= c[k] <= hi else d)
            return Policy(enabled=c.get("enabled") is True,
                          urgent="speak_over_jarvis" if c.get("urgent") == "speak_over_jarvis" else "off",
                          act_allow=tuple(a for a in allow if a not in CONSEQUENTIAL),
                          dedupe_ttl_s=float(num("dedupe_ttl_s", 24 * 3600.0, 60, 30 * 86400)),
                          min_relevance=float(num("min_relevance", 0.2, 0, 1)),
                          quiet_after_user_s=float(num("quiet_after_user_s", 2.0, 0, 60)),
                          max_per_hour=int(num("max_per_hour", 6, 1, 100)),
                          urgent_on_call=c.get("urgent_on_call") is True,
                          quiet_hours=tuple(qh) if qh else None,
                          tz=c.get("tz") if isinstance(c.get("tz"), str) else "America/Chicago")
        except Exception:
            return Policy()


@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str
    key: str
    kind: str
    severity: str
    at: float
    attempts: int = 0
    flags: tuple = ()

    def line(self) -> str:
        """The single log line for this decision: codes and the key only."""
        return (f"[proactive] decision outcome={self.outcome} reason={self.reason} kind={self.kind} "
                f"sev={self.severity} key={self.key[:KEY_MAX]}"
                + (f" flags={','.join(self.flags)}" if self.flags else "")
                + (f" attempts={self.attempts}" if self.attempts else ""))


@dataclass
class _Pending:
    event: Event
    first_seen: float
    attempts: int = 0


def _in_quiet_hours(policy, now) -> bool:
    if not policy.quiet_hours:
        return False
    try:
        t = datetime.fromtimestamp(now, ZoneInfo(policy.tz))
        mins = t.hour * 60 + t.minute
        a, b = (int(x[:2]) * 60 + int(x[3:]) for x in policy.quiet_hours)
    except Exception:
        return False
    return (a <= mins < b) if a <= b else (mins >= a or mins < b)


class Engine:
    def __init__(self, policy=None):
        self.policy = policy or Policy()
        self._pending = collections.OrderedDict()      # key -> _Pending
        self._seen = {}                                 # key -> remembered until (epoch seconds)
        self._sent = collections.defaultdict(collections.deque)   # kind -> delivery times (rate accounting)
        self._requeues = collections.Counter()          # key -> how many times a flushed line was put back
        self.stats = collections.Counter()

    # -- the rules ----------------------------------------------------------------

    def evaluate(self, e: Event, ctx: Context, now: float, attempts=0) -> Decision:
        p = self.policy

        def d(outcome, reason, flags=()):
            return Decision(outcome, reason, e.key[:KEY_MAX], e.kind, e.severity, now, attempts, tuple(flags))

        if not p.enabled:
            return d(IGNORE, "disabled")
        if e.kind not in KINDS or e.severity not in SEVERITIES or not e.key:
            return d(IGNORE, "malformed")
        if e.expires_at is not None and now > e.expires_at and e.kind not in USER_REQUESTED:
            return d(IGNORE, "expired")
        if self._seen.get(e.key, 0) > now:
            return d(IGNORE, "duplicate")
        flags = []
        if e.verified in ("UNVERIFIED", "VERIFY_FAILED", "FAILED"):
            flags.append("unverified_evidence")         # informational: the sentence builder already withholds the numbers
        if e.action is not None:
            act = str(e.action).lower()
            if act in p.act_allow and act not in CONSEQUENTIAL:
                return d(ACT, "act_allowed")
            flags.append("act_not_authorized")          # never performed; it is only announced (or waits) below
        if e.severity == LOW or (e.relevance < p.min_relevance and e.severity != URGENT):
            return d(QUEUE, "low_value", flags)
        if ctx.user_speaking or (ctx.since_user_speech_s is not None and ctx.since_user_speech_s < p.quiet_after_user_s):
            return d(WAIT, "user_speaking", flags)
        if not ctx.brain_ready or ctx.settling:
            return d(WAIT, "not_ready", flags)
        if ctx.zipped:
            return d(WAIT, "zipped", flags)
        if ctx.on_call and not (e.severity == URGENT and p.urgent_on_call):
            return d(WAIT, "on_call", flags)
        if ctx.user_away and e.kind not in USER_REQUESTED and e.severity != URGENT:
            return d(WAIT, "user_away", flags)
        if ctx.screen_busy and e.kind not in USER_REQUESTED and e.severity != URGENT:
            return d(WAIT, "screen_busy", flags)     # background news only; reminders and urgent lines are never held for this
        if ctx.camera_busy and e.kind not in USER_REQUESTED and e.severity != URGENT:
            return d(WAIT, "camera_busy", flags)     # background news only: a hidden board tab also runs the camera, so reminders still speak
        if e.kind not in USER_REQUESTED and not ctx.conversation_active:
            return d(WAIT, "not_in_conversation", flags)
        if e.severity != URGENT and _in_quiet_hours(p, now):
            return d(WAIT, "quiet_hours", flags)
        if ctx.mouth_busy or ctx.fg_active:
            if e.severity == URGENT and p.urgent == "speak_over_jarvis":
                return d(NOTIFY, "urgent_over_jarvis", flags)
            return d(WAIT, "jarvis_busy", flags)
        if e.kind not in USER_REQUESTED and e.severity != URGENT:
            recent = [t for t in self._sent[e.kind] if now - t < 3600]
            if len(recent) >= p.max_per_hour:
                return d(QUEUE, "rate_limited", flags)
        return d(NOTIFY, "clear", flags)

    # -- the queue ----------------------------------------------------------------

    def offer(self, e: Event, now: float) -> bool:
        """Queue an event unless that key is already remembered or queued. True if it was added."""
        if not self.policy.enabled or not e.key or e.key in self._pending or self._seen.get(e.key, 0) > now:
            self.stats["offer_duplicate"] += 1
            return False
        if len(self._pending) >= QUEUE_MAX:
            drop = min(self._pending.values(), key=lambda x: (SEVERITIES.index(x.event.severity), x.first_seen))
            del self._pending[drop.event.key]
            self.stats["queue_overflow_drop"] += 1
        self._pending[e.key] = _Pending(e, now)
        return True

    def tick(self, ctx: Context, now: float):
        """Decisions that need the caller to act, at most `max_per_tick`. Urgent first, then oldest. Everything that
        is not NOTIFY/ACT is settled here: WAIT stays queued, IGNORE and QUEUE leave."""
        self._prune(now)
        out = []
        order = sorted(self._pending.values(), key=lambda x: (-SEVERITIES.index(x.event.severity), x.first_seen))
        for pend in order:
            dec = self.evaluate(pend.event, ctx, now, pend.attempts)
            self.stats[dec.outcome] += 1
            if dec.outcome == WAIT:
                continue
            if dec.outcome in (IGNORE, QUEUE):
                del self._pending[pend.event.key]
                if dec.outcome == IGNORE:
                    self._seen[pend.event.key] = now + self.policy.dedupe_ttl_s
                out.append(dec)                          # reported so the caller can log it; no action required
                continue
            if sum(1 for x in out if x.outcome in (NOTIFY, ACT)) >= self.policy.max_per_tick:
                continue                                  # one announcement per tick; the rest stay queued
            out.append(dec)
        return out

    def waiting(self, ctx: Context, now: float):
        """[(key, reason)] for everything queued that would WAIT right now. For logging; changes nothing."""
        out = []
        for key, pend in self._pending.items():
            dec = self.evaluate(pend.event, ctx, now, pend.attempts)
            if dec.outcome == WAIT:
                out.append((key, dec.reason))
        return out

    def event_for(self, key):
        p = self._pending.get(key)
        return p.event if p else None

    def delivered(self, key, now):
        """The announcement was handed to the mouth (handed off, not necessarily heard). Dedupe and rate accounting."""
        p = self._pending.pop(key, None)
        self._seen[key] = now + self.policy.dedupe_ttl_s
        kind = p.event.kind if p else None
        if kind:
            q = self._sent[kind]
            q.append(now)
            while q and now - q[0] >= 3600:
                q.popleft()
        self.stats["delivered"] += 1

    def requeue(self, key, event, now) -> bool:
        """The mouth flushed a handed-off line before it was heard: forget the delivery and queue it again, a bounded
        number of times. False when the bound is spent (the caller leaves it to the next-turn note path)."""
        prior = self._seen.pop(key, None)
        n = self._requeues
        n[key] += 1
        if n[key] > MAX_REQUEUES:
            if prior is not None:
                self._seen[key] = prior
            self.stats["requeue_exhausted"] += 1
            return False
        self._pending[key] = _Pending(event, now, attempts=n[key])
        self.stats["requeued"] += 1
        return True

    def forget(self, key):
        """Drop everything remembered about a key (delivered, requeue count, queued). For a reminder whose line could not
        be heard and whose scheduler retry is about to come back under the same fire key."""
        self._seen.pop(key, None)
        self._requeues.pop(key, None)
        self._pending.pop(key, None)

    def pending_keys(self):
        return list(self._pending)

    def _prune(self, now):
        for k in [k for k, until in self._seen.items() if until <= now]:
            del self._seen[k]
