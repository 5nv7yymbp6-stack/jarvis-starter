"""The proactive runner (roadmap Phase F): the ONE place the scheduler, the decision engine, the ledger and the mouth meet.

It owns no policy of its own. `step(now)` does one pass, and every outward effect goes through an injected callable, so the
whole thing runs headless in tests with a fake mouth and a fake clock:

    1 ingest reminder requests dropped in `<scheduler dir>/inbox/` by the remind CLI (the voice process holds the scheduler
      lock, so the CLI never writes the scheduler itself)
    2 claim what the scheduler says is due (a missed one-time reminder is reported, not dropped)
    3 mirror the ledger's deferred background outcomes as Events (they STAY in `SpeechPolicy.deferred` until actually heard)
    4 resolve lines already handed to the mouth: heard -> mark delivered; flushed by a user action -> put back (bounded)
    5 ask the engine what to do; for NOTIFY, build the sentence ONLY from code (speech_policy.task_line / action_line, or
      "Reminder: <the user's own words, sanitized>") and hand it to the single injected sink

NEVER: calls the addressee gate (agent speech must not reopen or extend the listening window), speaks model-authored text,
reads `pending_reports`, performs an action (no ACT runner exists), or marks something delivered before it was heard.
Delivery is AT-LEAST-ONCE with dedupe on the scheduler's `fire_key` / the ledger's delivery record, bounded by MAX_REQUEUES.
A failure of any one step is logged and never stops the others or the loop.
"""
import dataclasses
import json
import os
import time

from backtalk import ledger as L
from backtalk import proactive as P
from backtalk import speech_policy as SP

INBOX = "inbox"
TICK_S = 2.0                       # a short fixed tick re-reading the WALL clock: sleep/wake safe (asyncio.sleep pauses with the Mac)
STARTUP_QUIET_S = 20.0             # nothing is spoken just after start: the greeting and the audio devices come first
WAKE_QUIET_S = 8.0                 # ...and just after the Mac wakes
WAKE_JUMP_S = 5.0                  # wall clock minus monotonic clock moved by this much: the machine slept
INFLIGHT_TIMEOUT_S = 90.0          # a line neither heard nor flushed after this long is treated as flushed
HEARTBEAT_S = 300.0
REQUEST_MAX_BYTES = 4096
_NOW_KINDS = {L.K_FAILED: "bg_failure"}


def inbox_dir(sched_dir):
    return os.path.join(sched_dir, INBOX)


def _json_request(path):
    with open(path, "rb") as fh:
        raw = fh.read(REQUEST_MAX_BYTES + 1)
    if len(raw) > REQUEST_MAX_BYTES:
        raise ValueError("too big")
    req = json.loads(raw)
    if not isinstance(req, dict) or req.get("v") != 1 or not isinstance(req.get("id"), str) or not req["id"]:
        raise ValueError("not a request")
    return req


class Runner:
    def __init__(self, *, engine, scheduler, speech, ledger, say, mouth_state, context, log, clock=time.time,
                 monotonic=time.monotonic, sched_dir=None):
        self.engine, self.sched, self.speech, self.ledger = engine, scheduler, speech, ledger
        self._say, self._mouth, self._context, self._log = say, mouth_state, context, log
        self._clock, self._mono = clock, monotonic
        self.sched_dir = sched_dir
        self.started = clock()
        self._drift = clock() - monotonic()
        self._quiet_until = self.started + STARTUP_QUIET_S
        self._fires = {}                 # event key -> scheduler Fire (reminders)
        self._inflight = {}              # event key -> dict(gen, at, text, event, seen_busy)
        self._wait_logged = {}           # event key -> last logged WAIT reason
        self._last_beat = clock()
        self.errors = 0

    # -- helpers ---------------------------------------------------------------

    def _settling(self, now):
        drift = now - self._mono()
        if abs(drift - self._drift) > WAKE_JUMP_S:
            self._drift = drift
            self._quiet_until = max(self._quiet_until, now + WAKE_QUIET_S)
            self._log("[proactive] wake detected: holding speech briefly")
        return now < self._quiet_until

    def _guard(self, what, fn, *a):
        try:
            return fn(*a)
        except Exception as e:           # one failing step never stops the others
            self.errors += 1
            self._log(f"[proactive] step failed what={what} error={e.__class__.__name__}")
            return None

    # -- 1: inbox --------------------------------------------------------------

    def ingest_inbox(self, now):
        if not self.sched_dir:
            return
        d = inbox_dir(self.sched_dir)
        if not os.path.isdir(d):
            return
        for name in sorted(n for n in os.listdir(d) if n.endswith(".json")):
            path = os.path.join(d, name)
            outcome = "done"
            try:
                req = _json_request(path)
                self._apply_request(req)
            except Exception as e:
                outcome = "rejected"
                self._log(f"[proactive] request rejected file={name[:60]} error={e.__class__.__name__}")
            try:
                dest = os.path.join(d, outcome)
                os.makedirs(dest, mode=0o700, exist_ok=True)
                os.replace(path, os.path.join(dest, name))
            except OSError:
                self._log("[proactive] could not archive a request file")

    def _apply_request(self, req):
        op, key = req.get("op"), f"inbox:{req['id']}"
        if op == "cancel":
            tid = req.get("task_id")
            if isinstance(tid, str) and self.sched.get(tid) and self.sched.cancel(tid):
                self._log(f"[proactive] reminder cancelled task={tid}")
            return
        if op != "add":
            raise ValueError("unknown op")
        if self.sched.has_key(key):
            return                       # a re-read of a request already applied (crash between apply and archive)
        title = req.get("title")
        sev = req.get("severity") if req.get("severity") in (P.NORMAL, P.URGENT) else P.NORMAL
        kind, payload = req.get("kind"), {"severity": sev}
        if kind == "once":
            t = self.sched.add_once(title, req.get("due_at"), key=key, payload=payload)
        elif kind == "every":
            t = self.sched.add_every(title, req.get("interval_s"), key=key, payload=payload)
        elif kind == "daily":
            t = self.sched.add_daily(title, req.get("hhmm"), tz=req.get("tz") or "America/Chicago", key=key, payload=payload)
        else:
            raise ValueError("unknown kind")
        self._log(f"[proactive] reminder scheduled task={t.id} kind={t.kind} sev={sev}")

    # -- 2 and 3: events in ----------------------------------------------------

    def collect_due(self, now):
        for f in self.sched.due(now):
            sev = f.payload.get("severity") if f.payload.get("severity") in (P.NORMAL, P.URGENT) else P.NORMAL
            if f.missed:
                ev = P.Event("overdue", f"{f.fire_key}:missed", sev, now, source_id=f.task_id)
            else:
                ev = P.Event("reminder", f.fire_key, sev, f.slot, source_id=f.task_id)
            self._fires[ev.key] = f
            if not self.engine.offer(ev, now):
                self._settle_fire(ev.key, ok=True, why="duplicate")

    def collect_deferred(self, now):
        for c in list(self.speech.deferred):
            row = None
            if c.kind == SP.BG_COMPLETION:
                row = self.ledger.tasks.get(c.source_id)
                if row is None or self.ledger.has_delivery("task", c.source_id, outcome=row.get("state")):
                    continue
                kind = _NOW_KINDS.get(row.get("state"), "bg_completion")
                key = f"task:{c.source_id}:{self.ledger.delivery_fingerprint('task', row)}"
                verified = (row.get("result_verification") or {}).get("state")
            elif c.kind == SP.ACTION_BAD_NEWS:
                row = self.ledger.actions.get(c.source_id)
                if row is None or self.ledger.has_delivery("action", c.source_id):
                    continue
                kind, verified = "action_bad_news", (row.get("verification") or {}).get("state")
                key = f"action:{c.source_id}:{self.ledger.delivery_fingerprint('action', row)}"
            else:
                continue
            self.engine.offer(P.Event(kind, key, P.NORMAL, now, source_id=c.source_id, verified=verified), now)

    # -- text ------------------------------------------------------------------

    def line_for(self, ev):
        """The sentence for an event, built from code only. None means there is nothing safe to say."""
        if ev.kind in ("reminder", "overdue"):
            f = self._fires.get(ev.key)
            title = SP.sanitize(f.title, 160) if f else ""
            if not title:
                return None
            return f"Reminder: {title}." if ev.kind == "reminder" else f"I couldn't tell you in time, so: reminder, {title}."
        if ev.kind in ("bg_completion", "bg_failure"):
            row = self.ledger.tasks.get(ev.source_id)
            return SP.task_line(row) if row else None
        if ev.kind == "action_bad_news":
            row = self.ledger.actions.get(ev.source_id)
            return SP.action_line(row) if row else None
        return None

    # -- 4: lines already with the mouth ---------------------------------------

    def resolve_inflight(self, now):
        busy, gen, recent = self._mouth()
        for key, st in list(self._inflight.items()):
            ev = st["event"]
            if gen != st["gen"]:                                  # a user action flushed the mouth before it was heard
                del self._inflight[key]
                self._flushed(key, ev, now)
                continue
            if busy:
                st["seen_busy"] = True
            started = st["seen_busy"] or any(st["text"][:40] in c for c in recent)
            if started and not busy:
                del self._inflight[key]
                self._heard(key, ev, st["text"], now)
            elif now - st["at"] > INFLIGHT_TIMEOUT_S:
                del self._inflight[key]
                self._flushed(key, ev, now)

    def _heard(self, key, ev, text, now):
        if ev.kind in ("reminder", "overdue"):
            self._settle_fire(key, ok=True, why="heard")
        else:
            kind = "action" if ev.kind == "action_bad_news" else "task"
            self.ledger.mark_spoken(kind, ev.source_id, "proactive", "spoken")
            self.speech.discard(SP.ACTION_BAD_NEWS if kind == "action" else SP.BG_COMPLETION, ev.source_id)
        self.speech.note_proactive(text, now)
        self._log(f"[proactive] heard kind={ev.kind} key={key[:100]}")

    def _flushed(self, key, ev, now):
        if self.engine.requeue(key, ev, now):
            self._log(f"[proactive] flushed before it was heard, queued again kind={ev.kind} key={key[:100]}")
        else:
            self._log(f"[proactive] flushed, requeue bound spent kind={ev.kind} key={key[:100]}")
            if ev.kind in ("reminder", "overdue"):
                self.engine.forget(key)                              # its retry returns under the same fire key and must be heard
                self._settle_fire(key, ok=False, why="flushed")      # the scheduler's bounded retry takes over

    def _settle_fire(self, key, *, ok, why):
        f = self._fires.pop(key, None)
        if f is None or f.missed:
            return
        (self.sched.complete if ok else self.sched.fail)(*((f.run_id,) if ok else (f.run_id, why)))

    # -- 5: decide and speak ---------------------------------------------------

    def act_on(self, now):
        ctx = dataclasses.replace(self._context(), settling=self._settling(now))
        for dec in self.engine.tick(ctx, now):
            ev = self.engine.event_for(dec.key)
            if dec.outcome == P.NOTIFY and ev is not None:
                self._notify(dec, ev, now)
            elif dec.outcome == P.ACT:
                self._log(dec.line() + " (no action runner exists: nothing was done)")
                self.engine.delivered(dec.key, now)
            else:                                    # QUEUE / IGNORE
                self._log(dec.line())
                if dec.outcome == P.IGNORE:
                    self._settle_fire(dec.key, ok=True, why="ignored")
        for key, why in self.engine.waiting(ctx, now):
            if self._wait_logged.get(key) != why:
                self._wait_logged[key] = why
                self._log(f"[proactive] waiting reason={why} key={key[:100]}")

    def _notify(self, dec, ev, now):
        if ev.kind in ("bg_completion", "bg_failure", "action_bad_news"):
            kind = "action" if ev.kind == "action_bad_news" else "task"
            row = (self.ledger.actions if kind == "action" else self.ledger.tasks).get(ev.source_id)
            if row is None or self.ledger.has_delivery(kind, ev.source_id, **({} if kind == "action" else {"outcome": row.get("state")})):
                self.engine.delivered(dec.key, now)          # the model was already handed it, or it is gone: do not repeat it
                self.speech.discard(SP.ACTION_BAD_NEWS if kind == "action" else SP.BG_COMPLETION, ev.source_id)
                self._log(f"[proactive] skipped, already handed off kind={ev.kind} key={dec.key[:100]}")
                return
        text = self.line_for(ev)
        if not text:
            self.engine.delivered(dec.key, now)
            self._settle_fire(dec.key, ok=True, why="nothing_to_say")
            self._log(dec.line() + " (nothing safe to say)")
            return
        self._log(dec.line())
        self._say(text)
        _, gen, _ = self._mouth()
        self._inflight[dec.key] = {"gen": gen, "at": now, "text": text, "event": ev, "seen_busy": False}
        self.engine.delivered(dec.key, now)
        self._wait_logged.pop(dec.key, None)

    # -- the pass --------------------------------------------------------------

    def step(self, now=None):
        now = self._clock() if now is None else now
        self._guard("inbox", self.ingest_inbox, now)
        self._guard("due", self.collect_due, now)
        self._guard("deferred", self.collect_deferred, now)
        self._guard("resolve", self.resolve_inflight, now)
        self._guard("act", self.act_on, now)
        if now - self._last_beat >= HEARTBEAT_S:
            self._last_beat = now
            self._log(f"[proactive] alive pending={len(self.engine.pending_keys())} inflight={len(self._inflight)} "
                      f"scheduled={len(self.sched.tasks('scheduled'))} errors={self.errors}")

    async def run(self, sleep):
        """The loop. `sleep` is asyncio.sleep, injected. Never raises: the caller's done-callback restarts it once."""
        while True:
            self.step()
            await sleep(TICK_S)
