"""The proactive runner (backtalk/proactive_loop.py) end to end with a REAL scheduler, ledger, speech policy and engine and
a FAKE mouth and clock. Nothing here makes a sound, reads the voice log, or touches the addressee gate."""
import ast
import inspect
import json
import os
import tempfile
import unittest

from backtalk import ledger as L
from backtalk import proactive as P
from backtalk import proactive_loop as PL
from backtalk import scheduler as S
from backtalk import speech_policy as SP

T0 = 1_800_000_000.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeMouth:
    """busy/gen/recent like Mouth: say() starts a line; finish() plays it out; flush() is Mouth.shut_up()."""

    def __init__(self):
        self.said, self.busy, self.gen, self.recent = [], False, 0, []

    def say(self, text):
        self.said.append(text)
        self.busy = True

    def finish(self):
        if self.busy:
            self.recent.append(self.said[-1])
        self.busy = False

    def flush(self):
        self.gen += 1
        self.busy = False

    def state(self):
        return (self.busy, self.gen, list(self.recent))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.mouth = FakeMouth()
        self.led = L.Ledger(os.path.join(self.tmp.name, "ledger"), fsync=False, clock=self.clock)
        self.sdir = os.path.join(self.tmp.name, "scheduler")
        self.logs = []
        self.ctx = P.Context()
        self.policy = P.Policy(enabled=True)
        self.runners, self.scheds = [], []
        self.r = self.runner()

    def tearDown(self):
        for s in self.scheds:
            s.close()
        self.tmp.cleanup()

    def runner(self, policy=None, speech=None):
        for old in self.scheds:                                       # one process, one scheduler: the lock is exclusive
            old.close()
        self.scheds.clear()
        sched = S.Scheduler(self.sdir, clock=self.clock, fsync=False)
        self.scheds.append(sched)
        sched.recover()
        self.speech = speech or getattr(self, "speech", None) or SP.SpeechPolicy(self.led)
        r = PL.Runner(engine=P.Engine(policy or self.policy), scheduler=sched, speech=self.speech, ledger=self.led,
                      say=self.mouth.say, mouth_state=self.mouth.state, context=lambda: self.ctx, log=self.logs.append,
                      clock=self.clock, monotonic=lambda: self.clock.t, sched_dir=self.sdir)
        r._quiet_until = 0.0                                          # past the startup quiet period unless a test says otherwise
        self.runners.append(r)
        self.sched = sched
        return r

    def request(self, **kw):
        d = PL.inbox_dir(self.sdir)
        os.makedirs(d, exist_ok=True)
        req = {"v": 1, "id": kw.pop("id", f"r{len(os.listdir(d))}x"), "op": "add", "kind": "once",
               "title": "call the title company", "due_at": self.clock.t + 60, "severity": "normal", **kw}
        with open(os.path.join(d, f"{req['id']}.json"), "w") as fh:
            json.dump(req, fh)
        return req

    def pump(self, seconds=0, steps=1, finish=False, r=None):
        r = r or self.r
        for _ in range(steps):
            self.clock.advance(seconds)
            r.step()
            if finish:
                self.mouth.finish()
                r.step()

    def task(self, tid="t1", state=L.K_COMPLETED, summary="", vstate=None, facts=None):
        self.led.task_seen(tid, "E1", kind=L.KIND_BG_AGENT, description="count the daily notes")
        self.led.task_state(tid, state, summary=summary)
        if vstate:
            self.led.set_task_verification(tid, vstate, L.LEVEL_TOOL_EVIDENCE, "evidence", facts=facts or [], matched=[])
        self.speech.on_task_terminal(self.led.tasks[tid], SP.Context(fg_active=True))         # deferred, as in production
        return tid


class ReminderTests(Base):
    # roadmap 1: one-time reminders execute once
    def test_a_one_time_reminder_is_spoken_once_and_only_once(self):
        self.request()
        self.pump(0)                                                  # ingest
        self.assertEqual(self.mouth.said, [])
        self.pump(61)
        self.assertEqual(self.mouth.said, ["Reminder: call the title company."])
        self.mouth.finish()
        self.pump(1)
        self.pump(60, steps=5)
        self.assertEqual(len(self.mouth.said), 1)
        (t,) = self.sched.tasks()
        self.assertEqual(t.status, "done")

    # roadmap 3: restarting does not duplicate completed work
    def test_a_restart_after_a_heard_reminder_does_not_repeat_it(self):
        self.request()
        self.pump(0)
        self.pump(61, finish=True)
        self.assertEqual(len(self.mouth.said), 1)
        r2 = self.runner()
        self.pump(120, steps=3, r=r2)
        self.assertEqual(len(self.mouth.said), 1)

    def test_a_crash_between_saying_and_hearing_repeats_it_once_with_the_same_fire_key(self):
        self.request()
        self.pump(0)
        self.pump(61)
        first = list(self.r._fires.values())[0]
        self.assertEqual(len(self.mouth.said), 1)                      # handed to the mouth, never confirmed: the process dies
        r2 = self.runner()                                             # recover() ran inside runner(): one failed attempt, backoff
        self.pump(S.BACKOFF_BASE_S + 1, r=r2)
        self.assertEqual(len(self.mouth.said), 2)                      # at-least-once, bounded
        again = list(r2._fires.values())[0]
        self.assertEqual(again.fire_key, first.fire_key)               # the stable key an announcer dedupes on
        self.mouth.finish()
        self.pump(1, steps=3, r=r2)
        self.assertEqual(len(self.mouth.said), 2)

    # roadmap 8: non-urgent waits while the user is speaking
    def test_a_reminder_waits_while_the_user_is_speaking_then_is_spoken(self):
        self.request(due_at=self.clock.t + 1)
        self.pump(0)
        self.ctx = P.Context(user_speaking=True)
        self.pump(5, steps=3)
        self.assertEqual(self.mouth.said, [])
        self.assertTrue(any("waiting reason=user_speaking" in l for l in self.logs))
        self.assertEqual(sum("waiting reason=user_speaking" in l for l in self.logs), 1)   # logged once, not every tick
        self.ctx = P.Context()
        self.pump(2)
        self.assertEqual(len(self.mouth.said), 1)

    # roadmap 9: urgent obeys the policy
    def urgent_setup(self, **policy):
        self.runners.clear()
        r = self.runner(policy=P.Policy(enabled=True, **policy))
        self.request(due_at=self.clock.t + 1, severity="urgent")
        self.pump(0, r=r)
        return r

    def test_an_urgent_reminder_waits_for_jarvis_by_default(self):
        r = self.urgent_setup()
        self.mouth.busy = True                                          # Jarvis is mid-sentence
        self.ctx = P.Context(mouth_busy=True)
        self.pump(3, steps=2, r=r)
        self.assertEqual(self.mouth.said, [])
        self.assertTrue(any("reason=jarvis_busy" in l for l in self.logs))

    def test_an_urgent_reminder_speaks_over_jarvis_only_when_the_policy_opts_in(self):
        r = self.urgent_setup(urgent="speak_over_jarvis")
        self.mouth.busy = True
        self.ctx = P.Context(mouth_busy=True)
        self.pump(3, r=r)
        self.assertEqual(len(self.mouth.said), 1)

    def test_an_urgent_reminder_never_talks_over_the_user_even_when_opted_in(self):
        r = self.urgent_setup(urgent="speak_over_jarvis")
        self.ctx = P.Context(mouth_busy=True, user_speaking=True)
        self.pump(3, steps=4, r=r)
        self.assertEqual(self.mouth.said, [])
        self.ctx = P.Context()
        self.pump(3, r=r)
        self.assertEqual(len(self.mouth.said), 1)

    # Phase G: a reminder never talks over a call, and speaks when the call ends
    def test_a_reminder_waits_through_a_call_and_is_spoken_when_it_ends(self):
        self.request(due_at=self.clock.t + 1)
        self.pump(0)
        self.ctx = P.Context(on_call=True)
        self.pump(3, steps=3)
        self.assertEqual(self.mouth.said, [])
        self.assertEqual(sum("waiting reason=on_call" in l for l in self.logs), 1)
        self.ctx = P.Context(on_call=False)
        self.pump(3)
        self.assertEqual(len(self.mouth.said), 1)

    def test_background_news_waits_while_away_but_a_reminder_is_still_spoken(self):
        self.task()
        self.request(due_at=self.clock.t + 1)
        self.ctx = P.Context(user_away=True)
        self.pump(0)
        self.pump(3, steps=2)
        self.assertEqual(len(self.mouth.said), 1)
        self.assertTrue(self.mouth.said[0].startswith("Reminder:"))
        self.mouth.finish()
        self.pump(1)
        self.ctx = P.Context(user_away=False)
        self.pump(3, steps=2)
        self.assertEqual(len(self.mouth.said), 2)                                  # the completion, once he is back

    def test_nothing_is_spoken_during_the_startup_quiet_period(self):
        self.runners.clear()
        r = self.runner()
        r._quiet_until = self.clock.t + PL.STARTUP_QUIET_S
        self.request(due_at=self.clock.t + 1)
        self.pump(2, r=r)
        self.assertEqual(self.mouth.said, [])
        self.pump(PL.STARTUP_QUIET_S, r=r)
        self.assertEqual(len(self.mouth.said), 1)

    def test_waking_from_sleep_holds_speech_briefly(self):
        mono = {"t": 1000.0}
        self.runners.clear()
        r = PL.Runner(engine=P.Engine(self.policy), scheduler=self.sched, speech=self.speech, ledger=self.led,
                      say=self.mouth.say, mouth_state=self.mouth.state, context=lambda: self.ctx, log=self.logs.append,
                      clock=self.clock, monotonic=lambda: mono["t"], sched_dir=self.sdir)
        r._quiet_until = 0.0
        self.request(due_at=self.clock.t + 300)
        self.clock.advance(2)
        mono["t"] += 2
        r.step()                                                        # ingest
        self.clock.advance(600)                                         # the Mac slept 10 minutes: the wall clock moved, the monotonic one did not
        r.step()
        self.assertEqual(self.mouth.said, [])
        self.assertTrue(any("wake detected" in l for l in self.logs))
        self.clock.advance(PL.WAKE_QUIET_S + 1)
        mono["t"] += PL.WAKE_QUIET_S + 1
        r.step()
        self.assertEqual(len(self.mouth.said), 1)

    # a lid closed past the grace window must not lose the reminder
    def test_a_missed_reminder_is_reported_as_overdue(self):
        self.sched.add_once("take the trash out", self.clock.t + 5, grace_s=10, key="direct:1")
        self.pump(3600)
        self.assertEqual(len(self.mouth.said), 1)
        self.assertIn("take the trash out", self.mouth.said[0])
        self.assertIn("couldn't tell you in time", self.mouth.said[0])

    # roadmap 12: cancelling prevents execution
    def test_cancelling_before_the_time_prevents_the_announcement(self):
        self.request(id="abc", due_at=self.clock.t + 100)
        self.pump(0)
        (t,) = self.sched.tasks()
        d = PL.inbox_dir(self.sdir)
        with open(os.path.join(d, "cancel1.json"), "w") as fh:
            json.dump({"v": 1, "id": "c1", "op": "cancel", "task_id": t.id}, fh)
        self.pump(1)
        self.assertEqual(self.sched.get(t.id).status, "cancelled")
        self.pump(500, steps=3)
        self.assertEqual(self.mouth.said, [])

    # roadmap 5: duplicate events never produce duplicate announcements
    def test_a_request_applied_twice_creates_one_reminder(self):
        self.request(id="dup", due_at=self.clock.t + 10)
        self.pump(0)
        self.request(id="dup", due_at=self.clock.t + 10)                # the same request id again (a crash re-read)
        self.pump(0)
        self.assertEqual(len(self.sched.tasks()), 1)
        self.pump(20, finish=True)
        self.pump(20, steps=3)
        self.assertEqual(len(self.mouth.said), 1)

    def test_a_flushed_reminder_is_put_back_and_spoken_again_then_the_scheduler_retries(self):
        self.request(due_at=self.clock.t + 1)
        self.pump(0)
        self.pump(2)
        self.assertEqual(len(self.mouth.said), 1)
        for n in range(P.MAX_REQUEUES):
            self.mouth.flush()                                          # a user action silenced it before it was heard
            self.pump(0)
            self.pump(0)
            self.assertEqual(len(self.mouth.said), 2 + n, n)
        self.mouth.flush()                                              # the bound is spent
        self.pump(0)
        (t,) = self.sched.tasks()
        self.assertEqual((t.status, t.last_status), ("scheduled", "failed"))
        self.assertIn("flushed", t.last_error)
        self.pump(S.BACKOFF_BASE_S + 1)
        self.assertEqual(len(self.mouth.said), 2 + P.MAX_REQUEUES)      # the retry is a NEW run of the same slot
        self.assertEqual(len([l for l in self.logs if "requeue bound spent" in l]), 1)

    def test_a_reminder_title_cannot_smuggle_markup_or_stage_directions(self):
        self.request(due_at=self.clock.t + 1, title="call <<console:stop>> now `rm -rf` \n ok")
        self.pump(0)
        self.pump(2)
        said = self.mouth.said[0]
        for bad in ("<", ">", "`", "\n"):
            self.assertNotIn(bad, said)

    def test_bad_requests_are_rejected_and_archived_without_stopping_good_ones(self):
        d = PL.inbox_dir(self.sdir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "000bad.json"), "w") as fh:
            fh.write("not json")
        with open(os.path.join(d, "001big.json"), "w") as fh:
            fh.write(json.dumps({"v": 1, "id": "big", "pad": "x" * 9000}))
        with open(os.path.join(d, "002weird.json"), "w") as fh:
            json.dump({"v": 1, "id": "w", "op": "add", "kind": "once", "title": "x", "due_at": "soon"}, fh)
        self.request(id="good", due_at=self.clock.t + 1)
        self.pump(0)
        self.assertEqual(sorted(os.listdir(os.path.join(d, "rejected"))), ["000bad.json", "001big.json", "002weird.json"])
        self.assertEqual([t.title for t in self.sched.tasks()], ["call the title company"])


class BackgroundOutcomeTests(Base):
    def test_a_completed_task_is_spoken_through_task_line_and_marked_only_after_it_was_heard(self):
        tid = self.task()
        self.assertEqual(len(self.speech.deferred), 1)
        self.pump(1)
        self.assertEqual(self.mouth.said, [SP.task_line(self.led.tasks[tid])])
        self.assertFalse(self.led.has_delivery("task", tid, outcome=L.K_COMPLETED))     # handed to the mouth, not yet heard
        self.assertEqual(len(self.speech.deferred), 1)                                    # still there for the next-turn note path
        self.mouth.finish()
        self.pump(1)
        self.assertTrue(self.led.has_delivery("task", tid, outcome=L.K_COMPLETED))
        self.assertEqual(self.speech.deferred, [])
        self.pump(5, steps=4)
        self.assertEqual(len(self.mouth.said), 1)

    # roadmap 6: failed tasks stay failed
    def test_a_failed_task_is_announced_as_failed(self):
        tid = self.task(state=L.K_FAILED)
        self.pump(1)
        self.assertEqual(self.mouth.said, ["Background task count the daily notes failed."])

    # roadmap 7 and the 21-vs-20 case: a contradicted number is never repeated
    def test_the_21_vs_20_summary_is_never_announced_as_20(self):
        self.task(summary="20 daily note files found", vstate=L.V_FAILED, facts=[21])
        self.pump(1)
        said = self.mouth.said[0]
        self.assertIn("21", said)
        self.assertNotIn("20", said)
        self.assertIn("didn't match", said)

    def test_an_unverified_number_is_withheld_not_announced_as_a_success(self):
        self.task(summary="20 daily note files found", vstate=L.V_UNVERIFIED)
        self.pump(1)
        said = self.mouth.said[0]
        self.assertNotIn("20", said)
        self.assertIn("I've got its report", said)

    def test_a_verified_number_is_spoken_with_its_noun(self):
        tid = self.task(summary="21 files", vstate=L.V_VERIFIED, facts=[21])
        self.led.set_task_verification(tid, L.V_VERIFIED, L.LEVEL_TOOL_EVIDENCE, "ev", facts=[21],
                                       matched=[{"n": 21, "noun": "files"}])
        self.pump(1)
        self.assertIn("21 files", self.mouth.said[0])

    # roadmap 5: duplicates
    def test_the_same_task_deferred_twice_is_announced_once(self):
        tid = self.task()
        self.speech.deferred.append(SP.Candidate(SP.BG_COMPLETION, "code", tid))
        self.pump(1, steps=3)
        self.assertEqual(len(self.mouth.said), 1)

    def test_a_task_the_model_was_already_handed_is_not_spoken_again(self):
        tid = self.task()
        note = self.speech.note_prefix()                                # the user's next turn took it first
        self.assertIn("count the daily notes", note)
        self.pump(1, steps=3)
        self.assertEqual(self.mouth.said, [])

    def test_a_note_taken_while_the_line_waited_cancels_the_spoken_line(self):
        tid = self.task()
        self.ctx = P.Context(user_speaking=True)
        self.pump(1, steps=2)
        self.assertEqual(self.mouth.said, [])
        self.speech.note_prefix()                                       # the user spoke; the model got the note
        self.ctx = P.Context()
        self.pump(3, steps=2)
        self.assertEqual(self.mouth.said, [])
        self.assertTrue(any("already handed off" in l for l in self.logs))

    def test_a_flushed_task_line_stays_available_to_the_next_turn_note(self):
        tid = self.task()
        self.pump(1)
        for _ in range(P.MAX_REQUEUES + 1):
            self.mouth.flush()
            self.pump(0, steps=2)
        self.assertFalse(self.led.has_delivery("task", tid, outcome=L.K_COMPLETED))     # never marked: it was never heard
        self.assertEqual(len(self.speech.deferred), 1)
        self.assertIn("count the daily notes", self.speech.note_prefix())

    # roadmap 10: background work survives foreground interruptions
    def test_a_foreground_interruption_changes_nothing_in_the_task_ledger(self):
        self.led.task_seen("run1", "E1", kind=L.KIND_BG_AGENT, description="long export")
        self.led.task_state("run1", L.K_RUNNING)
        before = dict(self.led.tasks["run1"])
        self.mouth.flush()                                              # the user barged in
        self.pump(1, steps=3)
        self.assertEqual(self.led.tasks["run1"]["state"], L.K_RUNNING)
        self.assertEqual(self.led.tasks["run1"].get("cancel_intent"), before.get("cancel_intent"))
        self.led.task_state("run1", L.K_COMPLETED, summary="done")
        self.speech.on_task_terminal(self.led.tasks["run1"], SP.Context(fg_active=True))
        self.pump(1)
        self.assertIn("long export", self.mouth.said[0])                # and its result still reaches the user afterwards

    def test_a_background_outcome_needs_an_active_conversation_but_a_reminder_does_not(self):
        self.task()
        self.ctx = P.Context(conversation_active=False)
        self.pump(1, steps=2)
        self.assertEqual(self.mouth.said, [])
        self.request(due_at=self.clock.t + 1)
        self.pump(2, steps=2)
        self.assertEqual(len(self.mouth.said), 1)
        self.assertTrue(self.mouth.said[0].startswith("Reminder:"))

    def test_what_jarvis_said_unprompted_is_handed_to_the_model_on_the_next_turn(self):
        self.task()
        self.pump(1, finish=True)
        note = self.speech.note_prefix()
        self.assertIn("You just said this aloud on your own, unprompted:", note)
        self.assertIn("count the daily notes", note)
        self.assertEqual(self.speech.note_prefix(), "")                 # once

    # roadmap 11: unauthorized consequential actions are blocked
    def test_a_consequential_action_request_is_never_performed(self):
        runner = self.runner(policy=P.Policy.from_config({"enabled": True, "act_allow": ["send", "status_check"]}))
        self.assertEqual(runner.engine.policy.act_allow, ("status_check",))          # `send` cannot be allowlisted at all
        ev = P.Event("bg_completion", "task:x", P.NORMAL, T0, source_id="x", action="send")
        self.assertNotEqual(runner.engine.evaluate(ev, P.Context(), T0).outcome, P.ACT)

    def test_an_allowed_action_is_logged_but_nothing_is_performed_because_no_runner_exists(self):
        runner = self.runner(policy=P.Policy(enabled=True, act_allow=("status_check",)))
        runner.engine.offer(P.Event("bg_completion", "task:y", P.NORMAL, T0, source_id="y", action="status_check"), T0)
        runner.step()
        self.assertTrue(any("act_allowed" in l and "nothing was done" in l for l in self.logs))
        self.assertEqual(self.mouth.said, [])


class RobustnessTests(Base):
    def test_one_failing_step_does_not_stop_the_others(self):
        tid = self.task()
        self.r.collect_due = lambda now: (_ for _ in ()).throw(RuntimeError("boom"))
        self.pump(1)
        self.assertEqual(len(self.mouth.said), 1)                       # the ledger path still ran
        self.assertEqual(self.r.errors, 1)
        self.assertTrue(any("step failed what=due error=RuntimeError" in l for l in self.logs))

    def test_a_disabled_engine_speaks_nothing_and_keeps_nothing(self):
        r = self.runner(policy=P.Policy())
        self.request()
        self.task()
        self.pump(120, steps=3, r=r)
        self.assertEqual(self.mouth.said, [])
        self.assertEqual(r.engine.pending_keys(), [])

    def test_the_heartbeat_logs_once_per_interval(self):
        self.pump(PL.HEARTBEAT_S + 1)
        self.pump(10)
        self.assertEqual(sum("[proactive] alive" in l for l in self.logs), 1)

    def test_a_line_neither_heard_nor_flushed_is_eventually_treated_as_flushed(self):
        tid = self.task()
        self.pump(1)
        self.assertEqual(len(self.mouth.said), 1)
        self.pump(PL.INFLIGHT_TIMEOUT_S + 1)                            # the mouth never reports completion
        self.assertTrue(any("flushed before it was heard" in l or "requeue bound spent" in l for l in self.logs))

    def test_the_log_never_carries_the_message_text(self):
        self.request(title="secret title words")
        self.task()
        self.pump(0)
        self.pump(61, steps=2, finish=True)
        for l in self.logs:
            self.assertNotIn("secret title words", l)
            self.assertNotIn("count the daily notes", l)


class StaticPinTests(unittest.TestCase):
    def test_the_runner_never_touches_the_addressee_gate_or_pending_reports(self):
        tree = ast.parse(inspect.getsource(PL))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree)
                                                                           if isinstance(n, ast.Attribute)}
        for banned in ("_ADDRESSEE", "AddresseeGate", "open", "touch", "pending_reports", "PendingReport", "router",
                       "can_use_tool"):
            self.assertNotIn(banned, names - {"open"} if banned == "open" else names, banned)
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse([m for m in mods if "addressee" in m])

    def test_the_runner_imports_only_what_it_should(self):
        tree = ast.parse(inspect.getsource(PL))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"dataclasses", "json", "os", "time", "backtalk.ledger", "backtalk.proactive",
                                "backtalk.speech_policy"})


if __name__ == "__main__":
    unittest.main()
