"""Scheduler (backtalk/scheduler.py): persistence, firing, recurring math, restart, missed schedules, bounded retries,
cancellation. A fake clock drives everything; nothing sleeps and nothing speaks."""
import json
import os
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from backtalk import scheduler as S

T0 = 1_800_000_000.0          # an arbitrary fixed epoch second
CT = ZoneInfo("America/Chicago")


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.clock = Clock()
        self.open = []

    def tearDown(self):
        for s in self.open:
            s.close()
        self.tmp.cleanup()

    def sched(self, **kw):
        s = S.Scheduler(self.dir, clock=self.clock, **kw)
        self.open.append(s)
        return s

    def reopen(self, old):
        old.close()
        return self.sched()

    @staticmethod
    def read(path):
        with open(path, "rb") as fh:
            return fh.read()

    @staticmethod
    def write(path, data):
        with open(path, "wb") as fh:
            fh.write(data)

    def lines(self):
        with open(os.path.join(self.dir, "scheduler.jsonl")) as fh:
            return [json.loads(l) for l in fh if l.strip()]


class OneTimeTests(Base):
    def test_a_one_time_task_fires_once_and_only_once(self):
        s = self.sched()
        t = s.add_once("call the title company", T0 + 60)
        self.assertEqual(s.due(T0 + 59), [])
        fires = s.due(T0 + 60)
        self.assertEqual([f.task_id for f in fires], [t.id])
        self.assertEqual(fires[0].title, "call the title company")
        self.assertEqual(s.due(T0 + 61), [])                          # claimed: not offered again while running
        self.assertTrue(s.complete(fires[0].run_id))
        self.assertEqual(s.due(T0 + 10_000), [])
        self.assertEqual(s.get(t.id).status, "done")

    def test_a_fire_is_durable_before_the_caller_acts_and_restart_does_not_repeat_completed_work(self):
        s = self.sched()
        t = s.add_once("x", T0 + 10)
        f = s.due(T0 + 10)[0]
        s.complete(f.run_id)
        s2 = self.reopen(s)
        self.assertEqual(s2.due(T0 + 99_999), [])
        self.assertEqual(s2.get(t.id).status, "done")
        self.assertEqual(s2.recover(), [])

    def test_a_crash_mid_run_is_one_failed_attempt_never_a_success(self):
        s = self.sched()
        t = s.add_once("x", T0 + 10)
        s.due(T0 + 10)                                                # claimed, never completed: the process dies
        s2 = self.reopen(s)
        self.assertEqual(s2.recover(), [t.id])
        got = s2.get(t.id)
        self.assertEqual((got.status, got.last_status), ("scheduled", "failed"))
        self.assertIn("restart", got.last_error)
        self.assertEqual(s2.due(T0 + 10), [])                         # backoff: not immediately
        again = s2.due(T0 + 10 + S.BACKOFF_BASE_S)
        self.assertEqual(len(again), 1)
        self.assertEqual(again[0].attempt, 2)

    def test_a_late_one_time_task_is_missed_deterministically(self):
        s = self.sched()
        t = s.add_once("x", T0, grace_s=60)
        out = s.due(T0 + 61)                                          # too late: never run, but reported once
        self.assertEqual([(f.missed, f.task_id) for f in out], [(True, t.id)])
        self.assertEqual((s.get(t.id).status, s.get(t.id).last_status), ("missed", "missed"))
        self.assertIn("missed by 61s", s.get(t.id).last_error)
        u = s.add_once("y", T0, grace_s=60)
        self.assertEqual(len(s.due(T0 + 60)), 1)                      # exactly at the grace edge still fires
        self.assertEqual(s.get(u.id).status, "running")

    def test_cancelling_before_the_time_prevents_any_execution(self):
        s = self.sched()
        t = s.add_once("x", T0 + 5)
        self.assertTrue(s.cancel(t.id))
        self.assertEqual(s.due(T0 + 500), [])
        self.assertEqual(s.get(t.id).status, "cancelled")
        self.assertFalse(s.cancel(t.id))                              # already cancelled
        s2 = self.reopen(s)
        self.assertEqual(s2.due(T0 + 500), [])                        # and it stays cancelled across a restart

    def test_cancelling_a_run_in_flight_ignores_its_late_result(self):
        s = self.sched()
        t = s.add_every("x", 60, start_at=T0)
        f = s.due(T0)[0]
        self.assertTrue(s.cancel(t.id))
        self.assertEqual(s.get(t.id).status, "running")               # the run in flight is flagged, not killed
        self.assertTrue(s.complete(f.run_id))
        self.assertEqual(s.get(t.id).status, "cancelled")
        self.assertEqual(s.due(T0 + 10_000), [])

    def test_a_key_makes_add_idempotent_until_the_task_ends(self):
        s = self.sched()
        a = s.add_once("x", T0 + 5, key="reminder:call")
        b = s.add_once("x", T0 + 5, key="reminder:call")
        self.assertIs(a, b)
        self.assertEqual(len(s.tasks()), 1)
        s.cancel(a.id)
        c = s.add_once("x", T0 + 5, key="reminder:call")
        self.assertNotEqual(a.id, c.id)


class RetryTests(Base):
    def run_failures(self, s, n, kind="once"):
        waits, t = [], T0
        for i in range(n):
            fires = s.due(t)
            self.assertEqual(len(fires), 1, i)
            self.assertTrue(s.fail(fires[0].run_id, "boom"))
            task = s.get(fires[0].task_id)
            if task.retry_at is None:
                return waits, task
            waits.append(task.retry_at - self.clock.t)
            t = task.retry_at
        return waits, s.get(fires[0].task_id)

    def test_failures_back_off_and_are_bounded_then_stay_failed(self):
        s = self.sched()
        t = s.add_once("x", T0, max_retries=3)
        waits, last = self.run_failures(s, 4)
        self.assertEqual(waits, [30.0, 60.0, 120.0])                  # exponential backoff, three retries
        self.assertEqual((last.status, last.last_status, last.last_error), ("failed", "failed", "boom"))
        self.assertEqual(s.due(T0 + 99_999), [])                      # a failed task is never resurrected
        self.assertEqual(last.runs, 4)                                # the first attempt plus three retries

    def test_backoff_is_capped(self):
        self.assertEqual(S._backoff(1), 30.0)
        self.assertEqual(S._backoff(20), S.BACKOFF_MAX_S)

    def test_a_retry_that_succeeds_ends_done(self):
        s = self.sched()
        t = s.add_once("x", T0)
        f = s.due(T0)[0]
        s.fail(f.run_id, "transient")
        retry = s.due(s.get(t.id).retry_at)[0]
        self.assertEqual(retry.attempt, 2)
        s.complete(retry.run_id)
        self.assertEqual((s.get(t.id).status, s.get(t.id).last_error), ("done", None))

    def test_a_retry_ignores_the_grace_window(self):
        s = self.sched()
        t = s.add_once("x", T0, grace_s=1)
        f = s.due(T0)[0]
        s.fail(f.run_id, "transient")
        self.assertEqual(len(s.due(s.get(t.id).retry_at + 5_000)), 1)  # far past grace, still a retry, not "missed"

    def test_zero_retries_fails_at_once(self):
        s = self.sched()
        t = s.add_once("x", T0, max_retries=0)
        s.fail(s.due(T0)[0].run_id, "no")
        self.assertEqual(s.get(t.id).status, "failed")

    def test_a_recurring_task_keeps_going_after_a_final_failure_and_remembers_it(self):
        s = self.sched()
        t = s.add_every("x", 3600, start_at=T0, max_retries=0)
        s.fail(s.due(T0)[0].run_id, "no")
        got = s.get(t.id)
        self.assertEqual((got.status, got.failures, got.last_error, got.next), ("scheduled", 1, "no", T0 + 3600))
        self.assertEqual(len(s.due(T0 + 3600)), 1)

    def test_complete_and_fail_on_unknown_or_finished_runs_are_ignored(self):
        s = self.sched()
        t = s.add_once("x", T0)
        f = s.due(T0)[0]
        self.assertFalse(s.complete("nope:1"))
        self.assertTrue(s.complete(f.run_id))
        self.assertFalse(s.complete(f.run_id))
        self.assertFalse(s.fail(f.run_id, "late"))
        self.assertEqual(s.get(t.id).status, "done")


class RecurringTests(Base):
    def test_every_computes_the_next_run_and_runs_each_slot(self):
        s = self.sched()
        t = s.add_every("pulse", 600, start_at=T0)
        for k in range(3):
            f = s.due(T0 + 600 * k)
            self.assertEqual(len(f), 1)
            self.assertEqual((f[0].scheduled_for, f[0].coalesced), (T0 + 600 * k, 0))
            self.assertEqual(s.get(t.id).next, T0 + 600 * (k + 1))
            s.complete(f[0].run_id)

    def test_every_collapses_missed_slots_into_one_fire(self):
        s = self.sched()
        t = s.add_every("pulse", 600, start_at=T0)
        f = s.due(T0 + 600 * 5 + 7)                                   # the slots at 0..5 elapsed while nothing ran
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].coalesced, 5)
        self.assertEqual(s.get(t.id).next, T0 + 600 * 6)              # the first slot strictly after now
        self.assertEqual(f[0].late_s, 600 * 5 + 7)

    def test_a_long_outage_is_constant_time(self):
        s = self.sched()
        t = s.add_every("pulse", 1, start_at=T0)
        f = s.due(T0 + 31_536_000)                                    # a year of one-second slots
        self.assertEqual(f[0].coalesced, 31_536_000)                  # slots T0+1 .. T0+year all elapsed

    def test_a_run_in_flight_is_never_overlapped(self):
        s = self.sched()
        t = s.add_every("pulse", 60, start_at=T0)
        f = s.due(T0)[0]
        self.assertEqual(s.due(T0 + 60), [])                          # still running: this slot is skipped
        s.complete(f.run_id)
        self.assertEqual(len(s.due(T0 + 120)), 1)

    def test_every_before_its_start_waits(self):
        s = self.sched()
        s.add_every("x", 60, start_at=T0 + 1000)
        self.assertEqual(s.due(T0 + 999), [])
        self.assertEqual(len(s.due(T0 + 1000)), 1)

    def test_daily_runs_at_local_wall_clock_time(self):
        self.clock.t = datetime(2026, 10, 19, 20, 0, tzinfo=CT).timestamp()
        s = self.sched()
        t = s.add_daily("morning brief", "05:30")
        want = datetime(2026, 10, 20, 5, 30, tzinfo=CT).timestamp()
        self.assertEqual(t.next, want)
        self.assertEqual(s.due(want - 1), [])
        f = s.due(want)[0]
        self.assertEqual(s.get(t.id).next, datetime(2026, 10, 21, 5, 30, tzinfo=CT).timestamp())
        s.complete(f.run_id)

    def test_daily_is_dst_safe_across_the_autumn_change(self):
        # Chicago leaves daylight time on 2026-11-01: the wall-clock 05:30 slot must still be 05:30, a 25-hour day apart.
        self.clock.t = datetime(2026, 10, 31, 6, 0, tzinfo=CT).timestamp()
        s = self.sched()
        t = s.add_daily("x", "05:30")
        d1 = datetime(2026, 11, 1, 5, 30, tzinfo=CT).timestamp()
        self.assertEqual(t.next, d1)
        f = s.due(d1)[0]
        d2 = datetime(2026, 11, 2, 5, 30, tzinfo=CT).timestamp()
        self.assertEqual(s.get(t.id).next, d2)
        self.assertEqual((d2 - d1) / 3600, 24.0)
        self.assertEqual(datetime.fromtimestamp(d1, CT).hour, 5)

    def test_daily_missed_days_coalesce(self):
        self.clock.t = datetime(2026, 10, 19, 20, 0, tzinfo=CT).timestamp()
        s = self.sched()
        t = s.add_daily("x", "05:30")
        after = datetime(2026, 10, 23, 12, 0, tzinfo=CT).timestamp()  # Oct 20, 21, 22, 23 slots all elapsed
        f = s.due(after)[0]
        self.assertEqual(f.coalesced, 3)
        self.assertEqual(s.get(t.id).next, datetime(2026, 10, 24, 5, 30, tzinfo=CT).timestamp())

    def test_restart_keeps_the_schedule_and_does_not_replay_a_finished_slot(self):
        s = self.sched()
        t = s.add_every("x", 600, start_at=T0)
        f = s.due(T0)[0]
        s.complete(f.run_id)
        s2 = self.reopen(s)
        self.assertEqual(s2.get(t.id).next, T0 + 600)
        self.assertEqual(s2.due(T0 + 599), [])
        self.assertEqual(len(s2.due(T0 + 600)), 1)


class ValidationTests(Base):
    def test_bad_inputs_are_refused(self):
        s = self.sched()
        for bad in (lambda: s.add_once("", T0), lambda: s.add_once("x", "soon"), lambda: s.add_once("x", True),
                    lambda: s.add_every("x", 0), lambda: s.add_every("x", True), lambda: s.add_daily("x", "25:00"),
                    lambda: s.add_daily("x", "8:30"), lambda: s.add_daily("x", "08:30", tz="Mars/Base"),
                    lambda: s.add_once("x", T0, max_retries=11), lambda: s.add_once("x", T0, grace_s=-1)):
            with self.assertRaises(S.SchedulerError):
                bad()
        self.assertEqual(s.tasks(), [])

    def test_a_title_is_bounded_and_single_line(self):
        s = self.sched()
        t = s.add_once("a\n\n b " + "z" * 500, T0)
        self.assertLessEqual(len(t.title), S.TITLE_MAX)
        self.assertNotIn("\n", t.title)


class StorageTests(Base):
    def test_state_is_the_fold_of_an_fsynced_append_only_file(self):
        s = self.sched()
        t = s.add_once("x", T0)
        s.due(T0)
        ops = [r["op"] for r in self.lines()]
        self.assertEqual(ops, ["add", "fire"])
        self.assertTrue(all(r["t"] == "sched" for r in self.lines()))

    def test_a_torn_last_line_is_cut_off_and_nothing_is_lost(self):
        s = self.sched()
        t = s.add_once("x", T0)
        s.close()
        with open(os.path.join(self.dir, "scheduler.jsonl"), "ab") as f:
            f.write(b'{"t":"sched","op":"fire","id":"' + t.id.encode() + b'","at"')    # a crash mid-append
        s2 = self.sched()
        self.assertEqual(s2.get(t.id).status, "scheduled")
        self.assertEqual(len(s2.due(T0)), 1)                          # and the file is healthy for new appends
        s3 = self.reopen(s2)
        self.assertEqual(s3.get(t.id).status, "running")

    def test_a_complete_record_missing_only_its_newline_is_kept(self):
        s = self.sched()
        t = s.add_once("x", T0)
        s.close()
        path = os.path.join(self.dir, "scheduler.jsonl")
        raw = self.read(path)
        self.write(path, raw.rstrip(b"\n"))
        s2 = self.sched()
        self.assertEqual(s2.get(t.id).title, "x")

    def test_corruption_in_the_middle_fails_closed_and_is_not_repaired(self):
        s = self.sched()
        s.add_once("x", T0)
        s.add_once("y", T0)
        s.close()
        path = os.path.join(self.dir, "scheduler.jsonl")
        lines = self.read(path).split(b"\n")
        lines[0] = b"not json at all"
        self.write(path, b"\n".join(lines))
        before = self.read(path)
        with self.assertRaises(S.SchedulerCorrupt):
            S.Scheduler(self.dir, clock=self.clock)
        self.assertEqual(self.read(path), before)

    def test_a_foreign_record_type_fails_closed(self):
        s = self.sched()
        s.close()
        with open(os.path.join(self.dir, "scheduler.jsonl"), "ab") as f:
            f.write(b'{"t":"other","op":"add"}\n')
        with self.assertRaises(S.SchedulerCorrupt):
            S.Scheduler(self.dir, clock=self.clock)

    def test_a_second_process_on_the_same_directory_is_refused(self):
        self.sched()
        with self.assertRaises(S.SchedulerLocked):
            S.Scheduler(self.dir, clock=self.clock)

    def test_next_due_at_is_the_earliest_eligible_time(self):
        s = self.sched()
        self.assertIsNone(s.next_due_at())
        s.add_once("a", T0 + 50)
        s.add_once("b", T0 + 20)
        c = s.add_once("c", T0 + 5)
        s.cancel(c.id)
        self.assertEqual(s.next_due_at(), T0 + 20)


class ModuleHygieneTests(unittest.TestCase):
    def test_the_core_imports_nothing_from_the_voice_line(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(S))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module)
        self.assertEqual(mods, {"fcntl", "json", "math", "os", "re", "time", "dataclasses", "datetime", "zoneinfo"})


if __name__ == "__main__":
    unittest.main()


class ReviewFixTests(Base):
    """Regression tests for the 2026-10-19 architecture review of scheduler.py. Each one failed before its fix."""

    def test_the_fire_key_is_stable_across_retries_and_restarts_of_one_slot(self):
        s = self.sched()
        t = s.add_once("x", T0)
        f1 = s.due(T0)[0]
        s.fail(f1.run_id, "boom")
        f2 = s.due(s.get(t.id).retry_at)[0]
        self.assertNotEqual(f1.run_id, f2.run_id)
        self.assertEqual(f1.fire_key, f2.fire_key)                     # the announcer dedupes on this
        s2 = self.reopen(s)
        s2.recover()
        f3 = s2.due(s2.get(t.id).retry_at)[0]
        self.assertEqual(f3.fire_key, f1.fire_key)
        self.assertEqual(f1.fire_key, f"{t.id}@{T0}")

    def test_each_recurring_slot_has_its_own_fire_key(self):
        s = self.sched()
        s.add_every("x", 60, start_at=T0)
        a = s.due(T0)[0]
        s.complete(a.run_id)
        b = s.due(T0 + 60)[0]
        self.assertNotEqual(a.fire_key, b.fire_key)

    def test_a_missed_one_time_reminder_is_reported_not_silently_dropped(self):
        s = self.sched()
        t = s.add_once("call the title company", T0, grace_s=60)
        out = s.due(T0 + 1800)                                         # the lid was closed for half an hour
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].missed)
        self.assertEqual((out[0].title, out[0].task_id), ("call the title company", t.id))
        self.assertEqual(out[0].late_s, 1800)
        self.assertEqual(s.get(t.id).status, "missed")
        self.assertEqual(s.due(T0 + 1801), [])                         # reported exactly once

    def test_the_default_grace_is_hours_so_a_closed_lid_does_not_lose_a_reminder(self):
        self.assertGreaterEqual(S.DEFAULT_GRACE_S, 4 * 3600)
        s = self.sched()
        s.add_once("x", T0)
        out = s.due(T0 + 3600)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].missed)

    def test_a_failed_append_inside_due_does_not_strand_earlier_claims(self):
        s = self.sched()
        for i in range(3):
            s.add_once(f"t{i}", T0 + i)
        real, calls = s._append, [0]

        def flaky(rec):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError("disk")
            return real(rec)
        s._append = flaky
        fires = s.due(T0 + 10)
        self.assertEqual(len(fires), 2)                                # the two durable claims reach the caller
        self.assertTrue(all(s.get(f.task_id).status == "running" for f in fires))
        self.assertTrue(s.errors)

    def test_a_write_failure_poisons_the_scheduler_and_leaves_the_file_clean(self):
        s = self.sched()
        t = s.add_once("x", T0)
        path = os.path.join(self.dir, "scheduler.jsonl")
        before = self.read(path)

        class Boom:
            def __init__(self, fh):
                self.fh = fh

            def write(self, b):
                self.fh.write(b[: len(b) // 2])                       # a partial write, then the failure
                raise OSError(28, "No space left on device")

            def __getattr__(self, n):
                return getattr(self.fh, n)
        s._fh = Boom(s._fh)
        with self.assertRaises(S.SchedulerError):
            s.cancel(t.id)
        self.assertEqual(self.read(path), before)                      # the fragment was truncated back
        with self.assertRaises(S.SchedulerError):                      # fail closed: no more writes this process
            s.add_once("y", T0)
        self.assertEqual(s.get(t.id).status, "scheduled")              # and memory never moved ahead of the disk
        s2 = self.reopen(s) if False else None
        s._fh = s._fh.fh
        s.close()
        self.open.remove(s)
        again = self.sched()                                           # a restart reads a healthy file
        self.assertEqual(again.get(t.id).status, "scheduled")

    def test_non_finite_and_absurd_numbers_are_refused(self):
        s = self.sched()
        nan, inf = float("nan"), float("inf")
        for bad in (lambda: s.add_every("x", nan), lambda: s.add_every("x", inf), lambda: s.add_every("x", 60, start_at=nan),
                    lambda: s.add_once("x", nan), lambda: s.add_once("x", inf), lambda: s.add_once("x", T0, grace_s=nan),
                    lambda: s.add_once("x", T0, grace_s=inf), lambda: s.add_every("x", 1e300),
                    lambda: s.add_once("x", 1e300)):
            with self.assertRaises(S.SchedulerError):
                bad()
        self.assertEqual(s.tasks(), [])
        s.add_once("fine", T0)
        self.assertEqual(len(s.due(T0)), 1)                            # nothing was poisoned

    def test_a_complete_but_invalid_last_line_fails_closed_with_the_right_exception(self):
        s = self.sched()
        s.add_once("x", T0)
        s.close()
        with open(os.path.join(self.dir, "scheduler.jsonl"), "ab") as f:
            f.write(b'{"t":"sched","op":"fire","id":"nope"}')          # valid JSON, unknown task, no newline
        with self.assertRaises(S.SchedulerCorrupt):
            S.Scheduler(self.dir, clock=self.clock)

    def test_a_key_reused_with_a_different_schedule_is_an_error_not_a_silent_old_task(self):
        s = self.sched()
        a = s.add_once("x", T0 + 60, key="k")
        self.assertIs(s.add_once("x", T0 + 60, key="k"), a)            # identical request: idempotent
        with self.assertRaises(S.SchedulerError):
            s.add_once("x", T0 + 600, key="k")
        with self.assertRaises(S.SchedulerError):
            s.add_once("other title", T0 + 60, key="k")
        b = s.add_once("y", T0 + 60, key=1)
        c = s.add_once("z", T0 + 70, key=True)                         # 1 and True are different keys
        self.assertNotEqual(b.id, c.id)

    def test_has_key_sees_finished_tasks_too(self):
        s = self.sched()
        t = s.add_once("x", T0, key="inbox:abc")
        f = s.due(T0)[0]
        s.complete(f.run_id)
        self.assertTrue(s.has_key("inbox:abc"))
        self.assertFalse(s.has_key("inbox:zzz"))

    def test_a_clock_set_backwards_re_anchors_a_recurring_task(self):
        s = self.sched()
        t = s.add_every("pulse", 60, start_at=T0)
        for k in range(3):
            s.complete(s.due(T0 + 60 * k)[0].run_id)
        self.assertEqual(s.get(t.id).next, T0 + 180)
        back = T0 + 60 * 2 - 7200                                      # the system clock was set two hours back
        self.assertEqual(s.due(back), [])                              # re-anchored, not stalled behind a far future slot
        self.assertLess(s.get(t.id).next - back, 61)
        self.assertEqual(len(s.due(s.get(t.id).next)), 1)

    def test_read_only_view_lists_tasks_without_the_lock(self):
        s = self.sched()
        t = s.add_once("x", T0, key="k")
        view = S.Scheduler.read_only(self.dir)
        self.assertEqual([x.id for x in view.tasks()], [t.id])
        with self.assertRaises(S.SchedulerError):
            view.add_once("y", T0)


class WeakSpotTests(Base):
    """Targets the mutants the independent verification agent found surviving the first 33 tests."""

    def test_durability_every_append_is_fsynced(self):
        calls = []
        real = os.fsync
        os.fsync = lambda fd: (calls.append(fd), real(fd))[1]
        try:
            s = S.Scheduler(self.dir, clock=self.clock, fsync=True)
            self.open.append(s)
            t = s.add_once("x", T0)
            s.due(T0)
        finally:
            os.fsync = real
        self.assertEqual(len(calls), 2)                                # one per record: add, fire

    def test_tasks_due_at_the_same_instant_fire_in_insertion_order(self):
        s = self.sched()
        ids = [s.add_once(f"t{i}", T0).id for i in range(20)]
        self.assertEqual([f.task_id for f in s.due(T0)], ids)

    def test_a_stale_complete_for_an_older_run_cannot_close_a_newer_run(self):
        s = self.sched()
        t = s.add_once("x", T0)
        f1 = s.due(T0)[0]
        s.fail(f1.run_id, "boom")
        f2 = s.due(s.get(t.id).retry_at)[0]
        self.assertFalse(s.complete(f1.run_id))                        # the first run is over; it cannot close the second
        self.assertEqual(s.get(t.id).status, "running")
        self.assertTrue(s.complete(f2.run_id))

    def test_cancelling_during_retry_backoff_leaves_no_retry(self):
        s = self.sched()
        t = s.add_once("x", T0)
        s.fail(s.due(T0)[0].run_id, "boom")
        self.assertIsNotNone(s.get(t.id).retry_at)
        self.assertTrue(s.cancel(t.id))
        self.assertIsNone(s.get(t.id).retry_at)
        self.assertEqual(s.due(T0 + 99_999), [])
        self.assertEqual(s.get(t.id).status, "cancelled")
        self.assertEqual(self.reopen(s).get(t.id).status, "cancelled")

    def test_a_cancelled_run_in_flight_that_then_fails_is_cancelled_not_retried(self):
        s = self.sched()
        t = s.add_once("x", T0)
        f = s.due(T0)[0]
        s.cancel(t.id)
        s.fail(f.run_id, "boom")
        self.assertEqual((s.get(t.id).status, s.get(t.id).retry_at), ("cancelled", None))
        self.assertEqual(s.due(T0 + 99_999), [])

    def test_a_task_flagged_for_cancellation_is_never_claimed(self):
        s = self.sched()
        t = s.add_every("x", 60, start_at=T0)
        f = s.due(T0)[0]
        s.cancel(t.id)                                                 # flagged while running
        s.get(t.id).status = "scheduled"                               # even if some path put it back, due() must skip it
        self.assertEqual(s.due(T0 + 10_000), [])

    def test_recover_twice_is_a_no_op_the_second_time(self):
        s = self.sched()
        t = s.add_once("x", T0)
        s.due(T0)
        s2 = self.reopen(s)
        self.assertEqual(s2.recover(), [t.id])
        self.assertEqual(s2.recover(), [])
        self.assertEqual(s2.get(t.id).attempts, 1)

    def test_a_task_that_needs_no_retries_fails_when_a_crash_interrupts_it(self):
        s = self.sched()
        t = s.add_once("x", T0, max_retries=0)
        s.due(T0)
        s2 = self.reopen(s)
        s2.recover()
        self.assertEqual(s2.get(t.id).status, "failed")

    def test_a_payload_that_is_not_json_is_refused_before_anything_is_written(self):
        s = self.sched()
        path = os.path.join(self.dir, "scheduler.jsonl")
        before = self.read(path) if os.path.exists(path) else b""
        with self.assertRaises(TypeError):
            s.add_once("x", T0, payload={"o": object()})
        self.assertEqual(s.tasks(), [])
        self.assertEqual(self.read(path) if os.path.exists(path) else b"", before)

    def test_daily_0230_across_the_spring_forward_gap_fires_once_each_day(self):
        self.clock.t = datetime(2026, 3, 7, 12, 0, tzinfo=CT).timestamp()       # the gap is 2026-03-08 02:00 to 03:00
        s = self.sched()
        t = s.add_daily("x", "02:30")
        seen = []
        for _ in range(4):
            nxt = s.get(t.id).next
            f = s.due(nxt)[0]
            seen.append(datetime.fromtimestamp(f.scheduled_for, CT))
            s.complete(f.run_id)
        self.assertEqual([d.day for d in seen], [8, 9, 10, 11])         # exactly one fire per calendar day, none skipped or doubled
        self.assertEqual([d.date().day for d in seen], sorted({d.date().day for d in seen}))
        for d in seen[1:]:
            self.assertEqual((d.hour, d.minute), (2, 30))
        self.assertIn(seen[0].hour, (1, 3))                             # the nonexistent 02:30 lands on the gap's edge

    def test_daily_0130_across_the_autumn_overlap_fires_once(self):
        self.clock.t = datetime(2026, 10, 31, 12, 0, tzinfo=CT).timestamp()      # 2026-11-01 01:00 to 02:00 happens twice
        s = self.sched()
        t = s.add_daily("x", "01:30")
        days = []
        for _ in range(3):
            f = s.due(s.get(t.id).next)[0]
            days.append(datetime.fromtimestamp(f.scheduled_for, CT).date().isoformat())
            s.complete(f.run_id)
        self.assertEqual(days, ["2026-11-01", "2026-11-02", "2026-11-03"])

    def test_a_year_long_outage_does_not_iterate(self):
        s = self.sched()
        t = s.add_every("x", 1, start_at=T0)
        calls = [0]
        real = S._next_after

        def counting(*a):
            calls[0] += 1
            return real(*a)
        S._next_after = counting
        try:
            s.due(T0 + 31_536_000)
        finally:
            S._next_after = real
        self.assertLessEqual(calls[0], 3)                               # arithmetic, not a loop over 31 million slots

    def test_the_history_is_capped(self):
        s = self.sched()
        t = s.add_every("x", 1, start_at=T0)
        for k in range(30):
            f = s.due(T0 + k)[0]
            s.complete(f.run_id)
        self.assertLessEqual(len(s.get(t.id).history), 20)

    def test_a_second_process_really_cannot_take_the_lock_until_the_first_closes(self):
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, %r); from backtalk import scheduler as S\n"
                "try:\\n S.Scheduler(%r)\\nexcept S.SchedulerLocked:\\n print('locked')\\n" % (
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.dir))
        code = code.replace("\\n", "\n")
        s = self.sched()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.stdout.strip(), "locked")
        s.close()
        self.open.remove(s)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.stdout.strip(), "")                        # free after close: the other process took it without error
