"""Background-agent failure and recovery matrix (2026-10-21 reliability brief, item 7).

Each scenario drives the real Router, Ledger and SpeechPolicy through a fake client, then asserts three
things: the ledger task state, the taskstate decision (status, failure class, recovery, whether the result
may be claimed), and what the user would actually be told. Simulation only: no live service is touched."""
import tempfile

from backtalk import ledger as L
from backtalk import router as R
from backtalk import speech_policy as SP
from backtalk import taskstate as TS
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_task_verification import Base, TID, LAUNCH_TUID, CHILD_TUID

GOOD = "21 daily note files."
BAD = "20 daily note files."


class Matrix(Base):
    def view(self):
        return TS.task_view(self.led, TID)

    def told(self):
        return SP.task_line(self.led.tasks[TID]) + " | " + self.note()

    async def test_s1_success_verified_may_be_claimed(self):
        await self.launch(); await self.child_runs(); await self.finish(GOOD)
        v = self.view()
        self.assertEqual(v.status, TS.SUCCEEDED)
        self.assertTrue(v.claim_allowed)
        self.assertEqual(v.recovery.action, TS.NONE)

    async def test_s2_wrong_count_is_partial_and_never_spoken(self):
        await self.launch(); await self.child_runs(); await self.finish(BAD)
        v = self.view()
        self.assertEqual(v.status, TS.PARTIAL)
        self.assertFalse(v.claim_allowed)
        self.assertEqual(v.failure_class, TS.PARTIAL_SUCCESS)
        self.assertNotRegex(self.told(), r"\b20\b")

    async def test_s3_task_reported_failed(self):
        await self.launch(); await self.child_runs()
        self.client.feed(F.task_updated(TID, "failed"),
                         F.task_note(TID, LAUNCH_TUID, "failed", "could not read the folder"),
                         *F.unsolicited_run("report"))
        await settle()
        self.assertEqual(self.led.tasks[TID]["state"], L.K_FAILED)
        v = self.view()
        self.assertEqual(v.status, TS.FAILED)
        self.assertFalse(v.claim_allowed)
        self.assertNotEqual(v.recovery.action, TS.NONE)      # a failure always carries a recovery decision

    async def test_s4_user_stop_is_cancelled_not_failed(self):
        await self.launch()
        ok = await self.r.stop_task(TID, "USER_STOP")
        self.assertTrue(ok)
        self.client.feed(F.task_updated(TID, "stopped"), F.task_note(TID, LAUNCH_TUID, "stopped", ""))
        await settle()
        self.assertEqual(self.led.tasks[TID]["state"], L.K_STOPPED_USER)
        v = self.view()
        self.assertEqual(v.status, TS.CANCELLED)
        self.assertEqual(v.cancel_cause, "user")
        self.assertFalse(v.claim_allowed)

    async def test_s4b_stop_without_intent_is_unattributed(self):
        await self.launch()
        self.client.feed(F.task_updated(TID, "stopped"), F.task_note(TID, LAUNCH_TUID, "stopped", ""))
        await settle()
        self.assertEqual(self.led.tasks[TID]["state"], L.K_STOPPED_UNATTRIBUTED)
        self.assertEqual(self.view().cancel_cause, "unattributed")

    async def test_s6_late_duplicate_cannot_rewrite_the_outcome(self):
        await self.launch(); await self.child_runs(); await self.finish(GOOD)
        self.client.feed(F.task_updated(TID, "failed"), F.task_note(TID, LAUNCH_TUID, "failed", "late"))
        await settle()
        self.assertEqual(self.led.tasks[TID]["state"], L.K_COMPLETED)
        self.assertEqual(self.view().status, TS.SUCCEEDED)

    async def test_s8_outcome_is_handed_over_once(self):
        await self.launch(); await self.child_runs(); await self.finish(GOOD)
        first = self.note()
        second = self.note()
        self.assertIn("21", first)
        self.assertNotIn("21", second)


class CrashRecovery(Base):
    async def test_s7_a_crash_leaves_live_tasks_unknown_never_completed(self):
        await self.launch()
        self.assertIn(self.led.tasks[TID]["state"], L.TASK_LIVE)
        for t in (self.r._pump_task, self.r._watch_task, *self.r._escalations):
            if t:
                t.cancel()
        self.led.close()                                   # no CLOSED_CLEAN marker: a hard kill
        led2 = L.Ledger(self.tmp.name, fsync=False)
        out = led2.reconcile()
        self.assertGreaterEqual(out["crashed_epochs"], 1)
        state = led2.tasks[TID]["state"]
        self.assertIn(state, (L.K_UNKNOWN_CRASH, L.K_LOST_REBUILD, L.K_DEAD_UNKNOWN))
        v = TS.task_view(led2, TID)
        self.assertEqual(v.status, TS.UNKNOWN)
        self.assertFalse(v.claim_allowed)
        self.assertEqual(led2.reconcile()["crashed_epochs"], 0)      # idempotent: a second pass changes nothing
        led2.close()
        self.led = L.Ledger(self.tmp.name, fsync=False)              # so tearDown has something to close
