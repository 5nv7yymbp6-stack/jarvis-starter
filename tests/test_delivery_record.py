"""F-09: a durable record of the hand-off of an outcome to the user, and a startup scan for eligible outcomes whose
hand-off was never recorded. "Delivery" here means HANDED OFF (a line queued to the mouth, a note taken for the next
turn), not heard, and it is never evidence that an action worked."""
import asyncio
import tempfile
import time
import unittest

from backtalk import ledger as L
from backtalk import speech_policy as SP
from tests.test_brain_router import Base as BrainBase

DAY = 24 * 3600.0
T0 = 1_800_000_000.0


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = [T0]
        self.led = L.Ledger(self.tmp.name, fsync=False, clock=lambda: self.t[0])
        self.ep = self.led.epoch_open("s", 1)
        self.p = SP.SpeechPolicy(self.led)

    def tearDown(self):
        self.led.close()
        self.tmp.cleanup()

    def reopen(self):
        self.led.close()
        self.led = L.Ledger(self.tmp.name, fsync=False, clock=lambda: self.t[0])
        self.p = SP.SpeechPolicy(self.led)
        return self.led

    def task(self, tid, state=L.K_COMPLETED, *, epoch=None, kind=L.KIND_BG_AGENT):
        self.led.task_seen(tid, epoch or self.ep, kind=kind, description=f"job {tid}")
        self.led.task_state(tid, L.K_RUNNING)
        self.led.task_state(tid, state, cause="t", terminal_at=self.t[0])
        return self.led.tasks[tid]

    def action(self, tuid, *, error=True, epoch=None):
        """error=True: an action with something to report (its read-back failed)."""
        aid = self.led.action_seen(tuid, epoch or self.ep, "Bash")
        self.led.action_result(tuid, is_error=False, content="x")
        if error:
            self.led.set_verification(aid, L.V_FAILED, L.LEVEL_READBACK, "gone", "file")
        return aid

    def queued(self):
        return sorted((c.kind, c.source_id) for c in self.p.deferred)

    def note(self):
        return self.p.note_prefix()


class DetectionTests(LedgerCase):
    def test_1_a_finished_background_task_with_no_delivery_is_found(self):
        self.task("a")
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [(SP.BG_COMPLETION, "a")])

    def test_2_a_failed_background_task_with_no_delivery_is_found(self):
        self.task("a", L.K_FAILED)
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [(SP.BG_COMPLETION, "a")])
        self.assertIn("failed", self.note())

    def test_3_crashed_and_lost_background_tasks_with_no_delivery_are_found(self):
        for i, st in enumerate((L.K_LOST_REBUILD, L.K_UNKNOWN_CRASH, L.K_ORPHAN_RUNNING, L.K_DEAD_UNKNOWN)):
            self.task(f"t{i}", st)
        self.p.scan_undelivered()
        self.assertEqual(len(self.p.deferred), 4)
        self.assertIn("can't tell whether", self.note())

    def test_a_user_stop_is_not_an_undelivered_outcome_and_neither_is_a_foreground_tool_task(self):
        self.task("s", L.K_STOPPED_USER)
        self.task("f", L.K_COMPLETED, kind=L.KIND_FG_TOOL)
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [])

    def test_4_an_action_that_has_something_to_report_and_no_delivery_is_found(self):
        a = self.action("e1")                                                     # a failed read-back
        b = self.led.action_seen("l1", self.ep, "Bash")
        self.led.action_state(b, L.A_LOST, cause="x")                           # a lost result
        c = self.action("v1", error=False)
        self.led.set_verification(c, L.V_FAILED, L.LEVEL_READBACK, "gone", "file")   # a failed read-back
        self.action("ok1", error=False)                                   # nothing to report
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), sorted([(SP.ACTION_BAD_NEWS, a), (SP.ACTION_BAD_NEWS, b), (SP.ACTION_BAD_NEWS, c)]))

    def test_5_an_outcome_whose_delivery_was_recorded_is_skipped(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "next_turn_note", "model_note")
        a = self.action("e1")
        self.led.mark_spoken("action", a, "next_turn_note", "model_note")
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [])

    def test_a_scanned_outcome_rides_the_normal_next_turn_note_and_is_then_recorded_once(self):
        self.task("a")
        self.p.scan_undelivered()
        note = self.note()
        self.assertTrue(note.startswith("[System note, from the task ledger, not from the user:"))
        self.assertIn("job a finished", note)
        self.assertEqual(self.note(), "")
        self.assertEqual(len(self.led.tasks["a"]["deliveries"]), 1)
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_scanning_twice_before_the_note_is_taken_queues_once(self):
        self.task("a"); self.action("e1")
        self.p.scan_undelivered(); self.p.scan_undelivered()
        self.assertEqual(len(self.p.deferred), 2)


class DeliveryRecordTests(LedgerCase):
    def test_6_the_delivery_record_survives_reopening_the_ledger(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "bg_completion", "spoken")
        led = self.reopen()
        d = led.tasks["a"]["deliveries"]
        self.assertEqual(len(d), 1)
        self.assertEqual((d[0]["kind"], d[0]["line_id"], d[0]["epoch"], d[0]["outcome"], d[0]["verification"], d[0]["at"]),
                         ("spoken", "bg_completion", self.ep, L.K_COMPLETED, None, T0))
        self.assertEqual(set(d[0]), {"at", "kind", "line_id", "epoch", "outcome", "verification"})

    def test_7_marking_the_same_handoff_twice_is_idempotent(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "next_turn_note", "model_note")
        seq, row = self.led._seq, dict(self.led.tasks["a"])
        self.t[0] += 5
        self.assertTrue(self.led.mark_spoken("task", "a", "next_turn_note", "model_note"))
        self.assertEqual(self.led._seq, seq)                                       # nothing was appended
        self.assertEqual(len(self.led.tasks["a"]["deliveries"]), 1)
        self.assertEqual(self.led.tasks["a"]["user_visible"], row["user_visible"])

    def test_8_different_actions_and_tasks_cannot_collide(self):
        self.task("a"); self.task("b")
        a, b = self.action("e1"), self.action("e2")
        self.led.mark_spoken("task", "a", "x", "model_note")
        self.led.mark_spoken("action", a, "x", "model_note")
        self.assertEqual([len(self.led.tasks[t].get("deliveries") or []) for t in "ab"], [1, 0])
        self.assertEqual([len(self.led.actions[i].get("deliveries") or []) for i in (a, b)], [1, 0])
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), sorted([(SP.BG_COMPLETION, "b"), (SP.ACTION_BAD_NEWS, b)]))

    def test_9_different_outcome_fingerprints_are_distinguishable(self):
        self.task("a", L.K_UNKNOWN_CRASH)
        self.led.mark_spoken("task", "a", "x", "model_note")
        self.led.task_state("a", L.K_ORPHAN_RUNNING, cause="refined")
        self.led.mark_spoken("task", "a", "y", "model_note")
        d = self.led.tasks["a"]["deliveries"]
        self.assertEqual([x["outcome"] for x in d], [L.K_UNKNOWN_CRASH, L.K_ORPHAN_RUNNING])
        self.assertNotEqual(self.led.delivery_fingerprint("task", self.led.tasks["a"]),
                            self.led.delivery_fingerprint("task", {"state": L.K_UNKNOWN_CRASH}))

    def test_a_new_outcome_after_a_delivery_is_found_by_the_scan(self):
        self.task("a", L.K_UNKNOWN_CRASH)
        self.led.mark_spoken("task", "a", "x", "model_note")
        self.led.task_state("a", L.K_ORPHAN_RUNNING, cause="refined")
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [(SP.BG_COMPLETION, "a")])

    def test_the_same_outcome_handed_off_on_a_second_surface_is_a_second_entry_but_not_a_third(self):
        self.task("a")
        for surface in ("spoken", "model_note", "spoken", "model_note"):
            self.led.mark_spoken("task", "a", "x", surface)
        self.assertEqual([d["kind"] for d in self.led.tasks["a"]["deliveries"]], ["spoken", "model_note"])

    def test_19_spoken_and_user_visible_keep_their_meaning(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "bg_completion", "spoken")
        r = self.led.tasks["a"]
        self.assertEqual((r["spoken"]["line_id"], r["user_visible"]["surface"]), ("bg_completion", "spoken"))
        self.task("b")
        self.led.mark_spoken("task", "b", "next_turn_note", "model_note")
        r = self.led.tasks["b"]
        self.assertIsNone(r["spoken"])                                             # a note to the model is not speech
        self.assertEqual(r["user_visible"]["surface"], "model_note")
        self.assertEqual(r["deliveries"][0]["kind"], "model_note")


class LegacyRecordTests(LedgerCase):
    def test_an_outcome_handed_off_before_deliveries_existed_is_not_found_again(self):
        self.task("a"); a = self.action("e1")
        for kind, ident in (("task", "a"), ("action", a)):                      # the old record: user_visible only
            self.led._append({"t": kind, "id": ident, "spoken": None,
                              "user_visible": {"at": T0, "surface": "model_note"}})
        self.assertIsNone(self.led.tasks["a"].get("deliveries"))
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])


class VerificationChangeTests(LedgerCase):
    def verdict(self, tid, state):
        self.led.set_task_verification(tid, state, L.LEVEL_TOOL_EVIDENCE if state == L.V_VERIFIED else L.LEVEL_NONE, "e", "task-claims")

    def test_10_a_verification_that_the_policy_would_surface_makes_one_update_after_a_restart(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "next_turn_note", "model_note")           # delivered with no verdict
        self.verdict("a", L.V_VERIFIED)                                              # the verdict came after
        self.reopen()
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])                                       # not a new outcome to announce again
        note = self.note()
        self.assertIn("Update on background task job a", note)
        self.assertIn("VERIFIED", note)
        self.assertEqual(self.note(), "")                                           # and only once
        self.assertEqual([d["verification"] for d in self.led.tasks["a"]["deliveries"]], [None, L.V_VERIFIED])

    def test_10_a_verification_state_the_policy_ignores_is_not_a_new_delivery(self):
        self.task("a")
        self.led.mark_spoken("task", "a", "next_turn_note", "model_note")
        self.verdict("a", L.V_NOT_REQUIRED)
        self.reopen()
        self.p.scan_undelivered()
        self.assertEqual((self.p.deferred, self.note()), ([], ""))
        self.assertEqual(len(self.led.tasks["a"]["deliveries"]), 1)

    def test_1_same_outcome_and_same_verification_never_delivers_twice(self):
        self.task("a"); self.verdict("a", L.V_UNVERIFIED)
        self.led.mark_spoken("task", "a", "x", "model_note")
        self.led.mark_spoken("task", "a", "y", "model_note")
        self.reopen(); self.p.scan_undelivered()
        self.assertEqual(len(self.led.tasks["a"]["deliveries"]), 1)
        self.assertEqual((self.p.deferred, self.note()), ([], ""))

    def test_11_a_delivery_never_creates_or_changes_verification(self):
        self.task("a")
        a = self.action("e1", error=False)
        before = (dict(self.led.actions[a]["verification"]), self.led.tasks["a"].get("result_verification"))
        self.led.mark_spoken("action", a, "x", "model_note")
        self.led.mark_spoken("task", "a", "x", "spoken")
        self.assertEqual(self.led.actions[a]["verification"], before[0])
        self.assertEqual(self.led.tasks["a"].get("result_verification"), before[1])
        self.assertNotEqual(self.led.actions[a]["verification"]["state"], L.V_VERIFIED)
        self.assertNotIn(L.V_VERIFIED, str(self.led.actions[a]["deliveries"][0]))

    def test_a_delivery_record_is_not_evidence_that_the_action_worked(self):
        a = self.action("e1")                                                      # an action whose read-back failed
        self.led.mark_spoken("action", a, "x", "model_note")
        self.assertEqual(self.led.actions[a]["verification"]["state"], L.V_FAILED)
        self.assertIsNotNone(SP.action_line(self.led.actions[a]))                  # still bad news, delivered or not
        self.assertTrue(self.led.actions[a]["deliveries"])

    def test_12_a_lost_action_still_ends_unverified_after_f08_and_deliveries_do_not_disturb_it(self):
        aid = self.led.action_seen("r1", self.ep, "Bash")
        self.led.lose_epoch_actions(self.ep, "closed")
        self.assertEqual((self.led.actions[aid]["state"], self.led.actions[aid]["verification"]["state"]),
                         (L.A_LOST, L.V_UNVERIFIED))
        self.p.scan_undelivered(); self.note()
        self.assertEqual(self.led.lose_epoch_actions(self.ep, "again"), 0)
        self.assertEqual(self.led.actions[aid]["verification"]["state"], L.V_UNVERIFIED)
        self.assertEqual(len(self.led.actions[aid]["deliveries"]), 1)


class EpochTests(LedgerCase):
    def test_13_records_from_different_epochs_do_not_leak_into_each_other(self):
        e2 = self.led.epoch_open("s2", 2)
        self.task("old", epoch=self.ep); self.task("new", epoch=e2)
        self.led.mark_spoken("task", "old", "x", "model_note")                      # current epoch is e2
        self.assertEqual(self.led.tasks["old"]["deliveries"][0]["epoch"], e2)       # the epoch it was handed off in
        self.assertEqual(self.led.tasks["old"]["epoch"], self.ep)                   # the task's own epoch is unchanged
        self.p.scan_undelivered()
        self.assertEqual(self.queued(), [(SP.BG_COMPLETION, "new")])                # a delivery in one does not hide the other


class WindowTests(LedgerCase):
    def at_age(self, age):
        self.task("a")                                                              # finished at T0
        self.t[0] = T0 + age
        self.p.scan_undelivered()
        return len(self.p.deferred)

    def test_14_inside_the_window_is_eligible(self):
        self.assertEqual(self.at_age(DAY - 1), 1)

    def test_15_exactly_at_the_boundary_is_eligible_and_one_millisecond_past_it_is_not(self):
        self.assertEqual(self.at_age(DAY), 1)                                      # inclusive: age <= 24 h
        self.p.deferred.clear()
        self.t[0] = T0 + DAY + 0.001
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_16_older_than_the_window_is_excluded(self):
        self.assertEqual(self.at_age(DAY + 3600), 0)
        self.assertEqual(self.at_age(30 * DAY), 0)

    def test_the_window_uses_the_durable_timestamp_not_when_the_process_started(self):
        self.task("a")
        self.t[0] = T0 + DAY - 10
        led = self.reopen()                                                         # a "restart" 10 s before it ages out
        self.p.scan_undelivered()
        self.assertEqual(len(self.p.deferred), 1)
        self.t[0] = T0 + DAY + 10
        self.p.deferred.clear()
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_a_row_with_no_durable_timestamp_is_not_scanned(self):
        self.task("a")
        self.led.tasks["a"]["history"] = []
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_a_verification_that_came_later_moves_the_outcome_time_forward(self):
        a = self.action("v1", error=False)
        self.t[0] = T0 + DAY - 5
        self.led.set_verification(a, L.V_FAILED, L.LEVEL_READBACK, "gone", "file")
        self.t[0] = T0 + DAY + 100                                                  # 100 s after the verdict, > 24 h after the call
        self.p.scan_undelivered()
        self.assertEqual(len(self.p.deferred), 1)

    def test_the_window_applies_to_actions_too(self):
        self.action("e1")
        self.t[0] = T0 + DAY
        self.p.scan_undelivered()
        self.assertEqual(len(self.p.deferred), 1)                                  # inclusive boundary
        self.p.deferred.clear(); self.p._bad_news.clear()
        self.t[0] = T0 + DAY + 1
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_when_more_are_eligible_than_the_limit_the_newest_win_and_queue_in_order(self):
        for i in range(6):
            self.t[0] = T0 + i * 60
            self.task(f"t{i}")
        self.p.scan_undelivered(limit=3)
        self.assertEqual([c.source_id for c in self.p.deferred], ["t3", "t4", "t5"])
        self.assertEqual(self.p.last_scan["eligible"], 6)

    def test_a_task_that_has_not_finished_is_never_an_undelivered_outcome(self):
        self.led.task_seen("run", self.ep, kind=L.KIND_BG_AGENT, description="still going")
        self.led.task_seen("run2", self.ep, kind=L.KIND_BG_AGENT, description="going")
        self.led.task_state("run2", L.K_RUNNING)
        self.p.scan_undelivered()
        self.assertEqual(self.p.deferred, [])

    def test_17_the_scan_is_bounded_however_large_the_history(self):
        for i in range(3000):
            self.task(f"old{i}")
        self.t[0] = T0 + 10 * DAY
        for i in range(25):
            self.task(f"new{i}")
        self.p.scan_undelivered(max_rows=200, limit=7)
        self.assertEqual(len(self.p.deferred), 7)
        self.assertLessEqual(self.p.last_scan["examined"], 200)
        self.assertTrue(all(c.source_id.startswith("new") for c in self.p.deferred))
        n = self.p.last_scan["examined"]
        self.p.deferred.clear()
        self.p.scan_undelivered()                                                   # the defaults are bounded too
        self.assertLessEqual(self.p.last_scan["examined"], SP.SCAN_MAX_ROWS)
        self.assertLessEqual(len(self.p.deferred), SP.SCAN_LIMIT)
        self.assertGreater(n, 0)


class DedupeReconstructionTests(LedgerCase):
    def test_18_durable_records_rebuild_the_dedupe_state_so_nothing_is_replayed(self):
        self.task("a"); a = self.action("e1")
        self.p.scan_undelivered()
        self.note()                                                                 # handed off: recorded
        self.reopen()
        self.assertEqual(self.p.scan_undelivered(), 0)
        self.p.on_action_bad_news(self.led.actions[a])                              # the live path asks again after restart
        self.assertEqual(self.p.deferred, [])
        self.assertEqual(self.note(), "")
        self.assertEqual([len(self.led.tasks["a"]["deliveries"]), len(self.led.actions[a]["deliveries"])], [1, 1])


class StartupTests(BrainBase):
    """The scan runs when a brain opens an epoch, after reconcile, so what a crash left behind is found."""

    async def test_20_a_prior_runs_unreported_outcomes_ride_the_first_note_and_only_once(self):
        from backtalk import brain as B
        from tests.router_fakes import settle
        with tempfile.TemporaryDirectory() as d:
            led = L.Ledger(d, fsync=False)
            ep = led.epoch_open("s", 1)                                             # never closed: a crash
            led.task_seen("done", ep, kind=L.KIND_BG_AGENT, description="export notes")
            led.task_state("done", L.K_RUNNING); led.task_state("done", L.K_COMPLETED, cause="w", terminal_at=time.time())
            led.task_seen("mid", ep, kind=L.KIND_BG_AGENT, description="long crawl")
            led.task_state("mid", L.K_RUNNING)
            led.close()
            led = L.Ledger(d, fsync=False)
            brain = B.WarmBrain(model="m", ledger=led)
            await brain.start()
            await settle()
            note = brain.note_prefix()
            self.assertIn("export notes finished", note)
            self.assertIn("can't tell whether long crawl finished", note)
            self.assertEqual(brain.note_prefix(), "")
            self.assertEqual(len(led.tasks["done"]["deliveries"]), 1)
            r = brain.router
            for t in (r._pump_task, r._watch_task):
                t.cancel()
            await asyncio.sleep(0)
            led.close()
