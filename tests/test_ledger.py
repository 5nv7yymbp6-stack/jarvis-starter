"""Phase 1: the Action Ledger (backtalk/ledger.py). Stdlib unittest."""
import json
import os
import tempfile
import unittest

from backtalk import ledger as L
from backtalk.ledger import Ledger


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.led = Ledger(self.dir)

    def tearDown(self):
        self.led.close()
        self.tmp.cleanup()

    def reopen(self):
        self.led.close()
        self.led = Ledger(self.dir)
        return self.led

    def lines(self):
        with open(os.path.join(self.dir, "ledger.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]


class PersistenceTests(Base):
    def test_every_event_is_one_appended_line_and_seq_increases(self):
        e = self.led.epoch_open("s1", 123)
        self.led.epoch_closing(e, "test")
        seqs = [r["seq"] for r in self.lines()]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), len(set(seqs)))

    def test_fsync_is_called_per_event(self):
        calls = []
        real = os.fsync
        os.fsync = lambda fd: calls.append(fd) or real(fd)
        try:
            n0 = len(calls)
            self.led.epoch_open("s1", 1)
            self.assertGreater(len(calls), n0)
        finally:
            os.fsync = real

    def test_replay_restores_identical_state(self):
        e = self.led.epoch_open("s1", 1)
        t = self.led.turn_new(e, "foreground", "u1", "hello")
        self.led.turn_state(t, L.T_SENT)
        self.led.action_seen("toolu_1", e, "Bash", turn_id=t, intent="ls")
        self.led.action_result("toolu_1", is_error=False, content="out")
        self.led.task_seen("task1", e, kind=L.KIND_BG_AGENT, tool_use_id="toolu_2",
                           description="bg")
        before = (json.dumps(self.led.turns, sort_keys=True),
                  json.dumps(self.led.actions, sort_keys=True),
                  json.dumps(self.led.tasks, sort_keys=True),
                  json.dumps(self.led.epochs, sort_keys=True))
        led2 = self.reopen()
        after = (json.dumps(led2.turns, sort_keys=True),
                 json.dumps(led2.actions, sort_keys=True),
                 json.dumps(led2.tasks, sort_keys=True),
                 json.dumps(led2.epochs, sort_keys=True))
        self.assertEqual(before, after)
        self.assertEqual(led2.action_id_for("toolu_1"), self.led.action_id_for("toolu_1"))

    def test_torn_last_line_after_a_crash_is_skipped_not_fatal(self):
        e = self.led.epoch_open("s1", 1)
        self.led.close()
        with open(os.path.join(self.dir, "ledger.jsonl"), "a") as f:
            f.write('{"t": "task", "id": "half')          # power cut mid-write
        led2 = Ledger(self.dir)
        self.assertEqual(led2.corrupt_lines, 1)
        self.assertIn(e, led2.epochs)
        led2.close()

    def test_result_bodies_are_never_stored(self):
        e = self.led.epoch_open("s1", 1)
        self.led.action_seen("toolu_1", e, "mcp__x__send_message")
        self.led.action_result("toolu_1", is_error=False, content="TOP SECRET BODY")
        raw = open(os.path.join(self.dir, "ledger.jsonl")).read()
        self.assertNotIn("TOP SECRET BODY", raw)
        r = self.led.actions[self.led.action_id_for("toolu_1")]["result"]
        self.assertEqual(r["bytes"], len("TOP SECRET BODY"))

    def test_intent_is_redacted_and_truncated(self):
        e = self.led.epoch_open("s1", 1)
        t = self.led.turn_new(e, "foreground", "u", "curl -H 'Authorization: Bearer abc123SECRET' " + "x" * 500)
        s = self.led.turns[t]["intent"]
        self.assertNotIn("abc123SECRET", s)
        self.assertLessEqual(len(s), 210)


class TransitionTests(Base):
    def setUp(self):
        super().setUp()
        self.e = self.led.epoch_open("s1", 1)

    def test_turn_lifecycle_and_terminal_states_are_sticky(self):
        t = self.led.turn_new(self.e, "foreground", "u")
        for s in (L.T_SENT, L.T_CLAIMED, L.T_RUNNING, L.T_RESULT, L.T_DONE):
            self.assertTrue(self.led.turn_state(t, s), s)
        self.assertFalse(self.led.turn_state(t, L.T_RUNNING))     # no going back
        self.assertEqual(self.led.turns[t]["state"], L.T_DONE)

    def test_refused_transition_is_recorded_as_a_note(self):
        t = self.led.turn_new(self.e, "foreground", "u")
        self.led.turn_state(t, L.T_SENT)
        self.assertFalse(self.led.turn_state(t, L.T_RESULT))      # skipped CLAIMED
        self.assertTrue(any(r.get("note") == "refused_transition" for r in self.lines()))

    def test_interrupted_turn_stays_interrupted(self):
        t = self.led.turn_new(self.e, "foreground", "u")
        for s in (L.T_SENT, L.T_CLAIMED, L.T_RUNNING, L.T_INTERRUPTED):
            self.led.turn_state(t, s)
        self.assertFalse(self.led.turn_state(t, L.T_DONE))
        self.assertEqual(self.led.turns[t]["state"], L.T_INTERRUPTED)

    def test_action_has_no_executed_state(self):
        self.assertFalse(hasattr(L, "A_EXECUTED"))
        all_states = {L.A_REQUESTED, L.A_ALLOWED, L.A_DENIED, L.A_OK, L.A_ERROR, L.A_LOST, L.A_ABORTED}
        self.assertNotIn("EXECUTED", all_states)

    def test_a_requested_action_has_not_returned(self):
        self.led.action_seen("toolu_1", self.e, "Bash")
        a = self.led.actions[self.led.action_id_for("toolu_1")]
        self.assertEqual(a["state"], L.A_REQUESTED)
        self.assertNotIn("returned_at", a)

    def test_action_seen_is_idempotent(self):
        a1 = self.led.action_seen("toolu_1", self.e, "Bash")
        a2 = self.led.action_seen("toolu_1", self.e, "Bash", turn_id="T1")
        self.assertEqual(a1, a2)
        self.assertEqual(len([a for a in self.led.actions.values()]), 1)

    def test_a_legacy_denied_row_does_not_become_returned(self):
        aid = self.led.action_seen("toolu_1", self.e, "Bash")
        self.led.action_state(aid, L.A_DENIED)            # nothing writes this any more; an older ledger can hold it
        self.led.action_result("toolu_1", is_error=True, content="Blocked")
        a = self.led.actions[self.led.action_id_for("toolu_1")]
        self.assertEqual(a["state"], L.A_DENIED)

    def test_task_lifecycle_and_stickiness(self):
        self.led.task_seen("t1", self.e, kind=L.KIND_BG_AGENT, description="bg")
        self.assertTrue(self.led.task_state("t1", L.K_RUNNING))
        self.assertTrue(self.led.task_state("t1", L.K_COMPLETED))
        self.assertFalse(self.led.task_state("t1", L.K_RUNNING))
        self.assertFalse(self.led.task_state("t1", L.K_LOST_REBUILD))
        self.assertEqual(self.led.tasks["t1"]["state"], L.K_COMPLETED)

    def test_cancel_intent_is_recorded_before_effect(self):
        self.led.task_seen("t1", self.e, kind=L.KIND_BG_AGENT)
        self.led.task_cancel_intent("t1", "USER_STOP")
        self.assertEqual(self.led.tasks["t1"]["cancel_intent"], "USER_STOP")
        self.assertEqual(self.led.tasks["t1"]["state"], L.K_LAUNCHED)

    def test_live_tasks_filters(self):
        self.led.task_seen("a", self.e, kind=L.KIND_BG_AGENT)
        self.led.task_seen("b", self.e, kind=L.KIND_SUBAGENT)
        self.led.task_state("a", L.K_RUNNING)
        self.assertEqual({t["id"] for t in self.led.live_tasks()}, {"a", "b"})
        self.assertEqual({t["id"] for t in self.led.live_tasks(kind=L.KIND_BG_AGENT)}, {"a"})
        self.led.task_state("a", L.K_COMPLETED)
        self.assertEqual({t["id"] for t in self.led.live_tasks(kind=L.KIND_BG_AGENT)}, set())


class VerificationTests(Base):
    def setUp(self):
        super().setUp()
        self.e = self.led.epoch_open("s1", 1)
        self.aid = self.led.action_seen("toolu_1", self.e, "Write")

    def v(self):
        return self.led.actions[self.aid]["verification"]

    def test_starts_pending(self):
        self.assertEqual(self.v()["state"], L.V_PENDING)

    def test_self_report_can_never_be_verified(self):
        self.led.set_verification(self.aid, L.V_VERIFIED, L.LEVEL_SELF, "tool said ok")
        self.assertEqual(self.v()["state"], L.V_UNVERIFIED)
        self.assertIn("self-report", self.v()["evidence"])

    def test_no_level_can_never_be_verified(self):
        self.led.set_verification(self.aid, L.V_VERIFIED, L.LEVEL_NONE, "")
        self.assertEqual(self.v()["state"], L.V_UNVERIFIED)

    def test_read_back_can_be_verified(self):
        self.led.action_result("toolu_1", is_error=False, content="ok")          # F-12: only an action that returned OK can be VERIFIED
        self.led.set_verification(self.aid, L.V_VERIFIED, L.LEVEL_READBACK, "file hash matches", "file")
        self.assertEqual(self.v()["state"], L.V_VERIFIED)

    def test_read_back_can_fail(self):
        self.led.set_verification(self.aid, L.V_FAILED, L.LEVEL_READBACK, "hash differs")
        self.assertEqual(self.v()["state"], L.V_FAILED)


class ReconcileTests(Base):
    def crashed_ledger(self):
        e = self.led.epoch_open("s1", 4242)
        t = self.led.turn_new(e, "foreground", "u")
        self.led.turn_state(t, L.T_SENT)
        self.led.action_seen("toolu_1", e, "Bash", turn_id=t)
        self.led.task_seen("bg1", e, kind=L.KIND_BG_AGENT, tool_use_id="toolu_9", description="job")
        self.led.task_state("bg1", L.K_RUNNING)
        self.led.task_set("bg1", process_snapshot=[{"pid": 111, "lstart": "Wed Sep 30 01:00:00 2026"}])
        self.led.task_seen("done1", e, kind=L.KIND_BG_AGENT)
        self.led.task_state("done1", L.K_COMPLETED)
        return e

    def test_open_epoch_at_startup_is_a_crash(self):
        e = self.crashed_ledger()
        led2 = self.reopen()
        out = led2.reconcile()
        self.assertEqual(led2.epochs[e]["state"], L.EPOCH_CRASHED)
        self.assertEqual(out["crashed_epochs"], 1)

    def test_crash_never_records_a_live_task_as_stopped(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile()
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_UNKNOWN_CRASH)
        self.assertNotIn(led2.tasks["bg1"]["state"],
                         (L.K_STOPPED_USER, L.K_STOPPED_UNATTRIBUTED, L.K_LOST_REBUILD))
        self.assertFalse(led2.tasks["bg1"]["outcome_known"])

    def test_completed_task_survives_a_crash_untouched(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile()
        self.assertEqual(led2.tasks["done1"]["state"], L.K_COMPLETED)

    def test_crash_loses_unreturned_actions_and_aborts_turns(self):
        e = self.crashed_ledger()
        led2 = self.reopen()
        out = led2.reconcile()
        self.assertEqual(led2.actions[led2.action_id_for("toolu_1")]["state"], L.A_LOST)
        self.assertEqual(out["actions_lost"], 1)
        self.assertTrue(all(t["state"] == L.T_ABORTED for t in led2.turns.values()))

    def test_closing_epoch_is_also_a_crash(self):
        e = self.crashed_ledger()
        self.led.epoch_closing(e, "rebuild")
        led2 = self.reopen()
        led2.reconcile()
        self.assertEqual(led2.epochs[e]["state"], L.EPOCH_CRASHED)
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_UNKNOWN_CRASH)

    def test_clean_epoch_leftover_live_task_is_lost_on_rebuild(self):
        e = self.crashed_ledger()
        self.led.epoch_closing(e, "rebuild")
        self.led.epoch_closed_clean(e)
        led2 = self.reopen()
        out = led2.reconcile()
        self.assertEqual(led2.epochs[e]["state"], L.EPOCH_CLOSED_CLEAN)
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_LOST_REBUILD)
        self.assertEqual(out["tasks_lost"], 1)

    def test_reconcile_is_idempotent(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile()
        n = len(led2.notes)
        seq = led2._seq
        out2 = led2.reconcile()
        self.assertEqual(sum(out2.values()), 0)
        self.assertEqual(led2._seq, seq)

    def test_process_evidence_alive_refines_to_orphan_running(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile(proc_check=lambda pid, ls: True)
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_ORPHAN_RUNNING)
        self.assertEqual(led2.tasks["bg1"]["process_evidence"], "ALIVE")

    def test_process_evidence_dead_is_still_outcome_unknown(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile(proc_check=lambda pid, ls: False)
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_DEAD_UNKNOWN)
        self.assertFalse(led2.tasks["bg1"]["outcome_known"])

    def test_uncheckable_process_stays_unknown_after_crash(self):
        self.crashed_ledger()
        led2 = self.reopen()
        led2.reconcile(proc_check=lambda pid, ls: None)
        self.assertEqual(led2.tasks["bg1"]["state"], L.K_UNKNOWN_CRASH)

    def test_only_recorded_processes_are_ever_checked(self):
        e = self.led.epoch_open("s1", 1)
        self.led.task_seen("nosnap", e, kind=L.KIND_BG_AGENT)
        led2 = self.reopen()
        seen = []
        led2.reconcile(proc_check=lambda pid, ls: seen.append(pid) or True)
        self.assertEqual(seen, [])                       # nothing recorded, nothing probed
        self.assertEqual(led2.tasks["nosnap"]["state"], L.K_UNKNOWN_CRASH)

    def test_process_alive_helper_never_signals(self):
        # read-only: a nonexistent pid is simply "not there"
        self.assertIs(L.process_alive(999999, "Wed Sep 30 01:00:00 2026"), False)
        # our own pid is alive but with a different start time -> not the same process
        self.assertIs(L.process_alive(os.getpid(), "Thu Jan  1 00:00:00 1970"), False)


if __name__ == "__main__":
    unittest.main()
