"""Typed task state, failure classes and recovery decisions (backtalk/taskstate.py), the history CLI and the live wiring."""
import ast
import inspect
import io
import itertools
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from backtalk import ledger as L
from backtalk import main as M
from backtalk import speech_policy as SP
from backtalk import tasks as TASKS
from backtalk import taskstate as T

T0 = 1_800_000_000.0


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        self.t += 1
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.led = L.Ledger(self.tmp.name, fsync=False, clock=self.clock)
        self.n = 0

    def tearDown(self):
        self.led.close()
        self.tmp.cleanup()

    def put(self, table, state, **fields):
        """A row in exactly `state`, written through the ledger's own append so replay sees it."""
        self.n += 1
        rid = f"{table[0].upper()}{self.n}"
        self.led._append({"t": table, "id": rid, "to": state, **fields})
        return rid

    def turn(self, state, cause="", **kw):
        rid = self.put("turn", state, kind="foreground", epoch="E1", intent="x", silent=False, **kw)
        if cause:
            self.led._append({"t": "turn", "id": rid, "to": state, "cause": cause})
        return rid

    def action(self, state, tool="Bash", turn_id=None, verification=None, epoch="E1", **kw):
        return self.put("action", state, tool=tool, turn_id=turn_id, epoch=epoch,
                        verification={"state": verification or L.V_NOT_REQUIRED, "level": "NONE", "evidence": "", "verifier": ""},
                        **kw)

    def task(self, state, rv=None, **kw):
        return self.put("task", state, task_kind=L.KIND_BG_AGENT, epoch="E1", description="job",
                        **({"result_verification": {"state": rv}} if rv else {}), **kw)


class ExhaustivenessTests(Base):
    """Adding a ledger state without mapping it must fail here, not in production."""

    def constants(self, prefix, skip=()):
        return {v for k, v in vars(L).items() if k.startswith(prefix) and isinstance(v, str) and k not in skip}

    def test_every_turn_state_maps(self):
        states = self.constants("T_") - {"TURN_TERMINAL"}
        self.assertGreaterEqual(len(states), 9)
        for st in states:
            v = T.turn_view(self.led, self.turn(st))
            self.assertIn(v.status, T.STATUSES, st)

    def test_every_action_state_maps(self):
        states = self.constants("A_")
        self.assertGreaterEqual(len(states), 7)
        for st in states:
            v = T.action_view(self.led, self.action(st))
            self.assertIn(v.status, T.STATUSES, st)

    def test_every_task_state_maps(self):
        states = {v for k, v in vars(L).items() if k.startswith("K_") and isinstance(v, str)}
        self.assertGreaterEqual(len(states), 10)
        for st in states:
            v = T.task_view(self.led, self.task(st))
            self.assertIn(v.status, T.STATUSES, st)

    def test_the_waiting_permission_status_is_never_produced(self):
        seen = set()
        for st in self.constants("T_") - {"TURN_TERMINAL"}:
            seen.add(T.turn_view(self.led, self.turn(st)).status)
        for st in self.constants("A_"):
            seen.add(T.action_view(self.led, self.action(st)).status)
        for k, st in vars(L).items():
            if k.startswith("K_") and isinstance(st, str):
                seen.add(T.task_view(self.led, self.task(st)).status)
        self.assertNotIn(T.WAITING_PERMISSION, seen)

    def test_the_real_ledger_maps_completely(self):
        real = L.Ledger("ledger", fsync=False) if os.path.exists("ledger/ledger.jsonl") else None
        if real is None:
            self.skipTest("no live ledger on this machine")
        for kind, tbl in (("turn", real.turns), ("action", real.actions), ("task", real.tasks)):
            for ident in tbl:
                self.assertIn(T.view_of(real, kind, ident).status, T.STATUSES)


class MappingTests(Base):
    def test_turn_states(self):
        want = {L.T_CREATED: T.PLANNED, L.T_SENT: T.WAITING_RESULT, L.T_CLAIMED: T.RUNNING, L.T_RUNNING: T.RUNNING,
                L.T_RESULT: T.RUNNING, L.T_DONE: T.SUCCEEDED, L.T_FAILED: T.FAILED, L.T_INTERRUPTED: T.CANCELLED,
                L.T_ABORTED: T.CANCELLED}
        for st, status in want.items():
            self.assertEqual(T.turn_view(self.led, self.turn(st)).status, status, st)

    def test_action_states_and_verification(self):
        self.assertEqual(T.action_view(self.led, self.action(L.A_REQUESTED)).status, T.WAITING_RESULT)
        self.assertEqual(T.action_view(self.led, self.action(L.A_ERROR)).status, T.FAILED)
        self.assertEqual(T.action_view(self.led, self.action(L.A_LOST)).status, T.UNKNOWN)
        self.assertEqual(T.action_view(self.led, self.action(L.A_ABORTED)).status, T.CANCELLED)
        pending = T.action_view(self.led, self.action(L.A_OK, verification=L.V_PENDING))
        self.assertEqual(pending.status, T.VERIFYING)
        self.assertFalse(pending.claim_allowed)

    def test_an_action_whose_check_says_it_did_not_land_is_failed_not_succeeded(self):
        v = T.action_view(self.led, self.action(L.A_OK, tool="Write", verification=L.V_FAILED))
        self.assertEqual((v.status, v.execution, v.failure_class), (T.FAILED, "FAILURE", T.FC_UNKNOWN))
        self.assertEqual(v.recovery.action, T.CHECK_FIRST)                    # the effect is unknown: never a blind retry

    def test_a_claim_may_be_reported_only_when_verified_or_not_required(self):
        for rv, allowed in ((L.V_VERIFIED, True), (L.V_NOT_REQUIRED, True), (L.V_UNVERIFIED, False), (None, False)):
            v = T.task_view(self.led, self.task(L.K_COMPLETED, rv=rv))
            self.assertEqual((v.status, v.claim_allowed), (T.SUCCEEDED, allowed), rv)
            a = T.action_view(self.led, self.action(L.A_OK, verification=rv or L.V_UNVERIFIED))
            self.assertEqual(a.claim_allowed, allowed if rv else False, rv)

    def test_a_task_that_contradicts_its_own_evidence_is_partial_and_reports_nothing_as_fact(self):
        v = T.task_view(self.led, self.task(L.K_COMPLETED, rv=L.V_FAILED))
        self.assertEqual((v.status, v.execution, v.claim_allowed, v.failure_class), (T.PARTIAL, "SUCCESS", False, T.PARTIAL_SUCCESS))

    def test_a_task_still_being_checked_is_verifying(self):
        self.assertEqual(T.task_view(self.led, self.task(L.K_COMPLETED, rv=L.V_PENDING)).status, T.VERIFYING)

    # roadmap: cancellation semantics
    def test_cancellation_causes_are_distinct(self):
        cases = {L.K_STOPPED_USER: "user", L.K_KILLED_INTERRUPT: "interrupt", L.K_STOPPED_UNATTRIBUTED: "unattributed"}
        for st, cause in cases.items():
            v = T.task_view(self.led, self.task(st))
            self.assertEqual((v.status, v.execution, v.cancel_cause), (T.CANCELLED, "CANCELLED", cause), st)
            self.assertEqual(v.recovery.action, T.NONE)                       # a cancellation is not a failure to recover from
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_INTERRUPTED)).cancel_cause, "interrupt")
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_ABORTED)).cancel_cause, "aborted")

    def test_outcomes_we_cannot_know_are_unknown_never_success(self):
        for st in (L.K_LOST_REBUILD, L.K_UNKNOWN_CRASH, L.K_DEAD_UNKNOWN, L.K_ORPHAN_RUNNING):
            v = T.task_view(self.led, self.task(st))
            self.assertEqual((v.status, v.recovery.action), (T.UNKNOWN, T.CHECK_FIRST), st)
        self.assertTrue(T.task_view(self.led, self.task(L.K_ORPHAN_RUNNING)).maybe_running)
        self.assertFalse(T.task_view(self.led, self.task(L.K_LOST_REBUILD)).maybe_running)

    def test_reported_reflects_a_recorded_hand_off(self):
        tid = self.task(L.K_COMPLETED, rv=L.V_VERIFIED)
        self.assertFalse(T.task_view(self.led, tid).reported)
        self.led.mark_spoken("task", tid, "x", "spoken")
        self.assertTrue(T.task_view(self.led, tid).reported)

    def test_times_and_parent_links(self):
        tn = self.turn(L.T_DONE)
        a = self.action(L.A_OK, turn_id=tn)
        v = T.action_view(self.led, a)
        self.assertEqual(v.parent, tn)
        self.assertLess(v.started_at, v.ended_at + 1)


class ClassificationTests(unittest.TestCase):
    def test_tool_error_families(self):
        table = [
            ("Operation timed out after 30s", T.RETRYABLE), ("HTTP 503 Service Unavailable", T.RETRYABLE),
            ("rate limit exceeded, try again later", T.RETRYABLE), ("ECONNRESET", T.RETRYABLE),
            ("Permission denied: '/etc/hosts'", T.PERMISSION_REQUIRED), ("operation not permitted", T.PERMISSION_REQUIRED),
            ("HTTP 403 Forbidden", T.PERMISSION_REQUIRED),
            ("No such file or directory: 'x'", T.INVALID_INPUT), ("syntax error near unexpected token", T.INVALID_INPUT),
            ("usage: git commit [options]", T.INVALID_INPUT),
            ("please sign in to continue", T.USER_ACTION_REQUIRED), ("your session expired", T.USER_ACTION_REQUIRED),
            ("something exploded", T.TOOL_ERROR), ("", T.TOOL_ERROR), (None, T.TOOL_ERROR),
            # review 2026-10-19, finding 5
            ("file /tmp/network_log.txt: No such file", T.INVALID_INPUT), ("network error: host unreachable", T.RETRYABLE),
            ("401 Unauthorized", T.PERMISSION_REQUIRED), ("HTTP 500 Internal Server Error", T.RETRYABLE),
            ("HTTP 501", T.RETRYABLE), ("sudo: a password is required", T.USER_ACTION_REQUIRED),
            ("login: command not found", T.INVALID_INPUT), ("please log in to continue", T.USER_ACTION_REQUIRED),
            ("saw 5021 items", T.TOOL_ERROR), ("port 4290 open", T.TOOL_ERROR),
        ]
        for text, want in table:
            self.assertEqual(T.classify_tool_error("Bash", text), want, text)

    def test_precedence_user_action_then_permission_then_retry_then_input(self):
        self.assertEqual(T.classify_tool_error("x", "sign in required; permission denied"), T.USER_ACTION_REQUIRED)
        self.assertEqual(T.classify_tool_error("x", "permission denied, try again"), T.PERMISSION_REQUIRED)
        self.assertEqual(T.classify_tool_error("x", "timed out: file not found"), T.RETRYABLE)

    def test_block_list_content_and_hostile_input(self):
        self.assertEqual(T.classify_tool_error("x", [{"type": "text", "text": "Permission denied"}]), T.PERMISSION_REQUIRED)
        self.assertEqual(T.classify_tool_error("x", [object(), 5]), T.TOOL_ERROR)
        self.assertEqual(T.classify_tool_error("x", "a" * 10_000_000 + " timed out"), T.TOOL_ERROR)       # only the head is read

    def test_turn_causes(self):
        want = {"no_echo": T.RETRYABLE, "send_failed:OSError": T.RETRYABLE, "api_error": T.RETRYABLE,
                "stream_lost": T.RETRYABLE, "oversize": T.USER_ACTION_REQUIRED, "max_turns": T.PARTIAL_SUCCESS,
                "error_max_turns": T.PARTIAL_SUCCESS, "something_new": T.FC_UNKNOWN, "": T.FC_UNKNOWN, None: T.FC_UNKNOWN}
        for cause, fc in want.items():
            self.assertEqual(T.classify_turn_cause(cause), fc, cause)

    def test_nothing_ran_only_for_the_causes_where_the_query_never_started(self):
        for c in ("no_echo", "send_failed", "send_failed:TimeoutError"):
            self.assertTrue(T.nothing_ran(c), c)
        for c in ("api_error", "stream_lost", "oversize", "max_turns", "", None):
            self.assertFalse(T.nothing_ran(c), c)

    def test_read_only_tools_are_a_closed_list_and_unknown_tools_have_side_effects(self):
        for t in ("Read", "Glob", "Grep", "WebFetch"):
            self.assertTrue(T.is_read_only(t))
        for t in ("Bash", "Write", "Edit", "mcp__stv__tv_play", "mcp__playwright__browser_click", "Agent", "NewTool",
                  "Skill", None):
            self.assertFalse(T.is_read_only(t), t)
        for tier in (0, 1, 2, 3, None):                                      # the legacy tier is IGNORED: 35 of 60 Bash rows say 0
            self.assertFalse(T.is_read_only("Bash", tier=tier), tier)


class RecoveryTests(unittest.TestCase):
    def test_only_failed_partial_and_unknown_need_a_recovery(self):
        for st in (T.PLANNED, T.RUNNING, T.WAITING_RESULT, T.VERIFYING, T.SUCCEEDED, T.CANCELLED):
            self.assertEqual(T.recover(st, T.RETRYABLE, side_effects=False), T.Recovery(T.NONE, "not_failed"), st)

    def test_a_retry_is_recommended_only_when_nothing_with_side_effects_could_have_run(self):
        r = T.recover(T.FAILED, T.RETRYABLE, side_effects=False)
        self.assertEqual((r.action, r.safe_to_retry), (T.RETRY_ONCE, True))
        for se in (True, None):
            r = T.recover(T.FAILED, T.RETRYABLE, side_effects=se)
            self.assertEqual((r.action, r.safe_to_retry), (T.CHECK_FIRST, False), se)

    def test_never_a_blind_retry_for_anything_that_may_have_side_effects_in_any_class(self):
        for st, fc, se, streak, retried in itertools.product((T.FAILED, T.PARTIAL, T.UNKNOWN), T.FAILURE_CLASSES + (None,),
                                                             (True, None), (0, 1, 5), (False, True)):
            r = T.recover(st, fc, side_effects=se, recent_same_cause=streak, retried=retried)
            self.assertNotEqual(r.action, T.RETRY_ONCE, (st, fc, se, streak, retried))
            self.assertFalse(r.safe_to_retry)

    def test_a_retry_happens_at_most_once_and_never_after_the_same_cause_repeated(self):
        self.assertEqual(T.recover(T.FAILED, T.RETRYABLE, side_effects=False, retried=True).action, T.ASK_USER)
        self.assertEqual(T.recover(T.FAILED, T.RETRYABLE, side_effects=False, recent_same_cause=T.MAX_SAME_CAUSE).action, T.ASK_USER)
        self.assertEqual(T.recover(T.FAILED, T.RETRYABLE, side_effects=False, recent_same_cause=0).action, T.RETRY_ONCE)
        self.assertEqual(T.recover(T.FAILED, T.RETRYABLE, side_effects=False, recent_same_cause=1).action, T.ASK_USER)

    def test_classes_that_need_the_user(self):
        for fc in (T.USER_ACTION_REQUIRED, T.PERMISSION_REQUIRED, T.INVALID_INPUT, T.PARTIAL_SUCCESS):
            self.assertEqual(T.recover(T.FAILED, fc, side_effects=False).action, T.ASK_USER, fc)

    def test_a_plain_tool_error_is_reported_and_an_unknown_outcome_is_checked_first(self):
        self.assertEqual(T.recover(T.FAILED, T.TOOL_ERROR, side_effects=True).action, T.REPORT)
        self.assertEqual(T.recover(T.UNKNOWN, None, side_effects=False).action, T.CHECK_FIRST)
        self.assertEqual(T.recover(T.FAILED, None, side_effects=False).action, T.REPORT)
        self.assertEqual(T.recover(T.FAILED, None, side_effects=True).action, T.CHECK_FIRST)

    def test_every_recovery_action_is_a_known_code(self):
        for st, fc, se in itertools.product(T.STATUSES, T.FAILURE_CLASSES + (None,), (True, False, None)):
            self.assertIn(T.recover(st, fc, side_effects=se).action, T.RECOVERY_ACTIONS)


class TurnViewTests(Base):
    def test_a_turn_that_never_started_is_safe_to_retry(self):
        v = T.turn_view(self.led, self.turn(L.T_FAILED, "no_echo"))
        self.assertEqual((v.failure_class, v.side_effects, v.recovery.action), (T.RETRYABLE, False, T.RETRY_ONCE))

    def test_a_stream_lost_is_never_assumed_to_have_run_nothing(self):
        # a tool_use can be lost with the stream before its action row exists, so stream_lost is unknown, not "nothing ran"
        for acts in ((), ("Read",), ("Write",)):
            self.turn(L.T_DONE)                                                  # a success between, so no streak
            tid = self.turn(L.T_FAILED, "stream_lost")
            for tool in acts:
                self.action(L.A_OK, tool=tool, turn_id=tid)
            v = T.turn_view(self.led, tid)
            self.assertEqual((v.side_effects, v.recovery.action), (None, T.CHECK_FIRST), acts)

    def test_an_api_error_after_only_reads_may_be_retried_once(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Read", turn_id=tid)
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.RETRY_ONCE)

    def test_an_api_error_after_a_write_must_be_checked_not_retried(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Write", turn_id=tid)
        v = T.turn_view(self.led, tid)
        self.assertEqual((v.side_effects, v.recovery.action, v.recovery.reason), (True, T.CHECK_FIRST, "retryable_but_may_have_run"))

    # review 2026-10-19, finding 1: a legacy tier 0 is not evidence
    def test_a_legacy_tier_zero_bash_action_is_not_read_only(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Bash", turn_id=tid, tier=0)
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.CHECK_FIRST)

    # finding 3: actions the turn link misses
    def test_an_unlinked_action_inside_the_turns_lifetime_counts(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Bash", turn_id=None)                          # the router sets turn_id None when the handle is gone
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.CHECK_FIRST)

    def test_an_unlinked_read_only_action_or_one_from_another_epoch_does_not(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Read", turn_id=None)
        self.put("action", L.A_OK, tool="Bash", turn_id=None, epoch="E9",
                 verification={"state": L.V_NOT_REQUIRED, "level": "NONE", "evidence": "", "verifier": ""})
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.RETRY_ONCE)

    def test_a_subagents_write_counts_through_its_parent_call_even_to_any_depth(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.action(L.A_OK, tool="Agent", turn_id=tid, tool_use_id="tu_parent")
        self.action(L.A_OK, tool="Read", turn_id="Tx", tool_use_id="tu_mid", parent_tool_use_id="tu_parent")
        self.action(L.A_OK, tool="Write", turn_id=None, tool_use_id="tu_leaf", parent_tool_use_id="tu_mid", epoch="E9")
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.CHECK_FIRST)
        self.assertTrue(T.turn_view(self.led, tid).side_effects)

    def test_a_background_task_launched_in_the_turn_counts_even_with_no_action_row(self):
        tid = self.turn(L.T_FAILED, "api_error")
        self.put("task", L.K_RUNNING, task_kind=L.KIND_BG_AGENT, epoch="E1", description="job", launched_in_turn=tid)
        self.assertEqual(T.turn_view(self.led, tid).recovery.action, T.CHECK_FIRST)

    # finding 4
    def test_a_turn_never_claims_a_verified_result(self):
        self.assertFalse(T.turn_view(self.led, self.turn(L.T_DONE)).claim_allowed)

    def test_the_seventeen_api_errors_in_a_row_stop_being_retryable(self):
        views = [T.turn_view(self.led, self.turn(L.T_FAILED, "api_error")) for _ in range(6)]
        self.assertEqual([v.recovery.action for v in views],
                         [T.RETRY_ONCE] + [T.ASK_USER] * 5)                       # once means once
        self.assertEqual(views[-1].recovery.reason, "keeps_failing")

    def test_barge_in_turns_do_not_hide_a_repeating_failure(self):
        self.turn(L.T_FAILED, "api_error")
        self.turn(L.T_INTERRUPTED)
        self.turn(L.T_ABORTED)
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_FAILED, "api_error")).recovery.reason, "keeps_failing")

    def test_causes_of_one_family_count_together_and_old_failures_do_not(self):
        self.turn(L.T_FAILED, "send_failed:OSError")
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_FAILED, "send_failed:TypeError")).recovery.reason, "keeps_failing")
        a = self.turn(L.T_FAILED, "api_error")
        self.clock.t += T.STREAK_GAP_S + 5                                       # ten minutes later it is a separate incident
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_FAILED, "api_error")).recovery.action, T.RETRY_ONCE)

    def test_the_streak_walk_is_fast_on_a_big_ledger(self):
        import time
        for _ in range(20_000):
            self.turn(L.T_DONE)
        tid = self.turn(L.T_FAILED, "api_error")
        t0 = time.perf_counter()
        T.same_cause_streak(self.led, tid)
        self.assertLess(time.perf_counter() - t0, 0.05)                          # walks from the end; no whole-ledger copy

    def test_a_success_in_between_resets_the_streak(self):
        for _ in range(3):
            self.turn(L.T_FAILED, "api_error")
        self.turn(L.T_DONE)
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_FAILED, "api_error")).recovery.action, T.RETRY_ONCE)

    def test_a_different_cause_resets_the_streak(self):
        self.turn(L.T_FAILED, "api_error")
        self.turn(L.T_FAILED, "api_error")
        self.turn(L.T_FAILED, "stream_lost")
        self.assertEqual(T.turn_view(self.led, self.turn(L.T_FAILED, "api_error")).recovery.action, T.RETRY_ONCE)

    def test_a_turn_not_failed_has_no_recovery_and_no_streak_lookup_cost(self):
        v = T.turn_view(self.led, self.turn(L.T_DONE))
        self.assertEqual(v.recovery.action, T.NONE)


class RobustnessTests(Base):
    def test_a_malformed_row_cannot_crash_the_history_view(self):
        self.turn(L.T_DONE)
        rid = self.put("action", L.A_OK, tool="Bash", epoch="E1", verification="not a dict")      # a row with a bad shape
        self.led.actions[rid]["history"] = [5]
        out = T.recent(self.led, 10)
        self.assertTrue(any(v.kind == "turn" for v in out))                         # the good rows still come back

    def test_classification_of_a_huge_adversarial_string_is_fast(self):
        import time
        t0 = time.perf_counter()
        for text in ("a" * 4000, "timed " * 700, "not " * 1000 + "found", "5" * 4000):
            T.classify_tool_error("Bash", text * 100)
        self.assertLess(time.perf_counter() - t0, 0.5)


class ActionViewTests(Base):
    def test_the_stored_failure_class_is_used_and_garbage_falls_back_to_tool_error(self):
        a = self.action(L.A_ERROR, failure_class=T.PERMISSION_REQUIRED)
        b = self.action(L.A_ERROR, failure_class="made-up")
        c = self.action(L.A_ERROR)
        self.assertEqual(T.action_view(self.led, a).failure_class, T.PERMISSION_REQUIRED)
        self.assertEqual(T.action_view(self.led, b).failure_class, T.TOOL_ERROR)
        self.assertEqual(T.action_view(self.led, c).failure_class, T.TOOL_ERROR)

    def test_a_retryable_error_on_a_side_effecting_tool_is_checked_first_on_a_read_tool_retried_once(self):
        w = T.action_view(self.led, self.action(L.A_ERROR, tool="Write", failure_class=T.RETRYABLE))
        r = T.action_view(self.led, self.action(L.A_ERROR, tool="Read", failure_class=T.RETRYABLE))
        self.assertEqual((w.recovery.action, r.recovery.action), (T.CHECK_FIRST, T.RETRY_ONCE))

    def test_legacy_denied_rows_map_to_permission_required(self):
        v = T.action_view(self.led, self.action(L.A_DENIED))
        self.assertEqual((v.status, v.failure_class, v.recovery.action), (T.FAILED, T.PERMISSION_REQUIRED, T.ASK_USER))


class HistoryTests(Base):
    def test_recent_is_newest_first_and_bounded_and_filterable(self):
        ids = [self.turn(L.T_DONE) for _ in range(5)]
        bad = self.turn(L.T_FAILED, "oversize")
        self.task(L.K_FAILED)
        out = T.recent(self.led, 3)
        self.assertEqual(len(out), 3)
        times = [v.ended_at or v.started_at for v in out]
        self.assertEqual(times, sorted(times, reverse=True))
        failed = T.recent(self.led, 50, statuses={T.FAILED})
        self.assertEqual({v.kind for v in failed}, {"turn", "task"})
        self.assertTrue(all(v.status == T.FAILED for v in failed))
        self.assertEqual([v.id for v in T.recent(self.led, 50, kinds=("turn",), statuses={T.FAILED})], [bad])

    def test_views_are_read_only(self):
        self.turn(L.T_FAILED, "no_echo")
        self.action(L.A_ERROR)
        with open(self.led.path, "rb") as fh:
            before = fh.read()
        for v in T.recent(self.led, 50):
            T.explain(v)
        with open(self.led.path, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_explain_carries_codes_only(self):
        tid = self.turn(L.T_FAILED, "api_error")
        line = T.explain(T.turn_view(self.led, tid))
        self.assertIn("class=RETRYABLE", line)
        self.assertIn("recovery=RETRY_ONCE(retryable_nothing_ran)", line)
        self.assertNotIn("\n", line)
        ok = T.explain(T.task_view(self.led, self.task(L.K_COMPLETED, rv=L.V_UNVERIFIED)))
        self.assertIn("claim_not_allowed", ok)
        self.assertIn("maybe_still_running", T.explain(T.task_view(self.led, self.task(L.K_ORPHAN_RUNNING))))


class CliTests(Base):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = TASKS.main(list(argv), ledger_dir=self.tmp.name)
        return code, out.getvalue()

    def test_failures_view_lists_only_what_went_wrong_with_the_decision(self):
        self.turn(L.T_DONE)
        self.turn(L.T_FAILED, "oversize")
        self.task(L.K_COMPLETED, rv=L.V_FAILED)
        code, out = self.run_cli("--failures")
        self.assertEqual(code, 0)
        self.assertIn("class=USER_ACTION_REQUIRED recovery=ASK_USER", out)
        self.assertIn("PARTIAL", out)
        self.assertNotIn("SUCCEEDED", out)

    def test_filters_and_bad_arguments(self):
        self.turn(L.T_DONE)
        self.task(L.K_STOPPED_USER)
        self.assertIn("cancel=user", self.run_cli("--kind", "task")[1])
        self.assertIn("nothing recorded", self.run_cli("--status", "UNKNOWN")[1])
        self.assertEqual(self.run_cli("--n", "0")[0], 2)

    def test_the_cli_never_writes_the_ledger(self):
        self.turn(L.T_FAILED, "no_echo")
        before = os.stat(self.led.path).st_size
        self.run_cli("--failures")
        self.assertEqual(os.stat(self.led.path).st_size, before)


class WiringTests(Base):
    def test_new_failure_lines_replace_the_generic_apology_for_known_causes(self):
        self.assertIn("couldn't reach my brain", SP.failure_line("api_error"))
        self.assertIn("never left my side", SP.failure_line("send_failed:OSError"))
        self.assertEqual(SP.failure_line("send_failed"), SP.failure_line("send_failed:TimeoutError"))
        self.assertIn("Something broke on my end", SP.failure_line("never_heard_of_it"))
        for old in ("oversize", "stream_lost", "no_echo", "max_turns"):                 # the pinned lines are unchanged
            self.assertEqual(SP.failure_line(old), SP.FAILURE_LINES[old])

    def test_the_recovery_log_line_uses_the_ledger_turn_when_it_can_and_the_cause_when_it_cannot(self):
        logs = []
        real = M.log
        M.log = logs.append
        try:
            tid = self.turn(L.T_FAILED, "api_error")
            M._log_recovery(SimpleNamespace(ledger=self.led), SimpleNamespace(turn_id=tid, reason="api_error"))
            M._log_recovery(SimpleNamespace(ledger=self.led), SimpleNamespace(turn_id=None, reason="send_failed:OSError"))
            M._log_recovery(SimpleNamespace(ledger=self.led), SimpleNamespace(turn_id=None, reason="stream_lost"))
            M._log_recovery(None, None)                                                    # garbage in: no raise
        finally:
            M.log = real
        self.assertIn(f"[recovery] turn {tid} FAILED cause=api_error class=RETRYABLE recovery=RETRY_ONCE", logs[0])
        self.assertIn("class=RETRYABLE recovery=RETRY_ONCE(retryable_nothing_ran)", logs[1])
        self.assertIn("recovery=CHECK_FIRST(retryable_but_may_have_run)", logs[2])          # cause alone: side effects unknown
        self.assertEqual(len(logs), 3)

    def test_nothing_in_speak_reply_retries_a_turn(self):
        src = inspect.getsource(M.speak_reply)
        self.assertEqual(src.count("brain.ask_stream("), 1)                              # the single-send invariant stays pinned
        self.assertIn("_log_recovery(brain, e)", src)

    def test_taskstate_is_pure(self):
        tree = ast.parse(inspect.getsource(T))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"re", "dataclasses.dataclass", "backtalk.ledger"})
        for banned in (".write(", "open(", "_append(", "mark_spoken", "action_set", "turn_state(", "task_state("):
            self.assertNotIn(banned, inspect.getsource(T).split('"""', 2)[2], banned)


if __name__ == "__main__":
    unittest.main()
