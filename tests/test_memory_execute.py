"""Memory 2.0, Phase 8.2d: the execute flow and the typed-word reconcile (backtalk/memory_execute.py). Every test runs on a
throwaway vault and receipts folder in a temp directory with a deterministic clock; the real vault is never opened, and a
check in the base class proves it. The terminal, clock, nonce and plan builder are fakes.

WHAT THESE TESTS DO NOT SHOW. A crash is an exception raised at a chosen step, not a process kill. The typed challenge is an
accidental-error confirmation, not authentication. The receipts lock does not exclude a non-cooperating store writer: test
`test_a_foreign_install_inside_the_window...` pins what happens instead (it cannot authorize a write, and it blocks)."""
import ast
import copy
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

from backtalk import memory_commit as C
from backtalk import memory_execute as EX
from backtalk import memory_receipts as RC
from backtalk import memory_review as R
from backtalk import memory_store as S
from tests.fake_clock import FakeClock
from tests.receipts_harness import item, read_log_bytes, tree_state
from tests.test_memory_commit_plan import NOWS
from tests.test_memory_review import Machine

NONCE = "ab" * 16
PACK_SHA = "a" * 64


class ExecCase(Machine):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        prod = os.path.realpath(S.REAL_VAULT)
        self.assertFalse(self.root == prod or self.root.startswith(prod + os.sep), "tests must never use the real vault")
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))
        self.vault = os.path.join(self.root, "vault")
        os.mkdir(self.vault)
        self.rdir = os.path.join(self.root, "receipts")
        os.mkdir(self.rdir, 0o700)
        os.chmod(self.rdir, 0o700)
        self.clock = FakeClock(NOWS)
        self.store = S.MemoryStore(self.vault, clock=self.clock)
        self.tick = 0.0
        self.mono_extra = 0.0
        self.jump_on = None                   # the monotonic call (counted down) on which the clock jumps
        self.jump_by = 0.0
        self.run_n = 0
        self.nonce = NONCE
        self.extra_before_replan = 0.0
        self.said, self.typed, self.asked = [], [], []
        self.plan_calls = 0
        self.lock("occupation")
        self.cand = self.cid("occupation")

    # -- injected fakes --
    def mono(self):
        self.tick += 0.5
        if self.jump_on is not None:
            self.jump_on -= 1
            if self.jump_on == 0:
                self.mono_extra += self.jump_by
        return self.tick + self.mono_extra

    def make_plan(self):
        self.plan_calls += 1
        if self.plan_calls == 2:
            self.mono_extra += self.extra_before_replan          # time passes between the answer and the re-plan
        return C.plan(pack=self.pack, pack_sha=PACK_SHA, records=self.state.records, private=None, candidate=self.cand,
                      store=self.store, provenance=lambda v: [], rederive=lambda v: copy.deepcopy(v),
                      clock=self.clock.stamp, monotonic=self.mono, preview_budget_s=EX.PREVIEW_BUDGET_S)

    def answer(self):
        return C.expected_answer(self.make_plan(), self.nonce)

    def next_answer(self):
        """The exact answer for the NEXT `run_commit` (which draws a fresh nonce)."""
        self.nonce = f"{self.run_n + 1:032x}"
        return self.answer()

    def ask(self, prompt):
        self.asked.append(prompt)
        return self.typed.pop(0)

    def run_commit(self, typed=None, **over):
        self.run_n += 1
        self.nonce = f"{self.run_n:032x}"                          # a fresh challenge nonce for every attempt
        self.typed = [self.answer()] if typed is None else list(typed)
        self.plan_calls = 0
        kw = dict(make_plan=self.make_plan, store=self.store, receipts_dir=self.rdir, ask=self.ask, say=self.said.append,
                  clock=self.clock.stamp, monotonic=self.mono, nonce_fn=lambda: self.nonce)
        kw.update(over)
        return EX.commit(**kw)

    def run_reconcile(self, typed):
        self.typed = list(typed)
        return EX.reconcile(store=self.store, receipts_dir=self.rdir, ask=self.ask, say=self.said.append,
                            clock=self.clock.stamp)

    # -- observation --
    def log(self):
        return RC.ReceiptLog(self.rdir)

    def actions(self):
        return [r["action"] for r in self.log().records()]

    def assess(self):
        return RC.assess(self.rdir, S.MemoryStore(self.vault, clock=self.clock))

    def memory_ids(self):
        return [s.item.id for s in self.store.list_items()] if os.path.isdir(self.store.dir) else []

    def receipts_files(self):
        return sorted(os.listdir(self.rdir))

    def spy_store(self):
        """Count store installs that carry an Authorization."""
        calls = []
        real_put, real_apply = S.MemoryStore.put, S.MemoryStore.apply

        def put(this, *a, **k):
            calls.append(("put", k.get("authorization") is not None))
            return real_put(this, *a, **k)

        def apply(this, *a, **k):
            calls.append(("apply", k.get("authorization") is not None or len(a) > 3))
            return real_apply(this, *a, **k)
        for target, name, fn in ((S.MemoryStore, "put", put), (S.MemoryStore, "apply", apply)):
            p = mock.patch.object(target, name, fn)
            p.start()
            self.addCleanup(p.stop)
        return calls


class CommitFlowTests(ExecCase):
    def test_E1_a_first_commit_goes_through_every_step_and_ends_clean(self):
        calls = self.spy_store()
        door = []
        real_door = RC.ReceiptLog.append_intent

        def spy_door(this, *a, **k):
            door.append(a[0]["action"])
            return real_door(this, *a, **k)
        with mock.patch.object(RC.ReceiptLog, "append_intent", spy_door):
            out = self.run_commit()
        self.assertEqual((out.status, out.code), ("committed", "committed"))
        self.assertEqual(self.actions(), ["open", "intent", "done"])
        self.assertEqual(door, ["intent"])                                       # the only door, once
        self.assertEqual(calls, [("put", True)])                                 # one authorized install
        self.assertEqual(len(self.memory_ids()), 1)
        a = self.assess()
        self.assertEqual((a.proceed, a.row), (True, 6))
        self.assertTrue(any("type exactly" in p for p in self.asked))
        self.assertIn(C.NOTICE, self.said)
        self.assertEqual(self.plan_calls, 2)                                     # the confirmed plan and the fresh re-plan

    def test_E2_a_wrong_answer_changes_nothing(self):
        calls = self.spy_store()
        before = tree_state(self.root)
        out = self.run_commit(typed=["commit 00000000"])
        self.assertEqual((out.status, out.code), ("declined", "challenge_declined"))
        self.assertEqual(tree_state(self.root), before)
        self.assertEqual(calls, [])
        self.assertEqual(self.plan_calls, 1)                                     # no re-plan for a declined challenge

    def test_E3_an_expired_challenge_changes_nothing(self):
        self.run_n += 1
        self.nonce = f"{self.run_n:032x}"
        answer = self.answer()
        self.typed = [answer]
        before = tree_state(self.root)

        def slow(prompt):
            self.clock.advance(C.CHALLENGE_SECONDS + 1)
            return self.typed.pop(0)
        self.run_n -= 1                                           # run_commit re-derives the same nonce
        out = self.run_commit(typed=[answer], ask=slow)
        self.assertEqual((out.status, out.code), ("declined", "challenge_expired"))
        self.assertEqual(tree_state(self.root), before)

    def test_E4_a_plan_that_changed_after_the_answer_aborts_before_any_receipt_or_store_call(self):
        calls = self.spy_store()
        self.nonce = f"{self.run_n + 1:032x}"
        answer = self.answer()

        def answer_then_change(prompt):
            reply = self.typed.pop(0)
            self.store.put(item(5, attribute="hobby", value="chess"), "I like chess")      # a write the person never saw
            return reply
        calls.clear()
        out = self.run_commit(typed=[answer], ask=answer_then_change)
        self.assertEqual((out.status, out.code), ("refused", "plan_changed"))
        self.assertIn("state_digest", out.details)
        self.assertEqual(calls, [("put", False)])                                # only the foreign unauthorized put
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())                     # no receipts were created

    def test_E5_the_startup_check_refuses_before_a_plan_is_built(self):
        for label, arrange, row in (
                ("marker", lambda: open(os.path.join(self.rdir, RC.BLOCKED), "w").close(), 1),
                ("folder mode", lambda: os.chmod(self.rdir, 0o755), 2)):
            with self.subTest(label):
                arrange()
                out = self.run_commit(typed=["x"])
                self.assertEqual(out.status, "refused")
                self.assertEqual(out.details[0], f"row_{row}")
                self.assertEqual(self.plan_calls, 0)
                os.chmod(self.rdir, 0o700)
                if os.path.exists(os.path.join(self.rdir, RC.BLOCKED)):
                    os.unlink(os.path.join(self.rdir, RC.BLOCKED))

    def test_E6_an_open_intent_and_a_state_gap_refuse_the_commit(self):
        self.run_commit()                                                        # one committed memory
        plan2_nonce = "cd" * 16
        snap = self.store.snapshot()                                             # an open intent, install never run
        open_fields = RC.intent_fields(snap, self._second_plan(), nonce=plan2_nonce,
                                       confirmed_at=self.clock.stamp(), apply_now=self.clock.stamp())
        self.log().append_intent(open_fields, self.clock.stamp(), self.store)
        out = self.run_commit(typed=["x"])
        self.assertEqual((out.status, out.code), ("refused", "reconcile_required"))
        self.assertEqual(self.plan_calls, 0)
        os.unlink(os.path.join(self.rdir, RC.RECEIPTS))                          # and a log that does not explain the store
        out = self.run_commit(typed=["x"])
        self.assertEqual((out.status, out.code), ("refused", "receipts_missing"))
        self.assertEqual(self.plan_calls, 0)

    def _second_plan(self):
        from types import SimpleNamespace
        from backtalk import memory_gate as G
        it = item(6, attribute="employer", value="Acme")
        ev = "I work at Acme"
        decision = G.evaluate(G.Proposal(it, ev), [s.item for s in self.store.list_items()], self.clock.stamp())
        return SimpleNamespace(candidate="goal_cash_buyer_list_deep", lock_hash="a" * 64, context_digest="c" * 64,
                               operation="put", item=it, evidence=ev, decision=decision,
                               item_digest=S.item_digest(it, ev, (), "put"))

    def test_E7_the_rest_deadline_aborts_after_the_intent_without_calling_the_store(self):
        calls = self.spy_store()
        real_door = RC.ReceiptLog.append_intent

        def slow_door(this, *a, **k):
            rec = real_door(this, *a, **k)
            self.jump_on, self.jump_by = 2, EX.REST_S + 1                         # time passes before the store call
            return rec
        with mock.patch.object(RC.ReceiptLog, "append_intent", slow_door):
            out = self.run_commit()
        self.assertEqual((out.status, out.code), ("aborted", "not_written"))
        self.assertEqual(out.details, ("rest_exceeded",))
        self.assertEqual(calls, [])
        self.assertEqual(self.actions(), ["open", "intent", "aborted"])
        self.assertEqual(self.assess().row, 6)

    def test_E8_the_e3_deadline_refuses_before_any_receipt(self):
        calls = self.spy_store()
        self.nonce = f"{self.run_n + 1:032x}"
        answer = self.answer()
        self.extra_before_replan = EX.E3_DEADLINE_S + 1
        out = self.run_commit(typed=[answer])
        self.assertEqual((out.status, out.code), ("refused", "deadline_exceeded"))
        self.assertEqual(calls, [])
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_E9_a_store_refusal_is_settled_from_the_state_and_the_decision_can_be_retried(self):
        real_put = S.MemoryStore.put
        state = {"fail": True}

        def put(this, *a, **k):
            if state["fail"]:
                raise S.StoreError("authorization_invalid", ("expired",))
            return real_put(this, *a, **k)
        with mock.patch.object(S.MemoryStore, "put", put):
            out = self.run_commit()
            self.assertEqual((out.status, out.code), ("aborted", "not_written"))
            self.assertEqual(out.details[0], "authorization_invalid")
            self.assertEqual(self.actions(), ["open", "intent", "aborted"])
            self.assertEqual(self.memory_ids(), [])
            state["fail"] = False
            self.clock.advance(10)
            out = self.run_commit()                                              # the same decision, retried
        self.assertEqual(out.status, "committed")
        self.assertEqual(self.actions(), ["open", "intent", "aborted", "intent", "done"])
        self.assertEqual(self.assess().row, 6)

    def test_E10_the_retry_limit_applies_through_execute(self):
        real_put = S.MemoryStore.put
        with mock.patch.object(S.MemoryStore, "put", side_effect=S.StoreError("write_failed")):
            for k in range(3):
                self.clock.advance(5)
                self.assertEqual(self.run_commit().status, "aborted")
            self.clock.advance(5)
            out = self.run_commit()
        self.assertEqual((out.status, out.code), ("refused", "receipts:record_refused"))
        self.assertEqual(out.details, ("RETRY_LIMIT",))
        self.assertEqual(self.actions().count("intent"), 3)
        self.assertEqual(self.assess().row, 6)

    def test_E11_an_interrupt_during_the_install_leaves_an_open_intent_that_reconcile_closes(self):
        with mock.patch.object(S.MemoryStore, "put", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()
        self.assertEqual(self.actions(), ["open", "intent"])
        a = self.assess()
        self.assertEqual((a.proceed, a.row, a.proof.kind), (False, 7, "NOT_WRITTEN"))
        out = self.run_reconcile(["reconcile"])
        self.assertEqual((out.status, out.code), ("reconciled", "aborted"))
        self.assertEqual(self.actions(), ["open", "intent", "aborted"])
        self.assertEqual(self.assess().row, 6)

    def test_E12_a_crash_after_the_install_is_proved_committed_and_reconciled_as_done(self):
        real_put = S.MemoryStore.put

        def install_then_die(this, *a, **k):
            real_put(this, *a, **k)
            raise KeyboardInterrupt
        with mock.patch.object(S.MemoryStore, "put", install_then_die):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()
        a = self.assess()
        self.assertEqual((a.row, a.proof.kind), (7, "COMMITTED"))
        out = self.run_reconcile(["reconcile"])
        self.assertEqual((out.status, out.code), ("reconciled", "done"))
        self.assertTrue(self.log().records()[-1]["recovered"])
        self.assertEqual(self.assess().row, 6)

    def test_E13_a_foreign_install_inside_the_window_cannot_authorize_a_write_and_blocks(self):
        real_door = RC.ReceiptLog.append_intent

        def foreign_after_intent(this, *a, **k):
            rec = real_door(this, *a, **k)
            self.store.put(item(5, attribute="hobby", value="chess"), "I like chess")     # a non-cooperating writer
            return rec
        with mock.patch.object(RC.ReceiptLog, "append_intent", foreign_after_intent):
            out = self.run_commit()
        self.assertEqual((out.status, out.code), ("anomaly", "inconsistent"))        # classified from the state
        self.assertEqual(self.memory_ids(), [item(5).id])                           # our memory was never installed
        self.assertEqual(self.actions(), ["open", "intent", "anomaly"])
        a = self.assess()
        self.assertEqual((a.proceed, a.row, a.code), (False, 8, "state_gap"))

    def test_E14_the_supersession_path_installs_through_apply(self):
        old = item(7, attribute="occupation", value="plumber", valid_from="2026-01-01T00:00:00Z",
                   created="2026-01-01T00:00:00Z", updated="2026-01-01T00:00:00Z")
        self.store.put(old, "I am a plumber by trade")
        snap = self.store.snapshot()
        self._baseline_over_existing_store(snap)                                  # a log whose baseline IS this store
        calls = self.spy_store()
        out = self.run_commit()
        self.assertEqual(out.status, "committed", out)
        self.assertIn(("apply", True), calls)
        statuses = {s.item.id: s.item.status for s in self.store.list_items()}
        self.assertEqual(statuses[old.id], "superseded")
        self.assertEqual(self.assess().row, 6)

    def _baseline_over_existing_store(self, snap):
        """Test-only: write the first receipt record by hand (`create` refuses a non-empty store, by design)."""
        rec = R.new_record([], self.clock.stamp(), RC.open_fields(snap))
        with open(os.path.join(self.rdir, RC.RECEIPTS), "wb") as f:
            f.write((R.canonical_json(rec) + "\n").encode())
        os.chmod(os.path.join(self.rdir, RC.RECEIPTS), 0o600)
        open(os.path.join(self.rdir, RC.LOCK), "wb").close()
        os.chmod(os.path.join(self.rdir, RC.LOCK), 0o600)

    def test_E15_a_plan_that_cannot_be_built_is_refused_with_its_code(self):
        def broken():
            raise C.CommitError("lock_too_old")
        out = self.run_commit(make_plan=broken)
        self.assertEqual((out.status, out.code), ("refused", "plan_refused:lock_too_old"))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_E16_a_bad_nonce_source_is_refused(self):
        out = self.run_commit(nonce_fn=lambda: "short")
        self.assertEqual((out.status, out.code), ("refused", "bad_nonce"))


class ReconcileTests(ExecCase):
    def test_R1_nothing_to_reconcile_asks_nothing(self):
        self.run_commit()
        self.asked.clear()
        self.said.clear()
        out = self.run_reconcile([])
        self.assertEqual((out.status, out.code), ("refused", "clean"))
        self.assertEqual(self.asked, [])

    def test_R2_a_blocked_log_is_not_overridable_by_the_word(self):
        self.run_commit()
        self.asked.clear()
        open(os.path.join(self.rdir, RC.BLOCKED), "w").close()
        before = tree_state(self.rdir)
        out = self.run_reconcile(["reconcile"])
        self.assertEqual((out.status, out.code), ("refused", "receipts_blocked"))
        self.assertEqual(tree_state(self.rdir), before)
        self.assertEqual(self.asked, [])

    def test_R3_only_the_exact_word_counts_and_nothing_is_appended_otherwise(self):
        with mock.patch.object(S.MemoryStore, "put", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()
        before = read_log_bytes_of(self.rdir)
        for word in ("", "y", "Reconcile", "reconcile ", "yes"):
            out = self.run_reconcile([word])
            self.assertEqual((out.status, out.code), ("declined", "not_confirmed"), repr(word))
            self.assertEqual(read_log_bytes_of(self.rdir), before)

    def test_R4_the_state_moving_between_the_proposal_and_the_word_refuses(self):
        with mock.patch.object(S.MemoryStore, "put", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()
        before = read_log_bytes_of(self.rdir)

        def move_then_type(prompt):
            self.store.put(item(5, attribute="hobby", value="chess"), "I like chess")
            return "reconcile"
        out = EX.reconcile(store=self.store, receipts_dir=self.rdir, ask=move_then_type, say=self.said.append,
                           clock=self.clock.stamp)
        self.assertEqual(out.status, "refused")
        self.assertEqual(read_log_bytes_of(self.rdir), before)

    def test_R5_an_inconsistent_proof_appends_only_an_anomaly(self):
        with mock.patch.object(S.MemoryStore, "put", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()
        os.makedirs(self.store.dir, exist_ok=True)
        open(os.path.join(self.store.dir, ".lock"), "w").close()
        with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:     # a stray row: neither side of the intent
            f.write("%s %s %s\n" % ("c" * 32, "mem_000000000099", "2026-03-02T09:59:00Z"))
        out = self.run_reconcile(["reconcile"])
        self.assertEqual((out.status, out.code), ("reconciled", "anomaly"))
        self.assertEqual(self.actions()[-1], "anomaly")
        self.assertFalse(self.assess().proceed)


def read_log_bytes_of(rdir):
    with open(os.path.join(rdir, RC.RECEIPTS), "rb") as f:
        return f.read()


class ScriptedMonotonic:
    """A monotonic clock that returns the given values in order (then the last one): it pins which operand of which
    deadline test fires, and the exact boundary."""

    def __init__(self, *values):
        self.values, self.i = list(values), 0

    def __call__(self):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


class DeadlineBoundaryTests(ExecCase):
    """`commit`'s own monotonic calls, in order: (1) t_answer, (2) after the re-plan, (3) t_intent, (4) before the store
    call (first operand), then (5) the rest check. The plan's own timing uses the plan builder's clock, not this one."""

    def go(self, *values):
        return self.run_commit(monotonic=ScriptedMonotonic(*values))

    def test_all_five_values_equal_is_the_happy_path(self):
        out = self.go(100, 100, 100, 100, 100)
        self.assertEqual(out.status, "committed")

    def test_exactly_the_e3_deadline_after_the_replan_is_allowed(self):
        self.assertEqual(self.go(100, 190.0, 190.0, 190.0, 190.0).status, "committed")

    def test_one_hundredth_over_the_e3_deadline_after_the_replan_is_refused_before_any_receipt(self):
        out = self.go(100, 190.01)
        self.assertEqual((out.status, out.code), ("refused", "deadline_exceeded"))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())                 # before any receipt

    def test_the_first_operand_of_the_pre_install_check(self):
        out = self.go(100, 100, 100, 190.01, 100)
        self.assertEqual((out.status, out.code, out.details), ("aborted", "not_written", ("deadline_exceeded",)))
        self.assertEqual(self.actions(), ["open", "intent", "aborted"])

    def test_the_second_operand_of_the_pre_install_check(self):
        out = self.go(100, 100, 190.01, 100, 100)                              # t_intent - t_answer is over
        self.assertEqual((out.status, out.details), ("aborted", ("deadline_exceeded",)))

    def test_the_second_operand_exactly_at_the_limit_is_allowed(self):
        self.assertEqual(self.go(100, 100, 190.0, 100, 100).status, "committed")

    def test_the_rest_deadline_exactly_at_the_limit_is_allowed(self):
        self.assertEqual(self.go(100, 100, 100, 100, 130.0).status, "committed")

    def test_the_rest_deadline_one_hundredth_over_is_refused(self):
        out = self.go(100, 100, 100, 100, 130.01)
        self.assertEqual((out.status, out.details), ("aborted", ("rest_exceeded",)))


class RefusalDetailTests(ExecCase):
    def test_a_clock_that_is_not_a_stamp_is_a_refusal_not_an_exception(self):
        for bad in ("garbage", None, 5, ""):
            with self.subTest(bad=bad):
                out = self.run_commit(clock=lambda b=bad: b)
                self.assertEqual((out.status, out.code), ("refused", "bad_clock"))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_a_clock_that_breaks_after_the_challenge_is_a_refusal(self):
        answer = self.next_answer()
        state = {"calls": 0}

        def clock():
            state["calls"] += 1
            return self.clock.stamp() if state["calls"] == 1 else "garbage"      # fine at issue, broken at the answer
        out = self.run_commit(typed=[answer], clock=clock)
        self.assertEqual((out.status, out.code), ("refused", "bad_clock"))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_the_outcome_is_an_immutable_value(self):
        out = EX.Outcome("refused", "x")
        import dataclasses
        with self.assertRaises(dataclasses.FrozenInstanceError):
            out.status = "committed"

    def test_a_failing_replan_is_refused_with_its_code_and_details(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 2:
                raise C.CommitError("lock_too_old", ("detail",))
            return self.make_plan()
        out = self.run_commit(typed=[self.next_answer()], make_plan=flaky)
        self.assertEqual((out.status, out.code, out.details), ("refused", "replan_refused:lock_too_old", ("detail",)))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_a_clock_that_runs_backwards_between_the_answer_and_the_replan(self):
        answer = self.next_answer()
        state = {"n": 0}

        def ask_then_forward(prompt):
            self.clock.advance(5)                                               # answered at +5
            return self.typed.pop(0)
        real_plan = self.make_plan

        def back_up():
            state["n"] += 1
            if state["n"] == 2:
                self.clock.advance(-3)                                          # re-planned at +2: before the answer
            return real_plan()
        out = self.run_commit(typed=[answer], ask=ask_then_forward, make_plan=back_up)
        self.assertEqual((out.status, out.code), ("refused", "clock_regressed"))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_the_second_startup_check_can_refuse_after_the_person_answered(self):
        answer = self.next_answer()

        def answer_then_block(prompt):
            reply = self.typed.pop(0)
            open(os.path.join(self.rdir, RC.BLOCKED), "w").close()               # appears while the person was typing
            return reply
        out = self.run_commit(typed=[answer], ask=answer_then_block)
        self.assertEqual((out.status, out.code, out.details), ("refused", "receipts_blocked", ("row_1",)))
        self.assertNotIn(RC.RECEIPTS, self.receipts_files())

    def test_startup_details_are_truncated_to_four(self):
        os.makedirs(self.store.dir)
        open(os.path.join(self.store.dir, ".lock"), "w").close()
        rows = "".join("%032x mem_%012x 2026-03-01T00:00:00Z\n" % (i + 1, i + 1) for i in range(6))
        with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:
            f.write(rows)
        rec = R.new_record([], self.clock.stamp(), RC.open_fields(S.Snapshot(True, "ok", "initialized", (), (), (),
                                                                             S.manifest_digest([]),
                                                                             S.snapshot_digest([], []))))
        with open(os.path.join(self.rdir, RC.RECEIPTS), "wb") as f:                # a log that knows no nonce row
            f.write((R.canonical_json(rec) + "\n").encode())
        os.chmod(os.path.join(self.rdir, RC.RECEIPTS), 0o600)
        open(os.path.join(self.rdir, RC.LOCK), "wb").close()
        os.chmod(os.path.join(self.rdir, RC.LOCK), 0o600)
        out = self.run_commit(typed=["x"])
        self.assertEqual((out.status, out.code), ("refused", "unreceipted_commit"))
        self.assertEqual(len(out.details), 5)                                       # "row_9" and exactly four rows
        self.assertEqual(out.details[0], "row_9")
        out = self.run_reconcile([])
        self.assertEqual((out.status, out.code), ("refused", "unreceipted_commit"))
        self.assertEqual(len(out.details), 5)                                       # the same four-row cut in reconcile
        self.assertEqual(out.details[0], "row_9")

    def test_store_refusal_details_are_truncated_to_three(self):
        err = S.StoreError("authorization_invalid", ("a", "b", "c", "d", "e"))
        with mock.patch.object(S.MemoryStore, "put", side_effect=err):
            out = self.run_commit()
        self.assertEqual((out.status, out.details), ("aborted", ("authorization_invalid", "a", "b", "c")))


class SettleOutcomeTests(ExecCase):
    def after_install_snapshot_not_ok(self, install_error=None):
        state = {"installed": False}
        real_put, real_snapshot = S.MemoryStore.put, S.MemoryStore.snapshot

        def put(this, *a, **k):
            state["installed"] = True
            if install_error is not None:
                raise install_error
            return real_put(this, *a, **k)

        def snapshot(this, **k):
            if state["installed"]:
                return S.Snapshot(False, "lock_busy", None, None, None, None, None, None)
            return real_snapshot(this, **k)
        for name, fn in (("put", put), ("snapshot", snapshot)):
            p = mock.patch.object(S.MemoryStore, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def test_a_snapshot_that_is_not_ok_after_the_install_is_unclassified_and_appends_nothing(self):
        self.after_install_snapshot_not_ok()
        out = self.run_commit()
        self.assertEqual((out.status, out.code, out.details), ("unclassified", "snapshot:lock_busy", ()))
        self.assertEqual(out.intent_seq, 1)
        self.assertEqual(out.memory_id, self.log().records()[1]["memory_id"])
        self.assertEqual(self.actions(), ["open", "intent"])                         # nothing was told to the log

    def test_the_same_with_a_store_refusal_keeps_the_reason(self):
        self.after_install_snapshot_not_ok(S.StoreError("authorization_invalid", ("expired",)))
        out = self.run_commit()
        self.assertEqual((out.status, out.code), ("unclassified", "snapshot:lock_busy"))
        self.assertEqual(out.details, ("authorization_invalid", "expired"))

    def test_a_receipt_that_cannot_be_appended_leaves_the_open_intent_and_says_so(self):
        err = RC.ReceiptError("receipts_append_ambiguous", ("x",))
        with mock.patch.object(RC.ReceiptLog, "append", side_effect=err):
            out = self.run_commit()
        self.assertEqual((out.status, out.code, out.details), ("unclassified", "receipts:receipts_append_ambiguous", ("x",)))
        self.assertEqual(out.intent_seq, 1)
        self.assertEqual(len(self.memory_ids()), 1)                                  # the install did happen
        a = self.assess()
        self.assertEqual((a.row, a.proof.kind), (7, "COMMITTED"))


class ReconcileDetailTests(ExecCase):
    def interrupted(self):
        with mock.patch.object(S.MemoryStore, "put", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_commit()

    def test_what_is_shown_before_the_word_is_typed(self):
        self.interrupted()
        self.said.clear()
        self.run_reconcile(["reconcile"])
        self.assertEqual(self.said[-1], "the one record that would be appended: aborted")
        self.assertEqual(len(self.said), 4)                                           # two facts, one inference, the record

    def test_an_anomaly_proposal_lists_its_codes(self):
        self.interrupted()
        os.makedirs(self.store.dir, exist_ok=True)
        open(os.path.join(self.store.dir, ".lock"), "w").close()
        with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:
            f.write("%s %s %s\n" % ("c" * 32, "mem_000000000099", "2026-03-02T09:59:00Z"))
        self.said.clear()
        self.run_reconcile(["reconcile"])
        proposed = self.assess().proposed
        self.assertEqual(self.said[-1], "the one record that would be appended: anomaly " + ", ".join(proposed["codes"]))
        self.assertIn(", ", self.said[-1]) if len(proposed["codes"]) > 1 else None

    def test_a_proposal_that_changes_while_the_row_stays_seven_is_refused(self):
        self.interrupted()                                                           # proposal: aborted (NOT_WRITTEN)
        before = read_log_bytes_of(self.rdir)

        def stray_row_then_word(prompt):
            os.makedirs(self.store.dir, exist_ok=True)
            open(os.path.join(self.store.dir, ".lock"), "w").close()
            with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:     # files unchanged: still row 7
                f.write("%s %s %s\n" % ("c" * 32, "mem_000000000099", "2026-03-02T09:59:00Z"))
            return "reconcile"
        out = EX.reconcile(store=self.store, receipts_dir=self.rdir, ask=stray_row_then_word, say=self.said.append,
                           clock=self.clock.stamp)
        self.assertEqual((out.status, out.code, out.details), ("refused", "state_changed", ("row_7",)))
        self.assertEqual(read_log_bytes_of(self.rdir), before)

    def test_a_log_that_cannot_take_the_record_is_refused_with_its_reason(self):
        self.interrupted()
        err = RC.ReceiptError("receipts_changed", ("x",))
        with mock.patch.object(RC.ReceiptLog, "append", side_effect=err):
            out = self.run_reconcile(["reconcile"])
        self.assertEqual((out.status, out.code, out.details), ("refused", "receipts:receipts_changed", ("x",)))

    def test_the_stamp_for_the_record_must_be_valid(self):
        self.interrupted()
        out = EX.reconcile(store=self.store, receipts_dir=self.rdir, ask=lambda p: "reconcile", say=self.said.append,
                           clock=lambda: "garbage")
        self.assertEqual((out.status, out.code), ("refused", "receipts:bad_clock"))


class StructureTests(unittest.TestCase):
    SRC = inspect.getsource(EX)

    def test_the_only_intent_door_and_only_these_appends(self):
        tree = ast.parse(self.SRC)
        door, appends = 0, []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "append_intent":
                    door += 1
                if n.func.attr == "append":
                    appends.append(ast.unparse(n.args[0]) if n.args else "")
                if n.func.attr in ("_write", "_put", "create") and n.func.attr != "create":
                    self.fail("a private writer is called: " + n.func.attr)
        self.assertEqual(door, 1)
        self.assertEqual(sorted(appends), sorted([
            "RC.done_fields(intent['seq'], snap, recovered=install_error is not None)",
            "RC.aborted_fields(intent['seq'], 'crashed_before_write', snap.state_digest)",
            "RC.anomaly_fields(intent['seq'], proof.codes, snap.state_digest)",
            "b.proposed"]))

    def test_the_fresh_replan_and_comparison_come_before_any_receipt_or_store_call(self):
        src = inspect.getsource(EX.commit)
        order = [src.index(s) for s in ("RC.assess(receipts_dir, store)", "first = make_plan()", "C.check_answer",
                                        "plan = make_plan()", "C.compare_plans(first, plan)", "log.create(",
                                        "log.append_intent(", "store.put(", "_settle(")]
        self.assertEqual(order, sorted(order))

    def test_budgets_and_limits_are_stated(self):
        self.assertEqual((EX.PREVIEW_BUDGET_S, EX.E3_DEADLINE_S, EX.REST_S), (45, 90, 30))
        doc = " ".join(EX.__doc__.split())
        for phrase in ("PROVISIONAL and were NOT measured by W2", "does not exclude a non-cooperating store writer",
                       "accidental-error confirmation, not authentication", "A Plan, token or answer is never itself an "
                       "authorization", "FRESH re-plan"):
            self.assertIn(phrase, doc)

    def test_it_imports_only_the_layers_it_composes(self):
        mods = set()
        for n in ast.walk(ast.parse(self.SRC)):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"re", "dataclasses.dataclass", "backtalk.memory", "backtalk.memory_commit",
                                "backtalk.memory_gate", "backtalk.memory_receipts", "backtalk.memory_store"})

    def test_it_reads_no_clock_and_no_keyboard_and_no_file_itself(self):
        tree = ast.parse(self.SRC)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"now", "utcnow", "time", "monotonic_ns", "input", "open", "environ", "getenv", "listdir",
                                 "system", "subprocess", "unlink", "rename", "makedirs", "mkdir"}, used)


if __name__ == "__main__":
    unittest.main()
