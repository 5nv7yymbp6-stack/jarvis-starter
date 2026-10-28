"""Memory 2.0, Phase 8.2c-2 amendment: retrying an intent after `aborted(crashed_before_write)`, the closing snapshot
digest at row 6, `check_new_intent`, and `append_intent` as the only door for an intent (backtalk/memory_receipts.py).
Every test runs on a throwaway vault and receipts folder in a temp directory with a deterministic clock; nothing reads or
writes the real vault.

WHAT THESE TESTS DO NOT SHOW. The receipts lock does not exclude a writer that does not take it, and the interval between
the snapshot and the write is NOT atomic against such a writer (tests C1/C2 pin exactly what happens instead: it can never
authorize an install, and what it cannot prevent is detected and blocks). A crash is an exception at a chosen step, not a
real process kill."""
import ast
import inspect
import os
import random
import threading
import unittest
from unittest import mock

from backtalk import memory_receipts as RC
from backtalk import memory_store as S
from tests.receipts_harness import (Crash, Rig, day, envelope_free, ev, item, mid, read_log_bytes, snapshot_of,
                                    tree_state, write_log)

CLEAN = "crashed_before_write"


class Case(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(self)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))

    def attempt(self, n=1, **over):
        """A fresh plan, a fresh authorization (new nonce) and its appended intent for item n."""
        self.rig.clock.advance(5)
        plan = self.rig.plan(n, **over)
        auth = self.rig.authorization(plan)
        return plan, auth, self.rig.begin(plan, auth)

    def abort(self, intent, code=CLEAN):
        return self.rig.log.append(RC.aborted_fields(intent["seq"], code, self.rig.store.state_digest()),
                                   self.rig.clock.stamp())

    def refused(self, reason, details, fn, *a, **k):
        with self.assertRaises(RC.ReceiptError) as cm:
            fn(*a, **k)
        self.assertEqual((cm.exception.reason, cm.exception.details), (reason, details))

    def start(self):
        return RC.assess(self.rig.rdir, self.rig.fresh_store())

    def actions(self):
        return [r["action"] for r in self.rig.log.records()]


# ---- R: history rules and the cap -------------------------------------------------------------------------------------

class RetryRuleTests(Case):
    def test_R1_a_clean_abort_can_be_retried_and_the_retry_commits(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first)
        self.rig.clock.advance(30)
        plan2, auth2, second = self.attempt()
        self.assertEqual(first["memory_id"], second["memory_id"])                      # same decision, same id
        self.assertNotEqual(first["nonce_hash"], second["nonce_hash"])
        self.rig.install(plan2, auth2)
        done = self.rig.finish(second)
        self.assertEqual(self.actions(), ["open", "intent", "aborted", "intent", "done"])
        self.assertEqual(done["intent_seq"], second["seq"])
        a = self.start()
        self.assertEqual((a.proceed, a.row), (True, 6))
        self.assertIn(plan.item.id + ".md", dict(self.rig.store.snapshot().files))

    def test_R2_a_retry_may_not_reuse_the_aborted_attempts_nonce(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        self.refused("record_refused", ("DUPLICATE_NONCE",), self.rig.log.append_intent, fields,
                     self.rig.clock.stamp(), self.rig.store)

    def test_R3_the_cap_is_three_total_intents_and_the_fourth_is_retry_limit(self):
        self.rig.baseline()
        for k in range(3):
            plan, auth, intent = self.attempt()
            self.abort(intent)
        before = read_log_bytes(self.rig)
        self.rig.clock.advance(5)
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        self.refused("record_refused", ("RETRY_LIMIT",), self.rig.begin, plan, auth)
        self.assertEqual(read_log_bytes(self.rig), before)                             # a refusal changes nothing
        self.assertEqual(self.actions().count("intent"), 3)
        a = self.start()                                                               # and the log itself is clean
        self.assertEqual((a.proceed, a.row), (True, 6))

    def test_R3b_the_third_attempt_may_still_succeed(self):
        self.rig.baseline()
        for k in range(2):
            plan, auth, intent = self.attempt()
            self.abort(intent)
        plan, auth, third = self.attempt()
        self.rig.install(plan, auth)
        self.rig.finish(third)
        self.assertEqual(self.actions().count("intent"), 3)
        self.assertEqual(self.start().row, 6)

    def test_R4_history_rule_for_every_status(self):
        clean = {"status": "aborted", "code": CLEAN, "anomalous": False}
        cases = (
            ([], None), ([clean], None), ([clean, clean], None),
            ([clean, clean, clean], "RETRY_LIMIT"),
            ([{"status": "done", "code": None, "anomalous": False}], "DUPLICATE_INTENT"),
            ([{"status": "open", "code": None, "anomalous": False}], "DUPLICATE_INTENT"),
            ([dict(clean, anomalous=True)], "DUPLICATE_INTENT"),
            ([dict(clean, code="user_declined")], "DUPLICATE_INTENT"),
            ([dict(clean, code=None)], "DUPLICATE_INTENT"),
            ([clean, dict(clean, anomalous=True)], "DUPLICATE_INTENT"),
            ([dict(clean, code="user_declined"), clean, clean], "DUPLICATE_INTENT"),    # history beats the cap
        )
        for prior, expected in cases:
            with self.subTest(prior=[(p["status"], p["code"], p["anomalous"]) for p in prior]):
                self.assertEqual(RC.retry_refusal(prior), expected)
        self.assertEqual(RC.MAX_INTENTS_PER_ID, 3)
        self.assertEqual(RC.RETRYABLE_ABORT_CODES, frozenset({"crashed_before_write"}))

    def test_R5_an_anomaly_permanently_prevents_automatic_retry(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        live = self.rig.store.state_digest()
        self.rig.log.append(RC.anomaly_fields(first["seq"], ["STATE_GAP"], live), self.rig.clock.stamp())
        self.abort(first)                                          # a person closed it after the anomaly
        self.rig.clock.advance(5)
        plan2 = self.rig.plan(1)
        self.refused("record_refused", ("DUPLICATE_INTENT",), self.rig.begin, plan2, self.rig.authorization(plan2))

    def test_R6_only_the_allowlisted_abort_code_permits_a_retry(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first, code="user_declined")
        self.rig.clock.advance(5)
        plan2 = self.rig.plan(1)
        self.refused("record_refused", ("DUPLICATE_INTENT",), self.rig.begin, plan2, self.rig.authorization(plan2))

    def test_R7_an_open_intent_blocks_any_new_intent_for_the_same_or_another_id(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        for n, kw in ((1, {}), (2, {"attribute": "employer", "value": "Acme"})):
            self.rig.clock.advance(5)
            p = self.rig.plan(n, **kw)
            with self.subTest(n=n):
                self.refused("record_refused", ("INTENT_OPEN",), self.rig.begin, p, self.rig.authorization(p))
        self.assertEqual(self.actions(), ["open", "intent"])

    def test_R8_an_unrelated_commit_between_the_abort_and_the_retry(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first)
        self.rig.clock.advance(5)
        self.rig.commit(self.rig.plan(2, attribute="employer", value="Acme"))
        plan2, auth2, second = self.attempt()
        self.rig.install(plan2, auth2)
        self.rig.finish(second)
        self.assertEqual(self.actions(), ["open", "intent", "aborted", "intent", "done", "intent", "done"])
        self.assertEqual(self.start().row, 6)

    def test_R13_a_forged_abort_with_the_wrong_digest_is_still_refused(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        fields = [envelope_free(r) for r in self.rig.log.records()]
        write_log(self.rig, fields + [RC.aborted_fields(first["seq"], CLEAN, "f" * 64)])
        self.assertEqual(RC.validate_records(RC._read_records(self.rig.rpath()))[0], ["CLOSE_DIGEST_MISMATCH"])

    def test_R14_random_histories_agree_with_an_independent_model(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        o, base = [envelope_free(r) for r in self.rig.log.records()]
        rnd = random.Random(8)
        closers = ("clean", "other_code", "anomaly_then_clean", "clean", "clean")
        checked = 0
        for trial in range(120):
            fields, history = [o], []                                   # history: "clean" | "dirty"
            for k in range(5):
                intent = dict(base, nonce_hash=f"{trial * 10 + k + 1:032x}")
                problems = RC.validate_records(self._chain(fields + [intent]))[0]
                if any(h != "clean" for h in history):
                    expected = ["DUPLICATE_INTENT"]
                elif len(history) >= 3:
                    expected = ["RETRY_LIMIT"]
                else:
                    expected = []
                self.assertEqual(problems, expected, (trial, k, history))
                checked += 1
                if expected:
                    break
                seq = len(fields)
                kind = rnd.choice(closers)
                fields.append(intent)
                if kind == "other_code":
                    fields.append(RC.aborted_fields(seq, "user_declined", base["state_digest_before"]))
                elif kind == "anomaly_then_clean":
                    fields.append(RC.anomaly_fields(seq, ["X"], None))
                    fields.append(RC.aborted_fields(seq, CLEAN, base["state_digest_before"]))
                else:
                    fields.append(RC.aborted_fields(seq, CLEAN, base["state_digest_before"]))
                history.append("clean" if kind == "clean" else "dirty")
        self.assertGreater(checked, 200)

    def _chain(self, fields_list):
        from backtalk import memory_review as R
        records = []
        for f in fields_list:
            records.append(R.new_record(records, "2026-02-01T00:00:00Z", f))
        return records


# ---- the closing snapshot digest, row 6 -------------------------------------------------------------------------------

class ClosingSnapshotTests(Case):
    def inject(self, row):
        os.makedirs(self.rig.memdir, exist_ok=True)
        for name in (".lock",):
            if not os.path.exists(self.rig.path(name)):
                open(self.rig.path(name), "w").close()
        with open(self.rig.path(".authorizations"), "a") as f:
            f.write("%s %s %s\n" % row)

    def test_the_expected_closing_snapshot_for_each_closing_record(self):
        self.rig.baseline()
        records = self.rig.log.records()
        state = RC.validate_records(records)[1]
        self.assertEqual(state.closing_snapshot_digest, records[0]["baseline_snapshot_digest"])
        plan, auth, intent = self.attempt()
        self.assertEqual(RC.validate_records(self.rig.log.records())[1].closing_snapshot_digest,
                         records[0]["baseline_snapshot_digest"])                       # an open intent changes nothing
        self.abort(intent)
        state = RC.validate_records(self.rig.log.records())[1]
        self.assertEqual(state.closing_snapshot_digest, intent["snapshot_digest_before"])
        plan2, auth2, second = self.attempt()
        self.rig.install(plan2, auth2)
        done = self.rig.finish(second)
        state = RC.validate_records(self.rig.log.records())[1]
        self.assertEqual(state.closing_snapshot_digest, done["snapshot_digest"])
        self.assertEqual(state.closing_snapshot_digest, self.rig.store.snapshot().snapshot_digest)

    def test_R10_a_nonce_row_bound_to_an_aborted_intent_blocks_row_6(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first)
        self.assertEqual(self.start().row, 6)
        self.inject((first["nonce_hash"], first["memory_id"], first["confirmed_at"]))
        before = tree_state(self.rig.root)
        a = self.start()
        self.assertEqual((a.proceed, a.row, a.code), (False, 9, "nonce_rows_changed"))
        self.assertEqual(a.details, ("the nonce rows differ from the last closing record",))
        report = RC.audit(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual((report.verdict, report.proposed_record), ("BLOCKED", None))
        self.assertEqual(tree_state(self.rig.root), before)                            # reading changed nothing

    def test_a_row_with_an_unknown_hash_keeps_the_specific_code(self):
        self.rig.baseline()
        self.inject(("a" * 32, mid(9), "2026-01-31T12:00:00Z"))
        a = self.start()
        self.assertEqual((a.row, a.code, a.details), (9, "unreceipted_commit", ("a" * 32,)))

    def test_editing_any_field_of_a_committed_row_blocks_row_6(self):
        self.rig.baseline()
        intent, done, auth = self.rig.commit(self.rig.plan(1))
        path = self.rig.path(".authorizations")
        with open(path) as f:
            row = f.read().split()
        good = " ".join(row) + "\n"
        edits = {"hash": ("b" * 32, row[1], row[2]), "memory_id": (row[0], mid(77), row[2]),
                 "confirmed_at": (row[0], row[1], "2026-01-31T00:00:01Z")}
        for label, edited in edits.items():
            with open(path, "w") as f:
                f.write("%s %s %s\n" % edited)
            with self.subTest(label):
                a = self.start()
                self.assertFalse(a.proceed)
                self.assertEqual(a.row, 9)
        with open(path, "w") as f:
            f.write(good)
        self.assertEqual(self.start().row, 6)

    def test_removing_the_row_blocks_row_6(self):
        self.rig.baseline()
        self.rig.commit(self.rig.plan(1))
        os.unlink(self.rig.path(".authorizations"))
        a = self.start()
        self.assertEqual((a.row, a.code), (9, "nonce_rows_changed"))


# ---- check_new_intent: each live comparison --------------------------------------------------------------------------

class CheckNewIntentTests(Case):
    def setUp(self):
        super().setUp()
        self.rig.baseline()
        plan, auth, first = self.attempt()
        self.abort(first)
        self.first = first
        self.rig.clock.advance(5)
        self.plan = self.rig.plan(1)
        self.auth = self.rig.authorization(self.plan)

    def state(self):
        return RC.validate_records(self.rig.log.records())[1]

    def fields(self, **over):
        f = RC.intent_fields(self.rig.store.snapshot(), self.plan, nonce=self.auth.nonce,
                             confirmed_at=self.auth.confirmed_at, apply_now=self.rig.clock.stamp())
        f.update(over)
        return f

    def check(self, fields=None, snapshot=None, state=None):
        return RC.check_new_intent(state or self.state(), snapshot or self.rig.store.snapshot(),
                                   fields if fields is not None else self.fields())

    def test_a_consistent_retry_is_admissible(self):
        self.assertIsNone(self.check())

    def test_R9_every_live_comparison_has_its_own_code(self):
        good = self.fields()
        bad_digest = "f" * 64
        self.assertEqual(self.check(fields="nope"), "bad_fields")
        self.assertEqual(self.check(fields=dict(good, action="done")), "bad_fields")
        self.assertEqual(self.check(snapshot=S.Snapshot(False, "lock_busy", None, None, None, None, None, None)),
                         "snapshot_not_ok")
        self.assertEqual(self.check(snapshot="nope"), "snapshot_not_ok")
        self.assertEqual(RC.check_new_intent(None, self.rig.store.snapshot(), good), "log_blocked")
        self.assertEqual(self.check(fields=dict(good, state_digest_before=bad_digest)), "state_stale")
        self.assertEqual(self.check(fields=dict(good, manifest_before=[[mid(9) + ".md", "e" * 64]])), "manifest_stale")
        self.assertEqual(self.check(fields=dict(good, snapshot_digest_before=bad_digest)), "snapshot_stale")
        self.assertEqual(self.check(fields=dict(good, nonces_before=[["a" * 32, mid(9), "2026-01-31T00:00:00Z"]])),
                         "nonces_stale")
        self.assertEqual(self.check(fields=dict(good, nonce_hash=self.first["nonce_hash"])), "nonce_reused")

    def test_the_memory_file_already_present(self):
        good = self.fields()
        self.rig.seed(1)                                                       # the id's file appears in the store
        self.assertEqual(self.check(fields=good), "memory_file_present")

    def test_the_store_moved_since_the_log_closed(self):
        good = self.fields()
        self.rig.seed(5, attribute="hobby", value="chess")
        self.assertEqual(self.check(fields=good), "state_gap")

    def test_a_live_row_bound_to_an_aborted_or_open_intent(self):
        good = self.fields()
        os.makedirs(self.rig.memdir, exist_ok=True)
        open(self.rig.path(".lock"), "w").close()
        with open(self.rig.path(".authorizations"), "w") as f:
            f.write("%s %s %s\n" % (self.first["nonce_hash"], self.first["memory_id"], self.first["confirmed_at"]))
        self.assertEqual(self.check(fields=good), "nonce_row_bound_to_undone_intent")

    def test_any_other_nonce_change_since_the_log_closed(self):
        good = self.fields()
        os.makedirs(self.rig.memdir, exist_ok=True)
        open(self.rig.path(".lock"), "w").close()
        with open(self.rig.path(".authorizations"), "w") as f:
            f.write("%s %s %s\n" % ("a" * 32, mid(9), "2026-01-31T12:00:00Z"))
        self.assertEqual(self.check(fields=good), "nonce_rows_changed")

    def test_an_open_intent_and_a_global_anomaly(self):
        self.rig.begin(self.plan, self.auth)
        self.assertEqual(self.check(), "intent_open")
        rig2 = Rig(self)
        rig2.baseline()
        recs = [envelope_free(r) for r in rig2.log.records()]
        write_log(rig2, recs + [RC.anomaly_fields(None, ["X"], None)])
        state = RC.validate_records(RC._read_records(rig2.rpath()))[1]
        self.assertTrue(state.blocked)
        self.assertEqual(RC.check_new_intent(state, rig2.store.snapshot(), self.fields()), "log_blocked")

    def test_the_log_state_carries_what_the_check_needs(self):
        s = self.state()
        self.assertEqual(s.undone_nonce_hashes, frozenset({self.first["nonce_hash"]}))
        self.assertEqual(s.nonce_hashes, frozenset({self.first["nonce_hash"]}))
        self.assertIsNone(s.open_intent)


# ---- R11 and R12: prune bounds and crashes inside a retry ------------------------------------------------------------

class RetryIntegrationTests(Case):
    def test_R11_the_retry_is_classified_by_its_own_confirmed_at(self):
        """Three receipted commits leave rows at known ages; the retry happens a day later, so one row is older than
        the prune floor, one is in the band and one is young. The store prunes by its own clock and the proof agrees."""
        self.rig.baseline()
        t0 = self.rig.clock.t
        rows = []
        for n, offset in ((2, 0), (3, 100), (4, 200)):
            self.rig.clock.t = t0 + __import__("datetime").timedelta(seconds=offset)
            intent, done, auth = self.rig.commit(self.rig.plan(n, attribute=f"attr{n}", value=f"v{n}"))
            rows.append((intent["nonce_hash"], mid(n), auth.confirmed_at))
        self.rig.clock.t = t0 + __import__("datetime").timedelta(seconds=300)
        plan, auth, first = self.attempt()
        self.abort(first)
        self.rig.clock.t = t0 + __import__("datetime").timedelta(seconds=86400 + 60)
        plan2, auth2, second = self.attempt()
        self.rig.install(plan2, auth2)
        after = self.rig.store.snapshot()
        proof = RC.classify_open_intent(second, after)
        self.assertEqual(proof.kind, "COMMITTED", proof)
        live = {r[0] for r in after.nonces}
        self.assertNotIn(rows[0][0], live)                                   # older than the floor: pruned
        self.assertIn(rows[2][0], live)                                      # at or after the upper bound: kept
        self.rig.finish(second)
        self.assertEqual(self.start().row, 6)

    def test_R12_a_crash_inside_the_retry_classifies_only_the_retry(self):
        for label, crash in (("before the journal", "journal"), ("after the commit marker", "marker")):
            with self.subTest(label):
                self.rig = Rig(self)
                self.rig.baseline()
                plan, auth, first = self.attempt()
                self.abort(first)
                plan2, auth2, second = self.attempt()
                if crash == "journal":
                    real = S._write_atomic
                    with mock.patch.object(S, "_write_atomic", side_effect=Crash()), \
                            mock.patch.object(S.MemoryStore, "_rollback", side_effect=Crash()):
                        with self.assertRaises(Crash):
                            self.rig.install(plan2, auth2)
                    a = self.start()
                    self.assertEqual((a.row, a.proof.kind), (7, "NOT_WRITTEN"))
                    self.assertEqual(a.proposed["intent_seq"], second["seq"])
                else:
                    real_unlink = S.os.unlink

                    def dying(path, *args, **kw):
                        if str(path).endswith(".txn.json"):
                            raise Crash()
                        return real_unlink(path, *args, **kw)
                    with mock.patch.object(S.os, "unlink", dying):
                        with self.assertRaises(Crash):
                            self.rig.install(plan2, auth2)
                    self.assertEqual(self.start().code, "snapshot:recovery_pending")
                    self.assertEqual(self.rig.fresh_store().verify(), ())
                    a = self.start()
                    self.assertEqual((a.row, a.proof.kind), (7, "COMMITTED"))
                    self.assertEqual(a.proposed["intent_seq"], second["seq"])
                    self.rig.log.append(a.proposed, self.rig.clock.stamp())
                    self.assertEqual(self.start().row, 6)


# ---- C: concurrency and call order -----------------------------------------------------------------------------------

class ConcurrencyTests(Case):
    def test_C1_a_foreign_install_inside_the_window_is_written_stale_but_can_never_authorize_a_write(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        foreign = self.rig.plan(5, attribute="hobby", value="chess")
        foreign_auth = self.rig.authorization(foreign)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        real_put = RC.ReceiptLog._put

        def foreign_install_then_write(this, data, creating):
            self.rig.last_apply_now = self.rig.clock.stamp()
            self.rig.install(foreign, foreign_auth)                      # a non-cooperating writer, inside the window
            return real_put(this, data, creating)
        with mock.patch.object(RC.ReceiptLog, "_put", foreign_install_then_write):
            intent = self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(intent["action"], "intent")                       # the limitation: the stale intent IS written
        after_foreign = self.rig.memory_files()
        self.rig.last_apply_now = fields["apply_now"]
        with self.assertRaises(S.StoreError) as cm:
            self.rig.install(plan, auth)                                   # but it cannot authorize the install
        self.assertEqual((cm.exception.reason, cm.exception.details), ("authorization_invalid", ("state_changed",)))
        self.assertEqual(self.rig.memory_files(), after_foreign)
        a = self.start()
        self.assertEqual((a.proceed, a.row, a.code), (False, 8, "state_gap"))
        self.assertIsNone(RC.audit(self.rig.rdir, self.rig.fresh_store()).proposed_record)

    def test_C2_a_foreign_nonce_only_change_installs_but_is_proved_inconsistent(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        real_put = RC.ReceiptLog._put
        stray = ("c" * 32, mid(99), auth.confirmed_at)

        def foreign_row_then_write(this, data, creating):
            os.makedirs(self.rig.memdir, exist_ok=True)
            open(self.rig.path(".lock"), "w").close()
            with open(self.rig.path(".authorizations"), "w") as f:
                f.write("%s %s %s\n" % stray)
            return real_put(this, data, creating)
        with mock.patch.object(RC.ReceiptLog, "_put", foreign_row_then_write):
            intent = self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        self.rig.last_apply_now = fields["apply_now"]
        self.rig.install(plan, auth)                                       # the Authorization does not bind nonce rows
        proof = RC.classify_open_intent(intent, self.rig.store.snapshot())
        self.assertEqual(proof.kind, "INCONSISTENT")
        self.assertIn("NONCE_ROW_MISMATCH", proof.codes)
        a = self.start()
        self.assertEqual((a.row, a.proof.kind), (7, "INCONSISTENT"))
        self.assertEqual(a.proposed["action"], "anomaly")
        self.assertEqual(RC.audit(self.rig.rdir, self.rig.fresh_store()).verdict, "BLOCKED")

    def test_C3_a_second_committer_is_refused_while_the_first_holds_the_lock(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        other = self.rig.plan(2, attribute="employer", value="Acme")
        other_auth = self.rig.authorization(other)
        in_check, release = threading.Event(), threading.Event()
        real_check, outcome = RC.check_new_intent, {}

        def slow_check(state, snapshot, fields):
            if threading.current_thread().name == "first":
                in_check.set()
                self.assertTrue(release.wait(10))
            return real_check(state, snapshot, fields)

        def first():
            outcome["rec"] = self.rig.begin(plan, auth)
        with mock.patch.object(RC, "check_new_intent", slow_check):
            t = threading.Thread(target=first, name="first", daemon=True)
            t.start()
            self.assertTrue(in_check.wait(10))
            before = read_log_bytes(self.rig)
            snap = self.rig.store.snapshot()
            fields = RC.intent_fields(snap, other, nonce=other_auth.nonce, confirmed_at=other_auth.confirmed_at,
                                      apply_now=self.rig.clock.stamp())
            self.refused("receipts_busy", (), self.rig.log.append_intent, fields, self.rig.clock.stamp(),
                         self.rig.store)
            self.assertEqual(read_log_bytes(self.rig), before)
            release.set()
            t.join(10)
        self.assertEqual(outcome["rec"]["action"], "intent")
        self.assertEqual(self.actions(), ["open", "intent"])

    def test_C4_a_second_intent_is_never_written_while_one_is_open(self):
        self.rig.baseline()
        plan, auth, first = self.attempt()
        other = self.rig.plan(2, attribute="employer", value="Acme")
        before = read_log_bytes(self.rig)
        self.refused("record_refused", ("INTENT_OPEN",), self.rig.begin, other, self.rig.authorization(other))
        self.assertEqual(read_log_bytes(self.rig), before)

    def test_C5_the_order_is_lock_then_read_then_snapshot_then_check_then_write(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        events = []
        real_flock, real_read, real_check = RC._IO.flock, RC._read_records, RC.check_new_intent
        real_write, real_snapshot = RC._IO.write, self.rig.store.snapshot
        with mock.patch.object(RC._IO, "flock", lambda fd, op: (events.append("lock"), real_flock(fd, op))[1]), \
                mock.patch.object(RC, "_read_records", lambda path: (events.append("read"), real_read(path))[1]), \
                mock.patch.object(self.rig.store, "snapshot", lambda **k: (events.append("snapshot"), real_snapshot(**k))[1]), \
                mock.patch.object(RC, "check_new_intent",
                                  lambda *a: (events.append("check"), real_check(*a))[1]), \
                mock.patch.object(RC._IO, "write", lambda fd, d: (events.append("write"), real_write(fd, d))[1]):
            self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(events, ["lock", "read", "snapshot", "check", "write"])


# ---- G: the only door ------------------------------------------------------------------------------------------------

class GuardTests(Case):
    def valid_fields(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        return plan, auth, RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce,
                                            confirmed_at=auth.confirmed_at, apply_now=self.rig.clock.stamp())

    def test_G1_ordinary_append_cannot_write_an_intent_even_when_everything_is_valid(self):
        plan, auth, fields = self.valid_fields()
        before = tree_state(self.rig.rdir)
        self.refused("use_append_intent", (), self.rig.log.append, fields, self.rig.clock.stamp())
        self.assertEqual(tree_state(self.rig.rdir), before)
        self.assertFalse(self.rig.log.poisoned)
        self.assertEqual(self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)["action"], "intent")

    def test_G1c_append_refuses_before_it_reaches_the_writer(self):
        plan, auth, fields = self.valid_fields()
        with mock.patch.object(RC.ReceiptLog, "_write", side_effect=AssertionError("append reached the writer")):
            self.refused("use_append_intent", (), self.rig.log.append, fields, self.rig.clock.stamp())

    def test_G1b_the_private_writer_refuses_an_intent_it_was_not_given_a_store_for(self):
        plan, auth, fields = self.valid_fields()
        self.refused("use_append_intent", (), self.rig.log._write, fields, self.rig.clock.stamp(), None, False)

    def test_G2_a_refusal_from_the_live_check_stops_the_write_and_changes_nothing(self):
        plan, auth, fields = self.valid_fields()
        before = tree_state(self.rig.rdir)
        with mock.patch.object(RC, "check_new_intent", return_value="forced"):
            self.refused("intent_refused", ("forced",), self.rig.log.append_intent, fields, self.rig.clock.stamp(),
                         self.rig.store)
        self.assertEqual(tree_state(self.rig.rdir), before)
        self.assertFalse(self.rig.log.poisoned)
        self.assertFalse(os.path.exists(self.rig.rpath(RC.BLOCKED)))

    def test_G3_the_check_is_invoked_exactly_once_with_the_state_a_fresh_snapshot_and_the_fields(self):
        plan, auth, fields = self.valid_fields()
        seen = []
        real = RC.check_new_intent
        with mock.patch.object(RC, "check_new_intent", lambda *a: (seen.append(a), real(*a))[1]):
            self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(len(seen), 1)
        state, snapshot, got = seen[0]
        self.assertIsInstance(state, RC.LogState)
        self.assertTrue(isinstance(snapshot, S.Snapshot) and snapshot.ok)
        self.assertIs(got, fields)

    def test_G4_structure_only_append_intent_reaches_the_intent_write_and_it_checks_first(self):
        tree = ast.parse(inspect.getsource(RC))
        log = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReceiptLog")
        methods = {n.name: n for n in log.body if isinstance(n, ast.FunctionDef)}
        callers = {}
        for name, fn in methods.items():
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_write":
                    callers.setdefault(name, []).append({k.arg for k in n.keywords})
        self.assertEqual(set(callers), {"create", "append", "append_intent"})
        passes_store = {name for name, kws in callers.items() if any("store" in k for k in kws)}
        self.assertEqual(passes_store, {"append_intent"})
        used = {n.id for n in ast.walk(methods["_write"]) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(methods["_write"]) if isinstance(n, ast.Attribute)}
        self.assertTrue({"check_new_intent", "snapshot", "validate_records"} <= used)
        src = inspect.getsource(RC.ReceiptLog._write)
        self.assertLess(src.index("self.records()"), src.index("store.snapshot()"))
        self.assertLess(src.index("store.snapshot()"), src.index("self._put("))
        self.assertIn("use_append_intent", inspect.getsource(RC.ReceiptLog.append))

    def test_G5_live_refusals_through_the_only_door_leave_the_file_unchanged(self):
        plan, auth, fields = self.valid_fields()
        scenarios = {}
        self.rig.seed(1)                                     # the memory file is already there
        scenarios["memory_file_present"] = None
        before = tree_state(self.rig.rdir)
        self.refused("intent_refused", ("memory_file_present",), self.rig.log.append_intent, fields,
                     self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(tree_state(self.rig.rdir), before)
        os.unlink(self.rig.path(mid(1) + ".md"))
        self.rig.seed(5, attribute="hobby", value="chess")   # the store moved since the log closed
        self.refused("intent_refused", ("state_gap",), self.rig.log.append_intent, fields, self.rig.clock.stamp(),
                     self.rig.store)
        self.assertEqual(tree_state(self.rig.rdir), before)
        os.unlink(self.rig.path(mid(5) + ".md"))
        with open(self.rig.path(".authorizations"), "w") as f:       # a nonce-only change since the snapshot
            f.write("%s %s %s\n" % ("a" * 32, mid(9), "2026-01-31T12:00:00Z"))
        self.refused("intent_refused", ("nonce_rows_changed",), self.rig.log.append_intent, fields,
                     self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(tree_state(self.rig.rdir), before)

    def test_G6_the_harness_door_is_the_guarded_one(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        with mock.patch.object(RC.ReceiptLog, "append_intent", autospec=True,
                               side_effect=RC.ReceiptLog.append_intent) as door:
            self.rig.begin(plan, auth)
        self.assertEqual(door.call_count, 1)

    def test_G7_append_intent_refuses_non_intents_and_a_missing_store(self):
        plan, auth, fields = self.valid_fields()
        self.refused("bad_fields", (), self.rig.log.append_intent, RC.aborted_fields(0, CLEAN, "a" * 64),
                     self.rig.clock.stamp(), self.rig.store)
        self.refused("bad_fields", (), self.rig.log.append_intent, "nope", self.rig.clock.stamp(), self.rig.store)
        self.refused("bad_fields", ("store",), self.rig.log.append_intent, fields, self.rig.clock.stamp(), None)


if __name__ == "__main__":
    unittest.main()
