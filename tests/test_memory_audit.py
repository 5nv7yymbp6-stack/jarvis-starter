"""Memory 2.0, Phase 8.2c-2: the proof, the startup matrix, the audit and the crash-injection matrix
(backtalk/memory_receipts.py). Every test runs on a throwaway vault and receipts folder in a temp directory with a
deterministic clock; nothing reads or writes the real vault, and the harness composes the commit sequence ONLY here.

WHAT THESE TESTS DO NOT SHOW. A crash is simulated by an exception at a chosen step (or an injected I/O failure), not by a
real process kill. The hashes are unkeyed: code running as the same user can forge a consistent log and store, and a
consistent rollback of both together is a valid earlier state (pinned below as a limit, not a bug)."""
import ast
import dataclasses
import fcntl
import glob
import inspect
import itertools
import os
import stat
import unittest
from datetime import timedelta
from unittest import mock

from backtalk import memory as M
from backtalk import memory_receipts as RC
from backtalk import memory_review as R
from backtalk import memory_store as S
from tests.receipts_harness import (Crash, Rig, day, envelope_free, ev, item, mid, read_log_bytes, snapshot_of,
                                    tree_state, write_log)

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtalk")
H = {"a": "a" * 32, "b": "b" * 32, "c": "c" * 32, "d": "d" * 32}


class Case(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(self)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))

    # a fresh process: new store object, new log object, no shared state
    def start(self):
        return RC.assess(self.rig.rdir, self.rig.fresh_store())

    def prepare(self, supersede):
        """A baseline and, when asked, one RECEIPTED commit of an older memory that the next install will supersede."""
        self.rig.baseline()
        if supersede:
            self.rig.commit(self.rig.plan(2, value="Kansas City", valid_from=day(1), created=day(1), updated=day(1)))

    def audit(self):
        return RC.audit(self.rig.rdir, self.rig.fresh_store())

    def assertBlocked(self, row, code, a=None):
        a = a or self.start()
        self.assertFalse(a.proceed)
        self.assertFalse(a.first_use)
        self.assertEqual((a.row, a.code), (row, code))
        return a

    def committed(self, n=1, **over):
        """A baseline plus one fully receipted commit; returns (intent, done, auth)."""
        if not os.path.exists(self.rig.rpath()):
            self.rig.baseline()
        return self.rig.commit(self.rig.plan(n, **over))

    def open_intent_before(self, n=1, **over):
        """A baseline plus an intent that was appended but whose install never started."""
        self.rig.baseline()
        plan = self.rig.plan(n, **over)
        auth = self.rig.authorization(plan)
        return plan, auth, self.rig.begin(plan, auth)


def rows_at(c, *offsets):
    """Nonce rows whose confirmed_at is `offset` seconds from `c`, with distinct hashes."""
    base = M.parse_ts(c)
    out = []
    for i, off in enumerate(offsets):
        at = (base + timedelta(seconds=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append((f"{i + 1:032x}", mid(100 + i), at))
    return out


# ---- the proof ------------------------------------------------------------------------------------------------------

class ProofTests(Case):
    def setup_pair(self, with_unrelated=True):
        """(intent fields, snapshot before, snapshot after) from one real authorized install, receipts not involved."""
        if with_unrelated:
            self.rig.seed(5, attribute="hobby", value="chess")
        self.rig.seed(2)
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        before = self.rig.store.snapshot()
        fields = RC.intent_fields(before, plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        self.rig.last_apply_now = fields["apply_now"]
        self.rig.install(plan, auth)
        after = self.rig.store.snapshot()
        return fields, before, after, plan, auth

    def mutate(self, snap, *, files=None, add_files=(), drop_files=(), nonces=None):
        f = dict(snap.files)
        f.update(files or {})
        f.update(dict(add_files))
        for n in drop_files:
            f.pop(n)
        return snapshot_of(f.items(), snap.nonces if nonces is None else nonces)

    def test_the_two_real_states_classify_as_committed_and_not_written(self):
        fields, before, after, plan, auth = self.setup_pair()
        self.assertEqual(RC.classify_open_intent(fields, after), RC.Proof("COMMITTED", ()))
        self.assertEqual(RC.classify_open_intent(fields, before), RC.Proof("NOT_WRITTEN", ()))
        self.assertEqual(after.nonces[-1][0], fields["nonce_hash"])

    def test_every_single_contradiction_is_inconsistent_on_both_sides(self):
        fields, before, after, plan, auth = self.setup_pair()
        sha = "e" * 64
        unrelated = mid(5) + ".md"
        old = mid(2) + ".md"
        new = mid(1) + ".md"
        extra_row = (H["d"], mid(77), "2026-01-31T12:00:00Z")
        for side_name, snap in (("before", before), ("after", after)):
            cases = {
                "extra_file": self.mutate(snap, add_files=[(mid(9) + ".md", sha)]),
                "deleted_unrelated": self.mutate(snap, drop_files=[unrelated]),
                "modified_unrelated": self.mutate(snap, files={unrelated: sha}),
                "extra_nonce_row": self.mutate(snap, nonces=tuple(snap.nonces) + (extra_row,)),
            }
            if side_name == "after":
                cases["expected_file_wrong_sha"] = self.mutate(snap, files={new: sha})
                cases["superseded_file_wrong_sha"] = self.mutate(snap, files={old: sha})
                cases["required_nonce_row_missing"] = self.mutate(snap, nonces=())
                cases["new_file_missing"] = self.mutate(snap, drop_files=[new])
            else:
                cases["new_file_present_nonce_absent"] = self.mutate(snap, add_files=[(new, dict(after.files)[new])])
                cases["old_file_ended_early"] = self.mutate(snap, files={old: dict(after.files)[old]})
            for label, mutated in cases.items():
                with self.subTest(side=side_name, case=label):
                    proof = RC.classify_open_intent(fields, mutated)
                    self.assertEqual(proof.kind, "INCONSISTENT", (label, proof))
                    self.assertTrue(proof.codes)

    def test_specific_codes(self):
        fields, before, after, plan, auth = self.setup_pair()
        sha = "e" * 64
        by = lambda snap, **kw: RC.classify_open_intent(fields, self.mutate(snap, **kw)).codes
        self.assertIn("UNRECEIPTED_FILE", by(after, add_files=[(mid(9) + ".md", sha)]))
        self.assertIn("MISSING_FILE", by(after, drop_files=[mid(5) + ".md"]))
        self.assertIn("MODIFIED_FILE", by(after, files={mid(5) + ".md": sha}))
        self.assertIn("DIGEST_MISMATCH", by(after, files={mid(5) + ".md": sha}))
        self.assertIn("NONCE_ROW_MISMATCH", by(after, nonces=()))
        self.assertIn("NONCE_ROWS_CHANGED", by(before, nonces=((H["d"], mid(8), "2026-01-31T00:00:00Z"),)))

    def test_a_snapshot_that_is_not_ok_never_proves_anything(self):
        fields, before, after, plan, auth = self.setup_pair()
        bad = S.Snapshot(False, "lock_busy", None, None, None, None, None, None)
        self.assertEqual(RC.classify_open_intent(fields, bad), RC.Proof("INCONSISTENT", ("SNAPSHOT_NOT_OK",)))
        self.assertEqual(RC.classify_open_intent(fields, "nope"), RC.Proof("INCONSISTENT", ("SNAPSHOT_NOT_OK",)))

    def test_state_digest_alone_is_never_the_proof(self):
        """A snapshot whose digest field equals the intent's after-digest but whose files differ is not COMMITTED."""
        fields, before, after, plan, auth = self.setup_pair()
        liar = S.Snapshot(True, "ok", "initialized", before.files, before.items, tuple(after.nonces),
                          fields["state_digest_after"], after.snapshot_digest)
        self.assertEqual(RC.classify_open_intent(fields, liar).kind, "INCONSISTENT")
        liar_before = S.Snapshot(True, "ok", "initialized", after.files, after.items, before.nonces,
                                 fields["state_digest_before"], before.snapshot_digest)
        self.assertEqual(RC.classify_open_intent(fields, liar_before).kind, "INCONSISTENT")

    def test_the_prune_bounds_are_exact(self):
        self.rig.seed(2)
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        c = auth.confirmed_at
        old_must_go, band_lo, band_hi, young_must_stay = rows_at(c, -86400 - 1, -86400, -86400 + 119, -86400 + 120)
        rows = [old_must_go, band_lo, band_hi, young_must_stay]
        before = self.rig.store.snapshot()
        before = snapshot_of(before.files, rows, before.items)
        fields = RC.intent_fields(before, plan, nonce=auth.nonce, confirmed_at=c, apply_now=c)
        new_row = (fields["nonce_hash"], plan.item.id, c)
        after_files = dict(fields["manifest_after"])

        def verdict(kept):
            return RC.classify_open_intent(fields, snapshot_of(after_files.items(), list(kept) + [new_row])).kind
        self.assertEqual(verdict([young_must_stay]), "COMMITTED")                               # band rows pruned
        self.assertEqual(verdict([band_lo, band_hi, young_must_stay]), "COMMITTED")             # band rows kept
        self.assertEqual(verdict([band_lo, young_must_stay]), "COMMITTED")                      # band is either way
        self.assertEqual(verdict([old_must_go, young_must_stay]), "INCONSISTENT")               # older than the floor, kept
        self.assertEqual(verdict([band_lo, band_hi]), "INCONSISTENT")                           # young row pruned too early
        self.assertEqual(verdict([]), "INCONSISTENT")
        codes = RC.classify_open_intent(fields, snapshot_of(after_files.items(), [old_must_go, young_must_stay, new_row]))
        self.assertIn("NONCE_PRUNE_VIOLATION", codes.codes)
        young_gone = RC.classify_open_intent(fields, snapshot_of(after_files.items(), [band_lo, band_hi, new_row]))
        self.assertIn("NONCE_PRUNE_VIOLATION", young_gone.codes)

    def test_the_retention_constant_is_the_stores(self):
        self.assertEqual(RC.NONCE_KEEP, S._AUTH_KEEP)

    def test_not_written_requires_the_nonce_rows_unchanged_including_young_ones(self):
        self.rig.seed(2)
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        rows = rows_at(auth.confirmed_at, -10, -86400 - 5)
        before = self.rig.store.snapshot()
        before = snapshot_of(before.files, rows, before.items)
        fields = RC.intent_fields(before, plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=auth.confirmed_at)
        files = dict(before.files)
        self.assertEqual(RC.classify_open_intent(fields, snapshot_of(files.items(), rows)).kind, "NOT_WRITTEN")
        self.assertEqual(RC.classify_open_intent(fields, snapshot_of(files.items(), rows[:1])).kind, "INCONSISTENT")


# ---- the startup matrix --------------------------------------------------------------------------------------------

class StartupMatrixTests(Case):
    def test_row_6_a_clean_log_proceeds_after_a_commit_and_after_an_abort(self):
        self.committed()
        a = self.start()
        self.assertEqual((a.proceed, a.row, a.code, a.first_use), (True, 6, "clean", False))
        plan = self.rig.plan(2, attribute="employer", value="Acme")
        auth = self.rig.authorization(plan)
        intent = self.rig.begin(plan, auth)
        self.rig.log.append(RC.aborted_fields(intent["seq"], "crashed_before_write", self.rig.store.state_digest()),
                            self.rig.clock.stamp())
        self.assertEqual(self.start().row, 6)

    def test_row_3_first_use_with_an_absent_empty_or_lock_only_store(self):
        for label, arrange in (("absent", lambda: None), ("empty", lambda: os.makedirs(self.rig.memdir)),
                               ("lock only", lambda: (os.makedirs(self.rig.memdir, exist_ok=True),
                                                      open(self.rig.path(".lock"), "wb").close()))):
            with self.subTest(label):
                arrange()
                before = tree_state(self.rig.root)
                a = self.start()
                self.assertEqual((a.proceed, a.row, a.code, a.first_use), (True, 3, "first_use", True))
                self.assertEqual(tree_state(self.rig.root), before)
                self.assertFalse(os.path.exists(self.rig.rpath()))

    def test_row_3b_never_baselines_an_existing_store(self):
        self.rig.seed(2)
        self.assertBlocked(3, "receipts_missing")
        self.assertFalse(os.path.exists(self.rig.rpath()))
        self.assertTrue(self.rig.log is not None)
        os.unlink(self.rig.path(mid(2) + ".md"))
        with open(self.rig.path(".authorizations"), "w") as f:
            f.write("%s %s 2026-01-31T00:00:00Z\n" % (H["a"], mid(2)))
        self.assertBlocked(3, "receipts_missing")                            # a lone nonce row is not an empty store

    def test_missing_receipts_with_an_unreadable_store_is_the_snapshot_reason(self):
        os.makedirs(self.rig.memdir)
        open(self.rig.path(".lock"), "w").close()
        open(self.rig.path("notes.txt"), "w").close()
        self.assertBlocked(5, "snapshot:store_unsound")
        os.unlink(self.rig.path(".lock"))
        self.assertBlocked(5, "snapshot:lock_missing")

    def test_row_1_the_marker_in_every_state_blocks(self):
        self.committed()
        for label, make in (("file", lambda: open(self.rig.rpath(RC.BLOCKED), "w").close()),
                            ("directory", lambda: os.mkdir(self.rig.rpath(RC.BLOCKED))),
                            ("symlink", lambda: os.symlink("/nonexistent", self.rig.rpath(RC.BLOCKED)))):
            with self.subTest(label):
                make()
                self.assertBlocked(1, "receipts_blocked")
                if os.path.isdir(self.rig.rpath(RC.BLOCKED)) and not os.path.islink(self.rig.rpath(RC.BLOCKED)):
                    os.rmdir(self.rig.rpath(RC.BLOCKED))
                else:
                    os.unlink(self.rig.rpath(RC.BLOCKED))
        real = os.lstat

        def eio(path, *a, **k):
            if str(path).endswith(RC.BLOCKED):
                raise OSError(5, "I/O error")
            return real(path, *a, **k)
        with mock.patch.object(RC._IO, "lstat", eio):
            self.assertBlocked(1, "receipts_blocked")                        # unreadable counts as present
        self.assertEqual(self.start().row, 6)

    def test_row_2_the_folder_and_file_problems_block(self):
        self.committed()
        os.chmod(self.rig.rdir, 0o755)
        self.assertBlocked(2, "receipts_unavailable")
        os.chmod(self.rig.rdir, 0o700)
        os.chmod(self.rig.rpath(), 0o644)
        self.assertBlocked(2, "receipts_unavailable")
        os.chmod(self.rig.rpath(), 0o600)
        os.rename(self.rig.rpath(RC.LOCK), self.rig.rpath(RC.LOCK) + ".x")
        self.assertBlocked(2, "receipts_lock_missing")
        os.rename(self.rig.rpath(RC.LOCK) + ".x", self.rig.rpath(RC.LOCK))
        import fcntl
        fd = os.open(self.rig.rpath(RC.LOCK), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.assertBlocked(2, "receipts_busy")
        finally:
            os.close(fd)
        os.rename(self.rig.rpath(), self.rig.rpath() + ".real")
        os.symlink(self.rig.rpath() + ".real", self.rig.rpath())
        self.assertBlocked(2, "receipts_unavailable")
        os.unlink(self.rig.rpath())
        os.rename(self.rig.rpath() + ".real", self.rig.rpath())
        gone = Rig(self)
        os.rmdir(gone.rdir)
        a = RC.startup_check(gone.rdir, gone.store)
        self.assertEqual((a.proceed, a.row, a.code), (False, 2, "receipts_unavailable"))
        self.assertEqual(self.start().row, 6)

    def test_row_4_every_mid_record_truncation_is_corrupt_and_nothing_is_repaired(self):
        self.committed()
        data = read_log_bytes(self.rig)
        ends = [i + 1 for i, b in enumerate(data) if b == 0x0A]
        for cut in sorted({1, 17, ends[0] - 1, ends[0] + 1, ends[0] + 40, ends[1] - 1, ends[1] + 5, ends[2] - 1}):
            with open(self.rig.rpath(), "wb") as f:
                f.write(data[:cut])
            with self.subTest(cut=cut):
                self.assertBlocked(4, "receipts_corrupt")
                with open(self.rig.rpath(), "rb") as f:
                    self.assertEqual(f.read(), data[:cut])
        for label, bad in (("empty", b""), ("blank line", data + b"\n"), ("invalid utf8", b"\xff\xfe\n"),
                           ("spaces", data.replace(b'":', b'": ', 1)), ("not json", b"hello\n")):
            with open(self.rig.rpath(), "wb") as f:
                f.write(bad)
            with self.subTest(label):
                self.assertBlocked(4, "receipts_corrupt")

    def test_cuts_at_a_record_boundary_leave_a_valid_log_that_the_store_then_contradicts(self):
        intent, done, auth = self.committed()
        data = read_log_bytes(self.rig)
        ends = [i + 1 for i, b in enumerate(data) if b == 0x0A]
        with open(self.rig.rpath(), "wb") as f:
            f.write(data[:ends[1]])                                   # open + intent: the done was deleted
        a = self.assertBlocked(7, "reconcile_required")
        self.assertEqual(a.details, ())
        self.assertEqual(self.audit().proposed_record["action"], "done")
        with open(self.rig.rpath(), "wb") as f:
            f.write(data[:ends[0]])                                   # open only: intent and done both deleted
        self.assertBlocked(8, "state_gap")

    def test_row_5_a_snapshot_that_is_not_ok_blocks_with_its_reason(self):
        self.committed()
        with self.rig.store._locked(write=True):
            a = RC.startup_check(self.rig.rdir, self.rig.store)
        self.assertEqual((a.row, a.code), (5, "snapshot:lock_busy"))
        with open(self.rig.path(".txn.json"), "w") as f:
            f.write('{"state": "prepared", "ops": []}\n')
        self.assertBlocked(5, "snapshot:recovery_pending")
        os.unlink(self.rig.path(".txn.json"))
        open(self.rig.path("stray.txt"), "w").close()
        self.assertBlocked(5, "snapshot:store_unsound")

    def test_row_7_one_open_intent_and_a_store_at_its_before_or_its_after(self):
        plan, auth, intent = self.open_intent_before()
        a = self.assertBlocked(7, "reconcile_required")
        self.assertEqual(a.proof.kind, "NOT_WRITTEN")
        self.rig.install(plan, auth)
        a = self.assertBlocked(7, "reconcile_required")
        self.assertEqual(a.proof.kind, "COMMITTED")

    def test_startup_check_is_the_projection_of_the_same_assessment(self):
        self.committed()
        a, s = RC.assess(self.rig.rdir, self.rig.fresh_store()), RC.startup_check(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual((s.proceed, s.row, s.code, s.first_use, s.details), (a.proceed, a.row, a.code, a.first_use,
                                                                              a.details))

    def test_row_8_every_other_relation_is_a_state_gap(self):
        self.committed()
        name = self.rig.path(mid(1) + ".md")
        with open(name, "rb") as f:
            good = f.read()
        with open(name, "wb") as f:                                   # a manual edit
            f.write(good.replace(b"San Jose", b"San Jos2"))
        self.assertBlocked(8, "state_gap")
        with open(name, "wb") as f:
            f.write(good)
        self.assertEqual(self.start().row, 6)
        os.unlink(name)                                               # a deleted memory file
        self.assertBlocked(8, "state_gap")
        with open(name, "wb") as f:
            f.write(good)
        extra = S.serialize(item(9, attribute="employer", value="Acme"), ev("Acme"))
        with open(self.rig.path(mid(9) + ".md"), "w") as f:           # an extra, valid memory file
            f.write(extra)
        self.assertBlocked(8, "state_gap")

    def test_row_9_a_nonce_row_with_no_intent_blocks(self):
        self.committed()
        with open(self.rig.path(".authorizations"), "a") as f:
            f.write("%s %s 2026-01-31T12:00:00Z\n" % (H["d"], mid(1)))
        a = self.assertBlocked(9, "unreceipted_commit")
        self.assertEqual(a.details, (H["d"],))

    def test_a_recorded_global_anomaly_blocks_even_when_the_store_matches(self):
        intent, done, auth = self.committed()
        recs = self.rig.log.records()
        fields = [envelope_free(r) for r in recs] + [RC.anomaly_fields(None, ["MANUAL_NOTE"], None)]
        write_log(self.rig, fields)
        self.assertBlocked(8, "anomaly_recorded")

    def test_deleting_an_anomaly_record_does_not_lift_a_block_derived_from_state(self):
        plan, auth, intent = self.open_intent_before()
        recs = [envelope_free(r) for r in self.rig.log.records()]
        write_log(self.rig, recs + [RC.anomaly_fields(intent["seq"], ["STATE_GAP"], None)])
        self.assertBlocked(7, "reconcile_required")
        write_log(self.rig, recs)                                     # the anomaly record removed
        self.assertBlocked(7, "reconcile_required")

    def test_two_open_intents_cannot_be_written_and_a_forged_pair_is_corrupt(self):
        plan, auth, intent = self.open_intent_before()
        second = RC.intent_fields(self.rig.store.snapshot(), self.rig.plan(2, attribute="employer", value="Acme"),
                                  nonce="zz", confirmed_at=self.rig.clock.stamp(), apply_now=self.rig.clock.stamp())
        self.assertEqual(self.refused_append(second), "record_refused")
        write_log(self.rig, [envelope_free(r) for r in self.rig.log.records()] + [second])
        a = self.assertBlocked(4, "receipts_corrupt")
        self.assertEqual(a.details, ("INTENT_OPEN",))

    def refused_append(self, fields):
        with self.assertRaises(RC.ReceiptError) as cm:
            self.rig.log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        return cm.exception.reason

    # -- limits, pinned so they are not hidden --

    def test_limit_deleting_a_trailing_intent_and_aborted_pair_is_undetected(self):
        self.committed()
        before = read_log_bytes(self.rig)
        plan = self.rig.plan(2, attribute="employer", value="Acme")
        intent = self.rig.begin(plan, self.rig.authorization(plan))
        self.rig.log.append(RC.aborted_fields(intent["seq"], "crashed_before_write", self.rig.store.state_digest()),
                            self.rig.clock.stamp())
        self.assertGreater(len(read_log_bytes(self.rig)), len(before))
        with open(self.rig.rpath(), "wb") as f:
            f.write(before)                                           # history only: no store change to notice
        self.assertEqual(self.start().row, 6)

    def test_limit_a_consistent_rollback_of_store_and_log_together_passes(self):
        self.committed()
        snap_files, snap_log = self.rig.memory_files(), read_log_bytes(self.rig)
        self.committed(2, attribute="employer", value="Acme")
        for name in os.listdir(self.rig.memdir):
            os.unlink(self.rig.path(name)) if os.path.isfile(self.rig.path(name)) else None
        for name, data in snap_files.items():
            with open(self.rig.path(name), "wb") as f:
                f.write(data)
        with open(self.rig.rpath(), "wb") as f:
            f.write(snap_log)
        self.assertEqual(self.start().row, 6)


# ---- audit ---------------------------------------------------------------------------------------------------------

class AuditTests(Case):
    def test_clean_first_use_and_clean_log(self):
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.code, a.proposed_record), ("CLEAN", 3, "first_use", None))
        self.committed()
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.proposed_record), ("CLEAN", 6, None))
        self.assertTrue(a.facts)

    def test_reconcile_required_proposes_exactly_one_record_that_it_never_appends(self):
        plan, auth, intent = self.open_intent_before()
        before_bytes = read_log_bytes(self.rig)
        a = self.audit()
        self.assertEqual((a.verdict, a.proof.kind), ("RECONCILE_REQUIRED", "NOT_WRITTEN"))
        self.assertEqual(a.proposed_record, RC.aborted_fields(intent["seq"], "crashed_before_write",
                                                              self.rig.store.state_digest()))
        self.assertTrue(a.inferences and a.facts)
        self.assertEqual(read_log_bytes(self.rig), before_bytes)
        self.rig.install(plan, auth)
        a = self.audit()
        self.assertEqual((a.verdict, a.proof.kind), ("RECONCILE_REQUIRED", "COMMITTED"))
        self.assertEqual(a.proposed_record["action"], "done")
        self.assertTrue(a.proposed_record["recovered"])
        self.assertEqual(read_log_bytes(self.rig), before_bytes)

    def test_a_proposed_record_is_appendable_and_closes_the_gap(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        proposed = self.audit().proposed_record
        self.rig.log.append(proposed, self.rig.clock.stamp())          # a person's act, done by the test
        self.assertEqual(self.start().row, 6)
        self.assertEqual(self.audit().verdict, "CLEAN")

    def test_an_inconsistent_proof_is_blocked_with_an_anomaly_proposal(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        os.unlink(self.rig.path(mid(1) + ".md"))                        # files are at "before", the nonce row is "after"
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.proof.kind), ("BLOCKED", 7, "INCONSISTENT"))
        self.assertEqual(a.proposed_record["action"], "anomaly")
        self.assertIn("NONCE_ROWS_CHANGED", a.proposed_record["codes"])        # the files say "before", the row says "after"

    def test_a_store_that_is_neither_side_is_a_state_gap_with_no_proposal(self):
        self.committed()
        plan = self.rig.plan(2, attribute="employer", value="Acme")
        self.rig.begin(plan, self.rig.authorization(plan))
        with open(self.rig.path(mid(1) + ".md"), "w") as f:             # a valid file with different content
            f.write(S.serialize(item(1, value="Fresno"), ev("Fresno")))
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.code, a.proposed_record), ("BLOCKED", 8, "state_gap", None))

    def test_inconsistent_files_with_a_matching_digest_cannot_exist_but_nonce_mismatch_is_caught(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        os.unlink(self.rig.path(".authorizations"))                     # files == after, nonce row gone
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.proof.kind), ("BLOCKED", 7, "INCONSISTENT"))
        self.assertEqual(a.proposed_record["action"], "anomaly")
        self.assertIn("NONCE_ROW_MISMATCH", a.proposed_record["codes"])

    def test_blocked_reports_have_no_proposed_record(self):
        self.committed()
        with open(self.rig.rpath(RC.BLOCKED), "w"):
            pass
        a = self.audit()
        self.assertEqual((a.verdict, a.row, a.proposed_record), ("BLOCKED", 1, None))

    def corrupt_tail(self):
        with open(self.rig.rpath(), "ab") as f:
            f.write(b"{")

    def make_marker(self):
        with open(self.rig.rpath(RC.BLOCKED), "w"):
            pass

    def test_audit_and_startup_check_change_nothing_anywhere(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        for label, mutate in (("open intent", lambda: None),
                              ("corrupt tail", self.corrupt_tail), ("marker", self.make_marker)):
            mutate()
            before = (tree_state(self.rig.root), self.rig.memory_files())
            with self.subTest(label):
                RC.audit(self.rig.rdir, self.rig.fresh_store())
                RC.startup_check(self.rig.rdir, self.rig.fresh_store())
                self.assertEqual((tree_state(self.rig.root), self.rig.memory_files()), before)

    def test_no_write_flag_and_no_write_call_is_ever_made_by_the_read_only_functions(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        flags = []
        real_open = RC._IO.open

        def spy(path, fl, *a):
            flags.append(fl)
            return real_open(path, fl, *a)
        writes = mock.Mock(side_effect=AssertionError("a read-only function wrote"))
        with mock.patch.object(RC._IO, "open", spy), mock.patch.object(RC._IO, "write", writes), \
                mock.patch.object(RC._IO, "fsync", writes):
            RC.audit(self.rig.rdir, self.rig.fresh_store())
            RC.startup_check(self.rig.rdir, self.rig.fresh_store())
        self.assertTrue(flags)
        for fl in flags:
            self.assertFalse(fl & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND), fl)


# ---- the crash matrix ----------------------------------------------------------------------------------------------

class CrashMatrixTests(Case):
    """The commit sequence with a failure injected at every step. A "fresh process" is a new store and log object."""

    def journal_writes(self, plan):
        """How many `_write_atomic` calls an authorized install makes: journal, one per file, commit marker."""
        return 1 + len([1 for _ in range(1 + len(plan.decision.supersedes) + 1)]) + 1

    def install_dying_at(self, plan, auth, crash_at):
        real, count = S._write_atomic, [0]

        def dying(path, text):
            count[0] += 1
            if count[0] == crash_at:
                raise Crash()
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", dying), \
                mock.patch.object(S.MemoryStore, "_rollback", side_effect=Crash()):
            with self.assertRaises(Crash):
                self.rig.install(plan, auth)

    def recover(self):
        """The store's own recovery, run by an ordinary operation, as the design says."""
        fresh = self.rig.fresh_store()
        self.assertEqual(fresh.verify(), ())

    def test_A1_a_crash_before_the_intent_is_written_leaves_nothing_to_reconcile(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        self.rig.authorization(plan)
        snap = self.rig.store.snapshot()
        RC.intent_fields(snap, plan, nonce="n", confirmed_at=self.rig.clock.stamp(), apply_now=self.rig.clock.stamp())
        a = self.start()                                              # the process died before append
        self.assertEqual((a.proceed, a.row), (True, 6))

    def test_A2_a_torn_intent_is_corrupt_and_the_store_was_never_touched(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        snap = self.rig.store.snapshot()
        fields = RC.intent_fields(snap, plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        log, real, calls = self.rig.log, os.write, [0]

        def half_then_die(fd, data):
            calls[0] += 1
            if calls[0] == 1:
                return real(fd, bytes(data[:len(data) // 2]))
            raise OSError(5, "I/O error")
        files_before = self.rig.memory_files()
        with mock.patch.object(RC._IO, "write", half_then_die):
            with self.assertRaises(RC.ReceiptError):
                log.append_intent(fields, self.rig.clock.stamp(), self.rig.store)
        self.assertEqual(self.rig.memory_files(), files_before)
        self.assertBlocked(1, "receipts_blocked")                      # the marker is evidence and blocks first
        os.unlink(self.rig.rpath(RC.BLOCKED))                           # even without it, the torn tail blocks
        self.assertBlocked(4, "receipts_corrupt")

    def test_A3_A4_A5_a_complete_intent_with_an_fsync_error_means_the_store_is_never_called(self):
        for label, failing_call in (("file fsync", 1),):
            with self.subTest(label):
                rig = Rig(self)
                self.rig = rig
                rig.baseline()
                plan = rig.plan(1)
                auth = rig.authorization(plan)
                fields = RC.intent_fields(rig.store.snapshot(), plan, nonce=auth.nonce,
                                          confirmed_at=auth.confirmed_at, apply_now=rig.clock.stamp())
                log = rig.log
                with mock.patch.object(RC._IO, "fsync", side_effect=OSError(5, "I/O error")):
                    with self.assertRaises(RC.ReceiptError):
                        log.append_intent(fields, rig.clock.stamp(), rig.store)
                self.assertTrue(log.poisoned)
                self.assertBlocked(1, "receipts_blocked")
                os.unlink(rig.rpath(RC.BLOCKED))
                a = self.assertBlocked(7, "reconcile_required")         # the intent did land; the store is at "before"
                self.assertEqual(a.proof.kind, "NOT_WRITTEN")
                self.assertEqual(self.audit().proposed_record["action"], "aborted")

    def test_C1_to_C4_a_crash_during_the_install_is_undone_by_the_stores_own_recovery(self):
        for supersede in (False, True):
            probe = Rig(self)
            if supersede:
                probe.seed(2)
            expected_writes = self.journal_writes(probe.plan(1))
            for crash_at in range(1, expected_writes + 1):             # journal ... commit marker (not yet durable)
                with self.subTest(supersede=supersede, crash_at=crash_at):
                    self.rig = Rig(self)
                    self.prepare(supersede)
                    plan = self.rig.plan(1)
                    self.assertEqual(plan.operation, "apply" if supersede else "put")
                    auth = self.rig.authorization(plan)
                    intent = self.rig.begin(plan, auth)
                    files_before = self.rig.memory_files()
                    self.install_dying_at(plan, auth, crash_at)
                    if crash_at == 1:                                   # no journal yet: the store was never changed
                        a = self.assertBlocked(7, "reconcile_required")
                        self.assertEqual(a.proof.kind, "NOT_WRITTEN")
                        continue
                    self.assertBlocked(5, "snapshot:recovery_pending")  # a journal is left: nothing is classified
                    self.assertEqual(self.audit().verdict, "BLOCKED")
                    self.recover()
                    a = self.assertBlocked(7, "reconcile_required")
                    self.assertEqual(a.proof.kind, "NOT_WRITTEN")
                    self.assertEqual({k: v for k, v in self.rig.memory_files().items() if k.startswith("mem_")},
                                     {k: v for k, v in files_before.items() if k.startswith("mem_")})
                    self.assertEqual(self.audit().proposed_record["intent_seq"], intent["seq"])

    def test_C5_a_crash_after_the_commit_marker_is_completed_and_proves_committed(self):
        for supersede in (False, True):
            with self.subTest(supersede=supersede):
                self.rig = Rig(self)
                self.prepare(supersede)
                plan = self.rig.plan(1)
                auth = self.rig.authorization(plan)
                intent = self.rig.begin(plan, auth)
                real_unlink = S.os.unlink

                def dying_unlink(path, *a, **k):
                    if str(path).endswith(".txn.json"):
                        raise Crash()
                    return real_unlink(path, *a, **k)
                with mock.patch.object(S.os, "unlink", dying_unlink):
                    with self.assertRaises(Crash):
                        self.rig.install(plan, auth)
                self.assertTrue(os.path.exists(self.rig.path(".txn.json")))
                self.assertBlocked(5, "snapshot:recovery_pending")
                self.recover()
                a = self.assertBlocked(7, "reconcile_required")
                self.assertEqual(a.proof.kind, "COMMITTED")
                self.assertEqual(self.audit().proposed_record["action"], "done")

    def test_C6_a_finished_install_with_no_done_record_proves_committed(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        a = self.assertBlocked(7, "reconcile_required")
        self.assertEqual(a.proof.kind, "COMMITTED")
        self.rig.log.append(self.audit().proposed_record, self.rig.clock.stamp())
        self.assertEqual(self.start().row, 6)

    def test_D2_a_torn_done_is_corrupt_and_the_store_is_at_its_after(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        snap = self.rig.store.snapshot()
        log, real, calls = self.rig.log, os.write, [0]

        def half_then_die(fd, data):
            calls[0] += 1
            if calls[0] == 1:
                return real(fd, bytes(data[:len(data) // 2]))
            raise OSError(5, "I/O error")
        with mock.patch.object(RC._IO, "write", half_then_die):
            with self.assertRaises(RC.ReceiptError):
                log.append(RC.done_fields(intent["seq"], snap), self.rig.clock.stamp())
        self.assertBlocked(1, "receipts_blocked")
        os.unlink(self.rig.rpath(RC.BLOCKED))
        self.assertBlocked(4, "receipts_corrupt")

    def test_D3_D4_a_done_that_landed_or_did_not_is_clean_or_reconcilable(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        log = self.rig.log
        with mock.patch.object(RC._IO, "fsync", side_effect=OSError(5, "I/O error")):
            with self.assertRaises(RC.ReceiptError):
                log.append(RC.done_fields(intent["seq"], self.rig.store.snapshot()), self.rig.clock.stamp())
        os.unlink(self.rig.rpath(RC.BLOCKED))
        self.assertEqual(self.start().row, 6)                           # the bytes landed: a valid done
        data = read_log_bytes(self.rig)
        with open(self.rig.rpath(), "wb") as f:
            f.write(data[:data.rindex(b"\n", 0, len(data) - 1) + 1])    # ... or it did not
        self.assertBlocked(7, "reconcile_required")

    def test_the_marker_failing_to_write_still_blocks_from_derived_state(self):
        plan, auth, intent = self.open_intent_before()
        self.rig.install(plan, auth)
        log, real_open = self.rig.log, os.open

        def no_marker(path, flags, *a):
            if str(path).endswith(RC.BLOCKED):
                raise OSError(5, "I/O error")
            return real_open(path, flags, *a)
        with mock.patch.object(RC._IO, "write", side_effect=OSError(5, "x")), \
                mock.patch.object(RC._IO, "open", no_marker):
            with self.assertRaises(RC.ReceiptError):
                log.append(RC.done_fields(intent["seq"], self.rig.store.snapshot()), self.rig.clock.stamp())
        self.assertFalse(os.path.exists(self.rig.rpath(RC.BLOCKED)))
        a = self.start()
        self.assertFalse(a.proceed)
        self.assertEqual((a.row, a.code), (7, "reconcile_required"))

    def test_the_intent_row_is_what_the_store_logs_for_both_operations(self):
        for supersede in (False, True):
            with self.subTest(supersede=supersede):
                self.rig = Rig(self)
                self.prepare(supersede)
                intent, done, auth = self.rig.commit(self.rig.plan(1))
                snap = self.rig.store.snapshot()
                self.assertIn((intent["nonce_hash"], intent["memory_id"], auth.confirmed_at), snap.nonces)
                self.assertEqual(done["state_digest"], intent["state_digest_after"])
                self.assertEqual(intent["operation"], "apply" if supersede else "put")
                self.assertEqual(self.start().row, 6)

    def test_two_commits_chain_and_a_forged_intermediate_is_a_gap(self):
        first = self.committed()
        second = self.committed(2, attribute="employer", value="Acme")
        recs = self.rig.log.records()
        self.assertEqual([r["action"] for r in recs], ["open", "intent", "done", "intent", "done"])
        self.assertEqual(recs[3]["state_digest_before"], recs[2]["state_digest"])
        self.assertEqual(self.start().row, 6)
        lines = read_log_bytes(self.rig).split(b"\n")
        with open(self.rig.rpath(), "wb") as f:
            f.write(b"\n".join([lines[0], lines[3], lines[4], b""]))
        self.assertEqual(self.start().row, 4)                           # a spliced log is a broken chain


# ---- pinned behavior (each assertion exists because a mutation survived without it) -------------------------------

class PinnedTests(Case):
    def test_the_decided_names_sizes_and_file_type_masks(self):
        self.assertEqual((RC.RECEIPTS, RC.LOCK, RC.BLOCKED), ("commits.jsonl", "commits.lock", "commits.blocked"))
        self.assertEqual(RC.MAX_BYTES, 8 * 1024 * 1024)
        self.assertEqual((RC._IFMT, RC._IFREG, RC._IFDIR), (stat.S_IFMT(0xFFFF), stat.S_IFREG, stat.S_IFDIR))
        self.assertEqual(RC.FORMAT, "memory-commit-receipts/1")

    def test_the_error_text_is_the_reason_then_the_codes(self):
        self.assertEqual(str(RC.ReceiptError("x")), "x")
        self.assertEqual(str(RC.ReceiptError("x", ("a", "b"))), "x: a, b")
        e = RC.ReceiptError("x", ["a"])
        self.assertEqual((e.reason, e.details), ("x", ("a",)))

    def test_builders_defaults_and_refusal_details(self):
        self.rig.baseline()
        snap = self.rig.store.snapshot()
        self.assertIs(RC.done_fields(0, snap)["recovered"], False)
        self.assertIs(RC.done_fields(0, snap, recovered=True)["recovered"], True)
        for fn in (lambda x: RC.open_fields(x), lambda x: RC.intent_fields(x, None, nonce="n", confirmed_at="t",
                                                                              apply_now="t")):
            with self.assertRaises(RC.ReceiptError) as cm:
                fn("nope")
            self.assertEqual((cm.exception.reason, cm.exception.details), ("snapshot_not_ok", ("not_a_snapshot",)))
            with self.assertRaises(RC.ReceiptError) as cm:
                fn(S.Snapshot(False, "lock_busy", None, None, None, None, None, None))
            self.assertEqual(cm.exception.details, ("lock_busy",))
        self.assertEqual(RC.open_fields(snap)["action"], "open")

    def test_the_envelope_and_open_fields_are_each_checked(self):
        recs = Rig(self)
        recs.baseline()
        rec = recs.log.records()[0]
        for field, bad in (("seq", True), ("seq", "0"), ("prev", "x"), ("at", "now"), ("hash", "x"),
                           ("baseline_state_digest", "x"), ("baseline_snapshot_digest", "x"),
                           ("baseline_state_digest", None), ("baseline_snapshot_digest", 5)):
            with self.subTest(field=field, value=repr(bad)):
                self.assertTrue(RC.field_problem(dict(rec, **{field: bad})).startswith("FIELDS:open:"))
        self.assertEqual(RC.field_problem(dict(rec, seq=True)), "FIELDS:open:envelope")
        self.assertEqual(RC.field_problem(dict(rec, baseline_state_digest="x")), "FIELDS:open:values")
        self.assertIsNone(RC.field_problem(rec))

    def test_anomaly_code_count_and_every_code_are_checked(self):
        base = R.new_record([], "2026-02-01T00:00:00Z", RC.anomaly_fields(None, ["X"], None))
        for codes, ok in ((["X"], True), (["X"] * 16, True), (["X"] * 17, False), ([], False), (["X", "a b"], False),
                          (["X", 5], False)):
            with self.subTest(codes=len(codes), ok=ok):
                got = RC.field_problem(dict(base, codes=codes))
                self.assertEqual(got is None, ok)

    def test_validate_records_details(self):
        rig = Rig(self)
        rig.baseline()
        rig.commit(rig.plan(1))
        recs = rig.log.records()
        o, i, d = [envelope_free(r) for r in recs]
        self.assertEqual(RC.validate_records(recs[:1] + ["x"]), (["FIELDS:not_a_record"], None))
        bad_done = dict(d, state_digest="x")
        write_log(rig, [o, i, bad_done])
        problems, state = RC.validate_records(RC._read_records(rig.rpath()))
        self.assertEqual((problems, state), (["FIELDS:done:values"], None))
        write_log(rig, [o])
        problems, state = RC.validate_records(RC._read_records(rig.rpath()))
        self.assertEqual(problems, [])
        self.assertEqual((state.open_anomalous, state.blocked, state.open_intent, state.count), (False, False, None, 1))
        self.assertEqual(state.last_hash, RC._read_records(rig.rpath())[-1]["hash"])
        write_log(rig, [o, i, RC.anomaly_fields(1, ["X"], None), d])
        problems, state = RC.validate_records(RC._read_records(rig.rpath()))
        self.assertEqual((problems, state.open_anomalous, state.open_intent), ([], False, None))
        records = RC._read_records(rig.rpath())
        self.assertEqual((state.last_hash, state.count), (records[-1]["hash"], 4))
        write_log(rig, [o, i, d, RC.done_fields(99, rig.store.snapshot())])
        self.assertEqual(RC.validate_records(RC._read_records(rig.rpath()))[0], ["CLOSE_WITHOUT_INTENT"])
        write_log(rig, [o, i, d, RC.done_fields(1, rig.store.snapshot())])
        self.assertEqual(RC.validate_records(RC._read_records(rig.rpath()))[0], ["DOUBLE_CLOSE"])
        # the same memory id again after a clean abort is a permitted retry (amendment); see test_memory_retry.py
        aborted = RC.aborted_fields(1, "crashed_before_write", i["state_digest_before"])
        again = dict(i, nonce_hash="c" * 32)
        write_log(rig, [o, i, aborted, again])
        self.assertEqual(RC.validate_records(RC._read_records(rig.rpath()))[0], [])

    def test_prediction_details(self):
        self.rig.seed(2)
        snap = self.rig.store.snapshot()
        before = S.normalized_entries(snap.files)
        with self.assertRaises(RC.ReceiptError) as cm:
            RC.predict_after(before, snap.items, item(2), ev("Kansas City"), [], "2026-02-01T00:00:00Z")
        self.assertEqual(cm.exception.details, ("memory_id_exists",))
        for items, links in ((snap.items, [mid(9)]), ((), [mid(2)])):
            with self.assertRaises(RC.ReceiptError) as cm:
                RC.predict_after(before, items, item(1), ev(), links, "2026-02-01T00:00:00Z")
            self.assertEqual(cm.exception.details, ("link_not_stored",))

    def test_read_errors_name_what_is_wrong(self):
        rig = Rig(self)
        log = rig.log
        with self.assertRaises(RC.ReceiptError) as cm:
            log.records()
        self.assertEqual((cm.exception.reason, cm.exception.details), ("receipts_missing", ()))
        rig.baseline()
        good = read_log_bytes(rig)
        for data, details in ((b"\xff\n", ("utf8",)), (good[:-1], ("unterminated",)), (b"nope\n", ("json:0",)),
                              (good + b"{ }\n", ("not_canonical:1",))):
            with open(rig.rpath(), "wb") as f:
                f.write(data)
            with self.assertRaises(RC.ReceiptError) as cm:
                log.records()
            self.assertEqual((cm.exception.reason, cm.exception.details), ("receipts_corrupt", details))

    def test_the_size_cap_and_the_bounded_read_loop(self):
        rig = Rig(self)
        rig.baseline()
        data = read_log_bytes(rig)
        with mock.patch.object(RC, "MAX_BYTES", len(data)):
            self.assertEqual(len(RC._read_records(rig.rpath())), 1)                 # exactly at the cap: fine
            with mock.patch.object(RC, "_CHUNK", 1):                                # one byte per read: len + EOF
                self.assertEqual(len(RC._read_records(rig.rpath())), 1)
        with mock.patch.object(RC, "MAX_BYTES", len(data) - 1):
            with self.assertRaises(RC.ReceiptError) as cm:
                RC._read_records(rig.rpath())
            self.assertEqual(cm.exception.reason, "receipts_unavailable")
        real = os.fstat

        class Small:
            def __init__(self, st):
                self.st_mode, self.st_uid, self.st_nlink, self.st_size = st.st_mode, st.st_uid, st.st_nlink, 1
        with mock.patch.object(RC, "MAX_BYTES", len(data) - 1), \
                mock.patch.object(RC._IO, "fstat", lambda fd: Small(real(fd))):
            with self.assertRaises(RC.ReceiptError) as cm:                           # grows past the cap while read
                RC._read_records(rig.rpath())
            self.assertEqual(cm.exception.reason, "receipts_unavailable")
        calls = []
        with mock.patch.object(RC._IO, "read", lambda fd, n: (calls.append(n), b"x")[1]):
            with self.assertRaises(RC.ReceiptError) as cm:
                RC._read_records(rig.rpath())
        self.assertEqual(cm.exception.reason, "receipts_unavailable")
        self.assertEqual(len(calls), RC.MAX_BYTES // RC._CHUNK + 2)                 # never an endless loop

    def test_append_details_and_second_marker_check_and_prev_check(self):
        rig = Rig(self)
        first = rig.baseline()
        plan = rig.plan(1)
        auth = rig.authorization(plan)
        intent = rig.begin(plan, auth)
        fields = RC.aborted_fields(intent["seq"], "crashed_before_write", rig.store.state_digest())
        at = rig.clock.stamp()
        with self.assertRaises(RC.ReceiptError) as cm:
            rig.log.append(fields, at, expect_prev=first["hash"])                    # not the tail any more
        self.assertEqual(cm.exception.reason, "receipts_changed")
        with self.assertRaises(RC.ReceiptError) as cm:
            rig.log.append(fields, "not a time")
        self.assertEqual((cm.exception.reason, cm.exception.details), ("bad_fields", ("record",)))
        real = RC._marker_present
        with mock.patch.object(RC, "_marker_present", side_effect=[False, True]):     # appears after the lock is held
            with self.assertRaises(RC.ReceiptError) as cm:
                rig.log.append(fields, at)
            self.assertEqual(cm.exception.reason, "receipts_blocked")
        self.assertEqual(real(rig.rdir), False)
        self.assertEqual(rig.log.append(fields, at, expect_prev=intent["hash"])["action"], "aborted")

    def test_a_sidecar_that_is_a_link_or_has_the_wrong_mode_is_unavailable(self):
        rig = Rig(self)
        rig.baseline()
        plan = rig.plan(1)
        fields = RC.intent_fields(rig.store.snapshot(), plan, nonce="n", confirmed_at=rig.clock.stamp(),
                                  apply_now=rig.clock.stamp())
        os.rename(rig.rpath(RC.LOCK), rig.rpath(RC.LOCK) + ".real")
        os.symlink(rig.rpath(RC.LOCK) + ".real", rig.rpath(RC.LOCK))
        for fn in (lambda: rig.log.append_intent(fields, rig.clock.stamp(), rig.store),):
            with self.assertRaises(RC.ReceiptError) as cm:
                fn()
            self.assertEqual(cm.exception.reason, "receipts_unavailable")
        a = RC.assess(rig.rdir, rig.store)
        self.assertEqual((a.row, a.code), (2, "receipts_unavailable"))
        os.unlink(rig.rpath(RC.LOCK))
        os.rename(rig.rpath(RC.LOCK) + ".real", rig.rpath(RC.LOCK))
        os.chmod(rig.rpath(RC.LOCK), 0o644)
        a = RC.assess(rig.rdir, rig.store)
        self.assertEqual((a.row, a.code), (2, "receipts_unavailable"))

    def test_a_receipts_path_that_cannot_be_inspected_is_unavailable(self):
        rig = Rig(self)
        rig.baseline()
        real = os.lstat

        def eio(path, *a, **k):
            if str(path).endswith(RC.RECEIPTS):
                raise OSError(5, "I/O error")
            return real(path, *a, **k)
        with mock.patch.object(RC._IO, "lstat", eio):
            a = RC.assess(rig.rdir, rig.store)
        self.assertEqual((a.proceed, a.row, a.code), (False, 2, "receipts_unavailable"))

    def test_writes_are_validated_for_progress(self):
        rig = Rig(self)
        rig.baseline()
        plan = rig.plan(1)
        fields = RC.intent_fields(rig.store.snapshot(), plan, nonce="n", confirmed_at=rig.clock.stamp(),
                                  apply_now=rig.clock.stamp())
        at = rig.clock.stamp()
        real = os.write
        with mock.patch.object(RC._IO, "write", lambda fd, data: real(fd, bytes(data[:1]))):   # one byte per call
            rec = rig.log.append_intent(fields, at, rig.store)
        self.assertEqual(rig.log.records()[-1], rec)
        plan2 = rig.plan(2, attribute="employer", value="Acme")
        for label, result in (("zero", lambda d: 0), ("negative", lambda d: -1), ("too many", lambda d: len(d) + 5)):
            log, calls = rig.log, []
            f2 = RC.aborted_fields(rec["seq"], "crashed_before_write", rig.store.state_digest())
            with self.subTest(label), mock.patch.object(RC._IO, "write",
                                                        lambda fd, d: (calls.append(1), result(d))[1]):
                with self.assertRaises(RC.ReceiptError) as cm:
                    log.append(f2, at)
                self.assertEqual(cm.exception.reason, "receipts_append_ambiguous")
                self.assertTrue(log.poisoned)
                self.assertEqual(len(calls), 2)                     # the refused write, then the evidence marker: no retries
            if os.path.exists(rig.rpath(RC.BLOCKED)):
                os.unlink(rig.rpath(RC.BLOCKED))
        self.assertTrue(plan2)

    def test_the_value_types_are_frozen(self):
        a = RC.assess(self.rig.rdir, self.rig.store)
        s = RC.startup_check(self.rig.rdir, self.rig.store)
        report = RC.audit(self.rig.rdir, self.rig.store)
        self.rig.baseline()
        self.rig.commit(self.rig.plan(1))
        state = RC.validate_records(self.rig.log.records())[1]
        for obj in (a, s, report, state, RC.Proof("COMMITTED", ())):
            with self.subTest(type=type(obj).__name__), self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, "row" if hasattr(obj, "row") else "kind" if hasattr(obj, "kind") else "count", 0)

    def test_reports_say_what_they_observed_in_plain_words(self):
        a = RC.assess(self.rig.rdir, self.rig.store)
        self.assertEqual(a.facts, ("no receipts file", "the store is empty"))
        self.committed()
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual(a.facts, ("the log's last closing digest equals the live store",))
        os.unlink(self.rig.path(mid(1) + ".md"))
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual(a.details, ("live digest differs from the log's last closing digest",))

    def test_open_intent_facts_and_inferences(self):
        plan, auth, intent = self.open_intent_before()
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual(a.facts, (f"open intent seq {intent['seq']} for {mid(1)}",
                                   "live digest equals the intent's before digest"))
        self.assertEqual(a.inferences, ("consistent with a crash before the store wrote anything",))
        self.rig.install(plan, auth)
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual(a.facts[1], "live digest equals the intent's after digest")
        self.assertEqual(a.inferences, ("consistent with a crash after the store commit and before the done record",))
        os.unlink(self.rig.path(".authorizations"))
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual(a.inferences, ("the store matches neither side of the intent",))

    def test_a_state_that_is_neither_side_of_an_open_intent_says_so(self):
        plan, auth, intent = self.open_intent_before()
        os.makedirs(self.rig.memdir)
        open(self.rig.path(".lock"), "w").close()
        with open(self.rig.path(mid(7) + ".md"), "w") as f:
            f.write(S.serialize(item(7, attribute="hobby", value="chess"), ev("chess")))
        a = RC.assess(self.rig.rdir, self.rig.fresh_store())
        self.assertEqual((a.row, a.code), (8, "state_gap"))
        self.assertEqual(a.details, ("an open intent and a store that is neither its before nor its after",))


# ---- structure -----------------------------------------------------------------------------------------------------

class StructureTests(unittest.TestCase):
    SRC = inspect.getsource(RC)

    def test_the_read_only_functions_never_reach_a_writing_name(self):
        tree = ast.parse(self.SRC)
        fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        banned = {"_write", "_put", "_lock", "_mark_blocked", "O_WRONLY", "O_RDWR", "O_CREAT",
                  "O_TRUNC", "O_APPEND", "write", "fsync", "unlink", "rename", "replace", "mkdir", "makedirs",
                  "truncate", "chmod", "LOCK_EX"}
        for name in ("assess", "startup_check", "audit", "classify_open_intent", "_read_records", "_marker_present",
                     "_folder_ok", "validate_records", "field_problem", "intent_problem", "predict_after",
                     "intent_fields"):
            attrs = {n.attr for n in ast.walk(fns[name]) if isinstance(n, ast.Attribute)}
            names = {n.id for n in ast.walk(fns[name]) if isinstance(n, ast.Name)}
            self.assertFalse(attrs & banned, (name, attrs & banned))
            self.assertFalse(names & {"_write", "_put", "_lock", "_mark_blocked", "ReceiptLog"}, (name, names))

    def test_only_the_log_class_writes(self):
        tree = ast.parse(self.SRC)
        writers = set()
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
                if names & {"write", "fsync"}:
                    writers.add((cls.name, fn.name))
        self.assertEqual(writers, {("ReceiptLog", "_put"), ("ReceiptLog", "_mark_blocked")})

    def test_it_never_names_what_other_layers_own_and_reads_no_clock(self):
        for banned in ("memory_commit", "memory_gate", "memory_extract", "memory_retrieval", "memory_keys",
                       "fake_clock", "store_arrange", "receipts_harness"):
            self.assertNotIn(banned, self.SRC)
        tree = ast.parse(self.SRC)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"Authorization", "Confirmation", "now", "utcnow", "today", "time", "monotonic",
                                 "environ", "getenv", "system"}, used)

    def test_it_imports_only_what_it_should(self):
        mods = set()
        for n in ast.walk(ast.parse(self.SRC)):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"fcntl", "hashlib", "json", "os", "re", "dataclasses.dataclass", "dataclasses.replace",
                                "datetime.timedelta", "backtalk.memory", "backtalk.memory_review",
                                "backtalk.memory_store"})

    def test_the_import_graph_among_the_memory_modules_has_no_cycle_and_nothing_imports_this_one(self):
        names = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, "memory*.py"))}
        graph = {}
        for name in sorted(names):
            with open(os.path.join(PKG, name + ".py"), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            edges = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    mod = n.module or ""
                    if mod == "backtalk":
                        edges |= {a.name for a in n.names} & names
                    elif mod.startswith("backtalk."):
                        edges |= {mod.split(".")[1]} & names
                elif isinstance(n, ast.Import):
                    edges |= {a.name.split(".")[1] for a in n.names if a.name.startswith("backtalk.")} & names
            graph[name] = edges - {name}
        self.assertEqual(graph["memory_receipts"], {"memory", "memory_review", "memory_store"})
        for name, edges in graph.items():
            if name != "memory_execute":                       # the 8.2d layer is the one importer
                self.assertNotIn("memory_receipts", edges, name)
        color = {}

        def visit(node, stack):
            color[node] = 1
            for nxt in sorted(graph[node]):
                if color.get(nxt) == 1:
                    self.fail("cycle: " + " -> ".join(stack + [node, nxt]))
                if nxt not in color:
                    visit(nxt, stack + [node])
            color[node] = 2
        for node in sorted(graph):
            if node not in color:
                visit(node, [])

    def test_nothing_outside_the_tests_imports_the_receipt_module_or_its_harness(self):
        root = os.path.dirname(PKG)
        for path in glob.glob(os.path.join(root, "backtalk", "**", "*.py"), recursive=True):
            if os.path.basename(path) in ("memory_receipts.py", "memory_execute.py", "commit_cli.py"):    # the 8.2d layer
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("memory_receipts", text, path)
            self.assertNotIn("receipts_harness", text, path)

    def test_the_limitation_statement_is_in_the_docstring(self):
        doc = " ".join(RC.__doc__.split())
        for phrase in ("The hashes are unkeyed", "A consistent rollback of store and log together is a valid earlier state",
                       "undetectable and loses audit history only", "that it never appends", "never from a digest alone"):
            self.assertIn(phrase, doc)

    def test_the_default_location_is_not_in_the_module(self):
        self.assertNotIn(".backtalk-memory-review", self.SRC)
        self.assertNotIn("expanduser", self.SRC)


if __name__ == "__main__":
    unittest.main()
