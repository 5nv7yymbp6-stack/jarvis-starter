"""Memory 2.0, Phase 8.2c-2: the receipt log (backtalk/memory_receipts.py): record schemas, the state machine, the file and
its append rules, failure and poison behavior, and `predict_after`. Every test runs on a throwaway vault and a throwaway
receipts folder in a temp directory with a deterministic clock; nothing reads or writes the real vault.

WHAT THESE TESTS DO NOT SHOW. The hashes are unkeyed: they detect accident and naive edits, not code running as the same
user, which can compute a consistent forged log. A receipt is a claim; the proof of a commit is the store's own files."""
import copy
import fcntl
import os
import unittest
from dataclasses import replace
from unittest import mock

from backtalk import memory_gate as G
from backtalk import memory_receipts as RC
from backtalk import memory_review as R
from backtalk import memory_store as S
from tests.receipts_harness import (Crash, Rig, envelope_free, ev, item, mid, day, read_log_bytes, tree_state,
                                    write_log)

H64 = "d" * 64


class Case(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(self)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))

    def refused(self, reason, fn, *a, **k):
        with self.assertRaises(RC.ReceiptError) as cm:
            fn(*a, **k)
        self.assertEqual(cm.exception.reason, reason, str(cm.exception))
        return cm.exception

    def three(self):
        """open, intent, done from one real authorized commit; returns the three records."""
        if getattr(self, "_three", None) is None:
            self.rig.baseline()
            self.rig.commit(self.rig.plan(1))
            self._three = self.rig.log.records()
        return self._three


# ---- schemas ---------------------------------------------------------------------------------------------------------

class SchemaTests(Case):
    def test_every_record_of_a_real_commit_has_exactly_its_fields(self):
        recs = self.three()
        self.assertEqual([r["action"] for r in recs], ["open", "intent", "done"])
        for r in recs:
            self.assertIsNone(RC.field_problem(r))
            self.assertEqual(set(r), RC.ENVELOPE | RC.SPECS[r["action"]])
        self.assertEqual(recs[0]["format"], RC.FORMAT)

    def test_aborted_and_anomaly_records_validate(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        intent = self.rig.begin(plan, self.rig.authorization(plan))
        live = self.rig.store.state_digest()
        recs = []
        for fields in (RC.anomaly_fields(intent["seq"], ["STATE_GAP"], live), RC.anomaly_fields(None, ["X"], None),
                       RC.aborted_fields(intent["seq"], "crashed_before_write", live)):
            rec = R.new_record(self.rig.log.records(), self.rig.clock.stamp(), fields)
            self.assertIsNone(RC.field_problem(rec))
            recs.append(rec)

    def test_missing_extra_and_unknown_fields_are_refused(self):
        for rec in self.three():
            action = rec["action"]
            for key in sorted(RC.SPECS[action] | {"action", "seq", "prev", "at", "hash"}):
                broken = {k: v for k, v in rec.items() if k != key}
                with self.subTest(action=action, missing=key):
                    self.assertIsNotNone(RC.field_problem(broken))
            with self.subTest(action=action, extra=True):
                self.assertEqual(RC.field_problem(dict(rec, surprise=1)), f"FIELDS:{action}:keys")
        self.assertEqual(RC.field_problem(dict(self.three()[0], action="reconcile")), "FORMAT_UNKNOWN:action")
        self.assertEqual(RC.field_problem(dict(self.three()[0], format="memory-commit-receipts/2")),
                         "FORMAT_UNKNOWN:format")
        self.assertEqual(RC.field_problem("not a record"), "FIELDS:not_a_record")

    def test_each_intent_value_is_checked_for_type_and_format(self):
        intent = self.three()[1]
        good_pairs = intent["manifest_after"]
        bad = {
            "candidate": ["", "x" * 81, "has space", 5, None],
            "lock_hash": ["A" * 64, "a" * 63, 5, None],
            "context_digest": ["g" * 64, "", None],
            "operation": ["delete", "PUT", None],
            "memory_id": ["mem_1", "MEM_000000000001", 5],
            "links": ["mem_000000000002", [5], ["mem_000000000002", "mem_000000000002"], [intent["memory_id"]]],
            "item_digest": ["x", None],
            "nonce_hash": ["a" * 31, "A" * 32, None],
            "confirmed_at": ["2026-13-45T00:00:00Z", "2026-01-10 00:00:00Z", 5, None],
            "apply_now": ["", "now", None],
            "state_digest_before": ["z" * 64, None],
            "snapshot_digest_before": ["z" * 64, None],
            "state_digest_after": ["z" * 64, None],
            "manifest_before": [{}, "x", [["notes.md", "a" * 64]], [["mem_000000000001.md", "A" * 64]],
                                [[3, 4]]],
            "manifest_after": [{}, [["mem_000000000002.md"]], [[1, 2]], [list(reversed(p)) for p in good_pairs],
                               good_pairs + good_pairs],
            "nonces_before": ["x", [["a" * 32]], [["a" * 32, "mem_000000000001", "bad"]], [5]],
        }
        for field, values in bad.items():
            for value in values:
                with self.subTest(field=field, value=repr(value)[:40]):
                    self.assertEqual(RC.field_problem(dict(intent, **{field: value})), "FIELDS:intent:values")

    def test_done_aborted_and_anomaly_values(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        intent = self.rig.begin(plan, self.rig.authorization(plan))
        cases = {
            "done": (RC.done_fields(0, self.rig.store.snapshot()),
                     {"intent_seq": [-1, "1", True, None], "state_digest": ["x", None],
                      "snapshot_digest": ["x", None], "recovered": [0, "yes", None]}),
            "aborted": (RC.aborted_fields(0, "crashed_before_write", H64),
                        {"intent_seq": [-1, None], "code": ["", "a b", None], "state_digest": ["x", None]}),
            "anomaly": (RC.anomaly_fields(0, ["X"], H64),
                        {"intent_seq": [-1, "0"], "codes": [[], "X", [5], ["a b"], ["X"] * 17],
                         "state_digest": ["x", 5]}),
        }
        for action, (fields, bad) in cases.items():
            rec = R.new_record([], "2026-02-01T00:00:00Z", fields)
            self.assertIsNone(RC.field_problem(rec), action)
            for field, values in bad.items():
                for value in values:
                    with self.subTest(action=action, field=field, value=repr(value)):
                        self.assertEqual(RC.field_problem(dict(rec, **{field: value})), f"FIELDS:{action}:values")
        self.assertEqual(intent["action"], "intent")

    def test_an_intent_must_agree_with_itself(self):
        base = self.three()[1]
        before, after = base["manifest_before"], base["manifest_after"]
        cases = {
            "digest_before": dict(base, state_digest_before=H64),
            "digest_after": dict(base, state_digest_after=H64),
            "snapshot_before": dict(base, snapshot_digest_before=H64),
            "put_with_links": dict(base, links=[mid(2)]),
            "new_file": dict(base, manifest_before=after, state_digest_before=base["state_digest_after"],
                             snapshot_digest_before=S.snapshot_digest(after, base["nonces_before"])),
            "nonce_already_logged": dict(base, nonces_before=[[base["nonce_hash"], mid(9), base["confirmed_at"]]],
                                         snapshot_digest_before=S.snapshot_digest(before, [
                                             [base["nonce_hash"], mid(9), base["confirmed_at"]]])),
        }
        for code, rec in cases.items():
            with self.subTest(code):
                self.assertEqual(RC.intent_problem(rec), code)
        extra = S.normalized_entries(list(after) + [[mid(7) + ".md", "e" * 64]])
        wrong_set = dict(base, manifest_after=extra, state_digest_after=S.manifest_digest(extra))
        self.assertEqual(RC.intent_problem(wrong_set), "changed_set")
        self.assertIsNone(RC.intent_problem(base))
        gone = dict(base, manifest_after=[], state_digest_after=S.manifest_digest([]))
        self.assertEqual(RC.intent_problem(gone), "new_file")
        stray_link = dict(base, operation="apply", links=[mid(5)])
        self.assertEqual(RC.intent_problem(stray_link), "link_not_stored")


# ---- the state machine ---------------------------------------------------------------------------------------------

class StateMachineTests(Case):
    def fields(self):
        """open, intent, done as bare field dicts from one real commit (a base to rearrange)."""
        return [envelope_free(r) for r in self.three()]

    def valid(self, fields_list):
        write_log(self.rig, fields_list)
        return RC.validate_records(RC._read_records(self.rig.rpath()))

    def test_valid_sequences(self):
        o, i, d = self.fields()
        for label, seq in (("open", [o]), ("committed", [o, i, d]),
                           ("aborted", [o, i, RC.aborted_fields(1, "crashed_before_write", i["state_digest_before"])])):
            with self.subTest(label):
                problems, state = self.valid(seq)
                self.assertEqual(problems, [])
                self.assertEqual(state.count, len(seq))
        problems, state = self.valid([o, i])
        self.assertEqual(state.open_intent["memory_id"], i["memory_id"])
        self.assertFalse(state.open_anomalous)
        problems, state = self.valid([o, i, RC.anomaly_fields(1, ["STATE_GAP"], None)])
        self.assertEqual(problems, [])
        self.assertTrue(state.open_anomalous)
        problems, state = self.valid([o, i, RC.anomaly_fields(1, ["STATE_GAP"], None), d])      # a closing may follow
        self.assertEqual(problems, [])
        self.assertIsNone(state.open_intent)

    def test_closing_digests_chain_into_the_next_intent(self):
        o, i, d = self.fields()
        problems, state = self.valid([o, i, d])
        self.assertEqual(state.closing_digest, i["state_digest_after"])
        second = dict(i, memory_id=mid(2), nonce_hash="b" * 32, state_digest_before=i["state_digest_after"],
                      manifest_before=i["manifest_after"])
        after = S.normalized_entries(list(i["manifest_after"]) + [[mid(2) + ".md", "b" * 64]])
        second.update(manifest_after=after, state_digest_after=S.manifest_digest(after),
                      snapshot_digest_before=S.snapshot_digest(i["manifest_after"], i["nonces_before"]))
        self.assertEqual(self.valid([o, i, d, second])[0], [])

    def test_each_invalid_transition_has_its_own_code(self):
        o, i, d = self.fields()
        cases = {
            "CHAIN:first_not_open": [i, d],
            "SECOND_OPEN": [o, o],
            "INTENT_OPEN": [o, i, dict(i, memory_id=mid(9), nonce_hash="c" * 32)],
            "STATE_GAP": [o, dict(i, state_digest_before="e" * 64)],
            "CLOSE_WITHOUT_INTENT": [o, d],
            "DOUBLE_CLOSE": [o, i, d, d],
            "CLOSE_DIGEST_MISMATCH": [o, i, dict(d, state_digest="f" * 64)],
            "AFTER_ANOMALY": [o, RC.anomaly_fields(None, ["GAP"], None), i],
        }
        for code, seq in cases.items():
            with self.subTest(code):
                problems, state = self.valid(seq)
                self.assertEqual(problems, [code])
                self.assertIsNone(state)
        self.assertEqual(self.valid([o, i, RC.aborted_fields(1, "x", "f" * 64)])[0], ["CLOSE_DIGEST_MISMATCH"])
        self.assertEqual(self.valid([o, i, RC.anomaly_fields(7, ["X"], None)])[0], ["CLOSE_WITHOUT_INTENT"])
        self.assertEqual(RC.validate_records([]), (["EMPTY"], None))
        self.assertEqual(RC.validate_records(["x"]), (["FIELDS:not_a_record"], None))

    def test_duplicate_nonce_and_duplicate_memory_id_across_intents(self):
        o, i, d = self.fields()
        mk = lambda **kw: dict(i, state_digest_before=i["state_digest_after"], manifest_before=i["manifest_after"],
                               snapshot_digest_before=S.snapshot_digest(i["manifest_after"], i["nonces_before"]), **kw)

        def with_new_file(rec, mem):
            after = S.normalized_entries(list(rec["manifest_before"]) + [[mem + ".md", "b" * 64]])
            return dict(rec, memory_id=mem, manifest_after=after, state_digest_after=S.manifest_digest(after))
        same_nonce = with_new_file(mk(nonce_hash=i["nonce_hash"]), mid(2))
        same_memory = dict(mk(nonce_hash="c" * 32), memory_id=i["memory_id"])
        self.assertEqual(self.valid([o, i, d, same_nonce])[0], ["DUPLICATE_NONCE"])
        problems = self.valid([o, i, d, same_memory])[0]
        self.assertTrue(problems and problems[0].startswith(("INTENT_INCONSISTENT", "DUPLICATE_INTENT")), problems)

    def test_an_edited_unrehashed_log_is_a_broken_chain(self):
        self.three()
        lines = read_log_bytes(self.rig).decode().split("\n")
        edits = {
            "altered": [lines[0], lines[1].replace('"candidate":"goal', '"candidate":"Goal'), *lines[2:]],
            "reordered": [lines[1], lines[0], *lines[2:]],
            "gap": [lines[0], lines[2], lines[3]],
        }
        for label, edited in edits.items():
            with open(self.rig.rpath(), "w") as f:
                f.write("\n".join(edited))
            with self.subTest(label):
                err = self.refused("receipts_corrupt", self.rig.log.records)
                self.assertTrue(err.details[0].startswith("CHAIN:"), err.details)


# ---- the file and the append rules -------------------------------------------------------------------------------

class FileTests(Case):
    def test_create_makes_a_private_file_a_sidecar_and_an_open_record(self):
        rec = self.rig.baseline()
        self.assertEqual((rec["seq"], rec["prev"]), (0, R.GENESIS))
        for name in (RC.RECEIPTS, RC.LOCK):
            st = os.lstat(self.rig.rpath(name))
            self.assertEqual((st.st_mode & 0o777, st.st_uid, st.st_nlink), (0o600, os.getuid(), 1))
        self.assertEqual(self.rig.log.records(), [rec])
        self.assertEqual(rec["baseline_state_digest"], S.manifest_digest([]))

    def test_create_fsyncs_the_file_and_the_folder(self):
        synced = []
        real = RC._IO.fsync
        with mock.patch.object(RC._IO, "fsync", lambda fd: (synced.append(os.fstat(fd).st_mode & 0o170000), real(fd))[1]):
            self.rig.baseline()
        self.assertEqual(synced, [0o100000, 0o040000])                      # the file, then the folder

    def test_append_fsyncs_the_file_only(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        synced = []
        real = RC._IO.fsync
        with mock.patch.object(RC._IO, "fsync", lambda fd: (synced.append(os.fstat(fd).st_mode & 0o170000), real(fd))[1]):
            self.rig.begin(plan, self.rig.authorization(plan))
        self.assertEqual(synced, [0o100000])

    def test_create_refusals(self):
        self.rig.seed(2)
        snap = self.rig.store.snapshot()
        self.refused("baseline_refused", self.rig.log.create, snap, "2026-02-01T00:00:00Z")      # never baseline a store
        self.assertFalse(os.path.exists(self.rig.rpath()))
        bad = S.Snapshot(False, "lock_busy", None, None, None, None, None, None)
        self.refused("snapshot_not_ok", self.rig.log.create, bad, "2026-02-01T00:00:00Z")
        os.makedirs(self.rig.rdir, exist_ok=True)
        rig2 = Rig(self)
        rig2.baseline()
        self.refused("receipts_exists", rig2.log.create, rig2.store.snapshot(), "2026-02-01T00:00:00Z")
        self.refused("bad_fields", rig2.log.append, {"action": "open"}, "2026-02-01T00:00:00Z")
        self.refused("bad_fields", rig2.log.append, "nope", "2026-02-01T00:00:00Z")
        self.refused("bad_fields", rig2.log.append, RC.aborted_fields(0, "x", H64), "not a time")

    def test_the_folder_is_never_created(self):
        gone = os.path.join(self.rig.root, "missing-folder")
        log = RC.ReceiptLog(gone)
        self.refused("receipts_unavailable", log.create, self.rig.store.snapshot(), "2026-02-01T00:00:00Z")
        self.assertFalse(os.path.exists(gone))

    def test_folder_and_file_modes_owners_and_types_are_enforced(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce="n" * 8, confirmed_at=self.rig.clock.stamp(),
                                  apply_now=self.rig.clock.stamp())
        at = self.rig.clock.stamp()
        before = read_log_bytes(self.rig)

        def attempt(label, reason="receipts_unavailable"):
            with self.subTest(label):
                self.refused(reason, self.rig.log.append_intent, fields, at, self.rig.store)
                self.assertEqual(read_log_bytes(self.rig), before)
        os.chmod(self.rig.rdir, 0o755)
        attempt("folder mode")
        os.chmod(self.rig.rdir, 0o700)
        os.chmod(self.rig.rpath(), 0o644)
        attempt("file mode")
        os.chmod(self.rig.rpath(), 0o600)
        with mock.patch.object(RC.os, "getuid", return_value=os.getuid() + 1):
            attempt("owner")
        os.link(self.rig.rpath(), os.path.join(self.rig.root, "second-name"))
        attempt("hard link")
        os.unlink(os.path.join(self.rig.root, "second-name"))
        os.rename(self.rig.rpath(), self.rig.rpath() + ".real")
        os.symlink(self.rig.rpath() + ".real", self.rig.rpath())
        attempt("symlink")
        os.unlink(self.rig.rpath())
        os.rename(self.rig.rpath() + ".real", self.rig.rpath())
        os.rename(self.rig.rpath(RC.LOCK), self.rig.rpath(RC.LOCK) + ".x")
        attempt("sidecar missing", "receipts_lock_missing")
        os.rename(self.rig.rpath(RC.LOCK) + ".x", self.rig.rpath(RC.LOCK))
        os.chmod(self.rig.rpath(RC.LOCK), 0o644)
        attempt("sidecar mode")
        os.chmod(self.rig.rpath(RC.LOCK), 0o600)
        self.assertEqual(self.rig.log.append_intent(fields, at, self.rig.store)["action"], "intent")

    def test_a_directory_in_place_of_the_file_and_an_oversize_file_are_refused(self):
        self.rig.baseline()
        with mock.patch.object(RC, "MAX_BYTES", 10):
            self.refused("receipts_unavailable", self.rig.log.records)
        os.rename(self.rig.rpath(), self.rig.rpath() + ".real")
        os.mkdir(self.rig.rpath())
        self.refused("receipts_unavailable", self.rig.log.records)

    def test_a_busy_sidecar_is_refused_not_waited_for(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce="n", confirmed_at=self.rig.clock.stamp(),
                                  apply_now=self.rig.clock.stamp())
        fd = os.open(self.rig.rpath(RC.LOCK), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.refused("receipts_busy", self.rig.log.append_intent, fields, self.rig.clock.stamp(), self.rig.store)
        finally:
            os.close(fd)

    def test_expect_prev_and_the_marker(self):
        first = self.rig.baseline()
        plan = self.rig.plan(1)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce="n", confirmed_at=self.rig.clock.stamp(),
                                  apply_now=self.rig.clock.stamp())
        at = self.rig.clock.stamp()
        self.refused("receipts_changed", self.rig.log.append_intent, fields, at, self.rig.store, expect_prev="e" * 64)
        with open(self.rig.rpath(RC.BLOCKED), "w"):
            pass
        self.refused("receipts_blocked", self.rig.log.append_intent, fields, at, self.rig.store, expect_prev=first["hash"])
        os.unlink(self.rig.rpath(RC.BLOCKED))
        self.assertEqual(self.rig.log.append_intent(fields, at, self.rig.store, expect_prev=first["hash"])["seq"], 1)

    def test_a_record_that_breaks_the_state_machine_is_refused_before_any_write(self):
        self.rig.baseline()
        before = tree_state(self.rig.rdir)
        stray_done = RC.done_fields(5, self.rig.store.snapshot())
        self.refused("record_refused", self.rig.log.append, stray_done, self.rig.clock.stamp())
        self.assertEqual(tree_state(self.rig.rdir), before)
        self.assertFalse(self.rig.log.poisoned)


# ---- failures and poison ---------------------------------------------------------------------------------------------

class FailureTests(Case):
    def prepared(self):
        self.rig.baseline()
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        fields = RC.intent_fields(self.rig.store.snapshot(), plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        return fields, self.rig.clock.stamp()

    def assert_poisoned_and_marked(self, log):
        self.assertTrue(log.poisoned)
        self.assertTrue(os.path.exists(self.rig.rpath(RC.BLOCKED)))
        self.refused("receipts_poisoned", log.append, RC.anomaly_fields(None, ["X"], None), self.rig.clock.stamp())

    def test_a_write_that_only_ever_makes_partial_progress_still_completes(self):
        fields, at = self.prepared()
        real = os.write
        with mock.patch.object(RC._IO, "write", lambda fd, data: real(fd, bytes(data[:7]))):
            rec = self.rig.log.append_intent(fields, at, self.rig.store)
        self.assertEqual(self.rig.log.records()[-1], rec)

    def test_a_write_error_after_partial_bytes_poisons_and_leaves_a_torn_tail_never_extended(self):
        fields, at = self.prepared()
        real, calls = os.write, [0]

        def half_then_fail(fd, data):
            calls[0] += 1
            if calls[0] == 1:
                return real(fd, bytes(data[:len(data) // 2]))
            raise OSError(5, "I/O error")
        log = self.rig.log
        with mock.patch.object(RC._IO, "write", half_then_fail):
            self.refused("receipts_append_ambiguous", log.append_intent, fields, at, self.rig.store)
        self.assert_poisoned_and_marked(log)
        torn = read_log_bytes(self.rig)
        self.assertFalse(torn.endswith(b"\n"))
        os.unlink(self.rig.rpath(RC.BLOCKED))                       # a person clears the marker; the log is still corrupt
        self.refused("receipts_corrupt", self.rig.log.records)
        fresh = self.rig.log
        self.refused("receipts_corrupt", fresh.append, RC.anomaly_fields(None, ["X"], None), at)
        self.assertEqual(read_log_bytes(self.rig), torn)               # never truncated, repaired or extended

    def test_fsync_failures_are_ambiguous_and_poison(self):
        fields, at = self.prepared()
        log = self.rig.log
        with mock.patch.object(RC._IO, "fsync", side_effect=OSError(5, "I/O error")):
            self.refused("receipts_append_ambiguous", log.append_intent, fields, at, self.rig.store)
        self.assert_poisoned_and_marked(log)
        self.assertEqual(len(read_log_bytes(self.rig).splitlines()), 2)    # the bytes did land; the log cannot know

    def test_a_folder_fsync_failure_on_create_is_ambiguous(self):
        calls = [0]
        real = os.fsync

        def second_fails(fd):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError(5, "I/O error")
            return real(fd)
        log = self.rig.log
        with mock.patch.object(RC._IO, "fsync", second_fails):
            self.refused("receipts_append_ambiguous", log.create, self.rig.store.snapshot(), self.rig.clock.stamp())
        self.assert_poisoned_and_marked(log)

    def test_a_write_that_makes_no_progress_is_ambiguous(self):
        fields, at = self.prepared()
        log = self.rig.log
        with mock.patch.object(RC._IO, "write", lambda fd, data: 0):
            self.refused("receipts_append_ambiguous", log.append_intent, fields, at, self.rig.store)
        self.assertTrue(log.poisoned)

    def test_keyboard_interrupt_poisons_and_propagates(self):
        fields, at = self.prepared()
        log = self.rig.log
        with mock.patch.object(RC._IO, "write", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                log.append_intent(fields, at, self.rig.store)
        self.assertTrue(log.poisoned)
        self.assertTrue(os.path.exists(self.rig.rpath(RC.BLOCKED)))

    def test_the_marker_failing_changes_nothing_else(self):
        fields, at = self.prepared()
        log = self.rig.log
        real_open = os.open

        def no_marker(path, flags, *a):
            if str(path).endswith(RC.BLOCKED):
                raise OSError(5, "I/O error")
            return real_open(path, flags, *a)
        with mock.patch.object(RC._IO, "write", side_effect=OSError(5, "x")), \
                mock.patch.object(RC._IO, "open", no_marker):
            self.refused("receipts_append_ambiguous", log.append_intent, fields, at, self.rig.store)
        self.assertTrue(log.poisoned)
        self.assertFalse(os.path.exists(self.rig.rpath(RC.BLOCKED)))

    def test_refusals_before_the_write_do_not_poison(self):
        fields, at = self.prepared()
        log = self.rig.log
        self.refused("receipts_changed", log.append_intent, fields, at, self.rig.store, expect_prev="e" * 64)
        self.assertFalse(log.poisoned)
        self.assertEqual(log.append_intent(fields, at, self.rig.store)["seq"], 1)

    def test_the_marker_is_created_exclusively_with_private_mode_and_no_secrets(self):
        fields, at = self.prepared()
        log = self.rig.log
        with mock.patch.object(RC._IO, "fsync", side_effect=OSError(5, "x")):
            self.refused("receipts_append_ambiguous", log.append_intent, fields, at, self.rig.store)
        st = os.lstat(self.rig.rpath(RC.BLOCKED))
        self.assertEqual(st.st_mode & 0o777, 0o600)
        with open(self.rig.rpath(RC.BLOCKED)) as f:
            self.assertEqual(f.read(), '{"format":"memory-commit-receipts/1","reason":"append_ambiguous"}\n')


# ---- prediction ---------------------------------------------------------------------------------------------------

class PredictionTests(Case):
    def after_files(self):
        return S.normalized_entries(
            [[n, __import__("hashlib").sha256(b).hexdigest()] for n, b in self.rig.memory_files().items()
             if n.startswith("mem_")])

    def predicted_equals_real(self, plan, links=None):
        snap = self.rig.store.snapshot()
        links = list(plan.decision.supersedes) if links is None else links
        now = self.rig.clock.stamp()
        predicted = RC.predict_after(S.normalized_entries(snap.files), snap.items, plan.item, plan.evidence, links, now)
        self.rig.last_apply_now = now
        self.rig.install(plan, self.rig.authorization(plan))
        self.assertEqual(predicted, self.after_files())
        return predicted

    def test_a_plain_put(self):
        self.predicted_equals_real(self.rig.plan(1))

    def test_a_supersession_with_one_link(self):
        self.rig.seed(2)
        plan = self.rig.plan(1)
        self.assertEqual(plan.operation, "apply")
        self.assertEqual(len(plan.decision.supersedes), 1)
        self.predicted_equals_real(plan)

    def test_awkward_text_and_a_later_apply_stamp(self):
        self.rig.seed(2, value="Café “Old”")
        self.rig.clock.advance(3600)
        plan = self.rig.plan(1, value="Café “New” — Zürich", evidence="I live in Café “New” — Zürich now")
        self.predicted_equals_real(plan)

    def test_the_intent_carries_the_prediction_and_the_nonce_hash_the_store_logs(self):
        self.rig.baseline()
        self.rig.seed(2)
        plan = self.rig.plan(1)
        auth = self.rig.authorization(plan)
        snap = self.rig.store.snapshot()
        fields = RC.intent_fields(snap, plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=self.rig.clock.stamp())
        self.assertEqual(fields["links"], list(plan.decision.supersedes))
        self.assertEqual(fields["manifest_before"], S.normalized_entries(snap.files))
        self.rig.last_apply_now = fields["apply_now"]
        self.rig.install(plan, auth)
        after = self.rig.store.snapshot()
        self.assertEqual(S.normalized_entries(after.files), fields["manifest_after"])
        self.assertEqual(after.state_digest, fields["state_digest_after"])
        self.assertEqual(after.nonces, ((fields["nonce_hash"], plan.item.id, auth.confirmed_at),))

    def test_prediction_refuses_what_it_cannot_predict(self):
        self.rig.seed(2)
        snap = self.rig.store.snapshot()
        before = S.normalized_entries(snap.files)
        it = item(1)
        self.refused("prediction_failed", RC.predict_after, before, snap.items, it, ev(), [mid(9)], NOW_)
        self.refused("prediction_failed", RC.predict_after, before, (), it, ev(), [mid(2)], NOW_)
        self.refused("prediction_failed", RC.predict_after, before, snap.items, item(2), ev("Kansas City"), [], NOW_)
        self.refused("prediction_failed", RC.predict_after, before, snap.items, it, "", [], NOW_)
        self.refused("prediction_failed", RC.predict_after, before, snap.items, replace(it, value=""), ev(), [], NOW_)

    def test_the_links_order_given_is_the_order_written(self):
        """The gate reports one link in every arrangement it can reach today, so a two-link install cannot be driven
        through the real store. The prediction still must not re-sort: the new file records links in the order given,
        which is the plan's gate-decision order (`intent_fields` passes it through untouched)."""
        self.rig.seed(2, attribute="employer", value="Initech", valid_from=day(1))
        self.rig.seed(3, attribute="employer", value="Initrode", valid_from=day(2), created=day(2), updated=day(2))
        snap = self.rig.store.snapshot()
        before = S.normalized_entries(snap.files)
        it = item(1, attribute="employer", value="Acme", valid_from=day(10))
        forward = RC.predict_after(before, snap.items, it, ev("Acme"), [mid(2), mid(3)], NOW_)
        backward = RC.predict_after(before, snap.items, it, ev("Acme"), [mid(3), mid(2)], NOW_)
        self.assertNotEqual(dict(forward)[mid(1) + ".md"], dict(backward)[mid(1) + ".md"])
        self.assertEqual({n: h for n, h in forward if n != mid(1) + ".md"},
                         {n: h for n, h in backward if n != mid(1) + ".md"})        # the ended records do not depend on it


NOW_ = "2026-02-01T00:00:00Z"


if __name__ == "__main__":
    unittest.main()
