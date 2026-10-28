"""Memory 2.0: the PRODUCTION plan builder (`commit_cli.make_plan_builder`) reading a real review pack and a real review
ledger from disk, with no injected plan builder and no mocked pipeline.

WHAT IS REAL HERE. A temp vault holds a `VAULT-INDEX.md`; an empty temp transcripts folder stands in for the transcripts. The
pack is written by the production scanner (`scan.run_pipeline` then `scan.write_pack`), the ledger by the production review
session (`review_cli.Session`, whose `push` is the production `append_line`), and the plan by the production builder, which
re-reads the pack bytes, the ledger and the private side file, re-opens the source (`verify_provenance`) and re-derives the
candidate by running `scan.run_pipeline` again (a spy proves it ran; it is wrapped, not replaced).

WHAT IS NOT. Only the clocks are fakes (a deterministic wall clock and a counting monotonic clock; the planner takes them as
arguments). The store is an ordinary temp store; the real-vault constructor path is covered in `test_memory_real_vault_path`.
The vault contents are a tiny fixture, not the user's notes. No test starts a subprocess, uses the network or the Keychain,
or touches the original real-vault path (the guard in `tests/realvault_guard.py` raises if it is reached).

REFUSAL CODES below are the ones the current implementation produces (`memory_commit.plan` and the builder); they were
observed by running it, not assumed. A refusal here is a `CommitError` raised before any receipt, evidence record, approval or
store call exists."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest import mock

from backtalk import memory_commit as C
from backtalk import memory_execute as EX
from backtalk import memory_extract as X
from backtalk import memory_review as R
from backtalk import memory_store as S
from backtalk.evals.memory_extraction import commit_cli as CLI
from backtalk.evals.memory_extraction import review_cli as RV
from backtalk.evals.memory_extraction import scan
from tests import realvault_guard as G
from tests.fake_clock import FakeClock
from tests.receipts_harness import tree_state
from tests.test_memory_commit_plan import NOWS
from tests.test_memory_review import INDEX

_OPENED, _RECORDING = [], []


def _record(event, args):
    if event == "open" and _RECORDING and isinstance(args[0], str):
        _OPENED.append(args[0])


sys.addaudithook(_record)


class DiskCase(unittest.TestCase):
    def setUp(self):
        for target, name in ((subprocess, "run"), (subprocess, "Popen"), (urllib.request, "urlopen")):
            p = mock.patch.object(target, name, side_effect=AssertionError(f"{name} was called"))
            p.start()
            self.addCleanup(p.stop)
        self.touched = G.arm(self)                      # refuses the ORIGINAL real-vault path (audit hook + stat wrappers)
        self.addCleanup(lambda: self.assertEqual(self.touched, []))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = os.path.realpath(tmp.name)
        self.vault = os.path.join(self.top, "v", "vault")
        self.transcripts = os.path.join(self.top, "tr")
        self.root = os.path.join(self.top, "review")
        os.makedirs(self.vault)
        os.mkdir(self.transcripts)
        os.mkdir(self.root, 0o700)
        with open(os.path.join(self.vault, "VAULT-INDEX.md"), "w", encoding="utf-8") as f:
            f.write(INDEX)
        self.clock = FakeClock(NOWS)
        self.generated_at = self.clock.stamp()
        cands, conflicts, meta, _ = scan.run_pipeline(self.transcripts, self.vault, self.generated_at)
        forbidden = [self.vault, os.path.dirname(self.vault), self.transcripts]
        self.run_dir = scan.write_pack(self.root, cands, conflicts, meta, forbidden=forbidden)["dir"]
        self.clock.advance(60)
        self.said = []
        session = RV.Session(self.root, self.run_dir, self.vault, self.transcripts, lambda p: "", self.said.append,
                             self.clock.stamp)
        self.cid = next(c for c, v in session.pack.views.items() if v["attribute"] == "occupation")
        self.other = next(c for c, v in session.pack.views.items() if v["attribute"] == "goal_cash_buyer_list_deep")
        self.clock.advance(60)
        self.assertIsNotNone(session.do(R.lock, self.cid, "lock", problems=session.problems(self.cid)))
        self.store = S.MemoryStore(self.vault, clock=self.clock)
        self.tick = 0.0

    # -- helpers --
    def mono(self):
        self.tick += 0.5
        return self.tick

    def builder(self, candidate=None):
        return CLI.make_plan_builder(self.root, self.run_dir, candidate or self.cid, self.store, self.vault,
                                     self.transcripts, self.clock.stamp, self.mono)

    def pack_path(self):
        return os.path.join(self.run_dir, "review_pack.json")

    def ledger_path(self):
        return os.path.join(self.root, RV.LEDGER)

    def assert_refused(self, code, details=None, candidate=None):
        with self.assertRaises(C.CommitError) as cm:
            self.builder(candidate)()
        self.assertEqual(cm.exception.code, code)
        if details is not None:
            self.assertEqual(tuple(cm.exception.details), tuple(details))
        return cm.exception

    def snapshot(self):
        return (tree_state(self.top), sorted(os.listdir(self.root)))

    def assert_no_write_artifacts(self):
        names = os.listdir(self.root)
        self.assertEqual([n for n in names if n.startswith("commits")], [])      # no receipts, no marker
        self.assertNotIn("tier2-evidence.jsonl", names)                          # no evidence record
        self.assertNotIn("pilot-approval.json", names)                           # no approval created or consumed
        self.assertFalse(os.path.exists(os.path.join(self.vault, S.MEMORY_DIR)))  # no memory written


class ValidDiskInputTests(DiskCase):
    def test_the_production_builder_reads_the_pack_and_ledger_from_disk_and_builds_the_plan(self):
        with mock.patch.object(CLI.scan, "run_pipeline", wraps=scan.run_pipeline) as pipeline:
            del _OPENED[:]
            _RECORDING.append(1)
            self.addCleanup(_RECORDING.clear)
            plan = self.builder()()
            _RECORDING.clear()
        # the inputs really came from disk
        opened = set(_OPENED)
        self.assertIn(self.pack_path(), opened)
        self.assertIn(self.ledger_path(), opened)
        self.assertIn(os.path.join(self.vault, "VAULT-INDEX.md"), opened)        # provenance and re-derivation
        self.assertNotIn(os.path.join(self.root, RV.PRIVATE), opened)            # absent: only an existence check
        # the production re-derivation ran the real pipeline with the pack's own timestamp
        pipeline.assert_called_once_with(self.transcripts, self.vault, self.generated_at)
        # the plan matches what is on disk
        with open(self.pack_path(), "rb") as f:
            raw = f.read()
        with open(self.ledger_path(), encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        lock = next(r for r in records if r["action"] == "lock")
        self.assertEqual(plan.candidate, self.cid)
        self.assertEqual(plan.run, os.path.basename(self.run_dir))
        self.assertEqual(plan.pack_sha, X.sha(raw.decode("utf-8")))
        self.assertEqual(plan.lock_hash, lock["hash"])
        self.assertEqual(json.loads(plan.lock_record_json)["hash"], lock["hash"])
        self.assertEqual((plan.operation, plan.kind), ("put", "commit"))
        self.assertEqual(plan.item.attribute, "occupation")
        self.assertEqual(plan.item.value, "wholesale real estate")
        self.assertEqual(plan.evidence, "I wholesale real estate for a living.")
        self.assertEqual(plan.decision.decision, "ACCEPT")
        self.assertEqual(int(plan.age_seconds), 0)                                # planned at the lock instant
        self.assertEqual(plan.checks[:6], ("inputs", "replay", "committable", "age", "provenance", "rederive"))
        self.assert_no_write_artifacts()

    def test_the_plan_is_rebuilt_from_scratch_each_call_and_is_stable(self):
        build = self.builder()
        first, second = build(), build()
        self.assertEqual(C.compare_plans(first, second), [])
        with mock.patch.object(CLI.scan, "run_pipeline", wraps=scan.run_pipeline) as pipeline:
            build()
            build()
        self.assertEqual(pipeline.call_count, 2)                                   # re-derived every time

    def test_the_whole_execute_flow_accepts_the_disk_backed_plan_up_to_the_challenge(self):
        """The production builder feeds `memory_execute.commit`; the person declines at the challenge, so nothing is written."""
        shown = []
        out = EX.commit(make_plan=self.builder(), store=self.store, receipts_dir=self.root,
                        ask=lambda p: (shown.append(p), "no")[1], say=self.said.append, clock=self.clock.stamp,
                        monotonic=self.mono, nonce_fn=lambda: "ab" * 16)
        self.assertEqual((out.status, out.code), ("declined", "challenge_declined"))
        self.assertEqual(len(shown), 1)
        self.assert_no_write_artifacts()


class InvalidDiskInputTests(DiskCase):
    def test_a_missing_pack_is_inputs_unreadable(self):
        os.unlink(self.pack_path())
        self.assert_refused("inputs_unreadable", ("FileNotFoundError",))

    def test_a_truncated_pack_is_inputs_unreadable(self):
        with open(self.pack_path(), "rb") as f:
            raw = f.read()
        with open(self.pack_path(), "wb") as f:
            f.write(raw[:len(raw) // 2])
        self.assert_refused("inputs_unreadable", ("JSONDecodeError",))

    def test_a_missing_ledger_is_bad_input(self):
        os.unlink(self.ledger_path())
        self.assert_refused("bad_input", ("records",))

    def test_a_truncated_ledger_is_inputs_unreadable(self):
        with open(self.ledger_path(), "rb") as f:
            raw = f.read()
        with open(self.ledger_path(), "wb") as f:
            f.write(raw[:-20])
        self.assert_refused("inputs_unreadable", ("ReviewError",))

    def test_a_tampered_lock_record_breaks_the_hash_chain(self):
        with open(self.ledger_path(), encoding="utf-8") as f:
            lines = f.read().split("\n")
        rec = json.loads(lines[1])
        rec["value"] = "something else"
        lines[1] = R.canonical_json(rec)
        with open(self.ledger_path(), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.assert_refused("ledger_replay_invalid", ("hash:1",))

    def test_a_ledger_missing_its_first_record_is_invalid(self):
        with open(self.ledger_path(), encoding="utf-8") as f:
            lines = [l for l in f.read().split("\n") if l]
        with open(self.ledger_path(), "w", encoding="utf-8") as f:
            f.write("\n".join(lines[1:]) + "\n")
        self.assert_refused("ledger_replay_invalid", ("seq:0", "first_not_open"))

    def test_a_pack_changed_after_the_lock_no_longer_replays(self):
        with open(self.pack_path(), "a", encoding="utf-8") as f:
            f.write("\n")
        self.assert_refused("ledger_replay_invalid", ("replay:0:mismatch",))

    def test_a_candidate_that_was_never_locked_is_not_committable(self):
        self.assert_refused("not_committable", ("pending",), candidate=self.other)

    def test_an_unknown_candidate_is_bad_input(self):
        self.assert_refused("bad_input", ("candidate",), candidate="cand_does_not_exist")

    def test_a_source_that_changed_since_the_lock_is_stale(self):
        path = os.path.join(self.vault, "VAULT-INDEX.md")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace("wholesale real estate", "sell used cars"))
        self.assert_refused("source_stale")

    def test_a_lock_older_than_fourteen_days_is_refused_and_exactly_at_the_limit_is_not(self):
        self.clock.advance(C.LOCK_MAX_AGE_DAYS * 86400)                            # the lock is exactly 14 days old
        self.assertEqual(int(self.builder()().age_seconds), C.LOCK_MAX_AGE_DAYS * 86400)
        self.clock.advance(1)
        self.assert_refused("lock_too_old", ())

    def test_a_malformed_private_side_file_is_inputs_unreadable(self):
        with open(os.path.join(self.root, RV.PRIVATE), "w", encoding="utf-8") as f:
            f.write("not json\n")
        self.assert_refused("inputs_unreadable", ("ReviewError",))


class RefusalNeverWritesTests(DiskCase):
    """Each case runs the production builder through `memory_execute.commit`: it is refused as `plan_refused:<code>` before
    the person is asked, and nothing is written (no receipts, evidence, approval or memory)."""

    def run_flow(self):
        before = self.snapshot()
        asked = []
        out = EX.commit(make_plan=self.builder(), store=self.store, receipts_dir=self.root,
                        ask=lambda p: asked.append(p) or "x", say=self.said.append, clock=self.clock.stamp,
                        monotonic=self.mono, nonce_fn=lambda: "cd" * 16)
        self.assertEqual(asked, [])
        self.assertEqual(self.snapshot(), before)
        self.assert_no_write_artifacts()
        return out

    def test_a_missing_pack(self):
        os.unlink(self.pack_path())
        out = self.run_flow()
        self.assertEqual((out.status, out.code), ("refused", "plan_refused:inputs_unreadable"))

    def test_a_tampered_ledger(self):
        with open(self.ledger_path(), encoding="utf-8") as f:
            lines = f.read().split("\n")
        rec = json.loads(lines[1])
        rec["value"] = "x" * 8
        lines[1] = R.canonical_json(rec)
        with open(self.ledger_path(), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        out = self.run_flow()
        self.assertEqual((out.status, out.code), ("refused", "plan_refused:ledger_replay_invalid"))

    def test_a_stale_source(self):
        path = os.path.join(self.vault, "VAULT-INDEX.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("nothing here any more\n")
        out = self.run_flow()
        self.assertEqual((out.status, out.code), ("refused", "plan_refused:source_stale"))

    def test_an_expired_lock(self):
        self.clock.advance(C.LOCK_MAX_AGE_DAYS * 86400 + 3600)
        out = self.run_flow()
        self.assertEqual((out.status, out.code), ("refused", "plan_refused:lock_too_old"))


class IsolationTests(DiskCase):
    def test_the_guards_are_live(self):
        with self.assertRaises(AssertionError):
            subprocess.run(["true"])
        with self.assertRaises(AssertionError):
            urllib.request.urlopen("https://example.invalid")
        with self.assertRaises(AssertionError):
            os.listdir(G.ORIGINAL)                                  # refused before it runs
        del self.touched[:]

    def test_every_fixture_lives_in_the_temp_directory(self):
        for p in (self.vault, self.root, self.transcripts, self.run_dir):
            self.assertTrue(p.startswith(self.top + os.sep), p)
        self.assertFalse(G.is_protected(self.top))


if __name__ == "__main__":
    unittest.main()
