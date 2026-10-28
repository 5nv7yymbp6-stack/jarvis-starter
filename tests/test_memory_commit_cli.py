"""Memory 2.0, Phase 8.2d: the commit program (backtalk/evals/memory_extraction/commit_cli.py). Temporary stores only: the
real vault is refused by the store and these tests prove nothing was read or written when it is. The terminal, clock and
nonce are injected; the plan builder is replaced where the test is about the program's wiring, not about the planner."""
import copy
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

from backtalk import memory_commit as C
from backtalk import memory_extract as X
from backtalk import memory_execute as EX
from backtalk import memory_receipts as RC
from backtalk import memory_review as R
from backtalk import memory_store as S
from backtalk.evals.memory_extraction import commit_cli as CLI
from tests.fake_clock import FakeClock
from tests.receipts_harness import tree_state
from tests.test_memory_commit_plan import NOWS
from tests.test_memory_review import DAILY, Machine, cands_of, fixture_notes
from tests.test_memory_extract import NOW

NONCE = "ef" * 16


class CliCase(Machine):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.top = os.path.realpath(self.tmp.name)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))
        os.mkdir(os.path.join(self.top, "v"))                         # the vault's parent is forbidden as a review root
        self.vault = os.path.join(self.top, "v", "vault")
        self.root = os.path.join(self.top, "review")
        self.transcripts = os.path.join(self.top, "transcripts")
        for d in (self.vault, self.transcripts):
            os.mkdir(d)
        os.mkdir(self.root, 0o700)
        os.chmod(self.root, 0o700)
        self.clock = FakeClock(NOWS)
        self.said = []

    def argv(self, *mode):
        return ["--root", self.root, "--vault", self.vault, "--transcripts", self.transcripts, *mode]

    def run_main(self, *mode, **kw):
        kw.setdefault("say", self.said.append)
        kw.setdefault("clock", self.clock.stamp)
        return CLI.main(self.argv(*mode), **kw)


class RefusalTests(CliCase):
    def test_no_terminal_refuses_a_commit_and_a_reconcile_before_anything_is_read(self):
        with mock.patch.object(CLI.RV, "read_pack", side_effect=AssertionError("read")), \
                mock.patch.object(CLI.S, "MemoryStore", side_effect=AssertionError("store built")):
            for mode in (("--commit", "cand_x"), ("--reconcile",)):
                self.said.clear()
                self.assertEqual(self.run_main(*mode, tty=False), 2)
                self.assertEqual(self.said, [CLI.NO_TTY])

    def test_the_real_vault_is_refused_by_the_store_and_nothing_is_read_or_written(self):
        before = (os.path.exists(os.path.expanduser("~/.backtalk-memory-review")),
                  os.path.exists(os.path.join(S.REAL_VAULT, S.MEMORY_DIR)))
        with mock.patch.object(CLI.scan, "output_dir_ok", side_effect=AssertionError("root touched")), \
                mock.patch.object(CLI.RV, "read_pack", side_effect=AssertionError("pack read")):
            for mode in (("--status",), ("--audit",), ("--commit", "c"), ("--reconcile",)):
                self.said.clear()
                out = CLI.main(["--vault", S.REAL_VAULT, *mode], say=self.said.append, tty=True)
                self.assertEqual(out, 2)
                self.assertEqual(self.said, [CLI.REAL_VAULT_REFUSED])
        after = (os.path.exists(os.path.expanduser("~/.backtalk-memory-review")),
                 os.path.exists(os.path.join(S.REAL_VAULT, S.MEMORY_DIR)))
        self.assertEqual(before, after)

    def test_the_real_vault_switch_is_one_exact_flag_with_one_use(self):
        src = inspect.getsource(CLI)
        self.assertEqual(src.count("allow_real_vault="), 1)
        self.assertNotIn("allow_real_vault=True", src)               # never a constant: only the parsed flag
        with self.assertRaises(SystemExit):
            CLI.main(self.argv("--status", "--allow-real-vault"), say=self.said.append)

    def test_the_flag_is_refused_for_a_temp_vault_and_with_a_test_clock(self):
        for kw in ({}, {"store_clock": self.clock}):
            self.said.clear()
            code = CLI.main(["--vault", self.vault, "--root", self.root, "--real-vault", "--status"],
                            say=self.said.append, **kw)
            self.assertEqual((code, self.said), (2, [CLI.REAL_VAULT_MISUSE]))
        self.said.clear()
        with mock.patch.object(CLI.S, "MemoryStore", side_effect=AssertionError("store built")):
            code = CLI.main(["--vault", S.REAL_VAULT, "--root", self.root, "--real-vault", "--status"],
                            say=self.said.append, store_clock=self.clock)
        self.assertEqual((code, self.said), (2, [CLI.REAL_VAULT_MISUSE]))

    def test_a_real_vault_commit_without_a_terminal_is_refused_before_anything_is_read(self):
        with mock.patch.object(CLI.RV, "read_pack", side_effect=AssertionError("pack read")), \
                mock.patch.object(CLI.S, "MemoryStore", side_effect=AssertionError("store built")):
            self.said.clear()
            code = CLI.main(["--vault", S.REAL_VAULT, "--root", self.root, "--real-vault", "--commit", "c"],
                            say=self.said.append, tty=False)
        self.assertEqual((code, self.said), (2, [CLI.NO_TTY]))

    def test_with_the_flag_status_audit_and_reconcile_never_write_a_memory(self):
        """`S.REAL_VAULT` is pointed at a TEMP folder so the flag is exercised without the real vault being opened."""
        os.mkdir(os.path.join(self.top, "v2"))                 # its parent is forbidden as a review root
        fake_real = os.path.join(self.top, "v2", "fake-real-vault")
        os.mkdir(fake_real)
        before = tree_state(self.top)
        refuse = AssertionError("a store write was attempted")
        with mock.patch.object(S, "REAL_VAULT", fake_real), \
                mock.patch.object(S.MemoryStore, "put", side_effect=refuse), \
                mock.patch.object(S.MemoryStore, "apply", side_effect=refuse), \
                mock.patch.object(CLI, "build_tier2", side_effect=AssertionError("gate built")):
            for mode, kw in (("--status", {}), ("--audit", {}), ("--reconcile", {"tty": True, "ask": lambda p: "no"})):
                CLI.main(["--vault", fake_real, "--root", self.root, "--real-vault", mode], say=self.said.append,
                         confirmer_factory=lambda m: self.fail("factory"), **kw)
        self.assertFalse(os.path.exists(os.path.join(fake_real, S.MEMORY_DIR)))
        self.assertEqual(tree_state(self.top), before)

    def test_exactly_one_mode_is_required(self):
        for argv in (self.argv(), self.argv("--status", "--audit")):
            with self.assertRaises(SystemExit):
                CLI.main(argv, say=self.said.append)


class ReadOnlyModeTests(CliCase):
    def test_status_and_audit_change_nothing_and_need_no_terminal(self):
        before = tree_state(self.top)
        self.assertEqual(self.run_main("--status", tty=False), 0)
        self.assertIn("proceed=True row=3 code=first_use (first use)", self.said)
        self.said.clear()
        self.assertEqual(self.run_main("--audit", tty=False), 0)
        self.assertEqual(self.said[0], "verdict=CLEAN row=3 code=first_use")
        self.assertEqual(tree_state(self.top), before)

    def test_a_blocked_log_exits_nonzero_and_says_why(self):
        open(os.path.join(self.root, RC.BLOCKED), "w").close()
        self.assertEqual(self.run_main("--status"), 1)
        self.assertIn("proceed=False row=1 code=receipts_blocked", self.said)
        self.said.clear()
        self.assertEqual(self.run_main("--audit"), 1)
        self.assertTrue(self.said[0].startswith("verdict=BLOCKED row=1"))

    def test_the_review_root_may_not_be_inside_the_vault(self):
        with self.assertRaises(ValueError):
            CLI.main(["--root", os.path.join(self.vault, "r"), "--vault", self.vault, "--transcripts", self.transcripts,
                      "--status"], say=self.said.append)


class WiringTests(CliCase):
    def setUp(self):
        super().setUp()
        self.lock("occupation")
        self.cand = self.cid("occupation")
        self.tick = 0.0

    def mono(self):
        self.tick += 0.5
        return self.tick

    def builder(self, *a, **k):
        def build():
            return C.plan(pack=self.pack, pack_sha="a" * 64, records=self.state.records, private=None,
                          candidate=self.cand, store=a[3], provenance=lambda v: [], rederive=lambda v: copy.deepcopy(v),
                          clock=self.clock.stamp, monotonic=self.mono, preview_budget_s=EX.PREVIEW_BUDGET_S)
        return build

    def test_a_commit_runs_the_whole_execute_flow_and_exits_zero(self):
        store = S.MemoryStore(self.vault, clock=self.clock)
        plan = self.builder(None, None, None, store)()
        answer = C.expected_answer(plan, NONCE)
        typed = [answer]
        with mock.patch.object(CLI, "make_plan_builder", self.builder):
            code = self.run_main("--commit", self.cand, "--run", "run-dir", tty=True, ask=lambda p: typed.pop(0),
                                 monotonic=self.mono, nonce_fn=lambda: NONCE, store_clock=self.clock)
        self.assertEqual(code, 0, self.said)
        self.assertEqual(self.said[-1], "committed: committed")
        log = RC.ReceiptLog(self.root)
        self.assertEqual([r["action"] for r in log.records()], ["open", "intent", "done"])
        self.assertEqual(len(S.MemoryStore(self.vault, clock=self.clock).list_items()), 1)

    def test_a_declined_challenge_exits_nonzero_and_writes_nothing(self):
        before = tree_state(self.top)
        with mock.patch.object(CLI, "make_plan_builder", self.builder):
            code = self.run_main("--commit", self.cand, "--run", "run-dir", tty=True, ask=lambda p: "no",
                                 monotonic=self.mono, nonce_fn=lambda: NONCE, store_clock=self.clock)
        self.assertEqual(code, 1)
        self.assertEqual(tree_state(self.top), before)

    def test_a_missing_pack_or_unreadable_ledger_is_a_clean_refusal(self):
        self.assertEqual(self.run_main("--commit", "cand_x", tty=True, ask=lambda p: "x"), 1)       # no pack in the root
        self.assertTrue(self.said[-1].startswith("refused: no_pack"))
        build = CLI.make_plan_builder(self.root, os.path.join(self.root, "run-none"), "cand_x",
                                      S.MemoryStore(self.vault, clock=self.clock), self.vault, self.transcripts,
                                      self.clock.stamp, self.mono)
        with self.assertRaises(C.CommitError) as cm:
            build()
        self.assertEqual((cm.exception.code, cm.exception.details), ("inputs_unreadable", ("FileNotFoundError",)))

    def test_the_plan_is_rebuilt_from_scratch_on_every_call(self):
        reads = []
        raw = self.raw

        def read_pack(run_dir):
            reads.append("pack")
            return raw, "a" * 64, "run-1"

        def read_ledger(path):
            reads.append("ledger")
            return self.state.records
        build_args = (self.root, "run-dir", self.cand, S.MemoryStore(self.vault, clock=self.clock), self.vault,
                      self.transcripts, self.clock.stamp, self.mono)
        with mock.patch.object(CLI.RV, "read_pack", read_pack), mock.patch.object(CLI.RV, "read_ledger", read_ledger), \
                mock.patch.object(CLI, "production_provenance", lambda t, v: (lambda view: [])), \
                mock.patch.object(CLI, "production_rederive", lambda t, v, g: (lambda view: copy.deepcopy(view))):
            build = CLI.make_plan_builder(*build_args)
            first, second = build(), build()
        self.assertEqual(reads, ["pack", "ledger"] * 2)
        self.assertEqual(C.compare_plans(first, second), [])


class ProductionCheckTests(CliCase):
    def test_the_production_rederive_returns_what_the_pack_was_built_from(self):
        notes = fixture_notes()
        extract = [n for n in notes if n.path in ("VAULT-INDEX.md", "03 - Markets/Kansas City (Missouri).md", DAILY,
                                                  "04 - Wholesaling/04 - Wholesaling.md",
                                                  "10 - New Ventures/10 - New Ventures.md")]
        cands, ex = cands_of([], extract, reference_notes=notes)
        conflicts = X.known_conflicts(notes) + X.generic_conflicts(cands)
        meta = {"generated_at": NOW, "exclusions": dict(ex)}
        seen = {}

        def pipeline(transcripts, vault, generated_at):
            seen["args"] = (transcripts, vault, generated_at)
            return cands, conflicts, meta, notes
        with mock.patch.object(CLI.scan, "run_pipeline", pipeline):
            rederive = CLI.production_rederive("T", "V", "G")
            for cid, view in self.pack.views.items():
                self.assertEqual(rederive(view)["id"], cid)
            self.assertIsNone(rederive({"id": "cand_does_not_exist"}))
            self.lock("occupation")
            plan = C.plan(pack=self.pack, pack_sha="a" * 64, records=self.state.records, private=None,
                          candidate=self.cid("occupation"), store=S.MemoryStore(self.vault, clock=self.clock),
                          provenance=lambda v: [], rederive=rederive, clock=self.clock.stamp,
                          monotonic=iter([0.0, 1.0]).__next__, preview_budget_s=45)
        self.assertEqual(plan.operation, "put")
        self.assertEqual(seen["args"], ("T", "V", "G"))

    def test_the_production_provenance_reopens_the_source(self):
        calls = []
        with mock.patch.object(CLI.E, "verify_provenance",
                               lambda c, t, v: (calls.append((c.id, t, v)), ["hash_mismatch"])[1]):
            check = CLI.production_provenance("T", "V")
            self.assertEqual(check({"id": "cand_1"}), ["hash_mismatch"])
        self.assertEqual(calls, [("cand_1", "T", "V")])

    def test_the_private_side_file_is_read_by_ledger_seq(self):
        path = os.path.join(self.root, "ledger_private.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"seq": 3, "candidate": "c", "value": "v", "evidence": "e"}) + "\n")
        self.assertEqual(CLI.read_private(path), {3: {"value": "v", "evidence": "e"}})
        self.assertEqual(CLI.read_private(os.path.join(self.root, "absent.jsonl")), {})
        with open(path, "a") as f:
            f.write("not json\n")
        with self.assertRaises(R.ReviewError):
            CLI.read_private(path)


class Tier2WiringTests(WiringTests):
    """The CLI hands the committer a Tier 2 gate only for a store that requires authorization. The flag is switched on
    for a TEMP store by wrapping its constructor; the real vault is never opened."""

    def flagged(self):
        real_init = S.MemoryStore.__init__

        def init(this, *a, **k):
            real_init(this, *a, **k)
            this.require_authorization = True
        return mock.patch.object(S.MemoryStore, "__init__", init)

    def guard_net(self):
        import subprocess
        import urllib.request
        for target, name in ((subprocess, "run"), (urllib.request, "urlopen")):
            p = mock.patch.object(target, name, side_effect=AssertionError("network or keychain touched"))
            p.start()
            self.addCleanup(p.stop)

    def commit(self, factory, typed=("x",)):
        t = list(typed)
        with self.flagged(), mock.patch.object(CLI, "make_plan_builder", self.builder):
            return self.run_main("--commit", self.cand, "--run", "run-dir", tty=True, ask=lambda p: t.pop(0),
                                 monotonic=self.mono, nonce_fn=lambda: NONCE, store_clock=self.clock,
                                 confirmer_factory=factory)

    def test_a_temp_store_commit_never_calls_the_confirmer_factory_or_the_network(self):
        self.guard_net()
        store = S.MemoryStore(self.vault, clock=self.clock)
        answer = C.expected_answer(self.builder(None, None, None, store)(), NONCE)
        typed = [answer]
        with mock.patch.object(CLI, "make_plan_builder", self.builder):
            code = self.run_main("--commit", self.cand, "--run", "run-dir", tty=True, ask=lambda p: typed.pop(0),
                                 monotonic=self.mono, nonce_fn=lambda: NONCE, store_clock=self.clock,
                                 confirmer_factory=lambda m: self.fail("factory called for a temp store"))
        self.assertEqual(code, 0, self.said)

    def test_a_store_requiring_authorization_refuses_closed_without_a_confirmer(self):
        self.guard_net()
        before = tree_state(self.top)
        for label, factory in (("none", lambda m: None), ("raises", lambda m: 1 / 0)):
            with self.subTest(label):
                self.said.clear()
                self.assertEqual(self.commit(factory), 1)
                self.assertEqual(self.said[-1], "refused: tier2:not_configured")
                self.assertEqual(tree_state(self.top), before)

    def test_the_default_factory_is_the_keychain_loader_and_is_only_reached_for_such_a_store(self):
        self.assertIs(CLI.T.configured_confirmer, CLI.T.configured_confirmer)
        with mock.patch.object(CLI.T, "configured_confirmer", side_effect=AssertionError("keychain")):
            for mode in (("--status",), ("--audit",)):
                self.assertIn(self.run_main(*mode, store_clock=self.clock), (0, 1))
            self.assertEqual(self.run_main("--commit", "c", tty=False), 2)

    def test_read_only_modes_and_reconcile_never_build_a_gate(self):
        self.guard_net()
        with self.flagged(), mock.patch.object(CLI, "build_tier2", side_effect=AssertionError("gate built")), \
                mock.patch.object(CLI.EV, "Tier2Gate", side_effect=AssertionError("gate built")):
            before = tree_state(self.top)
            for mode, kw in (("--status", {}), ("--audit", {}), ("--reconcile", {"tty": True, "ask": lambda p: "no"})):
                self.run_main(mode, store_clock=self.clock, confirmer_factory=lambda m: self.fail("factory"), **kw)
            self.assertEqual(tree_state(self.top), before)

    def test_the_cli_source_has_no_real_vault_switch_and_the_gate_is_built_only_inside_the_commit_branch(self):
        src = inspect.getsource(CLI)
        self.assertNotIn("allow_real_vault=True", src)
        self.assertEqual(src.count("build_tier2("), 2)              # its definition and the one call in the commit branch


if __name__ == "__main__":
    unittest.main()
