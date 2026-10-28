"""Memory 2.0: the `--real-vault --commit` composition, end to end, on TEMPORARY resources only.

HOW THE REAL-VAULT RULES ARE EXERCISED WITHOUT THE REAL VAULT. `S.REAL_VAULT` is patched to a temp folder, so the store
takes its genuine `allow_real_vault=True` constructor path: `require_authorization` on, no injected clock allowed, the
store's own clock used. That clock is `S._system_utc`, looked up when the store is constructed (memory_store.py: `self._clock =
clock if clock is not None else _system_utc`), so patching the module attribute swaps only the time source, not the rule.
The review ledger and pack are the existing `Machine` fixtures handed in through the CLI's `make_plan_builder` seam (the
same seam the other CLI tests use), the Tier 2 phone side is a fake confirmer handed in through `confirmer_factory`, and the
approval file is written by the test into the temp review root, mode 0600.

WHAT THESE TESTS DO NOT SHOW. Live Telegram behavior, the Keychain item, the 90 s and 30 s budgets, the production plan
builder reading a real ledger and pack from disk, and anything about the real vault itself. A guard (tests/realvault_guard.py) fails the test if the
ORIGINAL real-vault path is opened, listed, stat-ed, created or modified; subprocess and the network fail the test too."""
import copy
import hashlib
import json
import os
import subprocess
import urllib.request
from types import SimpleNamespace
from unittest import mock

from backtalk import memory_commit as C
from backtalk import memory_evidence as EV
from backtalk import memory_execute as EX
from backtalk import memory_receipts as RC
from backtalk import memory_store as S
from backtalk.evals.memory_extraction import commit_cli as CLI
from tests import realvault_guard as G
from tests.test_memory_commit_cli import CliCase
from tests.test_memory_commit_cli import NONCE
from tests.test_memory_tier2_integration import FakeConfirmer

ORIGINAL_REAL_VAULT = G.ORIGINAL


class RealVaultPathCase(CliCase):
    def setUp(self):
        super().setUp()
        self.lock("occupation")
        self.cand = self.cid("occupation")
        self.tick = 0.0
        self.install_guards()
        # the store's real-vault constructor path, on a deterministic clock (no clock is ever injected into the store)
        self.patch(mock.patch.object(S, "REAL_VAULT", self.vault))
        self.patch(mock.patch.object(S, "_system_utc", self.clock))

    # -- guards --
    def patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def install_guards(self):
        def no_run(*a, **k):
            raise AssertionError("a subprocess was started")

        def no_net(*a, **k):
            raise AssertionError("the network was contacted")
        self.patch(mock.patch.object(subprocess, "run", no_run))
        self.patch(mock.patch.object(subprocess, "Popen", no_run))
        self.patch(mock.patch.object(urllib.request, "urlopen", no_net))
        self.touched = G.arm(self)                          # audit hook + metadata wrappers, for this test only

    # -- fixtures --
    def mono(self):
        self.tick += 0.5
        return self.tick

    def builder(self, *a, **k):
        def build():
            return C.plan(pack=self.pack, pack_sha="a" * 64, records=self.state.records, private=None,
                          candidate=self.cand, store=a[3], provenance=lambda v: [], rederive=lambda v: copy.deepcopy(v),
                          clock=self.clock.stamp, monotonic=self.mono, preview_budget_s=EX.PREVIEW_BUDGET_S)
        return build

    def real_store(self):
        return S.MemoryStore(self.vault, allow_real_vault=True)

    def plan(self):
        return self.builder(None, None, None, self.real_store())()

    def write_approval(self, mode=0o600, **over):
        plan = self.plan()
        data = {"candidate": plan.candidate, "lock_hash": plan.lock_hash, "approved_at": self.clock.stamp()}
        data.update(over)
        path = os.path.join(self.root, EV.APPROVAL_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.chmod(path, mode)
        return data

    def commit(self, confirmer=None, nonce=NONCE, typed=None, factory=None):
        if typed is None:
            plan = self.plan()
            typed = [C.expected_answer(plan, nonce)]
        self.said.clear()
        conf = confirmer if confirmer is not None else FakeConfirmer()
        with mock.patch.object(CLI, "make_plan_builder", self.builder):
            return CLI.main(["--root", self.root, "--vault", self.vault, "--transcripts", self.transcripts,
                             "--real-vault", "--commit", self.cand, "--run", "run-dir"],
                            tty=True, ask=lambda p: typed.pop(0), say=self.said.append, clock=self.clock.stamp,
                            monotonic=self.mono, nonce_fn=lambda: nonce,
                            confirmer_factory=factory or (lambda m: conf))

    # -- observation --
    def items(self):
        return self.real_store().list_items() if os.path.isdir(os.path.join(self.vault, S.MEMORY_DIR)) else []

    def receipt_actions(self):
        log = RC.ReceiptLog(self.root)
        return [r["action"] for r in log.records()] if os.path.exists(log.path) else []

    def evidence_events(self):
        return [r["event"] for r in EV.EvidenceLog(self.root).records()]

    def assert_original_untouched(self):
        self.assertEqual(self.touched, [])

    def assert_nothing_installed(self):
        self.assertEqual(self.items(), [])
        self.assertEqual(self.receipt_actions(), [])           # no intent, and no `done` falsely recorded


class GuardTests(RealVaultPathCase):
    """The guards themselves fail loudly (the audit hook refuses BEFORE the operation runs, so these touch nothing)."""

    def test_the_original_real_vault_path_cannot_be_listed_opened_or_created(self):
        for op in (lambda: os.listdir(ORIGINAL_REAL_VAULT),
                   lambda: open(os.path.join(ORIGINAL_REAL_VAULT, "x"), "w"),
                   lambda: os.mkdir(os.path.join(ORIGINAL_REAL_VAULT, S.MEMORY_DIR))):
            with self.assertRaises(AssertionError):
                op()
        self.assertEqual(len(self.touched), 3)
        del self.touched[:]

    def test_subprocess_and_the_network_fail_immediately(self):
        with self.assertRaises(AssertionError):
            subprocess.run(["true"])
        with self.assertRaises(AssertionError):
            urllib.request.urlopen("https://example.invalid")


class RealVaultSuccessTests(RealVaultPathCase):
    def test_the_genuine_real_vault_constructor_path_is_the_one_under_test(self):
        store = self.real_store()
        self.assertTrue(store.require_authorization)                      # the real-vault rule is on
        with self.assertRaises(S.StoreError) as cm:
            S.MemoryStore(self.vault, allow_real_vault=True, clock=self.clock)
        self.assertEqual(cm.exception.reason, "clock_not_allowed")        # and no clock can be injected
        self.assertIs(store._clock, self.clock)                           # its time source is the (patched) system clock

    def test_a_confirmed_commit_runs_every_step_and_the_approval_is_then_spent(self):
        self.write_approval()
        code = self.commit()
        self.assertEqual((code, self.said[-1]), (0, "committed: committed"), self.said)
        self.assertEqual(self.receipt_actions(), ["open", "intent", "done"])
        self.assertEqual(self.evidence_events(), ["tier2_sent", "tier2_confirmed"])
        intent = RC.ReceiptLog(self.root).records()[1]
        want = hashlib.sha256(NONCE.encode()).hexdigest()[:32]
        self.assertEqual(intent["nonce_hash"], want)
        for rec in EV.EvidenceLog(self.root).records():
            self.assertEqual(rec["nonce_hash"], want)
            self.assertEqual((rec["candidate"], rec["lock_hash"]), (intent["candidate"], intent["lock_hash"]))
        self.assertEqual(len(self.items()), 1)
        # a second attempt is refused and installs nothing more
        again = self.commit(nonce="cd" * 16, typed=["x"])      # refused at the first plan: it is never asked
        self.assertEqual(again, 1, self.said)
        self.assertEqual(len(self.items()), 1)
        self.assertEqual(self.receipt_actions(), ["open", "intent", "done"])
        # the approval itself is spent: the gate says so on its own
        gate = EV.Tier2Gate(FakeConfirmer(), self.root, self.clock.stamp, lambda: RC.ReceiptLog(self.root).records())
        self.assertIsNone(gate.prepare())
        spent = SimpleNamespace(kind="commit", candidate=self.cand, lock_hash=gate._approval["lock_hash"])
        self.assertEqual(gate.check_plan(spent), "approval_consumed")
        self.assert_original_untouched()


class RealVaultRefusalTests(RealVaultPathCase):
    def refused(self, code_text, *, evidence):
        self.assertEqual(self.commit_code, 1, self.said)
        self.assertEqual(self.said[-1], code_text)
        self.assert_nothing_installed()
        self.assertEqual(self.evidence_events() if os.path.exists(os.path.join(self.root, EV.EVIDENCE_FILE)) else [],
                         evidence)
        self.assert_original_untouched()

    def test_a_missing_approval_file_refuses_before_anything_is_planned(self):
        self.commit_code = self.commit(typed=["x"])
        self.refused("refused: tier2:approval_missing", evidence=[])

    def test_an_approval_file_with_mode_0644_is_refused_and_not_repaired(self):
        self.write_approval(mode=0o644)
        self.commit_code = self.commit(typed=["x"])
        self.refused("refused: tier2:approval_permissions", evidence=[])
        self.assertEqual(os.stat(os.path.join(self.root, EV.APPROVAL_FILE)).st_mode & 0o777, 0o644)

    def test_an_approval_for_another_lock_hash_is_refused_before_the_person_is_asked(self):
        self.write_approval(lock_hash="f" * 64)
        asked = []
        plan = self.plan()
        self.said.clear()
        with mock.patch.object(CLI, "make_plan_builder", self.builder):
            code = CLI.main(["--root", self.root, "--vault", self.vault, "--transcripts", self.transcripts,
                             "--real-vault", "--commit", self.cand, "--run", "run-dir"], tty=True,
                            ask=lambda p: asked.append(p) or "x", say=self.said.append, clock=self.clock.stamp,
                            monotonic=self.mono, nonce_fn=lambda: NONCE, confirmer_factory=lambda m: FakeConfirmer())
        self.commit_code = code
        self.assertEqual(asked, [])
        self.refused("refused: tier2:approval_lock_mismatch", evidence=[])

    def test_a_confirmer_that_returns_no_confirmation_leaves_only_the_counted_send(self):
        self.write_approval()
        self.commit_code = self.commit(confirmer=FakeConfirmer(ok=False, code="timeout"))
        self.refused("refused: tier2:timeout", evidence=["tier2_sent"])      # the send is counted; nothing else is written

    def test_a_confirmer_that_declines_leaves_only_the_counted_send(self):
        self.write_approval()
        self.commit_code = self.commit(confirmer=FakeConfirmer(ok=False, code="declined"))
        self.refused("refused: tier2:declined", evidence=["tier2_sent"])

    def test_no_confirmer_at_all_refuses_closed(self):
        self.write_approval()
        self.commit_code = self.commit(factory=lambda m: None)
        self.refused("refused: tier2:not_configured", evidence=[])
