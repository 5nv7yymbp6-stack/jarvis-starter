"""Regression tests for the confirmed Memory 2.0 trust-boundary bypasses (Phase 7 integration audit, Phase 8 design).

Each test states the DESIRED behavior. Phase 8.0 added them as `expectedFailure` markers while the bypass existed; Phase
8.1A closed the `put` bypass, so those four are now plain tests. The two authorization markers became plain
tests with Phase 8.1B. Everything runs on a temp vault.
"""
import ast
import glob
import os
import tempfile
import unittest

from backtalk import memory as M
from backtalk import memory_gate as G
from backtalk import memory_store as S

EV = "I live in San Jose now"
NOW = "2026-02-01T00:00:00Z"


def make(n=1, **over):
    d = {"id": f"mem_{n:012x}", "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": "2026-01-10T00:00:00Z", "updated": "2026-01-10T00:00:00Z", "source": {"kind": "user", "ref": "turn 41"},
         "confidence": "stated", "valid_from": "2026-01-10T00:00:00Z", "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


class PutBypassMarkers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.store = S.MemoryStore(self.root)

    def memory_files(self):
        d = os.path.join(self.root, S.MEMORY_DIR)
        return sorted(f for f in os.listdir(d) if f != ".lock") if os.path.isdir(d) else []

    def assert_gate_refuses(self, it, evidence=EV):
        verdict = G.evaluate(G.Proposal(it, evidence), [], NOW)
        self.assertNotEqual(verdict.decision, G.ACCEPT, "premise: the gate must not accept this")
        return verdict

    def assert_put_refused_and_nothing_written(self, it, evidence=EV):
        with self.assertRaises(S.StoreError):
            self.store.put(it, evidence)
        self.assertEqual(self.memory_files(), [])

    def test_put_refuses_a_stated_fact_whose_source_is_a_note(self):
        it = make(1, source={"kind": "note", "ref": "Profile.md#L3"})
        self.assert_gate_refuses(it)
        self.assert_put_refused_and_nothing_written(it)

    def test_put_refuses_a_stated_instruction_whose_source_is_a_note(self):
        it = make(1, type="instruction", attribute="tone", value="keep replies short",
                  source={"kind": "note", "ref": "Profile.md#L3"})
        self.assert_gate_refuses(it, "keep replies short")
        self.assert_put_refused_and_nothing_written(it, "keep replies short")

    def test_put_refuses_a_private_item_without_a_confirmation(self):
        it = make(1, sensitivity="private")
        self.assertEqual(self.assert_gate_refuses(it).decision, G.REQUIRE_CONFIRMATION)
        self.assert_put_refused_and_nothing_written(it)

    def test_put_refuses_evidence_that_does_not_support_the_value(self):
        it = make(1)
        self.assert_gate_refuses(it, "I like tea")
        self.assert_put_refused_and_nothing_written(it, "I like tea")

    def test_apply_already_refuses_the_same_inputs(self):
        """The control: the one validated path (apply) already refuses each of the above, so the bypass is in put alone."""
        cases = [(make(1, source={"kind": "note", "ref": "Profile.md#L3"}), EV),
                 (make(1, sensitivity="private"), EV),
                 (make(1), "I like tea")]
        for it, ev in cases:
            prop = G.Proposal(it, ev)
            with self.assertRaises(S.StoreError):
                self.store.apply(G.evaluate(prop, [], NOW), prop, NOW)
        self.assertEqual(self.memory_files(), [])


class WritePathShapeTests(unittest.TestCase):
    """Static checks: one commit step, and no test-only helper in production."""
    PKG = os.path.dirname(os.path.abspath(S.__file__))

    @staticmethod
    def parse(path):
        with open(path, encoding="utf-8") as f:
            return ast.parse(f.read())

    def test_only_put_and_apply_call_the_commit_step(self):
        tree = self.parse(S.__file__)
        callers = set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_commit":
                        callers.add(fn.name)
        self.assertEqual(callers, {"put", "apply"})

    def test_no_public_store_method_writes_except_put_and_apply(self):
        tree = self.parse(S.__file__)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryStore")
        writers = set()
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_"):
                names = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                         for n in ast.walk(fn) if isinstance(n, ast.Call)}
                if names & {"_commit", "_install", "_write_atomic", "_replace", "_rollback"}:
                    writers.add(fn.name)
        self.assertEqual(writers, {"put", "apply"})

    def test_no_production_module_imports_the_test_only_arrange_helper_or_anything_from_tests(self):
        bad = []
        for path in glob.glob(os.path.join(self.PKG, "**", "*.py"), recursive=True):
            for n in ast.walk(self.parse(path)):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    mods = [n.module or ""] + [(n.module or "") + "." + a.name for a in n.names]
                if any(m.split(".")[0] == "tests" or "store_arrange" in m for m in mods):
                    bad.append(os.path.relpath(path, self.PKG))
                if isinstance(n, (ast.Name, ast.Attribute)) and getattr(n, "id", getattr(n, "attr", "")) == "arrange":
                    bad.append(os.path.relpath(path, self.PKG) + ":arrange")
        self.assertEqual(bad, [])

    def test_the_helper_is_not_reachable_from_the_store_module(self):
        self.assertFalse(hasattr(S, "arrange"))
        self.assertFalse(hasattr(S.MemoryStore, "arrange"))

    def test_the_real_vault_refusal_is_unchanged(self):
        with self.assertRaises(S.StoreError) as cm:
            S.MemoryStore(S.REAL_VAULT)
        self.assertEqual(cm.exception.reason, "real_vault_refused")


class AuthorizationMarkers(unittest.TestCase):
    """The authorization object does not exist yet (Phase 8.1). These only record that nothing in the store API accepts one."""

    def test_apply_accepts_an_authorization_argument(self):
        import inspect
        self.assertIn("authorization", inspect.signature(S.MemoryStore.apply).parameters)

    def test_the_gate_accepts_a_confirmation_argument(self):
        import inspect
        self.assertIn("confirmation", inspect.signature(G.evaluate).parameters)


if __name__ == "__main__":
    unittest.main()
