"""Memory 2.0, Phase 7: the versioned key registry (backtalk/memory_keys.py)."""
import ast
import inspect
import unittest

from backtalk import memory_keys as K


class RegistryTests(unittest.TestCase):
    def test_the_registry_is_versioned_and_pinned(self):
        self.assertEqual(K.REGISTRY_VERSION, 1)
        self.assertEqual(K.SINGLETONS, ("residence", "occupation", "business_start", "operating_focus", "language", "address_as",
                                        "vehicle", "new_ventures_priority"))
        self.assertEqual(K.SET_PREFIXES, ("pref_", "goal_", "rule_", "dislike_"))
        self.assertEqual(K.ALIASES, {"home": "residence", "address": "residence", "job": "occupation", "work": "occupation",
                                     "business_started": "business_start", "started_business": "business_start",
                                     "focus_market": "operating_focus", "market_focus": "operating_focus",
                                     "preferred_language": "language", "shopify_priority": "new_ventures_priority",
                                     "venture_priority": "new_ventures_priority"})

    def test_kinds(self):
        for k in K.SINGLETONS:
            self.assertEqual(K.kind_of(k), "singleton", k)
        for k in ("pref_x", "goal_close_deals", "rule_a", "dislike_b"):
            self.assertEqual(K.kind_of(k), "set", k)
        self.assertEqual(K.kind_of("pref_a"), "set")
        for k in ("pref_", "goal_", "weird", "", "residence_x", "Pref_x", None, 5):
            self.assertEqual(K.kind_of(k), "unknown", k)

    def test_aliases_resolve_to_the_canonical_key(self):
        for alias, canon in K.ALIASES.items():
            self.assertEqual(K.canonical(alias), canon)
            self.assertEqual(K.kind_of(alias), "singleton")
        self.assertEqual(K.canonical("residence"), "residence")
        self.assertEqual(K.canonical("pref_x"), "pref_x")
        self.assertIsNone(K.canonical(None))

    def test_describe(self):
        self.assertEqual(K.describe("job"), {"kind": "singleton", "canonical": "occupation", "version": 1})
        self.assertEqual(K.describe("pref_a"), {"kind": "set", "canonical": "pref_a", "version": 1})
        self.assertEqual(K.describe("zzz"), {"kind": "unknown", "canonical": "zzz", "version": 1})

    def test_validate_key(self):
        self.assertEqual(K.validate_key("residence"), ())
        self.assertEqual(K.validate_key("goal_close_ten_deals"), ())
        self.assertEqual(K.validate_key("weird"), ("not_in_registry",))
        self.assertEqual(K.validate_key("pref_"), ("not_in_registry",))
        self.assertEqual(K.validate_key("home"), ("use_canonical:residence",))
        for bad in ("Residence", "has space", "", None, 5, "x" * 70, "-x"):
            self.assertEqual(K.validate_key(bad), ("invalid_name",), repr(bad))

    def test_collisions_are_checked_on_the_canonical_key(self):
        self.assertEqual(K.validate_key("residence", {"residence": "cand_a"}), ("key_taken",))
        self.assertEqual(K.validate_key("home", {"residence": "cand_a"}), ("use_canonical:residence", "key_taken"))
        self.assertEqual(K.validate_key("pref_a", {"pref_b": "c"}), ())
        self.assertEqual(K.validate_key("weird", {"weird": "c"}), ("not_in_registry", "key_taken"))
        self.assertEqual(K.validate_key("residence", {}), ())
        self.assertEqual(K.validate_key("residence", None), ())

    def test_the_module_is_pure(self):
        tree = ast.parse(inspect.getsource(K))
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(mods, {"memory"})
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertFalse(names & {"open", "os", "socket", "print"})


if __name__ == "__main__":
    unittest.main()
