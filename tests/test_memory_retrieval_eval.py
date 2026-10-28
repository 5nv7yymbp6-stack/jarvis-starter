"""Memory 2.0, Phase 4: the offline retrieval evaluation (backtalk/evals/memory_retrieval). Synthetic data only."""
import json
import os
import unittest

from backtalk import memory as M
from backtalk import memory_retrieval as R
from backtalk import memory_store as S
from backtalk.evals.memory_retrieval import harness as H

CATEGORIES = ("direct exact key query", "synonym/lexical variation", "stated vs inferred", "current vs superseded",
              "unrelated memories", "duplicate memories", "memory already in context", "multiple relevant memories",
              "budget pressure", "historical query", "security rejection", "temporal boundary")


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.cases = H.load()
        cls.report = H.evaluate()

    def test_the_dataset_loads_and_every_corpus_item_is_a_valid_memory(self):
        self.assertGreaterEqual(len(self.corpus), 15)
        self.assertEqual(len({s.item.id for s in self.corpus}), len(self.corpus))
        for s in self.corpus:
            self.assertEqual(M.validate(s.item), ())

    def test_every_required_category_is_covered(self):
        cats = " | ".join(c["category"] for c in self.cases)
        for want in CATEGORIES:
            self.assertIn(want, cats)

    def test_every_case_refers_only_to_corpus_ids_and_has_a_unique_name(self):
        known = {s.item.id for s in self.corpus}
        self.assertEqual(len({c["id"] for c in self.cases}), len(self.cases))
        for c in self.cases:
            self.assertTrue(set(c["expected"]) | set(c["forbidden"]) <= known, c["id"])
            self.assertFalse(set(c["expected"]) & set(c["forbidden"]), c["id"])

    def test_the_dataset_holds_nothing_private_or_real(self):
        with open(H.DATASET, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("synthetic fixture", text)
        self.assertNotIn(os.path.expanduser("~"), text)
        self.assertNotIn("Users/", text)

    def test_metrics_are_deterministic(self):
        self.assertEqual(json.dumps(self.report, sort_keys=True), json.dumps(H.evaluate(), sort_keys=True))

    def test_the_expected_relevant_items_are_found_and_no_forbidden_one_is_returned(self):
        self.assertEqual(self.report["violations"], 0)
        for row in self.report["rows"]:
            self.assertEqual(sorted(row["returned"]), sorted(row["expected"]), row["id"])
        self.assertEqual((self.report["precision_at_k"], self.report["recall_at_k"], self.report["mrr"]), (1.0, 1.0, 1.0))

    def test_every_forbidden_item_really_is_a_trap(self):
        """Not vacuous: with the filters removed from the data (everything current, plain, unique), a forbidden item
        would match its query. Proven here by checking that each one is lexically relevant to its case's query."""
        by_id = {s.item.id: s for s in self.corpus}
        traps = 0
        for c in self.cases:
            words = [w for w in R._tokens(R.normalize_query(c["query"])) if w not in R.STOPWORDS]
            for fid in c["forbidden"]:
                e = R._Entry(fid, by_id[fid].item, by_id[fid].evidence, M.key_of(by_id[fid].item))
                if R._lexical(words, e) > 0:
                    traps += 1
        self.assertGreaterEqual(traps, 8)

    def test_budget_use_and_result_counts_are_reported_and_bounded(self):
        self.assertTrue(0.0 <= self.report["avg_budget"] <= 1.0)
        self.assertGreater(self.report["avg_results"], 0.0)
        for row in self.report["rows"]:
            self.assertTrue(0.0 <= row["budget"] <= 1.0)

    def test_metric_definitions_on_a_hand_computed_case(self):
        one = {"id": "t", "category": "x", "query": "where do I live", "now": "2026-10-01T00:00:00Z",
               "expected": [self.corpus[0].item.id, "mem_0000000000ff"], "forbidden": []}
        row = H.run_case(self.corpus, one)
        self.assertEqual((row["precision"], row["recall"], row["rr"]), (1.0, 0.5, 1.0))
        empty = H.run_case(self.corpus, {**one, "query": "weather in paris", "expected": []})
        self.assertEqual((empty["precision"], empty["recall"], empty["rr"]), (1.0, 1.0, None))
        missed = H.run_case(self.corpus, {**one, "query": "weather in paris"})
        self.assertEqual((missed["precision"], missed["recall"], missed["rr"]), (0.0, 0.0, 0.0))

    def test_the_harness_never_touches_the_real_vault(self):
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        H.evaluate()
        self.assertEqual(os.path.exists(real), existed)

    def test_a_corpus_item_that_does_not_parse_is_an_error_not_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump({"corpus": [{"item": {"id": "x"}, "evidence": "e"}], "cases": []}, f)
            with self.assertRaises(ValueError):
                H.load(p)


if __name__ == "__main__":
    unittest.main()
