"""The Memory 2.0 retrieval evaluation harness: run every case in dataset.json through backtalk.memory_retrieval and
report precision, recall and rank. Deterministic and offline: the corpus is synthetic, semantic scores and key hints
come from fixed numbers in the case, and no clock is read.

    python -m backtalk.evals.memory_retrieval.harness

A case lists the ids that SHOULD come back (`expected`) and the ids that must NEVER come back (`forbidden`: stale,
duplicate, redundant or unsafe ones). Metrics, per case then averaged over cases (K is the case's own limit):
    precision@K   returned-and-expected / returned.  Nothing returned: 1.0 if nothing was expected, else 0.0
    recall@K      returned-and-expected / expected.  Nothing expected: 1.0
    MRR           1 / rank of the first expected id returned (0 if none).  Cases that expect nothing are left out
    avg_results   mean number of results
    avg_budget    mean of chars_used / max_chars
    violations    forbidden ids that were returned (must be 0)
"""
import json
import os
from dataclasses import dataclass

from backtalk import memory as M
from backtalk import memory_retrieval as R

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


@dataclass(frozen=True)
class Stored:                       # the shape of memory_store.Stored, without importing the store
    item: object
    evidence: str


def load(path=DATASET):
    """(stored memories, cases). A corpus item that does not parse is an error, never skipped."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    corpus = []
    for rec in data["corpus"]:
        item, errors = M.parse(rec["item"])
        if item is None:
            raise ValueError(f"corpus item does not parse: {errors}")
        corpus.append(Stored(item, rec["evidence"]))
    return tuple(corpus), tuple(data["cases"])


def _provider(spec):
    signals = R.ProviderSignals(semantic=dict(spec.get("semantic", {})), keys=dict(spec.get("keys", {})))
    return lambda query, docs: signals


def run_case(corpus, case):
    providers = (_provider(case["provider"]),) if "provider" in case else ()
    limit, max_chars = case.get("limit", R.DEFAULT_LIMIT), case.get("max_chars", R.DEFAULT_MAX_CHARS)
    result = R.retrieve(case["query"], corpus, now=case.get("now"), context_items=case.get("context", ()),
                        limit=limit, max_chars=max_chars, include_history=case.get("include_history", False),
                        providers=providers)
    got, want = [h.id for h in result.hits], list(case["expected"])
    hit = [i for i in got if i in want]
    precision = len(hit) / len(got) if got else (1.0 if not want else 0.0)
    recall = len(set(hit)) / len(want) if want else 1.0
    first = next((n for n, i in enumerate(got, 1) if i in want), None)
    return {"id": case["id"], "category": case["category"], "returned": got, "expected": want,
            "precision": precision, "recall": recall,
            "rr": None if not want else (1.0 / first if first else 0.0),
            "results": len(got), "budget": result.chars_used / max_chars if max_chars else 0.0,
            "violations": [i for i in got if i in case["forbidden"]]}


def evaluate(path=DATASET):
    corpus, cases = load(path)
    rows = [run_case(corpus, c) for c in cases]
    rr = [r["rr"] for r in rows if r["rr"] is not None]
    n = len(rows)
    return {"cases": n,
            "precision_at_k": sum(r["precision"] for r in rows) / n,
            "recall_at_k": sum(r["recall"] for r in rows) / n,
            "mrr": sum(rr) / len(rr) if rr else 0.0, "mrr_cases": len(rr),
            "avg_results": sum(r["results"] for r in rows) / n,
            "avg_budget": sum(r["budget"] for r in rows) / n,
            "violations": sum(len(r["violations"]) for r in rows), "rows": rows}


def main():
    rep = evaluate()
    print(f"{'case':24} {'P@K':>5} {'R@K':>5} {'RR':>5} {'n':>2}  returned / expected")
    for r in rep["rows"]:
        rr = "  -  " if r["rr"] is None else f"{r['rr']:.2f}"
        flag = "  VIOLATION " + ",".join(r["violations"]) if r["violations"] else ""
        print(f"{r['id']:24} {r['precision']:5.2f} {r['recall']:5.2f} {rr:>5} {r['results']:2}  "
              f"{[i[-2:] for i in r['returned']]} / {[i[-2:] for i in r['expected']]}{flag}")
    print(f"\ncases {rep['cases']}  P@K {rep['precision_at_k']:.3f}  R@K {rep['recall_at_k']:.3f}  "
          f"MRR {rep['mrr']:.3f} (over {rep['mrr_cases']})  avg results {rep['avg_results']:.2f}  "
          f"avg budget use {rep['avg_budget']:.3f}  violations {rep['violations']}")


if __name__ == "__main__":
    main()
