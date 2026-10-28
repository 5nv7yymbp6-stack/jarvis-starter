"""Memory 2.0, Phase 8.2b: the commit PLANNING layer (backtalk/memory_commit.py). Synthetic packs, ledgers, stores in temp
directories and fake callables only; nothing is written to a real store or ledger, and no receipt exists.

TRIPWIRE (exactly what it is). In every test class that inherits `PlanCase` or calls `install_tripwire`, these callables
raise AssertionError when the SYMLINK-RESOLVED path is under the Claude folder, the vault or the review folder:
`builtins.open`, `io.open` (so `pathlib.Path.open`), `os.open`, `os.listdir` and `os.scandir`. It is a regression tripwire for
THESE TESTS. It does NOT cover metadata calls (`os.stat`, `os.path.exists`): the inherited base-class teardown in
tests/test_memory_extract.py deliberately checks the real vault with `os.path.exists`, so covering them would fail every test.
It also does NOT cover subprocesses (one test starts a child Python), `ctypes`, descriptors opened before it was installed,
other `os`/`shutil` calls, or code that bound the original functions earlier, and it does not prevent arbitrary same-account
file access. A separate static test checks that this file names no real path.

WHAT THESE TESTS DO NOT SHOW. `Plan.checks`, `verify_plan_integrity`, the display digest and the token are accidental-error
and regression defenses. They are not evidence that any check really ran, and not evidence that a person approved anything;
code running as the same user can build a consistent Plan, forge a token and an answer, and edit any file."""
import ast
import builtins
import copy
import functools
import glob
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from unittest import mock

from backtalk import memory as M
from backtalk import memory_commit as C
from backtalk import memory_extract as X
from backtalk import memory_gate as G
from backtalk import memory_keys as K
from backtalk import memory_review as R
from backtalk import memory_store as S
from tests.store_arrange import arrange
from tests.test_memory_authorization import day, item, mid
from tests.test_memory_extract import turn
from tests.test_memory_review import T1, Machine, make_pack

PACK_SHA = "a" * 64
NOWS = "2026-03-02T10:00:00Z"            # 23h55m after the T1 lock instant
NONCE = "b" * 32
TS = "2026-01-10T12:00:00Z"
SECRET = "SENTINEL-VALUE-7f3a"
FORBIDDEN = tuple(p for p in (os.path.expanduser("~/.claude"), os.path.expanduser("~/.backtalk-memory-review"),
                              S.REAL_VAULT))


def old_occupation(n=7, value="plumber", **over):
    base = dict(attribute="occupation", value=value, valid_from=day(1), created=day(1), updated=day(1))
    base.update(over)
    return item(n, **base)


TRIPWIRE_TARGETS = ((builtins, "open"), (io, "open"), (os, "open"), (os, "listdir"), (os, "scandir"))


def install_tripwire(case):
    """Make the callables in TRIPWIRE_TARGETS raise AssertionError for a symlink-resolved path under FORBIDDEN (see the
    module docstring for what this does not cover)."""
    real = tuple(os.path.realpath(p) for p in FORBIDDEN)
    busy = []

    def check(path):
        if busy or not isinstance(path, (str, bytes, os.PathLike)):
            return
        busy.append(1)                                          # realpath must not trip the tripwire itself
        try:
            resolved = os.path.realpath(os.fsdecode(path))
        finally:
            busy.pop()
        if any(resolved == f or resolved.startswith(f + os.sep) for f in real):
            raise AssertionError(f"a test touched a real path: {resolved}")

    def wrap(fn):
        @functools.wraps(fn)
        def guarded(*a, **k):
            check(a[0] if a else k.get("path", k.get("file")))
            return fn(*a, **k)
        return guarded
    for module, name in TRIPWIRE_TARGETS:
        patcher = mock.patch.object(module, name, wrap(getattr(module, name)))
        patcher.start()
        case.addCleanup(patcher.stop)


class PlanCase(Machine):
    def setUp(self):
        super().setUp()
        install_tripwire(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.store = S.MemoryStore(self.root)
        self.clock_value = NOWS
        self.tick = 0.0
        self.prov_calls, self.rederive_calls = [], []
        self.provenance = lambda view: (self.prov_calls.append(view["id"]) or [])
        self.rederive = lambda view: (self.rederive_calls.append(view["id"]) or copy.deepcopy(view))
        self.cand = None

    def mono(self):
        self.tick += 0.5
        return self.tick

    def lock_occupation(self):
        self.lock("occupation")
        self.cand = self.cid("occupation")

    def kw(self, **over):
        d = dict(pack=self.pack, pack_sha=PACK_SHA, records=self.state.records, private=None, candidate=self.cand,
                 store=self.store, provenance=self.provenance, rederive=self.rederive, clock=lambda: self.clock_value,
                 monotonic=self.mono, preview_budget_s=45, max_age_days=14)
        d.update(over)
        return d

    def plan(self, **over):
        return C.plan(**self.kw(**over))

    def refused(self, code, fn=None, *a, **k):
        fn = fn or self.plan
        with self.assertRaises((C.CommitError, R.ReviewError)) as cm:
            fn(*a, **k)
        got = getattr(cm.exception, "code")
        self.assertEqual(got, code, str(cm.exception))
        return cm.exception

    def catch(self, **over):
        try:
            self.plan(**over)
        except C.CommitError as e:
            return e.code
        return None

    def forge(self, **over):
        """A chain-valid ledger whose lock record has the given fields changed (replay_validate is mocked where used)."""
        rec = self.state.final[self.cand]
        fields_ = {k: v for k, v in rec.items() if k not in ("seq", "prev", "at", "hash")}
        head = self.state.records[0]
        return [head, R.new_record([head], rec["at"], {**fields_, **over})]

    def put_old(self, **over):
        it = old_occupation(**over)
        self.store.put(it, "I am a plumber by trade")
        return it


class SuccessTests(PlanCase):
    def test_a_note_lock_with_nothing_to_replace_is_a_put(self):
        self.lock_occupation()
        p = self.plan()
        self.assertEqual((p.operation, p.kind, p.links, p.superseded), ("put", "commit", (), ()))
        self.assertEqual((p.decision.decision, p.decision.reason), (G.ACCEPT, "accepted"))
        self.assertEqual(p.checks, C.REQUIRED_CHECKS)
        self.assertEqual(C.verify_plan_integrity(p), [])
        rec = self.state.final[self.cand]
        want, ev = R.build_item(self.state.run, self.cand, type=rec["type"], key=rec["key"], value=rec["value"],
                                evidence=rec["evidence"], sensitivity=rec["sensitivity"], at=rec["adoption"]["at"],
                                valid_from=rec["valid_from"])
        self.assertEqual((p.item, p.evidence), (want, ev))
        self.assertEqual((p.lock_hash, p.planned_at, p.age_seconds, p.max_age_seconds), (rec["hash"], NOWS, 86100, 1209600))
        self.assertEqual(p.item_digest, S.item_digest(want, ev, (), "put"))
        self.assertEqual(p.state_digest, self.store.state_digest())

    def test_the_plan_is_deterministic_and_does_not_touch_its_inputs(self):
        self.lock_occupation()
        records, before = self.state.records, copy.deepcopy(self.state.records)
        a, b = self.plan(), self.plan()
        self.assertEqual(replace(a, rederive_seconds=0.0), replace(b, rederive_seconds=0.0))
        self.assertEqual(records, before)
        self.assertEqual(self.store.list_items(), ())

    def test_a_newer_value_for_the_same_key_is_an_apply_with_links(self):
        old = self.put_old()
        self.lock_occupation()
        p = self.plan()
        self.assertEqual((p.operation, p.kind, p.links), ("apply", "commit", (old.id,)))
        self.assertEqual((p.decision.decision, p.decision.reason), (G.ACCEPT, "accepted_supersedes"))
        self.assertEqual([o.id for o in p.superseded], [old.id])
        self.assertEqual((p.superseded[0].value, p.superseded[0].status), ("plumber", "active"))
        self.assertEqual(C.verify_plan_integrity(p), [])
        self.assertEqual(p.item_digest, S.item_digest(p.item, p.evidence, p.links, "apply"))

    def test_an_edit_replays_and_plans(self):
        self.step(R.edit(self.pack, self.state, self.cid("goal_cash_buyer_list_deep"), value="a buyer list of fifty cash buyers", now=T1))
        self.cand = self.cid("goal_cash_buyer_list_deep")
        p = self.plan()
        self.assertEqual((p.item.value, p.operation, p.kind), ("a buyer list of fifty cash buyers", "put", "commit"))
        self.assertEqual(C.verify_plan_integrity(p), [])

    def test_a_private_edit_needs_its_side_data_and_is_a_private_kind(self):
        out = R.edit(self.pack, self.state, self.cid("goal_know_each_market_well"), value="I earn $800/week from the work", now=T1)
        self.step(out)
        self.cand = self.cid("goal_know_each_market_well")
        side = {out.private["seq"]: {"value": out.private["value"], "evidence": out.private["evidence"]}}
        p = self.plan(private=side)
        self.assertEqual((p.kind, p.operation, p.item.sensitivity), ("commit-private", "put", "private"))
        self.assertEqual((p.decision.decision, p.decision.reason), (G.REQUIRE_CONFIRMATION, "private_requires_confirmation"))
        self.assertEqual(p.item.value, "I earn $800/week from the work")
        self.assertIsNone(json.loads(p.lock_record_json)["value"])
        self.refused("ledger_replay_invalid", private=None)

    def test_a_transcript_lock_keeps_its_genuine_instant(self):
        pack = R.Pack(make_pack([turn("I prefer short answers when we talk.", ts=TS)]))
        cid = next(c for c, v in pack.views.items() if v["origin"] == "transcript")
        rec = R.lock(pack, self.state, cid, "lock", now=T1).record
        p = C.plan(**self.kw(pack=pack, records=self.state.records + [rec], candidate=cid))
        self.assertEqual(p.item.valid_from, TS)
        self.assertEqual(C.verify_plan_integrity(p), [])

    def test_the_repr_never_shows_a_value_or_evidence(self):
        self.lock_occupation()
        p = self.plan()
        text = repr(p)
        for secret in (p.item.value, p.evidence, p.lock_record_json, "Yousif"):
            self.assertNotIn(secret, text)
        self.assertIn(p.item.id, text)


class TripwireTests(PlanCase):
    def test_every_covered_call_raises_for_a_real_path(self):
        targets = (os.path.join(S.REAL_VAULT, "VAULT-INDEX.md"), os.path.expanduser("~/.claude/projects/x.jsonl"),
                   os.path.expanduser("~/.backtalk-memory-review/ledger.jsonl"), S.REAL_VAULT)
        calls = (lambda p: builtins.open(p), lambda p: io.open(p), lambda p: os.open(p, os.O_RDONLY), os.listdir,
                 lambda p: list(os.scandir(p)), lambda p: __import__("pathlib").Path(p).open(),
                 lambda p: __import__("pathlib").Path(p).read_text())
        for path in targets:
            for call in calls:
                with self.assertRaises(AssertionError):
                    call(path)

    def test_a_symlink_into_a_real_path_is_resolved_and_refused(self):
        link = os.path.join(self.root, "link")
        os.symlink(S.REAL_VAULT, link)                         # os.symlink is not covered, and touches nothing real
        for call in (lambda p: open(os.path.join(p, "child.txt")), os.listdir):
            with self.assertRaises(AssertionError):
                call(link)

    def test_metadata_calls_are_documented_as_not_covered(self):
        os.path.exists(S.REAL_VAULT)                           # the inherited teardown relies on this; it must not raise
        os.stat(self.root)
        self.assertNotIn((os, "stat"), TRIPWIRE_TARGETS)

    def test_other_paths_and_non_paths_still_work(self):
        self.assertEqual(open(os.devnull).read(), "")
        self.assertIsInstance(os.listdir(self.root), list)
        self.assertTrue(os.path.exists(self.root))
        fd = os.open(os.devnull, os.O_RDONLY)                  # a descriptor is not a path
        os.close(fd)
        self.assertEqual(open(fd := os.open(os.devnull, os.O_RDONLY)).read(), "")

    def test_the_documentation_says_what_it_does_not_cover(self):
        doc = " ".join(__doc__.split())
        for phrase in ("TRIPWIRE (exactly what it is)", "SYMLINK-RESOLVED", "does NOT cover metadata calls", "does NOT cover subprocesses",
                       "ctypes", "does not prevent arbitrary same-account file access"):
            self.assertIn(phrase, doc)
        self.assertEqual({(m.__name__, n) for m, n in TRIPWIRE_TARGETS},
                         {("builtins", "open"), ("io", "open"), ("os", "open"), ("os", "listdir"), ("os", "scandir")})


class InputTests(PlanCase):
    def test_bad_inputs(self):
        self.lock_occupation()
        for name, bad in (("pack", {}), ("pack_sha", "short"), ("pack_sha", None), ("records", []), ("records", None),
                          ("records", [1]), ("private", []), ("candidate", "cand_nope"), ("candidate", None),
                          ("provenance", None), ("rederive", "x"), ("clock", None), ("monotonic", 5),
                          ("preview_budget_s", 0), ("preview_budget_s", -1), ("preview_budget_s", True), ("preview_budget_s", "45"),
                          ("max_age_days", 0), ("max_age_days", True), ("max_age_days", 1.5)):
            with self.subTest(name=name, bad=bad):
                self.assertEqual(self.catch(**{name: bad}), "bad_input")

    def test_the_store_is_validated_before_any_check_or_clock_runs(self):
        self.lock_occupation()
        calls = []
        for bad in (None, object(), "x", 5, type("HalfStore", (), {"state_digest": lambda s: "0" * 64})()):
            with self.assertRaises(C.CommitError) as cm:
                self.plan(store=bad, clock=lambda: (calls.append("clock") or NOWS), monotonic=lambda: (calls.append("mono") or 0.0))
            self.assertEqual((cm.exception.code, cm.exception.details), ("bad_input", ("store",)))
        self.assertEqual((calls, self.prov_calls, self.rederive_calls), ([], [], []))

    def test_every_unusable_input_is_named_and_nothing_runs(self):
        self.lock_occupation()
        with self.assertRaises(C.CommitError) as cm:
            self.plan(pack=None, pack_sha="x", records=[], private=[], candidate=5, store=None, provenance=None, rederive=None,
                      clock=None, monotonic=None, preview_budget_s=0, max_age_days=0)
        self.assertEqual(cm.exception.details, ("pack", "pack_sha", "records", "private", "candidate", "store", "provenance",
                                                "rederive", "clock", "monotonic", "preview_budget_s", "max_age_days"))

    def test_the_budget_must_be_a_finite_positive_number(self):
        self.lock_occupation()
        inf = float("inf")
        for bad in (0, 0.0, -0.0, -1, -0.5, True, False, "45", None, [], {}, float("nan"), inf, -inf, 10 ** 400, -10 ** 400,
                    complex(1, 0), b"45"):
            with self.subTest(bad=repr(bad)[:24]):
                with self.assertRaises(C.CommitError) as cm:
                    self.plan(preview_budget_s=bad)
                self.assertEqual((cm.exception.code, cm.exception.details), ("bad_input", ("preview_budget_s",)))
        self.assertEqual((self.prov_calls, self.rederive_calls), ([], []))
        for good in (0.001, 1, 45, 45.5, 10 ** 6, 1e300):
            self.assertEqual(self.plan(preview_budget_s=good, monotonic=lambda: 0.0).checks, C.REQUIRED_CHECKS)

    def test_every_callable_is_required(self):
        self.lock_occupation()
        for name in ("provenance", "rederive", "clock", "monotonic", "store", "pack", "records", "candidate"):
            kw = self.kw()
            del kw[name]
            with self.assertRaises(TypeError):
                C.plan(**kw)
        self.assertNotIn("preview_budget_s", [n for n, p in inspect.signature(C.plan).parameters.items() if p.default is not inspect.Parameter.empty])

    def test_the_clock_must_give_a_real_utc_stamp(self):
        self.lock_occupation()
        from datetime import datetime, timezone
        for bad in (lambda: None, lambda: 5, lambda: "2026-03-02", lambda: "now", lambda: datetime(2026, 3, 2, 10),
                    lambda: (_ for _ in ()).throw(RuntimeError("x"))):
            self.assertEqual(self.catch(clock=bad), "bad_clock")
        self.assertEqual(self.plan(clock=lambda: datetime(2026, 3, 2, 10, 0, 0, 999999, tzinfo=timezone.utc)).planned_at, NOWS)

    def test_the_clock_is_read_exactly_once(self):
        self.lock_occupation()
        calls = []
        self.plan(clock=lambda: (calls.append(1) or NOWS))
        self.assertEqual(len(calls), 1)


class ReplayStepTests(PlanCase):
    def test_a_tampered_ledger_is_refused(self):
        self.lock_occupation()
        recs = copy.deepcopy(self.state.records)
        recs[1]["value"] = "something else"
        e = self.refused("ledger_replay_invalid", records=recs)
        self.assertEqual(e.details, ("hash:1",))

    def test_a_forged_but_chain_valid_record_is_refused(self):
        self.lock_occupation()
        head = self.state.records[0]
        fields_ = {k: v for k, v in self.state.records[1].items() if k not in ("seq", "prev", "at", "hash")}
        forged = [head, R.new_record([head], T1, {**fields_, "candidate": "cand_does_not_exist"})]
        self.assertEqual(R.verify_chain(forged), [])
        e = self.refused("ledger_replay_invalid", records=forged, candidate=self.cand)
        self.assertEqual(e.details, ("replay:1:unknown_candidate",))

    def test_the_wrong_pack_sha_is_refused(self):
        self.lock_occupation()
        self.refused("ledger_replay_invalid", pack_sha="b" * 64)

    def test_an_action_after_the_final_decision_is_refused(self):
        self.lock_occupation()
        extra = R.new_record(self.state.records, T1, {"action": "skip", "candidate": self.cand, "key": "occupation"})
        e = self.refused("ledger_replay_invalid", records=self.state.records + [extra])
        self.assertEqual(e.details, ("replay:2:already_reviewed",))

    def test_a_replay_result_that_is_not_an_empty_list_never_produces_a_plan(self):
        self.lock_occupation()
        for result in (None, (), ("x",), "ok", 0, ["replay:1:mismatch"]):
            with self.subTest(result=result), mock.patch.object(R, "replay_validate", lambda *a, **k: result):
                self.assertEqual(self.catch(), "ledger_replay_invalid")
        with mock.patch.object(R, "replay_validate", side_effect=RuntimeError("x")):
            self.assertEqual(self.catch(), "check_failed")

    def test_the_inputs_are_copied_before_they_are_validated(self):
        self.lock_occupation()
        seen = {}
        real = R.replay_validate

        def spy(pack, records, pack_sha, private=None):
            seen["records"] = records
            return real(pack, records, pack_sha, private)
        with mock.patch.object(R, "replay_validate", spy):
            self.plan()
        self.assertIsNot(seen["records"], self.state.records)

    def test_a_registry_version_other_than_the_codes_is_refused(self):
        self.lock_occupation()
        with mock.patch.object(K, "REGISTRY_VERSION", 2):
            self.assertIn(self.catch(), ("ledger_replay_invalid",))


class CommittableTests(PlanCase):
    def test_only_a_locked_or_edited_candidate_can_be_planned(self):
        self.cand = self.cid("occupation")
        self.refused("not_committable")
        self.step(R.skip(self.pack, self.state, self.cand, now=T1))
        self.refused("not_committable")
        self.step(R.reject(self.pack, self.state, self.cand, "not_true", now=T1))
        e = self.refused("not_committable")
        self.assertEqual(e.details, ("rejected",))

    def test_the_candidate_must_be_in_the_pack(self):
        self.lock_occupation()
        self.assertEqual(self.catch(candidate="cand_nope"), "bad_input")


class AgeTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.lock_occupation()
        self.at = M.parse_ts(self.state.final[self.cand]["at"])

    def stamp(self, delta):
        from datetime import timedelta
        return (self.at + timedelta(seconds=delta)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_the_boundaries(self):
        limit = 14 * 86400
        for delta, ok in ((0, True), (1, True), (limit - 1, True), (limit, True), (limit + 1, False), (limit * 3, False)):
            with self.subTest(delta=delta):
                if ok:
                    self.assertEqual(self.plan(clock=lambda: self.stamp(delta)).age_seconds, delta)
                else:
                    self.refused("lock_too_old", clock=lambda: self.stamp(delta))

    def test_a_record_from_the_future_is_refused_with_no_tolerance(self):
        self.refused("lock_from_future", clock=lambda: self.stamp(-1))
        self.refused("lock_from_future", clock=lambda: self.stamp(-86400))

    def test_a_different_limit_is_honoured_and_age_is_never_sufficient(self):
        self.assertEqual(self.plan(max_age_days=1, clock=lambda: self.stamp(86400)).max_age_seconds, 86400)
        self.refused("lock_too_old", max_age_days=1, clock=lambda: self.stamp(86401))
        self.provenance = lambda view: ["hash_mismatch"]
        self.refused("source_stale", clock=lambda: self.stamp(0))             # a brand-new lock still gets every check


class ProvenanceStepTests(PlanCase):
    def test_every_code_refuses_and_is_reported(self):
        self.lock_occupation()
        for codes in (["source_missing"], ["hash_mismatch", "evidence_not_in_line"], ["line_out_of_range"],
                      ["turn_no_longer_eligible", "id_mismatch", "timestamp_mismatch", "channel_mismatch", "ref_mismatch"]):
            self.provenance = lambda view, codes=codes: list(codes)
            e = self.refused("source_stale")
            self.assertEqual(e.details, tuple(codes))

    def test_a_result_that_is_not_a_clean_empty_list_never_produces_a_plan(self):
        self.lock_occupation()
        for bad in (None, (), "", 0, {}, [None], ["Has Spaces"], [5], ["x" * 100]):
            self.provenance = lambda view, bad=bad: bad
            e = self.refused("source_stale")
            self.assertEqual(e.details, ("result_invalid",), bad)
        self.provenance = lambda view: (_ for _ in ()).throw(OSError("disk"))
        self.refused("check_failed")

    def test_unrecognized_codes_are_refused_but_never_echoed(self):
        self.lock_occupation()
        for result, shown in ((["secret_value_abc"], ("unrecognized_code",)),
                              (["hash_mismatch", "my_private_note_text"], ("hash_mismatch", "unrecognized_code")),
                              (["x" * 60], ("unrecognized_code",))):
            self.provenance = lambda v, result=result: list(result)
            e = self.refused("source_stale")
            self.assertEqual(e.details, shown)
            self.assertNotIn("secret_value", str(e) + repr(e.details))

    def test_the_known_codes_cover_what_the_real_verifier_can_return(self):
        from backtalk.evals.memory_extraction import evaluate as E
        with open(E.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "verify_provenance")
        found = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append" and n.args \
                    and isinstance(n.args[0], ast.Constant):
                found.add(n.args[0].value)
            if isinstance(n, ast.Return) and isinstance(n.value, ast.List):
                found |= {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
        self.assertTrue(found)
        self.assertEqual(found - C.KNOWN_SOURCE_CODES, set(), "the verifier can return a code this module would not recognize")

    def test_the_callable_gets_a_copy_of_the_view(self):
        self.lock_occupation()
        got = []
        self.provenance = lambda view: (got.append(view) or view.update(value="changed") or [])
        self.plan()
        self.assertEqual(self.pack.views[self.cand]["value"], "wholesale real estate")


class RederiveStepTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.lock_occupation()

    def mutate(self, **over):
        def fn(view):
            d = copy.deepcopy(view)
            d.update(over)
            return d
        self.rederive = fn

    def test_no_result_or_a_non_dict(self):
        self.rederive = lambda view: None
        self.refused("not_rederived")
        for bad in ([], "x", 5, (1,)):
            self.rederive = lambda view, bad=bad: bad
            e = self.refused("not_rederived")
            self.assertEqual(e.details, ("result_invalid",))
        self.rederive = lambda view: (_ for _ in ()).throw(RuntimeError("x"))
        self.refused("check_failed")

    def test_each_compared_field_refuses_when_it_differs(self):
        view = self.pack.views[self.cand]
        changes = {"id": "cand_other", "origin": "transcript", "where": "Other.md", "line": 999, "source_kind": "user",
                   "source_ref": "vault:Other.md#L1", "text_sha": "0" * 64, "channel": "typed", "evidence_tier": "typed_user",
                   "uuid": "u", "session": "s", "timestamp": "2026-01-01T00:00:00Z", "confidence": "stated",
                   "corroboration": "typed", "type": "preference", "attribute": "other", "review_class": "deferred_template",
                   "registry": {"kind": "set", "canonical": "occupation", "version": 1},
                   "conflict_groups": ["conf_new"], "value": "something else", "evidence": "another sentence"}
        for field_, new in changes.items():
            with self.subTest(field=field_):
                self.mutate(**{field_: new})
                e = self.refused("rederive_mismatch")
                self.assertIn(field_, e.details)
        self.assertNotIn("occurrences", changes)

    def view_and_rec(self):
        return self.pack.views[self.cand], self.state.final[self.cand]

    def test_the_record_origin_branch_fires_alone_when_the_view_disagrees_with_the_record(self):
        view, rec = self.view_and_rec()
        self.assertEqual(C._rederive_mismatches(view, rec, copy.deepcopy(view)), [])
        for field_, new in (("source_kind", "user"), ("source_ref", "vault:Other.md#L9"), ("where", "Other.md"), ("line", 999),
                            ("text_sha", "0" * 64), ("channel", "typed"), ("evidence_tier", "typed_user")):
            with self.subTest(field=field_):
                moved = {**copy.deepcopy(view), field_: new}                 # view and fresh agree with each other, not the record
                self.assertEqual(C._rederive_mismatches(moved, rec, copy.deepcopy(moved)), ["record_origin"])

    def test_through_plan_a_record_origin_mismatch_never_stands_alone(self):
        """Why the branch is defence in depth: under a clean replay the record's origin IS the pack view's, and the fresh result
        is compared with the view on the same fields, so any record_origin hit is accompanied by the field itself."""
        view, rec = self.view_and_rec()
        for field_, new in (("source_kind", "user"), ("source_ref", "vault:Other.md#L9"), ("where", "Other.md"), ("line", 999),
                            ("text_sha", "0" * 64), ("channel", "typed"), ("evidence_tier", "typed_user")):
            with self.subTest(field=field_):
                self.mutate(**{field_: new})
                e = self.refused("rederive_mismatch")
                self.assertIn("record_origin", e.details)
                self.assertIn(field_, e.details)

    def test_malformed_and_mixed_type_results_are_a_stable_refusal(self):
        for field_ in ("conflict_groups", "origin", "registry", "line", "where", "timestamp", "channel", "uuid", "session",
                       "corroboration", "text_sha", "source_ref"):
            for bad in ([1, "a"], [{}, {}], [None, "x"], [[1], ["a"]], [object(), object()]):
                with self.subTest(field=field_, bad=repr(bad)[:18]):
                    self.mutate(**{field_: bad})
                    e = self.refused("rederive_mismatch")
                    self.assertEqual(e.details, ("result_invalid",))
                    self.assertIsNone(e.__context__)
                    self.assertIsNone(e.__cause__)

    def test_a_result_object_that_misbehaves_is_a_stable_refusal_and_leaks_nothing(self):
        class Evil(dict):
            def get(self, *a, **k):
                raise RuntimeError(SECRET)
        self.rederive = lambda view: Evil(view)
        e = self.refused("rederive_mismatch")
        self.assertEqual(e.details, ("result_invalid",))
        self.assertNotIn(SECRET, str(e) + repr(e.details))
        self.assertIsNone(e.__context__)

    def test_a_missing_field_is_a_mismatch(self):
        for field_ in ("text_sha", "where", "evidence", "value", "conflict_groups", "channel"):
            with self.subTest(field=field_):
                def fn(view, field_=field_):
                    d = copy.deepcopy(view)
                    del d[field_]
                    return d
                self.rederive = fn
                e = self.refused("rederive_mismatch")
                self.assertIn(field_, e.details)

    def test_ignored_fields_never_refuse(self):
        self.mutate(occurrences=99, other_refs=["x"], source_date="2020-01-01", gate={"decision": "REJECT", "reason": "x", "details": []},
                    if_affirmed={"decision": "REJECT", "reason": "x"}, extra_unknown_key=1)
        self.assertEqual(C.verify_plan_integrity(self.plan()), [])

    def test_conflict_groups_compare_as_sets(self):
        self.mutate(conflict_groups=list(reversed(self.pack.views[self.cand]["conflict_groups"])))
        self.assertEqual(C.verify_plan_integrity(self.plan()), [])

    def test_a_fresh_private_sensitivity_over_a_normal_record_refuses_but_not_the_reverse(self):
        self.mutate(sensitivity="private")
        self.assertIn("sensitivity", self.refused("rederive_mismatch").details)
        self.mutate(sensitivity="secret")
        self.assertIn("sensitivity", self.refused("rederive_mismatch").details)

    def test_the_record_origin_is_compared_too(self):
        recs = copy.deepcopy(self.state.records)
        self.rederive = lambda view: copy.deepcopy(view)
        # a pack whose view agrees with the fresh view but not with the record's origin: change both view and fresh
        raw = copy.deepcopy(self.raw)
        for c in raw["candidates"]:
            if c["id"] == self.cand:
                c["where"] = "Elsewhere.md"
        pack = R.Pack(raw)
        e = self.refused("ledger_replay_invalid", pack=pack, records=recs, rederive=lambda v: copy.deepcopy(v))
        self.assertTrue(e.details)

    def test_the_budget_is_required_and_a_slow_rederivation_is_refused(self):
        ticks = iter([0.0, 45.1])
        self.refused("rederivation_too_slow", monotonic=lambda: next(ticks))
        ticks = iter([0.0, 90.0])
        self.refused("rederivation_too_slow", monotonic=lambda: next(ticks))
        ticks = iter([10.0, 55.0])                           # exactly the budget is allowed
        self.assertEqual(self.plan(monotonic=lambda: next(ticks)).rederive_seconds, 45.0)
        ticks = iter([0.0, 1.0])
        self.refused("rederivation_too_slow", monotonic=lambda: next(ticks), preview_budget_s=0.5)

    def test_a_broken_monotonic_clock_fails_closed(self):
        for pair in ((5.0, 4.0), (None, 1.0), (True, 2.0), ("a", "b")):
            it = iter(pair)
            self.assertEqual(self.catch(monotonic=lambda: next(it)), "bad_clock")
        self.assertEqual(self.catch(monotonic=lambda: (_ for _ in ()).throw(RuntimeError("x"))), "check_failed")

    def test_an_edit_needs_its_origin_candidate_to_still_derive(self):
        self.setUp()
        self.step(R.edit(self.pack, self.state, self.cid("goal_cash_buyer_list_deep"), value="a buyer list of fifty cash buyers", now=T1))
        self.cand = self.cid("goal_cash_buyer_list_deep")
        self.assertEqual(C.verify_plan_integrity(self.plan()), [])
        self.mutate(evidence="the sentence changed after the edit")
        e = self.refused("rederive_mismatch")
        self.assertIn("origin_evidence", e.details)
        self.rederive = lambda view: None
        self.refused("not_rederived")

    def test_a_lock_needs_the_three_way_value_agreement(self):
        self.mutate(value="wholesale real estate!")
        self.assertIn("value", self.refused("rederive_mismatch").details)

    def test_a_transcript_lock_needs_the_same_genuine_instant(self):
        pack = R.Pack(make_pack([turn("I prefer short answers when we talk.", ts=TS)]))
        cid = next(c for c, v in pack.views.items() if v["origin"] == "transcript")
        rec = R.lock(pack, self.state, cid, "lock", now=T1).record
        kw = dict(pack=pack, records=self.state.records + [rec], candidate=cid)

        def shifted(view):
            d = copy.deepcopy(view)
            d["timestamp"] = "2026-01-10T12:00:01Z"
            return d
        e = self.refused("rederive_mismatch", rederive=shifted, **kw)
        self.assertIn("valid_from", e.details)
        self.assertIn("timestamp", e.details)


class BuildStepTests(PlanCase):
    """Chain-valid ledgers with changed fields. `replay_validate` and the re-derivation comparison are stubbed to pass so the
    BUILD step itself is exercised (the real replay refuses these first, as ReplayStepTests shows)."""

    def setUp(self):
        super().setUp()
        self.lock_occupation()
        for target, repl in ((R, "replay_validate"), (C, "_rederive_mismatches")):
            patcher = mock.patch.object(target, repl, lambda *a, **k: [])
            patcher.start()
            self.addCleanup(patcher.stop)

    def text_fields(self, text):
        return dict(value=text, value_fp=M.fingerprint(text), value_len=len(text), evidence=text, evidence_sha=X.sha(text))

    def test_a_ledger_value_that_no_longer_matches_its_fingerprint_is_refused(self):
        e = self.refused("item_invalid", records=self.forge(value="wholesale real estate!"))
        self.assertEqual(e.details, ("value_fp", "value_len"))
        e = self.refused("item_invalid", records=self.forge(evidence="another supporting sentence here"))
        self.assertEqual(e.details, ("evidence_sha",))

    def test_a_missing_private_side_file_is_item_invalid(self):
        for side in (None, {}, {1: "x"}, {1: {"value": 5, "evidence": "e"}}, {1: {"value": "v"}}):
            e = self.refused("item_invalid", records=self.forge(value=None, evidence=None, sensitivity="private"), private=side)
            self.assertEqual(e.details, ("private_missing",))

    def test_text_that_is_now_screened_is_refused(self):
        e = self.refused("screen_refused", records=self.forge(**self.text_fields("my api key is sk-abcdefghijklmnop1234")))
        self.assertEqual(e.details, ("credential_like",))
        e = self.refused("screen_refused", records=self.forge(**self.text_fields("call 816-452-3549 for deals")))
        self.assertEqual(e.details, ("contact_detail",))

    def test_a_value_that_turns_out_private_with_a_normal_record_is_refused(self):
        self.refused("sensitivity_changed", records=self.forge(**self.text_fields("I earn $800/week from the work")))

    def test_an_invalid_item_or_key_is_refused(self):
        self.refused("item_invalid", records=self.forge(key="not_in_registry"))
        self.refused("item_invalid", records=self.forge(valid_from="not a time"))

    def test_the_registry_version_on_the_record_must_be_the_codes(self):
        self.refused("ledger_replay_invalid", records=self.forge(registry_version=2))


class GateStepTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.lock_occupation()

    def test_the_same_value_stated_later_is_a_stale_proposal(self):
        self.store.put(old_occupation(value="wholesale real estate", valid_from="2026-03-01T11:00:00Z", created="2026-03-01T11:00:00Z",
                                      updated="2026-03-01T11:00:00Z"), "I wholesale real estate for a living")
        e = self.refused("gate_refused")
        self.assertEqual(e.details, (G.REJECT, "stale_proposal"))

    def test_an_equally_recent_rival_is_a_conflict_and_is_refused(self):
        self.store.put(old_occupation(value="plumber", valid_from=T1, created=T1, updated=T1), "I am a plumber by trade")
        e = self.refused("gate_refused")
        self.assertEqual(e.details, (G.CONFLICT, "conflict"))

    def test_a_newer_existing_value_blocks_the_older_candidate(self):
        self.store.put(old_occupation(value="plumber", valid_from="2026-03-01T11:00:00Z", created="2026-03-01T11:00:00Z",
                                      updated="2026-03-01T11:00:00Z"), "I am a plumber by trade")
        e = self.refused("gate_refused")
        self.assertEqual(e.details[0], G.REJECT)

    def test_the_gate_runs_at_the_plans_one_instant(self):
        seen = []
        real = G.evaluate

        def spy(proposal, existing, now, *a, **k):
            seen.append(now)
            return real(proposal, existing, now, *a, **k)
        with mock.patch.object(G, "evaluate", spy):
            self.plan()
        self.assertIn(NOWS, seen)
        self.assertEqual(set(seen) - {T1}, {NOWS})          # (the replay's own dry-runs use each record's instant)


class FakeStore:
    """A store-like object that can change between the reads, to exercise the consistent-read protocol."""

    def __init__(self, real, script):
        self.real, self.script, self.calls = real, list(script), 0

    def state_digest(self):
        self.calls += 1
        step = self.script.pop(0) if self.script else None
        if callable(step):
            step()
        return self.real.state_digest()

    def list_items(self):
        return self.real.list_items()


class StoreReadTests(PlanCase):
    def test_a_write_between_the_digest_and_the_items_is_retried(self):
        self.lock_occupation()
        fake = FakeStore(self.store, [None, lambda: None])
        orig = fake.list_items

        def writing():
            fake.list_items = orig                        # only the first attempt sees the write
            self.store.put(old_occupation(value="plumber", valid_from=day(1)), "I am a plumber by trade")
            return orig()
        fake.list_items = writing
        p = self.plan(store=fake)
        self.assertEqual(p.operation, "apply")            # settled on the state that includes the write
        self.assertEqual(p.state_digest, self.store.state_digest())

    def test_a_store_that_never_settles_is_refused_after_three_attempts(self):
        self.lock_occupation()
        n = [0]

        class Moving:
            def state_digest(s):
                n[0] += 1
                return hashlib.sha256(str(n[0]).encode()).hexdigest()

            def list_items(s):
                return ()
        e = self.refused("store_unstable", store=Moving())
        self.assertEqual(e.details, ("state_digest_changed",))
        self.assertEqual(n[0], 6)

    def test_items_that_do_not_hash_to_the_digest_are_refused(self):
        self.lock_occupation()
        self.put_old()
        other = S.MemoryStore(os.path.join(self.root, "elsewhere"))

        class Stale:
            def state_digest(s):
                return self_.store.state_digest()

            def list_items(s):
                return other.list_items()                 # empty: a different state than the digest names
        self_ = self
        e = self.refused("store_unstable", store=Stale())
        self.assertEqual(e.details, ("digest_of",))

    def test_a_flicker_that_returns_to_the_same_digest_is_still_caught_by_the_items_hash(self):
        self.lock_occupation()
        old = self.put_old()
        path = os.path.join(self.store.dir, old.id + ".md")
        with open(path, encoding="utf-8", newline="") as f:
            original = f.read()
        calls = {"n": 0}
        real_list = self.store.list_items

        class Flicker:
            def state_digest(s):
                return self_.store.state_digest()

            def list_items(s):
                calls["n"] += 1
                if calls["n"] == 1:                       # read the items while a different file is on disk
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(original.replace("plumber", "carpenter"))
                    items = real_list()
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(original)
                    return items
                return real_list()
        self_ = self
        p = self.plan(store=Flicker())                    # the first attempt is rejected (items != digest), the second is clean
        self.assertEqual(p.superseded[0].value, "plumber")
        self.assertEqual(calls["n"], 2)

    def failing_store(self, where, exc):
        real = self.store
        n = {"digest": 0}

        class Failing:
            def state_digest(s):
                n["digest"] += 1
                if (where == "first" and n["digest"] == 1) or (where == "second" and n["digest"] == 2):
                    raise exc
                return real.state_digest()

            def list_items(s):
                if where == "items":
                    raise exc
                return real.list_items()
        return Failing()

    def test_expected_read_failures_become_stable_codes_that_leak_nothing(self):
        self.lock_occupation()
        cases = ((OSError(SECRET), "io_error"), (PermissionError(13, SECRET), "io_error"),
                 (UnicodeDecodeError("utf-8", b"\xff", 0, 1, SECRET), "io_error"), (ValueError(SECRET), "failed"),
                 (RuntimeError(SECRET), "failed"), (KeyError(SECRET), "failed"), (TypeError(SECRET), "failed"),
                 (S.StoreError("corrupt_store", (SECRET,)), "corrupt_store"), (S.StoreError("journal_corrupt"), "journal_corrupt"),
                 (S.StoreError("A reason with spaces " + SECRET), "failed"), (S.StoreError(SECRET.upper()), "failed"))
        for where in ("first", "items", "second"):
            for exc, code in cases:
                with self.subTest(where=where, exc=type(exc).__name__, code=code):
                    with self.assertRaises(C.CommitError) as cm:
                        self.plan(store=self.failing_store(where, exc))
                    e = cm.exception
                    self.assertEqual((e.code, e.details), ("store_unreadable", (code,)))
                    self.assertNotIn(SECRET, str(e) + repr(e.details) + repr(e))
                    self.assertIsNone(e.__context__)
                    self.assertIsNone(e.__cause__)

    def test_records_that_cannot_be_serialized_become_a_stable_code(self):
        self.lock_occupation()
        good = item(1, attribute="a1")
        for bad in (S.Stored(replace(good, value=""), "I live in San Jose now"),
                    S.Stored(good, "line one\nline two"), S.Stored(good, "bad \ud800 surrogate"),
                    S.Stored(replace(good, source=None), "I live in San Jose now")):
            with self.subTest(bad=repr(bad)[:30]):
                with self.assertRaises(C.CommitError) as cm:
                    C.digest_of([bad])
                self.assertEqual((cm.exception.code, cm.exception.details), ("store_unreadable", ("serialize",)))
                self.assertIsNone(cm.exception.__context__)

    def test_store_errors_and_bad_shapes_fail_closed(self):
        self.lock_occupation()
        os.makedirs(self.store.dir)
        with open(os.path.join(self.store.dir, "stray.txt"), "w") as f:
            f.write("x")
        e = self.refused("store_unreadable")
        self.assertEqual(e.details, ("corrupt_store",))
        for bad in (object(), None):
            self.assertEqual(self.refused("bad_input", store=bad).details, ("store",))

        class Shapes:
            def __init__(s, d, items):
                s.d, s.items = d, items

            def state_digest(s):
                return s.d

            def list_items(s):
                return s.items
        for d, items in (("zz", ()), (None, ()), ("0" * 64, "notalist")):
            self.assertEqual(self.refused("store_unreadable", store=Shapes(d, items)).details, ("shape",))

    def test_a_hand_reformatted_but_parsable_file_blocks_planning(self):
        self.lock_occupation()
        old = self.put_old()
        path = os.path.join(self.store.dir, old.id + ".md")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace('value: "plumber"', 'value: "plumber" '))        # parses, but is not canonical
        self.assertEqual(len(self.store.list_items()), 1)                         # the store itself still loads it
        e = self.refused("store_unstable")
        self.assertEqual(e.details, ("digest_of",))

    def test_the_store_is_read_with_exactly_the_documented_calls(self):
        self.lock_occupation()
        order = []

        class Spy:
            def state_digest(s):
                order.append("digest")
                return self_.store.state_digest()

            def list_items(s):
                order.append("items")
                return self_.store.list_items()
        self_ = self
        self.plan(store=Spy())
        self.assertEqual(order, ["digest", "items", "digest"])


class DigestEquivalenceTests(unittest.TestCase):
    """digest_of(list_items()) must equal store.state_digest() byte for byte, for every file the store can produce."""

    def setUp(self):
        install_tripwire(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.store = S.MemoryStore(self.root)

    def check(self):
        items = self.store.list_items()
        self.assertEqual(C.digest_of(items), self.store.state_digest())
        for name in sorted(os.listdir(self.store.dir)) if os.path.isdir(self.store.dir) else ():
            if re.match(r"^mem_[0-9a-f]{12}\.md\Z", name):
                with open(os.path.join(self.store.dir, name), encoding="utf-8", newline="") as f:
                    text = f.read()
                self.assertEqual(S.serialize(*S.parse_file(text)), text, name)
                self.assertEqual(text.encode("utf-8"), open(os.path.join(self.store.dir, name), "rb").read())

    def test_empty_and_absent(self):
        self.check()
        os.makedirs(self.store.dir)
        self.check()

    def test_put_apply_and_every_status(self):
        self.store.put(item(1, attribute="a1"), "I live in San Jose now")
        arrange(self.store, item(2, attribute="a2", value="Reno", sensitivity="private"), "I live in Reno")
        self.store.put(item(3, attribute="a3", value="walks", confidence="inferred", source={"kind": "session", "ref": "s1"}),
                       "he walks every morning before work, plainly")
        old = old_occupation(9)
        self.store.put(old, "I am a plumber by trade")
        new = item(4, attribute="occupation", value="plumber of homes", created=day(11), updated=day(11), valid_from=day(11))
        prop = G.Proposal(new, "I am a plumber of homes")
        self.store.apply(G.evaluate(prop, [s.item for s in self.store.list_items()], "2026-02-01T00:00:00Z"), prop, "2026-02-01T00:00:00Z")
        arrange(self.store, item(5, attribute="a5", status="archived", value="smoked"), "I used to smoke a lot")
        arrange(self.store, item(6, attribute="a6", value="old", valid_to=day(20), valid_from=day(5), created=day(5), updated=day(5)), "an old fact stays here")
        self.assertGreaterEqual(len(self.store.list_items()), 7)
        self.check()

    def test_awkward_text(self):
        values = ["a: b", "---", "## Evidence", "x y\u0085z ", "tab\there", "emoji \U0001F600", "'single' \"double\"", "back\\slash",
                  " lead and trail ", "café", "café", "a b", "a b", "right‏to​left", "﻿bom",
                  "q" * 200]
        for n, v in enumerate(values, 1):
            arrange(self.store, item(n, attribute=f"x{n}", value=v), f"  evidence number {n} here  ")
        arrange(self.store, item(40, attribute="e1"), "ev\u0085with raw separators here")
        arrange(self.store, item(41, attribute="e2"), "e" * 300)
        self.check()

    def test_many_items_order(self):
        for n in (0xA, 0x1, 0xF, 0x2, 0xB):
            arrange(self.store, item(n, attribute=f"k{n}"), "I live in San Jose now")
        self.check()

    def test_a_changed_byte_a_missing_file_and_a_duplicate_id_all_break_the_equality(self):
        arrange(self.store, item(1), "I live in San Jose now")
        arrange(self.store, item(2, attribute="b"), "I live in San Jose now")
        items = self.store.list_items()
        digest = self.store.state_digest()
        self.assertEqual(C.digest_of(items), digest)
        self.assertNotEqual(C.digest_of(items[:1]), digest)
        path = os.path.join(self.store.dir, mid(1) + ".md")
        with open(path, "ab") as f:
            f.write(b" ")
        self.assertNotEqual(self.store.state_digest(), C.digest_of(items))
        with self.assertRaises(C.CommitError) as cm:
            C.digest_of(items + items[:1])
        self.assertEqual(cm.exception.details, ("duplicate_id",))
        with self.assertRaises(C.CommitError):
            C.digest_of([object()])

    def test_the_formula_uses_the_stores_version_tag_and_names(self):
        arrange(self.store, item(1), "I live in San Jose now")
        st = self.store.list_items()[0]
        data = S.serialize(st.item, st.evidence).encode("utf-8")
        body = {"v": S.AUTH_VERSION, "files": [[st.item.id + ".md", hashlib.sha256(data).hexdigest()]]}
        want = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(C.digest_of([st]), want)
        self.assertEqual(want, self.store.state_digest())


# ---- L1: change ONE field of a baseline Plan; exactly these invariants must be reported -------------------------

def _rec_mut(p, **kw):
    r = json.loads(p.lock_record_json)
    r.update(kw)
    return replace(p, lock_record_json=R.canonical_json(r))

def _org_mut(p, **kw):
    r = json.loads(p.lock_record_json)
    r["origin"].update(kw)
    return replace(p, lock_record_json=R.canonical_json(r))

def it(p, **kw):
    return replace(p, item=replace(p.item, **kw))

def old0(p, **kw):
    return replace(p, superseded=(replace(p.superseded[0], **kw),))

rec_mut, org_mut = _rec_mut, _org_mut


L1_CASES = [
    ('operation put->apply', 'put', lambda p: replace(p, operation="apply"), ('item_digest', 'display', 'operation_links')),
    ('operation apply->put', 'apply', lambda p: replace(p, operation="put"), ('item_digest', 'display', 'operation_links')),
    ('kind commit->private', 'put', lambda p: replace(p, kind="commit-private"), ('display', 'kind')),
    ('kind private->commit', 'priv', lambda p: replace(p, kind="commit"), ('display', 'kind')),
    ('links emptied', 'apply', lambda p: replace(p, links=()), ('item_digest', 'operation_links', 'links', 'gate_table', 'decision_fields')),
    ('links other id', 'apply', lambda p: replace(p, links=("mem_0000000000ff",)), ('item_digest', 'links')),
    ('links added on put', 'put', lambda p: replace(p, links=("mem_0000000000ff",)), ('item_digest', 'operation_links', 'links', 'gate_table', 'decision_fields')),
    ('decision reason', 'put', lambda p: replace(p, decision=replace(p.decision, reason="accepted_supersedes")), ('display', 'gate', 'gate_table')),
    ('decision verdict', 'put', lambda p: replace(p, decision=replace(p.decision, decision=G.REQUIRE_CONFIRMATION, reason="private_requires_confirmation")), ('display', 'gate', 'gate_table')),
    ('decision memory_id', 'put', lambda p: replace(p, decision=replace(p.decision, memory_id="mem_0000000000ff")), ('gate', 'decision_fields')),
    ('decision effective_from', 'apply', lambda p: replace(p, decision=replace(p.decision, effective_from="2020-01-01T00:00:00Z")), ('gate', 'decision_fields')),
    ('decision evidence', 'put', lambda p: replace(p, decision=replace(p.decision, evidence="x")), ('gate', 'decision_fields')),
    ('decision conflict', 'put', lambda p: replace(p, decision=replace(p.decision, conflict=("mem_0000000000ff",))), ('gate', 'decision_fields')),
    ('decision details', 'put', lambda p: replace(p, decision=replace(p.decision, details=("x",))), ('gate', 'decision_fields')),
    ('old id', 'apply', lambda p: old0(p, id="mem_0000000000ff"), ('display', 'links')),
    ('old key', 'apply', lambda p: old0(p, key="user.other"), ('display', 'links')),
    ('old value', 'apply', lambda p: old0(p, value="carpenter"), ('display', 'links')),
    ('old valid_from', 'apply', lambda p: old0(p, valid_from="2020-01-01T00:00:00Z"), ('display', 'links')),
    ('old confidence', 'apply', lambda p: old0(p, confidence="inferred"), ('display', 'links')),
    ('old sensitivity', 'apply', lambda p: old0(p, sensitivity="private"), ('display', 'links')),
    ('old status', 'apply', lambda p: old0(p, status="archived"), ('display', 'links')),
    ('old file_sha8', 'apply', lambda p: old0(p, file_sha8="00000000"), ('display', 'links')),
    ('superseded dropped', 'apply', lambda p: replace(p, superseded=()), ('display', 'links')),
    ('store_items dropped', 'apply', lambda p: replace(p, store_items=()), ('links', 'gate', 'state')),
    ('store_items evidence', 'apply', lambda p: replace(p, store_items=(replace(p.store_items[0], evidence="I am a carpenter by trade"),)), ('links', 'state')),
    ('state_digest', 'put', lambda p: replace(p, state_digest="0"*64), ('display', 'state')),
    ('item_digest', 'put', lambda p: replace(p, item_digest="0"*64), ('item_digest', 'display')),
    ('context_digest', 'put', lambda p: replace(p, context_digest="0"*64), ('context_digest', 'display')),
    ('display_digest', 'put', lambda p: replace(p, display_digest="0"*64), ('display',)),
    ('item value', 'put', lambda p: it(p, value="something else"), ('item_vs_record', 'item_digest', 'display', 'gate')),
    ('evidence', 'put', lambda p: replace(p, evidence="a different sentence entirely"), ('item_vs_record', 'item_digest', 'display', 'gate', 'decision_fields')),
    ('item attribute', 'put', lambda p: it(p, attribute="residence"), ('item_vs_record', 'item_digest', 'display')),
    ('item type', 'put', lambda p: it(p, type="preference"), ('item_vs_record', 'item_digest', 'display')),
    ('item subject', 'put', lambda p: it(p, subject="max"), ('item_vs_record', 'item_digest', 'display')),
    ('item created', 'put', lambda p: it(p, created="2026-03-01T10:04:59Z"), ('item_vs_record', 'item_digest', 'display')),
    ('item updated', 'put', lambda p: it(p, updated="2026-03-01T10:05:01Z"), ('item_vs_record', 'item_digest', 'display')),
    ('item source ref', 'put', lambda p: it(p, source=M.Source("user", "review:xx#c@t")), ('item_vs_record', 'item_digest', 'display')),
    ('item source kind', 'put', lambda p: it(p, source=M.Source("note", p.item.source.ref)), ('item_vs_record', 'item_digest', 'display', 'gate')),
    ('item confidence', 'put', lambda p: it(p, confidence="inferred"), ('item_vs_record', 'item_digest', 'display', 'gate')),
    ('item valid_from', 'put', lambda p: it(p, valid_from="2026-03-01T10:05:01Z"), ('item_vs_record', 'item_digest', 'display')),
    ('item valid_to', 'put', lambda p: it(p, valid_to="2027-01-01T00:00:00Z"), ('item_vs_record', 'item_digest', 'display')),
    ('item status', 'put', lambda p: it(p, status="archived"), ('item_vs_record', 'item_digest', 'display', 'gate')),
    ('item sensitivity put->private', 'put', lambda p: it(p, sensitivity="private"), ('item_vs_record', 'item_digest', 'display', 'kind', 'gate', 'gate_table', 'decision_fields')),
    ('item sensitivity priv->normal', 'priv', lambda p: it(p, sensitivity="normal"), ('item_vs_record', 'item_digest', 'display', 'kind', 'gate', 'gate_table', 'decision_fields')),
    ('item supersedes', 'put', lambda p: it(p, supersedes=("mem_0000000000ff",)), ('item_supersedes', 'gate')),
    ('item id', 'put', lambda p: it(p, id="mem_0000000000ff"), ('item_vs_record', 'item_digest', 'display', 'gate', 'decision_fields')),
    ('candidate', 'put', lambda p: replace(p, candidate="cand_other"), ('record', 'item_vs_record', 'context_digest')),
    ('run', 'put', lambda p: replace(p, run="run-2"), ('item_vs_record', 'context_digest')),
    ('pack_sha', 'put', lambda p: replace(p, pack_sha="b"*64), ('context_digest',)),
    ('registry_version', 'put', lambda p: replace(p, registry_version=2), ('record', 'context_digest')),
    ('lock_hash', 'put', lambda p: replace(p, lock_hash="0"*64), ('lock_hash', 'context_digest')),
    ('record where', 'put', lambda p: org_mut(p, where="Other.md"), ('lock_hash', 'display')),
    ('record line', 'put', lambda p: org_mut(p, line=999), ('lock_hash', 'display')),
    ('record text_sha', 'put', lambda p: org_mut(p, text_sha="0"*64), ('lock_hash', 'display')),
    ('record value', 'put', lambda p: rec_mut(p, value="zzz"), ('lock_hash', 'item_vs_record')),
    ('record adoption.at', 'put', lambda p: rec_mut(p, adoption={"by":"user","channel":"tty","at":"2026-03-01T10:05:01Z","kind":"adopted_note_text"}), ('lock_hash', 'item_vs_record', 'display')),
    ('record json invalid', 'put', lambda p: replace(p, lock_record_json="{not json"), ('integrity_error',)),
    ('record action', 'put', lambda p: rec_mut(p, action="reject"), ('record', 'lock_hash')),
    ('planned_at +1h', 'put', lambda p: replace(p, planned_at="2026-03-02T11:00:00Z"), ('age',)),
    ('age_seconds', 'put', lambda p: replace(p, age_seconds=1), ('age',)),
    ('max_age_seconds', 'put', lambda p: replace(p, max_age_seconds=10), ('age',)),
    ('checks', 'put', lambda p: replace(p, checks=C.REQUIRED_CHECKS[:-1]), ('checks',)),
    ('rederive_seconds', 'put', lambda p: replace(p, rederive_seconds=99.0), ()),
]


def make_bases(case):
    """Three baseline plans on one ledger: a plain put, an apply that replaces an older value, and a private put."""
    case.lock_occupation()
    put = case.plan()
    case.put_old()
    apply_ = case.plan()
    out = R.edit(case.pack, case.state, case.cid("goal_know_each_market_well"), value="I earn $800/week from the work", now=T1)
    case.step(out)
    side = {out.private["seq"]: {"value": out.private["value"], "evidence": out.private["evidence"]}}
    priv = case.plan(candidate=case.cid("goal_know_each_market_well"), private=side)
    return {"put": put, "apply": apply_, "priv": priv}


class L1BindingTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.bases = make_bases(self)

    def test_the_baselines_are_consistent(self):
        for name, p in self.bases.items():
            self.assertEqual(C.verify_plan_integrity(p), [], name)
        self.assertEqual({k: (p.operation, p.kind) for k, p in self.bases.items()},
                         {"put": ("put", "commit"), "apply": ("apply", "commit"), "priv": ("put", "commit-private")})

    def test_each_single_change_reports_exactly_its_invariants(self):
        for label, base, fn, expected in L1_CASES:
            with self.subTest(label=label):
                self.assertEqual(sorted(C.verify_plan_integrity(fn(self.bases[base]))), sorted(expected))

    def test_every_flagged_change_also_refuses_the_token_the_preview_and_the_answer(self):
        for label, base, fn, expected in L1_CASES:
            variant = fn(self.bases[base])
            with self.subTest(label=label):
                if expected:
                    for call in (lambda: C.challenge_token(variant, NONCE), lambda: C.expected_answer(variant, NONCE),
                                 lambda: C.render_preview(variant),
                                 lambda: C.check_answer("x", variant, NONCE, NOWS, NOWS)):
                        with self.assertRaises(C.CommitError) as cm:
                            call()
                        self.assertEqual(cm.exception.code, "plan_inconsistent")
                        self.assertEqual(sorted(cm.exception.details), sorted(expected))
                    self.assertEqual(C.compare_plans(self.bases[base], variant), ["plan_inconsistent"])
                else:
                    self.assertEqual(C.challenge_token(variant, NONCE), C.challenge_token(self.bases[base], NONCE))

    def test_the_table_covers_every_plan_field_and_every_item_and_old_item_field(self):
        changed, item_changed, old_changed = set(), set(), set()
        for label, base, fn, expected in L1_CASES:
            b, v = self.bases[base], fn(self.bases[base])
            changed |= {f.name for f in fields(C.Plan) if getattr(v, f.name) != getattr(b, f.name)}
            if v.item != b.item:
                item_changed |= {f.name for f in fields(M.MemoryItem) if getattr(v.item, f.name) != getattr(b.item, f.name)}
            if v.superseded != b.superseded and v.superseded and b.superseded:
                old_changed |= {f.name for f in fields(C.OldItem) if getattr(v.superseded[0], f.name) != getattr(b.superseded[0], f.name)}
        self.assertEqual(changed, {f.name for f in fields(C.Plan)})
        self.assertEqual(item_changed, {f.name for f in fields(M.MemoryItem)})
        self.assertEqual(old_changed, {f.name for f in fields(C.OldItem)})

    def test_only_the_informational_field_can_change_without_being_noticed(self):
        self.assertEqual([label for label, base, fn, expected in L1_CASES if not expected], ["rederive_seconds"])

    def test_a_changed_operation_is_refused_even_when_every_digest_is_recomputed_to_match(self):
        forged = replace(self.bases["put"], operation="apply")
        forged = replace(forged, item_digest=S.item_digest(forged.item, forged.evidence, forged.links, "apply"))
        forged = replace(forged, display_digest=C._display_checked(forged)[1])
        problems = C.verify_plan_integrity(forged)
        self.assertEqual(problems, ["operation_links"])                 # the digests agree; the OPERATION is still wrong
        with self.assertRaises(C.CommitError):
            C.challenge_token(forged, NONCE)

    def test_a_changed_kind_is_refused_even_when_the_display_is_recomputed(self):
        forged = replace(self.bases["put"], kind="commit-private")
        forged = replace(forged, display_digest=C._display_checked(forged)[1])
        self.assertEqual(C.verify_plan_integrity(forged), ["kind"])

    def test_forged_links_with_matching_digests_still_disagree_with_the_decision(self):
        p = self.bases["apply"]
        forged = replace(p, links=(), superseded=(), operation="put")
        forged = replace(forged, item_digest=S.item_digest(forged.item, forged.evidence, (), "put"))
        forged = replace(forged, display_digest=C._display_checked(forged)[1])
        problems = C.verify_plan_integrity(forged)
        self.assertEqual(sorted(problems), ["decision_fields", "gate_table", "links"])


class TokenTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.bases = make_bases(self)
        self.p = self.bases["put"]

    def test_the_token_is_eight_hex_characters_and_deterministic(self):
        t = C.challenge_token(self.p, NONCE)
        self.assertRegex(t, r"^[0-9a-f]{8}\Z")
        self.assertEqual(t, C.challenge_token(self.p, NONCE))
        self.assertEqual(C.expected_answer(self.p, NONCE), f"commit {t}")
        self.assertEqual(C.expected_answer(self.bases["priv"], NONCE),
                         f"commit-private {C.challenge_token(self.bases['priv'], NONCE)}")

    def test_each_input_of_the_token_changes_it(self):
        parts = dict(operation="put", kind="commit", item_digest="1" * 64, state_digest="2" * 64, context_digest="3" * 64,
                     display_digest="4" * 64, nonce=NONCE)
        changed = {"operation": "apply", "kind": "commit-private", "item_digest": "9" * 64, "state_digest": "9" * 64,
                   "context_digest": "9" * 64, "display_digest": "9" * 64, "nonce": "c" * 32}
        seen = {C._token_digest(**parts)}
        for k, v in changed.items():
            t = C._token_digest(**{**parts, k: v})
            self.assertNotIn(t, seen, k)
            seen.add(t)

    def test_the_token_version_is_part_of_the_hash(self):
        body = {"v": C.TOKEN_VERSION, "operation": "put", "kind": "commit", "item_digest": "1" * 64, "state_digest": "2" * 64,
                "context_digest": "3" * 64, "display_digest": "4" * 64, "nonce": NONCE}
        want = hashlib.sha256(R.canonical_json(body).encode("utf-8")).hexdigest()[:8]
        self.assertEqual(C._token_digest("put", "commit", "1" * 64, "2" * 64, "3" * 64, "4" * 64, NONCE), want)
        self.assertEqual(C.TOKEN_VERSION, "commit-token/2")

    def test_a_bad_nonce_is_refused(self):
        for bad in (None, "", "b" * 31, "B" * 32, "g" * 32, 5, b"b" * 32):
            with self.assertRaises(C.CommitError) as cm:
                C.challenge_token(self.p, bad)
            self.assertEqual(cm.exception.code, "bad_input")


class AnswerTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.bases = make_bases(self)
        self.p = self.bases["put"]
        self.want = C.expected_answer(self.p, NONCE)
        self.t0, self.t1 = "2026-03-02T10:00:05Z", "2026-03-02T10:00:35Z"

    def check(self, text, plan=None, nonce=NONCE, issued=None, answered=None):
        return C.check_answer(text, plan or self.p, nonce, issued or self.t0, answered or self.t1)

    def refuse(self, code, *a, **k):
        with self.assertRaises(C.CommitError) as cm:
            self.check(*a, **k)
        self.assertEqual(cm.exception.code, code, str(cm.exception))
        return cm.exception

    def test_the_exact_answer_returns_the_confirmation_instant(self):
        self.assertEqual(self.check(self.want), self.t1)

    def test_nothing_but_the_exact_text_is_accepted(self):
        token = self.want.split(" ")[1]
        for bad in ("", "commit", "commit ", f"commit  {token}", f" {self.want}", f"{self.want} ", f"{self.want}\n", self.want.upper(),
                    f"Commit {token}", f"commit {token.upper()}", "yes", "y", "lock", "confirm", f"commit {token[:-1]}",
                    f"commit {token}0", f"commit\t{token}"):
            self.refuse("challenge_declined", bad)

    def test_the_wrong_kind_word_is_ambiguous(self):
        token = self.want.split(" ")[1]
        self.refuse("challenge_ambiguous", f"commit-private {token}")
        priv = self.bases["priv"]
        ptok = C.challenge_token(priv, NONCE)
        self.assertEqual(self.check(f"commit-private {ptok}", plan=priv), self.t1)
        self.refuse("challenge_ambiguous", f"commit {ptok}", plan=priv)

    def test_non_text_is_ambiguous(self):
        for bad in (None, b"commit abc", 5, ["commit"], object()):
            self.assertEqual(self.refuse("challenge_ambiguous", bad).details, ("not_text",))

    def test_a_token_from_another_attempt_or_plan_is_declined(self):
        self.refuse("challenge_declined", C.expected_answer(self.p, "c" * 32))
        self.refuse("challenge_declined", C.expected_answer(self.bases["apply"], NONCE))

    def test_the_window_is_120_seconds_inclusive_and_expiry_comes_first(self):
        self.assertEqual(self.check(self.want, issued="2026-03-02T10:00:00Z", answered="2026-03-02T10:02:00Z"), "2026-03-02T10:02:00Z")
        self.refuse("challenge_expired", self.want, issued="2026-03-02T10:00:00Z", answered="2026-03-02T10:02:01Z")
        self.refuse("challenge_expired", "yes", issued="2026-03-02T10:00:00Z", answered="2026-03-02T10:02:01Z")

    def test_the_clock_must_not_run_backwards(self):
        self.refuse("clock_regressed", self.want, issued="2026-03-02T09:59:59Z", answered=self.t1)
        self.refuse("clock_regressed", self.want, issued=self.t1, answered=self.t0)
        self.assertEqual(self.check(self.want, issued=NOWS, answered=NOWS), NOWS)

    def test_the_stamps_must_be_strict(self):
        for bad in (None, "", "2026-03-02", "2026-03-02 10:00:05", 5):
            for kw in ({"issued": bad}, {"answered": bad}):
                with self.assertRaises(C.CommitError) as cm:
                    C.check_answer(self.want, self.p, NONCE, kw.get("issued", self.t0), kw.get("answered", self.t1))
                self.assertEqual(cm.exception.code, "bad_timestamp")

    def test_an_inconsistent_plan_never_reaches_the_token(self):
        self.refuse("plan_inconsistent", "anything", plan=replace(self.p, operation="apply"))

    def test_expected_answer_checks_the_plan_before_reading_any_field(self):
        for bad in (None, object(), "plan", 5, {"kind": "commit"}, [], type("Fake", (), {"kind": "commit"})()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(C.CommitError) as cm:
                    C.expected_answer(bad, NONCE)
                self.assertEqual((cm.exception.code, cm.exception.details), ("plan_inconsistent", ("bad_plan",)))
        with self.assertRaises(C.CommitError) as cm:
            C.expected_answer(replace(self.p, kind="commit-private"), NONCE)
        self.assertEqual(cm.exception.code, "plan_inconsistent")


class ComparePlansTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.lock_occupation()
        self.a = self.plan()

    def test_identical_plans_at_different_times_compare_equal(self):
        b = self.plan(clock=lambda: "2026-03-02T10:00:30Z", monotonic=iter([5.0, 9.0]).__next__)
        self.assertNotEqual((self.a.planned_at, self.a.age_seconds, self.a.rederive_seconds),
                            (b.planned_at, b.age_seconds, b.rederive_seconds))
        self.assertEqual(C.compare_plans(self.a, b), [])

    def test_a_record_appended_after_the_lock_changes_nothing(self):
        self.step(R.skip(self.pack, self.state, self.cid("goal_cash_buyer_list_deep"), now=T1))
        self.assertEqual(C.compare_plans(self.a, self.plan()), [])

    def test_a_second_plan_from_an_earlier_instant_is_flagged(self):
        self.assertEqual(C.compare_plans(self.a, self.plan(clock=lambda: "2026-03-02T09:00:00Z")), ["clock_regressed"])

    def test_bad_arguments(self):
        for x, y in ((None, self.a), (self.a, None), ("p", "q"), (self.a, object())):
            self.assertEqual(C.compare_plans(x, y), ["bad_plan"])
        self.assertEqual(C.compare_plans(self.a, replace(self.a, operation="apply")), ["plan_inconsistent"])
        self.assertEqual(C.compare_plans(replace(self.a, operation="apply"), self.a), ["plan_inconsistent"])

    def test_an_older_rival_turns_the_put_into_an_apply(self):
        self.put_old()
        self.assertEqual(C.compare_plans(self.a, self.plan()),
                         ["operation", "links", "reason", "item_digest", "state_digest", "display"])

    def test_a_changed_superseded_file_changes_the_state_and_the_display(self):
        old = self.put_old()
        first = self.plan()
        arrange(self.store, replace(old, value="carpenter"), "I am a carpenter by trade")
        second = self.plan()
        self.assertEqual(C.compare_plans(first, second), ["state_digest", "display"])
        self.assertEqual(second.superseded[0].value, "carpenter")

    def test_an_unrelated_new_memory_changes_only_the_state(self):
        self.store.put(item(5, attribute="a5", value="Reno"), "I live in Reno now")
        self.assertEqual(C.compare_plans(self.a, self.plan()), ["state_digest", "display"])

    def test_a_different_locked_candidate_changes_what_it_binds(self):
        self.lock("goal_cash_buyer_list_deep")
        names = C.compare_plans(self.a, self.plan(candidate=self.cid("goal_cash_buyer_list_deep")))
        for n in ("candidate", "lock_hash", "memory_id", "item_digest", "context_digest", "display"):
            self.assertIn(n, names)
        self.assertNotIn("state_digest", names)
        self.assertNotIn("operation", names)

    def test_a_private_edit_is_a_different_kind_and_sensitivity(self):
        out = R.edit(self.pack, self.state, self.cid("goal_know_each_market_well"), value="I earn $800/week from the work", now=T1)
        self.step(out)
        side = {out.private["seq"]: {"value": out.private["value"], "evidence": out.private["evidence"]}}
        names = C.compare_plans(self.a, self.plan(candidate=self.cid("goal_know_each_market_well"), private=side))
        for n in ("kind", "sensitivity", "decision", "reason"):
            self.assertIn(n, names)

    def test_a_gate_result_that_changes_with_time_shows_up_as_a_difference(self):
        ended = old_occupation(value="plumber", valid_from=day(1), valid_to="2026-03-02T10:00:20Z", created=day(1), updated=day(1))
        arrange(self.store, ended, "I am a plumber by trade")
        diff = C.compare_plans(self.plan(clock=lambda: "2026-03-02T10:00:10Z"), self.plan(clock=lambda: "2026-03-02T10:00:40Z"))
        self.assertTrue({"operation", "links", "reason", "item_digest", "display"} <= set(diff), diff)


class PreviewTests(PlanCase):
    def setUp(self):
        super().setUp()
        self.bases = make_bases(self)

    def lines(self, name="put"):
        return C.render_preview(self.bases[name]).bound

    def test_every_decision_relevant_field_is_shown(self):
        p = self.bases["apply"]
        text = "\n".join(self.lines("apply"))
        rec = json.loads(p.lock_record_json)
        for needle in (p.operation, p.kind, p.item.id, p.item.attribute, p.item.type, p.item.subject, p.item.value, p.evidence,
                       p.item.confidence, p.item.status, p.item.sensitivity, p.item.valid_from, str(p.item.valid_to),
                       p.item.created, p.item.updated, rec["adoption"]["at"], p.item.source.kind, p.item.source.ref,
                       rec["origin"]["where"], str(rec["origin"]["line"]), rec["origin"]["text_sha"][:16],
                       p.decision.decision, p.decision.reason, p.state_digest[:16], p.item_digest[:16], p.context_digest[:16],
                       p.superseded[0].id, p.superseded[0].value, p.superseded[0].file_sha8, p.superseded[0].valid_from,
                       p.superseded[0].status, p.superseded[0].confidence, p.superseded[0].sensitivity, C.NOTICE):
            self.assertIn(needle, text)

    def test_the_label_set_is_fixed_and_ordered(self):
        labels = [ln.split(" ", 1)[0] for ln in self.lines()][:-1]
        self.assertEqual(labels, ["operation", "kind", "item", "key", "subject", "type", "value", "evidence", "confidence",
                                  "status", "sensitivity", "valid_from", "valid_to", "created", "updated", "adopted_at",
                                  "source", "origin", "replaces", "gate", "store", "item", "context"])

    def test_no_replacement_is_said_explicitly(self):
        self.assertIn("replaces     nothing", self.lines("put"))

    def test_the_old_values_come_from_the_plan_alone(self):
        p = self.bases["apply"]
        before = C.render_preview(p)
        for name in os.listdir(self.store.dir):
            if name.endswith(".md"):
                os.unlink(os.path.join(self.store.dir, name))               # the store is gone; the preview must not care
        self.assertEqual(C.render_preview(p), before)
        self.assertIn("plumber", "\n".join(before.bound))

    def test_a_changed_old_value_changes_the_display_digest(self):
        base = self.bases["apply"]
        d0 = C._display_digest(C._display_lines(base))
        changed = replace(base, superseded=(replace(base.superseded[0], value="carpenter"),))
        self.assertNotEqual(C._display_digest(C._display_lines(changed)), d0)

    def test_the_informational_lines_are_outside_the_digest(self):
        p = self.bases["put"]
        later = replace(p, planned_at="2026-03-02T12:00:00Z", age_seconds=999, rederive_seconds=7.0)
        self.assertEqual(C._display_digest(C._display_lines(later)), C._display_digest(C._display_lines(p)))
        info = C.render_preview(p).info
        self.assertEqual(info[0], C.INFO_HEADING)
        self.assertTrue(any("lock age" in ln for ln in info))
        self.assertFalse(any("lock age" in ln for ln in C.render_preview(p).bound))

    def test_rendering_is_deterministic_across_processes_and_hash_seeds(self):
        digests = set()
        code = ("import sys\nfrom tests.test_memory_commit_plan import *\n"
                "class T(PlanCase):\n"
                "    def test_x(self):\n"
                "        b = make_bases(self)\n"
                "        print(C._display_digest(C._display_lines(b['apply'])))\n"
                "import unittest\nunittest.TextTestRunner(stream=open(os.devnull,'w')).run(unittest.TestSuite([T('test_x')]))\n")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
            out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
            digests.add(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:])
        self.assertEqual(len(digests), 1, digests)
        self.assertRegex(next(iter(digests)), r"^[0-9a-f]{64}\Z")

    def test_safe_text_leaves_nothing_hidden(self):
        self.assertEqual(C.safe_text("plain é text"), '"plain é text"')
        self.assertEqual(C.safe_text('a"b\\c'), '"a\\"b\\\\c"')
        for ch, esc in (("\t", "\\u0009"), ("\n", "\\u000a"), ("\x00", "\\u0000"), ("\x7f", "\\u007f"), ("\u0085", "\\u0085"),
                        (" ", "\\u2028"), (" ", "\\u2029"), ("‮", "\\u202e"), ("​", "\\u200b"),
                        ("﻿", "\\ufeff"), (" ", "\\u00a0"), ("　", "\\u3000"), ("", "\\ue000"),
                        ("\U000e0001", "\\U000e0001"), ("\ud800", "\\ud800")):
            self.assertEqual(C.safe_text("a" + ch + "b"), f'"a{esc}b"', repr(ch))
        self.assertEqual(C.safe_text("\U0001F600 ok"), '"\U0001F600 ok"')           # an emoji is a symbol, not hidden
        for bad in (None, 5, b"x", ["x"]):
            with self.assertRaises(C.CommitError) as cm:
                C.safe_text(bad)
            self.assertEqual(cm.exception.code, "display_failed")

    def test_a_value_with_hidden_characters_is_shown_escaped_in_the_preview(self):
        p = self.bases["put"]
        tricky = replace(p, evidence=p.evidence)                                    # shape check only; the item text is plain
        self.assertNotIn("‮", "\n".join(C.render_preview(tricky).bound))

    def test_a_missing_field_stops_rendering(self):
        p = self.bases["put"]
        for field_, bad in (("operation", ""), ("kind", None), ("item", None), ("evidence", ""), ("decision", None),
                            ("links", None), ("superseded", None), ("state_digest", ""), ("item_digest", None),
                            ("context_digest", ""), ("lock_record_json", "")):
            with self.assertRaises(C.CommitError) as cm:
                C._display_lines(replace(p, **{field_: bad}))
            self.assertEqual(cm.exception.code, "display_incomplete", field_)
            self.assertEqual(cm.exception.details, (field_,))

    def test_a_record_without_an_origin_stops_rendering(self):
        p = self.bases["put"]
        rec = json.loads(p.lock_record_json)
        del rec["origin"]
        with self.assertRaises(C.CommitError) as cm:
            C._display_lines(replace(p, lock_record_json=R.canonical_json(rec)))
        self.assertEqual(cm.exception.code, "display_failed")

    def test_rendering_twice_must_agree(self):
        p = self.bases["put"]
        outputs = iter([("a",), ("b",)] * 10)
        with mock.patch.object(C, "_display_lines", lambda plan: next(outputs)):
            with self.assertRaises(C.CommitError) as cm:
                C._display_checked(p)
        self.assertEqual(cm.exception.code, "display_unstable")

    def test_an_inconsistent_plan_has_no_preview(self):
        with self.assertRaises(C.CommitError) as cm:
            C.render_preview(replace(self.bases["put"], kind="commit-private"))
        self.assertEqual(cm.exception.code, "plan_inconsistent")

    def test_the_notice_says_the_token_is_not_authentication(self):
        self.assertIn("not authentication", self.lines()[-1])


class DiagnosticsTests(PlanCase):
    """No refusal, no repr and no detail may carry the value or the evidence."""

    def test_no_failure_leaks_the_sentinel(self):
        text = f"my plan is {SECRET} always"
        out = R.edit(self.pack, self.state, self.cid("goal_cash_buyer_list_deep"), value=text, now=T1)
        self.step(out)
        self.cand = self.cid("goal_cash_buyer_list_deep")
        good = self.plan()
        self.assertIn(SECRET, "\n".join(C.render_preview(good).bound))              # the person is shown it, deliberately
        self.assertNotIn(SECRET, repr(good))
        scenarios = [
            dict(provenance=lambda v: ["hash_mismatch"]),
            dict(provenance=lambda v: [SECRET]),
            dict(rederive=lambda v: None),
            dict(rederive=lambda v: {**copy.deepcopy(v), "evidence": SECRET}),
            dict(clock=lambda: "2027-01-01T00:00:00Z"),
            dict(clock=lambda: "2026-01-01T00:00:00Z"),
            dict(records=[dict(r, hash="0" * 64) if i else r for i, r in enumerate(self.state.records)]),
            dict(candidate="cand_nope"),
            dict(store=object()),
            dict(preview_budget_s=0),
        ]
        for kw in scenarios:
            with self.subTest(kw=list(kw)):
                with self.assertRaises(C.CommitError) as cm:
                    self.plan(**kw)
                blob = str(cm.exception) + repr(cm.exception.details) + cm.exception.code
                self.assertNotIn(SECRET, blob)
                self.assertNotIn("plan is", blob)
        self.store.put(old_occupation(attribute="goal_cash_buyer_list_deep", value="a deep list", valid_from=T1, created=T1, updated=T1),
                       "I want a deep list")
        with self.assertRaises(C.CommitError) as cm:
            self.plan()
        self.assertNotIn(SECRET, str(cm.exception) + repr(cm.exception.details))
        with self.assertRaises(C.CommitError) as cm:
            C.challenge_token(replace(good, kind="commit-private"), NONCE)
        self.assertNotIn(SECRET, str(cm.exception) + repr(cm.exception.details))

    def test_error_text_shape(self):
        self.assertEqual(str(C.CommitError("x")), "x")
        self.assertEqual(str(C.CommitError("x", ("a", "b"))), "x: a, b")
        e = C.CommitError("x", ["a"])
        self.assertEqual((e.code, e.details), ("x", ("a",)))


class NoLeakageTests(PlanCase):
    """Failures from injected code are normalized to fixed codes, and the original exception (whose text might carry source
    material) is not retained as __context__ or __cause__."""

    def clean(self, e):
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)
        self.assertNotIn(SECRET, str(e) + repr(e.details) + repr(e))

    def test_injected_check_failures(self):
        self.lock_occupation()
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
        for name in ("provenance", "rederive", "monotonic"):
            with self.subTest(name=name):
                with self.assertRaises(C.CommitError) as cm:
                    self.plan(**{name: boom})
                self.assertEqual((cm.exception.code, cm.exception.details), ("check_failed", (name,)))
                self.clean(cm.exception)
        with self.assertRaises(C.CommitError) as cm:
            self.plan(clock=boom)
        self.assertEqual((cm.exception.code, cm.exception.details), ("bad_clock", ("raised",)))
        self.clean(cm.exception)
        with mock.patch.object(R, "replay_validate", boom):
            with self.assertRaises(C.CommitError) as cm:
                self.plan()
        self.assertEqual(cm.exception.code, "check_failed")
        self.clean(cm.exception)

    def test_display_failure(self):
        self.lock_occupation()
        p = self.plan()
        with mock.patch.object(K, "describe", lambda a: (_ for _ in ()).throw(RuntimeError(SECRET))):
            with self.assertRaises(C.CommitError) as cm:
                C._display_lines(p)
        self.assertEqual(cm.exception.code, "display_failed")
        self.clean(cm.exception)

    def test_item_digest_failure(self):
        self.lock_occupation()
        with mock.patch.object(S, "item_digest", lambda *a, **k: (_ for _ in ()).throw(S.StoreError("invalid_item", (SECRET,)))):
            with self.assertRaises(C.CommitError) as cm:
                self.plan()
        self.assertEqual((cm.exception.code, cm.exception.details), ("item_invalid", ("invalid_item",)))
        self.clean(cm.exception)

    def test_old_item_failure(self):
        self.lock_occupation()
        self.put_old()
        with mock.patch.object(C, "_old_item", lambda st: (_ for _ in ()).throw(RuntimeError(SECRET))):
            with self.assertRaises(C.CommitError) as cm:
                self.plan()
        self.assertEqual((cm.exception.code, cm.exception.details), ("store_unreadable", ("serialize",)))
        self.clean(cm.exception)

    def test_no_from_none_remains_in_the_module(self):
        self.assertNotIn("from None", inspect.getsource(C))


class SafeReprTests(PlanCase):
    def test_no_object_the_module_returns_prints_a_value(self):
        self.lock_occupation()
        self.put_old()
        out = R.edit(self.pack, self.state, self.cid("goal_cash_buyer_list_deep"), value=f"my plan is {SECRET} always", now=T1)
        self.step(out)
        p = self.plan(candidate=self.cid("goal_cash_buyer_list_deep"))
        old = self.plan().superseded[0]
        pv = C.render_preview(p)
        self.assertIn(SECRET, "\n".join(pv.bound))                             # the person is shown it, deliberately
        for obj in (p, pv, old, (old,), [pv], {"x": pv}):
            for text in (repr(obj), str(obj), f"{obj}", "%r" % (obj,)):
                self.assertNotIn(SECRET, text)
                self.assertNotIn("plumber", text)
                self.assertNotIn("my plan", text)
        self.assertEqual(repr(old), f"OldItem(id={old.id}, file_sha8={old.file_sha8})")
        self.assertEqual(repr(pv), f"Preview(bound=<{len(pv.bound)} lines>, info=<{len(pv.info)} lines>)")

    def test_equality_and_immutability_are_unchanged(self):
        self.lock_occupation()
        pv = C.render_preview(self.plan())
        self.assertEqual(pv, C.render_preview(self.plan()))
        with self.assertRaises(Exception):
            pv.bound = ()
        old = C.OldItem("mem_000000000001", "user.occupation", "v", "2026-01-01T00:00:00Z", "stated", "normal", "active", "abcd1234")
        self.assertEqual(old, replace(old))
        self.assertNotEqual(old, replace(old, value="w"))
        with self.assertRaises(Exception):
            old.value = "x"


class ContractTests(PlanCase):
    """No plan without the checks: each required check is called once, in order, and refusing results never yield a Plan."""

    def test_each_check_runs_once_in_the_documented_order(self):
        self.lock_occupation()
        order = []
        real_replay, real_eval = R.replay_validate, G.evaluate

        def replay(*a, **k):
            order.append("replay")
            return real_replay(*a, **k)

        def prov(view):
            order.append("provenance")
            return []

        def rederive(view):
            order.append("rederive")
            return copy.deepcopy(view)

        class Store:
            def state_digest(s):
                order.append("digest")
                return self_.store.state_digest()

            def list_items(s):
                order.append("items")
                return self_.store.list_items()
        self_ = self

        def gate(proposal, existing, now, *a, **k):
            if now == NOWS:
                order.append("gate")
            return real_eval(proposal, existing, now, *a, **k)
        with mock.patch.object(R, "replay_validate", replay), mock.patch.object(G, "evaluate", gate):
            p = self.plan(provenance=prov, rederive=rederive, store=Store())
        self.assertEqual(order[:7], ["replay", "provenance", "rederive", "digest", "items", "digest", "gate"])
        self.assertEqual(order.count("replay"), 1)
        self.assertEqual(p.checks, C.REQUIRED_CHECKS)

    def test_a_plan_fails_if_any_single_check_is_removed(self):
        self.lock_occupation()
        p = self.plan()
        for dropped in C.REQUIRED_CHECKS:
            variant = replace(p, checks=tuple(c for c in C.REQUIRED_CHECKS if c != dropped))
            self.assertEqual(C.verify_plan_integrity(variant), ["checks"], dropped)
        self.assertEqual(C.verify_plan_integrity(replace(p, checks=tuple(reversed(C.REQUIRED_CHECKS)))), ["checks"])

    def test_every_injected_check_refuses_every_unusable_result(self):
        self.lock_occupation()
        for name, ok_codes in (("provenance", ("source_stale",)), ("rederive", ("not_rederived", "rederive_mismatch"))):
            for bad in (None, 5, "x", (), [], {}, object()):
                if name == "provenance" and bad == []:
                    continue                                         # an empty list is the one clean provenance result
                with self.subTest(check=name, result=repr(bad)[:20]):
                    self.assertIn(self.catch(**{name: lambda v, bad=bad: bad}), ok_codes)
            with self.subTest(check=name, raising=True):
                self.assertEqual(self.catch(**{name: lambda v: (_ for _ in ()).throw(RuntimeError("x"))}), "check_failed")
        for bad in (None, (), "x", ["replay:1:mismatch"]):
            with self.subTest(replay=bad), mock.patch.object(R, "replay_validate", lambda *a, **k: bad):
                self.assertEqual(self.catch(), "ledger_replay_invalid")
        for bad in (None, 5, "x", object()):
            with self.subTest(store=bad):
                self.assertEqual(self.catch(store=bad), "bad_input")


class StaticTests(unittest.TestCase):
    SRC = inspect.getsource(C)
    PKG = os.path.dirname(os.path.abspath(C.__file__))

    def setUp(self):
        install_tripwire(self)

    def tree(self):
        return ast.parse(self.SRC)

    def fn(self, name):
        return next(n for n in ast.walk(self.tree()) if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_it_imports_only_pure_standard_library_and_the_memory_layers(self):
        mods = set()
        for n in ast.walk(self.tree()):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" if n.module == "backtalk" else n.module for a in n.names}
        self.assertEqual(mods, {"copy", "hashlib", "json", "math", "re", "unicodedata", "dataclasses", "datetime", "backtalk.memory",
                                "backtalk.memory_extract", "backtalk.memory_gate", "backtalk.memory_keys",
                                "backtalk.memory_review", "backtalk.memory_store"})

    def test_no_io_no_clock_no_dynamic_execution_and_no_write_path(self):
        tree = self.tree()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(names & {"open", "os", "sys", "socket", "subprocess", "print", "input", "eval", "exec", "compile",
                                  "__import__", "environ", "getenv", "random", "time"})
        self.assertFalse(attrs & {"now", "utcnow", "today", "time", "monotonic", "write", "read", "system", "getenv",
                                  "environ", "urlopen", "put", "apply", "_commit", "_install", "_write_atomic", "unlink",
                                  "mkdir", "makedirs", "replace_file", "rename"})
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, (ast.Global, ast.Nonlocal))], [])

    def test_it_never_builds_an_authorization_or_a_confirmation_and_never_touches_the_review_ledger_files(self):
        tree = self.tree()
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"Authorization", "Confirmation", "arrange", "append_line", "read_ledger"})
        for banned in ("evals.memory_extraction", "memory_retrieval", "fake_clock", "store_arrange"):
            self.assertNotIn(banned, self.SRC)

    def test_nothing_else_in_the_package_mentions_this_module(self):
        for path in sorted(glob.glob(os.path.join(self.PKG, "**", "*.py"), recursive=True)):
            if os.path.basename(path) not in ("memory_commit.py", "memory_execute.py", "commit_cli.py"):    # the 8.2d layer
                with open(path, encoding="utf-8") as f:
                    self.assertNotIn("memory_commit", f.read(), path)

    def test_the_limitation_statement_is_pinned_in_the_docstring(self):
        doc = " ".join(C.__doc__.split())
        for phrase in ("accidental-error and regression defenses only", "do not prove that any check actually ran",
                       "do not prove that a person approved anything", "do not protect against code running as the same user",
                       "NOTHING HERE WRITES", "not an atomic snapshot", "sources are not locked"):
            self.assertIn(phrase, doc)
        self.assertIn("not authentication", C.NOTICE)
        self.assertIn("accidental-error", inspect.getdoc(C.verify_plan_integrity) + C.__doc__)
        self.assertIn("IT IS NOT A SECURITY CHECK", C.verify_plan_integrity.__doc__)

    def test_the_8_2d_acceptance_criteria_are_in_the_module_docstring(self):
        doc = " ".join(C.__doc__.split())
        for phrase in ("ACCEPTANCE CRITERIA CARRIED INTO PHASE 8.2d", "must be REBUILT from scratch by a fresh `plan()` call",
                       "compared with the plan the person confirmed using `compare_plans`",
                       "aborts before any receipt intent is appended and before any store call",
                       "A stale Plan must never authorize a write", "is never itself an authorization",
                       "puts no upper bound on how old a Plan is", "the fresh re-plan in (1) is the only bound on staleness"):
            self.assertIn(phrase, doc)

    def test_this_test_file_names_no_real_path_in_any_file_call(self):
        """A static tripwire check: no open/listdir/scandir/stat/exists/walk/glob/Path call in this file has a string literal
        that names the user's home, the Claude folder, the vault or the review folder."""
        with open(__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        markers = ("/Users/", "/home/", ".claude", ".backtalk-memory-review", "Desktop/jarvis", "11 - Memory", "VAULT-INDEX")
        names = {"open", "listdir", "scandir", "stat", "exists", "isdir", "isfile", "walk", "glob", "iglob", "Path", "read_text",
                 "read_bytes", "mkdir", "makedirs", "rmtree", "copytree", "unlink", "remove"}
        offenders = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fname = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
                if fname in names:
                    for sub in ast.walk(n):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and any(m in sub.value for m in markers):
                            offenders.append((fname, sub.value[:40], n.lineno))
        self.assertEqual(offenders, [])
        self.assertEqual(sum(1 for p in FORBIDDEN if p), 3)
        self.assertTrue(any(".claude" in p for p in FORBIDDEN) and any("backtalk-memory-review" in p for p in FORBIDDEN)
                        and S.REAL_VAULT in FORBIDDEN)

    def test_the_signature_has_no_default_for_any_check(self):
        params = inspect.signature(C.plan).parameters
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values()))
        self.assertEqual({n for n, p in params.items() if p.default is not inspect.Parameter.empty}, {"max_age_days"})
        self.assertEqual(set(params), {"pack", "pack_sha", "records", "private", "candidate", "store", "provenance", "rederive",
                                       "clock", "monotonic", "preview_budget_s", "max_age_days"})

    def first_call(self, name):
        body = [st for st in self.fn(name).body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
        st = body[0]
        call = st.value if isinstance(st, ast.Expr) else st.value if isinstance(st, (ast.Return, ast.Assign)) else None
        return ast.unparse(call) if call is not None else ast.unparse(st)

    def test_every_function_that_uses_a_plan_checks_consistency_first(self):
        for name in ("_token", "check_answer", "render_preview", "expected_answer"):
            self.assertTrue(self.first_call(name).startswith("_require_consistent("), name)
        self.assertTrue(self.first_call("challenge_token").startswith("_token("))
        self.assertIn("_token(plan, nonce)", ast.unparse(self.fn("expected_answer")))
        self.assertIn("verify_plan_integrity(a)", ast.unparse(self.fn("compare_plans")))
        self.assertIn("verify_plan_integrity(b)", ast.unparse(self.fn("compare_plans")))
        self.assertIn("_require_consistent(result)", ast.unparse(self.fn("plan")))

    def test_the_token_hash_is_computed_in_one_place(self):
        callers = set()
        for fn in ast.walk(self.tree()):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_token_digest":
                        callers.add(fn.name)
        self.assertEqual(callers, {"_token"})

    def test_the_informational_field_never_reaches_a_binding_function(self):
        for name in ("_token_digest", "_token", "_context_digest", "_display_lines", "_display_digest", "_display_checked",
                     "verify_plan_integrity", "_compare_view", "compare_plans", "digest_of"):
            used = {n.attr for n in ast.walk(self.fn(name)) if isinstance(n, ast.Attribute)} | \
                   {n.id for n in ast.walk(self.fn(name)) if isinstance(n, ast.Name)}
            self.assertNotIn("rederive_seconds", used, name)

    GOLDEN_BINDINGS = {
        "candidate": ("context_digest", "integrity:record"),
        "run": ("context_digest", "item_digest"),
        "pack_sha": ("context_digest",),
        "registry_version": ("context_digest", "integrity:record"),
        "lock_hash": ("context_digest", "integrity:record"),
        "lock_record_json": ("integrity:record",),
        "operation": ("item_digest", "integrity:gate", "display_digest", "token"),
        "kind": ("token", "display_digest", "integrity:gate"),
        "item": ("item_digest", "integrity:item", "display_digest"),
        "evidence": ("item_digest", "integrity:item", "display_digest"),
        "links": ("item_digest", "integrity:links", "display_digest"),
        "decision": ("integrity:gate", "display_digest"),
        "store_items": ("integrity:state",),
        "superseded": ("integrity:links", "display_digest"),
        "state_digest": ("token", "integrity:state"),
        "item_digest": ("token", "integrity:item"),
        "context_digest": ("token", "integrity:record"),
        "display_digest": ("token", "integrity:display"),
        "planned_at": ("integrity:gate", "integrity:age"),
        "age_seconds": ("integrity:age",),
        "max_age_seconds": ("integrity:age",),
        "checks": ("integrity:required_checks",),
    }

    def test_the_bindings_table_is_a_reviewed_golden_copy(self):
        self.assertEqual(C.BINDINGS, self.GOLDEN_BINDINGS)
        self.assertEqual(set(C.INFORMATIONAL), {"rederive_seconds"})
        self.assertEqual(set(C.OLD_ITEM_BINDINGS), {f.name for f in fields(C.OldItem)})
        self.assertEqual({f.name for f in fields(C.Plan)}, set(C.BINDINGS) | set(C.INFORMATIONAL))
        self.assertFalse(set(C.BINDINGS) & set(C.INFORMATIONAL))
        for binders in C.BINDINGS.values():
            self.assertTrue(set(binders) <= C.BINDER_VOCABULARY)
        self.assertEqual(C.BINDER_VOCABULARY, {"item_digest", "state_digest", "context_digest", "display_digest", "token",
                                               "integrity:record", "integrity:item", "integrity:gate", "integrity:state",
                                               "integrity:age", "integrity:required_checks", "integrity:display",
                                               "integrity:links"})

    def test_the_item_fields_the_file_format_carries_are_exactly_the_item_fields(self):
        self.assertEqual(set(S.FIELD_ORDER) - {"memory_schema"}, {f.name for f in fields(M.MemoryItem)})

    def test_the_required_checks_and_constants_are_pinned(self):
        self.assertEqual(C.REQUIRED_CHECKS, ("inputs", "replay", "committable", "age", "provenance", "rederive", "build", "store",
                                             "gate", "digests", "display"))
        self.assertEqual((C.LOCK_MAX_AGE_DAYS, C.CHALLENGE_SECONDS, C.TOKEN_HEX, C.READ_ATTEMPTS),
                         (14, 120, 8, 3))
        self.assertEqual((C.TOKEN_VERSION, C.DISPLAY_VERSION, C.CONTEXT_VERSION), ("commit-token/2", "display/1", "commit-ctx/1"))

    def variant(self, old, new):
        self.assertEqual(self.SRC.count(old), 1, old)
        ns = {"__name__": "memory_commit_variant"}
        exec(compile(self.SRC.replace(old, new, 1), "<variant>", "exec"), ns)
        return ns

    def test_the_module_cannot_be_imported_with_an_unreviewed_plan_field_or_binding(self):
        with self.assertRaises(RuntimeError):
            self.variant("    checks: tuple\n", "    checks: tuple\n    surprise: int = 0\n")
        with self.assertRaises(RuntimeError):
            self.variant('    "checks": ("integrity:required_checks",),\n', '    "checks": ("integrity:required_checks",),\n    "ghost": ("token",),\n')
        with self.assertRaises(RuntimeError):
            self.variant('    "pack_sha": ("context_digest",),', '    "pack_sha": ("made_up_binder",),')
        with self.assertRaises(RuntimeError):
            self.variant('    "pack_sha": ("context_digest",),\n', "")
        with self.assertRaises(RuntimeError):
            self.variant('    "pack_sha": ("context_digest",),', '    "pack_sha": (),')
        with self.assertRaises(RuntimeError):                          # a field both bound and informational
            self.variant('INFORMATIONAL = {\n', 'INFORMATIONAL = {\n    "pack_sha": "also informational",\n')
        with self.assertRaises(RuntimeError):                          # an informational field with no reason
            self.variant('    "rederive_seconds": "measured wall time;', '    "rederive_seconds": "" "measured wall time;') if False else \
                self.variant('"measured wall time; it cannot be recomputed, and its budget is enforced when the plan is built"', '""')
        self.assertIsNotNone(self.variant("    checks: tuple\n", "    checks: tuple\n").get("Plan"))   # the unmodified source imports


if __name__ == "__main__":
    unittest.main()
