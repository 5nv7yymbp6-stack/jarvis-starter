"""Memory 2.0, Phase 8.2c-1: the snapshot manifest (backtalk/memory_manifest.py). Pure value layer: schema, strict
parsing, validation against a snapshot, and the pinned empty encoding. Synthetic temp data only. These are integrity
checks, not authentication: nothing here claims to show who produced a manifest."""
import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import tempfile
import unittest

from backtalk import memory as M
from backtalk import memory_manifest as MM
from backtalk import memory_store as S

DAY = "2026-01-10T00:00:00Z"
EMPTY_STATE = "52647f02f82bcd18f407dc1bf8416ff6558b15de102610c3f219b86e3f073482"
EMPTY_SNAPSHOT = "a37591a81e08fc3bdab5adb94c338772b7e70f88c80eaff21158f4a31ed52a7d"
EMPTY_TEXT = ('{"count":0,"files":[],"nonces":[],"schema":"memory-manifest/1","snapshot_digest":"%s",'
              '"state_digest":"%s"}' % (EMPTY_SNAPSHOT, EMPTY_STATE))
NONCE = "0123456789abcdef0123456789abcdef"
ROW = (NONCE, "mem_000000000001", DAY)
LOG_LINE = "%s %s %s\n" % ROW


def item(n, **over):
    d = {"id": f"mem_{n:012x}", "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": DAY, "updated": DAY, "source": {"kind": "user", "ref": "turn 41"}, "confidence": "stated",
         "valid_from": DAY, "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))
        self.store = S.MemoryStore(self.root)

    def populated(self):
        if getattr(self, "_snap", None) is not None:
            return self._snap
        self.store.put(item(1), "I live in San Jose now")
        self.store.put(item(2, attribute="employer", value="Acme"), "I work at Acme")
        with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:
            f.write(LOG_LINE)
        snap = self.store.snapshot()
        self.assertTrue(snap.ok, snap)
        self._snap = snap
        return snap


class BuildAndRoundTrip(Case):
    def test_M4_the_empty_manifest_is_the_pinned_text(self):
        snap = self.store.snapshot()                                   # folder absent: ok, empty
        self.assertTrue(snap.ok)
        m = MM.build(snap)
        self.assertEqual((m.files, m.nonces, m.count), ((), (), 0))
        self.assertEqual(MM.canonical_text(m), EMPTY_TEXT)
        self.assertEqual(MM.parse(EMPTY_TEXT), m)

    def test_M1_round_trip_is_a_single_canonical_line(self):
        snap = self.populated()
        m = MM.build(snap)
        text = MM.canonical_text(m)
        self.assertEqual(MM.parse(text), m)
        self.assertNotIn("\n", text)
        self.assertTrue(text.isascii())
        self.assertEqual(m.count, 2)
        self.assertEqual(m.nonces, (ROW,))                          # the complete row, not just its hash
        self.assertEqual([n for n, _ in m.files], sorted(n for n, _ in m.files))
        MM.validate_against(m, snap)
        self.assertEqual(json.loads(text)["files"], [[n, h] for n, h in snap.files])      # a list of pairs
        self.assertEqual(json.loads(text)["nonces"], [list(ROW)])                         # full rows, as lists
        self.assertEqual(MM.parse(text).nonces, (ROW,))

    def test_build_refuses_a_snapshot_that_is_not_ok(self):
        self.store = S.MemoryStore(os.path.join(self.root, "nope"))
        bad = self.store.snapshot()
        self.assertFalse(bad.ok)
        with self.assertRaises(MM.ManifestError) as cm:
            MM.build(bad)
        self.assertEqual((cm.exception.reason, cm.exception.details), ("snapshot_not_ok", (bad.reason,)))
        with self.assertRaises(MM.ManifestError) as cm:
            MM.build("not a snapshot")
        self.assertEqual((cm.exception.reason, cm.exception.details), ("snapshot_not_ok", ("not_a_snapshot",)))
        locked = S.Snapshot(False, "lock_busy", None, None, None, None, None, None)
        with self.assertRaises(MM.ManifestError) as cm:
            MM.build(locked)
        self.assertEqual(cm.exception.details, ("lock_busy",))
        ok_but_flagged = S.Snapshot(False, "ok", None, (), (), (), None, None)         # ok=False is what refuses
        with self.assertRaises(MM.ManifestError):
            MM.build(ok_but_flagged)

    def test_the_digests_are_recomputed_not_copied(self):
        snap = self.populated()
        forged = S.Snapshot(True, "ok", snap.origin, snap.files, snap.items, snap.nonces, "0" * 64, "1" * 64)
        m = MM.build(forged)
        self.assertEqual((m.state_digest, m.snapshot_digest), (snap.state_digest, snap.snapshot_digest))
        with self.assertRaises(MM.ManifestError) as cm:
            MM.validate_against(m, forged)                             # the snapshot's own fields disagree
        self.assertEqual((cm.exception.reason, cm.exception.details), ("manifest_mismatch", ("snapshot_fields",)))


class StrictParsing(Case):
    def good(self):
        return MM.to_dict(MM.build(self.populated()))

    def refuse(self, text, code=None):
        with self.assertRaises(MM.ManifestError) as cm:
            MM.parse(text)
        self.assertEqual(cm.exception.reason, "bad_manifest")
        if code:
            self.assertEqual(cm.exception.details, (code,))

    def dump(self, d, **kw):
        return json.dumps(d, **kw)

    def test_M2_non_canonical_spellings_are_refused(self):
        d = self.good()
        canon = MM.canonical_text(MM.build(self.populated()))
        self.refuse(json.dumps(d), "not_canonical")                                       # spaces, insertion order
        self.refuse(canon + "\n", "not_canonical")
        self.refuse(" " + canon, "not_canonical")
        self.refuse(canon.replace('"count"', '"c\\u006funt"'), "not_canonical")           # an escaped key name
        d["nonces"] = [list(ROW)]
        self.refuse(json.dumps(d, sort_keys=True, indent=2), "not_canonical")

    def test_M2_duplicate_keys_extra_and_missing_keys_wrong_types(self):
        canon = MM.canonical_text(MM.build(self.populated()))
        self.refuse(canon[:-1] + ',"count":2}', "json")                                   # duplicate key
        d = self.good()
        for mutate, code in (
                (lambda x: x.update(extra=1), "keys"),
                (lambda x: x.pop("nonces"), "keys"),
                (lambda x: x.update(schema="memory-manifest/2"), "schema"),
                (lambda x: x.update(files={"a": "b"}), "shape"),
                (lambda x: x.update(files="not a list"), "shape"),
                (lambda x: x.update(files=[["mem_000000000001.md"]]), "shape"),
                (lambda x: x.update(files=[["mem_000000000001.md", "a" * 64, "x"]]), "shape"),
                (lambda x: x.update(files=["ab"]), "shape"),
                (lambda x: x.update(files=[{"a": 1, "b": 2}]), "shape"),
                (lambda x: x.update(nonces="abc"), "shape"),
                (lambda x: x.update(nonces={}), "shape"),
                (lambda x: x.update(nonces=[NONCE]), "shape"),                       # a bare token is not a row
                (lambda x: x.update(nonces=[[NONCE, "mem_000000000001"]]), "shape"),
                (lambda x: x.update(nonces=[list(ROW) + ["x"]]), "shape"),
                (lambda x: x.update(nonces=[{"a": 1, "b": 2, "c": 3}]), "shape"),
                (lambda x: x.update(count=99), "count"),
                (lambda x: x.update(count=True), "count"),
                (lambda x: x.update(count=2.0), "count"),                # equal in value, wrong in type
                (lambda x: x.update(count="2"), "count"),
                (lambda x: x.update(state_digest="0" * 64), "digest"),
                (lambda x: x.update(snapshot_digest="0" * 64), "digest"),
        ):
            bad = copy.deepcopy(d)
            mutate(bad)
            with self.subTest(code=code, keys=sorted(bad)):
                self.refuse(json.dumps(bad, sort_keys=True, separators=(",", ":")), code)
        for text, code in (("", "json"), ("null", "keys"), ("[]", "keys"), ("{", "json"), ('{"count":NaN}', "json"),
                           ('{"count":Infinity}', "json"), (5, "type"), (None, "type"), (b"{}", "type")):
            with self.subTest(text=repr(text)):
                self.refuse(text, code)

    def test_a_count_of_true_is_refused_even_when_one_file_makes_it_equal(self):
        self.store.put(item(1), "I live in San Jose now")
        snap = self.store.snapshot()
        d = MM.to_dict(MM.build(snap))
        self.assertEqual(d["count"], 1)
        d["count"] = True                                                # True == 1 in Python
        self.refuse(json.dumps(d, sort_keys=True, separators=(",", ":")), "count")
        d["count"] = 1.0
        self.refuse(json.dumps(d, sort_keys=True, separators=(",", ":")), "count")

    def test_the_two_json_hooks_name_their_refusals(self):
        with self.assertRaisesRegex(ValueError, "^duplicate_key$"):
            MM._no_duplicate_keys([("a", 1), ("a", 2)])
        self.assertEqual(MM._no_duplicate_keys([("a", 1), ("b", 2)]), {"a": 1, "b": 2})
        with self.assertRaisesRegex(ValueError, "^non_finite$"):
            MM._refuse_constant("NaN")

    def test_M2_malformed_entries_and_ordering_are_refused(self):
        d = self.good()
        for files in ([["mem_000000000001.md", "A" * 64]], [["notes.md", "a" * 64]], [["mem_000000000001.md", "a" * 63]],
                      [["mem_000000000001.md", "a" * 64], ["mem_000000000001.md", "a" * 64]],
                      [[1, 2]], [["mem_000000000001.md", 7]]):
            bad = dict(d, files=files, count=len(files))
            with self.subTest(files=files):
                self.refuse(json.dumps(bad, sort_keys=True, separators=(",", ":")), "bad_manifest_entries")
        rev = dict(d, files=list(reversed(d["files"])))
        self.refuse(json.dumps(rev, sort_keys=True, separators=(",", ":")), "not_canonical")   # order is canonical
        for nonces in ([["x" * 32, ROW[1], ROW[2]]], [list(ROW), list(ROW)], [[NONCE.upper(), ROW[1], ROW[2]]],
                       [[NONCE, "mem_00000000000", ROW[2]]], [[NONCE, ROW[1], "2026-13-45T00:00:00Z"]],
                       [[NONCE, ROW[1], 7]], [list(ROW), [NONCE, "mem_000000000002", ROW[2]]]):   # repeated hash
            bad = dict(d, nonces=nonces)
            with self.subTest(nonces=nonces):
                self.refuse(json.dumps(bad, sort_keys=True, separators=(",", ":")), "bad_nonces")
        rev = dict(d, nonces=[["f" * 32, ROW[1], ROW[2]], list(ROW)])
        self.refuse(json.dumps(rev, sort_keys=True, separators=(",", ":")), "digest")     # unsorted rows: digest differs


class RowFieldsInTheManifest(Case):
    """A change to any one field of an authorization row is visible: in the parsed text and against a snapshot."""

    CHANGES = {"hash32": (NONCE, "0123456789abcdef0123456789abcde0"),
               "memory_id": ("mem_000000000001", "mem_000000000009"),
               "confirmed_at": (DAY, "2026-01-10T00:00:01Z")}

    def log_with(self, row):
        with open(os.path.join(self.store.dir, ".authorizations"), "w") as f:
            f.write("%s %s %s\n" % row)

    def test_a_manifest_text_with_one_row_field_edited_is_refused(self):
        text = MM.canonical_text(MM.build(self.populated()))
        for field, (old, new) in self.CHANGES.items():
            edited = text.replace(f'"{old}"', f'"{new}"', 1) if field != "hash32" else text.replace(old, new, 1)
            with self.subTest(field=field):
                self.assertNotEqual(edited, text)
                with self.assertRaises(MM.ManifestError) as cm:
                    MM.parse(edited)
                self.assertEqual((cm.exception.reason, cm.exception.details), ("bad_manifest", ("digest",)))

    def test_a_manifest_fails_validation_after_any_row_field_changes(self):
        snap = self.populated()
        manifest = MM.build(snap)
        MM.validate_against(manifest, snap)
        for field, row in (("hash32", (self.CHANGES["hash32"][1], ROW[1], ROW[2])),
                           ("memory_id", (ROW[0], self.CHANGES["memory_id"][1], ROW[2])),
                           ("confirmed_at", (ROW[0], ROW[1], self.CHANGES["confirmed_at"][1]))):
            self.log_with(row)
            after = self.store.snapshot()
            with self.subTest(field=field):
                self.assertTrue(after.ok, after)
                self.assertEqual(after.state_digest, snap.state_digest)               # files are untouched
                self.assertNotEqual(after.snapshot_digest, snap.snapshot_digest)
                with self.assertRaises(MM.ManifestError) as cm:
                    MM.validate_against(manifest, after)
                self.assertEqual((cm.exception.reason, cm.exception.details), ("manifest_mismatch", ("nonces",)))
        self.log_with(ROW)
        MM.validate_against(manifest, self.store.snapshot())


class AgainstASnapshot(Case):
    def test_M3_a_manifest_that_differs_in_any_field_is_refused(self):
        snap = self.populated()
        m = MM.build(snap)
        MM.validate_against(m, snap)
        for field, value in (("files", m.files[:1]), ("nonces", ()), ("state_digest", "f" * 64),
                             ("snapshot_digest", "e" * 64)):
            other = MM.Manifest(**{**m.__dict__, field: value})
            with self.subTest(field=field), self.assertRaises(MM.ManifestError) as cm:
                MM.validate_against(other, snap)
            self.assertEqual((cm.exception.reason, cm.exception.details), ("manifest_mismatch", (field,)))

    def test_a_manifest_of_one_state_does_not_validate_against_another(self):
        before = self.populated()
        m = MM.build(before)
        self.store.put(item(3, attribute="hobby", value="chess"), "I play chess")
        after = self.store.snapshot()
        with self.assertRaises(MM.ManifestError) as cm:
            MM.validate_against(m, after)
        self.assertEqual(cm.exception.reason, "manifest_mismatch")

    def test_validate_refuses_non_manifests_and_failed_snapshots(self):
        snap = self.populated()
        with self.assertRaises(MM.ManifestError) as cm:
            MM.validate_against({"files": []}, snap)
        self.assertEqual((cm.exception.reason, cm.exception.details), ("bad_manifest", ("type",)))
        failed = S.Snapshot(False, "lock_busy", None, None, None, None, None, None)
        with self.assertRaises(MM.ManifestError) as cm:
            MM.validate_against(MM.build(snap), failed)
        self.assertEqual((cm.exception.reason, cm.exception.details), ("snapshot_not_ok", ("lock_busy",)))
        with self.assertRaises(MM.ManifestError) as cm:
            MM.validate_against(MM.build(snap), "not a snapshot")
        self.assertEqual((cm.exception.reason, cm.exception.details), ("snapshot_not_ok", ("not_a_snapshot",)))


class ErrorAndValueShape(unittest.TestCase):
    def test_the_error_text_is_the_reason_then_the_codes(self):
        self.assertEqual(str(MM.ManifestError("x")), "x")
        self.assertEqual(str(MM.ManifestError("x", ("a", "b"))), "x: a, b")
        e = MM.ManifestError("x", ["a"])
        self.assertEqual((e.reason, e.details), ("x", ("a",)))

    def test_a_manifest_is_a_frozen_value(self):
        m = MM.parse(EMPTY_TEXT)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            m.files = ()
        self.assertEqual(m.count, 0)


class PurityAndScope(unittest.TestCase):
    SRC = inspect.getsource(MM)

    def test_it_holds_no_digest_formula_and_no_io(self):
        tree = ast.parse(self.SRC)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"sha256", "hashlib", "AUTH_VERSION", "open", "listdir", "flock", "unlink", "write",
                                 "mkdir", "makedirs", "replace", "rename", "environ", "now", "time"}, used)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"json", "dataclasses.dataclass", "backtalk.memory_store"})

    def test_the_limitation_statement_is_in_the_docstring(self):
        doc = " ".join(MM.__doc__.split())
        for phrase in ("integrity checks, not authentication", "any code running as the same user",
                       "an observation made at one moment"):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main()
