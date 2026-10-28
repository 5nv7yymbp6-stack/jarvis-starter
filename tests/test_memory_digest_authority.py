"""Memory 2.0, Phase 8.2c-1: ONE authoritative digest implementation. The store defines `manifest_digest` and
`snapshot_digest`; the planning layer and the manifest layer call them and carry no formula of their own. Golden values
below were captured from the UNMODIFIED store before the refactor; the reference derivation in these tests uses plain
hashlib/json and imports no store code, so the check is not circular."""
import ast
import glob
import hashlib
import inspect
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

from backtalk import memory as M
from backtalk import memory_commit as C
from backtalk import memory_manifest as MM
from backtalk import memory_store as S
from tests.store_arrange import arrange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "backtalk")
DAY = "2026-01-10T00:00:00Z"

# captured from the unmodified store (state_digest on an absent folder, and on two put items)
GOLDEN_EMPTY = "52647f02f82bcd18f407dc1bf8416ff6558b15de102610c3f219b86e3f073482"
GOLDEN_ONE = "60ff17bad1984c1a85828d80c4fee529663a9a2765237421170cea4d37c8a30c"
GOLDEN_THREE = "d71e06f4f349f5cd61cbedcfefd70ad36c103f8ec04d99f40dc2934fa55a149f"
GOLDEN_STORE_TWO_ITEMS = "92e0a851d9148d64960682b3b5be53390050f7a561008f0a14fdec4163363bed"
GOLDEN_SNAPSHOT_EMPTY = "a37591a81e08fc3bdab5adb94c338772b7e70f88c80eaff21158f4a31ed52a7d"
# two files and two COMPLETE rows (hash32, memory_id, confirmed_at), derived with plain hashlib/json; the token-only
# digest this replaces was 96fffc3797c7431ca9d5d77eb4242c89a51ea58f5317b99c6cbaff9d58e99e86
GOLDEN_SNAPSHOT_TWO = "94ed3ea93074ae2a682605a539ffbd4f6a6f8abad4b5be7c4a979c1f5b131007"
ROW_A = ("1" * 32, "mem_000000000001", "2026-01-10T00:00:00Z")
ROW_B = ("2" * 32, "mem_0123456789ab", "2026-02-01T12:30:45Z")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def reference_state(entries):
    """Independent re-derivation of the state digest: plain hashlib and json."""
    body = {"v": "auth/1", "files": sorted([list(e) for e in entries])}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def reference_snapshot(entries, rows):
    body = {"schema": "memory-snapshot/1", "files": sorted([list(e) for e in entries]),
            "nonces": sorted([list(r) for r in rows])}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def item(n, **over):
    d = {"id": f"mem_{n:012x}", "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": DAY, "updated": DAY, "source": {"kind": "user", "ref": "turn 41"}, "confidence": "stated",
         "valid_from": DAY, "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


CORPUS_THREE = [["mem_000000000001.md", sha(b"one")], ["mem_0123456789ab.md", sha(b"two")],
                ["mem_ffffffffffff.md", sha("café".encode())]]


class Definitions(unittest.TestCase):
    def test_D1_the_empty_and_fixed_corpora_equal_values_captured_before_the_refactor(self):
        self.assertEqual(S.manifest_digest([]), GOLDEN_EMPTY)
        self.assertEqual(S.manifest_digest(CORPUS_THREE[:1]), GOLDEN_ONE)
        self.assertEqual(S.manifest_digest(CORPUS_THREE), GOLDEN_THREE)
        self.assertEqual(S.snapshot_digest([], []), GOLDEN_SNAPSHOT_EMPTY)
        self.assertEqual(S.snapshot_digest([["mem_000000000001.md", "a" * 64], ["mem_0123456789ab.md", "b" * 64]],
                                           [ROW_A, ROW_B]), GOLDEN_SNAPSHOT_TWO)
        self.assertEqual(S.snapshot_digest([["mem_000000000001.md", "a" * 64], ["mem_0123456789ab.md", "b" * 64]],
                                           [list(ROW_B), list(ROW_A)]), GOLDEN_SNAPSHOT_TWO)       # lists, any order

    def test_D2_an_independent_reference_agrees_including_non_ascii_content(self):
        rnd = random.Random(8)
        for k in range(40):
            names = rnd.sample([f"mem_{i:012x}.md" for i in range(1, 60)], rnd.randrange(0, 8))
            entries = [[n, sha(bytes(rnd.randrange(256) for _ in range(rnd.randrange(0, 20))) + "é’".encode())]
                       for n in names]
            nonces = [("%032x" % rnd.getrandbits(128), "mem_%012x" % rnd.getrandbits(48),
                       "2026-%02d-%02dT%02d:%02d:%02dZ" % (rnd.randrange(1, 13), rnd.randrange(1, 29),
                                                           rnd.randrange(24), rnd.randrange(60), rnd.randrange(60)))
                      for _ in range(rnd.randrange(0, 4))]
            self.assertEqual(S.manifest_digest(entries), reference_state(entries))
            self.assertEqual(S.snapshot_digest(entries, nonces), reference_snapshot(entries, nonces))

    def test_D3_order_is_decided_in_one_place_and_malformed_input_is_refused(self):
        a = S.manifest_digest(CORPUS_THREE)
        for perm in itertools.permutations(CORPUS_THREE):
            self.assertEqual(S.manifest_digest(list(perm)), a)
            self.assertEqual(S.manifest_digest(tuple(tuple(e) for e in perm)), a)
        bad = ([["mem_000000000001.md", "a" * 64]] * 2, [["notes.md", "a" * 64]], [["mem_000000000001.md", "A" * 64]],
               [["mem_000000000001.md", "a" * 63]], [["mem_000000000001.md"]], [[1, 2]], 7, [["mem_000000000001.md", None]])
        for entries in bad:
            with self.subTest(entries=repr(entries)[:40]), self.assertRaisesRegex(ValueError, "^bad_manifest_entries$"):
                S.manifest_digest(entries)
        good = list(ROW_A)
        for nonces in (["x"], ["1" * 32], [3], 5, [good[:2]], [good + ["x"]], [[good[0], good[1], 7]],
                       [["A" * 32, good[1], good[2]]], [[good[0], "mem_00000000000", good[2]]],
                       [[good[0], "MEM_000000000001", good[2]]], [[good[0], good[1], "2026-13-45T00:00:00Z"]],
                       [[good[0], good[1], "2026-01-10 00:00:00Z"]], [[good[0], good[1], "2026-01-10T00:00:00"]],
                       [[good[0] + " ", good[1], good[2]]], [[good[0], good[1] + " x", good[2]]],
                       [ROW_A, ROW_A], [ROW_A, (ROW_A[0], "mem_000000000002", ROW_A[2])]):
            with self.subTest(nonces=nonces), self.assertRaisesRegex(ValueError, "^bad_nonces$"):
                S.snapshot_digest([], nonces)

    def test_D5_state_and_snapshot_digests_never_coincide_and_nonces_move_only_the_snapshot_digest(self):
        s, n = S.manifest_digest(CORPUS_THREE), S.snapshot_digest(CORPUS_THREE, [])
        self.assertNotEqual(s, n)
        self.assertEqual(S.manifest_digest(CORPUS_THREE), s)
        self.assertNotEqual(S.snapshot_digest(CORPUS_THREE, [ROW_A]), n)
        self.assertNotEqual(S.snapshot_digest([], []), S.manifest_digest([]))

    def test_D6_a_changed_byte_name_or_membership_changes_both_digests(self):
        base = [list(e) for e in CORPUS_THREE]
        variants = [
            [[base[0][0], sha(b"ONE")]] + base[1:],                       # one byte of content
            [["mem_000000000009.md", base[0][1]]] + base[1:],             # a name
            base[1:],                                                     # a removed file
            base + [["mem_000000000002.md", sha(b"x")]],                  # an added file
        ]
        for v in variants:
            self.assertNotEqual(S.manifest_digest(v), S.manifest_digest(base))
            self.assertNotEqual(S.snapshot_digest(v, []), S.snapshot_digest(base, []))


class NonceRowDigest(unittest.TestCase):
    """Each field of an authorization row is covered by `snapshot_digest`, independently; the row order is not."""

    FILES = CORPUS_THREE

    def digest(self, *rows):
        return S.snapshot_digest(self.FILES, list(rows))

    def test_changing_only_the_hash_changes_the_digest(self):
        self.assertNotEqual(self.digest(ROW_A), self.digest(("3" * 32,) + ROW_A[1:]))

    def test_changing_only_the_memory_id_changes_the_digest(self):
        self.assertNotEqual(self.digest(ROW_A), self.digest((ROW_A[0], "mem_000000000002", ROW_A[2])))

    def test_changing_only_confirmed_at_changes_the_digest(self):
        self.assertNotEqual(self.digest(ROW_A), self.digest(ROW_A[:2] + ("2026-01-10T00:00:01Z",)))

    def test_every_field_change_is_distinct_and_leaves_the_state_digest_alone(self):
        variants = {self.digest(ROW_A), self.digest(("3" * 32,) + ROW_A[1:]),
                    self.digest((ROW_A[0], "mem_000000000002", ROW_A[2])),
                    self.digest(ROW_A[:2] + ("2026-01-10T00:00:01Z",)), self.digest()}
        self.assertEqual(len(variants), 5)
        self.assertEqual(S.manifest_digest(self.FILES), S.manifest_digest(self.FILES))   # the state digest has no rows

    def test_row_order_does_not_change_the_digest_and_rows_are_sorted_as_whole_tuples(self):
        self.assertEqual(self.digest(ROW_A, ROW_B), self.digest(ROW_B, ROW_A))
        rows = S.normalized_nonce_rows([ROW_B, ROW_A])
        self.assertEqual(rows, [list(ROW_A), list(ROW_B)])

    def test_the_repeated_hash_is_refused_even_when_the_other_fields_differ(self):
        for other in ((ROW_A[0], "mem_000000000002", ROW_A[2]), ROW_A[:2] + ("2026-01-10T00:00:01Z",), ROW_A):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "^bad_nonces$"):
                self.digest(ROW_A, other)

    def test_a_row_is_tuple_or_list_of_exactly_three_strings(self):
        self.assertEqual(self.digest(ROW_A), S.snapshot_digest(self.FILES, [list(ROW_A)]))
        self.assertEqual(S.normalized_nonce_rows([]), [])
        self.assertEqual(S.snapshot_digest([], []), GOLDEN_SNAPSHOT_EMPTY)           # the empty encoding is unchanged


class AgainstTheStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        existed = os.path.exists(real)
        self.addCleanup(lambda: self.assertEqual(os.path.exists(real), existed, "the real vault was touched"))
        self.store = S.MemoryStore(self.root)

    def agree(self):
        snap = self.store.snapshot()
        self.assertTrue(snap.ok, snap)
        self.assertEqual(snap.state_digest, self.store.state_digest())
        self.assertEqual(snap.state_digest, C.digest_of(self.store.list_items()))
        self.assertEqual(snap.state_digest, S.manifest_digest(snap.files))
        self.assertEqual(snap.snapshot_digest, S.snapshot_digest(snap.files, snap.nonces))
        self.assertEqual(MM.build(snap).state_digest, snap.state_digest)
        return snap

    def test_D4_the_store_the_planner_and_the_manifest_agree_in_every_state(self):
        self.assertEqual(self.agree().state_digest, GOLDEN_EMPTY)                     # absent folder
        os.makedirs(self.store.dir)
        self.assertEqual(self.agree().state_digest, GOLDEN_EMPTY)                     # empty folder (no lock yet)
        self.store.put(item(1), "I live in San Jose now")
        self.agree()
        self.store.put(item(2, attribute="employer", value="Café “Acme”"), "I work at Café Acme")
        self.assertEqual(self.agree().state_digest, GOLDEN_STORE_TWO_ITEMS)           # captured before the refactor
        # a record the gate would never create (history that already ended), written the way the store tests do
        arrange(self.store, item(5, attribute="employer", value="Initech", status="superseded",
                                 valid_from="2025-01-01T00:00:00Z", created="2025-01-01T00:00:00Z",
                                 updated="2025-06-01T00:00:00Z", valid_to="2025-06-01T00:00:00Z"),
                "I work at Initech")
        self.agree()


class Authority(unittest.TestCase):
    def test_A1_digest_of_carries_no_formula_of_its_own(self):
        src = inspect.getsource(C.digest_of)
        self.assertIn("manifest_digest", src)
        for banned in ("json.dumps", "sort_keys", "AUTH_VERSION", '"files"', "separators"):
            self.assertNotIn(banned, src)

    def test_A2_the_body_construction_exists_once_in_the_package(self):
        hits = []
        for path in sorted(glob.glob(os.path.join(PKG, "*.py"))):
            text = source(path)
            if '{"v": AUTH_VERSION, "files"' in text or '{"v": S.AUTH_VERSION, "files"' in text:
                hits.append(os.path.basename(path))
        self.assertEqual(hits, ["memory_store.py"])
        for name in ("memory_commit.py", "memory_manifest.py"):
            tree = ast.parse(source(os.path.join(PKG, name)))
            used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                   {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            self.assertNotIn("AUTH_VERSION", used, name)

    def test_A4_the_import_graph_among_the_memory_modules_has_no_cycle(self):
        names = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, "memory*.py"))}
        graph = {}
        for name in sorted(names):
            tree = ast.parse(source(os.path.join(PKG, name + ".py")))
            edges = set()
            for n in ast.walk(tree):                                      # function-local imports included
                if isinstance(n, ast.ImportFrom):
                    mod = n.module or ""
                    if n.level:                                           # relative: from . import x / from .x import y
                        mod = ("backtalk." + mod) if mod else "backtalk"
                    if mod == "backtalk":
                        edges |= {a.name for a in n.names} & names
                    elif mod.startswith("backtalk."):
                        edges |= {mod.split(".")[1]} & names
                elif isinstance(n, ast.Import):
                    edges |= {a.name.split(".")[1] for a in n.names if a.name.startswith("backtalk.")} & names
            graph[name] = edges - {name}
        self.assertIn("memory_manifest", graph)
        color, stack = {}, []

        def visit(node):
            color[node] = 1
            stack.append(node)
            for nxt in sorted(graph[node]):
                if color.get(nxt) == 1:
                    self.fail("import cycle: " + " -> ".join(stack[stack.index(nxt):] + [nxt]))
                if nxt not in color:
                    visit(nxt)
            stack.pop()
            color[node] = 2
        for node in sorted(graph):
            if node not in color:
                visit(node)
        self.assertFalse(graph["memory_store"] & {"memory_manifest", "memory_commit"}, graph["memory_store"])
        self.assertNotIn("memory_commit", graph["memory_manifest"])
        self.assertEqual(graph["memory_manifest"], {"memory_store"})

    PROBE = r"""
import json, sys
bad = []
def hook(event, args):
    if event == "open":
        path, mode, flags = args
        m = mode or ""
        if any(c in m for c in "wax+") or (isinstance(flags, int) and flags & (0x1 | 0x2 | 0x200 | 0x400 | 0x8)):
            bad.append(["open", str(path), m])
    elif event in ("os.mkdir", "os.rename", "os.remove", "os.rmdir", "shutil.move", "os.chmod"):
        bad.append([event, [str(a) for a in args]])
sys.addaudithook(hook)
sys.path.insert(0, sys.argv[1])
for name in sys.argv[2].split(","):
    __import__("backtalk." + name)
import backtalk
print(json.dumps({"bad": bad, "file": backtalk.__file__, "path0": sys.path[:1]}))
"""

    def test_A3_every_import_order_works_in_an_isolated_interpreter(self):
        """`-I` ignores PYTHONPATH and user-site and does not put the working directory on sys.path; `-B` stops
        bytecode writes. The project root is passed as an ABSOLUTE ARGUMENT. A decoy `backtalk` package in the working
        directory (and on PYTHONPATH) raises if isolation ever fails."""
        with tempfile.TemporaryDirectory() as decoy:
            os.makedirs(os.path.join(decoy, "backtalk"))
            with open(os.path.join(decoy, "backtalk", "__init__.py"), "w") as f:
                f.write("raise RuntimeError('the decoy package was imported')\n")
            env = {"PYTHONPATH": decoy, "PATH": os.environ.get("PATH", "")}
            for order in itertools.permutations(("memory_store", "memory_manifest", "memory_commit")):
                with self.subTest(order=order):
                    proc = subprocess.run([sys.executable, "-I", "-B", "-c", self.PROBE, ROOT, ",".join(order)],
                                          cwd=decoy, env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    out = json.loads(proc.stdout)
                    self.assertEqual(out["bad"], [])
                    self.assertTrue(out["file"].startswith(ROOT + os.sep), out["file"])
                    self.assertEqual(out["path0"], [ROOT])


if __name__ == "__main__":
    unittest.main()
