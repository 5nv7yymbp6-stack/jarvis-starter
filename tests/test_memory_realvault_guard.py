"""The test guard for the original real-vault path (tests/realvault_guard.py), tested on a SYNTHETIC SENTINEL folder in a temp
directory. The real vault path is never probed here: the guard is told to protect the sentinel as well, and every claim
below is about the sentinel.

These tests pin exactly which filesystem operations the guard stops BEFORE they run, which it does not stop, and that
ordinary temp-directory work and the store's own platform check still function while it is armed. A guard of this kind is a
tripwire inside one Python process, not a sandbox; the uncovered operations are asserted (not hidden) in the last class."""
import contextlib
import glob
import os
import pathlib
import shutil
import tempfile
import unittest

from backtalk import memory_store as S
from tests import realvault_guard as G
from tests.fake_clock import FakeClock
from tests.test_memory_commit_plan import NOWS


class SentinelCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = os.path.realpath(tmp.name)
        self.prot = os.path.join(self.base, "Protected")
        self.free = os.path.join(self.base, "free")
        os.mkdir(self.prot)
        os.mkdir(self.free)
        os.mkdir(os.path.join(self.prot, "sub"))
        self.f = os.path.join(self.prot, "f.txt")
        with open(self.f, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.chmod(self.f, 0o644)
        self.pre_fd = os.open(self.prot, os.O_RDONLY)                 # opened BEFORE the guard is armed (see the gap tests)
        self.addCleanup(os.close, self.pre_fd)
        self.cwd = os.getcwd()
        self.touched = G.arm(self, also_protect=[self.prot])

    @contextlib.contextmanager
    def unguarded(self):
        saved = list(G.ARMED)
        G.ARMED.clear()
        try:
            yield
        finally:
            G.ARMED.extend(saved)

    def state(self):
        with self.unguarded():
            tree = sorted(os.path.relpath(os.path.join(r, n), self.prot) for r, ds, fs in os.walk(self.prot)
                          for n in ds + fs)
            st = os.stat(self.f)
            with open(self.f, encoding="utf-8") as fh:
                return (tree, fh.read(), st.st_mode, st.st_mtime_ns, st.st_nlink, os.getcwd())


class StoppedBeforeTheyRunTests(SentinelCase):
    """Every operation here is refused with the guard's error AND leaves the sentinel exactly as it was."""

    def ops(self):
        p, f = self.prot, self.f
        sub = os.path.join(p, "sub")
        return {
            # audit-event operations
            "builtin open (read)": lambda: open(f),
            "builtin open (write)": lambda: open(os.path.join(p, "new.txt"), "w"),
            "os.open": lambda: os.open(f, os.O_RDONLY),
            "os.listdir": lambda: os.listdir(p),
            "os.scandir": lambda: list(os.scandir(p)),
            "os.walk": lambda: list(os.walk(p)),
            "glob.glob": lambda: glob.glob(p + "/*"),
            "os.mkdir": lambda: os.mkdir(os.path.join(p, "d")),
            "os.makedirs": lambda: os.makedirs(os.path.join(p, "e", "g")),
            "os.chmod": lambda: os.chmod(f, 0o600),
            "os.chown": lambda: os.chown(f, -1, -1),
            "os.utime": lambda: os.utime(f, (1, 1)),
            "os.truncate": lambda: os.truncate(f, 0),
            "os.link (source inside)": lambda: os.link(f, os.path.join(self.free, "hard")),
            "os.link (target inside)": lambda: os.link(os.path.join(self.free, "src"), os.path.join(p, "hard")),
            "os.symlink (target inside)": lambda: os.symlink(f, os.path.join(p, "ln")),
            "os.rename (from inside)": lambda: os.rename(f, os.path.join(self.free, "moved")),
            "os.rename (into)": lambda: os.rename(os.path.join(self.free, "src"), os.path.join(p, "moved")),
            "os.replace": lambda: os.replace(f, os.path.join(self.free, "moved")),
            "os.remove": lambda: os.remove(f),
            "os.unlink": lambda: os.unlink(f),
            "os.rmdir": lambda: os.rmdir(sub),
            "os.chdir": lambda: os.chdir(p),
            "shutil.copy out": lambda: shutil.copy(f, self.free),
            "shutil.copy in": lambda: shutil.copy(os.path.join(self.free, "src"), p),
            "shutil.rmtree": lambda: shutil.rmtree(sub),
            "pathlib read_text": lambda: pathlib.Path(f).read_text(),
            "pathlib write_text": lambda: pathlib.Path(f).write_text("y"),
            "pathlib iterdir": lambda: list(pathlib.Path(p).iterdir()),
            "tempfile.mkstemp(dir=)": lambda: tempfile.mkstemp(dir=p),
            # metadata operations with NO audit event: stopped by the wrappers
            "os.stat": lambda: os.stat(f),
            "os.lstat": lambda: os.lstat(p),
            "os.access": lambda: os.access(f, os.R_OK),
            "os.readlink": lambda: os.readlink(f),
            "os.statvfs": lambda: os.statvfs(p),
            "os.pathconf": lambda: os.pathconf(p, "PC_NAME_MAX"),
            "os.path.exists": lambda: os.path.exists(f),
            "os.path.isdir": lambda: os.path.isdir(p),
            "os.path.isfile": lambda: os.path.isfile(f),
            "os.path.islink": lambda: os.path.islink(f),
            "os.path.getsize": lambda: os.path.getsize(f),
            "os.path.getmtime": lambda: os.path.getmtime(f),
            "os.path.realpath": lambda: os.path.realpath(f),
            "os.path.samefile": lambda: os.path.samefile(f, f),
            "pathlib exists": lambda: pathlib.Path(f).exists(),
            "pathlib stat": lambda: pathlib.Path(p).stat(),
            "pathlib resolve": lambda: pathlib.Path(f).resolve(),
            "pathlib is_dir": lambda: pathlib.Path(p).is_dir(),
        }

    def test_each_operation_is_refused_before_it_runs_and_the_sentinel_is_unchanged(self):
        with self.unguarded():
            with open(os.path.join(self.free, "src"), "w", encoding="utf-8") as fh:
                fh.write("s")
        before = self.state()
        for label, op in self.ops().items():
            del self.touched[:]
            with self.subTest(label):
                with self.assertRaises(AssertionError, msg=label) as cm:
                    op()
                self.assertIn("protected real-vault path", str(cm.exception))
                self.assertTrue(self.touched, label)                     # recorded even if a caller swallows the error
        self.assertEqual(self.state(), before)
        with self.unguarded():
            self.assertEqual(sorted(os.listdir(self.free)), ["src"])     # nothing was created or moved out either

    def test_a_swallowed_refusal_is_still_recorded(self):
        try:
            os.stat(self.f)
        except AssertionError:
            pass
        self.assertEqual(self.touched, ["os.stat"])

    def test_dot_dot_upper_case_bytes_pathlike_and_trailing_slash_forms_are_all_recognized(self):
        forms = [os.path.join(self.free, "..", "Protected", "f.txt"), self.f.upper(), os.fsencode(self.f),
                 pathlib.Path(self.f), self.prot + os.sep, os.path.join(self.prot, ".", "sub", "..")]
        for form in forms:
            del self.touched[:]
            with self.subTest(form=repr(form)):
                with self.assertRaises(AssertionError):
                    os.stat(form)
                self.assertEqual(self.touched, ["os.stat"])

    def test_a_sibling_with_the_same_prefix_is_not_protected(self):
        sibling = self.prot + "-sibling"
        with self.unguarded():
            os.mkdir(sibling)
        self.assertTrue(os.path.isdir(sibling))
        self.assertEqual(self.touched, [])


class OrdinaryWorkStillWorksTests(SentinelCase):
    def test_temp_directory_work_is_untouched_by_the_guard(self):
        path = os.path.join(self.free, "a.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.chmod(path, 0o600)
        self.assertTrue(os.path.exists(path) and os.path.isfile(path) and not os.path.islink(path))
        self.assertEqual(os.stat(path).st_size, 2)
        os.rename(path, path + ".2")
        self.assertEqual(os.listdir(self.free), ["a.txt.2"])
        os.makedirs(os.path.join(self.free, "x", "y"))
        shutil.rmtree(os.path.join(self.free, "x"))
        os.remove(path + ".2")
        self.assertEqual(self.touched, [])

    def test_the_stores_own_platform_check_still_passes_while_armed(self):
        vault = os.path.join(self.free, "vault")
        os.mkdir(vault)
        store = S.MemoryStore(vault, clock=FakeClock(NOWS))
        snap = store.snapshot()
        self.assertNotEqual(snap.reason, "platform_unsupported")
        self.assertTrue(snap.ok, snap.reason)
        self.assertEqual(self.touched, [])

    def test_building_a_temp_store_never_resolves_the_original_path(self):
        self.assertNotEqual(S.REAL_VAULT, G.ORIGINAL)                    # the decoy the guard installs
        S.MemoryStore(self.free, clock=FakeClock(NOWS))
        self.assertEqual(self.touched, [])

    def test_the_check_itself_never_touches_the_filesystem(self):
        self.assertTrue(G.is_protected(G.ORIGINAL))
        self.assertTrue(G.is_protected(os.path.join(G.ORIGINAL, "11 - Memory", "a.md")))
        self.assertFalse(G.is_protected(self.free))
        self.assertFalse(G.is_protected(None))
        self.assertFalse(G.is_protected(7))
        self.assertEqual(self.touched, [])                               # (a stat of the original would have been recorded)


class GuardLifecycleTests(unittest.TestCase):
    def test_arming_installs_wrappers_registers_them_and_cleanup_restores_everything(self):
        originals = {n: getattr(os, n) for n in G._WRAPPED if hasattr(os, n)}
        memberships = {n: {s for s in G._SUPPORT_SETS if originals[n] in getattr(os, s)} for n in originals}
        case = unittest.TestCase()
        case.tearDown = lambda: None
        G.arm(case)
        try:
            self.assertTrue(G.ARMED)
            for n, real in originals.items():
                wrapped = getattr(os, n)
                self.assertIsNot(wrapped, real)
                self.assertIs(wrapped.__wrapped__, real)
                for s in memberships[n]:                                  # platform checks that test membership still pass
                    self.assertIn(wrapped, getattr(os, s), (n, s))
        finally:
            case.doCleanups()
        self.assertFalse(G.ARMED)
        for n, real in originals.items():
            self.assertIs(getattr(os, n), real)
            for s in G._SUPPORT_SETS:
                self.assertEqual(real in getattr(os, s), s in memberships[n], (n, s))
        self.assertEqual(G.ORIGINAL, S.REAL_VAULT)                        # the decoy patch is undone too

    def test_the_guard_is_off_unless_a_test_armed_it(self):
        self.assertFalse(G.ARMED)
        os.stat(tempfile.gettempdir())                                     # a plain stat works when nothing is armed


class KnownGapTests(SentinelCase):
    """What the guard does NOT stop. These assert the gap, so the limit is stated rather than assumed. They touch only the
    synthetic sentinel."""

    def test_a_symlink_elsewhere_that_points_into_the_protected_tree_is_not_recognized(self):
        link = os.path.join(self.free, "alias")
        with self.unguarded():
            os.symlink(self.prot, link)
        os.stat(os.path.join(link, "f.txt"))                              # reaches the sentinel through the alias
        self.assertEqual(self.touched, [])

    def test_a_relative_name_used_with_dir_fd_is_not_recognized(self):
        parent_fd = os.open(self.base, os.O_RDONLY)
        try:
            os.stat("Protected", dir_fd=parent_fd)
            os.stat(os.path.join("Protected", "f.txt"), dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        self.assertEqual(self.touched, [])

    def test_operations_on_an_already_open_descriptor_are_not_recognized(self):
        os.fstat(self.pre_fd)
        self.assertEqual(sorted(os.listdir(self.pre_fd)), ["f.txt", "sub"])
        self.assertEqual(self.touched, [])


if __name__ == "__main__":
    unittest.main()
