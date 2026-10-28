"""Memory 2.0, Phase 8.2c-1: `MemoryStore.snapshot()` (backtalk/memory_store.py). Locking, read-only enforcement, the empty
and changed-during-snapshot states, and the single-byte-source races. Every test runs on a throwaway vault in a temp
directory; nothing here reads or writes the real vault, and a guard in the base class proves it.

WHAT THESE TESTS CLAIM. A successful snapshot is a consistent OBSERVATION at the completed validation boundary. Tests
that name a "limit" document what is NOT detected; they are not proof of a stronger guarantee. Nothing here claims a
directory "stayed empty throughout": only that listings and metadata comparisons agreed (or did not).
"""
import ast
import collections
import dataclasses
import fcntl
import hashlib
import inspect
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from backtalk import memory as M
from backtalk import memory_store as S

EV = "I live in San Jose now"
DAY = "2026-01-10T00:00:00Z"
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
FILE = "mem_%012x.md"


def mid(n):
    return f"mem_{n:012x}"


def ev(value="San Jose"):
    """Evidence that supports `value`, so the gate accepts it."""
    return f"I live in {value} now"


def item(n=1, **over):
    d = {"id": mid(n), "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": DAY, "updated": DAY, "source": {"kind": "user", "ref": "turn 41"}, "confidence": "stated",
         "valid_from": DAY, "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


def tree_state(root):
    """Everything observable about a tree without following links: names, types, modes, owners, sizes, mtime/ctime,
    inodes, link counts, link targets and the sha256 of every regular file's bytes (atime is excluded: an OS may touch
    it on a read, and that is not an action of the snapshot)."""
    out = {}

    def entry(path, rel):
        st = os.lstat(path)
        rec = [st.st_mode, st.st_uid, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_nlink]
        kind = st.st_mode & 0o170000
        if kind == 0o120000:
            rec.append(os.readlink(path))
        elif kind == 0o100000:
            with open(path, "rb") as f:
                rec.append(hashlib.sha256(f.read()).hexdigest())
        out[rel] = tuple(rec)
        if kind == 0o040000:
            for name in sorted(os.listdir(path)):
                entry(os.path.join(path, name), rel + "/" + name)

    if os.path.lexists(root):
        entry(root, ".")
    return out


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def count_fds():
    return len(os.listdir("/dev/fd"))


class Hook(S._Ops):
    """A deterministic test double for the store's OS-call seam. It runs real calls and lets a test run an action at one
    exact point. Every `open` is checked: a write flag is a test failure, so each test also proves read-only opens."""

    def __init__(self):
        self.opens, self.read_total = [], collections.Counter()
        self.listdir_calls = 0
        self.before_open = {}                 # name -> action, run once, before the real open
        self.on_open_error = {}               # name -> action, run once, after a FileNotFoundError from open
        self.after_listdir = {}               # call number -> action, run after that listing was computed
        self.after_flock = None
        self.at_eof = None                    # action run when a read returns b"" (end of that file)
        self.after_first_read = None
        self.on_validated = None
        self.reads = collections.defaultdict(list)   # fd -> [chunks] while open
        self.streams = []                     # the complete bytes read through each descriptor, in close order
        self.fstat_override = None            # callable(st, fd) -> st
        self.read_sizes = []                  # len of every chunk returned, in order (b"" is 0)
        self.short = False                    # every read returns b"" (a short read)
        self.endless = False                  # every read returns one byte and never reaches the end
        self.stat_raise = {}                  # path -> exception raised by stat(path)
        self.stat_fake = {}                   # path -> callable(real_stat) -> replacement
        self.lstat_calls = collections.Counter()
        self.lstat_raise = {}                 # (name, nth call) -> exception
        self.lstat_fake = {}                  # (name, nth call) -> callable(real_stat) -> replacement

    def open(self, name, flags, dir_fd=None):
        assert not flags & WRITE_FLAGS, f"snapshot opened {name!r} with a write flag"
        self.opens.append((name, flags))
        action = self.before_open.pop(name, None)
        if action:
            action()
        try:
            return os.open(name, flags, dir_fd=dir_fd) if dir_fd is not None else os.open(name, flags)
        except FileNotFoundError:
            action = self.on_open_error.pop(name, None)
            if action:
                action()
            raise

    def listdir(self, fd):
        self.listdir_calls += 1
        names = os.listdir(fd)
        action = self.after_listdir.pop(self.listdir_calls, None)
        if action:
            action()
        return names

    def flock(self, fd, op):
        fcntl.flock(fd, op)
        if self.after_flock:
            self.after_flock()

    def stat(self, path, *args, **kw):
        if path in self.stat_raise:
            raise self.stat_raise[path]
        st = os.stat(path, *args, **kw)
        return self.stat_fake[path](st) if path in self.stat_fake else st

    def lstat(self, name, dir_fd=None):
        self.lstat_calls[name] += 1
        exc = self.lstat_raise.get((name, self.lstat_calls[name]))
        if exc:
            raise exc
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        fake = self.lstat_fake.get((name, self.lstat_calls[name]))
        return fake(st) if fake else st

    def read(self, fd, n):
        if self.endless:
            self.read_sizes.append(1)
            return b"x"
        chunk = b"" if self.short else os.read(fd, n)
        self.read_sizes.append(len(chunk))
        self.reads[fd].append(chunk)
        if chunk == b"" and self.at_eof:
            action, self.at_eof = self.at_eof, None
            action()
        if chunk and self.after_first_read:
            action, self.after_first_read = self.after_first_read, None
            action()
        return chunk

    def close(self, fd):
        if fd in self.reads:
            self.streams.append(b"".join(self.reads.pop(fd)))
        os.close(fd)

    def fstat(self, fd):
        st = os.fstat(fd)
        return self.fstat_override(st, fd) if self.fstat_override else st

    def validated(self):
        if self.on_validated:
            self.on_validated()


class FakeDirStat:
    """The real device/inode/mode of a directory with CONSTANT times and link count."""

    def __init__(self, st):
        self.st_dev, self.st_ino, self.st_mode, self.st_nlink = st.st_dev, st.st_ino, st.st_mode, 2
        self.st_mtime_ns = self.st_ctime_ns = 1_000_000_000_000_000_000


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        prod = os.path.realpath(S.REAL_VAULT)
        self.assertFalse(self.root == prod or self.root.startswith(prod + os.sep), "tests must never use the real vault")
        self.real_mem = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        self.real_existed = os.path.exists(self.real_mem)
        self.memdir = os.path.join(self.root, S.MEMORY_DIR)
        self.store = S.MemoryStore(self.root)

    def tearDown(self):
        self.assertEqual(os.path.exists(self.real_mem), self.real_existed, "the real vault was touched")

    # -- arranging (test-side writes, always OUTSIDE snapshot()) --
    def mkdir(self):
        os.makedirs(self.memdir, exist_ok=True)

    def lock(self):
        self.mkdir()
        with open(os.path.join(self.memdir, ".lock"), "ab"):
            pass

    def mkfile(self, n=1, **over):
        """A canonical memory file written directly (no lock, no `.lock` created)."""
        self.mkdir()
        S._write_atomic(os.path.join(self.memdir, FILE % n),
                        S.serialize(item(n, **over), ev(over.get("value", "San Jose"))))
        return FILE % n

    def put(self, n=1, **over):
        return self.store.put(item(n, **over), ev(over.get("value", "San Jose")))

    def write(self, name, data, mode="wb"):
        self.mkdir()
        with open(os.path.join(self.memdir, name), mode) as f:
            f.write(data)

    def path(self, name):
        return os.path.join(self.memdir, name)

    def swap_in(self, hook, snapshot_kwargs=None):
        with mock.patch.object(S, "_OPS", hook):
            return self.store.snapshot(**(snapshot_kwargs or {}))

    def bounded(self, fn, seconds=10):
        """Run fn on a thread and fail (rather than hang the run) if it does not finish."""
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
        t.start()
        t.join(seconds)
        self.assertFalse(t.is_alive(), "snapshot hung")
        return box["v"]

    def assertNotOk(self, snap, reason):
        self.assertFalse(snap.ok, snap)
        self.assertEqual(snap.reason, reason)
        for field in ("origin", "files", "items", "nonces", "state_digest", "snapshot_digest"):
            self.assertIsNone(getattr(snap, field), field)


# ---- locking ---------------------------------------------------------------------------------------

CHILD = ("import fcntl, os, sys\nfd = os.open(sys.argv[1], os.O_RDWR)\nfcntl.flock(fd, fcntl.LOCK_EX)\n"
         "sys.stdout.write('locked\\n'); sys.stdout.flush()\nsys.stdin.read()\n")


class LockingTests(Base):
    """Real flock calls, deterministic synchronisation (events and pipes). What each proves is in its docstring."""

    def pause_in_read(self):
        """A hook whose snapshot thread parks inside its first file read until `release` is set."""
        hook, entered, release = Hook(), threading.Event(), threading.Event()
        me = threading.current_thread()
        plain = hook.read

        def read(fd, n):
            if threading.current_thread() is not me and not entered.is_set():
                entered.set()
                self.assertTrue(release.wait(10))
            return plain(fd, n)
        hook.read = read
        return hook, entered, release

    def background_snapshot(self, hook):
        box = {}

        def run():
            with mock.patch.object(S, "_OPS", hook):
                box["snap"] = self.store.snapshot()
        t = threading.Thread(target=run, daemon=True)
        return t, box

    def test_shared_snapshots_coexist(self):
        """Proves LOCK_SH with LOCK_SH does not conflict (two open file descriptions, one process): a second snapshot
        succeeds while the first is parked holding its lock. Assumes nothing about other platforms."""
        self.put(1)
        hook, entered, release = self.pause_in_read()
        t, box = self.background_snapshot(hook)
        t.start()
        self.assertTrue(entered.wait(10))
        other = self.store.snapshot()
        release.set()
        t.join(10)
        self.assertTrue(other.ok and box["snap"].ok)
        self.assertEqual(other.snapshot_digest, box["snap"].snapshot_digest)

    def test_a_held_shared_lock_conflicts_with_exclusive_and_a_cooperating_writer_waits(self):
        """Proves (1) deterministically: while a snapshot holds LOCK_SH, a non-blocking LOCK_EX on the same `.lock` is
        refused; (2) a cooperating writer (`put`, which takes LOCK_EX) does not complete until the snapshot releases,
        and the snapshot returns the PRE-write state. (2) uses one bounded negative wait that can only produce a false
        pass, never a false failure. Does not cover writers that ignore flock."""
        self.put(1)
        before = self.store.snapshot()
        hook, entered, release = self.pause_in_read()
        t, box = self.background_snapshot(hook)
        t.start()
        self.assertTrue(entered.wait(10))
        probe = os.open(self.path(".lock"), os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)
        done = threading.Event()
        order, real_flock = [], fcntl.flock

        def spy(fd, op):
            if threading.current_thread().name == "writer" and op & fcntl.LOCK_EX:
                order.append("writer_requests_exclusive")
            return real_flock(fd, op)
        writer = threading.Thread(target=lambda: (self.put(2, attribute="employer", value="Acme"), done.set()),
                                  name="writer", daemon=True)
        with mock.patch.object(S.fcntl, "flock", spy):
            writer.start()
            self.assertFalse(done.wait(0.5), "the writer finished while a snapshot held the shared lock")
            self.assertEqual(order, ["writer_requests_exclusive"])
            release.set()
            writer.join(10)
        t.join(10)
        self.assertTrue(done.is_set())
        self.assertTrue(box["snap"].ok)
        self.assertEqual(box["snap"].snapshot_digest, before.snapshot_digest)     # the pre-write state
        after = self.store.snapshot()
        self.assertTrue(after.ok)
        self.assertNotEqual(after.snapshot_digest, before.snapshot_digest)
        self.assertEqual(len(after.items), 2)

    def test_the_existing_readers_take_the_exclusive_lock(self):
        """Ties the claim above to the code: `list_items` requests LOCK_EX, so a held LOCK_SH blocks it too."""
        self.put(1)
        ops = []
        real_flock = fcntl.flock
        with mock.patch.object(S.fcntl, "flock", lambda fd, op: (ops.append(op), real_flock(fd, op))[1]):
            self.store.list_items()
        self.assertIn(fcntl.LOCK_EX, ops)
        self.assertNotIn(fcntl.LOCK_SH, ops)

    def test_a_writer_holding_the_lock_gives_lock_busy_and_nothing_else(self):
        """Proves a held LOCK_EX is refused by the default non-blocking attempt with a stable reason, no data and no
        side effect, and that the same call succeeds once the writer releases."""
        self.put(1)
        before = tree_state(self.root)
        with self.store._locked(write=True):
            busy = self.bounded(self.store.snapshot)
        self.assertNotOk(busy, "lock_busy")
        self.assertEqual(tree_state(self.root), before)
        self.assertTrue(self.store.snapshot().ok)

    def test_a_lock_held_by_another_process_gives_lock_busy(self):
        """Proves the conflict holds across processes (a separate interpreter holds LOCK_EX, synchronised by a pipe)."""
        self.put(1)
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", CHILD, self.path(".lock")], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            self.assertNotOk(self.bounded(self.store.snapshot), "lock_busy")
        finally:
            child.stdin.close()
            child.wait(timeout=10)
            child.stdout.close()
        self.assertTrue(self.store.snapshot().ok)

    def test_block_true_waits_with_a_blocking_shared_request(self):
        """Proves `block=True` requests LOCK_SH without LOCK_NB, and the default requests it with LOCK_NB."""
        self.put(1)
        seen = []
        hook = Hook()
        hook.flock = lambda fd, op: (seen.append(op), fcntl.flock(fd, op))[1]
        self.assertTrue(self.swap_in(hook, {"block": True}).ok)
        self.assertTrue(self.swap_in(hook).ok)
        self.assertEqual(seen, [fcntl.LOCK_SH, fcntl.LOCK_SH | fcntl.LOCK_NB])

    def test_acquisition_failures_are_stable_and_create_nothing(self):
        """Proves an OS failure to open or lock `.lock` is `snapshot_unreadable`, with the tree untouched."""
        self.put(1)
        before = tree_state(self.root)

        hook = Hook()

        def boom_flock(fd, op):
            raise PermissionError(1, "denied")
        hook.flock = boom_flock
        self.assertNotOk(self.swap_in(hook), "snapshot_unreadable")

        hook = Hook()
        plain = hook.open

        def deny_lock(name, flags, dir_fd=None):
            if name == ".lock":
                raise PermissionError(13, "denied")
            return plain(name, flags, dir_fd)
        hook.open = deny_lock
        self.assertNotOk(self.swap_in(hook), "snapshot_unreadable")
        self.assertEqual(tree_state(self.root), before)

    def test_the_lock_and_every_descriptor_are_released_on_success_failure_and_exception(self):
        """Proves no descriptor leaks and no lock is left held: a following `put` is not blocked."""
        self.put(1)
        baseline = count_fds()
        self.assertTrue(self.store.snapshot().ok)
        with self.store._locked(write=True):
            self.assertNotOk(self.bounded(self.store.snapshot), "lock_busy")
        self.assertEqual(count_fds(), baseline)

        hook = Hook()
        hook.after_listdir[1] = lambda: (_ for _ in ()).throw(RuntimeError("injected"))
        with self.assertRaises(RuntimeError):
            self.swap_in(hook)
        self.assertEqual(count_fds(), baseline)
        self.assertEqual(self.bounded(lambda: self.put(3, attribute="employer", value="Acme")), mid(3))

    def test_concurrent_writers_and_a_snapshot_loop_only_ever_see_committed_states(self):
        """Invariants, not an order (scheduling is not deterministic): every successful snapshot is internally
        consistent, only ever grows, and never shows a file that no writer committed."""
        self.put(1)
        committed = {mid(1)}
        lock = threading.Lock()

        def writer(base):
            for k in range(4):
                n = base + k
                self.store.put(item(n, attribute=f"hobby{n}", value="chess"), ev("chess"))
                with lock:
                    committed.add(mid(n))
        threads = [threading.Thread(target=writer, args=(b,), daemon=True) for b in (10, 20, 30)]
        snaps, busy = [], 0
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            s = self.store.snapshot()
            if s.ok:
                snaps.append(s)
            else:
                self.assertEqual(s.reason, "lock_busy")
                busy += 1
        for t in threads:
            t.join(10)
        final = self.store.snapshot()
        self.assertTrue(final.ok)
        self.assertEqual({st.item.id for st in final.items}, committed)
        prev = set()
        for s in snaps + [final]:
            names = {n for n, _ in s.files}
            self.assertEqual(s.state_digest, S.manifest_digest(s.files))
            self.assertEqual(names, {st.item.id + ".md" for st in s.items})
            self.assertTrue(prev <= names, "a snapshot lost a file a previous one showed")
            self.assertTrue({n[:-3] for n in names} <= committed)
            prev = names


# ---- read-only enforcement -------------------------------------------------------------------------

class ReadOnlyTests(Base):
    """Every scenario takes a tree state before and after; a snapshot, successful or not, changes nothing."""

    MUTATORS = ("unlink", "remove", "rename", "replace", "rmdir", "mkdir", "makedirs", "truncate", "ftruncate",
                "write", "chmod", "fchmod", "chown", "fchown", "utime", "symlink", "link", "mkfifo", "fsync")

    def check(self, expected_reason, *, origin=None, extra_roots=(), snapshot_kwargs=None):
        before = {r: tree_state(r) for r in (self.root,) + tuple(extra_roots)}
        hook = Hook()
        refuse = mock.Mock(side_effect=AssertionError("snapshot called a mutating function"))
        patches = [mock.patch.object(os, n, refuse) for n in self.MUTATORS if hasattr(os, n)]
        patches += [mock.patch.object(tempfile, "mkstemp", refuse), mock.patch.object(shutil, "rmtree", refuse),
                    mock.patch.object(shutil, "move", refuse)]
        for p in patches:
            p.start()
        try:
            snap = self.bounded(lambda: self.swap_in(hook, snapshot_kwargs))
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(snap.reason, expected_reason, snap)
        self.assertEqual(snap.ok, expected_reason == "ok")
        if snap.ok:
            self.assertEqual(snap.origin, origin)
        else:
            self.assertNotOk(snap, expected_reason)
        self.assertEqual({r: tree_state(r) for r in before}, before, "the snapshot changed the filesystem")
        self.assertTrue(hook.opens, "the open spy never ran")
        return snap

    def test_missing_folder_is_an_empty_observation_and_creates_nothing(self):
        snap = self.check("ok", origin="absent")
        self.assertEqual((snap.files, snap.items, snap.nonces), ((), (), ()))
        self.assertFalse(os.path.exists(self.memdir))

    def test_missing_vault_root_is_not_an_empty_snapshot(self):
        self.store = S.MemoryStore(os.path.join(self.root, "no-such-vault"))
        before = tree_state(self.root)
        with mock.patch.object(S, "_OPS", Hook()):
            snap = self.store.snapshot()
        self.assertNotOk(snap, "store_root_missing")
        self.assertEqual(tree_state(self.root), before)

    def test_healthy_store(self):
        self.put(1)
        self.put(2, attribute="employer", value="Acme")
        snap = self.check("ok", origin="initialized")
        self.assertEqual(len(snap.items), 2)

    def test_missing_lock_with_an_empty_folder_creates_no_lock(self):
        self.mkdir()
        self.check("ok", origin="uninitialized")
        self.assertEqual(os.listdir(self.memdir), [])

    def test_missing_lock_with_files_is_refused(self):
        self.mkfile(1)
        self.check("lock_missing")
        self.assertEqual(os.listdir(self.memdir), [FILE % 1])

    def test_a_symlinked_memory_folder_is_refused_and_not_followed(self):
        elsewhere = os.path.join(self.root, "elsewhere")
        os.makedirs(elsewhere)
        with open(os.path.join(elsewhere, ".lock"), "wb"):
            pass
        os.symlink(elsewhere, self.memdir)
        self.check("root_symlink", extra_roots=(elsewhere,))

    def test_a_symlinked_vault_root_is_followed_by_design(self):
        self.put(1)
        link = os.path.join(self.tmp.name + "-link")
        os.symlink(self.root, link)
        self.addCleanup(lambda: os.path.lexists(link) and os.unlink(link))
        self.store = S.MemoryStore(link)
        self.check("ok", origin="initialized", extra_roots=(link,))

    def test_symlinked_memory_files_are_refused_inside_outside_and_dangling(self):
        self.put(1)
        outside = os.path.join(self.root, "outside.md")
        with open(outside, "wb") as f:
            f.write(b"x")
        for target in (self.path(FILE % 1), outside, os.path.join(self.root, "nowhere.md")):
            link = self.path(FILE % 2)
            os.symlink(target, link)
            with self.subTest(target=os.path.basename(target)):
                self.check("store_unsound", extra_roots=(outside,))
            os.unlink(link)

    def test_unexpected_files_are_refused(self):
        self.put(1)
        for name in ("notes.txt", ".hidden", ".tmp-abc.part", "mem_zzzzzzzzzzzz.md", "MEM_000000000009.md"):
            self.write(name, b"x")
            with self.subTest(name=name):
                self.check("store_unsound")
            os.unlink(self.path(name))

    def test_a_directory_or_fifo_in_place_of_a_file_is_refused_and_does_not_hang(self):
        self.put(1)
        os.mkdir(self.path(FILE % 2))
        self.check("store_unsound")
        os.rmdir(self.path(FILE % 2))
        os.mkfifo(self.path(FILE % 2))
        self.check("store_unsound")

    def test_a_hard_linked_file_is_refused(self):
        self.put(1)
        os.link(self.path(FILE % 1), os.path.join(self.root, "second-name"))      # mem_1 itself now has two links
        self.check("store_unsound")
        os.unlink(os.path.join(self.root, "second-name"))
        self.check("ok", origin="initialized")
        os.link(self.path(FILE % 1), self.path(FILE % 2))                         # and a link under another memory name
        self.check("store_unsound")

    def test_pending_committed_and_corrupt_journals_are_reported_and_never_recovered(self):
        self.put(1)
        self.mkfile(2, attribute="employer", value="Acme")          # a half-installed file beside the journal
        prepared = '{"state": "prepared", "ops": [{"file": "%s", "prior": null}]}\n' % (FILE % 2)
        for text, reason in ((prepared, "recovery_pending"),
                             (prepared.replace("prepared", "committed"), "recovery_pending"),
                             ("{not json", "journal_corrupt"), ('{"state": "weird", "ops": []}', "journal_corrupt"),
                             ('{"state": "prepared"}', "journal_corrupt")):
            self.write(".txn.json", text.encode())
            with self.subTest(text=text[:20]):
                self.check(reason)
                self.assertTrue(os.path.exists(self.path(FILE % 2)))
        os.unlink(self.path(".txn.json"))
        os.mkdir(self.path(".txn.json"))
        self.check("store_unsound")

    def test_non_canonical_and_mismatched_files_are_refused(self):
        name = self.mkfile(1)
        self.lock()
        good = read_bytes(self.path(name))
        for label, data in (("crlf", good.replace(b"\n", b"\r\n")), ("extra", good + b"\n"), ("empty", b""),
                            ("binary", b"\xff\xfe\x00"), ("huge", good + b"x" * (1 << 20))):
            self.write(name, data)
            with self.subTest(label):
                self.check("store_unsound")
        self.write(name, good)
        self.write(FILE % 2, good)                                   # the id inside does not match the name
        self.check("store_unsound")

    def test_authorization_log_problems(self):
        self.put(1)
        row = "0123456789abcdef0123456789abcdef mem_000000000001 2026-01-10T00:00:00Z\n"
        self.write(".authorizations", b"garbage\n")
        self.check("authorization_log_corrupt")
        self.write(".authorizations", (row + row).encode())
        self.check("authorization_log_corrupt")
        os.unlink(self.path(".authorizations"))
        os.mkdir(self.path(".authorizations"))
        self.check("store_unsound")
        os.rmdir(self.path(".authorizations"))
        self.write(".authorizations", row.encode())
        snap = self.check("ok", origin="initialized")
        self.assertEqual(snap.nonces, (("0123456789abcdef0123456789abcdef", "mem_000000000001",
                                        "2026-01-10T00:00:00Z"),))                # the complete row

    def test_a_lock_that_is_a_link_directory_or_fifo_is_refused(self):
        self.mkdir()
        os.symlink(self.path("target"), self.path(".lock"))
        self.check("store_unsound")
        os.unlink(self.path(".lock"))
        os.mkdir(self.path(".lock"))
        self.check("store_unsound")
        os.rmdir(self.path(".lock"))
        os.mkfifo(self.path(".lock"))
        self.check("store_unsound")

    def test_a_regular_file_in_place_of_the_memory_folder_is_refused(self):
        with open(self.memdir, "wb") as f:
            f.write(b"x")
        snap = self.check("store_unsound")
        self.assertEqual(snap.detail, (S.MEMORY_DIR,))

    @unittest.skipIf(os.geteuid() == 0, "permissions are not enforced for root")
    def test_permission_failures_are_snapshot_unreadable_and_change_nothing(self):
        self.put(1)
        for target in (self.memdir, self.path(FILE % 1)):
            os.chmod(target, 0)
            try:
                snap = self.swap_in(Hook())
            finally:
                os.chmod(target, 0o700 if target == self.memdir else 0o600)
            with self.subTest(target=os.path.basename(target)):
                self.assertNotOk(snap, "snapshot_unreadable")
        self.assertTrue(self.swap_in(Hook()).ok)

    def test_snapshot_source_never_references_a_writing_path(self):
        """Static: the method bodies never name recovery, locking-for-write, installing or any write primitive."""
        forbidden = {"_recover", "_locked", "_Lock", "_install", "_write_atomic", "_rollback", "_replace", "_commit",
                     "O_CREAT", "O_WRONLY", "O_RDWR", "O_TRUNC", "O_APPEND", "LOCK_EX", "makedirs", "unlink", "rename",
                     "mkstemp", "fsync", "put", "apply"}
        tree = ast.parse(inspect.getsource(S))
        bodies = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name in ("snapshot", "_snapshot", "_snapshot_absent", "_snapshot_result")]
        self.assertEqual(len(bodies), 4)
        for fn in bodies:
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
                   {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
            self.assertFalse(used & forbidden, (fn.name, used & forbidden))


# ---- empty and changed-during-snapshot states ----------------------------------------------------

class EmptyStateTests(Base):
    """E1-E16. Each claims only what the listings and metadata comparisons actually establish."""

    PAST = 1_000_000_000_000_000_000            # ns: a fixed time long before any test runs

    def set_dir_time(self, path):
        os.utime(path, ns=(self.PAST, self.PAST))

    def snap(self, hook):
        return self.bounded(lambda: self.swap_in(hook))

    def test_E1_absent_folder_returns_ok_absent_when_the_observations_agree(self):
        snap = self.snap(Hook())
        self.assertTrue(snap.ok)
        self.assertEqual(snap.origin, "absent")
        self.assertFalse(os.path.exists(self.memdir))
        self.assertEqual(snap.state_digest, S.manifest_digest([]))

    def test_E2_empty_folder_without_a_lock_returns_ok_uninitialized_when_the_observations_agree(self):
        self.mkdir()
        snap = self.snap(Hook())
        self.assertTrue(snap.ok)
        self.assertEqual(snap.origin, "uninitialized")
        self.assertFalse(os.path.exists(self.path(".lock")))

    def test_E3_lock_only_folder_returns_ok_initialized_with_the_same_digests(self):
        absent = self.snap(Hook())
        self.mkdir()
        bare = self.snap(Hook())
        self.lock()
        snap = self.snap(Hook())
        self.assertEqual((snap.ok, snap.origin), (True, "initialized"))
        self.assertEqual((snap.state_digest, snap.snapshot_digest), (absent.state_digest, absent.snapshot_digest))
        self.assertEqual((snap.state_digest, snap.snapshot_digest), (bare.state_digest, bare.snapshot_digest))

    def test_E4_a_file_appearing_between_the_listings_without_a_lock_is_not_ok(self):
        self.mkdir()
        hook = Hook()
        hook.after_listdir[1] = lambda: self.mkfile(1)
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E5_a_lock_appearing_between_the_listings_is_not_ok(self):
        self.mkdir()
        hook = Hook()
        hook.after_listdir[1] = self.lock
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E6_a_create_and_delete_between_the_listings_is_caught_by_the_metadata_comparison(self):
        """Claims only that the folder's recorded times differed at the two ends in this arrangement (its mtime was set
        far in the past first). It is not evidence about every create-and-delete sequence."""
        self.mkdir()
        self.set_dir_time(self.memdir)

        def churn():
            self.write("scratch", b"x")
            os.unlink(self.path("scratch"))
        hook = Hook()
        hook.after_listdir[1] = churn
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E7_files_present_without_a_lock_is_lock_missing(self):
        self.mkfile(1)
        self.assertNotOk(self.snap(Hook()), "lock_missing")

    def test_E8_a_file_added_by_a_non_cooperating_writer_after_the_first_listing_is_not_ok(self):
        self.lock()
        hook = Hook()
        hook.after_listdir[1] = lambda: self.mkfile(1)
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E9_the_folder_appearing_between_the_two_absence_probes_is_not_ok(self):
        os.makedirs(self.root, exist_ok=True)
        hook = Hook()
        hook.on_open_error[S.MEMORY_DIR] = self.mkdir
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E10_a_create_and_remove_of_the_folder_between_probes_moves_the_parent_times(self):
        self.set_dir_time(self.root)

        def churn():
            self.mkdir()
            os.rmdir(self.memdir)
        hook = Hook()
        hook.on_open_error[S.MEMORY_DIR] = churn
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_E11_a_missing_vault_root_is_not_an_empty_snapshot(self):
        self.store = S.MemoryStore(os.path.join(self.root, "absent-root"))
        self.assertNotOk(self.snap(Hook()), "store_root_missing")

    def test_E12_a_lock_that_is_a_link_directory_or_fifo_is_unsound_and_does_not_hang(self):
        self.mkdir()
        for make in (lambda: os.symlink(self.path("x"), self.path(".lock")), lambda: os.mkdir(self.path(".lock")),
                     lambda: os.mkfifo(self.path(".lock"))):
            make()
            self.assertNotOk(self.snap(Hook()), "store_unsound")
            if os.path.isdir(self.path(".lock")) and not os.path.islink(self.path(".lock")):
                os.rmdir(self.path(".lock"))
            else:
                os.unlink(self.path(".lock"))

    def test_E13_a_lock_replaced_after_it_was_taken_is_lock_replaced(self):
        self.put(1)
        hook = Hook()

        def replace_lock():
            self.write(".lock.new", b"")
            os.replace(self.path(".lock.new"), self.path(".lock"))
        hook.after_flock = replace_lock
        self.assertNotOk(self.snap(hook), "lock_replaced")

    def test_E13b_a_lock_removed_after_it_was_taken_is_lock_replaced(self):
        self.put(1)
        hook = Hook()
        hook.after_flock = lambda: os.unlink(self.path(".lock"))
        self.assertNotOk(self.snap(hook), "lock_replaced")

    def test_E13c_a_lock_replaced_after_the_first_check_is_caught_by_the_final_one(self):
        self.put(1)
        hook = Hook()

        def replace_lock():
            self.write(".lock.new", b"")
            os.replace(self.path(".lock.new"), self.path(".lock"))
        hook.after_listdir[1] = replace_lock
        self.assertNotOk(self.snap(hook), "lock_replaced")

    def test_E13d_a_lock_that_vanishes_at_the_final_check_is_lock_replaced(self):
        self.put(1)
        hook = Hook()
        hook.lstat_raise[(".lock", 2)] = FileNotFoundError(2, "gone")        # call 1: after flock; call 2: final
        self.assertNotOk(self.snap(hook), "lock_replaced")

    def test_E13e_a_lock_that_vanishes_before_classification_is_changed_during_read(self):
        self.put(1)
        hook = Hook()
        hook.after_listdir[1] = lambda: os.unlink(self.path(".lock"))
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, ())                                     # decided by the second listing

    def test_E13f_a_lock_swapped_and_put_back_is_still_caught_by_the_first_check(self):
        """The first identity check (right after the lock is taken) is independent of the final one: swap the lock
        file out and put the original inode back before the final check, and only the first check can see it."""
        self.put(1)
        hook = Hook()

        def swap_out():
            os.rename(self.path(".lock"), self.path(".lock.orig"))
            self.write(".lock", b"")

        def put_back():
            os.replace(self.path(".lock.orig"), self.path(".lock"))
        hook.after_flock = swap_out
        hook.after_listdir[1] = put_back
        self.assertNotOk(self.snap(hook), "lock_replaced")

    def test_E14a_a_change_after_the_validation_boundary_is_not_detected(self):
        """DOCUMENTED LIMITATION, not a guarantee. The snapshot is an observation made at the validation boundary; a
        write after that point cannot be seen by it. The result describes the pre-write state, and a fresh snapshot
        differs."""
        self.put(1)
        pre = self.store.snapshot()
        hook = Hook()
        hook.on_validated = lambda: self.mkfile(2, attribute="employer", value="Acme")
        snap = self.snap(hook)
        self.assertTrue(snap.ok)
        self.assertEqual((snap.state_digest, snap.snapshot_digest), (pre.state_digest, pre.snapshot_digest))
        fresh = self.store.snapshot()
        self.assertTrue(fresh.ok)
        self.assertNotEqual(fresh.snapshot_digest, snap.snapshot_digest)

    def test_E14b_a_snapshot_is_a_frozen_value_that_does_not_follow_the_store(self):
        self.put(1)
        snap = self.store.snapshot()
        digest = snap.snapshot_digest
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.ok = False
        self.put(2, attribute="employer", value="Acme")
        self.assertEqual(snap.snapshot_digest, digest)
        self.assertNotEqual(self.store.snapshot().snapshot_digest, digest)
        self.assertEqual({f.name for f in dataclasses.fields(snap)} & {"expires", "valid_until", "at", "time"}, set())

    def test_E14c_a_change_before_the_final_folder_comparison_is_caught_and_one_after_it_is_the_undetected_window(self):
        self.mkdir()
        self.set_dir_time(self.memdir)
        hook = Hook()
        hook.after_listdir[2] = lambda: self.write("late", b"x")      # listing 2 already returned []; D1 will differ
        self.assertNotOk(self.snap(hook), "changed_during_read")
        os.unlink(self.path("late"))
        self.set_dir_time(self.memdir)
        hook = Hook()
        hook.on_validated = lambda: self.write("later", b"x")         # after D1: the documented undetected window
        snap = self.snap(hook)
        self.assertTrue(snap.ok)                                       # asserted only to document the limit
        self.assertEqual(snap.files, ())

    def test_E16_decision_rule_only_equal_observations_give_ok(self):
        """Tests the DECISION RULE, not the filesystem: with the folder's recorded identity and times forced equal at
        both ends and both listings empty, the function returns ok. It shows the result depends only on those
        comparisons, which is exactly why they are the stated limit."""
        self.mkdir()
        hook = Hook()
        real_dir = os.stat(self.memdir)
        hook.fstat_override = lambda st, fd: FakeDirStat(st) if (st.st_dev, st.st_ino) == (real_dir.st_dev, real_dir.st_ino) \
            or st.st_mode & 0o170000 == 0o040000 else st
        hook.after_listdir[1] = lambda: (self.write("x", b"1"), os.unlink(self.path("x")))
        snap = self.snap(hook)
        self.assertTrue(snap.ok)
        self.assertEqual(snap.origin, "uninitialized")


# ---- single byte source and races -----------------------------------------------------------------

class SingleSourceTests(Base):
    def setUp(self):
        super().setUp()
        self.put(1)
        self.put(2, attribute="employer", value="Café “Acme”")

    def snap(self, hook):
        return self.bounded(lambda: self.swap_in(hook))

    def test_each_file_is_opened_once_and_read_once_and_digest_and_item_come_from_those_bytes(self):
        hook = Hook()
        snap = self.snap(hook)
        self.assertTrue(snap.ok)
        opened = collections.Counter(n for n, _ in hook.opens)
        for n in (FILE % 1, FILE % 2):
            self.assertEqual(opened[n], 1)
        for n, flags in hook.opens:
            if n.startswith("mem_"):
                self.assertTrue(flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK and flags & os.O_CLOEXEC)
                self.assertFalse(flags & WRITE_FLAGS)
        streams = hook.streams
        by_sha = {hashlib.sha256(b).hexdigest(): b for b in streams}
        for (name, sha), stored in zip(snap.files, snap.items):
            data = by_sha[sha]                                    # the hashed bytes ARE bytes that were read
            self.assertEqual(S.serialize(stored.item, stored.evidence).encode("utf-8"), data)
        self.assertEqual(sum(len(b) for b in streams), sum(os.path.getsize(self.path(n)) for n, _ in snap.files))

    def test_a_file_swapped_for_a_symlink_after_classification_and_before_open_is_unsound(self):
        hook = Hook()

        def swap():
            os.unlink(self.path(FILE % 1))
            os.symlink(self.path(FILE % 2), self.path(FILE % 1))
        hook.before_open[FILE % 1] = swap
        self.assertNotOk(self.snap(hook), "store_unsound")

    def test_a_file_replaced_by_another_regular_file_before_open_is_read_consistently(self):
        """Indistinguishable from a write that completed before the snapshot, so it is not an error: the NEW file's
        bytes are what the digest and the item both come from."""
        hook = Hook()
        new = S.serialize(item(1, value="Fresno"), "I moved to Fresno").encode("utf-8")

        def replace():
            self.write("incoming", new)
            os.replace(self.path("incoming"), self.path(FILE % 1))
        hook.before_open[FILE % 1] = replace
        snap = self.snap(hook)
        self.assertTrue(snap.ok)
        self.assertEqual(dict(snap.files)[FILE % 1], hashlib.sha256(new).hexdigest())
        self.assertEqual({s.item.id: s.item.value for s in snap.items}[mid(1)], "Fresno")

    def test_a_file_replaced_after_it_was_read_is_changed_during_read(self):
        hook = Hook()

        def replace():
            data = read_bytes(self.path(FILE % 1))
            self.write("incoming", data)
            os.replace(self.path("incoming"), self.path(FILE % 1))
        hook.at_eof = replace
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_bytes_appended_during_the_read_are_changed_during_read(self):
        hook = Hook()
        hook.after_first_read = lambda: self.write(FILE % 1, b"x", "ab")
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_a_file_deleted_after_listing_is_changed_during_read(self):
        hook = Hook()
        hook.after_listdir[1] = lambda: os.unlink(self.path(FILE % 2))
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_a_file_added_after_the_first_listing_is_changed_during_read(self):
        hook = Hook()
        hook.after_listdir[1] = lambda: self.mkfile(3, attribute="hobby", value="chess")
        self.assertNotOk(self.snap(hook), "changed_during_read")

    def test_the_memory_folder_renamed_away_and_replaced_is_root_replaced(self):
        hook = Hook()

        def swap():
            os.rename(self.memdir, self.memdir + ".old")
            os.mkdir(self.memdir)
        hook.after_listdir[2] = swap
        self.assertNotOk(self.snap(hook), "root_replaced")

    def test_the_result_never_contains_content_in_its_detail(self):
        self.write("notes.txt", b"SECRET-CONTENT")
        snap = self.snap(Hook())
        self.assertNotOk(snap, "store_unsound")
        self.assertEqual(snap.detail, ("notes.txt",))
        self.assertNotIn("SECRET", repr(snap))


class AuthorizationRowTests(Base):
    """The snapshot carries and hashes COMPLETE authorization rows: (hash32, memory_id, confirmed_at)."""

    H1, H2 = "0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"
    T1, T2 = "2026-01-10T00:00:00Z", "2026-01-10T00:00:01Z"

    def setUp(self):
        super().setUp()
        self.put(1)

    def log(self, *rows):
        self.write(".authorizations", "".join("%s %s %s\n" % r for r in rows).encode())
        snap = self.store.snapshot()
        self.assertTrue(snap.ok, snap)
        return snap

    def test_the_snapshot_holds_full_rows_sorted_regardless_of_file_order(self):
        a, b = (self.H1, mid(1), self.T1), (self.H2, mid(2), self.T2)
        forward, reverse = self.log(a, b), self.log(b, a)
        self.assertEqual(forward.nonces, (a, b))
        self.assertEqual(reverse.nonces, (a, b))
        self.assertEqual(forward.snapshot_digest, reverse.snapshot_digest)

    def test_each_row_field_alone_changes_snapshot_digest_but_not_state_digest(self):
        base = self.log((self.H1, mid(1), self.T1))
        for label, row in (("hash32", (self.H2, mid(1), self.T1)), ("memory_id", (self.H1, mid(2), self.T1)),
                           ("confirmed_at", (self.H1, mid(1), self.T2))):
            snap = self.log(row)
            with self.subTest(label):
                self.assertEqual(snap.nonces, (row,))
                self.assertEqual(snap.state_digest, base.state_digest)
                self.assertNotEqual(snap.snapshot_digest, base.snapshot_digest)
        self.assertEqual(self.log((self.H1, mid(1), self.T1)).snapshot_digest, base.snapshot_digest)

    def test_a_repeated_hash_is_corrupt_whatever_the_other_fields_are(self):
        for second in ((self.H1, mid(1), self.T1), (self.H1, mid(2), self.T1), (self.H1, mid(1), self.T2)):
            self.write(".authorizations", ("%s %s %s\n" * 2 % ((self.H1, mid(1), self.T1) + second)).encode())
            with self.subTest(second=second):
                self.assertNotOk(self.store.snapshot(), "authorization_log_corrupt")

    def test_malformed_rows_are_corrupt(self):
        for line in (f"{self.H1} {mid(1)}\n", f"{self.H1} {mid(1)} {self.T1} extra\n", f"{self.H1} mem_1 {self.T1}\n",
                     f"{self.H1.upper()} {mid(1)} {self.T1}\n", f"{self.H1} {mid(1)} 2026-13-45T00:00:00Z\n",
                     f"{self.H1} {mid(1)} 2026-01-10 00:00:00Z\n"):
            self.write(".authorizations", line.encode())
            with self.subTest(line=line):
                self.assertNotOk(self.store.snapshot(), "authorization_log_corrupt")


class PlatformTests(Base):
    """Each capability the snapshot needs, removed one at a time, gives `platform_unsupported` and touches nothing."""

    def check_unsupported(self):
        before = tree_state(self.root)
        snap = self.store.snapshot()
        self.assertNotOk(snap, "platform_unsupported")
        self.assertEqual(tree_state(self.root), before)

    def test_missing_function_support_is_refused_one_capability_at_a_time(self):
        self.put(1)
        for attr, func in (("supports_dir_fd", os.open), ("supports_dir_fd", os.stat),
                           ("supports_follow_symlinks", os.stat), ("supports_fd", os.listdir)):
            with self.subTest(attr=attr, func=func.__name__):
                with mock.patch.object(os, attr, set(getattr(os, attr)) - {func}):
                    self.check_unsupported()
        self.assertTrue(self.store.snapshot().ok)

    def test_missing_open_flags_are_refused_one_at_a_time(self):
        self.put(1)
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC"):
            saved = getattr(os, name)
            delattr(os, name)
            try:
                with self.subTest(flag=name):
                    self.check_unsupported()
            finally:
                setattr(os, name, saved)
        self.assertTrue(self.store.snapshot().ok)


class RootAndRaceBranchTests(Base):
    def setUp(self):
        super().setUp()
        self.put(1)

    def snap(self, hook):
        return self.bounded(lambda: self.swap_in(hook))

    def test_the_folder_path_vanishing_at_the_final_check_is_root_replaced(self):
        hook = Hook()
        hook.stat_raise[self.store.dir] = FileNotFoundError(2, "gone")
        self.assertNotOk(self.snap(hook), "root_replaced")

    def test_the_folder_entry_vanishing_at_the_final_check_is_root_replaced(self):
        hook = Hook()
        hook.lstat_raise[(S.MEMORY_DIR, 1)] = FileNotFoundError(2, "gone")
        self.assertNotOk(self.snap(hook), "root_replaced")

    def test_the_folder_path_pointing_elsewhere_at_the_final_check_is_root_replaced(self):
        hook = Hook()
        hook.stat_fake[self.store.dir] = lambda st: types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino + 1)
        self.assertNotOk(self.snap(hook), "root_replaced")

    def test_the_vault_root_pointing_elsewhere_at_the_final_check_is_root_replaced(self):
        hook = Hook()
        hook.stat_fake[self.store.vault_root] = lambda st: types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino + 1)
        self.assertNotOk(self.snap(hook), "root_replaced")

    def test_absent_folder_with_the_vault_root_vanishing_or_moving_is_root_replaced(self):
        shutil.rmtree(self.memdir)
        hook = Hook()
        hook.stat_raise[self.store.vault_root] = FileNotFoundError(2, "gone")
        self.assertNotOk(self.snap(hook), "root_replaced")
        hook = Hook()
        hook.stat_fake[self.store.vault_root] = lambda st: types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino + 1)
        self.assertNotOk(self.snap(hook), "root_replaced")
        self.assertTrue(self.snap(Hook()).ok)

    def test_a_file_vanishing_between_classification_and_open_is_changed_during_read(self):
        hook = Hook()
        hook.before_open[FILE % 1] = lambda: os.unlink(self.path(FILE % 1))
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

    def test_the_final_file_checks_each_see_what_they_are_for(self):
        """The validation loop is independent of the read-time check: each case below passes the read-time check."""
        hook = Hook()                                   # the entry vanishes between the listing and the final stat
        hook.lstat_raise[(FILE % 1, 2)] = FileNotFoundError(2, "gone")
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

        def replace():                                  # a new inode with the same bytes, after the last listing
            data = read_bytes(self.path(FILE % 1))
            self.write("incoming", data)
            os.replace(self.path("incoming"), self.path(FILE % 1))
        hook = Hook()
        hook.after_listdir[2] = replace
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

        hook = Hook()                                   # same ids, but the entry is now not a regular file
        hook.lstat_fake[(FILE % 1, 2)] = lambda st: types.SimpleNamespace(
            st_mode=0o120777, st_dev=st.st_dev, st_ino=st.st_ino, st_size=st.st_size, st_mtime_ns=st.st_mtime_ns,
            st_ctime_ns=st.st_ctime_ns)
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

    def test_a_file_vanishing_after_it_was_read_is_changed_during_read(self):
        hook = Hook()
        hook.at_eof = lambda: os.unlink(self.path(FILE % 1))
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

    def test_special_files_swapped_in_before_open_are_unsound_and_never_hang(self):
        def to_fifo():
            os.unlink(self.path(FILE % 1))
            os.mkfifo(self.path(FILE % 1))

        def to_dir():
            os.unlink(self.path(FILE % 1))
            os.mkdir(self.path(FILE % 1))

        def to_socket():
            os.unlink(self.path(FILE % 1))
            cwd, sock = os.getcwd(), socket.socket(socket.AF_UNIX)
            os.chdir(self.memdir)
            try:
                sock.bind(FILE % 1)                                  # a relative, short name: no path-length limit
            finally:
                os.chdir(cwd)
                sock.close()
        for label, swap in (("fifo", to_fifo), ("dir", to_dir), ("socket", to_socket)):
            shutil.rmtree(self.memdir)
            self.put(1)
            hook = Hook()
            hook.before_open[FILE % 1] = swap
            with self.subTest(label):
                snap = self.snap(hook)
                self.assertNotOk(snap, "store_unsound")
                self.assertEqual(snap.detail, (FILE % 1,))

    def test_a_file_that_cannot_be_opened_for_a_reason_other_than_its_type_is_unreadable(self):
        hook = Hook()
        plain = hook.open

        def deny(name, flags, dir_fd=None):
            if name == FILE % 1:
                raise PermissionError(13, "denied")
            return plain(name, flags, dir_fd)
        hook.open = deny
        self.assertNotOk(self.snap(hook), "snapshot_unreadable")

    def test_a_short_read_is_changed_during_read(self):
        hook = Hook()
        hook.short = True
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

    def test_a_timestamp_change_with_the_same_bytes_is_changed_during_read(self):
        hook = Hook()
        hook.after_first_read = lambda: os.utime(self.path(FILE % 1), ns=(1_000_000_000, 1_000_000_000))
        snap = self.snap(hook)
        self.assertNotOk(snap, "changed_during_read")
        self.assertEqual(snap.detail, (FILE % 1,))

    def test_a_read_that_never_ends_is_refused_not_looped(self):
        hook = Hook()
        hook.endless = True
        snap = self.snap(hook)
        self.assertNotOk(snap, "store_unsound")
        self.assertLessEqual(len(hook.read_sizes), S._MAX_FILE // S._CHUNK + 2)

    def test_the_size_cap_is_exact_and_an_oversized_file_is_never_read(self):
        big = FILE % 9
        self.write(big, b"x" * S._MAX_FILE)                       # exactly at the cap: read fully, then unsound by form
        hook = Hook()
        snap = self.snap(hook)
        self.assertNotOk(snap, "store_unsound")
        self.assertEqual(snap.detail, (big,))
        self.assertEqual(hook.read_sizes[-(S._MAX_FILE // S._CHUNK + 1):], [S._CHUNK] * (S._MAX_FILE // S._CHUNK) + [0])
        self.write(big, b"x" * (S._MAX_FILE + 1))                 # one byte over: refused before any read of it
        hook = Hook()
        snap = self.snap(hook)
        self.assertNotOk(snap, "store_unsound")
        self.assertEqual(snap.detail, (big,))
        self.assertEqual(len(hook.read_sizes), 2)                 # only mem_1 (its bytes, then end of file)

    def test_a_file_that_grows_past_the_cap_during_the_read_is_refused_at_the_cap(self):
        big = FILE % 9
        self.write(big, b"x" * (S._MAX_FILE + 5))
        hook = Hook()

        def small(st, fd):
            if st.st_size <= S._MAX_FILE:
                return st
            return types.SimpleNamespace(st_mode=st.st_mode, st_nlink=st.st_nlink, st_size=10, st_dev=st.st_dev,
                                         st_ino=st.st_ino, st_mtime_ns=st.st_mtime_ns, st_ctime_ns=st.st_ctime_ns)
        hook.fstat_override = small
        snap = self.snap(hook)
        self.assertNotOk(snap, "store_unsound")
        self.assertEqual(snap.detail, (big,))
        self.assertEqual(sum(hook.read_sizes[2:]), S._MAX_FILE + 5)   # stopped on the chunk that crossed the cap


class RealOpsTests(Base):
    """No hook at all: the shipped `_Ops` seam, so its own `lstat` is exercised (it must not follow a symlink)."""

    def test_the_shipped_seam_classifies_symlinks_without_following_them(self):
        self.put(1)
        self.assertIs(S._OPS.__class__, S._Ops)
        os.symlink(self.path(FILE % 1), self.path(FILE % 2))
        snap = self.store.snapshot()
        self.assertNotOk(snap, "store_unsound")
        self.assertEqual(snap.detail, (FILE % 2,))
        os.unlink(self.path(FILE % 2))
        self.assertTrue(self.store.snapshot().ok)

    def test_the_size_cap_is_the_decided_one_mebibyte(self):
        self.assertEqual(S._MAX_FILE, 1024 * 1024)


class StaticScopeTests(unittest.TestCase):
    def test_this_file_names_no_real_path(self):
        src = read_bytes(__file__).decode("utf-8")
        self.assertNotIn("/Users" + "/", src)


if __name__ == "__main__":
    unittest.main()
