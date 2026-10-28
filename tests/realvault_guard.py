"""TEST-ONLY guard that keeps a test from reaching the ORIGINAL real-vault path (`S.REAL_VAULT` as it is at import time).
Never imported by production code.

TWO MECHANISMS, because Python gives no single place to intercept everything:

  1. AN AUDIT HOOK (registered once, live only while a test has called `arm`). The interpreter reports these operations as
     audit events BEFORE they run, so raising from the hook stops them: open (builtin open, os.open, pathlib reads and
     writes, shutil copies), os.listdir, os.scandir, os.walk, glob.glob, os.mkdir (and makedirs), os.chmod, os.chown,
     os.utime, os.truncate, os.link, os.symlink, os.rename (and replace), os.remove (and unlink), os.rmdir, os.chdir.
  2. WRAPPERS around the path-taking metadata calls that emit NO audit event (measured, CPython 3.12 on macOS): os.stat,
     os.lstat, os.access, os.readlink, os.statvfs, os.pathconf. Everything built on them is covered through them:
     os.path.exists / isdir / isfile / islink / getsize / realpath / samefile and pathlib's exists / stat / resolve. Each
     wrapper is also registered in the same `os.supports_*` sets as the function it wraps, because the store's own platform
     check (`os.stat in os.supports_dir_fd`) would otherwise report `platform_unsupported`.

THE CHECK IS LEXICAL ON PURPOSE. Resolving a path (realpath) stats every component, which would itself inspect the
protected path before the guard could refuse. So a candidate is made absolute with `os.path.abspath` (no filesystem calls
except getcwd), lower-cased (the usual macOS volume is case-insensitive), and compared with the protected prefix.

WHAT IS NOT COVERED (each is pinned by a test in `test_memory_realvault_guard`, so the limit is stated, not assumed):
  * a path that reaches the protected tree through a SYMLINK or hard link elsewhere (resolving it would inspect it);
  * a relative name used with `dir_fd=` (the directory is a file descriptor, not a path);
  * file-descriptor operations (os.fstat, os.listdir(fd), reads on an already-open descriptor);
  * anything outside Python's os layer (ctypes, a subprocess, another process). Subprocess and the network are refused by
    separate patches in the tests that use this guard.
A guard of this kind is a tripwire, not a sandbox. Real prevention would need either dependency-injected filesystem access
in the code under test or a test run in a sandbox that has no route to the real vault."""
import os
import sys
import tempfile
from unittest import mock

from backtalk import memory_store as S

ORIGINAL = S.REAL_VAULT                      # captured at import, before any test patches S.REAL_VAULT
ARMED, TOUCHED = [], []

# audit event -> how many leading arguments are paths
_EVENTS = {"open": 1, "os.listdir": 1, "os.scandir": 1, "os.walk": 1, "glob.glob": 1, "os.mkdir": 1, "os.chmod": 1,
           "os.chown": 1, "os.utime": 1, "os.truncate": 1, "os.link": 2, "os.symlink": 2, "os.rename": 2,
           "os.remove": 1, "os.rmdir": 1, "os.chdir": 1}
# path-taking metadata calls with no audit event
_WRAPPED = ("stat", "lstat", "access", "readlink", "statvfs", "pathconf")
_SUPPORT_SETS = ("supports_dir_fd", "supports_fd", "supports_follow_symlinks", "supports_effective_ids")


def _norm(path) -> str:
    return os.path.abspath(os.fsdecode(path)).lower()


def is_protected(path, protected=None) -> bool:
    """True if `path` is, or is inside, a protected prefix. Lexical: it never touches the filesystem."""
    if path is None or isinstance(path, int):
        return False
    try:
        candidate = _norm(path)
    except (TypeError, ValueError):
        return False
    return any(candidate == p or candidate.startswith(p + "/") for p in (protected or _prefixes()))


_EXTRA = []


def _prefixes():
    return [_norm(ORIGINAL)] + [_norm(p) for p in _EXTRA]


def _audit(event, args):
    n = _EVENTS.get(event)
    if n and ARMED:
        for p in args[:n]:
            if isinstance(p, (str, bytes, os.PathLike)) and is_protected(p):
                TOUCHED.append(event)
                raise AssertionError(f"a protected real-vault path was reached by {event}")


sys.addaudithook(_audit)


def _wrapper(name, real):
    def wrapper(*a, **k):
        path = a[0] if a else k.get("path")
        if ARMED and k.get("dir_fd") is None and is_protected(path):
            TOUCHED.append("os." + name)
            raise AssertionError(f"a protected real-vault path was reached by os.{name}")
        return real(*a, **k)
    wrapper.__name__ = getattr(real, "__name__", name)
    wrapper.__wrapped__ = real
    return wrapper


def arm(case, *, also_protect=()):
    """Turn the guard on for one test and turn it off again in the test's cleanup. `also_protect` adds synthetic
    prefixes (the guard's own tests use a temp sentinel; nothing else should)."""
    del TOUCHED[:]
    del _EXTRA[:]
    _EXTRA.extend(also_protect)
    added = []
    for name in _WRAPPED:
        real = getattr(os, name, None)
        if real is None:
            continue
        wrapped = _wrapper(name, real)
        patcher = mock.patch.object(os, name, wrapped)
        patcher.start()
        case.addCleanup(patcher.stop)
        for set_name in _SUPPORT_SETS:
            members = getattr(os, set_name)
            if real in members:
                members.add(wrapped)
                added.append((members, wrapped))
    # `MemoryStore.__init__` resolves `S.REAL_VAULT` (realpath, which stats every component) to decide whether a vault is
    # the real one. That is production behavior, and it would inspect the protected path every time a temp store is built.
    # A guarded test therefore points `S.REAL_VAULT` at a decoy that does not exist; the store still decides "not the real
    # vault" for any temp vault, exactly as before, and the original path is never resolved. (Tests that need a "real"
    # vault patch `S.REAL_VAULT` to their own temp folder after this call.)
    decoy = os.path.join(tempfile.gettempdir(), f"decoy-real-vault-{os.getpid()}")
    decoy_patch = mock.patch.object(S, "REAL_VAULT", decoy)
    decoy_patch.start()
    case.addCleanup(decoy_patch.stop)
    ARMED.append(1)

    # The older base classes check in tearDown that `<real vault>/11 - Memory` still does not exist (one metadata read of
    # that single path, by design, in every test that derives from them). The guard is suspended for exactly that call so
    # those checks keep running unchanged; it is otherwise live for the whole test body.
    original_tear_down = case.tearDown

    def tear_down():
        ARMED.clear()
        original_tear_down()
    case.tearDown = tear_down

    def disarm():
        ARMED.clear()
        for members, wrapped in added:
            members.discard(wrapped)
        del _EXTRA[:]
    case.addCleanup(disarm)
    return TOUCHED
