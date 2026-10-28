"""The Memory 2.0 repository (Phase 3): persists what the gate accepted, atomically, and nothing else.

Phase 2 (memory_gate) decides whether a proposal is acceptable. This module only writes. It repairs nothing, guesses
nothing and invents nothing: a memory that does not validate is refused with a reason code, never "fixed". It is
connected to nothing in Jarvis yet.

NAMESPACE. Every memory is one file, `<vault>/11 - Memory/<id>.md`, and nothing else lives in that folder. A file
is also marked inside (`memory_schema: 1`), so a memory is recognisable wherever it is copied. The folder is created
on the first write, never by constructing a store, and ordinary notes are never opened, listed or changed.
(Machine-owned: edit through the store. A hand edit that stops the file parsing strictly makes the store refuse
to read until it is put right: it fails closed.)

FILE (strict, deterministic, human-readable, one field per line, every value JSON so no YAML ambiguity):

    ---
    memory_schema: 1
    id: "mem_0123456789ab"
    type: "fact"
    subject: "user"
    attribute: "residence"
    value: "San Jose"
    created: "2026-01-10T00:00:00Z"
    updated: "2026-01-10T00:00:00Z"
    source: {"kind": "user", "ref": "turn 41"}
    confidence: "stated"
    valid_from: "2026-01-10T00:00:00Z"
    valid_to: null
    status: "active"
    supersedes: []
    sensitivity: "normal"
    ---

    ## Evidence

    I live in San Jose now

The fields are exactly the Phase 1 MemoryItem; the evidence is exactly the bounded, single-line text the gate allowed.
The id INSIDE the file is authoritative and must equal the file name.

ATOMIC. A file is written to a temporary name, fsynced, then moved into place with one rename. A supersession
changes several files; it is journaled: the journal (`.txn.json`) records every file's previous content BEFORE
anything is touched, and an interrupted transaction is rolled back to exactly that state the next time the store is
used. A transaction is finished only when its journal says so. One advisory lock (`.lock`) serialises writers and
readers, in this process and in others.

AUTHORIZATION (Phase 8.1B). A write may carry an `Authorization`: a record that one exact operation, on one exact store
state, was confirmed within the last 120 seconds. The store checks it itself, inside the write lock, immediately before
installing, against ITS OWN clock (read once; a store on the real vault refuses any other clock) and a digest of the
exact memory and of every stored file. The nonce is recorded in `.authorizations` in the SAME journalled transaction as
the memory file, so both land or neither does. Writes to the real vault, and every private memory, require one.
THIS IS AN API SAFETY CONTROL, NOT PROOF OF A HUMAN: an Authorization is plain data built from public digests, so any
code running as the same user can forge one, call the private install path, or write the files directly. It stops
accidents, stale or mismatched approvals and replays through the public API; it does not authenticate a person.

SNAPSHOT (Phase 8.2c-1). `snapshot()` is a strictly read-only, single-pass OBSERVATION of the folder: it opens nothing
for writing, creates no `.lock`, never recovers, repairs, truncates or deletes, and reads every file exactly once from
one descriptor, so the digest and the parsed memory come from the same bytes. A successful snapshot is a consistent
observation at the completed validation boundary, NOT a guarantee that the folder is unchanged afterwards. It reports
what the directory listings and metadata comparisons actually showed; it does not claim anything held "throughout".
`manifest_digest` and `snapshot_digest` are the ONE authoritative digest definitions (integrity, not authentication).

THE GATE IS NOT BYPASSED. `apply` accepts only an ACCEPT decision, then re-runs the gate against what is stored NOW and
refuses unless it gives the same decision (a REQUIRE_CONFIRMATION, a forged or a stale decision all fail). `put` and
`apply` also refuse, on their own, never-expose, credential-like and assistant-steering content.
"""
import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from backtalk import memory as M
from backtalk import memory_gate as G

MEMORY_DIR = "11 - Memory"
SCHEMA_VERSION = 1
REAL_VAULT = "/Users/yousif/Desktop/jarvis/jarvis"

FIELD_ORDER = ("memory_schema", "id", "type", "subject", "attribute", "value", "created", "updated", "source",
               "confidence", "valid_from", "valid_to", "status", "supersedes", "sensitivity")
_FILE_RE = re.compile(r"^mem_[0-9a-f]{12}\.md\Z")
_JOURNAL = ".txn.json"
_LOCK = ".lock"
_TMP_PREFIX = ".tmp-"
_AUTH_LOG = ".authorizations"
AUTH_WINDOW = 120                       # seconds an Authorization stays usable; fixed, not caller-chosen
AUTH_VERSION = "auth/1"
AUTH_METHODS = ("typed-terminal",)
_AUTH_KEEP = 24 * 3600                  # nonce records older than this are expired many times over and are pruned
_HEX64, _HEX32 = re.compile(r"^[0-9a-f]{64}\Z"), re.compile(r"^[0-9a-f]{32}\Z")
_LOG_LINE = re.compile(r"^([0-9a-f]{32}) (mem_[0-9a-f]{12}) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\Z")


class StoreError(Exception):
    """A refusal. `reason` is a short code; `details` are codes and ids only, never any content."""

    def __init__(self, reason, details=()):
        super().__init__(reason + (": " + ", ".join(map(str, details)) if details else ""))
        self.reason, self.details = reason, tuple(details)


@dataclass(frozen=True)
class Stored:
    item: object                  # a memory.MemoryItem
    evidence: str


@dataclass(frozen=True)
class Authorization:
    """A recorded, typed yes for ONE operation on ONE store state. Plain data; the store verifies every field itself."""
    operation: str                # "put" | "apply"
    item_digest: str              # 64 hex: see item_digest()
    state_digest: str             # 64 hex: MemoryStore.state_digest() when the yes was given
    nonce: str                    # 32 hex, single use
    confirmed_at: str             # strict UTC timestamp of the yes
    context_digest: str           # 64 hex; opaque to the store (the caller's review artifact)
    method: str                   # "typed-terminal": a label the store cannot verify


@dataclass(frozen=True)
class Applied:
    memory_id: str
    superseded: tuple             # ids of the items that were ended by this write, sorted


# ---- the file format -----------------------------------------------------------------------------

def _js(value) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "), allow_nan=False)
    return text.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def serialize(item, evidence) -> str:
    """The exact text of a memory file. Raises StoreError for anything that is not a valid item or usable evidence."""
    errs = M.validate(item)
    if errs:
        raise StoreError("invalid_item", errs)
    if not G.evidence_usable(evidence):
        raise StoreError("invalid_evidence")
    d = M.to_dict(item)
    d["memory_schema"] = SCHEMA_VERSION
    lines = ["---"] + [f"{k}: {_js(d[k])}" for k in FIELD_ORDER] + ["---", "", "## Evidence", "", evidence]
    return "\n".join(lines) + "\n"


def parse_file(text):
    """(item, evidence) from a memory file, or StoreError. Strict: exact field order, exact layout, the Phase 1
    parser for the item, nothing coerced or defaulted."""
    if not isinstance(text, str) or not text.endswith("\n"):
        raise StoreError("bad_format", ("ending",))
    lines = text[:-1].split("\n")
    n = len(FIELD_ORDER)
    if len(lines) < n + 6 or lines[0] != "---" or lines[n + 1] != "---" or lines[n + 2] != "" \
            or lines[n + 3] != "## Evidence" or lines[n + 4] != "" or len(lines) != n + 6:
        raise StoreError("bad_format", ("layout",))
    data = {}
    for key, line in zip(FIELD_ORDER, lines[1:n + 1]):
        prefix = key + ": "
        if not line.startswith(prefix):
            raise StoreError("bad_format", (f"field:{key}",))
        try:
            data[key] = json.loads(line[len(prefix):])
        except ValueError:
            raise StoreError("bad_format", (f"value:{key}",)) from None
    if type(data["memory_schema"]) is not int or data["memory_schema"] != SCHEMA_VERSION:
        raise StoreError("bad_format", ("schema",))
    del data["memory_schema"]
    item, errs = M.parse(data)
    if item is None:
        raise StoreError("invalid_item", errs)
    evidence = lines[n + 5]
    if not G.evidence_usable(evidence):
        raise StoreError("invalid_evidence")
    return item, evidence


# ---- file system primitives (module level so tests can make them fail) ------------------------------

def _write_atomic(path, text):
    """Install `text` at `path`: temp file in the same folder, fsync, one rename. A failure removes the temp file
    and leaves whatever was at `path` exactly as it was."""
    folder = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=_TMP_PREFIX, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _sync_dir(folder)


def _replace(src, dst):
    os.replace(src, dst)


def _sync_dir(folder):
    try:
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _read(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def _system_utc():
    """The store's one wall-clock read."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def item_digest(item, evidence, links, operation) -> str:
    """The digest an Authorization must carry: the exact file the write would produce (every field and the evidence),
    the supersession links, and the operation."""
    body = {"v": AUTH_VERSION, "op": operation, "links": sorted(links),
            "file": serialize(replace(item, supersedes=tuple(sorted(links))), evidence)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_log(text):
    """[(nonce_hash, memory_id, confirmed_at)] for a well-formed nonce log, else None."""
    out = []
    for line in text.split("\n"):
        if not line:
            continue
        m = _LOG_LINE.match(line)
        if not m or M.parse_ts(m.group(3)) is None:
            return None
        out.append(m.groups())
    return out


def _stamp(now) -> str:
    if isinstance(now, datetime):
        if now.tzinfo is None or now.utcoffset() is None:
            raise StoreError("invalid_time")
        now = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if M.parse_ts(now) is None:
        raise StoreError("invalid_time")
    return now


# ---- digests: the ONE authoritative definitions (integrity checks, not authentication) -------------------------

SNAPSHOT_SCHEMA = "memory-snapshot/1"


def canonical_text(obj) -> str:
    """The canonical JSON every digest in this module hashes: sorted keys, compact, ASCII-only, one line."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))      # ensure_ascii is json's default (True)


def canonical_bytes(obj) -> bytes:
    return canonical_text(obj).encode()                              # ASCII only, so UTF-8 by construction


def normalized_entries(entries):
    """[[name, sha256_hex], ...] sorted ascending by name. ValueError("bad_manifest_entries") for anything malformed
    or duplicated, so ordering and shape are decided in exactly one place."""
    try:
        pairs = [(name, sha) for name, sha in entries]
    except (TypeError, ValueError):
        raise ValueError("bad_manifest_entries") from None
    seen = set()
    for name, sha in pairs:
        if not (isinstance(name, str) and isinstance(sha, str) and _FILE_RE.match(name) and _HEX64.match(sha)) \
                or name in seen:
            raise ValueError("bad_manifest_entries")
        seen.add(name)
    return [[name, sha] for name, sha in sorted(pairs)]


def manifest_digest(entries) -> str:
    """sha256 of canonical_json({"v": AUTH_VERSION, "files": [[name, sha256_hex], ...]}), names ascending. Byte for
    byte the digest `state_digest()` has always produced; an empty set is `{"files":[],"v":"auth/1"}`."""
    body = {"v": AUTH_VERSION, "files": normalized_entries(entries)}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def normalized_nonce_rows(rows):
    """[[hash32, memory_id, confirmed_at], ...] ascending. ValueError("bad_nonces") for a row that is not exactly three
    strings forming a valid nonce-log line (the store's own `_LOG_LINE` and timestamp rules), or a repeated nonce hash."""
    try:
        found = [tuple(r) for r in rows]
    except TypeError:
        raise ValueError("bad_nonces") from None
    for r in found:
        if not (len(r) == 3 and all(isinstance(x, str) for x in r) and _LOG_LINE.match(" ".join(r))
                and M.parse_ts(r[2]) is not None):
            raise ValueError("bad_nonces")
    if len({r[0] for r in found}) != len(found):
        raise ValueError("bad_nonces")
    return [list(r) for r in sorted(found)]


def snapshot_digest(entries, nonces) -> str:
    """sha256 of canonical_json({"schema": SNAPSHOT_SCHEMA, "files": [[name, sha], ...],
    "nonces": [[hash32, memory_id, confirmed_at], ...]}); names and rows ascending, nonce hashes unique. Parsed items,
    the journal, `.lock`, times and inodes are not inputs."""
    body = {"schema": SNAPSHOT_SCHEMA, "files": normalized_entries(entries), "nonces": normalized_nonce_rows(nonces)}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """One observation of the memory folder. `ok` False means `reason` is a stable code and every other value is None.
    `origin` ("absent" | "uninitialized" | "initialized") records what was observed and is NOT part of any digest.
    No clock, no expiry: it is an observation made at the validation boundary, nothing more."""
    ok: bool
    reason: str                   # "ok" or a stable code
    origin: object
    files: object                 # tuple of (name, sha256_hex), ascending by name
    items: object                 # tuple of Stored, ascending by id, parsed from the very bytes that were hashed
    nonces: object                # tuple of (hash32, memory_id, confirmed_at) rows, ascending
    state_digest: object
    snapshot_digest: object
    detail: tuple = ()            # file names / codes only, never content


class _SnapStop(Exception):
    def __init__(self, reason, detail=()):
        super().__init__(reason)
        self.reason, self.detail = reason, tuple(detail)


class _Ops:
    """Every OS call `snapshot()` makes, in one place so a test can hook them deterministically. Read-only calls only.
    `lstat` is stat without following a symlink; `validated` is a no-op hook called once, after the last comparison."""
    open = staticmethod(os.open)
    close = staticmethod(os.close)
    fstat = staticmethod(os.fstat)
    listdir = staticmethod(os.listdir)
    read = staticmethod(os.read)
    flock = staticmethod(fcntl.flock)
    stat = staticmethod(os.stat)

    @staticmethod
    def lstat(name, dir_fd=None):
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)

    @staticmethod
    def validated():
        return None


_OPS = _Ops()
_MAX_FILE = 1 << 20                     # a memory file is a few hundred bytes; anything larger is not one
_CHUNK = 1 << 16
_IFMT, _IFREG, _IFDIR, _IFLNK = 0o170000, 0o100000, 0o040000, 0o120000


def _kind(st):
    """The file-type bits of a stat result (compare with _IFREG, _IFDIR, _IFLNK); anything else is a special file."""
    return st.st_mode & _IFMT


def _dir_id(st):
    return (st.st_dev, st.st_ino, st.st_nlink, st.st_mtime_ns, st.st_ctime_ns)


def _file_id(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


# ---- the repository -------------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, vault_root, *, allow_real_vault=False, clock=None):
        real = os.path.realpath(vault_root)
        prod = os.path.realpath(REAL_VAULT)
        on_real = real == prod or real.startswith(prod + os.sep)
        if not allow_real_vault and on_real:
            raise StoreError("real_vault_refused")
        if on_real and clock is not None:
            raise StoreError("clock_not_allowed")
        self.vault_root = vault_root
        self.dir = os.path.join(vault_root, MEMORY_DIR)
        self.require_authorization = on_real              # fixed at construction
        self._clock = clock if clock is not None else _system_utc

    # -- locking and recovery --

    def _locked(self, write):
        return _Lock(self, write)

    def _path(self, memory_id):
        return os.path.join(self.dir, memory_id + ".md")

    def _recover(self):
        """Finish or undo an interrupted transaction, and sweep temp files. Inside the lock, before any read."""
        jpath = os.path.join(self.dir, _JOURNAL)
        if os.path.exists(jpath):
            try:
                journal = json.loads(_read(jpath))
                state, ops = journal["state"], journal["ops"]
                assert state in ("prepared", "committed") and isinstance(ops, list)
            except (ValueError, KeyError, AssertionError, TypeError):
                raise StoreError("journal_corrupt") from None
            if state == "prepared":
                self._rollback(ops)
            os.unlink(jpath)
            _sync_dir(self.dir)
        for name in os.listdir(self.dir):
            if name.startswith(_TMP_PREFIX):
                try:
                    os.unlink(os.path.join(self.dir, name))
                except OSError:
                    pass

    def _rollback(self, ops):
        for op in reversed(ops):
            path = os.path.join(self.dir, op["file"])
            try:
                if op["prior"] is None:
                    if os.path.exists(path):
                        os.unlink(path)
                else:
                    _write_atomic(path, op["prior"])
            except OSError:
                raise StoreError("recovery_failed", (op["file"],)) from None

    # -- reading --

    def _scan(self):
        """({id: Stored}, problems). Never raises for content: problems are codes."""
        found, problems = {}, []
        if not os.path.isdir(self.dir):
            return found, problems
        for name in sorted(os.listdir(self.dir)):
            if name in (_LOCK, _AUTH_LOG):
                continue
            if not _FILE_RE.match(name):
                problems.append(f"unexpected_file:{name}")
                continue
            try:
                item, evidence = parse_file(_read(os.path.join(self.dir, name)))
            except StoreError as e:
                problems.append(f"{name}:{e.reason}" + (f":{','.join(e.details)}" if e.details else ""))
                continue
            except (OSError, UnicodeDecodeError):
                problems.append(f"{name}:unreadable")
                continue
            if item.id != name[:-3]:
                problems.append(f"{name}:id_mismatch")
                continue
            found[item.id] = Stored(item, evidence)
        if not problems:
            res = M.resolve([s.item for s in found.values()], "2000-01-01T00:00:00Z")
            problems += [f"{ident}:{','.join(errs)}" for ident, errs in res.rejected]
        return found, problems

    def _all(self):
        found, problems = self._scan()
        if problems:
            raise StoreError("corrupt_store", problems)
        return found

    def load(self, memory_id):
        """The stored memory with this id. StoreError if there is none or the store is not sound."""
        with self._locked(write=False):
            found = self._all()
        if memory_id not in found:
            raise StoreError("not_found", (memory_id,))
        return found[memory_id]

    def list_items(self):
        """Every memory, sorted by id, history included. A store with any malformed file refuses to list."""
        with self._locked(write=False):
            found = self._all()
        return tuple(found[i] for i in sorted(found))

    def verify(self):
        """Problem codes for every malformed file and cross-item inconsistency; () for a sound store. Read-only."""
        with self._locked(write=False):
            problems = list(self._scan()[1])
            if self._read_log() is None:
                problems.append("authorizations:corrupt")
            return tuple(problems)

    def state_digest(self) -> str:
        """The digest an Authorization's `state_digest` must equal: every stored memory file's name and bytes."""
        with self._locked(write=False):
            return self._state_digest()

    def _state_digest(self) -> str:
        entries = []
        if os.path.isdir(self.dir):
            for name in sorted(os.listdir(self.dir)):
                if _FILE_RE.match(name):
                    with open(os.path.join(self.dir, name), "rb") as f:
                        entries.append([name, hashlib.sha256(f.read()).hexdigest()])
        return manifest_digest(entries)

    def snapshot(self, *, block=False) -> Snapshot:
        """A strictly read-only, single-pass observation of the memory folder (see the module docstring). Never creates,
        repairs, recovers, truncates or deletes anything and never takes the exclusive lock. By default one
        non-blocking shared-lock attempt: a writer holding the lock gives `lock_busy` (the caller decides whether to
        retry); `block=True` waits for it. Never raises for store content."""
        try:
            return self._snapshot(block)
        except _SnapStop as stop:
            return Snapshot(False, stop.reason, None, None, None, None, None, None, stop.detail)
        except OSError:
            return Snapshot(False, "snapshot_unreadable", None, None, None, None, None, None)

    def _snapshot(self, block):
        ops = _OPS
        if not (os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
                and os.stat in os.supports_follow_symlinks and os.listdir in os.supports_fd
                and all(hasattr(os, n) for n in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC"))):
            raise _SnapStop("platform_unsupported")
        base = os.O_RDONLY | os.O_CLOEXEC
        fds = []

        def open_fd(name, flags, dir_fd=None):
            fd = ops.open(name, flags, dir_fd=dir_fd) if dir_fd is not None else ops.open(name, flags)
            fds.append(fd)
            return fd

        def kind_of(dir_fd, name):
            try:
                return _kind(ops.lstat(name, dir_fd=dir_fd))
            except FileNotFoundError:
                return None

        def read_once(dfd, name):
            """(bytes, file_id): ONE open, ONE read to EOF through that descriptor, bracketed by two fstat calls."""
            try:
                fd = open_fd(name, base | os.O_NOFOLLOW | os.O_NONBLOCK, dfd)
            except FileNotFoundError:
                raise _SnapStop("changed_during_read", (name,)) from None
            except OSError:
                if kind_of(dfd, name) not in (None, _IFREG):
                    raise _SnapStop("store_unsound", (name,)) from None
                raise
            try:
                st = ops.fstat(fd)
                if _kind(st) != _IFREG or st.st_nlink != 1 or st.st_size > _MAX_FILE:
                    raise _SnapStop("store_unsound", (name,))
                chunks, total = [], 0
                for _ in range(_MAX_FILE // _CHUNK + 2):            # bounded: a read loop can never spin forever
                    chunk = ops.read(fd, _CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_FILE:
                        raise _SnapStop("store_unsound", (name,))
                    chunks.append(chunk)
                else:
                    raise _SnapStop("store_unsound", (name,))
                data = b"".join(chunks)
                after = ops.fstat(fd)
            finally:
                ops.close(fd)
                fds.remove(fd)
            if _file_id(st) != _file_id(after) or len(data) != st.st_size:
                raise _SnapStop("changed_during_read", (name,))
            return data, _file_id(after)

        try:
            # ---- the vault root and the memory folder: opened ONCE; from here only the descriptors are used ----
            try:
                pfd = open_fd(self.vault_root, base | os.O_DIRECTORY)
            except FileNotFoundError:
                raise _SnapStop("store_root_missing") from None
            ps = ops.fstat(pfd)
            p0 = (ps.st_dev, ps.st_ino, ps.st_mtime_ns, ps.st_ctime_ns)
            try:
                dfd = open_fd(MEMORY_DIR, base | os.O_DIRECTORY | os.O_NOFOLLOW, pfd)
            except FileNotFoundError:
                return self._snapshot_absent(ops, pfd, p0, kind_of)
            except OSError:
                kind = kind_of(pfd, MEMORY_DIR)
                if kind == _IFLNK:
                    raise _SnapStop("root_symlink") from None
                if kind not in (None, _IFDIR):
                    raise _SnapStop("store_unsound", (MEMORY_DIR,)) from None
                raise
            d0 = _dir_id(ops.fstat(dfd))
            root_id = d0[:2]

            def check_root():
                """The folder we hold is still the one the vault root (by descriptor and by path) points at."""
                try:
                    entry = ops.lstat(MEMORY_DIR, dir_fd=pfd)
                    by_path = ops.stat(self.dir)
                    by_root = ops.stat(self.vault_root)
                except FileNotFoundError:
                    raise _SnapStop("root_replaced") from None
                proot = ops.fstat(pfd)
                if _kind(entry) != _IFDIR or (entry.st_dev, entry.st_ino) != root_id \
                        or (by_path.st_dev, by_path.st_ino) != root_id \
                        or (by_root.st_dev, by_root.st_ino) != (proot.st_dev, proot.st_ino):
                    raise _SnapStop("root_replaced")

            # ---- the lock: opened read-only, never created, shared, non-blocking by default ----
            lock_id = None
            try:
                lfd = open_fd(_LOCK, base | os.O_NOFOLLOW | os.O_NONBLOCK, dfd)
            except FileNotFoundError:
                lfd = None
            except OSError:
                if kind_of(dfd, _LOCK) not in (None, _IFREG):
                    raise _SnapStop("store_unsound", (_LOCK,)) from None
                raise
            if lfd is not None:
                lst = ops.fstat(lfd)
                if _kind(lst) != _IFREG:
                    raise _SnapStop("store_unsound", (_LOCK,))
                try:
                    ops.flock(lfd, fcntl.LOCK_SH if block else fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise _SnapStop("lock_busy") from None
                lock_id = (lst.st_dev, lst.st_ino)
                try:
                    cur = ops.lstat(_LOCK, dir_fd=dfd)
                except FileNotFoundError:
                    raise _SnapStop("lock_replaced") from None
                if _kind(cur) != _IFREG or (cur.st_dev, cur.st_ino) != lock_id:
                    raise _SnapStop("lock_replaced")

            names = sorted(ops.listdir(dfd))
            if lfd is None:
                # State B: no lock to hold, so the only provable thing is "listed empty, twice, with the same folder
                # identity and times at both ends". Anything else is not an empty snapshot.
                if names:
                    raise _SnapStop("lock_missing")
                if kind_of(dfd, _LOCK) is not None or sorted(ops.listdir(dfd)) != []:
                    raise _SnapStop("changed_during_read")
                if _dir_id(ops.fstat(dfd)) != d0:
                    raise _SnapStop("changed_during_read")
                check_root()
                ops.validated()
                return self._snapshot_result("uninitialized", [], [], [])

            # State A: the shared lock is held; classify, then read every file exactly once.
            kinds = {}
            for name in names:
                if name == _LOCK:
                    continue
                kinds[name] = kind_of(dfd, name)
                if kinds[name] is None:
                    raise _SnapStop("changed_during_read", (name,))
            if _JOURNAL in names:
                if kinds[_JOURNAL] != _IFREG:
                    raise _SnapStop("store_unsound", (_JOURNAL,))
                data, _ = read_once(dfd, _JOURNAL)
                try:
                    journal = json.loads(data.decode())
                    ok = isinstance(journal, dict) and journal.get("state") in ("prepared", "committed") \
                        and isinstance(journal.get("ops"), list)
                except (ValueError, UnicodeDecodeError):
                    ok = False
                raise _SnapStop("recovery_pending" if ok else "journal_corrupt")
            unsound = [n for n in names if n != _LOCK and n != _AUTH_LOG and not (_FILE_RE.match(n) and kinds[n] == _IFREG)]
            if unsound or (_AUTH_LOG in names and kinds[_AUTH_LOG] != _IFREG):
                raise _SnapStop("store_unsound", unsound or (_AUTH_LOG,))

            ids, files, items = {}, [], []
            for name in names:
                if not _FILE_RE.match(name):
                    continue
                data, fid = read_once(dfd, name)
                try:
                    item, evidence = parse_file(data.decode())
                    sound = item.id == name[:-3] and serialize(item, evidence).encode() == data
                except (StoreError, UnicodeDecodeError):
                    sound = False
                if not sound:
                    raise _SnapStop("store_unsound", (name,))
                ids[name] = fid
                files.append((name, hashlib.sha256(data).hexdigest()))
                items.append(Stored(item, evidence))
            nonces = []
            if _AUTH_LOG in names:
                data, fid = read_once(dfd, _AUTH_LOG)
                ids[_AUTH_LOG] = fid
                try:
                    rows = _parse_log(data.decode())
                except UnicodeDecodeError:
                    rows = None
                nonces = [tuple(r) for r in rows] if rows is not None else None
                if nonces is None or len({r[0] for r in nonces}) != len(nonces):
                    raise _SnapStop("authorization_log_corrupt")

            # ---- validate: same names, same files (identity, size, times), same folder ----
            if sorted(ops.listdir(dfd)) != names:
                raise _SnapStop("changed_during_read")
            for name, fid in ids.items():
                try:
                    cur = ops.lstat(name, dir_fd=dfd)
                except FileNotFoundError:
                    raise _SnapStop("changed_during_read", (name,)) from None
                if _kind(cur) != _IFREG or _file_id(cur) != fid:
                    raise _SnapStop("changed_during_read", (name,))
            try:
                cur = ops.lstat(_LOCK, dir_fd=dfd)
            except FileNotFoundError:
                raise _SnapStop("lock_replaced") from None
            if _kind(cur) != _IFREG or (cur.st_dev, cur.st_ino) != lock_id:
                raise _SnapStop("lock_replaced")
            check_root()
            ops.validated()
            return self._snapshot_result("initialized", files, items, nonces)
        finally:
            for fd in fds:
                try:
                    ops.close(fd)
                except OSError:
                    pass

    def _snapshot_absent(self, ops, pfd, p0, kind_of):
        """State C. `11 - Memory` was absent on the first probe. Proven only if it is absent again on a second probe
        and the vault root's identity and times are equal at both ends; otherwise it is not an empty snapshot."""
        if kind_of(pfd, MEMORY_DIR) is not None:
            raise _SnapStop("changed_during_read")
        now = ops.fstat(pfd)
        if (now.st_dev, now.st_ino, now.st_mtime_ns, now.st_ctime_ns) != p0:
            raise _SnapStop("changed_during_read")
        try:
            by_root = ops.stat(self.vault_root)
        except FileNotFoundError:
            raise _SnapStop("root_replaced") from None
        if (by_root.st_dev, by_root.st_ino) != (now.st_dev, now.st_ino):
            raise _SnapStop("root_replaced")
        ops.validated()
        return self._snapshot_result("absent", [], [], [])

    @staticmethod
    def _snapshot_result(origin, files, items, nonces):
        files, nonces = tuple(files), tuple(sorted(nonces))
        return Snapshot(True, "ok", origin, files, tuple(items), nonces,
                        manifest_digest(files), snapshot_digest(files, nonces))

    def _read_log(self):
        """The nonce records, [] when there is no log, None when it is malformed."""
        path = os.path.join(self.dir, _AUTH_LOG)
        if not os.path.exists(path):
            return []
        try:
            return _parse_log(_read(path))
        except (OSError, UnicodeDecodeError):
            return None

    # -- writing --

    def _check_new(self, item, evidence):
        errs = M.validate(item)
        if errs:
            raise StoreError("invalid_item", errs)
        if not G.evidence_usable(evidence):
            raise StoreError("invalid_evidence")
        if item.sensitivity == "never-expose":
            raise StoreError("never_expose_not_storable")
        hit = G.screen_item(item, evidence)
        if hit:
            raise StoreError(hit[0], hit[1])

    def put(self, item, evidence, now=None, authorization=None):
        """Store one new memory that replaces nothing, through the same gate, the same authorization routine and the same
        transaction as `apply`. The gate must ACCEPT it against what is stored right now, and the acceptance must not
        supersede anything (that is `apply`'s job). `now` defaults to the item's own `updated` stamp, so an
        unauthorized put never reads the clock; an authorized one defaults to the store clock."""
        self._check_new(item, evidence)
        if item.supersedes:
            raise StoreError("supersedes_requires_apply")
        if authorization is None and self.require_authorization:
            raise StoreError("authorization_required")        # before the lock: a refused write creates nothing
        proposal = G.Proposal(item, evidence)
        stamp = None if (now is None and authorization is not None) else _stamp(item.updated if now is None else now)
        with self._locked(write=True):
            found = self._all()
            if item.id in found:
                raise StoreError("duplicate_id", (item.id,))
            decision, extra, stamp = self._decide_and_authorize("put", proposal, stamp, found, authorization, None)
            return self._commit(decision, proposal, stamp, found, require_in_force=False,
                                journal=False, extra_ops=extra).memory_id

    def apply(self, decision, proposal, now, authorization=None):
        """Persist an ACCEPT decision. A new memory is stored; the memories it supersedes are ended in the SAME
        transaction (status superseded, valid_to the new valid_from, updated `now`), or nothing changes. A private
        memory's REQUIRE_CONFIRMATION decision is accepted here only together with a valid Authorization."""
        if not (isinstance(decision, G.Decision) and isinstance(proposal, G.Proposal)
                and isinstance(proposal.item, M.MemoryItem)):
            raise StoreError("malformed_request")
        if authorization is None or decision.decision not in (G.ACCEPT, G.REQUIRE_CONFIRMATION):
            self._require_accept(decision)
        if authorization is None and self.require_authorization:
            raise StoreError("authorization_required")        # before the lock: a refused write creates nothing
        stamp = _stamp(now)
        self._check_new(proposal.item, proposal.evidence)
        with self._locked(write=True):
            found = self._all()
            decision, extra, stamp = self._decide_and_authorize("apply", proposal, stamp, found, authorization, decision)
            return self._commit(decision, proposal, stamp, found, require_in_force=True, extra_ops=extra)

    @staticmethod
    def _require_accept(decision):
        if decision.decision == G.REQUIRE_CONFIRMATION:
            raise StoreError("confirmation_required")
        if decision.decision != G.ACCEPT:
            raise StoreError("decision_not_accept", (decision.decision, decision.reason))

    def _decide_and_authorize(self, operation, proposal, stamp, found, authorization, claimed):
        """THE shared check for `put` and `apply`: called inside the write lock, immediately before installing.
        Returns (the ACCEPT Decision to commit, extra files for the same transaction, the stamp used). Order: gate against what is stored
        now; refuse anything not acceptable; then, when an Authorization is present or required, every check below, with
        nothing written until all pass. A Confirmation is derived here, never accepted from the caller."""
        t = self._now() if authorization is not None else None                # the one clock read
        if stamp is None:
            stamp = t
        d0 = G.evaluate(proposal, [s.item for s in found.values()], stamp)
        if claimed is not None and d0 != claimed:
            raise StoreError("stale_decision", (d0.decision, d0.reason))
        private = d0.decision == G.REQUIRE_CONFIRMATION and d0.reason == "private_requires_confirmation"
        if authorization is None:
            self._require_accept(d0)
            if operation == "put" and d0.supersedes:
                raise StoreError("supersedes_requires_apply")
            if self.require_authorization:
                raise StoreError("authorization_required")
            return d0, None, stamp
        if not private:
            self._require_accept(d0)
        if operation == "put" and d0.supersedes:
            raise StoreError("supersedes_requires_apply")
        self._check_authorization(authorization, operation, proposal, d0, found, stamp, t)
        if private:
            digest = G.proposal_digest(proposal.item, proposal.evidence, d0.supersedes)
            d1 = G.evaluate(proposal, [s.item for s in found.values()], stamp, G.Confirmation(digest, "private"))
            if d1.decision != G.ACCEPT:
                raise StoreError("decision_not_accept", (d1.decision, d1.reason))
        else:
            d1 = d0
        return d1, {os.path.join(self.dir, _AUTH_LOG): self._next_log(authorization, proposal.item.id, t)}, stamp

    def _now(self):
        try:
            return _stamp(self._clock())
        except StoreError:
            raise
        except Exception:
            raise StoreError("clock_failed") from None

    def _check_authorization(self, auth, operation, proposal, d0, found, stamp, t):
        def refuse(code):
            raise StoreError("authorization_invalid", (code,))
        if not (isinstance(auth, Authorization) and auth.operation in ("put", "apply")
                and all(isinstance(v, str) for v in (auth.item_digest, auth.state_digest, auth.nonce, auth.confirmed_at,
                                                    auth.context_digest, auth.method))
                and _HEX64.match(auth.item_digest) and _HEX64.match(auth.state_digest)
                and _HEX64.match(auth.context_digest) and _HEX32.match(auth.nonce)
                and auth.method in AUTH_METHODS and M.parse_ts(auth.confirmed_at) is not None):
            refuse("malformed")
        if auth.operation != operation:
            refuse("operation_mismatch")
        age = (M.parse_ts(t) - M.parse_ts(auth.confirmed_at)).total_seconds()
        if age < 0:
            refuse("not_yet_valid")
        if age > AUTH_WINDOW:
            refuse("expired")
        if abs((M.parse_ts(stamp) - M.parse_ts(t)).total_seconds()) > AUTH_WINDOW:
            refuse("now_not_current")
        if auth.item_digest != item_digest(proposal.item, proposal.evidence, d0.supersedes, operation):
            refuse("item_mismatch")
        if auth.state_digest != self._state_digest():
            refuse("state_changed")
        log = self._read_log()
        if log is None:
            refuse("authorization_log_corrupt")
        if hashlib.sha256(auth.nonce.encode("ascii")).hexdigest()[:32] in {row[0] for row in log}:
            refuse("reused")

    def _next_log(self, auth, memory_id, t):
        """The nonce log as it will be after this write: expired records dropped, this one appended."""
        floor = M.parse_ts(t).timestamp() - _AUTH_KEEP
        keep = [row for row in self._read_log() if M.parse_ts(row[2]).timestamp() >= floor]
        keep.append((hashlib.sha256(auth.nonce.encode("ascii")).hexdigest()[:32], memory_id, auth.confirmed_at))
        return "".join(f"{a} {b} {c}\n" for a, b, c in keep)

    def _commit(self, decision, proposal, stamp, found, require_in_force, journal=True, extra_ops=None):
        """The one place a memory is written. Caller holds the write lock and has just evaluated `decision` against
        `found`."""
        if decision.decision != G.ACCEPT:           # the one writer writes only an accepted decision
            raise StoreError("decision_not_accept", (decision.decision, decision.reason))
        item = proposal.item
        links = tuple(decision.supersedes)
        new = replace(item, supersedes=links)
        ops = {self._path(new.id): serialize(new, proposal.evidence)}
        if extra_ops:
            ops.update(extra_ops)
            journal = True                      # the nonce record must land in the same transaction as the memory
        after = {i: s.item for i, s in found.items()}
        after[new.id] = new
        for oid in links:
            old = found[oid]                    # the fresh gate decision equals this one, so it names stored items
            if old.item.status != "active":
                raise StoreError("supersede_target_invalid", (oid,))
            if M.parse_ts(new.valid_from) <= M.parse_ts(old.item.valid_from):
                raise StoreError("supersession_chronology", (oid,))
            if M.parse_ts(stamp) < M.parse_ts(old.item.updated):
                raise StoreError("chronology_updated", (oid,))
            # the gate only names memories still in force at the new one's start, so their end (if any) lies
            # after it: the new valid_from can only shorten it
            ended = replace(old.item, status="superseded", updated=stamp, valid_to=new.valid_from)
            errs = M.validate(ended)
            if errs:
                raise StoreError("supersession_chronology", (oid,) + errs)
            after[oid] = ended
            ops[self._path(oid)] = serialize(ended, old.evidence)
        self._guard_resolution(list(after.values()), new, stamp, require_in_force=require_in_force)
        if journal:
            self._install(ops)
        else:                                   # one new file, nothing to roll back: a single atomic replace
            try:
                _write_atomic(self._path(new.id), ops[self._path(new.id)])
            except OSError as e:
                raise StoreError("write_failed") from e
        return Applied(new.id, tuple(sorted(links)))

    def _guard_resolution(self, items, new, now, require_in_force=False):
        """The store as it WOULD be must resolve cleanly: no rejected record, no conflict on the new item's key
        involving it, and (for a supersession) the new item is the one in force."""
        at = now if M.parse_ts(now) >= M.parse_ts(new.valid_from) else new.valid_from
        res = M.resolve(items, at)
        if res.rejected:
            raise StoreError("postcondition_failed", tuple(f"{i}:{','.join(e)}" for i, e in res.rejected))
        key = M.key_of(new)
        if any(c.key == key and any(x.id == new.id for x in c.candidates) for c in res.conflicts):
            raise StoreError("postcondition_failed", ("conflict",))
        if require_in_force and new.id not in [x.id for x in res.active]:
            raise StoreError("postcondition_failed", ("new_item_not_in_force",))

    def _install(self, ops):
        """Journal the previous content of every file, install all of them, mark committed, drop the journal. Any
        failure rolls back to the journalled state."""
        jpath = os.path.join(self.dir, _JOURNAL)
        journal = {"state": "prepared", "ops": []}
        for path in sorted(ops):
            journal["ops"].append({"file": os.path.basename(path), "prior": _read(path) if os.path.exists(path) else None})
        _write_atomic(jpath, _js(journal) + "\n")
        try:
            for path in sorted(ops):
                _write_atomic(path, ops[path])
        except BaseException as e:
            try:
                self._rollback(journal["ops"])
                os.unlink(jpath)
            except (StoreError, OSError):
                raise StoreError("write_failed_recovery_pending") from e
            raise StoreError("write_failed") from e
        journal["state"] = "committed"
        _write_atomic(jpath, _js(journal) + "\n")
        os.unlink(jpath)
        _sync_dir(self.dir)


class _Lock:
    """The advisory lock on the memory folder. Writers create the folder; a reader of a folder that does not exist
    takes no lock and sees an empty store."""

    def __init__(self, store, write):
        self.store, self.write, self.fd = store, write, None

    def __enter__(self):
        if self.write:
            os.makedirs(self.store.dir, exist_ok=True)
        elif not os.path.isdir(self.store.dir):
            return self
        self.fd = os.open(os.path.join(self.store.dir, _LOCK), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            self.store._recover()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
        return False
