"""TEST-ONLY harness for the Memory 2.0 receipt log (Phase 8.2c-2). NOT part of the product and never imported by
production code. It builds a throwaway vault and a throwaway receipts folder in a temp directory and runs the commit
SEQUENCE the later execute layer will run (snapshot, intent, append, store install, snapshot, done), one step at a time, so
a test can stop, fail or crash between any two steps. Nothing here reads or writes the real vault."""
import hashlib
import os
import tempfile
from types import SimpleNamespace

from backtalk import memory as M
from backtalk import memory_gate as G
from backtalk import memory_receipts as RC
from backtalk import memory_store as S
from tests.fake_clock import FakeClock
from tests.store_arrange import arrange

NOW = "2026-02-01T00:00:00Z"
CTX = "c" * 64
LOCK_HASH = "a" * 64
DAY = "2026-01-10T00:00:00Z"


class Crash(BaseException):
    """The process dying: nothing after it runs."""


def mid(n):
    return f"mem_{n:012x}"


def day(n, hour=0):
    return f"2026-01-{n:02d}T{hour:02d}:00:00Z"


def item(n=1, **over):
    d = {"id": mid(n), "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": DAY, "updated": DAY, "source": {"kind": "user", "ref": "turn 41"}, "confidence": "stated",
         "valid_from": DAY, "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


def ev(value="San Jose"):
    return f"I live in {value} now"


class Rig:
    """One temp vault + one temp receipts folder + a store with a deterministic clock."""

    def __init__(self, case):
        self.tmp = tempfile.TemporaryDirectory()
        case.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        prod = os.path.realpath(S.REAL_VAULT)
        case.assertFalse(self.root == prod or self.root.startswith(prod + os.sep), "tests must never use the real vault")
        self.vault = os.path.join(self.root, "vault")
        os.mkdir(self.vault)
        self.rdir = os.path.join(self.root, "receipts")
        os.mkdir(self.rdir, 0o700)
        os.chmod(self.rdir, 0o700)
        self.clock = FakeClock(NOW)
        self.store = S.MemoryStore(self.vault, clock=self.clock)
        self.n = 0

    # -- paths and files --
    @property
    def memdir(self):
        return os.path.join(self.vault, S.MEMORY_DIR)

    def path(self, name=""):
        return os.path.join(self.memdir, name)

    def rpath(self, name=RC.RECEIPTS):
        return os.path.join(self.rdir, name)

    @property
    def log(self):
        return RC.ReceiptLog(self.rdir)

    def memory_files(self):
        out = {}
        if os.path.isdir(self.memdir):
            for name in sorted(os.listdir(self.memdir)):
                p = os.path.join(self.memdir, name)
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        out[name] = f.read()
        return out

    # -- arranging --
    def seed(self, n, **over):
        """A stored memory written directly (no gate), as the store tests do. Returns the item."""
        base = {"value": "Kansas City", "valid_from": day(1), "created": day(1), "updated": day(1)}
        base.update(over)
        it = item(n, **base)
        arrange(self.store, it, ev(base["value"]))
        return it

    def baseline(self):
        return self.log.create(self.store.snapshot(), self.clock.stamp())

    def plan(self, n=1, operation=None, evidence=None, **over):
        """A duck-typed plan for installing item n through the real gate against what is stored now."""
        it = item(n, **over)
        evidence = evidence if evidence is not None else ev(over.get("value", "San Jose"))
        proposal = G.Proposal(it, evidence)
        decision = G.evaluate(proposal, [s.item for s in self.store.list_items()], self.clock.stamp())
        assert decision.decision == G.ACCEPT, (decision.decision, decision.reason)
        op = operation or ("apply" if decision.supersedes else "put")
        return SimpleNamespace(candidate="goal_cash_buyer_list_deep", lock_hash=LOCK_HASH, context_digest=CTX,
                               operation=op, item=it, evidence=evidence, decision=decision, proposal=proposal,
                               item_digest=S.item_digest(it, evidence, decision.supersedes, op))

    def authorization(self, plan, **over):
        self.n += 1
        d = {"operation": plan.operation, "item_digest": plan.item_digest, "state_digest": self.store.state_digest(),
             "nonce": hashlib.sha256(f"nonce{self.n}".encode()).hexdigest()[:32], "confirmed_at": self.clock.stamp(),
             "context_digest": CTX, "method": "typed-terminal"}
        d.update(over)
        return S.Authorization(**d)

    # -- the sequence, one step at a time --
    def begin(self, plan, auth, apply_now=None, snapshot=None):
        """Steps 1-4: snapshot, build the intent, append it. Returns the appended intent record."""
        snap = snapshot or self.store.snapshot()
        assert snap.ok, snap.reason
        fields = RC.intent_fields(snap, plan, nonce=auth.nonce, confirmed_at=auth.confirmed_at,
                                  apply_now=apply_now or self.clock.stamp())
        self.last_apply_now = fields["apply_now"]
        return self.log.append_intent(fields, self.clock.stamp(), self.store)

    def install(self, plan, auth):
        """Step 5: the store install itself (authorized, as the real vault requires)."""
        if plan.operation == "put":
            return self.store.put(plan.item, plan.evidence, now=self.last_apply_now, authorization=auth)
        return self.store.apply(plan.decision, plan.proposal, self.last_apply_now, authorization=auth)

    def finish(self, intent):
        """Steps 6-7: a fresh snapshot, then the done record."""
        snap = self.store.snapshot()
        assert snap.ok, snap.reason
        return self.log.append(RC.done_fields(intent["seq"], snap), self.clock.stamp())

    def commit(self, plan, auth=None):
        auth = auth or self.authorization(plan)
        intent = self.begin(plan, auth)
        self.install(plan, auth)
        return intent, self.finish(intent), auth

    def fresh_store(self):
        """A store object with no shared state: what a new process would build."""
        return S.MemoryStore(self.vault, clock=self.clock)


# ---- helpers shared by the receipt tests ---------------------------------------------------------------------------

def tree_state(root):
    """Everything observable about a tree without following links: names, types, modes, owners, sizes, mtime/ctime,
    inodes, link counts, link targets and the sha256 of every regular file's bytes."""
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


def envelope_free(rec):
    return {k: v for k, v in rec.items() if k not in ("seq", "prev", "at", "hash")}


def write_log(rig, fields_list, at=NOW, sidecar=True):
    """Write a receipts file from a list of field dicts, chaining them with the review ledger's own `new_record`, so a
    test can craft any state machine. Mode 0600, with its sidecar lock. Returns the records."""
    from backtalk import memory_review as R
    records = []
    for fields in fields_list:
        records.append(R.new_record(records, at, fields))
    text = "".join(R.canonical_json(r) + "\n" for r in records)
    with open(rig.rpath(), "wb") as f:
        f.write(text.encode("utf-8"))
    os.chmod(rig.rpath(), 0o600)
    if sidecar and not os.path.exists(rig.rpath(RC.LOCK)):
        with open(rig.rpath(RC.LOCK), "wb"):
            pass
        os.chmod(rig.rpath(RC.LOCK), 0o600)
    return records


def read_log_bytes(rig):
    with open(rig.rpath(), "rb") as f:
        return f.read()


def snapshot_of(files, nonces=(), items=()):
    """A synthetic successful Snapshot (pure value) for tests of the proof function."""
    files = tuple(sorted(files))
    nonces = tuple(sorted(nonces))
    return S.Snapshot(True, "ok", "initialized", files, tuple(items), nonces, S.manifest_digest(files),
                      S.snapshot_digest(files, nonces))
