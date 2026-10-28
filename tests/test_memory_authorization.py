"""Memory 2.0, Phase 8.1B: the store's authorization boundary (Authorization, the gate's Confirmation, the nonce log).
Every test runs on a throwaway vault in a temp directory with a deterministic FakeClock; the "real vault" tests point the
store's REAL_VAULT constant at a temp folder, so the actual vault is never opened.

WHAT THESE TESTS DO NOT SHOW: an Authorization is plain data built from public digests. Code running as the same user can
forge one, call private install methods, or write the files directly. These tests prove the PUBLIC API refuses stale,
mismatched, expired, replayed and unauthorized writes; they do not (and cannot) prove who typed anything."""
import ast
import glob
import hashlib
import inspect
import os
import shutil
import stat
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest import mock

from backtalk import memory as M
from backtalk import memory_gate as G
from backtalk import memory_store as S
from tests.fake_clock import FakeClock
from tests.store_arrange import arrange

NOW = "2026-02-01T00:00:00Z"
EV = "I live in San Jose now"
CTX = "c" * 64


def day(n, hour=0):
    return f"2026-01-{n:02d}T{hour:02d}:00:00Z"


def mid(n):
    return f"mem_{n:012x}"


def raw(n=1, **over):
    d = {"id": mid(n), "type": "fact", "subject": "user", "attribute": "residence", "value": "San Jose",
         "created": day(10), "updated": day(10), "source": {"kind": "user", "ref": "turn 41"},
         "confidence": "stated", "valid_from": day(10), "status": "active", "sensitivity": "normal"}
    d.update(over)
    return d


def item(n=1, **over):
    it, errs = M.parse(raw(n, **over))
    assert it is not None, errs
    return it


def nonce_hash(nonce):
    return hashlib.sha256(nonce.encode("ascii")).hexdigest()[:32]


class Crash(BaseException):
    """The process dying: nothing after it runs."""


class AuthCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        prod = os.path.realpath(S.REAL_VAULT)
        self.assertFalse(self.root == prod or self.root.startswith(prod + os.sep), "tests must never use the real vault")
        self.real_mem = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)
        self.real_existed = os.path.exists(self.real_mem)
        self.clock = FakeClock(NOW)
        self.store = S.MemoryStore(self.root, clock=self.clock)
        self.n = 0

    def tearDown(self):
        self.assertEqual(os.path.exists(self.real_mem), self.real_existed, "the real vault was touched")

    @property
    def memdir(self):
        return os.path.join(self.root, S.MEMORY_DIR)

    @property
    def logpath(self):
        return os.path.join(self.memdir, ".authorizations")

    def files(self, store_dir=None):
        d = store_dir or self.memdir
        out = {}
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p) and name != ".lock":
                    with open(p, encoding="utf-8", newline="") as f:
                        out[name] = f.read()
        return out

    def log_rows(self):
        text = self.files().get(".authorizations", "")
        return [line.split(" ") for line in text.split("\n") if line]

    def auth(self, it, ev=EV, op="put", links=(), **over):
        self.n += 1
        d = {"operation": op, "item_digest": S.item_digest(it, ev, links, op), "state_digest": self.store.state_digest(),
             "nonce": hashlib.sha256(f"nonce{self.n}".encode()).hexdigest()[:32], "confirmed_at": self.clock.stamp(),
             "context_digest": CTX, "method": "typed-terminal"}
        d.update(over)
        return S.Authorization(**d)

    def refused(self, reason, fn, *a, **k):
        with self.assertRaises(S.StoreError) as cm:
            fn(*a, **k)
        self.assertEqual(cm.exception.reason, reason, str(cm.exception))
        return cm.exception

    def invalid(self, code, fn, *a, **k):
        e = self.refused("authorization_invalid", fn, *a, **k)
        self.assertEqual(e.details, (code,))
        return e

    def untouched(self, fn, *a, **k):
        """Run `fn`, expecting a refusal; every file in the folder (journal and nonce log included) must be unchanged."""
        before = self.files()
        with self.assertRaises(S.StoreError) as cm:
            fn(*a, **k)
        self.assertEqual(self.files(), before, "a refused write changed the store")
        return cm.exception

    def supersede_setup(self):
        arrange(self.store, item(2, value="Kansas City", valid_from=day(1), created=day(1), updated=day(1)),
                "I live in Kansas City")
        prop = G.Proposal(item(1), EV)
        d = G.evaluate(prop, [s.item for s in self.store.list_items()], NOW)
        self.assertEqual((d.decision, d.supersedes), (G.ACCEPT, (mid(2),)))
        return prop, d


class WindowTests(AuthCase):
    def test_an_authorization_made_now_is_accepted_and_the_item_is_stored(self):
        it = item(1)
        self.assertEqual(self.store.put(it, EV, authorization=self.auth(it)), mid(1))
        self.assertEqual(self.store.load(mid(1)).item, it)

    def test_the_window_is_exactly_120_seconds_inclusive(self):
        for age, ok in ((0, True), (1, True), (119, True), (120, True), (121, False), (600, False)):
            with self.subTest(age=age):
                shutil.rmtree(self.memdir, ignore_errors=True)
                self.clock.t = M.parse_ts(NOW)
                it = item(1)
                a = self.auth(it)
                self.clock.advance(age)
                if ok:
                    self.assertEqual(self.store.put(it, EV, authorization=a), mid(1))
                else:
                    self.invalid("expired", self.store.put, it, EV, authorization=a)

    def test_an_authorization_from_the_future_is_refused(self):
        it = item(1)
        a = self.auth(it, confirmed_at=self.clock.stamp(1))
        self.untouched(self.store.put, it, EV, authorization=a)
        self.invalid("not_yet_valid", self.store.put, it, EV, authorization=a)

    def test_the_window_constant_is_120_and_not_a_field(self):
        self.assertEqual(S.AUTH_WINDOW, 120)
        self.assertNotIn("expires_at", {f for f in S.Authorization.__dataclass_fields__})

    def test_confirmed_at_must_be_a_strict_utc_timestamp(self):
        it = item(1)
        for bad in ("2026-02-01 00:00:00", "2026-02-01T00:00:00+00:00", "2026-02-01T00:00:00", "2026-13-01T00:00:00Z",
                    "2026-02-30T00:00:00Z", "", " 2026-02-01T00:00:00Z", "2026-02-01T00:00:00.5Z", 1769904000,
                    datetime(2026, 2, 1, tzinfo=timezone.utc), None):
            with self.subTest(bad=bad):
                self.untouched(self.store.put, it, EV, authorization=self.auth(it, confirmed_at=bad))
                self.invalid("malformed", self.store.put, it, EV, authorization=self.auth(it, confirmed_at=bad))

    def test_the_callers_now_must_be_close_to_the_store_clock(self):
        it = item(1)
        a = self.auth(it)
        self.invalid("now_not_current", self.store.put, it, EV, now="2026-01-31T00:00:00Z", authorization=a)
        self.invalid("now_not_current", self.store.put, it, EV, now="2026-02-02T00:00:00Z", authorization=a)
        self.assertEqual(self.store.put(it, EV, now=self.clock.stamp(-120), authorization=a), mid(1))

    def test_an_authorized_put_with_no_now_uses_the_store_clock_not_the_items_updated_stamp(self):
        it = item(1)                                      # updated 2026-01-10, 22 days before the clock
        self.assertEqual(self.store.put(it, EV, authorization=self.auth(it)), mid(1))

    def test_an_unauthorized_put_reads_no_clock(self):
        boom = mock.Mock(side_effect=AssertionError("clock read"))
        store = S.MemoryStore(self.root, clock=boom)
        store.put(item(1), EV)
        boom.assert_not_called()

    def test_the_clock_is_read_exactly_once_per_authorized_write(self):
        calls = mock.Mock(side_effect=lambda: self.clock())
        store = S.MemoryStore(self.root, clock=calls)
        it = item(1)
        store.put(it, EV, authorization=self.auth(it))
        self.assertEqual(calls.call_count, 1)

    def test_a_naive_or_failing_clock_fails_closed(self):
        it = item(1)
        a = self.auth(it)
        naive = S.MemoryStore(self.root, clock=lambda: datetime(2026, 2, 1))
        self.untouched(naive.put, it, EV, authorization=a)
        self.refused("invalid_time", naive.put, it, EV, authorization=a)

        def broken():
            raise RuntimeError("no clock")
        self.refused("clock_failed", S.MemoryStore(self.root, clock=broken).put, it, EV, authorization=a)
        self.assertEqual(self.files(), {})

    def test_sub_second_clock_values_are_truncated_not_rounded(self):
        it = item(1)
        a = self.auth(it)
        store = S.MemoryStore(self.root, clock=lambda: self.clock().replace(microsecond=999999))
        self.assertEqual(store.put(it, EV, authorization=a), mid(1))


class RealVaultTests(AuthCase):
    def real_shaped(self, **kw):
        patcher = mock.patch.object(S, "REAL_VAULT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        return S.MemoryStore(self.root, allow_real_vault=True, **kw)

    def test_a_real_vault_store_still_needs_the_explicit_flag(self):
        with mock.patch.object(S, "REAL_VAULT", self.root):
            self.refused("real_vault_refused", S.MemoryStore, self.root)

    def test_a_real_vault_store_refuses_a_custom_clock(self):
        with mock.patch.object(S, "REAL_VAULT", self.root):
            self.refused("clock_not_allowed", S.MemoryStore, self.root, allow_real_vault=True, clock=self.clock)
            self.refused("clock_not_allowed", S.MemoryStore, os.path.join(self.root, "sub"), allow_real_vault=True,
                         clock=lambda: None)

    def test_a_real_vault_store_uses_the_system_clock(self):
        store = self.real_shaped()
        self.assertIs(store._clock, S._system_utc)
        self.assertTrue(store.require_authorization)

    def test_a_real_vault_store_refuses_every_write_without_an_authorization(self):
        store = self.real_shaped()
        self.refused("authorization_required", store.put, item(1), EV)
        prop = G.Proposal(item(1), EV)
        self.refused("authorization_required", store.apply, G.evaluate(prop, [], NOW), prop, NOW)
        self.assertFalse(os.path.exists(self.memdir))             # refused before the lock: not even the folder appears

    def test_a_real_vault_store_writes_with_a_valid_authorization_on_the_system_clock(self):
        store = self.real_shaped()
        it = item(1)
        stamp = S._system_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
        a = S.Authorization("put", S.item_digest(it, EV, (), "put"), store.state_digest(), "d" * 32, stamp, CTX,
                            "typed-terminal")
        self.assertEqual(store.put(it, EV, authorization=a), mid(1))
        self.assertEqual(len(self.log_rows()), 1)

    def test_a_temp_store_does_not_require_one_for_ordinary_items(self):
        self.assertFalse(self.store.require_authorization)
        self.assertEqual(self.store.put(item(1), EV), mid(1))
        self.assertEqual(self.files().keys() - {mid(1) + ".md"}, set())


class FormAndBindingTests(AuthCase):
    def test_anything_that_is_not_an_authorization_is_malformed(self):
        it = item(1)
        good = self.auth(it)
        for bad in ({}, dict(good.__dict__), "x", 5, [good], object()):
            with self.subTest(bad=type(bad).__name__):
                self.untouched(self.store.put, it, EV, authorization=bad)
                self.invalid("malformed", self.store.put, it, EV, authorization=bad)

    def test_each_field_is_checked_for_shape(self):
        it = item(1)
        for field, bad in (("operation", "delete"), ("operation", 5), ("item_digest", "A" * 64), ("item_digest", "a" * 63),
                           ("state_digest", "g" * 64), ("state_digest", None), ("nonce", "A" * 32), ("nonce", "a" * 31),
                           ("nonce", "a" * 33), ("context_digest", "c" * 63), ("context_digest", ""), ("method", "telegram"),
                           ("method", None)):
            with self.subTest(field=field, bad=bad):
                a = self.auth(it, **{field: bad})
                self.untouched(self.store.put, it, EV, authorization=a)
                self.invalid("malformed", self.store.put, it, EV, authorization=a)

    def test_a_guessed_digest_does_not_match(self):
        it = item(1)
        self.invalid("item_mismatch", self.store.put, it, EV, authorization=self.auth(it, item_digest="0" * 64))
        self.invalid("state_changed", self.store.put, it, EV, authorization=self.auth(it, state_digest="0" * 64))

    def test_it_binds_the_candidate_value_evidence_source_and_confidence(self):
        base = item(1)
        a = self.auth(base)
        for label, other, ev in (("another id", item(3), EV), ("value", item(1, value="Oakland"), "I live in Oakland now"),
                                 ("evidence", base, "I live in San Jose these days"),
                                 ("source ref", item(1, source={"kind": "user", "ref": "turn 42"}), EV),
                                 ("source kind", item(1, source={"kind": "session", "ref": "turn 41"}, confidence="inferred"),
                                  "he said he lives in San Jose now"),
                                 ("subject", item(1, subject="max"), EV), ("attribute", item(1, attribute="home"), EV),
                                 ("valid_from", item(1, valid_from=day(11)), EV), ("created", item(1, created=day(9)), EV),
                                 ("sensitivity", item(1, sensitivity="private"), EV)):
            with self.subTest(changed=label):
                e = self.untouched(self.store.put, other, ev, authorization=a)
                self.assertIn(e.reason, ("authorization_invalid", "confirmation_required"))
                if e.reason == "authorization_invalid":
                    self.assertEqual(e.details, ("item_mismatch",))

    def test_it_binds_the_operation_both_ways(self):
        it = item(1)
        self.invalid("operation_mismatch", self.store.put, it, EV, authorization=self.auth(it, op="apply"))
        prop = G.Proposal(it, EV)
        d = G.evaluate(prop, [], NOW)
        self.invalid("operation_mismatch", self.store.apply, d, prop, NOW, authorization=self.auth(it, op="put"))
        self.assertEqual(self.files(), {})

    def test_the_item_digest_includes_the_supersession_links(self):
        prop, d = self.supersede_setup()
        before = self.files()
        wrong = self.auth(prop.item, op="apply", links=())
        self.invalid("item_mismatch", self.store.apply, d, prop, NOW, authorization=wrong)
        self.assertEqual(self.files(), before)
        ok = self.auth(prop.item, op="apply", links=d.supersedes)
        applied = self.store.apply(d, prop, NOW, authorization=ok)
        self.assertEqual((applied.memory_id, applied.superseded), (mid(1), (mid(2),)))
        self.assertEqual(self.store.load(mid(2)).item.status, "superseded")
        self.assertEqual(len(self.log_rows()), 1)

    def test_it_binds_the_current_store_state(self):
        it = item(1)
        a = self.auth(it)
        self.store.put(item(5, attribute="other"), "I live in San Jose now")      # the store changes after the yes
        self.untouched(self.store.put, it, EV, authorization=a)
        self.invalid("state_changed", self.store.put, it, EV, authorization=a)

    def test_a_deleted_or_edited_file_changes_the_state_digest(self):
        self.store.put(item(5, attribute="other"), EV)
        d0 = self.store.state_digest()
        path = os.path.join(self.memdir, mid(5) + ".md")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text + " ")
        self.assertNotEqual(self.store.state_digest(), d0)
        os.unlink(path)
        self.assertNotEqual(self.store.state_digest(), d0)

    def test_the_state_digest_ignores_the_nonce_log_and_is_stable_for_an_empty_store(self):
        empty = self.store.state_digest()
        self.assertEqual(S.MemoryStore(os.path.join(self.root, "elsewhere")).state_digest(), empty)
        it = item(1)
        self.store.put(it, EV, authorization=self.auth(it))
        with_log = self.store.state_digest()
        os.unlink(self.logpath)
        self.assertEqual(self.store.state_digest(), with_log)

    def test_a_stale_decision_is_still_refused_for_an_authorized_apply(self):
        prop, d = self.supersede_setup()
        a = self.auth(prop.item, op="apply", links=d.supersedes)
        arrange(self.store, item(7, value="Reno", attribute="residence", valid_from=day(5), created=day(5), updated=day(5)),
                "I live in Reno")
        e = self.untouched(self.store.apply, d, prop, NOW, authorization=self.auth(prop.item, op="apply", links=d.supersedes))
        self.assertEqual(e.reason, "stale_decision")
        self.assertEqual(a.operation, "apply")

    def test_rejected_conflicting_and_unsupported_items_are_refused_even_with_a_valid_authorization(self):
        self.store.put(item(2, value="Austin"), "I live in Austin")
        for label, it, ev, want in (("conflict", item(1), EV, (G.CONFLICT, "conflict")),
                                    ("note-sourced stated", item(3, attribute="x1", source={"kind": "note", "ref": "N.md#L1"}),
                                     "I like green tea a lot", (G.REJECT, "stated_requires_user_source")),
                                    ("evidence mismatch", item(4, attribute="x2"), "I like tea", (G.REJECT, "evidence_does_not_support_value"))):
            with self.subTest(case=label):
                e = self.untouched(self.store.put, it, ev, authorization=self.auth(it, ev))
                self.assertEqual((e.reason, e.details), ("decision_not_accept", want))
        self.assertEqual(sorted(self.files()), [mid(2) + ".md"])

    def test_a_supersession_cannot_arrive_through_put_even_when_authorized(self):
        prop, d = self.supersede_setup()
        a = self.auth(prop.item, links=d.supersedes)
        self.untouched(self.store.put, prop.item, EV, authorization=a)
        self.assertEqual(self.store.load(mid(2)).item.status, "active")


class ReplayTests(AuthCase):
    def test_the_same_authorization_cannot_be_used_twice(self):
        it = item(1)
        a = self.auth(it)
        self.store.put(it, EV, authorization=a)
        again = item(1, attribute="x")                            # even for a different item the digests refuse it
        e = self.untouched(self.store.put, it, EV, authorization=a)
        self.assertEqual(e.reason, "duplicate_id")                # the id is spent, which refuses a straight replay first
        self.invalid("item_mismatch", self.store.put, item(9, attribute="x"), EV, authorization=a)

    def test_the_nonce_log_blocks_a_replay_when_the_state_matches_again(self):
        it = item(1)
        a = self.auth(it)
        self.store.put(it, EV, authorization=a)
        os.unlink(os.path.join(self.memdir, mid(1) + ".md"))      # the state digest is back to what the yes covered
        self.assertEqual(self.store.state_digest(), a.state_digest)
        e = self.untouched(self.store.put, it, EV, authorization=a)
        self.assertEqual(e.details, ("reused",))

    def test_a_reused_nonce_on_a_new_item_is_refused(self):
        it, other = item(1), item(2, attribute="other")
        a = self.auth(it)
        self.store.put(it, EV, authorization=a)
        b = self.auth(other, nonce=a.nonce)
        e = self.untouched(self.store.put, other, EV, authorization=b)
        self.assertEqual(e.details, ("reused",))

    def test_a_captured_authorization_cannot_authorize_another_candidate(self):
        a, b = item(1), item(2, attribute="other")
        auth_a = self.auth(a)
        self.store.put(a, EV, authorization=auth_a)
        self.untouched(self.store.put, b, EV, authorization=auth_a)

    def test_four_threads_with_one_authorization_store_exactly_one(self):
        it = item(1)
        a = self.auth(it)
        stores = [S.MemoryStore(self.root, clock=self.clock) for _ in range(4)]
        out, lock = [], threading.Lock()

        def run(store):
            try:
                store.put(it, EV, authorization=a)
                r = "ok"
            except S.StoreError as e:
                r = e.reason
            with lock:
                out.append(r)
        threads = [threading.Thread(target=run, args=(s,)) for s in stores]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(out), ["duplicate_id", "duplicate_id", "duplicate_id", "ok"], out)
        self.assertEqual(len(self.log_rows()), 1)
        self.assertEqual(self.store.verify(), ())

    def test_each_successful_write_invalidates_every_other_outstanding_authorization(self):
        a, b = item(1, attribute="a1"), item(2, attribute="a2")
        auth_a, auth_b = self.auth(a), self.auth(b)               # both made against the same (empty) state
        self.store.put(a, EV, authorization=auth_a)
        self.invalid("state_changed", self.store.put, b, EV, authorization=auth_b)


class NonceLogTests(AuthCase):
    def test_the_log_holds_a_hash_the_id_and_the_time_and_nothing_else(self):
        it = item(1)
        a = self.auth(it)
        self.store.put(it, EV, authorization=a)
        self.assertEqual(self.log_rows(), [[nonce_hash(a.nonce), mid(1), a.confirmed_at]])
        text = self.files()[".authorizations"]
        for secret in (a.nonce, "San Jose", EV, a.item_digest, a.state_digest, CTX):
            self.assertNotIn(secret, text)

    def test_the_log_and_the_memory_file_are_private_to_the_owner(self):
        it = item(1)
        self.store.put(it, EV, authorization=self.auth(it))
        for path in (self.logpath, os.path.join(self.memdir, mid(1) + ".md")):
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600, path)

    def test_old_records_are_pruned_and_recent_ones_kept_in_the_same_write(self):
        os.makedirs(self.memdir)
        old = f"{'1' * 32} {mid(8)} 2026-01-30T23:59:59Z\n"        # 24h + 1s before the clock
        edge = f"{'2' * 32} {mid(9)} 2026-01-31T00:00:00Z\n"       # exactly 24h before
        recent = f"{'3' * 32} {mid(7)} 2026-01-31T12:00:00Z\n"
        with open(self.logpath, "w") as f:
            f.write(old + edge + recent)
        it = item(1)
        a = self.auth(it)
        self.store.put(it, EV, authorization=a)
        self.assertEqual([r[0] for r in self.log_rows()], ["2" * 32, "3" * 32, nonce_hash(a.nonce)])

    def test_a_malformed_log_stops_authorized_writes_and_is_reported_by_verify(self):
        os.makedirs(self.memdir)
        it = item(1)
        for text in ("garbage\n", f"{'1' * 32} {mid(1)} not-a-time\n", f"{'z' * 32} {mid(1)} {NOW}\n",
                     f"{'1' * 32} {mid(1)}\n", f"{'1' * 32} {mid(1)} {NOW} extra\n", f"{'1' * 32}  {mid(1)} {NOW}\n"):
            with self.subTest(text=text):
                with open(self.logpath, "w") as f:
                    f.write(text)
                e = self.untouched(self.store.put, it, EV, authorization=self.auth(it))
                self.assertEqual((e.reason, e.details), ("authorization_invalid", ("authorization_log_corrupt",)))
                self.assertIn("authorizations:corrupt", self.store.verify())

    def test_a_malformed_log_does_not_stop_reads_or_unauthorized_writes(self):
        self.store.put(item(5, attribute="other"), EV)
        with open(self.logpath, "w") as f:
            f.write("garbage\n")
        self.assertEqual(len(self.store.list_items()), 1)
        self.store.put(item(6, attribute="third"), EV)

    def test_a_missing_or_valid_log_is_not_a_problem_and_is_not_an_unexpected_file(self):
        self.assertEqual(self.store.verify(), ())
        it = item(1)
        self.store.put(it, EV, authorization=self.auth(it))
        self.assertEqual(self.store.verify(), ())
        self.assertEqual(len(self.store.list_items()), 1)

    def test_an_unauthorized_write_never_touches_the_log(self):
        self.store.put(item(1), EV)
        self.assertNotIn(".authorizations", self.files())


class PrivateTests(AuthCase):
    def priv(self, n=1, **over):
        return item(n, sensitivity="private", **over)

    def test_a_private_item_needs_an_authorization_even_in_a_temp_store(self):
        it = self.priv()
        e = self.untouched(self.store.put, it, EV)
        self.assertEqual(e.reason, "confirmation_required")
        self.assertEqual([n for n in self.files() if n != ".lock"], [])

    def test_a_private_item_is_stored_with_its_authorization(self):
        it = self.priv()
        self.assertEqual(self.store.put(it, EV, authorization=self.auth(it)), mid(1))
        self.assertEqual(self.store.load(mid(1)).item.sensitivity, "private")

    def test_apply_accepts_the_requires_confirmation_decision_only_with_an_authorization(self):
        it = self.priv()
        prop = G.Proposal(it, EV)
        d = G.evaluate(prop, [], NOW)
        self.assertEqual((d.decision, d.reason), (G.REQUIRE_CONFIRMATION, "private_requires_confirmation"))
        self.untouched(self.store.apply, d, prop, NOW)
        self.assertEqual(self.store.apply(d, prop, NOW, authorization=self.auth(it, op="apply")).memory_id, mid(1))

    def test_an_authorization_for_another_private_item_is_refused(self):
        a, b = self.priv(1, attribute="a1"), self.priv(2, attribute="a2")
        e = self.untouched(self.store.put, b, EV, authorization=self.auth(a))
        self.assertEqual(e.details, ("item_mismatch",))

    def test_a_forged_accept_for_a_private_item_is_still_refused(self):
        it = self.priv()
        prop = G.Proposal(it, EV)
        forged = replace(G.evaluate(prop, [], NOW), decision=G.ACCEPT, reason="accepted_confirmed")
        e = self.untouched(self.store.apply, forged, prop, NOW)
        self.assertEqual(e.reason, "stale_decision")
        self.untouched(self.store.apply, forged, prop, NOW, authorization=self.auth(it, op="apply"))

    def test_neither_public_method_accepts_a_confirmation(self):
        for fn in (S.MemoryStore.put, S.MemoryStore.apply):
            self.assertNotIn("confirmation", inspect.signature(fn).parameters)
        it = self.priv()
        digest = G.proposal_digest(it, EV, ())
        with self.assertRaises(TypeError):
            self.store.put(it, EV, confirmation=G.Confirmation(digest))
        prop = G.Proposal(it, EV)
        with self.assertRaises(TypeError):
            self.store.apply(G.evaluate(prop, [], NOW), prop, NOW, confirmation=G.Confirmation(digest))
        self.assertFalse(os.path.exists(self.memdir))

    def test_a_valid_authorization_cannot_make_a_non_private_rejection_acceptable(self):
        it = item(1, source={"kind": "note", "ref": "N.md#L1"}, sensitivity="private")
        e = self.untouched(self.store.put, it, EV, authorization=self.auth(it))
        self.assertEqual((e.reason, e.details), ("decision_not_accept", (G.REJECT, "stated_requires_user_source")))


class GateConfirmationTests(unittest.TestCase):
    def prop(self, **over):
        return G.Proposal(item(1, sensitivity="private", **over), EV)

    def test_a_confirmation_is_immutable(self):
        c = G.Confirmation("a" * 64)
        self.assertEqual(c.kind, "private")
        with self.assertRaises(Exception):
            c.kind = "other"
        with self.assertRaises(Exception):
            c.item_digest = "b" * 64

    def test_no_confirmation_is_the_old_behavior(self):
        d = G.evaluate(self.prop(), [], NOW)
        self.assertEqual((d.decision, d.reason), (G.REQUIRE_CONFIRMATION, "private_requires_confirmation"))
        self.assertEqual(d, G.evaluate(self.prop(), [], NOW, None))

    def test_a_matching_confirmation_accepts_exactly_that_proposal(self):
        p = self.prop()
        c = G.Confirmation(G.proposal_digest(p.item, p.evidence, ()))
        d = G.evaluate(p, [], NOW, c)
        self.assertEqual((d.decision, d.reason, d.sensitivity), (G.ACCEPT, "accepted_confirmed", "private"))

    def test_a_confirmation_for_anything_else_changes_nothing(self):
        p = self.prop()
        good = G.proposal_digest(p.item, p.evidence, ())
        other = self.prop(value="Oakland")
        for c in (G.Confirmation("0" * 64), G.Confirmation(good, "other"), G.Confirmation(G.proposal_digest(other.item, other.evidence, ())),
                  G.Confirmation(G.proposal_digest(p.item, "I live in San Jose these days", ())),
                  G.Confirmation(G.proposal_digest(p.item, p.evidence, (mid(9),))), good, {"item_digest": good}, object()):
            with self.subTest(c=repr(c)[:40]):
                self.assertEqual(G.evaluate(p, [], NOW, c).decision, G.REQUIRE_CONFIRMATION)

    def test_a_confirmation_never_overrides_a_rejection_or_a_conflict(self):
        bad = G.Proposal(item(1, sensitivity="private", source={"kind": "note", "ref": "N.md#L1"}), EV)
        c = G.Confirmation(G.proposal_digest(bad.item, bad.evidence, ()))
        self.assertEqual((G.evaluate(bad, [], NOW, c).decision, G.evaluate(bad, [], NOW, c).reason), (G.REJECT, "stated_requires_user_source"))
        rival = item(2, value="Austin")
        p = self.prop()
        c = G.Confirmation(G.proposal_digest(p.item, p.evidence, ()))
        self.assertEqual(G.evaluate(p, [rival], NOW, c).decision, G.CONFLICT)

    def test_a_confirmation_on_a_non_private_proposal_is_ignored(self):
        p = G.Proposal(item(1), EV)
        d = G.evaluate(p, [], NOW, G.Confirmation(G.proposal_digest(p.item, p.evidence, ())))
        self.assertEqual((d.decision, d.reason), (G.ACCEPT, "accepted"))

    def test_the_digest_covers_the_links_and_every_field(self):
        p = self.prop()
        base = G.proposal_digest(p.item, p.evidence, ())
        self.assertNotEqual(base, G.proposal_digest(p.item, p.evidence, (mid(2),)))
        self.assertEqual(G.proposal_digest(p.item, p.evidence, (mid(3), mid(2))), G.proposal_digest(p.item, p.evidence, (mid(2), mid(3))))
        for over in ({"value": "Reno"}, {"attribute": "home"}, {"subject": "max"}, {"valid_from": day(11)},
                     {"source": {"kind": "user", "ref": "turn 42"}}):
            self.assertNotEqual(base, G.proposal_digest(item(1, sensitivity="private", **over), p.evidence, ()))
        self.assertEqual(len(base), 64)


class InstallAtomicityTests(AuthCase):
    """The memory file and the nonce log are installed by the existing journalled _install: they land together or not at
    all, at every interruption point, and an authorization is reusable only after a PROVEN full rollback."""

    def run_crash(self, crash_at, it, a):
        real, count = S._write_atomic, [0]

        def dying(path, text):
            count[0] += 1
            if count[0] == crash_at:
                raise Crash()
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", dying), mock.patch.object(S.MemoryStore, "_rollback", side_effect=Crash()):
            with self.assertRaises(Crash):
                self.store.put(it, EV, authorization=a)

    def consistent(self, memory_id, nonce):
        fs = self.files()
        has_file = memory_id + ".md" in fs
        has_nonce = nonce_hash(nonce) in fs.get(".authorizations", "")
        self.assertEqual(has_file, has_nonce, f"memory file present={has_file} but nonce recorded={has_nonce}")
        return has_file

    def test_the_log_is_part_of_the_journal(self):
        it = item(1)
        seen = []
        real = S._write_atomic

        def spy(path, text):
            seen.append(os.path.basename(path))
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", spy):
            self.store.put(it, EV, authorization=self.auth(it))
        self.assertEqual(seen, [".txn.json", ".authorizations", mid(1) + ".md", ".txn.json"])

    def test_a_crash_at_every_step_before_the_commit_marker_leaves_neither_file(self):
        for crash_at in (1, 2, 3, 4):                              # journal, log, memory file, commit marker
            with self.subTest(crash_at=crash_at):
                shutil.rmtree(self.memdir, ignore_errors=True)
                it = item(1)
                a = self.auth(it)
                self.run_crash(crash_at, it, a)
                if crash_at > 1:
                    self.assertIn(".txn.json", self.files())       # the torn state is really on disk
                fresh = S.MemoryStore(self.root, clock=self.clock)
                self.assertEqual(fresh.verify(), ())
                self.assertEqual(self.files(), {})                 # fully rolled back: no file, no nonce, no journal
                self.assertFalse(self.consistent(mid(1), a.nonce))

    def test_a_crash_after_the_commit_marker_completes_both_files(self):
        it = item(1)
        a = self.auth(it)
        real_unlink = S.os.unlink

        def dying_unlink(path, *args, **kw):
            if str(path).endswith(".txn.json"):
                raise Crash()
            return real_unlink(path, *args, **kw)
        with mock.patch.object(S.os, "unlink", dying_unlink):
            with self.assertRaises(Crash):
                self.store.put(it, EV, authorization=a)
        self.assertIn(".txn.json", self.files())
        fresh = S.MemoryStore(self.root, clock=self.clock)
        self.assertEqual(fresh.verify(), ())
        self.assertTrue(self.consistent(mid(1), a.nonce))
        self.assertNotIn(".txn.json", self.files())
        e = self.untouched(fresh.put, it, EV, authorization=a)                       # and it is spent
        self.assertEqual(e.reason, "duplicate_id")

    def test_after_a_proven_full_rollback_the_same_authorization_works_once(self):
        it = item(1)
        a = self.auth(it)
        for crash_at in (1, 2, 3, 4):
            with self.subTest(crash_at=crash_at):
                shutil.rmtree(self.memdir, ignore_errors=True)
                self.run_crash(crash_at, it, a)
                S.MemoryStore(self.root, clock=self.clock).verify()           # recovery runs
                self.assertEqual(self.files(), {})                            # proven: byte-identical to "nothing"
                self.assertEqual(S.MemoryStore(self.root, clock=self.clock).put(it, EV, authorization=a), mid(1))
                self.assertTrue(self.consistent(mid(1), a.nonce))
                self.untouched(self.store.put, it, EV, authorization=a)

    def test_a_partial_install_failure_rolls_back_both_files_and_the_authorization_survives(self):
        it = item(1)
        a = self.auth(it)
        real, count = S._write_atomic, [0]

        def flaky(path, text):
            count[0] += 1
            if count[0] == 3:                                                 # journal ok, log ok, memory file fails
                raise OSError("disk")
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", flaky):
            self.refused("write_failed", self.store.put, it, EV, authorization=a)
        self.assertEqual(self.files(), {})                                    # log was written first: it was rolled back too
        self.assertEqual(self.store.put(it, EV, authorization=a), mid(1))
        self.assertTrue(self.consistent(mid(1), a.nonce))

    def test_when_the_rollback_itself_fails_the_authorization_is_not_reusable_until_recovery_succeeds(self):
        it = item(1)
        a = self.auth(it)
        real, count = S._write_atomic, [0]

        def flaky(path, text):
            count[0] += 1
            if count[0] == 3:
                raise OSError("disk")
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", flaky), \
                mock.patch.object(S.MemoryStore, "_rollback", side_effect=S.StoreError("recovery_failed")):
            self.refused("write_failed_recovery_pending", self.store.put, it, EV, authorization=a)
        torn = self.files()
        self.assertIn(".txn.json", torn)
        with mock.patch.object(S.MemoryStore, "_rollback", side_effect=S.StoreError("recovery_failed")):
            self.refused("recovery_failed", self.store.put, it, EV, authorization=a)       # cannot prove a rollback: no reuse
        self.assertEqual(self.files(), torn)
        self.assertEqual(S.MemoryStore(self.root, clock=self.clock).put(it, EV, authorization=a), mid(1))   # recovery, then reuse
        self.assertTrue(self.consistent(mid(1), a.nonce))

    def test_a_supersession_and_its_nonce_are_one_transaction(self):
        prop, d = self.supersede_setup()
        a = self.auth(prop.item, op="apply", links=d.supersedes)
        before = self.files()
        real, count = S._write_atomic, [0]

        def dying(path, text):
            count[0] += 1
            if count[0] == 4:                                   # journal, log, then the first memory file lands, then dies
                raise Crash()
            return real(path, text)
        with mock.patch.object(S, "_write_atomic", dying), mock.patch.object(S.MemoryStore, "_rollback", side_effect=Crash()):
            with self.assertRaises(Crash):
                self.store.apply(d, prop, NOW, authorization=a)
        S.MemoryStore(self.root, clock=self.clock).verify()
        self.assertEqual(self.files(), before)
        self.assertEqual(self.store.apply(d, prop, NOW, authorization=a).superseded, (mid(2),))


class DefensiveBranchTests(AuthCase):
    def test_an_authorization_is_immutable(self):
        a = self.auth(item(1))
        with self.assertRaises(Exception):
            a.nonce = "0" * 32
        with self.assertRaises(Exception):
            del a.operation

    def test_the_system_clock_is_whole_seconds_utc(self):
        t = S._system_utc()
        self.assertEqual((t.microsecond, t.utcoffset().total_seconds()), (0, 0))

    def test_reading_the_state_digest_creates_nothing(self):
        self.store.state_digest()
        self.store.verify()
        self.assertFalse(os.path.exists(self.memdir))

    def test_an_unauthorized_put_that_would_supersede_is_sent_to_apply(self):
        arrange(self.store, item(2, value="Kansas City", valid_from=day(1), created=day(1), updated=day(1)),
                "I live in Kansas City")
        before = self.files()
        e = self.untouched(self.store.put, item(1), EV, now=NOW)
        self.assertEqual(e.reason, "supersedes_requires_apply")
        self.assertEqual(self.files(), before)

    def test_an_authorized_put_that_would_supersede_is_sent_to_apply(self):
        prop, d = self.supersede_setup()
        e = self.untouched(self.store.put, prop.item, EV, authorization=self.auth(prop.item, links=d.supersedes))
        self.assertEqual(e.reason, "supersedes_requires_apply")

    def test_the_shared_routine_itself_refuses_a_real_vault_write_with_no_authorization(self):
        with mock.patch.object(S, "REAL_VAULT", self.root):
            store = S.MemoryStore(self.root, allow_real_vault=True)
        prop = G.Proposal(item(1), EV)
        with store._locked(write=False):
            found = store._all()
        for op in ("put", "apply"):
            self.refused("authorization_required", store._decide_and_authorize, op, prop, NOW, found, None, None)

    def test_the_store_checks_the_gate_again_after_deriving_the_confirmation(self):
        it = item(1, sensitivity="private")
        a = self.auth(it)
        real = G.evaluate
        calls = []

        def second_call_rejects(*args, **kw):
            calls.append(1)
            return real(*args, **kw) if len(calls) == 1 else G.Decision(G.REJECT, "later_verdict")
        with mock.patch.object(G, "evaluate", second_call_rejects):
            e = self.untouched(self.store.put, it, EV, authorization=a)
        self.assertEqual((e.reason, e.details), ("decision_not_accept", (G.REJECT, "later_verdict")))
        self.assertEqual(len(calls), 2)

    def test_the_commit_step_writes_only_an_accept_decision(self):
        prop = G.Proposal(item(1), EV)
        os.makedirs(self.memdir)
        for d in (G.Decision(G.REJECT, "x"), G.Decision(G.REQUIRE_CONFIRMATION, "private_requires_confirmation"),
                  G.Decision(G.CONFLICT, "conflict")):
            self.refused("decision_not_accept", self.store._commit, d, prop, NOW, {}, True)
        self.assertEqual([n for n in self.files() if n != ".lock"], [])


class StructureTests(unittest.TestCase):
    PKG = os.path.dirname(os.path.abspath(S.__file__))

    @staticmethod
    def parse(path):
        with open(path, encoding="utf-8") as f:
            return ast.parse(f.read())

    def calls_in(self, tree, fname):
        found = set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == fname:
                        found.add(fn.name)
        return found

    def test_put_and_apply_both_and_only_they_call_the_shared_routine(self):
        self.assertEqual(self.calls_in(self.parse(S.__file__), "_decide_and_authorize"), {"put", "apply"})

    def test_the_shared_routine_runs_inside_the_write_lock_right_before_the_commit(self):
        tree = self.parse(S.__file__)
        for name in ("put", "apply"):
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            locked = next(n for n in ast.walk(fn) if isinstance(n, ast.With))
            self.assertIn("_locked", ast.unparse(locked.items[0].context_expr))
            body = [ast.unparse(st) for st in locked.body]
            call = next(i for i, t in enumerate(body) if "_decide_and_authorize" in t)
            commit = next(i for i, t in enumerate(body) if "_commit" in t)
            self.assertEqual(commit, call + 1, name)

    def test_the_only_wall_clock_read_is_in_system_utc(self):
        tree = self.parse(S.__file__)
        reads = []
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Attribute) and n.attr in ("now", "utcnow", "today", "time", "monotonic"):
                        reads.append((fn.name, n.attr))
        self.assertEqual(reads, [("_system_utc", "now")])

    def test_only_the_gate_and_the_store_mention_the_authorization_classes(self):
        users = set()
        for path in glob.glob(os.path.join(self.PKG, "**", "*.py"), recursive=True):
            names = {n.id for n in ast.walk(self.parse(path)) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(self.parse(path)) if isinstance(n, ast.Attribute)}
            if names & {"Authorization", "Confirmation"}:
                users.add(os.path.basename(path))
        self.assertEqual(users, {"memory_store.py", "memory_gate.py", "memory_execute.py"})   # execute builds the Authorization

    def test_the_store_builds_the_gate_confirmation_in_exactly_one_place(self):
        tree = self.parse(S.__file__)
        builders = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                    for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == "Confirmation"}
        self.assertEqual(builders, {"_decide_and_authorize"})

    def test_no_production_module_imports_the_test_clock(self):
        for path in glob.glob(os.path.join(self.PKG, "**", "*.py"), recursive=True):
            for n in ast.walk(self.parse(path)):
                mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [(n.module or "")] + [(n.module or "") + "." + a.name for a in n.names] if isinstance(n, ast.ImportFrom) else []
                self.assertFalse(any("fake_clock" in m or m.split(".")[0] == "tests" for m in mods), path)

    def test_the_markers_are_now_real_signatures(self):
        self.assertIn("authorization", inspect.signature(S.MemoryStore.apply).parameters)
        self.assertIn("authorization", inspect.signature(S.MemoryStore.put).parameters)
        self.assertIn("confirmation", inspect.signature(G.evaluate).parameters)
        self.assertIsNone(inspect.signature(G.evaluate).parameters["confirmation"].default)


if __name__ == "__main__":
    unittest.main()
