"""Memory 2.0, Tier 2 integration (D1-D7): the second confirmation wired into `memory_execute.commit` through an injected
gate, the evidence log, the pilot approval file and the send limits. Everything runs on throwaway stores and temp
directories with fake Telegram responses and a fake clock. `subprocess.run` and `urllib.request.urlopen` are replaced by
functions that fail the test if called, so no test can reach the Keychain or the network.

WHAT THESE TESTS DO NOT SHOW. The confirmation is a possession factor checked by this code, not identity authentication,
and the evidence chain detects damage, not a writer who recomputes it. Same-account code that bypasses the committer or
edits the vault, the evidence log or the approval file is outside what is tested. Nothing here enables the real vault:
the store under test is a temp store whose `require_authorization` flag is switched on to exercise the real-vault path."""
import ast
import inspect
import json
import os
import unittest
import re
import subprocess
import tempfile
import urllib.request
from types import SimpleNamespace
from unittest import mock

from backtalk import memory_evidence as EV
from backtalk import memory_receipts as RC
from backtalk import memory_tier2 as T
from tests.test_memory_execute import ExecCase
from tests.receipts_harness import item, tree_state

USER = CHAT = 4242
SENT = 1_800_000_000


class ReplyingTransport:
    """A fake Telegram: records what was sent and queues a scripted reply built from the code in the message."""

    def __init__(self, reply="good", webhook="", baseline=10, sender=USER, chat=CHAT, text=None, date_offset=5):
        self.reply, self.webhook, self.baseline = reply, webhook, baseline
        self.sender, self.chat, self.text, self.date_offset = sender, chat, text, date_offset
        self.sent, self.queue, self.calls = [], [], []

    def webhook_url(self):
        self.calls.append("webhook")
        return self.webhook

    def latest_update_id(self):
        self.calls.append("latest")
        return self.baseline

    def send(self, text):
        self.calls.append("send")
        self.sent.append(text)
        code = re.search(r"confirm ([0-9a-f]{12})", text).group(1)
        body = self.text if self.text is not None else f"confirm {code}"
        uid = self.baseline + len(self.sent)
        upd = {"update_id": uid, "message": {"message_id": uid, "from": {"id": self.sender},
                                             "chat": {"id": self.chat, "type": "private"},
                                             "date": SENT + self.date_offset, "text": body}}
        if self.reply == "good":
            self.queue = [[upd]]
        elif self.reply == "replay":
            upd["update_id"] = self.baseline               # already pending before the message: never accepted
            self.queue = [[upd]]
        elif self.reply == "duplicate":
            self.queue = [[upd, dict(upd)]]
        else:
            self.queue = []
        return {"message_id": uid, "date": SENT}

    def updates(self, after_id, timeout_s):
        self.calls.append("updates")
        return self.queue.pop(0) if self.queue else []


class Mono:
    def __init__(self, step=1.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class FakeConfirmer:
    """A confirmer that records its calls (used where the phone side is not the subject)."""

    def __init__(self, ok=True, code="confirmed", webhook_ok=True):
        self.ok, self.code, self.webhook_ok = ok, code, webhook_ok
        self.calls = []

    def preflight(self):
        self.calls.append("preflight")
        return SimpleNamespace(ok=self.webhook_ok, code="ok" if self.webhook_ok else "webhook_configured")

    def confirm(self, plan, nonce, lines):
        self.calls.append(("confirm", plan.candidate, nonce))
        return SimpleNamespace(ok=self.ok, code=self.code, update_id=77 if self.ok else None,
                               reply_date=SENT + 5 if self.ok else None, sent_date=SENT)


class TierCase(ExecCase):
    def setUp(self):
        super().setUp()

        def no_run(*a, **k):
            raise AssertionError("the Keychain command was run")

        def no_net(*a, **k):
            raise AssertionError("the network was contacted")
        for target, name, fn in ((subprocess, "run", no_run), (urllib.request, "urlopen", no_net)):
            p = mock.patch.object(target, name, fn)
            p.start()
            self.addCleanup(p.stop)
        self.store.require_authorization = True        # a TEMP store switched to the real-vault rule; never the real one

    # -- helpers --
    def approval(self, **over):
        if not hasattr(self, "_ap"):
            plan = self.make_plan()
            self.plan_calls = 0
            self._ap = (plan.candidate, plan.lock_hash)
        data = {"candidate": self._ap[0], "lock_hash": self._ap[1], "approved_at": self.clock.stamp()}
        data.update(over)
        path = os.path.join(self.rdir, EV.APPROVAL_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.chmod(path, 0o600)
        return path, data

    def gate(self, confirmer=None, **kw):
        def records():
            log = self.log()
            try:
                return log.records() if os.path.exists(log.path) else []
            except RC.ReceiptError:
                raise EV.EvidenceError("receipts_unreadable") from None
        return EV.Tier2Gate(confirmer if confirmer is not None else FakeConfirmer(), self.rdir, self.clock.stamp,
                            records, **kw)

    def real_confirmer(self, transport):
        return T.Confirmer(transport, USER, CHAT, Mono())

    def commit_with(self, gate, **over):
        return self.run_commit(tier2=gate, **over)

    def commit_files(self):
        return [n for n in os.listdir(self.rdir) if n.startswith("commits")]

    def evidence(self):
        return EV.EvidenceLog(self.rdir).records()


class RequirementTests(TierCase):
    def test_a_store_that_requires_authorization_refuses_without_the_second_confirmation(self):
        before = tree_state(self.root)
        out = self.run_commit()
        self.assertEqual((out.status, out.code), ("refused", "tier2_required"))
        self.assertEqual(tree_state(self.root), before)
        self.assertEqual(self.plan_calls, 0)

    def test_a_temp_store_commit_never_contacts_telegram_or_the_keychain(self):
        self.store.require_authorization = False        # the patched subprocess/urlopen would fail the test otherwise
        out = self.run_commit()
        self.assertEqual((out.status, out.code), ("committed", "committed"))
        self.assertNotIn(EV.EVIDENCE_FILE, os.listdir(self.rdir))


class HappyPathTests(TierCase):
    def test_a_confirmed_commit_records_its_evidence_before_the_intent_and_binds_everything(self):
        self.approval()
        transport = ReplyingTransport()
        order = []
        real_append, real_intent = EV.EvidenceLog.append, RC.ReceiptLog.append_intent

        def spy_append(this, event, fields, ts):
            order.append(event)
            return real_append(this, event, fields, ts)

        def spy_intent(this, *a, **k):
            order.append("intent")
            return real_intent(this, *a, **k)
        with mock.patch.object(EV.EvidenceLog, "append", spy_append), \
                mock.patch.object(RC.ReceiptLog, "append_intent", spy_intent):
            out = self.commit_with(self.gate(self.real_confirmer(transport)))
        self.assertEqual((out.status, out.code), ("committed", "committed"))
        self.assertEqual(order, ["tier2_sent", "tier2_confirmed", "intent"])
        self.assertEqual(self.actions(), ["open", "intent", "done"])
        sent, conf = self.evidence()
        intent = self.log().records()[1]
        for rec in (sent, conf):
            self.assertEqual(rec["candidate"], intent["candidate"])
            self.assertEqual(rec["lock_hash"], intent["lock_hash"])
            self.assertEqual(rec["nonce_hash"], intent["nonce_hash"])
            self.assertEqual(rec["item_digest"], intent["item_digest"])
            self.assertEqual(rec["context_digest"], intent["context_digest"])
            for k in ("operation", "state_digest", "display_digest"):
                self.assertTrue(rec[k])
        self.assertEqual(conf["update_id"], transport.baseline + 1)
        self.assertEqual((conf["sent_date"], conf["reply_date"]), (SENT, SENT + 5))
        self.assertEqual(len(transport.sent), 1)

    def test_confirmed_at_is_the_confirmation_instant_and_the_receipt_log_stays_valid(self):
        self.approval()
        t0 = self.clock.stamp()

        def ask_slowly(prompt):
            self.clock.advance(60)                      # a slow typist
            return self.typed.pop(0)
        gate = self.gate(FakeConfirmer())
        real_confirm = gate.confirm

        def slow_confirm(*a):
            self.clock.advance(100)                     # a slow phone reply: longer than the typed-to-store window alone
            return real_confirm(*a)
        gate.confirm = slow_confirm
        out = self.commit_with(gate, ask=ask_slowly)
        self.assertEqual((out.status, out.code), ("committed", "committed"))
        intent = self.log().records()[1]
        self.assertGreater(intent["confirmed_at"], t0)
        self.assertTrue(self.assess().proceed)


class RefusalLeavesNothingTests(TierCase):
    def assert_nothing_changed(self, out, code):
        self.assertEqual((out.status, out.code), ("refused", code))
        self.assertEqual(self.commit_files(), [])
        self.assertEqual(self.memory_ids(), [])

    def test_a_declined_or_silent_phone_changes_no_receipt_and_no_store(self):
        self.approval()
        t = ReplyingTransport(reply="silence")
        self.assert_nothing_changed(self.commit_with(self.gate(self.real_confirmer(t))), "tier2:timeout")
        self.assertEqual([r["event"] for r in self.evidence()], ["tier2_sent"])

    def test_invalid_wrong_sender_wrong_chat_expired_replayed_and_duplicated_replies_fail_closed(self):
        self.approval()
        cases = [("wrong_text", dict(text="confirm 000000000000"), "tier2:invalid_response"),
                 ("wrong_sender", dict(sender=999), "tier2:wrong_sender"),
                 ("wrong_chat", dict(chat=555), "tier2:wrong_chat"),
                 ("expired", dict(date_offset=T.WINDOW_S + 1), "tier2:expired"),
                 ("replayed", dict(reply="replay"), "tier2:timeout")]
        for label, kw, code in cases:
            with self.subTest(label):
                self.clock.advance(EV.MIN_SPACING_S + 1)       # stay clear of the send-spacing limit
                self.approval()                                 # a fresh approval resets the count
                out = self.commit_with(self.gate(self.real_confirmer(ReplyingTransport(**kw))))
                self.assert_nothing_changed(out, code)
        with self.subTest("duplicated update is one confirmation"):
            self.clock.advance(EV.MIN_SPACING_S + 1)
            self.approval()
            out = self.commit_with(self.gate(self.real_confirmer(ReplyingTransport(reply="duplicate"))))
            self.assertEqual((out.status, out.code), ("committed", "committed"))
            self.assertEqual(len(self.log().records()), 3)

    def test_a_plan_changed_after_the_phone_confirmed_refuses_before_any_receipt(self):
        self.approval()
        gate = self.gate(FakeConfirmer())
        real_confirm = gate.confirm

        def confirm_then_change(*a):
            res = real_confirm(*a)
            self.store.require_authorization = False
            self.store.put(item(5, attribute="hobby", value="chess"), "I like chess")
            self.store.require_authorization = True
            return res
        gate.confirm = confirm_then_change
        out = self.commit_with(gate)
        self.assertEqual((out.status, out.code), ("refused", "plan_changed"))
        self.assertEqual(self.commit_files(), [])
        self.assertEqual([r["event"] for r in self.evidence()], ["tier2_sent", "tier2_confirmed"])

    def test_a_configured_webhook_refuses_with_its_code_before_a_plan_is_built(self):
        self.approval()
        t = ReplyingTransport(webhook="https://example.invalid/hook")
        out = self.commit_with(self.gate(self.real_confirmer(t)))
        self.assert_nothing_changed(out, "tier2:webhook_configured")
        self.assertEqual((self.plan_calls, self.said, t.sent), (0, [], []))

    def test_an_unconfigured_confirmer_refuses_before_a_plan_is_built(self):
        self.approval()
        gate = EV.Tier2Gate(None, self.rdir, self.clock.stamp, lambda: [])
        out = self.commit_with(gate)
        self.assert_nothing_changed(out, "tier2:not_configured")
        self.assertEqual(self.plan_calls, 0)


class EvidenceTests(TierCase):
    def test_evidence_failure_before_the_send_sends_nothing_and_writes_no_intent(self):
        self.approval()
        t = ReplyingTransport()
        with mock.patch.object(EV.EvidenceLog, "append", side_effect=EV.EvidenceError("evidence_unwritable")):
            out = self.commit_with(self.gate(self.real_confirmer(t)))
        self.assertEqual((out.status, out.code), ("refused", "tier2:evidence_unwritable"))
        self.assertEqual((t.sent, self.commit_files()), ([], []))

    def test_evidence_failure_after_the_confirmation_refuses_before_any_intent(self):
        self.approval()
        real = EV.EvidenceLog.append

        def fail_second(this, event, fields, ts):
            if event == "tier2_confirmed":
                raise EV.EvidenceError("evidence_unwritable")
            return real(this, event, fields, ts)
        calls = self.spy_store()
        with mock.patch.object(EV.EvidenceLog, "append", fail_second):
            out = self.commit_with(self.gate(FakeConfirmer()))
        self.assertEqual((out.status, out.code), ("refused", "tier2:evidence_failed:evidence_unwritable"))
        self.assertEqual((self.commit_files(), calls, self.memory_ids()), ([], [], []))

    def test_a_tampered_chain_is_detected_and_refuses_the_commit(self):
        self.approval()
        EV.EvidenceLog(self.rdir).append("tier2_sent", {"candidate": "c", "lock_hash": "l"}, self.clock.stamp())
        path = os.path.join(self.rdir, EV.EVIDENCE_FILE)
        with open(path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        rec["candidate"] = "someone-else"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        with self.assertRaises(EV.EvidenceError) as ctx:
            EV.EvidenceLog(self.rdir).records()
        self.assertEqual(ctx.exception.reason, "evidence_corrupt")
        out = self.commit_with(self.gate(FakeConfirmer()))
        self.assertEqual((out.status, out.code), ("refused", "tier2:evidence_corrupt"))

    def test_a_removed_middle_record_breaks_the_chain(self):
        log = EV.EvidenceLog(self.rdir)
        for i in range(3):
            log.append("tier2_sent", {"candidate": "c", "lock_hash": "l"}, self.clock.stamp())
        path = log.path
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(path, "w", encoding="utf-8") as f:
            f.writelines([lines[0], lines[2]])
        with self.assertRaises(EV.EvidenceError):
            log.records()


class ApprovalTests(TierCase):
    def check(self, **over):
        path, data = self.approval(**over)
        gate = self.gate(FakeConfirmer())
        self.assertIsNone(gate.prepare())
        return gate, gate.check_plan(SimpleNamespace(kind="commit", candidate=data["candidate"],
                                                     lock_hash=data["lock_hash"]))

    def test_a_valid_approval_passes(self):
        self.assertEqual(self.check()[1], None)

    def test_a_mismatched_lock_hash_or_candidate_is_rejected(self):
        gate, _ = self.check()
        plan = SimpleNamespace(kind="commit", candidate=gate._approval["candidate"], lock_hash="f" * 64)
        self.assertEqual(gate.check_plan(plan), "approval_lock_mismatch")
        plan.lock_hash, plan.candidate = gate._approval["lock_hash"], "other"
        self.assertEqual(gate.check_plan(plan), "approval_candidate_mismatch")

    def test_an_approval_older_than_24_hours_is_rejected_and_exactly_24_hours_is_not(self):
        gate, _ = self.check()
        plan = SimpleNamespace(kind="commit", candidate=gate._approval["candidate"],
                               lock_hash=gate._approval["lock_hash"])
        self.clock.advance(EV.APPROVAL_MAX_AGE_S)
        self.assertIsNone(gate.check_plan(plan))
        self.clock.advance(1)
        self.assertEqual(gate.check_plan(plan), "approval_expired")

    def test_a_missing_loose_or_malformed_approval_is_refused_and_never_repaired(self):
        gate = self.gate()
        self.assertEqual(gate.prepare(), "approval_missing")
        self.assertNotIn(EV.APPROVAL_FILE, os.listdir(self.rdir))                # the gate never creates it
        path, _ = self.approval()
        os.chmod(path, 0o644)
        self.assertEqual(self.gate().prepare(), "approval_permissions")
        self.assertEqual(stat_mode(path), 0o644)                                 # and never repairs it
        self.approval(extra="x")
        self.assertEqual(self.gate().prepare(), "approval_shape")
        os.unlink(os.path.join(self.rdir, EV.APPROVAL_FILE))
        os.symlink(os.path.join(self.rdir, "elsewhere"), os.path.join(self.rdir, EV.APPROVAL_FILE))
        self.assertEqual(self.gate().prepare(), "approval_not_a_file")

    def test_a_private_kind_is_not_allowed_in_the_pilot(self):
        gate, _ = self.check()
        plan = SimpleNamespace(kind="commit-private", candidate=gate._approval["candidate"],
                               lock_hash=gate._approval["lock_hash"])
        self.assertEqual(gate.check_plan(plan), "kind_not_allowed")

    def test_done_consumes_the_approval_and_a_new_approval_is_needed_for_the_next_commit(self):
        self.approval()
        out = self.commit_with(self.gate(FakeConfirmer()))
        self.assertEqual(out.status, "committed")
        gate = self.gate(FakeConfirmer())
        self.assertIsNone(gate.prepare())
        data = gate._approval
        same = SimpleNamespace(kind="commit", candidate=data["candidate"], lock_hash=data["lock_hash"])
        self.assertEqual(gate.check_plan(same), "approval_consumed")
        self.clock.advance(EV.MIN_SPACING_S + 1)
        self.approval(candidate="another-candidate")                              # a new approval, a different candidate
        gate = self.gate(FakeConfirmer())
        self.assertIsNone(gate.prepare())
        other = SimpleNamespace(kind="commit", candidate="another-candidate", lock_hash=gate._approval["lock_hash"])
        self.assertIsNone(gate.check_plan(other))                  # since 2026-10-19 the cap is 10: a new approval is honoured
        real = EV.PILOT_MAX_COMMITS
        EV.PILOT_MAX_COMMITS = 1                                   # ...and the cap itself is still enforced at its limit
        try:
            self.assertEqual(gate.check_plan(other), "pilot_limit_reached")
        finally:
            EV.PILOT_MAX_COMMITS = real

    def test_an_aborted_commit_does_not_consume_the_approval(self):
        self.approval()
        out = self.commit_with(self.gate(FakeConfirmer()), typed=["commit 00000000"])
        self.assertEqual(out.status, "declined")
        gate = self.gate(FakeConfirmer())
        self.assertIsNone(gate.prepare())
        self.assertIsNone(gate.check_plan(SimpleNamespace(kind="commit", candidate=gate._approval["candidate"],
                                                          lock_hash=gate._approval["lock_hash"])))


def stat_mode(path):
    return os.stat(path).st_mode & 0o777


class SendLimitTests(TierCase):
    def plan(self, gate):
        return SimpleNamespace(kind="commit", candidate=gate._approval["candidate"],
                               lock_hash=gate._approval["lock_hash"], operation="put", item_digest="1" * 64,
                               state_digest="2" * 64, context_digest="3" * 64, display_digest="4" * 64)

    def test_constants_are_explicit(self):
        self.assertEqual((EV.MAX_SENDS, EV.MIN_SPACING_S, EV.APPROVAL_MAX_AGE_S, EV.PILOT_MAX_COMMITS),
                         (3, 30, 86400, 10))

    def test_three_sends_then_blocked_and_a_new_approval_resets(self):
        self.approval()
        conf = FakeConfirmer(ok=False, code="timeout")
        gate = self.gate(conf)
        self.assertIsNone(gate.prepare())
        plan = self.plan(gate)
        for i in range(EV.MAX_SENDS):
            self.assertEqual(gate.confirm(plan, f"{i + 1:032x}", ("x",)).code, "timeout")
            self.clock.advance(EV.MIN_SPACING_S)
        self.assertEqual(gate.check_plan(plan), "send_limit_reached")
        res = gate.confirm(plan, "f" * 32, ("x",))
        self.assertEqual((res.ok, res.code), (False, "send_limit_reached"))
        self.assertEqual(len([c for c in conf.calls if c != "preflight"]), EV.MAX_SENDS)
        self.approval()                                                          # a new user-created approval
        gate = self.gate(conf)
        self.assertIsNone(gate.prepare())
        self.assertIsNone(gate.check_plan(self.plan(gate)))

    def test_less_than_30_seconds_between_sends_is_blocked_and_exactly_30_is_allowed(self):
        self.approval()
        conf = FakeConfirmer(ok=False, code="timeout")
        gate = self.gate(conf)
        self.assertIsNone(gate.prepare())
        plan = self.plan(gate)
        gate.confirm(plan, "1" * 32, ("x",))
        self.clock.advance(EV.MIN_SPACING_S - 1)
        self.assertEqual(gate.check_plan(plan), "send_too_soon")
        self.assertEqual(gate.confirm(plan, "2" * 32, ("x",)).code, "send_too_soon")
        self.clock.advance(1)
        self.assertIsNone(gate.check_plan(plan))
        self.assertEqual(gate.confirm(plan, "3" * 32, ("x",)).code, "timeout")

    def test_the_count_is_per_candidate_and_lock_hash(self):
        self.approval()
        gate = self.gate(FakeConfirmer(ok=False, code="timeout"))
        gate.prepare()
        plan = self.plan(gate)
        gate.confirm(plan, "1" * 32, ("x",))
        other = SimpleNamespace(**dict(vars(plan), lock_hash="e" * 64))
        self.clock.advance(EV.MIN_SPACING_S)
        self.assertEqual(len(gate.log.sent_since(plan.candidate, plan.lock_hash, gate._approval["approved_at"])), 1)
        self.assertEqual(gate.log.sent_since(other.candidate, other.lock_hash, gate._approval["approved_at"]), [])


class ReconcileAndReadOnlyTests(TierCase):
    def test_reconcile_never_calls_the_confirmer_and_decides_from_store_proof(self):
        self.approval()
        conf = FakeConfirmer()
        real_install = self.store.put

        def crash(*a, **k):
            real_install(*a, **k)
            raise KeyboardInterrupt
        self.store.put = crash
        with self.assertRaises(KeyboardInterrupt):
            self.commit_with(self.gate(conf))
        self.store.put = real_install
        before = list(conf.calls)
        out = self.run_reconcile(["reconcile"])
        self.assertEqual(out.status, "reconciled")
        self.assertEqual(self.actions()[-1], "done")                              # from the store's own proof
        self.assertEqual(conf.calls, before)
        self.assertEqual(len(self.memory_ids()), 1)

    def test_reconcile_with_nothing_to_do_does_not_create_a_write(self):
        conf = FakeConfirmer()
        out = self.run_reconcile([])
        self.assertEqual(out.status, "refused")
        self.assertEqual((conf.calls, self.memory_ids(), self.commit_files()), ([], [], []))

    def test_the_read_only_cli_modes_never_touch_the_network_or_the_keychain(self):
        from backtalk.evals.memory_extraction import commit_cli as CLI
        out = []
        with tempfile.TemporaryDirectory() as other:
            root = os.path.join(os.path.realpath(other), "review")
            os.mkdir(root, 0o700)
            for flag in ("--status", "--audit"):
                CLI.main([flag, "--vault", self.vault, "--root", root], say=out.append, store_clock=self.clock)
            self.assertEqual(os.listdir(root), [])
        self.assertTrue(out)


class EvidenceModuleStructureTests(unittest.TestCase):
    SRC = inspect.getsource(EV)

    def test_it_imports_no_network_subprocess_or_terminal_layer(self):
        mods = set()
        for n in ast.walk(ast.parse(self.SRC)):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"errno", "fcntl", "hashlib", "json", "os", "re", "stat",
                                "datetime.datetime", "datetime.timezone"})

    def test_the_closed_components_do_not_know_the_second_confirmation(self):
        for name in ("memory_store.py", "memory_receipts.py", "memory_commit.py"):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtalk", name)
            with open(path, encoding="utf-8") as f:
                self.assertNotIn("tier2", f.read().lower(), name)

    def test_the_gate_never_creates_or_repairs_the_approval(self):
        used = {n.attr for n in ast.walk(ast.parse(self.SRC)) if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == "os"}
        self.assertFalse(used & {"chmod", "unlink", "rename", "replace", "mkdir", "makedirs", "symlink"}, used)


class EvidenceRecoveryTests(TierCase):
    def seed(self, n=2):
        log = EV.EvidenceLog(self.rdir)
        for i in range(n):
            self.clock.advance(1)
            log.append("tier2_sent", {"candidate": "c", "lock_hash": "l", "nonce_hash": f"{i:032x}"}, self.clock.stamp())
        return log

    def bytes_of(self, log):
        with open(log.path, "rb") as f:
            return f.read()

    def put(self, log, raw):
        with open(log.path, "wb") as f:
            f.write(raw)
        os.chmod(log.path, 0o600)

    def test_a_clean_log_diagnoses_ok(self):
        d = self.seed(3).diagnose()
        self.assertEqual((d["ok"], d["records"], d["problem"], d["torn_tail"]), (True, 3, None, False))

    def test_a_truncated_final_line_is_fail_closed_diagnosed_and_never_rewritten(self):
        log = self.seed(3)
        raw = self.bytes_of(log)
        self.put(log, raw[:-17])                                     # a crash in the middle of the last append
        torn = self.bytes_of(log)
        d = log.diagnose()
        self.assertEqual((d["ok"], d["records"], d["bad_line"], d["torn_tail"]), (False, 2, 3, True))
        self.assertEqual(self.bytes_of(log), torn)                   # diagnosis wrote nothing
        with self.assertRaises(EV.EvidenceError):
            log.records()
        with self.assertRaises(EV.EvidenceError):
            log.append("tier2_sent", {"candidate": "c"}, self.clock.stamp())
        self.assertEqual(self.bytes_of(log), torn)                   # an append never repairs, truncates or extends it

    def test_a_malformed_complete_record_a_broken_link_and_a_forged_hash_are_diagnosed(self):
        log = self.seed(3)
        lines = self.bytes_of(log).split(b"\n")[:-1]
        cases = {"malformed": [lines[0], b"{not json", lines[2]], "link": [lines[0], lines[2]],
                 "forged": [lines[0], lines[1].replace(b'"candidate":"c"', b'"candidate":"x"'), lines[2]]}
        for label, ls in cases.items():
            with self.subTest(label):
                self.put(log, b"\n".join(ls) + b"\n")
                d = log.diagnose()
                self.assertFalse(d["ok"])
                self.assertEqual(d["records"], 1)
                self.assertEqual(d["bad_line"], 2)
                with self.assertRaises(EV.EvidenceError):
                    log.records()

    def test_a_corrupt_log_refuses_the_gate_and_the_confirmer_is_never_called(self):
        self.approval()
        log = self.seed(2)
        self.put(log, self.bytes_of(log)[:-5])
        conf = FakeConfirmer()
        out = self.commit_with(self.gate(conf))
        self.assertEqual((out.status, out.code), ("refused", "tier2:evidence_corrupt"))
        self.assertEqual((self.commit_files(), self.memory_ids(), self.plan_calls), ([], [], 0))
        self.assertNotIn(("confirm",), [c[:1] for c in conf.calls if isinstance(c, tuple)])

    def test_send_counts_survive_a_valid_log_and_a_sent_record_is_never_read_as_a_confirmation(self):
        self.approval()
        log = self.seed(2)
        gate = self.gate(FakeConfirmer(ok=False, code="timeout"))
        self.assertIsNone(gate.prepare())
        self.assertEqual(len(log.records()), 2)
        self.assertTrue(all(r["event"] == "tier2_sent" for r in log.records()))
        out = self.commit_with(self.gate(FakeConfirmer(ok=False, code="timeout")))
        self.assertEqual(out.status, "refused")                      # a sent record alone authorizes nothing
        self.assertEqual(self.commit_files(), [])

    def test_setting_the_log_aside_requires_a_new_approval_made_afterwards(self):
        self.approval()
        log = self.seed(2)
        os.rename(log.path, os.path.join(self.rdir, EV.QUARANTINE_PREFIX + ".1"))
        self.assertEqual(self.gate().prepare(), "approval_predates_quarantine")
        self.assertEqual(EV.EvidenceLog(self.rdir).records(), [])    # nothing was recreated or repaired by the gate
        path = os.path.join(self.rdir, EV.APPROVAL_FILE)
        q = os.path.join(self.rdir, EV.QUARANTINE_PREFIX + ".1")
        os.utime(q, (1_000_000, 1_000_000))
        os.utime(path, (2_000_000, 2_000_000))                       # a person-made approval, later than the quarantine
        self.assertIsNone(self.gate().prepare())

    def test_an_interrupted_append_leaves_no_partial_record_that_validates(self):
        log = self.seed(1)
        raw = self.bytes_of(log)
        with mock.patch("os.fsync", side_effect=OSError(5, "io")):
            with self.assertRaises(EV.EvidenceError):
                log.append("tier2_sent", {"candidate": "c"}, self.clock.stamp())
        after = self.bytes_of(log)
        self.assertTrue(after.startswith(raw))
        d = log.diagnose()
        self.assertIn(d["records"], (1, 2))                          # complete or absent, never half-accepted


class ConfigRefusalTests(TierCase):
    def test_an_invalid_keychain_payload_is_not_configured_and_never_contacts_telegram(self):
        for out in ("", "not json", json.dumps({"token": "x", "chat_id": 1, "user_id": 1}),
                    json.dumps({"token": "123456789:" + "A" * 30, "chat_id": True, "user_id": 1})):
            run = lambda *a, **k: SimpleNamespace(returncode=0, stdout=out)
            self.assertIsNone(T.configured_confirmer(Mono(), run=run, opener=lambda *a, **k: 1 / 0))
        run = lambda *a, **k: SimpleNamespace(returncode=44, stdout="")
        self.assertIsNone(T.configured_confirmer(Mono(), run=run))
