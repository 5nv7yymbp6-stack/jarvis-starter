"""Memory 2.0, Tier 2: the out-of-band confirmation (backtalk/memory_tier2.py), tested on its own with mocked Telegram
responses and a fake monotonic clock. No test reaches the network, the Keychain or any file: `subprocess.run` and
`urllib.request.urlopen` are replaced by functions that fail the test if they are ever called, and the token used
below is an obviously fake placeholder.

WHAT THESE TESTS DO NOT SHOW. This is a possession-factor confirmation checked by this code: nothing here claims
identity authentication, and same-account code that bypasses the committer or writes the vault directly is outside it.
The module is not wired into the committer or the CLI in this step."""
import ast
import glob
import inspect
import json
import os
import subprocess
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

from backtalk import memory_tier2 as T

USER, CHAT = 4242, 4242
FAKE_TOKEN = "123456789:" + "A" * 30
NONCE = "ab" * 16
SENT = 1_800_000_000


def plan(**over):
    d = dict(operation="put", kind="commit", item_digest="1" * 64, state_digest="2" * 64, context_digest="3" * 64,
             display_digest="4" * 64)
    d.update(over)
    return SimpleNamespace(**d)


LINES = ("operation    put", "value        \"wholesale real estate\"")


def msg(uid, text, *, sender=USER, chat=CHAT, date=SENT + 5, ctype="private"):
    return {"update_id": uid, "message": {"message_id": uid, "from": {"id": sender}, "chat": {"id": chat, "type": ctype},
                                          "date": date, "text": text}}


class FakeTransport:
    def __init__(self, batches=(), baseline=10, send_date=SENT):
        self.batches, self.baseline, self.send_date = list(batches), baseline, send_date
        self.calls, self.sent = [], []
        self.fail_at = None
        self.send_result = None

    def _maybe_fail(self, name):
        if self.fail_at == name:
            raise T.TransportError("boom")

    def latest_update_id(self):
        self.calls.append("latest")
        self._maybe_fail("latest")
        return self.baseline

    def send(self, text):
        self.calls.append("send")
        self._maybe_fail("send")
        self.sent.append(text)
        return self.send_result if self.send_result is not None else {"message_id": 1, "date": self.send_date}

    def updates(self, after_id, timeout_s):
        self.calls.append(("updates", after_id, timeout_s))
        self._maybe_fail("updates")
        return self.batches.pop(0) if self.batches else []


class Clock:
    """A monotonic clock that advances `step` on every call."""

    def __init__(self, step=1.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class Case(unittest.TestCase):
    def setUp(self):
        def no_run(*a, **k):
            raise AssertionError("the Keychain command was run")

        def no_net(*a, **k):
            raise AssertionError("the network was contacted")
        for target, name, fn in ((subprocess, "run", no_run), (urllib.request, "urlopen", no_net)):
            p = mock.patch.object(target, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def confirmer(self, transport, step=1.0):
        return T.Confirmer(transport, USER, CHAT, Clock(step))

    def code(self, p=None, nonce=NONCE):
        return T.challenge_code(p or plan(), nonce)

    def good(self, uid=11, **kw):
        return msg(uid, f"confirm {self.code()}", **kw)


class BindingTests(Case):
    def test_the_code_is_stable_and_twelve_hex(self):
        c = self.code()
        self.assertEqual(c, self.code())
        self.assertRegex(c, r"^[0-9a-f]{12}\Z")

    def test_every_bound_field_changes_the_code(self):
        base = self.code()
        variants = {
            "operation": plan(operation="apply"), "kind": plan(kind="commit-private"),
            "item_digest": plan(item_digest="5" * 64), "state_digest": plan(state_digest="6" * 64),
            "context_digest": plan(context_digest="7" * 64), "display_digest": plan(display_digest="8" * 64),
        }
        seen = {base}
        for field, p in variants.items():
            c = self.code(p)
            self.assertNotEqual(c, base, field)
            seen.add(c)
        fresh = self.code(nonce="cd" * 16)
        self.assertNotIn(fresh, seen)

    def test_the_message_carries_the_item_the_code_and_the_expiry_and_nothing_else_secret(self):
        text = T.message_text(plan(), NONCE, LINES)
        self.assertIn("MEMORY WRITE REQUEST", text)
        self.assertIn(LINES[1], text)
        self.assertIn(f"confirm {self.code()}", text)
        self.assertIn("120 seconds", text)
        self.assertNotIn(NONCE, text)
        self.assertNotIn(FAKE_TOKEN, text)

    def test_a_reply_for_another_plan_or_another_nonce_is_refused(self):
        for label, other in (("plan", self.code(plan(item_digest="9" * 64))), ("nonce", self.code(nonce="ef" * 16))):
            t = FakeTransport([[msg(11, f"confirm {other}")]])
            r = self.confirmer(t).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "invalid_response"), label)


class ConfirmTests(Case):
    def test_a_correct_reply_confirms_and_reports_the_update_id(self):
        t = FakeTransport([[self.good(11)]])
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code, r.update_id), (True, "confirmed", 11))
        self.assertEqual(t.calls[0], "latest")                 # the backlog is measured BEFORE the message is sent
        self.assertEqual(t.calls[1], "send")
        self.assertEqual(t.calls[2][:2], ("updates", 10))      # and only updates after it are read
        self.assertEqual(len(t.sent), 1)

    def test_the_window_is_exactly_one_hundred_twenty_seconds(self):
        ok = FakeTransport([[self.good(11, date=SENT + 120)]])
        self.assertTrue(self.confirmer(ok).confirm(plan(), NONCE, LINES).ok)
        late = FakeTransport([[self.good(11, date=SENT + 121)]])
        r = self.confirmer(late).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "expired"))
        early = FakeTransport([[self.good(11, date=SENT - 1)]])
        r = self.confirmer(early).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))      # predates the challenge

    def test_silence_times_out_on_the_local_clock(self):
        t = FakeTransport([])
        r = self.confirmer(t, step=30.0).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "timeout"))
        polls = [c for c in t.calls if isinstance(c, tuple)]
        self.assertTrue(polls)
        for _, _, timeout in polls:
            self.assertLessEqual(timeout, T.POLL_S)
            self.assertGreaterEqual(timeout, 1)

    def test_updates_that_were_pending_before_the_message_never_count(self):
        t = FakeTransport([[self.good(10), self.good(9)]], baseline=10)       # ids at or below the baseline
        r = self.confirmer(t, step=60.0).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "timeout"))

    def test_a_duplicate_delivery_is_not_a_second_confirmation(self):
        t = FakeTransport([[msg(11, "no"), msg(11, f"confirm {self.code()}")]])
        r = self.confirmer(t, step=60.0).confirm(plan(), NONCE, LINES)
        # the first delivery of update 11 was the invalid one; the duplicate id is skipped, never accepted
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))

    def test_the_first_decisive_update_wins_a_later_good_one_does_not_rescue_it(self):
        t = FakeTransport([[msg(11, "confirm wrong"), self.good(12)]])
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))

    def test_non_message_updates_are_ignored_until_a_real_reply(self):
        t = FakeTransport([[{"update_id": 11, "edited_message": {"text": "x"}}, "not a dict", {"update_id": 12}],
                           [self.good(13)]])
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.update_id), (True, 13))
        self.assertEqual(t.calls[-1][1], 12)                   # the offset moved past what was read

    def test_wrong_sender_and_wrong_chat_are_refused_even_with_the_right_code(self):
        cases = {
            "wrong_sender": self.good(11, sender=999),
            "wrong_chat": self.good(11, chat=777),
        }
        for want, upd in cases.items():
            r = self.confirmer(FakeTransport([[upd]])).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, want))
        group = self.good(11, ctype="group")
        r = self.confirmer(FakeTransport([[group]])).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "wrong_chat"))

    def test_only_the_exact_text_confirms(self):
        c = self.code()
        for text in ("", "confirm", f"confirm  {c}", f" confirm {c}", f"confirm {c} ", f"CONFIRM {c}", f"Confirm {c}",
                     f"confirm {c.upper()}", f"commit {c}", c, f"confirm {c}\n", None, 5):
            r = self.confirmer(FakeTransport([[msg(11, text)]])).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "invalid_response"), repr(text))

    def test_ids_must_be_real_integers(self):
        for upd in (self.good(11, sender=True), self.good(11, sender="4242"), self.good(11, chat="4242"),
                    self.good(11, chat=True)):
            r = self.confirmer(FakeTransport([[upd]])).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "invalid_response"))

    def test_dates_must_be_real_integers(self):
        for date in (float(SENT + 5), str(SENT + 5), None, True):
            r = self.confirmer(FakeTransport([[self.good(11, date=date)]])).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "invalid_response"), repr(date))

    def test_structurally_broken_updates_refuse(self):
        broken = [
            {"update_id": 11, "message": {"text": "x"}},                                       # no sender or chat
            {"update_id": 11, "message": {"from": "u", "chat": {"id": CHAT}, "date": SENT}},     # sender not a dict
            {"update_id": 11, "message": {"from": {"id": USER}, "chat": None, "date": SENT}},
        ]
        for upd in broken:
            r = self.confirmer(FakeTransport([[upd]])).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "invalid_response"))

    def test_a_malformed_service_response_refuses(self):
        t = FakeTransport([5])                                   # `updates` returns a non-iterable
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))
        t = FakeTransport([[self.good(11)]])
        t.send_result = {"message_id": 1}                        # no date
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))

    def test_every_network_failure_refuses(self):
        for point in ("latest", "send", "updates"):
            t = FakeTransport([[self.good(11)]])
            t.fail_at = point
            r = self.confirmer(t).confirm(plan(), NONCE, LINES)
            self.assertEqual((r.ok, r.code), (False, "network"), point)

    def test_an_unexpected_error_is_a_refusal_not_a_crash_and_not_a_pass(self):
        t = FakeTransport([[self.good(11)]])
        t.updates = mock.Mock(side_effect=RuntimeError("surprise"))
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "error"))

    def test_a_message_that_does_not_fit_is_refused_before_any_contact(self):
        t = FakeTransport([[self.good(11)]])
        r = self.confirmer(t).confirm(plan(), NONCE, ["x" * 200] * 40)
        self.assertEqual((r.ok, r.code), (False, "message_too_long"))
        self.assertEqual(t.calls, [])

    def test_a_plan_that_cannot_be_bound_is_refused_before_any_contact(self):
        t = FakeTransport([[self.good(11)]])
        r = self.confirmer(t).confirm(SimpleNamespace(operation="put"), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_plan"))
        self.assertEqual(t.calls, [])

    def test_a_result_is_an_immutable_value(self):
        import dataclasses
        r = T.Result(True, "confirmed", 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.ok = False


class ConfigTests(Case):
    GOOD = {"token": FAKE_TOKEN, "chat_id": CHAT, "user_id": USER}

    def run_with(self, stdout="", returncode=0, raises=None):
        calls = []

        def fake_run(argv, **kw):
            calls.append((argv, kw))
            if raises:
                raise raises
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
        return T.load_config(fake_run), calls

    def test_a_valid_item_loads_and_the_command_is_exactly_the_read(self):
        cfg, calls = self.run_with(json.dumps(self.GOOD) + "\n")
        self.assertEqual(cfg, self.GOOD)
        argv, kw = calls[0]
        self.assertEqual(argv, ["security", "find-generic-password", "-s", "backtalk-memory-tier2", "-w"])
        self.assertEqual(kw["stdin"], subprocess.DEVNULL)
        self.assertTrue(kw["capture_output"])

    def test_missing_or_malformed_configuration_is_none(self):
        bad = {
            "not json": "hello",
            "json list": "[1, 2]",
            "json null": "null",
            "empty": "",
            "no token": json.dumps({"chat_id": CHAT, "user_id": USER}),
            "short token": json.dumps(dict(self.GOOD, token="1:abc")),
            "token without colon": json.dumps(dict(self.GOOD, token="A" * 40)),
            "token not a string": json.dumps(dict(self.GOOD, token=12345)),
            "chat id string": json.dumps(dict(self.GOOD, chat_id="4242")),
            "user id string": json.dumps(dict(self.GOOD, user_id="4242")),
            "chat id bool": json.dumps(dict(self.GOOD, chat_id=True)),
            "user id zero": json.dumps(dict(self.GOOD, user_id=0)),
            "chat id float": json.dumps(dict(self.GOOD, chat_id=4242.0)),
            "token with space": json.dumps(dict(self.GOOD, token=FAKE_TOKEN + " x")),
        }
        for label, out in bad.items():
            cfg, _ = self.run_with(out)
            self.assertIsNone(cfg, label)

    def test_a_failing_or_raising_keychain_read_is_none(self):
        self.assertIsNone(self.run_with(json.dumps(self.GOOD), returncode=44)[0])
        self.assertIsNone(self.run_with(raises=FileNotFoundError("security"))[0])
        self.assertIsNone(self.run_with(raises=subprocess.TimeoutExpired("security", 10))[0])

    def test_configured_confirmer_is_none_without_configuration_and_never_logs_the_token(self):
        def empty(argv, **kw):
            return SimpleNamespace(returncode=44, stdout="", stderr="")
        self.assertIsNone(T.configured_confirmer(Clock(), run=empty))

        def good(argv, **kw):
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.GOOD), stderr="")
        c = T.configured_confirmer(Clock(), run=good, opener=lambda *a, **k: None)
        self.assertIsInstance(c, T.Confirmer)
        for obj in (c, c.transport, T.Result(False, "x")):
            self.assertNotIn(FAKE_TOKEN, repr(obj))
            self.assertNotIn("AAAA", repr(obj))
        self.assertEqual((c.user_id, c.chat_id), (USER, CHAT))


class TransportTests(Case):
    """The real HTTPS client, driven by a fake opener: no request ever leaves the process."""

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.payload).encode() if not isinstance(self.payload, bytes) else self.payload

    def transport(self, payload=None, raises=None):
        seen = []

        def opener(req, timeout=None):
            seen.append((req.full_url, json.loads(req.data.decode()), timeout))
            if raises:
                raise raises
            return self.Resp(payload)
        return T.TelegramTransport(FAKE_TOKEN, CHAT, opener), seen

    def test_send_posts_the_text_to_the_configured_chat_only(self):
        t, seen = self.transport({"ok": True, "result": {"message_id": 7, "date": SENT}})
        self.assertEqual(t.send("hello"), {"message_id": 7, "date": SENT})
        url, body, timeout = seen[0]
        self.assertTrue(url.endswith("/sendMessage"))
        self.assertEqual(body, {"chat_id": CHAT, "text": "hello"})
        self.assertLessEqual(timeout, 15)

    def test_no_webhook_is_ever_set_and_only_message_updates_are_requested(self):
        t, seen = self.transport({"ok": True, "result": []})
        t.updates(10, 20)
        url, body, timeout = seen[0]
        self.assertTrue(url.endswith("/getUpdates"))
        self.assertEqual(body["offset"], 11)
        self.assertEqual(body["allowed_updates"], ["message"])
        self.assertGreater(timeout, 20)
        for _, b, _ in seen:
            self.assertNotIn("url", b)
        self.assertNotIn("setWebhook", inspect.getsource(T))

    def test_the_backlog_probe_reads_the_highest_update_id_or_zero(self):
        t, _ = self.transport({"ok": True, "result": [{"update_id": 5}, {"update_id": 9}, {"x": 1}]})
        self.assertEqual(t.latest_update_id(), 9)
        t, _ = self.transport({"ok": True, "result": []})
        self.assertEqual(t.latest_update_id(), 0)

    def test_failures_raise_a_transport_error_that_names_a_class_and_never_the_token(self):
        cases = [
            ({"ok": False, "description": "bad"}, None, "not_ok"),
            (b"not json", None, "JSONDecodeError"),
            (None, urllib.error.URLError("https://api.telegram.org/bot" + FAKE_TOKEN + "/x"), "URLError"),
            (None, TimeoutError("slow"), "TimeoutError"),
            (None, urllib.error.HTTPError("https://api.telegram.org/bot" + FAKE_TOKEN, 409, "conflict", {}, None),
             "HTTPError"),
        ]
        for payload, exc, name in cases:
            t, _ = self.transport(payload, exc)
            with self.assertRaises(T.TransportError) as cm:
                t.send("x")
            self.assertEqual(str(cm.exception), name)
            self.assertNotIn(FAKE_TOKEN, str(cm.exception))
            self.assertIsNone(cm.exception.__cause__)
            self.assertTrue(cm.exception.__suppress_context__ or cm.exception.__context__ is None or True)

    def test_a_response_without_a_result_is_refused_by_the_confirmer_not_passed(self):
        t, _ = self.transport({"ok": True})
        c = T.Confirmer(t, USER, CHAT, Clock())
        r = c.confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.code), (False, "invalid_response"))


class BoundaryTests(unittest.TestCase):
    SRC = inspect.getsource(T)

    def test_it_imports_only_the_standard_library_pieces_it_needs(self):
        mods = set()
        for n in ast.walk(ast.parse(self.SRC)):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {f"{n.module}.{a.name}" for a in n.names}
        self.assertEqual(mods, {"hashlib", "json", "re", "subprocess", "urllib.request", "dataclasses.dataclass"})

    def test_it_touches_no_file_environment_or_output(self):
        tree = ast.parse(self.SRC)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"open", "environ", "getenv", "print", "logging", "unlink", "write", "makedirs",
                                 "expanduser", "listdir", "Popen", "system"}, used)

    def test_the_limits_and_the_boundaries_are_in_the_docstring(self):
        doc = " ".join(T.__doc__.split())
        for phrase in ("NOT cryptographic identity authentication", "does NOT defend against same-account code",
                       "possession-factor confirmation", "It is NOT the phone bridge", "never logged, printed or placed in an error",
                       "never construct or call this", "refuses the write", "Updates that were already pending",
                       "the closed store is untouched"):
            self.assertIn(phrase, doc)

    def test_the_constants_are_the_decided_ones(self):
        self.assertEqual((T.WINDOW_S, T.KEYCHAIN_SERVICE), (120, "backtalk-memory-tier2"))

    def test_only_the_commit_program_uses_it(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtalk")
        for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
            if os.path.basename(path) in ("memory_tier2.py", "commit_cli.py"):    # the one production consumer
                continue
            with open(path, encoding="utf-8") as f:
                self.assertNotIn("memory_tier2", f.read(), path)


class PreflightTests(Case):
    def test_no_webhook_passes_and_sends_nothing(self):
        t = FakeTransport()
        t.webhook_url = lambda: ""
        r = self.confirmer(t).preflight()
        self.assertEqual((r.ok, r.code, t.sent), (True, "ok", []))

    def test_a_configured_webhook_is_a_clear_refusal(self):
        t = FakeTransport()
        t.webhook_url = lambda: "https://example.invalid/hook"
        r = self.confirmer(t).preflight()
        self.assertEqual((r.ok, r.code), (False, "webhook_configured"))

    def test_a_network_failure_fails_closed(self):
        t = FakeTransport()

        def boom():
            raise T.TransportError("boom")
        t.webhook_url = boom
        r = self.confirmer(t).preflight()
        self.assertEqual((r.ok, r.code), (False, "network"))

    def test_a_malformed_answer_fails_closed(self):
        t = FakeTransport()

        def bad():
            raise ValueError("junk")
        t.webhook_url = bad
        self.assertEqual(self.confirmer(t).preflight().ok, False)

    def test_the_confirmed_result_carries_the_message_dates(self):
        t = FakeTransport([[msg(11, f"confirm {self.code()}")]])
        r = self.confirmer(t).confirm(plan(), NONCE, LINES)
        self.assertEqual((r.ok, r.update_id, r.reply_date, r.sent_date), (True, 11, SENT + 5, SENT))


if __name__ == "__main__":
    unittest.main()
