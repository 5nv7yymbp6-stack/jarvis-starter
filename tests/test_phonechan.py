import importlib.util
import json
import os
import tempfile
import unittest

from backtalk import phonechan as P

KEY = "k" * 64
NOW = 1_800_000_000.0
BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phone-bridge", "phone_queue.py")


def req(**over):
    r = {"v": 1, "id": "ab12cd34ef56ab12", "type": "remind", "title": "call the title company", "due_at": NOW + 600.0,
         "created_at": NOW}
    r.update(over)
    r["sig"] = over.get("sig") or P.sign(KEY, r)
    return r


class ValidateTests(unittest.TestCase):
    def test_a_good_request_is_accepted(self):
        self.assertEqual(P.validate(req(), KEY, NOW), (True, ""))

    def test_a_bad_or_missing_signature_is_rejected(self):
        self.assertEqual(P.validate(req(sig="0" * 64), KEY, NOW), (False, "rejected"))
        self.assertEqual(P.validate(req(), "other-key", NOW), (False, "rejected"))
        self.assertEqual(P.validate(req(), None, NOW), (False, "rejected"))
        r = req()
        r["title"] = "something else"                          # changed after signing
        self.assertEqual(P.validate(r, KEY, NOW), (False, "rejected"))

    def test_shape_is_strict(self):
        for bad in (req(type="shell"), req(v=2), req(id="short"), req(id="G" * 16), req(title=""), req(title="x" * 201),
                    req(title="line\nbreak"), req(title="  padded"), req(title="two  spaces"), req(title="bell\x07"),
                    req(due_at="soon", sig="0" * 64), req(due_at=float("nan")), req(due_at=True)):
            self.assertEqual(P.validate(bad, KEY, NOW), (False, "rejected"), bad)
        extra = req()
        extra["cmd"] = "rm -rf ~"
        self.assertEqual(P.validate(extra, KEY, NOW), (False, "rejected"))
        self.assertEqual(P.validate("not a dict", KEY, NOW), (False, "rejected"))

    def test_time_bounds(self):
        self.assertEqual(P.validate(req(due_at=NOW + P.MAX_AHEAD_S + 10), KEY, NOW)[0], False)
        self.assertEqual(P.validate(req(due_at=NOW - P.PAST_GRACE_S - 10), KEY, NOW), (False, "expired"))
        self.assertEqual(P.validate(req(due_at=NOW - 60), KEY, NOW), (True, ""))          # just missed: still spoken
        self.assertEqual(P.validate(req(created_at=NOW + 3600), KEY, NOW)[0], False)       # from the future
        self.assertEqual(P.validate(req(created_at=NOW - P.MAX_AGE_S - 10, due_at=NOW + 60), KEY, NOW)[0], False)


def instant(type_, title=None, **over):
    base = {"v": 1, "id": "cd34ef56ab12cd34", "type": type_, "title": title or ("status" if type_ == "status" else "buy milk"),
            "due_at": NOW, "created_at": NOW}
    base.update(over)
    base["sig"] = over.get("sig") or P.sign(KEY, base)
    return base


class InstantTypeValidationTests(unittest.TestCase):
    def test_note_and_status_validate_when_well_formed(self):
        self.assertEqual(P.validate(instant("note"), KEY, NOW + 5), (True, ""))
        self.assertEqual(P.validate(instant("status"), KEY, NOW + 5), (True, ""))

    def test_status_must_be_exactly_status_and_instants_must_have_no_schedule(self):
        self.assertEqual(P.validate(instant("status", title="do something"), KEY, NOW)[0], False)
        self.assertEqual(P.validate(instant("note", due_at=NOW + 3600), KEY, NOW)[0], False)
        self.assertEqual(P.validate(instant("status", due_at=NOW + 3600), KEY, NOW)[0], False)

    def test_an_instant_request_that_waited_too_long_is_dropped(self):
        self.assertEqual(P.validate(instant("note"), KEY, NOW + P.INSTANT_MAX_AGE_S + 5), (False, "expired"))
        self.assertEqual(P.validate(instant("status"), KEY, NOW + P.INSTANT_MAX_AGE_S + 5), (False, "expired"))

    def test_unknown_types_and_forgeries_are_rejected(self):
        for t in ("shell", "open", "delete", "REMIND", ""):
            self.assertEqual(P.validate(instant("note", type=t), KEY, NOW)[0], False, t)
        self.assertEqual(P.validate(instant("note", sig="0" * 64), KEY, NOW)[0], False)


class NoteTests(unittest.TestCase):
    def test_a_note_is_one_bullet_in_a_new_note_with_frontmatter(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "00 - Inbox", "Phone Notes.md")
            P.append_note("buy milk", NOW, path)
            P.append_note("call the title company", NOW + 60, path)
            text = open(path).read()
            self.assertTrue(text.startswith("---\nstatus: active"))
            self.assertEqual([l for l in text.splitlines() if l.startswith("- ")].__len__(), 2)
            self.assertIn("buy milk", text)

    def test_note_text_cannot_become_structure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.md")
            P.append_note("[[Secrets]] `rm -rf` <script> # heading", NOW, path)
            line = open(path).read().splitlines()[-1]
            for bad in ("[[", "]]", "`", "<", ">"):
                self.assertNotIn(bad, line)
            self.assertTrue(line.startswith("- "))

    def test_a_full_note_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.md")
            open(path, "w").write("x" * (P.NOTE_MAX_FILE + 1))
            with self.assertRaises(OSError):
                P.append_note("one more", NOW, path)


class StatusTests(unittest.TestCase):
    def test_the_summary_is_short_and_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "dream.log")
            open(log, "w").write("=== Tue Oct 20 05:00:01 CDT 2026 : dream run starting ===\n=== Tue Oct 20 05:02:07 CDT 2026 : dream run finished (exit 0) ===\n")
            out = P.compose_status(NOW, list_tasks=lambda: ["sch_1  scheduled once  next Wed 09:00  Start cold calls", "(1 request(s) queued)"],
                                   port_open=lambda: True, dream_log=log)
        self.assertIn("Voice line: up", out)
        self.assertIn("Start cold calls", out)
        self.assertIn("05:02:07", out)
        self.assertLess(len(out), 900)

    def test_failures_degrade_to_honest_unknowns(self):
        def boom():
            raise RuntimeError("x")
        out = P.compose_status(NOW, list_tasks=boom, port_open=boom, dream_log="/nonexistent")
        self.assertIn("DOWN", out)
        self.assertIn("nothing", out)
        self.assertIn("unknown", out)


class InstantPollTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.seen = P.Seen(os.path.join(self.d.name, "seen.json"))
        self.acks, self.notes, self.replies, self.inbox = [], [], [], []

    def run_(self, reqs, **kw):
        def runner(verb):
            if verb == "fetch":
                return 0, "".join(json.dumps(r) + "\n" for r in reqs)
            self.acks.append(verb)
            return 0, ""
        return P.poll(runner=runner, key=KEY, seen=self.seen, now=NOW + 2, write=self.inbox.append,
                      note=lambda title, now: self.notes.append(title), reply=self.replies.append, **kw)

    def test_a_note_is_appended_and_a_status_is_replied_and_neither_touches_the_scheduler(self):
        c = self.run_([instant("note"), instant("status", id="ee11ee11ee11ee11")])
        self.assertEqual(c["accepted"], 2)
        self.assertEqual(self.notes, ["buy milk"])
        self.assertEqual(len(self.replies), 1)
        self.assertEqual(self.inbox, [])
        self.assertEqual(len(self.acks), 2)

    def test_a_replayed_note_is_written_once(self):
        self.run_([instant("note")])
        self.run_([instant("note")])
        self.assertEqual(self.notes, ["buy milk"])

    def test_a_failing_action_leaves_it_queued_and_unrecorded(self):
        def boom(text):
            raise OSError("telegram down")
        def runner(verb):
            return (0, json.dumps(instant("status")) + "\n") if verb == "fetch" else (self.acks.append(verb) or (0, ""))
        c = P.poll(runner=runner, key=KEY, seen=self.seen, now=NOW + 2, write=self.inbox.append, reply=boom)
        self.assertEqual((c["error"], c["accepted"], self.acks), (1, 0, []))
        self.assertFalse(self.seen.has("cd34ef56ab12cd34"))

    def test_the_token_never_appears_in_what_the_status_function_sends(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            open(env, "w").write("TELEGRAM_BOT_TOKEN=123456789:SECRETSECRETSECRETSECRETSECRET\nALLOWED_TELEGRAM_IDS=42\n")
            sent = []

            class R:
                status = 200
                def __enter__(s): return s
                def __exit__(s, *a): return False
            P.send_telegram("hello", env_path=env, opener=lambda req, timeout=0: (sent.append(req), R())[1])
            (req,) = sent
            self.assertIn("SECRETSECRET", req.full_url)                      # the URL needs it ...
            self.assertNotIn("SECRET", req.data.decode())                    # ... the body does not carry it
            self.assertIn("chat_id=42", req.data.decode())


def act(action="screenshot", **over):
    return instant("act", title=action, **dict({"id": "aa11bb22cc33dd44"}, **over))


def confirm(code, **over):
    return instant("confirm", title=code, **dict({"id": "bb22cc33dd44ee55"}, **over))


class Tier2ValidationTests(unittest.TestCase):
    def test_act_only_for_listed_actions_and_confirm_only_for_six_digits(self):
        self.assertEqual(P.validate(act("screenshot"), KEY, NOW + 2), (True, ""))
        self.assertEqual(P.validate(act("restart_voice"), KEY, NOW + 2), (True, ""))
        for bad in ("shell", "rm -rf", "lock", "screenshot ", "Screenshot", ""):
            self.assertEqual(P.validate(act(bad), KEY, NOW)[0], False, bad)
        self.assertEqual(P.validate(confirm("123456"), KEY, NOW + 2), (True, ""))
        for bad in ("12345", "1234567", "abcdef", "12 456", "123456 "):
            self.assertEqual(P.validate(confirm(bad), KEY, NOW)[0], False, bad)

    def test_they_carry_no_schedule_and_expire(self):
        self.assertEqual(P.validate(act(due_at=NOW + 600), KEY, NOW)[0], False)
        self.assertEqual(P.validate(act(), KEY, NOW + P.INSTANT_MAX_AGE_S + 5), (False, "expired"))
        self.assertEqual(P.validate(act(sig="0" * 64), KEY, NOW)[0], False)


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.path = os.path.join(self.d.name, "ch.json")

    def ch(self):
        return P.Challenges(self.path)

    def test_a_code_is_six_digits_valid_once_for_its_action(self):
        c = self.ch()
        code = c.issue("r1", "screenshot", NOW)
        self.assertRegex(code, r"^\d{6}$")
        self.assertEqual(c.check(code, NOW + 10), ("ok", "screenshot"))
        self.assertEqual(c.check(code, NOW + 11)[0], "wrong")                         # used

    def test_the_code_is_stored_hashed_and_the_file_is_private(self):
        c = self.ch()
        code = c.issue("r1", "restart_voice", NOW)
        stored = json.load(open(self.path))["pending"]["r1"]
        self.assertEqual(set(stored), {"action", "salt", "h", "exp"})                  # no field holds the code itself
        self.assertNotEqual(stored["h"], code)
        self.assertRegex(stored["h"], r"^[0-9a-f]{64}$")
        self.assertEqual(oct(os.stat(self.path).st_mode & 0o777), "0o600")

    def test_it_survives_a_new_process_and_expires(self):
        code = self.ch().issue("r1", "screenshot", NOW)
        self.assertEqual(self.ch().check(code, NOW + P.CHALLENGE_TTL_S - 1), ("ok", "screenshot"))
        code = self.ch().issue("r2", "screenshot", NOW)
        self.assertEqual(self.ch().check(code, NOW + P.CHALLENGE_TTL_S + 1)[0], "wrong")

    def test_only_a_few_codes_are_outstanding(self):
        c = self.ch()
        codes = [c.issue(f"r{i}", "screenshot", NOW + i) for i in range(P.CHALLENGE_PENDING_MAX + 2)]
        self.assertEqual(len(c.pending), P.CHALLENGE_PENDING_MAX)
        self.assertEqual(c.check(codes[0], NOW + 10)[0], "wrong")                        # the oldest was dropped

    def test_wrong_guesses_lock_everything_for_ten_minutes(self):
        c = self.ch()
        good = c.issue("r1", "screenshot", NOW)
        wrong = "000000" if good != "000000" else "111111"
        for i in range(P.GUESS_LIMIT - 1):
            self.assertEqual(c.check(wrong, NOW + i)[0], "wrong")
        self.assertEqual(c.check(wrong, NOW + 10)[0], "locked")
        self.assertEqual(c.check(good, NOW + 11)[0], "locked")                           # even the right code is refused now
        self.assertIsNone(c.issue("r2", "screenshot", NOW + 12))
        later = NOW + 10 + P.LOCKOUT_S + 1
        self.assertIsNotNone(self.ch().issue("r3", "screenshot", later))

    def test_a_wrong_code_does_not_consume_the_right_one(self):
        c = self.ch()
        good = c.issue("r1", "screenshot", NOW)
        c.check("000000" if good != "000000" else "111111", NOW + 1)
        self.assertEqual(c.check(good, NOW + 2), ("ok", "screenshot"))


class Tier2PollTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.seen = P.Seen(os.path.join(self.d.name, "seen.json"))
        self.ch = P.Challenges(os.path.join(self.d.name, "ch.json"))
        self.acks, self.replies, self.ran, self.inbox = [], [], [], []

    def run_(self, reqs, now=NOW + 2):
        def runner(verb):
            if verb == "fetch":
                return 0, "".join(json.dumps(r) + "\n" for r in reqs)
            self.acks.append(verb)
            return 0, ""
        ex = {"screenshot": lambda: (self.ran.append("screenshot"), "shot sent")[1],
              "restart_voice": lambda: (self.ran.append("restart_voice"), "restarting")[1]}
        return P.poll(runner=runner, key=KEY, seen=self.seen, now=now, write=self.inbox.append, reply=self.replies.append,
                      challenges=self.ch, executors=ex)

    def code_from_reply(self):
        import re
        return re.search(r"/yes (\d{6})", self.replies[-1]).group(1)

    def test_a_request_alone_runs_nothing_and_texts_a_code(self):
        self.run_([act("screenshot")])
        self.assertEqual(self.ran, [])
        self.assertRegex(self.replies[-1], r"/yes \d{6}")
        self.assertIn("screenshot", self.replies[-1])
        self.assertEqual(len(self.inbox), 1)                                             # and a spoken notice for the Mac
        self.assertIn("waiting for your code", self.inbox[0]["title"])

    def test_the_right_code_runs_the_action_once(self):
        self.run_([act("restart_voice")])
        code = self.code_from_reply()
        self.run_([confirm(code)], now=NOW + 30)
        self.assertEqual(self.ran, ["restart_voice"])
        self.assertEqual(self.replies[-1], "restarting")
        self.run_([confirm(code, id="cc33dd44ee55ff66")], now=NOW + 40)                  # the same code again
        self.assertEqual(self.ran, ["restart_voice"])
        self.assertIn("wrong or has expired", self.replies[-1])

    def test_a_wrong_code_or_no_challenge_runs_nothing(self):
        self.run_([confirm("123456")])
        self.assertEqual(self.ran, [])
        self.assertIn("wrong or has expired", self.replies[-1])

    def test_a_forged_act_or_confirm_is_rejected_before_anything_happens(self):
        c = self.run_([act("screenshot", sig="0" * 64), confirm("123456", sig="0" * 64)])
        self.assertEqual((c["rejected"], self.ran, self.replies), (2, [], []))
        self.assertEqual(self.ch.pending, {})

    def test_a_replayed_confirm_request_cannot_run_it_twice(self):
        self.run_([act("screenshot")])
        code = self.code_from_reply()
        r = confirm(code)
        self.run_([r], now=NOW + 30)
        self.run_([r], now=NOW + 31)                                                      # the same signed request again
        self.assertEqual(self.ran, ["screenshot"])

    def test_an_action_that_fails_after_approval_says_so_and_does_not_retry(self):
        self.run_([act("screenshot")])
        code = self.code_from_reply()

        def boom():
            raise OSError("no permission")
        def runner(verb):
            return (0, json.dumps(confirm(code)) + "\n") if verb == "fetch" else (self.acks.append(verb) or (0, ""))
        P.poll(runner=runner, key=KEY, seen=self.seen, now=NOW + 30, write=self.inbox.append, reply=self.replies.append,
               challenges=self.ch, executors={"screenshot": boom})
        self.assertIn("failed (OSError)", self.replies[-1])
        self.assertEqual(self.ch.pending, {})

    def test_the_lockout_message_after_too_many_wrong_codes(self):
        self.run_([act("screenshot")])
        for i in range(P.GUESS_LIMIT):
            self.run_([confirm("000000", id=f"{i:016x}")], now=NOW + 5 + i)
        self.assertIn("locked", self.replies[-1])


class ExecutorTests(unittest.TestCase):
    def test_screenshot_sends_a_photo_and_deletes_the_file(self):
        sent, paths = [], []

        def run(cmd, **kw):
            paths.append(cmd[-1])
            open(cmd[-1], "wb").write(b"\xff\xd8" + b"x" * 8000)
        out = P.run_screenshot(now=NOW, run=run, photo=lambda path, cap: (sent.append((os.path.getsize(path), cap)), None)[1])
        self.assertEqual(out, "Screenshot sent, sir.")
        self.assertEqual(sent[0][0], 8002)
        self.assertFalse(os.path.exists(paths[0]))

    def test_an_empty_capture_is_an_error_not_a_blank_photo(self):
        def run(cmd, **kw):
            open(cmd[-1], "wb").write(b"x")
        with self.assertRaises(OSError):
            P.run_screenshot(now=NOW, run=run, photo=lambda *a: None)

    def test_restart_voice_interrupts_only_the_voice_process_then_relaunches(self):
        calls, killed = [], []
        listing = ("  111 /Users/x/backtalk/.venv/bin/python3 -m backtalk.main\n  222 uv run python -m backtalk.main\n"
                   "  333 /Users/x/backtalk/.venv/bin/python -m backtalk.phonechan poll\n  444 /usr/bin/other\n")
        state = {"n": 0}

        def run(cmd, **kw):
            calls.append(cmd[0])
            if cmd[0] == "ps":
                state["n"] += 1
                class R: stdout = listing if state["n"] == 1 else "  444 /usr/bin/other\n"
                return R
        out = P.run_restart_voice(run=run, kill=lambda pid, sig: killed.append(pid), sleep=lambda s: None)
        self.assertEqual(killed, [111])
        self.assertEqual(calls[-1], "open")
        self.assertIn("Restarting", out)

    def test_send_photo_puts_the_token_in_the_url_only(self):
        with tempfile.TemporaryDirectory() as d:
            env, img = os.path.join(d, ".env"), os.path.join(d, "a.jpg")
            open(env, "w").write("TELEGRAM_BOT_TOKEN=123456789:SECRETSECRETSECRETSECRETSECRET\nALLOWED_TELEGRAM_IDS=42\n")
            open(img, "wb").write(b"JPEGDATA")
            sent = []

            class R:
                status = 200
                def __enter__(s): return s
                def __exit__(s, *a): return False
            P.send_photo(img, "cap", env_path=env, opener=lambda req, timeout=0: (sent.append(req), R())[1])
            (req,) = sent
            self.assertIn("SECRETSECRET", req.full_url)
            self.assertNotIn(b"SECRET", req.data)
            self.assertIn(b"JPEGDATA", req.data)
            self.assertIn(b'name="chat_id"\r\n\r\n42', req.data)


class KeyTests(unittest.TestCase):
    def test_only_a_private_nonempty_file_is_a_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k")
            self.assertIsNone(P.load_key(p))
            open(p, "w").write("  secret\n")
            os.chmod(p, 0o644)
            self.assertIsNone(P.load_key(p))
            os.chmod(p, 0o600)
            self.assertEqual(P.load_key(p), "secret")
            open(p, "w").write("\n")
            self.assertIsNone(P.load_key(p))


class PollTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.seen = P.Seen(os.path.join(self.d.name, "seen.json"))
        self.queue, self.acks, self.written, self.logs = [], [], [], []

    def runner(self, verb):
        if verb == "fetch":
            return 0, "".join(json.dumps(r) + "\n" for r in self.queue)
        self.acks.append(verb)
        return 0, ""

    def poll(self, **kw):
        return P.poll(runner=kw.pop("runner", self.runner), key=KEY, seen=self.seen, now=NOW, write=self.written.append,
                      log=self.logs.append, **kw)

    def test_an_accepted_reminder_lands_in_the_scheduler_inbox_and_is_acked(self):
        self.queue = [req()]
        c = self.poll()
        self.assertEqual(c["accepted"], 1)
        (w,) = self.written
        self.assertEqual((w["op"], w["kind"], w["title"], w["severity"]), ("add", "once", "call the title company", "normal"))
        self.assertEqual(w["due_at"], NOW + 600.0)
        self.assertEqual(self.acks, ["ack ab12cd34ef56ab12 ok"])

    def test_the_same_request_twice_is_a_replay(self):
        self.queue = [req()]
        self.poll()
        c = self.poll()
        self.assertEqual((c["accepted"], c["replay"]), (0, 1))
        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.acks[-1], "ack ab12cd34ef56ab12 replay")

    def test_replay_protection_survives_a_restart(self):
        self.queue = [req()]
        self.poll()
        self.seen = P.Seen(self.seen.path)                           # a new process reading the saved state
        self.assertEqual(self.poll()["replay"], 1)

    def test_forged_and_expired_requests_are_acked_so_they_drain_and_never_written(self):
        self.queue = [req(sig="0" * 64, id="1" * 16), req(id="2" * 16, due_at=NOW - 10 * 3600)]
        c = self.poll()
        self.assertEqual((c["rejected"], c["expired"], c["accepted"]), (1, 1, 0))
        self.assertEqual(self.written, [])
        self.assertEqual(sorted(self.acks), ["ack " + "1" * 16 + " rejected", "ack " + "2" * 16 + " expired"])

    def test_garbage_lines_never_crash_the_pass(self):
        def runner(verb):
            return (0, "not json\n[1,2]\n{\"id\": 5}\n\n") if verb == "fetch" else (0, "")
        c = self.poll(runner=runner)
        self.assertEqual(c["accepted"], 0)
        self.assertEqual(self.written, [])

    def test_a_failed_fetch_changes_nothing(self):
        c = self.poll(runner=lambda v: (255, ""))
        self.assertEqual((c["error"], self.written, self.acks), (1, [], []))

    def test_a_scheduler_write_failure_leaves_it_queued(self):
        self.queue = [req()]

        def boom(r):
            raise OSError("disk")
        c = P.poll(runner=self.runner, key=KEY, seen=self.seen, now=NOW, write=boom, log=self.logs.append)
        self.assertEqual((c["error"], c["accepted"], self.acks), (1, 0, []))
        self.assertFalse(self.seen.has("ab12cd34ef56ab12"))             # so the retry is not mistaken for a replay

    def test_the_daily_limit_leaves_the_rest_queued(self):
        self.seen.times = [NOW - 100 + i for i in range(P.MAX_PER_DAY)]
        self.queue = [req()]
        c = self.poll()
        self.assertEqual((c["limited"], c["accepted"], self.acks, self.written), (1, 0, [], []))

    def test_no_key_means_nothing_is_processed(self):
        c = P.poll(runner=self.runner, key="", seen=self.seen, now=NOW, write=self.written.append, log=self.logs.append)
        self.assertEqual((c["error"], self.written), (1, []))

    def test_the_log_never_contains_the_reminder_text_or_the_signature(self):
        self.queue = [req()]
        self.poll()
        blob = " ".join(self.logs)
        self.assertNotIn("title company", blob)
        self.assertNotIn(self.queue[0]["sig"], blob)


class KillSwitchAndSshTests(unittest.TestCase):
    def test_the_switch_file_stops_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            sw = os.path.join(d, "off")
            open(sw, "w").close()
            old, P.SWITCH_OFF = P.SWITCH_OFF, sw
            try:
                called = []
                real = P.poll
                P.poll = lambda **k: called.append(1)
                self.assertEqual(P.main(["poll"]), 0)
                self.assertEqual(called, [])
            finally:
                P.SWITCH_OFF, P.poll = old, real

    def test_ssh_is_restricted_and_runs_only_the_given_verb(self):
        import inspect
        src = inspect.getsource(P.ssh_runner)
        for need in ("BatchMode=yes", "IdentitiesOnly=yes", "StrictHostKeyChecking=yes", "capture_output=True"):
            self.assertIn(need, src)
        self.assertNotIn("shell=True", src)


@unittest.skipUnless(os.path.exists(BRIDGE), "phone-bridge not next to backtalk")
class CrossCompatibilityTests(unittest.TestCase):
    def test_act_and_confirm_requests_built_by_the_server_code_verify_here(self):
        spec = importlib.util.spec_from_file_location("phone_queue", BRIDGE)
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        with tempfile.TemporaryDirectory() as d:
            q.handle_command("/screenshot", KEY, d, NOW)
            q.handle_command("/restartvoice", KEY, d, NOW + 0.1)
            q.handle_command("/yes 482913", KEY, d, NOW + 0.2)
            reqs = [json.load(open(os.path.join(d, n))) for n in q.pending(d)]
        self.assertEqual(sorted((r["type"], r["title"]) for r in reqs),
                         [("act", "restart_voice"), ("act", "screenshot"), ("confirm", "482913")])
        for r in reqs:
            self.assertEqual(P.validate(r, KEY, NOW + 5), (True, ""), r["title"])

    """The server's signer and the Mac's verifier are separate files; this proves they agree."""

    def test_a_request_built_by_the_server_code_verifies_here(self):
        spec = importlib.util.spec_from_file_location("phone_queue", BRIDGE)
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        title, due = q.parse_remind("/remind in 10m call the title company", NOW)
        r = q.build_request(KEY, title, due, NOW)
        self.assertEqual(P.validate(r, KEY, NOW + 5), (True, ""))
        self.assertEqual(P.sign(KEY, r), r["sig"])
        self.assertEqual(P.validate(r, "wrong", NOW + 5), (False, "rejected"))

    def test_status_and_note_requests_built_by_the_server_code_verify_here(self):
        spec = importlib.util.spec_from_file_location("phone_queue", BRIDGE)
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        with tempfile.TemporaryDirectory() as d:
            q.handle_command("/status", KEY, d, NOW)
            q.handle_command("/note call the title company", KEY, d, NOW)
            reqs = [json.load(open(os.path.join(d, n))) for n in q.pending(d)]
        self.assertEqual(sorted(r["type"] for r in reqs), ["note", "status"])
        for r in reqs:
            self.assertEqual(P.validate(r, KEY, NOW + 5), (True, ""), r["type"])


LATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phone-bridge", "late_reminders.py")


class FakeView:
    def __init__(self, tasks):
        self._t = tasks

    def tasks(self):
        return self._t


def task(id_, title, status="scheduled", next_=2_000_000_000.0, retry_at=None, slot=None):
    from types import SimpleNamespace
    return SimpleNamespace(id=id_, title=title, status=status, next=next_, retry_at=retry_at, slot=slot)


class MirrorTasksTests(unittest.TestCase):
    def test_scheduled_and_running_tasks_are_mirrored_soonest_first_others_are_not(self):
        view = FakeView([task("sch_b", "second", next_=2_000_000_200.0), task("sch_a", "first", next_=2_000_000_100.0),
                         task("sch_c", "done", status="done"), task("sch_d", "failed", status="failed"),
                         task("sch_e", "cancelled", status="cancelled")])
        out = P.mirror_tasks(view=view)
        self.assertEqual([t["title"] for t in out], ["first", "second"])
        self.assertEqual(out[0]["k"], "sch_a@2000000100")

    def test_a_fired_but_unfinished_task_is_keyed_by_the_slot_it_was_due_at(self):
        out = P.mirror_tasks(view=FakeView([task("sch_a", "x", status="running", next_=None, slot=1_900_000_000.0)]))
        self.assertEqual(out, [{"k": "sch_a@1900000000", "due": 1_900_000_000.0, "title": "x"}])

    def test_a_retried_task_keeps_its_original_due_slot_not_the_retry_time(self):
        out = P.mirror_tasks(view=FakeView([task("sch_a", "x", next_=1_900_000_030.0, retry_at=1_900_000_050.0, slot=1_900_000_000.0)]))
        self.assertEqual(out[0]["due"], 1_900_000_000.0)                      # when he expected it, and the identity does not change on a retry
        self.assertEqual(out[0]["k"], "sch_a@1900000000")

    def test_titles_are_cleaned_unusable_tasks_are_skipped_and_the_list_is_capped(self):
        view = FakeView([task("sch_a", "a\nb\x07 c"), task("sch_b", "   "), task("sch_c", "x", next_=None), task("sch_d", "y", next_=float("nan"))]
                        + [task(f"sch_{i:03d}", f"t{i}", next_=2_000_000_000.0 + i) for i in range(80)])
        out = P.mirror_tasks(view=view)
        self.assertEqual(out[0]["title"], "a b c")
        self.assertEqual(len(out), P.MIRROR_MAX_TASKS)
        self.assertNotIn("sch_b", " ".join(t["k"] for t in out))


class SyncMirrorTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.old = (P.MIRROR_OFF, P.MIRROR_STATE)
        P.MIRROR_OFF, P.MIRROR_STATE = os.path.join(self.d.name, "off"), os.path.join(self.d.name, "last.json")
        self.addCleanup(lambda: setattr(P, "MIRROR_OFF", self.old[0]) or setattr(P, "MIRROR_STATE", self.old[1]))
        self.sent, self.logs = [], []

    def runner(self, verb, stdin=None, code=0):
        self.sent.append((verb, stdin))
        return code, ""

    def test_it_sends_one_signed_document_the_servers_verifier_accepts(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("late_reminders", LATE)
        L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(L)
        tasks = [{"k": "sch_a@2000000100", "due": 2_000_000_100.0, "title": "call the title company"}]
        self.assertTrue(P.sync_mirror(KEY, NOW, runner=self.runner, tasks=tasks, log=self.logs.append))
        (verb, doc), = self.sent
        self.assertEqual(verb, "mirror")
        m = L.accept_mirror(doc, KEY, NOW + 5)
        self.assertIsNotNone(m)
        self.assertEqual(m["tasks"], tasks)
        self.assertIsNone(L.accept_mirror(doc, "other-key", NOW + 5))

    def test_an_empty_list_is_still_sent_so_a_cleared_reminder_is_known(self):
        self.assertTrue(P.sync_mirror(KEY, NOW, runner=self.runner, tasks=[], log=self.logs.append))
        self.assertEqual(json.loads(self.sent[0][1])["tasks"], [])

    def test_the_off_switch_sends_nothing(self):
        open(P.MIRROR_OFF, "w").close()
        self.assertFalse(P.sync_mirror(KEY, NOW, runner=self.runner, tasks=[], log=self.logs.append))
        self.assertEqual(self.sent, [])

    def test_a_refused_or_crashing_send_is_reported_and_never_raised(self):
        self.assertFalse(P.sync_mirror(KEY, NOW, runner=lambda v, stdin=None: (255, ""), tasks=[], log=self.logs.append))
        self.assertIn("not accepted (exit 255)", self.logs[-1])

        def boom(v, stdin=None):
            raise OSError("ssh")
        self.assertFalse(P.sync_mirror(KEY, NOW, runner=boom, tasks=[], log=self.logs.append))
        self.assertIn("mirror failed (OSError)", self.logs[-1])

    def test_it_logs_only_when_the_set_of_reminders_changes(self):
        t1 = [{"k": "sch_a@2000000100", "due": 2_000_000_100.0, "title": "a"}]
        for _ in range(3):
            P.sync_mirror(KEY, NOW, runner=self.runner, tasks=t1, log=self.logs.append)
        self.assertEqual(sum("mirror synced" in l for l in self.logs), 1)
        P.sync_mirror(KEY, NOW, runner=self.runner, tasks=[], log=self.logs.append)
        self.assertEqual(sum("mirror synced" in l for l in self.logs), 2)

    def test_the_real_ssh_runner_feeds_stdin_to_the_restricted_command_only(self):
        import inspect
        src = inspect.getsource(P.ssh_runner)
        self.assertIn("input=stdin", src)
        self.assertNotIn("shell=True", src)


class PollMirrorWiringTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.seen = P.Seen(os.path.join(self.d.name, "seen.json"))
        self.calls = []

    def poll(self, runner, key=KEY, mirror="default"):
        mirror = (lambda k, n: self.calls.append((k, n))) if mirror == "default" else mirror
        return P.poll(runner=runner, key=key, seen=self.seen, now=NOW, write=lambda r: None, log=lambda l: None, mirror=mirror,
                      challenges=P.Challenges(os.path.join(self.d.name, "ch.json")))

    def test_the_mirror_runs_once_after_a_good_pass_even_with_an_empty_queue(self):
        self.poll(lambda v: (0, ""))
        self.assertEqual(len(self.calls), 1)

    def test_it_is_skipped_when_the_server_cannot_be_reached_or_there_is_no_key(self):
        self.poll(lambda v: (255, ""))
        self.poll(lambda v: (0, ""), key="")
        self.assertEqual(self.calls, [])

    def test_a_crashing_mirror_does_not_change_the_pass_result(self):
        def boom(k, n):
            raise RuntimeError("x")
        c = self.poll(lambda v: (0, ""), mirror=boom)
        self.assertEqual(c["error"], 0)

    def test_main_wires_the_real_mirror_and_the_default_is_off_for_tests(self):
        import inspect
        self.assertIn("mirror=lambda key, now: sync_mirror(key, now, log=_log)", inspect.getsource(P.main))
        self.assertIsNone(inspect.signature(P.poll).parameters["mirror"].default)


if __name__ == "__main__":
    unittest.main()


class MirrorResTests(unittest.TestCase):
    """What the Mac's REAL scheduler recorded for a reminder that left the pending list, and that the server's real checker reads it
    (no mock of either side)."""

    def setUp(self):
        from backtalk import scheduler as S
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = [1_800_000_000.0]
        self.s = S.Scheduler(self.tmp.name, clock=lambda: self.t[0], fsync=False)
        self.addCleanup(self.s.close)

    def res(self):
        return {r["k"]: r["r"] for r in P.mirror_res(self.t[0], view=self.s)}

    def test_heard_cancelled_failed_and_missed_are_each_named_and_pending_ones_are_not(self):
        a = self.s.add_once("heard", self.t[0] + 10).id
        b = self.s.add_once("cancelled", self.t[0] + 20).id
        c = self.s.add_once("pending", self.t[0] + 9999).id
        d = self.s.add_once("failing", self.t[0] + 10, max_retries=0).id
        self.s.cancel(b)
        self.t[0] += 15
        fires = {f.title: f for f in self.s.due()}
        self.s.complete(fires["heard"].run_id)
        self.s.fail(fires["failing"].run_id, "flushed")
        got = self.res()
        self.assertEqual(got[f"{a}@{int(fires['heard'].slot)}"], "done")
        self.assertEqual(got[f"{d}@{int(fires['failing'].slot)}"], "failed")
        self.assertIn("cancelled", got.values())
        self.assertFalse([k for k in got if k.startswith(c)])

    def test_a_running_or_retrying_reminder_has_no_final_outcome(self):
        a = self.s.add_once("x", self.t[0] + 5).id
        self.t[0] += 10
        f = self.s.due()[0]
        self.assertEqual(self.res(), {})                                    # running
        self.s.fail(f.run_id, "flushed")                                   # fails with a retry pending
        self.assertEqual(self.res(), {})

    def test_old_outcomes_age_out(self):
        self.s.add_once("x", self.t[0] + 5).id
        self.t[0] += 10
        self.s.complete(self.s.due()[0].run_id)
        self.assertEqual(len(self.res()), 1)
        self.t[0] += P.RES_WINDOW_S + 5
        self.assertEqual(self.res(), {})

    @unittest.skipUnless(os.path.exists(LATE), "phone-bridge not next to backtalk")
    def test_the_servers_checker_resolves_exactly_what_the_mac_recorded(self):
        spec = importlib.util.spec_from_file_location("late_reminders", LATE)
        L = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(L)
        sent, key = [], "k" * 64
        chk = L.Checker(os.path.join(self.tmp.name, "late.json"), os.path.join(self.tmp.name, "mirror.json"), key, 1,
                        send=lambda c, text: (sent.append(text) or True, 500 + len(sent)))
        a = self.s.add_once("heard one", self.t[0] + 10).id
        b = self.s.add_once("late one", self.t[0] + 10).id
        due = self.t[0] + 10

        def sync(now):
            tasks = P.mirror_tasks(now, view=self.s)
            doc = P.build_mirror(key, tasks, now, P.mirror_res(now, view=self.s))
            with open(os.path.join(self.tmp.name, "mirror.json"), "w") as f:
                json.dump(doc, f)
            return chk.run_once(now)
        self.t[0] = due - 5
        sync(self.t[0])                                                     # both pending and known to the server
        self.t[0] = due + 1
        fires = {f.title: f for f in self.s.due()}
        self.s.complete(fires["heard one"].run_id)                          # the Mac heard the first one
        self.t[0] = due + 200
        sync(self.t[0])                                                     # the second is still running: no outcome
        state = json.load(open(os.path.join(self.tmp.name, "late.json")))
        self.assertEqual(state[f"{a}@{int(due)}"]["status"], "completed_on_mac")
        self.assertEqual(len(sent), 1, sent)                                # only the unconfirmed one was texted
        self.assertIn("late one", sent[0])
        self.assertEqual(state[f"{b}@{int(due)}"]["status"], "sent")
        self.s.complete(fires["late one"].run_id)                           # it is heard afterwards
        sync(due + 230)
        state = json.load(open(os.path.join(self.tmp.name, "late.json")))
        self.assertEqual(state[f"{b}@{int(due)}"]["status"], "sent_then_completed_on_mac")
        self.assertEqual(len(sent), 1)
        text = L.summary(state, due + 240)
        self.assertIn("completed it", text)


class SshErrorTailTests(unittest.TestCase):
    def test_a_failed_pass_logs_the_last_stderr_line_sanitised_and_bounded(self):
        from types import SimpleNamespace
        from unittest import mock
        fake = SimpleNamespace(returncode=255, stdout="", stderr="warn\nssh: connect to host x port 22: Network is unreachable\x07" + "z" * 300)
        with mock.patch.object(P.subprocess, "run", return_value=fake):
            self.assertEqual(P.ssh_runner("fetch"), (255, ""))
        self.assertLessEqual(len(P.LAST_SSH_ERR), 120)
        self.assertNotIn("\x07", P.LAST_SSH_ERR)
        logs = []
        with tempfile.TemporaryDirectory() as d:
            P.poll(runner=lambda verb, stdin=None: (255, ""), key="k" * 64, seen=P.Seen(os.path.join(d, "seen.json")), log=logs.append)
        self.assertTrue(any(l.startswith("fetch failed (exit 255)") and "Network" in l for l in logs), logs)
