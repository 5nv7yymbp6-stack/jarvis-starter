"""Wiring of proactive behavior (main.py helpers, brain options, the remind CLI). Default OFF; a broken scheduler must never
take the voice line down; the CLI's requests are what the runner really consumes."""
import asyncio
import inspect
import io
import json
import os
import stat
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import main as M
from backtalk import mouth as MOUTH
from backtalk import proactive as P
from backtalk import proactive_loop as PL
from backtalk import remind as R
from backtalk import scheduler as S
from backtalk import speech_policy as SP

T0 = 1_800_000_000.0


class FakeMouth:
    busy = False
    generation = 0

    def say(self, text):
        pass

    def recent_speech(self, window_s=30.0):
        return []


def fake_brain(tmp):
    led = L.Ledger(os.path.join(tmp, "ledger"), fsync=False)
    return SimpleNamespace(ledger=led, speech=SP.SpeechPolicy(led), router=object(), _need_rebuild=None)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs = []
        self.saved = dict(M.CFG)
        self.real_log = M.log
        M.log = self.logs.append
        self.brain = fake_brain(self.tmp.name)
        self.sdir = os.path.join(self.tmp.name, "scheduler")
        self.built = []

    def tearDown(self):
        M.log = self.real_log
        M.CFG.clear()
        M.CFG.update(self.saved)
        for r in self.built:
            r.sched.close()
        M._PROACTIVE.update(runner=None, restarts=0)
        self.tmp.cleanup()

    def build(self, **kw):
        r = M._proactive_build(self.brain, FakeMouth(), lambda: P.Context(), sched_dir=self.sdir, **kw)
        if r:
            self.built.append(r)
        return r

    def test_it_is_off_by_default_and_builds_nothing(self):
        M.CFG.pop("proactive", None)
        self.assertIsNone(self.build())
        self.assertFalse(os.path.exists(self.sdir))                    # not even a directory
        M.CFG["proactive"] = {"enabled": "true"}                        # only the literal true enables
        self.assertIsNone(self.build())
        self.assertIsNone(M._proactive_start(self.brain, FakeMouth(), lambda: P.Context()))

    def test_enabled_builds_a_runner_with_a_scheduler_in_the_ledger_directory(self):
        M.CFG["proactive"] = {"enabled": True}
        r = self.build()
        self.assertIsInstance(r, PL.Runner)
        self.assertTrue(r.engine.policy.enabled)
        self.assertTrue(any(l.startswith("[proactive] on ") for l in self.logs))
        self.assertTrue(os.path.isdir(self.sdir))

    def test_the_default_scheduler_directory_lives_under_the_ledger_directory_so_tests_stay_isolated(self):
        src = inspect.getsource(M._proactive_build)
        self.assertIn('ledger_mod.default_dir()', src)
        self.assertIn('"scheduler"', src)
        self.assertTrue(str(L.default_dir()).startswith(tempfile.gettempdir().rstrip("/")) or "backtalk-tests" in str(L.default_dir()))

    def test_a_corrupt_scheduler_file_means_proactive_is_off_and_the_voice_line_starts(self):
        M.CFG["proactive"] = {"enabled": True}
        os.makedirs(self.sdir)
        with open(os.path.join(self.sdir, "scheduler.jsonl"), "w") as fh:
            fh.write("garbage\n{\"t\":\"sched\"}\n")
        self.assertIsNone(self.build())
        self.assertTrue(any("could not start (SchedulerCorrupt)" in l for l in self.logs))

    def test_a_locked_scheduler_means_proactive_is_off_and_nothing_raises(self):
        M.CFG["proactive"] = {"enabled": True}
        first = self.build()
        self.assertIsNotNone(first)
        self.assertIsNone(self.build())                                 # a second build in the same directory: locked
        self.assertTrue(any("could not start (SchedulerLocked)" in l for l in self.logs))

    def test_a_bad_config_section_is_off_not_an_error(self):
        for bad in (None, "on", 5, [], {"enabled": []}):
            M.CFG["proactive"] = bad
            self.assertIsNone(self.build(), bad)

    def test_the_voice_context_reads_the_ptt_key_and_never_the_addressee_gate_for_speaking(self):
        src = inspect.getsource(M.amain)
        body = src[src.index("def _proactive_context"):src.index("_proactive_start(brain, mouth, _proactive_context)")]
        self.assertIn('user_speaking=bool(_MIC["btn"])', body)
        self.assertNotIn(".open(", body)
        self.assertNotIn(".touch(", body)

    def test_the_start_call_comes_after_the_brain_exists_and_before_the_main_loop(self):
        src = inspect.getsource(M.amain)
        self.assertLess(src.index("brain.get_speech_context = _speech_context"), src.index("_proactive_start(brain, mouth"))
        self.assertLess(src.index("_proactive_start(brain, mouth"), src.index("typed_q: \"queue.Queue[str]\""))


class PresenceWiringTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(M.CFG)
        self.real_log, self.logs = M.log, []
        M.log = self.logs.append
        self.real_mon = M.presence_mod.PresenceMonitor
        M._PRESENCE["monitor"] = None

    def tearDown(self):
        M.log = self.real_log
        M.presence_mod.PresenceMonitor = self.real_mon
        M.CFG.clear()
        M.CFG.update(self.saved)
        stop = getattr(M._PRESENCE["monitor"], "stop", None)
        if stop:
            stop()
        M._PRESENCE["monitor"] = None

    def test_it_is_off_by_default_and_only_literal_true_starts_it(self):
        for v in (None, "true", 1, [], {"enabled": "yes"}):
            M.CFG["presence"] = v if isinstance(v, dict) else {"enabled": v}
            self.assertIsNone(M._presence_start(), v)
        M.CFG["presence"] = "garbage"
        self.assertIsNone(M._presence_start())
        self.assertIsNone(M._PRESENCE["monitor"])

    def test_enabled_starts_a_monitor_and_logs_one_enum_line(self):
        started = []

        class Fake:
            def __init__(self, **kw):
                self.kw = kw

            def refresh(self):
                pass

            def latest(self):
                return M.presence_mod.Presence(state="active", app_category="terminal", output="speakers", at=time.time())

            def start(self):
                started.append(1)

            def stop(self):
                pass
        M.presence_mod.PresenceMonitor = Fake
        M.CFG["presence"] = {"enabled": True}
        mon = M._presence_start()
        self.assertIsInstance(mon, Fake)
        self.assertEqual(started, [1])
        self.assertTrue(self.logs[0].startswith("[presence] on state=active app=terminal call=no output=speakers"))

    def test_a_failure_to_start_is_logged_and_never_raises(self):
        class Boom:
            def __init__(self, **kw):
                raise RuntimeError("no pyobjc")
        M.presence_mod.PresenceMonitor = Boom
        M.CFG["presence"] = {"enabled": True}
        self.assertIsNone(M._presence_start())
        self.assertTrue(any("could not start (RuntimeError)" in l for l in self.logs))

    def test_flags_come_from_a_fresh_reading_only(self):
        pr = M.presence_mod.Presence
        now = 1_000_000.0
        for reading, want in ((pr(state="active", app_category="call", likely_call=True, at=now - 1), (True, False)),
                              (pr(state="locked", at=now - 1), (False, True)),
                              (pr(state="away", at=now - 1), (False, True)),
                              (pr(state="idle", at=now - 1), (False, False)),
                              (pr(state="unknown", at=now - 1), (False, False)),
                              (pr(state="locked", likely_call=True, at=now - M.PRESENCE_TRUST_S - 1), (False, False))):   # too old
            M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self, r=reading: r})()
            self.assertEqual(M._presence_flags(now), want, reading)
        M._PRESENCE["monitor"] = None
        self.assertEqual(M._presence_flags(now), (False, False))
        M._PRESENCE["monitor"] = object()                                          # a broken monitor: no raise
        self.assertEqual(M._presence_flags(now), (False, False))

    def test_the_context_reads_presence_and_the_monitor_starts_before_the_runner(self):
        src = inspect.getsource(M.amain)
        body = src[src.index("def _proactive_context"):src.index("_proactive_start(brain, mouth, _proactive_context)")]
        self.assertIn("on_call=_presence_flags()[0]", body)
        self.assertIn("user_away=_presence_flags()[1]", body)
        self.assertLess(src.index("    _presence_start()"), src.index("    _proactive_start(brain, mouth, _proactive_context)"))

    def test_camera_state_reaches_the_context_and_the_desktop_string_only_from_a_fresh_reading(self):
        pr = M.presence_mod.Presence
        now = 1_000_000.0
        fresh = lambda **kw: type("F", (), {"latest": lambda self: pr(at=now - 1, **kw)})()
        M._PRESENCE["monitor"] = fresh(state="active", app_category="browser", camera_in_use=True)
        self.assertTrue(M._presence_camera(now))
        self.assertEqual(M._presence_desktop(now), "active/browser/cam")
        M._PRESENCE["monitor"] = fresh(state="active", app_category="call", likely_call=True, camera_in_use=True)
        self.assertEqual(M._presence_desktop(now), "active/call/call/cam")
        for cam in (False, None):
            M._PRESENCE["monitor"] = fresh(state="active", camera_in_use=cam)
            self.assertFalse(M._presence_camera(now))
        M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self: pr(at=now - M.PRESENCE_TRUST_S - 1, camera_in_use=True)})()
        self.assertFalse(M._presence_camera(now))                               # too old: unknown is not evidence
        M._PRESENCE["monitor"] = None
        self.assertFalse(M._presence_camera(now))
        M._PRESENCE["monitor"] = object()
        self.assertFalse(M._presence_camera(now))
        self.assertIn("camera_busy=_presence_camera()", inspect.getsource(M.amain))

    def test_fullscreen_reaches_the_context_and_the_desktop_string_only_from_a_fresh_reading(self):
        pr = M.presence_mod.Presence
        now = 1_000_000.0
        fresh = lambda **kw: type("F", (), {"latest": lambda self: pr(at=now - 1, **kw)})()
        M._PRESENCE["monitor"] = fresh(state="active", app_category="media", fullscreen=True, camera_in_use=True)
        self.assertTrue(M._presence_fullscreen(now))
        self.assertEqual(M._presence_desktop(now), "active/media/cam/full")
        for v in (False, None):
            M._PRESENCE["monitor"] = fresh(state="active", fullscreen=v)
            self.assertFalse(M._presence_fullscreen(now))
        M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self: pr(at=now - M.PRESENCE_TRUST_S - 1, fullscreen=True)})()
        self.assertFalse(M._presence_fullscreen(now))
        M._PRESENCE["monitor"] = None
        self.assertFalse(M._presence_fullscreen(now))
        self.assertIn("screen_busy=_presence_fullscreen()", inspect.getsource(M.amain))

    def test_the_situation_desktop_slot_is_one_enum_string_from_a_fresh_reading(self):
        pr = M.presence_mod.Presence
        now = 1_000_000.0
        M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self: pr(state="active", app_category="call", likely_call=True, at=now - 1)})()
        self.assertEqual(M._presence_desktop(now), "active/call/call")
        M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self: pr(state="idle", app_category="editor", at=now - 1)})()
        self.assertEqual(M._presence_desktop(now), "idle/editor")
        M._PRESENCE["monitor"] = type("F", (), {"latest": lambda self: pr(state="idle", at=now - M.PRESENCE_TRUST_S - 1)})()
        self.assertIsNone(M._presence_desktop(now))
        M._PRESENCE["monitor"] = None
        self.assertIsNone(M._presence_desktop(now))
        M._PRESENCE["monitor"] = object()
        self.assertIsNone(M._presence_desktop(now))
        self.assertIn("desktop=_presence_desktop", inspect.getsource(M.amain))

    def test_presence_reaches_nothing_the_model_sees(self):
        for fn in (M._presence_start, M._presence_flags):
            self.assertNotIn("model_input", inspect.getsource(fn))
        self.assertNotIn("presence", inspect.getsource(M._mem2_compose))
        self.assertNotIn("presence", inspect.getsource(M._sitctx_compose))


class SpawnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.real_log, self.logs = M.log, []
        M.log = self.logs.append
        M._PROACTIVE.update(runner=None, restarts=0)

    async def asyncTearDown(self):
        M.log = self.real_log
        M._PROACTIVE.update(runner=None, restarts=0)

    async def test_the_loop_runs_and_a_death_is_restarted_exactly_once(self):
        calls = []

        class Boom:
            async def run(self, sleep):
                calls.append(1)
                raise RuntimeError("dead")
        holder = []
        M._proactive_spawn(Boom(), holder)
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertEqual(len(calls), 2)                                 # the first run and one restart, never a third
        self.assertEqual(len(holder), 2)
        self.assertEqual(sum("loop stopped (RuntimeError)" in l for l in self.logs), 2)

    async def test_a_cancelled_loop_is_not_restarted(self):
        class Forever:
            async def run(self, sleep):
                await asyncio.sleep(3600)
        holder = []
        t = M._proactive_spawn(Forever(), holder)
        await asyncio.sleep(0)
        t.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(len(holder), 1)


class BrainTests(unittest.TestCase):
    def test_the_model_is_only_told_about_reminders_and_denied_self_scheduling_when_proactive_is_on(self):
        saved = dict(B.CFG)
        try:
            B.CFG.pop("proactive", None)
            self.assertFalse(B.proactive_enabled())
            B.CFG["proactive"] = {"enabled": True}
            self.assertTrue(B.proactive_enabled())
            B.CFG["proactive"] = {"enabled": 1}
            self.assertFalse(B.proactive_enabled())
        finally:
            B.CFG.clear()
            B.CFG.update(saved)

    def test_the_rule_names_the_cli_and_says_queued_and_the_denied_tools_are_the_self_schedulers(self):
        self.assertIn("backtalk.remind add", B.REMINDER_RULE)
        self.assertIn("QUEUED", B.REMINDER_RULE)
        self.assertIn("never schedule a wake-up", B.REMINDER_RULE)
        self.assertEqual(set(B.SELF_SCHEDULING_TOOLS), {"ScheduleWakeup", "CronCreate", "CronDelete", "CronList"})

    def test_the_options_use_the_flag_for_both_the_prompt_and_the_denied_tools(self):
        src = inspect.getsource(B.WarmBrain.start)
        self.assertIn("proactive_on = proactive_enabled()", src)
        self.assertIn("REMINDER_RULE", src)
        self.assertIn("disallowed_tools=list(SELF_SCHEDULING_TOOLS) if proactive_on else []", src)

    def test_the_mouth_exposes_its_flush_counter(self):
        self.assertIsInstance(MOUTH.Mouth.generation, property)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "scheduler")
        self.real = R.sched_dir
        R.sched_dir = lambda: self.dir

    def tearDown(self):
        R.sched_dir = self.real
        self.tmp.cleanup()

    def run_cli(self, *argv, now=T0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = R.main(list(argv), now=now)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def requests(self):
        d = os.path.join(self.dir, "inbox")
        return sorted(n for n in os.listdir(d)) if os.path.isdir(d) else []

    def test_parse_helpers(self):
        self.assertEqual([R.parse_duration(x) for x in ("90s", "10m", "2h", "1d", " 5 M ")], [90, 600, 7200, 86400, 300])
        for bad in ("", "0m", "10", "m", "-5m", "1.5h", "10 minutes"):
            with self.assertRaises(ValueError):
                R.parse_duration(bad)

    def test_at_picks_the_next_occurrence_today_or_tomorrow(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        z = ZoneInfo("America/Chicago")
        now = datetime(2026, 10, 20, 12, 0, tzinfo=z).timestamp()
        self.assertEqual(R.next_local("14:30", now), datetime(2026, 10, 20, 14, 30, tzinfo=z).timestamp())
        self.assertEqual(R.next_local("09:00", now), datetime(2026, 10, 21, 9, 0, tzinfo=z).timestamp())
        with self.assertRaises(ValueError):
            R.next_local("25:00", now)

    def test_add_writes_one_atomic_private_request_and_says_queued_not_scheduled(self):
        code, out, err = self.run_cli("add", "--in", "10m", "call", "the", "title company")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("queued", out)
        self.assertNotIn("scheduled for", out)
        (name,) = self.requests()
        self.assertTrue(name.endswith(".json") and not name.startswith(".tmp"))
        path = os.path.join(self.dir, "inbox", name)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)
        req = load_json(path)
        self.assertEqual((req["kind"], req["title"], req["severity"], req["due_at"]), ("once", "call the title company", "normal", T0 + 600))

    def test_every_form_and_urgent(self):
        for argv, kind in ((("add", "--daily", "08:30", "brief"), "daily"), (("add", "--every", "2h", "stretch"), "every"),
                           (("add", "--at", "14:30", "offer"), "once"), (("add", "--in", "5m", "--urgent", "oven"), "once")):
            self.assertEqual(self.run_cli(*argv)[0], 0, argv)
        kinds = [load_json(os.path.join(self.dir, "inbox", n))["kind"] for n in self.requests()]
        self.assertEqual(sorted(kinds), sorted(["daily", "every", "once", "once"]))
        urgent = [load_json(os.path.join(self.dir, "inbox", n))["severity"] for n in self.requests()]
        self.assertEqual(urgent.count("urgent"), 1)

    def test_bad_arguments_exit_two_and_write_nothing(self):
        for argv in (("add", "--in", "10m"), ("add", "hello"), ("add", "--in", "10m", "--at", "14:30", "x"),
                     ("add", "--in", "ten", "x"), ("add", "--daily", "8:30", "x"), ("cancel", "nope")):
            code, out, err = self.run_cli(*argv)
            self.assertEqual(code, 2, argv)
        self.assertEqual(self.requests(), [])

    def test_a_long_title_is_bounded(self):
        self.run_cli("add", "--in", "1m", "x" * 900)
        (name,) = self.requests()
        self.assertEqual(len(load_json(os.path.join(self.dir, "inbox", name))["title"]), R.TITLE_MAX)

    def test_the_cli_request_is_exactly_what_the_runner_ingests_and_list_confirms_it(self):
        self.run_cli("add", "--in", "10m", "call the title company")
        code, out, _ = self.run_cli("list")
        self.assertIn("1 request(s) queued", out)
        sched = S.Scheduler(self.dir, clock=lambda: T0, fsync=False)
        try:
            logs = []
            r = PL.Runner(engine=P.Engine(P.Policy(enabled=True)), scheduler=sched, speech=SP.SpeechPolicy(None),
                          ledger=L.Ledger(os.path.join(self.tmp.name, "ledger"), fsync=False),
                          say=lambda t: None, mouth_state=lambda: (False, 0, []), context=lambda: P.Context(),
                          log=logs.append, clock=lambda: T0, monotonic=lambda: T0, sched_dir=self.dir)
            r.ingest_inbox(T0)
            (t,) = sched.tasks()
            self.assertEqual((t.title, t.kind, t.next), ("call the title company", "once", T0 + 600))
            self.assertEqual(self.requests(), ["done"])
        finally:
            sched.close()
        code, out, _ = self.run_cli("list")
        self.assertIn("call the title company", out)
        self.assertNotIn("request(s) queued", out)
        tid = t.id
        self.assertEqual(self.run_cli("cancel", tid)[0], 0)
        sched = S.Scheduler(self.dir, clock=lambda: T0, fsync=False)
        try:
            r2 = PL.Runner(engine=P.Engine(P.Policy(enabled=True)), scheduler=sched, speech=SP.SpeechPolicy(None),
                           ledger=L.Ledger(os.path.join(self.tmp.name, "ledger2"), fsync=False), say=lambda t: None,
                           mouth_state=lambda: (False, 0, []), context=lambda: P.Context(), log=logs.append,
                           clock=lambda: T0, monotonic=lambda: T0, sched_dir=self.dir)
            r2.ingest_inbox(T0)
            self.assertEqual(sched.get(tid).status, "cancelled")
        finally:
            sched.close()

    def test_list_works_while_the_voice_line_holds_the_scheduler_lock(self):
        self.run_cli("add", "--in", "10m", "x")
        sched = S.Scheduler(self.dir, clock=lambda: T0, fsync=False)           # the voice line
        try:
            sched.add_once("held task", T0 + 60)
            code, out, err = self.run_cli("list")
            self.assertEqual(code, 0)
            self.assertIn("held task", out)
        finally:
            sched.close()

    def test_nothing_is_scheduled_message(self):
        self.assertEqual(self.run_cli("list")[1].strip(), "nothing is scheduled")


if __name__ == "__main__":
    unittest.main()


class VoicePreloadTests(unittest.TestCase):
    def test_the_voice_loads_before_the_other_engines_compete_for_the_cpu(self):
        src = inspect.getsource(M.amain)
        self.assertIn("threading.Thread(target=_voice_preload", src)
        pre = src.index("threading.Thread(target=_voice_preload")
        for later in ("loop.run_in_executor(None, warm_ears)", "loop.run_in_executor(None, vaultrecall.warm)",
                      "brain = WarmBrain(", "await asyncio.wait_for(brain.start(), 120)"):
            self.assertLess(pre, src.index(later), later)

    def test_a_preload_failure_is_logged_not_raised(self):
        body = inspect.getsource(M.amain)
        i = body.index("def _voice_preload")
        seg = body[i:i + 400]
        self.assertIn("except Exception", seg)
