"""Phase 6: main.py on the router: what reaches the mouth, the stop controls,
and structural guarantees about the stream and the mouth."""
import asyncio
import os
import re
import tempfile
import unittest
from unittest import mock

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import main as M
from backtalk import speech_policy as SP
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle

PKG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtalk")


class FakeMouth:
    def __init__(self):
        self.said, self.chunks = [], []
        self.busy = False
        self.speaking = False
        self.audible = False
        self.held = False
        self.shut = 0

    def say(self, t):
        self.said.append(t)

    def say_chunk(self, t, directions=None):
        self.chunks.append(t)

    def shut_up(self):
        self.shut += 1

    def release(self, resume=True):
        pass


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.clients = []
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there."))

        def factory(options=None):
            c = FakeClient()
            c.options = options
            c.on_query = lambda msg, c=c: self.script(c, msg)
            self.clients.append(c)
            return c
        self.patches = [mock.patch.object(B, "ClaudeSDKClient", factory),
                        mock.patch.object(B, "build_session_brief", lambda: ""),
                        mock.patch.object(M.signals, "set_state"),
                        mock.patch.object(M.signals, "static_stop"),
                        mock.patch.object(M.signals, "static_start")]
        for p in self.patches:
            p.start()
        self.brain = B.WarmBrain(model="m", ledger=self.led)
        self.mouth = FakeMouth()
        self.brain.on_say = self.mouth.say
        await self.brain.start()

    async def asyncTearDown(self):
        for p in self.patches:
            p.stop()
        r = self.brain.router
        if r:
            for t in (r._pump_task, r._watch_task, *r._escalations):
                if t:
                    t.cancel()
        await asyncio.sleep(0)
        self.led.close()
        self.tmp.cleanup()

    @property
    def client(self):
        return self.clients[-1]

    async def reply(self, text="hi"):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        await M.speak_reply(self.brain, self.mouth, text, vault_fut=fut)


class SpeakReplyTests(Base):
    async def test_a_normal_answer_reaches_the_mouth_sentence_by_sentence(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("One. Two."), F.asst("x"), F.result("x"))
        await self.reply()
        self.assertEqual(self.mouth.chunks, ["One.", "Two."])

    async def test_20_background_report_never_reaches_the_mouth_directly(self):
        # a background task finishes and its model-authored report run arrives
        # while nothing is being asked
        self.client.feed(F.task_started("tk", "tu", "export", task_type="local_agent"),
                         F.task_note("tk", "tu", "completed", "42 rows"),
                         *F.unsolicited_run("Sir, your export finished with 42 rows"))
        await settle()
        self.assertEqual(self.mouth.said, [])
        self.assertEqual(self.mouth.chunks, [])
        await self.reply()                                  # and the next answer is its own
        self.assertEqual(self.mouth.chunks, ["Hello there."])
        self.assertNotIn("42 rows", " ".join(self.mouth.chunks))

    async def test_20_synthetic_lost_task_report_after_a_rebuild_is_silent(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk1", "toolu_1", "export"),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.script = script
        await self.reply("run it")
        self.mouth.said.clear(); self.mouth.chunks.clear()
        await self.brain.rebuild("desync")
        self.client.feed(F.task_note("tk1", "toolu_1", "stopped", "No completion record was found"),
                         *F.synthetic_run())
        await settle()
        self.assertEqual((self.mouth.said, self.mouth.chunks), ([], []))
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_LOST_REBUILD)
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Back."))
        await self.reply("you there?")
        self.assertEqual(self.mouth.chunks, ["Back."])

    async def test_oversize_says_which_thing_broke(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.tool_use("tb", "Read", {}),
                                          ValueError("JSON message exceeded maximum buffer size of 1048576 bytes"))
        await self.reply()
        self.assertEqual(self.mouth.said, [SP.failure_line("oversize")])
        self.assertIn("too big", self.mouth.said[0])
        self.assertEqual(self.led.actions[self.led.action_id_for("tb")]["state"], L.A_LOST)

    async def test_failed_step_limit_is_named(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.result("", subtype="error_max_turns", is_error=True, terminal="max_turns"))
        await self.reply()
        self.assertEqual(self.mouth.said, [SP.failure_line("max_turns")])

    async def test_a_turn_after_a_lost_stream_recovers_by_itself(self):
        await self.reply("one")
        self.client.eof()
        await settle()
        self.mouth.chunks.clear()
        await self.brain.reset_turn()
        await self.reply("two")
        self.assertEqual(self.mouth.chunks, ["Hello there."])


class StopControlTests(Base):
    async def launch(self, tid="tk1", desc="export"):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg(tid, "toolu_" + tid, desc),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.script = script
        await self.reply("go")
        self.mouth.said.clear()
        self.mouth.chunks.clear()

    def wire_kill(self):
        self.client.on_stop = lambda tid: self.client.feed(
            F.task_updated(tid, "killed"), F.task_note(tid, "toolu_" + tid, "stopped"))

    async def test_no_background_tasks(self):
        await M.do_stop_background(self.brain, self.mouth)
        self.assertEqual(self.mouth.said, ["There are no background tasks running."])

    async def test_stop_one_is_scoped_and_confirmed_only_after_the_ledger_sees_it(self):
        await self.launch()
        self.wire_kill()
        await M.do_stop_background(self.brain, self.mouth)
        self.assertEqual(self.client.interrupts, 0)
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_STOPPED_USER)
        self.assertEqual(self.mouth.said, ["Stopped export."])

    async def test_stop_one_without_a_stop_notice_does_not_claim_success(self):
        await self.launch()
        real_sleep = asyncio.sleep
        with mock.patch.object(M.asyncio, "sleep", new=lambda s: real_sleep(0)):
            await M.do_stop_background(self.brain, self.mouth)
        self.assertEqual(self.mouth.said, ["I asked it to stop, but I haven't seen it stop yet."])
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)

    async def test_several_tasks_ask_instead_of_guessing(self):
        await self.launch("a", "alpha job")
        await self.launch("b", "beta job")
        await M.do_stop_background(self.brain, self.mouth)
        self.assertEqual(self.client.stops, [])
        self.assertIn("2 background tasks", self.mouth.said[0])
        self.assertIn("stop everything", self.mouth.said[0])

    async def test_stop_everything_stops_tasks_and_the_run_and_speaks_once(self):
        await self.launch("a", "alpha job")
        await self.launch("b", "beta job")
        # a foreground turn is running
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        task = asyncio.create_task(self.reply("long one"))
        await settle(10)
        self.wire_kill()
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        await M.do_stop_everything(self.brain, self.mouth, task)
        await settle()
        self.assertEqual(self.mouth.said, ["Stopped 2 background tasks and the current request."])
        self.assertEqual(self.led.tasks["a"]["state"], L.K_STOPPED_USER)
        self.assertEqual(self.led.tasks["b"]["state"], L.K_STOPPED_USER)
        self.assertFalse(self.brain.force_interrupt)
        # and the next question is clean
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Ready."))
        self.mouth.chunks.clear()
        await self.brain.reset_turn()
        await self.reply("hello")
        self.assertEqual(self.mouth.chunks, ["Ready."])

    async def test_stop_everything_when_idle(self):
        await M.do_stop_everything(self.brain, self.mouth, None)
        self.assertEqual(self.mouth.said, ["Nothing is running."])


class BargeInTests(Base):
    async def test_barge_in_with_a_protected_task_does_not_kill_it(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk1", "toolu_1", "export"), F.delta("Working"))
        self.script = script
        task = asyncio.create_task(self.reply("go"))
        await settle(10)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.client.interrupts, 0)
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)


class ConsoleMatchTests(unittest.TestCase):
    def test_new_verbs_match_naturally(self):
        for t in ("stop the background task", "Jarvis, stop the background task",
                  "stop the background task please"):
            self.assertEqual(M.console_match(t), "stopbg", t)
        for t in ("stop everything", "Stop everything.", "Jarvis stop everything",
                  "stop everything you're doing"):
            self.assertEqual(M.console_match(t), "stopall", t)

    def test_they_do_not_fire_on_ordinary_speech(self):
        for t in ("stop", "stop the music", "don't stop everything", "I said stop everything earlier and it was fine",
                  "please stop talking", "background music is fine"):
            self.assertNotIn(M.console_match(t), ("stopbg", "stopall"), t)

    def test_negated_contractions_never_fire_any_verb(self):
        for t in ("don't go hands free", "don\u2019t go hands free", "don't stop everything",
                  "didn't say stop everything", "won't stop the background task"):
            self.assertIsNone(M.console_match(t), t)

    def test_existing_verbs_are_unchanged(self):
        self.assertEqual(M.console_match("go hands free"), "micopen")
        self.assertEqual(M.console_match("zip it"), "zip")
        self.assertEqual(M.console_match("clear the session"), "clear")


class StructureTests(unittest.TestCase):
    """No production path can bypass the router or send a background report
    to the mouth. These read the source."""

    def code(self, name):
        src = open(os.path.join(PKG, name)).read()
        return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())

    def strip_docstrings(self, s):
        return re.sub(r'""".*?"""', "", s, flags=re.S)

    def test_the_stream_has_exactly_one_reader(self):
        hits = []
        for f in sorted(os.listdir(PKG)):
            if f.endswith(".py"):
                body = self.strip_docstrings(self.code(f))
                for m in re.finditer(r"\.receive_messages\(|\.receive_response\(", body):
                    hits.append((f, m.group(0)))
        self.assertEqual(hits, [("router.py", ".receive_messages(")])

    def test_no_drain_or_dirty_flag_survives(self):
        for f in ("brain.py", "main.py"):
            body = self.strip_docstrings(self.code(f))
            self.assertNotIn("_drain", body, f)
            self.assertNotIn("self._dirty", body, f)
            self.assertNotIn("receive_response", body, f)

    def test_router_ledger_verify_never_touch_the_mouth(self):
        for f in ("router.py", "ledger.py", "verify.py", "speech_policy.py"):
            body = self.strip_docstrings(self.code(f))
            for bad in (r"\bmouth\b", r"say_chunk", r"\.say\("):
                self.assertIsNone(re.search(bad, body), (f, bad))

    def test_the_only_mouth_paths_in_brain_are_the_policy_ones(self):
        body = self.strip_docstrings(self.code("brain.py"))
        uses = re.findall(r"self\.on_say\((.*)\)\s*$", body, re.M)
        self.assertEqual(sorted(uses), sorted(["SP.task_line(row)", "line"]))

    def test_model_report_text_is_never_read_into_a_spoken_line(self):
        body = self.strip_docstrings(self.code("speech_policy.py"))
        self.assertNotIn(".reports", body)
        self.assertNotIn("report_text", body)
        self.assertNotIn("pending_reports", self.strip_docstrings(self.code("main.py")))


if __name__ == "__main__":
    unittest.main()


class LiveDefectRegressionTests(Base):
    """The exact 2026-09-30 21:05 case, end to end through speak_reply: an
    Agent launched with no run_in_background that the CLI nevertheless ran in
    the background."""

    def launch_script(self, c, m):
        c.feed(F.init(), F.echo(m["uuid"]),
               F.tool_use("toolu_live1", "Agent", {"description": "Count daily notes in vault",
                                                   "prompt": "count", "subagent_type": "general-purpose"}),
               F.task_started("a16ff5bdae", "toolu_live1", "Count daily notes in vault",
                              task_type="local_agent", is_backgrounded=True),
               F.tool_result("toolu_live1", "Async agent launched successfully."),
               F.delta("Counting now."), F.asst("Counting now."), F.result("Counting now."))

    async def test_task_is_a_bg_agent_and_its_result_is_never_spoken_directly(self):
        self.script = self.launch_script
        await self.reply("Launch a background agent to count my daily notes")
        self.assertEqual(self.mouth.chunks, ["Counting now."])
        self.assertEqual(self.led.tasks["a16ff5bdae"]["task_kind"], L.KIND_BG_AGENT)
        self.mouth.said.clear(); self.mouth.chunks.clear()
        # the subagent's own find returns 21 paths (B1: the router keeps that as
        # evidence; without it the summary's "21" would be an unverified model
        # number and, since Phase B, would not be repeated)
        paths = "\n".join(f"/v/01 - Daily Notes/09 - September 2026/2026-09-{d:02d}.md" for d in range(9, 30))
        self.client.feed(F.tool_use("toolu_live_find", "Bash", {"command": "find /v -type f -name '*.md' | sort"},
                                    parent="toolu_live1"),
                         F.tool_result("toolu_live_find", paths, parent="toolu_live1"))
        await settle()
        # completes after the turn, with the model's own report run
        self.client.feed(F.task_updated("a16ff5bdae", "completed"),
                         F.task_note("a16ff5bdae", "toolu_live1", "completed",
                                     "21 daily note files, dated 2026-09-09 to 2026-09-30."),
                         *F.unsolicited_run("Sir, twenty-one daily notes, September ninth through today."))
        await settle()
        self.assertEqual((self.mouth.said, self.mouth.chunks), ([], []))     # nothing spoken
        self.assertEqual(self.led.tasks["a16ff5bdae"]["state"], L.K_COMPLETED)
        # the outcome waits as a ledger-built note for the next turn
        note = self.brain.note_prefix()
        self.assertIn("Count daily notes in vault finished", note)
        self.assertIn("21 daily note files", note)
        self.assertNotIn("Sir", note)                                        # not the model's report text
        self.assertEqual(self.brain.note_prefix(), "")
        # and the foreground turns around it are clean
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Four."))
        await self.reply("what's two plus two")
        self.assertEqual(self.mouth.chunks, ["Four."])
        self.assertEqual(self.brain.mgr.quarantine.counts["unowned_result"], 0)

    async def test_stop_background_and_barge_in_protection_apply_to_it(self):
        self.script = self.launch_script
        await self.reply("go")
        self.assertEqual([t["id"] for t in self.brain.live_background()], ["a16ff5bdae"])
        self.client.on_stop = lambda tid: self.client.feed(F.task_updated(tid, "killed"),
                                                           F.task_note(tid, "toolu_live1", "stopped"))
        self.mouth.said.clear()
        await M.do_stop_background(self.brain, self.mouth)
        self.assertEqual(self.led.tasks["a16ff5bdae"]["state"], L.K_STOPPED_USER)
        self.assertEqual(self.mouth.said, ["Stopped Count daily notes in vault."])
        self.assertEqual(self.client.interrupts, 0)

    async def test_announced_completion_is_spoken_only_from_the_ledger(self):
        spoken = []
        self.brain.on_say = spoken.append
        self.brain.get_speech_context = lambda: SP.Context()
        self.script = self.launch_script
        await self.reply("go")
        self.led.task_set("a16ff5bdae", announce=True)
        self.client.feed(F.task_note("a16ff5bdae", "toolu_live1", "completed", "21 files"),
                         *F.unsolicited_run("model-authored words that must not be spoken"))
        await settle()
        self.assertEqual(len(spoken), 1)
        self.assertIn("finished", spoken[0])
        self.assertNotIn("model-authored", spoken[0])
