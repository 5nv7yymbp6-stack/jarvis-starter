"""Phase 5: WarmBrain on the router. The SDK client is replaced by a fake; the
brain, router, ledger, verifier and speech policy are the real ones."""
import asyncio
import tempfile
import unittest
from unittest import mock

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import router as R
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.clients = []
        self.script = lambda c, msg: c.feed(*F.fg_run(msg["uuid"], "Hello there. How are you?"))

        def factory(options=None):
            c = FakeClient()
            c.options = options
            c.on_query = lambda msg, c=c: self.script(c, msg)
            self.clients.append(c)
            return c
        self.p1 = mock.patch.object(B, "ClaudeSDKClient", factory)
        self.p2 = mock.patch.object(B, "build_session_brief", lambda: "")
        self.p1.start(); self.p2.start()
        self.brain = B.WarmBrain(model="m", ledger=self.led)
        await self.brain.start()

    async def asyncTearDown(self):
        self.p1.stop(); self.p2.stop()
        r = self.brain.router
        if r:
            for t in (r._pump_task, r._watch_task, *r._escalations):
                if t:
                    t.cancel()
        await asyncio.sleep(0)
        self.led.close()
        self.tmp.cleanup()

    async def ask(self, text="hi", **kw):
        return [s async for s in self.brain.ask_stream(text, **kw)]

    @property
    def client(self):
        return self.clients[-1]


class StreamingTests(Base):
    async def test_ask_stream_yields_sentences_from_its_own_turn(self):
        self.assertEqual(await self.ask(), ["Hello there.", "How are you?"])
        self.assertEqual(await self.ask("again"), ["Hello there.", "How are you?"])

    async def test_sentence_splitting_is_unchanged(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.delta("Hello there. How "), F.delta("are you? Fine"),
                                          F.asst("x"), F.result("x"))
        self.assertEqual(await self.ask(), ["Hello there.", "How are you?", "Fine"])

    async def test_client_is_asked_to_echo_user_messages(self):
        self.assertEqual(self.clients[0].options.extra_args, {"replay-user-messages": None})

    async def test_query_carries_our_uuid_and_session_state_is_kept(self):
        await self.ask("q")
        sent = self.client.sent[-1]
        self.assertEqual(sent["type"], "user")
        self.assertTrue(sent["uuid"])
        self.assertEqual(self.brain.session_id, F.SID)
        self.assertEqual(self.brain.session["turns"], 1)

    async def test_exactly_one_stream_reader_for_the_session(self):
        for _ in range(3):
            await self.ask()
        self.assertEqual(self.client.reader_calls, 1)
        self.assertEqual(self.client.max_readers, 1)

    async def test_on_tool_still_fires(self):
        names = []
        self.script = lambda c, m: c.feed(
            F.init(), F.echo(m["uuid"]),
            F.StreamEvent(uuid="s", session_id="x", event={
                "type": "content_block_start", "content_block": {"type": "tool_use", "name": "Bash"}}),
            F.delta("ok."), F.asst("ok."), F.result("ok."))
        await self.ask(on_tool=names.append)
        self.assertEqual(names, ["Bash"])

    async def test_background_completion_before_the_next_question_cannot_be_its_answer(self):
        h0 = None

        def script(c, m):
            c.feed(*F.fg_run(m["uuid"], "Paris." if "France" in m["message"]["content"] else "Rome."))
        self.script = script
        # a background task finishes while nobody has a turn open
        self.client.feed(F.task_started("tk", "tu", "job", task_type="local_agent"),
                         F.task_note("tk", "tu", "completed", "GAMMA=1234"),
                         *F.unsolicited_run("GAMMA=1234"))
        await settle()
        self.assertEqual(await self.ask("France?"), ["Paris."])
        self.assertEqual(await self.ask("Italy?"), ["Rome."])

    async def test_failed_turn_raises_with_a_reason(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.result("", subtype="error_max_turns", is_error=True, terminal="max_turns"))
        with self.assertRaises(R.TurnFailed) as cm:
            await self.ask()
        self.assertEqual(cm.exception.reason, "max_turns")


class ContaminationTests(Base):
    async def test_a_folded_task_notice_holds_the_rest_of_the_answer(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), F.delta("First part. "), F.asst("First part."))
        self.script = script
        gen = self.brain.ask_stream("q")
        got = []

        async def consume():
            async for s in gen:
                got.append(s)
        t = asyncio.create_task(consume())
        await settle(10)
        self.assertEqual(got, ["First part."])                  # released before the notice
        self.client.feed(F.echo("inj", "<task-notification><task-id>tk9</task-id></task-notification>"),
                         F.delta("The background job was stopped. "), F.asst("x"),
                         F.delta("Second part."), F.asst("y"), F.result("done"))
        await settle(10)
        await asyncio.wait_for(t, 2)
        self.assertEqual(got, ["First part.", "The background job was stopped.", "Second part."])
        self.assertTrue(any(n.get("note") == "held_answer_released" for n in self.led.notes))


class MergedTurnHoldTests(Base):
    """The same audit finding at the speech boundary: a notice that arrived
    BEFORE the turn's own echo must hold the answer exactly like one that
    arrives in the middle of it (ContaminationTests above)."""

    async def test_an_answer_in_a_run_that_began_with_a_task_notice_is_held_to_the_end(self):
        def script(c, m):
            c.feed(F.init(), F.echo("inj", "<task-notification><task-id>tk7</task-id></task-notification>"),
                   F.echo(m["uuid"]), F.delta("The job found twenty. "), F.asst("x"))
        self.script = script
        gen = self.brain.ask_stream("what did it find")
        got = []

        async def consume():
            async for s in gen:
                got.append(s)
        t = asyncio.create_task(consume())
        await settle(12)
        self.assertEqual(got, [])                               # NOT streamed to the mouth
        self.client.feed(F.delta("And more."), F.asst("y"), F.result("done"))
        await settle(12)
        await asyncio.wait_for(t, 2)
        self.assertEqual(got, ["The job found twenty.", "And more."])     # released together, at the end
        released = [n for n in self.led.notes if n.get("note") == "held_answer_released"]
        self.assertEqual(released[0]["task_ids"], ["tk7"])

    async def test_a_clean_turn_still_streams_sentence_by_sentence(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), F.delta("First part. "), F.asst("First part."))
        self.script = script
        gen = self.brain.ask_stream("q")
        got = []

        async def consume():
            async for s in gen:
                got.append(s)
        t = asyncio.create_task(consume())
        await settle(12)
        self.assertEqual(got, ["First part."])                  # released immediately
        self.client.feed(F.delta("Second part."), F.asst("y"), F.result("done"))
        await settle(12)
        await asyncio.wait_for(t, 2)
        self.assertFalse([n for n in self.led.notes if n.get("note") == "held_answer_released"])


class InterruptAndRebuildTests(Base):
    async def test_interrupt_ends_the_turn_and_the_next_query_is_clean(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        gen = self.brain.ask_stream("long")
        t = asyncio.create_task(gen.__anext__())
        await settle(10)
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        self.assertEqual(await self.brain.interrupt(), "aborted")
        await self.brain.reset_turn()
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Fresh."))
        self.assertEqual(await self.ask("next"), ["Fresh."])

    async def test_lost_stream_rebuilds_the_same_conversation_before_the_next_turn(self):
        await self.ask("one")
        first_epoch = self.brain.router.epoch
        self.client.feed(ValueError("JSON message exceeded maximum buffer size of 1048576 bytes"))
        await settle()
        self.assertEqual(self.brain._need_rebuild, "oversize")
        await self.brain.reset_turn()
        self.assertEqual(len(self.clients), 2)
        self.assertEqual(self.clients[1].options.resume, F.SID)      # same conversation
        self.assertNotEqual(self.brain.router.epoch, first_epoch)
        self.assertEqual(self.led.epochs[first_epoch]["state"], L.EPOCH_CLOSED_CLEAN)
        self.assertEqual(await self.ask("two"), ["Hello there.", "How are you?"])
        self.assertEqual(self.clients[1].reader_calls, 1)

    async def test_ask_after_a_lost_stream_rebuilds_instead_of_failing(self):
        await self.ask("one")
        self.client.eof()
        await settle()
        self.assertEqual(await self.ask("two"), ["Hello there.", "How are you?"])
        self.assertEqual(len(self.clients), 2)

    async def test_stop_closes_the_epoch_cleanly(self):
        ep = self.brain.router.epoch
        await self.brain.stop()
        self.assertEqual(self.led.epochs[ep]["state"], L.EPOCH_CLOSED_CLEAN)
        self.assertTrue(self.clients[0].disconnected)


class CommandTests(Base):
    async def test_command_returns_the_cli_reply_text(self):
        self.script = lambda c, m: c.feed(F.init(), F.asst("Set effort level to low"),
                                          F.echo(m["uuid"], "<command-name>/effort"),
                                          F.result("Set effort level to low", num_turns=0))
        self.assertEqual(await self.brain.command("/effort low"), "Set effort level to low")

    async def test_command_does_not_count_as_a_turn(self):
        self.script = lambda c, m: c.feed(F.init(), F.asst("ok"), F.echo(m["uuid"]), F.result("ok", num_turns=0))
        await self.brain.command("/model x")
        self.assertEqual(self.brain.session["turns"], 0)


class BackgroundControlTests(Base):
    async def start_task(self, tid="tk1"):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg(tid, "toolu_" + tid, "export"),
                   F.delta("Started."), F.asst("Started."), F.result("Started."))
        self.script = script
        await self.ask("run export in the background")

    async def test_stop_background_is_scoped(self):
        await self.start_task()
        self.assertEqual([t["id"] for t in self.brain.live_background()], ["tk1"])
        self.client.on_stop = lambda tid: self.client.feed(F.task_updated(tid, "killed"),
                                                           F.task_note(tid, "toolu_tk1", "stopped"))
        self.assertTrue(await self.brain.stop_background("tk1"))
        await settle()
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_STOPPED_USER)
        self.assertEqual(self.client.interrupts, 0)

    async def test_completion_goes_to_the_ledger_and_a_next_turn_note_not_the_mouth(self):
        spoken = []
        self.brain.on_say = spoken.append
        await self.start_task()
        self.client.feed(F.task_note("tk1", "toolu_tk1", "completed", "42 rows"),
                         *F.unsolicited_run("The export finished with 42 rows, sir"))
        await settle()
        self.assertEqual(spoken, [])                        # nothing said unprompted
        note = self.brain.note_prefix()
        self.assertIn("export finished", note)
        self.assertNotIn("sir", note)                       # the model's report text is not in it
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_announced_task_is_spoken_from_the_ledger_when_quiet(self):
        spoken = []
        self.brain.on_say = spoken.append
        self.brain.get_speech_context = lambda: __import__("backtalk.speech_policy", fromlist=["x"]).Context()
        await self.start_task()
        self.led.task_set("tk1", announce=True)
        self.client.feed(F.task_note("tk1", "toolu_tk1", "completed", "42 rows"),
                         *F.unsolicited_run("model-authored report text"))
        await settle()
        self.assertEqual(len(spoken), 1)
        self.assertIn("export finished", spoken[0])
        self.assertNotIn("model-authored", spoken[0])

    async def test_stop_everything_sets_the_task_intent_and_interrupts(self):
        await self.start_task()
        # a second turn is running when the user says stop everything
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        gen = self.brain.ask_stream("another")
        t = asyncio.create_task(gen.__anext__())
        await settle(10)
        self.client.on_stop = lambda tid: self.client.feed(F.task_updated(tid, "killed"),
                                                           F.task_note(tid, "toolu_tk1", "stopped"))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        n, res = await self.brain.stop_everything()
        await settle()
        self.assertEqual((n, res), (1, "aborted"))
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_STOPPED_USER)


if __name__ == "__main__":
    unittest.main()


class ImmediateBadNewsTests(Base):
    """F-06: an action whose outcome is known the moment its result lands (error, denial, read-only,
    self-report) must reach the same bad-news path as one a verifier finished later, once."""
    BAD = "A consequential action failed."

    async def run_tool(self, tuid, tool, *, decision="ALLOW", error, inp=None, twice=False):
        F.seen(self.led, tuid, decision, epoch=self.led.current_epoch, tool=tool)
        res = [F.tool_result(tuid, "boom" if error else "ok", is_error=error)]
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Done.", tools=[F.tool_use(tuid, tool, inp or {})] + res * (2 if twice else 1)))
        await self.ask()
        await settle()

    async def test_a_tool_error_is_not_announced_as_a_consequential_failure(self):
        await self.run_tool("toolu_e", "Bash", error=True)
        self.assertNotIn(self.BAD, self.brain.note_prefix())
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_repeated_error_result_adds_no_note(self):
        await self.run_tool("toolu_d", "Bash", error=True, twice=True)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_denied_action_is_not_bad_news(self):
        await self.run_tool("toolu_n", "Bash", decision="DENY", error=True)
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_n")]["state"], L.A_DENIED)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_read_only_error_is_not_bad_news(self):
        await self.run_tool("toolu_r", "Read", error=True)
        await self.run_tool("toolu_l", "Bash", error=True)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_successful_unverified_consequential_action_is_not_bad_news(self):
        await self.run_tool("toolu_s", "Bash", error=False)
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_s")]["verification"]["state"], L.V_UNVERIFIED)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_verified_write_is_not_bad_news(self):
        with tempfile.TemporaryDirectory() as d:
            path = d + "/ok.txt"
            open(path, "w").write("hello")
            await self.run_tool("toolu_w", "Write", error=False,
                                inp={"file_path": path, "content": "hello"})
            await asyncio.sleep(0.2)
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_w")]["verification"]["state"], L.V_VERIFIED)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_failed_verification_is_still_reported_once(self):
        await self.run_tool("toolu_f", "Write", error=False,
                            inp={"file_path": "/nonexistent-dir-f06/x.txt", "content": "hello"})
        await asyncio.sleep(0.2)
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_f")]["verification"]["state"], L.V_FAILED)
        note = self.brain.note_prefix()
        self.assertEqual(note.count("my check says it did not land"), 1)
        self.assertEqual(self.brain.note_prefix(), "")

