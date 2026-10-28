"""F-01: a session-control confirmation ("Cleared.", "Compacted.", "Deep model online.") is spoken only for a
command that actually succeeded. The real brain, router and ledger run against the fake client; the success
line comes from main.command_confirmation, the same function the console's tail speaks."""
import ast
import asyncio
import inspect
from unittest import mock

from backtalk import main as M
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_main_router import Base

CLEARED = "Cleared. Fresh slate."
COMPACTED = "Compacted. Same conversation, smaller footprint."
DEEP = "Deep model online, for this session only."


def answer(text, *, sid=None):
    """The CLI's reply to a slash command, as captured: assistant text, then the echo, then the result."""
    def script(c, m):
        with mock.patch.object(F, "SID", sid or F.SID):
            c.feed(F.init(), F.asst(text), F.echo(m["uuid"], "<command-name>/x"), F.result(text, num_turns=0))
    return script


class ConsoleConfirmationTests(Base):
    async def confirm(self, verb, cmd, say_after, script=None):
        if script:
            self.script = script
        before = M.command_snapshot(self.brain)
        resp = await self.brain.command(cmd)
        return resp, M.command_confirmation(self.brain, before, verb, resp, say_after)

    # ---- the happy paths still confirm ----
    async def test_a_clear_that_started_a_new_session_is_confirmed(self):
        _, line = await self.confirm("clear", "/clear", CLEARED, answer("", sid="sess-2"))
        self.assertEqual(line, CLEARED)

    async def test_a_compact_that_finished_is_confirmed(self):
        _, line = await self.confirm("compact", "/compact", COMPACTED, answer("Compacted"))
        self.assertEqual(line, COMPACTED)

    async def test_the_models_own_set_reply_confirms_a_model_switch_and_an_effort_change(self):
        _, line = await self.confirm("deep", "/model x", DEEP, answer("Set model to `Opus 5` for this session only"))
        self.assertEqual(line, DEEP)
        _, line = await self.confirm("effort:low", "/effort low", "Effort set.", answer("Set effort level to low (this session only)"))
        self.assertEqual(line, "Effort set.")

    # ---- and every non-success is silent about success ----
    async def test_a_command_aborted_by_the_epoch_closing_is_not_confirmed(self):
        self.script = lambda c, m: None                                   # the CLI never answers
        before = M.command_snapshot(self.brain)
        task = asyncio.create_task(self.brain.command("/compact"))
        await settle()
        await self.brain.router.close("test")
        resp = await task
        self.assertEqual(resp, "")                                        # what the old code read as "no error"
        self.assertEqual(M.command_confirmation(self.brain, before, "compact", resp, COMPACTED), M.COMMAND_FAILED_LINE)

    async def test_an_interrupted_command_is_not_confirmed(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"], "<command-name>/compact"))   # claimed, never finishes
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        before = M.command_snapshot(self.brain)
        task = asyncio.create_task(self.brain.command("/compact"))
        await settle()
        await self.brain.router.interrupt(reason="TEST", force=True, wait=2.0)
        resp = await task
        self.assertEqual(resp, "")
        self.assertEqual(M.command_confirmation(self.brain, before, "compact", resp, COMPACTED), M.COMMAND_FAILED_LINE)

    async def test_an_empty_reply_is_not_success_for_a_model_switch_or_an_effort_change(self):
        for verb, cmd in (("deep", "/model x"), ("fast", "/model y"), ("effort:high", "/effort high")):
            resp, line = await self.confirm(verb, cmd, "OK", answer(""))
            self.assertEqual((resp, line), ("", M.COMMAND_FAILED_LINE), verb)

    async def test_a_reply_that_is_not_the_cli_set_wording_is_not_success(self):
        _, line = await self.confirm("deep", "/model nope", DEEP, answer("Unknown model nope"))   # no "error" in it
        self.assertEqual(line, M.COMMAND_FAILED_LINE)

    async def test_a_clear_that_left_the_session_unchanged_is_not_confirmed(self):
        await self.reply("hello")                                          # the session id is now known
        _, line = await self.confirm("clear", "/clear", CLEARED, answer(""))
        self.assertEqual(line, M.COMMAND_FAILED_LINE)

    async def test_no_command_turn_at_all_is_not_success(self):
        before = M.command_snapshot(self.brain)
        self.assertEqual(M.command_confirmation(self.brain, before, "compact", "", COMPACTED), M.COMMAND_FAILED_LINE)

    async def test_a_command_that_was_already_done_before_does_not_count(self):
        await self.confirm("compact", "/compact", COMPACTED, answer("Compacted"))
        before = M.command_snapshot(self.brain)                           # an older DONE command exists; none is new
        self.assertEqual(M.command_confirmation(self.brain, before, "compact", "", COMPACTED), M.COMMAND_FAILED_LINE)

    # ---- existing error messages still work ----
    async def test_a_failed_turn_still_speaks_the_cli_error_not_a_success(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"], "<command-name>/compact"),
                                          F.result("", is_error=True, subtype="error_during_execution", terminal="x"))
        resp, line = await self.confirm("compact", "/compact", COMPACTED)
        self.assertTrue(resp.startswith("error:"))
        self.assertEqual(line, resp[:160])

    async def test_the_cli_error_text_still_outranks_the_success_line(self):
        _, line = await self.confirm("deep", "/model q", DEEP, answer("Invalid model: q"))
        self.assertEqual(line, "Invalid model: q")
        self.assertEqual(M.command_confirmation(self.brain, M.command_snapshot(self.brain), "compact",
                                                "error: the command timed out", COMPACTED),
                         "error: the command timed out")


class ConsoleWiringTests(Base):
    """The console's own tail must use the judgement, once, and the effort default must not be saved for a
    command that failed."""

    def console(self):
        src = inspect.getsource(M)
        tree = ast.parse(src)
        amain = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "amain")
        fn = next(n for n in ast.walk(amain) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_console_inner")
        return ast.get_source_segment(src, fn)

    async def test_the_tail_speaks_exactly_one_line_chosen_by_command_confirmation(self):
        body = self.console()
        self.assertEqual(body.count("command_confirmation("), 1)
        self.assertEqual(body.count("command_snapshot("), 1)
        self.assertNotIn("mouth.say(say_after)", body)
        self.assertEqual(body.count("mouth.say(line)"), 1)
        self.assertLess(body.index("command_snapshot("), body.index('brain.command("/clear")'))

    async def test_the_failure_line_is_real_speech_and_claims_nothing_done(self):
        low = M.COMMAND_FAILED_LINE.lower()
        self.assertGreater(len(M.COMMAND_FAILED_LINE.split()), 3)
        for word in ("cleared", "compacted", "online", "done", "set to"):
            self.assertNotIn(word, low)
        self.assertNotIn(M.COMMAND_FAILED_LINE, (CLEARED, COMPACTED, DEEP))

    async def test_the_effort_default_is_only_saved_after_the_command_succeeded(self):
        body = self.console()
        i = body.index('_write_config_key("effort"')
        self.assertIn("command_succeeded(brain, before, verb, resp)", body[i - 150:i])
