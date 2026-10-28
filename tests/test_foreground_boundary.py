"""B5: the foreground model trust boundary.

THE PROBLEM (live, 2026-10-01). A background agent's tool returned 21 files and
its own report said 20. Task SPEECH was already safe, but the CLI hands every
background agent's RAW final text to the foreground model (a <task-notification>
prompt with a <result> block), runs an unsolicited model turn on it, and keeps
that turn in the conversation. The foreground model then told the user the agent
"correctly reported 20".

THE BOUNDARY. brain.WarmBrain._gate_prompt (a UserPromptSubmit hook) withholds
that raw notice; the model learns about the task only through the ledger-built
note (speech_policy.take_notes), which carries the task's verification state and
the TOOL'S number, never the report's. These tests prove the properties at that
boundary rather than the implementation: what the CLI is sent, what the hook lets
through, and what the note says in each verification state."""
import asyncio
import tempfile
import unittest
from unittest import mock

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import speech_policy as SP
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle

NOTES_DIR = "/v/01 - Daily Notes/09 - September 2026/"
FIND_CMD = 'find "/v/01 - Daily Notes/" -type f -name "*.md" | sort'
FIND_OUTPUT = "\n".join(f"{NOTES_DIR}2026-09-{d:02d}.md" for d in range(9, 31) if d != 17)   # 21 paths
LAUNCH, CHILD, TID = "toolu_launch", "toolu_child", "bgtask1"
WRONG = "20 daily note files found"                      # the agent's report; the tool counted 21
HEAD = "[System note, from the task ledger, not from the user: "
NAME = "Background task Count daily notes in vault"


def raw_notice(result, tid=TID, tuid=LAUNCH):
    """The prompt the CLI really builds for a finished background agent (captured
    live 2026-10-01), with the agent's raw final text in <result>."""
    return (f"<task-notification>\n<task-id>{tid}</task-id>\n<tool-use-id>{tuid}</tool-use-id>\n"
            "<output-file>/tmp/x/tasks/x.output</output-file>\n<status>completed</status>\n"
            '<summary>Agent "Count daily notes in vault" finished</summary>\n'
            "<note>A task-notification fires each time this agent stops.</note>\n"
            f"<result>{result}</result>\n</task-notification>")


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.clients = []
        self.script = lambda c, msg: c.feed(*F.fg_run(msg["uuid"], "Noted."))

        def factory(options=None):
            c = FakeClient()
            c.options = options
            c.on_query = lambda msg, c=c: self.script(c, msg)
            self.clients.append(c)
            return c
        self.patches = [mock.patch.object(B, "ClaudeSDKClient", factory),
                        mock.patch.object(B, "build_session_brief", lambda: "")]
        for p in self.patches:
            p.start()
        self.brain = B.WarmBrain(model="m", ledger=self.led)
        self.brain.get_speech_context = lambda: SP.Context(fg_active=True)
        await self.brain.start()
        self.client = self.clients[-1]

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

    # ---- the background task, as the router sees it ---------------------------
    async def launch(self, tid=TID, tuid=LAUNCH):
        h = await self.brain.router.send("Launch a background agent to count my daily notes")
        self.client.feed(
            F.init(), F.echo(h.uuid),
            F.tool_use(tuid, "Agent", {"description": "Count daily notes in vault"}),
            F.task_started(tid, tuid, "Count daily notes in vault", task_type="local_agent",
                           is_backgrounded=True),
            F.tool_result(tuid, "Async agent launched successfully."),
            F.delta("Counting."), F.asst("Counting."), F.result("Counting."))
        await settle(12)

    async def child(self, cmd=FIND_CMD, output=FIND_OUTPUT, tuid=CHILD, parent=LAUNCH, hold=False):
        self.client.feed(F.tool_use(tuid, "Bash", {"command": cmd}, parent=parent))
        await settle(8)
        F.seen(self.led, tuid, "ALLOW", tool="Bash")
        if not hold:
            self.client.feed(F.tool_result(tuid, output, parent=parent))
            await settle(8)

    async def result_lands(self, output=FIND_OUTPUT, tuid=CHILD, parent=LAUNCH):
        self.client.feed(F.tool_result(tuid, output, parent=parent))
        await settle(8)

    async def finish(self, summary, tid=TID, tuid=LAUNCH, status="completed"):
        self.client.feed(F.task_updated(tid, status), F.task_note(tid, tuid, status, ""),
                         F.task_note(tid, tuid, status, summary), *F.unsolicited_run("report"))
        await settle(12)

    def verdict(self, tid=TID):
        return self.led.tasks[tid]["result_verification"]["state"]

    async def block(self, prompt):
        return await self.brain._gate_prompt({"prompt": prompt}, None, None)


class GateTests(Base):
    async def test_the_gate_is_registered_as_a_userpromptsubmit_hook(self):
        hooks = self.client.options.hooks
        self.assertEqual(sorted(hooks), ["PostToolUse", "PostToolUseFailure", "PreToolUse", "UserPromptSubmit"])
        self.assertEqual(hooks["UserPromptSubmit"][0].hooks, [self.brain._gate_prompt])

    async def test_no_tool_call_hook_or_approval_callback_is_registered(self):
        b = B.WarmBrain(model="m", ledger=self.led)
        await b.start()
        opts = self.clients[-1].options
        self.assertEqual([(m.matcher, m.hooks) for m in opts.hooks["PreToolUse"]], [("Bash", [b.guard.pre_tool])])   # the Bash deny-list only
        self.assertIsNone(opts.can_use_tool)
        self.assertEqual(opts.permission_mode, "bypassPermissions")

    async def test_it_can_be_turned_off_by_config(self):
        with mock.patch.dict(B.CFG, {"background": {"withhold_raw_reports": False}}):
            b = B.WarmBrain(model="m", ledger=self.led)
            await b.start()
        self.assertEqual(sorted(self.clients[-1].options.hooks), ["PostToolUse", "PostToolUseFailure", "PreToolUse"])    # no gate; Bash deny-list and fetchguard remain

    async def test_a_raw_background_agent_report_is_blocked(self):
        await self.launch()
        out = await self.block(raw_notice(WRONG))
        self.assertEqual(out["decision"], "block")
        self.assertIn("task ledger", out["reason"])
        self.assertNotIn("20", out["reason"])
        n = [x for x in self.led.notes if x.get("note") == "raw_task_report_withheld"]
        self.assertEqual((len(n), n[0]["task_ids"]), (1, [TID]))
        self.assertGreater(n[0]["bytes"], 100)
        self.assertNotIn("20 daily", repr(n[0]))                      # sizes and ids only, never the body

    async def test_a_notice_for_a_task_the_ledger_does_not_know_fails_closed(self):
        await self.launch()
        self.assertEqual((await self.block(raw_notice(WRONG, tid="ghost")))["decision"], "block")
        self.assertEqual((await self.block("<task-notification>\nno id\n</task-notification>"))["decision"], "block")

    async def test_a_notice_naming_an_agent_and_something_else_is_blocked(self):
        await self.launch()
        self.client.feed(F.task_started("sh1", "toolu_sh", "build", task_type="local_bash", is_backgrounded=False))
        await settle(6)
        both = raw_notice("x").replace(f"<task-id>{TID}</task-id>", f"<task-id>{TID}</task-id><task-id>sh1</task-id>")
        self.assertEqual((await self.block(both))["decision"], "block")

    async def test_a_background_shell_commands_output_is_not_withheld(self):
        """Its notice is the only way the model learns what the command printed."""
        self.client.feed(F.task_started("sh1", "toolu_sh", "build", task_type="local_bash", is_backgrounded=False))
        await settle(6)
        self.assertEqual(self.led.tasks["sh1"]["task_kind"], L.KIND_FG_TOOL)
        self.assertEqual(await self.block(raw_notice("built ok", tid="sh1", tuid="toolu_sh")), {})
        self.assertFalse([x for x in self.led.notes if x.get("note") == "raw_task_report_withheld"])

    async def test_ordinary_prompts_are_never_touched(self):
        for prompt in ("what's the weather", "", "yes", "<task>not a notice</task>", HEAD + "Background task x finished.]",
                       "Please summarise <task-notification> handling", "  \n  hello"):
            self.assertEqual(await self.block(prompt), {}, prompt)
        self.assertEqual(await self.brain._gate_prompt(None, None, None), {})
        self.assertEqual(await self.brain._gate_prompt({}, None, None), {})
        self.assertEqual(await self.brain._gate_prompt({"prompt": None}, None, None), {})
        self.assertEqual(await self.brain._gate_prompt({"prompt": 42}, None, None), {})

    async def test_the_gate_cannot_deafen_the_user_when_it_errors(self):
        await self.launch()
        with mock.patch.object(self.brain, "ledger", mock.Mock(tasks=mock.Mock(get=mock.Mock(side_effect=RuntimeError("x"))))):
            self.assertEqual((await self.block(raw_notice(WRONG)))["decision"], "block")   # a raw notice: fail closed
            self.assertEqual(await self.block("what's the weather"), {})                    # a user prompt: never blocked


class ForegroundRepresentation(Base):
    """Exactly what the foreground model is told, per verification state."""

    async def test_verified_21(self):
        await self.launch()
        await self.child()
        await self.finish("21 daily note files found")
        self.assertEqual(self.verdict(), L.V_VERIFIED)
        self.assertEqual(self.brain.note_prefix(),
                         HEAD + NAME + " finished. It reported 21 daily note files."
                                       " (Ledger: VERIFIED against its own tool output.)]")

    async def test_verify_failed_20_against_21_never_shows_the_20(self):
        await self.launch()
        await self.child()
        await self.finish(WRONG)
        self.assertEqual(self.verdict(), L.V_FAILED)
        note = self.brain.note_prefix()
        self.assertEqual(note, HEAD + NAME + " finished. The count in its report didn't match what its own tool "
                                             "returned, which was 21. (Ledger: VERIFY_FAILED. Only the tool's number is "
                                             "a fact; the report's numbers are not.)]")
        self.assertNotRegex(note, r"\b20\b")
        self.assertNotIn("daily note files found", note)

    async def test_an_unverifiable_number_is_not_presented_as_a_fact(self):
        await self.launch()                                       # no tool evidence at all
        await self.finish(WRONG)
        self.assertEqual(self.verdict(), L.V_UNVERIFIED)
        note = self.brain.note_prefix()
        self.assertEqual(note, HEAD + NAME + " finished. I've got its report."
                                             " (Ledger: UNVERIFIED. None of the numbers in its report is a fact.)]")
        self.assertNotRegex(note, r"\b20\b")

    async def test_a_withheld_report_is_saved_and_the_model_is_pointed_at_it(self):
        # 2026-10-25: a blocked report used to be lost for good, so Jarvis could never say what an agent found.
        await self.launch()
        await self.finish(WRONG)
        out = await self.block(raw_notice("Full findings: " + WRONG))
        self.assertEqual(out["decision"], "block")                # the raw notice still never becomes a model turn
        path = self.led.tasks[TID]["report_path"]
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "Full findings: " + WRONG)
        note = self.brain.note_prefix()
        self.assertIn(f"Its full report is saved at {path}. Read that file now and tell him what it found", note)
        self.assertIn("UNVERIFIED", note)                         # its numbers stay labelled as claims
        self.assertNotIn("20 daily", note)
        self.assertNotIn(path, SP.task_line(self.led.tasks[TID]))  # a file path is never spoken aloud

    async def test_a_task_with_no_number_keeps_its_useful_summary(self):
        await self.launch()
        await self.child()
        await self.finish("Finished searching the September daily-notes directory.")
        self.assertEqual(self.verdict(), L.V_NOT_REQUIRED)
        self.assertEqual(self.brain.note_prefix(),
                         HEAD + NAME + " finished. It reported: Finished searching the September daily-notes directory.]")

    async def test_only_numbers_the_evidence_supports_become_factual_context(self):
        await self.launch()
        await self.child()
        await self.finish("Found 21 files, including 8 from one group and 12 from another.")
        self.assertEqual(self.verdict(), L.V_VERIFIED)
        note = self.brain.note_prefix()
        self.assertIn("It reported 21 files.", note)
        self.assertNotRegex(note, r"\b(8|12)\b")                  # unsupported parts are not facts
        self.assertEqual([m["n"] for m in self.led.tasks[TID]["result_verification"]["matched"]], [21])

    async def test_a_failed_task_gives_its_state_and_manufactures_no_result(self):
        await self.launch()
        await self.child()                                        # a real count exists, but the task FAILED
        await self.finish("I counted 20 files before it broke", status="failed")
        self.assertEqual(self.led.tasks[TID]["state"], L.K_FAILED)
        note = self.brain.note_prefix()
        self.assertEqual(note, HEAD + NAME + " failed.]")
        self.assertNotRegex(note, r"\b(20|21)\b")

    async def test_the_spoken_line_is_unchanged(self):
        await self.launch()
        await self.child()
        await self.finish(WRONG)
        self.assertEqual(SP.task_line(self.led.tasks[TID]),
                         NAME + " finished. The count in its report didn't match what its own tool returned, which was 21.")


class WhatTheCliIsActuallySent(Base):
    """THE CRITICAL TEST. Reproduce the exact B5 case end to end and inspect the
    payload that goes to the CLI for the foreground turn."""

    async def test_the_foreground_payload_has_the_tools_21_and_never_the_agents_20(self):
        await self.launch()
        await self.child()                                        # tool evidence: 21 paths
        await self.finish(WRONG)                                  # the agent's report: 20
        self.assertEqual(self.verdict(), L.V_FAILED)
        # the CLI now offers the model the raw notice: the gate withholds it
        raw = raw_notice("Based on the output, the files before 2026-09-30 are 20. **" + WRONG + "**")
        self.assertEqual((await self.block(raw))["decision"], "block")
        # the CLI's blocked-run result carries the original prompt text: it must stay inert
        self.client.feed(F.init(), F.result("UserPromptSubmit operation blocked by hook:\nwithheld\n\nOriginal prompt: "
                                            + raw, origin="task-notification", num_turns=0))
        await settle(8)
        # the next foreground turn, built exactly as main.handle builds it
        text = f"{self.brain.note_prefix()} how many daily notes did the agent find"
        spoken = [s async for s in self.brain.ask_stream(text)]
        sent = self.client.sent[-1]["message"]["content"]
        self.assertIn("21", sent)
        self.assertIn("VERIFY_FAILED", sent)
        self.assertNotRegex(sent, r"\b20\b")
        self.assertNotIn("<task-notification>", sent)
        self.assertNotIn("<result>", sent)
        self.assertNotIn("daily note files found", sent)
        self.assertNotIn("Original prompt", sent)
        self.assertEqual(spoken, ["Noted."])                      # the blocked run produced nothing for the mouth

    async def test_the_unsolicited_run_that_follows_a_blocked_notice_never_reaches_a_turn(self):
        await self.launch()
        await self.child()
        await self.finish(WRONG)
        self.client.feed(F.init(), F.asst("The background agent confirmed 20 daily note files."),
                         F.result("The background agent confirmed 20 daily note files.", origin="task-notification"))
        await settle(8)
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Four."))
        spoken = [s async for s in self.brain.ask_stream("what is two plus two")]
        self.assertEqual(spoken, ["Four."])
        self.assertNotIn("20", self.client.sent[-1]["message"]["content"])
        self.assertEqual(self.brain.mgr.quarantine.counts["unowned_result"], 0)

    async def test_a_matching_report_reaches_the_model_as_a_verified_fact(self):
        await self.launch()
        await self.child()
        await self.finish("21 daily note files found")
        text = f"{self.brain.note_prefix()} how many daily notes did the agent find"
        [s async for s in self.brain.ask_stream(text)]
        sent = self.client.sent[-1]["message"]["content"]
        self.assertIn("It reported 21 daily note files.", sent)
        self.assertIn("Ledger: VERIFIED", sent)


class LateEvidenceReachesTheForeground(Base):
    async def test_before_the_evidence_no_unchecked_number_is_a_fact_and_after_it_the_final_state_follows(self):
        await self.launch()
        await self.child(hold=True)                               # the task will end BEFORE its result lands
        await self.finish(WRONG)
        self.assertEqual(self.verdict(), L.V_UNVERIFIED)
        first = self.brain.note_prefix()
        self.assertNotRegex(first, r"\b(20|21)\b")
        self.assertIn("Ledger: UNVERIFIED", first)
        self.assertEqual(self.brain.note_prefix(), "")            # surfaced exactly once
        await self.result_lands()                                 # the router rechecks the owning task
        self.assertEqual(self.verdict(), L.V_FAILED)
        update = self.brain.note_prefix()
        self.assertEqual(update, HEAD + "Update on background task Count daily notes in vault: The count in its report "
                                         "didn't match what its own tool returned, which was 21. (Ledger: VERIFY_FAILED. "
                                         "Only the tool's number is a fact; the report's numbers are not.)]")
        self.assertNotRegex(update, r"\b20\b")
        self.assertEqual(self.brain.note_prefix(), "")            # and only once

    async def test_late_evidence_can_upgrade_to_verified(self):
        await self.launch()
        await self.child(hold=True)
        await self.finish("21 daily note files found")
        self.brain.note_prefix()
        await self.result_lands()
        self.assertEqual(self.verdict(), L.V_VERIFIED)
        self.assertIn("Update on background task", self.brain.note_prefix())

    async def test_evidence_that_lands_before_the_note_is_taken_gives_one_final_note_and_no_update(self):
        await self.launch()
        await self.child(hold=True)
        await self.finish(WRONG)
        await self.result_lands()
        note = self.brain.note_prefix()
        self.assertIn("Ledger: VERIFY_FAILED", note)
        self.assertNotIn("Update on", note)
        self.assertEqual(self.brain.note_prefix(), "")

    async def test_a_duplicate_completion_cannot_change_the_verdict_or_the_foreground_view(self):
        await self.launch()
        await self.child()
        await self.finish(WRONG)
        before = self.brain.note_prefix()
        self.client.feed(F.task_note(TID, LAUNCH, "completed", "21 daily note files found"))
        await settle(8)
        self.assertEqual(self.verdict(), L.V_FAILED)
        self.assertEqual(self.led.tasks[TID]["result_verification"]["failed"], [20])
        self.assertEqual(self.brain.note_prefix(), "")            # nothing new to say
        self.assertIn("VERIFY_FAILED", before)

    async def test_a_state_that_never_moved_produces_no_update(self):
        await self.launch()
        await self.child()
        await self.finish("All present.")
        self.brain.note_prefix()
        self.client.feed(F.tool_use("toolu_x", "Bash", {"command": "ls -1"}), F.tool_result("toolu_x", "a\nb"))
        await settle(8)
        self.assertEqual(self.brain.note_prefix(), "")


class PolicyLevel(unittest.TestCase):
    def task(self, state=L.K_COMPLETED, rv=None, summary=""):
        t = {"id": "t", "description": "count", "state": state, "summary": summary}
        if rv:
            t["result_verification"] = rv
        return t

    def test_the_helpers_never_raise_and_only_match_the_cli_notice(self):
        self.assertTrue(SP.is_task_notification(raw_notice("x")))
        self.assertTrue(SP.is_task_notification("  \n<task-notification>"))
        for v in (None, "", 5, "hello <task-notification>", HEAD + "x]", object()):
            self.assertFalse(SP.is_task_notification(v), v)
        self.assertEqual(SP.task_notice_ids(raw_notice("x")), [TID])
        self.assertEqual(SP.task_notice_ids(None), [])

    def test_the_tag_only_exists_for_a_finished_task_that_has_a_verdict(self):
        self.assertEqual(SP.verification_tag(self.task(rv={"state": L.V_NOT_REQUIRED})), "")
        self.assertEqual(SP.verification_tag(self.task()), "")
        self.assertEqual(SP.verification_tag(self.task(L.K_FAILED, {"state": L.V_FAILED})), "")
        for st in (L.V_VERIFIED, L.V_FAILED, L.V_UNVERIFIED):
            self.assertIn(st, SP.verification_tag(self.task(rv={"state": st})))

    def test_the_tag_never_carries_a_number(self):
        for st in (L.V_VERIFIED, L.V_FAILED, L.V_UNVERIFIED):
            self.assertNotRegex(SP.verification_tag(self.task(rv={"state": st, "facts": [21], "failed": [20]})), r"\d")


if __name__ == "__main__":
    unittest.main()
