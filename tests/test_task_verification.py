"""Phase B (B1-B4): a background task's reported numbers are checked against the
tool output the router actually saw, and an unverified model number is never
spoken.

THE INCIDENT, 2026-09-30 21:40 (ledger task af719640aff837f83). A background
agent ran one `find ... | sort`; the router saw 21 paths come back (a 1826-byte
result). The agent's summary said "20 daily note files found (September 2026
subfolder only)". Backtalk's ledger kept only the result's size and hash, then
built "It reported: 20 daily note files found..." from the model's summary and
handed it to the next turn. The user was told 20; the filesystem said 21."""
import asyncio
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import router as R
from backtalk import speech_policy as SP
from backtalk import verify as V
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle

NOTES_DIR = "/Users/yousif/Desktop/jarvis/jarvis/01 - Daily Notes/09 - September 2026/"
FIND_CMD = ('find "/Users/yousif/Desktop/jarvis/jarvis/01 - Daily Notes/" -type f '
            '-name "*.md" ! -name "Daily Note Template.md" | sort')
LAUNCH_TUID = "toolu_live_launch"
CHILD_TUID = "toolu_live_child"
TID = "af719640aff837f83"
# 2026-09-09 .. 2026-09-30 is 22 days; the vault had 21 notes
MISSING = date(2026, 9, 17)
DAYS = [date(2026, 9, 9) + timedelta(days=i) for i in range(22)]
PATHS = [f"{NOTES_DIR}{d.isoformat()}.md" for d in DAYS if d != MISSING]
FIND_OUTPUT = "\n".join(PATHS)
WRONG_SUMMARY = ("20 daily note files found (September 2026 subfolder only): "
                 + ", ".join(d.isoformat() for d in DAYS[:5]) + ", ...")


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.client = FakeClient()
        self.mgr = R.EpochManager(self.led, echo_timeout_s=0.25, barge_escalate_s=0.15, proc=False)
        self.r = await self.mgr.open(self.client, session_id="s", cli_pid=None)
        self.policy = SP.SpeechPolicy(self.led)
        self.mgr.tasks.listeners.append(
            lambda row: self.policy.on_task_terminal(row, SP.Context(fg_active=True)))

    async def asyncTearDown(self):
        for t in (self.r._pump_task, self.r._watch_task, *self.r._escalations):
            if t:
                t.cancel()
        await asyncio.sleep(0)
        self.led.close()
        self.tmp.cleanup()

    async def launch(self, tid=TID, tuid=LAUNCH_TUID):
        """The captured launch: an Agent with no run_in_background flag whose
        start message says is_backgrounded."""
        h = await self.r.send("Launch a background agent to count my daily notes")
        self.client.feed(
            F.init(), F.echo(h.uuid),
            F.tool_use(tuid, "Agent", {"description": "Count daily notes in vault",
                                       "prompt": "count them", "subagent_type": "general-purpose"}),
            F.task_started(tid, tuid, "Count daily notes in vault", task_type="local_agent",
                           is_backgrounded=True),
            F.tool_result(tuid, "Async agent launched successfully."),
            F.delta("Counting."), F.asst("Counting."), F.result("Counting."))
        await settle()
        return h

    async def child_runs(self, cmd=FIND_CMD, output=FIND_OUTPUT, tuid=CHILD_TUID,
                         parent=LAUNCH_TUID, is_error=None):
        """The subagent's own tool call and result, as the router sees them."""
        self.client.feed(F.tool_use(tuid, "Bash", {"command": cmd}, parent=parent))
        await settle()
        # a subagent's Bash call passes the same PreToolUse hook BEFORE it runs
        F.seen(self.led, tuid, "ALLOW", tool="Bash")
        self.client.feed(F.tool_result(tuid, output, is_error=is_error, parent=parent))
        await settle()

    async def finish(self, summary, tid=TID, tuid=LAUNCH_TUID, first_empty=True):
        """Completion exactly as captured: task_updated, an empty-summary
        notification, then the real one, then the unsolicited report run."""
        msgs = [F.task_updated(tid, "completed")]
        if first_empty:
            msgs.append(F.task_note(tid, tuid, "completed", ""))
        msgs += [F.task_note(tid, tuid, "completed", summary), *F.unsolicited_run("report")]
        self.client.feed(*msgs)
        await settle()

    def note(self):
        return self.policy.note_prefix()


class TheTwentyVersusTwentyOneIncident(Base):
    async def test_the_incident_replayed_end_to_end_never_speaks_20(self):
        await self.launch()
        await self.child_runs()
        await self.finish(WRONG_SUMMARY)
        task = self.led.tasks[TID]
        self.assertEqual(task["state"], L.K_COMPLETED)
        # the router really did see 21 paths come back
        self.assertEqual(len(PATHS), 21)
        # what the next turn is told
        line = SP.task_line(task)
        note = self.note()
        for said in (line, note):
            self.assertNotRegex(said, r"\b20\b", said)           # the model's number is never repeated
            self.assertNotIn("daily note files found", said)    # nor its sentence
            self.assertIn("21", said)                            # the tool's number is stated
        self.assertIn("didn't match", note)
        # and the ledger says why
        rv = task["result_verification"]
        self.assertEqual(rv["state"], L.V_FAILED)
        self.assertEqual(rv["level"], L.LEVEL_TOOL_EVIDENCE)
        self.assertEqual(rv["facts"], [21])

    async def test_the_same_run_with_a_correct_summary_is_verified_and_spoken(self):
        await self.launch()
        await self.child_runs()
        await self.finish("21 daily note files, dated 2026-09-09 to 2026-09-30.")
        task = self.led.tasks[TID]
        self.assertEqual(task["result_verification"]["state"], L.V_VERIFIED)
        self.assertEqual(task["result_verification"]["level"], L.LEVEL_TOOL_EVIDENCE)
        line = SP.task_line(task)
        self.assertIn("21", line)
        self.assertIn("daily note files", line)
        self.assertNotIn("dated", line)                  # only the verified claim is spoken, by code

    async def test_the_evidence_is_correlated_by_parent_tool_use_id(self):
        await self.launch()
        await self.child_runs()
        # an unrelated action, outside the task, with a different count
        self.client.feed(F.tool_use("toolu_other", "Bash", {"command": "ls -1 /tmp"}),
                         F.tool_result("toolu_other", "a\nb\nc"))
        await settle()
        await self.finish("21 daily note files.")
        children = self.led.child_actions(LAUNCH_TUID)
        self.assertEqual([a["tool_use_id"] for a in children], [CHILD_TUID])
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_VERIFIED)

    async def test_the_summary_arriving_late_still_gets_checked(self):
        """The first completion signal has an empty summary and fires the
        listener; the real summary comes in a duplicate."""
        await self.launch()
        await self.child_runs()
        self.client.feed(F.task_updated(TID, "completed"),
                         F.task_note(TID, LAUNCH_TUID, "completed", ""))
        await settle()
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_NOT_REQUIRED)
        self.client.feed(F.task_note(TID, LAUNCH_TUID, "completed", WRONG_SUMMARY))
        await settle()
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_FAILED)
        self.assertNotRegex(self.note(), r"\b20\b")

    async def test_the_incident_through_the_brain_note_prefix(self):
        """The same replay through WarmBrain: this is the path whose output
        was prepended to the next turn's text."""
        clients = []

        def factory(options=None):
            c = FakeClient()
            c.options = options
            clients.append(c)
            return c
        with mock.patch.object(B, "ClaudeSDKClient", factory), \
                mock.patch.object(B, "build_session_brief", lambda: ""):
            brain = B.WarmBrain(model="m", ledger=self.led)
            brain.get_speech_context = lambda: SP.Context(fg_active=True)
            await brain.start()
            c = clients[-1]
            h = await brain.router.send("Launch a background agent to count my daily notes")
            c.feed(F.init(), F.echo(h.uuid),
                   F.tool_use(LAUNCH_TUID, "Agent", {"description": "Count daily notes in vault"}),
                   F.task_started(TID, LAUNCH_TUID, "Count daily notes in vault",
                                  task_type="local_agent", is_backgrounded=True),
                   F.tool_result(LAUNCH_TUID, "launched"), F.delta("ok."), F.asst("ok."), F.result("ok."),
                   F.tool_use(CHILD_TUID, "Bash", {"command": FIND_CMD}, parent=LAUNCH_TUID),
                   F.tool_result(CHILD_TUID, FIND_OUTPUT, parent=LAUNCH_TUID),
                   F.task_updated(TID, "completed"),
                   F.task_note(TID, LAUNCH_TUID, "completed", WRONG_SUMMARY),
                   *F.unsolicited_run("report"))
            await settle(12)
            note = brain.note_prefix()
            r = brain.router
            for t in (r._pump_task, r._watch_task, *r._escalations):
                if t:
                    t.cancel()
        self.assertTrue(note.startswith("[System note, from the task ledger"), note)
        self.assertNotRegex(note, r"\b20\b")
        self.assertIn("21", note)


class WhatTheVerifierDecides(Base):
    def verdict(self, tid=TID):
        return self.led.tasks[tid]["result_verification"]

    async def test_numbers_with_no_evidence_are_unverified_and_not_repeated(self):
        await self.launch()
        await self.finish("20 daily note files found")
        self.assertEqual(self.verdict()["state"], L.V_UNVERIFIED)
        for said in (SP.task_line(self.led.tasks[TID]), self.note()):
            self.assertNotRegex(said, r"\b20\b")
            self.assertIn("I've got its report", said)

    async def test_a_summary_without_numbers_is_spoken_as_before(self):
        await self.launch()
        await self.child_runs()
        await self.finish("All the daily notes are in the September folder.")
        self.assertEqual(self.verdict()["state"], L.V_NOT_REQUIRED)
        self.assertIn("It reported: All the daily notes are in the September folder.",
                      SP.task_line(self.led.tasks[TID]))

    async def test_a_count_from_wc_is_evidence_too(self):
        await self.launch()
        await self.child_runs(cmd=FIND_CMD.replace("| sort", "| wc -l"), output="      21")
        await self.finish("20 daily notes")
        self.assertEqual(self.verdict()["state"], L.V_FAILED)
        self.assertEqual(self.verdict()["facts"], [21])

    async def test_a_zsh_array_count_with_the_cwd_line_is_evidence(self):
        await self.launch()
        await self.child_runs(cmd='cd "/x/" && setopt extended_glob null_glob; files=(**/*.md); echo ${#files[@]}',
                              output="21\nShell cwd was reset to /private/tmp/x")
        await self.finish("There are 21 notes.")
        self.assertEqual(self.verdict()["state"], L.V_VERIFIED)

    async def test_truncated_output_is_never_used_to_call_a_summary_wrong(self):
        await self.launch()
        big = "\n".join(f"{NOTES_DIR}file-{i:05d}.md" for i in range(400))
        await self.child_runs(output=big)
        await self.finish("20 daily note files")
        self.assertEqual(self.verdict()["state"], L.V_UNVERIFIED)

    async def test_a_failed_tool_call_gives_no_count(self):
        await self.launch()
        await self.child_runs(output="find: permission denied", is_error=True)
        await self.finish("20 daily note files")
        self.assertEqual(self.verdict()["state"], L.V_UNVERIFIED)

    async def test_another_tasks_evidence_is_not_used(self):
        await self.launch()
        await self.child_runs()                                        # task TID: 21
        await self.launch(tid="other", tuid="toolu_other_launch")
        await self.child_runs(cmd="find . -type f", output="a\nb\nc\nd\ne",
                              tuid="toolu_other_child", parent="toolu_other_launch")   # task other: 5
        await self.finish("5 daily note files", tid="other", tuid="toolu_other_launch")
        await self.finish("21 daily note files")
        self.assertEqual(self.verdict("other")["state"], L.V_VERIFIED)
        self.assertEqual(self.verdict("other")["facts"], [5])
        self.assertEqual(self.verdict(TID)["state"], L.V_VERIFIED)
        self.assertEqual(self.verdict(TID)["facts"], [21])

    async def test_evidence_from_a_nested_subagent_counts(self):
        await self.launch()
        self.client.feed(F.tool_use("toolu_inner", "Agent", {"description": "inner"}, parent=LAUNCH_TUID),
                         F.tool_result("toolu_inner", "started", parent=LAUNCH_TUID))
        await settle()
        await self.child_runs(tuid="toolu_deep", parent="toolu_inner")
        await self.finish(WRONG_SUMMARY)
        self.assertEqual(self.verdict()["state"], L.V_FAILED)

    async def test_a_claim_about_another_kind_of_thing_is_not_called_wrong(self):
        await self.launch()
        await self.child_runs()
        await self.finish("21 daily note files in 3 folders, drafted 4 emails")
        v = self.verdict()
        self.assertEqual(v["state"], L.V_VERIFIED)
        self.assertEqual([m["n"] for m in v["matched"]], [21])
        self.assertGreaterEqual(v["unchecked"], 2)
        line = SP.task_line(self.led.tasks[TID])
        self.assertIn("21 daily note files", line)
        self.assertNotRegex(line, r"\b(3|4)\b")                        # the unchecked numbers are not spoken

    async def test_only_unverifiable_numbers_are_not_repeated(self):
        await self.launch()
        await self.child_runs()
        await self.finish("Estimated ARV is 250000")
        self.assertEqual(self.verdict()["state"], L.V_UNVERIFIED)
        self.assertNotIn("250000", SP.task_line(self.led.tasks[TID]))

    async def test_a_year_in_the_summary_is_not_a_folder_count_that_contradicts_the_tool(self):
        """Found live 2026-10-01: "September 2026 daily notes directory" was read
        as 2026 directories. With a real folder count of 3 in hand that would have
        been reported as a mismatch."""
        await self.launch()
        await self.child_runs(cmd=f'find "{NOTES_DIR}" -type d | sort', output="/a\n/a/b\n/a/c")
        await self.finish("3 folders found in the September 2026 daily notes directory")
        v = self.verdict()
        self.assertEqual(v["state"], L.V_VERIFIED, v)
        self.assertEqual([m["n"] for m in v["matched"]], [3])
        self.assertEqual(v["failed"], [])
        self.assertEqual([c["n"] for c in self.led.tasks[TID]["claims"]], [3])

    async def test_the_live_summary_with_its_year_phrase_is_still_failed_only_for_the_20(self):
        await self.launch()
        await self.child_runs()
        await self.finish("Command executed successfully. The find command located all markdown files "
                          "in the September 2026 daily notes directory. 20 daily note files found.")
        v = self.verdict()
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))
        self.assertEqual([c["n"] for c in self.led.tasks[TID]["claims"]], [20])

    async def test_a_late_correct_summary_after_a_wrong_one_is_rechecked_once(self):
        await self.launch()
        await self.child_runs()
        n0 = sum(1 for _ in open(self.led.path))
        await self.finish("21 daily note files.", first_empty=False)
        n1 = sum(1 for _ in open(self.led.path))
        out = V.verify_task_result(self.led, TID)                      # nothing new: records nothing
        self.assertEqual(out.state, L.V_VERIFIED)
        self.assertEqual(sum(1 for _ in open(self.led.path)), n1)
        self.assertGreater(n1, n0)


class ReadOnlyResultsAreEvidence(Base):
    """B4: a read-only action's result is captured because a claim may need it."""

    async def test_a_bash_result_carries_evidence_and_stays_unverified(self):
        await self.launch()
        await self.child_runs()
        a = self.led.actions[self.led.action_id_for(CHILD_TUID)]
        self.assertEqual(a["evidence"]["kind"], "list")
        self.assertEqual(a["evidence"]["lines"], 21)
        self.assertEqual(V.plan(a, {"command": FIND_CMD})[0].state, L.V_UNVERIFIED)

    async def test_every_tool_result_gets_evidence_not_only_bash(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.tool_use("toolu_g", "Glob", {"pattern": "*.md"}),
                         F.tool_result("toolu_g", "a.md\nb.md\nc.md"),
                         F.tool_use("toolu_r", "Read", {"file_path": "/x"}),
                         F.tool_result("toolu_r", "text"))
        await settle()
        g = self.led.actions[self.led.action_id_for("toolu_g")]
        self.assertEqual(g["evidence"]["lines"], 3)
        self.assertIn("evidence", self.led.actions[self.led.action_id_for("toolu_r")])

    async def test_a_denied_action_gets_no_evidence(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_d", "Bash", {"command": "rm x"}))
        await settle()
        F.seen(self.led, "toolu_d", "DENY", tool="Bash")
        self.client.feed(F.tool_result("toolu_d", "Blocked: dangerous", is_error=True))
        await settle()
        self.assertNotIn("evidence", self.led.actions[self.led.action_id_for("toolu_d")])

    async def test_an_extractor_that_blows_up_cannot_cost_the_result(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_x", "Bash", {"command": "ls"}))
        await settle()
        with mock.patch("backtalk.router.E.extract_evidence", side_effect=RuntimeError("x")):
            self.client.feed(F.tool_result("toolu_x", "a"))
            await settle()
        a = self.led.actions[self.led.action_id_for("toolu_x")]
        self.assertEqual(a["state"], L.A_OK)
        self.assertNotIn("evidence", a)


class LedgerRulesForTaskVerification(Base):
    async def test_a_summary_can_never_verify_itself(self):
        await self.launch()
        self.led.set_task_verification(TID, L.V_VERIFIED, L.LEVEL_SELF, "the agent said so")
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_UNVERIFIED)
        self.led.set_task_verification(TID, L.V_VERIFIED, L.LEVEL_TOOL_EVIDENCE, "matches")
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_VERIFIED)

    async def test_evidence_and_verification_survive_a_replay_and_hold_no_bodies(self):
        await self.launch()
        await self.child_runs()
        await self.finish(WRONG_SUMMARY)
        self.led.close()
        again = L.Ledger(self.tmp.name, fsync=False)
        self.assertEqual(again.tasks[TID]["result_verification"]["state"], L.V_FAILED)
        self.assertEqual(again.actions[again.action_id_for(CHILD_TUID)]["evidence"]["lines"], 21)
        raw = open(again.path, encoding="utf-8").read()
        self.assertNotIn("2026-09-20.md", raw)                         # the result body was never stored
        again.close()

    async def test_unknown_task_is_none(self):
        self.assertIsNone(V.verify_task_result(self.led, "nope"))
        self.assertIsNone(V.check_task_summary(self.led, "nope", "20 files"))


class SpeechLineFromVerification(unittest.TestCase):
    def task(self, rv=None, summary="", state=L.K_COMPLETED):
        t = {"id": "t", "description": "count notes", "state": state, "summary": summary}
        if rv is not None:
            t["result_verification"] = rv
        return t

    def test_each_state_speaks_only_what_it_may(self):
        v = SP.task_line(self.task({"state": L.V_VERIFIED, "matched": [{"n": 21, "noun": "daily note files"}]},
                                   "21 daily note files, 3 folders"))
        self.assertIn("It reported 21 daily note files.", v)
        self.assertNotIn("folders", v)
        f = SP.task_line(self.task({"state": L.V_FAILED, "facts": [21], "failed": [20]}, "20 daily note files"))
        self.assertIn("didn't match", f)
        self.assertIn("21", f)
        self.assertNotRegex(f, r"\b20\b")
        u = SP.task_line(self.task({"state": L.V_UNVERIFIED}, "20 daily note files"))
        self.assertIn("I've got its report", u)
        self.assertNotRegex(u, r"\b20\b")
        n = SP.task_line(self.task({"state": L.V_NOT_REQUIRED}, "all present"))
        self.assertIn("It reported: all present.", n)

    def test_with_no_verification_on_record_a_number_is_still_not_repeated(self):
        line = SP.task_line(self.task(None, "20 daily note files found"))
        self.assertNotRegex(line, r"\b20\b")
        self.assertIn("I've got its report", line)
        self.assertIn("It reported: fine.", SP.task_line(self.task(None, "fine")))

    def test_several_tool_counts_are_all_named(self):
        f = SP.task_line(self.task({"state": L.V_FAILED, "facts": [21, 20, 5, 9]}, "x"))
        self.assertIn("21 and 20 and 5", f)

    def test_non_numeric_matched_values_cannot_break_the_line(self):
        v = SP.task_line(self.task({"state": L.V_VERIFIED, "matched": [{"n": "x"}, {"noun": "files"}]}, "x"))
        self.assertTrue(v.startswith("Background task"))

    def test_failed_and_stopped_tasks_are_unchanged(self):
        self.assertIn("failed", SP.task_line(self.task(None, "", L.K_FAILED)))
        self.assertIn("Stopped", SP.task_line(self.task(None, "", L.K_STOPPED_USER)))


class HookFirstOrdering(Base):
    """Audit 2026-10-01: the PreToolUse hook and the stream's tool_use are not
    ordered. When the hook created the row first, the later stream event never
    filled in parent_tool_use_id, so a subagent's call was orphaned and its
    evidence never reached the task."""

    def test_ledger_backfills_the_parent_when_the_stream_event_comes_second(self):
        tmp = tempfile.TemporaryDirectory()
        led = L.Ledger(tmp.name, fsync=False)
        ep = led.epoch_open("s", 1)
        F.seen(led, "toolu_c", "ALLOW", tool="Bash")             # hook first
        row = led.actions[led.action_id_for("toolu_c")]
        self.assertIsNone(row.get("parent_tool_use_id"))
        led.action_seen("toolu_c", ep, "Bash", turn_id="T1", parent_tool_use_id="toolu_launch",
                        intent="count the notes")                                         # stream second
        row = led.actions[led.action_id_for("toolu_c")]
        self.assertEqual(row["parent_tool_use_id"], "toolu_launch")
        self.assertEqual(row["turn_id"], "T1")
        self.assertTrue(row["intent"])
        self.assertEqual(len(led.actions), 1)                                             # still one row
        self.assertEqual([a["tool_use_id"] for a in led.child_actions("toolu_launch")], ["toolu_c"])
        led.close()
        tmp.cleanup()

    def test_a_known_parent_is_never_overwritten(self):
        tmp = tempfile.TemporaryDirectory()
        led = L.Ledger(tmp.name, fsync=False)
        ep = led.epoch_open("s", 1)
        led.action_seen("toolu_c", ep, "Bash", parent_tool_use_id="toolu_first")
        led.action_seen("toolu_c", ep, "Bash", parent_tool_use_id="toolu_other")
        self.assertEqual(led.actions[led.action_id_for("toolu_c")]["parent_tool_use_id"], "toolu_first")
        led.close()
        tmp.cleanup()

    async def test_hook_first_order_still_finds_the_child_and_detects_the_mismatch(self):
        await self.launch()
        # the hook decision lands BEFORE the stream has shown the tool_use
        F.seen(self.led, CHILD_TUID, "ALLOW", tool="Bash")
        self.client.feed(F.tool_use(CHILD_TUID, "Bash", {"command": FIND_CMD}, parent=LAUNCH_TUID))
        await settle()
        self.client.feed(F.tool_result(CHILD_TUID, FIND_OUTPUT, parent=LAUNCH_TUID))
        await settle()
        a = self.led.actions[self.led.action_id_for(CHILD_TUID)]
        self.assertEqual(a["parent_tool_use_id"], LAUNCH_TUID)
        self.assertEqual([x["tool_use_id"] for x in self.led.child_actions(LAUNCH_TUID)], [CHILD_TUID])
        await self.finish(WRONG_SUMMARY)
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))
        self.assertNotRegex(self.note(), r"\b20\b")


class EvidenceProvenanceAtTaskLevel(Base):
    async def test_a_literal_echo_cannot_verify_a_claim_of_the_same_number(self):
        await self.launch()
        await self.child_runs(cmd="echo 20", output="20")
        await self.finish("20 daily note files found")
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual(v["state"], L.V_UNVERIFIED)
        self.assertNotEqual(v["state"], L.V_VERIFIED)
        self.assertEqual(v["facts_all"], [])
        line = SP.task_line(self.led.tasks[TID])
        self.assertNotRegex(line, r"\b20\b")
        self.assertIn("I've got its report", line)

    async def test_echo_cannot_launder_a_number_next_to_a_real_count_either(self):
        await self.launch()
        await self.child_runs()                                          # a real find: 21
        await self.child_runs(cmd="echo 20", output="20", tuid="toolu_echo")
        await self.finish("20 daily note files found")
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))

    async def test_output_padded_by_an_earlier_echo_is_not_a_count(self):
        padded = "Counting:\n" + FIND_OUTPUT                           # 22 lines for 21 files
        await self.launch()
        await self.child_runs(cmd=f'echo "Counting:" && {FIND_CMD}', output=padded)
        await self.finish("21 daily note files")                         # TRUE claim
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual(v["state"], L.V_UNVERIFIED)                     # never "failed" off a padded 22
        self.assertEqual(v["facts_all"], [])

    async def test_legitimate_find_evidence_still_verifies_after_a_silent_cd(self):
        await self.launch()
        await self.child_runs(cmd=f'cd "{NOTES_DIR}" && {FIND_CMD}', output=FIND_OUTPUT)
        await self.finish("21 daily note files")
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_VERIFIED)


class FailClosedAtTaskLevel(Base):
    """The audit phrases, end to end: with real evidence in hand, a number the
    extractor cannot check makes the task UNVERIFIED and is never spoken."""

    PHRASES = ["Found one file.", "zero files matched", "no files found", "a hundred files", "2k files",
               "ARV is 250k", "Estimated ARV is 2000", "price $1950", "5th note", "20x more files",
               "about two dozen files", "one of the files is missing", "dozens of files", "~20 files",
               "21 of 22 files", "20+ files"]

    async def test_no_unverified_number_reaches_the_task_line_or_the_note(self):
        for k, phrase in enumerate(self.PHRASES):
            tid, tuid = f"t{k}", f"toolu_l{k}"
            await self.launch(tid=tid, tuid=tuid)
            await self.child_runs(tuid=f"toolu_c{k}", parent=tuid)
            await self.finish(phrase, tid=tid, tuid=tuid)
            v = self.led.tasks[tid]["result_verification"]
            self.assertIn(v["state"], (L.V_UNVERIFIED, L.V_FAILED, L.V_VERIFIED), phrase)
            line = SP.task_line(self.led.tasks[tid])
            self.assertNotIn("It reported:", line, phrase)               # the model's words are not repeated
            for digits in ("250", "1950", "2000", "2k", "20x", "5th", "dozen", "hundred", "zero", "one file"):
                self.assertNotIn(digits, line, (phrase, line))

    async def test_a_summary_with_no_number_at_all_is_still_spoken(self):
        await self.launch()
        await self.child_runs()
        await self.finish("All the daily notes are in the September folder.")
        self.assertIn("It reported: All the daily notes are in the September folder.",
                      SP.task_line(self.led.tasks[TID]))


class LateEvidence(Base):
    async def finish_then_evidence(self, summary, cmd=FIND_CMD, output=FIND_OUTPUT):
        await self.launch()
        self.client.feed(F.tool_use(CHILD_TUID, "Bash", {"command": cmd}, parent=LAUNCH_TUID))
        await settle()
        F.seen(self.led, CHILD_TUID, "ALLOW", tool="Bash")
        await self.finish(summary)                                       # the task ends BEFORE the result lands
        v0 = self.led.tasks[TID]["result_verification"]["state"]
        self.client.feed(F.tool_result(CHILD_TUID, output, parent=LAUNCH_TUID))
        await settle()
        return v0, self.led.tasks[TID]["result_verification"]

    async def test_evidence_that_lands_after_the_task_finished_is_checked(self):
        v0, v1 = await self.finish_then_evidence(WRONG_SUMMARY)
        self.assertEqual(v0, L.V_UNVERIFIED)                             # nothing to check against yet
        self.assertEqual((v1["state"], v1["facts"], v1["failed"]), (L.V_FAILED, [21], [20]))
        self.assertNotRegex(self.note(), r"\b20\b")

    async def test_late_evidence_can_also_verify(self):
        v0, v1 = await self.finish_then_evidence("21 daily note files.")
        self.assertEqual((v0, v1["state"]), (L.V_UNVERIFIED, L.V_VERIFIED))

    async def test_late_nested_subagent_evidence_reaches_the_task(self):
        await self.launch()
        self.client.feed(F.tool_use("toolu_inner", "Agent", {"description": "inner"}, parent=LAUNCH_TUID),
                         F.tool_result("toolu_inner", "started", parent=LAUNCH_TUID))
        await settle()
        self.client.feed(F.tool_use("toolu_deep", "Bash", {"command": FIND_CMD}, parent="toolu_inner"))
        await settle()
        F.seen(self.led, "toolu_deep", "ALLOW", tool="Bash")
        await self.finish(WRONG_SUMMARY)
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_UNVERIFIED)
        self.client.feed(F.tool_result("toolu_deep", FIND_OUTPUT, parent="toolu_inner"))
        await settle()
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_FAILED)

    async def test_a_recorded_failed_verdict_is_not_overwritten_by_later_evidence(self):
        await self.launch()
        await self.child_runs()                                          # 21
        await self.finish(WRONG_SUMMARY)                                 # FAILED for 20
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_FAILED)
        await self.child_runs(cmd="find /other -type f", output="\n".join(["x"] * 20),
                              tuid="toolu_later", parent=LAUNCH_TUID)    # a later list of exactly 20
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))

    async def test_a_recorded_verified_verdict_is_not_overwritten_by_later_evidence(self):
        await self.launch()
        await self.child_runs()
        await self.finish("21 daily note files.")
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_VERIFIED)
        await self.child_runs(cmd="find /other -type f", output="\n".join(["x"] * 5),
                              tuid="toolu_later", parent=LAUNCH_TUID)
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_VERIFIED)

    async def test_a_late_or_duplicate_summary_cannot_overwrite_a_verdict(self):
        await self.launch()
        await self.child_runs()
        await self.finish(WRONG_SUMMARY)
        self.client.feed(F.task_note(TID, LAUNCH_TUID, "completed", "21 daily note files."))
        await settle()
        v = self.led.tasks[TID]["result_verification"]
        self.assertEqual((v["state"], v["failed"]), (L.V_FAILED, [20]))
        self.assertEqual(self.led.tasks[TID]["summary"][:2], "20")

    async def test_unrelated_late_results_do_not_disturb_a_finished_task(self):
        await self.launch()
        await self.child_runs()
        await self.finish("All present.")
        before = sum(1 for _ in open(self.led.path))
        self.client.feed(F.tool_use("toolu_x", "Bash", {"command": "ls -1"}),
                         F.tool_result("toolu_x", "a\nb"))
        await settle()
        self.assertEqual(self.led.tasks[TID]["result_verification"]["state"], L.V_NOT_REQUIRED)
        self.assertLessEqual(sum(1 for _ in open(self.led.path)) - before, 4)   # the action's own rows only


class FailedVerdictReportsOnlyRelevantFacts(Base):
    async def two_kinds_of_evidence(self, summary):
        await self.launch()
        await self.child_runs()                                                    # 21 files
        await self.child_runs(cmd=f'find "{NOTES_DIR}" -type d | wc -l', output="3",
                              tuid="toolu_dirs", parent=LAUNCH_TUID)               # 3 folders
        await self.finish(summary)
        return self.led.tasks[TID]["result_verification"]

    async def test_a_wrong_file_count_is_reported_against_the_file_count_only(self):
        v = await self.two_kinds_of_evidence("20 daily note files found")
        self.assertEqual((v["state"], v["facts"], v["facts_all"]), (L.V_FAILED, [21], [3, 21]))
        self.assertEqual(len(v["evidence_actions"]), 1)
        line = SP.task_line(self.led.tasks[TID])
        self.assertIn("which was 21.", line)
        self.assertNotRegex(line, r"\b3\b")
        self.assertNotRegex(line, r"\b20\b")

    async def test_a_wrong_folder_count_is_reported_against_the_folder_count_only(self):
        v = await self.two_kinds_of_evidence("2 folders found")
        self.assertEqual((v["state"], v["facts"]), (L.V_FAILED, [3]))
        line = SP.task_line(self.led.tasks[TID])
        self.assertIn("which was 3.", line)
        self.assertNotIn("21", line)

    async def test_a_verified_count_does_not_carry_unrelated_facts_either(self):
        v = await self.two_kinds_of_evidence("21 daily note files")
        self.assertEqual((v["state"], v["facts"]), (L.V_VERIFIED, [21]))
        self.assertEqual(v["facts_all"], [3, 21])


class SubCountsAtTaskLevel(Base):
    """Found live 2026-10-01: a correct total was reported VERIFY_FAILED because
    the summary also listed breakdown counts the evidence never produced
    ([1, 8, 12, 20] failed). Only a primary claim the evidence contradicts, or a
    breakdown the evidence itself supports and contradicts, is a mismatch."""

    LIVE_SHAPE = ("Based on the command results, I can examine the file list: The files found are: "
                  "- 2026-09-09.md through 2026-09-16.md (8 files) - 2026-09-18.md through 2026-09-29.md "
                  "(12 files) - 2026-09-30.md (1 file) Files dated before 2026-09-30: 20 files. "
                  "20 daily note files found.")

    async def verdict_for(self, summary, *, extra_finds=()):
        await self.launch()
        await self.child_runs()                                                       # the total: 21
        for k, n in enumerate(extra_finds):
            await self.child_runs(cmd=f"find /part{k} -type f", output="\n".join(["x"] * n),
                                  tuid=f"toolu_part{k}", parent=LAUNCH_TUID)
        await self.finish(summary)
        return self.led.tasks[TID]["result_verification"]

    async def test_a_correct_total_with_unsupported_breakdown_numbers_is_verified(self):
        for k, summary in enumerate(("21 files total: 8 + 12 + 1", "21 files (8 + 13)",
                                     "21 daily note files: 8 files, 12 files and 1 file")):
            tid, tuid = f"t{k}", f"toolu_l{k}"
            await self.launch(tid=tid, tuid=tuid)
            await self.child_runs(tuid=f"toolu_c{k}", parent=tuid)
            await self.finish(summary, tid=tid, tuid=tuid)
            v = self.led.tasks[tid]["result_verification"]
            self.assertEqual(v["state"], L.V_VERIFIED, summary)
            self.assertNotEqual(v["state"], L.V_FAILED, summary)
            self.assertEqual([m["n"] for m in v["matched"]], [21], summary)
            self.assertEqual(v["failed"], [], summary)
            self.assertTrue(v["breakdown_unchecked"], summary)
            self.assertEqual(v["breakdown_matched"], [], summary)
            self.assertIn("no supporting evidence", v["evidence"])
            line = SP.task_line(self.led.tasks[tid])
            self.assertIn("It reported 21", line)
            for part in ("8", "12", "13", "1 "):
                self.assertNotRegex(line, rf"\b{part.strip()}\b(?! files)" if part == "1 " else rf"\b{part}\b")

    async def test_a_wrong_total_is_still_a_mismatch(self):
        v = await self.verdict_for("20 files total")
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))
        self.assertNotRegex(SP.task_line(self.led.tasks[TID]), r"\b20\b")

    async def test_the_exact_live_failure_now_fails_on_the_20_only(self):
        v = await self.verdict_for(self.LIVE_SHAPE)
        self.assertEqual((v["state"], v["facts"]), (L.V_FAILED, [21]))
        self.assertEqual(v["failed"], [20])                                 # was [1, 8, 12, 20]
        self.assertEqual(v["breakdown_unchecked"], [1, 8, 12])
        line = SP.task_line(self.led.tasks[TID])
        self.assertIn("didn't match", line)
        self.assertIn("which was 21", line)
        self.assertNotRegex(line, r"\b(1|8|12|20)\b")

    async def test_the_original_incident_reproduction_still_fails(self):
        v = await self.verdict_for(WRONG_SUMMARY)
        self.assertEqual((v["state"], v["facts"], v["failed"]), (L.V_FAILED, [21], [20]))

    async def test_the_incident_summary_with_a_scope_note_still_fails(self):
        v = await self.verdict_for("20 daily note files found (September 2026 subfolder only)")
        self.assertEqual((v["state"], v["failed"]), (L.V_FAILED, [20]))

    async def test_a_breakdown_the_evidence_supports_is_verified(self):
        v = await self.verdict_for("21 files: 8 + 12 + 1", extra_finds=(8, 12, 1))
        self.assertEqual(v["state"], L.V_VERIFIED)
        self.assertEqual((v["breakdown_matched"], v["breakdown_unchecked"], v["failed"]), ([1, 8, 12], [], []))
        self.assertIn("is supported by the tool output", v["evidence"])

    async def test_a_wrong_breakdown_the_evidence_can_refute_is_a_mismatch(self):
        v = await self.verdict_for("21 files: 8 + 12 + 2", extra_finds=(8, 12, 1))
        self.assertEqual(v["state"], L.V_FAILED)
        self.assertEqual(v["failed"], [2])
        self.assertEqual(v["breakdown_matched"], [8, 12])
        self.assertIn("didn't match", SP.task_line(self.led.tasks[TID]))

    async def test_evidence_that_does_not_add_up_to_the_total_supports_no_breakdown(self):
        # audit F-13 (this used to expect VERIFIED): the tools counted 21 AND two unrelated numbers (8, 5) that do not add up to it, so a
        # summary total of 21 equals one of several counts without being tied to a result. Not a failure, but not verified either.
        v = await self.verdict_for("21 files: 8 + 13", extra_finds=(8, 5))
        self.assertEqual((v["state"], v["failed"]), (L.V_UNVERIFIED, []))
        self.assertEqual(v["ambiguous"], [21])
        self.assertEqual(v["breakdown_unchecked"], [8, 13])

    async def test_parts_that_do_add_up_to_the_total_keep_it_verified(self):
        v = await self.verdict_for("21 files: 8 + 13", extra_finds=(8, 13))
        self.assertEqual((v["state"], v["failed"], v["ambiguous"]), (L.V_VERIFIED, [], []))

    async def test_states_are_not_collapsed(self):
        seen = {}
        for k, (summary, finds) in enumerate((("21 files (8 + 13)", ()), ("20 files total", ()),
                                             ("split (8 files) and (12 files)", ()), ("All present.", ()))):
            tid, tuid = f"s{k}", f"toolu_s{k}"
            await self.launch(tid=tid, tuid=tuid)
            await self.child_runs(tuid=f"toolu_sc{k}", parent=tuid)
            await self.finish(summary, tid=tid, tuid=tuid)
            seen[summary] = self.led.tasks[tid]["result_verification"]["state"]
        self.assertEqual(seen, {"21 files (8 + 13)": L.V_VERIFIED, "20 files total": L.V_FAILED,
                                "split (8 files) and (12 files)": L.V_UNVERIFIED, "All present.": L.V_NOT_REQUIRED})


if __name__ == "__main__":
    unittest.main()
