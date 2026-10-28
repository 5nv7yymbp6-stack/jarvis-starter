"""Phase 2: the message router (backtalk/router.py). No network, no CLI: a fake
client replays message sequences shaped like the ones captured in the
2026-09-30 spikes. Stdlib unittest."""
import asyncio
import json
import tempfile
import unittest

from backtalk import ledger as L
from backtalk import router as R
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.client = FakeClient()
        self.mgr = R.EpochManager(self.led, echo_timeout_s=0.25,
                                  barge_escalate_s=0.15, proc=False)
        self.r = await self.mgr.open(self.client, session_id="s", cli_pid=None)
        self.extra = []

    async def asyncTearDown(self):
        for r in [self.r] + self.extra:
            for t in (r._pump_task, r._watch_task, *r._escalations):
                if t:
                    t.cancel()
        await asyncio.sleep(0)
        self.led.close()
        self.tmp.cleanup()

    async def fg(self, text, answer, **kw):
        h = await self.r.send(text, **kw)
        self.client.feed(*F.fg_run(h.uuid, answer))
        await settle()
        return h

    async def drain(self, h):
        out = []
        async def go():
            async for t, m in h.events():
                out.append((t, m))
        await asyncio.wait_for(go(), 2)
        return out

    def result_text(self, h):
        return h.result.result if h.result else None

    def task(self, tid):
        return self.led.tasks[tid]


class ForegroundTests(Base):
    async def test_1_uuid_correlation_claims_only_on_echo(self):
        h = await self.r.send("hello")
        self.assertEqual(h.state, L.T_SENT)
        self.client.feed(F.init())
        await settle()
        self.assertEqual(h.state, L.T_SENT)                  # init is NOT a claim
        self.client.feed(F.echo(h.uuid), F.delta("Hi"), F.asst("Hi"), F.result("Hi"))
        await settle()
        self.assertEqual(h.state, L.T_DONE)
        self.assertEqual(self.led.turns[h.turn_id]["state"], L.T_DONE)

    async def test_2_result_is_attributed_to_its_own_turn(self):
        h = await self.fg("q", "the answer")
        ev = await self.drain(h)
        self.assertEqual([t for t, _ in ev], ["msg", "msg", "result"])
        self.assertEqual(self.result_text(h), "the answer")

    async def test_3_back_to_back_queries_get_their_own_answers(self):
        h1 = await self.r.send("one")
        h2 = await self.r.send("two")
        self.client.feed(*F.fg_run(h1.uuid, "alpha"))
        self.client.feed(*F.fg_run(h2.uuid, "beta"))
        await settle()
        self.assertEqual(self.result_text(h1), "alpha")
        self.assertEqual(self.result_text(h2), "beta")
        self.assertEqual((h1.state, h2.state), (L.T_DONE, L.T_DONE))

    async def test_result_without_a_claim_never_completes_a_waiting_turn(self):
        h = await self.r.send("waiting")
        self.client.feed(F.init(), F.asst("stray"), F.result("stray"))
        await settle()
        self.assertEqual(h.state, L.T_SENT)
        self.assertEqual(self.mgr.quarantine.counts["unowned_result"], 1)
        self.assertIsNone(h.result)

    async def test_a_self_woken_result_is_never_spoken_or_paired_but_is_no_longer_silent(self):
        # 2026-10-15 22:01: a model-scheduled "checking back in ten minutes" turn fired with no waiting echo; its result
        # was quarantined and nothing marked that it had ever existed. It must still never pair with a waiting turn or
        # reach speech, but its text size is now a ledger note and one record-only report.
        h = await self.r.send("waiting")
        self.client.feed(F.init(), F.asst("Ten minutes are up, still no word from the sessions."),
                         F.result("Ten minutes are up, still no word from the sessions."))
        await settle()
        self.assertEqual(h.state, L.T_SENT)                              # never paired with the waiting turn
        self.assertIsNone(h.result)
        self.assertEqual(self.mgr.quarantine.counts["unowned_result"], 1)  # still quarantined
        self.assertEqual(self.r.stats["unowned_report"], 1)
        reports = self.r.pending_reports()
        self.assertEqual(len(reports), 1)
        self.assertFalse(reports[0].synthetic)
        notes = [n for n in self.led.notes if n.get("note") == "unowned_result_text"]
        self.assertEqual(len(notes), 1)
        self.assertGreater(notes[0]["chars"], 20)
        self.assertNotIn("Ten minutes", json.dumps(notes))                # the note carries a size, never the words

    async def test_an_empty_stray_result_records_nothing_extra(self):
        h = await self.r.send("waiting")
        self.client.feed(F.init(), F.result(""))
        await settle()
        self.assertEqual(self.r.stats["unowned_report"], 0)
        self.assertEqual(self.r.pending_reports(), [])

    async def test_error_result_fails_the_turn_with_a_reason(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.result("", subtype="error_max_turns", is_error=True,
                                  terminal="max_turns"))
        await settle()
        self.assertEqual(h.state, L.T_FAILED)
        with self.assertRaises(R.TurnFailed) as cm:
            await self.drain(h)
        self.assertEqual(cm.exception.reason, "max_turns")

    async def test_18_no_echo_fails_the_turn_never_borrows_a_result(self):
        h = await self.r.send("lonely")
        await asyncio.sleep(0.9)                              # watchdog ticks every 0.5s
        self.assertEqual(h.state, L.T_FAILED)
        self.assertEqual(h.fail_reason, "no_echo")

    async def test_streaming_events_reach_only_the_owner(self):
        h = await self.r.send("s")
        self.client.feed(F.init(), F.delta("early"))          # before any claim
        await settle()
        self.assertTrue(h.q.empty())
        self.client.feed(F.echo(h.uuid), F.delta("late"))
        await settle()
        self.assertEqual(h.q.qsize(), 1)

    async def test_only_one_reader_of_the_stream(self):
        await self.fg("a", "1")
        await self.fg("b", "2")
        self.assertEqual(self.client.reader_calls, 1)
        self.assertEqual(self.client.max_readers, 1)


class FranceItalyTests(Base):
    async def test_14_the_original_desync_cannot_happen(self):
        # 1) a background task is launched and its turn returns
        h0 = await self.r.send("start job")
        self.client.feed(F.init(), F.echo(h0.uuid),
                         *F.launch_bg("tk1", "toolu_bg"), F.asst("launched"), F.result("launched"))
        await settle()
        self.assertEqual(h0.state, L.T_DONE)
        # 2) it finishes while NOBODY has a turn open: an unsolicited run
        self.client.feed(F.task_note("tk1", "toolu_bg", "completed", "GAMMA=1234"),
                         *F.unsolicited_run("GAMMA=1234"))
        await settle()
        # 3) only now do questions arrive
        france = await self.r.send("capital of France?")
        self.client.feed(*F.fg_run(france.uuid, "Paris"))
        await settle()
        italy = await self.r.send("capital of Italy?")
        self.client.feed(*F.fg_run(italy.uuid, "Rome"))
        await settle()
        self.assertEqual(self.result_text(france), "Paris")
        self.assertEqual(self.result_text(italy), "Rome")
        texts = [rep.text for rep in self.r.pending_reports()]
        self.assertTrue(any("GAMMA" in t for t in texts))

    async def test_unsolicited_run_between_send_and_echo_does_not_steal_the_turn(self):
        h = await self.r.send("capital of France?")
        self.client.feed(*F.unsolicited_run("GAMMA=1234"))    # arrives first, queued ahead
        self.client.feed(*F.fg_run(h.uuid, "Paris"))
        await settle()
        self.assertEqual(self.result_text(h), "Paris")
        self.assertEqual(len(self.r.pending_reports()), 1)


class BackgroundTests(Base):
    async def start_bg(self, tid="tk1", tuid="toolu_bg1", desc="job"):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg(tid, tuid, desc),
                         F.asst("launched"), F.result("launched"))
        await settle()
        return h

    async def test_4_background_completion_is_a_task_event_not_an_answer(self):
        seen = []
        self.mgr.tasks.listeners.append(lambda row: seen.append(row["id"]))
        await self.start_bg()
        self.assertEqual(self.task("tk1")["task_kind"], L.KIND_BG_AGENT)
        self.assertEqual(self.task("tk1")["state"], L.K_RUNNING)
        self.client.feed(F.task_updated("tk1", "completed"),
                         F.task_note("tk1", "toolu_bg1", "completed", "DONE-A"),
                         *F.unsolicited_run("Background agent finished: DONE-A"))
        await settle()
        self.assertEqual(self.task("tk1")["state"], L.K_COMPLETED)
        self.assertEqual(seen, ["tk1"])                       # duplicate signals fire once
        self.assertEqual(len(self.r.pending_reports()), 1)

    async def test_5_multiple_tasks_finish_in_arbitrary_order(self):
        h = await self.r.send("two jobs")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("a", "toolu_a"),
                         *F.launch_bg("b", "toolu_b"), F.asst("launched"), F.result("launched"))
        await settle()
        self.client.feed(F.task_note("b", "toolu_b", "completed", "B"),
                         *F.unsolicited_run("b done"))
        await settle()
        self.assertEqual(self.task("b")["state"], L.K_COMPLETED)
        self.assertEqual(self.task("a")["state"], L.K_RUNNING)
        self.client.feed(F.task_note("a", "toolu_a", "completed", "A"),
                         *F.unsolicited_run("a done"))
        await settle()
        self.assertEqual(self.task("a")["state"], L.K_COMPLETED)

    async def test_6_stop_task_a_while_b_continues(self):
        h = await self.r.send("two jobs")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("a", "toolu_a"),
                         *F.launch_bg("b", "toolu_b"), F.asst("launched"), F.result("launched"))
        await settle()
        intents = []

        def on_stop(tid):
            intents.append(self.task(tid)["cancel_intent"])   # recorded BEFORE the call
            self.client.feed(F.task_updated(tid, "killed"),
                             F.task_note(tid, "toolu_a", "stopped", "a"))
        self.client.on_stop = on_stop
        self.assertTrue(await self.r.stop_task("a", "USER_STOP"))
        await settle()
        self.assertEqual(intents, ["USER_STOP"])
        self.assertEqual(self.task("a")["state"], L.K_STOPPED_USER)
        self.assertEqual(self.task("b")["state"], L.K_RUNNING)
        self.assertEqual(self.client.interrupts, 0)

    async def test_stop_of_an_unexplained_task_is_not_called_user_stopped(self):
        await self.start_bg()
        self.client.feed(F.task_updated("tk1", "killed"), F.task_note("tk1", "toolu_bg1", "stopped"))
        await settle()
        self.assertEqual(self.task("tk1")["state"], L.K_STOPPED_UNATTRIBUTED)

    async def test_7_stop_task_during_foreground_leaves_the_run_alive(self):
        await self.start_bg()
        h = await self.r.send("long question")
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("Work"))
        await settle()
        self.client.on_stop = lambda tid: self.client.feed(
            F.task_updated(tid, "killed"), F.task_note(tid, "toolu_bg1", "stopped"),
            F.echo("inj", "<task-notification>\n<task-id>tk1</task-id>\n</task-notification>"))
        await self.r.stop_task("tk1", "USER_STOP")
        await settle()
        self.assertEqual(h.state, L.T_RUNNING)                # not touched
        self.assertEqual(h.contaminated_by, ["tk1"])          # but flagged
        self.client.feed(F.asst("Work done"), F.result("Work done"))
        await settle()
        self.assertEqual(h.state, L.T_DONE)
        self.assertEqual(self.task("tk1")["state"], L.K_STOPPED_USER)

    async def test_task_kinds_are_told_apart(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.tool_use("toolu_fg", "Bash", {"command": "sleep 1"}),
                         F.task_started("fgtask", "toolu_fg", "Sleep"),
                         F.task_started("inner", "toolu_in", "inner", owned=True),
                         *F.launch_bg("bg", "toolu_b"))
        await settle()
        self.assertEqual(self.task("fgtask")["task_kind"], L.KIND_FG_TOOL)
        self.assertEqual(self.task("inner")["task_kind"], L.KIND_SUBAGENT)
        self.assertEqual(self.task("bg")["task_kind"], L.KIND_BG_AGENT)
        self.assertEqual({t["id"] for t in self.r.live_bg()}, {"bg"})

    async def test_subagent_messages_never_reach_the_foreground(self):
        await self.start_bg()
        h = await self.r.send("q")
        self.client.feed(F.init(), F.asst("SUBAGENT TEXT", parent="toolu_bg1"),
                         F.echo(h.uuid), F.delta("mine"), F.asst("mine"), F.result("mine"))
        await settle()
        texts = [getattr(b, "text", "") for t, m in await self.drain(h)
                 for b in getattr(m, "content", []) or []]
        self.assertNotIn("SUBAGENT TEXT", texts)

    async def test_tool_lifecycle_lands_in_the_ledger_not_as_executed(self):
        rows = []
        self.r.action_listeners.append(lambda row, inp: rows.append((row["state"], inp)))
        h = await self.r.send("write")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.tool_use("toolu_w", "Write", {"file_path": "/tmp/x", "content": "hi"}))
        await settle()
        a = self.led.actions[self.led.action_id_for("toolu_w")]
        self.assertEqual(a["state"], L.A_REQUESTED)
        self.client.feed(F.tool_result("toolu_w", "wrote"))
        await settle()
        self.assertEqual(self.led.actions[a["id"]]["state"], L.A_OK)
        self.assertEqual(rows[0][1]["content"], "hi")
        self.client.feed(F.tool_use("toolu_e", "Bash", {}), F.tool_result("toolu_e", "boom", is_error=True))
        await settle()
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_e")]["state"], L.A_ERROR)


class FailureClassificationTests(Base):
    """Phase C: a tool error is classified once, from its transient text; only the CODE is stored."""

    async def run_tool_error(self, text, tool="Bash"):
        h = await self.r.send("do it")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_c", tool, {}),
                         F.tool_result("toolu_c", text, is_error=True))
        await settle()
        return self.led.actions[self.led.action_id_for("toolu_c")]

    async def test_the_class_is_stored_and_the_error_text_is_not(self):
        row = await self.run_tool_error("curl: (28) Operation timed out after 30000 milliseconds, SECRETTEXT-9f3")
        self.assertEqual(row["failure_class"], "RETRYABLE")
        self.assertNotIn("SECRETTEXT-9f3", json.dumps(row))
        with open(self.led.path, "r", encoding="utf-8") as fh:
            self.assertNotIn("SECRETTEXT-9f3", fh.read())

    async def test_each_family_is_classified(self):
        for text, want in (("permission denied: /etc/x", "PERMISSION_REQUIRED"), ("No such file or directory", "INVALID_INPUT"),
                           ("please sign in again", "USER_ACTION_REQUIRED"), ("boom", "TOOL_ERROR")):
            row = await self.run_tool_error(text)
            self.assertEqual(row["failure_class"], want, text)
            self.led.actions.clear()
            self.led._by_tool_use.clear()

    async def test_a_successful_result_gets_no_failure_class(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_ok", "Read", {}), F.tool_result("toolu_ok", "fine"))
        await settle()
        self.assertNotIn("failure_class", self.led.actions[self.led.action_id_for("toolu_ok")])

    async def test_a_failed_turn_carries_its_turn_id_for_the_recovery_decision(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid), F.result("", subtype="error_max_turns", is_error=True, terminal="max_turns"))
        await settle()
        with self.assertRaises(R.TurnFailed) as cm:
            await self.drain(h)
        self.assertEqual(cm.exception.turn_id, h.turn_id)


class InterruptTests(Base):
    def wire_interrupt(self, kill=()):
        def on_int():
            for tid, tuid in kill:
                self.client.feed(F.task_updated(tid, "killed"), F.task_note(tid, tuid, "stopped"))
            self.client.feed(F.result("", is_error=True, subtype="error_during_execution",
                                      terminal="aborted_tools"))
        self.client.on_interrupt = on_int

    async def test_8_interrupt_foreground_then_new_query_is_not_confused(self):
        h = await self.r.send("long")
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("Wor"))
        await settle()
        self.wire_interrupt()
        self.assertEqual(await self.r.interrupt(), "aborted")
        self.assertEqual(h.state, L.T_INTERRUPTED)
        h2 = await self.r.send("next")                        # immediately
        self.client.feed(*F.fg_run(h2.uuid, "NEWQ-ok"))
        await settle()
        self.assertEqual(self.result_text(h2), "NEWQ-ok")
        self.assertEqual(h.state, L.T_INTERRUPTED)            # the old one stayed old

    async def test_12_stale_aborted_result_closes_the_old_turn_only(self):
        h = await self.r.send("long")
        self.client.feed(F.init(), F.echo(h.uuid))
        await settle()
        h2_holder = []

        def on_int():                        # the aborted result arrives AFTER the next send
            pass
        self.client.on_interrupt = on_int
        t = asyncio.create_task(self.r.interrupt(wait=1))
        await settle()
        h2 = await self.r.send("next")
        self.client.feed(F.result("", is_error=True, subtype="error_during_execution",
                                  terminal="aborted_tools"))
        await t
        self.client.feed(*F.fg_run(h2.uuid, "fresh"))
        await settle()
        self.assertEqual(h.state, L.T_INTERRUPTED)
        self.assertEqual(self.result_text(h2), "fresh")

    async def test_9_interrupt_with_background_tasks_records_why_they_died(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        self.r.protect_background = False
        self.wire_interrupt(kill=[("tk1", "toolu_bg1")])
        self.assertEqual(await self.r.interrupt(reason="INTERRUPT"), "aborted")
        await settle()
        self.assertEqual(self.task("tk1")["state"], L.K_KILLED_INTERRUPT)

    async def test_barge_in_with_protected_task_mutes_and_does_not_interrupt(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        self.assertTrue(self.task("tk1")["protected"])
        res = await self.r.interrupt(reason="BARGE_IN")
        self.assertEqual(res, "protected")
        self.assertEqual(self.client.interrupts, 0)
        self.assertEqual(self.task("tk1")["state"], L.K_RUNNING)
        self.assertTrue(h.abandoned)
        # the abandoned run finishes on its own and is ledgered, never spoken
        self.client.feed(F.asst("late"), F.result("late"))
        await settle()
        self.assertEqual(h.state, L.T_DONE)

    async def test_protected_barge_in_escalates_after_the_bound(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        self.wire_interrupt(kill=[("tk1", "toolu_bg1")])
        await self.r.interrupt(reason="BARGE_IN")
        await asyncio.sleep(0.45)
        self.assertEqual(self.client.interrupts, 1)
        self.assertEqual(self.task("tk1")["state"], L.K_KILLED_INTERRUPT)
        self.assertEqual(h.state, L.T_INTERRUPTED)

    async def test_stop_everything_stops_tasks_by_name_then_the_run(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("a", "toolu_a"),
                         *F.launch_bg("b", "toolu_b"))
        await settle()
        order = []
        self.client.on_stop = lambda tid: (order.append(("stop", tid)), self.client.feed(
            F.task_updated(tid, "killed"), F.task_note(tid, "t", "stopped")))
        self.client.on_interrupt = lambda: (order.append(("interrupt",)), self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools")))
        n, res = await self.r.stop_all()
        await settle()
        self.assertEqual(n, 2)
        self.assertEqual(order[-1], ("interrupt",))
        self.assertEqual(self.task("a")["state"], L.K_STOPPED_USER)
        self.assertEqual(self.task("b")["state"], L.K_STOPPED_USER)
        self.assertEqual(h.state, L.T_INTERRUPTED)

    async def test_abandoned_unclaimed_query_is_stale_when_it_finally_runs(self):
        h = await self.r.send("never mind")
        self.assertEqual(await self.r.interrupt(), "none")
        self.assertEqual(h.state, L.T_ABORTED)
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("x"), F.asst("x"), F.result("x"))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 1)
        self.assertEqual(h.state, L.T_ABORTED)
        self.assertGreaterEqual(self.client.interrupts, 1)    # no live bg tasks: stopped


class SyntheticAndEpochTests(Base):
    async def rebuild(self, reason="desync", proc_check=None, fail=False):
        old = self.r
        self.client.fail_disconnect = fail
        ok = await old.close(reason, self.client.disconnect, proc_check=proc_check)
        self.client2 = FakeClient()
        self.r = await self.mgr.open(self.client2, session_id="s", cli_pid=None)
        self.extra.append(old)
        return ok

    async def test_10_synthetic_lost_task_report_after_rebuild_is_silent_and_inert(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        self.assertTrue(await self.rebuild())
        self.assertEqual(self.task("tk1")["state"], L.K_LOST_REBUILD)
        self.client2.feed(F.task_note("tk1", "toolu_bg1", "stopped", "No completion record was found"),
                          *F.synthetic_run())
        await settle()
        self.assertEqual(self.task("tk1")["state"], L.K_LOST_REBUILD)      # state untouched
        self.assertEqual(self.task("tk1")["cli_reported"], "stopped")
        reps = self.r.pending_reports()
        self.assertEqual([x.synthetic for x in reps], [True])
        # and the first real question gets its own answer, not the empty run
        h2 = await self.r.send("hello?")
        self.client2.feed(*F.fg_run(h2.uuid, "Hello"))
        await settle()
        self.assertEqual(self.result_text(h2), "Hello")

    async def test_11_old_epoch_messages_never_reach_the_new_epoch(self):
        h = await self.r.send("q")
        old = self.r
        await self.rebuild()
        self.assertEqual(h.state, L.T_ABORTED)
        n = self.mgr.quarantine.counts["message_after_epoch_end"]
        old.dispatch(F.echo(h.uuid))                           # a late straggler
        old.dispatch(F.result("late"))
        self.assertEqual(self.mgr.quarantine.counts["message_after_epoch_end"], n + 2)
        h2 = await self.r.send("new")
        self.assertIsNone(h2.result)
        # the dead handle's uuid means nothing to the new epoch
        self.client2.feed(F.init(), F.echo(h.uuid, "old"), F.result("late"))
        await settle()
        self.assertEqual(h2.state, L.T_SENT)
        self.assertGreaterEqual(self.mgr.quarantine.counts["unmatched_user_message"], 1)

    async def test_clean_rebuild_marks_epoch_and_tasks(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        old_epoch = self.r.epoch
        await self.rebuild(reason="desync")
        self.assertEqual(self.led.epochs[old_epoch]["state"], L.EPOCH_CLOSED_CLEAN)
        self.assertEqual(self.led.epochs[old_epoch]["close_reason"], "desync")
        self.assertEqual(self.task("tk1")["state"], L.K_LOST_REBUILD)
        self.assertEqual(self.task("tk1")["cancel_intent"], "REBUILD")

    async def test_failed_disconnect_is_a_crash_not_a_clean_loss(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        old_epoch = self.r.epoch
        ok = await self.rebuild(fail=True)
        self.assertFalse(ok)
        self.assertEqual(self.task("tk1")["state"], L.K_UNKNOWN_CRASH)
        self.assertEqual(self.led.epochs[old_epoch]["state"], L.EPOCH_CRASHED)

    async def test_process_still_alive_after_clean_close_is_not_called_lost(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        self.led.task_set("tk1", process_snapshot=[{"pid": 1234, "lstart": "x"}])
        await self.rebuild(proc_check=lambda pid, ls: True)
        self.assertEqual(self.task("tk1")["state"], L.K_ORPHAN_RUNNING)

    async def test_17_crash_between_runs_is_detected_by_the_next_process(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"))
        await settle()
        # simulate the process dying: no close(); a new process opens the ledger
        self.led.close()
        led2 = L.Ledger(self.tmp.name, fsync=False)
        mgr2 = R.EpochManager(led2, proc=False)
        out = mgr2.reconcile_once(proc_check=lambda pid, ls: None)
        self.assertEqual(out["crashed_epochs"], 1)
        self.assertEqual(led2.tasks["tk1"]["state"], L.K_UNKNOWN_CRASH)
        led2.close()
        self.led = L.Ledger(self.tmp.name, fsync=False)

    async def test_reconcile_runs_once_per_process(self):
        a = self.mgr.reconcile_once()
        self.assertIs(a, self.mgr.reconcile_once())


class FailureTests(Base):
    async def test_15_stream_failure_fails_turns_and_loses_pending_actions(self):
        lost = []
        self.r.on_lost = lost.append
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_x", "Bash", {}))
        await settle()
        self.client.feed(RuntimeError("Command failed with exit code -9"))
        await settle()
        self.assertEqual(h.state, L.T_FAILED)
        self.assertEqual(h.fail_reason, "stream_lost")
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_x")]["state"], L.A_LOST)
        self.assertEqual(len(lost), 1)
        self.assertFalse(self.r.usable)
        with self.assertRaises(R.RouterClosed):
            await self.r.send("again")

    async def test_16_oversized_tool_result_is_named_as_such(self):
        h = await self.r.send("read the huge file")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_big", "Read", {}))
        await settle()
        self.client.feed(ValueError("Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes"))
        await settle()
        self.assertEqual(h.fail_reason, "oversize")
        self.assertEqual(self.led.actions[self.led.action_id_for("toolu_big")]["state"], L.A_LOST)
        with self.assertRaises(R.TurnFailed) as cm:
            await self.drain(h)
        self.assertEqual(cm.exception.reason, "oversize")

    async def test_stream_end_is_a_loss_too(self):
        h = await self.r.send("go")
        self.client.eof()
        await settle()
        self.assertEqual(h.state, L.T_FAILED)

    async def test_13_unmatched_messages_are_quarantined_and_bounded(self):
        q = self.mgr.quarantine
        for i in range(500):
            self.client.feed(F.echo(f"nobody-{i}", "stray"))
        await settle(20)
        self.assertEqual(q.counts["unmatched_user_message"], 500)
        self.assertEqual(len(q.ring), 200)                    # bounded
        self.assertEqual(self.r.state, "OPEN")                # and the reader survived

    async def test_a_router_bug_does_not_stop_the_reader(self):
        orig = self.r._on_assistant

        def boom(m):
            raise ValueError("bug")
        self.r._on_assistant = boom
        self.client.feed(F.asst("x"))
        await settle()
        self.r._on_assistant = orig
        h = await self.r.send("still alive?")
        self.client.feed(*F.fg_run(h.uuid, "yes"))
        await settle()
        self.assertEqual(self.result_text(h), "yes")

    async def test_unsolicited_result_inside_a_foreground_run_does_not_end_it(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("a"),
                         F.result("x", origin="task-notification"))
        await settle()
        self.assertEqual(h.state, L.T_RUNNING)
        self.assertEqual(self.mgr.quarantine.counts["task_result_inside_foreground_run"], 1)


class CommandTests(Base):
    async def test_command_reply_before_echo_is_claimed_retroactively(self):
        h = await self.r.send("/effort low", kind="command")
        self.client.feed(F.init(), F.asst("Set effort level to low"), F.echo(h.uuid, "<command-name>"),
                         F.result("Set effort level to low", num_turns=0))
        await settle()
        self.assertEqual(h.state, L.T_DONE)
        ev = await self.drain(h)
        self.assertEqual([t for t, _ in ev], ["msg", "result"])

    async def test_a_foreground_turn_never_claims_content_that_came_before_its_echo(self):
        h = await self.r.send("hello")
        self.client.feed(F.init(), F.asst("who am i"), F.echo(h.uuid), F.asst("real"), F.result("real"))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["content_before_echo"], 1)
        texts = [getattr(b, "text", "") for t, m in await self.drain(h)
                 for b in getattr(m, "content", []) or []]
        self.assertEqual(texts, ["real"])


if __name__ == "__main__":
    unittest.main()


class ProcessEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Read-only process evidence, using real child processes that THIS TEST
    starts and stops itself. Nothing else on the machine is looked at."""

    async def test_proc_tree_sees_our_child_and_process_alive_matches_it_exactly(self):
        import os
        import subprocess
        child = subprocess.Popen(["sleep", "30"])
        try:
            await asyncio.sleep(0.2)
            tree = R.proc_tree(os.getpid())
            mine = [p for p in tree if p["pid"] == child.pid]
            self.assertEqual(len(mine), 1)
            self.assertTrue(L.process_alive(child.pid, mine[0]["lstart"]))
            self.assertFalse(L.process_alive(child.pid, "Thu Jan  1 00:00:00 1970"))   # same pid, other process
        finally:
            child.terminate()
            child.wait()
        self.assertFalse(L.process_alive(child.pid, mine[0]["lstart"]))

    async def test_task_processes_are_recorded_for_backtalks_own_bg_tasks_only(self):
        import os
        import subprocess
        tmp = tempfile.TemporaryDirectory()
        led = L.Ledger(tmp.name, fsync=False)
        client = FakeClient()
        mgr = R.EpochManager(led, proc=True)
        r = await mgr.open(client, session_id="s", cli_pid=os.getpid())
        pre = subprocess.Popen(["sleep", "30"])          # existed before the task: baseline
        try:
            await asyncio.sleep(0.2)
            await r.capture_baseline()
            h = await r.send("go")
            client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_1"))
            await settle()
            task_child = subprocess.Popen(["sleep", "31"])   # appears after the task began
            await asyncio.sleep(0.2)
            client.feed(F.task_progress("tk1", "toolu_1"))
            for _ in range(40):
                await asyncio.sleep(0.05)
                if led.tasks["tk1"].get("process_snapshot"):
                    break
            pids = {p["pid"] for p in led.tasks["tk1"]["process_snapshot"]}
            self.assertIn(task_child.pid, pids)
            self.assertNotIn(pre.pid, pids)                  # baseline excluded
            self.assertEqual(led.tasks["tk1"]["process_snapshot_confidence"], "approximate")
            task_child.terminate(); task_child.wait()
        finally:
            pre.terminate(); pre.wait()
            for t in (r._pump_task, r._watch_task):
                t.cancel()
            await asyncio.sleep(0)
            led.close(); tmp.cleanup()


class CloseIsBoundedTests(Base):
    async def test_a_reader_that_will_not_die_cannot_hang_close(self):
        async def stubborn():
            ignored = 0
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    ignored += 1
                    if ignored > 1:              # ignores the first cancel only, so the
                        return                   # test's own teardown can still finish it
        self.r._pump_task.cancel()
        self.r._pump_task = asyncio.create_task(stubborn())
        await asyncio.sleep(0.05)                # let it start, so cancel() really hits it
        t0 = asyncio.get_running_loop().time()
        ok = await self.r.close("test", self.client.disconnect)
        self.assertTrue(ok)
        took = asyncio.get_running_loop().time() - t0
        self.assertGreater(took, 2.5)            # it really waited for the stubborn task...
        self.assertLess(took, 5)                 # ...and then gave up
        self.assertEqual(self.led.epochs[self.r.epoch]["state"], L.EPOCH_CLOSED_CLEAN)


class MergedRunTests(Base):
    """The CLI can deliver a queued user message INTO a run that is still
    working (seen against the live CLI, 2026-09-30)."""

    async def test_query_sent_during_an_abandoned_run_takes_it_over(self):
        a = await self.r.send("long request")
        self.client.feed(F.init(), F.echo(a.uuid), F.delta("Working"))
        await settle()
        a.abandoned = True                                    # the user barged in; A is muted
        b = await self.r.send("next question")
        self.client.feed(F.echo(b.uuid, "next question"), F.delta("B answer."),
                         F.asst("B answer."), F.result("B answer."))
        await settle()
        self.assertEqual(a.state, L.T_ABORTED)
        self.assertEqual(a.fail_reason, "superseded_by_next_turn")
        self.assertEqual(b.state, L.T_DONE)
        self.assertEqual(self.result_text(b), "B answer.")
        ev = await self.drain(b)
        self.assertEqual([t for t, _ in ev], ["msg", "msg", "result"])       # only what came after its echo
        self.assertTrue(self.led.turns[b.turn_id]["merged_into_run"])
        self.assertTrue(any(n.get("note") == "turn_merged_into_run" for n in self.led.notes))

    async def test_a_run_still_being_consumed_is_never_taken(self):
        a = await self.r.send("live one")
        self.client.feed(F.init(), F.echo(a.uuid), F.delta("A speaking"))
        await settle()
        b = await self.r.send("intruder")
        self.client.feed(F.echo(b.uuid, "intruder"), F.delta("more"), F.asst("x"), F.result("x"))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["echo_inside_owned_run"], 1)
        self.assertEqual(a.state, L.T_DONE)                   # A's own run, A's own result
        self.assertEqual(b.state, L.T_SENT)                   # never handed A's answer
        self.assertIsNone(b.result)

    async def test_query_delivered_into_an_unsolicited_report_run_takes_it_over(self):
        b = await self.r.send("question")
        self.client.feed(F.init(), F.asst("Background report text"))       # unsolicited, unowned
        await settle()
        self.client.feed(F.task_note("t", "tu", "completed"))              # no ledger row: quarantined, harmless
        self.client.feed(F.echo("nobody"), )                               # unmatched: quarantined
        self.r.run.owner = R.UNSOLICITED                                   # its origin marker was seen
        self.client.feed(F.echo(b.uuid, "question"), F.delta("Real answer."), F.asst("Real answer."),
                         F.result("Real answer."))
        await settle()
        self.assertEqual(b.state, L.T_DONE)
        self.assertEqual(self.result_text(b), "Real answer.")

    async def test_a_taken_over_run_ends_once_and_the_next_query_is_clean(self):
        a = await self.r.send("long")
        self.client.feed(F.init(), F.echo(a.uuid))
        await settle()
        a.abandoned = True
        b = await self.r.send("two")
        self.client.feed(F.echo(b.uuid, "two"), F.asst("two."), F.result("two."))
        await settle()
        c = await self.r.send("three")
        self.client.feed(*F.fg_run(c.uuid, "three."))
        await settle()
        self.assertEqual([self.result_text(b), self.result_text(c)], ["two.", "three."])
        self.assertIsNone(self.r.run)

    async def test_interrupted_owner_hands_over_too(self):
        a = await self.r.send("long")
        self.client.feed(F.init(), F.echo(a.uuid))
        await settle()
        a.interrupt_requested = True
        b = await self.r.send("two")
        self.client.feed(F.echo(b.uuid, "two"), F.asst("two."), F.result("two."))
        await settle()
        self.assertEqual(self.result_text(b), "two.")


NOTICE = "<task-notification>\n<task-id>tk1</task-id>\n<status>completed</status>\n</task-notification>"


class MergedTurnContaminationTests(Base):
    """Audit 2026-10-01: a task notice that reached the run BEFORE the foreground
    turn did marked the run UNSOLICITED but not the turn that later took it
    over, so brain.ask_stream streamed that turn's answer with no hold."""

    async def sent_then_notice_then_echo(self, notice=NOTICE, user_text="what's the weather"):
        h = await self.r.send(user_text)
        self.client.feed(F.init(), F.echo("inj", notice))                  # the notice comes first
        await settle()
        self.assertEqual(self.r.run.owner, R.UNSOLICITED)
        self.client.feed(F.echo(h.uuid, user_text), F.delta("Answer."), F.asst("Answer."), F.result("Answer."))
        await settle()
        return h

    async def test_a_notice_before_the_echo_contaminates_the_turn_that_takes_the_run(self):
        h = await self.sent_then_notice_then_echo()
        self.assertEqual(h.contaminated_by, ["tk1"])
        self.assertTrue(self.led.turns[h.turn_id]["merged_into_run"])
        n = [x for x in self.led.notes if x.get("note") == "foreground_contaminated"]
        self.assertEqual((n[0]["turn_id"], n[0]["task_ids"], n[0]["how"]), (h.turn_id, ["tk1"], "merged_into_run"))
        self.assertEqual(self.r.stats["contaminated_foreground"], 1)
        self.assertEqual(h.state, L.T_DONE)

    async def test_a_notice_without_a_task_id_still_contaminates(self):
        h = await self.sent_then_notice_then_echo("<task-notification>no id here</task-notification>")
        self.assertEqual(h.contaminated_by, ["?"])

    async def test_several_notices_before_the_echo_are_all_recorded(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo("inj1", NOTICE),
                         F.echo("inj2", NOTICE.replace("tk1", "tk2")))
        await settle()
        self.client.feed(F.echo(h.uuid, "q"), F.asst("a."), F.result("a."))
        await settle()
        self.assertEqual(sorted(h.contaminated_by), ["tk1", "tk2"])

    async def test_a_turn_that_takes_over_an_abandoned_contaminated_run_inherits_it(self):
        a = await self.r.send("long one")
        self.client.feed(F.init(), F.echo(a.uuid), F.echo("inj", NOTICE))     # the notice lands INSIDE A's run
        await settle()
        self.assertEqual(a.contaminated_by, ["tk1"])
        a.abandoned = True
        b = await self.r.send("next question")
        self.client.feed(F.echo(b.uuid, "next question"), F.asst("B."), F.result("B."))
        await settle()
        self.assertEqual(b.contaminated_by, ["tk1"])

    async def test_a_merge_with_no_background_notice_is_not_contaminated(self):
        a = await self.r.send("long")
        self.client.feed(F.init(), F.echo(a.uuid))
        await settle()
        a.abandoned = True
        b = await self.r.send("two")
        self.client.feed(F.echo(b.uuid, "two"), F.asst("two."), F.result("two."))
        await settle()
        self.assertEqual(b.contaminated_by, [])
        self.assertFalse([x for x in self.led.notes if x.get("note") == "foreground_contaminated"])

    async def test_the_contamination_does_not_leak_into_the_next_clean_turn(self):
        await self.sent_then_notice_then_echo()
        c = await self.r.send("another")
        self.client.feed(*F.fg_run(c.uuid, "clean."))
        await settle()
        self.assertEqual(c.contaminated_by, [])
        self.assertEqual(self.r.run, None)

    async def test_a_notice_inside_a_claimed_run_is_still_flagged_as_before(self):
        h = await self.r.send("q")
        self.client.feed(F.init(), F.echo(h.uuid), F.echo("inj", NOTICE))
        await settle()
        self.assertEqual(h.contaminated_by, ["tk1"])


class BackgroundClassificationTests(Base):
    """Regression for the live defect of 2026-09-30 21:05: the CLI ran an Agent
    in the background although its input had NO run_in_background, and the
    router, trusting that flag, recorded it as a foreground tool task. The
    authority is `is_backgrounded` on the task's own start message."""

    def agent_without_flag(self, tid="a16ff5bdae", tuid="toolu_live1", wire=True, extra=None):
        # the shape captured live: description/prompt/subagent_type, no flag
        return [F.tool_use(tuid, "Agent", {"description": "Count daily notes in vault",
                                           "prompt": "count them", "subagent_type": "general-purpose"}),
                F.task_started(tid, tuid, "Count daily notes in vault", task_type="local_agent",
                               is_backgrounded=wire, **(extra or {})),
                F.tool_result(tuid, "Async agent launched successfully.")]

    async def test_flag_unset_but_wire_says_backgrounded_is_a_bg_agent(self):
        seen = []
        self.mgr.tasks.listeners.append(lambda row: seen.append((row["id"], row["state"])))
        h = await self.r.send("Launch a background agent to count my daily notes")
        self.client.feed(F.init(), F.echo(h.uuid), *self.agent_without_flag(),
                         F.delta("Counting now."), F.asst("Counting now."), F.result("Counting now."))
        await settle()
        t = self.led.tasks["a16ff5bdae"]
        self.assertEqual(t["task_kind"], L.KIND_BG_AGENT)
        self.assertEqual(t["kind_basis"], "wire:is_backgrounded")
        self.assertTrue(t["wire_is_backgrounded"])
        self.assertEqual(t["wire_task_type"], "local_agent")
        self.assertTrue(t["protected"])                               # barge-in protection applies
        self.assertIn("a16ff5bdae", {x["id"] for x in self.r.live_bg()})
        # the launching foreground turn finished normally and on its own
        self.assertEqual(h.state, L.T_DONE)
        self.assertEqual(self.result_text(h), "Counting now.")
        # the action that launched it records that the input had no flag
        a = self.led.actions[self.led.action_id_for("toolu_live1")]
        self.assertIsNone(a.get("run_in_background"))
        self.assertEqual(a.get("task_id"), "a16ff5bdae")
        # the task finishes AFTER the turn, exactly as live: two completion signals,
        # the first with an empty summary
        self.client.feed(F.task_updated("a16ff5bdae", "completed"),
                         F.task_note("a16ff5bdae", "toolu_live1", "completed", ""),
                         F.task_note("a16ff5bdae", "toolu_live1", "completed",
                                     "21 daily note files, dated 2026-09-09 to 2026-09-30."),
                         *F.unsolicited_run("Sir, there are twenty-one daily notes"))
        await settle()
        t = self.led.tasks["a16ff5bdae"]
        self.assertEqual(t["state"], L.K_COMPLETED)
        self.assertEqual(t["summary"], "21 daily note files, dated 2026-09-09 to 2026-09-30.")
        self.assertEqual(seen, [("a16ff5bdae", L.K_COMPLETED)])       # listener fired once
        self.assertEqual(len(self.r.pending_reports()), 1)            # the report was captured, not routed
        self.assertEqual(h.state, L.T_DONE)

    async def test_explicit_foreground_agent_stays_a_foreground_task(self):
        h = await self.r.send("count them and wait")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.tool_use("toolu_fgag", "Agent", {"description": "d", "prompt": "p",
                                                            "run_in_background": False}),
                         F.task_started("fgag", "toolu_fgag", "d", task_type="local_agent",
                                        is_backgrounded=False))
        await settle()
        t = self.led.tasks["fgag"]
        self.assertEqual(t["task_kind"], L.KIND_FG_TOOL)
        self.assertEqual(t["kind_basis"], "wire:is_backgrounded=false")
        self.assertFalse(t["protected"])
        self.assertEqual(self.r.live_bg(), [])

    async def test_foreground_bash_task_is_not_a_background_agent(self):
        h = await self.r.send("run it")
        self.client.feed(F.init(), F.echo(h.uuid), F.tool_use("toolu_b", "Bash", {"command": "x"}),
                         F.task_started("bx", "toolu_b", "Sleep for 5 seconds", task_type="local_bash",
                                        is_backgrounded=False))
        await settle()
        self.assertEqual(self.led.tasks["bx"]["task_kind"], L.KIND_FG_TOOL)

    async def test_subagent_internal_tasks_win_over_everything(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.task_started("inner", "toolu_i", "inner", owned=True, is_backgrounded=True))
        await settle()
        self.assertEqual(self.led.tasks["inner"]["task_kind"], L.KIND_SUBAGENT)

    async def test_when_the_input_flag_and_the_wire_disagree_the_work_is_protected(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid),
                         F.tool_use("toolu_c", "Agent", {"run_in_background": True}),
                         F.task_started("conf", "toolu_c", "d", task_type="local_agent", is_backgrounded=False))
        await settle()
        self.assertEqual(self.led.tasks["conf"]["task_kind"], L.KIND_BG_AGENT)
        self.assertEqual(self.led.tasks["conf"]["kind_basis"], "conflict:input_flag_true_wire_false")
        self.assertTrue(any(n.get("note") == "task_kind_conflict" for n in self.led.notes))

    async def test_a_cli_without_is_backgrounded_still_classifies_by_the_input_flag(self):
        h = await self.r.send("x")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("old", "toolu_old"))
        await settle()
        self.assertEqual(self.led.tasks["old"]["task_kind"], L.KIND_BG_AGENT)
        self.assertEqual(self.led.tasks["old"]["kind_basis"], "fallback:input_run_in_background")

    async def test_stop_task_still_works_on_a_task_classified_from_the_wire(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *self.agent_without_flag(), F.result("ok"))
        await settle()
        self.client.on_stop = lambda tid: self.client.feed(F.task_updated(tid, "killed"),
                                                           F.task_note(tid, "toolu_live1", "stopped"))
        self.assertTrue(await self.r.stop_task("a16ff5bdae", "USER_STOP"))
        await settle()
        self.assertEqual(self.led.tasks["a16ff5bdae"]["state"], L.K_STOPPED_USER)
        self.assertEqual(self.client.interrupts, 0)

    async def test_foreground_routing_is_unchanged_around_it(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *self.agent_without_flag(), F.result("ok"))
        await settle()
        self.client.feed(F.task_note("a16ff5bdae", "toolu_live1", "completed", "done"),
                         *F.unsolicited_run("report"))
        await settle()
        h2 = await self.r.send("capital of France?")
        self.client.feed(*F.fg_run(h2.uuid, "Paris"))
        await settle()
        self.assertEqual(self.result_text(h2), "Paris")
        self.assertEqual(self.mgr.quarantine.counts["unowned_result"], 0)
