"""F-11: a terminal outcome is first-write-wins and immutable. Once an action or task has reached a terminal state, a
later callback (a duplicate, a conflicting result, a late result from another epoch) cannot change its state,
overwrite the result/evidence recorded when it got there, re-run its verification, or revive it. A conflicting
report is recorded once, as evidence, and nothing else happens. The real router, ledger and verifier run against the
fake client."""
import asyncio
import os

from backtalk import ledger as L
from backtalk import verify as V
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle
from tests.test_router import Base


class ImmutabilityBase(Base):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.calls = []
        self.r.action_listeners.append(lambda row, inp: self.calls.append((row["id"], row["state"])))

    async def start(self, r=None, client=None):
        r, client = r or self.r, client or self.client
        h = await r.send("go")
        client.feed(F.init(), F.echo(h.uuid))
        await settle()

    def act(self, tuid):
        return self.led.actions[self.led.action_id_for(tuid)]

    def facts(self, tuid):
        a = self.act(tuid)
        return (a["state"], a.get("is_error"), a.get("returned_at"), dict(a.get("result") or {}),
                dict(a.get("evidence") or {}), dict(a["verification"]), list(a["history"]))

    def notes(self, name):
        return [n for n in self.led.notes if n.get("note") == name]

    async def call(self, tuid="t", *, error=False, body="x\ny\n", decide=True):
        if decide:
            F.seen(self.led, tuid, "ALLOW", epoch=self.r.epoch, tool="Bash")
        self.client.feed(F.tool_use(tuid, "Bash", {"command": "ls"}), F.tool_result(tuid, body, is_error=error))
        await settle()


class ActionTerminalTests(ImmutabilityBase):
    async def test_ok_then_error_keeps_the_ok_and_everything_recorded_with_it(self):
        await self.start(); await self.call(error=False)
        before = self.facts("t")
        self.client.feed(F.tool_result("t", "boom", is_error=True)); await settle()
        self.assertEqual(self.facts("t"), before)
        self.assertEqual(len(self.calls), 1)                                        # verification ran once, for the first result
        c = self.act("t")["conflicting_result"]
        self.assertEqual((c["recorded"], c["reported"]), (L.A_OK, L.A_ERROR))
        self.assertEqual(len(self.notes("conflicting_terminal_result")), 1)

    async def test_error_then_ok_keeps_the_error(self):
        await self.start(); await self.call(error=True)
        before = self.facts("t")
        self.client.feed(F.tool_result("t", "fine", is_error=False)); await settle()
        self.assertEqual(self.facts("t"), before)
        self.assertEqual(self.act("t")["conflicting_result"]["reported"], L.A_OK)
        self.assertEqual(len(self.calls), 1)

    async def test_a_duplicate_result_is_a_no_op_that_cannot_overwrite_the_evidence(self):
        await self.start(); await self.call(body="x\ny\n")
        before, seq = self.facts("t"), self.led._seq
        for _ in range(3):
            self.client.feed(F.tool_result("t", "a completely different and much longer body\nz\nw\nv\n")); await settle()
        self.assertEqual(self.facts("t"), before)
        self.assertEqual(self.led._seq, seq)                                       # not one record was written
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("conflicting_result", self.act("t"))                      # same outcome: a duplicate, not a conflict
        self.assertEqual(self.r.stats["late_tool_result"], 3)                       # counted, so a storm of them is visible

    async def test_a_repeated_conflicting_result_is_recorded_once(self):
        await self.start(); await self.call(error=False)
        for _ in range(4):
            self.client.feed(F.tool_result("t", "boom", is_error=True)); await settle()
        self.assertEqual(len(self.notes("conflicting_terminal_result")), 1)
        self.assertEqual(self.notes("refused_transition"), [])

    async def test_a_result_for_a_lost_action_does_not_revive_it_or_get_it_verified(self):
        q = V.VerifyQueue(self.led)
        self.r.action_listeners.append(lambda row, inp: q.submit(row, inp))
        path = os.path.join(self.tmp.name, "w.txt"); open(path, "w").write("hello")
        await self.start()
        F.seen(self.led, "w", "ALLOW", epoch=self.r.epoch, tool="Write")
        self.client.feed(F.tool_use("w", "Write", {"file_path": path, "content": "hello"})); await settle()
        self.led.action_state(self.led.action_id_for("w"), L.A_LOST, cause="lost")
        self.led.set_verification(self.led.action_id_for("w"), L.V_UNVERIFIED, L.LEVEL_NONE, "lost", "t")
        self.client.feed(F.tool_result("w", "File created")); await settle(); await asyncio.sleep(0.15)
        a = self.act("w")
        self.assertEqual((a["state"], a["verification"]["state"]), (L.A_LOST, L.V_UNVERIFIED))
        self.assertEqual(a["conflicting_result"]["recorded"], L.A_LOST)
        self.assertEqual(self.calls, [])

    async def test_a_late_result_through_the_next_epochs_router_cannot_verify_an_action_epoch_close_lost(self):
        q = V.VerifyQueue(self.led)
        self.r.action_listeners.append(lambda row, inp: q.submit(row, inp))
        path = os.path.join(self.tmp.name, "late.txt"); open(path, "w").write("hello")
        await self.start()
        F.seen(self.led, "w", "ALLOW", epoch=self.r.epoch, tool="Write")
        self.client.feed(F.tool_use("w", "Write", {"file_path": path, "content": "hello"})); await settle()
        old = self.r
        await old.close("rebuild", self.client.disconnect)
        self.extra.append(old)
        c2 = FakeClient()
        self.r = await self.mgr.open(c2, session_id="s", cli_pid=None)
        self.r.action_listeners.append(lambda row, inp: q.submit(row, old._inputs.get("w")))
        await self.start(self.r, c2)
        self.assertEqual((self.act("w")["state"], self.act("w")["verification"]["state"]), (L.A_LOST, L.V_UNVERIFIED))
        c2.feed(F.tool_result("w", "File created")); await settle(); await asyncio.sleep(0.15)
        a = self.act("w")
        self.assertEqual((a["state"], a["verification"]["state"]), (L.A_LOST, L.V_UNVERIFIED))
        self.assertEqual(self.led.notes[-1]["note"], "conflicting_terminal_result")

    async def test_epoch_close_and_reconcile_cannot_replace_a_terminal_result(self):
        await self.start(); await self.call("ok1"); await self.call("bad1", error=True)
        before = (self.facts("ok1"), self.facts("bad1"))
        await self.r.close("rebuild", self.client.disconnect)
        self.assertEqual(self.led.lose_epoch_actions(self.r.epoch, "again"), 0)
        self.led.reconcile()
        self.assertEqual((self.facts("ok1"), self.facts("bad1")), before)

    async def test_a_denied_action_still_gets_its_result_once_and_a_repeat_changes_nothing(self):
        q = V.VerifyQueue(self.led)
        self.r.action_listeners.append(lambda row, inp: q.submit(row, inp))
        await self.start()
        F.seen(self.led, "d", "DENY", epoch=self.r.epoch, tool="Bash")
        self.client.feed(F.tool_use("d", "Bash", {}), F.tool_result("d", "blocked", is_error=True)); await settle()
        a = self.act("d")
        self.assertEqual((a["state"], a["verification"]["state"], len(self.calls)), (L.A_DENIED, L.V_NOT_REQUIRED, 1))
        before = self.facts("d")
        self.client.feed(F.tool_result("d", "blocked again", is_error=True)); await settle()
        self.assertEqual((self.facts("d"), len(self.calls)), (before, 1))

    async def test_a_burst_of_conflicting_callbacks_is_first_wins_every_time(self):
        for i in range(25):
            tuid = f"b{i}"
            await self.start() if i == 0 else None
            first = bool(i % 2)
            F.seen(self.led, tuid, "ALLOW", epoch=self.r.epoch, tool="Bash")
            self.client.feed(F.tool_use(tuid, "Bash", {}), F.tool_result(tuid, "1", is_error=first),
                             F.tool_result(tuid, "2", is_error=not first), F.tool_result(tuid, "3", is_error=first))
            await settle()
            a = self.act(tuid)
            self.assertEqual(a["state"], L.A_ERROR if first else L.A_OK, tuid)
            self.assertEqual(a["result"]["bytes"], 1, tuid)
        self.assertEqual(len(self.calls), 25)                                      # one verification per action, not per callback


class TaskTerminalTests(ImmutabilityBase):
    async def launch(self, tid="tk"):
        await self.start()
        self.client.feed(*F.launch_bg(tid, "toolu_" + tid, "job")); await settle()

    def task(self, tid="tk"):
        return self.led.tasks[tid]

    async def test_a_user_stop_then_a_completed_report_keeps_the_stop_and_records_the_conflict(self):
        await self.launch()
        self.led.task_cancel_intent("tk", "USER_STOP")
        self.client.feed(F.task_note("tk", "toolu_tk", "stopped")); await settle()
        before = (self.task()["state"], list(self.task()["history"]))
        self.client.feed(F.task_updated("tk", "completed"), F.task_note("tk", "toolu_tk", "completed", "done")); await settle()
        t = self.task()
        self.assertEqual((t["state"], t["history"]), before)
        self.assertEqual([c["wire"] for c in t["conflicting_terminal"]], ["completed"])
        self.assertEqual(len(self.notes("conflicting_terminal_status")), 1)

    async def test_completed_then_failed_keeps_completed_and_records_it_once(self):
        await self.launch()
        self.client.feed(F.task_note("tk", "toolu_tk", "completed", "ok")); await settle()
        for _ in range(3):
            self.client.feed(F.task_note("tk", "toolu_tk", "failed")); await settle()
        self.assertEqual(self.task()["state"], L.K_COMPLETED)
        self.assertEqual([c["wire"] for c in self.task()["conflicting_terminal"]], ["failed"])
        self.assertEqual(len(self.notes("conflicting_terminal_status")), 1)

    async def test_a_report_that_agrees_with_the_recorded_outcome_is_not_a_conflict(self):
        await self.launch()
        self.led.task_cancel_intent("tk", "USER_STOP")
        self.client.feed(F.task_updated("tk", "killed"), F.task_note("tk", "toolu_tk", "stopped"), F.task_note("tk", "toolu_tk", "killed")); await settle()
        self.assertEqual(self.task()["state"], L.K_STOPPED_USER)
        self.assertNotIn("conflicting_terminal", self.task())
        self.assertEqual(self.notes("conflicting_terminal_status"), [])

    async def test_a_late_summary_still_reaches_the_task_and_is_not_mistaken_for_a_conflict(self):
        await self.launch()
        self.client.feed(F.task_updated("tk", "completed")); await settle()
        self.client.feed(F.task_note("tk", "toolu_tk", "completed", "all done")); await settle()
        self.assertEqual(self.task()["state"], L.K_COMPLETED)
        self.assertEqual(self.task().get("summary"), "all done")
        self.assertNotIn("conflicting_terminal", self.task())

    async def test_a_report_from_another_epoch_still_never_changes_the_recorded_state(self):
        await self.launch()
        self.client.feed(F.task_note("tk", "toolu_tk", "completed", "ok")); await settle()
        old = self.r
        await old.close("rebuild", self.client.disconnect); self.extra.append(old)
        c2 = FakeClient()
        self.r = await self.mgr.open(c2, session_id="s", cli_pid=None)
        c2.feed(F.task_note("tk", "toolu_tk", "failed")); await settle()
        self.assertEqual(self.task()["state"], L.K_COMPLETED)
