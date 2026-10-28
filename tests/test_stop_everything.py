"""F-02: "stop everything" says N background tasks were stopped only for tasks whose STOPPED_BY_USER state the
ledger actually observed. The real do_stop_everything, brain, router and ledger run against the fake client;
the only thing the tests choose is what the (fake) CLI does when told to stop."""
import asyncio
from unittest import mock

from backtalk import ledger as L
from backtalk import main as M
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_main_router import Base


class StopEverythingTests(Base):
    async def launch(self, tid, desc=None):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg(tid, "toolu_" + tid, desc or tid + " job"),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.script = script
        await self.reply("go")
        self.mouth.said.clear()
        self.mouth.chunks.clear()

    def on_stop(self, behaviour):
        """behaviour: {task_id: "kill" | "fail" | "ignore" | "kill_twice" | "raise"}"""
        def stop(tid):
            how = behaviour.get(tid, "ignore")
            if how in ("kill", "kill_twice"):
                self.client.feed(F.task_updated(tid, "killed"), F.task_note(tid, "toolu_" + tid, "stopped"))
            if how == "kill_twice":
                self.client.feed(F.task_updated(tid, "killed"), F.task_note(tid, "toolu_" + tid, "stopped"),
                                 F.task_note(tid, "toolu_" + tid, "killed"))
            if how == "fail":
                self.client.feed(F.task_note(tid, "toolu_" + tid, "failed"))
            if how == "raise":
                raise RuntimeError("stop refused")
        self.client.on_stop = stop

    async def stop(self, task=None):
        """Run the real do_stop_everything with the confirmation wait collapsed (it is 3 s in production)."""
        real_sleep = asyncio.sleep
        with mock.patch.object(M.asyncio, "sleep", new=lambda s: real_sleep(0)):
            await M.do_stop_everything(self.brain, self.mouth, task)
        await settle()

    def state(self, tid):
        return self.led.tasks[tid]["state"]

    # ---- counting ----
    async def test_two_tasks_that_both_stop_are_counted_two(self):
        await self.launch("a"); await self.launch("b")
        self.on_stop({"a": "kill", "b": "kill"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["Stopped 2 background tasks."])
        self.assertEqual((self.state("a"), self.state("b")), (L.K_STOPPED_USER, L.K_STOPPED_USER))

    async def test_only_the_task_that_actually_stopped_is_counted(self):
        await self.launch("a"); await self.launch("b")
        self.on_stop({"a": "kill", "b": "ignore"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["Stopped 1 background task. I couldn't confirm the other one stopped."])
        self.assertEqual((self.state("a"), self.state("b")), (L.K_STOPPED_USER, L.K_RUNNING))

    async def test_a_stop_that_was_only_requested_is_not_a_stop(self):
        await self.launch("a"); await self.launch("b")
        self.on_stop({})                                              # acknowledged, nothing ever happens
        await self.stop()
        self.assertEqual(self.mouth.said, ["I asked them to stop, but I couldn't confirm they stopped."])
        self.assertEqual((self.state("a"), self.state("b")), (L.K_RUNNING, L.K_RUNNING))

    async def test_a_stop_request_that_errors_leaves_the_task_uncounted(self):
        await self.launch("a")
        self.on_stop({"a": "raise"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["I asked it to stop, but I couldn't confirm it stopped."])
        self.assertEqual(self.state("a"), L.K_RUNNING)

    async def test_a_task_that_failed_instead_of_stopping_is_not_counted(self):
        await self.launch("a"); await self.launch("b")
        self.on_stop({"a": "fail", "b": "kill"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["Stopped 1 background task. I couldn't confirm the other one stopped."])
        self.assertEqual((self.state("a"), self.state("b")), (L.K_FAILED, L.K_STOPPED_USER))

    async def test_a_task_that_was_already_finished_is_not_counted_as_newly_stopped(self):
        await self.launch("a"); await self.launch("b")
        self.client.feed(F.task_note("a", "toolu_a", "completed", "done"))
        await settle()
        self.assertEqual(self.state("a"), L.K_COMPLETED)
        self.mouth.said.clear()
        self.on_stop({"a": "kill", "b": "kill"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["Stopped 1 background task."])
        self.assertEqual(self.state("a"), L.K_COMPLETED)

    async def test_repeated_stop_signals_never_count_a_task_twice(self):
        await self.launch("a"); await self.launch("b")
        self.on_stop({"a": "kill_twice", "b": "kill_twice"})
        await self.stop()
        self.assertEqual(self.mouth.said, ["Stopped 2 background tasks."])
        self.mouth.said.clear()
        await self.stop()                                             # asking again: nothing is left to stop
        self.assertEqual(self.mouth.said, ["Nothing is running."])

    # ---- the zero cases keep their wording ----
    async def test_nothing_running_and_nothing_to_stop(self):
        await self.stop()
        self.assertEqual(self.mouth.said, ["Nothing is running."])

    async def test_a_running_request_with_no_tasks_is_just_stopped(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        task = asyncio.create_task(self.reply("long one"))
        await settle(10)
        await self.stop(task)
        self.assertEqual(self.mouth.said, ["Stopped."])

    # ---- the run itself ----
    async def test_the_current_request_is_still_named_when_it_was_stopped(self):
        await self.launch("a"); await self.launch("b")
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        task = asyncio.create_task(self.reply("long one"))
        await settle(10)
        self.on_stop({"a": "kill", "b": "kill"})
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        await self.stop(task)
        self.assertEqual(self.mouth.said, ["Stopped 2 background tasks and the current request."])

    async def test_an_interrupt_that_was_never_acknowledged_is_not_claimed_as_stopping_the_request(self):
        await self.launch("a"); await self.launch("b")
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        task = asyncio.create_task(self.reply("long one"))
        await settle(10)
        self.on_stop({"a": "kill", "b": "kill"})
        self.client.on_interrupt = lambda: None                       # the CLI never ends the run
        router, real = self.brain.router, self.brain.router.interrupt

        async def quick(*a, **k):
            k["wait"] = 0.05
            return await real(*a, **k)
        router.interrupt = quick
        await self.stop(task)
        self.assertEqual(self.mouth.said, ["Stopped 2 background tasks."])
        task.cancel()

    async def test_the_request_is_named_separately_when_no_task_was_seen_to_stop(self):
        await self.launch("a")
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        task = asyncio.create_task(self.reply("long one"))
        await settle(10)
        self.on_stop({})
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        await self.stop(task)
        self.assertEqual(self.mouth.said, ["Stopped the current request. I asked it to stop, but I couldn't confirm it stopped."])
