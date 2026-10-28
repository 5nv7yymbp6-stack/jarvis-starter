"""Barge-in while a PROTECTED background task is running: the router mutes the old foreground run and lets it finish (the task must not be
killed). Nothing that run says afterwards may reach the mouth, the new request is answered, and the task is untouched. Real router, ledger,
brain and speak_reply; only the SDK client and the mouth are fakes (so this is an automated integration test, not a live voice test)."""
import asyncio

from backtalk import ledger as L
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_main_router import Base


class StaleAnswerAfterProtectedBargeIn(Base):
    async def test_the_old_answer_never_resumes_the_new_one_is_spoken_and_the_task_survives(self):
        def first(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk1", "toolu_1", "harmless export"), F.delta("First part kept. Then, while the export runs,"))
        self.script = first
        old = asyncio.create_task(self.reply("start the export and explain it"))
        await settle(10)
        self.assertEqual(self.mouth.chunks, ["First part kept."])
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)

        old.cancel()                                                 # the cut-in phrase arrives: main cancels the reply, which calls brain.interrupt()
        await asyncio.gather(old, return_exceptions=True)
        h = next(x for x in self.brain.router.handles.values() if x.state not in (L.T_DONE, L.T_FAILED))
        self.assertTrue(h.abandoned, "a protected barge-in must mute the old run")
        self.assertEqual(self.client.interrupts, 0)                  # the background task was not interrupted
        # the muted run keeps producing text and then finishes
        self.client.feed(F.delta(" Stale second sentence. Stale third sentence."), F.asst("x"), F.result("x"))
        await settle(10)
        self.assertEqual(self.mouth.chunks, ["First part kept."], "obsolete speech reached the mouth after the cut-in")

        # the new request is handled normally
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Fresh answer."))
        await self.brain.reset_turn()
        await self.reply("what time is it")
        self.assertEqual(self.mouth.chunks, ["First part kept.", "Fresh answer."])
        self.assertNotIn("Stale", " ".join(self.mouth.chunks))
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)    # still alive, still the same task
        self.assertEqual(self.client.interrupts, 0)
