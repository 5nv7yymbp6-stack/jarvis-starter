"""The stale-interrupt rate guard (router.py, 2026-10-05).

An abandoned query's run still starts when its echo arrives, and _interrupt_stale() asks the CLI to stop
it. When several abandoned queries are queued their echoes arrive within milliseconds and each one used
to send its own CLI interrupt; two interrupt control requests a few ms apart killed the CLI (AbortError
-> stream_lost) or interrupted the live query, reproduced with the real CLI. The router now sends a stale
interrupt only if no other went out in the last STALE_INTERRUPT_MIN_GAP_S. It never DELAYS one (a delayed
interrupt lands on whatever run is current by then) and never touches the live handle's own interrupt().

Fake SDK client, real router. Nothing here touches the live voice process or its files."""
import asyncio
import unittest

from backtalk import ledger as L
from backtalk import router as R
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle
from tests.test_router import Base

ABORTED = dict(is_error=True, subtype="error_during_execution", terminal="aborted_streaming")


class GuardTests(Base):
    async def abandoned(self, n, r=None, client=None):
        """n queries sent and abandoned before any was claimed (what a burst of stitches leaves
        behind), and the handles."""
        r = r or self.r
        hs = []
        for i in range(n):
            h = await r.send(f"fragment {i}")
            self.assertEqual(await r.interrupt(), "none")
            self.assertEqual(h.state, L.T_ABORTED)
            hs.append(h)
        return hs

    def stale_runs(self, hs):
        """The CLI draining its queue: each abandoned query starts a run, echoes, and is aborted, all
        within one burst of messages."""
        msgs = []
        for h in hs:
            msgs += [F.init(), F.echo(h.uuid), F.result("", **ABORTED)]
        return msgs

    async def test_the_constant_is_the_documented_gap(self):
        self.assertEqual(R.STALE_INTERRUPT_MIN_GAP_S, 1.2)

    async def test_two_stale_echoes_within_100ms_are_both_quarantined_but_send_one_interrupt(self):
        hs = await self.abandoned(2)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 2)
        self.assertEqual(self.client.interrupts, 1)
        self.assertEqual(self.r.stats["stale_interrupt_suppressed"], 1)

    async def test_three_stale_echoes_within_a_second_send_exactly_one_interrupt(self):
        hs = await self.abandoned(3)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        await asyncio.sleep(0.3)
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 3)
        self.assertEqual(self.client.interrupts, 1)
        self.assertEqual(self.r.stats["stale_interrupt_suppressed"], 2)

    async def test_four_stale_echoes_still_send_exactly_one_interrupt(self):
        hs = await self.abandoned(4)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 4)
        self.assertEqual(self.client.interrupts, 1)

    async def test_stale_echoes_spaced_at_least_the_gap_apart_can_each_interrupt(self):
        hs = await self.abandoned(2)
        self.client.feed(*self.stale_runs(hs[:1]))
        await settle()
        self.assertEqual(self.client.interrupts, 1)
        await asyncio.sleep(R.STALE_INTERRUPT_MIN_GAP_S + 0.05)
        self.client.feed(*self.stale_runs(hs[1:]))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 2)
        self.assertEqual(self.client.interrupts, 2)
        self.assertEqual(self.r.stats["stale_interrupt_suppressed"], 0)

    async def test_two_stale_echoes_in_the_same_tick_send_one_even_if_the_interrupt_takes_time(self):
        """A real client's interrupt() suspends; the second echo's task must not slip in behind it."""
        sent = []

        async def slow_interrupt():
            sent.append(1)
            await asyncio.sleep(0.05)
        self.client.interrupt = slow_interrupt
        hs = await self.abandoned(3)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        await asyncio.sleep(0.2)
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 3)
        self.assertEqual(len(sent), 1)

    async def test_the_gap_is_measured_from_the_last_interrupt_actually_sent_not_the_last_one_suppressed(self):
        hs = await self.abandoned(3)
        self.client.feed(*self.stale_runs(hs[:1]))
        await settle()
        self.assertEqual(self.client.interrupts, 1)                  # t = 0: sent
        await asyncio.sleep(0.7)
        self.client.feed(*self.stale_runs(hs[1:2]))
        await settle()
        self.assertEqual(self.client.interrupts, 1)                  # t = 0.7: suppressed
        await asyncio.sleep(0.6)
        self.client.feed(*self.stale_runs(hs[2:3]))
        await settle()
        self.assertEqual(self.client.interrupts, 2)                  # t = 1.3: 1.3 s after the SENT one, so it goes
        self.assertEqual(self.r.stats["stale_interrupt_suppressed"], 1)

    async def test_the_interrupt_is_never_delayed_or_queued_for_later(self):
        hs = await self.abandoned(2)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        self.assertEqual(self.client.interrupts, 1)
        await asyncio.sleep(R.STALE_INTERRUPT_MIN_GAP_S + 0.3)       # long after the gap: still nothing owed
        self.assertEqual(self.client.interrupts, 1)

    async def test_a_suppressed_stale_interrupt_does_not_touch_the_live_handle(self):
        stale = await self.abandoned(3)
        live = await self.r.send("the real request")
        self.client.feed(*self.stale_runs(stale))
        await settle()
        self.assertEqual(self.client.interrupts, 1)                  # only the first stale run was stopped
        self.client.feed(*F.fg_run(live.uuid, "the real answer"))
        await settle()
        self.assertEqual(live.state, L.T_DONE)                       # the live turn ran and finished untouched
        self.assertEqual(self.result_text(live), "the real answer")
        self.assertEqual(self.client.interrupts, 1)

    async def test_the_live_handles_own_interrupt_is_never_rate_limited_by_the_stale_guard(self):
        stale = await self.abandoned(2)
        self.client.feed(*self.stale_runs(stale))
        await settle()
        self.assertEqual(self.client.interrupts, 1)                  # a stale interrupt just went out
        live = await self.r.send("the real request")
        self.client.feed(F.init(), F.echo(live.uuid), F.delta("working"))
        await settle()
        self.assertIn(live.state, (L.T_CLAIMED, L.T_RUNNING))
        self.client.on_interrupt = lambda: self.client.feed(F.result("", **ABORTED))
        self.assertEqual(await self.r.interrupt(reason="BARGE_IN"), "aborted")
        self.assertEqual(self.client.interrupts, 2)                  # sent at once, well inside 1.2 s
        self.assertEqual(live.state, L.T_INTERRUPTED)

    async def test_a_stale_echo_is_quarantined_whether_or_not_its_interrupt_was_sent(self):
        hs = await self.abandoned(3)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        reasons = [rec["reason"] for rec in self.mgr.quarantine.ring]
        self.assertEqual(reasons.count("stale_echo"), 3)
        for h in hs:
            self.assertEqual(h.state, L.T_ABORTED)                   # an abandoned handle stays abandoned

    async def test_a_new_epoch_starts_with_no_history(self):
        hs = await self.abandoned(1)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        self.assertEqual(self.client.interrupts, 1)
        self.assertIsNotNone(self.r._last_stale_interrupt)

        client2 = FakeClient()
        r2 = await self.mgr.open(client2, session_id="s2", cli_pid=None)
        self.extra.append(r2)
        self.assertIsNone(r2._last_stale_interrupt)                   # nothing inherited from the old epoch
        hs2 = await self.abandoned(1, r=r2)
        client2.feed(*self.stale_runs(hs2))
        await settle()
        self.assertEqual(client2.interrupts, 1)                       # sent, though the old epoch's was < 1.2 s ago
        self.assertEqual(r2.stats["stale_interrupt_suppressed"], 0)

    async def test_an_old_epoch_cannot_suppress_the_new_one_even_when_both_are_within_the_gap(self):
        hs = await self.abandoned(1)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        client2 = FakeClient()
        r2 = await self.mgr.open(client2, session_id="s3", cli_pid=None)
        self.extra.append(r2)
        for _ in range(2):
            hs2 = await self.abandoned(1, r=r2)
            client2.feed(*self.stale_runs(hs2))
            await settle()
        self.assertEqual(client2.interrupts, 1)                       # its own guard works within the new epoch
        self.assertEqual(r2.stats["stale_interrupt_suppressed"], 1)

    async def test_the_existing_stale_behaviour_is_unchanged_for_a_single_stale_echo(self):
        h = await self.r.send("never mind")
        self.assertEqual(await self.r.interrupt(), "none")
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("x"), F.asst("x"), F.result("x"))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 1)
        self.assertEqual(h.state, L.T_ABORTED)
        self.assertGreaterEqual(self.client.interrupts, 1)

    async def test_with_live_background_tasks_a_stale_echo_still_sends_no_interrupt_at_all(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.result("started"))
        await settle()
        self.assertTrue(self.r.live_bg())                            # the background task outlives the turn
        hs = await self.abandoned(1)
        self.client.feed(*self.stale_runs(hs))
        await settle()
        self.assertEqual(self.mgr.quarantine.counts["stale_echo"], 1)
        self.assertEqual(self.client.interrupts, 0)                  # unchanged: an interrupt would kill the task
        self.assertEqual(self.r.stats["stale_interrupt_suppressed"], 0)


class CallSiteTests(unittest.TestCase):
    def test_the_guard_applies_only_to_the_stale_echo_path(self):
        import inspect
        src = inspect.getsource(R.EpochRouter)
        self.assertEqual(src.count("self._interrupt_stale()"), 1)
        i = src.index("async def interrupt(self")
        live = src[i:src.index("def _schedule_escalation", i)]
        self.assertNotIn("_last_stale_interrupt", live)               # the live handle's interrupt() is a different path
        self.assertNotIn("STALE_INTERRUPT_MIN_GAP_S", live)


if __name__ == "__main__":
    unittest.main()
