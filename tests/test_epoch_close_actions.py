"""F-08: closing an epoch leaves no action of that epoch in flight. The real router, ledger and epoch manager run
against the fake client; actions are put in each state the way production puts them (the stream's tool_use,
the permission hook's decision, the tool result)."""
from backtalk import ledger as L
from backtalk import router as R
from tests import router_fakes as F
from tests.router_fakes import FakeClient, settle
from tests.test_router import Base


class CloseActionsTests(Base):
    async def rebuild(self, reason="desync", fail=False):
        """What WarmBrain.rebuild does to the router: close the old epoch, open the next on a new client."""
        old = self.r
        self.client.fail_disconnect = fail
        ok = await old.close(reason, self.client.disconnect)
        self.client2 = FakeClient()
        self.r = await self.mgr.open(self.client2, session_id="s", cli_pid=None)
        self.extra.append(old)
        return ok

    async def start_run(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid))
        await settle()
        return h

    async def requested(self, tuid, tool="Bash"):
        self.client.feed(F.tool_use(tuid, tool, {}))
        await settle()

    async def allowed(self, tuid, tool="Bash"):
        F.seen(self.led, tuid, "LEGACY_ALLOWED", epoch=self.r.epoch, tool=tool)
        await self.requested(tuid, tool)

    async def returned(self, tuid, *, error=False):
        await self.allowed(tuid)
        self.client.feed(F.tool_result(tuid, "boom" if error else "ok", is_error=error))
        await settle()

    def act(self, tuid):
        return self.led.actions[self.led.action_id_for(tuid)]

    def in_flight(self, epoch):
        return [a["id"] for a in self.led.actions.values()
                if a.get("epoch") == epoch and a.get("state") in (L.A_REQUESTED, L.A_ALLOWED)]

    # ---- the stale state, and its fix ----
    async def test_a_requested_action_is_lost_when_its_epoch_closes(self):
        await self.start_run(); await self.requested("t1")
        self.assertEqual(self.act("t1")["state"], L.A_REQUESTED)
        old = self.r.epoch
        await self.rebuild()
        a = self.act("t1")
        self.assertEqual(a["state"], L.A_LOST)
        self.assertEqual(a["verification"]["state"], L.V_UNVERIFIED)
        self.assertEqual(self.in_flight(old), [])

    async def test_a_legacy_allowed_action_is_lost_when_its_epoch_closes(self):
        await self.start_run(); await self.allowed("t2")
        self.assertEqual(self.act("t2")["state"], L.A_ALLOWED)
        await self.rebuild()
        a = self.act("t2")
        self.assertEqual(a["state"], L.A_LOST)
        self.assertEqual(a["verification"]["state"], L.V_UNVERIFIED)
        self.assertIn("epoch", a["history"][-1][2])                # the cause says why

    async def test_there_is_no_running_action_state_so_requested_and_allowed_are_all_that_can_be_in_flight(self):
        self.assertEqual(set(L._ACTION_NEXT), {L.A_REQUESTED, L.A_ALLOWED})
        self.assertNotIn("RUNNING", {s for v in L._ACTION_NEXT.values() for s in v} | set(L._ACTION_NEXT))

    async def test_nothing_from_the_closed_epoch_is_left_in_flight_or_pending(self):
        await self.start_run()
        await self.requested("a"); await self.allowed("b"); await self.returned("c"); await self.returned("d", error=True)
        old = self.r.epoch
        F.seen(self.led, "e", "DENY", epoch=old, tool="Bash")      # denied, no result seen
        await self.rebuild()
        self.assertEqual(self.in_flight(old), [])
        self.assertEqual([self.act(t)["state"] for t in "abcd"],
                         [L.A_LOST, L.A_LOST, L.A_OK, L.A_ERROR])
        for t in "ab":
            self.assertEqual(self.act(t)["verification"]["state"], L.V_UNVERIFIED)
        self.assertEqual(self.act("e")["state"], L.A_DENIED)
        self.assertEqual(self.act("e")["verification"]["state"], L.V_NOT_REQUIRED)

    async def test_actions_that_finished_before_the_close_are_unchanged(self):
        await self.start_run(); await self.returned("ok1"); await self.returned("bad1", error=True)
        before = {t: (self.act(t)["state"], dict(self.act(t)["verification"]), len(self.act(t)["history"])) for t in ("ok1", "bad1")}
        await self.rebuild()
        after = {t: (self.act(t)["state"], dict(self.act(t)["verification"]), len(self.act(t)["history"])) for t in ("ok1", "bad1")}
        self.assertEqual(before, after)
        self.assertEqual((before["ok1"][0], before["bad1"][0]), (L.A_OK, L.A_ERROR))

    async def test_a_lost_stream_then_a_close_still_ends_with_no_pending_verification(self):
        await self.start_run(); await self.requested("t3")
        self.client.feed(RuntimeError("Command failed with exit code -9"))
        await settle()
        self.assertEqual(self.act("t3")["state"], L.A_LOST)
        self.assertEqual(self.act("t3")["verification"]["state"], L.V_PENDING)     # the stream-lost path leaves it
        await self.rebuild()
        self.assertEqual(self.act("t3")["state"], L.A_LOST)
        self.assertEqual(self.act("t3")["verification"]["state"], L.V_UNVERIFIED)

    async def test_a_failed_disconnect_also_loses_the_in_flight_actions(self):
        await self.start_run(); await self.allowed("t4")
        old = self.r.epoch
        self.assertFalse(await self.rebuild(fail=True))
        self.assertEqual(self.act("t4")["state"], L.A_LOST)
        self.assertEqual(self.in_flight(old), [])

    # ---- isolation ----
    async def test_another_epochs_actions_are_never_touched(self):
        await self.start_run(); await self.requested("old1")
        stranger = self.led.action_seen("stranger", "E-other", "Bash")             # an epoch that is not closing
        old = self.r.epoch
        await self.rebuild()
        h = await self.r.send("again")                                            # the NEW router's epoch
        self.client2.feed(F.init(), F.echo(h.uuid), F.tool_use("new1", "Bash", {}))
        await settle()
        self.assertEqual(self.act("old1")["state"], L.A_LOST)
        self.assertEqual(self.act("new1")["state"], L.A_REQUESTED)
        self.assertEqual(self.led.actions[stranger]["state"], L.A_REQUESTED)
        self.assertNotEqual(self.r.epoch, old)
        await self.r.close("quit", self.client2.disconnect)                       # now the second epoch closes
        self.assertEqual(self.act("new1")["state"], L.A_LOST)
        self.assertEqual(self.led.actions[stranger]["state"], L.A_REQUESTED)      # still not ours

    # ---- idempotence and callbacks ----
    async def test_closing_twice_changes_nothing_the_second_time(self):
        await self.start_run(); await self.requested("t5"); await self.allowed("t6")
        old = self.r
        await old.close("desync", self.client.disconnect)
        n = (self.led._seq, [list(self.act(t)["history"]) for t in ("t5", "t6")])
        await old.close("desync", self.client.disconnect)
        self.assertEqual((self.led._seq, [list(self.act(t)["history"]) for t in ("t5", "t6")]), n)
        self.assertEqual(self.led.lose_epoch_actions(old.epoch, "again"), 0)      # and the helper alone is idempotent too
        self.assertEqual(self.led._seq, n[0])

    async def test_closing_calls_no_action_listener_and_cannot_verify_a_lost_action(self):
        seen = []
        self.r.action_listeners.append(lambda row, inp: seen.append(row["id"]))
        await self.start_run(); await self.allowed("t7")
        await self.rebuild()
        self.assertEqual(seen, [])                                                # no result arrived, so no verifier ran
        self.assertNotEqual(self.act("t7")["verification"]["state"], L.V_VERIFIED)

    async def test_a_result_that_arrives_after_the_close_cannot_revive_a_lost_action(self):
        await self.start_run(); await self.allowed("t8")
        old = self.r
        await self.rebuild()
        old.dispatch(F.tool_result("t8", "late"))
        self.assertEqual(self.act("t8")["state"], L.A_LOST)
        self.assertGreaterEqual(self.mgr.quarantine.counts["message_after_epoch_end"], 1)

    async def test_the_helper_reports_exactly_how_many_actions_it_newly_lost(self):
        await self.start_run(); await self.requested("h1"); await self.allowed("h2"); await self.returned("h3")
        self.assertEqual(self.led.lose_epoch_actions(self.r.epoch, "x"), 2)
        self.assertEqual([self.act(t)["state"] for t in ("h1", "h2", "h3")], [L.A_LOST, L.A_LOST, L.A_OK])
