"""Phase 3: verification (backtalk/verify.py)."""
import asyncio
import http.server
import json
import os
import tempfile
import threading
import time
import unittest

from backtalk import ledger as L
from tests import router_fakes as F
from backtalk import verify as V


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.led = L.Ledger(os.path.join(self.tmp.name, "led"), fsync=False)
        self.ep = self.led.epoch_open("s", 1)
        self.q = V.VerifyQueue(self.led, timeout_s=1.0)

    async def asyncTearDown(self):
        self.led.close()
        self.tmp.cleanup()

    def action(self, tool, inp, *, error=False, denied=False):
        tuid = f"toolu_{len(self.led.actions)}"
        aid = self.led.action_seen(tuid, self.ep, tool)
        if denied:
            F.seen(self.led, tuid, "DENY")
        self.led.action_result(tuid, is_error=error, content="x")
        return aid

    def ver(self, aid):
        return self.led.actions[aid]["verification"]

    async def run_q(self, aid, inp):
        t = self.q.submit(self.led.actions[aid], inp)
        if t:
            await t
        return self.ver(aid)


class StateMachineTests(Base):
    async def test_21_reads_are_not_required(self):
        for tool in ("Read", "Grep", "WebSearch"):
            aid = self.action(tool, {})
            self.assertEqual((await self.run_q(aid, {}))["state"], L.V_NOT_REQUIRED)

    async def test_a_plain_bash_command_has_nothing_independent_to_check(self):
        aid = self.action("Bash", {"command": "ls"})
        v = await self.run_q(aid, {"command": "ls"})
        self.assertEqual((v["state"], v["level"]), (L.V_UNVERIFIED, L.LEVEL_SELF))

    async def test_errors_and_denials_are_not_verified_as_successes(self):
        e = self.action("Write", {}, error=True)
        d = self.action("Write", {}, denied=True)
        self.assertEqual((await self.run_q(e, {}))["state"], L.V_NOT_REQUIRED)
        self.assertEqual((await self.run_q(d, {}))["state"], L.V_NOT_REQUIRED)

    async def test_5_tool_self_report_is_never_verified(self):
        aid = self.action("mcp__gmail__send_message", {"to": "x"})
        v = await self.run_q(aid, {"to": "x"})
        self.assertEqual(v["state"], L.V_UNVERIFIED)
        self.assertEqual(v["level"], L.LEVEL_SELF)
        self.assertNotEqual(v["state"], L.V_VERIFIED)

    async def test_a_verifier_claiming_verified_on_self_report_is_downgraded(self):
        aid = self.action("Bash", {"command": "touch x"})
        self.led.set_verification(aid, L.V_VERIFIED, L.LEVEL_SELF, "exit 0")
        self.assertEqual(self.ver(aid)["state"], L.V_UNVERIFIED)

    async def test_starts_pending_and_ends_final(self):
        p = os.path.join(self.tmp.name, "f.txt")
        open(p, "w").write("hello")
        aid = self.action("Write", {"file_path": p, "content": "hello"})
        self.assertEqual(self.ver(aid)["state"], L.V_PENDING)
        t = self.q.submit(self.led.actions[aid], {"file_path": p, "content": "hello"})
        self.assertEqual(self.ver(aid)["state"], L.V_PENDING)      # submit() returned at once
        await t
        self.assertEqual(self.ver(aid)["state"], L.V_VERIFIED)

    async def test_6_verification_never_blocks_the_caller(self):
        def slow(inp):
            time.sleep(0.5)
            return V.Outcome(L.V_VERIFIED, L.LEVEL_READBACK, "slow", "t")
        aid = self.action("Write", {})
        old = V.verify_write
        V.verify_write = slow
        try:
            t0 = time.monotonic()
            task = self.q.submit(self.led.actions[aid], {})
            self.assertLess(time.monotonic() - t0, 0.05)
            await task
        finally:
            V.verify_write = old

    async def test_slow_verifier_times_out_as_unverified(self):
        aid = self.action("Write", {})
        old = V.verify_write
        V.verify_write = lambda inp: time.sleep(2) or V.Outcome(L.V_VERIFIED, L.LEVEL_READBACK)
        try:
            v = await self.run_q(aid, {})
        finally:
            V.verify_write = old
        self.assertEqual(v["state"], L.V_UNVERIFIED)
        self.assertIn("timed out", v["evidence"])

    async def test_crashing_verifier_is_unverified_not_an_exception(self):
        aid = self.action("Write", {})
        old = V.verify_write

        def boom(inp):
            raise RuntimeError("x")
        V.verify_write = boom
        try:
            v = await self.run_q(aid, {})
        finally:
            V.verify_write = old
        self.assertEqual(v["state"], L.V_UNVERIFIED)
        self.assertIn("crashed", v["evidence"])


class OnDoneTests(Base):
    """F-06: on_done fires exactly once per submitted action, whether the outcome was known at once or
    came from a worker."""
    async def done_for(self, aid, inp=None):
        seen = []
        self.q.on_done = seen.append
        await self.run_q(aid, inp)
        return seen

    async def test_immediate_outcomes_call_on_done_once(self):
        cases = {"error": self.action("Bash", {}, error=True),
                 "denied": self.action("Bash", {}, denied=True),
                 "read": self.action("Read", {}),
                 "self_report": self.action("Bash", {"command": "curl -X POST x"})}
        for name, aid in cases.items():
            self.assertEqual(await self.done_for(aid), [aid], name)

    async def test_a_worker_outcome_still_calls_on_done_once(self):
        path = os.path.join(self.tmp.name, "w.txt")
        open(path, "w").write("a")
        aid = self.action("Write", {})
        self.assertEqual(await self.done_for(aid, {"file_path": path, "content": "a"}), [aid])

    async def test_a_full_queue_and_a_missing_event_loop_also_call_on_done_once(self):
        aid = self.action("Write", {})
        self.q.max_pending = 0
        self.assertEqual(await self.done_for(aid, {"file_path": "/x", "content": "y"}), [aid])    # queue full
        self.assertEqual(self.ver(aid)["evidence"], "verification queue full; not checked")
        self.q.max_pending = 64
        seen = []
        self.q.on_done = seen.append
        aid2 = self.action("Write", {})
        await asyncio.to_thread(self.q.submit, self.led.actions[aid2], {"file_path": "/x", "content": "y"})
        self.assertEqual(seen, [aid2])                              # no running loop in that thread
        self.assertEqual(self.ver(aid2)["state"], L.V_UNVERIFIED)

    async def test_a_failing_on_done_cannot_break_submit(self):
        def boom(_):
            raise RuntimeError("x")
        self.q.on_done = boom
        aid = self.action("Bash", {}, error=True)
        self.assertIsNone(self.q.submit(self.led.actions[aid], {}))
        self.assertEqual(self.ver(aid)["state"], L.V_NOT_REQUIRED)


class FileVerifierTests(Base):
    async def test_write_verified_by_reading_back(self):
        p = os.path.join(self.tmp.name, "a.txt")
        open(p, "w").write("data")
        aid = self.action("Write", {})
        v = await self.run_q(aid, {"file_path": p, "content": "data"})
        self.assertEqual((v["state"], v["level"]), (L.V_VERIFIED, L.LEVEL_READBACK))

    async def test_write_that_did_not_land_fails_verification(self):
        aid = self.action("Write", {})
        v = await self.run_q(aid, {"file_path": os.path.join(self.tmp.name, "nope"), "content": "x"})
        self.assertEqual(v["state"], L.V_FAILED)

    async def test_write_with_different_content_fails_verification(self):
        p = os.path.join(self.tmp.name, "b.txt")
        open(p, "w").write("other")
        aid = self.action("Write", {})
        v = await self.run_q(aid, {"file_path": p, "content": "data"})
        self.assertEqual(v["state"], L.V_FAILED)

    async def test_edit_verified_and_failed(self):
        p = os.path.join(self.tmp.name, "c.txt")
        aid = self.action("Edit", {})
        open(p, "w").write("alpha NEW omega")                       # the edit changed the file, after the decision
        ok = await self.run_q(aid, {"file_path": p, "old_string": "OLD", "new_string": "NEW"})
        self.assertEqual(ok["state"], L.V_VERIFIED)
        aid2 = self.action("Edit", {})
        bad = await self.run_q(aid2, {"file_path": p, "old_string": "alpha", "new_string": "ZZZ"})
        self.assertEqual(bad["state"], L.V_FAILED)


class BoardVerifierTests(Base):
    def serve(self, state):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(state).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def cmd(self, body, returned_at=None):
        # the action returned a moment ago; the queue puts these times into a verifier's input (F-07)
        return {"command": "barehands/bin/board.sh '" + json.dumps(body) + "'",
                V._TIMES_KEY: (time.time() - 5, time.time() - 5 if returned_at is None else returned_at)}

    async def test_card_on_the_board_is_verified(self):
        port = self.serve({"items": [{"title": "TRICERATOPS"}], "ts": time.time()})
        out = V.verify_board(self.cmd({"a": "add_card", "title": "TRICERATOPS"}), port=port, wait_s=1)
        self.assertEqual((out.state, out.level), (L.V_VERIFIED, L.LEVEL_READBACK))

    async def test_card_missing_from_a_live_board_fails(self):
        port = self.serve({"items": [{"title": "OTHER"}], "ts": time.time()})
        out = V.verify_board(self.cmd({"a": "present", "title": "TRICERATOPS"}), port=port, wait_s=0.6)
        self.assertEqual(out.state, L.V_FAILED)

    async def test_dark_board_is_unverified_not_failed(self):
        out = V.verify_board(self.cmd({"a": "add_card", "title": "X"}), port=1, wait_s=0.3)
        self.assertEqual(out.state, L.V_UNVERIFIED)

    async def test_board_actions_without_a_readback_stay_unverified(self):
        out = V.verify_board(self.cmd({"a": "lock", "title": "X"}), port=1, wait_s=0.1)
        self.assertEqual((out.state, out.level), (L.V_UNVERIFIED, L.LEVEL_SELF))


if __name__ == "__main__":
    unittest.main()
