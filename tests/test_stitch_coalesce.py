"""Stitch coalescing (main.py, 2026-10-05).

A stitch used to cancel the turn and resend the accumulated text for EVERY fragment. A burst of
fragments therefore sent several queries back to back, the router answered each abandoned one with a
CLI interrupt, and two interrupts a few ms apart killed the CLI (reproduced with the real CLI:
AbortError -> stream_lost). A stitch now retracts the turn once and HOLDS the accumulated request until
no fragment has arrived for STITCH_HOLD_S, then sends ONE replacement.

What is real here and what is pinned by structure:
  * StitchHold      the REAL class, driven by a fake clock: accumulation, ordering, the quiet time, the
                    hard cap, "send at most once", epoch refusal, and that it owns no timer or task.
  * release_held    the REAL coroutine the main loop calls when the hold may be due, with a fake handle()
                    and a fake brain (the epoch).
  * decide_during_turn  the REAL pure decision: the 20 s window and an explicit stop are unchanged.
  * handle() and the main loop are closures inside amain() and cannot be called from a test, so the parts
    of them that matter are pinned by structure, the way tests/test_voice_responsiveness.py does it.
Nothing here touches the live voice process, the production log, ledger or session file."""
import asyncio
import inspect
import types
import unittest
from unittest import mock

from backtalk import main as M


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def fresh():
    c = Clock()
    return M.StitchHold(clock=c), c


class ConstantsTests(unittest.TestCase):
    def test_the_stitch_window_is_still_twenty_seconds(self):
        self.assertEqual(M.STITCH_MAX_S, 20.0)

    def test_the_hold_is_the_documented_quiet_time_with_a_hard_cap(self):
        self.assertGreaterEqual(M.STITCH_HOLD_S, 1.2)
        self.assertLessEqual(M.STITCH_HOLD_S, 1.5)
        self.assertEqual(M.STITCH_HOLD_MAX_S, 3.0)
        self.assertGreater(M.STITCH_HOLD_MAX_S, M.STITCH_HOLD_S)


class CoalescingTests(unittest.TestCase):
    def test_three_fragments_inside_a_second_are_one_replacement_in_order(self):
        h, c = fresh()
        h.begin("KOD Jarvis, check the bridge.", "E1")
        c.advance(0.3)
        h.add("HUD.")
        c.advance(0.3)
        h.add("God.")
        c.advance(0.3)                       # 0.9 s after the first: still inside the quiet time
        self.assertFalse(h.due())
        self.assertIsNone(h.take("E1"))      # not due: nothing is sent and nothing is lost
        c.advance(M.STITCH_HOLD_S)
        self.assertTrue(h.due())
        self.assertEqual(h.take("E1"), ("KOD Jarvis, check the bridge. HUD. God.", 3))
        self.assertFalse(h.active)

    def test_five_or_more_fragments_inside_the_window_are_still_one_replacement(self):
        h, c = fresh()
        h.begin("one", "E1")
        for word in ("two", "three", "four", "five", "six"):
            c.advance(0.2)
            h.add(word)
        c.advance(M.STITCH_HOLD_S)
        self.assertEqual(h.take("E1"), ("one two three four five six", 6))
        self.assertIsNone(h.take("E1"))      # and it can only be sent once

    def test_every_fragment_extends_the_quiet_time(self):
        h, c = fresh()
        h.begin("a", "E1")
        c.advance(1.0)
        h.add("b")
        c.advance(1.0)                       # 2.0 s after the first, 1.0 s after the last
        self.assertFalse(h.due())
        self.assertAlmostEqual(h.remaining(), M.STITCH_HOLD_S - 1.0, places=6)
        c.advance(0.4)
        self.assertTrue(h.due())

    def test_fragments_separated_by_more_than_the_quiet_time_are_separate_requests(self):
        h, c = fresh()
        h.begin("first request", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertEqual(h.take("E1"), ("first request", 1))
        c.advance(0.5)
        h.begin("second request", "E1")      # a later stitch starts a new hold, with no carry-over
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertEqual(h.take("E1"), ("second request", 1))

    def test_steady_chatter_cannot_starve_the_request(self):
        h, c = fresh()
        h.begin("start", "E1")
        t0 = c.t
        sent_at = None
        for i in range(40):                  # a fragment every second, for 40 s
            c.advance(1.0)
            if h.due():
                sent_at = c.t - t0
                break
            h.add(f"w{i}")
        self.assertIsNotNone(sent_at)
        self.assertLessEqual(sent_at, M.STITCH_HOLD_MAX_S + 1e-6)
        self.assertEqual(h.take("E1")[1] >= 2, True)
        self.assertFalse(h.active)           # what arrives after that starts a new hold

    def test_the_deadline_never_passes_the_hard_cap(self):
        h, c = fresh()
        h.begin("a", "E1")
        c.advance(2.9)
        h.add("b")
        self.assertLessEqual(h.remaining(), 0.1 + 1e-6)
        c.advance(0.1 + 1e-6)
        self.assertTrue(h.due())              # due at the cap even though a fragment just arrived

    def test_no_duplicated_text(self):
        h, c = fresh()
        h.begin("the retracted turn text and the first new fragment", "E1")
        h.add("second")
        h.add("third")
        c.advance(M.STITCH_HOLD_S + 0.1)
        text, n = h.take("E1")
        self.assertEqual(text, "the retracted turn text and the first new fragment second third")
        self.assertEqual(text.count("the retracted turn text"), 1)
        self.assertEqual(n, 3)
        h.begin("fresh", "E1")               # a new hold starts empty: nothing left over from the last
        self.assertEqual(h.text, "fresh")

    def test_add_without_a_hold_is_an_error_not_a_silent_new_request(self):
        h, _ = fresh()
        with self.assertRaises(RuntimeError):
            h.add("orphan")


class StaleStateTests(unittest.TestCase):
    def test_it_owns_no_timer_and_no_task(self):
        src = inspect.getsource(M.StitchHold)
        for needle in ("asyncio", "create_task", "call_later", "call_at", "Timer", "threading"):
            self.assertNotIn(needle, src, needle)

    def test_a_replaced_or_cleared_request_can_never_be_sent_later(self):
        h, c = fresh()
        h.begin("old request", "E1")
        h.clear()                            # a new turn, a stop, a console command, "that's all"
        c.advance(60)
        self.assertFalse(h.due())
        self.assertIsNone(h.take("E1"))
        self.assertIsNone(h.remaining())

    def test_a_replacement_hold_cannot_be_sent_with_the_old_text(self):
        h, c = fresh()
        h.begin("old", "E1")
        h.clear()
        c.advance(0.5)
        h.begin("new", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertEqual(h.take("E1"), ("new", 1))

    def test_a_request_said_in_one_session_is_never_sent_into_another(self):
        h, c = fresh()
        h.begin("said in epoch one", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertIsNone(h.take("E2"))      # the session was replaced: dropped, and
        self.assertFalse(h.active)           # nothing remains to fire later
        c.advance(60)
        self.assertIsNone(h.take("E1"))

    def test_remaining_is_none_when_nothing_is_held(self):
        h, _ = fresh()
        self.assertIsNone(h.remaining())
        self.assertFalse(h.due())


class ReleaseTests(unittest.IsolatedAsyncioTestCase):
    def brain(self, epoch="E1"):
        return types.SimpleNamespace(router=types.SimpleNamespace(epoch=epoch))

    def recorder(self, verdict=True):
        calls = []

        async def handle(text, spoke_from=None, released=False):
            calls.append((text, released))
            return verdict
        return handle, calls

    async def test_a_due_hold_sends_exactly_one_replacement_with_every_fragment(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("a b", "E1")
        h.add("c")
        h.add("d")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertTrue(await M.release_held(h, self.brain(), handle))
        self.assertEqual(calls, [("a b c d", True)])
        self.assertTrue(await M.release_held(h, self.brain(), handle))   # woken again: nothing more to send
        self.assertEqual(len(calls), 1)

    async def test_a_hold_that_is_not_due_sends_nothing_and_keeps_its_text(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("a b", "E1")
        c.advance(0.5)
        self.assertTrue(await M.release_held(h, self.brain(), handle))
        self.assertEqual(calls, [])
        self.assertTrue(h.active)
        self.assertEqual(h.text, "a b")

    async def test_a_hold_from_a_replaced_session_is_dropped_not_sent(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("said before the rebuild", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertTrue(await M.release_held(h, self.brain("E2"), handle))
        self.assertEqual(calls, [])
        self.assertFalse(h.active)

    async def test_a_wake_before_the_hold_is_due_has_no_side_effects_at_all(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("a b", "E1")
        c.advance(0.4)
        with mock.patch.object(M, "log") as log, mock.patch.object(M.signals, "set_state") as state:
            self.assertTrue(await M.release_held(h, self.brain(), handle))
        log.assert_not_called()               # no "dropped" line for a request that is merely still held
        state.assert_not_called()             # and the UI is not flipped to idle while it waits
        self.assertEqual(calls, [])
        self.assertTrue(h.active)

    async def test_a_dropped_hold_goes_idle_and_says_why(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("said before the rebuild", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        with mock.patch.object(M, "log") as log, mock.patch.object(M.signals, "set_state") as state:
            await M.release_held(h, self.brain("E2"), handle)
        state.assert_called_once_with("idle")
        self.assertIn("session changed", log.call_args[0][0])
        self.assertEqual(calls, [])

    async def test_a_cleared_hold_never_fires(self):
        h, c = fresh()
        handle, calls = self.recorder()
        h.begin("a", "E1")
        h.clear()
        c.advance(60)
        await M.release_held(h, self.brain(), handle)
        self.assertEqual(calls, [])

    async def test_a_quit_from_handle_is_passed_up(self):
        h, c = fresh()
        handle, calls = self.recorder(verdict=False)
        h.begin("goodbye", "E1")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertFalse(await M.release_held(h, self.brain(), handle))


class DecisionUnchangedTests(unittest.TestCase):
    """The pure decision handle() asks is untouched: the 20 s window and an explicit stop."""

    def d(self, text, *, busy=False, elapsed=5.0, last="the request", age=1.0):
        return M.decide_during_turn(text, mouth_busy=busy, elapsed_s=elapsed,
                                    last_turn_text=last, last_turn_age_s=age)

    def test_fragments_inside_twenty_seconds_still_stitch(self):
        self.assertEqual(self.d("and also the notes", age=0.5), "stitch")
        self.assertEqual(self.d("and also the notes", age=M.STITCH_MAX_S), "stitch")

    def test_the_window_still_expires_after_twenty_seconds(self):
        self.assertEqual(self.d("and also the notes", age=M.STITCH_MAX_S + 0.1), "interrupt")
        self.assertEqual(self.d("and also the notes", last=""), "interrupt")

    def test_an_explicit_stop_is_still_a_stop_not_a_fragment(self):
        for t in ("stop", "wait", "Jarvis, stop!", "okay wait", "hold on", "shut up", "enough", "quiet"):
            self.assertEqual(self.d(t, age=0.5), "stop", t)


class WiringTests(unittest.TestCase):
    """handle() and the main loop are closures; their stitch/hold wiring is pinned by structure."""

    @classmethod
    def setUpClass(cls):
        with open(M.__file__, encoding="utf-8") as fh:
            cls.src = fh.read()
        a = cls.src.index("async def handle(text")
        cls.handle_src = cls.src[a:cls.src.index("def over_speech(", a)]
        i = cls.handle_src.index('if action == "stitch":')
        cls.stitch = cls.handle_src[i:cls.handle_src.index("        if speak_task:", i)]
        j = cls.stitch.index("\n            else:\n")
        cls.stitch_branch, cls.interrupt_branch = cls.stitch[:j], cls.stitch[j:]

    def test_handle_still_asks_the_pure_decision_exactly_once(self):
        self.assertEqual(self.handle_src.count("decide_during_turn("), 1)

    def test_a_stitch_retracts_the_turn_once_and_holds_instead_of_resending(self):
        b = self.stitch_branch
        self.assertIn("_HOLD.begin(", b)
        self.assertIn("_HOLD.add(", b)
        self.assertIn("speak_task.cancel()", b)
        self.assertIn("hold = True", b)
        for banned in ("speak_reply(", "create_task", "brain.reset_turn", "brain.interrupt"):
            self.assertNotIn(banned, b, banned)

    def test_a_held_fragment_returns_before_any_new_turn_is_started(self):
        i = self.handle_src.index("        if hold:")
        block = self.handle_src[i:i + 700]
        self.assertIn("return True", block)
        self.assertLess(i, self.handle_src.index("await brain.reset_turn()"))
        self.assertLess(i, self.handle_src.index("asyncio.create_task(\n            speak_reply("))

    def test_a_replaced_turn_or_a_stop_ends_the_hold(self):
        self.assertIn("_HOLD.clear()", self.interrupt_branch)
        i = self.handle_src.index('if action == "stop":')
        stop = self.handle_src[i:self.handle_src.index('if action == "checkin":', i)]
        self.assertIn("_HOLD.clear()", stop)
        self.assertLess(stop.index("_HOLD.clear()"), stop.index("await stop_current_turn("))
        self.assertIn("await stop_current_turn(brain, mouth, speak_task,", stop)
        self.assertNotIn("speak_reply(", stop)

    def test_a_released_request_is_not_logged_or_judged_as_new_speech(self):
        self.assertIn("released: bool = False", self.handle_src)
        self.assertIn("if not released:\n            log(f\"[you]", self.handle_src)
        self.assertIn("_HOLD.active and not released", self.handle_src)

    def test_the_main_loop_wakes_on_the_hold_with_no_timer_of_its_own(self):
        self.assertIn("timeout=_HOLD.remaining()", self.src)
        self.assertIn("await release_held(_HOLD, brain, handle)", self.src)
        i = self.src.index("timeout=_HOLD.remaining()")
        loop_part = self.src[i - 200:i + 400]
        for banned in ("call_later", "call_at", "Timer(", "create_task", "ensure_future"):
            self.assertNotIn(banned, loop_part, banned)
            self.assertNotIn(banned, inspect.getsource(M.release_held), banned)

    def test_the_end_of_a_conversation_and_a_console_command_end_the_hold(self):
        i = self.src.index('if d.reason == "end_conversation":')
        self.assertIn("_HOLD.clear()", self.src[i:i + 900])
        j = self.src.index("async def _run_console_inner(verb):")
        self.assertIn("_HOLD.clear()", self.src[j:j + 300])


if __name__ == "__main__":
    unittest.main()
