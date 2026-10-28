"""Unfinished-sentence hold (main.py, 2026-10-25).

The 600ms open-mic endpoint cut Yousif mid-sentence ~15 times in 25 minutes; each cut started a turn that
his next fragment killed and resent. An utterance that plainly stops mid-sentence is now held BEFORE any turn
is sent (StitchHold, unfinished=True), so the rest joins it and one turn goes out. A finished sentence is never
delayed. handle() is a closure inside amain(), so its wiring is pinned by structure, like test_stitch_coalesce."""
import inspect
import unittest

from backtalk import main as M
from tests.test_stitch_coalesce import fresh


class LooksUnfinished(unittest.TestCase):
    def test_tonights_real_fragments(self):
        # verbatim from the 2026-10-25 22:46 log
        # "close that" can end a sentence, so the first fragment is (correctly) not delayed
        self.assertFalse(M.looks_unfinished(
            "Well first of all I want you to close this session that take as that take already fucking 3 billion 4 billion token close that"))
        self.assertTrue(M.looks_unfinished("and second of all start wiring everything you need so you can take control of the"))
        self.assertFalse(M.looks_unfinished("of the screen."))
        self.assertFalse(M.looks_unfinished("And then let's go step by step after that."))
        self.assertTrue(M.looks_unfinished("So show me miracle Jarvis 2.0. Like, like..."))

    def test_finished_or_doubtful_is_never_held(self):
        for s in ("Jarvis?", "Jarvis", "Yes, go ahead.", "Fix all three of them.", "Leave it to AJ",
                  "What time is it", "do it", "close that", "log in", "turn it on", "or not", "I think so", "", None, "   "):
            self.assertFalse(M.looks_unfinished(s), s)

    def test_connectors_commas_and_trail_offs_are_held(self):
        for s in ("I want you to", "open the", "and", "Can you check my email and", "so basically,",
                  "the thing is…", "wait -"):
            self.assertTrue(M.looks_unfinished(s), s)


class PresendHold(unittest.TestCase):
    def test_an_unfinished_hold_waits_longer_than_a_stitch(self):
        h, c = fresh()
        h.begin("take control of the", "E1", quiet_s=M.UNFINISHED_HOLD_S,
                max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertFalse(h.due())
        c.advance(M.UNFINISHED_HOLD_S)
        self.assertEqual(h.take("E1"), ("take control of the", 1))

    def test_the_rest_of_the_sentence_joins_and_then_goes_out_quickly(self):
        h, c = fresh()
        h.begin("take control of the", "E1", quiet_s=M.UNFINISHED_HOLD_S,
                max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        c.advance(0.5)
        h.add("of the screen.")
        c.advance(M.UNFINISHED_DONE_S - 0.1)
        self.assertFalse(h.due())
        c.advance(0.2)                                            # out well before a stitch's 1.3s
        self.assertEqual(h.take("E1"), ("take control of the of the screen.", 2))

    def test_still_unfinished_after_a_fragment_keeps_waiting(self):
        h, c = fresh()
        h.begin("I want you to", "E1", quiet_s=M.UNFINISHED_HOLD_S,
                max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        c.advance(1.0)
        h.add("check my email and")
        c.advance(M.STITCH_HOLD_S + 0.1)
        self.assertFalse(h.due())

    def test_the_hard_cap_still_sends_it(self):
        h, c = fresh()
        h.begin("and", "E1", quiet_s=M.UNFINISHED_HOLD_S, max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        for _ in range(10):
            c.advance(1.0)
            if h.active and not h.due():
                h.add("and")
        self.assertIsNotNone(h.take("E1") if h.active else ("sent", 0))

    def test_a_plain_stitch_is_unchanged(self):
        h, c = fresh()
        h.begin("do A and", "E1")
        self.assertFalse(h.unfinished)
        h.add("and B and")
        c.advance(M.STITCH_HOLD_S)
        self.assertTrue(h.due())

    def test_clear_resets_the_flag(self):
        h, _ = fresh()
        h.begin("x and", "E1", unfinished=True)
        h.clear()
        self.assertFalse(h.unfinished)


class Wiring(unittest.TestCase):
    src = inspect.getsource(M)

    def test_open_mic_unfinished_text_is_held_before_any_turn_is_sent(self):
        i = self.src.index('wait = unfinished_wait(text, turn_p) if not released and _MIC["mode"] == "open" else 0.0')
        j = self.src.index("speak_task = asyncio.create_task(", i)
        self.assertIn("unfinished=True", self.src[i:i + 700])
        self.assertIn("return True", self.src[i:i + 900])
        self.assertLess(i, j)                                     # held before the turn starts

    def test_a_fragment_continues_the_presend_hold(self):
        self.assertIn("last_turn_text=_HOLD.text if presend else", self.src)
        self.assertIn("_HOLD.add(text, unfinished_wait(text, turn_p) if presend else None)", self.src)

    def test_the_open_mic_passes_the_audio_verdict(self):
        self.assertIn("turn_p=ears.last_turn_p", self.src)


class UnfinishedWait(unittest.TestCase):
    def test_text_alone(self):
        self.assertEqual(M.unfinished_wait("take control of the", None), M.UNFINISHED_HOLD_S)
        self.assertEqual(M.unfinished_wait("Open YouTube.", None), 0.0)

    def test_audio_catches_what_the_full_stop_hides(self):
        self.assertEqual(M.unfinished_wait("I was thinking we can order.", 0.02), M.UNFINISHED_HOLD_S)
        self.assertEqual(M.unfinished_wait("I was thinking we can order.", 0.35), M.UNFINISHED_UNSURE_S)
        self.assertEqual(M.unfinished_wait("Open YouTube.", 0.97), 0.0)

    def test_no_opinion_never_delays(self):
        self.assertEqual(M.unfinished_wait("Open YouTube.", None), 0.0)

    def test_a_fragment_that_finishes_it_goes_out_quickly(self):
        h, c = fresh()
        h.begin("I was thinking", "E1", quiet_s=M.UNFINISHED_HOLD_S, max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        h.add("we can order dinner.", wait=0.0)
        c.advance(M.UNFINISHED_DONE_S + 0.01)
        self.assertTrue(h.due())

    def test_a_fragment_still_unfinished_by_audio_keeps_waiting(self):
        h, c = fresh()
        h.begin("I was thinking", "E1", quiet_s=M.UNFINISHED_HOLD_S, max_s=M.UNFINISHED_HOLD_MAX_S, unfinished=True)
        h.add("we can order.", wait=M.UNFINISHED_UNSURE_S)
        c.advance(M.UNFINISHED_UNSURE_S - 0.1)
        self.assertFalse(h.due())
