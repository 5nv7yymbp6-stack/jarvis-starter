"""Backchannel classification (backtalk/turnclass.py) and its two wiring sites in main.py (default OFF)."""
import ast
import inspect
import unittest
from types import SimpleNamespace

from backtalk import main as M
from backtalk import turnclass as TC


class ClassifierTests(unittest.TestCase):
    def test_acknowledgements_are_backchannels(self):
        for t in ("Okay.", "okay", "Mm-hmm.", "mm hmm", "Uh-huh", "uh huh", "Yeah", "Right.", "I see.", "Got it", "Thank you.",
                  "Thanks!", "Oh okay", "Cool", "Oh, nice.", "OK cool", "hmm", "mhm", "Wow", "Oh I see"):
            self.assertTrue(TC.is_backchannel(t), t)

    def test_requests_questions_and_longer_speech_are_not(self):
        for t in ("what time is it", "okay do it", "okay so what about the leads", "can you check that", "yeah that is wrong",
                  "check the vault", "okay thanks jarvis", "I see you found it", "", "   ", "...", None):
            self.assertFalse(TC.is_backchannel(t), t)

    def test_more_than_three_words_is_never_a_backchannel(self):
        self.assertFalse(TC.is_backchannel("okay okay okay okay"))
        self.assertFalse(TC.is_backchannel("oh okay cool"))                # three words are allowed only if the phrase is on the list
        self.assertFalse(TC.is_backchannel("yeah yeah yeah"))              # not on the list

    def test_stop_cancel_and_negative_words_are_never_backchannels(self):
        for t in ("stop", "wait", "no", "nope", "okay stop", "okay wait", "hold on", "enough", "pause", "quiet", "cancel", "never mind",
                  "not okay", "no thanks", "okay no", "don't"):
            self.assertFalse(TC.is_backchannel(t), t)

    def test_the_agents_name_means_it_is_addressing_not_acknowledging(self):
        for t in ("okay jarvis", "Jarvis", "thanks boss", "yes sir", "hey okay", "okay please"):
            self.assertFalse(TC.is_backchannel(t), t)

    def test_an_answer_word_after_a_question_is_an_answer_not_a_backchannel(self):
        for t in ("yes", "yeah", "yep", "sure", "okay", "ok", "right", "alright", "exactly", "fine", "good", "perfect", "oh okay"):
            self.assertFalse(TC.is_backchannel(t, last_spoken_was_question=True), t)
            self.assertTrue(TC.is_backchannel(t, last_spoken_was_question=False), t)

    def test_non_answer_acknowledgements_stay_backchannels_even_after_a_question(self):
        for t in ("mm-hmm", "uh huh", "hmm", "I see", "got it", "wow", "interesting", "thank you"):
            self.assertTrue(TC.is_backchannel(t, last_spoken_was_question=True), t)

    def test_the_list_is_self_consistent(self):
        for phrase in TC.BACKCHANNELS:
            ws = TC.words(phrase)
            self.assertTrue(1 <= len(ws) <= TC.MAX_WORDS, phrase)
            self.assertFalse(set(ws) & TC.NEVER_WORDS, phrase)
            self.assertTrue(TC.is_backchannel(phrase), phrase)

    def test_no_existing_stop_vocabulary_can_ever_be_a_backchannel(self):
        vocab = set(M._STOP_WORDS) | set(M._STOP_PHRASES) | set(M.QUIT_PHRASES)
        self.assertTrue(vocab)
        for v in vocab:
            self.assertFalse(TC.is_backchannel(v), v)
            self.assertFalse(TC.is_backchannel(f"okay {v}"), v)
            self.assertFalse(TC.is_backchannel(f"{v} okay"), v)

    def test_spoken_was_question(self):
        self.assertTrue(TC.spoken_was_question(["It is done.", "Want me to read it?"]))
        # review 2026-10-19: a question that is NOT the last chunk still makes "yes" an answer
        self.assertTrue(TC.spoken_was_question(["Want me to send it?", "I'll draft it meanwhile."]))
        self.assertTrue(TC.spoken_was_question(['He asked, "Send it?"']))                   # a trailing quote
        self.assertTrue(TC.spoken_was_question(["Send it? (yes or no)?)"]))
        self.assertFalse(TC.spoken_was_question(["Done.", "All set."]))
        self.assertFalse(TC.spoken_was_question([]))
        self.assertFalse(TC.spoken_was_question(None))
        self.assertTrue(TC.spoken_was_question(["  Say it again?  ", "  "]))               # blank trailing chunks are ignored

    def test_the_module_is_pure(self):
        tree = ast.parse(inspect.getsource(TC))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module)
        self.assertEqual(mods, {"re"})


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(M.CFG)

    def tearDown(self):
        M.CFG.clear()
        M.CFG.update(self.saved)

    def mouth(self, chunks=()):
        return SimpleNamespace(recent_speech=lambda w=30.0: list(chunks))

    def test_it_is_off_by_default_and_only_literal_true_turns_it_on(self):
        M.CFG.pop("conversation", None)
        self.assertFalse(M._backchannel_on())
        for v in ("true", 1, "yes", None, [], {}):
            M.CFG["conversation"] = {"backchannel": v}
            self.assertFalse(M._backchannel_on(), v)
        M.CFG["conversation"] = "garbage"
        self.assertFalse(M._backchannel_on())
        M.CFG["conversation"] = {"backchannel": True}
        self.assertTrue(M._backchannel_on())

    def test_off_means_even_an_obvious_backchannel_is_not_classified(self):
        M.CFG["conversation"] = {"backchannel": False}
        self.assertFalse(M._is_backchannel_now("okay", self.mouth()))

    def test_on_uses_the_voices_last_chunk_to_tell_an_answer_from_an_acknowledgement(self):
        M.CFG["conversation"] = {"backchannel": True}
        M._BC_LAST.update(text=None, at=0.0)
        self.assertTrue(M._is_backchannel_now("okay", self.mouth(["Here is the result."]), now=10.0))
        self.assertFalse(M._is_backchannel_now("okay", self.mouth(["Want me to run it?"]), now=100.0))
        self.assertTrue(M._is_backchannel_now("mm-hmm", self.mouth(["Want me to run it?"]), now=200.0))

    def test_with_nothing_recently_spoken_an_answer_word_is_not_swallowed(self):
        M.CFG["conversation"] = {"backchannel": True}
        M._BC_LAST.update(text=None, at=0.0)
        self.assertFalse(M._is_backchannel_now("okay", self.mouth([]), now=100.0))          # the voice may be mid-question
        self.assertTrue(M._is_backchannel_now("mm-hmm", self.mouth([]), now=100.0))

    def test_the_same_acknowledgement_twice_within_eight_seconds_is_meant_and_let_through(self):
        M.CFG["conversation"] = {"backchannel": True}
        M._BC_LAST.update(text=None, at=0.0)
        m = self.mouth(["Here is the result."])
        self.assertTrue(M._is_backchannel_now("okay", m, now=100.0))
        self.assertFalse(M._is_backchannel_now("Okay.", m, now=104.0))                       # a repeat: let it through
        self.assertTrue(M._is_backchannel_now("okay", m, now=105.0))                         # and the memory was reset
        self.assertTrue(M._is_backchannel_now("okay", m, now=200.0))                         # long after: swallowed again
        self.assertTrue(M._is_backchannel_now("mm-hmm", m, now=201.0))                       # a different one is independent

    def test_a_broken_mouth_never_raises_and_means_not_a_backchannel(self):
        M.CFG["conversation"] = {"backchannel": True}
        self.assertFalse(M._is_backchannel_now("okay", None))
        self.assertFalse(M._is_backchannel_now("okay", SimpleNamespace()))

    def test_handle_checks_before_the_voice_is_flushed_and_never_cancels_anything(self):
        src = inspect.getsource(M.amain)
        h = src[src.index("async def handle("):src.index("async def handle(") + 6000]
        i_check, i_flush = h.index("_is_backchannel_now(text, mouth)"), h.index("if mouth.held:\n            # words, not noise")
        self.assertLess(i_check, i_flush)                                    # before the reply is flushed as 'words, not noise'
        body_start = h.index("backchannel swallowed", i_check)                 # the swallow BODY, after the condition
        block = h[body_start:h.index("return True", body_start) + 12]
        for banned in ("cancel(", "shut_up", "_HOLD", "brain.", "stop_current_turn"):
            self.assertNotIn(banned, block, banned)
        self.assertIn("mouth.release()", block)                              # a paused voice carries on

    # review 2026-10-19, findings 2 and 3
    def test_only_the_open_mic_call_site_is_eligible_and_a_held_stitch_is_not(self):
        src = inspect.getsource(M.amain)
        h = src[src.index("async def handle("):src.index("async def handle(") + 6000]
        self.assertIn("backchannel_ok: bool = False", h)
        self.assertIn("if (backchannel_ok and not released and not _HOLD.active", h)
        calls = [l.strip() for l in src.splitlines() if "await handle(" in l]
        self.assertEqual(len([c for c in calls if "backchannel_ok=True" in c]), 1)      # exactly one site: the open mic
        typed = [c for c in calls if c == "if text and not await handle(text):"]
        self.assertEqual(len(typed), 1)                                                  # typed input: not eligible
        self.assertTrue([c for c in calls if "spoke_from=press_t" in c and "backchannel_ok" not in c])   # the talk key: not eligible
        self.assertNotIn("backchannel_ok", inspect.getsource(M.release_held))             # a released stitch is never eligible

    def test_over_speech_has_the_same_check_before_the_cut_in(self):
        src = inspect.getsource(M.amain)
        o = src[src.index("def over_speech("):src.index("def on_barge(")]
        self.assertLess(o.index("_is_backchannel_now(text, mouth)"), o.index("cut in over the voice"))
        self.assertIn("return False", o[o.index("_is_backchannel_now"):o.index("cut in over the voice")])

    def test_the_decision_table_and_the_single_send_are_untouched(self):
        self.assertEqual(M.decide_during_turn("okay", mouth_busy=True, elapsed_s=1.0, last_turn_text="x", last_turn_age_s=3.0),
                         "stitch")                                           # the pure decision is unchanged
        self.assertEqual(inspect.getsource(M.speak_reply).count("brain.ask_stream("), 1)
        self.assertEqual(inspect.getsource(M.amain).count("decide_during_turn("), 1)


class ElongationAndRepetitionTests(unittest.TestCase):
    """Live 2026-10-19: 'Mmm.' was swallowed but 'Mmmmmm. Mmmmmm.' cut the voice off, because the list was exact-match only."""

    def test_the_live_miss_is_now_a_backchannel(self):
        self.assertTrue(TC.is_backchannel("Mmmmmm. Mmmmmm."))
        self.assertTrue(TC.is_backchannel("Mmmmmm."))
        self.assertTrue(TC.is_backchannel("Mmm."))

    def test_elongated_listed_words(self):
        for t in ("hmmmmm", "okayyy", "yeahhh", "ooooh", "ahhh", "uhhh huhhh", "coooool", "greeeat", "mhmmmm",
                  "mm-hmm mm-hmm", "mm mm mm", "hmmm hmmm"):
            self.assertTrue(TC.is_backchannel(t), t)

    def test_stretching_never_hides_a_stop_or_a_request(self):
        for t in ("nooo", "stoppp", "waaait", "noooo noooo", "pleaseee", "jarvisss", "cancel cancel", "okay stoppp", "mmm no"):
            self.assertFalse(TC.is_backchannel(t), t)

    def test_repetition_is_only_of_one_listed_acknowledgement(self):
        for t in ("mm okay", "okay thanks thanks", "dog dog", "banana banana", "mm mm mm mm", "send it send it", "okay okay",
                  "yeah yeah", "right right", "nice nice"):
            self.assertFalse(TC.is_backchannel(t), t)

    def test_the_question_guard_still_applies_to_stretched_and_repeated_answers(self):
        self.assertFalse(TC.is_backchannel("yeahhh", last_spoken_was_question=True))
        self.assertFalse(TC.is_backchannel("mm yeahhh", last_spoken_was_question=True))
        self.assertTrue(TC.is_backchannel("mmmmm", last_spoken_was_question=True))           # a hum is not an answer

    def test_unlisted_words_are_not_rescued_by_collapsing(self):
        for t in ("maybe", "reallyyy", "sooo", "weird", "sorry"):
            self.assertFalse(TC.is_backchannel(t), t)


if __name__ == "__main__":
    unittest.main()
