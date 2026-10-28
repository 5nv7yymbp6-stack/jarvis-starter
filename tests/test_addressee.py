import re
import unittest
from pathlib import Path

from backtalk.addressee import ACTIVE, IDLE, AddresseeGate


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def gate(mode="wake", window=180.0, max_active=300.0):
    c = Clock()
    g = AddresseeGate(["jarvis", "hey jarvis"], ["jarvus", "jervis"], mode,
                      window, clock=c, max_active_s=max_active)
    return g, c


class WakeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.g, _ = gate()

    def test_addressed_sentences_wake(self):
        for t in ("Jarvis, what's the weather?", "Hey Jarvis, open YouTube.",
                  "Jarvis can you help me?", "Jarvis open safari",
                  "okay Jarvis what time is it", "Jarvis", "Jarvis?",
                  "Jarvus open safari", "Jervis what time is it"):
            self.assertTrue(self.g.is_wake(t), t)

    def test_mentions_of_the_name_do_not_wake(self):
        for t in ("I wasn't talking to you, Jarvis.",
                  "My friend said Jarvis is cool.",
                  "Did you see the Jarvis video?",
                  "Jarvis was mentioned in that movie.",
                  "Jarvis is a great assistant",
                  "open YouTube, Jarvis",          # known cost: trailing vocative
                  "the harvest is late", "travis said hi",
                  "what's the weather like today"):
            self.assertFalse(self.g.is_wake(t), t)


class IdleTests(unittest.TestCase):
    def test_A_idle_rejects_ordinary_speech(self):
        g, _ = gate()
        d = g.check("I think the weather looks nice today")
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, "idle_no_wake")
        self.assertEqual(g.state, IDLE)

    def test_B_idle_accepts_a_wake_phrase(self):
        g, _ = gate()
        d = g.check("Jarvis, what's the weather?")
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, "wake")
        self.assertEqual(g.state, ACTIVE)

    def test_C_idle_rejects_mention_only(self):
        g, _ = gate()
        self.assertFalse(g.check("I wasn't talking to you, Jarvis.").accepted)
        self.assertEqual(g.state, IDLE)

    def test_D_idle_rejects_third_party_mention(self):
        g, _ = gate()
        self.assertFalse(g.check("My friend said Jarvis is cool.").accepted)
        self.assertEqual(g.state, IDLE)

    def test_no_busy_admission_exists(self):
        """Speech is never admitted just because the voice is speaking: the
        gate has no busy input at all, and IDLE rejects."""
        g, _ = gate()
        with self.assertRaises(TypeError):
            g.check("random background talk", busy=True)
        self.assertFalse(g.check("random background talk").accepted)

    def test_J_exempt_stop_over_voice_does_not_activate(self):
        g, _ = gate()
        d = g.check("stop", exempt=True)          # caller: stop phrase + voice on
        self.assertTrue(d.accepted)
        self.assertEqual(g.state, IDLE)


class ActiveTests(unittest.TestCase):
    def test_E_F_followups_need_no_wake(self):
        g, c = gate()
        g.check("Jarvis, what's the weather?")
        c.t += 20
        self.assertTrue(g.check("What about tomorrow?").accepted)
        c.t += 20
        d = g.check("And San Jose?")
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, "active")

    def test_G_explicit_end_returns_to_idle_and_is_not_sent(self):
        g, c = gate()
        g.check("Jarvis, what's the weather?")
        d = g.check("That's all.")
        self.assertFalse(d.accepted)               # never reaches the brain
        self.assertEqual(d.reason, "end_conversation")
        self.assertEqual(g.state, IDLE)
        c.t += 5
        self.assertFalse(g.check("What about tomorrow?").accepted)

    def test_all_end_phrases(self):
        for phrase in ("That's all", "I'm done", "Stop listening", "Go idle",
                       "Never mind", "nevermind", "that will be all",
                       "okay, that's all, Jarvis", "Jarvis, never mind"):
            g, _ = gate()
            g.check("Jarvis hello")
            d = g.check(phrase)
            self.assertEqual(d.reason, "end_conversation", phrase)
            self.assertEqual(g.state, IDLE, phrase)

    def test_dismissal_of_the_agent_ends_it(self):
        g, _ = gate()
        g.check("Jarvis hello")
        d = g.check("No, I wasn't talking to you, Jarvis.")
        self.assertEqual(d.reason, "end_conversation")
        self.assertEqual(g.state, IDLE)

    def test_end_phrase_in_idle_is_just_rejected(self):
        g, _ = gate()
        d = g.check("that's all")
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, "idle_no_wake")

    def test_K_wake_followup_end_followup_rejected(self):
        g, c = gate()
        self.assertTrue(g.check("Jarvis what time is it").accepted)
        c.t += 10
        self.assertTrue(g.check("and in Tokyo").accepted)
        c.t += 10
        self.assertEqual(g.check("never mind").reason, "end_conversation")
        c.t += 10
        self.assertFalse(g.check("and in London").accepted)
        c.t += 10
        self.assertTrue(g.check("Jarvis, London please").accepted)   # re-wake

    def test_idle_window_expires_180s_after_last_human_turn(self):
        g, c = gate()
        g.check("Jarvis hello")
        c.t += 179
        self.assertTrue(g.check("still here").accepted)
        c.t += 181
        self.assertFalse(g.check("what about tomorrow").accepted)
        self.assertEqual(g.state, IDLE)

    def test_H_agent_reply_cannot_refresh_the_timer(self):
        """The gate has no touch(): only a human follow-up moves the idle
        window. A reply finishing (or being spoken) is invisible to it."""
        g, c = gate()
        self.assertFalse(hasattr(g, "touch"))
        g.check("Jarvis, tell me a story")
        c.t += 179           # a long spoken reply happens; nothing calls the gate
        c.t += 2
        self.assertEqual(g.state, IDLE)
        self.assertFalse(g.check("and then?").accepted)

    def test_main_does_not_touch_the_gate_on_reply_or_handle(self):
        src = (Path(__file__).resolve().parent.parent
               / "backtalk" / "main.py").read_text()
        self.assertNotIn("_ADDRESSEE.touch", src)
        self.assertNotRegex(src, r"add_done_callback\([^)]*_ADDRESSEE")
        self.assertNotIn("busy=busy", src)

    def test_I_ambient_speech_cannot_slide_the_window_forever(self):
        g, c = gate(window=180, max_active=300)
        g.check("Jarvis hello")
        accepted_until = None
        for _ in range(40):                # ambient chatter every 30s for 20min
            c.t += 30
            if g.check("some unrelated chatter in the room").accepted:
                accepted_until = c.t - 1000
        self.assertLessEqual(accepted_until, 300)   # hard cap from the wake
        self.assertEqual(g.state, IDLE)
        self.assertFalse(g.check("more chatter").accepted)

    def test_rewake_restarts_the_hard_cap(self):
        g, c = gate()
        g.check("Jarvis hello")
        c.t += 250
        g.check("Jarvis, one more thing")          # re-wake at t=250: cap -> 550
        c.t += 170
        self.assertTrue(g.check("and another thing").accepted)   # t=420
        c.t += 100                        # t=520: past the FIRST cap (300)
        self.assertTrue(g.check("and one more").accepted)
        c.t += 100                        # t=620: past the SECOND cap (550)
        self.assertFalse(g.check("and yet another").accepted)

    def test_hard_cap_logged_as_max_lifetime(self):
        import backtalk.addressee as a
        seen = []
        orig, a.log = a.log, seen.append
        try:
            g, c = gate()
            g.check("Jarvis hello")
            for _ in range(9):
                c.t += 30
                g.check("chatter")
            c.t += 60
            g.check("chatter")
        finally:
            a.log = orig
        self.assertTrue(any("cause=max_lifetime" in s for s in seen), seen)


class PendingAndExemptTests(unittest.TestCase):
    def test_a_bare_yes_is_not_admitted_without_a_wake_phrase(self):
        # There is no pending-question exemption: an idle gate admits only a wake phrase.
        g, c = gate()
        c.t += 5000
        d = g.check("yes")
        self.assertFalse(d.accepted)
        self.assertEqual(g.state, IDLE)

    def test_zip_unzip_exempt_and_never_activate(self):
        g, _ = gate()
        self.assertTrue(g.check("unzip", exempt=True).accepted)
        self.assertEqual(g.state, IDLE)

    def test_explicit_open_for_held_key_and_typed(self):
        g, c = gate()
        g.open("held_key")
        self.assertEqual(g.state, ACTIVE)
        self.assertTrue(g.check("and what about tomorrow").accepted)

    def test_dev_always_mode(self):
        g, _ = gate(mode="always")
        self.assertTrue(g.check("random talk").accepted)

    def test_from_config(self):
        g = AddresseeGate.from_config({}, "Jarvis")
        self.assertTrue(g.check("hey jarvis").accepted)
        self.assertEqual(g.max_active, 300.0)
        g2 = AddresseeGate.from_config(
            {"addressee": {"wake_words": ["computer"], "active_window_s": 5,
                           "max_active_s": 9}}, "Jarvis")
        self.assertFalse(g2.check("jarvis hi").accepted)
        self.assertTrue(g2.check("computer hi").accepted)
        self.assertEqual(g2.max_active, 9.0)

    def test_logs_never_contain_the_transcript(self):
        import backtalk.addressee as a
        seen = []
        orig, a.log = a.log, seen.append
        try:
            gate()[0].check("my secret password is hunter2")
        finally:
            a.log = orig
        self.assertTrue(seen)
        self.assertNotIn("hunter2", " ".join(seen))


if __name__ == "__main__":
    unittest.main()
