"""The existing voice behavior that must stay untouched."""
import unittest

import backtalk.main as M
from backtalk.addressee import AddresseeGate


class ExistingVoiceBehaviourTests(unittest.TestCase):
    def test_stop_phrases_still_recognised(self):
        for t in ("stop", "Jarvis!", "okay wait", "hold on", "shut up"):
            self.assertTrue(M._is_stop_phrase(t), t)
        self.assertFalse(M._is_stop_phrase("Jarvis, do it"))

    def test_checkin_shortcut_still_recognised(self):
        self.assertTrue(M._is_checkin("you there?"))
        self.assertTrue(M._is_checkin("did you find it"))
        self.assertFalse(M._is_checkin("open safari"))

    def test_console_and_zip_verbs_still_match(self):
        self.assertEqual(M.console_match("zip it"), "zip")
        self.assertEqual(M.console_match("unzip"), "unzip")
        self.assertEqual(M.console_match("go hands free"), "micopen")
        self.assertIsNone(M.console_match("what's the weather"))

    def test_echo_filters_still_work(self):
        self.assertTrue(M._is_echo("Right here, boss.", ["Right here, boss."]))
        self.assertFalse(M._is_echo("open safari please now",
                                    ["Right here, boss."]))

    def test_model_control_tags_still_parse(self):
        self.assertEqual(M._console_direction_verb("effort:low"), "effort:low")
        self.assertEqual(M._console_direction_verb("model:deep"), "deep")

    def test_ordinary_harmless_conversation_reaches_the_agent_and_runs_tools(self):
        g = AddresseeGate.from_config({}, "Jarvis")
        self.assertTrue(g.check("Jarvis, what's the time?").accepted)
        self.assertTrue(g.check("and in Tokyo?").accepted)       # follow-up


if __name__ == "__main__":
    unittest.main()
