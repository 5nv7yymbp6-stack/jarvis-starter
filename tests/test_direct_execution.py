"""Tool calls run directly. There is no permission tier, no spoken "do you want me to?", no pending
confirmation, no approval callback and no PreToolUse hook standing between the model and a tool.

These tests pin that down so an approval step cannot come back unnoticed."""
import importlib
import inspect
import unittest

import backtalk.main as M
from backtalk import brain as B, config
from backtalk.addressee import AddresseeGate
from tests import router_fakes as F
from tests.test_brain_router import Base

# calls the old gate classified as consequential or dangerous (Tier 2 / Tier 3)
RISKY = (
    ("Bash", {"command": "rm -rf /tmp/backtalk-direct-test"}),
    ("Bash", {"command": "sudo ls /"}),
    ("Bash", {"command": "git push origin main"}),
    ("mcp__gmail__send_message", {"to": "a@example.com", "body": "hi"}),
    ("Write", {"file_path": "/tmp/backtalk-direct-test/BOOT-CONFIG.md", "content": "x"}),
    ("WebFetch", {"url": "https://example.com"}),
)


class ToolCallsRunDirectlyTests(Base):
    async def test_the_session_is_opened_with_no_approval_machinery(self):
        opts = self.clients[0].options
        self.assertEqual(opts.permission_mode, "bypassPermissions")
        self.assertIsNone(opts.can_use_tool)
        pre = (opts.hooks or {}).get("PreToolUse") or []           # only the short Bash deny-list guard (2026-10-22), nothing else
        self.assertEqual([(m.matcher, m.hooks) for m in pre], [("Bash", [self.brain.guard.pre_tool])])

    async def test_ordinary_and_risky_tool_calls_return_without_any_question(self):
        said = []
        self.brain.on_say = said.append
        for n, (tool, inp) in enumerate(RISKY):
            tuid = f"toolu_direct_{n}"
            self.script = lambda c, m, tuid=tuid, tool=tool, inp=inp: c.feed(*F.fg_run(
                m["uuid"], "Done.",
                tools=[F.tool_use(tuid, tool, inp), F.tool_result(tuid, "ok")]))
            self.assertEqual(await self.ask(f"go {n}"), ["Done."])
            row = self.led.actions[self.led.action_id_for(tuid)]
            self.assertEqual(row["state"], "RETURNED_OK", tool)
            self.assertNotIn("hook_decision", row)
            self.assertNotIn("tier", row)
        self.assertEqual(said, [])                       # nothing was asked out loud
        self.assertEqual(self.client.sent[-1]["type"], "user")

    async def test_a_bare_yes_is_an_ordinary_turn_because_nothing_is_waiting_on_it(self):
        self.assertEqual(await self.ask("yes"), ["Hello there.", "How are you?"])
        self.assertEqual(self.client.sent[-1]["type"], "user")


class RemovedSurfaceTests(unittest.TestCase):
    def test_the_permissions_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("backtalk.permissions")

    def test_the_brain_takes_no_approval_arguments(self):
        params = set(inspect.signature(B.WarmBrain.__init__).parameters)
        self.assertFalse(params & {"can_use_tool", "pre_tool_hook", "hook_timeout_s"}, params)
        self.assertFalse(hasattr(B.WarmBrain, "set_permission_mode"))

    def test_main_has_no_gate_state_or_voice_toggle(self):
        for name in ("make_permission_gate", "_deny_pending", "_perm_pending", "_confirm_pending",
                     "_GATE_CONFIRM", "_PERM", "_CONFIRM", "_AUTOAPPROVE", "ConfirmationGate"):
            self.assertFalse(hasattr(M, name), name)
        self.assertNotIn("noask", M.CONSOLE_VERBS)
        self.assertNotIn("ask", M.CONSOLE_VERBS)
        self.assertIsNone(M.console_match("stop asking for permission"))
        self.assertIsNone(M.console_match("start asking again"))
        self.assertIsNone(M._console_direction_verb("permission:noask"))
        self.assertIsNone(M._console_direction_verb("permission:ask"))

    def test_the_config_has_no_permission_settings(self):
        for key in ("permission_mode", "permissions"):
            self.assertNotIn(key, config.DEFAULTS, key)
            self.assertNotIn(key, config.CFG, key)

    def test_the_addressee_gate_has_no_pending_confirmation_path(self):
        self.assertNotIn("pending_confirmation", inspect.signature(AddresseeGate.check).parameters)
        g = AddresseeGate.from_config({}, "Jarvis")
        d = g.check("yes")
        self.assertFalse(d.accepted)
        self.assertNotEqual(d.reason, "pending_confirmation")
        self.assertTrue(g.check("Jarvis, delete the old drafts").accepted)   # wake phrase still works

    def test_wake_word_stop_phrase_and_voice_verbs_are_untouched(self):
        self.assertTrue(M._is_stop_phrase("hold on"))
        self.assertEqual(M.console_match("go hands free"), "micopen")
        self.assertEqual(M.console_match("zip it"), "zip")
        self.assertEqual(M._console_direction_verb("effort:low"), "effort:low")


if __name__ == "__main__":
    unittest.main()
