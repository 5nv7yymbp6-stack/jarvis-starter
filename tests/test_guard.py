"""Destructive-command guard: classifier table, spoken yes/no, fail-closed paths, wiring."""
import asyncio
import os
import unittest

from backtalk import guard as G

HOME = os.path.expanduser("~")

BLOCK = [
    "sudo rm file", "ls && sudo reboot", "echo hi; sudo -s", "bash -c 'sudo ls'", "echo $(sudo whoami)", "env FOO=1 sudo ls", "/usr/bin/sudo ls",
    "rm -rf /", "rm -rf ~", "rm -rf ~/", "rm -rf $HOME", "rm -rf ${HOME}/*", "rm -fr /Users", "rm -rf " + HOME, "rm -r /*", "rm --recursive --force ~", "sudo-less; rm -rf /",
    "git push --force", "git push -f origin main", "git push origin +main", "git push --force-with-lease", "git -C repo push -fu origin x", "cd x && git push origin main -f",
    "security find-generic-password -s x", "security find-internet-password -a y", "security dump-keychain", "security export -k login.keychain",
    "sh -c \"git push --force\"", "`sudo id`",
    "bash -lc 'sudo ls'", "zsh -ic \"rm -rf ~\"", "eval \"sudo ls\"", "eval git push -f", "security -v find-generic-password -s x", "security -q -p x dump-keychain",
    "git --git-dir x push --force", "git -c a=b push -f",
]
ALLOW = [
    "ls -la", "rm file.txt", "rm -rf ~/Desktop/jarvis/tmp-scratch", "rm -rf /tmp/x", "rm -rf ./build", "git push origin main", "git push -u origin feature", "git push --follow-tags",
    "git status", "echo sudo", "echo 'git push --force'", "grep -r sudo .", "security list-keychains", "pseudo ls", "python3 -c 'print(1)'", "", "rm -i ~/old.txt",
    "git log --grep push -f", "git commit -m push -f", "command -v sudo", "bash -lc 'ls -la'", "eval echo hi", "security -v list-keychains",
    "security find-certificate -a",
]


class Classifier(unittest.TestCase):
    def test_blocked(self):
        for c in BLOCK:
            self.assertTrue(G.classify_command(c), c)

    def test_allowed(self):
        for c in ALLOW:
            self.assertIsNone(G.classify_command(c), c)

    def test_yes_words(self):
        for t in ["yes", "Yes.", "yeah go ahead", "go ahead", "do it", "Yes, boss.", "Yeah, go ahead and do it.", "yes please", "go for it", "yep yep"]:
            self.assertTrue(G.is_yes(t), t)
        for t in ["", "no", "yes but wait", "don't do it", "not yet", "what", "go and ahead", "yesterday", "stop, yes", "maybe",
                  "yes, what's that?", "yes what is it deleting", "sure why", "ok", "boss", "Sure, proceed with the other one", "yes do the other thing"]:
            self.assertFalse(G.is_yes(t), t)


def hook(cmd, tool="Bash"):
    return {"tool_name": tool, "tool_input": {"command": cmd}}


def decision(out):
    return (out.get("hookSpecificOutput") or {}).get("permissionDecision")


class SpokenTarget(unittest.TestCase):
    def test_the_question_names_the_target(self):
        self.assertEqual(G.classify_command("git push -f origin main"), "that's a force push to origin main")
        self.assertEqual(G.classify_command("rm -rf ~"), "that deletes your whole home folder")
        self.assertEqual(G.classify_command("rm -rf /"), "that deletes the whole disk")
        self.assertEqual(G.classify_command("sudo reboot"), "that runs sudo reboot")
        self.assertEqual(G.classify_command("security dump-keychain"), "that dumps the keychain")

    def test_credentials_and_urls_are_never_spoken(self):
        r = G.classify_command("git push --force https://user:tok3n@github.com/a/b.git main")
        self.assertNotIn("tok3n", r)
        self.assertNotIn("github", r)
        self.assertIn("a remote address", r)
        self.assertNotIn("x@y", G.classify_command("git push -f x@y:repo main"))


class Hook(unittest.IsolatedAsyncioTestCase):
    async def test_harmless_and_non_bash_pass_untouched(self):
        g = G.Guard(say=lambda t: self.fail("must not ask"))
        self.assertEqual(await g.pre_tool(hook("ls"), "t", None), {})
        self.assertEqual(await g.pre_tool(hook("sudo ls", tool="Read"), "t", None), {})

    async def test_yes_allows_and_question_is_spoken(self):
        said = []
        g = G.Guard(say=said.append, timeout_s=5)
        task = asyncio.create_task(g.pre_tool(hook("git push --force"), "t", None))
        await asyncio.sleep(0.05)
        self.assertTrue(g.pending)
        self.assertIn("force push", said[0])
        self.assertTrue(g.answer("yes go ahead", None))
        self.assertEqual(decision(await task), "allow")
        self.assertFalse(g.pending)

    async def test_no_other_answer_denies(self):
        for ans in ("no", "what was that", "yes but wait"):
            g = G.Guard(say=lambda t: None, timeout_s=5)
            task = asyncio.create_task(g.pre_tool(hook("sudo ls"), "t", None))
            await asyncio.sleep(0.05)
            g.answer(ans, None)
            self.assertEqual(decision(await task), "deny", ans)

    async def test_silence_times_out_to_deny(self):
        g = G.Guard(say=lambda t: None, timeout_s=0.1)
        self.assertEqual(decision(await g.pre_tool(hook("sudo ls"), "t", None)), "deny")
        self.assertFalse(g.pending)

    async def test_no_voice_denies(self):
        self.assertEqual(decision(await G.Guard(say=None).pre_tool(hook("sudo ls"), "t", None)), "deny")

    async def test_speech_that_began_before_the_question_is_not_an_answer(self):
        g = G.Guard(say=lambda t: None, timeout_s=5)
        task = asyncio.create_task(g.pre_tool(hook("sudo ls"), "t", None))
        await asyncio.sleep(0.05)
        self.assertFalse(g.answer("yes", g.asked_at - 5))
        self.assertTrue(g.pending)
        g.deny_pending()
        self.assertEqual(decision(await task), "deny")

    async def test_second_dangerous_command_while_asking_is_denied_not_queued(self):
        g = G.Guard(say=lambda t: None, timeout_s=5)
        first = asyncio.create_task(g.pre_tool(hook("sudo a"), "t", None))
        await asyncio.sleep(0.05)
        self.assertEqual(decision(await g.pre_tool(hook("sudo b"), "t2", None)), "deny")
        self.assertTrue(g.pending)
        g.answer("yes", None)
        self.assertEqual(decision(await first), "allow")

    async def test_a_talk_key_answer_stamped_with_monotonic_time_approves(self):
        import time
        g = G.Guard(say=lambda t: None, timeout_s=5)
        task = asyncio.create_task(g.pre_tool(hook("sudo ls"), "t", None))
        await asyncio.sleep(0.05)
        press_t = time.monotonic()                      # exactly what main.py passes as spoke_from on the talk key
        self.assertTrue(g.answer("yes", press_t))
        self.assertEqual(decision(await task), "allow")

    async def test_a_raising_voice_denies(self):
        def boom(t):
            raise RuntimeError("audio device gone")
        self.assertEqual(decision(await G.Guard(say=boom).pre_tool(hook("sudo ls"), "t", None)), "deny")

    async def test_a_classifier_error_denies(self):
        from unittest import mock
        with mock.patch.object(G, "classify_command", side_effect=ValueError("bug")):
            self.assertEqual(decision(await G.Guard(say=lambda t: None).pre_tool(hook("ls"), "t", None)), "deny")

    async def test_cancellation_while_asking_denies_and_clears_the_question(self):
        g = G.Guard(say=lambda t: None, timeout_s=5)
        task = asyncio.create_task(g.pre_tool(hook("sudo ls"), "t", None))
        await asyncio.sleep(0.05)
        task.cancel()
        self.assertEqual(decision(await task), "deny")
        self.assertFalse(g.pending)

    async def test_the_log_never_carries_the_command(self):
        from unittest import mock
        lines = []
        with mock.patch.object(G, "log", side_effect=lines.append):
            g = G.Guard(say=lambda t: None, timeout_s=0.05)
            await g.pre_tool(hook("git push --force https://u:SECRET@h/x main"), "tu1", None)
        self.assertTrue(lines and all("SECRET" not in x and "tu1" in x for x in lines), lines)
        self.assertIn("denied", lines[-1])

    async def test_disabled_guard_passes_everything(self):
        self.assertEqual(await G.Guard(say=None, enabled=False).pre_tool(hook("sudo ls"), "t", None), {})


class TalkKey(unittest.IsolatedAsyncioTestCase):
    """The talk key during an open safety question answers it instead of cancelling it (mic test 2026-10-22 found every
    talk-key answer pre-empted: the press cancelled the turn, the hook was cancelled and denied before the words were heard)."""

    async def asyncSetUp(self):
        from backtalk import main as M
        self.M = M
        self.turn = asyncio.create_task(asyncio.sleep(30))          # a turn in flight
        self.addCleanup(self.turn.cancel)

    async def test_a_press_during_an_open_question_keeps_the_question_and_captures_the_answer(self):
        import time
        g = G.Guard(say=lambda t: None, timeout_s=5)
        q = asyncio.create_task(g.pre_tool(hook("sudo -n true"), "tu_a", None))
        await asyncio.sleep(0.05)
        self.assertFalse(self.M.press_cancels_turn(self.turn, g))  # the turn (and with it the question) survives the press
        press_t = time.monotonic()
        await asyncio.sleep(0.05)                                   # he talks, releases; the words arrive
        self.assertTrue(g.answer("Yes.", press_t))
        self.assertEqual(decision(await q), "allow")
        self.assertFalse(self.turn.done())

    async def test_a_press_with_no_open_question_still_interrupts(self):
        self.assertTrue(self.M.press_cancels_turn(self.turn, G.Guard(say=lambda t: None)))

    async def test_a_press_with_no_turn_in_flight_cancels_nothing(self):
        self.assertFalse(self.M.press_cancels_turn(None, G.Guard(say=lambda t: None)))

    async def test_talk_key_no_ambiguous_and_empty_capture_deny(self):
        import time
        for ans in ("No.", "yes, what's that?", None):             # None = tap, empty transcript or an ears error
            g = G.Guard(say=lambda t: None, timeout_s=5)
            q = asyncio.create_task(g.pre_tool(hook("sudo -n true"), "tu_b", None))
            await asyncio.sleep(0.05)
            self.assertFalse(self.M.press_cancels_turn(self.turn, g))
            if ans is None:
                g.deny_pending()                                    # what main.py does on an empty capture
            else:
                g.answer(ans, time.monotonic())
            self.assertEqual(decision(await q), "deny", ans)


class Binding(unittest.IsolatedAsyncioTestCase):
    async def test_yes_approves_only_the_pending_call_and_the_next_risky_call_asks_again(self):
        said = []
        g = G.Guard(say=said.append, timeout_s=5)
        first = asyncio.create_task(g.pre_tool(hook("sudo -n true"), "tu_1", None))
        await asyncio.sleep(0.05)
        self.assertIn("tool_use_id=tu_1", g.pending_call)
        g.answer("yes", None)
        self.assertEqual(decision(await first), "allow")
        self.assertIsNone(g.pending_call)
        self.assertFalse(g.answer("yes", None))                     # a late or repeated yes has nothing to approve
        second = asyncio.create_task(g.pre_tool(hook("sudo whoami"), "tu_2", None))
        await asyncio.sleep(0.05)
        self.assertEqual(len(said), 2)                              # asked again: no inherited approval
        self.assertIn("tool_use_id=tu_2", g.pending_call)
        g.answer("no", None)
        self.assertEqual(decision(await second), "deny")

    async def test_a_modified_command_is_a_new_call_with_its_own_question(self):
        g = G.Guard(say=lambda t: None, timeout_s=0.1)
        a = asyncio.create_task(g.pre_tool(hook("git push -f origin main"), "tu_x", None))
        await asyncio.sleep(0.02)
        tag_a = g.pending_call
        g.answer("yes", None)
        self.assertEqual(decision(await a), "allow")
        b = await g.pre_tool(hook("git push -f origin other"), "tu_y", None)   # nobody answers this one
        self.assertEqual(decision(b), "deny")
        self.assertIn("tool_use_id=tu_x", tag_a)                    # the yes was recorded against tu_x only


class Wiring(unittest.TestCase):
    def test_brain_registers_a_bash_pretooluse_hook_and_main_routes_answers(self):
        pkg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtalk")
        with open(os.path.join(pkg, "brain.py")) as f:
            b = f.read()
        with open(os.path.join(pkg, "main.py")) as f:
            m = f.read()
        self.assertIn('hooks["PreToolUse"] = [HookMatcher(matcher="Bash", hooks=[self.guard.pre_tool]', b)
        self.assertIn("brain.guard.pending and brain.guard.answer(text, spoke_from)", m)
        self.assertIn("if press_cancels_turn(speak_task, brain.guard):", m)          # the press handler uses the guard-aware rule
        self.assertEqual(m.count("speak_task.cancel()          # the button = interrupt"), 1)
        self.assertIn("brain.guard.deny_pending()", m)                                 # empty capture during a question = no
