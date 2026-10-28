import asyncio
import os
import tempfile
import unittest
from datetime import date
from unittest import mock

import backtalk.brief as B
import backtalk.brain as brain_mod
import backtalk.vaultrecall as VR


def _vault(index=True, prio=True, notes=True, daily=True, **over):
    d = tempfile.mkdtemp()
    def w(rel, txt):
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(txt)
    if index:
        w("VAULT-INDEX.md", "## Who I Am\n\nI'm Yousif. I wholesale.\n\n"
          "## My Preferences for Working with AI\n\n- **Be direct.** No hedging. More.\n"
          "- **Do it right.** Not v2.\n\n## Other\n")
    if prio:
        w("Active Priorities.md", "### Open Tasks\n\n" + "".join(
            f"- [ ] **Item {i}** (meta) — details {i}\n" for i in range(20))
          + "\n### Completed Tasks\n\n- [x] **Done** (meta)\n")
    if notes:
        for _, rel in B.PROJECT_NOTES:
            w(rel, "# N\n\n**Status: live and working.**\n")
    if daily:
        t = date.today().isoformat()
        w(f"01 - Daily Notes/m/{t}.md", "## Index\n\n- **Topic** — did a thing.\n\n## Session 1\n")
    for k, v in over.items():
        w(k, v)
    return d


class BriefTests(unittest.TestCase):
    def build(self, vault, **kw):
        with mock.patch.object(B, "VAULT", vault):
            return B.build_session_brief(**kw)

    def test_sections_present(self):
        out = self.build(_vault())
        for s in ("[USER]", "[CURRENT PRIORITIES]", "[ACTIVE PROJECTS]",
                  "[RECENT CONTEXT]", "[SESSION]", "no permissions"):
            self.assertIn(s, out)
        self.assertIn("Item 0", out)
        self.assertNotIn("Item 6", out)          # bounded to 6
        self.assertNotIn("Done", out)            # completed section excluded

    def test_hard_size_limit(self):
        big = "x" * 20000
        v = _vault(**{"VAULT-INDEX.md": "## Who I Am\n\n" + big + "\n"})
        out = self.build(v)
        self.assertLessEqual(len(out), B.MAX_CHARS)
        out = self.build(_vault(), max_chars=600)
        self.assertLessEqual(len(out), 600)

    def test_missing_sources_do_not_crash(self):
        out = self.build(tempfile.mkdtemp())
        self.assertIn("JARVIS SESSION BRIEF", out)
        self.assertNotIn("[USER]", out)

    def test_malformed_or_empty_sources(self):
        v = _vault(**{"VAULT-INDEX.md": "", "Active Priorities.md": "\x00garbage ### ##",
                      "09 - Resources/Barehands.md": "no status here"})
        out = self.build(v)
        self.assertIn("[SESSION]", out)
        with open(os.path.join(v, "09 - Resources/Voice Line (Backtalk).md"), "wb") as f:
            f.write(b"\xff\xfe\xfa bad utf8")
        self.assertIsInstance(self.build(v), str)

    def test_old_daily_note_omitted(self):
        d = _vault(daily=False)
        p = os.path.join(d, "01 - Daily Notes/m/2020-01-01.md")
        os.makedirs(os.path.dirname(p))
        with open(p, "w") as f:
            f.write("## Index\n\n- old\n")
        self.assertNotIn("[RECENT CONTEXT]", self.build(d))

    def test_no_secrets_or_raw_paths_logged(self):
        v = _vault(**{"VAULT-INDEX.md": "## Who I Am\n\nSECRETBIO123\n"})
        logs = []
        with mock.patch.object(B, "log", logs.append):
            self.build(v)
        joined = " ".join(logs)
        self.assertIn("[brief] loaded sections=", joined)
        self.assertNotIn("SECRETBIO123", joined)

    def test_disabled(self):
        with mock.patch.dict(B.CFG, {"session_brief": {"enabled": False}}):
            self.assertEqual(self.build(_vault()), "")


class InjectionTests(unittest.TestCase):
    # ONE loop per test: the router's reader task lives on the loop that started
    # it, so start(), ask_stream() and stop() must all run on the same one.
    def setUp(self):
        self._loop = asyncio.new_event_loop()

    def tearDown(self):
        self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _brain(self, resume=None):
        opts = []
        # The client contract is now receive_messages() + a uuid'd query()
        # (router.py); the shared fake answers every query with one clean run.
        from tests import router_fakes as RF

        class FakeClient(RF.FakeClient):
            def __init__(s, options):
                super().__init__()
                opts.append(options)
                s.on_query = lambda msg: s.feed(*RF.fg_run(msg["uuid"], "ok."))
        b = brain_mod.WarmBrain(resume_id=resume)
        return b, opts, FakeClient

    def test_injected_once_at_start_not_per_turn(self):
        b, opts, Fake = self._brain()
        with mock.patch.object(brain_mod, "ClaudeSDKClient", Fake), \
             mock.patch.object(brain_mod, "build_session_brief",
                               return_value="BRIEF-X") as bsb:
            self._run(b.start())
            self.assertEqual(bsb.call_count, 1)
            self.assertIn("BRIEF-X", opts[0].system_prompt["append"])
            self.assertTrue(opts[0].system_prompt["append"].startswith(
                brain_mod.DISCIPLINE))
            # a user turn goes through ask_stream, which never rebuilds it
            async def go():
                async for _ in b.ask_stream("hello there"):
                    pass
                await b.stop()
            self._run(go())
            self.assertEqual(bsb.call_count, 1)

    def test_resume_still_works_and_gets_brief(self):
        b, opts, Fake = self._brain(resume="abcdef123456")
        with mock.patch.object(brain_mod, "ClaudeSDKClient", Fake), \
             mock.patch.object(brain_mod, "build_session_brief",
                               return_value="BRIEF-X") as bsb:
            self._run(b.start())
            self._run(b.stop())
        self.assertEqual(opts[0].resume, "abcdef123456")
        self.assertIn("BRIEF-X", opts[0].system_prompt["append"])
        self.assertEqual(bsb.call_count, 1)

    def test_brief_failure_never_blocks_start(self):
        b, opts, Fake = self._brain()
        with mock.patch.object(brain_mod, "ClaudeSDKClient", Fake), \
             mock.patch.object(brain_mod, "build_session_brief",
                               side_effect=RuntimeError("boom")):
            self._run(b.start())
            self._run(b.stop())
        self.assertEqual(opts[0].system_prompt["append"], brain_mod.DISCIPLINE)

    def test_discipline_no_longer_contradicts_boot(self):
        self.assertNotIn("Skip any startup sequence", brain_mod.DISCIPLINE)
        self.assertIn("SESSION BRIEF", brain_mod.DISCIPLINE)
        self.assertIn("safety", brain_mod.DISCIPLINE)

    def test_vaultrecall_unchanged_and_independent(self):
        self.assertIsNone(VR.context_for("too short"))
        self.assertEqual((VR.MIN_SIM, VR.MAX_INJECT_TOKENS, VR.TOP_N,
                          VR.MIN_WORDS), (0.6, 2000, 4, 5))
        self.assertFalse(hasattr(VR, "build_session_brief"))


if __name__ == "__main__":
    unittest.main()
