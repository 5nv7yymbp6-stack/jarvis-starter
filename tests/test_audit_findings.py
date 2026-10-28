"""Regression tests for the open findings of the 2026-10-06 trust-layer audit (F-03, F-04, F-10, F-12 to F-15), reproduced and fixed
2026-10-21. Each test states the failing scenario it pins."""
import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest

from backtalk import evidence as E
from backtalk import ledger as L
from backtalk import verify as V
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_brain_router import Base


class F15AbortedTailTests(Base):
    """F-15: an answer the router ended as ABORTED / INTERRUPTED must not have its half-finished buffered tail spoken."""

    async def test_an_aborted_turns_partial_sentence_is_not_yielded(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Hello there. How a"))
        gen = self.brain.ask_stream("q")
        first = await gen.__anext__()
        self.assertEqual(first, "Hello there.")
        handle = next(iter(self.brain.router.handles.values()))
        handle.finish(L.T_ABORTED, "superseded_by_next_turn")
        rest = [s async for s in gen]
        self.assertEqual(rest, [])                                  # was ["How a"]: a fragment of an answer that no longer stands

    async def test_an_interrupted_turn_drops_its_tail_but_a_finished_turn_keeps_it(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("One. Two. Thr"))
        gen = self.brain.ask_stream("q")
        self.assertEqual(await gen.__anext__(), "One.")
        handle = next(iter(self.brain.router.handles.values()))
        handle.finish(L.T_INTERRUPTED, "interrupt")
        self.assertEqual([s async for s in gen], [])                # was ["Two.", "Thr"]: sentences the cut-in answer never got to say
        # a turn that completed normally still flushes its last, un-punctuated words
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there. Fine"))
        self.assertEqual(await self.ask("again"), ["Hello there.", "Fine"])

    async def test_a_drop_is_recorded_in_the_ledger_so_it_is_not_silent(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Hello there. How a"))
        gen = self.brain.ask_stream("q")
        await gen.__anext__()
        handle = next(iter(self.brain.router.handles.values()))
        handle.finish(L.T_ABORTED, "superseded_by_next_turn")
        [s async for s in gen]
        notes = [n for n in self.led.notes if n.get("note") == "aborted_answer_tail_dropped"]
        self.assertEqual(len(notes), 1)
        self.assertEqual((notes[0]["state"], notes[0]["chars"] > 0), (L.T_ABORTED, True))


class ProtectedBargeInTest(Base):
    async def test_a_muted_protected_turn_stays_muted(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("One. Two. Thr"))
        gen = self.brain.ask_stream("q")
        self.assertEqual(await gen.__anext__(), "One.")
        handle = next(iter(self.brain.router.handles.values()))
        handle.abandoned = True                                     # router.interrupt(): a protected barge-in mutes the run and lets it finish
        self.assertEqual([s async for s in gen], [])                # was ["Two.", "Thr"]: stale speech over the user's new request


class F13VerdictTests(unittest.TestCase):
    """F-13 end to end: the task verdict for a summary that equals only one of several conflicting counts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.addCleanup(self.led.close)
        self.e = self.led.epoch_open("s1", 1)

    def task_with_counts(self, *counts):
        self.led.task_new = getattr(self.led, "task_new")
        tid = self.led.task_new(self.e, "tk1", "toolu_task", "count the notes", "BG_AGENT") if hasattr(self.led, "task_new") else None
        return tid

    def verdict(self, summary, counts):
        # build the shape verify_task_result reads: a task row with a tool_use_id, and child actions carrying count facts
        row = {"id": "tk1", "tool_use_id": "toolu_task", "claims": E.extract_claims(summary)}
        self.led.tasks["tk1"] = dict(row, history=[])
        for i, n in enumerate(counts):
            aid = f"A{i}"
            self.led.actions[aid] = {"id": aid, "state": L.A_OK, "parent_tool_use_id": "toolu_task", "tool_use_id": f"toolu_c{i}",
                                     "evidence": {"kind": "count", "n": n, "group": "item"}, "history": []}
        written = []
        self.led.set_task_verification = lambda tid, state, level, evidence="", verifier="", **extra: written.append((state, evidence, extra)) or True
        self.led.task_set = lambda *a, **k: None
        out = V.verify_task_result(self.led, "tk1")
        return out, written

    def test_21_and_20_with_a_summary_of_20_is_unverified_and_says_why(self):
        orig = E.fact_of
        E.fact_of = lambda ev: (ev["n"], ev["group"]) if isinstance(ev, dict) and "n" in ev else orig(ev)
        try:
            out, written = self.verdict("20 daily note files found", [21, 20])
        finally:
            E.fact_of = orig
        self.assertEqual(out.state, L.V_UNVERIFIED)
        self.assertIn("disagree", out.evidence)
        self.assertIn("21", out.evidence)
        self.assertEqual(written[-1][2]["ambiguous"], [20])

    def test_a_single_matching_count_is_still_verified_and_a_wrong_one_still_fails(self):
        orig = E.fact_of
        E.fact_of = lambda ev: (ev["n"], ev["group"]) if isinstance(ev, dict) and "n" in ev else orig(ev)
        try:
            self.assertEqual(self.verdict("21 daily note files found", [21])[0].state, L.V_VERIFIED)
            self.assertEqual(self.verdict("20 daily note files found", [21])[0].state, L.V_FAILED)
        finally:
            E.fact_of = orig


class F12LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.addCleanup(self.led.close)
        self.e = self.led.epoch_open("s1", 1)

    def action(self, tuid="toolu_1"):
        return self.led.action_seen(tuid, self.e, "Write")

    def test_verified_is_refused_for_every_action_state_except_ok(self):
        # an action still REQUESTED, one that errored, one that was denied, one whose result was lost
        pending = self.action("toolu_a")
        errored = self.action("toolu_b")
        self.led.action_result("toolu_b", is_error=True, content="boom")
        denied = self.action("toolu_c")
        self.led.action_state(denied, L.A_DENIED)
        lost = self.action("toolu_d")
        self.led.action_state(lost, L.A_LOST)
        for aid in (pending, errored, denied, lost):
            self.led.set_verification(aid, L.V_VERIFIED, L.LEVEL_READBACK, "read back fine", "file")
            v = self.led.actions[aid]["verification"]
            self.assertEqual(v["state"], L.V_UNVERIFIED, self.led.actions[aid]["state"])
            self.assertIn("cannot be verified", v["evidence"])

    def test_an_ok_action_can_still_be_verified(self):
        aid = self.action()
        self.led.action_result("toolu_1", is_error=False, content="ok")
        self.led.set_verification(aid, L.V_VERIFIED, L.LEVEL_READBACK, "hash matches", "file")
        self.assertEqual(self.led.actions[aid]["verification"]["state"], L.V_VERIFIED)

    def test_a_failed_check_and_not_required_are_unaffected_by_the_action_state(self):
        aid = self.action()
        self.led.set_verification(aid, L.V_FAILED, L.LEVEL_READBACK, "hash differs")
        self.assertEqual(self.led.actions[aid]["verification"]["state"], L.V_FAILED)

    def test_a_disk_error_neither_raises_nor_leaves_memory_behind_the_truth(self):
        class BrokenFile:
            def write(self, s):
                raise OSError(28, "No space left on device")

            def flush(self):
                pass

            def close(self):
                pass

            def fileno(self):
                return 0
        self.led._fh = BrokenFile()
        aid = self.action("toolu_x")                                                  # must not raise
        self.assertEqual(self.led.actions[aid]["state"], L.A_REQUESTED)               # memory still has the row
        self.assertTrue(self.led.action_result("toolu_x", is_error=False, content="ok"))
        self.assertEqual(self.led.actions[aid]["state"], L.A_OK)
        self.assertGreaterEqual(self.led.write_failures, 1)
        self.assertEqual(self.led.last_write_error, "OSError")

    def test_after_a_disk_error_the_next_append_reopens_the_file_and_persists_again(self):
        class Broken:
            def write(self, s):
                raise OSError("full")

            def flush(self):
                pass

            def close(self):
                pass
        self.led._fh = Broken()
        self.action("toolu_1")                      # writes two records: the index record hits the error, the action record reopens the file
        self.assertEqual(self.led.write_failures, 1)
        self.action("toolu_2")                      # a healthy disk again: both of its records persist
        self.assertEqual(self.led.write_failures, 1)
        self.led.close()
        recs = [json.loads(l) for l in open(self.led.path, encoding="utf-8") if l.strip()]
        index_ids = [r["tool_use_id"] for r in recs if r.get("t") == "index"]
        action_tuids = [r.get("tool_use_id") for r in recs if r.get("t") == "action"]
        self.assertEqual(index_ids, ["toolu_2"])                       # the lost record is exactly the one that hit the error: stated, not hidden
        self.assertIn("toolu_1", action_tuids)
        self.assertIn("toolu_2", action_tuids)


class F04BashFileTests(unittest.TestCase):
    """F-04: a plain file command is checked by reading the filesystem back; anything else stays unverified."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = os.path.realpath(self.tmp.name)

    def inp(self, cmd, began, returned=None):
        return {"command": cmd, V._TIMES_KEY: (began, returned or time.time())}

    def at(self, name):
        return os.path.join(self.d, name)

    # -- the parser accepts only plain commands --------------------------------------------------------------------------------------
    def test_only_a_single_plain_file_command_on_absolute_paths_is_recognised(self):
        self.assertEqual(V.bash_file_op(f"mkdir -p {self.d}/a"), ("mkdir", ["-p"], [f"{self.d}/a"]))
        self.assertEqual(V.bash_file_op(f"cp -r {self.d}/a {self.d}/b")[0], "cp")
        for bad in ("ls /tmp", "mkdir relative/dir", "rm -rf /tmp/x; echo hi", "rm /tmp/a && rm /tmp/b", "cp /tmp/a /tmp/b | cat",
                    "touch $HOME/x", "touch ~/x", "rm /tmp/*.log", "cp /a/f > /b/g", "mkdir `whoami`", "mv /a/f", "rm", "mkdir -m 777 /tmp/q",
                    "rm --no-preserve-root /x", "git rm /a/f", "touch /a\n/b", "rm /tmp/a $(id)", "", None, "mkdir /a/b /c/d /e/f /g/h /i/j /k/l /m/n /o/p /q/r"):
            self.assertIsNone(V.bash_file_op(bad), bad)

    def test_the_planner_routes_only_those_commands_to_the_verifier(self):
        row = {"tool": "Bash", "state": L.A_OK}
        self.assertIs(V.plan(row, {"command": f"touch {self.d}/x"})[1], V.verify_bash_files)
        out, fn = V.plan(row, {"command": "make test"})
        self.assertIsNone(fn)
        self.assertEqual((out.state, out.level), (L.V_UNVERIFIED, L.LEVEL_SELF))

    # -- mkdir / touch -----------------------------------------------------------------------------------------------------------------
    def test_mkdir_that_created_the_directory_is_verified(self):
        began = time.time() - 1
        os.mkdir(self.at("new"))
        o = V.verify_bash_files(self.inp(f"mkdir {self.at('new')}", began))
        self.assertEqual((o.state, o.level), (L.V_VERIFIED, L.LEVEL_READBACK))

    def test_mkdir_p_on_a_directory_that_already_existed_is_not_proof(self):
        os.mkdir(self.at("old"))
        time.sleep(0.05)
        o = V.verify_bash_files(self.inp(f"mkdir -p {self.at('old')}", time.time()))
        self.assertEqual(o.state, L.V_UNVERIFIED)
        self.assertIn("not changed by this action", o.evidence)

    def test_a_missing_directory_after_mkdir_is_a_failure(self):
        o = V.verify_bash_files(self.inp(f"mkdir {self.at('ghost')}", time.time() - 1))
        self.assertEqual((o.state, o.level), (L.V_FAILED, L.LEVEL_READBACK))

    def test_touch_creates_or_updates_and_a_stale_file_is_not_proof(self):
        began = time.time() - 1
        open(self.at("t"), "w").close()
        self.assertEqual(V.verify_bash_files(self.inp(f"touch {self.at('t')}", began)).state, L.V_VERIFIED)
        old = self.at("old.txt")
        open(old, "w").close()
        time.sleep(0.05)
        self.assertEqual(V.verify_bash_files(self.inp(f"touch {old}", time.time())).state, L.V_UNVERIFIED)
        self.assertEqual(V.verify_bash_files(self.inp(f"touch {self.at('nope')}", time.time() - 1)).state, L.V_FAILED)

    # -- rm ------------------------------------------------------------------------------------------------------------------------------
    def test_rm_is_verified_when_the_path_is_gone_and_failed_when_it_is_not(self):
        p = self.at("victim")
        open(p, "w").close()
        self.assertEqual(V.verify_bash_files(self.inp(f"rm {p}", time.time())).state, L.V_FAILED)
        os.unlink(p)
        o = V.verify_bash_files(self.inp(f"rm {p}", time.time()))
        self.assertEqual((o.state, o.level), (L.V_VERIFIED, L.LEVEL_READBACK))
        self.assertIn("plain rm fails on a missing path", o.evidence)

    # -- mv / cp -------------------------------------------------------------------------------------------------------------------------
    def test_mv_needs_the_destination_present_and_the_source_gone(self):
        src, dst = self.at("a.txt"), self.at("b.txt")
        open(src, "w").write("x")
        began = time.time() - 1
        os.rename(src, dst)
        self.assertEqual(V.verify_bash_files(self.inp(f"mv {src} {dst}", began)).state, L.V_VERIFIED)
        open(src, "w").write("again")                                       # the source is back: the move did not leave it gone
        self.assertEqual(V.verify_bash_files(self.inp(f"mv {src} {dst}", began)).state, L.V_FAILED)

    def test_mv_into_a_directory_uses_the_basename(self):
        os.mkdir(self.at("dir"))
        src = self.at("f.txt")
        open(src, "w").write("x")
        began = time.time() - 1
        os.rename(src, self.at("dir/f.txt"))
        self.assertEqual(V.verify_bash_files(self.inp(f"mv {src} {self.at('dir')}", began)).state, L.V_VERIFIED)

    def test_cp_compares_sizes_and_a_preserved_mtime_does_not_fool_it(self):
        src, dst = self.at("s.txt"), self.at("d.txt")
        open(src, "w").write("hello world")
        os.utime(src, (1_000_000_000, 1_000_000_000))                       # an old mtime, as `cp -p` would carry over
        began = time.time() - 1
        import shutil
        shutil.copy2(src, dst)
        self.assertEqual(V.verify_bash_files(self.inp(f"cp -p {src} {dst}", began)).state, L.V_VERIFIED)
        open(dst, "w").write("short")
        o = V.verify_bash_files(self.inp(f"cp {src} {dst}", began, returned=time.time() - 0.5))
        self.assertEqual(o.state, L.V_FAILED)
        self.assertIn("differs in size", o.evidence)

    def test_a_copy_destination_that_existed_before_is_not_proof(self):
        src, dst = self.at("s.txt"), self.at("d.txt")
        open(src, "w").write("same")
        open(dst, "w").write("same")
        time.sleep(0.05)
        self.assertEqual(V.verify_bash_files(self.inp(f"cp {src} {dst}", time.time())).state, L.V_UNVERIFIED)

    def test_without_the_action_clock_nothing_is_claimed(self):
        o = V.verify_bash_files({"command": f"touch {self.at('x')}"})
        self.assertEqual(o.state, L.V_UNVERIFIED)

    def test_the_verifier_never_writes(self):
        before = sorted(os.listdir(self.d))
        V.verify_bash_files(self.inp(f"mkdir {self.at('a')}", time.time()))
        V.verify_bash_files(self.inp(f"rm -rf {self.at('b')}", time.time()))
        self.assertEqual(sorted(os.listdir(self.d)), before)


class F14ActionEpochTests(unittest.TestCase):
    """F-14: the hook that made action rows take the ledger's global epoch is gone; a row takes the epoch it is given."""

    def test_an_action_row_keeps_the_epoch_it_was_seen_in(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        led = L.Ledger(tmp.name, fsync=False)
        self.addCleanup(led.close)
        e1 = led.epoch_open("s1", 1)
        e2 = led.epoch_open("s2", 2)                                          # now the ledger's "current" epoch
        aid = led.action_seen("toolu_old", e1, "Write")
        self.assertEqual(led.actions[aid]["epoch"], e1)
        self.assertNotEqual(led.current_epoch, e1)

    def test_nothing_in_the_code_creates_action_rows_from_a_hook_any_more(self):
        import glob
        import re
        calls = []
        for f in glob.glob(os.path.join(os.path.dirname(L.__file__), "*.py")):
            calls += [(os.path.basename(f), m.group(0)) for m in re.finditer(r"\.action_seen\(", open(f).read())]
        self.assertEqual(sorted({n for n, _ in calls}), ["router.py"])             # the only caller is the router, with its own epoch


class F10Test(unittest.TestCase):
    def test_the_permission_callback_the_finding_was_about_no_longer_exists(self):
        import glob
        files = {}
        for f in glob.glob(os.path.join(os.path.dirname(L.__file__), "*.py")):
            with open(f, encoding="utf-8") as fh:
                files[os.path.basename(f)] = fh.read()
        self.assertNotIn("can_use_tool", "".join(files.values()))
        # 2026-10-22: the only tool-call hook is the short Bash deny-list (guard.py, registered in brain.py); no tier system, no callback.
        self.assertEqual(sorted(n for n, t in files.items() if "PreToolUse" in t), ["brain.py", "guard.py"])


class ReviewRound3Tests(unittest.TestCase):
    """Defects an independent reviewer confirmed in the F-fixes (2026-10-21)."""

    def test_the_same_count_stated_twice_is_still_ambiguous(self):
        from backtalk import evidence as E
        c = {"n": 21, "group": "g", "role": "primary"}
        f = lambda n: {"n": n, "group": "g", "action": "a%d" % n}
        r = E.check_claims([dict(c), dict(c)], [f(21), f(20)])
        self.assertEqual(len(r["ambiguous"]), 2)
        self.assertEqual(len(r["matched"]), 0)

    def test_a_total_explained_by_many_parts_is_not_ambiguous(self):
        from backtalk import evidence as E
        parts = list(range(1, 14)) + [20, 79 - 0]
        total = 100
        facts = [{"n": total, "group": "g", "action": "t"}] + [{"n": n, "group": "g", "action": "p%d" % n} for n in parts]
        r = E.check_claims([{"n": total, "group": "g", "role": "primary"}], facts)
        self.assertEqual(len(r["matched"]), 1)
        self.assertEqual(len(r["ambiguous"]), 0)

    def test_ledger_survives_unserialisable_record_and_never_glues_lines(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        led = L.Ledger(tmp.name, fsync=False)
        self.addCleanup(led.close)
        led._append({"t": "x", "bad": object()})                 # json default=str handles it; must not raise
        led._fh.close()                                          # a closed handle: write raises ValueError
        led._append({"t": "y"})
        self.assertEqual(led.write_failures, 1)
        led._append({"t": "z"})                                  # reopened, and starts on a fresh line
        led.close()
        with open(led.path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    json.loads(ln)

    def _bash(self, cmd, **extra):
        d = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        return d, {"command": cmd.replace("@", d), V._TIMES_KEY: (time.time() - 1, time.time()), **extra}

    def test_trailing_slash_dotdot_and_background_are_not_judged(self):
        d = tempfile.gettempdir()
        self.assertIsNone(V.bash_file_op(f"cp -R {d}/a/ {d}/b"))
        self.assertIsNone(V.bash_file_op(f"mkdir -p {d}/a/../b"))
        self.assertIsNone(V.bash_file_op(f"touch {d}/a^b"))
        _, inp = self._bash("rm @/x", run_in_background=True)
        self.assertEqual(V.verify_bash_files(inp).state, L.V_UNVERIFIED)

    def test_rm_f_of_a_path_that_never_existed_is_not_verified(self):
        _, inp = self._bash("rm -f @/never")
        self.assertEqual(V.verify_bash_files(inp).state, L.V_UNVERIFIED)
        _, inp = self._bash("rm @/never")
        self.assertEqual(V.verify_bash_files(inp).state, L.V_VERIFIED)

    def test_a_change_after_the_action_returned_is_not_credited_to_it(self):
        d = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, "dir")
        os.mkdir(p)
        now = time.time()
        inp = {"command": f"mkdir -p {p}", V._TIMES_KEY: (now - 100, now - 50)}   # returned long ago; ctime is newer
        self.assertEqual(V.verify_bash_files(inp).state, L.V_UNVERIFIED)

    def test_copying_a_directory_is_not_called_verified(self):
        d = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "s")); os.makedirs(os.path.join(d, "t", "s"))
        inp = {"command": f"cp -R {d}/s {d}/t", V._TIMES_KEY: (time.time() - 5, time.time())}
        self.assertEqual(V.verify_bash_files(inp).state, L.V_UNVERIFIED)
