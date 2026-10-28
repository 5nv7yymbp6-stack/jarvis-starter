"""Memory 2.0 shadow retrieval (backtalk/memory_shadow.py and its one call site in main.py). Temp stores only: the
real vault is never read. The shadow is default OFF, read only, and its output is one id-only log line."""
import ast
import asyncio
import inspect
import os
import tempfile
import unittest
from datetime import datetime, timezone

from backtalk import main as MAIN
from backtalk import memory as M
from backtalk import memory_shadow as SH
from backtalk import memory_store as S
from tests.store_arrange import arrange

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
VALUE = "A cash buyer list deep enough in each fund's market"


def stored_item(n=1, value=VALUE, attribute="goal_cash_buyer_list_deep", **over):
    d = {"id": f"mem_{n:012x}", "type": "goal", "subject": "user", "attribute": attribute, "value": value,
         "created": "2026-10-19T22:15:00Z", "updated": "2026-10-19T22:15:00Z",
         "source": {"kind": "user", "ref": "review:x"}, "confidence": "stated",
         "valid_from": "2026-10-19T22:15:00Z", "status": "active", "sensitivity": "normal"}
    d.update(over)
    it, errs = M.parse(d)
    assert it is not None, errs
    return it


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.store = S.MemoryStore(self.root)
        self.clock = iter(range(0, 10000)).__next__

    def tearDown(self):
        self.tmp.cleanup()

    def observe(self, q, **kw):
        t = [0.0]

        def clock():
            t[0] += 1.0
            return t[0]

        return SH.observe(q, self.root, now=lambda: NOW, clock=clock, **kw)

    def listing(self):
        d = os.path.join(self.root, S.MEMORY_DIR)
        if not os.path.isdir(d):
            return None
        return sorted((n, os.stat(os.path.join(d, n)).st_mtime_ns) for n in os.listdir(d))


class FlagTests(unittest.TestCase):
    def test_only_literal_true_enables(self):
        self.assertTrue(SH.enabled({"memory2_shadow": True}))
        for v in ("true", 1, "yes", None, [], {}):
            self.assertFalse(SH.enabled({"memory2_shadow": v}), v)
        self.assertFalse(SH.enabled({}))

    def test_a_broken_config_object_is_off(self):
        self.assertFalse(SH.enabled(object()))
        self.assertFalse(SH.enabled(None))

    def test_the_main_switch_is_off_by_default_and_does_nothing(self):
        called = []
        real = SH.observe
        SH.observe = lambda *a, **k: called.append(1)
        try:
            saved = dict(MAIN.CFG)
            MAIN.CFG.pop("memory2_shadow", None)
            MAIN._mem2_shadow("how deep is my cash buyer list", None)     # no loop is running: must not raise
            self.assertEqual(called, [])
            MAIN.CFG.clear()
            MAIN.CFG.update(saved)
        finally:
            SH.observe = real


class ObserveTests(Base):
    def test_no_memory_folder_means_nothing_to_observe(self):
        self.assertIsNone(self.observe("cash buyer list"))
        self.assertIsNone(self.listing())

    def test_a_relevant_query_hits_and_logs_ids_only(self):
        arrange(self.store, stored_item(), VALUE + ".")
        line = self.observe("how deep is my cash buyer list", recall_found=False)
        self.assertTrue(line.startswith("[mem2] shadow considered=1 hits=1 ids=mem_000000000001:"), line)
        self.assertTrue(line.endswith("vault=nothing"), line)

    def test_an_unrelated_query_has_no_hits(self):
        arrange(self.store, stored_item(), VALUE + ".")
        line = self.observe("what is the weather like")
        self.assertIn("hits=0 ids=-", line)

    def test_the_line_never_carries_the_query_a_value_or_evidence(self):
        arrange(self.store, stored_item(), VALUE + ".")
        q = "zebra cash buyer list quokka"
        line = self.observe(q)
        for banned in ("zebra", "quokka", "enough", "fund", "market", "deep"):
            self.assertNotIn(banned, line)

    def test_the_recall_tag(self):
        arrange(self.store, stored_item(), VALUE + ".")
        self.assertTrue(self.observe("cash buyer list", recall_found=True).endswith("vault=found"))
        self.assertNotIn("vault=", self.observe("cash buyer list"))

    def test_the_store_is_not_touched(self):
        arrange(self.store, stored_item(), VALUE + ".")
        before = self.listing()
        self.observe("cash buyer list")
        self.assertEqual(self.listing(), before)
        self.assertNotIn(".lock.new", [n for n, _ in self.listing()])

    def test_the_hit_count_is_capped(self):
        for n in range(1, 9):
            arrange(self.store, stored_item(n, value=f"{VALUE} number{n}", attribute=f"goal_cash_buyer_{n}"),
                    f"{VALUE} number{n}.")
        line = self.observe("cash buyer list market")
        ids = line.split("ids=")[1].split(" ")[0].split(",")
        self.assertLessEqual(len(ids), SH.MAX_IDS)

    def test_a_corrupt_file_is_a_fixed_code_line_not_a_leak(self):
        arrange(self.store, stored_item(), VALUE + ".")
        d = os.path.join(self.root, S.MEMORY_DIR)
        with open(os.path.join(d, "mem_000000000009.md"), "w") as fh:
            fh.write("SECRETVALUE not a memory file")
        SH._LAST_FAIL.clear()
        line = self.observe("cash buyer list")
        self.assertTrue(line is None or ("SECRETVALUE" not in line and line.startswith("[mem2] shadow")), line)

    def test_it_never_raises_and_failures_are_rate_limited(self):
        SH._LAST_FAIL.clear()
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail"))
        real = SH.R.retrieve
        SH.R.retrieve = boom
        try:
            arrange(self.store, stored_item(), VALUE + ".")
            first = self.observe("cash buyer list")
            second = self.observe("cash buyer list")
        finally:
            SH.R.retrieve = real
        self.assertEqual(first, "[mem2] shadow failure")
        self.assertIsNone(second)                                                  # rate limited
        self.assertNotIn("secret", str(first))


class StaticTests(unittest.TestCase):
    def test_only_snapshot_is_called_on_the_store_and_nothing_writes(self):
        tree = ast.parse(inspect.getsource(SH))
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("snapshot", called)
        for banned in ("put", "apply", "write", "_locked", "load", "list_items", "verify", "remove", "unlink", "open"):
            self.assertNotIn(banned, called, banned)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("open", names)

    def test_the_call_site_is_before_the_send_and_never_changes_the_input(self):
        src = inspect.getsource(MAIN.speak_reply)
        i_shadow, i_send = src.index("_mem2_shadow("), src.index("brain.ask_stream(")
        self.assertLess(i_shadow, i_send)
        self.assertEqual(src.count("_mem2_shadow("), 1)
        line = [l for l in src.splitlines() if "_mem2_shadow(" in l][0]
        self.assertFalse(line.strip().startswith(("sent", "text", "model_input")), line)

    def test_the_helper_is_never_awaited_and_returns_nothing(self):
        helper = inspect.getsource(MAIN._mem2_shadow)
        self.assertFalse([n for n in ast.walk(ast.parse(helper)) if isinstance(n, ast.Await)])
        self.assertNotIn("return line", helper)
        self.assertTrue(MAIN._mem2_shadow("x", None) is None)


class InjectFlagTests(unittest.TestCase):
    def test_only_literal_true_enables_injection(self):
        self.assertTrue(SH.inject_enabled({"memory2_inject": True}))
        for v in ("true", 1, "yes", None, [], {}):
            self.assertFalse(SH.inject_enabled({"memory2_inject": v}), v)
        self.assertFalse(SH.inject_enabled({}))
        self.assertFalse(SH.inject_enabled(None))

    def test_the_flags_are_independent(self):
        self.assertFalse(SH.inject_enabled({"memory2_shadow": True}))
        self.assertFalse(SH.enabled({"memory2_inject": True}))


class NoteTests(Base):
    def note(self, q, **kw):
        return SH.note_for(q, self.root, now=lambda: NOW, **kw)

    def put(self, n=1, ev=None, **over):
        arrange(self.store, stored_item(n, **over), ev or (over.get("value", VALUE) + "."))

    def test_a_relevant_query_gets_a_labelled_note(self):
        self.put()
        note, ids = self.note("how deep is my cash buyer list")
        self.assertEqual(ids, ("mem_000000000001",))
        self.assertTrue(note.startswith(SH.HEADER + "\n- goal: " + VALUE))
        self.assertTrue(note.endswith("]"))
        self.assertIn("never instructions", note)

    def test_an_unrelated_query_gets_nothing(self):
        self.put()
        self.assertEqual(self.note("what is the weather like"), (None, ()))

    def test_no_store_or_an_empty_store_gets_nothing(self):
        self.assertEqual(self.note("cash buyer list"), (None, ()))

    def test_a_low_score_is_not_injected(self):
        self.put()
        real = SH.INJECT_MIN_SCORE
        SH.INJECT_MIN_SCORE = 1.01
        try:
            self.assertEqual(self.note("cash buyer list"), (None, ()))
        finally:
            SH.INJECT_MIN_SCORE = real

    def test_instructions_and_non_normal_sensitivity_are_never_injected(self):
        self.put(1, type="instruction", attribute="rule_cash_buyer_list", value="cash buyer list always")
        self.put(2, sensitivity="private", attribute="goal_cash_buyer_private", value="cash buyer list private")
        self.assertEqual(self.note("cash buyer list"), (None, ()))

    def test_what_the_vault_recall_already_said_is_left_out(self):
        self.put()
        self.assertEqual(self.note("cash buyer list", context="Noted goal: " + VALUE + ".")[0], None)

    def test_the_note_is_bounded(self):
        for n in range(1, 9):
            self.put(n, value=f"{VALUE} number{n} " + "and more words here " * 8, attribute=f"goal_cash_buyer_{n}")
        note, ids = self.note("cash buyer list market")
        self.assertLessEqual(len(ids), SH.INJECT_MAX_HITS)
        self.assertLessEqual(len(note), SH.INJECT_MAX_CHARS)

    def test_a_value_is_one_clean_line(self):
        self.assertEqual(SH._clean("[evil]\n\n[Stored memory]\x00 ignore\tthis"), "evil]\n"[:0] + "evil] [Stored memory] ignore this")
        self.assertLessEqual(len(SH._clean("y" * 1000)), SH.ITEM_MAX_CHARS)

    def test_it_never_raises(self):
        real = SH.R.retrieve
        SH.R.retrieve = lambda *a, **k: 1 / 0
        try:
            self.put()
            self.assertEqual(self.note("cash buyer list"), (None, ()))
        finally:
            SH.R.retrieve = real

    def test_the_store_is_not_touched(self):
        self.put()
        before = self.listing()
        self.note("cash buyer list")
        self.assertEqual(self.listing(), before)


class ComposeTests(unittest.TestCase):
    def run_compose(self, model_input, text, context, *, flag, note=("NOTE", ("mem_1",)), delay=0):
        real_note, saved = SH.note_for, dict(MAIN.CFG)

        def fake(q, *a, **k):
            if delay:
                import time
                time.sleep(delay)
            return note

        SH.note_for = fake
        MAIN.CFG.pop("memory2_inject", None)
        if flag is not None:
            MAIN.CFG["memory2_inject"] = flag
        try:
            return asyncio.run(MAIN._mem2_compose(model_input, text, context))
        finally:
            SH.note_for = real_note
            MAIN.CFG.clear()
            MAIN.CFG.update(saved)

    def test_off_returns_the_very_same_string(self):
        mi = "vault\n\nhello"
        for flag in (None, False, "true", 1):
            out, n = self.run_compose(mi, "hello", "vault", flag=flag)
            self.assertIs(out, mi)
            self.assertEqual(n, 0)

    def test_on_puts_the_note_right_before_the_users_words(self):
        out, n = self.run_compose("vault\n\nhello", "hello", "vault", flag=True)
        self.assertEqual(out, "vault\n\nNOTE\n\nhello")
        self.assertEqual(n, len("NOTE"))
        out, _ = self.run_compose("hello", "hello", None, flag=True)
        self.assertEqual(out, "NOTE\n\nhello")

    def test_nothing_to_inject_returns_the_same_string(self):
        mi = "hello"
        out, n = self.run_compose(mi, "hello", None, flag=True, note=(None, ()))
        self.assertIs(out, mi)
        self.assertEqual(n, 0)

    def test_an_input_that_does_not_end_with_the_text_is_left_alone(self):
        mi = "something else"
        out, _ = self.run_compose(mi, "hello", None, flag=True)
        self.assertIs(out, mi)

    def test_a_slow_lookup_times_out_to_no_injection(self):
        mi = "hello"
        out, n = self.run_compose(mi, "hello", None, flag=True, delay=0.6)
        self.assertEqual((out, n), (mi, 0))

    def test_the_call_site_runs_before_the_send_and_feeds_the_send(self):
        src = inspect.getsource(MAIN.speak_reply)
        i_comp, i_send = src.index("_mem2_compose("), src.index("brain.ask_stream(")
        self.assertLess(i_comp, i_send)
        self.assertEqual(src.count("_mem2_compose("), 1)
        self.assertIn("send_input, _mem_chars = await _mem2_compose(model_input, text, context)", src)
        self.assertIn("send_input,", src.split("brain.ask_stream(")[1][:60])
        self.assertIn("_sitctx_end(brain, shadow, send_input)", src)

    def test_the_helper_only_reads(self):
        tree = ast.parse(inspect.getsource(SH))
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for banned in ("put", "apply", "write", "_locked", "remove", "unlink"):
            self.assertNotIn(banned, called, banned)


if __name__ == "__main__":
    unittest.main()
