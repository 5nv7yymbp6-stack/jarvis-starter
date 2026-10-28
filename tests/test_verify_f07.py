"""F-07 (Tier A): what a file or board read-back may claim, using only the action's own clock (when it was decided,
when it returned), the file's mtime and the board snapshot's `ts`. Direct calls give the verifier explicit times, so
every case is deterministic; a few go through the real queue to prove the row's own times are what reach it."""
import http.server
import json
import os
import tempfile
import threading
import time
import unittest

from backtalk import ledger as L
from backtalk import verify as V

BEGAN, RETURNED = 1_000_000_000.0, 1_000_000_010.0
TIMES = {V._TIMES_KEY: (BEGAN, RETURNED)}
VERIFIED, FAILED, UNVERIFIED = L.V_VERIFIED, L.V_FAILED, L.V_UNVERIFIED


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, text, mtime):
        p = os.path.join(self.tmp.name, name)
        with open(p, "w", newline="") as f:
            f.write(text)
        os.utime(p, (mtime, mtime))
        return p

    def write(self, path, content, times=TIMES):
        return V.verify_write({"file_path": path, "content": content, **times})

    def edit(self, path, old=None, new=None, times=TIMES, **kw):
        return V.verify_edit({"file_path": path, "old_string": old, "new_string": new, **kw, **times})

    def multi(self, path, edits, times=TIMES):
        return V.verify_edit({"file_path": path, "edits": edits, **times})

    # ---- Write ----
    def test_1_exact_write_is_still_verified(self):
        out = self.write(self.file("a", "hello\nworld\n", BEGAN + 3), "hello\nworld\n")
        self.assertEqual((out.state, out.level), (VERIFIED, L.LEVEL_READBACK))

    def test_a_write_whose_content_was_already_there_still_counts_as_the_requested_end_state(self):
        self.assertEqual(self.write(self.file("a", "same", BEGAN - 500), "same").state, VERIFIED)   # not causal proof, but true

    def test_2_a_mismatch_in_a_file_modified_after_the_write_returned_is_unverified_not_failed(self):
        out = self.write(self.file("a", "changed later", RETURNED + 100), "what was written")
        self.assertEqual((out.state, out.level), (UNVERIFIED, L.LEVEL_NONE))
        self.assertIn("modified after", out.evidence)

    def test_a_mismatch_in_a_file_not_modified_since_the_write_returned_is_still_a_failure(self):
        for mtime in (BEGAN - 500, BEGAN + 5, RETURNED, RETURNED + 0.004):               # not later (5 ms of slack)
            self.assertEqual(self.write(self.file("a", "stale", mtime), "new").state, FAILED, mtime)

    def test_without_the_actions_times_a_write_mismatch_is_still_a_failure(self):
        self.assertEqual(self.write(self.file("a", "x", RETURNED + 99), "y", times={}).state, FAILED)

    def test_10_the_legitimate_write_failures_remain_failures(self):
        self.assertEqual(self.write(os.path.join(self.tmp.name, "none"), "x").state, FAILED)                  # missing
        self.assertEqual(self.write(self.file("w", "other", BEGAN + 1), "data").state, FAILED)                # wrong content
        self.assertEqual(self.write(self.file("p", "NEW con", BEGAN + 1), "NEW content").state, FAILED)       # partial
        self.assertEqual(self.write(self.file("e", "", BEGAN + 1), "something").state, FAILED)                # empty, non-empty expected
        self.assertEqual(self.write(self.file("ok", "", BEGAN + 1), "").state, VERIFIED)                      # empty, empty expected
        a, b = self.file("a2", "other", BEGAN + 1), self.file("b2", "expected", BEGAN + 1)
        self.assertEqual(self.write(a, "expected").state, FAILED)                                             # text sits in b, not a
        self.assertEqual(V.verify_write({"file_path": a, **TIMES}).state, UNVERIFIED)                         # no content at all

    def test_9_a_relative_path_is_unverified_and_is_never_resolved_against_a_folder(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            open("rel.txt", "w").write("x")
            self.assertEqual(self.write("rel.txt", "x").state, UNVERIFIED)                                   # it exists, right here
            self.assertEqual(self.edit("rel.txt", "a", "x").state, UNVERIFIED)
            self.assertIn("not absolute", self.write("rel.txt", "x").evidence)
        finally:
            os.chdir(cwd)

    def test_a_home_relative_path_is_expanded_not_rejected_as_relative(self):
        self.assertEqual(self.write("~/this-file-does-not-exist-f07", "x").state, FAILED)

    # ---- Edit ----
    def test_5_an_edit_that_changed_the_file_during_the_action_is_verified(self):
        for mtime in (BEGAN, BEGAN + 3, RETURNED + 100):
            out = self.edit(self.file("e", "alpha NEW omega", mtime), "OLD", "NEW")
            self.assertEqual((out.state, out.level), (VERIFIED, L.LEVEL_READBACK), mtime)

    def test_3_matching_text_in_a_file_not_touched_since_the_action_began_is_unverified(self):
        out = self.edit(self.file("e", "NEW was already here", BEGAN - 100), "OLD", "NEW")
        self.assertEqual(out.state, UNVERIFIED)
        self.assertIn("not modified since", out.evidence)
        self.assertEqual(self.edit(self.file("e2", "alpha NEW omega", BEGAN - 0.1), "OLD", "NEW").state, UNVERIFIED)

    def test_4_a_tiny_new_text_that_was_already_there_is_unverified(self):
        self.assertEqual(self.edit(self.file("e", "the cat sat", BEGAN - 100), "dog", "a").state, UNVERIFIED)

    def test_an_edit_with_no_start_time_cannot_be_shown_to_have_changed_the_file(self):
        self.assertEqual(self.edit(self.file("e", "NEW", BEGAN + 1), "OLD", "NEW", times={V._TIMES_KEY: (None, RETURNED)}).state, UNVERIFIED)
        self.assertEqual(self.edit(self.file("e2", "NEW", BEGAN + 1), "OLD", "NEW", times={}).state, UNVERIFIED)

    def test_the_content_checks_are_not_weakened(self):
        self.assertEqual(self.edit(self.file("a", "alpha OLD omega", BEGAN + 1), "OLD", "NEW").state, FAILED)   # did not land
        self.assertEqual(self.edit(self.file("b", "nothing relevant", BEGAN + 1), "OLD", "NEW").state, FAILED)  # neither text
        self.assertEqual(self.edit(self.file("c", "has NEW and OLD", BEGAN + 1), "OLD", "NEW").state, FAILED)   # old still there
        self.assertEqual(self.edit(os.path.join(self.tmp.name, "none"), "a", "b").state, FAILED)               # missing file
        self.assertEqual(self.edit(self.file("d", "x", BEGAN + 1), "a", None).state, UNVERIFIED)               # no new text

    def test_an_edit_mismatch_in_a_file_modified_after_it_returned_is_unverified_not_failed(self):
        out = self.edit(self.file("a", "alpha OLD omega", RETURNED + 100), "OLD", "NEW")
        self.assertEqual(out.state, UNVERIFIED)
        self.assertIn("modified after", out.evidence)

    # ---- MultiEdit and replace_all ----
    def test_6_a_dependent_multiedit_whose_edits_both_landed_is_verified(self):
        edits = [{"old_string": "a", "new_string": "b"}, {"old_string": "b", "new_string": "c"}]
        self.assertEqual(self.multi(self.file("m", "c", BEGAN + 1), edits).state, VERIFIED)

    def test_a_dependent_multiedit_whose_second_edit_did_not_land_fails(self):
        edits = [{"old_string": "a", "new_string": "b"}, {"old_string": "b", "new_string": "c"}]
        self.assertEqual(self.multi(self.file("m", "b", BEGAN + 1), edits).state, FAILED)      # only the first landed
        self.assertEqual(self.multi(self.file("m2", "z", BEGAN + 1), edits).state, FAILED)     # neither

    def test_a_later_edit_that_changes_part_of_an_earlier_new_text_is_followed(self):
        edits = [{"old_string": "x", "new_string": "foo bar"}, {"old_string": "bar", "new_string": "baz"}]
        self.assertEqual(self.multi(self.file("m", "foo baz", BEGAN + 1), edits).state, VERIFIED)
        self.assertEqual(self.multi(self.file("m2", "foo bar", BEGAN + 1), edits).state, FAILED)

    def test_a_later_edit_that_puts_an_earlier_old_text_back_is_not_a_failure(self):
        edits = [{"old_string": "a", "new_string": "x"}, {"old_string": "y", "new_string": "a"}]
        self.assertEqual(self.multi(self.file("m", "x a", BEGAN + 1), edits).state, VERIFIED)

    def test_the_first_edits_new_text_missing_is_not_excused_just_because_a_later_edit_exists(self):
        edits = [{"old_string": "a", "new_string": "unrelated first"}, {"old_string": "q", "new_string": "r"}]
        self.assertEqual(self.multi(self.file("m", "r only", BEGAN + 1), edits).state, FAILED)

    def test_7_replace_all_with_an_old_occurrence_left_is_a_failure(self):
        out = self.edit(self.file("r", "x NEW y OLD z", BEGAN + 1), "OLD", "NEW", replace_all=True)
        self.assertEqual(out.state, FAILED)
        self.assertEqual(self.edit(self.file("r2", "x NEW y NEW z", BEGAN + 1), "OLD", "NEW", replace_all=True).state, VERIFIED)

    def test_8_replace_all_where_the_old_text_is_part_of_the_new_text_keeps_the_existing_rule(self):
        out = self.edit(self.file("r", "foo bar x foo bar", BEGAN + 1), "foo", "foo bar", replace_all=True)
        self.assertEqual(out.state, VERIFIED)                                 # `old` inside `new` is exempt, as before
        self.assertEqual(self.edit(self.file("r2", "x NEW y OLD z", BEGAN + 1), "OLD", "NEW").state, FAILED)  # and without replace_all

    # ---- through the real queue: the row's own times are what the verifier gets ----
    def test_the_queue_gives_the_verifier_the_actions_own_times_and_a_model_cannot_supply_its_own(self):
        import asyncio

        async def go():
            led = L.Ledger(os.path.join(self.tmp.name, "led"), fsync=False)
            ep = led.epoch_open("s", 1)
            q = V.VerifyQueue(led, timeout_s=2.0)
            p = os.path.join(self.tmp.name, "q.txt")
            open(p, "w").write("alpha NEW omega")
            time.sleep(0.02)
            led.action_seen("t1", ep, "Edit")
            led.action_result("t1", is_error=False, content="x")
            aid = led.action_id_for("t1")
            began = led.actions[aid]["history"][0][0]
            os.utime(p, (began - 100, began - 100))                           # the file predates the decision
            spoof = {"file_path": p, "old_string": "OLD", "new_string": "NEW", V._TIMES_KEY: (0.0, 0.0)}
            await q.submit(led.actions[aid], spoof)
            self.assertEqual(led.actions[aid]["verification"]["state"], UNVERIFIED)   # the spoofed (0, 0) did not count
            self.assertIn("not modified since", led.actions[aid]["verification"]["evidence"])
            os.utime(p, (began + 1, began + 1))
            await q.submit(led.actions[aid], {"file_path": p, "old_string": "OLD", "new_string": "NEW"})
            self.assertEqual(led.actions[aid]["verification"]["state"], VERIFIED)
            led.close()
        asyncio.run(go())

    def test_the_time_the_call_was_first_seen_is_the_start(self):
        row = {"history": [[BEGAN + 7, "REQUESTED", ""]], "returned_at": RETURNED}
        self.assertEqual(V._times(row), (BEGAN + 7, RETURNED))
        self.assertEqual(V._times({}), (None, None))


class Board(unittest.TestCase):
    def setUp(self):
        outer = self
        self.state = {"items": []}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(outer.state).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.shutdown)
        self.port = self.srv.server_address[1]
        self.returned = time.time() - 1                       # the action returned a second ago

    def snap(self, titles, ts=None, **kw):
        self.state = {"event": "snapshot", "items": [{"kind": "note", "title": t, "scale": 1.0} for t in titles],
                      "ts": time.time() if ts is None else ts, **kw}

    def run_board(self, body, wait=0.3, returned=None, command=None):
        cmd = command or "barehands/bin/board.sh '" + json.dumps(body) + "'"
        r = self.returned if returned is None else returned
        return V.verify_board({"command": cmd, V._TIMES_KEY: (r - 5, r)}, port=self.port, wait_s=wait)

    def add(self, title, **kw):
        return self.run_board({"a": "add_card", "title": title}, **kw)

    def test_12_a_fresh_snapshot_with_the_exact_title_is_verified(self):
        self.snap(["TRICERATOPS", "OTHER"])
        out = self.add("TRICERATOPS")
        self.assertEqual((out.state, out.level), (VERIFIED, L.LEVEL_READBACK))
        self.assertNotIn("spotlight", out.evidence)

    def test_present_keeps_its_existence_check_and_says_the_spotlight_was_not_checked(self):
        self.snap(["TRICERATOPS"])
        out = self.run_board({"a": "present", "title": "TRICERATOPS"})
        self.assertEqual(out.state, VERIFIED)
        self.assertIn("spotlight not checked", out.evidence)
        self.snap(["OTHER"])
        self.assertEqual(self.run_board({"a": "present", "title": "TRICERATOPS"}).state, FAILED)

    def test_13_a_fresh_snapshot_without_the_title_is_a_failure(self):
        self.snap(["OTHER"])
        out = self.add("TRICERATOPS", wait=0.5)
        self.assertEqual((out.state, out.level), (FAILED, L.LEVEL_READBACK))

    def test_14_a_stale_snapshot_that_has_the_title_is_unverified(self):
        self.snap(["TRICERATOPS"], ts=self.returned - 3600)
        self.assertEqual(self.add("TRICERATOPS").state, UNVERIFIED)
        self.snap(["TRICERATOPS"], ts=self.returned)                                   # strictly newer is required
        self.assertEqual(self.add("TRICERATOPS").state, UNVERIFIED)

    def test_15_a_stale_snapshot_without_the_title_is_unverified_never_failed(self):
        self.snap([], ts=self.returned - 3600)
        out = self.add("TRICERATOPS", wait=0.5)
        self.assertEqual((out.state, out.level), (UNVERIFIED, L.LEVEL_NONE))
        self.assertIn("newer than this action", out.evidence)

    def test_16_a_snapshot_with_no_usable_timestamp_is_unverified(self):
        for ts in (None, "now", True, [1]):
            self.snap(["TRICERATOPS"])
            self.state["ts"] = ts
            self.assertEqual(self.add("TRICERATOPS").state, UNVERIFIED, repr(ts))
        self.snap(["TRICERATOPS"])
        del self.state["ts"]
        self.assertEqual(self.add("TRICERATOPS").state, UNVERIFIED)

    def test_an_action_with_no_return_time_cannot_be_judged_against_any_snapshot(self):
        self.snap(["TRICERATOPS"])
        out = V.verify_board({"command": "barehands/bin/board.sh '" + json.dumps({"a": "add_card", "title": "TRICERATOPS"}) + "'"},
                             port=self.port, wait_s=0.2)
        self.assertEqual(out.state, UNVERIFIED)

    def test_a_snapshot_that_becomes_fresh_while_waiting_is_used(self):
        self.snap(["TRICERATOPS"], ts=self.returned - 10)
        threading.Timer(0.5, lambda: self.snap(["TRICERATOPS"])).start()
        self.assertEqual(self.add("TRICERATOPS", wait=3.0).state, VERIFIED)

    def test_a_card_that_appears_in_a_later_fresh_snapshot_is_found(self):
        self.snap(["OTHER"])
        threading.Timer(0.5, lambda: self.snap(["OTHER", "SLOW"])).start()
        self.assertEqual(self.add("SLOW", wait=3.0).state, VERIFIED)

    def test_11_a_long_title_matches_the_title_the_board_actually_stored(self):
        long = "L" * 100
        self.snap(["L" * 80])
        self.assertEqual(self.add(long).state, VERIFIED)
        self.snap([long])                                                              # the board never stores 100
        self.assertEqual(self.add(long, wait=0.5).state, FAILED)
        self.snap(["L" * 79])
        self.assertEqual(self.add(long, wait=0.5).state, FAILED)                       # 79 is not the 80-char title

    def test_the_truncation_counts_utf16_units_like_the_boards_javascript(self):
        title = "x" * 79 + "\U0001F600" + "tail"                                      # the emoji straddles the 80-unit cut
        self.assertEqual(V._board_title(title), "x" * 79 + "\ud83d")                   # JS: a lone high surrogate
        self.assertEqual(V._board_title("é" * 100), "é" * 80)
        self.snap(["x" * 79 + "\ud83d"])
        self.assertEqual(self.add(title).state, VERIFIED)

    def test_17_more_than_one_board_command_in_a_call_is_unverified_even_if_both_cards_exist(self):
        self.snap(["FIRST", "SECOND"])
        one = "barehands/bin/board.sh '" + json.dumps({"a": "add_card", "title": "FIRST"}) + "'"
        two = one + " && barehands/bin/board.sh '" + json.dumps({"a": "add_card", "title": "SECOND"}) + "'"
        self.assertEqual(self.run_board(None, command=one).state, VERIFIED)
        out = self.run_board(None, command=two)
        self.assertEqual((out.state, out.level), (UNVERIFIED, L.LEVEL_NONE))
        self.assertIn("more than one", out.evidence)

    def test_18_a_title_that_only_contains_or_is_contained_in_the_requested_one_is_not_a_match(self):
        self.snap(["TRICERATOPS 2", "TRICERAT"])
        self.assertEqual(self.add("TRICERATOPS", wait=0.5).state, FAILED)

    def test_19_the_wrong_item_does_not_verify(self):
        self.snap(["SOMETHING ELSE", "TRICERATOPS"], )
        self.assertEqual(self.add("SOMETHING", wait=0.5).state, FAILED)

    def test_20_duplicate_cards_of_one_title_prove_no_more_than_existence_does(self):
        self.snap(["A", "A", "A"])
        out = self.add("A")
        self.assertEqual(out.state, VERIFIED)                       # Tier A cannot tell a pre-existing card from a new one
        self.assertNotIn("3", out.evidence)                         # and it does not pretend to have counted them

    def test_unsupported_board_actions_and_a_dark_board_stay_unverified(self):
        out = self.run_board({"a": "lock", "title": "X"})
        self.assertEqual((out.state, out.level), (UNVERIFIED, L.LEVEL_SELF))
        self.srv.shutdown()
        self.assertEqual(self.add("X", wait=0.3).state, UNVERIFIED)

    def test_the_queue_puts_the_rows_return_time_into_the_boards_input(self):
        import asyncio

        async def go():
            tmp = tempfile.mkdtemp()
            led = L.Ledger(tmp, fsync=False)
            ep = led.epoch_open("s", 1)
            q = V.VerifyQueue(led, timeout_s=3.0)
            led.action_seen("b1", ep, "Bash")
            led.action_result("b1", is_error=False, content="204")
            aid = led.action_id_for("b1")
            cmd = {"command": "barehands/bin/board.sh '" + json.dumps({"a": "add_card", "title": "Q"}) + "'"}
            with unittest.mock.patch.dict(os.environ, {"HOLO_PORT": str(self.port)}):
                self.snap(["Q"], ts=led.actions[aid]["returned_at"] - 50)
                await q.submit(led.actions[aid], cmd)
                self.assertEqual(led.actions[aid]["verification"]["state"], UNVERIFIED)         # older than the action
                self.snap(["Q"], ts=led.actions[aid]["returned_at"] + 1)
                await q.submit(led.actions[aid], cmd)
                self.assertEqual(led.actions[aid]["verification"]["state"], VERIFIED)
            led.close()
        asyncio.run(go())


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
