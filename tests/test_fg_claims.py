"""G-03 option B (2026-10-25): counts from compound commands whose output layout can be proven, and the LOG-ONLY check of the
numbers a foreground turn spoke against its own tool counts."""
import tempfile
import unittest

from backtalk import evidence as E
from backtalk import ledger as L
from backtalk import verify as V


def ev(cmd, out):
    return E.extract_evidence("Bash", {"command": cmd}, out)


class CompoundCounts(unittest.TestCase):
    def test_the_real_turn_from_2026_10_25(self):
        cmd = ('cd /Users/yousif/Desktop/jarvis/jarvis && find . -name "*.md" -not -path "*/.obsidian/*" | wc -l\n'
               'echo "---newest---"\n'
               'find . -name "*.md" -not -path "*/.obsidian/*" -print0 | xargs -0 stat -f "%m %N" | sort -rn | head -5')
        e = ev(cmd, "     175\n---newest---\n1792953899 ./a.md\n1792922604 ./b.md\n")
        self.assertEqual(e["kind"], E.KIND_COUNTS)
        self.assertEqual(E.facts_of(e), [(175, "item")])
        self.assertIsNone(E.fact_of(e))                    # the background verifier's input is unchanged

    def test_labelled_counts_in_sequence(self):
        cmd = 'echo "notes:"; find . -name "*.md" | wc -l; echo "folders:"; find . -type d | wc -l'
        e = ev(cmd, "notes:\n42\nfolders:\n7\n")
        self.assertEqual(E.facts_of(e), [(42, "item"), (7, "dir")])

    def test_a_shifted_layout_drops_the_counts_instead_of_misreading_them(self):
        cmd = 'echo "notes:"; find . -name "*.md" | wc -l; echo "folders:"; find . -type d | wc -l'
        e = ev(cmd, "find: ./x: Permission denied\nnotes:\n42\nfolders:\n7\n")   # stderr pushed everything down a line
        self.assertEqual(E.facts_of(e), [])
        e = ev(cmd, "notes:\n42\nfind: ./y: Permission denied\nfolders:\n7\n")
        self.assertEqual(E.facts_of(e), [(42, "item")])     # the first count is still provable; the second is not

    def test_two_unlabelled_counts_only_the_first_is_provable(self):
        e = ev("find . | wc -l; find . -type d | wc -l", "40\n6\n")
        self.assertEqual(E.facts_of(e), [(40, "item")])

    def test_unknowable_layouts_give_no_counts(self):
        for cmd, out in [
            ("ls; find . | wc -l", "a\nb\n12\n"),                         # ls prints an unknown number of lines first
            ("echo -n x; find . | wc -l", "x12\n"),                       # echo flag
            ('echo "$HOME"; find . | wc -l', "/Users/y\n12\n"),            # expansion
            ("echo *; find . | wc -l", "a b\n12\n"),                      # glob
            ('echo "n:"; find . | wc -l || echo none', "n:\n12\n"),        # || in a compound: what ran is unknowable
            ("x=$(find . | wc -l); echo $x", "12\n"),                     # substitution
            ("echo 20; echo 21", "20\n21\n"),                             # literals measure nothing
            ("cat notes.txt", "175\n"),                                   # a number read from a file is not a count
        ]:
            self.assertEqual(E.facts_of(ev(cmd, out)), [], cmd)

    def test_single_counts_keep_their_old_kind(self):
        e = ev("find . -name '*.md' | wc -l", "  21\n")
        self.assertEqual((e["kind"], e["int"]), (E.KIND_INTEGER, 21))
        self.assertEqual(E.facts_of(e), [(21, "item")])


class Numbers(unittest.TestCase):
    def test_numbers_only_and_nothing_that_identifies_a_person(self):
        nums, partial = E.numbers_in("54 open, call 555-123-4567 or (555) 123 4567, $1,950, id 12345678, on 2026-10-21")
        self.assertEqual(nums, [10, 21, 54, 1950, 2026])
        self.assertFalse(partial)
        self.assertNotIn(4567, nums)
        self.assertNotIn(12345678, nums)

    def test_too_many_numbers_is_marked_partial(self):
        nums, partial = E.numbers_in(" ".join(str(i) for i in range(1000, 1400)))
        self.assertEqual(len(nums), E._NUMS_MAX)
        self.assertTrue(partial)

    def test_every_result_records_its_numbers_never_its_text(self):
        e = E.extract_evidence("Read", {"file_path": "/x/Dashboard.md"}, "You have 54 open priorities and 3 done.\n")
        self.assertEqual(e["nums"], [3, 54])
        self.assertNotIn("priorities", repr(e))

    def test_a_truncated_result_is_partial(self):
        e = E.extract_evidence("Bash", {"command": "cat big"}, "7\n" + "x" * 30000)
        self.assertTrue(e["nums_partial"])


class Shadow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.led = L.Ledger(self.tmp.name, fsync=False)
        self.addCleanup(self.led.close)
        self.n = 0

    def action(self, turn, evidence, ok=True):
        self.n += 1
        aid = f"A{self.n}"
        self.led.actions[aid] = {"id": aid, "turn_id": turn, "state": L.A_OK if ok else L.A_ERROR, "evidence": evidence}
        return aid

    def check(self, turn, said, asked=""):
        return V.shadow_check_foreground(self.led, turn, said, asked)

    def notes(self):
        return [x for x in self.led.notes if x.get("note") == "fg_claim_shadow"]

    def test_a_counted_number_is_counted(self):
        self.action("T1", ev("find . -name '*.md' | wc -l", "175\n"))
        r = self.check("T1", "175 notes total.")
        self.assertEqual((r["verdict"], r["counted"]), ("accounted", [175]))
        self.assertEqual(len(self.notes()), 1)

    def test_a_wrong_count_is_contradicted(self):
        self.action("T1", ev("find . -name '*.md' | wc -l", "175\n"))
        r = self.check("T1", "157 notes total.")
        self.assertEqual((r["verdict"], r["contradicted"], r["counted_by_tools"]), ("CONTRADICTED", [157], [175]))

    def test_a_number_read_from_a_note_is_in_output(self):
        self.action("T1", ev("cat Dashboard.md", "54 open priorities\n"))
        r = self.check("T1", "You have 54 open priorities.")
        self.assertEqual((r["verdict"], r["in_output"]), ("accounted", [54]))

    def test_a_number_the_tools_never_showed_is_nowhere(self):
        self.action("T1", ev("cat Dashboard.md", "54 open priorities\n"))
        r = self.check("T1", "You have 45 open priorities.")
        self.assertEqual((r["verdict"], r["nowhere"]), ("NOWHERE", [45]))

    def test_a_number_from_his_own_request_is_not_blamed_on_the_tools(self):
        self.action("T1", ev("ls -t notes | head -5", "a.md\nb.md\n"))
        r = self.check("T1", "Here are the five newest.", "summarize the five newest notes")
        self.assertEqual((r["verdict"], r["in_request"]), ("accounted", [5]))

    def test_absence_from_a_partial_output_proves_nothing(self):
        self.action("T1", ev("cat big", "7\n" + "x" * 30000))
        r = self.check("T1", "There are 12 of them.")
        self.assertEqual((r["verdict"], r["unknown"]), ("unknown", [12]))

    def test_a_turn_without_tools_or_without_numbers(self):
        r = self.check("T9", "It's about 3 miles.")
        self.assertEqual((r["verdict"], r["nowhere"]), ("no_tools", [3]))
        self.action("T2", ev("find . | wc -l", "4\n"))
        self.assertIsNone(self.check("T2", "Done, boss."))
        self.assertEqual(len(self.notes()), 1)

    def test_only_this_turns_successful_actions_count(self):
        self.action("T1", ev("find . | wc -l", "9\n"))
        self.action("T2", ev("find . | wc -l", "175\n"))
        self.action("T2", ev("find . | wc -l", "999\n"), ok=False)
        r = self.check("T2", "175 items, not 999 or 9.")
        self.assertEqual(r["counted"], [175])
        self.assertEqual(sorted(r["nowhere"]), [9, 999])

    def test_spoken_times_are_clock_readings_not_claims(self):
        self.action("T1", ev("date", "Sun Oct 25 14:00:00 CDT 2026\n"))
        for said in ("Just about eleven at night.", "Five fifty PM here, boss.", "It's half past six.", "Three o'clock sharp."):
            self.assertIsNone(self.check("T1", said), said)

    def test_the_real_22_vs_23_from_2026_10_05(self):
        cmd = "find /x/01\\ -\\ Daily\\ Notes -type f -regex '.*/[0-9]\\{4\\}-[0-9]\\{2\\}-[0-9]\\{2\\}\\.md'"
        out = "".join(f"/x/01 - Daily Notes/2026-09-{d:02d}.md\n" for d in range(1, 24))     # 23 files
        self.action("T1", E.extract_evidence("Bash", {"command": cmd}, out))
        r = self.check("T1", "Found 22 daily note files, the most recent being 2026-10-05.")
        self.assertEqual((r["verdict"], r["contradicted"], r["counted_by_tools"]), ("CONTRADICTED", [22], [23]))

    def test_it_never_raises(self):
        class Broken:
            actions = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertIsNone(V.shadow_check_foreground(Broken(), "T1", "5 files"))

    def test_the_log_carries_numbers_never_the_spoken_text(self):
        from unittest import mock
        self.action("T1", ev("find . | wc -l", "175\n"))
        lines = []
        with mock.patch.object(V, "log", side_effect=lines.append):
            self.check("T1", "Your secret project has 157 files.", "how big is my secret project")
        self.assertEqual(len(lines), 1)
        self.assertNotIn("secret", lines[0])
        self.assertIn("CONTRADICTED", lines[0])


class Wiring(unittest.TestCase):
    def test_ask_stream_runs_the_check_after_the_last_sentence_and_only_for_spoken_turns(self):
        import os
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtalk", "brain.py"), encoding="utf-8") as f:
            src = f.read()
        i = src.index("async def ask_stream")
        body = src[i:src.index("def _mentions_background")]
        self.assertIn("if said and not silent:\n            shadow_check_foreground(self.ledger, h.turn_id, \" \".join(said), utterance)", body)
        self.assertGreater(body.index("shadow_check_foreground(self.ledger"), body.rindex("yield out.pop(0)"))
        self.assertEqual(body.count("said.append(out[0])"), 2)


if __name__ == "__main__":
    unittest.main()
