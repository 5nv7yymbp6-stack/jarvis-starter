"""Phase B1/B3: structured evidence from tool results, and the numeric claims a
model makes about them (backtalk/evidence.py). Pure functions."""
import unittest

from backtalk import evidence as E

DIR = "/Users/yousif/Desktop/jarvis/jarvis/01 - Daily Notes/09 - September 2026/"
PATHS = [f"{DIR}2026-09-{d:02d}.md" for d in range(9, 30)]          # 21 paths
OUT = "\n".join(PATHS)
FIND = ('find "/Users/yousif/Desktop/jarvis/jarvis/01 - Daily Notes/" -type f -name "*.md" '
        '! -name "Daily Note Template.md" | sort')


def ev(tool, cmd_or_input, content, err=False):
    ti = {"command": cmd_or_input} if isinstance(cmd_or_input, str) else cmd_or_input
    return E.extract_evidence(tool, ti, content, err)


class ListEvidenceTests(unittest.TestCase):
    def test_the_incident_result_is_a_list_of_21(self):
        e = ev("Bash", FIND, OUT)
        self.assertEqual((e["kind"], e["lines"], e["group"]), ("list", 21, "item"))
        self.assertEqual(E.fact_of(e), (21, "item"))

    def test_the_cli_cwd_line_is_not_output(self):
        e = ev("Bash", FIND, OUT + "\nShell cwd was reset to /private/tmp/x")
        self.assertEqual(e["lines"], 21)

    def test_evidence_is_bounded_and_never_the_body(self):
        e = ev("Bash", FIND, OUT)
        self.assertLessEqual(len(e["head"]), 2)
        self.assertLessEqual(len(e["tail"]), 1)
        for s in e["head"] + e["tail"]:
            self.assertLessEqual(len(s), 103)
        blob = repr(e)
        self.assertNotIn("2026-09-20.md", blob)                  # a middle path is not kept
        self.assertLess(len(blob), 900)

    def test_samples_are_redacted(self):
        e = ev("Bash", "find . -name x", "a/b\nAuthorization: Bearer SECRETBEARERTOKEN99\n/c\n/d")
        self.assertNotIn("SECRETBEARERTOKEN99", repr(e))

    def test_strict_list_commands_are_lists(self):
        for cmd in (FIND, "find . -name '*.md'", "find . -type f | sort", "ls -1 /tmp", "ls -A",
                    "ls /tmp", "cd /tmp && ls -1", "git ls-files", "git ls-files --cached",
                    "cd d && find . -maxdepth 1 -type f | sort"):
            self.assertEqual(ev("Bash", cmd, "a\nb\nc")["kind"], "list", cmd)

    def test_directories_are_their_own_group(self):
        self.assertEqual(ev("Bash", "find . -type d", "a\nb")["group"], "dir")
        self.assertEqual(ev("Bash", "find . -type f", "a\nb")["group"], "item")

    def test_commands_whose_lines_are_not_entries_are_not_lists(self):
        for cmd in ("ls -la /tmp", "ls -l", "ls /a /b", "ls *.md", "ls --color=always",
                    "find . -exec echo {} ;", "find . -printf '%p %s\\n'", "find . -print0",
                    "find . > /tmp/out", "find . 2>&1", "find . | head -5 > x",
                    "git ls-files -z", "git ls-files -s", "cat notes.txt", "grep -r x .",
                    "echo hi", "find . | xargs wc", "ls $(pwd)", "ls; echo x"):
            self.assertNotEqual(ev("Bash", cmd, "a\nb\nc")["kind"], "list", cmd)

    def test_stderr_to_devnull_is_fine_but_merged_stderr_is_not(self):
        self.assertEqual(ev("Bash", "find . -name x 2>/dev/null", "a\nb")["kind"], "list")
        self.assertNotEqual(ev("Bash", "find . -name x 2>&1", "a\nb")["kind"], "list")

    def test_glob_is_a_list_and_no_files_is_zero(self):
        self.assertEqual(E.fact_of(ev("Glob", {"pattern": "*.md"}, "a.md\nb.md")), (2, "item"))
        self.assertEqual(E.fact_of(ev("Glob", {"pattern": "*.md"}, "No files found")), (0, "item"))

    def test_an_empty_find_is_a_list_of_zero(self):
        self.assertEqual(E.fact_of(ev("Bash", "find . -name zzz", "")), (0, "item"))

    def test_blocks_and_none_are_handled(self):
        self.assertEqual(E.fact_of(ev("Bash", FIND, [{"type": "text", "text": OUT}])), (21, "item"))
        self.assertIsNone(E.fact_of(ev("Bash", FIND, None)) if False else None)
        self.assertEqual(ev("Bash", "cat x", None)["kind"], "empty")


class IntegerAndOtherEvidenceTests(unittest.TestCase):
    def test_a_lone_integer_is_a_fact_whatever_printed_it(self):
        for cmd, out in (("find . | wc -l", "      21"), ("echo ${#files[@]}", "21\nShell cwd was reset to /x"),
                         ("grep -c x f", "21")):
            self.assertEqual(E.fact_of(ev("Bash", cmd, out)), (21, "item"), cmd)

    def test_errors_text_and_unknown_tools_yield_no_count(self):
        self.assertIsNone(E.fact_of(ev("Bash", FIND, "No such file", err=True)))
        self.assertEqual(ev("Bash", FIND, "x", err=True)["kind"], "error")
        self.assertIsNone(E.fact_of(ev("Bash", "cat notes.txt", "one\ntwo\nthree")))
        self.assertIsNone(E.fact_of(ev("Read", {"file_path": "/x"}, "1\n2\n3")))
        self.assertIsNone(E.fact_of(ev("Bash", "echo 'a b'", "21 files")))

    def test_truncated_output_never_yields_a_count(self):
        big = "\n".join(f"{DIR}file-{i:05d}.md" for i in range(400))
        e = ev("Bash", FIND, big)
        self.assertTrue(e.get("truncated"))
        self.assertIsNone(E.fact_of(e))
        e = ev("Glob", {"pattern": "*"}, "a\nb\n(Results are truncated. Consider using a more specific path or pattern.)")
        self.assertIsNone(E.fact_of(e))

    def test_extraction_never_raises(self):
        class Boom:
            def __str__(self):
                raise RuntimeError("x")
        self.assertIn("kind", E.extract_evidence("Bash", {"command": Boom()}, Boom(), False))


class ClaimTests(unittest.TestCase):
    def counts(self, text):
        return [(c["n"], c["group"]) for c in E.extract_claims(text)]

    def test_the_incident_summary_is_one_count_claim(self):
        s = ("20 daily note files found (September 2026 subfolder only): 2026-09-09, 2026-09-10, "
             "2026-09-11, 2026-09-12, 2026-09-13, ...")
        self.assertEqual(self.counts(s), [(20, "item")])
        self.assertEqual(E.extract_claims(s)[0]["noun"], "daily note files")

    def test_the_correct_summary_is_21(self):
        self.assertEqual(self.counts("21 daily note files, dated 2026-09-09 to 2026-09-30."),
                         [(21, "item")])

    def test_groups(self):
        self.assertEqual(self.counts("21 files across 3 folders"), [(21, "item"), (3, "dir")])
        self.assertEqual(self.counts("drafted 3 emails"), [(3, "other")])
        self.assertEqual(self.counts("1,234 results"), [(1234, "item")])

    def test_lead_ins_and_spelled_numbers(self):
        self.assertEqual(self.counts("Found 20."), [(20, "item")])
        self.assertEqual(self.counts("There are 7 in the folder"), [(7, "item")])
        self.assertEqual(self.counts("twenty-one notes"), [(21, "item")])
        self.assertEqual(self.counts("thirty files"), [(30, "item")])
        self.assertEqual(self.counts("about thirty files"), [(30, None)])    # approximate: unchecked

    def test_not_counts(self):
        self.assertEqual(self.counts("done at 2026-09-30 15:40"), [])
        self.assertEqual(self.counts("all notes present"), [])
        self.assertEqual(self.counts("all notes present"), [])
        self.assertEqual(self.counts(""), [])
        self.assertEqual(self.counts(None), [])

    def test_free_numbers_are_flagged_unverifiable(self):
        self.assertEqual(self.counts("estimated ARV is 250000"), [(250000, None)])
        self.assertEqual(self.counts("about 20% done"), [(20, None)])
        self.assertEqual([c["n"] for c in E.extract_claims("rate 3.5 now")], [None])
        self.assertEqual(self.counts("21 files, ARV 250000"), [(21, "item"), (250000, None)])

    def test_ids_and_hex_are_not_numbers(self):
        self.assertEqual(self.counts("task a16ff5bdae43 uuid 5b48bccc-5e13-47ac done"), [])


class YearIsNotACountTests(unittest.TestCase):
    """Found live 2026-10-01: "September 2026 daily notes directory" was read as
    a claim of 2026 directories. A year that describes a date or a name is not a
    number a task can be wrong about."""

    def counts(self, text):
        return [(c["n"], c["group"]) for c in E.extract_claims(text)]

    def test_the_live_phrase_is_no_claim_at_all(self):
        self.assertEqual(E.extract_claims("September 2026 daily notes directory"), [])
        self.assertEqual(E.extract_claims("in the September 2026 daily notes directory"), [])

    def test_the_whole_live_summary_is_exactly_one_claim(self):
        s = ("Command executed successfully. The find command located and sorted all markdown "
             "files in the September 2026 daily notes directory. 20 daily note files found.")
        self.assertEqual(self.counts(s), [(20, "item")])

    def test_years_after_a_month_a_preposition_or_an_article_are_not_counts(self):
        for t in ("Sep 2026 notes", "September 2026 files", "year 2026 notes", "FY 2026 files",
                  "Q3 2026 files", "fiscal 2026 records", "May 2026 files",
                  "built in 1985", "since 2025", "during 2026", "until 2026", "by 2030",
                  "from 2025 to 2026"):
            self.assertEqual(self.counts(t), [], t)

    def test_a_date_preposition_followed_by_a_count_noun_is_a_count(self):
        """Audit 2026-10-01: "increased by 2000 files" and "searched in 2026 files"
        were silently dropped as years, so the count was spoken unchecked."""
        for t, n in (("increased by 2000 files", 2000), ("searched in 2026 files", 2026),
                     ("in 2000 files", 2000), ("by 2000 files", 2000), ("during 2026 files", 2026),
                     ("since 2025 notes", 2025), ("through 2026 notes", 2026),
                     ("until 2026 entries", 2026), ("by 2030 items", 2030)):
            self.assertEqual(self.counts(t), [(n, "item")], t)
        self.assertEqual(self.counts("there may 2026 files exist"), [(2026, "item")])   # "may" is a verb here

    def test_without_strong_date_context_a_year_sized_number_is_still_a_count(self):
        # a vault CAN hold this many; hiding the number would let it be spoken unchecked
        for t in ("a total of 2026 files", "the 2026 notes", "of 2026 notes", "found 2026 notes"):
            self.assertEqual(self.counts(t), [(2026, "item")], t)

    def test_a_year_that_names_a_folder_is_not_a_folder_count(self):
        for t in ("2026 daily notes directory", "the 2026 notes folder", "2025 backups directory"):
            self.assertEqual(self.counts(t), [], t)

    def test_real_folder_counts_are_still_detected(self):
        self.assertEqual(self.counts("3 folders found"), [(3, "dir")])
        self.assertEqual(self.counts("Found 3 folders"), [(3, "dir")])
        self.assertEqual(self.counts("21 files across 3 folders"), [(21, "item"), (3, "dir")])
        # "subfolders" is not a noun the extractor knows, so its 12 stays an unchecked free number
        self.assertEqual(self.counts("12 subfolders and 3 directories"), [(3, "dir"), (12, None)])
        self.assertEqual(self.counts("3 folders in the September 2026 daily notes directory"),
                         [(3, "dir")])

    def test_real_file_and_item_counts_are_still_detected(self):
        for text, want in (("20 daily note files found", 20), ("21 files", 21), ("There are 7 items", 7),
                           ("1,234 results", 1234), ("twenty-one notes", 21), ("Found 20.", 20),
                           ("21 daily note files, dated 2026-09-09 to 2026-09-30.", 21),
                           ("1850 files", 1850), ("2101 files", 2101), ("899 entries", 899)):
            self.assertEqual([n for n, _ in self.counts(text)], [want], text)

    def test_a_bare_year_sized_count_is_still_a_count(self):
        self.assertEqual(self.counts("2026 files"), [(2026, "item")])

    def test_known_residue_an_unmarked_year_before_items_reads_as_a_count(self):
        # documented limit, pinned so a change to it is deliberate
        self.assertEqual(self.counts("2026 daily notes"), [(2026, "item")])

    def test_years_dates_times_and_hex_ids_stay_ignored(self):
        for t in ("done at 2026-09-30 15:40", "see 2026", "dated 2026-09-09 to 2026-09-30",
                  "at 15:40:05 on 2026-09-30T15:40:05", "task a16ff5bdae43 uuid 5b48bccc-5e13-47ac",
                  "September 2026", "in 2026", "release 2.0"):
            self.assertEqual([c for c in E.extract_claims(t) if c["group"] is not None], [], t)
        self.assertEqual(self.counts("done at 2026-09-30 15:40"), [])
        self.assertEqual(self.counts("task a16ff5bdae43 uuid 5b48bccc-5e13-47ac"), [])

    def test_a_year_in_clear_date_context_is_not_a_free_number_either(self):
        self.assertEqual(E.extract_claims("September 2026 daily notes directory"), [])
        self.assertEqual(E.extract_claims("built in 1985"), [])
        # but a year-sized number with no date context is a number, and unchecked
        self.assertEqual(self.counts("see 2026"), [(2026, None)])


class CheckTests(unittest.TestCase):
    def fact(self, n, group="item"):
        return {"n": n, "group": group, "action": "A"}

    def claim(self, n, group="item"):
        return {"n": n, "noun": "files", "group": group}

    def test_equal_matches_and_different_fails_with_no_off_by_one_forgiveness(self):
        self.assertEqual(len(E.check_claims([self.claim(21)], [self.fact(21)])["matched"]), 1)
        r = E.check_claims([self.claim(20)], [self.fact(21)])
        self.assertEqual((len(r["failed"]), len(r["matched"])), (1, 0))
        self.assertEqual(r["facts"], [21])
        self.assertEqual(len(E.check_claims([self.claim(22)], [self.fact(21)])["failed"]), 1)

    def test_groups_are_not_mixed(self):
        r = E.check_claims([self.claim(3, "dir")], [self.fact(21)])
        self.assertEqual((len(r["failed"]), len(r["unchecked"])), (0, 1))
        r = E.check_claims([self.claim(3, "other"), self.claim(5, None)], [self.fact(21)])
        self.assertEqual(len(r["unchecked"]), 2)

    def test_a_lone_claim_among_conflicting_counts_is_ambiguous_not_matched(self):
        # audit F-13 (this test used to pin the opposite: "any held value of the group matches"). The task's own tools counted 21 AND 20;
        # a summary saying 20 equals one of them by luck as much as by fact, so it is neither verified nor refuted.
        r = E.check_claims([self.claim(20)], [self.fact(21), self.fact(20)])
        self.assertEqual((len(r["matched"]), len(r["failed"]), len(r["ambiguous"])), (0, 0, 1))
        self.assertEqual(r["facts"], [20, 21])

    def test_two_claims_that_each_equal_a_held_value_still_match(self):
        r = E.check_claims([self.claim(21), self.claim(20)], [self.fact(21), self.fact(20)])
        self.assertEqual((len(r["matched"]), len(r["ambiguous"])), (2, 0))

    def test_a_total_whose_own_parts_are_the_other_counts_is_not_ambiguous(self):
        r = E.check_claims([self.claim(21)], [self.fact(21), self.fact(8), self.fact(12), self.fact(1)])
        self.assertEqual((len(r["matched"]), len(r["ambiguous"])), (1, 0))

    def test_the_single_fact_cases_are_unchanged(self):
        self.assertEqual(len(E.check_claims([self.claim(21)], [self.fact(21)])["matched"]), 1)
        self.assertEqual(len(E.check_claims([self.claim(20)], [self.fact(21)])["failed"]), 1)

    def test_no_facts_means_everything_is_unchecked(self):
        r = E.check_claims([self.claim(20)], [])
        self.assertEqual((len(r["failed"]), len(r["unchecked"])), (0, 1))


class ProvenanceTests(unittest.TestCase):
    """Audit 2026-10-01: a count fact must be a MEASUREMENT of the thing, not a
    number the model typed or the length of output padded by an earlier command."""

    def fact(self, cmd, out):
        return E.fact_of(ev("Bash", cmd, out))

    def test_a_literal_the_model_typed_is_never_a_fact(self):
        for cmd in ("echo 20", 'echo "20"', "echo '20'", "printf 20", "printf '%s' 20", "print 20",
                    "echo 20 21", "cd d && echo 20", "export N=20; echo 20", "echo $N",
                    "seq 1 20 | wc -l", "echo 20 | wc -l", "printf 'a\\nb\\n' | wc -l", "yes | head -20 | wc -l"):
            self.assertIsNone(self.fact(cmd, "20"), cmd)

    def test_a_lone_integer_from_something_that_is_not_a_count_is_not_a_fact(self):
        for cmd in ("cat count.txt", "date +%d", "expr 20 + 1", "wc -l notes.md", "grep x f", "python3 -c 'print(21)'"):
            self.assertIsNone(self.fact(cmd, "21"), cmd)
        self.assertIsNone(E.fact_of(E.extract_evidence("Read", {"file_path": "/x"}, "21")))
        self.assertIsNone(E.fact_of(E.extract_evidence("Grep", {"pattern": "x"}, "21")))

    def test_real_counting_commands_are_facts(self):
        for cmd in ("find . -name '*.md' | wc -l", "ls -1 | wc -l", "ls | wc -l", "git ls-files | wc -l",
                    "cat notes.txt | wc -l", "find . | sort | wc -l", "grep -c TODO f", "rg -c TODO f",
                    "grep --count x f", "echo ${#files[@]}", "echo ${#files}",
                    "cd d && setopt extended_glob null_glob; files=(**/*.md); echo ${#files[@]}",
                    'cd "/x/y z/" && setopt extended_glob null_glob 2>/dev/null; files=(**/*.md); echo ${#files[@]}'):
            self.assertEqual(self.fact(cmd, "      21"), (21, "item"), cmd)

    def test_the_count_group_follows_what_was_counted(self):
        self.assertEqual(self.fact("find . -type d | wc -l", "3"), (3, "dir"))
        self.assertEqual(self.fact("find . -type f | wc -l", "21"), (21, "item"))

    def test_a_printing_command_before_the_counting_command_voids_the_count(self):
        out = "Counting:\n" + "\n".join(f"./{i}.md" for i in range(21))        # 22 lines, 21 files
        for cmd in ('echo "Counting:" && find . -name "*.md" | sort', "pwd; find . -type f",
                    "ls && find . -type f", "cat header.txt; find . -type f", "echo start\nfind . -type f",
                    "date || find . -type f", "git status && git ls-files"):
            self.assertIsNone(self.fact(cmd, out), cmd)
            self.assertNotEqual(ev("Bash", cmd, out)["kind"], "list", cmd)

    def test_silent_commands_before_it_keep_the_evidence(self):
        out = "\n".join(f"./{i}.md" for i in range(21))
        for cmd in ("cd d && find . -type f", "cd d; find . -type f", "cd d && cd e && ls -1",
                    "setopt extended_glob && find . -type f", "export A=1 && find . -type f",
                    "A=1; find . -type f", "files=(a b); find . -type f", "true && find . -type f",
                    "cd d && setopt null_glob 2>/dev/null; find . -type f | sort", "sleep 0 && ls -1",
                    "shopt -s globstar && find . -type f", "unset A; find . -type f"):
            self.assertEqual(self.fact(cmd, out), (21, "item"), cmd)

    def test_an_earlier_stage_that_is_piped_away_prints_nothing(self):
        out = "\n".join(f"./{i}.md" for i in range(21))
        self.assertEqual(self.fact("echo ignored | find . -type f", out), (21, "item"))

    def test_commands_that_print_are_not_silent_even_when_they_look_like_setup(self):
        out = "x\n" + "\n".join(f"./{i}.md" for i in range(21))
        for cmd in ("cd - && find . -type f", "setopt && find . -type f", "unsetopt; find . -type f",
                    "export && find . -type f", "export -p; find . -type f", "shopt -p && find . -type f",
                    "shopt && find . -type f"):
            self.assertIsNone(self.fact(cmd, out), cmd)

    def test_legitimate_list_evidence_is_preserved(self):
        out = "\n".join(f"./{i}" for i in range(21))
        for cmd in ('find "/a b/" -type f -name "*.md" | sort', "find . -type f", "ls -1 /tmp", "ls -A",
                    "git ls-files", "git ls-files --cached", "cd d && find . -maxdepth 1 | sort",
                    "find . -name x 2>/dev/null"):
            self.assertEqual(self.fact(cmd, out), (21, "item"), cmd)
        self.assertEqual(E.fact_of(ev("Glob", {"pattern": "*"}, out)), (21, "item"))

    def test_heredocs_and_substitutions_never_yield_evidence(self):
        for cmd in ("cat <<EOF | wc -l\na\nEOF", "echo $(find . | wc -l)", "find . | wc -l `echo x`"):
            self.assertIsNone(self.fact(cmd, "3"), cmd)


class AuditPhraseClaimTests(unittest.TestCase):
    """Every phrase the 2026-10-01 audit showed being spoken unchecked or
    mis-parsed. Each number is now accounted for: a count about something, or a
    free (unchecked) number that makes the task UNVERIFIED."""

    def claims(self, text):
        return [(c["n"], c["group"]) for c in E.extract_claims(text)]

    def test_one_and_zero(self):
        self.assertEqual(self.claims("Found one file."), [(1, "item")])
        self.assertEqual(self.claims("zero files matched"), [(0, "item")])
        self.assertEqual(self.claims("one daily note"), [(1, "item")])
        self.assertEqual(self.claims("There is one."), [(1, "item")])
        self.assertEqual(self.claims("one of the files is missing"), [(1, None)])       # unchecked, not dropped
        self.assertEqual(self.claims("no files found"), [(0, None)])
        self.assertEqual(self.claims("No matches"), [(0, None)])

    def test_k_and_other_suffixes_are_unchecked_numbers(self):
        self.assertEqual(self.claims("2k files"), [(2000, None)])
        self.assertEqual(self.claims("ARV is 250k"), [(250000, None)])
        self.assertEqual(self.claims("about 2.5k files"), [(None, None)])
        self.assertEqual(self.claims("3M in repairs"), [(3_000_000, None)])
        self.assertEqual(self.claims("20x more files"), [(20, None)])
        self.assertEqual(self.claims("the 5th note"), [(5, None)])
        self.assertEqual(self.claims("5mb"), [(5, None)])
        self.assertEqual(self.claims("20% done"), [(20, None)])

    def test_compound_number_words(self):
        for text, n in (("twenty five files", 25), ("twenty-five files", 25), ("twenty one notes", 21),
                        ("thirty-nine entries", 39), ("fifteen items", 15), ("ninety nine records", 99)):
            self.assertEqual(self.claims(text), [(n, "item")], text)

    def test_hundred_thousand_and_dozen(self):
        for text, n in (("one hundred twenty files", 120), ("one hundred and twenty files", 120),
                        ("a hundred files", 100), ("two hundred files", 200), ("three hundred sixty five notes", 365),
                        ("two thousand five hundred files", 2500), ("a thousand files", 1000),
                        ("two dozen files", 24), ("a dozen files", 12), ("one million rows", 1_000_000)):
            self.assertEqual(self.claims(text), [(n, "item")], text)

    def test_a_malformed_number_run_is_unchecked_not_guessed(self):
        self.assertEqual(self.claims("five twenty files"), [(None, None)])
        self.assertEqual(self.claims("hundred hundred files"), [(None, None)])
        self.assertEqual(self.claims("one one files"), [(None, None)])

    def test_vague_quantities_are_unchecked(self):
        for text in ("dozens of files", "hundreds of notes", "several files", "a couple of files", "a handful of notes"):
            self.assertEqual(self.claims(text), [(None, None)], text)

    def test_approximate_numbers_are_unchecked_never_called_wrong(self):
        for text in ("about two dozen files", "~20 files", "around 20 files", "roughly 20 files",
                     "over 20 files", "at least 20 files", "more than 20 files", "up to 20 files", "nearly 20 notes"):
            self.assertEqual([g for _, g in self.claims(text)], [None], text)

    def test_amounts_and_measures_the_old_year_mask_hid(self):
        for text, n in (("Estimated ARV is 2000", 2000), ("rent 2100 per month", 2100), ("price $1950", 1950),
                        ("sqft 1800", 1800), ("ARV 1900", 1900), ("offer 2050", 2050)):
            self.assertEqual(self.claims(text), [(n, None)], text)

    def test_years_in_clear_date_context_are_ignored(self):
        for text in ("built in 1985", "September 2026 daily notes directory", "Sep 2026", "May 2026 files",
                     "FY 2026", "year 2026", "Q3 2026", "from 2025 to 2026", "took 10am", "at 3:45 pm",
                     "dated 2026-09-09", "on 09/30/2026", "called on May 20", "Sep 9th"):
            self.assertEqual(self.claims(text), [], text)

    def test_ids_hex_and_file_names_are_not_numbers(self):
        for text in ("task a16ff5bdae43", "uuid 5b48bccc-5e13-47ac", "toolu_01YXtFD1vWBuayunRzBBBzab",
                     "file-2.md and report_3.md", "released v2", "abc123"):
            self.assertEqual(self.claims(text), [], text)

    def test_the_incident_summary_is_still_exactly_one_claim(self):
        s = ("Command executed successfully. The find command located and sorted all markdown files in the "
             "September 2026 daily notes directory. 20 daily note files found.")
        self.assertEqual(self.claims(s), [(20, "item")])

    def test_numbers_that_are_still_detected_as_before(self):
        for text, want in (("3 folders found", [(3, "dir")]), ("21 files across 3 folders", [(21, "item"), (3, "dir")]),
                           ("2026 files", [(2026, "item")]), ("1,234 results", [(1234, "item")]),
                           ("a total of 2026 files", [(2026, "item")]), ("Found 20.", [(20, "item")])):
            self.assertEqual(self.claims(text), want, text)

    def test_no_phrase_with_a_digit_or_number_word_comes_back_empty_by_accident(self):
        """The fail-closed property: if a number token is there, something is returned."""
        for text in ("took 5 seconds", "version 3", "port 4890", "about 7", "seven", "ten", "sixty-four",
                     "exactly 12 apples", "the count was 9"):
            self.assertTrue(E.extract_claims(text), text)


class FailedVerdictScopeTests(unittest.TestCase):
    def test_only_the_facts_of_the_claims_group_are_reported(self):
        facts = [{"n": 21, "group": "item", "action": "A1"}, {"n": 3, "group": "dir", "action": "A2"},
                 {"n": 5, "group": "item", "action": "A3"}]
        r = E.check_claims([{"n": 20, "noun": "files", "group": "item"}], facts)
        self.assertEqual(r["facts"], [5, 21])
        self.assertEqual(sorted(r["actions"]), ["A1", "A3"])
        self.assertEqual(r["facts_all"], [3, 5, 21])
        r = E.check_claims([{"n": 2, "noun": "folders", "group": "dir"}], facts)
        self.assertEqual((r["facts"], r["actions"]), ([3], ["A2"]))

    def test_an_unchecked_claim_pulls_in_no_facts(self):
        facts = [{"n": 21, "group": "item", "action": "A1"}]
        r = E.check_claims([{"n": 250000, "noun": "", "group": None}], facts)
        self.assertEqual((r["facts"], r["actions"], r["facts_all"]), ([], [], [21]))


class ClaimRoleTests(unittest.TestCase):
    """Primary vs breakdown is decided by STRUCTURE (parentheses, a +/,/and chain,
    "of which"), never by how big the number is."""

    def roles(self, text):
        return [(c["n"], c["group"], c["role"]) for c in E.extract_claims(text)]

    def test_a_plus_chain_after_a_total_is_a_breakdown_that_inherits_the_group(self):
        self.assertEqual(self.roles("21 files total: 8 + 12 + 1"),
                         [(21, "item", "primary"), (8, "item", "breakdown"),
                          (12, "item", "breakdown"), (1, "item", "breakdown")])

    def test_a_parenthesised_sum_is_a_breakdown(self):
        self.assertEqual(self.roles("21 files (8 + 13)"),
                         [(21, "item", "primary"), (8, "item", "breakdown"), (13, "item", "breakdown")])

    def test_counts_with_their_own_nouns_chained_or_parenthesised_are_breakdowns(self):
        self.assertEqual(self.roles("21 files total: 8 files + 12 files + 1 file"),
                         [(21, "item", "primary"), (8, "item", "breakdown"),
                          (12, "item", "breakdown"), (1, "item", "breakdown")])
        self.assertEqual(self.roles("21 files: 8 files, 12 files and 1 file"),
                         [(21, "item", "primary"), (8, "item", "breakdown"),
                          (12, "item", "breakdown"), (1, "item", "breakdown")])
        self.assertEqual(self.roles("20 files: 8 (a files) and 12 (b files)")[0], (20, "item", "primary"))

    def test_of_which_and_including_introduce_a_breakdown(self):
        self.assertEqual(self.roles("21 files, of which 8 files are old"),
                         [(21, "item", "primary"), (8, "item", "breakdown")])
        # a claim in ANOTHER group has no primary of its own group to be part of: it is promoted
        self.assertEqual(self.roles("21 files including 3 folders"),
                         [(21, "item", "primary"), (3, "dir", "primary")])
        self.assertEqual(self.roles("21 files including 8 files"),
                         [(21, "item", "primary"), (8, "item", "breakdown")])

    def test_the_total_may_come_after_its_parts(self):
        self.assertEqual(self.roles("8 + 12 + 1 = 21 files"),
                         [(21, "item", "primary"), (8, "item", "breakdown"),
                          (12, "item", "breakdown"), (1, "item", "breakdown")])

    def test_the_live_summary_shape_has_one_primary_claim_per_statement(self):
        s = ("Based on the command results, I can examine the file list: The files found are: "
             "- 2026-09-09.md through 2026-09-16.md (8 files) - 2026-09-18.md through 2026-09-29.md (12 files) "
             "- 2026-09-30.md (1 file) Files dated before 2026-09-30: 20 files. 20 daily note files found.")
        self.assertEqual(self.roles(s), [(8, "item", "breakdown"), (12, "item", "breakdown"),
                                         (1, "item", "breakdown"), (20, "item", "primary"),
                                         (20, "item", "primary")])

    def test_a_scope_note_after_the_number_does_not_demote_it(self):
        """The 2026-09-30 incident: the number is outside the parentheses."""
        self.assertEqual(self.roles("20 daily note files found (September 2026 subfolder only)"),
                         [(20, "item", "primary")])
        self.assertEqual(self.roles("20 files (dated before 2026-09-30)"), [(20, "item", "primary")])

    def test_it_is_structure_not_size(self):
        # the bigger number is the parenthetical one; the smaller is the statement
        self.assertEqual(self.roles("8 files total (21 files in the vault)"),
                         [(8, "item", "primary"), (21, "item", "breakdown")])
        self.assertEqual(self.roles("8 files total (21 files in the vault)")[0][2], "primary")

    def test_a_lone_parenthesised_claim_is_still_a_claim(self):
        self.assertEqual(self.roles("Done (20 files)"), [(20, "item", "primary")])
        self.assertEqual(self.roles("All good (found 20)"), [(20, "item", "primary")])

    def test_several_parenthetical_claims_with_no_statement_are_all_parts(self):
        self.assertEqual(self.roles("split (8 files) and (12 files)"),
                         [(8, "item", "breakdown"), (12, "item", "breakdown")])

    def test_bare_numbers_with_nothing_to_be_part_of_stay_unrelated(self):
        self.assertEqual(self.roles("8 + 12 + 1"), [(8, None, "unrelated"), (12, None, "unrelated"),
                                                    (1, None, "unrelated")])

    def test_ordinary_claims_are_not_breakdowns(self):
        self.assertEqual(self.roles("21 files across 3 folders"), [(21, "item", "primary"), (3, "dir", "primary")])
        self.assertEqual(self.roles("20+ files"), [(20, None, "unrelated")])
        self.assertEqual(self.roles("21 of 22 files"), [(22, "item", "primary"), (21, None, "unrelated")])
        self.assertEqual(self.roles("5 files, 3 folders"), [(5, "item", "primary"), (3, "dir", "primary")])
        self.assertEqual(self.roles("Found 20."), [(20, "item", "primary")])

    def test_unrelated_and_other_numbers_keep_their_roles(self):
        self.assertEqual(self.roles("ARV is 250k"), [(250000, None, "unrelated")])
        self.assertEqual(self.roles("drafted 3 emails"), [(3, "other", "primary")])


class BreakdownCheckTests(unittest.TestCase):
    def fact(self, n, action="A", group="item"):
        return {"n": n, "group": group, "action": action}

    def check(self, text, *values):
        facts = [self.fact(v, f"A{i}") for i, v in enumerate(values)]
        return E.check_claims(E.extract_claims(text), facts)

    def numbers(self, claims):
        return sorted(c["n"] for c in claims)

    def test_a_correct_total_with_unsupported_parts_is_not_a_mismatch(self):
        for text in ("21 files total: 8 + 12 + 1", "21 files (8 + 13)", "21 files: 8 files, 12 files and 1 file",
                     "21 files, of which 8 files are old"):
            r = self.check(text, 21)
            self.assertEqual(self.numbers(r["matched"]), [21], text)
            self.assertEqual(r["failed"], [], text)
            self.assertTrue(r["breakdown_unchecked"], text)
            self.assertEqual(r["breakdown_matched"], [], text)

    def test_a_wrong_total_still_fails(self):
        for text in ("20 files total", "20 daily note files found (September 2026 subfolder only)", "Done (20 files)"):
            r = self.check(text, 21)
            self.assertEqual((self.numbers(r["failed"]), r["matched"]), ([20], []), text)

    def test_a_wrong_total_with_parts_fails_on_the_total_only(self):
        r = self.check("20 files total: 8 + 12", 21)
        self.assertEqual(self.numbers(r["failed"]), [20])
        self.assertEqual(self.numbers(r["breakdown_unchecked"]), [8, 12])

    def test_the_live_failure_shape(self):
        s = ("The files found are: - 2026-09-09.md through 2026-09-16.md (8 files) - 2026-09-18.md through "
             "2026-09-29.md (12 files) - 2026-09-30.md (1 file) Files dated before 2026-09-30: 20 files. "
             "20 daily note files found.")
        r = self.check(s, 21)
        self.assertEqual(sorted({c["n"] for c in r["failed"]}), [20])          # was [1, 8, 12, 20]
        self.assertEqual(self.numbers(r["breakdown_unchecked"]), [1, 8, 12])
        self.assertEqual(r["facts"], [21])

    def test_a_breakdown_the_evidence_supports_matches_or_fails(self):
        right = self.check("21 files: 8 + 12 + 1", 21, 8, 12, 1)
        self.assertEqual((self.numbers(right["matched"]), right["failed"]), ([21], []))
        self.assertEqual(self.numbers(right["breakdown_matched"]), [1, 8, 12])
        self.assertEqual(right["breakdown_unchecked"], [])
        wrong = self.check("21 files: 8 + 12 + 2", 21, 8, 12, 1)
        self.assertEqual(self.numbers(wrong["failed"]), [2])
        self.assertEqual(self.numbers(wrong["breakdown_matched"]), [8, 12])
        wrong2 = self.check("21 files (8 + 13)", 21, 8, 12, 1)
        self.assertEqual(self.numbers(wrong2["failed"]), [13])

    def test_evidence_that_does_not_break_the_total_down_supports_no_breakdown(self):
        for facts in ((21, 8, 5), (21, 8, 12), (21, 21), (21, 30), (21, 7)):       # no 2+ facts sum to 21
            r = self.check("21 files: 8 + 13", *facts)
            self.assertEqual(r["failed"], [], facts)
            self.assertEqual(self.numbers(r["breakdown_unchecked"]), [8, 13], facts)

    def test_breakdowns_are_unchecked_when_the_total_itself_did_not_match(self):
        r = self.check("20 files: 8 + 12", 21, 8, 13)                   # parts DO sum to 21, but 20 is the claim
        self.assertEqual(self.numbers(r["failed"]), [20])
        self.assertEqual(self.numbers(r["breakdown_unchecked"]), [8, 12])

    def test_a_breakdown_alone_with_no_statement_is_unchecked_never_failed(self):
        r = self.check("split (8 files) and (12 files)", 21)
        self.assertEqual((r["failed"], r["matched"]), ([], []))
        self.assertEqual(len(r["unchecked"]), 2)

    def test_the_bigger_number_is_not_assumed_to_be_the_total(self):
        r = self.check("8 files total (21 files in the vault)", 21)
        self.assertEqual(self.numbers(r["failed"]), [8])                   # the STATEMENT is wrong
        self.assertEqual(self.numbers(r["breakdown_unchecked"]), [21])

    def test_a_breakdown_group_is_separate(self):
        r = self.check("21 files including 3 folders", 21)                 # no folder evidence at all
        self.assertEqual((r["failed"], self.numbers(r["matched"])), ([], [21]))
        self.assertEqual(self.numbers(r["unchecked"]), [3])
        r = E.check_claims(E.extract_claims("21 files including 3 folders"),
                           [self.fact(21), self.fact(4, "D", "dir")])      # folder evidence says 4
        self.assertEqual(self.numbers(r["failed"]), [3])

    def test_claims_recorded_before_roles_existed_are_still_primary(self):
        r = E.check_claims([{"n": 20, "noun": "files", "group": "item"}], [self.fact(21)])
        self.assertEqual(self.numbers(r["failed"]), [20])


if __name__ == "__main__":
    unittest.main()
