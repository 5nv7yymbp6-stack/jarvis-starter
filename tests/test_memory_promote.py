import os
import tempfile
import unittest
from collections import Counter

from backtalk import memory_extract as X
from backtalk.evals.memory_extraction import promote as P
from backtalk.evals.memory_extraction import scan

GEN = "2026-10-21T06:00:00Z"
PREF = ('- [x] **PREFERENCE** (2026-10-06, inferred, voice, project: jarvis) Prefers short answers during calls '
        '— said: "keep it short when I am on a call" <!-- cand:aaaaaaaaaaaa -->')
GOAL = ('- [x] **PROJECT** (2026-10-06, inferred, voice, project: wholesaling) Wants to hire cold callers with side income '
        '— said: "so we can hire cold callers" <!-- cand:bbbbbbbbbbbb -->')
FACT = ('- [x] **IMPORTANT_FACT** (2026-10-06, inferred, voice, project: personal) Drives a GMC Acadia '
        '— said: "I drive a GMC Acadia" <!-- cand:cccccccccccc -->')
OPEN = ('- [ ] **PREFERENCE** (2026-10-07, inferred, voice, project: jarvis) Likes tea — said: "I like tea" <!-- cand:dddddddddddd -->')
INBOX_HEAD = "---\nstatus: active\nproject: meta\ntype: log\n---\n# Memory Candidates\n\n## Candidates\n"


class Vault:
    def __init__(self, test, lines):
        self.dir = tempfile.TemporaryDirectory()
        test.addCleanup(self.dir.cleanup)
        os.makedirs(os.path.join(self.dir.name, "00 - Inbox"))
        self.write(P.INBOX, INBOX_HEAD + "\n".join(lines) + "\n")
        lock = tempfile.NamedTemporaryFile(delete=False)
        lock.close()
        self.old_lock, P.LOCK = P.LOCK, lock.name
        test.addCleanup(lambda: setattr(P, "LOCK", self.old_lock))
        test.addCleanup(lambda: os.unlink(lock.name))

    def path(self, rel):
        return os.path.join(self.dir.name, rel)

    def write(self, rel, text):
        with open(self.path(rel), "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, rel):
        return open(self.path(rel), encoding="utf-8").read()


class AcceptanceTests(unittest.TestCase):
    def test_supported_classes_are_accepted(self):
        for line in (PREF, GOAL):
            self.assertEqual(P.accepted(line), (True, ""), line)
        lesson = PREF.replace("PREFERENCE", "LESSON")
        self.assertEqual(P.accepted(lesson), (True, ""))

    def test_a_fact_has_no_generic_key_and_stays_in_the_inbox(self):
        self.assertEqual(P.accepted(FACT), (False, "approved_class_needs_registry_key"))

    def test_malformed_secret_like_and_oversize_lines_are_refused(self):
        self.assertEqual(P.accepted("- [x] just some text")[0], False)
        secret = PREF.replace("short answers", "token 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ012345678")
        self.assertEqual(P.accepted(secret)[0], False)
        long = PREF.replace("short answers during calls", "x " * 200)
        self.assertEqual(P.accepted(long)[0], False)


class PromoteTests(unittest.TestCase):
    def test_ticked_supported_lines_move_verbatim_to_the_end_and_leave_the_inbox(self):
        v = Vault(self, [PREF, OPEN, GOAL, FACT])
        r = P.promote(v.dir.name)
        self.assertEqual((r["promoted"], r["already"], dict(r["left"])), (2, 0, {"approved_class_needs_registry_key": 1}))
        appr = v.read(P.APPROVED).split("\n")
        self.assertEqual(appr[-3:-1], [PREF, GOAL])                                    # verbatim, in inbox order, at the end
        inbox = v.read(P.INBOX)
        self.assertNotIn("cand:aaaaaaaaaaaa", inbox)
        self.assertNotIn("cand:bbbbbbbbbbbb", inbox)
        self.assertIn(OPEN, inbox)                                                     # an unticked line is untouched
        self.assertIn(FACT, inbox)                                                     # an unsupported ticked line stays, ticked

    def test_the_approved_note_is_created_with_frontmatter_and_the_append_only_warning(self):
        v = Vault(self, [PREF])
        P.promote(v.dir.name)
        text = v.read(P.APPROVED)
        self.assertTrue(text.startswith("---\nstatus: active"))
        self.assertIn("APPEND-ONLY", text)

    def test_it_is_idempotent_and_never_duplicates(self):
        v = Vault(self, [PREF])
        P.promote(v.dir.name)
        first = v.read(P.APPROVED)
        v.write(P.INBOX, v.read(P.INBOX) + PREF + "\n")                               # the same tag ticked again
        r = P.promote(v.dir.name)
        self.assertEqual((r["promoted"], r["already"]), (0, 1))
        self.assertEqual(v.read(P.APPROVED), first)
        self.assertNotIn("cand:aaaaaaaaaaaa", v.read(P.INBOX))

    def test_appending_never_shifts_an_earlier_lines_number(self):
        v = Vault(self, [PREF])
        P.promote(v.dir.name)
        before = [i for i, l in enumerate(v.read(P.APPROVED).split("\n"), 1) if "cand:aaaaaaaaaaaa" in l]
        v.write(P.INBOX, v.read(P.INBOX) + GOAL + "\n")
        P.promote(v.dir.name)
        after = [i for i, l in enumerate(v.read(P.APPROVED).split("\n"), 1) if "cand:aaaaaaaaaaaa" in l]
        self.assertEqual(before, after)

    def test_a_crash_after_the_approved_write_leaves_only_a_clearable_duplicate(self):
        v = Vault(self, [PREF])
        real = P._write_atomic
        calls = []

        def flaky(path, text):
            calls.append(path)
            if path.endswith("Memory Candidates.md"):
                raise OSError("disk full")
            real(path, text)
        P._write_atomic = flaky
        try:
            with self.assertRaises(OSError):
                P.promote(v.dir.name)
        finally:
            P._write_atomic = real
        self.assertIn("cand:aaaaaaaaaaaa", v.read(P.INBOX))                            # still there
        self.assertIn("cand:aaaaaaaaaaaa", v.read(P.APPROVED))                         # and already appended
        r = P.promote(v.dir.name)
        self.assertEqual((r["promoted"], r["already"]), (0, 1))                        # the next run only clears it
        self.assertEqual(v.read(P.APPROVED).count("cand:aaaaaaaaaaaa"), 1)

    def test_it_touches_only_the_two_inbox_notes(self):
        v = Vault(self, [PREF, GOAL])
        os.makedirs(v.path("11 - Memory"))
        v.write("11 - Memory/sentinel.txt", "x")
        P.promote(v.dir.name)
        self.assertEqual(open(v.path("11 - Memory/sentinel.txt")).read(), "x")
        self.assertEqual(sorted(os.listdir(v.path("11 - Memory"))), ["sentinel.txt"])
        self.assertEqual(sorted(os.listdir(v.path("00 - Inbox"))), ["Approved Memory Statements.md", "Memory Candidates.md"])

    def test_no_inbox_is_a_quiet_no_op(self):
        v = Vault(self, [PREF])
        os.unlink(v.path(P.INBOX))
        self.assertEqual(P.promote(v.dir.name)["promoted"], 0)

    def test_it_never_prints_the_statements(self):
        import contextlib
        import io
        v = Vault(self, [PREF, FACT])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(["--vault", v.dir.name])
        self.assertNotIn("short", buf.getvalue())
        self.assertNotIn("Acadia", buf.getvalue())
        self.assertIn("1 promoted", buf.getvalue())


class ExtractorIntegrationTests(unittest.TestCase):
    def cands(self, lines):
        text = "\n".join(lines) + "\n"
        note = X.Note(X.INBOX_APPROVED, text, X.sha(text))
        return X.build_candidates([], [note], generated_at=GEN)

    def test_a_promoted_line_is_offered_as_a_note_sourced_candidate_that_needs_his_affirmation(self):
        (c,), excl = self.cands([PREF])
        self.assertEqual((c.type, c.channel, c.source_kind, c.confidence), ("preference", "note", "note", "inferred"))
        self.assertTrue(c.attribute.startswith("pref_"))
        self.assertTrue(c.requires_confirmation)
        self.assertIn("note_authored_by_assistant", c.confirmation_reasons)
        self.assertIn(c.value, c.evidence)
        self.assertEqual(c.where, X.INBOX_APPROVED)

    def test_goal_and_lesson_mappings_and_unticked_or_unsupported_lines_are_ignored(self):
        cands, excl = self.cands([GOAL, PREF.replace("PREFERENCE", "LESSON").replace("aaaaaaaaaaaa", "eeeeeeeeeeee"), OPEN, FACT])
        types = sorted((c.type, c.attribute.split("_")[0]) for c in cands)
        self.assertEqual(types, [("goal", "goal"), ("instruction", "rule")])
        self.assertEqual(excl["approved_class_needs_registry_key"], 1)

    def test_the_candidate_points_at_a_stable_line_number(self):
        head = "# Approved\n\n## Approved\n"
        text = head + PREF + "\n"
        (c,), _ = X.build_candidates([], [X.Note(X.INBOX_APPROVED, text, X.sha(text))], generated_at=GEN)
        text2 = text + GOAL + "\n"
        cands2, _ = X.build_candidates([], [X.Note(X.INBOX_APPROVED, text2, X.sha(text2))], generated_at=GEN)
        same = [d for d in cands2 if d.id == c.id][0]
        self.assertEqual(same.line, c.line)

    def test_the_scan_reads_the_approved_note_and_nothing_in_the_inbox_note_itself(self):
        self.assertIn(X.INBOX_APPROVED, scan.EXTRACT_NOTES)
        self.assertNotIn(P.INBOX, scan.EXTRACT_NOTES)                                  # unticked proposals are never extracted

    def test_it_creates_no_memory_and_imports_no_store(self):
        import inspect
        src = inspect.getsource(P)
        for banned in ("memory_store", "memory_execute", "memory_commit", "Authorization", "tier2", "11 - Memory"):
            self.assertNotIn(banned, src.replace("11 - Memory", "") if banned == "11 - Memory" else src, banned)


if __name__ == "__main__":
    unittest.main()
