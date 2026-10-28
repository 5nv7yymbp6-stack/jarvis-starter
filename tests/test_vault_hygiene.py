"""status/vault_hygiene.py: read-only structural checks of the vault. A temp vault; the real one is never read."""
import importlib.util
import json
import os
import tempfile
import time
import unittest

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "status", "vault_hygiene.py")
NOW = 2_000_000_000.0
FM = "---\nstatus: active\nproject: meta\ntype: log\n---\n"


@unittest.skipUnless(os.path.exists(PATH), "status/vault_hygiene.py is not on this machine")
class HygieneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = os.path.join(self.tmp.name, "v")
        os.makedirs(self.vault)
        self.index = os.path.join(self.tmp.name, "index.json")
        spec = importlib.util.spec_from_file_location("hyg_under_test", PATH)
        self.h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.h)

    def tearDown(self):
        self.tmp.cleanup()

    def note(self, rel, text=FM + "body\n", mtime=NOW - 100):
        p = os.path.join(self.vault, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        os.utime(p, (mtime, mtime))
        return p

    def write_index(self, files):
        with open(self.index, "w") as fh:
            json.dump({"files": files, "entries": []}, fh)
        os.utime(self.index, (NOW - 60, NOW - 60))

    def run_h(self):
        return self.h.analyze(self.vault, self.index, NOW)

    def test_a_clean_vault_is_clean(self):
        self.note("a.md")
        self.write_index({"a.md": NOW - 100})
        res = self.run_h()
        lines, level = self.h.report(res)
        self.assertEqual(level, 0, lines)
        self.assertEqual(res["missing_frontmatter"], [])

    def test_missing_frontmatter_is_found_and_the_memory_folder_is_exempt(self):
        self.note("a.md", "no frontmatter\n")
        self.note("b.md", "---\nstatus: active\n---\nbody")                   # missing project and type
        self.note("11 - Memory/mem_1.md", '---\nstatus: "active"\n---\nbody')   # machine-owned: no project key by design
        self.write_index({})
        self.assertEqual(self.run_h()["missing_frontmatter"], ["a.md", "b.md"])

    def test_duplicate_names_ignore_daily_notes(self):
        self.note("06 - Learning/Same.md")
        self.note("09 - Resources/Same.md")
        self.note("01 - Daily Notes/10 - October 2026/2026-10-19.md")
        self.note("01 - Daily Notes/09 - September 2026/2026-10-19.md")
        self.write_index({})
        self.assertEqual(self.run_h()["duplicate_names"], [["06 - Learning/Same.md", "09 - Resources/Same.md"]])

    def test_index_lag_missing_and_stale_notes(self):
        self.note("fresh.md", mtime=NOW - 10)
        self.note("old.md", mtime=NOW - 5000)
        self.note("new.md", mtime=NOW - 5)
        self.note("a transcript.md", mtime=NOW - 5)
        self.write_index({"fresh.md": NOW - 500, "old.md": NOW - 5000})
        res = self.run_h()
        self.assertEqual(res["index_stale_files"], ["fresh.md"])                  # changed after it was indexed
        self.assertEqual(res["index_missing_files"], ["new.md"])                  # never indexed; transcripts are skipped by design
        self.assertAlmostEqual(res["index_age_s"], 60, delta=1)

    def test_an_unreadable_or_missing_index_is_reported_not_fatal(self):
        self.note("a.md")
        self.assertIsNone(self.run_h()["index_age_s"])
        lines, level = self.h.report(self.run_h())
        self.assertEqual(level, 1)
        with open(self.index, "w") as fh:
            fh.write("{broken")
        self.assertIsNone(self.run_h()["index_age_s"])

    def test_priorities_size_and_stale_linked_notes(self):
        items = "\n".join(f"- [ ] item {i} see [[Old Note]]" for i in range(61))
        self.note("Active Priorities.md", FM + items + "\n- [x] done [[Fresh Note]]\n", mtime=NOW - 10)
        self.note("Old Note.md", mtime=NOW - 40 * 86400)
        self.note("Fresh Note.md", mtime=NOW - 86400)
        self.note("01 - Daily Notes/2026-01-01.md", mtime=NOW - 90 * 86400)
        self.write_index({})
        res = self.run_h()
        self.assertEqual(res["open_priorities"], 61)
        self.assertEqual(res["stale_linked"], ["Old Note.md"])
        lines, level = self.h.report(res, show_files=True)
        self.assertEqual(level, 1)
        self.assertTrue(any("61 open items" in l for l in lines))
        self.assertTrue(any("Old Note.md" in l for l in lines))

    def test_the_report_never_contains_note_text(self):
        self.note("a.md", FM + "SECRET-BODY-TEXT-77\n")
        self.write_index({"a.md": NOW - 100})
        lines, _ = self.h.report(self.run_h(), show_files=True)
        self.assertNotIn("SECRET-BODY-TEXT-77", "\n".join(lines))

    def test_it_never_writes_anything(self):
        self.note("a.md")
        self.write_index({})
        before = sorted((n, os.stat(os.path.join(self.vault, n)).st_mtime) for n in os.listdir(self.vault))
        self.run_h()
        self.assertEqual(sorted((n, os.stat(os.path.join(self.vault, n)).st_mtime) for n in os.listdir(self.vault)), before)


if __name__ == "__main__":
    unittest.main()
