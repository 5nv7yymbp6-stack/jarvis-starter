import unittest

from backtalk import main as M


class CodeRevision(unittest.TestCase):
    def run_with(self, sha, status):
        return M.code_revision(lambda *a: sha if a[0] == "rev-parse" else status)

    def test_clean_dirty_and_runtime_files_ignored(self):
        self.assertEqual(self.run_with("abc1234567\n", ""), "abc1234567 clean")
        self.assertEqual(self.run_with("abc1234567\n", " M backtalk/brain.py\n"), "abc1234567 dirty")
        self.assertEqual(self.run_with("abc1234567\n", " M .backtalk_session\n M watcher.log\n"), "abc1234567 clean")

    def test_unknown_when_git_cannot_answer(self):
        self.assertEqual(self.run_with("", ""), "unknown")

        def boom(*a):
            raise OSError("no git")
        self.assertEqual(M.code_revision(boom), "unknown")

    def test_the_real_tree_answers(self):
        self.assertRegex(M.code_revision(), r"^[0-9a-f]{10} (clean|dirty)$|^unknown$")


if __name__ == "__main__":
    unittest.main()
