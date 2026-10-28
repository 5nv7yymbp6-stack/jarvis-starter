"""Tests must never read or write the live voice session's files: its resume
pointer (.backtalk_session), its log, or its ledger. See tests/__init__.py."""
import os
import tempfile
import unittest
from types import SimpleNamespace

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import vlog
from backtalk.config import CFG, REPO

TMP = os.path.realpath(tempfile.gettempdir())


def inside_tmp(path):
    return os.path.realpath(str(path)).startswith(TMP + os.sep)


class LiveStateIsolationTests(unittest.TestCase):
    def test_the_session_file_is_a_temp_path_not_the_production_pointer(self):
        production = os.path.join(CFG["signals_dir"], ".backtalk_session")
        self.assertNotEqual(os.path.realpath(B.SESSION_FILE), os.path.realpath(production))
        self.assertTrue(inside_tmp(B.SESSION_FILE), B.SESSION_FILE)

    def test_the_log_is_a_temp_path_not_the_production_log(self):
        production = REPO / "logs" / "backtalk.log"
        self.assertNotEqual(os.path.realpath(vlog.LOG_PATH), os.path.realpath(production))
        self.assertTrue(inside_tmp(vlog.LOG_PATH), vlog.LOG_PATH)

    def test_the_default_ledger_is_a_temp_path_not_the_production_ledger(self):
        self.assertNotEqual(os.path.realpath(L.default_dir()), os.path.realpath(REPO / "ledger"))
        self.assertTrue(inside_tmp(L.default_dir()), L.default_dir())

    def test_remembering_a_session_writes_the_temp_file_only(self):
        with tempfile.TemporaryDirectory() as d:
            brain = B.WarmBrain(model="m", ledger=L.Ledger(d, fsync=False))
            brain._remember_session(SimpleNamespace(session_id="sess-isolation-guard"))
            self.assertTrue(CFG.get("resume_last_session"), "the write path under test is off")
            with open(B.SESSION_FILE) as f:
                self.assertEqual(f.read(), "sess-isolation-guard")
            brain.ledger.close()


if __name__ == "__main__":
    unittest.main()
