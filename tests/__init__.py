"""Phase 1 tests (stdlib unittest). Run from the repo root:

    .venv/bin/python -m unittest discover -s tests -t . -v

The log is redirected first: a test must never write into the live session
log (see vlog.py)."""
import os
import tempfile

os.environ["BACKTALK_LOG"] = os.path.join(tempfile.gettempdir(),
                                          "backtalk-tests.log")

os.environ["BACKTALK_LEDGER_DIR"] = tempfile.mkdtemp(prefix="backtalk-tests-ledger-")

# ---- the live session's files are not the tests' files ----------------------
#
# brain.SESSION_FILE (<signals_dir>/.backtalk_session) is the LIVE voice
# session's resume pointer, and vlog.LOG_PATH is its log. Tests that drive
# WarmBrain write their fake session id ("sess-1") through _remember_session,
# and on 2026-10-01 a suite run overwrote the production pointer that way; a
# run started without `-t .` also put ~270 test lines into the production log.
#
# Both are module attributes fixed at import time, and
# `discover -s tests` (no `-t .`) imports backtalk.brain and backtalk.vlog
# BEFORE this package, so an environment variable set here would be too late.
# Rebinding the attributes holds in either import order. (CFG["signals_dir"] is
# also where the barehands/voice signal files live; the main-router tests patch
# those out themselves.)
from pathlib import Path  # noqa: E402

from backtalk import brain as _brain  # noqa: E402
from backtalk import vlog as _vlog  # noqa: E402

_STATE_DIR = tempfile.mkdtemp(prefix="backtalk-tests-state-")
_brain.SESSION_FILE = os.path.join(_STATE_DIR, ".backtalk_session")
_vlog.LOG_PATH = Path(os.environ["BACKTALK_LOG"])

# ---- the live config is not the tests' config --------------------------------------------------------------------
# CFG is loaded from the machine's real backtalk.json. Proactive behavior (reminders, spoken completions, the reminder rule in
# the system prompt, the denied self-scheduling tools) is switched ON there, and tests that assert the default prompt or the
# default wiring must not change with it. Anything that needs it ON sets it itself and restores it.
from backtalk.config import CFG as _CFG  # noqa: E402

_CFG["proactive"] = {**(_CFG.get("proactive") or {}), "enabled": False}
