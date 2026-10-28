"""Sign-in guard: one clear message, no restart storm.

THE PROBLEM THIS SOLVES. When the Claude Code login has expired, the brain connect fails within seconds on EVERY start.
Backtalk spoke "couldn't reach my brain" and exited with code 1, and whatever relaunched it (a terminal loop, the watcher
agent) started it again, so the same error was spoken every ~40 seconds, indefinitely.

WHAT IT DOES.
  * After a brain-connect failure it asks the CLI, LOCALLY and without a model call, whether the login is signed in
    (`claude auth status`). Only a definite "signed out" is an AUTHENTICATION failure. Anything else (signed in, or the
    check itself could not answer) is a TRANSIENT failure and keeps the existing behavior exactly: the same spoken
    message and exit code 1. There is no message-text matching and there are no retries anywhere in this module.
  * An authentication failure speaks ONE sign-in message, writes a small marker file, and exits with code 78
    (EX_CONFIG: "configuration needed, do not relaunch").
  * While the marker exists, a start does one cheap local check BEFORE loading any model or speaking: still signed out
    means print the instruction and exit 78 immediately and silently; signed in means clear the marker and start
    normally. So an external relauncher that keeps relaunching costs about a second and says nothing.
  * A successful brain connect clears the marker.

NOT IN SCOPE. It does not touch credentials (it only reads whether the CLI says it is logged in), does not sign in for
the person, and does not stop an external loop from running; it makes each iteration cheap, silent and recognisable
(exit code 78).
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

SIGNED_IN, SIGNED_OUT, UNKNOWN = "signed_in", "signed_out", "unknown"
EXIT_SIGN_IN_REQUIRED = 78
EXIT_TRANSIENT = 1
MARKER_NAME = ".voice_auth_required"            # the .voice_* glob is already untracked
STATUS_TIMEOUT_S = 10
SIGNIN_COMMAND = "claude auth login"

# Exactly the text main.py used for every connect failure before this guard; transient failures keep it unchanged.
TRANSIENT_SPOKEN = ("Bad news. The voice and the face are fine, but I "
                    "couldn't reach my brain, the Claude Code session. "
                    "Check this window for the error. The usual causes: "
                    "Claude Code isn't signed in, the internet is down, "
                    "or the plan is out of usage.")
SIGNIN_SPOKEN = ("Claude Code needs you to sign in again. Open a terminal, run claude auth login, and start me again "
                 "afterwards. I won't keep retrying until you do.")
SIGNIN_PRINTED = ("[backtalk] SIGN-IN REQUIRED: the Claude Code login has expired. Run this in a terminal, finish the "
                  f"browser step, then start backtalk again:  {SIGNIN_COMMAND}")


def cli_path():
    """The Claude CLI the brain uses: the SDK's bundled binary when present, else one on PATH, else None."""
    try:
        import claude_agent_sdk
        bundled = os.path.join(os.path.dirname(claude_agent_sdk.__file__), "_bundled", "claude")
        if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
            return bundled
    except Exception:
        pass
    return shutil.which("claude")


def signed_in_status(cli=None, run=subprocess.run, timeout=STATUS_TIMEOUT_S) -> str:
    """SIGNED_IN / SIGNED_OUT from `claude auth status` (a local check, no model call), or UNKNOWN when it cannot say."""
    cli = cli or cli_path()
    if not cli:
        return UNKNOWN
    try:
        r = run([cli, "auth", "status"], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        out = r.stdout or ""
        start = out.find("{")
        data = json.loads(out[start:]) if start >= 0 else None
    except Exception:
        return UNKNOWN
    if not isinstance(data, dict):
        return UNKNOWN
    logged = data.get("loggedIn")
    if logged is True:
        return SIGNED_IN
    if logged is False:
        return SIGNED_OUT
    return UNKNOWN


def default_marker_path() -> str:
    from backtalk.config import CFG
    return os.path.join(CFG["signals_dir"], MARKER_NAME)


def marker_present(path) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


def write_marker(path, now=None) -> bool:
    """Atomically write the marker (a timestamp and the reason: no secrets). Never raises; False if it could not."""
    try:
        body = json.dumps({"at": int(now if now is not None else time.time()), "reason": "signed_out"})
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".voice_auth_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception:
        return False


def clear_marker(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def startup_gate(path, status_fn=signed_in_status):
    """Run at the very start of a launch. Returns EXIT_SIGN_IN_REQUIRED when the marker is present AND the login is
    still signed out (the caller prints SIGNIN_PRINTED and exits), else None to continue. A signed-in answer clears the
    marker; an UNKNOWN answer lets the start proceed so a broken status check can never lock the person out."""
    if not marker_present(path):
        return None
    st = status_fn()
    if st == SIGNED_OUT:
        return EXIT_SIGN_IN_REQUIRED
    if st == SIGNED_IN:
        clear_marker(path)
    return None


@dataclass(frozen=True)
class Decision:
    kind: str                 # "auth" | "transient"
    speak: object             # text to say, or None for silence
    exit_code: int
    write_marker: bool


def decide_connect_failure(status, marker_was_present) -> Decision:
    """What to do after a brain-connect failure, from the LOCAL sign-in status only (never from error text).
    Authentication: speak once (not again while the marker already exists), mark, exit 78. Everything else is
    transient and unchanged: the original message, exit 1, no marker. Nothing here retries."""
    if status == SIGNED_OUT:
        return Decision("auth", None if marker_was_present else SIGNIN_SPOKEN, EXIT_SIGN_IN_REQUIRED, True)
    return Decision("transient", TRANSIENT_SPOKEN, EXIT_TRANSIENT, False)
