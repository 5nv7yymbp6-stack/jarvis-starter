# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The audio watchdog -- restart the voice line in place when an audio
call hangs.

Seen 2026-09-13 on macOS: a PortAudio stream stop (AudioOutputUnitStop)
never returned. It sat waiting on a CoreAudio lock, and the next mic open
blocked behind it. Nothing crashed and nothing was logged: the state said
"listening" and the voice line simply never heard another word until it
was restarted by hand.

A thread stuck inside CoreAudio can't be freed from Python, and a fresh
stream in the same process waits on the same lock, so rebuilding the
audio in-process doesn't recover. The process does: every blocking audio
call runs inside watch(), and a checker thread that finds one older than
STUCK_AFTER_S re-executes the voice line in place -- same pid, same
Terminal window -- resuming the same conversation. MAX_RESTARTS bounds
it, so audio that is broken for good can't turn into a restart loop.
"""
import itertools
import os
import signal
import subprocess
import sys
import threading
import time

from backtalk.vlog import log

STUCK_AFTER_S = 10.0     # no mic read, speaker write, or stream stop takes this long
CHECK_EVERY_S = 1.0
MAX_RESTARTS = 3         # within RESTART_WINDOW_S; past that, log it and stay down
RESTART_WINDOW_S = 600
# A hang within this long of an audio DEVICE error is the device, not a
# stuck process, and a restart can't fix it: seen 2026-09-13 after waking
# with AirPods, when every restart hit the same dead mic.
DEVICE_ERROR_WINDOW_S = 60.0

RESTARTS_ENV = "BACKTALK_AUDIO_RESTARTS"   # wall-clock times of recent restarts
RESUME_ENV = "BACKTALK_RESUME_ID"          # the conversation to pick back up
RECOVERED_ENV = "BACKTALK_AUDIO_RECOVERED"  # set on the restarted process

_calls: dict[int, tuple[str, float]] = {}   # id -> (what, monotonic start)
_lock = threading.Lock()
_ids = itertools.count()
_hooks: list = []
_session_source = None
_started = False
_device_error_at = -1e9          # monotonic time of the last device error
_declined: set = set()           # stuck calls already stood down for
_failure_hooks: list = []


class watch:
    """Time one blocking audio call:  with watch("stopping the mic"): ..."""
    __slots__ = ("what", "_id")

    def __init__(self, what: str):
        self.what = what

    def __enter__(self):
        self._id = next(_ids)
        with _lock:
            _calls[self._id] = (self.what, time.monotonic())
        return self

    def __exit__(self, *exc):
        with _lock:
            _calls.pop(self._id, None)
        return False


def before_restart(fn):
    """Run fn just before a restart. exec skips atexit, so anything that
    must be undone (the terminal mode, ducked music) registers here too."""
    _hooks.append(fn)


def set_session_source(fn):
    """fn() -> the current conversation's session id, or None."""
    global _session_source
    _session_source = fn


def device_error():
    """The audio system just reported a device-level failure."""
    global _device_error_at
    _device_error_at = time.monotonic()


def on_device_failure(fn):
    """Run fn when the watchdog stands down for a broken device (e.g. to
    tell the person out loud)."""
    _failure_hooks.append(fn)


def recovered() -> bool:
    """True once, in a process the watchdog just restarted."""
    return bool(os.environ.pop(RECOVERED_ENV, None))


def resume_id() -> str | None:
    """The conversation the restarted process should resume, once."""
    return os.environ.pop(RESUME_ENV, None) or None


def start():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_run, name="audio-watchdog", daemon=True).start()


def _oldest():
    with _lock:
        if not _calls:
            return None
        call_id, (what, since) = min(_calls.items(), key=lambda item: item[1][1])
    return call_id, what, time.monotonic() - since


def _check_once() -> bool:
    """One look at the audio calls. False means stop watching."""
    found = _oldest()
    if not found:
        return True
    call_id, what, age = found
    if age < STUCK_AFTER_S or call_id in _declined:
        return True
    if time.monotonic() - _device_error_at < DEVICE_ERROR_WINDOW_S:
        _declined.add(call_id)
        log(f"[watchdog] audio stuck {what} for {age:.0f}s right after an audio "
            f"device error -- not restarting, a restart can't fix the device. "
            f"Restart the Mac's audio (sudo killall coreaudiod) or reconnect "
            f"the mic, then start the voice line again.")
        for fn in _failure_hooks:
            try:
                fn()
            except Exception:
                pass
        return True
    return _restart(what, age)


def _run():
    while True:
        time.sleep(CHECK_EVERY_S)
        if not _check_once():
            return              # gave up: said so once, not every second


def _recent_restarts(now: float) -> list[float]:
    recent = []
    for part in os.environ.get(RESTARTS_ENV, "").split(","):
        try:
            t = float(part)
        except ValueError:
            continue
        if 0 <= now - t < RESTART_WINDOW_S:
            recent.append(t)
    return recent


def _restart(what: str, stuck_s: float) -> bool:
    now = time.time()
    recent = _recent_restarts(now)
    if len(recent) >= MAX_RESTARTS:
        log(f"[watchdog] audio stuck {what} for {stuck_s:.0f}s again, after "
            f"{len(recent)} restarts in {RESTART_WINDOW_S // 60} minutes -- "
            f"not restarting. The audio system itself looks broken: check "
            f"the devices, then start the voice line again.")
        return False
    log(f"[watchdog] audio stuck {what} for {stuck_s:.0f}s -- restarting "
        f"the voice line in place")
    for fn in _hooks:
        try:
            fn()
        except Exception:
            pass
    sid = None
    if _session_source is not None:
        try:
            sid = _session_source()
        except Exception:
            sid = None
    env = dict(os.environ)
    env[RESTARTS_ENV] = ",".join(str(t) for t in recent + [now])
    env[RECOVERED_ENV] = "1"
    if sid:
        env[RESUME_ENV] = sid
    else:
        env.pop(RESUME_ENV, None)
    _stop_children()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
    except OSError as e:
        log(f"[watchdog] restart failed: {e!r}")
    return False


def _alive(pid: int) -> bool:
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)     # reaps it if it has exited
        return done == 0
    except ChildProcessError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _stop_children():
    """The Claude Code session runs as a child of this process. Stop it
    (and any other direct child) before exec, so the resumed session is
    the only one writing that conversation."""
    try:
        out = subprocess.run(["pgrep", "-P", str(os.getpid())],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return
    pids = [int(p) for p in out.split() if p.isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 3
    while pids and time.monotonic() < deadline:
        pids = [p for p in pids if _alive(p)]
        if pids:
            time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
