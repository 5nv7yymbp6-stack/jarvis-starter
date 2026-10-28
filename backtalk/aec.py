"""Echo cancellation: the mic and speaker through Apple voice processing.

Jarvis's voice comes out of the MacBook speakers and straight back into the
MacBook mic. Without cancellation the ears can only tell the person from the
voice by comparing transcripts after the fact (main._is_echo), which drops real
answers and can never let the person cut in the moment they start talking.

aec/jarvis-aec (Swift, built locally with swiftc; see its header) runs Apple's
VoiceProcessingIO unit: it plays what we send and records the mic with that
playback removed. Measured on this MacBook Air (2026-09-14): a spoken clip that
reached the mic at -30 dBFS through the plain path came back at the -64 dBFS
noise floor, and Whisper transcribed nothing.

This module is the Python side: one helper process shared by the ears (a
stand-in for sounddevice.InputStream) and the mouth (a stand-in for
sounddevice.OutputStream). Config key "echo_cancel" (see config.py). Off, or
the helper missing, or the default output not the speakers (headphones have no
echo) -> the normal PortAudio path, unchanged.
"""
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from backtalk.config import CFG
from backtalk import audiodev
from backtalk.vlog import log

HELPER = Path(__file__).resolve().parent.parent / "aec" / "jarvis-aec"
MIC_RATE = 16000       # what the ears read
HELPER_MIC_RATE = 24000  # what the helper records (the unit refuses 16 kHz)
PLAY_RATE = 24000
LEAD_S = 0.2           # how far ahead of the speaker a write may run
PLAY_CALL_MAX_S = 6.0  # longest one write may take, pauses included (watchdog is 10 s)
MIC_KEEP_S = 10        # unread mic audio kept before the oldest is dropped


class AecUnavailable(RuntimeError):
    pass


_decided: bool | None = None
_decided_at: float = 0.0
_decide_lock = threading.Lock()
# How often to re-check the default output device. Not every call: the ears
# and the mouth both call enabled() independently and must keep agreeing
# with each other mid-utterance, so this trades a few seconds of possible
# lag for that agreement instead of re-deciding on every call.
RECHECK_S = 3.0


def _settings() -> dict:
    return CFG.get("echo_cancel") or {}


def _decide() -> bool:
    """The actual on/off logic, unmemoized. Called by enabled() under the
    lock, on a cadence, not every time -- see RECHECK_S."""
    cfg = _settings()
    if not cfg.get("enabled"):
        return False
    if not HELPER.exists():
        log(f"[aec] echo_cancel is on but {HELPER.name} isn't built -- "
            f"using the plain mic and speaker")
        return False
    speaker = str(cfg.get("speaker") or "MacBook Air Speakers")
    if cfg.get("only_on_speakers", True):
        # audiodev, NOT sounddevice: PortAudio freezes its idea of the
        # default output at startup and never updates it, so asking it here
        # meant this decision could never change for the life of the
        # process. That is exactly how Yousif ended up wearing AirPods
        # while every word came out of the MacBook speaker on 2026-09-22 --
        # making this re-decide every 3 s (the 2026-09-21 fix) did nothing,
        # because it kept re-deciding from the same frozen answer.
        out = audiodev.default_output_name()
        if not out:
            log("[aec] could not read the default output -- "
                "echo cancellation off")
            return False
        if speaker.lower() not in out.lower():
            log(f"[aec] default output is {out!r}, not the speakers -- "
                f"no echo to cancel, so echo cancellation is off")
            return False
    return True


def enabled() -> bool:
    """Whether echo cancellation should be in use right now.

    Re-decided every RECHECK_S seconds rather than once for the life of
    the process (the original behavior, kept as the default cadence
    floor). Found 2026-09-21: deciding once at launch meant that plugging
    in / connecting AirPods mid-session after the voice line started on
    the MacBook speakers left AEC "on" forever, still driving its helper
    process as if the speakers were still the output, while the real
    default output had silently become the AirPods underneath it -- the
    voice line went silent and nothing in the code ever re-checked.
    A device FLIP (on<->off) also tears down the running helper, so the
    next engine() call starts a fresh one against the current device
    instead of a stale one."""
    global _decided, _decided_at
    with _decide_lock:
        now = time.monotonic()
        if _decided is not None and (now - _decided_at) < RECHECK_S:
            return _decided
        new = _decide()
        first_decision = _decided is None
        flipped = not first_decision and new != _decided
        _decided, _decided_at = new, now
        if flipped:
            log(f"[aec] output device changed -- echo cancellation now "
                f"{'on' if new else 'off'}, restarting the helper")
            _reset_engine()
        elif first_decision and new:
            log("[aec] echo cancellation on")
        return new


_out_is_speaker = None
_out_checked_at = 0.0


def on_speakers() -> bool:
    """True when the Mac's default output is the built-in speaker, so the
    voice can leak back into the mic.

    False on headphones or AirPods, where the mic never hears the voice at
    all. That's why cutting in is safe there even with echo cancellation
    off: found 2026-09-21, Yousif on AirPods turned echo cancellation off,
    which also switched off the instant pause, so the voice talked straight
    over him. Cached on the same cadence as enabled()."""
    global _out_is_speaker, _out_checked_at
    with _decide_lock:
        now = time.monotonic()
        if _out_is_speaker is not None and (now - _out_checked_at) < RECHECK_S:
            return _out_is_speaker
        speaker = str(_settings().get("speaker") or "MacBook Air Speakers")
        out = audiodev.default_output_name()   # CoreAudio, see _decide()
        if out:
            is_speaker = speaker.lower() in out.lower()
        else:
            log("[aec] could not read the default output -- "
                "assuming the speakers")
            is_speaker = True            # unknown: assume the risky case
        _out_is_speaker, _out_checked_at = is_speaker, now
        return is_speaker


class _Engine:
    """One jarvis-aec process: mic audio in, speaker audio out."""

    def __init__(self):
        cfg = _settings()
        self.proc = subprocess.Popen(
            [str(HELPER), str(cfg.get("mic") or "MacBook Air Microphone"),
             str(cfg.get("speaker") or "MacBook Air Speakers")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        first: list[str] = []
        got = threading.Event()

        def _first_line():
            first.append(self.proc.stderr.readline().decode(errors="replace").strip())
            got.set()
        threading.Thread(target=_first_line, daemon=True).start()
        if not got.wait(5.0) or not first or not first[0].startswith("[aec] ready"):
            said = first[0] if first else "nothing within 5 s"
            self._kill()
            raise AecUnavailable(f"jarvis-aec did not start: {said}")
        log(first[0])
        self._buf = bytearray()
        self._cond = threading.Condition()
        self._dead = False
        self._send_lock = threading.Lock()
        self._clock: float | None = None   # when the audio already sent runs out
        self._paused_at: float | None = None
        threading.Thread(target=self._read_mic, daemon=True).start()
        threading.Thread(target=self._read_log, daemon=True).start()

    # ---- mic --------------------------------------------------------------
    def _to_mic_rate(self, b: bytes) -> bytes:
        """24 kHz int16 bytes -> 16 kHz int16 bytes, streaming: a sample every
        1.5 input samples, linearly interpolated. Leftover bytes and the
        fractional read position carry over to the next call."""
        b = self._byte_carry + b
        whole = len(b) - (len(b) % 2)
        self._byte_carry = b[whole:]
        x = np.concatenate([self._carry,
                            np.frombuffer(b[:whole], dtype="<i2").astype(np.float32)])
        step = HELPER_MIC_RATE / MIC_RATE
        if len(x) < 2 or self._pos > len(x) - 2:
            self._carry = x
            return b""
        n = int((len(x) - 2 - self._pos) // step) + 1
        ps = self._pos + step * np.arange(n)
        i = ps.astype(np.int64)
        f = (ps - i).astype(np.float32)
        y = x[i] * (1 - f) + x[i + 1] * f
        nxt = ps[-1] + step
        drop = min(int(nxt), len(x))
        self._carry = x[drop:]
        self._pos = nxt - drop
        return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()

    def _read_mic(self):
        keep = MIC_RATE * 2 * MIC_KEEP_S
        self._byte_carry = b""
        self._carry = np.zeros(0, dtype=np.float32)
        self._pos = 0.0
        while True:
            try:
                b = self.proc.stdout.read(4096)
            except Exception:
                b = b""
            if not b:
                break
            b = self._to_mic_rate(b)
            if not b:
                continue
            with self._cond:
                self._buf.extend(b)
                if len(self._buf) > keep:
                    del self._buf[:len(self._buf) - keep]
                self._cond.notify_all()
        with self._cond:
            self._dead = True
            self._cond.notify_all()

    def _read_log(self):
        for line in self.proc.stderr:
            text = line.decode(errors="replace").rstrip()
            if text:
                log(text)

    @property
    def alive(self) -> bool:
        return not self._dead and self.proc.poll() is None

    def available(self) -> int:
        with self._cond:
            return len(self._buf) // 2

    def read(self, frames: int, timeout: float = 2.0) -> np.ndarray:
        need = frames * 2
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._buf) < need:
                if self._dead:
                    raise AecUnavailable("jarvis-aec exited")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AecUnavailable("echo-cancelled mic stalled")
                self._cond.wait(left)
            data = bytes(self._buf[:need])
            del self._buf[:need]
        return np.frombuffer(data, dtype="<i2").astype(np.int16).reshape(-1, 1)

    # ---- speaker ----------------------------------------------------------
    def _send(self, kind: bytes, payload: bytes = b""):
        with self._send_lock:
            try:
                self.proc.stdin.write(kind + struct.pack("<I", len(payload)) + payload)
            except (BrokenPipeError, OSError, ValueError) as e:
                raise AecUnavailable(f"jarvis-aec gone: {e}")

    def play(self, pcm, rate: int):
        """Queue audio and return at a real speaker's pace, never more than
        LEAD_S ahead, so a flush can still cut it (and the mouth's timing,
        waveform and state stay true to what is audible). While paused it
        waits like a speaker that isn't taking audio. One call never runs
        past PLAY_CALL_MAX_S, pauses included, so the audio watchdog (10 s)
        never sees a hang: a first version waited out each pause and then a
        resume-stretched pacing sleep in the same call, and tripped it."""
        deadline = time.monotonic() + PLAY_CALL_MAX_S
        while self._paused_at is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if not pcm.size:
            return
        if rate != PLAY_RATE:
            n = int(round(pcm.size * PLAY_RATE / rate))
            pcm = np.interp(np.linspace(0, pcm.size - 1, n), np.arange(pcm.size),
                            pcm).astype(np.int16)
        self._send(b"A", pcm.astype("<i2").tobytes())
        now = time.monotonic()
        start = max(now, self._clock or now)
        self._clock = start + pcm.size / PLAY_RATE
        while time.monotonic() < deadline:
            clock = self._clock
            if self._paused_at is None and (clock is None or clock - LEAD_S <= time.monotonic()):
                break
            time.sleep(0.01)

    def flush(self):
        """Barge-in: everything not yet played is dropped, now."""
        self._clock = None
        self._paused_at = None
        self._send(b"F")

    def pause(self):
        """Something may be cutting in: play silence, keep the place."""
        if self._paused_at is None:
            self._paused_at = time.monotonic()
        self._send(b"P")

    def resume(self):
        """It was noise: carry on from where it paused. The queue's run-out
        time moves back by however long the pause lasted."""
        if self._paused_at is not None and self._clock is not None:
            self._clock += time.monotonic() - self._paused_at
        self._paused_at = None
        self._send(b"R")

    def _kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def close(self):
        try:
            self._send(b"Q")
            self.proc.wait(timeout=1.0)
        except Exception:
            self._kill()


_engine: _Engine | None = None
_engine_lock = threading.Lock()


def engine() -> _Engine:
    """The shared helper, started (or restarted, if it died) on demand."""
    global _engine
    with _engine_lock:
        if _engine is None or not _engine.alive:
            _engine = _Engine()
        return _engine


def _reset_engine():
    """Kill the running helper (if any) without starting a new one -- the
    next engine() call rebuilds fresh. Called when enabled() sees the
    output device change out from under it, so a stale helper never keeps
    running against a device that's no longer real."""
    global _engine
    with _engine_lock:
        old, _engine = _engine, None
    if old is not None:
        try:
            old.close()
        except Exception:
            pass


def _now(action: str):
    """Send the running helper flush / pause / resume, from any thread.
    Never raises, never starts a helper just to do it."""
    with _engine_lock:
        e = _engine
    if e is not None and e.alive:
        try:
            getattr(e, action)()
        except Exception:
            pass


def flush_now():
    _now("flush")


def pause_now():
    _now("pause")


def resume_now():
    _now("resume")


class InputStream:
    """Stands in for sounddevice.InputStream in the ears (int16, 16 kHz, mono)."""

    def start(self):
        engine()

    @property
    def read_available(self) -> int:
        return engine().available()

    def read(self, frames: int):
        return engine().read(frames), False

    def stop(self):
        pass

    def close(self, ignore_errors=True):
        pass


class OutputStream:
    """Stands in for sounddevice.OutputStream in the mouth."""

    def __init__(self, rate: int):
        self.rate = rate
        self.active = True

    def start(self):
        engine()

    def write(self, pcm):
        engine().play(pcm, self.rate)

    def flush(self):
        engine().flush()

    def close(self, ignore_errors=True):
        pass
