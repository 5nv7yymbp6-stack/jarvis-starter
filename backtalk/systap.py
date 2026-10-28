"""What the Mac itself is playing, so the voice line can ignore it.

A video or music playing on this Mac comes out of its speakers and into the
mic, and the voice line took it for the person (2026-09-14: printer tutorials
transcribed as [you] and answered). Echo cancellation only removes the voice's
own playback, and voice ID couldn't separate Yousif from a narrator (shadow
scores overlapped, 2026-09-15). This compares instead: what did the Mac play
while the mic heard that?

aec/jarvis-systap (Swift, a CoreAudio process tap) streams every app's output as
int16 mono 16 kHz. This module keeps the last KEEP_S seconds with their times,
so main.py can transcribe what the Mac played during an utterance and drop the
utterance when the words match. Config: "ignore_mac_audio" (see config.py).
"""
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from backtalk.config import CFG
from backtalk.vlog import log

HELPER = Path(__file__).resolve().parent.parent / "aec" / "jarvis-systap"
RATE = 16000
KEEP_S = 60

START_WAIT_S = 15      # how long a starting helper gets to say "ready"
RETRY_S = 30           # after a failed start, the next try waits this long

_decided: bool | None = None
_lock = threading.Lock()
_tap = None
_starting = False
_next_try = 0.0


def enabled() -> bool:
    """Decided once per launch."""
    global _decided
    with _lock:
        if _decided is None:
            want = bool((CFG.get("ignore_mac_audio") or {}).get("enabled"))
            if want and not HELPER.exists():
                log(f"[systap] ignore_mac_audio is on but {HELPER.name} isn't built -- off")
            _decided = want and HELPER.exists()
        return _decided


class _Tap:
    def __init__(self):
        self.proc = subprocess.Popen([str(HELPER)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0)
        first: list[str] = []
        got = threading.Event()

        def _first_line():
            first.append(self.proc.stderr.readline().decode(errors="replace").strip())
            got.set()
        threading.Thread(target=_first_line, daemon=True).start()
        if not got.wait(START_WAIT_S) or not first or not first[0].startswith("[systap] ready"):
            said = first[0] if first else f"nothing within {START_WAIT_S} s"
            try:
                self.proc.kill()
            except Exception:
                pass
            raise RuntimeError(f"jarvis-systap did not start: {said}")
        log(first[0])
        self._buf = np.zeros(RATE * KEEP_S, dtype=np.int16)
        self._n = 0                    # samples received in total
        self._t_last: float | None = None
        self._buf_lock = threading.Lock()
        self._dead = False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_log, daemon=True).start()

    def _read(self):
        carry = b""
        size = len(self._buf)
        while True:
            try:
                b = self.proc.stdout.read(4096)
            except Exception:
                b = b""
            if not b:
                break
            b = carry + b
            whole = len(b) - (len(b) % 2)
            carry = b[whole:]
            x = np.frombuffer(b[:whole], dtype="<i2")
            if not x.size:
                continue
            with self._buf_lock:
                idx = (self._n + np.arange(x.size)) % size
                self._buf[idx] = x
                self._n += x.size
                self._t_last = time.monotonic()
        self._dead = True

    def _read_log(self):
        for line in self.proc.stderr:
            text = line.decode(errors="replace").rstrip()
            if text:
                log(text)

    @property
    def alive(self) -> bool:
        return not self._dead and self.proc.poll() is None

    def window(self, t0: float, t1: float) -> np.ndarray | None:
        """What the Mac played between two time.monotonic() times."""
        size = len(self._buf)
        with self._buf_lock:
            if self._t_last is None:
                return None
            n, t_last = self._n, self._t_last
            i0 = n - int(round((t_last - t0) * RATE))
            i1 = n - int(round((t_last - t1) * RATE))
            i0 = max(i0, n - size, 0)
            i1 = min(i1, n)
            if i1 <= i0:
                return None
            return self._buf[np.arange(i0, i1) % size].copy()


def start():
    """Start (or restart) the helper in the background. Never blocks the
    caller: the first version started it inline and gave up after 5 s, and
    because a failed start was retried on every utterance, each line could
    stall the voice line for 5 s (live, 2026-09-15 00:24 AM, when the start
    came right after the previous voice line's audio helper was killed)."""
    global _starting, _next_try
    with _lock:
        if _starting or (_tap is not None and _tap.alive) or time.monotonic() < _next_try:
            return
        _starting = True

    def _run():
        global _tap, _starting, _next_try
        try:
            t = _Tap()
            with _lock:
                _tap = t
        except Exception as e:
            log(f"[systap] could not start ({e}) -- retrying in {RETRY_S} s; "
                f"Mac audio isn't ignored until then")
            with _lock:
                _next_try = time.monotonic() + RETRY_S
        finally:
            with _lock:
                _starting = False
    threading.Thread(target=_run, daemon=True).start()


def window(t0: float, t1: float) -> np.ndarray | None:
    """What the Mac played, or None while the helper isn't running (it is
    started or retried in the background). Never blocks, never raises."""
    with _lock:
        t = _tap
    if t is None or not t.alive:
        start()
        return None
    try:
        return t.window(t0, t1)
    except Exception:
        return None


# --- Is the mic just hearing what the Mac played? ---------------------------
# The word check above (_without_mac_audio in main.py) is the ground truth,
# but it needs a whole finished utterance and a Whisper pass, so it lands
# seconds late. This is the cheap version, for decisions that can't wait:
# compare the SHAPE of the two signals -- 10 ms RMS envelopes over the last
# fraction of a second -- searching a lag range for the output path's delay.
# Envelopes rather than waveforms because a Bluetooth and acoustic path
# wrecks the phase but leaves the syllable rhythm intact. Pure numpy on half
# a second of audio: microseconds.
#
# LINEAR envelopes, not dB, and that is the whole ballgame. Measured over
# 2,892 cases built from real speech (2026-09-22): a dB envelope scores an
# unrelated speaker as high as 0.98, because when two people both stop
# talking at the same moment, the two ~50 dB cliffs into silence line up and
# that single shared step carries the whole correlation. Linear treats
# silence as zero, so the loud syllables -- the actual content -- decide it.
# Best accuracy dB 86.1%, linear 99.9%.
HOP = 160                  # 10 ms at 16 kHz
ECHO_WIN_S = 0.6           # most mic tail compared
ECHO_MIN_HOPS = 25         # ...and least (250 ms), or it can't be told
ECHO_MAX_LAG_S = 0.4       # DAC + Bluetooth + acoustic delay searched
ECHO_MIN_DBFS = -55.0      # quieter than this, the Mac played nothing to echo
# The threshold is deliberately tuned to catch ALL echo, because the two
# mistakes cost wildly different amounts. A missed echo latches the cut-in,
# pauses the voice, and PAUSE_MAX_S throws the whole reply away. A false veto
# holds a real cut-in back for ONE more 180 ms run and then re-decides on
# fresh audio. Over those 2,892 cases: 0.88 caught 100% of echo (worst real
# echo scored 0.894, through a 24-tap lowpass and hard clipping at -42 dBFS)
# while deferring 4.5% of genuine cut-in windows -- about 8 ms of added delay
# once you account for the retry. Raising it to 0.94 misses 10% of echo, and
# at ~30 windows per spoken reply that is a self-inflicted cut-off on
# basically every reply. Cheap mistake over expensive one, every time.
ECHO_CORR = 0.88


def _env(x: np.ndarray, n: int) -> np.ndarray | None:
    """Linear RMS envelope: n points, one per HOP samples, ending where
    x ends. See the note above for why this is not in dB."""
    if len(x) < n * HOP:
        return None
    f = x[-n * HOP:].astype(np.float32).reshape(n, HOP)
    return np.sqrt(np.mean(f * f, axis=1)) / 32768.0


def echo_score(mic: np.ndarray, end: float) -> tuple[float, float] | None:
    """How much of `mic` (int16 mono 16 kHz, ending at time.monotonic()
    `end`) is the Mac's own output coming back into the mic.

    Returns (correlation, lag in seconds) -- 1.0 is "identical shape", 0 is
    "nothing to do with each other" -- or None when it genuinely can't be
    told: the tap isn't running, or there's too little mic audio to compare.
    Never blocks, never raises."""
    if not enabled():
        return None
    n = min(len(mic) // HOP, int(ECHO_WIN_S * RATE / HOP))
    if n < ECHO_MIN_HOPS:
        return None
    mic_env = _env(mic, n)
    if mic_env is None:
        return None
    win_s = n * HOP / RATE
    k_max = int(ECHO_MAX_LAG_S * RATE / HOP)
    played = window(end - win_s - ECHO_MAX_LAG_S - 0.05, end)
    if played is None:
        return None
    lvl = 20.0 * np.log10(
        np.sqrt(np.mean(played.astype(np.float32) ** 2)) / 32768 + 1e-9)
    if lvl < ECHO_MIN_DBFS:
        return 0.0, 0.0        # the Mac was silent: whatever that was, it wasn't us
    p_env = _env(played, n + k_max)
    if p_env is None:
        return None
    # every lag at once: row k is the played envelope slid back k hops
    idx = (len(p_env) - n) - np.arange(k_max + 1)[:, None] + np.arange(n)[None, :]
    rows = p_env[idx]
    m = mic_env - mic_env.mean()
    r = rows - rows.mean(axis=1, keepdims=True)
    corr = (r @ m) / (np.sqrt((m * m).sum() * (r * r).sum(axis=1)) + 1e-9)
    k = int(np.argmax(corr))
    return float(corr[k]), k * HOP / RATE
