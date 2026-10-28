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
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Chunks are synthesized on their own thread, AHEAD of playback, so each one
is ready the moment the one before it ends, and the first is audible while
later ones are still rendering. Playback is cancellable mid-word: shut_up()
and the speaker goes silent within one audio block plus the device buffer
(~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from backtalk import aec, audiodev, audiowatch
from backtalk.config import CFG
from backtalk.vlog import log

KOKORO_RATE = 24000
# A pause for a possible cut-in that is still going after this long is a person
# really talking: the paused reply is dropped. Raised from 4.0 to 8.0 on
# 2026-09-22: ears.py lets a single utterance run up to MAX_UTTER_S (30s), so
# 4.0 was dropping real, still-in-progress replies constantly (10+ times on
# 2026-09-22 alone, logged at 03:33-04:01 and 14:57-14:59) well before the
# person had even finished talking. Capped at 8.0, not higher, to stay clear
# of the audio watchdog's STUCK_AFTER_S (10.0) -- a paused speaker write is a
# silent one, and the watchdog restarts the whole voice line if it sees no
# speaker write for that long.
PAUSE_MAX_S = 8.0
# How often the speaker re-asks CoreAudio which device the Mac is really
# playing out of. A query is ~70 microseconds, but _get_out runs per audio
# chunk, so it is still worth an interval rather than asking every block.
OUT_RECHECK_S = 1.0
# Audio law #2's head-start buffer (see module docstring): how much audio
# is banked before a chunk starts playing, so a slow machine never
# underruns into slow-motion garble. Applies to every chunk EXCEPT the
# first one after silence/a barge-in, which uses FIRST_CHUNK_PREBUFFER_S
# instead -- shorter, because that's the one prebuffer wait Yousif is
# actually sitting in silence for; every later chunk in the same reply
# has his own listening time to render ahead into, so the full 0.75s
# there costs nothing he'd notice. If this turns out to still be tight
# enough to glitch on a slow machine, raise FIRST_CHUNK_PREBUFFER_S back
# toward PREBUFFER_S rather than dropping the guard entirely.
PREBUFFER_S = 0.75
FIRST_CHUNK_PREBUFFER_S = 0.30
EL_RATE = 44100
# Kokoro starts every chunk with about 0.36 s of digital near-silence (13 of 13
# sentences measured 2026-10-03, bm_lewis at speed 1.15: the signal stays under 30
# of 32767 until 350-377 ms, then jumps to speech). On the FIRST chunk of a reply
# that is dead air he waits through before the first word, whatever the device.
# So the first chunk is cut to start LEAD_KEEP_S before the first sample above
# LEAD_THRESH (speech onsets measured >1000 within ~15 ms of crossing 30; the
# threshold sits well above the pad and the pad kept covers a soft consonant).
# Only the first chunk: later chunks keep the pause that gives sentences their
# cadence, and nothing is trimmed from the middle or the end.
LEAD_THRESH = 60
LEAD_KEEP_S = 0.04
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_pipe = None
_pipe_lock = threading.Lock()
_device = None         # PHASE J2: "mps" or "cpu" -- whichever _pipe actually ended up on
_mps_disabled = False  # PHASE J2: set True after any real MPS failure; sticky for this process


def _tts_device_setting() -> str:
    """CFG["tts_device"]: "cpu" (the default), "mps" or "auto" (both mean MPS
    when this machine has it). Anything else is read as "cpu"."""
    v = str(CFG.get("tts_device") or "cpu").strip().lower()
    return v if v in ("cpu", "mps", "auto") else "cpu"


def _pick_device() -> str:
    """The device Kokoro runs on: the CPU unless tts_device asks for MPS
    (Apple Silicon's GPU) AND this machine has it. Never CUDA -- there is no
    CUDA on Apple Silicon, and it's Kokoro's own device auto-detect (when
    device=None) checking ONLY for CUDA that left this always landing on CPU
    before PHASE J2. torch is imported lazily here, same convention as
    kokoro's own lazy import below, so this costs nothing when Kokoro is
    never used (the ElevenLabs-only path).

    WHY THE DEFAULT IS THE CPU (voice-responsiveness phase, 2026-10-03). Time
    to first audio is the part of a reply Yousif waits through in silence, and
    MPS is the slower way to get it. A Metal shader is compiled per distinct
    input SHAPE, i.e. per text length, and nearly every reply is a new length,
    so the compile is paid on almost every first chunk. Measured on this
    machine (M4, bm_lewis at speed 1.15, a NEW sentence each time with idle
    gaps, like real replies, 16 sentences): CPU 0.33 s median / 0.69 s worst to
    first audio, MPS 0.78 s median / 1.09 s worst. In the live log (since the
    MPS switch) the same step measured 1.52 s median and 3.89 s p90, because
    the GPU is also running Whisper. MPS is faster per second of audio once a
    shape is compiled (about 13x realtime against 9x), but only the first
    chunk of a reply is waited on; every later chunk is rendered ahead of
    playback either way. So the GPU stays free for Whisper and the first word
    comes sooner. "mps" / "auto" opt back in."""
    if _mps_disabled or _tts_device_setting() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_pipe(lang: str, device: str):
    from kokoro import KPipeline
    return KPipeline(lang_code=lang, device=device)


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


def warm():
    """Load the Kokoro pipeline (first call downloads the model to the
    HF cache). Called at startup while the greeting text is composed.

    PHASE J2 (2026-09-29): tries MPS first when available, with CPU as the
    automatic, permanent-for-this-process fallback the moment MPS does
    anything wrong -- either failing to initialize, or failing on the
    one-line warm-up synthesis below. That warm-up line exists specifically
    so an MPS problem is caught HERE, quietly, before it can surface on the
    actual greeting. It forces the first Metal shader compile off one
    short, fixed phrase -- not the greeting's own text, and not repeated
    for other lengths. That is deliberate: a real Metal shader is compiled
    per distinct input SHAPE (text length), so this absorbs exactly one
    shape's cold-compile cost, not every shape a real conversation will
    ever use. It does not claim to solve shape-compilation for arbitrary
    later replies -- see the measurements this phase reported."""
    global _pipe, _device, _mps_disabled
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
            # The voice name's first letter IS the language pipeline:
            # a=American English, b=British English, e/f/h/i/j/p/z = the
            # other shipped languages. bm_lewis -> 'b'.
            lang = (CFG["voice"] or "bm_lewis")[0]
            device = _pick_device()
            log(f"[mouth] loading kokoro (lang '{lang}', "
                f"voice {CFG['voice']}, device '{device}')...")
            try:
                pipe = _load_pipe(lang, device)
                # One short line through the real path on any device: it primes
                # the phonemizer as well as MPS's first shader compile, so the
                # first real reply is not the one that pays for either.
                for _ in pipe("Hello.", voice=CFG["voice"], speed=1.0):
                    pass
                _pipe, _device = pipe, device
            except Exception as e:
                if device != "mps":
                    raise   # a CPU failure is a real Kokoro problem, not an MPS one -- don't hide it
                log(f"[mouth] kokoro failed on mps ({e!r}) -- "
                    f"falling back to cpu for this session")
                _mps_disabled = True
                _pipe = _load_pipe(lang, "cpu")
                _device = "cpu"
            log(f"[mouth] voice ready (device: {_device})")
    return _pipe


def trim_lead(pcm: np.ndarray, rate: int, carry: np.ndarray | None = None):
    """(audio from LEAD_KEEP_S before the first real sound, found).

    `pcm` is int16. While nothing has crossed LEAD_THRESH yet, the whole block is
    leading silence: it is dropped and the caller keeps looking in the next one,
    which is how a streaming engine that yields small pieces is handled too.
    `carry` is the tail of the silence already dropped, so the pad before the
    sound is the same length when the sound starts exactly on a piece boundary."""
    if carry is not None and carry.size:
        pcm = np.concatenate([carry, pcm])
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > LEAD_THRESH)
    if not loud.size:
        return pcm[:0], False
    return pcm[max(0, int(loud[0]) - int(rate * LEAD_KEEP_S)):], True


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _stream_kokoro(text: str):
    """One sentence -> int16 PCM chunks at 24kHz, in-process.

    PHASE J2: the warm-up synthesis inside warm() catches most MPS problems
    before any real text reaches this function, but a synthesis-time
    failure is still possible on a shape/text warm-up never exercised
    (a NotImplementedError for some op, an OOM, etc.). If THIS pipe is on
    MPS and synthesis throws, fall back to CPU and retry this same text
    once so the utterance isn't dropped -- and flip _mps_disabled so every
    later call goes straight to CPU instead of failing the same way again."""
    global _pipe, _device, _mps_disabled
    pipe = warm()
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0

    def _run_pipe(p):
        for _, _, audio in p(text, voice=CFG["voice"], speed=speed):
            a = np.asarray(audio, dtype=np.float32)
            if a.size:
                yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)

    try:
        yield from _run_pipe(pipe)
    except Exception as e:
        if _device != "mps" or _mps_disabled:
            raise   # not an MPS problem (or already on the fallback) -- a real synthesis error
        log(f"[mouth] kokoro synthesis failed on mps ({e!r}) -- "
            f"falling back to cpu for the rest of this session")
        _mps_disabled = True
        lang = (CFG["voice"] or "bm_lewis")[0]
        with _pipe_lock:
            _pipe = _load_pipe(lang, "cpu")
            _device = "cpu"
        yield from _run_pipe(_pipe)


def _stream_elevenlabs(text: str, timeout: float, model: str | None = None):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability 0.5, similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text,
                                    "model_id": model or el["model"],
                                    "voice_settings": {
                                        "stability": 0.5,
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    proc.stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _has_arabic(text: str) -> bool:
    """Kokoro's shipped language packs don't include Arabic at all (it
    phonemizes through espeak-ng's Latin-script rules and just spells
    the letters out) — so Arabic text needs ElevenLabs' multilingual
    model even when the normal Kokoro/ElevenLabs voice config is
    English-only."""
    return bool(_ARABIC_RE.search(text))


def synth_stream(text: str, timeout: float = 30.0):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS
    renders. ElevenLabs when configured, Kokoro otherwise — and Kokoro
    as the fallback on ANY ElevenLabs failure. Degrade, never mute.

    Arabic text is a special case: it always routes to ElevenLabs on
    the multilingual model (never eleven_turbo_v2_5, which is
    English-tuned and mangles Arabic), using the configured voice_id if
    ElevenLabs is set up at all — even if the normal `enabled` flag is
    off, since that flag governs the English voice choice, not whether
    Arabic can be spoken at all. Falls back to Kokoro spelling out the
    letters (current broken behavior) only if no ElevenLabs key/voice
    is configured."""
    el = CFG["elevenlabs"]
    if _has_arabic(text) and el.get("voice_id") and _get_elevenlabs_key():
        try:
            for pcm in _stream_elevenlabs(text, timeout,
                                           model="eleven_multilingual_v2"):
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs (arabic) failed ({str(e)[:60]}) — "
                f"falling back to {CFG['voice']}")
    elif _elevenlabs_ready():
        try:
            for pcm in _stream_elevenlabs(text, timeout):
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) — "
                f"falling back to {CFG['voice']}")
    for pcm in _stream_kokoro(text):
        yield KOKORO_RATE, pcm


class _Rendering:
    """One chunk of speech: its text, and the audio the synth thread
    renders into `pcm` as (rate, samples), ending with None."""
    __slots__ = ("gen", "text", "directions", "pcm", "first", "t_enq", "t_synth")

    def __init__(self, gen: int, text: str, directions, first: bool = False):
        self.gen = gen
        self.text = text
        self.directions = directions
        # monotonic: when the chunk was queued, and when its first audio was
        # rendered. For the one [timing] line per reply (see _play_rendering).
        self.t_enq = None
        self.t_synth = None
        self.pcm: queue.Queue = queue.Queue()
        # True only for a chunk queued while the mouth was fully idle: the
        # first thing spoken after silence or a barge-in. Lets playback use
        # a shorter prebuffer for just that one chunk (see _play_rendering).
        self.first = first


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        # (gen, text, directions) waiting for the synth thread.
        self._q: queue.Queue = queue.Queue()
        # _Rendering, in order, handed from the synth thread to playback.
        self._play_q: queue.Queue = queue.Queue()
        self._speaking = threading.Event()
        # shut_up() bumps this; a chunk tagged with an older generation is
        # dropped wherever it is: waiting, rendering, or playing.
        self._gen = 0
        # hold(): something started talking over the voice, so playback is
        # paused in place until it's known to be words (cut) or noise
        # (release, and it carries on).
        self._held = False
        self._hold_token = 0          # which hold a PAUSE_MAX_S timer belongs to
        # True only while audio is actually being written out, not while a
        # reply is queued or the voice model is still loading.
        self._audible = False
        # Chunks said and not yet played out or dropped. The reply has
        # stopped talking only at zero, not in the gap while the next
        # chunk renders.
        self._outstanding = 0
        self._count_lock = threading.Lock()
        # (monotonic start, text) of the chunks played most recently, so
        # the ears can tell the voice's own echo from the person talking.
        self._spoken: deque = deque(maxlen=8)
        # The one persistent output stream (audio law #1).
        # Playback-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        # Which device the open stream is actually playing into, and when
        # that was last checked against CoreAudio. Audio law #1 keeps one
        # stream for the life of the process, so without this the voice
        # stays on whatever device it started on forever (2026-09-22).
        self._out_device: str | None = None
        self._out_checked_at: float = 0.0
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        self._synth = threading.Thread(target=self._synth_run, daemon=True)
        self._synth.start()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def busy(self) -> bool:
        """Anything queued, loading or actually playing -- i.e. the voice
        is not silent. The turn narrator and the check-in answer both read
        this so neither can ever talk over a real reply."""
        return self._outstanding > 0 or self._speaking.is_set()

    def recent_speech(self, window_s: float = 30.0) -> list[str]:
        """The chunks the voice started playing in the last window_s
        seconds, for telling its own echo from the person talking over it."""
        cutoff = time.monotonic() - window_s
        return [text for t, text in list(self._spoken) if t >= cutoff]

    @property
    def generation(self) -> int:
        """How many times the queue has been flushed (shut_up). A proactive line handed over under one value and found under
        another was flushed before it was heard."""
        return self._gen

    def say(self, text: str):
        """Queue text (split to sentences) for speech."""
        for s in split_sentences(text):
            self._enqueue(s, None)

    def say_chunk(self, text: str, directions=None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if text:
            self._enqueue(text, directions or None)

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued."""
        with self._count_lock:
            self._gen += 1
        if aec.enabled():
            aec.flush_now()          # silent now, not after the next block
        for q in (self._q, self._play_q):
            try:
                while True:
                    q.get_nowait()
                    self._finish_one()
            except queue.Empty:
                pass

    @property
    def held(self) -> bool:
        return self._held

    @property
    def audible(self) -> bool:
        """Sound is actually coming out (not just a reply queued or loading)."""
        return self._audible

    def hold(self):
        """Something is talking over the voice: pause it mid-word, keeping
        its place. The caller then cuts it (words: shut_up) or lets it
        carry on (noise: release). Still paused after PAUSE_MAX_S, the reply
        is dropped and the voice stays quiet until release()."""
        self._held = True
        self._hold_token += 1
        if aec.enabled():
            aec.pause_now()
        t = threading.Timer(PAUSE_MAX_S, self._hold_expired,
                            args=(self._hold_token,))
        t.daemon = True
        t.start()

    def _hold_expired(self, token: int):
        if not self._held or token != self._hold_token:
            return
        log(f"[turn] still talking after {PAUSE_MAX_S:.0f}s -- reply dropped")
        self.shut_up()
        if aec.enabled():
            aec.pause_now()          # the flush unpaused it; stay quiet for them

    def release(self, resume: bool = True):
        self._held = False
        self._hold_token += 1
        if resume and aec.enabled():
            aec.resume_now()

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until the queue is drained and playback finished."""
        import time
        deadline = None if timeout is None else time.time() + timeout
        while self._outstanding > 0 or self._speaking.is_set():
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _enqueue(self, text: str, directions):
        with self._count_lock:
            # Fully idle right before this chunk joins the queue: it's the
            # first thing that will play after silence or a barge-in, so it
            # gets tagged for the shorter prebuffer in _play_rendering.
            is_first = self._outstanding == 0
            self._outstanding += 1
            self._q.put((self._gen, text, directions, is_first, time.monotonic()))

    def _finish_one(self):
        """One chunk is over: played, cut, or dropped."""
        from backtalk import signals
        with self._count_lock:
            self._outstanding = max(0, self._outstanding - 1)
            last = self._outstanding == 0
        if last and self._speaking.is_set():
            self._speaking.clear()
            # The reply has genuinely stopped talking, as opposed to
            # the gap between two chunks of the same reply.
            signals.reply_done()
            self.ducker.speech_end()
            signals.set_state("idle")

    def _synth_run(self):
        """Render chunks AHEAD of playback, so the next one is ready the
        moment the current one ends. Playback used to render each chunk
        only after the previous one finished playing, which put about
        1.3 s of silence before every chunk after the first (measured
        2026-09-13: a two-sentence chunk on an M4)."""
        while True:
            gen, text, directions, is_first, t_enq = self._q.get()
            if not text or gen != self._gen:
                self._finish_one()
                continue
            r = _Rendering(gen, text, directions, is_first)
            r.t_enq = t_enq
            self._play_q.put(r)
            lead = is_first               # only the chunk he is waiting for loses its lead-in
            dropped = []                  # lead-in held back, in case nothing ever follows it
            carry = None                  # its last LEAD_KEEP_S, to pad the sound when it comes
            try:
                for rate, pcm in synth_stream(text):
                    if r.gen != self._gen:
                        break
                    if lead:
                        whole = pcm
                        pcm, found = trim_lead(whole, rate, carry)
                        lead = not found
                        if lead:
                            dropped.append((rate, whole))
                            keep = int(rate * LEAD_KEEP_S)
                            joined = whole if carry is None else np.concatenate([carry, whole])
                            carry = joined[len(joined) - min(keep, len(joined)):]
                            continue
                    if r.t_synth is None:
                        r.t_synth = time.monotonic()
                    r.pcm.put((rate, pcm))
                if lead and r.gen == self._gen:
                    # A chunk with no sound above the threshold anywhere is not
                    # speech with a lead-in; play it exactly as rendered.
                    for item in dropped:
                        r.pcm.put(item)
            except Exception as e:
                log(f"[mouth] synth error: {e}")
            finally:
                r.pcm.put(None)

    def _run(self):
        from backtalk import signals
        while True:
            r = self._play_q.get()
            if r.gen != self._gen:
                self._finish_one()
                continue
            self._speaking.set()
            self.ducker.speech_start()
            signals.static_stop()     # thinking sound dies when speech starts
            signals.set_state("speaking")
            self._spoken.append((time.monotonic(), r.text))
            try:
                self._play_rendering(r)
            except Exception as e:
                log(f"[mouth] play error: {e}")
            finally:
                self._audible = False
                self._finish_one()

    def _next_pcm(self, r: _Rendering):
        """The next (rate, pcm) of a chunk, or None when it has ended or
        shut_up() dropped it. Polls, so a stop never waits on a render."""
        while True:
            try:
                return r.pcm.get(timeout=0.05)
            except queue.Empty:
                if r.gen != self._gen:
                    return None

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened only when the
        sample rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback:
        rare, costs at most one blip on the switch) or when aec.enabled()
        has flipped since the stream we're holding was opened (the
        output device changed underneath us -- see aec.enabled()'s
        docstring). Without this check, a flip from on to off would keep
        writing into the old aec.OutputStream, whose helper enabled()
        just told to tear itself down -- silence, not a clean handoff."""
        want_aec = aec.enabled()
        have_aec = isinstance(self._out, aec.OutputStream)
        if self._out is not None and want_aec != have_aec:
            self._drop_out()
        if want_aec:
            # the echo-cancelled speaker (one helper process for life)
            if not isinstance(self._out, aec.OutputStream) or self._out_rate != rate:
                with audiowatch.watch("starting the speaker"):
                    out = aec.OutputStream(rate)
                    out.start()
                self._out, self._out_rate = out, rate
            return self._out
        # DID THE OUTPUT DEVICE CHANGE UNDER US? Audio law #1 keeps one
        # stream for the life of the process, and the only rebuild triggers
        # used to be a rate change, an aec flip, or the stream dying. None
        # of those fire when Yousif puts AirPods in while echo cancellation
        # is already off -- so the voice went on playing into the built-in
        # speaker while he was wearing headphones (2026-09-22).
        #
        # Asked of CoreAudio, never of PortAudio, whose idea of "the
        # default output" is frozen at startup and is not repaired even by
        # a terminate()/initialize(). See audiodev.
        if self._out is not None:
            now = time.monotonic()
            if now - self._out_checked_at >= OUT_RECHECK_S:
                self._out_checked_at = now
                real = audiodev.default_output_name()
                if real and self._out_device and real != self._out_device:
                    log(f"[mouth] output device changed: "
                        f"{self._out_device!r} -> {real!r} — "
                        f"moving the voice onto it")
                    self._drop_out()
        if self._out is not None and self._out_rate == rate:
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    with audiowatch.watch("starting the speaker"):
                        self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        self._drop_out()
        # Open against an EXPLICIT device index resolved from CoreAudio's
        # name. Passing no device lets PortAudio pick its frozen default,
        # which is the whole bug. If the name isn't in PortAudio's list at
        # all it is a device that appeared after startup, so refresh the
        # list once and look again.
        want = audiodev.default_output_name()
        idx = audiodev.output_index(want) if want else None
        if want and idx is None:
            audiodev.refresh()
            idx = audiodev.output_index(want)
        with audiowatch.watch("opening the speaker"):
            self._out = sd.OutputStream(samplerate=rate, channels=1,
                                        dtype="int16", device=idx)
            self._out_rate = rate
            self._out_device = want or None
            self._out_checked_at = time.monotonic()
            self._out.start()
        log(f"[mouth] speaking through {want or 'the default device'}"
            + (f" (device {idx})" if idx is not None else " (PortAudio default)"))
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        if isinstance(self._out, aec.OutputStream):
            try:
                self._out.flush()    # the helper drops what's queued: no tail
            except Exception:
                self._drop_out()
            return
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            with audiowatch.watch("padding the speaker"):
                for _ in range(3):
                    self._out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                with audiowatch.watch("closing the speaker"):
                    self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None
        self._out_device = None

    def _play_rendering(self, r: _Rendering, block: int = 2205,
                        prebuffer_s: float | None = None):
        """Play one chunk from the audio the synth thread renders into
        r.pcm, with the head-start buffer (audio law #2). shut_up() reacts
        within one block. The sample rate comes from whichever engine
        actually answered."""
        from backtalk import signals
        if prebuffer_s is None:
            prebuffer_s = FIRST_CHUNK_PREBUFFER_S if r.first else PREBUFFER_S
        head: list = []
        banked = 0
        rate = None
        ended = False
        while True:
            item = self._next_pcm(r)
            if item is None:
                ended = True
                break
            rate, pcm = item
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None or r.gen != self._gen:
            return
        try:
            out = self._get_out(rate)
            self._audible = True
            if r.first and r.t_enq is not None:
                # The reply's first audio, from the moment its text was queued:
                # the TTS startup he waits through. One line per reply, so it
                # can be measured in the live log from now on.
                now = time.monotonic()
                synth = (r.t_synth - r.t_enq) if r.t_synth is not None else 0.0
                log(f"[timing] tts first chunk: audio {now - r.t_enq:.2f}s after "
                    f"queued (synth {synth:.2f}s, {_device or 'engine'}, "
                    f"{len(r.text)} chars)")
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            if r.directions:
                signals.direction(r.directions)

            def _write(pcm):
                for i in range(0, len(pcm), block):
                    if r.gen != self._gen:
                        return False
                    with audiowatch.watch("playing audio"):
                        out.write(pcm[i:i + block])
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if r.gen != self._gen:
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
            while not ended:
                item = self._next_pcm(r)
                if item is None:
                    if r.gen != self._gen:
                        self._cut()
                    return
                if not _write(item[1]):
                    self._cut()
                    return
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
