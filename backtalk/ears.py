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
"""The ears — mic capture with VAD endpointing, transcribed in-process
by faster-whisper. Local, free, no server, no API key.

record_held() is the hold-to-talk capture (the button is the VAD).
Ears.listen_once() is the legacy open-mic mode: blocks until one
complete utterance is heard, then returns its transcript. Endpointing:
an utterance opens after ~120ms of sustained speech, closes after
`silence_ms` of trailing quiet. A `gate` callable can suppress
listening (so the open mic ignores the speakers unless barge-in is on).
"""
import platform
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad

from backtalk import aec, audiowatch, systap
from backtalk.config import CFG
from backtalk.vlog import log
from backtalk import turnend

RATE = 16000
FRAME_MS = 30
FRAME_LEN = RATE * FRAME_MS // 1000  # samples per frame
OPEN_FRAMES = 4        # ~120ms speech to open an utterance
MAX_UTTER_S = 30
# With echo cancellation the mic hears only the person, so speech over the
# voice pauses it at once: after 300ms of UNBROKEN speech while sound is
# actually playing. Measured through the echo-cancelled MacBook mic
# (2026-09-14, webrtcvad aggressiveness 2): a quiet room's longest speech run
# was 150ms (25 of 200 frames flagged), the reactor heartbeat 120ms. A first
# version counted flagged frames across the utterance instead of in a row,
# and room noise paused the voice for nothing.
# 2026-09-21: 10 frames (300ms) -> 6 (180ms), so being cut off feels like
# Claude's or ChatGPT's voice mode. Still clear of the runs measured above
# (150ms room, 120ms heartbeat), and BARGE_MIN_DBFS drops quiet room noise
# before it ever counts. If it starts pausing for nothing, put this back up.
BARGE_FRAMES = 6
# ...and only speech louder than this counts while the voice is audible or
# paused for a possible cut-in. The voice's own echo isn't removed perfectly:
# through the echo-cancelled mic its leftover still looked like speech to the
# VAD for up to 540ms, but quietly (speech-flagged frames median -60 dBFS,
# loudest -41). A quiet room is about -65; the voice itself reached the mic at
# about -39 dBFS without cancellation, and a person talking nearby is louder.
BARGE_MIN_DBFS = -50.0
# How much mic audio the cut-in check compares against what the Mac played
# (20 frames = 600ms; less than that is used when less is on hand).
ECHO_FRAMES = 20

_NONSPEECH = re.compile(r"[\[(][^\])]*[\])]")

# What Whisper writes for room noise, breath, and the tail of the voice's
# own audio. From this machine's log (2026-09-13): "!" 154 times, "you" 49,
# "Thank you." 12, out of 1,238 heard lines, and 24 of 55 cut-off replies
# were cut by lines like these.
_NOISE = {"you", "thank you", "thanks for watching", "uh", "um", "hmm",
          "mm", "ahem", "a", "the", "so", "music", "applause"}


def _is_noise(text: str) -> bool:
    """True for a transcript with no real words in it."""
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    words = [w.strip("'") for w in words if w.strip("'")]
    if not words:
        return True
    return (" ".join(words) in _NOISE or set(words) <= {"music"}
            or _loop_share(words) >= 0.5)


# Whisper's repeat loop on unclear audio: one short phrase over and over.
# From this machine's log (2026-09-16): "Spin number." x40 after a few real
# words, and "Beep" x120, both taken as the person. A person doesn't say the
# same 1-3 words eight times running, so a line mostly made of such a run is
# dropped. "no no no no stop" (4 in a row) is kept.
_LOOP_MIN_REPS = 8


def _loop_share(words: list[str]) -> float:
    """Share of the words that sit in a run of one 1-3 word phrase
    repeated _LOOP_MIN_REPS or more times in a row."""
    best = 0
    for n in (1, 2, 3):
        i = 0
        while i + n <= len(words):
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == words[i:i + n]:
                reps += 1
            if reps >= _LOOP_MIN_REPS:
                best = max(best, reps * n)
            i += reps * n if reps > 1 else 1
    return best / len(words)

_model = None
_model_lock = threading.Lock()
_backend = None          # "mlx" once the GPU path loads, else "faster-whisper"


def _apple_gpu_available() -> bool:
    """Apple Silicon only. CTranslate2, the runtime under faster-whisper,
    has no Metal backend, so on every Mac it transcribes on the CPU while
    the GPU sits idle. mlx-whisper runs the SAME model on the GPU.

    Measured on an M4 Max, small.en, a 6.5s clip, warm: 0.88s on the CPU
    path against 0.12s on the GPU, with a character-identical transcript
    on three of four test clips and a two-comma difference on the fourth.

    Not a second product and not a user-facing choice: same model name
    from the same config key, same text out, one platform finally running
    it properly. Anything that is not an Apple Silicon Mac keeps
    faster-whisper, which already uses CUDA wherever it exists."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper                       # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_repo(model_name: str) -> str:
    """A faster-whisper model name -> its MLX conversion on the Hub."""
    return f"mlx-community/whisper-{model_name}-mlx"


_mic_checked = False


_mic_device_warned = False


def _mic_index():
    """Resolve mic_device (a device NAME) to an index, or None for the default.

    A NAME and never an index, because indices shift every time a device
    connects or disconnects, which is the exact event this setting exists
    to survive. Measured on a real machine: plugging a USB microphone in
    moved the default pair from [-1, 1] to [1, 3], silently changing the
    OUTPUT device too.

    Re-resolved on every stream open rather than cached at startup, for
    the same reason. Exact name wins, then the first case-insensitive
    substring, so a precise name can never be beaten by a loose one.
    """
    global _mic_device_warned
    want = str(CFG.get("mic_device", "") or "").strip()
    if not want:
        return None
    try:
        devices = sd.query_devices()
    except Exception as e:
        log(f"[ears] could not list audio devices ({e}) -- using the "
            f"default mic")
        return None
    ins = [(i, d) for i, d in enumerate(devices)
           if d.get("max_input_channels", 0) > 0]
    for i, d in ins:
        if d["name"] == want:
            _mic_device_warned = False
            return i
    low = want.lower()
    for i, d in ins:
        if low in d["name"].lower():
            _mic_device_warned = False
            return i
    if not _mic_device_warned:          # once per disappearance, not per press
        _mic_device_warned = True
        log(f"[ears] mic_device {want!r} not found -- using the system "
            f"default. Inputs I can see: {[d['name'] for _, d in ins]}")
    return None


def _open_mic():
    """Open the capture stream on the configured mic.

    Degrades to the system default if that device will not open --
    unplugged between the lookup and the open, busy, or refusing the
    sample rate. The mic gets worse; it never goes mute.
    """
    if aec.enabled():
        return aec.InputStream()     # echo-cancelled, same int16 16 kHz mono
    dev = _mic_index()
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=FRAME_LEN)
    try:
        return sd.InputStream(device=dev, **opts)
    except Exception as e:
        audiowatch.device_error()          # a hang in the fallback below isn't a restartable one
        if dev is not None:
            log(f"[ears] could not open mic_device {CFG.get('mic_device')!r} "
                f"({e}) -- using the system default")
            try:
                return sd.InputStream(**opts)
            except Exception:
                pass                   # fall through to the rebuild below
        return _reopen_after_device_change(opts)


def _reopen_after_device_change(opts):
    """Last resort: rebuild the audio system, then open the mic once more.

    PortAudio caches the device list when it initialises, so a device that
    disappears afterwards leaves a stale entry behind. A Bluetooth headset
    flipping between listening and call modes does this every time the mic
    opens, and from then on EVERY capture fails while the voice line looks
    perfectly healthy and simply never hears another word.

    Rebuilding refreshes the list. It also closes every open stream, the
    speaking one included, which is why Mouth._get_out rebuilds a stream
    it finds dead rather than trusting the one it is holding. Do not
    remove that guard without removing this.
    """
    log("[ears] the audio devices changed -- rebuilding and reopening")
    try:
        sd._terminate()
    except Exception:
        pass                           # already down; re-initialising is the point
    sd._initialize()
    return sd.InputStream(**opts)


_shared_mic = None
_open_lock = threading.Lock()
_read_lock = threading.Lock()


def _drop_shared_mic():
    """Forget the shared stream after an error WITHOUT stopping it: a stop is
    the very call that hangs here. The next listen opens a fresh one; the old
    one is released when the process exits."""
    global _shared_mic
    _shared_mic = None


class _WatchedMic:
    """The mic, as ONE long-lived capture stream shared by every listen.

    It used to open, start, stop and close a stream for every utterance.
    Twice on 2026-09-13 (5:08 PM and 10:26 PM) that stop never returned --
    PortAudio's FinishStoppingStream waiting on a CoreAudio lock -- and the
    voice line went deaf. The speaker already keeps one stream for life
    (audio law #1 in mouth.py) and never hung, so the mic does the same.
    Every blocking call still runs under the audio watchdog. Audio buffered
    while nobody was reading is flushed on entry, so a listen starts from
    now. Trade-off: the mic stays open while the voice line runs."""

    def __enter__(self):
        global _shared_mic
        try:
            with _open_lock:
                # aec.enabled() can flip mid-session (see its docstring —
                # the output device changing, e.g. AirPods connecting,
                # re-decides echo cancellation on a cadence). If the shared
                # mic we're holding doesn't match the current decision, it's
                # reading through the wrong path (or a helper that just got
                # torn down) — drop it so the block below opens a fresh one.
                if _shared_mic is not None:
                    want_aec = aec.enabled()
                    have_aec = isinstance(_shared_mic, aec.InputStream)
                    if want_aec != have_aec:
                        _drop_shared_mic()
                        _shared_mic = None
                if _shared_mic is None:
                    with audiowatch.watch("opening the mic"):
                        stream = _open_mic()
                        stream.start()
                    _shared_mic = stream
                self.stream = _shared_mic
            with _read_lock, audiowatch.watch("reading the mic"):
                stale = self.stream.read_available
                if stale > 0:
                    self.stream.read(stale)
        except Exception as e:
            self._failed(e)
            raise
        return self

    def read(self, frames):
        try:
            with _read_lock, audiowatch.watch("reading the mic"):
                return self.stream.read(frames)
        except Exception as e:
            self._failed(e)
            raise

    @staticmethod
    def _failed(exc):
        _drop_shared_mic()
        if type(exc).__name__ == "PortAudioError":
            audiowatch.device_error()

    def __exit__(self, *exc):
        return False            # the stream keeps running for the next listen


_mic_warned = False

# Substrings PortAudio uses when the problem is the DEVICE rather than the
# audio. Matched on the message because the exception TYPE is the same
# PortAudioError whether a device vanished or a stream merely glitched.
_DEVICE_ERROR_HINTS = ("error querying device", "invalid device",
                       "device unavailable", "no default input",
                       "invalid number of channels", "device not found")


def _mic_message(detail: str) -> list[str]:
    """The one explanation, so startup and mid-session say the same thing."""
    return [
        "[ears] NO WORKING MICROPHONE. Nothing can be recorded on this "
        "machine, so the talk key will have nothing to send.",
        f"[ears] the audio system said: {detail}",
        "[ears] plug one in and start the voice line again. If one IS "
        "plugged in, check it is allowed in this system's microphone "
        "privacy settings -- and if you have several, put part of the "
        "one you want in \"mic_device\" in backtalk.json.",
    ]


def explain_audio_failure(exc) -> bool:
    """Turn a device-level audio failure into plain words. Returns True
    when it handled the message, so the caller can skip the raw repr.

    The startup pre-flight cannot cover a microphone that is unplugged or
    dies MID-SESSION, and that person gets the worst version of this:
    no warning at all, and a raw PortAudioError on every single press,
    forever. The key hook keeps working throughout, so it still looks
    like it is listening. This says the same sentences the pre-flight
    would have said, at the moment it becomes true.

    Said in full once, then briefly, because a message repeated on every
    key press stops being information and becomes noise.
    """
    global _mic_warned
    text = str(exc).lower()
    # THE TWO HALVES OF THIS TEST ARE NOT DOING THE SAME JOB. Do not
    # simplify it to one. Measured on Windows: the SAME missing microphone
    # produces "Error querying device -1" when it is absent at startup and
    # "A device ID has been used that is out of range for your system
    # [MME error 2]" when it is unplugged mid-stream. The second matches
    # not one hint below, and was caught only by the type check -- on the
    # very first real test of the case this function exists for. The
    # hints catch device failures raised as something other than a
    # PortAudioError; the type catches PortAudio wording nobody predicted.
    if type(exc).__name__ != "PortAudioError" and \
            not any(h in text for h in _DEVICE_ERROR_HINTS):
        return False
    audiowatch.device_error()
    if _mic_warned:
        log("[ears] still no working microphone.")
        return True
    _mic_warned = True
    for line in _mic_message(f"{type(exc).__name__}: {exc}"):
        log(line)
    return True


def check_microphone() -> bool:
    """Say whether recording is possible at all, BEFORE the greeting.

    Without this the voice line boots on a machine with no microphone,
    warms, speaks its greeting and presents a working push-to-talk
    prompt. The key hook works perfectly throughout, so the user is
    given every impression it is listening -- and the only sign of
    trouble is a raw PortAudioError AFTER they have held the key and
    spoken. It then repeats forever, because holding a key again cannot
    conjure a device.
    """
    global _mic_checked
    if _mic_checked:
        return True
    _mic_checked = True
    try:
        sd.check_input_settings(device=_mic_index(), channels=1,
                                samplerate=RATE, dtype="int16")
        return True
    except Exception as e:
        global _mic_warned
        _mic_warned = True      # said it here; do not repeat on first press
        for line in _mic_message(str(e)):
            log(line)
        return False


def _probe(model):
    """Run a tenth of a second of silence through the real path.

    faster-whisper is lazy: transcribe() returns a generator and does no
    work until it is iterated, so the list() is what actually exercises
    the backend and is not redundant.
    """
    segments, _ = model.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                   language="en")
    list(segments)


def warm():
    """Load the STT model (first call downloads it to the HF cache).
    Called at startup while the greeting plays, so the first real
    utterance doesn't pay the load."""
    global _model, _backend
    check_microphone()
    with _model_lock:
        if _model is None:
            if _apple_gpu_available():
                import mlx_whisper
                repo = _mlx_repo(CFG["stt_model"])
                log(f"[ears] loading {CFG['stt_model']} on the Apple GPU...")
                # This API has no separate load call: the first transcribe
                # pulls and caches the weights. Warm on a beat of silence so
                # the first real utterance does not pay for it.
                mlx_whisper.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                       path_or_hf_repo=repo, language="en",
                                       verbose=None)
                _model, _backend = repo, "mlx"
            else:
                from faster_whisper import WhisperModel
                want = CFG["stt_device"]
                log(f"[ears] loading {CFG['stt_model']} "
                    f"({want}/{CFG['stt_compute']})...")
                _model = WhisperModel(CFG["stt_model"], device=want,
                                      compute_type=CFG["stt_compute"])
                # PROVE the device before the greeting, not at the first
                # spoken sentence. WhisperModel CONSTRUCTS perfectly well
                # against a GPU it cannot actually use: "auto" picks CUDA
                # on any NVIDIA machine, and the CUDA runtime is not
                # loaded until the first inference. So warm-up logged
                # "model ready", startup reported healthy, and a missing
                # cublas DLL only surfaced when the user finally spoke --
                # long after the greeting, in a place they could not
                # connect to a setting. The Apple-GPU branch above has
                # always done this; this one never did.
                try:
                    _probe(_model)
                except Exception as e:
                    if want == "cpu":
                        raise
                    log(f"[ears] {want!r} does not work on this machine "
                        f"({type(e).__name__}: {e}).")
                    log("[ears] falling back to the CPU. Set "
                        "\"stt_device\": \"cpu\" in backtalk.json to skip "
                        "this check in future.")
                    _model = WhisperModel(CFG["stt_model"], device="cpu",
                                          compute_type=CFG["stt_compute"])
                    _probe(_model)
                _backend = "faster-whisper"
            log(f"[ears] model ready ({_backend})")
    return _model


_voiceid_warned = False


def _passes_voice_lock(pcm: np.ndarray) -> bool:
    """Gate an utterance on the enrolled speaker, if voice_lock is on.
    Checked BEFORE transcription so a stranger's speech never reaches
    Whisper at all -- it is dropped exactly like silence. Fails open:
    any problem loading the model or the profile lets the utterance
    through rather than going deaf over a missing dependency."""
    global _voiceid_warned
    if not (CFG.get("voice_lock") or {}).get("enabled"):
        return True
    try:
        from backtalk import voiceid
        if not voiceid.has_profile():
            if not _voiceid_warned:
                _voiceid_warned = True
                log("[ears] voice_lock is enabled but no voiceprint is "
                    "enrolled -- run `uv run python -m backtalk.voiceid` "
                    "to enroll. Letting everything through until then.")
            return True
        passed, score = voiceid.verify(pcm)
        if (CFG.get("voice_lock") or {}).get("shadow"):
            # Shadow mode (2026-09-15): score and log every utterance but
            # block nothing, so the threshold is picked from real scores
            # (the person vs videos and other voices) before voice-lock
            # can lock the person out. The line after this in the log shows
            # what was heard, so each score can be matched to a speaker.
            log(f"[ears] voice_lock shadow: score {score:.2f} "
                f"({'would pass' if passed else 'would reject'})")
            return True
        if not passed:
            log(f"[ears] voice_lock: rejected an utterance (score "
                f"{score:.2f})")
        return passed
    except Exception as e:
        if not _voiceid_warned:
            _voiceid_warned = True
            log(f"[ears] voice_lock check failed, letting speech "
                f"through: {e!r}")
        return True


def _own_voice(frames: list[np.ndarray]) -> tuple[bool, str]:
    """Is the speech heard over the voice actually the VOICE, coming back
    into the mic? Returns (don't cut in, why).

    Only asked with echo cancellation OFF -- with it on, the mic feed has
    the voice taken out of it already. Off is the headphone case, and the
    2026-09-21 rule that headphones are safe because "nothing it says
    reaches the mic" turned out to be wrong: AirPods Pro leak into the
    MacBook Air mic at -15 to -30 dBFS, loud enough to trip the cut-in
    detector on their own. Live on 2026-09-22, 02:53-02:54, that was six
    self-inflicted cut-offs in under two minutes -- each one paused the
    voice, and PAUSE_MAX_S then threw the whole reply away. The word-level
    check in main.py catches the same echo correctly, but only after a
    finished utterance and a Whisper pass, seconds after the reply is gone.
    This is the same question asked in time to matter."""
    if aec.enabled():
        return False, ""
    got = systap.echo_score(np.concatenate(frames[-ECHO_FRAMES:]),
                            time.monotonic())
    if got is None:
        return True, ("no Mac-audio tap, so the voice can't be told from a "
                      "person with echo cancellation off")
    corr, lag = got
    if corr >= systap.ECHO_CORR:
        return True, (f"that's the voice coming back in (envelope r={corr:.2f} "
                      f"at {lag * 1000:.0f} ms)")
    return False, f", r={corr:.2f} vs the Mac"


def transcribe(pcm: np.ndarray) -> str:
    """int16 mono 16kHz -> text. Bracketed non-speech markers that
    whisper emits ([BLANK_AUDIO], [SIGHS], (coughs)...) are stripped;
    if nothing remains, it was silence."""
    model = warm()
    audio = pcm.astype(np.float32) / 32768.0
    lang = "en" if CFG["stt_model"].endswith(".en") else None
    t0 = time.monotonic()
    if _backend == "mlx":
        import mlx_whisper
        text = mlx_whisper.transcribe(audio, path_or_hf_repo=model,
                                      temperature=0.0, language=lang,
                                      verbose=None)["text"].strip()
    else:
        segments, _ = model.transcribe(audio, temperature=0.0, language=lang)
        text = "".join(s.text for s in segments).strip()
    log(f"[timing] stt ({_backend}) {time.monotonic() - t0:.2f}s "
        f"for {len(pcm) / RATE:.1f}s of audio")
    text = _NONSPEECH.sub("", text).strip()
    if _is_noise(text):
        if text:
            log(f"[ears] ignored noise: {text!r}")
        return ""
    return text


class Ears:
    def __init__(self, aggressiveness: int = 2, silence_ms: int = 480):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = silence_ms // FRAME_MS
        self.heard_while_speaking = False
        self.last_turn_p = None       # turnend.p_complete() of the last utterance, None = no opinion
        # True while an utterance is being captured (speech has opened it and it has not closed): the proactive layer reads this
        # so a reminder never starts talking over someone mid-sentence on the open mic. A plain bool, written only here.
        self.hearing = False
        # when the last utterance ended (time.monotonic()) and how long it
        # ran, so what the Mac played at that moment can be compared with it
        self.last_end = 0.0
        self.last_seconds = 0.0

    def listen_once(self, gate=None, timeout_s: float | None = None,
                    abort=None, speaking=None, barge=None,
                    audible=None) -> str | None:
        try:
            return self._listen_once(gate, timeout_s, abort, speaking, barge, audible)
        finally:
            self.hearing = False

    def _listen_once(self, gate=None, timeout_s: float | None = None,
                     abort=None, speaking=None, barge=None,
                     audible=None) -> str | None:
        """Block until one utterance completes; return transcript
        (or None on timeout). An `abort` callable is checked every
        frame; returning True ends the listen and returns None, which
        is how a live switch back to push-to-talk shuts the open mic
        down promptly instead of after one more utterance.

        `speaking` (a callable) is sampled every frame. When most of the
        utterance was captured while it returned True, heard_while_speaking
        is set, so the caller can treat it as heard over the voice (which
        may be the voice itself, coming back through the speakers).

        `barge` (a callable taking True/False), for echo cancellation:
        called with True once BARGE_FRAMES of unbroken speech are heard while
        `audible()` says sound is coming out, so the voice pauses while the
        person is still talking instead of after they finish; called with
        False if that utterance is then thrown away, so it carries on."""
        self.heard_while_speaking = False
        self.last_turn_p = None
        barged = False
        over_run = 0                  # unbroken speech frames while audible
        veto_logged = 0.0             # last time an ignored cut-in was logged
        levels: list[float] = []      # dBFS of this utterance's speech frames
        frames: list[np.ndarray] = []
        ring: list[np.ndarray] = []   # pre-roll so the first syllable survives
        ring_spk: list[bool] = []     # was the voice talking, per ring frame
        spk = 0                       # utterance frames captured over the voice
        speech_run = 0
        silence_run = 0
        speech_total = 0
        in_utterance = False
        elapsed = 0.0

        with _WatchedMic() as stream:
            while True:
                if gate and gate():
                    # the talk key is recording on the shared mic: stay off
                    # it (reading here would steal half its frames)
                    ring.clear()
                    ring_spk.clear()
                    time.sleep(FRAME_MS / 1000)
                    elapsed += FRAME_MS / 1000
                    if abort and abort():
                        return None
                    continue
                block, _ = stream.read(FRAME_LEN)
                elapsed += FRAME_MS / 1000
                if abort and abort():
                    return None
                if timeout_s and elapsed > timeout_s and not in_utterance:
                    return None
                mono = block[:, 0].copy()
                over = bool(speaking and speaking())
                is_speech = self.vad.is_speech(mono.tobytes(), RATE)
                lvl = -120.0
                if barge is not None:
                    # echo cancellation on: quiet "speech" is the leftover of
                    # the voice's own echo or the room, not the person. Gated
                    # always, not just while the voice is audible: live on
                    # 2026-09-14, room noise between Yousif's words (utterance
                    # speech medians -56 to -67 dBFS, his words peaking -11 to
                    # -13) kept utterances open, "Take a look at this." ran to
                    # 7.5 s and replies took 9-10 s to start.
                    lvl = float(20 * np.log10(np.sqrt(np.mean(
                        mono.astype(np.float32) ** 2)) / 32768 + 1e-9))
                    if is_speech and lvl < BARGE_MIN_DBFS:
                        is_speech = False
                if not in_utterance:
                    ring.append(mono)
                    ring_spk.append(over)
                    if len(ring) > 8:
                        ring.pop(0)
                        ring_spk.pop(0)
                    speech_run = speech_run + 1 if is_speech else 0
                    if speech_run >= OPEN_FRAMES:
                        in_utterance = True
                        self.hearing = True
                        frames = ring[:]
                        spk = sum(ring_spk)
                        silence_run = 0
                        over_run = speech_run if (audible and audible()) else 0
                        levels = []
                else:
                    frames.append(mono)
                    spk += over
                    if is_speech:
                        speech_total += 1
                        silence_run = 0
                        if barge is not None:
                            levels.append(lvl)
                        over_run = over_run + 1 if (audible and audible()) else 0
                        if barge and not barged and over_run >= BARGE_FRAMES:
                            mine, why = _own_voice(frames)
                            if mine:
                                # Reset the run instead of pausing: this costs
                                # one check per BARGE_FRAMES rather than one
                                # per frame, and a real cut-in on top of the
                                # echo still lands on the next run, ~180ms on.
                                over_run = 0
                                now = time.monotonic()
                                if now - veto_logged > 2.0:
                                    veto_logged = now
                                    log(f"[ears] cut-in ignored -- {why}")
                            else:
                                barged = True
                                log(f"[ears] cut-in: {over_run * FRAME_MS} ms "
                                    f"of speech over the voice at "
                                    f"{lvl:.0f} dBFS{why}")
                                barge(True)
                    else:
                        silence_run += 1
                        over_run = 0
                    if silence_run >= self.silence_frames or \
                       len(frames) * FRAME_MS / 1000 > MAX_UTTER_S:
                        if speech_total < 8:
                            # <240ms of actual speech: a noise blip, not
                            # a sentence — keep listening
                            in_utterance = False
                            self.hearing = False
                            frames, ring, ring_spk = [], [], []
                            spk = 0
                            speech_run = speech_total = 0
                            continue
                        pcm = np.concatenate(frames)
                        if not _passes_voice_lock(pcm):
                            if barged:
                                barged = False
                                barge(False)
                            in_utterance = False
                            self.hearing = False
                            frames, ring, ring_spk = [], [], []
                            spk = 0
                            speech_run = speech_total = 0
                            continue
                        self.heard_while_speaking = spk * 2 >= len(frames)
                        # is he done, or pausing mid-thought? (turnend.py; None = no opinion)
                        self.last_turn_p = turnend.p_complete(pcm)
                        self.last_end = time.monotonic()
                        self.last_seconds = len(frames) * FRAME_MS / 1000
                        if barge is not None and levels:
                            # calibration for BARGE_MIN_DBFS, from real voices
                            log(f"[ears] heard {len(frames) * FRAME_MS / 1000:.1f}s, "
                                f"speech level median {np.median(levels):.0f} "
                                f"max {max(levels):.0f} dBFS")
                        return transcribe(pcm)


def record_held(is_held, max_s: float = 60.0, min_s: float = 0.25) -> str | None:
    """Hold-to-talk capture: record raw audio while is_held() is True,
    then transcribe. The button is the VAD — no endpointing. Returns
    None for taps shorter than min_s (accidental presses)."""
    frames: list[np.ndarray] = []
    with _WatchedMic() as stream:
        while is_held() and len(frames) * FRAME_MS / 1000 < max_s:
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
        # a small tail so the last word isn't clipped at release
        for _ in range(6):
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
    if len(frames) * FRAME_MS / 1000 < min_s:
        return None
    pcm = np.concatenate(frames)
    if not _passes_voice_lock(pcm):
        return None
    return transcribe(pcm)


if __name__ == "__main__":
    import time
    print("[ears] listening — say something...", flush=True)
    ears = Ears()
    start = time.time()
    while time.time() - start < 30:
        text = ears.listen_once(timeout_s=30 - (time.time() - start))
        if text:
            print(f"[ears] heard: {text!r}", flush=True)
            break
        if text is None:
            print("[ears] timed out with no speech", flush=True)
            break
        print("[ears] (noise/empty — still listening)", flush=True)
