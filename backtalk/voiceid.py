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
"""voice-lock — speaker verification on top of the ears.

Optional, off by default (CFG["voice_lock"]["enabled"]). When on, every
utterance is checked against an enrolled voiceprint BEFORE it reaches
Whisper: close enough, it proceeds like normal; not a match, it is
dropped silently, same as if nobody had spoken.

Built on Resemblyzer (local, free, no API key) -- the same d-vector
speaker-embedding approach used for speaker diarization. One profile
per install: PROFILE_PATH. Run `python -m backtalk.voiceid` to enroll
or re-enroll.
"""
import sys
import threading

import numpy as np

from backtalk.config import CFG, REPO
from backtalk.vlog import log

PROFILE_PATH = REPO / "voiceprint.npy"
RATE = 16000

_encoder = None
_encoder_lock = threading.Lock()
_profile = None
_profile_loaded = False


def _load_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                log("[voiceid] loading the speaker-embedding model...")
                from resemblyzer import VoiceEncoder
                _encoder = VoiceEncoder()
    return _encoder


def _load_profile():
    global _profile, _profile_loaded
    if not _profile_loaded:
        _profile_loaded = True
        if PROFILE_PATH.exists():
            _profile = np.load(PROFILE_PATH)
    return _profile


def has_profile() -> bool:
    return PROFILE_PATH.exists()


def _embed(pcm: np.ndarray) -> np.ndarray:
    """int16 mono 16kHz -> a unit-length speaker embedding."""
    from resemblyzer import preprocess_wav
    audio = pcm.astype(np.float32) / 32768.0
    wav = preprocess_wav(audio, source_sr=RATE)
    return _load_encoder().embed_utterance(wav)


def enroll_from_clips(clips: list[np.ndarray]) -> None:
    """Average the embeddings of several enrollment clips into one
    profile and save it. Averaging over a few utterances (not just one)
    is what keeps the profile from overfitting a single recording's
    room tone and mic gain."""
    global _profile, _profile_loaded
    embeds = [_embed(c) for c in clips]
    profile = np.mean(embeds, axis=0)
    profile = profile / np.linalg.norm(profile)
    np.save(PROFILE_PATH, profile)
    _profile, _profile_loaded = profile, True
    log(f"[voiceid] enrolled from {len(clips)} clips -> {PROFILE_PATH}")


def verify(pcm: np.ndarray) -> tuple[bool, float]:
    """Does this utterance match the enrolled voice? Returns (passed,
    score). No profile enrolled yet -> passes everything (fails open,
    never locks someone out of a feature they never set up)."""
    profile = _load_profile()
    if profile is None:
        return True, 1.0
    try:
        e = _embed(pcm)
    except Exception as ex:
        # Too short / silent / malformed clip: let it through rather
        # than silently eating an utterance over a preprocessing quirk.
        log(f"[voiceid] embedding failed, letting it through: {ex!r}")
        return True, 0.0
    score = float(np.dot(e, profile))
    threshold = float((CFG.get("voice_lock") or {}).get("threshold", 0.75))
    return score >= threshold, score


def _record_clip(seconds: float) -> np.ndarray:
    """Record a fixed-length clip straight from the mic, independent of
    the VAD-based capture paths in ears.py (enrollment wants a known
    length, not endpointing)."""
    import sounddevice as sd
    frames = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1,
                    dtype="int16")
    sd.wait()
    return frames[:, 0].copy()


def enroll_interactive(n_clips: int = 3, seconds: float = 5.0) -> None:
    """CLI enrollment: record a few clips of the person talking and
    save the averaged profile. `python -m backtalk.voiceid`."""
    print(f"[voiceid] Enrolling a voiceprint -- {n_clips} clips, "
          f"{seconds:.0f}s each. Talk naturally in each one (read a "
          f"sentence, count, whatever -- the words don't matter, only "
          f"the voice).")
    clips = []
    for i in range(n_clips):
        input(f"[voiceid] Clip {i + 1}/{n_clips}: press Enter, then "
              f"start talking immediately...")
        print(f"[voiceid] recording for {seconds:.0f}s...")
        clips.append(_record_clip(seconds))
        print("[voiceid] got it.")
    enroll_from_clips(clips)
    print(f"[voiceid] Done. Voice lock will use this profile once "
          f"\"voice_lock\": {{\"enabled\": true}} is set in "
          f"backtalk.json.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    enroll_interactive(n, secs)
