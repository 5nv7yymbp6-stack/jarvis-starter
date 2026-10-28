"""Labeled utterance capture for Stage 0. Interactive. Records ONLY people who
know and have agreed.

    # Yusuf, plain mic (safe to run while Backtalk is up; it stays IDLE unless
    # a prompt contains its wake word, and none of these do):
    python -m backtalk.evals.speaker.capture --label YUSUF --path plain --n 60

    # another person, with their consent (they type the agreement themselves):
    python -m backtalk.evals.speaker.capture --label OTHER_SPEAKER --free 25

    # the AEC path needs the echo-cancel helper, which the running voice line
    # owns. This tool REFUSES to run it while Backtalk is up.
    python -m backtalk.evals.speaker.capture --label YUSUF --path aec --n 60

Stored per sample: label, duration, path, time, session, consent note. Audio is
kept only as a short-lived WAV under the eval data dir (outside the repo) so
several models can be compared on the same samples; delete it afterwards with
    python -m backtalk.evals.speaker.evaluate --purge-audio
Never stored: names, transcripts, or the words spoken."""
import argparse
import itertools
import socket
import sys

import numpy as np

from backtalk.evals.speaker.store import LABELS, PATHS, RATE, Sample, Store, now_iso

FRAME = RATE * 30 // 1000
SHORT = ["what about tomorrow", "and in Tokyo", "yes please", "open YouTube",
         "how much is that", "check the weather", "not right now",
         "okay go on", "what time is it", "read it again", "and then",
         "sounds good", "tell me more", "who is that", "turn it up",
         "what's next", "remind me later", "show me", "try again",
         "that's wrong"]
MEDIUM = ["I was thinking we could order dinner around seven tonight",
          "can you look up the address for that house on Maple Street",
          "what did the last message say about the appointment",
          "let me know when the file has finished downloading",
          "I'm not sure which one I want, read me the options again",
          "the meeting moved to Thursday afternoon, please update it",
          "how long would it take to drive there from here",
          "make a note that I need to call the title company"]
LONG = ["So the plan for the week is to finish the list, make the calls in "
        "the morning, and then review everything with you in the evening",
        "I keep going back and forth on this, but I think the safest option "
        "is to wait until we have the numbers before I decide anything",
        "Yesterday I spoke to a few people about the property and they all "
        "said roughly the same thing, so let's write that down properly",
        "Can you walk me through what happened today from the beginning, "
        "including anything that went wrong and what you did about it"]
BANK = {"short": SHORT, "medium": MEDIUM, "long": LONG}


class Endpointer:
    """Frame-by-frame utterance segmenter that mirrors ears.listen_once:
    opens after `open_frames` speech frames (with an 8-frame pre-roll),
    closes after `close_frames` of silence (600 ms) or `max_frames`. Returns
    the utterance PCM (pre-roll + speech + trailing silence, exactly the shape
    production scores) or None. Utterances with fewer than `min_speech`
    speech frames are discarded as noise."""

    def __init__(self, preroll=8, open_frames=4, close_frames=20,
                 max_frames=400, min_speech=8):
        self.preroll, self.open_frames = preroll, open_frames
        self.close_frames, self.max_frames = close_frames, max_frames
        self.min_speech = min_speech
        self.reset()

    def reset(self):
        self.ring, self.frames = [], []
        self.run = self.silence = self.speech = 0
        self.open = False

    def feed(self, frame: np.ndarray, is_speech: bool):
        if not self.open:
            self.ring.append(frame)
            self.ring = self.ring[-self.preroll:]
            self.run = self.run + 1 if is_speech else 0
            if self.run >= self.open_frames:
                self.open = True
                self.frames = list(self.ring)
                self.silence = 0
                self.speech = self.run
            return None
        self.frames.append(frame)
        if is_speech:
            self.speech += 1
            self.silence = 0
        else:
            self.silence += 1
        if self.silence >= self.close_frames or len(self.frames) >= self.max_frames:
            pcm = np.concatenate(self.frames)
            ok = self.speech >= self.min_speech
            self.reset()
            return pcm if ok else None
        return None


def _backtalk_running() -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8791))
        return False            # we could bind: nobody holds the lock
    except OSError:
        return True
    finally:
        s.close()


def _open_stream(path: str):
    if path == "aec":
        if _backtalk_running():
            raise SystemExit(
                "Backtalk is running and owns the echo-cancel helper, so the "
                "AEC path cannot be captured now. Stop it first, or use "
                "--path plain (production shadow scores can be joined later "
                "with evaluate --join-log).")
        from backtalk import aec
        st = aec.InputStream()
        st.start()
        return st, lambda: st.read(FRAME)[0][:, 0].copy()
    import sounddevice as sd
    from backtalk.config import CFG
    want = str(CFG.get("mic_device") or "").strip().lower()
    dev = None
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and want and want in d["name"].lower():
            dev = i
            break
    st = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        blocksize=FRAME, device=dev)
    st.start()
    return st, lambda: st.read(FRAME)[0][:, 0].copy()


def record_one(read_frame, timeout_s=25.0) -> np.ndarray | None:
    import webrtcvad
    vad = webrtcvad.Vad(2)
    ep = Endpointer()
    for _ in range(int(timeout_s * 1000 / 30)):
        fr = read_frame()
        pcm = ep.feed(fr, vad.is_speech(fr.tobytes(), RATE))
        if pcm is not None:
            return pcm
    return None


def consent_flow() -> str:
    print("\nCONSENT. This records your voice for a speaker-identification "
          "test on this Mac.\n- Only short voice clips and a similarity score "
          "are kept, on this computer, outside the code repository.\n- Nothing "
          "is uploaded. Audio files are deleted after the test.\n- You can stop "
          "any time, and ask for your samples to be deleted.\n")
    if input('The person being recorded types  I AGREE  to consent: ').strip() != "I AGREE":
        raise SystemExit("No consent given. Nothing recorded.")
    who = input("Relationship/initials for the record (not a full name): ").strip()
    return f"agreed {now_iso()} ({who or 'unspecified'})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--path", default="plain", choices=PATHS)
    ap.add_argument("--n", type=int, default=0, help="prompted samples")
    ap.add_argument("--free", type=int, default=0,
                    help="capture the next N natural utterances, no prompts")
    ap.add_argument("--bucket", default="mixed",
                    choices=["short", "medium", "long", "mixed"])
    ap.add_argument("--session", default=None)
    ap.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    if not (a.n or a.free):
        ap.error("give --n or --free")
    consent = consent_flow() if a.label == "OTHER_SPEAKER" else None
    store = Store(a.dir)
    session = a.session or f"{a.label.lower()}-{now_iso()}"
    stream, read = _open_stream(a.path)
    got = 0
    if a.bucket == "mixed":
        order = itertools.cycle(["short", "short", "short", "medium",
                                 "medium", "long"])
    else:
        order = itertools.cycle([a.bucket])
    cycles = {k: itertools.cycle(v) for k, v in BANK.items()}
    total = a.free or a.n
    try:
        while got < total:
            if a.free:
                print(f"[{got + 1}/{total}] speak naturally...", flush=True)
                bucket = "free"
            else:
                bucket = next(order)
                print(f"\n[{got + 1}/{total}] ({bucket}) say:  "
                      f"\"{next(cycles[bucket])}\"")
                cmd = input("Enter to record, r to skip, q to quit: ").strip()
                if cmd == "q":
                    break
                if cmd == "r":
                    continue
            pcm = record_one(read)
            if pcm is None:
                print("  (nothing heard, try again)")
                continue
            s = Sample(store.new_id(a.label), a.label, len(pcm) / RATE,
                       a.path, now_iso(), session, consent, bucket)
            store.add(s, pcm)
            got += 1
            print(f"  saved {s.sample_id}: {s.duration_s:.1f}s")
    finally:
        try:
            stream.stop()
        except Exception:
            pass
    print(f"\n{got} sample(s) saved to {store.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
