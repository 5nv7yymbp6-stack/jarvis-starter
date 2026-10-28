"""Labeled sample store. Metadata is JSONL; audio, if kept, is a short-lived
WAV that purge_audio() deletes. Lives outside the repo so it can never be
committed. No sample without a label; no OTHER_SPEAKER sample without a
recorded consent."""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RATE = 16000
LABELS = ("YUSUF", "OTHER_SPEAKER", "TV_RECORDED", "SYNTHETIC_TTS")
PATHS = ("plain", "aec")


def default_dir() -> Path:
    return Path(os.environ.get("BACKTALK_SPEAKER_EVAL_DIR")
                or Path.home() / ".backtalk-speaker-eval")


@dataclass
class Sample:
    sample_id: str
    label: str
    duration_s: float
    path: str                  # "plain" | "aec"
    timestamp: str             # local ISO time of capture END
    session: str
    consent: str | None = None  # required for OTHER_SPEAKER
    note: str = ""             # never a name or transcript
    extra: dict = field(default_factory=dict)


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else default_dir()
        self.audio_dir = self.root / "audio"
        self.meta = self.root / "samples.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    @staticmethod
    def validate(s: Sample) -> None:
        if s.label not in LABELS:
            raise ValueError(f"unknown label {s.label!r}; use one of {LABELS}")
        if s.path not in PATHS:
            raise ValueError(f"unknown path {s.path!r}; use one of {PATHS}")
        if s.label == "OTHER_SPEAKER" and not (s.consent or "").strip():
            raise ValueError("OTHER_SPEAKER samples need a recorded consent")
        if s.duration_s <= 0:
            raise ValueError("duration must be positive")

    def new_id(self, label: str) -> str:
        n = len(self.load()) + 1
        return f"{label[:2].lower()}{n:04d}"

    def add(self, s: Sample, pcm: np.ndarray | None = None) -> None:
        self.validate(s)
        if pcm is not None:
            import soundfile as sf
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            sf.write(self.audio_dir / f"{s.sample_id}.wav", pcm, RATE,
                     subtype="PCM_16")
        with open(self.meta, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s)) + "\n")

    def load(self) -> list[Sample]:
        if not self.meta.exists():
            return []
        out = []
        for line in self.meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Sample(**json.loads(line)))
        return out

    def audio(self, sample_id: str) -> np.ndarray | None:
        p = self.audio_dir / f"{sample_id}.wav"
        if not p.exists():
            return None
        import soundfile as sf
        data, _ = sf.read(p, dtype="int16")
        return data

    def purge_audio(self) -> int:
        """Delete every kept WAV. Metadata and embeddings stay."""
        n = 0
        if self.audio_dir.exists():
            for p in self.audio_dir.glob("*.wav"):
                p.unlink()
                n += 1
        return n


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
