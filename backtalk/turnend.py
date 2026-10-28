"""Is he done talking? An audio end-of-turn check for the open mic.

The open mic ends an utterance after silence_ms (600) of quiet. Yousif pauses
mid-thought for longer than that all the time (median ~1.3 s), and Whisper puts
a full stop on most fragments, so the transcript cannot tell a pause from an
ending (2026-10-25: a text-only check caught 6 of 43 real cuts). This module
listens to HOW the utterance ended -- intonation, rhythm, a trailing "and..." --
with Pipecat's Smart Turn v3.2 (BSD-2, Whisper-tiny encoder + linear head,
int8 ONNX, ~8.7 MB, sha256 2bb02631...7e4f, from huggingface.co/pipecat-ai/
smart-turn-v3) and returns the probability that the turn is COMPLETE.

main.py uses it only to decide whether to WAIT for more before sending: a low
probability holds the utterance (the same pre-send StitchHold the text check
uses). It never drops or rewrites anything. Every failure returns None, which
means "no opinion": the voice line behaves exactly as it did without it.
"""
import os
import threading
import time

import numpy as np

from backtalk.config import CFG
from backtalk.vlog import log

RATE = 16000
WINDOW_S = 8                       # the model sees the last 8 s, left-padded
KEEP_TAIL_SILENCE_S = 0.2          # trim the endpoint's trailing quiet to what the model was trained on
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "smart-turn", "smart-turn-v3.2-cpu.onnx")

_lock = threading.Lock()
_state = {"session": None, "fx": None, "failed": False}


def _cfg():
    c = CFG.get("turn_end") or {}
    return {"enabled": bool(c.get("enabled", True)),
            "threshold": float(c.get("threshold", 0.5))}


def enabled() -> bool:
    return _cfg()["enabled"] and not _state["failed"]


def threshold() -> float:
    return _cfg()["threshold"]


def _load(block=True):
    """Build the ONNX session and feature extractor once. Never raises. With
    block=False a load already under way elsewhere (warm() at startup) is not
    waited for: the caller gets False and carries on with no opinion, so his
    first sentence after launch is never held up by a model still loading."""
    if _state["session"] is not None or _state["failed"]:
        return _state["session"] is not None
    if not _lock.acquire(blocking=block):
        return False
    try:
        return _load_locked()
    finally:
        _lock.release()


def _load_locked():
    if _state["session"] is not None or _state["failed"]:
        return _state["session"] is not None
    try:
        t0 = time.monotonic()
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _state["session"] = ort.InferenceSession(MODEL_PATH, sess_options=so,
                                                 providers=["CPUExecutionProvider"])
        _state["fx"] = WhisperFeatureExtractor(chunk_length=WINDOW_S)
        log(f"[turnend] smart-turn v3.2 loaded in {time.monotonic() - t0:.2f}s")
        return True
    except Exception as e:
        _state["failed"] = True
        log(f"[turnend] could not load the end-of-turn model, carrying on without it: {e!r}")
        return False


def warm():
    """Load and run once on silence so the first real utterance pays nothing."""
    if _cfg()["enabled"] and _load():
        p_complete(np.zeros(RATE, dtype=np.int16), trim=False)


def _prepare(pcm, trim=True):
    a = np.asarray(pcm)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    else:
        a = a.astype(np.float32)
    a = a.reshape(-1)
    if trim and a.size:
        # drop the endpoint's trailing quiet beyond KEEP_TAIL_SILENCE_S
        frame = RATE * 30 // 1000
        n = a.size // frame
        if n:
            rms = np.sqrt(np.mean(a[:n * frame].reshape(n, frame) ** 2, axis=1))
            loud = np.nonzero(rms > max(1e-4, rms.max() * 0.05))[0]
            if loud.size:
                end = min(a.size, (loud[-1] + 1) * frame + int(KEEP_TAIL_SILENCE_S * RATE))
                a = a[:end]
    max_n = WINDOW_S * RATE
    if a.size > max_n:
        a = a[-max_n:]
    elif a.size < max_n:
        a = np.pad(a, (max_n - a.size, 0))
    return a


def p_complete(pcm, trim=True):
    """Probability (0..1) that the speaker has FINISHED, or None for no opinion.
    pcm: 16 kHz mono, int16 or float. Never raises."""
    if not _cfg()["enabled"] or not _load(block=False):
        return None
    try:
        a = _prepare(pcm, trim)
        feats = _state["fx"](a, sampling_rate=RATE, return_tensors="np", padding="max_length",
                             max_length=WINDOW_S * RATE, truncation=True, do_normalize=True)
        x = feats.input_features.squeeze(0).astype(np.float32)[None, ...]
        out = _state["session"].run(None, {"input_features": x})
        return float(np.asarray(out[0]).reshape(-1)[0])
    except Exception as e:
        log(f"[turnend] check failed, no opinion: {e!r}")
        return None


def incomplete(p) -> bool:
    """True only when the model is confident he is NOT done."""
    return p is not None and p < threshold()
