"""turnend.py: the audio end-of-turn model (Smart Turn v3.2). Runs the REAL model on synthetic audio and checks
that every failure is "no opinion" (None), never an exception and never a delay."""
import unittest
from unittest import mock

import numpy as np

from backtalk import turnend as T


class RealModel(unittest.TestCase):
    def test_the_model_file_is_the_verified_one(self):
        import hashlib
        with open(T.MODEL_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(),
                             "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f")

    def test_it_returns_a_probability(self):
        t = np.arange(16000 * 2) / 16000
        tone = (np.sin(2 * np.pi * 180 * t) * 8000).astype(np.int16)
        p = T.p_complete(np.concatenate([tone, np.zeros(9600, np.int16)]))
        self.assertIsNotNone(p)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_long_and_float_input(self):
        p = T.p_complete(np.random.default_rng(0).normal(0, 0.05, 16000 * 12).astype(np.float32))
        self.assertTrue(p is None or 0.0 <= p <= 1.0)


class FailsOpen(unittest.TestCase):
    def test_a_load_still_running_elsewhere_is_no_opinion_not_a_wait(self):
        import time
        with mock.patch.dict(T._state, {"session": None, "failed": False}):
            T._lock.acquire()
            try:
                t0 = time.monotonic()
                self.assertIsNone(T.p_complete(np.zeros(16000, np.int16)))
                self.assertLess(time.monotonic() - t0, 0.5)
            finally:
                T._lock.release()

    def test_disabled_by_config_is_no_opinion(self):
        with mock.patch.dict(T.CFG, {"turn_end": {"enabled": False}}):
            self.assertIsNone(T.p_complete(np.zeros(16000, np.int16)))

    def test_a_crash_inside_is_no_opinion(self):
        with mock.patch.object(T, "_prepare", side_effect=RuntimeError("boom")):
            self.assertIsNone(T.p_complete(np.zeros(16000, np.int16)))

    def test_incomplete_needs_a_confident_low_score(self):
        self.assertFalse(T.incomplete(None))
        self.assertFalse(T.incomplete(0.9))
        self.assertTrue(T.incomplete(0.1))

    def test_trailing_silence_is_trimmed_to_the_trained_tail(self):
        a = np.concatenate([np.full(16000, 3000, np.int16), np.zeros(16000, np.int16)])
        out = T._prepare(a)
        self.assertEqual(out.size, T.WINDOW_S * T.RATE)
        nz = np.nonzero(out)[0]
        tail = (out.size - 1 - nz[-1]) / T.RATE
        self.assertLessEqual(tail, T.KEEP_TAIL_SILENCE_S + 0.05)
