"""Tests for the Stage 0 evaluation tooling ONLY (backtalk/evals/speaker).
Nothing here touches the addressee gate or the voice
line."""
import os
import tempfile
import time
import unittest

import numpy as np

from backtalk.evals.speaker import enroll, metrics
from backtalk.evals.speaker.capture import Endpointer
from backtalk.evals.speaker.evaluate import analyze, join_production_scores
from backtalk.evals.speaker.store import Sample, Store


class MetricsTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(metrics.bucket(0.5), "<2s")
        self.assertEqual(metrics.bucket(1.99), "<2s")
        self.assertEqual(metrics.bucket(2.0), "2-5s")
        self.assertEqual(metrics.bucket(4.99), "2-5s")
        self.assertEqual(metrics.bucket(5.0), ">=5s")

    def test_far_frr(self):
        far, frr = metrics.far_frr([0.9, 0.8, 0.4], [0.7, 0.3, 0.2], 0.75)
        self.assertAlmostEqual(frr, 1 / 3)     # 0.4 rejected
        self.assertAlmostEqual(far, 0.0)       # none of the impostors >= .75
        far, _ = metrics.far_frr([0.9], [0.8, 0.3], 0.75)
        self.assertAlmostEqual(far, 0.5)

    def test_empty_sides_give_none_not_invented_numbers(self):
        far, frr = metrics.far_frr([], [0.5], 0.5)
        self.assertIsNone(frr)
        self.assertIsNone(metrics.eer([], [0.5]))
        self.assertIsNone(metrics.overlap([0.5], []))
        self.assertIsNone(metrics.d_prime([0.5], [0.4]))   # n<2

    def test_eer_zero_for_perfect_separation(self):
        e, _ = metrics.eer([0.9, 0.85, 0.95], [0.1, 0.2, 0.3])
        self.assertEqual(e, 0.0)

    def test_eer_half_for_identical_distributions(self):
        s = [0.1, 0.3, 0.5, 0.7, 0.9]
        e, _ = metrics.eer(s, s)
        self.assertGreaterEqual(e, 0.3)

    def test_overlap_and_dprime_order_with_separation(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(0.8, 0.05, 200)
        far = rng.normal(0.3, 0.05, 200)
        near = rng.normal(0.75, 0.05, 200)
        self.assertLess(metrics.overlap(pos, far), metrics.overlap(pos, near))
        self.assertGreater(metrics.d_prime(pos, far), metrics.d_prime(pos, near))

    def test_threshold_table_and_roc_monotonic(self):
        pos, neg = [0.9, 0.7, 0.6], [0.5, 0.4, 0.65]
        tab = metrics.threshold_table(pos, neg, [0.3, 0.6, 0.95])
        self.assertEqual(tab[0]["frr"], 0.0)
        self.assertEqual(tab[2]["far"], 0.0)
        fars = [p[1] for p in metrics.roc(pos, neg)]
        # thresholds ascend, so FAR can only fall and FRR can only rise
        self.assertTrue(all(a >= b for a, b in zip(fars, fars[1:])))
        frrs = [p[2] for p in metrics.roc(pos, neg)]
        self.assertTrue(all(a <= b for a, b in zip(frrs, frrs[1:])))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.store = Store(self.d)

    def s(self, label="YUSUF", **kw):
        base = dict(sample_id="x1", label=label, duration_s=1.5, path="plain",
                    timestamp="2026-09-29T20:00:00", session="t")
        base.update(kw)
        return Sample(**base)

    def test_other_speaker_requires_consent(self):
        with self.assertRaises(ValueError):
            self.store.add(self.s("OTHER_SPEAKER"))
        self.store.add(self.s("OTHER_SPEAKER", consent="agreed (mother)"))
        self.assertEqual(len(self.store.load()), 1)

    def test_rejects_bad_label_path_duration(self):
        for kw in ({"label": "MOM"}, {"path": "usb"}, {"duration_s": 0}):
            with self.assertRaises(ValueError):
                self.store.add(self.s(**kw))

    def test_audio_is_short_lived(self):
        pcm = (np.random.randn(16000) * 1000).astype(np.int16)
        self.store.add(self.s(), pcm)
        self.assertIsNotNone(self.store.audio("x1"))
        self.assertEqual(self.store.purge_audio(), 1)
        self.assertIsNone(self.store.audio("x1"))
        self.assertEqual(len(self.store.load()), 1)     # metadata survives

    def test_no_audio_written_when_none_given(self):
        self.store.add(self.s())
        self.assertFalse(self.store.audio_dir.exists()
                         and any(self.store.audio_dir.iterdir()))

    def test_default_dir_is_outside_the_repo(self):
        from backtalk.evals.speaker.store import default_dir
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertFalse(str(default_dir()).startswith(repo))

    def test_metadata_holds_no_transcript_field(self):
        fields = set(Sample.__dataclass_fields__)
        self.assertFalse(fields & {"transcript", "text", "words", "name"})


class EnrollTests(unittest.TestCase):
    def test_centroid_is_unit_and_scores_members_high(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=64)
        embs = [base + rng.normal(scale=0.2, size=64) for _ in range(8)]
        c = enroll.centroid(embs)
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=6)
        self.assertGreater(enroll.score_centroid(embs[0], c), 0.8)
        self.assertLess(enroll.score_centroid(rng.normal(size=64), c), 0.5)

    def test_multi_embedding_uses_top_k(self):
        a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        e = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(enroll.score_multi(e, [a, b], k=1), 1.0)
        self.assertAlmostEqual(enroll.score_multi(e, [a, b], k=2), 0.5)

    def test_blocks_are_contiguous_disjoint_and_cover(self):
        b = enroll.blocks(23, 5)
        flat = [i for blk in b for i in blk]
        self.assertEqual(flat, list(range(23)))
        self.assertTrue(all(blk == list(range(blk[0], blk[-1] + 1)) for blk in b))
        self.assertEqual(len(enroll.blocks(3, 10)), 3)   # never empty blocks


def _samples_and_embeds(n_pos=30, n_neg=15, sep=1.0, seed=3):
    rng = np.random.default_rng(seed)
    yus_c, oth_c = rng.normal(size=32), rng.normal(size=32)
    samples, embeds = [], {}
    for i in range(n_pos):
        sid = f"y{i:03d}"
        dur = [1.0, 3.0, 6.0][i % 3]
        samples.append(Sample(sid, "YUSUF", dur, "plain" if i % 2 else "aec",
                              "2026-09-29T20:00:00", "s"))
        embeds[sid] = yus_c + rng.normal(scale=0.4 / sep, size=32)
    for i in range(n_neg):
        sid = f"o{i:03d}"
        samples.append(Sample(sid, "OTHER_SPEAKER", [1.0, 3.0, 6.0][i % 3],
                              "plain", "2026-09-29T20:10:00", "s",
                              consent="agreed (test)"))
        embeds[sid] = oth_c + rng.normal(scale=0.4 / sep, size=32)
    return samples, embeds, enroll.centroid([embeds[f"y{i:03d}"] for i in range(min(5, n_pos))])


class AnalyzeTests(unittest.TestCase):
    def test_separable_data_gives_low_eer_everywhere(self):
        samples, embeds, prof = _samples_and_embeds(sep=3.0)
        r = analyze(samples, embeds, existing_profile=prof, k_folds=5)
        for key in ("existing_profile", "cv_centroid", "cv_multi"):
            blk = r[key]["OTHER_SPEAKER"]["all"]
            self.assertLess(blk["eer"], 0.1, key)
            self.assertEqual(blk["pos"]["n"], 30)
        self.assertIn("<2s", r["cv_centroid"]["OTHER_SPEAKER"])
        self.assertIn(">=5s", r["cv_centroid"]["OTHER_SPEAKER"])

    def test_identical_classes_give_high_eer(self):
        samples, embeds, prof = _samples_and_embeds(sep=3.0)
        rng = np.random.default_rng(9)
        c = rng.normal(size=32)
        for s in samples:                      # everyone from ONE distribution
            embeds[s.sample_id] = c + rng.normal(scale=0.4, size=32)
        r = analyze(samples, embeds, existing_profile=prof, k_folds=5)
        self.assertGreater(r["cv_centroid"]["OTHER_SPEAKER"]["all"]["eer"], 0.25)

    def test_cv_never_scores_a_sample_against_a_profile_it_was_enrolled_in(self):
        """Poison the enrollment: if a sample leaked into its own profile it
        would score ~1.0 against it. Held-out scores must stay well below."""
        rng = np.random.default_rng(5)
        samples, embeds = [], {}
        for i in range(20):                    # every Yusuf sample is unique noise
            sid = f"y{i:03d}"
            samples.append(Sample(sid, "YUSUF", 3.0, "plain",
                                  "2026-09-29T20:00:00", "s"))
            embeds[sid] = rng.normal(size=256)
        samples.append(Sample("o000", "OTHER_SPEAKER", 3.0, "plain",
                              "2026-09-29T20:00:00", "s", consent="c"))
        embeds["o000"] = rng.normal(size=256)
        r = analyze(samples, embeds, k_folds=5)
        self.assertLess(r["cv_centroid"]["OTHER_SPEAKER"]["all"]["pos"]["max"], 0.6)

    def test_missing_classes_are_omitted_not_invented(self):
        samples, embeds, prof = _samples_and_embeds()
        only_yus = [s for s in samples if s.label == "YUSUF"]
        r = analyze(only_yus, embeds, existing_profile=prof)
        self.assertNotIn("OTHER_SPEAKER", r.get("existing_profile", {}))

    def test_too_few_samples_skips_cross_validation(self):
        samples, embeds, prof = _samples_and_embeds(n_pos=3)
        self.assertNotIn("cv_centroid", analyze(samples, embeds, prof))


class JoinLogTests(unittest.TestCase):
    def test_matches_on_time_and_duration_only(self):
        s1 = Sample("a", "YUSUF", 1.6, "aec", "2026-09-29T22:00:10", "s")
        s2 = Sample("b", "YUSUF", 3.0, "aec", "2026-09-29T22:05:00", "s")
        log = [
            "2026-09-29 22:00:11 [ears] voice_lock shadow: score 0.53 (would reject)",
            "2026-09-29 22:00:11 [ears] heard 1.6s, speech level median -40 max -30 dBFS",
            "2026-09-29 22:05:01 [ears] voice_lock shadow: score 0.71 (would reject)",
            "2026-09-29 22:05:01 [ears] heard 1.1s, speech level median -40 max -30 dBFS",
        ]
        got = join_production_scores([s1, s2], log)
        self.assertEqual(got, {"a": 0.53})     # b's duration does not match


class EndpointerTests(unittest.TestCase):
    def run_frames(self, pattern):
        ep, out = Endpointer(), []
        for sp in pattern:
            r = ep.feed(np.zeros(480, np.int16), sp)
            if r is not None:
                out.append(len(r))
        return out

    def test_speech_then_silence_makes_one_utterance_with_tail(self):
        out = self.run_frames([False] * 10 + [True] * 30 + [False] * 25)
        self.assertEqual(len(out), 1)
        # pre-roll(8) + speech run, opened after 4 frames + tail (20 frames)
        self.assertGreaterEqual(out[0], (8 + 30 + 20) * 480 - 4 * 480)

    def test_short_blip_is_discarded(self):
        self.assertEqual(self.run_frames([True] * 5 + [False] * 30), [])

    def test_two_utterances_in_a_row(self):
        pat = [True] * 20 + [False] * 25 + [True] * 20 + [False] * 25
        self.assertEqual(len(self.run_frames(pat)), 2)


if __name__ == "__main__":
    unittest.main()
