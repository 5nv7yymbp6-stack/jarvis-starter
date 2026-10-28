"""Voice responsiveness (2026-10-03): the first word out sooner, a stop that stops, and
proof that going faster did not cost correctness.

What is real here and what is faked:
  * Mouth      the REAL class (its synth thread, playback thread, generation counter and
               hold timer), with a fake synthesizer, a fake output stream that records every
               block with a timestamp, and the signal bus / music ducker patched out.
  * Ears       the REAL Ears.listen_once, fed synthetic frames and a fake VAD, so detection
               and endpoint latency are counted in FRAMES (deterministic, not wall-clock).
  * brain      the REAL WarmBrain / router / ledger / speak_reply over the fake SDK client.
Nothing here touches the live voice process, the production session file, log or ledger
(tests/__init__.py rebinds them), or the live .voice_* signal files (patched)."""
import asyncio
import contextlib
import random
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from backtalk import brain as B
from backtalk import ears as EARS
from backtalk import ledger as L
from backtalk import main as M
from backtalk import mouth as MOUTH
from backtalk.config import CFG
from tests import router_fakes as F
from tests import test_main_router as TMR
from tests.router_fakes import settle

BLOCK = 2205                       # samples per playback block (Mouth._play_rendering)


# ======================================================================================
# the real Mouth, with fake audio
# ======================================================================================

class FakeOut:
    """sd.OutputStream stand-in. Every block is recorded with its marker (the first
    sample: the fake synth fills a chunk's blocks with that chunk's marker, so a block
    can be traced to the chunk that made it) and a monotonic timestamp. Zero blocks
    are the silence padding Mouth._cut() writes and are not audio."""

    def __init__(self, block_s=0.0):
        self.block_s = block_s
        self.writes = []                     # (monotonic, marker)
        self.blocks = []                     # every block written, whole
        self.active = True

    def write(self, pcm):
        self.writes.append((time.monotonic(), int(pcm[0])))
        self.blocks.append(np.array(pcm, copy=True))
        if self.block_s:
            time.sleep(self.block_s)

    def start(self):
        pass

    def close(self, ignore_errors=True):
        pass

    def audio(self, marker=None):
        return [(t, m) for t, m in self.writes if m != 0 and (marker is None or m == marker)]


class FakeDucker:
    def speech_start(self): pass
    def speech_end(self, debounce=0.5): pass
    def restore_now(self): pass


def wait_until(cond, timeout=3.0, step=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


class MouthCase(unittest.TestCase):
    """Builds a real Mouth against fakes. `synth` maps a chunk's text to
    (first_audio_delay_s, n_blocks, gap_between_blocks_s)."""

    def build(self, synth=None, block_s=0.0, pause_max_s=None, pcm_for=None):
        synth = synth or (lambda text: (0.0, 4, 0.0))
        self.out = FakeOut(block_s)
        self.synth_started = {}              # text -> monotonic when its synthesis began
        self.synth_pulled = {}               # text -> blocks the consumer has pulled
        self.log_lines = []
        out = self.out

        def fake_synth_stream(text, timeout=30.0):
            self.synth_started[text] = time.monotonic()
            delay, n, gap = synth(text)
            time.sleep(delay)
            if pcm_for is not None:          # the test supplies the exact pieces the engine yields
                for piece in pcm_for(text):
                    yield MOUTH.KOKORO_RATE, piece
                return
            marker = ord(text[0])
            for i in range(n):
                self.synth_pulled[text] = i + 1
                yield MOUTH.KOKORO_RATE, np.full(BLOCK, marker, dtype=np.int16)
                if gap:
                    time.sleep(gap)

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(MOUTH, "synth_stream", fake_synth_stream))
        stack.enter_context(mock.patch.object(MOUTH.Mouth, "_get_out", lambda self, rate: out))
        stack.enter_context(mock.patch("backtalk.aec.enabled", return_value=False))
        stack.enter_context(mock.patch.object(MOUTH.audiowatch, "watch",
                                              lambda *a, **k: contextlib.nullcontext()))
        stack.enter_context(mock.patch("backtalk.ducking.Ducker", FakeDucker))
        stack.enter_context(mock.patch.object(MOUTH, "log", self.log_lines.append))
        for name in ("set_state", "static_stop", "reply_done", "direction", "feed_waveform"):
            stack.enter_context(mock.patch(f"backtalk.signals.{name}"))
        if pause_max_s is not None:
            stack.enter_context(mock.patch.object(MOUTH, "PAUSE_MAX_S", pause_max_s))
        self.mouth = MOUTH.Mouth()
        return self.mouth


class TtsPipeliningTests(MouthCase):
    """B. Claude text -> chunk -> TTS -> playback, with later chunks rendering while
    earlier ones play."""

    def test_the_first_word_does_not_wait_for_the_whole_reply(self):
        m = self.build(synth=lambda t: (0.15, 8, 0.0))
        t0 = time.monotonic()
        for c in "ABCDE":
            m.say_chunk(c)
        self.assertTrue(wait_until(lambda: self.out.audio(), 2.0))
        first = self.out.audio()[0][0] - t0
        self.assertLess(first, 0.15 + 0.30)                  # its own synth, not five of them (0.75 s)
        self.assertTrue(wait_until(lambda: not m.busy, 5.0))

    def test_later_chunks_render_while_the_earlier_one_is_still_playing(self):
        m = self.build(synth=lambda t: (0.10, 10, 0.0), block_s=0.03)
        for c in "ABC":
            m.say_chunk(c)
        self.assertTrue(wait_until(lambda: not m.busy, 6.0))
        last_a = max(t for t, _ in self.out.audio(ord("A")))
        # B's synthesis began while A was still being heard, not after it ended
        self.assertLess(self.synth_started["B"], last_a - 0.05)
        self.assertLess(self.synth_started["C"], max(t for t, _ in self.out.audio(ord("B"))))

    def test_chunks_play_in_order_with_nothing_lost_or_doubled(self):
        m = self.build(synth=lambda t: (0.02, 5, 0.0))
        for c in "ABCDEF":
            m.say_chunk(c)
        self.assertTrue(wait_until(lambda: not m.busy, 5.0))
        order = [mk for _, mk in self.out.audio()]
        self.assertEqual(order, [ord(c) for c in "ABCDEF" for _ in range(5)])

    def test_exactly_one_timing_line_per_reply_not_per_chunk(self):
        m = self.build(synth=lambda t: (0.02, 4, 0.0))
        for c in "ABC":
            m.say_chunk(c)
        self.assertTrue(wait_until(lambda: not m.busy, 3.0))
        lines = [l for l in self.log_lines if l.startswith("[timing] tts first chunk")]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"audio \d+\.\d\ds after queued \(synth \d+\.\d\ds, \S+, 1 chars\)")
        m.say_chunk("D")                                     # the next reply
        self.assertTrue(wait_until(lambda: not m.busy, 3.0))
        self.assertEqual(len([l for l in self.log_lines if l.startswith("[timing] tts first chunk")]), 2)


RATE = MOUTH.KOKORO_RATE
PAD = int(RATE * MOUTH.LEAD_KEEP_S)            # the lead-in the trim leaves in front of the first sound


def silence(n):
    return np.zeros(n, dtype=np.int16)


def speech(n, level=5000):
    return np.full(n, level, dtype=np.int16)


class LeadSilenceTrimTests(MouthCase):
    """Kokoro opens every chunk with ~0.36 s of digital near-silence. On the reply's FIRST
    chunk that is dead air before the first word, so it is cut; nothing else is touched."""

    LEAD = 8640                                        # 0.36 s at 24 kHz, as measured

    def played(self):
        return np.concatenate(self.out.blocks) if self.out.blocks else silence(0)

    def reply(self, *texts, pcm_for):
        m = self.build(pcm_for=pcm_for)
        for t in texts:
            m.say_chunk(t)
        self.assertTrue(wait_until(lambda: m.busy, 1.0))
        self.assertTrue(wait_until(lambda: not m.busy, 5.0))
        return self.played()

    def test_the_first_chunk_starts_a_beat_before_its_first_sound_not_0_36_s_before(self):
        got = self.reply("First", pcm_for=lambda t: [np.concatenate([silence(self.LEAD), speech(4410)])])
        self.assertEqual(len(got), PAD + 4410)
        self.assertTrue((got[:PAD] == 0).all())
        self.assertTrue((got[PAD:] == 5000).all())     # the sound itself is untouched

    def test_the_very_first_block_the_listener_gets_already_holds_the_word(self):
        self.reply("First", pcm_for=lambda t: [np.concatenate([silence(self.LEAD), speech(8820)])])
        # untrimmed, the first three written blocks would be pure silence
        self.assertTrue((self.out.blocks[0] != 0).any())

    def test_a_streaming_engine_that_yields_small_pieces_is_trimmed_the_same_way(self):
        def pieces(text):
            for _ in range(self.LEAD // 480):
                yield silence(480)
            for _ in range(10):
                yield speech(480)
        got = self.reply("First", pcm_for=pieces)
        # the sound starts exactly on a piece boundary, and still gets the same pad
        self.assertEqual(len(got), PAD + 4800)
        self.assertTrue((got[:PAD] == 0).all())
        self.assertTrue((got[PAD:] == 5000).all())

    def test_only_the_first_chunk_is_trimmed_later_chunks_keep_their_own_pause(self):
        got = self.reply("First", "Second",
                         pcm_for=lambda t: [np.concatenate([silence(self.LEAD), speech(4410)])])
        self.assertEqual(len(got), (PAD + 4410) + (self.LEAD + 4410))

    def test_a_chunk_with_no_sound_in_it_is_played_exactly_as_rendered(self):
        quiet = np.full(5000, MOUTH.LEAD_THRESH, dtype=np.int16)          # at the threshold, not above it
        got = self.reply("First", pcm_for=lambda t: [quiet[:2000], quiet[2000:]])
        self.assertEqual(len(got), 5000)

    def test_the_trim_never_touches_the_end_or_the_middle(self):
        body = np.concatenate([speech(3000), silence(2000), speech(3000, 9000), silence(4000)])
        got = self.reply("First", pcm_for=lambda t: [np.concatenate([silence(self.LEAD), body])])
        self.assertTrue((got[PAD:] == body).all())

    def test_a_pause_in_the_middle_of_a_streamed_first_chunk_survives(self):
        def pieces(text):
            yield silence(self.LEAD)
            yield speech(2205)
            yield silence(2205)                          # a real pause, after the sound began
            yield speech(2205, 9000)
        got = self.reply("First", pcm_for=pieces)
        self.assertEqual(len(got), PAD + 2205 * 3)
        self.assertTrue((got[PAD + 2205:PAD + 4410] == 0).all())

    def test_a_cut_while_the_lead_in_is_still_being_dropped_plays_nothing(self):
        gate = threading.Event()

        def slow(text):
            yield silence(self.LEAD // 2)
            gate.wait(2.0)                               # the engine is still in the lead-in
            yield silence(self.LEAD // 2)
            yield speech(4410)
        m = self.build(pcm_for=slow)
        m.say_chunk("First")
        time.sleep(0.15)
        m.shut_up()
        gate.set()
        self.assertTrue(wait_until(lambda: not m.busy, 3.0))
        time.sleep(0.2)
        self.assertEqual([w for w in self.out.writes if w[1] != 0], [])

    def test_the_timing_line_still_comes_once_per_reply(self):
        self.reply("First", "Second", pcm_for=lambda t: [np.concatenate([silence(self.LEAD), speech(4410)])])
        self.assertEqual(len([l for l in self.log_lines if l.startswith("[timing] tts first chunk")]), 1)


class TrimLeadFunctionTests(unittest.TestCase):
    """The pure function behind it: where the first real sound is."""

    def test_samples_up_to_the_threshold_are_lead_in_and_the_first_one_above_it_is_sound(self):
        pcm = np.concatenate([silence(5000), np.array([MOUTH.LEAD_THRESH], dtype=np.int16),
                              np.array([MOUTH.LEAD_THRESH + 1], dtype=np.int16), speech(100)])
        out, found = MOUTH.trim_lead(pcm, RATE)
        self.assertTrue(found)
        self.assertEqual(len(out), PAD + 101)                       # 5001 is the first loud sample
        self.assertEqual(int(out[PAD]), MOUTH.LEAD_THRESH + 1)

    def test_a_negative_swing_counts_as_sound_even_the_most_negative_sample(self):
        for v in (-MOUTH.LEAD_THRESH - 1, -32768):
            pcm = np.concatenate([silence(5000), np.array([v], dtype=np.int16), speech(50)])
            out, found = MOUTH.trim_lead(pcm, RATE)
            self.assertTrue(found, v)
            self.assertEqual(int(out[PAD]), v)

    def test_sound_near_the_start_keeps_everything_and_never_indexes_before_zero(self):
        pcm = np.concatenate([silence(100), speech(5000)])       # longer than the pad, so a
        out, found = MOUTH.trim_lead(pcm, RATE)                  # negative index would not clamp
        self.assertTrue(found)
        self.assertEqual(len(out), 5100)
        self.assertTrue((out[:100] == 0).all() and (out[100:] == 5000).all())

    def test_the_pad_comes_from_the_silence_already_dropped_when_the_sound_starts_a_piece(self):
        carry = silence(PAD + 500)
        out, found = MOUTH.trim_lead(speech(480), RATE, carry)
        self.assertTrue(found)
        self.assertEqual(len(out), PAD + 480)
        out, found = MOUTH.trim_lead(speech(480), RATE, None)
        self.assertEqual(len(out), 480)                              # no carry, no pad to give

    def test_pure_silence_reports_nothing_found_and_gives_back_nothing(self):
        out, found = MOUTH.trim_lead(silence(9000), RATE)
        self.assertFalse(found)
        self.assertEqual(len(out), 0)

    def test_the_pad_is_the_same_length_in_time_at_any_sample_rate(self):
        pcm = np.concatenate([silence(30000), speech(10)])
        out, _ = MOUTH.trim_lead(pcm, MOUTH.EL_RATE)
        self.assertEqual(len(out) - 10, int(MOUTH.EL_RATE * MOUTH.LEAD_KEEP_S))

    def test_the_constants_are_what_the_measurement_justified(self):
        # Kokoro's lead-in measured 338-377 ms and every sentence was under level 30 until
        # then; the pad covers a soft consonant but must stay far below the 0.36 s it replaces.
        self.assertLessEqual(MOUTH.LEAD_KEEP_S, 0.06)
        self.assertGreater(MOUTH.LEAD_KEEP_S, 0.0)
        self.assertLess(MOUTH.LEAD_THRESH, 200)
        self.assertGreater(MOUTH.LEAD_THRESH, 30)


class CancellationDuringPlaybackTests(MouthCase):
    """C/E. A cut stops the sound inside one block, leaves nothing stale behind, and
    leaves the mouth ready."""

    def test_shut_up_during_playback_stops_within_one_block(self):
        m = self.build(synth=lambda t: (0.0, 80, 0.0), block_s=0.02)
        m.say_chunk("A")
        self.assertTrue(wait_until(lambda: len(self.out.audio()) >= 3, 2.0))
        t_cut = time.monotonic()
        m.shut_up()
        time.sleep(0.4)
        late = [(t, mk) for t, mk in self.out.audio() if t > t_cut + 0.06]
        self.assertEqual(late, [])                           # nothing audible after the cut + one block
        self.assertLessEqual(len([1 for _, mk in self.out.writes if mk == 0]), 3)   # only the silence pad
        self.assertTrue(wait_until(lambda: not m.busy and not m.speaking, 1.0))

    def test_shut_up_during_synthesis_never_plays_stale_audio(self):
        m = self.build(synth=lambda t: (0.0, 40, 0.03))
        m.say_chunk("A")
        self.assertTrue(wait_until(lambda: self.synth_pulled.get("A", 0) >= 2, 2.0))
        t_cut = time.monotonic()
        pulled_at_cut = self.synth_pulled["A"]
        m.shut_up()
        time.sleep(0.5)
        # the synthesizer stops being consumed within a block or two ...
        self.assertLessEqual(self.synth_pulled["A"] - pulled_at_cut, 2)
        # ... and no block of the cut reply is written after it
        self.assertEqual([1 for t, mk in self.out.audio(ord("A")) if t > t_cut + 0.06], [])
        self.assertTrue(wait_until(lambda: not m.busy, 1.0))

    def test_a_new_reply_after_a_cut_plays_whole_and_nothing_stale_with_it(self):
        m = self.build(synth=lambda t: (0.0, 30 if t == "A" else 5, 0.0), block_s=0.02)
        m.say_chunk("A")
        self.assertTrue(wait_until(lambda: len(self.out.audio()) >= 2, 2.0))
        t_cut = time.monotonic()
        m.shut_up()
        m.say_chunk("B")
        self.assertTrue(wait_until(lambda: not m.busy, 5.0))
        stale = [t - t_cut for t, mk in self.out.audio(ord("A")) if t > t_cut + 0.06]
        self.assertEqual(stale, [])                          # none of the cut reply outlived the cut
        self.assertEqual(len(self.out.audio(ord("B"))), 5)   # and the new reply played whole
        self.assertTrue(all(t >= t_cut for t, _ in self.out.audio(ord("B"))))

    def test_rapid_repeated_interruptions_leave_no_stale_audio_and_a_clean_mouth(self):
        rnd = random.Random(7)
        m = self.build(synth=lambda t: (0.0, 25, 0.0), block_s=0.01)
        cuts = {}
        for i in range(25):
            ch = chr(65 + i)
            m.say_chunk(ch)
            time.sleep(rnd.uniform(0.0, 0.04))
            cuts[ord(ch)] = time.monotonic()
            m.shut_up()
        m.say_chunk("Z")
        self.assertTrue(wait_until(lambda: not m.busy and not m.speaking, 5.0))
        stale = [(mk, t - cuts[mk]) for t, mk in self.out.audio()
                 if mk != ord("Z") and t > cuts[mk] + 0.06]
        self.assertEqual(stale, [])                          # no block outlived its own cut
        self.assertEqual(len(self.out.audio(ord("Z"))), 25)  # the final reply played whole
        self.assertEqual(m._outstanding, 0)
        self.assertFalse([l for l in self.log_lines if "error" in l])

    def test_a_pause_that_turns_out_to_be_noise_carries_on_and_a_long_one_drops_the_reply(self):
        m = self.build(synth=lambda t: (0.0, 40, 0.0), block_s=0.02, pause_max_s=0.25)
        m.say_chunk("A")
        self.assertTrue(wait_until(lambda: len(self.out.audio()) >= 2, 2.0))
        m.hold()
        m.release()                                          # it was noise
        self.assertTrue(wait_until(lambda: not m.busy, 5.0))
        self.assertEqual(len(self.out.audio(ord("A"))), 40)  # played to the end
        self.assertFalse([l for l in self.log_lines if "reply dropped" in l])
        # still talking after the limit: the reply is dropped, and stays dropped
        m.say_chunk("B")
        self.assertTrue(wait_until(lambda: len(self.out.audio(ord("B"))) >= 2, 2.0))
        m.hold()
        self.assertTrue(wait_until(lambda: any("reply dropped" in l for l in self.log_lines), 2.0))
        n = len(self.out.audio(ord("B")))
        time.sleep(0.3)
        self.assertLessEqual(len(self.out.audio(ord("B"))) - n, 1)
        self.assertLess(len(self.out.audio(ord("B"))), 40)


class DevicePolicyTests(unittest.TestCase):
    """A. Kokoro runs on the CPU unless asked otherwise, and the CPU is warmed too."""

    def fake_torch(self, mps):
        t = types.ModuleType("torch")
        t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        return t

    def test_the_default_is_the_cpu_even_on_a_machine_with_mps(self):
        with mock.patch.dict(CFG, {"tts_device": "cpu"}), \
                mock.patch.dict(sys.modules, {"torch": self.fake_torch(True)}), \
                mock.patch.object(MOUTH, "_mps_disabled", False):
            self.assertEqual(MOUTH._pick_device(), "cpu")
        self.assertEqual(CFG["tts_device"], "cpu")           # the shipped default

    def test_mps_and_auto_opt_back_in_only_where_mps_exists(self):
        for want in ("mps", "auto", "MPS", " auto "):
            with mock.patch.dict(CFG, {"tts_device": want}), \
                    mock.patch.object(MOUTH, "_mps_disabled", False):
                with mock.patch.dict(sys.modules, {"torch": self.fake_torch(True)}):
                    self.assertEqual(MOUTH._pick_device(), "mps", want)
                with mock.patch.dict(sys.modules, {"torch": self.fake_torch(False)}):
                    self.assertEqual(MOUTH._pick_device(), "cpu", want)

    def test_anything_unrecognised_is_the_cpu_and_a_real_mps_failure_is_sticky(self):
        for bad in ("cuda", "gpu", "", None, 7):
            with mock.patch.dict(CFG, {"tts_device": bad}), \
                    mock.patch.dict(sys.modules, {"torch": self.fake_torch(True)}), \
                    mock.patch.object(MOUTH, "_mps_disabled", False):
                self.assertEqual(MOUTH._pick_device(), "cpu", bad)
        with mock.patch.dict(CFG, {"tts_device": "mps"}), \
                mock.patch.dict(sys.modules, {"torch": self.fake_torch(True)}), \
                mock.patch.object(MOUTH, "_mps_disabled", True):
            self.assertEqual(MOUTH._pick_device(), "cpu")    # J2's fallback still wins

    def test_warm_loads_on_the_chosen_device_and_primes_it_with_one_line(self):
        calls = []

        class Pipe:
            def __call__(self, text, voice=None, speed=1.0):
                calls.append(text)
                yield None, None, np.zeros(10, dtype=np.float32)
        loaded = []
        with mock.patch.dict(CFG, {"tts_device": "cpu"}), \
                mock.patch.object(MOUTH, "_pipe", None), mock.patch.object(MOUTH, "_device", None), \
                mock.patch.object(MOUTH, "_load_pipe", lambda lang, dev: loaded.append(dev) or Pipe()), \
                mock.patch.object(MOUTH, "_ensure_espeak"), \
                mock.patch.object(MOUTH, "_sweep_orphan_espeak_tempdirs"), \
                mock.patch.object(MOUTH, "log"):
            MOUTH.warm()
            self.assertEqual(loaded, ["cpu"])
            self.assertEqual(calls, ["Hello."])              # warmed on the CPU too
            self.assertEqual(MOUTH._device, "cpu")


# ======================================================================================
# the real Ears.listen_once, counted in frames
# ======================================================================================

class FakeVad:
    def is_speech(self, buf, rate):
        return int(np.abs(np.frombuffer(buf, dtype=np.int16)).max()) > 2000


class FakeMic:
    """script characters: S loud speech, F a distinctive first speech frame, q speech
    quieter than BARGE_MIN_DBFS, _ silence."""
    LEVEL = {"S": 6000, "F": 9000, "q": 60, "_": 0}

    def __init__(self, script):
        self.script, self.reads = list(script), 0

    def read(self, n):
        c = self.script[self.reads] if self.reads < len(self.script) else "_"
        self.reads += 1
        return np.full((n, 1), self.LEVEL[c], dtype=np.int16), False

    @property
    def exhausted(self):
        return self.reads >= len(self.script)


class EarsCase(unittest.TestCase):
    def listen(self, script, *, silence_ms=600, barge=True, gate=None, speaking=True, audible=True,
               max_loops=None):
        mic = FakeMic(script)
        loops = {"n": 0}

        def abort():
            # the listen ends when the script is used up, or after max_loops passes
            # through the loop (needed when a closed gate means nothing is ever read)
            loops["n"] += 1
            return mic.exhausted or (max_loops is not None and loops["n"] > max_loops)
        self.events, self.captured = [], []

        class Ctx:
            def __enter__(s):
                return mic

            def __exit__(s, *a):
                return False

        def transcribe(pcm):
            self.captured.append(pcm)
            self.reads_at_transcribe = mic.reads
            self.hearing_at_transcribe = self.ears.hearing
            return "hello there"
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(EARS, "_WatchedMic", Ctx))
        stack.enter_context(mock.patch.object(EARS, "transcribe", transcribe))
        stack.enter_context(mock.patch.object(EARS, "_passes_voice_lock", lambda pcm: True))
        stack.enter_context(mock.patch.object(EARS.aec, "enabled", return_value=True))   # no echo to police
        stack.enter_context(mock.patch.object(EARS, "log"))
        e = EARS.Ears(silence_ms=silence_ms)
        e.vad = FakeVad()
        self.mic = mic
        self.ears = e
        text = e.listen_once(
            gate=gate, abort=abort,
            speaking=(lambda: speaking), audible=(lambda: audible),
            barge=(lambda start: self.events.append((mic.reads, start))) if barge else None)
        return text


class HearingFlagTests(EarsCase):
    """Phase G follow-up: Ears.hearing is True exactly while an utterance is being captured, so the proactive layer never
    starts a reminder over someone mid-sentence on the open mic."""

    def test_it_is_false_at_rest_true_during_an_utterance_and_false_after(self):
        self.assertFalse(EARS.Ears().hearing)
        self.listen("_" * 5 + "F" + "S" * 15 + "_" * 25)
        self.assertTrue(self.hearing_at_transcribe)
        self.assertFalse(self.ears.hearing)

    def test_a_noise_blip_never_sets_it_for_good(self):
        self.listen("_" * 5 + "S" * 5 + "_" * 30)
        self.assertFalse(self.ears.hearing)

    def test_an_abort_mid_utterance_clears_it(self):
        self.listen("_" * 5 + "F" + "S" * 40)                  # the script runs out while the person is still talking
        self.assertFalse(self.ears.hearing)

    def test_the_context_reads_it(self):
        import inspect
        from backtalk import main as M
        self.assertIn('getattr(ears, "hearing", False)', inspect.getsource(M.amain))


class BargeInDetectionTests(EarsCase):
    """C. Detection latency, preservation of the person's first words, and no double
    capture, all counted in 30 ms frames."""

    def test_speech_over_the_voice_is_detected_after_exactly_six_frames(self):
        pre = 5
        self.listen("_" * pre + "F" + "S" * 15 + "_" * 25)
        start, first = self.events[0]
        self.assertTrue(first)                               # barge(True): the voice pauses
        frames_from_onset = start - pre                      # reads since the first speech frame, inclusive
        self.assertEqual(frames_from_onset, EARS.BARGE_FRAMES)
        self.assertEqual(frames_from_onset * EARS.FRAME_MS, 180)   # the median in the live log

    def test_a_noise_blip_shorter_than_the_threshold_never_pauses_the_voice(self):
        self.listen("_" * 5 + "S" * 5 + "_" * 30)
        self.assertEqual(self.events, [])

    def test_quiet_speech_below_the_floor_never_pauses_the_voice(self):
        self.listen("_" * 5 + "q" * 20 + "_" * 30)
        self.assertEqual(self.events, [])

    def test_no_cut_in_when_the_voice_is_not_audible(self):
        self.listen("_" * 5 + "S" * 15 + "_" * 25, audible=False)
        self.assertEqual(self.events, [])

    def test_the_first_words_survive_because_the_pre_roll_is_kept(self):
        self.listen("_" * 6 + "F" + "S" * 14 + "_" * 25)
        pcm = self.captured[0]
        # The VAD only opens after OPEN_FRAMES (4) of speech, so by then the first
        # speech frame is already 3 frames old. The ring of 8 frames keeps it, and
        # the 4 frames of silence in front of it.
        first_nonzero = int(np.nonzero(pcm)[0][0])
        self.assertEqual(first_nonzero, 4 * EARS.FRAME_LEN)
        self.assertEqual(int(pcm[first_nonzero]), 9000)      # the very first speech frame is in the utterance

    def test_the_utterance_ends_after_exactly_silence_ms_of_quiet(self):
        text = self.listen("_" * 3 + "F" + "S" * 15 + "_" * 40, silence_ms=600, barge=False, speaking=False)
        self.assertEqual(text, "hello there")
        last_speech_read = 3 + 1 + 15
        self.assertEqual(self.reads_at_transcribe - last_speech_read, 600 // EARS.FRAME_MS)   # 20 frames

    def test_the_open_mic_yields_while_the_button_records_so_one_utterance_is_never_two(self):
        text = self.listen("S" * 50, gate=lambda: True, barge=False, speaking=False, max_loops=4)
        self.assertIsNone(text)
        self.assertEqual(self.mic.reads, 0)                  # it never read a frame, so it cannot have heard it twice
        self.assertEqual(self.captured, [])


# ======================================================================================
# the in-flight decision (pure), and the stop that stops
# ======================================================================================

class DecideDuringTurnTests(unittest.TestCase):
    def d(self, text, *, busy=False, elapsed=5.0, last="do the thing", age=3.0):
        return M.decide_during_turn(text, mouth_busy=busy, elapsed_s=elapsed,
                                    last_turn_text=last, last_turn_age_s=age)

    def test_a_bare_stop_word_is_a_stop_whether_the_voice_is_silent_or_not(self):
        for t in ("stop", "wait", "Jarvis, stop!", "okay wait", "hold on", "shut up", "enough", "quiet"):
            self.assertEqual(self.d(t, busy=False), "stop", t)       # was "stitch": the tool carried on
            self.assertEqual(self.d(t, busy=True), "stop", t)
            self.assertEqual(self.d(t, busy=False, age=999), "stop", t)   # was a brand-new model turn
            self.assertEqual(self.d(t, busy=False, last=""), "stop", t)

    def test_the_bare_name_is_a_check_in_not_a_stop(self):
        self.assertEqual(self.d("Jarvis?"), "checkin")
        self.assertEqual(self.d("Jarvis"), "checkin")
        self.assertEqual(self.d("you there?"), "checkin")
        self.assertEqual(self.d("did you find it"), "checkin")

    def test_the_bare_name_over_a_talking_voice_keeps_its_old_meaning_and_is_not_made_a_stop(self):
        # The check-in rule needs a silent mouth, so this is the case that tells "the name is
        # not a stop word" apart from "the name happened to be read as a check-in first".
        # It was and remains a replacement of the turn (the old behaviour), not the new stop.
        self.assertEqual(self.d("Jarvis", busy=True), "interrupt")
        self.assertEqual(self.d("Jarvis!", busy=True, age=999), "interrupt")
        self.assertFalse(M.is_stop_command("Jarvis"))
        self.assertFalse(M.is_stop_command("hey Jarvis"))
        self.assertTrue(M._is_stop_phrase("Jarvis"))        # over speech it still silences the voice

    def test_a_check_in_is_only_a_check_in_into_silence_after_the_minimum_time(self):
        self.assertEqual(self.d("you there?", busy=True), "stitch")        # said over a reply: a real barge-in
        self.assertEqual(self.d("you there?", elapsed=0.5), "stitch")      # too soon: still him talking
        self.assertEqual(self.d("you there?", busy=True, age=999), "interrupt")

    def test_a_stop_with_other_words_is_a_request_not_a_stop(self):
        self.assertEqual(self.d("wait, also check the vault"), "stitch")
        self.assertEqual(self.d("wait what did you say", age=999), "interrupt")
        self.assertEqual(self.d("stop the background task"), "stitch")     # a verb: handled before this is asked
        self.assertFalse(M.is_stop_command("stop everything"))

    def test_the_rest_of_a_request_is_stitched_not_sent_as_a_second_turn(self):
        self.assertEqual(self.d("and also the notes"), "stitch")
        self.assertEqual(self.d("and also the notes", busy=True), "stitch")
        self.assertEqual(self.d("and also the notes", age=M.STITCH_MAX_S), "stitch")

    def test_a_stale_or_absent_request_is_replaced_not_stitched(self):
        self.assertEqual(self.d("and also the notes", age=M.STITCH_MAX_S + 0.1), "interrupt")
        self.assertEqual(self.d("and also the notes", last=""), "interrupt")

    def test_the_old_decisions_are_otherwise_unchanged(self):
        # (what the inline code did before: check-in, then stitch, else interrupt)
        for text in ("open safari", "what time is it", "and then the other one", "never mind that"):
            for busy in (False, True):
                for age in (1.0, 19.9, 20.1, 100.0):
                    want = ("stitch" if age <= M.STITCH_MAX_S and (not busy or not M._is_stop_phrase(text))
                            else "interrupt")
                    self.assertEqual(self.d(text, busy=busy, age=age), want, (text, busy, age))


class StopCurrentTurnTests(TMR.Base):
    """F/E/G. A stop during slow tool work ends the turn without a new model turn, keeps
    the session usable, and never touches protected background tasks."""

    def wire_abort(self):
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))

    async def start_tool_turn(self, text="run it"):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.tool_use("tb", "Bash", {"command": "sleep 30"}))
        task = asyncio.create_task(self.reply(text))
        await settle(10)
        self.wire_abort()
        return task

    async def test_stop_while_a_tool_runs_cancels_it_says_so_and_starts_no_new_turn(self):
        task = await self.start_tool_turn()
        took = await M.stop_current_turn(self.brain, self.mouth, task, voice_was_busy=False)
        self.assertEqual(self.client.interrupts, 1)
        self.assertEqual(len(self.client.sent), 1)                       # no "stop" turn went to the model
        self.assertEqual(self.mouth.said, ["Stopped."])                  # silent work: he is told it worked
        self.assertTrue(task.done())
        self.assertGreaterEqual(self.mouth.shut, 1)
        self.assertLess(took, 2.0)
        await settle()
        self.assertTrue(all(t["state"] in L.TURN_TERMINAL for t in self.led.turns.values()))

    async def test_stop_over_a_voice_that_is_talking_is_silent(self):
        task = await self.start_tool_turn()
        await M.stop_current_turn(self.brain, self.mouth, task, voice_was_busy=True)
        self.assertEqual(self.mouth.said, [])                            # the silence is the answer

    async def test_the_next_question_after_a_stop_is_answered_by_itself(self):
        task = await self.start_tool_turn()
        await M.stop_current_turn(self.brain, self.mouth, task, voice_was_busy=False)
        await settle()
        self.mouth.said.clear(); self.mouth.chunks.clear()
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Back."))
        await self.brain.reset_turn()
        await self.reply("you there?")
        self.assertEqual(self.mouth.chunks, ["Back."])                   # its own answer, nothing stale
        self.assertEqual(len(self.client.sent), 2)                       # exactly one query per utterance

    async def test_stop_does_not_kill_a_protected_background_task(self):
        def launch(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk1", "toolu_tk1", "export"),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.script = launch
        await self.reply("go")
        self.mouth.said.clear(); self.mouth.chunks.clear()
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working"))
        task = asyncio.create_task(self.reply("and now this"))
        await settle(10)
        await M.stop_current_turn(self.brain, self.mouth, task, voice_was_busy=False)
        self.assertEqual(self.client.interrupts, 0)                      # interrupt would have killed it
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)
        self.assertTrue(task.done())

    async def test_how_long_the_cancel_took_is_logged(self):
        task = await self.start_tool_turn()
        with mock.patch.object(M, "log") as lg:
            await M.stop_current_turn(self.brain, self.mouth, task, voice_was_busy=False)
        lines = [c.args[0] for c in lg.call_args_list]
        self.assertTrue(any(l.startswith("[timing] stop: the turn was cancelled in ") for l in lines), lines)
        self.assertTrue(any(l.startswith("[timing] interrupt aborted in ") for l in lines), lines)


class StreamingAndCancellationTests(TMR.Base):
    """A/E/G/H. Streaming, cancelling mid-generation, repeated cuts and what survives."""

    async def test_the_first_sentence_is_handed_up_before_the_model_has_finished(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("First sentence. Sec"))
        agen = self.brain.ask_stream("hi")
        first = await asyncio.wait_for(agen.__anext__(), 2)
        self.assertEqual(first, "First sentence.")                       # the run is still open
        self.assertIsNotNone(self.brain.router.active_handle())
        self.client.feed(F.delta("ond one."), F.asst("x"), F.result("x"))
        rest = [s async for s in agen]
        self.assertEqual(rest, ["Second one."])

    async def test_speech_before_a_tool_call_is_flushed_not_held_until_the_tool_ends(self):
        from claude_agent_sdk import StreamEvent
        stop = StreamEvent(uuid="se", session_id=F.SID, event={"type": "content_block_stop"})
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Checking the vault"), stop)
        agen = self.brain.ask_stream("hi")
        self.assertEqual(await asyncio.wait_for(agen.__anext__(), 2), "Checking the vault")

    async def test_cancelling_during_generation_stops_the_flow_to_the_mouth_for_good(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working on it. Still"))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        task = asyncio.create_task(self.reply("go"))
        await settle(10)
        self.assertEqual(self.mouth.chunks, ["Working on it."])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.client.interrupts, 1)
        # the dead turn's late tail arrives: it must not reach the mouth
        self.client.feed(F.delta(" going. More text here."), F.asst("x"))
        await settle(10)
        self.assertEqual(self.mouth.chunks, ["Working on it."])
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Back."))
        await self.brain.reset_turn()
        await self.reply("again")
        self.assertEqual(self.mouth.chunks, ["Working on it.", "Back."])

    async def test_rapid_repeated_interruptions_send_one_query_each_and_stay_consistent(self):
        n = 8
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Thinking out loud. And"))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        for i in range(n):
            task = asyncio.create_task(self.reply(f"question {i}"))
            await settle(8)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await settle(4)
        self.assertEqual(len(self.client.sent), n)                       # one per utterance, no duplicates
        self.assertEqual(len({m["uuid"] for m in self.client.sent}), n)
        self.assertEqual(self.client.interrupts, n)
        self.assertTrue(all(t["state"] in L.TURN_TERMINAL for t in self.led.turns.values()))
        self.mouth.chunks.clear()
        self.script = lambda c, m: c.feed(*F.fg_run(m["uuid"], "Final."))
        await self.brain.reset_turn()
        await self.reply("last one")
        self.assertEqual(self.mouth.chunks, ["Final."])                  # its own answer, exactly once
        self.assertEqual(len(self.client.sent), n + 1)
        self.assertEqual(self.brain.router.stats["router_error"], 0)

    async def test_cancelling_while_a_tool_is_running_is_clean_and_the_tool_is_known(self):
        seen = []
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.tool_use("tb", "Bash", {"command": "sleep 30"}))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        task = asyncio.create_task(M.speak_reply(self.brain, self.mouth, "run it", M.TurnProgress(),
                                                 vault_fut=self._done_future()))
        await settle(10)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.client.interrupts, 1)
        self.assertEqual(self.mouth.chunks, [])
        self.assertTrue(all(t["state"] in L.TURN_TERMINAL for t in self.led.turns.values()))

    def _done_future(self):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    async def test_the_progress_the_check_in_reads_knows_which_tool_is_running(self):
        self.script = lambda c, m: c.feed(
            F.init(), F.echo(m["uuid"]),
            F.delta("On it."),
            __import__("claude_agent_sdk").StreamEvent(uuid="s", session_id=F.SID, event={
                "type": "content_block_start", "content_block": {"type": "tool_use", "name": "Bash"}}))
        prog = M.TurnProgress()
        task = asyncio.create_task(M.speak_reply(self.brain, self.mouth, "go", prog,
                                                 vault_fut=self._done_future()))
        await settle(10)
        self.assertEqual(prog.tool, "Bash")
        self.assertIn("running a command", prog.status())
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class BackgroundSurvivesForegroundCutsTests(TMR.Base):
    async def launch(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk1", "toolu_tk1", "export"),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.script = script
        await self.reply("go")
        self.mouth.chunks.clear(); self.mouth.said.clear()

    async def test_repeated_foreground_cuts_never_kill_a_protected_task_or_speak_its_report(self):
        await self.launch()
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Working. More"))
        for i in range(5):
            task = asyncio.create_task(self.reply(f"q{i}"))
            await settle(8)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.client.interrupts, 0)
        self.assertEqual(self.led.tasks["tk1"]["state"], L.K_RUNNING)
        # the task finishes during the chaos: a model-written report arrives
        self.client.feed(F.task_updated("tk1", "completed"), F.task_note("tk1", "toolu_tk1", "completed", "42 rows"),
                         *F.unsolicited_run("Sir, your export finished with 42 rows"))
        await settle(12)
        self.assertNotIn("42 rows", " ".join(self.mouth.chunks + self.mouth.said))


class InvariantsAfterInterruptionTests(TMR.Base):
    """13/14/15. Going faster did not loosen the verification of results or the
    foreground boundary."""

    async def test_the_prompt_gate_is_still_wired_and_no_tool_hook_is(self):
        b = B.WarmBrain(model="m", ledger=self.led)
        self.addCleanup(lambda: [x.cancel() for x in (b.router._pump_task, b.router._watch_task) if x])
        await b.start()
        hooks = self.clients[-1].options.hooks
        self.assertEqual([m.matcher for m in hooks["PreToolUse"]], ["Bash"])      # the deny-list guard, nothing broader
        self.assertEqual(hooks["UserPromptSubmit"][0].hooks, [b._gate_prompt])
        self.assertEqual(self.clients[-1].options.permission_mode, "bypassPermissions")

    def test_main_builds_the_brain_without_a_tool_hook(self):
        with open(M.__file__, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("pre_tool_hook", src)
        self.assertNotIn("can_use_tool", src)

    async def test_a_result_landing_after_a_cut_is_still_queued_for_verification(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]),
                                          F.tool_use("tb", "Bash", {"command": "ls"}))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        task = asyncio.create_task(self.reply("list it"))
        await settle(10)
        F.seen(self.led, "tb", "ALLOW", tool="Bash")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        with mock.patch.object(self.brain.verify, "submit") as submit:
            self.client.feed(F.tool_result("tb", "a\nb\nc"))
            await settle(10)
        self.assertEqual(submit.call_count, 1)
        row = self.led.actions[self.led.action_id_for("tb")]
        self.assertEqual(row["state"], L.A_OK)
        self.assertEqual((row.get("evidence") or {}).get("kind"), "list")    # the bounded fact was still taken

    async def test_a_raw_background_report_is_still_withheld_after_repeated_interruptions(self):
        self.script = lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Thinking. And"))
        self.client.on_interrupt = lambda: self.client.feed(
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_tools"))
        for i in range(4):
            task = asyncio.create_task(self.reply(f"q{i}"))
            await settle(8)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        notice = ("<task-notification>\n<task-id>unknown-task</task-id>\n"
                  "<result>The agent reported 20 files.</result>\n</task-notification>")
        out = await self.brain._gate_prompt({"prompt": notice}, None, None)
        self.assertEqual(out.get("decision"), "block")                    # fails closed for an unknown task too
        self.assertEqual(await self.brain._gate_prompt({"prompt": "what time is it"}, None, None), {})


class WiringTests(unittest.TestCase):
    """handle() is a closure inside amain() and cannot be called from a test, so the
    parts of it that matter are pinned by structure: it must ask decide_during_turn
    and act on "stop" with stop_current_turn, and the cancel timing must be logged."""

    @classmethod
    def setUpClass(cls):
        with open(M.__file__, encoding="utf-8") as fh:
            cls.src = fh.read()
        a = cls.src.index("async def handle(text")
        cls.handle_src = cls.src[a:cls.src.index("def over_speech(", a)]

    def test_handle_asks_the_pure_decision_once_a_turn_is_in_flight(self):
        self.assertEqual(self.handle_src.count("decide_during_turn("), 1)
        self.assertIn("if speak_task and not speak_task.done():", self.handle_src)
        self.assertLess(self.handle_src.index("if speak_task and not speak_task.done():"),
                        self.handle_src.index("decide_during_turn("))

    def test_a_stop_goes_through_the_helper_and_ends_without_a_new_turn(self):
        i = self.handle_src.index('if action == "stop":')
        block = self.handle_src[i:self.handle_src.index('if action == "checkin":', i)]
        self.assertIn("await stop_current_turn(brain, mouth, speak_task,", block)
        self.assertIn("speak_task = None", block)
        self.assertIn("return True", block)
        self.assertNotIn("speak_reply(", block)                  # no model turn for "stop"
        self.assertNotIn("asyncio.create_task", block)

    def test_every_old_branch_is_still_there(self):
        for needle in ('if action == "checkin":', 'if action == "stitch":',
                       '"[turn] interrupted mid-reply by new input"',
                       "mouth.shut_up()", "await speak_task", "await brain.reset_turn()"):
            self.assertIn(needle, self.handle_src, needle)

    def test_cancellation_is_timed_in_both_places(self):
        self.assertIn("[timing] cancel:", self.handle_src)
        self.assertIn("[timing] interrupt {verdict} in", self.src)
        self.assertIn("[timing] stop: the turn was cancelled in", self.src)

    def test_the_cancel_handler_still_interrupts_the_brain(self):
        i = self.src.index("except asyncio.CancelledError:\n        t_int")
        self.assertIn("await brain.interrupt()", self.src[i:i + 400])
        self.assertIn("        raise", self.src[i:i + 600])      # and re-raises: a cancel stays a cancel


if __name__ == "__main__":
    unittest.main()
