"""Situation Phase 3A: the pure model-facing projector (situation_context.py).

Nothing here touches the live voice process, and nothing in production imports the module yet.
Snapshots come from the real SituationReader over simple fakes; the projector is exercised through
its three-argument API only."""
import ast
import calendar
import dataclasses
import inspect
import os
import re
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from backtalk import situation as S
from backtalk import situation_context as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TZ_SAVED = {}


def setUpModule():
    """Local-time formatting depends on the process time zone; pin it so the text is exact."""
    _TZ_SAVED["TZ"] = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()


def tearDownModule():
    if _TZ_SAVED["TZ"] is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = _TZ_SAVED["TZ"]
    time.tzset()


def at(h, m=0, day=6):
    """Wall-clock epoch for 2026-10-<day> h:m UTC (2026-10-06 is a Tuesday)."""
    return float(calendar.timegm((2026, 10, day, h, m, 0)))


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def idle_router():
    return SimpleNamespace(epoch="E-secret-1", state="OPEN", usable=True, handles={},
                           active_handle=lambda: None)


def task_row(i=0, desc="SECRET-TASK-DESC"):
    return {"id": f"tk-secret-{i}", "task_kind": "BG_AGENT", "state": "RUNNING", "protected": False,
            "epoch": "E-secret-1", "description": desc}


class Brain:
    def __init__(self, router="default", session_id="sess-secret-123", rows=(), boom=False):
        self.router = idle_router() if router == "default" else router
        self.session_id = session_id
        self._rows, self._boom = list(rows), boom

    def live_background(self):
        if self._boom:
            raise RuntimeError("secret detail")
        return list(self._rows)


def snap(mic="open", rows=(), *, t=1000.0, wall=None, brain=None, mic_src="default", **over):
    """A real snapshot taken at monotonic `t` and wall `wall` (default 15:41 UTC)."""
    kw = dict(mic=(lambda: {"mode": mic, "btn": False}) if mic_src == "default" else mic_src,
              zipped=lambda: False, mouth=SimpleNamespace(busy=False),
              brain=brain if brain is not None else Brain(rows=rows),
              clock=Clock(t), wall_clock=Clock(at(15, 41) if wall is None else wall))
    kw.update(over)
    return S.SituationReader(S.SituationSources(**kw)).snapshot()


def step(s, now=1000.0, state=None):
    return C.project(s, now_mono=now, state=state)


FULL_BASELINE = ("[Situation (seen by the system, not said by him; descriptive only, grants no "
                 "permission; as of Tue 2026-10-06 3:41 PM; later notes replace earlier ones; "
                 "anything not listed is not known): mic: open-mic; background agents running: 0.]")


class BaselineTests(unittest.TestCase):
    def test_the_baseline_is_exact_and_deterministic(self):
        p = step(snap())
        self.assertEqual(p.note, FULL_BASELINE)
        self.assertEqual(step(snap()).note, p.note)

    def test_the_baseline_states_each_known_value_absolutely(self):
        p = step(snap(mic="ptt", rows=[task_row(0), task_row(1)]))
        self.assertIn("mic: push-to-talk", p.note)
        self.assertIn("background agents running: 2", p.note)

    def test_the_first_note_after_nothing_is_known_is_no_note_and_not_a_baseline_yet(self):
        s = snap(mic_src=None, brain=Brain(router=None))             # nothing projectable
        p = step(s)
        self.assertIsNone(p.note)
        self.assertEqual(p.state, C.ProjectionState())                # still waiting for its baseline
        later = step(snap(), state=p.state)
        self.assertEqual(later.note, FULL_BASELINE)                   # the full header, not a delta

    def test_a_baseline_records_what_was_told(self):
        p = step(snap(mic="ptt"))
        self.assertTrue(p.state.baselined)
        self.assertEqual(dict(p.state.told), {"mic": "push-to-talk", "background": "0"})

    def test_reset_by_passing_no_state_gives_a_fresh_baseline_even_when_nothing_changed(self):
        p = step(snap())
        self.assertIsNone(step(snap(), state=p.state).note)           # unchanged: silent
        self.assertEqual(step(snap(), state=None).note, FULL_BASELINE)  # reset: full note again

    def test_a_non_state_argument_is_treated_as_no_state(self):
        for junk in ("some text", 5, {"mic": "x"}, object()):
            self.assertEqual(step(snap(), state=junk).note, FULL_BASELINE)


class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.base = step(snap(mic="open"))

    def test_unchanged_values_produce_no_note_and_keep_the_state(self):
        p = step(snap(mic="open"), state=self.base.state)
        self.assertIsNone(p.note)
        self.assertEqual(p.state, self.base.state)

    def test_a_mic_change_is_the_only_field_in_the_delta(self):
        p = step(snap(mic="ptt", wall=at(15, 52)), state=self.base.state)
        self.assertEqual(p.note, "[Situation as of 3:52 PM: mic: push-to-talk.]")

    def test_a_background_count_change_is_the_only_field_in_the_delta(self):
        p = step(snap(rows=[task_row(0)], wall=at(15, 52)), state=self.base.state)
        self.assertEqual(p.note, "[Situation as of 3:52 PM: background agents running: 1.]")

    def test_simultaneous_changes_are_one_note_in_a_fixed_order(self):
        p = step(snap(mic="ptt", rows=[task_row(0), task_row(1), task_row(2)], wall=at(16, 3)),
                 state=self.base.state)
        self.assertEqual(p.note,
                         "[Situation as of 4:03 PM: mic: push-to-talk; background agents running: 3.]")

    def test_a_delta_never_repeats_an_unchanged_field(self):
        p = step(snap(mic="open", rows=[task_row(0)]), state=self.base.state)
        self.assertNotIn("mic", p.note)

    def test_the_state_follows_the_notes_a_change_then_back_is_told_again(self):
        a = step(snap(mic="ptt"), state=self.base.state)
        b = step(snap(mic="ptt"), state=a.state)
        c = step(snap(mic="open"), state=b.state)
        self.assertIsNotNone(a.note)
        self.assertIsNone(b.note)
        self.assertIn("mic: open-mic", c.note)

    def test_not_adopting_the_returned_state_means_the_same_note_is_offered_again(self):
        s = snap(mic="ptt")
        first = step(s, state=self.base.state)
        again = step(s, state=self.base.state)                         # the send failed; state not adopted
        self.assertEqual(first.note, again.note)


class OmissionTests(unittest.TestCase):
    """Anything not known and fresh is absent. Never a default."""

    def test_unknown_mic_is_omitted_and_background_still_projected(self):
        p = step(snap(mic_src=None))
        self.assertNotIn("mic", p.note)
        self.assertIn("background agents running: 0", p.note)

    def test_unavailable_background_is_omitted_and_mic_still_projected(self):
        p = step(snap(brain=Brain(boom=True)))
        self.assertIn("mic: open-mic", p.note)
        self.assertNotIn("background", p.note)

    def test_an_unwired_brain_omits_background_and_keeps_mic(self):
        s = S.SituationReader(S.SituationSources(
            mic=lambda: {"mode": "ptt", "btn": False}, clock=Clock(1000.0), wall_clock=Clock(at(15, 41)))).snapshot()
        self.assertEqual(s.meta_of("work.background").reason, "unwired")
        p = step(s)
        self.assertIn("mic: push-to-talk", p.note)
        self.assertNotIn("background", p.note)

    def test_stale_fields_are_omitted(self):
        s = snap(t=1000.0)
        self.assertEqual(step(s, now=1004.9).note, FULL_BASELINE)       # inside the 5 s TTL
        self.assertEqual(step(s, now=1005.0).note, FULL_BASELINE)       # exactly the TTL is still fresh
        self.assertIsNone(step(s, now=1005.1).note)                     # both are past it
        self.assertIsNone(step(s, now=900.0).note)                      # a clock behind the read

    def test_inside_the_ttl_both_fields_are_projected(self):
        s = snap(t=1000.0)
        self.assertEqual(s.meta_of("audio.mic_mode").ttl_s, s.meta_of("work.background").ttl_s)
        self.assertIsNotNone(step(s, now=1003.0).note)

    def test_no_router_omits_background_and_keeps_mic(self):
        s = snap(brain=Brain(router=None))
        self.assertEqual(s.meta_of("work.background").state, "known")   # known-empty, read from the brain
        p = step(s)
        self.assertIn("mic: open-mic", p.note)
        self.assertNotIn("background", p.note)                          # "no router" is not "0 agents"

    def test_no_metadata_omits_the_field(self):
        ttl = {k: v for k, v in S._TTL_S.items() if k != "audio.mic_mode"}
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            s = snap()
        self.assertEqual(s.meta_of("audio.mic_mode").reason, "no_metadata")
        self.assertEqual(s.audio.mic_mode, "open")                     # the value is there; it is not trusted
        p = step(s)
        self.assertNotIn("mic", p.note)
        ttl = {k: v for k, v in S._TTL_S.items() if k != "work.background"}
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            s = snap()
        self.assertNotIn("background", step(s).note)

    def test_unavailable_fields_are_omitted(self):
        def boom():
            raise RuntimeError("secret detail")
        s = snap(mic_src=boom)
        self.assertEqual(s.meta_of("audio.mic_mode").reason, "unavailable")
        self.assertNotIn("mic", step(s).note)
        s = snap(brain=Brain(boom=True))
        self.assertNotIn("background", step(s).note)

    def test_everything_unavailable_is_no_note(self):
        s = S.SituationReader(S.SituationSources()).snapshot()
        p = step(s)
        self.assertIsNone(p.note)
        self.assertEqual(p.state, C.ProjectionState())


class RetractionTests(unittest.TestCase):
    def setUp(self):
        self.base = step(snap(mic="open", rows=[task_row(0)]))

    def lost_mic(self, wall=at(16, 0)):
        return snap(mic_src=None, rows=[task_row(0)], wall=wall)

    def test_a_told_field_that_stops_being_known_is_retracted_once(self):
        p = step(self.lost_mic(), state=self.base.state)
        self.assertEqual(p.note, "[Situation as of 4:00 PM: mic: no longer known.]")

    def test_the_retraction_does_not_repeat(self):
        p = step(self.lost_mic(), state=self.base.state)
        for _ in range(3):
            q = step(self.lost_mic(at(16, 5)), state=p.state)
            self.assertIsNone(q.note)
            self.assertEqual(q.state, p.state)

    def test_a_field_that_becomes_known_again_is_told_absolutely(self):
        p = step(self.lost_mic(), state=self.base.state)
        q = step(snap(mic="ptt", rows=[task_row(0)], wall=at(16, 10)), state=p.state)
        self.assertEqual(q.note, "[Situation as of 4:10 PM: mic: push-to-talk.]")
        r = step(snap(mic="ptt", rows=[task_row(0)], wall=at(16, 11)), state=q.state)
        self.assertIsNone(r.note)

    def test_a_field_that_returns_to_the_same_value_is_still_told_again(self):
        p = step(self.lost_mic(), state=self.base.state)
        q = step(snap(mic="open", rows=[task_row(0)], wall=at(16, 10)), state=p.state)
        self.assertIn("mic: open-mic", q.note)                         # the model was told "no longer known"

    def test_the_background_retraction_wording(self):
        s = snap(brain=Brain(router=None), wall=at(16, 0))
        p = step(s, state=self.base.state)
        self.assertEqual(p.note, "[Situation as of 4:00 PM: background agents running: no longer known.]")

    def test_stale_fields_are_retracted_too(self):
        late = snap(t=1000.0)
        p = step(late, now=1100.0, state=self.base.state)
        self.assertEqual(p.note, "[Situation as of 3:41 PM: mic: no longer known; "
                                 "background agents running: no longer known.]")

    def test_both_retracted_then_both_recovered(self):
        gone = step(snap(mic_src=None, brain=Brain(router=None), wall=at(16, 0)), state=self.base.state)
        self.assertIn("mic: no longer known", gone.note)
        self.assertIn("background agents running: no longer known", gone.note)
        back = step(snap(mic="open", rows=[], wall=at(16, 9)), state=gone.state)
        self.assertEqual(back.note,
                         "[Situation as of 4:09 PM: mic: open-mic; background agents running: 0.]")

    def test_nothing_is_retracted_that_was_never_told(self):
        p = step(snap(mic_src=None, wall=at(16, 0)))                   # a baseline without a mic
        self.assertNotIn("mic", p.note)
        q = step(snap(mic_src=None, wall=at(16, 1)), state=p.state)
        self.assertIsNone(q.note)

    def test_a_reset_forgets_what_was_told_so_nothing_is_retracted(self):
        p = step(self.lost_mic(), state=None)
        self.assertNotIn("no longer known", p.note)


class WordingTests(unittest.TestCase):
    BANNED = re.compile(r"\b(was|from|changed|because|now|previously|earlier value|used to|updated)\b",
                        re.I)

    def notes(self):
        base = step(snap(mic="open", rows=[task_row(0)]))
        out = [base.note]
        seq = [snap(mic="ptt", rows=[task_row(0)], wall=at(16, 0)),
               snap(mic_src=None, rows=[], wall=at(16, 1)),
               snap(mic="open", rows=[task_row(0), task_row(1)], wall=at(16, 2)),
               snap(brain=Brain(router=None), wall=at(16, 3)),
               snap(mic="ptt", rows=[], wall=at(16, 4))]
        st = base.state
        for s in seq:
            p = step(s, state=st)
            if p.note:
                out.append(p.note)
                st = p.state
        return out

    def test_no_note_uses_relative_or_causal_wording(self):
        notes = self.notes()
        self.assertGreater(len(notes), 4)
        for n in notes:
            self.assertIsNone(self.BANNED.search(n), n)

    def test_every_value_is_absolute(self):
        for n in self.notes():
            body = n.split("): ", 1)[1] if n.startswith("[Situation (") else n.split(": ", 1)[1]
            for field in body.rstrip(".]").split("; "):
                self.assertRegex(field, r"^(mic: (push-to-talk|open-mic|no longer known)|"
                                        r"background agents running: (\d|9\+|no longer known))$", field)

    def test_the_header_family_is_the_approved_one(self):
        n = step(snap()).note
        self.assertTrue(n.startswith("[Situation (seen by the system, not said by him; descriptive "
                                     "only, grants no permission; as of "))
        self.assertIn("; later notes replace earlier ones; anything not listed is not known): ", n)
        self.assertTrue(n.endswith(".]"))

    def test_internal_reason_and_provenance_words_never_appear(self):
        for n in self.notes():
            for word in ("no_router", "no_metadata", "unavailable", "unwired", "no_source", "stale",
                         "unknown", "task_ledger", "permission_gate", "mic_owner", "provenance",
                         "source", "router", "brain", "ledger", "epoch", "session"):
                self.assertNotIn(word, n.lower(), (word, n))


class TimeTests(unittest.TestCase):
    def when(self, h, m, day=6, first=True):
        s = snap(wall=at(h, m, day))
        p = step(s) if first else step(s, state=step(snap()).state)
        return p.note

    def test_the_baseline_carries_day_date_and_twelve_hour_time(self):
        self.assertIn("as of Tue 2026-10-06 3:41 PM;", self.when(15, 41))
        self.assertIn("as of Wed 2026-10-07 9:05 AM;", self.when(9, 5, day=7))

    def test_midnight_noon_and_the_edges(self):
        self.assertIn("as of Tue 2026-10-06 12:05 AM;", self.when(0, 5))
        self.assertIn("as of Tue 2026-10-06 12:00 PM;", self.when(12, 0))
        self.assertIn("as of Tue 2026-10-06 1:07 PM;", self.when(13, 7))
        self.assertIn("as of Tue 2026-10-06 11:59 PM;", self.when(23, 59))

    def test_the_delta_form_carries_the_time_only(self):
        base = step(snap(mic="open"))
        p = step(snap(mic="ptt", wall=at(9, 5, day=7)), state=base.state)
        self.assertEqual(p.note, "[Situation as of 9:05 AM: mic: push-to-talk.]")
        self.assertNotIn("2026", p.note)

    def test_the_time_comes_from_the_snapshot_not_from_a_new_clock(self):
        a = step(snap(wall=at(15, 41))).note
        time.sleep(0.01)
        self.assertEqual(step(snap(wall=at(15, 41))).note, a)
        self.assertIn("as of Tue 2026-10-06 8:00 AM;", step(snap(wall=at(8, 0))).note)

    def test_the_timestamp_does_not_decide_freshness(self):
        far = at(15, 41, day=6) + 60 * 60 * 24 * 365 * 50          # a valid, very different wall time
        self.assertIsNotNone(step(snap(wall=far)).note)             # fresh by the monotonic clock
        stale = snap(t=1000.0, wall=at(15, 41))                      # a perfectly good wall time...
        self.assertIsNone(step(stale, now=2000.0).note)             # ...but stale by the monotonic clock

    def test_an_unusable_timestamp_fails_closed(self):
        for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -5.0, 5e9, None, "yesterday", True):
            p = step(snap(wall_clock=Clock(bad)))                       # exactly what the clock returned
            self.assertIsNone(p.note, bad)
            self.assertEqual(p.state, C.ProjectionState(), bad)


class SizeTests(unittest.TestCase):
    def test_the_cap_is_400(self):
        self.assertEqual(C.MAX_CHARS, 400)

    def test_every_realistic_note_is_well_inside_it(self):
        worst = step(snap(mic="ptt", rows=[task_row(i) for i in range(30)])).note
        self.assertLess(len(worst), 300)
        self.assertIn("background agents running: 9+", worst)

    def test_the_count_text_is_bounded(self):
        for n, want in ((0, "0"), (1, "1"), (9, "9"), (10, "9+"), (500, "9+")):
            note = step(snap(rows=[task_row(i) for i in range(n)])).note
            self.assertIn(f"background agents running: {want}.", note)

    def test_a_field_that_does_not_fit_is_dropped_whole_and_offered_next_time(self):
        full = step(snap(mic="ptt"))
        both = len(full.note)
        mic_only = len(C._HEADER_FULL.format(when="Tue 2026-10-06 3:41 PM", body="mic: push-to-talk"))
        with mock.patch.object(C, "MAX_CHARS", both - 1):
            p = step(snap(mic="ptt"))
        self.assertEqual(len(p.note), mic_only)
        self.assertIn("mic: push-to-talk.]", p.note)
        self.assertNotIn("background", p.note)                          # whole field, no fragment
        self.assertEqual(dict(p.state.told), {"mic": "push-to-talk"})  # the dropped one is not "told"
        nxt = step(snap(mic="ptt", wall=at(15, 50)), state=p.state)
        self.assertEqual(nxt.note, "[Situation as of 3:50 PM: background agents running: 0.]")

    def test_nothing_fits_means_no_note_and_an_unchanged_state(self):
        with mock.patch.object(C, "MAX_CHARS", 20):
            p = step(snap())
        self.assertIsNone(p.note)
        self.assertEqual(p.state, C.ProjectionState())

    def test_the_cap_is_a_hard_limit_and_a_note_is_never_malformed_at_any_cap(self):
        well_formed = re.compile(
            r"^\[Situation \(seen by the system[^\]]*\): (mic: [a-z-]+|background agents running: [\d+]+)"
            r"(; background agents running: [\d+]+)?\.\]$")
        for cap in range(0, 420):
            with mock.patch.object(C, "MAX_CHARS", cap):
                note = step(snap(mic="ptt")).note
            if note is not None:
                self.assertLessEqual(len(note), cap, cap)
                self.assertRegex(note, well_formed, cap)

    def test_a_delta_is_far_below_the_cap(self):
        base = step(snap())
        p = step(snap(mic="ptt", rows=[task_row(0)], wall=at(15, 50)), state=base.state)
        self.assertLess(len(p.note), 100)


class AllowlistTests(unittest.TestCase):
    SECRETS = ("sess-secret-123", "E-secret-1", "tk-secret", "SECRET-TASK-DESC", "SECRET-MODE",
               "secret detail", "/Users/", "rm -rf", "hunter2")
    PROHIBITED_WORDS = ("ptt", "push-to-talk key", "held", "zip", "mouth", "addressee", "wake", "idle",
                        "active", "window", "foreground", "pending", "confirm", "voice", "speaker",
                        "device", "desktop", "location", "turn", "tier", "latency",
                        "seconds")

    def rich(self):
        return snap(mic="open", rows=[task_row(0), task_row(1)],
                    addressee=SimpleNamespace(mode="wake", state="ACTIVE", window=180.0, max_active=300.0))

    def test_only_the_allowlisted_fields_and_no_secret_ever_reach_a_note(self):
        s = self.rich()
        note = step(s).note
        for secret in self.SECRETS:
            self.assertNotIn(secret, note)
        for word in self.PROHIBITED_WORDS:
            self.assertNotIn(word, note.lower().replace("push-to-talk", "").replace("open-mic", ""), word)
        self.assertEqual(note.count(";"), 5)                    # 4 in the header + 1 between the two fields

    def test_the_body_holds_exactly_the_two_allowlisted_fields(self):
        body = step(self.rich()).note.split("): ", 1)[1].rstrip(".]")
        self.assertEqual(body.split("; "), ["mic: open-mic", "background agents running: 2"])

    def test_an_unexpected_mic_value_is_never_copied(self):
        for weird in ("SECRET-MODE", "PTT", "Open", "", "ptt ", None, 5, "open-mic", "push-to-talk"):
            note = step(snap(mic=weird)).note
            self.assertIsNotNone(note, weird)                  # the background field is still projected
            self.assertNotIn("mic", note.split("): ", 1)[1], weird)
            if isinstance(weird, str) and weird:
                self.assertNotIn(weird.strip(), note.replace("open-mic", "").replace("push-to-talk", ""))

    def test_task_text_ids_and_kinds_never_appear(self):
        rows = [dict(task_row(i), task_kind="SECRET-KIND", description="SECRET-TASK-DESC") for i in range(3)]
        note = step(snap(rows=rows)).note
        for secret in ("SECRET", "tk-secret", "E-secret"):
            self.assertNotIn(secret, note)
        self.assertIn("background agents running: 3", note)

    def test_the_allowlist_is_exactly_mic_background_and_time(self):
        self.assertEqual(C._KEYS, ("mic", "background"))
        self.assertEqual(set(C._MIC_WORDS), {"ptt", "open"})
        self.assertEqual({C._MIC_PATH, C._BACKGROUND_PATH}, {"audio.mic_mode", "work.background"})


def projector_references(source: str):
    """(import scopes, use scopes) for the projector module in `source`. A scope is the NEAREST enclosing
    def's name ("<module>" outside any def). Uses are every `situation_context_mod` name; imports are every
    import of `situation_context`. Comments and docstrings are not code and are ignored."""
    import ast
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def scope(node):
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
        return "<module>"
    imports, uses = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any("situation_context" in (a.name or "") for a in node.names):
            imports.append(scope(node))
        elif isinstance(node, ast.Import) and any("situation_context" in (a.name or "") for a in node.names):
            imports.append(scope(node))
        elif isinstance(node, ast.Name) and node.id == "situation_context_mod":
            uses.append(scope(node))
    return sorted(imports), sorted(uses)


class PurityTests(unittest.TestCase):
    def test_the_module_imports_only_math_dataclasses_and_datetime(self):
        tree = ast.parse(inspect.getsource(C))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported, {"math", "dataclasses", "datetime"})

    def test_it_calls_no_io_logging_clock_or_dynamic_code(self):
        tree = ast.parse(inspect.getsource(C))
        names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(names & {"open", "print", "log", "exec", "eval", "setattr", "delattr", "input",
                                  "__import__"}, names)
        self.assertFalse(attrs & {"now", "utcnow", "time", "monotonic", "write", "send", "submit", "sleep",
                                  "append_to", "request", "get_event_loop"}, attrs)
        self.assertNotIn("global ", inspect.getsource(C))

    def test_importing_it_pulls_in_no_authority_or_runtime_module(self):
        code = ("import sys, backtalk.situation_context\n"
                "bad = {'backtalk.addressee', 'backtalk.router', 'backtalk.brain',"
                " 'backtalk.ledger', 'backtalk.main', 'backtalk.situation', 'backtalk.config',"
                " 'backtalk.vlog', 'backtalk.mouth', 'backtalk.ears'}\n"
                "print(sorted(bad & set(sys.modules)))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr[-300:])

    def test_no_other_production_module_references_the_projector_and_main_py_only_inside_its_helpers(self):
        pkg = os.path.join(REPO, "backtalk")
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py") and name not in ("situation_context.py", "main.py"):
                with open(os.path.join(pkg, name), encoding="utf-8") as f:
                    self.assertNotIn("situation_context", f.read(), name)        # authority modules stay clean
        with open(os.path.join(pkg, "main.py"), encoding="utf-8") as f:
            src = f.read()
        imports, uses = projector_references(src)
        self.assertEqual(imports, ["<module>"])                                  # one import, at module level
        self.assertTrue(uses)                                                    # and it is actually used
        # every use is INSIDE the approved shadow helpers, never in unrelated main.py code
        self.assertEqual(sorted(set(uses)), ["_sitctx_begin", "_sitctx_line"])
        self.assertEqual(sorted(uses), ["_sitctx_begin", "_sitctx_line", "_sitctx_line"])

    def test_the_reference_guard_catches_projector_use_added_anywhere_else_in_main_py(self):
        with open(os.path.join(REPO, "backtalk", "main.py"), encoding="utf-8") as f:
            base = f.read()
        good = projector_references(base)
        for extra in ("\ndef handle_like():\n    return situation_context_mod.project(None, now_mono=0.0)\n",
                      "\nasync def speak_like():\n    x = situation_context_mod.ProjectionState()\n",
                      "\nsituation_context_mod.project\n",                                   # module level
                      "\ndef _sitctx_new():\n    return situation_context_mod.Projection\n",   # a NEW helper name
                      "\nfrom backtalk import situation_context as sc2\n"):                  # a second import
            self.assertNotEqual(projector_references(base + extra), good, extra)
        # a use that is moved out of its approved helper into another function is also caught
        moved = base.replace("proj = situation_context_mod.project(", "proj = (lambda *a, **k: None)(", 1) \
            + "\ndef _elsewhere():\n    return situation_context_mod.project\n"
        self.assertNotEqual(projector_references(moved), good)
        self.assertEqual(projector_references(base + "\ndef f():\n    pass\n"), good)

    def test_the_api_cannot_carry_text_in(self):
        self.assertEqual(list(inspect.signature(C.project).parameters), ["snapshot", "now_mono", "state"])
        self.assertEqual([f.name for f in dataclasses.fields(C.ProjectionState)], ["baselined", "told"])
        self.assertEqual([f.name for f in dataclasses.fields(C.Projection)], ["note", "state"])

    def test_results_and_states_are_frozen(self):
        p = step(snap())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.note = "x"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.state.baselined = False
        self.assertIsInstance(p.state.told, tuple)

    def test_projecting_does_not_change_the_snapshot_or_the_state_it_was_given(self):
        s = snap(rows=[task_row(0)])
        before = repr(s)
        st = step(snap()).state
        st_before = repr(st)
        for _ in range(5):
            step(s, state=st)
            step(s)
        self.assertEqual(repr(s), before)
        self.assertEqual(repr(st), st_before)

    def test_repeated_calls_with_the_same_inputs_are_identical(self):
        s = snap(mic="ptt", rows=[task_row(0)])
        st = step(snap()).state
        runs = [step(s, state=st) for _ in range(4)] + [step(s, state=st)]
        self.assertEqual(len({(r.note, r.state) for r in runs}), 1)

    def test_nothing_is_printed(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            step(snap())
            step(snap(mic_src=None))
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))


class FailClosedTests(unittest.TestCase):
    def test_a_malformed_snapshot_gives_no_note_and_never_raises(self):
        st = step(snap()).state
        for junk in (object(), None, "snapshot", 5, SimpleNamespace(), SimpleNamespace(meta=())):
            p = step(junk, state=st)
            self.assertIsNone(p.note)
            self.assertEqual(p.state, st)

    def test_a_phase_1_style_snapshot_without_metadata_projects_nothing(self):
        bare = S.Snapshot(taken_at_wall=at(15, 41), taken_at_mono=1000.0,
                          audio=S.AudioView(mic_mode="open"), work=S.WorkView(foreground=(), active_turn_id=None,
                                                                              background=()))
        self.assertIsNone(step(bare).note)

    def test_a_bad_now_mono_gives_no_note(self):
        for bad in (None, "later", float("nan")):
            self.assertIsNone(C.project(snap(), now_mono=bad).note)

    def test_the_result_is_always_a_projection_with_a_state(self):
        for s in (snap(), snap(mic_src=None), object()):
            p = step(s)
            self.assertIsInstance(p, C.Projection)
            self.assertIsInstance(p.state, C.ProjectionState)

    def test_background_requires_a_tuple_of_work_not_a_list_or_text(self):
        s = snap()
        broken = dataclasses.replace(s, work=dataclasses.replace(s.work, background="3 agents"))
        self.assertNotIn("background", step(broken).note)
        broken = dataclasses.replace(s, work=dataclasses.replace(s.work, background=["a", "b"]))
        self.assertNotIn("background", step(broken).note)


# =====================================================================================
# REVIEW FIXES: router usability gate, local-time pinning, independent freshness, cap and state.
# =====================================================================================

def router(state="OPEN", usable=True):
    return SimpleNamespace(epoch="E-secret-1", state=state, usable=usable, handles={},
                           active_handle=lambda: None)


def with_meta(s, path, **changes):
    """The same snapshot with one field's metadata replaced (to test the projector's gate alone)."""
    meta = tuple(dataclasses.replace(m, **changes) if m.path == path else m for m in s.meta)
    return dataclasses.replace(s, meta=meta)


class RouterUsabilityGateTests(unittest.TestCase):
    """A router that exists but is not OPEN may still have task rows that are no longer true."""

    def test_a_usable_router_with_a_fresh_flag_projects_the_count(self):
        s = snap(brain=Brain(router=router("OPEN", True), rows=[task_row(0), task_row(1)]))
        self.assertEqual(s.meta_of("conversation.router_usable").state, "known")
        self.assertIs(s.conversation.router_usable, True)
        p = step(s)
        self.assertIn("background agents running: 2", p.note)
        self.assertIn("mic: open-mic", p.note)

    def test_a_lost_closing_or_closed_router_with_a_live_ledger_row_projects_no_count(self):
        for state in ("LOST", "CLOSING", "CLOSED"):
            s = snap(brain=Brain(router=router(state, False), rows=[task_row(0)]))
            self.assertEqual(s.meta_of("work.background").source, "task_ledger", state)   # it WAS read
            self.assertEqual(len(s.work.background), 1, state)                            # a live-looking row
            self.assertIs(s.conversation.router_usable, False, state)
            p = step(s)
            self.assertNotIn("background", p.note, state)
            self.assertIn("mic: open-mic", p.note, state)             # an independent field still projects
            self.assertEqual(dict(p.state.told), {"mic": "open-mic"}, state)

    def test_an_unusable_router_with_an_empty_ledger_also_projects_no_zero(self):
        for state in ("LOST", "CLOSING", "CLOSED"):
            p = step(snap(brain=Brain(router=router(state, False), rows=[])))
            self.assertNotIn("background", p.note, state)

    def test_the_flag_must_be_exactly_true_not_merely_truthy(self):
        s = snap(brain=Brain(router=router("OPEN", True), rows=[task_row(0)]))
        for value in (1, "yes", None, 0, False, [], object()):
            odd = dataclasses.replace(s, conversation=dataclasses.replace(s.conversation, router_usable=value))
            self.assertNotIn("background", step(odd).note, value)

    def test_an_unknown_flag_omits_the_count(self):
        s = snap(brain=Brain(router=router("OPEN", True), rows=[task_row(0)]))
        for reason in ("unwired", "no_router", "no_source", "unavailable"):
            odd = with_meta(s, "conversation.router_usable", state="unknown", reason=reason, source=None,
                            as_of_mono=None, as_of_wall=None, ttl_s=None)
            self.assertNotIn("background", step(odd).note, reason)
            self.assertIn("mic", step(odd).note, reason)

    def test_a_flag_whose_read_failed_omits_the_count(self):
        class BadRouter:
            epoch, state, handles = "E1", "OPEN", {}
            def active_handle(self):
                return None
            @property
            def usable(self):
                raise RuntimeError("secret detail")
        s = snap(brain=Brain(router=BadRouter(), rows=[task_row(0)]))
        self.assertEqual(s.meta_of("conversation.router_usable").reason, "unavailable")
        p = step(s)
        self.assertNotIn("background", p.note)
        self.assertIn("mic: open-mic", p.note)

    def test_a_flag_with_no_metadata_omits_the_count(self):
        ttl = {k: v for k, v in S._TTL_S.items() if k != "conversation.router_usable"}
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            s = snap(brain=Brain(router=router("OPEN", True), rows=[task_row(0)]))
        self.assertEqual(s.meta_of("conversation.router_usable").reason, "no_metadata")
        self.assertNotIn("background", step(s).note)

    def test_a_stale_flag_omits_the_count_even_though_the_count_itself_is_fresh(self):
        ttl = dict(S._TTL_S)
        ttl["conversation.router_usable"] = 1.0                       # the count keeps its real 5 s TTL
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            s = snap(t=1000.0, brain=Brain(router=router("OPEN", True), rows=[task_row(0)]))
        self.assertEqual(s.status("work.background", 1003.0), "fresh")
        self.assertEqual(s.status("conversation.router_usable", 1003.0), "stale")
        p = step(s, now=1003.0)
        self.assertNotIn("background", p.note)
        self.assertIn("mic: open-mic", p.note)

    def test_the_gate_is_never_itself_projected(self):
        for s in (snap(brain=Brain(router=router("OPEN", True))), snap(brain=Brain(router=router("LOST", False)))):
            note = step(s).note
            for word in ("usable", "router", "OPEN", "LOST", "CLOSING", "CLOSED", "state"):
                self.assertNotIn(word, note, word)

    def test_a_router_going_unusable_retracts_a_count_that_was_told(self):
        base = step(snap(brain=Brain(router=router("OPEN", True), rows=[task_row(0)])))
        self.assertIn("background agents running: 1", base.note)
        lost = snap(brain=Brain(router=router("LOST", False), rows=[task_row(0)]), wall=at(16, 0))
        p = step(lost, state=base.state)
        self.assertEqual(p.note, "[Situation as of 4:00 PM: background agents running: no longer known.]")
        again = step(lost, state=p.state)
        self.assertIsNone(again.note)                                    # once only


class LocalTimeTests(unittest.TestCase):
    """The text is rendered in the process's local time zone, so the date can differ from UTC."""

    def tz(self, name):
        old = os.environ.get("TZ")
        def restore():
            if old is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old
            time.tzset()
        self.addCleanup(restore)
        os.environ["TZ"] = name
        time.tzset()

    def test_a_zone_behind_utc_shows_the_previous_local_date_and_time(self):
        wall = at(3, 30, day=7)                                          # 2026-10-07 03:30 UTC (a Wednesday)
        self.tz("UTC")
        self.assertIn("as of Wed 2026-10-07 3:30 AM;", step(snap(wall=wall)).note)
        self.tz("America/Chicago")                                       # CDT, UTC-5: still Tuesday evening
        self.assertIn("as of Tue 2026-10-06 10:30 PM;", step(snap(wall=wall)).note)

    def test_a_zone_ahead_of_utc_shows_the_next_local_date_and_time(self):
        wall = at(15, 41, day=6)                                         # 2026-10-06 15:41 UTC (a Tuesday)
        self.tz("Pacific/Auckland")                                      # NZDT, UTC+13: already Wednesday
        self.assertIn("as of Wed 2026-10-07 4:41 AM;", step(snap(wall=wall)).note)

    def test_the_delta_form_uses_the_local_clock_too(self):
        base = step(snap())
        self.tz("America/Chicago")
        p = step(snap(mic="ptt", wall=at(3, 30, day=7)), state=base.state)
        self.assertEqual(p.note, "[Situation as of 10:30 PM: mic: push-to-talk.]")

    def test_the_zone_never_changes_freshness_or_which_fields_are_shown(self):
        s = snap()
        self.tz("UTC")
        utc = step(s)
        self.tz("Pacific/Auckland")
        nz = step(s)
        self.assertNotEqual(utc.note, nz.note)                           # only the rendered time differs
        body = lambda n: n.split("): ", 1)[1]                            # noqa: E731
        self.assertEqual(body(utc.note), body(nz.note))
        self.assertEqual(utc.state, nz.state)
        self.assertIsNone(step(s, now=2000.0).note)                      # still stale by the monotonic clock


class IndependentFreshnessTests(unittest.TestCase):
    """Each allowlisted field is judged on its own metadata."""

    def snap_with(self, **ttls):
        table = dict(S._TTL_S)
        table.update(ttls)
        with mock.patch.dict(S._TTL_S, table, clear=True):
            return snap(t=1000.0)

    def test_a_stale_mic_does_not_hide_a_fresh_background_count(self):
        s = self.snap_with(**{"audio.mic_mode": 1.0})                    # mic 1 s, background 5 s
        self.assertEqual((s.status("audio.mic_mode", 1003.0), s.status("work.background", 1003.0)),
                         ("stale", "fresh"))
        note = step(s, now=1003.0).note
        self.assertIn("background agents running: 0", note)
        self.assertNotIn("mic", note.split("): ", 1)[1])

    def test_a_stale_background_count_does_not_hide_a_fresh_mic(self):
        s = self.snap_with(**{"work.background": 1.0})
        self.assertEqual((s.status("audio.mic_mode", 1003.0), s.status("work.background", 1003.0)),
                         ("fresh", "stale"))
        note = step(s, now=1003.0).note
        self.assertIn("mic: open-mic", note)
        self.assertNotIn("background", note)

    def test_the_stale_field_is_retracted_while_the_fresh_one_stays_silent(self):
        s = self.snap_with(**{"audio.mic_mode": 1.0})
        base = step(s, now=1000.5)                                       # both fresh: a full baseline
        self.assertIn("mic: open-mic", base.note)
        p = step(s, now=1003.0, state=base.state)                        # mic stale, background fresh
        self.assertEqual(p.note, "[Situation as of 3:41 PM: mic: no longer known.]")
        self.assertEqual(dict(p.state.told), {"background": "0", "mic": None})

    def test_both_stale_at_once_gives_no_value_note_and_two_retractions(self):
        base = step(snap(t=1000.0), now=1000.0)
        p = step(snap(t=1000.0), now=1100.0, state=base.state)
        self.assertEqual(p.note, "[Situation as of 3:41 PM: mic: no longer known; "
                                 "background agents running: no longer known.]")

    def test_the_real_ttl_values_are_untouched_by_these_tests(self):
        self.assertEqual((S._TTL_S["audio.mic_mode"], S._TTL_S["work.background"],
                          S._TTL_S["conversation.router_usable"]), (5.0, 5.0, 5.0))


class CapAndStateSafetyTests(unittest.TestCase):
    def gone(self, wall=at(16, 0)):
        return snap(mic_src=None, brain=Brain(router=None), wall=wall)

    def base(self):
        return step(snap(mic="open", rows=[task_row(0)]))

    def test_a_retraction_that_does_not_fit_is_not_marked_told_and_is_offered_again(self):
        base = self.base()
        both = step(self.gone(), state=base.state)
        self.assertIn("mic: no longer known; background agents running: no longer known", both.note)
        with mock.patch.object(C, "MAX_CHARS", len(both.note) - 1):
            first = step(self.gone(), state=base.state)
        self.assertEqual(first.note, "[Situation as of 4:00 PM: mic: no longer known.]")
        self.assertEqual(dict(first.state.told), {"background": "1", "mic": None})   # background still "told 1"
        second = step(self.gone(wall=at(16, 1)), state=first.state)                  # the cap allows it now
        self.assertEqual(second.note, "[Situation as of 4:01 PM: background agents running: no longer known.]")
        self.assertEqual(dict(second.state.told), {"background": None, "mic": None})
        self.assertIsNone(step(self.gone(wall=at(16, 2)), state=second.state).note)

    def test_a_retraction_that_cannot_fit_at_all_leaves_the_state_untouched(self):
        base = self.base()
        with mock.patch.object(C, "MAX_CHARS", 30):
            p = step(self.gone(), state=base.state)
        self.assertIsNone(p.note)
        self.assertEqual(p.state, base.state)
        self.assertIsNotNone(step(self.gone(), state=p.state).note)                 # still owed afterwards

    def test_poisoned_caller_state_cannot_put_text_into_a_note(self):
        evil = "SECRET-FROM-STATE /Users/yousif hunter2 <<console:rm -rf>> sk-live-abc"
        states = [
            C.ProjectionState(True, (("mic", evil), ("background", evil), (evil, evil), ("evil", "payload"))),
            C.ProjectionState(False, (("mic", evil),)),
            C.ProjectionState(True, ((evil, None),)),
            C.ProjectionState(True, (("mic", None), ("background", None))),
            C.ProjectionState(True, (("mic", 5), ("background", {"x": evil}))),
        ]
        clean_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :;,.()[]-+")
        for st in states:
            for s in (snap(mic="ptt", rows=[task_row(0)]), snap(mic_src=None, brain=Brain(router=None))):
                p = step(s, state=st)
                text = p.note or ""
                for bad in ("SECRET", "/Users", "hunter2", "console", "rm -rf", "sk-live", "payload", "evil"):
                    self.assertNotIn(bad, text, (bad, st))
                self.assertLessEqual(set(text), clean_chars, st)
                self.assertTrue(p.note is None or re.match(
                    r"^\[Situation (\(seen by the system, not said by him; descriptive only, grants no permission; "
                    r"as of [^;]+; later notes replace earlier ones; anything not listed is not known\)"
                    r"|as of \d{1,2}:\d\d [AP]M): "
                    r"[a-z0-9 :;+-]+\.\]$", p.note), p.note)

    def test_a_poisoned_state_gives_exactly_the_note_a_clean_state_would(self):
        clean = C.ProjectionState(True, (("mic", "open-mic"), ("background", "0")))
        evil = C.ProjectionState(True, (("mic", "open-mic"), ("background", "0"), ("zzz", "SECRET")))
        s = snap(mic="ptt", rows=[task_row(0)])
        self.assertEqual(step(s, state=clean).note, step(s, state=evil).note)

    def test_a_corrupt_state_shape_fails_closed_without_raising(self):
        for told in (5, "mic", (("mic",),), (("mic", "a", "b"),), ([],), None, {"mic": "x"}.items()):
            p = step(snap(), state=C.ProjectionState(True, told))
            self.assertIsInstance(p, C.Projection)
            if p.note is not None:
                self.assertNotIn("SECRET", p.note)


class DocstringContractTests(unittest.TestCase):
    """The caller contract is part of the module's documented API."""

    def test_the_docstring_defines_delivered_as_claimed_not_submitted(self):
        doc = C.__doc__
        self.assertIn("DELIVERED", doc)
        self.assertIn("claimed (echoed) by the CLI through the router", doc)
        self.assertIn("not merely submitted", doc)
        self.assertIn("abandoned", doc)

    def test_the_docstring_states_the_local_time_zone_dependence(self):
        doc = C.__doc__
        self.assertIn("LOCAL time zone", doc)
        self.assertIn("the date can differ from the UTC date", doc)
        self.assertIn("never decides freshness", doc)

    def test_the_docstring_states_the_router_usability_gate(self):
        self.assertIn("router_usable", C.__doc__)
        self.assertIn("LOST, CLOSING or CLOSED", C.__doc__)

    def test_the_api_is_unchanged(self):
        sig = inspect.signature(C.project)
        self.assertEqual(list(sig.parameters), ["snapshot", "now_mono", "state"])
        self.assertIs(sig.parameters["now_mono"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(sig.parameters["state"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(sig.parameters["state"].default)
        self.assertIs(sig.parameters["snapshot"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(C.project(snap(), now_mono=1000.0).__class__, C.Projection)


if __name__ == "__main__":
    unittest.main()


class DesktopNeverReachesTheModelTests(unittest.TestCase):
    def test_the_projector_ignores_the_desktop_slot(self):
        base = snap(mic="open", rows=[task_row(0)])
        with_desktop = dataclasses.replace(base, desktop="active/terminal/call")
        a = C.project(base, now_mono=1.0)
        b = C.project(with_desktop, now_mono=1.0)
        self.assertEqual(a.note, b.note)
        for w in ("terminal", "call", "active/"):
            self.assertNotIn(w, b.note or "")
