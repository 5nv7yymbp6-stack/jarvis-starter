"""Phase 1 of the Situation / Context Engine (situation.py): a read-only snapshot.

Real owners (AddresseeGate, Ledger, EpochRouter) behind fakes only where the
owner needs a process (the SDK client, the mouth). Nothing here touches the live voice process."""
import asyncio
import copy
import dataclasses
import inspect
import pickle
import unittest
from types import SimpleNamespace

from backtalk import addressee as A
from backtalk import ledger as L
from backtalk import situation as S
from backtalk.situation import UNKNOWN
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_router import Base


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def gate(clock):
    return A.AddresseeGate(["jarvis", "hey jarvis"], clock=clock)


class Fake:
    """A stand-in for WarmBrain: only what the reader is allowed to touch."""
    def __init__(self, router=None, session_id=None, live=()):
        self.router, self.session_id, self._live = router, session_id, list(live)

    def live_background(self):
        return list(self._live)


def reader(**kw):
    return S.SituationReader(S.SituationSources(**kw))


class UnknownTests(unittest.TestCase):
    def test_unknown_is_a_singleton_and_survives_copy_and_pickle(self):
        self.assertIs(S._Unknown(), UNKNOWN)
        self.assertIs(copy.deepcopy(UNKNOWN), UNKNOWN)
        self.assertIs(copy.copy(UNKNOWN), UNKNOWN)
        self.assertIs(pickle.loads(pickle.dumps(UNKNOWN)), UNKNOWN)

    def test_unknown_cannot_be_used_as_a_boolean(self):
        with self.assertRaises(TypeError):
            bool(UNKNOWN)
        with self.assertRaises(TypeError):
            if UNKNOWN:
                pass

    def test_a_snapshot_with_nothing_wired_is_unknown_everywhere(self):
        snap = reader().snapshot()
        for name in ("audio", "conversation", "work", "device", "desktop", "location"):
            self.assertIs(getattr(snap, name), UNKNOWN, name)
        self.assertEqual(snap.unavailable, ())

    def test_nothing_is_ever_guessed_for_the_groups_with_no_owner(self):
        snap = reader(addressee=gate(Clock()), brain=Fake(),
                      mic=lambda: {"mode": "open", "btn": False}, zipped=lambda: False,
                      mouth=SimpleNamespace(busy=False)).snapshot()
        self.assertIs(snap.device, UNKNOWN)
        self.assertIs(snap.desktop, UNKNOWN)
        self.assertIs(snap.location, UNKNOWN)
        self.assertIs(snap.audio.voice_state, UNKNOWN)      # no public owner accessor
        self.assertIs(snap.audio.speaker, UNKNOWN)          # voice-lock is shadow-only
        self.assertIs(snap.conversation.router_usable, UNKNOWN)   # brain has no router here

    def test_partly_wired_audio_keeps_the_unwired_parts_unknown(self):
        snap = reader(zipped=lambda: True).snapshot()
        self.assertIs(snap.audio.zipped, True)
        self.assertIs(snap.audio.mic_mode, UNKNOWN)
        self.assertIs(snap.audio.mouth_busy, UNKNOWN)

    def test_a_source_that_raises_makes_its_group_unknown_and_is_reported(self):
        def boom():
            raise RuntimeError("secret detail that must not be copied")
        snap = reader(mic=boom, addressee=gate(Clock())).snapshot()
        self.assertIs(snap.audio, UNKNOWN)
        self.assertEqual(snap.unavailable, ("audio",))
        self.assertEqual(snap.conversation.addressee_state, "IDLE")   # other groups unaffected
        self.assertNotIn("secret", repr(snap))


class AudioTests(unittest.TestCase):
    def test_mic_zip_and_mouth_are_reflected(self):
        mic, zipped = {"mode": "open", "btn": False, "gen": 7}, {"on": False}
        mouth = SimpleNamespace(busy=False)
        rd = reader(mic=lambda: mic, zipped=lambda: zipped["on"], mouth=mouth)
        a = rd.snapshot().audio
        self.assertEqual((a.mic_mode, a.ptt_held, a.zipped, a.mouth_busy), ("open", False, False, False))
        mic.update(mode="ptt", btn=True)
        zipped["on"] = True
        mouth.busy = True
        a = rd.snapshot().audio
        self.assertEqual((a.mic_mode, a.ptt_held, a.zipped, a.mouth_busy), ("ptt", True, True, True))


class AddresseeTests(unittest.TestCase):
    def test_idle_active_and_lapse_are_represented_from_the_gate_itself(self):
        clk = Clock()
        g = gate(clk)
        rd = reader(addressee=g)
        c = rd.snapshot().conversation
        self.assertEqual((c.addressee_mode, c.addressee_state), ("wake", "IDLE"))
        self.assertEqual((c.addressee_window_s, c.addressee_max_active_s), (180.0, 300.0))
        g.open("wake")
        self.assertEqual(rd.snapshot().conversation.addressee_state, "ACTIVE")
        clk.t += 181                                     # idle window passes with no follow-up
        self.assertEqual(rd.snapshot().conversation.addressee_state, "IDLE")

    def test_always_mode_is_reported_as_such(self):
        g = A.AddresseeGate(["jarvis"], mode="always", clock=Clock())
        self.assertEqual(reader(addressee=g).snapshot().conversation.addressee_mode, "always")

    def test_reading_the_gate_changes_nothing_and_logs_nothing(self):
        clk = Clock()
        g = gate(clk)
        g.open("wake")
        clk.t += 400                                     # lapsed, but nobody has noticed yet
        before = (g._idle_until, g._hard_until, g._logged)
        logged = []
        real = A.log
        A.log = logged.append
        try:
            for _ in range(3):
                self.assertEqual(reader(addressee=g).snapshot().conversation.addressee_state, "IDLE")
        finally:
            A.log = real
        self.assertEqual((g._idle_until, g._hard_until, g._logged), before)
        self.assertEqual(logged, [])                     # the lapse is still the gate's to announce


class RouterBacked(Base):
    """Foreground / background / session / epoch, from a real router, ledger and fake client."""

    def brain(self, **kw):
        live = self.led.live_tasks(kind=L.KIND_BG_AGENT, epoch=self.r.epoch)
        return Fake(router=self.r, session_id=kw.pop("session_id", "sess-1"), live=live)

    def snap(self, **kw):
        return reader(brain=self.brain(**kw)).snapshot()

    async def test_session_and_epoch_are_reflected(self):
        s = self.snap(session_id="abc123")
        c = s.conversation
        self.assertEqual((c.session_id, c.epoch, c.epoch_state, c.router_usable),
                         ("abc123", self.r.epoch, "OPEN", True))

    async def test_no_conversation_id_is_known_none_not_unknown(self):
        self.assertIsNone(self.snap(session_id=None).conversation.session_id)

    async def test_a_new_epoch_replaces_the_old_one_and_a_lost_one_says_so(self):
        first = self.snap().conversation.epoch
        self.r.state = "LOST"
        c = self.snap().conversation
        self.assertEqual((c.epoch, c.epoch_state, c.router_usable), (first, "LOST", False))
        client2 = F.FakeClient()
        r2 = await self.mgr.open(client2, session_id="s2", cli_pid=None)
        self.extra.append(r2)
        c = reader(brain=Fake(router=r2, session_id="s2")).snapshot().conversation
        self.assertNotEqual(c.epoch, first)
        self.assertEqual((c.epoch_state, c.router_usable), ("OPEN", True))

    async def test_no_router_means_known_no_epoch_and_known_no_work(self):
        s = reader(brain=Fake(router=None, session_id="x")).snapshot()
        self.assertIsNone(s.conversation.epoch)
        self.assertIs(s.conversation.epoch_state, UNKNOWN)
        self.assertEqual((s.work.foreground, s.work.active_turn_id, s.work.background), ((), None, ()))

    async def test_an_idle_router_has_no_foreground_or_background_work(self):
        w = self.snap().work
        self.assertEqual((w.foreground, w.active_turn_id, w.background), ((), None, ()))

    async def test_a_sent_turn_is_in_flight_but_not_yet_the_active_run(self):
        h = await self.r.send("hello there")
        w = self.snap().work
        self.assertEqual(len(w.foreground), 1)
        t = w.foreground[0]
        self.assertEqual((t.turn_id, t.kind, t.state, t.silent), (h.turn_id, "foreground", L.T_SENT, False))
        self.assertGreaterEqual(t.age_s, 0.0)
        self.assertIsNone(w.active_turn_id)

    async def test_a_running_turn_is_the_active_one_and_a_finished_turn_disappears(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), F.delta("working"))
        await settle()
        w = self.snap().work
        self.assertEqual(w.active_turn_id, h.turn_id)
        self.assertEqual(w.foreground[0].state, h.state)
        self.client.feed(F.asst("working"), F.result("working"))
        await settle()
        w = self.snap().work
        self.assertEqual((w.foreground, w.active_turn_id), ((), None))

    async def test_a_live_background_task_is_represented_and_a_finished_one_is_not(self):
        h = await self.r.send("start a job")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1", "build the thing"),
                         F.result("started"))
        await settle()
        w = self.snap().work
        self.assertEqual(len(w.background), 1)
        t = w.background[0]
        self.assertEqual((t.task_id, t.kind, t.epoch), ("tk1", L.KIND_BG_AGENT, self.r.epoch))
        self.assertIn(t.state, L.TASK_LIVE)
        self.assertEqual(w.foreground, ())                  # background work is not foreground work
        self.assertNotIn("build the thing", repr(w))        # the description stays in the ledger
        self.led.task_state("tk1", L.K_COMPLETED)
        self.assertEqual(self.snap().work.background, ())

    async def test_foreground_and_background_are_reported_independently(self):
        h = await self.r.send("start a job")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.result("started"))
        await settle()
        h2 = await self.r.send("what else")
        w = self.snap().work
        self.assertEqual([t.turn_id for t in w.foreground], [h2.turn_id])
        self.assertEqual([t.task_id for t in w.background], ["tk1"])


class ReadOnlyTests(RouterBacked):
    """Taking a snapshot changes nothing in any owner, and the snapshot cannot change them."""

    async def test_taking_snapshots_leaves_every_owner_unchanged(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        clk = Clock()
        g = gate(clk)
        g.open("typed")
        mic, zipped = {"mode": "open", "btn": False, "gen": 3}, {"on": False}
        owners = lambda: (  # noqa: E731
            dict(mic), dict(zipped), g._idle_until, g._hard_until, g._logged,
            self.r.state, self.r.epoch, list(self.r.handles), {u: x.state for u, x in self.r.handles.items()},
            self.r.stats.copy(), len(self.r.reports), dict(self.led.tasks["tk1"]),
            len(self.led.turns), len(self.led.actions), self.r._last_stale_interrupt)
        size = self.led_size()
        before = owners()
        rd = reader(mic=lambda: mic, zipped=lambda: zipped["on"], mouth=SimpleNamespace(busy=False),
                    addressee=g, brain=self.brain())
        for _ in range(10):
            rd.snapshot()
        self.assertEqual(owners(), before)
        self.assertEqual(self.led_size(), size)             # nothing was appended to the ledger file

    def led_size(self):
        import os
        return os.path.getsize(os.path.join(self.tmp.name, "ledger.jsonl"))

    async def test_a_snapshot_is_frozen_all_the_way_down(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        snap = reader(addressee=gate(Clock()), brain=self.brain(),
                      mic=lambda: {"mode": "open", "btn": False}).snapshot()

        def frozen(obj):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, "mutated", 1)
        frozen(snap)
        for grp in (snap.audio, snap.conversation, snap.work, snap.work.foreground[0],
                    snap.work.background[0]):
            frozen(grp)
        self.assertIsInstance(snap.work.foreground, tuple)
        self.assertIsInstance(snap.work.background, tuple)
        self.assertIsInstance(snap.unavailable, tuple)

    async def test_nothing_in_a_snapshot_refers_to_a_live_owner(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        snap = reader(addressee=gate(Clock()),
                      brain=self.brain(), mic=lambda: {"mode": "ptt", "btn": False}).snapshot()
        owners = (self.r, self.led, self.client, *self.r.handles.values())
        allowed = (str, float, int, bool, type(None), tuple, type(UNKNOWN))

        def walk(o):
            if dataclasses.is_dataclass(o):
                for f in dataclasses.fields(o):
                    walk(getattr(o, f.name))
            else:
                self.assertIsInstance(o, allowed, repr(o))
                self.assertFalse(any(o is x for x in owners))
                if isinstance(o, tuple):
                    for i in o:
                        walk(i)
        walk(snap)

    async def test_mutating_what_a_snapshot_exposes_cannot_reach_the_owner(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        snap = reader(brain=self.brain()).snapshot()
        row_before = dict(self.led.tasks["tk1"])
        copy.copy(snap.work.background[0])                  # a copy is a different object anyway
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.work.background[0].state = "COMPLETED"
        with self.assertRaises(TypeError):
            snap.work.background[0] = None                  # tuples do not assign
        self.assertEqual(self.led.tasks["tk1"], row_before)

    async def test_the_snapshot_survives_the_owner_moving_on(self):
        mic = {"mode": "open", "btn": False}
        rd = reader(mic=lambda: mic)
        old = rd.snapshot()
        mic["mode"] = "ptt"
        self.assertEqual(old.audio.mic_mode, "open")        # a snapshot is a moment, not a view
        self.assertEqual(rd.snapshot().audio.mic_mode, "ptt")


class SourceDisciplineTests(unittest.TestCase):
    def test_the_module_reaches_for_no_private_owner_state(self):
        src = inspect.getsource(S)
        body = src.split("class SituationReader", 1)[1]
        for private in ("._idle_until", "._hard_until", "._logged", "._clock", "._need_rebuild",
                        "._lock", "._table", "._inputs", "._pump_task", "._handles", "._last_"):
            self.assertNotIn(private, body, private)

    def test_the_module_imports_no_owner_and_writes_nothing(self):
        import ast
        tree = ast.parse(inspect.getsource(S))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module)
        self.assertEqual(imported, {"time", "dataclasses"})        # no owner, no I/O library
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        attrs = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(called & {"open", "log", "print", "exec", "eval", "setattr", "delattr"}, called)
        self.assertFalse(attrs & {"write", "submit", "open", "end", "request", "send", "interrupt",
                                  "put", "add", "update", "pop", "clear", "set"}, attrs)


# =====================================================================================
# PHASE 2: timestamps, provenance, TTL, staleness, and the passive [sit] line.
# =====================================================================================

import re

LEAVES = (
    "audio.mic_mode", "audio.ptt_held", "audio.zipped", "audio.mouth_busy", "audio.voice_state",
    "audio.speaker",
    "conversation.addressee_mode", "conversation.addressee_state", "conversation.addressee_window_s",
    "conversation.addressee_max_active_s", "conversation.session_id", "conversation.epoch",
    "conversation.epoch_state", "conversation.router_usable", "conversation.rebuild_pending",
    "work.foreground", "work.active_turn_id", "work.background",
    "device", "desktop", "location")

# Written out independently of the module, so a wrong or name-derived table fails here.
EXPECTED_SOURCE = {
    "audio.mic_mode": "mic_owner", "audio.ptt_held": "mic_owner", "audio.zipped": "zip_owner",
    "audio.mouth_busy": "mouth",
    "conversation.addressee_mode": "addressee_gate", "conversation.addressee_state": "addressee_gate",
    "conversation.addressee_window_s": "addressee_gate",
    "conversation.addressee_max_active_s": "addressee_gate",
    "conversation.session_id": "brain", "conversation.epoch": "router",
    "conversation.epoch_state": "router", "conversation.router_usable": "router",
    "work.foreground": "router", "work.active_turn_id": "router", "work.background": "task_ledger",
    "desktop": "presence_monitor"}
EXPECTED_TTL = {
    "audio.ptt_held": 1.0, "audio.mouth_busy": 1.0, "work.foreground": 2.0, "work.active_turn_id": 2.0,
    "audio.mic_mode": 5.0, "audio.zipped": 5.0, "conversation.addressee_state": 5.0,
    "conversation.epoch_state": 5.0, "conversation.router_usable": 5.0, "work.background": 5.0,
    "conversation.epoch": 10.0, "conversation.addressee_mode": 30.0,
    "conversation.addressee_window_s": 30.0, "conversation.addressee_max_active_s": 30.0,
    "conversation.session_id": 30.0, "desktop": 15.0}
NO_SOURCE = ("audio.voice_state", "audio.speaker", "conversation.rebuild_pending", "device",
             "location")


def idle_router():
    """The router's public surface for an idle, open epoch (no handles, nothing running)."""
    return SimpleNamespace(epoch="E1", state="OPEN", usable=True, handles={}, active_handle=lambda: None)


def full(clk=None, wall=None, lines=None, **over):
    """Every group wired to simple fakes (an idle open router, an idle gate)."""
    kw = dict(addressee=gate(clk or Clock()),
              brain=Fake(router=idle_router(), session_id="sess-1"),
              mic=lambda: {"mode": "open", "btn": False}, zipped=lambda: False,
              mouth=SimpleNamespace(busy=False), clock=clk or Clock(),
              wall_clock=wall or Clock(1.7e9), desktop=lambda: "active/terminal")
    if lines is not None:
        kw["observer"] = lines.append
    kw.update(over)
    return reader(**kw)


class MetaShapeTests(unittest.TestCase):
    def test_every_field_has_exactly_one_meta_in_a_fixed_order(self):
        for snap in (full().snapshot(), reader().snapshot()):
            self.assertEqual(tuple(m.path for m in snap.meta), LEAVES)

    def test_meta_is_a_tuple_of_frozen_records(self):
        snap = full().snapshot()
        self.assertIsInstance(snap.meta, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.meta[0].ttl_s = 999.0
        with self.assertRaises(TypeError):
            snap.meta[0] = None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.meta = ()

    def test_meta_of_unknown_path_raises_rather_than_guessing(self):
        with self.assertRaises(KeyError):
            full().snapshot().meta_of("audio.nonsense")

    def test_phase_1_values_are_unchanged_by_the_metadata(self):
        snap = full().snapshot()
        self.assertEqual((snap.audio.mic_mode, snap.conversation.addressee_state,
                          snap.conversation.session_id), ("open", "IDLE", "sess-1"))


class TimestampTests(unittest.TestCase):
    def test_known_fields_carry_the_monotonic_and_wall_time_of_the_read(self):
        clk, wall = Clock(500.0), Clock(1.7e9)
        snap = full(clk, wall).snapshot()
        self.assertEqual((snap.taken_at_mono, snap.taken_at_wall), (500.0, 1.7e9))
        for m in snap.meta:
            if m.state == "known":
                self.assertEqual((m.as_of_mono, m.as_of_wall), (500.0, 1.7e9), m.path)

    def test_unknown_fields_carry_no_time_and_no_ttl(self):
        for m in reader().snapshot().meta:
            self.assertEqual((m.state, m.source, m.as_of_mono, m.as_of_wall, m.ttl_s),
                             ("unknown", None, None, None, None), m.path)

    def test_each_snapshot_is_stamped_at_its_own_read_time(self):
        clk = Clock(10.0)
        rd = full(clk)
        a = rd.snapshot()
        clk.t = 14.0
        b = rd.snapshot()
        self.assertEqual((a.meta_of("audio.mic_mode").as_of_mono, b.meta_of("audio.mic_mode").as_of_mono),
                         (10.0, 14.0))


class ProvenanceTests(unittest.TestCase):
    def test_the_source_of_every_known_field_is_the_explicit_owner(self):
        snap = full().snapshot()
        got = {m.path: m.source for m in snap.meta if m.state == "known"}
        self.assertEqual(got, EXPECTED_SOURCE)

    def test_provenance_is_not_inferred_from_the_field_name(self):
        snap = full().snapshot()
        self.assertEqual(snap.meta_of("conversation.session_id").source, "brain")      # not "conversation"
        self.assertEqual(snap.meta_of("conversation.epoch").source, "router")
        self.assertEqual(snap.meta_of("work.background").source, "task_ledger")        # not "work"/"brain"
        self.assertEqual(snap.meta_of("audio.zipped").source, "zip_owner")             # not "audio"
        self.assertEqual(snap.meta_of("audio.mic_mode").source, "mic_owner")

    def test_every_known_source_is_a_named_owner_and_unknown_fields_have_none(self):
        for m in full().snapshot().meta:
            if m.state == "known":
                self.assertRegex(m.source, r"^[a-z_]+(\+[a-z_]+)*$")
            else:
                self.assertIsNone(m.source)


class UnknownVersusStaleTests(unittest.TestCase):
    def test_unwired_sources_are_unknown_with_reason_unwired(self):
        snap = reader().snapshot()
        for p in LEAVES:
            if p not in NO_SOURCE:
                self.assertEqual((snap.meta_of(p).state, snap.meta_of(p).reason), ("unknown", "unwired"), p)

    def test_fields_with_no_owner_are_unknown_even_when_everything_is_wired(self):
        snap = full().snapshot()
        for p in NO_SOURCE:
            m = snap.meta_of(p)
            self.assertEqual((m.state, m.reason, m.source), ("unknown", "no_source", None), p)
        self.assertIs(snap.device, UNKNOWN)
        self.assertIs(snap.conversation.rebuild_pending, UNKNOWN)

    def test_a_failed_read_is_unknown_with_reason_unavailable(self):
        def boom():
            raise RuntimeError("x")
        snap = full(mic=boom).snapshot()
        self.assertEqual((snap.meta_of("audio.mic_mode").state, snap.meta_of("audio.mic_mode").reason),
                         ("unknown", "unavailable"))
        self.assertEqual(snap.meta_of("audio.zipped").reason, "unavailable")        # the whole group
        self.assertEqual(snap.meta_of("conversation.addressee_state").state, "known")  # others unaffected

    def test_unknown_never_becomes_stale_and_has_no_age(self):
        snap = reader().snapshot()
        for now in (0.0, 1e3, 1e9):
            self.assertEqual(snap.status("audio.mic_mode", now), "unknown")
            self.assertIsNone(snap.age_s("audio.mic_mode", now))
        self.assertEqual(snap.stale_paths(1e9), ())

    def test_a_known_value_past_its_ttl_is_stale_and_keeps_its_value(self):
        clk = Clock(100.0)
        snap = full(clk).snapshot()
        self.assertEqual(snap.status("audio.mic_mode", 100.0), "fresh")
        self.assertEqual(snap.status("audio.mic_mode", 106.0), "stale")
        self.assertEqual(snap.audio.mic_mode, "open")             # stale is a judgment; the value is not replaced
        self.assertIs(snap.device, UNKNOWN)

    def test_stale_and_unknown_are_three_different_statuses(self):
        snap = full(Clock(1000.0)).snapshot()
        self.assertEqual(snap.status("audio.mic_mode", 1000.0), "fresh")
        self.assertEqual(snap.status("audio.mic_mode", 1006.0), "stale")
        self.assertEqual(snap.status("device", 1000.0), "unknown")
        self.assertEqual(snap.status("device", 1006.0), "unknown")        # time never turns unknown into stale


class TtlTests(unittest.TestCase):
    def test_the_ttl_values_are_the_documented_conservative_ones(self):
        snap = full().snapshot()
        got = {m.path: m.ttl_s for m in snap.meta if m.state == "known"}
        self.assertEqual(got, EXPECTED_TTL)
        self.assertTrue(all(0 < t <= 30.0 for t in got.values()))

    def test_boundaries_age_equal_to_ttl_is_fresh_and_just_over_is_stale(self):
        clk = Clock(0.0)
        snap = full(clk).snapshot()
        for path, ttl in EXPECTED_TTL.items():
            self.assertEqual(snap.status(path, 0.0), "fresh", path)
            self.assertEqual(snap.status(path, ttl - 1e-9), "fresh", path)
            self.assertEqual(snap.status(path, ttl), "fresh", path)               # exactly at the TTL
            self.assertEqual(snap.status(path, ttl + 1e-6), "stale", path)

    def test_a_negative_age_is_stale_never_fresh(self):
        snap = full(Clock(100.0)).snapshot()
        self.assertEqual(snap.status("audio.mic_mode", 99.0), "stale")
        self.assertEqual(snap.age_s("audio.mic_mode", 99.0), -1.0)

    def test_stale_paths_order_matches_the_ttl_ladder(self):
        snap = full(Clock(0.0)).snapshot()
        self.assertEqual(snap.stale_paths(0.5), ())
        self.assertEqual(set(snap.stale_paths(1.5)), {"audio.ptt_held", "audio.mouth_busy"})
        self.assertIn("work.foreground", snap.stale_paths(2.5))
        self.assertNotIn("audio.mic_mode", snap.stale_paths(2.5))
        self.assertIn("audio.mic_mode", snap.stale_paths(5.5))
        self.assertNotIn("conversation.epoch", snap.stale_paths(5.5))
        self.assertEqual(set(snap.stale_paths(31.0)), set(EXPECTED_TTL))

    def test_age_is_now_minus_the_read_time(self):
        snap = full(Clock(40.0)).snapshot()
        self.assertEqual(snap.age_s("work.foreground", 43.5), 3.5)


class MonotonicTests(unittest.TestCase):
    def test_freshness_ignores_the_wall_clock_entirely(self):
        clk, wall = Clock(100.0), Clock(1.7e9)
        snap = full(clk, wall).snapshot()
        before = [snap.status(p, 102.0) for p in EXPECTED_TTL]
        wall.t = 0.0                                   # the wall clock jumps back to 1970
        self.assertEqual([snap.status(p, 102.0) for p in EXPECTED_TTL], before)
        wall.t = 9e12
        self.assertEqual([snap.status(p, 102.0) for p in EXPECTED_TTL], before)

    def test_a_wall_clock_that_moves_backwards_between_reads_cannot_make_a_field_stale(self):
        clk, wall = Clock(10.0), Clock(2e9)
        rd = full(clk, wall)
        a = rd.snapshot()
        wall.t = 1e9                                   # an NTP step backwards
        clk.t = 10.5
        b = rd.snapshot()
        self.assertGreater(a.taken_at_wall, b.taken_at_wall)
        self.assertEqual(b.status("audio.ptt_held", 10.5), "fresh")
        self.assertEqual(a.status("audio.ptt_held", 10.5), "fresh")   # age 0.5 s on the monotonic clock

    def test_status_takes_the_monotonic_time_from_the_caller_and_has_no_default(self):
        snap = full().snapshot()
        with self.assertRaises(TypeError):
            snap.status("audio.mic_mode")
        src = inspect.getsource(S.Snapshot)
        self.assertNotIn("time.time", src)
        self.assertNotIn("as_of_wall", src.split("def meta_of", 1)[1])      # no judgment reads it

    def test_the_reader_judges_with_the_clock_it_was_given_not_the_real_one(self):
        snap = full(Clock(5.0)).snapshot()
        self.assertEqual(snap.meta_of("audio.mic_mode").as_of_mono, 5.0)    # a fake, not time.monotonic()


class SitLineTests(RouterBacked):
    SECRETS = ("SECRET-TASK-DESC", "rm -rf /home/secret", "sess-secret-123", "delete:secret-key",
               "/Users/yousif", "hunter2", "sk-live-abc")

    async def poisoned(self):
        h = await self.r.send("please do hunter2 sk-live-abc /Users/yousif/x")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk9", "toolu_bg9", "SECRET-TASK-DESC"),
                         F.delta("x"))
        await settle()
        lines = []
        rd = reader(addressee=gate(Clock()),
                    brain=Fake(router=self.r, session_id="sess-secret-123",
                               live=self.led.live_tasks(kind=L.KIND_BG_AGENT, epoch=self.r.epoch)),
                    mic=lambda: {"mode": "open", "btn": False}, zipped=lambda: False,
                    mouth=SimpleNamespace(busy=False), observer=lines.append)
        return rd, lines

    async def test_the_line_carries_no_value_id_text_path_or_secret(self):
        rd, lines = await self.poisoned()
        snap = rd.snapshot()
        self.assertEqual(len(lines), 1)
        line = lines[0]
        for secret in self.SECRETS + (self.r.epoch, "tk9", "toolu_bg9", "tuSecret", "Bash"):
            self.assertNotIn(secret, line)
        self.assertNotIn("/", line)
        self.assertNotIn("sess-", line)

    async def test_the_line_matches_a_strict_whitelist_shape(self):
        rd, lines = await self.poisoned()
        rd.snapshot()
        pat = (r"^\[sit\] known=\d+ unknown=\d+ stale=\d+ unavailable=[a-z_,]+|- "
               r"sources=[a-z_+:,\d]+|- unknown_fields=[a-z_.,()]+|- stale_fields=[a-z_.,]+|- ttl_s=[\d.\-]+|-$")
        strict = re.compile(r"^\[sit\] known=\d+ unknown=\d+ stale=\d+ unavailable=\S+ sources=\S+ "
                            r"unknown_fields=\S+ stale_fields=\S+ ttl_s=\S+$")
        self.assertRegex(lines[0], strict)
        self.assertNotRegex(lines[0], r"[^a-z0-9_.,:()=+\- \[\]]")        # nothing outside the safe alphabet

    async def test_the_line_reports_counts_sources_and_unknown_reasons(self):
        lines = []
        full(lines=lines).snapshot()
        line = lines[0]
        self.assertIn("known=16 unknown=5 stale=0 unavailable=-", line)
        self.assertIn("addressee_gate:4", line)
        self.assertIn("router:", line)
        self.assertIn("device(no_source)", line)
        self.assertIn("conversation.rebuild_pending(no_source)", line)
        self.assertIn("stale_fields=- ttl_s=1-30", line)

    async def test_a_held_snapshot_judged_late_reports_its_stale_fields_by_name(self):
        snap = full(Clock(0.0)).snapshot()
        fresh, late = S.sit_line(snap, 0.5), S.sit_line(snap, 3.0)
        self.assertIn("stale=0 ", fresh)
        self.assertIn("stale_fields=- ", fresh)
        self.assertIn("stale=4 ", late)                          # the 1 s and 2 s fields, past their TTL at 3.0 s
        self.assertIn("stale_fields=audio.ptt_held,audio.mouth_busy,work.foreground,work.active_turn_id ", late)
        self.assertRegex(S.sit_line(snap, 100.0), r"stale=16 ")             # every known field
        self.assertRegex(S.sit_line(snap, 100.0), r"unknown=5 ")            # unknown does not become stale

    async def test_unwired_and_unavailable_show_up_by_name_only(self):
        lines = []
        def boom():
            raise RuntimeError("secret detail")
        reader(mic=boom, observer=lines.append).snapshot()
        self.assertIn("unavailable=audio", lines[0])
        self.assertIn("audio.mic_mode(unavailable)", lines[0])
        self.assertNotIn("secret detail", lines[0])

    async def test_one_line_per_snapshot_and_none_without_an_observer(self):
        lines = []
        rd = full(lines=lines)
        for _ in range(7):
            rd.snapshot()
        self.assertEqual(len(lines), 7)
        self.assertEqual(len(set(lines)), 1)                        # same situation, same line
        full().snapshot()                                          # no observer: nothing to call, nothing raised

    async def test_an_observer_that_raises_never_costs_the_read(self):
        def bad(_line):
            raise RuntimeError("log failed")
        snap = full(observer=bad).snapshot()
        self.assertEqual(snap.audio.mic_mode, "open")

    async def test_the_name_filter_hides_anything_that_is_not_a_static_identifier(self):
        for bad in ("a/b", "has space", "UPPER", "x=1", "", "naïve", "/Users/y"):
            self.assertEqual(S._name(bad), "?", bad)
        self.assertEqual(S._name("audio.mic_mode"), "audio.mic_mode")
        self.assertEqual(S._name("mic_owner+zip_owner"), "mic_owner+zip_owner")


class SideEffectTests(RouterBacked):
    async def test_repeated_observed_reads_change_no_owner_and_write_nothing(self):
        h = await self.r.send("go")
        self.client.feed(F.init(), F.echo(h.uuid), *F.launch_bg("tk1", "toolu_bg1"), F.delta("x"))
        await settle()
        clk = Clock()
        g = gate(clk)
        g.open("typed")
        mic, zipped = {"mode": "open", "btn": False, "gen": 3}, {"on": False}
        owners = lambda: (  # noqa: E731
            dict(mic), dict(zipped), g._idle_until, g._hard_until, g._logged, self.r.state,
            self.r.epoch, list(self.r.handles), {u: x.state for u, x in self.r.handles.items()},
            self.r.stats.copy(), len(self.r.reports), dict(self.led.tasks["tk1"]), len(self.led.turns),
            len(self.led.actions), self.r._last_stale_interrupt, self.r._last_submit)
        size, before, lines = self.led_size(), owners(), []
        rd = reader(mic=lambda: mic, zipped=lambda: zipped["on"], mouth=SimpleNamespace(busy=False),
                    addressee=g, brain=self.brain(),
                    observer=lines.append)
        snaps = [rd.snapshot() for _ in range(25)]
        self.assertEqual(owners(), before)
        self.assertEqual(self.led_size(), size)
        self.assertEqual(len(lines), 25)
        shape = lambda sn: tuple((m.path, m.state, m.source, m.reason, m.ttl_s) for m in sn.meta)  # noqa: E731
        self.assertEqual({shape(sn) for sn in snaps}, {shape(snaps[0])})      # same situation, same metadata

    def led_size(self):
        import os
        return os.path.getsize(os.path.join(self.tmp.name, "ledger.jsonl"))

    async def test_observed_reads_do_not_advance_the_addressee_timers_or_log_its_lapse(self):
        clk = Clock()
        g = gate(clk)
        g.open("wake")
        clk.t += 400
        before = (g._idle_until, g._hard_until, g._logged)
        logged = []
        real = A.log
        A.log = logged.append
        try:
            lines = []
            rd = reader(addressee=g, observer=lines.append)
            for _ in range(3):
                rd.snapshot()
        finally:
            A.log = real
        self.assertEqual((g._idle_until, g._hard_until, g._logged), before)
        self.assertEqual(logged, [])

    async def test_the_snapshot_is_still_frozen_with_metadata_attached(self):
        snap = full().snapshot()
        for obj in (snap, snap.meta[0], snap.audio, snap.conversation):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, "mutated", 1)


def snapshot_call_sites(source: str):
    """Every `<receiver>.snapshot()` call in `source`, as sorted (enclosing function, receiver) pairs.
    The enclosing function is the NEAREST def around the call; a call outside any def is "<module>"."""
    import ast
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "snapshot":
            owner, cur = "<module>", node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner = cur.name
                    break
            sites.append((owner, ast.unparse(node.func.value)))
    return sorted(sites)


class MainWiringTests(unittest.TestCase):
    def setUp(self):
        from backtalk import main
        self.main = main
        self.src = inspect.getsource(main)

    def test_the_reader_is_wired_with_the_log_as_its_observer_and_called_only_by_the_passive_helper(self):
        self.assertIn("observer=log", self.src)
        # A CLOSED list of snapshot call sites in main.py, by the function that contains each one: the
        # telemetry reader's passive helper (Phase 2) and the shadow projection's begin step (Phase 3B).
        # Any other `.snapshot()` call, anywhere (module level, handle(), amain(), speak_reply(), a new
        # helper), changes this list and fails the test.
        self.assertEqual(snapshot_call_sites(self.src),
                         [("_observe_situation", "_SITUATION"), ("_sitctx_begin", "reader")])
        helper = inspect.getsource(self.main._observe_situation)
        self.assertIn("_SITUATION.snapshot()", helper)
        sites = [ln for ln in self.src.splitlines() if ln.strip().startswith("_observe_situation()")]
        self.assertEqual(len(sites), 1)                                     # one call site: once per turn

    def test_the_call_site_guard_catches_a_new_snapshot_call_anywhere(self):
        base = self.src
        for extra in ("\n_SITUATION.snapshot()\n",                                   # module level
                      "\ndef _new_helper():\n    return _SITUATION.snapshot()\n",      # a new helper
                      "\nasync def handle2():\n    x = reader.snapshot()\n",            # another function
                      "\ndef _observe_situation2():\n    _SITUATION.snapshot()\n"):    # a look-alike name
            self.assertNotEqual(snapshot_call_sites(base + extra), snapshot_call_sites(base), extra)
        moved = base.replace("_SITUATION.snapshot()", "pass", 1) + "\ndef _elsewhere():\n    _SITUATION.snapshot()\n"
        self.assertNotEqual(snapshot_call_sites(moved), snapshot_call_sites(base))      # a call MOVED out of its helper
        self.assertEqual(snapshot_call_sites(base + "\ndef f():\n    pass\n"), snapshot_call_sites(base))

    def test_the_snapshot_result_is_discarded_and_never_reaches_the_model(self):
        self.assertRegex(self.src, r"\n\s+_observe_situation\(\)\s+# passive")   # a bare statement
        self.assertNotRegex(self.src, r"=\s*_SITUATION\.snapshot\(\)|\(\s*_SITUATION\.snapshot")
        for name in ("brain.py", "brief.py", "config.py", "router.py", "speech_policy.py"):
            path = __import__("os").path.join(__import__("os").path.dirname(self.main.__file__), name)
            self.assertNotIn("situation", open(path).read().lower(), name)

    def test_the_passive_helper_never_raises_and_does_nothing_when_unwired(self):
        saved = self.main._SITUATION
        try:
            self.main._SITUATION = None
            self.main._observe_situation()
            self.main._SITUATION = SimpleNamespace(snapshot=lambda: (_ for _ in ()).throw(RuntimeError("x")))
            self.main._observe_situation()                                  # swallowed
            seen = []
            self.main._SITUATION = SimpleNamespace(snapshot=lambda: seen.append(1))
            self.main._observe_situation()
            self.assertEqual(seen, [1])
        finally:
            self.main._SITUATION = saved


# =====================================================================================
# PHASE 2 HARDENING: accurate unknown reasons, fail-closed metadata, observer failure signal.
# =====================================================================================

from unittest import mock


def brain_without_router(**kw):
    return Fake(router=None, session_id=kw.pop("session_id", "sess-1"), **kw)


class RouterAbsentReasonTests(unittest.TestCase):
    def test_a_wired_brain_with_no_router_says_no_router_not_unwired(self):
        snap = full(brain=brain_without_router()).snapshot()
        for p in ("conversation.epoch_state", "conversation.router_usable"):
            m = snap.meta_of(p)
            self.assertEqual((m.state, m.reason, m.source, m.as_of_mono, m.ttl_s),
                             ("unknown", "no_router", None, None, None), p)
        self.assertIs(snap.conversation.epoch_state, UNKNOWN)          # the value is still not invented

    def test_known_empty_state_keeps_its_value_and_credits_the_brain_that_was_read(self):
        snap = full(brain=brain_without_router()).snapshot()
        self.assertIsNone(snap.conversation.epoch)
        self.assertEqual((snap.work.foreground, snap.work.active_turn_id, snap.work.background),
                         ((), None, ()))                                  # values unchanged from Phase 1
        for p in ("conversation.epoch", "work.foreground", "work.active_turn_id", "work.background"):
            m = snap.meta_of(p)
            self.assertEqual((m.state, m.source), ("known", "brain"), p)
            self.assertEqual(m.ttl_s, EXPECTED_TTL[p], p)                 # TTLs untouched
        self.assertEqual(snap.meta_of("conversation.session_id").source, "brain")

    def test_a_genuinely_unwired_brain_is_still_unwired(self):
        snap = reader(addressee=gate(Clock())).snapshot()
        for p in ("conversation.epoch", "conversation.epoch_state", "conversation.router_usable",
                  "conversation.session_id", "work.foreground", "work.active_turn_id", "work.background"):
            m = snap.meta_of(p)
            self.assertEqual((m.state, m.reason), ("unknown", "unwired"), p)

    def test_the_three_situations_are_distinguishable(self):
        unwired = reader().snapshot().meta_of("conversation.epoch_state")
        absent = full(brain=brain_without_router()).snapshot().meta_of("conversation.epoch_state")
        present = full().snapshot().meta_of("conversation.epoch_state")
        self.assertEqual((unwired.reason, absent.reason, present.reason), ("unwired", "no_router", None))
        self.assertEqual((unwired.state, absent.state, present.state), ("unknown", "unknown", "known"))
        self.assertEqual(present.source, "router")

    def test_a_failed_group_is_unavailable_not_no_router(self):
        class Boom(Fake):
            @property
            def session_id(self):
                raise RuntimeError("secret detail")
            @session_id.setter
            def session_id(self, v):
                pass
        snap = full(brain=Boom(router=None)).snapshot()
        self.assertEqual(snap.meta_of("conversation.epoch_state").reason, "unavailable")
        self.assertEqual(snap.meta_of("conversation.epoch").reason, "unavailable")
        self.assertEqual(snap.meta_of("work.foreground").state, "known")       # the work group read fine

    def test_the_router_coming_back_restores_the_router_provenance(self):
        brain = brain_without_router()
        rd = full(brain=brain)
        self.assertEqual(rd.snapshot().meta_of("conversation.epoch").source, "brain")
        brain.router = idle_router()
        snap = rd.snapshot()
        self.assertEqual(snap.meta_of("conversation.epoch").source, "router")
        self.assertEqual(snap.meta_of("conversation.epoch_state").state, "known")

    def test_no_router_never_becomes_stale_and_known_empty_still_ages_normally(self):
        snap = full(brain=brain_without_router(), clk=Clock(0.0)).snapshot()
        self.assertEqual(snap.status("conversation.epoch_state", 1e6), "unknown")
        self.assertIsNone(snap.age_s("conversation.router_usable", 1e6))
        self.assertEqual(snap.status("work.foreground", 2.0), "fresh")
        self.assertEqual(snap.status("work.foreground", 2.5), "stale")          # same TTL rule as before

    def test_the_line_names_the_reason(self):
        lines = []
        full(brain=brain_without_router(), lines=lines).snapshot()
        self.assertIn("conversation.epoch_state(no_router)", lines[0])
        self.assertIn("conversation.router_usable(no_router)", lines[0])
        self.assertIn("known=14 unknown=7 ", lines[0])
        self.assertIn("brain:5", lines[0])                                       # session id + the four read from it


class MissingMetadataTests(unittest.TestCase):
    def test_a_known_field_with_no_ttl_cannot_crash_a_snapshot(self):
        ttl = {k: v for k, v in S._TTL_S.items() if k != "audio.mic_mode"}
        lines = []
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            snap = full(lines=lines).snapshot()                                 # must not raise KeyError
        m = snap.meta_of("audio.mic_mode")
        self.assertEqual((m.state, m.reason, m.source, m.as_of_mono, m.as_of_wall, m.ttl_s),
                         ("unknown", "no_metadata", None, None, None, None))
        self.assertEqual(snap.status("audio.mic_mode", 0.0), "unknown")         # never fresh, never stale
        self.assertIsNone(snap.age_s("audio.mic_mode", 1e6))
        self.assertIn("audio.mic_mode", snap.unknown_paths())
        self.assertEqual(len(lines), 1)                                          # the observer still ran
        self.assertIn("audio.mic_mode(no_metadata)", lines[0])

    def test_no_ttl_is_ever_guessed_for_the_missing_entry(self):
        ttl = {k: v for k, v in S._TTL_S.items() if k != "work.foreground"}
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            snap = full().snapshot()
        m = snap.meta_of("work.foreground")
        self.assertIsNone(m.ttl_s)
        self.assertEqual(snap.status("work.foreground", 0.0), "unknown")
        self.assertEqual(snap.status("work.foreground", 1e9), "unknown")

    def test_a_known_field_with_no_provenance_entry_fails_closed_too(self):
        src = {k: v for k, v in S._SOURCE.items() if k != "audio.zipped"}
        with mock.patch.dict(S._SOURCE, src, clear=True):
            snap = full().snapshot()
        m = snap.meta_of("audio.zipped")
        self.assertEqual((m.state, m.reason, m.source, m.ttl_s), ("unknown", "no_metadata", None, None))

    def test_the_other_fields_and_the_view_values_are_untouched_by_a_missing_entry(self):
        ttl = {k: v for k, v in S._TTL_S.items() if k != "audio.mic_mode"}
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            snap = full(Clock(0.0)).snapshot()
        self.assertEqual(snap.audio.mic_mode, "open")                           # the value is not rewritten
        self.assertEqual(snap.meta_of("audio.ptt_held").ttl_s, 1.0)
        self.assertEqual(snap.status("audio.ptt_held", 1.0), "fresh")
        self.assertEqual(snap.status("audio.ptt_held", 1.5), "stale")

    def test_the_real_tables_cover_every_known_path_so_the_fail_closed_branch_is_not_taken(self):
        snap = full(desktop=lambda: "active/terminal").snapshot()
        self.assertEqual({m.reason for m in snap.meta if m.state == "unknown"}, {"no_source"})


class ObserverFailureSignalTests(unittest.TestCase):
    def boom(self, line):
        raise RuntimeError("secret /Users/yousif/x sess-secret-123")

    def test_a_failing_observer_never_costs_the_read_and_reports_a_fixed_line(self):
        out = []
        snap = full(observer=self.boom, on_failure=out.append).snapshot()
        self.assertEqual(snap.audio.mic_mode, "open")
        self.assertEqual(out, ["[sit] observer_failure"])
        for leak in ("secret", "/Users", "sess-", "RuntimeError"):
            self.assertNotIn(leak, out[0])

    def test_repeated_failures_are_rate_limited_and_counted_not_flooded(self):
        out, clk = [], Clock(100.0)
        rd = full(clk, observer=self.boom, on_failure=out.append, failure_interval_s=60.0)
        for _ in range(10):
            rd.snapshot()
        self.assertEqual(out, ["[sit] observer_failure"])                       # 1 line for 10 failures
        clk.t = 159.9
        rd.snapshot()
        self.assertEqual(len(out), 1)                                            # still inside the interval
        clk.t = 160.0                                                            # exactly the interval: allowed
        rd.snapshot()
        self.assertEqual(out[-1], "[sit] observer_failure suppressed=10")
        rd.snapshot()
        self.assertEqual(len(out), 2)                                            # a new quiet period began
        clk.t = 400.0
        rd.snapshot()
        self.assertEqual(out[-1], "[sit] observer_failure suppressed=1")

    def test_categories_are_limited_independently(self):
        out, clk = [], Clock(0.0)
        rd = full(clk, on_failure=out.append)
        rd.report_failure("observer_failure")
        rd.report_failure("snapshot_failure")
        rd.report_failure("observer_failure")
        self.assertEqual(out, ["[sit] observer_failure", "[sit] snapshot_failure"])

    def test_an_unknown_category_is_never_echoed(self):
        out = []
        rd = full(on_failure=out.append)
        rd.report_failure("/Users/yousif/secret sess-123")
        self.assertEqual(out, ["[sit] failure"])

    def test_a_clock_that_goes_backwards_cannot_wedge_the_limiter_shut(self):
        out, clk = [], Clock(500.0)
        rd = full(clk, on_failure=out.append)
        rd.report_failure("observer_failure")
        clk.t = 10.0
        rd.report_failure("observer_failure")
        self.assertEqual(len(out), 2)

    def test_no_failure_sink_is_silent_and_a_failing_sink_is_swallowed(self):
        rd = full(observer=self.boom)                                            # on_failure is None
        rd.snapshot()
        def bad_sink(line):
            raise RuntimeError("log write failed")
        rd = full(observer=self.boom, on_failure=bad_sink)
        self.assertEqual(rd.snapshot().audio.mic_mode, "open")
        rd.report_failure("snapshot_failure")                                    # never raises

    def test_a_healthy_observer_reports_no_failure(self):
        out, lines = [], []
        full(observer=lines.append, on_failure=out.append).snapshot()
        self.assertEqual((len(lines), out), (1, []))

    def test_the_signal_adds_no_thread_timer_or_logging_dependency(self):
        import ast
        tree = ast.parse(inspect.getsource(S))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported, {"time", "dataclasses"})


class UnknownPathsTests(unittest.TestCase):
    def test_unknown_paths_lists_exactly_the_unknown_fields_in_order(self):
        self.assertEqual(full().snapshot().unknown_paths(), NO_SOURCE)
        self.assertEqual(reader().snapshot().unknown_paths(), LEAVES)

    def test_unknown_paths_tracks_a_failed_group_a_router_less_brain_and_missing_metadata(self):
        def boom():
            raise RuntimeError("x")
        got = full(mic=boom).snapshot().unknown_paths()
        self.assertTrue({"audio.mic_mode", "audio.ptt_held", "audio.zipped", "audio.mouth_busy"} <= set(got))
        got = full(brain=brain_without_router()).snapshot().unknown_paths()
        self.assertIn("conversation.epoch_state", got)
        self.assertNotIn("conversation.epoch", got)
        self.assertNotIn("audio.mic_mode", full().snapshot().unknown_paths())

    def test_stale_fields_are_not_unknown_paths(self):
        snap = full(Clock(0.0)).snapshot()
        self.assertEqual(snap.unknown_paths(), NO_SOURCE)
        self.assertIn("work.foreground", snap.stale_paths(1e6))
        self.assertNotIn("work.foreground", snap.unknown_paths())


class UnchangedContractTests(unittest.TestCase):
    def test_the_21_path_order_is_the_same_in_every_state(self):
        def boom():
            raise RuntimeError("x")
        ttl = {k: v for k, v in S._TTL_S.items() if k != "audio.mic_mode"}
        states = [full().snapshot(), reader().snapshot(), full(mic=boom).snapshot(),
                  full(brain=brain_without_router()).snapshot()]
        with mock.patch.dict(S._TTL_S, ttl, clear=True):
            states.append(full().snapshot())
        for snap in states:
            self.assertEqual(tuple(m.path for m in snap.meta), LEAVES)
            self.assertEqual(len(snap.meta), 21)

    def test_ttl_values_and_status_rules_are_exactly_as_before(self):
        self.assertEqual(S._TTL_S, EXPECTED_TTL)
        snap = full(Clock(0.0)).snapshot()
        for path, ttl in EXPECTED_TTL.items():
            self.assertEqual(snap.status(path, ttl), "fresh", path)
            self.assertEqual(snap.status(path, ttl + 1e-6), "stale", path)
        self.assertEqual(snap.status("audio.mic_mode", -1.0), "stale")
        self.assertEqual(snap.status("device", 1e9), "unknown")

    def test_rebuild_pending_is_still_explicitly_unknown_with_no_source(self):
        for snap in (full().snapshot(), full(brain=brain_without_router()).snapshot()):
            m = snap.meta_of("conversation.rebuild_pending")
            self.assertEqual((m.state, m.reason), ("unknown", "no_source"))


class MainFailureWiringTests(unittest.TestCase):
    def setUp(self):
        from backtalk import main
        self.main = main

    def test_the_failure_sink_is_the_log_and_the_helper_reports_a_snapshot_failure(self):
        src = inspect.getsource(self.main)
        self.assertIn("observer=log, on_failure=log", src)
        saved, got = self.main._SITUATION, []
        try:
            def bad():
                raise RuntimeError("secret detail")
            self.main._SITUATION = SimpleNamespace(snapshot=bad, report_failure=got.append)
            self.main._observe_situation()
            self.assertEqual(got, ["snapshot_failure"])                          # a category, never the exception
            def worse(_c):
                raise RuntimeError("sink failed too")
            self.main._SITUATION = SimpleNamespace(snapshot=bad, report_failure=worse)
            self.main._observe_situation()                                       # still swallowed
        finally:
            self.main._SITUATION = saved


if __name__ == "__main__":
    unittest.main()


class DesktopSlotTests(unittest.TestCase):
    """Phase G: the `desktop` leaf is one enum string from the presence monitor, or UNKNOWN. Observability only."""

    def test_unwired_or_no_fresh_reading_is_unknown_never_a_guess(self):
        for src in (None, lambda: None):
            snap = full(desktop=src).snapshot()
            self.assertIs(snap.desktop, UNKNOWN)
            self.assertEqual(snap.meta_of("desktop").state, "unknown")
        self.assertEqual(full(desktop=None).snapshot().meta_of("desktop").reason, "unwired")

    def test_a_reading_is_known_fresh_then_stale_after_its_ttl(self):
        clk = Clock(0.0)
        snap = full(clk, desktop=lambda: "idle/browser/call").snapshot()
        self.assertEqual(snap.desktop, "idle/browser/call")
        m = snap.meta_of("desktop")
        self.assertEqual((m.state, m.source, m.ttl_s), ("known", "presence_monitor", 15.0))
        self.assertEqual(snap.status("desktop", 15.0), "fresh")
        self.assertEqual(snap.status("desktop", 15.001), "stale")

    def test_anything_that_is_not_a_plain_identifier_is_unknown(self):
        for bad in ("Zoom Meeting - secret.docx", "a\nb", "x" * 60, "", 5, ["active"], "active/Chrome", "a b"):
            self.assertIs(full(desktop=lambda b=bad: b).snapshot().desktop, UNKNOWN, repr(bad))

    def test_a_raising_source_poisons_only_the_desktop_leaf(self):
        def boom():
            raise RuntimeError("secret /Users/x")
        snap = full(desktop=boom).snapshot()
        self.assertIs(snap.desktop, UNKNOWN)
        self.assertEqual(snap.meta_of("desktop").reason, "unavailable")
        self.assertEqual(snap.audio.mic_mode, "open")
        self.assertIn("desktop", snap.unavailable)

    def test_the_sit_line_carries_the_name_only_never_the_value(self):
        lines = []
        full(desktop=lambda: "active/terminal/call", observer=lines.append).snapshot()
        self.assertNotIn("terminal", lines[0])
        self.assertNotIn("call", lines[0].replace("stale_fields", ""))
        self.assertIn("presence_monitor:1", lines[0])
