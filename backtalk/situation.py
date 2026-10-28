"""A read-only snapshot of what the voice line currently knows about its situation.

PHASE 1 of the Situation / Context Engine: one place that can answer "what is the state right
now?" by READING the objects that already own that state. Nothing here owns, caches, polls,
persists or decides anything; no behavior in the process depends on it, and nothing is shown to
the model.

RULES
  * Read only. Every owner is read through a public attribute, property or method, and only
    plain values are copied out. No owner is written to, no lock is taken, nothing is logged by
    the owners as a side effect of being read (AddresseeGate.state is a pure clock comparison;
    _sync(), which logs, is never called here).
  * The snapshot is frozen: dataclasses are frozen, collections are tuples, and nothing in it
    refers back to a live owner object.
  * Unknown stays unknown. A field with no source, a source that was not wired, and a source that
    raised are all UNKNOWN: never False, never an empty tuple, never a default. UNKNOWN refuses to
    be used as a boolean, so a caller cannot quietly treat "don't know" as "no".
  * No guessing. Fields that no owner exposes publicly yet (voice state, the router's pending
    rebuild, device, desktop, location, the addressee's deadlines) are UNKNOWN in this phase.
  * Free text is never copied: task descriptions and utterances stay with
    their owners. Only identifiers, enum-like words, numbers and booleans.

PHASE 2 adds time and provenance, still without owning anything and still without telling the
model. Every field has a FieldMeta in `Snapshot.meta`: which owner it was read from (an explicit
table, never derived from the field's name), when (monotonic and wall clock), and how long a
held observation may be trusted (ttl_s).
  * UNKNOWN and stale are different. UNKNOWN: no trustworthy source exists (not wired, no public
    accessor, or the read failed); there is no value, no age and no TTL. Stale: a source exists and
    the value was read, but the observation is older than its TTL at the moment it is judged.
    Neither is ever turned into a guessed value.
  * Freshness uses the MONOTONIC clock only. The wall clock is recorded for the log and never
    decides anything. A snapshot is a moment: it is judged against a caller-supplied `now_mono`,
    which is what makes a held snapshot able to go stale. At the instant of the read every known
    field has age 0.
  * Staleness boundary: stale means age > ttl_s (exactly at the TTL is still fresh). A negative age
    (the caller passed a clock that is behind the observation) is treated as stale, never fresh.
  * Observability goes through an injected `observer(line)`, so this module still imports no
    logging, I/O or owner code. The line holds names, counts and statuses only; never a value.
"""
import dataclasses
import time
from dataclasses import dataclass, field


class _Unknown:
    """The value of a field nobody can vouch for."""
    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNKNOWN"

    def __bool__(self):
        raise TypeError("UNKNOWN is neither true nor false; test `is UNKNOWN` first")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_Unknown, ())


UNKNOWN = _Unknown()


# ---- the views (all frozen, all plain data) ------------------------------------

@dataclass(frozen=True)
class AudioView:
    mic_mode: object = UNKNOWN          # "ptt" | "open"
    ptt_held: object = UNKNOWN          # bool: the talk key is down right now
    zipped: object = UNKNOWN            # bool: replies withheld until "unzip"
    mouth_busy: object = UNKNOWN        # bool: the voice is speaking or queued
    voice_state: object = UNKNOWN       # no public owner accessor yet (a signals file only)
    speaker: object = UNKNOWN           # voice-lock is shadow-only; nothing is established


@dataclass(frozen=True)
class ConversationView:
    addressee_mode: object = UNKNOWN    # "wake" | "always"
    addressee_state: object = UNKNOWN   # "IDLE" | "ACTIVE"
    addressee_window_s: object = UNKNOWN
    addressee_max_active_s: object = UNKNOWN
    session_id: object = UNKNOWN        # str | None (None: no conversation id yet, which is known)
    epoch: object = UNKNOWN             # str | None (None: no router open)
    epoch_state: object = UNKNOWN       # "OPEN" | "CLOSING" | "CLOSED" | "LOST"
    router_usable: object = UNKNOWN     # bool
    rebuild_pending: object = UNKNOWN   # no public accessor on the brain yet: always UNKNOWN


@dataclass(frozen=True)
class TurnView:
    turn_id: str
    kind: str                           # foreground | command
    state: str                          # ledger turn state
    age_s: float                        # since it was handed to the SDK
    silent: bool


@dataclass(frozen=True)
class TaskView:
    task_id: str
    kind: str                           # BG_AGENT ...
    state: str                          # LAUNCHED | RUNNING
    protected: bool
    epoch: object                       # str | None


@dataclass(frozen=True)
class WorkView:
    foreground: object = UNKNOWN        # tuple[TurnView]; () means known-none
    active_turn_id: object = UNKNOWN    # str | None: the run that is claimed and working
    background: object = UNKNOWN        # tuple[TaskView]; () means known-none


@dataclass(frozen=True)
class FieldMeta:
    path: str                           # "audio.mic_mode", "device", ...
    state: str                          # "known" | "unknown"
    source: object = None               # the owner it was read from (known only)
    reason: object = None               # why unknown: "unwired" (the owner was not passed in) |
                                        # "unavailable" (the read raised) | "no_source" (no owner
                                        # exists yet) | "no_router" (the owner IS wired, but the
                                        # router it would read is absent right now) |
                                        # "no_metadata" (a value was read but its provenance or TTL
                                        # is missing, so it is not trusted)
    as_of_mono: object = None           # monotonic time of the read (known only)
    as_of_wall: object = None           # wall-clock time of the read, for the log only
    ttl_s: object = None                # how long a held observation may be trusted (known only)


# Every field's owner, written out. Never derived from the field's name. A field absent from this
# table has no source that exists yet and is always UNKNOWN (reason "no_source").
_SOURCE = {
    "audio.mic_mode": "mic_owner", "audio.ptt_held": "mic_owner",
    "audio.zipped": "zip_owner", "audio.mouth_busy": "mouth",
    "conversation.addressee_mode": "addressee_gate", "conversation.addressee_state": "addressee_gate",
    "conversation.addressee_window_s": "addressee_gate",
    "conversation.addressee_max_active_s": "addressee_gate",
    "conversation.session_id": "brain",
    "conversation.epoch": "router", "conversation.epoch_state": "router",
    "conversation.router_usable": "router",
    "work.foreground": "router", "work.active_turn_id": "router", "work.background": "task_ledger",
    "desktop": "presence_monitor",
}

# Conservative, short, explicit (seconds). The rule: the faster the owner can change it, the
# shorter the TTL, and nothing is trusted for longer than half a minute.
#   1 s   things a person changes by moving: the talk key, the voice speaking or going quiet
#   2 s   a turn appearing or resolving
#   5 s   things changed by a spoken or typed command or by a timer: mic mode, zip, addressee
#         state (its window is 180 s, but a lapse should not go unnoticed for long), epoch state,
#         router usability, background tasks
#   10 s  the epoch id (changes only on a rebuild)
#   30 s  values fixed for the life of the process or conversation: addressee mode/window/cap,
#         session id
_TTL_S = {
    "audio.ptt_held": 1.0, "audio.mouth_busy": 1.0,
    "work.foreground": 2.0, "work.active_turn_id": 2.0,
    "audio.mic_mode": 5.0, "audio.zipped": 5.0, "conversation.addressee_state": 5.0,
    "conversation.epoch_state": 5.0, "conversation.router_usable": 5.0, "work.background": 5.0,
    "conversation.epoch": 10.0,
    "conversation.addressee_mode": 30.0, "conversation.addressee_window_s": 30.0,
    "conversation.addressee_max_active_s": 30.0, "conversation.session_id": 30.0,
    "desktop": 15.0,                  # the presence monitor refreshes every few seconds; an older reading is not trusted
}

# Fields that can only be read from a live router. When the brain is wired but has no router
# open, these are UNKNOWN for that reason ("no_router"), not because anything is unwired.
_NEEDS_ROUTER = frozenset({"conversation.epoch_state", "conversation.router_usable"})
# Fields that are KNOWN-empty when the brain has no router (no epoch, nothing running). What was
# read is the brain's own `router` attribute, so that, not the router, is the provenance then.
_ROUTER_OR_BRAIN = frozenset({"conversation.epoch", "work.foreground", "work.active_turn_id",
                              "work.background"})

FRESH, STALE, UNKNOWN_STATUS = "fresh", "stale", "unknown"

# The only things the failure signal may ever say. Anything else is shown as plain "failure".
_FAILURE_CATEGORIES = frozenset({"observer_failure", "snapshot_failure"})


@dataclass(frozen=True)
class Snapshot:
    taken_at_wall: float
    taken_at_mono: float
    audio: object = UNKNOWN
    conversation: object = UNKNOWN
    work: object = UNKNOWN
    # No owner exists for these yet. They are listed so their absence is explicit, not so a
    # reader can wonder whether they were forgotten.
    device: object = UNKNOWN
    desktop: object = UNKNOWN
    location: object = UNKNOWN
    unavailable: tuple = field(default=())   # groups whose source was wired but failed to read
    meta: tuple = field(default=())          # one FieldMeta per field, in a fixed order

    def meta_of(self, path: str) -> FieldMeta:
        for m in self.meta:
            if m.path == path:
                return m
        raise KeyError(path)

    def status(self, path: str, now_mono: float) -> str:
        """"fresh" | "stale" | "unknown" for one field, judged at `now_mono` (monotonic)."""
        m = self.meta_of(path)
        if m.state != "known":
            return UNKNOWN_STATUS
        age = now_mono - m.as_of_mono
        return FRESH if 0 <= age <= m.ttl_s else STALE

    def age_s(self, path: str, now_mono: float):
        """Seconds since the field was read, or None when it is UNKNOWN (there is nothing to age)."""
        m = self.meta_of(path)
        return None if m.state != "known" else now_mono - m.as_of_mono

    def stale_paths(self, now_mono: float) -> tuple:
        return tuple(m.path for m in self.meta if self.status(m.path, now_mono) == STALE)

    def unknown_paths(self) -> tuple:
        return tuple(m.path for m in self.meta if m.state != "known")


# ---- the sources ----------------------------------------------------------------

@dataclass
class SituationSources:
    """The owners to read. Anything left None is simply not wired, and its fields are UNKNOWN."""
    mic: object = None            # callable -> mapping with "mode" and "btn" (main._MIC)
    zipped: object = None         # callable -> bool
    mouth: object = None          # object with a `busy` attribute
    addressee: object = None      # AddresseeGate
    brain: object = None          # WarmBrain: .router, .session_id, .live_background()
    desktop: object = None        # callable -> "state/app[/call]" enum string, or None when there is no fresh reading
    clock: object = time.monotonic            # MONOTONIC: decides freshness
    wall_clock: object = time.time            # recorded for the log only; decides nothing
    observer: object = None                   # callable(line: str): the passive [sit] line
    on_failure: object = None                 # callable(line: str): a fixed, safe failure line
    failure_interval_s: float = 60.0          # at most one failure line per category per interval


class SituationReader:
    """Builds a Snapshot on demand. Holds only references to the owners."""

    def __init__(self, sources: SituationSources):
        self._s = sources
        self._failures = {}        # category -> (monotonic time of the last line, suppressed since)

    def snapshot(self) -> Snapshot:
        mono, wall = self._s.clock(), self._s.wall_clock()
        failed = []
        groups = {}
        router_absent = {}         # group -> True when the brain is wired but has no router open
        for name, build in (("audio", self._audio),
                            ("conversation", lambda: self._conversation(router_absent)),
                            ("work", lambda: self._work(router_absent))):
            try:
                groups[name] = build()
            except Exception:
                groups[name] = UNKNOWN
                failed.append(name)
        try:
            groups["desktop"] = self._desktop()
        except Exception:
            groups["desktop"] = UNKNOWN
            failed.append("desktop")
        snap = Snapshot(taken_at_wall=wall, taken_at_mono=mono, unavailable=tuple(failed),
                        meta=self._meta(groups, failed, mono, wall, router_absent), **groups)
        if self._s.observer is not None:
            try:
                self._s.observer(sit_line(snap, mono))
            except Exception:
                self.report_failure("observer_failure")    # never costs a read, never says why
        return snap

    def report_failure(self, category: str) -> None:
        """A fixed, rate-limited "[sit] <category>" line through `on_failure`. It carries no
        value and no exception text (only a fixed category and, after a quiet period, a count of
        how many were suppressed), starts no thread or timer, and can never raise."""
        try:
            if self._s.on_failure is None:
                return
            cat = category if category in _FAILURE_CATEGORIES else "failure"
            now = self._s.clock()
            last, held = self._failures.get(cat, (None, 0))
            if last is not None and 0 <= now - last < self._s.failure_interval_s:
                self._failures[cat] = (last, held + 1)
                return
            self._failures[cat] = (now, 0)
            self._s.on_failure("[sit] %s%s" % (cat, " suppressed=%d" % held if held else ""))
        except Exception:
            pass

    def _meta(self, groups, failed, mono, wall, router_absent) -> tuple:
        out = []
        for path, value, group in _leaves(groups):
            if value is UNKNOWN:
                if group in failed:
                    reason = "unavailable"
                elif path in _NEEDS_ROUTER and router_absent.get(group):
                    reason = "no_router"
                elif path not in _SOURCE:
                    reason = "no_source"
                else:
                    reason = "unwired"
                out.append(FieldMeta(path, "unknown", reason=reason))
                continue
            src = _SOURCE.get(path)
            if src and path in _ROUTER_OR_BRAIN and router_absent.get(group):
                src = "brain"
            ttl = _TTL_S.get(path)
            if not src or ttl is None:
                # A value was read but its provenance or TTL is missing. Fail closed: no guessed
                # TTL, no invented source, and nothing that could ever count as fresh.
                out.append(FieldMeta(path, "unknown", reason="no_metadata"))
                continue
            out.append(FieldMeta(path, "known", source=src, as_of_mono=mono, as_of_wall=wall,
                                 ttl_s=ttl))
        return tuple(out)

    # Each builder returns UNKNOWN for a group whose every source is unwired, and otherwise a
    # view in which the unwired parts are individually UNKNOWN. A source that raises poisons the
    # whole group (the caller records it in `unavailable`) rather than half-reporting it.

    def _desktop(self):
        """One enum string from the presence owner, or UNKNOWN: unwired, no fresh reading, or anything that is not a plain
        identifier (so no free text can ever ride in this field)."""
        if self._s.desktop is None:
            return UNKNOWN
        v = self._s.desktop()
        if not isinstance(v, str) or not v or len(v) > 48 or not set(v) <= _SAFE_DESKTOP:
            return UNKNOWN
        return v

    def _audio(self):
        s = self._s
        if s.mic is None and s.zipped is None and s.mouth is None:
            return UNKNOWN
        kw = {}
        if s.mic is not None:
            mic = s.mic()
            kw["mic_mode"] = str(mic["mode"])
            kw["ptt_held"] = bool(mic["btn"])
        if s.zipped is not None:
            kw["zipped"] = bool(s.zipped())
        if s.mouth is not None:
            kw["mouth_busy"] = bool(s.mouth.busy)
        return AudioView(**kw)

    def _conversation(self, router_absent):
        s = self._s
        a, b = s.addressee, s.brain
        if a is None and b is None:
            return UNKNOWN
        kw = {}
        if a is not None:
            kw["addressee_mode"] = str(a.mode)
            kw["addressee_state"] = str(a.state)
            kw["addressee_window_s"] = float(a.window)
            kw["addressee_max_active_s"] = float(a.max_active)
        if b is not None:
            sid = b.session_id
            kw["session_id"] = None if sid is None else str(sid)
            r = b.router
            if r is None:
                kw["epoch"] = None
                router_absent["conversation"] = True
            else:
                kw["epoch"] = str(r.epoch)
                kw["epoch_state"] = str(r.state)
                kw["router_usable"] = bool(r.usable)
        return ConversationView(**kw)

    def _work(self, router_absent):
        b = self._s.brain
        if b is None:
            return UNKNOWN
        r = b.router
        if r is None:
            router_absent["work"] = True
            return WorkView(foreground=(), active_turn_id=None, background=())
        now = self._s.clock()
        fg = tuple(TurnView(str(h.turn_id), str(h.kind), str(h.state),
                            max(0.0, now - float(h.sent_at)), bool(h.silent))
                   for h in list(r.handles.values()) if not h.terminal)
        active = r.active_handle()
        bg = tuple(TaskView(str(t["id"]), str(t.get("task_kind")), str(t.get("state")),
                            bool(t.get("protected")), t.get("epoch"))
                   for t in b.live_background())
        return WorkView(foreground=fg,
                        active_turn_id=None if active is None else str(active.turn_id),
                        background=bg)


# ---- leaves and the passive [sit] line ------------------------------------------

_GROUPS = (("audio", AudioView), ("conversation", ConversationView), ("work", WorkView))


def _leaves(groups):
    """(path, value, group) for every field, in a fixed order; UNKNOWN groups yield UNKNOWN fields."""
    for gname, cls in _GROUPS:
        g = groups[gname]
        for f in dataclasses.fields(cls):
            yield f"{gname}.{f.name}", (UNKNOWN if g is UNKNOWN else getattr(g, f.name)), gname
    for name in ("device", "desktop", "location"):
        yield name, groups.get(name, UNKNOWN), name       # only `desktop` has an owner (presence); the rest are always UNKNOWN


_SAFE_NAME = frozenset("abcdefghijklmnopqrstuvwxyz_.+")
_SAFE_DESKTOP = frozenset("abcdefghijklmnopqrstuvwxyz_/")


def _name(x) -> str:
    """A static identifier only; anything else is shown as '?' so no value can ride in a name."""
    x = str(x)
    return x if x and set(x) <= _SAFE_NAME else "?"


def sit_line(snap: Snapshot, now_mono: float) -> str:
    """The passive observability line for one snapshot. Field, group and source NAMES, counts and
    statuses only: no field value, no id, no text, no path. Everything is static or counted."""
    known = [m for m in snap.meta if m.state == "known"]
    unknown = [m for m in snap.meta if m.state != "known"]
    sources = {}
    for m in known:
        for name in str(m.source).split("+"):
            sources[name] = sources.get(name, 0) + 1
    ttls = [m.ttl_s for m in known]
    stale = snap.stale_paths(now_mono)
    return ("[sit] known=%d unknown=%d stale=%d unavailable=%s sources=%s unknown_fields=%s stale_fields=%s ttl_s=%s"
            % (len(known), len(unknown), len(stale),
               ",".join(_name(g) for g in snap.unavailable) or "-",
               ",".join("%s:%d" % (_name(k), sources[k]) for k in sorted(sources)) or "-",
               ",".join("%s(%s)" % (_name(m.path), _name(m.reason)) for m in unknown) or "-",
               ",".join(_name(p) for p in stale) or "-",
               ("%g-%g" % (min(ttls), max(ttls))) if ttls else "-"))
