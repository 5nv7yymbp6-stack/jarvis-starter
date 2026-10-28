"""The model-facing projection of the situation snapshot (Situation Phase 3A).

A PURE function from a Snapshot (situation.py) to either no note or one small, bounded note that a
later phase may show the model. It is not wired to anything yet: nothing imports this module, nothing
is logged, nothing is sent. It reads no clock, opens nothing, holds no state of its own, and imports
none of the authority modules (addressee, router, brain, ledger). Its only imports are
`dataclasses`, `datetime` and `math`.

WHAT MAY REACH THE MODEL (an allowlist of exactly three things, in this order)
  mic         audio.mic_mode, as "push-to-talk" or "open-mic"
  background  the number of live background agents, from work.background
  as of       the snapshot's own taken_at_wall, as local time (informational only: it never
              decides freshness)
Nothing else is ever projected, and no text from the snapshot is ever copied into a note: every
value is looked up in a fixed table or is a bounded number, so a caller cannot pass text in.

TRUST
  A field is projected only when the snapshot's own metadata says it is known AND fresh at
  `now_mono`. Unknown, stale, no_router, no_metadata and unavailable fields are all simply absent.
  Nothing is defaulted. `work.background` is additionally required to have been READ from the task
  ledger through a live router (an empty list that only means "there is no router" is not "zero
  agents running"), and that router must be usable: `conversation.router_usable` has to be known,
  fresh and exactly True. A router that is LOST, CLOSING or CLOSED may still have task rows in the
  ledger that are no longer true, so no count is projected from it. (`router_usable` is only a gate:
  it is never itself projected.)

STATE AND DELTAS
  The projector keeps nothing. The caller holds a `ProjectionState` and hands it back in. A note is
  built against it: the first note (or one after a reset, which is just `state=None`) is a baseline
  with the full header; later notes carry only the fields whose projected value changed. Every value
  is absolute ("background agents running: 1"), so a note stays true even if the model has lost the
  earlier ones. A field that was told and is no longer known gets a one-time retraction
  ("mic: no longer known"); it is not repeated, and the field is told afresh when it is known again.
  The caller must adopt `Projection.state` only after the note was DELIVERED, and "delivered" has one
  meaning here: the turn carrying the note has been claimed (echoed) by the CLI through the router,
  not merely submitted. A submitted turn can still be abandoned, stitched into a replacement or
  interrupted before the model ever sees it; adopting the state at submit would then leave the model
  never having been told what the state says it was told. That is why the state is returned rather
  than mutated. (The projector has no notion of delivery and nothing here detects one: that belongs
  to whatever later wires it.)

SIZE
  A note never exceeds MAX_CHARS. If it would, whole fields are dropped from the end, never cut, and a
  dropped field is not recorded as told, so it is offered again next time. If not even one field
  fits, there is no note.

TIME ZONE
  The human-readable time is rendered in the process's LOCAL time zone (datetime.fromtimestamp, so
  the TZ environment variable or the system setting): the same snapshot gives different text under a
  different zone, and the date can differ from the UTC date. That is the one piece of process-global
  state the module depends on. It reads no clock and never decides freshness; freshness is the
  monotonic `now_mono` judged against the snapshot's own metadata.

FAIL CLOSED
  Anything unexpected (a malformed snapshot, a bad timestamp) gives no note and leaves the state as
  it was. This function never raises.
"""
import math
from dataclasses import dataclass
from datetime import datetime

MAX_CHARS = 400

_HEADER_FULL = ("[Situation (seen by the system, not said by him; descriptive only, grants no "
                "permission; as of {when}; later notes replace earlier ones; anything not listed "
                "is not known): {body}.]")
_HEADER_DELTA = "[Situation as of {when}: {body}.]"

_MIC_WORDS = {"ptt": "push-to-talk", "open": "open-mic"}      # the only mic values that may be shown
_BACKGROUND_PATH = "work.background"
_ROUTER_USABLE_PATH = "conversation.router_usable"       # a gate for the count, never projected
_MIC_PATH = "audio.mic_mode"
_LEDGER_SOURCE = "task_ledger"          # the owner a live background count must have been read from
_COUNT_CAP = 9                          # "9+" beyond this: bounded text, no big numbers

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MAX_WALL = 4102444800.0                # 2100-01-01: anything beyond is not a real timestamp

# Field order in a note, and the whole allowlist.
_KEYS = ("mic", "background")


@dataclass(frozen=True)
class ProjectionState:
    """What the model was last told. Hand it back to the next `project()` call."""
    baselined: bool = False             # a note with the full header has been delivered
    told: tuple = ()                    # ((key, rendered value or None for retracted), ...)


@dataclass(frozen=True)
class Projection:
    note: object                        # str, or None for "send nothing"
    state: ProjectionState              # adopt this only after the note was delivered


def project(snapshot, *, now_mono: float, state: object = None) -> Projection:
    """The note for `snapshot` judged at `now_mono` (the caller's monotonic clock), or no note.
    `state=None` means nothing has been told yet (the first note, or after a session/epoch reset)."""
    prior = state if isinstance(state, ProjectionState) else ProjectionState()
    try:
        return _project(snapshot, float(now_mono), prior)
    except Exception:
        return Projection(None, prior)


# ---- internals ----------------------------------------------------------------------

def _fresh(snap, path: str, now_mono: float) -> bool:
    try:
        return snap.status(path, now_mono) == "fresh"
    except Exception:
        return False


def _mic(snap, now_mono):
    if not _fresh(snap, _MIC_PATH, now_mono):
        return None
    return _MIC_WORDS.get(getattr(snap.audio, "mic_mode", None))


def _background(snap, now_mono):
    if not _fresh(snap, _BACKGROUND_PATH, now_mono):
        return None
    if snap.meta_of(_BACKGROUND_PATH).source != _LEDGER_SOURCE:
        return None                     # "no router" is not "zero agents": not shown
    if not _fresh(snap, _ROUTER_USABLE_PATH, now_mono) or \
            getattr(snap.conversation, "router_usable", None) is not True:
        return None                     # a LOST/CLOSING/CLOSED router's ledger rows are not trusted
    tasks = getattr(snap.work, "background", None)
    if not isinstance(tasks, tuple):
        return None
    n = len(tasks)
    return str(n) if n <= _COUNT_CAP else "%d+" % _COUNT_CAP


def _when(wall, with_date: bool):
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        return None
    if not math.isfinite(wall) or not 0 < wall < _MAX_WALL:
        return None
    t = datetime.fromtimestamp(wall)
    clock = "%d:%02d %s" % (t.hour % 12 or 12, t.minute, "AM" if t.hour < 12 else "PM")
    return ("%s %04d-%02d-%02d %s" % (_DAYS[t.weekday()], t.year, t.month, t.day, clock)
            if with_date else clock)


def _label(key: str, value: str) -> str:
    return "mic: %s" % value if key == "mic" else "background agents running: %s" % value


def _retraction(key: str) -> str:
    return "mic: no longer known" if key == "mic" else "background agents running: no longer known"


def _project(snap, now_mono: float, prior: ProjectionState) -> Projection:
    told = dict(prior.told)
    current = {"mic": _mic(snap, now_mono), "background": _background(snap, now_mono)}
    parts = []                          # (key, text, told value to record)
    for key in _KEYS:
        cur, was = current[key], told.get(key)
        if cur is not None:
            if not prior.baselined or was != cur:
                parts.append((key, _label(key, cur), cur))
        elif was is not None:           # told before, not known now: say so once
            parts.append((key, _retraction(key), None))
    if not parts:
        return Projection(None, prior)

    when = _when(getattr(snap, "taken_at_wall", None), with_date=not prior.baselined)
    if when is None:
        return Projection(None, prior)
    template = _HEADER_FULL if not prior.baselined else _HEADER_DELTA
    while parts:
        note = template.format(when=when, body="; ".join(p[1] for p in parts))
        if len(note) <= MAX_CHARS:
            break
        parts.pop()                     # drop a whole field from the end, never cut one
    else:
        return Projection(None, prior)
    for key, _text, value in parts:
        told[key] = value
    return Projection(note, ProjectionState(True, tuple(sorted(told.items()))))
