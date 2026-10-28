# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""backtalk — talk to your Claude Code agent out loud.

Flow: hold the key and speak -> local transcription -> your agent's warm
Claude session streams the reply -> sentences go to the mouth the moment
they complete (~1-2s to first audio on warm turns). The greeting plays
over a hidden warmup query so the first real turn is already hot.

Typing in this terminal is a first-class turn too: same conversation,
spoken reply, and typing while it talks interrupts it.

THE VOICE CONSOLE: exact phrases, spoken (or typed) alone, control the
session itself so you never go back to the keyboard: "clear the
session" / "compact the session" / "switch to the deep model" / "back
to the fast model" / "set effort to low" (or medium, high, max) /
"usage report" / "go hands free" and "push to talk mode" (the MIC).
Tool calls run directly; there is no approval step.

Flags:
  --open-mic   start in hands-free listening for this session (the
               config key mic_mode makes it the standing default, and
               the voice can switch live either way: "go hands free" /
               "push to talk mode"). Know the tradeoff: room audio (a
               video, music, another voice assistant) can trigger
               replies to speech never meant for the agent. The talk
               key keeps working: it interrupts, and holding it always
               gets you heard.
  --barge-in   with --open-mic: keep listening WHILE speaking.
               HEADPHONES REQUIRED — with open speakers the mic hears
               the reply and the agent interrupts itself.
  --model X    override the model for this session (full id).

Say "goodbye <name>" / "end voice mode" to hang up. Ctrl-C works.
"""
import asyncio
import dataclasses
import difflib
import json
import os
import queue
import re
import socket
import sys
import threading
import time

from backtalk import (aec, audiowatch, authguard, board, sessions, signals, systap,
                      vaultrecall)
from backtalk import memory_shadow
from backtalk import situation as situation_mod
from backtalk import situation_context as situation_context_mod
from backtalk import ledger as ledger_mod
from backtalk import presence as presence_mod
from backtalk import gestures as gestures_mod
from backtalk import proactive as proactive_mod
from backtalk import proactive_loop
from backtalk import scheduler as scheduler_mod
from backtalk import taskstate
from backtalk import turnclass
from backtalk import turnend
from backtalk import speech_policy as SP
from backtalk.addressee import AddresseeGate
from backtalk.brain import WarmBrain
from backtalk.router import TurnFailed
from backtalk.config import CFG
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           warm as warm_ears)
from backtalk.mouth import Mouth
from backtalk.mouth import warm as warm_mouth
from backtalk.ptt import PTTListener
from backtalk.vlog import log

NAME = CFG["name"]
QUIT_PHRASES = CFG["quit_phrases"]

_ADDRESSEE = None    # AddresseeGate, set in amain()
_SITUATION = None    # SituationReader (situation.py), set in amain(); read-only, logs a passive [sit] line, never shown to the model


def _observe_situation():
    """One passive [sit] log line per turn (the snapshot emits it). It reads owners and writes
    nothing back; it can never raise into, delay or alter a turn."""
    try:
        if _SITUATION is not None:
            _SITUATION.snapshot()
    except Exception:
        try:
            _SITUATION.report_failure("snapshot_failure")    # fixed, rate-limited, no detail
        except Exception:
            pass


# ---- Situation Phase 3B/3C: shadow projection, and (default OFF) model-facing injection. ----------
# At the one real send point (speak_reply, just before brain.ask_stream) a FRESH snapshot is projected
# by situation_context.project() and one metadata-only "[sitctx]" line is logged. By default (Phase 3B)
# the note is built and dropped: it never reaches `sent`, a tool input or any model input.
# PHASE 3C: only when the config key `situation_context_inject` is exactly true (absent means OFF) the
# note is added to the model input, in speak_reply and nowhere else: vault + "\n\n" + note + "\n\n" + text.
# The note is descriptive context. It is never consulted by any gate and grants nothing. The shadow
# state moves only when the router had CLAIMED the turn that was sent (see _sitctx_end); submitting,
# a stream starting or a first token is not delivery.
_SITCTX_READER = None                      # a SituationReader with NO observer: the [sit] line is untouched
# The shadow ProjectionState and the Claude context it was told into: (epoch, session id, generation).
_SITCTX = {"state": None, "epoch": None, "session": None, "generation": None}
# main.py's own context-generation counter. /clear and /compact change what the model can still see
# WITHOUT changing the router epoch (a /clear starts a new session id inside the same epoch; a /compact
# may keep both), so each of them bumps it, and a state told under an older value is never reused.
_SITCTX_GEN = {"n": 0}
_SITCTX_FAIL = {}                          # fixed category -> monotonic time of its last failure line
_SITCTX_FAIL_EVERY_S = 60.0
_SITCTX_FIELDS = ("mic", "background")     # the fixed allowlist names: the only field names ever logged


def _sitctx_fail(category: str) -> None:
    """One fixed, rate-limited failure line. The category is always a literal chosen by the caller:
    no exception text, no value, ever. Never raises."""
    try:
        now = time.monotonic()
        last = _SITCTX_FAIL.get(category)
        if last is not None and 0 <= now - last < _SITCTX_FAIL_EVERY_S:
            return
        _SITCTX_FAIL[category] = now
        log(f"[sitctx] projection_failure category={category}")
    except Exception:
        pass


def _sitctx_identity(brain):
    """(epoch, session id, generation) of the Claude context a send goes into. Reads only; never raises."""
    try:
        router = getattr(brain, "router", None)
        return (getattr(router, "epoch", None), getattr(brain, "session_id", None), _SITCTX_GEN["n"])
    except Exception:
        return (None, None, None)


def _sitctx_invalidate() -> None:
    """A /clear or /compact is being run: whatever the model was told may no longer be in its context.
    Called BEFORE the command goes out, so a command that fails or times out still counts. The next real
    send then compares unequal to the adopted identity and gets a fresh baseline. Never raises."""
    try:
        _SITCTX_GEN["n"] += 1
    except Exception:
        pass


def _sitctx_same_context(brain, rec, handle) -> bool:
    """Is the context the handle was sent into the SAME one the projection was made for? Router instance,
    epoch, session id and generation must all be unchanged; any difference means "not the same context"."""
    epoch, session, generation = rec["identity"]
    return (getattr(brain, "router", None) is rec["router"]
            and getattr(handle, "epoch", None) == epoch
            and getattr(brain, "session_id", None) == session
            and _SITCTX_GEN["n"] == generation)


def _mem2_shadow(text, context) -> None:
    """Memory 2.0 shadow retrieval. Only when the config key `memory2_shadow` is exactly true (default OFF) one
    metadata-only "[mem2]" line is logged, from a read-only snapshot, in a worker thread that is never awaited:
    it cannot delay or alter the turn and its result goes nowhere but the log. Never raises."""
    try:
        if not memory_shadow.enabled(CFG):
            return

        def work():
            line = memory_shadow.observe(text, recall_found=bool(context))
            if line:
                log(line)

        asyncio.get_running_loop().run_in_executor(None, work)
    except Exception:
        pass


_SITCTX_FLAG = "situation_context_inject"      # the config key; absent or anything but literal true means OFF
_SITCTX_NOTE_MAX = 400                         # the projector's own cap (a test pins equality); defence in depth


def _sitctx_inject_enabled() -> bool:
    """Phase 3C switch. DEFAULT OFF: only the JSON value `true` turns it on (a string, a number or a missing
    key do not). Independent of every addressee and ledger setting. Never raises."""
    try:
        return CFG.get(_SITCTX_FLAG) is True
    except Exception:
        return False


def _sitctx_note_for(rec):
    """The note to put in the model input for this send, or None. Fails closed: no flag, no record, no
    projection, or anything that is not exactly one well-formed single-line note of at most 400 characters
    gives None, and nothing is injected. Never raises."""
    try:
        if rec is None or not _sitctx_inject_enabled():
            return None
        proj = rec["proj"]
        note = getattr(proj, "note", None)
        if not isinstance(note, str) or not note or len(note) > _SITCTX_NOTE_MAX or "\n" in note:
            return None
        if not (note.startswith("[Situation") and note.endswith(".]")):
            return None
        return note
    except Exception:
        return None


def _sitctx_compose(sent: str, context, text: str, rec):
    """(model input, note characters injected). `sent` is the existing composition (vault + text) and is
    returned UNCHANGED, as the very same string, whenever nothing is injected, so with the flag OFF the model
    input is byte-for-byte what it was before 3C. With a valid note: vault + "\n\n" + note + "\n\n" + text
    (no vault: note + "\n\n" + text). Never raises."""
    note = _sitctx_note_for(rec)
    if note is None:
        return sent, 0
    return "\n\n".join(([context] if context else []) + [note, text]), len(note)


def _mem2_inject_enabled() -> bool:
    """Memory 2.0 injection switch. DEFAULT OFF: only the JSON value `true` turns it on. Never raises."""
    try:
        return memory_shadow.inject_enabled(CFG)
    except Exception:
        return False


async def _mem2_compose(model_input: str, text: str, context):
    """(model input, note characters injected). With the flag OFF (or anything not a clean win, or slower than
    0.3 s) the very same `model_input` string is returned untouched. Otherwise the labelled memory note is put
    immediately before the user's words, which every composition ends with. Never raises, never blocks longer
    than the timeout, never writes."""
    try:
        if not _mem2_inject_enabled() or not model_input.endswith(text):
            return model_input, 0
        note, ids = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(
            None, lambda: memory_shadow.note_for(text, context=context)), 0.3)
        if not note:
            return model_input, 0
        final = model_input[:len(model_input) - len(text)] + note + "\n\n" + text
        log(f"[meminj] flag=on injected=yes ids={','.join(ids)} note_chars={len(note)} "
            f"base_chars={len(model_input)} final_chars={len(final)}")
        return final, len(note)
    except Exception:
        return model_input, 0


def _sitctx_marker(base: str, final: str, note_chars: int) -> None:
    """The send-time audit line, logged BEFORE the send: flag state, whether a note was injected, and three
    integers. Never the note, the text or any id. A separate prefix; the [sitctx] line is untouched."""
    try:
        log(f"[sitinj] flag={'on' if _sitctx_inject_enabled() else 'off'} "
            f"injected={'yes' if note_chars else 'no'} note_chars={int(note_chars)} "
            f"base_chars={len(base)} final_chars={len(final)}")
    except Exception:
        pass


def _sitctx_begin(brain):
    """At the send point, just before ask_stream: one fresh snapshot, one projection. Returns a record
    for _sitctx_end, or None. Reads only; never raises; never touches the prompt."""
    try:
        reader = _SITCTX_READER
        if reader is None:
            return None
        router = brain.router
        identity = _sitctx_identity(brain)
        # a state told into another epoch, session or generation is never reused: that is a fresh baseline
        prior = _SITCTX["state"] if (_SITCTX["epoch"], _SITCTX["session"], _SITCTX["generation"]) == identity else None
        proj = None
        try:
            proj = situation_context_mod.project(reader.snapshot(), now_mono=time.monotonic(),
                                                 state=prior)
        except Exception:
            _sitctx_fail("projection")
        keys = frozenset(router.handles) if router is not None else frozenset()
        return {"router": router, "keys": keys, "prior": prior, "proj": proj, "identity": identity}
    except Exception:
        _sitctx_fail("projection")
        return None


def _sitctx_line(prior, proj, delivery: str) -> str:
    """The shadow log line: kind, fixed field names, length and delivery. Never the note, a value or
    a timestamp."""
    if proj.note is None:
        return f"[sitctx] projection=none delivery={delivery}"
    old = dict(prior.told) if isinstance(prior, situation_context_mod.ProjectionState) else {}
    new = dict(proj.state.told)
    changed = [k for k in _SITCTX_FIELDS if k in new and new.get(k) != old.get(k)]
    baseline = not (isinstance(prior, situation_context_mod.ProjectionState) and prior.baselined)
    kind = ("baseline" if baseline else
            "retraction" if changed and all(new[k] is None for k in changed) else "delta")
    return (f"[sitctx] projection={kind} fields={','.join(changed)} length={len(proj.note)} "
            f"delivery={delivery}")


def _sitctx_end(brain, rec, sent: str) -> None:
    """After the send (also on cancel or error): attribute the send to the ONE new foreground handle
    the router created for it, and adopt the shadow state only if that handle was CLAIMED.

    Attribution safety comes from the COMBINATION of four things, not from any one of them:
      1. prior-handle exclusion: every handle that existed before the send (the key set captured in
         _sitctx_begin) is ignored, so an old handle can never be taken for this send;
      2. the same router instance: if the router was replaced, every handle of the new one counts as new;
      3. exactly one new foreground handle: zero means nothing was sent, several mean "unattributed";
      4. the handle's own `claimed_at`, set by the router for that handle alone when ITS uuid echo
         arrives (a stale or abandoned handle's echo is quarantined and never sets it).
    Sends are serialized (handle() awaits the previous speak task before starting another), which is
    what makes "exactly one new handle" hold.

    ADOPTION additionally requires that the context did not change between the projection and the send:
    the router instance, the handle's epoch, the session id and the generation counter must all equal
    what _sitctx_begin captured. If any differs (a rebuild or a /clear during the send), nothing is adopted
    and the outcome is logged as "unattributed": the shadow fails closed and a baseline is offered again.

    `h.intent == ledger.redact(sent)` is only a WEAK CROSS-CHECK, not an identity proof. The ledger
    keeps just the first 200 characters, so two different requests that share a long prefix (a vault
    recall block, for example) have EQUAL intents. The comparison can reject an obvious mismatch; it can
    never confirm that a handle is this send. Anything that fails the checks above adopts nothing, which
    only means a baseline is offered again later. Reads public router state only; never raises."""
    try:
        if rec is None or rec["proj"] is None:
            return
        router = brain.router
        if router is None:
            return
        seen = rec["keys"] if router is rec["router"] else frozenset()
        new = [h for u, h in list(router.handles.items())
               if u not in seen and getattr(h, "kind", None) == "foreground"]
        if not new:
            return                                       # nothing was actually sent: no projection
        h = new[0] if len(new) == 1 else None
        delivery = "unattributed"
        if h is not None and h.intent == ledger_mod.redact(sent):      # weak cross-check only; see docstring
            delivery = "claimed" if getattr(h, "claimed_at", None) is not None else "unclaimed"
        proj = rec["proj"]
        if delivery == "claimed" and not _sitctx_same_context(brain, rec, h):
            delivery = "unattributed"            # claimed, but not into the context this projection describes
        if delivery == "claimed" and proj.note is not None:
            _SITCTX["state"] = proj.state
            _SITCTX["epoch"], _SITCTX["session"], _SITCTX["generation"] = rec["identity"]
        log(_sitctx_line(rec["prior"], proj, delivery))
    except Exception:
        _sitctx_fail("delivery")


# The microphone mode, switchable live by voice. "ptt" = mic closed
# except while the key is held. "open" = hands-free listening (VAD).
# The key keeps working in open mode: it interrupts, and holding it
# always gets you heard. gen bumps on every switch so an in-flight
# open-mic capture from before the switch gets discarded, never
# processed.
_MIC = {"mode": "ptt", "gen": 0, "btn": False}
# "Zip it": open-mic ambient listening keeps hearing everything, but nothing
# unprompted gets a spoken reply until "unzip" is said. Distinct from _MIC
# (which closes the microphone itself) -- this just withholds the mouth.
_ZIPPED = {"on": False}
# The last thing asked, and when the turn for it started, so a
# follow-on fragment can be stitched onto it instead of replacing
# it. Cleared once a reply is actually spoken.
_LAST_TURN = {"text": "", "at": 0.0}
# How long an unanswered request stays open for stitching. Generous
# on purpose: Yousif thinks out loud and pauses well past the 1.6s
# endpoint, and the real guard is that the mouth must still be
# silent -- once he has heard anything, new speech interrupts.
STITCH_MAX_S = 20.0
# A stitch used to cancel the turn and resend the accumulated text for EVERY
# fragment. Fragments that land a few hundred ms apart therefore sent several
# queries back to back, and the router answered each abandoned one with a CLI
# interrupt; two interrupts a few ms apart killed the CLI (AbortError ->
# stream_lost, reproduced with the real CLI on 2026-10-05). A stitch now
# RETRACTS the turn once and holds the accumulated request until no further
# fragment has arrived for STITCH_HOLD_S, then sends ONE replacement. The hold
# can be extended by later fragments but never past STITCH_HOLD_MAX_S from the
# first retract, so steady chatter cannot starve the request.
STITCH_HOLD_S = 1.3
STITCH_HOLD_MAX_S = 3.0
# 2026-10-25: the 600ms endpoint cut him mid-sentence ~15 times in 25 minutes
# ("...take control of the" / "of the screen."), and every cut started a turn
# that the next fragment then killed and resent. An open-mic utterance that
# plainly is NOT finished (ends on a connector word, a comma or a trail-off) is
# now held BEFORE anything is sent, with the same StitchHold, so the rest of
# the sentence joins it and ONE turn goes out. A finished sentence is sent at
# once, exactly as before: no added latency where it isn't needed.
UNFINISHED_HOLD_S = 1.6
UNFINISHED_HOLD_MAX_S = 6.0
UNFINISHED_DONE_S = 0.5      # once the held sentence reads finished
UNFINISHED_UNSURE_S = 1.0    # the audio model leans "not done" but isn't sure
TURNEND_SURE_P = 0.2         # below this the audio model is confident he isn't done
_UNFINISHED_TAIL = frozenset("""
and or but because cause if than
the a an my your our their its every
of to for with from at into onto under like as
i we they he she
just very um uh er erm hmm
""".split())    # only words that (almost) never end a sentence: "do it", "close that", "log in" are finished


def looks_unfinished(text: str) -> bool:
    """True when an utterance plainly stops mid-sentence: a trail-off ('...'),
    a trailing comma or dash, or a last word that cannot end a sentence ('the',
    'and', 'of', 'like'). Terminal punctuation or any other last word counts as
    finished, so a doubtful case is never delayed. Cannot raise."""
    try:
        s = str(text or "").strip()
        if not s:
            return False
        if s.endswith(("...", "…", ",", "-", "—", "–")):
            return True
        if s[-1] in ".?!":
            return False
        last = re.sub(r"[^a-z']", "", s.split()[-1].lower())
        return last in _UNFINISHED_TAIL
    except Exception:
        return False


def unfinished_wait(text: str, turn_p) -> float:
    """How long to wait for the rest of an open-mic utterance before sending it;
    0.0 = send now. The text check catches a trailing "the"/"and"/"..."; the audio
    model (turnend.py) catches the rest -- a pause mid-thought that Whisper still
    ended with a full stop. turn_p None = the model had no opinion."""
    if looks_unfinished(text):
        return UNFINISHED_HOLD_S
    if turnend.incomplete(turn_p):
        return UNFINISHED_HOLD_S if turn_p < TURNEND_SURE_P else UNFINISHED_UNSURE_S
    return 0.0


class StitchHold:
    """The one held (retracted, not yet resent) stitched request. Pure state
    and a clock: no timer, no task, so there is nothing to leak or fire late.
    The main loop asks remaining() for its wait timeout and take() once the hold
    is due; clear() ends it. `epoch` ties the text to the session it was said
    in: take() refuses it under a different epoch."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.clear()

    def clear(self):
        self.text = ""
        self.epoch = None
        self.n = 0
        self.unfinished = False
        self._deadline = None
        self._hard = None

    @property
    def active(self) -> bool:
        return self._deadline is not None

    def begin(self, text: str, epoch, *, quiet_s: float = STITCH_HOLD_S,
              max_s: float = STITCH_HOLD_MAX_S, unfinished: bool = False) -> str:
        now = self._clock()
        self.text, self.epoch, self.n = text, epoch, 1
        self.unfinished = unfinished      # held before any turn was sent (looks_unfinished)
        self._hard = now + max_s
        self._deadline = min(now + quiet_s, self._hard)
        return self.text

    def add(self, fragment: str, wait: float | None = None) -> str:
        """Append one more fragment, in order, and push the deadline out. A
        pre-send hold waits again while the sentence is still unfinished
        (`wait`, from unfinished_wait() on the new fragment; None = judge the
        text alone) and only briefly once it reads finished."""
        if not self.active:
            raise RuntimeError("no stitched request is being held")
        now = self._clock()
        self.text = f"{self.text} {fragment}".strip()
        self.n += 1
        quiet = STITCH_HOLD_S
        if self.unfinished:
            # nothing was sent yet, so there is no burst of retracts to coalesce
            if wait is None:
                wait = UNFINISHED_HOLD_S if looks_unfinished(self.text) else 0.0
            quiet = wait or UNFINISHED_DONE_S
        self._deadline = min(now + quiet, self._hard)
        return self.text

    def remaining(self):
        """Seconds until the hold is due, or None when nothing is held."""
        if not self.active:
            return None
        return max(0.0, self._deadline - self._clock())

    def due(self) -> bool:
        return self.active and self._clock() >= self._deadline

    def take(self, epoch):
        """(text, n_fragments) when due and still for this epoch, else None.
        Ends the hold either way, so a request can be sent at most once."""
        if not self.due():
            return None
        text, n, held_epoch = self.text, self.n, self.epoch
        self.clear()
        if held_epoch != epoch:
            return None
        return text, n


_HOLD = StitchHold()


def _brain_epoch(brain):
    r = getattr(brain, "router", None)
    return getattr(r, "epoch", None)


async def release_held(hold: StitchHold, brain, handle) -> bool:
    """The main loop's wake-up when the quiet time on a held stitched request may
    have run out. Sends ONE replacement (every fragment, in order) through
    handle(released=True) unless the session it was said in has changed, and
    never more than once: take() ends the hold whether or not it was sent.
    Returns handle()'s verdict (False = quit); True when nothing was sent."""
    if not hold.due():
        return True
    n = hold.n
    got = hold.take(_brain_epoch(brain))
    if got is None:
        log("[turn] held request dropped: its session changed")
        signals.set_state("idle")
        return True
    log(f"[turn] sending one replacement for {n} fragment(s): {got[0]!r}")
    return await handle(got[0], released=True)

def _norm_speech(text):
    """Lowercase, every non-letter to space, collapse. Whisper loves
    interior commas ("yes, confirm"); end-stripping alone misses them."""
    out = []
    for ch in text.lower():
        out.append(ch if "a" <= ch <= "z" else " ")
    return " ".join("".join(out).split())


# Said over the voice, a bare stop silences it without starting a new turn
# (see over_speech in amain). Anything else said over it is a cut-in.
_STOP_WORDS = {"stop", "wait", "enough", "pause", "quiet"}
_STOP_PHRASES = ("hold on", "hang on", "shut up", "be quiet")
_STOP_FILLER = {"okay", "ok", "hey", "please", "sir", "just", "a", "sec",
                "second", "now", "oh", "alright", "right"}


def _is_name(word):
    """The agent's name, allowing Whisper's near misses ('javis')."""
    import difflib
    name = NAME.lower()
    return word == name or (len(word) >= 4 and difflib.SequenceMatcher(
        None, word, name).ratio() >= 0.8)


def _is_stop_phrase(text):
    """'Stop', 'hold on', 'Jarvis!', 'okay wait': nothing but stop words,
    the agent's name, and filler. 'Jarvis, do it' is not a stop; it's an
    instruction."""
    words = _norm_speech(text).split()
    norm = f" {' '.join(words)} "
    for p in _STOP_PHRASES:
        norm = norm.replace(f" {p} ", " stop ")
    rest = norm.split()
    stops = [w for w in rest if w in _STOP_WORDS or _is_name(w)]
    return bool(stops) and all(
        w in _STOP_WORDS or _is_name(w) or w in _STOP_FILLER for w in rest)


_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million"}


def _echo_chars(text):
    """The words' letters run together, with numbers dropped. Whisper
    re-spells the echo: the voice said "Reactor's up, pid ninety one two
    three five, sir." and it came back "React is up. PID 91-235, sir."
    (2026-09-13). _norm_speech already drops digits; number words go too,
    and spacing and contractions stop mattering once the letters are
    joined."""
    return "".join(w for w in _norm_speech(text).split()
                   if w not in _NUMBER_WORDS)


def _is_echo(text, chunks):
    """True when a transcript is the voice's own recent speech heard back
    through the speakers. `chunks` are the texts it recently started
    playing. The echo comes back a chunk at a time (Kokoro pauses between
    chunks), so:
    - naming the agent means the person, unless the voice just said its
      own name
    - one or two words are echo only when a whole short chunk says them;
      the same words inside a longer chunk ('go ahead' in 'want me to go
      ahead?') are the person answering over it
    - longer: echo when either comparison says so. Words: at least 60%
      of its words fall in runs of two or more shared with what was just
      said (this catches echo glued to other room audio). Letters: at
      least 70% of its letters, numbers dropped, fall in runs of four or
      more (this catches the echo Whisper re-spelled, which the word
      check missed and the voice answered itself)."""
    import difflib
    t = _norm_speech(text).split()
    said = [_norm_speech(c).split() for c in chunks]
    s = [w for c in said for w in c]
    if not t or not s:
        return False
    if any(_is_name(w) for w in t) and not any(_is_name(w) for w in s):
        return False
    if len(t) <= 2:
        return any(len(c) <= len(t) + 1
                   and f" {' '.join(t)} " in f" {' '.join(c)} " for c in said)
    words = difflib.SequenceMatcher(None, t, s, autojunk=False)\
        .get_matching_blocks()
    if sum(b.size for b in words if b.size >= 2) >= 0.6 * len(t):
        return True
    tc = _echo_chars(text)
    sc = "".join(_echo_chars(c) for c in chunks)
    if not tc or not sc:
        return False
    letters = difflib.SequenceMatcher(None, tc, sc, autojunk=False)\
        .get_matching_blocks()
    return sum(b.size for b in letters if b.size >= 4) >= 0.7 * len(tc)


# Words only the voice says. The person calls it by name; it calls them boss
# and sir. A transcript carrying these is far likelier to be its own echo.
_VOICE_ONLY_WORDS = {"boss", "sir"}


def _is_echo_after(text, chunks):
    """The echo check for speech heard AFTER the voice stopped. Stricter than
    _is_echo, because a real answer reuses the question's words: asked "is it
    just sitting on a home screen?", the person said "It is on the home
    screen." and _is_echo dropped it (2026-09-14). Here, the voice's own
    vocabulary (boss, sir) still gets the loose check; otherwise:
    - up to four words: echo only inside a short chunk that said them
    - longer: echo when 70% of its words fall in shared runs of three or
      more, or 75% of its letters in shared runs of ten or more"""
    import difflib
    t = _norm_speech(text).split()
    said = [_norm_speech(c).split() for c in chunks]
    s = [w for c in said for w in c]
    if not t or not s:
        return False
    if any(_is_name(w) for w in t) and not any(_is_name(w) for w in s):
        return False
    if _VOICE_ONLY_WORDS & set(t):
        return _is_echo(text, chunks)
    if len(t) <= 4:
        return any(f" {' '.join(t)} " in f" {' '.join(c)} " and len(c) <= len(t) + 2
                   for c in said)
    words = difflib.SequenceMatcher(None, t, s, autojunk=False)\
        .get_matching_blocks()
    if sum(b.size for b in words if b.size >= 3) >= 0.7 * len(t):
        return True
    tc = _echo_chars(text)
    sc = "".join(_echo_chars(c) for c in chunks)
    if not tc or not sc:
        return False
    letters = difflib.SequenceMatcher(None, tc, sc, autojunk=False)\
        .get_matching_blocks()
    return sum(b.size for b in letters if b.size >= 10) >= 0.75 * len(tc)


def _is_name_over_echo(text, chunks):
    """The person said the agent's name over the voice's own echo: "Jarvis"
    glued to a sentence it was just saying. _is_echo lets any name through as
    the person, so the whole echoed sentence went in as their words and got
    answered (2026-09-14, 10:34 PM). Saying the name over it means stop."""
    words = _norm_speech(text).split()
    if not any(_is_name(w) for w in words):
        return False
    rest = [w for w in words if not _is_name(w)]
    return len(rest) >= 4 and _is_echo(" ".join(rest), chunks)


# Words too common to say anything about who said a line.
_MAC_COMMON = {"a", "an", "the", "and", "or", "but", "i", "you", "it", "is",
               "to", "of", "in", "on", "that", "this", "so", "we", "me", "my",
               "be", "are", "was", "for", "with", "just", "like", "yeah"}


def _without_mac_audio(text, seconds, end):
    """The line the mic heard, minus what the Mac itself was playing (a
    video, music, the voice): "" when it was all Mac audio. A line is
    checked when most of its words are in what the Mac played over the same
    stretch (systap.py). Lines under three meaningful words always count as
    the person, so "stand by" gets through a video."""
    if not text or not systap.enabled() or seconds <= 0:
        return text
    import numpy as np
    from backtalk.ears import transcribe
    mic = set(_norm_speech(text).split()) - _MAC_COMMON
    if len(mic) < 3:
        return text
    played = systap.window(end - seconds - 0.6, end + 0.3)
    if played is None or len(played) < 16000 // 2:
        return text
    level = 20 * np.log10(np.sqrt(np.mean(played.astype(np.float32) ** 2)) / 32768 + 1e-9)
    if level < -50:
        return text                      # the Mac was quiet
    heard = transcribe(played)
    mac = set(_norm_speech(heard).split()) - _MAC_COMMON
    overlap = len(mic & mac) / len(mic)
    log(f"[ears] Mac audio check: {overlap:.0%} of the line is what the Mac "
        f"played ({level:.0f} dBFS): {heard[:70]!r}")
    if overlap < 0.6:
        return text
    return _mac_leftover(text, heard)


def _mac_leftover(text, heard):
    """The person's words around the Mac's audio in one line: `text` minus
    the stretches (2+ words in a row) that match `heard`. A leftover stretch
    counts only at 3+ words with 2+ meaningful ones, so a word Whisper heard
    differently doesn't become a line. (2026-09-16: "...or something else
    entirely? Yes, I want you just to remember the word pineapple" was
    dropped whole, the person's answer with it.)"""
    import difflib
    toks = text.split()
    flat = [(w, i) for i, t in enumerate(toks) for w in _norm_speech(t).split()]
    sm = difflib.SequenceMatcher(None, [w for w, _ in flat],
                                 _norm_speech(heard).split(), autojunk=False)
    matched = {flat[k][1] for b in sm.get_matching_blocks() if b.size >= 2
               for k in range(b.a, b.a + b.size)}
    runs, cur = [], []
    for i, t in enumerate(toks):
        if i in matched:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(cur)
    keep = []
    for run in runs:
        words = _norm_speech(" ".join(run)).split()
        if len(words) >= 3 and len(set(words) - _MAC_COMMON) >= 2:
            keep.append(" ".join(run))
    return " ".join(keep)


# ---- THE VOICE CONSOLE: session verbs, spoken. Exact phrases only,
# spoken alone, so ordinary sentences can never trigger them. (Grown
# from a community member's own build shared in the Discord.)
CONSOLE_VERBS = {
    "clear":     ("clear the session", "clear the context",
                  "clear context", "fresh slate", "slash clear"),
    "compact":   ("compact the session", "compact the context",
                  "compact context", "slash compact"),
    "deep":      ("switch to the deep model", "use the deep model",
                  "slash model deep"),
    "fast":      ("switch to the fast model", "use the fast model",
                  "back to the fast model", "slash model fast"),
    "usage":     ("usage report", "slash usage"),
    "micopen":   ("go hands free", "hands free mode",
                  "hands free listening", "open mic", "open the mic"),
    "micptt":    ("push to talk", "push to talk mode",
                  "back to push to talk", "back to the button"),
    # Background-task controls (router.py). "stopbg" stops the background
    # task(s) and leaves the current answer alone; "stopall" stops those AND
    # the current run. Handled ahead of the normal barge-in path in handle().
    "stopbg":    ("stop the background task", "stop the background tasks",
                  "stop background task", "cancel the background task",
                  "kill the background task", "stop the background job"),
    "stopall":   ("stop everything", "cancel everything", "stop all tasks",
                  "stop all background tasks", "kill everything"),
    "zip":       ("zip it", "zip it for a second", "zip it for now",
                  "go quiet", "pipe down", "hold your tongue"),
    "unzip":     ("unzip", "unzip it", "you can talk again",
                  "you can talk now", "talk again", "unzip yourself"),
}
_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _strip_name(norm: str) -> str:
    """A normalized utterance with the agent's name (and Whisper's
    near-misses for it) taken off either end.

    Found live 2026-09-20: "Clear the session Jarvis" silently fell
    through as a normal sentence instead of firing the console command,
    because the match required the phrase completely alone. Check-ins
    ("you there, Jarvis?") need the same treatment, so it lives here
    rather than twice."""
    words = norm.split()
    while words and _is_name(words[0]):
        words = words[1:]
    while words and _is_name(words[-1]):
        words = words[:-1]
    return " ".join(words).strip(" .,!?")


# Extra bare forms people actually say, kept out of CONSOLE_VERBS above so
# the canonical list stays readable. These exist because the exact-match
# table demanded the magic phrase: on 2026-09-22 Yousif said "every time I
# tell him go hands-free or push to talk I have to say the exact same
# phrase or he's not gonna understand it."
_CONSOLE_EXTRA = {"micopen": ("hands free",), "micptt": ("push to talk",)}
# A command is a short instruction, so an utterance much longer than the
# phrase is somebody talking ABOUT it, not issuing it ("I was telling him
# to go hands free earlier"). Short phrases get less room than long ones,
# which is what stops "talk again about the deal" from unzipping.
_CONSOLE_SLACK_SHORT = 2
_CONSOLE_SLACK_LONG = 3
# Whisper mangles command words the same way it mangles the name
# ("push to tug", "push to tuck"). Scored whole-string, and only against
# phrases already length-filtered above, so ordinary speech can't reach it.
_CONSOLE_FUZZY_CUTOFF = 0.75
# "don't go hands free" must never flip the mic.
_CONSOLE_NEGATIONS = {"dont", "do", "not", "never", "without", "instead",
                      "didnt", "wont", "cant", "shouldnt", "wouldnt", "isnt",
                      "doesnt"}


def _negated(words):
    """True if any word negates. Apostrophes are dropped first, straight AND
    curly: this guard compared "don't" against "dont" and so never fired for
    a contraction -- "don't go hands free" flipped the mic, and a spoken "don't
    stop everything" would have stopped everything (found 2026-09-30, when
    "stop everything" became a verb)."""
    return bool(_CONSOLE_NEGATIONS & {w.replace("'", "").replace("\u2019", "")
                                      for w in words})


def _console_phrases(verb):
    return tuple(CONSOLE_VERBS[verb]) + _CONSOLE_EXTRA.get(verb, ())


def apply_mic_direction(d: str) -> bool:
    """A <<mic:...>> stage direction from the model. Returns True if it was
    one and it was applied.

    This exists because a phrase list cannot understand speech. Yousif,
    2026-09-22: "he dosent understand basic stuff like why I have to give
    him an exact command to go hands-free... he need to have more common
    sens." The matcher below is still the instant path for the obvious
    phrasings; this is what catches "stop making me hold the button" and
    every other way a person actually says it. The model already knows what
    he meant -- it just had no way to act on it, and was told to recite a
    magic phrase at him instead."""
    want = d.strip().lower().replace(" ", "").replace("_", "-")
    if not want.startswith("mic:"):
        return False
    want = want[4:]
    mode = ("open" if want in ("hands-free", "handsfree", "open", "openmic")
            else "ptt" if want in ("push-to-talk", "pushtotalk", "ptt",
                                   "button") else None)
    if mode is None:
        log(f"[console] ignoring unknown mic direction {d!r}")
        return False
    if _MIC["mode"] == mode:
        log(f"[console] mic already {mode}, direction {d!r} was a no-op")
        return True
    _MIC["mode"] = mode
    _MIC["gen"] += 1
    _write_config_key("mic_mode", mode)
    log(f"[console] mic_mode -> {mode} (the model did it: {d!r})")
    return True


def _console_direction_verb(d: str) -> str | None:
    """A <<console:...>>, <<model:...>>, or
    <<effort:...>> stage direction from the model — the natural-language
    counterpart to console_match(), for the verbs that still demanded an
    exact spoken phrase. Pure lookup, no side effects: returns the same
    verb string console_match() would, or None if this isn't one of
    these tags.
"""
    raw = d.strip().lower()
    if ":" not in raw:
        return None
    ns, _, val = raw.partition(":")
    ns = ns.strip()
    val = val.strip().replace(" ", "-").replace("_", "-")
    if ns == "console":
        if val in ("clear", "compact", "usage"):
            return val
        if val in ("deep", "model-deep"):
            return "deep"
        if val in ("fast", "model-fast"):
            return "fast"
        return None
    if ns == "model":
        if val == "deep":
            return "deep"
        if val == "fast":
            return "fast"
        return None
    if ns == "effort":
        if val in _EFFORTS:
            return f"effort:{val}"
        return None
    return None


def console_match(text):
    norm = " ".join(text.lower().replace("-", " ").split()).strip(" .,!?")
    norm_no_name = _strip_name(norm)
    for verb, phrases in CONSOLE_VERBS.items():
        if norm in phrases or norm_no_name in phrases:
            return verb
    for lvl in _EFFORTS:
        if norm in (f"set effort to {lvl}", f"effort {lvl}",
                    f"slash effort {lvl}"):
            return f"effort:{lvl}"
    # Exact match missed. Try the command phrase sitting INSIDE a natural
    # sentence next ("switch to hands free", "push to talk please"), then
    # fall back to sound-alikes. Every pass is length-guarded, and the
    # whole thing is skipped for a negated sentence.
    candidates = [c for c in (norm, norm_no_name)
                  if c and not _negated(c.split())]
    for cand in candidates:
        words = cand.split()
        for verb in CONSOLE_VERBS:
            for phrase in _console_phrases(verb):
                pw = phrase.split()
                slack = (_CONSOLE_SLACK_SHORT if len(pw) <= 2
                         else _CONSOLE_SLACK_LONG)
                if len(words) > len(pw) + slack:
                    continue
                for i in range(len(words) - len(pw) + 1):
                    if words[i:i + len(pw)] == pw:
                        return verb
    # zip/unzip are deliberately NOT sound-alike-matched here: they have
    # their own matcher below with a curated table and a two-word ceiling.
    # Letting the general pass at them made "skip it" and "ship it" zip,
    # which are ordinary things to say.
    best, score = None, 0.0
    for cand in candidates:
        words = cand.split()
        for verb in CONSOLE_VERBS:
            if verb in ("zip", "unzip"):
                continue
            for phrase in _console_phrases(verb):
                pw = phrase.split()
                slack = (_CONSOLE_SLACK_SHORT if len(pw) <= 2
                         else _CONSOLE_SLACK_LONG)
                if len(words) > len(pw) + slack:
                    continue
                ratio = difflib.SequenceMatcher(None, cand, phrase).ratio()
                if ratio > score:
                    best, score = verb, ratio
    if score >= _CONSOLE_FUZZY_CUTOFF:
        log(f"[console] heard {text!r} as {best} (sound-alike, {score:.2f})")
        return best
    return _zip_soundalike(text)


# Whisper's spellings of "zip it" and "unzip", from this machine's log
# (2026-09-16: "Java's... Zibbit.", "I said, exhibit.", "Jarvis Zibit" never
# zipped). Matched only as the whole line, after dropping the agent's name
# and "I said", so "exhibit" inside a sentence never zips.
_ZIP_SOUNDALIKES = {
    "zip": {"zip", "zip it", "zip it up", "zipit", "zippit", "zibit",
            "zibbit", "zip bit", "exhibit"},
    "unzip": {"unzip", "unzip it", "un zip", "unzipped", "unzipit",
              "unzippit", "unzibit", "unzibbit"},
}
_ZIP_LEAD = {"i", "said", "s", "hey", "okay", "ok"}
# How close a bare line has to sound to a known zip form before it counts,
# and how short it has to be. Two words is the ceiling because a zip is
# always said alone -- that is precisely what keeps "skip it", "ship it",
# "split it", "so be it" and "submit" from ever zipping. Verified against
# all of those plus "is it", "did it", "sit" and "zoom in": none match.
_ZIP_FUZZY_CUTOFF = 0.8
_ZIP_FUZZY_MAX_WORDS = 2


def _zip_filler(word):
    # the name as Whisper mangles it here ("java", "javanis"); safe
    # because what's left must be a zip phrase and nothing else
    return word in _ZIP_LEAD or _is_name(word) or word.startswith("jav")


def _zip_soundalike(text):
    words = _norm_speech(text).split()
    # the name and filler go wherever they are: "Zip it, Jarvis." and
    # "Zip it, Jarvis, zip it." both failed to zip (2026-09-16)
    words = [w for w in words if not _zip_filler(w)]
    half = len(words) // 2
    if half and len(words) % 2 == 0 and words[:half] == words[half:]:
        words = words[:half]         # said twice: "zip it, zip it"
    phrase = " ".join(words)
    for verb, forms in _ZIP_SOUNDALIKES.items():
        if phrase in forms:
            return verb
    # While zipped, Whisper hears "un" as "and", "en", "ends": four tries
    # at unzip were swallowed before one landed (2026-09-16). Zipping again
    # means nothing then, so these count as unzip.
    if _ZIPPED["on"] and words:
        if phrase in _UNZIP_WHILE_ZIPPED:
            return "unzip"
        if (words[0] in _UNZIP_HEADS and
                " ".join(words[1:]) in _ZIP_SOUNDALIKES["zip"] | {"zipped"}):
            return "unzip"
    # Last resort, added 2026-09-22. Whisper swaps s and z freely -- they are
    # the same sibilant, voiced and unvoiced -- so "Jarvis zip it" came back
    # as "Java Sibit" twice while Yousif was trying to shut the voice up, and
    # was ignored both times. One consonant: "Java Zibit" zipped fine. Adding
    # another literal every time it mangles a new way is whack-a-mole (that
    # cycle already ran on 2026-09-16), so score the line against the known
    # forms instead and the whole sound-alike class lands at once. Being
    # ignored is the expensive failure here; a stray zip costs one "unzip".
    if words and len(words) <= _ZIP_FUZZY_MAX_WORDS:
        best, score = None, 0.0
        for verb, forms in _ZIP_SOUNDALIKES.items():
            for form in forms:
                ratio = difflib.SequenceMatcher(None, phrase, form).ratio()
                if ratio > score:
                    best, score = verb, ratio
        if score >= _ZIP_FUZZY_CUTOFF:
            log(f"[console] heard {phrase!r} as {best} "
                f"(sound-alike, {score:.2f})")
            return best
    return None


# Whisper's spellings of the name beyond what _is_name allows (this log).
_NAME_MISHEARD = {"java", "javanis", "jarvus", "jervis"}


def _names_agent(text):
    """The line calls the agent by name, anywhere in it."""
    return any(_is_name(w) or w in _NAME_MISHEARD
               for w in _norm_speech(text).split())


def _fix_name(text):
    """Whisper's misspellings of the name, written as the name, so the brain
    knows it was addressed ("Java's put it in my hand." got "Still not for
    me, sir." at 6:52 PM on 2026-09-16)."""
    def sub(m):
        w = m.group(0)
        base = re.sub(r"'s$", "", w.lower())
        if base == NAME.lower():
            return w
        if _is_name(base):
            return NAME
        # real words ("java") only where he'd be called: first, before a
        # comma, or as "Java's"; "some Java code" stays as said
        if base in _NAME_MISHEARD and (
                not text[:m.start()].strip(" .,!?") or w.endswith("'s")
                or text[m.end():m.end() + 1] == ","):
            return NAME
        return w
    return re.sub(r"[A-Za-z]+(?:'s)?", sub, text)


_UNZIP_HEADS = {"and", "an", "en", "in", "on", "un", "ends"}
_UNZIP_WHILE_ZIPPED = {"enzip", "inzip", "anzip", "onzip", "enzipped",
                       "enzip it", "ends a bit", "ends it", "enzibbit",
                       "enzibit"}


def _write_config_key(key, value):
    """The agent rewrites the config; the person never hand-edits it.
    Returns True on a persisted write. A file that fails to PARSE is
    left untouched (rewriting from {} would wipe every other setting);
    the in-memory CFG updates either way so the session behaves."""
    from backtalk.config import CONFIG_PATH
    CFG[key] = value
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log(f"[console] config not writable/parsable, session-only: {e}")
        return False
    data[key] = value
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log(f"[console] config write failed, session-only: {e}")
        return False
    return True


def _fmt_tokens(n):
    if n >= 1_000_000:
        return f"about {round(n / 1_000_000, 1):g} million tokens"
    if n >= 1000:
        return f"about {round(n / 1000)} thousand tokens"
    return f"{n} tokens"


def _spoken_usage(sess, ctx_usage):
    """A short CFO brief of the session, written for the ear: plain
    numerals only (the TTS reads "40" fine; symbols come out garbled)."""
    turns = sess["turns"]
    parts = [f"{turns} turn{'s' if turns != 1 else ''} this session",
             _fmt_tokens(sess["out_tokens"]) + " spoken out"]
    cents = round(sess["cost"] * 100)
    if cents >= 1:
        parts.append(f"roughly {cents} cents" if cents < 100
                     else f"roughly {round(cents / 100)} dollars")
    try:
        cats = (getattr(ctx_usage, "categories", None)
                or (ctx_usage or {}).get("categories") or [])
        # the breakdown includes "Free space" and the autocompact
        # buffer; only OCCUPIED categories belong in the spoken number
        total = sum(int(c.get("tokens") or 0) for c in cats
                    if isinstance(c, dict)
                    and "free" not in str(c.get("name", "")).lower()
                    and "buffer" not in str(c.get("name", "")).lower())
        if total:
            parts.append(_fmt_tokens(total)
                         + " sitting in the context window")
    except Exception:
        pass
    return ". ".join(parts) + "."

_PASTE_ON = "\x1b[200~"    # bracketed-paste markers (we enable the mode below)
_PASTE_OFF = "\x1b[201~"


# <<anything>> is a stage direction: lifted out, never spoken, published on
# the bus when the audio carrying it starts. Bounded so a runaway model cannot
# swallow a paragraph into one "tag".
_DIRECTION_TAG = re.compile(r"<<([^<>]{1,80})>>")


def _clean_typed(line: str) -> str:
    """Scrub terminal-copy artifacts: blockquote gutter glyphs and stray
    whitespace (copying from a CLI chat render drags bars along)."""
    line = line.strip()
    while line[:1] in ("▎", "│", ">"):
        line = line[1:].lstrip()
    return line


def _join_paste(body: str) -> str:
    """Pasted blob -> one clean message (gutters scrubbed, lines joined)."""
    parts = [_clean_typed(l) for l in body.split("\n")]
    return " ".join(" ".join(p for p in parts if p).split())


def _typed_reader_pipe(q: "queue.Queue[str]", fd: int):
    """Non-tty stdin (pipes/tests): line assembly with paste markers."""
    import os
    pend = ""
    while True:
        try:
            b = os.read(fd, 65536)
        except OSError:
            return
        if not b:
            return
        pend += b.decode("utf-8", "replace")
        while True:
            if _PASTE_ON in pend:
                if _PASTE_OFF not in pend:
                    break
                head, rest = pend.split(_PASTE_ON, 1)
                body, pend = rest.split(_PASTE_OFF, 1)
                *hlines, hpart = head.split("\n")
                for l in hlines:
                    l = _clean_typed(l)
                    if l:
                        q.put(l)
                text = _join_paste(hpart + body)
                if text:
                    q.put(text)
                continue
            if "\n" in pend:
                line, pend = pend.split("\n", 1)
                line = _clean_typed(line)
                if line:
                    q.put(line)
                continue
            break


def _typed_reader_simple(q: "queue.Queue[str]"):
    """Windows (no termios): plain line input on a thread. Pastes work;
    they just echo normally instead of collapsing to a count."""
    while True:
        try:
            line = _clean_typed(input())
        except (EOFError, OSError):
            return
        if line:
            q.put(line)


def _typed_reader(q: "queue.Queue[str]"):
    """Terminal stdin -> typed messages (daemon thread). Typed lines are
    first-class turns: same pipeline as a spoken utterance, spoken reply.

    On a POSIX tty we OWN the input line (cbreak: no kernel echo, no
    canonical buffering — the little line editor below echoes keys,
    handles backspace, and assembles bracketed pastes invisibly). The
    kernel's canonical mode is unfixable for pastes: it echoes the
    markers as visible junk and holds unfinished marker lines hostage.
    Pastes show as `[pasted N chars]`; Enter sends everything as ONE
    message. Ctrl-C still works (ISIG stays on); termios restored at
    exit."""
    import atexit
    import os
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        _typed_reader_pipe(q, fd)
        return
    try:
        import termios
        import tty as _tty
    except ImportError:            # Windows: no termios — simple reader
        _typed_reader_simple(q)
        return
    old = termios.tcgetattr(fd)
    _tty.setcbreak(fd)                      # ECHO+ICANON off, ISIG kept
    sys.stdout.write("\x1b[?2004h")         # bracket pastes, please
    sys.stdout.flush()

    def _restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    atexit.register(_restore)
    audiowatch.before_restart(_restore)     # exec skips atexit

    MARKS = (_PASTE_ON, _PASTE_OFF)

    def _partial_tail(s: str) -> int:
        """Length of a trailing partial paste-marker (hold it for the
        next read)."""
        for m in MARKS:
            for k in range(min(len(s), len(m) - 1), 0, -1):
                if m.startswith(s[-k:]):
                    return k
        return 0

    buf = ""          # the input line being composed
    paste = None      # accumulating paste body, or None
    pend = ""
    while True:
        try:
            b = os.read(fd, 4096)
        except OSError:
            _restore()
            return
        if not b:
            _restore()
            return
        pend += b.decode("utf-8", "replace")
        keep = _partial_tail(pend)
        proc = pend[:len(pend) - keep] if keep else pend
        pend = pend[len(pend) - keep:] if keep else ""
        i = 0
        while i < len(proc):
            if paste is not None:
                j = proc.find(_PASTE_OFF, i)
                if j < 0:
                    paste += proc[i:]
                    break
                paste += proc[i:j]
                i = j + len(_PASTE_OFF)
                text = _join_paste(paste)
                paste = None
                if text:
                    if buf and not buf.endswith(" "):
                        buf += " "
                    buf += text
                    sys.stdout.write(text if len(text) <= 60
                                     else f"[pasted {len(text)} chars]")
                    sys.stdout.flush()
                continue
            if proc.startswith(_PASTE_ON, i):
                paste = ""
                i += len(_PASTE_ON)
                continue
            ch = proc[i]
            i += 1
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = buf.strip()
                buf = ""
                if line:
                    q.put(line)
            elif ch in ("\x7f", "\x08"):     # backspace
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch >= " " or ch == "\t":    # printable: echo + collect
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()


# --- The turn never goes silent on him -------------------------------------
# From the 66 utterances that got no reply at all between 2026-09-19 and 21:
#   * 26 of them were Yousif speaking again WHILE a turn worked, which
#     cancelled the turn that was about to answer him. Now a check-in keeps
#     the turn alive and gets a status line instead (see _is_checkin).
#   * 7 came back empty and said nothing and logged nothing (see speak_reply).
# (A third fix, a spoken narrator for long silent tool turns, shipped
# 2026-09-22 and was killed 2026-09-23 -- Yousif still heard it as robotic
# filler after the "don't announce" prompt fix, so it's gone rather than
# tuned again.)

# What a tool is, said out loud. The check-in answer reads from here, so
# Jarvis describes his work the same way every time.
_TOOL_SPEECH = {
    "Read": "reading a file", "Glob": "looking through the files",
    "Grep": "searching the files", "Bash": "running a command",
    "Edit": "making a change", "Write": "writing a file",
    "NotebookEdit": "editing a notebook", "TodoWrite": "sorting the list",
    "WebSearch": "searching the web", "WebFetch": "reading a web page",
    "Task": "running a background job", "AskUserQuestion": "thinking it over",
}


def _tool_speech(name: str) -> str:
    if not name:
        return "still thinking it through"
    if name in _TOOL_SPEECH:
        return _TOOL_SPEECH[name]
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) > 1:
            return "using " + parts[1].replace("-", " ").replace("_", " ")
    return "working on it"


_STATUS_OPENERS = ("Still on it, boss", "Yeah, still working",
                   "Still digging, boss", "Hang tight")


class TurnProgress:
    """What the turn in flight is actually doing.

    So a check-in ("you there?") can be answered with the truth instead of
    killing the turn to ask. Written by speak_reply, read by handle() --
    one event loop, plain attributes, no locking needed."""

    def __init__(self):
        self.start()

    def start(self):
        self.t0 = time.monotonic()
        self.tool = ""
        self.tools = 0
        self.said = 0             # chunks of REAL reply spoken this turn
        self.checkins = 0
        self.narrated = 0
        self.last_voice = self.t0

    def on_tool(self, name: str):
        self.tool = name
        self.tools += 1

    def on_speech(self):
        self.said += 1
        self.last_voice = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def status(self, n: int = 0) -> str:
        """One spoken sentence: what is happening right now, for real.

        Each opener in _STATUS_OPENERS is used at most once per turn; once
        they're all spent, later narrations drop the opener and just state
        the facts rather than cycling back to "Still on it, boss" verbatim
        -- a repeated canned phrase is exactly the "call-centre script"
        the spoken-delivery discipline warns against."""
        self.last_voice = time.monotonic()
        secs = int(self.elapsed)
        doing = _tool_speech(self.tool)
        prefix = f"{_STATUS_OPENERS[n]} — " if n < len(_STATUS_OPENERS) else ""
        if self.tools >= 2:
            return (f"{prefix}{secs} seconds in, {doing}, "
                    f"{self.tools} steps so far.")
        return f"{prefix}{secs} seconds in, {doing}."


# Short "are you still alive" utterances. Said while a turn is WORKING,
# these used to cancel it -- he asked "did you find it?", which killed the
# turn that was about to answer him.
#
# The list is deliberately tight, and nothing that could carry information
# belongs in it: a false positive answers a real instruction with "still
# working" and then drops it. Matched only against the whole utterance,
# after the agent's name is stripped off either end.
_CHECKINS = {
    "hello", "hey", "you there", "are you there", "still there",
    "you still there", "are you still there", "you alive", "are you alive",
    "youre there", "youre still there", "youre quiet", "still alive",
    "did you find it", "you find it", "did you find anything",
    "did you get it", "any luck", "anything yet", "still working",
    "are you working", "you working", "still on it", "are you still on it",
    "you good", "what is taking so long", "whats taking so long",
    "whats the holdup", "whats going on",
    "taking so long", "still with me", "you still with me",
    "are you still with me", "you busy", "are you busy",
}

# Before this much of a turn has passed it is not a check-in, it is him
# still talking.
CHECKIN_MIN_S = 2.0


def _is_checkin(text: str) -> bool:
    """A "you still alive?" with no instruction in it.

    Apostrophes are dropped before matching, straight AND curly: Whisper
    emits the curly one constantly, so "what's taking so long" and
    "what’s taking so long" both have to land on the same entry
    without the set carrying three spellings of every contraction."""
    norm = text.lower().replace("-", " ").replace("’", "").replace("'", "")
    norm = " ".join(norm.split()).strip(" .,!?")
    bare = _strip_name(norm)
    if not bare:
        return _names_agent(text)      # just his name: "Jarvis?"
    return bare in _CHECKINS


def _has_stop_word(text: str) -> bool:
    """A real stop word or stop phrase in the line, not just the agent's name.
    _is_stop_phrase counts the bare name as a stop (said over the voice,
    "Jarvis!" means shut up); said into silence it is a check-in, so the two
    must be told apart."""
    norm = f" {' '.join(_norm_speech(text).split())} "
    for p in _STOP_PHRASES:
        norm = norm.replace(f" {p} ", " stop ")
    return any(w in _STOP_WORDS for w in norm.split())


def is_stop_command(text: str) -> bool:
    """"Stop", "wait", "hold on", "okay stop, Jarvis": nothing but stop words,
    the name and filler, with at least one actual stop word."""
    return _has_stop_word(text) and _is_stop_phrase(text)


def decide_during_turn(text: str, *, mouth_busy: bool, elapsed_s: float,
                       last_turn_text: str, last_turn_age_s: float) -> str:
    """What a new utterance means while a turn is in flight. One of:

      "checkin"   "did you find it?" into dead air: answer it, keep the turn.
      "stop"      a bare stop word: end the turn, say nothing new. This used to
                  be "stitch" while the voice was silent (a tool running) and
                  inside the stitch window, so "stop" became extra words on the
                  request and the work went on; and a fresh model turn
                  otherwise. Stop only worked while the voice was audible.
      "stitch"    the back half of the same request, nothing answered yet:
                  fold it into the unanswered request (one turn, not two).
      "interrupt" anything else: a real cut-in; the turn is replaced.

    Order matters and is the old order with "stop" added before "stitch"."""
    if _is_checkin(text) and not mouth_busy and elapsed_s >= CHECKIN_MIN_S:
        return "checkin"
    if is_stop_command(text):
        return "stop"
    if (last_turn_text and last_turn_age_s <= STITCH_MAX_S
            and (not mouth_busy or not _is_stop_phrase(text))):
        return "stitch"
    return "interrupt"


async def stop_current_turn(brain: WarmBrain, mouth: Mouth,
                            task: "asyncio.Task | None", *,
                            voice_was_busy: bool) -> float:
    """End the turn in flight because the person said stop. Returns how long
    the cancel took.

    The turn's own cancel handler calls brain.interrupt(), which is where the
    background-task protection lives: with a protected background task alive the
    run is muted and left to finish, and the task is NOT killed ("stop
    everything" is the verb that kills tasks). No new model turn is started.
    If the voice was already silent (a tool was working) there is no sound to
    tell him it worked, so say so; if it was talking, silence is the answer."""
    t0 = time.monotonic()
    if task is not None and not task.done():
        task.cancel()
    mouth.shut_up()
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    signals.static_stop()
    signals.set_state("idle")
    if not voice_was_busy:
        mouth.say("Stopped.")
    took = time.monotonic() - t0
    log(f"[timing] stop: the turn was cancelled in {took:.2f}s")
    return took


async def _log_time_to_sound(mouth: Mouth, t0: float, cap_s: float = 20.0):
    """Wait for the first audio to actually leave the speaker and log how
    long that really took. Cheap poll; dies quietly if nothing ever plays
    (a cancelled turn, a zipped reply)."""
    try:
        while time.time() - t0 < cap_s:
            if mouth.audible:
                log(f"[turn] {time.time() - t0:.1f}s to SOUND "
                    f"(what he actually waited)")
                return
            await asyncio.sleep(0.02)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def speak_reply(brain: WarmBrain, mouth: Mouth, text: str,
                      progress: "TurnProgress | None" = None,
                      addressed: bool = True,
                      vault_fut: "asyncio.Future | None" = None,
                      run_console=None):
    """Every sentence ships the moment it's ready -- no holding a spoken
    sentence back to wait for the next one. The TTS synthesis pipeline
    (Mouth._synth_run) still renders chunks ahead of playback in the
    background, so this doesn't cost anything in choppiness; it just
    stops adding an artificial "wait for sentence 3" delay before
    sentence 2 is allowed to play.

    `vault_fut`, if given, is a vault-recall executor future started
    EARLIER by the caller (handle()), so its ~0-500ms cost overlaps with
    other per-turn setup instead of sitting fully in series here. If not
    given, one is started now -- same behavior as before this existed.

    `run_console`, if given, is handle()'s console-verb executor. A
    <<console:...>>/<<model:...>>/<<effort:...>>
    direction is only ever CAPTURED here, never executed mid-stream --
    it runs after ask_stream's loop fully completes, once the pipe is
    aligned again (brain._dirty is False). Firing brain.command() while
    ask_stream is still iterating the SAME client's receive_response()
    is exactly the off-by-one desync brain.reset_turn() exists to
    recover from; deferring avoids ever causing it."""
    t0 = time.time()
    first = True
    pending: list[str] = []          # directions waiting for their chunk
    console_verb: str | None = None  # captured, run only after the loop

    def emit(raw: str):
        nonlocal first, pending, console_verb
        # STAGE DIRECTIONS: your agent may write <<anything>> inline. It is
        # lifted out here, never spoken, and published on the signal bus when
        # this chunk's audio starts (signals.direction). backtalk has no
        # opinion on what a direction means; something watching the bus does.
        #
        # This used to strip only the ANGLE BRACKETS, which left the tag body
        # in the sentence and the TTS read it aloud.
        found = _DIRECTION_TAG.findall(raw)
        if found:
            # A mic switch or console verb is CONSUMED here, not published:
            # both are controls, not a stage direction for a face to react
            # to. First console verb in a turn wins; extras are ignored
            # rather than queued, since only one action makes sense per turn.
            remaining = []
            for d in found:
                if apply_mic_direction(d):
                    continue
                v = _console_direction_verb(d)
                if v:
                    if console_verb is None:
                        console_verb = v
                    else:
                        log(f"[console] ignoring extra direction {d!r} -- "
                            f"already running {console_verb!r} this turn")
                    continue
                remaining.append(d)
            pending += [d.strip() for d in remaining if d.strip()]
        raw = _DIRECTION_TAG.sub(" ", raw)
        # TTS hygiene: backticks and markdown fences are never speakable.
        s = " ".join(raw.replace("`", "").split()).strip()
        if not s:
            return
        if progress:
            progress.on_speech()
        if first:
            log(f"[{NAME}] ({time.time()-t0:.1f}s to first) {s}"
                + (f"  <directions: {pending}>" if pending else ""))
            # The line above is when the TEXT was ready, which is NOT when he
            # hears anything: TTS synthesis still has to run (measured
            # 0.5-1.2s on this machine). Reporting text time as "speed" hid
            # the real gap for days, and the thinking sound covered what was
            # left. Time the sound itself, so the number means what it says.
            asyncio.create_task(_log_time_to_sound(mouth, t0))
            first = False
        else:
            log(f"[{NAME}] {s}" + (f"  <directions: {pending}>" if pending else ""))
        mouth.say_chunk(s, pending)
        pending = []

    try:
        # Item 6, Jarvis Feel and Speed Plan: pull relevant vault matches
        # in automatically, off the event loop so a slow/dead Ollama can
        # never stall a turn. context_for() enforces its own similarity
        # floor and token cap; None here just means nothing scored well
        # enough to be worth injecting, which is the normal case.
        if vault_fut is None:
            vault_fut = asyncio.get_running_loop().run_in_executor(
                None, vaultrecall.context_for, text)
        vault_wait_t0 = time.monotonic()
        try:
            context = await asyncio.wait_for(vault_fut, 0.5)
        except Exception:
            context = None
        # How long THIS turn actually blocked on it here -- not the total
        # recall time, which may have started earlier in handle() and
        # already been partly hidden behind reset_turn()/board.changes().
        log(f"[timing] vault recall blocked speak_reply for "
            f"{time.monotonic() - vault_wait_t0:.2f}s"
            + (", found" if context else ", nothing"))
        sent = f"{context}\n\n{text}" if context else text
        if _SITCTX_READER is not None:
            # Settle the session BEFORE the shadow captures its context identity: reset_turn() is the same
            # idempotent check handle() already ran (a rebuild only if one is due, no suspension otherwise),
            # so no rebuild can land between that capture and the send. ask_stream's own check then agrees.
            await brain.reset_turn()
        _mem2_shadow(text, context)         # Memory 2.0 shadow (default OFF): one id-only log line, never model input
        shadow = _sitctx_begin(brain)       # shadow capture: always runs; `sent` itself is never changed by it
        # Phase 3C: with situation_context_inject on (default OFF) the situation note joins the model input
        # here and nowhere else; otherwise model_input IS `sent`. The audit line is written before the send.
        model_input, injected = _sitctx_compose(sent, context, text, shadow)
        _sitctx_marker(sent, model_input, injected)
        send_input, _mem_chars = await _mem2_compose(model_input, text, context)   # Memory 2.0 inject (default OFF)
        try:
            async for sentence in brain.ask_stream(
                    send_input, on_tool=progress.on_tool if progress else None):
                emit(sentence)
        finally:
            _sitctx_end(brain, shadow, send_input)
        # A captured console verb runs NOW, never earlier: ask_stream's
        # loop just ended, so its ResultMessage was already consumed
        # (brain._dirty is False) and the shared client pipe is aligned.
        # Routed through the SAME run_console() the exact/fuzzy spoken
        # path uses, so every safeguard applies identically.
        if console_verb and run_console:
            log(f"[console] natural-language direction -> {console_verb!r}")
            await run_console(console_verb)
        if first and not console_verb:
            # Zero sentences yielded: nothing will ever dequeue, so nothing
            # resets the bus — park it here.
            #
            # And SAY so, when it was meant for him. Two very different
            # things land here and they must not be treated alike:
            #   * open-mic judgment deliberately answering ambient room
            #     talk with silence, which the spoken discipline explicitly
            #     asks for — that must STAY silent;
            #   * a turn that genuinely came back with nothing, which used
            #     to vanish with no word and no log line at all. 7 of the 66
            #     unanswered utterances between 2026-09-19 and 21 were this,
            #     and from the outside it is indistinguishable from Jarvis
            #     being broken.
            # `addressed` is the difference: push-to-talk, or his name in it.
            # A console verb with no other words (e.g. just the tag) is
            # NOT this case either -- something real happened, run_console
            # already spoke its own confirmation.
            log(f"[{NAME}] no reply — empty turn after {time.time()-t0:.1f}s"
                + (", and it was meant for me" if addressed
                   else ", ambient — staying quiet on purpose"))
            if addressed:
                mouth.say("That one came back empty, boss. Say it again?")
            signals.static_stop()
            signals.set_state("idle")
    except asyncio.CancelledError:
        t_int = time.monotonic()
        verdict = "error"
        try:
            verdict = await brain.interrupt()
        except Exception:
            pass
        log(f"[timing] interrupt {verdict} in {time.monotonic() - t_int:.2f}s")
        raise
    except TurnFailed as e:
        # A turn the router ended as FAILED, with a reason it knows (an
        # oversized tool result, a lost stream, a step limit). Say WHICH, in
        # words built here, instead of the generic apology.
        log(f"[{NAME}] the turn failed: {e.reason}")
        _log_recovery(brain, e)
        signals.static_stop()
        signals.set_state("idle")
        try:
            mouth.say(SP.failure_line(e.reason))
        except Exception:
            pass
    except Exception as e:
        # A turn that dies on an exception used to leave the bus spinning
        # and Yousif staring at silence with nothing in the log either.
        log(f"[{NAME}] the turn failed: {e!r}")
        signals.static_stop()
        signals.set_state("idle")
        try:
            mouth.say("Something broke on my end, boss, and that turn died. "
                      "Say it again?")
        except Exception:
            pass


# ---- session-control commands: a confirmation is spoken only for a command that succeeded ----

COMMAND_FAILED_LINE = "That one didn't go through. Say it again?"
# The CLI's own wording for a command it carried out (seen in the SDK drain log and in session
# transcripts: "Set model to `Opus 5` for this session only", "Set effort level to low ..."). /clear
# answers with nothing and /compact's wording is not pinned for the SDK path, so neither is read for it.
_COMMAND_REPLY = {"deep": "Set model to", "fast": "Set model to", "effort": "Set effort level to"}


def command_snapshot(brain):
    """What the ledger and the session looked like BEFORE a session-control command is sent."""
    return set(brain.ledger.turns), brain.session_id


def command_succeeded(brain, before, verb, resp) -> bool:
    """Did the command this console verb sent actually succeed? Only evidence counts, never the absence of
    an error: the command's own turn (new since `before`) reached DONE, which an abort, an interrupt, a
    timeout, a closed epoch or a failed turn never does; /clear also started a NEW session (the CLI's reply
    to a /clear carries the new id, 4 of 4 in the production ledger); /model and /effort also carry the
    CLI's own "Set ..." reply. An empty reply is not success on its own."""
    seen, sid = before
    turns = [t for i, t in brain.ledger.turns.items() if i not in seen and t.get("kind") == "command"]
    if not turns or turns[-1].get("state") != ledger_mod.T_DONE:
        return False
    if verb == "clear":
        return bool(brain.session_id) and brain.session_id != sid
    want = _COMMAND_REPLY.get(verb.split(":", 1)[0])
    if want:
        return (resp or "").lstrip().startswith(want)
    return True


def command_confirmation(brain, before, verb, resp, say_after) -> str:
    """The one line to speak after a session-control command: the CLI's own error text when it sent one,
    else `say_after` only if the command succeeded, else the failure line."""
    low = (resp or "").lower()
    if resp and ("error" in low or "invalid" in low):
        return resp[:160]
    if not command_succeeded(brain, before, verb, resp):
        return COMMAND_FAILED_LINE
    return say_after


# ---- background-task voice controls (module level so they can be tested) ------

async def do_stop_background(brain, mouth):
    """"Stop the background task": a scoped stop; the current answer keeps
    going. What is said is built from the ledger, and "Stopped X" is only said
    (by the brain) once the ledger has seen the task stop."""
    live = brain.live_background()
    if not live:
        mouth.say("There are no background tasks running.")
        return
    if len(live) > 1:
        names = ", ".join(SP.sanitize(t.get("description"), 40) or "a task"
                          for t in live[:4])
        mouth.say(f"There are {len(live)} background tasks: {names}. "
                  "Say stop everything, or tell me which.")
        return
    t = live[0]
    try:
        await brain.stop_background(t["id"])
    except Exception as e:
        log(f"[bg] stop_task failed: {e!r}")
        mouth.say("I couldn't stop it.")
        return
    for _ in range(30):                     # the confirmation is the notification
        await asyncio.sleep(0.1)
        if brain.ledger.tasks[t["id"]]["state"] not in ledger_mod.TASK_LIVE:
            return                          # the brain speaks "Stopped X." itself
    mouth.say("I asked it to stop, but I haven't seen it stop yet.")


async def do_stop_everything(brain, mouth, task=None):
    """"Stop everything": silence the mouth, stop every background task by
    name, then the current run. `task` is the running turn's asyncio task, if
    any; it is cancelled here (its own handler calls brain.interrupt(), which
    is forced past barge-in protection for this).

    What is said is built from the ledger. A task counts as stopped only when its
    STOPPED_BY_USER state was SEEN: a stop that was requested, acknowledged or
    interrupted is not a stop (a task can ignore it, fail, finish, or be lost). The
    tasks judged are the ones live when the request came in."""
    targets = [t["id"] for t in brain.live_background()]
    mouth.shut_up()
    mouth.release(resume=False)
    brain.force_interrupt = True
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except BaseException:
            pass
    verdict = "none"
    try:
        _, verdict = await brain.stop_everything()
    finally:
        brain.force_interrupt = False
    signals.static_stop()
    signals.set_state("idle")
    for _ in range(30):                     # the confirmation is the ledger seeing each one stop
        if all(brain.ledger.tasks[i]["state"] not in ledger_mod.TASK_LIVE for i in targets):
            break
        await asyncio.sleep(0.1)
    stopped = sum(1 for i in targets if brain.ledger.tasks[i]["state"] == ledger_mod.K_STOPPED_USER)
    missing = len(targets) - stopped
    run = " and the current request" if task is not None and verdict != "timeout" else ""
    if missing and not stopped:
        it, they = ("it", "it") if missing == 1 else ("them", "they")
        mouth.say(("Stopped the current request. " if run else "")
                  + f"I asked {it} to stop, but I couldn't confirm {they} stopped.")
    elif stopped:
        say = f"Stopped {stopped} background task{'s' if stopped != 1 else ''}{run}."
        if missing:
            say += (" I couldn't confirm the other one stopped." if missing == 1
                    else f" I couldn't confirm the other {missing} stopped.")
        mouth.say(say)
    elif task is not None:
        mouth.say("Stopped.")
    else:
        mouth.say("Nothing is running.")


# PHASE J1: long-running background tasks (currently just the vault-recall
# keep-warm heartbeat) are held here so asyncio can't garbage-collect them
# mid-flight -- a bare `asyncio.create_task(...)` with no stored reference is
# only weakly held by the loop and can be silently dropped.
_bg_tasks: list[asyncio.Task] = []


# ---- PROACTIVE BEHAVIOR (roadmap Phase F). DEFAULT OFF. -----------------------------------------------------
# Only the JSON value `true` for CFG["proactive"]["enabled"] starts it; otherwise nothing below runs and nothing about the
# voice line changes. When on, ONE periodic task (a Runner, backtalk/proactive_loop.py) ticks the persistent scheduler,
# mirrors the ledger's deferred background outcomes, and speaks through mouth.say only when the engine says the room is
# clear. It never touches the addressee gate, never speaks model-authored text, and a failure to build it (a corrupt or
# locked scheduler file, a bad config) logs once and leaves the voice line exactly as it was.
_PROACTIVE = {"runner": None, "restarts": 0}
_PRESENCE = {"monitor": None}
PRESENCE_TRUST_S = 15.0          # a presence reading older than this (the thread died or stalled) is ignored, never trusted


def _presence_start(log_fn=None):
    """Desktop presence (roadmap Phase G): a daemon thread of cheap, permission-free, enum-only readings (idle time, frontmost app
    CATEGORY, lock/display state, audio output class, a fresh Bear Hands count). DEFAULT OFF: only the JSON value true for
    CFG["presence"]["enabled"] starts it. It changes nothing the model sees; the proactive layer reads it to avoid speaking
    over a call and to hold background news while nobody is at the machine. Never raises."""
    try:
        if (CFG.get("presence") or {}).get("enabled") is not True:
            return None
        mon = presence_mod.PresenceMonitor(on_change=lambda p: (log_fn or log)(p.line()))
        mon.refresh()
        (log_fn or log)("[presence] on " + mon.latest().line()[len("[presence] "):])
        mon.start()
        _PRESENCE["monitor"] = mon
        return mon
    except Exception as e:
        (log_fn or log)(f"[presence] could not start ({e.__class__.__name__})")
        return None


def _presence_camera(now=None):
    """True only when a FRESH presence reading says a camera is running; False otherwise (unknown is not evidence)."""
    try:
        mon = _PRESENCE["monitor"]
        r = None if mon is None else mon.latest()
        if r is None or not 0 <= (time.time() if now is None else now) - r.at <= PRESENCE_TRUST_S:
            return False
        return r.camera_in_use is True
    except Exception:
        return False


def _presence_fullscreen(now=None):
    """True only when a FRESH presence reading says the frontmost window is full-screen; unknown is not evidence."""
    try:
        mon = _PRESENCE["monitor"]
        r = None if mon is None else mon.latest()
        if r is None or not 0 <= (time.time() if now is None else now) - r.at <= PRESENCE_TRUST_S:
            return False
        return r.fullscreen is True
    except Exception:
        return False


def _presence_desktop(now=None):
    """The latest FRESH presence reading as one enum string ("state/app" plus "/call"), or None. Enums only."""
    try:
        mon = _PRESENCE["monitor"]
        r = None if mon is None else mon.latest()
        if r is None or not 0 <= (time.time() if now is None else now) - r.at <= PRESENCE_TRUST_S:
            return None
        return f"{r.state}/{r.app_category}" + ("/call" if r.likely_call else "") + ("/cam" if r.camera_in_use is True else "") + ("/full" if r.fullscreen is True else "")
    except Exception:
        return None


_GESTURES = {"watcher": None}


def _gestures_start(mouth, log_fn=None):
    """Both open hands held about a second silences the voice (roadmap Phase G). DEFAULT OFF: only the JSON value true for
    CFG["gestures"]["hush"] starts it. The action is `mouth.shut_up()` only, and only when the voice is actually busy. Never raises."""
    try:
        if (CFG.get("gestures") or {}).get("hush") is not True:
            return None
        say = log_fn or log

        def hush():
            if mouth.busy:
                mouth.shut_up()
                say("[gesture] both hands open held: voice stopped")
            else:
                say("[gesture] both hands open held: nothing was speaking")
        w = gestures_mod.GestureWatcher(hush, log=say)
        w.start()
        _GESTURES["watcher"] = w
        say("[gesture] hush on (both open hands held about a second)")
        return w
    except Exception as e:
        (log_fn or log)(f"[gesture] could not start ({e.__class__.__name__})")
        return None


def _presence_flags(now=None):
    """(on_call, user_away) from the latest fresh presence reading; (False, False) when there is none or it is old."""
    try:
        mon = _PRESENCE["monitor"]
        if mon is None:
            return False, False
        p = mon.latest()
        if (time.time() if now is None else now) - p.at > PRESENCE_TRUST_S:
            return False, False
        return bool(p.likely_call), bool(p.user_away)
    except Exception:
        return False, False


def _backchannel_on() -> bool:
    """Phase D: a spoken acknowledgement ("okay", "mm-hmm") heard while Jarvis talks or works is not a cut-in. DEFAULT OFF: only
    the JSON value true for CFG["conversation"]["backchannel"] turns it on. Never raises."""
    try:
        return (CFG.get("conversation") or {}).get("backchannel") is True
    except Exception:
        return False


_BC_LAST = {"text": None, "at": 0.0}
BACKCHANNEL_REPEAT_S = 8.0       # the same acknowledgement again this soon is the user MEANING it: let it through


def _is_backchannel_now(text, mouth, now=None) -> bool:
    """Is this utterance a spoken acknowledgement to ignore? Off unless enabled. Conservative on every doubt: with nothing recently
    spoken the voice may be mid-question, so an answer-type word is NOT swallowed; the same word twice within 8 s is not swallowed.
    Never raises."""
    try:
        if not _backchannel_on():
            return False
        chunks = mouth.recent_speech()
        q = turnclass.spoken_was_question(chunks) or not chunks
        if not turnclass.is_backchannel(text, last_spoken_was_question=q):
            return False
        now = time.monotonic() if now is None else now
        key = " ".join(turnclass.words(text))
        if _BC_LAST["text"] == key and now - _BC_LAST["at"] <= BACKCHANNEL_REPEAT_S:
            _BC_LAST["text"] = None                       # a repeat is deliberate: let it through as real input
            return False
        _BC_LAST["text"], _BC_LAST["at"] = key, now
        return True
    except Exception:
        return False


def _log_recovery(brain, exc) -> None:
    """Phase C: one audit line with the failure class and the recovery DECISION for a failed turn. Codes and ids only. It only
    decides; nothing here retries anything (a retry would send the turn twice, and speak_reply sends exactly once). Never raises."""
    try:
        tid = getattr(exc, "turn_id", None)
        if tid and tid in brain.ledger.turns:
            log("[recovery] " + taskstate.explain(taskstate.turn_view(brain.ledger, tid)))
        else:
            fc = taskstate.classify_turn_cause(exc.reason)
            rec = taskstate.recover(taskstate.FAILED, fc, side_effects=False if taskstate.nothing_ran(exc.reason) else None)
            log(f"[recovery] turn cause={str(exc.reason)[:40]} class={fc} recovery={rec.action}({rec.reason})")
    except Exception:
        pass


def _proactive_policy():
    try:
        return proactive_mod.Policy.from_config(CFG.get("proactive"))
    except Exception:
        return proactive_mod.Policy()


def _proactive_build(brain, mouth, context, *, policy=None, sched_dir=None):
    """The Runner, or None (disabled, or it could not be built). Never raises."""
    try:
        policy = policy or _proactive_policy()
        if not policy.enabled:
            return None
        d = sched_dir or os.path.join(str(ledger_mod.default_dir()), "scheduler")
        sched = scheduler_mod.Scheduler(d)
        sched.recover()
        runner = proactive_loop.Runner(
            engine=proactive_mod.Engine(policy), scheduler=sched, speech=brain.speech, ledger=brain.ledger,
            say=mouth.say, mouth_state=lambda: (bool(mouth.busy), mouth.generation, mouth.recent_speech(30.0)),
            context=context, log=log, sched_dir=d)
        log(f"[proactive] on urgent={policy.urgent} act_allow={len(policy.act_allow)} "
            f"quiet_hours={'set' if policy.quiet_hours else 'none'} scheduled={len(sched.tasks('scheduled'))}")
        return runner
    except Exception as e:
        log(f"[proactive] could not start ({e.__class__.__name__}); the voice line runs without it")
        return None


def _proactive_spawn(runner, holder=None):
    """Run the loop as a held task, restarting it ONCE if it ever dies. Returns the task."""
    holder = _bg_tasks if holder is None else holder

    async def _run():
        await runner.run(asyncio.sleep)

    task = asyncio.create_task(_run())
    holder.append(task)

    def _done(t):
        if t.cancelled():
            return
        err = t.exception()
        log(f"[proactive] loop stopped ({err.__class__.__name__ if err else 'returned'})")
        if _PROACTIVE["restarts"] < 1:
            _PROACTIVE["restarts"] += 1
            try:
                _proactive_spawn(runner, holder)
            except Exception:
                pass

    task.add_done_callback(_done)
    return task


def _proactive_start(brain, mouth, context):
    runner = _proactive_build(brain, mouth, context)
    if runner is None:
        return None
    _PROACTIVE["runner"] = runner
    _proactive_spawn(runner)
    return runner


async def amain():
    open_mic = "--open-mic" in sys.argv
    barge_in = "--barge-in" in sys.argv
    model = None
    if "--model" in sys.argv:
        try:
            model = sys.argv[sys.argv.index("--model") + 1]
        except IndexError:
            pass

    _MIC["mode"] = "open" if (open_mic
                              or CFG.get("mic_mode") == "open") else "ptt"
    # resume_last_session: reattach to the saved conversation, if any --
    # but only while it is still small enough to be worth carrying. A
    # session left to grow re-sends its whole context on every turn, and
    # re-caches it at 1.25x after any idle gap; that is where Yousif's
    # usage was actually going (see sessions.py for the measurements).
    resume_id = None
    if CFG.get("resume_last_session"):
        try:
            from backtalk.brain import SESSION_FILE
            with open(SESSION_FILE) as f:
                resume_id = f.read().strip() or None
        except OSError:
            resume_id = None
        resume_id = sessions.resumable(resume_id)

    # Restarted in place by the audio watchdog? Pick the same conversation
    # back up, whatever resume_last_session says.
    recovered = audiowatch.recovered()
    resume_id = audiowatch.resume_id() or resume_id

    mouth = Mouth()
    # Load the voice FIRST, in the background, before the other engines (speech-to-text, vault embeddings, the brain connect) start
    # competing for the CPU. It used to load only when the greeting was queued, behind all of them: ~80 s to the first spoken word
    # on a restart (measured 2026-10-19), against ~21 s when it has the machine to itself. A failure here is not fatal: the first
    # say() calls warm() itself and raises exactly as before.
    def _voice_preload():
        try:
            warm_mouth()
        except Exception as e:
            log(f"[mouth] preload failed ({e.__class__.__name__}); it will retry on first use")
    threading.Thread(target=_voice_preload, name="voice-preload", daemon=True).start()
    threading.Thread(target=turnend.warm, name="turnend-preload", daemon=True).start()
    ears = Ears(silence_ms=int(CFG.get("silence_ms", 1000)))
    # Echo cancellation (aec.py): decided once, so ears and mouth agree. On,
    # the mic can't hear the voice, so nothing heard is echo and the person
    # cuts in the moment they start talking.
    aec_on = aec.enabled()
    if systap.enabled():
        systap.start()           # in the background; never holds up launch
    # ADDRESSEE GATE (see addressee.py).
    # --no-wake is for development and testing only.
    global _ADDRESSEE
    _ADDRESSEE = AddresseeGate.from_config(CFG, NAME,
                                           always="--no-wake" in sys.argv)
    # The action ledger (ledger.py). If its directory cannot be used the voice
    # line must still come up, so fall back to a throwaway one and say so.
    try:
        action_ledger = ledger_mod.Ledger()
    except Exception as e:
        import tempfile
        log(f"[ledger] could not open the ledger ({e!r}); using a temporary one")
        action_ledger = ledger_mod.Ledger(tempfile.mkdtemp(prefix="backtalk-ledger-"))
    log(f"[gate] addressee mode={_ADDRESSEE.mode} "
        f"window={_ADDRESSEE.window:.0f}s max={_ADDRESSEE.max_active:.0f}s "
        f"wake={_ADDRESSEE.wake}")
    brain = WarmBrain(model=model,
                      resume_id=resume_id,
                      ledger=action_ledger)
    brain.on_say = mouth.say
    # A hung audio call restarts the process instead of leaving it deaf
    # (see audiowatch). The music is restored first; the talk resumes.
    audiowatch.set_session_source(lambda: brain.session_id)
    audiowatch.before_restart(mouth.ducker.restore_now)
    audiowatch.on_device_failure(lambda: mouth.say(
        "My microphone stopped working, so I can't hear you. Restart the "
        "Mac's audio, then open me again."))
    audiowatch.start()

    mode = ("hands-free listening (the talk key still works)"
            if _MIC["mode"] == "open"
            else f"push-to-talk ({CFG['ptt_key']})")
    log(f"[backtalk] code revision {code_revision()}")
    log(f"[backtalk] up — agent={NAME} dir={CFG['agent_dir']} "
        f"model={brain.model} mic={mode} "
        f"(say 'goodbye {NAME.lower()}' to hang up)")
    if recovered:
        log("[watchdog] back up after an audio lockup")
        mouth.say("My audio locked up, so I restarted myself. I'm back.")
    else:
        mouth.say(CFG["greeting"])

    loop = asyncio.get_event_loop()
    # Warm the engines while the greeting plays: the STT model load and
    # the brain's prompt-cache toll both hide behind the spoken line.
    loop.run_in_executor(None, warm_ears)
    # Same trick for vault recall: load the embedding model now, behind the
    # greeting, so the first real question isn't the one that pays for it.
    loop.run_in_executor(None, vaultrecall.warm)
    # PHASE J1: keep it warm for the rest of the session too -- one warm()
    # at startup only covers the FIRST 30 minutes; a long quiet stretch
    # after that lets Ollama's keep_alive lapse, and a cold model can't
    # recover through ordinary conversation (see vaultrecall.keep_warm_loop's
    # docstring). Stored on _bg_tasks so the task object isn't GC'd out from
    # under the running loop -- it runs for the life of the process either way.
    _bg_tasks.append(asyncio.create_task(vaultrecall.keep_warm_loop()))
    # THE BRAIN CONNECT, guarded. This is the one startup step that
    # needs a signed-in Claude Code, internet, and available usage.
    # When it fails or hangs, the mouth still works, so SAY SO instead
    # of dying silently with the face stuck on idle (a real field
    # case: the greeting played, then nothing, and on Windows the
    # window closed before anyone could read the error).
    log("[backtalk] connecting the brain...")
    try:
        await asyncio.wait_for(brain.start(), 120)

        async def _warmup():
            async for _ in brain.ask_stream(
                    "Warmup ping - reply with the single word: ready"):
                pass
        await asyncio.wait_for(_warmup(), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        # Tell a signed-out login from a transient failure by asking the CLI
        # locally (no model call). A signed-out login speaks ONE sign-in
        # message and exits 78 so a relauncher does not loop on it; every other
        # failure keeps the original message and exit code 1. No retries.
        _marker = authguard.default_marker_path()
        _was_marked = authguard.marker_present(_marker)
        _status = await asyncio.get_running_loop().run_in_executor(
            None, authguard.signed_in_status)
        _decision = authguard.decide_connect_failure(_status, _was_marked)
        if _decision.write_marker:
            authguard.write_marker(_marker)
        log(f"[backtalk] brain connect failure classified: {_decision.kind} "
            f"(sign-in status {_status})")
        if _decision.kind == "auth":
            print(authguard.SIGNIN_PRINTED, flush=True)
        if _decision.speak:
            mouth.say(_decision.speak)
            mouth.wait_done(timeout=30)
        raise SystemExit(_decision.exit_code)
    authguard.clear_marker(authguard.default_marker_path())
    log("[backtalk] brain warm")
    # the hidden warmup ping is plumbing, not conversation
    brain.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
    # a configured effort level applies at launch (saved by the spoken
    # "set effort to X", or written by the person's agent on request)
    boot_effort = str(CFG.get("effort") or "").strip().lower()
    if boot_effort in _EFFORTS:
        await brain.command(f"/effort {boot_effort}")
        log(f"[backtalk] effort set to {boot_effort} (from config)")
    elif boot_effort:
        log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")

    speak_task: asyncio.Task | None = None

    def _speech_context():
        """What the speech policy needs to know about the room right now."""
        conv = True
        if _MIC["mode"] == "open" and _ADDRESSEE is not None:
            conv = _ADDRESSEE.state == "ACTIVE"
        return SP.Context(
            fg_active=bool(speak_task and not speak_task.done()),
            mouth_busy=bool(mouth.busy), conversation_active=conv,
            zipped=bool(_ZIPPED["on"]))
    brain.get_speech_context = _speech_context
    # The situation snapshot: a reader over the owners above. It only holds references and
    # changes nothing it reads. Phase 2: each snapshot logs one passive [sit] line (names and
    # counts only) through `observer`; the model is never shown it.
    global _SITUATION, _SITCTX_READER
    _sit_sources = situation_mod.SituationSources(
        mic=lambda: _MIC, zipped=lambda: _ZIPPED["on"], mouth=mouth,
        addressee=_ADDRESSEE,
        brain=brain, desktop=_presence_desktop, observer=log, on_failure=log)
    _SITUATION = situation_mod.SituationReader(_sit_sources)
    # Phase 3B: a second reader over the SAME owners with no observer, so a shadow snapshot never
    # produces a second [sit] telemetry line.
    _SITCTX_READER = situation_mod.SituationReader(
        dataclasses.replace(_sit_sources, observer=None, on_failure=None))
    def _proactive_context():
        """What the proactive engine needs to know about the room. Reads only."""
        conv = True
        if _MIC["mode"] == "open" and _ADDRESSEE is not None:
            conv = _ADDRESSEE.state == "ACTIVE"
        last = _LAST_TURN["at"]
        return proactive_mod.Context(
            user_speaking=bool(_MIC["btn"]) or bool(getattr(ears, "hearing", False)),
            since_user_speech_s=(time.monotonic() - last) if last else None,
            mouth_busy=bool(mouth.busy), fg_active=bool(speak_task and not speak_task.done()),
            zipped=bool(_ZIPPED["on"]), conversation_active=conv,
            on_call=_presence_flags()[0], user_away=_presence_flags()[1], camera_busy=_presence_camera(), screen_busy=_presence_fullscreen(),
            brain_ready=getattr(brain, "router", None) is not None and not getattr(brain, "_need_rebuild", False))
    _presence_start()
    _gestures_start(mouth)
    _proactive_start(brain, mouth, _proactive_context)
    typed_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_typed_reader, args=(typed_q,), daemon=True).start()
    typed_fut: asyncio.Future | None = None

    async def run_console(verb):
        """One voice-console verb. The current reply was already
        cancelled and awaited by handle(); the pipe gets drained here
        before the command goes out. A verb that blows up must never
        take the whole voice session down with it."""
        try:
            await _run_console_inner(verb)
        except Exception as e:
            log(f"[console] {verb} failed: {e}")
            mouth.say("That command hit an error. Check the log.")
            signals.set_state("idle")

    async def _run_console_inner(verb):
        _HOLD.clear()                # a console command replaces whatever was being held
        await brain.reset_turn()
        before = command_snapshot(brain)     # to judge, after the command, whether it really succeeded
        say_after = None
        if verb == "clear":
            _sitctx_invalidate()     # before the command: a /clear that fails still invalidates
            resp = await brain.command("/clear")
            say_after = "Cleared. Fresh slate."
        elif verb == "compact":
            mouth.say("Compacting. One moment.")
            _sitctx_invalidate()     # before the command: a /compact that fails still invalidates
            resp = await brain.command("/compact")
            say_after = "Compacted. Same conversation, smaller footprint."
        elif verb == "deep":
            mouth.say("Switching to the deep model. Heads up, replies "
                      "get slower. Say back to the fast model when "
                      "you're done.")
            resp = await brain.command(f"/model {CFG['deep_model']}")
            say_after = "Deep model online, for this session only."
        elif verb == "fast":
            resp = await brain.command(f"/model {CFG['model']}")
            say_after = "Back on the fast model."
        elif verb.startswith("effort:"):
            lvl = verb.split(":", 1)[1]
            resp = await brain.command(f"/effort {lvl}")
            saved = (command_succeeded(brain, before, verb, resp)
                     and _write_config_key("effort", lvl))
            say_after = (f"Effort set to {lvl}, and saved as your "
                         "default." if saved else
                         f"Effort set to {lvl} for this session. The "
                         "config file couldn't be written, so it won't "
                         "stick past a restart.")
        elif verb == "usage":
            resp = ""
            mouth.say(_spoken_usage(brain.session,
                                    await brain.context_usage()))
        elif verb == "micopen":
            resp = ""
            if _MIC["mode"] == "open":
                mouth.say("Already in hands-free listening.")
            else:
                _MIC["mode"] = "open"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "open")
                log("[console] mic_mode -> open (hands-free listening)")
                # Was a 38-word tutorial, spoken in full EVERY time.
                # Yousif, 2026-09-22: "he tell me the exact same thing
                # every single time and I don't need that he's
                # literally acting like a robot." He knows what it
                # means; the console facts are in the prompt if he asks.
                mouth.say("Hands-free.")
        elif verb == "micptt":
            resp = ""
            if _MIC["mode"] == "ptt":
                mouth.say("Already on push to talk.")
            else:
                _MIC["mode"] = "ptt"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "ptt")
                log("[console] mic_mode -> ptt")
                mouth.say("Push to talk.")
        elif verb == "stopbg":
            resp = ""
            await stop_background_intent()
        elif verb == "stopall":
            resp = ""
            await stop_everything_intent()
        elif verb == "zip":
            resp = ""
            if _ZIPPED["on"]:
                mouth.say("Already zipped.")
            else:
                _ZIPPED["on"] = True
                log("[console] zipped -- withholding unprompted replies")
                mouth.say("Zipped, sir.")
        elif verb == "unzip":
            resp = ""
            if not _ZIPPED["on"]:
                mouth.say("Wasn't zipped, sir.")
            else:
                _ZIPPED["on"] = False
                log("[console] unzipped")
                mouth.say("Unzipped, sir. Back in the mix.")
        else:
            resp = ""
        if say_after:
            # the CLI answers slash commands with its own text (confirmations, API errors); an error
            # outranks our line, and our line is only for a command that demonstrably succeeded
            line = command_confirmation(brain, before, verb, resp, say_after)
            mouth.say(line)
            if line != say_after:
                log(f"[console] {verb} not confirmed: {(resp or '')[:120]!r}")
        signals.set_state("idle")

    async def stop_background_intent():
        await do_stop_background(brain, mouth)

    async def stop_everything_intent():
        nonlocal speak_task
        was_running = bool(speak_task and not speak_task.done())
        await do_stop_everything(brain, mouth, speak_task if was_running else None)
        if was_running:
            speak_task = None

    progress = TurnProgress()

    async def handle(text: str, spoke_from: float | None = None,
                     released: bool = False, backchannel_ok: bool = False,
                     turn_p: float | None = None) -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed.
        `released` is the held stitched request coming due (see StitchHold):
        text that was already heard and logged, sent now as one turn."""
        nonlocal speak_task
        cancel_t0 = 0.0               # when the turn in flight was cancelled, if it was
        hold = False                  # this fragment joined a held stitched request
        if not released:
            log(f"[you]    {text}")
        if not released and brain.guard.pending and brain.guard.answer(text, spoke_from):
            log(f"[guard] spoken answer to the safety question ({brain.guard.last_answered}): {text}")
            return True
        if (backchannel_ok and not released and not _HOLD.active
                and (mouth.held or mouth.busy or (speak_task and not speak_task.done()))
                and _is_backchannel_now(text, mouth)):
            # Phase D: an acknowledgement HEARD ON THE OPEN MIC is not a request and not a cut-in. Typed input and the talk key are
            # explicit address and never come here; neither is a fragment of a held stitched request. Neither the voice nor the
            # turn nor any work behind it is cancelled; a paused voice simply carries on.
            log(f"[turn] backchannel swallowed, not a cut-in -- carrying on: {text}")
            if mouth.held:
                mouth.release()
            return True
        if mouth.held:
            # words, not noise: the paused reply is over. Flush it BEFORE
            # releasing, or the release would play the rest of it.
            mouth.shut_up()
            mouth.release(resume=False)
        verb = None
        if any(q in text.lower() for q in QUIT_PHRASES):
            if speak_task and not speak_task.done():
                speak_task.cancel()
            mouth.shut_up()
            mouth.say(CFG["signoff"])
            mouth.wait_done(timeout=15)
            mark_intentional_quit()
            return False
        if _ZIPPED["on"] and verb is None:
            # Zipped: still hearing everything, but nothing unprompted
            # gets a reply. "unzip" is the one phrase that still lands.
            verb = console_match(text)
            # Zipped means zipped. There used to be an exception here that
            # answered anyway whenever the name appeared in the sentence
            # (added 2026-09-16, reasoning: staying silent hides that it was
            # zipped). In practice that is the opposite of what "zip it"
            # means -- Yousif zips it to talk to someone ELSE in the room,
            # his name lands in the mic in passing, and the voice pipes up.
            # 2026-09-22: "he still speak when I bring his name." Only
            # "unzip" gets through now.
            #
            # CONSOLE COMMANDS STILL RUN THOUGH. Zip suppresses REPLIES, not
            # the control surface -- the first cut of this swallowed
            # everything but "unzip", so "push to talk" while zipped was
            # recognised and then thrown away (live at 19:29:05: heard
            # 'Wish to talk.' as micptt, then swallowed), and he had to
            # unzip before he could touch his own mic settings.
            if verb is None:
                log(f"[zip] swallowed: {text}")
                return True
        early = verb or console_match(text)
        if early == "stopbg":
            log(f"[bg] stop the background task: {text!r}")
            await stop_background_intent()
            return True
        if early == "stopall":
            log(f"[bg] stop everything: {text!r}")
            await stop_everything_intent()
            return True
        if (speak_task and not speak_task.done()) or (_HOLD.active and not released):
            # A CHECK-IN, into dead air, while the turn is still working:
            # answer it out loud and let the turn finish. This is the single
            # biggest cause of "Jarvis never answered me" — 26 of the 66
            # dead utterances between 2026-09-19 and 21 were Yousif asking
            # "did you find it?", which cancelled the very turn that was
            # about to tell him. Costs no tokens: the status line is built
            # here, not asked of the model.
            #
            # Gated on the mouth being SILENT, so the same words spoken over
            # a reply in progress are still a real barge-in and still cancel.
            # While a stitched request is only HELD there is no turn working,
            # so there is nothing to check in on: elapsed 0 keeps a "did you
            # find it?" from being answered as a status line.
            # A pre-send hold (an unfinished sentence, nothing sent yet) is the
            # request this fragment continues, whatever the last turn was.
            presend = _HOLD.active and _HOLD.unfinished and not (speak_task and not speak_task.done())
            action = decide_during_turn(
                text, mouth_busy=bool(mouth.busy),
                elapsed_s=(progress.elapsed if speak_task and not speak_task.done() else 0.0),
                last_turn_text=_HOLD.text if presend else _LAST_TURN["text"],
                last_turn_age_s=0.0 if presend else time.monotonic() - _LAST_TURN["at"])
            if action == "stop":
                log(f"[turn] stopped by voice while working: {text}")
                _HOLD.clear()          # a held request is stopped with the turn
                await stop_current_turn(brain, mouth, speak_task,
                                        voice_was_busy=bool(mouth.busy))
                speak_task = None
                return True
            if action == "checkin":
                line = progress.status(progress.checkins + progress.narrated)
                progress.checkins += 1
                log(f"[turn] check-in kept the turn alive: {text!r}")
                mouth.say(line)
                return True
            # STITCH: he is still adding to the same request. The mouth has
            # not said a word yet, so nothing he asked for has been answered
            # -- this is the back half of a sentence, not an interruption.
            #
            # Without this, "do A" starts a turn, "and B" CANCELS it and
            # starts over with only "B", so A is silently dropped and B
            # arrives late. That is exactly what Yousif reported on
            # 2026-09-22: "when I ask him for multiple stuff at the same
            # time he paused for a second and then answer me later." His
            # speech genuinely arrives in pieces -- one complaint that night
            # landed as eight separate utterances, split mid-phrase.
            #
            # Originally gated on the mouth being SILENT: once a reply was
            # audible, new speech was always treated as a real barge-in.
            # 2026-09-23: that gate now ALSO allows mouth.busy, because
            # closing the (now much faster) false endpoint plus this
            # session's other latency work means Jarvis often starts
            # speaking a premature reply before he finishes a naturally
            # paused sentence -- the mouth being audible no longer proves
            # the fragment isn't still the same thought. A real stop
            # command must still stop it rather than get folded into the
            # resent question, so once mouth IS busy, the same stop-phrase
            # check over_speech() uses gates whether this counts as a
            # continuation or falls through to the interrupt path below.
            #
            # 2026-10-05: cancel-and-resend per fragment is what turned a burst
            # of fragments into several back-to-back queries, so the turn is
            # retracted ONCE (here, the same cancel as before) and the request
            # is HELD, not resent, until the fragments stop (StitchHold). The
            # first fragment starts the hold from the retracted turn's text;
            # each later one is appended in order and extends it.
            if action == "stitch":
                if _HOLD.active:
                    text = _HOLD.add(text, unfinished_wait(text, turn_p) if presend else None)
                else:
                    text = _HOLD.begin(f"{_LAST_TURN['text']} {text}".strip(),
                                       _brain_epoch(brain))
                hold = True
                log(f"[turn] stitched onto the unanswered request: {text!r}")
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    cancel_t0 = time.monotonic()
                mouth.shut_up()
            else:
                log("[turn] interrupted mid-reply by new input")
                _HOLD.clear()        # a held request is replaced like a turn is
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    cancel_t0 = time.monotonic()
                mouth.shut_up()
        if speak_task:
            # Let the cancellation fully land (its brain.interrupt()
            # included) BEFORE anything else touches the brain —
            # otherwise the dead turn's stop signal can race in after
            # the new query and kill the new answer (half of the
            # off-by-one bug; see brain.reset_turn for the other half).
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            speak_task = None
            if cancel_t0:
                log(f"[timing] cancel: the previous turn was gone "
                    f"{time.monotonic() - cancel_t0:.2f}s after it was replaced")
        if hold:
            # Nothing is sent yet: the main loop sends ONE replacement when
            # the quiet time on the held request runs out (StitchHold).
            log(f"[turn] holding {_HOLD.n} fragment(s) for the replacement "
                f"({STITCH_HOLD_S:.1f}s of quiet, at most {STITCH_HOLD_MAX_S:.1f}s)")
            signals.set_state("thinking")
            return True
        verb = verb or console_match(text)
        if verb:
            await run_console(verb)
            return True
        wait = unfinished_wait(text, turn_p) if not released and _MIC["mode"] == "open" else 0.0
        if turn_p is not None and not released:
            log(f"[turnend] p_complete={turn_p:.2f} wait={wait:.1f}s")    # for measuring it on real speech
        if wait:
            # He paused mid-sentence: wait for the rest instead of starting a
            # turn the next fragment would kill (unfinished_wait).
            _HOLD.begin(text, _brain_epoch(brain), quiet_s=wait,
                        max_s=UNFINISHED_HOLD_MAX_S, unfinished=True)
            log(f"[turn] sounds unfinished, waiting for the rest "
                f"({wait:.1f}s of quiet, at most {UNFINISHED_HOLD_MAX_S:.1f}s): {text!r}")
            return True
        signals.set_state("thinking")
        signals.static_start()
        # Clean the pipe: drain the interrupted turn's leftovers so the
        # new question can't pair with a stale ResultMessage.
        await brain.reset_turn()
        fixed = _fix_name(text)
        if fixed != text:
            log(f"[turn] name fixed: {fixed}")
            text = fixed
        # Vault recall and the board-changes check are independent (one
        # searches the vault by similarity, the other polls the physical
        # board), so they run concurrently instead of back-to-back. Kicked
        # off on the name-fixed text -- NOT the eventual board-changes-
        # prefixed one below, so a garbled name is never what gets searched
        # on, and searching drifts none from what speak_reply always used.
        vault_fut = loop.run_in_executor(None, vaultrecall.context_for, text)
        seen = await asyncio.to_thread(board.changes)
        if seen:
            log(f"[board] {seen}")
            text = f"{seen} {text}"
        # Push-to-talk means he pressed the key and spoke to Jarvis, so an
        # empty turn is a failure. On the open mic it is only "meant for me"
        # when his name is in it — otherwise silence may be the right answer.
        addressed = (_MIC["mode"] != "open") or _names_agent(text)
        # Background outcomes that were not announced ride in as a short,
        # ledger-built note (same pattern as the holo-board note), so the
        # model can mention them when they matter. Computed AFTER `addressed`
        # so a task name can never make an ambient line look addressed.
        bg_note = brain.note_prefix()
        if bg_note:
            log(f"[bg] note for this turn: {bg_note[:160]}")
            text = f"{bg_note} {text}"
        progress.start()
        _LAST_TURN["text"] = text
        _LAST_TURN["at"] = time.monotonic()
        speak_task = asyncio.create_task(
            speak_reply(brain, mouth, text, progress, addressed, vault_fut,
                       run_console))
        _observe_situation()      # passive [sit] line; nothing is sent to the model
        return True

    def over_speech(text: str) -> bool:
        """Something was heard while the voice was talking: the person
        talking over it, or the voice itself coming back through the
        speakers. Its own recent words are dropped as echo. A bare stop
        silences it and ends the turn, like the talk key. Anything else is
        the person cutting in ("okay, I get it, do it"): the voice stops
        and the words go through as the next thing said (True)."""
        if any(q in text.lower() for q in QUIT_PHRASES):
            return True
        # With echo cancellation the mic can't hear the voice, so nothing
        # heard over it is echo: no transcript comparison at all.
        name_stop = (not aec_on
                     and _is_name_over_echo(text, mouth.recent_speech()))
        if (not aec_on and not name_stop
                and _is_echo(text, mouth.recent_speech())):
            log(f"[ears] ignored echo: {text}")
            return False
        if name_stop or _is_stop_phrase(text):
            if speak_task and not speak_task.done():
                speak_task.cancel()
            log(f"[turn] stopped by voice: {text}")
            mouth.shut_up()
            mouth.release(resume=False)
            signals.static_stop()
            signals.set_state("listening")
            return False
        if _is_backchannel_now(text, mouth):
            log(f"[turn] backchannel over the voice -- carrying on: {text}")
            mouth.release()
            return False
        log(f"[turn] cut in over the voice: {text}")
        mouth.shut_up()
        return True

    def on_barge(start: bool):
        """Echo cancellation's instant pause, called from the mic thread:
        speech started over the voice (True: pause it in place), or what
        paused it was thrown away before it became a transcript (False:
        carry on)."""
        if start:
            log("[turn] you started talking -- voice paused")
            mouth.hold()
        else:
            mouth.release()

    def barge_ok() -> bool:
        """Whether cutting in should pause the voice on the spot.

        Always safe with echo cancellation on: the mic feed has the voice
        taken out of it. On headphones it is off, and the 2026-09-21 version
        of this said headphones were safe anyway because "nothing it says
        reaches the mic" -- wrong. AirPods Pro leak into the MacBook Air mic
        at -15 to -30 dBFS, so Jarvis cut HIMSELF off six times in under two
        minutes on 2026-09-22 and PAUSE_MAX_S dropped every one of those
        replies. So headphones now need the Mac-audio tap running to police
        it (ears._own_voice vetoes a cut-in that matches what the Mac just
        played); with no tap, headphones go back to no instant pause rather
        than back to that bug. Checked per listen rather than once at launch,
        so connecting AirPods mid-session gets whichever answer is true now.
        """
        return aec.enabled() or (not aec.on_speakers() and systap.enabled())

    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        ptt = PTTListener(CFG["ptt_key"])
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns). It keeps
        # listening while the mouth speaks, but without barge-in only a
        # stop phrase heard over the voice gets through (over_speech).
        mic_gate = lambda: _MIC["btn"]
        mic_fails = 0
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch
                if press_fut is not None and press_fut.done():
                    press_fut.result(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.result(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(None, typed_q.get)
            if press_fut is None:
                press_fut = loop.run_in_executor(None, ptt.wait_press)
            waiters = {press_fut, typed_fut}
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    mic_fut = loop.run_in_executor(
                        None, lambda g=g: (g, ears.listen_once(
                            gate=mic_gate,
                            speaking=lambda: mouth.speaking,
                            barge=on_barge if barge_ok() else None,
                            audible=lambda: mouth.audible,
                            abort=lambda: _MIC["gen"] != g),
                            ears.heard_while_speaking))
                waiters.add(mic_fut)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED,
                timeout=_HOLD.remaining())
            if not done:
                # The quiet time on a held stitched request ran out (or this
                # was a spurious wake): send ONE replacement, built from every
                # fragment in order, unless the session it was said in is gone.
                if not await release_held(_HOLD, brain, handle):
                    return
                continue
            if typed_fut in done:
                text = typed_fut.result(); typed_fut = None
                if text and _ADDRESSEE is not None:
                    _ADDRESSEE.open("typed")      # keyboard = explicit address
                if text and not await handle(text):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, text, over = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if not text and mouth.held:
                    # it cut the voice, then came to nothing (noise): let
                    # the rest of the reply carry on
                    log("[turn] that was noise -- carrying on")
                    mouth.release()
                if text:
                    kept = _without_mac_audio(text, ears.last_seconds,
                                              ears.last_end)
                    if not kept:
                        log(f"[ears] ignored Mac audio: {text}")
                        if mouth.held:
                            mouth.release()      # the voice carries on
                        continue
                    if kept != text:
                        log(f"[ears] kept your words around Mac audio: {kept}")
                        text = kept
                if text and _ADDRESSEE is not None:
                    # Speech is NOT admitted merely because the voice is
                    # talking. What is always allowed: zip/unzip, and a bare
                    # stop phrase while the voice is speaking (silencing it is
                    # harmless). Anything else needs ACTIVE or a wake phrase.
                    voice_on = mouth.busy or bool(speak_task
                                                  and not speak_task.done())
                    d = _ADDRESSEE.check(
                        text,
                        exempt=(console_match(text) in ("zip", "unzip")
                                or (voice_on and _is_stop_phrase(text))))
                    if d.reason == "end_conversation":
                        # "that's all" / "never mind": back to IDLE, silence
                        # the voice, and do NOT send it to the brain.
                        if speak_task and not speak_task.done():
                            speak_task.cancel()
                        _HOLD.clear()        # a held stitched request ends with the conversation
                        mouth.shut_up()
                        mouth.release(resume=False)
                        signals.static_stop()
                        signals.set_state("idle")
                        continue
                    if not d.accepted:
                        if mouth.held:
                            mouth.release()      # the voice carries on
                        continue
                if text and over and not barge_in and not over_speech(text):
                    continue
                # Echo can arrive after playback has already stopped (room
                # tail/reverb lags mouth.speaking flipping False), so it
                # never went "over" anything and skipped over_speech
                # entirely. Catch it here too, whether or not it overlapped.
                if (text and not over and not aec_on
                        and _is_echo_after(text, mouth.recent_speech())):
                    log(f"[ears] ignored echo (post-speech): {text}")
                    continue
                if text and not await handle(text, backchannel_ok=True,      # the open mic only (Phase D)
                                             turn_p=ears.last_turn_p):
                    return
                continue
            if press_fut in done:
                press_fut.result(); press_fut = None
                press_t = time.monotonic()
                if press_cancels_turn(speak_task, brain.guard):
                    log("[turn] interrupted mid-reply — key pressed")
                    speak_task.cancel()          # the button = interrupt
                elif brain.guard.pending:
                    log(f"[guard] key pressed while the safety question is open: capturing the answer, question kept ({brain.guard.pending_call})")
                mouth.shut_up()
                signals.static_stop()            # button kills the static too
                signals.set_state("listening")
                mouth.ducker.speech_start()      # duck NOW, while you talk
                print("[ptt] recording (release to send)...", flush=True)
                _MIC["btn"] = True               # open mic yields to the button
                try:
                    text = await loop.run_in_executor(
                        None, lambda: record_held(ptt.is_held))
                except Exception as e:
                    # A device-level failure gets plain words instead of a
                    # raw exception. The pre-flight at startup cannot catch
                    # a microphone unplugged mid-session, and that is the
                    # case where the old message was worst: jargon, on
                    # every press, with the key hook still working so it
                    # looked like it was listening.
                    if explain_audio_failure(e):
                        mouth.say("I can't hear you. There's no working "
                                  "microphone I can use.")
                    else:
                        log(f"[ears] record/transcribe failed: {e!r}")
                        mouth.say("My ears hit an error. Check this "
                                  "window for the details.")
                    text = None
                finally:
                    _MIC["btn"] = False
                mouth.ducker.speech_end(0.2)     # snap back fast on release
                if not text:
                    log("[ptt] (tap or empty — ignored)")
                    if brain.guard.pending:      # no usable answer (tap, empty, or an ears error) to an open safety question = no
                        log(f"[guard] no usable answer on the talk key: denying ({brain.guard.pending_call})")
                        brain.guard.deny_pending()
                    signals.set_state("idle")
                    continue
                if _ADDRESSEE is not None:
                    _ADDRESSEE.open("held_key")   # held key = explicit address
                if not await handle(text, spoke_from=press_t):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        # Capture this before any await below: None for returns and a caught
        # Ctrl-C, CancelledError for the asyncio Ctrl-C / closed-window path,
        # the exception itself for a real crash (verified on Python 3.12).
        _exc_at_exit = sys.exc_info()[0]
        _MIC["gen"] += 1     # abort any live open-mic capture promptly
        if speak_task and not speak_task.done():
            speak_task.cancel()
        mouth.shutdown()  # restores the music on Ctrl-C / crash paths too
        signals.static_stop()
        signals.set_state("idle")
        await brain.stop()
        log("[backtalk] hung up")
        if _exc_at_exit in (None, asyncio.CancelledError):
            _exit_now()      # clean shutdowns only; a crash keeps its traceback


# Loopback port used purely as a mutex. Nothing is ever served on it.
_INSTANCE_PORT = 8791
_instance_lock = None


def _claim_single_instance() -> bool:
    """Refuse to be the second voice line on this machine, out loud.

    Two instances both hold the keyboard hook and both open the
    microphone, and the result looks EXACTLY like a broken talk key:
    presses register, the audio goes to whichever process won the
    device, and the loser reports an ignored tap. Nothing warned about
    it, so a user who double-clicks the Talk icon twice concludes the
    product is broken. The tell, when it was finally caught, was the
    same sentence transcribed twice at an identical timestamp.

    A bound socket is the mutex rather than a pid file, because the
    operating system releases it when this process dies HOWEVER it dies.
    A pid file outlives a crash or a force-kill and then lies about a
    process that is long gone, which is the failure it would exist to
    prevent.
    """
    global _instance_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # No SO_REUSEADDR here on purpose: reuse is exactly what would let a
    # second instance bind alongside the first and defeat the whole point.
    try:
        s.bind(("127.0.0.1", _INSTANCE_PORT))
        s.listen(1)
    except OSError:
        s.close()
        return False
    _instance_lock = s
    return True


def _exit_now():
    """Leave the process now, after a clean shutdown.

    backtalk waits for typed input and the talk key on Python's default
    worker pool (run_in_executor). Those calls block forever, and at
    interpreter shutdown Python waits for that pool, so after "hung up"
    the process never exited on its own (seen 2026-09-12: still alive
    26 s later, which made every relaunch wait for a force-kill). Run the
    registered exit hooks first (they restore the terminal), flush, then
    exit without waiting on the pool."""
    import atexit
    import os
    try:
        import sounddevice as sd
        # PortAudio's exit handler stops every open stream, and the mic now
        # stays open for the life of the process. That stop is the call that
        # hangs on this machine; the OS releases the devices when we exit.
        atexit.unregister(sd._exit_handler)
    except Exception:
        pass
    try:
        atexit._run_exitfuncs()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(0)


HANGUP_BACKSTOP_S = 5.0   # a closed window exits within this long, however shutdown goes


def _install_hangup_logger():
    """Closing the Terminal window (or clicking Terminate) sends SIGHUP.
    Python's default is to die on the spot: no log line, no cleanup,
    music left ducked. Seen live 2026-09-12, twice, and it took the
    macOS system log to find out why the voice line had vanished. Log
    it, then take the exact Ctrl-C path so shutdown stays clean."""
    import os
    import signal as _signal

    fired = []

    def _on_hangup(signum, frame):
        # Terminal sends SIGHUP to uv and to Python, so it can arrive several
        # times (three at once at 10:27 PM on 2026-09-13). A second SIGINT
        # landing inside the shutdown interrupted it and left the process
        # stuck in Python's exit. Only the first one counts.
        if fired:
            return
        fired.append(True)
        try:
            log("[backtalk] Terminal window closed (SIGHUP) — hanging up")
        except Exception:
            pass
        # Backstop: the window is gone, so leave even if the clean shutdown
        # stalls. asyncio.run waits on the blocked worker threads before it
        # re-raises the interrupt, which can take minutes.
        backstop = threading.Timer(HANGUP_BACKSTOP_S, _exit_now)
        backstop.daemon = True
        backstop.start()
        os.kill(os.getpid(), _signal.SIGINT)

    _signal.signal(_signal.SIGHUP, _on_hangup)


QUIT_MARKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".voice_quit")


def mark_intentional_quit(path=None) -> None:
    """A spoken hang-up ("goodbye Jarvis") is intentional: leave a marker so the watcher does not relaunch the voice line. run.sh removes
    it on the next launch. A crash or a closed Terminal window writes nothing, so those ARE restarted. Never raises: hanging up must work."""
    try:
        with open(path or QUIT_MARKER, "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        log("[backtalk] hung up on purpose: the watcher will not relaunch until it is started by hand")
    except Exception:
        pass


def press_cancels_turn(speak_task, guard) -> bool:
    """A talk-key press interrupts the turn in flight, EXCEPT while the destructive-command guard is waiting on a spoken answer: then
    the press is how he answers, and cancelling the turn would cancel the question (denied before he spoke; mic test 2026-10-22)."""
    return bool(speak_task and not speak_task.done() and not guard.pending)


def code_revision(run=None):
    """'<short sha> clean|dirty' of the source tree this process was started from, so a log line proves WHICH code is running ('dirty' = tracked
    source files differ from that commit; runtime files such as the session id and watcher.log are not counted). 'unknown' if git cannot answer."""
    import subprocess
    run = run or (lambda *a: subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *a],
                                            capture_output=True, text=True, timeout=5).stdout)
    try:
        sha = run("rev-parse", "--short=10", "HEAD").strip()
        changed = [l[3:] for l in run("status", "--porcelain", "--untracked-files=no").splitlines()
                   if l[3:] and not l[3:].startswith((".backtalk_session", "watcher.log"))]
        return f"{sha} {'dirty' if changed else 'clean'}" if sha else "unknown"
    except Exception:
        return "unknown"


def main():
    # A previous run ended because the Claude Code login expired: do one cheap
    # local check before loading anything, and stay silent while still signed
    # out, so a relauncher cannot turn this into a loop of spoken errors.
    _gate = authguard.startup_gate(authguard.default_marker_path())
    if _gate is not None:
        print(authguard.SIGNIN_PRINTED, flush=True)
        sys.exit(_gate)
    if not _claim_single_instance():
        print("[backtalk] ANOTHER VOICE LINE IS ALREADY RUNNING on this "
              "machine, so this one is stopping.", flush=True)
        print("[backtalk] Two of them fight over the microphone and the "
              "talk key, which looks exactly like the talk key being "
              "broken. Use the window that is already open, or close it "
              "and start again.", flush=True)
        sys.exit(1)
    _install_hangup_logger()
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[backtalk] interrupted — hanging up", flush=True)
        # an interrupt that escaped amain's own shutdown: still leave now, or
        # Python waits forever on the blocked worker threads
        _exit_now()


if __name__ == "__main__":
    main()
