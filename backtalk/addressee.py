"""The addressee gate: is this open-mic speech meant for the agent?

STATE MACHINE (every transition is explicit and logged):

    IDLE --valid wake / held key / typed line--> ACTIVE
    ACTIVE --explicit end phrase---------------> IDLE   (end())
    ACTIVE --idle_window_s with no user turn---> IDLE   (idle_timeout)
    ACTIVE --max_active_s since the last wake--> IDLE   (max_lifetime)
    ACTIVE --valid wake again------------------> ACTIVE (hard cap restarts)

TIMERS
  * idle window (active_window_s, 180): pushed out ONLY by an accepted human
    follow-up. Never by the agent speaking, a reply finishing, or a turn
    running.
  * hard cap (max_active_s, 300): measured from the last explicit address
    (wake phrase, held key, typed line). Follow-ups cannot extend it, so
    ambient speech cannot keep the mic open indefinitely. Saying the wake
    phrase again restarts it.

WAKE is conservative and deterministic: the name must OPEN the sentence
(optionally after "hey", "okay", ...), and must not be followed by a
statement verb ("Jarvis is cool", "Jarvis was mentioned"). A name anywhere
else ("I wasn't talking to you, Jarvis", "my friend said Jarvis...") is not
an address. Whisper's punctuation is not trusted: a comma after the name
helps but is not required. Known cost: a trailing vocative ("open YouTube,
Jarvis") does not wake from IDLE; it works while ACTIVE.

END phrases ("that's all", "I'm done", "stop listening", "go idle",
"never mind", and "I wasn't talking to you") leave ACTIVE and are NOT sent to
the agent.

There is no "busy" admission: speech is never admitted merely because the
agent is speaking. Legitimate interruptions are ACTIVE-state follow-ups (the
speaker is mid-conversation), or a bare stop phrase, which the caller exempts.

THIS IS AN ADDRESSEE GATE, NOT SPEAKER AUTHENTICATION. Anyone or anything
that says a valid wake phrase activates it, and while ACTIVE any voice is
treated as the conversation. Nothing downstream asks for approval, so speech
this gate admits runs tools directly.

Applies to open-mic transcripts only. A held talk key and typed input are
explicit addresses: the caller opens the conversation with open().
"""
import difflib
import re
import time
from dataclasses import dataclass

from backtalk.vlog import log

IDLE = "IDLE"
ACTIVE = "ACTIVE"

# Words that may precede the name and still be addressing it.
_INTERJECTIONS = {"hey", "hi", "hello", "yo", "okay", "ok", "alright", "so",
                  "well", "um", "uh", "oh", "ay"}
# "Jarvis <verb>" with one of these next is a statement ABOUT the name.
_STATEMENT_WORDS = {"is", "was", "are", "were", "has", "had", "said", "says",
                    "means", "seems", "looks", "sounds", "isnt", "wasnt",
                    "arent", "werent"}
# Whole-utterance end phrases (name and fillers stripped, apostrophes gone).
END_PHRASES = {"thats all", "that will be all", "im done", "i am done",
               "were done", "stop listening", "go idle", "never mind",
               "nevermind"}
# "I wasn't / I'm not talking to you (Jarvis)": the user says it was not for
# the agent. Ends the conversation from ACTIVE.
_NOT_FOR_ME = re.compile(
    r"\b(not|wasnt|isnt|arent|werent|never|nobody)\s+(?:been\s+|be\s+)?"
    r"(talking|speaking)\s+to\s+(you|jarvis|u)\b")
_END_FILLER = {"ok", "okay", "alright", "well", "so", "yeah", "yes", "um",
               "uh", "hey", "please", "thanks"}


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", (text or "").lower()).replace(
        "'", "").split()


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str           # wake | active | exempt |
                          # dev_always | idle_no_wake | end_conversation


class AddresseeGate:
    def __init__(self, wake_words, heard_as=(), mode: str = "wake",
                 active_window_s: float = 180.0, clock=time.monotonic,
                 max_active_s: float = 300.0):
        self.wake = [_words(w) for w in wake_words if _words(w)]
        self.heard_as = {w for h in heard_as for w in _words(h)}
        self.mode = mode if mode in ("wake", "always") else "wake"
        self.window = float(active_window_s)
        self.max_active = float(max_active_s)
        self._clock = clock
        self._idle_until = 0.0     # follow-up deadline (extended by users)
        self._hard_until = 0.0     # hard cap (only a new address moves it)
        self._logged = IDLE        # last state we logged a transition for
        self._names = {p[-1] for p in self.wake}
        self._prefix = _INTERJECTIONS | {w for p in self.wake for w in p[:-1]}

    @classmethod
    def from_config(cls, cfg: dict, name: str, always: bool = False):
        a = cfg.get("addressee") or {}
        low = name.lower()
        wake = a.get("wake_words") or [low, f"hey {low}"]
        return cls(wake, a.get("heard_as") or [],
                   "always" if always else a.get("mode", "wake"),
                   a.get("active_window_s", 180.0),
                   max_active_s=a.get("max_active_s", 300.0))

    # ---- state ----------------------------------------------------------

    @property
    def state(self) -> str:
        now = self._clock()
        return ACTIVE if now < min(self._idle_until, self._hard_until) \
            else IDLE

    def _sync(self) -> None:
        """Log a lapse the first time it is noticed, with its real cause."""
        if self._logged == ACTIVE and self.state == IDLE:
            cause = ("max_lifetime" if self._clock() >= self._hard_until
                     and self._hard_until <= self._idle_until
                     else "idle_timeout")
            self._logged = IDLE
            log(f"[gate] event=state_transition from=ACTIVE to=IDLE "
                f"cause={cause}")

    def open(self, cause: str = "wake") -> None:
        """An explicit address (wake phrase, held key, typed line): start or
        restart the conversation, including the hard cap."""
        self._sync()
        now = self._clock()
        was = self.state
        self._idle_until = now + self.window
        self._hard_until = now + self.max_active
        self._logged = ACTIVE
        log(f"[gate] event=state_transition from={was} to=ACTIVE "
            f"cause={cause} idle={self.window:.0f}s "
            f"max={self.max_active:.0f}s")

    def _follow_up(self) -> None:
        """An accepted human follow-up: extend the idle window, never past
        the hard cap (state's min() enforces that)."""
        self._idle_until = self._clock() + self.window

    def end(self, cause: str = "end_phrase") -> None:
        """Leave ACTIVE now."""
        was = self.state
        self._idle_until = 0.0
        self._hard_until = 0.0
        self._logged = IDLE
        log(f"[gate] event=state_transition from={was} to=IDLE "
            f"cause={cause}")

    # ---- wake / end detection ------------------------------------------

    @staticmethod
    def _word_is_name(w: str, target: str) -> bool:
        return (w == target or (len(w) >= 4 and len(target) >= 4 and
                difflib.SequenceMatcher(None, w, target).ratio() >= 0.8))

    def _is_name(self, w: str) -> bool:
        return w in self.heard_as or any(self._word_is_name(w, t)
                                         for t in self._names)

    def is_wake(self, text: str) -> bool:
        """Is this an address TO the agent? The name must open the sentence."""
        toks = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text or "")
        i = 0
        while i < len(toks):
            w = toks[i].lower().replace("'", "")
            if w in self._prefix and w not in self._names \
                    and not self._is_name(w):
                i += 1
                continue
            break
        if i >= len(toks):
            return False
        first = toks[i].lower().replace("'", "")
        if not self._is_name(first):
            return False
        rest = toks[i + 1:]
        if not rest:
            return True                                  # "Jarvis"
        nxt = rest[0]
        if not re.match(r"[A-Za-z0-9']", nxt):
            return True                                  # "Jarvis, ..."
        if nxt.lower().replace("'", "") in _STATEMENT_WORDS:
            return False                                 # "Jarvis is cool"
        return True                                      # "Jarvis open ..."

    def is_end(self, text: str) -> bool:
        ws = _words(text)
        norm = " ".join(ws)
        if _NOT_FOR_ME.search(norm):
            return True
        core = [w for w in ws
                if not self._is_name(w) and w not in _END_FILLER]
        return " ".join(core) in END_PHRASES

    # ---- the decision ---------------------------------------------------

    def check(self, text: str, *, exempt: bool = False) -> Decision:
        """Should this open-mic transcript become an agent turn?
        `exempt`: caller-owned always-allowed controls
        (zip/unzip, a stop phrase over the voice); never activates."""
        self._sync()
        if self.mode == "always":
            d = Decision(True, "dev_always")
        elif exempt:
            d = Decision(True, "exempt")
        elif self.state == ACTIVE and self.is_end(text):
            d = Decision(False, "end_conversation")
        elif self.is_wake(text):
            d = Decision(True, "wake")
        elif self.state == ACTIVE:
            d = Decision(True, "active")
        else:
            d = Decision(False, "idle_no_wake")

        if d.reason == "end_conversation":
            self.end("end_phrase")
            log("[gate] event=conversation_ended cause=end_phrase "
                f"words={len(_words(text))}")
        elif d.accepted:
            if d.reason == "wake":
                self.open("wake")
            elif d.reason == "active":
                self._follow_up()
            log(f"[gate] event=addressee_accepted reason={d.reason} "
                f"state={self.state}")
        else:
            log(f"[gate] event=addressee_rejected reason={d.reason} "
                f"words={len(_words(text))}")
        return d
