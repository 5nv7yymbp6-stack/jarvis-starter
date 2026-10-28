"""Backchannel classification (roadmap Phase D). PURE: no audio, no model, no state, no clock, no config.

A BACKCHANNEL is a short acknowledgement a person makes while someone else talks ("okay", "mm-hmm", "right", "I see", "thanks").
It is not a request and not a cut-in, so it must not cancel the reply being spoken or the work behind it. Evidence (the live log,
2026-09-10 to 10-19): 179 of 439 interrupts were triggered by a transcript of two words or fewer ("okay", "thank you", "oh",
"mm-hmm"), 514 voice pauses and 128 dropped replies. Before this, `over_speech` and `handle()` treated every non-stop utterance
heard over the voice as a cut-in.

CONSERVATIVE BY DESIGN (a missed backchannel costs what it costs today; a swallowed real request costs the request):
  * only a short CLOSED list, one to three words;
  * never if any word is a stop or cancel word (stop wait enough pause quiet no hold cancel never mind) or the agent's name
    (that is addressing it, not acknowledging it);
  * never an answer-type word ("yes", "yeah", "yep", "sure", "okay", "ok", "right", "alright") when the voice's last spoken
    chunk was a QUESTION: then it IS the answer;
  * anything not on the list is NOT a backchannel. Two spellings of the list are allowed: an elongated word ("mmmmmm", "okayyy",
    "hmmmm") collapses to its listed form, and one listed HUM said two or three times in a row ("mm mm") counts; a repeated word ("yeah yeah") does not. Neither relaxes the stop-word, name or question guards (checked on every spelling, e.g. "nooo", "stoppp").
"""
import re

BACKCHANNELS = frozenset({
    "okay", "ok", "k", "yeah", "yep", "yes", "yup", "right", "sure", "alright", "all right", "oh", "ah", "huh", "hm", "hmm", "mm", "mmm",
    "mhm", "mmhmm", "mm hmm", "mm-hmm", "uh huh", "uh-huh", "i see", "got it", "gotcha", "thanks", "thank you", "cool", "nice",
    "great", "wow", "interesting", "good", "fine", "perfect", "exactly", "true", "oh okay", "oh ok", "okay cool", "ok cool",
    "yeah okay", "yeah ok", "okay thanks", "ok thanks", "okay great", "oh right", "oh i see", "oh nice", "oh cool",
})
ANSWER_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "k", "right", "alright", "all", "exactly", "true",
                          "fine", "good", "perfect"})
NEVER_WORDS = frozenset({"stop", "wait", "enough", "pause", "quiet", "no", "nope", "nah", "hold", "hang", "cancel", "never", "mind",
                         "jarvis", "boss", "sir", "hey", "please", "don't", "dont", "not"})
MAX_WORDS = 3
_PUNCT = re.compile(r"[^\w\s'\-]")
# Every single token that appears inside a listed phrase: the vocabulary an elongated word ("mmmmmm", "okayyy") may collapse into.
VOCAB = frozenset(w for phrase in BACKCHANNELS for w in re.split(r"[\s\-]+", phrase) if w)
# Only HUMS may be repeated ("mm mm"): a repeated word ("yeah yeah", "okay okay", "right right") is as often impatience as agreement.
REPEATABLE = frozenset({"hm", "hmm", "mm", "mmm", "mhm", "mmhmm", "mm-hmm", "huh", "ah", "oh"})
_RUN3 = re.compile(r"(.)\1{2,}")
_RUN2 = re.compile(r"(.)\1+")


def words(text) -> list:
    """Lowercase words, punctuation stripped (hyphens and apostrophes kept inside a word)."""
    return [w for w in _PUNCT.sub(" ", str(text or "").lower()).split() if w]


def _shapes(word) -> list:
    """The word as heard, then with runs of 3+ equal letters cut to 2, then to 1: how an elongated hum or "okayyy" is spelled."""
    return [word, _RUN3.sub(r"\1\1", word), _RUN2.sub(r"\1", word)]


def _normalize(word) -> str:
    """The first shape of the word that is in the backchannel vocabulary, else the word unchanged."""
    for shape in _shapes(word):
        if shape in VOCAB:
            return shape
    return word


def spoken_was_question(chunks) -> bool:
    """True when ANY recent chunk the voice spoke ends in a question mark (a trailing quote or bracket is ignored). `chunks` is
    Mouth.recent_speech(): the last few chunks, up to 30 s. Any, not just the last: an answer to a question asked two chunks ago
    ("Want me to send it?" ... "I'll draft it meanwhile.") must not be swallowed as an acknowledgement."""
    try:
        return any(str(c).rstrip(" \t\n\"')]}").endswith("?") for c in (chunks or []) if str(c).strip())
    except Exception:
        return False


def is_backchannel(text, *, last_spoken_was_question=False) -> bool:
    raw = words(text)
    if not 1 <= len(raw) <= MAX_WORDS:
        return False
    # a stop or cancel word wins however it is stretched or spelled ("nooo", "stoppp")
    if any(shape in NEVER_WORDS for w in raw for shape in _shapes(w)):
        return False
    ws = [_normalize(w) for w in raw]
    phrase = " ".join(ws)
    listed = phrase in BACKCHANNELS or phrase.replace("-", " ") in BACKCHANNELS
    # the same hum said twice or three times ("mm mm", "mm-hmm mm-hmm")
    repeated = len(ws) > 1 and len(set(ws)) == 1 and ws[0] in REPEATABLE
    if not (listed or repeated):
        return False
    if last_spoken_was_question and any(w in ANSWER_WORDS for w in ws):
        return False
    return True
