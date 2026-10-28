"""How big a Claude Code session has actually got, and whether it is still
worth reattaching to.

Why this exists: on 2026-09-22 Yousif said that opening Jarvis eats 2-3% of
his usage, and that a single Cmd-J sometimes took 40%. Measured across all
126 sessions in this project (8,107 turns, ~290 M token-equivalents):

  * 53% of ALL spend came from the 1,636 turns run at a context over 500 k
    tokens. Peak context seen: 965,774.
  * 13% came from just 58 cold-start turns, each re-caching a whole huge
    session. Worst single turn: a 936,150-token cache write.
  * The 2-3% per open is a WARM resume of a ~740 k session -- about 74 k
    token-equivalents every turn just to re-read context that never changes.
    The 40% is a COLD one: at 03:08:04 that morning, after a 44-hour gap,
    one turn paid a 701,202-token cache write. One turn.

Reattaching forever is also the opposite of how this agent is supposed to
work. BOOT-CONFIG is explicit: the memory lives in the vault, not in the
context window -- "hold the job and know where the rest is." A 900 k-token
session is context hoarding, and it is billed on every single turn.

So: resuming stays the default, because remembering across a Terminal close
is genuinely useful -- but only while the conversation is still small enough
to be worth carrying. Past the cap it starts fresh and says so out loud. The
vault carries everything that mattered.

The audio watchdog's in-place restart is deliberately NOT capped: that is a
crash recovery in the middle of a live conversation, and dropping context
there would be worse than the cost.
"""
import glob
import json
import os

from backtalk.config import CFG
from backtalk.vlog import log

# Past this many tokens of context, a saved conversation is not reattached.
# Tuned from the measurements above: the cheap sessions all sit well under
# 150 k, and the spend curve turns sharply upward past ~500 k. 150 k keeps a
# long working conversation resumable while making a 700 k monster
# impossible to silently inherit.
DEFAULT_MAX_TOKENS = 150_000

_TAIL_BYTES = 512 * 1024      # enough to hold the last several turns


def _transcript(session_id: str) -> str | None:
    """The session's JSONL, wherever Claude Code filed it."""
    if not session_id:
        return None
    hits = glob.glob(os.path.expanduser(
        f"~/.claude/projects/*/{session_id}.jsonl"))
    return hits[0] if hits else None


def context_tokens(session_id: str) -> int | None:
    """Tokens the LAST turn of this session carried, or None if unknown.

    That is the real cost of reattaching: every turn re-sends the whole
    context, cheaply while the cache is warm and at 1.25x when it is not.
    Reads only the tail of the file -- one of these is 224 MB.
    """
    path = _transcript(session_id)
    if not path:
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
                f.readline()              # drop the partial line
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    best = None
    for line in tail.splitlines():
        if '"usage"' not in line:
            continue
        try:
            u = ((json.loads(line).get("message") or {}).get("usage")) or {}
        except Exception:
            continue
        n = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
             + u.get("cache_creation_input_tokens", 0))
        if n:
            best = n
    return best


def resumable(session_id: str | None) -> str | None:
    """The session id to actually reattach to -- or None to start fresh
    because carrying this one costs more than it is worth."""
    if not session_id:
        return None
    cap = int(CFG.get("resume_max_tokens") or DEFAULT_MAX_TOKENS)
    if cap <= 0:
        return session_id                 # capping switched off on purpose
    n = context_tokens(session_id)
    if n is None:
        return session_id                 # can't tell; don't punish it
    if n <= cap:
        log(f"[session] resuming {session_id[:8]} -- {n:,} tokens of context")
        return session_id
    log(f"[session] NOT resuming {session_id[:8]}: {n:,} tokens of context, "
        f"over the {cap:,} cap. Every turn would re-read that, and a cold "
        f"start would re-cache it at 1.25x. Starting fresh -- the vault has "
        f"the memory.")
    return None
