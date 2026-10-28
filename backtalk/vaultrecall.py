"""Automatic vault recall: inject the strongest semantic-search matches
into a turn before it reaches the brain, so relevant memory shows up
without Yousif deciding to go looking for it (item 6, Jarvis Feel and
Speed Plan).

Held back on 2026-09-22 pending a hard token cap: the same night, 53% of
all token spend was measured as coming from oversized context (see
Session Cost and Context Discipline) -- this must not quietly undo the
150k resume cap by fattening every turn. Two guards enforce that:
  - MIN_SIM: only injects on a genuinely strong match, not any match.
  - MAX_INJECT_TOKENS: a hard cap on what one turn can add.
Both bound a SINGLE injection, not cumulative growth over a long
session -- if repeated injections turn out to matter in practice,
status/usage.py already measures where spend goes and would show it.
"""
import asyncio
import json
import os
import time

import numpy as np
import requests

from backtalk.vlog import log

INDEX_FILE = "/Users/yousif/Desktop/jarvis/vault-search/index.json"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

MIN_SIM = 0.6               # cosine similarity floor -- below this, skip
MAX_INJECT_TOKENS = 2000    # hard per-turn cap, ~4 chars/token estimate
TOP_N = 4                   # candidates considered before the cap trims them
# Recall sits in front of the brain on every turn, so its worst case IS the
# turn's added dead air. Cut from 1.5s to 0.4s on 2026-09-22 after Yousif
# said replies felt slow: measured calls here ran 1.12s and 1.31s cold,
# ~0.04s warm, meaning a cold Ollama was adding over a second of silence to
# every question he asked. Recall is a bonus, never a requirement -- when
# it can't answer fast, the right move is to skip it and let him have his
# reply sooner.
EMBED_TIMEOUT_S = 0.4
# Short conversational lines ("you there?", "Jarvis?") have nothing to
# recall and were matching junk anyway -- "Jarvis are you here?" pulled
# 1,644 tokens of livestream transcript on 2026-09-22. Anything under this
# many words skips the query entirely, which also spares it the round trip.
MIN_WORDS = 5

_index_cache: dict | None = None
_index_mtime = 0.0


def _load_index() -> dict | None:
    global _index_cache, _index_mtime
    try:
        mtime = os.path.getmtime(INDEX_FILE)
    except OSError:
        return None
    if _index_cache is None or mtime != _index_mtime:
        with open(INDEX_FILE, encoding="utf-8") as f:
            _index_cache = json.load(f)
        _index_mtime = mtime
    return _index_cache


# Ollama unloads an idle model (5 min by default), and reloading it costs
# ~0.55s -- which lands as dead air in front of Yousif's reply. Asking it to
# stay resident makes the normal path ~0.015s, a 25x margin inside the
# timeout above, so recall is effectively free instead of a gamble.
KEEP_ALIVE = "30m"


def _embed(text: str, timeout: float = EMBED_TIMEOUT_S) -> np.ndarray:
    r = requests.post(OLLAMA_URL,
                      json={"model": MODEL, "prompt": text,
                            "keep_alive": KEEP_ALIVE},
                      timeout=timeout)
    r.raise_for_status()
    return np.array(r.json()["embedding"])


def warm() -> None:
    """Load the embedding model and start its keep-alive clock, so the
    first real question of a session doesn't pay the cold load. Called off
    the startup path; failure here is harmless (recall just skips until
    Ollama is up)."""
    try:
        t0 = time.monotonic()
        _embed("warm", timeout=30)
        log(f"[vault] recall warm ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        log(f"[vault] recall not available: {e!r}")


# PHASE J1 (2026-09-28): Ollama's 30m keep_alive above is only ever refreshed
# by an actual embed call. A long gap between vault-eligible utterances (an
# hour is normal) lets it lapse -- and once cold, the conversational path can
# never re-warm it on its own: EMBED_TIMEOUT_S (0.4s) is shorter than a real
# cold load (measured 0.54-1.9s), and a client that gives up at 0.4s cancels
# Ollama's in-flight load rather than letting it finish in the background
# (confirmed directly: killing a request at 0.4s left `ollama ps` empty
# afterward). So a cold model, mid-conversation, stays cold turn after turn.
# This ping stays well inside the 30m window so it never has the chance to
# lapse during a normal session -- same model, same warm() call, same 30m
# keep_alive; nothing new is loaded.
REWARM_INTERVAL_S = 20 * 60


async def keep_warm_loop(interval_s: float = REWARM_INTERVAL_S) -> None:
    """Background heartbeat: re-warm the embedding model every ~20 minutes
    for as long as the process runs. Sleeps on the event loop and runs the
    (blocking) warm() call in a thread executor each time, so this never
    competes with speech, STT, or a Claude request for the event loop.
    warm() already swallows and logs its own failures (Ollama down, etc.)
    without raising; the try/except here is a second layer so a truly
    unexpected error -- in the executor plumbing itself, not the HTTP call
    -- still can't kill this loop or the process."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(interval_s)
        log("[vault] periodic re-warm: pinging embedding model")
        try:
            await loop.run_in_executor(None, warm)
        except Exception as e:
            log(f"[vault] periodic re-warm crashed unexpectedly: {e!r}")


def context_for(utterance: str) -> str | None:
    """Top vault matches for this utterance, formatted for injection, or
    None if nothing scored well enough to be worth the tokens (or the
    index/Ollama isn't available). Never raises -- a dead or slow Ollama
    just means no recall this turn, not a broken one, so this always
    runs off the async path with its own timeout."""
    if len(utterance.split()) < MIN_WORDS:
        return None
    idx = _load_index()
    if not idx or not idx.get("entries"):
        return None
    t0 = time.monotonic()
    try:
        qvec = _embed(utterance)
    except Exception as e:
        log(f"[vault] recall skipped -- embed failed: {e!r}")
        return None
    qnorm = qvec / np.linalg.norm(qvec)

    scored = []
    for e in idx["entries"]:
        v = np.array(e["vec"])
        sim = float(np.dot(qnorm, v) / np.linalg.norm(v))
        scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = [(sim, e) for sim, e in scored[:TOP_N] if sim >= MIN_SIM]
    if not hits:
        return None

    lines = []
    used_files = []
    budget = MAX_INJECT_TOKENS * 4     # chars, ~4 chars/token estimate
    for sim, e in hits:
        snippet = e["text"].strip()
        if not snippet or budget <= 0:
            break
        if len(snippet) > budget:
            snippet = snippet[:budget]
        lines.append(f"- {e['file']} (match {sim:.2f}): {snippet}")
        used_files.append(e["file"])
        budget -= len(snippet)

    if not lines:
        return None
    block = ("[From the vault, background only -- use it if it helps, "
             "don't read it out loud or mention it unless he asks:\n"
             + "\n".join(lines) + "\n]")
    log(f"[vault] recall: {len(lines)} match(es) from "
        f"{', '.join(used_files)}, ~{len(block) // 4} tokens, "
        f"{time.monotonic() - t0:.2f}s")
    return block
