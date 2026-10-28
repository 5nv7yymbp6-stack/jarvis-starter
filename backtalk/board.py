"""What's on the HOLO board, for the voice brain (2026-09-16).

Upstream's stage fed its Jarvis a text event for every grab and placement
(livestream 0:22, 3:48), but the feed lagged by minutes and, with chat on
top, buried him in events. Here nothing is pushed: at the start of each
turn, the board's last snapshot (the page posts one every second to the
holo server's /api/state) is compared with what the brain was last told,
and only the difference rides along with what the person said. Always
current, never chatter, and a dark board costs nothing.

Config: board_url (the /api/state URL; empty turns this off).
"""
import json
import time
import urllib.request
from collections import Counter

from backtalk.config import CFG

_last: Counter | None = None      # what the brain was last told is on it
STALE_S = 5.0                      # an older snapshot means the page is closed


TITLE_MAX = 60                     # a card title is untrusted text (vault notes, add_card)
KIND_MAX = 16
ITEMS_MAX = 40


def _clean(v, cap) -> str:
    """Untrusted text made safe to quote in a prompt note: printable single line, no quotes or brackets
    (so it can't close the note or fake another), capped. Non-strings become empty."""
    if not isinstance(v, str):
        return ""
    v = "".join(c if c.isprintable() and c not in '"[]{}<>`\\' else " " for c in v)
    return " ".join(v.split())[:cap].strip()


def _label(item) -> str:
    if not isinstance(item, dict):
        return "item"
    kind = _clean(item.get("kind"), KIND_MAX) or "item"
    kind = {"obj": "3D model"}.get(kind, kind)
    title = _clean(item.get("title"), TITLE_MAX)
    return f'{kind} "{title}"' if title else kind


_SKIP = object()                   # no snapshot this instant; board still up


def _read():
    """Counter of what's on the board; None when it's dark (no server, no
    page open); _SKIP when the page's last post was an event, not a
    snapshot (the next snapshot replaces it within a second)."""
    url = CFG.get("board_url") or ""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=0.3) as r:
            data = json.loads(r.read() or b"{}")
    except Exception:
        return None                    # server down: the board is dark
    if time.time() - float(data.get("ts") or 0) > STALE_S:
        return None                    # no page posting
    items = data.get("items")
    if items is None:
        return _SKIP
    if not isinstance(items, list):
        return None
    return Counter(_label(i) for i in items[:ITEMS_MAX])


def _list(c: Counter) -> str:
    return ", ".join(f"{k} x{n}" if n > 1 else k for k, n in sorted(c.items()))


def changes() -> str:
    """A note for the brain about the board, or "" when there's nothing new
    to say: the board is dark, or nothing changed since the last turn."""
    global _last
    now = _read()
    if now is _SKIP:
        return ""
    if now is None:
        _last = None                   # say what's there when it comes back
        return ""
    if _last is None:
        _last = now
        what = _list(now) if now else "empty"
        return f"[Holo board (seen by the system, not said by him): {what}.]"
    added, removed = now - _last, _last - now
    _last = now
    if not added and not removed:
        return ""
    parts = []
    if added:
        parts.append(f"added {_list(added)}")
    if removed:
        parts.append(f"removed {_list(removed)}")
    return ("[Holo board changed since last turn (seen by the system, not "
            f"said by him): {'; '.join(parts)}.]")
