"""A blocked web fetch is not a dead end: open the page in the browser.

2026-10-26: Yousif asked Jarvis to check his LLC on Missouri's business search.
WebFetch got HTTP 403, and Jarvis told him "Can't pull it directly... you'll
need to run it yourself". Asked again ("look it up, figure it out"), Jarvis
opened the Playwright browser and had the answer in 40 s. The browser was
there the whole time; nothing pointed him at it after the fetch failed.

These hooks (PostToolUse and PostToolUseFailure, matcher WebFetch) watch for a
fetch that came back blocked or failed and add a note to the tool result
telling the model to open the same page in the Playwright browser before
answering, and never to hand the lookup back to him. They only add context;
they never change or block anything, and every error is swallowed.
"""
import re

from backtalk.vlog import log

_HTTP_RE = re.compile(r"\b(?:returned\s+)?HTTP\s+(\d{3})\b", re.I)
_FAIL_RE = re.compile(r"\b(forbidden|access denied|blocked|captcha|unable to fetch|"
                      r"failed to fetch|timed? ?out|request failed|cloudflare)\b", re.I)


def blocked(tool_response) -> str | None:
    """A short reason when the fetch came back blocked or failed, else None."""
    try:
        if isinstance(tool_response, dict):
            code = tool_response.get("code")
            if isinstance(code, int) and code >= 400:
                return f"HTTP {code}"
            text = str(tool_response.get("result") or "")
        else:
            text = str(tool_response or "")
        m = _HTTP_RE.search(text)
        if m and int(m.group(1)) >= 400:
            return f"HTTP {m.group(1)}"
        # only trust failure words when the fetch returned almost nothing:
        # a real page can mention "blocked" or "Cloudflare" in its content
        if len(text) < 400:
            w = _FAIL_RE.search(text)
            if w:
                return w.group(1).lower()
    except Exception:
        pass
    return None


def note(url: str, reason: str) -> str:
    return (f"That fetch failed ({reason}), which only means this tool was refused, not "
            f"that the page can't be read. Open {url or 'the same page'} in the Playwright "
            f"browser now (mcp__playwright__browser_navigate, then read the page or fill its "
            f"search form) and get the answer yourself before replying. Do not tell Yousif "
            f"to look it up himself or offer to walk him through it. Only if the browser "
            f"also fails, say plainly what you tried and what stopped you.")


def _out(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


async def post_tool(input_data, tool_use_id, context):
    """PostToolUse hook (matcher WebFetch). Never raises."""
    try:
        data = input_data or {}
        if data.get("tool_name") != "WebFetch":
            return {}
        reason = blocked(data.get("tool_response"))
        if not reason:
            return {}
        url = str((data.get("tool_input") or {}).get("url") or "")
        log(f"[fetchguard] fetch blocked ({reason}), pointing at the browser: {url[:120]}")
        return _out("PostToolUse", note(url, reason))
    except Exception:
        return {}


async def post_tool_failure(input_data, tool_use_id, context):
    """PostToolUseFailure hook (matcher WebFetch). Never raises."""
    try:
        data = input_data or {}
        if data.get("tool_name") != "WebFetch" or data.get("is_interrupt"):
            return {}
        reason = blocked(data.get("error")) or "error"
        url = str((data.get("tool_input") or {}).get("url") or "")
        log(f"[fetchguard] fetch failed ({reason}), pointing at the browser: {url[:120]}")
        return _out("PostToolUseFailure", note(url, reason))
    except Exception:
        return {}
