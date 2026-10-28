"""Destructive-command guard: a SHORT deny-list that asks by voice (Yousif's decision, 2026-10-22; the full tier system stays removed).

Only the Bash tool is checked, and every command segment (after && ; | & newlines, inside $( ) and backticks, inside `bash -c` / `bash -lc`
and `eval`). Blocked until a spoken yes: sudo; rm with recursive/force flags aimed at /, ~, $HOME, /Users or the home folder; git push
--force / -f / --force-with-lease / +refspec; keychain reads (security find-*-password, dump-keychain, export). Everything else runs directly.

Fails closed, everywhere: no answer in `timeout_s`, any answer that is not yes-words only, no voice to ask with, or ANY error or
cancellation inside the hook => deny. (Verified live 2026-10-22: a PreToolUse hook that raises lets the CLI run the command, so nothing
may escape pre_tool.) The question names the target ("a force push to origin main"); URLs and anything credential-shaped are never spoken.
The log records the decision, the tool-use id and a hash of the command, never the command itself.
It sees commands as written; it does not see what a script does inside. The phone bridge and the dream job have no Bash at all.
"""
import asyncio
import hashlib
import os
import re
import shlex
import time

from backtalk.vlog import log

HOME = os.path.expanduser("~")
_WRAPPERS = {"env", "command", "nohup", "time", "exec", "builtin", "nice", "caffeinate", "xargs", "stdbuf"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
_SPLIT = re.compile(r"&&|\|\||[;|&\n]")
_SUBST = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_DIR_NAMES = {"/": "the whole disk", "/Users": "every user folder", os.path.normpath(HOME): "your whole home folder"}
_SEC_BAD = {"dump-keychain": "that dumps the keychain", "export": "that exports keychain items",
            "find-generic-password": "that reads a keychain password", "find-internet-password": "that reads a keychain password"}
_GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix", "--config-env"}

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "affirmative"}
_FILLER = {"boss", "sir", "please", "jarvis", "ok", "okay"}
_YES_PHRASES = {"go ahead", "do it", "go for it", "proceed", "approved", "confirmed", "go ahead and do it", "go ahead do it"}
_NO_WORDS = re.compile(r"\b(no|not|don'?t|do not|stop|cancel|wait|hold|never|nope|negative|abort)\b")


def _norm_path(tok: str) -> str:
    p = tok.replace("${HOME}", HOME).replace("$HOME", HOME)
    if p == "~" or p.startswith("~/"):
        p = HOME + p[1:]
    p = os.path.normpath(p)
    if p == "/*":
        return "/"
    return p[:-2] if p.endswith("/*") else p


def _say_safe(tok: str) -> str:
    """A token fit to say aloud: plain names only. URLs, user@host and anything long or odd become a generic phrase."""
    t = tok.lstrip("+")
    if "://" in t or "@" in t or not re.fullmatch(r"[A-Za-z0-9._/-]{1,40}", t):
        return "a remote address"
    return t


def _strip(tokens):
    """Drop leading VAR=x assignments and wrapper commands (and their flags). `command -v X` only looks X up: nothing runs."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if os.path.basename(t) == "command" and i + 1 < len(tokens) and tokens[i + 1] in ("-v", "-V"):
            return []
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) or os.path.basename(t) in _WRAPPERS or (i > 0 and t.startswith("-") and os.path.basename(tokens[0]) in _WRAPPERS):
            i += 1
            continue
        break
    return tokens[i:]


def _git_sub(t):
    """git's subcommand index (first non-option word after `git`), skipping global options and their values."""
    i = 1
    while i < len(t):
        x = t[i]
        if x in _GIT_OPTS_WITH_ARG:
            i += 2
        elif x.startswith("-"):
            i += 1
        else:
            return i
    return None


def _check_tokens(tokens, depth):
    t = _strip(tokens)
    if not t:
        return None
    name = os.path.basename(t[0])
    if name in ("sudo", "doas"):
        what = next((os.path.basename(x) for x in t[1:] if not x.startswith("-")), "")
        return "that runs sudo" + (" " + _say_safe(what) if what else "")
    if name == "rm":
        flags = [x for x in t[1:] if x.startswith("-") and x != "--"]
        risky = any((x.startswith("--") and x in ("--recursive", "--force", "--no-preserve-root")) or (not x.startswith("--") and re.search(r"[rRf]", x)) for x in flags)
        if risky:
            for x in t[1:]:
                if not x.startswith("-") and _norm_path(x) in _DIR_NAMES:
                    return "that deletes " + _DIR_NAMES[_norm_path(x)]
    if name == "git":
        s = _git_sub(t)
        if s is not None and t[s] == "push":
            args = t[s + 1:]
            forced = any(x in ("--force", "-f", "--force-if-includes") or x.startswith("--force-with-lease")
                         or re.match(r"^-[A-Za-z]*f[A-Za-z]*$", x) or (x.startswith("+") and len(x) > 1) for x in args)
            if forced:
                where = [_say_safe(x) for x in args if not x.startswith("-")][:2]
                return "that's a force push" + (" to " + " ".join(where) if where else "")
    if name == "security":
        rest, skip = [], False
        for x in t[1:]:
            if skip:
                skip = False
            elif x == "-p":
                skip = True
            elif not x.startswith("-"):
                rest.append(x)
        if rest and rest[0] in _SEC_BAD:
            return _SEC_BAD[rest[0]]
    if name == "eval" and len(t) > 1:
        return classify_command(" ".join(t[1:]), depth + 1)
    if name in _SHELLS:
        for k, x in enumerate(t[1:-1], 1):
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", x):
                return classify_command(t[k + 1], depth + 1)
    return None


def classify_command(cmd: str, _depth: int = 0):
    """The spoken reason a command needs a yes, or None."""
    if not cmd or _depth > 4:
        return None
    for m in _SUBST.finditer(cmd):
        r = classify_command(m.group(1) or m.group(2) or "", _depth + 1)
        if r:
            return r
    for seg in _SPLIT.split(cmd):
        seg = seg.strip().lstrip("({").strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        r = _check_tokens(toks, _depth)
        if r:
            return r
    return None


def is_yes(text: str) -> bool:
    """True only for an answer made of yes-words and nothing else: 'yes', 'yeah, go ahead', 'yes boss'. 'Yes, what is that?' is not a yes."""
    words = re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()
    if not words or _NO_WORDS.search(" ".join(words)):
        return False
    rest = " ".join(w for w in words if w not in _YES_WORDS and w not in _FILLER)
    has_yes = any(w in _YES_WORDS for w in words)
    if rest == "":
        return has_yes
    return rest in _YES_PHRASES


def _deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": f"blocked by the safety guard ({reason}); no clear spoken yes"}}


class Guard:
    def __init__(self, say=None, timeout_s: float = 30.0, enabled: bool = True):
        self.say, self.timeout_s, self.enabled = say, float(timeout_s), enabled
        self._fut = None
        self.asked_at = 0.0               # time.monotonic(): the same clock main.py stamps a talk-key press with
        self.pending_call = None          # the one tool call the open question is about; an answer resolves only that call
        self.last_answered = None

    @property
    def pending(self) -> bool:
        return self._fut is not None and not self._fut.done()

    def answer(self, text: str, spoke_from=None) -> bool:
        """Consume an utterance as the answer to the pending question. False = not an answer (it began before the question was asked).
        spoke_from must be time.monotonic(), as main.py's talk-key path passes it."""
        if not self.pending:
            return False
        if spoke_from is not None and spoke_from < self.asked_at:
            return False
        self.last_answered = self.pending_call
        self._fut.set_result(is_yes(text))
        return True

    def deny_pending(self):
        if self.pending:
            self._fut.set_result(False)

    async def pre_tool(self, input_data, tool_use_id, context):
        """PreToolUse hook (matcher Bash). Nothing may escape: a raising hook makes the CLI run the command. Any error => deny."""
        try:
            return await self._decide(input_data, tool_use_id)
        except BaseException as e:            # includes cancellation: deny rather than let the CLI fall through to running it
            try:
                log(f"[guard] denied: internal {type(e).__name__} tool_use_id={tool_use_id}")
            except BaseException:
                pass
            return _deny("guard error")

    async def _decide(self, input_data, tool_use_id):
        data = input_data or {}
        if not self.enabled or data.get("tool_name") != "Bash":
            return {}
        cmd = str((data.get("tool_input") or {}).get("command") or "")
        reason = classify_command(cmd)
        if not reason:
            return {}
        tag = f"tool_use_id={tool_use_id} cmd_sha={hashlib.sha256(cmd.encode()).hexdigest()[:12]}"
        if self.say is None or self.pending:
            log(f"[guard] denied ({'no voice' if self.say is None else 'another question is open'}): {reason} {tag}")
            return _deny(reason)
        allow = False
        self._fut = asyncio.get_running_loop().create_future()
        self.pending_call = tag
        try:
            self.asked_at = time.monotonic()
            log(f"[guard] asking: {reason} {tag}")
            self.say(f"Hold on, boss: {reason}. Go ahead?")
            try:
                allow = bool(await asyncio.wait_for(asyncio.shield(self._fut), self.timeout_s))
            except asyncio.TimeoutError:
                allow = False
        finally:
            if not self._fut.done():
                self._fut.set_result(False)
            self._fut = None
            self.pending_call = None
        log(f"[guard] {'allowed by spoken yes' if allow else 'denied'}: {reason} {tag}")
        if not allow:
            return _deny(reason)
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                       "permissionDecisionReason": "approved by voice"}}
