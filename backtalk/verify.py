"""Post-action verification: did the thing a tool call said it did actually
happen?

THE RULE THAT SHAPES THIS FILE. A tool's own "success" is not verification.
There are three levels of evidence:

  NONE       nothing was checked                          -> UNVERIFIED
  SELF_REPORT the tool returned without an error          -> UNVERIFIED
  READ_BACK  an independent, read-only look at the world  -> VERIFIED / VERIFY_FAILED

Only READ_BACK can produce VERIFIED (ledger.set_verification enforces it even
if a verifier tries otherwise). So an email that "sent", a TV that "powered
on" and a shell command that "exited 0" all stay UNVERIFIED until something
independent has looked.

WHAT EXISTS NOW: deterministic, local, read-only checks only:
  * Write / Edit / MultiEdit: read the file back and compare.
  * barehands board.sh add_card / present: read the board's own state.
There is deliberately NO remote/MCP verifier and NO second model client yet.

TASK RESULTS (check_task_summary). A background task's summary is the MODEL'S
sentence about what its tools returned. The router captures each tool result as
a bounded fact (evidence.py); this module compares the numbers in the summary
with the counts those facts hold, for the tool calls the task made itself
(found through parent_tool_use_id). The outcome is recorded on the task as
`result_verification`, at level TOOL_EVIDENCE: the claim is faithful to what the
tool returned. It is not a claim that the tool looked in the right place.

NEVER ON THE STREAM'S PATH. submit() is called from the router's dispatch and
returns immediately; the check runs in a worker thread with a timeout, and its
result is written to the ledger afterwards. A slow or crashing verifier cannot
delay or corrupt message routing.
"""
import asyncio
import hashlib
import json
import os
import re
import shlex
import time
import urllib.request

from backtalk import evidence as E
from backtalk import ledger as L
from backtalk.vlog import log

# Tools that only read: nothing changed, so there is nothing to verify.
_READ_TOOLS = {"Read", "Glob", "Grep", "WebSearch", "WebFetch", "TodoWrite",
               "ToolSearch", "AskUserQuestion", "ExitPlanMode", "EnterPlanMode",
               "BashOutput", "Skill", "ListAgents"}
_BOARD_RE = re.compile(r"barehands/bin/board\.sh\s+(['\"])(\{.*?\})\1", re.S)


class Outcome:
    __slots__ = ("state", "level", "evidence", "verifier")

    def __init__(self, state, level, evidence="", verifier=""):
        self.state, self.level = state, level
        self.evidence, self.verifier = evidence, verifier

    def __repr__(self):
        return f"Outcome({self.state},{self.level},{self.evidence!r})"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---- the local verifiers (sync, read-only, run in a worker thread) -----------
#
# WHAT A READ-BACK CAN PROVE. A verifier sees the world NOW, some time after the action returned. The action's own
# clock (when it was decided, when it returned) is the only causal evidence this module has, and it is used for
# exactly three things (F-07, Tier A):
#   * a mismatch in a file that was modified AFTER the action returned proves nothing about the action: UNVERIFIED;
#   * an Edit is VERIFIED only if the file was modified at or after the action began, because matching text that
#     predates the action is not evidence that the action happened;
#   * a board snapshot is evidence only if it was taken after the action returned (its own `ts`); an older or
#     undated one can neither confirm nor refute anything: UNVERIFIED, never VERIFY_FAILED.
# Nothing here knows the state BEFORE the action (that needs Tier B plumbing), so a card that already existed with the
# requested title, a card the board rejected, and `present`'s spotlight (no threshold has been measured) are
# still not distinguished from success.

_TIMES_KEY = "__action_times__"          # (began_at, returned_at), put into the verifier's input by the queue
_SLACK_S = 0.005                         # ledger timestamps are rounded to 1 ms; file clocks are not
_BOARD_TITLE_UNITS = 80                  # the board slices an added card's title to 80 UTF-16 units (holo.html)


def _times(row):
    """(began_at, returned_at) of an action from its own ledger row. began_at is the time the call was first seen."""
    h = row.get("history") or []
    began = h[0][0] if h and h[0] else None
    return began, row.get("returned_at")


def _action_times(inp):
    t = (inp or {}).get(_TIMES_KEY)
    return (t[0], t[1]) if isinstance(t, (tuple, list)) and len(t) == 2 else (None, None)


def _mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _changed_after(path, returned_at) -> bool:
    m = _mtime(path)
    return m is not None and returned_at is not None and m > returned_at + _SLACK_S


def verify_write(inp) -> Outcome:
    path = os.path.expanduser(str((inp or {}).get("file_path") or ""))
    want = (inp or {}).get("content")
    if not path or want is None:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "no path or content to compare", "file")
    if not os.path.isabs(path):
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "the path is not absolute; it is not resolved against any folder", "file")
    _, returned_at = _action_times(inp)
    try:
        with open(path, "rb") as f:
            got = f.read()
    except FileNotFoundError:
        return Outcome(L.V_FAILED, L.LEVEL_READBACK, "file does not exist after the write", "file")
    except OSError as e:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, f"could not read it back: {type(e).__name__}", "file")
    exp = str(want).encode("utf-8")
    if _sha(got) == _sha(exp):
        return Outcome(L.V_VERIFIED, L.LEVEL_READBACK,
                       f"file read back, {len(got)} bytes, hash matches what was sent", "file")
    if _changed_after(path, returned_at):
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "the file differs, but it was modified after this write returned; cannot say the write failed", "file")
    return Outcome(L.V_FAILED, L.LEVEL_READBACK,
                   f"file exists but differs ({len(got)} bytes vs {len(exp)} sent) and was not modified since", "file")


def _project(new, later):
    """What `new` becomes once the LATER edits of the same MultiEdit have run over it (a->b then b->c leaves c)."""
    out = str(new)
    for e in later:
        old, rep = e.get("old_string"), e.get("new_string")
        if old and rep is not None and old in out:
            out = out.replace(old, str(rep)) if e.get("replace_all") else out.replace(old, str(rep), 1)
    return out


def _edit_mismatch(text, edits, replace_all):
    """Why the file cannot be the result of these edits, or None. Each edit is judged as it stands AFTER the edits
    that follow it: its new text as the later edits left it must be there, and its old text must be gone unless
    its own result, or a later edit, put it back."""
    for i, e in enumerate(edits):
        new, old = e.get("new_string"), e.get("old_string")
        later = edits[i + 1:]
        shown = _project(new, later)
        if shown not in text:
            return "the new text is not in the file"
        if old and old not in shown and not any(old in str(x.get("new_string") or "") for x in later) \
                and old in text:
            return "the old text is still in the file"
    return None


def verify_edit(inp) -> Outcome:
    inp = inp or {}
    path = os.path.expanduser(str(inp.get("file_path") or ""))
    edits = inp.get("edits")
    if edits is None:
        edits = [{"old_string": inp.get("old_string"), "new_string": inp.get("new_string"),
                  "replace_all": inp.get("replace_all")}]
    if not path or not edits:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "no edit to compare", "file")
    if not os.path.isabs(path):
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "the path is not absolute; it is not resolved against any folder", "file")
    began_at, returned_at = _action_times(inp)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return Outcome(L.V_FAILED if isinstance(e, FileNotFoundError) else L.V_UNVERIFIED,
                       L.LEVEL_READBACK if isinstance(e, FileNotFoundError) else L.LEVEL_NONE,
                       f"could not read the file back: {type(e).__name__}", "file")
    if any(e.get("new_string") is None for e in edits):
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "edit has no new text", "file")
    why = _edit_mismatch(text, edits, bool(inp.get("replace_all")))
    if why:
        if _changed_after(path, returned_at):
            return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                           f"{why}, but the file was modified after this edit returned; cannot say it failed", "file")
        return Outcome(L.V_FAILED, L.LEVEL_READBACK, why, "file")
    m = _mtime(path)
    if began_at is None or m is None:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "the text matches, but there is no start time to show the file changed during this action", "file")
    if m + _SLACK_S < began_at:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "the text matches, but the file was not modified since this action began; it may predate it", "file")
    return Outcome(L.V_VERIFIED, L.LEVEL_READBACK,
                   f"file read back; {len(edits)} edit(s) present", "file")


def _board_state(port, timeout=1.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _board_title(title) -> str:
    """The title as the board stores it: JavaScript's String(title).slice(0, 80), which counts UTF-16 units."""
    raw = str(title).encode("utf-16-le", "surrogatepass")[:2 * _BOARD_TITLE_UNITS]
    return raw.decode("utf-16-le", "surrogatepass")


def _fresh(snapshot, returned_at) -> bool:
    ts = snapshot.get("ts") if isinstance(snapshot, dict) else None
    return (returned_at is not None and isinstance(ts, (int, float)) and not isinstance(ts, bool)
            and ts > returned_at)


def verify_board(inp, *, port=None, wait_s=3.0) -> Outcome:
    port = port or os.environ.get("HOLO_PORT", "4890")
    cmd = str((inp or {}).get("command") or "")
    if cmd.count("barehands/bin/board.sh") > 1:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "more than one board command in this call; only one can be read back", "board")
    m = _BOARD_RE.search(cmd)
    if not m:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "board command not recognised", "board")
    try:
        body = json.loads(m.group(2))
    except ValueError:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "board command JSON not readable", "board")
    act, title = body.get("a"), str(body.get("title") or "")
    if act not in ("add_card", "present") or not title:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_SELF,
                       f"board action {act!r} has no read-back check yet", "board")
    _, returned_at = _action_times(inp)
    if returned_at is None:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "no return time for this action, so no snapshot can be shown to be newer", "board")
    want = _board_title(title)
    deadline = time.monotonic() + wait_s
    saw_any = saw_fresh = False
    while True:
        st = _board_state(port)
        if isinstance(st, dict) and st.get("items") is not None:
            saw_any = True
            if _fresh(st, returned_at):
                saw_fresh = True
                if any(str(i.get("title") or "") == want for i in st["items"]):
                    return Outcome(L.V_VERIFIED, L.LEVEL_READBACK,
                                   f"a board snapshot taken after the action shows an item titled {want[:40]!r}"
                                   + (" (spotlight not checked)" if act == "present" else ""), "board")
        if time.monotonic() >= deadline:
            break
        time.sleep(0.4)          # the page pushes its snapshot about once a second
    if not saw_any:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "the board reported no state (page closed or server down)", "board")
    if not saw_fresh:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                       "no board snapshot newer than this action arrived (page hidden or closed?)", "board")
    return Outcome(L.V_FAILED, L.LEVEL_READBACK,
                   f"every board snapshot taken after the action lacks an item titled {want[:40]!r} after {wait_s:.0f}s", "board")


# ---- deciding what can be checked -------------------------------------------

# ---- Bash: only plain file commands (audit F-04) ----------------------------------------------------------------------------------
#
# A shell command's exit code is the tool's own word, not a read-back. For ONE plain `mkdir`, `touch`, `rm`, `mv` or `cp` on absolute paths,
# with no shell syntax at all, the result is a fact about the filesystem that can be looked at afterwards, read-only. Everything else
# (scripts, pipes, git, network, anything with a variable or a glob) is deliberately NOT handled and stays UNVERIFIED, as before.
# "Created or changed by this action" is judged from the inode CHANGE time (st_ctime): unlike mtime it cannot be preserved by `cp -p` or `touch -r`,
# so a path that merely EXISTED before the action is not mistaken for proof of it.

_SHELL_SYNTAX = re.compile(r"[;&|<>`$(){}*?\[\]!#\\~^%\n\r]")
_FILE_CMDS = {"mkdir": {"-p"}, "touch": set(), "rm": {"-f", "-r", "-R", "-rf", "-fr", "-Rf", "-fR"}, "mv": {"-f"},
              "cp": {"-r", "-R", "-p", "-f", "-a", "-rp", "-pr"}}
_MAX_PATHS = 8


def bash_file_op(command):
    """(program, flags, [absolute paths]) for a single plain file command, else None."""
    cmd = str(command or "").strip()
    if not cmd or _SHELL_SYNTAX.search(cmd):
        return None
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    if not parts or parts[0] not in _FILE_CMDS:
        return None
    prog, flags, paths, ended = parts[0], [], [], False
    for a in parts[1:]:
        if not ended and a == "--":
            ended = True
        elif not ended and a.startswith("-"):
            if a not in _FILE_CMDS[prog]:
                return None
            flags.append(a)
        else:
            paths.append(a)
    if not paths or len(paths) > _MAX_PATHS or any(not os.path.isabs(x) or "\0" in x or x.endswith("/")
                                                  or ".." in x.split("/") for x in paths):
        return None
    if prog in ("mv", "cp") and len(paths) < 2:
        return None
    return prog, flags, [os.path.normpath(x) for x in paths]


def _ctime(path):
    try:
        return os.lstat(path).st_ctime
    except OSError:
        return None


def verify_bash_files(inp) -> Outcome:
    if (inp or {}).get("run_in_background"):
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "ran in the background; it had not finished when it returned", "bash-files")
    op = bash_file_op((inp or {}).get("command"))
    if op is None:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "not a plain file command; no independent check exists", "bash-files")
    prog, flags, paths = op
    began, returned = _action_times(inp)
    if began is None:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "no action clock to tell this action's effect from older state", "bash-files")

    def fresh(p):                                         # changed at or after the action began (the same slack the file verifiers use)
        c = _ctime(p)
        return (c is not None and c >= began - _SLACK_S
                and (returned is None or c <= returned + _SLACK_S))     # a later touch by anything else is not this action

    problems, unproven = [], []
    if prog == "rm":
        still = [p for p in paths if os.path.lexists(p)]
        if still:
            return Outcome(L.V_FAILED, L.LEVEL_READBACK, f"still exists after rm: {len(still)} of {len(paths)} path(s)", "bash-files")
        if "-f" in flags or any(f.startswith("-") and "f" in f for f in flags):
            return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "rm -f succeeds on paths that never existed, so absence proves nothing", "bash-files")
        return Outcome(L.V_VERIFIED, L.LEVEL_READBACK, f"{len(paths)} path(s) do not exist now (a plain rm fails on a missing path)", "bash-files")
    if prog in ("mv", "cp"):
        srcs, dst = paths[:-1], paths[-1]
        for src in srcs:
            target = os.path.join(dst, os.path.basename(src)) if (os.path.isdir(dst) and (len(srcs) > 1 or not dst == src)) else dst
            if not os.path.lexists(target):
                problems.append(f"{os.path.basename(target)} is missing")
                continue
            if prog == "cp" and os.path.isdir(src):
                unproven.append("a copied directory's contents are not compared")
            elif prog == "mv":
                if os.path.lexists(src):
                    problems.append(f"{os.path.basename(src)} is still at the source")
                elif not fresh(target):
                    unproven.append(f"{os.path.basename(target)} exists but was not changed by this action")
            else:
                if not os.path.lexists(src):
                    unproven.append("the source is gone, so the copy cannot be compared")
                elif os.path.isfile(src) and os.path.isfile(target) and os.path.getsize(src) != os.path.getsize(target):
                    if _changed_after(src, returned):
                        unproven.append(f"{os.path.basename(src)} changed after the copy returned")
                    else:
                        problems.append(f"{os.path.basename(target)} differs in size from its source")
                elif not fresh(target):
                    unproven.append(f"{os.path.basename(target)} exists but was not changed by this action")
    else:                                                  # mkdir, touch
        for p in paths:
            if prog == "mkdir" and not os.path.isdir(p):
                problems.append(f"{os.path.basename(p)} is not a directory after mkdir")
            elif prog == "touch" and not os.path.lexists(p):
                problems.append(f"{os.path.basename(p)} does not exist after touch")
            elif not fresh(p):
                unproven.append(f"{os.path.basename(p)} exists but was not changed by this action")
    if problems:
        return Outcome(L.V_FAILED, L.LEVEL_READBACK, "; ".join(problems)[:280], "bash-files")
    if unproven:
        return Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, "; ".join(unproven)[:280], "bash-files")
    return Outcome(L.V_VERIFIED, L.LEVEL_READBACK, f"{prog}: {len(paths)} path(s) read back and changed by this action", "bash-files")


def plan(row, tool_input):
    """(outcome_now, verifier_fn). Exactly one is not None."""
    tool = row.get("tool") or ""
    st = row.get("state")
    if st == L.A_DENIED:
        return Outcome(L.V_NOT_REQUIRED, L.LEVEL_NONE, "denied; nothing ran", "plan"), None
    if st == L.A_ERROR:
        return Outcome(L.V_NOT_REQUIRED, L.LEVEL_NONE, "the tool returned an error; the action failed", "plan"), None
    if tool in _READ_TOOLS:
        return Outcome(L.V_NOT_REQUIRED, L.LEVEL_NONE, "read-only action", "plan"), None
    if tool == "Write":
        return None, verify_write
    if tool in ("Edit", "MultiEdit"):
        return None, verify_edit
    if tool == "Bash" and "barehands/bin/board.sh" in str((tool_input or {}).get("command") or ""):
        return None, verify_board
    if tool == "Bash" and bash_file_op((tool_input or {}).get("command")) is not None:
        return None, verify_bash_files
    # nothing independent to look at: the tool's own return is all we have
    return Outcome(L.V_UNVERIFIED, L.LEVEL_SELF,
                   f"{tool or 'tool'} returned without an error; no independent check exists yet",
                   "plan"), None


# ---- a background task's reported numbers ----------------------------------------

def verify_task_result(ledger, task_id):
    """Check the claims recorded on a task against the count facts of its own
    tool calls, and record `result_verification`. Synchronous and in-memory
    only: it reads rows the router already holds and appends one ledger line.
    Returns the Outcome, or None for an unknown task."""
    row = ledger.tasks.get(task_id)
    if row is None:
        return None
    prev = row.get("result_verification") or {}
    if prev.get("state") in (L.V_VERIFIED, L.V_FAILED):
        # A recorded verdict is final. Late evidence and a late or duplicate
        # summary can move a task OUT of "could not check", never overwrite a
        # verdict that was reached (the claims themselves never change).
        return Outcome(prev["state"], prev.get("level", L.LEVEL_NONE),
                       prev.get("evidence", ""), prev.get("verifier", "task-claims"))
    claims = row.get("claims") or []
    facts = []
    if claims and row.get("tool_use_id"):
        for a in ledger.child_actions(row["tool_use_id"]):
            if a.get("state") != L.A_OK:
                continue
            f = E.fact_of(a.get("evidence"))
            if f is not None:
                facts.append({"n": f[0], "group": f[1], "action": a["id"]})
    res = E.check_claims(claims, facts)
    action_ids = res["actions"]
    if not claims:
        out = Outcome(L.V_NOT_REQUIRED, L.LEVEL_NONE, "no number in the task's summary", "task-claims")
    elif not facts:
        out = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                      "the summary states numbers but none of the task's own tool results "
                      "gives a count to check them against", "task-claims")
    elif res["failed"]:
        out = Outcome(L.V_FAILED, L.LEVEL_TOOL_EVIDENCE,
                      f"summary says {sorted({c['n'] for c in res['failed']})}; the task's tool "
                      f"output counted {res['facts']}", "task-claims")
    elif res["ambiguous"]:
        counted = sorted({f["n"] for f in facts if f["group"] in {c["group"] for c in res["ambiguous"]}})
        out = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                      f"the task's own tool results disagree (they counted {counted}), so the summary's "
                      f"{sorted({c['n'] for c in res['ambiguous']})} cannot be attributed to one of them", "task-claims")
    elif res["matched"]:
        note = ""
        if res["breakdown_unchecked"]:
            note = (f"; breakdown values {sorted({c['n'] for c in res['breakdown_unchecked']})} "
                    "have no supporting evidence (unchecked, not a mismatch)")
        elif res["breakdown_matched"]:
            note = f"; breakdown {sorted({c['n'] for c in res['breakdown_matched']})} is supported by the tool output"
        out = Outcome(L.V_VERIFIED, L.LEVEL_TOOL_EVIDENCE,
                      f"summary count {sorted({c['n'] for c in res['matched']})} equals the task's "
                      f"tool output count {res['facts']}{note}", "task-claims")
    else:
        out = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                      "none of the numbers in the summary is about something the task's tool "
                      "output counts", "task-claims")
    # `facts` is only what bears on the claims that were compared; everything
    # else the task's tools counted is kept apart in `facts_all` and never spoken
    extra = {"facts": res["facts"], "facts_all": res["facts_all"],
             "matched": [{"n": c["n"], "noun": c["noun"]} for c in res["matched"]][:3],
             "failed": sorted({c["n"] for c in res["failed"]})[:5],
             "breakdown_matched": sorted({c["n"] for c in res["breakdown_matched"]})[:8],
             "breakdown_unchecked": sorted({c["n"] for c in res["breakdown_unchecked"]})[:8],
             "ambiguous": sorted({c["n"] for c in res["ambiguous"]})[:5],
             "unchecked": len(res["unchecked"]), "evidence_actions": action_ids[:8]}
    if (prev.get("state"), prev.get("facts"), prev.get("matched"), prev.get("failed"), prev.get("ambiguous"),
            prev.get("facts_all")) == (out.state, extra["facts"], extra["matched"],
                                       extra["failed"], extra["ambiguous"], extra["facts_all"]):
        return out                                   # nothing new to record
    ledger.set_task_verification(task_id, out.state, out.level, out.evidence, out.verifier, **extra)
    log(f"[verify] task={task_id} result={out.state} facts={extra['facts']} failed={extra['failed']}")
    return out


def check_task_summary(ledger, task_id, summary):
    """Record the numbers in `summary` as the task's claims, then verify them.
    `summary` is the FULL text from the completion notice, before the ledger's
    short redacted copy is made. Never raises."""
    try:
        if task_id not in ledger.tasks:
            return None
        ledger.task_set(task_id, claims=E.extract_claims(summary))
        return verify_task_result(ledger, task_id)
    except Exception as e:
        log(f"[verify] task result check failed: {type(e).__name__}")
        return None


_HOUR_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|noon|midnight"
_MIN_WORDS = r"(?:oh[\s-]+)?(?:[a-z]+(?:[\s-][a-z]+)?)"
# a time said in words ("eleven at night", "five fifty PM", "half past six") is a clock reading, not a claim about tool output;
# digit times are already ignored by extract_claims. Used by the foreground check only.
_SPOKEN_TIME = re.compile(
    rf"\b(?:(?:half|quarter)\s+(?:past|to)\s+(?:{_HOUR_WORDS})|(?:{_HOUR_WORDS})(?:[\s-]+{_MIN_WORDS})?\s*"
    rf"(?:a\.?m\.?|p\.?m\.?|o'?clock|at night|in the (?:morning|afternoon|evening)|tonight))\b", re.I)


def shadow_check_foreground(ledger, turn_id, spoken_text, request_text=""):
    """G-03, option B (his choice 2026-10-25): LOG-ONLY check of the numbers Jarvis just SAID in a foreground turn. It never changes,
    holds or delays speech: it runs after the turn's last sentence was handed up, reads rows the router already holds, writes one
    ledger note and one log line, and never raises. Turns without a spoken number are not recorded.

    Each spoken number lands in exactly one bucket, strongest evidence first:
      counted       equals a count the turn's own tools proved (evidence.facts_of)
      contradicted  is about something a tool counted, and differs from that count
      in_output     appears among the numbers a tool result showed (evidence "nums": numbers only, never text)
      in_request    appears in what was sent to the model (his words plus injected context), e.g. "the five newest"
      unknown       not found, but some tool output was too long to list every number, so absence proves nothing
      nowhere       not found anywhere the turn could have read it: a signal (computed, recalled, or made up), not a verdict
    Returns the note's fields (for tests), or None."""
    try:
        claims = [c for c in E.extract_claims(_SPOKEN_TIME.sub(" ", spoken_text or "")) if isinstance(c.get("n"), int)]
        if not claims:
            return None
        acts = [a for a in list(ledger.actions.values()) if a.get("turn_id") == turn_id and a.get("state") == L.A_OK]
        facts, shown, partial = [], set(), False
        for a in acts:
            ev = a.get("evidence") if isinstance(a.get("evidence"), dict) else {}
            for n, g in E.facts_of(ev):
                facts.append({"n": n, "group": g, "action": a["id"]})
            shown.update(x for x in ev.get("nums") or [] if isinstance(x, int))
            partial = partial or bool(ev.get("nums_partial"))
        res = E.check_claims(claims, facts) if facts else None
        counted = {c["n"] for c in (res["matched"] + res["breakdown_matched"])} if res else set()
        failed = {c["n"] for c in res["failed"]} if res else set()
        asked = set(E.numbers_in(request_text or "")[0]) | {c["n"] for c in E.extract_claims(request_text or "")}
        buckets = {k: [] for k in ("counted", "contradicted", "in_output", "in_request", "unknown", "nowhere")}
        for n in sorted({c["n"] for c in claims}):
            if n in failed:
                k = "contradicted"
            elif n in counted:
                k = "counted"
            elif n in shown:
                k = "in_output"
            elif n in asked:
                k = "in_request"
            elif partial:
                k = "unknown"
            else:
                k = "nowhere"
            buckets[k].append(n)
        if buckets["contradicted"]:
            verdict = "CONTRADICTED"
        elif not acts:
            verdict = "no_tools"
        elif buckets["nowhere"]:
            verdict = "NOWHERE"
        elif buckets["unknown"]:
            verdict = "unknown"
        else:
            verdict = "accounted"
        fields = {"turn_id": turn_id, "verdict": verdict, "numbers": sum(len(v) for v in buckets.values()), "tools": len(acts),
                  "facts": len(facts), "counted_by_tools": sorted({f["n"] for f in facts})[:8], **{k: v[:6] for k, v in buckets.items()}}
        ledger.note("fg_claim_shadow", **fields)
        log(f"[verify-fg] turn={turn_id} {verdict} tools={len(acts)} "
            + " ".join(f"{k}={v[:6]}" for k, v in buckets.items() if v))
        return fields
    except Exception as e:
        try:
            log(f"[verify-fg] check skipped: {type(e).__name__}")
        except Exception:
            pass
        return None


class VerifyQueue:
    """Runs verifiers off the stream's path. submit() never blocks."""

    def __init__(self, ledger, *, timeout_s=6.0, max_pending=64):
        self.ledger = ledger
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self.pending = set()
        self.done_count = 0
        self.on_done = None            # callable(action_id) after a check finishes

    def submit(self, row, tool_input=None):
        aid = row.get("id")
        try:
            now, fn = plan(row, tool_input)
        except Exception as e:
            now, fn = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE, f"planner crashed: {type(e).__name__}", "plan"), None
        if now is not None:
            self.ledger.set_verification(aid, now.state, now.level, now.evidence, now.verifier)
            self._notify(aid)
            return None
        if len(self.pending) >= self.max_pending:
            self.ledger.set_verification(aid, L.V_UNVERIFIED, L.LEVEL_NONE,
                                         "verification queue full; not checked", "queue")
            self._notify(aid)
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.ledger.set_verification(aid, L.V_UNVERIFIED, L.LEVEL_NONE,
                                         "no event loop to run the check", "queue")
            self._notify(aid)
            return None
        t = loop.create_task(self._run(aid, fn, self._with_times(row, tool_input)))
        self.pending.add(t)
        t.add_done_callback(self.pending.discard)
        return t

    @staticmethod
    def _with_times(row, tool_input):
        """The verifier's input: the tool's own input plus the action's decision and return times. The key is set
        AFTER the copy, so nothing a model put in its tool input can stand in for the real times."""
        ti = dict(tool_input) if isinstance(tool_input, dict) else {}
        ti[_TIMES_KEY] = _times(row)
        return ti

    async def _run(self, aid, fn, tool_input):
        loop = asyncio.get_running_loop()
        try:
            out = await asyncio.wait_for(loop.run_in_executor(None, fn, tool_input),
                                         self.timeout_s)
        except asyncio.TimeoutError:
            out = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                          f"verifier timed out after {self.timeout_s:.0f}s", "queue")
        except Exception as e:
            out = Outcome(L.V_UNVERIFIED, L.LEVEL_NONE,
                          f"verifier crashed: {type(e).__name__}", "queue")
        self.ledger.set_verification(aid, out.state, out.level, out.evidence, out.verifier)
        self.done_count += 1
        log(f"[verify] action={aid} state={out.state} level={out.level} by={out.verifier}")
        self._notify(aid)

    def _notify(self, aid):
        """The check for `aid` is finished, whether it was settled at once or by a worker."""
        if self.on_done:
            try:
                self.on_done(aid)
            except Exception as e:
                log(f"[verify] on_done failed: {e!r}")
