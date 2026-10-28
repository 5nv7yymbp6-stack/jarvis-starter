"""Structured evidence from tool results, and the numeric claims a model makes
about them. Pure functions: no I/O, no ledger writes, nothing that can block the
stream.

WHY THIS EXISTS. On 2026-09-30 a background agent ran one `find` whose result
the router saw: 21 paths. Its summary said "20 daily note files". The ledger had
kept only the result's size and hash, so nothing could say the summary was
wrong, and the speech policy repeated it. The model's sentence about a result
is not the result. This module turns the result into a small, bounded fact
(a count, a single integer) that a claim can be checked against.

BOUNDED, NEVER A BODY. A fact holds counts, sizes and at most two head lines and
one tail line, each redacted and cut to 100 characters. The result text itself
is never stored.

PROVENANCE. A count is derived only from a command that COMPUTES one:
  * a strict list (find, ls -1, git ls-files, Glob), counted by its lines, or
  * a counting command: `... | wc -l`, `grep -c`, `rg -c`, or `echo ${#array[@]}`.
Every command that runs before it and prints (an `echo`, a `pwd`, an `ls`) would
add lines to the result, so each such segment must be SILENT (cd, setopt,
export NAME=..., an assignment...) or be piped away. A literal the model typed
(`echo 20`, `printf 20`) is not a measurement and never becomes a fact; neither
does a lone integer from `cat`, `date`, or any non-Bash tool. Truncated output
yields no count.

PRIMARY OR BREAKDOWN. A summary's count claims are not all the same kind of
statement. "21 files (8 + 13)" makes ONE claim about the whole (21) and two about
its parts. Each count claim gets a `role` from its position alone, never from its
size: a number inside parentheses, chained to another number by `+`/`,`/`and`, or
introduced by "of which"/"including" is a BREAKDOWN; the rest are PRIMARY. A
primary claim is checked against the tool's count. A breakdown claim is checked
only when the evidence itself breaks the total down (facts that sum to it);
without that it is unchecked, which is never a mismatch.

FAIL CLOSED ON CLAIMS. Every number in a model's summary is accounted for:
a count about something (checked against a fact), or a FREE number (unchecked,
so the task is UNVERIFIED and the number is not repeated). Nothing is dropped
silently: spelled-out numbers, `one`, `zero`, `250k`, `20x`, `5th`, `$1950`
and a bare year-sized amount are all free numbers. Only dates, times, and a year
in clear date context are ignored. An off-by-one is never forgiven.
"""
import collections
import re
import shlex

KIND_LIST, KIND_INTEGER = "list", "integer"
KIND_TEXT, KIND_ERROR, KIND_EMPTY = "text", "error", "empty"
GROUP_ITEM, GROUP_DIR = "item", "dir"

_CWD_LINE = "Shell cwd was reset to"
_MAX_SCAN = 400_000                 # never walk an unbounded string
_TRUNC_AT = 29_000                  # the CLI cuts a Bash result at 30000 chars
_SAMPLE = 100

# ---- the result ---------------------------------------------------------------


def result_text(content) -> str:
    """The text of a tool result, whether it arrived as a string or as blocks."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text", "")))
            else:
                parts.append(str(getattr(b, "text", b)))
        content = "\n".join(parts)
    return str(content)[:_MAX_SCAN]


def _lines(text: str) -> list:
    return [ln for ln in text.splitlines()
            if ln.strip() and not ln.startswith(_CWD_LINE)]


def _looks_truncated(text: str, lines: list) -> bool:
    if len(text) >= _TRUNC_AT:
        return True
    for ln in lines:
        if "truncat" in ln.lower() and not ln.startswith(("/", ".", "~")):
            return True
    return False


_LIST_FIND_BAD = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-ls",
                            "-printf", "-print0", "-fprint", "-fprint0", "-fprintf",
                            "-fls"})
_GIT_LSFILES_OK = frozenset({"-c", "--cached", "-o", "--others", "--exclude-standard",
                             "-m", "--modified", "-d", "--deleted"})
_DEVNULL = re.compile(r"\d?>\s*/dev/null|&>\s*/dev/null")
_DEVNULL_TOKEN = re.compile(r"^(?:\d?>|&>)\s*/dev/null$")
_ASSIGN_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
_SILENT_CMDS = frozenset({"cd", "true", "false", ":", "sleep", "test", "[", "unset"})
# producers whose output is whatever the model typed, so a count of it measures nothing
_LITERAL_PRODUCERS = frozenset({"echo", "printf", "print", "seq", "yes", "jot"})
_LEN_EXPANSION = re.compile(r"^\$\{#[A-Za-z_][A-Za-z0-9_]*(?:\[[@*]\])?\}$")


def _split_ops(cmd):
    """[(segment, operator_after)] on UNQUOTED && || ; newline |. The operator
    after a segment says whether its stdout is printed (&& || ; newline) or
    piped onward (|). The caller has already rejected unbalanced quotes and
    background/grouping constructs."""
    out, cur, i, n = [], [], 0, len(cmd)
    in_s = in_d = False
    while i < n:
        c = cmd[i]
        if in_s:
            cur.append(c)
            in_s = c != "'"
            i += 1
            continue
        if in_d:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            in_d = c != '"'
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur += [c, cmd[i + 1]]
            i += 2
            continue
        if c in "'\"":
            in_s, in_d = c == "'", c == '"'
            cur.append(c)
            i += 1
            continue
        two = cmd[i:i + 2]
        op = two if two in ("&&", "||") else (c if c in ";\n|" else None)
        if op:
            out.append(("".join(cur).strip(), "|" if op == "|" else op))
            cur = []
            i += len(op)
            continue
        cur.append(c)
        i += 1
    out.append(("".join(cur).strip(), None))
    return [(t, op) for t, op in out if t]


def _tokens(seg):
    return [t for t in shlex.split(seg) if not _DEVNULL_TOKEN.match(t)]


def _command(toks):
    """(name, args) with leading NAME=value assignments removed (an array
    assignment `files=(a b)` spans several tokens)."""
    while toks and _ASSIGN_TOKEN.match(toks[0]):
        first = toks[0]
        toks = toks[1:]
        if "=(" in first and not first.endswith(")"):
            while toks and not toks[0].endswith(")"):
                toks = toks[1:]
            toks = toks[1:]                   # the token holding the closing )
    if not toks:
        return "", []
    return toks[0].rsplit("/", 1)[-1], toks[1:]


def _is_silent(seg) -> bool:
    """True if this segment prints nothing on stdout."""
    name, args = _command(_tokens(seg))
    if not name:
        return True                               # nothing but assignments
    if name in _SILENT_CMDS:
        return not (name == "cd" and args[:1] == ["-"])      # `cd -` prints the directory
    if name in ("setopt", "unsetopt"):
        return bool(args)                         # bare `setopt` lists the options
    if name == "shopt":
        flags = [t for t in args if t.startswith("-")]
        return bool([t for t in args if not t.startswith("-")]) and all(
            f in ("-s", "-u", "-q") for f in flags)
    if name == "export":
        return bool(args) and not any(t.startswith("-") for t in args)
    return False


def _is_sort(seg) -> bool:
    name, args = _command(_tokens(seg))
    return name == "sort" and not any(t.startswith("-o") or t == "--output" for t in args)


def _list_group(name, args):
    """GROUP_* if this command prints one entry per line by construction."""
    if name == "find":
        if any(t in _LIST_FIND_BAD for t in args):
            return None
        for k, t in enumerate(args[:-1]):
            if t == "-type" and args[k + 1] == "d":
                return GROUP_DIR
        return GROUP_ITEM
    if name == "ls":
        flags = [t for t in args if t.startswith("-")]
        pos = [t for t in args if not t.startswith("-")]
        if any(f.startswith("--") or not set(f[1:]) <= set("1A") for f in flags):
            return None
        if len(pos) > 1 or any(c in p for p in pos for c in "*?["):
            return None                       # several directories print headers
        return GROUP_ITEM
    if name == "git" and args[:1] == ["ls-files"]:
        flags = [t for t in args[1:] if t.startswith("-")]
        return GROUP_ITEM if all(f in _GIT_LSFILES_OK for f in flags) else None
    return None


def _bash_provenance(cmd):
    """("list", GROUP) when the command's stdout is one entry per line by
    construction; ("count", GROUP) when it computes a single count; else None.
    Either way every segment that runs before the counting command and whose
    stdout is printed must be silent, or its lines would be counted too."""
    try:
        from backtalk.shellscan import _scan_bash
        cmd = cmd or ""
        segs_ok, unsafe = _scan_bash(cmd)
        if unsafe is not None or not segs_ok or "<<" in cmd \
                or any(x in cmd for x in ("$(", "`", "<(")):
            return None
        if ">" in _DEVNULL.sub("", cmd):
            return None                       # stdout redirected, or stderr merged into it
        segs = _split_ops(cmd)
        if not segs:
            return None
        i = len(segs) - 1
        while i >= 1 and segs[i - 1][1] == "|" and _is_sort(segs[i][0]):
            i -= 1                            # `| sort` keeps one entry per line
        if _is_sort(segs[i][0]) and i == 0:
            return None
        for j in range(i):                    # what runs BEFORE the counting command
            if segs[j][1] != "|" and not _is_silent(segs[j][0]):
                return None
        name, args = _command(shlex.split(segs[i][0]))
        listed = _list_group(name, args)
        if listed is not None:
            return "list", listed
        if name == "wc":
            if args != ["-l"] or i == 0 or segs[i - 1][1] != "|":
                return None
            k = i - 1                         # walk back to the SOURCE of the pipeline
            while k >= 1 and segs[k - 1][1] == "|":
                k -= 1
            stages = [_command(shlex.split(segs[x][0])) for x in range(k, i)]
            groups = [g for g in (_list_group(n, a) for n, a in stages) if g]
            if stages[0][0] in _LITERAL_PRODUCERS and not groups:
                return None                   # counting lines of text the model typed
            return "count", GROUP_DIR if GROUP_DIR in groups else GROUP_ITEM
        if name in ("grep", "egrep", "fgrep", "rg") and any(
                a in ("-c", "--count") for a in args):
            return "count", GROUP_ITEM
        if name == "echo" and len(args) == 1 and _LEN_EXPANSION.match(args[0]):
            return "count", GROUP_ITEM        # ${#files[@]}: a length, never a literal
    except Exception:
        return None
    return None


KIND_COUNTS = "counts"
_NUMS_MAX = 256
# phone-shaped runs (555-123-4567, (555) 123 4567, +1 555 123 4567) are removed before numbers are collected, and any digit run
# longer than 6 is skipped whole: the ledger may keep amounts and counts, never something that identifies a person
_PHONE = re.compile(r"(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_NUM = re.compile(r"(?<![\d,.])(\d{1,3}(?:,\d{3})+|\d+)(?![\d,]*\d)")


def numbers_in(text):
    """(sorted unique integers of at most 6 digits that appear in `text`, partial) -- the numbers only, never the text around them.
    `partial` is True when more than _NUMS_MAX distinct numbers were present, so an absent number proves nothing."""
    seen = set()
    for m in _NUM.finditer(_PHONE.sub(" ", text or "")):
        raw = m.group(1).replace(",", "")
        if len(raw) <= 6:
            seen.add(int(raw))
    nums = sorted(seen)
    return nums[:_NUMS_MAX], len(nums) > _NUMS_MAX
_ECHO_FLAG = re.compile(r"^-[neE]+$")


def _echo_literal(seg):
    """The single line a plain `echo <literal>` prints ('' for a bare `echo`: a blank line, which _lines drops), or None when
    that output is not knowable from the text: an echo flag (-n -e -E), or any escape, expansion or substitution character."""
    toks = _tokens(seg)
    name, args = _command(toks)
    if name != "echo" or len(toks) - len(args) != 1 or (args and _ECHO_FLAG.match(args[0])):
        return None
    if any(c in seg for c in "\\$`*?[~{"):
        return None                                    # escapes, expansions, globs, tilde and braces change what is printed
    return " ".join(args).strip()


def _bash_sections(cmd):
    """A compound command read statement by statement (G-03, 2026-10-25), for output whose line layout can be PROVEN:
    [(line_index, group, label_before)] for each counting statement, or None.

    Statements are split on unquoted ; && newline (a `||` makes what runs unknowable: None). In order, each statement must be
    silent (0 lines), a plain `echo <literal>` (exactly the line it prints), or a counting pipeline as _bash_provenance defines it
    (exactly one integer line). The walk stops at the first statement whose output size is unknown: everything after it is
    simply not counted. A count is used only at line 0 or right after a literal label, which the caller checks line for line,
    so an unexpected line (stderr, a failed `&&`) shifts the layout into a mismatch and the counts are dropped, never misread."""
    try:
        from backtalk.shellscan import _scan_bash
        cmd = cmd or ""
        segs_ok, unsafe = _scan_bash(cmd)
        if unsafe is not None or not segs_ok or "<<" in cmd or any(x in cmd for x in ("$(", "`", "<(")):
            return None
        if ">" in _DEVNULL.sub("", cmd):
            return None
        segs = _split_ops(cmd)
        if any(op == "||" for _, op in segs):
            return None
        stmts, cur = [], []
        for seg, op in segs:
            cur.append(seg)
            if op != "|":
                stmts.append(cur)
                cur = []
        if cur:
            stmts.append(cur)
        plan, idx, label = [], 0, None
        for st in stmts:
            text = " | ".join(st)
            if len(st) == 1 and _is_silent(st[0]):
                continue
            lit = _echo_literal(st[0]) if len(st) == 1 else None
            if lit is not None:
                if lit:
                    plan.append((idx, "label", lit))
                    label, idx = lit, idx + 1
                continue
            prov = _bash_provenance(text)
            if prov and prov[0] == "count":
                plan.append((idx, prov[1], label))
                label, idx = None, idx + 1
                continue
            break
        return plan if any(g != "label" for _, g, _ in plan) else None
    except Exception:
        return None


def _section_counts(plan, lines):
    """[{n, group}] read from `lines` by a _bash_sections plan; stops at the first line that does not fit the proven layout."""
    out = []
    for idx, group, text in plan:
        if idx >= len(lines):
            break
        line = lines[idx].strip()
        if group == "label":
            if line != text:
                break                                  # layout shifted: nothing from here on can be attributed
            continue
        if not re.fullmatch(r"\d{1,12}", line):
            break
        if idx != 0 and (text is None or lines[idx - 1].strip() != text):
            break                                      # a count must sit at line 0 or right under its own label
        out.append({"n": int(line), "group": group})
    return out


def _sample(lines, redact):
    cut = lambda s: redact(s, _SAMPLE)       # noqa: E731
    return [cut(s) for s in lines[:2]], [cut(s) for s in lines[-1:]] if len(lines) > 2 else []


def extract_evidence(tool, tool_input, content, is_error=False) -> dict:
    """The bounded fact for one tool result. Never raises."""
    try:
        from backtalk.ledger import redact
        text = result_text(content)
        ev = {"bytes": len(text.encode("utf-8", "replace"))}
        if is_error:
            ev.update(kind=KIND_ERROR, lines=0)
            return ev
        lines = _lines(text)
        ev["lines"] = len(lines)
        ev["nums"], partial = numbers_in(text)              # G-03 option B: which numbers the tool showed (numbers only)
        if partial:
            ev["nums_partial"] = True
        if _looks_truncated(text, lines):
            ev.update(kind=KIND_TEXT, truncated=True, nums_partial=True)
            return ev
        ti = tool_input if isinstance(tool_input, dict) else {}
        prov = None
        if tool == "Glob":
            prov = ("list", GROUP_ITEM)
            if len(lines) == 1 and lines[0].strip().lower().startswith("no files found"):
                lines, ev["lines"] = [], 0
        elif tool == "Bash":
            prov = _bash_provenance(ti.get("command"))
        if prov and prov[0] == "list":
            ev.update(kind=KIND_LIST, group=prov[1])
            ev["head"], ev["tail"] = _sample(lines, redact)
            return ev
        if prov and prov[0] == "count" and len(lines) == 1 \
                and re.fullmatch(r"\s*\d{1,12}\s*", lines[0]):
            ev.update(kind=KIND_INTEGER, int=int(lines[0]), group=prov[1])
            return ev
        if tool == "Bash" and prov is None and lines:
            plan = _bash_sections(ti.get("command"))
            counts = _section_counts(plan, lines) if plan else []
            if counts:
                ev.update(kind=KIND_COUNTS, counts=counts[:8])
                return ev
        ev["kind"] = KIND_EMPTY if not lines else KIND_TEXT
        return ev
    except Exception:
        return {"kind": KIND_TEXT, "bytes": -1, "lines": 0}


def fact_of(evidence):
    """(count, group) if this evidence yields a trustworthy count, else None."""
    if not isinstance(evidence, dict) or evidence.get("truncated"):
        return None
    if evidence.get("kind") == KIND_INTEGER and isinstance(evidence.get("int"), int):
        return evidence["int"], evidence.get("group") or GROUP_ITEM
    if evidence.get("kind") == KIND_LIST and isinstance(evidence.get("lines"), int):
        return evidence["lines"], evidence.get("group") or GROUP_ITEM
    return None


def facts_of(evidence):
    """Every trustworthy (count, group) in this evidence: fact_of's single count, or each proven count of a compound command
    (KIND_COUNTS). fact_of itself is unchanged, so the background-summary verifier keeps exactly its old inputs."""
    one = fact_of(evidence)
    if one is not None:
        return [one]
    if isinstance(evidence, dict) and evidence.get("kind") == KIND_COUNTS and not evidence.get("truncated"):
        return [(c["n"], c["group"]) for c in evidence.get("counts") or []
                if isinstance(c, dict) and isinstance(c.get("n"), int) and c.get("group")]
    return []


# ---- the claims ---------------------------------------------------------------
#
# Every number in a summary is ACCOUNTED FOR. extract_claims returns, in this
# order: count claims (a number joined to a countable noun), lead-in claims
# ("found 20"), then FREE numbers (anything else numeric). A free number has
# group None: it cannot be checked, so it makes the task UNVERIFIED and is never
# repeated. Dates, times and a year in clear date context are the only things
# ignored.

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_WORD_ALT = "|".join(sorted([*_UNITS, *_TEENS, *_TENS, *_SCALES, "dozen"], key=len, reverse=True))
# a run of number words: "twenty five", "twenty-five", "one hundred and twenty",
# "a hundred", "two dozen". An odd run ("five twenty") is a free number, not a guess.
_WORD_RUN = (rf"(?:a\s+(?=(?:hundred|thousand|million|dozen)\b))?\b(?:{_WORD_ALT})\b"
             rf"(?:[\s-]+(?:and\s+)?(?:{_WORD_ALT})\b)*")
_VAGUE = r"\b(?:dozens|hundreds|thousands|millions|couple|several|handful)\b"
_SUFFIX = r"(?:k|m|b|x|st|nd|rd|th|kb|mb|gb|tb|ms|s|h|d)"
_TOKEN_RE = re.compile(
    rf"(?<![\w.])(?P<digit>\d[\d,]*(?:\.\d+)?)(?P<suf>{_SUFFIX}|%)?(?![\w])(?!\.[A-Za-z])"
    rf"|(?P<vague>{_VAGUE})"
    rf"|(?P<words>{_WORD_RUN})", re.I)

_ITEM_NOUNS = ("files", "file", "notes", "note", "items", "item", "entries", "entry",
               "results", "result", "matches", "match", "lines", "line",
               "documents", "document", "records", "record", "rows", "row",
               "pages", "page")
_DIR_NOUNS = ("folders", "folder", "directories", "directory", "dirs", "dir")
_OTHER_NOUNS = ("emails", "email", "messages", "message", "tasks", "task",
                "contacts", "leads", "properties", "deals", "calls", "texts")
_NOUN = "|".join(sorted([*_ITEM_NOUNS, *_DIR_NOUNS, *_OTHER_NOUNS], key=len, reverse=True))
# words that mean the number is NOT directly counting the noun after them
# ("7 in the folder", "5 of the files"): a claim needs the number and its noun
# joined by at most two descriptive words
_FUNCTION_WORDS = ("(?:in|of|the|a|an|to|from|on|at|for|with|and|or|that|which|"
                   "are|were|is|was|by|as|than|more|less)")
_NOUN_AFTER = re.compile(
    rf"(?P<words>(?:\s+(?!{_FUNCTION_WORDS}\b)[A-Za-z][A-Za-z'-]*){{0,2}})\s+(?P<noun>{_NOUN})\b", re.I)
_NO_NOUN = re.compile(
    rf"\bno(?P<words>(?:\s+(?!{_FUNCTION_WORDS}\b)[A-Za-z][A-Za-z'-]*){{0,2}})\s+(?P<noun>{_NOUN})\b", re.I)
_LEAD_BEFORE = re.compile(
    r"\b(?:found|counted|count(?:\s+is|:)?|total(?:\s+of|:)?|there\s+(?:are|were|is))\s*$", re.I)
# "about 20", "~20", "over 20", "at least 20": a rounded or bounded number cannot be
# checked for equality, so it is unchecked, never called wrong
_APPROX_LEAD = re.compile(
    r"(?:\b(?:about|around|roughly|approximately|approx\.?|nearly|almost|over|under|above|"
    r"below|some|(?:more|less|fewer|greater)\s+than|at\s+(?:least|most)|up\s+to)\s+|[~≈]\s*)$",
    re.I)

_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
# months are case-insensitive except "may", which is a month only as "May"
_MONTH_NAME = rf"(?:(?i:{_MONTHS})|May)"
_MONTH_LEAD = re.compile(rf"\b{_MONTH_NAME}\.?\s+$")
_YEAR_DESIGNATOR_LEAD = re.compile(r"\b(?:year|fiscal|fy|q[1-4])\.?\s+$", re.I)
_DATE_PREP_LEAD = re.compile(r"\b(?:in|during|since|until|through|by|from|before|after)\s+$", re.I)
_YEAR_RANGE_LEAD = re.compile(r"\b(?:from|between)\s+(?:19|20)\d{2}\s+(?:to|and|through|-)\s+$", re.I)
# dates and clock times are not numbers a task can be wrong about
_DATE_MASKS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\b\d{1,4}/\d{1,2}/\d{1,4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]m)?\b", re.I),
    re.compile(r"\b\d{1,2}\s?[ap]m\b", re.I),
    re.compile(rf"\b{_MONTH_NAME}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\d)"),
)


def _words_to_int(run: str):
    """The value of a run of number words, or None for a run that is not a
    well-formed number (so it stays an unchecked free number)."""
    toks = [t for t in re.split(r"[\s-]+", run.lower()) if t and t not in ("a", "and")]
    if toks == ["zero"]:
        return 0
    total = cur = 0
    last = None
    for t in toks:
        if t in _UNITS and t != "zero":
            if last not in (None, "tens", "hundred", "scale"):
                return None
            cur += _UNITS[t]
            last = "unit"
        elif t in _TEENS:
            if last not in (None, "hundred", "scale"):
                return None
            cur += _TEENS[t]
            last = "teen"
        elif t in _TENS:
            if last not in (None, "hundred", "scale"):
                return None
            cur += _TENS[t]
            last = "tens"
        elif t == "hundred":
            if last not in (None, "unit") or cur >= 10:
                return None
            cur = (cur or 1) * 100
            last = "hundred"
        elif t in ("thousand", "million"):
            if last == "scale":
                return None
            total += (cur or 1) * _SCALES[t]
            cur, last = 0, "scale"
        elif t == "dozen":
            if last not in (None, "unit", "teen"):
                return None
            cur = (cur or 1) * 12
            last = "dozen"
        else:
            return None                                   # "zero" inside a longer run
    return total + cur


def _group_of(noun: str):
    n = noun.lower()
    if n in _ITEM_NOUNS:
        return GROUP_ITEM
    if n in _DIR_NOUNS:
        return GROUP_DIR
    return "other"


def _mask_dates(s: str) -> str:
    for pat in _DATE_MASKS:
        s = pat.sub(lambda mm: " " * (mm.end() - mm.start()), s)
    return s


def _year_to_ignore(before: str, n, noun_m) -> bool:
    """A year-sized number is ignored only in CLEAR date/year context."""
    if _MONTH_LEAD.search(before) or _YEAR_DESIGNATOR_LEAD.search(before):
        return True                                       # "September 2026 notes", "FY 2026 files"
    if noun_m is None:
        return bool(_DATE_PREP_LEAD.search(before) or _YEAR_RANGE_LEAD.search(before))   # "built in 1985"; "in 2026 files" counts
    return bool(noun_m.group("words").split()) and noun_m.group("noun").lower() in _DIR_NOUNS


_BREAKDOWN_LEAD = re.compile(
    r"\b(?:of\s+which|including|incl\.?|comprising|made\s+up\s+of|consisting\s+of|"
    r"split\s+(?:into|as)|broken\s+down\s+(?:as|into)|breakdown(?:\s+is)?)\s*:?\s*$", re.I)
_LINK_GAP = re.compile(r"\s*(?:,\s*and|,|;|\+|&|\band\b)\s*", re.I)


def _structure_roles(masked, entries):
    """Give each entry a `role` from structure alone: inside parentheses, chained
    to a neighbouring number by a list separator, or introduced by "of which" =
    breakdown; otherwise primary. Bare numbers in a breakdown inherit the group
    of the nearest primary claim. Mutates the entries' claim dicts."""
    seq = sorted((e for e in entries if e["span"]), key=lambda e: e["start"])
    for i, e in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        e["chained"] = e.get("chained", False)
        if nxt is not None and _LINK_GAP.fullmatch(masked[e["ext"]:nxt["start"]]):
            e["chained"] = nxt["chained"] = True
    for e in seq:
        before = masked[:e["start"]]
        shaped = (before.count("(") > before.count(")") or e["chained"]
                  or bool(_BREAKDOWN_LEAD.search(before[-48:])))
        e["shaped"] = shaped
        c = e["claim"]
        if c["group"] in (GROUP_ITEM, GROUP_DIR, "other"):
            c["role"] = "breakdown" if shaped else "primary"
        else:
            c["role"] = "breakdown" if (shaped and e["inherit"]) else "unrelated"
    for g in (GROUP_ITEM, GROUP_DIR):                     # a lone parenthesised claim is still a claim
        prim = [e for e in seq if e["claim"]["group"] == g and e["claim"]["role"] == "primary"]
        part = [e for e in seq if e["claim"]["group"] == g and e["claim"]["role"] == "breakdown"]
        if not prim and len(part) == 1:
            part[0]["claim"]["role"] = "primary"
    primaries = [e for e in seq if e["claim"]["role"] == "primary"
                 and e["claim"]["group"] in (GROUP_ITEM, GROUP_DIR)]
    for e in seq:
        c = e["claim"]
        if c["group"] is None and c["role"] == "breakdown":
            near = ([p for p in primaries if p["start"] < e["start"]][-1:]
                    or [p for p in primaries if p["start"] > e["start"]][:1])
            if near:
                c["group"] = near[0]["claim"]["group"]    # "8 + 12 + 1" after "21 files": items
            else:
                c["role"] = "unrelated"                   # nothing to be a part OF: unchecked


def extract_claims(text) -> list:
    """Every number in a model's summary, each as {n, noun, group, role}. group
    None means it cannot be checked (and must not be repeated as if it had been);
    role is "primary", "breakdown" or "unrelated" (see the module docstring)."""
    s = str(text or "")[:4000]
    masked = _mask_dates(s)
    entries = []                                          # in discovery order within each kind
    for m in _NO_NOUN.finditer(masked):                   # "no files found": zero, unchecked
        entries.append({"kind": "free", "span": None, "inherit": False, "start": m.start(),
                        "ext": m.end(), "claim": {"n": 0, "noun": " ".join(
                            (m.group("words") + " " + m.group("noun")).split())[:40],
                            "group": None, "role": "unrelated"}})
    for m in _TOKEN_RE.finditer(masked):
        before = masked[max(0, m.start() - 48):m.start()]
        after = masked[m.end():m.end() + 80]
        free_only = False
        if m.group("vague"):
            n, free_only = None, True
        elif m.group("digit"):
            raw = m.group("digit")
            n = None if "." in raw else int(raw.replace(",", ""))
            if m.group("suf"):                            # 250k, 20x, 5th, 3gb, 50%
                free_only = True
                mult = {"k": 1000, "m": 10 ** 6, "b": 10 ** 9}.get(m.group("suf").lower())
                if mult and n is not None:
                    n *= mult
        else:
            n = _words_to_int(m.group("words"))
        noun_m = _NOUN_AFTER.match(after)
        if (m.group("digit") and not m.group("suf") and n is not None and 1900 <= n <= 2100
                and "," not in m.group("digit") and len(m.group("digit")) == 4
                and _year_to_ignore(before, n, noun_m)):
            continue
        approx = bool(_APPROX_LEAD.search(before))
        e = {"span": m.span(), "start": m.start(), "ext": m.end(), "inherit": False}
        if free_only or approx or n is None:
            e.update(kind="free", claim={"n": n, "noun": "", "group": None, "role": "unrelated"})
        elif noun_m:
            noun = " ".join((noun_m.group("words") + " " + noun_m.group("noun")).split())[:40]
            e.update(kind="count", ext=m.end() + noun_m.end(),
                     claim={"n": n, "noun": noun, "group": _group_of(noun_m.group("noun")),
                            "role": "primary"})
        elif _LEAD_BEFORE.search(before):
            e.update(kind="lead", claim={"n": n, "noun": "", "group": GROUP_ITEM, "role": "primary"})
        else:
            e.update(kind="free", inherit=not m.group("vague"),
                     claim={"n": n, "noun": "", "group": None, "role": "unrelated"})
        entries.append(e)
    _structure_roles(masked, entries)
    return ([e["claim"] for e in entries if e["kind"] == "count"]
            + [e["claim"] for e in entries if e["kind"] == "lead"]
            + [e["claim"] for e in entries if e["kind"] == "free"])


def _partition_values(group_facts, totals):
    """The values of the facts that BREAK DOWN a matched total, or None. The
    evidence supports a breakdown only if, apart from the fact that gave the
    total, at least two facts of the group sum to it."""
    for t in sorted(totals, reverse=True):
        whole = next((f for f in group_facts if f["n"] == t), None)
        if whole is None or t <= 0:
            continue
        parts = [f["n"] for f in group_facts if f is not whole and f["n"] > 0][:200]
        reach = {0: 0}                                  # subset-sum DP: reachable sum -> most parts used (capped at 2)
        for v in parts:
            for sm, k in list(reach.items()):
                if sm + v <= t:
                    reach[sm + v] = min(2, max(reach.get(sm + v, 0), k + 1))
        if reach.get(t, 0) >= 2:
            return set(parts)
    return None


def check_claims(claims, facts) -> dict:
    """Compare claims with count facts (each {n, group, action}).

    A PRIMARY claim is compared only with facts of its own group: `matched` when
    its number equals one, `failed` when it equals none, `unchecked` when it has
    no group we hold a fact for (including every free number). Exact equality.

    AMBIGUOUS (audit F-13): a lone primary claim whose group holds SEVERAL different counts, and where the other counts are not the
    parts of the claimed total (see _partition_values), equals one of them but the evidence contradicts itself, so it is neither `matched`
    nor `failed`: the caller must not call it verified. Two or more primary claims in a group, or a total that its own parts explain, are
    matched as before.

    A BREAKDOWN claim is compared only when the evidence itself breaks a matched
    total down (see _partition_values). Then it is `breakdown_matched` if it equals
    one of those facts and `failed` if it does not (a wrong breakdown the evidence
    can refute). Without that support it is `unchecked` and
    `breakdown_unchecked`: not backed by evidence, and never a mismatch.

    `facts` and `actions` are the facts RELEVANT to the verdict: those of the
    groups the failed claims are about (or, with nothing failed, the matched
    ones). Facts about other things stay in `facts_all`."""
    group_facts = {}
    for f in facts:
        group_facts.setdefault(f["group"], []).append(f)
    by_group = {g: {f["n"] for f in fs} for g, fs in group_facts.items()}
    primaries = collections.Counter(g for g, _ in {(c.get("group"), c.get("n")) for c in claims
                                                   if (c.get("role") or "primary") != "breakdown" and c.get("n") is not None})
    matched, failed, unchecked, ambiguous = [], [], [], []
    breakdown_matched, breakdown_unchecked, breakdowns = [], [], []
    for c in claims:
        if (c.get("role") or "primary") == "breakdown" and c.get("group"):
            breakdowns.append(c)
            continue
        held = by_group.get(c.get("group"))
        if not held or c.get("n") is None:
            unchecked.append(c)
        elif c["n"] in held:
            lone = len(held) > 1 and primaries[c.get("group")] == 1
            if lone and _partition_values(group_facts[c["group"]], {c["n"]}) is None:
                ambiguous.append(c)               # audit F-13: ONE claim, SEVERAL different counts in its group, and the others are not parts
            else:                                 # of it: it equals one of them by luck as much as by fact, so it is not "verified" (a summary
                matched.append(c)                 # saying 20 against tool outputs of 21 and 20 is exactly the roadmap's failure case)
        else:
            failed.append(c)
    for c in breakdowns:
        g = c["group"]
        parts = _partition_values(group_facts.get(g, []),
                                  {m["n"] for m in matched if m.get("group") == g})
        if c.get("n") is None or parts is None:
            unchecked.append(c)
            breakdown_unchecked.append(c)
        elif c["n"] in parts:
            breakdown_matched.append(c)
        else:
            failed.append(c)
    groups = {c["group"] for c in (failed or ambiguous or matched)}
    relevant = [f for f in facts if f["group"] in groups]
    return {"matched": matched, "failed": failed, "unchecked": unchecked, "ambiguous": ambiguous,
            "breakdown_matched": breakdown_matched, "breakdown_unchecked": breakdown_unchecked,
            "facts": sorted({f["n"] for f in relevant}),
            "actions": [f["action"] for f in relevant],
            "facts_all": sorted({f["n"] for f in facts})}
