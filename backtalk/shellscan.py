"""Shell-command helpers shared by the evidence and ledger modules.

_scan_bash splits a command on UNQUOTED separators and reports constructs it
cannot vouch for (evidence.py uses it to decide whether a command's stdout is
safe to count). _PREVIEW_REDACTIONS scrubs secrets out of text before it is
logged or written to the ledger (ledger.redact uses it).
"""
import re


# zsh glob qualifiers that are safe: file-type/permission/sort/null-glob
# letters. `e` (eval a string) and `+` (call a function) are deliberately
# absent, as are `:` modifiers and subscripts.
_SAFE_QUAL = frozenset(".@/=p*%rwxAIERWXsStdlugonOmacLNMTDU^-,0123456789")
_UNSAFE_ZSH = ("zsh glob qualifier or expansion I can't verify",
               "run a command with a shell pattern I can't verify")
_UNSAFE_BG = ("backgrounds a command", "run something in the background")
_UNSAFE_GROUP = ("shell grouping or process substitution",
                 "run a command with shell grouping I can't verify")


def _scan_bash(cmd: str):
    """(segments, unsafe). `segments` is the list of simple commands split on
    UNQUOTED separators, or None if the structure could not be understood
    (unbalanced quote or parenthesis, trailing backslash); `unsafe` is
    (reason, verb) when a construct was found that must not be treated as a
    plain read-only command, else None. Either of those means the command
    cannot be vouched for."""
    segs, cur, word = [], [], []
    i, n = 0, len(cmd)
    in_s = in_d = False
    arr = 0

    def flush():
        t = "".join(cur).strip()
        if t:
            segs.append(t)
        cur.clear()

    while i < n:
        c = cmd[i]
        if in_s:                                   # '...' : nothing is special
            cur.append(c)
            if c == "'":
                in_s = False
            i += 1
            continue
        if in_d:                                   # "..." : backslash escapes
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            if c == '"':
                in_d = False
            elif c == "$" and cmd.startswith("{(", i + 1):
                return None, _UNSAFE_ZSH           # ${(e)x} still expands in ""
            i += 1
            continue
        if c == "\\":
            if i + 1 >= n:
                return None, None
            cur.append(c)
            cur.append(cmd[i + 1])
            word.append("x")
            i += 2
            continue
        if c == "'":
            in_s = True
            cur.append(c)
            word.append("'")
            i += 1
            continue
        if c == '"':
            in_d = True
            cur.append(c)
            word.append('"')
            i += 1
            continue
        if c == "$" and cmd.startswith("{(", i + 1):
            return None, _UNSAFE_ZSH               # ${(e)...}: parameter flags run code
        if arr and c in ";\n|&":
            return None, _UNSAFE_GROUP             # a separator inside an array body
        if cmd[i:i + 2] in ("&&", "||"):
            flush()
            word.clear()
            i += 2
            continue
        if c in ";\n":
            flush()
            word.clear()
            i += 1
            continue
        if c == "|":
            if cmd[i + 1:i + 2] == "&":
                return None, _UNSAFE_BG
            flush()
            word.clear()
            i += 1
            continue
        if c == "&":
            prev = cmd[i - 1] if i else ""
            nxt = cmd[i + 1] if i + 1 < n else ""
            if prev in "<>" or nxt == ">":         # 2>&1, <&3, &>file: a redirect
                cur.append(c)
                word.append(c)
                i += 1
                continue
            return None, _UNSAFE_BG                # a lone & backgrounds the command
        if c in " \t":
            cur.append(c)
            word.clear()
            i += 1
            continue
        if c == "(":
            w = "".join(word)
            if not arr and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\+?=", w):
                arr = 1                            # name=( ... ) array assignment
                cur.append(c)
                word.clear()
                i += 1
                continue
            if w in ("", "="):                     # ( subshell, =(cmd) temp-file form
                return None, _UNSAFE_GROUP
            j = cmd.find(")", i + 1)               # a qualifier on a glob word: *(...)
            if j < 0:
                return None, None
            body = cmd[i + 1:j]
            if not body or not set(body) <= _SAFE_QUAL:
                return None, _UNSAFE_ZSH
            cur.append(cmd[i:j + 1])
            word.append("q")
            i = j + 1
            continue
        if c == ")":
            if arr:
                arr = 0
                cur.append(c)
                word.clear()
                i += 1
                continue
            return None, _UNSAFE_GROUP
        cur.append(c)
        word.append(c)
        i += 1
    if in_s or in_d or arr:
        return None, None
    flush()
    return segs, None


# Best-effort redaction, applied to the whole text BEFORE any preview is cut,
# so a secret straddling the cut can't leak its first half. Order matters:
# specific patterns first, the generic NAME=value and long-blob catch-alls
# last. This is a LOG aid, not a security control.
_PREVIEW_REDACTIONS = [
    (re.compile(r"(://)[^/\s:@]+:[^/\s@]+@"), r"\1***@"),        # url user:pass@
    (re.compile(r"(?i)(authorization\s*:\s*\w+\s+)[^\s'\"]+"), r"\1***"),
    (re.compile(r"(?i)\b(bearer\s+)[^\s'\"]+"), r"\1***"),
    (re.compile(r"(?i)\b((?:x-)?api[-_]?key|x-auth-token|set-cookie|cookie)"
                r"\s*:\s*[^'\"\n]+"), r"\1: ***"),                # secret headers
    (re.compile(r"(?i)(--?(?:password|passwd|pass|token|secret|api[-_]?key|"
                r"auth|access[-_]?key|client[-_]?secret)(?:=|\s+))"
                r"(\"[^\"]*\"|'[^']*'|\S+)"), r"\1***"),          # secret flags
    (re.compile(r"(?<!\S)(-u|--user)\s+\S+"), r"\1 ***"),        # curl -u user:pass
    (re.compile(r"(?<!\S)(-d|--data(?:-raw|-binary|-urlencode)?|-F|--form)"
                r"(?:=|\s+)(\"[^\"]*\"|'[^']*'|\S+)"), r"\1 ***"),  # request bodies
    (re.compile(r"(?<![\w-])([A-Za-z_][A-Za-z0-9_]*=)(\"[^\"]*\"|'[^']*'|\S+)"),
     r"\1***"),                                                   # any NAME=value
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), "***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "***"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*"),
     "***"),                                                      # JWT
    (re.compile(r"\b[A-Za-z0-9+_=-]{32,}\b"), "***"),            # opaque blobs
]
