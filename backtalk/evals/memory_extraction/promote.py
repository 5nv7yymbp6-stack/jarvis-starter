"""Promote ticked memory candidates into the Memory 2.0 candidate pipeline. Writes NO memory.

    python -m backtalk.evals.memory_extraction.promote [--vault PATH]

`00 - Inbox/Memory Candidates.md` holds statements the nightly capture proposes. Ticking one (`- [x]`) is Yousif's approval to PROMOTE it:
this program moves each ticked, supported line, verbatim, from that note to the END of `00 - Inbox/Approved Memory Statements.md`, a note that
is append-only on purpose (Memory 2.0 reviews cite source line numbers; appending never shifts them, so a reviewed candidate cannot go
stale). `memory_extract` reads that note as one more note source, so the next `scan` OFFERS the statement as a candidate, and from there it
still needs everything a memory always needs: his affirmation in the interactive review, the gate, the approval file he creates, the typed
challenge and the phone reply. This program touches neither `11 - Memory/` nor any ledger and holds no approval.

A ticked line is promoted only if the extractor would actually accept it (class mapped, key derivable, evidence usable, passes the secret
and injection screen); otherwise it stays in the inbox and the reason is counted (never the text). Idempotent: a candidate id already in the
approved note is only removed from the inbox. Both files are replaced atomically; the approved note is written first, so a crash can only
leave a duplicate that the next run clears.
"""
import argparse
import fcntl
import os
import sys
import tempfile
from collections import Counter

from backtalk import memory_extract as X

DEFAULT_VAULT = "/Users/yousif/Desktop/jarvis/jarvis"
INBOX = "00 - Inbox/Memory Candidates.md"
APPROVED = X.INBOX_APPROVED
LOCK = os.path.expanduser("~/.backtalk-memory-review/promote.lock")
HEADER = """---
status: active
project: meta
type: log
---
# Approved Memory Statements

APPEND-ONLY. Never edit, reorder or delete a line: Memory 2.0 reviews cite these lines by number. Each line is a statement Yousif ticked in
[[Memory Candidates]], moved here verbatim by `promote.py`. Being here is NOT being a memory: the next extraction offers it as a candidate
and it still needs his affirmation in the interactive review, the approval file, the typed challenge and the phone reply.

## Approved
"""


def _write_atomic(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-promote-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def accepted(line) -> tuple:
    """(ok, reason) for one ticked line: would the extractor accept it as a candidate source?"""
    if not X.APPROVED_LINE.match(line):
        return False, "malformed"
    excl = Counter()
    raws = X._approved_raws(X.Note(X.INBOX_APPROVED, line, X.sha(line)), excl)
    if not raws:
        return False, next(iter(excl), "not_extractable")
    kept = [r for r in raws if X.screen_raw(r, excl, "")]
    return (True, "") if kept else (False, next(iter(excl), "screened_out"))


def promote(vault=DEFAULT_VAULT) -> dict:
    result = {"promoted": 0, "already": 0, "left": Counter()}
    os.makedirs(os.path.dirname(LOCK), mode=0o700, exist_ok=True)
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        inbox_path, appr_path = os.path.join(vault, INBOX), os.path.join(vault, APPROVED)
        if not os.path.isfile(inbox_path):
            return result
        with open(inbox_path, encoding="utf-8") as f:
            inbox = f.read()
        appr = open(appr_path, encoding="utf-8").read() if os.path.isfile(appr_path) else HEADER
        have = {m.group("tag") for m in (X.APPROVED_LINE.match(l) for l in appr.split("\n")) if m}
        keep, moved = [], []
        for line in inbox.split("\n"):
            if not line.startswith("- [x] "):
                keep.append(line)
                continue
            m = X.APPROVED_LINE.match(line)
            tag = m.group("tag") if m else None
            if tag and tag in have:
                result["already"] += 1                 # already promoted earlier: just clear it from the inbox
                continue
            ok, why = accepted(line)
            if not ok:
                result["left"][why] += 1
                keep.append(line)
                continue
            moved.append(line)
            have.add(tag)
        if moved:
            _write_atomic(appr_path, appr.rstrip("\n") + "\n" + "\n".join(moved) + "\n")
            result["promoted"] = len(moved)
        if moved or result["already"]:
            _write_atomic(inbox_path, "\n".join(keep))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault", default=DEFAULT_VAULT)
    r = promote(ap.parse_args(argv).vault)
    print(f"promote: {r['promoted']} promoted, {r['already']} already promoted, "
          f"{sum(r['left'].values())} left in the inbox {dict(sorted(r['left'].items())) if r['left'] else ''}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
