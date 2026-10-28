"""The Phase 6 scanner: READ the local Claude transcripts and the eligible vault notes, run the pure extractor and the
gate dry-run, and WRITE a review pack to a folder outside the vault, the repository, the transcripts and the store.

    python -m backtalk.evals.memory_extraction.scan [--out DIR] [--vault DIR] [--transcripts DIR] [--now TS]

Everything it opens, it opens for reading ('r'/'rb'); the only writes are the pack files, in `write_pack`, into a new
private (0700) folder whose location is checked first. No network, no model, no Ollama. Transcript text is read into
memory, screened, reduced to short verbatim evidence quotes, and never copied anywhere else.
"""
import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone

from backtalk import memory as M
from backtalk import memory_extract as X

HOME = os.path.expanduser("~")
DEFAULT_TRANSCRIPTS = os.path.join(HOME, ".claude/projects/-Users-yousif-Desktop-jarvis")
DEFAULT_VAULT = "/Users/yousif/Desktop/jarvis/jarvis"
DEFAULT_OUT = os.path.join(HOME, ".backtalk-memory-review")
MAX_LINE_BYTES = 4_000_000

EXTRACT_NOTES = ("VAULT-INDEX.md", "03 - Markets/Kansas City (Missouri).md", "04 - Wholesaling/04 - Wholesaling.md",
                 "10 - New Ventures/10 - New Ventures.md", X.INBOX_APPROVED)
QUOTE_NOTES = ("LESSONS.md", "BOOT-CONFIG.md", "VAULT-INDEX.md")     # rule documents: quoting them is not a new statement
_DAILY_DIR = "01 - Daily Notes"
PACK_FILES = ("review_pack.json", "report.md", "private_review.json")


def probe_paths() -> set:
    return {path for probe in X.KNOWN_PROBES for _, path, _ in probe.sides}


# ---- reading (read-only) -------------------------------------------------------------------------

def transcript_files(folder) -> list:
    """Regular, non-symlink *.jsonl files, sorted."""
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if name.endswith(".jsonl") and os.path.isfile(p) and not os.path.islink(p):
            out.append(p)
    return out


def read_turns(folder):
    """(turns, stats). A malformed, oversize or odd line is counted and skipped; it can never stop the scan."""
    turns, stats = [], Counter()
    for path in transcript_files(folder):
        stats["files"] += 1
        name = os.path.basename(path)
        with open(path, "rb") as f:
            for lineno, raw in enumerate(f, 1):
                stats["lines"] += 1
                if len(raw) > MAX_LINE_BYTES:
                    stats["oversize_lines"] += 1
                    continue
                try:
                    entry = json.loads(raw.decode("utf-8", "replace"))
                except (ValueError, RecursionError):
                    stats["malformed_lines"] += 1
                    continue
                if not isinstance(entry, dict):
                    stats["malformed_lines"] += 1
                    continue
                if entry.get("type") == "user":
                    stats["user_entries"] += 1
                turn, why = X.classify_turn(entry, file=name, line=lineno)
                if turn is not None:
                    turns.append(turn)
                    stats["turn_" + turn.channel] += 1
                elif entry.get("type") == "user":
                    stats["excluded:" + (why or "unknown")] += 1
    return turns, stats


def load_notes(vault) -> list:
    wanted = set(EXTRACT_NOTES) | probe_paths() | set(QUOTE_NOTES)
    base = os.path.join(vault, _DAILY_DIR)
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), vault).replace(os.sep, "/")
                if X._DAILY_RX.match(rel):
                    wanted.add(rel)
    notes = []
    for rel in sorted(wanted):
        p = os.path.join(vault, *rel.split("/"))
        if not os.path.isfile(p) or os.path.islink(p):
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        notes.append(X.Note(rel, text, X.sha(text)))
    return notes


# ---- the pipeline --------------------------------------------------------------------------------

def run_pipeline(transcripts, vault, generated_at):
    turns, tstats = read_turns(transcripts)
    notes = load_notes(vault)
    cands, cexcl = X.build_candidates(turns, [n for n in notes if _is_extract(n.path)], generated_at=generated_at,
                                      reference_notes=notes)
    conflicts = X.known_conflicts(notes) + X.generic_conflicts(cands)
    exclusions = {"turns": dict(sorted(tstats.items())), "candidates": dict(sorted(cexcl.items()))}
    meta = {"generated_at": generated_at, "transcript_files": tstats["files"], "notes_read": len(notes),
            "notes_fingerprint": X.sha("".join(n.path + n.sha for n in notes)), "exclusions": exclusions,
            "channels": {"typed": tstats["turn_typed"], "voice": tstats["turn_voice"]}}
    return cands, conflicts, meta, notes


def _is_extract(path) -> bool:
    return path in EXTRACT_NOTES or bool(X._DAILY_RX.match(path))


# ---- writing the pack (the only writes) ----------------------------------------------------------

def output_dir_ok(out, *, forbidden) -> str:
    """The real path of `out`, or ValueError if it is inside any forbidden root (vault, repository, transcripts, store)."""
    target = os.path.realpath(out)
    for root in forbidden:
        r = os.path.realpath(root)
        if target == r or target.startswith(r + os.sep):
            raise ValueError("the review pack must be written outside the vault, the repository, the transcripts and the store")
    return target


def _write_private(path, text):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_pack(out, cands, conflicts, meta, *, private=True, forbidden=()) -> dict:
    """Write review_pack.json, report.md and (unless private=False) private_review.json into out/<run id>/."""
    root = output_dir_ok(out, forbidden=forbidden)
    run = os.path.join(root, "run-" + re.sub(r"\D", "", meta["generated_at"]))
    if not os.path.isdir(root):
        os.makedirs(root, mode=0o700)                       # the review root is private too, not just each run
    os.makedirs(run, mode=0o700, exist_ok=True)
    pack = X.review_pack(cands, conflicts, meta)
    files = {"review_pack.json": X.dumps(pack), "report.md": X.report(pack)}
    if private:
        files["private_review.json"] = X.dumps(X.private_companion(cands))
    for name, text in files.items():
        _write_private(os.path.join(run, name), text)
    return {"dir": run, "files": sorted(files)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--vault", default=DEFAULT_VAULT)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--now", default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    ap.add_argument("--no-private", action="store_true")
    a = ap.parse_args(argv)
    if M.parse_ts(a.now) is None:
        sys.exit("--now must be YYYY-MM-DDTHH:MM:SSZ")
    forbidden = [a.vault, os.path.dirname(os.path.realpath(a.vault)), a.transcripts]
    cands, conflicts, meta, _ = run_pipeline(a.transcripts, a.vault, a.now)
    done = write_pack(a.out, cands, conflicts, meta, private=not a.no_private, forbidden=forbidden)
    s = X.review_pack(cands, conflicts, meta)["summary"]
    print(f"wrote {done['files']} to {done['dir']}")
    print(f"candidates {s['candidates']}  gate {s['gate_decisions']}  confirm {s['requires_confirmation']}")


if __name__ == "__main__":
    main()
