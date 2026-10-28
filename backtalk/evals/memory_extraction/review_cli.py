"""The Memory 2.0 review program (Phase 7): you, at a terminal, deciding one candidate at a time. REVIEW ONLY.

    python -m backtalk.evals.memory_extraction.review_cli            # interactive review (needs a real terminal)
    python -m backtalk.evals.memory_extraction.review_cli --status    # counts, no values        (no terminal needed)
    python -m backtalk.evals.memory_extraction.review_cli --pilot     # the proposed pilot list  (no terminal needed)
    python -m backtalk.evals.memory_extraction.review_cli --verify    # check the ledger's hash chain

HARD RULES, all enforced in code and tests:
  * Deciding needs a TERMINAL on both input and output. A pipe, a file, a script or a process an agent launched is refused
    before anything is read. Voice and chat are never an approval channel.
  * It reads the review pack (outside the vault) and the vault/transcripts only to re-check that a source still says what it
    said. It writes ONLY the review ledger (`ledger.jsonl`) and, for a private edit, `ledger_private.jsonl`, both mode 0600
    in a 0700 folder under ~/.backtalk-memory-review/. It never writes to the vault, the Memory 2.0 store, the index or the
    dreaming system, and it does not import the store.
  * It never reads `private_review.json`. Private candidates are not offered in this phase.
  * `lock` must be typed in full. Enter, `y` or anything else is not a decision.
"""
import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from backtalk import memory as M
from backtalk import memory_extract as X
from backtalk import memory_review as R
from backtalk.evals.memory_extraction import evaluate as E
from backtalk.evals.memory_extraction import scan

LEDGER, PRIVATE = "ledger.jsonl", "ledger_private.jsonl"
NO_TTY = "A review decision needs a real terminal for both input and output. Nothing was read or written."


def latest_run(root) -> str:
    runs = sorted(d for d in os.listdir(root) if d.startswith("run-") and os.path.isfile(os.path.join(root, d, "review_pack.json")))
    if not runs:
        raise R.ReviewError("no_pack")
    return os.path.join(root, runs[-1])


def read_pack(run_dir):
    path = os.path.join(run_dir, "review_pack.json")
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), X.sha(raw.decode("utf-8")), os.path.basename(run_dir)


def read_ledger(path) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                raise R.ReviewError("ledger_corrupt", ("unreadable_line",)) from None
    return out


def append_line(path, obj, *, expect_prev=None):
    """Append one JSON line under an exclusive lock. When `expect_prev` is given the file's last record must still be that
    hash (a second writer cannot interleave). The file is created 0600."""
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.fchmod(fd, 0o600)
        if expect_prev is not None:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l for l in f.read().split("\n") if l]
            last = json.loads(lines[-1])["hash"] if lines else R.GENESIS
            if last != expect_prev:
                raise R.ReviewError("ledger_changed")
        os.write(fd, (R.canonical_json(obj) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def source_problems(view, transcripts, vault) -> list:
    """Re-open the candidate's source and report what no longer matches (empty = still says what it said)."""
    return E.verify_provenance(SimpleNamespace(**view), transcripts, vault)


class Session:
    def __init__(self, root, run_dir, vault, transcripts, ask, say, now_fn):
        self.root, self.vault, self.transcripts = root, vault, transcripts
        self.ask, self.say, self.now_fn = ask, say, now_fn
        raw, self.pack_sha, self.run = read_pack(run_dir)
        self.pack = R.Pack(raw)
        self.ledger_path, self.private_path = os.path.join(root, LEDGER), os.path.join(root, PRIVATE)
        self.records = read_ledger(self.ledger_path)
        if self.records:
            self.state = R.State(self.records)
            self.push(R.session_record(self.records, self.now(), run=self.run, pack_sha=self.pack_sha))
        else:
            self.push(R.open_record(self.records, self.now(), run=self.run, pack_sha=self.pack_sha, pack_format=raw["format"]))

    def now(self) -> str:
        return self.now_fn()

    def push(self, rec, private=None):
        prev = self.records[-1]["hash"] if self.records else R.GENESIS
        if private is not None:
            append_line(self.private_path, {"seq": rec["seq"], "candidate": rec["candidate"], **private})
        append_line(self.ledger_path, rec, expect_prev=prev)
        self.records.append(rec)
        self.state = R.State(self.records)

    def problems(self, cid):
        return source_problems(self.pack.views[cid], self.transcripts, self.vault)

    # -- one item --

    def show(self, cid):
        self.say("")
        self.say("=" * 78)
        for line in R.why(self.pack, self.state, cid):
            self.say(line)
        self.say("-" * 78)

    def do(self, fn, *a, **k):
        """Run a state-machine action, record it, and say why if it is refused (a code, never a value)."""
        try:
            out = fn(self.pack, self.state, *a, now=self.now(), **k)
        except R.ReviewError as e:
            self.say(f"refused: {e}")
            return None
        self.push(out.record, out.private)
        return out.record

    def resolve(self, gid):
        g = self.pack.groups[gid]
        self.say(f"\nCONFLICT {gid}  {g['key']}  [{g['scope']}]  (nothing is decided for you)")
        for i, s in enumerate(g["sides"], 1):
            q = s.get("quote") or "[private]"
            self.say(f"  {i}. {s['label']}: {s['path']} line {s['line']}" + (f" ({s['date_hint']})" if s.get("date_hint") else "") + f" - \"{q}\"")
        if g["scope"] == "document_drift":
            if self.ask("This is a difference between your ordinary notes, not a memory. Type `drift` to acknowledge: ").strip() == "drift":
                return self.do(R.conflict_choice, gid, "drift")
            return None
        c = self.ask("choose: a number = that side is current / own = I'll type it / both / neither / back > ").strip()
        if c.isdigit() and 1 <= int(c) <= len(g["sides"]):
            return self.do(R.conflict_choice, gid, "side", label=g["sides"][int(c) - 1]["label"])
        if c in ("own", "both", "neither"):
            return self.do(R.conflict_choice, gid, c)
        return None

    def item(self, cid) -> bool:
        """Review one candidate. False = the person quit."""
        v = self.pack.views[cid]
        problems = self.problems(cid)
        self.show(cid)
        if problems:
            self.say("STALE: the source no longer matches (" + ", ".join(problems) + "). Re-extract before it can be reviewed.")
            self.do(R.stale, cid, problems)
            return True
        while True:
            for gid in v["conflict_groups"]:
                if self.pack.groups.get(gid, {}).get("scope") == "memory" and gid not in self.state.choices:
                    self.resolve(gid)
            cmd = self.ask("[lock | edit | reject | skip | defer | why | status | quit] > ").strip()
            if cmd == "quit":
                return False
            if cmd == "why":
                self.show(cid)
            elif cmd == "status":
                self.say(json.dumps(R.summary(self.pack, self.state), sort_keys=True))
            elif cmd == "lock":
                typed = self.ask("Type the word  lock  to adopt this exactly as shown: ").strip()
                if self.do(R.lock, cid, typed, problems=self.problems(cid)):
                    self.say("locked (recorded in the review ledger only; nothing was written to memory)")
                    return True
            elif cmd == "edit":
                value = self.ask("your value (in your words): ")
                sent = self.ask("the sentence it rests on (blank = same as the value): ")
                key = self.ask(f"key (blank = {v['attribute']}): ").strip() or None
                priv = self.ask("is it private? [n/y] ").strip().lower() == "y"
                if self.do(R.edit, cid, value=value, evidence=sent.strip() or None, key=key,
                           sensitivity="private" if priv else None, problems=self.problems(cid)):
                    self.say("edited (recorded in the review ledger only; nothing was written to memory)")
                    return True
            elif cmd == "reject":
                reason = self.ask("why? " + " / ".join(R.REJECT_REASONS) + " > ").strip()
                if self.do(R.reject, cid, reason):
                    return True
            elif cmd in ("skip", "defer"):
                if self.do(getattr(R, cmd), cid):
                    return True
            else:
                self.say("not a command (a decision needs the exact word)")


def review(session) -> None:
    while True:
        q = R.queue(session.pack, session.state)
        pending = [c for c in q if session.state.status(c) == "pending"]
        if not pending:
            session.say("\nNo candidate is waiting for a first decision. (skipped, deferred and stale ones can be reopened by running again)")
            return
        if not session.item(pending[0]):
            return


def main(argv=None, *, ask=None, say=None, tty=None, now_fn=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=scan.DEFAULT_OUT)
    ap.add_argument("--run")
    ap.add_argument("--vault", default=scan.DEFAULT_VAULT)
    ap.add_argument("--transcripts", default=scan.DEFAULT_TRANSCRIPTS)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    say = say or print
    ask = ask or input
    now_fn = now_fn or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    inspect_only = a.status or a.pilot or a.verify
    if not inspect_only:
        if tty is None:                       # (`tty` is a test hook; the command line has no way to set it)
            tty = sys.stdin.isatty() and sys.stdout.isatty()
        if not tty:
            say(NO_TTY)
            return 2
    root = scan.output_dir_ok(a.root, forbidden=[a.vault, os.path.dirname(os.path.realpath(a.vault)), a.transcripts])
    run_dir = a.run or latest_run(root)
    if inspect_only:
        raw, pack_sha, run = read_pack(run_dir)
        pack = R.Pack(raw)
        records = read_ledger(os.path.join(root, LEDGER))
        problems = R.verify_chain(records)
        if a.verify:
            say("ledger chain intact" if not problems else "ledger chain BROKEN: " + ", ".join(problems))
            return 0 if not problems else 1
        if problems:
            say("ledger chain BROKEN: " + ", ".join(problems))
            return 1
        state = R.State(records)
        if a.pilot:
            for cid, key in R.pilot(pack):
                say(f"{cid}  {key}  [{state.status(cid)}]")
            say(f"{len(R.pilot(pack))} candidates (no private, instruction or template-default candidate is ever offered)")
        else:
            say(json.dumps(R.summary(pack, state), sort_keys=True, indent=2))
        return 0
    os.makedirs(root, mode=0o700, exist_ok=True)
    session = Session(root, run_dir, a.vault, a.transcripts, ask, say, now_fn)
    review(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
