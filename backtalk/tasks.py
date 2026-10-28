"""`python -m backtalk.tasks`: the typed, persistent task history, read-only (roadmap Phase C, item 5).

    tasks                       the 20 newest turns, actions and background tasks, newest first
    tasks --failures            only what went wrong (FAILED, PARTIAL, UNKNOWN) with the failure class and recovery decision
    tasks --kind task --n 50    one kind, more rows
    tasks --status CANCELLED    one status

It replays the ledger file and prints ids, states, classes and times. It never writes, never opens a write handle, and never
prints message text (the ledger does not keep bodies; summaries are not shown).
"""
import argparse
import sys
import time

from backtalk import ledger as L
from backtalk import taskstate as T


def render(views):
    lines = []
    for v in views:
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(v.ended_at or v.started_at or 0))
        lines.append(f"{ts}  {T.explain(v)}")
    return lines or ["nothing recorded"]


def main(argv=None, ledger_dir=None):
    p = argparse.ArgumentParser(prog="tasks", description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--kind", choices=("turn", "action", "task"))
    p.add_argument("--status", choices=T.STATUSES)
    p.add_argument("--failures", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.n <= 500:
        print("tasks: --n must be 1 to 500", file=sys.stderr)
        return 2
    led = L.Ledger(ledger_dir, fsync=False)
    statuses = {T.FAILED, T.PARTIAL, T.UNKNOWN} if a.failures else ({a.status} if a.status else None)
    print("\n".join(render(T.recent(led, a.n, kinds=(a.kind,) if a.kind else ("turn", "action", "task"), statuses=statuses))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
