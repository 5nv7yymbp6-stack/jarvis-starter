"""`python -m backtalk.remind`: queue, list or cancel a spoken reminder.

    remind add --in 10m "call the title company"         once, 10 minutes from now (s, m, h, d)
    remind add --at 14:30 "send the offer"                once, the next 14:30 local time
    remind add --daily 08:30 "check the lead list"        every day at 08:30 local time
    remind add --every 2h "stretch"                       every two hours
    remind add --in 5m --urgent "the oven"                 urgent (still never over the user's voice)
    remind list                                            what is scheduled, and what is still waiting to be picked up
    remind cancel sch_ab12cd34ef56                         cancel one

The running voice line owns the scheduler (it holds an exclusive lock), so this program never writes it. `add` and `cancel`
drop one small request file into `<scheduler dir>/inbox/` (written to a temp name, then renamed: a reader never sees half a
file) and the voice line applies it within a few seconds. So the honest word for what `add` did is QUEUED, not scheduled:
`remind list` shows it once the voice line has picked it up. Jarvis itself speaks the reminder when it is due, from the words
given here; the model is never in that path. Prints no secret and no message text beyond what the user typed.
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backtalk import ledger as L

TZ = "America/Chicago"
_DUR = re.compile(r"(\d+)\s*([smhd])\Z")
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)\Z")
_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400}
TITLE_MAX = 200


def sched_dir():
    return os.path.join(str(L.default_dir()), "scheduler")


def parse_duration(text):
    m = _DUR.match(str(text).strip().lower())
    if not m or int(m.group(1)) <= 0:
        raise ValueError("a duration looks like 90s, 10m, 2h or 1d")
    return int(m.group(1)) * _UNIT[m.group(2)]


def next_local(hhmm, now, tz=TZ):
    if not _HHMM.match(hhmm):
        raise ValueError("a time looks like 14:30")
    h, mi = int(hhmm[:2]), int(hhmm[3:])
    z = ZoneInfo(tz)
    base = datetime.fromtimestamp(now, z)
    day = base.date()
    for _ in range(3):
        cand = datetime(day.year, day.month, day.day, h, mi, tzinfo=z)
        if cand.timestamp() > now:
            return cand.timestamp()
        day += timedelta(days=1)
    raise ValueError("no such time")


def write_request(req, directory=None):
    d = os.path.join(directory or sched_dir(), "inbox")
    os.makedirs(d, mode=0o700, exist_ok=True)
    name = f"{int(time.time() * 1000):015d}-{req['id']}.json"
    tmp = os.path.join(d, f".tmp-{req['id']}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(req, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(d, name))
    return name


def build_add(args, now):
    title = " ".join(" ".join(args.text).split())[:TITLE_MAX]
    if not title:
        raise ValueError("say what to remind you of")
    req = {"v": 1, "id": uuid.uuid4().hex[:12], "op": "add", "title": title,
           "severity": "urgent" if args.urgent else "normal"}
    chosen = [x for x in (args.in_, args.at, args.daily, args.every) if x]
    if len(chosen) != 1:
        raise ValueError("give exactly one of --in, --at, --daily, --every")
    if args.in_:
        req.update(kind="once", due_at=now + parse_duration(args.in_))
    elif args.at:
        req.update(kind="once", due_at=next_local(args.at, now))
    elif args.daily:
        if not _HHMM.match(args.daily):
            raise ValueError("a time looks like 08:30")
        req.update(kind="daily", hhmm=args.daily, tz=TZ)
    else:
        req.update(kind="every", interval_s=float(parse_duration(args.every)))
    return req


def describe(req):
    z = ZoneInfo(TZ)
    if req["kind"] == "once":
        return "at " + datetime.fromtimestamp(req["due_at"], z).strftime("%a %H:%M")
    if req["kind"] == "daily":
        return f"every day at {req['hhmm']}"
    return f"every {int(req['interval_s'])} seconds"


def list_tasks(directory=None):
    from backtalk import scheduler as S
    d = directory or sched_dir()
    lines = []
    if os.path.exists(os.path.join(d, "scheduler.jsonl")):
        view = S.Scheduler.read_only(d)
        z = ZoneInfo(TZ)
        for t in sorted(view.tasks(), key=lambda x: x.next):
            if t.status in ("scheduled", "running"):
                when = datetime.fromtimestamp(t.retry_at if t.retry_at is not None else t.next, z).strftime("%a %H:%M")
                lines.append(f"{t.id}  {t.status:<9} {t.kind:<5} next {when}  {t.title}")
    inbox = os.path.join(d, "inbox")
    waiting = [n for n in os.listdir(inbox) if n.endswith(".json")] if os.path.isdir(inbox) else []
    if waiting:
        lines.append(f"({len(waiting)} request(s) queued, not yet picked up by the voice line)")
    return lines or ["nothing is scheduled"]


def main(argv=None, now=None):
    p = argparse.ArgumentParser(prog="remind", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--in", dest="in_")
    a.add_argument("--at")
    a.add_argument("--daily")
    a.add_argument("--every")
    a.add_argument("--urgent", action="store_true")
    a.add_argument("text", nargs="+")
    sub.add_parser("list")
    c = sub.add_parser("cancel")
    c.add_argument("task_id")
    args = p.parse_args(argv)
    now = time.time() if now is None else now
    try:
        if args.cmd == "add":
            req = build_add(args, now)
            write_request(req)
            print(f"queued {req['id']}: \"{req['title']}\" {describe(req)}. The voice line applies it within a few seconds; "
                  "`remind list` confirms.")
        elif args.cmd == "cancel":
            if not re.fullmatch(r"sch_[0-9a-f]{12}", args.task_id):
                raise ValueError("a task id looks like sch_ab12cd34ef56")
            write_request({"v": 1, "id": uuid.uuid4().hex[:12], "op": "cancel", "task_id": args.task_id})
            print(f"cancel queued for {args.task_id}")
        else:
            print("\n".join(list_tasks()))
    except (ValueError, OSError) as e:
        print(f"remind: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
