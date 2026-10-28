"""Phone -> Mac request channel, MAC side (roadmap Phase H; design: vault note "Phone-to-Mac Channel Design").

    python -m backtalk.phonechan poll      one pass: fetch, verify, hand reminders to the scheduler, ack. Run every 30 s by a LaunchAgent.

The Mac connects OUTBOUND over ssh with a key that the server restricts to a forced command (`drain_queue.py`: verbs `fetch` and
`ack <id> <outcome>` only). Nothing listens on this Mac. Each request is accepted only if ALL of these hold:
  * the HMAC-SHA256 signature verifies (constant time) with the key in KEY_PATH (a 0600 file; a looser file is refused)
  * the type is `remind`, the id is 16 hex characters, no unknown fields, the title is one printable line of at most 200 characters
  * due_at is within [now - 6 h, now + 30 d] and created_at is not in the future and not older than 7 days
  * the id has not been accepted before (a persisted, capped replay set)
  * fewer than MAX_PER_DAY reminders were accepted in the last 24 h
Three request types, a closed list (anything else is rejected): `remind` (spoken reminder), `note` (one line appended to the vault inbox
note NOTE_PATH, data only) and `status` (the Mac texts a short, read-only summary straight to his own Telegram chat).
Accepted reminders are written to the SAME inbox the `remind` CLI uses, so the voice line speaks them with its code-built line; this
module never speaks, never runs a command and never interprets the text. Every outcome is acked so the server's queue drains, except a
failure to write the inbox (left unacked, retried next pass). Kill switch: the file SWITCH_OFF existing makes `poll` do nothing.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid

from backtalk import remind

HOST = "jarvis@137.184.130.111"
VAULT = "/Users/yousif/Desktop/jarvis/jarvis"
NOTE_PATH = os.path.join(VAULT, "00 - Inbox", "Phone Notes.md")
NOTE_MAX_FILE = 200_000
ENV_PATH = "/Users/yousif/Desktop/jarvis/phone-bridge/.env"
DREAM_LOG = "/Users/yousif/Desktop/jarvis/dreaming/dream.log"
TYPES = ("remind", "note", "status", "act", "confirm")
# Tier 2 actions (change something): a closed list. Each needs a one-time code the Mac texts to his chat DIRECTLY (it never passes through
# the server), then his typed `/yes <code>` within CHALLENGE_TTL_S. A request alone never runs anything.
ACTION_TEXT = {"screenshot": "take a screenshot of this Mac and send it to your chat",
               "restart_voice": "restart the Jarvis voice line"}
CHALLENGE_PATH = os.path.expanduser("~/.config/jarvis/phone_challenges.json")
CHALLENGE_TTL_S = 300
CHALLENGE_PENDING_MAX = 3
GUESS_LIMIT = 5                 # wrong codes within GUESS_WINDOW_S lock the whole thing for LOCKOUT_S and drop every pending challenge
GUESS_WINDOW_S = 300
LOCKOUT_S = 600
_CODE = re.compile(r"\d{6}\Z")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_phonechan")
KEY_PATH = os.path.expanduser("~/.config/jarvis/phone_channel.key")
SWITCH_OFF = "/Users/yousif/Desktop/jarvis/status/.phone_channel_off"
MIRROR_OFF = "/Users/yousif/Desktop/jarvis/status/.mirror_off"     # creating it stops the reminder mirror only (no late-reminder text can then be sent)
MIRROR_STATE = os.path.expanduser("~/.config/jarvis/mirror_last.json")
MIRROR_MAX_TASKS = 50
MIRROR_MAX_BYTES = 60000                            # the server accepts 65536; leave headroom
STATE_PATH = os.path.expanduser("~/.config/jarvis/phone_channel_seen.json")
LOG_PATH = os.path.expanduser("~/Library/Logs/jarvis/phone-channel.log")
TITLE_MAX = 200
PAST_GRACE_S = 6 * 3600
MAX_AHEAD_S = 30 * 86400
MAX_AGE_S = 7 * 86400
INSTANT_MAX_AGE_S = 6 * 3600          # a note or status request that waited longer than this (Mac off) is dropped
MAX_PER_DAY = 30
SEEN_CAP = 500
FIELDS = {"v", "id", "type", "title", "due_at", "created_at", "sig"}
_ID = re.compile(r"[0-9a-f]{16}\Z")


def canonical(req) -> bytes:
    return "\n".join(["v1", str(req["id"]), str(req["type"]), str(req["title"]), f"{float(req['due_at']):.3f}",
                      f"{float(req['created_at']):.3f}"]).encode("utf-8")


def sign(key, req) -> str:
    return hmac.new(key.encode("utf-8"), canonical(req), hashlib.sha256).hexdigest()


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) < 1e12


def validate(req, key, now):
    """(True, "") or (False, outcome) where outcome is 'rejected' or 'expired'. Pure."""
    if not isinstance(req, dict) or set(req) != FIELDS:
        return False, "rejected"
    if req["v"] != 1 or req["type"] not in TYPES or not isinstance(req["id"], str) or not _ID.match(req["id"]):
        return False, "rejected"
    t = req["title"]
    if not isinstance(t, str) or not t or len(t) > TITLE_MAX or not t.isprintable() or t != " ".join(t.split()):
        return False, "rejected"
    if not _num(req["due_at"]) or not _num(req["created_at"]) or not isinstance(req["sig"], str):
        return False, "rejected"
    if not key or not hmac.compare_digest(sign(key, req).encode(), req["sig"].encode("utf-8", "ignore")):
        return False, "rejected"
    if req["created_at"] > now + 300 or now - req["created_at"] > MAX_AGE_S or req["due_at"] > now + MAX_AHEAD_S:
        return False, "rejected"
    if req["type"] in ("note", "status", "act", "confirm"):
        # instant requests carry no schedule: due_at must equal created_at, and a stale one is dropped rather than run late
        if abs(req["due_at"] - req["created_at"]) > 1.0 or (req["type"] == "status" and t != "status"):
            return False, "rejected"
        if (req["type"] == "act" and t not in ACTION_TEXT) or (req["type"] == "confirm" and not _CODE.match(t)):
            return False, "rejected"
        if now - req["created_at"] > INSTANT_MAX_AGE_S:
            return False, "expired"
        return True, ""
    if req["due_at"] < now - PAST_GRACE_S:
        return False, "expired"
    return True, ""


def load_key(path=KEY_PATH):
    """The key, or None when missing/empty or when the file is readable by anyone but the owner."""
    try:
        st = os.stat(path)
        if st.st_mode & 0o077:
            return None
        with open(path) as f:
            k = f.read().strip()
        return k or None
    except OSError:
        return None


class Seen:
    """Persisted replay protection: ids already accepted, plus a day's accept times for the rate cap."""

    def __init__(self, path=STATE_PATH):
        self.path = path
        try:
            with open(path) as f:
                d = json.load(f)
            self.ids = {str(k): float(v) for k, v in dict(d.get("ids", {})).items()}
            self.times = [float(x) for x in d.get("times", [])]
        except (OSError, ValueError, TypeError, AttributeError):
            self.ids, self.times = {}, []

    def has(self, rid):
        return rid in self.ids

    def rate_ok(self, now):
        return len([t for t in self.times if now - t < 86400]) < MAX_PER_DAY

    def add(self, rid, now):
        self.ids[rid] = now
        self.times = [t for t in self.times if now - t < 86400] + [now]
        for k in sorted(self.ids, key=self.ids.get)[:-SEEN_CAP]:
            del self.ids[k]
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"ids": self.ids, "times": self.times}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def ssh_runner(verb, stdin=None):
    """Run one forced-command verb on the server (optionally feeding it stdin). Returns (exit_code, stdout). The verb is the only thing the
    server will accept."""
    r = subprocess.run(["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10",
                        "-o", "StrictHostKeyChecking=yes", HOST, verb], capture_output=True, text=True, encoding="utf-8", timeout=30, input=stdin)
    global LAST_SSH_ERR
    tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
    LAST_SSH_ERR = "".join(c if c.isprintable() else " " for c in tail[0])[:120]       # why a failed pass failed (ssh's own last line)
    return r.returncode, r.stdout


LAST_SSH_ERR = ""


# ---- the reminder mirror (late-reminder fallback; see phone-bridge/late_reminders.py for the server half) ---------------------------

def _clean_title(text):
    kept = "".join(c if c.isprintable() else " " for c in str(text))
    return " ".join(kept.split())[:TITLE_MAX].strip()


def mirror_tasks(now=None, view=None):
    """The voice line's pending reminders as [{"k", "due", "title"}]: scheduled or fired-but-not-done, soonest first, capped. The key is the task
    id plus the ORIGINAL due slot, which does not change when the voice line retries a failed delivery (so a retry can never look like a new
    reminder and be texted twice); `due` is that same original time, because that is when he expected it. A plain scheduled task is keyed by
    its next time, so each occurrence of a repeating reminder is its own key. Read-only; the voice line keeps the lock."""
    if view is None:
        from backtalk import scheduler as S
        view = S.Scheduler.read_only(remind.sched_dir())
    out = []
    for t in view.tasks():
        if t.status not in ("scheduled", "running"):
            continue
        retrying = t.status == "running" or t.retry_at is not None
        due = t.slot if (retrying and t.slot) else t.next
        if not isinstance(due, (int, float)) or isinstance(due, bool) or due != due or due <= 0:
            continue
        title = _clean_title(t.title)
        if title:
            out.append({"k": f"{t.id}@{int(due)}", "due": round(float(due), 3), "title": title})
    return sorted(out, key=lambda x: x["due"])[:MIRROR_MAX_TASKS]


RES_WINDOW_S = 24 * 3600


def mirror_res(now=None, view=None):
    """What the Mac's own scheduler recorded for reminders that are no longer pending, as [{"k", "r"}] with r in done|cancelled|failed|missed,
    recent ones only. `done` means the scheduler COMPLETED it: normally after it was heard (proactive_loop._heard), but also for a duplicate, a deliberate
    ignore or an empty title, so it is NOT proof of speech; `cancelled`/`failed`/`missed` are named for what they are. Keys match mirror_tasks (task id @ the original slot). A running
    task, or one waiting to retry, has no final outcome and is not listed. Read-only."""
    now = time.time() if now is None else now
    if view is None:
        from backtalk import scheduler as S
        view = S.Scheduler.read_only(remind.sched_dir())
    out = []
    for t in view.tasks():
        last = t.history[-1][1] if t.history else 0
        if now - last > RES_WINDOW_S or t.status == "running" or t.retry_at is not None:
            continue

        def add(slot, r):
            if isinstance(slot, (int, float)) and not isinstance(slot, bool) and slot == slot and slot > 0:
                out.append({"k": f"{t.id}@{int(slot)}", "r": r, "_at": last})
        if t.status == "cancelled":
            add(t.next, "cancelled")
            add(t.slot, "cancelled")
        elif t.status == "missed":
            add(t.next, "missed")
        elif t.last_status in ("done", "failed") and t.slot:
            add(t.slot, "done" if t.last_status == "done" else "failed")
    seen, res = set(), []
    for r in sorted(out, key=lambda x: -x["_at"]):
        if (r["k"], r["r"]) not in seen:
            seen.add((r["k"], r["r"]))
            res.append({"k": r["k"], "r": r["r"]})
    return res[:MIRROR_MAX_TASKS]


def mirror_canonical(m) -> bytes:
    parts = ["v1", f"{float(m['at']):.3f}"]
    for t in sorted(m["tasks"], key=lambda x: x["k"]):
        parts.append(f"{t['k']}|{float(t['due']):.3f}|{t['title']}")
    for r in sorted(m.get("res") or [], key=lambda x: x["k"]):
        parts.append(f"r|{r['k']}|{r['r']}")
    return "\n".join(parts).encode("utf-8")


def build_mirror(key, tasks, now, res=None):
    m = {"v": 1, "at": round(float(now), 3), "tasks": tasks}
    if res:
        m["res"] = res
    m["sig"] = hmac.new(key.encode("utf-8"), mirror_canonical(m), hashlib.sha256).hexdigest()
    return m


def sync_mirror(key, now=None, runner=None, tasks=None, log=lambda line: None, res=None):
    """Send the mirror to the server. Returns True only when the server accepted it (exit 0). Never raises; a failure is logged and retried
    on the next pass."""
    try:
        if os.path.exists(MIRROR_OFF):
            return False
        now = time.time() if now is None else now
        tasks = mirror_tasks(now) if tasks is None else tasks
        if res is None:
            try:
                res = mirror_res(now)
            except Exception:
                res = []
        doc = None
        while True:                                          # stay inside the server's byte limit (multibyte titles are bigger than they look)
            doc = json.dumps(build_mirror(key, tasks, now, res), separators=(",", ":"), ensure_ascii=False)
            if len(doc.encode("utf-8")) <= MIRROR_MAX_BYTES or not tasks:
                break
            tasks = tasks[:-1]
            if not tasks and res:
                res = res[:-1]
        code, _ = (runner or ssh_runner)("mirror", stdin=doc)
        if code != 0:
            log(f"mirror not accepted (exit {code})")
            return False
        digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:12]
        try:
            last = json.load(open(MIRROR_STATE)).get("digest")
        except (OSError, ValueError, AttributeError):
            last = None
        if digest != last:
            log(f"mirror synced: {len(tasks)} pending reminder(s)")
            os.makedirs(os.path.dirname(MIRROR_STATE), mode=0o700, exist_ok=True)
            with open(MIRROR_STATE, "w") as f:
                json.dump({"digest": digest, "at": now}, f)
        return True
    except Exception as e:
        log(f"mirror failed ({e.__class__.__name__})")
        return False


def to_inbox(req):
    """The scheduler-inbox request the `remind` CLI would have written for this reminder."""
    return {"v": 1, "id": uuid.uuid4().hex[:12], "op": "add", "title": req["title"], "severity": "normal",
            "kind": "once", "due_at": float(req["due_at"])}


class Challenges:
    """One-time codes for Tier 2 actions, persisted (0600) so a poller restart does not lose them. A code is 6 random digits, stored only as a
    salted hash, valid for CHALLENGE_TTL_S, usable once, at most CHALLENGE_PENDING_MAX outstanding. Wrong guesses are counted globally: after
    GUESS_LIMIT inside GUESS_WINDOW_S everything is cleared and refused for LOCKOUT_S."""

    def __init__(self, path=CHALLENGE_PATH):
        self.path = path
        try:
            with open(path) as f:
                d = json.load(f)
            self.pending = dict(d.get("pending", {}))
            self.fails = [float(x) for x in d.get("fails", [])]
            self.locked_until = float(d.get("locked_until", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            self.pending, self.fails, self.locked_until = {}, [], 0.0

    def _save(self):
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"pending": self.pending, "fails": self.fails, "locked_until": self.locked_until}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    @staticmethod
    def _hash(salt, code):
        return hashlib.sha256((salt + code).encode()).hexdigest()

    def locked(self, now):
        return now < self.locked_until

    def issue(self, rid, action, now):
        """A fresh 6-digit code for `action`, or None while locked out."""
        if self.locked(now):
            return None
        self.pending = {k: v for k, v in self.pending.items() if v["exp"] > now}
        while len(self.pending) >= CHALLENGE_PENDING_MAX:
            del self.pending[min(self.pending, key=lambda k: self.pending[k]["exp"])]
        code = f"{secrets.randbelow(10 ** 6):06d}"
        salt = secrets.token_hex(8)
        self.pending[rid] = {"action": action, "salt": salt, "h": self._hash(salt, code), "exp": now + CHALLENGE_TTL_S}
        self._save()
        return code

    def check(self, code, now):
        """('ok', action) and the challenge is consumed; ('wrong', None); or ('locked', None)."""
        if self.locked(now):
            return "locked", None
        self.pending = {k: v for k, v in self.pending.items() if v["exp"] > now}
        for k, v in list(self.pending.items()):
            if hmac.compare_digest(v["h"], self._hash(v["salt"], code)):
                del self.pending[k]
                self._save()
                return "ok", v["action"]
        self.fails = [t for t in self.fails if now - t < GUESS_WINDOW_S] + [now]
        if len(self.fails) >= GUESS_LIMIT:
            self.pending, self.fails, self.locked_until = {}, [], now + LOCKOUT_S
            self._save()
            return "locked", None
        self._save()
        return "wrong", None


def send_photo(path, caption, env_path=ENV_PATH, opener=None):
    """Send one image file to his Telegram chat from the Mac (multipart, outbound only)."""
    import urllib.request
    env = {}
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip("\"'")
    chat = env["ALLOWED_TELEGRAM_IDS"].split(",")[0].strip()
    boundary = "----jarvis" + secrets.token_hex(8)
    with open(path, "rb") as f:
        img = f.read()
    parts = []
    for name, val in (("chat_id", chat), ("caption", caption[:200])):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="screen.jpg"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n'.encode() + img + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendPhoto", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with (opener or urllib.request.urlopen)(req, timeout=30) as r:
        if r.status != 200:
            raise OSError("telegram refused the photo")


def run_screenshot(now=None, run=subprocess.run, photo=None):
    """Capture the main display to a private temp file, send it to his chat, delete it. Needs Screen Recording permission for the process
    that runs the poller; without it macOS returns an image that is only the wallpaper, so the caption says exactly what it is."""
    import tempfile
    now = time.time() if now is None else now
    with tempfile.TemporaryDirectory(prefix="jarvis-shot-") as d:
        path = os.path.join(d, "shot.jpg")
        run(["/usr/sbin/screencapture", "-x", "-t", "jpg", path], check=True, timeout=20, capture_output=True)
        if not os.path.exists(path) or os.path.getsize(path) < 5_000:
            raise OSError("screenshot came back empty")
        (photo or send_photo)(path, "Mac screen at " + time.strftime("%H:%M", time.localtime(now)))
    return "Screenshot sent, sir."


def run_restart_voice(run=subprocess.run, kill=os.kill, sleep=time.sleep):
    """Interrupt the running voice line (the same SIGINT the terminal's Ctrl-C sends) and relaunch it through its app."""
    import signal
    out = run(["ps", "-eo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    pids = [int(l.split(None, 1)[0]) for l in out.splitlines()
            if "/.venv/bin/python3 -m backtalk.main" in l and "uv run" not in l and "phonechan" not in l]
    for pid in pids:
        try:
            kill(pid, signal.SIGINT)
        except OSError:
            pass
    for _ in range(15):
        sleep(1)
        out = run(["ps", "-eo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
        if not any(str(p) == l.split(None, 1)[0] for p in pids for l in out.splitlines() if l.strip()):
            break
    run(["open", "/Users/yousif/Desktop/jarvis/Jarvis Voice.app"], check=True, timeout=20)
    return "Restarting the voice line, sir. The first word takes about 25 seconds."


EXECUTORS = {"screenshot": run_screenshot, "restart_voice": run_restart_voice}


def append_note(title, now, path=NOTE_PATH):
    """Append ONE bullet to the inbox note (created with frontmatter if missing). The text is data: it has already been validated as
    one printable line; link/heading/code syntax is defused so a note can never become structure. Raises OSError if it cannot write."""
    safe = title.replace("[[", "[ [").replace("]]", "] ]").replace("`", "'").replace("<", "(").replace(">", ")")
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > NOTE_MAX_FILE:
        raise OSError("inbox note is full")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("---\nstatus: active\nproject: meta\ntype: log\n---\n# Phone Notes\n\n"
                    "One line per note sent from the phone with /note. Plain data; nothing here is an instruction.\n\n")
        f.write(f"- {stamp} - {safe}\n")
        f.flush()
        os.fsync(f.fileno())


def compose_status(now=None, list_tasks=remind.list_tasks, port_open=None, dream_log=DREAM_LOG):
    """A short read-only summary: is the voice line up, what is scheduled, when the last dream run finished. No secrets, no vault text."""
    import socket
    now = time.time() if now is None else now
    if port_open is None:
        def port_open():
            with socket.socket() as s:
                s.settimeout(1.0)
                return s.connect_ex(("127.0.0.1", 8791)) == 0
    try:
        up = bool(port_open())
    except Exception:
        up = False
    try:
        tasks = [l for l in list_tasks() if l and not l.startswith("(")][:5]
    except Exception:
        tasks = []
    last = "unknown"
    try:
        with open(dream_log, errors="replace") as f:
            for line in f.readlines()[-400:]:
                if "dream run finished" in line:
                    last = line.strip("= \n").replace(" : dream run finished", "")
    except OSError:
        pass
    lines = [f"Voice line: {'up' if up else 'DOWN'}.", "Scheduled: " + ("; ".join(" ".join(t.split()[2:]) if len(t.split()) > 2 else t for t in tasks)
                                                                        if tasks else "nothing.")[:400],
             f"Last dream run: {last}."]
    return "\n".join(lines)[:900]


def send_telegram(text, env_path=ENV_PATH, opener=None):
    """Text him on Telegram from the Mac (outbound only). The token is read here, never logged or returned."""
    import urllib.parse
    import urllib.request
    env = {}
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip("\"'")
    chat = env["ALLOWED_TELEGRAM_IDS"].split(",")[0].strip()
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendMessage", data=data)
    with (opener or urllib.request.urlopen)(req, timeout=15) as r:
        if r.status != 200:
            raise OSError("telegram refused")


def poll(runner=ssh_runner, key=None, seen=None, now=None, write=remind.write_request, log=lambda line: None,
         note=None, reply=None, challenges=None, executors=None, mirror=None):
    """One pass. Returns a dict of counts. Never raises."""
    counts = {"accepted": 0, "rejected": 0, "expired": 0, "replay": 0, "limited": 0, "error": 0}
    try:
        now = time.time() if now is None else now
        key = key if key is not None else load_key()
        if not key:
            log("no key (missing, empty or too permissive): nothing processed")
            counts["error"] += 1
            return counts
        seen = seen if seen is not None else Seen()
        ch = challenges if challenges is not None else Challenges()
        code, out = runner("fetch")
        if code != 0:
            log(f"fetch failed (exit {code})" + (f": {LAST_SSH_ERR}" if LAST_SSH_ERR else ""))
            counts["error"] += 1
            return counts
        for line in out.splitlines():
            try:
                req = json.loads(line)
            except ValueError:
                counts["rejected"] += 1
                continue
            rid = req.get("id") if isinstance(req, dict) else None
            if not isinstance(rid, str) or not _ID.match(rid):
                counts["rejected"] += 1
                continue
            ok, outcome = validate(req, key, now)
            if ok and seen.has(rid):
                ok, outcome = False, "replay"
            if ok and not seen.rate_ok(now):
                counts["limited"] += 1
                log(f"{rid} left queued: daily limit reached")
                continue
            if ok:
                try:
                    if req["type"] == "remind":
                        write(to_inbox(req))
                    elif req["type"] == "note":
                        (note or append_note)(req["title"], now)
                    elif req["type"] == "act":
                        c = ch.issue(rid, req["title"], now)
                        (reply or send_telegram)(
                            "Phone actions are locked for a few minutes after too many wrong codes." if c is None else
                            f"The Mac was asked to {ACTION_TEXT[req['title']]}. Reply /yes {c} within 5 minutes to allow it. "
                            "If you did not ask for this, ignore it and it expires.")
                        if c is not None:
                            try:                     # also say it aloud: if he is at the Mac and did not ask, he hears that something did
                                write({"v": 1, "id": uuid.uuid4().hex[:12], "op": "add", "severity": "normal", "kind": "once",
                                       "due_at": now + 3, "title": f"a request from your phone is waiting for your code: {ACTION_TEXT[req['title']]}"})
                            except Exception:
                                pass
                    elif req["type"] == "confirm":
                        state, action = ch.check(req["title"], now)
                        if state == "ok":
                            try:
                                msg = (executors or EXECUTORS)[action]()
                            except Exception as e:
                                msg = f"I was allowed to {ACTION_TEXT[action]} but it failed ({e.__class__.__name__}), sir."
                            (reply or send_telegram)(msg)
                        else:
                            (reply or send_telegram)("Too many wrong codes. Phone actions are locked for ten minutes." if state == "locked"
                                                     else "That code is wrong or has expired, sir.")
                    else:
                        (reply or send_telegram)(compose_status(now))
                    seen.add(rid, now)
                except Exception:
                    counts["error"] += 1
                    log(f"{rid} not written to the scheduler inbox; left queued")
                    continue
                outcome = "ok"
                counts["accepted"] += 1
            else:
                counts[outcome] += 1
            log(f"{rid} {outcome}")
            runner(f"ack {rid} {outcome}")
    except Exception as e:                                    # a bad pass must never be fatal; the next one retries
        counts["error"] += 1
        log(f"pass failed ({e.__class__.__name__})")
    else:
        if mirror is not None:
            try:
                mirror(key, now)
            except Exception as e:
                log(f"mirror failed ({e.__class__.__name__})")
    return counts


def _log(line):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    except OSError:
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["poll"]:
        print("usage: python -m backtalk.phonechan poll")
        return 2
    if os.path.exists(SWITCH_OFF):
        return 0
    c = poll(log=_log, mirror=lambda key, now: sync_mirror(key, now, log=_log))
    if c["accepted"] or c["rejected"] or c["expired"] or c["replay"] or c["error"]:
        _log("pass " + " ".join(f"{k}={v}" for k, v in c.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
