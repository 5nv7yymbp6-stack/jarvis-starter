"""Memory 2.0, Tier 2: an OUT-OF-BAND CONFIRMATION for writes to the real vault (Option B of the recorded decision).

WHAT THIS IS. A possession-factor confirmation: a message describing the exact write is sent to a DEDICATED Telegram bot's
private chat, and the write may proceed only if a reply of exactly `confirm <code>` comes back from the one configured
Telegram user id, in that chat, within 120 seconds of the message, and the code is bound to this exact operation, the
plan, state, context and display digests and a fresh nonce.

WHAT THIS IS NOT. It is NOT cryptographic identity authentication, and it does NOT defend against same-account code that
bypasses the committer, calls the store directly, or writes the vault files directly (the receipts lock does not exclude
non-cooperating writers). The verifier is this code, so code that edits it defeats it. It adds a third-party service and a
network dependency. Same-account code can also compose a misleading message, because the message is composed by the
program being approved. What it does stop: a script that only drives the terminal and types the answer, a stale or
replayed approval, and a write that was approved for a different plan.

BOUNDARIES (each is a refusal, never a repair; each is tested):
  * It is NOT the phone bridge: a separate bot, polled directly (no webhook, no shared secret file). The bridge runs
    Claude on allowlisted messages; a confirmation reply there would be an instruction.
  * Configuration comes from the macOS Keychain item `backtalk-memory-tier2` (JSON: token, chat_id, user_id). Missing or
    malformed means `not_configured`, and the commit refuses. The token is never logged, printed or placed in an error.
  * Read-only operations and temporary stores never construct or call this. The committer asks for it only when the store
    itself requires authorization (the real vault).
  * Every failure (no configuration, network error, a message too long to show, wrong sender, wrong chat, an invalid or
    wrong reply, an expired reply, silence) refuses the write. Updates that were already pending before the message was
    sent are never accepted.
  * `Authorization.method` is NOT changed (the closed store is untouched): the store cannot verify this confirmation, so
    the label stays what the store already knows. The evidence (the Telegram update id) is carried in the outcome.
"""
import hashlib
import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass

KEYCHAIN_SERVICE = "backtalk-memory-tier2"
WINDOW_S = 120
MAX_MESSAGE = 3800                       # Telegram's limit is 4096; a message that does not fit is refused, not cut
POLL_S = 20
_TOKEN = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{20,}\Z")
VERSION = "tier2/1"


@dataclass(frozen=True)
class Result:
    ok: bool
    code: str                    # "confirmed" or a stable refusal code
    update_id: object = None
    reply_date: object = None    # Telegram's epoch seconds for the confirming reply (None unless confirmed)
    sent_date: object = None     # Telegram's epoch seconds for our message (None if it was never sent)


def challenge_code(plan, nonce) -> str:
    """12 hex characters over everything the confirmation must be bound to. Any change to the operation, the plan, state,
    context or display digests, or the nonce, is a different code."""
    body = {"v": VERSION, "operation": plan.operation, "kind": plan.kind, "item_digest": plan.item_digest,
            "state_digest": plan.state_digest, "context_digest": plan.context_digest,
            "display_digest": plan.display_digest, "nonce": nonce}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def message_text(plan, nonce, lines) -> str:
    code = challenge_code(plan, nonce)
    head = ["MEMORY WRITE REQUEST (real vault)", ""]
    tail = ["", f"Reply exactly:  confirm {code}", f"This expires in {WINDOW_S} seconds. If you did not start this, ignore it."]
    return "\n".join(head + [str(l) for l in lines] + tail)


class TransportError(Exception):
    """Any failure to talk to the service. Never carries the URL or token."""


class TelegramTransport:
    """The three calls needed, over plain HTTPS, no webhook. `opener` is injected for tests."""

    def __init__(self, token, chat_id, opener=urllib.request.urlopen):
        self._token, self._chat, self._open = token, chat_id, opener

    def __repr__(self):
        return "TelegramTransport(<redacted>)"

    def _call(self, method, params, timeout):
        failed = None
        try:
            req = urllib.request.Request(f"https://api.telegram.org/bot{self._token}/{method}",
                                         data=json.dumps(params).encode(), headers={"Content-Type": "application/json"})
            with self._open(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            if not (isinstance(data, dict) and data.get("ok") is True):
                failed = "not_ok"
        except Exception as e:                          # the original exception (it can carry the URL) is not kept
            failed = type(e).__name__
        if failed is not None:
            raise TransportError(failed)
        return data["result"]

    def latest_update_id(self) -> int:
        res = self._call("getUpdates", {"offset": -1, "limit": 1, "timeout": 0}, 15)
        return max((u["update_id"] for u in res if isinstance(u, dict) and isinstance(u.get("update_id"), int)), default=0)

    def webhook_url(self) -> str:
        """The configured webhook URL ("" when none). `getUpdates` cannot be trusted while one is set."""
        res = self._call("getWebhookInfo", {}, 15)
        url = res.get("url") if isinstance(res, dict) else None
        if not isinstance(url, str):
            raise TransportError("bad_webhook_info")
        return url

    def send(self, text) -> dict:
        res = self._call("sendMessage", {"chat_id": self._chat, "text": text}, 15)
        return {"message_id": res["message_id"], "date": res["date"]}

    def updates(self, after_id, timeout_s) -> list:
        res = self._call("getUpdates", {"offset": after_id + 1, "timeout": int(timeout_s),
                                        "allowed_updates": ["message"]}, int(timeout_s) + 15)
        return [u for u in res if isinstance(u, dict)]


class Confirmer:
    """`confirm(plan, nonce, lines)` returns a Result and never raises for a service or reply problem."""

    def __init__(self, transport, user_id, chat_id, monotonic):
        self.transport, self.user_id, self.chat_id, self.monotonic = transport, user_id, chat_id, monotonic

    def __repr__(self):
        return "Confirmer(<redacted>)"

    def preflight(self) -> Result:
        """Fail closed if a webhook is configured on this bot (polling is then unreliable and a second consumer could
        be reading replies), or if the service cannot be asked. Sends nothing."""
        try:
            url = self.transport.webhook_url()
        except TransportError:
            return Result(False, "network")
        except Exception:
            return Result(False, "error")
        return Result(url == "", "ok" if url == "" else "webhook_configured")

    def confirm(self, plan, nonce, lines) -> Result:
        try:
            text = message_text(plan, nonce, lines)
            want = f"confirm {challenge_code(plan, nonce)}"
        except (AttributeError, TypeError, ValueError):
            return Result(False, "invalid_plan")           # a plan that cannot be bound is never confirmed
        if len(text) > MAX_MESSAGE:
            return Result(False, "message_too_long")
        try:
            baseline = self.transport.latest_update_id()          # nothing pending before the message can count
            sent = self.transport.send(text)
            sent_date = sent["date"]
            deadline = self.monotonic() + WINDOW_S
            sent_date = sent_date if type(sent_date) is int else None
            if sent_date is None:
                return Result(False, "invalid_response")
            seen, after = set(), baseline
            while True:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    return Result(False, "timeout")
                for upd in self.transport.updates(after, min(POLL_S, max(1, int(remaining)))):
                    if not isinstance(upd, dict):
                        continue                                  # junk can never confirm anything
                    uid = upd.get("update_id")
                    if not isinstance(uid, int) or uid <= baseline or uid in seen:
                        continue
                    seen.add(uid)
                    after = max(after, uid)
                    verdict = self._judge(upd, want, sent_date)
                    if verdict is not None:
                        if verdict != "confirmed":
                            return Result(False, verdict, sent_date=sent_date)
                        return Result(True, verdict, uid, upd["message"]["date"], sent_date)
        except TransportError:
            return Result(False, "network")
        except (KeyError, TypeError, ValueError, AttributeError):
            return Result(False, "invalid_response")
        except Exception:                                   # anything unexpected is still a refusal, never a pass
            return Result(False, "error")

    def _judge(self, upd, want, sent_date):
        msg = upd.get("message")
        if not isinstance(msg, dict):
            return None                                          # an edit, a channel post, ...: not a reply
        sender, chat = msg.get("from"), msg.get("chat")
        if not (isinstance(sender, dict) and isinstance(chat, dict)):
            return "invalid_response"
        sid, cid = sender.get("id"), chat.get("id")
        if type(sid) is not int or type(cid) is not int:        # a bool or a string is never an id
            return "invalid_response"
        if sid != self.user_id:
            return "wrong_sender"
        if cid != self.chat_id or chat.get("type") != "private":
            return "wrong_chat"
        date = msg.get("date")
        if type(date) is not int:
            return "invalid_response"
        if date < sent_date:
            return "invalid_response"                           # predates the challenge: not a reply to it
        if date - sent_date > WINDOW_S:
            return "expired"
        if msg.get("text") != want:
            return "invalid_response"
        return "confirmed"


def load_config(run=subprocess.run):
    """{"token", "chat_id", "user_id"} from the Keychain, or None for anything missing or malformed. The secret is read
    only here, only when a real-vault commit asks, and is never printed."""
    failed = None
    try:
        r = run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"], capture_output=True, text=True,
                timeout=10, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            return None
        cfg = json.loads(r.stdout.strip())
    except Exception:
        failed = True
    if failed or not isinstance(cfg, dict):
        return None
    token, chat_id, user_id = cfg.get("token"), cfg.get("chat_id"), cfg.get("user_id")
    ok = (isinstance(token, str) and _TOKEN.match(token) and all(
        isinstance(v, int) and not isinstance(v, bool) and v != 0 for v in (chat_id, user_id)))
    return {"token": token, "chat_id": chat_id, "user_id": user_id} if ok else None


def configured_confirmer(monotonic, *, run=subprocess.run, opener=urllib.request.urlopen):
    """A Confirmer from the Keychain configuration, or None (the caller then refuses)."""
    cfg = load_config(run)
    if cfg is None:
        return None
    return Confirmer(TelegramTransport(cfg["token"], cfg["chat_id"], opener), cfg["user_id"], cfg["chat_id"], monotonic)
