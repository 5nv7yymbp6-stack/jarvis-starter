# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import time
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

from backtalk import fetchguard
from backtalk import guard as G
from backtalk import ledger as L
from backtalk import signals
from backtalk import speech_policy as SP
from backtalk.brief import build_session_brief
from backtalk.config import CFG, DISCIPLINE
from backtalk.router import EpochManager, RouterClosed, TurnFailed
from backtalk.verify import VerifyQueue, shadow_check_foreground
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# A clause break, for the early-flush fallback below: never used unless a
# real sentence boundary is taking too long to arrive.
_CLAUSE_END = re.compile(r"(?<=[,;:—-])\s")
# How much text the FIRST chunk of a turn may accumulate before an early
# clause-boundary flush kicks in. Only ever consulted when no sentence
# boundary has appeared yet, so a short reply that ends in "." well under
# this length flushes exactly as it always did -- this is strictly an
# early-out for a long opening sentence, never an added minimum wait.
_FIRST_CHUNK_CLAUSE_CHARS = 120


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")

# ---- proactive behavior: what the MODEL is told and denied when it is on (default OFF; see proactive.py) ----------
# The 2026-10-15 defect: the model promised "checking back in ten minutes" and used the CLI's own wake-up scheduling; the
# woken turn's result had no owner and was silently quarantined. With proactive behavior on, the model schedules nothing
# itself: it queues a reminder through the remind CLI and Jarvis speaks it from the user's own words when due.
SELF_SCHEDULING_TOOLS = ("ScheduleWakeup", "CronCreate", "CronDelete", "CronList")
REMINDER_RULE = (
    "REMINDERS: when the user asks to be reminded of something, or to be told something later, never promise it yourself "
    "and never schedule a wake-up or a cron job. Queue it by running, from the backtalk folder: "
    "`.venv/bin/python -m backtalk.remind add --in 10m \"the user's words\"` (or --at 14:30, --daily 08:30, --every 2h; "
    "add --urgent only if the user asks). Then say it is QUEUED: the voice line applies it within seconds and speaks the "
    "reminder itself when it is due. `remind list` shows what is scheduled and `remind cancel <id>` cancels one."
)


def proactive_enabled() -> bool:
    try:
        from backtalk import proactive as _P
        return _P.Policy.from_config(CFG.get("proactive")).enabled
    except Exception:
        return False



def first_event_line(h, ev) -> str:
    """One log line for the first stream event of a turn (the boundary the existing
    `[timing] claude first token` line already measures). Only identifiers, the event
    and delta TYPE names, and times; never any text. Never raises."""
    try:
        now = time.monotonic()
        sa, ca = getattr(h, "sent_at", None), getattr(h, "claimed_at", None)
        et = (ev or {}).get("type")
        dt = ((ev or {}).get("delta") or {}).get("type")
        return (f"[req] first stream event turn={h.turn_id} uuid={h.uuid[:8]} "
                f"event={et if isinstance(et, str) and et.replace('_', '').isalnum() else '?'} "
                f"delta={dt if isinstance(dt, str) and dt.replace('_', '').isalnum() else '?'} "
                f"since_submit={('%.2fs' % (now - sa)) if sa else 'n/a'} "
                f"claim_to_first={('%.2fs' % (now - ca)) if ca else 'n/a'}")
    except Exception:
        return "[req] first stream event (unavailable)"


class WarmBrain:
    def __init__(self, model: str | None = None,
                 resume_id: str | None = None, ledger=None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the first start (config key
        # resume_last_session). Consumed on use. A rebuild sets it again to
        # the current conversation, so the rebuild keeps the memory.
        # (Community proposal, issue #1.)
        self._resume_id = resume_id
        # THE ROUTER (router.py). There is no "_dirty" any more: the stream
        # has exactly one reader (the router's pump) for the life of an epoch,
        # every message is owned by a claimed turn, a background task or the
        # quarantine, and a result is never paired with a query by arriving
        # next. A query is correlated by the uuid it was sent with.
        bg = CFG.get("background") or {}
        self.ledger = ledger or L.Ledger()
        self.mgr = EpochManager(
            self.ledger,
            protect_background=bool(bg.get("protect_on_barge_in", True)),
            echo_timeout_s=float(bg.get("echo_timeout_s", 20)),
            barge_escalate_s=float(bg.get("barge_escalate_s", 20)))
        gcfg = CFG.get("guard") or {}
        self.guard = G.Guard(say=lambda t: self.on_say and self.on_say(t), timeout_s=float(gcfg.get("timeout_s", 30)),
                             enabled=bool(gcfg.get("enabled", True)))
        self.router = None
        self.verify = VerifyQueue(self.ledger)
        self.verify.on_done = self._on_verified
        self.speech = SP.SpeechPolicy(self.ledger)
        self.mgr.tasks.listeners.append(self._on_task_terminal)
        # The CLI hands the foreground model every background agent's RAW final
        # text (a <task-notification> prompt), then runs an unsolicited model
        # turn on it that stays in the conversation. Withheld by default: see
        # _gate_prompt.
        self._withhold_raw_reports = bool(bg.get("withhold_raw_reports", True))
        # set by main: speech context for the policy, and how to say a line
        self.get_speech_context = None      # callable -> speech_policy.Context
        self.on_say = None                  # callable(str)
        self._need_rebuild: str | None = None
        self.force_interrupt = False        # "stop everything": interrupt even with protected tasks
        # The current conversation, kept whatever resume_last_session says:
        # the audio watchdog resumes it after restarting the process.
        self.session_id: str | None = None

    async def start(self):
        resume, self._resume_id = self._resume_id, None   # consume once
        # The session brief rides in the system prompt, built ONCE per
        # start (fresh, resumed, or desync rebuild), never per turn and
        # never into the transcript. Context only; see brief.py.
        try:
            brief = build_session_brief()
        except Exception:
            brief = ""
        append = DISCIPLINE + ("\n\n" + brief if brief else "")
        proactive_on = proactive_enabled()
        if proactive_on:
            append += "\n\n" + REMINDER_RULE

        hooks = {}
        if self._withhold_raw_reports:
            hooks["UserPromptSubmit"] = [HookMatcher(hooks=[self._gate_prompt], timeout=10.0)]

        hooks["PreToolUse"] = [HookMatcher(matcher="Bash", hooks=[self.guard.pre_tool], timeout=self.guard.timeout_s + 10.0)]
        # a blocked WebFetch points the model at the browser (fetchguard.py)
        hooks["PostToolUse"] = [HookMatcher(matcher="WebFetch", hooks=[fetchguard.post_tool], timeout=5.0)]
        hooks["PostToolUseFailure"] = [HookMatcher(matcher="WebFetch", hooks=[fetchguard.post_tool_failure], timeout=5.0)]

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": append},
                include_partial_messages=True,
                # Tool calls run directly: no approval step anywhere.
                permission_mode="bypassPermissions",
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                resume=rid,
                # ask the CLI to echo each user message back with the uuid we
                # sent it with: that echo is what claims a run for a turn
                # (router.py). Verified in scratch 2026-09-30.
                extra_args={"replay-user-messages": None},
                hooks=hooks or None,
                disallowed_tools=list(SELF_SCHEDULING_TOOLS) if proactive_on else [],
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                await self._open_epoch("resume")
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()
        await self._open_epoch("fresh" if not resume else "fresh_after_failed_resume")

    @staticmethod
    def _cli_pid(client):
        """The CLI process id, for read-only process evidence. Reaches a
        private attribute of the SDK, so it is best effort: None on any
        failure and everything else keeps working."""
        try:
            return int(client._transport._process.pid)
        except Exception:
            return None

    async def _open_epoch(self, reason):
        self.router = await self.mgr.open(
            self._client, session_id=self.session_id, reason=reason,
            cli_pid=self._cli_pid(self._client))
        self.router.on_lost = self._on_router_lost
        self.router.action_listeners.append(self._on_action)
        self._need_rebuild = None
        log(f"[brain] epoch {self.router.epoch} open ({reason}); "
            f"one stream reader")
        try:                                     # outcomes whose hand-off was never recorded ride the next note
            n = self.speech.scan_undelivered()
            if n:
                log(f"[ledger] {n} outcome(s) with no recorded hand-off queued for the next turn")
        except Exception as e:
            log(f"[ledger] undelivered-outcome scan failed: {type(e).__name__}")

    # ---- router callbacks (all synchronous, on the event loop) -------------

    def _on_action(self, row, tool_input):
        """A tool result came back. Verification is queued, never awaited."""
        self.verify.submit(row, tool_input)

    def _on_verified(self, aid):
        row = self.ledger.actions.get(aid)
        if row:
            self.speech.on_action_bad_news(row)

    def _on_task_terminal(self, row):
        """A background task reached a terminal state. Whatever is said about
        it is built from the ledger by the speech policy, never taken from the
        model's own report."""
        try:
            ctx = self.get_speech_context() if self.get_speech_context else SP.Context()
            if row.get("state") == L.K_STOPPED_USER:
                # The user asked for this. Confirm it once, from the ledger,
                # only now that the stop has actually been seen. "Stop
                # everything" says its own single line instead.
                if row.get("cancel_intent") == "STOP_ALL":
                    return
                cand = SP.Candidate(SP.BG_STOP_CONFIRM, "code", row["id"])
                if self.speech.decide(cand, ctx) == SP.SPEAK and self.on_say:
                    self.on_say(SP.task_line(row))
                    self.ledger.mark_spoken("task", row["id"], "bg_stop_confirm", "spoken")
                return
            line = self.speech.on_task_terminal(row, ctx)
            if line and self.on_say:
                self.on_say(line)
                self.ledger.mark_spoken("task", row["id"], "bg_completion", "spoken")
        except Exception as e:
            log(f"[brain] task terminal handling failed: {e!r}")

    def _on_router_lost(self, cause):
        text = str(cause)
        self._need_rebuild = "oversize" if "maximum buffer size" in text else "stream_lost"
        log(f"[brain] stream lost ({self._need_rebuild}); will rebuild before the next turn")
        for a in self.ledger.actions.values():
            if a.get("epoch") == (self.router.epoch if self.router else None) \
                    and a.get("state") == L.A_LOST:
                self.speech.on_action_bad_news(a)

    async def _gate_prompt(self, input_data, tool_use_id, context):
        """UserPromptSubmit hook: THE FOREGROUND TRUST BOUNDARY.

        A finished background agent's raw report is model-written text about what
        its tools returned, and the CLI would put it straight into the foreground
        model's context (a <task-notification> prompt with a <result> block) and
        let the model answer it. Whatever number is in it then reads as fact: on
        2026-10-01 the foreground model told the user the agent "correctly
        reported 20" about a find that returned 21. So the raw notice is BLOCKED
        here (the CLI then neither shows it to the model nor starts a model turn
        on it), and the model is told about the task only through take_notes():
        built from the ledger, carrying the verification state and the tool's
        own number, never the report's.

        Only notices for background AGENTS (or tasks this ledger does not know,
        which fails closed) are withheld; a background shell command's output
        and every ordinary prompt pass untouched. A hook that raises makes the
        CLI run the prompt, so nothing here can raise."""
        try:
            prompt = str((input_data or {}).get("prompt") or "")
            if not SP.is_task_notification(prompt):
                return {}
            ids = SP.task_notice_ids(prompt)
            rows = [self.ledger.tasks.get(i) for i in ids]
            others = bool(ids) and all(r and r.get("task_kind") != L.KIND_BG_AGENT for r in rows)
            if others:
                return {}                                  # a shell task's output, not an agent's report
            saved = self._save_report(ids, prompt)
            self.ledger.note("raw_task_report_withheld", task_ids=ids, bytes=len(prompt),
                             saved=bool(saved), epoch=self.router.epoch if self.router else None)
            log(f"[brain] raw background report withheld from the foreground model "
                f"(tasks {ids or '?'}); it is told through the task ledger instead")
            return {"decision": "block",
                    "reason": "withheld: background results reach the model through the task ledger"}
        except Exception as e:
            log(f"[brain] prompt gate error {type(e).__name__}")
            try:
                if SP.is_task_notification((input_data or {}).get("prompt")):
                    return {"decision": "block", "reason": "withheld: gate error"}
            except Exception:
                pass
            return {}

    def _save_report(self, ids, prompt):
        """Keep the withheld report on disk so it is not lost (2026-10-25: Yousif
        asked twice what a finished agent found and Jarvis had nothing, because the
        report was blocked and TaskOutput cannot fetch an agent's result). The
        foreground model is pointed at this file by take_notes(), with the
        report's numbers marked as the agent's claims. Never raises."""
        try:
            body = SP.task_notice_result(prompt)
            if not body:
                return None
            d = self.ledger.dir / "reports"
            d.mkdir(parents=True, exist_ok=True)
            path = None
            for tid in (ids or [f"unknown-{int(time.time())}"]):
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(tid))[:80]
                path = d / f"{safe}.md"
                path.write_text(body, encoding="utf-8")
                row = self.ledger.tasks.get(tid)
                if row is not None:
                    row["report_path"] = str(path)
            return path
        except Exception as e:
            log(f"[brain] could not save background report: {type(e).__name__}")
            return None

    async def _ensure_router(self):
        if self._client is None:
            await self.start()
        elif self._need_rebuild or self.router is None or not self.router.usable:
            await self.rebuild(self._need_rebuild or "router_unusable")

    async def rebuild(self, reason):
        """Close this session cleanly and open a new one on the SAME
        conversation. The order lives in EpochRouter.close(): mark closing,
        abort turns, disconnect, judge the tasks, write CLOSED_CLEAN."""
        log(f"[brain] rebuilding the session ({reason}), reopening the same conversation")
        old, client = self.router, self._client
        self._resume_id = self.session_id
        self._client = None
        self.router = None
        try:
            if old is not None:
                await old.close(reason, client.disconnect if client else None)
            elif client is not None:
                await client.disconnect()
        except Exception as e:
            log(f"[brain] close during rebuild failed: {e!r}")
        await self.start()

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        sid = getattr(rm, "session_id", None)
        if sid:
            if sid != self.session_id and self.router is not None:
                try:
                    self.ledger.epoch_set(self.router.epoch, session_id=sid)
                except Exception:
                    pass
            self.session_id = sid
        if not CFG.get("resume_last_session") or not sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) and return whatever text the CLI answered with
        (confirmations, errors). Slash-command replies arrive as COMPLETE
        AssistantMessages, not stream deltas, and BEFORE their uuid echo, so a
        command turn (and only a command turn) may claim its run retroactively
        (router.py). Bounded: an unanswered command must not deafen the voice
        loop."""
        await self._ensure_router()
        h = await self.router.send(cmd, kind="command")
        texts = []

        async def _collect():
            async for tag, msg in h.events():
                if tag == "msg" and type(msg).__name__ == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif tag == "result":
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            await self.router.interrupt(reason="COMMAND_TIMEOUT", force=True)
            return "error: the command timed out"
        except TurnFailed as e:
            return f"error: {e.reason}"
        return " ".join(texts).strip()

    async def interrupt(self, reason="BARGE_IN"):
        """Cancel the foreground run. interrupt() also kills every background
        task, so with live protected tasks the router mutes the run and lets
        it finish (escalating after a bound); force_interrupt (set for "stop
        everything") overrides that. Returns the router's verdict."""
        r = self.router
        if r is None or not r.usable:
            return "none"
        force, self.force_interrupt = self.force_interrupt, False
        res = await r.interrupt(reason=("STOP_ALL" if force else reason), force=force,
                                wait=5.0)
        if res == "timeout":
            # the aborted result never came: the run cannot be trusted to end
            self._need_rebuild = "interrupt_timeout"
            log("[brain] interrupt was not acknowledged; will rebuild before the next turn")
        return res

    async def reset_turn(self, timeout: float = 8.0):
        """Called before a new turn goes out. It used to interrupt and drain a
        shared pipe so the next query would not pair with a dead turn's
        leftovers. The router makes that impossible by construction (a query
        is claimed by its uuid, not by order), so this now only makes sure the
        session underneath is usable, rebuilding it if the stream was lost."""
        if self._client is None:
            return
        if self._need_rebuild or self.router is None or not self.router.usable:
            await self.rebuild(self._need_rebuild or "router_unusable")

    async def stop(self):
        if self.router is not None:
            client, self._client = self._client, None
            try:
                await self.router.close("quit", client.disconnect if client else None)
            finally:
                self.router = None
        elif self._client:
            await self._client.disconnect()
            self._client = None

    # ---- background work: what the voice controls call ---------------------

    def live_background(self):
        return self.router.live_bg() if self.router is not None else []

    async def stop_background(self, task_id):
        """Scoped stop: one task, the foreground run untouched."""
        return await self.router.stop_task(task_id, "USER_STOP")

    async def stop_everything(self):
        """(tasks stopped, interrupt verdict). Each task is stopped by name
        first so the ledger records who stopped it, then the run is
        interrupted."""
        if self.router is None:
            return 0, "none"
        return await self.router.stop_all("STOP_ALL")

    def note_prefix(self):
        return self.speech.note_prefix()

    async def ask_stream(self, utterance: str, on_tool=None, silent: bool = False):
        """Yield complete sentences as they stream out of the model.

        The turn is a TurnHandle: only messages from the run that this
        query's own uuid claimed ever reach it (router.py). `on_tool(name)`
        fires as each tool call starts, so the caller can tell Yousif what
        the turn is ACTUALLY doing during a long silent stretch instead of
        guessing. Never allowed to break a turn.

        If a background task's notification was folded into this run (the CLI
        does that), the rest of the answer is held to the end so a task-stop
        notice cannot be spoken as part of it."""
        await self._ensure_router()
        t0 = time.monotonic()
        first_token_logged = False
        # True once anything has been yielded this turn. The early
        # clause-boundary flush below is an optimization for a long FIRST
        # sentence only -- after the first chunk is out, later sentences go
        # back to full-sentence-only flushing, same as before this change.
        flushed_once = False
        h = await self.router.send(utterance, kind="foreground", silent=silent)
        buf = ""
        out: list[str] = []            # sentences ready to hand up
        said: list[str] = []           # everything handed up, for the log-only number check at the end (G-03)
        spoken_held = False

        def emit(x):
            nonlocal flushed_once
            flushed_once = True
            out.append(x)

        def ended_early() -> bool:
            """The router ended this turn as ABORTED or INTERRUPTED (superseded, stopped, cut in on): whatever of the answer is still
            buffered belongs to an answer that no longer stands and must never be spoken (audit finding F-15)."""
            return h.state in (L.T_ABORTED, L.T_INTERRUPTED) or bool(getattr(h, "abandoned", False))   # abandoned = a protected barge-in muted it

        def drop_unspoken():
            nonlocal buf
            dropped = len(buf.strip()) + sum(len(x) for x in out)
            buf = ""
            out.clear()
            if dropped:
                self.ledger.note("aborted_answer_tail_dropped", turn_id=h.turn_id, state=h.state, chars=dropped)

        async for tag, msg in h.events():
            if tag == "msg":
                t = type(msg).__name__
                if t == "StreamEvent":
                    ev = getattr(msg, "event", {}) or {}
                    if ev.get("type") == "content_block_delta":
                        if not first_token_logged:
                            first_token_logged = True
                            log(f"[timing] claude first token {time.monotonic() - t0:.2f}s")
                            log(first_event_line(h, ev))      # observability only: same instant, extra detail
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            buf += delta.get("text", "")
                            # emit any complete sentences
                            while True:
                                m = _SENTENCE_END.search(buf)
                                if not m:
                                    break
                                sentence, buf = (buf[:m.end()].strip(),
                                                 buf[m.end():])
                                if sentence:
                                    emit(sentence)
                            # Early-flush fallback: no sentence boundary yet, but
                            # this is the still-unflushed FIRST chunk and it has
                            # run long enough that a clause boundary is a better
                            # bet than waiting for the model to finish the whole
                            # sentence. Never fires for a short reply -- a normal
                            # "." well under _FIRST_CHUNK_CLAUSE_CHARS is caught
                            # by the sentence check above before this is reached.
                            if not flushed_once and len(buf) >= _FIRST_CHUNK_CLAUSE_CHARS:
                                m = None
                                for m in _CLAUSE_END.finditer(buf):
                                    pass    # last clause boundary in the buffer
                                if m:
                                    clause, buf = (buf[:m.end()].strip(),
                                                  buf[m.end():])
                                    if clause:
                                        emit(clause)
                    elif ev.get("type") == "content_block_start" and on_tool:
                        # A tool is starting: the turn is about to go quiet for
                        # a while. Guarded because this reaches into the raw
                        # stream shape, which is not ours to depend on.
                        try:
                            blk = ev.get("content_block", {}) or {}
                            if blk.get("type") == "tool_use":
                                on_tool(blk.get("name") or "")
                        except Exception:
                            pass
                    elif ev.get("type") == "content_block_stop":
                        # End of a speech block (e.g. right before a tool
                        # call): flush NOW. Without this, pre-tool filler
                        # ("On it — let me grab that.") sits silent in the
                        # buffer through the whole tool run, then plays
                        # glued to the answer: long dead air, then two
                        # thoughts at once.
                        tail = buf.strip()
                        buf = ""
                        if tail:
                            emit(tail)
            elif tag == "result":
                self._tally(msg)
                self._remember_session(msg)
                await self._pull_rate_limits()
            # Hand sentences up as they are ready, unless a task notification
            # was folded into this run: then hold the rest until the run ends.
            if not h.contaminated_by:
                while out:
                    if ended_early():
                        drop_unspoken()
                        return
                    said.append(out[0])
                    yield out.pop(0)
            elif not spoken_held:
                spoken_held = True
                log(f"[brain] answer held: task notification folded into this run "
                    f"({h.contaminated_by})")
        if ended_early():
            drop_unspoken()
            return
        tail = buf.strip()
        if tail:
            out.append(tail)
        if h.contaminated_by and out:
            mentions = self._mentions_background(" ".join(out), h.contaminated_by)
            self.ledger.note("held_answer_released", turn_id=h.turn_id,
                             task_ids=h.contaminated_by, mentions_background=mentions)
        while out:
            if ended_early():
                drop_unspoken()
                return
            said.append(out[0])
            yield out.pop(0)
        # The turn finished normally and everything it said has been handed up: compare its numbers with its own tool counts.
        # Log only; nothing here can change, hold or delay speech (G-03 option B).
        if said and not silent:
            shadow_check_foreground(self.ledger, h.turn_id, " ".join(said), utterance)

    def _mentions_background(self, text, task_ids):
        """Does a held answer talk about the task that was folded into it?
        Audit only: it is recorded, not used to edit what was said."""
        low = text.lower()
        for tid in task_ids:
            row = self.ledger.tasks.get(tid) or {}
            d = str(row.get("description") or "").lower()
            if str(tid).lower() in low or (d and d in low):
                return True
        return False


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
