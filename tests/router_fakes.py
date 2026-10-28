"""Test doubles for the router: a fake SDK client and builders for the real
SDK message classes, shaped like the sequences captured in the 2026-09-30
spikes (scratchpad/spikes)."""
import asyncio

from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent,
                              SystemMessage, TextBlock, ToolResultBlock,
                              ToolUseBlock, UserMessage)
from claude_agent_sdk.types import (TaskNotificationMessage,
                                    TaskProgressMessage, TaskStartedMessage,
                                    TaskUpdatedMessage)

_EOF = object()
SID = "sess-1"


class FakeClient:
    """Stands in for ClaudeSDKClient. Counts concurrent receive_messages()
    readers so a test can prove there is exactly one."""

    def __init__(self):
        self.inbox = asyncio.Queue()
        self.sent = []                 # user-message dicts written by query()
        self.readers = 0
        self.max_readers = 0
        self.reader_calls = 0
        self.interrupts = 0
        self.stops = []
        self.disconnected = False
        self.on_interrupt = None
        self.on_stop = None
        self.on_query = None
        self.fail_disconnect = False

    async def receive_messages(self):
        self.reader_calls += 1
        self.readers += 1
        self.max_readers = max(self.max_readers, self.readers)
        try:
            while True:
                m = await self.inbox.get()
                if m is _EOF:
                    return
                if isinstance(m, BaseException):
                    raise m
                yield m
        finally:
            self.readers -= 1

    async def query(self, prompt, session_id="default"):
        async for msg in prompt:
            self.sent.append(msg)
            if self.on_query:
                self.on_query(msg)

    async def interrupt(self):
        self.interrupts += 1
        if self.on_interrupt:
            self.on_interrupt()

    async def stop_task(self, task_id):
        self.stops.append(task_id)
        if self.on_stop:
            self.on_stop(task_id)

    async def connect(self):
        self.connected = True

    async def get_context_usage(self):
        return None

    async def disconnect(self):
        if self.fail_disconnect:
            raise RuntimeError("disconnect blew up")
        self.disconnected = True

    def feed(self, *msgs):
        for m in msgs:
            self.inbox.put_nowait(m)

    def eof(self):
        self.inbox.put_nowait(_EOF)

    def uuid_of(self, i=-1):
        return self.sent[i]["uuid"]


async def settle(n=6):
    for _ in range(n):
        await asyncio.sleep(0)
    await asyncio.sleep(0.005)


# ---- message builders --------------------------------------------------------

def init():
    return SystemMessage(subtype="init", data={"type": "system", "subtype": "init"})


def echo(u, text="hi"):
    return UserMessage(content=text, uuid=u)


def asst(text, parent=None):
    return AssistantMessage(content=[TextBlock(text=text)], model="m",
                            parent_tool_use_id=parent)


def tool_use(tuid, name, inp=None, parent=None):
    return AssistantMessage(content=[ToolUseBlock(id=tuid, name=name, input=inp or {})],
                            model="m", parent_tool_use_id=parent)


def tool_result(tuid, content="ok", is_error=None, parent=None):
    return UserMessage(content=[ToolResultBlock(tool_use_id=tuid, content=content,
                                                is_error=is_error)],
                       uuid="tr-" + tuid, parent_tool_use_id=parent)


def delta(text):
    return StreamEvent(uuid="se", session_id=SID, event={
        "type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})


def result(text="", *, origin=None, num_turns=1, terminal="completed",
           is_error=False, subtype="success"):
    return ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1,
                         is_error=is_error, num_turns=num_turns, session_id=SID,
                         result=text, terminal_reason=terminal,
                         origin=({"kind": origin} if origin else None))


def fg_run(u, text, *, tools=()):
    """init -> echo -> [stream] -> assistant -> result, as captured."""
    out = [init(), echo(u), delta(text), asst(text)]
    out += list(tools)
    out.append(result(text))
    return out


def unsolicited_run(text="report"):
    return [init(), asst(text), result(text, origin="task-notification")]


def synthetic_run():
    return [init(), result("", origin="task-notification", num_turns=0)]


def task_started(tid, tuid, desc="bg", task_type=None, owned=False, is_backgrounded=None):
    data = {"type": "system", "subtype": "task_started", "task_id": tid}
    if owned:
        data["owned_by_subagent"] = True
    if is_backgrounded is not None:
        data["is_backgrounded"] = is_backgrounded
    if task_type:
        data["task_type"] = task_type
    return TaskStartedMessage(subtype="task_started", data=data, task_id=tid,
                              description=desc, uuid="u-" + tid, session_id=SID,
                              tool_use_id=tuid, task_type=task_type)


def task_progress(tid, tuid):
    return TaskProgressMessage(subtype="task_progress", data={}, task_id=tid,
                               description="p", usage={"total_tokens": 0, "tool_uses": 0,
                                                       "duration_ms": 0},
                               uuid="p-" + tid, session_id=SID, tool_use_id=tuid)


def task_note(tid, tuid, status, summary="s"):
    return TaskNotificationMessage(subtype="task_notification", data={}, task_id=tid,
                                   status=status, output_file="", summary=summary,
                                   uuid="n-" + tid + status, session_id=SID,
                                   tool_use_id=tuid)


def task_updated(tid, status):
    return TaskUpdatedMessage(subtype="task_updated", data={}, task_id=tid,
                              patch={"status": status}, status=status,
                              session_id=SID, uuid="up-" + tid + status)


def launch_bg(tid="tk1", tuid="toolu_bg1", desc="job"):
    """The captured launch: Agent(run_in_background) -> TaskStarted."""
    return [tool_use(tuid, "Agent", {"run_in_background": True, "description": desc}),
            task_started(tid, tuid, desc, task_type="local_agent"),
            tool_result(tuid, "Async agent launched")]


def seen(led, tool_use_id, decision="ALLOW", *, epoch=None, tool="", **_ignored):
    """Put an action row in the ledger the way the stream does: REQUESTED, on
    its way to a result. "DENY" and "LEGACY_ALLOWED" drive the row to the
    DENIED / ALLOWED states, which nothing writes any more but an older
    ledger.jsonl can still hold."""
    from backtalk import ledger as L
    aid = led.action_seen(tool_use_id, epoch or led.current_epoch, tool)
    if decision == "DENY":
        led.action_state(aid, L.A_DENIED)
    elif decision == "LEGACY_ALLOWED":
        led.action_state(aid, L.A_ALLOWED)
    return aid
