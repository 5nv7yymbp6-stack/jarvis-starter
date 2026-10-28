"""Passive SDK/CLI observability (router.py, brain.py, 2026-10-06).

A first-token wait of 9 to 27 s was traced to the span between the CLI accepting a message and
the first stream event, and the router dropped every non-init system message and rate-limit
event, so a retry/backoff or a throttle would have been invisible. These tests pin what is now
LOGGED and, as importantly, that nothing about routing, claiming, retries or results changed,
and that no prompt text, transcript, credential or free-text error body can reach the log.

Fake SDK client, real router. Nothing here touches the live voice process or its files."""
import asyncio
import inspect
import unittest
from unittest import mock

from claude_agent_sdk import RateLimitEvent, RateLimitInfo, SystemMessage

from backtalk import brain as B
from backtalk import ledger as L
from backtalk import router as R
from tests import router_fakes as F
from tests.router_fakes import settle
from tests.test_router import Base

SECRETS = ["sk-ant-api03-SECRETSECRETSECRETSECRETSECRETSECRETSECRET", "Bearer abc.def.ghi",
           "session=Cookie123", "hunter2-password", "the user said launch the missiles",
           "Overloaded: please retry, request id req_01SECRETBODY", "/Users/yousif/.ssh/id_rsa"]


def sysmsg(sub, **data):
    return SystemMessage(subtype=sub, data=data)


def line_of(d):
    return str(d)


class DescribeSystemTests(unittest.TestCase):
    def test_a_status_requesting_message_is_described_with_its_state(self):
        d = R.describe_sdk_system(sysmsg("status", status="requesting", uuid="u-1",
                                         session_id="82e10862-a524-45b9-b2ee-391c41a42e82"))
        self.assertEqual(d["subtype"], "status")
        self.assertEqual(d["fields"], {"status": "requesting"})
        self.assertEqual(d["session"], "82e10862")                 # a short prefix, never the full id
        self.assertEqual(d["retry"], "none_exposed")               # said plainly, not invented

    def test_retry_and_backoff_fields_are_recorded_directly(self):
        d = R.describe_sdk_system(sysmsg("api_retry", attempt=2, max_retries=10, retry_delay_ms=2000,
                                         error_status=529, retryable=True))
        self.assertEqual(d["fields"], {"attempt": "2", "max_retries": "10", "retry_delay_ms": "2000",
                                       "error_status": "529", "retryable": "true"})
        self.assertEqual(d["retry"], "present")

    def test_rate_limit_and_overload_style_fields_are_recorded(self):
        d = R.describe_sdk_system(sysmsg("rate_limited", retry_after=12.5, status_code=429, error_type="rate_limit_error",
                                         backoff_ms=4000))
        self.assertEqual(d["fields"]["retry_after"], "12.5")
        self.assertEqual(d["fields"]["status_code"], "429")
        self.assertEqual(d["fields"]["error_type"], "rate_limit_error")
        self.assertEqual(d["retry"], "present")

    def test_an_unmodelled_retry_field_is_visible_by_name_not_by_value(self):
        d = R.describe_sdk_system(sysmsg("api_retry", next_retry_in_flight_ms=900, note="free text"))
        self.assertEqual(d["retry"], "present")                    # the NAME says retry
        self.assertIn("next_retry_in_flight_ms", d["keys"])
        self.assertNotIn("free text", line_of(d))

    def test_init_is_not_described(self):
        self.assertIsNone(R.describe_sdk_system(sysmsg("init", apiKeySource="env", tools=["Bash"], cwd="/x")))

    def test_secrets_prompts_and_error_bodies_never_reach_the_description(self):
        d = R.describe_sdk_system(sysmsg(
            "api_retry", attempt=1, error=SECRETS[5], message=SECRETS[4], prompt=SECRETS[4], text=SECRETS[4],
            content=SECRETS[4], apiKey=SECRETS[0], api_key=SECRETS[0], authorization=SECRETS[1],
            headers={"Authorization": SECRETS[1], "Cookie": SECRETS[2]}, cookie=SECRETS[2], token=SECRETS[3],
            password=SECRETS[3], cwd=SECRETS[6], path=SECRETS[6], tools=["Bash", SECRETS[6]], reason=SECRETS[5],
            error_type=SECRETS[0], status=SECRETS[1], category="Mixed Case Words With Spaces"))
        out = line_of(d)
        for secret in SECRETS:
            self.assertNotIn(secret, out)
        for frag in ("SECRET", "Bearer", "Cookie123", "hunter2", "missiles", "id_rsa"):
            self.assertNotIn(frag, out)
        # allowlisted keys that were handed text are marked, not echoed
        self.assertEqual(d["fields"]["reason"], "<text>")
        self.assertEqual(d["fields"]["error_type"], "<text>")
        self.assertEqual(d["fields"]["status"], "<text>")
        self.assertEqual(d["fields"]["category"], "<text>")
        self.assertEqual(d["fields"]["attempt"], "1")

    def test_a_string_that_starts_like_a_key_or_holds_digits_is_not_logged_even_if_it_looks_like_a_word(self):
        for v in ("sk-live-abc", "hunter-two2", "abc123", "Mixed", "has space", "x" * 41, ""):
            d = R.describe_sdk_system(sysmsg("x", reason=v))
            self.assertEqual(d["fields"]["reason"], "<text>", v)
        for v in ("requesting", "allowed_warning", "rate_limit_error", "overloaded_error", "five_hour", "skipped"):
            self.assertEqual(R.describe_sdk_system(sysmsg("x", status=v))["fields"]["status"], v)

    def test_missing_odd_or_hostile_shapes_do_not_crash(self):
        class Bare: pass
        for m in (SystemMessage(subtype="status", data={}), SystemMessage(subtype="status", data=None),
                  SystemMessage(subtype=None, data={"status": "requesting"}), SystemMessage(subtype="x", data=[1, 2]),
                  SystemMessage(subtype="x", data={1: "a", None: "b", "k" * 80: 1, "ok": object()}),
                  Bare(), None, SystemMessage(subtype="x" * 90, data={"status": 5})):
            R.describe_sdk_system(m)                                # must not raise
        d = R.describe_sdk_system(SystemMessage(subtype="x", data={"k" * 80: 1, "estimated_tokens": 3.0, "retryable": False}))
        self.assertEqual(d["fields"]["retryable"], "false")
        self.assertEqual(d["fields"]["estimated_tokens"], "3")
        self.assertNotIn("k" * 80, d["keys"])
        self.assertEqual(R.describe_sdk_system(SystemMessage(subtype="y" * 90, data={}))["subtype"], "?")

    def test_the_list_of_other_key_names_is_bounded(self):
        d = R.describe_sdk_system(sysmsg("x", **{f"k{i}": i for i in range(60)}))
        self.assertLessEqual(len(d["keys"]), 14)


class DescribeRateLimitTests(unittest.TestCase):
    def ev(self, **kw):
        info = RateLimitInfo(status=kw.pop("status", "allowed"), raw=kw.pop("raw", {}), **kw)
        return RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s")

    def test_the_modelled_fields_are_recorded(self):
        d = R.describe_sdk_rate_limit(self.ev(status="allowed_warning", rate_limit_type="five_hour", utilization=0.74,
                                              resets_at=1791270600, overage_status="rejected",
                                              overage_disabled_reason="org_level_disabled"))
        self.assertEqual(d["fields"], {"status": "allowed_warning", "rate_limit_type": "five_hour", "utilization": "0.74",
                                       "resets_at": "1791270600", "overage_status": "rejected",
                                       "overage_disabled_reason": "org_level_disabled"})

    def test_unmodelled_raw_fields_are_listed_by_name_only(self):
        d = R.describe_sdk_rate_limit(self.ev(raw={"unifiedWindows": {"x": SECRETS[0]}, "isUsingOverage": SECRETS[2],
                                                   "Weird Key": 1}))
        self.assertEqual(d["keys"], ["isUsingOverage", "unifiedWindows"])
        for secret in SECRETS:
            self.assertNotIn(secret, line_of(d))

    def test_missing_fields_and_garbage_do_not_crash(self):
        self.assertEqual(R.describe_sdk_rate_limit(self.ev())["fields"]["status"], "allowed")
        for m in (None, object(), RateLimitEvent(rate_limit_info=None, uuid="u", session_id="s")):
            R.describe_sdk_rate_limit(m)


class RouterLoggingTests(Base):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.lines = []
        p = mock.patch.object(R, "log", side_effect=lambda s: self.lines.append(s))
        p.start()
        self.addCleanup(p.stop)

    def sdk(self):
        return [l for l in self.lines if l.startswith("[sdk]")]

    def req(self):
        return [l for l in self.lines if l.startswith("[req]")]

    async def test_a_non_init_system_message_is_now_observable_with_epoch_and_time_since_submit(self):
        h = await self.r.send("hello")
        self.client.feed(F.init(), sysmsg("status", status="requesting", session_id="82e10862-a524-45b9-b2ee"), F.echo(h.uuid))
        await settle()
        got = self.sdk()
        self.assertEqual(len(got), 1)
        self.assertIn("subtype=status", got[0])
        self.assertIn("status=requesting", got[0])
        self.assertIn(f"epoch={self.r.epoch}", got[0])
        self.assertIn("since_submit=", got[0])
        self.assertIn("retry_metadata=none_exposed", got[0])

    async def test_init_is_handled_exactly_as_before_and_logs_no_sdk_line(self):
        await self.r.send("hello")
        self.client.feed(F.init())
        await settle()
        self.assertTrue(self.r.run is not None and self.r.run.saw_init)
        self.assertEqual(self.sdk(), [])

    async def test_a_retry_message_is_logged_with_its_real_fields(self):
        await self.r.send("hello")
        self.client.feed(F.init(), sysmsg("api_retry", attempt=3, max_retries=10, retry_delay_ms=4000, error_status=529))
        await settle()
        line = self.sdk()[0]
        for frag in ("subtype=api_retry", "attempt=3", "max_retries=10", "retry_delay_ms=4000", "error_status=529",
                     "retry_metadata=present"):
            self.assertIn(frag, line)

    async def test_a_burst_of_thinking_tokens_is_throttled_to_about_one_line(self):
        await self.r.send("hello")
        self.client.feed(F.init(), *[sysmsg("thinking_tokens", estimated_tokens=i * 10, estimated_tokens_delta=10)
                                     for i in range(1, 8)])
        await settle()
        self.assertEqual(len([l for l in self.sdk() if "thinking_tokens" in l]), 1)

    async def test_a_rate_limit_event_is_observable(self):
        await self.r.send("hello")
        info = RateLimitInfo(status="allowed_warning", rate_limit_type="five_hour", utilization=0.8, raw={"unifiedWindows": {}})
        self.client.feed(F.init(), RateLimitEvent(rate_limit_info=info, uuid="u", session_id="s"))
        await settle()
        line = [l for l in self.sdk() if "rate_limit" in l][0]
        for frag in ("status=allowed_warning", "rate_limit_type=five_hour", "utilization=0.8", "raw_keys=unifiedWindows"):
            self.assertIn(frag, line)

    async def test_a_message_with_secrets_logs_none_of_them(self):
        await self.r.send("hello")
        self.client.feed(F.init(), sysmsg("api_retry", attempt=1, error=SECRETS[5], apiKey=SECRETS[0], authorization=SECRETS[1],
                                         headers={"Cookie": SECRETS[2]}, reason=SECRETS[3]))
        await settle()
        blob = "\n".join(self.lines)
        for secret in SECRETS:
            self.assertNotIn(secret, blob)

    async def test_the_submit_boundary_is_logged_before_the_query_and_never_contains_the_text(self):
        seen = []
        self.client.on_query = lambda msg: seen.append(list(self.req()))
        text = "a very private sentence about the deal"
        h = await self.r.send(text)
        self.assertEqual(len(seen[0]), 1)                          # already logged when the SDK got the query
        line = seen[0][0]
        self.assertIn(f"turn={h.turn_id}", line)
        self.assertIn(f"uuid={h.uuid[:8]}", line)
        self.assertIn(f"chars={len(text)}", line)
        self.assertNotIn("private", "\n".join(self.lines))

    async def test_the_echo_is_logged_at_the_claim_with_time_since_submit(self):
        h = await self.r.send("hello")
        self.client.feed(F.init(), F.echo(h.uuid))
        await settle()
        echo = [l for l in self.req() if l.startswith("[req] echo")]
        self.assertEqual(len(echo), 1)
        self.assertIn(f"turn={h.turn_id}", echo[0])
        self.assertIn("since_submit=", echo[0])

    async def test_a_turn_merged_into_an_abandoned_run_is_marked_in_the_echo_line(self):
        a = await self.r.send("first")
        self.client.feed(F.init(), F.echo(a.uuid))
        await settle()
        a.abandoned = True                                         # the owner is no longer wanted: the next turn takes the run
        b = await self.r.send("second")
        self.client.feed(F.echo(b.uuid))
        await settle()
        self.assertEqual(b.state, L.T_CLAIMED)
        mine = [l for l in self.req() if l.startswith("[req] echo") and f"turn={b.turn_id}" in l]
        self.assertEqual(len(mine), 1)
        self.assertIn("merged_into_run", mine[0])

    async def test_a_logging_failure_cannot_stop_a_query_or_a_dispatch(self):
        def boom(s):
            raise OSError("disk full")
        with mock.patch.object(R, "log", side_effect=boom):
            h = await self.r.send("hello")                          # the submit log raises: the send still goes out
            self.client.feed(F.init(), sysmsg("status", status="requesting"),
                             sysmsg("api_retry", attempt=1), *F.fg_run(h.uuid, "fine")[1:])
            await settle()
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(h.state, L.T_DONE)
        self.assertEqual(self.result_text(h), "fine")

    async def test_interleaved_system_messages_change_nothing_about_the_turn(self):
        a = await self.r.send("one")
        self.client.feed(*F.fg_run(a.uuid, "answer"))
        await settle()
        plain = (a.state, self.result_text(a), dict(self.mgr.quarantine.counts), self.r.stats["duplicate_echo"])

        client2 = F.FakeClient()
        r2 = await self.mgr.open(client2, session_id="s2", cli_pid=None)
        self.extra.append(r2)
        b = await r2.send("one")
        run = F.fg_run(b.uuid, "answer")
        client2.feed(run[0], sysmsg("status", status="requesting"), sysmsg("thinking_tokens", estimated_tokens=5),
                     RateLimitEvent(rate_limit_info=RateLimitInfo(status="allowed"), uuid="u", session_id="s"), *run[1:])
        await settle()
        noisy = (b.state, b.result.result if b.result else None, dict(self.mgr.quarantine.counts), r2.stats["duplicate_echo"])
        self.assertEqual(noisy[:2], plain[:2])
        self.assertEqual(noisy[3], plain[3])
        self.assertEqual(noisy[2].get("unknown_message_type", 0), plain[2].get("unknown_message_type", 0))

    async def test_no_retry_or_recovery_logic_was_added(self):
        src = inspect.getsource(R.EpochRouter._observe_system) + inspect.getsource(R.EpochRouter._observe_rate_limit) \
            + inspect.getsource(R.EpochRouter._log_claim)
        for banned in ("interrupt", "send(", "finish(", "ledger", "_lost", "sleep", "create_task", "raise "):
            self.assertNotIn(banned, src, banned)


class FirstEventLineTests(unittest.TestCase):
    class H:
        turn_id = "T123"
        uuid = "abcdef12-0000"
        sent_at = None
        claimed_at = None

    def test_the_line_has_identifiers_event_and_delta_types_and_times_only(self):
        import time
        h = self.H()
        h.sent_at = time.monotonic() - 2.5
        h.claimed_at = time.monotonic() - 0.5
        line = B.first_event_line(h, {"type": "content_block_delta",
                                      "delta": {"type": "text_delta", "text": "the private reply text"}})
        for frag in ("turn=T123", "uuid=abcdef12", "event=content_block_delta", "delta=text_delta", "since_submit=2.5", "claim_to_first=0.5"):
            self.assertIn(frag, line)
        self.assertNotIn("private", line)

    def test_missing_times_and_odd_events_do_not_crash(self):
        h = self.H()
        self.assertIn("since_submit=n/a", B.first_event_line(h, {}))
        self.assertIn("event=?", B.first_event_line(h, {"type": "has space; or text"}))
        self.assertIn("delta=?", B.first_event_line(h, {"type": "content_block_delta", "delta": {"type": "<script>"}}))
        self.assertIn("unavailable", B.first_event_line(None, None))

    def test_the_existing_first_token_line_and_its_clock_are_unchanged(self):
        src = inspect.getsource(B.WarmBrain.ask_stream)
        self.assertIn('log(f"[timing] claude first token {time.monotonic() - t0:.2f}s")', src)
        self.assertIn("t0 = time.monotonic()", src)
        i = src.index('[timing] claude first token')
        self.assertIn("first_event_line(h, ev)", src[i:i + 300])
        self.assertEqual(src.count("first_token_logged = True"), 1)          # still logged once, at the same instant


if __name__ == "__main__":
    unittest.main()
