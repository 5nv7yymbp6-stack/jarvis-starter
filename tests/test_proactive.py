"""Proactive decision engine (backtalk/proactive.py): every rule, in order, with a fake clock and no speech."""
import ast
import inspect
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from backtalk import proactive as P

NOW = 1_800_000_000.0
ON = P.Policy(enabled=True)
CLEAR = P.Context()


def ev(key="k1", kind="reminder", severity=P.NORMAL, **kw):
    return P.Event(kind=kind, key=key, severity=severity, at=NOW, source_id="x", **kw)


def eng(**policy):
    return P.Engine(P.Policy(enabled=True, **policy))


class RuleTests(unittest.TestCase):
    def test_a_disabled_engine_ignores_everything_and_queues_nothing(self):
        e = P.Engine(P.Policy())
        self.assertEqual(e.evaluate(ev(), CLEAR, NOW).reason, "disabled")
        self.assertFalse(e.offer(ev(), NOW))
        self.assertEqual(e.pending_keys(), [])

    def test_a_clear_room_notifies(self):
        d = eng().evaluate(ev(), CLEAR, NOW)
        self.assertEqual((d.outcome, d.reason), (P.NOTIFY, "clear"))

    def test_malformed_events_are_ignored(self):
        e = eng()
        for bad in (P.Event(kind="nope", key="k"), P.Event(kind="reminder", key=""),
                    P.Event(kind="reminder", key="k", severity="shouting")):
            self.assertEqual(e.evaluate(bad, CLEAR, NOW).outcome, P.IGNORE)

    def test_expired_events_are_ignored(self):
        d = eng().evaluate(ev(kind="bg_completion", expires_at=NOW - 1), CLEAR, NOW)
        self.assertEqual((d.outcome, d.reason), (P.IGNORE, "expired"))
        self.assertEqual(eng().evaluate(ev(kind="bg_completion", expires_at=NOW), CLEAR, NOW).outcome, P.NOTIFY)   # the edge is valid

    def test_low_severity_and_low_relevance_are_queued_not_spoken(self):
        e = eng()
        self.assertEqual(e.evaluate(ev(severity=P.LOW), CLEAR, NOW).outcome, P.QUEUE)
        self.assertEqual(e.evaluate(ev(relevance=0.05), CLEAR, NOW).reason, "low_value")
        self.assertEqual(e.evaluate(ev(relevance=0.05, severity=P.URGENT), CLEAR, NOW).outcome, P.NOTIFY)

    # roadmap test 8: non-urgent announcements wait while the user is speaking
    def test_the_user_speaking_makes_every_severity_wait(self):
        e = eng(urgent="speak_over_jarvis")
        for sev in (P.NORMAL, P.URGENT):
            d = e.evaluate(ev(severity=sev), P.Context(user_speaking=True), NOW)
            self.assertEqual((d.outcome, d.reason), (P.WAIT, "user_speaking"), sev)

    def test_recent_user_speech_counts_as_speaking(self):
        e = eng(quiet_after_user_s=2.0)
        self.assertEqual(e.evaluate(ev(), P.Context(since_user_speech_s=1.9), NOW).outcome, P.WAIT)
        self.assertEqual(e.evaluate(ev(), P.Context(since_user_speech_s=2.0), NOW).outcome, P.NOTIFY)
        self.assertEqual(e.evaluate(ev(), P.Context(since_user_speech_s=None), NOW).outcome, P.NOTIFY)

    # roadmap test 9: urgent announcements obey the configured policy
    def test_urgent_waits_for_jarvis_by_default(self):
        d = eng().evaluate(ev(severity=P.URGENT), P.Context(mouth_busy=True), NOW)
        self.assertEqual((d.outcome, d.reason), (P.WAIT, "jarvis_busy"))
        self.assertEqual(eng().evaluate(ev(severity=P.URGENT), P.Context(fg_active=True), NOW).outcome, P.WAIT)

    def test_urgent_may_speak_over_jarvis_only_when_the_policy_says_so(self):
        e = eng(urgent="speak_over_jarvis")
        for ctx in (P.Context(mouth_busy=True), P.Context(fg_active=True)):
            self.assertEqual(e.evaluate(ev(severity=P.URGENT), ctx, NOW).reason, "urgent_over_jarvis")
        self.assertEqual(e.evaluate(ev(severity=P.NORMAL), P.Context(mouth_busy=True), NOW).outcome, P.WAIT)

    def test_zip_beats_even_an_opted_in_urgent(self):
        e = eng(urgent="speak_over_jarvis")
        self.assertEqual(e.evaluate(ev(severity=P.URGENT), P.Context(zipped=True), NOW).reason, "zipped")

    def test_quiet_hours_hold_everything_but_urgent(self):
        chi = ZoneInfo("America/Chicago")
        night = datetime(2026, 10, 20, 2, 0, tzinfo=chi).timestamp()
        day = datetime(2026, 10, 20, 12, 0, tzinfo=chi).timestamp()
        e = eng(quiet_hours=("23:00", "07:00"))
        self.assertEqual(e.evaluate(ev(), CLEAR, night).reason, "quiet_hours")
        self.assertEqual(e.evaluate(ev(severity=P.URGENT), CLEAR, night).outcome, P.NOTIFY)
        self.assertEqual(e.evaluate(ev(), CLEAR, day).outcome, P.NOTIFY)
        same_day = eng(quiet_hours=("13:00", "15:00"))
        self.assertEqual(same_day.evaluate(ev(), CLEAR, datetime(2026, 10, 20, 14, 0, tzinfo=chi).timestamp()).outcome, P.WAIT)

    # roadmap test 11: unauthorized consequential actions are blocked
    def test_an_action_is_performed_only_when_allowlisted_and_not_consequential(self):
        self.assertEqual(eng(act_allow=("status_check",)).evaluate(ev(action="status_check"), CLEAR, NOW).outcome, P.ACT)

    def test_every_consequential_action_is_blocked_even_if_someone_allowlists_it(self):
        for act in sorted(P.CONSEQUENTIAL):
            e = eng(act_allow=(act,))
            d = e.evaluate(ev(action=act), CLEAR, NOW)
            self.assertNotEqual(d.outcome, P.ACT, act)
            self.assertIn("act_not_authorized", d.flags)
        self.assertNotEqual(eng().evaluate(ev(action="anything"), CLEAR, NOW).outcome, P.ACT)   # default allowlist is empty

    def test_a_blocked_action_is_still_announced_not_silently_dropped(self):
        d = eng().evaluate(ev(action="send"), CLEAR, NOW)
        self.assertEqual((d.outcome, d.reason), (P.NOTIFY, "clear"))
        self.assertIn("act_not_authorized", d.flags)

    def test_unverified_evidence_is_flagged(self):
        for v in ("UNVERIFIED", "VERIFY_FAILED"):
            self.assertIn("unverified_evidence", eng().evaluate(ev(kind="bg_completion", verified=v), CLEAR, NOW).flags)
        self.assertNotIn("unverified_evidence", eng().evaluate(ev(kind="bg_completion", verified="VERIFIED"), CLEAR, NOW).flags)

    def test_a_reminder_never_expires_but_an_ordinary_event_does(self):
        for kind in ("reminder", "overdue"):
            self.assertEqual(eng().evaluate(ev(kind=kind, expires_at=NOW - 99_999), CLEAR, NOW).outcome, P.NOTIFY, kind)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion", expires_at=NOW - 1), CLEAR, NOW).reason, "expired")

    def test_nothing_speaks_while_the_brain_is_not_ready_or_just_started(self):
        e = eng(urgent="speak_over_jarvis")
        for ctx in (P.Context(brain_ready=False), P.Context(settling=True)):
            for sev in (P.NORMAL, P.URGENT):
                self.assertEqual(e.evaluate(ev(severity=sev), ctx, NOW).reason, "not_ready", (ctx, sev))

    def test_a_background_completion_needs_an_active_conversation_but_a_reminder_does_not(self):
        idle = P.Context(conversation_active=False)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion"), idle, NOW).reason, "not_in_conversation")
        self.assertEqual(eng().evaluate(ev(kind="reminder"), idle, NOW).outcome, P.NOTIFY)
        self.assertEqual(eng().evaluate(ev(kind="overdue"), idle, NOW).outcome, P.NOTIFY)

    # Phase G: presence
    def test_a_likely_call_holds_everything_including_reminders_and_urgent_by_default(self):
        for kind in ("reminder", "bg_completion", "overdue"):
            for sev in (P.NORMAL, P.URGENT):
                d = eng().evaluate(ev(kind=kind, severity=sev), P.Context(on_call=True), NOW)
                self.assertEqual((d.outcome, d.reason), (P.WAIT, "on_call"), (kind, sev))

    def test_an_urgent_line_may_speak_during_a_call_only_with_the_explicit_opt_in(self):
        e = eng(urgent_on_call=True)
        self.assertEqual(e.evaluate(ev(severity=P.URGENT), P.Context(on_call=True), NOW).outcome, P.NOTIFY)
        self.assertEqual(e.evaluate(ev(severity=P.NORMAL), P.Context(on_call=True), NOW).reason, "on_call")

    def test_the_user_speaking_still_beats_a_call_opt_in(self):
        e = eng(urgent_on_call=True)
        self.assertEqual(e.evaluate(ev(severity=P.URGENT), P.Context(on_call=True, user_speaking=True), NOW).reason, "user_speaking")

    def test_while_nobody_is_at_the_machine_news_waits_but_a_requested_reminder_does_not(self):
        away = P.Context(user_away=True)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion"), away, NOW).reason, "user_away")
        self.assertEqual(eng().evaluate(ev(kind="action_bad_news"), away, NOW).reason, "user_away")
        for kind in ("reminder", "overdue"):
            self.assertEqual(eng().evaluate(ev(kind=kind), away, NOW).outcome, P.NOTIFY, kind)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion", severity=P.URGENT), away, NOW).outcome, P.NOTIFY)

    def test_away_news_is_delivered_when_the_user_returns(self):
        e = eng()
        e.offer(ev("t", kind="bg_completion"), NOW)
        self.assertEqual(e.tick(P.Context(user_away=True), NOW + 1), [])
        self.assertEqual(e.pending_keys(), ["t"])
        (d,) = e.tick(P.Context(user_away=False), NOW + 600)
        self.assertEqual((d.outcome, d.key), (P.NOTIFY, "t"))

    def test_a_running_camera_holds_background_news_but_never_a_reminder_or_urgent(self):
        busy = P.Context(camera_busy=True)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion"), busy, NOW).reason, "camera_busy")
        self.assertEqual(eng().evaluate(ev(kind="action_bad_news"), busy, NOW).reason, "camera_busy")
        for kind in ("reminder", "overdue"):
            self.assertEqual(eng().evaluate(ev(kind=kind), busy, NOW).outcome, P.NOTIFY, kind)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion", severity=P.URGENT), busy, NOW).outcome, P.NOTIFY)

    def test_a_fullscreen_window_holds_background_news_but_never_a_reminder_or_urgent(self):
        busy = P.Context(screen_busy=True)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion"), busy, NOW).reason, "screen_busy")
        for kind in ("reminder", "overdue"):
            self.assertEqual(eng().evaluate(ev(kind=kind), busy, NOW).outcome, P.NOTIFY, kind)
        self.assertEqual(eng().evaluate(ev(kind="bg_completion", severity=P.URGENT), busy, NOW).outcome, P.NOTIFY)
        self.assertFalse(P.Context().screen_busy)

    def test_news_held_for_the_camera_is_delivered_when_it_stops(self):
        e = eng()
        e.offer(ev("t", kind="bg_completion"), NOW)
        self.assertEqual(e.tick(P.Context(camera_busy=True), NOW + 1), [])
        (d,) = e.tick(P.Context(camera_busy=False), NOW + 700)
        self.assertEqual((d.outcome, d.key), (P.NOTIFY, "t"))

    def test_presence_defaults_change_nothing(self):
        c = P.Context()
        self.assertEqual((c.on_call, c.user_away, c.camera_busy), (False, False, False))
        self.assertEqual(eng().evaluate(ev(), c, NOW).reason, "clear")
        self.assertFalse(P.Policy().urgent_on_call)
        self.assertTrue(P.Policy.from_config({"urgent_on_call": True}).urgent_on_call)
        for v in ("true", 1, None):
            self.assertFalse(P.Policy.from_config({"urgent_on_call": v}).urgent_on_call, v)

    def test_the_hourly_cap_queues_but_never_reminders_or_urgent(self):
        e = eng(max_per_hour=2)
        for i in range(2):
            e.offer(ev(f"c{i}", kind="bg_completion"), NOW)
            e.delivered(f"c{i}", NOW + i)
        self.assertEqual(e.evaluate(ev("c9", kind="bg_completion"), CLEAR, NOW + 5).reason, "rate_limited")
        self.assertEqual(e.evaluate(ev("c9", kind="bg_completion", severity=P.URGENT), CLEAR, NOW + 5).outcome, P.NOTIFY)
        for i in range(5):
            e.offer(ev(f"r{i}", kind="reminder"), NOW)
            e.delivered(f"r{i}", NOW + i)
        self.assertEqual(e.evaluate(ev("r9", kind="reminder"), CLEAR, NOW + 9).outcome, P.NOTIFY)
        self.assertEqual(e.evaluate(ev("c9", kind="bg_completion"), CLEAR, NOW + 3601).outcome, P.NOTIFY)   # the hour rolled over


class QueueTests(unittest.TestCase):
    # roadmap test 5: duplicate events never produce duplicate announcements
    def test_a_duplicate_key_is_not_queued_twice(self):
        e = eng()
        self.assertTrue(e.offer(ev("same"), NOW))
        self.assertFalse(e.offer(ev("same"), NOW))
        self.assertEqual(e.pending_keys(), ["same"])

    def test_a_delivered_event_is_remembered_and_never_announced_again(self):
        e = eng()
        e.offer(ev("same"), NOW)
        (d,) = e.tick(CLEAR, NOW)
        self.assertEqual(d.outcome, P.NOTIFY)
        e.delivered("same", NOW)
        self.assertFalse(e.offer(ev("same"), NOW + 1))
        self.assertEqual(e.tick(CLEAR, NOW + 2), [])
        self.assertEqual(e.evaluate(ev("same"), CLEAR, NOW + 2).reason, "duplicate")

    def test_the_memory_expires_after_the_ttl(self):
        e = eng(dedupe_ttl_s=100)
        e.offer(ev("same"), NOW)
        e.delivered("same", NOW)
        self.assertFalse(e.offer(ev("same"), NOW + 99))
        e.tick(CLEAR, NOW + 101)                                   # prune
        self.assertTrue(e.offer(ev("same"), NOW + 101))

    def test_wait_stays_queued_and_is_delivered_when_the_user_stops_speaking(self):
        e = eng()
        e.offer(ev("r"), NOW)
        self.assertEqual(e.tick(P.Context(user_speaking=True), NOW + 1), [])
        self.assertEqual(e.pending_keys(), ["r"])
        (d,) = e.tick(CLEAR, NOW + 60)
        self.assertEqual((d.outcome, d.key), (P.NOTIFY, "r"))

    def test_one_announcement_per_tick_urgent_first_then_oldest(self):
        e = eng()
        e.offer(ev("old"), NOW)
        e.offer(ev("newer"), NOW + 1)
        e.offer(ev("hot", severity=P.URGENT), NOW + 2)
        order = []
        t = NOW + 3
        while True:
            got = [d for d in e.tick(CLEAR, t) if d.outcome == P.NOTIFY]
            if not got:
                break
            self.assertEqual(len(got), 1)
            order.append(got[0].key)
            e.delivered(got[0].key, t)
            t += 1
        self.assertEqual(order, ["hot", "old", "newer"])

    def test_queue_and_ignore_leave_the_queue_for_good(self):
        e = eng()
        e.offer(ev("low", severity=P.LOW), NOW)
        e.offer(ev("exp", kind="bg_completion", expires_at=NOW - 5), NOW)
        decisions = e.tick(CLEAR, NOW)
        self.assertEqual({d.key: d.outcome for d in decisions}, {"low": P.QUEUE, "exp": P.IGNORE})
        self.assertEqual(e.pending_keys(), [])
        self.assertFalse(e.offer(ev("exp"), NOW + 1))             # an ignored key is remembered too

    def test_the_queue_is_bounded_and_drops_the_least_important_oldest(self):
        e = eng()
        e.offer(ev("keep", severity=P.URGENT), NOW)
        for i in range(P.QUEUE_MAX + 5):
            e.offer(ev(f"n{i}"), NOW + 1 + i)
        keys = e.pending_keys()
        self.assertLessEqual(len(keys), P.QUEUE_MAX)
        self.assertIn("keep", keys)
        self.assertEqual(e.stats["queue_overflow_drop"], 6)

    def test_a_flushed_announcement_is_requeued_a_bounded_number_of_times(self):
        e = eng()
        event = ev("r")
        e.offer(event, NOW)
        e.tick(CLEAR, NOW)
        e.delivered("r", NOW)
        for n in range(1, P.MAX_REQUEUES + 1):
            self.assertTrue(e.requeue("r", event, NOW + n), n)
            (d,) = e.tick(CLEAR, NOW + n)
            self.assertEqual((d.outcome, d.attempts), (P.NOTIFY, n))
            e.delivered("r", NOW + n)
        self.assertFalse(e.requeue("r", event, NOW + 10))         # bound spent
        self.assertFalse(e.offer(event, NOW + 11))                # and it stays remembered as delivered

    def test_an_act_decision_is_returned_without_waiting_for_a_quiet_room(self):
        e = eng(act_allow=("status_check",))
        e.offer(ev("a", action="status_check"), NOW)
        (d,) = e.tick(P.Context(user_speaking=True, mouth_busy=True), NOW)
        self.assertEqual(d.outcome, P.ACT)


class PolicyTests(unittest.TestCase):
    def test_only_literal_true_enables(self):
        self.assertTrue(P.Policy.from_config({"enabled": True}).enabled)
        for v in ("true", 1, "yes", None, [], {}):
            self.assertFalse(P.Policy.from_config({"enabled": v}).enabled, v)
        for junk in (None, [], "x", 5):
            self.assertEqual(P.Policy.from_config(junk), P.Policy())

    def test_malformed_values_fall_back_to_the_safe_defaults(self):
        p = P.Policy.from_config({"enabled": True, "urgent": "always", "quiet_hours": ["25:00", "07:00"],
                                  "max_per_hour": -3, "dedupe_ttl_s": "soon", "quiet_after_user_s": True})
        self.assertEqual((p.urgent, p.quiet_hours, p.max_per_hour, p.dedupe_ttl_s, p.quiet_after_user_s),
                         ("off", None, 6, 24 * 3600.0, 2.0))

    def test_consequential_actions_cannot_be_allowlisted_through_config(self):
        p = P.Policy.from_config({"enabled": True, "act_allow": ["status_check", "Send", "delete", 7]})
        self.assertEqual(p.act_allow, ("status_check",))

    def test_urgent_policy_must_be_the_exact_opt_in_string(self):
        self.assertEqual(P.Policy.from_config({"urgent": "speak_over_jarvis"}).urgent, "speak_over_jarvis")
        self.assertEqual(P.Policy.from_config({"urgent": "SPEAK_OVER_JARVIS"}).urgent, "off")


class HygieneTests(unittest.TestCase):
    def test_a_decision_line_carries_codes_and_the_key_never_text(self):
        d = eng().evaluate(ev("reminder:sch_1:1"), CLEAR, NOW)
        line = d.line()
        self.assertTrue(line.startswith("[proactive] decision outcome=NOTIFY reason=clear kind=reminder sev=normal key=reminder:sch_1:1"))
        self.assertNotIn("\n", line)

    def test_the_module_is_pure_stdlib(self):
        tree = ast.parse(inspect.getsource(P))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module)
        self.assertEqual(mods, {"collections", "re", "dataclasses", "datetime", "zoneinfo"})
        src = inspect.getsource(P)
        body = src.split('"""', 2)[2]
        for banned in ("time.time", "time.monotonic", "open(", ".say(", "say_chunk", "socket", "subprocess", "os."):
            self.assertNotIn(banned, body, banned)


if __name__ == "__main__":
    unittest.main()
