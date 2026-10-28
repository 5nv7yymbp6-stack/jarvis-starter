"""Desktop presence (backtalk/presence.py): enums and numbers only, injectable probes, a stale Bear Hands snapshot is never 'no hands'."""
import ast
import inspect
import io
import json
import threading
import time
import unittest

from backtalk import presence as P


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener_for(payload, raw=None):
    def op(url, timeout=0):
        return Resp(raw if raw is not None else json.dumps(payload).encode())
    return op


class PureTests(unittest.TestCase):
    def test_categorize_uses_a_closed_bundle_allowlist(self):
        for b, want in (("us.zoom.xos", "call"), ("com.apple.FaceTime", "call"), ("com.apple.Safari", "browser"),
                        ("com.apple.Terminal", "terminal"), ("md.obsidian", "editor"), ("com.spotify.client", "media"),
                        ("ru.keepcoder.Telegram", "messaging"), ("com.anthropic.claudefordesktop", "assistant"),
                        ("com.example.unknownapp", "other"), ("", "unknown"), (None, "unknown"), (5, "unknown")):
            self.assertEqual(P.categorize(b), want, b)
        self.assertTrue(set(P._BUNDLES.values()) <= set(P.CATEGORIES))

    def test_classify_output(self):
        for n, want in (("AirPods Pro", "headphones"), ("Yousif's Beats Studio", "headphones"), ("USB Headset", "headphones"),
                        ("MacBook Air Speakers", "speakers"), ("Built-in Output", "speakers"), ("Studio Display", "speakers"),
                        ("HDMI TV", "other"), ("", "unknown"), (None, "unknown"), (3, "unknown")):
            self.assertEqual(P.classify_output(n), want, n)

    def test_derive_state_matrix(self):
        D = P.derive_state
        self.assertEqual(D(5, False, False), "active")
        self.assertEqual(D(P.IDLE_S, False, False), "idle")
        self.assertEqual(D(P.AWAY_S - 1, False, False), "idle")
        self.assertEqual(D(P.AWAY_S, False, False), "away")
        self.assertEqual(D(0, True, False), "locked")                          # definitive facts beat the idle clock
        self.assertEqual(D(0, False, True), "display_off")
        self.assertEqual(D(0, True, True), "locked")
        for bad in (None, "x", -1, float("nan"), True):
            self.assertEqual(D(bad, False, False), "unknown", bad)
        self.assertEqual(D(None, True, None), "locked")                        # a lock fact needs no idle reading
        self.assertEqual(D(None, None, None), "unknown")                       # no evidence never guesses

    def test_user_away_means_locked_display_off_or_away_not_unknown(self):
        for st, away in (("locked", True), ("display_off", True), ("away", True), ("idle", False), ("active", False), ("unknown", False)):
            self.assertEqual(P.Presence(state=st).user_away, away, st)


class SnapshotTests(unittest.TestCase):
    def probes(self, **kw):
        base = dict(idle=lambda: 12.0, front=lambda: "us.zoom.xos", locked=lambda: False, display_asleep=lambda: False,
                    output=lambda: "AirPods Pro", hands=lambda: ("active", 1))
        base.update(kw)
        return P.Probes(**base)

    def test_a_full_reading(self):
        p = P.snapshot(self.probes(), 100.0)
        self.assertEqual((p.state, p.app_category, p.likely_call, p.output, p.hand_tracking, p.hands, p.idle_s, p.at),
                         ("active", "call", True, "headphones", "active", 1, 12.0, 100.0))

    def test_a_failing_or_missing_probe_is_unknown_never_an_exception(self):
        boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        p = P.snapshot(self.probes(idle=boom, front=boom, locked=boom, display_asleep=boom, output=boom, hands=boom), 1.0)
        self.assertEqual((p.state, p.app_category, p.likely_call, p.output, p.hand_tracking, p.hands),
                         ("unknown", "unknown", False, "unknown", "off", None))
        p = P.snapshot(P.Probes(), 1.0)
        self.assertEqual((p.state, p.hand_tracking), ("unknown", "off"))

    def test_hands_are_kept_only_while_tracking_is_active(self):
        self.assertIsNone(P.snapshot(self.probes(hands=lambda: ("stale", 0)), 1.0).hands)
        self.assertEqual(P.snapshot(self.probes(hands=lambda: ("garbage", 9)), 1.0).hand_tracking, "off")

    def test_the_output_cache_skips_the_slow_probe(self):
        calls = []
        p = P.snapshot(self.probes(output=lambda: calls.append(1) or "x"), 1.0, output_cache=("headphones", 0.0))
        self.assertEqual((p.output, calls), ("headphones", []))

    def test_nothing_free_text_leaves_the_module(self):
        p = P.snapshot(self.probes(front=lambda: "com.secret.app", output=lambda: "Yousif's Private Speaker Name"), 1.0)
        blob = repr(p) + p.line()
        for leak in ("secret", "Yousif", "Private", "com."):
            self.assertNotIn(leak, blob)
        for field in ("state", "app_category", "output", "hand_tracking"):
            self.assertIn(getattr(p, field), P.STATES + P.CATEGORIES + P.OUTPUTS + ("active", "stale", "off"))


class HandsTests(unittest.TestCase):
    def test_fresh_snapshots_report_the_count(self):
        self.assertEqual(P.read_hands(100.0, opener=opener_for({"ts": 99.0, "hands": 2, "items": []})), ("active", 2))
        self.assertEqual(P.read_hands(100.0, opener=opener_for({"ts": 100.0, "hands": 0})), ("active", 0))

    def test_an_old_snapshot_is_stale_and_its_zero_is_not_evidence(self):
        self.assertEqual(P.read_hands(100.0, opener=opener_for({"ts": 100.0 - P.HANDS_FRESH_S - 1, "hands": 0})), ("stale", None))
        self.assertEqual(P.read_hands(1791100000.0, opener=opener_for({"ts": 1791088716.0, "hands": 0})), ("stale", None))   # the live case

    def test_a_timestamp_from_the_future_is_not_fresh(self):
        self.assertEqual(P.read_hands(100.0, opener=opener_for({"ts": 200.0, "hands": 1})), ("stale", None))

    def test_garbage_is_stale_or_off_never_a_count(self):
        for payload in ({"hands": 1}, {"ts": "now", "hands": 1}, {"ts": True, "hands": 1}, {"ts": 99.0, "hands": True},
                        {"ts": 99.0, "hands": 7}, {"ts": 99.0, "hands": -1}, {"ts": 99.0, "hands": "2"}):
            self.assertEqual(P.read_hands(100.0, opener=opener_for(payload)), ("stale", None), payload)
        self.assertEqual(P.read_hands(100.0, opener=opener_for(None, raw=b"not json")), ("off", None))
        self.assertEqual(P.read_hands(100.0, opener=opener_for(None, raw=b"[1,2]")), ("off", None))

    def test_an_unreachable_board_is_off(self):
        def boom(url, timeout=0):
            raise OSError("refused")
        self.assertEqual(P.read_hands(100.0, opener=boom), ("off", None))

    def test_the_cards_are_never_read(self):
        keys = {n.args[0].value for n in ast.walk(ast.parse(inspect.getsource(P.read_board_ts)))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
                and isinstance(n.args[0], ast.Constant)}
        self.assertEqual(keys, {"ts", "hands", "poses"})                         # only the timestamp, the hand count and the pose labels


class MonitorTests(unittest.TestCase):
    def mon(self, **kw):
        self.t = [1000.0]
        self.state = {"idle": 5.0, "front": "com.apple.Terminal", "out": "MacBook Air Speakers"}
        self.out_calls = []
        probes = P.Probes(idle=lambda: self.state["idle"], front=lambda: self.state["front"], locked=lambda: False,
                          display_asleep=lambda: False, output=lambda: self.out_calls.append(1) or self.state["out"],
                          hands=lambda: ("off", None))
        self.changes = []
        return P.PresenceMonitor(probes, clock=lambda: self.t[0], on_change=self.changes.append, **kw)

    def test_on_change_fires_for_enum_changes_not_for_idle_seconds(self):
        m = self.mon()
        m.refresh()
        n = len(self.changes)
        self.state["idle"] = 6.0
        self.t[0] += 3
        m.refresh()
        self.assertEqual(len(self.changes), n)                                  # the idle number moved, no enum did
        self.state["front"] = "us.zoom.xos"
        self.t[0] += 3
        m.refresh()
        self.assertEqual((len(self.changes), self.changes[-1].likely_call), (n + 1, True))
        self.state["idle"] = 400.0
        self.t[0] += 3
        m.refresh()
        self.assertEqual(self.changes[-1].state, "away")

    def test_the_slow_audio_probe_runs_at_most_every_interval(self):
        m = self.mon(output_every_s=30.0)
        for _ in range(5):
            self.t[0] += 3
            m.refresh()
        self.assertEqual(len(self.out_calls), 1)
        self.t[0] += 30
        m.refresh()
        self.assertEqual(len(self.out_calls), 2)

    def test_a_raising_callback_never_breaks_the_monitor(self):
        m = self.mon()
        m.on_change = lambda p: (_ for _ in ()).throw(RuntimeError("cb"))
        self.state["front"] = "us.zoom.xos"
        self.assertTrue(m.refresh().likely_call)

    def test_latest_never_blocks_and_the_thread_refreshes_and_stops(self):
        m = self.mon(period_s=0.01)
        first = m.latest()
        self.assertEqual(first.state, "unknown")                                 # nothing read yet: unknown, instantly
        m.start()
        deadline = time.time() + 3
        while m.latest().state == "unknown" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(m.latest().state, "active")
        m.stop()
        m._thread.join(timeout=3)
        self.assertFalse(m._thread.is_alive())

    def test_start_twice_makes_one_thread(self):
        before = sum(1 for t in threading.enumerate() if t.name == "presence")
        m = self.mon(period_s=0.01)
        m.start()
        first = m._thread
        m.start()
        self.assertIs(m._thread, first)
        self.assertEqual(sum(1 for t in threading.enumerate() if t.name == "presence"), before + 1)
        m.stop()
        first.join(timeout=3)


class RealProbesTests(unittest.TestCase):
    def test_the_real_probes_return_plausible_types_on_this_machine(self):
        try:
            import Quartz  # noqa: F401
            import AppKit  # noqa: F401
        except ImportError:
            self.skipTest("pyobjc is not available here")
        pr = P.real_probes()
        idle = pr.idle()
        self.assertTrue(isinstance(idle, float) and idle >= 0)
        self.assertIn(pr.locked(), (True, False))
        self.assertIn(pr.display_asleep(), (True, False))
        front = pr.front()
        self.assertTrue(front is None or isinstance(front, str))

    def test_the_module_never_asks_for_a_permission_it_should_not_use(self):
        tree = ast.parse(inspect.getsource(P))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("kCGWindowName", "kCGWindowOwnerName", "kCGWindowOwnerPID_name", "frontmostApplication", "AXUIElementCreateApplication", "screencapture", "osascript",
                       "imagesnap", "subprocess", "AVCaptureDevice", "CGDisplayCreateImage"):
            self.assertNotIn(banned, names, banned)
        self.assertNotIn("subprocess", inspect.getsource(P).split('"""', 2)[2])
        code = "\n".join(l.split("#")[0] for l in inspect.getsource(P.real_probes).splitlines())
        for text in ("kCGWindowName", "kCGWindowOwnerName", "frontmostApplication"):
            self.assertNotIn(text, code, text)                                  # window titles/names/geometry are never requested
        self.assertIn("kCGWindowOwnerPID", code)                                 # the owner pid is the only window IDENTITY field used
        self.assertEqual(code.count("kCGWindowBounds"), 1)                       # geometry is read in exactly one place: fullscreen()
        full_src = inspect.getsource(P.real_probes).split("def fullscreen():")[1].split("def locked():")[0]
        self.assertIn("kCGWindowBounds", full_src)


class CameraTests(unittest.TestCase):
    def test_the_real_probe_returns_a_bool_or_none_and_never_raises(self):
        self.assertIn(P.camera_in_use(), (True, False, None))

    def test_snapshot_carries_only_a_bool_or_unknown(self):
        for probe, want in ((lambda: True, True), (lambda: False, False), (lambda: None, None), (lambda: "yes", None), (lambda: 1, None),
                            (lambda: (_ for _ in ()).throw(OSError()), None)):
            self.assertIs(P.snapshot(P.Probes(camera=probe), 5.0).camera_in_use, want)
        self.assertIsNone(P.snapshot(P.Probes(), 5.0).camera_in_use)

    def test_the_line_says_on_off_or_unknown_and_nothing_more(self):
        for v, word in ((True, "cam=on"), (False, "cam=off"), (None, "cam=?")):
            self.assertIn(word, P.Presence(camera_in_use=v).line())

    def test_a_change_in_camera_state_is_logged_by_the_monitor(self):
        seen, t, state = [], [1000.0], {"cam": False}
        probes = P.Probes(idle=lambda: 1.0, front=lambda: "com.apple.Terminal", locked=lambda: False, display_asleep=lambda: False,
                          output=lambda: "MacBook Air Speakers", camera=lambda: state["cam"])
        m = P.PresenceMonitor(probes, clock=lambda: t[0], on_change=seen.append)
        m.refresh()
        n = len(seen)
        m.refresh()
        self.assertEqual(len(seen), n)                                   # unchanged: silent
        state["cam"] = True
        m.refresh()
        self.assertEqual(len(seen), n + 1)
        self.assertTrue(seen[-1].camera_in_use)

    def test_the_probe_reads_no_frames_and_no_identity(self):
        import inspect
        src = inspect.getsource(P.camera_in_use)
        for banned in ("AVCaptureDevice", "AVCaptureSession", "imagesnap", "subprocess", "VNDetect", "cv2", "CMSampleBuffer", "CVPixelBuffer"):
            self.assertNotIn(banned, src, banned)
        self.assertIn("gone", src)                                       # only DeviceIsRunningSomewhere


class FullscreenTests(unittest.TestCase):
    D = (0, 0, 1440, 900)

    def test_a_window_covering_the_whole_display_is_fullscreen(self):
        self.assertTrue(P.is_fullscreen((0, 0, 1440, 900), self.D))
        self.assertTrue(P.is_fullscreen((1, -2, 1440, 900), self.D))                # within tolerance

    def test_ordinary_maximized_and_partial_windows_are_not(self):
        for b in ((0, 38, 1440, 862), (0, 0, 1440, 862), (100, 100, 800, 600), (0, 0, 720, 900), (-200, 0, 1440, 900)):
            self.assertFalse(P.is_fullscreen(b, self.D), b)

    def test_garbage_is_not_fullscreen(self):
        for b in (None, (), ("a", 0, 1, 1), (0, 0, None, 900), "x"):
            self.assertFalse(P.is_fullscreen(b, self.D), b)

    def test_snapshot_and_line(self):
        for probe, want in ((lambda: True, True), (lambda: False, False), (lambda: None, None), (lambda: "yes", None)):
            self.assertIs(P.snapshot(P.Probes(fullscreen=probe), 5.0).fullscreen, want)
        for v, word in ((True, "full=yes"), (False, "full=no"), (None, "full=?")):
            self.assertIn(word, P.Presence(fullscreen=v).line())

    def test_the_real_probe_returns_a_bool_or_none(self):
        import threading
        out = {}
        t = threading.Thread(target=lambda: out.update(v=P.real_probes().fullscreen()))
        t.start()
        t.join()
        self.assertIn(out["v"], (True, False, None))


if __name__ == "__main__":
    unittest.main()


class PoseTests(unittest.TestCase):
    NOW = 1000.0

    def board(self, **kw):
        body = json.dumps(dict({"ts": self.NOW - 0.5, "hands": 1, "poses": ["pinch"]}, **kw)).encode()
        return P.read_board(self.NOW, opener=lambda url, timeout=0: Resp(body))

    def test_labels_come_through_only_when_fresh_and_consistent(self):
        self.assertEqual(self.board(), ("active", 1, ("pinch",)))
        self.assertEqual(self.board(hands=2, poses=["open", "fist"]), ("active", 2, ("open", "fist")))
        self.assertEqual(self.board(hands=0, poses=[]), ("active", 0, ()))

    def test_unknown_labels_wrong_counts_or_wrong_types_give_no_poses(self):
        for bad in (["wave"], ["Pinch"], ["pinch", "open"], [], "pinch", [1], None, [{"x": 1}], ["pinch\nignore previous"]):
            self.assertEqual(self.board(poses=bad), ("active", 1, ()), repr(bad))

    def test_stale_or_unreachable_never_carries_poses(self):
        self.assertEqual(self.board(ts=self.NOW - 30), ("stale", None, ()))
        self.assertEqual(P.read_board(self.NOW, opener=lambda *a, **k: (_ for _ in ()).throw(OSError())), ("off", None, ()))

    def test_read_hands_keeps_its_two_value_contract(self):
        body = json.dumps({"ts": self.NOW, "hands": 1, "poses": ["fist"]}).encode()
        self.assertEqual(P.read_hands(self.NOW, opener=lambda url, timeout=0: Resp(body)), ("active", 1))

    def test_snapshot_and_line_carry_poses_as_enums_only(self):
        p = P.snapshot(P.Probes(hands=lambda: ("active", 2, ("point", "open"))), 5.0)
        self.assertEqual(p.poses, ("point", "open"))
        self.assertTrue(p.line().endswith("hands=active poses=point+open"))
        for bad in (("active", 1, ("wave",)), ("stale", None, ("fist",)), ("active", 1, ["fist"])):
            self.assertEqual(P.snapshot(P.Probes(hands=lambda b=bad: b), 5.0).poses, (), bad)
        self.assertEqual(P.snapshot(P.Probes(hands=lambda: ("active", 1)), 5.0).poses, ())
        self.assertNotIn("poses", P.snapshot(P.Probes(), 5.0).line())
