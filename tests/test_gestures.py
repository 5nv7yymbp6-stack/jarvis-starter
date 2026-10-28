import ast
import inspect
import unittest
from types import SimpleNamespace

from backtalk import gestures as G
from backtalk import main as M

OPEN2 = ("active", 2, ("open", "open"))


def det(**kw):
    return G.HushDetector(**kw)


class DetectorTests(unittest.TestCase):
    def test_two_open_hands_held_across_new_snapshots_fires_once(self):
        d = det()
        self.assertFalse(d.feed(*OPEN2, 100.0, 100.0))
        self.assertFalse(d.feed(*OPEN2, 100.6, 100.6))                 # 0.6 s: not yet
        self.assertTrue(d.feed(*OPEN2, 101.3, 101.3))                  # 1.3 s held, 3 snapshots
        self.assertFalse(d.feed(*OPEN2, 102.4, 102.4))                 # still held: no second fire

    def test_the_same_snapshot_read_twice_is_not_new_evidence(self):
        d = det()
        d.feed(*OPEN2, 100.0, 100.0)
        for t in (100.5, 101.0, 101.5, 102.0):
            self.assertFalse(d.feed(*OPEN2, 100.0, t))                 # ts never advanced: a frozen page

    def test_one_hand_or_other_poses_never_fire(self):
        for tracking, hands, poses in (("active", 1, ("open",)), ("active", 2, ("open", "fist")), ("active", 2, ("fist", "fist")),
                                       ("active", 2, ("pinch", "open")), ("active", 2, ("open", "other")), ("active", 0, ())):
            d = det()
            fired = [d.feed(tracking, hands, poses, 100.0 + i * 0.5, 100.0 + i * 0.5) for i in range(8)]
            self.assertFalse(any(fired), (hands, poses))

    def test_unknown_labels_or_no_poses_never_fire(self):
        d = det()
        fired = [d.feed("active", 2, (), 100.0 + i, 100.0 + i) for i in range(5)]
        self.assertFalse(any(fired))

    def test_a_break_resets_the_hold(self):
        d = det()
        d.feed(*OPEN2, 100.0, 100.0)
        d.feed("active", 2, ("open", "fist"), 100.7, 100.7)            # one hand closed
        self.assertFalse(d.feed(*OPEN2, 101.4, 101.4))                 # the hold starts over
        self.assertTrue(d.feed(*OPEN2, 102.7, 102.7))

    def test_stale_or_off_tracking_resets_everything(self):
        d = det()
        d.feed(*OPEN2, 100.0, 100.0)
        d.feed("stale", None, (), None, 100.5)
        self.assertFalse(d.feed(*OPEN2, 101.0, 101.0))                 # a fresh start, not a continuation of the old hold
        d.feed("off", None, (), None, 101.5)
        self.assertFalse(d.feed(*OPEN2, 102.0, 102.0))

    def test_it_must_see_the_hands_change_before_firing_again_and_respects_the_cooldown(self):
        d = det(cooldown_s=6.0)
        d.feed(*OPEN2, 100.0, 100.0)
        self.assertTrue(d.feed(*OPEN2, 101.5, 101.5))
        d.feed("active", 0, (), 102.0, 102.0)                           # hands down: re-armed
        d.feed(*OPEN2, 103.0, 103.0)
        self.assertFalse(d.feed(*OPEN2, 104.5, 104.5))                  # held again but inside the 6 s cooldown
        d.feed("active", 0, (), 106.0, 106.0)
        d.feed(*OPEN2, 108.0, 108.0)
        self.assertTrue(d.feed(*OPEN2, 109.4, 109.4))

    def test_the_hold_time_is_measured_on_the_snapshot_clock(self):
        d = det()
        d.feed(*OPEN2, 100.0, 100.0)
        self.assertFalse(d.feed(*OPEN2, 100.4, 150.0))                  # wall time far ahead, snapshots only 0.4 s apart
        self.assertTrue(d.feed(*OPEN2, 101.3, 150.5))


class WatcherTests(unittest.TestCase):
    def test_it_calls_hush_once_when_the_gesture_completes_and_swallows_action_errors(self):
        script = iter([(*OPEN2, 100.0), (*OPEN2, 100.7), (*OPEN2, 101.4), (*OPEN2, 102.1)])
        called, logs = [], []
        t = [100.0]
        w = G.GestureWatcher(lambda: called.append(1), fetch=lambda now: next(script), clock=lambda: t[0], log=logs.append)
        results = []
        for _ in range(4):
            results.append(w.step())
            t[0] += 0.7
        self.assertEqual(results, [False, False, True, False])
        self.assertEqual(called, [1])

        def boom():
            raise RuntimeError("x")
        script2 = iter([(*OPEN2, 100.0), (*OPEN2, 101.4)])
        w2 = G.GestureWatcher(boom, fetch=lambda now: next(script2), clock=lambda: 100.0, log=logs.append)
        w2.step()
        self.assertTrue(w2.step())
        self.assertTrue(any("hush action failed (RuntimeError)" in l for l in logs))

    def test_a_failing_fetch_is_survivable_in_the_thread_loop(self):
        w = G.GestureWatcher(lambda: None, fetch=lambda now: (_ for _ in ()).throw(OSError()), period_s=0.01)
        w.start()
        w.stop()


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(M.CFG)
        self.logs = []

    def tearDown(self):
        M.CFG.clear()
        M.CFG.update(self.saved)
        w = M._GESTURES["watcher"]
        if w is not None:
            w.stop()
        M._GESTURES["watcher"] = None

    def test_off_by_default_and_only_literal_true_starts_it(self):
        mouth = SimpleNamespace(busy=True, shut_up=lambda: None)
        for v in (None, "true", 1, [], {"hush": "yes"}, {"hush": False}):
            M.CFG["gestures"] = v if isinstance(v, dict) else {"hush": v}
            self.assertIsNone(M._gestures_start(mouth, self.logs.append), v)
        M.CFG["gestures"] = "garbage"
        self.assertIsNone(M._gestures_start(mouth, self.logs.append))
        self.assertIsNone(M._GESTURES["watcher"])

    def test_enabled_starts_a_watcher_and_the_action_only_silences_a_busy_voice(self):
        silenced = []
        mouth = SimpleNamespace(busy=True, shut_up=lambda: silenced.append(1))
        M.CFG["gestures"] = {"hush": True}
        orig = M.gestures_mod.GestureWatcher
        made = {}

        class Fake:
            def __init__(self, on_hush, **kw):
                made["hush"] = on_hush

            def start(self):
                made["started"] = True

            def stop(self):
                pass
        M.gestures_mod.GestureWatcher = Fake
        try:
            w = M._gestures_start(mouth, self.logs.append)
        finally:
            M.gestures_mod.GestureWatcher = orig
        self.assertIsInstance(w, Fake)
        self.assertTrue(made["started"])
        made["hush"]()
        self.assertEqual(silenced, [1])
        mouth.busy = False
        made["hush"]()
        self.assertEqual(silenced, [1])                                  # nothing speaking: no call
        self.assertTrue(any("voice stopped" in l for l in self.logs) and any("nothing was speaking" in l for l in self.logs))

    def test_a_start_failure_degrades_quietly(self):
        M.CFG["gestures"] = {"hush": True}
        orig = M.gestures_mod.GestureWatcher
        M.gestures_mod.GestureWatcher = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        try:
            self.assertIsNone(M._gestures_start(SimpleNamespace(busy=True, shut_up=lambda: None), self.logs.append))
        finally:
            M.gestures_mod.GestureWatcher = orig
        self.assertTrue(any("could not start (RuntimeError)" in l for l in self.logs))

    def test_it_is_started_in_amain_and_reaches_no_model_input(self):
        self.assertIn("_gestures_start(mouth)", inspect.getsource(M.amain))
        src = inspect.getsource(M._gestures_start)
        for banned in ("model_input", "brain", "send_user_message", "ask_stream"):
            self.assertNotIn(banned, src)
        tree = ast.parse(inspect.getsource(G))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("subprocess", "screencapture", "imagesnap", "CGWindowListCopyWindowInfo", "AVCaptureDevice"):
            self.assertNotIn(banned, names)


if __name__ == "__main__":
    unittest.main()
