"""Gesture control (roadmap Phase G). ONE gesture, one effect: BOTH HANDS OPEN and held for about a second (a "stop" sign) silences the
voice. It stops playback and flushes what is queued (`mouth.shut_up()`), exactly what talking over the voice does, and nothing else: it
cancels no work, sends nothing, changes no setting.

Input is only the Bear Hands board's pose labels (presence.read_board_ts: an enum per hand, never landmarks, never video). OFF unless the
JSON value true is set for CFG["gestures"]["hush"].

HushDetector is pure (feed it readings and times). A reading only counts when it is a NEW snapshot (its own timestamp advanced), so a stale
page cannot hold a gesture open; any snapshot that is not "two open hands" ends the hold; after it fires it must see the hands change
before it can fire again, and it has a cooldown. GestureWatcher is the small polling thread around it.
"""
import threading
import time

from backtalk import presence

HOLD_S = 1.2            # snapshots arrive about once a second: this needs the pose in at least two consecutive new ones
MIN_SNAPSHOTS = 2
COOLDOWN_S = 6.0
POLL_S = 0.4


class HushDetector:
    def __init__(self, hold_s=HOLD_S, min_snapshots=MIN_SNAPSHOTS, cooldown_s=COOLDOWN_S):
        self.hold_s, self.min_snapshots, self.cooldown_s = hold_s, min_snapshots, cooldown_s
        self._last_ts = None
        self._start = None
        self._count = 0
        self._armed = True
        self._cooldown_until = 0.0

    @staticmethod
    def is_stop_sign(tracking, hands, poses) -> bool:
        return tracking == "active" and hands == 2 and tuple(poses) == ("open", "open")

    def feed(self, tracking, hands, poses, ts, now) -> bool:
        """True exactly when the gesture completes. `ts` is the snapshot's own timestamp (None when there is no fresh snapshot)."""
        if tracking != "active" or ts is None:
            self._reset()
            return False
        if self._last_ts is not None and ts <= self._last_ts:
            return False                                   # the same snapshot read again: not new evidence
        self._last_ts = ts
        if not self.is_stop_sign(tracking, hands, poses):
            self._reset(keep_ts=True)
            self._armed = True                             # the hands changed: it may fire again later
            return False
        if self._start is None:
            self._start, self._count = ts, 0
        self._count += 1
        if self._armed and now >= self._cooldown_until and self._count >= self.min_snapshots and ts - self._start >= self.hold_s:
            self._armed = False
            self._cooldown_until = now + self.cooldown_s
            return True
        return False

    def _reset(self, keep_ts=False):
        self._start, self._count = None, 0
        if not keep_ts:
            self._last_ts = None


class GestureWatcher:
    """Polls the board snapshot and calls `on_hush()` (from its own thread) when the gesture completes. Never raises."""

    def __init__(self, on_hush, *, fetch=None, clock=time.time, period_s=POLL_S, detector=None, log=None):
        self.on_hush, self.clock, self.period_s = on_hush, clock, period_s
        self._fetch = fetch or (lambda now: presence.read_board_ts(now))
        self.detector = detector or HushDetector()
        self._log = log or (lambda line: None)
        self._stop = threading.Event()
        self._thread = None

    def step(self):
        now = self.clock()
        tracking, hands, poses, ts = self._fetch(now)
        if self.detector.feed(tracking, hands, poses, ts, now):
            try:
                self.on_hush()
            except Exception as e:
                self._log(f"[gesture] hush action failed ({e.__class__.__name__})")
            return True
        return False

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                pass
            self._stop.wait(self.period_s)
