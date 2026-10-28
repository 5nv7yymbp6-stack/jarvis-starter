"""Desktop presence (roadmap Phase G: structured perception, screen awareness). READ-ONLY and PRIVACY-SAFE by construction.

It answers four cheap questions about the Mac, with NO permission prompt and no content:
    state        active | idle | away | locked | display_off | unknown     (from seconds since the last keyboard/mouse event, the session
                                                                           lock flag and the display sleep flag)
    app_category call | browser | terminal | editor | media | messaging | office | assistant | other | unknown
                 (the FRONTMOST application's bundle id mapped through a fixed allowlist; the app name itself is never kept)
    likely_call  the frontmost app is a call app. "Likely", never a fact.
    output       headphones | speakers | other | unknown                    (the default audio output device, classified by name)
    hand_tracking active | stale | off, `hands` (a count) and `poses` (one of pinch|fist|point|peace|open|other per hand) from the Bear Hands board's /api/state, ONLY when that snapshot is fresh:
                 a snapshot older than HANDS_FRESH_S means the browser page that does the tracking is closed or hidden, and the
                 board's own "hands: 0" is then NOT evidence that no hand is there. The board's cards (user content) are never read.

NEVER read, logged or stored: window titles, screen text, screenshots, the raw application name, the raw audio device name. A reader
that wants those needs Screen Recording / Accessibility permission, which this module deliberately does not use. Every field is an
enum or a number, matching the Situation layer's rule of never copying free text.

PURE CORE (`derive_state`, `categorize`, `classify_output`, `Presence`) plus injectable `Probes` (real ones use Quartz, AppKit and
sounddevice, imported lazily, each wrapped so a failure yields None) and a `PresenceMonitor` thread that refreshes a snapshot every few
seconds, so a reader on the event loop never blocks on a probe (idle takes about 40 ms, the audio device about 150 ms).
"""
import json
import threading
import time
import urllib.request
from dataclasses import dataclass

IDLE_S = 60.0                 # no keyboard or mouse for this long: idle
AWAY_S = 300.0                # ... and for this long: away
HANDS_URL = "http://127.0.0.1:4890/api/state"
HANDS_FRESH_S = 3.0           # the page posts about once a second while it is open and visible
STATES = ("active", "idle", "away", "locked", "display_off", "unknown")
CATEGORIES = ("call", "browser", "terminal", "editor", "media", "messaging", "office", "assistant", "other", "unknown")
OUTPUTS = ("headphones", "speakers", "other", "unknown")

# frontmost app -> category, by bundle id (a closed allowlist; anything else is "other")
_BUNDLES = {
    "us.zoom.xos": "call", "com.apple.FaceTime": "call", "com.microsoft.teams": "call", "com.microsoft.teams2": "call",
    "com.cisco.webexmeetingsapp": "call", "com.skype.skype": "call", "com.google.GoogleMeet": "call",
    "com.apple.Safari": "browser", "com.google.Chrome": "browser", "org.mozilla.firefox": "browser",
    "company.thebrowser.Browser": "browser", "com.microsoft.edgemac": "browser", "com.brave.Browser": "browser",
    "com.apple.Terminal": "terminal", "com.googlecode.iterm2": "terminal", "dev.warp.Warp-Stable": "terminal",
    "com.mitchellh.ghostty": "terminal",
    "com.microsoft.VSCode": "editor", "com.apple.dt.Xcode": "editor", "md.obsidian": "editor", "com.todesktop.230313mzl4w4u92": "editor",
    "com.apple.TextEdit": "editor", "com.apple.Notes": "editor",
    "com.spotify.client": "media", "com.apple.Music": "media", "com.apple.TV": "media", "com.apple.QuickTimePlayerX": "media",
    "org.videolan.vlc": "media", "com.apple.Podcasts": "media",
    "com.apple.MobileSMS": "messaging", "net.whatsapp.WhatsApp": "messaging", "ru.keepcoder.Telegram": "messaging",
    "com.hnc.Discord": "messaging", "com.tinyspeck.slackmacgap": "messaging", "com.apple.mail": "messaging",
    "com.apple.iWork.Pages": "office", "com.apple.iWork.Numbers": "office", "com.apple.iWork.Keynote": "office",
    "com.microsoft.Word": "office", "com.microsoft.Excel": "office", "com.microsoft.Powerpoint": "office",
    "com.anthropic.claudefordesktop": "assistant",
}
# transient system surfaces that own an on-screen window without being "the app in front"
_SYSTEM_UI = frozenset({"com.apple.WindowManager", "com.apple.dock", "com.apple.controlcenter", "com.apple.notificationcenterui",
                        "com.apple.loginwindow", "com.apple.Spotlight", "com.apple.systemuiserver"})
_HEADPHONE_WORDS = ("airpods", "headphone", "headset", "beats", "earbuds", "buds", "bluetooth")
_SPEAKER_WORDS = ("speaker", "macbook", "built-in", "builtin", "display", "studio display", "imac")


POSES = ("pinch", "fist", "point", "peace", "open", "other")


def read_board_ts(now, url=HANDS_URL, opener=urllib.request.urlopen, timeout=0.3):
    """(hand_tracking, hands, poses, ts): as read_board, plus the snapshot's own timestamp (None unless active) so a caller can tell a NEW
    snapshot from the same one read twice."""
    try:
        with opener(url, timeout=timeout) as r:
            data = json.loads(r.read(65536))
        ts, hands = data.get("ts"), data.get("hands")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not 0 <= now - ts <= HANDS_FRESH_S:
            return "stale", None, (), None
        if not (isinstance(hands, int) and not isinstance(hands, bool) and 0 <= hands <= 2):
            return "stale", None, (), None
        p = data.get("poses")
        poses = tuple(p) if isinstance(p, list) and len(p) == hands and all(isinstance(x, str) and x in POSES for x in p) else ()
        return "active", hands, poses, float(ts)
    except Exception:
        return "off", None, (), None


def read_board(now, url=HANDS_URL, opener=urllib.request.urlopen, timeout=0.3):
    """(hand_tracking, hands, poses): see read_hands. `poses` is a tuple of allowlisted labels, one per hand, only when the snapshot is
    fresh and the labels agree with the count; anything else gives (). Reads only `ts`, `hands`, `poses`."""
    return read_board_ts(now, url, opener, timeout)[:3]


def read_hands(now, url=HANDS_URL, opener=urllib.request.urlopen, timeout=0.3):
    """(hand_tracking, hands) from the board's state snapshot. 'off' = unreachable; 'stale' = reachable but its timestamp is old (the page
    is closed or hidden: the count is NOT trustworthy); 'active' = fresh, with the number of hands. Never raises; reads only `ts` and `hands`."""
    t, h, _ = read_board(now, url, opener, timeout)
    return t, h


def camera_in_use():
    """True when ANY camera on this Mac is running for ANY app, False when none is, None when it cannot be read. Uses CoreMediaIO's
    `DeviceIsRunningSomewhere` property: read-only, needs no camera permission, never opens the camera and never lights its indicator. It
    says nothing about WHICH app or what is in view. Caveat: the Bear Hands board's own hand tracking also runs the camera."""
    try:
        import ctypes

        class Addr(ctypes.Structure):
            _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]

        def cc(tag):
            return int.from_bytes(tag.encode("ascii"), "big")
        lib = ctypes.CDLL("/System/Library/Frameworks/CoreMediaIO.framework/CoreMediaIO")
        u32, p = ctypes.c_uint32, ctypes.POINTER
        lib.CMIOObjectGetPropertyDataSize.argtypes = [u32, p(Addr), u32, ctypes.c_void_p, p(u32)]
        lib.CMIOObjectGetPropertyDataSize.restype = ctypes.c_int32
        lib.CMIOObjectGetPropertyData.argtypes = [u32, p(Addr), u32, ctypes.c_void_p, u32, p(u32), ctypes.c_void_p]
        lib.CMIOObjectGetPropertyData.restype = ctypes.c_int32
        devs = Addr(cc("dev#"), cc("glob"), 0)
        size = u32()
        if lib.CMIOObjectGetPropertyDataSize(1, ctypes.byref(devs), 0, None, ctypes.byref(size)) != 0:
            return None
        n = size.value // 4
        if n == 0:
            return False
        ids, used = (u32 * n)(), u32()
        if lib.CMIOObjectGetPropertyData(1, ctypes.byref(devs), 0, None, size.value, ctypes.byref(used), ids) != 0:
            return None
        running = Addr(cc("gone"), cc("glob"), 0)
        answered = False
        for dev in ids:
            val, got = u32(), u32()
            if lib.CMIOObjectGetPropertyData(dev, ctypes.byref(running), 0, None, 4, ctypes.byref(got), ctypes.byref(val)) == 0:
                answered = True
                if val.value:
                    return True
        return False if answered else None
    except Exception:
        return None


def is_fullscreen(window_bounds, display_bounds, tol=3.0) -> bool:
    """True when a window's rectangle covers the whole display, menu bar included (a macOS full-screen Space: a movie, a presentation). Pure
    geometry on two (x, y, width, height) rectangles; a maximized but ordinary window leaves the menu bar uncovered and is False."""
    try:
        wx, wy, ww, wh = (float(v) for v in window_bounds)
        dx, dy, dw, dh = (float(v) for v in display_bounds)
    except (TypeError, ValueError):
        return False
    return abs(wx - dx) <= tol and abs(wy - dy) <= tol and ww >= dw - tol and wh >= dh - tol


def categorize(bundle_id) -> str:
    """The app category for a bundle id, or 'unknown' for no frontmost app. Anything not on the allowlist is 'other'."""
    if not isinstance(bundle_id, str) or not bundle_id:
        return "unknown"
    return _BUNDLES.get(bundle_id, "other")


def classify_output(device_name) -> str:
    """headphones | speakers | other | unknown from a device name. The name itself is not kept."""
    if not isinstance(device_name, str) or not device_name.strip():
        return "unknown"
    n = device_name.lower()
    if any(w in n for w in _HEADPHONE_WORDS):
        return "headphones"
    if any(w in n for w in _SPEAKER_WORDS):
        return "speakers"
    return "other"


def derive_state(idle_s, locked, display_asleep) -> str:
    """Locked and a sleeping display are definitive; otherwise the time since the last input decides. Unknown inputs never guess:
    no idle reading and no lock/display fact gives 'unknown'."""
    if locked is True:
        return "locked"
    if display_asleep is True:
        return "display_off"
    if not isinstance(idle_s, (int, float)) or isinstance(idle_s, bool) or idle_s != idle_s or idle_s < 0:
        return "unknown"
    if idle_s >= AWAY_S:
        return "away"
    return "idle" if idle_s >= IDLE_S else "active"


@dataclass(frozen=True)
class Presence:
    state: str = "unknown"
    idle_s: object = None
    app_category: str = "unknown"
    likely_call: bool = False
    output: str = "unknown"
    at: float = 0.0
    hand_tracking: str = "off"
    hands: object = None          # a count, only while hand_tracking == "active"
    poses: tuple = ()             # one allowlisted label per hand (pinch|fist|point|peace|open|other), only while active
    camera_in_use: object = None  # True | False | None (unknown): some camera is running for some app (see camera_in_use())
    fullscreen: object = None     # True | False | None (unknown): the frontmost window covers the whole display

    @property
    def user_away(self) -> bool:
        """Nobody is at the machine: locked, display asleep, or idle for AWAY_S. (Not 'unknown': that is not evidence.)"""
        return self.state in ("locked", "display_off", "away")

    def line(self) -> str:
        """The audit line: enums only."""
        cam = {True: "on", False: "off"}.get(self.camera_in_use, "?")
        return (f"[presence] state={self.state} app={self.app_category} call={'yes' if self.likely_call else 'no'} "
                f"output={self.output} cam={cam} full={ {True: 'yes', False: 'no'}.get(self.fullscreen, '?') } hands={self.hand_tracking}" + (f" poses={'+'.join(self.poses)}" if self.poses else ""))


@dataclass
class Probes:
    """Four callables, each returning a value or None (unknown). `front` returns a bundle id string."""
    idle: object = None
    front: object = None
    locked: object = None
    display_asleep: object = None
    output: object = None
    hands: object = None          # () -> (tracking, hands)
    camera: object = None         # () -> True | False | None
    fullscreen: object = None     # () -> True | False | None


def _safe(fn):
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def snapshot(probes: Probes, now: float, output_cache=None) -> Presence:
    """One reading from the probes. `output_cache` is (value, at) from an earlier reading to avoid the slow audio probe."""
    idle = _safe(probes.idle)
    bundle = _safe(probes.front)
    cat = categorize(bundle)
    out = output_cache[0] if output_cache else classify_output(_safe(probes.output))
    h = _safe(probes.hands)
    cam = _safe(probes.camera)
    full = _safe(probes.fullscreen)
    ok = isinstance(h, tuple) and len(h) in (2, 3) and h[0] in ("active", "stale", "off")
    tracking, hands = (h[0], h[1]) if ok else ("off", None)
    poses = tuple(x for x in h[2] if x in POSES) if ok and len(h) == 3 and isinstance(h[2], tuple) and tracking == "active" else ()
    return Presence(derive_state(idle, _safe(probes.locked), _safe(probes.display_asleep)),
                    float(idle) if isinstance(idle, (int, float)) and not isinstance(idle, bool) and idle == idle else None,
                    cat, cat == "call", out, now, tracking, hands if tracking == "active" else None, poses,
                    cam if isinstance(cam, bool) else None, full if isinstance(full, bool) else None)


# ---- the real probes (lazy imports; a missing module or a failed call is just None) ---------------------------------------

def real_probes() -> Probes:
    def idle():
        from Quartz import CGEventSourceSecondsSinceLastEventType, kCGAnyInputEventType, kCGEventSourceStateCombinedSessionState
        return float(CGEventSourceSecondsSinceLastEventType(kCGEventSourceStateCombinedSessionState, kCGAnyInputEventType))

    def front():
        # NOT NSWorkspace.frontmostApplication(): in a process with no main run loop it is cached at first use and never updates
        # (proved live 2026-10-19: it stayed on one app across switches). The on-screen window list is front-to-back and needs no
        # permission for the owner pid; the window NAME (which would need Screen Recording) is never requested.
        from AppKit import NSRunningApplication
        from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly
        for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []:
            pid = w.get("kCGWindowOwnerPID")
            if w.get("kCGWindowLayer") != 0 or not pid:
                continue
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
            bid = str(app.bundleIdentifier()) if app is not None and app.bundleIdentifier() else None
            if bid and bid not in _SYSTEM_UI:
                return bid
        return None

    def fullscreen():
        # same window list as front(): the topmost ordinary window's SIZE against the main display's. Titles and contents are never read.
        from Quartz import (CGDisplayBounds, CGMainDisplayID, CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly)
        from AppKit import NSRunningApplication
        db = CGDisplayBounds(CGMainDisplayID())
        for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []:
            pid = w.get("kCGWindowOwnerPID")
            if w.get("kCGWindowLayer") != 0 or not pid:
                continue
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
            bid = str(app.bundleIdentifier()) if app is not None and app.bundleIdentifier() else None
            if not bid or bid in _SYSTEM_UI:
                continue
            b = w.get("kCGWindowBounds") or {}
            return is_fullscreen((b.get("X"), b.get("Y"), b.get("Width"), b.get("Height")),
                                 (db.origin.x, db.origin.y, db.size.width, db.size.height))
        return None

    def locked():
        from Quartz import CGSessionCopyCurrentDictionary
        d = CGSessionCopyCurrentDictionary()
        return None if d is None else bool(d.get("CGSSessionScreenIsLocked", False))      # the key is absent when unlocked

    def display_asleep():
        from Quartz import CGDisplayIsAsleep, CGMainDisplayID
        return bool(CGDisplayIsAsleep(CGMainDisplayID()))

    def output():
        import sounddevice
        return sounddevice.query_devices(kind="output").get("name")

    return Probes(idle, front, locked, display_asleep, output, lambda: read_board(time.time()), camera_in_use, fullscreen)


class PresenceMonitor:
    """A daemon thread that refreshes the snapshot every `period_s`; `latest()` never blocks and never raises. The slow audio-device probe
    runs every `output_every_s`. `on_change(presence)` is called (from the thread) when the enums change, never for the idle seconds."""

    def __init__(self, probes=None, *, period_s=3.0, output_every_s=30.0, clock=time.time, on_change=None):
        self.probes = probes or real_probes()
        self.period_s, self.output_every_s, self._clock = period_s, output_every_s, clock
        self.on_change = on_change
        self._latest = Presence(at=clock())
        self._out = None                       # (value, at)
        self._stop = threading.Event()
        self._thread = None

    def refresh(self):
        now = self._clock()
        if self._out is None or now - self._out[1] >= self.output_every_s:
            self._out = (classify_output(_safe(self.probes.output)), now)
        p = snapshot(self.probes, now, self._out)
        prev, self._latest = self._latest, p
        if self.on_change and (p.state, p.app_category, p.likely_call, p.output, p.hand_tracking, p.poses, p.camera_in_use, p.fullscreen) != (
                prev.state, prev.app_category, prev.likely_call, prev.output, prev.hand_tracking, prev.poses, prev.camera_in_use, prev.fullscreen):
            try:
                self.on_change(p)
            except Exception:
                pass
        return p

    def latest(self) -> Presence:
        return self._latest

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="presence", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:
                pass
            self._stop.wait(self.period_s)
