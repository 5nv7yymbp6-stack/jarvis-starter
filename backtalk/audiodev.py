"""What the Mac is ACTUALLY playing out of, asked of CoreAudio directly.

WHY THIS EXISTS, and why sounddevice is not enough: PortAudio resolves the
DEFAULT output device when it initialises and then never updates it for the
life of the process. Reproduced on this machine 2026-09-22 -- with PortAudio
held open, the Mac's default output was switched from the AirPods to the
built-in speakers, and PortAudio went on reporting the AirPods. **A
terminate()/initialize() did NOT fix it either**, which is the part worth
remembering: the ears' existing `_reopen_after_device_change` rebuild would
never have cured this. PortAudio's device LIST is refreshed and correct;
only its notion of "the default" is frozen.

That single stale value is the whole bug Yousif hit: wearing AirPods, his
replies came out of the MacBook speaker. `aec.enabled()` asked PortAudio
whether the output was still the built-in speakers, PortAudio said yes
forever, the echo-cancellation helper stayed bound to the speakers, and
every word went there. The 2026-09-21 fix that made `aec.enabled()`
re-decide every 3 seconds did not help, because it re-decided from the same
frozen source.

So the rule here: **CoreAudio for the truth, PortAudio only for the device
LIST, and every output stream opened against an EXPLICIT device index
resolved from the true name** -- never against PortAudio's idea of the
default. A CoreAudio query costs about 473 microseconds, so it is cheap
enough to ask on a short interval rather than trust a cache.

Everything here is guarded. A failure returns "" or None and the caller
falls back to the old behaviour; audio must never take the voice line down.
"""
import ctypes
import ctypes.util
import struct
import sys
import threading

from backtalk.vlog import log

_IS_MAC = sys.platform == "darwin"
_lock = threading.Lock()
_ca = _cf = None
_warned = False


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


_SYSTEM_OBJECT = 1
_GLOBAL = _fourcc("glob")
_DEFAULT_OUTPUT = _fourcc("dOut")
_NAME = _fourcc("lnam")
_UTF8 = 0x08000100


class _Address(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


def _libs():
    """CoreAudio + CoreFoundation, loaded once. None off macOS."""
    global _ca, _cf
    if not _IS_MAC:
        return None, None
    with _lock:
        if _ca is None:
            ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
            cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
            cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.c_long, ctypes.c_uint32]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            _ca, _cf = ca, cf
        return _ca, _cf


def default_output_name() -> str:
    """The device the Mac is playing out of RIGHT NOW, or "" if it can't
    be read. Never raises, never caches."""
    global _warned
    try:
        ca, cf = _libs()
        if ca is None:
            import sounddevice as sd       # not macOS: PortAudio is all there is
            return str(sd.query_devices(kind="output")["name"])
        addr = _Address(_DEFAULT_OUTPUT, _GLOBAL, 0)
        dev = ctypes.c_uint32(0)
        size = ctypes.c_uint32(4)
        if ca.AudioObjectGetPropertyData(
                ctypes.c_uint32(_SYSTEM_OBJECT), ctypes.byref(addr), 0, None,
                ctypes.byref(size), ctypes.byref(dev)) != 0 or not dev.value:
            return ""
        addr = _Address(_NAME, _GLOBAL, 0)
        ref = ctypes.c_void_p()
        size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        if ca.AudioObjectGetPropertyData(
                ctypes.c_uint32(dev.value), ctypes.byref(addr), 0, None,
                ctypes.byref(size), ctypes.byref(ref)) != 0 or not ref:
            return ""
        buf = ctypes.create_string_buffer(256)
        got = cf.CFStringGetCString(ref, buf, 256, _UTF8)
        cf.CFRelease(ref)
        return buf.value.decode("utf-8", "replace") if got else ""
    except Exception as e:
        if not _warned:
            _warned = True
            log(f"[audiodev] can't read the default output ({e!r}) -- "
                f"falling back to PortAudio's idea of it")
        try:
            import sounddevice as sd
            return str(sd.query_devices(kind="output")["name"])
        except Exception:
            return ""


def output_index(name: str):
    """PortAudio's index for a device NAME, or None.

    The list is trustworthy even when the default is not, so this is the
    safe half of PortAudio. Exact match first, then case-insensitive, then
    a substring either way (CoreAudio and PortAudio occasionally disagree
    on decoration around the same device)."""
    if not name:
        return None
    try:
        import sounddevice as sd
        devs = [(i, d) for i, d in enumerate(sd.query_devices())
                if d.get("max_output_channels", 0) > 0]
        for i, d in devs:
            if d["name"] == name:
                return i
        low = name.lower()
        for i, d in devs:
            if d["name"].lower() == low:
                return i
        for i, d in devs:
            dn = d["name"].lower()
            if low in dn or dn in low:
                return i
    except Exception:
        pass
    return None


def refresh():
    """Rebuild PortAudio's device list (it closes every open stream).

    Only worth calling when a device is missing from the list entirely --
    a genuinely new device that appeared after PortAudio started. It does
    NOT repair the stale default; that is what default_output_name() is
    for."""
    try:
        import sounddevice as sd
        try:
            sd._terminate()
        except Exception:
            pass
        sd._initialize()
        return True
    except Exception as e:
        log(f"[audiodev] device-list refresh failed: {e!r}")
        return False
