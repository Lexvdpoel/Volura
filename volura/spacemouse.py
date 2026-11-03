"""3D mice from 3Dconnexion: SpaceMouse, SpaceNavigator, SpacePilot, SpaceExplorer, ...

Three ways to read the motion; the first one that works is used:

* **3DxWare** (Windows, with the 3Dconnexion driver): through ``siappdll.dll``, the
  classic interface that both 3DxWare 10 and the older 3DxSoftware support. The driver
  then knows Volura is a 3D application: it makes a profile "Volura" of its own
  (sensitivity, inverted axes, buttons) and stops emulating the mouse wheel.
* **Raw Input** (Windows, without a driver): read the HID device directly.
  Older devices (SpaceNavigator, SpacePilot, SpaceExplorer, SpaceMouse Pro) send
  translation and rotation in two separate reports; newer ones (SpaceMouse Wireless,
  Compact, Enterprise, the Universal Receiver) in one report.
* **spacenavd** (Linux): the free background daemon, through its socket.

Everything comes out as :class:`Motion` in the 3Dconnexion coordinate system: X to the
right, Y up, Z towards the user; rotations right-handed about those axes. A full
deflection is about 1.0.
"""
from __future__ import annotations

import ctypes
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, QSocketNotifier, Signal
from PySide6.QtWidgets import QApplication

from .i18n import tr

FULL_SCALE = 350.0          # raw value at full deflection
DEADZONE = 0.02
MAX_DT = 0.1                # a pause longer than this does not count as motion

# The buttons 3DxWare passes on (virtual keys V3DK_*) and what Volura does with them.
V3DK_COMMANDS = {2: "fit", 3: "top", 4: "left", 5: "right", 6: "front", 7: "bottom",
                 8: "back", 11: "iso", 12: "iso"}
# Without a driver: button 1 (left) to iso, button 2 (right) to zoom extents.
RAW_BUTTON_COMMANDS = {0: "iso", 1: "fit"}


@dataclass
class Motion:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    dt: float = 1 / 60          # seconds since the previous motion

    def is_zero(self):
        return not any((self.tx, self.ty, self.tz, self.rx, self.ry, self.rz))


def normalize(values, dt=1 / 60) -> Motion:
    """Six raw values (in the 3Dconnexion coordinate system) to a Motion with a dead zone."""
    out = []
    for v in values:
        v = float(v) / FULL_SCALE
        out.append(0.0 if abs(v) < DEADZONE else max(-2.0, min(2.0, v)))
    return Motion(*out, dt=max(0.001, min(MAX_DT, float(dt))))


def hid_to_si(tx, ty, tz, rx, ry, rz):
    """HID axes (X right, Y towards the user, Z down) to those of 3Dconnexion."""
    return tx, -tz, ty, rx, -rz, ry


class HidState:
    """Keeps track of the last translation, rotation and buttons of a HID device."""

    def __init__(self):
        self.trans = [0, 0, 0]
        self.rot = [0, 0, 0]
        self.buttons = 0

    def feed(self, report: bytes):
        """Process a HID report. Returns ("motion", 6 values) or ("buttons", pressed)."""
        if not report:
            return None
        rid, data = report[0], bytes(report[1:])
        if rid == 1 and len(data) >= 12:           # new: translation and rotation together
            v = struct.unpack_from("<6h", data)
            self.trans, self.rot = list(v[:3]), list(v[3:])
        elif rid == 1 and len(data) >= 6:          # old: translation only
            self.trans = list(struct.unpack_from("<3h", data))
        elif rid == 2 and len(data) >= 6:          # old: rotation only
            self.rot = list(struct.unpack_from("<3h", data))
        elif rid == 3:                             # buttons as a bit mask
            mask = int.from_bytes(data[:4].ljust(4, b"\0"), "little")
            pressed = mask & ~self.buttons
            self.buttons = mask
            return ("buttons", pressed)
        else:
            return None
        return ("motion", hid_to_si(*self.trans, *self.rot))


def bits(mask):
    return [i for i in range(32) if mask >> i & 1]


class _Timer:
    """The time between two motions, so the speed does not depend on the sample rate."""

    def __init__(self):
        self.last = None

    def dt(self):
        now = time.monotonic()
        dt = 1 / 60 if self.last is None else now - self.last
        self.last = now
        return dt if dt <= MAX_DT else 1 / 60


# ---------------------------------------------------------------- Windows: 3DxWare
SI_EVENT_TYPES = {"button": 1, "motion": 2, "zero": 4, "button_press": 12, "button_release": 13}
SI_NOT_EVENT = 7
SI_ANY_DEVICE = -1
SI_EVENT = 1
_EVENT_SIZE = 8192            # large enough for any SiSpwEvent


def parse_si_event(buf: bytes):
    """A SiSpwEvent: ("motion", 6 values, period in ms), ("press", button) or
    ("buttons", pressed mask); otherwise None."""
    kind = struct.unpack_from("<i", buf, 0)[0]
    if kind == SI_EVENT_TYPES["motion"]:
        # union at 4 bytes: SiButtonData (4 x uint32), mData[6] (int32), period (int32)
        _last, _cur, pressed, _rel = struct.unpack_from("<4I", buf, 4)
        values = struct.unpack_from("<6i", buf, 20)
        period = struct.unpack_from("<i", buf, 44)[0]
        return ("motion", values, period)
    if kind == SI_EVENT_TYPES["button"]:
        _last, _cur, pressed, _rel = struct.unpack_from("<4I", buf, 4)
        return ("buttons", pressed)
    if kind == SI_EVENT_TYPES["button_press"]:
        return ("press", struct.unpack_from("<I", buf, 4)[0])
    if kind == SI_EVENT_TYPES["zero"]:
        return ("motion", (0, 0, 0, 0, 0, 0), 0)
    return None


class _NativeFilter(QAbstractNativeEventFilter):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG" or bytes(event_type) == b"windows_generic_MSG":
            try:
                from ctypes import wintypes
                msg = wintypes.MSG.from_address(int(message))
                self.handler(msg)
            except Exception:              # a 3D mouse must never crash the application
                pass
        return False, 0


class SiAppBackend:
    """3DxWare through siappdll.dll; Volura then gets a profile of its own in 3DxWare."""

    name = "3DxWare"

    def __init__(self, owner, hwnd):
        self.owner = owner
        dll = ctypes.WinDLL("siappdll")
        dll.SiInitialize.restype = ctypes.c_int
        dll.SiOpenWinInit.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        dll.SiOpen.restype = ctypes.c_void_p
        dll.SiOpen.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p)
        dll.SiGetEventWinInit.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                          ctypes.c_ssize_t)
        dll.SiGetEvent.restype = ctypes.c_int
        dll.SiGetEvent.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        dll.SiClose.argtypes = (ctypes.c_void_p,)
        self.dll = dll
        if dll.SiInitialize() != 0:
            raise OSError(tr("3DxWare does not respond"))
        self.open_data = ctypes.create_string_buffer(4096)
        dll.SiOpenWinInit(self.open_data, ctypes.c_void_p(hwnd))
        self.handle = dll.SiOpen(b"Volura", SI_ANY_DEVICE, None, SI_EVENT, self.open_data)
        if not self.handle:
            dll.SiTerminate()
            raise OSError(tr("3DxWare found no 3D mouse"))
        self.event_data = ctypes.create_string_buffer(64)
        self.event = ctypes.create_string_buffer(_EVENT_SIZE)
        self.filter = _NativeFilter(self.on_msg)
        QApplication.instance().installNativeEventFilter(self.filter)

    def on_msg(self, msg):
        if msg.message < 0xC000:           # 3DxWare uses a registered message
            return
        self.dll.SiGetEventWinInit(self.event_data, msg.message, msg.wParam or 0, msg.lParam or 0)
        ctypes.memset(self.event, 0, 64)
        if self.dll.SiGetEvent(self.handle, 0, self.event_data, self.event) == SI_NOT_EVENT:
            return
        ev = parse_si_event(self.event.raw)
        if ev is None:
            return
        if ev[0] == "motion":
            period = ev[2] / 1000.0 if 0 < ev[2] < 1000 else self.owner.timer.dt()
            self.owner.emit_motion(ev[1], period)
        elif ev[0] == "press":
            self.owner.emit_command(V3DK_COMMANDS.get(ev[1]))
        elif ev[0] == "buttons":
            for b in bits(ev[1]):
                self.owner.emit_command(V3DK_COMMANDS.get(b))

    def close(self):
        app = QApplication.instance()
        if app is not None:
            app.removeNativeEventFilter(self.filter)
        if self.handle:
            self.dll.SiClose(self.handle)
            self.handle = None
            self.dll.SiTerminate()


# ---------------------------------------------------------------- Windows: Raw Input
WM_INPUT = 0x00FF
RID_INPUT = 0x10000003
RIM_TYPEHID = 2
RIDEV_REMOVE = 0x00000001
USAGE_PAGE_GENERIC = 0x01
USAGE_MULTI_AXIS = 0x08


class _RawInputDevice(ctypes.Structure):
    _fields_ = [("usUsagePage", ctypes.c_ushort), ("usUsage", ctypes.c_ushort),
                ("dwFlags", ctypes.c_uint), ("hwndTarget", ctypes.c_void_p)]


class RawInputBackend:
    """Read the HID device itself, when 3DxWare is not there."""

    name = "Raw Input"

    def __init__(self, owner, hwnd):
        self.owner = owner
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetRawInputData.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_uint), ctypes.c_uint)
        self.user32.GetRawInputData.restype = ctypes.c_uint
        dev = _RawInputDevice(USAGE_PAGE_GENERIC, USAGE_MULTI_AXIS, 0, ctypes.c_void_p(hwnd))
        if not self.user32.RegisterRawInputDevices(ctypes.byref(dev), 1, ctypes.sizeof(dev)):
            raise OSError(tr("Raw Input not available"))
        self.header = 8 + 2 * ctypes.sizeof(ctypes.c_void_p)      # RAWINPUTHEADER
        self.states = {}
        self.filter = _NativeFilter(self.on_msg)
        QApplication.instance().installNativeEventFilter(self.filter)

    def on_msg(self, msg):
        if msg.message != WM_INPUT:
            return
        size = ctypes.c_uint(0)
        hdr = self.header
        self.user32.GetRawInputData(ctypes.c_void_p(msg.lParam), RID_INPUT, None,
                                    ctypes.byref(size), hdr)
        if not size.value or size.value > 4096:
            return
        buf = ctypes.create_string_buffer(size.value)
        if self.user32.GetRawInputData(ctypes.c_void_p(msg.lParam), RID_INPUT, buf,
                                       ctypes.byref(size), hdr) != size.value:
            return
        raw = buf.raw
        kind = struct.unpack_from("<I", raw, 0)[0]
        if kind != RIM_TYPEHID:
            return
        device = raw[8:8 + ctypes.sizeof(ctypes.c_void_p)]
        each, count = struct.unpack_from("<II", raw, hdr)
        state = self.states.setdefault(device, HidState())
        for i in range(count):
            report = raw[hdr + 8 + i * each: hdr + 8 + (i + 1) * each]
            self.owner.handle_hid(state, report)

    def close(self):
        app = QApplication.instance()
        if app is not None:
            app.removeNativeEventFilter(self.filter)
        dev = _RawInputDevice(USAGE_PAGE_GENERIC, USAGE_MULTI_AXIS, RIDEV_REMOVE, None)
        self.user32.RegisterRawInputDevices(ctypes.byref(dev), 1, ctypes.sizeof(dev))


# ---------------------------------------------------------------- Linux: spacenavd
SPNAV_SOCKETS = ("/var/run/spnav.sock", "/run/spnav.sock")


def parse_spnav(packet: bytes):
    """A message from spacenavd (8 x int32): motion or a pressed button."""
    v = struct.unpack("=8i", packet)
    if v[0] == 0:
        x, y, z, rx, ry, rz = v[1:7]
        return ("motion", (x, y, -z, rx, ry, -rz), v[7])   # spacenavd: Z into the screen
    if v[0] == 1:
        return ("press", v[1])
    return None


class SpnavBackend:
    name = "spacenavd"

    def __init__(self, owner, hwnd=None):
        self.owner = owner
        path = next((p for p in SPNAV_SOCKETS if os.path.exists(p)), None)
        if path is None:
            raise OSError(tr("spacenavd is not running"))
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.sock.setblocking(False)
        self.buf = b""
        self.notifier = QSocketNotifier(self.sock.fileno(), QSocketNotifier.Read)
        self.notifier.activated.connect(self.read)

    def read(self, *args):
        try:
            data = self.sock.recv(4096)
        except BlockingIOError:
            return
        if not data:
            self.notifier.setEnabled(False)
            return
        self.buf += data
        while len(self.buf) >= 32:
            packet, self.buf = self.buf[:32], self.buf[32:]
            ev = parse_spnav(packet)
            if ev is None:
                continue
            if ev[0] == "motion":
                period = ev[2] / 1000.0 if 0 < ev[2] < 1000 else self.owner.timer.dt()
                self.owner.emit_motion(ev[1], period)
            else:
                self.owner.emit_command(RAW_BUTTON_COMMANDS.get(ev[1]))

    def close(self):
        self.notifier.setEnabled(False)
        self.sock.close()


# ---------------------------------------------------------------- together
def backends():
    if sys.platform == "win32":
        return (SiAppBackend, RawInputBackend)
    if sys.platform.startswith("linux"):
        return (SpnavBackend,)
    return ()


class SpaceMouse(QObject):
    """Reads a 3D mouse and emits :class:`Motion` and button commands as signals."""

    motion = Signal(object)
    command = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend = None
        self.timer = _Timer()
        self.problems = []

    @property
    def name(self):
        return self.backend.name if self.backend is not None else None

    def start(self, hwnd=0):
        """Try the backends in order; returns the name of the one that works, or None."""
        self.stop()
        self.problems = []
        for cls in backends():
            try:
                self.backend = cls(self, hwnd)
                return self.backend.name
            except Exception as e:          # no driver, no device, ...
                self.problems.append(f"{cls.name}: {e}")
        return None

    def stop(self):
        if self.backend is not None:
            try:
                self.backend.close()
            finally:
                self.backend = None

    def handle_hid(self, state, report):
        ev = state.feed(report)
        if ev is None:
            return
        if ev[0] == "motion":
            self.emit_motion(ev[1], self.timer.dt())
        else:
            for b in bits(ev[1]):
                self.emit_command(RAW_BUTTON_COMMANDS.get(b))

    def emit_motion(self, values, dt):
        if dt is None:
            dt = self.timer.dt()
        else:
            self.timer.last = time.monotonic()
        m = normalize(values, dt)
        if not m.is_zero():
            self.motion.emit(m)

    def emit_command(self, name):
        if name:
            self.command.emit(name)
