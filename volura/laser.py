"""The laser on the other end of the cable: a board that answers, and a job that waits.

No Qt in this file. :mod:`volura.grbl` knows the words; this module knows the
conversation: which port the board hangs on, the thread that reads and writes it, and
what the board means by what it says while a job runs. The window that shows all of this
lives in :mod:`volura.laser_window`.

A GRBL board answers every block with `ok` or `error:N`, and no faster than it plans
them out of a buffer of 127 bytes. Writing more than the buffer holds loses characters,
which on a laser is a burn in the wrong place, so the blocks of a job go out through
:func:`volura.grbl.Sender`, which keeps track of the bytes the board has not answered for
yet.

The board says nothing unless it is asked, so the thread asks for a status report (`?`)
every `poll` seconds; that is where the position on the screen and the state of the board
come from. The real-time commands (`?`, `!`, `~` and the reset byte) are read *between*
the blocks that are already waiting, which is why they go out as single bytes without a
newline and why pausing works while a job is running.

Threading: the cable is read on a thread of its own, because a board that is planning
holds the port open longer than a window may wait. Everything the conversation knows
lives in :class:`Session`, which has no thread and no clock of its own; :class:`Worker` is
the only thing that writes it, and the window only reads. Every attribute of a session is
a plain object that is replaced whole, so the window never sees half a position.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import grbl
from .i18n import tr

#: how often the window looks at the session (seconds)
UPDATE_SECONDS = 0.2

#: how long the thread waits when the cable has nothing to say (seconds)
IDLE_SECONDS = 0.01

#: the distances of the jog pad, in millimetres
JOG_STEPS = (0.1, 1.0, 5.0, 10.0, 50.0)

#: the speeds of the jog pad, in millimetres per minute
JOG_FEEDS = (500, 1000, 3000, 6000)

#: the settings keys of the window, so the board of yesterday is the board of today
PORT_KEY = "laser/port"
BAUD_KEY = "laser/baud"
STEP_KEY = "laser/step"
FEED_KEY = "laser/jog_feed"


class CableError(RuntimeError):
    """The port is not there, is busy, or is not a board Volura can talk to."""


# --------------------------------------------------------------------------
#  the cable
# --------------------------------------------------------------------------

def ports() -> list[tuple[str, str]]:
    """The serial ports the computer sees, as ``(device, description)``.

    Empty when pyserial is not installed: a window that cannot see the board is better
    than a program that will not start, and the window says what is missing.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [(p.device, p.description) for p in list_ports.comports()]


def has_serial() -> bool:
    """Whether the computer has the package that opens a serial port."""
    try:
        import serial                                    # noqa: F401
    except ImportError:
        return False
    return True


def open_port(device, baud=grbl.DEFAULT_BAUD):
    """The open serial port of a GRBL board.

    pyserial is imported here and not at the top: it is a small package, but it is
    needed for nothing else in Volura and a build without it should still open.

    The read timeout is nothing at all, because the thread reads what is there and never
    waits for more; the write timeout is generous, because a board that is planning a
    long block can hold the cable for a moment.
    """
    try:
        import serial
    except ImportError as e:
        raise CableError(tr("Volura cannot talk to a laser without the pyserial package. "
                            "Install it with: pip install pyserial")) from e
    try:
        return serial.Serial(port=str(device), baudrate=int(baud), timeout=0,
                             write_timeout=2.0)
    except Exception as e:
        raise CableError(tr("{port} is not a port Volura can open: {why}").format(
            port=device, why=e)) from e


# --------------------------------------------------------------------------
#  the conversation
# --------------------------------------------------------------------------

class Session:
    """What the board is and does, and what is still waiting to go out to it.

    The session has no thread and no clock: :meth:`pending` says what may go out now and
    :meth:`feed` tells it what came back. :class:`Worker` is what makes that happen on a
    real cable, and a test can do it by hand.
    """

    def __init__(self, buffer=grbl.BUFFER_BYTES):
        self.buffer = int(buffer)
        self.state = ""                 #: what the board says it is: "Idle", "Run", ...
        self.mpos = None                #: the machine position, three numbers
        self.wpos = None                #: the position in the work
        self.wco = None                 #: the work offset
        self.feed_rate = 0.0            #: the feed the board is moving at, mm/min
        self.speed = 0.0                #: the spindle or laser power it is using now
        self.line = None                #: the block of the job the board is at
        self.feed_override = 100.0
        self.rapid_override = 100.0
        self.power_override = 100.0
        self.boot = ""                  #: what the board said when the cable was opened
        self.messages: list[str] = []   #: what $$ and [MSG:] answered, newest last
        self.errors: list[str] = []     #: what the board refused, newest last
        self.sender: grbl.Sender | None = None
        self._urgent: deque = deque()   #: the bytes that may not wait for the job

    # -- what the board is
    @property
    def state_label(self) -> str:
        """The state of the board in the words of the interface."""
        return grbl.state_label(self.state)

    @property
    def running(self) -> bool:
        """Whether a job is going out to the board (a finished one is not)."""
        return self.sender is not None and not self.sender.done

    @property
    def paused(self) -> bool:
        return self.sender is not None and self.sender.paused

    @property
    def fraction(self) -> float:
        """How far the job is, 0.0 to 1.0; nothing to do counts as nothing done."""
        return self.sender.fraction if self.sender is not None else 0.0

    @property
    def position(self):
        """The position to show: the work when the board knows it, else the machine."""
        return self.wpos if self.wpos is not None else self.mpos

    # -- what goes out
    def send(self, text) -> None:
        """A block to the board: the text with a newline behind it, as the board reads it."""
        text = str(text).strip()
        if text:
            self._urgent.append((text + "\n").encode("ascii", "replace"))

    def realtime(self, cmd) -> None:
        """A real-time command: one byte the board reads between the blocks it has."""
        if cmd:
            self._urgent.append(str(cmd).encode("ascii", "replace"))

    def override(self, kind, action) -> None:
        """Change the feed, the rapid or the power while the board is moving."""
        byte = grbl.override(kind, action)
        if byte:
            self._urgent.append(byte)

    def start(self, lines) -> bool:
        """Begin a job of the blocks in `lines`; False when a job is already running."""
        if self.running:
            return False
        self.sender = grbl.Sender(lines, buffer=self.buffer)
        return True

    def pause(self) -> None:
        """Feed hold: the board slows to a stop with the beam off and keeps what it has."""
        if self.sender is not None:
            self.sender.pause()
        self.realtime(grbl.FEED_HOLD)

    def resume(self) -> None:
        if self.sender is not None:
            self.sender.resume()
        self.realtime(grbl.CYCLE_START)

    def stop(self) -> None:
        """End the job and empty what the board has queued; it lands in the alarm state.

        The sender is cancelled and not thrown away: the thread may be standing in
        :meth:`pending` with it in its hand, and the numbers of the job on the screen are
        better left where they are than forgotten on the way out.
        """
        if self.sender is not None:
            self.sender.cancel()
        self.realtime(grbl.RESET)

    def pending(self) -> bytes:
        """What may go out now: what may not wait first, then as much of the job as fits."""
        if self._urgent:
            return self._urgent.popleft()
        if self.sender is not None:
            data = self.sender.take()
            if data:
                return data
        return b""

    # -- what comes back
    def feed(self, line) -> None:
        """One line of the board, as it stands and without its newline."""
        kind, rest = grbl.classify(line)
        if kind == "ok":
            if self.sender is not None:
                self.sender.confirm()
        elif kind == "error":
            if self.sender is not None:
                self.sender.confirm(ok=False)
                self.sender.error(rest)
            self.errors.append(grbl.error_text(rest))
        elif kind == "status":
            self._report(grbl.parse_report(line))
        elif kind == "alarm":
            self.state = "Alarm"
            self.errors.append(rest)
        elif kind == "message":
            self.messages.append(rest)
        elif kind == "text":
            # the first thing a board says is who it is; the rest is the answer of $$
            if self.boot:
                self.messages.append(rest)
            else:
                self.boot = rest

    def _report(self, info: dict) -> None:
        """Take the numbers of a status report into the session; a field that is not
        there says nothing and leaves what the board said before."""
        if not info:
            return
        if info.get("state"):
            self.state = info["state"]
        for key, name in (("mpos", "mpos"), ("wpos", "wpos"), ("wco", "wco"),
                          ("feed", "feed_rate"), ("speed", "speed"), ("line", "line")):
            if key in info:
                setattr(self, name, info[key])
        for name in ("feed_override", "rapid_override", "power_override"):
            if info.get(name) is not None:
                setattr(self, name, info[name])


class Worker(threading.Thread):
    """The thread that reads and writes the cable, so the window keeps answering.

    The thread is the only one that touches the port and the only one that feeds the
    session. It asks for a status report whenever it has nothing else to write, and it
    stops by itself when the cable breaks; :attr:`problem` then says what happened.
    """

    def __init__(self, port, session=None, poll=0.2, name="volura-laser"):
        super().__init__(daemon=True, name=name)
        self.port = port
        self.session = session if session is not None else Session()
        self.poll = float(poll)
        self.problem = ""
        self._stop = threading.Event()
        self._left = b""

    def run(self) -> None:
        asked = time.monotonic()
        while not self._stop.is_set():
            try:
                self._read()
                out = self.session.pending()
                if not out and time.monotonic() - asked >= self.poll:
                    out = grbl.STATUS.encode("ascii")
                    asked = time.monotonic()
                if out:
                    self.port.write(out)
                else:
                    time.sleep(IDLE_SECONDS)
            except Exception as e:                      # the cable is gone: say so and end
                self.problem = str(e) or type(e).__name__
                break
        self._stop.set()
        self._let_go()

    def _read(self) -> None:
        """Everything waiting on the cable, cut into the lines the board sends."""
        waiting = int(self.port.in_waiting or 0)
        if waiting <= 0:
            return
        data = self.port.read(waiting)
        if not data:
            return
        lines = (self._left + bytes(data)).split(b"\n")
        self._left = lines.pop()                        # the next line is not finished yet
        for raw in lines:
            self.session.feed(raw.decode("ascii", "replace").strip())

    def shutdown(self, timeout=3.0) -> None:
        """Stop the thread and let go of the port, so the next board can be opened."""
        self._stop.set()
        if self.is_alive():
            self.join(timeout)
        self._let_go()

    def _let_go(self) -> None:
        try:
            self.port.close()
        except Exception:
            pass                                        # already closed, or no cable at all
