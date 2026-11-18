"""GRBL: the words a laser controller speaks, both ways.

No Qt in this file. What goes to the board and what comes back from it is plain text, so
it can be tested without a machine and reused by the controller in ``volura/laser.py``.

A **GRBL board** is a small microcontroller between the computer and the laser. The
computer writes blocks to its serial port, the board plans the moves and switches the
beam with the words of the spindle: **M3** puts it on at a fixed power while the head
moves, **M4** lets the power follow the feed so a corner burns less, **M5** switches it
off. The power is the **S word**, a whole number up to `$30` of the board.

Everything the board sends back is one of a handful of lines: **`ok`** for a block it
accepted, **`error:N`** for one it refused, **`echo:`** when echo (`$E`) is on,
**`[MSG: ...]`** for what `$` commands answer and a **status report** between angle
brackets, which is what the position on the screen comes from. The report is only sent
when the computer asks for it with `?`, because the board has nothing else to say.

Two things make this more than a text file over a cable. The board has a small input
buffer, so the computer may not write faster than the board reads (:class:`Sender`);
and the **real-time commands** (`?`, `!`, `~` and the reset byte) are read between the
blocks that are already there, which is why pausing and stopping work while a job runs
and why they are single bytes without a newline.
"""
from __future__ import annotations

from collections import deque

from .i18n import tr

#: the serial speeds a GRBL board is set to; 115200 is the default of GRBL 1.1
BAUD_RATES = (115200, 57600, 38400, 19200, 9600)
DEFAULT_BAUD = 115200

#: bytes the board keeps for the blocks it has not read yet; one is left for the newline
BUFFER_BYTES = 127

#: decimals of a coordinate; GRBL works in thousandths of a millimetre
DECIMALS = 3

#: the axes a board can drive: X Y Z, and A and B on a mill with rotary axes, in degrees
AXES = "XYZAB"

# Real-time commands: read between the blocks that are already waiting, so they act now.
#: ask for a status report
STATUS = "?"
#: feed hold: the head slows to a stop and the beam goes off, the queue stays
FEED_HOLD = "!"
#: go on after a feed hold
CYCLE_START = "~"
#: reset: the planner and the buffer are emptied and the board lands in the alarm state
RESET = "\x18"

#: the override bytes, per what they change and per how far they change it
OVERRIDES = {
    "feed": {"set": b"\x90", "up10": b"\x91", "down10": b"\x92", "up1": b"\x93",
             "down1": b"\x94"},
    "rapid": {"set": b"\x95", "half": b"\x96", "quarter": b"\x97"},
    "power": {"set": b"\x99", "up10": b"\x9a", "down10": b"\x9b", "up1": b"\x9c",
              "down1": b"\x9d"},
}

#: the state of the board as the user reads it
STATE_LABELS = {
    "Idle": tr("Idle"), "Run": tr("Cutting"), "Hold": tr("Paused"), "Jog": tr("Jogging"),
    "Check": tr("Checking"), "Home": tr("Homing"), "Queue": tr("Queued"),
    "Sleep": tr("Sleeping"), "Alarm": tr("Alarm"), "Door": tr("Door open"),
}

#: the error numbers of GRBL, so a refused block can be explained instead of quoted
ERRORS = {
    "1": tr("Expected command"),
    "2": tr("Reportable error"),
    "3": tr("Bad number format"),
    "4": tr("Negative value"),
    "5": tr("Hard limit"),
    "6": tr("Soft limit"),
    "8": tr("Critical event"),
    "9": tr("Alarm abort"),
    "10": tr("Feed max exceed"),
    "11": tr("Conversion alarm"),
    "12": tr("Invalid statement"),
    "13": tr("Set position under motion"),
    "20": tr("Axis words"),
    "21": tr("Modal group"),
    "22": tr("Coordinate values"),
    "23": tr("Probed axis values"),
    "24": tr("G2/G3 arc values"),
    "25": tr("Feed rate"),
    "26": tr("Words used twice"),
    "27": tr("Expected axis word"),
    "28": tr("Invalid axis word"),
    "29": tr("G-code words"),
    "30": tr("Value out of range"),
    "31": tr("Probing failure"),
    "32": tr("Check mode"),
    "33": tr("Min travel limit exceeded"),
    "34": tr("Radius value"),
    "35": tr("Axis word(s) not allowed"),
    "36": tr("Zero-length tool vector"),
    "37": tr("Number of tool axis words"),
}


def _num(v) -> str:
    """A coordinate as the board reads it: always the same number of decimals."""
    return f"{float(v):.{DECIMALS}f}"


# --------------------------------------------------------------------------
#  from a G-code file to the blocks the board gets
# --------------------------------------------------------------------------

def job_lines(text) -> list[str]:
    """The G-code of a job as the blocks to write, in order.

    Comments (`;` to the end of the line) and empty lines are dropped: the board answers
    them with `ok` too, and every block costs a round trip in the character-counting
    :class:`Sender`. The rest goes over as it stands, `$` settings commands included,
    because those are blocks like any other.
    """
    out = []
    for line in str(text).splitlines():
        line = line.split(";", 1)[0].strip()
        if line:
            out.append(line)
    return out


# --------------------------------------------------------------------------
#  what comes back
# --------------------------------------------------------------------------

def classify(line: str) -> tuple[str, str]:
    """What a line from the board means: ``("ok", "")``, ``("error", "3")``, ...

    The kinds are "ok", "error", "echo", "message", "alarm", "status" and "text"; the
    second half is the number of the error, the text of the message or the report itself.
    A board that is told to do something it may not do while it is in the alarm state
    answers with "Alarm: unlock required", which is not an error number.
    """
    text = str(line).strip()
    if not text:
        return ("empty", "")
    if text == "ok":
        return ("ok", "")
    if text.startswith("error:"):
        return ("error", text[len("error:"):].strip())
    if text.lower().startswith("echo:"):
        return ("echo", text[len("echo:"):].strip())
    if text.startswith("[MSG:") and text.endswith("]"):
        return ("message", text[len("[MSG:"):].rstrip("]").strip())
    if text.lower().startswith("alarm"):
        return ("alarm", text)
    if text.startswith("<") and text.endswith(">"):
        return ("status", text)
    return ("text", text)


def error_text(number: str) -> str:
    """What an error number of the board means, in words."""
    num = str(number).strip()
    return ERRORS.get(num, tr("Unknown error {n}")).format(n=num)


def _triplet(value: str):
    """`12.000,3.000,0.000` as three numbers; None when the board sends something else."""
    parts = str(value).split(",")
    if len(parts) < 2:
        return None
    try:
        nums = [float(p) for p in parts[:3]]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.append(0.0)
    return tuple(nums)


def _numbers(value: str, at_most=5):
    """`12.000,3.000,0.000,90.000` as the numbers it holds, at most `at_most` of them.

    A board with rotary axes sends four or five of them and a board of GRBL 1.1 three;
    what a board sends behind the fifth axis is not ours to keep. None when the board
    sends something else, and a board that names fewer than three leaves the rest at zero.
    """
    parts = str(value).split(",")
    if len(parts) < 2:
        return None
    try:
        nums = [float(p) for p in parts[:at_most]]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.append(0.0)
    return tuple(nums)


#: the fields of a status report that carry a value of their own; the state is the one
#: field without a name, so it is the first field that is not one of these
REPORT_FIELDS = ("mpos", "wpos", "wco", "fs", "ov", "coor", "pnx", "bf", "wd", "al", "sys")


def parse_report(line: str) -> dict:
    """The numbers of a status report `<Idle|MPos:12.000,3.000,0.000|FS:0,0|Ov:100,100,100>`.

    The dict holds ``state`` (the word without its ":line" and without the "Check:"
    prefix of check mode), ``mpos``, ``wpos`` and ``wco`` (three numbers on a plain board
    and four or five on one with rotary axes, or None), ``feed`` and ``speed``, ``line``
    (how far the job is), ``coord`` (the work system, G54 to G59), and the three overrides
    as ``feed_override``, ``rapid_override`` and ``power_override``. Fields the board does
    not send are simply not in the dict, so a board of an older version still gives a
    usable position.

    The state is the only field the board does not name, which is why the first field
    that is not one of :data:`REPORT_FIELDS` is read as the state; a number behind it is
    the block the board is at, except behind "Door", where it says how the door is.

    An empty dict when the line is not a report: a board that is asked too often says
    nothing at all, and a line that is only half a report is not worth half a position.
    """
    text = str(line).strip()
    if not (text.startswith("<") and text.endswith(">")):
        return {}
    out: dict = {"state": ""}
    for part in text[1:-1].split("|"):
        key, _sep, value = part.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key not in REPORT_FIELDS:
            words = [w.strip() for w in part.split(":")]
            if words[0] == "Check" and len(words) > 1:
                words = words[1:]     # <Check:Idle|…>: the board plans, but is idle
            out["state"] = words[0]
            if len(words) > 1 and words[0] != "Door":
                # the number behind Door is the state of the door, not how far the job is
                try:
                    out["line"] = int(words[1])
                except ValueError:
                    pass
        elif key in ("mpos", "wpos", "wco"):
            out[key] = _numbers(value)
        elif key == "fs":
            nums = _triplet(value)
            if nums is not None:
                out["feed"], out["speed"] = nums[0], nums[1]
        elif key == "ov":
            nums = _triplet(value)
            if nums is not None:
                out["feed_override"], out["rapid_override"], out["power_override"] = nums
        elif key == "coor":
            out["coord"] = value
        elif key == "pnx":
            out["pinx"] = value
        else:
            out.setdefault("other", {})[key] = value
    return out


def axes_of(report_or_position) -> int:
    """How many axes the board drives, out of a status report or a position of one.

    A board of GRBL 1.1 names three numbers and grblHAL or FluidNC four or five when it
    drives a rotary A or B axis; a board that has not said anything yet is a plain three.
    """
    where = report_or_position
    if isinstance(where, dict):
        where = next((where[key] for key in ("mpos", "wpos", "wco") if where.get(key)),
                     None)
    if not where:
        return 3
    return max(3, min(len(where), len(AXES)))


def state_label(state: str) -> str:
    """The state of the board in the words of the interface; an unknown state as it says."""
    return STATE_LABELS.get(str(state), str(state))


# --------------------------------------------------------------------------
#  the blocks a controller window sends
# --------------------------------------------------------------------------

def jog(dx=0.0, dy=0.0, dz=0.0, feed=1000.0, da=0.0, db=0.0) -> str:
    """A relative move in jog mode: GRBL 1.1 only allows jogging through `$J=`.

    The axes that do not move are left out, and a move of nothing at all is no block, so a
    click on the pad never sends an empty `$J=` the board refuses. A and B are the rotary
    axes of a mill with a 4th or 5th axis and stand in degrees.
    """
    words = [f"{axis}{_num(v)}" for axis, v in (("X", dx), ("Y", dy), ("Z", dz),
                                                ("A", da), ("B", db))
             if abs(float(v)) > 1e-9]
    if not words:
        return ""
    return f"$J=G21 G91 {' '.join(words)} F{float(feed):g}"


def move(x=None, y=None, z=None, feed=None) -> str:
    """An absolute rapid move to a point of the work, in millimetres.

    Only the axes that are given are named; `feed` turns the rapid into a fed move, which
    is what a move with the beam on has to be.
    """
    words = [f"{axis}{_num(v)}" for axis, v in (("X", x), ("Y", y), ("Z", z)) if v is not None]
    if not words:
        return ""
    return f"G90 G0 {' '.join(words)}" + (f" F{float(feed):g}" if feed else "")


def zero_here(x=True, y=True, z=False, a=False, b=False) -> str:
    """The block that says "the head is at zero here", for the axes that are asked for.

    `G10 L20` moves the work offset **relative** to the position the board reports now, so
    naming only X leaves the Y and Z offsets exactly as they were; `L2` would need the
    machine position of every axis and is easy to get wrong. The rotary axes A and B are
    zeroed the same way, in degrees.
    """
    words = [f"{axis}0" for axis, on in (("X", x), ("Y", y), ("Z", z), ("A", a), ("B", b))
             if on]
    if not words:
        return ""
    return f"G10 L20 P1 {' '.join(words)}"


def zero_at(x=0.0, y=0.0, z=None) -> str:
    """The block that puts the work origin at a point of the machine, in machine
    millimetres (`G10 L2` names the offset itself, not the position it brings)."""
    words = [f"X{_num(x)}", f"Y{_num(y)}"]
    if z is not None:
        words.append(f"Z{_num(z)}")
    return f"G10 L2 P1 {' '.join(words)}"


def beam(power, s_max=1000, dynamic=True) -> str:
    """The block that puts the beam on at `power` percent of what the board allows.

    The S word is a whole number up to `$30` of the board, so the percentage is scaled to
    `s_max`. M4 or M3 is the same choice the G-code export makes: M4 follows the feed, so
    a slow start gives less power.
    """
    power = max(0.0, min(100.0, float(power)))
    s = int(round(power * float(s_max) / 100.0))
    return f"{'M4' if dynamic else 'M3'} S{s}"


#: the block that switches the beam off
BEAM_OFF = "M5"

#: the blocks of the machine itself
UNLOCK = "$X"
HOME = "$H"
SETTINGS = "$$"
HELP = "$"
#: check mode: the board plans and refuses but moves nothing; `$C` turns it off again
CHECK = "$C"


def override(kind: str, action: str) -> bytes:
    """The real-time byte that changes the feed, the rapid or the power; b"" if unknown."""
    return OVERRIDES.get(str(kind), {}).get(str(action), b"")


# --------------------------------------------------------------------------
#  sending a job without losing anything
# --------------------------------------------------------------------------

class Sender:
    """The blocks of a job, handed out as fast as the board can take them.

    The board reads one block at a time and answers it with `ok`, and its input buffer
    holds `buffer` bytes. Writing more than that loses characters, so the sender keeps
    track of how many bytes it has written without an answer and waits when the buffer is
    full: **character counting**, the way every GRBL sender does it. A block longer than
    the whole buffer still goes on its own, since nothing may follow it anyway.

    :meth:`pause` stops handing out new blocks without forgetting the ones that are left;
    the board runs empty what it already has, which is what the feed hold button does.
    """

    def __init__(self, lines, buffer=BUFFER_BYTES):
        self.lines = [str(line) for line in lines]
        self.total = len(self.lines)
        self.buffer = max(1, int(buffer))
        self.index = 0              #: the next block to hand out
        self.acked = 0              #: the blocks the board answered, with ok or with error
        self.errors: list[tuple[int, str]] = []   #: (number of the block, error number)
        self.paused = False
        self._sizes = deque()       #: the bytes of every block written without an answer

    @property
    def in_flight(self) -> int:
        """How many bytes the board has not answered for yet."""
        return sum(self._sizes)

    def take(self, extra=0) -> bytes:
        """As much of the job as the board still has room for, as the bytes to write.

        `extra` is what else the board holds without an answer - blocks that are not part
        of the job, such as a `$$` sent in between - so the whole stays under the buffer.
        Empty when the job is over, when it is paused or when the buffer is full; the
        caller comes back after the next `ok`.
        """
        if self.paused or self.index >= self.total:
            return b""
        extra = max(0, int(extra))
        out, sizes = [], []
        while self.index < self.total:
            size = len(self.lines[self.index]) + 1        # the newline is part of it
            room = extra + self.in_flight + sum(sizes) + size
            if (out or self._sizes or extra) and room > self.buffer:
                break
            out.append(self.lines[self.index])
            sizes.append(size)
            self.index += 1
        if not out:
            return b""
        self._sizes.extend(sizes)
        return ("\n".join(out) + "\n").encode("ascii", "replace")

    def confirm(self, ok=True) -> None:
        """The board answered the oldest block that was still waiting.

        `ok` is False for `error:N`: the bytes come back in the same way, but the block is
        remembered in :attr:`errors` so the job can say what it could not cut.
        """
        if not self._sizes:
            self.acked = min(self.total, self.acked + 1)
            return
        self._sizes.popleft()
        self.acked += 1
        if not ok:
            self.errors.append((self.acked, ""))

    def error(self, number: str) -> None:
        """The error number that belongs to the block that was refused just now."""
        if self.errors:
            self.errors[-1] = (self.errors[-1][0], str(number))

    @property
    def done(self) -> bool:
        """Everything handed out and everything answered."""
        return self.index >= self.total and not self._sizes

    @property
    def fraction(self) -> float:
        """How far the job is, 0.0 to 1.0; a job with nothing in it is finished."""
        return 1.0 if not self.total else min(1.0, self.acked / float(self.total))

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def cancel(self) -> None:
        """Nothing more goes out, and the job is over."""
        self.index = self.total
        self._sizes.clear()
        self.paused = False
