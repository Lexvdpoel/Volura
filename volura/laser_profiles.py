"""Machines that are known: the presets of the open-source GRBL lasers.

No Qt in this file. A preset says what the board of a certain laser is set to and how big
its work area is, so the controller window does not have to guess: the highest S word
(`$30`) decides how the percentage of a layer becomes an S word in the G-code, laser mode
(`$32`) decides whether the beam goes dark during a rapid move, and the travel of the
machine decides how far the pad may jog before the head pulls the belt out of its ends.

What a preset never touches is the number of steps per millimetre (`$100`, `$101`) and
anything that has to do with limits and homing (`$20`, `$22`, `$132`): the first is how
the machine was calibrated and a wrong number there moves the head somewhere the drawing
does not say it is, and the second are there to save the machine from itself. A preset
that switches one of those off is not a preset but a broken cable waiting to happen.

The board is the truth and a preset is a starting point, so the settings the board itself
reports (`$$`) can be read back: :func:`parse_settings` turns those lines into numbers and
the window offers to take the work area and the S maximum from the board instead of from
the table. That matters because these machines change: the same model has been shipped
with three boards, and the numbers below are what the model was built with, not a promise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .i18n import tr

#: `$30=1000 (max spindle speed)`: the number of a setting and its value
SETTING = re.compile(r"^\$(\d+)\s*=\s*(-?\d*\.?\d+)")

#: the settings a preset may never send: calibration, limits and homing
NEVER = (100, 101, 102, 112, 120, 121, 122, 132, 20, 21, 22, 23, 24, 25, 26, 27)

#: the settings of the S maximum and the laser mode, so the window can find them back
S_MAX_SETTING = 30
LASER_MODE_SETTING = 32


@dataclass(frozen=True)
class Machine:
    """One laser: how big it is and what its board is set to.

    `x` and `y` are the travel of the head in millimetres; zero means "unknown", which
    keeps the pad free and leaves the travel settings alone. `jog` is the feed of the pad
    in mm per minute, `rapid` the `F` of a travel move and also what `$110` and `$111`
    are set to. `open_source` marks the machines whose design or controller board is
    published; the others are commercial machines that speak plain GRBL.
    """

    key: str
    label: str
    x: float = 0.0
    y: float = 0.0
    s_max: int = 1000
    dynamic: bool = True
    jog: int = 4000
    rapid: int = 6000
    baud: int = 115200
    open_source: bool = False
    note: str = ""

    @property
    def known(self) -> bool:
        """Whether this preset knows how big the machine is."""
        return float(self.x) > 0.0 and float(self.y) > 0.0

    def inside(self, x, y, margin=0.0) -> bool:
        """Whether a point of the work is on the machine; anything is on an unknown one."""
        if not self.known:
            return True
        return -float(margin) <= float(x) <= float(self.x) + float(margin) and \
            -float(margin) <= float(y) <= float(self.y) + float(margin)

    def settings(self, travel=None) -> list[str]:
        """The blocks that set the board to what this machine is, in the order of `$`.

        `travel` overrides the work area of the preset with what is on the screen, because
        the user may have measured something else than the brochure says.

        Laser mode is what makes a GRBL laser a laser: without `$32=1` the beam stays on
        during a rapid move and burns a line across the whole sheet on its way to the
        first path. The travel (`$130`, `$131`) goes along without the soft limits
        (`$20`), because GRBL only allows those together with homing and a preset that
        refuses to move because it has never been home is not a help.
        """
        x, y = (float(self.x), float(self.y)) if travel is None else \
            (float(travel[0]), float(travel[1]))
        out = ["$32=1", f"$30={int(self.s_max)}",
               f"$110={int(self.rapid)}", f"$111={int(self.rapid)}"]
        if x > 0.0 and y > 0.0:
            out += [f"$130={x:g}", f"$131={y:g}"]
        out = [line for line in out if int(SETTING.match(line).group(1)) not in NEVER]
        return sorted(out, key=lambda line: int(SETTING.match(line).group(1)))


#: what to start with when nobody knows: a 32-bit board on a cable
CUSTOM = Machine(key="custom", label=tr("Own machine"), note=tr("What the controller says"))

#: the machines: the open designs first, then the commercial ones that speak GRBL
MACHINES = (
    Machine("grbl8", "GRBL 8-bit (Arduino Nano, 3018)", 300, 180, s_max=255, jog=2000,
            rapid=3000, open_source=True,
            note=tr("An old 8-bit board: the S word goes to 255")),
    Machine("grbl32", "GRBL 32-bit (BtGRBL, BlackBird)", 400, 400, open_source=True,
            note=tr("The board most open lasers are built with")),
    Machine("blackbird", "OpenBuilds BlackBird", 500, 500, rapid=6000, open_source=True,
            note=tr("The open-source laser controller of OpenBuilds")),
    Machine("leadstone", "OpenBuilds LEAD / Leadstone 10", 1016, 1016, rapid=8000,
            open_source=True, note=tr("A BlackBird on a LEAD frame")),
    Machine("workhorse", "Workhorse 3500", 350, 350, rapid=5000, open_source=True,
            note=tr("The open laser design of Jacques Bell")),
    Machine("longrun", "Longrun LZ-40417", 400, 400),
    Machine("tts20", "TwoTrees TTS-20 / TTS-20 Pro", 400, 400),
    Machine("s9", "Sculpfun S9 / S10", 400, 415),
    Machine("s30", "Sculpfun S30", 410, 400, rapid=8000),
    Machine("lm2", "Ortur Laser Master 2 / LM2 Pro", 400, 350),
    Machine("a5", "Atomstack A5 / A5 Pro", 400, 375),
    Machine("x7", "Atomstack X7 / A70", 400, 410, rapid=8000),
    Machine("falcon12", "Creality Falcon 1.2", 400, 400),
    Machine("falcon2", "Creality Falcon 2", 400, 400, rapid=8000),
    Machine("z2", "NEJE Z2 / Z2 Pro", 400, 400, dynamic=False,
            note=tr("Not every board of this one knows M4")),
)

#: the settings keys of the controller window
MACHINE_KEY = "laser/machine"
TRAVEL_X_KEY = "laser/travel_x"
TRAVEL_Y_KEY = "laser/travel_y"
CLAMP_KEY = "laser/clamp_jog"


def names() -> list[str]:
    """The presets the user can choose from, as ``(key, label)`` pairs."""
    return [(m.key, m.label) for m in MACHINES]


def choices() -> list[str]:
    """The presets with the own machine in front: what the combo box of the window holds."""
    return [(CUSTOM.key, CUSTOM.label)] + names()


def match(x, y, s_max=None) -> Machine | None:
    """The preset that describes a work area and an S maximum; None when none does."""
    x, y = float(x or 0.0), float(y or 0.0)
    for machine in MACHINES:
        if abs(float(machine.x) - x) < 0.51 and abs(float(machine.y) - y) < 0.51 and \
                (s_max is None or int(machine.s_max) == int(s_max)):
            return machine
    return None


def find(key) -> Machine:
    """The preset a key names; the own machine when the key is unknown or empty."""
    for machine in MACHINES:
        if machine.key == str(key):
            return machine
    return CUSTOM


def parse_settings(lines) -> dict:
    """What the board answered to `$$`, as {number: value}.

    The board answers `$30=1000 (max spindle speed)` and friends, one per line, mixed
    with its greeting and the `$G`-codes; only the numbered settings are of interest
    here, and a value that is not a number is left out rather than guessed.
    """
    out: dict = {}
    for line in lines:
        m = SETTING.match(str(line).strip())
        if m is None:
            continue
        try:
            out[int(m.group(1))] = float(m.group(2))
        except ValueError:
            continue
    return out


def s_max_of(settings: dict, default=1000) -> int:
    """The highest S word the board itself reported; `default` when it did not say."""
    value = settings.get(S_MAX_SETTING)
    return int(value) if value and value > 0 else int(default)


def travel_of(settings: dict) -> tuple:
    """The travel the board itself reported as `$130` and `$131`; (0, 0) when unknown."""
    x = float(settings.get(130, 0.0) or 0.0)
    y = float(settings.get(131, 0.0) or 0.0)
    return (x, y) if x > 0.0 and y > 0.0 else (0.0, 0.0)


def laser_mode(settings: dict):
    """Whether the board says laser mode is on; None when the board did not answer."""
    value = settings.get(LASER_MODE_SETTING)
    return None if value is None else value >= 1.0
