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

import json
import re
from dataclasses import dataclass

from .i18n import tr
from .settings import qsettings

#: `$30=1000 (max spindle speed)`: the number of a setting and its value
SETTING = re.compile(r"^\$(\d+)\s*=\s*(-?\d*\.?\d+)")

#: the settings a preset may never send: calibration, limits and homing
NEVER = (100, 101, 102, 112, 120, 121, 122, 132, 20, 21, 22, 23, 24, 25, 26, 27)

#: the settings of the S maximum and the laser mode, so the window can find them back
S_MAX_SETTING = 30
LASER_MODE_SETTING = 32

#: the key of the presets the user saved, and the front of the key of such a preset
PRESETS_KEY = "laser/presets"
USER_PREFIX = "user:"

#: what the numbers of `$$` mean, in the groups GRBL itself uses
GROUPS = (
    ((0, 1, 2, 3, 4, 5, 6), tr("Pulse and inversion")),
    ((10, 11, 12, 13), tr("Reports and deviation")),
    ((20, 21, 22, 23, 24, 25, 26, 27), tr("Limits and homing")),
    ((30, 31, 32), tr("Spindle and laser")),
    ((100, 101, 102), tr("Steps per millimetre")),
    ((110, 111, 112), tr("Maximum feed")),
    ((120, 121, 122), tr("Acceleration")),
    ((130, 131, 132), tr("Maximum travel")),
)

#: the last three numbers of a group are X, Y and Z
AXES = "XYZ"


def setting_group(n) -> str:
    """What the board calls the setting with this number, in the words of the interface."""
    for numbers, name in GROUPS:
        if int(n) in numbers:
            return name
    return tr("Setting {n}").format(n=int(n))


def setting_name(n) -> str:
    """The name of a setting with its axis: `110` is the maximum feed of X."""
    n = int(n)
    group = setting_group(n)
    if 100 <= n <= 132 and n % 10 < 3:
        return f"{group} ({AXES[n % 10]})"
    return group


def is_risky(n) -> bool:
    """Whether sending this setting to another machine can do damage."""
    return int(n) in NEVER


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
    extra: dict = None            #: the settings of the board itself, for a saved preset

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

        A preset the user saved carries the settings of its board in `extra` and sends
        those; a preset of the list sends what is written above.
        """
        x, y = (float(self.x), float(self.y)) if travel is None else \
            (float(travel[0]), float(travel[1]))
        if self.extra:
            values = {int(n): float(v) for n, v in self.extra.items()}
        else:
            values = {32: 1, 30: int(self.s_max),
                      110: int(self.rapid), 111: int(self.rapid)}
            if x > 0.0 and y > 0.0:
                values[130], values[131] = x, y
            values = {n: v for n, v in values.items() if n not in NEVER}
        return [f"${n}={v:g}" for n, v in sorted(values.items())]


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
    Machine("k40", "K40 (K40-Nano, GRBL)", 300, 200, s_max=255, jog=2000, rapid=3000,
            note=tr("The Chinese 40 W CO2 on a Nano board; the original M2 board of a K40 "
                    "does not speak GRBL")),
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
    """The presets with the own machine in front and the saved ones behind: for the combo box."""
    return ([(CUSTOM.key, CUSTOM.label)] + names() +
            [(user_key(str(p.get("name", ""))), str(p.get("name", ""))) for p in presets()])


def user_key(name) -> str:
    """The key of a saved preset of the user."""
    return USER_PREFIX + str(name)


def is_user_key(key) -> bool:
    return str(key).startswith(USER_PREFIX)


# --------------------------------------------------------------------------
#  presets of the user: everything a board answered, saved under a name
# --------------------------------------------------------------------------

def new_preset(name, settings=None, x=0.0, y=0.0, s_max=1000, dynamic=True,
               jog=4000, rapid=6000, baud=115200, send=None) -> dict:
    """An empty preset of the user, with the numbers of a board in it.

    `settings` is everything the board answered to `$$`; `send` is what may go back to a
    board, and defaults to the settings that are safe to send: the numbers of the
    calibration and of the limits are kept, because a preset that goes to another machine
    should not move that machine's steps per millimetre along with it.
    """
    settings = {int(n): float(v) for n, v in (settings or {}).items()}
    if S_MAX_SETTING in settings and settings[S_MAX_SETTING] > 0:
        s_max = int(settings[S_MAX_SETTING])       # the board knows its own S maximum
    if send is None:
        send = [n for n in sorted(settings) if n not in NEVER]
    return {"name": str(name), "x": float(x), "y": float(y), "s_max": int(s_max),
            "dynamic": bool(dynamic), "jog": int(jog), "rapid": int(rapid),
            "baud": int(baud),
            "settings": {str(n): float(v) for n, v in sorted(settings.items())},
            "send": [int(n) for n in send]}


def presets() -> list:
    """The presets the user saved, as they stand in the settings."""
    text = str(qsettings().value(PRESETS_KEY, "") or "")
    return presets_from_json(text)


def presets_from_json(text) -> list:
    """The presets in a JSON string; nothing at all when it is not presets."""
    if not str(text).strip():
        return []
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        data = [data]
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        out.append(new_preset(item.get("name"), item.get("settings"), item.get("x", 0.0),
                              item.get("y", 0.0), item.get("s_max", 1000),
                              item.get("dynamic", True), item.get("jog", 4000),
                              item.get("rapid", 6000), item.get("baud", 115200),
                              item.get("send")))
    return out


def presets_to_json(items) -> str:
    """The presets as JSON, for the settings and for a file to give to someone else."""
    return json.dumps([{**item} for item in items], indent=1, sort_keys=True)


def store_presets(items) -> list:
    """Keep the presets of the user; what was stored is what comes back."""
    q = qsettings()
    q.setValue(PRESETS_KEY, presets_to_json(items))
    q.sync()
    return items


def find_preset(name):
    """The saved preset of that name; None when there is no such preset."""
    for preset in presets():
        if preset["name"] == str(name):
            return preset
    return None


def add_preset(preset) -> list:
    """Save a preset under its name, overwriting the one that had it."""
    items = [item for item in presets() if item["name"] != preset["name"]]
    items.append(preset)
    return store_presets(items)


def remove_preset(name) -> list:
    """Forget the preset of that name."""
    return store_presets([item for item in presets() if item["name"] != str(name)])


def preset_machine(preset, include=None) -> Machine:
    """A saved preset as a machine of the list, with the settings it may send in it."""
    settings = {int(n): float(v) for n, v in (preset.get("settings") or {}).items()}
    send = preset.get("send")
    if send is None:
        send = [n for n in sorted(settings) if n not in NEVER]
    if include is not None:
        send = [int(n) for n in include]
    return Machine(key=user_key(preset["name"]), label=str(preset["name"]),
                   x=float(preset.get("x", 0.0)), y=float(preset.get("y", 0.0)),
                   s_max=int(preset.get("s_max", 1000)),
                   dynamic=bool(preset.get("dynamic", True)),
                   jog=int(preset.get("jog", 4000)), rapid=int(preset.get("rapid", 6000)),
                   baud=int(preset.get("baud", 115200)),
                   note=str(preset.get("note", "")),
                   extra={n: settings[n] for n in send if n in settings})


def preset_lines(preset, include=None, travel=None) -> list[str]:
    """The blocks a saved preset sends back to a board, in the order of `$`."""
    return preset_machine(preset, include).settings(travel)


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
    key = str(key)
    if is_user_key(key):
        preset = find_preset(key[len(USER_PREFIX):])
        return preset_machine(preset) if preset is not None else CUSTOM
    for machine in MACHINES:
        if machine.key == key:
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
