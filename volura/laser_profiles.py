"""Machines that are known: the presets of GRBL lasers and GRBL mills.

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

The sizes come from the makers' own pages and manuals (checked November 2026). Where a
maker does not publish the top speed of its machine the preset leaves `$110` and `$111`
alone (`rapid` zero) instead of guessing, and a mill never gets an S maximum from a
preset: the spindle of a mill is what the board of that one mill was set up for.
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
CNC_PRESETS_KEY = "cnc/presets"
USER_PREFIX = "user:"

#: the two kinds of machine on a cable
LASER = "laser"
CNC = "cnc"


def presets_key(kind=LASER) -> str:
    """Where the saved presets of a laser or of a mill are kept: each its own list."""
    return CNC_PRESETS_KEY if kind == CNC else PRESETS_KEY

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
    keeps the pad free and leaves the travel settings alone; `z` is the travel of a mill
    in height, for information. `jog` is the feed of the pad in mm per minute, `rapid`
    what `$110` and `$111` are set to; zero leaves them as the board has them.
    `open_source` marks the machines whose design or controller board is published; the
    others are commercial machines that speak plain GRBL. `kind` is laser or cnc, and
    `axes` how many axes the board drives (4 with a rotary A axis).
    """

    key: str
    label: str
    x: float = 0.0
    y: float = 0.0
    s_max: int = 1000
    dynamic: bool = True
    jog: int = 3000
    rapid: int = 0
    baud: int = 115200
    open_source: bool = False
    note: str = ""
    extra: dict = None            #: the settings of the board itself, for a saved preset
    kind: str = LASER
    z: float = 0.0
    axes: int = 3

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
        the user may have measured something else than the brochure says. A mill may
        give a third number, its height: that goes to `$132`, which a laser leaves alone.

        Laser mode is what makes a GRBL laser a laser: without `$32=1` the beam stays on
        during a rapid move and burns a line across the whole sheet on its way to the
        first path. A mill wants it off (`$32=0`): in laser mode GRBL does not wait for
        the spindle and lets it run down at every G0. The travel (`$130`, `$131`) goes along without the soft limits
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
            if self.kind == CNC:
                values = {32: 0}
            else:
                values = {32: 1, 30: int(self.s_max)}
            if int(self.rapid) > 0:
                values[110] = values[111] = int(self.rapid)
            if x > 0.0 and y > 0.0:
                values[130], values[131] = x, y
            values = {n: v for n, v in values.items() if n not in NEVER}
            z = float(self.z) if travel is None or len(travel) < 3 else float(travel[2])
            if self.kind == CNC and z > 0.0:
                values[132] = z         # the height of a mill, next to its X and Y
        return [f"${n}={v:g}" for n, v in sorted(values.items())]


#: what a key that names nothing falls back to: a board of which nothing is known
CUSTOM = Machine(key="custom", label=tr("Unknown GRBL board"),
                 note=tr("What the controller says"))
CUSTOM_CNC = Machine(key="custom", label=tr("Unknown GRBL board"), kind="cnc",
                     dynamic=False, note=tr("What the controller says"))

L = Machine     # a laser


def M(key, label, x, y, z, rapid=0, open_source=False, note="", axes=3, jog=2000):
    """A mill: no laser mode, no dynamic power, slower on the pad."""
    return Machine(key, label, x, y, rapid=rapid, jog=jog, open_source=open_source,
                   note=note, kind=CNC, dynamic=False, z=z, axes=axes)


#: the lasers: the generic boards and the open designs first, then the commercial ones
MACHINES = (
    L("grbl8", "GRBL 1.1 (Arduino Nano / Uno)", 300, 180, rapid=500, jog=500,
      open_source=True, note=tr("The 8-bit board of the 3018 kits; GRBL 1.1 comes with "
                                "$30=1000 and a slow $110")),
    L("grbl32", tr("GRBL 32-bit (ESP32 / STM32, MKS DLC32)"), open_source=True,
      note=tr("A 32-bit board: take the work area from the board")),
    L("acro55", "OpenBuilds ACRO 55", 300, 300, open_source=True, note=tr("The open laser frame of OpenBuilds, on a BlackBox board")),
    L("acro510", "OpenBuilds ACRO 510", 300, 800, open_source=True, note=tr("The open laser frame of OpenBuilds, on a BlackBox board")),
    L("acro1010", "OpenBuilds ACRO 1010", 800, 800, open_source=True, note=tr("The open laser frame of OpenBuilds, on a BlackBox board")),
    L("acro1510", "OpenBuilds ACRO 1510", 1300, 800, open_source=True, note=tr("The open laser frame of OpenBuilds, on a BlackBox board")),
    L("acro1515", "OpenBuilds ACRO 1515", 1300, 1300, open_source=True, note=tr("The open laser frame of OpenBuilds, on a BlackBox board")),
    L("k40", "K40 + Cohesion3D (GRBL-LPC)", 300, 200, jog=2000, open_source=True,
      note=tr("The 40 W CO2 with a GRBL board in it; the original M2 board does not speak "
              "GRBL. Measure the bed: often 320 × 230")),
    L("lm2pro", "Ortur Laser Master 2 Pro S2", 400, 400, rapid=15000),
    L("lm3", "Ortur Laser Master 3", 400, 380, note=tr("The 10 W version is 400 × 400")),
    L("s9", "Sculpfun S9", 410, 420, note=tr("The S9 Pro is 400 × 410")),
    L("s30", "Sculpfun S30 / S30 Pro", 380, 385),
    L("s30pm", "Sculpfun S30 Pro Max", 410, 400, note=tr("Usable about 370 × 363")),
    L("sfa9", "Sculpfun SF-A9", 400, 400, rapid=36000),
    L("a5pro", "Atomstack A5 Pro", 410, 400),
    L("x7pro", "Atomstack X7 Pro / A10 Pro", 410, 400),
    L("a70max", "Atomstack A70 Max", 850, 800),
    L("tts20pro", "TwoTrees TTS-20 Pro", 418, 418,
      note=tr("The honeycomb bed that comes with it is 350 × 350")),
    L("ts2", "TwoTrees TS2", 450, 450),
    L("falcon10", "Creality CR-Laser Falcon 10W", 400, 415),
    L("falcon2", "Creality Falcon2 22W", 400, 415, rapid=25000),
    L("falcon2pro", "Creality Falcon2 Pro", 400, 415),
    L("ray5", "Longer Ray5 5W / 10W", 400, 400, rapid=10000,
      note=tr("The 20 W version is 400 × 365")),
    L("comgrow_z1", "Comgrow Z1", 400, 400),
)

#: the mills: gantry routers and desktop mills that speak GRBL, grblHAL or FluidNC
MILLS = (
    M("grbl_mill", tr("GRBL mill (any board)"), 0, 0, 0, open_source=True,
      note=tr("Take the work area from the board")),
    M("3018pro", "Genmitsu 3018-PRO", 300, 180, 45, rapid=1000, jog=1000, open_source=True,
      note=tr("GRBL 1.1 on an 8-bit board")),
    M("3018prover2", "Genmitsu 3018-PROVer V2", 290, 180, 40, jog=1000),
    M("4040pro", "Genmitsu 4040-PRO", 400, 400, 78),
    M("lead1010", "OpenBuilds LEAD 1010", 730, 810, 100, open_source=True, note=tr("Open design on a BlackBox board")),
    M("lead1515", "OpenBuilds LEAD 1515", 1170, 1250, 90, open_source=True, note=tr("Open design on a BlackBox board")),
    M("minimill", "OpenBuilds MiniMill", 120, 180, 80, open_source=True, note=tr("Open design on a BlackBox board")),
    M("cbeam", "OpenBuilds C-Beam Machine", 350, 280, 85, open_source=True, note=tr("Open design on a BlackBox board")),
    M("workbee1010", "WorkBee 1010", 824, 780, 122, open_source=True,
      note=tr("The 1510 is 824 × 1280")),
    M("longmill30", "Sienci LongMill MK2 30×30", 813, 864, 114, rapid=4000, open_source=True),
    M("longmill48", "Sienci LongMill MK2 48×30", 1270, 864, 114, rapid=4000,
      open_source=True),
    M("longmill_vortex", "Sienci LongMill + Vortex (A axis)", 813, 864, 114, rapid=4000,
      open_source=True, axes=4,
      note=tr("A fourth, rotating axis: needs grblHAL on the SLB board")),
    M("altmill", "Sienci AltMill 4×4", 1245, 1245, 140, rapid=15000, open_source=True,
      axes=4, note=tr("grblHAL; a rotary A axis can be added")),
    M("shapeoko3", "Shapeoko 3", 425, 425, 75, rapid=5000,
      note=tr("The XL is 830 × 430, the XXL 830 × 850")),
    M("shapeoko4xxl", "Shapeoko 4 XXL", 838, 838, 101, rapid=5000),
    M("shapeoko5pro", "Shapeoko 5 Pro 4×4", 1237, 1237, 101),
    M("xcarve1000", "Inventables X-Carve 1000", 750, 750, 65),
    M("xcarvepro44", "Inventables X-Carve Pro 4×4", 1219, 1219, 101,
      note=tr("The 4×2 is 1219 × 610")),
    M("fox4040xe", "FoxAlien 4040-XE", 400, 400, 65),
    M("masuterpro", "FoxAlien Masuter Pro", 400, 400, 60, note=tr("Usable about 377 × 377")),
    M("ttc450", "TwoTrees TTC450", 460, 460, 80),
    M("lowrider4", "V1 Engineering LowRider v4", 0, 0, 100, rapid=5000, open_source=True,
      note=tr("FluidNC on a Jackpot board; you choose the size yourself")),
    M("mpcnc", "V1 Engineering MPCNC Primo", 0, 0, 0, open_source=True, note=tr("FluidNC on a Jackpot board; you choose the size yourself")),
)


def machines(kind=LASER) -> tuple:
    """The list of a kind: the lasers or the mills."""
    return MILLS if kind == CNC else MACHINES


#: the settings keys of the controller window
MACHINE_KEY = "laser/machine"
TRAVEL_X_KEY = "laser/travel_x"
TRAVEL_Y_KEY = "laser/travel_y"
TRAVEL_Z_KEY = "laser/travel_z"
CLAMP_KEY = "laser/clamp_jog"


def names(kind=LASER) -> list[str]:
    """The presets the user can choose from, as ``(key, label)`` pairs."""
    return [(m.key, m.label) for m in machines(kind)]


def choices(kind=LASER) -> list[str]:
    """The machines of the list and the presets of the user behind them: for the combo box.

    There is no "own machine" in it: a board that is not in the list is read with "Take
    from the board" and kept as a preset of its own."""
    return (names(kind) +
            [(user_key(str(p.get("name", ""))), str(p.get("name", ""))) for p in presets(kind)])


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


def presets(kind=LASER) -> list:
    """The presets the user saved, as they stand in the settings."""
    text = str(qsettings().value(presets_key(kind), "") or "")
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


def store_presets(items, kind=LASER) -> list:
    """Keep the presets of the user; what was stored is what comes back."""
    q = qsettings()
    q.setValue(presets_key(kind), presets_to_json(items))
    q.sync()
    return items


def find_preset(name, kind=LASER):
    """The saved preset of that name; None when there is no such preset."""
    for preset in presets(kind):
        if preset["name"] == str(name):
            return preset
    return None


def add_preset(preset, kind=LASER) -> list:
    """Save a preset under its name, overwriting the one that had it."""
    items = [item for item in presets(kind) if item["name"] != preset["name"]]
    items.append(preset)
    return store_presets(items, kind)


def remove_preset(name, kind=LASER) -> list:
    """Forget the preset of that name."""
    return store_presets([item for item in presets(kind) if item["name"] != str(name)], kind)


def preset_machine(preset, include=None, kind=LASER) -> Machine:
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
                   extra={n: settings[n] for n in send if n in settings},
                   kind=CNC if kind == CNC else LASER)


def preset_lines(preset, include=None, travel=None, kind=LASER) -> list[str]:
    """The blocks a saved preset sends back to a board, in the order of `$`."""
    return preset_machine(preset, include, kind).settings(travel)


def match(x, y, s_max=None, kind=LASER) -> Machine | None:
    """The preset that describes a work area and an S maximum; None when none does.

    A mill matches on its work area alone: its S maximum is its spindle, not the table."""
    x, y = float(x or 0.0), float(y or 0.0)
    for machine in machines(kind):
        if machine.known and abs(float(machine.x) - x) < 0.51 and \
                abs(float(machine.y) - y) < 0.51 and \
                (s_max is None or kind == CNC or int(machine.s_max) == int(s_max)):
            return machine
    return None


def unknown(kind=LASER) -> Machine:
    """What a key that names nothing falls back to."""
    return CUSTOM_CNC if kind == CNC else CUSTOM


def find(key, kind=LASER) -> Machine:
    """The preset a key names; an unknown board when the key names nothing."""
    key = str(key)
    if is_user_key(key):
        preset = find_preset(key[len(USER_PREFIX):], kind)
        return preset_machine(preset, kind=kind) if preset is not None else unknown(kind)
    for machine in machines(kind):
        if machine.key == key:
            return machine
    return unknown(kind)


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


def height_of(settings: dict) -> float:
    """How far Z goes, as the board reported it in `$132`; 0 when it did not say."""
    z = float(settings.get(132, 0.0) or 0.0)
    return z if z > 0.0 else 0.0


def travel_of(settings: dict) -> tuple:
    """The travel the board itself reported as `$130` and `$131`; (0, 0) when unknown."""
    x = float(settings.get(130, 0.0) or 0.0)
    y = float(settings.get(131, 0.0) or 0.0)
    return (x, y) if x > 0.0 and y > 0.0 else (0.0, 0.0)


def laser_mode(settings: dict):
    """Whether the board says laser mode is on; None when the board did not answer."""
    value = settings.get(LASER_MODE_SETTING)
    return None if value is None else value >= 1.0
