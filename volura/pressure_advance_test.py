"""Pressure advance tests: the computational core of a hollow tower whose segments print with a
rising pressure advance value.

No Qt in this file except the values, which come from `fonts` the way the engraved spans of a bridge
test do. It gives the tower as one closed volume, so it can be tested on its own and reused in the
model and the export, and it gives the plan of values the printer has to follow, together with the
command every firmware wants to set one, so a slice can be told to change pressure halfway.

A **pressure advance test** is the test print that says how hard the printer has to push the filament
before the pressure in the nozzle is right for the corners: a hollow square box of `FOOT` by `FOOT`
millimetres with sharp corners, `segments` segments of `SEG_H` millimetres high, every one of them
printed with a higher value than the one under it. The corners of a segment show the result, because
that is where the nozzle turns and the pressure has to catch up: too little pressure advance leaves
gaps at the corners and in the short walls next to them, too much makes the corners bulge and ooze,
and the lowest segment that came out sharp is the one to read.

The wall is `WALL_LINES` lines of the nozzle wide and the box is hollow above a floor of `FLOOR`
millimetres, with no top: nothing hides those corners behind a wall of plastic, and the print is
quick and uses little filament. The value of every segment stands engraved `ENGRAVE` millimetres deep
in the middle of the front wall, in letters `CAP` millimetres high, so the number is read while the
corners are and the outside of the tower stays `FOOT` millimetres wide.

The tower is useless unless the printer really sets a new value at every segment, so the part carries
its plan: a list of ``[z_from, value]``, the lowest first, which `commands` turns into the command
lines of a firmware - `SET_PRESSURE_ADVANCE ADVANCE=…` for Klipper, `M900 K…` for Marlin and
`M572 D0 S…` for RepRapFirmware. Every segment asks for its value, the lowest one too, because a
printer often sits on another value than the start of the test. The commands go in at the first layer
at or above their height, which is what the generalised `slicer.with_commands` does, the same
mechanism as a colour change by height; because a change is asked for by height, a tower that stands
on the bed next to other parts changes the pressure advance for the whole bed.

The result is one closed volume on `z = 0`, centred in x and y, that prints standing up without
support; without fonts the walls carry no numbers.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr, tr_noop

#: mm a segment is tall
SEG_H = 5.0
#: mm a side of the tower is wide (the tower is a square box with sharp corners)
FOOT = 60.0
#: mm thick the floor under the hollow box is; the box has no top
FLOOR = 1.0
#: mm wide one line of the nozzle is, which is how thick a wall of the tower is
NOZZLE = 0.4
#: how many lines a wall is printed with: two of them is the thinnest wall that stands on its own
WALL_LINES = 2
#: mm thick a wall of the tower is
WALL = WALL_LINES * NOZZLE
#: mm deep the value of a segment is engraved in the front wall
ENGRAVE = 0.3
#: mm of wall that has to stay behind an engraving, so the wall stays a wall
ROOM = 0.4
#: mm high the letters of a value are, small enough to fit inside a segment of SEG_H
CAP = 3.0
#: mm of wall that has to stay at both ends of a value, so it does not run around a corner
EDGE = 6.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5

#: the fewest and the most segments a tower can have
SEGMENTS_MIN, SEGMENTS_MAX = 2, 20
#: the smallest and the biggest pressure advance value a firmware takes
VALUE_MIN, VALUE_MAX = 0.0, 5.0
#: the smallest and the biggest step between two segments
STEP_MIN, STEP_MAX = 0.001, 1.0

#: every firmware that can set the pressure advance, with the command it wants and what it is called
FIRMWARES = {"klipper": ("SET_PRESSURE_ADVANCE ADVANCE={v}", tr_noop("Klipper")),
             "marlin": ("M900 K{v}", tr_noop("Marlin")),
             "rrf": ("M572 D0 S{v}", tr_noop("RepRapFirmware"))}
#: the firmware to fall back on when nothing says what the printer runs
FIRMWARE_DEFAULT = "marlin"
#: the presets of the dialog, in its order: name -> (firmware, start, end, step)
PRESETS = {"klipper_direct": ("klipper", 0.0, 0.1, 0.005),
           "klipper_bowden": ("klipper", 0.0, 1.0, 0.05),
           "marlin_direct": ("marlin", 0.0, 0.1, 0.005),
           "marlin_bowden": ("marlin", 0.0, 1.0, 0.05),
           "rrf": ("rrf", 0.0, 0.2, 0.01)}
#: what the dialog calls every preset: the firmware and how far its extruder stands from the nozzle
PRESET_LABELS = {"klipper_direct": tr_noop("Klipper, direct drive"),
                 "klipper_bowden": tr_noop("Klipper, bowden"),
                 "marlin_direct": tr_noop("Marlin, direct drive"),
                 "marlin_bowden": tr_noop("Marlin, bowden"),
                 "rrf": tr_noop("RepRapFirmware")}
#: the printer under Window > Preferences that says the printer runs Klipper
KLIPPER_KIND = "moonraker"


def firmware_label(firmware) -> str:
    """What the dialog calls a firmware; one it does not know is its own name."""
    return tr(FIRMWARES[firmware][1]) if firmware in FIRMWARES else str(firmware)


def preset_label(name) -> str:
    """What the dialog calls a preset; one it does not know is its own name."""
    return tr(PRESET_LABELS[name]) if name in PRESET_LABELS else str(name)


def default_firmware(printers=()) -> str:
    """The firmware to offer first: Klipper for a Klipper printer, Marlin for anything else.

    `printers` is what `Settings.printers` holds (the connections of `volura.printer_link`), of which
    only the kind matters: a Moonraker printer answers to a Klipper machine. None, an empty list or a
    list of things that are no printers all say Marlin, which is what most printers run.
    """
    try:
        klipper = any(getattr(conn, "kind", None) == KLIPPER_KIND for conn in printers)
    except TypeError:
        klipper = False
    return "klipper" if klipper else FIRMWARE_DEFAULT


def default_preset(firmware) -> str:
    """The preset to offer for a firmware: its first one, which is its direct drive one."""
    for name, (fw, *_rest) in PRESETS.items():
        if fw == firmware:
            return name
    return next(iter(PRESETS))


def value_text(value) -> str:
    """A pressure advance value the way a command and an engraving write it: 0.035, not
    0.035000000000000003.

    The values come out of a walk by `step`, so they carry the noise of that walk; `:g}` gives the
    shortest form that means the same number. A value that is no number is written as it stands, and
    one so small that it prints nothing is written as 0.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(v) and abs(v) < 1e-9:
        v = 0.0                             # -0.0 and -1e-17 are the same value here
    return f"{v:g}"


def _number(value, lo, hi):
    """A number of the test, clamped to what a firmware takes; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the pressure advance values of a pressure advance test must be numbers"))
    return max(lo, min(hi, v))


def _values(start, end, step) -> list:
    """The pressure advance values of the segments, the lowest first, with the test's validation.

    `start` is the value of the lowest segment and every next segment stands `step` higher, walking
    from `start` towards `end` and stopping as soon as the next value would pass `end` - so the tower
    has `(end - start) / step` segments and the value of the top one is the last one that stays under
    `end`. A range that walks down works just the same, because a test can be read from the top too.
    A step of 0 (or less) would never reach the other value and a start equal to the end is one value
    and no test, so both are refused; so are a tower of fewer than `SEGMENTS_MIN` segments, which is
    nothing to read, and one of more than `SEGMENTS_MAX`, which does not print in reasonable time.
    """
    a = _number(start, VALUE_MIN, VALUE_MAX)
    b = _number(end, VALUE_MIN, VALUE_MAX)
    try:
        d = float(step)
    except (TypeError, ValueError):
        d = math.nan
    if not math.isfinite(d):
        raise ValueError(tr("the pressure advance values of a pressure advance test must be numbers"))
    if d <= 0.0:
        raise ValueError(tr("the step of a pressure advance test must be more than 0"))
    if abs(a - b) < 1e-9:
        raise ValueError(tr("the end value of a pressure advance test must differ from the start"))
    n = int(math.floor(abs(a - b) / d + 1e-9))
    if n < SEGMENTS_MIN:
        raise ValueError(tr("a pressure advance test needs at least {n} segments").format(n=SEGMENTS_MIN))
    if n > SEGMENTS_MAX:
        raise ValueError(tr("a pressure advance test has at most {n} segments").format(n=SEGMENTS_MAX))
    direction = 1.0 if b > a else -1.0
    # the values are rounded to what a firmware reads, so a walk of 0.005 steps carries no noise and
    # the last one of a walk down is 0 instead of the -0.0 that floating point leaves behind
    return [round(a + k * d * direction, 6) + 0.0 for k in range(n)]


def steps(start=0.0, end=0.1, step=0.005) -> list:
    """The plan of a pressure advance test, as ``[[z_from mm, value], ...]``, the lowest first.

    Segment `i` stands from `i * SEG_H` millimetres up and prints with value `i`, so its `z_from` is
    the height at which the printer has to set that value; the lowest segment starts on the bed, so
    its `z_from` is 0 and its command lands on the very first layer. That command is not too much: a
    printer often sits on another value than the start of the test. The numbers are validated as in
    `pressure_advance_test`.
    """
    return [[round(i * SEG_H, 6), v] for i, v in enumerate(_values(start, end, step))]


def sizes(start=0.0, end=0.1, step=0.005) -> dict:
    """The numbers a pressure advance test is made of, so a dialog can show them.

    `segments` is how many bands the tower has and `values` the pressure advance value of every one of
    them, the lowest first; `size` is the outside (width, depth, height) in millimetres, which for a
    tower is `FOOT` square and `segments * SEG_H` millimetres high, because it stands on its own floor
    and has no top. The numbers are validated as in `pressure_advance_test` and anything but a number
    is a ValueError.
    """
    vals = _values(start, end, step)
    return {"segments": len(vals), "values": vals, "segment_h": SEG_H,
            "size": (FOOT, FOOT, len(vals) * SEG_H)}


def command(firmware, value) -> str:
    """The command line with which a firmware sets the pressure advance to `value`.

    Klipper wants `SET_PRESSURE_ADVANCE ADVANCE=…`, Marlin `M900 K…` and RepRapFirmware
    `M572 D0 S…`; a firmware Volura does not know is a ValueError, because guessing there would write
    a line the printer does not understand.
    """
    if firmware not in FIRMWARES:
        raise ValueError(tr("Volura does not know the firmware {firmware}")
                         .format(firmware=firmware))
    return FIRMWARES[firmware][0].format(v=value_text(value))


def commands(firmware, plan) -> list:
    """The plan of `steps` as ``[[z_from mm, command line], ...]``, which is what the part keeps.

    Every segment gets its own command, the lowest one too, because the printer may sit on another
    value than the start of the test. An empty plan gives no commands; `firmware` is checked once, so
    a plan of a hundred segments does not ask twenty times.
    """
    made = list(plan or [])
    if made and firmware not in FIRMWARES:
        raise ValueError(tr("Volura does not know the firmware {firmware}")
                         .format(firmware=firmware))
    return [[float(z), command(firmware, v)] for z, v in made]


def _box(n) -> mf.Manifold:
    """The hollow box of `n` segments: a square tube `FOOT` wide, with a floor and no top.

    The wall is `WALL` millimetres thick, which is `WALL_LINES` lines of the nozzle, and the inside is
    open all the way up above the floor of `FLOOR` millimetres. The hole runs `MARGIN` past the top, so
    the top edge of the wall is one sharp square and nothing stands in the way of a corner.
    """
    top = n * SEG_H
    outer = mf.Manifold.cube([FOOT, FOOT, top]).translate((-0.5 * FOOT, -0.5 * FOOT, 0.0))
    inside = FOOT - 2.0 * WALL
    hole = mf.Manifold.cube([inside, inside, top - FLOOR + MARGIN]) \
        .translate((-0.5 * inside, -0.5 * inside, FLOOR))
    return outer - hole


def _loops(text) -> list:
    """The outline of a value, `CAP` millimetres high and centred on the origin.

    A value that is too wide for the wall between the corners is made smaller, which keeps the letters
    legible: `EDGE` millimetres of wall stay at both ends of it. Empty when this core runs without a
    Qt app or without fonts, or when the text yields nothing: a tower without numbers is not an error.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # numbers need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(text, height=CAP)
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    made = [np.asarray(L, dtype=float) - mid for L in loops]
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    room = FOOT - 2.0 * EDGE
    if wide > room > 0.0:
        made = [one * (room / wide) for one in made]
    return [np.ascontiguousarray(one) for one in made]


def _cut(text, z_from) -> mf.Manifold:
    """The cut of one value into the outside of the front wall of the segment that starts at `z_from`.

    The outline is drawn flat and pulled `ENGRAVE + MARGIN` millimetres up, so the cut starts `MARGIN`
    millimetres in front of the wall and engraves `ENGRAVE` millimetres into it, which leaves at least
    `ROOM` millimetres of wall behind the numbers. Turning the cut about x stands the letters upright
    on the front wall and reads them from the front, the way the calibration cube engraves its axes;
    it lands in the middle of its own segment, so the walls between two segments stay whole. None when
    this system has no fonts, so the segment simply carries no number.
    """
    loops = _loops(text)
    if not loops:
        return None
    cut = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), ENGRAVE + MARGIN)
    return cut.rotate((90.0, 0.0, 0.0)) \
        .translate((0.0, ENGRAVE - 0.5 * FOOT, z_from + 0.5 * SEG_H))


def _engraved(vals, box) -> mf.Manifold:
    """The box with the value of every segment engraved in the middle of its front wall.

    The cuts are taken out of the box instead of stood on it, so the tower stays `FOOT` millimetres
    wide and nothing hangs over; `ENGRAVE` out of a wall of `WALL` leaves a wall behind the numbers.
    The box as it stands when this system has no fonts.
    """
    cuts = [c for c in (_cut(value_text(v), i * SEG_H) for i, v in enumerate(vals))
            if c is not None]
    if not cuts:
        return box
    return box - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The tower on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def pressure_advance_test(start=0.0, end=0.1, step=0.005, labels=True) -> mf.Manifold:
    """One printable pressure advance test: a hollow tower, one segment per pressure advance value.

    The tower is a square box of `FOOT` by `FOOT` millimetres with sharp corners and walls of
    `WALL_LINES` lines of the nozzle, hollow above a floor of `FLOOR` millimetres and without a top, so
    the corners are all there is to look at. It has one segment of `SEG_H` millimetres per value, from
    `start` by `step` towards `end`, and so is `segments * SEG_H` millimetres tall in total. With
    `labels` the value of every segment is engraved `ENGRAVE` millimetres deep in the middle of the
    front wall of that segment, in letters `CAP` millimetres high; nothing is stood on the box, so the
    tower stays `FOOT` millimetres across in every direction.

    The result is one closed volume on `z = 0`, centred in x and y, that prints standing up with
    nothing to support. `start` and `end` are clamped to what a firmware takes, but a step that is no
    number, a step of 0 or less, a start equal to the end and a test with fewer than `SEGMENTS_MIN` or
    more than `SEGMENTS_MAX` segments are a ValueError; without fonts the walls carry no numbers. See
    `steps` and `commands` for the plan of values the printer has to follow.
    """
    vals = _values(start, end, step)
    tower = _box(len(vals))
    return _centred(_engraved(vals, tower) if labels else tower)
