"""Max speed tests: the computational core of a vase-like single-wall cylinder whose segments
print at a rising volumetric flow.

No Qt in this file except the labels, which come from `fonts` the way the engraved values of a
pressure advance test do. It gives the test as one closed volume, so it can be tested on its own
and reused in the model and the export, and it gives the plan of flows the printer has to follow,
together with the `M220` line every segment is worth at the layer height and line width the slice
uses, so a slice can be told to speed up halfway.

A **max speed test** is the test print that says how many cubic millimetres of plastic per second
this hotend can actually melt: a cylinder `DIAMETER` millimetres across and `segments` segments of
`SEG_H` millimetres tall, printed as one wall of a line of the nozzle with no infill and no top -
a vase, in other words - in which every segment prints with a higher volumetric flow than the one
under it. Too little flow prints clean anywhere; past what the hotend can melt the wall goes rough,
thin or glassy, and the highest segment that still came out smooth is the flow to keep under. The
segments are counted on the print itself by the little ribs stood on the wall at their boundaries,
and the flow of the first and last segment stands engraved on a small tab at the foot, so the
lowest and the highest number on the tower are read without counting to twenty.

The speed of a segment follows from its flow, because a wall of `layer_height` by `line_width`
square millimetres laid at `speed` millimetres per second deposits exactly
`layer_height * line_width * speed` cubic millimetres per second: the speed is
`flow / (layer_height * line_width)`, and it is the only way to reach a volumetric flow by
printing faster. That is also why a segment whose speed would pass `SPEED_MAX` millimetres per
second is refused - no printer moves there sanely, and the message names the segment so the user
sees which flow to stop at.

Unlike a temperature tower or a pressure advance test, this test cannot carry finished commands on
itself: the percent depends on the layer height and line width, and those only stand known when
the user picks the printer in the slice dialog. So the part keeps the plan of flows, and the
slice turns it into commands with the stacks it is about to print with. CuraEngine has no
per-mesh volumetric flow either, and its vase mode is not per-mesh - `magic_spiralize` is not
`settable_per_mesh`, so a mesh cannot ask to spiralise - which leaves the setting every firmware
has anyway: the test prints every segment at a fixed **base** speed - `mesh_settings` hands
`speed_print`, `speed_wall`, `speed_wall_0` and `speed_topbottom` of its own mesh, so 100 % is
known - and asks for every segment's speed with `M220 S<percent>` at its height, which the
firmware applies to those feedrates while printing. The wall is then single-lined per mesh the
way a vase is: `wall_line_count` 1, `infill_sparse_density` 0 and `top_layers` 0, all of which
CuraEngine does accept for a single mesh; other parts on the bed print with the print's own
settings, only faster or slower while this tower prints.

The plan is the same mechanism as the height commands of the other tests: a list of
``[z_from, flow]`` kept in the part's `g.meta["speed_steps"]`, which `commands` turns into
``[z_from, "M220 S…"]`` - every segment asks for its percent, the lowest one too, because a
printer often sits on another speed factor. `slicer.with_commands` writes those at the first layer
at or above their height, the same way the pressure advance test's commands and a colour change
by height go in.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr, tr_noop

#: mm across the cylinder (its wall is one line of the nozzle thick)
DIAMETER = 40.0
#: mm a segment is tall
SEG_H = 5.0
#: mm wide one line of the nozzle is, which is how thick the wall of the cylinder is
NOZZLE = 0.4
#: how many faces the cylinder is drawn with
SEGMENTS = 64
#: mm tall the rib at a segment boundary is, a step the fingers and the eye catch
RIB_H = 0.8
#: mm the rib stands out of the wall
RIB_OUT = 0.8
#: mm wide the rib is along the wall
RIB_W = 3.0
#: mm the rib sinks into the wall, so it is one volume with the cylinder
RIB_IN = 0.4
#: mm the tab at the foot stands out past the wall
TAB_OUT = 11.5
#: mm the tab sinks into the wall, so it is one volume with the cylinder
TAB_IN = 0.4
#: mm wide the tab at the foot is
TAB_W = 24.0
#: mm thick the tab at the foot is: layers that are all bottom, so it needs no top
TAB_T = 1.2
#: mm deep the flows are engraved in the tab
ENGRAVE = 0.3
#: mm high the letters of a flow are
CAP = 3.0
#: mm of tab that has to stay around a number, so it does not run over an edge
EDGE = 2.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5

#: the fewest and the most segments a test can have
SEGMENTS_MIN, SEGMENTS_MAX = 2, 20
#: mm³/s the leanest and the richest a segment prints at
FLOW_MIN, FLOW_MAX = 1.0, 60.0
#: the smallest and the biggest step between two segments
STEP_MIN, STEP_MAX = 0.1, 30.0
#: mm/s the speed a segment may not pass: above this no printer moves sanely
SPEED_MAX = 500.0
#: mm/s the wall prints at for 100 %: the base speed the M220 percents are relative to
BASE_SPEED = 100.0
#: the default start, end and step of the dialog: nine segments from 5 to 25 in steps of 2.5
FLOW_START, FLOW_END, FLOW_STEP = 5.0, 25.0, 2.5
#: the layer height and line width the speeds assume when nothing is known of the printer yet
DEFAULT_LAYER_HEIGHT, DEFAULT_LINE_WIDTH = 0.2, 0.4

#: the presets of the dialog, in its order: name -> (start, end, step)
PRESETS = {"standard_hotend": (5.0, 20.0, 2.5), "high_flow": (10.0, 40.0, 5.0)}
#: what the dialog calls every preset: the hotend and the flow it is known to melt
PRESET_LABELS = {"standard_hotend": tr_noop("Standard hotend 5-20"),
                 "high_flow": tr_noop("High flow 10-40")}


def preset_label(name) -> str:
    """What the dialog calls a preset; one it does not know is its own name."""
    return tr(PRESET_LABELS[name]) if name in PRESET_LABELS else str(name)


def default_preset() -> str:
    """The preset to offer when nothing was chosen: the first one, the standard hotend."""
    return next(iter(PRESETS))


def flow_text(flow) -> str:
    """A volumetric flow the way an engraving and a summary write it: 7.5, not 7.500000000000001.

    The flows come out of a walk by `step`, so they carry the noise of that walk; `:g` gives the
    shortest form that means the same number. A flow that is no number is written as it stands, and
    one so small that it prints nothing is written as 0.
    """
    try:
        v = float(flow)
    except (TypeError, ValueError):
        return str(flow)
    if math.isfinite(v) and abs(v) < 1e-9:
        v = 0.0                             # -0.0 and -1e-17 are the same value here
    return f"{v:g}"


def _number(value, lo, hi):
    """A number of the test, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the flows of a max speed test must be numbers"))
    return max(lo, min(hi, v))


def _measure(value) -> float:
    """A layer height or a line width the slice uses; only a number above 0 makes sense here."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(tr("the layer height and line width of a max speed test must be "
                            "numbers above 0"))
    return v


def _flows(start, end, step) -> list:
    """The flows of the segments, the lowest first, with the test's validation.

    `start` is the flow of the lowest segment and every next segment stands `step` higher, walking
    from `start` towards `end` and stopping as soon as the next flow would pass `end` - so the test
    has `(end - start) / step + 1` segments and both `start` and the last flow at or under `end`
    are printed. A range that walks down works just the same, because a tower can be read from the
    top too. A step of 0 (or less) would never reach the other flow, so it is refused; so are a
    tower of fewer than `SEGMENTS_MIN` segments, which is nothing to read, and one of more than
    `SEGMENTS_MAX`, which does not print in reasonable time.
    """
    a = _number(start, FLOW_MIN, FLOW_MAX)
    b = _number(end, FLOW_MIN, FLOW_MAX)
    try:
        d = float(step)
    except (TypeError, ValueError):
        d = math.nan
    if not math.isfinite(d):
        raise ValueError(tr("the flows of a max speed test must be numbers"))
    if d <= 0.0:
        raise ValueError(tr("the step of a max speed test must be more than 0"))
    n = int(math.floor(abs(a - b) / d + 1e-9)) + 1
    if n < SEGMENTS_MIN:
        raise ValueError(tr("a max speed test needs at least {n} segments").format(n=SEGMENTS_MIN))
    if n > SEGMENTS_MAX:
        raise ValueError(tr("a max speed test has at most {n} segments").format(n=SEGMENTS_MAX))
    direction = 1.0 if b >= a else -1.0
    # the flows are rounded to a thousandth, so a walk of 2.5 steps carries no noise and the last
    # one of a walk down is 0.0 instead of the -0.0 that floating point leaves behind
    return [round(a + k * d * direction, 3) + 0.0 for k in range(n)]


def flows(start=FLOW_START, end=FLOW_END, step=FLOW_STEP) -> list:
    """The volumetric flow of every segment, lowest first, validated as in `max_speed_test`.

    These are the mm³/s the segments print at; `speeds` says what that means in mm/s at the layer
    height and line width of the print, and `commands` what the printer has to be told. The
    numbers are validated exactly as the geometry is, so the same start, end and step that make
    the tower also give its flows."""
    return _flows(start, end, step)


def steps(start=FLOW_START, end=FLOW_END, step=FLOW_STEP) -> list:
    """The plan of a max speed test, as ``[[z_from mm, flow mm³/s], ...]``, the lowest first.

    Segment `i` stands from `i * SEG_H` millimetres up and prints with flow `i`, so its `z_from`
    is the height at which the printer has to set that flow; the lowest segment starts on the bed,
    so its `z_from` is 0 and its command lands on the very first layer. The numbers are validated
    as in `max_speed_test`.
    """
    return [[round(i * SEG_H, 6), f] for i, f in enumerate(_flows(start, end, step))]


def sizes(start=FLOW_START, end=FLOW_END, step=FLOW_STEP) -> dict:
    """The numbers a max speed test is made of, so a dialog can show them.

    `segments` is how many bands the tower has and `flows` the volumetric flow of every one of
    them, lowest first; `size` is the outside (width, depth, height) in millimetres, which is the
    cylinder's `DIAMETER` with the tab at the foot sticking out of it in one direction, and
    `segments * SEG_H` millimetres high. The numbers are validated as in `max_speed_test` and
    anything but a number is a ValueError.
    """
    fl = _flows(start, end, step)
    return {"segments": len(fl), "flows": fl, "segment_h": SEG_H,
            "size": (DIAMETER + TAB_OUT, DIAMETER, len(fl) * SEG_H)}


def speed(flow, layer_height, line_width) -> float:
    """The mm/s at which a wall of `layer_height` by `line_width` lays `flow` mm³/s.

    A wall of that square cross-section laid at a millimetre per second moves exactly one cubic
    millimetre of plastic, so the speed is the flow divided by the cross-section - which is also
    why the same flow prints slower on a thick, wide wall than on a thin, narrow one. The layer
    height and line width are what the slice really uses, not a guess of the model; a flow of 0
    is allowed here (it just stops the extruder), a layer height or line width of 0 or less is a
    ValueError, because nothing can be laid with it.
    """
    return float(_as_flow(flow)) / (_measure(layer_height) * _measure(line_width))


def _as_flow(flow) -> float:
    """A volumetric flow of the test; anything but a number is refused."""
    try:
        v = float(flow)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the flows of a max speed test must be numbers"))
    return v


def speeds(flows, layer_height, line_width) -> list:
    """The mm/s every segment prints at, with the test's validation.

    Every flow is converted as in `speed`; a segment whose speed would pass `SPEED_MAX`
    millimetres per second is refused with a message that names the segment, counting the lowest
    one as 1, because that flow is past what this slice may chase at any speed a printer moves."""
    out = []
    for i, flow in enumerate(flows or []):
        v = speed(flow, layer_height, line_width)
        if v > SPEED_MAX + 1e-9:
            raise ValueError(tr("segment {n} of a max speed test would print at {speed} mm/s, "
                                "more than {max} mm/s")
                             .format(n=i + 1, speed=f"{v:g}", max=f"{SPEED_MAX:g}"))
        out.append(v)
    return out


def percent(flow, layer_height, line_width) -> int:
    """The `M220` number with which the printer lays `flow`: the percent of `BASE_SPEED`.

    The wall runs at the base speed, so the firmware's speed factor does the rest; the percent is
    rounded up at a half to a whole number, because a printer reads whole percents, and never
    below 1, because `M220 S0` would stop the print.
    """
    return _rounded(speeds([flow], layer_height, line_width)[0] / BASE_SPEED * 100.0)


def percents(flows, layer_height, line_width) -> list:
    """The `M220` number of every segment, lowest first, validated as in `speeds`."""
    return [_rounded(v / BASE_SPEED * 100.0) for v in speeds(flows, layer_height, line_width)]


def _rounded(value) -> int:
    """A number of the test rounded to the nearest whole, halves up - `round` is not, because it
    rounds a half to the nearest even number and two tests of the same half could disagree.

    A half of a measure like 5 / (0.2 * 0.4) comes out of floating point a hair under 62.5, so the
    half is taken from the number rounded to a billionth first; no measure means two billionths."""
    return max(1, int(math.floor(round(value, 9) + 0.5)))


def command(flow, layer_height, line_width) -> str:
    """The line with which the printer sets its speed factor to lay `flow`.

    Every firmware that speaks G-code knows `M220 S<percent>`; the percent is what `flow` means in
    mm/s at this layer height and line width, relative to `BASE_SPEED`, so a printer on the speed
    factor of an earlier test is set right by the lowest segment too."""
    return f"M220 S{percent(flow, layer_height, line_width)}"


def commands(plan, layer_height, line_width) -> list:
    """The plan of `steps` as ``[[z_from mm, command line], ...]``, which is what the slice wants.

    Every segment gets its own percent, the lowest one too, because the printer may sit on another
    speed factor than the start of the test. The flows are checked as in `speeds`, so a plan whose
    top segment would pass `SPEED_MAX` at this layer height and line width is refused with the
    segment named - which is what the slice does with a part that was made for a thinner slice.
    An empty plan gives no commands."""
    made = [(float(z), f) for z, f in (plan or [])]
    per = percents([f for _z, f in made], layer_height, line_width)
    return [[z, f"M220 S{p}"] for (z, _f), p in zip(made, per)]


def mesh_settings(base=BASE_SPEED) -> dict:
    """The CuraEngine settings the test writes on its own part.

    The part prints at the base speed - `speed_print` for the moves around it, `speed_wall` and
    `speed_wall_0` for its walls and `speed_topbottom` for its bottom and the tab at the foot, so
    100 % is a known speed everywhere on the tower - and it prints like a vase that cannot spiral:
    one line of wall, no infill and no top, because `magic_spiralize` is not `settable_per_mesh`
    while these three are. `slicer.per_mesh_settings` keeps what the printer's definition allows
    for a single mesh, so a part made elsewhere loses nothing by carrying all of them."""
    base = float(base)
    return {"speed_print": base, "speed_wall": base, "speed_wall_0": base,
            "speed_topbottom": base, "wall_line_count": 1, "infill_sparse_density": 0.0,
            "top_layers": 0}


def _tube(n) -> mf.Manifold:
    """The tube of `n` segments: a cylinder `DIAMETER` across with a wall of one nozzle line.

    The wall is `NOZZLE` millimetres thick, which is one line of the nozzle - what the single
    wall line of the slice prints - and the inside stays open all the way, because the test has
    no floor of its own: the bottom layers of the slice close it on the bed.
    """
    top = n * SEG_H
    outer = mf.Manifold.cylinder(top, 0.5 * DIAMETER, 0.5 * DIAMETER, SEGMENTS)
    inside = 0.5 * DIAMETER - NOZZLE
    hole = mf.Manifold.cylinder(top + 2.0 * MARGIN, inside, inside, SEGMENTS) \
        .translate((0.0, 0.0, -MARGIN))
    return outer - hole


def _rib(z_from) -> mf.Manifold:
    """The rib at one segment boundary: a step of `RIB_H` stood on the wall.

    The rib sinks `RIB_IN` millimetres into the wall, so it becomes one volume with the cylinder,
    and stands `RIB_OUT` millimetres out of it, where the finger and the eye find it. It sits at
    the boundary in the middle of the wall's run of flat faces, the same spot on every boundary,
    so the ribs stand in one row and count themselves.
    """
    r = 0.5 * DIAMETER
    return mf.Manifold.cube([RIB_IN + RIB_OUT, RIB_W, RIB_H]) \
        .translate((r - RIB_IN, -0.5 * RIB_W, z_from - 0.5 * RIB_H))


def _ribs(n) -> list:
    """The ribs of the boundaries between the `n` segments - the boundaries between two segments,
    not the bottom on the bed and not the top."""
    return [_rib(i * SEG_H) for i in range(1, n)]


def _tab() -> mf.Manifold:
    """The small plate at the foot that carries the two engraved flows.

    The tab stands out of the wall at `z = 0`, `TAB_T` millimetres thick - layers that are all
    bottom to the slice, so it needs no top - and sinks into the wall, so it is one volume with
    the cylinder. It is where the lowest and the highest number of the tower are read, so it
    stays out of the way of the wall itself."""
    r = 0.5 * DIAMETER
    return mf.Manifold.cube([TAB_IN + TAB_OUT, TAB_W, TAB_T]) \
        .translate((r - TAB_IN, -0.5 * TAB_W, 0.0))


def _loops(text) -> list:
    """The outline of one flow, `CAP` millimetres high and centred on the origin.

    A number that is too wide for its half of the tab is made smaller, which keeps the letters
    legible: `EDGE` millimetres of tab stay at its ends. Empty when this core runs without a Qt
    app or without fonts, or when the text yields nothing: a tab without numbers is not an error.
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
    room = 0.5 * TAB_W - 2.0 * EDGE
    if wide > room > 0.0:
        made = [one * (room / wide) for one in made]
    return [np.ascontiguousarray(one) for one in made]


def _cut(text, cy) -> mf.Manifold | None:
    """The cut that engraves one flow `ENGRAVE` millimetres deep in the tab at `y = cy`.

    The outline is drawn flat and pulled `ENGRAVE + MARGIN` millimetres up from the depth of the
    engraving, so the cut starts `MARGIN` millimetres above the tab and bites `ENGRAVE`
    millimetres into it, and `TAB_T - ENGRAVE` of tab stays under it. None when this system has
    no fonts, so the tab simply carries no number there.
    """
    loops = _loops(text)
    if not loops:
        return None
    made = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), ENGRAVE + MARGIN)
    return made.translate((0.5 * DIAMETER + 0.5 * (TAB_OUT - TAB_IN), cy, TAB_T - ENGRAVE))


def _labelled(fl, tab) -> mf.Manifold:
    """The tab with the flow of the first segment engraved at its front end and that of the last
    segment at its back end.

    Only the two end flows are engraved, because counting the ribs gives every segment in between
    its number; the cuts are taken out of the tab instead of stood on it, so nothing hides the
    wall above it. The tab as it stands when this system has no fonts.
    """
    cuts = [c for c in (_cut(flow_text(fl[0]), -0.25 * TAB_W),
                        _cut(flow_text(fl[-1]), 0.25 * TAB_W)) if c is not None]
    if not cuts:
        return tab
    return tab - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def max_speed_test(start=FLOW_START, end=FLOW_END, step=FLOW_STEP, labels=True) -> mf.Manifold:
    """One printable max speed test: a single-walled tube, one segment per volumetric flow.

    The tower is a cylinder of `DIAMETER` millimetres across with a wall of one line of the
    nozzle and no floor, `segments` segments of `SEG_H` millimetres, from `start` by `step`
    towards `end` millimetres cubed per second, and so `segments * SEG_H` millimetres tall in
    total. A rib of `RIB_H` millimetres stands on the wall at every boundary between two
    segments, so the segments are counted on the print, and with `labels` the flow of the first
    and the last segment is engraved `ENGRAVE` millimetres deep in the little tab that stands at
    the foot, in letters `CAP` millimetres high.

    The result is one closed volume on `z = 0`, centred on the cylinder's axis, that prints
    standing up with nothing to support; the slice is what makes it a vase, through
    `mesh_settings`. `start` and `end` are clamped to what prints, but a step that is no number,
    a step of 0 or less and a test with fewer than `SEGMENTS_MIN` or more than `SEGMENTS_MAX`
    segments are a ValueError; without fonts the tab carries no numbers. See `steps` for the plan
    of flows and `commands` for the lines the slice writes with it."""
    fl = _flows(start, end, step)
    tab = _labelled(fl, _tab()) if labels else _tab()
    return _tube(len(fl)) + mf.Manifold.batch_boolean(_ribs(len(fl)) + [tab], mf.OpType.Add)