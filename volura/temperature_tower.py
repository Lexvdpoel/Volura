"""Temperature towers: the computational core of a stack of segments, one per nozzle temperature.

No Qt in this file except the temperatures, which come from `fonts` the way the raised labels of a
clearance test do. It gives the tower as one closed volume, so it can be tested on its own and
reused in the model and the export, and it gives the plan of temperatures the printer has to
follow, so a slice can be told to change the nozzle temperature between every two segments.

A **temperature tower** is the test print that says how a printer behaves at every nozzle
temperature: a tower of `segments` segments of `SEG_H` millimetres, each printed at its own
temperature, the hottest at the bottom and the coolest at the top, so the first layers stick best
when the plastic is hot. Every segment is a block with a `45°` overhang on one side, a small
bridge of `BRIDGE_SPAN` millimetres in front of the lowest one and, on the whole tower, a thin
cone up top; the temperature of every segment stands `raised` on its front face in letters at
least `CAP` millimetres high, so you can tell one band from the next while you read the print.

The tower is useless unless the printer really changes temperature at every segment, so the part
carries its plan: a list of ``[z_from, temperature]``, one per segment, the hottest first. When
the tower is sliced the first temperature becomes the print temperature of the slice and every
other one is asked for with `M104` at the first layer at or above its `z_from`. Because the change
is asked for by height, a tower that stands on the bed next to other parts changes the temperature
for the whole bed - which is what the dialog warns about.

The whole thing stands on a plate of `BASE` millimetres that is a little bigger than the tower, so
no segment overhangs the bed itself. The result is one closed volume on `z = 0`, centred in x and
y, that prints without support; without fonts the bands carry no numbers.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import geom
from .i18n import tr

#: mm a segment is tall
SEG_H = 10.0
#: mm thick the plate under the tower is
BASE = 1.0
#: mm a side of a segment is wide (the tower is a square column)
FOOT = 24.0
#: mm of plate that lies beside the tower
EDGE = 5.0
#: mm the thin cone on top rises above the top segment
SPIRE = 2.0
#: mm the cap height of a temperature (readable while you read the print)
CAP = 6.0
#: mm a temperature stands out in front of a segment, and mm it bites into it
LABEL_PROTR, LABEL_EMBED = 0.8, 0.6
#: mm the 45° overhang reaches out on a side, and mm it bites into the segment
OVG, OVG_EMBED = 4.0, 1.0
#: mm the gap the bridge spans, how wide its two posts are and how thick its beam
BRIDGE_SPAN, BRIDGE_POST, BRIDGE_T = 10.0, 3.0, 1.2
#: mm a bridge post stands out in front of a segment, and mm it bites into it
BRIDGE_PROTR, BRIDGE_EMBED = 3.0, 1.0
#: mm how much two neighbouring segments and the lowest one and the plate overlap, so they fuse
OVERLAP = 0.2
#: mm past a face the cone bites into the top segment
SPIRE_EMBED = 0.5
#: the straight pieces the cone is approximated with
SEGMENTS = 48

#: the temperatures of the material presets: (hottest, coolest, step)
PRESETS = {"PLA": (220, 190, 5), "PETG": (250, 220, 5), "ABS": (260, 230, 5),
           "ASA": (260, 230, 5), "TPU": (235, 210, 5)}
#: the default material of the dialog
PRESET_DEFAULT = "PLA"
#: the fewest and the most segments a tower can have
SEGMENTS_MIN, SEGMENTS_MAX = 2, 12
#: °C the coolest and the hottest a nozzle prints at
TEMP_MIN, TEMP_MAX = 150.0, 300.0
#: °C the smallest and the biggest step between two segments
STEP_MIN, STEP_MAX = 1.0, 15.0


def _number(value, lo, hi):
    """A number of the tower, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the temperatures of a temperature tower must be numbers"))
    return max(lo, min(hi, v))


def _temps(start, end, step) -> list:
    """The temperatures of the segments, hottest first, with the validation of the tower.

    A step of 0 °C (or less) would never reach the other temperature and a start equal to the end
    is one segment, so both are refused. The temperatures walk from `start` towards `end` by `step`,
    and stop as soon as the next one would pass `end`, so the last one never goes past the coolest
    temperature asked for. A tower of fewer than `SEGMENTS_MIN` segments is not a tower and one of
    more than `SEGMENTS_MAX` does not print in reasonable time, so both are refused too.
    """
    a = _number(start, TEMP_MIN, TEMP_MAX)
    b = _number(end, TEMP_MIN, TEMP_MAX)
    try:
        d = float(step)
    except (TypeError, ValueError):
        d = math.nan
    if not math.isfinite(d):
        raise ValueError(tr("the temperatures of a temperature tower must be numbers"))
    if d <= 0.0:
        raise ValueError(tr("the step of a temperature tower must be more than 0 °C"))
    if abs(a - b) < 1e-6:
        raise ValueError(tr("the end temperature of a temperature tower must differ from the start"))
    n = int(math.floor(abs(a - b) / d + 1e-9)) + 1
    if n < SEGMENTS_MIN:
        raise ValueError(tr("a temperature tower needs at least {n} segments").format(n=SEGMENTS_MIN))
    if n > SEGMENTS_MAX:
        raise ValueError(tr("a temperature tower has at most {n} segments").format(n=SEGMENTS_MAX))
    direction = 1.0 if b > a else -1.0
    return [a + k * d * direction for k in range(n)]


def steps(start=220.0, end=190.0, step=5.0) -> list:
    """The plan of a temperature tower, as ``[[z_from mm, temperature °C], ...]``, hottest first.

    Segment `i` stands from `BASE + i * SEG_H` millimetres up and is printed at temperature `i`, so
    its `z_from` is the height at which the printer has to switch to that temperature; the first
    entry is the print temperature of the slice itself and needs no switch. Temperatures are whole
    degrees, which is all `M104` understands. The numbers are validated as in `temperature_tower`.
    """
    return [[round(BASE + i * SEG_H, 6), int(round(t))]
            for i, t in enumerate(_temps(start, end, step))]


def sizes(start=220.0, end=190.0, step=5.0) -> dict:
    """The numbers a temperature tower is made of, so a dialog can show them.

    `segments` is how many bands the tower has and `temps` the temperature of every one of them,
    hottest first; `size` is the outside (width, depth, height) in millimetres, which for a tower
    that stands on the bed is the plate it prints on and `segments * SEG_H + BASE` millimetres
    high. The numbers are validated as in `temperature_tower` and anything but a number is a
    ValueError.
    """
    temps = _temps(start, end, step)
    plate = FOOT + 2.0 * EDGE
    return {"segments": len(temps), "temps": [int(round(t)) for t in temps],
            "segment_h": SEG_H, "size": (plate, plate, len(temps) * SEG_H + BASE)}


def _plate() -> mf.Manifold:
    """The plate the tower stands on, `BASE` millimetres thick and `EDGE` beside the column."""
    w = FOOT + 2.0 * EDGE
    return mf.Manifold.cube([w, w, BASE]).translate((-0.5 * w, -0.5 * w, 0.0))


def _block(i, z_from, top) -> mf.Manifold:
    """The block of segment `i`: the square column `FOOT` wide, from `z_from` up to `top`.

    It reaches `OVERLAP` below `z_from` and into the segment under it - or into the plate for the
    lowest one - so the whole tower is one solid instead of `segments` ones that only touch.
    """
    h = top - z_from + OVERLAP
    return mf.Manifold.cube([FOOT, FOOT, h]).translate((-0.5 * FOOT, -0.5 * FOOT, z_from - OVERLAP))


def _overhang(z_top) -> mf.Manifold:
    """The 45° overhang on the +x side of a segment that ends at `z_top`.

    The wedge bites `OVG_EMBED` into the block and reaches `OVG` out of it, so its underside is one
    flat face that leans out at 45 degrees - the overhang the test is about - and it is fused to
    the block instead of only touching it.
    """
    f = 0.5 * FOOT
    tri = [(f - OVG_EMBED, z_top), (f - OVG_EMBED, z_top - (OVG + OVG_EMBED)), (f + OVG, z_top)]
    rot = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return mf.Manifold.extrude(mf.CrossSection([np.ascontiguousarray(np.array(tri, float))]),
                               FOOT).transform(geom.affine(rot, (0.0, 0.5 * FOOT, 0.0)))


def _bridge(z_bottom) -> mf.Manifold:
    """The small bridge in front of the lowest segment: two posts and a beam over a 10 mm gap.

    Two short posts stand out of the front face with `BRIDGE_SPAN` millimetres of open air between
    them and the beam lies across their tops, so the underside of the beam over the gap is a bridge
    with nothing under it; both posts bite into the block, so the bridge is part of the tower.
    """
    y0, y1 = -0.5 * FOOT - BRIDGE_PROTR, -0.5 * FOOT + BRIDGE_EMBED
    t = z_bottom + BASE + 0.4                      # mm above the bed the beam lies on
    parts = [mf.Manifold.cube([BRIDGE_POST, y1 - y0, BRIDGE_T + 1.0])
             .translate((s - BRIDGE_POST * (1 if s > 0 else 0), y0, t))
             for s in (-(0.5 * BRIDGE_SPAN + BRIDGE_POST), 0.5 * BRIDGE_SPAN)]
    parts.append(mf.Manifold.cube([BRIDGE_SPAN + 2.0 * BRIDGE_POST, y1 - y0, BRIDGE_T])
                 .translate((-(0.5 * BRIDGE_SPAN + BRIDGE_POST), y0, t)))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _spire(z_top) -> mf.Manifold:
    """The thin cone on top of the tower: `SPIRE` millimetres from the top segment to its tip.

    It bites `SPIRE_EMBED` into the top segment so it is fused to it, and its tip lands on the top
    of the band, so the tower stays `segments * SEG_H` tall.
    """
    return mf.Manifold.cylinder(SPIRE + SPIRE_EMBED, 4.0, 0.3, SEGMENTS) \
        .translate((0.0, 0.0, z_top - SPIRE_EMBED))


def _tower(s) -> mf.Manifold:
    """The plate with every segment on it, its overhangs, the bridge, the spire and the numbers."""
    n = len(s["temps"])
    tops = [BASE + (i + 1) * SEG_H for i in range(n)]
    tops[-1] -= SPIRE                              # the last band ends its block early for the spire
    parts = [_plate()]
    for i in range(n):
        z_from = BASE + i * SEG_H
        parts.append(_block(i, z_from, tops[i]))
        parts.append(_overhang(tops[i]))
    parts.append(_bridge(BASE))
    parts.append(_spire(tops[-1]))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _labels(s, tower) -> mf.Manifold:
    """The tower with the temperature of every segment stood on its front face.

    Each number stands in the middle of the front face of its own band, in letters `CAP` millimetres
    high, biting `LABEL_EMBED` into the block and standing `LABEL_PROTR` out of it, so it reads from
    the front and is fused to the tower. Empty when this system has no fonts, so the tower simply
    carries no numbers.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return tower
    from . import fonts                            # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return tower
    made = []
    for i, t in enumerate(s["temps"]):
        try:
            loops = fonts.text_loops(f"{int(round(t))}", height=CAP)
        except Exception:
            continue
        if not loops:
            continue
        allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
        mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
        loops = [np.ascontiguousarray(np.asarray(L, dtype=float) - mid) for L in loops]
        slab = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero),
                                   LABEL_PROTR + LABEL_EMBED)
        rot = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        y0 = -0.5 * FOOT + LABEL_EMBED
        zc = BASE + (i + 0.5) * SEG_H if i < len(s["temps"]) - 1 else BASE + i * SEG_H + 0.5 * (SEG_H - SPIRE)
        made.append(slab.transform(geom.affine(rot, (0.0, y0, zc))))
    if not made:
        return tower
    return mf.Manifold.batch_boolean([tower] + made, mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The tower on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def temperature_tower(start=220.0, end=190.0, step=5.0, labels=True) -> mf.Manifold:
    """One printable temperature tower: a stack of segments, one per nozzle temperature.

    The tower has one segment per temperature from `start` towards `end` by `step` degrees, hottest
    first, each `SEG_H` millimetres tall and standing on a plate of `BASE` millimetres that is
    `EDGE` wider than the column all round. A segment is a block `FOOT` millimetres wide with a 45°
    overhang on one side, the lowest one carries a small bridge of `BRIDGE_SPAN` millimetres in
    front and the whole column ends in a thin cone; with `labels` the temperature of every band
    stands raised on its front face in letters `CAP` millimetres high.

    The result is one closed volume on `z = 0`, centred in x and y, that prints without support and
    is `segments * SEG_H + BASE` millimetres high. `start` and `end` are clamped to what a nozzle
    prints at and `step` to a sensible size, but a step that is no number, a start equal to the end
    and a tower with fewer than `SEGMENTS_MIN` or more than `SEGMENTS_MAX` segments are a
    ValueError; without fonts the bands carry no numbers. See `steps` for the plan of temperatures
    the printer has to follow.
    """
    s = {"temps": _temps(start, end, step)}
    tower = _tower(s)
    return _centred(_labels(s, tower) if labels else tower)
