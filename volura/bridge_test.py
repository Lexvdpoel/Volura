"""Bridge tests: the computational core of a row of bridges over gaps that grow wider.

No Qt in this file except the spans, which come from `fonts` the way a hex key holder's sizes
do. It gives the test as one closed volume, so it can be tested on its own and reused in the
model and the export.

A **bridge test** is the test print that says how far a printer bridges a gap without support:
a row of `count` bridges, standing on pillars of `PILLAR` millimetres, over gaps that start at
`first` millimetres and grow by `step` millimetres from left to right. Every bridge is a slab of
`BRIDGE` millimetres that runs straight over its gap and sinks into the pillars on both sides, so
its underside is one flat horizontal overhang with nothing under it - which is exactly what the
test is about. The bridges hang `height` millimetres above the bed, so the printer cannot be
saved by what stands under them, and they are `width` millimetres deep, so a bridge is a wall laid
flat and not a thread.

The **span** of every bridge stands engraved `ENGRAVE` millimetres deep in the top of that
bridge, in letters half the width of a bridge high and made smaller when they would run past
their gap. The top of the bridges lies flush with the tops of the pillars, so the row prints
without a single support and the number of the bridge you are reading is never in the way.

Print it once and read the widest span that still came out clean: that is how far this printer
bridges, and the gap to stay under when a part has an overhang that lies flat. The whole test
stands on a plate of `PLATE` millimetres that keeps `EDGE` millimetres of plastic beside the row,
so no pillar stands against the edge of the plate. Without fonts the row simply has no
numbers. The widest spans are dropped when the row would grow past `BED` millimetres, so it
always fits the bed. The result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the shortest and the longest gap of the first bridge
FIRST_MIN, FIRST_MAX = 5.0, 30.0
#: mm the smallest and the biggest step between two bridges
STEP_MIN, STEP_MAX = 5.0, 20.0
#: the fewest and the most bridges in a row
COUNT_MIN, COUNT_MAX = 2, 8
#: mm the lowest and the highest a bridge hangs above the bed
HEIGHT_MIN, HEIGHT_MAX = 4.0, 20.0
#: mm the narrowest and the widest a bridge is
WIDTH_MIN, WIDTH_MAX = 5.0, 20.0
#: mm thick the plate is, and mm thick a bridge
PLATE, BRIDGE = 1.2, 1.2
#: mm wide a pillar is, in the direction of the row
PILLAR = 4.0
#: mm of plate that lies beside the row of bridges
EDGE = 3.0
#: mm deep the spans are engraved in the top of a bridge
ENGRAVE = 0.6
#: the height of a letter as a part of the width of a bridge
LETTER = 0.5
#: mm taken off a span, so its number keeps plastic around it
LABEL_ROOM = 4.0
#: mm the whole row has to stay within, so it fits the bed
BED = 250.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
OVERCUT = 0.5


def _number(value, lo, hi):
    """A number of the test, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the numbers of a bridge test must be numbers"))
    return max(lo, min(hi, v))


def _sizes(first, step, count, height, width) -> dict:
    """The clamped numbers of the test, the places of its pillars and the size it measures.

    The bridges are the spans of the row, `first` millimetres for the first one and `step` wider
    for every one after it, and `mids[i]` is the middle of the gap bridge `i` lies over. A pillar
    stands between two gaps and one at both ends, so the row is `length` long; `depth` is the
    width of a bridge with the plate beside it and `top` is the height of the bridges, which is
    also the height of the pillars.

    The row that no bed holds is a row that prints on two sheets, so the widest spans are dropped
    until the whole test fits `BED` millimetres; the smallest bridges are the ones worth printing
    and those are the ones that stay.
    """
    f = _number(first, FIRST_MIN, FIRST_MAX)
    s = _number(step, STEP_MIN, STEP_MAX)
    n = int(_number(count, COUNT_MIN, COUNT_MAX))
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    spans = [f + k * s for k in range(n)]
    while len(spans) > COUNT_MIN and sum(spans) + (len(spans) + 1) * PILLAR > BED:
        spans.pop()
    length = sum(spans) + (len(spans) + 1) * PILLAR
    mids, x = [], 0.0
    for span in spans:
        mids.append(x + PILLAR + 0.5 * span)
        x += PILLAR + span
    return {"first": f, "step": s, "count": len(spans), "height": h, "width": w,
            "spans": spans, "mids": mids, "length": length, "depth": w + 2.0 * EDGE,
            "top": h + BRIDGE}


def sizes(first=10.0, step=10.0, count=5, height=8.0, width=8.0) -> dict:
    """The numbers a bridge test is made of, so a dialog can show them.

    `first` is the span of the shortest bridge and `step` how much wider every next one is, of
    which there are `count` in the row; `height` is how high the bridges hang above the bed and
    `width` how wide a bridge is. `spans` are the spans of the row as it prints: the widest ones
    drop out when the row would grow past the bed. `size` is the outside (length, depth, height)
    in millimetres. Every number is clamped to what prints and anything but a number is a
    ValueError.
    """
    s = _sizes(first, step, count, height, width)
    return {"first": s["first"], "step": s["step"], "count": s["count"], "height": s["height"],
            "width": s["width"], "spans": s["spans"],
            "size": (s["length"], s["depth"], s["top"])}


def _box(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """The block between two opposite corners, so plate, pillars and bridges read as boxes."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate((x0, y0, z0))


def _row(s) -> mf.Manifold:
    """The plate with the pillars on it and a bridge over every gap, as one solid.

    The row is built from `x = 0` towards `length`, a pillar of `PILLAR` at a time, and the
    bridge over the gap that follows it reaches `PILLAR` millimetres into the pillar on both
    sides - which is also what fuses the two of them into one solid. A bridge ends where the
    pillars end, so its top is flush with theirs and its underside is flat over the gap.
    """
    parts = [_box(0.0, s["length"], 0.0, s["depth"], 0.0, PLATE)]
    x = 0.0
    for span in s["spans"]:
        parts.append(_box(x, x + PILLAR, EDGE, EDGE + s["width"], PLATE, s["top"]))
        parts.append(_box(x, x + PILLAR + span + PILLAR, EDGE, EDGE + s["width"],
                          s["height"], s["top"]))
        x += PILLAR + span
    parts.append(_box(x, x + PILLAR, EDGE, EDGE + s["width"], PLATE, s["top"]))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _letters(s) -> list:
    """The spans to engrave, one list of loops per bridge, centred on the bridge it names.

    Empty when this core runs without a Qt app or without fonts, or when a number yields
    nothing: a bridge test without numbers is not an error. A number is drawn flat and centred
    on the middle of its gap, and made smaller when it would run past that gap, so the
    engraving stops short of the pillars on both sides of it.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    made = []
    for x, span in zip(s["mids"], s["spans"]):
        try:
            loops = fonts.text_loops(f"{span:g}", height=LETTER * s["width"])
        except Exception:
            continue
        if not loops:
            continue
        allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
        wide = float(allp[:, 0].max() - allp[:, 0].min())
        scale = 1.0
        if wide > 1e-9 and span - LABEL_ROOM > 0.0:
            scale = min(1.0, (span - LABEL_ROOM) / wide)
        mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
        made.append([np.ascontiguousarray((np.asarray(L, dtype=float) - mid) * scale
                                          + np.array([x, EDGE + 0.5 * s["width"]]))
                     for L in loops])
    return made


def _engraved(s, row) -> mf.Manifold:
    """The row with every span cut `ENGRAVE` millimetres deep in the top of its bridge."""
    cuts = [mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), ENGRAVE + OVERCUT)
            .translate((0.0, 0.0, s["top"] - ENGRAVE)) for loops in _letters(s)]
    if not cuts:
        return row
    return row - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The test on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def bridge_test(first=10.0, step=10.0, count=5, height=8.0, width=8.0) -> mf.Manifold:
    """One printable bridge test: a row of bridges over gaps that grow wider.

    The row is `count` bridges over gaps of `first`, `first + step` and so on, standing on
    pillars of `PILLAR` millimetres that rise from a plate of `PLATE`. Every bridge is a slab of
    `BRIDGE` millimetres that runs over its whole gap and into the pillars beside it, so its
    underside is a flat overhang with nothing under it; it hangs `height` millimetres above the
    bed and is `width` millimetres deep. The span of every bridge is engraved `ENGRAVE`
    millimetres deep in the top above its gap, in letters half the width of a bridge high.

    The result is one closed volume on `z = 0`, centred in x and y, that prints without support.
    Every number is clamped to what prints and anything but a number is a ValueError; the widest
    spans are dropped when the row would grow past `BED` millimetres, and without fonts the
    bridges carry no numbers.
    """
    s = _sizes(first, step, count, height, width)
    return _centred(_engraved(s, _row(s)))
