"""First layer tests: the computational core of thin squares in the corners and the middle of the bed.

No Qt in this file. It gives the test as five separate closed volumes, so it can be tested on its
own and reused in the model and the export.

A **first layer test** is the test print that says whether the bed is level and the first layer is
squashed right everywhere on it: five thin squares of plastic, one or two layers high, standing in
the four corners of the bed and in the middle of it. They are far apart on purpose, because one
corner of a bed can be a hair too high while the middle is perfect - and that is exactly what a
single test in the middle can never show.

A square is `size` millimetres on a side and `height` millimetres high, so at 0.3 millimetres it
is a single layer of plastic and at 0.6 two of them. Where the bed is too far from the nozzle the
plastic lies loose and you can peel the square off with a fingernail; where the bed is too close
the nozzle drags through it, the surface goes rough and the edges curl. A bed that is level and a
first layer that is squashed right leave five squares that are equally stuck and equally smooth.

The squares lie where they lie: `margin` millimetres from the edge of the bed to a corner square
and the middle square in the middle, in bed coordinates from `0` to `bed_x` and `0` to `bed_y`.
The size is clamped so that the corner squares and the middle one never touch each other and all
five stay on the bed. The result is a list of five closed volumes on `z = 0`, kept apart on
purpose: one piece per place, so every one of them can be looked at on its own.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: mm the bed measures along x and along y
BED_MIN, BED_MAX = 80.0, 1000.0
#: mm the shortest and the longest side of a square
SIZE_MIN, SIZE_MAX = 10.0, 60.0
#: mm the thinnest and the thickest a square is
HEIGHT_MIN, HEIGHT_MAX = 0.1, 1.0
#: mm from the edge of the bed to the nearest side of a corner square
MARGIN_MIN, MARGIN_MAX = 5.0, 50.0


def _number(value, lo, hi):
    """A number of the test, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the numbers of a first layer test must be numbers"))
    return max(lo, min(hi, v))


def _sizes(bed_x, bed_y, size, height, margin) -> dict:
    """The clamped numbers of the test and the places of its squares, in bed coordinates.

    A corner square keeps `margin` millimetres of bed between it and the two edges it is nearest
    to, so its centre sits at `margin + size / 2` from the corner; the fifth square is in the
    middle of the bed. `centres` lists the five of them in the order front left, front right,
    back left, back right, middle.

    The middle square has to clear the four corner ones, which takes three sides of `size`
    between the edges of the bed minus twice the margin, so the size drops to a third of what is
    left when the squares would otherwise overlap or hang off a small bed.
    """
    bx = _number(bed_x, BED_MIN, BED_MAX)
    by = _number(bed_y, BED_MIN, BED_MAX)
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    m = _number(margin, MARGIN_MIN, MARGIN_MAX)
    w = _number(size, SIZE_MIN, SIZE_MAX)
    w = min(w, (min(bx, by) - 2.0 * m) / 3.0)
    c = m + 0.5 * w
    return {"bed_x": bx, "bed_y": by, "size": w, "height": h, "margin": m,
            "centres": [(c, c), (bx - c, c), (c, by - c), (bx - c, by - c),
                        (0.5 * bx, 0.5 * by)]}


def sizes(bed_x=220.0, bed_y=220.0, size=30.0, height=0.3, margin=15.0) -> dict:
    """The numbers a first layer test is made of, so a dialog can show them.

    `bed_x` and `bed_y` are the size of the bed, `size` the side of a square, `height` how thick a
    square is and `margin` how far a corner square stands from the edge of the bed. Every number
    is clamped to what prints and the size drops when the squares would overlap or leave the bed,
    so what stands here is what prints; `centres` are the five places in bed coordinates. Anything
    but a number is a ValueError.
    """
    s = _sizes(bed_x, bed_y, size, height, margin)
    return {"bed_x": s["bed_x"], "bed_y": s["bed_y"], "size": s["size"],
            "height": s["height"], "margin": s["margin"], "centres": s["centres"]}


def _square(w, h) -> mf.Manifold:
    """One square of `w` millimetres on a side and `h` millimetres high, centred on the origin."""
    return mf.Manifold.extrude(mf.CrossSection.square((w, w), True), h)


def first_layer_test(bed_x=220.0, bed_y=220.0, size=30.0, height=0.3,
                     margin=15.0) -> list:
    """Five thin squares on the bed: four in its corners and one in the middle.

    The squares are `size` millimetres on a side and `height` millimetres high and stand on
    `z = 0` in bed coordinates: `margin` millimetres from the edge of the bed to a corner square
    and one square in the middle of a bed of `bed_x` by `bed_y` millimetres.

    The result is a list of five closed volumes in the order front left, front right, back left,
    back right, middle. They are separate pieces on purpose, because where each of them lies on
    the bed is what the test is about. Every number is clamped to what prints and the size drops
    when the squares would overlap or leave the bed; anything but a number is a ValueError.
    """
    s = _sizes(bed_x, bed_y, size, height, margin)
    return [_square(s["size"], s["height"]).translate((x, y, 0.0)) for x, y in s["centres"]]
