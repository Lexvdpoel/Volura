"""Soap dishes: the computational core of a dish a bar of soap lies on while it dries.

No Qt in this file: the dish is a tray of extrusions and boxes, so it can be tested on its own and
reused in the model and the export.

A **soap dish** is a tray `length` millimetres long and `width` millimetres wide that stands `rim`
millimetres high, with a wall of `wall` millimetres around a floor of `FLOOR` millimetres. The soap
does not lie in the tray but on **ribs**: `ribs` bars of `RIB_W` millimetres that stand on the
floor and run from wall to wall, so the soap lies above the water instead of in it.

That water leaves through the **drain slots**: one slot of `SLOT_W` millimetres through the floor
for every gap between two neighbouring ribs, halfway between them, so the soap never lies over the
hole it drains through. A slot stops `SLOT_END` millimetres short of the wall at either end, so the
floor keeps an edge all round the dish.

The dish prints **flat on its floor**: every wall is vertical and every floor, rim and rib
horizontal, so nothing overhangs and nothing needs support - and support would only scratch the
bottom of the dish. The four corners round with `CORNER` millimetres and the inside with what is
left of that, so a thick wall does not eat the corner of the basin.

Every number is clamped to what prints and anything but a number is refused. The ribs are clamped
too, so that `RIB_GAP` millimetres is left between two of them: a narrower gap clogs with soap and
drains nothing. The result is one closed volume on `z = 0`, centred in x and y, `rim` millimetres
high.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: mm the shortest and the longest dish: long enough for a bar of soap, short enough for a shelf
LENGTH_MIN, LENGTH_MAX = 70.0, 160.0
#: mm the narrowest and the widest dish: wide enough that a wide bar still lies on the ribs
WIDTH_MIN, WIDTH_MAX = 50.0, 120.0
#: mm the least and the most the rim stands above the floor
RIM_MIN, RIM_MAX = 8.0, 30.0
#: mm the thinnest and the thickest wall of the tray
WALL_MIN, WALL_MAX = 1.6, 4.0
#: the fewest and the most ribs: a bar of soap lies on at least three of them
RIBS_MIN, RIBS_MAX = 3, 15
#: mm the floor is thick: thin enough for the water to run through, thick enough to stand on
FLOOR = 2.0
#: mm a rib is wide, the part of the rim it stands up to and mm it stands from the end walls
RIB_W, RIB_PART, RIB_END = 2.4, 0.6, 12.0
#: mm the least that has to stay between two ribs: a narrower gap clogs with soap
RIB_GAP = 6.0
#: mm a drain slot is wide and mm it stops short of the wall at either end
SLOT_W, SLOT_END = 3.0, 8.0
#: mm the corners of the dish round with, and the least of that the inside keeps
CORNER, CORNER_MIN = 10.0, 1.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
OVERCUT = 1.0
#: the straight pieces the rounded corners are approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the dish, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a soap dish must be numbers"))
    return max(lo, min(hi, v))


def _sizes(length, width, rim, wall, ribs) -> dict:
    """The clamped numbers of the dish, where its ribs and slots are and how big it is outside.

    The dish lies with its floor on `z = 0` and is centred on the origin, so the ribs stand at
    `rib_x` millimetres from the middle: the first and the last `RIB_END` millimetres from the
    inside of the end walls and the rest evenly between them. A rib runs from wall to wall and
    stands `RIB_PART` of the rim high, so the soap lies above the water but is still easy to pick
    up, and a slot lies halfway between two neighbouring ribs. `ribs` is clamped past its own range
    too, so that `RIB_GAP` millimetres is left between two of them.
    """
    L = _number(length, LENGTH_MIN, LENGTH_MAX)
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    h = _number(rim, RIM_MIN, RIM_MAX)
    t = _number(wall, WALL_MIN, WALL_MAX)
    span = L - 2.0 * t - 2.0 * RIB_END
    n = int(_number(ribs, RIBS_MIN, RIBS_MAX))
    n = max(RIBS_MIN, min(n, int(span // (RIB_W + RIB_GAP)) + 1))
    step = span / (n - 1)
    return {"length": L, "width": w, "rim": h, "wall": t, "ribs": n,
            "rib_x": [-0.5 * L + t + RIB_END + i * step for i in range(n)],
            "rib_len": w - 2.0 * t, "rib_top": FLOOR + RIB_PART * h,
            "slot_len": w - 2.0 * t - 2.0 * SLOT_END,
            "size_x": L, "size_y": w, "size_z": h}


def sizes(length=110.0, width=75.0, rim=15.0, wall=2.4, ribs=7) -> dict:
    """The numbers a soap dish is made of, so a dialog can show them.

    `length` is how long the dish is and `width` how wide, `rim` how high its rim stands and `wall`
    how thick that rim and the floor are; `ribs` is how many bars stand on the floor, clamped until
    `RIB_GAP` millimetres is left between two of them. `rib_x` are the places those ribs stand,
    measured from the middle of the dish, and `size_x`, `size_y` and `size_z` are the outside of the
    dish, which is how it stands on the bed. Every number is clamped to what prints and anything but
    a number is a ValueError.
    """
    s = _sizes(length, width, rim, wall, ribs)
    return {key: s[key] for key in ("length", "width", "rim", "wall", "ribs", "rib_x",
                                    "size_x", "size_y", "size_z")}


def _outline(s, inset) -> mf.CrossSection:
    """The dish `inset` millimetres in from every side, with its corners rounded.

    The corners round with what is left of `CORNER` once that much is taken off, so the basin keeps
    a corner of its own; a wall as thick as the radius would leave none, so `CORNER_MIN` stays.
    """
    r = max(CORNER - inset, CORNER_MIN)
    return (mf.CrossSection.square([s["length"] - 2.0 * inset, s["width"] - 2.0 * inset],
                                   center=True)
            .offset(-r, mf.JoinType.Miter)
            .offset(r, mf.JoinType.Round, circular_segments=SEGMENTS))


def _tray(s) -> mf.Manifold:
    """The tray on its own: the outside pulled up to the rim with the basin cut out of it.

    The basin is cut from the floor up, so what is left is a wall all round and the floor of
    `FLOOR` millimetres under it, which is the face the dish stands on.
    """
    outer = mf.Manifold.extrude(_outline(s, 0.0), s["rim"])
    basin = mf.Manifold.extrude(_outline(s, s["wall"]), s["rim"] - FLOOR + OVERCUT)
    return outer - basin.translate((0.0, 0.0, FLOOR))


def _slot(s, x) -> mf.Manifold:
    """One drain slot through the floor, `SLOT_W` millimetres wide and short of both side walls."""
    return mf.Manifold.cube([SLOT_W, s["slot_len"], FLOOR + 2.0 * OVERCUT]).translate(
        (x - 0.5 * SLOT_W, -0.5 * s["slot_len"], -OVERCUT))


def _rib(s, x) -> mf.Manifold:
    """One rib standing on the floor from wall to wall, so the soap lies above the water."""
    return mf.Manifold.cube([RIB_W, s["rib_len"], s["rib_top"] - FLOOR]).translate(
        (x - 0.5 * RIB_W, -0.5 * s["rib_len"], FLOOR))


def _slot_x(s) -> list:
    """The middle of every gap between two neighbouring ribs: where the drain slots are cut."""
    return [0.5 * (a + b) for a, b in zip(s["rib_x"], s["rib_x"][1:])]


def _dish(s) -> mf.Manifold:
    """The tray with its slots cut through the floor and its ribs stood up on it.

    The slots are cut before the ribs are added, so a rib always stands beside a slot instead of
    through it, and a rib reaches the walls exactly, which is where it gets its strength from.
    """
    dish = _tray(s)
    for x in _slot_x(s):
        dish = dish - _slot(s, x)
    for x in s["rib_x"]:
        dish = dish + _rib(s, x)
    return dish


def soap_dish(length=110.0, width=75.0, rim=15.0, wall=2.4, ribs=7) -> mf.Manifold:
    """One printable soap dish: a bar of soap lies on ribs above a floor the water runs through.

    The dish is `length` millimetres long, `width` millimetres wide and `rim` millimetres high, with
    a wall of `wall` millimetres and a floor of `FLOOR` millimetres. `ribs` ribs stand on that floor
    and run from wall to wall, and every gap between two of them carries a slot of `SLOT_W`
    millimetres through the floor, so the soap lies dry while the water runs down under it.

    The result is one closed volume on `z = 0`, centred in x and y and `rim` millimetres high, that
    prints flat on its floor: every face is vertical or horizontal, so nothing overhangs and nothing
    needs support. Every number is clamped to what prints - the ribs until `RIB_GAP` millimetres is
    left between two of them - and anything but a number is a ValueError.
    """
    return _dish(_sizes(length, width, rim, wall, ribs))
