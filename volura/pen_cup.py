"""Pen cups: the computational core of a round cup that stands on a desk and holds pens.

No Qt in this file: the cup is a cylinder with a band of windows cut through its wall, so it can be
tested on its own and reused in the model and the export.

A **pen cup** is a cylinder `diameter` millimetres across and `height` millimetres high, with a wall
of `wall` millimetres and a floor of `floor` millimetres. Its wall carries a **band of windows**:
`columns` **diamonds** round the cup, one above the other in rows, so the cup weighs little and
shows what stands in it.

A diamond is a square of side `side` turned 45 degrees, so it has a corner on top and one at the
bottom and its sides lean at 45 degrees. That is the whole trick: the upper sides of a window are
carried by the wall below them and the lower sides lean at exactly the angle a printer wants, so the
band cuts **without support** - support inside a window would only have to be broken off again.

The band hangs `BAND` millimetres clear of the rim and of the floor, so the cup keeps a strong edge
top and bottom, and the rows interleave: every second row stands half a column further round, which
leaves wall between two diamonds of neighbouring rows instead of a hole straight through the wall.

The cup prints **standing on its floor**: nothing overhangs by more than 45 degrees, so it needs no
support and comes off the bed with the bottom of the floor as the only face that touched anything.

Every number is clamped to what prints and anything but a number is refused. The result is one
closed volume on `z = 0`, centred in x and y, `height` millimetres high.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: mm the smallest and the widest cup across: small enough for a desk, big enough for scissors
DIAMETER_MIN, DIAMETER_MAX = 40.0, 120.0
#: mm the shortest and the tallest cup: pens stand in it, it must not topple on a shelf
HEIGHT_MIN, HEIGHT_MAX = 50.0, 160.0
#: mm the thinnest and the thickest wall of the cup
WALL_MIN, WALL_MAX = 1.6, 4.0
#: mm the thinnest and the thickest floor the cup stands on
FLOOR_MIN, FLOOR_MAX = 1.6, 4.0
#: the fewest and the most diamonds round the cup
COLUMNS_MIN, COLUMNS_MAX = 6, 24
#: mm the band of windows stays clear of the rim and of the floor
BAND = 15.0
#: the part of one column a diamond takes across: the rest is wall between two of them
DIAMOND_PART = 0.6
#: mm of wall between one row of diamonds and the next, on top of half a diamond
ROW_GAP = 2.0
#: mm the cutter is longer than the wall, so it reaches past both faces of the wall and cuts through
CUT_PAD = 4.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
OVERCUT = 1.0
#: the straight pieces the wall of the cup is approximated with
SEGMENTS = 128


def _number(value, lo, hi):
    """A number of the cup, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a pen cup must be numbers"))
    return max(lo, min(hi, v))


def _rows(height, step, diamond) -> int:
    """How many rows of diamonds fit in the band: from `BAND` up while the top stays below the rim.

    A row stands `step` millimetres above the previous one and the first hangs `BAND` millimetres
    under its own middle, so a row only counts when its top corner stays `BAND` under the rim.
    """
    n = 0
    z = BAND + 0.5 * diamond
    while z + 0.5 * diamond < height - BAND:
        n += 1
        z += step
    return n


def _sizes(diameter, height, wall, floor, windows, columns) -> dict:
    """The clamped numbers of the cup, how big a diamond is and how many rows of them fit.

    A diamond takes `DIAMOND_PART` of the `2 * pi * radius / columns` millimetres of one column, so
    40 percent of the wall is left between two of them. Being a square turned 45 degrees it is as
    high as it is wide: `diamond` millimetres. Rows stand every half of that plus `ROW_GAP`
    millimetres, so two rows of diamonds overlap in z but stand half a column apart round the cup.
    """
    d = _number(diameter, DIAMETER_MIN, DIAMETER_MAX)
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _number(wall, WALL_MIN, WALL_MAX)
    f = _number(floor, FLOOR_MIN, FLOOR_MAX)
    n = int(_number(columns, COLUMNS_MIN, COLUMNS_MAX))
    r = 0.5 * d
    side = DIAMOND_PART * (2.0 * math.pi * r / n) / math.sqrt(2.0)
    diamond = side * math.sqrt(2.0)
    rows = _rows(h, 0.5 * diamond + ROW_GAP, diamond) if windows else 0
    return {"diameter": d, "height": h, "wall": t, "floor": f, "windows": bool(windows),
            "columns": n, "radius": r, "side": side, "diamond": diamond,
            "step": 0.5 * diamond + ROW_GAP, "rows": rows,
            "size_x": d, "size_y": d, "size_z": h}


def sizes(diameter=75.0, height=100.0, wall=2.4, floor=2.4, windows=True, columns=12) -> dict:
    """The numbers a pen cup is made of, so a dialog can show them.

    `diameter` is how wide the cup is outside and `height` how tall it stands, `wall` is how thick
    that wall is and `floor` how thick the bottom it stands on; `windows` says whether the wall
    carries its band of diamonds and `columns` how many of them stand round the cup. `rows` is how
    many rows of diamonds fit between `BAND` millimetres above the floor and `BAND` under the rim -
    and so `rows * columns` how many windows the cup has - and `size_x`, `size_y` and `size_z` are
    the outside of the cup, which is how it stands on the bed. Every number is clamped to what
    prints and anything but a number is a ValueError.
    """
    s = _sizes(diameter, height, wall, floor, windows, columns)
    return {key: s[key] for key in ("diameter", "height", "wall", "floor", "windows", "columns",
                                    "rows", "size_x", "size_y", "size_z")}


def _cup(s) -> mf.Manifold:
    """The cup on its own: the outside pulled up to the rim with the inside cut out of it.

    The inside is cut from the floor up, so what is left is a wall all round and the floor of
    `floor` millimetres under it, which is the face the cup stands on.
    """
    outer = mf.Manifold.cylinder(s["height"], s["radius"], s["radius"], SEGMENTS)
    bore = mf.Manifold.cylinder(s["height"] - s["floor"] + OVERCUT,
                                s["radius"] - s["wall"], s["radius"] - s["wall"], SEGMENTS)
    return outer - bore.translate((0.0, 0.0, s["floor"]))


def _cutter(s) -> mf.Manifold:
    """One diamond cutter facing -y, centred on the origin: a square of side `side` turned 45.

    The cube is `CUT_PAD` millimetres longer along y than the wall is thick, so it reaches past both
    faces of the wall for every thickness and cuts a window right through; centred on the origin and
    turned 45 degrees about y it is a diamond in the x-z plane, whose upper sides lean at 45 degrees
    and need no support.
    """
    return (mf.Manifold.cube([s["side"], s["wall"] + CUT_PAD, s["side"]])
            .translate((-0.5 * s["side"], -0.5 * (s["wall"] + CUT_PAD), -0.5 * s["side"]))
            .rotate((0.0, 45.0, 0.0)))


def _cutters(s) -> list:
    """Every diamond of the band, placed round the cup: the row of `columns` at every row height.

    A cutter stands against the wall at the front, its long axis across the middle of the wall, and
    is turned about the axis of the cup to its column; every second row is turned half a column
    further, so the diamonds interleave and wall is left between two rows instead of one hole
    straight through the wall.
    """
    base = _cutter(s)
    cut = []
    for row in range(s["rows"]):
        off = 180.0 / s["columns"] if row % 2 else 0.0
        z = BAND + 0.5 * s["diamond"] + row * s["step"]
        for k in range(s["columns"]):
            cut.append(base.translate((0.0, 0.5 * s["wall"] - s["radius"], z))
                       .rotate((0.0, 0.0, 360.0 * k / s["columns"] + off)))
    return cut


def _pen_cup(s) -> mf.Manifold:
    """The cup with its band of windows cut through the wall, or a whole wall without them."""
    cup = _cup(s)
    if s["rows"]:
        cup = cup - mf.Manifold.batch_boolean(_cutters(s), mf.OpType.Add)
    return cup


def pen_cup(diameter=75.0, height=100.0, wall=2.4, floor=2.4, windows=True,
            columns=12) -> mf.Manifold:
    """One printable pen cup: a round cup for pens with a band of diamond windows in its wall.

    The cup is `diameter` millimetres across and `height` millimetres high, with a wall of `wall`
    millimetres and a floor of `floor` millimetres. With `windows` its wall carries `columns`
    diamonds round in rows that interleave, every one of them as wide as 60 percent of a column and
    leaning at 45 degrees, so the band prints open without support.

    The result is one closed volume on `z = 0`, centred in x and y and `height` millimetres high,
    that stands on its floor: nothing above the bed points down by more than 45 degrees, so nothing
    needs support. Every number is clamped to what prints and anything but a number is a ValueError.
    """
    return _pen_cup(_sizes(diameter, height, wall, floor, windows, columns))
