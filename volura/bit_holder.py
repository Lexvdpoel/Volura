"""Bit holders: the computational core of a block that stands screwdriver bits upright in rows.

No Qt in this file. It gives the holder as one manifold, so it can be tested on its own and reused
in the model and the export.

A **bit holder** is a block that lies on the bed with the bits sticking out of its top, `cols` by
`rows` of them, so a rack of driver bits stands on the bench or in a drawer and the one you want is
the one you see. Every bit is a standard 1/4 inch hex shank, `BIT` millimetres across the flats,
and it stands in a **socket** of that same hexagon with `fit` millimetres of play on every flat and
`depth` millimetres deep. The hexagon lies with two of its flats parallel to the rows, so the walls
between the sockets are straight along X and a socket that leans keeps its walls within 45 degrees.

The sockets are `pitch` millimetres apart in both directions and never closer than their own
diagonal plus `WALL_MIN` millimetres, which is all the wall that stays between two of them; the
block keeps `WALL` millimetres of wall around the outer sockets and `FLOOR` millimetres of floor
under the lowest point of a socket. Its corners are rounded and its top edge is chamfered.

With `angle` the rows lean backwards, so the bits lean towards you. Every socket turns about the X
axis over its own mouth and the top of the block rises with the tilt, so every mouth stays in that
top, every socket keeps its full `depth` along its own axis and every floor keeps its `FLOOR`
millimetres under its lowest point. The top stays flat and the only step in it is the ledge; the
block grows at the back by the distance the tilt carries the sockets, so the wall behind them stays
`WALL` millimetres. 30 degrees is the most the rows lean, which is also what keeps a tilted
hexagonal wall under 45 degrees, so nothing of the holder needs support.

The **ledge** is a flat `LEDGE` millimetres deep along the front of the block at its full width,
`LEDGE_DROP` millimetre below the top of the front row: a strip of paper or a printed label lies
there, in front of the bits it belongs to. Without it the front wall is as thick as the others.
"""
from __future__ import annotations

import math

import manifold3d as mf

from . import enclosure, geom
from .i18n import tr

#: how many bits there are in a row, and how many rows there are
COLS_MIN, COLS_MAX = 1, 12
ROWS_MIN, ROWS_MAX = 1, 4
#: mm deep a socket is, and mm of play between a bit and its socket
DEPTH_MIN, DEPTH_MAX = 6.0, 20.0
FIT_MIN, FIT_MAX = 0.0, 0.4
#: mm between the centres of two sockets next to each other
PITCH_MIN, PITCH_MAX = 8.5, 15.0
#: degrees the rows lean backwards, 0 standing straight up
ANGLE_MIN, ANGLE_MAX = 0.0, 30.0
#: mm across the flats of a standard 1/4 inch hex bit
BIT = 6.35
#: mm of wall around the outer sockets, mm of floor under a socket and mm of wall between two of
#: them, which is what the pitch is never allowed to come down to
WALL, FLOOR, WALL_MIN = 3.0, 2.0, 1.2
#: mm deep the ledge for a label is, and mm below the top of the front row it lies
LEDGE, LEDGE_DROP = 8.0, 1.0
#: mm the corners of the block are rounded to, and mm of chamfer on its top edge
CORNER, CHAMFER = 2.0, 0.5
#: mm of chamfer at the mouth of a socket, so a bit finds its way in
MOUTH = 0.6
#: mm past a cut, so it leaves no skin behind
MARGIN = 0.5
#: mm of slab at either end of a chamfer, of which the hull of a chamfer is made
SLIVER = 0.2


def _number(value, lo, hi):
    """A number of the holder, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a bit holder must be numbers"))
    return max(lo, min(hi, v))


def _count(value, lo, hi) -> int:
    """How many sockets there are in a row, or how many rows: a whole number of them."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the number of sockets of a bit holder must be a whole number"))
    return max(lo, min(hi, int(v)))


def _sizes(cols, rows, depth, fit, pitch, angle, label) -> dict:
    """The clamped numbers and the places of the sockets, in the frame the holder prints in.

    In that frame the block lies on the bed and is centred on the origin in plan: it is `width` by
    `length` and `height` high. A socket has its mouth in the top of the block, on the axis through
    (`x`, `y`) of `centres` that crosses the top at `z = height`; that axis leans back over X by
    `tilt`, so a bit in the socket leans towards the front. Along that axis `s` counts millimetres
    downwards from the mouth, and a socket runs from `s = 0` to `s = depth`.

    The tilt carries every socket `drift` millimetres towards the back and makes its mouth anything
    but level: the front edge of a mouth is `drop` millimetres below its back edge. That is why the
    block stands `drop` higher, why it grows `drift` at the back and why a cut along a tilted axis
    has to rise `rise` millimetres above the top to clear it everywhere.
    """
    n, m = _count(cols, COLS_MIN, COLS_MAX), _count(rows, ROWS_MIN, ROWS_MAX)
    d = _number(depth, DEPTH_MIN, DEPTH_MAX)
    f = _number(fit, FIT_MIN, FIT_MAX)
    a = _number(angle, ANGLE_MIN, ANGLE_MAX)
    flats = BIT + 2.0 * f
    # the sockets are never so close that the wall between two of them is thinner than WALL_MIN;
    # a hexagon is 2 / sqrt(3) wider across its corners than across its flats
    p = max(_number(pitch, PITCH_MIN, PITCH_MAX), 2.0 * flats / math.sqrt(3.0) + WALL_MIN)
    tilt = math.radians(a)
    drift, drop = d * math.sin(tilt), 0.5 * flats * math.sin(tilt)
    rise = (0.5 * flats + MARGIN) * math.tan(tilt) + MARGIN
    front = LEDGE if label else WALL
    width = (n - 1) * p + 2.0 * (WALL + flats / math.sqrt(3.0))
    length = (m - 1) * p + 2.0 * WALL + flats + front + drift
    # the floor under the lowest point of a socket: the mouth plane is that much lower at its front
    height = FLOOR + d * math.cos(tilt) + drop
    y_front = -0.5 * length + front + 0.5 * flats
    return {"cols": n, "rows": m, "depth": d, "fit": f, "pitch": p, "angle": a,
            "label": bool(label), "flats": flats, "width": width, "length": length,
            "height": height, "tilt": tilt, "drift": drift, "drop": drop, "rise": rise,
            "front": front,
            "centres": [((i - 0.5 * (n - 1)) * p, y_front + j * p)
                        for j in range(m) for i in range(n)]}


def sizes(cols=6, rows=2, depth=10.0, fit=0.15, pitch=10.0, angle=0.0, label=True) -> dict:
    """The numbers a bit holder is made of, so a dialog can show them.

    `cols` and `rows` are how many bits stand in a row and how many rows there are, `depth` how
    deep the sockets are and `fit` the play between a bit and its socket. `pitch` is the distance
    between the centres of two sockets next to each other and is raised when it would leave less
    than `WALL_MIN` millimetres of wall between them; `angle` is how far the rows lean backwards
    and `label` whether the front carries the ledge for a label. `flats` is the socket across the
    flats as it prints and `width`, `length` and `height` are the block. Every number is clamped to
    what prints; anything but a number is a ValueError.
    """
    s = _sizes(cols, rows, depth, fit, pitch, angle, label)
    return {key: s[key] for key in ("cols", "rows", "depth", "fit", "pitch", "angle", "label",
                                    "flats", "width", "length", "height")}


def _hex(across) -> mf.CrossSection:
    """A hexagon `across` millimetres across the flats, with two flats parallel to X.

    Those two are the walls a bit lies against: they look along Y, so the walls that run along X
    are straight and the sockets line up in rows.
    """
    r = across / math.sqrt(3.0)                     # the corners stand that much out
    return mf.CrossSection([[(r * math.cos(k * math.pi / 3.0), r * math.sin(k * math.pi / 3.0))
                             for k in range(6)]])


def _box(length, width, radius, height, z=0.0) -> mf.Manifold:
    """A box `length` by `width` with corners of `radius`, `height` high, standing at `z` and
    centred on the origin in plan."""
    return mf.Manifold.extrude(enclosure.rounded_rect(length, width, radius),
                               height).translate((0.0, 0.0, z))


def _chamfered(length, width, radius, height, z=0.0) -> mf.Manifold:
    """The same box with its top edge chamfered `CHAMFER` millimetres.

    The chamfer is the hull of a thin slab of the one outline and a thin slab of the same outline
    inset by `CHAMFER`: inset a rounded rectangle and its corners get a `CHAMFER` tighter, so the
    hull leans 45 degrees along the whole edge, also round the corner.
    """
    c = min(CHAMFER, 0.5 * height, radius)
    parts = [_box(length, width, radius, height - c, z)]
    parts.append(mf.Manifold.batch_hull([_box(length, width, radius, SLIVER, z + height - c),
                                         _box(length - 2.0 * c, width - 2.0 * c, radius - c,
                                              SLIVER, z + height - SLIVER)]))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _block(s) -> mf.Manifold:
    """The block on its own: the footprint with rounded corners and a chamfered top edge, with the
    ledge for a label cut out of the front when there is one.

    The ledge is the front wall, lowered: the cut reaches across the whole front at the full width
    of the block, so it runs into the rounded corners too and a label has all of it to lie on.
    """
    solid = _chamfered(s["width"], s["length"], CORNER, s["height"])
    if s["label"]:
        cut = mf.Manifold.cube([s["width"] + 2.0 * MARGIN, LEDGE + MARGIN, LEDGE_DROP + MARGIN])
        solid = solid - cut.translate((-0.5 * s["width"] - MARGIN, -0.5 * s["length"] - MARGIN,
                                       s["height"] - LEDGE_DROP))
    return solid


def _socket(s, x, y, cs, s_top, s_bottom) -> mf.Manifold:
    """The prism over the cross-section `cs` along the axis of the socket whose mouth is at
    (`x`, `y`) in the top of the block, from `s_top` to `s_bottom` millimetres down that axis.

    A negative `s` is above the top of the block, which is where a cut has to come from to leave
    nothing behind. The axis leans back over X by `tilt`, so a bit in the socket leans towards the
    front; that turn lays the cross-section out the way it is measured, with two flats along X.
    """
    tilt = s["tilt"]
    rot = geom.rotation_matrix((1.0, 0.0, 0.0), tilt)
    at = (x, y + s_bottom * math.sin(tilt), s["height"] - s_bottom * math.cos(tilt))
    return mf.Manifold.extrude(cs, s_bottom - s_top).transform(geom.affine(rot, at))


def _mouth(s, x, y) -> mf.Manifold:
    """The lead-in at the mouth of a socket: a chamfer of `MOUTH` millimetres at 45 degrees, so a
    bit finds its way in instead of standing on the edge of the hexagon.

    The wide end stands upright, because it is the top of the block that a bit passes first: drawn
    on that top, the chamfer stays `MOUTH` millimetres wide all round also when the socket leans.
    A hull is made of the two ends it runs between, so the wide end is a `SLIVER` wider than the
    chamfer to make the hull pass `MOUTH` wide over the top of the block.
    """
    wide = mf.Manifold.extrude(_hex(s["flats"] + 2.0 * (MOUTH + SLIVER)), SLIVER).translate(
        (x, y, s["height"]))
    deep = _socket(s, x, y, _hex(s["flats"]), MOUTH - SLIVER, MOUTH)
    return mf.Manifold.batch_hull([wide, deep])


def bit_holder(cols=6, rows=2, depth=10.0, fit=0.15, pitch=10.0, angle=0.0,
               label=True) -> mf.Manifold:
    """One printable bit holder, lying on `z = 0` and centred on the origin in plan.

    The block is `cols` by `rows` sockets big with `WALL` millimetres of wall around them, and
    every socket is the hexagon of a 1/4 inch bit with `fit` millimetres of play on every flat,
    `depth` millimetres deep, open at the top and chamfered at its mouth. The sockets are `pitch`
    millimetres apart, the rows lean back by `angle` degrees so the bits lean towards you, and the
    front carries a ledge of `LEDGE` millimetres for a label when `label`.

    Every number is clamped to what prints, the pitch is raised when it would leave too little wall
    between two sockets, the block grows with the tilt so every socket keeps its depth and its
    floor, and nothing hangs over steeper than 45 degrees; anything but a number is a ValueError.
    """
    s = _sizes(cols, rows, depth, fit, pitch, angle, label)
    cutters = []
    for x, y in s["centres"]:
        cutters.append(_socket(s, x, y, _hex(s["flats"]), -s["rise"], s["depth"]))
        cutters.append(_mouth(s, x, y))
    return _block(s) - mf.Manifold.batch_boolean(cutters, mf.OpType.Add)
