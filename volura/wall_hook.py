"""Wall hooks: the computational core of a J-shaped hook that is screwed to a wall.

No Qt in this file. It gives the hook as one manifold, so it can be tested on its own and reused
in the model and the export.

A **wall hook** is a J that you screw to a wall or to the side of a cupboard, for a coat, a bag,
headphones or a cable: a **back plate** of `plate` millimetres that lies against the wall, an
**arm** of `thickness` millimetres that leaves the lower end of that plate horizontally and
reaches `reach` millimetres out from the wall, and a **tip** that rises `rise` millimetres at the
end of the arm and stops what hangs on it from sliding off. The hook is that outline, seen from
the side, extruded `width` millimetres.

Both bends are rounded: the inside of the hook between arm and tip with a quarter of the reach and
at most 8 millimetres, the corner where the arm leaves the plate with a radius of the thickness
itself. A sharp inside corner bends about one line of plastic and is where a hook snaps; a rounded
one bends about a whole fillet. The top of the tip is a full half-round, so nothing cuts into what
hangs on it and the tip prints without support.

The hook prints **on its side**: the J lies on the bed and the hook stands `width` millimetres
high, so the layer lines run along the J instead of under it, which is the way a print takes a
load without snapping off along a layer. Every outside edge on top is chamfered half a millimetre.

The plate gets `holes` screw holes through its thickness, perpendicular to the wall and spread
evenly over the part of the plate that rises above the arm, one of them right in its middle.
Because the hook lies on its side those holes run horizontally in the print, so each one is a
teardrop: its top closes on a roof of 45 degrees instead of lying flat over the bed. The screws are
driven in from the front, away from the wall, so `countersink` sinks a flat head into that front
face until it is flush with it; without it the hole is straight and the head stands proud. A hole
never reaches into the bend of the arm, and a screw whose head does not fit across the width
becomes the largest one that does, so a hole can never cut the plate in two.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import geom, threads
from .i18n import tr

#: mm from the wall to the outside of the tip, and mm the tip rises above the arm
REACH_MIN, REACH_MAX = 10.0, 80.0
RISE_MIN, RISE_MAX = 5.0, 50.0
#: mm across the hook, which is how high it stands on the bed
WIDTH_MIN, WIDTH_MAX = 8.0, 40.0
#: mm thick the plate, the arm and the tip are
THICK_MIN, THICK_MAX = 3.0, 10.0
#: mm tall the plate may be at most
PLATE_MAX = 150.0
#: mm of plate a screw hole needs beside its head
PLATE_ROOM = 4.0
#: how many holes there may be in the plate
HOLES_MIN, HOLES_MAX = 1, 2
#: the screws the holes are made for
SCREWS = ("M3", "M4", "M5")
#: mm of straight arm that always stays between the two bends
ARM_MIN = 2.0
#: mm the bend between arm and tip is at most
BEND_MAX = 8.0
#: mm of plastic that always stays between a hole and the bend or the top of the plate
ROOM = 0.5
#: mm of chamfer on every outside edge on top
CHAMFER = 0.5
#: mm past an edge and past the top that a chamfer cut reaches, so it does not stop in the face
#: it is cutting
OVER = 0.2
#: mm past both ends of an edge that a chamfer cut reaches, so two of them overlap instead of
#: standing exactly against each other, which a boolean pays with a sliver
EDGE = 0.02
#: the straight pieces a bend or a round hole is approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the hook, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a wall hook must be numbers"))
    return max(lo, min(hi, v))


def _count(value):
    """How many holes the plate gets: a whole number of them, 1 or 2."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the number of holes in a wall hook must be a whole number"))
    return max(HOLES_MIN, min(HOLES_MAX, int(v)))


def _head_of(size, countersink) -> float:
    """The screw head that sits on the hole of `size`, as a printed hole needs it."""
    _hole_d, head_d, _head_h, cs_head_d = threads.SCREW_HOLES[size]
    return (cs_head_d if countersink else head_d) + threads.DEFAULT_HOLE_CLEARANCE


def _fits(size, width, countersink) -> bool:
    """Whether the hole of `size` fits across a plate that is `width` millimetres wide.

    The screw head needs room on the front face and the teardrop roof, which reaches further from
    the axis than the hole, needs room to the top; both with `ROOM` of plate left beside them. A
    hole whose head does not fit would cut the plate in two.
    """
    r = 0.5 * (threads.SCREW_HOLES[size][0] + threads.DEFAULT_HOLE_CLEARANCE)
    return (_head_of(size, countersink) + 2.0 * ROOM <= width
            and r * math.sqrt(2.0) + ROOM <= 0.5 * width)


#: mm of plate that always stays between a screw head and the end of the plate, or the bend
HEAD_EDGE = 3.0


def _hole_places(plate, root, head, n) -> list:
    """The places of the holes along the plate, measured from the bed.

    The free part of the plate begins `root` up, where the bend of the arm is behind you, and ends
    at the top of the plate; `n` holes are spread evenly over it, and a single one stands right in
    its middle. A head sits flat on the front face, with `HEAD_EDGE` millimetres of plate left
    between it and the bend and between it and the top, so the plate does not split there.
    """
    lo, hi = root + HEAD_EDGE + 0.5 * head, plate - HEAD_EDGE - 0.5 * head
    if n <= 1:
        return [0.5 * (lo + hi)]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def _sizes(reach, rise, width, thickness, plate, screw, countersink, holes) -> dict:
    """The clamped numbers and the places of the features, in the frame the hook prints in.

    In that frame the hook lies on its side: the wall is the plane `x = 0`, the plate runs up along
    +Y to `plate`, the arm runs out along +X to `reach` and the hook stands from `z = 0` to
    `z = width`. The underside of the arm and the bottom of the plate both lie on `y = 0`, the
    front face of the plate is `x = thickness` and the inside of the tip is `x = reach -
    thickness`. `bend` is the radius of the bend between arm and tip and `root` how far the bend of
    the arm reaches up the plate; `holes_y` are the places of the holes along the plate, `hole_d`
    is the clearance hole as it prints and `head` the screw head that sits on the front face.
    """
    t = _number(thickness, THICK_MIN, THICK_MAX)
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    n = _count(holes)
    size = str(screw)
    if size not in SCREWS:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    sink = bool(countersink)
    # a screw whose head does not fit across the width becomes the next smaller one
    while size != SCREWS[0] and not _fits(size, w, sink):
        size = SCREWS[SCREWS.index(size) - 1]
    hole_d = threads.SCREW_HOLES[size][0] + threads.DEFAULT_HOLE_CLEARANCE
    head = _head_of(size, sink)
    # the reach carries both bends and a piece of straight arm between them, so a thick hook is
    # never shorter than the room those need
    a = _number(reach, REACH_MIN, REACH_MAX)
    a = max(a, 3.0 * t + ARM_MIN + min(0.25 * a, BEND_MAX))
    # the bend between arm and tip is a quarter of the reach, but never wider than the hook
    r = min(0.25 * a, BEND_MAX, a - 3.0 * t - ARM_MIN)
    # a tip is never lower than its own bend and its own round top
    rise_ = max(_number(rise, RISE_MIN, RISE_MAX), 1.5 * t + r)
    root = 2.0 * t
    # the plate is tall enough for its holes next to the bend, and never bigger than a wall
    p = min(PLATE_MAX, max(root + 2.0 * HEAD_EDGE + n * head + (n - 1) * PLATE_ROOM,
                           _number(plate, 0.0, PLATE_MAX)))
    return {"reach": a, "rise": rise_, "width": w, "thickness": t, "plate": p, "screw": size,
            "countersink": sink, "holes": n, "hole_d": hole_d, "head": head, "bend": r,
            "root": root, "height": w, "holes_y": _hole_places(p, root, head, n)}


def sizes(reach=30.0, rise=15.0, width=15.0, thickness=5.0, plate=50.0, screw="M4",
          countersink=True, holes=2) -> dict:
    """The numbers a wall hook is made of, so a dialog can show them.

    `reach` is how far the hook stands out from the wall, `rise` how high its tip comes above the
    arm, `width` how wide the hook is and `thickness` how thick plate, arm and tip are. `plate` is
    how tall the plate that lies against the wall is, `holes` how many screws go through it, for a
    `screw` of `SCREWS`, and `countersink` whether a flat head sinks into the front face. `height`
    is what the hook occupies on the bed, which is its width because it prints on its side, and
    `holes_y` are the places of the holes along the plate, measured from the bed.

    A plate that is too short for its holes is made taller, a reach too short for the thickness is
    made longer, a tip too low for its own bend is made higher and a screw whose head does not fit
    across the width becomes the largest one that does. Every number is clamped to what prints;
    anything but a number, or a screw that is not in `SCREWS`, is a ValueError.
    """
    h = _sizes(reach, rise, width, thickness, plate, screw, countersink, holes)
    return {key: h[key] for key in ("reach", "rise", "width", "thickness", "plate", "screw",
                                    "countersink", "holes", "height", "holes_y", "hole_d",
                                    "head", "bend", "root")}


def _arc(cx, cy, r, a0, a1, steps) -> list:
    """The inside points of a circle of `r` around (`cx`, `cy`), from `a0` to `a1` radians.

    The two end points are left out, because the outline names them itself and a point that is
    there twice would make an edge of no length at the end of the arc.
    """
    ang = np.linspace(a0, a1, steps + 1)[1:-1]
    return [(cx + r * math.cos(x), cy + r * math.sin(x)) for x in ang]


def _outline(h) -> list:
    """The J of plate, arm and tip, with both inside bends rounded, in the plane of the bed.

    It runs counter-clockwise from the bottom of the wall face along the underside of the arm to
    the tip, over the half-round top of the tip, down its inside and through the bend back to the
    top of the arm, and up the front face of the plate to its top. A bend is the arc that is
    tangent to both faces it joins and bulges towards the corner, so it adds plastic to the bend
    instead of taking it away.
    """
    a, rise, t, r = h["reach"], h["rise"], h["thickness"], h["bend"]
    pts = [(0.0, 0.0), (a, 0.0), (a, rise - 0.5 * t)]
    pts += _arc(a - 0.5 * t, rise - 0.5 * t, 0.5 * t, 0.0, math.pi, SEGMENTS // 2)
    pts.append((a - t, rise - 0.5 * t))
    pts.append((a - t, t + r))
    pts += _arc(a - t - r, t + r, r, 0.0, -0.5 * math.pi, SEGMENTS // 4)
    pts.append((a - t - r, t))
    pts.append((2.0 * t, t))
    pts += _arc(2.0 * t, 2.0 * t, t, -0.5 * math.pi, -math.pi, SEGMENTS // 4)
    pts.append((t, 2.0 * t))
    pts += [(t, h["plate"]), (0.0, h["plate"])]
    # a tip that is exactly as high as its own bend needs leaves one of these points twice
    return [p for i, p in enumerate(pts) if math.dist(p, pts[i - 1]) > 1e-9]


def _wedge(p, q, c, height) -> mf.Manifold:
    """The 45 degree cut that lies along the edge from `p` to `q` of the outline.

    The wedge is `c` wide against the top of the part and nothing `c` below it, so what is left is
    an edge of 45 degrees. It reaches `OVER` past the edge and over the top and `EDGE` past both of
    its ends, so the cut does not stop in the middle of a face it is cutting and two wedges that
    meet on a corner overlap instead of leaving a sliver between them. Where the wedges of two
    edges cross, the cut follows the bisector of the corner, which mitres a corner and takes a
    bend round in the short pieces it is made of.
    """
    d = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    length = float(math.hypot(d[0], d[1]))
    t = d / length
    n = np.array([t[1], -t[0]])            # out of the part, for an outline that runs CCW
    tri = mf.CrossSection([[(OVER, height - c - OVER), (OVER, height + OVER),
                            (-c - OVER, height + OVER)]])
    rot = np.column_stack([[n[0], n[1], 0.0], [0.0, 0.0, 1.0], [-t[0], -t[1], 0.0]])
    at = (q[0] + EDGE * t[0], q[1] + EDGE * t[1], 0.0)
    return mf.Manifold.extrude(tri, length + 2.0 * EDGE).transform(geom.affine(rot, at))


def _body(h) -> mf.Manifold:
    """The prism over the J, standing `width` high, with every outside edge on top chamfered."""
    pts = _outline(h)
    c = min(CHAMFER, 0.5 * h["width"])
    cut = mf.Manifold.batch_boolean([_wedge(p, q, c, h["width"])
                                     for p, q in zip(pts, pts[1:] + pts[:1])], mf.OpType.Add)
    return mf.Manifold.extrude(mf.CrossSection([pts]), h["width"]) - cut


def wall_hook(reach=30.0, rise=15.0, width=15.0, thickness=5.0, plate=50.0, screw="M4",
              countersink=True, holes=2) -> mf.Manifold:
    """One printable wall hook, lying on its side on `z = 0`.

    The plate is `plate` millimetres tall and `thickness` thick and lies against the wall on
    `x = 0`; the arm leaves its lower end along +X and reaches `reach` millimetres out from the
    wall, and the tip rises `rise` millimetres at the end of it. The hook is `width` millimetres
    wide, which is how high it stands on the bed, and it is that J extruded, so the layer lines
    run along the hook instead of under it. Both inside bends are rounded, the one between arm and
    tip to a quarter of the reach and at most 8 millimetres and the one at the plate to the
    thickness, and the top of the tip is a full half-round.

    The plate gets `holes` teardrop holes for a `screw` of `SCREWS` through its thickness, spread
    over the part of the plate above the arm; `countersink` sinks a flat head into the front face,
    flush with it, because the screws are driven in from away from the wall. Every outside edge on
    top is chamfered `CHAMFER` millimetres.

    A plate too short for its holes is made taller, a reach too short for the thickness is made
    longer, a tip too low for its own bend is made higher and a screw whose head does not fit
    across the width becomes a smaller one; nothing hangs over steeper than 45 degrees. Anything
    but a number, or a screw that is not in `SCREWS`, is a ValueError.
    """
    h = _sizes(reach, rise, width, thickness, plate, screw, countersink, holes)
    m = _body(h)
    kind = "countersink" if h["countersink"] else "through"
    z = 0.5 * h["width"]
    for y in h["holes_y"]:
        m = threads.screw_hole(m, (h["thickness"], y, z), (-1.0, 0.0, 0.0), h["thickness"],
                               h["screw"], kind, segments=SEGMENTS, teardrop=True)
    return m
