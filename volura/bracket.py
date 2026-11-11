"""Corner brackets: the computational core of a printed L-shaped angle bracket with screw holes.

No Qt in this file. It gives the bracket as one manifold, so it can be tested on its own and
reused in the model and the export.

A **corner bracket** fastens two parts at a right angle: two flat **legs** of `thickness`
millimetres, `leg_a` and `leg_b` long measured on the outside from the corner and `width` wide,
with `holes` screw holes in each of them. The **inner corner** is rounded to `fillet`
millimetres: a sharp corner bends about one line of plastic, a rounded one about a whole fillet,
which is what makes the bend strong.

The bracket prints **on its side**: the L lies on the bed and the bracket stands `width`
millimetres high, so the layer lines run round the bend instead of along it, which is the way a
print takes a load without peeling apart. Leg A runs along +X and leg B along +Y, the outside
corner is at the origin and the bracket reaches from `z = 0` to `z = width`.

The holes go through the **thickness** of a leg, so they are horizontal in the print, and each one
is a teardrop: its top closes on a roof of 45 degrees instead of lying flat over the bed. In use
the two outside faces lie flat against the wall and the shelf, so the screws are driven in from
the inside of the L: `countersink` sinks a flat head into the inner face of a leg until it is
flush with it; without it the hole is straight and the head stands proud. The holes are centred
across the width and stand in the free part of a leg, where a whole head fits past the fillet and
past a gusset and clear of the free end. A screw whose head does not fit across the width becomes
the largest one that does, so a hole can never cut a leg in two.

The **gussets** are a triangular rib in the inner corner at both ends of the width, `thickness`
thick and reaching 40 % of the shorter leg along both inner faces. The lower one stands on the
bed; the underside of the upper one rises at 45 degrees from both inner faces, so it hangs over
the empty corner no further than that and prints without support.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import geom, threads
from .i18n import tr

#: mm the shortest and the longest leg, measured on the outside from the corner
LEG_MIN, LEG_MAX = 15.0, 150.0
#: mm across the bend, which is how high the bracket stands on the bed
WIDTH_MIN, WIDTH_MAX = 8.0, 100.0
#: mm thick each leg is: thin enough to bend into place, thick enough for a screw
THICK_MIN, THICK_MAX = 2.0, 12.0
#: the largest inner fillet, as a multiple of the thickness
FILLET_MAX = 3.0
#: how many holes there may be in a leg
HOLES_MIN, HOLES_MAX = 0, 3
#: the screws the holes are made for
SCREWS = ("M3", "M4", "M5", "M6")
#: how far of the shorter leg a gusset reaches along the inner faces
GUSSET_PART = 0.4
#: mm of plastic that always stays between a hole and the fillet or a gusset
ROOM = 0.5
#: mm past a rib the wedge that shapes its underside reaches, so the cut leaves no skin behind
MARGIN = 1.0
#: the straight pieces a fillet or a round hole is approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the bracket, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a corner bracket must be numbers"))
    return max(lo, min(hi, v))


def _count(value):
    """How many holes a leg gets: a whole number of them, 0 to 3."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the number of holes in a corner bracket must be a whole number"))
    return max(HOLES_MIN, min(HOLES_MAX, int(v)))


def _head_of(size, countersink) -> float:
    """The screw head that sits on the hole of `size`, as a printed hole needs it."""
    _hole_d, head_d, _head_h, cs_head_d = threads.SCREW_HOLES[size]
    return (cs_head_d if countersink else head_d) + threads.DEFAULT_HOLE_CLEARANCE


def _fits(size, width, countersink) -> bool:
    """Whether the hole of `size` fits across a leg that is `width` millimetres wide.

    The screw head needs room on the inner face and the teardrop roof, which reaches further
    from the axis than the hole, needs room to the top; both with `ROOM` of leg left beside them.
    A hole that does not fit would cut the leg in two.
    """
    r = 0.5 * (threads.SCREW_HOLES[size][0] + threads.DEFAULT_HOLE_CLEARANCE)
    return (_head_of(size, countersink) + 2.0 * ROOM <= width
            and r * math.sqrt(2.0) + ROOM <= 0.5 * width)


def _hole_places(leg, t, ahead, head, n) -> list:
    """The places of the holes along a leg, from the outside corner.

    The free part of a leg begins `ahead` past the corner, where the fillet and a gusset are
    behind you, and ends a screw head before the free end; `n` holes are spread evenly over it.
    The head sits on the inner face, right where that corner plastic begins, so a whole head has
    to fit between it and the first hole. Two heads need `head` millimetres next to each other, so
    a leg that is too short for all of them gets fewer, and one that is too short for a single
    hole gets none.
    """
    lo, hi = t + ahead + ROOM + 0.5 * head, leg - 0.5 * head
    k = min(n, 1 + int((hi - lo) // head)) if hi >= lo else 0
    if k <= 0:
        return []
    if k == 1:
        return [0.5 * (lo + hi)]
    return [lo + i * (hi - lo) / (k - 1) for i in range(k)]


def _sizes(leg_a, leg_b, width, thickness, holes, screw, countersink, gussets, fillet) -> dict:
    """The clamped numbers and the places of the features, in the frame the bracket prints in.

    In that frame the bracket lies on its side: leg A runs along +X and leg B along +Y, the
    outside corner is at the origin and the bracket stands from `z = 0` to `z = width`. A hole in
    leg A starts on the inner face at `y = thickness` and runs along -Y at `x = holes_a[i]`, and
    one in leg B does the same along -X at `y = holes_b[i]`, both centred on `z = width / 2`;
    `hole_d` is the clearance hole as it prints and `head` the screw head that sits on the inner
    face. The gusset lies along both inner faces and reaches `gusset` millimetres from the corner,
    0 when there is none, and is `rib` millimetres across the width.
    """
    a = _number(leg_a, LEG_MIN, LEG_MAX)
    b = _number(leg_b, LEG_MIN, LEG_MAX)
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    t = _number(thickness, THICK_MIN, THICK_MAX)
    n = _count(holes)
    size = str(screw)
    if size not in SCREWS:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    sink, ribs = bool(countersink), bool(gussets)
    # a fillet that is wider than the leg which is left over would eat its free end
    r = min(_number(fillet, 0.0, FILLET_MAX * t), a - t, b - t)
    # a screw that does not fit across the width becomes the next smaller one, and a leg that is
    # too narrow for the smallest of them stays whole
    while size != SCREWS[0] and not _fits(size, w, sink):
        size = SCREWS[SCREWS.index(size) - 1]
    hole_d = threads.SCREW_HOLES[size][0] + threads.DEFAULT_HOLE_CLEARANCE
    head = _head_of(size, sink)
    # a rib lies along both inner faces, so it too has to stay on the legs, and both ribs have
    # to fit in the width with the bend between them
    g = max(0.0, min(GUSSET_PART * min(a, b), a - t, b - t)) if ribs else 0.0
    ahead = max(r, g)
    free = n if _fits(size, w, sink) else 0
    return {"leg_a": a, "leg_b": b, "width": w, "thickness": t, "holes": n, "screw": size,
            "countersink": sink, "gussets": ribs, "fillet": r, "gusset": g, "hole_d": hole_d,
            "head": head, "rib": min(t, 0.5 * w),
            "holes_a": _hole_places(a, t, ahead, head, free),
            "holes_b": _hole_places(b, t, ahead, head, free)}


def sizes(leg_a=40.0, leg_b=40.0, width=20.0, thickness=4.0, holes=2, screw="M4",
          countersink=True, gussets=True, fillet=4.0) -> dict:
    """The numbers a corner bracket is made of, so a dialog can show them.

    `leg_a` and `leg_b` are the two legs long, measured on the outside from the corner, `width`
    is the bracket wide and `thickness` how thick each leg is. `holes` are the screw holes per
    leg, for a `screw` of `SCREWS`, `countersink` whether a flat head sinks into the inner face of
    the leg and `gussets` whether the inner corner is ribbed at both ends of the width; `fillet`
    is the radius of that corner. `hole_d` is the diameter of the clearance hole as it prints,
    `holes_a` and `holes_b` are the places of the holes along each leg, from the outside corner,
    and `gusset` is how far a rib reaches along the inner faces. A `screw` whose head does not fit
    across the width becomes the largest one that does. Every number is clamped to what prints;
    anything but a number is a ValueError.
    """
    b = _sizes(leg_a, leg_b, width, thickness, holes, screw, countersink, gussets, fillet)
    return {key: b[key] for key in ("leg_a", "leg_b", "width", "thickness", "holes", "screw",
                                    "countersink", "gussets", "fillet", "gusset", "hole_d",
                                    "holes_a", "holes_b")}


def _profile(b) -> mf.CrossSection:
    """The L of the two legs, with the inner corner rounded, in the plane of the bed.

    It runs counter-clockwise from the outside corner over the free end of leg A, along the inner
    faces to the free end of leg B and back down the outside of that leg. The fillet is the arc
    that is tangent to both inner faces and bulges towards the corner, so it adds plastic to the
    bend instead of taking it away.
    """
    a, leg, t, r = b["leg_a"], b["leg_b"], b["thickness"], b["fillet"]
    pts = [(0.0, 0.0), (a, 0.0), (a, t)]
    if r > 0.0:
        pts.append((t + r, t))
        steps = max(2, SEGMENTS // 4)
        ang = np.linspace(-0.5 * math.pi, -math.pi, steps + 1)[1:-1]
        pts += [(t + r + r * math.cos(x), t + r + r * math.sin(x)) for x in ang]
        pts.append((t, t + r))
    else:
        pts.append((t, t))
    return mf.CrossSection([pts + [(t, leg), (0.0, leg)]])


def _sideways(cs, axis, a0, a1) -> mf.Manifold:
    """The prism over the cross-section `cs`, extruded along `axis` from `a0` to `a1`.

    `cs` is written in the plane you look at that axis from: (x, z) for "y" and (y, z) for "x",
    so the 45 degree underside of a gusset is drawn the way it is measured. That turn lays the
    extrusion along +x but along -y, so a prism along y starts its length at `a1` and grows back
    to `a0`.
    """
    rot = (geom.rotation_matrix((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0) if axis == "x"
           else geom.rotation_matrix((1.0, 0.0, 0.0), 0.5 * math.pi))
    at = (a0, 0.0, 0.0) if axis == "x" else (0.0, a1, 0.0)
    return mf.Manifold.extrude(cs, a1 - a0).transform(geom.affine(rot, at))


def _gussets(b) -> list:
    """The two ribs in the inner corner, at both ends of the width; empty when there are none.

    Both stand over the triangle that the corner makes with the two inner faces and are as thick
    as the leg, or half the width when both of them have to fit in it together. The lower one
    lies on the bed with its top facing up, so it prints as it is. The upper one would hang over
    the empty corner with a flat ceiling, so its underside is cut back on two planes of 45
    degrees that rise from the inner faces: what is left is a rib along each of the two legs,
    rooted in the face it supports.
    """
    g, t, q, w = b["gusset"], b["thickness"], b["rib"], b["width"]
    if g <= 0.0:
        return []
    corner = mf.CrossSection([[(t, t), (t + g, t), (t, t + g)]])
    low = mf.Manifold.extrude(corner, q)
    up = mf.Manifold.extrude(corner, q).translate((0.0, 0.0, w - q))

    def wedge(axis, along):
        """The room a 45 degree underside needs between the inner face and the rib's top."""
        lo, hi = w - q - MARGIN, w + MARGIN     # past the rib, so the cut leaves no skin behind
        loop = [(along - MARGIN, lo), (along + g, lo), (along + g, min(hi, w - q + g))]
        if g > q + MARGIN:
            loop.append((along + q + MARGIN, hi))
        return _sideways(mf.CrossSection([loop]), axis, t - MARGIN, t + g + MARGIN)

    both = mf.Manifold.batch_boolean([wedge("y", t), wedge("x", t)], mf.OpType.Intersect)
    return [low, up - both]


def bracket(leg_a=40.0, leg_b=40.0, width=20.0, thickness=4.0, holes=2, screw="M4",
            countersink=True, gussets=True, fillet=4.0) -> mf.Manifold:
    """One printable corner bracket, lying on its side on `z = 0`.

    The two legs are `leg_a` and `leg_b` millimetres long, measured on the outside from the
    corner at the origin, `width` millimetres wide and `thickness` millimetres thick; leg A runs
    along +X and leg B along +Y, and the bracket stands from `z = 0` to `z = width`. The inner
    corner is rounded to `fillet` millimetres, which is what makes the bend strong.

    Each leg gets `holes` holes for a `screw` of `SCREWS` through its thickness, centred across
    the width and spread over the free part of the leg. The screws are driven in from the inside
    of the L, so `countersink` sinks a flat head into the inner face of the leg, flush with it,
    and without it the hole is straight. The holes are teardrops, so their top prints without
    support. `gussets` ribs the inner corner at both ends of the width. A leg that is too short
    for the holes asked gets fewer of them.

    Every number is clamped to what prints, a screw whose head does not fit across the width
    becomes a smaller one, neither a hole nor its head cuts into a gusset or into the fillet, and
    nothing hangs over steeper than 45 degrees; anything but a number, or a screw that is not in
    `SCREWS`, is a ValueError.
    """
    b = _sizes(leg_a, leg_b, width, thickness, holes, screw, countersink, gussets, fillet)
    m = mf.Manifold.extrude(_profile(b), b["width"])
    for rib in _gussets(b):
        m = m + rib
    kind = "countersink" if b["countersink"] else "through"
    z = 0.5 * b["width"]
    t = b["thickness"]
    for x in b["holes_a"]:
        m = threads.screw_hole(m, (x, t, z), (0.0, -1.0, 0.0), t, b["screw"], kind,
                               segments=SEGMENTS, teardrop=True)
    for y in b["holes_b"]:
        m = threads.screw_hole(m, (t, y, z), (-1.0, 0.0, 0.0), t, b["screw"], kind,
                               segments=SEGMENTS, teardrop=True)
    return m
