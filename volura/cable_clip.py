"""Cable clips: the computational core for a printable clip that holds a cable in place.

No Qt in this file. It gives the clip as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **cable clip** is the clip that keeps a cable, a hose or a bundle of both where it belongs: it
is screwed to a wall or under a desk and the cable is pushed into it. The clip is printed lying on
its side, so the ring is a wall that stands on the bed and needs no support, and the cable runs
across the bed instead of away from it.

The **ring** is what holds the cable: `cable_d` millimetres inside and `wall` millimetres thick,
open at the top with a mouth of `opening` times the cable diameter. The mouth is smaller than the
cable, so the cable has to be pushed past it and the ring springs back behind it: that is what
keeps the cable in.

The **foot** is what is screwed down: a flat tab of `base` millimetres thick that runs out on one
side of the ring, `cable_d + 2 wall + 8` millimetres long, with a screw hole of `bore` through its
thickness, in the middle of the tab and in the middle of the width. The hole leaves `TAB_ROOM`
millimetres of tab beside it on both sides of the width, or the clip breaks off at the screw.

`count` puts that many clips in a row along X, four millimetres apart, as one part: a row of
clips prints in one go and is broken apart after it.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .geom import arc_polyline
from .i18n import tr
from . import threads

#: the thinnest cable that is worth a clip and the thickest a clip still holds
CABLE_MIN, CABLE_MAX = 2.0, 40.0
#: the narrowest clip that takes a screw and the widest that is still a clip, not a bracket
WIDTH_MIN, WIDTH_MAX = 4.0, 40.0
#: the thinnest wall that survives snapping a cable in and the thickest that is still a clip
WALL_MIN, WALL_MAX = 0.8, 5.0
#: the mouth as a fraction of the cable: less than that does not close, more does not hold
OPENING_MIN, OPENING_MAX = 0.3, 0.95
#: the thinnest foot that holds a screw and the thickest that is still a foot
BASE_MIN, BASE_MAX = 1.5, 8.0
#: the fewest and the most clips in one row: a run along a desk holds more than one
COUNT_MIN, COUNT_MAX = 1, 12
#: mm of tab beside the screw hole, so the screw does not tear out of the side
TAB_ROOM = 1.0
#: mm of foot beyond the ring, so there is room for the screw around the ring
TAB_EXTRA = 8.0
#: mm of bed between two clips of a row, so they break apart without cutting them
ROW_ROOM = 4.0
#: the straight pieces a ring is approximated with
SEGMENTS = 64
#: mm past the bottom face and over the top, so a cut leaves no skin behind
MARGIN = 0.5


def _arc(radius, start, sweep):
    """`sweep` degrees of a circle of `radius`, from `start` degrees, as points of that ring."""
    a, s = math.radians(start), math.radians(sweep)
    n = max(2, int(round(SEGMENTS * abs(sweep) / 360.0)))
    return arc_polyline((0.0, 0.0), radius, (math.cos(a), math.sin(a)),
                        (-math.sin(a), math.cos(a)), s, n)


def _ring(cable_d, wall, opening) -> mf.CrossSection:
    """The ring from above: a band of `wall` round a bore of `cable_d`, open at the top.

    The mouth is `opening * cable_d` wide between its tips, which stand on the bore, so the
    opening is narrowest exactly where the cable passes it. The band is one contour: over the
    bottom along the bore, out at the left tip, back over the bottom along the outside and in at
    the right tip.
    """
    r_in, r_out = 0.5 * cable_d, 0.5 * cable_d + wall
    # the tip of the mouth stands at x = half the mouth on the bore and on the outside
    a_in, a_out = (math.degrees(math.acos(0.5 * opening * cable_d / r)) for r in (r_in, r_out))
    inner = _arc(r_in, a_in, -(180.0 + 2.0 * a_in))
    outer = _arc(r_out, 180.0 - a_out, 180.0 + 2.0 * a_out)
    return mf.CrossSection([np.concatenate([inner, outer])])


def _foot(cable_d, base, length) -> mf.CrossSection:
    """The foot from above: a tab of `base` thick from the bore out to `length`."""
    y = 0.5 * float(base)
    return mf.CrossSection([np.array([[0.5 * cable_d, -y], [length, -y],
                                      [length, y], [0.5 * cable_d, y]], dtype=float)])


def cable_clip(cable_d=6.0, width=8.0, wall=1.6, opening=0.8, base=3.0, bore="M3",
               count=1) -> mf.Manifold:
    """A printable clip that holds a cable or a bundle against a wall or a desk.

    The clip stands on `z = 0` and is centred on the origin in plan, `width` millimetres high,
    printed lying on its side so the ring needs no support. It is a ring of `cable_d` millimetres
    inside and `wall` millimetres thick, open at the top with a mouth of `opening` times the cable
    diameter, so the cable snaps in and is held. The ring stands on a flat foot of `base`
    millimetres thick that runs out on one side and takes a screw of `bore` through its
    thickness, in the middle of the tab and of the width.

    `count` stands that many clips in a row along X, four millimetres apart, as one part whose row
    is centred on the origin too.
    """
    d = float(cable_d)
    if d < CABLE_MIN or d > CABLE_MAX:
        raise ValueError(tr("the cable diameter of a clip must be {lo} to {hi} mm")
                         .format(lo=f"{CABLE_MIN:g}", hi=f"{CABLE_MAX:g}"))
    w = float(width)
    if w < WIDTH_MIN or w > WIDTH_MAX:
        raise ValueError(tr("the width of a cable clip must be {lo} to {hi} mm")
                         .format(lo=f"{WIDTH_MIN:g}", hi=f"{WIDTH_MAX:g}"))
    t = float(wall)
    if t < WALL_MIN or t > WALL_MAX:
        raise ValueError(tr("the wall of a cable clip must be {lo} to {hi} mm")
                         .format(lo=f"{WALL_MIN:g}", hi=f"{WALL_MAX:g}"))
    o = float(opening)
    if o < OPENING_MIN or o > OPENING_MAX:
        raise ValueError(tr("the opening of a cable clip must be {lo} to {hi} of the cable "
                            "diameter").format(lo=f"{OPENING_MIN:g}", hi=f"{OPENING_MAX:g}"))
    b = float(base)
    if b < BASE_MIN or b > BASE_MAX:
        raise ValueError(tr("the foot of a cable clip must be {lo} to {hi} mm")
                         .format(lo=f"{BASE_MIN:g}", hi=f"{BASE_MAX:g}"))
    if bore not in threads.SCREW_HOLES:
        raise ValueError(tr("unknown screw size {size}").format(size=bore))
    n = int(count)
    if n < COUNT_MIN or n > COUNT_MAX:
        raise ValueError(tr("the number of cable clips must be {lo} to {hi}")
                         .format(lo=COUNT_MIN, hi=COUNT_MAX))

    # the screw goes through the foot, so the hole is plain all the way through its thickness
    hole_d = threads.SCREW_HOLES[bore][0]
    if w - hole_d < 2.0 * TAB_ROOM:
        raise ValueError(tr("The screw hole does not fit in this clip."))

    length = d + 2.0 * t + TAB_EXTRA
    body = mf.Manifold.extrude(_ring(d, t, o), w) + mf.Manifold.extrude(_foot(d, b, length), w)
    x_hole = 0.5 * (0.5 * d + length)
    reach = b + 2.0 * MARGIN
    rod = mf.Manifold.cylinder(reach, 0.5 * hole_d, 0.5 * hole_d, SEGMENTS) \
        .translate((0.0, 0.0, -0.5 * reach)).rotate((90.0, 0.0, 0.0))
    body = body - rod.translate((x_hole, 0.0, 0.5 * w))
    # the clip is printed lying on its side, so it is put in the middle of the bed as it stands
    bb = np.asarray(body.bounding_box(), dtype=float)
    body = body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), 0.0))
    if n == 1:
        return body
    # a row: the clips print in one go, and 4 mm of bed makes them break apart
    spacing = float(bb[3] - bb[0]) + ROW_ROOM
    return mf.Manifold.batch_boolean(
        [body.translate(((i - 0.5 * (n - 1)) * spacing, 0.0, 0.0)) for i in range(n)],
        mf.OpType.Add)
