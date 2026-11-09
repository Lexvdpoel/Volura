"""Standoffs: the computational core for a printable spacer that holds a part off a surface.

No Qt in this file. It gives the standoff as one closed volume, so it can be tested on its own
and reused in the model and the export.

A **standoff** is the spacer between a board and the wall under it: a PCB, a panel or a cover
that has to lie flat without lying flat. It stands on the bed `height` millimetres high and is
centred on the origin, so it prints without support and lands where the other printed parts
land.

The **shape** is what the spacer looks like from above: `round` is a cylinder of `outer`
millimetres, `hex` a hexagon of `outer` millimetres across the flats, which stands still on a
tilted bed and grips in a slot.

The **mount** is how the screw is held. `through` cuts a clearance hole all the way, so a screw
goes through the standoff into what lies above it; `tap` cuts a pilot hole of 0.85 of the
nominal diameter, into which a self-tapping screw cuts its own thread and stays; `insert` cuts
a hole for a heat-set insert from the top, `INSERTS` wide and as deep as the insert is long,
with the clearance hole under it the rest of the way, so a bolt that comes through the board
reaches out of the bottom. `clearance` is added to the clearance holes, because a printed hole
comes out smaller than drawn; the pilot hole is left alone, because the screw has to find
plastic to bite in.

`count` puts that many standoffs in a row along X, `outer + 4` millimetres from centre to
centre, as one part: the four corners of a PCB print in one go and are broken apart after it.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .geom import circle_points
from .i18n import tr
from . import threads

#: the lowest spacer that keeps anything apart and the tallest that is worth printing
HEIGHT_MIN, HEIGHT_MAX = 2.0, 100.0
#: the thinnest standoff that still takes a screw and the widest that is a spacer, not a column
OUTER_MIN, OUTER_MAX = 3.0, 40.0
#: the fewest and the most standoffs in one row: a PCB has four corners, a large panel more
COUNT_MIN, COUNT_MAX = 1, 12
#: the largest clearance on a diameter, above which a standoff no longer holds its screw
CLEARANCE_MAX = 1.0
#: mm of plastic between the widest hole and the outside: less than that breaks off
WALL_MIN = 0.8
#: mm of standoff that is left above the hole of an insert, so the insert has room
INSERT_ROOM = 1.0
#: an insert hole goes this many nominal diameters deep, plus INSERT_EXTRA millimetres
INSERT_DEPTH, INSERT_EXTRA = 1.5, 2.0
#: a self-tapping pilot hole is this fraction of the nominal diameter
TAP_FACTOR = 0.85
#: mm of bed between two standoffs of a row, so they break apart without cutting them
ROW_ROOM = 4.0
#: the straight pieces a round body or a round hole is approximated with
SEGMENTS = 64
#: mm past the bottom face and over the top, so a cut leaves no skin behind
MARGIN = 0.5

STANDOFF_SHAPES = ("round", "hex")
STANDOFF_KINDS = ("through", "tap", "insert")

#: size: the hole a heat-set insert is pressed into, in mm
INSERTS = {"M2": 3.2, "M2.5": 3.6, "M3": 4.0, "M4": 5.6, "M5": 6.4}


def _outline(shape, outer):
    """The body from above: a circle of `outer` or a hexagon of `outer` across the flats.

    A corner of the hexagon stands at 0 degrees, so its flats face the y and it is `outer`
    millimetres between the parallel sides, the way a hex nut is measured.
    """
    if shape == "hex":
        return mf.CrossSection([circle_points(0.5 * outer / math.cos(math.pi / 6.0), 6)])
    return mf.CrossSection([circle_points(0.5 * outer, SEGMENTS)])


def standoff(height=10.0, outer=6.0, shape="round", bore="M3", kind="through",
             clearance=threads.DEFAULT_HOLE_CLEARANCE, count=1) -> mf.Manifold:
    """A printable spacer that holds a PCB or a panel off the surface under it.

    The standoff stands on `z = 0`, centred on the origin, `height` millimetres high and
    `outer` millimetres wide, of which `shape` is one of `STANDOFF_SHAPES`: "round" a cylinder
    or "hex" a hexagon across the flats.

    `kind` is the mount, one of `STANDOFF_KINDS`: "through" cuts a clearance hole of `bore` +
    `clearance` through the standoff, for a screw that goes into what lies above it; "tap" cuts
    a pilot hole of 0.85 of the nominal diameter, into which a self-tapping screw cuts its own
    thread; "insert" cuts a hole for a heat-set insert of `bore` from the top, `INSERTS[bore]`
    wide and 1.5 nominal diameters + 2 mm deep, with that clearance hole under it the rest of
    the way. `count` stands that many of them in a row along X, `outer` + 4 mm from centre to
    centre, as one part whose row is centred on the origin too.
    """
    h, o = float(height), float(outer)
    if h < HEIGHT_MIN or h > HEIGHT_MAX:
        raise ValueError(tr("the height of a standoff must be {lo} to {hi} mm")
                         .format(lo=f"{HEIGHT_MIN:g}", hi=f"{HEIGHT_MAX:g}"))
    if o < OUTER_MIN or o > OUTER_MAX:
        raise ValueError(tr("the outer size of a standoff must be {lo} to {hi} mm")
                         .format(lo=f"{OUTER_MIN:g}", hi=f"{OUTER_MAX:g}"))
    if shape not in STANDOFF_SHAPES:
        raise ValueError(tr("unknown standoff shape {shape}").format(shape=shape))
    if kind not in STANDOFF_KINDS:
        raise ValueError(tr("unknown standoff mount {kind}").format(kind=kind))
    if bore not in threads.SCREW_HOLES:
        raise ValueError(tr("unknown screw size {size}").format(size=bore))
    if kind == "insert" and bore not in INSERTS:
        raise ValueError(tr("no heat-set insert in {size}").format(size=bore))
    n = int(count)
    if n < COUNT_MIN or n > COUNT_MAX:
        raise ValueError(tr("the number of standoffs must be {lo} to {hi}")
                         .format(lo=COUNT_MIN, hi=COUNT_MAX))
    c = float(clearance)
    if c < 0.0 or c > CLEARANCE_MAX:
        raise ValueError(tr("The clearance must be between 0 and 1 mm."))

    # the mount: the hole the screw goes through, and the wider hole of an insert on top of it
    hole_d, nominal = threads.SCREW_HOLES[bore][0], float(bore[1:])
    if kind == "tap":
        r_hole, r_top = 0.5 * (TAP_FACTOR * nominal), 0.0
    elif kind == "insert":
        r_hole, r_top = 0.5 * (hole_d + c), 0.5 * INSERTS[bore]
    else:
        r_hole = r_top = 0.5 * (hole_d + c)
    # a hexagon is measured across the flats, so 0.5 * outer is its inscribed radius, which is
    # the plastic a hole has to leave room in for both shapes
    if 0.5 * o - max(r_hole, r_top) < WALL_MIN:
        raise ValueError(tr("The hole does not fit in this standoff."))

    body = mf.Manifold.extrude(_outline(shape, o), h)
    # the screw goes through the standoff, so the hole is plain all the way out at the top
    body = body - mf.Manifold.cylinder(h + 2.0 * MARGIN, r_hole, r_hole,
                                       SEGMENTS).translate((0.0, 0.0, -MARGIN))
    if kind == "insert":
        # the insert is pressed in from the top, so its hole is the wider one and stops as deep
        # below the top face as the insert is long
        depth = min(h - INSERT_ROOM, INSERT_DEPTH * nominal + INSERT_EXTRA)
        body = body - mf.Manifold.cylinder(depth + MARGIN, r_top, r_top,
                                           SEGMENTS).translate((0.0, 0.0, h - depth))
    if n == 1:
        return body
    # a row: the corners of a PCB print in one go, and 4 mm of bed makes them break apart
    spacing = o + ROW_ROOM
    return mf.Manifold.batch_boolean(
        [body.translate(((i - 0.5 * (n - 1)) * spacing, 0.0, 0.0)) for i in range(n)],
        mf.OpType.Add)
