"""Print-in-place hinges: the computational core for a hinge that prints as one piece.

No Qt in this file. It gives the hinge as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **print-in-place hinge** is a hinge that comes off the printer already assembled: the leaves,
the knuckles and the pin print in one go and the hinge turns as soon as it is off the bed. It is
used on a printed box, a tool case or a cover that has to open, and there is nothing to screw or
to glue.

The **leaves** are what is screwed to the two halves: two flat plates of `width` millimetres and
`thickness` millimetres, `length` millimetres long, that lie flat on the bed beside each other with
`LEAF_GAP` millimetres of bed between them, so they do not print together.

The **knuckles** are what joins them: `knuckles` barrels around one pin axis that runs along the
length of the hinge, parallel to the bed and one barrel radius above it, so the barrels stand on
the bed and the leaves lie beside them. The barrels alternate: the first, the third and every other
one belong to one leaf, the rest to the other, and neighbouring barrels leave `clearance`
millimetres of room between them so the hinge turns. The number of knuckles is odd, so both ends of
the hinge carry the pin and the pin cannot walk out.

The **pin** is what the knuckles turn on: a rod of `pin_d` millimetres along the whole hinge, part
of the leaf that carries it, while the knuckles of the other leaf have a hole of `pin_d + 2
clearance` around it. The inside edge of a leaf is rounded away from the axis by the same
clearance, so a leaf can never pinch the pin. The pin and the hole end in a 45 degree chamfer, so
neither of them needs support.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

#: the shortest hinge that is still one part and the longest that prints in one go
LENGTH_MIN, LENGTH_MAX = 10.0, 300.0
#: the narrowest leaf that is worth a screw and the widest that is still a leaf, per leaf
WIDTH_MIN, WIDTH_MAX = 5.0, 100.0
#: the thinnest leaf that survives being screwed on and the thickest that still turns
THICKNESS_MIN, THICKNESS_MAX = 1.0, 8.0
#: the fewest knuckles that hold a pin and the most that are worth it on one hinge
KNUCKLES_MIN, KNUCKLES_MAX = 3, 11
#: the least play that survives printing and the most that still leaves plastic in the knuckle
CLEARANCE_MIN, CLEARANCE_MAX = 0.1, 1.0
#: the thinnest pin that prints and the thickest that still turns in its hole
PIN_MIN, PIN_MAX = 1.0, 12.0
#: mm of bed between the two leaves, so they print apart instead of as one plate
LEAF_GAP = 0.8
#: mm of knuckle that has to stay on the pin, so a short hinge gets fewer knuckles
KNUCKLE_MIN = 2.0
#: mm of chamfer on the end of the pin and of the hole, which is what makes them print unsupported
CHAMFER = 0.4
#: the straight pieces a pin and a knuckle are approximated with
SEGMENTS = 64
#: mm past a cut face, so a cut leaves no skin behind
MARGIN = 0.5


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so the hinge is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _rod(x0, x1, r0, r1) -> mf.Manifold:
    """A piece along X from `x0` to `x1`, tapering from radius `r0` to radius `r1`, round the axis."""
    return mf.Manifold.cylinder(x1 - x0, r0, r1, SEGMENTS).rotate((0.0, 90.0, 0.0)) \
        .translate((x0, 0.0, 0.0))


def _knuckles(length, clearance, n):
    """The X spans of the knuckles: `n` barrels over `length`, `clearance` millimetres apart."""
    pitch = (length + clearance) / n
    return [(i * pitch, i * pitch + pitch - clearance) for i in range(n)]


def hinge(length=40.0, width=15.0, thickness=2.0, knuckles=5, clearance=0.3,
          pin_d=3.0) -> mf.Manifold:
    """A hinge that prints in one piece and turns as soon as it is off the printer.

    The hinge lies on the bed with both leaves flat beside each other and the pin above them, so it
    prints without support: two plates of `width` by `thickness` millimetres and `length`
    millimetres, joined by `knuckles` barrels around one pin of `pin_d` millimetres. Half of the
    barrels are part of one leaf and carry the pin, the other half have a hole of `pin_d + 2
    clearance` millimetres and turn on it, and neighbouring barrels leave `clearance` millimetres
    between them. The number of knuckles is made odd, so both ends of the hinge carry the pin.

    The result is centred on the origin in plan and stands on `z = 0`; it is one volume of two
    leaves that are not fused, `length` millimetres long, `2 width + LEAF_GAP` millimetres wide and
    one knuckle diameter high.
    """
    ln = _clamp(length, LENGTH_MIN, LENGTH_MAX)
    w = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    t = _clamp(thickness, THICKNESS_MIN, THICKNESS_MAX)
    c = _clamp(clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    rp = 0.5 * _clamp(pin_d, PIN_MIN, PIN_MAX)
    # every knuckle keeps KNUCKLE_MIN millimetres on the pin, so a short hinge gets fewer of them
    room = int((ln + c) // (KNUCKLE_MIN + c))
    n = max(KNUCKLES_MIN, min(int(knuckles), KNUCKLES_MAX, room if room % 2 else room - 1))
    if n % 2 == 0:
        n -= 1                        # odd: the leaf that carries the pin also carries both ends

    rh = rp + c                       # the hole the pin turns in
    ro = rh + t                       # the knuckle: the wall of a leaf around that hole
    ch = min(CHAMFER, 0.5 * t, 0.5 * rp)      # a chamfer never takes more than half the wall
    relief = rh + min(c, 0.5 * t)     # the inside edge of a leaf, clear of the pin and of the hole

    # the pin is one rod over the whole hinge, chamfered on both ends, part of the leaf that carries
    # it; the other leaf gets the same rod as a hole, chamfered outwards so its end prints flat
    pin = (_rod(0.0, ch, rp - ch, rp) + _rod(ch, ln - ch, rp, rp) + _rod(ln - ch, ln, rp, rp - ch))
    hole = (_rod(-MARGIN, 0.0, rh + ch, rh + ch) + _rod(0.0, ch, rh + ch, rh)
            + _rod(ch, ln - ch, rh, rh) + _rod(ln - ch, ln, rh, rh + ch)
            + _rod(ln, ln + MARGIN, rh + ch, rh + ch))

    spans = _knuckles(ln, c, n)
    leaves = []
    for side in (1, -1):
        own = 0 if side > 0 else 1
        plate = mf.Manifold.cube([ln, w, t]).translate(
            (0.0, 0.5 * LEAF_GAP if side > 0 else -0.5 * LEAF_GAP - w, -ro))
        plate = plate - _rod(-MARGIN, ln + MARGIN, relief, relief)
        for i, (x0, x1) in enumerate(spans):
            # a leaf is cut away where the knuckles of the other leaf turn, so the hinge can move
            if i % 2 != own:
                plate = plate - _rod(x0 - MARGIN, x1 + MARGIN, ro + c, ro + c)
        for i, (x0, x1) in enumerate(spans):
            if i % 2 == own:
                plate = plate + _rod(x0, x1, ro, ro)
        leaves.append(plate + pin if side > 0 else plate - hole)
    body = mf.Manifold.batch_boolean(leaves, mf.OpType.Add)

    # the hinge lies on the bed as it prints: centred in plan, with the leaves and the pin on z = 0
    bb = np.asarray(body.bounding_box(), dtype=float)
    return body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
