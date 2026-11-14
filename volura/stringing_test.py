"""Stringing tests: the computational core of a row of thin towers to travel between.

No Qt in this file. It gives the test as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **stringing test** is the test print that says how well a printer keeps its filament where it
belongs: `towers` thin towers of `diameter` millimetres, standing in a row `distance` millimetres
apart between the centres of two neighbouring ones. Nothing connects them, so on every layer the
nozzle has to travel through open air from one to the next - and every travel move is a chance to
leave a hair of plastic behind. Retraction that is too weak or a nozzle that is too hot strings
the row together like a cobweb; a good setting leaves the towers clean.

A tower is a cylinder of `diameter` millimetres that rises `height` millimetres above the bed.
With a pointed top the last `CONE` part of it is a cone that narrows to `TIP` millimetres, so the
nozzle travels to a point instead of to a flat top and the hairs are easy to see and to break off;
a cone that steep prints without support. The towers stand on a strip of `STRIP` millimetres that
keeps `EDGE` millimetres of plastic beside the row, so no tower stands against its edge.

Print it once and look at the air between the towers: the row that came out without a single
thread between two of them was printed with the retraction and the temperature that belong
together. The row is clamped down when it would grow past `BED` millimetres, so it always fits the
bed. The result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import enclosure
from .i18n import tr

#: the fewest and the most towers in the row
TOWERS_MIN, TOWERS_MAX = 2, 4
#: mm between the centres of two neighbouring towers
DISTANCE_MIN, DISTANCE_MAX = 10.0, 100.0
#: mm the lowest and the highest a tower stands
HEIGHT_MIN, HEIGHT_MAX = 10.0, 80.0
#: mm the thinnest and the thickest a tower is
DIAMETER_MIN, DIAMETER_MAX = 3.0, 12.0
#: mm thick the strip under the towers is
STRIP = 0.6
#: the part of a tower that a pointed top takes
CONE = 0.4
#: mm wide the top of a pointed tower is
TIP = 0.6
#: mm of strip that lies beside the row of towers, and the radius of its corners
EDGE = 2.0
#: mm the whole row has to stay within, so it fits the bed
BED = 250.0
#: segments around a tower
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the test, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the numbers of a stringing test must be numbers"))
    return max(lo, min(hi, v))


def _sizes(towers, distance, height, diameter, cone) -> dict:
    """The clamped numbers of the test, the places of its towers and the size it measures.

    `xs[i]` is the middle of tower `i`, measured from the middle of the first one, so the row is
    `(towers - 1) * distance` long between the first and the last centre. The strip under it
    reaches half a diameter past the outside of the first and the last tower and lies `EDGE`
    millimetres beside the row, which together are the length and the depth of the test; its
    height is the height of a tower.

    The row that no bed holds is a row that prints next to the bed, so towers drop off the end of
    it until the whole test fits `BED` millimetres; the towers that stay are the ones the nozzle
    travels between first.
    """
    n = int(_number(towers, TOWERS_MIN, TOWERS_MAX))
    d = _number(distance, DISTANCE_MIN, DISTANCE_MAX)
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    w = _number(diameter, DIAMETER_MIN, DIAMETER_MAX)
    while n > TOWERS_MIN and (n - 1) * d + 2.0 * w > BED:
        n -= 1
    return {"towers": n, "distance": d, "height": h, "diameter": w, "cone": bool(cone),
            "xs": [k * d for k in range(n)], "length": (n - 1) * d + 2.0 * w,
            "depth": w + 2.0 * EDGE, "top": h}


def sizes(towers=2, distance=40.0, height=30.0, diameter=6.0) -> dict:
    """The numbers a stringing test is made of, so a dialog can show them.

    `towers` is how many of them stand in the row and `distance` how far two neighbouring ones
    are apart between their centres; `height` is how high a tower stands and `diameter` how thick
    it is. `towers` drops when the row would grow past the bed, so what stands here is what
    prints. `size` is the outside (length, depth, height) in millimetres. Every number is clamped
    to what prints and anything but a number is a ValueError.
    """
    s = _sizes(towers, distance, height, diameter, True)
    return {"towers": s["towers"], "distance": s["distance"], "height": s["height"],
            "diameter": s["diameter"], "size": (s["length"], s["depth"], s["top"])}


def _strip(s) -> mf.Manifold:
    """The strip the towers stand on, `STRIP` millimetres thick and rounded at the corners.

    It is built centred on the origin and then slid along x to the places of the towers, so it
    reaches half the diameter of a tower past the outside of the first and the last one and
    `EDGE` millimetres of plastic beside the row - which is also what keeps a tower off the edge
    of the strip.
    """
    plate = mf.Manifold.extrude(enclosure.rounded_rect(s["length"], s["depth"], EDGE), STRIP)
    return plate.translate((0.5 * s["length"] - s["diameter"], 0.0, 0.0))


def _towers(s) -> mf.Manifold:
    """A tower on every place of the row, as one solid.

    A tower is a cylinder of `diameter` millimetres from the bed to `height`. With a pointed top
    the last `CONE` part of it is a cone that narrows to `TIP` millimetres, which is steep enough
    to print without support; without one the tower keeps its width up to its flat top.
    """
    r = 0.5 * s["diameter"]
    shaft = s["height"] * (1.0 - CONE) if s["cone"] else s["height"]
    parts = []
    for x in s["xs"]:
        tower = mf.Manifold.cylinder(shaft, r, r, SEGMENTS).translate((x, 0.0, 0.0))
        if s["cone"]:
            tower = tower + mf.Manifold.cylinder(s["height"] - shaft, r, 0.5 * TIP, SEGMENTS) \
                .translate((x, 0.0, shaft))
        parts.append(tower)
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _row(s) -> mf.Manifold:
    """The strip with a tower on every place of the row, as one solid.

    A tower starts on `z = 0` and not on the top of the strip, so the two of them grow into one
    volume instead of only touching each other on a face.
    """
    return mf.Manifold.batch_boolean([_strip(s), _towers(s)], mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The test on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def stringing_test(towers=2, distance=40.0, height=30.0, diameter=6.0,
                   cone=True) -> mf.Manifold:
    """One printable stringing test: thin towers the nozzle travels between.

    The row is `towers` towers of `diameter` millimetres, standing `distance` millimetres apart
    between their centres and `height` millimetres above the bed, on a strip of `STRIP`
    millimetres. With `cone` every tower ends in a point: the top `CONE` part of it narrows to
    `TIP` millimetres, which prints without support and gives the nozzle a point to travel to.
    Nothing connects the towers, so the nozzle crosses open air on every layer and whatever it
    leaves between them is what the test is about.

    The result is one closed volume on `z = 0`, centred in x and y, that prints without support.
    Every number is clamped to what prints and anything but a number is a ValueError; towers are
    dropped from the end of the row when it would grow past `BED` millimetres.
    """
    s = _sizes(towers, distance, height, diameter, cone)
    return _centred(_row(s))
