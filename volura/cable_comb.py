"""Cable combs: the computational core of a flat comb that keeps a bundle of cables in line.

No Qt in this file: the comb is a plate with a round seat and a narrow opening for every cable, so it
can be tested on its own and reused in the model and the export.

A **cable comb** is the flat bar you lay behind a PC or on a desk to keep a bundle of cables side by
side instead of in one knot. Every cable lies in its own **seat**: a round hole of `cable`
millimetres, `0.3` millimetres wider than the cable so the cable fits into it. Over the seat runs an
**opening** that is only `snap` times the cable diameter wide, so the cable has to be pushed through
it and the plastic springs back behind it: that is what holds the bundle in line and still lets a
cable out when you pull it.

The seats stand one every `cable + 3.5` millimetres, so `3.5` millimetres of plate is left between
two of them, and the plate keeps `EDGE` millimetres all round with its corners rounded. The seats sit
low in the plate, half a millimetre under its middle, so the wall above them - the one a cable is
snapped through - is the thinner one and the comb keeps a thick bottom to lie on.

Where an opening leaves the top edge it is widened by `CHAMFER` millimetres on both sides, a lead-in
that catches the cable and guides it into the seat: a cable dropped in from above never rests on a
sharp rim.

The comb prints **lying flat**: the outline is extruded `thickness` millimetres high, so every wall is
vertical, nothing overhangs and nothing needs support. `BED` millimetres is as long a comb as the bed
takes, so the number of cables is clamped to what still fits.

Every number is clamped to what prints and anything but a number is refused. The result is one closed
volume on `z = 0`, centred in x and y, `thickness` millimetres high.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: the fewest and the most cables a comb holds: a single cable is a clip, more than a bundle is not
CABLES_MIN, CABLES_MAX = 2, 16
#: mm the thinnest cable that is worth a seat and the thickest a comb still takes
CABLE_MIN, CABLE_MAX = 3.0, 12.0
#: mm the thinnest plate that holds a cable and the thickest that is still a comb, not a brick
THICKNESS_MIN, THICKNESS_MAX = 3.0, 8.0
#: the opening as a part of the cable: less does not close behind it, more does not hold it
SNAP_MIN, SNAP_MAX = 0.6, 0.9
#: mm of plate between two seats, on top of the cable diameter, and all round the seats
EDGE, GAP = 3.5, 3.5
#: mm the seat is wider than the cable, so the cable goes in without forcing
SEAT_CLEARANCE = 0.3
#: mm the seats stand under the middle of the plate, so the wall above them is the thinner one
SEAT_LIFT = 0.5
#: mm the lead-in by which an opening is widened where it leaves the top edge
CHAMFER = 0.8
#: mm the corners of the plate are rounded with
CORNER = 2.0
#: mm past the top edge that an opening is cut, so it does not stop in the edge it cuts
OVERCUT = 1.0
#: mm the longest comb that still fits on the bed
BED = 250.0
#: the straight pieces a seat is approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the comb, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a cable comb must be numbers"))
    return max(lo, min(hi, v))


def _sizes(cables, cable, thickness, snap) -> dict:
    """The clamped numbers of the comb, the width of an opening and where the seats stand.

    A seat stands `pitch` millimetres from the next one, which is the cable diameter plus the `GAP`
    of plate between two of them, and the plate is `EDGE` millimetres wider than the seats on every
    side. The seats all have their middle on `seat_y`, half a millimetre under the middle of the
    plate, and an opening is `snap * cable` wide.
    """
    n = int(_number(cables, CABLES_MIN, CABLES_MAX))
    d = _number(cable, CABLE_MIN, CABLE_MAX)
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    s = _number(snap, SNAP_MIN, SNAP_MAX)
    pitch = d + GAP
    n = min(n, int((BED - EDGE) // pitch))       # a comb that hangs off the bed cannot be printed
    return {"cables": n, "cable": d, "thickness": t, "snap": s, "opening": s * d, "pitch": pitch,
            "seat_y": EDGE + 0.5 * d + SEAT_LIFT, "size_x": n * pitch + EDGE,
            "size_y": d + 2.0 * EDGE, "size_z": t}


def sizes(cables=6, cable=6.0, thickness=4.0, snap=0.75) -> dict:
    """The numbers a cable comb is made of, so a dialog can show them.

    `cables` is how many the comb holds and `cable` how thick they are, `thickness` how high the
    plate stands and `snap` how wide an opening is as a part of the cable diameter. `opening` is that
    width in millimetres and `pitch` the distance between two seats, while `size_x`, `size_y` and
    `size_z` are the outside of the comb, which is how it stands on the bed. The number of cables is
    clamped down to what fits in `BED` millimetres and every other number to what prints; anything
    but a number is a ValueError.
    """
    s = _sizes(cables, cable, thickness, snap)
    return {key: s[key] for key in ("cables", "cable", "thickness", "snap", "opening", "pitch",
                                    "size_x", "size_y", "size_z")}


def _plate(s) -> mf.CrossSection:
    """The plate from above: `size_x` by `size_y` with its corners rounded, from the origin out.

    The corners are rounded by pulling the rectangle `CORNER` millimetres in with square joins and
    pushing it back out with round ones, which leaves the flat sides where they were and turns every
    corner into an arc of `CORNER` millimetres.
    """
    return (mf.CrossSection.square((s["size_x"], s["size_y"]), True)
            .offset(-CORNER, mf.JoinType.Miter, 2.0)
            .offset(CORNER, mf.JoinType.Round, 2.0)
            .translate((0.5 * s["size_x"], 0.5 * s["size_y"])))


def _cutters(s) -> list:
    """Every cut of the comb from above: the seat of every cable with its opening and lead-in.

    A seat is a circle of `cable + 0.3` millimetres on the middle of its cable. Over it runs the
    opening, `snap * cable` wide, from the middle of the seat past the top edge, so the cut goes
    right through that edge and the seat can be reached from above. Where the opening leaves the edge
    a trapezoid widens it by `CHAMFER` millimetres on both sides, which is the lead-in the cable
    runs down into the seat.
    """
    half = 0.5 * s["opening"]
    y0, y1 = s["size_y"] - CHAMFER, s["size_y"] + 0.01
    cut = []
    for k in range(s["cables"]):
        x = EDGE + 0.5 * s["cable"] + k * s["pitch"]
        cut.append(mf.CrossSection.circle(0.5 * (s["cable"] + SEAT_CLEARANCE), SEGMENTS)
                   .translate((x, s["seat_y"])))
        cut.append(mf.CrossSection.square((s["opening"], s["size_y"] + OVERCUT - s["seat_y"]), True)
                   .translate((x, 0.5 * (s["size_y"] + OVERCUT + s["seat_y"]))))
        cut.append(mf.CrossSection([np.array([[x - half - CHAMFER, y1], [x - half, y0],
                                              [x + half, y0], [x + half + CHAMFER, y1]],
                                             dtype=float)]))
    return cut


def _cable_comb(s) -> mf.Manifold:
    """The plate with every seat, opening and lead-in cut out of it, pulled up to its thickness."""
    plate = _plate(s) - mf.CrossSection.batch_boolean(_cutters(s), mf.OpType.Add)
    return mf.Manifold.extrude(plate, s["thickness"]).translate(
        (-0.5 * s["size_x"], -0.5 * s["size_y"], 0.0))


def cable_comb(cables=6, cable=6.0, thickness=4.0, snap=0.75) -> mf.Manifold:
    """One printable cable comb: a flat bar that keeps `cables` cables of `cable` millimetres in line.

    Every cable lies in its own round seat and is held by the opening above it, `snap` times the
    cable diameter wide, so the cable has to be snapped in and does not fall out again. The seats
    stand `cable + 3.5` millimetres apart, low in a plate of `thickness` millimetres that keeps
    3.5 millimetres of plastic all round and rounds its corners with 2.

    The result is one closed volume on `z = 0`, centred in x and y and `thickness` millimetres high,
    that lies flat on the bed: every wall is vertical and every face horizontal, so nothing overhangs
    and nothing needs support. Every number is clamped to what prints, the number of cables to what
    fits the bed, and anything but a number is a ValueError.
    """
    return _cable_comb(_sizes(cables, cable, thickness, snap))
