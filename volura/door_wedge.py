"""Door wedges: the computational core of a wedge you push under a door to hold it open.

No Qt in this file: the wedge is one side profile extruded straight up, so it can be tested on its
own and reused in the model and the export.

A **door wedge** is a block that is thick at one end and thin at the other: `length` millimetres
long, `height` millimetres high at the thick end and `tip` millimetres at the thin one, and `width`
millimetres wide. The thin end goes under the door and the door leans on the **slope**; the harder
the door pushes, the harder the wedge is driven against the floor, so a wedge holds a door without a
strap, a magnet or a screw in the floor.

**Grip ridges** are small saw teeth on the slope, from `RIDGE_START` millimetres in at either end:
each one runs along the slope for `RIDGE_BASE` millimetres and stands `RIDGE_HEIGHT` millimetres
above it, where its steep side faces the thin end. The wedge can still slide in, but when the door
tries to push it out the steep sides dig into the floor and it stays where it was.

A **hanging hole** of `HOLE_D` millimetres through the thick end hangs the wedge on a hook when it
is not holding a door. It is only cut when the wedge is high enough to carry it, because a hole that
breaks through the slope holds nothing.

The wedge prints **lying on its side**: the profile is one closed loop in the x-y plane extruded
`width` millimetres high, so every wall is vertical and nothing overhangs. The two corners of the
thick end round with `ROUND` millimetres, so nothing is sharp enough to cut a hand.

Every number is clamped to what prints and anything but a number is refused. The result is one
closed volume on `z = 0`, centred in x and y, `width` millimetres high.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the shortest and the longest wedge: long enough to reach under a door, short enough for a bag
LENGTH_MIN, LENGTH_MAX = 60.0, 200.0
#: mm the least and the most the wedge is high at its thick end
HEIGHT_MIN, HEIGHT_MAX = 10.0, 50.0
#: mm the least and the most the wedge is high at its thin end
TIP_MIN, TIP_MAX = 1.0, 5.0
#: mm the narrowest and the widest wedge: a wide one stands firmer and does not tip over
WIDTH_MIN, WIDTH_MAX = 20.0, 60.0
#: mm the thin end always stays under the thick one by this much, so a slope is left to bite with
TIP_GAP = 5.0
#: mm the first ridge stands from either end, mm between two ridges and mm of the base of one
RIDGE_START, RIDGE_PITCH, RIDGE_BASE = 15.0, 6.0, 4.0
#: mm a ridge stands above the slope, and mm it reaches into the wedge below it so it grows out of it
RIDGE_HEIGHT, RIDGE_IN = 1.2, 0.2
#: mm the diameter of the hanging hole and mm of its centre from the back of the wedge
HOLE_D, HOLE_X = 8.0, 10.0
#: mm the least a wedge has to be high at its thick end for the hole to fit inside the slope
HOLE_MIN = 16.0
#: mm the two corners of the thick end round with
ROUND = 3.0
#: the straight pieces the rounded corners and the hole are approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the wedge, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a door wedge must be numbers"))
    return max(lo, min(hi, v))


def _sizes(length, height, tip, width, ridges, hole) -> dict:
    """The clamped numbers of the wedge, the angle of its slope and the size of its profile.

    The profile is `length` millimetres long and `height` millimetres high at the thick end, and the
    wedge is `width` millimetres thick, which is how it stands on the bed. The thin end stays
    `TIP_GAP` millimetres under the thick one, so there is always a slope left, and the hole is only
    there when the wedge is high enough to carry it.
    """
    L = _number(length, LENGTH_MIN, LENGTH_MAX)
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _number(tip, TIP_MIN, TIP_MAX)
    b = _number(width, WIDTH_MIN, WIDTH_MAX)
    t = min(t, h - TIP_GAP)
    return {"length": L, "height": h, "tip": t, "width": b, "ridges": bool(ridges),
            "hole": bool(hole) and h >= HOLE_MIN,
            "angle": math.degrees(math.atan((h - t) / L)),
            "size_x": L, "size_y": h, "size_z": b}


def sizes(length=120.0, height=25.0, tip=2.0, width=35.0, ridges=True, hole=True) -> dict:
    """The numbers a door wedge is made of, so a dialog can show them.

    `length` is how long the wedge is, `height` how high it stands at the thick end and `tip` how
    high it is at the thin one; `width` is how wide the wedge is, `ridges` whether the slope carries
    grip ridges and `hole` whether the thick end is drilled for a hook. `angle` is the angle of the
    slope in degrees and `size_x`, `size_y` and `size_z` are the outside of the wedge, which is how
    it stands on the bed. Every number is clamped to what prints and anything but a number is a
    ValueError.
    """
    s = _sizes(length, height, tip, width, ridges, hole)
    return {key: s[key] for key in ("length", "height", "tip", "width", "ridges", "hole", "angle",
                                    "size_x", "size_y", "size_z")}


def _array(loop) -> np.ndarray:
    """A loop of points as the array a `CrossSection` wants."""
    return np.ascontiguousarray(np.asarray(loop, dtype=float))


def _rect(x0, x1, y0, y1):
    """One rectangle of the profile, as a loop that runs counter-clockwise."""
    return _array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _slope(s, x) -> float:
    """How high the wedge stands at `x`: the slope runs from `height` at the back down to `tip`."""
    return s["height"] + (s["tip"] - s["height"]) * x / s["length"]


def _ridge(s, x) -> np.ndarray:
    """One grip ridge: a saw tooth whose base lies on the slope from `x` to `x + RIDGE_BASE`.

    Its tip stands `RIDGE_HEIGHT` millimetres above the slope at `x + RIDGE_BASE`, so the steep side
    faces the thin end, and its base lies `RIDGE_IN` millimetres under the slope, so the ridge grows
    out of the wedge instead of only touching it.
    """
    return _array([(x, _slope(s, x) - RIDGE_IN),
                   (x + RIDGE_BASE, _slope(s, x + RIDGE_BASE) - RIDGE_IN),
                   (x + RIDGE_BASE, _slope(s, x + RIDGE_BASE) + RIDGE_HEIGHT)])


def _profile(s) -> mf.CrossSection:
    """The side profile of the wedge: the slope with its ridges and the hole in the thick end.

    The x axis runs along the wedge with the thick end at `x = 0` and the floor that lies on the bed
    on `y = 0`. The two corners of the thick end round with `ROUND` millimetres - shrinking the wedge
    and growing it back with round joins - and that takes an arc off every corner the wedge has, so
    the wedge is put back from the point where it is `3 * ROUND` high again: the rounding is then
    only where it was asked for, and the thin end keeps the height it was given. The ridges come
    after the rounding, so their tips stay sharp, and the hole is cut last.
    """
    wedge = mf.CrossSection([_array([(0.0, 0.0), (s["length"], 0.0),
                                     (s["length"], s["tip"]), (0.0, s["height"])])])
    rounded = (wedge.offset(-ROUND, mf.JoinType.Miter)
               .offset(ROUND, mf.JoinType.Round, circular_segments=SEGMENTS))
    thick = s["length"] * (s["height"] - 3.0 * ROUND) / (s["height"] - s["tip"])
    out = rounded + (wedge - mf.CrossSection([_rect(-1.0, thick, -1.0, s["height"] + 1.0)]))
    if s["ridges"]:
        x = RIDGE_START
        while x <= s["length"] - RIDGE_START:
            out = out + mf.CrossSection([_ridge(s, x)])
            x += RIDGE_PITCH
    if s["hole"]:
        out = out - mf.CrossSection.circle(0.5 * HOLE_D, SEGMENTS).translate(
            [HOLE_X, 0.5 * s["height"]])
    return out


def _centred(m) -> mf.Manifold:
    """The wedge on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def door_wedge(length=120.0, height=25.0, tip=2.0, width=35.0, ridges=True,
               hole=True) -> mf.Manifold:
    """One printable door wedge: a block you push under a door to hold it open.

    The wedge is `length` millimetres long, `height` millimetres high at the thick end and `tip`
    millimetres at the thin one, which is the end that goes under the door. Grip ridges stand on the
    slope so the wedge bites into the floor when the door tries to push it out, and a hole through
    the thick end hangs it on a hook when it is not working.

    The result is one closed volume on `z = 0`, centred in x and y and `width` millimetres high,
    that prints lying on its side: the profile is extruded straight up, so every face is vertical or
    horizontal and nothing needs support. Every number is clamped to what prints and anything but a
    number is a ValueError.
    """
    s = _sizes(length, height, tip, width, ridges, hole)
    return _centred(mf.Manifold.extrude(_profile(s), s["width"]))
