"""Phone stands: the computational core of a stand a phone or a tablet leans in.

No Qt in this file. It gives the stand as one manifold, so it can be tested on its own and reused
in the model and the export.

A **phone stand** is the side view of a leaning stand, drawn flat and pulled `width` millimetres
thick: a **base** of `WALL` millimetres that lies on the bed, a **back support** that leaves it at
`angle` degrees and rises `back` millimetres up the lean, and in front of that support a **ledge**
the device stands on with a **lip** that comes `lip` millimetres up at the front of it. The slot
between support and lip is a millimetre wider than the device, so a phone with its case in it
stands there on its foot and leans back against the support. The top of the support is a full
half-round, so the edge of a tablet does not lie against a sharp end, and where the support meets
the base a fillet of 4 millimetres carries the weight into the bed instead of into one line of
plastic.

The stand prints **lying on its side**: the profile lies on the bed and the stand stands `width`
millimetres high, so every wall is pulled straight up from the bed and nothing hangs over but the
short ceiling over the cable slot. The layers run along the profile instead of under it, which is
the way a print takes the weight of a tablet without peeling apart along a layer.

The length of the base is not asked for but **computed**: a tablet is heavy and leans far over its
foot, so the base is made long enough that the centre of a device of `back` × 1.8 millimetres
leaning against the support stays 10 millimetres in front of the back edge of the base. What the
base gains over the back of the support is what keeps the stand on the bed.

The **cable slot** is a gap of 12 millimetres right in the middle of the width, through the ledge,
the lip and the base under them, so a charging cable runs in behind the stand, between the two
rails that are left of the base, and up to the foot of the phone. A stand narrower than 40
millimetres has no room beside such a gap and gets none.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm across the stand, which is how high it stands on the bed
WIDTH_MIN, WIDTH_MAX = 30.0, 200.0
#: degrees the back support leans back from the bed
ANGLE_MIN, ANGLE_MAX = 50.0, 80.0
#: mm the support rises up the lean
BACK_MIN, BACK_MAX = 40.0, 160.0
#: mm thick the device is, its case included
DEVICE_MIN, DEVICE_MAX = 5.0, 16.0
#: mm the lip comes up above the ledge
LIP_MIN, LIP_MAX = 4.0, 20.0
#: mm thick the base, the support and the lip are
WALL = 3.0
#: mm of room between the device and the slot it stands in
SLOT_PLAY = 1.0
#: mm wide the cable slot is
CABLE = 12.0
#: mm of stand that has to stay beside a cable slot, or there is no cable slot
CABLE_MIN_WIDTH = 40.0
#: mm high the device is that decides how long the base is, as a multiple of the support
DEVICE_TALL = 1.8
#: mm of base that always stays behind the centre of the leaning device
BACK_SPARE = 10.0
#: mm of base that stays in front of the lip
FRONT_EDGE = 3.0
#: mm the fillet is that rounds where the support meets the base
FILLET = 4.0
#: mm past an edge that the cut of the cable slot reaches, so it does not stop in the face it cuts
OVER = 1.0
#: the straight pieces a fillet or a rounded top is approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the stand, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a phone stand must be numbers"))
    return max(lo, min(hi, v))


def _sizes(width, angle, back, device, lip, cable) -> dict:
    """The clamped numbers and the places of the features, in the frame the stand prints in.

    In that frame the profile lies on the bed: `x` runs to the front, from the back edge of the
    base at `x = 0`, and `y` is up, so the base runs from `x = 0` to `x = depth` and from `y = 0`
    to `y = WALL`, the front of the support leaves the top of the base at `x = support_x` and
    rises along the lean, and the stand is pulled from `z = 0` to `z = width`. `slot` is how wide
    the gap the device stands in is and `height` how far the top of the support comes up;
    `device_x` is where the centre of the device that decides the length of the base falls.
    """
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    deg = _number(angle, ANGLE_MIN, ANGLE_MAX)
    back_ = _number(back, BACK_MIN, BACK_MAX)
    d = _number(device, DEVICE_MIN, DEVICE_MAX)
    lip_ = _number(lip, LIP_MIN, LIP_MAX)
    a = math.radians(deg)
    sin, cos = math.sin(a), math.cos(a)
    # a gap of CABLE in the middle needs stand beside it on both sides, or it cuts the stand in two
    on = bool(cable) and w >= CABLE_MIN_WIDTH
    # the base begins so far behind the support that a device of DEVICE_TALL * back millimetres
    # keeps its centre BACK_SPARE millimetres in front of the back edge of the base, and the base
    # always reaches under the back of the rounded top, so nothing of the stand leans past it
    tall = DEVICE_TALL * back_
    x0 = max(BACK_SPARE + 0.5 * tall * cos - 0.5 * d * sin,
             back_ * cos + 0.5 * WALL * (1.0 + sin))
    return {"width": w, "angle": deg, "back": back_, "device": d, "lip": lip_, "cable": on,
            "wall": WALL, "slot": d + SLOT_PLAY, "support_x": x0,
            "depth": x0 + (d + SLOT_PLAY + WALL) / sin + FRONT_EDGE,
            "height": WALL + back_ * sin + 0.5 * WALL * (1.0 - cos),
            "device_x": x0 - 0.5 * tall * cos + 0.5 * d * sin}


def sizes(width=70.0, angle=65.0, back=90.0, device=11.0, lip=10.0, cable=True) -> dict:
    """The numbers a phone stand is made of, so a dialog can show them.

    `width` is the stand wide, which is how high it stands on the bed, `angle` is how far the back
    support leans back from the bed and `back` how far it rises up that lean. `device` is the
    thickness of the phone or tablet, its case included, and `lip` how far the lip comes up above
    the ledge the device stands on; `cable` is whether the cable slot is cut. `wall` is the
    thickness of every wall, `slot` the width of the gap the device stands in and `depth` and
    `height` how long the base is and how high the stand comes up, the last three of them following
    from the rest. Every number is clamped to what prints; anything but a number is a ValueError.
    """
    h = _sizes(width, angle, back, device, lip, cable)
    return {key: h[key] for key in ("width", "angle", "back", "device", "lip", "cable", "wall",
                                    "slot", "depth", "height", "support_x", "device_x")}


def _arc(cx, cy, r, a0, a1, steps) -> list:
    """The inside points of a circle of `r` around (`cx`, `cy`), from `a0` to `a1` radians.

    The two end points are left out, because the outline names them itself and a point that is
    there twice would make an edge of no length at the end of the arc.
    """
    ang = np.linspace(a0, a1, steps + 1)[1:-1]
    return [(cx + r * math.cos(x), cy + r * math.sin(x)) for x in ang]


def _outline(h) -> list:
    """The profile of base, support and lip, with the foot of the support rounded.

    It is drawn in the plane of the bed and runs counter-clockwise: from the back edge along the
    underside of the base to its front, up that front and along the shelf in front of the lip, up
    the lip, over its top and down its inside to the ledge, along the ledge the device stands on
    into the fillet, up the front of the support, over its half-round top and down its back to the
    base again, and back along the top of the base to its back edge. A point `up` millimetres up
    the lean stands `up * sin(angle)` high and falls `up * cos(angle)` back towards the back, so
    the faces of the lip, which lean with the support, meet the ledge further to the front the
    steeper the lean is.
    """
    a = math.radians(h["angle"])
    t, s, lip, depth = WALL, h["slot"], h["lip"], h["depth"]
    x0 = h["support_x"]
    sin, cos, tan = math.sin(a), math.cos(a), math.tan(a)
    front = x0 + (s + t) / sin          # where the outside of the lip meets the ledge
    back = x0 + s / sin                 # and where its inside does
    top = t + lip
    # the fillet is tangent to the ledge and to the front of the support and bulges into the
    # corner, so it adds plastic there; it never reaches further along the ledge than the lip
    r = min(FILLET, s / (sin * math.tan(0.5 * a)))
    d = r * math.tan(0.5 * a)
    pts = [(0.0, 0.0), (depth, 0.0), (depth, t), (front, t)]
    pts += [(front - lip / tan, top), (back - lip / tan, top), (back, t), (x0 + d, t)]
    pts += _arc(x0 + d, t + r, r, -0.5 * math.pi, -0.5 * math.pi - a, SEGMENTS // 4)
    pts.append((x0 - d * cos, t + d * sin))
    pts.append((x0 - h["back"] * cos, t + h["back"] * sin))
    mid = (x0 - h["back"] * cos - 0.5 * t * sin, t + h["back"] * sin - 0.5 * t * cos)
    pts += _arc(mid[0], mid[1], 0.5 * t, 0.5 * math.pi - a, 1.5 * math.pi - a, SEGMENTS // 2)
    pts += [(x0 - t / sin, t), (0.0, t)]
    # a fillet that is exactly as wide as the ledge needs leaves one of these points twice
    return [p for i, p in enumerate(pts) if math.dist(p, pts[i - 1]) > 1e-9]


def _cable_cuts(h) -> list:
    """The two boxes that cut the cable slot; empty when the stand does not get one.

    The gap is `CABLE` millimetres wide, right in the middle of the width. One box goes through
    the base from its back edge to the front of the lip, so a cable comes in behind the stand and
    runs to the device between the two rails that are left of the base; the other goes through the
    lip itself, from the ledge to above its top. The support keeps its foot, because the box of
    the base stops at the top of the base and the box of the lip stops at the slot.
    """
    if not h["cable"]:
        return []
    a = math.radians(h["angle"])
    t, s, lip = WALL, h["slot"], h["lip"]
    x0, z = h["support_x"], 0.5 * (h["width"] - CABLE)
    front = x0 + (s + t) / math.sin(a)
    back = x0 + s / math.sin(a) - lip / math.tan(a)      # its inside at the top of the lip
    base = mf.Manifold.cube([front + 2.0 * OVER, t + OVER, CABLE]).translate((-OVER, -OVER, z))
    ledge = mf.Manifold.cube([front - back + 2.0 * OVER, lip + 2.0 * OVER, CABLE]) \
        .translate((back - OVER, t - OVER, z))
    return [base, ledge]


def phone_stand(width=70.0, angle=65.0, back=90.0, device=11.0, lip=10.0,
                cable=True) -> mf.Manifold:
    """One printable phone or tablet stand, lying on its side on `z = 0`.

    The base lies on the bed and is `depth` millimetres long and `WALL` millimetres thick; the
    back support leaves it at `angle` degrees and rises `back` millimetres up the lean, with a
    half-round top, and in front of it the lip comes `lip` millimetres up above the ledge, so the
    slot between them is `device` plus a millimetre wide and a phone with its case in it stands on
    its foot and leans back. The stand is `width` millimetres across, which is how high it stands
    on the bed, and `cable` cuts a gap of 12 millimetres through the ledge, the lip and the base
    under them for a charging cable; a stand under 40 millimetres wide gets none.

    The base is as long as it has to be to keep the centre of a device of `back` × 1.8 millimetres
    `BACK_SPARE` millimetres in front of its back edge, so the stand does not tip over backwards,
    and it always reaches under the back of the rounded top, so nothing leans past the plate that
    lies on the bed. The foot of the support is rounded into the base with a fillet of 4
    millimetres. The profile is drawn with the back edge of the base on `x = 0` and the bed on
    `y = 0`; the stand that comes back is centred in x and y and stands on `z = 0`.

    Every number is clamped to what prints; anything but a number is a ValueError.
    """
    h = _sizes(width, angle, back, device, lip, cable)
    m = mf.Manifold.extrude(mf.CrossSection([_outline(h)]), h["width"])
    for cut in _cable_cuts(h):
        m = m - cut
    return m.translate((-0.5 * h["depth"], -0.5 * h["height"], 0.0))
