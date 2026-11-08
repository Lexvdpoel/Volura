"""Thumb knobs: the computational core for a printable knob that turns on a bolt or a shaft.

No Qt in this file. It gives the knob as one closed volume, so it can be tested on its own
and reused in the model and the export.

A **knob** is the round part you turn with two fingers: on a bolt that stands out of a part,
on the shaft of a potentiometer or on a motor axle. It stands on the bed `height` millimetres
high and is centred on the origin, so it prints without support and lands where the other
printed parts land.

The **grip** is the rim you hold. `knurl` puts straight ribs of 1 mm around the rim, about one
rib per 2 mm of circumference, the ridges that stay grippy with oily fingers; `lobes` makes a
star of rounded lobes, whose valleys stand at 0.7 of the diameter, which is easier to print
than fine ribs and still turns under a thumb; `smooth` leaves it a plain cylinder.

The **mount** is how the knob takes the part it turns. `nut` cuts a clearance hole through the
knob and a hexagonal pocket in the bottom face, so a hex nut is pressed in and the knob screws
onto a bolt that is already in the machine; `head` cuts a clearance hole through the knob and a
round counterbore in the bottom face, so the head of a socket head bolt is captured and the
knob turns with it; `shaft` cuts a blind D hole in the bottom face, the round shaft of a
potentiometer with one side filed flat, so the knob cannot spin on the axle. `clearance` is
added to every diameter, because a printed hole comes out smaller than drawn.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .geom import circle_points
from .i18n import tr
from . import threads

#: the smallest knob a hand still fits and the largest that is worth printing
DIAMETER_MIN, DIAMETER_MAX = 10.0, 120.0
#: the thickest and the thinnest knob: a knob of 4 mm is already hard to hold
HEIGHT_MIN, HEIGHT_MAX = 4.0, 60.0
#: the fewest and the most lobes a lobed rim can have
LOBES_MIN, LOBES_MAX = 3, 12
#: the largest clearance on a diameter, above which a knob no longer grips its bolt
CLEARANCE_MAX = 1.0
#: mm of plastic between the widest hole and the valleys of the rim, and above a blind pocket:
#: less than that breaks off in the hand
WALL_MIN, TOP_MIN = 1.2, 1.0
#: mm a rib of a knurled rim stands above the core, and the mm of circumference one rib takes
RIB_DEPTH, RIB_PITCH = 1.0, 2.0
#: the points one rib is made of: a crest of five of them, a valley of the three left
RIB_POINTS = 8
#: the valleys of a lobed rim stand at this fraction of the outer radius
VALLEY_OVER_RADIUS = 0.7
#: mm deeper than the nut or the head, so the part that is pressed in goes all the way down
POCKET_ROOM = 0.2
#: mm of knob that is left above a D shaft hole, so the end of the shaft has room
SHAFT_ROOM = 2.0
#: the straight pieces a round hole or a round rim is approximated with
SEGMENTS = 64
#: mm past the bottom face and over the top, so a cut leaves no skin behind
MARGIN = 0.5

KNOB_KINDS = ("nut", "head", "shaft")
KNOB_GRIPS = ("knurl", "lobes", "smooth")


def _ribs(radius, diameter):
    """The rim of a knurled knob: straight ribs of `RIB_DEPTH` mm around a circle.

    About one rib per `RIB_PITCH` millimetres of circumference, rounded to a multiple of four
    so that a rib stands at 0, 90, 180 and 270 degrees and the knob is exactly `diameter`
    millimetres across. Every rib is a flat crest of `radius` and a flat valley of
    `radius − RIB_DEPTH`, so the ribs are straight vertical ridges with nothing hanging.
    """
    ribs = max(4, 4 * int(round(math.pi * float(diameter) / RIB_PITCH / 4.0)))
    step = 2.0 * math.pi / ribs
    j = np.arange(RIB_POINTS) - RIB_POINTS // 2
    ang = np.arange(ribs)[:, None] * step + j * (step / RIB_POINTS)
    r = np.where(np.abs(j) <= RIB_POINTS // 4, radius, radius - RIB_DEPTH)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1).reshape(-1, 2)


def _lobed(radius, lobes):
    """The rim of a lobed knob: `lobes` rounded lobes with rounded valleys between them.

    The radius runs as a cosine with `lobes` periods round the axis, from `radius` on a lobe
    to `VALLEY_OVER_RADIUS` of it in a valley, so lobe and valley are both round and there is
    no sharp point to print. A lobe stands at 0 degrees, so the knob is `diameter` across.
    """
    low = VALLEY_OVER_RADIUS * radius
    mid, amp = 0.5 * (radius + low), 0.5 * (radius - low)
    ang = np.linspace(0.0, 2.0 * math.pi, max(4 * SEGMENTS, 24 * int(lobes)), endpoint=False)
    r = mid + amp * np.cos(int(lobes) * ang)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _d_profile(radius, flat):
    """The outline of a D shaft hole: a circle of `radius` with a flat of `flat` mm.

    The flat is the chord at `radius − flat` from the axis on the −x side, the side of a
    potentiometer shaft that is filed flat, so the knob cannot spin on the axle. Without a
    flat the outline is the plain circle of the shaft.
    """
    r, f = float(radius), float(flat)
    if f <= 0.0:
        return circle_points(r, SEGMENTS)
    a = math.acos(max(-1.0, min(1.0, -(r - f) / r)))
    ang = np.linspace(-a, a, SEGMENTS - 1)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def knob(diameter=30.0, height=12.0, grip="knurl", lobes=6, bore="M4", kind="nut",
         clearance=threads.DEFAULT_HOLE_CLEARANCE, shaft_d=6.0,
         flat=0.5) -> mf.Manifold:
    """A printable thumb knob that is screwed on a bolt or pushed on a shaft.

    The knob stands on `z = 0`, centred on the origin, `height` millimetres high and
    `diameter` millimetres across. `grip` is the rim you hold, one of `KNOB_GRIPS`: "knurl"
    straight ribs, "lobes" `lobes` rounded lobes or "smooth" a plain cylinder.

    `kind` is the mount, one of `KNOB_KINDS`: "nut" cuts a clearance hole for the bolt of
    `bore` through the knob and a hexagonal pocket for a nut of the same size in the bottom
    face; "head" cuts that clearance hole and a round counterbore for the head of a socket
    head bolt; "shaft" cuts a blind D hole of `shaft_d` with a flat of `flat` millimetres,
    `height − 2` deep, for a potentiometer or motor axle. `clearance` is added to every
    diameter, because a printed hole comes out smaller than drawn.
    """
    d, h = float(diameter), float(height)
    if d < DIAMETER_MIN or d > DIAMETER_MAX:
        raise ValueError(tr("the diameter of a knob must be {lo} to {hi} mm")
                         .format(lo=f"{DIAMETER_MIN:g}", hi=f"{DIAMETER_MAX:g}"))
    if h < HEIGHT_MIN or h > HEIGHT_MAX:
        raise ValueError(tr("the height of a knob must be {lo} to {hi} mm")
                         .format(lo=f"{HEIGHT_MIN:g}", hi=f"{HEIGHT_MAX:g}"))
    if grip not in KNOB_GRIPS:
        raise ValueError(tr("unknown knob grip {grip}").format(grip=grip))
    if kind not in KNOB_KINDS:
        raise ValueError(tr("unknown knob mount {kind}").format(kind=kind))
    if bore not in threads.SCREW_HOLES:
        raise ValueError(tr("unknown screw size {size}").format(size=bore))
    n_lobes = int(lobes)
    if n_lobes < LOBES_MIN or n_lobes > LOBES_MAX:
        raise ValueError(tr("the number of lobes must be {lo} to {hi}")
                         .format(lo=LOBES_MIN, hi=LOBES_MAX))
    c = float(clearance)
    if c < 0.0 or c > CLEARANCE_MAX:
        raise ValueError(tr("The clearance must be between 0 and 1 mm."))
    s_d, f = float(shaft_d), float(flat)
    if s_d <= 0.0:
        raise ValueError(tr("the shaft diameter must be greater than zero"))
    if f < 0.0:
        raise ValueError(tr("the flat may not be negative"))

    # the rim: the outline you hold, and the radius of its deepest point, which is where a
    # hole has to leave room
    r = 0.5 * d
    if grip == "knurl":
        outline, valley = mf.CrossSection([_ribs(r, d)]), r - RIB_DEPTH
    elif grip == "lobes":
        outline, valley = mf.CrossSection([_lobed(r, n_lobes)]), VALLEY_OVER_RADIUS * r
    else:
        outline, valley = mf.CrossSection([circle_points(r, SEGMENTS)]), r

    # the mount: how wide the widest cut is and how deep it comes from the bottom face
    hole_d, head_d, head_h, _cs = threads.SCREW_HOLES[bore]
    if kind == "nut":
        s, nut_h = threads.HEX_NUTS[bore]
        wide, deep = (s + c) / math.sqrt(3.0), nut_h + POCKET_ROOM
    elif kind == "head":
        wide, deep = 0.5 * (head_d + c), head_h + POCKET_ROOM
    else:
        wide, deep = 0.5 * (s_d + c), h - SHAFT_ROOM
    if f >= wide:
        raise ValueError(tr("the flat must be less than half the shaft diameter"))
    if valley - wide < WALL_MIN or h - deep < TOP_MIN:
        raise ValueError(tr("The hole does not fit in this knob."))

    body = mf.Manifold.extrude(outline, h)
    if kind == "shaft":
        # the D hole is blind, so it stops under the top and the shaft has room to go in
        body = body - mf.Manifold.extrude(mf.CrossSection([_d_profile(wide, f)]),
                                          deep + MARGIN).translate((0.0, 0.0, -MARGIN))
        return body
    # a bolt goes through the knob, so the hole is plain all the way out at the top and the
    # pocket for the nut or the head opens in the bottom face
    body = body - mf.Manifold.cylinder(h + 2.0 * MARGIN, 0.5 * (hole_d + c), 0.5 * (hole_d + c),
                                       SEGMENTS).translate((0.0, 0.0, -MARGIN))
    if kind == "nut":
        # the nut of the same size, held by its flats: a hexagon whose corners stand
        # (s + clearance) / sqrt(3) from the axis is that wide between the parallel sides
        pocket = mf.Manifold.extrude(mf.CrossSection([circle_points(wide, 6)]),
                                     deep + MARGIN).translate((0.0, 0.0, -MARGIN))
    else:
        pocket = mf.Manifold.cylinder(deep + MARGIN, wide, wide,
                                      SEGMENTS).translate((0.0, 0.0, -MARGIN))
    return body - pocket
