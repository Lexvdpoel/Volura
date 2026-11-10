"""Cable chains: the computational core for a drag chain that is printed in one piece.

No Qt in this file. It gives the chain as one manifold of separate links, so it can be tested on
its own and reused in the model and the export.

A **cable chain** guides the cables of a moving axis: a row of links that turn on each other, each
an open frame of two side plates and two cross bars with the cables inside. The size you enter is
the inside, `width` by `height`, because that is what the cables need.

Every link has a narrow **front** and a wide **rear**: the side plates of the front stand at the
inside width, those of the rear `wall + CLEARANCE` further out, so the front of the next link fits
between them. A **pin** on the outside of every front plate turns in a **hole** through the rear
plate of the link before it. The ends of the plates are round about the pin, so a link turns
without its corners catching, and the cross bars sit in the middle of a link, where the plates
step from front to rear.

The chain is printed **in place**: all links lie in a row on the bed, already in each other, with
`CLEARANCE` millimetres of room around every pin and between the plates, so they come off the bed
as one chain that bends. The links have no stops, so the chain bends both ways.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .geom import Plane, extrude_cs
from .i18n import tr

#: mm of inside: the smallest bundle that is worth a chain and the largest one that prints
WIDTH_MIN, WIDTH_MAX = 5.0, 80.0
HEIGHT_MIN, HEIGHT_MAX = 5.0, 50.0
#: mm of plate and bar
WALL_MIN, WALL_MAX = 1.2, 4.0
#: the number of links in one print
LINKS_MIN, LINKS_MAX = 2, 60
#: mm of room around a pin and between the plates of two links, so they turn after printing
CLEARANCE = 0.3
#: the pin is this part of the outside height across
PIN_PART = 0.36
#: mm that the pitch is longer than strictly needed, so the bars stay clear of the next link
PITCH_ROOM = 1.0
#: the straight pieces a round end or a pin is approximated with
SEGMENTS = 48
#: the plane of the side plates: local x, y are the world x, z, extruded along the world -y
SIDE = Plane((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def sizes(width=20.0, height=15.0, wall=2.0) -> dict:
    """The sizes that follow from the inside: outside height, pitch, pin and the plate faces."""
    w = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    h = _clamp(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _clamp(wall, WALL_MIN, WALL_MAX)
    outer_h = h + 2.0 * t
    pitch = outer_h + 2.0 * t + 2.0 * CLEARANCE + PITCH_ROOM
    return {"width": w, "height": h, "wall": t, "outer_h": outer_h, "pitch": pitch,
            "pin_r": 0.5 * PIN_PART * outer_h,
            # y of the faces of the side plates: front inside/outside, rear inside/outside
            "front_in": 0.5 * w, "front_out": 0.5 * w + t,
            "rear_in": 0.5 * w + t + CLEARANCE, "rear_out": 0.5 * w + 2.0 * t + CLEARANCE}


def _circle(cx, cy, r) -> list:
    return [[cx + r * math.cos(2.0 * math.pi * i / SEGMENTS),
             cy + r * math.sin(2.0 * math.pi * i / SEGMENTS)] for i in range(SEGMENTS)]


def _plate(profile, y_lo, y_hi) -> mf.Manifold:
    """A side plate of `profile` (in x, z) between the faces `y_lo` and `y_hi`."""
    return extrude_cs(profile, SIDE, -y_hi, -y_lo)


def chain_link(width=20.0, height=15.0, wall=2.0) -> mf.Manifold:
    """One link, with the pin centre of its front on x = 0 and the hole of its rear on x = pitch.

    The link stands on `z = 0` and is centred on `y = 0`; the next link of a chain starts at
    `x = pitch`.
    """
    s = sizes(width, height, wall)
    t, H, P, r = s["wall"], s["outer_h"], s["pitch"], 0.5 * s["outer_h"]
    mid = 0.5 * P
    front = mf.CrossSection([_circle(0.0, r, r)]) + mf.CrossSection.square([mid + t, H])
    rear = (mf.CrossSection([_circle(P, r, r)])
            + mf.CrossSection.square([P - mid + t, H]).translate([mid - t, 0.0]))
    hole = mf.CrossSection([_circle(P, r, s["pin_r"] + CLEARANCE)])
    rear = rear - hole
    bridge = mf.CrossSection.square([2.0 * t, H]).translate([mid - t, 0.0])
    parts = []
    for side in (-1.0, 1.0):
        def faces(a, b):
            return (a, b) if side > 0 else (-b, -a)
        parts.append(_plate(front, *faces(s["front_in"], s["front_out"])))
        parts.append(_plate(rear, *faces(s["rear_in"], s["rear_out"])))
        parts.append(_plate(bridge, *faces(s["front_in"], s["rear_out"])))
        # the pin: from the front plate out through the hole of the link before it
        pin = mf.CrossSection([_circle(0.0, r, s["pin_r"])])
        parts.append(_plate(pin, *faces(s["front_out"] - 0.5, s["rear_out"])))
    # the cross bars, under and over the cables, where the plates step from front to rear
    for z in (0.0, H - t):
        parts.append(mf.Manifold.cube([2.0 * t, 2.0 * s["front_in"] + 1.0, t])
                     .translate((mid - t, -s["front_in"] - 0.5, z)))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def cable_chain(width=20.0, height=15.0, wall=2.0, links=10) -> mf.Manifold:
    """`links` links in a row, each in the one before it, printed in place as one chain.

    The chain lies along x from the round front of the first link, stands on `z = 0` and is
    centred on the origin in plan.
    """
    n = max(LINKS_MIN, min(LINKS_MAX, int(links)))
    s = sizes(width, height, wall)
    if s["pin_r"] + CLEARANCE >= 0.5 * s["outer_h"] - s["wall"] * 0.5:
        raise ValueError(tr("the pins of this chain do not fit"))
    one = chain_link(width, height, wall)
    chain = mf.Manifold.batch_boolean([one.translate((i * s["pitch"], 0.0, 0.0))
                                       for i in range(n)], mf.OpType.Add)
    lo = chain.bounding_box()
    return chain.translate((-0.5 * (lo[0] + lo[3]), -0.5 * (lo[1] + lo[4]), -lo[2]))
