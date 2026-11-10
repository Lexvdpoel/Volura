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
as one chain that bends.

The **stops** make it bend one way only: on the outside of every front plate, behind the pin, a
small round **tab** stands in an arc-shaped **window** through the rear plate of the link before
it. The window is the path of the tab from straight to `bend` degrees, with `CLEARANCE` around it,
so the chain lies straight on a table, bends up to `bend` degrees the other way round its loop, and
carries itself over a span. A chain too small for a tab beside its pin gets no stops.
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
#: degrees one link may turn on the next: from a gentle curve to a quarter turn
BEND_MIN, BEND_MAX = 20.0, 90.0
#: the tab of a stop is this part of the outside height across, within these bounds in mm
TAB_PART, TAB_R_MIN, TAB_R_MAX = 0.12, 0.6, 1.2
#: mm of plate that stays between the window of a stop and the pin hole or the edge of the plate
TAB_WEB = 0.6
#: the steps in which the path of a tab is drawn, per 90 degrees
TAB_STEPS = 12
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


def _tab(s):
    """(radius of the tab, distance from the pin) of the stops of a chain, or None without room."""
    r = 0.5 * s["outer_h"]
    tab_r = max(TAB_R_MIN, min(TAB_R_MAX, TAB_PART * s["outer_h"]))
    inner = s["pin_r"] + CLEARANCE + TAB_WEB + CLEARANCE + tab_r
    outer = r - TAB_WEB - CLEARANCE - tab_r
    if inner > outer:
        return None
    return tab_r, 0.5 * (inner + outer)


def _window(s, bend, centre_x, tab):
    """The window in a rear plate for the tab of the next link, which turns on the pin at
    (`centre_x`, outside height / 2): the tab swept from straight to `bend` degrees, with
    `CLEARANCE` around it, in the x, z plane of the plate."""
    tab_r, rho = tab
    r = 0.5 * s["outer_h"]
    steps = max(2, int(math.ceil(TAB_STEPS * bend / 90.0)))
    spots = []
    for i in range(steps + 1):
        a = math.radians(bend * i / steps)
        # the tab stands behind the pin and climbs as the next link turns
        spots.append(mf.CrossSection([_circle(centre_x - rho * math.cos(a), r + rho * math.sin(a),
                                               tab_r + CLEARANCE)]))
    return mf.CrossSection.batch_boolean(
        [mf.CrossSection.batch_hull([a, b]) for a, b in zip(spots, spots[1:])], mf.OpType.Add)


def chain_link(width=20.0, height=15.0, wall=2.0, bend=60.0) -> mf.Manifold:
    """One link, with the pin centre of its front on x = 0 and the hole of its rear on x = pitch.

    The link stands on `z = 0` and is centred on `y = 0`; the next link of a chain starts at
    `x = pitch`. The tab and the window of the stops let the next link turn from straight to
    `bend` degrees and no further.
    """
    s = sizes(width, height, wall)
    turn = _clamp(bend, BEND_MIN, BEND_MAX)
    tab = _tab(s)
    t, H, P, r = s["wall"], s["outer_h"], s["pitch"], 0.5 * s["outer_h"]
    mid = 0.5 * P
    front = mf.CrossSection([_circle(0.0, r, r)]) + mf.CrossSection.square([mid + t, H])
    rear = (mf.CrossSection([_circle(P, r, r)])
            + mf.CrossSection.square([P - mid + t, H]).translate([mid - t, 0.0]))
    hole = mf.CrossSection([_circle(P, r, s["pin_r"] + CLEARANCE)])
    rear = rear - hole
    if tab is not None:
        rear = rear - _window(s, turn, P, tab)
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
        if tab is not None:
            # the tab of the stop: behind the pin, through the window of the link before it
            nub = mf.CrossSection([_circle(-tab[1], r, tab[0])])
            parts.append(_plate(nub, *faces(s["front_out"] - 0.5, s["rear_out"])))
    # the cross bars, under and over the cables, where the plates step from front to rear
    for z in (0.0, H - t):
        parts.append(mf.Manifold.cube([2.0 * t, 2.0 * s["front_in"] + 1.0, t])
                     .translate((mid - t, -s["front_in"] - 0.5, z)))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def cable_chain(width=20.0, height=15.0, wall=2.0, links=10, bend=60.0) -> mf.Manifold:
    """`links` links in a row, each in the one before it, printed in place as one chain.

    The chain lies along x from the round front of the first link, stands on `z = 0` and is
    centred on the origin in plan.
    """
    n = max(LINKS_MIN, min(LINKS_MAX, int(links)))
    s = sizes(width, height, wall)
    if s["pin_r"] + CLEARANCE >= 0.5 * s["outer_h"] - s["wall"] * 0.5:
        raise ValueError(tr("the pins of this chain do not fit"))
    one = chain_link(width, height, wall, bend)
    chain = mf.Manifold.batch_boolean([one.translate((i * s["pitch"], 0.0, 0.0))
                                       for i in range(n)], mf.OpType.Add)
    lo = chain.bounding_box()
    return chain.translate((-0.5 * (lo[0] + lo[3]), -0.5 * (lo[1] + lo[4]), -lo[2]))
