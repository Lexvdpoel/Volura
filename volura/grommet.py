"""Desk grommets: the computational core of a printed cable pass-through for a desk top.

No Qt in this file. It gives the grommet as one manifold, so it can be tested on its own and
reused in the model and the export.

A **desk grommet** finishes the edge of the hole in a desk top that the cables go through, so
they stop chafing on the wood and the hole looks as if it came from the factory. The
**flange** is a ring of `hole + 2 * flange` millimetres across that lies on the desk, 2
millimetres thick with a chamfer on the edge you see; the **sleeve** hangs below it in the hole,
`hole - 0.4` millimetres across so it slides in, and four **crush ribs** 0.4 millimetres proud
of it keep the grommet from turning or dropping through. The sleeve reaches as deep as the desk
is thick, but never deeper than 28 millimetres: the bottom of a thick desk holds nothing.

The **lid** is a disc that drops into the **recess** in the flange and closes the hole off, with
a **slot** from the rim inward for the cables that stay in the desk. It prints flat next to the
grommet and in use it rests on the cone at the bottom of the recess, `PLAY` millimetres clear
all round so it lifts out by hand.

The grommet prints **flange down**: `z = 0` is the edge that is visible above the desk and the
sleeve points up, so everything either lies on the bed or rises straight from it. The step from
the recess down to the bore is a cone of 45 degrees for that reason: flat, it would be a ceiling
of `THICK` millimetres over the bed.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: the smallest and the largest hole in a desk this is made for
HOLE_MIN, HOLE_MAX = 20.0, 120.0
#: the thinnest and the thickest desk top
DESK_MIN, DESK_MAX = 10.0, 60.0
#: the narrowest and the widest flange that covers a hole, and the thinnest and thickest wall
FLANGE_MIN, FLANGE_MAX = 4.0, 20.0
WALL_MIN, WALL_MAX = 2.0, 4.0
#: mm the sleeve is smaller than the hole, so that it slides into it
FIT = 0.4
#: mm the crush ribs stand proud of the sleeve, how wide they are and how many there are
RIB, RIB_W, RIBS = 0.4, 1.0, 4
#: mm the sleeve reaches into the hole at most, and mm of it that stays out of the hole
SLEEVE_MAX, SLEEVE_KEEP = 30.0, 2.0
#: mm thick the flange is, which is how deep the recess in it is
THICK = 2.0
#: mm of chamfer on the outer edge of the flange that is visible above the desk
CHAMFER = 0.8
#: mm the recess is wider than the bore, so half of it is the height of the cone between them
RECESS_ROOM = 2.0
#: mm of plastic that always stays between the recess and the outside of the sleeve: that ring
#: is all that joins the sleeve to the flange, so it must be a few lines of the nozzle thick
NECK = 1.2
#: mm of play all round between the lid and the recess
PLAY = 0.3
#: mm across the slot the cable lies in, the radius its corners are rounded to, and the ring of
#: lid that has to stay around it
SLOT_ACROSS, SLOT_R, LID_EDGE = 12.0, 2.0, 10.0
#: mm between the grommet and the lid on the bed
LID_GAP = 10.0
#: the straight pieces a round part is approximated with
SEGMENTS = 96
#: mm past the ends the passage is cut through, so the cut leaves no skin behind
MARGIN = 0.5


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _number(value, lo, hi):
    """A number of the grommet, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a desk grommet must be numbers"))
    return _clamp(v, lo, hi)


def _sizes(hole, desk, flange, wall, slot, lid) -> dict:
    """The clamped numbers and the places of the features along the axis.

    In that frame the grommet prints flange down: `z = 0` is the edge of the flange that is
    visible above the desk and the sleeve points up. The flange is `z = 0 .. THICK` with the
    recess open on the bed, the cone to the bore starts at `z = THICK` and the sleeve runs from
    there to `z = THICK + sleeve_len`. The lid lies flat next to the grommet, centred on
    `x = lid_x`; `slot_deep` is how far in from its rim the slot reaches and `rib_r` the radius
    the crush ribs are rooted at.
    """
    h = _number(hole, HOLE_MIN, HOLE_MAX)
    t = _number(desk, DESK_MIN, DESK_MAX)
    f = _number(flange, FLANGE_MIN, FLANGE_MAX)
    w = _number(wall, WALL_MIN, WALL_MAX)
    s = _number(slot, 0.0, 0.5 * h)
    sleeve_d = h - FIT
    bore_d = sleeve_d - 2.0 * w
    # a thin wall would leave only a hair of plastic between the recess and the outside, and that
    # ring is what holds the sleeve to the flange: the ledge the lid rests on gets narrower first
    recess_d = min(h - 2.0 * w + RECESS_ROOM, sleeve_d - 2.0 * NECK)
    lid_d = recess_d - 2.0 * PLAY
    return {"hole": h, "desk": t, "flange": f, "wall": w, "slot": s, "lid": bool(lid),
            "sleeve_d": sleeve_d, "bore_d": bore_d, "recess_d": recess_d, "lid_d": lid_d,
            "flange_d": h + 2.0 * f, "sleeve_len": min(t, SLEEVE_MAX) - SLEEVE_KEEP,
            "cone_h": 0.5 * (recess_d - bore_d), "lid_x": h + 2.0 * f + LID_GAP,
            "rib_r": max(0.5 * recess_d, 0.25 * (bore_d + sleeve_d)),
            "slot_deep": max(0.0, min(s, 0.5 * lid_d - LID_EDGE))}


def sizes(hole=60.0, desk=25.0, flange=8.0, wall=2.0, slot=20.0, lid=True) -> dict:
    """The numbers a desk grommet is made of, so a dialog can show them.

    `hole` is the diameter of the hole in the desk and `desk` the thickness of that desk top,
    `flange` how far the ring that lies on the desk comes out past the hole and `wall` the
    plastic of the sleeve. `slot` is how far in from the rim the slot of the lid reaches and
    `lid` whether the lid is made along with it. `sleeve_d` and `sleeve_len` are the outside and
    the length of the tube that goes into the hole, `flange_d` the outside of the ring and
    `bore_d` the inside of the sleeve; `recess_d` is the pocket in the flange that the lid drops
    into and `lid_d` the lid itself, `PLAY` millimetres smaller all round.
    """
    g = _sizes(hole, desk, flange, wall, slot, lid)
    return {key: g[key] for key in ("hole", "desk", "flange", "wall", "slot", "lid",
                                    "sleeve_d", "sleeve_len", "flange_d", "recess_d", "lid_d",
                                    "bore_d")}


def _revolved(points) -> mf.Manifold:
    """The solid of revolution of (radius, height) points that rise from the bottom to the top.

    Consecutive points are joined by a cone, so a step in the radius becomes a flat ring. The
    profile is closed along the axis and turned round it in one piece: cones stacked on each
    other only touch, and a union of solids that only touch falls apart in loose rings.
    """
    profile = [(0.0, points[0][1])]
    for p in list(points) + [(0.0, points[-1][1])]:
        if abs(p[0] - profile[-1][0]) > 1e-9 or abs(p[1] - profile[-1][1]) > 1e-9:
            profile.append((float(p[0]), float(p[1])))
    return mf.Manifold.revolve(mf.CrossSection([profile]), SEGMENTS)


def _outside(g) -> list:
    """The (radius, height) points of the outside, from the bed up to the tip of the sleeve.

    The flange stands on the bed with its visible edge chamfered, and out of its middle the
    sleeve rises; what is left of the flange around the sleeve is where it lies on the desk.
    """
    c = min(CHAMFER, THICK)
    top = THICK + g["sleeve_len"]
    return [(0.5 * g["flange_d"] - c, 0.0), (0.5 * g["flange_d"], c),
            (0.5 * g["flange_d"], THICK), (0.5 * g["sleeve_d"], THICK), (0.5 * g["sleeve_d"], top)]


def _bore(g) -> list:
    """The (radius, height) points of the passage: the recess, the cone and the bore.

    The recess is as deep as the flange is thick and opens on the bed, which is the side the lid
    comes in at; from there a cone of 45 degrees steps down to the bore, which is the inside of
    the sleeve and goes out at the tip.
    """
    return [(0.5 * g["recess_d"], -MARGIN), (0.5 * g["recess_d"], THICK),
            (0.5 * g["bore_d"], THICK + g["cone_h"]),
            (0.5 * g["bore_d"], THICK + g["sleeve_len"] + MARGIN)]


def _ribs(g) -> mf.Manifold:
    """The crush ribs along the sleeve: `RIB` millimetres proud of it, `RIB_W` wide, `RIBS` of them.

    They are what holds the grommet in the hole: the sleeve is a hair smaller than the hole and
    the ribs a hair larger, so the desk top has to be pushed over them and they bite into it.
    They root in the wall of the sleeve, but never in the recess the lid sits in.
    """
    reach = 0.5 * g["sleeve_d"] + RIB
    rib = (mf.CrossSection.square([reach - g["rib_r"], RIB_W], center=True)
           .translate([0.5 * (reach + g["rib_r"]), 0.0]))
    bars = mf.CrossSection()
    for i in range(RIBS):
        bars = bars + rib.rotate(i * 360.0 / RIBS)
    return mf.Manifold.extrude(bars, g["sleeve_len"]).translate((0.0, 0.0, THICK))


def _slot(g) -> mf.CrossSection:
    """The slot of the lid: `SLOT_ACROSS` millimetres across and `slot_deep` in from the rim.

    It reaches past the rim, so the slot stays open and a cable that is already lying in the
    desk can still be closed off; its corners are rounded, because a cable does not like a sharp
    corner either. The rectangle is made by inflating a smaller one, which adds `SLOT_R` to
    every side, so it is shifted out by that much to keep its bottom where it belongs.
    """
    length = g["slot_deep"] + MARGIN
    return (mf.CrossSection.square([length, SLOT_ACROSS - 2.0 * SLOT_R], center=True)
            .offset(SLOT_R, mf.JoinType.Round, circular_segments=SEGMENTS // 4)
            .translate([0.5 * g["lid_d"] - g["slot_deep"] + SLOT_R + 0.5 * length, 0.0]))


def _lid(g) -> mf.Manifold:
    """The lid that drops into the recess: a disc of `lid_d`, as thick as the flange is.

    It lies flat on the bed next to the grommet, so it prints with nothing under it. A lid whose
    slot would reach the middle is made without one: `slot_deep` is 0 there.
    """
    r = 0.5 * g["lid_d"]
    lid = mf.Manifold.cylinder(THICK, r, r, SEGMENTS)
    if g["slot_deep"] > 0.0:
        cut = mf.Manifold.extrude(_slot(g), THICK + 2.0 * MARGIN).translate((0.0, 0.0, -MARGIN))
        lid = lid - cut
    return lid.translate((g["lid_x"], 0.0, 0.0))


def grommet(hole=60.0, desk=25.0, flange=8.0, wall=2.0, slot=20.0, lid=True) -> mf.Manifold:
    """One printable desk grommet, with the lid that goes with it lying next to it.

    The grommet prints flange down on `z = 0`, which is the edge that is visible above the desk,
    and is `THICK + sleeve_len` millimetres high; `hole` is the diameter of the hole in the desk
    and `desk` the thickness of that desk top, which the sleeve reaches `SLEEVE_KEEP` millimetres
    short of. The flange lies on the desk and comes `flange` millimetres out past the hole, the
    sleeve is `wall` millimetres thick and `FIT` smaller than the hole, with `RIBS` crush ribs
    on it. The lid closes the hole off and has a slot of `slot` millimetres deep for the cables;
    `lid` is whether it is made along with the grommet, as a second solid at
    `hole + 2 * flange + LID_GAP`. Every number is clamped to what prints; anything but a number
    is a ValueError.
    """
    g = _sizes(hole, desk, flange, wall, slot, lid)
    m = _revolved(_outside(g)) - _revolved(_bore(g)) + _ribs(g)
    if g["lid"]:
        m = m + _lid(g)
    return m
