"""Hose adapters: the computational core of a printed reducer that joins two hoses.

No Qt in this file. It gives the adapter as one manifold, so it can be tested on its own and
reused in the model and the export.

A **hose adapter** is a hollow body of revolution that stands on the bed and prints upright:
hose `d1` goes over the **spigot** at the bottom and hose `d2` over the spigot at the top, with
`wall` millimetres of plastic around the water way everywhere. Between the two spigots stands
the **collar**, a stop of `COLLAR` millimetres high that the hoses are pushed up against, so
what is left of a hose is the same on both sides. Its underside is a cone of 45 degrees up from
the lower spigot, because a flat ring standing out in the air does not print without support.

A **barbed** spigot holds its hose with `barbs`: cones that widen from the hose diameter to
`BARB_GROW` of it over `BARB_LEN` millimetres going towards the middle and step straight back
afterwards. A hose rides over those shallow ramps and hangs behind the steps, which is how a
printed barb is made; the ramps are so shallow that they print upright without support. A
**straight** spigot is a plain cylinder of the hose diameter with a small chamfer at the tip,
for a tube that is pushed on and stays by friction or by a clamp.

The **bore** is the water way: `d - 2 * wall` at each spigot, joined through the collar by a cone
of 45 degrees, so the adapter prints upright whichever end is the fatter one. A bore that widens
upwards runs alongside the cone under the collar, `wall` millimetres inside it; one that narrows
upwards ends where the upper spigot begins, so neither ever comes closer to the outside than
`wall`. `length` is the
height of the whole part; by default it is what the barbs of both spigots and the collar need,
and what a longer length gives is plain tube to push a hose onto.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: how much fatter a barb is than the hose that is pushed over it
BARB_GROW = 1.08
#: mm of barb per mm of hose diameter, and the shortest barb that grips a hose
BARB_PART, BARB_MIN = 0.6, 4.0
#: mm high the collar between the spigots is, and mm it stands proud of the fatter spigot
COLLAR, COLLAR_ROOM = 3.0, 4.0
#: mm of spigot an end without barbs gets, so there is something to push a hose onto
STRAIGHT_MIN = 5.0
#: mm of chamfer at the tip of a straight spigot, never more than half the wall
CHAMFER = 0.5
#: the thinnest and the fattest hose this is made for, and the wall that goes with them
D_MIN, D_MAX = 3.0, 60.0
WALL_MIN, WALL_MAX = 0.8, 5.0
#: the fewest and the most barbs on one spigot
BARBS_MIN, BARBS_MAX = 1, 6
#: the narrowest water way that is worth printing
BORE_MIN = 1.0
#: the tallest adapter that is worth standing on the bed
LENGTH_MAX = 200.0
#: the straight pieces a round part is approximated with
SEGMENTS = 64
#: mm past the ends the bore is cut through, so the cut leaves no skin behind
MARGIN = 0.5


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _number(value, lo, hi):
    """A number of the adapter, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a hose adapter must be numbers"))
    return _clamp(v, lo, hi)


def _barb_len(d):
    """How far up a barb on a spigot of hose diameter `d` runs."""
    return max(BARB_PART * d, BARB_MIN)


def _sizes(d1, d2, length, wall, barbs, straight) -> dict:
    """The clamped numbers and the places of the features along the axis.

    In that frame the bed is on `z = 0`, where the tip of the lower spigot stands, and the tip of
    the upper one on `z = length`; `spigot1` and `spigot2` are the heights of the two spigots, of
    which the barbs take `barbs * barb_len`. Between them the 45 degree `chamfer` under the
    collar and the `COLLAR` itself, from `collar_z` up.
    """
    a = _number(d1, D_MIN, D_MAX)
    b = _number(d2, D_MIN, D_MAX)
    w = _number(wall, WALL_MIN, WALL_MAX)
    n = int(_number(barbs, BARBS_MIN, BARBS_MAX))
    if min(a, b) - 2.0 * w < BORE_MIN:
        raise ValueError(tr("the wall of a hose adapter leaves a bore of under {min} mm")
                         .format(min=f"{BORE_MIN:g}"))
    bl1, bl2 = _barb_len(a), _barb_len(b)
    room1 = n * bl1 if not straight else STRAIGHT_MIN
    room2 = n * bl2 if not straight else STRAIGHT_MIN
    collar_d = max(a, b) * BARB_GROW + COLLAR_ROOM
    chamfer = 0.5 * (collar_d - a)
    need = room1 + room2 + chamfer + COLLAR
    h = need if length is None else max(_number(length, 0.0, LENGTH_MAX), need)
    # what the barbs do not use is shared by the two spigots as plain tube to push a hose onto
    extra = 0.5 * (h - need)
    s1 = room1 + extra
    # the bore goes from one end to the other under 45 degrees, which is as much height as radius:
    # widening alongside the chamfer, or narrowing to end where the upper spigot begins
    r1, r2 = 0.5 * (a - 2.0 * w), 0.5 * (b - 2.0 * w)
    span = abs(r2 - r1)
    z0 = s1 if r2 >= r1 else s1 + chamfer + COLLAR - span
    return {"d1": a, "d2": b, "wall": w, "barbs": n, "straight": bool(straight),
            "length": h, "collar_d": collar_d, "chamfer": chamfer, "collar_z": s1 + chamfer,
            "bore1": 2.0 * r1, "bore2": 2.0 * r2,
            "barb_len1": 0.0 if straight else bl1, "barb_len2": 0.0 if straight else bl2,
            "spigot1": s1, "spigot2": room2 + extra, "cone_z0": z0, "cone_z1": z0 + span}


def sizes(d1=12.0, d2=8.0, length=None, wall=1.6, barbs=3, straight=False) -> dict:
    """The numbers a hose adapter is made of, so a dialog can show them.

    `d1` and `d2` are the inner diameters of the two hoses, `wall` the plastic around the water
    way, `barbs` the number of barbs per spigot and `length` the height of the part. `collar_d`
    is the outside of the stop between the spigots, `collar_z` the height its full width starts
    at and `chamfer` the height of the cone under it; `bore1` and `bore2` are the water way at
    each end, and `barb_len1` and `barb_len2` how far up a barb runs (0 for a straight end).
    """
    g = _sizes(d1, d2, length, wall, barbs, straight)
    return {key: g[key] for key in ("d1", "d2", "wall", "barbs", "length", "collar_d",
                                    "collar_z", "chamfer", "bore1", "bore2", "barb_len1",
                                    "barb_len2")}


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


def _spigot(d, room, barbs, barb_len, straight, wall) -> list:
    """The (radius, height) points of one spigot, from its tip at 0 up to its root at `room`.

    The barbs run from the tip towards the root: each one widens to `BARB_GROW` of the hose
    diameter and steps straight back, so a hose rides over the ramp and hangs behind the step.
    """
    r, r_b = 0.5 * d, 0.5 * d * BARB_GROW
    if straight:
        c = min(CHAMFER, 0.5 * wall, 0.5 * room)
        return [(r - c, 0.0), (r, c), (r, room)]
    points = [(r, 0.0)]
    for i in range(1, barbs + 1):
        points.append((r_b, i * barb_len))
        points.append((r, i * barb_len))
    points.append((r, room))
    return points


def _outside(g) -> mf.Manifold:
    """The wall of the adapter: the two spigots with their barbs and the collar between them."""
    h, s1, c = g["length"], g["spigot1"], g["collar_z"]
    bottom = _spigot(g["d1"], s1, g["barbs"], g["barb_len1"], g["straight"], g["wall"])
    top = _spigot(g["d2"], g["spigot2"], g["barbs"], g["barb_len2"], g["straight"], g["wall"])
    points = [(r, t) for r, t in bottom]
    points += [(0.5 * g["collar_d"], c), (0.5 * g["collar_d"], c + COLLAR)]
    points += [(r, h - t) for r, t in reversed(top)]
    return _revolved(points)


def _bore(g) -> mf.Manifold:
    """The water way: `d - 2 * wall` at each spigot, joined by a cone of 45 degrees.

    A long narrowing cone may start below the bed; the bore then starts as wide as the cone is
    there.
    """
    r1, r2 = 0.5 * g["bore1"], 0.5 * g["bore2"]
    z0, z1 = g["cone_z0"], g["cone_z1"]
    if z0 > -MARGIN:
        points = [(r1, -MARGIN), (r1, z0)]
    else:
        points = [(r1 + (r2 - r1) * (-MARGIN - z0) / (z1 - z0), -MARGIN)]
    return _revolved(points + [(r2, z1), (r2, g["length"] + MARGIN)])


def hose_adapter(d1=12.0, d2=8.0, length=None, wall=1.6, barbs=3, straight=False) -> mf.Manifold:
    """One printable hose adapter: a reducer with barbs between two hoses.

    The adapter stands on `z = 0` with its axis along Z and `length` millimetres high; `d1` is
    the inner diameter of the hose that goes over the spigot at the bottom and `d2` the one at
    the top, each spigot being `d` millimetres at its tip and `BARB_GROW` of that at its widest.
    `wall` is the plastic between the water way and the outside and `barbs` the number of barbs
    on each spigot; `straight` makes plain cylindrical ends with a chamfer at the tip instead,
    for a tube that is pushed on. Between the spigots stands a collar the hoses are pushed up
    against. `length` is by default what the barbs and the collar need, and a height that is too
    short for the barbs is raised to that; every number is clamped to what prints, except a wall
    that leaves no water way, which is a ValueError.
    """
    g = _sizes(d1, d2, length, wall, barbs, straight)
    return _outside(g) - _bore(g)
