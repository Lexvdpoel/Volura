"""Plant pots: the computational core of a flower pot that prints upright, with a saucer.

No Qt in this file. It gives the pot and its saucer as closed volumes, so they can be tested on
their own and reused in the model and the export.

A **plant pot** is the tapered pot a plant sits in: `diameter` millimetres across at the top,
narrowing to `taper` percent of that at the bottom, `height` millimetres high, with walls of `wall`
millimetres and a floor of `FLOOR`. It prints **upright**, on its bottom, so the wall leans outwards
and never further out than it stands high: a taper that would lean the wall past 45 degrees is made
smaller, because that is what a pot without support survives. The rim is rounded outward by a bead
of `BEAD` millimetres, so the pot has no sharp edge left to break off and the bead is the widest
thing about it; the bead grows out of the wall at 45 degrees, so it needs no support either.

The **shape** is what the pot looks like from above: "round", or "hexagon", a six-sided pot of the
same sizes measured across the flats, which stands still on a tilted bench and does not roll off
it.

The **drain holes** are `holes` in number: one in the middle of the floor and the rest on a circle
halfway to the wall, `HOLE_D` millimetres across, so the water a plant does not drink runs out
instead of standing round the roots. A floor too small to hold them gets smaller holes, never
holes that run into one another. Under the floor, channels of `CHANNEL_W` millimetres wide and
`CHANNEL_H` high run from the drain holes out to the edge, so the water runs out from under the
pot too and the pot does not sit in its own water. The pot stands on the rest of its bottom, so
the floor only bridges the narrow channels and the pot still prints without support.

The **saucer** is the tray that catches that water: the bottom of the pot plus `SAUCER_ROOM`
millimetres all round, `SAUCER_H` millimetres high, on a floor of `SAUCER_FLOOR` with walls of
`SAUCER_WALL`, of the same shape as the pot.

Every number is clamped to what prints. The pot and the saucer are separate closed volumes that
both stand on `z = 0`, `GAP` millimetres apart and centred together in x and y, so one print gives
a pot and a saucer and the pair lands in the middle of the bed.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm across the top of the smallest pot worth printing and the largest that is still a pot
D_MIN, D_MAX = 40.0, 250.0
#: mm the shortest and the tallest pot that is worth standing on a bench
HEIGHT_MIN, HEIGHT_MAX = 30.0, 250.0
#: percent of the top diameter the bottom narrows to: from a steep flower pot to a straight one
TAPER_MIN, TAPER_MAX = 50.0, 100.0
#: the thinnest wall that prints upright and the thickest that is still worth the plastic
WALL_MIN, WALL_MAX = 1.2, 4.0
#: the fewest and the most drain holes, the one in the middle of the floor included
HOLES_MIN, HOLES_MAX = 0, 9
#: mm thick the floor is: watertight, and still one piece after a summer of water
FLOOR = 2.5
#: mm high the channels under the floor are, so the water runs out from under the pot
CHANNEL_H = 2.0
#: mm wide a channel is: narrow enough for the floor to bridge it while it prints
CHANNEL_W = 5.0
#: how many channels a pot with only the middle drain hole gets; otherwise there is one under
#: every hole on the circle, running through the middle one
CHANNEL_N = 3
#: mm the rim rounds outward by
BEAD = 2.0
#: mm across a drain hole
HOLE_D = 8.0
#: mm of floor that stays between two drain holes, so they do not merge into one
HOLE_WEB = 0.5
#: mm the saucer is larger than the bottom of the pot all round, and mm high
SAUCER_ROOM, SAUCER_H = 10.0, 15.0
#: mm thick the wall and the floor of the saucer are
SAUCER_WALL, SAUCER_FLOOR = 2.0, 2.0
#: mm of bed between the pot and the saucer
GAP = 5.0
#: the shapes a pot comes in: a round one and a six-sided one
POT_SHAPES = ("round", "hexagon")
#: the straight pieces a round pot, a round foot or a round hole is approximated with
SEGMENTS = 64
#: the straight pieces the rounding of the rim is approximated with
BEAD_SEGMENTS = 6
#: a hexagon stands this much further out at its corners than at its flats
CORNER = 1.0 / math.cos(math.pi / 6.0)
#: the area of a hexagon is this much the square of the radius to its flats
HEX_AREA = 6.0 * math.tan(math.pi / 6.0)
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5


def _number(value, lo, hi):
    """A number of the pot, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a plant pot must be numbers"))
    return max(lo, min(hi, v))


def _sizes(diameter, height, taper, wall, holes, shape, saucer) -> dict:
    """The clamped numbers and the places of the features, in the frame the pot prints in.

    In that frame the pot stands on its bottom on `z = 0` and is centred on the origin. `radius` is
    the outside radius at the top of the wall and `bottom_radius` the one at its bottom, which is
    `taper` percent of the top unless that would lean the wall out further than it stands high;
    `wall_h` is the height the wall leans over. `hole_r` is the radius a drain hole is cut with
    and `ring` the radius of the circle on which all but the middle one stand; `saucer_radius` is
    the inside radius of the saucer.
    """
    D = _number(diameter, D_MIN, D_MAX)
    H = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _number(taper, TAPER_MIN, TAPER_MAX)
    w = _number(wall, WALL_MIN, WALL_MAX)
    n = int(_number(holes, HOLES_MIN, HOLES_MAX))
    sh = str(shape or "").strip().lower()
    R = 0.5 * D
    wall_h = H - 2.0 * BEAD - CHANNEL_H      # mm the wall stands from its bottom face to the bead
    # a wall that leans out by as much as it stands high leans 45 degrees, which is the steepest
    # that prints upright without support: a taper that leans further makes the bottom larger
    r_b = max(R * t / 100.0, R - wall_h)
    ring = 0.5 * (r_b - w)              # the drain holes stand halfway between the axis and the wall
    return {"diameter": D, "height": H, "taper": 100.0 * r_b / R, "wall": w, "holes": n,
            "shape": sh if sh in POT_SHAPES else POT_SHAPES[0], "saucer": bool(saucer),
            "radius": R, "bottom_radius": r_b, "wall_h": wall_h, "ring": ring,
            "hole_r": min(0.5 * HOLE_D, 0.5 * ring - HOLE_WEB),
            "saucer_radius": r_b + SAUCER_ROOM}


def sizes(diameter=100.0, height=90.0, taper=75.0, wall=2.0, holes=5, shape="round",
          saucer=True) -> dict:
    """The numbers a plant pot is made of, so a dialog can show them.

    `diameter` is the top of the pot and `height` how high it stands; `taper` is how wide the
    bottom is as a percentage of the top, `wall` how thick the plastic is and `holes` how many
    drain holes the floor gets. `shape` is one of `POT_SHAPES` and `saucer` says whether the pot
    comes with its saucer. `bottom` is the diameter of the bottom the pot stands on, `hole` the
    diameter a drain hole is cut with and `hole_ring` the diameter of the circle on which all but
    the middle one stand; `saucer_inside` is the inside of the saucer and `capacity` what the pot
    holds in litres. Every number is clamped to what prints; anything but a number is a ValueError.
    """
    g = _sizes(diameter, height, taper, wall, holes, shape, saucer)
    return {"diameter": g["diameter"], "height": g["height"], "taper": g["taper"],
            "wall": g["wall"], "holes": g["holes"], "shape": g["shape"], "saucer": g["saucer"],
            "bottom": 2.0 * g["bottom_radius"], "hole": 2.0 * g["hole_r"],
            "hole_ring": 2.0 * g["ring"], "saucer_inside": 2.0 * g["saucer_radius"],
            "capacity": _capacity(g)}


def _capacity(g) -> float:
    """What the pot holds in litres: the space between the top of the floor and the rim."""
    r0, r1 = _cavity(g)
    k = HEX_AREA if g["shape"] == "hexagon" else math.pi
    lean = g["wall_h"] - FLOOR
    return k * (lean * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0 + 2.0 * BEAD * r1 * r1) / 1e6


def _slope(g) -> float:
    """How much the wall leans out per millimetre it stands high."""
    return (g["radius"] - g["bottom_radius"]) / g["wall_h"]


def _cavity(g) -> tuple:
    """The radii of the soil space at the top of the floor and at the rim: the leaning wall
    `wall` millimetres inwards on every side, so the plastic stays the same thickness all the way
    up and the rim is `wall` wide."""
    return (g["bottom_radius"] + _slope(g) * FLOOR - g["wall"], g["radius"] - g["wall"])


def _places(n, at) -> list:
    """The places of `n` features evenly round the axis, each at radius `at`."""
    return [(at * math.cos(2.0 * math.pi * k / n), at * math.sin(2.0 * math.pi * k / n))
            for k in range(n)]


def _post(x, y, r, z0, z1) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, SEGMENTS).translate((x, y, z0))


def _revolved(points) -> mf.Manifold:
    """The solid of revolution of (radius, height) points that rise from the bottom to the top.

    Consecutive points are joined by a cone, so a step in the radius becomes a flat ring. The
    profile is closed along the axis and turned round it in one piece.
    """
    profile = [(0.0, points[0][1])]
    for p in list(points) + [(0.0, points[-1][1])]:
        if abs(p[0] - profile[-1][0]) > 1e-9 or abs(p[1] - profile[-1][1]) > 1e-9:
            profile.append((float(p[0]), float(p[1])))
    return mf.Manifold.revolve(mf.CrossSection([profile]), SEGMENTS)


def _loft(shape, points) -> mf.Manifold:
    """The solid whose outline at every height is `shape` of the radius the (radius, height)
    points give, rising in a straight line between two of them.

    A radius is measured to the flat of the shape, which for a round pot is its radius and for a
    hexagon half the distance across its flats, so both shapes are the size a pot is sold as. The
    round pot is turned round the axis in one go; the hexagon is the same profile pulled through
    six-sided outlines, one standing on the other, which share the whole face between them and so
    make one volume. A step out at the same height belongs to the piece above it, whose bottom
    face covers it.
    """
    if shape != "hexagon":
        return _revolved(points)
    out = None
    for (r0, z0), (r1, z1) in zip(points, points[1:]):
        if z1 <= z0:
            continue
        piece = mf.Manifold.cylinder(z1 - z0, CORNER * r0, CORNER * r1, 6).translate((0.0, 0.0, z0))
        out = piece if out is None else out + piece
    return out if out is not None else mf.Manifold()


def _outside(g) -> list:
    """The (radius, height) points of the outside of the pot, from its bottom face to its top.

    The wall leans outwards from the bottom to the height the bead starts at, the bead grows out
    by `BEAD` at 45 degrees, so it prints without support, and rounds back in to the top face.
    """
    R, top = g["radius"], g["height"] - BEAD
    pts = [(g["bottom_radius"], CHANNEL_H), (R, top - BEAD), (R + BEAD, top)]
    for k in range(1, BEAD_SEGMENTS + 1):        # the bead, from its widest point to the top face
        a = 0.5 * math.pi * k / BEAD_SEGMENTS
        pts.append((R + BEAD * math.cos(a), top + BEAD * math.sin(a)))
    return pts


def _inside(g) -> list:
    """The (radius, height) points of the soil space: the leaning wall `wall` millimetres inwards
    on every side up to the bead, and straight up through the rim above it, so the cut leaves no
    skin behind and the plastic keeps its thickness from the floor to the top face."""
    r0, r1 = _cavity(g)
    return [(r0, CHANNEL_H + FLOOR), (r1, CHANNEL_H + g["wall_h"]), (r1, g["height"] + MARGIN)]


def _base(g) -> mf.Manifold:
    """The bottom the pot stands on, `CHANNEL_H` high under the floor and reaching into it."""
    r = g["bottom_radius"]
    return _loft(g["shape"], [(r, 0.0), (r, CHANNEL_H + MARGIN)])


def _channel_angles(g) -> list:
    """The directions, in radians, of the channels under the floor: one through every drain hole
    on the circle, or `CHANNEL_N` from the middle hole when it is the only one; none without
    holes."""
    if g["holes"] <= 0:
        return []
    n = g["holes"] - 1 if g["holes"] > 1 else CHANNEL_N
    return [2.0 * math.pi * k / n for k in range(n)]


def _channels(g) -> list:
    """The channels that lead the water from the drain holes out from under the pot: each from
    the middle of the floor to past the edge of the bottom, `CHANNEL_W` wide and `CHANNEL_H`
    high."""
    reach = CORNER * g["bottom_radius"] + MARGIN
    return [mf.Manifold.cube((reach, CHANNEL_W, CHANNEL_H + MARGIN))
            .translate((0.0, -0.5 * CHANNEL_W, -MARGIN)).rotate((0.0, 0.0, math.degrees(a)))
            for a in _channel_angles(g)]


def _drains(g) -> list:
    """The drain holes, cut through the floor from under the bed to over it: the one in the middle
    of the floor and the rest on the circle halfway to the wall; none when the pot asks for none."""
    if g["holes"] <= 0:
        return []
    cuts = [_post(0.0, 0.0, g["hole_r"], -MARGIN, CHANNEL_H + FLOOR + MARGIN)]
    for x, y in _places(max(1, g["holes"] - 1), g["ring"]):
        cuts.append(_post(x, y, g["hole_r"], -MARGIN, CHANNEL_H + FLOOR + MARGIN))
    return cuts


def _pot(g) -> mf.Manifold:
    """One printable pot: the leaning wall with its bead, a floor on its bottom, the drain holes
    in it and the channels under it."""
    body = _loft(g["shape"], _outside(g)) - _loft(g["shape"], _inside(g)) + _base(g)
    cuts = _drains(g) + _channels(g)
    return body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add) if cuts else body


def _saucer(g) -> mf.Manifold:
    """The tray that catches what runs out: the bottom of the pot plus `SAUCER_ROOM` all round,
    `SAUCER_H` high, of the same shape as the pot, on a floor of `SAUCER_FLOOR`."""
    r = g["saucer_radius"]
    outer = _loft(g["shape"], [(r + SAUCER_WALL, 0.0), (r + SAUCER_WALL, SAUCER_H)])
    inner = _loft(g["shape"], [(r, SAUCER_FLOOR), (r, SAUCER_H + MARGIN)])
    return outer - inner


def _on_bed(made) -> list:
    """The pieces side by side on the bed, the pair centred in x and y.

    The pot stands on its bottom and the saucer on its floor, both on `z = 0`, with `GAP`
    millimetres of bed between them.
    """
    boxes = [np.asarray(m.bounding_box(), dtype=float) for m in made]
    wide = [b[3] - b[0] for b in boxes]
    x = -0.5 * (sum(wide) + GAP * (len(made) - 1))
    out = []
    for m, b, w in zip(made, boxes, wide):
        out.append(m.translate((x + 0.5 * w - 0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2])))
        x += w + GAP
    return out


def plant_pot(diameter=100.0, height=90.0, taper=75.0, wall=2.0, holes=5, shape="round",
              saucer=True) -> list:
    """One printable plant pot and, when it is asked for, its saucer beside it on the bed.

    The pot is `diameter` millimetres across at the top and `height` millimetres high, narrowing
    to `taper` percent of its top diameter at the bottom, with walls of `wall` millimetres and a
    floor of `FLOOR`; the rim is rounded outward by `BEAD` millimetres. `shape` is one of
    `POT_SHAPES`: "round" or "hexagon", a six-sided pot of the same sizes across the flats. The
    floor is drained by `holes` holes of which the first stands in the middle and the rest on a
    circle halfway to the wall, with channels under the floor that lead the water out from
    under the pot. With `saucer` the pot comes with a tray of the same shape, the bottom of
    the pot plus `SAUCER_ROOM` millimetres all round and `SAUCER_H` millimetres high.

    The result is a list of one or two closed volumes, the pot first, each standing on `z = 0` and
    the pair centred in x and y with `GAP` millimetres between them. Every number is clamped to
    what prints: the taper is made smaller before the wall would lean out further than it stands
    high, and the drain holes before the floor would not hold them; anything but a number is a
    ValueError.
    """
    g = _sizes(diameter, height, taper, wall, holes, shape, saucer)
    made = [_pot(g)]
    if g["saucer"]:
        made.append(_saucer(g))
    return _on_bed(made)
