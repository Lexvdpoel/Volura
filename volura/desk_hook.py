"""Desk hooks: the computational core of a hook that slides over the edge of a desk top.

No Qt in this file: the hook is one side profile extruded straight up, so it can be tested on its
own and reused in the model and the export.

A **desk hook** hangs below a desk top without a single screw: a top arm of `grip` millimetres lies
on the desk, an **upright** runs down along the desk's edge and an **under arm** clamps against the
bottom of it, where a short **bump** presses into the wood so the hook does not slide sideways.
Below the clamp a **drop** of `drop` millimetres carries a U-shaped **hook** whose opening is
`hook` millimetres wide, wide enough for the band of a bag or the headband of headphones.

The hook prints **lying on its side**: the profile is one closed loop in the x-y plane extruded
`width` millimetres high, so every wall is vertical and nothing overhangs. The upright stands
`PLAY` millimetres off the edge of the desk and the under arm `PLAY` millimetres under it, so the
hook still slides on; only the bump, which reaches `BUMP` millimetres into the desk, presses.
Outside corners round with `OUTSIDE` millimetres and inside ones with `INSIDE`, so nothing is sharp
enough to cut a hand or to catch a bag.

Every number is clamped to what prints and anything but a number is refused. The result is one
closed volume on `z = 0`, centred in x and y, `width` millimetres high.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the thinnest and the thickest desk top the hook slides over
DESK_MIN, DESK_MAX = 10.0, 50.0
#: mm the least and the most the hook reaches over the top of the desk
GRIP_MIN, GRIP_MAX = 20.0, 80.0
#: mm the least and the most the hook hangs below the desk
DROP_MIN, DROP_MAX = 20.0, 120.0
#: mm the narrowest and the widest opening of the hook
HOOK_MIN, HOOK_MAX = 15.0, 60.0
#: mm the thinnest and the thickest wall
WALL_MIN, WALL_MAX = 4.0, 10.0
#: mm the narrowest and the widest hook: a wide one holds more, a narrow one prints faster
WIDTH_MIN, WIDTH_MAX = 15.0, 50.0
#: mm of play at the edge and under the desk, so the hook still slides on
PLAY = 0.3
#: mm the bump presses into the bottom of the desk, so the hook clamps instead of rattling
BUMP = 0.2
#: mm the under arm reaches under the desk, and the x of the two sides of the bump
CLAMP = 15.0
BUMP_NEAR, BUMP_FAR = -10.0, -14.0
#: mm outside and inside corners round with
OUTSIDE, INSIDE = 2.0, 3.0
#: the most of the wall an outside corner may round with: a wall thinner than twice the radius
#: cannot carry the arc and would round away into nothing
ROUND_MAX = 0.4
#: how far the lip of the hook reaches up, as a multiple of the opening it protects
LIP_RATIO = 0.6
#: the straight pieces the rounded corners are approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the hook, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a desk hook must be numbers"))
    return max(lo, min(hi, v))


def _sizes(desk, grip, drop, hook, wall, width) -> dict:
    """The clamped numbers of the hook and the size of its profile.

    The profile is `grip + PLAY + 2 * wall + hook` millimetres across and `desk + 3 * wall + drop`
    millimetres high, and the hook is `width` millimetres thick, which is how it stands on the bed.
    """
    d = _number(desk, DESK_MIN, DESK_MAX)
    g = _number(grip, GRIP_MIN, GRIP_MAX)
    p = _number(drop, DROP_MIN, DROP_MAX)
    h = _number(hook, HOOK_MIN, HOOK_MAX)
    w = _number(wall, WALL_MIN, WALL_MAX)
    b = _number(width, WIDTH_MIN, WIDTH_MAX)
    return {"desk": d, "grip": g, "drop": p, "hook": h, "wall": w, "width": b,
            "size_x": g + PLAY + 2 * w + h, "size_y": d + 3 * w + p, "size_z": b}


def sizes(desk=25.0, grip=40.0, drop=50.0, hook=30.0, wall=6.0, width=25.0) -> dict:
    """The numbers a desk hook is made of, so a dialog can show them.

    `desk` is the thickness of the desk top, `grip` how far the hook reaches over it and `drop` how
    far it hangs below it; `hook` is the inside width of the hook's opening, `wall` the thickness of
    the plastic and `width` how wide the hook is. `size_x`, `size_y` and `size_z` are the outside of
    the hook, which is how it stands on the bed. Every number is clamped to what prints and
    anything but a number is a ValueError.
    """
    s = _sizes(desk, grip, drop, hook, wall, width)
    return {key: s[key] for key in ("desk", "grip", "drop", "hook", "wall", "width",
                                    "size_x", "size_y", "size_z")}


def _rect(x0, x1, y0, y1):
    """One rectangle of the profile, as a loop that runs counter-clockwise."""
    return np.ascontiguousarray(np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float))


def _profile(s) -> mf.CrossSection:
    """The side profile of the hook: the pieces of it as one loop, with every corner rounded.

    The x axis points away from the desk and y up, the top face of the desk is at `y = 0` and its
    edge at `x = 0`, with the desk itself at `x < 0`. The top arm lies on the desk and the upright
    runs down along its edge, `PLAY` millimetres away from it; the under arm clamps under the desk,
    `PLAY` millimetres of play below it, and the bump on top of it reaches `BUMP` millimetres into
    the desk. The drop carries the U of the hook, whose lip is `LIP_RATIO` of the opening high.

    The rectangles are unioned by the non-zero rule, so the pieces that only touch are still one
    piece. The inside corners are rounded first - growing the profile with `INSIDE` millimetres and
    shrinking it back - so the arcs fill the corners; the outside corners are then rounded the same
    way the other way around, which takes the sharpness off without moving a single wall. A thin
    wall carries less of that last arc, so it rounds with `ROUND_MAX` of its own thickness at most.
    """
    d, w = s["desk"], s["wall"]
    edge = PLAY + w                                    # the far side of the upright
    far = PLAY + 2 * w + s["hook"]                     # the far side of the hook, under its lip
    under = -d - PLAY - w                              # the under arm, at the bottom of its play
    bottom = -d - 2 * w - s["drop"]                    # the bottom of the hook
    rects = [_rect(-s["grip"], edge, 0.0, w),                        # top arm
             _rect(PLAY, edge, -d - w, w),                           # upright
             _rect(-CLAMP, edge, under, -d - PLAY),                  # under arm
             _rect(BUMP_FAR, BUMP_NEAR, -d - PLAY, -d + BUMP),       # bump
             _rect(PLAY, edge, bottom + w, -d - w),                  # drop
             _rect(PLAY, far, bottom, bottom + w),                   # bottom of the hook
             _rect(edge + s["hook"], far, bottom,
                   bottom + w + LIP_RATIO * s["hook"])]              # lip
    out = min(OUTSIDE, ROUND_MAX * w)
    rounded = (mf.CrossSection(rects, mf.FillRule.NonZero)
               .offset(INSIDE, mf.JoinType.Miter)
               .offset(-INSIDE, mf.JoinType.Round, circular_segments=SEGMENTS)
               .offset(-out, mf.JoinType.Miter)
               .offset(out, mf.JoinType.Round, circular_segments=SEGMENTS))
    # the rounding filled the corners the edge of the desk goes into: the desk and its play are
    # cut out again, square, and the bump that clamps it is put back
    desk_room = mf.CrossSection([_rect(-s["grip"] - 1.0, PLAY, -d - PLAY, 0.0)])
    bump = mf.CrossSection([_rect(BUMP_FAR, BUMP_NEAR, -d - PLAY - 0.5, -d + BUMP)])
    return rounded - desk_room + bump


def _centred(m) -> mf.Manifold:
    """The hook on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def desk_hook(desk=25.0, grip=40.0, drop=50.0, hook=30.0, wall=6.0, width=25.0) -> mf.Manifold:
    """One printable desk hook: it slides over the edge of a desk top and hangs below it.

    The top arm reaches `grip` millimetres over the desk and the upright runs down along its edge;
    the under arm clamps against the bottom of a desk `desk` millimetres thick, where a bump presses
    into it so the hook stays put, and the drop of `drop` millimetres carries a U whose opening is
    `hook` millimetres wide, for the band of a bag or the headband of headphones.

    The result is one closed volume on `z = 0`, centred in x and y and `width` millimetres high,
    that prints lying on its side: the profile is extruded straight up, so every face is vertical or
    horizontal and nothing needs support. Every number is clamped to what prints and anything but a
    number is a ValueError.
    """
    s = _sizes(desk, grip, drop, hook, wall, width)
    return _centred(mf.Manifold.extrude(_profile(s), s["width"]))
