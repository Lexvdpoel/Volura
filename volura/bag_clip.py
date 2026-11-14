"""Bag clips: the computational core of a spring clip that closes a bag.

No Qt in this file: the clip is one flat outline pulled straight up, so it can be tested on its own
and reused in the model and the export.

A **bag clip** is one flat piece that closes a bag: two straight arms lie side by side along x,
`length` millimetres long and `arm` millimetres wide each, with `gap` millimetres between them - the
bag goes in that gap. The left end joins the arms with a **spring loop**: a half ring of inner radius
`gap / 2 + loop` and a wall of `0.6 * arm`, so squeezing the arms opens the loop and the gap bites
on the bag. The right end is a **latch**: the lower arm runs past the upper one and up into an
upright that rises an arm's width over it and bends back with a 45 degree ramp, so the upper arm
snaps under that lip and the clip stays closed. The clip is modelled closed, as it prints and as it
is used: the upper arm stops `HOOK_GAP` millimetres short of the upright and stands `HOOK_LIFT`
millimetres under the lip, so nothing fuses and the piece stays one volume through the loop.

Optional **teeth** of `TOOTH` millimetres stand on the inner faces of both arms every `TOOTH_PITCH`
millimetres and interleave, so the clip holds a bag instead of sliding off it.

The piece prints **lying flat**: the outline is extruded `thickness` millimetres high, so every wall
is vertical, nothing overhangs and nothing needs support. The outside corners round with `CORNER`
millimetres, or less where the loop wall is thinner than twice that and the rounding would eat it
away. Every number is clamped to what prints: the gap grows to what the teeth need when they are on,
and the loop is never smaller than an arm, so the neck from the loop to an arm is never a point. The
result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the shortest and the longest clip: long enough to close a bag of coffee
LENGTH_MIN, LENGTH_MAX = 50.0, 200.0
#: mm the narrowest and the widest arm; the loop wall, the neck and the hook are made of it
ARM_MIN, ARM_MAX = 3.0, 10.0
#: mm the thinnest plate that still springs back and the thickest that is still a clip
THICKNESS_MIN, THICKNESS_MAX = 4.0, 12.0
#: mm the narrowest and the widest gap the bag goes in
GAP_MIN, GAP_MAX = 0.4, 3.0
#: mm the smallest and the biggest spring loop
LOOP_MIN, LOOP_MAX = 2.0, 10.0
#: mm the gap has to be with the teeth on, so the tips of two arms do not touch
TEETH_GAP = 1.4
#: mm a tooth stands out of the inner face of an arm, and mm between two teeth of one arm
TOOTH, TOOTH_PITCH = 0.6, 3.0
#: mm the first tooth stands from the loop and the last one from the end of the arm
TOOTH_START, TOOTH_END = 3.0, 5.0
#: mm the upper arm stops short of the hook upright and stands under its lip
HOOK_GAP, HOOK_LIFT = 0.4, 0.3
#: mm the outside corners round with, when there is plastic enough around them to keep
CORNER = 1.0
#: mm a piece reaches into the piece it meets, so the clip is one whole and not a touch along a line
OVERLAP = 0.3
#: degrees the loop overshoots a half circle at both ends, for the same reason
OVER_ANGLE = 5.0
#: the straight pieces a ring is approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the clip, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a bag clip must be numbers"))
    return max(lo, min(hi, v))


def _sizes(length, arm, thickness, gap, loop, teeth) -> dict:
    """The clamped numbers of the clip and the places of its arms, its loop and its latch.

    The clip lies on the bed and is centred on the origin in plan. The arms run along x from the
    neck at `neck` to `length`, the upper one above `y = gap / 2` and the lower one below
    `y = -gap / 2`; the loop is the half ring round the origin between `inner_radius` and
    `outer_radius`, and the latch stands at `length` with its lip over the upper arm.
    """
    L = _number(length, LENGTH_MIN, LENGTH_MAX)
    a = _number(arm, ARM_MIN, ARM_MAX)
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    g = _number(gap, GAP_MIN, GAP_MAX)
    lp = _number(loop, LOOP_MIN, LOOP_MAX)
    on = bool(teeth)
    if on:
        g = max(g, TEETH_GAP)                     # the teeth of the two arms bite past each other
    lp = max(lp, a)                               # the neck from the loop to an arm is no point
    r_i = 0.5 * g + lp
    r_o = r_i + 0.6 * a
    n = 2.0 * lp
    low = -r_o                                    # the loop is the lowest thing on the clip
    high = max(0.5 * g + 2.0 * a + HOOK_LIFT, r_o)     # the lip of the latch, or the loop
    return {"length": L, "arm": a, "thickness": t, "gap": g, "loop": lp, "teeth": on,
            "inner_radius": r_i, "outer_radius": r_o, "neck": n,
            "outside_length": L + a + r_o, "outside_width": high - low}


def sizes(length=100.0, arm=5.0, thickness=6.0, gap=1.0, loop=4.0, teeth=True) -> dict:
    """The numbers a bag clip is made of, so a dialog can show them.

    `length` is how long the arms are and `arm` how wide each of them is; `thickness` is how high the
    piece prints and `gap` is the room between the arms the bag goes in. `loop` is the size of the
    spring loop that joins the arms and `teeth` whether the inner faces are serrated. `outside_length`
    and `outside_width` are the whole clip. Every number is clamped to what prints and anything but a
    number is a ValueError.
    """
    s = _sizes(length, arm, thickness, gap, loop, teeth)
    return {key: s[key] for key in ("length", "arm", "thickness", "gap", "loop", "teeth",
                                    "outside_length", "outside_width")}


def _array(loop) -> np.ndarray:
    """A loop of points as the array a `CrossSection` wants."""
    return np.ascontiguousarray(np.asarray(loop, dtype=float))


def _rect(x0, x1, y0, y1) -> list:
    """A rectangle from (`x0`, `y0`) to (`x1`, `y1`), counter-clockwise."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _ring(r_i, r_o) -> list:
    """The left half ring between the radii `r_i` and `r_o` round the origin, as one loop.

    The loop overshoots a half circle by `OVER_ANGLE` degrees at both ends, so it runs into the two
    necks instead of touching them along a line.
    """
    half = SEGMENTS // 2
    step = (math.pi + 2.0 * math.radians(OVER_ANGLE)) / half
    start = math.pi / 2.0 - math.radians(OVER_ANGLE)
    outer = [(r_o * math.cos(start + k * step), r_o * math.sin(start + k * step))
             for k in range(half + 1)]
    inner = [(r_i * math.cos(start + k * step), r_i * math.sin(start + k * step))
             for k in range(half, -1, -1)]
    return outer + inner


def _outline(s) -> mf.CrossSection:
    """The flat clip: its two arms, the spring loop, the two necks and the latch, with its outside
    corners rounded.

    The pieces overlap a little where they meet, because a clip whose neck only touches its loop is
    not one piece. The corners round by shrinking the outline and growing it back, which rounds every
    corner that sticks out and leaves the straight sides where they were; the radius is taken smaller
    when the loop wall would not survive a whole `CORNER`.
    """
    g, a, L = s["gap"], s["arm"], s["length"]
    r_i, r_o, n = s["inner_radius"], s["outer_radius"], s["neck"]
    e = OVERLAP
    loops = [
        _rect(n, L - HOOK_GAP, 0.5 * g, 0.5 * g + a),                  # the upper arm
        _rect(n, L, -0.5 * g - a, -0.5 * g),                           # the lower arm
        _ring(r_i, r_o),                                               # the spring loop
        [(n + e, 0.5 * g), (n + e, 0.5 * g + a), (0.0, r_o),
         (0.0, r_i)],                                                  # the neck to the top arm
        [(n + e, -0.5 * g - a), (n + e, -0.5 * g), (0.0, -r_i),
         (0.0, -r_o)],                                                 # the neck to the bottom arm
        _rect(L, L + a, -0.5 * g - a, 0.5 * g + 2.0 * a + HOOK_LIFT),  # the upright of the hook
        [(L - 2.0 * a, 0.5 * g + a + HOOK_LIFT), (L + a, 0.5 * g + a + HOOK_LIFT),
         (L + a, 0.5 * g + 2.0 * a + HOOK_LIFT),                       # the lip over the arm; its
         (L - a, 0.5 * g + 2.0 * a + HOOK_LIFT)],                      # left end is the 45° ramp
    ]
    cs = mf.CrossSection([_array(p) for p in loops], mf.FillRule.NonZero)
    r = min(CORNER, 0.45 * (r_o - r_i))
    return cs.offset(-r, mf.JoinType.Miter, 2.0).offset(r, mf.JoinType.Round, 2.0)


def _grip(s) -> list:
    """The teeth that bite into the bag, one triangle per tooth.

    A tooth stands `TOOTH` millimetres out of the inner face of its arm and is based a little inside
    that face, so it grows out of the arm instead of touching it along a line. The teeth of the two
    arms are half a pitch apart, so they pass each other when the clip is squeezed.
    """
    if not s["teeth"]:
        return []
    g, n, L = s["gap"], s["neck"], s["length"]
    over = TOOTH + OVERLAP                # how far up the arm the base of a tooth reaches
    made = []
    x = n + TOOTH_START
    while x <= L - TOOTH_END:
        made.append([(x, 0.5 * g - TOOTH), (x + over, 0.5 * g + over), (x - over, 0.5 * g + over)])
        u = x + 0.5 * TOOTH_PITCH
        if u <= L - TOOTH_END:
            made.append([(u, -0.5 * g + TOOTH), (u - over, -0.5 * g - over),
                         (u + over, -0.5 * g - over)])
        x += TOOTH_PITCH
    return made


def _centred(m) -> mf.Manifold:
    """The clip on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def bag_clip(length=100.0, arm=5.0, thickness=6.0, gap=1.0, loop=4.0,
             teeth=True) -> mf.Manifold:
    """One printable bag clip: a flat spring clip that closes a bag.

    The two arms are `length` millimetres long and `arm` wide, `gap` millimetres apart, and the bag
    goes between them. The `loop` joins them at one end as a half ring whose wall flexes; at the
    other end the lower arm rises into a hook whose lip snaps over the upper arm, which stops just
    short of the upright and just under the lip so the clip is one piece that still opens. With
    `teeth` the inner faces are serrated for grip.

    The result is one closed volume on `z = 0`, centred in x and y, that prints lying flat: it is one
    outline extruded `thickness` millimetres high, so every wall is vertical and nothing overhangs.
    Every number is clamped to what prints and anything but a number is a ValueError.
    """
    s = _sizes(length, arm, thickness, gap, loop, teeth)
    m = mf.Manifold.extrude(_outline(s), s["thickness"])
    loops = _grip(s)
    if loops:
        m = m + mf.Manifold.extrude(mf.CrossSection([_array(t) for t in loops],
                                                    mf.FillRule.NonZero), s["thickness"])
    return _centred(m)
