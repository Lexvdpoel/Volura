"""Screw-top jars: the computational core of a round jar with a lid that screws on.

No Qt in this file. It gives the jar and its lid as closed volumes, so they can be tested on their
own and reused in the model and the export.

A **screw-top jar** is a round cup of `diameter` millimetres inside and `height` millimetres deep
above its floor of `FLOOR`, with walls of `wall` millimetres. Its top `thread_len` millimetres
carry an outside thread of `pitch` millimetres (the ISO profile of `threads.thread_solid`): the
root of the thread is the outside of the wall, so the crests stand out by the depth of the thread
and the inside stays `diameter` wide all the way up to the rim. The thread is bevelled at both
ends, so it starts without a sharp edge and its first turn finds the lid.

The **lid** is a disc of `LID_TOP` millimetres on a skirt with the matching inside thread, cut
`clearance` millimetres larger all round so it turns freely after printing; its skirt is `wall`
millimetres thick outside that thread and as deep as the thread is long, so the disc comes to rest
on the rim. `grip` ribs of `RIB_DEPTH` by `RIB_WIDTH` millimetres stand on the outside of the skirt
so it turns in a wet hand; 0 leaves the skirt smooth.

Both print **without support**: the jar stands on its floor and the lid lies on its top, upside
down, so the threads run round a vertical axis and their flanks lean as every printed bolt's do.
Every number is clamped to what prints. The jar and the lid stand on `z = 0` side by side, `GAP`
millimetres apart and centred together in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import threads
from .i18n import tr

#: mm inside the narrowest and the widest jar
D_MIN, D_MAX = 20.0, 150.0
#: mm deep the shallowest and the deepest jar is inside, above its floor
HEIGHT_MIN, HEIGHT_MAX = 15.0, 200.0
#: mm the thinnest wall that holds a thread and the thickest worth printing
WALL_MIN, WALL_MAX = 1.6, 4.0
#: mm the finest and the coarsest thread pitch: coarse threads print best on a jar
PITCH_MIN, PITCH_MAX = 2.0, 5.0
#: mm the shortest and the longest thread
THREAD_MIN, THREAD_MAX = 6.0, 20.0
#: mm of play between the threads of jar and lid
CLEARANCE_MIN, CLEARANCE_MAX = 0.1, 0.6
#: the most grip ribs on the lid, and the fewest when there are any
GRIP_MIN, GRIP_MAX = 12, 48
#: mm thick the floor of the jar and the top of the lid are
FLOOR, LID_TOP = 2.0, 2.0
#: mm of jar below the thread at least, so the thread never runs into the floor
BELOW_THREAD = 3.0
#: mm deep (outward) and wide a grip rib is
RIB_DEPTH, RIB_WIDTH = 1.0, 2.0
#: how deep an ISO thread is, as a part of its pitch (the radius from root to crest)
THREAD_DEPTH = 1.0825 / 2.0
#: how far the 45 degree collar under the thread stands out of the wall, as a part of the pitch:
#: past the bottom of the bevelled thread, which stands out by 0.041 of it
COLLAR = 0.1
#: mm the wall under the thread stands inside the root of the thread, so the two never share a face
CORE_IN = 0.05
#: mm of bed between the jar and the lid
GAP = 5.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5


def _number(value, lo, hi):
    """A number of the jar, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a screw-top jar must be numbers"))
    return max(lo, min(hi, v))


def _sizes(diameter, height, wall, pitch, thread_len, clearance, grip) -> dict:
    """The clamped numbers of the jar and the radii it is made of.

    `inner` is the radius of the inside, `outer` the outside of the wall and the root of the
    thread, and `major` the radius of the crests; `lid_outer` is the outside of the lid's skirt,
    ribs not counted. The thread is never longer than the jar is deep less `BELOW_THREAD`.
    """
    D = _number(diameter, D_MIN, D_MAX)
    H = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    w = _number(wall, WALL_MIN, WALL_MAX)
    p = _number(pitch, PITCH_MIN, PITCH_MAX)
    L = _number(thread_len, THREAD_MIN, min(THREAD_MAX, H - BELOW_THREAD))
    c = _number(clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    n = int(round(_number(grip, 0, GRIP_MAX)))
    if 0 < n < GRIP_MIN:
        n = GRIP_MIN
    inner = 0.5 * D
    outer = inner + w
    major = outer + THREAD_DEPTH * p
    return {"diameter": D, "height": H, "wall": w, "pitch": p, "thread_len": L,
            "clearance": c, "grip": n, "inner": inner, "outer": outer, "major": major,
            "lid_outer": major + c + w}


def sizes(diameter=60.0, height=60.0, wall=2.0, pitch=3.0, thread_len=10.0, clearance=0.3,
          grip=24) -> dict:
    """The numbers a screw-top jar is made of, so a dialog can show them.

    `diameter` and `height` are the inside of the jar, `wall` how thick its plastic is, `pitch`
    and `thread_len` the thread it closes with, `clearance` the play between the threads and
    `grip` how many ribs the lid has. `jar_outside` is the diameter over the crests of the thread,
    `lid_outside` the diameter of the lid over its ribs, `closed_height` how high the jar stands
    with its lid on and `volume` what it holds in millilitres. Every number is clamped to what
    prints; anything but a number is a ValueError.
    """
    g = _sizes(diameter, height, wall, pitch, thread_len, clearance, grip)
    rib = RIB_DEPTH if g["grip"] else 0.0
    return {"diameter": g["diameter"], "height": g["height"], "wall": g["wall"],
            "pitch": g["pitch"], "thread_len": g["thread_len"], "clearance": g["clearance"],
            "grip": g["grip"], "jar_outside": 2.0 * g["major"],
            "lid_outside": 2.0 * (g["lid_outer"] + rib),
            "closed_height": FLOOR + g["height"] + LID_TOP,
            "volume": math.pi * g["inner"] ** 2 * g["height"] / 1000.0}


def _segments(radius) -> int:
    """How many straight pieces a circle of `radius` is made of: about one per millimetre."""
    return int(max(64, min(256, round(2.0 * math.pi * radius))))


def _cylinder(radius, z0, z1, segments) -> mf.Manifold:
    return mf.Manifold.cylinder(z1 - z0, radius, radius, segments).translate((0.0, 0.0, z0))


def _jar(g) -> mf.Manifold:
    """The jar standing on its floor: the cup, the thread round its top, the inside cut out."""
    seg = _segments(g["major"])
    rim = FLOOR + g["height"]
    start = rim - g["thread_len"]
    cup = _cylinder(g["outer"], 0.0, start + MARGIN, seg)
    # the wall goes on up to the rim just inside the root of the thread: where the bevels at the
    # ends of a coarse thread dip deeper than the wall is thick, the wall stays whole
    cup = cup + _cylinder(g["outer"] - CORE_IN, start, rim, seg)
    thread =threads.thread_solid(2.0 * g["major"], g["pitch"], g["thread_len"],
                                  segments=seg).translate((0.0, 0.0, start))
    # the bevel at the start of the thread still stands a little past the wall; a collar that
    # widens at 45 degrees from the wall carries it, so nothing below the thread hangs flat
    lip = COLLAR * g["pitch"]
    collar = mf.Manifold.cylinder(lip, g["outer"], g["outer"] + lip, seg) \
        .translate((0.0, 0.0, start - lip))
    inside = _cylinder(g["inner"], FLOOR, rim + MARGIN, seg)
    return (cup + collar + thread) - inside


def _lid(g) -> mf.Manifold:
    """The lid as it prints: its top on the bed, the skirt with the inside thread above it."""
    seg = _segments(g["lid_outer"])
    L, top = g["thread_len"], LID_TOP
    skirt = _cylinder(g["lid_outer"], 0.0, top + L, seg)
    cut = threads.thread_solid(2.0 * g["major"], g["pitch"], L + MARGIN, internal=True,
                               clearance=g["clearance"], segments=seg) \
        .translate((0.0, 0.0, top))
    lid = skirt - cut
    if g["grip"]:
        rib = mf.Manifold.cube((RIB_DEPTH + MARGIN, RIB_WIDTH, top + L)) \
            .translate((g["lid_outer"] - MARGIN, -0.5 * RIB_WIDTH, 0.0))
        ribs = [rib.rotate((0.0, 0.0, 360.0 * k / g["grip"])) for k in range(g["grip"])]
        lid = lid + mf.Manifold.batch_boolean(ribs, mf.OpType.Add)
    return lid


def _on_bed(made) -> list:
    """The pieces side by side on the bed, `GAP` apart, the pair centred in x and y."""
    boxes = [np.asarray(m.bounding_box(), dtype=float) for m in made]
    wide = [b[3] - b[0] for b in boxes]
    x = -0.5 * (sum(wide) + GAP * (len(made) - 1))
    out = []
    for m, b, w in zip(made, boxes, wide):
        out.append(m.translate((x - b[0], -0.5 * (b[1] + b[4]), -b[2])))
        x += w + GAP
    return out


def screw_jar(diameter=60.0, height=60.0, wall=2.0, pitch=3.0, thread_len=10.0, clearance=0.3,
              grip=24) -> list:
    """A printable screw-top jar and its lid, side by side on the bed.

    The jar is `diameter` millimetres wide and `height` deep inside, above a floor of `FLOOR`,
    with walls of `wall` millimetres and an outside thread of `pitch` over its top `thread_len`
    millimetres. The lid has the matching inside thread with `clearance` millimetres of play and
    `grip` ribs round it, and lies on its top so it prints without support.

    The result is a list of two closed volumes, the jar first, each standing on `z = 0` and the
    pair centred in x and y with `GAP` millimetres between them. Every number is clamped to what
    prints and anything but a number is a ValueError.
    """
    g = _sizes(diameter, height, wall, pitch, thread_len, clearance, grip)
    return _on_bed([_jar(g), _lid(g)])
