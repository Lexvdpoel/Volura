"""Filament clips: the computational core of a clip that grips the rim of a filament spool.

No Qt in this file: the clip is one flat outline pulled straight up with holes cut through it, so it
can be tested on its own and reused in the model and the export.

A **filament clip** keeps the loose end of a filament from unwinding: it pushes onto the flange of a
spool and the end of the filament is pushed sideways into a hole, where a narrower slit holds it.
`flange` is the thickness of the rim it grips, `filament` the diameter of the filament - 1.75 or
2.85 millimetres - and `count` how many holes the clip has, so one clip holds several ends.

The clip grips with a **U** that opens downward: two jaws of `WALL` millimetres standing `JAW`
millimetres high under a bridge of `WALL` millimetres, that straddles the flange. The gap between
the jaws is exactly `flange` wide, and a `BUMP` millimetre grip bump on the inside of each jaw
pinches into that gap so the clip does not slide off. The outer bottom corner of each jaw is
chamfered `CHAMFER` millimetres, so the clip slides onto the flange instead of digging into it.

The **tab** hangs under the bridge, to the right of the clip, `TAB` millimetres high with its far
corners rounded `CORNER` millimetres. The **holes** are cut through the whole clip `CLEAR`
millimetres wider than the filament, `HOLE_EDGE` millimetres under the bridge and `HOLE_STEP`
millimetres apart, and every hole opens to the bottom edge of the tab through a slit `SLIT` times
the filament diameter: the filament is pushed in sideways through the slit, and the slit - narrower
than the hole - is what holds it.

The piece prints **lying flat**: one outline extruded `thickness` millimetres high with holes whose
walls are vertical, so every face is horizontal or vertical, nothing overhangs and nothing needs
support. The result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the thinnest and the thickest spool rim the clip grips
FLANGE_MIN, FLANGE_MAX = 1.0, 6.0
#: mm the thinnest and the thickest clip
THICKNESS_MIN, THICKNESS_MAX = 4.0, 10.0
#: the fewest and the most filament holes on one clip
COUNT_MIN, COUNT_MAX = 1, 3
#: the filament diameters the clip holds, in millimetres
FILAMENTS = ("1.75", "2.85")
#: mm the wall of a jaw, and mm thick the bridge that joins them
WALL = 2.2
#: mm high the jaws stand under the bridge
JAW = 14.0
#: mm high the tab hangs under the bridge
TAB = 8.0
#: mm the middle of a hole stands above the bottom edge of the tab
HOLE_EDGE = 4.0
#: mm between two holes, and mm the first one stands from the outer face of the clip
HOLE_STEP = 6.0
#: mm a grip bump pinches into the gap, whose base runs from BUMP_LOW to BUMP_HIGH millimetres up
BUMP, BUMP_LOW, BUMP_HIGH = 0.3, 1.0, 4.0
#: mm the outer bottom corner of a jaw is chamfered, so the clip slides onto the flange
CHAMFER = 1.0
#: mm the far corners of the tab are rounded with
CORNER = 2.0
#: mm a hole is bigger than the filament, and times the filament diameter a slit is wide
CLEAR, SLIT = 0.4, 0.6
#: the straight pieces a circle is approximated with
SEGMENTS = 128


def _number(value, lo, hi):
    """A number of the clip, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a filament clip must be numbers"))
    return max(lo, min(hi, v))


def _filament(value):
    """The filament the clip holds; anything but one of FILAMENTS is the first of them."""
    name = str(value)
    return name if name in FILAMENTS else FILAMENTS[0]


def _count(value):
    """How many holes the tab has, as a whole number of holes."""
    return int(round(_number(value, COUNT_MIN, COUNT_MAX)))


def _sizes(flange, filament, count, thickness) -> dict:
    """The clamped numbers of the clip and the faces its outline is drawn between.

    The outline is drawn with its left face on `x = 0` and its bottom on `y = 0`: `left` and `right`
    are the inner faces of the two jaws, so the gap that takes the flange is `flange` wide, and
    `outer` is the outer face of the right jaw. `tab` is the far edge of the tab, which is `length`
    millimetres from `x = 0`; `top` is the top of the bridge and also the top of the tab, and
    `tab_bottom` the edge the slits open through.
    """
    f = _number(flange, FLANGE_MIN, FLANGE_MAX)
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    n = _count(count)
    name = _filament(filament)
    d = float(name)
    outer = 2.0 * WALL + f
    return {"flange": f, "filament": name, "count": n, "thickness": t, "diameter": d,
            "hole": d + CLEAR, "slit": SLIT * d,
            "left": WALL, "right": WALL + f, "outer": outer, "tab": outer + HOLE_STEP * (n + 1),
            "top": JAW + WALL, "tab_bottom": JAW + WALL - TAB,
            "length": outer + HOLE_STEP * (n + 1), "width": JAW + WALL}


def sizes(flange=2.5, filament="1.75", count=1, thickness=5.0) -> dict:
    """The numbers a filament clip is made of, so a dialog can show them.

    `flange` is the thickness of the spool rim the clip grips and `filament` the diameter of the
    filament, "1.75" or "2.85" millimetres; `count` is how many holes the clip has and `thickness`
    how high it prints. `hole` is the diameter of a hole and `slit` the width of the slit that opens
    it to the edge of the tab; `length` and `width` are the outside of the clip, which lies flat.
    Every number is clamped to what prints and anything but a number is a ValueError; an unknown
    filament falls back to the default.
    """
    s = _sizes(flange, filament, count, thickness)
    return {key: s[key] for key in ("flange", "filament", "count", "thickness", "hole", "slit",
                                    "length", "width")}


def _arc(cx, cy, r, a0, a1):
    """The middle points of a round from `a0` to `a1` degrees, the ends of it left to the caller."""
    steps = SEGMENTS // 4
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / steps)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / steps)))
            for i in range(1, steps)]


def _outline(s) -> mf.CrossSection:
    """The clip as one outline: the U that grips the flange, with the tab next to it.

    The outline runs counter-clockwise: along the chamfered bottom of the left jaw, up the inside of
    the U over both grip bumps, down the outside of the right jaw and along the bottom of the tab to
    its rounded far corners, and back along the top, which is flat from the far end of the tab to
    the left face of the clip.
    """
    left, right = s["left"], s["right"]
    outer, tab = s["outer"], s["tab"]
    top, bottom = s["top"], s["tab_bottom"]
    tip = 0.5 * (BUMP_LOW + BUMP_HIGH)
    loop = [(CHAMFER, 0.0), (left, 0.0),
            (left, BUMP_LOW), (left + BUMP, tip), (left, BUMP_HIGH),
            (left, JAW), (right, JAW),
            (right, BUMP_HIGH), (right - BUMP, tip), (right, BUMP_LOW),
            (right, 0.0), (outer - CHAMFER, 0.0), (outer, CHAMFER),
            (outer, bottom), (tab - CORNER, bottom)]
    loop += _arc(tab - CORNER, bottom + CORNER, CORNER, -90.0, 0.0)
    loop += [(tab, top - CORNER)]
    loop += _arc(tab - CORNER, top - CORNER, CORNER, 0.0, 90.0)
    loop += [(outer, top), (0.0, top), (0.0, CHAMFER)]
    return mf.CrossSection([np.ascontiguousarray(np.asarray(loop, dtype=float))])


def _holes(s) -> mf.CrossSection:
    """The holes of the tab and the slit every one of them opens through, as one outline to cut.

    A hole is `hole` millimetres across and stands `HOLE_STEP` millimetres from its neighbour,
    `HOLE_EDGE` millimetres under the bridge. The slit is `slit` millimetres wide and runs from the
    middle of the hole past the bottom edge of the tab - a millimetre past it, so it is cut open
    instead of stopping short of the edge.
    """
    holes = mf.CrossSection()
    low = s["tab_bottom"] - 1.0
    for k in range(s["count"]):
        x = s["outer"] + HOLE_STEP * (k + 1)
        y = s["top"] - HOLE_EDGE
        holes = holes + mf.CrossSection.circle(0.5 * s["hole"], SEGMENTS).translate([x, y])
        holes = holes + mf.CrossSection.square([s["slit"], y - low],
                                               center=True).translate([x, 0.5 * (y + low)])
    return holes


def filament_clip(flange=2.5, filament="1.75", count=1, thickness=5.0) -> mf.Manifold:
    """One printable filament clip: a clip for the rim of a spool that holds the end of the thread.

    Two jaws of `WALL` millimetres straddle the flange in a gap as wide as `flange`, held there by
    a `BUMP` millimetre grip bump in each jaw, and are joined by a bridge under which the tab
    hangs. The tab carries `count` holes of `filament` plus `CLEAR` millimetres through the whole
    clip, each opening to its edge through a slit that is narrower than the hole.

    The result is one closed volume on `z = 0`, centred in x and y, `thickness` millimetres high,
    that prints lying flat: every face is horizontal or vertical, so nothing overhangs and nothing
    needs support. Every number is clamped to what prints and anything but a number is a ValueError;
    an unknown filament falls back to the default.
    """
    s = _sizes(flange, filament, count, thickness)
    m = mf.Manifold.extrude(_outline(s), s["thickness"])
    m = m - mf.Manifold.extrude(_holes(s), s["thickness"] + 2.0).translate((0.0, 0.0, -1.0))
    return m.translate((-0.5 * s["length"], -0.5 * s["width"], 0.0))
