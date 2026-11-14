"""Coasters: the computational core of a flat coaster with a rim and a non-slip pattern.

No Qt in this file: the coaster is a plate pulled straight up with grooves cut in its top, so it can
be tested on its own and reused in the model and the export.

A **coaster** is one flat piece that stands under a glass: an outline of `size` millimetres across,
extruded `thickness` millimetres high. `shape` is `round` (a circle), `hexagon` or `square`; `size`
is the diameter of the circle, the distance across the flats of the hexagon and the side of the
square, so all three feel the same size in the hand. The corners of the square round with `CORNER`
millimetres, so nothing cuts a hand.

The **rim** is what makes a coaster a coaster instead of a disc: when it is on, the top is cut out
to `thickness` millimetres up to `WALL` millimetres inside the edge, so the edge stands `rim`
millimetres high as a wall that catches what runs down the side of a glass. A coaster with `rim` 0
is a plain plate, which is what you want under a mug that has to stand flat.

The top is **grooved** for grip, `GROOVE` millimetres deep and `GROOVE_WIDTH` wide, so the glass
does not slide on a wet surface and the coaster does not suck itself on the glass. `rings` are
concentric rings every `RING_STEP` millimetres from the middle outward, `grid` is lines every
`GRID_STEP` millimetres in x and in y through the middle and `none` leaves the top flat. The grooves
stop `EDGE` millimetres before the rim, so they do not break into the wall of the recess.

The piece prints **lying flat**: it is one outline extruded straight up with cuts whose walls are
vertical, so every face is horizontal or vertical, nothing overhangs and nothing needs support. The
result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the smallest and the biggest coaster: from a liquor glass to a soup bowl
SIZE_MIN, SIZE_MAX = 60.0, 150.0
#: mm the thinnest and the thickest plate under the rim
THICKNESS_MIN, THICKNESS_MAX = 3.0, 8.0
#: mm the lowest and the highest rim; 0 leaves the top of the plate flat
RIM_MIN, RIM_MAX = 0.0, 3.0
#: the outlines the plate is cut to
SHAPES = ("round", "hexagon", "square")
#: what is grooved into the top of the plate
PATTERNS = ("rings", "grid", "none")
#: mm of outline left standing as a rim round the recess
WALL = 3.0
#: mm the grooves stay inside the recess, so they do not break into its wall
EDGE = 2.0
#: mm a groove is cut into the top of the plate, and mm wide it is
GROOVE, GROOVE_WIDTH = 0.8, 1.2
#: mm between two rings, and mm between two lines of a grid
RING_STEP, GRID_STEP = 6.0, 8.0
#: mm the corners of the square are rounded with
CORNER = 4.0
#: the straight pieces a circle is approximated with
SEGMENTS = 128


def _number(value, lo, hi):
    """A number of the coaster, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a coaster must be numbers"))
    return max(lo, min(hi, v))


def _shape(value):
    """The outline of the plate; anything but one of SHAPES is the first of them."""
    name = str(value)
    return name if name in SHAPES else SHAPES[0]


def _pattern(value):
    """The pattern of grooves; anything but one of PATTERNS is the first of them."""
    name = str(value)
    return name if name in PATTERNS else PATTERNS[0]


def _width(shape, r):
    """The largest distance from the middle to the edge of the outline, in x or in y."""
    if shape == "hexagon":
        return r / math.cos(math.radians(30))       # the corners stand on the x axis
    return r


def _sizes(shape, size, thickness, rim, pattern) -> dict:
    """The clamped numbers of the coaster and the heights its top is cut to.

    The coaster lies on `z = 0` and is centred on the origin: `radius` is half the `size` and the
    radius the outline is drawn with, `height` is how high the piece stands, which is the top of the
    rim and also the floor of the recess when there is no rim. The recess leaves `WALL` millimetres
    of outline standing all round and the grooves stay `EDGE` millimetres inside that.
    """
    sh = _shape(shape)
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    r = _number(rim, RIM_MIN, RIM_MAX)
    L = _number(size, SIZE_MIN, SIZE_MAX)
    return {"shape": sh, "pattern": _pattern(pattern), "size": L, "thickness": t, "rim": r,
            "radius": 0.5 * L, "height": t + r, "outside": 2.0 * _width(sh, 0.5 * L)}


def sizes(shape="round", size=95.0, thickness=4.0, rim=1.5, pattern="rings") -> dict:
    """The numbers a coaster is made of, so a dialog can show them.

    `shape` is the outline of the plate and `size` how big it is: the diameter of the circle, the
    distance across the flats of the hexagon and the side of the square. `thickness` is the plate
    under the rim and `rim` how high the rim stands above it, so the coaster is `thickness + rim`
    high; `rim` 0 leaves the top flat. `pattern` is what is grooved into the top for grip.
    `outside` is the largest width of the coaster in x or y and `height` how high it stands. Every
    number is clamped to what prints and anything but a number is a ValueError; an unknown shape or
    pattern falls back to the default.
    """
    s = _sizes(shape, size, thickness, rim, pattern)
    return {key: s[key] for key in ("shape", "size", "thickness", "rim", "pattern", "outside",
                                    "height")}


def _outline(shape, r) -> mf.CrossSection:
    """The outline of the plate, `2 * r` across, centred on the origin.

    The hexagon stands with a flat to the front and to the back, so its corners lie on the x axis
    and `2 * r` is the distance across its flats. The corners of the square round with `CORNER`,
    which is done by shrinking the outline and growing it back; the sides stay where they were.
    """
    if shape == "hexagon":
        reach = r / math.cos(math.radians(30))
        loop = [(reach * math.cos(math.radians(60 * k)), reach * math.sin(math.radians(60 * k)))
                for k in range(6)]
        return mf.CrossSection([np.ascontiguousarray(np.asarray(loop, dtype=float))])
    if shape == "square":
        return (mf.CrossSection.square([2.0 * r, 2.0 * r], center=True)
                .offset(-CORNER, mf.JoinType.Miter)
                .offset(CORNER, mf.JoinType.Round, circular_segments=SEGMENTS))
    return mf.CrossSection.circle(r, SEGMENTS)


def _plate(s) -> mf.Manifold:
    """The plate with its rim: the outline pulled up to `height`, with the recess out of its top.

    The recess is the same outline `WALL` millimetres smaller, cut from the floor of the recess
    above the top of the piece, so what is left of the outline stands `rim` millimetres high as a
    rim round it. Without a rim the top is the plain plate.
    """
    m = mf.Manifold.extrude(_outline(s["shape"], s["radius"]), s["height"])
    if s["rim"] > 0:
        recess = mf.Manifold.extrude(_outline(s["shape"], s["radius"] - WALL), s["rim"] + 1.0)
        m = m - recess.translate((0.0, 0.0, s["thickness"]))
    return m


def _rings(s) -> mf.CrossSection:
    """Concentric grooves of `GROOVE_WIDTH` every `RING_STEP` millimetres from the middle out.

    The rings stop at `RING_STEP` millimetres inside the edge, so the outermost one is never closer
    to the rim than the others are to each other.
    """
    half = 0.5 * GROOVE_WIDTH
    rings, r = mf.CrossSection(), RING_STEP
    while r <= s["radius"] - RING_STEP:
        rings = rings + (mf.CrossSection.circle(r + half, SEGMENTS)
                         - mf.CrossSection.circle(r - half, SEGMENTS))
        r += RING_STEP
    return rings


def _grid(s) -> mf.CrossSection:
    """Grooves of `GROOVE_WIDTH` every `GRID_STEP` millimetres across the plate, in x and in y.

    The lines run through the middle, so one stands in the middle of the coaster in both
    directions, and far enough past the edge to be cut off by it instead of stopping short.
    """
    reach = 2.0 * s["radius"]
    lines, n = mf.CrossSection(), int(s["radius"] // GRID_STEP) + 1
    for i in range(-n, n + 1):
        d = i * GRID_STEP
        lines = lines + mf.CrossSection.square([GROOVE_WIDTH, reach],
                                               center=True).translate([d, 0.0])
        lines = lines + mf.CrossSection.square([reach, GROOVE_WIDTH],
                                               center=True).translate([0.0, d])
    return lines


def _grooves(s):
    """The pattern to cut, limited to the middle of the plate; None when the top stays flat.

    The cutter is intersected with the outline `WALL + EDGE` millimetres inside the edge, so a
    groove never breaks into the wall of the recess and the rim stays whole.
    """
    if s["pattern"] == "none":
        return None
    piece = _rings(s) if s["pattern"] == "rings" else _grid(s)
    return piece ^ _outline(s["shape"], s["radius"] - WALL - EDGE)


def coaster(shape="round", size=95.0, thickness=4.0, rim=1.5, pattern="rings") -> mf.Manifold:
    """One printable coaster: a flat plate with a rim and a pattern that grips the glass.

    The plate is `size` millimetres across in `shape` - a circle, a hexagon across its flats or a
    square with rounded corners - and prints `thickness` millimetres high. The `rim` cuts the top
    out to `thickness` up to `WALL` millimetres inside the edge, so the edge stands `rim`
    millimetres high and catches what runs down a glass; 0 leaves the top flat. The top is grooved
    `GROOVE` millimetres deep in `pattern`: rings, a grid or nothing.

    The result is one closed volume on `z = 0`, centred in x and y, that prints lying flat: every
    face is horizontal or vertical, so nothing overhangs and nothing needs support. Every number is
    clamped to what prints and anything but a number is a ValueError; an unknown shape or pattern
    falls back to the default.
    """
    s = _sizes(shape, size, thickness, rim, pattern)
    m = _plate(s)
    cut = _grooves(s)
    if cut is not None:
        bottom = s["thickness"] - GROOVE
        m = m - mf.Manifold.extrude(cut, s["height"] - bottom).translate((0.0, 0.0, bottom))
    return m
