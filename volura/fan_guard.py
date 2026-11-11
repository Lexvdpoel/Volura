"""Fan guards: the computational core of a printed grille that keeps fingers off a fan.

No Qt in this file. It gives the guard as one manifold, so it can be tested on its own and
reused in the model and the export.

A **fan guard** is a flat plate that lies on the bed and prints in one piece: `fan` millimetres
square with the corners rounded, the four screw holes of the fan at their standard spacing, and
the blade circle of the fan (`fan - 2` millimetres across) open in the middle. That opening is
filled with a **grille**, and the point of a guard is that nothing you can reach through it is
wider than `gap` millimetres, so a finger stays out where the blade spins.

Three grilles are made. **rings** are concentric rings of `bar` millimetres with `gap` between
them, running from the edge of the opening inward and held together by four straight bars along
the axes; **spokes** are eight radial bars with one ring halfway the opening, which is the
strongest grille for its weight but leaves wide openings; **grid** is a square grid of `bar`
millimetres with `gap` between the bars, clipped to the opening. Rings and grid keep every
opening at `gap`, so they are the ones to put in front of a fan a child can reach.

The **hub** is the solid disc in the middle, `fan / 4` millimetres across but never more than 25,
that the rings and the spokes start from; a grid has none. Every bar of the grille bites `GRIP`
millimetres into the plate around the opening, so the guard comes off the bed as one piece and
not as a plate with loose change in it.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: fan size in mm -> distance between the centres of its screw holes in mm
FANS = {40: 32.0, 50: 40.0, 60: 50.0, 70: 60.0, 80: 71.5, 92: 82.5, 120: 105.0, 140: 124.5}
#: fan size in mm -> diameter of its screw holes in mm
SCREW_HOLE = {40: 3.4, 50: 3.4, 60: 3.4, 70: 3.4, 80: 4.4, 92: 4.4, 120: 4.4, 140: 4.4}
#: the grilles this can make
PATTERNS = ("rings", "spokes", "grid")
#: the thinnest and the thickest plate that is worth laying on the bed
THICK_MIN, THICK_MAX = 1.0, 6.0
#: the narrowest and the widest bar of a grille
BAR_MIN, BAR_MAX = 0.8, 5.0
#: the narrowest and the widest opening between the bars; `gap` is what keeps fingers out
GAP_MIN, GAP_MAX = 2.0, 12.0
#: the fraction of the fan size that the corners of the plate are rounded to
CORNER = 0.1
#: mm between the blade circle and the opening, so there is plate left to grip the fan
OPENING_EDGE = 2.0
#: the hub as a fraction of the fan size, and the widest hub that is worth making
HUB_PART, HUB_MAX = 0.25, 25.0
#: mm the grille bites into the plate around the opening, so the guard is one piece
GRIP = 0.5
#: the radial bars of a spoked grille
SPOKES = 8
#: an opening of the grille smaller than this part of `gap` squared is filled
SLIVER = 0.25
#: the straight pieces a round part is approximated with
SEGMENTS = 96


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _number(value, lo, hi):
    """A number of the guard, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a fan guard must be numbers"))
    return _clamp(v, lo, hi)


def _fan(value):
    """The fan size, which has to be one of the standard sizes this knows the holes of."""
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = 0
    if size not in FANS:
        raise ValueError(tr("the fan size of a fan guard must be one of {sizes} mm")
                         .format(sizes=", ".join(str(s) for s in sorted(FANS))))
    return size


def _pattern(value):
    """The grille to make; anything but one of PATTERNS is refused."""
    name = str(value)
    if name not in PATTERNS:
        raise ValueError(tr("the pattern of a fan guard must be one of {patterns}")
                         .format(patterns=", ".join(PATTERNS)))
    return name


def _sizes(fan, thickness, pattern, bar, gap) -> dict:
    """The clamped numbers and the places of the features on the plate.

    The guard lies on `z = 0` and is centred on the origin: `side` is the edge of the plate and
    `corner` the radius its corners are rounded to, `radius` and `hub` are the radii of the
    opening and of the hub, and the screw holes of `hole` millimetres stand at
    (+-spacing/2, +-spacing/2).
    """
    size = _fan(fan)
    p = _pattern(pattern)
    t = _number(thickness, THICK_MIN, THICK_MAX)
    w = _number(bar, BAR_MIN, BAR_MAX)
    g = _number(gap, GAP_MIN, GAP_MAX)
    return {"fan": size, "thickness": t, "pattern": p, "bar": w, "gap": g,
            "side": float(size), "corner": CORNER * size,
            "spacing": FANS[size], "hole": SCREW_HOLE[size],
            "radius": 0.5 * (size - OPENING_EDGE), "hub": 0.5 * min(HUB_PART * size, HUB_MAX)}


def sizes(fan=40, thickness=2.0, pattern="rings", bar=1.6, gap=4.0) -> dict:
    """The numbers a fan guard is made of, so a dialog can show them.

    `fan` is the size of the fan in millimetres and `thickness` the height of the plate, `pattern`
    the grille and `bar` and `gap` the width of its bars and the openings between them.
    `hole_spacing` is the distance between the centres of the screw holes and `screw_hole` their
    diameter, `opening` is the diameter of the hole the grille fills and `hub` the diameter of
    the solid disc in its middle; `open_area` is the area of the openings of the grille in mm².
    """
    g = _sizes(fan, thickness, pattern, bar, gap)
    return {"fan": g["fan"], "thickness": g["thickness"], "pattern": g["pattern"],
            "bar": g["bar"], "gap": g["gap"], "hole_spacing": g["spacing"],
            "screw_hole": g["hole"], "opening": 2.0 * g["radius"], "hub": 2.0 * g["hub"],
            "open_area": _open_area(g)}


def _outline(g) -> mf.CrossSection:
    """The outside of the plate: `fan` millimetres square with the corners rounded to `corner`."""
    return (mf.CrossSection.square([g["side"], g["side"]], center=True)
            .offset(-g["corner"], mf.JoinType.Miter)
            .offset(g["corner"], mf.JoinType.Round, circular_segments=SEGMENTS))


def _plate(g) -> mf.CrossSection:
    """The plate on its own: the outline with the opening and the four screw holes cut out."""
    plate = _outline(g) - mf.CrossSection.circle(g["radius"], SEGMENTS)
    for x in (-0.5 * g["spacing"], 0.5 * g["spacing"]):
        for y in (-0.5 * g["spacing"], 0.5 * g["spacing"]):
            plate = plate - mf.CrossSection.circle(0.5 * g["hole"], SEGMENTS).translate([x, y])
    return plate


def _bar(g, length) -> mf.CrossSection:
    """A straight bar of the grille, `bar` wide and `length` long, along Y through the middle."""
    return mf.CrossSection.square([g["bar"], length], center=True)


def _rings(g) -> mf.CrossSection:
    """Concentric rings from the edge of the opening inward, held by four bars along the axes.

    The rings step inward by `bar + gap`, so the opening between two of them is `gap` wide, and
    the last one is the ring that still reaches past the hub. The first starts `GRIP` millimetres
    out in the plate, so it grows into the plate instead of only touching it.
    """
    w, step = g["bar"], g["bar"] + g["gap"]
    rings, r = mf.CrossSection(), g["radius"] + GRIP
    while r > g["hub"]:
        ring = mf.CrossSection.circle(r, SEGMENTS)
        if r > w:
            ring = ring - mf.CrossSection.circle(r - w, SEGMENTS)
        rings = rings + ring
        r -= step
    hold = _bar(g, 2.0 * (g["radius"] + GRIP))
    return rings + hold + hold.rotate(90) + mf.CrossSection.circle(g["hub"], SEGMENTS)


def _spokes(g) -> mf.CrossSection:
    """The hub with eight radial bars and one ring of `bar` halfway the opening."""
    length = g["radius"] + 2.0 * GRIP
    spoke = _bar(g, length).translate([0.0, 0.5 * length - GRIP])
    bars = mf.CrossSection()
    for i in range(SPOKES):                 # a spoke runs from the hub out, so all the way round
        bars = bars + spoke.rotate(i * 360.0 / SPOKES)
    half = 0.5 * g["radius"]
    ring = (mf.CrossSection.circle(half, SEGMENTS)
            - mf.CrossSection.circle(half - g["bar"], SEGMENTS))
    return bars + ring + mf.CrossSection.circle(g["hub"], SEGMENTS)


def _grid(g) -> mf.CrossSection:
    """A square grid of bars of `bar` with openings of `gap`, over the whole opening."""
    step, reach = g["bar"] + g["gap"], g["radius"] + GRIP
    bars, n = mf.CrossSection(), int(reach // step) + 1
    for i in range(-n, n + 1):
        d = i * step
        bars = bars + mf.CrossSection.square([g["bar"], 2.0 * reach],
                                             center=True).translate([d, 0.0])
        bars = bars + mf.CrossSection.square([2.0 * reach, g["bar"]],
                                             center=True).translate([0.0, d])
    return bars


def _grille(g) -> mf.CrossSection:
    """The grille that fills the opening, cut to `GRIP` into the plate so it cannot reach a hole.

    The bars are made across the whole guard, but the plate around the opening is only
    `OPENING_EDGE` millimetres wide, so what is left of the grille ends halfway in it. Where the
    edge of the opening cuts a square of a grid off to a sliver, that sliver is filled: an
    opening smaller than `SLIVER` of a full one lets no air through worth having and prints as
    a crumb.
    """
    if g["pattern"] == "rings":
        piece = _rings(g)
    elif g["pattern"] == "spokes":
        piece = _spokes(g)
    else:
        piece = _grid(g)
    rim = mf.CrossSection.circle(g["radius"] + GRIP, SEGMENTS)
    piece = piece ^ rim
    least = SLIVER * g["gap"] ** 2
    opening = mf.CrossSection.circle(g["radius"], SEGMENTS)
    crumbs = [h for h in (opening - piece).decompose() if h.area() < least]
    return mf.CrossSection.batch_boolean([piece] + crumbs, mf.OpType.Add) ^ rim


def _section(g) -> mf.CrossSection:
    """The section of the whole guard: the plate with the grille in its opening."""
    return _plate(g) + _grille(g)


def _open_area(g) -> float:
    """mm² of openings in the grille: the part of the opening that is not plastic."""
    opening = mf.CrossSection.circle(g["radius"], SEGMENTS)
    return opening.area() - (opening ^ _grille(g)).area()


def fan_guard(fan=40, thickness=2.0, pattern="rings", bar=1.6, gap=4.0) -> mf.Manifold:
    """One printable fan guard: a plate with the screw holes of a fan and a grille over the blade.

    The guard lies on `z = 0` and is centred on the origin, `fan` millimetres square and
    `thickness` millimetres high; `fan` is the size of the fan and has to be one of the sizes in
    `FANS`, whose screw holes of `SCREW_HOLE` millimetres are cut at their standard spacing. The
    middle is opened to the blade circle of `fan - 2` millimetres and filled with `pattern`:
    `rings` and `grid` leave no opening wider than `gap`, `spokes` is the lightest of the three.
    `bar` is the width of every bar and the hub in the middle is a solid disc of a quarter of the
    fan size. Every number is clamped to what prints; an unknown fan or pattern is a ValueError.
    """
    g = _sizes(fan, thickness, pattern, bar, gap)
    return _section(g).extrude(g["thickness"])
