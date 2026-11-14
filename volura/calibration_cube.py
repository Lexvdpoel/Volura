"""Calibration cubes: the computational core of the classic XYZ cube you measure with calipers.

No Qt in this file except the letters, which come from `fonts` the way a cable winder's label
does. It gives the cube as one closed volume, so it can be tested on its own and reused in the
model and the export.

A **calibration cube** is the test print that says whether a printer keeps to size: a cube of
`size` millimetres, standing on the bed, that you measure with calipers after printing. Every axis
has its own way of going wrong - a bed that is not level, steps per millimetre that are off, an
extruder that pushes too little - and all three show up as a cube that is not `size` by `size` by
`size` millimetres.

With `letters` the three axes are named on the faces they run along, engraved `depth` millimetres
into the plastic: **Z** in the top face, **X** in the face that looks to the front and **Y** in
the face that looks to the right. The letters are half the size of the cube high and stand in the
middle of their face, so a caliper that lands on the letter still measures the whole face, and
each one is read from outside the cube, the way you read it while measuring. Without fonts the
cube is simply a cube.

The letters are cut in, never stood on, so the cube stays exactly `size` wide, deep and high and
nothing has to be subtracted from the measurement. The result is one closed volume on `z = 0`,
centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the smallest and the biggest cube worth measuring with calipers
SIZE_MIN, SIZE_MAX = 10.0, 50.0
#: mm the shallowest and the deepest the letters are engraved
DEPTH_MIN, DEPTH_MAX = 0.4, 1.2
#: the height of a letter as a part of the size of the cube
LETTER = 0.5
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5


def _number(value, lo, hi):
    """A number of the cube, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a calibration cube must be numbers"))
    return max(lo, min(hi, v))


def _sizes(size, letters, depth) -> dict:
    """The clamped numbers of the cube and the height its letters are drawn at.

    The cube stands on `z = 0` and is centred in x and y, so `half` is the wall every face lies
    behind; the letters are `LETTER` times the size high, which is how they stay in proportion
    with the cube.
    """
    L = _number(size, SIZE_MIN, SIZE_MAX)
    d = _number(depth, DEPTH_MIN, DEPTH_MAX)
    return {"size": L, "letters": bool(letters), "depth": d, "half": 0.5 * L,
            "letter": LETTER * L}


def sizes(size=20.0, letters=True, depth=0.6) -> dict:
    """The numbers a calibration cube is made of, so a dialog can show them.

    `size` is the edge of the cube in millimetres and also what a printed one has to measure
    between every pair of faces. `letters` says whether the axes are engraved in the faces and
    `depth` how deep they are cut. Every number is clamped to what prints and anything but a
    number is a ValueError.
    """
    g = _sizes(size, letters, depth)
    return {"size": g["size"], "letters": g["letters"], "depth": g["depth"]}


def _letters(g, text) -> list:
    """The outline of one letter, `g["letter"]` millimetres high and centred on the origin.

    Empty when this core runs without a Qt app or without fonts, or when the letter yields
    nothing: a cube without letters is not an error.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(text, height=g["letter"])
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    return [np.ascontiguousarray(np.asarray(L, dtype=float) - mid) for L in loops]


def _cutter(g, text, face) -> mf.Manifold:
    """The cut of one letter into a face: `face` is `top`, `front` or `right`.

    The outline is drawn flat and pulled `depth + MARGIN` millimetres up, so the cut reaches past
    the outside of the face and engraves `depth` millimetres into it. The letter of a side face
    stands upright there: the extrusion turns about x for the front face and about x and then z
    for the right one, which is also what keeps both of them readable from outside the cube.
    None when this system has no fonts, so the letter is simply not cut.
    """
    loops = _letters(g, text)
    if not loops:
        return None
    cut = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), g["depth"] + MARGIN)
    if face == "top":
        return cut.translate((0.0, 0.0, g["size"] - g["depth"]))
    if face == "front":
        return cut.rotate((90.0, 0.0, 0.0)) \
            .translate((0.0, g["depth"] - g["half"], g["half"]))
    return cut.rotate((90.0, 0.0, 0.0)).rotate((0.0, 0.0, 90.0)) \
        .translate((g["half"] - g["depth"], 0.0, g["half"]))


def calibration_cube(size=20.0, letters=True, depth=0.6) -> mf.Manifold:
    """One printable calibration cube: a cube of `size` with its axes engraved in the faces.

    The cube is `size` millimetres wide, deep and high and stands on `z = 0`. With `letters` the
    axis each direction runs along is engraved `depth` millimetres into the plastic: Z in the top
    face, X in the face that looks to the front and Y in the face that looks to the right, each
    half the size of the cube high and in the middle of its face. The letters are cut in and never
    stood on, so the cube stays `size` across in every direction.

    The result is one closed volume on `z = 0`, centred in x and y, that prints standing up with
    nothing to support. Every number is clamped to what prints and anything but a number is a
    ValueError; without fonts the cube is engraved with nothing.
    """
    g = _sizes(size, letters, depth)
    cube = mf.Manifold.cube([g["size"], g["size"], g["size"]]) \
        .translate((-g["half"], -g["half"], 0.0))
    if not g["letters"]:
        return cube
    cuts = [c for c in (_cutter(g, text, face) for text, face in
                        (("Z", "top"), ("X", "front"), ("Y", "right"))) if c is not None]
    if not cuts:
        return cube
    return cube - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
