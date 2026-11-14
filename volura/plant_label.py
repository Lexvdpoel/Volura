"""Plant labels: the computational core of a stake for a pot whose head carries a name.

No Qt in this file except the letters, which come from `fonts` the way the label of a cable winder
does. It gives the label as one closed volume, so it can be tested on its own and reused in the
model and the export.

A **plant label** is the stake you push into the soil of a pot, so you know again what it was you
sowed there: a flat head of `width` by `head` millimetres carries the name of the plant in letters
that stand `raised` millimetres out of the plastic, and below it an 8 millimetre wide **stake** of
`stake` millimetres goes down into the soil. The stake ends in a point, so it is easy to push in
and holds the head above the rim of the pot.

The head is a rectangle whose four corners round with `CORNER` millimetres, so nothing is sharp
enough to cut a hand, and the stake overlaps it by `OVERLAP` millimetres, so head and stake are
one piece and not a seam that snaps. The label is one plate of `thickness` millimetres with the
letters extruded straight up from the top face of the head, so every face is horizontal or
vertical: it prints **lying flat**, nothing overhangs and nothing needs support.

`text` is written in letters half as high as the head and made smaller when it would run past the
head less its margins, so the name stays centred on the head and inside it; empty text, or a
system without fonts, leaves a plain plate, which is a label you can still write on. Every number
is clamped to what prints and anything but a number is refused. The result is one closed volume on
`z = 0`, centred in x and y, `head + stake` millimetres long.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the narrowest and the widest head: narrow enough for a pot, wide enough to read
WIDTH_MIN, WIDTH_MAX = 30.0, 120.0
#: mm the shortest and the tallest head: a tall head is a sign, not a label
HEAD_MIN, HEAD_MAX = 12.0, 40.0
#: mm the shortest and the longest stake: a long one reaches the soil of a tall pot
STAKE_MIN, STAKE_MAX = 30.0, 200.0
#: the thinnest plate that survives the soil and the thickest that is still a plate
THICKNESS_MIN, THICKNESS_MAX = 1.6, 4.0
#: the least relief that reads as letters and the most that still reads as a label
RAISED_MIN, RAISED_MAX = 0.4, 2.0
#: mm the corners of the head round with, so nothing is sharp enough to cut a hand
CORNER = 3.0
#: mm wide the stake is and mm long the point its end tapers to
STAKE_WIDTH, POINT = 8.0, 8.0
#: mm the stake overlaps the head, so the two are one piece and not a seam that snaps
OVERLAP = 0.5
#: mm of head that has to stay around the letters, across and up
MARGIN_X, MARGIN_Y = 8.0, 6.0
#: the cap height of the letters, as a multiple of the height of the head
LETTER_RATIO = 0.5
#: the straight pieces the rounded corners are approximated with
SEGMENTS = 64


def _number(value, lo, hi):
    """A number of the label, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a plant label must be numbers"))
    return max(lo, min(hi, v))


def _sizes(text, width, head, stake, thickness, raised) -> dict:
    """The clamped numbers of the label and the length it lies on the bed with.

    The head is `width` by `head` millimetres and the stake `stake` millimetres long below it, so
    the label is `head + stake` millimetres long; the letters stand `raised` millimetres on top of
    the plate, which makes the label `thickness + raised` millimetres high.
    """
    w = _number(width, WIDTH_MIN, WIDTH_MAX)
    h = _number(head, HEAD_MIN, HEAD_MAX)
    k = _number(stake, STAKE_MIN, STAKE_MAX)
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    r = _number(raised, RAISED_MIN, RAISED_MAX)
    return {"text": str(text or ""), "width": w, "head": h, "stake": k, "thickness": t,
            "raised": r, "length": h + k}


def sizes(text="Basil", width=60.0, head=20.0, stake=80.0, thickness=2.4, raised=0.8) -> dict:
    """The numbers a plant label is made of, so a dialog can show them.

    `text` is the name on the head, `width` and `head` are the head in millimetres and `stake` how
    long the stake below it is; `thickness` is how thick the plate is and `raised` how high the
    letters stand out of it. `length` is the outside of the label, `head + stake` millimetres,
    which is how it lies on the bed. Every number is clamped to what prints and anything but a
    number is a ValueError.
    """
    s = _sizes(text, width, head, stake, thickness, raised)
    return {key: s[key] for key in ("text", "width", "head", "stake", "thickness", "raised",
                                    "length")}


def _head(s) -> mf.CrossSection:
    """The head: `width` by `head`, from `y = 0` up, with its four corners rounded.

    The corners are made by shrinking the rectangle and growing it back, so the sides stay where
    they were and only the corners become arcs of `CORNER` millimetres.
    """
    return (mf.CrossSection.square((s["width"], s["head"]), center=True)
            .translate([0.0, 0.5 * s["head"]])
            .offset(-CORNER, mf.JoinType.Miter)
            .offset(CORNER, mf.JoinType.Round, circular_segments=SEGMENTS))


def _stake(s) -> mf.CrossSection:
    """The stake: an 8 millimetre wide strip `stake` millimetres long that ends in a point.

    It starts `OVERLAP` millimetres above the bottom of the head, so it is welded into the head
    instead of only touching it, and the last `POINT` millimetres taper to the middle, so the
    stake is easy to push into the soil. The loop runs counter-clockwise - from the point up the
    right side and back along the top - so the union with the head keeps the strip instead of
    cutting it out.
    """
    half, tip = 0.5 * STAKE_WIDTH, s["stake"]
    loop = [(-half, -tip + POINT), (0.0, -tip), (half, -tip + POINT), (half, OVERLAP),
            (-half, OVERLAP)]
    return mf.CrossSection([np.ascontiguousarray(np.asarray(loop, dtype=float))])


def _letters(s) -> list:
    """The outlines of the name, centred on (`0`, `head / 2`) and shrunk to fit the head.

    Empty when there is no text, when this core runs without a Qt app or without fonts, or when
    the text yields nothing: a label without letters is a plain plate, not an error. The letters
    are half as high as the head and made smaller when they would run past the head less
    `MARGIN_X` across and `MARGIN_Y` up, so they always stay inside the head.
    """
    if not s["text"].strip():
        return []
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(s["text"], height=LETTER_RATIO * s["head"])
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    high = float(allp[:, 1].max() - allp[:, 1].min())
    room = (s["width"] - MARGIN_X, s["head"] - MARGIN_Y)
    scale = 1.0
    if wide > 1e-9 and room[0] > 0.0:
        scale = min(scale, room[0] / wide)
    if high > 1e-9 and room[1] > 0.0:
        scale = min(scale, room[1] / high)
    mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    return [np.ascontiguousarray((np.asarray(L, dtype=float) - mid) * scale
                                 + np.array([0.0, 0.5 * s["head"]])) for L in loops]


def _raised(s, plate) -> mf.Manifold:
    """The plate with the name standing `raised` millimetres on top of the head."""
    loops = _letters(s)
    if not loops:
        return plate
    letters = mf.CrossSection(loops, mf.FillRule.NonZero)
    return plate + mf.Manifold.extrude(letters, s["raised"]).translate(
        (0.0, 0.0, s["thickness"]))


def _centred(m) -> mf.Manifold:
    """The label on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def plant_label(text="Basil", width=60.0, head=20.0, stake=80.0, thickness=2.4,
                raised=0.8) -> mf.Manifold:
    """One printable plant label: a stake for a pot whose head carries the name of the plant.

    The head is `width` by `head` millimetres with its corners rounded `CORNER` millimetres and
    the stake below it is 8 millimetres wide, `stake` millimetres long and pointed at the end;
    together they are one plate of `thickness` millimetres. `text` stands on the head in letters
    half as high as the head and `raised` millimetres high, centred on it and shrunk to fit it, so
    the name is to be read and not only felt; empty text leaves a plain plate.

    The result is one closed volume on `z = 0`, centred in x and y, `head + stake` millimetres
    long and `thickness + raised` millimetres high, that prints lying flat: every face is
    horizontal or vertical, so nothing overhangs and nothing needs support. Every number is
    clamped to what prints and anything but a number is a ValueError.
    """
    s = _sizes(text, width, head, stake, thickness, raised)
    plate = mf.Manifold.extrude(_head(s) + _stake(s), s["thickness"])
    return _centred(_raised(s, plate))
