"""Label plates: the computational core for a printable name tag or drawer label.

No Qt in this file except the letters, which come from `fonts` the way the raised labels of a
clearance test do. It gives the plate as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **label plate** is the tag that says what a drawer holds, the name on a tool holder or the
sign above a switch: letters on a plate that stands on the bed, so it prints without support
and lands where the other printed parts land. `height` is the cap height of the letters,
because that is what a label is measured by; the plate is the text `margin` millimetres wider
all round, a rounded rectangle with corners of `radius`, so it is pleasant to hold.

The **style** is how the letters are made. `raised` stands them `relief` millimetres on top of
the plate, the way a clearance test writes its clearances, and is what a printer does best;
`engraved` cuts them `relief` millimetres deep in the top face, which still reads when the
plate is rubbed and when it is painted, and which leaves `ENGRAVE_ROOM` millimetres of plate
under the cut, so the plate does not become a stencil.

The **hole** is how the tag is hung: a round hole of `hole` millimetres through the plate,
`margin` left of the letters, which makes the plate that much longer for it, so the label hangs
on a hook, a nail or a screw.

Several **lines** go on one plate: `\n` in the text starts the next line, every line is centred
and the lines stand `1.4 × height` apart, the way a paragraph is set.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

from .i18n import tr
from . import enclosure

#: the smallest cap height that is still readable and the largest that is a label, not a sign
HEIGHT_MIN, HEIGHT_MAX = 3.0, 50.0
#: the thinnest plate that survives being handled and the thickest that is still a plate
THICKNESS_MIN, THICKNESS_MAX = 1.0, 10.0
#: the least relief that prints as letters and the most that still reads as a label
RELIEF_MIN, RELIEF_MAX = 0.2, 5.0
#: the narrowest and the widest band of plate around the letters
MARGIN_MIN, MARGIN_MAX = 0.0, 20.0
#: the largest hole through a label plate: above that it is a bracket, not a tag
HOLE_MAX = 15.0
#: the most lines and the most characters on one line: a label is read at a glance
LINES_MAX, CHARS_MAX = 4, 40
#: the baseline distance of two lines, as a multiple of the cap height
LINE_SPACING = 1.4
#: mm of plate that has to stay under an engraving, so the plate stays a plate
ENGRAVE_ROOM = 0.4
#: the straight pieces a hole is approximated with
SEGMENTS = 64
#: mm past the top face, so an engraving leaves no skin behind
CUT_MARGIN = 0.5

LABEL_MODES = ("raised", "engraved")


def _rows(text, family, height):
    """The lines of `text`: the outlines of every line, centred on x = 0, baseline on y = 0.

    A line that yields no letters (an empty line, or a font that draws nothing) comes back
    empty; the outlines are the ones the raised labels of a clearance test are made of.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    rows = []
    for line in str(text).split("\n"):
        try:
            loops = fonts.text_loops(line, family, height)
        except Exception:
            loops = []
        if not loops:
            rows.append([])
            continue
        pts = np.concatenate([np.asarray(L, dtype=float) for L in loops])
        mid = 0.5 * float(pts[:, 0].max() + pts[:, 0].min())
        rows.append([np.ascontiguousarray(np.asarray(L, dtype=float) - np.array([mid, 0.0]))
                     for L in loops])
    return rows


def label_plate(text="Volura", height=8.0, thickness=2.0, relief=0.8, mode="raised",
                margin=3.0, radius=2.0, hole=0.0, family=None) -> mf.Manifold:
    """A printable name tag or drawer label: the text on a plate.

    The plate stands on `z = 0` and is centred on the origin, `thickness` millimetres high.
    `text` is written in letters whose cap height is `height` millimetres, and the plate is the
    text `margin` millimetres wider all round, a rounded rectangle with corners of `radius`.
    A `\n` in the text makes the next line; the lines are stacked `1.4 × height` apart and each
    is centred.

    `mode` is one of `LABEL_MODES`: "raised" stands the letters `relief` millimetres on top of
    the plate, "engraved" cuts them `relief` millimetres deep in the top face. `hole` drills a
    round hole of that diameter through the plate `margin` left of the letters and makes the
    plate that much longer for it, so the tag hangs on a hook or a screw; the letters stay
    centred on the origin. `family` is the font, by default the one the text tool prefers.
    """
    s = str(text or "")
    if not s.strip():
        raise ValueError(tr("a label plate needs text"))
    lines = s.split("\n")
    if len(lines) > LINES_MAX or any(len(line) > CHARS_MAX for line in lines):
        raise ValueError(tr("a label plate is at most {lines} lines of {chars} characters")
                         .format(lines=LINES_MAX, chars=CHARS_MAX))
    h = float(height)
    if h < HEIGHT_MIN or h > HEIGHT_MAX:
        raise ValueError(tr("the text height of a label plate must be {lo} to {hi} mm")
                         .format(lo=f"{HEIGHT_MIN:g}", hi=f"{HEIGHT_MAX:g}"))
    t = float(thickness)
    if t < THICKNESS_MIN or t > THICKNESS_MAX:
        raise ValueError(tr("the thickness of a label plate must be {lo} to {hi} mm")
                         .format(lo=f"{THICKNESS_MIN:g}", hi=f"{THICKNESS_MAX:g}"))
    r = float(relief)
    if r < RELIEF_MIN or r > RELIEF_MAX:
        raise ValueError(tr("the relief of a label plate must be {lo} to {hi} mm")
                         .format(lo=f"{RELIEF_MIN:g}", hi=f"{RELIEF_MAX:g}"))
    if mode not in LABEL_MODES:
        raise ValueError(tr("unknown label plate style {mode}").format(mode=mode))
    m = float(margin)
    if m < MARGIN_MIN or m > MARGIN_MAX:
        raise ValueError(tr("the margin of a label plate must be {lo} to {hi} mm")
                         .format(lo=f"{MARGIN_MIN:g}", hi=f"{MARGIN_MAX:g}"))
    rad = float(radius)
    if rad < 0.0:
        raise ValueError(tr("the corner radius of a label plate may not be negative"))
    d = float(hole)
    if d < 0.0 or d > HOLE_MAX:
        raise ValueError(tr("the hole of a label plate must be 0 to {hi} mm")
                         .format(hi=f"{HOLE_MAX:g}"))
    if mode == "engraved" and r >= t - ENGRAVE_ROOM:
        raise ValueError(tr("an engraving must leave {room} mm of plate under it")
                         .format(room=f"{ENGRAVE_ROOM:g}"))

    from . import fonts           # Qt: only here, this core is Qt-free
    face = str(family or fonts.default_family())
    if not face:
        raise ValueError(tr("No font found for the text."))
    rows = _rows(s, face, h)
    # the lines stacked under each other, 1.4 cap heights from baseline to baseline
    loops = [np.ascontiguousarray(L + np.array([0.0, -i * LINE_SPACING * h]), dtype=float)
             for i, row in enumerate(rows) for L in row]
    if not loops:
        raise ValueError(tr("No font found for the text."))
    ink = np.concatenate(loops)
    low, high = ink.min(axis=0), ink.max(axis=0)
    mid = 0.5 * (low + high)
    loops = [np.ascontiguousarray(L - mid, dtype=float) for L in loops]
    wide, deep = float(high[0] - low[0]), float(high[1] - low[1])

    # the plate: the letters with margin all round, and as much room on the left as the hole
    # needs; the plate is shifted away from the letters for it, so the text stays centred
    length, depth = wide + 2.0 * m, max(deep, d) + 2.0 * m
    x_hole = -0.5 * wide - m - 0.5 * d
    if d > 0.0:
        length += d + m
    plate = mf.Manifold.extrude(enclosure.rounded_rect(length, depth, rad), t)
    if d > 0.0:
        plate = plate.translate((-0.5 * (d + m), 0.0, 0.0))
    letters = mf.CrossSection(loops, mf.FillRule.NonZero)
    if mode == "raised":
        plate = plate + mf.Manifold.extrude(letters, r).translate((0.0, 0.0, t))
    else:
        # the engraving is cut from the top face down, and past the top, so nothing is left
        plate = plate - mf.Manifold.extrude(letters, r + CUT_MARGIN) \
            .translate((0.0, 0.0, t - r))
    if d > 0.0:
        # the screw goes through the plate, so the hole is plain all the way through
        plate = plate - mf.Manifold.cylinder(t + 2.0 * CUT_MARGIN, 0.5 * d, 0.5 * d, SEGMENTS) \
            .translate((x_hole, 0.0, -CUT_MARGIN))
    return plate
