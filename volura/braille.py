"""Braille labels: the computational core for a label that is read with the fingers.

No Qt in this file except the letters above the Braille, which come from `fonts` the way the
raised letters of a label plate do. It gives the label as one closed volume, so it can be tested
on its own and reused in the model and the export.

A **Braille label** is the sign that says what a drawer holds for someone who reads with a
fingertip: raised dots on a plate that stands on the bed, so it prints without support. The dots
are the Marburg Medium ones and the label may carry the same text in letters above them, so that
sighted and blind people read the same sign.

**Braille** here is uncontracted (grade 1), as used for labels: every word is spelled letter by
letter. A cell holds six dots, numbered 1-2-3 down the left column and 4-5-6 down the right one;
the dots of a character are `BRAILLE`, whose source is the BANA/ICEB "English Braille" alphabet
for uncontracted (grade 1) reading and writing. Six dots are not enough for twenty-six letters,
so the letters k-t are the letters a-j with dot 3 added, and the digits 1-9 and 0 are the letters
a-j too; for that a capital is announced by a cell of dot 6 before it, a run of digits by the
number sign of dots 3-4-5-6, and a letter a-j that stands right after digits by the letter sign
of dots 5-6, so that "1a" is not read as the digit "1" again. `\n` starts a new line.

The **sizes** are the Marburg Medium ones, which are also what signage recommends: a dot of
`DOT_D` millimetres across at its base and `DOT_H` high, `DOT_PITCH` millimetres from its
neighbours in the cell, cells `CELL_PITCH` apart from left dot column to left dot column and
lines `LINE_PITCH` apart. The dots of a cell are counted from dot 1, top left, and a line is
**left aligned**, not centred: a reader's hand starts at the left edge of every line.

A dot is a spherical cap: the sphere that goes through its base circle and its top, cut off at
the plate, so it is smooth under a fingertip and needs no support.

The **letters above** are the same text in raised letters of `LETTER_HEIGHT` millimetres cap
height, so that sighted people read the label too; the plate grows for them. The **hole** is the
way the label plate has one: a round hole through the plate `margin` left of the first cell,
which makes the plate that much longer for it.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

from .i18n import tr
from . import enclosure

#: the base diameter and the height of one raised dot: Marburg Medium
DOT_D, DOT_H = 1.5, 0.6
#: dot centre to dot centre inside a cell, both from row to row and from column to column
DOT_PITCH = 2.5
#: left dot column to left dot column, from one cell to the next on a line
CELL_PITCH = 6.0
#: dot row 1 to dot row 1, from one line to the next
LINE_PITCH = 10.0
#: the cap height and the relief of the letters that stand above the Braille
LETTER_HEIGHT, LETTER_RELIEF = 5.0, 0.6
#: the baseline distance of two lines of letters, as a multiple of their cap height
LETTER_LINE_SPACING = 1.4
#: the narrowest and the widest band of plate around the dots
MARGIN_MIN, MARGIN_MAX = 2.0, 15.0
#: the thinnest plate that survives being handled and the thickest that is still a plate
THICKNESS_MIN, THICKNESS_MAX = 1.0, 10.0
#: the largest hole through a Braille label: above that it is a bracket, not a label
HOLE_MAX = 15.0
#: the most lines and the most characters on one line: a label is read at a glance
LINES_MAX, CHARS_MAX = 4, 40
#: the straight pieces a hole is approximated with, and the ones the sphere of a dot is made of
SEGMENTS, SPHERE_SEGMENTS = 64, 48
#: mm past the top face, so the hole leaves no skin behind
CUT_MARGIN = 0.5
#: a cell without dots: a space, and the end of a line in the flat list of `cells`
EMPTY_CELL = ()

#: the cell that announces a capital, a run of digits and a letter that follows digits
CAPITAL_SIGN, NUMBER_SIGN, LETTER_SIGN = (6,), (3, 4, 5, 6), (5, 6)
#: the letters that are also digits, so one of these after digits needs the letter sign
DIGIT_LETTERS = "abcdefghij"

#: one cell per character: the dots of the letters a-z and of the punctuation a label needs,
#: uncontracted (grade 1). The letters k-t are a-j with dot 3 added; the digits are in `DIGITS`,
#: a capital, a run of digits and a letter after digits get a sign of their own, and the source
#: is the BANA/ICEB "English Braille" alphabet for uncontracted (grade 1) reading and writing.
BRAILLE = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5), "f": (1, 2, 4),
    "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5), "k": (1, 3), "l": (1, 2, 3),
    "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5), "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5),
    "r": (1, 2, 3, 5), "s": (2, 3, 4), "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6),
    "w": (2, 4, 5, 6), "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
    " ": EMPTY_CELL, ",": (2,), ";": (2, 3), ":": (2, 5), ".": (2, 5, 6), "!": (2, 3, 5),
    "?": (2, 3, 6), "'": (3,), "-": (3, 6),
}
#: the digits 1-9 and 0 are the letters a-j; the number sign in front of the run says so
DIGITS = {"1": (1,), "2": (1, 2), "3": (1, 4), "4": (1, 4, 5), "5": (1, 5), "6": (1, 2, 4),
          "7": (1, 2, 4, 5), "8": (1, 2, 5), "9": (2, 4), "0": (2, 4, 5)}


def dot_xy(dot) -> tuple[float, float]:
    """Where one dot sits in its cell, seen from the origin of that cell: dot 1 stands there.

    The dots run 1-2-3 down the left column and 4-5-6 down the right one, so every step along
    that order goes `DOT_PITCH` to the right or `DOT_PITCH` down.
    """
    n = int(dot)
    if n < 1 or n > 6:
        raise ValueError(tr("{dot} is not a dot of a Braille cell").format(dot=dot))
    column, row = (0, n - 1) if n <= 3 else (1, n - 4)
    return column * DOT_PITCH, -row * DOT_PITCH


def _line_cells(line) -> list[tuple[int, ...]]:
    """The cells of one line of text: the characters with the signs a reader needs added."""
    out = []
    digits = False            # the cell before this one was a digit, so the number goes on
    for ch in str(line):
        if ch in DIGITS:
            if not digits:
                out.append(NUMBER_SIGN)
            out.append(DIGITS[ch])
            digits = True
            continue
        if ch in BRAILLE:
            cell = BRAILLE[ch]
        elif ch.isupper() and ch.lower() in BRAILLE:
            out.append(CAPITAL_SIGN)
            cell = BRAILLE[ch.lower()]
        else:
            raise ValueError(tr("{char} cannot be written in Braille").format(char=ch))
        if digits and ch in DIGIT_LETTERS:
            out.append(LETTER_SIGN)      # otherwise "1a" is read as the digit "1" once more
        out.append(cell)
        digits = False
    return out


def line_cells(text) -> list[list[tuple[int, ...]]]:
    """The cells of `text`, one list per line: a `\n` in the text starts a new line.

    A cell is the tuple of the numbers of the dots that are raised in it, so an empty tuple is a
    space, and the capital, number and letter signs are cells of their own. `cells` gives the
    same cells as one flat list.
    """
    s = str(text or "")
    if not s.strip():
        raise ValueError(tr("a Braille label needs text"))
    lines = s.split("\n")
    if len(lines) > LINES_MAX or any(len(line) > CHARS_MAX for line in lines):
        raise ValueError(tr("a Braille label is at most {lines} lines of {chars} characters")
                         .format(lines=LINES_MAX, chars=CHARS_MAX))
    return [_line_cells(line) for line in lines]


def cells(text) -> list[tuple[int, ...]]:
    """The dots of every cell of `text`, along the lines from left to right and top down.

    The signs are there too: "Ab" gives the capital sign, then the cells of A and of b. A `\n` in
    the text starts a new line, which is one empty cell in this flat list, like a space;
    `line_cells` gives the same cells per line.
    """
    out = []
    for i, row in enumerate(line_cells(text)):
        if i:
            out.append(EMPTY_CELL)
        out.extend(row)
    return out


def _centres(rows) -> list[tuple[float, float]]:
    """The centres of the dots of `line_cells` rows, in the grid of the label."""
    pts = []
    for j, row in enumerate(rows):
        for k, cell in enumerate(row):
            for dot in cell:
                x, y = dot_xy(dot)
                pts.append((k * CELL_PITCH + x, -j * LINE_PITCH + y))
    return pts


def dot_centres(text) -> list[tuple[float, float]]:
    """The centres of the raised dots of `text`, in the grid of the label.

    Cell 0 of a line begins at the origin, the cell next to it is `CELL_PITCH` to the right and
    the line below is `LINE_PITCH` lower; nothing is centred, because a reader's hand starts at
    the left edge of a line. An empty cell takes its place in the row but holds no dot.
    """
    return _centres(line_cells(text))


def _letter_rows(text, family):
    """The lines of `text` in letters: the outlines of every line, the left edge of each line on
    x = 0, the baseline of the top line on y = 0 and the lines below it lower. A line without
    letters (an empty line, or a font that draws nothing) comes back empty.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    rows = []
    for line in str(text).split("\n"):
        try:
            loops = fonts.text_loops(line, family, LETTER_HEIGHT)
        except Exception:
            loops = []
        if not loops:
            rows.append([])
            continue
        loops = [np.asarray(L, dtype=float) for L in loops]
        left = min(float(L[:, 0].min()) for L in loops)
        rows.append([np.ascontiguousarray(L - np.array([left, 0.0])) for L in loops])
    return rows


def braille_label(text="Volura", margin=4.0, thickness=2.0, radius=2.0, hole=0.0,
                  letters=True, family=None) -> mf.Manifold:
    """A label of raised Braille dots that stands on the bed of the printer.

    The dots are the cells of `text`: uncontracted Braille with the capital, number and letter
    signs. They stand `DOT_H` millimetres above the plate, are `DOT_D` millimetres across at
    their base, `DOT_PITCH` apart inside a cell, `CELL_PITCH` apart from cell to cell and
    `LINE_PITCH` apart from line to line, and the lines are left aligned the way a reader
    expects. The plate is the dots `margin` millimetres wider all round, a rounded rectangle
    with corners of `radius`, `thickness` millimetres high, standing on `z = 0` and centred on
    the origin in x and y. A `\n` in the text makes the next line of dots below the first.

    `letters` stands the same text in raised letters of `LETTER_HEIGHT` millimetres cap height
    and `LETTER_RELIEF` millimetres relief above the Braille, left aligned with the dots and the
    lines of letters under each other, so that sighted people read the label too; the plate grows
    `LETTER_HEIGHT + margin` for them. `hole` drills a round hole of that diameter through the
    plate `margin` left of the first cell, in the middle of the plate depth, and makes the plate
    that much longer for it, so the label hangs on a hook or a screw.
    `family` is the font of those letters, by default the one the text tool prefers.
    """
    rows = line_cells(text)
    m = float(margin)
    if m < MARGIN_MIN or m > MARGIN_MAX:
        raise ValueError(tr("the margin of a Braille label must be {lo} to {hi} mm")
                         .format(lo=f"{MARGIN_MIN:g}", hi=f"{MARGIN_MAX:g}"))
    t = float(thickness)
    if t < THICKNESS_MIN or t > THICKNESS_MAX:
        raise ValueError(tr("the thickness of a Braille label must be {lo} to {hi} mm")
                         .format(lo=f"{THICKNESS_MIN:g}", hi=f"{THICKNESS_MAX:g}"))
    rad = float(radius)
    if rad < 0.0:
        raise ValueError(tr("the corner radius of a Braille label may not be negative"))
    d = float(hole)
    if d < 0.0 or d > HOLE_MAX:
        raise ValueError(tr("the hole of a Braille label must be 0 to {hi} mm")
                         .format(hi=f"{HOLE_MAX:g}"))

    pts = _centres(rows)
    # the plate around the dots: their ink reaches half a dot past the outer centres
    left = min(p[0] for p in pts) - 0.5 * DOT_D
    right = max(p[0] for p in pts) + 0.5 * DOT_D
    low = min(p[1] for p in pts) - 0.5 * DOT_D
    high = max(p[1] for p in pts) + 0.5 * DOT_D
    x_min, x_max, y_min, y_max = left - m, right + m, low - m, high + m

    ink = []
    if letters:
        from . import fonts           # Qt: only here, this core is Qt-free
        face = str(family or fonts.default_family())
        if not face:
            raise ValueError(tr("No font found for the text."))
        for i, row in enumerate(_letter_rows(text, face)):
            ink += [np.ascontiguousarray(L + np.array([0.0, -i * LETTER_LINE_SPACING
                                                       * LETTER_HEIGHT]), dtype=float)
                    for L in row]
        if not ink:
            raise ValueError(tr("No font found for the text."))
        # the letters stand on the plate above the Braille, `margin` clear of the dots and of the
        # top edge, and left aligned with the dots; the plate grows the band they need
        all_pts = np.concatenate(ink)
        shift = np.array([left - float(all_pts[:, 0].min()),
                          high + m - float(all_pts[:, 1].min())])
        ink = [np.ascontiguousarray(L + shift, dtype=float) for L in ink]
        x_max = max(x_max, float(all_pts[:, 0].max() + shift[0]) + m)
        y_max = max(y_max, float(all_pts[:, 1].max() + shift[1]) + m)

    # the hole is `margin` left of the first cell and the plate grows `margin` further for it;
    # the plate also grows in depth, so a big hole keeps `margin` of plate all round
    x_hole = left - m - 0.5 * d
    if d > 0.0:
        x_min = min(x_min, x_hole - 0.5 * d - m)
        need = d + 2.0 * m
        if y_max - y_min < need:
            extra = 0.5 * (need - (y_max - y_min))
            y_min, y_max = y_min - extra, y_max + extra
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

    # everything is built in the grid of the dots and moved to the middle of the plate for it,
    # so that the plate is centred on the origin and the dots reach its edges evenly
    part = mf.Manifold.extrude(enclosure.rounded_rect(x_max - x_min, y_max - y_min, rad), t)
    if d > 0.0:
        # the screw goes through the plate, so the hole is plain all the way through, in the
        # middle of the plate depth
        part = part - mf.Manifold.cylinder(t + 2.0 * CUT_MARGIN, 0.5 * d, 0.5 * d, SEGMENTS) \
            .translate((x_hole - cx, 0.0, -CUT_MARGIN))
    if ink:
        part = part + mf.Manifold.extrude(mf.CrossSection(ink, mf.FillRule.NonZero),
                                          LETTER_RELIEF).translate((-cx, -cy, t))

    # a dot is the cap of the sphere that goes through its base circle and its top, cut at the
    # plate: nothing overhangs, so the label prints without support
    r = (0.25 * DOT_D * DOT_D + DOT_H * DOT_H) / (2.0 * DOT_H)
    box = mf.Manifold.cube([2.0 * r, 2.0 * r, DOT_H + CUT_MARGIN])
    caps = [box.translate((x - cx - r, y - cy - r, t))
            ^ mf.Manifold.sphere(r, SPHERE_SEGMENTS)
            .translate((x - cx, y - cy, t + DOT_H - r))
            for x, y in pts]
    return part + mf.Manifold.batch_boolean(caps, mf.OpType.Add)
