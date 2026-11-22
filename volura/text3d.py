"""3D text: the computational core for a part that is made of letters.

No Qt in this file except the letters, which come from `fonts` the way the raised labels of a
label plate do. It gives the text as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **3D text** is the word that is the part itself: a name on a box lid, a letter on a trophy, the
sign that is printed as one piece. `height` is the cap height of the letters, because that is
what text is measured by, and `depth` is how thick the letters are, standing on `z = 0` so the
text prints without support. The holes of o, a and 8 stay holes, so an "o" comes off the printer
as a ring and not as a disc.

Several **lines** go in one part: `\n` in the text starts the next line and the lines stand
`line_spacing × height` apart, the way a paragraph is set. `align` says how the lines stand under
each other: at the same left edge, centred, or at the same right edge; the whole block is centred
on the origin, because that is where a part is placed.

On an **arc** the text follows a circle instead of a straight line. `radius` is the circle the
letters stand on, their baseline; every letter is placed on its own and turned so that its baseline
touches that circle in the middle of the room it takes, and the room of all letters together is
walked along the circle, so the letters stand as far apart on the arc as they do on a straight line.
`inside` says which side of the circle the letters wear their feet to: on the outside they stand on
the circle and read clockwise over the top, the way a name stands on the lid of a round box, and on
the inside they turn round, feet outward, and read counter clockwise under the bottom, the way text
stands in a ring. The text is centred on the top (on the inside: on the bottom) of the circle and
the circle keeps its centre in the origin, because that is the place a round part is built from. An
arc takes one line of text, and text that would walk further than once round the circle is refused
with the radius it would need.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: the smallest cap height that is still readable and the largest that is a part, not a sign
HEIGHT_MIN, HEIGHT_MAX = 2.0, 200.0
#: the thinnest letter that survives being handled and the thickest that still reads as text
DEPTH_MIN, DEPTH_MAX = 0.4, 50.0
#: the baseline distance of two lines, as a multiple of the cap height
LINE_SPACING = 1.3
#: the narrowest and the widest band between two lines, as a multiple of the cap height
SPACING_MIN, SPACING_MAX = 0.5, 4.0
#: how the lines of several lines of text stand under each other
ALIGNMENTS = ("left", "centre", "right")
#: what the text follows: a straight line, or a circle the letters stand on
PATHS = ("straight", "arc")
#: the smallest circle that is still a part you can read and the largest that is not a wall
RADIUS_MIN, RADIUS_MAX = 5.0, 500.0
#: the radius the dialog starts with: a lid, not a ring
RADIUS_START = 60.0


def _face(family) -> str:
    """The font the letters are drawn with; empty when the system has no fonts at all."""
    from . import fonts           # Qt: only here, this core is Qt-free
    return str(family or fonts.default_family())


def _asked(text, family, height):
    """What `text_width` and its friends are asked for: the text, the font and a cap height in
    range, said plainly when one of them is not there, the way `text3d` says it."""
    s = str(text or "")
    if not s.strip():
        raise ValueError(tr("3D text needs text"))
    face = _face(family)
    if not face:
        raise ValueError(tr("No font found for the text."))
    return s, face, min(HEIGHT_MAX, max(HEIGHT_MIN, float(height)))


def _rows(text, family, height, bold):
    """The lines of `text`: the outlines of every line, first letter on x = 0, baseline on y = 0.

    A line that yields no letters (an empty line, or a font that draws nothing) comes back
    empty; the outlines are the ones the raised labels of a label plate are made of, holes
    included.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    rows = []
    for line in str(text).split("\n"):
        try:
            loops = fonts.text_loops(line, family, height, bold)
        except Exception:
            loops = []
        if not loops:
            rows.append([])
            continue
        rows.append([np.asarray(L, dtype=float) for L in loops])
    return rows


def arc_lines(text) -> int:
    """How many lines `text` has that draw something: an arc takes exactly one of them.

    The empty lines around a text are not counted, because they draw nothing and an arc misses
    nothing without them.
    """
    return sum(1 for line in str(text or "").split("\n") if line.strip())


def text_width(text, family=None, height=10.0, bold=False) -> float:
    """How wide `text` is in millimetres, taken the way a pen takes it: the room of all letters.

    The drawn letters are a little narrower, because a letter does not fill the room the font
    gives it; the room is what decides how far the next letter goes, so it is what an arc counts
    and what says how much circle a text needs.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    s, face, h = _asked(text, family, height)
    return sum(a for _loops, a in fonts.letter_loops(s, face, h, bool(bold)))


def min_radius(text, family=None, height=10.0, bold=False) -> float:
    """The smallest radius on which `text` still fits on an arc, in millimetres.

    Once the letters have walked the whole circle they run into each other, so that is where the
    radius has to start; `RADIUS_MIN` is the smallest the dialog can go in any case. The answer is
    rounded up to a tenth, so a text at this radius really does fit.
    """
    wide = text_width(text, family, height, bold)
    return max(RADIUS_MIN, math.ceil(wide / (2.0 * math.pi) * 10.0) / 10.0)


def _arc_letters(s, face, h, bold, radius, inside):
    """The letters of an arc text: the outlines of every letter, the room it takes and the angle
    it stands at, in the order they are read, together with the radius of the circle.

    The room of the letters is walked along the circle of `radius` at the baseline, so they stand
    as far apart on the arc as they do on a straight line, and the text is centred on the top of
    the circle, or on the bottom of it when `inside` turns the letters round with their feet
    outward. Outside the letters run clockwise, inside counter clockwise. Text of more than one
    line, text that draws nothing and text that would walk further than once round the circle are
    refused; the last one says which radius it needs.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    if arc_lines(s) > 1:
        raise ValueError(tr("An arc takes one line of text"))
    letters = []
    for line in s.split("\n"):    # an empty line around the one line draws nothing
        letters += [(loops, max(0.0, float(a)))
                    for loops, a in fonts.letter_loops(line, face, h, bool(bold))]
    if not any(loops for loops, _a in letters):
        raise ValueError(tr("No font found for the text."))
    r = min(RADIUS_MAX, max(RADIUS_MIN, float(radius)))
    wide = sum(a for _loops, a in letters)
    if wide > 2.0 * math.pi * r:
        raise ValueError(tr("this text needs a radius of at least {mm} mm on an arc")
                         .format(mm=f"{min_radius(s, face, h, bold):.1f}"))
    sign = 1.0 if inside else -1.0            # inside, the letters run counter clockwise
    middle = -0.5 * math.pi if inside else 0.5 * math.pi
    run, out = 0.0, []
    for loops, a in letters:
        out.append((loops, a, middle + sign * (run + 0.5 * a - 0.5 * wide) / r))
        run += a
    return out, r


def _arc_frame(ang, r, inside):
    """The frame of a letter on an arc: the point where its baseline touches the circle of radius
    `r` at the angle `ang`, and the directions of its own x (along the reading) and y (up the
    letter). On the inside the letters wear their feet outward, so up points to the centre.
    """
    away = np.array([math.cos(ang), math.sin(ang)])
    up = -away if inside else away
    return r * away, np.array([up[1], -up[0]]), up


def _arc_point(x, y, advance, frame):
    """The point of a letter on the arc, or all the points of one of its outlines: `x` and `y`
    take a point over the room of the letter and up from its baseline, `advance` is the room of the
    letter and `frame` puts that letter on the arc. Points that come in as arrays go out as (n, 2).
    """
    base, right, up = frame
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return base + (x - 0.5 * advance)[:, None] * right + y[:, None] * up


def arc_places(text, family=None, height=10.0, bold=False, radius=RADIUS_START,
               inside=False) -> list[tuple[float, float, float]]:
    """Where the letters of an arc text stand: one place for every character of `text`.

    A place is `(x, y, angle)`: the middle of the room the letter takes, half a cap height above
    its baseline, in millimetres with the centre of the arc in the origin, and the angle in radians
    where the letter touches the circle. The angles run clockwise for letters on the outside of the
    arc and counter clockwise for letters on the inside, and they keep running past -π and π, so
    the first letter and the last one compare without the circle getting them mixed up. A character
    that draws nothing (a space) has a place too, because it takes room like the letters do; the
    refusals of the arc are the refusals of `text3d`.
    """
    s, face, h = _asked(text, family, height)
    letters, r = _arc_letters(s, face, h, bool(bold), radius, bool(inside))
    out = []
    for _loops, _a, ang in letters:
        point = _arc_point(0.0, 0.5 * h, 0.0, _arc_frame(ang, r, bool(inside)))[0]
        out.append((float(point[0]), float(point[1]), float(ang)))
    return out


def _arc_text(s, face, h, d, bold, radius, inside):
    """The letters of an arc text as one closed volume, with the centre of the circle in the
    origin and the letters standing on `z = 0`.

    Every letter keeps the shape the font gives it and is turned to the tangent of the circle in
    the middle of its room; on a tight arc the corners of two letters can meet, so the letters are
    made first and joined afterwards. That also keeps the hole of an o a hole, whichever letter
    stands next to it.
    """
    letters, r = _arc_letters(s, face, h, bool(bold), radius, bool(inside))
    parts = []
    for loops, advance, ang in letters:
        if not loops:
            continue                # a space takes room on the arc but draws nothing
        frame = _arc_frame(ang, r, bool(inside))
        polys = [np.ascontiguousarray(_arc_point(L[:, 0], L[:, 1], advance, frame), dtype=float)
                 for L in loops]
        parts.append(mf.Manifold.extrude(mf.CrossSection(polys, mf.FillRule.NonZero), d))
    if not parts:
        raise ValueError(tr("No font found for the text."))
    if len(parts) == 1:
        return parts[0]
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def text3d(text, family=None, height=10.0, depth=3.0, bold=False, align="left",
           line_spacing=LINE_SPACING, path="straight", radius=RADIUS_START,
           inside=False) -> mf.Manifold:
    """The text as one closed volume: the letters, `depth` millimetres thick.

    The letters stand on `z = 0` and the block is centred on the origin in plan; `height` is the
    cap height of the letters. A `\\n` in the text makes the next line, `line_spacing` cap heights
    from baseline to baseline, and `align` (one of `ALIGNMENTS`) says how those lines stand under
    each other: at the same left edge, centred, or at the same right edge. `bold` asks the font
    for its bold face, which prints a letter that wears less quickly. `family` is the font, by
    default the one the text tool prefers.

    `path` is "straight" or "arc" (one of `PATHS`). On an arc the letters follow the circle of
    `radius` millimetres, one line of text and no more, each letter turned to the tangent of that
    circle in the middle of its room; the centre of the circle is the origin, and `inside` turns
    the letters round, feet outward, so they read counter clockwise under the bottom instead of
    clockwise over the top.
    """
    s = str(text or "")
    if not s.strip():
        raise ValueError(tr("3D text needs text"))
    h = min(HEIGHT_MAX, max(HEIGHT_MIN, float(height)))
    d = min(DEPTH_MAX, max(DEPTH_MIN, float(depth)))
    if align not in ALIGNMENTS:
        raise ValueError(tr("unknown text alignment {align}").format(align=align))
    if path not in PATHS:
        raise ValueError(tr("unknown text path {path}").format(path=path))
    sp = min(SPACING_MAX, max(SPACING_MIN, float(line_spacing)))

    face = _face(family)
    if not face:
        raise ValueError(tr("No font found for the text."))
    if path == "arc":
        return _arc_text(s, face, h, d, bool(bold), radius, bool(inside))
    rows = _rows(s, face, h, bool(bold))
    if not any(rows):
        raise ValueError(tr("No font found for the text."))

    # the lines as wide as the widest one, so align has something to line the lines up against
    wide = [float(np.concatenate(row)[:, 0].max()) if row else 0.0 for row in rows]
    block = max(wide) if wide else 0.0
    edge = {"left": lambda w: 0.0, "centre": lambda w: 0.5 * (block - w),
            "right": lambda w: block - w}[align]
    loops = [np.ascontiguousarray(L + np.array([edge(w), -i * sp * h]), dtype=float)
             for i, (row, w) in enumerate(zip(rows, wide)) for L in row]
    ink = np.concatenate(loops)
    mid = 0.5 * (ink.min(axis=0) + ink.max(axis=0))
    loops = [np.ascontiguousarray(L - mid, dtype=float) for L in loops]
    return mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), d)
