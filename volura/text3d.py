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
"""
from __future__ import annotations

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


def text3d(text, family=None, height=10.0, depth=3.0, bold=False, align="left",
           line_spacing=LINE_SPACING) -> mf.Manifold:
    """The text as one closed volume: the letters, `depth` millimetres thick.

    The letters stand on `z = 0` and the block is centred on the origin in plan; `height` is the
    cap height of the letters. A `\\n` in the text makes the next line, `line_spacing` cap heights
    from baseline to baseline, and `align` (one of `ALIGNMENTS`) says how those lines stand under
    each other: at the same left edge, centred, or at the same right edge. `bold` asks the font
    for its bold face, which prints a letter that wears less quickly. `family` is the font, by
    default the one the text tool prefers.
    """
    s = str(text or "")
    if not s.strip():
        raise ValueError(tr("3D text needs text"))
    h = min(HEIGHT_MAX, max(HEIGHT_MIN, float(height)))
    d = min(DEPTH_MAX, max(DEPTH_MIN, float(depth)))
    if align not in ALIGNMENTS:
        raise ValueError(tr("unknown text alignment {align}").format(align=align))
    sp = min(SPACING_MAX, max(SPACING_MIN, float(line_spacing)))

    from . import fonts           # Qt: only here, this core is Qt-free
    face = str(family or fonts.default_family())
    if not face:
        raise ValueError(tr("No font found for the text."))
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
