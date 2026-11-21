"""Flow rate tests: the computational core of a row of pads, one per material flow.

No Qt in this file except the flows, which come from `fonts` the way the engraved values of a
pressure advance test do. It gives the test as a number of separate closed volumes, so it can be
tested on its own and reused in the model and the export, and it gives the flow of every pad, so the
model can hand each pad its own `material_flow` to the slicer.

A **flow rate test** is the test print that says how much plastic this printer has to push to lay a
smooth, closed top surface: `pads` square pads of `PAD` by `PAD` millimetres and `HEIGHT` millimetres
tall - enough layers that the top surface is all you have to look at - standing `GAP` millimetres
apart in a row, or in two rows when a single row would become too wide for the bed. Every pad prints
with its own flow, from `start` to `end` by `step` percent, so nine pads from 90 to 110 in steps of
2.5 walk from too little plastic to too much.

Too little flow leaves the top surface with gaps and low lines that never quite close; too much
leaves ridges and blobs where the lines pile up on each other. The pad whose top is smooth and closed
- no gaps, no ridges - is the flow that is right, and its number stands engraved `ENGRAVE`
millimetres deep in the middle of that top surface, in letters `CAP` millimetres high, so the pad you
like is also the number you read. Because the number is cut out of the top instead of stood on it,
the pad stays `PAD` millimetres across and its surface stays the thing you judge.

Unlike a temperature tower or a pressure advance test, a flow rate test does not change anything by
height: each pad prints with a fixed flow for its whole height, so it carries no commands. The flow
goes to the slicer as a setting of that one pad, kept in the part's `mesh_settings`; a flow test
that stands on the bed next to other parts changes the flow of nothing but its own pads.

The result is a list of closed volumes on `z = 0`, centred as a group around the origin and kept
apart on purpose: one piece per pad, so every one of them can carry its own flow; without fonts the
pads carry no numbers.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm a side of a pad is (a pad is a square)
PAD = 30.0
#: mm a pad is tall: enough layers that the top surface is all you have to look at
HEIGHT = 3.0
#: mm between two neighbouring pads
GAP = 3.0
#: mm the flow of a pad is engraved deep in its top surface
ENGRAVE = 0.4
#: mm high the letters of a flow are
CAP = 9.0
#: mm past the top that the cut reaches, so it does not stop in the surface it cuts
MARGIN = 0.5
#: mm of pad that has to stay at both ends of a number, so it does not run over an edge
EDGE = 3.0
#: mm the widest a single row of pads is allowed to become before the test is split into two rows
MAX_ROW = 210.0

#: % the coolest and the richest a flow prints at
FLOW_MIN, FLOW_MAX = 50.0, 150.0
#: the fewest and the most pads a test can have
PADS_MIN, PADS_MAX = 2, 15
#: the default start, end and step of the dialog: nine pads from 90 to 110 in steps of 2.5
FLOW_START, FLOW_END, FLOW_STEP = 90.0, 110.0, 2.5


def flow_text(flow) -> str:
    """A flow the way an engraving and a group name write it: 97.5, not 97.50000000000001.

    The flows come out of a walk by `step`, so they carry the noise of that walk; `:g` gives the
    shortest form that means the same number. A flow that is no number is written as it stands.
    """
    try:
        v = float(flow)
    except (TypeError, ValueError):
        return str(flow)
    if math.isfinite(v) and abs(v) < 1e-9:
        v = 0.0                             # -0.0 and -1e-17 are the same value here
    return f"{v:g}"


def _number(value, lo, hi):
    """A number of the test, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the flows of a flow rate test must be numbers"))
    return max(lo, min(hi, v))


def _flows(start, end, step) -> list:
    """The flows of the pads, lowest first, with the test's validation.

    `start` is the flow of the first pad and every next pad stands `step` higher, walking from
    `start` towards `end` and stopping as soon as the next flow would pass `end` - so the test has
    `(end - start) / step + 1` pads and both `start` and the last flow at or under `end` are printed.
    A range that walks down works just the same, because the pads can be read from the right too.
    A step of 0 (or less) would never reach the other flow, so it is refused; so are a test of fewer
    than `PADS_MIN` pads, which is nothing to compare, and one of more than `PADS_MAX` pads, which
    does not print in reasonable time and no longer fits the bed.
    """
    a = _number(start, FLOW_MIN, FLOW_MAX)
    b = _number(end, FLOW_MIN, FLOW_MAX)
    try:
        d = float(step)
    except (TypeError, ValueError):
        d = math.nan
    if not math.isfinite(d):
        raise ValueError(tr("the flows of a flow rate test must be numbers"))
    if d <= 0.0:
        raise ValueError(tr("the step of a flow rate test must be more than 0 %"))
    n = int(math.floor(abs(a - b) / d + 1e-9)) + 1
    if n < PADS_MIN:
        raise ValueError(tr("a flow rate test needs at least {n} pads").format(n=PADS_MIN))
    if n > PADS_MAX:
        raise ValueError(tr("a flow rate test has at most {n} pads").format(n=PADS_MAX))
    direction = 1.0 if b >= a else -1.0
    # the flows are rounded to a whole tenth of a percent, so a walk of 2.5 steps carries no noise
    return [round(a + k * d * direction, 3) + 0.0 for k in range(n)]


def _grid(n) -> tuple:
    """How the `n` pads lie: ``(rows, cols)`` of the grid they fill.

    A single row is kept as long as it stays under `MAX_ROW` millimetres wide, so the test fits the
    bed of most printers; a longer row is folded into two of as equal a length as the count allows,
    because a row wider than the bed prints on nothing."""
    one = n * PAD + (n - 1) * GAP
    rows = 1 if one <= MAX_ROW or n <= 1 else 2
    cols = -(-n // rows)                           # ceil(n / rows)
    return rows, cols


def _centres(n) -> list:
    """The centres of the `n` pads, centred as a group around the origin, in reading order.

    Pad `i` lands in row `i // cols` and column `i % cols`, so the pads walk left to right and top to
    bottom in the order of their flows; the whole group is moved so its middle is the origin."""
    rows, cols = _grid(n)
    total_w = cols * PAD + (cols - 1) * GAP
    total_d = rows * PAD + (rows - 1) * GAP
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append((c * (PAD + GAP) + 0.5 * PAD - 0.5 * total_w,
                    r * (PAD + GAP) + 0.5 * PAD - 0.5 * total_d))
    return out


def flows(start=FLOW_START, end=FLOW_END, step=FLOW_STEP) -> list:
    """The flow of every pad, lowest first, validated as in `flow_rate_test`.

    These are the `material_flow` values the pads print with; the model gives every pad its own and
    the slicer reads it back from the part. The numbers are validated exactly as the geometry is, so
    the same start, end and step that make the pads also give their flows."""
    return _flows(start, end, step)


def sizes(start=FLOW_START, end=FLOW_END, step=FLOW_STEP) -> dict:
    """The numbers a flow rate test is made of, so a dialog can show them.

    `pads` is how many pads there are and `flows` the flow of every one of them, lowest first; `rows`
    and `cols` say how the pads lie and `size` is the whole footprint (width, depth, height) in
    millimetres. The numbers are validated as in `flow_rate_test` and anything but a number is a
    ValueError.
    """
    fl = _flows(start, end, step)
    rows, cols = _grid(len(fl))
    total_w = cols * PAD + (cols - 1) * GAP
    total_d = rows * PAD + (rows - 1) * GAP
    return {"pads": len(fl), "flows": fl, "rows": rows, "cols": cols,
            "pad": PAD, "height": HEIGHT, "size": (total_w, total_d, HEIGHT)}


def _loops(text) -> list:
    """The outline of a flow, `CAP` millimetres high and centred on the origin.

    A number that is too wide for the pad is made smaller, which keeps the digits legible: `EDGE`
    millimetres of pad stay at both ends of it. Empty when this core runs without a Qt app or without
    fonts, or when the text yields nothing: a pad without a number is not an error.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # numbers need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(text, height=CAP)
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    made = [np.asarray(L, dtype=float) - mid for L in loops]
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    room = PAD - 2.0 * EDGE
    if wide > room > 0.0:
        made = [one * (room / wide) for one in made]
    return [np.ascontiguousarray(one) for one in made]


def _cut(text) -> mf.Manifold | None:
    """The cut that engraves one flow `ENGRAVE` millimetres deep in a pad centred on the origin.

    The outline is drawn flat and pulled `ENGRAVE + MARGIN` millimetres up from `HEIGHT - ENGRAVE`, so
    the cut starts `MARGIN` millimetres above the top surface and bites `ENGRAVE` millimetres into it;
    seen from above the number reads the right way up. None when this system has no fonts."""
    loops = _loops(text)
    if not loops:
        return None
    return mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero), ENGRAVE + MARGIN) \
        .translate((0.0, 0.0, HEIGHT - ENGRAVE))


def _pad(flow, cx, cy, labels) -> mf.Manifold:
    """One pad of `flow`: the square block with its flow engraved in the middle of its top."""
    block = mf.Manifold.cube([PAD, PAD, HEIGHT]).translate((cx - 0.5 * PAD, cy - 0.5 * PAD, 0.0))
    if not labels:
        return block
    cut = _cut(flow_text(flow))
    if cut is None:
        return block
    return block - cut.translate((cx, cy, 0.0))


def flow_rate_test(start=FLOW_START, end=FLOW_END, step=FLOW_STEP, labels=True) -> list:
    """The pads of a flow rate test: one closed volume per flow, `pads` of them.

    Every pad is a square of `PAD` by `PAD` millimetres and `HEIGHT` millimetres tall, `GAP`
    millimetres from its neighbours, printed with its own flow from `start` to `end` by `step`
    percent. The pads lie in a row, or in two rows when a single row would become wider than the bed,
    in the order of their flows, and the whole group is centred around the origin on `z = 0`. With
    `labels` the flow of a pad is engraved `ENGRAVE` millimetres deep in the middle of its top
    surface, in letters `CAP` millimetres high, so the pad with a smooth, closed top is also the
    number you read; nothing is stood on the pad, so it stays `PAD` millimetres across.

    The result is a list of closed volumes, one per pad, kept apart on purpose because each of them
    prints with a different flow. `start` and `end` are clamped to what a printer prints at, but a
    step that is no number, a step of 0 or less and a test with fewer than `PADS_MIN` or more than
    `PADS_MAX` pads are a ValueError; without fonts the pads carry no numbers. See `flows` for the
    flow every pad prints with.
    """
    fl = _flows(start, end, step)
    centres = _centres(len(fl))
    return [_pad(flow, cx, cy, labels) for (cx, cy), flow in zip(centres, fl)]
