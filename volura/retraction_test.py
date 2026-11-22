"""Retraction tests: the computational core of two thin towers whose bands each print with a
longer retraction.

No Qt in this file except the numbers, which come from `fonts` the way the raised temperatures of
a temperature tower do. It gives the test as one closed volume, so it can be tested on its own and
reused in the model and the export, and it gives the plan of retraction lengths the slice has to
write into its G-code, band by band.

A **retraction test** is a stringing test that measures: `2` towers of `DIAMETER` millimetres,
standing `DISTANCE` millimetres apart between their centres on a base of `BASE` millimetres, so on
every layer the nozzle has to travel through open air from one tower to the next. Nothing connects
them, and every travel is a chance to leave a hair of plastic behind - but here the height is cut
into `bands` bands of `BAND_H` millimetres, and every band travels with a longer retraction than
the one under it. The band where the hairs disappear is the retraction this printer wants; the
band where the nozzle starts grinding or the walls go thin is the one where it retracts too far.

The bands count themselves: a notch of `RING` millimetres deep is cut all round one tower at every
boundary between two bands, and the length of the lowest and the highest band stands `raised` on a
tab at the foot of the base in letters `CAP` millimetres high, so the two ends of the walk are read
without counting the notches.

Unlike the other tests that change something per height, this one cannot ask for its change with a
command: a retraction is no G-code line but a length CuraEngine hides in the extrusion moves it
writes around a travel. So the part keeps the plan of lengths - ``[z_from, z_to, length]``, the
lowest band first, in `g.meta["retraction_bands"]`` - and the slice rewrites the G-code after
CuraEngine has written it: in every layer of a band, every move that pulls filament back with
nothing moved in x or y gets the length of the band, and the move that pushes it back after it the
same length, so the filament stays where it belongs. That is `slicer.rewrite_retractions`, which
reads absolute as well as relative extrusion (M82 and M83) and keeps a `G92 E0` working. Because
the lengths are applied by height, a test that stands on the bed next to other parts changes the
retraction for the whole bed - which is what the dialog says - and a printer whose firmware
retracts itself (`G10` and `G11`) keeps its own retraction, because there is nothing in the
G-code to change there.

The result is one closed volume on `z = 0`, centred in x and y, that prints without support;
without fonts the tab carries no numbers.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import enclosure
from .i18n import tr, tr_noop

#: mm round a tower is: thin, so a hair of plastic on it is easy to see
DIAMETER = 4.0
#: mm between the centres of the two towers, which is how far the nozzle travels
DISTANCE = 50.0
#: mm thick the base under the towers is
BASE = 1.0
#: mm tall a band is: one retraction length per band of this height
BAND_H = 5.0
#: mm of base that lies beside the row, and the radius of its corners
EDGE = 2.5
#: mm of base that stays free around a number on the tab
EDGE_T = 2.0
#: mm deep the notch at a band boundary is, cut all round one tower
RING = 0.6
#: mm tall that notch is
RING_H = 0.9
#: mm long the tab at the foot is (along the row of towers)
TAB_W = 22.0
#: mm the tab stands out past the edge of the base
TAB_OUT = 7.0
#: mm the tab bites into the base, so it is one volume with it
TAB_IN = 0.6
#: mm thick the tab is: a layer or two of bottom, so it needs no top
TAB_T = 1.2
#: mm high the letters of a length are
CAP = 3.0
#: mm a number stands above the tab, and mm it bites into it
LABEL_PROTR, LABEL_EMBED = 0.6, 0.5
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5
#: how many faces a tower is drawn with
SEGMENTS = 64

#: mm long the base is: half a diameter of tower past the outside of both towers
LENGTH = DISTANCE + DIAMETER
#: mm deep the test is, the tab at the foot of the base not counted
DEPTH = DIAMETER + 2.0 * EDGE

#: the fewest and the most bands a test can have
BANDS_MIN, BANDS_MAX = 2, 15
#: mm the shortest and the longest a printer retracts
LENGTH_MIN, LENGTH_MAX = 0.1, 10.0
#: mm the smallest and the biggest step between two bands, which is what the dialog spins over
STEP_MIN, STEP_MAX = 0.05, 5.0
#: the default start, end and step of the dialog: six bands from 0.5 to 3 in steps of 0.5
LENGTH_START, LENGTH_END, LENGTH_STEP = 0.5, 3.0, 0.5

#: the presets of the dialog, in its order: name -> (start, end, step)
PRESETS = {"direct_drive": (0.5, 3.0, 0.5), "bowden": (2.0, 7.0, 1.0)}
#: what the dialog calls every preset: how far the extruder stands from the nozzle
PRESET_LABELS = {"direct_drive": tr_noop("Direct drive 0.5-3"),
                 "bowden": tr_noop("Bowden 2-7")}


def preset_label(name) -> str:
    """What the dialog calls a preset; one it does not know is its own name."""
    return tr(PRESET_LABELS[name]) if name in PRESET_LABELS else str(name)


def default_preset() -> str:
    """The preset to offer when nothing was chosen: the first one, the direct drive."""
    return next(iter(PRESETS))


def length_text(length) -> str:
    """A retraction length the way a number on the tab and a summary write it: 0.5, not
    0.5000000000000001.

    The lengths come out of a walk by `step`, so they carry the noise of that walk; `:g` gives the
    shortest form that means the same number. A length that is no number is written as it stands,
    and one so small that nothing moves is written as 0.
    """
    try:
        v = float(length)
    except (TypeError, ValueError):
        return str(length)
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
        raise ValueError(tr("the retraction lengths of a retraction test must be numbers"))
    return max(lo, min(hi, v))


def _lengths(start, end, step) -> list:
    """The retraction lengths of the bands, the shortest first, with the test's validation.

    `start` is the length of the lowest band and every next band stands `step` longer, walking from
    `start` towards `end` and stopping as soon as the next length would pass `end` - so the test has
    `(end - start) / step + 1` bands and both `start` and the last length at or under `end` are
    printed. A range that walks down works just the same, because a test can be read from the top
    too. A step of 0 (or less) would never reach the other length, so it is refused; so are a test
    of fewer than `BANDS_MIN` bands, which is nothing to compare, and one of more than `BANDS_MAX`,
    which does not print in reasonable time.
    """
    a = _number(start, LENGTH_MIN, LENGTH_MAX)
    b = _number(end, LENGTH_MIN, LENGTH_MAX)
    try:
        d = float(step)
    except (TypeError, ValueError):
        d = math.nan
    if not math.isfinite(d):
        raise ValueError(tr("the retraction lengths of a retraction test must be numbers"))
    if d <= 0.0:
        raise ValueError(tr("the step of a retraction test must be more than 0"))
    n = int(math.floor(abs(a - b) / d + 1e-9)) + 1
    if n < BANDS_MIN:
        raise ValueError(tr("a retraction test needs at least {n} bands").format(n=BANDS_MIN))
    if n > BANDS_MAX:
        raise ValueError(tr("a retraction test has at most {n} bands").format(n=BANDS_MAX))
    direction = 1.0 if b >= a else -1.0
    # the lengths are rounded to a thousandth, so a walk of 0.5 steps carries no noise and the last
    # one of a walk down is 0.0 instead of the -0.0 that floating point leaves behind
    return [round(a + k * d * direction, 3) + 0.0 for k in range(n)]


def lengths(start=LENGTH_START, end=LENGTH_END, step=LENGTH_STEP) -> list:
    """The retraction length of every band, the shortest first, validated as in `retraction_test`.

    These are the millimetres of filament every band pulls back; `bands` says from which height to
    which the printer has to hold them, and the slice writes them into its own G-code."""
    return _lengths(start, end, step)


def bands(start=LENGTH_START, end=LENGTH_END, step=LENGTH_STEP) -> list:
    """The plan of a retraction test, as ``[[z_from, z_to, length], ...]``, the lowest band first.

    Band `i` stands from `i * BAND_H` up to `(i + 1) * BAND_H` millimetres and travels with length
    `i`, so the plan says for every height of the test how far the filament has to come back. The
    lowest band starts on the bed, so its `z_from` is 0, and the highest one ends on the top of the
    test, which is where the rewriting stops: a part that stands taller than the test keeps the
    retraction of the profile. The numbers are validated as in `retraction_test`.
    """
    return [[round(i * BAND_H, 6), round((i + 1) * BAND_H, 6), v]
            for i, v in enumerate(_lengths(start, end, step))]


def sizes(start=LENGTH_START, end=LENGTH_END, step=LENGTH_STEP) -> dict:
    """The numbers a retraction test is made of, so a dialog can show them.

    `bands` is how many bands the test has and `lengths` the retraction length of every one of
    them, the shortest first; `size` is the outside (length, depth, height) in millimetres, which
    is the base the towers stand on with the tab at its foot sticking out on one side, and
    `bands * BAND_H` millimetres high. The numbers are validated as in `retraction_test` and
    anything but a number is a ValueError.
    """
    ls = _lengths(start, end, step)
    return {"bands": len(ls), "lengths": ls, "band_h": BAND_H,
            "size": (LENGTH, DEPTH + TAB_OUT, len(ls) * BAND_H)}


def _base() -> mf.Manifold:
    """The base the two towers stand on, `BASE` millimetres thick and rounded at the corners.

    It reaches half the diameter of a tower past the outside of both of them and `EDGE`
    millimetres beside the row, which together are its length and its depth; a tower starts on
    `z = 0` and not on top of it, so the two of them grow into one volume."""
    return mf.Manifold.extrude(enclosure.rounded_rect(LENGTH, DEPTH, EDGE), BASE)


def _towers(top) -> mf.Manifold:
    """The two towers the nozzle travels between, from the bed up to `top`.

    They are `DIAMETER` millimetres round and `DISTANCE` millimetres apart between their centres,
    and nothing connects them, so the nozzle crosses open air on every layer - which is what the
    test is about."""
    parts = [mf.Manifold.cylinder(top, 0.5 * DIAMETER, 0.5 * DIAMETER, SEGMENTS)
             .translate((x, 0.0, 0.0)) for x in (-0.5 * DISTANCE, 0.5 * DISTANCE)]
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _notch(z_from) -> mf.Manifold:
    """The cut that makes the notch at one band boundary at `z_from`.

    The cutter is a tube around the nearest tower that reaches from `RING` millimetres under its
    skin to `MARGIN` millimetres past it, `RING_H` millimetres tall and centred on the boundary, so
    what is left there is a groove the finger and the eye catch all round the tower. Only one of
    the two towers carries the notches, which is enough to count the bands by."""
    r = 0.5 * DIAMETER
    ring = mf.Manifold.cylinder(RING_H, r + MARGIN, r + MARGIN, SEGMENTS)
    hole = mf.Manifold.cylinder(RING_H + 2.0 * MARGIN, r - RING, r - RING, SEGMENTS) \
        .translate((0.0, 0.0, -MARGIN))
    return (ring - hole).translate((-0.5 * DISTANCE, 0.0, z_from - 0.5 * RING_H))


def _notches(n) -> list:
    """The notches of the boundaries between the `n` bands - the boundaries between two bands, not
    the base on the bed and not the top."""
    return [_notch(i * BAND_H) for i in range(1, n)]


def _tab() -> mf.Manifold:
    """The plate at the foot of the base that carries the two raised lengths.

    The tab stands out of the long side of the base at `z = 0` and bites `TAB_IN` millimetres into
    it, so it is one volume with the base, and it is `TAB_T` millimetres thick - layers that are all
    bottom to the slice, so it needs no top. It is where the lowest and the highest number of the
    test are read, so it stays out of the way of the towers and of the air between them."""
    return mf.Manifold.cube([TAB_W, TAB_IN + TAB_OUT, TAB_T]) \
        .translate((-0.5 * TAB_W, -0.5 * DEPTH - TAB_OUT, 0.0))


def _loops(text) -> list:
    """The outline of one length, `CAP` millimetres high and centred on the origin.

    A number that is too wide for its half of the tab is made smaller, which keeps the letters
    legible: `EDGE_T` millimetres of tab stay at its ends. Empty when this core runs without a Qt
    app or without fonts, or when the text yields nothing: a tab without numbers is not an error.
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
    room = 0.5 * TAB_W - 2.0 * EDGE_T
    if wide > room > 0.0:
        made = [one * (room / wide) for one in made]
    return [np.ascontiguousarray(one) for one in made]


def _raised(text, cx) -> mf.Manifold | None:
    """The cut that stands one length `LABEL_PROTR` millimetres above the tab at `x = cx`.

    The outline is drawn flat and pulled from `LABEL_EMBED` under the top of the tab to
    `LABEL_PROTR` above it, so the number bites into the tab instead of standing on it and is one
    volume with it. It lands in the middle of the part of the tab that lies outside the base, so the
    base neither hides it nor is hidden by it. None when this system has no fonts, so the tab
    simply carries no number there.
    """
    loops = _loops(text)
    if not loops:
        return None
    made = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero),
                               LABEL_EMBED + LABEL_PROTR)
    cy = -0.5 * DEPTH + 0.5 * (TAB_IN - TAB_OUT)
    return made.translate((cx, cy, TAB_T - LABEL_EMBED))


def _labelled(ls, tab) -> mf.Manifold:
    """The tab with the length of the lowest band stood at its near end and that of the highest
    band at its far end.

    Only the two end lengths are stood on the tab, because the notches give every band in between
    its number; the numbers are added to the tab instead of cut out of it, so they are still read
    when the tab is covered in hairs of plastic. The tab as it stands when this system has no
    fonts.
    """
    made = [c for c in (_raised(length_text(ls[0]), -0.25 * TAB_W),
                        _raised(length_text(ls[-1]), 0.25 * TAB_W)) if c is not None]
    if not made:
        return tab
    return mf.Manifold.batch_boolean([tab] + made, mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The test on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def retraction_test(start=LENGTH_START, end=LENGTH_END, step=LENGTH_STEP,
                    labels=True) -> mf.Manifold:
    """One printable retraction test: two thin towers, one band per retraction length.

    The towers are `DIAMETER` millimetres round and stand `DISTANCE` millimetres apart between their
    centres on a base of `BASE` millimetres, and the test has one band of `BAND_H` millimetres per
    length from `start` by `step` towards `end` millimetres, and so is `bands * BAND_H` millimetres
    tall in total. Nothing connects the towers, so the nozzle crosses open air on every layer and
    retracts every time it does - with the length of the band it is printing. A notch of `RING`
    millimetres deep is cut all round one tower at every boundary between two bands, so the bands
    count themselves on the print, and with `labels` the lengths of the lowest and the highest band
    stand `LABEL_PROTR` millimetres above the tab at the foot of the base, in letters `CAP`
    millimetres high.

    The result is one closed volume on `z = 0`, centred in x and y, that prints standing up with
    nothing to support. `start` and `end` are clamped to what a printer retracts, but a step that is
    no number, a step of 0 or less and a test with fewer than `BANDS_MIN` or more than `BANDS_MAX`
    bands are a ValueError; without fonts the tab carries no numbers. See `bands` for the plan of
    lengths the slice writes into its G-code, band by band."""
    ls = _lengths(start, end, step)
    tab = _labelled(ls, _tab()) if labels else _tab()
    body = mf.Manifold.batch_boolean([_base(), _towers(len(ls) * BAND_H), tab], mf.OpType.Add)
    return _centred(body - mf.Manifold.batch_boolean(_notches(len(ls)), mf.OpType.Add))
