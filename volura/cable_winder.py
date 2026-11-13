"""Cable winders: the computational core of an H-shaped plate you wind a cable round.

No Qt in this file except the letters, which come from `fonts` the way a label plate does. It
gives the winder as one closed volume, so it can be tested on its own and reused in the model and
the export.

A **cable winder** is the flat piece you wrap a loose cable round and hang up: `length`
millimetres long, `width` wide and `thickness` millimetres thick. From each short end a **slot**
is cut in the middle, `slot` millimetres wide and `depth` millimetres deep, so the plate looks
like an H and the cable winds round the middle part that is left standing between the two slots.
The two ends are the flanges that keep the wound cable from sliding off; the middle part is the
barrel it winds on, `length - 2 * depth` millimetres long.

The plate prints **lying flat**, so every wall is pulled straight up and nothing overhangs: the
outside corners are rounded with a radius of `CORNER` millimetres and the closed end of every slot
with half the slot width, a round end, so nothing snags the cable. An optional **cable clamp** is
a keyhole: a notch of `KEYHOLE_W` millimetres wide and `KEYHOLE_LEN` millimetres long cut from the
bottom of one slot further into the plate, with a round end, so the stripped end of the cable is
pushed through the slot and pinched in the narrow part.

An optional **label** is engraved `ENGRAVE` millimetres deep in the top face, centred on the middle
part in letters of `LETTER_HEIGHT` millimetres, made smaller when a long text would run past the
middle part; empty text engraves nothing.

Every number is clamped to what prints: the slot is never wider than the plate less two flanges of
`FLANGE` millimetres and the slot is never deeper than a third of the length, so the middle part
always stands. The result is one closed volume on `z = 0`, centred in x and y.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import enclosure
from .i18n import tr

#: mm the shortest and the longest winder that is worth winding a cable round
LENGTH_MIN, LENGTH_MAX = 40.0, 200.0
#: mm the narrowest and the widest winder: wide ends keep the wound cable in
WIDTH_MIN, WIDTH_MAX = 20.0, 100.0
#: the thinnest plate that survives being handled and the thickest that is still a plate
THICKNESS_MIN, THICKNESS_MAX = 2.0, 6.0
#: mm the narrowest and the widest slot the cable winds between
SLOT_MIN, SLOT_MAX = 4.0, 30.0
#: mm the shallowest and the deepest slot cut from an end
DEPTH_MIN, DEPTH_MAX = 5.0, 60.0
#: mm the outside corners round with, so nothing is sharp enough to cut a cable
CORNER = 3.0
#: mm of plate that has to stay beside a slot, so the ends are still flanges worth the name
FLANGE = 6.0
#: mm wide and long the keyhole is that pinches the end of the cable
KEYHOLE_W, KEYHOLE_LEN = 3.0, 8.0
#: mm deep the label is engraved in the top face
ENGRAVE = 0.6
#: mm high the letters of the label are, before they are shrunk to fit the middle part
LETTER_HEIGHT = 6.0
#: mm of middle part that stays around the engraved letters
LABEL_MARGIN = 2.0
#: the straight pieces a round end is approximated with
SEGMENTS = 64
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5


def _number(value, lo, hi):
    """A number of the winder, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a cable winder must be numbers"))
    return max(lo, min(hi, v))


def _sizes(length, width, thickness, slot, depth, keyhole, label) -> dict:
    """The clamped numbers of the winder and the length of the middle part it winds on.

    The plate is `length` by `width` by `thickness`. The slot is never wider than the plate less
    two flanges of `FLANGE` millimetres, so a strip of plastic stays beside it on every end, and
    the slot is never deeper than a third of the length, so the middle part between the two slots
    always stands. `middle` is the length of that middle part.
    """
    L = _number(length, LENGTH_MIN, LENGTH_MAX)
    W = _number(width, WIDTH_MIN, WIDTH_MAX)
    T = _number(thickness, THICKNESS_MIN, THICKNESS_MAX)
    s = _number(slot, SLOT_MIN, min(SLOT_MAX, W - 2.0 * FLANGE))
    d = _number(depth, DEPTH_MIN, min(DEPTH_MAX, L / 3.0))
    return {"length": L, "width": W, "thickness": T, "slot": s, "depth": d,
            "keyhole": bool(keyhole), "label": str(label or ""), "middle": L - 2.0 * d}


def sizes(length=80.0, width=40.0, thickness=3.0, slot=10.0, depth=15.0, keyhole=True,
          label="") -> dict:
    """The numbers a cable winder is made of, so a dialog can show them.

    `length` is how long the plate is, `width` how wide and `thickness` how thick; `slot` is how
    wide a slot is cut from an end and `depth` how deep. `keyhole` says whether the cable clamp is
    cut and `label` is the text engraved on the middle part. `middle` is the length of the middle
    part the cable winds round, which is the length less two slot depths. Every number is clamped
    to what prints; anything but a number is a ValueError.
    """
    g = _sizes(length, width, thickness, slot, depth, keyhole, label)
    return {"length": g["length"], "width": g["width"], "thickness": g["thickness"],
            "slot": g["slot"], "depth": g["depth"], "keyhole": g["keyhole"], "label": g["label"],
            "middle": g["middle"]}


def _plate(g) -> mf.Manifold:
    """The whole plate: `length` by `width`, `thickness` high, its outside corners rounded."""
    return mf.Manifold.extrude(enclosure.rounded_rect(g["length"], g["width"], CORNER),
                               g["thickness"])


def _slot(g, sign) -> list:
    """The cut that opens a slot in the middle of one short end, `sign` the end it is on.

    The slot is `slot` millimetres wide and `depth` millimetres deep, cut from the end inwards, and
    its closed end is a round end of half the slot width so the cable finds its way in and does not
    snag. The cut reaches past the end, so it leaves no skin on the edge of the plate.
    """
    edge = sign * 0.5 * g["length"]
    r = 0.5 * g["slot"]
    center = edge - sign * (g["depth"] - r)         # the round end reaches `depth` into the plate
    a, b = center, edge + sign * MARGIN
    x0, x1 = min(a, b), max(a, b)
    cuts = [mf.Manifold.cylinder(g["thickness"] + 2.0 * MARGIN, r, r, SEGMENTS)
            .translate((center, 0.0, -MARGIN))]
    if x1 > x0:
        cuts.append(mf.Manifold.cube((x1 - x0, g["slot"], g["thickness"] + 2.0 * MARGIN))
                    .translate((x0, -0.5 * g["slot"], -MARGIN)))
    return cuts


def _keyhole(g) -> list:
    """The cable clamp: a keyhole cut from the bottom of the +x slot further into the plate.

    A notch of `KEYHOLE_W` millimetres wide runs `KEYHOLE_LEN` millimetres past the bottom of the
    slot along the long axis and ends in a round end, so the stripped end of the cable is pushed
    through the slot and pinched in the narrow part. The cut reaches into the slot, so the keyhole
    is open to it and not a hole buried in the plate.
    """
    bottom = 0.5 * g["length"] - g["depth"]         # the closed end of the +x slot
    far = bottom - KEYHOLE_LEN                       # the round end, deeper into the plate
    r = 0.5 * KEYHOLE_W
    return [mf.Manifold.cylinder(g["thickness"] + 2.0 * MARGIN, r, r, SEGMENTS)
            .translate((far, 0.0, -MARGIN)),
            mf.Manifold.cube((bottom + 0.5 * g["slot"] - far, KEYHOLE_W,
                              g["thickness"] + 2.0 * MARGIN))
            .translate((far, -r, -MARGIN))]


def _letters(g) -> list:
    """The outlines of the label, centred on the origin and shrunk to fit the middle part.

    Empty when there is no text, when this core runs without a Qt app or without fonts, or when the
    text yields nothing: a winder without a label is not an error. The letters are `LETTER_HEIGHT`
    millimetres high and made smaller when they would run past the middle part less `LABEL_MARGIN`,
    and with the cable clamp less its notch on both ends, so the letters stay centred and clear
    of it.
    """
    if not g["label"].strip():
        return []
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(g["label"], height=LETTER_HEIGHT)
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    high = float(allp[:, 1].max() - allp[:, 1].min())
    clamp = KEYHOLE_LEN if g["keyhole"] else 0.0
    room = (g["middle"] - 2.0 * (LABEL_MARGIN + clamp), g["width"] - 2.0 * LABEL_MARGIN)
    scale = 1.0
    if wide > 1e-9 and room[0] > 0.0:
        scale = min(scale, room[0] / wide)
    if high > 1e-9 and room[1] > 0.0:
        scale = min(scale, room[1] / high)
    mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    return [np.ascontiguousarray((np.asarray(L, dtype=float) - mid) * scale) for L in loops]


def _engraved(g, plate) -> mf.Manifold:
    """The plate with the label cut `ENGRAVE` millimetres deep in its top face, centred on x = 0."""
    loops = _letters(g)
    if not loops:
        return plate
    letters = mf.CrossSection(loops, mf.FillRule.NonZero)
    cut = mf.Manifold.extrude(letters, ENGRAVE + MARGIN) \
        .translate((0.0, 0.0, g["thickness"] - ENGRAVE))
    return plate - cut


def _centred(m) -> mf.Manifold:
    """The winder on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def cable_winder(length=80.0, width=40.0, thickness=3.0, slot=10.0, depth=15.0, keyhole=True,
                 label="") -> mf.Manifold:
    """One printable cable winder: an H-shaped plate to wind a cable round.

    The plate is `length` millimetres long, `width` wide and `thickness` thick, lying flat on the
    bed with its outside corners rounded by `CORNER` millimetres. A slot of `slot` millimetres wide
    and `depth` deep is cut in the middle of each short end, its closed end rounded to a half
    circle, so the plate looks like an H and the cable winds round the middle part between the two
    slots. With `keyhole` a small notch is cut from the bottom of one slot to pinch the end of the
    cable. `label` engraves text `ENGRAVE` millimetres deep in the top face, centred on the middle
    part and shrunk to fit it; empty text engraves nothing.

    The result is one closed volume on `z = 0`, centred in x and y, with no overhang: every wall is
    vertical and every other face horizontal. Every number is clamped to what prints - the slot to
    the plate less two flanges and the depth to a third of the length - and anything but a number
    is a ValueError.
    """
    g = _sizes(length, width, thickness, slot, depth, keyhole, label)
    cuts = _slot(g, 1.0) + _slot(g, -1.0)
    if g["keyhole"]:
        cuts += _keyhole(g)
    plate = _plate(g) - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
    return _centred(_engraved(g, plate))
