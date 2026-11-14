"""Hex key holders: the computational core of a block that stands a set of Allen keys upright.

No Qt in this file except the sizes, which come from `fonts` the way a label plate does. It gives
the holder as one closed volume, so it can be tested on its own and reused in the model and the
export.

A **hex key holder** keeps a set of hex (Allen) keys upright with the short leg down in a hexagonal
hole, so the set stands on the bench and the key you want is the one you read off the top.
`sizes_mm` names the keys by their size across the flats in millimetres - "1.5 2 2.5 3" is a set of
four - and they stand in one row, smallest to the left.

Every hole is a hexagon of the key size plus `clearance` millimetres across the flats, `depth`
millimetres deep, cut from the top face down. The hexagon stands with a flat towards the front, so
a key turns the same way every time and the plastic in front of it lies flat for its size. Two
holes keep `gap` millimetres of plastic between them and the row keeps `margin` millimetres all
round: the block is as long as the row needs, `depth + FLOOR` high - `FLOOR` millimetres of plastic
stay under the deepest hole - and as wide as the biggest hole plus two margins. Its outside
vertical edges round with `CORNER` millimetres.

A set of keys grows with its sizes, so the 1.5 mm key is a short one. With `steps` the top of the
block steps down from the biggest key to the smallest, one flat step per hole, in proportion to the
sizes and never more than half the depth per step lower than the next. Every hole keeps its full
`depth` below its own step, so the keys stick out as far as each other and a small key does not
have to be fished out of a deep block. Without `steps` the top is flat.

The **sizes** are engraved `ENGRAVE` millimetres deep in the top face in front of their own hole, in
letters of `LETTER_HEIGHT` millimetres and made smaller when they would run past their step or out
of the strip in front of the hole.

The holder prints on its bottom: every wall is pulled straight up and every step, every hole floor
and every engraved letter looks up, so nothing hangs over and nothing needs support. Every number is
clamped to what prints; the sizes are read out of the text - numbers separated by spaces or commas,
junk left out, sizes that are no hex key dropped, a size twice counted once and at most `COUNT_MAX`
of them - and a text without a single key is a ValueError. The result is one closed volume on
`z = 0`, centred in x and y.
"""
from __future__ import annotations

import math
import re

import manifold3d as mf
import numpy as np

from . import enclosure
from .i18n import tr

#: the keys a holder starts with, across their flats in millimetres
SIZES_TEXT = "1.5 2 2.5 3 4 5 6 8 10"
#: mm the smallest and the biggest hex key that is worth a hole of its own
SIZE_MIN, SIZE_MAX = 0.7, 19.0
#: how many keys a holder holds at most
COUNT_MAX = 15
#: mm of play between a key and the wall of its hole
CLEARANCE_MIN, CLEARANCE_MAX = 0.1, 0.6
#: mm the shallowest and the deepest hole
DEPTH_MIN, DEPTH_MAX = 6.0, 30.0
#: mm of plastic between two holes
GAP_MIN, GAP_MAX = 2.0, 10.0
#: mm of plastic round the row
MARGIN_MIN, MARGIN_MAX = 3.0, 10.0
#: mm of plastic that stays under the deepest hole
FLOOR = 3.0
#: the whole drop of the stepped top, as a fraction of the depth of a hole
STEP_DROP = 0.5
#: mm deep the sizes are engraved in the top face, and mm high their letters
ENGRAVE, LETTER_HEIGHT = 0.6, 3.5
#: mm of top face that has to stay around an engraved size
LABEL_MARGIN = 1.0
#: mm the outside vertical edges round with, so nothing is sharp enough to cut a hand
CORNER = 2.0
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
OVERCUT = 0.5


def _number(value, lo, hi):
    """A number of the holder, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a hex key holder must be numbers"))
    return max(lo, min(hi, v))


def _keys(sizes_mm) -> list:
    """The sizes of the keys that get a hole, smallest first, read out of the text.

    `sizes_mm` names the keys by their size across the flats in millimetres, separated by spaces or
    commas. A word or any other junk is left out, a size outside `SIZE_MIN` to `SIZE_MAX` is no hex
    key and is dropped, and a size that comes twice is one hole; what is left is sorted and cut to
    `COUNT_MAX`, so the smallest keys of a long list are kept. A text without a single key is a
    ValueError.
    """
    found = set()
    for word in re.split(r"[,\s]+", str(sizes_mm or "")):
        try:
            v = float(word)
        except ValueError:
            continue
        if SIZE_MIN <= v <= SIZE_MAX:
            found.add(round(v, 6))
    if not found:
        raise ValueError(tr("a hex key holder needs at least one key size"))
    return sorted(found)[:COUNT_MAX]


def _sizes(sizes_mm, clearance, depth, gap, margin, steps, labels) -> dict:
    """The clamped numbers of the holder, the places of its holes and the height of its steps.

    The block lies on the bed, is centred on the origin in plan and stands `height` high. The keys
    are `keys` and hole `i` is a hexagon of `holes[i]` millimetres across the flats on the axis
    through (`centres[i]`, 0), cut from the top of the block above it at `tops[i]` down to
    `tops[i] - depth`. The top of the block above hole `i` runs from `edges[i]` to `edges[i + 1]`,
    which is also the room its engraved size has.

    The steps take `STEP_DROP` of the depth off the top at the smallest key and spread that over the
    holes in proportion to their sizes, so no step stands more than half a depth below the next one
    and the lowest step is `depth + FLOOR` above the bed.
    """
    keys = _keys(sizes_mm)
    c = _number(clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    d = _number(depth, DEPTH_MIN, DEPTH_MAX)
    g = _number(gap, GAP_MIN, GAP_MAX)
    m = _number(margin, MARGIN_MIN, MARGIN_MAX)
    holes = [k + c for k in keys]
    # the hexagon stands with a flat to the front and to the back, so its corners point along the
    # row: that is how far a hole reaches in x, and where the next one may start
    corners = [h / math.sqrt(3.0) for h in holes]
    centres, edge = [], 0.0
    for r in corners:
        centres.append(edge + r)
        edge += 2.0 * r + g
    span = edge - g                                   # the row, corner to corner
    centres = [x - 0.5 * span for x in centres]
    drop = STEP_DROP * d if steps and holes[-1] > holes[0] else 0.0
    height = d + FLOOR + drop
    tops = [height - drop * (holes[-1] - h) / (holes[-1] - holes[0]) if drop > 0.0 else height
            for h in holes]
    length, width = span + 2.0 * m, holes[-1] + 2.0 * m
    edges = [-0.5 * length] + [0.5 * (a + b) for a, b in zip(centres, centres[1:])]
    edges.append(0.5 * length)
    return {"keys": keys, "clearance": c, "depth": d, "gap": g, "margin": m,
            "steps": bool(steps), "labels": bool(labels), "holes": holes, "centres": centres,
            "tops": tops, "edges": edges, "drop": drop, "length": length, "width": width,
            "height": height}


def sizes(sizes_mm=SIZES_TEXT, clearance=0.3, depth=12.0, gap=3.0, margin=4.0, steps=True,
          labels=True) -> dict:
    """The numbers a hex key holder is made of, so a dialog can show them.

    `sizes_mm` names the keys by their size across the flats in millimetres and `keys` are those
    sizes, sorted and clamped to what a holder holds; `clearance` is the play between a key and its
    hole and `depth` how deep a hole is cut from the top above it. `gap` is the plastic between two
    holes and `margin` the plastic round the row; `steps` says whether the top steps down towards
    the small keys and `labels` whether the sizes are engraved in front of them. `length`, `width`
    and `height` are the block. Every number is clamped to what prints; anything but a number is a
    ValueError, and so is a text without a single key size.
    """
    s = _sizes(sizes_mm, clearance, depth, gap, margin, steps, labels)
    return {key: s[key] for key in ("keys", "clearance", "depth", "gap", "margin", "steps",
                                    "labels", "length", "width", "height")}


def _hex(across) -> mf.CrossSection:
    """A hexagon `across` millimetres across the flats, with a flat to the front and to the back.

    Those two are the walls a key lies against: they look along Y, so the corners point along the
    row and the plastic between two holes is only `gap` wide where two corners face each other and
    widens towards the front and the back.
    """
    r = across / math.sqrt(3.0)                       # the corners stand that much out
    return mf.CrossSection([[(r * math.cos(k * math.pi / 3.0), r * math.sin(k * math.pi / 3.0))
                             for k in range(6)]])


def _block(s) -> mf.Manifold:
    """The block on its own: the footprint with its corners rounded, its top stepped down towards
    the small keys.

    A step is cut off the whole prism across the full width of the block, so it is one flat plane
    and the rounded corners of the footprint stay as they are. A cut reaches into the step next to
    it, which stands higher, so the steps leave no wall between them; the lowest step is the one
    that keeps `FLOOR` millimetres of plastic under its hole.
    """
    solid = mf.Manifold.extrude(enclosure.rounded_rect(s["length"], s["width"], CORNER),
                                s["height"])
    cuts = []
    for i, top in enumerate(s["tops"]):
        if s["height"] - top <= 1e-9:
            continue
        x1 = s["edges"][i + 1]
        cuts.append(mf.Manifold.cube([x1 - s["edges"][i] + OVERCUT, s["width"] + 2.0 * OVERCUT,
                                      s["height"] - top + OVERCUT])
                    .translate((s["edges"][i] - OVERCUT, -0.5 * s["width"] - OVERCUT, top)))
    if not cuts:
        return solid
    return solid - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def _letters(s) -> list:
    """The sizes to engrave, one list of loops per hole, with the height of the face they go in.

    Empty when the sizes are off, when this core runs without a Qt app or without fonts, or when a
    text yields nothing: a holder whose sizes are not engraved is not an error. A size stands on
    the axis of its own hole and in the strip between the front of the block and the flat wall of
    that hole; it is made smaller when it would run past its step or out of that strip, so the
    letters of a big key in a narrow margin are small but never cut away.
    """
    if not s["labels"]:
        return []
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    made = []
    for i, key in enumerate(s["keys"]):
        try:
            loops = fonts.text_loops(f"{key:g}", height=LETTER_HEIGHT)
        except Exception:
            continue
        if not loops:
            continue
        allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
        wide = float(allp[:, 0].max() - allp[:, 0].min())
        high = float(allp[:, 1].max() - allp[:, 1].min())
        x = s["centres"][i]
        room = (min(x - s["edges"][i], s["edges"][i + 1] - x) - LABEL_MARGIN,
                0.5 * (s["width"] - s["holes"][i]) - LABEL_MARGIN)
        scale = 1.0
        if wide > 1e-9 and room[0] > 0.0:
            scale = min(scale, room[0] / wide)
        if high > 1e-9 and room[1] > 0.0:
            scale = min(scale, room[1] / high)
        mid = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
        at = (x, -0.25 * (s["width"] + s["holes"][i]) + 0.5 * LABEL_MARGIN)
        made.append(([np.ascontiguousarray((np.asarray(L, dtype=float) - mid) * scale
                                           + np.array(at)) for L in loops], s["tops"][i]))
    return made


def _engraved(s, block) -> mf.Manifold:
    """The block with every size cut `ENGRAVE` millimetres deep in the top face above its hole."""
    cuts = []
    for loops, top in _letters(s):
        cs = mf.CrossSection(loops, mf.FillRule.NonZero)
        cuts.append(mf.Manifold.extrude(cs, ENGRAVE + OVERCUT).translate((0.0, 0.0,
                                                                          top - ENGRAVE)))
    if not cuts:
        return block
    return block - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def _centred(m) -> mf.Manifold:
    """The holder on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def hex_key_holder(sizes_mm=SIZES_TEXT, clearance=0.3, depth=12.0, gap=3.0, margin=4.0,
                   steps=True, labels=True) -> mf.Manifold:
    """One printable hex key holder: a block that stands a set of Allen keys upright.

    The keys are those of `sizes_mm`, smallest to the left, and every one of them stands in a
    hexagonal hole of its size plus `clearance` millimetres across the flats and `depth` millimetres
    deep, cut from the top face above it. Two holes keep `gap` millimetres of plastic between them
    and the row keeps `margin` millimetres all round; the outside vertical edges round with `CORNER`
    millimetres. With `steps` the top steps down from the biggest key to the smallest, so every key
    sticks out as far as the others, and with `labels` the sizes are engraved `ENGRAVE` millimetres
    deep in the top face in front of their own hole.

    The result is one closed volume on `z = 0`, centred in x and y, that prints on its bottom: every
    wall is vertical and every other face looks up. Every number is clamped to what prints, the
    sizes are read out of the text and a text without a single key size is a ValueError.
    """
    s = _sizes(sizes_mm, clearance, depth, gap, margin, steps, labels)
    cuts = [mf.Manifold.extrude(_hex(h), s["depth"] + OVERCUT)
            .translate((x, 0.0, top - s["depth"]))
            for x, top, h in zip(s["centres"], s["tops"], s["holes"])]
    block = _block(s) - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
    return _centred(_engraved(s, block))
