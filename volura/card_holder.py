"""Memory card holders: the computational core of a block that stands cards and sticks upright.

No Qt in this file. It gives the holder as one manifold, so it can be tested on its own and reused
in the model and the export.

A **memory card holder** is a block that lies on the bed with the cards standing in it, so the SD
card of the camera, the microSD card of the drone and the USB stick of the laptop stand in a row on
the desk and the one you want is the one you see. Every card stands upright on its short edge
and is held by its thickness: a **slot** of `width + 2 * fit` millimetres along X and
`thickness + 2 * fit` along Y, open at the top, so the card goes in with its face towards you.

The slots of one kind are a **column**: `count` of them behind each other along Y, `pitch`
millimetres apart. The columns stand side by side along X in the order SD, microSD and USB, and
`rows` puts a second one of everything behind the first. A kind without cards is left out and the
block is narrower for it. An SD or a USB slot is `depth` millimetres deep; a microSD slot is never
deeper than `MICRO_DEPTH`, because that card is only 15 mm tall and has to stick out far enough to
be grabbed.

The block keeps `WALL` millimetres of wall around everything, `GAP` millimetres between two columns
and `FLOOR` millimetres of floor under the deepest slot, so the floor of every slot is that much
above the bed. Its corners are rounded and its top edge is chamfered, the mouth of every slot has a
lead-in of 45 degrees and in front of every column the top dips down in a **finger notch**: a
half-round hollow, `NOTCH` millimetres wide and `NOTCH_DEPTH` deep, in which a fingertip comes down
in front of the card that stands there so you can pinch it out. Everything of that hollow looks up,
so the holder prints without support.
"""
from __future__ import annotations

import math

import manifold3d as mf

from . import enclosure
from .i18n import tr

#: how many cards of a kind there are, and how many rows of columns there are
SD_MIN, SD_MAX = 0, 12
MICRO_MIN, MICRO_MAX = 0, 12
USB_MIN, USB_MAX = 0, 12
ROWS_MIN, ROWS_MAX = 1, 2
#: mm deep a slot is, and mm of play between a card and its slot
DEPTH_MIN, DEPTH_MAX = 6.0, 20.0
FIT_MIN, FIT_MAX = 0.1, 0.6
#: the cards: mm wide along X, mm thick along Y and mm between the centres of two slots of one
#: column
SD_W, SD_T, SD_PITCH = 24.0, 2.1, 4.5
MICRO_W, MICRO_T, MICRO_PITCH = 11.0, 0.7, 4.5
USB_W, USB_T, USB_PITCH = 12.0, 4.5, 8.0
#: mm a microSD slot is never deeper than, so the card still stands out of the block
MICRO_DEPTH = 8.0
#: mm of wall around the slots, mm of wall between two columns and mm of floor under the deepest
#: slot
WALL, GAP, FLOOR = 3.0, 2.0, 1.5
#: mm wide and mm deep the dip in front of a column is, so the card there can be pinched out
NOTCH, NOTCH_DEPTH = 12.0, 3.0
#: mm the corners of the block are rounded to, and mm of chamfer on its top edge
CORNER, CHAMFER = 2.0, 0.5
#: mm of chamfer at the mouth of a slot, so a card finds its way in
MOUTH = 0.5
#: mm past a cut, so it leaves no skin behind
MARGIN = 0.5
#: mm of slab at either end of a chamfer, of which the hull of a chamfer is made
SLIVER = 0.2
#: how round the finger notch is
SEGMENTS = 64
#: the kinds of card, from left to right: their name, their width, their thickness and the pitch
#: of their column
KINDS = (("sd", SD_W, SD_T, SD_PITCH), ("micro", MICRO_W, MICRO_T, MICRO_PITCH),
         ("usb", USB_W, USB_T, USB_PITCH))


def _number(value, lo, hi):
    """A number of the holder, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a memory card holder must be numbers"))
    return max(lo, min(hi, v))


def _count(value, lo, hi) -> int:
    """How many cards of a kind there are, or how many rows there are: a whole number of them."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the number of cards of a memory card holder must be a whole number"))
    return max(lo, min(hi, int(v)))


def _kinds(counts, fit, depth) -> list:
    """The kinds that have cards, from left to right: how big their slot is and where their column
    stands.

    A column is one place along X with all the slots of one kind behind each other along Y. The
    columns stand side by side in the order SD, microSD and USB with `GAP` millimetres of wall
    between two of them, and the whole of it is centred on the origin. A slot is `fit` millimetres
    wider and thicker than its card on every side, and a microSD slot is never deeper than
    `MICRO_DEPTH`.
    """
    out, x = [], 0.0
    for name, w, t, pitch in KINDS:
        n = counts[name]
        if not n:
            continue
        sx, sy = w + 2.0 * fit, t + 2.0 * fit
        out.append({"name": name, "count": n, "sx": sx, "sy": sy, "pitch": pitch,
                    "depth": min(depth, MICRO_DEPTH) if name == "micro" else depth,
                    "x": x + 0.5 * sx})
        x += sx + GAP
    for k in out:                                # the columns as a whole stand on the origin
        k["x"] -= 0.5 * (x - GAP)
    return out


def _sizes(sd, micro, usb, rows, fit, depth) -> dict:
    """The clamped numbers and the places of the slots, in the frame the holder prints in.

    In that frame the block lies on the bed and is centred on the origin in plan: it is `width` by
    `length` and `height` high. The first slot of a column stands with its front edge on `y_edge`,
    `WALL` millimetres behind the front of the block, and the columns reach `reach` millimetres
    back from there; a second row of everything begins `step` millimetres behind the first, which
    is as far as the longest column reaches plus a wall.
    """
    counts = {"sd": _count(sd, SD_MIN, SD_MAX), "micro": _count(micro, MICRO_MIN, MICRO_MAX),
              "usb": _count(usb, USB_MIN, USB_MAX)}
    if not any(counts.values()):
        raise ValueError(tr("a memory card holder needs at least one slot"))
    r = _count(rows, ROWS_MIN, ROWS_MAX)
    f = _number(fit, FIT_MIN, FIT_MAX)
    d = _number(depth, DEPTH_MIN, DEPTH_MAX)
    kinds = _kinds(counts, f, d)
    reach = max(k["pitch"] * (k["count"] - 1) + k["sy"] for k in kinds)
    step = reach + GAP
    width = sum(k["sx"] for k in kinds) + GAP * (len(kinds) - 1) + 2.0 * WALL
    length = (r - 1) * step + reach + 2.0 * WALL
    height = FLOOR + max(k["depth"] for k in kinds)
    return {"sd": counts["sd"], "micro": counts["micro"], "usb": counts["usb"], "rows": r,
            "fit": f, "depth": d, "width": width, "length": length, "height": height,
            "reach": reach, "step": step, "y_edge": -0.5 * length + WALL, "kinds": kinds}


def sizes(sd=4, micro=6, usb=0, rows=1, fit=0.3, depth=12.0) -> dict:
    """The numbers a memory card holder is made of, so a dialog can show them.

    `sd`, `micro` and `usb` are how many cards of a kind stand in the block, `rows` how many
    rows of columns there are, `depth` how deep the slots are and `fit` the play between a card
    and its slot. At least one card has to be asked for; a microSD slot is never deeper than
    `MICRO_DEPTH` millimetres and `width`, `length` and `height` are the block around it. Every
    number is clamped to what prints; anything but a number is a ValueError.
    """
    s = _sizes(sd, micro, usb, rows, fit, depth)
    return {key: s[key] for key in ("sd", "micro", "usb", "rows", "fit", "depth",
                                    "width", "length", "height")}


def _slots(s) -> list:
    """Every slot of the holder: where its centre is, how big it is and how deep it is.

    The slots of a column follow each other along Y at the pitch of that kind, and a row of
    everything stands `step` millimetres behind the row in front.
    """
    out = []
    for k in s["kinds"]:
        for r in range(s["rows"]):
            for i in range(k["count"]):
                y = s["y_edge"] + 0.5 * k["sy"] + r * s["step"] + i * k["pitch"]
                out.append({"x": k["x"], "y": y, "sx": k["sx"], "sy": k["sy"],
                            "depth": k["depth"], "name": k["name"]})
    return out


def _box(length, width, radius, height, z=0.0) -> mf.Manifold:
    """A box `length` by `width` with corners of `radius`, `height` high, standing at `z` and
    centred on the origin in plan."""
    return mf.Manifold.extrude(enclosure.rounded_rect(length, width, radius),
                               height).translate((0.0, 0.0, z))


def _chamfered(length, width, radius, height, z=0.0) -> mf.Manifold:
    """The same box with its top edge chamfered `CHAMFER` millimetres.

    The chamfer is the hull of a thin slab of the one outline and a thin slab of the same outline
    inset by `CHAMFER`: inset a rounded rectangle and its corners get a `CHAMFER` tighter, so the
    hull leans 45 degrees along the whole edge, also round the corner.
    """
    c = min(CHAMFER, 0.5 * height, radius)
    parts = [_box(length, width, radius, height - c, z)]
    parts.append(mf.Manifold.batch_hull([_box(length, width, radius, SLIVER, z + height - c),
                                         _box(length - 2.0 * c, width - 2.0 * c, radius - c,
                                              SLIVER, z + height - SLIVER)]))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _block(s) -> mf.Manifold:
    """The block on its own: the footprint with rounded corners and a chamfered top edge."""
    return _chamfered(s["width"], s["length"], CORNER, s["height"])


def _slot(s, x, y, sx, sy, depth) -> mf.Manifold:
    """The prism of a slot: `sx` by `sy` millimetres, from `depth` millimetres below the top of the
    block to `MARGIN` above it, so the mouth is open and nothing is left behind."""
    return mf.Manifold.cube([sx, sy, depth + MARGIN]).translate((x - 0.5 * sx, y - 0.5 * sy,
                                                                 s["height"] - depth))


def _mouth(s, x, y, sx, sy) -> mf.Manifold:
    """The lead-in at the mouth of a slot: a chamfer of `MOUTH` millimetres at 45 degrees, so a
    card finds its way in instead of standing on the edge of the wall.

    A hull is made of the two ends it runs between, so the wide end is a `SLIVER` wider than the
    chamfer to make the hull pass `MOUTH` wide over the top of the block.
    """
    w, d = sx + 2.0 * (MOUTH + SLIVER), sy + 2.0 * (MOUTH + SLIVER)
    wide = mf.Manifold.cube([w, d, SLIVER]).translate((x - 0.5 * w, y - 0.5 * d, s["height"]))
    deep = mf.Manifold.cube([sx, sy, SLIVER]).translate((x - 0.5 * sx, y - 0.5 * sy,
                                                         s["height"] - MOUTH))
    return mf.Manifold.batch_hull([wide, deep])


def _notch(s, k) -> mf.Manifold:
    """The dip in front of a column: a half-round hollow in the top, `NOTCH` millimetres wide and
    `NOTCH_DEPTH` deep, that runs from the front of the block to the card that stands there.

    The hollow is a piece off a cylinder whose axis lies along Y above the block, so its floor is
    an arc that rises to the top at both sides and everything of it looks up. A fingertip comes
    down in front of the card and pinches it out.
    """
    r = (NOTCH ** 2 + 4.0 * NOTCH_DEPTH ** 2) / (8.0 * NOTCH_DEPTH)
    y0 = -0.5 * s["length"] - MARGIN
    return mf.Manifold.cylinder(s["y_edge"] + 0.5 * k["sy"] - y0, r, r, SEGMENTS) \
        .rotate((-90.0, 0.0, 0.0)) \
        .translate((k["x"], y0, s["height"] - NOTCH_DEPTH + r))


def card_holder(sd=4, micro=6, usb=0, rows=1, fit=0.3, depth=12.0) -> mf.Manifold:
    """One printable memory card holder, lying on `z = 0` and centred on the origin in plan.

    The block is `width` by `length` and `height` high, with `WALL` millimetres of wall around the
    slots and `FLOOR` millimetres of floor under the deepest of them. It holds `sd` SD cards,
    `micro` microSD cards and `usb` USB sticks, every kind in a column of its own and every card in
    a slot of its own size with `fit` millimetres of play, open at the top and chamfered at its
    mouth; `rows` puts a second row of columns behind the first.

    Every number is clamped to what prints, a kind without cards is left out, a microSD slot stays
    shallow so the card still sticks out, and nothing hangs over steeper than 45 degrees; anything
    but a number, or a holder without a single slot, is a ValueError.
    """
    s = _sizes(sd, micro, usb, rows, fit, depth)
    cutters = [_notch(s, k) for k in s["kinds"]]
    for k in _slots(s):
        cutters.append(_slot(s, k["x"], k["y"], k["sx"], k["sy"], k["depth"]))
        cutters.append(_mouth(s, k["x"], k["y"], k["sx"], k["sy"]))
    return _block(s) - mf.Manifold.batch_boolean(cutters, mf.OpType.Add)
