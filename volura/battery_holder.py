"""Battery holders: the computational core for a printable holder that keeps cells in a row.

No Qt in this file. It gives the holder as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **battery holder** is the block that keeps one or more round cells side by side: the cells are
pressed into the block from above and the block holds them by friction, so a pack of AA or a single
18650 stays where it belongs in a printed case, and nothing has to be screwed or glued.

The **cradles** are what the cells lie in: `count` horizontal half cylinders side by side, each
`d + 2 clearance` millimetres across and `L + 2 clearance` millimetres long (`d` and `L` are the
diameter and the length of the cell), cut out of the top of the block with their axis one cradle
radius plus `wall` millimetres above the bed. The block stands so far over the cells that the
opening between the two lips over a cradle is `d * (1 - grip)` millimetres across: narrower than
the cell, whatever the clearance, so the cell presses past the lips and snaps in and does not fall
out when the holder is turned over. With `grip = 0` the top stands on the axis, the cradle is open
all the way to it and the cell only lies in it. The play all round the cell is `clearance`.

The **walls** are what holds the holder together: `wall` millimetres between the cradles, around
them and under them, so the block is `L + 2 clearance + 2 wall` millimetres long,
`count (d + 2 clearance) + (count + 1) wall` millimetres wide and `floor + r + h` millimetres high,
with `r` the radius of a cradle, `h` the height above the axis on which a cradle has narrowed to
`d * (1 - grip)` and `floor` the wall under the cradles, which is `wall` unless a screw head needs
to sink in deeper than that.

The **contacts** are where the wires come in: at both ends of every cradle a vertical slot of
`SLOT_WIDE` millimetres goes through the end wall, from `0.4 d` under the axis of the cell up to the
top of the block, for a spring contact or a wire.

The **mounting holes** are how the holder is screwed to a base: one hole under the axis of the first
cradle near the `-x` end and one under the axis of the last cradle near the `+x` end, so the two
holes lie diagonally in the holder and always open in the middle of a cradle, also when the number
of cells is even. Every hole is the clearance hole of `screw` with a 90 degree countersink that opens
into the cradle, so a flat head screw sinks under the cell and the holder still lies flat on the
base; the floor under the cradles is thick enough that the cone does not cut through it. With
`screw = "none"` the floor is left alone.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr
from . import threads

#: the cells a holder is made for, as (diameter, length) in millimetres
CELLS = {"AA": (14.5, 50.5), "AAA": (10.5, 44.5), "18650": (18.6, 65.2),
         "21700": (21.7, 70.2), "CR123A": (17.0, 34.5)}
#: the fewest cell that is worth a holder and the most that still fits on a small bed
COUNT_MIN, COUNT_MAX = 1, 8
#: the thinnest wall that grips a cell and the thickest that is still a holder, not a box
WALL_MIN, WALL_MAX = 1.2, 4.0
#: the least play a cell still presses in with and the most that still holds it
CLEARANCE_MIN, CLEARANCE_MAX = 0.1, 1.0
#: how much narrower the opening over a cradle is than the cell, as a part of the cell diameter:
#: 0 leaves the cradle open to its axis, and at the most the cell snaps in and cannot fall out
GRIP_MIN, GRIP_MAX = 0.0, 0.15
#: mm across the slot in the end wall, which is what a spring contact sits in
SLOT_WIDE = 1.0
#: the slot as a part of the cell diameter, so a contact reaches the cell without touching it
SLOT_PART = 0.8
#: the screws the holder is mounted with; "none" leaves the floor alone
SCREWS = ("none", "M2.5", "M3", "M4")
#: mm of floor that stays under the countersink, so the cone does not cut through
FLOOR_UNDER_SINK = 0.6
#: mm between an end wall and the edge of the countersink, so the hole stays in the cradle
HOLE_ROOM = 1.0
#: the straight pieces a cradle is approximated with
SEGMENTS = 64
#: mm past a cut face, so a cut leaves no skin behind
MARGIN = 0.5


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so the holder is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _cell(cell):
    """The (diameter, length) of `cell`, or a ValueError when that cell is not one of `CELLS`."""
    try:
        return CELLS[str(cell)]
    except KeyError:
        raise ValueError(tr("unknown battery cell {cell}").format(cell=cell)) from None


def _cradle(x0, x1, r, y, z) -> mf.Manifold:
    """The piece of cradle along X from `x0` to `x1`, round the axis at `y`, `z`, radius `r`."""
    return mf.Manifold.cylinder(x1 - x0, r, r, SEGMENTS).rotate((0.0, 90.0, 0.0)) \
        .translate((x0, y, z))


def _mount(screw):
    """(hole radius, head radius, cone depth) of the mounting screw, or None for no holes."""
    size = str(screw)
    if size == "none":
        return None
    if size not in SCREWS:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    hole_d, _socket_d, _head_h, cs_head_d = threads.SCREW_HOLES[size]
    r_hole = 0.5 * (hole_d + threads.DEFAULT_HOLE_CLEARANCE)
    r_head = 0.5 * (cs_head_d + threads.DEFAULT_HOLE_CLEARANCE)
    return r_hole, r_head, r_head - r_hole


def battery_holder(cell="AA", count=2, wall=1.6, clearance=0.4, grip=0.05,
                   screw="M3") -> mf.Manifold:
    """A printable holder that keeps `count` cells of `cell` side by side.

    The holder is a block of `L + 2 clearance + 2 wall` by `count (d + 2 clearance) + (count + 1)
    wall` millimetres and `floor + r + h` millimetres high, in which `count` cradles of `d + 2
    clearance` millimetres across are cut out of the top, `L + 2 clearance` millimetres long, with
    their axis `floor + r` millimetres above the bed (`d` and `L` are the diameter and the length of
    the cell and `r` is `0.5 d + clearance`). The block stands `h` millimetres over that axis, and
    `h` is the height on which a cradle has narrowed to `d * (1 - grip)` millimetres across, so the
    opening between the two lips over a cradle is that much narrower than the cell, whatever the
    clearance, and the cell presses past the lips and snaps in; `grip = 0` puts the top on the axis,
    which leaves the cradle open and the cell only lying in it. The cell has `clearance` millimetres
    of play all round it, while `wall` millimetres of plastic stay between the cradles, around them
    and under them, and `floor` is that wall or as much as the countersink of a screw needs, so the
    cone never cuts through the bottom. At both ends of every cradle a slot goes through the end wall
    for a spring contact or a wire, and two holes mount the holder: one under the axis of the first
    cradle near the `-x` end and one under the axis of the last cradle near the `+x` end, so they lie
    diagonally and always in the middle of a cradle. Every hole is the clearance hole of `screw`, one
    of `SCREWS`, with a 90 degree countersink that opens into the cradle, so a flat head screw sinks
    under the cell; `screw = "none"` leaves the floor alone.

    The result is centred on the origin in plan and stands on `z = 0`. `count` is kept within
    `COUNT_MIN` to `COUNT_MAX` and `wall`, `clearance` and `grip` within their own bounds; a cell
    that is not in `CELLS` or a screw that is not in `SCREWS` is refused with a ValueError.
    """
    d, ln = _cell(cell)
    n = max(COUNT_MIN, min(COUNT_MAX, int(count)))
    t = _clamp(wall, WALL_MIN, WALL_MAX)
    c = _clamp(clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    g = _clamp(grip, GRIP_MIN, GRIP_MAX)
    mount = _mount(screw)
    r = 0.5 * d + c                       # the cradle: the cell with room all round it
    x_len = ln + 2.0 * c + 2.0 * t        # the cell, its play and an end wall on both sides
    y_len = n * 2.0 * r + (n + 1) * t     # the cradles and a wall between and around them
    # the wall under the cradles, or as much as the head needs to sink in without breaking through
    floor = t if mount is None else max(t, mount[2] + FLOOR_UNDER_SINK)
    axis = floor + r                      # the cells lie with their axis on this height
    # how far the block has to stand over the axis for the lips to leave an opening of d * (1 - g)
    lip = 0.0 if g <= GRIP_MIN else math.sqrt(r * r - (0.5 * d * (1.0 - g)) ** 2)
    top = axis + lip                      # the block stands over the cells, which is what grips them

    # the slot through the end wall, open at the top, for a spring contact or a wire
    slot_in = 0.5 * (ln + 2.0 * c) - MARGIN       # just inside the cradle
    slot_out = 0.5 * x_len + MARGIN               # just past the outside of the end wall
    slot_low = axis - 0.5 * SLOT_PART * d         # at the height of the cell
    cut = []
    for i in range(n):
        y = (i - 0.5 * (n - 1)) * (2.0 * r + t)
        # the cradle stops short of the end walls, which is what keeps the cell from walking out
        cut.append(_cradle(-0.5 * (ln + 2.0 * c), 0.5 * (ln + 2.0 * c), r, y, axis))
        slot = mf.Manifold.cube([slot_out - slot_in, SLOT_WIDE, top + MARGIN - slot_low]).translate(
            (slot_in, y - 0.5 * SLOT_WIDE, slot_low))
        cut.append(slot)
        cut.append(slot.translate((-slot_in - slot_out, 0.0, 0.0)))

    # the block stands centred on the origin, so the cradles and the holes are placed from its middle
    block = mf.Manifold.cube([x_len, y_len, top]).translate((-0.5 * x_len, -0.5 * y_len, 0.0))
    body = block - mf.Manifold.batch_boolean(cut, mf.OpType.Add)
    if mount is not None:
        # a flat head screw mounts the holder through the floor, sunk under the cell: the first
        # cradle at one end and the last at the other, so the two holes lie diagonally
        r_hole, r_head, _depth = mount
        x = 0.5 * (ln + 2.0 * c) - (r_head + HOLE_ROOM)
        for side, i in ((-1.0, 0), (1.0, n - 1)):
            y = (i - 0.5 * (n - 1)) * (2.0 * r + t)
            body = threads.screw_hole(body, (side * x, y, floor), (0.0, 0.0, -1.0), floor,
                                      str(screw), "countersink", segments=SEGMENTS)
    # the holder stands on the bed as it prints: centred in plan, with the floor on z = 0
    bb = np.asarray(body.bounding_box(), dtype=float)
    return body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
