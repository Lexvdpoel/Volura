"""Bin wall rack: the computational core for a plate on the wall that the storage bins hang on.

No Qt in this file. It gives the rack as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **bin wall rack** is a grid on the wall: `cols` by `rows` cells of `cell_w` by `cell_h`
millimetres. The plate is `plate` millimetres thick and is printed lying on its back, so the side
against the wall is on the bed. Every row has one **rail** across the whole width, at the top of
the row: a cleat of `CLEAT_THICK` millimetres that stands `CLEAT_OUT` millimetres out of the plate
and leans 45 degrees upwards, so between the rail and the plate a groove opens to the top. The hook
on the back of a storage bin (`storage_bin(hanger=True)`) drops in that groove and its weight
pulls it against the plate, so a bin hangs anywhere along a rail and slides sideways to make room.

The rail is placed so a bin hangs with its top edge level with the top of its row, so every bin up
to `cell_h` high fits in a row without touching the bin under it. The rail prints without support:
its underside, as it lies on the bed, stands at 45 degrees.

The **screw holes** hold the rack on the wall: in every row, halfway the free band under the rail,
one hole in the first and one in the last column (one in the middle with a single column), with a
90 degree countersink on the front, so a flat head screw sinks into the plate and a bin hangs flat
against it. With `screw = "none"` the plate has no holes, for a rack that is glued or taped.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

from . import threads
from .geom import extrude_cs
from .i18n import tr
from .storage_bin import CLEAT_OUT, CLEAT_THICK, FRONT

#: the number of cells across and up: from one bin to a wall full of them
COLS_MIN, COLS_MAX = 1, 10
ROWS_MIN, ROWS_MAX = 1, 10
#: mm of a cell across: the length of a bin with some room on both sides
CELL_W_MIN, CELL_W_MAX = 30.0, 300.0
#: mm of a cell up: the height of a bin, which has to leave room for the rail and a screw
CELL_H_MIN, CELL_H_MAX = 30.0, 200.0
#: mm of plate: thin enough to print fast, thick enough for the countersink of the screw
PLATE_MIN, PLATE_MAX = 2.0, 8.0
#: the screws the rack is mounted with; "none" leaves the plate whole
SCREWS = ("none", "M3", "M4")
#: mm of plate that stays under the countersink, so the cone does not cut through
PLATE_UNDER_SINK = 0.6
#: mm into the plate that a rail starts, so the two are one piece instead of only touching
MARGIN = 0.5


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so the rack is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _count(n, lo, hi):
    return max(lo, min(hi, int(n)))


def rail_height() -> float:
    """mm from the top of a row down to the bottom of its rail: a hook, a rail and the 45 degrees."""
    return 2.0 * CLEAT_THICK + CLEAT_OUT


def _rail(y_top, x_lo, x_hi, plate) -> mf.Manifold:
    """The rail under `y_top`, the top of its row, from `x_lo` to `x_hi`, on the plate.

    In the plane across the rail (along the wall, out of the wall) the rail is a parallelogram: it
    stands on the plate from `y0` to `y0 + CLEAT_THICK` and leans 45 degrees up to `CLEAT_OUT`
    out of it, so the groove between its upper face and the plate opens to the top.
    """
    y0 = y_top - rail_height()
    t, out = CLEAT_THICK, CLEAT_OUT
    loop = [[y0 - MARGIN, plate - MARGIN], [y0 + t - MARGIN, plate - MARGIN],
            [y0 + t + out, plate + out], [y0 + out, plate + out]]
    return extrude_cs(mf.CrossSection([loop]), FRONT, x_lo, x_hi)


def screw_places(cols=3, rows=2, cell_w=85.0, cell_h=50.0) -> list:
    """The (x, y) of every screw hole, with the rack from (0, 0) to its width and height."""
    nc = _count(cols, COLS_MIN, COLS_MAX)
    nr = _count(rows, ROWS_MIN, ROWS_MAX)
    w = _clamp(cell_w, CELL_W_MIN, CELL_W_MAX)
    h = _clamp(cell_h, CELL_H_MIN, CELL_H_MAX)
    xs = [0.5 * w] if nc == 1 else [0.5 * w, (nc - 0.5) * w]
    band = h - rail_height()                    # the free part of a row, under its rail
    return [(x, j * h + 0.5 * band) for j in range(nr) for x in xs]


def bin_rack(cols=3, rows=2, cell_w=85.0, cell_h=50.0, plate=4.0, screw="M4") -> mf.Manifold:
    """A plate of `cols` by `rows` cells with a rail at the top of every row, for hanging bins.

    The plate is `cols cell_w` by `rows cell_h` millimetres and `plate` millimetres thick and lies
    on `z = 0` with the side against the wall down; the rails stand `CLEAT_OUT` millimetres out of
    it, leaning 45 degrees to the top of the rack, which is `+y`. A bin with the wall hook, as high
    as `cell_h` or lower, hangs in a row with its top edge level with the top of that row. The
    screw holes are countersunk on the front for a flat head `screw`, one of `SCREWS`, and the
    plate is made thick enough for the countersink.

    The result is centred on the origin in plan. The sizes are kept within their own bounds, so a
    rack that cannot be printed becomes the nearest one that can; a screw that is not in `SCREWS`
    is refused with a ValueError.
    """
    size = str(screw)
    if size not in SCREWS:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    nc = _count(cols, COLS_MIN, COLS_MAX)
    nr = _count(rows, ROWS_MIN, ROWS_MAX)
    w = _clamp(cell_w, CELL_W_MIN, CELL_W_MAX)
    h = _clamp(cell_h, CELL_H_MIN, CELL_H_MAX)
    p = _clamp(plate, PLATE_MIN, PLATE_MAX)
    if size != "none":
        hole_d, _socket_d, _head_h, cs_head_d = threads.SCREW_HOLES[size]
        sink = 0.5 * (cs_head_d - hole_d)       # the countersink is a 90 degree cone
        p = max(p, sink + PLATE_UNDER_SINK)
    width, height = nc * w, nr * h

    body = mf.Manifold.cube([width, height, p])
    rails = [_rail((j + 1) * h, 0.0, width, p) for j in range(nr)]
    body = mf.Manifold.batch_boolean([body] + rails, mf.OpType.Add)
    if size != "none":
        for x, y in screw_places(nc, nr, w, h):
            body = threads.screw_hole(body, (x, y, p), (0.0, 0.0, -1.0), p, size, "countersink")
    # centred in plan, with the side against the wall on the bed
    bb = np.asarray(body.bounding_box(), dtype=float)
    return body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
