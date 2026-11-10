"""Gridfinity bins and baseplates: the computational core of the open modular storage standard.

No Qt in this file. It gives a bin and a baseplate as one closed volume each, so they can be
tested on their own and reused in the model and the export.

**Gridfinity** is a grid of squares of `GRID` millimetres on which bins stand: a bin is one or
more squares wide and a whole number of height units of `HEIGHT_UNIT` millimetres high, and a
baseplate is the plate that carries the grid. Nothing is glued or screwed: a bin stands on the
plate because the foot under every one of its squares drops into the hole of that square, and bins
stack because the lip on top of a bin is the same hole again. One standard, so a bin printed
yesterday still stands on a plate printed tomorrow, whoever printed either.

The **foot** is what makes the grid hold: the profile of a square, from the bed up, is a
45 degree chamfer of `FOOT_CHAMFER`, a vertical part of `FOOT_HEIGHT` and a 45 degree chamfer of
`NECK_CHAMFER`, `4.75` millimetres in all. It is narrow at the bottom and widest at the top, so
pushing the bin down pulls it into the plate instead of out of it, and the grid locates the bin in
plan. The hole of a baseplate is the same profile upside down, only `0.1` millimetre wider at the
bottom chamfer, which is the room the foot needs to go in and come out.

The **bin** stands on `4.75` millimetres of foot under every square of its `cols` by `rows`, and
above that the body is the outline of those squares, less `BIN_GAP` so a neighbour never touches
it, hollowed out above a floor of `floor` with walls of `wall`. `divisions` compartments along the
length are made by straight dividers of `wall` millimetres that stop `DIVIDER_DROP` millimetre
below the top edge, so you can still reach over them. With `magnets` the bottom of every foot gets
four round pockets, in which a magnet is glued flush, which is what keeps a bin on a steel shelf
and what holds a plate to a machine bed.

Two things make a bin easier to use. The **finger scoop** is a quarter-round fillet in the corner
between the floor and the inside of the front wall, over the whole inside length, so a small part
that lies flat is caught with a finger and slid out. The **label tab** is a shelf under the top of
the inside of the back wall, of `LABEL_DEPTH` millimetres, whose underside leans 45 degrees down
to the wall so it prints without support; a printed label stands against the wall under it and is
read from the front. Both are clipped to the inside, so a bin with them is the same size outside
and still stacks and stands on a plate.

The **stacking lip** on top of the wall is the top of a baseplate hole again, `LIP_HEIGHT`
millimetres high and standing on the wall, around the whole bin rather than around every square,
so nothing crosses the opening of a bin of more squares: the bin above drops into it with its
foot and is located by the same two chamfers, so a tower of bins stays a tower. The lip leans out over the
opening at its bottom and is widest at the top, which is also why nothing that stands in the bin
sticks out of it.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

from . import enclosure
from .geom import Plane, extrude_cs

#: mm from the centre of one square of the grid to the centre of the next
GRID = 42.0
#: mm of one height unit of a bin
HEIGHT_UNIT = 7.0
#: mm that a bin is smaller than its squares, so neighbours never touch
BIN_GAP = 0.5
#: mm of the vertical corner radius of a bin and of the opening of a square
OUTER_RADIUS = 3.75
#: mm of the vertical corner radius of a baseplate
PLATE_RADIUS = 4.0
#: mm across a bin or a plate per square, less the gap to a neighbour
CELL = GRID - BIN_GAP
#: mm of the 45 degree chamfer at the bottom of the foot of a bin
FOOT_CHAMFER = 0.8
#: mm of the vertical part of the foot of a bin, and of the hole of a plate
FOOT_HEIGHT = 1.8
#: mm of the 45 degree chamfer that opens the foot, and the hole of a plate, to the full width
NECK_CHAMFER = 2.15
#: mm of the 45 degree chamfer at the bottom of the hole of a plate: 0.1 less than the foot has,
#: which is the room the foot has to go in and come out
PLATE_CHAMFER = 0.7
#: mm across the vertical part of a foot and of a hole
FOOT_MID = CELL - 2.0 * NECK_CHAMFER
#: mm high the foot of a bin is, and the height it is counted in
BASE_HEIGHT = FOOT_CHAMFER + FOOT_HEIGHT + NECK_CHAMFER
#: mm high a baseplate is, open at the bottom
PLATE_HEIGHT = PLATE_CHAMFER + FOOT_HEIGHT + NECK_CHAMFER
#: mm that the stacking lip stands above the nominal height of a bin
LIP_HEIGHT = 4.4
#: mm across a magnet pocket in the bottom of a foot
MAGNET_DIA = 6.5
#: mm deep a magnet pocket is
MAGNET_DEPTH = 2.4
#: mm from the centre of a foot to the centre of its pockets, so they are 26 mm apart
MAGNET_OFFSET = 13.0
#: the straight pieces a magnet pocket is approximated with
SEGMENTS = 64
#: mm that the dividers stay under the top of the wall, so you can reach over them
DIVIDER_DROP = 1.0
#: mm past a cut, so it leaves no skin behind
MARGIN = 0.5
#: mm of slab at either end of a chamfer, of which the hull of a chamfer is made
SLIVER = 0.2
#: the fewest squares in a row and the most that is still one bin
COLS_MIN, COLS_MAX = 1, 10
#: the shortest bin that is worth a foot and the tallest that still stands on a plate
UNITS_MIN, UNITS_MAX = 2, 20
#: the thinnest wall that stands up straight and the thickest that is still a bin, not a block
WALL_MIN, WALL_MAX = 0.8, 3.0
#: the thinnest floor that holds the smallest thing in and the thickest that is worth the plastic
FLOOR_MIN, FLOOR_MAX = 0.6, 3.0
#: one open compartment, and the most compartments that leave room to pick something out
DIV_MIN, DIV_MAX = 1, 8
#: mm of the finger scoop between the floor and the front wall, and of the label tab under the top
#: of the back wall; both become less in a bin that is too low or too narrow to hold them
SCOOP_RADIUS, LABEL_DEPTH = 12.0, 12.0
#: the vertical plane across the bin, whose local x, y, z are the world y, z, x: the scoop and the
#: label tab are drawn in that plane and drawn out along the length of the bin
FRONT = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so a bin is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _count(value, lo, hi):
    """A number of squares, units or dividers, kept within `lo` and `hi`."""
    return max(lo, min(hi, int(value)))


def _box(length, width, radius, height, z=0.0) -> mf.Manifold:
    """A rounded box `length` by `width` with corners of `radius`, `height` high, standing at `z`
    and centred on the origin in plan."""
    return mf.Manifold.extrude(enclosure.rounded_rect(length, width, radius),
                               height).translate((0.0, 0.0, z))


def _cell_levels(bottom, length=CELL, width=CELL) -> tuple:
    """The three levels of the profile of a square from the bottom up, as (length, width, corner
    radius, mm of vertical part), where the chamfer at the bottom is `bottom` millimetres.

    The foot of a bin and the hole of a plate are the same profile, and only that chamfer
    differs: the corner radius follows the side, so a chamfer is 45 degrees also around the
    corner. The top level has no vertical part of its own; it is where the chamfer ends. The
    top is `length` by `width`: a square, or the outline of a whole bin for its stacking lip.
    """
    mid_r = OUTER_RADIUS - NECK_CHAMFER
    neck = 2.0 * NECK_CHAMFER
    return ((length - neck - 2.0 * bottom, width - neck - 2.0 * bottom, mid_r - bottom, 0.0),
            (length - neck, width - neck, mid_r, FOOT_HEIGHT),
            (length, width, OUTER_RADIUS, 0.0))


def _stepped(levels, z=0.0) -> mf.Manifold:
    """The profile `levels` describes, standing at `z` and centred on the origin in plan.

    Every level is followed by its vertical part and, when a wider level comes next, by a
    chamfer of 45 degrees. A chamfer is the hull of a thin slab of the one level and a thin slab
    of the next: the hull of two rounded squares keeps the corner on the side, so the side and
    the corner grow together and the chamfer is 45 degrees everywhere.
    """
    parts = []
    for i, (length, width, radius, height) in enumerate(levels):
        if height > 0.0:
            parts.append(_box(length, width, radius, height, z))
            z += height
        if i + 1 < len(levels):
            ul, uw, ur, _ = levels[i + 1]
            d = 0.5 * (ul - length)
            parts.append(mf.Manifold.batch_hull([_box(length, width, radius, SLIVER, z),
                                                 _box(ul, uw, ur, SLIVER, z + d)]))
            z += d
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _cell_cutter(bottom, z=0.0, below=0.0, above=0.0, length=CELL, width=CELL) -> mf.Manifold:
    """The profile of a square with the bottom chamfer of `bottom` millimetres, standing at `z`,
    as a cutter: carried `below` millimetres under its bottom and `above` millimetres past its
    top with the cross-sections it has there, so a cut comes right through instead of leaving a
    skin of the plate behind. `length` and `width` are the top of the profile.
    """
    levels = _cell_levels(bottom, length, width)
    bl, bw, radius, _ = levels[0]
    high = bottom + FOOT_HEIGHT + NECK_CHAMFER
    parts = [_stepped(levels, z)]
    if below > 0.0:
        parts.append(_box(bl, bw, radius, below, z - below))
    if above > 0.0:
        parts.append(_box(length + 2.0 * MARGIN, width + 2.0 * MARGIN, OUTER_RADIUS + MARGIN,
                          above, z + high))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _cells(cols, rows):
    """The centres of the `cols` by `rows` squares of a grid, as (x, y), about the origin."""
    return [((i - 0.5 * (cols - 1)) * GRID, (j - 0.5 * (rows - 1)) * GRID)
            for i in range(cols) for j in range(rows)]


def _scoop(inner_l, inner_w, bottom, H) -> mf.Manifold:
    """The finger scoop along the inside of the front wall: a quarter-round fillet between the
    floor and the wall, over the whole inside length, so small parts slide out with a finger.

    It is the block of `r` by `r` in the corner less a cylinder of `r` along the length, centred
    where that block touches the wall and the floor, so the rounding runs into both tangentially
    and there is no step left for a part to catch on.
    """
    r = min(SCOOP_RADIUS, 0.5 * (H - bottom))
    y0 = -0.5 * inner_w
    block = mf.Manifold.cube([inner_l, r, r]).translate((-0.5 * inner_l, y0, bottom))
    axis = extrude_cs(mf.CrossSection.circle(r, SEGMENTS), FRONT,
                      -0.5 * inner_l, 0.5 * inner_l).translate((0.0, y0 + r, bottom + r))
    return block - axis


def _label_tab(inner_l, inner_w, H) -> mf.Manifold:
    """The label tab along the inside of the back wall: a shelf of `d` millimetres under the top
    of the wall, over the whole inside length, against which a printed label stands.

    The triangle in the y-z plane has its top on the top of the wall and its underside at
    45 degrees down to the wall, so it prints without support.
    """
    d = min(LABEL_DEPTH, 0.5 * inner_w)
    y1 = 0.5 * inner_w
    loop = [[y1, H - d], [y1, H], [y1 - d, H]]
    return extrude_cs(mf.CrossSection([loop]), FRONT, -0.5 * inner_l, 0.5 * inner_l)


def gridfinity_bin(cols=1, rows=1, units=3, wall=1.2, floor=1.0, divisions=1, lip=True,
                   magnets=False, scoop=False, label=False) -> mf.Manifold:
    """A Gridfinity bin of `cols` by `rows` squares and `units` height units, as one closed volume.

    The bin is `GRID * cols − BIN_GAP` by `GRID * rows − BIN_GAP` millimetres and
    `units * HEIGHT_UNIT` millimetres high, in which the foot of `BASE_HEIGHT` millimetres is
    counted, and it stands on that foot under every one of its squares. Above the foot the body
    is hollowed from `BASE_HEIGHT + floor` up, with walls of `wall` and inside corners of
    `OUTER_RADIUS − wall`, so the wall stays the same thickness all the way round. `divisions`
    compartments along the length are made by straight dividers of `wall` millimetres that stop
    `DIVIDER_DROP` millimetre below the top edge. With `lip` the top of the wall carries the
    `LIP_HEIGHT` millimetres of stacking lip, so a bin of the same grid stands on it and the bin
    is that much taller than `units * HEIGHT_UNIT`. With `magnets` the bottom of every foot gets
    four pockets of `MAGNET_DIA` by `MAGNET_DEPTH`, for the magnets that hold a bin to a steel
    shelf. With `scoop` a quarter-round fillet of `SCOOP_RADIUS` millimetres stands in the corner
    between the floor and the inside of the front wall, or half the depth of the inside when the
    bin is too low for that, so you slide small parts out with a finger. With `label` a tab of
    `LABEL_DEPTH` millimetres runs along the inside of the back wall under the top edge, its
    underside at 45 degrees, so a printed label stands against the wall and is read from the front.
    Both are clipped to the inside, so neither sticks out of the bin or changes its size.

    The result is centred on the origin in plan and stands on `z = 0`. The sizes are kept within
    their own bounds, so a bin that cannot be printed becomes the nearest one that can.
    """
    nc, nr = _count(cols, COLS_MIN, COLS_MAX), _count(rows, COLS_MIN, COLS_MAX)
    u = _count(units, UNITS_MIN, UNITS_MAX)
    t = _clamp(wall, WALL_MIN, WALL_MAX)
    f = _clamp(floor, FLOOR_MIN, FLOOR_MAX)
    nd = _count(divisions, DIV_MIN, DIV_MAX)
    L, W = nc * GRID - BIN_GAP, nr * GRID - BIN_GAP
    H = u * HEIGHT_UNIT
    top = H + (LIP_HEIGHT if lip else 0.0)          # the lip stands on the top of the wall
    bottom = BASE_HEIGHT + f                        # where the inside starts: on the foot

    body = _box(L, W, OUTER_RADIUS, top - BASE_HEIGHT, BASE_HEIGHT)
    feet = [_stepped(_cell_levels(FOOT_CHAMFER)).translate((x, y, 0.0)) for x, y in _cells(nc, nr)]
    body = body + mf.Manifold.batch_boolean(feet, mf.OpType.Add)

    # the inside, up to the top of the wall: with a lip the wall top stays whole, because the lip
    # leans out over the opening and the foot of the bin above comes down into it
    inner_l, inner_w = L - 2.0 * t, W - 2.0 * t
    room = _box(inner_l, inner_w, max(OUTER_RADIUS - t, 0.0),
                H - bottom + (0.0 if lip else MARGIN), bottom)
    cut = [room]
    if lip:
        # the lip: the top of a plate hole again, standing on the wall, around the whole bin,
        # so nothing of it crosses the opening between two squares
        drop = PLATE_HEIGHT - LIP_HEIGHT
        cut.append(_cell_cutter(PLATE_CHAMFER, H - drop, above=MARGIN, length=L, width=W))
    body = body - mf.Manifold.batch_boolean(cut, mf.OpType.Add)

    # what stands in the bin again: the dividers, which stop under the top of the wall, and the
    # scoop and the label tab, which are added with them and clipped to the inside together
    add = []
    for i in range(1, nd):
        x = -0.5 * inner_l + i * inner_l / nd
        add.append(mf.Manifold.cube([t, inner_w, H - DIVIDER_DROP - bottom])
                   .translate((x - 0.5 * t, -0.5 * inner_w, bottom)))
    if scoop:
        add.append(_scoop(inner_l, inner_w, bottom, H))
    if label:
        add.append(_label_tab(inner_l, inner_w, H))
    if add:
        # nothing may stand outside the bin, so what goes back in is clipped to the inside, grown
        # into the wall by MARGIN so it fuses with the wall instead of only resting against it
        extra = mf.Manifold.batch_boolean(add, mf.OpType.Add)
        clip = _box(inner_l + 2.0 * MARGIN, inner_w + 2.0 * MARGIN,
                    max(OUTER_RADIUS - t, 0.0) + MARGIN, H - bottom, bottom)
        body = body + mf.Manifold.batch_boolean([extra, clip], mf.OpType.Intersect)
    if magnets:
        # the pockets for the magnets, in the bottom of every foot, flush with the bed
        r, d = 0.5 * MAGNET_DIA, MAGNET_DEPTH
        pockets = []
        for cx, cy in _cells(nc, nr):
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                pockets.append(mf.Manifold.cylinder(d + MARGIN, r,
                                                    circular_segments=SEGMENTS).translate(
                    (cx + sx * MAGNET_OFFSET, cy + sy * MAGNET_OFFSET, -MARGIN)))
        body = body - mf.Manifold.batch_boolean(pockets, mf.OpType.Add)
    # the bin stands on the bed as it prints: centred in plan, with the foot on z = 0
    bb = np.asarray(body.bounding_box(), dtype=float)
    return body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))


def gridfinity_baseplate(cols=3, rows=2) -> mf.Manifold:
    """A Gridfinity baseplate of `cols` by `rows` squares, as one closed volume.

    The plate is `GRID * cols` by `GRID * rows` millimetres and `PLATE_HEIGHT` millimetres high,
    with corners of `PLATE_RADIUS`, and every square is a hole of the same profile as the foot of
    a bin, upside down: `NECK_CHAMFER` at 45 degrees at the top, `FOOT_HEIGHT` vertical and
    `PLATE_CHAMFER` at 45 degrees at the bottom, where the plate is open. Neighbouring holes
    share the thin rim between them, which is what makes a plate of any size one printed piece.
    A foot drops into a hole from the top and the chamfers pull it in, so a bin stands on the
    plate without glue and is lifted off again without breaking anything.

    The result is centred on the origin in plan and stands on `z = 0`. The number of squares is
    kept within its bounds.
    """
    nc, nr = _count(cols, COLS_MIN, COLS_MAX), _count(rows, COLS_MIN, COLS_MAX)
    plate = _box(nc * GRID, nr * GRID, PLATE_RADIUS, PLATE_HEIGHT)
    holes = [_cell_cutter(PLATE_CHAMFER, 0.0, below=MARGIN, above=MARGIN).translate((x, y, 0.0))
             for x, y in _cells(nc, nr)]
    return plate - mf.Manifold.batch_boolean(holes, mf.OpType.Add)
