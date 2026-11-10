"""Storage bins: the computational core for a printable open-top box with compartments.

No Qt in this file. It gives the bin as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **storage bin** is the box you empty your pocket of screws into: walls you pick it up by, a
floor that keeps the smallest thing from walking out and an open top, so you see what is in it.
The size you enter is the outside of the bin, because that is what has to fit on the shelf or in
the drawer, and the wall and the floor are part of it. The vertical corners are rounded with
`radius`, which is nicer to hold than a sharp edge, and the inside follows with
`radius − wall`, so the wall stays the same thickness all the way round.

The **compartments** keep the small things apart: `cols` columns along the length and `rows` rows
along the width, made by straight dividers of `wall` millimetres that stand on the floor and reach
to `DIVIDER_DROP` millimetre below the top edge. They are lower than the walls on purpose, so you
can pick something out over the divider without lifting the bin, while nothing climbs on its own
from one compartment into the next.

The **label ledge** is where the name goes: along the inside of the front wall a ledge of
`LABEL_DEPTH` millimetres runs the whole length at the top, and its underside is one plane that
stands at 45 degrees. A label lies against the wall and on the ledge, where you read it from
above, and the ledge prints without support, because a face at 45 degrees is what a printer
carries on its own.

The top of the ledge has a shallow **pocket** for a printed label (`bin_label`): `LABEL_POCKET`
millimetres deep, from the front wall to `LABEL_LIP` millimetres before the edge of the ledge, so
the lip that stays holds the label in. The label is a thin plate of the size of the pocket, less
`LABEL_PLAY` all round, with its text standing on top, and the pocket is as deep as the plate and
the letters together, so even the letters stay under the top edge and a bin stands on it.

A **stacking** bin has a foot: the lowest `STACK_FOOT` millimetres of the outside step in to the
opening of a bin of the same size, less `STACK_PLAY` all round, so it drops into the bin under it
and cannot slide off. The foot stays behind the label ledge of the bin under it and its dividers
stop that much lower, so nothing is in its way; the floor inside starts on top of the foot.

The **wall hook** hangs the bin on the wall rack of `volura/bin_rack.py`: along the outside of the
back wall a cleat of `CLEAT_THICK` millimetres runs at the top and reaches `CLEAT_OUT − CLEAT_PLAY`
millimetres out, down at 45 degrees, so it drops behind a rail of the rack, whose face leans the
same way, and the weight of the bin pulls it in. A rib of the same depth along the bottom of the
back wall keeps the bin upright against the rack. The hook prints without support for the same
reason as the ledge, and the rib stands on the bed.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np

from . import enclosure
from .geom import Plane, extrude_cs
from .i18n import tr

#: the smallest bin that is worth printing and the largest that still fits on a bed
LENGTH_MIN, LENGTH_MAX = 20.0, 300.0
WIDTH_MIN, WIDTH_MAX = 20.0, 300.0
HEIGHT_MIN, HEIGHT_MAX = 10.0, 200.0
#: the thinnest wall that stands up straight and the thickest that is still a bin, not a block
WALL_MIN, WALL_MAX = 0.8, 5.0
#: the thinnest floor that holds and the thickest that is still worth the plastic
FLOOR_MIN, FLOOR_MAX = 0.6, 5.0
#: the corner radius: from a bin with sharp corners to one that is nearly a stadium
RADIUS_MIN, RADIUS_MAX = 0.0, 20.0
#: the fewest compartment in a row and the most that still leaves room to pick something out
COLS_MIN, COLS_MAX = 1, 10
#: mm of ledge along the front wall, on which a printed label lies flat and stays readable
LABEL_DEPTH = 12.0
#: mm of plate under the letters of a label
LABEL_THICK = 0.8
#: mm of ledge that stays whole along the inside, the lip that holds the label in its pocket
LABEL_LIP = 2.5
#: mm of room all round the label in its pocket
LABEL_PLAY = 0.2
#: mm that the letters stand on the label
LABEL_RELIEF = 0.6
#: mm deep the pocket in the top of the ledge is: the label with its letters, flush with the top
LABEL_POCKET = LABEL_THICK + LABEL_RELIEF
#: mm high the foot of a stacking bin is, and the room it has in the opening of the bin under it
STACK_FOOT, STACK_PLAY = 3.0, 0.3
#: mm that the letters stay from the edge of the label
LABEL_EDGE = 1.0
#: mm across the narrowest pocket that is still worth a label
LABEL_NARROWEST = 4.0
#: mm that the dividers stay under the top edge, so you can reach over them into the bin
DIVIDER_DROP = 1.0
#: mm past the top of the walls, so a cut leaves no skin behind
MARGIN = 0.5
#: the vertical plane across the bin, whose local x, y, z are the world y, z, x: the label ledge
#: is drawn in that plane and drawn out along the length of the bin
FRONT = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
#: mm that a rail of the wall rack stands out from its plate, and the hook of a bin reaches back
CLEAT_OUT = 6.0
#: mm of plastic in a rail and in a hook, measured along the wall
CLEAT_THICK = 4.0
#: mm that the tip of a hook stays off the plate, so the hook rests on the rail and not on the wall
CLEAT_PLAY = 0.4


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so the bin is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _count(n):
    """The number of compartments in a row, kept within `COLS_MIN` to `COLS_MAX`."""
    return max(COLS_MIN, min(COLS_MAX, int(n)))


def _ledge(y_wall, into, x_lo, x_hi, top, depth) -> mf.Manifold:
    """The label ledge along the inside of the front wall at `y_wall`, drawn from `x_lo` to `x_hi`.

    The underside runs from `top − depth` at the wall up to `top` at `depth` millimetres into the
    bin, so it stands at 45 degrees, and it reaches `into` millimetres into the wall, so the ledge
    is one piece with it instead of only touching it.
    """
    loop = [[y_wall - into, top - depth - into], [y_wall + depth, top], [y_wall - into, top]]
    return extrude_cs(mf.CrossSection([loop]), FRONT, x_lo, x_hi)


def _hanger(y_back, x_lo, x_hi, top, bottom=0.0) -> mf.Manifold:
    """The hook and the rib on the outside of the back wall at `y_back`, from `x_lo` to `x_hi`.

    The hook hangs from the top edge, `CLEAT_THICK` high, and reaches `CLEAT_OUT − CLEAT_PLAY` out
    and as far down, so its underside lies at 45 degrees on the face of a rail of the rack. The rib
    along the bottom reaches out as far, so the bin hangs upright instead of tilting to the wall;
    it starts at `bottom`, which is above the foot of a stacking bin.
    """
    out, t, inset = CLEAT_OUT - CLEAT_PLAY, CLEAT_THICK, MARGIN
    hook = [[y_back - inset, top - t + inset], [y_back + out, top - t - out],
            [y_back + out, top - out], [y_back - inset, top]]
    rib = [[y_back - inset, bottom], [y_back + out, bottom], [y_back + out, bottom + t],
           [y_back - inset, bottom + t]]
    return extrude_cs(mf.CrossSection([hook, rib]), FRONT, x_lo, x_hi)


def _ledge_depth(inner_w, H, f) -> float:
    """How deep the ledge is: never deeper than half the inside or the height of the bin."""
    return min(LABEL_DEPTH, 0.5 * inner_w, H - f)


def _pocket(inner_l, inner_w, depth):
    """The pocket in the ledge as (length, width, y of its front side), or None when the ledge
    leaves too little room for a label."""
    wide = depth - LABEL_LIP
    if wide < LABEL_NARROWEST or inner_l < LABEL_NARROWEST:
        return None
    return inner_l, wide, -0.5 * inner_w


def label_size(length=80, width=60, height=40, wall=1.6, floor=1.2):
    """(length, width) of the label that fits the pocket of this bin, or None when it has none."""
    L = _clamp(length, LENGTH_MIN, LENGTH_MAX)
    W = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    H = _clamp(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _clamp(wall, WALL_MIN, WALL_MAX)
    f = _clamp(floor, FLOOR_MIN, FLOOR_MAX)
    inner_l, inner_w = L - 2.0 * t, W - 2.0 * t
    pocket = _pocket(inner_l, inner_w, _ledge_depth(inner_w, H, f))
    if pocket is None:
        return None
    return pocket[0] - 2.0 * LABEL_PLAY, pocket[1] - 2.0 * LABEL_PLAY


def bin_label(text, length=80, width=60, height=40, wall=1.6, floor=1.2, family=None,
              bold=True) -> mf.Manifold:
    """The label for the pocket of a bin of these sizes: a plate with `text` standing on it.

    The plate is `LABEL_THICK` millimetres thick and as large as the pocket less `LABEL_PLAY` all
    round, so with its letters it lies flush in the ledge. The letters stand `LABEL_RELIEF` on it,
    as high as fits and made smaller when a long text would run over the edge; they stay
    `LABEL_EDGE` millimetres from it. The label lies on `z = 0`, centred on the origin, and the
    text reads along the length of the bin. A bin without room for a pocket, or a text without
    letters, is refused with a ValueError.
    """
    from . import text3d           # the letters need the fonts, which only this part does
    size = label_size(length, width, height, wall, floor)
    if size is None:
        raise ValueError(tr("this bin has no room for a label"))
    lx, ly = size
    letters = text3d.text3d(text, family, height=max(text3d.HEIGHT_MIN, 0.6 * ly),
                            depth=LABEL_RELIEF + MARGIN, bold=bold, align="centre")
    lo, hi = np.asarray(letters.bounding_box()[:3]), np.asarray(letters.bounding_box()[3:])
    room = (lx - 2.0 * LABEL_EDGE, ly - 2.0 * LABEL_EDGE)
    k = min(1.0, room[0] / (hi[0] - lo[0]), room[1] / (hi[1] - lo[1]))
    letters = letters.scale((k, k, 1.0))
    lo, hi = np.asarray(letters.bounding_box()[:3]), np.asarray(letters.bounding_box()[3:])
    letters = letters.translate((-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]),
                                 LABEL_THICK - MARGIN - lo[2]))
    plate = mf.Manifold.cube([lx, ly, LABEL_THICK]).translate((-0.5 * lx, -0.5 * ly, 0.0))
    return plate + letters


def storage_bin(length=80, width=60, height=40, wall=1.6, floor=1.2, radius=4,
                cols=1, rows=1, label=True, hanger=False, stack=False) -> mf.Manifold:
    """A printable open-top box with `cols` by `rows` compartments and a ledge for a label.

    The bin is `length` by `width` millimetres and `height` millimetres high, with walls of `wall`
    and a floor of `floor`, and its vertical corners are rounded with `radius` outside and
    `radius − wall` inside. `cols` columns along the length and `rows` rows along the width are
    made by straight dividers of `wall` millimetres, which stand on the floor and stop
    `DIVIDER_DROP` millimetre below the top edge. With `label` the inside of the front wall carries
    a ledge of `LABEL_DEPTH` millimetres at the top, whose underside stands at 45 degrees, so a
    label lies under it and the ledge prints without support. With `hanger` the back wall carries
    the hook and the rib that hang the bin on the wall rack of `volura/bin_rack.py`; they stick
    out behind the bin, so the box round the bin is then deeper than `width`. With `stack` the
    bottom gets the foot that drops into a bin of the same size under it (see the top of this
    file); the bin stays `height` high and its inside starts on top of the foot.

    The result is centred on the origin in plan and stands on `z = 0`. The sizes are kept within
    their own bounds and the radius within half the smallest side, so a bin that cannot be printed
    becomes the nearest one that can.
    """
    L = _clamp(length, LENGTH_MIN, LENGTH_MAX)
    W = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    H = _clamp(height, HEIGHT_MIN, HEIGHT_MAX)
    t = _clamp(wall, WALL_MIN, WALL_MAX)
    f = _clamp(floor, FLOOR_MIN, FLOOR_MAX)
    r = min(_clamp(radius, RADIUS_MIN, RADIUS_MAX), 0.5 * min(L, W))
    nc, nr = _count(cols), _count(rows)
    inner_l, inner_w = L - 2.0 * t, W - 2.0 * t
    inner_r = max(r - t, 0.5)                  # the inside follows the corner, so the wall stays even

    foot = STACK_FOOT if stack else 0.0
    bottom = foot + f                          # where the inside starts: on the foot, if any
    block = mf.Manifold.extrude(enclosure.rounded_rect(L, W, r), H)
    cavity = mf.Manifold.extrude(enclosure.rounded_rect(inner_l, inner_w, inner_r),
                                 H - bottom + MARGIN).translate((0.0, 0.0, bottom))

    # what stands in the bin again: the dividers, and along the front wall the ledge for a label;
    # the dividers of a stacking bin stay under the foot of the bin that stands in it
    add = []
    top = H - DIVIDER_DROP - foot
    f = bottom
    for i in range(1, nc):
        x = -0.5 * inner_l + i * inner_l / nc
        add.append(mf.Manifold.cube([t, inner_w + t, top - f])
                   .translate((x - 0.5 * t, -0.5 * (inner_w + t), f)))
    for j in range(1, nr):
        y = -0.5 * inner_w + j * inner_w / nr
        add.append(mf.Manifold.cube([inner_l + t, t, top - f])
                   .translate((-0.5 * (inner_l + t), y - 0.5 * t, f)))
    depth = _ledge_depth(inner_w, H, f)
    if label and depth > 0.0:
        add.append(_ledge(-0.5 * inner_w, t, -0.5 * (inner_l + t), 0.5 * (inner_l + t), H, depth))
    body = block - cavity
    if add:
        # nothing may stand outside the outline, so what goes back in is clipped to the block
        extra = mf.Manifold.batch_boolean(add, mf.OpType.Add)
        body = body + mf.Manifold.batch_boolean([extra, block], mf.OpType.Intersect)
    pocket = _pocket(inner_l, inner_w, depth) if label else None
    if pocket is not None:
        # the pocket for the label, in the top of the ledge and against the front wall
        lx, ly, y0 = pocket
        body = body - mf.Manifold.cube([lx, ly, LABEL_POCKET + MARGIN]).translate(
            (-0.5 * lx, y0, H - LABEL_POCKET))
    if stack:
        # the foot: the opening of the bin under it, less the play, and behind its label ledge
        lx = inner_l - 2.0 * STACK_PLAY
        y_lo = -0.5 * inner_w + STACK_PLAY + (depth if label and depth > 0.0 else 0.0)
        y_hi = 0.5 * inner_w - STACK_PLAY
        rr = max(0.0, min(inner_r - STACK_PLAY, 0.5 * lx, 0.5 * (y_hi - y_lo)))
        sole = mf.Manifold.extrude(enclosure.rounded_rect(lx, y_hi - y_lo, rr),
                                   foot + 2.0 * MARGIN).translate((0.0, 0.5 * (y_lo + y_hi), -MARGIN))
        slab = mf.Manifold.cube([L + 2.0, W + 2.0, foot + MARGIN]).translate(
            (-0.5 * L - 1.0, -0.5 * W - 1.0, -MARGIN))
        body = body - (slab - sole)
    # the bin stands on the bed as it prints: centred in plan, with the floor on z = 0
    bb = np.asarray(body.bounding_box(), dtype=float)
    body = body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
    if hanger:
        # on the flat part of the back wall, so the hook does not stick out of a rounded corner
        x_flat = 0.5 * L - max(r, t)
        if x_flat > 0.0:
            body = body + _hanger(0.5 * W, -x_flat, x_flat, H, foot)
    return body
