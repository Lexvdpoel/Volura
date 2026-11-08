"""Project boxes: the computational core for an enclosure with a fitting lid.

No Qt in this file. It gives the base and the lid of a box as two closed volumes, so it can
be tested on its own and reused in the model and the export.

A **project box** is a tray with an open top and a lid that slides over it. The user says
how big the **inside** has to be, because that is what must fit: the electronics. The wall
and the floor are added around that, so the box stands `length + 2 × wall` wide,
`width + 2 × wall` deep and `floor + height` high on the bed. The vertical corners are
rounded with `radius`, which is nicer to hold and stronger than a sharp corner; the inside
follows with `radius − wall`, so the wall stays the same thickness all the way round.

The lid is a plate of the same outline with a **lip** on its underside: a ring as thick as
the wall and `lip` millimetres high, that goes into the opening. Its outside is the opening
shrunk by the `clearance` on every side, the little room a printed part needs to slide into
another printed part; a lip without clearance is printed tight and the lid stays stuck.

The lid comes back **flipped for printing**: its flat top lies on the bed and the lip stands
up, so nothing hangs in the air and both parts print without support.
"""
from __future__ import annotations

import manifold3d as mf

from .i18n import tr

#: the thinnest wall, floor and lid that still print as more than one line of plastic
WALL_MIN, FLOOR_MIN, LID_MIN = 0.8, 0.4, 0.4
#: the largest clearance between the lip and the wall, above which the lid falls in
CLEARANCE_MAX = 1.0


def rounded_rect(length, width, radius) -> mf.CrossSection:
    """A rectangle `length` × `width` with corners of `radius`, centred on the origin.

    The corners are made by inflating a smaller rectangle: `offset` with `JoinType.Round`
    replaces every corner by an arc and moves every side out by `radius`, so the outline is
    exactly `length` × `width` along the flat sides. A radius over half the shortest side is
    taken as half of it, which gives a stadium or a circle instead of a contour that crosses
    itself.
    """
    L, W = float(length), float(width)
    r = max(0.0, min(float(radius), 0.5 * min(L, W)))
    if r == 0.0:
        return mf.CrossSection.square((max(L, 0.0), max(W, 0.0)), True)
    if L <= 2.0 * r and W <= 2.0 * r:            # nothing left of the rectangle: a circle
        return mf.CrossSection.circle(r)
    return mf.CrossSection.square((max(L - 2.0 * r, 0.0), max(W - 2.0 * r, 0.0)), True) \
        .offset(r, mf.JoinType.Round, 2.0)


def project_box(length=80, width=50, height=30, wall=2.0, floor=2.0, radius=3.0,
                lid=2.0, lip=3.0, clearance=0.2) -> tuple[mf.Manifold, mf.Manifold]:
    """A printed enclosure with a fitting lid: the base and the lid, side by side on the bed.

    `length`, `width` and `height` are the **inside** of the box, what the electronics need;
    `wall` is the thickness of the sides and of the lip, `floor` the thickness of the bottom
    and `lid` the thickness of the plate. `radius` rounds the vertical corners outside,
    `radius − wall` inside. The lid is a plate of the same outline as the base with a lip of
    `lip` millimetres on its underside, which slides into the opening with `clearance`
    millimetres of room on every side.

    Both parts stand on `z = 0` and are centred on the origin in plan. The base is the tray
    with its open top up; the lid is flipped for printing, so its flat top lies on the bed
    and the lip points up.
    """
    L, W, H = float(length), float(width), float(height)
    if L <= 0.0 or W <= 0.0 or H <= 0.0:
        raise ValueError(tr("The inside dimensions must be greater than zero."))
    t = float(wall)
    if t < WALL_MIN:
        raise ValueError(tr("The wall must be at least {wall} mm.").format(wall=WALL_MIN))
    f = float(floor)
    if f < FLOOR_MIN:
        raise ValueError(tr("The floor must be at least {floor} mm.").format(floor=FLOOR_MIN))
    plate_t = float(lid)
    if plate_t < LID_MIN:
        raise ValueError(tr("The lid must be at least {lid} mm.").format(lid=LID_MIN))
    c = float(clearance)
    if c < 0.0 or c > CLEARANCE_MAX:
        raise ValueError(tr("The clearance must be between 0 and 1 mm."))
    lip_h = float(lip)
    if lip_h < 0.0:
        raise ValueError(tr("The lip may not be negative."))
    if lip_h >= H:
        raise ValueError(tr("The lip must be lower than the inside height."))
    r = float(radius)
    if r < 0.0:
        raise ValueError(tr("The corner radius may not be negative."))

    # the tray: the outer block with the inside cut out of it, past the top so it stays open
    outline = rounded_rect(L + 2.0 * t, W + 2.0 * t, r)
    shell = mf.Manifold.extrude(outline, f + H)
    cavity = mf.Manifold.extrude(rounded_rect(L, W, max(0.0, r - t)), H + t) \
        .translate((0.0, 0.0, f))
    base = shell - cavity

    # the lid: the plate on the bed, the lip above it, so both print without support
    top = mf.Manifold.extrude(outline, plate_t)
    if lip_h == 0.0:
        return base, top
    lip_out = rounded_rect(L - 2.0 * c, W - 2.0 * c, max(0.0, r - t - c))
    lip_in = rounded_rect(L - 2.0 * (c + t), W - 2.0 * (c + t), max(0.0, r - 2.0 * t - c))
    ring = mf.Manifold.extrude(lip_out, lip_h) \
        - mf.Manifold.extrude(lip_in, lip_h + 2.0).translate((0.0, 0.0, -1.0))
    if ring.is_empty():                   # in a very small box nothing is left of the lip
        return base, top
    return base, top + ring.translate((0.0, 0.0, plate_t))
