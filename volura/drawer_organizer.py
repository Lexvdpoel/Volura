"""Drawer organizers: the computational core of a tray that divides the inside of a drawer.

No Qt in this file. It gives the organizer as one manifold, so it can be tested on its own and
reused in the model and the export.

A **drawer organizer** is the tray that drops into a drawer and keeps its contents apart: a floor
it stands on, walls around the edge and dividers between the compartments, and an open top so you
see and reach what is in it. It prints **upright**, so every wall is pulled straight up from the
floor and nothing hangs over but the finger notches, which are cut from the top.

The size you enter is the **inside of the drawer** (`length` by `width`), because that is what you
measure with a tape; the tray is made `clearance` millimetres smaller on every side, so it drops in
without forcing. It is `height` millimetres high, its walls and dividers are `wall` millimetres
thick and its floor is `floor` millimetres thick.

The **compartments** are `cols` columns along the length and `rows` rows along the width, all the
same size, made by walls of `wall` millimetres between them and around them. The inside corners of
every compartment are rounded with `radius`, so you scoop something out with a finger and the tray
cleans easily; the outside corners of the tray are rounded with `radius + wall`, so the wall stays
the same thickness all the way round, through the corners too.

A **finger notch** is a half-round cut in the top edge of every inner wall, at its middle, so you
get a finger down beside an item that stands between two dividers and lift it out. It is
`FINGER_W` millimetres wide and `FINGER_D` deep, never wider than a compartment nor deeper than the
wall stands above the floor. With a single compartment there is no inner wall, so there is no notch.

Every number is clamped to what prints: the walls are made thinner when there would not otherwise
be a compartment left to reach into, and the radius never exceeds a third of the smallest
compartment, so a tray asked for too full becomes the nearest one that still works.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from . import enclosure
from .geom import Plane, extrude_cs
from .i18n import tr

#: mm the inside of the smallest drawer worth a tray and the largest that still prints
LENGTH_MIN, LENGTH_MAX = 40.0, 300.0
WIDTH_MIN, WIDTH_MAX = 40.0, 300.0
#: mm the shortest and the tallest tray that is still worth standing something up in
HEIGHT_MIN, HEIGHT_MAX = 15.0, 120.0
#: mm the tray is made smaller than the drawer on every side, so it drops in
CLEAR_MIN, CLEAR_MAX = 0.0, 3.0
#: the fewest and the most compartments along the length and along the width
COLS_MIN, COLS_MAX = 1, 8
ROWS_MIN, ROWS_MAX = 1, 8
#: the thinnest wall that stands up straight and the thickest that still leaves room to fill
WALL_MIN, WALL_MAX = 1.2, 4.0
#: the thinnest floor that holds and the thickest that is still worth the plastic
FLOOR_MIN, FLOOR_MAX = 0.8, 3.0
#: the corner radius: from compartments with sharp corners to ones that are nearly round
RADIUS_MIN, RADIUS_MAX = 0.0, 15.0
#: the radius is never more than this many parts of the smallest compartment side
RADIUS_SHARE = 3.0
#: mm wide and deep a finger notch is, before it is clamped to the compartment and the height
FINGER_W, FINGER_D = 20.0, 15.0
#: mm a compartment keeps at least, so the walls are made thinner before a compartment vanishes
MIN_COMPARTMENT = 1.0
#: the straight pieces a round corner or notch is approximated with
SEGMENTS = 64
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
MARGIN = 0.5
#: the vertical planes across the tray whose local frame draws a finger notch and pulls it through
#: the thickness of a wall: one for a wall between two columns (thin in x), one for a wall between
#: two rows (thin in y); both take the notch width as their u and the height as their v
THROUGH_X = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
THROUGH_Y = Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))


def _number(value, lo, hi):
    """A number of the tray, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a drawer organizer must be numbers"))
    return max(lo, min(hi, v))


def _count(value, lo, hi):
    """How many compartments in a row: a whole number between `lo` and `hi`."""
    return int(_number(value, lo, hi))


def _sizes(length, width, height, clearance, cols, rows, wall, floor, radius, finger) -> dict:
    """The clamped numbers and the places of the features, in the frame the tray prints in.

    In that frame the tray stands on its floor on `z = 0` and is centred in x and y. `length` and
    `width` are the inside of the drawer and `outer_length`, `outer_width` the tray that is made
    from them, `clearance` smaller on every side. `cols` by `rows` compartments of
    `compartment_length` by `compartment_width` fill the inside, `wall` millimetres apart and from
    the edge; `radius` rounds their corners and `outer_radius` those of the tray, one wall thicker.
    `finger_w` and `finger_d` are the notch as it is actually cut, clamped to a compartment and to
    the wall above the floor.
    """
    L = _number(length, LENGTH_MIN, LENGTH_MAX)
    W = _number(width, WIDTH_MIN, WIDTH_MAX)
    H = _number(height, HEIGHT_MIN, HEIGHT_MAX)
    c = _number(clearance, CLEAR_MIN, CLEAR_MAX)
    nc, nr = _count(cols, COLS_MIN, COLS_MAX), _count(rows, ROWS_MIN, ROWS_MAX)
    t = _number(wall, WALL_MIN, WALL_MAX)
    f = _number(floor, FLOOR_MIN, FLOOR_MAX)
    r = _number(radius, RADIUS_MIN, RADIUS_MAX)
    outer_l, outer_w = L - 2.0 * c, W - 2.0 * c
    # the walls are made thinner when cols + 1 of them would leave a compartment too small to reach
    t = max(WALL_MIN, min(t, (outer_l - nc * MIN_COMPARTMENT) / (nc + 1),
                             (outer_w - nr * MIN_COMPARTMENT) / (nr + 1)))
    comp_l = (outer_l - (nc + 1) * t) / nc
    comp_w = (outer_w - (nr + 1) * t) / nr
    comp = min(comp_l, comp_w)
    r = min(r, comp / RADIUS_SHARE)          # a corner you scoop out, never a whole round
    # the notch is cut in the top of a wall: never wider than a compartment, never to the floor
    return {"length": L, "width": W, "height": H, "clearance": c, "cols": nc, "rows": nr,
            "wall": t, "floor": f, "radius": r, "finger": bool(finger),
            "outer_length": outer_l, "outer_width": outer_w, "outer_radius": r + t,
            "compartment_length": comp_l, "compartment_width": comp_w,
            "finger_w": min(FINGER_W, comp), "finger_d": min(FINGER_D, H - f - MARGIN)}


def sizes(length=200.0, width=150.0, height=50.0, clearance=1.0, cols=3, rows=2,
          wall=1.6, floor=1.2, radius=4.0, finger=True) -> dict:
    """The numbers a drawer organizer is made of, so a dialog can show them.

    `length` and `width` are the inside of the drawer, `height` how high the tray stands and
    `clearance` how much smaller it is made on every side so it drops in. `cols` by `rows` equal
    compartments fill it, `wall` millimetres apart and from the edge on a floor of `floor`, with
    the corners rounded `radius` and a `finger` notch in the top of every inner wall.
    `outer_length` and `outer_width` are the tray that is actually made and `compartment_length`,
    `compartment_width` the size of one compartment. Every number is clamped to what prints;
    anything but a number is a ValueError.
    """
    g = _sizes(length, width, height, clearance, cols, rows, wall, floor, radius, finger)
    return {key: g[key] for key in ("length", "width", "height", "clearance", "cols", "rows",
                                    "wall", "floor", "radius", "finger", "outer_length",
                                    "outer_width", "compartment_length", "compartment_width")}


def _pockets(g) -> list:
    """The compartments, cut out of the block from the top of the floor up through the top edge.

    They are laid out on a grid of `cols` by `rows` with a wall between every two, so the wall
    between two of them is `wall` millimetres where their flat sides face each other and, at the
    corners, the round of `radius` sits concentric in the round of `radius + wall` of the tray.
    """
    inner_l = g["outer_length"] - 2.0 * g["wall"]
    inner_w = g["outer_width"] - 2.0 * g["wall"]
    cl, cw = g["compartment_length"], g["compartment_width"]
    reach = g["height"] - g["floor"] + MARGIN
    cuts = []
    for i in range(g["cols"]):
        cx = -0.5 * inner_l + 0.5 * cl + i * (cl + g["wall"])
        for j in range(g["rows"]):
            cy = -0.5 * inner_w + 0.5 * cw + j * (cw + g["wall"])
            cs = enclosure.rounded_rect(cl, cw, g["radius"])
            cuts.append(mf.Manifold.extrude(cs, reach).translate((cx, cy, g["floor"])))
    return cuts


def _finger(fw, fd, height, reach, along, at) -> mf.Manifold:
    """The notch that cuts the wall standing at `at`: `reach` through its thickness and `fw` wide,
    with straight sides and a half-round bottom `fd` deep, open at the top edge.

    `along` is the direction the wall runs: "y" for a wall between two columns, which is thin in x,
    and "x" for a wall between two rows, which is thin in y. The notch sits at the middle of the
    wall, so it is centred on the tray across that direction. It is one closed profile pulled
    through the wall, so its sides and its round meet in one edge instead of two touching faces.
    """
    r = 0.5 * fw
    zc = height - fd + r                       # the centre of the half-round bottom
    top = height + MARGIN
    loop = [(-r, top), (-r, zc)]
    for k in range(1, SEGMENTS):               # the bottom, from one side over the low point to the other
        a = math.pi * (1.0 + k / SEGMENTS)
        loop.append((r * math.cos(a), zc + r * math.sin(a)))
    loop += [(r, zc), (r, top)]
    x0, x1 = at - 0.5 * reach, at + 0.5 * reach
    return extrude_cs(mf.CrossSection([loop]), THROUGH_Y if along == "x" else THROUGH_X, x0, x1)


def _notches(g) -> list:
    """A half-round finger notch in the top of every inner wall, at its middle; none without them.

    The walls between the columns run along the width and the walls between the rows along the
    length; a single compartment has neither, so it gets no notch.
    """
    if not g["finger"]:
        return []
    fw, fd = g["finger_w"], g["finger_d"]
    if fw <= 0.0 or fd <= 0.0:
        return []
    inner_l = g["outer_length"] - 2.0 * g["wall"]
    inner_w = g["outer_width"] - 2.0 * g["wall"]
    cl, cw, t = g["compartment_length"], g["compartment_width"], g["wall"]
    reach = t + 2.0 * MARGIN
    cuts = []
    for i in range(1, g["cols"]):        # the wall between column i-1 and i, at its middle
        at = -0.5 * inner_l + i * cl + (i - 0.5) * t
        cuts.append(_finger(fw, fd, g["height"], reach, "y", at))
    for j in range(1, g["rows"]):        # the wall between row j-1 and j, at its middle
        at = -0.5 * inner_w + j * cw + (j - 0.5) * t
        cuts.append(_finger(fw, fd, g["height"], reach, "x", at))
    return cuts


def drawer_organizer(length=200.0, width=150.0, height=50.0, clearance=1.0, cols=3, rows=2,
                     wall=1.6, floor=1.2, radius=4.0, finger=True) -> mf.Manifold:
    """One printable drawer organizer: a tray of `cols` by `rows` compartments that drops in a drawer.

    The tray is the inside of the drawer (`length` by `width`) less `clearance` on every side, so it
    drops in, and `height` millimetres high; it stands on a floor of `floor` millimetres and its
    walls and dividers are `wall` millimetres thick. The `cols` by `rows` compartments are all the
    same size, their inside corners rounded with `radius` and the corners of the tray with
    `radius + wall`, so the wall stays even. With `finger` the top of every inner wall has a
    half-round notch at its middle, wide enough and deep enough to get a finger down beside an item;
    a single compartment has no inner wall and so no notch.

    The result is one closed volume that stands on `z = 0` and is centred in x and y. Every number
    is clamped to what prints: the walls thin out before a compartment would vanish and the radius
    stays under a third of the smallest compartment; anything but a number is a ValueError.
    """
    g = _sizes(length, width, height, clearance, cols, rows, wall, floor, radius, finger)
    block = mf.Manifold.extrude(
        enclosure.rounded_rect(g["outer_length"], g["outer_width"], g["outer_radius"]),
        g["height"])
    cuts = _pockets(g) + _notches(g)
    body = block - mf.Manifold.batch_boolean(cuts, mf.OpType.Add) if cuts else block
    # the tray stands on the bed as it prints: centred in plan, with the floor on z = 0
    bb = np.asarray(body.bounding_box(), dtype=float)
    return body.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
