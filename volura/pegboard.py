"""Pegboard parts: the computational core of the hooks, shelves and holders that hang on a board.

No Qt in this file. It gives a part as one manifold, so it can be tested on its own and reused in
the model and the export.

A **pegboard** is a board of holes of `PEG_HOLE` millimetres across, on a square grid of
`PEG_PITCH`. A part hangs on two holes next to each other: an **upper peg** that goes in straight
and bends **down** behind the board, so the part hangs from it, and a **lower peg** `PEG_PITCH`
lower that comes no deeper than the back of the board and keeps the part from swinging out. Between
them the **back plate** lies against the front of the board, `PEG_PITCH + 10` millimetres high, and
the load of the part goes through the upper peg alone.

Everything is printed **lying on its side**: the profile is the shape that lies on the bed and the
part is extruded `width` millimetres straight up, so no part needs support and no part has a
bridge. The pegs take part in that: they are as thick as the print in one direction and `PEG_DIA`
in the other, which is how printed pegboard hooks are made, and what is left over in a board hole
is the play that lets you still get a hook off the board.

A **hook** is an arm of `reach` millimetres out of the bottom of the back plate with an upturned
**tip** at its end, so what hangs on it cannot slide off. A **shelf** is a flat shelf of `reach`,
stayed by a 45 degree **brace** under it and with a **lip** at the front edge. A **holder** is a
**ring** of inner diameter `size` at `reach` from the plate, for a screwdriver or a pair of
pliers, joined to the plate by an arm.
"""
from __future__ import annotations

import manifold3d as mf

from .i18n import tr

#: mm across a board hole (1/4 inch) and mm between the centres of two holes (1 inch)
PEG_HOLE, PEG_PITCH = 6.35, 25.4
#: mm of board the parts hang on: the thickness a part is made for, and the range worth asking for
BOARD, BOARD_MIN, BOARD_MAX = 5.0, 3.0, 7.0
#: mm taken off a board hole, so a peg goes into it by hand
PEG_FIT = 0.6
#: mm across a peg
PEG_DIA = PEG_HOLE - PEG_FIT
#: mm the upper peg comes through behind the board, and mm it hangs down there
PEG_THROUGH, PEG_DROP = 1.0, 6.0
#: mm of plate, wall, arm and lip
WALL, ARM = 3.0, 4.0
#: mm the back plate stands above the upper peg and below the lower one, together PEG_PITCH + 10
PLATE_TOP, PLATE_BOTTOM = 5.0, 5.0
#: mm high the tip of a hook stands up, and mm high the lip of a shelf
HOOK_TIP, SHELF_LIP = 8.0, 3.0
#: mm of print width, of reach out of the board and of the inside of a ring
WIDTH_DEFAULT, WIDTH_MIN, WIDTH_MAX = 8.0, 5.0, 20.0
REACH_DEFAULT, REACH_MIN, REACH_MAX = 40.0, 10.0, 100.0
SIZE_DEFAULT, SIZE_MIN, SIZE_MAX = 12.0, 5.0, 40.0
#: the brace of a shelf runs this part of the reach under the shelf, to a maximum
BRACE_PART, BRACE_MAX = 0.5, 20.0
#: the straight pieces a ring is approximated with
SEGMENTS = 48
#: what this module knows how to make
KINDS = ("hook", "shelf", "holder")


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _kind(kind) -> str:
    """The kind as it is spelled here, or a ValueError for anything else."""
    k = str(kind).strip().lower()
    if k not in KINDS:
        raise ValueError(tr("there is no such pegboard part"))
    return k


def _box(x0, x1, y0, y1, w) -> mf.Manifold:
    """The block between `x0`, `x1`, `y0` and `y1` of the profile, `w` mm high."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, w]).translate((x0, y0, 0.0))


def _prism(points, w) -> mf.Manifold:
    """The prism over a polygon of the profile, `w` mm high."""
    return mf.CrossSection([points]).extrude(w)


def _cyl(cx, cy, r, w) -> mf.Manifold:
    """A post of radius `r` about the point (`cx`, `cy`), `w` mm high."""
    return mf.Manifold.cylinder(w, r, r, SEGMENTS).translate((cx, cy, 0.0))


def _sizes(kind, width, reach, board, size) -> dict:
    """The clamped numbers and the places of the features, in the design frame.

    In that frame the front of the back plate is on `x = 0`, the axis of the upper peg on `y = 0`
    and the bed on `z = 0`; `dx` and `dy` are the shift that centres the part in plan.
    """
    k = _kind(kind)
    w = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    r = _clamp(reach, REACH_MIN, REACH_MAX)
    b = _clamp(board, BOARD_MIN, BOARD_MAX)
    d = _clamp(size, SIZE_MIN, SIZE_MAX)
    bore_r, ring_r = 0.5 * d, 0.5 * d + WALL
    y_bottom = -(PEG_PITCH + PLATE_BOTTOM)
    # a fat ring is pushed out until it stands clear of the plate instead of swallowing it
    ring_x = max(r, ring_r + WALL + 1.0)
    g = {"kind": k, "width": w, "reach": r, "board": b, "size": d,
         "x_back": -(b + PEG_THROUGH), "y_top": PLATE_TOP, "y_bottom": y_bottom,
         "ring_x": ring_x if k == "holder" else 0.0, "ring_y": y_bottom,
         "ring_r": ring_r, "bore_r": bore_r, "brace": BRACE_PART * r if k == "shelf" else 0.0}
    g["brace"] = min(g["brace"], BRACE_MAX)
    x_hi = g["ring_x"] + ring_r if k == "holder" else r
    y_lo = y_bottom - g["brace"]
    if k == "holder":
        y_lo = min(y_lo, y_bottom - ring_r)
    g["dx"] = -0.5 * (x_hi + g["x_back"] - 0.5 * PEG_DIA)
    g["dy"] = -0.5 * (y_lo + g["y_top"])
    return g


def sizes(kind="hook", width=WIDTH_DEFAULT, reach=REACH_DEFAULT, board=BOARD,
          size=SIZE_DEFAULT) -> dict:
    """The sizes of a part and the places of its features, in the frame `pegboard_part` returns.

    `x_face` is the front of the back plate, the face that lies against the board, and `x_back` the
    place behind the board where the upper peg bends down; `peg_y` and `lower_y` are the axes of
    the two pegs, and `ring_x`, `ring_y`, `ring_r` and `bore_r` the ring of a holder.
    """
    g = _sizes(kind, width, reach, board, size)
    dx, dy = g["dx"], g["dy"]
    return {"kind": g["kind"], "width": g["width"], "reach": g["reach"], "board": g["board"],
            "size": g["size"], "peg_dia": PEG_DIA, "wall": WALL,
            "x_face": dx, "x_back": dx + g["x_back"],
            "peg_y": dy, "lower_y": dy - PEG_PITCH,
            "y_top": dy + g["y_top"], "y_bottom": dy + g["y_bottom"],
            "ring_x": dx + g["ring_x"], "ring_y": dy + g["ring_y"],
            "ring_r": g["ring_r"], "bore_r": g["bore_r"]}


def _part(g) -> mf.Manifold:
    """The part of the design frame `g`: the mount with the two pegs, and what hangs on it."""
    w, b, r = g["width"], g["board"], g["reach"]
    y0, y1 = g["y_bottom"], g["y_top"]
    xb = g["x_back"]
    peg = 0.5 * PEG_DIA
    parts = [_box(0.0, WALL, y0, y1, w)]
    # the upper peg: straight through the board, then down behind it, so the part hangs
    parts.append(_box(xb, 0.0, -peg, peg, w))
    parts.append(_box(xb - peg, xb + peg, -PEG_DROP, peg, w))
    # the lower peg: as deep as the board and no deeper, so the part cannot swing
    parts.append(_box(-b, 0.0, -PEG_PITCH - peg, -PEG_PITCH + peg, w))
    if g["kind"] == "hook":
        parts.append(_box(0.0, r, y0, y0 + ARM, w))
        parts.append(_box(r - WALL, r, y0, y0 + HOOK_TIP, w))
    elif g["kind"] == "shelf":
        parts.append(_box(0.0, r, y0, y0 + ARM, w))
        parts.append(_box(r - WALL, r, y0 + ARM, y0 + ARM + SHELF_LIP, w))
        parts.append(_prism([[0.0, y0], [0.0, y0 - g["brace"]], [g["brace"], y0]], w))
    else:
        # the arm runs into the ring and the bore is cut afterwards, so the joint stays solid
        parts.append(_box(0.0, g["ring_x"] - g["bore_r"] + 1.0, g["ring_y"] - 0.5 * ARM,
                          g["ring_y"] + 0.5 * ARM, w))
        parts.append(_cyl(g["ring_x"], g["ring_y"], g["ring_r"], w))
    part = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if g["kind"] == "holder":
        part = part - _cyl(g["ring_x"], g["ring_y"], g["bore_r"], w)
    return part


def pegboard_part(kind="hook", width=WIDTH_DEFAULT, reach=REACH_DEFAULT, board=BOARD,
                  size=SIZE_DEFAULT) -> mf.Manifold:
    """One pegboard part of `kind`: a hook, a shelf or a holder.

    The part stands on `z = 0`, is `width` millimetres high and centred in plan, and is one closed
    volume. The board it is made for is `board` millimetres thick, the part reaches `reach`
    millimetres out of the board and a holder takes a tool of `size` millimetres across. Every
    number is clamped to what prints; an unknown kind is a ValueError.
    """
    g = _sizes(kind, width, reach, board, size)
    return _part(g).translate((g["dx"], g["dy"], 0.0))
