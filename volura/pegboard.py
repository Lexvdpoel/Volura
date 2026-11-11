"""Pegboard parts: the computational core of the board and the hooks, shelves and holders on it.

No Qt in this file. It gives a part as one manifold, so it can be tested on its own and reused in
the model and the export.

A **pegboard** is a board of holes of `PEG_HOLE` millimetres across, on a square grid of
`PEG_PITCH`. A part hangs on two holes above each other: an **upper peg** that goes through the
board **rising** into it, and a **lower peg** `PEG_PITCH` lower that comes no deeper than the back
of the board and keeps the part from swinging out. Between them the **back plate** lies against the
front of the board. The pegs are flat octagons, wider along the board than they are tall, so the
upper one has room in the round hole to rise through it. A printed peg is too thick for the bend a steel hook has behind the board - it would never
get through the hole - so the upper peg is straight and rises as steeply as the hole lets it through
a board of that thickness: to come off, the part would have to be lifted, and its own weight keeps
it on. It goes on by tilting the part, pushing the upper peg in and swinging the lower one in.

Everything that hangs is printed **lying on its side**: the profile is the shape that lies on the
bed and the part is extruded `width` millimetres straight up, so the layers run along the arm and
the part is as strong as a print gets where it bends. The pegs lie on the bed at one side of the
plate, so their flat underside needs no support.

A **hook** is an arm of `reach` millimetres out of the bottom of the back plate that rises
`HOOK_RISE` degrees, so what hangs on it slides towards the board instead of off it, with a rounded
end and a fillet under the plate where a printed hook would otherwise break. A **shelf** is a flat
shelf of `reach`, stayed by a 45 degree **brace** under it and with a **lip** at the front edge. A
**holder** is an upright **ring** of inner diameter `size` at `reach` from the plate, so a
screwdriver or a pair of pliers drops into it from above; it lies on its side on the bed, with a
45 degree roof in the bore and 45 degree flanks under the ring, so it prints without support.

The **board** itself is `cols` by `rows` holes on the pitch, `board` millimetres thick, with a rim
along its back edges that holds it `standoff` millimetres off the wall - room for the bend of the
upper pegs - and four bosses with a countersunk screw hole between the holes near the corners. It
prints face down, with the rim and the bosses standing up.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: mm across a board hole (1/4 inch) and mm between the centres of two holes (1 inch)
PEG_HOLE, PEG_PITCH = 6.35, 25.4
#: mm of board the parts hang on: the thickness a part is made for, and the range worth asking for
BOARD, BOARD_MIN, BOARD_MAX = 5.0, 3.0, 7.0
#: mm of a peg along the board (the height it prints), mm of it up and down, and mm of chamfer on
#: its four edges: a flat octagon, so a peg that rises still passes a round hole
PEG_WIDE, PEG_TALL, PEG_CHAMFER = 4.8, 3.8, 1.0
#: mm the upper peg comes out behind the board; mm of a hole a peg keeps free all round, so it goes
#: in by hand; and the steepest the upper peg rises, in degrees
PEG_BEYOND, PEG_CLEAR, PEG_RISE_MAX = 1.5, 0.25, 15.0
#: mm of plate, wall and shelf; mm of hook arm, which carries the most
WALL, ARM, HOOK_ARM = 3.0, 4.0, 5.0
#: mm the back plate stands above the upper peg and below the lower one
PLATE_TOP, PLATE_BOTTOM = 5.0, 5.0
#: degrees the arm of a hook rises, mm of fillet under the plate, mm high the lip of a shelf
HOOK_RISE, HOOK_FILLET, SHELF_LIP = 15.0, 6.0, 3.0
#: mm of print width, of reach out of the board and of the inside of a ring
WIDTH_DEFAULT, WIDTH_MIN, WIDTH_MAX = 8.0, 6.0, 20.0
REACH_DEFAULT, REACH_MIN, REACH_MAX = 40.0, 10.0, 100.0
SIZE_DEFAULT, SIZE_MIN, SIZE_MAX = 12.0, 5.0, 40.0
#: mm high the ring of a holder is, and the least plastic under its bore on the bed
HOLDER_HEIGHT, HOLDER_FLOOR = 10.0, 1.2
#: the brace of a shelf runs this part of the reach under the shelf, to a maximum
BRACE_PART, BRACE_MAX = 0.5, 20.0
#: holes of a board across and down, and how far it stands off the wall
COLS_DEFAULT, COLS_MIN, COLS_MAX = 6, 2, 12
ROWS_DEFAULT, ROWS_MIN, ROWS_MAX = 6, 2, 12
STANDOFF_DEFAULT, STANDOFF_MIN, STANDOFF_MAX = 15.0, 10.0, 40.0
#: mm of rim along the back of a board; mm of boss, screw hole and countersink at the corners
RIM, BOSS, SCREW, SCREW_HEAD = 4.0, 6.0, 4.4, 8.5
#: the straight pieces a ring is approximated with
SEGMENTS = 48
#: what this module knows how to make
KINDS = ("hook", "shelf", "holder", "board")


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


def _peg_corners() -> tuple:
    """The two corners of a quarter of the peg section: (along the board, up), from its axis."""
    a, b = 0.5 * PEG_WIDE, 0.5 * PEG_TALL
    return (a, b - PEG_CHAMFER), (a - PEG_CHAMFER, b)


def _octagon(cx, cy) -> mf.CrossSection:
    """The section of a peg about (`cx`, `cy`): `PEG_WIDE` along x, `PEG_TALL` along y."""
    (a, c), (d, b) = _peg_corners()
    return mf.CrossSection([[(cx + a, cy - c), (cx + a, cy + c), (cx + d, cy + b),
                             (cx - d, cy + b), (cx - a, cy + c), (cx - a, cy - c),
                             (cx - d, cy - b), (cx + d, cy - b)]])


def _peg_x(x0, x1, y) -> mf.Manifold:
    """A peg along the profile from `x0` to `x1` at height `y`, lying on the bed."""
    return (_octagon(0.5 * PEG_WIDE, y).extrude(x1 - x0)          # (z, y) section, along x
            .transform([[0.0, 0.0, 1.0, x0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))


def _peg_rise(b) -> float:
    """Degrees the upper peg rises through a board `b` mm thick and still passes the hole.

    The peg rises about the middle of the board, so at the front and the back of the board its
    section stands `b / 2 * tan` higher or lower, and at an angle it is `1 / cos` taller. Every
    corner of the octagon must then still lie inside the round hole less `PEG_CLEAR`; the
    steepest angle for which they do is found by halving.
    """
    room = 0.5 * PEG_HOLE - PEG_CLEAR

    def fits(deg):
        t = math.radians(deg)
        return all(math.hypot(y / math.cos(t) + 0.5 * b * math.tan(t), z) <= room
                   for z, y in _peg_corners())

    lo, hi = 0.0, PEG_RISE_MAX
    if fits(hi):
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


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
    # the axis of the ring lies on the bed's side as low as the 45 degree flanks under the ring
    # allow, and never so low that the bore comes closer to the bed than HOLDER_FLOOR
    ring_z = max(ring_r * math.cos(math.pi / 4.0), bore_r + HOLDER_FLOOR)
    y_bottom = -(PEG_PITCH + PLATE_BOTTOM)
    ring_x = max(r, ring_r + WALL + 1.0)
    g = {"kind": k, "width": w, "reach": r, "board": b, "size": d,
         "x_tip": -(b + PEG_BEYOND), "peg_rise": _peg_rise(b),
         "y_top": PLATE_TOP, "y_bottom": y_bottom,
         "ring_x": ring_x if k == "holder" else 0.0, "ring_y": y_bottom, "ring_z": ring_z,
         "ring_r": ring_r, "bore_r": bore_r,
         "rise": r * math.tan(math.radians(HOOK_RISE)) if k == "hook" else 0.0,
         "brace": min(BRACE_PART * r, BRACE_MAX) if k == "shelf" else 0.0}
    # the end of the risen peg is square to the peg, so its lower edge stands a little further back
    x_lo = g["x_tip"] - 0.5 * PEG_TALL * math.sin(math.radians(g["peg_rise"]))
    if k == "holder":
        x_hi = ring_x + _ring_half_width(g)
    elif k == "hook":
        x_hi = r + 0.5 * HOOK_ARM
    else:
        x_hi = r
    y_hi = max(g["y_top"], y_bottom + HOOK_ARM + g["rise"] + 0.5 * HOOK_ARM)
    y_lo = y_bottom - g["brace"]
    g["dx"] = -0.5 * (x_hi + x_lo)
    g["dy"] = -0.5 * (y_lo + y_hi)
    return g


def _ring_half_width(g) -> float:
    """How far the ring of a holder reaches to either side of its axis, on the bed or above."""
    return max(g["ring_r"], math.sqrt(2.0) * g["ring_r"] - g["ring_z"])


def sizes(kind="hook", width=WIDTH_DEFAULT, reach=REACH_DEFAULT, board=BOARD,
          size=SIZE_DEFAULT) -> dict:
    """The sizes of a part and the places of its features, in the frame `pegboard_part` returns.

    `x_face` is the front of the back plate, the face that lies against the board, and `x_tip`
    the end of the upper peg behind the board; `peg_y` and `lower_y` are the axes of the two pegs
    where they pass the middle of the board, `peg_rise` the degrees the upper one rises and `peg_z`
    the height of their axes above the bed. `ring_x`, `ring_y`, `ring_z`,
    `ring_r` and `bore_r` are the ring of a holder: its axis stands along y, from `ring_y` up
    `HOLDER_HEIGHT`.
    """
    g = _sizes(kind, width, reach, board, size)
    dx, dy = g["dx"], g["dy"]
    return {"kind": g["kind"], "width": g["width"], "reach": g["reach"], "board": g["board"],
            "size": g["size"], "peg_wide": PEG_WIDE, "peg_tall": PEG_TALL, "wall": WALL,
            "x_face": dx, "x_tip": dx + g["x_tip"], "peg_rise": g["peg_rise"],
            "peg_y": dy, "lower_y": dy - PEG_PITCH, "peg_z": 0.5 * PEG_WIDE,
            "y_top": dy + g["y_top"], "y_bottom": dy + g["y_bottom"],
            "ring_x": dx + g["ring_x"], "ring_y": dy + g["ring_y"], "ring_z": g["ring_z"],
            "ring_r": g["ring_r"], "bore_r": g["bore_r"], "rise": g["rise"]}


def _ring(g) -> tuple:
    """(solid, bore) of the ring of a holder, standing along y in the design frame.

    The section across the axis lies in the plane of x and z: the outside is the circle of
    `ring_r` with 45 degree flanks down to the bed and a 45 degree peak, the bore the circle of
    `bore_r` with a 45 degree roof under that peak, so neither has an overhang steeper than that
    and the plastic over the roof is as thick as the ring.
    """
    R, rb, zc = g["ring_r"], g["bore_r"], g["ring_z"]
    hw = max(math.sqrt(2.0) * R - zc, 1e-3)

    def peak(z):
        """A point at height `z` over the axis, as a sliver a hull can take."""
        return mf.CrossSection([[(-1e-3, z - 1e-3), (1e-3, z - 1e-3), (0.0, z)]])

    outside = mf.CrossSection.batch_hull([mf.CrossSection.circle(R, SEGMENTS).translate((0, zc)),
                                          mf.CrossSection.square((2.0 * hw, 1e-3))
                                          .translate((-hw, 0.0)),
                                          peak(zc + math.sqrt(2.0) * R)])
    outside = outside ^ mf.CrossSection.square((4.0 * R, 4.0 * R)).translate((-2.0 * R, 0.0))
    bore = mf.CrossSection.batch_hull([mf.CrossSection.circle(rb, SEGMENTS).translate((0, zc)),
                                       peak(zc + math.sqrt(2.0) * rb)])
    # the (x, z) section extruded along y: x stays, the extrusion becomes y, the section's y is z
    M = [[1.0, 0.0, 0.0, g["ring_x"]], [0.0, 0.0, 1.0, g["ring_y"]], [0.0, 1.0, 0.0, 0.0]]
    return (outside.extrude(HOLDER_HEIGHT).transform(M),
            bore.extrude(HOLDER_HEIGHT + 2.0).translate((0.0, 0.0, -1.0)).transform(M))


def _part(g) -> mf.Manifold:
    """The part of the design frame `g`: the mount with the two pegs, and what hangs on it."""
    w, b, r = g["width"], g["board"], g["reach"]
    y0, y1 = g["y_bottom"], g["y_top"]
    parts = [_box(0.0, WALL, y0, y1, w)]
    # the upper peg: straight, rising into the board about the middle of the hole
    rise = g["peg_rise"]
    pivot = -0.5 * b
    reach_back = (pivot - g["x_tip"]) / math.cos(math.radians(rise))
    reach_front = (0.5 * WALL - pivot) / math.cos(math.radians(rise))
    parts.append(_peg_x(pivot - reach_back, pivot + reach_front, 0.0)
                 .translate((-pivot, 0.0, 0.0)).rotate((0.0, 0.0, -rise))
                 .translate((pivot, 0.0, 0.0)))
    # the lower peg: as deep as the board and no deeper, so the part cannot swing
    parts.append(_peg_x(-b, 0.5 * WALL, -PEG_PITCH))
    if g["kind"] == "hook":
        t, rise = HOOK_ARM, g["rise"]
        # the arm rises from the bottom of the plate and ends in a rounded tip
        parts.append(_prism([[0.0, y0], [r, y0 + rise], [r, y0 + rise + t], [0.0, y0 + t]], w))
        parts.append(mf.Manifold.cylinder(w, 0.5 * t, 0.5 * t, SEGMENTS)
                     .translate((r, y0 + rise + 0.5 * t, 0.0)))
        # a fillet in the corner of arm and plate, where the load bends the hook
        f = HOOK_FILLET
        parts.append(_prism([[WALL, y0 + t + f], [WALL, y0 + t],
                             [WALL + f, y0 + t + (WALL + f) * math.tan(math.radians(HOOK_RISE))]],
                            w))
    elif g["kind"] == "shelf":
        parts.append(_box(0.0, r, y0, y0 + ARM, w))
        parts.append(_box(r - WALL, r, y0 + ARM, y0 + ARM + SHELF_LIP, w))
        parts.append(_prism([[0.0, y0], [0.0, y0 - g["brace"]], [g["brace"], y0]], w))
    else:
        ring, bore = _ring(g)
        # the arm runs into the ring and the bore is cut afterwards, so the joint stays solid
        parts.append(_box(0.0, g["ring_x"], y0, y0 + ARM, w))
        parts.append(ring)
    part = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if g["kind"] == "holder":
        part = part - bore
    return part


def board_sizes(cols=COLS_DEFAULT, rows=ROWS_DEFAULT, board=BOARD,
                standoff=STANDOFF_DEFAULT) -> dict:
    """The numbers of a board: holes across and down, thickness, standoff and outside size."""
    c = int(round(_clamp(cols, COLS_MIN, COLS_MAX)))
    r = int(round(_clamp(rows, ROWS_MIN, ROWS_MAX)))
    return {"cols": c, "rows": r, "board": _clamp(board, BOARD_MIN, BOARD_MAX),
            "standoff": _clamp(standoff, STANDOFF_MIN, STANDOFF_MAX),
            "length": c * PEG_PITCH, "height": r * PEG_PITCH}


def pegboard_board(cols=COLS_DEFAULT, rows=ROWS_DEFAULT, board=BOARD,
                   standoff=STANDOFF_DEFAULT) -> mf.Manifold:
    """A pegboard of `cols` by `rows` holes, face down on the bed and centred in plan.

    The holes stand half a pitch from the edges, so boards laid side by side keep the pitch. The
    rim and the four bosses stand `standoff` millimetres up from the back of the board; a boss
    sits between the four holes nearest a corner, with a screw hole that is countersunk in the
    face of the board.
    """
    s = board_sizes(cols, rows, board, standoff)
    L, H, t, h = s["length"], s["height"], s["board"], s["standoff"]
    face = mf.CrossSection.square((L, H))
    rim = face - mf.CrossSection.square((L - 2.0 * RIM, H - 2.0 * RIM)).translate((RIM, RIM))
    corners = [(x, y) for x in (PEG_PITCH, L - PEG_PITCH) for y in (PEG_PITCH, H - PEG_PITCH)]
    bosses = [mf.CrossSection.circle(BOSS, SEGMENTS).translate(c) for c in corners]
    body = face.extrude(t) + mf.CrossSection.batch_boolean([rim] + bosses, mf.OpType.Add) \
        .extrude(h).translate((0.0, 0.0, t))
    holes = [mf.CrossSection.circle(0.5 * PEG_HOLE, SEGMENTS)
             .translate(((i + 0.5) * PEG_PITCH, (j + 0.5) * PEG_PITCH))
             for i in range(s["cols"]) for j in range(s["rows"])]
    cut = [mf.CrossSection.batch_boolean(holes, mf.OpType.Add).extrude(t + 2.0)
           .translate((0.0, 0.0, -1.0))]
    sink = 0.5 * (SCREW_HEAD - SCREW)
    for x, y in corners:
        cut.append(mf.Manifold.cylinder(t + h + 2.0, 0.5 * SCREW, 0.5 * SCREW, SEGMENTS)
                   .translate((x, y, -1.0)))
        cut.append(mf.Manifold.cylinder(sink + 1.0, 0.5 * SCREW_HEAD + 1.0, 0.5 * SCREW,
                                        SEGMENTS).translate((x, y, -1.0)))
    part = body - mf.Manifold.batch_boolean(cut, mf.OpType.Add)
    return part.translate((-0.5 * L, -0.5 * H, 0.0))


def pegboard_part(kind="hook", width=WIDTH_DEFAULT, reach=REACH_DEFAULT, board=BOARD,
                  size=SIZE_DEFAULT, cols=COLS_DEFAULT, rows=ROWS_DEFAULT,
                  standoff=STANDOFF_DEFAULT) -> mf.Manifold:
    """One pegboard part of `kind`: a hook, a shelf, a holder or the board itself.

    A part that hangs stands on `z = 0`, is `width` millimetres high (a holder's ring may stand
    higher) and centred in plan, and is one closed volume. The board it is made for is `board`
    millimetres thick, the part reaches `reach` millimetres out of the board and a holder takes a
    tool of `size` millimetres across. A "board" is `pegboard_board` of `cols`, `rows`, `board` and
    `standoff`. Every number is clamped to what prints; an unknown kind is a ValueError.
    """
    if _kind(kind) == "board":
        return pegboard_board(cols, rows, board, standoff)
    g = _sizes(kind, width, reach, board, size)
    return _part(g).translate((g["dx"], g["dy"], 0.0))
