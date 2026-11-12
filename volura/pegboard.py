"""Pegboard parts: the computational core of the board and the hooks, shelves and holders on it.

No Qt in this file. It gives a part as one manifold, so it can be tested on its own and reused in
the model and the export.

A **pegboard** is a board of holes of `PEG_HOLE` millimetres across, on a square grid of
`PEG_PITCH`. A part hangs on two holes above each other, like a steel pegboard hook: an **upper
peg** that is an **L-hook** and a straight **lower peg** `PEG_PITCH` lower. Both are round rods of
`PEG_D`. The upper peg goes straight through the board and behind it bends up over a rounded
bend into a **tip** that points up, `TIP` above the axis of the peg, with a rounded end. Hung on
the board the tip stands behind the board above the hole, so the part cannot be pulled out, and
its weight keeps it on. The straight part is as long as the board plus `PEG_PLAY`, measured to
the inside of the tip, so the back plate lies against the front of the board and the tip against
its back with less than half a millimetre to spare and the part does not hang crooked. The lower
peg is as deep as the board and no deeper, with a chamfer at its end, so the part cannot swing.

It goes on as a steel hook does: tilt the part a quarter turn, front end up, so the tip points
into the hole, push the tip through, and swing the part down while the bend slides round the
back edge of the hole until the lower peg drops into its hole. `insertion_path` checks that
motion in the plane through the pegs; `PEG_D`, `BEND_R` and `TIP` are chosen so it clears every
board from `BOARD_MIN` to `BOARD_MAX`.

Everything that hangs is printed **lying on its side**: the profile is the shape that lies on the
bed and the part is extruded `width` millimetres up, so the layers run along the arm and the part
is as strong as a print gets where it bends. A part is symmetric about its middle plane; the pegs
are round with their axes in that plane, so they print with support under them.

A **hook** is an arm of `reach` millimetres out of the bottom of the back plate that rises
`HOOK_RISE` degrees, so what hangs on it slides towards the board instead of off it, with a rounded
end and a fillet under the plate where a printed hook would otherwise break. A **shelf** is a flat
shelf of `reach`, stayed by a 45 degree **brace** under it and with a **lip** at the front edge. A
**holder** is a plain round **ring** of inner diameter `size` at `reach` from the plate, its axis
upright as it hangs, so a screwdriver or a pair of pliers drops into it from above.

The **board** itself is `cols` by `rows` holes on the pitch, `board` millimetres thick, with a
chamfer at the front of every hole so a peg finds its way in, and a rim along its back edges that
holds it `standoff` millimetres off the wall - room for the tips behind it - and four bosses with a
countersunk screw hole between the holes near the corners. It prints face down, with the rim and
the bosses standing up.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm across a board hole (1/4 inch) and mm between the centres of two holes (1 inch)
PEG_HOLE, PEG_PITCH = 6.35, 25.4
#: mm of board the parts hang on: the thickness a part is made for, and the range worth asking for
BOARD, BOARD_MIN, BOARD_MAX = 5.0, 3.0, 7.0
#: mm across both pegs, like the 3/16 inch wire of a steel hook; the radius of the bend of the
#: upper peg, to the axis of the rod; and how high the end of its tip stands above its axis
PEG_D, BEND_R, TIP = 4.4, 4.0, 6.2
#: mm along the peg the part can move between the front of the board and the back: the play
PEG_PLAY = 0.4
#: mm of chamfer at the end of the lower peg and at the front of the holes of a board
PEG_CHAMFER, HOLE_CHAMFER = 0.5, 0.5
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
#: mm high the ring of a holder is
HOLDER_HEIGHT = 10.0
#: the brace of a shelf runs this part of the reach under the shelf, to a maximum
BRACE_PART, BRACE_MAX = 0.5, 20.0
#: holes of a board across and down, and how far it stands off the wall
COLS_DEFAULT, COLS_MIN, COLS_MAX = 6, 2, 12
ROWS_DEFAULT, ROWS_MIN, ROWS_MAX = 6, 2, 12
STANDOFF_DEFAULT, STANDOFF_MIN, STANDOFF_MAX = 15.0, 10.0, 40.0
#: mm of rim along the back of a board; mm of boss, screw hole and countersink at the corners
RIM, BOSS, SCREW, SCREW_HEAD = 4.0, 6.0, 4.4, 8.5
#: the straight pieces a round is approximated with, and the steps of the bend of a peg
SEGMENTS, BEND_STEPS = 48, 12
#: what this module knows how to make
KINDS = ("hook", "shelf", "holder", "board")
#: the insertion check: degrees per step, mm the part may shift per step, the grid it shifts on,
#: and the mm2 of overlap with the board that still counts as clear (the corners of the polygons)
INSERT_STEP, INSERT_SHIFT, INSERT_GRID, INSERT_OVERLAP = 3.0, 3.0, 0.1, 0.05


def _clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _kind(kind) -> str:
    """The kind as it is spelled here, or a ValueError for anything else."""
    k = str(kind).strip().lower()
    if k not in KINDS:
        raise ValueError(tr("there is no such pegboard part"))
    return k


def _box(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """The block between `x0`, `x1`, `y0`, `y1`, `z0` and `z1`."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate((x0, y0, z0))


def _prism(points, z0, z1) -> mf.Manifold:
    """The prism over a polygon of the profile, from `z0` to `z1`."""
    return mf.CrossSection([points]).extrude(z1 - z0).translate((0.0, 0.0, z0))


def _tip_x(b) -> float:
    """The axis of the tip behind a board `b` mm thick: its inside lies `PEG_PLAY` behind it."""
    return -(b + PEG_PLAY) - 0.5 * PEG_D


def _hook_path(b, x_start):
    """The axis of the upper peg in the profile: points and their directions, from the plate to
    the end of the straight part of the tip (the rounded end comes on top of that)."""
    xt = _tip_x(b)
    cx, cy = xt + BEND_R, BEND_R                # the centre of the bend
    pts = [((x_start, 0.0), (-1.0, 0.0))]
    for i in range(BEND_STEPS + 1):             # from pointing back to pointing up
        a = math.pi * (1.5 - 0.5 * i / BEND_STEPS)
        pts.append(((cx + BEND_R * math.cos(a), cy + BEND_R * math.sin(a)),
                    (math.sin(a), -math.cos(a))))
    if TIP - 0.5 * PEG_D > cy + 1e-6:          # a straight piece above the bend
        pts.append(((xt, TIP - 0.5 * PEG_D), (0.0, 1.0)))
    return pts


def _upper_peg(b, x_start, zc) -> mf.Manifold:
    """The L-hook: a round rod along `_hook_path` with a rounded end, its axis at height `zc`.

    It is one sweep - rings of points round the axis, joined into a closed mesh - so the straight
    part, the bend and the tip have no seams between them.
    """
    r = 0.5 * PEG_D
    n = SEGMENTS
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    cos, sin = np.cos(phi), np.sin(phi)
    path = _hook_path(b, x_start)
    rings = []
    for (px, py), (tx, ty) in path:
        nx, ny = -ty, tx                        # the normal in the profile plane
        rings.append([(px + r * c * nx, py + r * c * ny, zc + r * s) for c, s in zip(cos, sin)])
    (ex, ey), (tx, ty) = path[-1]
    nx, ny = -ty, tx
    quarter = n // 4
    for k in range(1, quarter):                 # the rounded end: a half sphere of rings
        a = 0.5 * math.pi * k / quarter
        ca, sa = math.cos(a), math.sin(a)
        rings.append([(ex + r * (ca * c * nx + sa * tx), ey + r * (ca * c * ny + sa * ty),
                       zc + r * ca * s) for c, s in zip(cos, sin)])
    verts = [v for ring in rings for v in ring]
    apex = len(verts)
    verts.append((ex + r * tx, ey + r * ty, zc))
    start = len(verts)
    verts.append((x_start, 0.0, zc))            # the middle of the flat start
    tris = []
    for i in range(len(rings) - 1):
        a0, a1 = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            tris += [(a0 + j, a0 + k, a1 + k), (a0 + j, a1 + k, a1 + j)]
    last = (len(rings) - 1) * n
    for j in range(n):
        k = (j + 1) % n
        tris.append((last + j, last + k, apex))
        tris.append((start, k, j))
    v = np.asarray(verts, dtype=np.float32)
    t = np.asarray(tris, dtype=np.uint32)
    m = mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=t))
    if m.status() != mf.Error.NoError or m.volume() <= 0:
        m = mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=np.ascontiguousarray(t[:, ::-1])))
    return m


def _lower_peg(b, x_start, y, zc) -> mf.Manifold:
    """The lower peg: a straight round rod from `x_start` to the back of the board, chamfered."""
    r = 0.5 * PEG_D
    length = x_start + b
    prof = mf.CrossSection([[(0.0, 0.0), (r - PEG_CHAMFER, 0.0), (r, PEG_CHAMFER), (r, length),
                             (0.0, length)]])
    # the revolved rod stands along z with its chamfered end at z = 0; laid along +x from there
    rod = mf.Manifold.revolve(prof, SEGMENTS)
    return rod.rotate((0.0, 90.0, 0.0)).translate((-b, y, zc))


def _sizes(kind, width, reach, board, size) -> dict:
    """The clamped numbers and the places of the features, in the design frame.

    In that frame the front of the back plate is on `x = 0`, the axis of the upper peg on `y = 0`
    and the bed on `z = 0`; the part is symmetric about `z = mid`; `dx` and `dy` are the shift
    that centres the part in plan.
    """
    k = _kind(kind)
    w = _clamp(width, WIDTH_MIN, WIDTH_MAX)
    r = _clamp(reach, REACH_MIN, REACH_MAX)
    b = _clamp(board, BOARD_MIN, BOARD_MAX)
    d = _clamp(size, SIZE_MIN, SIZE_MAX)
    bore_r, ring_r = 0.5 * d, 0.5 * d + WALL
    y_bottom = -(PEG_PITCH + PLATE_BOTTOM)
    ring_x = max(r, ring_r + WALL + 1.0)
    # a ring wider than the part lifts the middle plane, so the ring stands on the bed
    mid = max(0.5 * w, ring_r) if k == "holder" else 0.5 * w
    g = {"kind": k, "width": w, "reach": r, "board": b, "size": d, "mid": mid,
         "x_tip": _tip_x(b) - 0.5 * PEG_D, "y_top": PLATE_TOP, "y_bottom": y_bottom,
         "ring_x": ring_x if k == "holder" else 0.0, "ring_y": y_bottom,
         "ring_r": ring_r, "bore_r": bore_r,
         "rise": r * math.tan(math.radians(HOOK_RISE)) if k == "hook" else 0.0,
         "brace": min(BRACE_PART * r, BRACE_MAX) if k == "shelf" else 0.0}
    x_lo = g["x_tip"]
    if k == "holder":
        x_hi = ring_x + ring_r
    elif k == "hook":
        x_hi = r + 0.5 * HOOK_ARM
    else:
        x_hi = r
    y_hi = max(g["y_top"], TIP, y_bottom + HOOK_ARM + g["rise"] + 0.5 * HOOK_ARM)
    y_lo = y_bottom - g["brace"]
    g["dx"] = -0.5 * (x_hi + x_lo)
    g["dy"] = -0.5 * (y_lo + y_hi)
    return g


def sizes(kind="hook", width=WIDTH_DEFAULT, reach=REACH_DEFAULT, board=BOARD,
          size=SIZE_DEFAULT) -> dict:
    """The sizes of a part and the places of its features, in the frame `pegboard_part` returns.

    `x_face` is the front of the back plate, the face that lies against the board, and `x_tip`
    the back of the tip of the upper peg behind the board; `peg_y` and `lower_y` are the axes of
    the two pegs, `peg_z` the height of their axes (the middle plane of the part), `peg_d` their
    diameter and `tip` how high the end of the tip stands above the upper axis. `ring_x`,
    `ring_y`, `ring_z`, `ring_r` and `bore_r` are the ring of a holder: its axis stands along y,
    from `ring_y` up `HOLDER_HEIGHT`.
    """
    g = _sizes(kind, width, reach, board, size)
    dx, dy = g["dx"], g["dy"]
    return {"kind": g["kind"], "width": g["width"], "reach": g["reach"], "board": g["board"],
            "size": g["size"], "peg_d": PEG_D, "tip": TIP, "wall": WALL,
            "x_face": dx, "x_tip": dx + g["x_tip"],
            "peg_y": dy, "lower_y": dy - PEG_PITCH, "peg_z": g["mid"],
            "y_top": dy + g["y_top"], "y_bottom": dy + g["y_bottom"],
            "ring_x": dx + g["ring_x"], "ring_y": dy + g["ring_y"], "ring_z": g["mid"],
            "ring_r": g["ring_r"], "bore_r": g["bore_r"], "rise": g["rise"]}


def _ring(g) -> tuple:
    """(solid, bore) of the ring of a holder: plain round, standing along y in the design frame."""
    # the cylinder stands along z; z becomes y, and its y becomes z around the middle plane
    M = [[1.0, 0.0, 0.0, g["ring_x"]], [0.0, 0.0, 1.0, g["ring_y"]], [0.0, 1.0, 0.0, g["mid"]]]
    outside = mf.Manifold.cylinder(HOLDER_HEIGHT, g["ring_r"], g["ring_r"], SEGMENTS)
    bore = mf.Manifold.cylinder(HOLDER_HEIGHT + 2.0, g["bore_r"], g["bore_r"], SEGMENTS) \
        .translate((0.0, 0.0, -1.0))
    return outside.transform(M), bore.transform(M)


def _part(g) -> mf.Manifold:
    """The part of the design frame `g`: the mount with the two pegs, and what hangs on it."""
    w, b, r, mid = g["width"], g["board"], g["reach"], g["mid"]
    z0, z1 = mid - 0.5 * w, mid + 0.5 * w
    y0, y1 = g["y_bottom"], g["y_top"]
    parts = [_box(0.0, WALL, y0, y1, z0, z1),
             _upper_peg(b, 0.5 * WALL, mid),
             _lower_peg(b, 0.5 * WALL, -PEG_PITCH, mid)]
    if g["kind"] == "hook":
        t, rise = HOOK_ARM, g["rise"]
        # the arm rises from the bottom of the plate and ends in a rounded tip
        parts.append(_prism([[0.0, y0], [r, y0 + rise], [r, y0 + rise + t], [0.0, y0 + t]],
                            z0, z1))
        parts.append(mf.Manifold.cylinder(w, 0.5 * t, 0.5 * t, SEGMENTS)
                     .translate((r, y0 + rise + 0.5 * t, z0)))
        # a fillet in the corner of arm and plate, where the load bends the hook
        f = HOOK_FILLET
        parts.append(_prism([[WALL, y0 + t + f], [WALL, y0 + t],
                             [WALL + f, y0 + t + (WALL + f) * math.tan(math.radians(HOOK_RISE))]],
                            z0, z1))
    elif g["kind"] == "shelf":
        parts.append(_box(0.0, r, y0, y0 + ARM, z0, z1))
        parts.append(_box(r - WALL, r, y0 + ARM, y0 + ARM + SHELF_LIP, z0, z1))
        parts.append(_prism([[0.0, y0], [0.0, y0 - g["brace"]], [g["brace"], y0]], z0, z1))
    else:
        ring, bore = _ring(g)
        # the arm runs into the ring and the bore is cut afterwards, so the joint stays solid
        parts.append(_box(0.0, g["ring_x"], y0, y0 + ARM, z0, z1))
        parts.append(ring)
    part = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if g["kind"] == "holder":
        part = part - bore
    return part


# ---- going on the board
def mount_section(board=BOARD) -> mf.CrossSection:
    """The back plate and the two pegs cut through their axes, in the design frame."""
    b = _clamp(board, BOARD_MIN, BOARD_MAX)
    r = 0.5 * PEG_D
    path = _hook_path(b, 0.5 * WALL)
    left = [(px - r * ty, py + r * tx) for (px, py), (tx, ty) in path]
    right = [(px + r * ty, py - r * tx) for (px, py), (tx, ty) in path]
    (ex, ey), _t = path[-1]
    end = [(ex + r * math.cos(a), ey + r * math.sin(a))
           for a in np.linspace(0.0, math.pi, SEGMENTS // 2 + 1)]
    upper = mf.CrossSection([right + end[1:-1] + left[::-1]], mf.FillRule.NonZero)
    plate = mf.CrossSection.square((WALL, PLATE_TOP + PEG_PITCH + PLATE_BOTTOM)) \
        .translate((0.0, -PEG_PITCH - PLATE_BOTTOM))
    y, c = -PEG_PITCH, PEG_CHAMFER
    lower = mf.CrossSection([[(-b, y - r + c), (-b + c, y - r), (0.5 * WALL, y - r),
                              (0.5 * WALL, y + r), (-b + c, y + r), (-b, y + r - c)]])
    return mf.CrossSection.batch_boolean([plate, upper, lower], mf.OpType.Add)


def board_section(board=BOARD) -> mf.CrossSection:
    """A board `board` mm thick cut through two holes above each other: front face on `x = 0`, the
    upper hole on `y = 0`."""
    b = _clamp(board, BOARD_MIN, BOARD_MAX)
    h = 0.5 * PEG_HOLE
    slab = mf.CrossSection.square((b, 4.0 * PEG_PITCH)).translate((-b, -2.5 * PEG_PITCH))
    holes = [mf.CrossSection.square((b + 2.0, PEG_HOLE)).translate((-b - 1.0, y - h))
             for y in (0.0, -PEG_PITCH)]
    return slab - mf.CrossSection.batch_boolean(holes, mf.OpType.Add)


def hanging_drop() -> float:
    """How far below the middle of the hole the upper peg hangs: it rests on the bottom of it."""
    return 0.5 * (PEG_HOLE - PEG_D)


def placed(section, board, degrees, dx, dy) -> mf.CrossSection:
    """`section` turned `degrees` about the middle of the upper hole, then moved by `dx`, `dy`."""
    px = -0.5 * _clamp(board, BOARD_MIN, BOARD_MAX)
    return section.translate((-px, 0.0)).rotate(degrees).translate((px + dx, dy))


def insertion_path(board=BOARD):
    """How a part goes on a board `board` mm thick, or None when it does not.

    This is checked in the plane through the axes of the pegs, where a round rod in a round hole
    clears exactly when its section clears. The part starts turned a quarter turn (front end up),
    so the tip points into the upper hole, and is pushed in as far as it goes. Then it turns back
    `INSERT_STEP` degrees at a time about the middle of the upper hole, and at every step it may
    shift up to `INSERT_SHIFT` mm on a grid of `INSERT_GRID`, as a hand does; a position counts as
    clear when the part overlaps the board by no more than `INSERT_OVERLAP` mm2. The result is the
    list of (degrees, dx, dy) from the start to hanging, where the upper peg rests on the bottom of
    its hole.
    """
    b = _clamp(board, BOARD_MIN, BOARD_MAX)
    part, slab = mount_section(b), board_section(b)
    grid = INSERT_GRID

    def clear(deg, dx, dy):
        return (placed(part, b, deg, dx, dy) ^ slab).area() <= INSERT_OVERLAP

    goal = (0.0, -hanging_drop())
    start = None
    for i in range(-60, 61):                     # the tip in front of the hole, then pushed in
        dy = i * grid
        dx = 40.0
        if not clear(90.0, dx, dy):
            continue
        while dx > -20.0 and clear(90.0, dx - grid, dy):
            dx -= grid
        if start is None or dx < start[1]:
            start = (90.0, dx, dy)
    if start is None:
        return None
    path = [start]
    k = int(round(INSERT_SHIFT / grid))
    shifts = [(i * grid, j * grid) for i in range(-k, k + 1) for j in range(-k, k + 1)
              if i * i + j * j <= k * k]
    for s in range(1, int(round(90.0 / INSERT_STEP)) + 1):
        deg = 90.0 - s * INSERT_STEP
        _d, x0, y0 = path[-1]
        # of the positions within reach, the one nearest to hanging that is clear
        for ddx, ddy in sorted(shifts, key=lambda p: math.hypot(x0 + p[0] - goal[0],
                                                                y0 + p[1] - goal[1])):
            if clear(deg, x0 + ddx, y0 + ddy):
                path.append((deg, x0 + ddx, y0 + ddy))
                break
        else:
            return None
    _d, x0, y0 = path[-1]
    if math.hypot(x0 - goal[0], y0 - goal[1]) > INSERT_SHIFT or not clear(0.0, *goal):
        return None
    return path + [(0.0, goal[0], goal[1])]


# ---- the board
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

    The holes stand half a pitch from the edges, so boards laid side by side keep the pitch, and
    have a `HOLE_CHAMFER` at the front. The rim and the four bosses stand `standoff` millimetres
    up from the back of the board; a boss sits between the four holes nearest a corner, with a
    screw hole that is countersunk in the face of the board.
    """
    s = board_sizes(cols, rows, board, standoff)
    L, H, t, h = s["length"], s["height"], s["board"], s["standoff"]
    face = mf.CrossSection.square((L, H))
    inner = mf.CrossSection.square((L - 2.0 * RIM, H - 2.0 * RIM)).translate((RIM, RIM))
    # the rim is flush with the edges of the board, so it is cut out of one block that stands
    # the whole height: a frame that is stood on the back of the board touches it face to face,
    # and the seam between them is a sliver the model cannot close
    body = face.extrude(t + h) - inner.extrude(h + 1.0).translate((0.0, 0.0, t))
    corners = [(x, y) for x in (PEG_PITCH, L - PEG_PITCH) for y in (PEG_PITCH, H - PEG_PITCH)]
    bosses = [mf.CrossSection.circle(BOSS, SEGMENTS).translate(c) for c in corners]
    body = body + mf.CrossSection.batch_boolean(bosses, mf.OpType.Add).extrude(h) \
        .translate((0.0, 0.0, t))
    rh = 0.5 * PEG_HOLE
    centres = [((i + 0.5) * PEG_PITCH, (j + 0.5) * PEG_PITCH)
               for i in range(s["cols"]) for j in range(s["rows"])]
    holes = [mf.CrossSection.circle(rh, SEGMENTS).translate(c) for c in centres]
    cut = [mf.CrossSection.batch_boolean(holes, mf.OpType.Add).extrude(t + 2.0)
           .translate((0.0, 0.0, -1.0))]
    # the chamfer: a 45 degree cone from below the face to just inside the hole, so it crosses
    # the wall of the hole instead of ending on it
    over = 0.05
    cone = mf.Manifold.cylinder(1.0 + HOLE_CHAMFER + over, rh + HOLE_CHAMFER + 1.0, rh - over,
                                SEGMENTS).translate((0.0, 0.0, -1.0))
    cut += [cone.translate((x, y, 0.0)) for x, y in centres]
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
    higher) and centred in plan, symmetric about its middle plane and one closed volume. The board
    it is made for is `board` millimetres thick, the part reaches `reach` millimetres out of the
    board and a holder takes a tool of `size` millimetres across. A "board" is `pegboard_board` of
    `cols`, `rows`, `board` and `standoff`. Every number is clamped to what prints; an unknown
    kind is a ValueError.
    """
    if _kind(kind) == "board":
        return pegboard_board(cols, rows, board, standoff)
    g = _sizes(kind, width, reach, board, size)
    return _part(g).translate((g["dx"], g["dy"], 0.0))
