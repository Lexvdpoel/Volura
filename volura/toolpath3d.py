"""G-code for a 3D mill: roughing and finishing of the model, and indexed 4th-axis work.

No Qt in this file. The solids of the model (`volura/ops.py`) become the plain text a
hobby mill understands, so it can be tested on its own and reused by the dialog in
`volura/mill3d_dialog.py` and by the CNC Controller (`volura/laser_window.py`).

A **3D mill** does not follow drawn contours but the surface of the model itself. That
surface is read into a **height map**: the highest point of the model in every cell of a
grid, which is all a 3-axis machine with a vertical spindle can reach anyway. From the
height map the tool path follows with a **drop cutter**: the tool is slid downwards over
the map until its shape touches the model, which for every cell gives the highest Z the
**tip** of the tool may go to without cutting into the model. `tool_surface` is that map,
and every path here is written on top of it.

**Z = 0 is the top of the material**, exactly as for the 2.5D mill of
`volura/gcode_mill.py`, because a mill is zeroed on the material and not on some
theoretical plane of the model. The stock is the box around the model, its top 1 mm above
the highest point, so everything that cuts stands negative in the G-code and `safe_z` is
the height above the stock the cutter travels at.

**Roughing** takes the stock away level by level, from the top down to the bottom of the
model, and leaves a skin of `stock_to_leave` for the finishing pass; **finishing** runs
parallel lines over the surface at a small stepover and follows every bump in it. A flat
end mill cannot stand in a corner, so it stays as high as the highest model point inside
its own circle; a ball nose follows the shape and leaves a scallop between the lines.

The spindle is the word of the laser turned motor: **M3** starts it at the revolutions of
the `S` word, **G4 P2** gives it the time to come up to speed before the cutter touches
anything, and **M5** stops it at the end of the job.

The first step towards the axes a mill does not have is the **indexed 4th axis**: the A
axis of a 3+1 machine lies along X, so `indexed` turns the model about X to the angles it
should be milled from - 0 and 180 for a part that is milled on both sides - and writes
**G0 A** before the paths of every side. The machine then stops, the part is turned to
that angle, and the same program goes on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import __version__
from .gcode import _num

#: how far a point may stand off a straight line before it is kept in the G-code (mm)
COLLINEAR_TOL = 0.01

#: the height of the stock above the highest point of the model (mm)
STOCK_TOP_CLEARANCE = 1.0


# --------------------------------------------------------------------------
#  the tool and the stock
# --------------------------------------------------------------------------

@dataclass
class Tool:
    """The cutter: `kind` is "flat" (an end mill) or "ball" (a ball nose)."""

    kind: str = "flat"
    diameter: float = 6.0
    flute_length: float = 20.0

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


@dataclass
class Stock:
    """The block of material around the model; `z1` is the top the mill zeroes on."""

    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    x1: float = 100.0
    y1: float = 100.0
    z1: float = 0.0


def stock_around(vertices, margin=2.0) -> Stock:
    """The bounding box of the model, grown by `margin` in X and Y.

    The bottom of the stock is the bottom of the model - a part is milled flat on the
    waste board or on a spoil board - and the top is 1 mm above the highest point, so the
    cutter has room to travel and there is a little material left to face off.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.size == 0:
        return Stock()
    return Stock(x0=float(v[:, 0].min()) - margin,
                 y0=float(v[:, 1].min()) - margin,
                 z0=float(v[:, 2].min()),
                 x1=float(v[:, 0].max()) + margin,
                 y1=float(v[:, 1].max()) + margin,
                 z1=float(v[:, 2].max()) + STOCK_TOP_CLEARANCE)


# --------------------------------------------------------------------------
#  the height map
# --------------------------------------------------------------------------

def height_map(vertices, triangles, x0, y0, x1, y1, res):
    """The top surface Z of the model on a grid of `res` millimetre cells.

    Every triangle is rasterised into the grid and the highest Z in a cell wins, so an
    overhang cannot fool the tool: what hangs above is what the cutter meets first. A
    triangle that is seen exactly from the side is a wall and gives no top surface, and a
    cell without any model keeps the bottom of the model, because that is as deep as the
    cutter may go there.

    (xs, ys, Z) with Z shaped (len(ys), len(xs))
    """
    v = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    z0 = float(v[:, 2].min()) if len(v) else 0.0
    nx = int((x1 - x0) / res) + 1
    ny = int((y1 - y0) / res) + 1
    xs = x0 + np.arange(nx, dtype=np.float64) * res
    ys = y0 + np.arange(ny, dtype=np.float64) * res
    Z = np.full((ny, nx), z0, dtype=np.float64)
    if len(tri) == 0 or nx < 2 or ny < 2:
        return xs, ys, Z

    a = v[tri[:, 0]].tolist()
    b = v[tri[:, 1]].tolist()
    c = v[tri[:, 2]].tolist()
    for k in range(len(tri)):
        ax, ay, az = a[k]
        bx, by, bz = b[k]
        cx, cy, cz = c[k]
        det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(det) < 1e-12:
            continue                        # a wall seen from the side: no top here
        i0 = max(int(math.floor((min(ax, bx, cx) - x0) / res)), 0)
        i1 = min(int(math.ceil((max(ax, bx, cx) - x0) / res)), nx - 1)
        j0 = max(int(math.floor((min(ay, by, cy) - y0) / res)), 0)
        j1 = min(int(math.ceil((max(ay, by, cy) - y0) / res)), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        px = xs[i0:i1 + 1][None, :]
        py = ys[j0:j1 + 1][:, None]
        u = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / det
        w = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / det
        hit = (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
        if not hit.any():
            continue
        zc = az + u * (bz - az) + w * (cz - az)
        block = Z[j0:j1 + 1, i0:i1 + 1]
        np.maximum(block, np.where(hit, zc, -np.inf), out=block)
    return xs, ys, Z


# --------------------------------------------------------------------------
#  the drop cutter
# --------------------------------------------------------------------------

def tool_offsets(res, tool):
    """The grid offsets inside the footprint of the tool, with the profile height at r.

    A flat end mill has no profile: its tip stays level. A ball nose is a sphere, so at
    the distance r from the axis the ball already touches the model `R - sqrt(R^2 - r^2)`
    higher up, and the tip has to go that much higher to clear the model there.
    """
    r = tool.radius
    n = max(int(math.ceil(r / res)), 0)
    ball = tool.kind == "ball"
    offsets = []
    for dj in range(-n, n + 1):
        for di in range(-n, n + 1):
            d = math.hypot(di * res, dj * res)
            if d > r + 1e-9:
                continue
            offsets.append((di, dj, r - math.sqrt(max(r * r - d * d, 0.0)) if ball else 0.0))
    return offsets


def _max_filter(Z, offsets):
    """The maximum of Z over the given (di, dj, drop) offsets, one shifted maximum each.

    The whole footprint at once, per offset: a loop over the offsets of the tool and not
    over the cells of the map, so a 50 000 triangle model with a 0.5 mm grid stays a
    fraction of a second.
    """
    ny, nx = Z.shape
    n = max(max(abs(di) for di, _, _ in offsets), max(abs(dj) for _, dj, _ in offsets))
    pad = np.full((ny + 2 * n, nx + 2 * n), -np.inf, dtype=np.float64)
    pad[n:n + ny, n:n + nx] = Z
    out = np.full(Z.shape, -np.inf, dtype=np.float64)
    for di, dj, drop in offsets:
        np.maximum(out, pad[n + dj:n + dj + ny, n + di:n + di + nx] - drop, out=out)
    return out


def tool_surface(Z, res, tool):
    """The highest tool tip Z per cell that keeps the tool out of the model.

    The drop cutter on the height map: for every cell the maximum over the footprint of
    the tool of Z(there) minus the profile of the tool at that distance. The cell itself
    is always part of the footprint, so the surface never stands below the model.
    """
    return _max_filter(Z, tool_offsets(res, tool))


# --------------------------------------------------------------------------
#  the lines
# --------------------------------------------------------------------------

def _runs(mask):
    """The inclusive (start, stop) index ranges of the True runs in a 1D mask."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1),
                                                 np.flatnonzero(edges == -1))
            if b - a > 1]


def _line_indices(n, res, spacing):
    """The grid indices of the parallel lines, `spacing` millimetres apart over the grid.

    The lines are taken on the exact spacing and land on the nearest cell, so a stepover
    that does not divide the grid still gives evenly spaced cuts and not a drift that
    grows with every line.
    """
    spacing = max(float(spacing), res)
    count = int(math.floor((n - 1) * res / spacing + 1e-9))
    return [min(int(round(k * spacing / res)), n - 1) for k in range(count + 1)]


def roughing(Z, xs, ys, res, stock, tool, step_down=2.0, stepover=0.4,
             stock_to_leave=0.3):
    """The stock away, level by level: parallel zig-zag lines in X.

    The levels run from the top of the stock down to the bottom of the model in
    `step_down` steps, and the last one lands exactly on the bottom. At a level the tool
    may go down at a cell as soon as the tool surface plus `stock_to_leave` stands at or
    below that level: everything higher is still in the way and the tool would break
    itself against it. `stepover` is a fraction of the diameter, so a wide cutter makes
    wide lines.

    A line is split where the tool has to stop, and every piece comes out as its own
    polyline, in the order the mill visits them; `to_gcode` brings the cutter up to the
    safe height between the pieces.
    """
    surface = tool_surface(Z, res, tool)
    rows = _line_indices(len(ys), res, stepover * tool.diameter)
    levels = []
    z = stock.z1 - step_down
    while z > stock.z0 + 1e-9:
        levels.append(z)
        z -= step_down
    levels.append(stock.z0)
    paths = []
    for level in levels:
        allowed = surface + stock_to_leave <= level + 1e-9
        if not allowed.any():
            continue
        forward = True
        for j in rows:
            for i0, i1 in _runs(allowed[j]):
                cells = slice(i0, i1 + 1)
                line = np.stack([xs[cells], np.full(i1 - i0 + 1, ys[j]),
                                 np.full(i1 - i0 + 1, level)], axis=1)
                paths.append(np.ascontiguousarray(line[::-1] if not forward else line))
            forward = not forward
    return paths


def finishing(Z, xs, ys, res, tool, angle=0, stepover=None):
    """Parallel raster finishing: lines in X, or in Y at `angle` 90, over the tool surface.

    Every line follows the tool tip Z along its way, so a ball nose stays on the model
    also over a dome; the lines are cut zig-zag and split where the model runs out,
    because a line that goes on over the waste board burns nothing but time. `stepover`
    is in millimetres here and not a fraction of the diameter: it is the scallop that
    stands between two lines, and a tenth of the diameter is what a ball nose leaves.
    """
    if stepover is None:
        stepover = max(tool.diameter * 0.1, 0.3)
    surface = tool_surface(Z, res, tool)
    floor = float(Z.min())
    solid = surface > floor + 1e-9
    stepover = max(float(stepover), res)
    paths = []
    for n, line in enumerate(_line_indices(len(xs if angle == 90 else ys), res, stepover)):
        forward = n % 2 == 0
        if angle == 90:
            for j0, j1 in _runs(solid[:, line]):
                cells = slice(j0, j1 + 1)
                seg = np.stack([np.full(j1 - j0 + 1, xs[line]), ys[cells],
                                surface[cells, line]], axis=1)
                paths.append(np.ascontiguousarray(seg[::-1] if not forward else seg))
        else:
            for i0, i1 in _runs(solid[line]):
                cells = slice(i0, i1 + 1)
                seg = np.stack([xs[cells], np.full(i1 - i0 + 1, ys[line]),
                                surface[line, cells]], axis=1)
                paths.append(np.ascontiguousarray(seg[::-1] if not forward else seg))
    return paths


# --------------------------------------------------------------------------
#  the text for the machine
# --------------------------------------------------------------------------

def _drop_collinear(points):
    """The points that are not on a straight line, within COLLINEAR_TOL millimetres.

    A finishing line over a flat top is thousands of points that all say the same thing;
    the machine does not need them and the file becomes needlessly big with them in it.
    A point is only dropped against the last point that was kept, so a gentle curve keeps
    the points it needs even when every three of them look like a straight line.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    coords = pts.tolist()
    keep = [0]
    ax, ay, az = coords[0]
    for i in range(1, len(coords) - 1):
        bx, by, bz = coords[i]
        cx, cy, cz = coords[i + 1]
        ex, ey, ez = cx - ax, cy - ay, cz - az
        length = math.sqrt(ex * ex + ey * ey + ez * ez)
        if length <= 1e-9:
            continue
        ux, uy, uz = bx - ax, by - ay, bz - az
        gx, gy, gz = uy * ez - uz * ey, uz * ex - ux * ez, ux * ey - uy * ex
        if math.sqrt(gx * gx + gy * gy + gz * gz) / length > COLLINEAR_TOL:
            keep.append(i)
            ax, ay, az = coords[i]
    keep.append(len(coords) - 1)
    return pts[keep]


def to_gcode(paths, tool, feed, plunge, rpm, safe_z, stock, angles=None):
    """The toolpaths as G-code for a GRBL mill, with Z = 0 on the top of the stock.

    Between two polylines the cutter goes up to `safe_z`, travels and plunges at the
    plunge feed; along a polyline it cuts at `feed`. `angles` holds the A angle of every
    polyline for indexed 4th-axis work, and a `G0 A` line is written whenever that angle
    changes - the machine stops there and waits for the part to be turned.
    """
    lines = [f"; Volura {__version__}",
             f"; {tool.kind} tool {_num(tool.diameter)} mm",
             "G21", "G90", "G17", f"G0 Z{_num(safe_z)}",
             f"M3 S{int(round(rpm))}", "G4 P2"]
    last_angle = None
    for n, path in enumerate(paths):
        pts = _drop_collinear(path)
        if len(pts) < 2:
            continue
        if angles is not None:
            a = float(angles[n])
            if a != last_angle and not (a == 0.0 and last_angle is None):
                lines.append(f"G0 A{_num(a)}")
            last_angle = a
        lines.append(f"G0 Z{_num(safe_z)}")
        lines.append(f"G0 X{_num(pts[0][0])} Y{_num(pts[0][1])}")
        lines.append(f"G1 Z{_num(pts[0][2] - stock.z1)} F{plunge:g}")
        for p in pts[1:]:
            lines.append(f"G1 X{_num(p[0])} Y{_num(p[1])} Z{_num(p[2] - stock.z1)} "
                         f"F{feed:g}")
    lines += [f"G0 Z{_num(safe_z)}", "M5", "G0 X0 Y0"]
    return "\n".join(lines) + "\n"


def estimate(paths, feed, plunge, rapid):
    """What the mill takes for the toolpaths: the millimetres and the seconds.

    A cut goes at `feed`, a plunge - a move that changes nothing but Z - at `plunge`, and
    the travel between two polylines at `rapid`, all in millimetres per minute, so the
    length of a move divided by its speed is the fraction of a minute it takes.

    {"cut_mm", "travel_mm", "seconds"}
    """
    cut = travel = seconds = 0.0
    previous = None
    for path in paths:
        pts = np.asarray(path, dtype=np.float64)
        if len(pts) < 2:
            continue
        if previous is not None:
            move = float(math.hypot(pts[0, 0] - previous[0], pts[0, 1] - previous[1]))
            travel += move
            seconds += move / float(rapid) * 60.0 if rapid > 0 else 0.0
        previous = pts[-1]
        step = np.diff(pts, axis=0)
        seg = np.sqrt((step ** 2).sum(axis=1))
        cut += float(seg.sum())
        speed = np.where(np.abs(step[:, 0]) + np.abs(step[:, 1]) < 1e-9, plunge, feed)
        seconds += 60.0 * float(np.sum(np.divide(seg, speed, out=np.zeros_like(seg),
                                                where=speed > 0)))
    return {"cut_mm": cut, "travel_mm": travel, "seconds": seconds}


# --------------------------------------------------------------------------
#  the indexed 4th axis
# --------------------------------------------------------------------------

def rotate_about_x(vertices, degrees):
    """The model turned about the X axis: the A axis of a 3+1 mill, through its middle.

    The axis runs along X at the height of the middle of the model, so a part that is
    milled on both sides stays in the same box between the two sides and the stock does
    not have to be re-measured.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) == 0:
        return v.copy()
    cy = float(v[:, 1].min() + v[:, 1].max()) / 2.0
    cz = float(v[:, 2].min() + v[:, 2].max()) / 2.0
    cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    y, z = v[:, 1] - cy, v[:, 2] - cz
    out = v.copy()
    out[:, 1] = cy + y * cos - z * sin
    out[:, 2] = cz + y * sin + z * cos
    return out


def indexed(vertices, triangles, angles, rough_tool=None, finish_tool=None, res=0.5,
            step_down=2.0, stepover=0.4, stock_to_leave=0.3, finish_stepover=None,
            finish_angle=0, margin=2.0, report=None):
    """One side per A angle, each milled from the top: the first step towards multi-axis.

    The model is turned about the X axis to every angle in `angles` - 0 and 180 for a
    part that is milled on both sides - and roughed and finished from above there. The
    stock is the box around the model in **all** of its positions, so Z = 0 stays the
    same top for every side and the operator only zeroes once.

    `report(fraction)` may be given to follow the sides on a progress bar.

    (paths, angles, stock, kinds) with one A angle and one kind per polyline
    """
    v = np.asarray(vertices, dtype=np.float64)
    sides = [(float(a), rotate_about_x(v, a)) for a in angles]
    stock = stock_around(np.concatenate([s for _, s in sides]), margin) if sides else Stock()
    paths: list = []
    path_angles: list = []
    kinds: list = []
    for n, (a, rotated) in enumerate(sides):
        xs, ys, Z = height_map(rotated, triangles, stock.x0, stock.y0,
                               stock.x1, stock.y1, res)
        for kind, tool in (("rough", rough_tool), ("finish", finish_tool)):
            if tool is None:
                continue
            if kind == "rough":
                side = roughing(Z, xs, ys, res, stock, tool, step_down, stepover,
                                stock_to_leave)
            else:
                side = finishing(Z, xs, ys, res, tool, finish_angle, finish_stepover)
            paths += side
            path_angles += [a] * len(side)
            kinds += [kind] * len(side)
        if report is not None:
            report((n + 1) / len(sides))
    return paths, path_angles, stock, kinds
