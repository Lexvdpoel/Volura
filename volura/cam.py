"""The computational core for laser and mill: nesting depth, kerf/cutter compensation
and dogbones.

No Qt in this file. `nesting_depth`, `compensate` and `dogbones` work on the shared 2D
model from `volura/vector2d.py` and can therefore be tested on their own and reused
elsewhere (the layers panel, "Export for laser/mill" and nesting).

A laser cuts in the middle of the drawn line and removes half the kerf on both sides;
a mill has its own radius. The contour must therefore be moved sideways to give the
drawn size after cutting. Which way is told by the nesting depth: the outer boundary of
a part (depth 0) outwards, a hole (depth 1) the other way, an island in that hole
(depth 2) outwards again.

A round cutter cannot reach into a sharp inside corner. That needs a relief: a circle
with the cutter diameter through the corner point, along the bisector ("dogbone") or
along the longer of the two edges ("T-bone").
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import manifold3d as mf

from .i18n import tr
from .vector2d import (Drawing, VPath, _point_in_poly, arc_points, layer_param,
                       polygon_area, regions)

MAX_ANGLE = 179.0        # the mill leaves corners sharper than this standing
CURVE_TOL = 0.002        # deviation of the circle of a dogbone, in mm
MAX_RINGS = 4000         # a pocket area never goes on forever


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------

def _is_shape(path) -> bool:
    """Only a closed path with at least three points is a shape."""
    return bool(getattr(path, "closed", False)) and len(getattr(path, "pts", ())) >= 3


def _polygons(cs: mf.CrossSection) -> list[np.ndarray]:
    """The loops of a cross-section, holes and all, as arrays."""
    return [np.asarray(q, dtype=float) for q in cs.to_polygons() if len(q) >= 3]


def _section(loops) -> mf.CrossSection:
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def _copy(path: VPath) -> VPath:
    return VPath(np.asarray(path.pts, dtype=float).copy(), path.closed, path.layer,
                 path.filled, path.rule, path.src, dict(path.meta))


def _new(path: VPath, pts) -> VPath:
    """A new closed path on the layer of `path`; the meta of a circle is no longer right."""
    return VPath(np.asarray(pts, dtype=float), True, path.layer, path.filled,
                 path.rule, path.src, {})


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else np.zeros(2)


def _edge_distance(poly, q) -> float:
    """Smallest distance from point `q` to the edge of the closed polyline `poly`."""
    p = np.asarray(poly, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
    t = np.clip(np.sum((q - a) * d, axis=1) / L2, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - q, axis=1)))


def _circle(center, r, tol=CURVE_TOL) -> np.ndarray:
    """A circle as a closed polyline (the last point is dropped)."""
    pts = arc_points(np.asarray(center, dtype=float), abs(float(r)),
                     0.0, 2.0 * math.pi, tol)
    return np.ascontiguousarray(np.asarray(pts, dtype=float)[:-1])


# --------------------------------------------------------------------------
#  nesting depth
# --------------------------------------------------------------------------

def _interior_point(poly) -> np.ndarray:
    """A point just inside the loop: at a convex corner, close to the edge.

    The mean of the points is not a good representative point: for a plate with a
    hole the middle of the outer boundary lies right in the hole. The ear of a corner
    lies against the edge and so outside the contours that lie inside it.
    """
    p = np.asarray(poly, dtype=float)
    n = len(p)
    sgn = 1.0 if polygon_area(p) >= 0.0 else -1.0
    for i in range(n):
        a, v, b = p[i - 1], p[i], p[(i + 1) % n]
        d1, d2 = a - v, b - v
        L1, L2 = float(np.hypot(*d1)), float(np.hypot(*d2))
        if L1 <= 1e-12 or L2 <= 1e-12:
            continue
        if sgn * (d1[0] * d2[1] - d1[1] * d2[0]) >= 0.0:
            continue                        # concave corner: the bisector points outwards
        bis = _unit(d1) + _unit(d2)
        u = _unit(bis)
        if float(np.hypot(*u)) < 1e-9:
            continue
        step = 0.5 * min(L1, L2)
        for factor in (0.05, 0.15, 0.35):
            q = v + u * (step * factor)
            if _point_in_poly(p, q) and _edge_distance(p, q) > 1e-9:
                return q
    return p.mean(axis=0)


def nesting_depth(paths) -> list[int]:
    """How deep each closed path lies within the others: 0 outer boundary, 1 hole,
    2 island in a hole.

    The list matches the given paths; an open path or a path that is not a shape gets
    0. Paths of different layers do not count for each other, so whoever uses this for
    a whole drawing gets the depth within each layer by passing the paths per layer.
    """
    paths = list(paths.paths if isinstance(paths, Drawing) else paths)
    loops = [np.asarray(p.pts, dtype=float) for p in paths if _is_shape(p)]
    spots = [_interior_point(q) for q in loops]
    depths: list[int] = []
    for i, spot in enumerate(spots):
        d = 0
        for j, other in enumerate(loops):
            if i != j and _point_in_poly(other, spot):
                d += 1
        depths.append(d)
    out: list[int] = []
    k = 0
    for p in paths:
        if _is_shape(p):
            out.append(depths[k])
            k += 1
        else:
            out.append(0)
    return out


# --------------------------------------------------------------------------
#  offsetting
# --------------------------------------------------------------------------

def offset_loop(pts, delta, join=None) -> list[np.ndarray]:
    """Move a closed path `delta` millimetres sideways (positive outwards).

    Rounded corners (`mf.JoinType.Round`), because a laser and a mill turn.
    A hole that closes up or a contour that is too small for the cutter gives nothing.
    """
    delta = float(delta)
    if abs(delta) < 1e-12:
        return [np.asarray(pts, dtype=float).copy()]
    try:
        cs = _section([np.asarray(pts, dtype=float)]).offset(
            delta, join or mf.JoinType.Round, 2.0)
    except Exception:                       # manifold3d sometimes rejects a jump
        return []
    return [] if cs.is_empty() else _polygons(cs)


def region_of(paths) -> mf.CrossSection:
    """The area enclosed by the closed paths of a layer (holes removed)."""
    loops = [loop for shape in regions(paths) for loop in shape]
    return _section(loops) if loops else mf.CrossSection()


def pocket_paths(paths, tool_d, stepover=50.0) -> list[np.ndarray]:
    """The paths of a pocket: first the edge moved inwards by the cutter radius,
    then concentrically smaller by the stepover until the area is used up."""
    tool_d = abs(float(tool_d))
    if tool_d <= 0.0:
        return []
    region = region_of(paths)
    if region.is_empty():
        return []
    step = max(tool_d * max(1.0, min(100.0, float(stepover))) / 100.0, tool_d * 0.02)
    out: list[np.ndarray] = []
    reach = _reach(region)
    delta = 0.5 * tool_d
    for _ in range(MAX_RINGS):
        try:
            cs = region.offset(-delta, mf.JoinType.Round, 2.0)
        except Exception:
            break
        if cs.is_empty():
            break
        out.extend(_polygons(cs))
        delta += step
        if delta > reach + step:
            break
    return out


def _reach(cs: mf.CrossSection) -> float:
    """How far a path can go inwards at most: half the diagonal of the extent."""
    loops = _polygons(cs)
    if not loops:
        return 0.0
    pts = np.concatenate(loops, axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    return float(np.hypot(*span))


# --------------------------------------------------------------------------
#  dogbones
# --------------------------------------------------------------------------

def corner_spots(pts, tool_d, style="dogbone", max_angle=MAX_ANGLE) -> list[np.ndarray]:
    """The centres of the relief circles at the inside corners of a closed path.

    A corner is an inside corner when the area inside the path (the area to be milled)
    is sharper than `max_angle` degrees there: the cutter cannot reach it. The circle
    has the cutter diameter and passes through the corner point, so its centre is at a
    distance of the cutter radius: along the bisector ("dogbone") or along the longer
    of the two edges ("T-bone").

    The small polygons with which a circle or arc was flattened are not corners: the
    cutter rolls along there. A corner therefore only counts when the relief is coarse
    compared with the edges and the cutter radius itself.
    """
    r = abs(float(tool_d)) / 2.0
    p = np.asarray(pts, dtype=float)
    if r <= 0.0 or len(p) < 3:
        return []
    if polygon_area(p) < 0.0:               # counter-clockwise: inside is left of the edge
        p = p[::-1].copy()
    n = len(p)
    out: list[np.ndarray] = []
    for i in range(n):
        v = p[i]
        d1, d2 = p[i - 1] - v, p[(i + 1) % n] - v
        L1, L2 = float(np.hypot(*d1)), float(np.hypot(*d2))
        if L1 <= 1e-9 or L2 <= 1e-9:
            continue
        if d1[0] * d2[1] - d1[1] * d2[0] >= 0.0:
            continue                        # convex corner for the tool: it reaches that
        hoek = math.degrees(math.acos(max(-1.0, min(1.0, float(d1 @ d2) / (L1 * L2)))))
        if hoek >= float(max_angle):
            continue
        if 0.5 * float(np.hypot(*(d1 + d2))) < max(0.05 * min(L1, L2), 0.02 * r):
            continue                        # flattened curve, not a real corner
        if str(style).lower() == "tbone":
            weg = _unit(d1) if L1 >= L2 else _unit(d2)
        else:
            weg = _unit(_unit(d1) + _unit(d2))
        if float(np.hypot(*weg)) < 1e-9:
            continue
        out.append(v + weg * r)
    return out


def dogbone_loops(pts, tool_d, style="dogbone") -> list[np.ndarray]:
    """Unite a closed path with the relief circles; one closed path again."""
    spots = corner_spots(pts, tool_d, style)
    p = np.asarray(pts, dtype=float)
    if not spots:
        return [p.copy()]
    shapes = [_section([p])] + [_section([_circle(c, abs(float(tool_d)) / 2.0)])
                                for c in spots]
    try:
        whole = mf.CrossSection.batch_boolean(shapes, mf.OpType.Add)
    except Exception:
        return [p.copy()]
    loops = _polygons(whole)
    return loops or [p.copy()]


def dogbones(paths, tool_d, style="dogbone") -> list[VPath]:
    """Add a relief at every inside corner of the closed paths.

    The result is again a list of closed paths on the same layer, ready to mill;
    open paths stay as they are.
    """
    paths = list(paths.paths if isinstance(paths, Drawing) else paths)
    out: list[VPath] = []
    for p in paths:
        if not _is_shape(p):
            out.append(_copy(p))
            continue
        for loop in dogbone_loops(p.pts, tool_d, style):
            out.append(_new(p, loop))
    return out


# --------------------------------------------------------------------------
#  the compensation of a whole drawing
# --------------------------------------------------------------------------

def compensate_path(layer, path, depth=0) -> list[VPath]:
    """One path, offset as the operation of its layer requires."""
    op = getattr(layer, "op", "cut")
    if not _is_shape(path) or op in ("engrave", "raster", "drill", "none"):
        return [_copy(path)]
    if op == "cut":
        kerf = float(layer_param(layer, "kerf") or 0.0)
        if kerf <= 0.0:
            return [_copy(path)]
        delta = 0.5 * kerf * (1.0 if int(depth) % 2 == 0 else -1.0)
        loops = offset_loop(path.pts, delta)
    elif op == "mill_outside":
        loops = offset_loop(path.pts, float(layer_param(layer, "tool_d")) / 2.0)
    elif op == "mill_inside":
        tool_d = float(layer_param(layer, "tool_d"))
        loops = offset_loop(path.pts, -0.5 * tool_d)
        loops = _corners_of(loops, tool_d, layer_param(layer, "corner"))
    else:
        return [_copy(path)]
    return [_new(path, loop) for loop in loops]


def compensate_pocket(layer, paths) -> list[tuple[VPath, list[VPath]]]:
    """The concentric paths of a pocket, attached to the first contour of the layer."""
    shapes = [p for p in paths if _is_shape(p)]
    if not shapes:
        return [(p, [_copy(p)]) for p in paths]
    tool_d = float(layer_param(layer, "tool_d"))
    loops = pocket_paths(shapes, tool_d, layer_param(layer, "stepover"))
    loops = _corners_of(loops, tool_d, layer_param(layer, "corner"))
    made = [_new(shapes[0], loop) for loop in loops]
    out: list[tuple[VPath, list[VPath]]] = []
    for p in paths:
        if not _is_shape(p):
            out.append((p, [_copy(p)]))
        else:
            out.append((p, made if p is shapes[0] else []))
    return out


def _corners_of(loops, tool_d, style) -> list[np.ndarray]:
    """Dogbones on the inside corners of the loops, when the layer asks for them."""
    if not style or str(style) == "none":
        return list(loops)
    out: list[np.ndarray] = []
    for loop in loops:
        out.extend(dogbone_loops(loop, tool_d, style))
    return out


def compensate(drawing) -> Drawing:
    """A new drawing with the contours as the machine must follow them.

    Per layer, with the settings of that layer: "cut" with half the kerf outwards for
    the outer boundaries and inwards for the holes (the hole gets smaller, so it has the
    drawn size after cutting); "mill_outside" and "mill_inside" with the cutter radius
    outwards or inwards; "pocket" with the edge moved inwards by the cutter radius and
    then concentric paths with the stepover until the area is empty. "engrave",
    "raster", "drill" and open lines stay as they are.
    """
    drawing = drawing if isinstance(drawing, Drawing) else Drawing(paths=list(drawing))
    repl: dict[int, list[VPath]] = {}
    tight: set[int] = set()
    for name, layer in drawing.layers.items():
        mine = [p for p in drawing.paths if p.layer == name]
        if layer.op == "pocket":
            for original, made in compensate_pocket(layer, mine):
                repl[id(original)] = made
            continue
        depths = nesting_depth(mine)
        for p, depth in zip(mine, depths):
            made = compensate_path(layer, p, depth)
            repl[id(p)] = made
            if not made and _is_shape(p):
                tight.add(id(p))
    out = Drawing(layers={n: replace(l, params=dict(l.params))
                          for n, l in drawing.layers.items()},
                  warnings=list(drawing.warnings), **drawing._sheet_kwargs())
    skipped = 0
    for p in drawing.paths:
        made = repl.get(id(p))
        if made is None:
            out.paths.append(_copy(p))
            continue
        if id(p) in tight:
            skipped += 1
        out.paths.extend(made)
    if skipped:
        out.warn(tr("{n} contours are too small for the chosen cutter or kerf and "
                    "were skipped.").format(n=skipped))
    return out
