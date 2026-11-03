"""2D vector export from the 3D model: selected faces, top view and section.

The logic here is deliberately free of Qt: drawings (`volura.vector2d.Drawing`)
are taken from the document -- the selected faces at true size, the top view of
the solids and a section at height Z -- and passed to `volura.svg` or
`volura.dxf`. The interface in `volura.app` builds on this.

All sizes are millimetres with Y up. The top view and the section keep the x and y
of the model, so the drawing stays aligned with the model origin; selected faces
get the local frame of their own plane, and faces that do not lie in the same
plane are placed side by side with `GAP` between them.
"""
from __future__ import annotations

import numpy as np
import manifold3d as mf

from . import ops
from .geom import default_u, normalize
from .i18n import tr
from .vector2d import Drawing, VPath, regions

GAP = 5.0            # mm between shapes that do not lie in the same plane
CLEAN_TOL = 1e-6     # mm: points closer together than this merge
LAYER = "0"          # everything goes on layer "0"


# --------------------------------------------------------------------------
#  cleaning up 2D points
# --------------------------------------------------------------------------

def _dist_to_line(p, a, b):
    """Distance from `p` to the line through `a` and `b`."""
    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L <= 1e-12:
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / L


def clean_loop(pts, tol=CLEAN_TOL):
    """Clean up a closed polyline: remove duplicate points and points on a straight line.

    `Manifold.slice` and `Manifold.project` often return a corner twice (once per
    adjacent triangle), and a point in the middle of a straight line is redundant
    too; such points are dropped. A point is dropped when it lies closer than `tol`
    to the line of its neighbours, or closer than `tol` to the previous point. At
    least three points remain.
    """
    p = [np.asarray(q, dtype=float) for q in np.asarray(pts, dtype=float).reshape(-1, 2)]
    changed = True
    while changed and len(p) > 3:
        changed = False
        keep = []
        for i, cur in enumerate(p):
            prev = p[i - 1]
            nxt = p[(i + 1) % len(p)]
            if float(np.hypot(cur[0] - prev[0], cur[1] - prev[1])) <= tol:
                changed = True
                continue
            if _dist_to_line(cur, prev, nxt) <= tol:
                changed = True
                continue
            keep.append(cur)
        p = keep
    return np.ascontiguousarray(np.asarray(p, dtype=float).reshape(-1, 2))


# --------------------------------------------------------------------------
#  selected faces
# --------------------------------------------------------------------------

def plane_frame(n):
    """Right handed local frame `(n, u, v)` that depends only on the normal.

    Two coplanar faces thus get the same axes and keep their relative position. A
    normal pointing backwards is flipped (the same key as `brep.plane_key`), so a
    face is always viewed from the front: a flat drawing on the ground keeps its
    top side up.
    """
    n = normalize(np.asarray(n, dtype=float))
    for c in n:
        if abs(c) > 1e-9:
            if c < 0.0:
                n = -n
            break
    u = np.asarray(default_u(n), dtype=float)
    u = u - n * float(u @ n)
    if float(np.linalg.norm(u)) < 1e-9:
        u = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    u = normalize(u)
    return n, u, np.cross(n, u)


def _plane_key(n, origin):
    """Key of the plane: the same normal and the same distance to the origin."""
    return tuple(round(float(c), 6) for c in n) + (round(float(np.dot(n, origin)), 4),)


def _face_loops(ents, fid):
    """(normal, origin, loops) of a face, the loops in its own 2D frame."""
    face = ents.F[fid]
    n, u, v = plane_frame(face.normal)
    origin = n * float(np.dot(n, ents.V[face.loops[0][0]]))
    M = np.stack([u, v])
    loops = []
    for loop in face.loops:
        pts = np.array([ents.V[v_] for v_ in loop], dtype=float)
        q = clean_loop((pts - origin) @ M.T)
        if len(q) >= 3:
            loops.append(q)
    return n, origin, loops


def _shapes(loops, rule="nonzero"):
    """Loops as filled shapes [outer boundary, holes], via `vector2d.regions`."""
    paths = [VPath(np.ascontiguousarray(l, float), True, LAYER, True, rule, 1)
             for l in loops if len(l) >= 3]
    return regions(paths) if paths else []


def _bbox(loops):
    """(xmin, ymin, xmax, ymax) of a list of loops, or None if there are none."""
    pts = [np.asarray(l, dtype=float).reshape(-1, 2) for l in loops if len(l)]
    if not pts:
        return None
    P = np.concatenate(pts)
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max()), float(P[:, 1].max()))


def _layout(bboxes, gap=GAP):
    """Offset per group: the first stays in place, the rest move below it.

    The first group sets the position of the drawing; the other groups go in rows
    below it, `gap` mm apart, side by side in turn.
    """
    if len(bboxes) < 2 or any(b is None for b in bboxes):
        return [(0.0, 0.0) for _ in bboxes]
    shifts = [(0.0, 0.0)]
    left, bottom, right, _top = bboxes[0]
    wrap = max(right - left, max(b[2] - b[0] for b in bboxes[1:]))
    x, y = right + gap, bottom
    for b in bboxes[1:]:
        w, h = b[2] - b[0], b[3] - b[1]
        if x + w > left + wrap:                       # new row, lower down
            x, y = left, bottom - gap - h
            bottom = y
        shifts.append((x - b[0], y - b[1]))
        x += w + gap
    return shifts


def drawing_from_faces(doc, selection=None, gap=GAP) -> Drawing:
    """Every selected face at true size in its own plane.

    `selection` is a list of selection items as in `Document.selection`,
    `("f", ctx, fid)`; without an argument the active selection is used. Faces in
    the same plane keep their relative position, faces in different planes are
    placed side by side with `gap` mm between them. A face with holes gives one
    shape: the outer boundary and the holes belong to one source element (`src`).
    """
    items = doc.selection if selection is None else list(selection)
    order, by_plane = [], {}
    seen = set()
    for item in items:
        try:
            kind, ctx, fid = item
        except (TypeError, ValueError):
            continue
        if kind != "f" or (ctx, fid) in seen:
            continue
        ents = doc.ents(ctx)
        if ents is None or fid not in ents.F:
            continue
        seen.add((ctx, fid))
        n, origin, loops = _face_loops(ents, fid)
        if not loops:
            continue
        key = _plane_key(n, origin)
        if key not in by_plane:
            by_plane[key] = []
            order.append(key)
        by_plane[key].extend(loops)

    drawing = Drawing()
    groups = [g for g in (_shapes(by_plane[k]) for k in order) if g]
    boxes = [_bbox([l for shape in g for l in shape]) for g in groups]
    src = 0
    for group, (dx, dy) in zip(groups, _layout(boxes, gap)):
        for shape in group:
            src += 1
            for loop in shape:
                drawing.add_path(np.asarray(loop, dtype=float) + np.array([dx, dy]),
                                 closed=True, layer=LAYER, filled=True, rule="nonzero", src=src)
    return drawing


# --------------------------------------------------------------------------
#  top view and section
# --------------------------------------------------------------------------

def _sections(doc, only_selected, what, z=0.0):
    """(sections per part, warnings) for `project` or `slice`."""
    out, warn = [], []
    for part in ops.parts(doc, only_selected=bool(only_selected)):
        if part.manifold is None:
            warn.append(tr("{name} skipped: {problem}.").format(name=part.name, problem=part.problem))
            continue
        try:
            cs = part.manifold.project() if what == "top" else part.manifold.slice(float(z))
        except Exception as e:                          # a broken solid must never crash
            warn.append(tr("{name} skipped: {problem}.").format(name=part.name, problem=e))
            continue
        if cs.is_empty():
            continue
        polys = [clean_loop(q) for q in cs.to_polygons()]
        polys = [np.ascontiguousarray(q, dtype=float) for q in polys if len(q) >= 3]
        if not polys:
            continue
        if what == "top":
            # a projection is often self-intersecting; only the positive fill rule
            # turns it into a normal shape (as Manifold.project recommends)
            cs = mf.CrossSection(polys, mf.FillRule.NonZero)
            if cs.is_empty():
                continue
        out.append(cs)
    return out, warn


def _drawing_from_sections(sections, warn) -> Drawing:
    """The union of the sections as closed paths on layer "0"."""
    drawing = Drawing()
    for text in warn:
        drawing.warn(text)
    if not sections:
        return drawing
    whole = sections[0] if len(sections) == 1 else mf.CrossSection.batch_boolean(
        sections, mf.OpType.Add)
    if whole.is_empty():
        return drawing
    polys = [clean_loop(q) for q in whole.to_polygons()]
    for i, shape in enumerate(_shapes(polys, "evenodd"), 1):
        for loop in shape:
            drawing.add_path(np.asarray(loop, dtype=float), closed=True, layer=LAYER,
                             filled=True, rule="nonzero", src=i)
    return drawing


def drawing_top(doc, only_selected=False) -> Drawing:
    """Top view (outline) of the solids: `Manifold.project()` per part."""
    sections, warn = _sections(doc, only_selected, "top")
    return _drawing_from_sections(sections, warn)


def drawing_section(doc, z=0.0, only_selected=False) -> Drawing:
    """Section at height `z`: `Manifold.slice(z)` per part."""
    sections, warn = _sections(doc, only_selected, "section", z)
    return _drawing_from_sections(sections, warn)


def section_z_default(doc):
    """The vertical middle of the model: the default height for a section."""
    box = doc.model_bbox()
    if box is None:
        return 0.0
    return 0.5 * (float(box[0][2]) + float(box[1][2]))
