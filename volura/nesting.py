"""Automatic nesting: arrange parts as economically as possible on one or more sheets.

No Qt in this file. It works on the shared 2D model from `volura/vector2d.py` and the
nesting depth from `volura/cam.py`, and can therefore be tested on its own and reused
in the worksheet, the export and the milling paths.

A *part* (`Part`) is an outer boundary with everything that lies inside it (holes,
islands, engravings and open paths), so it moves across the sheet as one whole.
Loose open paths without an outer boundary form a part of their own on their bounding
box.

`nest` arranges the parts with "bottom-left fill": largest first, and per part and
rotation the lowest-then-leftmost place where it, inflated by half the gap, touches no
other part and falls within the sheet minus the margin. The sheet is always the same
size at a rotation of 0, 90, 180 or 270 degrees, so the geometry stays within the
sheet; the rotation happens around the part's own corner.

A placement is returned as `(part, sheet_index, angle, dx, dy)`: rotate the part by
`angle` degrees and put the bottom-left corner of its extent at `(dx, dy)` within that
sheet. `place_transform` turns it into the 3x3 matrix that carries all paths of the
part along; add the origin of the sheet (`sheet_origins`) to land on the worksheet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from . import cam
from .i18n import tr
from .vector2d import (SHEET_SPACING, Drawing, VPath, _point_in_poly, polygon_area)

EPS = 1e-9            # touching boxes do not count as overlap
AREA_EPS = 1e-6       # an intersection smaller than this (mm²) is not an overlap


# --------------------------------------------------------------------------
#  sheet geometry
# --------------------------------------------------------------------------

def sheet_origins(drawing) -> list[tuple[float, float]]:
    """The bottom-left corner of every sheet: sheet i is at (i * (width + 50), 0)."""
    w, h = drawing.sheet_w, drawing.sheet_h
    if not w or not h:
        return []
    n = max(1, int(drawing.sheet_count or 1))
    return [(i * (float(w) + SHEET_SPACING), 0.0) for i in range(n)]


def sheet_rects(drawing) -> list[tuple[float, float, float, float]]:
    """The full sheet area (x0, y0, x1, y1) per sheet."""
    w, h = float(drawing.sheet_w or 0.0), float(drawing.sheet_h or 0.0)
    return [(ox, oy, ox + w, oy + h) for ox, oy in sheet_origins(drawing)]


def usable_rects(drawing) -> list[tuple[float, float, float, float]]:
    """The usable area of every sheet: the sheet minus the margin along the edge."""
    m = float(drawing.sheet_margin or 0.0)
    return [(x0 + m, y0 + m, x1 - m, y1 - m) for x0, y0, x1, y1 in sheet_rects(drawing)]


def _points_in_rect(pts, rect, tol=EPS) -> bool:
    x0, y0, x1, y1 = rect
    p = np.asarray(pts, dtype=float)
    if not len(p):
        return True
    return (p[:, 0] >= x0 - tol).all() and (p[:, 0] <= x1 + tol).all() and \
           (p[:, 1] >= y0 - tol).all() and (p[:, 1] <= y1 + tol).all()


def outside_paths(drawing, paths=None) -> list:
    """Paths that lie (partly) outside every sheet or in the margin; empty without a sheet."""
    rects = usable_rects(drawing)
    if not rects:
        return []
    paths = list(paths if paths is not None else drawing.paths)
    return [p for p in paths if len(p.pts) and
            not any(_points_in_rect(p.pts, r) for r in rects)]


# --------------------------------------------------------------------------
#  parts
# --------------------------------------------------------------------------

def _bbox_of(paths):
    pts = [p.pts for p in paths if len(p.pts)]
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    allp = np.concatenate(pts, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _bbox_area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


@dataclass
class Part:
    """An outer boundary with everything inside it, or a bunch of loose open paths on their box."""

    paths: list = field(default_factory=list)
    outline: np.ndarray | None = None      # outer boundary (counter-clockwise) or None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    area: float = 0.0

    @property
    def shape(self) -> np.ndarray:
        """The outline with which the part is laid on the sheet."""
        if self.outline is not None:
            return np.asarray(self.outline, dtype=float)
        x0, y0, x1, y1 = self.bbox
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def part_shape(part: Part) -> np.ndarray:
    return part.shape


def part_area(part: Part) -> float:
    return float(part.area)


def parts_of(drawing, paths=None) -> list[Part]:
    """Group paths into parts: an outer boundary (depth 0) with everything inside it.

    Holes, islands, engravings and open paths that lie inside an outer boundary go
    along with that boundary. Loose open paths without an outer boundary form, together
    with the paths around the same point, a part of their own on their bounding box.
    """
    paths = list(paths if paths is not None else drawing.paths)
    closed = [p for p in paths if cam._is_shape(p)]
    others = [p for p in paths if not cam._is_shape(p) and len(p.pts)]
    depths = cam.nesting_depth(closed) if closed else []

    parts: list[Part] = []
    outer_index: list[tuple[np.ndarray, Part]] = []
    for p, d in zip(closed, depths):
        if d != 0:
            continue
        poly = np.asarray(p.pts, dtype=float)
        if polygon_area(poly) < 0:
            poly = poly[::-1].copy()
        part = Part(paths=[p], outline=poly)
        parts.append(part)
        outer_index.append((poly, part))

    for p, d in zip(closed, depths):
        if d == 0:
            continue
        spot = cam._interior_point(np.asarray(p.pts, dtype=float))
        owner = _owner(spot, outer_index)
        if owner is None:                       # deeper than 0 but without an outer boundary
            poly = np.asarray(p.pts, dtype=float)
            parts.append(Part(paths=[p], outline=poly))
        else:
            owner.paths.append(p)

    for p in others:
        spot = np.asarray(p.pts, dtype=float).mean(axis=0)
        owner = _owner(spot, outer_index)
        if owner is None:
            parts.append(Part(paths=[p]))       # loose open path: a part of its own on its box
        else:
            owner.paths.append(p)

    for part in parts:
        part.bbox = _bbox_of(part.paths)
        if part.outline is not None:
            part.area = abs(polygon_area(part.outline))
        else:
            part.area = _bbox_area(part.bbox)
    return [p for p in parts if p.paths]


def _owner(spot, outer_index) -> Part | None:
    for poly, part in outer_index:
        if _point_in_poly(poly, spot):
            return part
    return None


# --------------------------------------------------------------------------
#  transformation of a placement
# --------------------------------------------------------------------------

def _rot(angle_deg) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def _norm_rot(shape, angle_deg):
    """Rotate the shape and move it so its extent starts at the origin."""
    shape = np.asarray(shape, dtype=float)
    bmin = shape.min(axis=0)
    R = _rot(angle_deg)
    rot = (shape - bmin) @ R.T
    rmin = rot.min(axis=0)
    return rot - rmin, R, bmin, rmin


def place_transform(part: Part, angle, dx, dy, origin=(0.0, 0.0)) -> np.ndarray:
    """3x3 matrix that puts the part at `(dx, dy)` on a sheet (with rotation)."""
    norm, R, bmin, rmin = _norm_rot(part_shape(part), angle)
    M = np.eye(3, dtype=float)
    M[:2, :2] = R
    off = np.array([float(dx), float(dy)], dtype=float) - rmin - (bmin @ R.T)
    M[:2, 2] = off + np.asarray(origin, dtype=float)
    return M


# --------------------------------------------------------------------------
#  nesting
# --------------------------------------------------------------------------

def _section(loops) -> mf.CrossSection:
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def _inflated(norm_shape, gap) -> mf.CrossSection:
    """The keep-out space around a part: its shape grown by half the gap."""
    cs = _section([np.asarray(norm_shape, dtype=float)])
    if gap <= 0:
        return cs
    try:
        off = cs.offset(gap / 2.0, mf.JoinType.Round, 2.0)
    except Exception:
        return cs
    return cs if off.is_empty() else off


def _bbox_overlap(a, b) -> bool:
    return (a[0] < b[2] - EPS and b[0] < a[2] - EPS and
            a[1] < b[3] - EPS and b[1] < a[3] - EPS)


def _edge_candidates(placed, margin, gap):
    """Corners of the placed bounding boxes (with gap) and the sheet corner."""
    xs = {margin}
    ys = {margin}
    for entry in placed:
        x0, y0, x1, y1 = entry[2]
        xs.update((x0, x1 + gap))
        ys.update((y0, y1 + gap))
    xs = sorted(xs)
    ys = sorted(ys)
    return [(x, y) for y in ys for x in xs]


def _grid_candidates(margin, sheet_w, sheet_h, step):
    """Coarse grid as a fallback for shapes whose corner does not fall on a box corner."""
    step = max(float(step), 1.0)
    xs = np.arange(margin, sheet_w - margin + EPS, step)
    ys = np.arange(margin, sheet_h - margin + EPS, step)
    return [(float(x), float(y)) for y in ys for x in xs]


def _fits(cx, cy, w, h, infl, placed, sheet_w, sheet_h, margin, gap):
    """Does the part fit at `(cx, cy)`: within the margin and without overlap."""
    if (cx < margin - EPS or cy < margin - EPS or
            cx + w > sheet_w - margin + EPS or cy + h > sheet_h - margin + EPS):
        return None
    box = (cx, cy, cx + w, cy + h)
    half = gap / 2.0
    ibox = (cx - half, cy - half, cx + w + half, cy + h + half)
    moved = None
    for (_ox, _oy, obox, oibox, ocs) in placed:
        if not _bbox_overlap(ibox, oibox):
            continue
        if moved is None:
            moved = infl.translate(np.array([cx, cy], dtype=float))
        inter = mf.CrossSection.batch_boolean([moved, ocs], mf.OpType.Intersect)
        if inter.area() > AREA_EPS:
            return None
    if moved is None:
        moved = infl.translate(np.array([cx, cy], dtype=float))
    return (box, ibox, moved)


def _try_place(templates, placed, sheet_w, sheet_h, margin, gap):
    """First (lowest, then leftmost) place for a part on this sheet."""
    for (cx, cy) in _edge_candidates(placed, margin, gap):
        for (a, w, h, infl) in templates:
            res = _fits(cx, cy, w, h, infl, placed, sheet_w, sheet_h, margin, gap)
            if res is not None:
                return (a, cx, cy, res)
    step = max(gap, 1.0)
    grid = _grid_candidates(margin, sheet_w, sheet_h, step)
    for (cx, cy) in grid:
        for (a, w, h, infl) in templates:
            res = _fits(cx, cy, w, h, infl, placed, sheet_w, sheet_h, margin, gap)
            if res is not None:
                return (a, cx, cy, res)
    return None


def _templates(part: Part, rotations, usable_w, usable_h, gap):
    out = []
    shape = part_shape(part)
    for a in rotations:
        norm, _R, _bmin, _rmin = _norm_rot(shape, a)
        w, h = norm.max(axis=0)
        if w > usable_w + EPS or h > usable_h + EPS:
            continue
        out.append((float(a), float(w), float(h), _inflated(norm, gap)))
    return out


def nest(parts, sheet_w, sheet_h, margin, gap, rotations=(0, 90, 180, 270),
         max_sheets=20) -> list[tuple]:
    """Arrange the parts over as many sheets as needed, largest first.

    Returns `(part, sheet_index, angle, dx, dy)` per part. A part that is larger than
    the sheet (also when rotated) or more sheets than `max_sheets` gives a
    `ValueError` with a clear text.
    """
    parts = [p for p in parts if p.paths]
    if not parts:
        return []
    sheet_w, sheet_h = float(sheet_w), float(sheet_h)
    if sheet_w <= 0 or sheet_h <= 0:
        raise ValueError(tr("The sheet size must be greater than zero."))
    margin = max(0.0, float(margin))
    gap = max(0.0, float(gap))
    usable_w, usable_h = sheet_w - 2 * margin, sheet_h - 2 * margin
    if usable_w <= EPS or usable_h <= EPS:
        raise ValueError(tr("The margin is so wide that nothing fits on the sheet."))
    rots = tuple(rotations) if rotations else (0,)

    ordered = sorted(parts, key=lambda p: -part_area(p))
    placements: list[tuple] = []
    sheets: list[list] = []          # per sheet: (cx, cy, box, ibox, crosssection)
    for part in ordered:
        templates = _templates(part, rots, usable_w, usable_h, gap)
        if not templates:
            w, h = _footprint(part_shape(part))
            raise ValueError(
                tr("A part of {w:.0f} × {h:.0f} mm does not fit on a sheet of "
                   "{sheet_w:.0f} × {sheet_h:.0f} mm with a margin of {margin:g} mm.").format(
                    w=w, h=h, sheet_w=sheet_w, sheet_h=sheet_h, margin=margin))
        found = None
        index = None
        for si, placed in enumerate(sheets):
            found = _try_place(templates, placed, sheet_w, sheet_h, margin, gap)
            if found is not None:
                index = si
                break
        if found is None:
            if len(sheets) >= max_sheets:
                raise ValueError(tr("More than {max_sheets} sheets are needed for "
                                    "{n} parts.").format(max_sheets=max_sheets, n=len(parts)))
            sheets.append([])
            index = len(sheets) - 1
            found = _try_place(templates, sheets[index], sheet_w, sheet_h, margin, gap)
            if found is None:                        # empty and still too large
                w, h = _footprint(part_shape(part))
                raise ValueError(
                    tr("A part of {w:.0f} × {h:.0f} mm does not fit on the sheet.").format(
                        w=w, h=h))
        angle, cx, cy, (box, ibox, moved) = found
        sheets[index].append((cx, cy, box, ibox, moved))
        placements.append((part, index, angle, cx, cy))
    return placements


def _footprint(shape):
    shape = np.asarray(shape, dtype=float)
    span = shape.max(axis=0) - shape.min(axis=0)
    return float(span[0]), float(span[1])


# --------------------------------------------------------------------------
#  utilization and exporting per sheet
# --------------------------------------------------------------------------

def utilization(placements, sheet_w, sheet_h, sheet_index=None) -> float:
    """Fraction of the sheet area covered by parts (0..1)."""
    area = sum(part_area(p) for (p, si, *_rest) in placements
               if sheet_index is None or si == sheet_index)
    n = 1 if sheet_index is not None else max(1, _sheet_count(placements))
    total = float(sheet_w) * float(sheet_h) * n
    return 0.0 if total <= 0 else min(1.0, area / total)


def _sheet_count(placements) -> int:
    idx = [si for (_p, si, *_rest) in placements]
    return (max(idx) + 1) if idx else 0


def sheet_subdrawing(drawing, index, paths=None):
    """A separate drawing for one sheet: that sheet as the page, its origin as 0."""
    origins = sheet_origins(drawing)
    ox, oy = origins[index] if index < len(origins) else (0.0, 0.0)
    w, h = float(drawing.sheet_w or 0.0), float(drawing.sheet_h or 0.0)
    rect = (ox, oy, ox + w, oy + h)
    paths = list(paths if paths is not None else drawing.paths)
    kwargs = drawing._sheet_kwargs()
    kwargs.update(sheet_w=w, sheet_h=h, sheet_count=1)
    out = Drawing(layers={n: l for n, l in drawing.layers.items()}, **kwargs)
    for p in paths:
        if len(p.pts) and _points_in_rect(p.pts, rect, tol=1e-6):
            q = np.asarray(p.pts, dtype=float) - np.array([ox, oy])
            out.paths.append(VPath(np.ascontiguousarray(q), p.closed, p.layer, p.filled,
                                   p.rule, p.src, dict(p.meta)))
    for p in out.paths:
        out.ensure_layer(p.layer)
    return out
