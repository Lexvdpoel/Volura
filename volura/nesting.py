"""Automatisch nesten: onderdelen zo zuinig mogelijk op één of meer platen schikken.

Geen Qt in dit bestand. Het werkt op het gedeelde 2D-model uit `volura/vector2d.py`
en de nestingdiepte uit `volura/cam.py`, en is daarom los te testen en in het
werkblad, de export en de freespaden weer te gebruiken.

Een *onderdeel* (`Part`) is een buitenrand met alles wat erbinnen ligt (gaten,
eilanden, gravures en open paden), zodat het als één geheel over de plaat beweegt.
Losse open paden zonder buitenrand vormen een eigen onderdeel op hun bounding box.

`nest` schikt de onderdelen met "bottom-left fill": grootste eerst, en per
onderdeel en rotatie de laagste-dan-meest-linkse plek waar het, opgeblazen met de
halve tussenruimte, geen ander onderdeel raakt en binnen de plaat min de marge valt.
De plaat is bij een rotatie van 0, 90, 180 of 270 graden altijd even groot, dus de
geometrie blijft binnen de plaat; de rotatie gebeurt rond het eigen hoekpunt.

Een plaats wordt teruggegeven als `(part, sheet_index, angle, dx, dy)`: draai het
onderdeel met `angle` graden en zet de onderkant-linkerhoek van zijn omvang op
`(dx, dy)` binnen die plaat. `place_transform` maakt er de 3x3-matrix van die alle
paden van het onderdeel meenemen; tel daar de oorsprong van de plaat bij op
(`sheet_origins`) om op het werkblad te landen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from . import cam
from .vector2d import (SHEET_SPACING, Drawing, VPath, _point_in_poly, polygon_area)

EPS = 1e-9            # rakende boxen tellen niet als overlap
AREA_EPS = 1e-6       # een doorsnede kleiner dan dit (mm²) is geen overlap


# --------------------------------------------------------------------------
#  plaatmeetkunde
# --------------------------------------------------------------------------

def sheet_origins(drawing) -> list[tuple[float, float]]:
    """De linkeronderhoek van elke plaat: plaat i staat op (i * (breedte + 50), 0)."""
    w, h = drawing.sheet_w, drawing.sheet_h
    if not w or not h:
        return []
    n = max(1, int(drawing.sheet_count or 1))
    return [(i * (float(w) + SHEET_SPACING), 0.0) for i in range(n)]


def sheet_rects(drawing) -> list[tuple[float, float, float, float]]:
    """Het volle plaatvlak (x0, y0, x1, y1) per plaat."""
    w, h = float(drawing.sheet_w or 0.0), float(drawing.sheet_h or 0.0)
    return [(ox, oy, ox + w, oy + h) for ox, oy in sheet_origins(drawing)]


def usable_rects(drawing) -> list[tuple[float, float, float, float]]:
    """Het bruikbare vlak van elke plaat: de plaat min de marge langs de rand."""
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
    """Paden die (deels) buiten elke plaat of in de marge liggen; leeg zonder plaat."""
    rects = usable_rects(drawing)
    if not rects:
        return []
    paths = list(paths if paths is not None else drawing.paths)
    return [p for p in paths if len(p.pts) and
            not any(_points_in_rect(p.pts, r) for r in rects)]


# --------------------------------------------------------------------------
#  onderdelen
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
    """Een buitenrand met alles erbinnen, of een bos losse open paden op hun box."""

    paths: list = field(default_factory=list)
    outline: np.ndarray | None = None      # buitenrand (tegen de klok in) of None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    area: float = 0.0

    @property
    def shape(self) -> np.ndarray:
        """De omtrek waarmee het onderdeel op de plaat gelegd wordt."""
        if self.outline is not None:
            return np.asarray(self.outline, dtype=float)
        x0, y0, x1, y1 = self.bbox
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def part_shape(part: Part) -> np.ndarray:
    return part.shape


def part_area(part: Part) -> float:
    return float(part.area)


def parts_of(drawing, paths=None) -> list[Part]:
    """Groepeer paden in onderdelen: een buitenrand (diepte 0) met alles erbinnen.

    Gaten, eilanden, gravures en open paden die binnen een buitenrand liggen, gaan
    met die buitenrand mee. Losse open paden zonder buitenrand vormen samen met de
    paden rond hetzelfde punt een eigen onderdeel op hun bounding box.
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
        if owner is None:                       # dieper dan 0 maar zonder buitenrand
            poly = np.asarray(p.pts, dtype=float)
            parts.append(Part(paths=[p], outline=poly))
        else:
            owner.paths.append(p)

    for p in others:
        spot = np.asarray(p.pts, dtype=float).mean(axis=0)
        owner = _owner(spot, outer_index)
        if owner is None:
            parts.append(Part(paths=[p]))       # los open pad: eigen onderdeel op de box
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
#  transformatie van een plaats
# --------------------------------------------------------------------------

def _rot(angle_deg) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def _norm_rot(shape, angle_deg):
    """Vorm draaien en zo schuiven dat de omvang in de oorsprong begint."""
    shape = np.asarray(shape, dtype=float)
    bmin = shape.min(axis=0)
    R = _rot(angle_deg)
    rot = (shape - bmin) @ R.T
    rmin = rot.min(axis=0)
    return rot - rmin, R, bmin, rmin


def place_transform(part: Part, angle, dx, dy, origin=(0.0, 0.0)) -> np.ndarray:
    """3x3-matrix die het onderdeel naar `(dx, dy)` op een plaat zet (met rotatie)."""
    norm, R, bmin, rmin = _norm_rot(part_shape(part), angle)
    M = np.eye(3, dtype=float)
    M[:2, :2] = R
    off = np.array([float(dx), float(dy)], dtype=float) - rmin - (bmin @ R.T)
    M[:2, 2] = off + np.asarray(origin, dtype=float)
    return M


# --------------------------------------------------------------------------
#  nesten
# --------------------------------------------------------------------------

def _section(loops) -> mf.CrossSection:
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def _inflated(norm_shape, gap) -> mf.CrossSection:
    """De houd-over-ruimte rond een onderdeel: zijn vorm met de halve tussenruimte."""
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
    """Hoeken van de geplaatste omvangbakken (met tussenruimte) en de plaathoek."""
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
    """Grof raster als terugval voor vormen wier hoek niet op een boxhoek valt."""
    step = max(float(step), 1.0)
    xs = np.arange(margin, sheet_w - margin + EPS, step)
    ys = np.arange(margin, sheet_h - margin + EPS, step)
    return [(float(x), float(y)) for y in ys for x in xs]


def _fits(cx, cy, w, h, infl, placed, sheet_w, sheet_h, margin, gap):
    """Past het onderdeel op `(cx, cy)`: binnen de marge en zonder overlap."""
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
    """Eerste (laagste-, dan meest-linkse) plek voor een onderdeel op deze plaat."""
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
    """Schik de onderdelen over zoveel platen als nodig, grootste eerst.

    Geeft per onderdeel `(part, sheet_index, angle, dx, dy)`. Een onderdeel dat
    groter is dan de plaat (ook gedraaid) of meer platen dan `max_sheets` geeft een
    `ValueError` met een duidelijke tekst.
    """
    parts = [p for p in parts if p.paths]
    if not parts:
        return []
    sheet_w, sheet_h = float(sheet_w), float(sheet_h)
    if sheet_w <= 0 or sheet_h <= 0:
        raise ValueError("Het plaatformaat moet groter dan nul zijn.")
    margin = max(0.0, float(margin))
    gap = max(0.0, float(gap))
    usable_w, usable_h = sheet_w - 2 * margin, sheet_h - 2 * margin
    if usable_w <= EPS or usable_h <= EPS:
        raise ValueError("De marge is zo breed dat er niets op de plaat past.")
    rots = tuple(rotations) if rotations else (0,)

    ordered = sorted(parts, key=lambda p: -part_area(p))
    placements: list[tuple] = []
    sheets: list[list] = []          # per plaat: (cx, cy, box, ibox, crosssection)
    for part in ordered:
        templates = _templates(part, rots, usable_w, usable_h, gap)
        if not templates:
            w, h = _footprint(part_shape(part))
            raise ValueError(
                f"Onderdeel van {w:.0f} × {h:.0f} mm past niet op een plaat van "
                f"{sheet_w:.0f} × {sheet_h:.0f} mm met marge {margin:g} mm.")
        found = None
        index = None
        for si, placed in enumerate(sheets):
            found = _try_place(templates, placed, sheet_w, sheet_h, margin, gap)
            if found is not None:
                index = si
                break
        if found is None:
            if len(sheets) >= max_sheets:
                raise ValueError(f"Er zijn meer dan {max_sheets} platen nodig voor "
                                 f"{len(parts)} onderdelen.")
            sheets.append([])
            index = len(sheets) - 1
            found = _try_place(templates, sheets[index], sheet_w, sheet_h, margin, gap)
            if found is None:                        # leeg en toch te groot
                w, h = _footprint(part_shape(part))
                raise ValueError(
                    f"Onderdeel van {w:.0f} × {h:.0f} mm past niet op de plaat.")
        angle, cx, cy, (box, ibox, moved) = found
        sheets[index].append((cx, cy, box, ibox, moved))
        placements.append((part, index, angle, cx, cy))
    return placements


def _footprint(shape):
    shape = np.asarray(shape, dtype=float)
    span = shape.max(axis=0) - shape.min(axis=0)
    return float(span[0]), float(span[1])


# --------------------------------------------------------------------------
#  benuttingsgraad en per plaat exporteren
# --------------------------------------------------------------------------

def utilization(placements, sheet_w, sheet_h, sheet_index=None) -> float:
    """Deel van de plaatoppervlakte dat door onderdelen bedekt is (0..1)."""
    area = sum(part_area(p) for (p, si, *_rest) in placements
               if sheet_index is None or si == sheet_index)
    n = 1 if sheet_index is not None else max(1, _sheet_count(placements))
    total = float(sheet_w) * float(sheet_h) * n
    return 0.0 if total <= 0 else min(1.0, area / total)


def _sheet_count(placements) -> int:
    idx = [si for (_p, si, *_rest) in placements]
    return (max(idx) + 1) if idx else 0


def sheet_subdrawing(drawing, index, paths=None):
    """Een losse tekening voor één plaat: die plaat als pagina, zijn oorsprong als 0."""
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
