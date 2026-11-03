"""De rekenkern voor laser en frees: nestingdiepte, kerf-/freescorrectie en dogbones.

Geen Qt in dit bestand. `nesting_depth`, `compensate` en `dogbones` werken op het
gedeelde 2D-model uit `volura/vector2d.py` en zijn daarom los te testen en elders
weer te gebruiken (het lagenpaneel, "Exporteren voor laser/frees" en het nesten).

Een laser snijdt in het midden van de getekende lijn en haalt aan beide kanten de
halve kerf weg; een frees heeft een eigen straal. De contour moet dus opzij geschoven
worden om na het snijden de getekende maat te geven. Welke kant op zegt de
nestingdiepte: de buitenrand van een onderdeel (diepte 0) naar buiten, een gat
(diepte 1) de andere kant op, een eiland in dat gat (diepte 2) weer naar buiten.

Een ronde frees kan niet in een scherpe binnenhoek komen. Daar hoort een uitsparing
bij: een cirkel met de freesdiameter door het hoekpunt, langs de bissectrice
("dogbone") of langs de langste van de twee randen ("T-bone").
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import manifold3d as mf

from .vector2d import (Drawing, VPath, _point_in_poly, arc_points, layer_param,
                       polygon_area, regions)

MAX_ANGLE = 179.0        # hoeken die scherper zijn dan dit laat de frees staan
CURVE_TOL = 0.002        # afwijking van de cirkel van een dogbone, in mm
MAX_RINGS = 4000         # een uitkameringsgebied loopt nooit oneindig door


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------

def _is_shape(path) -> bool:
    """Alleen een gesloten pad met minstens drie punten is een vorm."""
    return bool(getattr(path, "closed", False)) and len(getattr(path, "pts", ())) >= 3


def _polygons(cs: mf.CrossSection) -> list[np.ndarray]:
    """De lussen van een doorsnede, gaten en al, als arrays."""
    return [np.asarray(q, dtype=float) for q in cs.to_polygons() if len(q) >= 3]


def _section(loops) -> mf.CrossSection:
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def _copy(path: VPath) -> VPath:
    return VPath(np.asarray(path.pts, dtype=float).copy(), path.closed, path.layer,
                 path.filled, path.rule, path.src, dict(path.meta))


def _new(path: VPath, pts) -> VPath:
    """Een nieuw gesloten pad op de laag van `path`; de meta van een cirkel klopt niet meer."""
    return VPath(np.asarray(pts, dtype=float), True, path.layer, path.filled,
                 path.rule, path.src, {})


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else np.zeros(2)


def _edge_distance(poly, q) -> float:
    """Kleinste afstand van punt `q` tot de rand van de gesloten polylijn `poly`."""
    p = np.asarray(poly, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
    t = np.clip(np.sum((q - a) * d, axis=1) / L2, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - q, axis=1)))


def _circle(center, r, tol=CURVE_TOL) -> np.ndarray:
    """Een cirkel als gesloten polylijn (laatste punt valt weg)."""
    pts = arc_points(np.asarray(center, dtype=float), abs(float(r)),
                     0.0, 2.0 * math.pi, tol)
    return np.ascontiguousarray(np.asarray(pts, dtype=float)[:-1])


# --------------------------------------------------------------------------
#  nestingdiepte
# --------------------------------------------------------------------------

def _interior_point(poly) -> np.ndarray:
    """Een punt net binnen de lus: bij een bolle hoek, dicht bij de rand.

    Het midden van de punten is geen goed vertegenwoordigend punt: bij een plaat met
    een gat ligt het midden van de buitenrand just in het gat. Het oortje van een hoek
    ligt tegen de rand aan en dus buiten de contouren die erin liggen.
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
            continue                        # holle hoek: de bissectrice wijst naar buiten
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
    """Hoe diep elk gesloten pad in de andere ligt: 0 buitenrand, 1 gat, 2 eiland in een gat.

    De lijst hoort bij de gegeven paden; een open pad of een pad dat geen vorm is
    krijgt 0. Paden van verschillende lagen tellen niet voor elkaar, dus wie dit voor
    een hele tekening gebruikt krijgt de diepte binnen de eigen laag terug door de
    paden per laag aan te bieden.
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
#  verschuiven
# --------------------------------------------------------------------------

def offset_loop(pts, delta, join=None) -> list[np.ndarray]:
    """Een gesloten pad `delta` millimeter opzij zetten (positief naar buiten).

    Afgeronde hoeken (`mf.JoinType.Round`), want een laser en een frees draaien.
    Een gat dat dichtloopt of een contour die te klein is voor de frees geeft niets.
    """
    delta = float(delta)
    if abs(delta) < 1e-12:
        return [np.asarray(pts, dtype=float).copy()]
    try:
        cs = _section([np.asarray(pts, dtype=float)]).offset(
            delta, join or mf.JoinType.Round, 2.0)
    except Exception:                       # manifold3d wijst soms een sprong af
        return []
    return [] if cs.is_empty() else _polygons(cs)


def region_of(paths) -> mf.CrossSection:
    """Het gebied dat de gesloten paden van een laag omsluiten (gaten eruit)."""
    loops = [loop for shape in regions(paths) for loop in shape]
    return _section(loops) if loops else mf.CrossSection()


def pocket_paths(paths, tool_d, stepover=50.0) -> list[np.ndarray]:
    """De paden van een uitkamering: eerst de rand met de freesradius naar binnen,
    daarna concentrisch smaller met de stapover totdat het gebied op is."""
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
    """Hoe ver een pad hooguit naar binnen kan: de halve diagonal van de omvang."""
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
    """De middelpunten van de uitsparingscirkels bij de binnenhoeken van een gesloten pad.

    Een hoek is een binnenhoek als het gebied binnen het pad (het uit te freesde
    gebied) daar scherper is dan `max_angle` graden: de frees komt er niet. De cirkel
    heeft de freesdiameter en gaat door het hoekpunt, dus zijn middelpunt staat op
    afstand van de freesradius: langs de bissectrice ("dogbone") of langs de langste
    van de twee randen ("T-bone").

    De veelhoekjes waarmee een cirkel of boog is afgevlakt zijn geen hoeken: daar rolt
    de frees langs. Een hoek telt daarom pas als de uitsparing grof is vergeleken met
    de ribben en de freesradius zelf.
    """
    r = abs(float(tool_d)) / 2.0
    p = np.asarray(pts, dtype=float)
    if r <= 0.0 or len(p) < 3:
        return []
    if polygon_area(p) < 0.0:               # tegen de klok in: binnen ligt links van de rand
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
            continue                        # bolle hoek voor het gereedschap: die haalt het
        hoek = math.degrees(math.acos(max(-1.0, min(1.0, float(d1 @ d2) / (L1 * L2)))))
        if hoek >= float(max_angle):
            continue
        if 0.5 * float(np.hypot(*(d1 + d2))) < max(0.05 * min(L1, L2), 0.02 * r):
            continue                        # afgevlakte kromme, geen echte hoek
        if str(style).lower() == "tbone":
            weg = _unit(d1) if L1 >= L2 else _unit(d2)
        else:
            weg = _unit(_unit(d1) + _unit(d2))
        if float(np.hypot(*weg)) < 1e-9:
            continue
        out.append(v + weg * r)
    return out


def dogbone_loops(pts, tool_d, style="dogbone") -> list[np.ndarray]:
    """Een gesloten pad met de uitsparingscirkels verenigen; weer één gesloten pad."""
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
    """Voeg bij elke binnenhoek van de gesloten paden een uitsparing toe.

    Het resultaat is weer een lijst gesloten paden op dezelfde laag, klaar om te
    frezen; open paden blijven zoals ze zijn.
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
#  de correctie van een hele tekening
# --------------------------------------------------------------------------

def compensate_path(layer, path, depth=0) -> list[VPath]:
    """Eén pad, verschoven zoals de bewerking van zijn laag dat vraagt."""
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
    """De concentrische paden van een uitkamering, aan de eerste contour van de laag."""
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
    """Dogbones op de binnenhoeken van de lussen, als de laag die vraagt."""
    if not style or str(style) == "none":
        return list(loops)
    out: list[np.ndarray] = []
    for loop in loops:
        out.extend(dogbone_loops(loop, tool_d, style))
    return out


def compensate(drawing) -> Drawing:
    """Een nieuwe tekening met de contouren zoals de machine ze moet volgen.

    Per laag, met de instellingen van die laag: "cut" met de halve kerf naar buiten
    voor de buitenranden en naar binnen voor de gaten (het gat wordt kleiner, zodat
    het na het snijden de getekende maat heeft); "mill_outside" en "mill_inside" met
    de freesradius naar buiten of naar binnen; "pocket" met de rand met de
    freesradius naar binnen en daarna concentrische paden met de stapover tot het
    gebied leeg is. "engrave", "raster", "drill" en open lijnen blijven zoals ze zijn.
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
        out.warn(f"{skipped} contouren zijn te klein voor de gekozen frees of kerf en "
                 "zijn overgeslagen.")
    return out
