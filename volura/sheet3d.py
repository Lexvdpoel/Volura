"""Van werkblad naar model en terug: de onderdelen van de plaat als 3D-objecten.

Elk onderdeel op het werkblad (een buitenrand met alles erbinnen, zoals het nesten
het ziet) wordt een plaat met de dikte van het werkblad: de contouren die door de
plaat gaan (snijden, frezen buiten en binnen, boren) vormen de rand en de gaten,
en een uitkamering haalt van boven de ingestelde diepte weg. Gravures en rasters
blijven alleen op het werkblad. De onderdelen staan in het model op dezelfde plek
als op het werkblad (X en Y gelijk), de onderkant op Z = 0.

De andere kant op (3D naar het werkblad) gaat met `vector_export`; hier staat
alleen hoe die tekening op een laag van het werkblad komt.
"""
from __future__ import annotations

import numpy as np
import manifold3d as mf

from . import cam, nesting
from .vector2d import Drawing, layer_param

#: Bewerkingen die helemaal door de plaat gaan: zij bepalen de vorm van een onderdeel.
THROUGH_OPS = ("cut", "mill_outside", "mill_inside", "drill")
#: Bewerkingen die maar tot een diepte gaan (van boven).
DEPTH_OPS = ("pocket",)

DEFAULT_THICKNESS = 3.0
NAME_PREFIX = "Werkblad"


def _op(drawing, path):
    lay = drawing.layers.get(path.layer)
    if lay is None or not lay.visible:
        return None
    return lay.op


def _section(paths) -> mf.CrossSection:
    loops = [np.ascontiguousarray(np.asarray(p.pts, dtype=float)) for p in paths
             if len(p.pts) >= 3]
    if not loops:
        return mf.CrossSection()
    return mf.CrossSection(loops, mf.FillRule.EvenOdd)


def part_solids(drawing: Drawing, thickness, paths=None):
    """[(naam, Manifold)] van de onderdelen die door de plaat gesneden worden.

    `paths` beperkt het tot de onderdelen waar die paden in zitten (de selectie).
    Een onderdeel zonder gesloten buitenrand op een snijlaag levert niets op.
    """
    t = float(thickness)
    if t <= 0:
        raise ValueError("De plaatdikte moet groter dan 0 zijn.")
    usable = [p for p in drawing.paths
              if cam._is_shape(p) and _op(drawing, p) in THROUGH_OPS + DEPTH_OPS]
    parts = nesting.parts_of(drawing, usable)
    wanted = None if paths is None else {id(p) for p in paths}
    out = []
    for part in parts:
        if part.outline is None:
            continue
        if wanted is not None and not any(id(p) in wanted for p in part.paths):
            continue
        outer = part.paths[0]
        if _op(drawing, outer) not in THROUGH_OPS:
            continue                       # een losse uitkamering is geen onderdeel
        through = [p for p in part.paths if _op(drawing, p) in THROUGH_OPS]
        cs = _section(through)
        if cs.is_empty():
            continue
        man = mf.Manifold.extrude(cs, t)
        for p in part.paths:
            if _op(drawing, p) not in DEPTH_OPS:
                continue
            depth = min(t, float(layer_param(drawing.layers.get(p.layer), "depth") or 0.0))
            if depth <= 0:
                continue
            pocket = mf.Manifold.extrude(_section([p]), depth + 1.0).translate((0, 0, t - depth))
            man = man - pocket
        if man.is_empty():
            continue
        out.append((f"{NAME_PREFIX} - deel {len(out) + 1}", man))
    return out


def solid_thickness(groups) -> float | None:
    """De dikte van platen in het model: hun hoogte als die voor allemaal gelijk is."""
    heights = set()
    for g in groups:
        bb = g.bbox()
        if bb is None:
            continue
        lo, hi = bb
        heights.add(round(float(hi[2] - lo[2]), 3))
    if len(heights) != 1:
        return None
    h = heights.pop()
    return h if h > 0 else None


def on_layer(drawing: Drawing, layer_name, color=None, op=None) -> Drawing:
    """Dezelfde tekening met alle paden op laag `layer_name` (de bewerking in 3D gekozen)."""
    out = Drawing(warnings=list(drawing.warnings))
    lay = out.ensure_layer(layer_name, color or "#000000", op)
    for p in drawing.paths:
        out.add_path(p.pts, closed=p.closed, layer=lay.name, filled=p.filled, rule=p.rule,
                     src=p.src)
    return out
