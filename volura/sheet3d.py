"""From worksheet to model and back: the parts of the sheet as 3D objects.

Every part on the worksheet (an outer boundary with everything inside it, as
nesting sees it) becomes a plate with the thickness of the worksheet: the contours
that go through the plate (cut, mill outside and inside, drill) form the edge and
the holes, and a pocket removes the set depth from the top. Engravings and rasters
stay on the worksheet only. The parts sit in the model at the same place as on the
worksheet (X and Y equal), the bottom at Z = 0.

The other way (3D to the worksheet) goes through `vector_export`; this module only
holds how that drawing ends up on a layer of the worksheet.
"""
from __future__ import annotations

import numpy as np
import manifold3d as mf

from . import cam, nesting
from .i18n import tr
from .vector2d import Drawing, layer_param

#: Operations that go all the way through the plate: they determine the shape of a part.
THROUGH_OPS = ("cut", "mill_outside", "mill_inside", "drill")
#: Operations that only go to a depth (from the top).
DEPTH_OPS = ("pocket",)

DEFAULT_THICKNESS = 3.0
NAME_PREFIX = tr("Worksheet")       # only shown in group names, never searched for


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
    """[(name, Manifold)] of the parts that are cut through the plate.

    `paths` limits it to the parts those paths are in (the selection).
    A part without a closed outer boundary on a cutting layer gives nothing.
    """
    t = float(thickness)
    if t <= 0:
        raise ValueError(tr("The sheet thickness must be greater than 0."))
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
            continue                       # a loose pocket is not a part
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
        out.append((tr("{prefix} - part {n}").format(prefix=NAME_PREFIX, n=len(out) + 1), man))
    return out


def solid_thickness(groups) -> float | None:
    """The thickness of plates in the model: their height when it is the same for all."""
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
    """The same drawing with all paths on layer `layer_name` (the operation chosen in 3D)."""
    out = Drawing(warnings=list(drawing.warnings))
    lay = out.ensure_layer(layer_name, color or "#000000", op)
    for p in drawing.paths:
        out.add_path(p.pts, closed=p.closed, layer=lay.name, filled=p.filled, rule=p.rule,
                     src=p.src)
    return out
