"""Importing vector files (SVG, DXF) into the 3D environment.

The logic here is deliberately free of Qt: it reads a drawing (via `volura.svg` or
`volura.dxf`), scales and anchors it, and turns it into faces, edges or an
extruded solid in the B-rep model. That way everything can be tested without a
window. The interface in `volura.app` and the tool in `volura.tools` build on this.

All 2D sizes are millimetres with Y up (the shared model of `volura.vector2d`);
`plane` is the plane to place on (the ground or a face of a model) with a right
handed local frame `(u, v, n)`.
"""
from __future__ import annotations

import os

import numpy as np
import manifold3d as mf

from . import dxf, svg
from .geom import extrude_cs
from .i18n import tr
from .vector2d import regions

VECTOR_EXTENSIONS = (".svg", ".dxf")


def is_vector(path) -> bool:
    """True if `path` is a vector file (SVG or DXF, also in upper case)."""
    return os.path.splitext(str(path))[1].lower() in VECTOR_EXTENSIONS


def load_vector(path):
    """An SVG or DXF file as a 2D drawing; the extension decides which."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".svg":
        return svg.read_svg(path)
    if ext == ".dxf":
        return dxf.read_dxf(path)
    raise ValueError(tr("Only SVG and DXF files can be imported as vectors."))


def load_svg_text(text):
    """SVG text (e.g. from the clipboard) as a 2D drawing."""
    return svg.read_svg(text)


def prepare(drawing, scale=1.0, target_width=None, anchor="center"):
    """Scale and anchor a drawing; return (regions, open paths).

    `target_width` (in mm) wins over `scale`: the width of the bounding box then
    becomes exactly `target_width` and the height scales along. Then the drawing is
    shifted so that the anchor -- the centre ("center") or the lower left corner
    ("lower_left") of the bounding box -- lies at the origin.

    The closed shapes come back as regions (via `vector2d.regions`, holes following
    the fill rule of the source) and the open paths as a list of `(N, 2)` arrays.
    """
    b = drawing.bounds()
    if b is None:
        return [], []
    xmin, ymin, xmax, ymax = b
    w, h = xmax - xmin, ymax - ymin
    factor = float(scale) if scale else 1.0
    if target_width is not None and w > 0.0:
        factor = float(target_width) / w
    if str(anchor) == "lower_left":
        ax, ay = xmin, ymin
    else:
        ax, ay = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    M = np.array([[factor, 0.0, -factor * ax],
                  [0.0, factor, -factor * ay],
                  [0.0, 0.0, 1.0]])
    scaled = drawing.transformed(M)
    closed = [p for p in scaled.paths if p.closed and len(p.pts) >= 3]
    open_paths = [np.asarray(p.pts, dtype=float) for p in scaled.paths
                  if not p.closed and len(p.pts) >= 2]
    return regions(closed), open_paths


def place_faces(ents, plane, regios, open_paden, offset):
    """Turn the regions into faces and the open paths into edges on `plane`.

    `regios` is the list of shapes from `prepare` (first loop the outer boundary, the
    rest holes); each becomes one face with `ents.add_shape`. `open_paden` become
    edges with `ents.add_segment`. `offset` is the 2D shift in the local frame of
    `plane` (where the anchor ends up). Returns the new face ids.
    """
    offset = np.asarray(offset, dtype=float).reshape(2)
    created = []
    for shape in regios:
        loops = [np.asarray(loop, dtype=float) + offset for loop in shape if len(loop) >= 3]
        if loops:
            created += list(ents.add_shape(plane, loops) or [])
    for path in open_paden or ():
        pts = np.asarray(path, dtype=float) + offset
        if len(pts) < 2:
            continue
        W = plane.to_world(pts)
        for i in range(len(W) - 1):
            ents.add_segment(W[i], W[i + 1])
    return created


def cross_section(regios, offset=(0.0, 0.0)):
    """One cross-section of all regions (holes stay holes), shifted by `offset`."""
    offset = np.asarray(offset, dtype=float).reshape(2)
    sections = []
    for shape in regios:
        loops = [np.ascontiguousarray(np.asarray(loop, dtype=float) + offset, float)
                 for loop in shape if len(loop) >= 3]
        if not loops:
            continue
        cs = mf.CrossSection(loops, mf.FillRule.NonZero)
        if not cs.is_empty():
            sections.append(cs)
    if not sections:
        return mf.CrossSection()
    if len(sections) == 1:
        return sections[0]
    return mf.CrossSection.batch_boolean(sections, mf.OpType.Add)


def place_extruded(doc, plane, regios, offset, height, color, name):
    """Extrude the regions into one group perpendicular to `plane`, along the normal.

    The height runs along the normal of `plane` (outwards for a model face); a
    negative height pushes in (to the other side of the face). The result is always
    one separate group: the underlying model stays unchanged. Returns the new group,
    or None if there was nothing to extrude.
    """
    from . import ops
    cs = cross_section(regios, offset)
    if cs.is_empty():
        return None
    height = float(height)
    if abs(height) < 1e-9:
        return None
    man = extrude_cs(cs, plane, min(0.0, height), max(0.0, height))
    if man.is_empty():
        return None
    return ops.group_from_manifold(doc, man, color, name)
