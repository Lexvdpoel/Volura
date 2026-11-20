"""The section view: cut the 3D view open with a plane you can move.

A section cuts the way you look at the model and never the model itself. The plane stands
on one of the three axes and slides through it; everything on the far side of the plane is
gone from the view, and where the plane cuts through material the cut surface is drawn as a
solid cap, so a part that is hollow inside also looks hollow inside.

The plane is kept as the equation ``a*x + b*y + c*z + d``, of which OpenGL's ``glClipPlane``
and the picking of the view keep everything with ``>= 0``. The caps come from the solid: the
part is turned so that the axis of the plane stands up, cut there and turned back again, the
same turn the thin wall view cuts with (see ``thin_walls.AXES``).

The section belongs to the view: it is not stored in the file, and it is off when Volura
starts, because a model that is half hidden at start-up looks broken.
"""
from __future__ import annotations

import math

import numpy as np

from .geom import triangulate_cs
from .i18n import tr
from .thin_walls import AXES as TURNS

#: the directions a section plane can stand in
AXES = ("x", "y", "z")
#: the cut of one axis: thin_walls.AXES holds, per axis, the turn of the solid that lifts
#: that axis to z and the same turn for the points (`p @ turn` sends a point along)
CUT = {"z": TURNS[0], "x": TURNS[1], "y": TURNS[2]}
#: mm a plane that lands in a flat face of the part is moved into the material
NUDGE = 1e-6

NO_AXIS = tr("a section plane stands along x, y or z")


def plane(axis, pos, flip=False):
    """The cut plane as ``(a, b, c, d)``, keeping every point with ``a*x + b*y + c*z + d >= 0``.

    Without Flip the half below `pos` stays - the lower half of a plane along z - so you
    look down into the model from above; the Flip checkbox of the panel keeps the other half."""
    if axis not in AXES:
        raise ValueError(NO_AXIS)
    normal = [0.0, 0.0, 0.0]
    normal[AXES.index(axis)] = 1.0 if flip else -1.0
    return (normal[0], normal[1], normal[2], -normal[AXES.index(axis)] * float(pos))


def keeps(eq, points):
    """Which of the `points` the plane equation `eq` keeps, as one bool per point.

    A point that stands exactly on the plane is kept: what is cut open is not gone, it is
    what the cap is drawn over."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    a, b, c, d = (float(v) for v in eq)
    return pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d >= 0.0


def span(box, axis):
    """The low and the high of the bounding box `box` along `axis`, or None without a model."""
    if box is None or axis not in AXES:
        return None
    k = AXES.index(axis)
    return float(box[0][k]), float(box[1][k])


def view_position(pos, lo, hi):
    """The position of the plane on an axis that runs from `lo` to `hi`.

    No position is the middle of the model, and a position beside the model - or no number
    at all - is moved to the nearest edge, so the plane always lands on the model."""
    at = 0.5 * (lo + hi) if pos is None or not math.isfinite(float(pos)) else float(pos)
    return min(max(at, lo), hi)


def view_plane(axis, pos, flip, box):
    """The plane the view cuts with, or None when there is no model to cut.

    `pos` and the bounding box `box` are as the panel and the view have them; see
    :func:`view_position` for what is made of a position that is missing or beside the model."""
    limits = span(box, axis)
    if limits is None:
        return None
    return plane(axis, view_position(pos, *limits), flip)


def cut_of(settings, box):
    """What the view of `settings` cuts, as ``(axis, pos, flip, equation)``, or None when off.

    The one place the names of the section settings are read: the view clips with this plane,
    the picking filters on it and the panel and the status bar show it, so all three cut the
    same model even when a position is missing or stands beside it (see :func:`view_position`)."""
    if settings is None or not getattr(settings, "section_on", False):
        return None
    axis = getattr(settings, "section_axis", "z")
    limits = span(box, axis)
    if limits is None:
        return None
    flip = bool(getattr(settings, "section_flip", False))
    pos = view_position(getattr(settings, "section_pos", None), *limits)
    return axis, pos, flip, plane(axis, pos, flip)


def ray_span(eq, o, d):
    """The stretch of the ray `o` + `d` * t that the plane equation `eq` keeps, as ``(lo, hi)``.

    A plane cuts a line in one point, so what stays of a ray is one stretch of it: everything
    before the plane for a ray that runs into the cut-away half, everything after it for one
    that runs out of it. None when the whole ray lies in the cut-away half - there is nothing
    left to hit - and an open end as ``math.inf``."""
    n = np.asarray(eq[:3], dtype=float)
    at = float(n @ np.asarray(o, dtype=float)) + float(eq[3])
    along = float(n @ np.asarray(d, dtype=float))
    if at >= 0.0:                    # the ray starts in what stays
        return (0.0, math.inf) if along >= 0.0 else (0.0, -at / along)
    if along <= 0.0:                 # it runs along or away from the plane: nothing is kept
        return None
    return -at / along, math.inf


def keep_span(eq, a, b):
    """Per segment from `a` to `b` the part the plane equation `eq` keeps, as ``(lo, hi)``.

    The parameters run along the segment itself, from 0 at `a` to 1 at `b`. A segment that
    hangs entirely in the cut-away half gets an empty span - `lo` greater than `hi` - and one
    that crosses the plane keeps the part that is on the side that stays."""
    n = np.asarray(eq[:3], dtype=float)
    d = float(eq[3])
    pa = np.asarray(a, dtype=float).reshape(-1, 3)
    pb = np.asarray(b, dtype=float).reshape(-1, 3)
    ga, gb = pa @ n + d, pb @ n + d
    diff = gb - ga
    at = np.clip(-ga / np.where(np.abs(diff) < 1e-12, 1.0, diff), 0.0, 1.0)
    lo = np.where((ga < 0.0) & (gb >= 0.0), at, 0.0)
    hi = np.where((ga >= 0.0) & (gb < 0.0), at, 1.0)
    gone = (ga < 0.0) & (gb < 0.0)
    return np.where(gone, 1.0, lo), np.where(gone, 0.0, hi)


def cap_triangles(manifold, axis, pos):
    """The surface the part gets where the plane cuts it, as ``(n, 3, 3)`` float32.

    The part is cut standing along its own z and the triangles are turned back, so they lie
    in the plane at `pos` in the coordinates of the part. A cut in nothing is no triangles
    at all: an empty part, a plane beside it, a part that is only a loose face or a
    manifold3d that refuses the cut. A plane that lands exactly in a flat face of the part
    is moved a hair into the material first, because a cut through a top or a bottom is a
    sliver of round-off."""
    empty = np.zeros((0, 3, 3), dtype=np.float32)
    if axis not in AXES:
        raise ValueError(NO_AXIS)
    if manifold is None:
        return empty
    try:
        angles, turn = CUT[axis]
        if manifold.is_empty():
            return empty
        solid = manifold if angles is None else manifold.rotate(angles)
        z = float(pos)
        face = _cross(solid, z)
        if face is None:
            face = _on_a_face(solid, z)
        if face is None:
            return empty
        pts, tri = triangulate_cs(face)
        if not len(tri):
            return empty
        out = np.empty((len(tri), 3, 3), dtype=np.float32)
        out[:, :, :2] = pts[tri]
        out[:, :, 2] = np.float32(z)
        if turn is not None:
            out = (out.reshape(-1, 3) @ turn.T).astype(np.float32)
        return out.reshape(-1, 3, 3)
    except Exception:               # a part that refuses to be cut says nothing about it
        return empty


def _cross(solid, z):
    """The cross-section of `solid` at z, or None when there is nothing there to cap."""
    try:
        face = solid.slice(z)
    except Exception:
        return None
    return None if face is None or face.is_empty() else face


def _on_a_face(solid, z):
    """The cross-section a plane gets when it lands in the bottom or the top face of the part.

    A cut through a top or a bottom is a sliver of round-off, so the plane is moved a hair
    into the material; a plane that lies beside the part stays beside it, and so does one
    that lands in the gap between two parts."""
    box = solid.bounding_box()
    lo, hi = float(box[2]), float(box[5])
    if hi - lo <= 2 * NUDGE:
        return None
    if abs(z - lo) <= NUDGE:
        return _cross(solid, lo + NUDGE)
    if abs(z - hi) <= NUDGE:
        return _cross(solid, hi - NUDGE)
    return None
