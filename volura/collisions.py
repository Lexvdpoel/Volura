"""Which parts of a model stand in one another.

A lid that is a hair too big for its box, a bracket that was moved into the wall, the same
part pasted twice on one spot: whatever stands inside another part is no longer two pieces,
and on the screen it looks like one. The collision view (View > Face Style > Collisions)
draws what two parts have in common in red, the way the tight fit view colours the places
that come too close to one another.

The rule is the intersection of the two solids: the overlap is a solid itself, and that solid
is what is drawn, so the red is exactly the material that has to move. Every part is asked
about every other one, and a pair whose boxes do not come near one another is answered by
those boxes alone - no boolean, no mesh, nothing - which is what keeps a model of a hundred
parts that stand apart cheap to look at.

Two solids that only touch at a face have an overlap of nothing: that is a fit and not a
collision, and the boolean answers it with an empty solid. A sliver of a few thousandths of a
cubic millimetre, which is what a boolean of solids that nearly touch leaves behind now and
then, stays under :data:`MIN_VOLUME` and is no collision either.
"""
from __future__ import annotations

import numpy as np

#: the smallest overlap in mm³ that still counts: a boolean of two solids that only nearly
#: touch leaves a sliver of that order behind, and no feature of a model is that small
MIN_VOLUME = 1e-3


def collisions(parts, min_volume=MIN_VOLUME):
    """Which pairs of the parts of a model stand in one another, and what they share.

    `parts` is a list of ``(key, manifold)``: one solid per part of the model, in the same
    millimetres the model is drawn in, with `key` whatever the caller uses to recognise the
    part again. The answer is one dict per pair that stands in one another, the biggest
    overlap first:

        ``{"a": key, "b": key, "volume": mm³, "mesh": (vertices, triangles)}``

    The two keys keep the order the parts stood in, so the caller reads the pair back as it
    gave it; equal volumes keep that order too, so the same model gives the same answer twice.
    `mesh` is the solid of the overlap: its vertices ``(n, 3)`` and its triangles ``(m, 3)``
    as indices into them, ready to draw.

    A pair of parts whose boxes do not touch is answered without a boolean at all, and so is a
    part that is no solid. `min_volume` is the smallest overlap that still counts; nothing, a
    negative or a nonsensical floor asks for :data:`MIN_VOLUME`-sized slivers to be dropped.
    """
    try:
        floor = MIN_VOLUME if min_volume is None else float(min_volume)
    except (TypeError, ValueError):
        floor = MIN_VOLUME
    if not floor >= 0.0:
        floor = 0.0
    solids = []
    for key, man in parts:
        box = _box(man)
        if box is not None:
            solids.append((key, man, box))
    hits = []
    for i in range(len(solids) - 1):
        ka, ma, ba = solids[i]
        for j in range(i + 1, len(solids)):
            kb, mb, bb = solids[j]
            if _apart(ba, bb):
                continue            # two boxes that do not touch share nothing to measure
            inter = _intersect(ma, mb)
            if inter is None:
                continue
            volume = _volume(inter)
            if volume is None or volume <= floor:
                continue            # touching at a face is a fit, not a collision
            mesh = _mesh(inter)
            if mesh is None:
                continue
            hits.append({"a": ka, "b": kb, "volume": volume, "mesh": mesh})
    hits.sort(key=lambda h: -h["volume"])       # a sort that is stable keeps the pairs of an
    return hits                                 # equal volume in the order the parts stood


def _intersect(a, b):
    """The solid two manifolds have in common, or None when the kernel will not make it.

    A pair the kernel refuses is no reason to say the two parts stand apart: it is answered
    with nothing, which is what a pair with no room in it says as well."""
    try:
        return a ^ b
    except Exception:
        return None


def _box(man):
    """The box of a solid as ``(min, max)``, or None when it is no solid or holds nothing."""
    if man is None:
        return None
    try:
        if man.is_empty():
            return None
        box = np.asarray(man.bounding_box(), dtype=float)
    except Exception:
        return None
    if box.size != 6 or not np.all(box[3:] >= box[:3]):
        return None
    return box[:3], box[3:]


def _apart(ba, bb):
    """Whether two boxes are further from one another than any solid of theirs can be.

    Two boxes that touch at a face or at an edge are not apart: the pair they stand for has
    to be asked, because the answer - an overlap of nothing - is the difference between a fit
    and a collision."""
    return bool(np.any(ba[0] > bb[1]) or np.any(bb[0] > ba[1]))


def _volume(man):
    """The content of a solid in mm³, or None when the kernel will not say."""
    try:
        return float(man.volume())
    except Exception:
        return None


def _mesh(man):
    """The triangles of a solid as ``(vertices, triangles)``, or None when it has none."""
    try:
        mesh = man.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    except Exception:
        return None
    if not len(verts) or not len(tris):
        return None
    return verts, tris
