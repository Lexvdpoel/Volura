"""Painted faces as volumes of their own: how a painted model is cut for a multi-material print.

No Qt in this file. A face painted with the Paint Bucket is no part: it is a color on the skin
of one. PrusaSlicer and OrcaSlicer turn such painted faces into volumes of their own when they
segment a model for a printer with several filaments: a skin of a few millimetres under every
painted face, with the rest of the part around it. `split_painted` does the same, so Volura's
own slicer can print a painted model on more than one filament.

Every painted triangle makes a prism: the triangle, and the same triangle moved `depth` mm
inwards along its normal. The prisms of neighbours are grown a little in their own plane, so
they reach past the edge they share and overlap there: without that, a sliver of the part is
left unclaimed between two prisms that stand at an angle to each other. The prisms of one color
are unioned in one batch and cut to the part; where the skins of two colors overlap, the color
that comes later keeps it, so the volumes never overlap and, together, are the part."""
from __future__ import annotations

import manifold3d as mf
import numpy as np

#: how far a painted triangle reaches past its own corners, in its own plane (mm): enough for
#: the prisms of neighbours to overlap, far too little to see in the print
GROW = 0.01
#: what is left of a boolean under this volume (mm3) is nothing
TINY = 0.001
#: the eight triangles of one prism, out of the six corners `_prisms` makes
PRISM = np.array([[0, 1, 2], [3, 5, 4], [1, 0, 3], [1, 3, 4],
                  [2, 1, 4], [2, 4, 5], [0, 2, 5], [0, 5, 3]], dtype=np.uint32)
#: the corners of one prism: the grown triangle, and the same triangle moved inwards
CORNERS = 6


def _unit(v):
    """Every vector of `v` as a unit vector; a vector of no length stays as it is."""
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, ln, out=np.zeros_like(v), where=ln > 1e-12)


def _grown(a, b, c, grow=GROW):
    """Every corner of these triangles moved `grow` mm out of the triangle, in its own plane.

    A corner moves along the bisector of the outward normals of the two edges it bounds, so the
    grown triangle covers the edges of the original one; for a sharp corner that way leads a
    long way out, so the distance is capped at four times `grow`, like a miter joint is capped.
    """
    n = _unit(np.cross(b - a, c - a))
    out = []
    for p, before, after in ((a, c, b), (b, a, c), (c, b, a)):
        m1 = _unit(np.cross(_unit(p - before), n))       # out of the edge that ends in p
        m2 = _unit(np.cross(_unit(after - p), n))        # out of the edge that starts in p
        sv = m1 + m2
        sq = np.maximum(np.einsum("ij,ij->i", sv, sv), 0.25)[:, None]
        out.append(p + sv * (2.0 * grow / sq))
    return out


def _prisms(a, b, c, depth):
    """The triangular prisms of these triangles, `depth` mm deep: one row of six corners each."""
    A, B, C = _grown(a, b, c)
    d = -_unit(np.cross(B - A, C - A)) * depth
    return np.stack([A, B, C, A + d, B + d, C + d], axis=1)


def _solids(per):
    """Every prism of `per` (n, 6, 3) as a Manifold, the empty and flat ones left out."""
    vp = np.zeros((len(per), CORNERS, 5), dtype=np.float32)
    vp[:, :, :3] = per
    out = []
    for k in range(len(per)):
        m = mf.Manifold(mf.Mesh(vp[k], PRISM))
        if m.status() == mf.Error.NoError and not m.is_empty():
            out.append(m)
    return out


def _union(solids):
    """All these Manifolds in one; the prisms do not overlap yet, so the batch does the work."""
    return mf.Manifold.batch_boolean(solids, mf.OpType.Add) if solids else mf.Manifold()


def _skin(manifold, sel, a, b, c, depth):
    """The painted skin of the triangles of `sel`: their prisms, unioned and cut to the part."""
    per = _prisms(a[sel], b[sel], c[sel], depth)
    prisms = [m for m in _solids(per) if m.volume() > TINY]
    if not prisms:
        return mf.Manifold()
    got = _union(prisms)
    if got.status() != mf.Error.NoError:
        raise ValueError("the painted faces do not become volumes")
    skin = mf.Manifold.batch_boolean([got, manifold], mf.OpType.Intersect)
    return skin if skin.status() == mf.Error.NoError else mf.Manifold()


def painted_colors(tri_colors, part_color):
    """The colors painted on the faces of a part in `part_color`, as they appear: [key]."""
    pc = str(part_color).lower()
    out, seen = [], set()
    for c in tri_colors:
        if c is None:
            continue
        key = str(c).lower()
        if key != pc and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def split_painted(manifold, tri_colors, part_color, depth=1.0):
    """``[(manifold, color)]`` of a part with painted faces: the skins of every color and the rest.

    `tri_colors` is the color of every triangle of `manifold` (None where it has none), the way
    `volura.fileio.triangle_colors` reads it back. A face painted in another color than
    `part_color` prints as a skin of `depth` mm under the painted surface, in the filament of
    its own color; what is left of the part is `part_color`.

    The colors are dealt with in the order in which they appear in `tri_colors`. Where the skins
    of two of them overlap - at the edge where two painted faces meet - the color that comes
    later keeps it, so the volumes never overlap and together make up the part. Every result is
    a valid Manifold; a piece under `TINY` mm3 is left out.
    """
    order = painted_colors(tri_colors, part_color)
    if not order or manifold.is_empty():
        return [(manifold, part_color)]
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), np.cross(b - a, c - a)))
    keep = {key: np.zeros(len(t), dtype=bool) for key in order}
    for k, col in enumerate(tri_colors[:len(t)]):
        if col is not None:
            key = str(col).lower()
            if key in keep:
                keep[key][k] = True
    skins = [(key, _skin(manifold, keep[key] & (area > 1e-12), a, b, c, depth)) for key in order]
    out, claimed = [], mf.Manifold()
    for key, skin in reversed(skins):
        if not claimed.is_empty() and not skin.is_empty():
            skin = skin - claimed
        if not skin.is_empty() and skin.volume() > TINY:
            out.append((skin, key))
            claimed = claimed + skin
    rest = manifold - claimed if not claimed.is_empty() else manifold
    out.reverse()
    if rest.is_empty() or rest.volume() <= TINY:
        return out
    return out + [(rest, part_color)]
