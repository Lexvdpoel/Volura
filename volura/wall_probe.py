"""How thick the material of a part is under a point of its surface.

A wall thinner than the printer lays down is no wall at once (see thin_walls), and a wall
thicker than it needs to be prints for hours without reason. The Wall Thickness probe
answers what the wall is under the cursor, in two ways. The first is the distance straight
through the part, from the face you clicked to the face it stands opposite: the way a
caliper measures a plate. The second is the wall as the material feels it, the diameter of
the largest ball that touches the surface at that point and stays inside the part - the
thickness a medial axis gives, and the number that says how strong the spot is. The two
differ where a wall leans away from the click: straight through the side of a 3 mm plate
you measure 20 mm of plate, while the ball that fits there is 3 mm across.

The straight ray starts just under the surface, so the face that was clicked is never its
own first hit, and stops at the first triangle it meets. A ball is free while its centre
clears every triangle by at least its own radius, and the search halves the interval on
that test. A mesh that is not closed has no wall where it is open - a ray that meets
nothing returns nothing, and it is up to the caller to say so.

The distance to the triangles runs over a block of them at once, which is what makes a click
answer in a fraction of a second, also on a part with a million of them. The search for the
ball first asks which triangles it can reach at all, and measures only those exactly.
"""
from __future__ import annotations

import math

import numpy as np

from .geom import normalize

#: below this a ray and a triangle count as parallel (the determinant of Moeller-Trumbore)
EPS = 1e-9
#: mm a ray starts under the surface, so the face it starts on is never its first hit
INSIDE = 1e-4
#: a hit nearer to the start of the ray than this is that same face
NEAR = 1e-6
#: how much a radius may fall short of clearing the triangles: a ball that touches a face
#: is still inside it, and a mesh of flat triangles is never exactly round
FIT = 1e-4
#: the search for the radius stops when the interval has shrunk to this fraction of the
#: radius, never below a tenth of a micron: the answer is shown in hundredths of a millimetre
STOP = 1e-4
#: how many times the interval of the radius search is halved at most
STEPS = 40
#: how many triangles, or triangles times rays, one round of the tests holds in memory
CELLS = 200_000


def _arrays(verts, tris):
    """The mesh as ``(verts, tris, p0, e1, e2)``: the corners ``(n, 3)``, the rows of them
    that make the triangles, and the corner and the two edges of every triangle."""
    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if not len(t) or not len(v):
        return None, None, None, None, None
    p0 = v[t[:, 0]]
    return v, t, p0, v[t[:, 1]] - p0, v[t[:, 2]] - p0


def _crossings(p0, e1, e2, origin, dirs):
    """The distance of every triangle for every direction: ``(m, k)`` with inf where a ray
    misses a triangle.

    The Moeller-Trumbore test, written for m rays and k triangles at once. A ray that
    runs along a triangle makes the determinant zero: its u, v and distance are nonsense
    there, and the test on the determinant is what decides that nothing was hit."""
    pv = np.cross(dirs[:, None, :], e2[None, :, :])
    det = np.einsum("ij,mij->mi", e1, pv)
    s = origin - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        u = np.einsum("ij,mij->mi", s, pv) * inv
        qv = np.cross(s[None, :, :], e1[None, :, :])
        w = np.einsum("mj,mij->mi", dirs, qv) * inv
        dist = np.einsum("ij,mij->mi", e2, qv) * inv
        near = ((np.abs(det) > EPS) & (u >= -EPS) & (w >= -EPS)
                & (u + w <= 1.0 + EPS) & (dist > NEAR))
    return np.where(near, dist, np.inf)


def ray_hits(verts, tris, origin, direction):
    """The distance of every triangle a ray hits, nearest first.

    `verts` are the corners ``(n, 3)`` of the mesh and `tris` the rows of `verts`
    ``(m, 3)`` that make its triangles. A hit closer than :data:`NEAR` is the surface the
    ray started on, which is no hit."""
    _v, _t, p0, e1, e2 = _arrays(verts, tris)
    d = normalize(np.asarray(direction, dtype=float).reshape(3))
    if p0 is None or not np.any(d):
        return np.zeros(0)
    dist = _crossings(p0, e1, e2, np.asarray(origin, dtype=float).reshape(3), d[None, :])
    dist = dist[0]
    return np.sort(dist[np.isfinite(dist)])


def first_hits(verts, tris, origin, dirs):
    """The distance of the nearest triangle every direction in `dirs` hits, one per row,
    and inf for a ray that hits nothing.

    The triangles go by blocks of :data:`CELLS` rays: every triangle of a block is tested
    against every ray in one pass, which is what a fan of them needs to be quick, and a big
    model may have more triangles than fit in memory at once."""
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 3)
    length = np.linalg.norm(dirs, axis=1)[:, None]
    dirs = dirs / np.where(length > 1e-12, length, 1.0)     # a ray of no length hits nothing
    out = np.full(len(dirs), np.inf)
    _v, t, p0, e1, e2 = _arrays(verts, tris)
    if p0 is None or not len(dirs):
        return out
    origin = np.asarray(origin, dtype=float).reshape(3)
    step = max(1, CELLS // len(dirs))
    for s in range(0, len(t), step):
        block = slice(s, s + step)
        out = np.minimum(out, np.min(_crossings(p0[block], e1[block], e2[block],
                                                origin, dirs), axis=1))
    return out


def _divide(num, den, ok):
    """`num` divided by `den` in place, and 0.0 where `den` vanishes: an edge of no length
    and a triangle with no surface divide to nothing, which is where they lie."""
    np.copyto(num, 0.0, where=~ok)
    np.divide(num, den, out=num, where=ok)


def _reuse(work, name, shape, dtype=float):
    """The array `name` of the working set `work`, made again only when it is missing or of
    another shape."""
    box = work.get(name)
    if box is None or box.shape != shape:
        box = work[name] = np.empty(shape, dtype=dtype)
    return box


def _nearest(a, ab, ac, point, work, size):
    """The squared distance ``(m,)`` of `point` to every triangle.

    The nearest point of a triangle lies in one of seven regions around it: beyond a corner,
    beside an edge, or over the face (Ericson, Real-Time Collision Detection 5.1.5). Every
    point starts on the face, at the barycentric coordinates ``u`` and ``v`` of the two edges
    that run from ``a``, and each region below takes the triangles that belong to it to its
    corner or edge. The regions do not overlap, so the order in which they are applied makes
    no difference.

    The work happens in the arrays of `work`, which are as wide as `size` triangles and used
    up to the ``m`` of the block: a caller that asks the same question about a big part over
    and over pays for its triangles, and not for memory again every time."""
    m = len(a)
    ap = _reuse(work, "ap", (size, 3))[:m]              # from a corner to the point
    q = _reuse(work, "q", (size, 3))[:m]                # the nearest point found so far
    on = _reuse(work, "on", (size, 3))[:m]              # a corner or a point on an edge
    edge = _reuse(work, "edge", (size, 3))[:m]          # the far edge of the triangle
    d1, d2, u, v, w, g, h, par = (_reuse(work, name, (size,))[:m]
                                  for name in ("d1", "d2", "u", "v", "w", "g", "h", "par"))
    mask = _reuse(work, "mask", (size,), bool)[:m]
    here = _reuse(work, "here", (size,), bool)[:m]
    square = _reuse(work, "square", (size,))[:m]

    np.subtract(point, a, out=ap)
    np.multiply(ap, ab, out=on)
    np.sum(on, axis=1, out=d1)                          # d1, d2: how far along the two edges
    np.multiply(ap, ac, out=on)
    np.sum(on, axis=1, out=d2)

    bb = np.einsum("ij,ij->i", ab, ab)                  # the edges with themselves
    bc = np.einsum("ij,ij->i", ab, ac)                  # and with each other
    cc = np.einsum("ij,ij->i", ac, ac)
    den = bb * cc - bc * bc                             # nonzero for a triangle of surface
    ok = den != 0.0
    np.multiply(d1, cc, out=u)
    np.multiply(d2, bc, out=w)
    np.subtract(u, w, out=u)                            # u and v are the barycentric
    np.multiply(d2, bb, out=v)                          # coordinates of the face
    np.multiply(d1, bc, out=w)
    np.subtract(v, w, out=v)
    np.add(u, v, out=w)
    np.subtract(den, w, out=w)                          # negative past the edge bc

    _divide(u, den, ok)
    _divide(v, den, ok)
    np.multiply(ab, u[:, None], out=on)
    np.add(a, on, out=q)
    np.multiply(ac, v[:, None], out=on)
    np.add(q, on, out=q)

    np.logical_and(d1 <= 0.0, d2 <= 0.0, out=mask)      # the corner a
    np.copyto(q, a, where=mask[:, None])
    np.subtract(d1, bb, out=g)                          # past the corner b?
    np.subtract(d2, bc, out=h)
    np.less_equal(h, g, out=mask)
    np.greater_equal(g, 0.0, out=here)
    mask &= here
    np.add(a, ab, out=on)
    np.copyto(q, on, where=mask[:, None])
    np.subtract(d2, cc, out=h)                          # past the corner c?
    np.subtract(d1, bc, out=g)
    np.less_equal(g, h, out=mask)
    np.greater_equal(h, 0.0, out=here)
    mask &= here
    np.add(a, ac, out=on)
    np.copyto(q, on, where=mask[:, None])

    np.copyto(par, d1)                                  # beside the edge ab
    _divide(par, bb, bb != 0.0)
    np.logical_and(v <= 0.0, d1 >= 0.0, out=mask)
    np.less_equal(d1, bb, out=here)
    mask &= here
    np.multiply(ab, par[:, None], out=on)
    np.add(a, on, out=on)
    np.copyto(q, on, where=mask[:, None])

    np.copyto(par, d2)                                  # beside the edge ac
    _divide(par, cc, cc != 0.0)
    np.logical_and(u <= 0.0, d2 >= 0.0, out=mask)
    np.less_equal(d2, cc, out=here)
    mask &= here
    np.multiply(ac, par[:, None], out=on)
    np.add(a, on, out=on)
    np.copyto(q, on, where=mask[:, None])

    np.less_equal(w, 0.0, out=mask)                     # beside the edge bc
    np.subtract(d2, bc, out=g)
    np.subtract(d1, bb, out=h)
    np.subtract(g, h, out=g)
    np.greater_equal(g, 0.0, out=here)
    mask &= here
    np.subtract(d1, bc, out=h)
    np.subtract(d2, cc, out=w)
    np.subtract(h, w, out=h)
    np.greater_equal(h, 0.0, out=here)
    mask &= here
    np.add(g, h, out=par)
    _divide(g, par, par != 0.0)                         # how far along the edge bc
    np.subtract(ac, ab, out=edge)
    np.multiply(edge, g[:, None], out=edge)
    np.add(a, ab, out=on)
    np.add(on, edge, out=on)
    np.copyto(q, on, where=mask[:, None])

    np.subtract(q, point, out=ap)
    np.multiply(ap, ap, out=on)
    np.sum(on, axis=1, out=square)
    return square


def point_mesh_distance(verts, tris, points, work=None):
    """The distance of every point in `points` to the nearest triangle of the mesh.

    `points` is one point ``(3,)`` or several ``(k, 3)``, and the answer is one distance per
    point. The triangles go by blocks of :data:`CELLS`, as in :func:`first_hits`, and the
    arrays of the search may be passed in as `work` and used again: a search that asks the
    same question a hundred times about a big part then costs its triangles and not new
    memory every round."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.full(len(pts), np.inf)
    _v, t, p0, e1, e2 = _arrays(verts, tris)
    if p0 is None or not len(pts):
        return out
    work = {} if work is None else work
    for i, pt in enumerate(pts):
        best = np.inf
        for s in range(0, len(t), CELLS):
            block = slice(s, s + CELLS)
            near = _nearest(p0[block], e1[block], e2[block], pt, work, CELLS)
            best = min(best, float(np.min(near)))
        out[i] = math.sqrt(best)
    return out


def thickness(verts, tris, point, normal):
    """The wall of a mesh under a point of its surface.

    `point` lies on the surface and `normal` is the outward normal of the face there. The
    answer is ``{"along_normal": (distance, exit), "local": (diameter, centre)}``: the wall
    straight through, and the ball of material that fits at that spot - the largest ball
    that touches the surface at `point` and stays inside the part, whose diameter is the
    thickness there. An exit is the point at which the ray leaves the material again, and a
    distance runs from `point` to that exit. Both are None when the ray meets nothing: an
    open mesh or a flat sketch has no wall there to measure, and it is up to the caller to
    say so."""
    p = np.asarray(point, dtype=float).reshape(3)
    n = normalize(np.asarray(normal, dtype=float).reshape(3))
    out = {"along_normal": None, "local": None}
    if not np.any(n):
        return out
    inward = -n
    t = first_hits(verts, tris, p + inward * INSIDE, inward[None, :])[0]
    if not np.isfinite(t):
        return out
    exit_at = p + inward * (INSIDE + float(t))
    through = float(np.linalg.norm(exit_at - p))
    out["along_normal"] = (through, exit_at)

    work = {}                                   # the arrays of the distance search, once
    _v, t, p0, e1, e2 = _arrays(verts, tris)
    third = (e1 + e2) / 3.0                     # from a corner to the centre of the triangle
    cent = p0 + third                           # the triangle and the sphere around it:
    rad = np.maximum.reduce([np.linalg.norm(third, axis=1),
                             np.linalg.norm(e1 - third, axis=1),
                             np.linalg.norm(e2 - third, axis=1)])

    def free(r):
        """Whether the ball of radius `r` on the line into the part still clears the
        material: its centre may not come closer to a triangle than its own radius.

        A triangle whose sphere lies further from the centre of the ball than its radius
        cannot be touched by the ball at all, so only the triangles it can reach are
        measured; on a big part with a thin wall that is a handful of them."""
        c = p - n * r
        reach = np.linalg.norm(cent - c, axis=1) - rad
        touch = reach < r
        if not np.any(touch):
            return True
        return (point_mesh_distance(verts, t[touch], c, work)[0]
                >= r - FIT * max(1.0, r))

    # the ball cannot be thicker than the wall straight through, and on a flat wall half of
    # that wall is free; so the search starts there and halves down to the largest radius
    # that still fits - which on a wall of two flat faces is exactly half of the wall
    lo, hi = 0.0, through / 2.0
    for _ in range(STEPS):
        if hi - lo < STOP * max(hi, 1.0):
            break
        mid = (lo + hi) / 2.0
        if free(mid):
            lo = mid
        else:
            hi = mid
    out["local"] = (2.0 * lo, p - n * lo)
    return out
