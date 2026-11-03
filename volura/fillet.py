"""Filleting (rounding) and chamfering of edges between planar faces.

Look perpendicular onto an edge between two faces. The two faces cut out an
angle of alpha = 180° - gamma there, with gamma the angle between the outward
normals of the faces. That angle is the "corner" of the edge:

* at a **convex** edge (think of the top edge of a cube) the corner holds the
  material of the solid: filleting removes a wedge from it;
* at a **concave** edge (the inner corner of an L profile) the corner holds
  air: filleting fills that corner with a rounded edge.

Both cases use the same wedge and the same formulas; only the direction of the
bisector and whether the wedge is subtracted or added differ.

A chain of edges (the smooth top edge of a cylinder is built from 48 small
edges) is filleted segment by segment. The cutter of a segment extends past a
vertex it shares with a neighbouring segment, and the two profiles that meet
there are joined with a loft (the convex hull of the two profile faces, placed
side by side) so the fillet runs on and no gap or ridge is left in the corner.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import manifold3d as mf

from .geom import Plane, extrude_cs, manifold_from_mesh, rotation_matrix
from .i18n import tr

MARGIN = 0.01            # mm: cutters extend past the ends, no seams
MIN_ANGLE = math.radians(0.4)     # flatter than this, the edge is already smooth
MAX_ANGLE = math.radians(179.6)   # almost folded double: nothing to round
RAY = np.array([1237.13, 917.71, 553.37])   # a direction that never lies on an axis


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def _union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    if not ms:
        return mf.Manifold()
    if len(ms) == 1:
        return ms[0]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)


def arc_segments(segments, sweep):
    """Number of line segments for an arc of `sweep` radians on a circle of
    `segments` pieces: the cylinder of a fillet gets 16 to 32."""
    full = max(16, int(segments))
    return max(2, int(round(full * abs(float(sweep)) / (2 * math.pi))))


# ------------------------------------------------------- the corner of an edge

def corner_frame(p0, p1, n1, n2, convex=True):
    """(d, L, r1, r2, b, alpha) of the corner of an edge, or None.

    r1 and r2 are the half-lines along which the two faces point away from the edge
    and b the bisector between them; alpha is the angle between r1 and r2 (always
    less than 180°). At a convex edge the material lies inside that angle, at a
    concave edge the air."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = _unit(p1 - p0)
    if d is None:
        return None
    L = float(np.linalg.norm(p1 - p0))
    u1 = np.asarray(n1, dtype=float) - d * float(np.dot(n1, d))
    u2 = np.asarray(n2, dtype=float) - d * float(np.dot(n2, d))
    u1, u2 = _unit(u1), _unit(u2)
    if u1 is None or u2 is None:
        return None
    gamma = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
    alpha = math.pi - gamma
    if not MIN_ANGLE < alpha < MAX_ANGLE:
        return None
    b = _unit((-1.0 if convex else 1.0) * (u1 + u2))
    if b is None:
        return None
    r1, r2 = _unit(np.cross(d, u1)), _unit(np.cross(d, u2))
    if r1 is None or r2 is None:
        return None
    if float(np.dot(r1, b)) < 0:
        r1 = -r1
    if float(np.dot(r2, b)) < 0:
        r2 = -r2
    if float(np.dot(r1, r2)) > math.cos(alpha) + 1e-6:
        return None
    return d, L, r1, r2, b, alpha


def corner_profile(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True, arc=0):
    """The profile of the cutter at p0, perpendicular to the edge, as 3D points, and
    the number of arc segments used for it. None when there is nothing to do.

    For "chamfer" that is the triangle between the edge and the two points at
    distance `size` in the faces; for "fillet" the wedge between the edge, the two
    tangent points at distance size/tan(alpha/2) and the circular arc of radius `size`."""
    fr = corner_frame(p0, p1, n1, n2, convex)
    if fr is None:
        return None
    d, L, r1, r2, b, alpha = fr
    p0 = np.asarray(p0, dtype=float)
    if mode == "chamfer":
        return np.array([p0, p0 + r1 * size, p0 + r2 * size], dtype=float), 0
    T = size / math.tan(alpha / 2.0)              # edge -> tangent point
    t1, t2 = p0 + r1 * T, p0 + r2 * T
    c = p0 + b * (size / math.sin(alpha / 2.0))   # centre of the circular arc
    a1, a2 = t1 - c, t2 - c
    sweep = math.acos(float(np.clip(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)),
                                    -1.0, 1.0)))
    axis = d if float(np.dot(np.cross(a1, a2), d)) > 0 else -d
    n = int(arc) if arc else arc_segments(segments, sweep)
    R = rotation_matrix(axis, sweep / n)
    pts, v = [p0, t1], a1
    for _ in range(n - 1):
        v = R @ v
        pts.append(c + v)
    pts.append(t2)
    return np.array(pts, dtype=float), n


def corner_wedge(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True):
    """The planar cross-section polygon of the corner of an edge (3D points)."""
    pr = corner_profile(p0, p1, n1, n2, size, mode, segments, convex)
    return None if pr is None else pr[0]


def extrude_profile(p0, d, L, pts, e0=0.0, e1=0.0):
    """The prism of a profile along the edge direction, from -e0 to L + e1."""
    if pts is None or len(pts) < 3:
        return None
    pl = Plane(p0, d, u=pts[1] - pts[0])
    cs = mf.CrossSection([pl.to_local(pts)], mf.FillRule.EvenOdd)
    cut = extrude_cs(cs, pl, -MARGIN - float(e0), L + MARGIN + float(e1))
    if cut.is_empty() or cut.status() != mf.Error.NoError:
        return None
    return cut


def edge_cutter(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True, extend=0.0):
    """The solid to subtract from the volume at an edge (convex edge) or to add to it
    (concave edge). None when there is nothing to do.

    The solid sticks out MARGIN mm past both ends of the edge, so cutting leaves no
    seams; `extend` (a number or (before, after)) extends it further, so the cutters
    of a chain of edges overlap each other."""
    pr = corner_profile(p0, p1, n1, n2, size, mode, segments, convex)
    if pr is None:
        return None
    pts, _arc = pr
    d = _unit(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    if d is None:
        return None
    L = float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))
    e0, e1 = (extend, extend) if np.isscalar(extend) else tuple(extend)
    return extrude_profile(p0, d, L, pts, e0, e1)


# ------------------------------------------------- joining two segments

def profile_at(pts, d, L, at_start):
    """The same profile, but at the other end of the edge."""
    return np.asarray(pts, dtype=float) if at_start else np.asarray(pts, dtype=float) + d * L


def loft(pa, pb):
    """The solid between two profiles at the same vertex: the convex hull of the
    two profile faces, matched point by point. That fills the corner between two
    consecutive segments of a chain without removing more than the fillet itself."""
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    n = len(pa)
    if n < 3 or len(pb) != n:
        return None
    verts = np.vstack([pa, pb])
    tris = []
    for i in range(1, n - 1):                     # the two end caps (fan from the corner)
        tris.append((0, i + 1, i))
        tris.append((n, n + i, n + i + 1))
    for i in range(n):                            # the side surface
        j = (i + 1) % n
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    tris = np.array(tris, dtype=np.int64)
    try:
        m = manifold_from_mesh(verts, tris)
        if m.is_empty() or m.status() != mf.Error.NoError or m.volume() < 0:
            m = manifold_from_mesh(verts, tris[:, ::-1])     # reversed winding
    except Exception:
        return None
    if m.is_empty() or m.status() != mf.Error.NoError or m.volume() <= 0:
        return None
    return m


# ------------------------------------------------------- convex or concave

def _inside(man, q, far):
    hits = man.ray_cast(tuple(np.asarray(q, dtype=float)),
                        tuple(np.asarray(q, dtype=float) + far))
    return len(hits) % 2 == 1


def is_convex(man, p0, p1, n1, n2, size=1.0):
    """Convex or concave: the two "sides" of an edge are air at a convex edge and
    material at a concave edge. The faces of the edge do not touch those sides, so
    two samples tell; if only one is inside, the edge is convex."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = _unit(p1 - p0)
    if d is None:
        return True
    w = np.asarray(n1, dtype=float) - np.asarray(n2, dtype=float)
    w = _unit(w - d * float(np.dot(w, d)))
    if w is None:
        return True
    bb = np.asarray(man.bounding_box(), dtype=float).reshape(-1)
    span = bb[3:] - bb[:3] if bb.size == 6 else np.zeros(3)
    far = RAY * max(1.0, float(np.linalg.norm(span)))
    mid = 0.5 * (p0 + p1)
    step = float(np.clip(min(size, float(np.linalg.norm(p1 - p0))) * 0.1, 0.02, 2.0))
    inside = int(_inside(man, mid + w * step, far)) + int(_inside(man, mid - w * step, far))
    return inside != 2


# ------------------------------------------------------------------ applying

def apply(man, edges, size, mode="fillet", segments=64, join=True):
    """Fillet or chamfer the edges `edges` = [(p0, p1, n1, n2)]; returns a new
    manifold. Convex edges are subtracted, concave ones added.

    Edges that share a vertex form a chain. At such a vertex the solid between the
    two profiles is added, so the fillet runs along the whole chain and no ridge is
    left in the corner (`join=False` skips that)."""
    size = float(size)
    if size <= 0:
        raise ValueError(tr("the size must be greater than 0"))
    items = []
    for (p0, p1, n1, n2) in edges:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        n1 = np.asarray(n1, dtype=float)
        n2 = np.asarray(n2, dtype=float)
        convex = is_convex(man, p0, p1, n1, n2, size)
        fr = corner_frame(p0, p1, n1, n2, convex)
        if fr is None:
            continue
        _d, _L, _r1, _r2, _b, alpha = fr
        items.append({"p0": p0, "p1": p1, "n1": n1, "n2": n2, "convex": convex,
                      "d": fr[0], "L": fr[1], "alpha": alpha})
    if not items:
        return man
    at = defaultdict(list)
    for i, it in enumerate(items):
        at[_vkey(it["p0"])].append(i)
        at[_vkey(it["p1"])].append(i)
    cutters = []
    for it in items:
        cut = edge_cutter(it["p0"], it["p1"], it["n1"], it["n2"], size, mode,
                          segments, it["convex"])
        cutters.append(None if cut is None or cut.volume() <= 0 else (cut, it["convex"]))
    if not any(c is not None for c in cutters):
        return man
    if join:
        for key, idxs in at.items():
            for i, j in combinations(sorted(set(idxs)), 2):
                if cutters[i] is None or cutters[j] is None:
                    continue
                if items[i]["convex"] != items[j]["convex"]:
                    continue
                jnt = joint(items[i], items[j], key, mode, size, segments)
                if jnt is not None:
                    cutters.append((jnt, items[i]["convex"]))
    sub = [c for c, convex in cutters if convex]
    add = [c for c, convex in cutters if not convex]
    res = man
    if sub:
        res = res - _union(sub)
    if add:
        res = res + _union(add)
    return res


def joint(a, b, key, mode, size, segments):
    """The solid that joins the cutters of two edges meeting at the same vertex, so
    the fillet runs on there."""
    pa, na = corner_profile(a["p0"], a["p1"], a["n1"], a["n2"], size, mode,
                            segments, a["convex"])
    pb, nb = corner_profile(b["p0"], b["p1"], b["n1"], b["n2"], size, mode,
                            segments, b["convex"])
    if pa is None or pb is None:
        return None
    if na != nb:                                  # joining needs the same sampling
        n = min(na, nb)
        pa, _ = corner_profile(a["p0"], a["p1"], a["n1"], a["n2"], size, mode, segments,
                               a["convex"], n)
        pb, _ = corner_profile(b["p0"], b["p1"], b["n1"], b["n2"], size, mode, segments,
                               b["convex"], n)
    pa = profile_at(pa, a["d"], a["L"], _vkey(a["p0"]) == key)
    pb = profile_at(pb, b["d"], b["L"], _vkey(b["p0"]) == key)
    return loft(pa, pb)


def _vkey(p):
    return tuple(np.round(np.asarray(p, dtype=float), 3))


# --------------------------------------------------------- edges of a model

def edge_normals(ents, eid):
    """(p0, p1, n1, n2) of an edge with exactly two faces, or None."""
    if eid not in ents.E:
        return None
    fs = sorted(f for f in ents.e2f.get(eid, ()) if f in ents.F)
    if len(fs) != 2:
        return None
    p0, p1 = ents.edge_points(eid)
    return (np.asarray(p0, dtype=float), np.asarray(p1, dtype=float),
            ents.F[fs[0]].normal.copy(), ents.F[fs[1]].normal.copy())


def edge_data(ents, eids):
    """(list of (eid, p0, p1, n1, n2), list of edges that were skipped)."""
    ok, bad = [], []
    for eid in eids:
        d = edge_normals(ents, eid)
        if d is None:
            if eid in ents.E:
                bad.append(eid)
            continue
        ok.append((eid,) + d)
    return ok, bad


def _continues(ents, e, w, shared, cos_lim):
    """Does edge `w` continue smoothly where edge `e` ends: they share face `shared`
    and the two faces behind them are nearly the same plane (the edges of one rounding)."""
    if e in ents.soft and w in ents.soft:
        return True
    ne = [ents.F[g].normal for g in ents.e2f.get(e, ()) if g != shared and g in ents.F]
    nw = [ents.F[g].normal for g in ents.e2f.get(w, ()) if g != shared and g in ents.F]
    if not ne or not nw:
        return False
    return any(abs(float(np.dot(a, b))) > cos_lim for a in ne for b in nw)


def soft_chain(ents, eid, angle=20.0):
    """All connected smooth edges that form one chain with `eid`: the wall of a
    cylinder, for example, is built from 48 small edges that together are one
    rounding. A right angle (cube) stops the chain.

    Two smooth edges need not touch at a vertex: on a cylinder the two seams of a
    wall face run right opposite each other, yet they are the same rounding. Edges
    that are not smooth are only chained when they touch at a vertex and the faces
    behind them are the same plane."""
    if eid not in ents.E:
        return []
    cos_lim = math.cos(math.radians(float(angle)))
    seen, stack = {eid}, [eid]
    while stack:
        e = stack.pop()
        for f in ents.e2f.get(e, ()):
            soft_face = e in ents.soft
            for w in ents.face_edges(ents.F[f]):
                if w in seen or w not in ents.E:
                    continue
                if not (soft_face and w in ents.soft) and not set(ents.E[w]) & set(ents.E[e]):
                    continue
                if _continues(ents, e, w, f, cos_lim):
                    seen.add(w)
                    stack.append(w)
    return sorted(seen)


smooth_chain = soft_chain        # the same chain, another name
