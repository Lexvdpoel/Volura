"""Geometry helpers: planes, 2D cross-sections and mesh analysis.

All solids in Volura are stored as ``manifold3d.Manifold`` objects, which are
guaranteed to be closed, oriented 2-manifolds. That is what makes every STL /
3MF export watertight. This module adds the "SketchUp view" on top of them:
planar faces (coplanar triangle regions), visible edges and inference points.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import manifold3d as mf

AXES = np.eye(3)
SPLIT_WIDTH = 1e-3  # half width of the cut strip used when a line splits a face


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def default_u(n):
    """A stable in-plane x-axis: world X for horizontal planes, horizontal otherwise."""
    if abs(n[2]) > 0.999:
        return np.array([1.0, 0.0, 0.0])
    return normalize(np.cross([0.0, 0.0, 1.0], n))


class Plane:
    """An oriented plane with a right handed local frame (u, v, n)."""

    __slots__ = ("origin", "n", "u", "v")

    def __init__(self, origin, normal, u=None):
        self.origin = np.asarray(origin, dtype=float).copy()
        self.n = normalize(normal)
        if u is None:
            u = default_u(self.n)
        u = np.asarray(u, dtype=float)
        u = u - self.n * np.dot(u, self.n)
        if np.linalg.norm(u) < 1e-9:
            u = default_u(self.n)
        self.u = normalize(u)
        self.v = np.cross(self.n, self.u)

    def to_local(self, p):
        d = np.asarray(p, dtype=float) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    def to_world(self, xy):
        xy = np.asarray(xy, dtype=float)
        return self.origin + xy[..., :1] * self.u + xy[..., 1:2] * self.v

    def matrix(self):
        """3x4 matrix mapping local (x, y, z) to world coordinates."""
        return np.column_stack([self.u, self.v, self.n, self.origin])

    def distance(self, p):
        return float(np.dot(np.asarray(p, dtype=float) - self.origin, self.n))

    def project(self, p):
        p = np.asarray(p, dtype=float)
        return p - self.n * self.distance(p)

    def intersect_ray(self, o, d, allow_behind=False):
        den = float(np.dot(d, self.n))
        if abs(den) < 1e-12:
            return None
        t = float(np.dot(self.origin - o, self.n)) / den
        if t < 0 and not allow_behind:
            return None
        return o + d * t

    def coplanar(self, other, tol=1e-4):
        return (abs(abs(float(np.dot(self.n, other.n))) - 1.0) < 1e-7
                and abs(self.distance(other.origin)) < tol)

    def transformed(self, M3, t):
        M3 = np.asarray(M3, dtype=float)
        n = np.linalg.inv(M3).T @ self.n
        return Plane(M3 @ self.origin + t, n, M3 @ self.u)

    def to_dict(self):
        return {"origin": self.origin.tolist(), "n": self.n.tolist(), "u": self.u.tolist()}

    @staticmethod
    def from_dict(d):
        return Plane(d["origin"], d["n"], d["u"])


GROUND = Plane((0, 0, 0), (0, 0, 1))


# --------------------------------------------------------------------------
# 2D cross-sections
# --------------------------------------------------------------------------

def cs_from_loops(loops, fill=mf.FillRule.EvenOdd):
    loops = [np.asarray(l, dtype=float) for l in loops if len(l) >= 3]
    if not loops:
        return mf.CrossSection()
    return mf.CrossSection(loops, fill)


def cs_polygons(cs):
    return [np.asarray(p, dtype=float) for p in cs.to_polygons()]


def triangulate_cs(cs):
    """Return (points2d (n,2), triangles (m,3)) for a cross-section."""
    polys = cs_polygons(cs)
    if not polys:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    tris = np.asarray(mf.triangulate(polys), dtype=np.int64).reshape(-1, 3)
    return np.concatenate(polys), tris


def point_in_polys(polys, p):
    """Even-odd point in polygon test for a list of closed polygons.

    The crossing test is the same ray-parity rule as before (count the edges that
    cross the horizontal ray to the left of `p`); it is evaluated per edge in plain
    Python because these polygons are small and the numpy version spent all its
    time in the overhead of ``np.roll`` on a handful of points."""
    x, y = float(p[0]), float(p[1])
    inside = False
    for poly in polys:
        pts = poly.tolist() if hasattr(poly, "tolist") else [tuple(q) for q in poly]
        n = len(pts)
        cnt = 0
        for i in range(n):
            xi, yi = pts[i][0], pts[i][1]
            xj, yj = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
            if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
                cnt += 1
        inside ^= (cnt & 1)
    return inside


def polyline_strip(points2d, half_width=SPLIT_WIDTH):
    """A thin cross-section covering an open polyline (used to cut faces)."""
    parts = []
    for a, b in zip(points2d[:-1], points2d[1:]):
        d = np.asarray(b) - np.asarray(a)
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        d /= L
        nrm = np.array([-d[1], d[0]]) * half_width
        a2 = a - d * half_width
        b2 = b + d * half_width
        parts.append(mf.CrossSection([np.array([a2 - nrm, b2 - nrm, b2 + nrm, a2 + nrm])]))
    if not parts:
        return mf.CrossSection()
    return mf.CrossSection.batch_boolean(parts, mf.OpType.Add)


def split_cs(cs, points2d):
    """Split a cross-section along an open polyline. Returns the pieces."""
    strip = polyline_strip(points2d)
    pieces = (cs - strip).decompose()
    if len(pieces) < 2:
        return [cs]
    out = []
    for p in pieces:
        grown = p.offset(SPLIT_WIDTH, mf.JoinType.Miter) ^ cs
        if grown.area() > 1e-6:
            out.append(grown)
    return out


def extrude_cs(cs, plane, z0, z1):
    """Extrude a cross-section (in plane coordinates) between local z0 and z1."""
    if z1 < z0:
        z0, z1 = z1, z0
    if z1 - z0 < 1e-9 or cs.is_empty():
        return mf.Manifold()
    m = mf.Manifold.extrude(cs, z1 - z0).translate((0.0, 0.0, z0))
    return m.transform(plane.matrix())


def circle_points(radius, segments, start_angle=0.0):
    a = start_angle + np.arange(segments) * (2 * math.pi / segments)
    return np.column_stack([np.cos(a) * radius, np.sin(a) * radius])


# --------------------------------------------------------------------------
# Arcs (the SketchUp arc tools)
# --------------------------------------------------------------------------

def arc_polyline(center, radius, u, v, sweep, segments):
    """`segments` segments of a circle around `center`, from u over sweep radians."""
    n = max(1, int(round(segments)))
    a = np.linspace(0.0, float(sweep), n + 1)
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.asarray(center, dtype=float) + radius * (np.cos(a)[:, None] * u +
                                                       np.sin(a)[:, None] * v)


def arc_from_center(center, start, angle_rad, plane, segments=12):
    """Arc around `center` starting at `start`, over `angle_rad` in `plane` (n+1 points)."""
    c, s = plane.project(center), plane.project(start)
    d = s - c
    r = float(np.linalg.norm(d))
    if r < 1e-9:
        return np.array([c, c])
    u = d / r
    return arc_polyline(c, r, u, np.cross(plane.n, u), angle_rad, segments)


def arc_through_3(p0, p_mid, p1, plane, segments=12):
    """The arc through three points in `plane`; collinear points give the chord."""
    a, m, b = (plane.project(p) for p in (p0, p_mid, p1))
    A, M, B = (plane.to_local(p) for p in (a, m, b))
    N = np.array([2 * (M - A), 2 * (B - A)], dtype=float)
    rhs = np.array([float(M @ M - A @ A), float(B @ B - A @ A)])
    det = float(N[0, 0] * N[1, 1] - N[0, 1] * N[1, 0])
    scale = max(1.0, float(np.abs(np.concatenate([A, M, B])).max()))
    if abs(det) < 1e-9 * scale * scale:
        return np.array([a, b])  # on one line: no arc
    c = plane.to_world(np.linalg.solve(N, rhs))
    e0, e1 = a - c, b - c
    r = float(np.linalg.norm(e0))
    if r < 1e-9:
        return np.array([a, b])
    e0, e1 = e0 / r, e1 / r
    sweep = math.atan2(float(np.cross(e0, e1) @ plane.n), float(e0 @ e1))
    turn = float(np.dot(np.cross(m - a, b - m), plane.n))
    if turn > 0 > sweep or turn < 0 < sweep:
        sweep += 2 * math.pi if turn > 0 else -2 * math.pi
    return arc_polyline(c, r, e0, np.cross(plane.n, e0), sweep, segments)


def arc_from_bulge(p0, p1, bulge_point, plane, segments=12):
    """Arc from p0 to p1 whose top lies above `bulge_point` (n+1 points).

    The bulge is symmetric about the chord, like the "Arc from 2 Points" tool:
    only the height of `bulge_point` above the chord in `plane` is used."""
    a, b = plane.project(p0), plane.project(p1)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.array([a, a])
    w = np.cross(plane.n, d / L)
    h = float(np.dot(plane.project(bulge_point) - 0.5 * (a + b), w))
    if abs(h) < 1e-9:
        return np.array([a, b])  # no bulge: a straight line
    return arc_through_3(a, 0.5 * (a + b) + w * h, b, plane, segments)


# --------------------------------------------------------------------------
# Freehand strokes (the "Uit de vrije hand" tool)
# --------------------------------------------------------------------------

def simplify_polyline(points, tol, min_len=None):
    """Simplify a drawn stroke like SketchUp's freehand tool does on mouse release.

    Points that deviate less than `tol` from the line between the points that are
    kept disappear (Douglas-Peucker), after which segments shorter than `min_len`
    (by default the same tolerance) are collapsed. End points are always kept.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 3:
        return pts.copy() if pts.ndim == 2 else np.zeros((0, 3))
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        L = float(np.linalg.norm(d))
        seg = pts[i + 1:j]
        if L > 1e-12:
            w = d / L
            rel = seg - a
            off = np.linalg.norm(rel - w * np.einsum("ij,j->i", rel, w)[:, None], axis=1)
        else:
            off = np.linalg.norm(seg - a, axis=1)
        k = int(np.argmax(off))
        if off[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    out = pts[keep]
    if min_len is None:
        min_len = tol
    if min_len > 0 and len(out) > 2:
        res = [out[0]]
        for p in out[1:-1]:
            if float(np.linalg.norm(p - res[-1])) >= min_len:
                res.append(p)
        if len(res) > 1 and float(np.linalg.norm(out[-1] - res[-1])) < min_len:
            res.pop()
        res.append(out[-1])
        out = np.asarray(res, dtype=float)
    return out


# --------------------------------------------------------------------------
# Rotated rectangle (the "Gedraaide rechthoek" tool)
# --------------------------------------------------------------------------

def rotated_rect(p0, p1, width, plane):
    """The four corners of a rotated rectangle: p0-p1 is the first side and `width`
    the perpendicular distance to the opposite side, both measured in `plane`.

    A positive width lies on the side of `plane.n` turned left from p0 to p1; the
    loop keeps that order, so it is counter clockwise around `plane.n`.
    """
    a, b = plane.project(p0), plane.project(p1)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.zeros((0, 3))
    w = np.cross(plane.n, d / L) * float(width)
    return np.array([a, b, b + w, a + w])


# --------------------------------------------------------------------------
# Mesh analysis
# --------------------------------------------------------------------------

class MeshData:
    """Derived data of a manifold used for display, picking and inference."""

    SOFT_ANGLE = 20.0  # edges with a smaller fold angle are hidden (softened)

    def __init__(self, man: mf.Manifold):
        mesh = man.to_mesh64()
        V = np.asarray(mesh.vert_properties, dtype=float)[:, :3].copy()
        T = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3).copy()
        self.verts, self.tris = V, T
        nt = len(T)
        p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        cr = np.cross(p1 - p0, p2 - p0)
        area2 = np.linalg.norm(cr, axis=1)
        self.normals = cr / np.maximum(area2, 1e-30)[:, None]
        self.tri_d = np.einsum("ij,ij->i", self.normals, p0)
        self.p0, self.e1, self.e2 = p0, p1 - p0, p2 - p0
        if nt:
            self.bbox = np.array([V.min(axis=0), V.max(axis=0)])
        else:
            self.bbox = np.zeros((2, 3))

        # --- triangle adjacency through shared edges
        ea = T.ravel()
        eb = T[:, [1, 2, 0]].ravel()
        etri = np.repeat(np.arange(nt), 3)
        lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
        key = lo * (len(V) + 1) + hi
        order = np.argsort(key, kind="stable")
        ks = key[order]
        idx = np.nonzero(ks[:-1] == ks[1:])[0] if nt else np.zeros(0, dtype=np.int64)
        ta, tb = etri[order[idx]], etri[order[idx + 1]]
        elo, ehi = lo[order[idx]], hi[order[idx]]

        # --- coplanar regions (SketchUp "faces")
        N = self.normals
        cosang = np.einsum("ij,ij->i", N[ta], N[tb]) if len(ta) else np.zeros(0)
        scale = max(1.0, float(np.abs(self.bbox).max()))
        same = (cosang > 1 - 1e-7) & (np.abs(self.tri_d[ta] - self.tri_d[tb]) < 1e-6 * scale)
        parent = list(range(nt))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in zip(ta[same].tolist(), tb[same].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(nt)], dtype=np.int64)
        uniq, region = np.unique(roots, return_inverse=True)
        self.region_of_tri = region
        self.region_count = len(uniq)
        order_r = np.argsort(region, kind="stable")
        bounds = np.searchsorted(region[order_r], np.arange(len(uniq) + 1))
        self.region_tris = [order_r[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]

        # --- visible (hard) edges between regions
        diff = region[ta] != region[tb]
        hard = diff & (cosang < math.cos(math.radians(self.SOFT_ANGLE)))
        self.edges = self._merge_collinear(elo[hard], ehi[hard])
        if len(self.edges):
            self.corners = np.unique(self.edges.reshape(-1, 3), axis=0)
            self.midpoints = self.edges.mean(axis=1)
        else:
            self.corners = np.zeros((0, 3))
            self.midpoints = np.zeros((0, 3))

        # --- render buffers
        self.tri_positions = V[T].reshape(-1, 3).astype(np.float32)
        self.tri_normals = np.repeat(N, 3, axis=0).astype(np.float32)
        self.edge_positions = self.edges.reshape(-1, 3).astype(np.float32)
        self._loops = {}

    def _merge_collinear(self, a, b):
        """Join hard edge segments that continue in a straight line."""
        V = self.verts
        if len(a) == 0:
            return np.zeros((0, 2, 3))
        adj = defaultdict(list)
        for i, j in zip(a.tolist(), b.tolist()):
            adj[i].append(j)
            adj[j].append(i)

        def passthrough(v):
            nb = adj[v]
            if len(nb) != 2:
                return False
            d1 = normalize(V[nb[0]] - V[v])
            d2 = normalize(V[nb[1]] - V[v])
            return float(np.dot(d1, d2)) < -0.99999

        segs, seen = [], set()
        for i, j in zip(a.tolist(), b.tolist()):
            if (i, j) in seen or (j, i) in seen:
                continue
            seen.add((i, j))
            ends = []
            for start, nxt in ((i, j), (j, i)):
                prev, cur = start, nxt
                while passthrough(cur):
                    n0, n1 = adj[cur]
                    step = n1 if n0 == prev else n0
                    seen.add((cur, step))
                    prev, cur = cur, step
                    if cur in (i, j):
                        break
                ends.append(cur)
            segs.append((V[ends[1]], V[ends[0]]))
        return np.array(segs, dtype=float).reshape(-1, 2, 3)

    # ------------------------------------------------------------------
    def ray_cast(self, o, d):
        """Nearest hit (t, triangle) of a ray, or None."""
        if not len(self.tris):
            return None
        e1, e2 = self.e1, self.e2
        pvec = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tvec = o - self.p0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            qvec = np.cross(tvec, e1)
            v = (qvec @ d) * inv
            t = np.einsum("ij,ij->i", e2, qvec) * inv
        ok = (np.abs(det) > 1e-14) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
        if not ok.any():
            return None
        cand = np.nonzero(ok)[0]
        k = cand[np.argmin(t[cand])]
        return float(t[k]), int(k)

    def region_plane(self, rid):
        tri = self.region_tris[rid][0]
        return Plane(self.verts[self.tris[tri, 0]], self.normals[tri])

    def region_loops(self, rid):
        """Boundary loops (world coordinates) of a planar region."""
        if rid in self._loops:
            return self._loops[rid]
        T = self.tris[self.region_tris[rid]]
        a = T.ravel().tolist()
        b = T[:, [1, 2, 0]].ravel().tolist()
        directed = set(zip(a, b))
        nxt = defaultdict(list)
        boundary = []
        for x, y in zip(a, b):
            if (y, x) not in directed:
                nxt[x].append(y)
                boundary.append((x, y))
        used, loops = set(), []
        for x, y in boundary:
            if (x, y) in used:
                continue
            used.add((x, y))
            loop, cur = [x], y
            guard = 0
            while cur != x and guard < len(boundary) + 2:
                guard += 1
                cands = [c for c in nxt[cur] if (cur, c) not in used]
                if not cands:
                    break
                used.add((cur, cands[0]))
                loop.append(cur)
                cur = cands[0]
            if len(loop) >= 3:
                loops.append(self.verts[loop])
        self._loops[rid] = loops
        return loops

    def region_cs(self, rid, plane):
        return cs_from_loops([plane.to_local(L) for L in self.region_loops(rid)])

    def region_positions(self, rid):
        return self.verts[self.tris[self.region_tris[rid]]].reshape(-1, 3).astype(np.float32)


def manifold_from_mesh(verts, tris):
    """Build a Manifold from a triangle soup, merging coincident vertices."""
    verts = np.asarray(verts, dtype=float).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    scale = max(1.0, float(np.abs(verts).max())) if len(verts) else 1.0
    q = np.round(verts / (scale * 1e-7)).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    verts = verts[first]
    tris = inverse[tris]
    good = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[good]
    mesh = mf.Mesh64(vert_properties=np.ascontiguousarray(verts, dtype=np.float64),
                     tri_verts=np.ascontiguousarray(tris, dtype=np.uint64))
    return mf.Manifold(mesh)


def mesh_arrays(man):
    mesh = man.to_mesh64()
    return (np.asarray(mesh.vert_properties, dtype=float)[:, :3].copy(),
            np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3).copy())


def rotation_matrix(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def affine(M3, t):
    return np.column_stack([np.asarray(M3, dtype=float), np.asarray(t, dtype=float)])


def closest_on_line_to_ray(p0, d, ro, rd):
    """Parameter s so that p0 + s*d is closest to the ray (ro, rd)."""
    w = p0 - ro
    a, b, c = np.dot(d, d), np.dot(d, rd), np.dot(rd, rd)
    dd, e = np.dot(d, w), np.dot(rd, w)
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None
    return float((b * e - c * dd) / den)
