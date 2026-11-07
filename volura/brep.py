"""SketchUp style geometry: vertices, edges and planar faces.

The model behaves like SketchUp's "sticky" geometry:

* edges split each other where they touch or cross,
* a closed planar loop of edges becomes a face,
* a line across a face splits it, erasing that line heals the face again,
* erasing an edge removes the faces that need it,
* push/pull extrudes a face and extends or cancels coplanar neighbours.

Faces are planar polygons (outer loop + holes) referring to vertex ids. All
faces of a plane are (re)computed from the edges lying in that plane by a
planar arrangement; a "coverage" list says which regions should be faces.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import manifold3d as mf

from .geom import MeshData, Plane, normalize, point_in_polys

TOL = 1e-4          # mm: points closer than this are the same vertex
CELL = 1e-3
ECELL = 1.0         # mm: coarse cell of the edge/vertex index used by add_segment
EGRID_CAP = 512     # an edge spanning more cells than this goes to a linear list
PLANAR_TOL = 1e-6   # mm: how flat the points of a face must be
SOFT_ANGLE = 20.0   # degrees: smaller folds made by push/pull are softened


def _loops_key(loops):
    """A rotation-invariant key for a face's loops (a tuple of vertex loops).

    A planar arrangement walks a cycle from whichever vertex it happens to reach
    first, so the same loop comes back as a rotation of what a face stores
    (``(1, 6, 4, 2)`` vs ``(6, 4, 2, 1)``). Rotating every loop to start at its
    smallest vertex id makes the comparison of a region with an existing face -
    and the region key of the match cache - independent of that choice, while the
    direction (which sets the normal) is kept, so a reversed loop stays distinct."""
    out = []
    for l in loops:
        t = tuple(l)
        if t:
            m = min(range(len(t)), key=lambda i: t[i])
            t = t[m:] + t[:m]
        out.append(t)
    return tuple(out)


class Face:
    __slots__ = ("loops", "normal", "color")

    def __init__(self, loops, normal, color=None):
        self.loops = tuple(tuple(int(v) for v in l) for l in loops)
        self.normal = np.asarray(normal, dtype=float)
        self.color = color

    def replaced(self, **kw):
        f = Face(kw.get("loops", self.loops), kw.get("normal", self.normal), kw.get("color", self.color))
        return f


def _face(loops, normal, color=None):
    """A ``Face`` whose loops are already a tuple of tuples of vertex ids.

    ``Face.__init__`` rebuilds the loops, so that a caller can hand it lists or
    numpy indices; ``from_manifold`` makes the loops of a whole mesh in one go and
    gives the faces what it made, instead of walking every id a second time."""
    f = Face((), normal, color)
    f.loops = loops
    return f


def _ekey(a, b):
    return (a, b) if a < b else (b, a)


TRI_TRIS = np.array([[0, 1, 2]], dtype=np.int64)     # the one triangulation of a triangle
TRI_TRIS.flags.writeable = False        # shared by every triangular face


def _triangle(V, loop, normal):
    """The single triangle of a three vertex face, or None when the general way has to do it.

    A triangle has only one triangulation, so ear clipping (and the plane it projects on)
    is only needed for faces with more vertices. That is only the same answer as
    ``mf.triangulate`` gives for a loop that runs counter clockwise around the face normal
    and stands out of it far enough: ``Plane.to_local`` keeps that winding, and the
    triangulation of a counter clockwise triangle is the identity, ``[[0, 1, 2]]`` (checked
    against manifold on thousands of random ones). A clockwise loop comes out as one of
    the three rotations of those indices and a flat loop as something else altogether, so
    all of those keep the general way. Read-only result: ``TRI_TRIS`` is shared by every
    triangular face."""
    pa, pb, pc = V[loop[0]], V[loop[1]], V[loop[2]]
    ax, ay, az = float(pa[0]), float(pa[1]), float(pa[2])
    ux, uy, uz = float(pb[0]) - ax, float(pb[1]) - ay, float(pb[2]) - az
    vx, vy, vz = float(pc[0]) - ax, float(pc[1]) - ay, float(pc[2]) - az
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    nn = cx * cx + cy * cy + cz * cz
    if nn <= 0.0:
        return None                                        # repeated or collinear vertex
    if nn <= 1e-24 * ((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz)):
        return None                                        # a sliver: another order there too
    if cx * float(normal[0]) + cy * float(normal[1]) + cz * float(normal[2]) <= 0.0:
        return None                                        # not counter clockwise around it
    out = np.empty((1, 3, 3))
    out[0] = (pa, pb, pc)
    return out


def _point_cells(P, size):
    """The cell of the `size` grid that holds every point of P, as (n, 3) integers."""
    return np.floor(P / size).astype(np.int64)


def _cell_groups(C, payload):
    """(cell, rows) for the cells of C: every row of `payload` under its own cell.

    The rows of a cell keep the order they came in, which is the order the
    incremental inserts of ``vertex`` and ``_einsert`` would have made. Everything is
    laid out in one go before the first cell is handed out, so a cell is a slice of
    the rows instead of a small loop of its own."""
    if not len(C):
        return
    o = np.lexsort((C[:, 2], C[:, 1], C[:, 0]))
    Cs = C[o]
    cut = (np.flatnonzero((Cs[1:, 0] != Cs[:-1, 0]) | (Cs[1:, 1] != Cs[:-1, 1]) |
                          (Cs[1:, 2] != Cs[:-1, 2])) + 1).tolist()
    rows = [payload[i] for i in o.tolist()]
    cells = list(map(tuple, Cs.tolist()))
    for s, e in zip([0] + cut, cut + [len(rows)]):
        yield cells[s], rows[s:e]


def _edge_cells(lo, hi, size, cap):
    """The cells of the `size` grid that cover the bounding boxes of the edges.

    Returns (cells, rows, big): `cells` holds one cell per row of `rows`, which is
    the edge that cell came from. Edges that span more than `cap` cells are not in
    the grid at all; `big` marks them so the caller can keep them in the linear list
    next to the grid."""
    c0 = np.floor(lo / size).astype(np.int64)
    c1 = np.floor(hi / size).astype(np.int64)
    ni = c1[:, 0] - c0[:, 0] + 1
    nj = c1[:, 1] - c0[:, 1] + 1
    nk = c1[:, 2] - c0[:, 2] + 1
    n = ni * nj * nk
    big = n > cap
    keep = ~big
    if not keep.any():
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64), big
    rows = np.flatnonzero(keep)
    n = n[keep]
    # the cell of an edge runs (i, j, k) with i slowest: count up from its first cell
    r = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    nkr = np.repeat(nk[keep], n)
    njr = np.repeat(nj[keep], n)
    C = np.empty((len(r), 3), np.int64)
    C[:, 0] = r // (nkr * njr) + np.repeat(c0[keep, 0], n)
    C[:, 1] = (r // nkr) % njr + np.repeat(c0[keep, 1], n)
    C[:, 2] = r % nkr + np.repeat(c0[keep, 2], n)
    return C, np.repeat(rows, n), big


def _merge_within_tol(pts):
    """(rep, kept): the point every point joins, and which points are vertices.

    The same walk as calling ``find_vertex`` on every point in order: a point takes
    the closest point before it that is still a vertex and within TOL (`rep` points
    at it, `kept` is False there), and otherwise becomes a vertex itself. Only
    points in the same or a neighbouring cell of a grid with cells of at least TOL
    can qualify, so the candidate pairs come from 27 lookups in the points sorted by
    cell. (Sorting on x alone made every pair on a plane of constant x a candidate:
    quadratic for a cylinder along X.) For a manifold, whose vertices are already
    merged, a point finds next to nothing but itself."""
    n = len(pts)
    kept = np.ones(n, dtype=bool)
    rep = np.arange(n)
    if n < 2:
        return rep, kept
    lo = pts.min(axis=0)
    # cells of TOL, or larger for a big model so the cell key fits in an int64
    size = max(TOL, float((pts.max(axis=0) - lo).max()) / 2e6)
    C = np.floor((pts - lo) / size).astype(np.int64) + 1
    ny, nz = int(C[:, 1].max()) + 2, int(C[:, 2].max()) + 2
    key = (C[:, 0] * ny + C[:, 1]) * nz + C[:, 2]
    o = np.argsort(key, kind="stable")
    ks = key[o]
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = key + (dx * ny + dy) * nz + dz
                a = np.searchsorted(ks, nk, "left")
                cnt = np.searchsorted(ks, nk, "right") - a
                if not cnt.any():
                    continue
                r = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                I.append(np.repeat(np.arange(n), cnt))
                J.append(o[np.repeat(a, cnt) + r])
    i, j = np.concatenate(I), np.concatenate(J)
    not_self = i != j
    i, j = i[not_self], j[not_self]
    d = np.sqrt(((pts[i] - pts[j]) ** 2).sum(axis=1))
    near = d < TOL
    i, j, d = i[near], j[near], d[near]
    order = np.lexsort((j, i))               # points in order, closest candidate first
    cur, best, bd = -1, -1, TOL
    for c, q, dd in zip(i[order].tolist(), j[order].tolist(), d[order].tolist()):
        if c != cur:
            if best >= 0:
                kept[cur] = False
                rep[cur] = best
            cur, best, bd = c, -1, TOL
        if q < c and kept[q] and dd < bd:
            best, bd = q, dd
    if best >= 0:
        kept[cur] = False
        rep[cur] = best
    return rep, kept


def _fold_edges(edge_of_pair, face_of_pair, nf, normals):
    """The edges whose two faces make a fold smaller than SOFT_ANGLE.

    The smoothing pass of ``from_manifold`` read the faces of an edge from ``e2f``;
    the (edge, face) pairs of the loops say the same thing, so the edges can be
    tested in one go."""
    both = np.unique(edge_of_pair * nf + face_of_pair)      # one entry per edge, face
    e = both // nf
    f = both % nf
    if len(e) < 2:
        return set()
    how = np.bincount(e, minlength=int(e.max()) + 1)
    pair = (e[:-1] == e[1:]) & (how[e[:-1]] == 2)
    dots = np.einsum("ij,ij->i", normals[f[:-1][pair]], normals[f[1:][pair]])
    cos_soft = math.cos(math.radians(SOFT_ANGLE))
    return {int(x) for x in e[:-1][pair][dots > cos_soft].tolist()}


def plane_key(plane):
    n = plane.n.copy()
    for c in n:
        if abs(c) > 1e-9:
            if c < 0:
                n = -n
            break
    d = float(np.dot(n, plane.origin))
    return tuple(np.round(n, 6)) + (round(d, 4),)


def plane_id(plane):
    """The exact identity of a plane, for the caches that store plane coordinates.

    Unlike ``plane_key`` this does not round or flip anything: cached results are
    only reused for a plane with bit-identical parameters, so the tolerance based
    plane tests of ``faces_in_plane`` and friends keep their exact behaviour."""
    return (plane.n[0], plane.n[1], plane.n[2], float(plane.n @ plane.origin),
            plane.u[0], plane.u[1], plane.u[2], plane.v[0], plane.v[1], plane.v[2])


def canon_plane(plane):
    """The plane with its origin moved onto the normal axis (to ``d * n``).

    ``plane_id`` identifies a plane by its normal, its offset ``d = n . origin``
    and its in-plane axes, but the local frame of ``Plane`` also depends on where
    the origin sits *inside* the plane: ``face_plane`` returns a plane whose origin
    is a vertex, so the same geometric plane comes back with a different origin
    every time. Moving the origin to ``d * n`` (the point of the plane closest to
    the world origin) makes ``to_local`` a function of ``plane_id`` alone, so the
    caches that store plane coordinates (``coverage``, the arrangement and the
    region matches) stay valid when that plane comes back.
    """
    d = float(np.dot(plane.n, plane.origin))
    return Plane(plane.n * d, plane.n, plane.u)


def canon_shift(plane):
    """``(canon_plane(plane), shift)`` where ``shift`` moves plane coordinates.

    A point that is ``q`` in ``plane`` is ``q + shift`` in ``canon_plane(plane)``,
    so 2d data a caller already computed in ``plane`` (the loops handed to
    ``add_shape``) can be reused in the canonical frame."""
    d = float(np.dot(plane.n, plane.origin))
    w = plane.origin - plane.n * d
    return canon_plane(plane), (float(w @ plane.u), float(w @ plane.v))


# --------------------------------------------------------------- meshes across processes
#
# ``to_manifold`` triangulates every face in Python, which is the heavy part of an
# export and is independent per group. The three functions below split that work
# so it can run in a worker process (see ``workers.parallel_map``): ``mesh_payload``
# takes a group out to picklable numpy arrays (no ``Entities``, no Qt),
# ``triangulate_payload`` does the per-face triangulation on those arrays, and
# ``manifold_from_tris`` rebuilds the ``Manifold`` in the main process - a
# ``Manifold`` cannot be pickled, so only the arrays travel between processes.


def mesh_payload(ents, fids=None, face_ids=None):
    """The faces as picklable arrays ``(verts, loops, normals, tags)``.

    ``loops`` holds one entry per face: its loops with the vertices renumbered to
    rows of ``verts``. It is exactly what ``face_triangles`` reads per face, so a
    worker can triangulate a group without the ``Entities`` object."""
    fids = list(ents.F) if fids is None else list(fids)
    if not fids:
        return (np.zeros((0, 3)), [], np.zeros((0, 3)),
                (np.zeros(0, dtype=np.uint64) if face_ids is not None else None))
    vids = sorted({v for f in fids for loop in ents.F[f].loops for v in loop})
    idx = {v: i for i, v in enumerate(vids)}
    verts = np.array([ents.V[v] for v in vids], dtype=np.float64)
    loops = [tuple(tuple(idx[v] for v in loop) for loop in ents.F[f].loops) for f in fids]
    normals = np.array([ents.F[f].normal for f in fids], dtype=np.float64)
    tags = (np.array([face_ids.get(f, 0) for f in fids], dtype=np.uint64)
            if face_ids is not None else None)
    return verts, loops, normals, tags


def _loop_triangles(verts, ls, normal):
    """The triangulation of one face given as loops of rows of ``verts``: what
    ``Entities.face_triangles`` gives, including its shortcut for a triangular face."""
    if len(ls) == 1 and len(ls[0]) == 3 and _triangle(verts, ls[0], normal) is not None:
        return TRI_TRIS
    pl = Plane(verts[ls[0][0]], normal)
    polys = [pl.to_local(verts[list(loop)]) for loop in ls]
    try:
        return np.asarray(mf.triangulate(polys), dtype=np.int64).reshape(-1, 3)
    except Exception:
        return np.zeros((0, 3), dtype=np.int64)


def _triangle_rows(loops):
    """The faces that are one loop of three vertices, as ``(rows, faces)``: their three
    vertex rows and the row each has in `loops`. Such a face has only one triangulation."""
    rows, faces = [], []
    for i, ls in enumerate(loops):
        if len(ls) == 1 and len(ls[0]) == 3:
            rows.append(ls[0])
            faces.append(i)
    if not rows:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.asarray(rows, dtype=np.int64), np.asarray(faces, dtype=np.int64)


def _triangles_direct(verts, rows, normals):
    """Which of the triangle `rows` ``_triangle`` takes directly, decided for all of them
    at once: the same cross product test on the same doubles, only without a call and a
    small array per face."""
    pa, pb, pc = verts[rows[:, 0]], verts[rows[:, 1]], verts[rows[:, 2]]
    ux, uy, uz = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1], pb[:, 2] - pa[:, 2]
    vx, vy, vz = pc[:, 0] - pa[:, 0], pc[:, 1] - pa[:, 1], pc[:, 2] - pa[:, 2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    nn = cx * cx + cy * cy + cz * cz
    sliver = 1e-24 * ((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz))
    return (nn > 0.0) & (nn > sliver) & \
           (cx * normals[:, 0] + cy * normals[:, 1] + cz * normals[:, 2] > 0.0)


def triangulate_payload(payload):
    """``(tris, tags)`` for a :func:`mesh_payload`: the triangles of every face.

    The same walk as ``to_manifold`` - a plane per face, its loops to local
    coordinates, ``manifold3d.triangulate``, corners back to rows of ``verts`` -
    so the result is identical whether it runs here or in a worker.

    The faces that are a single triangle (nearly every face of an imported mesh) are
    collected in one go instead of one ``_triangle`` call each, which is what made a
    group of 200k faces take seconds here; the faces that need ear clipping still walk
    ``_loop_triangles`` one by one."""
    verts, loops, normals, tags = payload
    rows, faces = _triangle_rows(loops)
    if len(rows):
        keep = _triangles_direct(verts, rows, normals[faces])
        rows, faces = rows[keep], faces[keep]
    counts = np.zeros(len(loops), dtype=np.int64)
    counts[faces] = 1
    done = set(faces.tolist())
    rest = []
    for i, ls in enumerate(loops):
        if i in done:
            continue
        ids = [v for loop in ls for v in loop]
        if not ids:
            continue
        t = _loop_triangles(verts, ls, normals[i])
        if len(t):
            rest.append((i, np.asarray(ids, dtype=np.int64)[t]))
            counts[i] = len(t)
    at = np.zeros(len(loops) + 1, dtype=np.int64)
    np.cumsum(counts, out=at[1:])
    tris = np.zeros((int(at[-1]), 3), dtype=np.int64)
    if len(faces):
        tris[at[faces]] = rows
    for i, t in rest:
        tris[at[i]:at[i] + len(t)] = t
    if tags is None:
        return tris, None
    return tris, np.repeat(tags, counts)


def manifold_from_tris(verts, tris, tags):
    """The ``Manifold`` of triangles from :func:`triangulate_payload` (built in the main process)."""
    if len(tris) == 0:
        return mf.Manifold()
    extra = {"face_id": np.ascontiguousarray(tags, dtype=np.uint64)} if tags is not None else {}
    mesh = mf.Mesh64(vert_properties=np.ascontiguousarray(verts, dtype=np.float64),
                     tri_verts=np.ascontiguousarray(tris, dtype=np.uint64), **extra)
    return mf.Manifold(mesh)


def triangulate_job(payload):
    """Worker entry point: the triangles of one group, as picklable arrays."""
    return triangulate_payload(payload)


def render_payload(ents):
    """``(verts, loops, normals, fids)`` for every face, for the render / pick triangulation.

    Like :func:`mesh_payload` but without color tags: it carries the face id so the
    triangles come back tagged with the face they belong to (``GeoCache`` needs that
    for picking and selection)."""
    fids = list(ents.F)
    if not fids:
        return np.zeros((0, 3)), [], np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    vids = sorted({v for f in fids for loop in ents.F[f].loops for v in loop})
    idx = {v: i for i, v in enumerate(vids)}
    verts = np.array([ents.V[v] for v in vids], dtype=np.float64)
    loops = [tuple(tuple(idx[v] for v in loop) for loop in ents.F[f].loops) for f in fids]
    normals = np.array([ents.F[f].normal for f in fids], dtype=np.float64)
    return verts, loops, normals, np.asarray(fids, dtype=np.int64)


def triangulate_render(payload):
    """``(T, tri_fid, tri_nrm)``: the world triangles of every face, as ``GeoCache`` builds them.

    The same walk as the loop in ``GeoCache._build`` - a plane per face, its loops to
    local coordinates, ``manifold3d.triangulate``, empty faces skipped - so the
    render and pick cache is identical whether it is built here or in a worker."""
    verts, loops, normals, fids = payload
    T, fid, nrm = [], [], []
    for i, ls in enumerate(loops):
        ids = [v for loop in ls for v in loop]
        if not ids:
            continue
        t = _loop_triangles(verts, ls, normals[i])
        if not len(t):
            continue
        T.append(verts[ids][t])
        fid.extend([int(fids[i])] * len(t))
        nrm.append(np.repeat(normals[i][None], len(t), axis=0))
    return (np.concatenate(T) if T else np.zeros((0, 3, 3)),
            np.asarray(fid, dtype=np.int64),
            np.concatenate(nrm) if nrm else np.zeros((0, 3)))


def render_triangulate_job(payload):
    """Worker entry point: the render triangles of one group, as picklable arrays."""
    return triangulate_render(payload)


class Entities:
    """A set of connected SketchUp style geometry (the model or a group)."""

    def __init__(self):
        self.V = {}          # vid -> position (never mutated in place)
        self.E = {}          # eid -> (a, b)
        self.F = {}          # fid -> Face
        self.soft = set()    # smoothed edges (not drawn as a crease)
        self.hidden_f = set()  # faces hidden by Edit > Hide
        self.hidden_e = set()  # edges hidden by Edit > Hide
        self.next_id = 1
        self._index()

    # ------------------------------------------------------------ bookkeeping
    def _index(self):
        """Rebuild every lookup table from V, E and F.

        The cells of the three spatial grids are computed for all vertices and all
        edges at once with numpy (``_point_cells`` and ``_edge_cells``); Python only
        fills the dictionaries, which is what ``vertex`` and ``_einsert`` do one
        item at a time while the model is drawn. The rows of a cell keep the order
        of ``V`` and ``E``, so the grids hold what inserting the same geometry edge
        by edge would have put in them."""
        vids = list(self.V)
        self.grid = defaultdict(list)
        self.vgrid = defaultdict(list)
        self.egrid = defaultdict(list)
        self.ebig = set()
        self.ekey = {}
        self.v2e = defaultdict(set)
        self.e2f = defaultdict(set)
        self._pos_cache = None
        self._dead_v = set(vids)
        P = None
        if vids:
            P = np.array([self.V[v] for v in vids], dtype=float)
            for cell, rows in _cell_groups(_point_cells(P, CELL), vids):
                self.grid[cell] = rows
            for cell, rows in _cell_groups(_point_cells(P, ECELL), vids):
                self.vgrid[cell] = rows
        eids = list(self.E)
        if eids and P is not None:
            ab = np.array([self.E[e] for e in eids], dtype=np.int64)
            row = {v: i for i, v in enumerate(vids)}
            ia = np.fromiter((row[v] for v in ab[:, 0].tolist()), np.int64, len(ab))
            ib = np.fromiter((row[v] for v in ab[:, 1].tolist()), np.int64, len(ab))
            PA, PB = P[ia], P[ib]
            C, rows, big = _edge_cells(np.minimum(PA, PB), np.maximum(PA, PB), ECELL, EGRID_CAP)
            el = np.asarray(eids, dtype=np.int64)
            for cell, grp in _cell_groups(C, el[rows].tolist()):
                self.egrid[cell] = grp
            self.ebig = {eids[i] for i in np.flatnonzero(big)}
            for e, a, b in zip(eids, ab[:, 0].tolist(), ab[:, 1].tolist()):
                self.ekey[(a, b) if a < b else (b, a)] = e
                self.v2e[a].add(e)
                self.v2e[b].add(e)
            self._dead_v -= set(ab.ravel().tolist())
            self._index_faces(ab, el)
        self._clear_caches()
        self.version = getattr(self, "version", 0) + 1

    def _index_faces(self, ab, eids):
        """e2f: the faces that use every edge, matched on their vertex pair in bulk.

        A loop pair with no edge between its vertices is skipped, exactly as
        ``face_edges`` skips the ``None`` of ``ekey.get``."""
        seq, starts, lens, fids = [], [], [], []
        for f, face in self.F.items():
            for loop in face.loops:
                if not len(loop):
                    continue
                starts.append(len(seq))
                lens.append(len(loop))
                fids.append(f)
                seq.extend(loop)
        if not seq or not len(ab):
            return
        L = np.asarray(seq, dtype=np.int64)
        nxt = np.roll(L, -1)
        first = np.asarray(starts, dtype=np.int64)
        last = first + np.asarray(lens, dtype=np.int64) - 1
        nxt[last] = L[first]                      # close every loop
        pos = {f: i for i, f in enumerate(self.F)}
        FPOS = np.repeat(np.fromiter((pos[f] for f in fids), np.int64, len(fids)),
                         np.asarray(lens, dtype=np.int64))     # the face of a loop point
        nv = max(int(L.max()), int(ab.max())) + 1
        lo = np.minimum(L, nxt)
        hi = np.maximum(L, nxt)
        key = lo * nv + hi
        elo = np.minimum(ab[:, 0], ab[:, 1])
        ehi = np.maximum(ab[:, 0], ab[:, 1])
        o = np.argsort(elo * nv + ehi, kind="stable")
        ks = (elo * nv + ehi)[o]
        at = np.searchsorted(ks, key)
        at = np.minimum(at, len(ks) - 1)
        hit = ks[at] == key
        pair = o[at[hit]]
        nf = len(self.F)
        both = np.unique(eids[pair] * nf + FPOS[hit])       # one entry per (edge, face)
        e2 = both // nf
        cut = (np.flatnonzero(e2[1:] != e2[:-1]) + 1).tolist()
        fa = np.asarray(list(self.F), dtype=np.int64)
        e2, fs = e2.tolist(), fa[both % nf].tolist()
        for s, e in zip([0] + cut, cut + [len(e2)]):
            self.e2f[e2[s]] = set(fs[s:e])

    def _clear_caches(self):
        """Drop everything that is derived from the geometry.

        The caches below only hold results recomputed from ``V``, ``E`` and ``F``,
        so moving vertices drops them all (see ``_moved``):

        ``_qcache``     faces and edges of a plane (keyed on the exact plane)
        ``_covcache``   the coverage entry of one face in one plane
        ``_arrcache``   the planar arrangement of a plane, keyed on its edge set
        ``_matchcache`` which coverage entry covers which region of a plane
        """
        self._qcache = {}
        self._covcache = {}
        self._arrcache = {}
        self._matchcache = {}

    def _changed(self):
        """Call after a vertex, edge or face was added or removed."""
        self._qcache.clear()

    def _face_changed(self):
        """Call when the loops, normal or colour of a face changed.

        ``_arrcache`` and ``_matchcache`` survive: an arrangement is keyed on the
        edges of its plane and a coverage match on the identity of the coverage
        entries, so both notice a changed face by themselves."""
        self._qcache.clear()
        self._covcache.clear()

    def state(self):
        return (dict(self.V), dict(self.E), dict(self.F), set(self.soft), self.next_id,
                (set(self.hidden_f), set(self.hidden_e)))

    def load_state(self, st):
        """Load a state() snapshot into this object, so that references to it (a
        group, an open edit) keep pointing at the geometry."""
        self.V, self.E, self.F = dict(st[0]), dict(st[1]), dict(st[2])
        self.soft, self.next_id = set(st[3]), st[4]
        hid = st[5] if len(st) > 5 else ((), ())
        self.hidden_f, self.hidden_e = set(hid[0]), set(hid[1])
        self._index()
        return self

    @classmethod
    def from_state(cls, st):
        ents = cls.__new__(cls)
        ents.version = 0
        return ents.load_state(st)

    def copy(self):
        return Entities.from_state(self.state())

    def _new_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def touch(self):
        self.version += 1

    @staticmethod
    def _cell(p):
        return (math.floor(p[0] / CELL), math.floor(p[1] / CELL), math.floor(p[2] / CELL))

    def find_vertex(self, p):
        cx, cy, cz = self._cell(p)
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        best, bd = None, TOL
        get = self.grid.get
        V = self.V
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for v in get((cx + dx, cy + dy, cz + dz), ()):
                        q = V[v]
                        qx, qy, qz = float(q[0]) - px, float(q[1]) - py, float(q[2]) - pz
                        d = math.sqrt(qx * qx + qy * qy + qz * qz)
                        if d < bd:
                            best, bd = v, d
        return best

    # ------------------------------------------------------- coarse spatial index
    # add_segment looks for the edges and vertices it must split or cross. Only
    # geometry near the new segment can qualify, so a coarse grid (ECELL mm cells)
    # over edge bounding boxes and vertex positions keeps that a local lookup
    # instead of a scan of the whole model. A query box already includes TOL, and an
    # edge that can qualify has a bounding box overlapping it, so the two share a cell:
    # the query returns a superset and the exact tests still decide, so the result is
    # identical to scanning everything; edges too long for the grid go to a linear list.
    @staticmethod
    def _ecell(p):
        return (math.floor(p[0] / ECELL), math.floor(p[1] / ECELL),
                math.floor(p[2] / ECELL))

    def _ecells(self, lo, hi):
        c0 = [math.floor(lo[i] / ECELL) for i in range(3)]
        c1 = [math.floor(hi[i] / ECELL) for i in range(3)]
        for i in range(c0[0], c1[0] + 1):
            for j in range(c0[1], c1[1] + 1):
                for k in range(c0[2], c1[2] + 1):
                    yield (i, j, k)

    def _einsert(self, e):
        pa, pb = self.V[self.E[e][0]], self.V[self.E[e][1]]
        c0 = [math.floor(min(pa[i], pb[i]) / ECELL) for i in range(3)]
        c1 = [math.floor(max(pa[i], pb[i]) / ECELL) for i in range(3)]
        n = (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) * (c1[2] - c0[2] + 1)
        if n > EGRID_CAP:
            self.ebig.add(e)
        else:
            for i in range(c0[0], c1[0] + 1):
                for j in range(c0[1], c1[1] + 1):
                    for k in range(c0[2], c1[2] + 1):
                        self.egrid[(i, j, k)].append(e)

    def _eremove(self, e, a, b):
        pa, pb = self.V[a], self.V[b]
        c0 = [math.floor(min(pa[i], pb[i]) / ECELL) for i in range(3)]
        c1 = [math.floor(max(pa[i], pb[i]) / ECELL) for i in range(3)]
        n = (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) * (c1[2] - c0[2] + 1)
        if n > EGRID_CAP:
            self.ebig.discard(e)
        else:
            for i in range(c0[0], c1[0] + 1):
                for j in range(c0[1], c1[1] + 1):
                    for k in range(c0[2], c1[2] + 1):
                        s = self.egrid.get((i, j, k))
                        if s and e in s:
                            s.remove(e)

    def _edges_near(self, lo, hi):
        out = set(self.ebig)
        upd = out.update
        for c in self._ecells(lo, hi):
            s = self.egrid.get(c)
            if s:
                upd(s)
        return out

    def _vertices_near(self, lo, hi):
        out = set()
        upd = out.update
        for c in self._ecells(lo, hi):
            s = self.vgrid.get(c)
            if s:
                upd(s)
        return out

    def vertex(self, p):
        p = np.asarray(p, dtype=float)
        v = self.find_vertex(p)
        if v is not None:
            return v
        v = self._new_id()
        self.V[v] = p.copy()
        self.grid[self._cell(p)].append(v)
        self.vgrid[self._ecell(p)].append(v)
        self._dead_v.add(v)
        pos = self._pos_cache
        if pos is not None:
            # grow the cached position array instead of dropping it on every vertex
            if v >= len(pos):
                pos = self._pos_cache = np.vstack([pos, np.zeros((max(v + 1 - len(pos), 64), 3))])
            pos[v] = p
        self._changed()
        return v

    def _remove_vertex(self, v):
        if self.v2e.get(v):
            return
        p = self.V.pop(v, None)
        self._dead_v.discard(v)
        if p is not None:
            pos = self._pos_cache
            if pos is not None and v < len(pos):
                pos[v] = 0.0
            cell = self.grid.get(self._cell(p))
            if cell and v in cell:
                cell.remove(v)
            cell = self.vgrid.get(self._ecell(p))
            if cell and v in cell:
                cell.remove(v)
        self.v2e.pop(v, None)
        self._changed()

    def edge_between(self, a, b):
        return self.ekey.get(_ekey(a, b))

    def _add_edge_raw(self, a, b, soft=False, hidden=False):
        if a == b:
            return None
        e = self.edge_between(a, b)
        if e is not None:
            return e
        e = self._new_id()
        self.E[e] = (a, b)
        self.ekey[_ekey(a, b)] = e
        self.v2e[a].add(e)
        self.v2e[b].add(e)
        self._dead_v.discard(a)
        self._dead_v.discard(b)
        if soft:
            self.soft.add(e)
        if hidden:
            self.hidden_e.add(e)
        self._einsert(e)
        self._changed()
        return e

    def _remove_edge_raw(self, e):
        a, b = self.E[e]
        self._eremove(e, a, b)
        self.E.pop(e)
        self.ekey.pop(_ekey(a, b), None)
        self.v2e[a].discard(e)
        self.v2e[b].discard(e)
        self.soft.discard(e)
        self.hidden_e.discard(e)
        self.e2f.pop(e, None)
        self._dead_v.add(a)
        self._dead_v.add(b)
        self._changed()

    def face_edges(self, face):
        out = []
        for loop in face.loops:
            n = len(loop)
            for i in range(n):
                e = self.ekey.get(_ekey(loop[i], loop[(i + 1) % n]))
                if e is not None:
                    out.append(e)
        return out

    def _add_face_raw(self, loops, normal, color=None, fid=None):
        for loop in loops:
            for i in range(len(loop)):
                self._add_edge_raw(loop[i], loop[(i + 1) % len(loop)])
        face = Face(loops, normal, color)
        fid = fid or self._new_id()
        self.F[fid] = face
        for e in self.face_edges(face):
            self.e2f[e].add(fid)
        self._changed()
        return fid

    def _remove_face_raw(self, fid):
        face = self.F.pop(fid)
        self.hidden_f.discard(fid)
        for e in self.face_edges(face):
            self.e2f[e].discard(fid)
        # only faces_in_plane changes: the coverage of the other faces is untouched
        # (a removed face id is never reused, so its cached entry is simply dead) and
        # rebuild_plane notices the missing coverage entry through its identity check
        self._changed()

    def set_face_color(self, fid, color):
        self.F[fid] = self.F[fid].replaced(color=color)
        self._face_changed()
        self.touch()

    def _reverse_face(self, fid):
        face = self.F[fid]
        self.F[fid] = Face(tuple(reversed(l) for l in face.loops), -face.normal, face.color)
        self._face_changed()

    def reverse_faces(self, fids):
        """Reverse the orientation of the given faces (SketchUp: Right Click > Reverse)."""
        for fid in fids:
            if fid in self.F:
                self._reverse_face(fid)
        self.touch()

    def orient_faces(self, fid):
        """Make all faces connected to fid consistently oriented (SketchUp: Orient Faces).

        Breadth first over shared edges: neighbours run the shared edge in
        opposite directions; a neighbour that does not is reversed."""
        if fid not in self.F:
            return
        seen = {fid}
        queue = [fid]
        while queue:
            f = queue.pop(0)
            for loop in self.F[f].loops:
                n = len(loop)
                for i in range(n):
                    a, b = loop[i], loop[(i + 1) % n]
                    e = self.ekey.get(_ekey(a, b))
                    if e is None:
                        continue
                    for g in self.e2f.get(e, ()):
                        if g in seen:
                            continue
                        seen.add(g)
                        queue.append(g)
                        if any(l2[k] == a and l2[(k + 1) % len(l2)] == b
                               for l2 in self.F[g].loops for k in range(len(l2))):
                            self._reverse_face(g)
        self.touch()

    # ------------------------------------------------------------ visibility
    def hide(self, fids=(), eids=()):
        """Edit > Hide: hide faces and edges (they stay part of the model)."""
        self.hidden_f |= {f for f in fids if f in self.F}
        self.hidden_e |= {e for e in eids if e in self.E}
        self.touch()

    def unhide(self, fids=(), eids=()):
        """Edit > Unhide: show the given faces and edges again."""
        self.hidden_f -= set(fids)
        self.hidden_e -= set(eids)
        self.touch()

    def unhide_all(self):
        """Edit > Unhide > All: show every hidden face and edge."""
        had = bool(self.hidden_f or self.hidden_e)
        self.hidden_f.clear()
        self.hidden_e.clear()
        if had:
            self.touch()
        return had

    def visible_faces(self):
        return [f for f in self.F if f not in self.hidden_f]

    def visible_edges(self):
        return [e for e in self.E if e not in self.hidden_e]

    # ------------------------------------------------------------ queries
    def face_plane(self, fid):
        f = self.F[fid]
        return Plane(self.V[f.loops[0][0]], f.normal)

    def face_points(self, fid):
        return [np.array([self.V[v] for v in loop]) for loop in self.F[fid].loops]

    def edge_points(self, e):
        a, b = self.E[e]
        return self.V[a], self.V[b]

    def faces_in_plane(self, plane):
        key = ("f", plane_id(plane))
        hit = self._qcache.get(key)
        if hit is not None:
            return list(hit)
        if not self.F:
            return []
        fids = list(self.F)
        pos = self._pos_by_id()
        N = np.array([self.F[f].normal for f in fids])
        P0 = pos[[self.F[f].loops[0][0] for f in fids]]
        dn = np.abs(np.abs(N @ plane.n) - 1.0)
        dist = np.abs((P0 - plane.origin) @ plane.n)
        ok = (dn < 1e-7) & (dist < TOL * 10)
        out = [fids[i] for i in np.nonzero(ok)[0]]
        self._qcache[key] = out
        return list(out)

    def _pos_by_id(self):
        """A (next_id, 3) array with pos[vid] the position of vertex vid.

        Building this once with numpy is much cheaper than the ``np.array([V[v]
        for v in ...])`` list comprehensions that used to rebuild positions from
        the dictionary on every edge/segment query. The array only depends on the
        vertex positions, so it is cached and dropped whenever a vertex is added,
        removed or moved (see ``vertex``, ``_remove_vertex`` and ``_index``)."""
        c = self._pos_cache
        if c is not None:
            return c
        pos = np.zeros((max(self.next_id, 1), 3))
        if self.V:
            pos[list(self.V)] = np.array(list(self.V.values()))
        self._pos_cache = pos
        return pos

    def edges_in_plane(self, plane):
        key = ("e", plane_id(plane))
        hit = self._qcache.get(key)
        if hit is not None:
            return list(hit)
        if not self.E:
            return []
        ids = list(self.E)
        ab = np.array(list(self.E.values()))
        pos = self._pos_by_id()
        n = plane.n
        da = np.abs((pos[ab[:, 0]] - plane.origin) @ n)
        db = np.abs((pos[ab[:, 1]] - plane.origin) @ n)
        ok = (da < TOL * 10) & (db < TOL * 10)
        out = [ids[i] for i in np.nonzero(ok)[0]]
        self._qcache[key] = out
        return list(out)

    def loose_edges(self):
        return [e for e in self.E if not self.e2f.get(e)]

    def bbox(self):
        if not self.V:
            return None
        P = np.array(list(self.V.values()))
        return np.array([P.min(axis=0), P.max(axis=0)])

    def connected(self, faces=(), edges=()):
        """All faces and edges connected to the given ones (SketchUp triple click)."""
        seen_e, seen_f = set(), set()
        stack_e = list(edges)
        for f in faces:
            stack_e.extend(self.face_edges(self.F[f]))
            seen_f.add(f)
        while stack_e:
            e = stack_e.pop()
            if e in seen_e or e not in self.E:
                continue
            seen_e.add(e)
            for f in self.e2f.get(e, ()):
                if f not in seen_f:
                    seen_f.add(f)
                    stack_e.extend(self.face_edges(self.F[f]))
            for v in self.E[e]:
                stack_e.extend(self.v2e[v] - seen_e)
        return seen_f, seen_e

    # ------------------------------------------------------------ splitting
    def _split_edge(self, e, v):
        """Split edge e at vertex v, keeping faces intact."""
        a, b = self.E[e]
        if v in (a, b):
            return
        faces = [(f, self.F[f]) for f in self.e2f.get(e, ())]
        soft = e in self.soft
        hidden = e in self.hidden_e
        hid_f = {f for f, _ in faces if f in self.hidden_f}
        for f, _ in faces:
            self._remove_face_raw(f)
        self._remove_edge_raw(e)
        self._add_edge_raw(a, v, soft, hidden)
        self._add_edge_raw(v, b, soft, hidden)
        for f, face in faces:
            loops = []
            for loop in face.loops:
                new = []
                n = len(loop)
                for i in range(n):
                    p, q = loop[i], loop[(i + 1) % n]
                    new.append(p)
                    if (p, q) in ((a, b), (b, a)):
                        new.append(v)
                loops.append(new)
            self._add_face_raw(loops, face.normal, face.color, fid=f)
        self.hidden_f |= hid_f

    def _split_edges_at(self, v):
        """Split the edge that passes through vertex v (in valid geometry at most one)."""
        if not self.E:
            return
        p = self.V[v]
        # all candidates at once: where many short edges meet (a flattened spline, a
        # thread) a cell holds thousands of them, and a loop per edge made every new
        # segment cost milliseconds
        ids = [e for e in sorted(self._edges_near(p - TOL, p + TOL))
               if e in self.E and v not in self.E[e]]
        if not ids:
            return
        pos = self._pos_by_id()
        ab = np.array([self.E[e] for e in ids])
        A, B = pos[ab[:, 0]], pos[ab[:, 1]]
        d = B - A
        l2 = np.einsum("ij,ij->i", d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.einsum("ij,ij->i", p - A, d) / l2
        dist = np.linalg.norm(A + d * t[:, None] - p, axis=1)
        hit = np.flatnonzero((l2 >= 1e-18) & (t > 0.0) & (t < 1.0) & (dist < TOL))
        if len(hit):
            self._split_edge(ids[hit[0]], v)       # the first in id order, as before

    def add_segment(self, p, q):
        """Add an edge from p to q, splitting and merging with existing geometry.

        Returns (eids along the segment, newly created eids)."""
        a = self.vertex(p)
        b = self.vertex(q)
        if a == b:
            return [], []
        self._split_edges_at(a)
        self._split_edges_at(b)
        P, Q = self.V[a], self.V[b]
        d = Q - P
        L2 = float(d @ d)
        params = {0.0: a, 1.0: b}
        eps = TOL / math.sqrt(L2)
        lo = np.minimum(P, Q) - TOL
        hi = np.maximum(P, Q) + TOL
        pos = self._pos_by_id()
        # existing vertices on the segment: only the ones near it can lie on it
        vids = list(self._vertices_near(lo, hi))
        if vids:
            X = pos[vids]
            t = ((X - P) @ d) / L2
            foot = P + np.outer(t, d)
            dist = np.linalg.norm(X - foot, axis=1)
            for i in np.nonzero((t > eps) & (t < 1 - eps) & (dist < TOL))[0]:
                params[float(t[i])] = vids[i]
        # crossings with existing edges: only edges whose box touches the segment can cross
        if self.E:
            ids = sorted(self._edges_near(lo, hi))
            if ids:
                ab = np.array([self.E[e] for e in ids])
                C = pos[ab[:, 0]]
                D = pos[ab[:, 1]]
                r = D - C
                w = P - C
                aa = L2
                bb = r @ d
                cc = np.einsum("ij,ij->i", r, r)
                dd = w @ d
                ee = np.einsum("ij,ij->i", r, w)
                den = aa * cc - bb * bb
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (bb * ee - cc * dd) / den
                    uu = (aa * ee - bb * dd) / den
                ok = np.abs(den) > 1e-12 * aa * np.maximum(cc, 1e-12)
                ok &= (s > eps) & (s < 1 - eps) & (uu > 0) & (uu < 1)
                for i in np.nonzero(ok)[0]:
                    X1 = P + d * s[i]
                    X2 = C[i] + r[i] * uu[i]
                    if np.linalg.norm(X1 - X2) >= TOL:
                        continue
                    e = ids[i]
                    if e not in self.E:
                        continue
                    x = self.vertex((X1 + X2) / 2)
                    ca, cb = self.E[e]
                    if x not in (ca, cb):
                        self._split_edge(e, x)
                    params[float(s[i])] = x
        chain = [params[k] for k in sorted(params)]
        along, new = [], []
        for u, v in zip(chain[:-1], chain[1:]):
            if u == v:
                continue
            existed = self.edge_between(u, v)
            e = self._add_edge_raw(u, v)
            along.append(e)
            if existed is None:
                new.append(e)
        self.touch()
        return along, new

    # ------------------------------------------------------------ arrangement
    def _arrangement(self, plane, eids):
        """_regions, cached on the set of edges that lie in the plane.

        The arrangement of a plane only depends on those edges (and on the vertex
        positions, which only change in ``transform``), so pushing a face without
        changing the edges of its plane reuses the whole planar graph."""
        plane = canon_plane(plane)
        pid = plane_id(plane)
        key = frozenset(eids)
        hit = self._arrcache.get(pid)
        if hit is not None and len(hit[0]) == len(key) and hit[0] == key:
            return hit[1], hit[2]
        regions, pos = self._regions(plane, eids)
        self._arrcache[pid] = (key, regions, pos)
        return regions, pos

    def _regions(self, plane, eids):
        """Bounded regions of the planar graph made by edges `eids` in `plane`.

        Returns a list of [outer_loop, hole_loop, ...] (vertex ids). Outer loops
        run counter clockwise around plane.n, holes clockwise."""
        adj = defaultdict(set)
        for e in eids:
            a, b = self.E[e]
            adj[a].add(b)
            adj[b].add(a)
        verts = list(adj)
        if verts:
            W = self._pos_by_id()[verts] - plane.origin
            loc = np.stack([W @ plane.u, W @ plane.v], axis=-1)
            pos = {v: loc[i] for i, v in enumerate(verts)}
            xy = {v: t for v, t in zip(verts, map(tuple, loc))}
        else:
            pos = {}
            xy = {}

        def prune():
            stack = [v for v in adj if len(adj[v]) <= 1]
            while stack:
                v = stack.pop()
                if v not in adj or len(adj[v]) > 1:
                    continue
                for w in list(adj[v]):
                    adj[w].discard(v)
                    if len(adj[w]) <= 1:
                        stack.append(w)
                del adj[v]

        prune()
        atan2 = math.atan2
        while True:
            order = {}
            rank = {}
            for v, nb in adj.items():
                xv, yv = xy[v]
                srt = order[v] = sorted(nb, key=lambda w: atan2(xy[w][1] - yv, xy[w][0] - xv))
                rank[v] = {w: i for i, w in enumerate(srt)}
            visited = set()
            cycles = []
            for v, nb in order.items():
                for w in nb:
                    if (v, w) in visited:
                        continue
                    cyc = []
                    a, b = v, w
                    guard = 0
                    while (a, b) not in visited and guard < 100000:
                        guard += 1
                        visited.add((a, b))
                        cyc.append(a)
                        nbb = order[b]
                        c = nbb[(rank[b][a] - 1) % len(nbb)]
                        a, b = b, c
                    cycles.append(cyc)
            bridges = set()
            for cyc in cycles:
                de = {(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))}
                for x, y in de:
                    if (y, x) in de:
                        bridges.add(_ekey(x, y))
            if not bridges:
                break
            for x, y in bridges:
                adj[x].discard(y)
                adj[y].discard(x)
            prune()

        # connected components
        comp = {}
        for v in adj:
            if v in comp:
                continue
            stack = [v]
            comp[v] = v
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in comp:
                        comp[y] = v
                        stack.append(y)

        def area(cyc):
            n = len(cyc)
            s1 = 0.0
            s2 = 0.0
            for i in range(n):
                ax, ay = xy[cyc[i]]
                bx, by = xy[cyc[(i + 1) % n]]
                s1 += ax * by
                s2 += bx * ay
            return 0.5 * (s1 - s2)

        pos_c, neg_c = [], []
        for cyc in cycles:
            if len(cyc) < 3:
                continue
            A = area(cyc)
            if A > 1e-9:
                pos_c.append((A, cyc))
            elif A < -1e-9:
                neg_c.append((A, cyc))
        regions = [[cyc] for _, cyc in pos_c]
        # Assign every negative cycle (a hole, or the outer face of a component) to
        # the smallest positive cycle of another component that contains it. A grid
        # over the bounding boxes of the positive cycles keeps this near linear: a
        # point can only lie in a polygon whose box covers the cell it falls in, so
        # only those few candidates need the (small) point-in-polygon test.
        if neg_c and pos_c:
            polys = [np.array([pos[v] for v in cyc]) for _, cyc in pos_c]
            pb = [(p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()) for p in polys]
            xmin = min(b[0] for b in pb)
            xmax = max(b[1] for b in pb)
            ymin = min(b[2] for b in pb)
            ymax = max(b[3] for b in pb)
            grid, g, cw, ch = None, 0, 0.0, 0.0
            if xmax - xmin > 1e-9 and ymax - ymin > 1e-9:
                g = int(math.isqrt(len(pos_c))) + 1
                cw = (xmax - xmin) / g
                ch = (ymax - ymin) / g
                grid = defaultdict(list)
                for k, (bx0, bx1, by0, by1) in enumerate(pb):
                    i0 = min(g - 1, max(0, int((bx0 - xmin) / cw)))
                    i1 = min(g - 1, max(0, int((bx1 - xmin) / cw)))
                    j0 = min(g - 1, max(0, int((by0 - ymin) / ch)))
                    j1 = min(g - 1, max(0, int((by1 - ymin) / ch)))
                    for ii in range(i0, i1 + 1):
                        for jj in range(j0, j1 + 1):
                            grid[(ii, jj)].append(k)
            for _, cyc in neg_c:
                c0 = comp[cyc[0]]
                probe = pos[cyc[0]]
                na = -area(cyc)
                if grid is None:
                    cand = range(len(pos_c))
                else:
                    ii = min(g - 1, max(0, int((float(probe[0]) - xmin) / cw)))
                    jj = min(g - 1, max(0, int((float(probe[1]) - ymin) / ch)))
                    cand = grid.get((ii, jj))
                    if not cand:
                        continue
                best, best_a = None, None
                for k in sorted(set(cand)):
                    A, pc = pos_c[k]
                    if comp[pc[0]] == c0:
                        continue
                    if A < na - 1e-12:
                        continue
                    if point_in_polys([polys[k]], probe) and (best is None or A < best_a):
                        best, best_a = k, A
                if best is not None:
                    regions[best].append(cyc)
        return regions, pos

    @staticmethod
    def _sample_point(polys):
        """A point strictly inside a polygon with holes."""
        if len(polys) == 1:
            P = np.asarray(polys[0], dtype=float)
            n = len(P)
            if n >= 3:
                L = P.tolist()
                pos = neg = 0
                for i in range(n):
                    ax, ay = L[i]
                    bx, by = L[(i + 1) % n]
                    cx, cy = L[(i + 2) % n]
                    cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
                    if cr > 1e-9:
                        pos += 1
                    elif cr < -1e-9:
                        neg += 1
                if pos == 0 or neg == 0:
                    return P.mean(axis=0)   # convex: the vertex mean is strictly inside
        try:
            tris = np.asarray(mf.triangulate([np.asarray(p) for p in polys]), dtype=np.int64).reshape(-1, 3)
        except Exception:
            tris = np.zeros((0, 3), dtype=np.int64)
        P = np.concatenate(polys)
        if not len(tris):
            return P.mean(axis=0)
        A = P[tris]
        u, w = A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]
        areas = np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0])
        k = int(np.argmax(areas))
        return A[k].mean(axis=0)

    def coverage(self, plane, fids, flip=()):
        """Describe faces as plane coverage entries (polygons in plane coords).

        One entry is cached per face and plane (``_covcache``): a face that did not
        change keeps its entry object, which is how ``rebuild_plane`` can tell that
        the regions matching it did not change either."""
        plane = canon_plane(plane)
        out = []
        if not fids:
            return out
        pos = self._pos_by_id()
        pid = plane_id(plane)
        memo = self._covcache
        for f in fids:
            e = memo.get((pid, f))
            if e is None:
                face = self.F[f]
                polys = [plane.to_local(pos[list(loop)]) for loop in face.loops]
                e = {"polys": polys, "sign": 1 if np.dot(face.normal, plane.n) > 0 else -1,
                     "color": face.color}
                memo[(pid, f)] = e
            if f in flip:
                e = {"polys": e["polys"], "sign": -e["sign"], "color": e["color"]}
            out.append(e)
        return out

    @staticmethod
    def cov_entry(polys, sign, color=None):
        return {"polys": [np.asarray(p, dtype=float) for p in polys], "sign": sign, "color": color}

    @staticmethod
    def cancel(cov):
        """Remove overlap between opposite facing coverage (SketchUp push/pull)."""
        if len({c["sign"] for c in cov}) < 2:
            # nothing can cancel out when everything faces the same way; the
            # entries are handed back as they are (and keep their identity, which
            # is what rebuild_plane matches regions on)
            return list(cov)
        cs = [mf.CrossSection(c["polys"], mf.FillRule.EvenOdd) for c in cov]
        for i in range(len(cov)):
            for j in range(i + 1, len(cov)):
                if cov[i]["sign"] == cov[j]["sign"]:
                    continue
                ov = cs[i] ^ cs[j]
                if ov.area() > 1e-6:
                    cs[i] = cs[i] - ov
                    cs[j] = cs[j] - ov
        out = []
        for c, s in zip(cov, cs):
            polys = [np.asarray(p) for p in s.to_polygons()]
            if polys and s.area() > 1e-6:
                out.append({"polys": polys, "sign": c["sign"], "color": c["color"]})
        return out

    def rebuild_plane(self, plane, cov, new_edges=frozenset(), hint=None):
        """Recompute the faces of a plane from its edges and a coverage list.

        A region whose face is still there - same loops, same orientation, same
        colour - is kept as it was, so a plane with hundreds of faces is rebuilt
        by looking at the few regions that really changed. Faces therefore also
        keep their id (and their hidden state) when nothing changed for them.

        Which coverage entry covers a region is cached per plane on the identity
        of the coverage entries (``coverage`` hands out the same entry object for
        a face that did not change), so only the regions that a new entry could
        reach are tested again."""
        plane = canon_plane(plane)
        plane_fids = self.faces_in_plane(plane)
        eids = self.edges_in_plane(plane)
        regions, pos = self._arrangement(plane, eids)
        old = {}
        for f in plane_fids:
            loops = self.F[f].loops
            old[_loops_key(loops)] = f
            old[_loops_key(tuple(reversed(l) for l in loops))] = -f

        pid = plane_id(plane)
        ids = [id(c) for c in cov]
        prev = self._matchcache.get(pid)
        pm = prev[1] if prev is not None else {}
        if prev is None or prev[0] == ids:
            changed = ()
        else:
            seen = set(prev[0])
            changed = tuple(i for i, k in enumerate(ids) if k not in seen)
        idpos = {k: i for i, k in enumerate(ids)}
        self._matchcache[pid] = (ids, {}, cov)
        matches = self._matchcache[pid][1]
        box = {}

        def covers(i, sp):
            """Does coverage entry i contain the point sp (plane coordinates)?"""
            c = cov[i]
            bb = box.get(i, 0)
            if bb == 0:
                bb = box[i] = self._cov_bb(c)
            if bb is not None and (sp[0] < bb[0] or sp[0] > bb[2] or sp[1] < bb[1] or sp[1] > bb[3]):
                return False
            return point_in_polys(c["polys"], sp)

        def search(sp):
            """The first coverage entry that contains sp."""
            if not cov:
                return None
            # a grid over the bounding boxes of the entries keeps this near linear:
            # a sample point only needs the few entries whose box covers the cell it
            # falls in (see _regions, which uses the same trick).
            g = box.get("grid", 0)
            if g == 0:
                g = box["grid"] = self._cov_grid(cov, box)
            if g is None:
                cand = range(len(cov))
            else:
                cw, ch, cells, grid = g
                ii = min(cells - 1, max(0, int((sp[0] - box["xmin"]) / cw)))
                jj = min(cells - 1, max(0, int((sp[1] - box["ymin"]) / ch)))
                cand = sorted(set(grid.get((ii, jj), ())))
            for i in cand:
                if covers(i, sp):
                    return cov[i]
            return None

        created = []
        kept = set()
        for loops in regions:
            key = _loops_key(loops)
            hit = pm.get(key)
            entry = None
            if hit is None:
                sp = self._sample_point([np.array([pos[v] for v in l]) for l in loops])
                entry = search(sp)
            else:
                eid, sp = hit[0], hit[3]
                j = idpos.get(eid) if eid is not None else None
                if j is None:
                    if eid is not None or changed:
                        entry = search(sp)
                elif not changed:
                    entry = cov[j]
                else:
                    entry = cov[j]
                    for i in changed:
                        if i < j and covers(i, sp):
                            entry = cov[i]
                            break
            if entry is not None:
                sign, color, eid = entry["sign"], entry["color"], id(entry)
            elif new_edges and self._uses(loops, new_edges):
                sign, color, eid = self._orient_new(plane, loops, hint), None, None
            else:
                matches[key] = (None, None, None, sp)
                continue
            matches[key] = (eid, sign, color, sp)
            want = key if sign > 0 else tuple(tuple(reversed(l)) for l in key)
            g = old.get(_loops_key(want))
            if g is not None and g > 0:
                face = self.F[g]
                if face.color == color and float(np.dot(face.normal, plane.n * sign)) > 1 - 1e-12:
                    kept.add(g)
                    continue
            created.append(self._add_face_raw([list(l) for l in want], plane.n * sign, color))
        for f in plane_fids:
            if f in self.F and f not in kept:
                self._remove_face_raw(f)
        before = set(self.F)
        self._fold_non_planar(created)
        created = [f for f in created if f in self.F] + [f for f in self.F if f not in before]
        self.touch()
        return created + [f for f in plane_fids if f in kept]

    @staticmethod
    def _cov_bb(c):
        """Bounding box of a coverage entry in plane coordinates, or None if empty."""
        bb = c.get("bb")
        if bb is not None:
            return bb
        bb = None
        polys = c["polys"]
        if len(polys):
            allp = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 2) for p in polys])
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            bb = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        c["bb"] = bb
        return bb

    @staticmethod
    def _cov_grid(cov, box):
        """A grid over the bounding boxes of the coverage entries, or None."""
        boxes = []
        for i, c in enumerate(cov):
            bb = box.get(i, 0)
            if bb == 0:
                bb = box[i] = Entities._cov_bb(c)
            boxes.append(bb)
        have = [b for b in boxes if b is not None]
        if not have or len(cov) <= 8:
            return None
        xmin = min(b[0] for b in have)
        xmax = max(b[2] for b in have)
        ymin = min(b[1] for b in have)
        ymax = max(b[3] for b in have)
        if xmax - xmin <= 1e-9 or ymax - ymin <= 1e-9:
            return None
        g = int(math.isqrt(len(cov))) + 1
        cw = (xmax - xmin) / g
        ch = (ymax - ymin) / g
        grid = defaultdict(list)
        for idx, bb in enumerate(boxes):
            if bb is None:
                continue
            i0 = min(g - 1, max(0, int((bb[0] - xmin) / cw)))
            i1 = min(g - 1, max(0, int((bb[2] - xmin) / cw)))
            j0 = min(g - 1, max(0, int((bb[1] - ymin) / ch)))
            j1 = min(g - 1, max(0, int((bb[3] - ymin) / ch)))
            for ii in range(i0, i1 + 1):
                for jj in range(j0, j1 + 1):
                    grid[(ii, jj)].append(idx)
        box["xmin"] = xmin
        box["ymin"] = ymin
        return cw, ch, g, grid

    def _uses(self, loops, eids):
        for loop in loops:
            for i in range(len(loop)):
                if self.edge_between(loop[i], loop[(i + 1) % len(loop)]) in eids:
                    return True
        return False

    def _orient_new(self, plane, loops, hint):
        """Orientation of a new face: consistent with neighbours, else like SketchUp."""
        outer = loops[0]
        for i in range(len(outer)):
            a, b = outer[i], outer[(i + 1) % len(outer)]
            e = self.edge_between(a, b)
            for f in self.e2f.get(e, ()):
                face = self.F[f]
                if abs(abs(float(np.dot(face.normal, plane.n))) - 1) < 1e-7:
                    continue
                for loop in face.loops:
                    n = len(loop)
                    for k in range(n):
                        if loop[k] == a and loop[(k + 1) % n] == b:
                            return -1
                        if loop[k] == b and loop[(k + 1) % n] == a:
                            return 1
        if abs(plane.n[2]) > 0.999 and abs(plane.origin[2]) < TOL * 10:
            return 1 if plane.n[2] < 0 else -1  # faces on the ground look down
        if hint is not None:
            return 1 if np.dot(plane.n, np.asarray(hint) - plane.origin) >= 0 else -1
        return 1

    def candidate_planes(self, eids):
        """Planes in which new edges may close a loop."""
        planes = {}
        for e in eids:
            if e not in self.E:
                continue
            a, b = self.E[e]
            A, B = self.V[a], self.V[b]
            d = normalize(B - A)
            for f in self.F:
                pl = self.face_plane(f)
                if abs(pl.distance(A)) < TOL * 10 and abs(pl.distance(B)) < TOL * 10:
                    planes.setdefault(plane_key(pl), pl)
            for v in (a, b):
                for e2 in self.v2e[v]:
                    if e2 == e:
                        continue
                    x, y = self.E[e2]
                    d2 = normalize(self.V[y] - self.V[x])
                    n = np.cross(d, d2)
                    if np.linalg.norm(n) < 1e-6:
                        continue
                    pl = Plane(A, normalize(n))
                    planes.setdefault(plane_key(pl), pl)
        return list(planes.values())

    # ------------------------------------------------------------ high level
    def add_line(self, p, q, hint=None):
        """Line tool: add an edge and create faces for loops it closes."""
        along, new = self.add_segment(p, q)
        if along:
            self.make_faces(along, hint)
        return along

    def make_faces(self, eids, hint=None):
        eids = set(eids)
        for pl in self.candidate_planes(eids):
            if not any(self._edge_in(pl, e) for e in eids):
                continue
            cov = self.coverage(pl, self.faces_in_plane(pl))
            self.rebuild_plane(pl, cov, eids, hint)
        self._cleanup_vertices()

    def _edge_in(self, pl, e):
        a, b = self.E[e]
        return abs(pl.distance(self.V[a])) < TOL * 10 and abs(pl.distance(self.V[b])) < TOL * 10

    def add_shape(self, plane, loops2d, color=None, sign=None, hint=None):
        """Rectangle / circle / text: add closed loops and make the enclosed face."""
        loops2d = [np.asarray(l, dtype=float) for l in loops2d if len(l) >= 3]
        if not loops2d:
            return []
        plane, shift = canon_shift(plane)
        if shift[0] or shift[1]:
            loops2d = [l + shift for l in loops2d]
        existing = self.faces_in_plane(plane)
        cov = self.coverage(plane, existing)
        for loop in loops2d:
            W = plane.to_world(loop)
            for i in range(len(W)):
                self.add_segment(W[i], W[(i + 1) % len(W)])
        if sign is None:
            sign = self._orient_shape(plane, loops2d, cov, hint)
        cov.append(self.cov_entry(loops2d, sign, color))
        created = self.rebuild_plane(plane, cov, hint=hint)
        self._cleanup_vertices()
        return created

    def add_shapes(self, plane, shapes, color=None, hint=None, progress=None):
        """Several shapes on one plane, as `add_shape` one after the other would make them,
        but with one rebuild of the plane at the end instead of one per shape: an imported
        drawing with hundreds of shapes otherwise rebuilds a growing plane hundreds of times.
        `progress(fraction)` is called after every shape. Returns the new face ids."""
        plane0 = plane
        shapes = [[np.asarray(l, dtype=float) for l in s if len(l) >= 3] for s in shapes]
        shapes = [s for s in shapes if s]
        if not shapes:
            return []
        plane, shift = canon_shift(plane0)
        if shift[0] or shift[1]:
            shapes = [[l + shift for l in s] for s in shapes]
        cov = self.coverage(plane, self.faces_in_plane(plane))
        for k, loops2d in enumerate(shapes):
            for loop in loops2d:
                W = plane.to_world(loop)
                for i in range(len(W)):
                    self.add_segment(W[i], W[(i + 1) % len(W)])
            cov.append(self.cov_entry(loops2d, self._orient_shape(plane, loops2d, cov, hint),
                                      color))
            if progress:
                progress(0.8 * (k + 1) / len(shapes))
        created = self.rebuild_plane(plane, cov, hint=hint)
        self._cleanup_vertices()
        if progress:
            progress(1.0)
        return created

    def _orient_shape(self, plane, loops2d, cov, hint):
        sp = self._sample_point([loops2d[0]])
        for c in cov:
            if point_in_polys(c["polys"], sp):
                return c["sign"]
        for c in cov:
            return c["sign"]
        if abs(plane.n[2]) > 0.999 and abs(plane.origin[2]) < TOL * 10:
            return 1 if plane.n[2] < 0 else -1
        if hint is not None:
            return 1 if np.dot(plane.n, np.asarray(hint) - plane.origin) >= 0 else -1
        return 1

    def erase_edges(self, eids):
        eids = {e for e in eids if e in self.E}
        if not eids:
            return
        affected = set()
        for e in eids:
            affected |= self.e2f.get(e, set())
        keep = set()
        for f in affected:
            ok = True
            fplane = self.face_plane(f)
            for e in set(self.face_edges(self.F[f])) & eids:
                fs = self.e2f[e]
                if len(fs) != 2:
                    ok = False
                    break
                for g in fs:
                    if np.dot(self.F[g].normal, fplane.n) < 1 - 1e-7 or \
                            abs(fplane.distance(self.V[self.F[g].loops[0][0]])) > TOL * 10:
                        ok = False
                if not ok:
                    break
            if ok:
                keep.add(f)
        planes = {}
        for f in affected:
            pl = self.face_plane(f)
            planes.setdefault(plane_key(pl), pl)
        covs = {}
        for k, pl in planes.items():
            fids = [f for f in self.faces_in_plane(pl) if f not in affected or f in keep]
            covs[k] = self.coverage(pl, fids)
        ends = set()
        for f in affected:
            self._remove_face_raw(f)
        for e in eids:
            ends |= set(self.E[e])
            self._remove_edge_raw(e)
        for k, pl in planes.items():
            self.rebuild_plane(pl, covs[k])
        self.heal(ends)
        self._cleanup_vertices()
        self.touch()

    def erase_faces(self, fids):
        for f in fids:
            if f in self.F:
                self._remove_face_raw(f)
        self.touch()

    def heal(self, vids):
        """Join two collinear edges at a vertex that has nothing else attached."""
        for v in list(vids):
            if v not in self.V:
                continue
            es = list(self.v2e.get(v, ()))
            if len(es) != 2:
                continue
            e1, e2 = es
            a = [x for x in self.E[e1] if x != v][0]
            c = [x for x in self.E[e2] if x != v][0]
            if a == c or self.edge_between(a, c) is not None:
                continue
            d1 = normalize(self.V[a] - self.V[v])
            d2 = normalize(self.V[c] - self.V[v])
            if float(np.dot(d1, d2)) > -1 + 1e-9:
                continue
            if self.e2f.get(e1, set()) != self.e2f.get(e2, set()):
                continue
            soft = e1 in self.soft and e2 in self.soft
            hidden = e1 in self.hidden_e and e2 in self.hidden_e
            faces = [(f, self.F[f]) for f in self.e2f.get(e1, set())]
            hid_f = {f for f, _ in faces if f in self.hidden_f}
            for f, _ in faces:
                self._remove_face_raw(f)
            self._remove_edge_raw(e1)
            self._remove_edge_raw(e2)
            self._add_edge_raw(a, c, soft, hidden)
            for f, face in faces:
                loops = [[x for x in loop if x != v] for loop in face.loops]
                self._add_face_raw(loops, face.normal, face.color, fid=f)
            self.hidden_f |= hid_f
            self._remove_vertex(v)

    def _cleanup_vertices(self):
        """Forget the vertices no edge uses any more.

        ``_dead_v`` holds the vertices that can have lost their last edge (new
        vertices and the ends of removed edges), so this does not have to walk
        over every vertex of the model after every operation."""
        for v in list(self._dead_v):
            if v in self.V and not self.v2e.get(v):
                self._remove_vertex(v)

    # ------------------------------------------------------------ push / pull
    def pushpull(self, fid, d, keep=False):
        """Extrude a face by d along its normal, SketchUp style.

        keep=True keeps the original face (Ctrl: new starting face).
        Returns the id of the new cap face (or None when it was cancelled)."""
        if fid not in self.F or abs(d) < 1e-9:
            return None
        F = self.F[fid]
        n = F.normal
        off = n * d
        fedges = self.face_edges(F)
        attached = any(len(self.e2f[e]) > 1 for e in fedges)
        keep_orig = keep or not attached
        standalone = not attached
        flip = standalone and d < 0
        loose_before = set(self.loose_edges())
        P0 = canon_plane(self.face_plane(fid))
        Pc = canon_plane(Plane(P0.origin + off, P0.n, P0.u))

        loops_xyz = [[self.V[v] for v in loop] for loop in F.loops]
        # side quads: (base vertex a, b, plane, sign relative to plane normal)
        quads = []
        for loop, L in zip(F.loops, loops_xyz):
            m = len(loop)
            for i in range(m):
                A, B = L[i], L[(i + 1) % m]
                qn = normalize(np.cross(B - A, off))
                if flip:
                    qn = -qn
                quads.append((loop[i], loop[(i + 1) % m], A, B, canon_plane(Plane(A, qn))))

        planes = {plane_key(P0): P0, plane_key(Pc): Pc}
        for q in quads:
            planes.setdefault(plane_key(q[4]), q[4])

        removed = set() if keep_orig else {fid}
        flipped = {fid} if (standalone and not keep and d > 0) else set()
        covs = {}
        for k, pl in planes.items():
            fids = [f for f in self.faces_in_plane(pl) if f not in removed]
            covs[k] = self.coverage(pl, fids, flip=flipped)
        cap_sign = (-1 if flip else 1) * (1 if np.dot(n, Pc.n) > 0 else -1)
        covs[plane_key(Pc)].append(self.cov_entry(
            [Pc.to_local(np.array(L) + off) for L in loops_xyz], cap_sign, F.color))
        for a, b, A, B, qp in quads:
            k = plane_key(qp)
            pts = np.array([A, B, B + off, A + off])
            sign = 1 if np.dot(qp.n, planes[k].n) > 0 else -1
            covs[k].append(self.cov_entry([planes[k].to_local(pts)], sign, F.color))

        # edges to merge away: base edges between a side quad and a coplanar,
        # equally oriented neighbour, and side edges between coplanar quads
        merge_pairs = []
        for a, b, A, B, qp in quads:
            e = self.edge_between(a, b)
            for g in self.e2f.get(e, ()):
                if g == fid:
                    continue
                gf = self.F[g]
                if np.dot(gf.normal, qp.n) > 1 - 1e-7 and abs(qp.distance(self.V[gf.loops[0][0]])) < TOL * 10:
                    merge_pairs.append((A, B))
        side_merge = []
        idx = 0
        for loop in F.loops:
            m = len(loop)
            group = quads[idx:idx + m]
            for i in range(m):
                q1, q2 = group[i], group[(i + 1) % m]
                if np.dot(q1[4].n, q2[4].n) > 1 - 1e-7:
                    side_merge.append(q1[3])  # shared base vertex position
            idx += m

        for k in covs:
            covs[k] = self.cancel(covs[k])

        # geometry: the faces of these planes are not removed here, rebuild_plane
        # keeps the ones that turn out unchanged; `gone` says which ones can still
        # disappear, so an edge only counts as free when nothing else needs it
        gone = {fid} if fid in self.F else set()
        for k, pl in planes.items():
            gone |= set(self.faces_in_plane(pl))
        touched = set()
        for loop, L in zip(F.loops, loops_xyz):
            m = len(loop)
            for i in range(m):
                A, B = L[i], L[(i + 1) % m]
                al, _ = self.add_segment(A + off, B + off)
                touched.update(al)
                al, _ = self.add_segment(A, A + off)
                touched.update(al)
        for A, B in merge_pairs:
            for e in self._edges_on_segment(A, B):
                if not (self.e2f.get(e, set()) - gone):
                    self._remove_edge_raw(e)
        for A in side_merge:
            for e in self._edges_on_segment(A, A + off):
                if not (self.e2f.get(e, set()) - gone):
                    self._remove_edge_raw(e)
        made = set(touched)
        for k, pl in planes.items():
            for f in self.rebuild_plane(pl, covs[k]):
                if f in self.F:
                    made.update(self.face_edges(self.F[f]))
        # remove edges that bound nothing any more (but keep user's loose lines)
        verts = set()
        for k, pl in planes.items():
            for e in self.edges_in_plane(pl):
                if e in self.E and not self.e2f.get(e) and e not in loose_before:
                    verts |= set(self.E[e])
                    self._remove_edge_raw(e)
        for e in list(touched):
            if e in self.E:
                verts |= set(self.E[e])
        for loop in F.loops:
            verts |= set(loop)
        self.heal(verts)
        # soften folds between side faces of curved profiles: only the edges this
        # push/pull made can start to qualify, no need to scan every edge again
        for e in made:
            fs = self.e2f.get(e, ())
            if len(fs) == 2 and e not in self.soft:
                f1, f2 = (self.F[x] for x in fs)
                if np.dot(f1.normal, f2.normal) > math.cos(math.radians(SOFT_ANGLE)) and \
                        abs(np.dot(f1.normal, n)) < 1e-6 and abs(np.dot(f2.normal, n)) < 1e-6:
                    a, b = self.E[e]
                    if abs(abs(np.dot(normalize(self.V[b] - self.V[a]), n)) - 1) < 1e-6:
                        self.soft.add(e)
        self._cleanup_vertices()
        self.touch()
        cap = [f for f in self.faces_in_plane(Pc)]
        for f in cap:
            if np.dot(self.F[f].normal, n * (-1 if flip else 1)) > 0:
                return f
        return None

    def _edges_on_segment(self, A, B):
        d = B - A
        L2 = float(d @ d)
        out = []
        if L2 < 1e-18:
            return out
        for e, (a, b) in self.E.items():
            pa, pb = self.V[a], self.V[b]
            ok = True
            for p in (pa, pb):
                t = float((p - A) @ d) / L2
                if t < -1e-9 or t > 1 + 1e-9 or np.linalg.norm(A + d * t - p) > TOL:
                    ok = False
                    break
            if ok:
                out.append(e)
        return out

    # ------------------------------------------------------------ transforms
    def transform(self, M3, t, vids=None):
        """Apply an affine transform to vertices (all when vids is None)."""
        M3 = np.asarray(M3, dtype=float)
        t = np.asarray(t, dtype=float)
        vids = set(self.V) if vids is None else set(vids)
        mirror = np.linalg.det(M3) < 0
        for v in vids:
            self.V[v] = M3 @ self.V[v] + t
        nm = np.linalg.inv(M3).T
        for f, face in list(self.F.items()):
            if all(v in vids for loop in face.loops for v in loop):
                loops = face.loops
                if mirror:
                    loops = [tuple(reversed(l)) for l in loops]
                self.F[f] = Face(loops, normalize(nm @ face.normal), face.color)
        self._index()
        if len(vids) != len(self.V):
            self._fix_planarity()
        self.touch()

    def _fold_face(self, fid):
        """Split a face whose points are not in one plane into triangles (auto-fold)."""
        face = self.F[fid]
        P = np.array([self.V[v] for v in face.loops[0]])
        c = P.mean(axis=0)
        nrm = np.zeros(3)
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            nrm += np.cross(a - c, b - c)
        if np.linalg.norm(nrm) < 1e-12:
            return
        nrm = normalize(nrm)
        plane = Plane(c, nrm if np.dot(nrm, face.normal) >= 0 else -nrm)
        polys = [plane.to_local(np.array([self.V[v] for v in loop])) for loop in face.loops]
        ids = [v for loop in face.loops for v in loop]
        try:
            tris = np.asarray(mf.triangulate(polys), dtype=np.int64).reshape(-1, 3)
        except Exception:
            return
        if not len(tris):
            return
        self._remove_face_raw(fid)
        for tri in tris:
            vs = [ids[i] for i in tri]
            A, B, C = (self.V[v] for v in vs)
            tn = normalize(np.cross(B - A, C - A))
            for i in range(3):
                x, y = vs[i], vs[(i + 1) % 3]
                if self.edge_between(x, y) is None:
                    self._add_edge_raw(x, y, soft=True)
            self._add_face_raw([vs], tn, face.color)

    def _fold_non_planar(self, fids, tol=PLANAR_TOL):
        """Fold the given faces that are not flat, so every face stays planar.

        Vertices that are almost (but not quite) the same point make a region of
        a plane slightly warped; such a region cannot be one face."""
        for fid in list(fids):
            face = self.F.get(fid)
            if face is None:
                continue
            pts = np.array([self.V[v] for loop in face.loops for v in loop])
            if np.max(np.abs((pts - self.V[face.loops[0][0]]) @ face.normal)) > tol:
                self._fold_face(fid)

    def _fix_planarity(self):
        """Faces bent by moving part of them are re-computed or triangulated."""
        for f, face in list(self.F.items()):
            P = np.array([self.V[v] for v in face.loops[0]])
            c = P.mean(axis=0)
            nrm = np.zeros(3)
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                nrm += np.cross(a - c, b - c)
            if np.linalg.norm(nrm) < 1e-12:
                continue
            nrm = normalize(nrm)
            allp = np.array([self.V[v] for loop in face.loops for v in loop])
            if np.max(np.abs((allp - c) @ nrm)) < TOL * 10:
                if np.dot(nrm, face.normal) < 0:
                    nrm = -nrm
                self.F[f] = face.replaced(normal=nrm)
                self._face_changed()
                continue
            # non planar: split into triangles (soft edges), like SketchUp's auto-fold
            self._fold_face(f)

    def merge_from(self, other, M3=None, t=None, hint=None):
        """Copy all geometry of another Entities into this one (sticky)."""
        M3 = np.eye(3) if M3 is None else np.asarray(M3, dtype=float)
        t = np.zeros(3) if t is None else np.asarray(t, dtype=float)
        mirror = np.linalg.det(M3) < 0
        nm = np.linalg.inv(M3).T
        planes = {}
        for f in other.F:
            pl = other.face_plane(f)
            pl = canon_plane(Plane(M3 @ pl.origin + t, normalize(nm @ pl.n)))
            planes.setdefault(plane_key(pl), pl)
        covs = {k: self.coverage(pl, self.faces_in_plane(pl)) for k, pl in planes.items()}
        for f, face in other.F.items():
            pl = other.face_plane(f)
            pl = canon_plane(Plane(M3 @ pl.origin + t, normalize(nm @ pl.n)))
            k = plane_key(pl)
            target = planes[k]
            polys = [target.to_local(np.array([M3 @ other.V[v] + t for v in loop])) for loop in face.loops]
            sign = 1 if np.dot(normalize(nm @ face.normal), target.n) > 0 else -1
            covs[k].append(self.cov_entry(polys, sign, face.color))
        for e, (a, b) in other.E.items():
            al, _ = self.add_segment(M3 @ other.V[a] + t, M3 @ other.V[b] + t)
            if e in other.soft:
                self.soft.update(al)
            if e in other.hidden_e:
                self.hidden_e.update(al)
        for k, pl in planes.items():
            self.rebuild_plane(pl, covs[k], hint=hint)
        self._cleanup_vertices()
        self.touch()

    def extract(self, fids, eids):
        """New Entities with the given faces and edges (plus face edges)."""
        out = Entities()
        eids = set(eids)
        for f in fids:
            eids |= set(self.face_edges(self.F[f]))
        for e in eids:
            a, b = self.E[e]
            na, nb = out.vertex(self.V[a]), out.vertex(self.V[b])
            out._add_edge_raw(na, nb, e in self.soft, e in self.hidden_e)
        for f in fids:
            face = self.F[f]
            loops = [[out.vertex(self.V[v]) for v in loop] for loop in face.loops]
            nf = out._add_face_raw(loops, face.normal, face.color)
            if f in self.hidden_f:
                out.hidden_f.add(nf)
        out.touch()
        return out

    def remove(self, fids, eids):
        """Delete faces, then the given edges and edges that bound nothing any more."""
        cand = set(eids)
        for f in fids:
            if f in self.F:
                cand |= set(self.face_edges(self.F[f]))
                self._remove_face_raw(f)
        ends = set()
        for e in cand:
            if e in self.E and not self.e2f.get(e):
                ends |= set(self.E[e])
                self._remove_edge_raw(e)
        self.heal(ends)
        self._cleanup_vertices()
        self.touch()

    # ------------------------------------------------------------ meshes
    def face_triangles(self, fid):
        """Triangles (k,3,3) of a face, wound counter clockwise around its normal."""
        face = self.F[fid]
        loops = face.loops
        if len(loops) == 1 and len(loops[0]) == 3:
            tri = _triangle(self.V, loops[0], face.normal)
            if tri is not None:
                return tri, TRI_TRIS
        pl = Plane(self.V[loops[0][0]], face.normal)
        pts3 = [np.array([self.V[v] for v in loop]) for loop in loops]
        polys = [pl.to_local(p) for p in pts3]
        try:
            tris = np.asarray(mf.triangulate(polys), dtype=np.int64).reshape(-1, 3)
        except Exception:
            tris = np.zeros((0, 3), dtype=np.int64)
        allp = np.concatenate(pts3)
        return allp[tris], tris

    def to_manifold(self, fids=None, face_ids=None):
        """The faces as a Manifold. `face_ids` ({face: int}) tags every triangle of a face;
        Manifold keeps these tags through booleans (the 3MF export uses them for colors)."""
        verts, loops, normals, tags = mesh_payload(self, fids, face_ids)
        tris, tags = triangulate_payload((verts, loops, normals, tags))
        return manifold_from_tris(verts, tris, tags)

    def shells(self):
        """Groups of faces connected through edges."""
        left = set(self.F)
        out = []
        while left:
            f0 = left.pop()
            comp = {f0}
            stack = [f0]
            while stack:
                f = stack.pop()
                for e in self.face_edges(self.F[f]):
                    for g in self.e2f.get(e, ()):
                        if g in left:
                            left.discard(g)
                            comp.add(g)
                            stack.append(g)
            out.append(comp)
        return out

    def open_edges(self, fids=None):
        """Edges that do not have exactly two faces (holes / non-manifold)."""
        fids = set(self.F) if fids is None else set(fids)
        bad = []
        for e in self.E:
            fs = self.e2f.get(e, set()) & fids
            if fs and len(fs) != 2:
                bad.append(e)
        return bad

    @classmethod
    def from_manifold(cls, man, color=None, progress=None):
        """A Manifold as SketchUp geometry: one face per coplanar region.

        The vertices of a manifold are already merged, so they are created in bulk
        instead of one ``find_vertex`` per loop point; points that still lie within
        TOL of each other share a vertex under the same rule as ``vertex`` (see
        ``_merge_within_tol``). The ids come out as they did when the regions were
        walked one by one: a region first takes the ids of the vertices it is the
        first to touch, then those of the edges its loops add, then the id of its
        face, and a fold smaller than SOFT_ANGLE is smoothed at the end.

        The loops are built in "slot" numbers first - one per vertex, in the order
        the loops walk them - because which edges a region adds decides how many ids
        the regions before it took, and so the number of the vertices that follow.

        ``progress(fraction)`` is called between the steps, for a progress bar: on a big
        mesh this takes seconds."""
        progress = progress or (lambda fraction: None)
        ents = cls()
        if man.is_empty():
            return ents
        md = MeshData(man)
        index_loops = md.region_index_loops()
        progress(0.2)
        nr = len(index_loops)
        # the mesh vertices in the order the loops walk them: the order in which
        # vertex() would have created them. Every loop of every region is one run of
        # that array, so a loop is described by where it starts and how long it is.
        nloops = sum(len(loops) for loops in index_loops)
        lens = np.fromiter((len(loop) for loops in index_loops for loop in loops),
                           np.int64, nloops)
        L = np.fromiter((v for loops in index_loops for loop in loops for v in loop),
                        np.int64, int(lens.sum()))
        if not len(L):
            return ents
        lreg = np.repeat(np.arange(nr, dtype=np.int64),
                         np.fromiter((len(loops) for loops in index_loops), np.int64, nr))
        VREG = np.repeat(lreg, lens)                 # the region of every loop point
        uniq, at_v = np.unique(L, return_index=True)
        first = np.argsort(at_v, kind="stable")
        order = uniq[first]
        freg_v = VREG[at_v][first]
        rep, kept = _merge_within_tol(md.verts[order])
        vrows = np.flatnonzero(kept)
        slot = np.cumsum(kept)[rep]                  # the slot of every mesh vertex
        mesh_slot = np.zeros(len(md.verts), dtype=np.int64)
        mesh_slot[order] = slot

        progress(0.25)
        # the loops of the merged vertices: a point that joined the point before it is
        # dropped (the first one too, when it repeats the point that closes the loop),
        # and what is left of a loop has to still be one
        S = mesh_slot[L]
        lrow = np.repeat(np.arange(nloops, dtype=np.int64), lens)   # the loop of a point
        lstart = np.repeat(np.cumsum(lens) - lens, lens)            # where its loop starts
        lend = np.repeat(lens, lens)                                # ... and how long it is
        row = np.arange(len(S))
        prev = np.where(row > lstart, row - 1, lstart + lend - 1)   # the point before it,
        keep = S != S[prev]                                         # the last one for the first
        llen = np.bincount(lrow[keep], minlength=nloops)             # points left in a loop
        rows = np.flatnonzero(llen >= 3)
        Lv = S[keep & (llen[lrow] >= 3)]
        lens2 = llen[rows]
        starts2 = np.cumsum(lens2) - lens2
        rs = lreg[rows]                              # the region of every loop that is left
        if len(rows):
            newface = np.empty(len(rs), dtype=bool)  # a region opens a face at its first loop
            newface[0] = True
            newface[1:] = rs[1:] != rs[:-1]
            ford = np.cumsum(newface) - 1            # the face of every loop
            rof = rs[newface]                        # the region of every face
        else:
            newface = ford = rof = np.zeros(0, dtype=np.int64)
        nf = len(rof)
        cf = np.zeros(nr, dtype=np.int64)
        if nf:
            cf[rof] = 1
        FR = RR = A = B = None
        at = inverse = None
        if nf:
            nxt = np.roll(Lv, -1)
            nxt[starts2 + lens2 - 1] = Lv[starts2]   # close every loop
            FR = np.repeat(ford, lens2)
            RR = np.repeat(rs, lens2)
            ok = Lv != nxt                           # _add_edge_raw makes nothing of a == b
            A, B, FR, RR = Lv[ok], nxt[ok], FR[ok], RR[ok]
            key = np.minimum(A, B) * (int(Lv.max()) + 1) + np.maximum(A, B)
            _, at, inverse = np.unique(key, return_index=True, return_inverse=True)
            ce = np.bincount(RR[at], minlength=nr)
        else:
            ce = np.zeros(nr, dtype=np.int64)

        # the ids of a region follow those of the regions before it
        cv = np.bincount(freg_v[kept], minlength=nr)
        before = np.cumsum(cv + ce + cf) - (cv + ce + cf)
        cumv = np.cumsum(cv)
        vid = np.empty(len(vrows) + 1, dtype=np.int64)
        vid[1:] = 1 + before[freg_v[vrows]] + np.arange(len(vrows)) -             (cumv[freg_v[vrows]] - cv[freg_v[vrows]])
        progress(0.45)
        block = md.verts[order[vrows]].copy()
        ents.V = dict(zip(vid[slot[vrows]].tolist(), list(block)))

        if nf:
            rank = np.argsort(at, kind="stable")     # the order in which the edges appear
            freg_e = RR[at]
            cume = np.cumsum(ce)
            erows = at[rank]
            eids = 1 + before[freg_e[rank]] + cv[freg_e[rank]] +                 np.arange(len(rank)) - (cume[freg_e[rank]] - ce[freg_e[rank]])
            eid = np.empty(len(rank), dtype=np.int64)
            eid[rank] = eids
            ents.E = dict(zip(eids.tolist(),
                              zip(vid[A[erows]].tolist(), vid[B[erows]].tolist())))
            face_id = 1 + before[rof] + cv[rof] + ce[rof]
            progress(0.55)
            normal = md.normals[np.fromiter((md.region_tris[rid][0] for rid in rof.tolist()),
                                            np.int64, nf)]
            pts = vid[Lv].tolist()                   # the vertex of every loop point
            loop_id = [tuple(pts[s:e]) for s, e in zip(starts2.tolist(),
                                                       np.cumsum(lens2).tolist())]
            frow = np.flatnonzero(newface).tolist()  # where the loops of a face start
            fend = frow[1:] + [len(loop_id)]
            for k, s, e, fid in zip(range(nf), frow, fend, face_id.tolist()):
                ents.F[fid] = _face(tuple(loop_id[s:e]), normal[k], color)
            ents.next_id = int(1 + len(vrows) + len(erows) + nf)
            ents.soft = _fold_edges(eid[inverse], FR, nf, normal)
        else:
            ents.next_id = int(1 + len(vrows))
        progress(0.65)
        ents._index()
        ents.touch()
        return ents
