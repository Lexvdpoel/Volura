"""Repairing a triangle mesh that is not one closed volume, so it can still be imported.

Meshes from other programs often have flaws a slicer silently repairs: faces exported on
both sides, triangles pointing the wrong way, small holes, loose "fin" triangles, parts
that touch along an edge, and inner walls where two rooms of a SketchUp model meet. Volura
works with closed volumes only, so `repair_mesh` tries, step by step and cheapest first:

1. merge the corners, drop degenerate and duplicate triangles (also a face that is there on
   both sides) and loose fins;
2. turn the triangles of every connected piece the same way round (the way most of them
   already were), so cavities stay cavities;
3. fill holes: every open boundary gets a lid, flat holes with ear clipping;
4. untangle edges that more than two triangles share (parts that touch along an edge, or
   a surface that passes through itself) by pairing the triangles by their angle around
   the edge, and give every fan of triangles around a corner its own copy of it; the
   closed shells are joined and cavities cut out, so overlapping parts become one volume;
5. a few knots left (edges of an odd number of triangles): cut them out and fill the
   holes, when the lids give back about the same surface;
6. inner walls (a SketchUp model of several rooms): keep only the triangles that can be
   seen from outside; else drop pieces whose whole rim lies on edges of three or more
   triangles, keeping the choice that closes the most volume.

Nothing here knows about the document or Qt. `repair_mesh` returns the closed Manifold (or
None) and a :class:`RepairReport` that says what was done.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import manifold3d as mf

from .geom import manifold_from_mesh, merge_mesh, mesh_arrays
from .i18n import tr, tr_noop

#: a hole with more corners than this gets a fan around its middle instead of ear clipping
EAR_CLIP_MAX = 400

#: at most this many inner-wall candidates are tried (every try builds the mesh once)
MAX_WALL_TRIES = 40
#: at most this many triangles at knots are cut out and refilled; more is no local flaw
MAX_KNOT_TRIANGLES = 200
#: and only when the lids give back the surface that was cut out to within this part of
#: the whole surface: then the knot was a small flaw in a smooth or flat surface
MAX_EXCISED_AREA = 0.02
#: the faces seen from outside are only worked out for meshes up to this many triangles
#: (every triangle casts rays at every other one)
MAX_VISIBILITY_TRIANGLES = 8000

#: inner walls are only looked for in meshes up to this many triangles (SketchUp models
#: are small; a big scanned or sculpted model has other flaws, and every try costs time)
MAX_WALL_TRIANGLES = 50_000


@dataclass
class RepairReport:
    duplicates: int = 0        # duplicate triangles dropped (also faces on both sides)
    fins: int = 0              # loose fin triangles dropped
    flipped: int = 0           # triangles turned round
    holes: int = 0             # holes filled
    crowded: int = 0           # edges of more than two triangles untangled
    walls: int = 0             # inner walls dropped (triangles)
    excised: int = 0           # triangles cut out around knots (and the holes refilled)
    status: object = None      # why it could not be repaired, when it could not

    @property
    def changed(self):
        return any((self.duplicates, self.fins, self.flipped, self.holes, self.crowded,
                    self.walls, self.excised))


# ------------------------------------------------------------------ edges

def _halfedges(t):
    """Every directed edge of the triangles: (a, b) of shape (3M, 2), triangle k at row
    k, M + k and 2M + k."""
    return np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])


def _edge_keys(t):
    """The undirected edge of every halfedge as one int, and how many halfedges share it."""
    h = _halfedges(t)
    n = int(t.max()) + 1 if len(t) else 1
    key = np.minimum(h[:, 0], h[:, 1]) * n + np.maximum(h[:, 0], h[:, 1])
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    return key, inverse.ravel(), count


def _pairs(t, use=None):
    """The triangle pairs that share a manifold edge (one used by exactly two triangles):
    (i, j, same) with `same` True when both run along the edge in the same direction."""
    key, inverse, count = _edge_keys(t)
    use = count[inverse] if use is None else use
    m = len(t)
    rows = np.flatnonzero(use == 2)
    order = rows[np.argsort(inverse[rows], kind="stable")]
    a, b = order[0::2], order[1::2]
    h = _halfedges(t)
    same = h[a, 0] == h[b, 0]
    return a % m, b % m, same


class _Union:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def _components(n, x, y):
    """Connected components of n nodes joined by the pairs (x, y): the smallest node of
    every component as its label. Fast when the components are small (the fan of
    triangles around a corner): the labels spread one step per round."""
    labels = np.arange(n)
    while True:
        low = np.minimum(labels[x], labels[y])
        new = labels.copy()
        np.minimum.at(new, x, low)
        np.minimum.at(new, y, low)
        new = new[new]
        if np.array_equal(new, labels):
            return labels
        labels = new


def _pieces(t):
    """A label per triangle: the triangles that hang together through manifold edges."""
    a, b, _ = _pairs(t)
    u = _Union(len(t))
    for i, j in zip(a.tolist(), b.tolist()):
        u.join(i, j)
    return u.labels()


# ------------------------------------------------------------------ the steps

def _clean(verts, tris, report):
    v, t = merge_mesh(verts, tris)
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    # duplicates regardless of their direction: one copy stays, its direction is settled
    # when the piece is oriented
    _, first = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    report.duplicates += len(t) - len(first)
    t = t[np.sort(first)]
    from .fileio import _drop_fins
    t, fins = _drop_fins(t)
    report.fins += fins
    return v, t


def _orient(t, report):
    """Turn the triangles of every piece the same way round as most of that piece."""
    if not len(t):
        return t
    a, b, same = _pairs(t)
    adj = [[] for _ in range(len(t))]
    for i, j, s in zip(a.tolist(), b.tolist(), same.tolist()):
        adj[i].append((j, s))
        adj[j].append((i, s))
    flip = np.zeros(len(t), dtype=bool)
    seen = np.zeros(len(t), dtype=bool)
    for start in range(len(t)):
        if seen[start]:
            continue
        seen[start] = True
        piece, stack = [start], [start]
        while stack:
            i = stack.pop()
            for j, s in adj[i]:
                if not seen[j]:
                    seen[j] = True
                    flip[j] = flip[i] ^ s      # running the same way: one of them turns
                    piece.append(j)
                    stack.append(j)
        if flip[piece].sum() * 2 > len(piece):  # keep the direction most of them had
            flip[piece] = ~flip[piece]
    report.flipped += int(flip.sum())
    t = t.copy()
    t[flip] = t[flip][:, [0, 2, 1]]
    return t


def _ear_clip(pts2):
    """Triangles (index triples) of a simple polygon given counter-clockwise; None when
    no ear is found (a polygon that crosses itself)."""
    idx = list(range(len(pts2)))
    out = []

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 4 * len(pts2) * len(pts2) + 16:
            return None
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            p0, p1, p2 = pts2[i0], pts2[i1], pts2[i2]
            if cross(p0, p1, p2) <= 1e-12:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = pts2[j]
                if (cross(p0, p1, q) >= 0 and cross(p1, p2, q) >= 0 and cross(p2, p0, q) >= 0):
                    inside = True
                    break
            if not inside:
                out.append((i0, i1, i2))
                del idx[k]
                break
        else:
            return None
    out.append(tuple(idx))
    return out


def _boundary_loops(t):
    """The open boundaries as loops of vertices, in the direction their triangles run."""
    key, inverse, count = _edge_keys(t)
    h = _halfedges(t)
    open_rows = np.flatnonzero(count[inverse] == 1)
    nxt = {}
    for a, b in h[open_rows].tolist():
        nxt.setdefault(a, []).append(b)
    loops = []
    while nxt:
        start = next(iter(nxt))
        path, at = [start], {start: 0}
        a = start
        while True:
            outs = nxt.get(a)
            if not outs:
                break                          # a boundary that does not close: left open
            b = outs.pop()
            if not outs:
                del nxt[a]
            if b in at:
                # back at a corner of the path: that stretch is a loop of its own (two
                # holes that touch in one corner are two loops)
                k = at[b]
                loop = path[k:]
                if len(loop) >= 3:
                    loops.append(loop)
                for c in loop[1:]:
                    del at[c]
                path = path[:k + 1]
                a = b
                if not nxt.get(a):
                    break
                continue
            at[b] = len(path)
            path.append(b)
            a = b
    return loops


def _fill_holes(v, t, report):
    """Close every open boundary with a lid running against it."""
    loops = _boundary_loops(t)
    if not loops:
        return v, t
    verts, new = [v], []
    count = len(v)
    for loop in loops:
        lid = loop[::-1]                       # the lid runs against the boundary
        P = v[lid]
        n = np.zeros(3)
        for k in range(len(P)):                # Newell: the normal of the lid
            p, q = P[k], P[(k + 1) % len(P)]
            n += np.array([(p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]),
                           (p[0] - q[0]) * (p[1] + q[1])])
        tris = None
        if np.linalg.norm(n) > 1e-12 and len(lid) <= EAR_CLIP_MAX:
            n = n / np.linalg.norm(n)
            u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
            u /= np.linalg.norm(u)
            w = np.cross(n, u)
            ear = _ear_clip([(float(p @ u), float(p @ w)) for p in P])
            if ear is not None:
                tris = [(lid[a], lid[b], lid[c]) for a, b, c in ear]
        if tris is None:                      # a fan around a new point in the middle
            verts.append(P.mean(axis=0)[None, :])
            c = count
            count += 1
            tris = [(lid[k], lid[(k + 1) % len(lid)], c) for k in range(len(lid))]
        new.extend(tris)
        report.holes += 1
    v = np.concatenate(verts)
    return v, np.concatenate([t, np.asarray(new, dtype=np.int64).reshape(-1, 3)])


def _flipped(m):
    v, t = mesh_arrays(m)
    return manifold_from_mesh(v, t[:, [0, 2, 1]])


def _resolve(shells):
    """One volume of closed shells: the outward ones joined, the inward ones (cavities)
    cut out of what holds them; an inward shell that nothing holds is turned round."""
    pos = [s for s in shells if s.volume() > 0]
    neg = [_flipped(s) for s in shells if s.volume() < 0]
    solid = mf.Manifold.batch_boolean(pos, mf.OpType.Add) if pos else mf.Manifold()
    holes = []
    for c in neg:
        if solid.is_empty() or (c - solid).volume() > 0.5 * c.volume():
            solid = solid + c                  # not inside anything: a solid turned inside out
        else:
            holes.append(c)
    if holes:
        solid = solid - mf.Manifold.batch_boolean(holes, mf.OpType.Add)
    return solid


def _pair_edges(v, t):
    """Pair every directed edge with one running the other way, also where more than two
    triangles meet at an edge: there the triangles are sorted by their angle around the
    edge and every one is paired with its neighbour on the side where the material is.
    Returns the pairs (halfedge rows) and the number of crowded edges they untangled, or
    None when an edge cannot be paired (an odd number of triangles, or directions that do
    not alternate around it)."""
    key, inverse, count = _edge_keys(t)
    h = _halfedges(t)
    m = len(t)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(count)])
    two = count == 2
    rows2 = order.reshape(-1)  # grouped by edge
    pairs = []
    # the common case at once: edges of exactly two triangles running opposite ways
    first = starts[:-1][two]
    a, b = rows2[first], rows2[first + 1]
    if (h[a, 0] == h[b, 0]).any():
        return None
    pairs.append(np.stack([a, b], axis=1))
    crowded = 0
    for e in np.flatnonzero(~two).tolist():
        rows = rows2[starts[e]:starts[e + 1]]
        if len(rows) % 2:
            return None
        crowded += 1
        p, q = h[rows[0]]
        A, B = v[p], v[q]
        d = B - A
        d = d / (np.linalg.norm(d) or 1.0)
        x = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(d, x)
        ang = []
        for r in rows:
            c = v[t[r % m]].sum(axis=0) - v[h[r, 0]] - v[h[r, 1]]   # the third corner
            w = c - A
            ang.append(np.arctan2(w @ y, w @ x))
        ring = [rows[k] for k in np.argsort(ang)]
        forward = [h[r, 0] == p for r in ring]
        # a triangle running q -> p has its material just counter-clockwise of it: it
        # pairs with the next triangle counter-clockwise, which has to run p -> q
        start = next((k for k in range(len(ring)) if not forward[k]), None)
        if start is None or sum(forward) * 2 != len(ring):
            return None
        ring = ring[start:] + ring[:start]
        forward = forward[start:] + forward[:start]
        if all(forward[k] == (k % 2 == 1) for k in range(len(ring))):
            pairs.append(np.array(ring).reshape(-1, 2))
        else:
            # the triangles cross each other around this edge (the surface passes through
            # itself): pair them in turn, the shape stays as the file has it
            back = [r for r, f in zip(ring, forward) if not f]
            ahead = [r for r, f in zip(ring, forward) if f]
            pairs.append(np.stack([back, ahead], axis=1))
    return np.concatenate(pairs), crowded


def _split_corners(t, pairs):
    """Give every fan of triangles around a corner its own copy of that corner: corners of
    paired triangles on the same vertex stay together, the rest come apart. Returns the
    new triangles and, per new vertex, the old vertex it copies."""
    m = len(t)
    h = _halfedges(t)
    # corner k*... : corner (triangle i, slot s) is row s * m + i, the start of halfedge s
    # halfedge r1 runs a -> b, r2 runs b -> a: corner a of r1 joins corner a of r2, which
    # is the end of r2, the start of the next halfedge of its triangle; the same for b
    i1, s1 = pairs[:, 0] % m, pairs[:, 0] // m
    i2, s2 = pairs[:, 1] % m, pairs[:, 1] // m
    x = np.concatenate([s1 * m + i1, ((s1 + 1) % 3) * m + i1])
    y = np.concatenate([((s2 + 1) % 3) * m + i2, s2 * m + i2])
    labels = _components(3 * m, x, y)
    _, new_id, = np.unique(labels, return_inverse=True)
    corner_vertex = h[:, 0]                       # the vertex of every corner
    source = np.zeros(int(new_id.max()) + 1, dtype=np.int64)
    source[new_id] = corner_vertex
    tris = np.stack([new_id[:m], new_id[m:2 * m], new_id[2 * m:]], axis=1)
    return tris, source


def _build(v, t):
    """The mesh as one volume: crowded edges untangled and corners split where only a
    corner or an edge is shared, every closed shell built on its own; overlapping shells
    are joined and cavities cut out. Returns (Manifold or None, pieces untangled, status)."""
    if not len(t):
        return None, 0, "Empty"
    paired = _pair_edges(v, t)
    if paired is None:
        return None, 0, mf.Error.NotManifold
    pairs, crowded = paired
    tris, source = _split_corners(t, pairs)
    mesh = mf.Mesh64(vert_properties=np.ascontiguousarray(v[source], dtype=np.float64),
                     tri_verts=np.ascontiguousarray(tris, dtype=np.uint64))
    whole = mf.Manifold(mesh)
    if whole.status() != mf.Error.NoError or whole.is_empty():
        return None, crowded, whole.status()
    shells = whole.decompose()
    solid = _resolve(shells) if (len(shells) > 1 or crowded or whole.volume() < 0) \
        else whole
    if solid.is_empty() or solid.status() != mf.Error.NoError:
        return None, crowded, solid.status()
    return solid, crowded, mf.Error.NoError


def _walls(t):
    """Candidate inner walls: pieces whose whole rim lies on edges that more than two
    triangles share (where a wall meets the outside)."""
    key, inverse, count = _edge_keys(t)
    use = count[inverse]
    m = len(t)
    labels = _pieces(t)
    rim_ok = {}
    for row in np.flatnonzero(use != 2).tolist():
        lab = labels[row % m]
        rim_ok[lab] = rim_ok.get(lab, True) and use[row] >= 3
    pieces = [lab for lab, ok in rim_ok.items() if ok]
    sizes = {lab: int((labels == lab).sum()) for lab in pieces}
    return labels, sorted(pieces, key=lambda lab: sizes[lab])


def _drop_walls(v, t, report):
    """Drop inner walls one by one: every time the candidate that leaves the most closed
    volume (or, while nothing closes yet, the fewest edges shared by three or more)."""
    tries = 0
    while tries < MAX_WALL_TRIES:
        labels, candidates = _walls(t)
        if not candidates:
            return None, t
        best = None
        for lab in candidates:
            if tries >= MAX_WALL_TRIES:
                break
            tries += 1
            rest = _orient(t[labels != lab], RepairReport())
            m, _, _ = _build(v, rest)
            if m is not None:
                score = (1, m.volume())
            else:
                _, inverse, count = _edge_keys(rest)
                score = (0, -int((count > 2).sum()))
            if best is None or score > best[0]:
                best = (score, lab, rest, m)
        if best is None:
            return None, t
        score, lab, rest, m = best
        report.walls += int((labels == lab).sum())
        t = rest
        if m is not None:
            return m, t
    return None, t


def _odd_edges(t):
    """The triangles at an edge that an odd number of triangles share (an open edge, or
    one of three): those can never be paired."""
    key, inverse, count = _edge_keys(t)
    odd = (count[inverse] % 2 == 1).reshape(3, -1).T.any(axis=1)
    return odd


def _area(v, t):
    tri = v[t]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _excise(v, t, report, rounds=3):
    """Cut out the triangles around a few knots (edges of an odd number of triangles) and
    fill the holes that leaves: a small local repair, as a slicer makes. Returns the mesh
    and the part of the surface that was cut out (0..1)."""
    total = float(_area(v, t).sum()) or 1.0
    cut = 0.0
    for _ in range(rounds):
        bad = _odd_edges(t)
        if not bad.any():
            break
        # the knot and the ring of triangles around its corners
        corners = np.unique(t[bad])
        near = np.isin(t, corners).any(axis=1)
        report.excised += int(near.sum())
        cut += float(_area(v, t[near]).sum())
        t = _orient(t[~near], RepairReport())
        v, t = _fill_holes(v, t, report)
        t = _orient(t, RepairReport())
    return v, t, cut / total


def _ray_hits(origins, direction, tri):
    """For rays from `origins` all along one `direction`: does each hit a triangle?

    Seen along the direction a ray is a point and a triangle a flat triangle: the ray
    hits when the point lies in it and the triangle is further along there."""
    d = direction / np.linalg.norm(direction)
    x = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    P = np.stack([tri @ x, tri @ y], axis=-1)              # (T, 3, 2)
    Z = tri @ d                                             # (T, 3)
    a, e1, e2 = P[:, 0], P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(den) > 1e-18                              # triangles seen edge-on: no hit
    a, e1, e2, den, Z = a[keep], e1[keep], e2[keep], den[keep], Z[keep]
    ca = a[:, 0] * e2[:, 1] - a[:, 1] * e2[:, 0]
    cb = e1[:, 0] * a[:, 1] - e1[:, 1] * a[:, 0]
    ox, oy, oz = origins @ x, origins @ y, origins @ d
    hit = np.zeros(len(origins), dtype=bool)
    step = max(1, 4_000_000 // max(len(den), 1))
    for k in range(0, len(origins), step):
        X, Y = ox[k:k + step, None], oy[k:k + step, None]
        u = (X * e2[:, 1] - Y * e2[:, 0] - ca) / den
        w = (e1[:, 0] * Y - e1[:, 1] * X - cb) / den
        z = Z[:, 0] + u * (Z[:, 1] - Z[:, 0]) + w * (Z[:, 2] - Z[:, 0])
        hit[k:k + step] = ((u >= 0) & (w >= 0) & (u + w <= 1)
                           & (z > oz[k:k + step, None])).any(axis=1)
    return hit


def _sphere(n):
    """n directions spread evenly over the sphere (a Fibonacci spiral)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)


def _seen_from_outside(v, t, rays=48):
    """Which triangles can be seen from outside the model: from at least one of its two
    sides a ray gets away without hitting another triangle. Inner walls cannot."""
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    c = tri.mean(axis=1)
    size = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) or 1.0
    eps = 1e-6 * size
    dirs = _sphere(2 * rays)
    seen = np.zeros(len(t), dtype=bool)
    for side in (1.0, -1.0):
        todo = np.flatnonzero(~seen)
        if not len(todo):
            break
        cos = (n[todo] * side) @ dirs.T                   # (todo, 2 * rays)
        out = np.zeros(len(todo), dtype=bool)
        for j in range(len(dirs)):
            rows = np.flatnonzero((cos[:, j] > 0.05) & ~out)
            if len(rows):
                origins = c[todo[rows]] + side * eps * n[todo[rows]]
                out[rows] |= ~_ray_hits(origins, dirs[j], tri)
        seen[todo[out]] = True
    return seen


def _attempt(v, t, report):
    """Build the mesh; when only a few knots are in the way, cut them out and try again.
    Returns (Manifold or None, crowded edges, status, verts, tris)."""
    m, crowded, status = _build(v, t)
    if m is None and int(_odd_edges(t).sum()) <= MAX_KNOT_TRIANGLES:
        r2 = RepairReport()
        v2, t2, cut = _excise(v, t, r2)
        before, after = float(_area(v, t).sum()), float(_area(v2, t2).sum())
        same = abs(after - before) <= MAX_EXCISED_AREA * before
        m2, crowded2, status2 = _build(v2, t2) if same else (None, 0, None)
        if m2 is not None:
            report.excised += r2.excised
            report.holes += r2.holes
            return m2, crowded2, status2, v2, t2
    return m, crowded, status, v, t


def repair_mesh(verts, tris, progress=None):
    """Try to make one closed volume of a triangle soup: ``(Manifold or None, report)``."""
    progress = progress or (lambda fraction: None)
    report = RepairReport()
    verts = np.asarray(verts, dtype=float).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if not len(tris):
        report.status = "Empty"
        return None, report
    v, t = _clean(verts, tris, report)
    progress(0.1)
    t = _orient(t, report)
    progress(0.2)
    v, t = _fill_holes(v, t, report)
    t = _orient(t, RepairReport())             # the lids follow their surroundings
    progress(0.3)
    m, crowded, status, v, t = _attempt(v, t, report)
    progress(0.6)
    if m is None and len(t) <= MAX_VISIBILITY_TRIANGLES:
        # inner walls (a SketchUp model of several rooms): keep what is seen from outside
        seen = _seen_from_outside(v, t)
        if not seen.all():
            r2 = RepairReport(walls=int((~seen).sum()))
            t2 = _orient(t[seen], r2)
            v2, t2 = _fill_holes(v, t2, r2)
            t2 = _orient(t2, RepairReport())
            m2, crowded2, status2, v2, t2 = _attempt(v2, t2, r2)
            if m2 is not None:
                m, crowded, v, t = m2, crowded2, v2, t2
                report.walls += r2.walls
                report.holes += r2.holes
                report.excised += r2.excised
    progress(0.8)
    if m is None and len(t) <= MAX_WALL_TRIANGLES:
        m, t = _drop_walls(v, t, report)
        if m is not None:
            crowded = _build(v, t)[1]
    if m is not None and m.volume() <= 1e-9 * max(1.0, _box_volume(v)):
        m, status = None, "Flat"                # a sheet folded onto itself holds nothing
    if m is None:
        report.status = status
        return None, report
    report.crowded = crowded
    progress(1.0)
    return m, report


def _box_volume(v):
    size = v.max(axis=0) - v.min(axis=0)
    return float(np.prod(size))


def describe(report):
    """What a repair did, as translated phrases ("3 holes filled", ...)."""
    said = []
    for n, text in ((report.holes, tr_noop("{n} holes filled")),
                    (report.flipped, tr_noop("{n} triangles turned round")),
                    (report.duplicates, tr_noop("{n} duplicate triangles removed")),
                    (report.fins, tr_noop("{n} loose triangles removed")),
                    (report.walls, tr_noop("{n} triangles of inner walls removed")),
                    (report.excised, tr_noop("{n} triangles around knots rebuilt")),
                    (report.crowded, tr_noop("{n} crossing edges untangled"))):
        if n:
            said.append(tr(text).format(n=n))
    return said
