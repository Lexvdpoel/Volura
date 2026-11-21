"""Which faces of a model come closer to another part than the printer's clearance.

A lid in a box, a pin in a hole, a hinge that has to move: parts that are printed with
less room between them than the machine lays down come off the bed as one brick, and
nothing moves any more. The tight fit view (View > Face Style > Tight Fits) colours those
faces, the way the thin wall view colours what is thinner than the nozzle.

The rule is the one no slicer can save: a face is tight where it stands closer to another
part than the clearance the printer proved. Every part is asked about every other one,
which starts with two cheap no's - the boxes of the two parts, and the gap between the
solids - before a single triangle is measured, so two parts that stand far apart cost
nothing at all.

What remains is the distance of a face centre to the surface of the other part, and
measuring every centre against every triangle of that part is what a heavy model cannot
pay. So the triangles stand sorted in cells the width of the clearance: a point is
measured against the triangles of its own cell and of the cells around it, which are all
of them that can be close, and of those only against the ones whose own box does come
within the clearance of it. A triangle whose own box covers more cells than that is
too big for such a cell to be worth its memory and is kept aside, to be asked by every
point. A centre with nothing near it is not thereby free: it may lie inside the other
part, which is the tightest place of all. That is answered by the even-odd test of a ray
straight up, and the cells of that grid are numbered with z first so the whole column a
ray runs through is one run of cells and one search after them.
"""
from __future__ import annotations

import numpy as np

from .i18n import tr

#: the most cells of the search grid on one axis; a wide part is searched with wider cells
MAX_CELLS = 128
#: how many (point, triangle) pairs one round of a test holds in memory
PAIRS = 200_000
#: a triangle whose box covers more cells than this is no longer worth a place in the grid
BIG = 128
#: how many parts keep their triangles and their grid ready at the same time
KEEP = 3
#: the cells one point is measured against: its own and every one around it
RING = [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
#: mm the ray of the even-odd test leans to the side of a point: too small for any feature
#: of a model, far too big to be lost by double arithmetic at the size of a print bed
HAIR = 1e-6
#: the two steps of those leans: two irrational ratios, so no two points lean alike
GOLD = (5 ** 0.5 - 1) / 2
SQRT2 = 2 ** 0.5 - 1.0


def tight_masks(parts, clearance):
    """Which points of every part of a model stand closer than `clearance` to another part.

    `parts` is a list of ``(manifold, points)``, one per part of the model: its solid and
    the centres of the faces of the view that draws it, ``(n, 3)`` in the same millimetres
    as the solid. The answer is one bool array per part, True where a point comes closer
    to another part than `clearance` - touching, overlapping and lying inside another part
    all count as tight, because none of them leaves room for the plastic a printer would
    put between the two.

    A pair of parts whose boxes, grown by the clearance, still do not touch is answered
    without measuring a single triangle, and so is a pair whose solids are further apart
    than the clearance; what the machine cannot keep that straight is not worth measuring
    to a hundredth of a millimetre.

    A `clearance` of zero or less is no room at all and raises ``ValueError``; a model of
    fewer than two parts, a part without points, a part that is no solid and a part that
    only another part touches where it is not close say nothing is tight.
    """
    if not clearance > 0:
        raise ValueError(tr("the clearance must be greater than 0 mm"))
    solids = [None if p is None or p[0] is None else p[0] for p in parts]
    pts = [np.asarray([] if p is None else p[1], dtype=float).reshape(-1, 3) for p in parts]
    tight = [np.zeros(len(p), dtype=bool) for p in pts]
    box = [None if s is None or s.is_empty() else np.asarray(s.bounding_box(), dtype=float)
           for s in solids]
    ready = {}
    for i in range(len(parts) - 1):
        for j in range(i + 1, len(parts)):
            if box[i] is None or box[j] is None or not len(pts[i]) or not len(pts[j]):
                continue
            if not _touch(box[i], box[j], clearance):
                continue
            gap = _gap(solids[i], solids[j], clearance + 0.01)
            if gap is not None and gap >= clearance:
                continue
            here = _ready(ready, solids, i, clearance, i, j)     # part i as a target
            there = _ready(ready, solids, j, clearance, i, j)    # and part j as one
            _near(pts[i], tight[i], there, clearance)
            _near(pts[j], tight[j], here, clearance)
    return tight


def _ready(ready, solids, k, clearance, i, j):
    """Part `k` as something to measure against: made once and kept, because the same part
    is a target for every part it comes close to.

    Only :data:`KEEP` of them are kept at a time; the ones this pair does not need go first,
    so a model of many parts that stand near one another does not hold a grid for every
    part of it at once."""
    got = ready.get(k)
    if got is None:
        ready[k] = got = _Target(solids[k], clearance)
        while len(ready) > KEEP:
            for old in sorted(ready):
                if old != i and old != j:
                    del ready[old]
                    break
            else:
                break
    return got


def _touch(a, b, clearance):
    """Whether the boxes `a` and `b`, grown by `clearance` millimetres, overlap at all."""
    return bool(np.all(a[:3] - clearance <= b[3:] + clearance)
                and np.all(b[:3] - clearance <= a[3:] + clearance))


def _gap(a, b, search):
    """The gap between two solids, no more than `search`; None when manifold3d cannot tell.

    Two solids that touch or run into each other have a gap of nothing, which is exactly
    what a search of the clearance is asked for. A pair the kernel will not measure is
    taken for close, because saying it is apart would hide a tight place in it."""
    try:
        return float(a.min_gap(b, search))
    except Exception:               # a pair the kernel refuses is no reason to let it pass
        return None


class _Target:
    """One part as something to measure the other parts against: its triangles, the box of
    every one of them, the box around the part and the grid of cells they stand sorted in.

    That work is done once for a part and asked again for every other part it comes close
    to, which is what makes the pairs of a model with many parts in one place no dearer
    than the pairs of a model of two."""

    def __init__(self, solid, clearance):
        mesh = solid.to_mesh()
        v = np.asarray(mesh.vert_properties, dtype=float).reshape(-1, 3)[:, :3]
        self.p = v[np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)]
        self.lo = self.p.min(axis=1)            # the box of every triangle
        self.hi = self.p.max(axis=1)
        self.box = np.asarray(solid.bounding_box(), dtype=float)
        self.grid = _Grid(self.p, self.box, clearance)

    def block(self, per):
        """How many points one round of a test may take, so that `per` triangles and their
        answers for each of them stay in memory together."""
        return max(1, PAIRS // max(1, per))


class _Grid:
    """The triangles of a part, sorted into cubic cells of the clearance.

    A point is measured against its own cell and the ones around it, and that is every
    triangle that can be within the clearance: a cell is at least as wide of that, so the
    sphere of that radius round the point never reaches into a cell that is not next to it.
    The grid stands one cell wider than the part on every side, so the cells beside the
    part are cells that hold nothing rather than cells that are not there; the triangles
    whose own box covers more than :data:`BIG` cells do not fit that idea and are kept in
    :attr:`always`, to be gathered by every point however far it stands.

    A cell is numbered with z first, so the cells of one column stand in one run of
    numbers and the part of a column above a point is one range of them."""

    def __init__(self, p, box, clearance):
        size = np.maximum(box[3:] - box[:3], 0.0)
        far = float(size.max())
        cell = max(float(clearance), far / MAX_CELLS if far > 0 else 0.0)
        self.lo, self.cell = box[:3], cell
        self.n = np.maximum(1, np.ceil(size / cell).astype(np.int64))
        self.wx = int(self.n[0]) + 2
        self.wy = int(self.n[1]) + 2
        self.wz = int(self.n[2]) + 2
        i0 = np.clip(np.floor((p.min(axis=1) - self.lo) / cell).astype(np.int64) + 1, 1, self.n)
        i1 = np.clip(np.floor((p.max(axis=1) - self.lo) / cell).astype(np.int64) + 1, 1, self.n)
        span = i1 - i0 + 1
        count = span[:, 0] * span[:, 1] * span[:, 2]
        self.always = np.flatnonzero(count > BIG)
        keep = np.flatnonzero(count <= BIG)
        key, tri = self._cells_of(i0[keep], span[keep], keep)
        order = np.argsort(key, kind="stable")
        key, self.data = key[order], tri[order]
        if len(key):
            first = np.r_[True, key[1:] != key[:-1]]
            self.U = key[first]
            self.start = np.r_[np.flatnonzero(first), len(key)]
        else:                       # every triangle of the part was too big for a cell
            self.U = key
            self.start = np.zeros(1, dtype=np.int64)

    def _cells_of(self, i0, span, tri):
        """The key of every cell the box of one of `tri` touches, with the triangle that
        belongs to it: the boxes go one axis at a time into lists that grow to hold them."""
        rows = np.arange(len(tri))
        key = np.zeros(len(rows), dtype=np.int64)
        if not len(rows):
            return key, np.zeros(0, dtype=np.int64)
        for axis, stride in enumerate((self.wz * self.wy, self.wz, 1)):
            count = span[rows, axis]
            inner = (np.arange(int(count.sum()))
                     - np.repeat(np.r_[0, np.cumsum(count)[:-1]], count))
            add = (np.repeat(i0[rows, axis], count) + inner) * stride
            rows, key = np.repeat(rows, count), np.repeat(key, count) + add
        return key, tri[rows]

    def cells(self, q):
        """The cell every point of `q` stands in, as coordinates on the three axes. A point
        beside the part stands in a cell outside the part, which holds nothing."""
        return np.floor((q - self.lo) / self.cell).astype(np.int64) + 1

    def _here(self, i, dx, dy, dz):
        """The cells `offsets` away from every cell of `i`, as keys and the rows they
        belong to; a cell outside the grid is dropped, so nothing from another column of
        the grid can be found under its number."""
        jx, jy, jz = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
        ok = ((jx >= 0) & (jx < self.wx) & (jy >= 0) & (jy < self.wy) & (jz >= 0) & (jz < self.wz))
        at = np.flatnonzero(ok)
        if not len(at):
            return at, np.zeros(0, dtype=np.int64)
        return at, jz[at] + self.wz * (jy[at] + self.wy * jx[at])

    def count(self, i, offsets):
        """How many triangles the cells `offsets` away from every cell of `i` hold."""
        found = np.zeros(len(i), dtype=np.int64)
        if not len(self.U):
            return found
        for dx, dy, dz in offsets:
            at, key = self._here(i, dx, dy, dz)
            if len(at):
                found[at] += self._sizes(key)
        return found

    def gather(self, i, offsets):
        """The triangles of the cells `offsets` away from every cell of `i`, as ``(row,
        tri)`` where `row` counts the cells that gathered something. A cell outside the
        grid and a cell with nothing in it gather nothing."""
        rows_out, tri_out = [], []
        if len(self.U):
            for dx, dy, dz in offsets:
                at, key = self._here(i, dx, dy, dz)
                if len(at):
                    rows_out.append(np.repeat(at, self._sizes(key)))
                    tri_out.append(self.data[self._runs(key)])
        if not rows_out:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(rows_out), np.concatenate(tri_out)

    def above(self, i):
        """The triangles of the cells at and above every cell of `i` in its own column, as
        ``(row, tri)``; the column of a cell outside the grid gathers nothing. A triangle
        whose box covers several cells of the column is named once for every one of them."""
        if not len(self.U):
            return np.zeros(len(i), dtype=np.int64), np.zeros(0, dtype=np.int64)
        count, begin = self._column(i)
        row = np.repeat(np.arange(len(i)), count)
        inner = np.arange(int(count.sum())) - np.repeat(np.r_[0, np.cumsum(count)[:-1]], count)
        return row, self.data[np.repeat(begin, count) + inner]

    def count_above(self, i):
        """How many triangles the column of every cell of `i` holds above it, `above`."""
        if not len(self.U):
            return np.zeros(len(i), dtype=np.int64)
        return self._column(i)[0]

    def _column(self, i):
        """Where in :attr:`data` the triangles of the column of every cell of `i` stand, and
        how many of them there are: one column is one run of cell numbers, so one pair of
        searches finds where it begins and where the part above the cell begins in it."""
        base = self.wz * (np.clip(i[:, 1], 0, self.wy - 1)
                          + self.wy * np.clip(i[:, 0], 0, self.wx - 1))
        lo, hi = np.searchsorted(self.U, [base + np.clip(i[:, 2], 0, self.wz), base + self.wz])
        # the searches answer with a place among the cells, and a cell begins where the run
        # of triangles it holds begins
        at, beg = self.start[lo], self.start[hi]
        here = ((i[:, 0] >= 0) & (i[:, 0] < self.wx)
                & (i[:, 1] >= 0) & (i[:, 1] < self.wy))
        return np.where(here, beg - at, 0), at

    def _sizes(self, key):
        """How many triangles every one of the cells of `key` holds; a cell that is not in
        the grid holds nothing, and its number may stand past the last one."""
        pos = np.searchsorted(self.U, key)
        got = (pos < len(self.U)) & (self.U[np.minimum(pos, len(self.U) - 1)] == key)
        return np.where(got, self.start[np.minimum(pos + 1, len(self.U))] - self.start[pos], 0)

    def _runs(self, key):
        """The places in :attr:`data` where the triangles of the cells of `key` stand, in
        one run: `count` of them, so a row can be repeated with them."""
        pos = np.searchsorted(self.U, key)
        here = self.start[np.minimum(pos, len(self.U) - 1)]
        count = self._sizes(key)
        inner = (np.arange(int(count.sum()))
                 - np.repeat(np.r_[0, np.cumsum(count)[:-1]], count))
        return np.repeat(here, count) + inner


def _near(points, hit, part, clearance):
    """Mark in `hit` which of `points` stand closer than `clearance` to `part`.

    Only the points within the clearance of the box of the part are asked - beside that box
    there is nothing of it to be close to - and a point that another part made tight is
    not asked again. What the distance cannot answer is asked of the surface: a point
    buried in the other part lies further from it than the clearance and is tight all the
    same, because there is no room left for plastic where two parts share one place."""
    todo = np.flatnonzero(_in_box(points, part.box, clearance) & ~hit)
    if not len(todo):
        return
    found = _distances(part, points[todo], clearance) < clearance * clearance
    hit[todo[found]] = True
    rest = todo[~found]
    if len(rest):
        hit[rest[_inside(part, points[rest])]] = True


def _in_box(points, box, clearance):
    """Whether every point of `points` lies in `box`, grown by `clearance` millimetres."""
    return (np.all(points >= box[None, :3] - clearance, axis=1)
            & np.all(points <= box[None, 3:] + clearance, axis=1))


def _rounds(count, budget):
    """The slices that cut a run of rows with `count` pairs each into rounds of no more
    than `budget` pairs; a row bigger than the budget on its own is a round of its own."""
    if not len(count):
        return []
    total = np.cumsum(count, dtype=np.int64)
    if not total[-1]:
        return []
    marks = np.arange(budget, int(total[-1]), budget, dtype=np.int64)
    cut = np.unique(np.r_[0, np.searchsorted(total, marks, side="left"), len(count)])
    return [slice(a, b) for a, b in zip(cut[:-1], cut[1:]) if b > a]


def _distances(part, q, clearance):
    """The squared distance of every point of `q` to the nearest triangle of `part`, or
    infinity where nothing of the part comes within `clearance` of it.

    A point is measured against the triangles of the cells around it, in rounds of as many
    points as fit in memory, and of those only against the ones that can still be the close
    one the view asks about."""
    grid = part.grid
    d2 = np.full(len(q), np.inf)
    cell = grid.cells(q)
    limit = clearance * clearance
    for rows in _rounds(grid.count(cell, RING), PAIRS):
        at, tri = grid.gather(cell[rows], RING)
        _take(d2[rows], q[rows][at], at, tri, part, limit)
    for here, row, tri in _big(part, len(q)):
        # a triangle too big for the grid can lie close to a point from any distance
        _take(d2[here], q[here][row], row, tri, part, limit)
    return d2


def _take(d2, p, row, tri, part, limit):
    """Fold the distance of every point of `p` to the triangle of `part` named by `tri`
    beside it into `d2`, the answer so far of every row of this round - a view of the
    answers of the whole, so what lands here lands in them.

    Two questions decide whether a triangle is measured at all: one triangle whose box
    stands further from the point than the clearance can never be the close one the view
    asks about, and one that stands further than the answer the point already has can
    neither make it nearer nor farther away. The cells round a point reach as far as a cell
    and hold everything that reaches that far, so most of what they hold answers both no."""
    off = np.maximum(part.lo[tri] - p, 0.0) + np.maximum(p - part.hi[tri], 0.0)
    near = (off * off).sum(axis=1) < np.minimum(limit, d2[row])
    if near.any():
        np.minimum.at(d2, row[near], _closest(p[near], part.p[tri[near]]) ** 2)


def _big(part, n):
    """The triangles that are too big for the grid, paired with every one of `n` points, in
    rounds that hold: the points of the round, which point of it every pair belongs to, and
    the triangle of it. Such a triangle can be close to a point from any distance, so it
    stands in every round."""
    always = part.grid.always
    if not len(always):
        return
    step = part.block(len(always))
    for s in range(0, n, step):
        here = slice(s, min(s + step, n))
        row = np.repeat(np.arange(here.stop - here.start), len(always))
        yield here, row, np.tile(always, (here.stop - here.start,))


def _inside(part, q):
    """Whether every point of `q` lies in `part`, as one bool per point.

    The even-odd test of a ray straight up: a point is inside a solid when its surface
    crosses a ray from it an odd number of times. Only the triangles of the column the ray
    runs through can cross it, and of those only the ones above the point, so a point is
    asked about a handful of the triangles of the part - which is what makes it worth
    asking, and only of the points that have nothing near them.

    The ray of every point leans a hair of a millimetre to its own side, by a step that
    never lands on the same place twice, because a point that stands exactly above the
    edge two faces share would count that edge twice or not at all and the answer would
    come out the wrong way round. The lean is never along an axis, a diagonal or an
    anti-diagonal, which are the four directions an edge of an upright model can run in,
    and so no point is left leaning along the very edge it meant to step over. The points
    this moves over an edge are the ones whose answer was never sure to begin with."""
    grid, p = part.grid, part.p
    cell = grid.cells(q)
    cross = np.zeros(len(q), dtype=np.int64)
    step = np.arange(len(q))
    x = 0.13 + 0.71 * ((step * GOLD) % 1.0)
    y = 0.29 + 0.53 * ((step * SQRT2) % 1.0)
    alike = np.abs(x - y) < 0.05
    y[alike] += 0.41
    q = q + HAIR * np.stack((x, y, np.zeros(len(q))), axis=1)
    for rows in _rounds(grid.count_above(cell), PAIRS):
        at, tri = grid.above(cell[rows])
        if len(tri):
            hit = _once(_crossings(q[rows][at], p[tri]), at, tri)
            cross[rows] += _count(hit, at, rows.stop - rows.start)
    for here, row, tri in _big(part, len(q)):
        # a triangle too big for the grid crosses the ray of every point that stands under it
        cross[here] += _count(_crossings(q[here][row], p[tri]), row, here.stop - here.start)
    return (cross & 1) == 1


def _once(hit, row, tri):
    """The crossings of `hit` that count: a triangle whose box covers several cells of one
    column is gathered from every one of them, but its surface crosses the ray of a point
    once, however many cells it stands in - and counted twice it turns a point inside the
    part into a point outside it again."""
    if len(row) < 2:
        return hit
    pair = row * (tri.max() + 1) + tri
    order = np.argsort(pair, kind="stable")
    seen = pair[order]
    again = np.ones(len(pair), dtype=bool)
    again[order[1:][seen[1:] == seen[:-1]]] = False
    return hit & again


def _count(hit, row, n):
    """How often every one of the `n` rows of a round is True in `hit`, as one array of
    counts of that length: the crossings a ray has had on its way up."""
    return np.bincount(row, minlength=n, weights=hit.astype(np.float64)).astype(np.int64)


def _crossings(p, tri):
    """Where the ray straight up from every point of `p` goes through the triangle that
    goes with it and stands above the point: the places to count of an even-odd test.

    A ray that is vertical is one point seen from above, so a triangle is crossed where its
    flat figure covers that point and the triangle is higher there than the point. A
    triangle that stands upright shows a line from above and crosses nothing - its
    determinant is nothing too, so a safe one is asked for and the answer it works out is
    dropped again."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flat = np.abs(det) > 1e-12
    dx, dy = p[:, 0] - tri[:, 0, 0], p[:, 1] - tri[:, 0, 1]
    safe = np.where(flat, det, 1.0)
    a = (dx * e2[:, 1] - dy * e2[:, 0]) / safe
    b = (e1[:, 0] * dy - e1[:, 1] * dx) / safe
    z = tri[:, 0, 2] + a * e1[:, 2] + b * e2[:, 2]
    return flat & (a >= 0.0) & (b >= 0.0) & (a + b <= 1.0) & (z > p[:, 2])


def _closest(p, tri):
    """The distance from every point of `p` to the triangle that goes with it.

    Where the foot of the perpendicular lands inside the triangle that is the distance;
    anywhere else the nearest place of a triangle lies on one of its edges, so the three of
    them are asked - of those points only, because a point measured against a whole layer
    of the part has its foot inside on all of them but a few. A triangle without a surface
    has no plane at all and is answered by its edges too, which for such a triangle are a
    line and two points."""
    a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    w = p - a
    n = np.cross(e1, e2)
    den = (n * n).sum(axis=1)                    # what the two edges span, squared
    flat = den > 1e-24
    wide = np.where(flat, den, 1.0)
    d1, d2 = (w * e1).sum(axis=1), (w * e2).sum(axis=1)
    f = (e1 * e2).sum(axis=1)
    s = (d1 * (e2 * e2).sum(axis=1) - d2 * f) / wide
    t = (d2 * (e1 * e1).sum(axis=1) - d1 * f) / wide
    d = np.sqrt(((w * n).sum(axis=1) ** 2) / wide)
    side = (s < 0.0) | (t < 0.0) | (s + t > 1.0) | ~flat
    if side.any():
        d[side] = np.minimum(_segment(p[side], a[side], tri[side, 1]),
                             np.minimum(_segment(p[side], a[side], tri[side, 2]),
                                        _segment(p[side], tri[side, 1], tri[side, 2])))
    return d


def _segment(p, a, b):
    """The distance from every point of `p` to the segment from `a` to `b`; a segment with
    no length is the point it starts at."""
    d = b - a
    dd = (d * d).sum(axis=1)
    t = np.clip(((p - a) * d).sum(axis=1) / np.where(dd > 0.0, dd, 1.0), 0.0, 1.0)
    return np.sqrt(((p - (a + t[:, None] * d)) ** 2).sum(axis=1))
