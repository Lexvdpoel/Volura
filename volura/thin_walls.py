"""Which faces of a part are thinner than a printer can print.

A wall thinner than what the printer lays down is no wall at all: a slicer drops it or
prints it as a few blobs, so a fin of 0.3 mm with a nozzle of 0.4 mm does not exist once
it is off the bed. The thin wall view (View > Face Style > Thin Walls) colours those
faces, the way the overhang view colours what needs support.

The rule is the one a slicer uses for a wall that is too thin: a place is thin where a
circle of the diameter of the thinnest printable wall no longer fits in the material.
That is measured by cutting the part in planes and comparing every cut with its own
opening - the cut shrunk by half the wall and grown back is exactly what such a circle
reaches, and what falls off is what it cannot reach. One axis only finds the material
that is thin across it, so all three are asked: z the walls that stand up, x and y the
floors and roofs that lie down.

The work is done per cut, never per triangle, and only for the points that stand above
that cut, so a heavy view costs one cut per plane instead of one test per triangle.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: how many cut planes per axis at most; a wider part is cut with a wider step
MAX_PLANES = 400
#: mm a face centre may stand beside the thin area and still count: it lies on its edge
GROW = 0.05
#: a piece of the thin area under this fraction of r squared (r = half the wall) is the
#: corner a round circle cannot reach: a square corner leaves 0.21 r^2, which is no thin
#: wall, while a corner sharper than about 60 degrees leaves more and is a tip too fine to print
PIECE = 0.5
#: how many of the three axes must find a point thin before it is
VOTES = 2
#: mm of slack on the plane window, so rounding never drops the point that stands on it
SLACK = 1e-6
#: how many (point, edge) pairs one round of the point in polygon test holds in memory
CELLS = 2_000_000

# The three axes the part is cut along: the turn of the manifold over `angles` degrees
# that lifts that axis to its z, and the same turn for the points (`p @ turn` sends a
# point along), written out so a quarter turn stays exact.
AXES = ((None, None),                                             # z stays z
        ([0.0, -90.0, 0.0], np.array([[0.0, 0.0, 1.0],            # (x, y, z) -> (-z, y, x)
                                     [0.0, 1.0, 0.0],
                                     [-1.0, 0.0, 0.0]])),
        ([90.0, 0.0, 0.0], np.array([[1.0, 0.0, 0.0],             # (x, y, z) -> (x, -z, y)
                                     [0.0, 0.0, 1.0],
                                     [0.0, -1.0, 0.0]])))


def thin_mask(manifold, points, min_wall, step=None):
    """Which of `points` lie in material thinner than `min_wall`, as one bool per point.

    `points` are the centres of the faces of the view, ``(n, 3)`` in the same millimetres
    as the part. A point is thin when the material there is thinner than `min_wall`:
    nowhere around it a circle of that diameter fits in the part. The solid is cut in
    planes `step` millimetres apart (half the wall by default, at most `MAX_PLANES` per
    axis, so a wide part is cut wider) along each of the three axes; a point is judged on
    every cut within one step of it - a curved wall has no plane that stands exactly under
    all of its faces - and a point that lies beside the part is never thin.

    A point is thin when at least two of the three axes say so. A thin wall is thin in the
    two cuts that stand across it, while a cut that only grazes a round part - a plane
    beside the axis of a cylinder - makes a narrow strip of solid material; that is one
    axis on its own, and no wall.

    A `min_wall` of zero or less is no wall at all and raises ``ValueError``; an empty
    solid, no points, or a part that refuses to be cut simply say nothing is thin.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if min_wall <= 0:
        raise ValueError(tr("the thinnest wall must be greater than 0 mm"))
    thin = np.zeros(len(pts), dtype=bool)
    if not len(pts) or manifold is None or manifold.is_empty():
        return thin
    step = float(step) if step is not None and step > 0 else 0.5 * min_wall
    votes = np.zeros(len(pts), dtype=np.int8)
    for k, (angles, turn) in enumerate(AXES):
        # a point that already has its two votes, or can no longer get them, is left out
        skip = (votes >= VOTES) | (votes + (len(AXES) - k) < VOTES)
        if skip.all():
            break
        hit = skip.copy()
        solid = manifold if angles is None else manifold.rotate(angles)
        _one_axis(solid, pts if turn is None else pts @ turn, hit, min_wall, step)
        votes += hit & ~skip
    return votes >= VOTES


def _one_axis(solid, q, thin, min_wall, step):
    """The thin points of one axis, filled into `thin`; `solid` is cut along its own z.

    Every plane is cut once and only when a point of the view stands above it, and the
    points a previous axis already marked are left out of the work - which is what keeps
    the second and the third axis about as cheap as the first."""
    box = solid.bounding_box()
    lo, hi = float(box[2]), float(box[5])
    count = max(1, int(math.ceil((hi - lo) / step - 1e-9)))
    if count > MAX_PLANES:
        step = (hi - lo) / MAX_PLANES
        count = MAX_PLANES
    # The planes stand a step apart and in the middle of the part, so none of them lies in
    # a flat face: a cut exactly through a top or a bottom is a sliver of round-off that
    # would make every flat lid look thin. A step wider than the part is one middle cut.
    lo += 0.5 * ((hi - lo) - (count - 1) * step)
    z = q[:, 2]
    near = (z - lo) / step                              # the plane nearest to every point
    near = np.where(np.isfinite(near), near, -2.0)
    inside = (near >= -0.5 - SLACK) & (near <= count - 0.5 + SLACK)
    at = np.clip(np.rint(near), 0, count - 1).astype(np.int64)
    at = np.where(inside & ~thin, at, -1)
    rows = np.flatnonzero(at >= 0)
    if not len(rows):
        return
    # A point is judged first on the plane nearest to it, and only when that says nothing
    # on the planes a step beside it: the cut of a plane half a step away already leans
    # past the point where a wall is not straight, and on a curved wall no plane sits
    # exactly under all of its faces - the cut on the wide side is what sees a thin wall.
    cand = np.stack((at[rows], at[rows] - 1, at[rows] + 1), axis=1)
    turn = np.broadcast_to(np.minimum(np.arange(cand.shape[1]), 1), cand.shape)
    height = lo + cand * step
    ok = ((cand >= 0) & (cand < count)
          & (np.abs(z[rows, None] - height) <= step + SLACK)).ravel()
    point = np.repeat(rows, cand.shape[1])[ok]
    plane, turn = cand.ravel()[ok], turn.ravel()[ok]
    cut = {}                          # the thin edges of a plane, cut at most once per axis
    for round_no in (0, 1):
        todo = np.flatnonzero((turn == round_no) & ~thin[point])
        if not len(todo):
            continue
        by_plane = np.argsort(plane[todo], kind="stable")
        starts = np.flatnonzero(np.r_[True, plane[todo][by_plane][1:]
                                      != plane[todo][by_plane][:-1]])
        for that, nxt in zip(starts, np.r_[starts[1:], len(todo)]):
            here = point[todo[by_plane[that:nxt]]]
            index = plane[todo[by_plane[that]]]
            if index not in cut:
                cut[index] = _thin_edges(solid.slice(lo + index * step), min_wall)
            if cut[index] is not None:
                thin[here] |= _inside(q[here], cut[index])


def _thin_edges(section, min_wall):
    """The edges of the area of `section` no circle of Ø `min_wall` reaches, or ``None``.

    The opening - in by half the wall and back out again, round both times - is what such
    a circle reaches, so what the cut loses is what it cannot reach. A circle never
    reaches into a corner either: what it leaves in a square or a blunter corner is under
    half of r squared, so such pieces are dropped - otherwise every face along the edge of
    a part would be thin."""
    if section.is_empty():
        return None
    r = 0.5 * min_wall
    try:
        opened = section.offset(-r, mf.JoinType.Round).offset(r, mf.JoinType.Round)
        thin = section - opened
    except Exception:               # a cut manifold3d refuses says nothing about this plane
        return None
    if thin.is_empty():
        return None
    small = PIECE * r * r
    pieces = [p for p in thin.decompose() if p.area() >= small]
    if not pieces:
        return None
    area = pieces[0] if len(pieces) == 1 else mf.CrossSection.batch_boolean(
        pieces, mf.OpType.Add)
    return _edges(area.offset(GROW, mf.JoinType.Round).to_polygons())


def _edges(polys):
    """The first and the last point of every edge of the loops of a ``to_polygons()``."""
    loops = [np.asarray(q, dtype=float).reshape(-1, 2) for q in polys if len(q) > 2]
    if not loops:
        return None
    return (np.concatenate(loops), np.concatenate([np.roll(q, -1, axis=0) for q in loops]))


def _inside(q, walls):
    """Which of the plane points `q` lie in the area bound by `walls`, as one bool per point.

    The even-odd crossing test of a point in a polygon, done for all points of one cut at
    once: a ray to the right counts the edges it crosses, and an odd count means inside.
    The loops of a hole cross along with those outside, so they cut themselves out. The
    points go by rounds, because a cut of a busy model has enough edges for that."""
    a, b = walls
    x0, y0 = a[:, 0], a[:, 1]
    dx, dy = b[:, 0] - x0, b[:, 1] - y0
    slope = dx / np.where(dy == 0.0, 1.0, dy)      # dy is 0 only where nothing crosses
    x, y = q[:, 0], q[:, 1]
    out = np.zeros(len(q), dtype=bool)
    chunk = max(64, min(4096, CELLS // max(1, len(a))))
    for s in range(0, len(q), chunk):
        rows = slice(s, min(s + chunk, len(q)))
        crosses = (y0 > y[rows, None]) != (y0 + dy > y[rows, None])
        hit = crosses & (x[rows, None] < x0 + (y[rows, None] - y0) * slope)
        out[rows] = (hit.sum(axis=1) & 1) == 1
    return out
