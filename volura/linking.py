"""The core of the live link between a part in the model and its outline on the worksheet.

No Qt in this file: it compares shapes and turns plates into solids and back, and can
therefore be tested on its own. ``volura/app`` decides when it is called (switching
between the 3D model and the 2D worksheet).

A link joins one source element on the worksheet (the paths with one ``src`` number,
see `volura/vector2d.py`) with one group in the model (see `volura/model.py`). Only the
*shape* travels: a plate that becomes 1 mm thicker must thicken its outline, but moving
or rotating the plate in 3D - or moving, arranging, rotating or mirroring its outline on
the sheet - is a change of *placement*, and every side keeps its own. That is why a
shape is compared without any rigid motion, which is what `fit_rigid` measures: the best
rotation (and reflection, since mirroring is allowed on the sheet) plus translation that
carries one set of loops onto the other, and the residual of that fit.

Shape goes 3D to 2D for every linked face, because the outline of a face is always well
defined. The other way only works for a *plate*: a prism with two parallel faces at the
distance of the thickness and side walls perpendicular to them, which is exactly what
`volura/sheet3d.py` puts on the model. `plate_of` recognises such a solid and returns
its frame, its thickness and the outline of its bottom face; `plate_solid` is the
inverse. A solid that is not a plate keeps its shape in 3D and only sends an outline to
the sheet.

The loops that travel through a link are always in their own 2D frame (the plane of the
face, or the bottom face of the plate), never in world or sheet coordinates, so the
placement of either side is never part of the comparison.

A vent pattern that was cut into one of those outlines is remembered the same way, as a
`Vent`: the outline it was made for and the holes it gave, so the pattern can be made
again when the outline of the panel changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from .vector2d import _point_in_poly, polygon_area

JSON_DECIMALS = 6      # lossless to 1e-6 mm, the same convention as the worksheet
CLOSE_TOL = 1e-4       # mm: a closing point this close to the start point is a duplicate
CLEAN_TOL = 1e-6       # mm: points closer together than this add nothing to the shape
AXIS_TOL = 1e-9        # 1 - cos between two face normals: still the same direction
MAX_AXES = 8           # face directions that are tried as the axis of a plate
TIE_TOL = 1e-6         # relative: two thicknesses this close are the same plate
SLICE_EPS = 1e-3       # fraction of the thickness: how far inside a plate a check slice is
KINDS = ("plate", "face")


# --------------------------------------------------------------------------
#  comparing shapes without placement
# --------------------------------------------------------------------------

def _loop(pts, tol=CLOSE_TOL) -> np.ndarray:
    """The points of a closed loop, without a closing point that repeats the start."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(p) >= 2 and float(np.hypot(*(p[-1] - p[0]))) <= tol:
        p = p[:-1]
    return np.ascontiguousarray(p)


def _kabsch(S, D, allow_mirror=True):
    """The best rigid map of the points `S` onto `D`, without scaling: (R, t, rms).

    Umeyama with the scale fixed at 1: the rotation is the one that aligns the two
    covariance matrices, and when a mirror is allowed the reflection with the smaller
    residual wins (a mirrored outline on the sheet is still the same part).
    """
    ms, md = S.mean(axis=0), D.mean(axis=0)
    U, _s, Vt = np.linalg.svd((S - ms).T @ (D - md))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0.0 else -1.0
    best = None
    for sign in (d, -d) if allow_mirror else (d,):
        R = Vt.T @ np.diag([1.0, sign]) @ U.T
        t = md - R @ ms
        rms = float(np.sqrt(np.mean(np.sum((S @ R.T + t - D) ** 2, axis=1))))
        if best is None or rms < best[2]:
            best = (R, t, rms)
    return best


def _readings(loop):
    """Every way to read a closed loop: each start point, forwards and backwards.

    Two loops of the same shape rarely start at the same vertex, and a mirror turns
    the order of the points around, so both are tried.
    """
    n = len(loop)
    for q in (loop, loop[::-1]):
        for s in range(n):
            yield np.roll(q, -s, axis=0)


def _best_reading(src, dst, R, t):
    """The reading of `src` that lands closest on `dst` under the map (R, t).

    The distance of a shift is a cross-correlation of the two loops, so the best
    start point follows from one FFT instead of one fit per vertex.
    """
    A = src @ R.T + t
    best = None
    for q, base in ((A, src), (A[::-1], src[::-1])):
        c = np.fft.irfft(np.fft.rfft(q, axis=0) * np.conj(np.fft.rfft(dst, axis=0)),
                         n=len(q), axis=0).sum(axis=1)
        s = int(np.argmax(c))
        if best is None or float(c[s]) > best[0]:
            best = (float(c[s]), np.roll(base, -s, axis=0))
    return best[1]


def fit_rigid(src_loops, dst_loops, allow_mirror=True, tol=CLOSE_TOL):
    """The rigid map that carries `src_loops` onto `dst_loops`: (M, rms) or None.

    Both are lists of closed loops ((n, 2) arrays); the loops are matched by number and
    order, and per loop the number of points must be the same (a closing point that
    repeats the start point does not count, and `tol` says how close counts as the same
    point). `M` is the 3x3 homogeneous matrix of the fit and `rms` the root mean square
    distance of the points after it, in millimetres. None when the loops do not match;
    a shape that was really changed has a large `rms`, not None.

    The map is found from the loop with the fewest points: every reading of that loop
    gives a candidate map, the other loops take the reading that map suggests, and the
    fit over all points together decides which candidate was the right one.
    """
    src = [_loop(l, tol) for l in src_loops]
    dst = [_loop(l, tol) for l in dst_loops]
    if not src or len(src) != len(dst):
        return None
    if any(len(a) != len(b) for a, b in zip(src, dst)):
        return None
    pairs = [(a, b) for a, b in zip(src, dst) if len(a)]
    if not pairs:
        return None
    anchor = int(np.argmin([len(a) for a, _b in pairs]))
    D = np.concatenate([b for _a, b in pairs])
    best = None
    for cand in _readings(pairs[anchor][0]):
        R, t, _rms = _kabsch(cand, pairs[anchor][1], allow_mirror)
        order = [cand if i == anchor else _best_reading(a, b, R, t)
                 for i, (a, b) in enumerate(pairs)]
        R, t, rms = _kabsch(np.concatenate(order), D, allow_mirror)
        if best is None or rms < best[2]:
            best = (R, t, rms)
    R, t, rms = best
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = t
    return M, rms


def same_shape(a_loops, b_loops, tol=1e-3, allow_mirror=True) -> bool:
    """Whether two sets of loops are the same shape within `tol` mm, wherever they are."""
    fit = fit_rigid(a_loops, b_loops, allow_mirror=allow_mirror)
    return fit is not None and fit[1] <= tol


def shape_changed(old, new, tol=1e-3, allow_mirror=True) -> bool:
    """Whether `new` is another shape than the one a link last synced.

    A fit that is clearly off is a change. Nothing to fit is only a change when there is
    another number of loops, which is a hole that came or went; the same loops with
    another number of points are one shape that two triangulations found differently,
    and a link does not move for that.
    """
    old = [l for l in old if len(np.asarray(l, dtype=float).reshape(-1, 2)) >= 3]
    new = [l for l in new if len(np.asarray(l, dtype=float).reshape(-1, 2)) >= 3]
    if len(old) != len(new):
        return True
    fit = fit_rigid(old, new, allow_mirror=allow_mirror)
    return fit is not None and fit[1] > tol


def _edge_lengths(loop) -> np.ndarray:
    """The lengths of the segments of a loop, sorted: its shape without its placement."""
    d = np.roll(loop, -1, axis=0) - loop
    return np.sort(np.hypot(d[:, 0], d[:, 1]))


def find_loop(loop, loops, tol=1e-3) -> int:
    """The index in `loops` of the loop that is the same shape as `loop`, or -1.

    One `same_shape` is one fit, and the loops that are looked through are often all alike
    - the holes of a grille, the copies of one shape - so the number of points and the
    lengths of the segments decide first which loops can still be alike. Only those are
    fitted, which is what keeps a search over a whole worksheet a short one.
    """
    a = _loop(loop)
    d = _edge_lengths(a)
    for i, b in enumerate(loops):
        q = _loop(b)
        if len(q) != len(a) or float(np.max(np.abs(_edge_lengths(q) - d))) > tol:
            continue
        if same_shape([a], [q], tol=tol):
            return i
    return -1


def sorted_loops(loops) -> list:
    """The loops in the order a link compares them: the largest first.

    `fit_rigid` compares loop by loop, so both sides of a link have to hand them over in
    the same order. The area counts without its sign, because the worksheet does not say
    which way a loop runs, and the outline of a plate and the paths of one source element
    come in the order their maker liked.
    """
    out = [np.asarray(l, dtype=float).reshape(-1, 2) for l in loops if len(l) >= 3]
    return sorted(out, key=lambda l: -abs(polygon_area(l)))


def place(loops, M) -> list:
    """The loops through the homogeneous map `M`, as `fit_rigid` returns it."""
    out = []
    for l in loops:
        p = np.asarray(l, dtype=float).reshape(-1, 2)
        out.append(np.ascontiguousarray(p @ np.asarray(M, dtype=float)[:2, :2].T
                                        + np.asarray(M, dtype=float)[:2, 2]))
    return out


def invert(M) -> np.ndarray:
    """The map that undoes the homogeneous map `M` of the plane."""
    R, t = np.asarray(M, dtype=float)[:2, :2], np.asarray(M, dtype=float)[:2, 2]
    out = np.eye(3)
    out[:2, :2] = np.linalg.inv(R)
    out[:2, 2] = -out[:2, :2] @ t
    return out


def placement(old, current) -> np.ndarray:
    """The map that carries the shape `old` onto where that shape is now (`current`).

    This is the placement one side kept: the rigid fit of the shape the link last synced
    onto the loops as they lie there now, so a new shape handed to `place` with this map
    lands where the old one was. When the two cannot be matched at all - another number
    of loops, a hole that came or went - there is no placement to read and the identity
    is returned, which leaves the new shape where it is.
    """
    fit = fit_rigid(old, current)
    return np.eye(3) if fit is None else fit[0]


# --------------------------------------------------------------------------
#  a plane in the model
# --------------------------------------------------------------------------

def face_frame(normal, origin, ref=None):
    """An orthonormal frame ``(origin, u, v, n)`` in the plane through `origin`.

    Without `ref` the x-axis of the plane is world X projected into it (world Y when
    the normal points along X), so the frame depends only on the normal and two faces
    in the same plane get the same axes. With `ref` (a 3D direction) u is `ref`
    projected into the plane: that is how a plate keeps the direction it had on the
    sheet. `v` is always ``n x u``, so the frame is right handed.
    """
    n = np.asarray(normal, dtype=float).reshape(3)
    L = float(np.linalg.norm(n))
    n = n / L if L > 1e-12 else np.array([0.0, 0.0, 1.0])
    bases = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    if ref is not None:
        bases.insert(0, np.asarray(ref, dtype=float).reshape(3))
    u = bases[-1]
    for base in bases:
        u = base - n * float(base @ n)
        if float(np.linalg.norm(u)) > 1e-9:
            break
    u = u / float(np.linalg.norm(u))
    return np.asarray(origin, dtype=float).reshape(3).copy(), u, np.cross(n, u), n


def to_plane(points3d, frame) -> np.ndarray:
    """Points in the model as the flat coordinates of `frame`: an (n, 2) array."""
    o, u, v, _n = frame
    d = np.asarray(points3d, dtype=float).reshape(-1, 3) - np.asarray(o, dtype=float)
    return np.ascontiguousarray(d @ np.stack([u, v], axis=1))


def from_plane(points2d, frame) -> np.ndarray:
    """Flat coordinates of `frame` back in the model: an (n, 3) array."""
    o, u, v, _n = frame
    q = np.asarray(points2d, dtype=float).reshape(-1, 2)
    return np.asarray(o, dtype=float) + q[:, :1] * u + q[:, 1:2] * v


def frame_to_list(frame) -> list:
    """A frame as the plain numbers a link keeps: its origin and its normal."""
    o, _u, _v, n = frame
    return [float(c) for c in o] + [float(c) for c in n]


def frame_of(v):
    """The frame of `frame_to_list`; the axes follow from the normal, as in `face_frame`."""
    if v is None or len(v) < 6:
        return None
    q = np.asarray(v, dtype=float).reshape(-1)
    return face_frame(q[3:6], q[0:3])


# --------------------------------------------------------------------------
#  plates: the parts that are a prism
# --------------------------------------------------------------------------

def _matrix34(M):
    """A 3x4 affine matrix as manifold wants it: contiguous float64."""
    return np.ascontiguousarray(np.asarray(M, dtype=float)[:3], dtype=np.float64)


def _canonical(d):
    """The direction of a plane, independent of which side it is seen from.

    The same rule as `brep.plane_key`: the first component that says something fixes
    the sign, so the two sides of one face share a direction.
    """
    for c in d:
        if abs(c) > 1e-9:
            return -d if c < 0.0 else d
    return d


def _dist_to_line(p, a, b):
    """Distance from `p` to the line through `a` and `b`."""
    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L <= 1e-12:
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / L


def _clean(loop, tol=CLEAN_TOL) -> np.ndarray:
    """A slice contour without double points and without points on a straight line.

    `Manifold.slice` returns a corner once per adjacent triangle and puts extra points
    where the diagonals of the side walls are cut; those points say nothing about the
    shape and would break the point counts that `fit_rigid` compares.
    """
    p = [np.asarray(q, dtype=float) for q in np.asarray(loop, dtype=float).reshape(-1, 2)]
    changed = True
    while changed and len(p) > 3:
        changed = False
        keep = []
        for i, cur in enumerate(p):
            prev, nxt = p[i - 1], p[(i + 1) % len(p)]
            if float(np.hypot(*(cur - prev))) <= tol or _dist_to_line(cur, prev, nxt) <= tol:
                changed = True
                continue
            keep.append(cur)
        p = keep
    return np.ascontiguousarray(np.asarray(p, dtype=float).reshape(-1, 2))


def _orient(loops):
    """The loops of one filled shape: outer boundary counter-clockwise, holes clockwise.

    Largest loop first, so the order of the loops says something about the shape and
    not about the order in which manifold happened to return the contours.
    """
    loops = [l for l in loops if len(l) >= 3]
    out = []
    for i, loop in enumerate(loops):
        depth = sum(1 for j, other in enumerate(loops)
                    if j != i and _point_in_poly(other, loop[0]))
        if (depth % 2 == 0) != (polygon_area(loop) > 0.0):
            loop = loop[::-1].copy()
        out.append(loop)
    out.sort(key=lambda l: -abs(polygon_area(l)))
    return out


def _axes(manifold):
    """The directions of the faces of a solid, the direction with the most area first.

    Triangles are grouped by direction: a flat face is triangulated however manifold
    likes, but every triangle of it has the same normal, so the faces of a solid become
    one group each and the largest faces are the candidates for the axis of a plate.
    """
    mesh = manifold.to_mesh()
    V = np.asarray(mesh.vert_properties, dtype=float)
    T = np.asarray(mesh.tri_verts, dtype=np.int64)
    if not len(T):
        return []
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    a = 0.5 * np.linalg.norm(n, axis=1)
    keep = a > 1e-12
    n, a = n[keep], a[keep]
    if not len(n):
        return []
    d = n / (2.0 * a)[:, None]
    dirs = np.empty((len(a), 3))
    area = np.zeros(len(a))
    groups = 0
    for i in np.argsort(-a):
        q = _canonical(d[i])
        if groups:
            dots = dirs[:groups] @ q
            j = int(np.argmax(dots))
            if 1.0 - float(dots[j]) <= AXIS_TOL:
                area[j] += float(a[i])
                continue
        dirs[groups] = q
        area[groups] = float(a[i])
        groups += 1
    return [dirs[i] for i in np.argsort(-area[:groups])[:MAX_AXES]]


def _prism(manifold, n, tol):
    """(frame, thickness, loops) of the solid as a prism along `n`, or None.

    The solid is turned so that `n` points along +Z; then it is a plate when the
    section is the same just above the bottom, in the middle and just below the top,
    and when the volume is that section times the height. A cone, a sphere or a cube
    with a block on top fails; a prism passes whatever its axis is.
    """
    o = np.asarray(manifold.bounding_box(), dtype=float).reshape(2, 3).mean(axis=0)
    _o, u, v, _n = face_frame(n, o)
    R = np.stack([u, v, n])
    local = manifold.transform(_matrix34(np.column_stack([R, -R @ o])))
    lo, hi = np.asarray(local.bounding_box(), dtype=float).reshape(2, 3)
    t = float(hi[2] - lo[2])
    if t <= 0.0:
        return None
    eps = max(t * SLICE_EPS, 1e-9)
    areas = [local.slice(lo[2] + eps).area(), local.slice(lo[2] + 0.5 * t).area(),
             local.slice(hi[2] - eps).area()]
    if min(areas) <= 0.0 or max(areas) - min(areas) > tol * max(areas):
        return None
    volume = local.volume()
    if abs(volume - areas[1] * t) > tol * areas[1] * t:
        return None
    loops = _orient([_clean(q) for q in local.slice(lo[2] + 0.5 * t).to_polygons()])
    if not loops:
        return None
    return (o + lo[2] * n, u, v, n), t, loops


def plate_of(manifold, tol=1e-4):
    """What a solid is as a plate: {"frame", "thickness", "loops"}, or None.

    A plate (a prism) returns the frame ``(origin, u, v, n)`` of its bottom face, its
    thickness and the outline of that face in the frame: the outer boundary
    counter-clockwise and the holes clockwise. Anything else - a sphere, a cone, a
    part with a boss on top - is not a plate and gives None, because there is no
    outline that describes it completely: its shape only lives in 3D.

    A box is a prism in three directions, so being a prism is not enough: the plate is
    the prism with the smallest thickness, and of two equally thin ones the one with
    the largest faces - a plate of 40 x 20 x 3 is 3 mm thick, not 40. `tol` is the
    relative deviation the sections and the volume are allowed to differ by.
    """
    if manifold is None or manifold.is_empty():
        return None
    best = None
    for n in _axes(manifold):           # the largest faces first, so ties go by area
        plate = _prism(manifold, n, tol)
        if plate is None:
            continue
        if best is None or plate[1] < best[1] * (1.0 - TIE_TOL):
            best = plate
    if best is None:
        return None
    frame, thickness, loops = best
    return {"frame": frame, "thickness": thickness, "loops": loops}


def plate_solid(loops, frame, thickness) -> mf.Manifold:
    """The plate with outline `loops` in `frame`, `thickness` mm along its normal.

    The inverse of `plate_of`: the loops are filled as on the worksheet (even-odd, so
    a loop inside another one is a hole) and extruded along the normal of the frame,
    starting at its origin.
    """
    cs = mf.CrossSection([np.ascontiguousarray(np.asarray(l, dtype=float).reshape(-1, 2))
                          for l in loops if len(np.asarray(l, dtype=float).reshape(-1, 2)) >= 3],
                         mf.FillRule.EvenOdd)
    if cs.is_empty():
        return mf.Manifold()
    o, u, v, n = frame
    M = np.column_stack([u, v, n, np.asarray(o, dtype=float)])
    return mf.Manifold.extrude(cs, float(thickness)).transform(_matrix34(M))


def face_shape(manifold, frame):
    """The outline of a solid seen along the normal of `frame`, in the coordinates of that frame.

    This is how a link of a *face* reads the model again: the projection along the normal
    of the plane is that face for every prism, and for anything else it is the silhouette
    in that direction. A solid that is gone, or that has nothing to show from that side,
    gives None, and the link then has no news from the model.
    """
    if manifold is None or manifold.is_empty():
        return None
    o, u, v, n = frame
    R = np.stack([u, v, n])
    local = manifold.transform(_matrix34(np.column_stack([R, -R @ o])))
    cs = local.project()
    if cs.is_empty():
        return None
    loops = _orient([_clean(q) for q in cs.to_polygons()])
    return loops or None


def _loops_json(loops) -> list:
    """The loops as the plain numbers a file keeps, rounded to the decimals above."""
    return [[[round(float(x), JSON_DECIMALS), round(float(y), JSON_DECIMALS)]
             for x, y in np.asarray(l, dtype=float).reshape(-1, 2)] for l in loops]


def _loops_of(v) -> list:
    """Loops as they come back from a file: (n, 2) arrays."""
    return [np.asarray(l, dtype=float).reshape(-1, 2) for l in (v or [])]


# --------------------------------------------------------------------------
#  what a link remembers
# --------------------------------------------------------------------------

@dataclass
class Link:
    """One live link: an outline on the worksheet and the part it belongs to in 3D.

    `src` is the source number of the paths on the worksheet and `group` the id of the
    group in the model. `kind` is "plate" when the part is a prism, so the shape may
    also travel from the sheet back into the model, and "face" when it is not, so the
    outline is only a copy of what 3D says. `shape` is the shape as it was last synced:
    the loops in their own frame, without placement, and `thickness` the thickness of
    the plate then. Comparing that with the current shape is what shows a change, and
    it is blind to moving or mirroring either side.

    `frame` is what a link of a face adds: the plane its outline lives in, as
    `frame_to_list` keeps it, so the model can be read from the same side again. A plate
    needs no frame, because `plate_of` finds the plane of the plate itself.
    """

    src: int
    group: int
    kind: str = "plate"
    shape: list = field(default_factory=list)
    thickness: float | None = None
    frame: list | None = None

    def to_dict(self) -> dict:
        """Plain values only, so a link goes into a file as it is."""
        return {"src": int(self.src), "group": int(self.group), "kind": str(self.kind),
                "shape": _loops_json(self.shape),
                "thickness": None if self.thickness is None
                else round(float(self.thickness), JSON_DECIMALS),
                "frame": None if self.frame is None
                else [round(float(c), JSON_DECIMALS) for c in self.frame]}

    @staticmethod
    def from_dict(d) -> "Link":
        kind = str(d.get("kind", "plate"))
        thickness, frame = d.get("thickness"), d.get("frame")
        return Link(int(d.get("src", 0)), int(d.get("group", 0)),
                    kind if kind in KINDS else "plate",
                    _loops_of(d.get("shape")),
                    None if thickness is None else float(thickness),
                    None if not frame else [float(c) for c in frame])


# --------------------------------------------------------------------------
#  what a vent pattern remembers
# --------------------------------------------------------------------------

@dataclass
class Vent:
    """One vent pattern the worksheet remembers, so it can be made again.

    A grille is cut for the outline it lies in, so an outline that changes shape leaves
    its holes cut off or sticking out. `src` is the source number of that outline on the
    worksheet and `shape`, `size`, `web` and `margin` the options the pattern was made
    with, which are the defaults of `volura/boxgen.py` when a file says nothing.

    `outline` is the outer loop the pattern was made for and `holes` the holes it gave,
    both in the `sorted_loops` form and in their own plane, never in the placement of the
    sheet. That is what makes the pattern recognisable again: `shape_changed` compares the
    outline with the loop that is there now and `same_shape` finds the holes back,
    wherever the panel was moved to or turned into on the way.
    """

    src: int
    shape: str = "hex"
    size: float = 6.0
    web: float = 2.0
    margin: float = 3.0
    outline: list = field(default_factory=list)
    holes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain values only, so a vent goes into a file as it is."""
        return {"src": int(self.src), "shape": str(self.shape),
                "size": round(float(self.size), JSON_DECIMALS),
                "web": round(float(self.web), JSON_DECIMALS),
                "margin": round(float(self.margin), JSON_DECIMALS),
                "outline": _loops_json(self.outline), "holes": _loops_json(self.holes)}

    @staticmethod
    def from_dict(d) -> "Vent":
        return Vent(int(d.get("src", 0)), str(d.get("shape", "hex")),
                    float(d.get("size", 6.0)), float(d.get("web", 2.0)),
                    float(d.get("margin", 3.0)),
                    _loops_of(d.get("outline")), _loops_of(d.get("holes")))
