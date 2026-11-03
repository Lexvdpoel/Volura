"""The computational core for cutting a model in two and joining the parts.

A large model is cut in two parts with a plane (``cut``) so it can be printed in
parts. On the cut plane come joints that click the parts back together: pins,
holes for loose pins or a dovetail. The clearance (``clearance``) is always on the
hole side, so the parts fit after printing.

Everything happens in ``manifold3d.Manifold`` volumes; the positions are computed in
the local coordinate system (u, v) of the cut plane and then transformed back to the
world with ``Plane.matrix()``.
"""
from __future__ import annotations

import math

import numpy as np
import manifold3d as mf

from .geom import Plane, normalize, point_in_polys, rotation_matrix
from .i18n import tr

CHAMFER = 0.3          # mm chamfer at the tip of a pin
EXTRA_DEPTH = 0.5      # mm: holes are a little deeper than the pin goes in

DEFAULTS = {
    "pin_diameter": 5.0,
    "pin_length": 10.0,
    "count": "auto",
    "wall": 2.0,
    "dovetail_width": 8.0,
    "dovetail_depth": 5.0,
    "dovetail_angle": 15.0,
    "clearance": 0.2,
    "segments": 48,
}

# Joint kinds: identifiers used in code (the dialog shows its own labels), not translated.
KINDS = ("geen", "pennen", "losse pennen", "zwaluwstaart")


# --------------------------------------------------------------------- helpers

def _params(params):
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})
    return p


def union_all(manifolds):
    ms = [m for m in manifolds if m is not None and not m.is_empty()]
    if not ms:
        return mf.Manifold()
    if len(ms) == 1:
        return ms[0]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)


def _matrix34(M):
    return np.ascontiguousarray(np.asarray(M, dtype=float)[:3], dtype=np.float64)


def _world_matrix(plane, local34):
    """Combine a plane-local 3x4 transform with the plane's frame to world."""
    P = np.eye(4)
    P[:3] = plane.matrix()
    L = np.eye(4)
    L[:3] = np.asarray(local34, dtype=float)
    return _matrix34(P @ L)


def align_rotation(a, b):
    """The rotation that maps direction ``a`` onto direction ``b``."""
    a, b = normalize(a), normalize(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        return rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi)
    return rotation_matrix(axis / s, math.atan2(s, c))


def _rot_from_z(d):
    """A rotation that maps the +Z axis onto the unit vector d."""
    return align_rotation(np.array([0.0, 0.0, 1.0]), d)


def _cs(polys):
    return mf.CrossSection([np.asarray(p, dtype=float) for p in polys], mf.FillRule.EvenOdd)


# --------------------------------------------------------------------- cut / section

def cut(manifold, origin, normal):
    """Split with ``split_by_plane``; part_a lies on the side of the normal."""
    n = normalize(normal)
    off = float(np.dot(np.asarray(origin, dtype=float), n))
    a, b = manifold.split_by_plane(n, off)
    return a, b


def section(manifold, origin, normal):
    """The cross-section perpendicular to ``normal``: (2D polygons in (u, v), transform).

    The polygons lie in the local coordinate system of the cut plane; the second
    element is the ``Plane`` with which points (u, v) are mapped back to the world."""
    plane = Plane(origin, normal)
    R = np.column_stack([plane.u, plane.v, plane.n])
    Rinv = R.T
    tinv = -Rinv @ plane.origin
    local = manifold.transform(_matrix34(np.column_stack([Rinv, tinv])))
    cs = local.slice(0.0)
    polys = [np.asarray(p, dtype=float) for p in cs.to_polygons()]
    return polys, plane


# --------------------------------------------------------------------- pin positions

def _to_crosssection(section_arg):
    if isinstance(section_arg, mf.CrossSection):
        return section_arg
    if isinstance(section_arg, tuple) and len(section_arg) == 2 and isinstance(section_arg[0], (list, tuple)):
        return _cs(section_arg[0])
    if isinstance(section_arg, (list, tuple)):
        return _cs(section_arg)
    raise TypeError("section must be (polys, plane), a list of polygons or a CrossSection")


def _polygon_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    A = 0.5 * float(cr.sum())
    if abs(A) < 1e-12:
        return poly.mean(axis=0)
    gx = float(((x + x2) * cr).sum() / (6 * A))
    gy = float(((y + y2) * cr).sum() / (6 * A))
    return np.array([gx, gy])


def _principal(points):
    """(centroid, principal axis) of a point cloud."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    c = P.mean(axis=0)
    Q = P - c
    cov = Q.T @ Q / max(len(Q), 1)
    w, V = np.linalg.eigh(cov)
    return c, V[:, int(np.argmax(w))]


def _inside(region, p):
    polys = [np.asarray(q, dtype=float) for q in region.to_polygons()]
    return point_in_polys(polys, p)


def _nearest_vertex(region, p):
    best, bd = None, np.inf
    for q in region.to_polygons():
        q = np.asarray(q, dtype=float)
        d = np.linalg.norm(q - p, axis=1)
        k = int(np.argmin(d))
        if d[k] < bd:
            bd, best = float(d[k]), q[k]
    return best if best is not None else np.asarray(p, dtype=float)


def _auto_pins(region, radius):
    """1 pin in the middle of a small region, 2 pins apart for a long one."""
    outer = np.asarray(region.to_polygons()[0], dtype=float)
    if len(outer) < 3:
        return []
    c, d = _principal(outer)
    proj = (outer - c) @ d
    length = float(proj.max() - proj.min())
    if length > 6 * radius:
        imin, imax = int(np.argmin(proj)), int(np.argmax(proj))
        return [outer[imin].copy(), outer[imax].copy()]
    cen = _polygon_centroid(outer)
    if _inside(region, cen):
        return [cen]
    return [_nearest_vertex(region, cen)]


def _spread_pins(shrunk, n):
    """N pins spread as evenly as possible along the long axis of the whole cross-section."""
    if n <= 0:
        return []
    polys = [np.asarray(p, dtype=float) for p in shrunk.to_polygons()]
    allp = np.concatenate(polys)
    c, d = _principal(allp)
    m = np.array([-d[1], d[0]])
    proj = (allp - c) @ d
    lo, hi = float(proj.min()), float(proj.max())
    out = []
    for k in range(n):
        t = lo + (hi - lo) * ((k + 0.5) / n)
        p = c + d * t
        if shrunk.area() > 0 and not any(_inside(r, p) for r in shrunk.decompose()):
            p = _nearest_vertex(shrunk.decompose()[0], p) if shrunk.decompose() else p
        out.append(p)
    return out


def pin_positions(section, radius, wall, count="auto"):
    """Point positions (in (u, v)) where a pin fits with at least ``wall`` mm around it.

    Returns ``(points, warnings)``; without room there are no points and there is a
    warning."""
    cs = _to_crosssection(section)
    if cs.is_empty() or cs.area() <= 1e-9:
        return [], [tr("The cross-section is empty: no pin can be placed.")]
    margin = radius + wall
    shrunk = cs.offset(-margin)
    if shrunk.is_empty() or shrunk.area() <= 1e-9:
        return [], [tr("The cross-section is too narrow for pins of {d:g} mm "
                       "with {wall:g} mm wall thickness.").format(d=2 * radius, wall=wall)]
    if isinstance(count, str) and count.lower() == "auto":
        points = []
        for region in shrunk.decompose():
            points += _auto_pins(region, radius)
    else:
        points = _spread_pins(shrunk, int(count))
    if not points:
        return [], [tr("There is too little material for the chosen number of pins.")]
    return points, []


# --------------------------------------------------------------------- solids

def _pin_solid(plane, uv, radius, length, chamfer, segments):
    """A pin sticking out of the cut plane along -normal (attached to part_a)."""
    chamfer = max(0.0, min(chamfer, radius * 0.5, length * 0.5))
    body = mf.Manifold.cylinder(length - chamfer, radius, radius, segments)
    body = body.translate((0.0, 0.0, -(length - chamfer)))
    parts = [body]
    if chamfer > 1e-6:
        tip = mf.Manifold.cylinder(chamfer, radius - chamfer, radius, segments)
        tip = tip.translate((0.0, 0.0, -length))
        parts.append(tip)
    pin = union_all(parts)
    m = np.column_stack([np.eye(3), np.array([uv[0], uv[1], 0.0])])
    return pin.transform(_world_matrix(plane, m))


def _hole_solid(plane, uv, radius, depth, sign, segments):
    """A hole from the cut plane; sign=-1 along -normal, +1 along +normal."""
    cyl = mf.Manifold.cylinder(depth, radius, radius, segments)
    if sign < 0:
        cyl = cyl.translate((0.0, 0.0, -depth))
    m = np.column_stack([np.eye(3), np.array([uv[0], uv[1], 0.0])])
    return cyl.transform(_world_matrix(plane, m))


def _cyl_between(p, q, radius, segments):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    d = q - p
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return mf.Manifold()
    R = _rot_from_z(d / L)
    M = np.column_stack([R, p])
    return mf.Manifold.cylinder(L, radius, radius, segments).transform(_matrix34(M))


def _dovetail_solids(cs, plane, width, depth, angle_deg, clearance, segments):
    """(tail, slot): a trapezoidal dovetail across the whole cross-section."""
    regions = [r for r in cs.decompose() if r.area() > 1e-9]
    if not regions:
        return mf.Manifold(), mf.Manifold()
    region = max(regions, key=lambda r: r.area())
    outer = np.asarray(region.to_polygons()[0], dtype=float)
    c, d = _principal(outer)
    proj = (outer - c) @ d
    dmin, dmax = float(proj.min()), float(proj.max())
    tan = math.tan(math.radians(angle_deg))

    def trap(grow):
        hw, hd = width / 2 + grow, depth + grow
        return np.array([[-hw, 0.0], [hw, 0.0], [hw + hd * tan, -hd], [-hw - hd * tan, -hd]])

    mdir = np.array([-d[1], d[0]])
    R = np.column_stack([np.array([mdir[0], mdir[1], 0.0]),
                         np.array([0.0, 0.0, 1.0]),
                         np.array([d[0], d[1], 0.0])])
    M = _world_matrix(plane, np.column_stack([R, np.array([c[0], c[1], 0.0])]))
    length = max(dmax - dmin, 1e-3)

    tail = mf.Manifold.extrude(_cs([trap(0.0)]), length).translate((0.0, 0.0, dmin)).transform(M)
    slot_cs = _cs([trap(clearance)]).offset(clearance, mf.JoinType.Miter)
    slot = mf.Manifold.extrude(slot_cs, length + 2 * clearance)
    slot = slot.translate((0.0, 0.0, dmin - clearance)).transform(M)

    # keep within the cross-section
    clip_depth = depth + 2 * clearance + 1.0
    se = mf.Manifold.extrude(cs, clip_depth).translate((0.0, 0.0, -clip_depth))
    se = se.transform(plane.matrix())
    grown = cs.offset(clearance, mf.JoinType.Miter)
    seg = mf.Manifold.extrude(grown, clip_depth).translate((0.0, 0.0, -clip_depth))
    seg = seg.transform(plane.matrix())
    tail = tail ^ se
    slot = slot ^ seg
    return tail, slot


# --------------------------------------------------------------------- top level

def split_with_joint(manifold, origin, normal, kind, params=None):
    """Cut ``manifold`` and build the joint. -> (part_a, part_b, extra_parts).

    ``kind`` is one of "geen" (none), "pennen" (pins), "losse pennen" (loose pins),
    "zwaluwstaart" (dovetail). ``extra_parts`` holds the loose pins (with "losse pennen")."""
    if kind not in KINDS:
        raise ValueError(tr("unknown joint: {kind}").format(kind=kind))
    if manifold.is_empty() or manifold.status() != mf.Error.NoError:
        raise ValueError(tr("The model is not a valid closed volume."))
    p = _params(params)
    a, b = cut(manifold, origin, normal)
    extras = []
    if kind == "geen":
        return a, b, extras

    polys, plane = section(manifold, origin, normal)
    cs = _cs(polys)
    if cs.is_empty() or cs.area() <= 1e-9:
        return a, b, extras

    if kind == "pennen":
        r = p["pin_diameter"] / 2
        clr = p["clearance"]
        pts, _ = pin_positions((polys, plane), r, p["wall"], p["count"])
        pins = [_pin_solid(plane, uv, r, p["pin_length"], CHAMFER, p["segments"]) for uv in pts]
        holes = [_hole_solid(plane, uv, r + clr, p["pin_length"] + EXTRA_DEPTH, -1, p["segments"])
                 for uv in pts]
        a = union_all([a] + pins)
        b = b - union_all(holes)
    elif kind == "losse pennen":
        r = p["pin_diameter"] / 2
        clr = p["clearance"]
        depth = p["pin_length"] / 2 + EXTRA_DEPTH
        pts, _ = pin_positions((polys, plane), r, p["wall"], p["count"])
        holes_a = [_hole_solid(plane, uv, r + clr, depth, +1, p["segments"]) for uv in pts]
        holes_b = [_hole_solid(plane, uv, r + clr, depth, -1, p["segments"]) for uv in pts]
        a = a - union_all(holes_a)
        b = b - union_all(holes_b)
        extras = _loose_pins(manifold, pts, plane, r, p["pin_length"], p["segments"])
    elif kind == "zwaluwstaart":
        tail, slot = _dovetail_solids(cs, plane, p["dovetail_width"], p["dovetail_depth"],
                                      p["dovetail_angle"], p["clearance"], p["segments"])
        a = union_all([a, tail])
        b = b - slot
    return a, b, extras


def _loose_pins(manifold, pts, plane, radius, length, segments):
    """Loose pins (diameter d, length L) lying flat on the bed, next to the model."""
    if not pts:
        return []
    bb = np.asarray(manifold.bounding_box())
    lo, hi = bb[:3], bb[3:]
    y = 0.5 * (lo[1] + hi[1])
    x0 = hi[0] + 10.0 + radius
    pins = []
    for i, uv in enumerate(pts):
        x = x0 + i * (length + 5.0)
        pins.append(_cyl_between((x, y, radius), (x + length, y, radius), radius, segments))
    return pins
