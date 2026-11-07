"""Screw threads (ISO metric) for 3D printing.

A thread is built as a "screw surface": a grid of rings in z (step at most
pitch/16) and angles (segments), where every point gets the radius r(theta, z)
that the ISO profile has at position

    u = (z - starts * pitch * theta / (2 pi)) mod pitch

along the helix. A flat cap at the top and bottom (a triangle fan to the centre)
closes the solid, so the result is always a closed manifold: suitable for
boolean operations and STL/3MF export.

ISO profile: flank angle 60 degrees, H = 0.866 * pitch, major diameter major_d,
minor diameter major_d - 1.0825 * pitch, flattened crest (H/8) and root (H/4).
``internal=True`` gives the solid that is cut out of a sleeve to make an internal
thread: the same shape with every radius enlarged to give the clearance.
"""
from __future__ import annotations

import math

import numpy as np
import manifold3d as mf

from .geom import affine, circle_points, manifold_from_mesh, normalize, rotation_matrix
from .i18n import tr

TAN30 = math.tan(math.pi / 6)

# coarse ISO pitch per common metric size (mm)
ISO_PITCH = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 16: 2.0, 20: 2.5}

DEFAULT_CLEARANCE = 0.15    # mm radial clearance: bolt and nut fit after printing


def iso_pitch(diameter):
    """Standard coarse pitch for a metric diameter (nearest size)."""
    sizes = sorted(ISO_PITCH)
    best = min(sizes, key=lambda d: abs(math.log(d / max(diameter, 1e-6))))
    return ISO_PITCH[best]


# ------------------------------------------------------------------ computing

def _profile_radius(u, pitch, r_major, r_minor):
    """Radius of the ISO profile at position u in [0, pitch) (0 = crest)."""
    H = math.sqrt(3.0) / 2.0 * pitch
    u_c = H / 8.0 * TAN30          # half width of the flattened crest
    q = np.minimum(u, pitch - u)   # distance to the nearest crest
    r = r_major - np.maximum(q - u_c, 0.0) / TAN30
    return np.maximum(r, r_minor)


def thread_solid(major_d, pitch, length, internal=False, clearance=0.0, starts=1,
                 left=False, segments=64, chamfer=True):
    """Thread solid along the Z axis from z=0 to z=length (manifold3d).

    ``internal=True`` is the cutting solid for an internal thread: the same shape with
    every radius + clearance. ``chamfer`` bevels the first and last half pitch at
    45 degrees, so the thread engages easily.
    """
    major_d, pitch, length = float(major_d), float(pitch), float(length)
    if major_d <= 0 or pitch <= 0 or length <= 0:
        raise ValueError(tr("major_d, pitch and length must be greater than 0"))
    starts = max(1, int(starts))
    segments = max(12, int(segments))
    r_major = major_d / 2.0 + clearance
    r_minor = (major_d - 1.0825 * pitch) / 2.0 + clearance
    if r_minor <= 0:
        raise ValueError(tr("pitch too coarse for this diameter"))
    sign = -1.0 if left else 1.0

    n = max(2, int(math.ceil(length / (pitch / 16.0))))
    zs = np.linspace(0.0, length, n + 1)
    th = np.arange(segments) * (2.0 * math.pi / segments)
    u = np.mod(zs[:, None] - sign * starts * pitch * th[None, :] / (2.0 * math.pi), pitch)
    r = _profile_radius(u, pitch, r_major, r_minor)

    if chamfer:
        half = pitch / 2.0
        cut = np.maximum(half - zs, 0.0)[:, None] + np.maximum(zs - (length - half), 0.0)[:, None]
        if internal:
            r = r + cut          # the cutting solid widens at the ends: a chamfered entry
        else:
            r = np.maximum(r - cut, max(0.02, 0.1 * r_minor))

    cos_t, sin_t = np.cos(th), np.sin(th)
    verts = np.stack([r * cos_t[None, :], r * sin_t[None, :],
                      np.broadcast_to(zs[:, None], r.shape)], axis=-1).reshape(-1, 3)
    c0, ct = len(verts), len(verts) + 1
    verts = np.vstack([verts, [[0.0, 0.0, 0.0], [0.0, 0.0, length]]])

    idx = np.arange((n + 1) * segments).reshape(n + 1, segments)
    a = idx[:-1].ravel()
    b = np.roll(idx[:-1], -1, axis=1).ravel()
    c = np.roll(idx[1:], -1, axis=1).ravel()
    d = idx[1:].ravel()
    tris = [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    ring0, ringN = idx[0], idx[-1]
    tris.append(np.stack([np.full(segments, c0), np.roll(ring0, -1), ring0], axis=1))
    tris.append(np.stack([np.full(segments, ct), ringN, np.roll(ringN, -1)], axis=1))
    man = manifold_from_mesh(verts, np.concatenate(tris))
    if man.status() == mf.Error.NoError and man.volume() < 0:
        man = manifold_from_mesh(verts, np.concatenate(tris)[:, ::-1])
    return man


def _place(man, center, axis):
    """Transform a solid lying along +Z from 0 to (center, axis)."""
    a = normalize(np.asarray(axis, dtype=float))
    center = np.asarray(center, dtype=float)
    z = np.array([0.0, 0.0, 1.0])
    w = np.cross(z, a)
    if np.linalg.norm(w) < 1e-12:
        M3 = np.eye(3) if float(z @ a) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        M3 = rotation_matrix(w, math.atan2(float(np.linalg.norm(w)), float(z @ a)))
    return man.transform(affine(M3, center))


def external_on(manifold, center, axis, radius, length, pitch=1.5, clearance=0.0,
                starts=1, left=False, segments=64):
    """Replace a cylinder in the solid by an external thread with major diameter 2*radius."""
    cyl = _place(mf.Manifold.cylinder(length, radius, radius, max(16, int(segments))),
                 center, axis)
    thread = _place(thread_solid(2.0 * radius, pitch, length, clearance=clearance,
                                 starts=starts, left=left, segments=segments), center, axis)
    return ((manifold - cyl) + thread) ^ (manifold + cyl)


def internal_in(manifold, center, axis, radius, length, clearance=DEFAULT_CLEARANCE,
                pitch=1.5, starts=1, left=False, segments=64):
    """Cut an internal thread into a hole with radius radius (major_d = 2*radius)."""
    thread = _place(thread_solid(2.0 * radius, pitch, length, internal=True,
                                 clearance=clearance, starts=starts, left=left,
                                 segments=segments), center, axis)
    return manifold - thread


def nut(major_d, pitch, height, outer_d=None, clearance=DEFAULT_CLEARANCE, hex=True,
        starts=1, left=False, segments=64):
    """Separate nut (hexagonal) or sleeve (round) with internal thread, from z=0 to z=height."""
    if outer_d is None:
        outer_d = major_d * 1.7
    if outer_d <= major_d:
        raise ValueError(tr("the outer size of a nut must be larger than the thread"))
    if hex:
        pts = circle_points(outer_d / 2.0 / math.cos(math.pi / 6.0), 6)
        body = mf.Manifold.extrude(mf.CrossSection([pts]), height)
    else:
        body = mf.Manifold.cylinder(height, outer_d / 2.0, outer_d / 2.0,
                                    max(24, int(segments)))
    hole = thread_solid(major_d, pitch, height, internal=True, clearance=clearance,
                        starts=starts, left=left, segments=segments)
    return body - hole


# ------------------------------------------------------------------ screw holes

# size: (clearance hole Ø, socket head Ø, head height, countersunk head Ø) in mm:
# clearance holes ISO 273 medium, socket heads ISO 4762, countersunk heads ISO 10642 (90°)
SCREW_HOLES = {
    "M2": (2.4, 3.8, 2.0, 4.4), "M2.5": (2.9, 4.5, 2.5, 5.5), "M3": (3.4, 5.5, 3.0, 6.7),
    "M4": (4.5, 7.0, 4.0, 9.0), "M5": (5.5, 8.5, 5.0, 11.2), "M6": (6.6, 10.0, 6.0, 13.4),
    "M8": (9.0, 13.0, 8.0, 17.9)}

SCREW_HOLE_KINDS = ("through", "counterbore", "countersink")

DEFAULT_HOLE_CLEARANCE = 0.2   # mm on every diameter of a hole: printed holes come out smaller
COUNTERBORE_ROOM = 0.2         # mm the counterbore goes deeper than the head, so it sits low


def screw_hole(manifold, center, axis, length, size, kind, clearance=DEFAULT_HOLE_CLEARANCE,
               segments=64):
    """A printable hole for a metric screw, cut along (center, axis) into the solid.

    ``kind`` is "through" (a straight clearance hole), "counterbore" (a cylinder of the
    socket head Ø + clearance, ``head height`` + 0.2 mm deep from the face, for a socket
    head screw) or "countersink" (a 90° cone from the countersunk head Ø + clearance at
    the face down to the hole, for a flat head screw). The hole goes through the whole
    ``length``, past the face and out at the back, so the cut leaves no skin behind.
    ``clearance`` (mm) is added to every diameter, because printed holes come out smaller.
    """
    if size not in SCREW_HOLES:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    if kind not in SCREW_HOLE_KINDS:
        raise ValueError(tr("unknown screw hole kind {kind}").format(kind=kind))
    length = float(length)
    if length <= 0:
        raise ValueError(tr("the depth of a screw hole must be greater than 0"))
    hole_d, head_d, head_h, cs_head_d = SCREW_HOLES[size]
    segments = max(12, int(segments))
    margin = max(0.5, 0.02 * length)      # mm past both faces, so no skin is left
    r_hole = (hole_d + clearance) / 2.0
    cutter = mf.Manifold.cylinder(length + 2.0 * margin, r_hole, r_hole, segments) \
        .translate((0.0, 0.0, -margin))
    if kind == "counterbore":
        depth = head_h + COUNTERBORE_ROOM + margin
        head = mf.Manifold.cylinder(depth, (head_d + clearance) / 2.0,
                                    (head_d + clearance) / 2.0, segments)
        cutter = cutter + head.translate((0.0, 0.0, -margin))
    elif kind == "countersink":
        # 90° included angle: the radius falls 1 mm per mm of depth, so the cone that ends
        # in the hole Ø is exactly as deep below the face as its radius is wide
        r_head = (cs_head_d + clearance) / 2.0
        head = mf.Manifold.cylinder(r_head - r_hole + margin, r_head + margin, r_hole,
                                    segments).translate((0.0, 0.0, -margin))
        cutter = cutter + head
    return manifold - _place(cutter, center, axis)


# ------------------------------------------------- selection -> thread context

def circle_loop(points, min_points=12):
    """(centre, radius) if the loop is a full circle, else (None, 0).

    A circle here is: at least ``min_points`` points at the same distance from
    their mean, covering the full angle.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < min_points:
        return None, 0.0
    c = P.mean(axis=0)
    d = np.linalg.norm(P - c, axis=1)
    r = float(d.mean())
    if r < 1e-6 or float(d.max() - d.min()) > 0.02 * r:
        return None, 0.0
    n = np.zeros(3)
    for a, b in zip(P - c, np.roll(P - c, -1, axis=0)):
        n += np.cross(a, b)
    if np.linalg.norm(n) < 1e-9:
        return None, 0.0
    n = normalize(n)
    u = normalize(P[0] - c)
    v = np.cross(n, u)
    ang = np.sort(np.arctan2((P - c) @ v, (P - c) @ u))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]]))
    if float(gaps.max()) > 2.5 * 2.0 * math.pi / len(P):
        return None, 0.0        # an arc, not a full circle
    return c, r


def _loop_area(P):
    """Area of a planar loop (Newell), independent of the orientation."""
    c = P.mean(axis=0)
    n = np.zeros(3)
    for a, b in zip(P - c, np.roll(P - c, -1, axis=0)):
        n += np.cross(a, b)
    return float(np.linalg.norm(n)) / 2.0


def face_thread_context(ents, fid, want="outer"):
    """(center, axis, radius, length) of a circular face or hole in a solid.

    ``want``: "outer" for the end face of a cylinder (external thread), "hole"
    for the opening of a hole (internal thread). The axis points into the solid.
    from_manifold does not guarantee that loops[0] is the outer loop, so the loops
    are compared by size.
    """
    if fid not in ents.F or not ents.V:
        return None
    plane = ents.face_plane(fid)
    loops = ents.face_points(fid)
    if not loops:
        return None
    biggest = int(np.argmax([_loop_area(np.asarray(L, dtype=float)) for L in loops]))
    P = np.array(list(ents.V.values()))
    dist = (P - plane.origin) @ plane.n
    axis = -plane.n if -float(dist.min()) >= float(dist.max()) else plane.n
    best = None
    for k, L in enumerate(loops):
        if want == "outer" and k != biggest:
            continue
        if want == "hole" and k == biggest:
            continue
        c, r = circle_loop(L)
        if c is None or (best is not None and best[2] >= r):
            continue
        along = (P - c) @ axis
        best = (c, axis, r, float(along.max() - along.min()))
    return best


def group_thread_context(ents, want="outer"):
    """Context from the largest circular face of a solid (e.g. a cylinder)."""
    best = None
    for fid in ents.visible_faces() if hasattr(ents, "visible_faces") else list(ents.F):
        tc = face_thread_context(ents, fid, want)
        if tc is not None and (best is None or tc[2] > best[2]):
            best = tc
    return best
