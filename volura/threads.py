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


def _rotation(axis):
    """The rotation that takes +Z to ``axis``, the one _place uses."""
    a = normalize(np.asarray(axis, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    w = np.cross(z, a)
    if np.linalg.norm(w) < 1e-12:
        return np.eye(3) if float(z @ a) > 0 else np.diag([1.0, -1.0, -1.0])
    return rotation_matrix(w, math.atan2(float(np.linalg.norm(w)), float(z @ a)))


def _place(man, center, axis):
    """Transform a solid lying along +Z from 0 to (center, axis)."""
    return man.transform(affine(_rotation(axis), np.asarray(center, dtype=float)))


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

# size: (width across flats, nut height) in mm, hex nuts ISO 4032
HEX_NUTS = {
    "M2": (4.0, 1.6), "M2.5": (5.0, 2.0), "M3": (5.5, 2.4), "M4": (7.0, 3.2),
    "M5": (8.0, 4.7), "M6": (10.0, 5.2), "M8": (13.0, 6.8)}

SCREW_HOLE_KINDS = ("through", "counterbore", "countersink", "nutpocket")

DEFAULT_HOLE_CLEARANCE = 0.2   # mm on every diameter of a hole: printed holes come out smaller
COUNTERBORE_ROOM = 0.2         # mm the counterbore goes deeper than the head, so it sits low


def _teardrop_loop(radius, up, segments):
    """The outline of a teardrop hole: a circle with a 45° roof towards ``up``.

    The roof is made of the two tangents from the point at ``radius`` * sqrt(2) in the
    direction of ``up``; they touch the circle at 45° left and right of it, so the top of
    the hole is a point. A loop in the xy-plane of a solid that runs along +Z.
    """
    a = math.atan2(float(up[1]), float(up[0]))
    # three quarters of the circle, from one tangent point round the bottom to the other
    ang = np.linspace(a + 0.25 * math.pi, a + 1.75 * math.pi, max(3, round(0.75 * segments)) + 1)
    pts = np.column_stack([np.cos(ang), np.sin(ang)]) * radius
    tip = np.array([math.cos(a), math.sin(a)]) * radius * math.sqrt(2.0)
    return np.vstack([pts, tip])


def screw_hole(manifold, center, axis, length, size, kind, clearance=DEFAULT_HOLE_CLEARANCE,
               segments=64, teardrop=False):
    """A printable hole for a metric screw, cut along (center, axis) into the solid.

    ``kind`` is "through" (a straight clearance hole), "counterbore" (a cylinder of the
    socket head Ø + clearance, ``head height`` + 0.2 mm deep from the face, for a socket
    head screw), "countersink" (a 90° cone from the countersunk head Ø + clearance at
    the face down to the hole, for a flat head screw) or "nutpocket" (a hexagonal pocket of
    the nut size + clearance, ``nut height`` + 0.2 mm deep from the face, that holds a hex
    nut while the screw goes through the rest). The hole goes through the whole
    ``length``, past the face and out at the back, so the cut leaves no skin behind.
    ``clearance`` (mm) is added to every diameter, because printed holes come out smaller.
    ``teardrop`` gives the straight hole a 45° roof that points up in the model, so a hole
    that runs sideways prints without support under its top; a hole that is more vertical
    than that needs no roof, and the room for the head always stays round.
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
    if teardrop and abs(float(normalize(np.asarray(axis, dtype=float))[2])) < 0.5:
        # the roof has to point up in the model, so the profile turns with the hole: this is
        # the direction in its own plane that _place later maps onto world +Z
        up = (_rotation(axis).T @ np.array([0.0, 0.0, 1.0]))[:2]
        hole = mf.Manifold.extrude(mf.CrossSection([_teardrop_loop(r_hole, up, segments)]),
                                   length + 2.0 * margin)
    else:
        hole = mf.Manifold.cylinder(length + 2.0 * margin, r_hole, r_hole, segments)
    cutter = hole.translate((0.0, 0.0, -margin))
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
    elif kind == "nutpocket":
        # the nut of the same size, held by its flats: a hexagon whose corners stand
        # (s + clearance) / sqrt(3) from the axis is that wide between the parallel sides
        s, nut_h = HEX_NUTS[size]
        pocket = mf.Manifold.extrude(
            mf.CrossSection([circle_points((s + clearance) / math.sqrt(3.0), 6)]),
            nut_h + COUNTERBORE_ROOM + margin).translate((0.0, 0.0, -margin))
        cutter = cutter + pocket
    return manifold - _place(cutter, center, axis)


# ------------------------------------------------------------- clearance test

CLEARANCE_TEST_THICKNESS = 4.0   # mm how thick the plate of a clearance test is
CLEARANCE_TEST_LABEL = 0.6       # mm how high the clearance stands above the plate
CLEARANCE_TEST_ROOM = 1.0        # mm between a hole and the letters that name it
CLEARANCE_TEST_CHAMFER = 0.4     # mm how the top edge of the pin is bevelled, so it enters easily


def _clearance_label(text, family, height, max_width):
    """The outlines of `text`, centred on the origin and at most `max_width` mm wide.

    Empty when there are no fonts or when the text yields nothing: a plate without labels is
    not an error, the row of holes works just as well.
    """
    from . import fonts           # Qt: only here, this core is Qt-free
    try:
        loops = fonts.text_loops(text, family, height) if fonts.has_fonts() else []
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    s = min(1.0, max_width / wide) if wide > 1e-9 else 1.0
    c = (allp.min(axis=0) + allp.max(axis=0)) / 2.0
    return [np.ascontiguousarray((np.asarray(L, dtype=float) - c) * s) for L in loops]


def clearance_test(pin_d=5.0, clearances=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                   thickness=CLEARANCE_TEST_THICKNESS, label_height=3.0, segments=64,
                   family=None):
    """A plate with a row of holes and a loose pin: what clearance does this printer like?

    Returns ``(plate, pin)`` as two manifolds in one frame, so the caller can move them
    together. The plate lies on z = 0 and is ``thickness`` mm high, with a hole of
    Ø pin_d + c for every clearance in ``clearances``: a row along X, ``pin_d + 6`` mm from
    centre to centre and 5 mm from the edge. Under every hole, on the top face, stands its
    clearance in letters of 0.6 mm, centred on the hole and inside the plate, which is that
    much deeper at the front. The pin is a cylinder of Ø pin_d and ``thickness`` + 4 mm high
    with a 0.4 mm chamfer on top, standing on the bed 5 mm behind the plate in +Y.

    With the clearances from small to large the holes grow from left to right, so the user
    prints the plate once, tries the pin in every hole and reads the clearance off the hole
    in which it fits best: that is the clearance to use in Screw Hole, threads and Cut and
    Join. Without fonts the plate simply has no labels, and the row still works.
    """
    pin_d = float(pin_d)
    if not 2.0 <= pin_d <= 20.0:
        raise ValueError(tr("the pin of a clearance test must be 2 to 20 mm wide"))
    clrs = [float(c) for c in clearances]
    if len(clrs) < 2:
        raise ValueError(tr("a clearance test needs at least two holes"))
    if any(not 0.0 <= c <= 2.0 for c in clrs):
        raise ValueError(tr("the clearance of a clearance test must be 0 to 2 mm"))
    thickness, label_height = float(thickness), float(label_height)
    segments = max(12, int(segments))
    spacing = pin_d + 6.0
    radii = [0.5 * (pin_d + c) for c in clrs]
    xs = [i * spacing for i in range(len(clrs))]
    # 5 mm of plate all around the holes, and as much more in front of them for the letters
    back, front = max(radii) + 5.0, max(radii) + 5.0 + CLEARANCE_TEST_ROOM + label_height
    lo, hi = -radii[0] - 5.0, xs[-1] + radii[-1] + 5.0
    plate = mf.Manifold.cube([hi - lo, back + front, thickness]).translate((lo, -front, 0.0))
    margin = 0.5                      # mm past both faces, so the hole leaves no skin behind
    for x, r in zip(xs, radii):
        hole = mf.Manifold.cylinder(thickness + 2.0 * margin, r, r, segments)
        plate = plate - hole.translate((x, 0.0, -margin))
    for x, c, r in zip(xs, clrs, radii):
        loops = _clearance_label(f"{round(c, 2):g}", family, label_height, spacing - 2.0)
        if loops:
            lab = mf.Manifold.extrude(mf.CrossSection(loops, mf.FillRule.NonZero),
                                      CLEARANCE_TEST_LABEL)
            plate = plate + lab.translate((x, -(r + CLEARANCE_TEST_ROOM
                                               + 0.5 * label_height), thickness))
    r, h = 0.5 * pin_d, thickness + 4.0
    body = mf.Manifold.cylinder(h - CLEARANCE_TEST_CHAMFER, r, r, segments)
    top = mf.Manifold.cylinder(CLEARANCE_TEST_CHAMFER, r, r - CLEARANCE_TEST_CHAMFER, segments)
    pin = (body + top.translate((0.0, 0.0, h - CLEARANCE_TEST_CHAMFER))
           ).translate((0.0, back + 5.0 + r, 0.0))
    return plate, pin


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
