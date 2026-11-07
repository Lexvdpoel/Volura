"""Modelling operations on a Document (no GUI code here)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import manifold3d as mf

from .brep import TOL, Entities, manifold_from_tris, mesh_payload, triangulate_job
from .geom import (Plane, default_u, extrude_cs, manifold_from_mesh, mesh_arrays, normalize,
                   rotation_matrix)
from .i18n import tr, tr_noop
from .model import ROOT, Document, Group
from .vector2d import VPath, polygon_area
from . import fillet, joints, nesting, workers

DEFAULT_COLOR = "#f2f2f2"

# The quality steps of Tools > Simplify Model: a fixed tolerance in mm and what it does.
# Aimed at a 0.4 mm nozzle and a 0.1 mm layer: a deviation far under the layer height is
# invisible in the print. The dialog and the tests read the same table; the dialog
# translates the texts when it shows them.
SIMPLIFY_STEPS = [
    (0.005, tr_noop("Lossless: no visible change, only removes what adds nothing.")),
    (0.01, tr_noop("Invisible in print: a tenth of a 0.1 mm layer.")),
    (0.02, tr_noop("Recommended: far below a layer; curved surfaces print the same.")),
    (0.05, tr_noop("Half a layer: large curves are fine; very fine details such as small text "
                   "may soften slightly.")),
    (0.1, tr_noop("One layer height: facets may show on large smooth curves; "
                  "fine details soften.")),
    (0.2, tr_noop("Half the nozzle: clearly coarser; only for drafts or hidden parts.")),
]
SIMPLIFY_DEFAULT_TOLERANCE = 0.02

# An import with more triangles than this gets the "simplify it now?" advice. Measured on
# the test machine (Qt offscreen, CPU only) by timing the per-frame work that scales with
# the triangle count - viewport._render_data() rebuilding the draw arrays of a sphere built
# from manifold (one face per triangle, as an imported STL becomes):
#     triangles   512  2048  4608   8192  18432  32768  73728  131072
#     frame (ms)  1.2   4.6  10.8   17.2   44.9   93.0  222.7   398.6
# about 0.003 ms per triangle, so a frame crosses 50 ms near 18000 triangles; 20000 is the
# first round step that is clearly past it. (Hovering and snapping stay ~1 ms at any size;
# the first hover after an import also rebuilds the snap cache, ~0.11 ms per triangle.)
LARGE_IMPORT_TRIANGLES = 20000

# An import whose largest size stays under this was almost certainly drawn in other units
# than millimetres (STL has no unit; converters often write inches): Volura asks whether
# to scale it up.
SMALL_IMPORT_MM = 5.0
# the units a too small model was probably drawn in, as (name, millimetres per unit)
IMPORT_UNITS = (("inches", 25.4), ("centimetres", 10.0), ("metres", 1000.0))


def union_all(manifolds):
    manifolds = [m for m in manifolds if not m.is_empty()]
    if not manifolds:
        return mf.Manifold()
    if len(manifolds) == 1:
        return manifolds[0]
    return mf.Manifold.batch_boolean(manifolds, mf.OpType.Add)


# ------------------------------------------------------------- groups

def make_group(doc: Document, items=None):
    """Move selected loose geometry (faces + edges) into a new group."""
    items = doc.selection if items is None else items
    ctx = doc.active
    ents = doc.active_ents()
    fs = [i for k, c, i in items if k == "f" and c == ctx]
    es = [i for k, c, i in items if k == "e" and c == ctx]
    if not fs and not es:
        return None
    sub = ents.extract(fs, es)
    ents.remove(fs, es)
    color = Counter(ents_face_colors(sub)).most_common(1)
    g = doc.add_group(sub, color[0][0] if color else DEFAULT_COLOR)
    if ctx != ROOT:
        # nested groups are not supported: the new group lives at top level
        pass
    return g


def ents_face_colors(ents):
    return [f.color for f in ents.F.values() if f.color]


def explode(doc: Document, g: Group):
    """Merge a group back into the loose geometry (faces take the group color)."""
    if g.locked:
        return
    sub = g.ents.copy()
    for f in list(sub.F):
        if sub.F[f].color is None:
            sub.set_face_color(f, g.color)
    doc.root.merge_from(sub)
    doc.remove_group(g.id)


def group_from_manifold(doc: Document, man, color, name=None):
    return doc.add_group(Entities.from_manifold(man), color, name)


def boolean(doc: Document, groups, op):
    """Union / subtract / intersect of solid groups. The first one keeps name and color."""
    if len(groups) < 2:
        return None
    ms = []
    for g in groups:
        m = g.ents.to_manifold()
        if m.status() != mf.Error.NoError or m.is_empty():
            raise ValueError(tr("{name} is not a closed volume.").format(name=g.name))
        ms.append(m)
    if op == "union":
        res = union_all(ms)
    elif op == "subtract":
        res = ms[0] - union_all(ms[1:])
    else:
        res = ms[0]
        for m in ms[1:]:
            res = res ^ m
    first = groups[0]
    for g in groups[1:]:
        doc.remove_group(g.id)
    if res.is_empty():
        doc.remove_group(first.id)
        return None
    first.ents = Entities.from_manifold(res)
    return first


def split_parts(doc: Document, g: Group):
    m = g.ents.to_manifold()
    parts = m.decompose() if m.status() == mf.Error.NoError else []
    if len(parts) < 2:
        return [g]
    g.ents = Entities.from_manifold(parts[0])
    out = [g]
    for p in parts[1:]:
        out.append(doc.add_group(Entities.from_manifold(p), g.color, g.name))
    return out


def tri_count(ents):
    """The number of triangles a group's geometry prints as (its manifold's triangles)."""
    m = ents.to_manifold()
    return 0 if m.is_empty() else len(m.to_mesh().tri_verts)


def has_painted_faces(g: Group):
    """True when a face of the group carries its own color (Tools > Simplify Model warns
    that simplifying drops those colors, because they do not survive the manifold)."""
    return any(f.color for f in g.ents.F.values())


def simplify_group(doc: Document, g: Group, tolerance):
    """Tools > Simplify Model: replace the geometry of `g` by a simplified copy.

    Returns (triangles before, triangles after). Name, color, visibility and id of the
    group stay the same; only its entities change. A locked group is left untouched.
    A group that is not a closed volume raises a ValueError with its name, as boolean().
    Face colors do not survive to_manifold()/from_manifold(), so the group keeps its own
    color (the dialog warns about painted faces beforehand)."""
    if g.locked:
        n = tri_count(g.ents)
        return n, n
    before, after, g.ents = simplified(g, tolerance)
    return before, after


def closed_manifold(g: Group):
    """The Manifold of `g`; a ValueError with its name when it is not a closed volume."""
    m = g.ents.to_manifold()
    if m.status() != mf.Error.NoError or m.is_empty():
        raise ValueError(tr("{name} is not a closed volume.").format(name=g.name))
    return m


def simplified(g: Group, tolerance, m=None):
    """(triangles before, triangles after, new Entities) of simplifying `g`.

    Reads the group and changes nothing, so it can run in a background task; `m` is the
    group's `closed_manifold` when the caller already has it. The counts are those of
    the manifold before and after `simplify`, so an estimate needs no new Entities."""
    m = closed_manifold(g) if m is None else m
    s = m.simplify(tolerance)
    return m.num_tri(), s.num_tri(), Entities.from_manifold(s)


def outside(m):
    """The outside of a solid: cavities and parts that lie inside another part go."""
    if m.status() != mf.Error.NoError:
        return m
    parts = [p for p in m.decompose() if p.volume() > 1e-9]
    if len(parts) > 1:
        keep = [p for i, p in enumerate(parts)
                if (p - union_all([q for j, q in enumerate(parts) if j != i])).volume() > 1e-9]
        parts = keep or parts
    return union_all(parts) if parts else m


def outer_shell(doc: Document, groups):
    """Tools > Outer Shell: one group of the outside of the selected groups.

    The groups are united and everything that is not on the outside is dropped:
    hollow cavities (decompose() returns those with a negative volume) and parts
    that lie completely inside another part. The first group keeps name and color."""
    if not groups:
        return None
    ms = []
    for g in groups:
        m = g.ents.to_manifold()
        if m.status() != mf.Error.NoError or m.is_empty():
            raise ValueError(tr("{name} is not a closed volume.").format(name=g.name))
        ms.append(m)
    res = outside(union_all(ms))
    first = groups[0]
    for g in groups[1:]:
        doc.remove_group(g.id)
    if res.is_empty():
        doc.remove_group(first.id)
        return None
    first.ents = Entities.from_manifold(res)
    return first


def trim(doc: Document, a: Group, b: Group):
    """Tools > Trim: b is trimmed with a (b becomes b - a), a stays as it is."""
    ms = []
    for g in (a, b):
        m = g.ents.to_manifold()
        if m.status() != mf.Error.NoError or m.is_empty():
            raise ValueError(tr("{name} is not a closed volume.").format(name=g.name))
        ms.append(m)
    res = ms[1] - ms[0]
    if res.is_empty():
        doc.remove_group(b.id)
        return None
    b.ents = Entities.from_manifold(res)
    return b


# ------------------------------------------------------------- intersect faces

def face_items(doc: Document, items):
    """(ctx, fid) of every face in items; a group contributes all its visible faces."""
    out = []
    for k, ctx, i in items:
        if k == "g":
            g = doc.group(i)
            if g is not None and g.visible:
                out += [(g.id, f) for f in g.ents.visible_faces()]
        elif k == "f":
            ents = doc.ents(ctx)
            if ents is not None and i in ents.F and i not in ents.hidden_f:
                out.append((ctx, i))
    return out


def _face_bbox(ents, fid):
    P = np.concatenate(ents.face_points(fid))
    return P.min(axis=0), P.max(axis=0)


def _boxes_overlap(a, b):
    return np.all(a[0] <= b[1] + TOL) and np.all(b[0] <= a[1] + TOL)


def _plane_line(pa, pb):
    """(point, direction) of the line where two planes cross; None when parallel."""
    w = np.cross(pa.n, pb.n)
    s = float(w @ w)
    if s < 1e-12:
        return None
    n1, n2 = pa.n, pb.n
    b = float(n1 @ n2)
    c1, c2 = float(n1 @ pa.origin), float(n2 @ pb.origin)
    return ((c1 - b * c2) * n1 + (c2 - b * c1) * n2) / s, w / math.sqrt(s)


def _polygon_intervals(loops, o, d, n_plane):
    """The stretches of the line o + t*d that a polygon covers (even odd)."""
    m = np.cross(n_plane, d)
    L = float(np.linalg.norm(m))
    if L < 1e-12:
        return []
    ts = []
    for loop in loops:
        P = np.asarray(loop, dtype=float)
        side = (P - o) @ m
        n = len(P)
        for i in range(n):
            s1, s2 = float(side[i]), float(side[(i + 1) % n])
            if (s1 > 0) == (s2 > 0):
                continue                          # the edge does not cross the line
            k = s1 / (s1 - s2)
            q = P[i] + (P[(i + 1) % n] - P[i]) * k
            ts.append(float((q - o) @ d))
    ts.sort()
    return [(ts[i], ts[i + 1]) for i in range(0, len(ts) - 1, 2)]


def face_intersection_segments(ents_a, fid_a, ents_b, fid_b):
    """The segments where two faces cut each other ([] when parallel or apart)."""
    pa, pb = ents_a.face_plane(fid_a), ents_b.face_plane(fid_b)
    line = _plane_line(pa, pb)
    if line is None:
        return []
    o, d = line
    ia = _polygon_intervals(ents_a.face_points(fid_a), o, d, pa.n)
    ib = _polygon_intervals(ents_b.face_points(fid_b), o, d, pb.n)
    out = []
    for a0, a1 in ia:
        for b0, b1 in ib:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 10 * TOL:
                out.append((o + d * lo, o + d * hi))
    return out


def intersect_faces(doc: Document, items_a, items_b):
    """Edit > Intersect Faces: add the cut lines between two sets of faces.

    Every face of `items_a` is cut with every face of `items_b` (groups contribute
    all their faces); the segments go into the active context, where they split the
    faces there, like SketchUp's "Intersect Faces > Model" and "> Selection".
    Returns the edges that were newly created by the cut lines."""
    ents = doc.active_ents()
    fa, fb = face_items(doc, items_a), face_items(doc, items_b)
    boxes = {k: _face_bbox(doc.ents(k[0]), k[1]) for k in set(fa) | set(fb)}
    done, along, new = set(), [], []
    for ka in fa:
        for kb in fb:
            if ka == kb or frozenset((ka, kb)) in done:
                continue
            done.add(frozenset((ka, kb)))
            if not _boxes_overlap(boxes[ka], boxes[kb]):
                continue
            ea, eb = doc.ents(ka[0]), doc.ents(kb[0])
            if ea is None or eb is None or ka[1] not in ea.F or kb[1] not in eb.F:
                continue
            for p, q in face_intersection_segments(ea, ka[1], eb, kb[1]):
                e, created = ents.add_segment(p, q)
                along += e
                new += created
    if along:
        ents.make_faces(along)
    return new


# ------------------------------------------------------------- transforms

def transform_selection(doc: Document, M3, t, copy=False, items=None):
    """Move/rotate/scale selected groups and loose geometry. Returns new selection."""
    items = doc.selection if items is None else items
    M3 = np.asarray(M3, dtype=float)
    t = np.asarray(t, dtype=float)
    new_sel = []
    for k, ctx, i in items:
        if k != "g":
            continue
        g = doc.group(i)
        if g is None:
            continue
        if g.locked:
            new_sel.append(("g", g.id, g.id))   # a locked group stays where it is
            continue
        if copy:
            ents = g.ents.copy()
            ents.transform(M3, t)
            n = doc.add_group(ents, g.color, g.name)
            new_sel.append(("g", n.id, n.id))
        else:
            g.ents.transform(M3, t)
            new_sel.append(("g", g.id, g.id))
    by_ctx = {}
    for k, ctx, i in items:
        if k in ("f", "e"):
            by_ctx.setdefault(ctx, ([], []))[0 if k == "f" else 1].append(i)
    for ctx, (fs, es) in by_ctx.items():
        ents = doc.ents(ctx)
        if ents is None:
            continue
        if copy:
            sub = ents.extract(fs, es)
            before_f, before_e = set(ents.F), set(ents.E)
            ents.merge_from(sub, M3, t)
            new_sel += [("f", ctx, f) for f in ents.F if f not in before_f]
            new_sel += [("e", ctx, e) for e in ents.E if e not in before_e]
        else:
            vids = set()
            for f in fs:
                for loop in ents.F[f].loops:
                    vids |= set(loop)
            for e in es:
                vids |= set(ents.E[e])
            ents.transform(M3, t, vids)
            new_sel += [("f", ctx, f) for f in fs if f in ents.F] + [("e", ctx, e) for e in es if e in ents.E]
    return new_sel


def selection_bbox(doc: Document, items=None):
    items = doc.selection if items is None else items
    pts = []
    for k, ctx, i in items:
        if k == "g":
            g = doc.group(i)
            if g is not None and g.bbox() is not None:
                pts.extend(g.bbox())
            continue
        if k != "e" and k != "f":
            continue                      # guides are not part of the volume
        ents = doc.ents(ctx)
        if ents is None:
            continue
        if k == "f":
            for loop in ents.F[i].loops:
                pts.extend(ents.V[v] for v in loop)
        else:
            pts.extend(ents.edge_points(i))
    if not pts:
        return None
    pts = np.asarray(pts)
    return np.array([pts.min(axis=0), pts.max(axis=0)])


def whole_items(doc: Document, groups=None):
    """Selection items for 'everything': all groups plus all loose geometry."""
    items = [("g", g.id, g.id) for g in (groups if groups is not None else doc.visible_groups())]
    if groups is None:
        items += [("f", ROOT, f) for f in doc.root.F] + [("e", ROOT, e) for e in doc.root.E]
    return items


def drop_to_bed(doc: Document, items):
    bb = selection_bbox(doc, items)
    if bb is not None:
        transform_selection(doc, np.eye(3), np.array([0, 0, -bb[0][2]]), items=items)


def center_on_bed(doc: Document, items, bed):
    bb = selection_bbox(doc, items)
    if bb is not None:
        c = bb.mean(axis=0)
        transform_selection(doc, np.eye(3), np.array([bed[0] / 2 - c[0], bed[1] / 2 - c[1], -bb[0][2]]),
                            items=items)


def lay_flat(doc: Document, gid, fid):
    """Lay Flat on This Face: turn a group so that one of its faces lies on the bed.

    The group turns about the centre of its own bounding box until the normal of face
    `fid` points straight down, the shortest way round (a face that already points down
    leaves the group where it is, one that points up turns it over), and afterwards the
    group is dropped on the bed, like "place on face" of a slicer. A locked group is not
    touched; the caller takes the checkpoint, so one undo brings the old place back."""
    g = doc.group(gid)
    if g is None or g.locked or fid not in g.ents.F:
        return
    bb = g.bbox()
    if bb is None:
        return
    c = 0.5 * (np.asarray(bb[0], dtype=float) + np.asarray(bb[1], dtype=float))
    n = np.asarray(g.ents.F[fid].normal, dtype=float)
    R = joints.align_rotation(n, np.array([0.0, 0.0, -1.0]))
    item = [("g", g.id, g.id)]
    transform_selection(doc, R, c - R @ c, items=item)
    drop_to_bed(doc, item)


# What Arrange on Bed leaves free: the same 5 mm a slicer usually keeps from the edge
# and between the parts.
BED_MARGIN = 5.0
BED_GAP = 5.0
FLAT_AREA = 1e-6          # mm²: below this a footprint has no area and is treated as flat


def _turn_2d(o, a, b) -> float:
    """Which way the way goes on from o via a to b: left (positive), straight, right (negative)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull_2d(pts) -> np.ndarray:
    """The convex hull of 2D points as a counter-clockwise loop (Andrew's monotone chain)."""
    P = np.unique(np.asarray(pts, dtype=float).reshape(-1, 2), axis=0)
    if len(P) < 3:
        return P

    def chain(points):
        """Half of the hull: every point that would make a right turn is dropped again."""
        out = []
        for p in points:
            while len(out) >= 2 and _turn_2d(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    return np.array(chain(P) + chain(P[::-1])[:-1], dtype=float)


def footprint_part(ents) -> nesting.Part | None:
    """The footprint of a piece of geometry as a nesting part: the hull of its vertices on XY."""
    if not ents.V:
        return None
    P = np.array(list(ents.V.values()), dtype=float)[:, :2]
    hull = hull_2d(P)
    if len(hull) < 3 or abs(polygon_area(hull)) < FLAT_AREA:
        # a group that stands flat on the bed has no area: nest on its bounding box instead
        lo, hi = P.min(axis=0), P.max(axis=0)
        hull = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=float)
    if polygon_area(hull) < 0:
        hull = hull[::-1].copy()
    lo, hi = hull.min(axis=0), hull.max(axis=0)
    return nesting.Part(paths=[VPath(np.ascontiguousarray(np.vstack([hull, hull[:1]])), True)],
                        outline=hull, bbox=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                        area=abs(polygon_area(hull)))


def arrange_placements(doc: Document, items, bed, gap=BED_GAP, margin=BED_MARGIN):
    """The move per group that lays the parts side by side on the bed: `(gid, M3, t)`.

    Every group is one part: the convex hull of its vertices projected on XY. The parts
    are placed with the same nesting as the worksheet (largest first, in the lowest and
    then leftmost free spot) on a bed of `bed[0]` × `bed[1]` mm, with `margin` mm along
    the edge and `gap` mm between the parts, rotated 0, 90, 180 or 270 degrees about
    their own footprint. Locked groups and groups without geometry are left out, and the
    height is not touched: `arrange_on_bed` drops the parts on the bed afterwards.

    Nothing is moved here; a `ValueError` is raised when the parts do not fit on one bed.
    """
    parts, owners, seen = [], {}, set()
    for k, _ctx, i in items:
        g = doc.group(i) if k == "g" else None
        if g is None or g.locked or g.id in seen:
            continue
        part = footprint_part(g.ents)
        if part is None:
            continue
        seen.add(g.id)
        parts.append(part)
        owners[id(part)] = g
    if not parts:
        return []
    try:
        placements = nesting.nest(parts, float(bed[0]), float(bed[1]), margin, gap,
                                  rotations=(0, 90, 180, 270), max_sheets=1)
    except ValueError:
        raise ValueError(tr("The parts do not fit on one bed of {w:.0f} × {h:.0f} mm.").format(
            w=float(bed[0]), h=float(bed[1])))
    moves = []
    for part, _sheet, angle, dx, dy in placements:
        M = nesting.place_transform(part, angle, dx, dy)
        M3 = np.eye(3)
        M3[:2, :2] = M[:2, :2]           # turn about the vertical axis, keep the height
        moves.append((owners[id(part)].id, M3, np.array([M[0, 2], M[1, 2], 0.0])))
    return moves


def arrange_on_bed(doc: Document, items, bed, gap=BED_GAP, margin=BED_MARGIN):
    """Arrange on Bed: lay the groups side by side on the print bed, like a slicer.

    See `arrange_placements` for the search for a place; here every group is turned to
    its place and dropped on the bed, so all parts stand on the bed without overlapping.
    The caller takes the checkpoint, so one undo brings the old places back. Returns how
    many groups did not get a place; when the bed is too small nothing moves and a
    `ValueError` tells so.
    """
    wanted = {i for k, _ctx, i in items if k == "g" and doc.group(i) is not None
              and not doc.group(i).locked}
    moves = arrange_placements(doc, items, bed, gap=gap, margin=margin)
    for gid, M3, t in moves:
        sel = [("g", gid, gid)]
        transform_selection(doc, M3, t, items=sel)
        drop_to_bed(doc, sel)
    return max(0, len(wanted) - len(moves))


# The default chamfer of Tools > Print Tools > Elephant Foot Chamfer. The first layer of a
# print is squashed a little wider than the rest of the part, so it sticks out: an
# "elephant's foot". A chamfer of a few tenths of a millimetre along the bottom edges makes
# room for that squashed layer.
ELEPHANT_FOOT_SIZE = 0.4
BOTTOM_TOL = 0.01           # mm: both ends of a bottom edge lie within this of the lowest z
DOWN_NORMAL = -0.999        # the normal of a face that points straight down


def bottom_edges(ents: Entities):
    """The ids of the edges along the bottom of a part: both end points at the lowest z of
    the part and the edge belonging to a face that points straight down. A part that only
    touches the bed with a vertex or with a tilted edge has no bottom edges."""
    bb = ents.bbox()
    if bb is None:
        return []
    z = float(bb[0][2])
    eids = set()
    for f in ents.F.values():
        if float(f.normal[2]) > DOWN_NORMAL:
            continue
        for eid in ents.face_edges(f):
            p0, p1 = ents.edge_points(eid)
            if abs(float(p0[2]) - z) <= BOTTOM_TOL and abs(float(p1[2]) - z) <= BOTTOM_TOL:
                eids.add(eid)
    return sorted(eids)


def elephant_foot_chamfer(g: Group, size, segments=64):
    """The geometry of `g` with its bottom edges chamfered by `size` mm, or None when there
    is nothing to chamfer.

    Reads the group and changes nothing, so it can run in a background task, as `simplified`
    does. None for a locked group, one that is not a closed volume, one without bottom edges
    (see `bottom_edges`) and one that is too small for the size: a `ValueError` of the
    chamfer leaves the group as it is."""
    if g.locked:
        return None
    eids = bottom_edges(g.ents)
    if not eids:
        return None
    ok, _bad = fillet.edge_data(g.ents, eids)
    if not ok:
        return None
    try:
        man = g.ents.to_manifold()
    except Exception:
        return None
    if man.is_empty() or man.status() != mf.Error.NoError:
        return None
    edges = [(p0, p1, n1, n2) for (_eid, p0, p1, n1, n2) in ok]
    try:
        res = fillet.apply(man, edges, float(size), "chamfer", int(segments))
    except ValueError:                      # the size is too large for this volume
        return None
    if res.is_empty() or res.status() != mf.Error.NoError:
        return None
    return Entities.from_manifold(res)


def elephant_foot(doc: Document, items, size, segments=64):
    """Elephant Foot Chamfer: a small chamfer along the bottom edges of the given groups.

    The first printed layer is squashed wider than the rest of the part; the chamfer takes
    the same amount off the bottom, so the foot of the print does not stick out. Every group
    keeps its name and color, only its geometry changes; the caller takes the checkpoint, so
    one undo brings the old shape back. Returns (groups chamfered, groups skipped); see
    `elephant_foot_chamfer` for the groups that are skipped."""
    changed = skipped = 0
    seen = set()
    for k, _ctx, i in items:
        g = doc.group(i) if k == "g" else None
        if g is None or g.id in seen:
            continue
        seen.add(g.id)
        new = elephant_foot_chamfer(g, size, segments)
        if new is None:
            skipped += 1
        else:
            g.ents = new
            changed += 1
    return changed, skipped


# The defaults of Tools > Print Tools > Mouse Ears: thin round discs on the bed under the
# sharp bottom corners of a part, so those corners do not lift (warp) off the bed while
# printing - a classic trick for ABS and ASA and for large flat parts. The discs are cut off
# after printing. A hull corner is "sharp" when its interior angle is at most this many
# degrees: a box has four corners of 90° and gets an ear on each, a cylinder's hull corners
# are nearly flat and get none.
MOUSE_EAR_DIAMETER = 10.0
MOUSE_EAR_HEIGHT = 0.4
MOUSE_EAR_MAX_ANGLE = 100.0


def _sharp_hull_corners(hull, max_angle):
    """The corners of a convex hull whose interior angle is at most `max_angle` degrees.
    `hull_2d` repeats the point where its two chains meet, so consecutive duplicates go
    first; a corner between two identical points has no angle and would otherwise be lost."""
    limit = math.radians(max_angle)
    pts = [p for i, p in enumerate(hull) if not np.array_equal(p, hull[i - 1])]
    sharp = []
    for i, cur in enumerate(pts):
        a, b = pts[i - 1] - cur, pts[(i + 1) % len(pts)] - cur
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            continue
        if math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))) <= limit:
            sharp.append(cur)
    return sharp


def mouse_ears_part(g: Group, diameter, height, max_angle=MOUSE_EAR_MAX_ANGLE, segments=48):
    """The geometry of `g` with a thin disc under every sharp bottom corner, or None when
    there is nothing to add.

    Reads the group and changes nothing, so it can run in a background task, as
    `elephant_foot_chamfer` does. The vertices at the lowest z of the group (within
    `BOTTOM_TOL`) are put on their 2D convex hull; every hull corner sharper than `max_angle`
    degrees gets a cylinder of `diameter` and `height` standing on that z, and all of them are
    unioned with the part. None for a locked group, one that is not a closed volume, one that
    does not stand flat on its bottom (fewer than three bottom vertices), one without sharp
    corners and one whose union does not come out right."""
    if g.locked:
        return None
    bb = g.ents.bbox()
    if bb is None or not g.ents.V:
        return None
    z = float(bb[0][2])
    bottom = np.array([p for p in g.ents.V.values() if abs(float(p[2]) - z) <= BOTTOM_TOL],
                      dtype=float)
    if len(bottom) < 3:
        return None
    hull = hull_2d(bottom[:, :2])
    if len(hull) < 3:
        return None
    corners = _sharp_hull_corners(hull, max_angle)
    if not corners:
        return None
    try:
        man = g.ents.to_manifold()
    except Exception:
        return None
    if man.is_empty() or man.status() != mf.Error.NoError:
        return None
    r = float(diameter) / 2.0
    ears = [mf.Manifold.cylinder(float(height), r, r, int(segments)).translate((c[0], c[1], z))
            for c in corners]
    res = union_all([man] + ears)
    if res.is_empty() or res.status() != mf.Error.NoError:
        return None
    return Entities.from_manifold(res)


def mouse_ears(doc: Document, items, diameter, height, segments=48):
    """Mouse Ears: a thin disc on the bed under the sharp bottom corners of the given groups.

    A sharp corner of a flat bottom lifts off the bed while printing (warping), most of all
    with ABS and ASA and on large flat parts; a thin disc under each corner holds it down and
    is cut off after printing. Every group keeps its name and color, only its geometry
    changes; the caller takes the checkpoint, so one undo brings the old shape back. Returns
    (groups with ears, groups skipped); see `mouse_ears_part` for the groups that are skipped."""
    changed = skipped = 0
    seen = set()
    for k, _ctx, i in items:
        g = doc.group(i) if k == "g" else None
        if g is None or g.id in seen:
            continue
        seen.add(g.id)
        new = mouse_ears_part(g, diameter, height, segments=segments)
        if new is None:
            skipped += 1
        else:
            g.ents = new
            changed += 1
    return changed, skipped


# Orient for Printing (Tools > Print Tools) turns a part so that it stands on the face that
# needs the least support, like the auto-orient of a slicer. The faces a part can stand on
# are the faces of its convex hull: only those are flat and large enough to print on, and on
# a scanned model there are far fewer of them than of triangles. The hull faces whose normals
# lie within a degree of each other count as one (the side of a cylinder of 64 segments is 64
# of them), and of the rest only the largest `max_candidates` are tried: a part never stands
# on a face of a square micrometre, and a hull of a big scan has thousands.
ORIENT_MERGE_DEG = 1.0        # hull faces whose normals lie within this are one candidate
ORIENT_SAME_DEG = 1.0         # a direction this close to straight down needs no turning
ORIENT_TIE = 0.01             # scores within 1% of the best score are a tie
ORIENT_EPS = 1e-9             # mm²: what no program can measure, so what counts as nothing
DOWN_DIR = np.array([0.0, 0.0, -1.0])


def is_down(d, tol_deg=ORIENT_SAME_DEG) -> bool:
    """Whether the direction `d` already points straight down, within `tol_deg` degrees."""
    cos = float(np.dot(normalize(d), DOWN_DIR))
    return math.degrees(math.acos(min(1.0, max(-1.0, cos)))) <= tol_deg


def overhang_area(V, T, d, limit_deg) -> float:
    """The area of the triangles that need support when the part stands with `d` down.

    The triangles of `V` (n, 3) and `T` (m, 3) are turned until `d` points straight down and
    marked with `overhang_mask`, with the lowest turned vertex as the print bed; the areas of
    the marked ones are added. Turning does not change an area, so the answer is the same in
    every frame of reference."""
    tri = np.asarray(V, dtype=float)[np.asarray(T, dtype=np.int64).reshape(-1, 3)]
    tri = tri @ joints.align_rotation(d, DOWN_DIR).T
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mark = overhang_mask(n, tri.mean(axis=1)[:, 2], float(tri[:, :, 2].min()), limit_deg)
    return float(0.5 * np.linalg.norm(n[mark], axis=1).sum())


def hull_directions(man, merge_deg=ORIENT_MERGE_DEG, max_candidates=40):
    """The directions a part can stand on, as `(direction, area)` of the convex hull faces.

    The outward normals of the faces of the hull of the manifold `man`, largest first. Faces
    whose normals lie within `merge_deg` of each other are one direction and add their areas,
    and the `max_candidates` largest remain. The direction that is already straight down is
    always there, also when it is not a face of the hull or did not survive the cut, so that
    a part that is already right is not turned for nothing."""
    mesh = man.hull().to_mesh()
    V = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    T = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    length = np.linalg.norm(n, axis=1)
    limit = math.cos(math.radians(merge_deg))
    dirs, areas = [], []
    for i in np.argsort(-length):
        if length[i] <= ORIENT_EPS:
            continue
        u = n[i] / length[i]
        for j, w in enumerate(dirs):
            if float(np.dot(u, w)) >= limit:
                areas[j] += 0.5 * float(length[i])
                break
        else:
            dirs.append(u)
            areas.append(0.5 * float(length[i]))
    out = [(dirs[i], areas[i]) for i in np.argsort(-np.asarray(areas))[:int(max_candidates)]]
    if not any(float(np.dot(d, DOWN_DIR)) >= limit for d, _area in out):
        out.append((DOWN_DIR.copy(), 0.0))
    return out


def best_down_direction(V, T, limit_deg=45.0, max_candidates=40):
    """The way a part has to point down to need the least support, as `(direction, area)`.

    `V` (n, 3) and `T` (m, 3) are the triangles of the part, for instance of
    `mesh_arrays(g.ents.to_manifold())`. The candidates to stand on are the faces of the
    convex hull, see `hull_directions`; for each of them the part is turned until that face
    points straight down and `overhang_area` says how much has to be supported. The least of
    that wins. A tie of 1% goes to the candidate with the largest hull face, because that
    stands the most steadily on the bed, and a tie that is left goes to the way the part
    already stands, so a part that is fine is not turned. The area is the overhang area of
    the chosen way, 0.0 for a part that prints without support."""
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64).reshape(-1, 3)
    if not len(T):
        return DOWN_DIR.copy(), 0.0
    scored = [(overhang_area(V, T, d, limit_deg), area, d)
              for d, area in hull_directions(manifold_from_mesh(V, T),
                                              max_candidates=max_candidates)]
    best = min(score for score, _area, _d in scored)
    tied = [c for c in scored if c[0] <= best * (1.0 + ORIENT_TIE) + ORIENT_EPS]
    top = max(area for _score, area, _d in tied)
    tied = [c for c in tied if c[1] >= top * (1.0 - ORIENT_TIE) - ORIENT_EPS]
    for score, _area, d in tied:
        if is_down(d):
            return d, score
    score, _area, d = min(tied, key=lambda c: c[0])
    return d, score


def orient_direction(g: Group, limit_deg):
    """The way the part of `g` has to point down, or None when `g` cannot be turned.

    Reads the group and changes nothing, so it can run in a background task, as
    `elephant_foot_chamfer` does. None for a locked group and for one that is not a closed
    volume; see `best_down_direction` for the way itself."""
    if g.locked:
        return None
    try:
        V, T = mesh_arrays(closed_manifold(g))
    except Exception:
        return None
    if not len(T):
        return None
    return best_down_direction(V, T, limit_deg)[0]


def orient_group(doc: Document, gid, d) -> bool:
    """Turn a group until `d` points straight down and drop it on the bed, like `lay_flat`.

    The group turns about the centre of its own bounding box; one that already stands with
    `d` down is only dropped on the bed. True when the group was turned, False when it only
    came down. A locked group is not touched; the caller takes the checkpoint, so one undo
    brings the old place back."""
    g = doc.group(gid)
    if g is None or g.locked:
        return False
    item = [("g", g.id, g.id)]
    if is_down(d):
        drop_to_bed(doc, item)
        return False
    bb = g.bbox()
    if bb is None:
        return False
    c = 0.5 * (np.asarray(bb[0], dtype=float) + np.asarray(bb[1], dtype=float))
    R = joints.align_rotation(d, DOWN_DIR)
    transform_selection(doc, R, c - R @ c, items=item)
    drop_to_bed(doc, item)
    return True


def orient_for_printing(doc: Document, items, limit_deg):
    """Orient for Printing: turn the given groups so each stands on the face that needs the
    least support, like the auto-orient of a slicer.

    Every group turns about the centre of its own bounding box until that face points
    straight down and is dropped on the bed afterwards; a group that already stands the right
    way round is only dropped on the bed. The caller takes the checkpoint, so one undo brings
    the old places back. Returns (groups turned, groups that already stood right, groups
    skipped); see `orient_direction` for the groups that are skipped."""
    turned = unchanged = skipped = 0
    seen = set()
    for k, _ctx, i in items:
        g = doc.group(i) if k == "g" else None
        if g is None or g.id in seen:
            continue
        seen.add(g.id)
        d = orient_direction(g, limit_deg)
        if d is None:
            skipped += 1
        elif orient_group(doc, g.id, d):
            turned += 1
        else:
            unchanged += 1
    return turned, unchanged, skipped


def mirror(doc: Document, items, axis):
    bb = selection_bbox(doc, items)
    if bb is None:
        return
    c = bb.mean(axis=0)
    S = np.eye(3)
    S[axis, axis] = -1
    transform_selection(doc, S, c - S @ c, items=items)


def resize_group(doc: Document, g: Group, size):
    if g.locked:
        return
    lo, hi = g.bbox()
    cur = hi - lo
    f = np.array([size[i] / cur[i] if cur[i] > 1e-9 else 1.0 for i in range(3)])
    S = np.diag(f)
    g.ents.transform(S, lo - S @ lo)


def inlay(doc: Document, ents: Entities, fid, d, color):
    """Volura extra (Alt + push/pull): the extruded volume becomes its own colored part.

    d < 0: a pocket is pushed into the face and filled with a new part (inlay).
    d > 0: a new part is placed on the face, the face itself is not changed."""
    plane = ents.face_plane(fid)
    polys = [plane.to_local(L) for L in ents.face_points(fid)]
    cs = mf.CrossSection(polys, mf.FillRule.EvenOdd)
    prism = extrude_cs(cs, plane, min(0.0, d), max(0.0, d))
    if d < 0:
        ents.pushpull(fid, d)
    return doc.add_group(Entities.from_manifold(prism), color)


# ------------------------------------------------------------- follow me

def loop_plane(loop):
    """(origin, normal) of a planar loop; the normal follows the loop's winding."""
    P = np.asarray(loop, dtype=float)
    c = P.mean(axis=0)
    n = np.zeros(3)
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        n += np.cross(a - c, b - c)
    return c, normalize(n)


def loop_area_vector(loop, c):
    P = np.asarray(loop, dtype=float) - c
    return 0.5 * np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0)


def _swing(a, b):
    """The rotation that turns the unit vector a into b (rotating about a x b)."""
    a, b = normalize(a), normalize(b)
    f = np.cross(a, b)
    s, c = float(np.linalg.norm(f)), float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return rotation_matrix(normalize(np.cross(default_u(a), a)), math.pi)
    return rotation_matrix(normalize(f), math.atan2(s, c))


def _in_plane_basis(d):
    """An orthonormal (p, q) with p x q = d."""
    p = normalize(default_u(d) - d * np.dot(default_u(d), d))
    return p, np.cross(d, p)


def follow_me(profile_loops_world, path_points, closed):
    """Sweep a profile along a path, like SketchUp's 'Volg mij'.

    `profile_loops_world` is the outer loop of the profile followed by its holes
    (world coordinates); `path_points` are the points of the path in order (for a
    closed path the first point is not repeated). The profile is carried along the
    path: on a straight run it stands perpendicular to the path, and at a corner it
    is projected along the incoming direction into the bissectrice plane, which is
    what makes the sweep mitre like SketchUp's. Returns the swept solid."""
    loops = [np.asarray(L, dtype=float).reshape(-1, 3) for L in profile_loops_world]
    loops = [L for L in loops if len(L) >= 3]
    P = np.asarray(path_points, dtype=float).reshape(-1, 3)
    if len(P) > 1:
        P = P[[0] + [i for i in range(1, len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-9]]
    if closed and len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 1e-9:
        P = P[:-1]
    if not loops or len(P) < (3 if closed else 2):
        return mf.Manifold()
    n = len(P)

    c, m = loop_plane(loops[0])                       # the profile's own plane
    loops = [loops[0]] + [L[::-1] if np.dot(loop_area_vector(L, c), m) > 0 else L for L in loops[1:]]
    if closed:                                        # start where the profile is
        k = int(np.argmin(np.linalg.norm(P - c, axis=1)))
        go, come = normalize(P[(k + 1) % n] - P[k]), normalize(P[k] - P[k - 1])
        if abs(float(m @ come)) > abs(float(m @ go)):     # the profile stands on the other segment
            k = (k - 1) % n
        P = np.roll(P, -k, axis=0)

    d = [normalize(P[(i + 1) % n] - P[i]) for i in (range(n) if closed else range(n - 1))]
    if float(m @ d[0]) < 0:      # the profile runs the same way round as the path
        loops = [L[::-1] for L in loops]
    p, q = _in_plane_basis(d[0])
    coords = [np.stack([(L - P[0]) @ p, (L - P[0]) @ q], axis=1) for L in loops]

    frames = [(p, q)]
    for i in range(1, len(d)):
        R = _swing(d[i - 1], d[i])
        frames.append((R @ frames[-1][0], R @ frames[-1][1]))

    copies = []
    for i in range(n):
        if not closed and i == 0:
            A, B = frames[0]
        elif not closed and i == n - 1:
            A, B = frames[-1]
        else:
            din = d[i - 1]
            nb = normalize(din + d[i])
            den = float(din @ nb)
            if den < 1e-9:                       # the path folds back on itself
                return mf.Manifold()
            A, B = frames[i - 1]
            A = A - np.dot(A, nb) / den * din    # the section projected into the mitre plane
            B = B - np.dot(B, nb) / den * din
        copies.append([P[i] + np.outer(xy[:, 0], A) + np.outer(xy[:, 1], B) for xy in coords])

    index, verts = {}, []
    for i in range(n):
        for k, Q in enumerate(copies[i]):
            index[(i, k)] = np.arange(len(verts), len(verts) + len(Q))
            verts.extend(Q)
    verts = np.asarray(verts, dtype=float).reshape(-1, 3)

    tris = []
    steps = range(n) if closed else range(n - 1)
    for i in steps:
        j = (i + 1) % n
        for k, Q in enumerate(copies[i]):
            a, b = index[(i, k)], index[(j, k)]
            for t in range(len(Q)):
                w, x = a[t], a[(t + 1) % len(Q)]
                y, z = b[(t + 1) % len(Q)], b[t]
                tris.append((w, x, z))
                tris.append((x, y, z))

    try:
        cap = np.asarray(mf.triangulate(coords), dtype=np.int64).reshape(-1, 3)
    except Exception:
        cap = np.zeros((0, 3), dtype=np.int64)
    if not len(cap):
        return mf.Manifold()
    if not closed:
        # mf.triangulate numbers the points of the loops one after another
        for i, rev in ((n - 1, False), (0, True)):
            at = np.concatenate([index[(i, k)] for k in range(len(coords))])
            for t in cap:
                tris.append(tuple(at[t[k]] for k in (2, 1, 0)) if rev else tuple(at[t[k]] for k in (0, 1, 2)))

    man = manifold_from_mesh(verts, np.asarray(tris, dtype=np.int64))
    if man.status() == mf.Error.NoError and man.volume() < 0:
        man = manifold_from_mesh(verts, np.asarray(tris, dtype=np.int64)[:, ::-1])
    return man


# ------------------------------------------------------------- primitives

def primitive(kind, a, b=0.0, c=0.0, segments=64):
    if kind == "box":
        return mf.Manifold.cube((a, b, c))
    if kind == "cylinder":
        return mf.Manifold.cylinder(b, a, a, segments)
    if kind == "cone":
        return mf.Manifold.cylinder(c, a, b, segments)
    if kind == "sphere":
        return mf.Manifold.sphere(a, segments).translate((0, 0, a))
    if kind == "torus":
        circle = mf.CrossSection.circle(b, max(16, segments // 2)).translate((a, 0))
        return mf.Manifold.revolve(circle, segments).translate((0, 0, b))
    raise ValueError(kind)


# ------------------------------------------------------------- printing

class Part:
    def __init__(self, name, color, manifold=None, problem=None):
        self.name = name
        self.color = color
        self.manifold = manifold
        self.problem = problem


def parts(doc: Document, only_selected=False):
    """Printable parts: every visible group and every closed shell of loose geometry."""
    groups = doc.selected_groups() if only_selected else doc.visible_groups()
    items = [(g.name, g.color, g.ents, list(g.ents.F)) for g in groups]
    if not only_selected:
        for k, shell in enumerate(doc.root.shells(), 1):
            colors = Counter(doc.root.F[f].color for f in shell if doc.root.F[f].color)
            color = colors.most_common(1)[0][0] if colors else DEFAULT_COLOR
            items.append((tr("Loose geometry {k}").format(k=k), color, doc.root, shell))
    return build_parts(items)


def color_id(color):
    """'#rrggbb' as a number + 1, to tag triangles with their face color (0 = no color,
    so black stays distinguishable)."""
    try:
        return int(str(color).lstrip("#")[:6], 16) + 1
    except ValueError:
        return 0


def id_color(value):
    """The color of a triangle tag, or None for an untagged triangle."""
    value = int(value)
    return f"#{(value - 1) & 0xFFFFFF:06x}" if value > 0 else None


def _finish_part(name, color, m):
    """The Part a triangulated, closed shell makes once its Manifold is built."""
    if m.status() != mf.Error.NoError or m.is_empty():
        return Part(name, color, problem=tr("not a valid volume ({status})").format(status=m.status().name))
    if m.volume() < 0:
        return Part(name, color, problem=tr("faces are inside out"))
    return Part(name, color, m)


def _part(name, color, ents, fids):
    open_e = ents.open_edges(fids)
    if open_e:
        return Part(name, color, problem=tr("not closed: {n} open or doubled edge(s)").format(n=len(open_e)))
    # every triangle carries the color of its face (a painted face, or the part color)
    m = ents.to_manifold(fids, face_ids={f: color_id(ents.F[f].color or color) for f in fids})
    return _finish_part(name, color, m)


def part_check_job(payload):
    """Worker entry point: ``(volume, surface, problem)`` of a closed shell's ``mesh_payload``.

    `problem` is None, ``("invalid", status name)`` or ``("inside_out", None)``; the main
    process turns it into text, because a worker does not load the translations."""
    tris, tags = triangulate_job(payload)
    m = manifold_from_tris(payload[0], tris, tags)
    if m.status() != mf.Error.NoError or m.is_empty():
        return 0.0, 0.0, ("invalid", m.status().name)
    if m.volume() < 0:
        return 0.0, 0.0, ("inside_out", None)
    return m.volume(), m.surface_area(), None


def part_summary(name, color, ents, pool=False):
    """``(volume, surface, problem)`` of a group as Entity Info shows it: what ``_part``
    finds, with `problem` the text of what is wrong, or None when it prints.

    With `pool` the triangulation, the heavy part, runs in a worker process (when the
    pool is on), so a background thread that waits for it does not hold the GIL and
    the window stays smooth while a big group is checked."""
    fids = list(ents.F)
    open_e = ents.open_edges(fids)
    if open_e:
        return 0.0, 0.0, tr("not closed: {n} open or doubled edge(s)").format(n=len(open_e))
    payload = mesh_payload(ents, fids)
    if pool:
        vol, area, problem = workers.parallel_map(part_check_job, [payload])[0]
    else:
        vol, area, problem = part_check_job(payload)
    if problem is None:
        return vol, area, None
    if problem[0] == "invalid":
        return 0.0, 0.0, tr("not a valid volume ({status})").format(status=problem[1])
    return 0.0, 0.0, tr("faces are inside out")


def filament_use(volume_mm3, density_g_cm3, diameter_mm, price_per_kg):
    """``(grams, metres, cost)`` of filament for a solid of `volume_mm3` mm³.

    The weight is the solid volume times the material density; the length follows from
    the cross-section of the filament and the cost from the price per kilogram. A print
    with infill is lighter, so this is the most a part of this volume can use."""
    grams = volume_mm3 / 1000.0 * density_g_cm3
    length_mm = volume_mm3 / (math.pi * (diameter_mm / 2.0) ** 2)
    return grams, length_mm / 1000.0, grams / 1000.0 * price_per_kg


# Triangulating a group in a worker only pays once it is big enough to beat the
# cost of pickling its loops over and the triangles back. Measured on the
# benchmark model (24 groups, 16k faces) ops.parts falls from ~2.0 s to ~0.7 s
# with the pool; a group of a few hundred faces gains nothing, so small models
# and VOLURA_WORKERS=0 keep the work in the main process.
PARTS_POOL_MIN_GROUPS = 2
PARTS_POOL_MIN_FACES = 1200


def build_parts(items):
    """The parts for ``[(name, color, ents, fids)]``, triangulated over the cores when that pays.

    The open-edge check reads the live geometry and stays here; the per-face
    triangulation - the heavy part of an export - goes to the pool as picklable
    arrays (see ``brep.mesh_payload``) and every ``Manifold`` is rebuilt here from
    the triangles that come back. The result is identical to calling ``_part`` on
    each item, only faster for a model with enough faces to spread."""
    prepared = []
    for name, color, ents, fids in items:
        open_e = ents.open_edges(fids)
        if open_e:
            prepared.append((name, color, open_e, None))
        else:
            face_ids = {f: color_id(ents.F[f].color or color) for f in fids}
            prepared.append((name, color, open_e, mesh_payload(ents, fids, face_ids)))
    jobs = [p for _, _, _, p in prepared if p is not None]
    faces = sum(len(p[1]) for p in jobs)
    if len(jobs) >= PARTS_POOL_MIN_GROUPS and faces >= PARTS_POOL_MIN_FACES:
        results = workers.parallel_map(triangulate_job, jobs, min_items=PARTS_POOL_MIN_GROUPS)
    else:
        results = [triangulate_job(p) for p in jobs]
    out, it = [], iter(results)
    for name, color, open_e, payload in prepared:
        if payload is None:
            out.append(Part(name, color,
                            problem=tr("not closed: {n} open or doubled edge(s)").format(n=len(open_e))))
        else:
            tris, tags = next(it)
            out.append(_finish_part(name, color, manifold_from_tris(payload[0], tris, tags)))
    return out


def print_parts(part_list):
    """One watertight solid per color; where colors overlap the later part wins."""
    valid = [p for p in part_list if p.manifold is not None]
    resolved = []
    for i, p in enumerate(valid):
        m = p.manifold
        later = [q.manifold for q in valid[i + 1:] if q.color != p.color]
        if later:
            m = m - union_all(later)
        resolved.append((p.color, m))
    by_color = {}
    for color, m in resolved:
        by_color.setdefault(color, []).append(m)
    out = []
    for color, ms in by_color.items():
        u = union_all(ms)
        if not u.is_empty():
            out.append((color, u))
    return out


def check_parts(part_list, bed=None):
    msgs = []
    for p in part_list:
        if p.problem:
            msgs.append(("fout", tr("{name}: {problem}.").format(name=p.name, problem=p.problem)))
            continue
        m = p.manifold
        bb = m.bounding_box()
        lo, hi = np.array(bb[:3]), np.array(bb[3:])
        msgs.append(("ok", tr("{name}: closed volume, {tris} triangles, volume {vol:.2f} cm³.")
                     .format(name=p.name, tris=m.num_tri(), vol=m.volume() / 1000)))
        if lo[2] < -1e-4:
            msgs.append(("waarschuwing", tr("{name}: sticks out {d:.2f} mm below the print bed.")
                         .format(name=p.name, d=-lo[2])))
        if bed and (lo[0] < -1e-4 or lo[1] < -1e-4 or hi[0] > bed[0] + 1e-4 or hi[1] > bed[1] + 1e-4
                    or hi[2] > bed[2] + 1e-4):
            dx, dy, dz = hi - lo
            bx, by, bz = bed
            if dx <= bx + 1e-4 and dy <= by + 1e-4 and dz <= bz + 1e-4:
                msgs.append(("waarschuwing",
                             tr("{name}: lies (partly) outside the print volume; Arrange on Bed puts "
                                "it on the bed.").format(name=p.name)))
            elif dy <= bx + 1e-4 and dx <= by + 1e-4 and dz <= bz + 1e-4:
                msgs.append(("waarschuwing",
                             tr("{name}: only fits on the bed turned a quarter turn.").format(name=p.name)))
            else:
                msgs.append(("waarschuwing",
                             tr("{name}: is {dx:.0f} × {dy:.0f} × {dz:.0f} mm, larger than the print "
                                "volume of {bx:.0f} × {by:.0f} × {bz:.0f} mm; Cut and Join splits it "
                                "into parts that fit.").format(name=p.name, dx=dx, dy=dy, dz=dz,
                                                               bx=bx, by=by, bz=bz)))
    valid = [p for p in part_list if p.manifold is not None]
    if valid:
        zmin = min(p.manifold.bounding_box()[2] for p in valid)
        if zmin > 1e-4:
            msgs.append(("waarschuwing", tr("The model floats {z:.2f} mm above the bed.").format(z=zmin)))
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            if a.color == b.color:
                continue
            ov = (a.manifold ^ b.manifold).volume()
            if ov > 1e-6:
                msgs.append(("info", tr("{a} and {b} overlap ({ov:.2f} mm³); on export {b} wins.")
                             .format(a=a.name, b=b.name, ov=ov)))
    return msgs


# ------------------------------------------------------------- overhangs
#
# An FDM printer cannot print a face that hangs over too far without support; slicers
# use 45° from vertical as the usual limit. The overhang view (View > Face Style >
# Overhangs) marks those faces; this is the pure classification it and the tests use.

def overhang_mask(normals, centroids_z, bed_z, limit_deg):
    """Which faces hang over too far to print without support, as a bool array.

    The overhang angle of a face is measured from the vertical: ``a = degrees(asin(max(0,
    -n.z)))`` for the unit outward normal ``n``, so a vertical wall is 0° and a face pointing
    straight down is 90°. A face is marked when ``a`` is greater than `limit_deg`: the
    underside of a ledge sloping 30° from horizontal has a = 60° and is marked, one sloping
    60° from horizontal has a = 30° and is not. The exception is the bottom of the model: a
    face that lies on the print bed (its centroid within 0.01 mm of `bed_z`, the lowest z of
    the geometry) prints fine even pointing straight down, so it is never marked.

    `normals` is ``(n, 3)`` and `centroids_z` ``(n,)``; the answer is ``(n,)``.
    """
    n = np.asarray(normals, dtype=float)
    if not len(n):
        return np.zeros(0, dtype=bool)
    length = np.linalg.norm(n, axis=1)
    length[length == 0.0] = 1.0
    angle = np.degrees(np.arcsin(np.clip(-n[:, 2] / length, 0.0, 1.0)))
    on_bed = np.abs(np.asarray(centroids_z, dtype=float) - bed_z) <= 0.01
    # a face right at the limit is allowed; the margin keeps rounding from marking it
    return (angle > limit_deg + 1e-6) & ~on_bed


# ------------------------------------------------------------- print inspection
#
# What the SketchUp extension "Solid Inspector" shows: why a part is not a solid.
# Every part is a group or one closed shell of loose geometry; every problem
# carries the geometry it is about, so the panel can select and mark it.

OPEN_EDGE = tr("open edge")
MANY_FACES = tr("doubled edge")
LOOSE_EDGE = tr("loose edge")
INTERIOR_FACE = tr("interior face")
REVERSED_FACE = tr("reversed face")
NO_VOLUME = tr("no volume")

# a direction that never runs along an axis or a face diagonal of a box
INSIDE_DIR = normalize(np.array([0.313, 0.571, 0.759]))


class Problem:
    """One reason why a part (a group or a loose shell) is no printable volume."""

    def __init__(self, kind, ctx, part="", edges=(), faces=(), point=None, detail=""):
        self.kind = kind
        self.ctx = ctx                       # ROOT or the id of the group
        self.part = part
        self.edges = list(edges)
        self.faces = list(faces)
        self.point = None if point is None else np.asarray(point, dtype=float)
        self.detail = detail

    def items(self):
        """Selection items of the geometry this problem is about."""
        return [("e", self.ctx, e) for e in self.edges] + [("f", self.ctx, f) for f in self.faces]

    def text(self):
        return f"{self.kind} ({self.detail})" if self.detail else self.kind

    def __repr__(self):
        return f"<Problem {self.part}: {self.text()}>"


def _pair(a, b):
    return (a, b) if a < b else (b, a)


def _edge_mid(ents, eid):
    A, B = ents.edge_points(eid)
    return (np.asarray(A, dtype=float) + np.asarray(B, dtype=float)) / 2


def _face_center(ents, fid):
    """A point strictly inside a face (the middle of its largest triangle)."""
    tri, _ = ents.face_triangles(fid)
    if not len(tri):
        return np.mean(np.concatenate(ents.face_points(fid)), axis=0)
    ar = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return tri[int(np.argmax(ar))].mean(axis=0)


def _shells_of(ents, fids):
    """Connected pieces (through edges) of the faces in fids."""
    left = {f for f in fids if f in ents.F}
    out = []
    while left:
        f0 = left.pop()
        comp, stack = {f0}, [f0]
        while stack:
            f = stack.pop()
            for loop in ents.F[f].loops:
                n = len(loop)
                for i in range(n):
                    e = ents.ekey.get(_pair(loop[i], loop[(i + 1) % n]))
                    for g in ents.e2f.get(e, ()):
                        if g in left:
                            left.discard(g)
                            comp.add(g)
                            stack.append(g)
        out.append(comp)
    return out


def _part_edges(ents, fids):
    """(edges with one face, edges with more than two faces) of the faces in fids."""
    fids, one, many, seen = set(fids), [], [], set()
    for f in fids:
        for e in ents.face_edges(ents.F[f]):
            if e in seen:
                continue
            seen.add(e)
            k = len(ents.e2f.get(e, set()) & fids)
            if k == 1:
                one.append(e)
            elif k > 2:
                many.append(e)
    return sorted(one), sorted(many)


def _consistent_signs(ents, fids):
    """fid -> +1/-1 so that neighbours run their shared edge in opposite directions."""
    fids = set(fids)
    signs = {}
    for seed in sorted(fids):
        if seed in signs:
            continue
        signs[seed] = 1
        queue = [seed]
        while queue:
            f = queue.pop(0)
            for loop in ents.F[f].loops:
                n = len(loop)
                for i in range(n):
                    a, b = loop[i], loop[(i + 1) % n]
                    e = ents.ekey.get(_pair(a, b))
                    if e is None:
                        continue
                    for g in ents.e2f.get(e, ()):
                        if g not in fids or g in signs:
                            continue
                        same = any(l2[k] == a and l2[(k + 1) % len(l2)] == b
                                   for l2 in ents.F[g].loops for k in range(len(l2)))
                        signs[g] = -signs[f] if same else signs[f]
                        queue.append(g)
    return signs


def _signed_volume(ents, fids, signs=None):
    """The volume of a closed shell; negative when the normals point inwards."""
    signs = signs or {}
    v = 0.0
    for f in fids:
        tri, _ = ents.face_triangles(f)
        if not len(tri):
            continue
        v += signs.get(f, 1) * float(np.einsum("ij,ij->i", tri[:, 0],
                                               np.cross(tri[:, 1], tri[:, 2])).sum())
    return v / 6.0


def _faces_to_reverse(ents, fids):
    """The faces to flip so every closed shell in fids points outwards."""
    out = set()
    for comp in _shells_of(ents, fids):
        if not ents.open_edges(comp):     # an open shell has no inside: leave it alone
            signs = _consistent_signs(ents, comp)
            v = _signed_volume(ents, comp, signs)
            if abs(v) < 1e-9:
                continue
            want = 1 if v > 0 else -1     # the sign that makes the volume positive
            out |= {f for f in comp if signs.get(f, 1) != want}
    return out


def _points_in_mesh(tri, points, chunk=64):
    """Which of `points` lie inside the closed triangle soup `tri` (ray parity)."""
    out = np.zeros(len(points), dtype=bool)
    if not len(tri) or not len(points):
        return out
    p0 = tri[:, 0]
    e1, e2 = tri[:, 1] - p0, tri[:, 2] - p0
    d = INSIDE_DIR
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)
        for s in range(0, len(points), chunk):
            tvec = points[s:s + chunk][:, None, :] - p0
            u = np.einsum("ctj,tj->ct", tvec, pvec) * inv
            qvec = np.cross(tvec, e1)
            v = np.einsum("ctj,j->ct", qvec, d) * inv
            t = np.einsum("ctj,tj->ct", qvec, e2) * inv
            hit = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
            out[s:s + chunk] = hit.sum(axis=1) % 2 == 1
    return out


def _solid_triangles(ents):
    """The triangles of every closed shell of ents, for point-in-solid tests."""
    refs = []
    for shell in ents.shells():
        if ents.open_edges(shell):
            continue
        m = ents.to_manifold(shell)
        if m.status() != mf.Error.NoError or m.is_empty() or abs(m.volume()) < 1e-9:
            continue
        V, T = mesh_arrays(m)
        if len(T):
            refs.append(V[T])
    return np.concatenate(refs) if refs else np.zeros((0, 3, 3))


def _interior_faces(ents, fids=None, tri=None):
    """Faces that lie inside a closed volume: solid on both sides of the face."""
    cands = sorted(set(ents.F) if fids is None else {f for f in fids if f in ents.F})
    if not cands:
        return []
    tri = _solid_triangles(ents) if tri is None else tri
    if not len(tri):
        return []
    bb = ents.bbox()
    diag = float(np.linalg.norm(bb[1] - bb[0])) if bb is not None else 1.0
    eps = min(0.5, max(1e-3, 1e-4 * diag))
    pts = []
    for f in cands:
        c, n = _face_center(ents, f), ents.F[f].normal
        pts += [c + n * eps, c - n * eps]
    inside = _points_in_mesh(tri, np.asarray(pts, dtype=float))
    return [f for k, f in enumerate(cands) if bool(inside[2 * k]) and bool(inside[2 * k + 1])]


def _inspect_part(ents, ctx, name, fids, loose=(), tri=None):
    """The problems of one part: the faces `fids` and the loose edges `loose`."""
    fids = {f for f in fids if f in ents.F}
    loose = [e for e in loose if e in ents.E]
    probs = []
    if loose:
        probs.append(Problem(LOOSE_EDGE, ctx, name, edges=loose, point=_edge_mid(ents, loose[0]),
                             detail=tr("{n} edge(s) without a face").format(n=len(loose))))
    one, many = _part_edges(ents, fids)
    if one:
        probs.append(Problem(OPEN_EDGE, ctx, name, edges=one, point=_edge_mid(ents, one[0]),
                             detail=tr("{n} edge(s) with 1 face: hole").format(n=len(one))))
    if many:
        probs.append(Problem(MANY_FACES, ctx, name, edges=many, point=_edge_mid(ents, many[0]),
                             detail=tr("{n} edge(s) with more than 2 faces").format(n=len(many))))
    inner = _interior_faces(ents, fids, tri)
    if inner:
        probs.append(Problem(INTERIOR_FACE, ctx, name, faces=inner,
                             point=_face_center(ents, inner[0]),
                             detail=tr("{n} face(s) inside a volume").format(n=len(inner))))
    flipped = _faces_to_reverse(ents, fids) if not one and not many else set()
    if flipped:
        probs.append(Problem(REVERSED_FACE, ctx, name, faces=sorted(flipped),
                             point=_face_center(ents, sorted(flipped)[0]),
                             detail=tr("{n} face(s) with the normal pointing inwards")
                             .format(n=len(flipped))))
    if fids and not probs:
        m = ents.to_manifold(fids)
        if m.status() != mf.Error.NoError or m.is_empty() or abs(m.volume()) < 1e-9:
            oorzaak = m.status().name if m.status() != mf.Error.NoError else tr("empty")
            probs.append(Problem(NO_VOLUME, ctx, name, faces=sorted(fids),
                                 point=_face_center(ents, sorted(fids)[0]),
                                 detail=tr("manifold: {cause}").format(cause=oorzaak)))
    return probs


def _inspect_context(ents, ctx, name):
    """Problems of everything in one context: one part per shell of faces."""
    out = []
    shells = _shells_of(ents, set(ents.F))
    tri = _solid_triangles(ents)
    loose = [e for e in ents.E if not ents.e2f.get(e)]
    left = set(loose)
    for k, shell in enumerate(sorted(shells, key=lambda s: min(s)), 1):
        vids = {v for f in shell for loop in ents.F[f].loops for v in loop}
        mine = [e for e in left if set(ents.E[e]) & vids]
        left -= set(mine)
        label = name if len(shells) == 1 else f"{name} {k}"
        out += _inspect_part(ents, ctx, label, shell, mine, tri)
    if left:
        out += _inspect_part(ents, ctx, tr("{name} (loose edges)").format(name=name), set(),
                             sorted(left), tri)
    return out


def inspect(doc: Document):
    """Printability problems per part, like Solid Inspector: a list of Problem."""
    out = []
    for g in doc.groups:
        if g.visible:
            out += _inspect_context(g.ents, g.id, g.name)
    out += _inspect_context(doc.root, ROOT, tr("Loose geometry"))
    return out


def _repair_ents(ents):
    """Make one context as closed and solid as possible; True when something changed."""
    changed = False
    for _ in range(3):
        before = (len(ents.F), len(ents.E))
        inner = _interior_faces(ents)
        if inner:
            ents.erase_faces(inner)
            changed = True
        loose = ents.loose_edges()
        if loose:
            ents.remove([], loose)
            changed = True
        ents.heal(list(ents.V))
        open_e = ents.open_edges()
        if open_e:
            ents.make_faces(open_e)
            changed = True
        flip = _faces_to_reverse(ents, set(ents.F))
        if flip:
            ents.reverse_faces(flip)
            changed = True
        if (len(ents.F), len(ents.E)) == before:
            break
    ents.touch()
    return changed


def repair_all(doc: Document):
    """Print Check > Repair All: one undo step that closes and orients every part.

    Loose edges go, collinear edges are joined, faces are oriented outwards, faces
    inside a volume are removed and flat open loops get a face."""
    before = doc._state()
    changed = _repair_ents(doc.root)
    for g in doc.groups:
        if not g.locked:
            changed |= _repair_ents(g.ents)
    if changed:
        doc.push_state(before)        # one undo step, and only when there is something to undo
    doc.notify()
    return changed
