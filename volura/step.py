"""STEP export: the parts as solids with flat faces, for other CAD programs.

An STL or a 3MF is a bag of triangles; a CAD program like Fusion, FreeCAD, SolidWorks or
Onshape wants a solid it can measure, cut and fillet. STEP (ISO 10303-21, schema AP214
"automotive design") carries that: every part becomes a MANIFOLD_SOLID_BREP of planar
faces, edges and vertices.

The faces are the flat faces of the part, not its triangles: the triangles that lie in
one plane and touch along an edge are merged into one ADVANCED_FACE, with the outline of
the region as its outer bound and every hole in it as an inner bound. A box comes out as
six faces, a plate with a hole as a plate face with a hole in it. Curved surfaces stay
the flat facets they are in Volura (a cylinder of 96 segments has 96 side faces).

Every edge of the mesh between two faces is one EDGE_CURVE on a LINE, used once in each
direction by the two faces it separates; that is what makes the shell closed. A straight
outline cut into several mesh edges stays several edges: that is valid STEP and keeps the
two faces along it agreeing about the vertices they share. No Qt in this file.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from . import __version__
from .geom import mesh_arrays
from .i18n import tr

#: how far apart two triangles may be in direction (length of the difference of their unit
#: normals) and in distance from the origin (mm) and still be one flat face
PLANE_TOL = 1e-6
#: the distance below which two points are the same, as written in the file (mm)
UNCERTAINTY = 1e-6


def _real(x) -> str:
    """A number as STEP writes it: always with a decimal point (``25.``, ``1.5E-07``)."""
    s = "%.12G" % float(x)
    if s in ("-0", "0"):
        return "0."
    mant, _, exp = s.partition("E")
    if "." not in mant:
        mant += "."
    return mant + ("E" + exp if exp else "")


def _text(s) -> str:
    """A string as STEP writes it: in single quotes, a quote doubled, plain ASCII."""
    s = str(s).encode("ascii", "replace").decode("ascii")
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _perpendicular(n):
    """A unit vector at right angles to the unit vector `n`."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    return u / np.linalg.norm(u)


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def flat_faces(verts, tris):
    """The flat faces of a closed mesh: a group number per triangle and the planes.

    Two triangles are in one group when they share an edge and lie in the same plane
    within PLANE_TOL. Returns `(group, normals, offsets, twins)`: `group[t]` is the face
    of triangle t (0, 1, 2, ...), `normals[g]` the outward unit normal of face g,
    `offsets[g]` its distance from the origin along it and `twins[t, k]` the triangle on
    the other side of edge k of triangle t (-1 if none). A triangle without area joins a
    neighbour, so it never splits a face.
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    n = np.zeros_like(cross)
    ok = area2 > 1e-12
    n[ok] = cross[ok] / area2[ok, None]
    d = np.einsum("ij,ij->i", n, a)
    nt = len(tris)
    # every directed edge (a, b) of a triangle, and the triangle that has (b, a)
    he_from = tris.reshape(-1)
    he_to = tris[:, [1, 2, 0]].reshape(-1)
    he_tri = np.repeat(np.arange(nt), 3)
    key = he_from.astype(np.int64) * (len(verts) + 1) + he_to
    twin_key = he_to.astype(np.int64) * (len(verts) + 1) + he_from
    order = np.argsort(key)
    pos = np.searchsorted(key[order], twin_key)
    pos = np.clip(pos, 0, len(key) - 1)
    found = key[order][pos] == twin_key
    twin_tri = np.where(found, he_tri[order][pos], -1)
    parent = list(range(nt))
    u_all = np.maximum(twin_tri, 0)
    same = ((twin_tri > he_tri) & ok[he_tri] & ok[u_all]
            & (np.linalg.norm(n[he_tri] - n[u_all], axis=1) <= PLANE_TOL)
            & (np.abs(d[he_tri] - d[u_all]) <= PLANE_TOL))
    for t, u in zip(he_tri[same].tolist(), twin_tri[same].tolist()):
        rt, ru = _find(parent, t), _find(parent, u)
        if rt != ru:
            parent[ru] = rt
    # a sliver without area joins a neighbour, so it never breaks up a face
    for t, u in zip(he_tri.tolist(), twin_tri.tolist()):
        if not ok[t] and u >= 0 and ok[u]:
            rt, ru = _find(parent, t), _find(parent, u)
            if rt != ru:
                parent[rt] = ru
    roots = np.array([_find(parent, t) for t in range(nt)])
    _, group = np.unique(roots, return_inverse=True)
    groups = group.max() + 1 if nt else 0
    weights = np.zeros((groups, 3))
    np.add.at(weights, group, cross)
    normals = weights / np.maximum(np.linalg.norm(weights, axis=1), 1e-300)[:, None]
    centre = (a + b + c) / 3.0
    sums = np.zeros((groups, 3))
    np.add.at(sums, group, centre * area2[:, None])
    total = np.zeros(groups)
    np.add.at(total, group, area2)
    points = sums / np.maximum(total, 1e-300)[:, None]
    offsets = np.einsum("ij,ij->i", normals, points)
    return group, normals, offsets, twin_tri.reshape(-1, 3)


def _loops(edges, at):
    """Chain directed boundary edges (a, b) into closed loops of vertex numbers.

    The face lies to the left of its edges. Where two loops of one face touch in a vertex,
    the way on is the edge that turns furthest to the right, which stays on the loop that
    came in, so each loop is a simple polygon. `at(i)` is vertex i in the plane.
    """
    out = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
    loops = []
    while out:
        start = next(iter(out))
        loop = [start]
        cur, prev = start, None
        while True:
            ways = out[cur]
            if len(ways) > 1 and prev is not None:
                here = at(cur)
                bx, by = at(prev) - here
                back = math.atan2(by, bx)

                def turn(b):
                    dx, dy = at(b) - here
                    return (math.atan2(dy, dx) - back) % (2.0 * math.pi)

                ways.sort(key=turn)          # the last one turns furthest to the right
            nxt = ways.pop()
            if not out[cur]:
                del out[cur]
            if nxt == start:
                break
            loop.append(nxt)
            cur, prev = nxt, cur
            if cur not in out:          # an open chain: the mesh is not closed here
                break
        loops.append(loop)
    return loops


class _Writer:
    """The numbered entities of a STEP file, in the order they are made."""

    def __init__(self):
        self.lines = []

    def add(self, text) -> int:
        self.lines.append(text)
        return len(self.lines)


def _shell(w, man):
    """The CLOSED_SHELL of one connected piece of a manifold; returns its entity number.

    The faces keep the direction of the triangles: out of the material, also for the
    inside of a hollow part, whose faces then point into the hollow. The points, vertices
    and edges are made for all faces at once, so a part of many facets stays quick.
    """
    verts, tris = mesh_arrays(man)
    group, normals, offsets, twins = flat_faces(verts, tris)
    # a point and a vertex for every corner of the mesh
    text = [", ".join(_real(c) for c in p) for p in verts.tolist()]
    point = np.empty(len(verts), dtype=np.int64)
    vertex = np.empty(len(verts), dtype=np.int64)
    for i, xyz in enumerate(text):
        point[i] = w.add(f"CARTESIAN_POINT('',({xyz}))")
        vertex[i] = w.add(f"VERTEX_POINT('',#{point[i]})")
    # the boundary edges of each face: the edges whose other triangle is in another face
    tri_of = np.repeat(np.arange(len(tris)), 3)
    a = tris.reshape(-1)
    b = tris[:, [1, 2, 0]].reshape(-1)
    other = twins.reshape(-1)
    edge_group = group[tri_of]
    border = (other < 0) | (group[np.maximum(other, 0)] != edge_group)
    a, b, tri_of, edge_group = a[border], b[border], tri_of[border], edge_group[border]
    # one EDGE_CURVE per edge, from its lower to its higher vertex number
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys, first, which = np.unique(lo * (len(verts) + 1) + hi, return_index=True,
                                   return_inverse=True)
    lo, hi = lo[first], hi[first]
    vec = verts[hi] - verts[lo]
    length = np.linalg.norm(vec, axis=1)
    unit = np.where(length[:, None] > 0, vec / np.maximum(length, 1e-300)[:, None],
                    np.array([1.0, 0.0, 0.0]))
    curve = np.empty(len(keys), dtype=np.int64)
    for e, (d, n) in enumerate(zip(unit.tolist(), length.tolist())):
        direction = w.add("DIRECTION('',(" + ", ".join(_real(c) for c in d) + "))")
        vector = w.add(f"VECTOR('',#{direction},{_real(n)})")
        line = w.add(f"LINE('',#{point[lo[e]]},#{vector})")
        curve[e] = w.add(f"EDGE_CURVE('',#{vertex[lo[e]]},#{vertex[hi[e]]},#{line},.T.)")
    edge_of = {(int(x), int(y)): (int(curve[c]), x < y)
               for x, y, c in zip(a.tolist(), b.tolist(), which.reshape(-1).tolist())}
    # the faces, each with the plane it lies in
    us = np.where(np.abs(normals[:, :1]) < 0.9, np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]))
    us = np.cross(normals, us)
    us /= np.linalg.norm(us, axis=1)[:, None]
    vs = np.cross(normals, us)
    order = np.argsort(edge_group, kind="stable")
    starts = np.searchsorted(edge_group[order], np.arange(len(normals) + 1))
    faces = []
    for g in range(len(normals)):
        idx = order[starts[g]:starts[g + 1]]
        if len(idx) == 0:
            continue
        if len(idx) == 3 and len(set(tri_of[idx].tolist())) == 1:
            t = tris[tri_of[idx[0]]].tolist()     # a face of one triangle: its own outline
            loops, outer = [t], 0
        else:
            u, v = us[g], vs[g]
            loops = _loops(list(zip(a[idx].tolist(), b[idx].tolist())),
                           lambda i, u=u, v=v: np.array([verts[i] @ u, verts[i] @ v]))
            areas = []
            for loop in loops:
                q = verts[loop]
                x, y = q @ u, q @ v
                areas.append(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
            outer = int(np.argmax(areas))
        refs = []
        for i, loop in enumerate(loops):
            oes = []
            for j in range(len(loop)):
                c, forward = edge_of[(loop[j], loop[(j + 1) % len(loop)])]
                oes.append(w.add(f"ORIENTED_EDGE('',*,*,#{c},{'.T.' if forward else '.F.'})"))
            el = w.add("EDGE_LOOP('',(" + ",".join(f"#{e}" for e in oes) + "))")
            kind = "FACE_OUTER_BOUND" if i == outer else "FACE_BOUND"
            refs.append(w.add(f"{kind}('',#{el},.T.)"))
        n, u = normals[g].tolist(), us[g].tolist()
        # the plane goes through a corner of the face, so that corner is exactly on it
        corner = int(point[loops[outer][0]])
        dn = w.add("DIRECTION('',(" + ", ".join(_real(c) for c in n) + "))")
        du = w.add("DIRECTION('',(" + ", ".join(_real(c) for c in u) + "))")
        ax = w.add(f"AXIS2_PLACEMENT_3D('',#{corner},#{dn},#{du})")
        plane = w.add(f"PLANE('',#{ax})")
        faces.append(w.add("ADVANCED_FACE('',(" + ",".join(f"#{r}" for r in refs)
                           + f"),#{plane},.T.)"))
    return w.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")


def _inside(box, outer):
    """Whether the bounding box `box` lies within the bounding box `outer`."""
    return all(outer[i] <= box[i] + 1e-9 and box[i + 3] <= outer[i + 3] + 1e-9 for i in range(3))


def _solids(w, name, man):
    """The solids of one part; returns their entity numbers.

    A STEP solid is one closed shell, so a part of loose pieces (a grommet and its lid)
    becomes one solid per piece. A hollow inside a piece is a shell of negative volume:
    it goes with the piece around it as a void (BREP_WITH_VOIDS).
    """
    pieces = man.decompose()
    solid = [p for p in pieces if p.volume() > 0.0]
    hollow = [p for p in pieces if p.volume() <= 0.0]
    voids = {i: [] for i in range(len(solid))}
    for h in hollow:
        around = [i for i, p in enumerate(solid) if _inside(h.bounding_box(), p.bounding_box())]
        if around:
            voids[min(around, key=lambda i: solid[i].volume())].append(h)
    out = []
    for i, p in enumerate(solid):
        label = name if len(solid) == 1 else f"{name} {i + 1}"
        shell = _shell(w, p)
        if voids[i]:
            inner = [w.add(f"ORIENTED_CLOSED_SHELL('',*,#{_shell(w, h)},.F.)") for h in voids[i]]
            out.append(w.add(f"BREP_WITH_VOIDS({_text(label)},#{shell},("
                             + ",".join(f"#{v}" for v in inner) + "))"))
        else:
            out.append(w.add(f"MANIFOLD_SOLID_BREP({_text(label)},#{shell})"))
    return out


def step_text(solids, name="Volura") -> str:
    """The STEP file of `solids`, a list of (name, manifold3d.Manifold), as text."""
    solids = [(n, m) for n, m in solids if m is not None and not m.is_empty()]
    if not solids:
        raise ValueError(tr("There is nothing to export."))
    w = _Writer()
    ctx_app = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',"
          f"2000,#{ctx_app})")
    ctx_prod = w.add(f"PRODUCT_CONTEXT('',#{ctx_app},'mechanical')")
    product = w.add(f"PRODUCT({_text(name)},{_text(name)},'',(#{ctx_prod}))")
    w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))")
    formation = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    ctx_def = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx_app},'design')")
    definition = w.add(f"PRODUCT_DEFINITION('design','',#{formation},#{ctx_def})")
    shape = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
    length = w.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    angle = w.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid_angle = w.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    accuracy = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_real(UNCERTAINTY)}),"
                     f"#{length},'distance_accuracy_value','confusion accuracy')")
    context = w.add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
                    f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{accuracy})) "
                    f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) "
                    f"REPRESENTATION_CONTEXT('','') )")
    o = w.add("CARTESIAN_POINT('',(0.,0.,0.))")
    z = w.add("DIRECTION('',(0.,0.,1.))")
    x = w.add("DIRECTION('',(1.,0.,0.))")
    placement = w.add(f"AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")
    breps = [b for n, m in solids for b in _solids(w, n, m)]
    rep = w.add("ADVANCED_BREP_SHAPE_REPRESENTATION(" + _text(name) + ",(#" + str(placement)
                + "".join(f",#{b}" for b in breps) + f"),#{context})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{rep})")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    head = ["ISO-10303-21;", "HEADER;",
            "FILE_DESCRIPTION(('Volura model'),'2;1');",
            f"FILE_NAME({_text(name)},'{stamp}',(''),(''),"
            f"{_text('Volura ' + __version__)},{_text('Volura ' + __version__)},'');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
            "ENDSEC;", "DATA;"]
    body = [f"#{i}={line};" for i, line in enumerate(w.lines, 1)]
    return "\n".join(head + body + ["ENDSEC;", "END-ISO-10303-21;", ""])


def write_step(path, solids, name=None):
    """Write `solids` [(name, manifold)] to `path` as a STEP file; returns the face count."""
    from .fileio import atomic_write           # fileio imports a lot: only the writer needs it
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    text = step_text(solids, name)

    def write(tmp):
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write(text)

    atomic_write(path, write)
    return text.count("ADVANCED_FACE(")
