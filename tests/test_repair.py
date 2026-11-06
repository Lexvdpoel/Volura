"""Repairing meshes that are not one closed volume (`volura.repair`).

Every flaw is made by hand on a cube or a box, the way other programs get it wrong: faces
the wrong way round, faces on both sides, holes (also a concave one), parts that share an
edge, a cavity, inner walls of a SketchUp model and a flat sheet that holds nothing.

Run with:  python -m pytest tests
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from volura import repair  # noqa: E402

CORNERS = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


def quads_to_tris(quads):
    return np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])


def cube(size=10.0, at=(0, 0, 0)):
    return CORNERS * size + np.asarray(at, dtype=float), quads_to_tris(QUADS)


def fixed(v, t):
    m, report = repair.repair_mesh(v, t)
    assert m is not None, report
    return m, report


def test_a_closed_mesh_comes_back_unchanged():
    m, report = fixed(*cube())
    assert abs(m.volume() - 1000) < 1e-9 and not report.changed


def test_triangles_the_wrong_way_round_are_turned():
    v, t = cube()
    t[[0, 5, 7]] = t[[0, 5, 7]][:, [0, 2, 1]]
    m, report = fixed(v, t)
    assert report.flipped == 3 and abs(m.volume() - 1000) < 1e-9


def test_a_cube_turned_inside_out_comes_out_right():
    v, t = cube()
    m, report = fixed(v, t[:, [0, 2, 1]][:10].tolist() + t[10:].tolist())
    assert abs(m.volume() - 1000) < 1e-9


def test_faces_on_both_sides_count_once():
    v, t = cube()
    m, report = fixed(v, np.concatenate([t, t[:, [0, 2, 1]]]))
    assert report.duplicates == 12 and abs(m.volume() - 1000) < 1e-9


def test_a_missing_face_gets_a_lid():
    v, t = cube()
    m, report = fixed(v, t[2:])
    assert report.holes == 1 and abs(m.volume() - 1000) < 1e-9


def test_a_concave_hole_is_filled_flat():
    # an L-shaped prism without its top: ear clipping must not cut across the notch
    L = np.array([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20.0)])
    n = len(L)
    v = np.concatenate([np.c_[L, np.zeros(n)], np.c_[L, np.full(n, 5.0)]])
    # the bottom, facing down
    bottom = [(0, 2, 1), (0, 3, 2), (0, 5, 4), (0, 4, 3)]
    sides = [t for k in range(n) for t in ((k, (k + 1) % n, n + (k + 1) % n),
                                           (k, n + (k + 1) % n, n + k))]
    m, report = fixed(v, np.array(bottom + sides))
    assert report.holes == 1
    assert abs(m.volume() - 300 * 5) < 1e-6


def test_two_cubes_that_share_an_edge_come_apart_and_both_stay():
    va, ta = cube()
    vb, tb = cube(at=(10, 10, 0))
    m, report = fixed(np.concatenate([va, vb]), np.concatenate([ta, tb + 8]))
    assert report.crowded == 1 and abs(m.volume() - 2000) < 1e-6


def test_a_cavity_stays_a_cavity():
    vo, to = cube()
    vi, ti = cube(4.0, at=(3, 3, 3))
    # the outer cube misses a face (so it needs a repair), the inner one faces inwards
    v = np.concatenate([vo, vi])
    t = np.concatenate([to[2:], ti[:, [0, 2, 1]] + 8])
    m, report = fixed(v, t)
    assert report.holes == 1 and abs(m.volume() - (1000 - 64)) < 1e-6


def sketchup_two_rooms():
    """A 20 x 10 x 10 box drawn as two rooms: the wall between them at x = 10 is a face
    of its own, and every edge around it is shared by three faces."""
    xs = (0.0, 10.0, 20.0)
    idx = {}
    v = []
    for x in xs:
        for y in (0.0, 10.0):
            for z in (0.0, 10.0):
                idx[(x, y, z)] = len(v)
                v.append((x, y, z))
    q = []
    for x0, x1 in ((0.0, 10.0), (10.0, 20.0)):
        P = lambda x, y, z: idx[(x, y, z)]  # noqa: E731
        q += [(P(x0, 0, 0), P(x0, 10, 0), P(x1, 10, 0), P(x1, 0, 0)),       # bottom
              (P(x0, 0, 10), P(x1, 0, 10), P(x1, 10, 10), P(x0, 10, 10)),   # top
              (P(x0, 0, 0), P(x1, 0, 0), P(x1, 0, 10), P(x0, 0, 10)),       # front
              (P(x0, 10, 0), P(x0, 10, 10), P(x1, 10, 10), P(x1, 10, 0))]   # back
    P = lambda x, y, z: idx[(x, y, z)]  # noqa: E731
    q += [(P(0, 0, 0), P(0, 0, 10), P(0, 10, 10), P(0, 10, 0)),             # left end
          (P(20, 0, 0), P(20, 10, 0), P(20, 10, 10), P(20, 0, 10)),         # right end
          (P(10, 0, 0), P(10, 10, 0), P(10, 10, 10), P(10, 0, 10))]         # inner wall
    return np.array(v), quads_to_tris(q)


def test_an_inner_wall_between_two_rooms_is_dropped():
    m, report = fixed(*sketchup_two_rooms())
    assert report.walls == 2 and abs(m.volume() - 2000) < 1e-6


def test_a_flat_sheet_cannot_be_repaired():
    v = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0.0)])
    m, report = repair.repair_mesh(v, np.array([(0, 1, 2), (0, 2, 3)]))
    assert m is None and report.status is not None


def test_a_cube_seen_from_outside_has_no_hidden_faces_and_a_wall_does():
    v, t = cube()
    assert repair._seen_from_outside(v, t).all()
    v, t = sketchup_two_rooms()
    seen = repair._seen_from_outside(v, t)
    assert (~seen).sum() == 2 and np.allclose(v[t[~seen]][:, :, 0], 10)


def test_the_report_reads_as_a_list_of_what_was_done():
    _, report = fixed(*cube()[:1], cube()[1][2:])
    assert repair.describe(report) == ["1 holes filled"]
