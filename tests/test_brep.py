"""Geometry behaviour compared with SketchUp 2017.

The expected numbers (faces, edges, face areas, volumes) were recorded by running
the same operations in SketchUp 2017.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


def P(x, y, z=0):
    return np.array([x, y, z], dtype=float)


def rect(ents, x0, y0, x1, y1, z=0):
    pl = Plane((0, 0, z), (0, 0, 1))
    hit = [f for f in ents.faces_in_plane(pl)]
    if hit:
        pl = ents.face_plane(hit[0])
        pl = Plane((0, 0, z), pl.n)
    loop = pl.to_local(np.array([P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)]))
    return ents.add_shape(pl, [loop])


def face_at(ents, x, y, z):
    p = P(x, y, z)
    from volura.geom import point_in_polys
    for f in ents.F:
        pl = ents.face_plane(f)
        if abs(pl.distance(p)) > 1e-6:
            continue
        polys = [pl.to_local(L) for L in ents.face_points(f)]
        if point_in_polys(polys, pl.to_local(p)):
            return f
    raise AssertionError(f"no face at {p}")


def edge_at(ents, x, y, z):
    p = P(x, y, z)
    for e in ents.E:
        A, B = ents.edge_points(e)
        d = B - A
        t = np.clip((p - A) @ d / (d @ d), 0, 1)
        if np.linalg.norm(A + d * t - p) < 1e-6:
            return e
    raise AssertionError(f"no edge at {p}")


def box(w=40, d=30, h=10):
    ents = Entities()
    rect(ents, 0, 0, w, d)
    f = face_at(ents, w / 2, d / 2, 0)
    assert ents.F[f].normal[2] < 0  # ground faces look down, like SketchUp
    ents.pushpull(f, -h)
    return ents


def areas(ents):
    out = []
    for f in ents.F:
        tri, _ = ents.face_triangles(f)
        out.append(round(float(np.abs(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)).sum() / 2), 2))
    return sorted(out)


def volume(ents):
    m = ents.to_manifold()
    assert m.status().name == "NoError", m.status()
    return round(m.volume(), 2)


def counts(ents):
    return len(ents.F), len(ents.E)


def test_01_three_lines_close_triangle():
    e = Entities()
    e.add_line(P(0, 0), P(30, 0))
    e.add_line(P(30, 0), P(15, 20))
    e.add_line(P(15, 20), P(0, 0))
    assert counts(e) == (1, 3) and areas(e) == [300.0]


def test_02_line_splits_face():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    e.add_line(P(15, 0), P(15, 30))
    assert counts(e) == (2, 7) and areas(e) == [450.0, 750.0]


def test_03_erase_split_line_heals():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    e.add_line(P(15, 0), P(15, 30))
    e.erase_edges([edge_at(e, 15, 15, 0)])
    assert counts(e) == (1, 4) and areas(e) == [1200.0]


def test_04_crossing_lines_split_and_erase_part():
    e = Entities()
    e.add_line(P(0, 15), P(40, 15))
    e.add_line(P(20, 0), P(20, 30))
    assert counts(e) == (0, 4)
    e.erase_edges([edge_at(e, 8, 15, 0)])
    assert counts(e) == (0, 3)


def test_06_erase_rect_edge_removes_face():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    e.erase_edges([edge_at(e, 20, 0, 0)])
    assert counts(e) == (0, 3)


def test_07_redraw_edge_restores_face():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    e.erase_edges([edge_at(e, 20, 0, 0)])
    e.add_line(P(0, 0), P(40, 0))
    assert counts(e) == (1, 4) and areas(e) == [1200.0]


def test_08_half_line_does_not_split():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    e.add_line(P(15, 0), P(15, 15))
    assert counts(e) == (1, 6) and len(e.loose_edges()) == 1


def test_09_rect_in_corner():
    e = Entities()
    rect(e, 0, 0, 40, 30)
    rect(e, 0, 0, 15, 10)
    assert counts(e) == (2, 8) and areas(e) == [150.0, 1050.0]


def test_10_overlapping_rects_split():
    e = Entities()
    rect(e, 0, 0, 30, 20)
    rect(e, 15, 10, 45, 30)
    assert areas(e) == [150.0, 450.0, 450.0]


def test_11_box():
    e = box()
    assert counts(e) == (6, 12) and volume(e) == 12000.0


def test_12_half_top_up():
    e = box()
    e.add_line(P(15, 0, 10), P(15, 30, 10))
    e.pushpull(face_at(e, 5, 15, 10), 5)
    assert counts(e) == (8, 18) and volume(e) == 14250.0
    assert areas(e) == [150.0, 300.0, 450.0, 450.0, 475.0, 475.0, 750.0, 1200.0]


def test_13_half_top_down():
    e = box()
    e.add_line(P(15, 0, 10), P(15, 30, 10))
    e.pushpull(face_at(e, 5, 15, 10), -4)
    assert counts(e) == (8, 18) and volume(e) == 10200.0


def test_14_half_pushed_through():
    e = box()
    e.add_line(P(15, 0, 10), P(15, 30, 10))
    e.pushpull(face_at(e, 5, 15, 10), -10)
    assert counts(e) == (6, 12) and volume(e) == 7500.0


def test_15_pocket():
    e = box()
    rect(e, 10, 10, 25, 20, 10)
    e.pushpull(face_at(e, 17, 15, 10), -4)
    assert counts(e) == (11, 24) and volume(e) == 11400.0


def test_16_hole():
    e = box()
    rect(e, 10, 10, 25, 20, 10)
    e.pushpull(face_at(e, 17, 15, 10), -10)
    assert counts(e) == (10, 24) and volume(e) == 10500.0


def test_17_boss():
    e = box()
    rect(e, 10, 10, 25, 20, 10)
    e.pushpull(face_at(e, 17, 15, 10), 6)
    assert counts(e) == (11, 24) and volume(e) == 12900.0


def test_18_front_part_out():
    e = box()
    e.add_line(P(15, 0, 0), P(15, 0, 10))
    e.pushpull(face_at(e, 5, 0, 5), 5)
    assert counts(e) == (8, 18) and volume(e) == 12750.0


def test_19_corner_rect_down():
    e = box()
    rect(e, 0, 0, 15, 10, 10)
    e.pushpull(face_at(e, 5, 5, 10), -4)
    assert counts(e) == (9, 21) and volume(e) == 11400.0


def test_20_erase_vertical_edge():
    e = box()
    e.erase_edges([edge_at(e, 0, 0, 5)])
    assert counts(e) == (4, 11)


def test_21_erase_top_face():
    e = box()
    e.erase_faces([face_at(e, 20, 15, 10)])
    assert counts(e) == (5, 12)


def test_box_up_again_is_one_box():
    e = box()
    e.pushpull(face_at(e, 20, 15, 10), 5)
    assert counts(e) == (6, 12) and volume(e) == 18000.0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as ex:
                fails += 1
                print("FAIL", name, type(ex).__name__, ex)
    print("failures:", fails)
