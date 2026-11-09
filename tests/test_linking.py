"""The live link between a part in the model and its outline on the worksheet.

`volura/linking.py` (without Qt) compares shapes without their placement (`fit_rigid`,
`same_shape`), puts a plane of the model in flat coordinates (`face_frame`, `to_plane`,
`from_plane`), recognises a plate in a solid (`plate_of`) and builds one back
(`plate_solid`), and remembers what a link last synced (`Link`).

Run with:  python -m pytest tests
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402

from volura import linking  # noqa: E402
from volura.vector2d import polygon_area  # noqa: E402


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def circle(cx, cy, r, n=64):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def tee():
    """An outline without any symmetry: mirrored it is a different part."""
    return np.array([[0, 0], [30, 0], [30, 12], [18, 12], [18, 26], [6, 26], [6, 12],
                     [0, 12]], dtype=float)


def turn(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def turn3(axis, angle):
    """Rotation in 3D about a unit axis (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(k, k)


def section(loops):
    return mf.CrossSection([np.ascontiguousarray(np.asarray(l, dtype=float))
                            for l in loops], mf.FillRule.EvenOdd)


def plate(loops=None, thickness=3.0):
    return mf.Manifold.extrude(section(loops or [rect(40, 20)]), thickness)


def placed(manifold, R, t):
    return manifold.transform(np.ascontiguousarray(np.column_stack([R, t]), dtype=np.float64))


# ---------------------------------------------------------------- fit_rigid

def test_fit_rigid_recovers_rotation_and_translation():
    src = [rect(40, 20), rect(8, 6, 12, 7)]
    R, t = turn(math.radians(37.0)), np.array([13.0, -4.0])
    dst = [l @ R.T + t for l in src]
    M, rms = linking.fit_rigid(src, dst)
    assert rms == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(M[:2, :2], R, atol=1e-9)
    assert np.allclose(M[:2, 2], t, atol=1e-9)
    assert np.allclose(M[2], [0.0, 0.0, 1.0])


def test_fit_rigid_mirror_only_when_it_is_allowed():
    src = [tee()]
    dst = [src[0] * np.array([-1.0, 1.0])]
    M, rms = linking.fit_rigid(src, dst, allow_mirror=True)
    assert rms == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.det(M[:2, :2]) == pytest.approx(-1.0)
    assert linking.fit_rigid(src, dst, allow_mirror=False)[1] > 1.0
    assert linking.same_shape(src, dst)
    assert not linking.same_shape(src, dst, allow_mirror=False)


def test_fit_rigid_ignores_the_start_point_and_the_direction():
    src = [tee()]
    for q in (np.roll(src[0], -3, axis=0), np.roll(src[0][::-1], 5, axis=0)):
        M, rms = linking.fit_rigid(src, [q])
        assert rms == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(M, np.eye(3), atol=1e-9)


def test_fit_rigid_ignores_a_repeated_closing_point():
    src = [rect(40, 20)]
    q = rect(40, 20) + np.array([10.0, 5.0])
    M, rms = linking.fit_rigid([np.vstack([src[0], src[0][0]])], [np.vstack([q, q[0]])])
    assert rms == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(M[:2, 2], [10.0, 5.0], atol=1e-9)


def test_fit_rigid_needs_the_same_number_of_points():
    assert linking.fit_rigid([rect(40, 20)], [circle(0, 0, 20)]) is None
    assert linking.fit_rigid([rect(40, 20), rect(8, 6)], [rect(40, 20)]) is None
    assert linking.fit_rigid([], []) is None


# ---------------------------------------------------------------- same_shape

def test_same_shape_ignores_placement():
    src = [tee(), circle(40, 40, 6)]
    R, t = turn(math.radians(120.0)), np.array([-25.0, 60.0])
    assert linking.same_shape(src, [l @ R.T + t for l in src])


def test_same_shape_sees_a_vertex_moved_half_a_millimetre():
    src = [rect(40, 20)]
    dst = [src[0].copy()]
    dst[0][0] += 0.5
    assert not linking.same_shape(src, dst)
    assert linking.same_shape(src, dst, tol=1.0)


def test_same_shape_needs_the_same_loops():
    assert not linking.same_shape([rect(40, 20)], [rect(40, 20), rect(8, 6)])
    assert not linking.same_shape([rect(40, 20)], [rect(40, 20), rect(8, 6, 12, 7)])


# ------------------------------------------------------------------- planes

def test_face_frame_is_a_right_handed_frame_in_the_plane():
    n = np.array([0.3, -0.5, 0.81])
    o, u, v, nn = linking.face_frame(n, (12.0, -3.0, 5.0))
    assert np.linalg.norm(nn) == pytest.approx(1.0)
    assert np.allclose(o, [12.0, -3.0, 5.0])
    for a in (u, v):
        assert np.linalg.norm(a) == pytest.approx(1.0)
        assert float(a @ nn) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.cross(nn, u), v)
    assert np.allclose(linking.face_frame(-n, (12.0, -3.0, 5.0))[3], -nn)


def test_face_frame_axis_is_stable_and_follows_a_reference():
    _o, u, _v, _n = linking.face_frame((0.0, 0.0, 1.0), (0, 0, 0))
    assert np.allclose(u, [1.0, 0.0, 0.0])
    _o, u, _v, _n = linking.face_frame((1.0, 0.0, 0.0), (0, 0, 0))       # normal along X
    assert np.allclose(u, [0.0, 1.0, 0.0])
    _o, u, _v, _n = linking.face_frame((0.0, 0.0, 1.0), (0, 0, 0), ref=(1.0, 2.0, 9.0))
    assert np.allclose(u, [1.0, 2.0, 0.0] / np.linalg.norm([1.0, 2.0, 0.0]))
    _o, u, _v, _n = linking.face_frame((0.0, 0.0, 1.0), (0, 0, 0), ref=(0.0, 0.0, -1.0))
    assert np.allclose(u, [1.0, 0.0, 0.0])                               # ref along n


def test_to_plane_and_from_plane_round_trip():
    frame = linking.face_frame((0.4, 0.2, -0.9), (5.0, -8.0, 2.0))
    o, u, v, _n = frame
    ab = np.array([[0.0, 0.0], [40.0, 20.0], [-3.5, 12.0], [7.0, -7.0]])
    pts = o + ab[:, :1] * u + ab[:, 1:2] * v
    assert np.allclose(linking.to_plane(pts, frame), ab)
    assert np.allclose(linking.from_plane(ab, frame), pts)
    off = pts + 6.0 * frame[3]
    assert np.allclose(linking.from_plane(linking.to_plane(off, frame), frame), pts)


# ------------------------------------------------------------------- plates

def test_plate_of_a_box_is_3_mm_thick():
    plate = linking.plate_of(mf.Manifold.cube((40, 20, 3)))
    assert plate is not None
    assert plate["thickness"] == pytest.approx(3.0, abs=1e-6)
    loops = plate["loops"]
    assert len(loops) == 1 and len(loops[0]) == 4
    assert polygon_area(loops[0]) == pytest.approx(800.0)          # outer, counter-clockwise
    o, u, v, n = plate["frame"]
    assert np.linalg.norm(n) == pytest.approx(1.0)
    assert np.allclose(linking.from_plane(np.zeros((1, 2)), plate["frame"])[0], o)


def test_plate_of_a_plate_with_a_hole():
    solid = plate([rect(40, 20), rect(8, 6, 12, 7)])
    plate_ = linking.plate_of(solid)
    assert plate_ is not None
    assert plate_["thickness"] == pytest.approx(3.0, abs=1e-6)
    outer, hole = plate_["loops"]
    assert polygon_area(outer) == pytest.approx(800.0)
    assert polygon_area(hole) == pytest.approx(-48.0)              # holes run the other way
    assert sum(polygon_area(l) for l in plate_["loops"]) == pytest.approx(752.0)
    assert linking.same_shape(plate_["loops"], [rect(40, 20), rect(8, 6, 12, 7)])


def test_plate_of_a_plate_rotated_and_moved():
    loops = [rect(40, 20), rect(8, 6, 12, 7)]
    R = turn3((1.0, 0.0, 0.0), math.radians(30.0))
    solid = placed(plate(loops), R, np.array([12.0, -30.0, 4.0]))
    plate_ = linking.plate_of(solid)
    assert plate_ is not None
    assert plate_["thickness"] == pytest.approx(3.0, abs=1e-4)
    assert sum(polygon_area(l) for l in plate_["loops"]) == pytest.approx(752.0, rel=1e-4)
    assert linking.same_shape(plate_["loops"], loops)
    axis = float(np.dot(plate_["frame"][3], R @ np.array([0.0, 0.0, 1.0])))
    assert abs(axis) == pytest.approx(1.0, abs=1e-6)


def test_moving_or_turning_a_plate_keeps_its_shape():
    a = linking.plate_of(plate([tee()]))
    b = linking.plate_of(plate([tee()]).rotate((0.0, 0.0, 25.0)).translate((100.0, -30.0, 5.0)))
    assert b is not None
    assert linking.same_shape(a["loops"], b["loops"])
    assert a["thickness"] == pytest.approx(b["thickness"])


def test_a_plate_stays_a_plate_when_it_is_thicker():
    thin = linking.plate_of(plate([tee()], 3.0))
    thick = linking.plate_of(plate([tee()], 4.0))
    assert linking.same_shape(thin["loops"], thick["loops"])       # only the thickness changed
    assert thick["thickness"] == pytest.approx(4.0)


def test_a_solid_with_a_boss_is_not_a_plate():
    bottom = mf.Manifold.cube((20, 20, 5))
    boss = mf.Manifold.cube((10, 10, 5)).translate((5.0, 5.0, 5.0))
    solid = bottom + boss
    assert not solid.is_empty()
    assert linking.plate_of(solid) is None


def test_a_sphere_is_not_a_plate():
    assert linking.plate_of(mf.Manifold.sphere(12.0, 16)) is None


def test_a_wedge_is_not_a_plate():
    box = mf.Manifold.cube((40, 20, 3))
    wedge = box - box.rotate((1.0, 2.0, 0.0))         # an axis that is not one of its own
    assert not wedge.is_empty()
    assert linking.plate_of(wedge) is None


def test_a_cylinder_is_a_plate():
    """The caps are triangulated however manifold likes; the plate is still recognised."""
    plate_ = linking.plate_of(mf.Manifold.cylinder(3.0, 10.0, 10.0, 48))
    assert plate_ is not None
    assert plate_["thickness"] == pytest.approx(3.0, abs=1e-6)
    assert len(plate_["loops"]) == 1
    assert polygon_area(plate_["loops"][0]) == pytest.approx(math.pi * 100.0, rel=1e-2)


# ------------------------------------------------------------ plate_solid

FRAME_FLAT = linking.face_frame((0.0, 0.0, 1.0), (5.0, -2.0, 9.0))
FRAME_TILTED = linking.face_frame((0.4, -0.7, 0.6), (1.0, 2.0, -3.0))
FRAME_SIDE = linking.face_frame((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), ref=(0.0, 3.0, 4.0))


@pytest.mark.parametrize("frame", (FRAME_FLAT, FRAME_TILTED, FRAME_SIDE))
def test_plate_solid_round_trip(frame):
    loops = [tee(), rect(4.0, 4.0, 10.0, 4.0)]        # the second loop is a hole
    solid = linking.plate_solid(loops, frame, 3.0)
    outer, hole = abs(polygon_area(loops[0])), abs(polygon_area(loops[1]))
    assert solid.volume() == pytest.approx((outer - hole) * 3.0, rel=1e-6)
    verts = np.asarray(solid.to_mesh().vert_properties, dtype=float)[:, :3]
    along = verts @ frame[3]
    assert float(along.max() - along.min()) == pytest.approx(3.0, abs=1e-6)
    back = linking.plate_of(solid)
    assert back is not None
    assert back["thickness"] == pytest.approx(3.0, abs=1e-4)     # vertices are stored single
    assert linking.same_shape(back["loops"], loops)
    assert sum(abs(polygon_area(l)) for l in back["loops"]) == pytest.approx(outer + hole,
                                                                             rel=1e-6)


def test_plate_solid_of_nothing_is_empty():
    assert linking.plate_solid([], FRAME_FLAT, 3.0).is_empty()
    assert linking.plate_of(None) is None


# ------------------------------------------------------------------- Link

def test_link_goes_to_a_file_and_back():
    link = linking.Link(src=3, group=7, kind="plate",
                        shape=[rect(40, 20), rect(8, 6, 12, 7)], thickness=3.0)
    d = json.loads(json.dumps(link.to_dict()))
    assert d["src"] == 3 and d["group"] == 7 and d["kind"] == "plate"
    assert d["thickness"] == pytest.approx(3.0)
    assert len(d["shape"]) == 2 and len(d["shape"][0]) == 4
    back = linking.Link.from_dict(d)
    assert (back.src, back.group, back.kind) == (link.src, link.group, link.kind)
    assert back.thickness == pytest.approx(link.thickness)
    assert len(back.shape) == 2
    assert np.allclose(back.shape[0], link.shape[0])
    assert np.allclose(back.shape[1], link.shape[1])


def test_link_of_a_face_has_no_thickness():
    link = linking.Link(src=1, group=2, kind="face", shape=[rect(30, 20)])
    d = json.loads(json.dumps(link.to_dict()))
    assert d["thickness"] is None
    back = linking.Link.from_dict(d)
    assert back.kind == "face" and back.thickness is None
    assert linking.Link.from_dict({}).kind == "plate"
    assert linking.Link.from_dict({"kind": "wobbly"}).kind == "plate"
    assert linking.Link.from_dict({"kind": "face", "thickness": 2}).thickness == pytest.approx(2.0)
