"""The Wall Thickness probe: the rays of volura/wall_probe and Tools > Wall Thickness.

The rays run without Qt; the tool runs the window without a display (Qt offscreen
platform), like tests/test_guides.py, and is clicked through the mouse handlers of the
viewport.
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import ops, wall_probe  # noqa: E402
from volura.geom import Plane, mesh_arrays  # noqa: E402
from volura.model import ROOT  # noqa: E402
from volura.tools import fmt_wall  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


def box(lx=20.0, ly=10.0, lz=3.0):
    """A closed box from the origin to (lx, ly, lz), as the rays read a mesh."""
    v = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                  [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]], dtype=float)
    loops = ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3])
    return v, np.array([[q[0], q[i], q[i + 1]] for q in loops for i in (1, 2)], dtype=np.int64)


# ------------------------------------------------------------------ the rays

def test_ray_hits_lists_every_triangle_the_ray_crosses():
    v, t = box()
    # through the top face into the material and out again through the bottom
    assert wall_probe.ray_hits(v, t, (10.0, 3.0, 8.0), (0, 0, -1.0)) == pytest.approx([5.0, 8.0])
    # beside the box nothing is crossed, and towards the sky the floor is no hit either
    assert not len(wall_probe.ray_hits(v, t, (30.0, 3.0, 8.0), (0, 0, -1.0)))
    assert not len(wall_probe.ray_hits(v, t, (10.0, 3.0, 3.0), (0, 0, 1.0)))


def test_rays_of_no_mesh_and_of_no_direction_hit_nothing():
    v, t = box()
    assert not len(wall_probe.ray_hits(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                                       (0, 0, 0), (0, 0, -1)))
    assert not len(wall_probe.ray_hits(v, t, (10.0, 5.0, 3.0), (0.0, 0.0, 0.0)))
    assert not np.isfinite(wall_probe.first_hits(v, t, (60.0, 5.0, 3.0),
                                                 np.array([[0.0, 0.0, -1.0]]))[0])


def test_the_distance_of_points_to_a_triangle():
    """The nearest point of a triangle is over the face, beside an edge or at a corner."""
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    t = np.array([[0, 1, 2]])
    points = [[2, 2, 3],        # over the face: straight down
              [6, 6, 0],        # over the face, past the hypotenuse: to the edge
              [-1, -1, 0],      # beside the edge from a to b, before a: to the corner a
              [0, -3, 4],       # below a corner
              [1, 1, 0],        # a point of the triangle itself
              [12, 0, 0]]       # past the corner b, along its edge
    want = [3.0, math.sqrt(2.0), math.sqrt(2.0), 5.0, 0.0, 2.0]
    assert wall_probe.point_mesh_distance(v, t, points) == pytest.approx(want)
    # one point on its own reads as one distance, also as a row of a table
    assert wall_probe.point_mesh_distance(v, t, [2, 2, 3]) == pytest.approx([3.0])
    assert wall_probe.point_mesh_distance(v, t, np.array([[[2, 2, 3], [1, 1, 0]]])).shape == (2,)


def test_the_distance_of_a_mesh_of_no_triangles_and_of_no_points():
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    assert wall_probe.point_mesh_distance(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                                         [[1, 2, 3]]) == pytest.approx([np.inf])
    assert not len(wall_probe.point_mesh_distance(v, np.array([[0, 1, 2]]), []))


def test_a_triangle_of_no_area_is_no_obstacle_and_the_blocks_agree(monkeypatch):
    """A triangle with no surface has no distance to keep: the search may not divide by zero
    on it, and a round of a few triangles answers the same as one of many."""
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]], float)
    t = np.array([[0, 1, 2], [3, 4, 5], [0, 0, 1]])
    points = [[2, 2, 3], [6, 6, 0], [12, 0, 0]]
    want = wall_probe.point_mesh_distance(v, t, points)
    assert np.all(np.isfinite(want)), "a triangle of no surface does not hide the real ones"
    monkeypatch.setattr(wall_probe, "CELLS", 1)
    assert wall_probe.point_mesh_distance(v, t, points) == pytest.approx(want)


def test_the_wall_straight_through_a_box():
    """The test box: 20 mm along x, 10 along y, 3 along z."""
    v, t = box()
    for p, n, want in (((10.0, 5.0, 3.0), (0, 0, 1), 3.0),        # the top face
                       ((20.0, 5.0, 1.5), (1, 0, 0), 20.0),       # a face across x
                       ((10.0, 0.0, 1.5), (0, -1, 0), 10.0)):     # a face across y
        distance, end = wall_probe.thickness(v, t, p, n)["along_normal"]
        assert distance == pytest.approx(want), f"the wall under {p}"
        assert np.linalg.norm(np.asarray(end) - np.asarray(p)) == pytest.approx(want, abs=1e-6)


def test_the_exit_lies_on_the_far_side_of_the_wall():
    v, t = box()
    distance, end = wall_probe.thickness(v, t, (10.0, 5.0, 3.0), (0, 0, 1))["along_normal"]
    assert np.allclose(end, (10.0, 5.0, 0.0)), "the exit lies on the bottom face"
    assert distance == pytest.approx(3.0)


def test_the_wall_where_the_material_is_thinnest():
    """Straight through a plate of 20 mm you measure 20 mm; the wall itself is 3 mm."""
    v, t = box()
    answer = wall_probe.thickness(v, t, (10.0, 5.0, 3.0), (0, 0, 1))
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01), "over a flat wall they are equal"
    assert np.allclose(answer["local"][1], (10.0, 5.0, 1.5), atol=0.01), \
        "the ball stands in the middle of the wall"

    answer = wall_probe.thickness(v, t, (20.0, 5.0, 1.5), (1, 0, 0))
    assert answer["along_normal"][0] == pytest.approx(20.0)
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01), "the ball fits in the wall"
    diameter, centre = answer["local"]
    assert np.linalg.norm(np.asarray(centre) - (20.0, 5.0, 1.5)) == pytest.approx(diameter / 2.0), \
        "the centre lies as deep as the ball is wide, on the line into the part"

    answer = wall_probe.thickness(v, t, (10.0, 0.0, 1.5), (0, -1, 0))
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01)


def test_the_ball_of_a_solid_cube_fills_it():
    v, t = box(lx=20.0, ly=20.0, lz=20.0)
    answer = wall_probe.thickness(v, t, (10.0, 10.0, 20.0), (0, 0, 1))
    assert answer["along_normal"][0] == pytest.approx(20.0)
    assert answer["local"][0] == pytest.approx(20.0, abs=0.01)


def test_the_wall_of_a_tube_is_its_wall_and_not_its_diameter():
    tube = (mf.Manifold.cylinder(20.0, 10.0, 10.0, 96, False)
            - mf.Manifold.cylinder(20.0, 8.0, 8.0, 96, False))
    v, t = mesh_arrays(tube)
    # the middle of an outside facet, halfway up the tube: 2 mm of wall, and a 96-gon is a
    # hair narrower than the circle it stands in
    p = np.array([10.0 * math.cos(math.pi / 96), 0.0, 10.0])
    answer = wall_probe.thickness(v, t, p, (1, 0, 0))
    flat = 2.0 * math.cos(math.pi / 96)
    assert answer["along_normal"][0] == pytest.approx(flat, abs=0.02)
    assert answer["local"][0] == pytest.approx(flat, abs=0.03), "not the 20 mm it stands over"
    diameter, centre = answer["local"]
    assert np.linalg.norm(np.asarray(centre) - p) == pytest.approx(diameter / 2.0)


def test_the_wall_of_a_sphere_is_its_diameter():
    """A mesh of flat triangles is never exactly round: the ball that touches the facets is
    a hair smaller than the sphere they stand on."""
    v, t = mesh_arrays(mf.Manifold.sphere(5.0, 96))
    for face in (t[0], t[len(t) // 3], t[-1]):
        p = v[face].mean(axis=0)            # the middle of a triangle lies on the mesh
        answer = wall_probe.thickness(v, t, p, p / np.linalg.norm(p))
        assert answer["along_normal"][0] == pytest.approx(10.0, abs=0.05)
        assert answer["local"][0] == pytest.approx(10.0, abs=0.25)


def test_a_mesh_that_is_not_closed_has_no_wall():
    one = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float), np.array([[0, 1, 2]])
    answers = [wall_probe.thickness(*one, (3.0, 3.0, 0.0), (0, 0, 1)),
               wall_probe.thickness(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                                    (0, 0, 0), (0, 0, 1)),
               wall_probe.thickness(*one, (3.0, 3.0, 0.0), (0.0, 0.0, 0.0))]
    for answer in answers:
        assert answer["along_normal"] is None and answer["local"] is None


def test_a_hairline_wall_is_still_measured():
    """A wall thinner than a nozzle is still a wall: the start of the rays inside the
    material may not eat a part of it."""
    v, t = box(lx=8.0, ly=8.0, lz=0.05)
    answer = wall_probe.thickness(v, t, (4.0, 4.0, 0.05), (0, 0, 1))
    assert answer["along_normal"][0] == pytest.approx(0.05, abs=1e-6)
    assert answer["local"][0] == pytest.approx(0.05, abs=1e-3)


def test_a_normal_of_any_length_and_a_triangle_of_no_area_change_nothing():
    v, t = box()
    flat = np.vstack([v, [[10.0, 5.0, 3.0]] * 3])          # a triangle with no surface
    answer = wall_probe.thickness(flat, np.vstack([t, [[len(v), len(v) + 1, len(v) + 2]]]),
                                 (10.0, 5.0, 3.0), (0.0, 0.0, 4.0))
    assert answer["along_normal"][0] == pytest.approx(3.0, abs=1e-6)
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("cells", (wall_probe.CELLS, 3))
def test_the_triangles_go_by_blocks(monkeypatch, cells):
    """A round holds CELLS triangles: what does not fit in one round is no different."""
    monkeypatch.setattr(wall_probe, "CELLS", cells)
    v, t = box()
    answer = wall_probe.thickness(v, t, (20.0, 5.0, 1.5), (1, 0, 0))
    assert answer["along_normal"][0] == pytest.approx(20.0)
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01)


def test_a_mesh_of_many_triangles_answers_the_same(monkeypatch):
    """A part triangulated a thousand times over still measures 3 mm of wall."""
    v, t = mesh_arrays(mf.Manifold.cube([20.0, 10.0, 3.0], True).refine(20))
    monkeypatch.setattr(wall_probe, "CELLS", 1000)
    assert len(t) > 3 * wall_probe.CELLS, "many triangles indeed"
    answer = wall_probe.thickness(v, t, (0.0, 0.0, 1.5), (0, 0, 1))
    assert answer["along_normal"][0] == pytest.approx(3.0, abs=1e-6)
    assert answer["local"][0] == pytest.approx(3.0, abs=0.01)


# ------------------------------------------------------------------ the tool

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    # whatever an earlier test of this process left behind: the probe measures the faces
    # of the model as they are drawn, with no section plane and no coloured view in the way
    for name in ("section_on", "show_hidden", "show_overhangs", "show_thin_walls",
                 "show_tight_fits"):
        setattr(w.settings, name, False)
    w.view.cam.width, w.view.cam.height = 800, 600
    w.view.cam.set_view("iso")
    yield w
    w.doc.dirty = False
    w.close()


def box_group(win, size=(20.0, 20.0, 3.0)):
    """A closed box as a group, standing on the origin; returns the group."""
    lx, ly, lz = size
    win.doc.root.add_shape(GROUND, [np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], float)])
    f = next(iter(win.doc.root.F))
    win.doc.root.pushpull(f, -lz)
    group = ops.make_group(win.doc, [("f", ROOT, fid) for fid in win.doc.root.F])
    win.view.zoom_extents()
    return group


def click(win, p):
    """A left click at a world point, through the mouse handlers of the viewport."""
    s, ok = win.view.cam.project(np.asarray(p, dtype=float)[None])
    assert ok[0], "point is off screen"
    pos = QPointF(float(s[0][0]), float(s[0][1]))
    win.view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos, Qt.NoButton,
                                        Qt.NoButton, Qt.NoModifier))
    win.view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, pos, Qt.LeftButton,
                                         Qt.LeftButton, Qt.NoModifier))


def probe(win, p):
    """Click and wait for the rays: a big part is measured in the background."""
    tool = win.view.tools["wall_probe"]
    click(win, p)
    if tool.job is not None:
        tool.job.wait()
    return tool


def test_the_menu_calls_the_probe(win):
    assert win.a_wall_probe.text() == "&Wall Thickness"
    win.a_wall_probe.setChecked(False)
    win.a_wall_probe.trigger()
    assert win.view.tool.name == "wall_probe"
    assert win.hint.text() == "Click a face of a part to measure its wall thickness."


def test_a_click_on_the_top_of_a_box_reads_its_thickness(win):
    box_group(win)
    win.set_tool("wall_probe")
    tool = probe(win, [10.0, 10.0, 3.0])
    assert tool.wall[0] == pytest.approx(3.0, abs=1e-6)
    assert tool.local[0] == pytest.approx(3.0, abs=0.01)
    assert np.allclose(tool.wall[1], (10.0, 10.0, 0.0), atol=1e-6), "the line ends under the box"
    assert win.hint.text() == \
        f"Straight through {fmt_wall(3.0)} · wall {fmt_wall(3.0)}.", win.hint.text()


def test_a_click_on_the_side_of_a_plate_says_the_wall_too(win):
    """Straight through the side the material is 20 mm thick; the plate itself is 3 mm."""
    box_group(win)
    win.set_tool("wall_probe")
    tool = probe(win, [20.0, 10.0, 1.5])
    assert tool.wall[0] == pytest.approx(20.0, abs=1e-6)
    assert tool.local[0] == pytest.approx(3.0, abs=0.01)
    assert "20.00 mm" in win.hint.text() and "3.00 mm" in win.hint.text(), win.hint.text()


def test_a_click_in_empty_space_asks_for_a_part(win):
    box_group(win)
    win.set_tool("wall_probe")
    tool = probe(win, [200.0, 200.0, 0.0])
    assert win.hint.text() == "Click a face of a part"
    assert tool.wall is None and tool.local is None and tool.point is None


def test_an_answer_of_a_click_that_was_already_replaced_is_dropped(win):
    """A slow probe may not answer a click the user already left behind."""
    box_group(win)
    win.set_tool("wall_probe")
    tool = win.view.tools["wall_probe"]
    click(win, [10.0, 10.0, 3.0])
    first = tool.job
    click(win, [200.0, 200.0, 0.0])
    assert tool.point is None and tool.wall is None
    assert first is not None and first.wait(), "the first probe is still on its way"
    assert tool.wall is None, "the answer of the replaced click is dropped"
    assert win.hint.text() == "Click a face of a part", win.hint.text()


def test_a_click_on_a_flat_sketch_has_no_wall(win):
    """A face on its own is a drawing and not a part: the rays meet nothing."""
    win.doc.root.add_shape(GROUND, [np.array([[0, 0], [30, 0], [30, 30], [0, 30]], float)])
    win.view.zoom_extents()
    win.set_tool("wall_probe")
    tool = probe(win, [15.0, 15.0, 0.0])
    assert win.hint.text() == "Click a face of a part"
    assert tool.wall is None and tool.local is None


def test_the_last_answer_stays_on_screen(win):
    box_group(win)
    win.set_tool("wall_probe")
    tool = probe(win, [10.0, 10.0, 3.0])
    s, _ok = win.view.cam.project(np.array([[120.0, 120.0, 0.0]]))
    pos = QPointF(float(s[0][0]), float(s[0][1]))
    win.view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, pos, Qt.NoButton,
                                        Qt.NoButton, Qt.NoModifier))
    assert tool.wall is not None and tool.wall[0] == pytest.approx(3.0, abs=1e-6), \
        "the measured wall stays until the next click"


def test_esc_leaves_the_probe(win):
    box_group(win)
    win.set_tool("wall_probe")
    probe(win, [10.0, 10.0, 3.0])
    win.view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert win.view.tool.name == "select"
    assert not win.tool_actions["wall_probe"].isChecked()
