"""Snapping to the drawing axes, as in SketchUp: the axes behave like edges."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import ops  # noqa: E402
from volura.geom import Plane  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def view():
    from volura.app import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    v = w.view
    v.cam.width, v.cam.height = 900, 700
    v.cam.set_view("iso")
    v.cam.target = np.array([0.0, 0.0, 0.0])
    v.cam.distance = 600.0
    yield v
    w.doc.dirty = False
    w.close()


def screen(view, world, off=(0.0, 0.0)):
    s, _ = view.cam.project(np.array(world, dtype=float))
    return float(s[0][0] + off[0]), float(s[0][1] + off[1])


@pytest.mark.parametrize("px", (3, 6, 8))
def test_near_the_red_axis_snaps_onto_it(view, px):
    inf = view.engine.infer(*screen(view, [120, 0, 0], (0, px)), anchor=np.array([50.0, 80.0, 30.0]))
    assert inf.kind == "axisline" and inf.axis == 0
    assert abs(inf.point[1]) < 1e-9 and abs(inf.point[2]) < 1e-9
    assert inf.label == "On Red Axis"


def test_parallel_to_green_ends_exactly_on_the_red_axis(view):
    inf = view.engine.infer(*screen(view, [50, 0, 0], (4, 4)), anchor=np.array([50.0, 80.0, 0.0]))
    assert inf.kind == "axisline" and inf.axis == 0
    assert np.allclose(inf.point, [50.0, 0.0, 0.0])


def test_axis_wins_over_a_face_under_the_cursor(view):
    doc = view.doc
    ops.group_from_manifold(doc, mf.Manifold.cube((200, 60, 10)).translate((0, -30, -10)), "#f2f2f2", "Plate")
    doc.notify()
    view.engine.invalidate()
    inf = view.engine.infer(*screen(view, [120, 0, 0], (0, 4)))
    assert inf.kind == "axisline" and inf.axis == 0


def test_far_from_the_axis_no_snap(view):
    inf = view.engine.infer(*screen(view, [120, 0, 0], (0, 30)))
    assert inf.kind != "axisline"


def test_axis_outside_the_drawing_plane_is_ignored(view):
    plane = Plane(np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0]))
    inf = view.engine.infer(*screen(view, [120, 0, 0], (0, 3)), anchor=np.array([0.0, 50.0, 20.0]),
                            plane=plane)
    assert inf.kind != "axisline"
    assert abs(inf.point[2] - 20.0) < 1e-6
