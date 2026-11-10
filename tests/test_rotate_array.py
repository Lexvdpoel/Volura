"""Circular arrays with the Rotate tool, like the linear arrays of the Move tool.

After a Ctrl-copy with Rotate, typing "5x" in the Measurements box makes 5 copies at
the same angle step, and "/3" divides the angle into 3 equal steps (like SketchUp).
"""
import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import ops  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.inference import Inference  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


class FakeView:
    """A view that remembers what a tool drew."""

    def __init__(self):
        self.lines = []
        self.cam = SimpleNamespace(pixel_size=lambda p: 1.0)

    def draw_lines(self, pts, color, width, stipple=False, on_top=False):
        self.lines.append((np.asarray(pts, dtype=float), bool(stipple)))

    def draw_strip(self, pts, color, width, on_top=False):
        pass

    def draw_points(self, pts, color, size):
        pass

    def add_label(self, point, text):
        pass

    def draw_inference(self, inf):
        pass

    def segment_to(self, point):
        """The single line the tool draws to the point under the cursor."""
        for pts, stip in self.lines:
            if len(pts) == 2 and np.allclose(pts[1], point):
                return stip
        return None


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def spoke(win, x=30.0):
    """A 10 mm cube whose centre sits x mm from the origin, as its own group."""
    man = mf.Manifold.cube((10, 10, 10)).translate((x - 5.0, -5.0, -5.0))
    return ops.group_from_manifold(win.doc, man, "#ffffff")


def start_rotation(tool, g):
    tool.reset()
    tool.items = [("g", g.id, g.id)]
    tool.center = np.zeros(3)
    tool.plane = Plane(np.zeros(3), np.array([0.0, 0, 1]))
    tool.ref = np.array([1.0, 0, 0])
    tool.copy = True


def centre(g):
    return np.asarray(g.ents.bbox(), float).mean(axis=0)


def angles(doc):
    """The angle around Z of every group centre, sorted."""
    out = []
    for g in doc.groups:
        c = centre(g)
        out.append(round(math.degrees(math.atan2(c[1], c[0])) % 360.0, 3))
    return sorted(out)


def test_rotate_copy_then_5x_makes_a_circle_of_spokes(win):
    g = spoke(win)
    tool = win.view.tools["rotate"]
    start_rotation(tool, g)
    tool.apply(60.0)
    assert len(win.doc.groups) == 2, "one copy after the rotate"

    tool.vcb_enter("5x")
    assert len(win.doc.groups) == 6, "original + 5 copies"
    vols = {round(g.ents.to_manifold().volume(), 3) for g in win.doc.groups}
    assert vols == {1000.0}, "every copy keeps the same volume"
    for g in win.doc.groups:                        # all centres on one circle
        c = centre(g)
        assert np.linalg.norm(c[:2]) == pytest.approx(30.0, abs=1e-6)
        assert c[2] == pytest.approx(0.0, abs=1e-6)
    assert angles(win.doc) == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


def test_rotate_array_x_forms_and_divide(win):
    g = spoke(win)
    tool = win.view.tools["rotate"]
    start_rotation(tool, g)
    tool.apply(60.0)
    tool.vcb_enter("x5")
    assert angles(win.doc) == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]

    win.doc.undo()                                  # one undo removes the whole array
    assert len(win.doc.groups) == 1, "only the original is left"


def test_rotate_array_divides_the_angle(win):
    g = spoke(win)
    tool = win.view.tools["rotate"]
    start_rotation(tool, g)
    tool.apply(90.0)
    tool.vcb_enter("/3")
    assert len(win.doc.groups) == 4, "original + 3 copies"
    assert angles(win.doc) == [0.0, 30.0, 60.0, 90.0]

    win.doc.undo()
    assert len(win.doc.groups) == 1, "one undo leaves only the original"


def test_the_line_from_the_centre_to_the_cursor_while_the_angle_starts(win):
    g = spoke(win)
    tool = win.view.tools["rotate"]
    tool.reset()
    tool.items = [("g", g.id, g.id)]
    tool.inf = Inference("endpoint", np.array([25.0, 8.0, 0.0]))
    view = FakeView()
    tool.draw(view)
    assert view.segment_to(tool.inf.point) is None, "no centre yet, so no line"

    tool.center = np.zeros(3)
    tool.plane = Plane(np.zeros(3), np.array([0.0, 0, 1]))
    view = FakeView()
    tool.draw(view)
    assert view.segment_to(tool.inf.point) is True, "a stippled line to the start point"

    tool.ref = np.array([1.0, 0, 0])
    view = FakeView()
    tool.draw(view)
    assert view.segment_to(tool.inf.point) is False, "the angle line, as before"


def test_parse_array_helper():
    from volura.tools import parse_array
    assert parse_array("5x") == (5, False)
    assert parse_array("x5") == (5, False)
    assert parse_array("5 *") == (5, False)
    assert parse_array("/5") == (5, True)
    assert parse_array("  / 3 ") == (3, True)
    assert parse_array("45") is None
    assert parse_array("2,5") is None
