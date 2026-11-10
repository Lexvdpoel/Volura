"""A live link of a face keeps its face when the solid is turned over.

A link of a *face* remembers the plane of that face in world coordinates, so it can read
the model again from the same side. Turning the solid puts another face in that plane: the
40 x 20 top of a box comes to lie where the 20 x 10 side was, and the link would send that
top to the worksheet. The link follows its shape instead of its old place, and the Rotate
and Move tools take the plane of the linked face with the solid, so the outline on the
worksheet stays the outline that went there.

These tests (Qt offscreen) link the side face of a box, turn the box and look at the sheet.

Run with:  python -m pytest tests/test_linking_rotate.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def box(win, size=(40.0, 20.0, 10.0)):
    """A block in the model, at the origin."""
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube(list(size)), "#3b7dd8", "Block")
    win.doc.notify()
    return g


def side_face(win, g):
    """The +x side face of the block: the 20 x 10 one."""
    return next(f for f, F in g.ents.F.items() if F.normal[0] > 0.99)


def to_sheet(win, g, fid):
    """That face on the worksheet, as "To Laser or Mill" puts it there."""
    win.set_mode("3d")
    win.doc.set_selection([("f", g.id, fid)])
    cut = win.doc.sheet.default_layer("cut")
    added = win.send_selection_to_sheet(layer=cut)
    assert added, "the face went to the worksheet"
    win.set_mode("3d")                            # that command ends on the worksheet
    return int(added[0].src)


def outline(win, src):
    """The size of the outline on the worksheet, smallest span first."""
    pts = np.vstack([p.pts for p in win.doc.sheet.paths if int(p.src) == src and p.closed])
    return np.sort(pts.max(axis=0) - pts.min(axis=0))


def turn(win, g, deg, axis):
    """The block turned about the origin, with the Rotate tool; the matrix it used."""
    tool = win.view.tools["rotate"]
    tool.reset()
    tool.items = [("g", g.id, g.id)]
    tool.center = np.zeros(3)
    tool.plane = Plane(np.zeros(3), np.asarray(axis, dtype=float))
    tool.ref = np.array([1.0, 0.0, 0.0])
    M = tool.matrix(float(deg))
    tool.apply(float(deg))
    return M


def reshape(win, g, size, M=None):
    """The block gets another shape in the model (what push/pull would do), where it lies."""
    win.set_mode("3d")
    man = mf.Manifold.cube(list(size))
    if M is not None:
        man = man.transform(np.ascontiguousarray(np.c_[np.asarray(M, float)[:3, :3],
                                                       np.asarray(M, float)[:3, 3]]))
    win.doc.group(g.id).ents = Entities.from_manifold(man)
    win.doc.notify()


def test_the_side_face_goes_to_the_sheet_as_a_face_link(win):
    g = box(win)
    src = to_sheet(win, g, side_face(win, g))
    assert len(win.doc.links) == 1
    assert win.doc.links[0].kind == "face"
    assert outline(win, src) == pytest.approx([10.0, 20.0], abs=1e-3)


def test_turning_about_z_keeps_the_side_face_on_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, side_face(win, g))
    turn(win, g, 90.0, (0.0, 0.0, 1.0))
    win.set_mode("2d")
    assert outline(win, src) == pytest.approx([10.0, 20.0], abs=1e-3), "still the side face"


def test_turning_about_y_does_not_put_the_top_on_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, side_face(win, g))
    turn(win, g, 90.0, (0.0, 1.0, 0.0))
    win.set_mode("2d")
    assert outline(win, src) == pytest.approx([10.0, 20.0], abs=1e-3), "not the 40 x 20 top"


def test_the_rotate_tool_takes_the_plane_of_the_link_along(win):
    g = box(win)
    to_sheet(win, g, side_face(win, g))
    turn(win, g, 90.0, (0.0, 0.0, 1.0))
    frame = np.asarray(win.doc.links[0].frame, dtype=float)
    assert frame[3:6] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9), "the face now looks along +y"
    assert frame[1] == pytest.approx(40.0, abs=1e-6), "and lies in the plane y = 40"


def test_a_link_follows_a_solid_that_was_turned_without_the_tool(win):
    g = box(win)
    src = to_sheet(win, g, side_face(win, g))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    g.ents.transform(R, np.zeros(3))       # turned over without a link in mind
    win.doc.notify()
    win.set_mode("2d")
    assert outline(win, src) == pytest.approx([10.0, 20.0], abs=1e-3), "the shape found its face"


def test_a_longer_side_face_still_comes_to_the_sheet_after_the_turn(win):
    g = box(win)
    src = to_sheet(win, g, side_face(win, g))
    M = turn(win, g, 90.0, (0.0, 0.0, 1.0))
    reshape(win, g, (40.0, 30.0, 10.0), M)
    win.set_mode("2d")
    assert outline(win, src) == pytest.approx([10.0, 30.0], abs=1e-3), "the sheet follows"
