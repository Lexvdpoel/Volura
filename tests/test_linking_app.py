"""The live link between a part in the model and its outline on the worksheet.

A plate that goes "To Laser or Mill" (or comes back with "Parts to 3D Model") stays linked
to the outline of its source element on the sheet. Only the shape travels, when the user
switches between 3D and 2D or chooses Tools > Update Links: each side keeps its own place,
so arranging the sheet never moves the model and moving the plate in 3D never moves the
sheet. A plate can also change on the sheet and be rebuilt in the model; a face only goes
from the model to the sheet.

These tests (Qt offscreen) make a plate, link it, change one side and look at the other.

Run with:  python -m pytest tests/test_linking_app.py
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
from volura import fileio, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402

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


def plate(win, size=(40.0, 20.0, 3.0), at=(10.0, 20.0, 0.0)):
    """A plate in the model, selected."""
    man = mf.Manifold.cube(list(size)).translate(at)
    g = ops.group_from_manifold(win.doc, man, "#3b7dd8", "Plate")
    win.doc.set_selection([("g", g.id, g.id)])
    win.doc.notify()
    return g


def to_sheet(win, g):
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    cut = win.doc.sheet.default_layer("cut")
    added = win.send_selection_to_sheet(layer=cut)
    assert added, "the plate went to the worksheet"
    return int(added[0].src)


def outline(win, src):
    pts = np.vstack([p.pts for p in win.doc.sheet.paths if int(p.src) == src and p.closed])
    return pts.min(axis=0), pts.max(axis=0)


def size_of(win, g):
    lo, hi = win.doc.group(g.id).bbox()
    return np.asarray(hi) - np.asarray(lo), np.asarray(lo)


def reshape(win, g, size, at=(10.0, 20.0, 0.0)):
    """The plate gets another shape in the model (what push/pull would do), in 3D."""
    win.set_mode("3d")
    win.doc.group(g.id).ents = Entities.from_manifold(mf.Manifold.cube(list(size)).translate(at))
    win.doc.notify()


def move_on_sheet(win, src, d):
    for p in win.doc.sheet.paths:
        if int(p.src) == src:
            p.pts = np.asarray(p.pts, dtype=float) + np.asarray(d, dtype=float)
    win.doc.notify()


def test_to_laser_makes_a_plate_link(win):
    g = plate(win)
    src = to_sheet(win, g)
    assert len(win.doc.links) == 1
    link = win.doc.links[0]
    assert (link.kind, int(link.group), int(link.src)) == ("plate", g.id, src)
    assert link.thickness == pytest.approx(3.0)


def test_a_wider_plate_widens_its_outline_where_it_lies_on_the_sheet(win):
    g = plate(win)
    src = to_sheet(win, g)
    move_on_sheet(win, src, (100.0, 50.0))           # arranged on the sheet
    lo0, hi0 = outline(win, src)
    reshape(win, g, (60.0, 20.0, 3.0))
    win.set_mode("2d")
    lo, hi = outline(win, src)
    span = sorted(hi - lo)
    assert span == pytest.approx([20.0, 60.0], abs=1e-3), "the outline has the new shape"
    centre0 = 0.5 * (lo0 + hi0)
    assert np.linalg.norm(0.5 * (lo + hi) - centre0) < 15.0, "it stays where it was arranged"
    assert win.doc.links[0].shape is not None


def test_moving_the_outline_on_the_sheet_leaves_the_model_alone(win):
    g = plate(win)
    src = to_sheet(win, g)
    before, lo_before = size_of(win, g)
    move_on_sheet(win, src, (70.0, -30.0))
    win.set_mode("3d")
    after, lo_after = size_of(win, g)
    assert after == pytest.approx(before) and lo_after == pytest.approx(lo_before)


def test_a_new_shape_on_the_sheet_rebuilds_the_plate_at_its_place_in_3d(win):
    g = plate(win)
    src = to_sheet(win, g)
    lo, hi = outline(win, src)
    # the outline gets 10 mm longer on the sheet
    for p in win.doc.sheet.paths:
        if int(p.src) == src:
            q = np.asarray(p.pts, dtype=float).copy()
            q[q[:, 0] > lo[0] + 1.0, 0] += 10.0
            p.pts = q
    win.doc.notify()
    win.set_mode("3d")
    size, low = size_of(win, g)
    assert sorted(size[:2]) == pytest.approx([20.0, 50.0], abs=1e-3)
    assert size[2] == pytest.approx(3.0, abs=1e-6), "the same thickness"
    assert low[2] == pytest.approx(0.0, abs=1e-6), "still on the bed"
    assert abs(low[0] - 10.0) < 10.5 and abs(low[1] - 20.0) < 10.5, "where the plate was"


def test_a_sync_is_one_undo_step(win):
    g = plate(win)
    src = to_sheet(win, g)
    lo0, hi0 = outline(win, src)
    reshape(win, g, (60.0, 20.0, 3.0))
    win.doc.checkpoint()
    steps = len(win.doc.undo_stack)
    win.set_mode("2d")
    assert len(win.doc.undo_stack) == steps + 1
    win.doc.undo()
    lo, hi = outline(win, src)
    assert lo == pytest.approx(lo0) and hi == pytest.approx(hi0)


def test_nothing_changed_is_no_undo_step(win):
    g = plate(win)
    to_sheet(win, g)
    steps = len(win.doc.undo_stack)
    win.update_links()
    assert len(win.doc.undo_stack) == steps
    assert win.hint.text() == "Nothing to update."


def test_the_links_are_saved_with_the_file(win, tmp_path):
    g = plate(win)
    src = to_sheet(win, g)
    path = str(tmp_path / "linked.volura")
    fileio.save_document(win.doc, path)
    win.doc.links = []
    fileio.load_document(win.doc, path)
    assert len(win.doc.links) == 1 and int(win.doc.links[0].src) == src


def test_unlink_removes_the_link_of_the_selected_plate(win):
    g = plate(win)
    to_sheet(win, g)
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    win.unlink()
    assert win.doc.links == []
