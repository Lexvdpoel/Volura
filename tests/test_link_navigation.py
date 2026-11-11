"""Jumping between a linked outline on the worksheet and the part it belongs to.

A linked outline is marked on the worksheet with a small chain badge at its top-left
corner, and the two sides hold each the way to the other: the context menu of the
worksheet has "Go to Linked Part", which selects the part in 3D and frames it, and the
context menu of the model has "Go to Linked Outline", which selects the paths of that
link on the worksheet and centres the view on them. A link of a face goes to the face
itself, not to the whole part.

These tests (Qt offscreen) link a plate and walk between the two sides.

Run with:  python -m pytest tests/test_link_navigation.py
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


def to_sheet(win, g, face=None):
    """The selection on the worksheet, as "To Laser or Mill" puts it there."""
    win.set_mode("3d")
    win.doc.set_selection([("f", g.id, face)] if face is not None else [("g", g.id, g.id)])
    cut = win.doc.sheet.default_layer("cut")
    added = win.send_selection_to_sheet(layer=cut)
    assert added, "the part went to the worksheet"
    return int(added[0].src)


def top_face(win, g):
    """The face on top of the block."""
    return max((f for f, F in g.ents.F.items() if F.normal[2] > 0.99 and F.loops),
               key=lambda f: float(g.ents.V[g.ents.F[f].loops[0][0]][2]))


def paths_of(win, src):
    return [p for p in win.doc.sheet.paths if int(p.src) == src]


def labels(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


def action(menu, text):
    return next(a for a in menu.actions() if a.text().replace("&", "") == text)


def test_a_linked_outline_carries_a_badge_at_its_corner(win):
    g = plate(win)
    src = to_sheet(win, g)
    v = win.sheet_view
    badges = v._link_badges()
    assert len(badges) == 1 and int(badges[0][1].src) == src
    pts = np.vstack([p.pts for p in paths_of(win, src)])
    corner = v.to_screen((float(pts[:, 0].min()), float(pts[:, 1].max())))
    assert badges[0][0].x() == pytest.approx(corner.x(), abs=1e-6)
    assert badges[0][0].y() == pytest.approx(corner.y(), abs=1e-6), "the top-left corner"
    own = win.doc.sheet.add_path(
        np.array([[200.0, 200.0], [240.0, 200.0], [240.0, 230.0]], dtype=float),
        closed=True, layer="0")
    win.doc.notify()
    assert len(v._link_badges()) == 1, "a contour that is not linked adds no badge"
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    win.unlink()
    assert v._link_badges() == [], "an outline without a link carries no badge"
    assert own in win.doc.sheet.paths


def test_an_outline_that_is_not_linked_has_no_way_to_a_part(win):
    src = to_sheet(win, plate(win))
    win.doc.links = []
    v = win.sheet_view
    v.set_selection(paths_of(win, src))
    assert "Go to Linked Part" not in labels(v.build_context_menu())


def test_the_worksheet_menu_of_a_linked_outline_goes_to_the_part(win):
    g = plate(win)
    src = to_sheet(win, g)
    v = win.sheet_view
    drawn = win.doc.sheet.add_path(
        np.array([[200.0, 200.0], [240.0, 200.0], [240.0, 230.0]], dtype=float),
        closed=True, layer="0")
    v.set_selection([drawn])
    assert "Go to Linked Part" not in labels(v.build_context_menu())
    v.set_selection(paths_of(win, src) + [drawn])
    assert "Go to Linked Part" not in labels(v.build_context_menu()), "two outlines are no part"
    v.set_selection(paths_of(win, src))
    action(v.build_context_menu(), "Go to Linked Part").trigger()
    assert win.mode == "3d"
    assert win.doc.selection == [("g", g.id, g.id)]


def test_a_linked_face_goes_to_that_face(win):
    g = plate(win, size=(40.0, 20.0, 20.0))
    fid = top_face(win, g)
    src = to_sheet(win, g, face=fid)
    assert win.doc.links[0].kind == "face"
    v = win.sheet_view
    v.set_selection(paths_of(win, src))
    action(v.build_context_menu(), "Go to Linked Part").trigger()
    assert win.mode == "3d"
    assert win.doc.selection == [("f", g.id, fid)]


def test_the_model_menu_of_a_linked_part_goes_to_the_outline(win):
    g = plate(win)
    src = to_sheet(win, g)
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    assert "Go to Linked Outline" in labels(win.build_context_menu())
    action(win.build_context_menu(), "Go to Linked Outline").trigger()
    assert win.mode == "2d"
    assert {int(p.src) for p in win.sheet_view.selected_paths()} == {src}


def test_an_unlinked_part_has_no_way_to_an_outline(win):
    g = plate(win)
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    assert "Go to Linked Outline" not in labels(win.build_context_menu())


def test_the_linked_face_of_a_solid_gives_the_way_back(win):
    g = plate(win, size=(40.0, 20.0, 20.0))
    fid = top_face(win, g)
    to_sheet(win, g, face=fid)
    win.set_mode("3d")
    win.doc.set_selection([])
    win._context_face = (g.id, fid)
    try:
        assert "Go to Linked Outline" in labels(win.build_context_menu())
    finally:
        win._context_face = None


def test_painting_the_worksheet_with_a_link(win):
    to_sheet(win, plate(win))
    v = win.sheet_view
    v.zoom_extents()
    assert not v.grab().isNull()
