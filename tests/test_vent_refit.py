"""A vent pattern is made again when the outline it was cut into changes shape.

`apply_vent_pattern` remembers the pattern of a contour (`Document.vents`): its options,
the outline it was made for and the holes it gave. When that outline is another shape the
next time the window looks (`refit_vents`, called on a switch between 2D and 3D and by
Update Links), the holes of the pattern go and the pattern is made again for the new
outline, with the same options; a hole you drew yourself in the same contour stays. On a
plate that is linked to the model, the plate gets the new grille too.

A rectangle of 100 x 60 with hexagons of 6, a web of 2 and a margin of 3 holds 85 holes;
the same rectangle shrunk to 60 x 60 holds what `boxgen.vent_pattern` gives for it.

Run with:  python -m pytest tests/test_vent_refit.py
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
from volura import boxgen, fileio, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.vector2d import next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

OPTIONS = {"shape": "hex", "size": 6.0, "web": 2.0, "margin": 3.0}
RECT = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]])
SQUARE = np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]])


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


def grille(win, pts=RECT):
    """A contour on the cut layer with a vent pattern in it; returns the contour."""
    sheet = win.doc.sheet
    p = sheet.add_path(pts, closed=True, layer=sheet.default_layer("cut"),
                       src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.set_selection([p])
    made = win.apply_vent_pattern(dict(OPTIONS))
    assert made, "the pattern made holes"
    return p


def holes_of(win, outline):
    return [p for p in win.doc.sheet.paths
            if int(p.src) == int(outline.src) and p is not outline and p.closed]


def expected(pts):
    return len(boxgen.vent_pattern(pts, **OPTIONS))


def inside(win, outline, pts):
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    for h in holes_of(win, outline):
        q = np.asarray(h.pts, dtype=float)
        assert (q.min(axis=0) >= lo - 1e-6).all() and (q.max(axis=0) <= hi + 1e-6).all()


def test_the_pattern_is_remembered(win):
    p = grille(win)
    assert len(win.doc.vents) == 1 and int(win.doc.vents[0].src) == int(p.src)
    assert len(holes_of(win, p)) == expected(RECT) == 85


def test_a_smaller_outline_gets_a_smaller_pattern(win):
    p = grille(win)
    p.pts = SQUARE.copy()
    assert win.refit_vents() == 1
    assert len(holes_of(win, p)) == expected(SQUARE) < 85
    inside(win, p, SQUARE)
    assert win.hint.text() == "1 vent pattern fitted to its new outline."


def test_the_same_outline_changes_nothing(win):
    grille(win)
    steps = len(win.doc.undo_stack)
    assert win.refit_vents() == 0
    assert len(win.doc.undo_stack) == steps


def test_a_hole_you_drew_yourself_stays(win):
    p = grille(win)
    own = win.doc.sheet.add_path([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]], closed=True,
                                 layer=p.layer, src=p.src)
    p.pts = SQUARE.copy()
    win.refit_vents()
    assert any(h is own for h in win.doc.sheet.paths)
    assert len(holes_of(win, p)) == expected(SQUARE) + 1


def test_a_refit_is_one_undo_step(win):
    p = grille(win)
    p.pts = SQUARE.copy()
    steps = len(win.doc.undo_stack)
    win.refit_vents()
    assert len(win.doc.undo_stack) == steps + 1
    win.doc.undo()                      # undo gives new path objects: the outline counts too
    assert sum(1 for q in win.doc.sheet.paths if int(q.src) == int(p.src)) == 85 + 1


def test_a_switch_to_3d_refits(win):
    p = grille(win)
    p.pts = SQUARE.copy()
    win.set_mode("3d")
    assert len(holes_of(win, p)) == expected(SQUARE)


def test_a_new_pattern_in_the_same_contour_replaces_the_old_one(win):
    p = grille(win)
    win.sheet_view.set_selection([p])
    win.apply_vent_pattern(dict(OPTIONS, size=10.0))
    assert len(win.doc.vents) == 1
    assert len(holes_of(win, p)) == len(boxgen.vent_pattern(RECT, **dict(OPTIONS, size=10.0)))


def test_the_vents_are_saved_with_the_file(win, tmp_path):
    p = grille(win)
    path = str(tmp_path / "vent.volura")
    fileio.save_document(win.doc, path)
    win.doc.vents = []
    fileio.load_document(win.doc, path)
    assert len(win.doc.vents) == 1 and int(win.doc.vents[0].src) == int(p.src)


def genus(win, g):
    return win.doc.group(g.id).ents.to_manifold().genus()


def test_a_linked_plate_gets_the_new_grille(win):
    man = mf.Manifold.cube([100.0, 60.0, 3.0])
    g = ops.group_from_manifold(win.doc, man, "#3b7dd8", "Plate")
    win.doc.notify()
    win.set_mode("3d")
    win.doc.set_selection([("g", g.id, g.id)])
    added = win.send_selection_to_sheet(layer=win.doc.sheet.default_layer("cut"))
    src = int(added[0].src)
    win.set_mode("2d")
    outline = next(p for p in win.doc.sheet.paths if int(p.src) == src and p.closed)
    win.sheet_view.set_selection([outline])
    win.apply_vent_pattern(dict(OPTIONS))
    win.set_mode("3d")
    assert genus(win, g) == 85, "the plate has the holes"
    # the plate gets narrower in the model: the sheet follows and so does the grille
    win.doc.group(g.id).ents = Entities.from_manifold(mf.Manifold.cube([60.0, 60.0, 3.0]))
    win.doc.notify()
    win.set_mode("2d")
    outline = max((p for p in win.doc.sheet.paths if int(p.src) == src and p.closed),
                  key=lambda p: np.ptp(np.asarray(p.pts), axis=0).prod())
    n = len(holes_of(win, outline))
    assert n == expected(SQUARE)
    win.set_mode("3d")
    assert genus(win, g) == n, "the plate has the new grille"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
