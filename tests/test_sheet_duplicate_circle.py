"""Duplicate in a circle (polar array) on the 2D worksheet.

`polar_angles` gives the angles of the copies, `SheetView.duplicate_circle` places the
copies of every selected path in one undo step, around the centre of the selection, the
origin or a point of choice. With "Rotate the copies" off the copies only travel along
the circle and keep their own orientation.

Run with:  python -m pytest tests/test_sheet_duplicate_circle.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura.model import Document  # noqa: E402
from volura.sheettools import circle_loop, polar_angles  # noqa: E402
from volura.sheetview import DuplicateCircleSheetDialog, SheetView  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def square(doc, x=50.0, half=2.0):
    """A small square whose centre stands at (x, 0)."""
    doc.sheet.add_path([[x - half, -half], [x + half, -half],
                        [x + half, half], [x - half, half]], closed=True)
    return doc.sheet.paths[-1]


def sheet_view(doc):
    v = SheetView(doc)
    v.set_selection(list(doc.sheet.paths))
    return v


def centre(path):
    return np.asarray(path.pts, dtype=float).mean(axis=0)


def directions(path):
    """The direction of every edge, as a unit vector, ignoring where the path stands."""
    pts = np.asarray(path.pts, dtype=float)
    segs = np.roll(pts, -1, axis=0) - pts
    segs = segs[np.hypot(segs[:, 0], segs[:, 1]) > 1e-9]
    return sorted(tuple(np.round(s / np.hypot(*s), 6)) for s in segs)


# ------------------------------------------------------------------ the angles

def test_polar_angles_of_a_full_circle():
    assert [round(a, 9) for a in polar_angles(6)] == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


def test_polar_angles_of_an_arc_end_on_the_angle():
    assert [round(a, 9) for a in polar_angles(4, 90.0)] == [0.0, 30.0, 60.0, 90.0]


def test_polar_angles_need_at_least_two_copies():
    for count in (1, 0):
        with pytest.raises(ValueError):
            polar_angles(count)


# ------------------------------------------------------------------ the copies

def test_duplicate_circle_around_the_origin():
    doc = Document()
    square(doc)
    v = sheet_view(doc)
    made = v.duplicate_circle(4, 360.0, (0.0, 0.0))
    assert len(made) == 3
    assert [tuple(np.round(centre(p), 9)) for p in made] == [(0.0, 50.0), (-50.0, 0.0),
                                                             (0.0, -50.0)]
    centres = sorted(tuple(np.round(centre(p), 9)) for p in doc.sheet.paths)
    assert centres == [(-50.0, 0.0), (0.0, -50.0), (0.0, 50.0), (50.0, 0.0)]
    assert len(doc.sheet.paths) == 4


def test_duplicate_circle_selects_the_original_with_the_copies():
    doc = Document()
    p = square(doc)
    v = sheet_view(doc)
    made = v.duplicate_circle(3, 360.0, (0.0, 0.0))
    assert v.selected_paths() == [p] + made


def test_duplicate_circle_is_one_undo_step():
    doc = Document()
    square(doc)
    v = sheet_view(doc)
    steps = len(doc.undo_stack)
    v.duplicate_circle(4, 360.0, (0.0, 0.0))
    assert len(doc.undo_stack) == steps + 1
    doc.undo()
    assert len(doc.sheet.paths) == 1


def test_duplicate_circle_copies_the_circle_meta():
    doc = Document()
    doc.sheet.add_path(circle_loop([50.0, 0.0], 5.0), closed=True, filled=True,
                       meta={"kind": "circle", "center": [50.0, 0.0], "radius": 5.0})
    v = sheet_view(doc)
    v.duplicate_circle(4, 360.0, (0.0, 0.0))
    centres = sorted(tuple(np.round(p.meta["center"], 9)) for p in doc.sheet.paths)
    assert centres == [(-50.0, 0.0), (0.0, -50.0), (0.0, 50.0), (50.0, 0.0)]
    assert all(p.meta["radius"] == pytest.approx(5.0) for p in doc.sheet.paths)


def test_duplicate_circle_without_rotating_only_moves_the_copies():
    doc = Document()
    p = square(doc)
    v = sheet_view(doc)
    made = v.duplicate_circle(4, 360.0, (0.0, 0.0), rotate=False)
    assert len(made) == 3
    centres = sorted(tuple(np.round(centre(q), 9)) for q in made)
    assert centres == [(-50.0, 0.0), (0.0, -50.0), (0.0, 50.0)]
    assert all(directions(q) == directions(p) for q in made), "the edges keep their direction"


def test_duplicate_circle_around_the_centre_of_the_selection():
    doc = Document()
    square(doc, x=50.0)
    square(doc, x=50.0 + 40.0)
    v = sheet_view(doc)
    made = v.duplicate_circle(2, 180.0)
    centres = sorted(tuple(np.round(centre(p), 9)) for p in made)
    assert centres == [(50.0, 0.0), (90.0, 0.0)], "halfway turns swap the two squares"


def test_duplicate_circle_without_a_selection_is_not_an_undo_step():
    doc = Document()
    square(doc)
    v = SheetView(doc)
    msgs = []
    v.status_changed.connect(msgs.append)
    steps = len(doc.undo_stack)
    assert v.duplicate_circle(4, 360.0, (0.0, 0.0)) == []
    assert len(doc.undo_stack) == steps
    assert len(doc.sheet.paths) == 1
    assert msgs, "the worksheet says what went wrong"


# ------------------------------------------------------------------ the dialog

def test_the_dialog_remembers_what_it_offers():
    doc = Document()
    square(doc)
    v = SheetView(doc)
    dlg = DuplicateCircleSheetDialog(v)
    assert dlg.copies.value() == 6
    assert float(dlg.angle.value()) == pytest.approx(360.0)
    assert dlg.rotate.isChecked()
    assert dlg.values() == (6, 360.0, None, True)
    assert not dlg.px.isEnabled() and not dlg.py.isEnabled()
    dlg.copies.setValue(8)
    dlg.angle.setValue(180.0)
    dlg.rotate.setChecked(False)
    assert dlg.values()[:2] == (8, 180.0) and dlg.values()[3] is False


def test_the_dialog_asks_for_a_point_of_its_own():
    doc = Document()
    square(doc)
    v = SheetView(doc)
    dlg = DuplicateCircleSheetDialog(v)
    dlg.centre.setCurrentIndex(1)
    assert dlg.values()[2] == (0.0, 0.0)
    assert not dlg.px.isEnabled()
    dlg.centre.setCurrentIndex(2)
    assert dlg.px.isEnabled() and dlg.py.isEnabled()
    dlg.px.setValue(12.0)
    dlg.py.setValue(34.0)
    assert dlg.values()[2] == (12.0, 34.0)
