"""Holding tabs: small uncut bridges that keep a part in the sheet on the laser or mill.

`sheettools.holding_tabs` cuts `count` gaps of `width` millimetres evenly out of a closed
contour and returns the open pieces that are left to cut, in the direction of the contour.
The first tab is centred at 1/(2*count) of the perimeter, so a tab never sits on the start
point, and a gap that falls over a corner cuts the two segments exactly at the arc-length
positions. When there is no room for the tabs the contour comes back closed as it was.

In the window (Qt offscreen) a closed contour on a cutting layer is replaced by its open
pieces in one undo step, keeping its layer and source number; open paths and engravings are
skipped, and nothing selected or nothing to change leaves the undo stack untouched.

Run with:  python -m pytest tests/test_holding_tabs.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura.sheettools import holding_tabs  # noqa: E402
from volura.vector2d import next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

RECT = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])


def perimeter(pts):
    """Length of the closed loop `pts` (last point back to the first)."""
    q = np.asarray(pts, dtype=float)
    return float(np.sum(np.hypot(*(np.diff(np.vstack([q, q[:1]]), axis=0)).T)))


def piece_length(piece):
    return float(np.sum(np.hypot(*(np.diff(np.asarray(piece, dtype=float), axis=0)).T)))


def total_length(pieces):
    return sum(piece_length(p) for p in pieces)


def has_point(pieces, p, tol=1e-6):
    return any(np.all(np.abs(q - np.asarray(p, float)) <= tol, axis=1).any() for q in pieces)


# ------------------------------------------------------------------ the function

def test_the_pieces_add_up_to_the_perimeter_less_the_tabs():
    for count, width in ((1, 1.0), (4, 1.0), (7, 0.5)):
        pieces = holding_tabs(RECT, count, width)
        assert total_length(pieces) == pytest.approx(perimeter(RECT) - count * width, abs=1e-9)


def test_there_are_count_pieces_and_every_one_is_open():
    pieces = holding_tabs(RECT, 4, 1.0)
    assert len(pieces) == 4
    for q in pieces:
        assert len(q) >= 2
        assert not np.allclose(q[0], q[-1]), "a piece that is cut is open at both ends"


def test_no_tab_sits_on_the_start_point():
    pieces = holding_tabs(RECT, 4, 1.0)
    start = RECT[0]
    for q in pieces:
        assert not np.allclose(q[0], start) and not np.allclose(q[-1], start)
    assert has_point(pieces, start), "the start point stays inside a piece"


def test_a_gap_over_a_corner_cuts_both_segments():
    # a 30 mm perimeter with three tabs puts the middle tab centred on the corner (10, 5)
    pieces = holding_tabs(RECT, 3, 2.0)
    assert not has_point(pieces, (10.0, 5.0)), "the corner falls in the gap"
    assert has_point(pieces, (10.0, 4.0)) and has_point(pieces, (9.0, 5.0)), \
        "the two segments are cut where the gap crosses them"


def test_no_room_for_the_tabs_leaves_the_contour_closed():
    for count, width in ((20, 1.0), (0, 1.0), (4, 0.0), (4, -1.0)):
        pieces = holding_tabs(RECT, count, width)
        assert len(pieces) == 1 and np.array_equal(pieces[0], RECT)


def test_a_short_contour_can_be_left_alone():
    pieces = holding_tabs(RECT, 4, 1.0, skip_short=100.0)
    assert len(pieces) == 1 and np.array_equal(pieces[0], RECT)
    assert len(holding_tabs(RECT, 4, 1.0, skip_short=10.0)) == 4


# ------------------------------------------------------------------ in the window

@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


def contour(win, pts, op="cut", closed=True):
    """A path on a layer of the worksheet by operation, selected."""
    sheet = win.doc.sheet
    layer = sheet.default_layer(op) or sheet.free_layer_name(op)
    p = sheet.add_path(pts, closed=closed, layer=layer, src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.add_to_selection([p])
    return p


def test_a_rectangle_gets_its_tabs_in_one_undo_step(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.tab_count, win.settings.tab_width = 4, 1.0
    p = contour(win, [[0, 0], [60, 0], [60, 40], [0, 40]])
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    made = win.holding_tabs_dialog()
    assert len(win.doc.undo_stack) == steps + 1, "one undo step for all the pieces"
    assert made is not None and len(made) == 4
    assert all(not q.closed for q in made), "the pieces are open"
    assert all(q.layer == p.layer and int(q.src) == int(p.src) for q in made), \
        "same layer and source, so the part still moves and nests as one"
    assert win.hint.text() == "1 contours got 4 tabs each."
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_open_paths_and_engraving_are_skipped(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.tab_count, win.settings.tab_width = 4, 1.0
    rect = contour(win, [[0, 0], [40, 0], [40, 20], [0, 20]])
    open_line = contour(win, [[0, 30], [20, 30], [20, 40]], closed=False)
    engraved = contour(win, [[50, 0], [70, 0], [70, 20], [50, 20]], op="engrave")
    made = win.holding_tabs_dialog()
    assert made is not None and len(made) == 4
    assert open_line in win.doc.sheet.paths and engraved in win.doc.sheet.paths
    assert engraved.closed, "the engraving keeps its shape"
    assert "2 paths were skipped (open or not a cutting layer)." in win.hint.text()


def test_only_skipped_paths_changes_nothing(win, monkeypatch):
    accept_dialog(monkeypatch)
    contour(win, [[0, 30], [20, 30], [20, 40]], closed=False)
    contour(win, [[50, 0], [70, 0], [70, 20], [50, 20]], op="engrave")
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.holding_tabs_dialog() is None
    assert len(win.doc.undo_stack) == steps, "nothing changed, so no undo step"
    assert win.doc.sheet.paths == before


def test_nothing_selected_is_no_undo_step(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing selected: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    win.set_mode("2d")
    win.sheet_view.clear_selection()
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.holding_tabs_dialog() is None
    assert len(win.doc.undo_stack) == steps
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select the closed contours to tab first."


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    win.settings.tab_count, win.settings.tab_width = 6, 2.5
    contour(win, [[0, 0], [60, 0], [60, 40], [0, 40]])
    seen = {}

    def fill(dlg):
        seen["start"] = (dlg.findChild(QSpinBox).value(), round(dlg.findChild(
            QDoubleSpinBox).value(), 2))
        dlg.findChild(QSpinBox).setValue(3)
        dlg.findChild(QDoubleSpinBox).setValue(1.5)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.holding_tabs_dialog()
    assert seen["start"] == (6, 2.5)
    assert win.settings.tab_count == 3
    assert win.settings.tab_width == pytest.approx(1.5)
