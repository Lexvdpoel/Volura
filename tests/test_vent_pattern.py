"""Vent pattern: a grille or a speaker cover of holes inside any closed contour.

`boxgen.vent_pattern` stands the holes on a triangular grid, a step `size + web` from the
next hole, so the web of material between two neighbours is `web` wide everywhere:
hexagons of `size` across the flats (the tightest way to fill a panel) or circles of `size`
diameter, on the same grid. The grid is centred on the bounding box of the contour, so a
rectangle or a circle is symmetric on every side, and a hole only stays when it lies whole
inside the contour shrunk by `margin`. On a rectangle of 100 × 60 mm with a hole size of 6,
a web of 2 and a margin of 3 that is 85 holes, all inside [3, 97] × [3, 57] and with their
centres 8 mm apart; a circle of radius 50 holds 109 and a square of 5 × 5 none at all.

These tests (Qt offscreen) measure that pattern and check in the window: one selected closed
contour gets its holes on the cut layer in one undo step, with its source number, while
nothing selected, an open path or a contour too small leaves the undo stack untouched with a
clear text in the status bar.

Run with:  python -m pytest tests/test_vent_pattern.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import boxgen  # noqa: E402
from volura.sheettools import circle_loop  # noqa: E402
from volura.vector2d import next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

RECT = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]])
SIZE, WEB, MARGIN = 6.0, 2.0, 3.0
HOLES = 85


def pattern(outline=RECT, **options):
    o = {"shape": "hex", "size": SIZE, "web": WEB, "margin": MARGIN}
    o.update(options)
    return boxgen.vent_pattern(outline, **o)


def centres(holes):
    return np.array([q.mean(axis=0) for q in holes])


def nearest_centre_distance(holes):
    """The distance from every centre to its nearest neighbour."""
    c = centres(holes)
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def as_sets(holes):
    """The holes as a sorted list of rounded points, so two patterns can be compared."""
    return sorted(np.round(q, 9).tolist() for q in holes)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def contour(win, pts, closed=True, op="cut"):
    """One path on a layer of the worksheet by operation, selected."""
    sheet = win.doc.sheet
    p = sheet.add_path(pts, closed=closed, layer=sheet.default_layer(op),
                       src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.set_selection([p])
    return p


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


# ------------------------------------------------------------------ the pattern

def test_the_rectangle_holds_the_expected_number_of_holes():
    holes = pattern()
    assert len(holes) == HOLES


def test_every_hole_lies_inside_the_rectangle_shrunk_by_the_margin():
    for q in pattern():
        assert q[:, 0].min() >= MARGIN - 1e-9 and q[:, 0].max() <= 100.0 - MARGIN + 1e-9
        assert q[:, 1].min() >= MARGIN - 1e-9 and q[:, 1].max() <= 60.0 - MARGIN + 1e-9


def test_the_web_between_two_neighbours_is_web_wide():
    # centres a size + web apart: the flats of two hexagons stand 2 mm from each other
    assert nearest_centre_distance(pattern()) == pytest.approx(SIZE + WEB, abs=1e-6)


def test_a_hexagon_is_a_regular_hexagon_of_size_across_the_flats():
    for q in pattern():
        assert len(q) == 6
        assert float(q[:, 1].max() - q[:, 1].min()) == pytest.approx(SIZE, abs=1e-9)


def test_a_circle_outline_is_filled_the_same_way():
    assert len(pattern(circle_loop((0.0, 0.0), 50.0))) == 109


def test_round_holes_are_circles_of_the_chosen_diameter():
    holes = pattern(shape="round")
    assert len(holes) == HOLES
    for q in holes:
        c = q.mean(axis=0)
        assert np.max(np.hypot(*(q - c).T)) == pytest.approx(0.5 * SIZE, abs=1e-9)
        assert np.min(np.hypot(*(q - c).T)) == pytest.approx(0.5 * SIZE, abs=1e-9)


def test_a_contour_that_is_too_small_holds_no_hole():
    small = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    assert pattern(small) == []
    assert pattern(RECT, margin=40.0) == []


def test_a_repeated_last_point_gives_the_same_holes():
    closed = np.vstack([RECT, RECT[:1]])       # imports often close a contour by hand
    assert as_sets(pattern(closed)) == as_sets(pattern())


def test_impossible_input_raises():
    with pytest.raises(ValueError):
        pattern(shape="square")
    with pytest.raises(ValueError):
        pattern(size=0.0)
    with pytest.raises(ValueError):
        pattern(web=0.0)


# ------------------------------------------------------------------ in the window

def test_the_action_fills_the_selected_contour(win, monkeypatch):
    accept_dialog(monkeypatch)
    outline = contour(win, RECT)
    added = win.vent_pattern_dialog()
    assert added is not None and len(added) == HOLES
    assert all(p.closed for p in added), "a hole is a closed contour"
    cut = win.doc.sheet.default_layer("cut")
    assert all(p.layer == cut for p in added)
    assert all(int(p.src) == int(outline.src) for p in added), "the pattern moves with the panel"
    assert win.hint.text() == "85 holes added."
    assert all(p in win.sheet_view.selected_paths() for p in added)


def test_the_holes_come_and_go_in_one_undo_step(win, monkeypatch):
    accept_dialog(monkeypatch)
    contour(win, RECT)
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    win.vent_pattern_dialog()
    assert len(win.doc.undo_stack) == steps + 1, "one undo step for all holes together"
    assert len(win.doc.sheet.paths) == len(before) + HOLES
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    win.settings.vent_shape, win.settings.vent_size = "round", 9.0
    contour(win, RECT)
    seen = {}

    def fill(dlg):
        shape = dlg.findChild(QComboBox)
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [str(shape.currentData())] + [round(sp.value(), 2) for sp in spins]
        shape.setCurrentIndex(0)
        spins[0].setValue(4.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.vent_pattern_dialog()
    assert seen["start"] == ["round", 9.0, 2.0, 3.0]
    assert win.settings.vent_shape == "hex"
    assert win.settings.vent_size == pytest.approx(4.0)
    s = app_module.Settings()
    assert s.vent_shape == "hex" and s.vent_size == pytest.approx(4.0)


def test_a_contour_too_small_for_a_hole_leaves_nothing(win, monkeypatch):
    accept_dialog(monkeypatch)
    contour(win, [[0, 0], [5, 0], [5, 5], [0, 5]])
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.vent_pattern_dialog() is None
    assert len(win.doc.undo_stack) == steps, "nothing changed, so no undo step"
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "No hole fits inside this contour."


def test_an_open_path_is_no_contour(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing to fill: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    contour(win, [[0, 0], [40, 0], [40, 20]], closed=False)
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.vent_pattern_dialog() is None
    assert len(win.doc.undo_stack) == steps
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select one closed contour on the worksheet first."


def test_nothing_selected_is_a_clear_message(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing selected: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    win.set_mode("2d")
    win.sheet_view.clear_selection()
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.vent_pattern_dialog() is None
    assert len(win.doc.undo_stack) == steps
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select one closed contour on the worksheet first."


def test_apply_vent_pattern_needs_no_dialog(win):
    outline = contour(win, RECT)
    made = win.apply_vent_pattern({"shape": "hex", "size": SIZE, "web": WEB, "margin": MARGIN})
    assert made is not None and len(made) == HOLES
    assert all(int(p.src) == int(outline.src) for p in made)
    assert win.hint.text() == "85 holes added."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
