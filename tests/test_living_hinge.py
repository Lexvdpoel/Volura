"""Living hinge: a field of straight slits that makes plywood bend on the laser.

`boxgen.living_hinge` gives the classic pattern. The slits stand in columns `pitch`
apart, starting `margin` from the left edge; in a column the slits are `slit` long with
a bridge of `gap` between them, and every second column is shifted by half the period
`slit + gap`, so the bridges of neighbouring columns never lie in a row. That zigzag of
material is what holds the panel together while it bends. On a rectangle of 60 × 40 mm
with the defaults (slits 15, bridges 3, pitch 2, margin 3) the inner band of 54 mm takes
28 columns; an even column takes two slits (3-18, 21-36), a shifted one three (3-9,
12-27, 30-37, short pieces at both margins): 70 slits, all inside the rectangle shrunk
by the margin, and no slit runs from the bottom margin to the top one.

These tests (Qt offscreen) check the pattern on that rectangle, the turned version and
the impossible sizes, and in the window: one selected rectangle gets its slits in one
undo step, anything else a clear text in the status bar.

Run with:  python -m pytest tests/test_living_hinge.py
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
from volura.vector2d import next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

RECT = (0.0, 0.0, 60.0, 40.0)
SLIT, GAP, PITCH, MARGIN = 15.0, 3.0, 2.0, 3.0
PERIOD = SLIT + GAP
COLUMNS = 28
SLITS = 14 * 2 + 14 * 3            # even columns two slits, shifted columns three


def pattern(rect=RECT, **options):
    o = {"slit": SLIT, "gap": GAP, "pitch": PITCH, "margin": MARGIN}
    o.update(options)
    return boxgen.living_hinge(*rect, **o)


def columns_of(slits):
    """The slits per column, in the order of the columns, and the x of every column."""
    xs = sorted({round(float(p[0, 0]), 6) for p in slits})
    return [[p for p in slits if abs(float(p[0, 0]) - x) < 1e-9] for x in xs], xs


def gap_centres(column):
    """The y in the middle of every bridge of a column (of slits running along Y)."""
    return [0.5 * (float(a[1, 1]) + float(b[0, 1])) for a, b in zip(column, column[1:])]


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    # start from the defaults, whatever an earlier, interrupted run left in the settings
    w.settings.hinge_slit, w.settings.hinge_gap = 15.0, 3.0
    w.settings.hinge_pitch, w.settings.hinge_margin = 2.0, 3.0
    w.settings.hinge_vertical = True
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def rectangle(win, rect=(0.0, 0.0, 60.0, 40.0)):
    """One closed rectangle on the worksheet, selected."""
    x0, y0, x1, y1 = rect
    sheet = win.doc.sheet
    p = sheet.add_path([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], closed=True,
                       layer=sheet.default_layer("cut"), src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.set_selection([p])
    return p


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


# ------------------------------------------------------------------ the pattern

def test_every_slit_lies_inside_the_rectangle_shrunk_by_the_margin():
    s = pattern()
    assert s
    for p in s:
        assert p.shape == (2, 2)
        assert p[:, 0].min() >= MARGIN - 1e-9 and p[:, 0].max() <= 60.0 - MARGIN + 1e-9
        assert p[:, 1].min() >= MARGIN - 1e-9 and p[:, 1].max() <= 40.0 - MARGIN + 1e-9


def test_all_slits_run_along_y():
    for p in pattern():
        assert p[0, 0] == pytest.approx(p[1, 0]), "the slit is a straight vertical line"
        assert p[1, 1] > p[0, 1]


def test_the_columns_stand_a_pitch_apart_from_the_left_margin():
    cols, xs = columns_of(pattern())
    assert len(cols) == COLUMNS
    assert xs == [MARGIN + k * PITCH for k in range(COLUMNS)]
    assert xs[-1] <= 60.0 - MARGIN + 1e-9


def test_neighbouring_columns_have_their_bridges_staggered():
    cols, _xs = columns_of(pattern())
    for a, b in zip(cols, cols[1:]):
        assert gap_centres(a) and gap_centres(b), "a column without a bridge hinges nothing"
        for ca in gap_centres(a):
            for cb in gap_centres(b):
                assert abs(((cb - ca) % PERIOD) - PERIOD / 2.0) < 1e-6


def test_no_column_is_cut_loose_by_one_slit():
    inner = (RECT[3] - RECT[1]) - 2.0 * MARGIN
    for p in pattern():
        assert float(p[1, 1]) - float(p[0, 1]) < inner


def test_the_number_of_slits():
    assert len(pattern()) == SLITS
    cols, _xs = columns_of(pattern())
    assert [len(c) for c in cols] == [2, 3] * (COLUMNS // 2), "a shifted column misses a piece"


def test_horizontal_is_the_same_pattern_turned():
    up = np.array(sorted(np.round(p, 6).tolist() for p in pattern(RECT)))
    side = np.array(sorted(np.round(p[:, ::-1], 6).tolist()
                           for p in pattern((0.0, 0.0, 40.0, 60.0), vertical=False)))
    assert len(side) == len(up) == SLITS
    assert np.allclose(side, up)
    for p in pattern((0.0, 0.0, 40.0, 60.0), vertical=False):
        assert p[0, 1] == pytest.approx(p[1, 1]), "the slit is a straight horizontal line"


def test_impossible_sizes_raise():
    with pytest.raises(ValueError):
        pattern(slit=3.0, gap=3.0)
    with pytest.raises(ValueError):
        pattern(slit=2.0, gap=3.0)
    with pytest.raises(ValueError):
        pattern(pitch=0.0)
    with pytest.raises(ValueError):
        pattern(rect=(0.0, 0.0, 6.0, 40.0))          # no room for two columns
    with pytest.raises(ValueError):
        pattern(rect=(0.0, 0.0, 60.0, 6.0))          # no room for a slit between the margins


# ------------------------------------------------------------------ in the window

def test_the_action_fills_the_selected_rectangle(win, monkeypatch):
    accept_dialog(monkeypatch)
    rect = rectangle(win)
    added = win.living_hinge_dialog()
    assert added is not None and len(added) == SLITS
    assert all(not p.closed for p in added), "a slit is an open line"
    cut = win.doc.sheet.default_layer("cut")
    assert all(p.layer == cut for p in added)
    assert all(int(p.src) == int(rect.src) for p in added), "the pattern moves with the panel"
    assert win.hint.text() == "70 slits added: the panel bends there."
    assert all(p in win.sheet_view.selected_paths() for p in added)


def test_the_slits_come_and_go_in_one_undo_step(win, monkeypatch):
    accept_dialog(monkeypatch)
    rectangle(win)
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    win.living_hinge_dialog()
    assert len(win.doc.undo_stack) == steps + 1, "one undo step for all slits together"
    assert len(win.doc.sheet.paths) == len(before) + SLITS
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    accept_dialog(monkeypatch)
    rectangle(win)
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins]
        seen["direction"] = dlg.findChild(QComboBox).currentText()
        spins[0].setValue(20.0)
        dlg.findChild(QComboBox).setCurrentIndex(1)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.hinge_slit = 25.0
    win.settings.hinge_vertical = False
    win.living_hinge_dialog()
    assert seen["start"] == [25.0, 3.0, 2.0, 3.0]
    assert seen["direction"] == "Bend around a horizontal axis"
    assert win.settings.hinge_slit == pytest.approx(20.0)
    assert win.settings.hinge_vertical is False


def test_a_circle_is_not_a_rectangle(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing to hinge: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    sheet = win.doc.sheet
    t = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    p = sheet.add_path(np.column_stack([30.0 + 20.0 * np.cos(t), 20.0 + 20.0 * np.sin(t)]),
                       closed=True, layer=sheet.default_layer("cut"))
    win.set_mode("2d")
    win.sheet_view.set_selection([p])
    before = list(sheet.paths)
    assert win.living_hinge_dialog() is None
    assert sheet.paths == before, "a circle keeps its shape"
    assert win.hint.text() == "Select one rectangle on the worksheet first."


def test_nothing_selected_is_a_clear_message(win):
    win.set_mode("2d")
    win.sheet_view.clear_selection()
    before = list(win.doc.sheet.paths)
    assert win.living_hinge_dialog() is None
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select one rectangle on the worksheet first."
