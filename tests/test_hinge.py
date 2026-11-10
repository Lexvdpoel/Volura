"""Print-in-place hinges: a hinge that prints as one piece and turns as soon as it is off the bed.

`hinge.hinge` gives the hinge as one closed volume that lies flat on `z = 0` and is centred on the
origin in plan: two leaves of `width` by `thickness` millimetres and `length` millimetres beside
each other, joined by `knuckles` barrels around one pin of `pin_d` millimetres. Half of the barrels
are part of the leaf that carries the pin, the other half have a hole of `pin_d + 2 clearance` and
turn on it, and neighbouring barrels leave `clearance` millimetres between them, so nothing is
assembled afterwards.

These tests weigh a hinge against what it promises: it is one volume of two leaves that are not
fused, the pin stands in its hole with the play asked for, and the box round it is the size asked
for. A size that cannot be printed becomes the nearest one that can. The place in the interface is
checked too (Parts right after the cable clip, one group in one undo step, and the dialog that
remembers its values).

Run with:  python -m pytest tests/test_hinge.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import hinge  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

L, W, T, N, C, PIN = 40.0, 15.0, 2.0, 5, 0.3, 3.0
BASE = dict(length=L, width=W, thickness=T, knuckles=N, clearance=C, pin_d=PIN)
OPTIONS = dict(BASE)


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def make(**kw):
    return hinge.hinge(**dict(BASE, **kw))


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


# ------------------------------------------------------------------ the hinge

def test_the_default_hinge_is_one_valid_volume_that_lies_on_the_bed():
    m = make()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() > 0.0, "the hinge is not a volume"
    lo, _hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the hinge does not lie on the bed"


def test_the_hinge_is_two_leaves_that_are_not_fused():
    parts = make().decompose()
    assert len(parts) == 2, "the two leaves printed into one plate"
    a, b = parts
    assert a.min_gap(b, 1.0) >= 0.9 * C, "the pin is fused to the leaf with the hole"


def test_the_box_round_the_hinge_is_the_size_asked():
    lo, hi = bbox(make())
    assert hi[0] - lo[0] == pytest.approx(L), "the hinge is not as long as asked"
    assert hi[1] - lo[1] == pytest.approx(2 * W + hinge.LEAF_GAP), "the leaves are not side by side"
    assert hi[2] - lo[2] == pytest.approx(PIN + 2 * C + 2 * T), "the knuckle is not one barrel high"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the hinge is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the hinge is not centred"


def test_an_odd_number_of_knuckles_holds_the_pin_at_both_ends():
    assert make(knuckles=4).volume() == pytest.approx(make(knuckles=3).volume(), rel=1e-6), \
        "an even number of knuckles is not made odd"


def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can():
    m = make(length=5000.0, width=1000.0, thickness=0.1, knuckles=99, clearance=0.01, pin_d=99.0)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(hinge.LENGTH_MAX)
    assert hi[1] - lo[1] == pytest.approx(2 * hinge.WIDTH_MAX + hinge.LEAF_GAP)
    assert hi[2] - lo[2] == pytest.approx(hinge.PIN_MAX + 2 * hinge.CLEARANCE_MIN
                                          + 2 * hinge.THICKNESS_MIN)
    assert len(m.decompose()) == 2, "the clamped hinge is not two leaves"
    short = bbox(make(length=0.1))
    assert short[1][0] - short[0][0] == pytest.approx(hinge.LENGTH_MIN)


# ------------------------------------------------------------------ in the window

def test_the_hinge_is_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_hinge_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no hinge in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the hinge is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Print-in-place hinge"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(make().volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_hinge_lies_on_the_bed_next_to_the_model(win):
    win.make_hinge_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no hinge in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    bx, by, _bz = win.settings.bed_size
    assert lo[2] == pytest.approx(0.0), "the hinge does not lie on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the hinge is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    assert pump_until(lambda: "Print-in-place hinge" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Print-in-place hinge of 40 mm added."
    win.make_hinge_model(dict(OPTIONS, width=25.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second hinge is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the hinges are not 10 mm apart"


def test_the_hinge_follows_the_cable_clip_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Print-in-Place Hinge") == labels.index("Cable Clip") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [round(dlg.length.value(), 2), round(dlg.width.value(), 2),
                         round(dlg.thickness.value(), 2), int(dlg.knuckles.value()),
                         round(dlg.clearance.value(), 2), round(dlg.pin_d.value(), 2)]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        dlg.length.setValue(60.0)
        dlg.width.setValue(20.0)
        dlg.thickness.setValue(2.5)
        dlg.knuckles.setValue(7)
        dlg.clearance.setValue(0.4)
        dlg.pin_d.setValue(4.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.hinge_length, win.settings.hinge_width = 30.0, 12.0
    win.settings.hinge_thickness, win.settings.hinge_knuckles = 1.5, 3
    win.settings.hinge_clearance, win.settings.hinge_pin_d = 0.25, 2.0
    win.hinge_dialog()
    assert seen["title"] == "Print-in-place hinge"
    assert seen["start"] == [30.0, 12.0, 1.5, 3, 0.25, 2.0]
    assert seen["spins"] == 5
    assert pump_until(lambda: len(win.doc.groups) == 1), "no hinge in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(60.0), "the length is not as asked"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 2, \
        "the knuckles did not reach the model"
    assert win.settings.hinge_knuckles == 7
    s = app_module.Settings()
    assert (s.hinge_length, s.hinge_width, s.hinge_thickness, s.hinge_knuckles,
            s.hinge_clearance, s.hinge_pin_d) == (pytest.approx(60.0), pytest.approx(20.0),
                                                  pytest.approx(2.5), 7, pytest.approx(0.4),
                                                  pytest.approx(4.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
