"""Overhang Test: a plate with a row of fins that lean further over, which shows up to which
angle this printer prints an overhang without support.

`threads.overhang_test` makes one manifold that stands on z = 0: a plate of ``plate`` mm high
with a row of fins along X, ``wall + length + 6`` mm from post to post. Every fin is a post of
``wall`` from which a slab of ``wall`` leans out at its angle measured from the vertical, so 0
degrees is a straight wall and 90 a flat bridge; the underside of the slab is the overhang and
nothing holds it up there. In front of every fin its angle stands on the plate in raised letters,
the way a clearance test names its holes, so without fonts the plate is simply the plate with the
fins on it. With the angles from small to large the fins lean further over from left to right, and
the top of a fin comes out lower the further it leans, so the bounding box of a one-angle test
falls as the angle grows.

These tests (Qt offscreen) check that solid, the row, the leaning, the refused input and the place
in the interface (Tools > Print Tools, one group, one undo).

Run with:  python -m pytest tests/test_overhang_test.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QSpinBox  # noqa: E402

from volura import app as app_module, fonts, ops, threads  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ANGLES = (30, 40, 50, 60, 70)
WIDTH, LENGTH, WALL = 10.0, 10.0, 2.0
PLATE = threads.OVERHANG_TEST_PLATE
POST = threads.OVERHANG_TEST_POST


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def nolabels(monkeypatch):
    """No fonts: the plate is then exactly the plate with the fins on it."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def bbox(m):
    bb = np.asarray(m.bounding_box(), dtype=float)
    return bb[:3], bb[3:]


def fin_top(angle):
    """The height the top of a fin reaches: the post less half a wall, plus the slab that leans."""
    return PLATE + POST - 0.5 * WALL + LENGTH * math.cos(math.radians(angle))


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the geometry

def test_the_test_is_one_valid_solid_on_the_bed(nolabels):
    m = threads.overhang_test()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.genus() == 0, "a plate with fins on it has no hole through it"
    lo, _hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the plate does not stand on the bed"


def test_the_fins_stand_up_from_the_plate(nolabels):
    _lo, hi = bbox(threads.overhang_test())
    assert hi[2] > PLATE, "nothing stands on the plate"


def test_the_row_grows_with_the_number_of_fins(nolabels):
    lo2, hi2 = bbox(threads.overhang_test(angles=(30, 40)))
    lo5, hi5 = bbox(threads.overhang_test(angles=ANGLES))
    assert hi5[0] - lo5[0] > hi2[0] - lo2[0], "more fins did not make the row longer"


def test_a_fin_that_leans_further_over_comes_out_lower(nolabels):
    _lo, hi_small = bbox(threads.overhang_test(angles=(30,), width=WIDTH, length=LENGTH,
                                               wall=WALL))
    _lo, hi_big = bbox(threads.overhang_test(angles=(60,), width=WIDTH, length=LENGTH,
                                             wall=WALL))
    assert hi_small[2] > hi_big[2], "the fin that leans more does not reach lower"
    assert hi_small[2] == pytest.approx(fin_top(30), rel=1e-3)
    assert hi_big[2] == pytest.approx(fin_top(60), rel=1e-3)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="1 to 10"):
        threads.overhang_test(angles=())
    with pytest.raises(ValueError, match="1 to 10"):
        threads.overhang_test(angles=tuple(range(11)))
    with pytest.raises(ValueError, match="85"):
        threads.overhang_test(angles=(90,))
    with pytest.raises(ValueError, match="85"):
        threads.overhang_test(angles=(-5.0, 30.0))
    with pytest.raises(ValueError, match="positive"):
        threads.overhang_test(width=0.0)
    with pytest.raises(ValueError, match="positive"):
        threads.overhang_test(length=-1.0)
    with pytest.raises(ValueError, match="positive"):
        threads.overhang_test(wall=0.0)


# -------------------------------------------------------------------- the window

def test_the_action_makes_one_group_that_is_selected(win):
    win._apply_overhang_test(30.0, 10.0, 6)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no plate"
    g = win.doc.groups[0]
    assert g.name == "Overhang test"
    assert g.ents.to_manifold().status() == mf.Error.NoError
    assert [i[2] for i in win.doc.selection] == [g.id]
    assert pump_until(lambda: "angle" in win.hint.text()), win.hint.text()


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_overhang_test(30.0, 10.0, 6)
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_test_lands_next_to_what_is_already_in_the_model(win):
    ops.group_from_manifold(win.doc, mf.Manifold.cube([20.0, 20.0, 10.0]), "#cccccc", "block")
    win._apply_overhang_test(30.0, 10.0, 6)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no plate"
    plate_g = win.doc.groups[1]
    assert pump_until(lambda: plate_g.bbox() is not None and plate_g.bbox()[0][0] >= 29.9), \
        "the test did not land 10 mm to the right of the model"
    assert plate_g.bbox()[0][2] == pytest.approx(0.0), "the plate does not stand on the bed"


def test_the_row_stops_at_85_degrees(win):
    """The spin boxes together allow more than 85 degrees: then the step is made smaller."""
    win._apply_overhang_test(60.0, 30.0, 6)
    assert pump_until(lambda: len(win.doc.groups) == 1), "the row was refused instead of clamped"
    _lo, hi = bbox(win.doc.groups[0].ents.to_manifold())
    # the step falls to (85 - 60) / 5, so the tallest fin is still the first one at 60 degrees
    assert hi[2] == pytest.approx(fin_top(60), rel=1e-3)


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        count = dlg.findChild(QSpinBox)
        seen["start"] = [spins[0].value(), spins[1].value(), count.value()]
        spins[0].setValue(40.0)
        spins[1].setValue(5.0)
        count.setValue(8)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.overhang_test_first = 20.0
    win.settings.overhang_test_step = 15.0
    win.settings.overhang_test_count = 4
    win.overhang_test_dialog()
    assert seen["start"] == [pytest.approx(20.0), pytest.approx(15.0), 4]
    assert pump_until(lambda: len(win.doc.groups) == 1)
    s = app_module.Settings()
    assert (s.overhang_test_first, s.overhang_test_step) == (pytest.approx(40.0),
                                                             pytest.approx(5.0))
    assert s.overhang_test_count == 8


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("overhang_test_first", "overhang_test_step", "overhang_test_count")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (200.0, "wide", 99)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.overhang_test_first == 85.0
        assert s.overhang_test_step == 10.0
        assert s.overhang_test_count == 10
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_print_tools_menu_after_the_clearance_test(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.print_tools_menu)]
    at = labels.index("Thread")
    assert labels[at:at + 8] == ["Thread", "External Thread on Cylinder", "Internal Thread in Hole",
                                 "New Bolt", "New Nut or Sleeve", "Screw Hole", "Clearance Test",
                                 "Overhang Test"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
