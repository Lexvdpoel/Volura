"""First Layer Test: thin squares in the corners and the middle of the bed, which shows how the
first layer lies.

`first_layer_test.first_layer_test` gives the test as five separate closed volumes standing on
`z = 0`, in bed coordinates: squares of `size` millimetres on a side and `height` millimetres
high, `margin` millimetres from the edge of the bed in the four corners and one in the middle of
a bed of `bed_x` by `bed_y` millimetres. They are far apart on purpose, because a bed can be a
hair too high in one corner and perfect in the middle. The size drops when the squares would
overlap or leave the bed.

These tests (Qt offscreen) weigh that: five closed volumes on the bed that are each a square of
`size` and as high as asked, their centres in the four corners and the middle of the bed for two
beds and none of them touching another, the size that drops on a small bed, every range reached
by clamping and a number that is no number refused, and the place in the interface (3D Printing,
right after the stringing test, five groups where they lie on the bed, one undo step, remembered
values).

Run with:  python -m pytest tests/test_first_layer_test.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox,  # noqa: E402
                               QLabel)

from volura import app as app_module, first_layer_test  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs Qt

NUMBERS = dict(bed_x=220.0, bed_y=220.0, size=30.0, height=0.3, margin=15.0)
#: the squares are exact polygons, so a part is at most this far off
ABS = 1e-4
#: the beds the places are checked on: square and wide
BEDS = ((220.0, 220.0), (300.0, 200.0))
#: the names the five groups get, in the order of the places
NAMES = ["First layer front left", "First layer front right", "First layer back left",
         "First layer back right", "First layer middle"]


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


def made(**kw) -> list:
    return first_layer_test.first_layer_test(**dict(NUMBERS, **kw))


def numbers(**kw) -> dict:
    return first_layer_test.sizes(**dict(NUMBERS, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def places(s):
    """The five places of the test: four corners `margin` from the edges and the middle."""
    c = s["margin"] + 0.5 * s["size"]
    return [(c, c), (s["bed_x"] - c, c), (c, s["bed_y"] - c), (s["bed_x"] - c, s["bed_y"] - c),
            (0.5 * s["bed_x"], 0.5 * s["bed_y"])]


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the squares

@pytest.mark.parametrize("bed_x, bed_y", BEDS)
def test_the_test_is_five_closed_volumes_that_stand_on_the_bed(bed_x, bed_y):
    s, ms = numbers(bed_x=bed_x, bed_y=bed_y), made(bed_x=bed_x, bed_y=bed_y)
    assert len(ms) == 5, "the test is not five squares"
    for m in ms:
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert len(m.decompose()) == 1, "a square fell apart"
        lo, hi = bbox(m)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the square does not stand on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "not as high as asked"


@pytest.mark.parametrize("bed_x, bed_y", BEDS)
def test_every_square_is_a_square_of_the_size_asked(bed_x, bed_y):
    s, ms = numbers(bed_x=bed_x, bed_y=bed_y, height=0.6), made(bed_x=bed_x, bed_y=bed_y,
                                                               height=0.6)
    for m in ms:
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["size"], abs=ABS), "not as wide as asked"
        assert hi[1] - lo[1] == pytest.approx(s["size"], abs=ABS), "not as deep as asked"
        assert hi[2] - lo[2] == pytest.approx(0.6, abs=ABS), "two layers is twice one layer"


@pytest.mark.parametrize("bed_x, bed_y", BEDS)
def test_the_squares_stand_in_the_corners_and_the_middle_of_the_bed(bed_x, bed_y):
    s, ms = numbers(bed_x=bed_x, bed_y=bed_y), made(bed_x=bed_x, bed_y=bed_y)
    want = places(s)
    for got, (x, y) in zip(s["centres"], want):
        assert got[0] == pytest.approx(x, abs=ABS) and got[1] == pytest.approx(y, abs=ABS)
    for m, (x, y) in zip(ms, want):
        lo, hi = bbox(m)
        assert 0.5 * (lo[0] + hi[0]) == pytest.approx(x, abs=ABS), "not there along x"
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(y, abs=ABS), "not there along y"
        assert lo[0] >= -ABS and hi[0] <= bed_x + ABS, "the square hangs off the bed along x"
        assert lo[1] >= -ABS and hi[1] <= bed_y + ABS, "the square hangs off the bed along y"


@pytest.mark.parametrize("bed_x, bed_y", BEDS)
def test_no_two_squares_touch_each_other(bed_x, bed_y):
    ms = made(bed_x=bed_x, bed_y=bed_y)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            assert (ms[i] ^ ms[j]).volume() == pytest.approx(0.0, abs=1e-6), \
                "two squares of the test grew together"


# ------------------------------------------------------------------ the numbers of the test

def test_sizes_reports_the_test_and_its_places():
    s = first_layer_test.sizes(bed_x=300.0, bed_y=200.0, size=25.0, height=0.5, margin=20.0)
    assert s["bed_x"] == 300.0
    assert s["bed_y"] == 200.0
    assert s["size"] == 25.0
    assert s["height"] == 0.5
    assert s["margin"] == 20.0
    assert s["centres"] == places(s)
    assert s["centres"][4] == (150.0, 100.0)
    assert s["centres"][0] == (32.5, 32.5)


def test_a_small_bed_makes_smaller_squares():
    """Three sides of `size` and twice the margin have to fit the shortest way across the bed."""
    s = numbers(bed_x=80.0, bed_y=80.0, size=60.0)
    assert s["size"] == pytest.approx((80.0 - 2.0 * s["margin"]) / 3.0)
    assert s["size"] < 60.0, "the squares still overlap on a small bed"
    ms = made(bed_x=80.0, bed_y=80.0, size=60.0)
    for m in ms:
        lo, hi = bbox(m)
        assert lo[0] >= -ABS and hi[0] <= 80.0 + ABS and lo[1] >= -ABS and hi[1] <= 80.0 + ABS, \
            "a square of a small bed hangs off it"
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            assert (ms[i] ^ ms[j]).volume() == pytest.approx(0.0, abs=1e-6)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(bed_x=10.0, bed_y=10.0, size=1.0, height=0.0, margin=0.0)
    assert small["bed_x"] == first_layer_test.BED_MIN
    assert small["bed_y"] == first_layer_test.BED_MIN
    assert small["size"] == first_layer_test.SIZE_MIN
    assert small["height"] == first_layer_test.HEIGHT_MIN
    assert small["margin"] == first_layer_test.MARGIN_MIN
    big = numbers(bed_x=9999.0, bed_y=9999.0, size=999.0, height=99.0, margin=999.0)
    assert big["bed_x"] == first_layer_test.BED_MAX
    assert big["bed_y"] == first_layer_test.BED_MAX
    assert big["size"] == first_layer_test.SIZE_MAX
    assert big["height"] == first_layer_test.HEIGHT_MAX
    assert big["margin"] == first_layer_test.MARGIN_MAX
    lo, hi = bbox(made(bed_x=9999.0, bed_y=9999.0, size=999.0, height=99.0, margin=999.0)[0])
    assert hi[2] - lo[2] == pytest.approx(first_layer_test.HEIGHT_MAX, abs=ABS)
    assert hi[0] - lo[0] == pytest.approx(first_layer_test.SIZE_MAX, abs=ABS)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("bed_x", "bed_y", "size", "height", "margin"):
        with pytest.raises(ValueError):
            first_layer_test.first_layer_test(**dict(NUMBERS, **{key: bad}))
        with pytest.raises(ValueError):
            first_layer_test.sizes(**dict(NUMBERS, **{key: bad}))


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The first layer test settings back at what the program starts with."""
    for key, value in (("first_layer_test_size", 30.0), ("first_layer_test_height", 0.3),
                       ("first_layer_test_margin", 15.0)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_builds_the_squares_and_remembers_it(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [sp.value() for sp in spins]
        seen["text"] = " ".join(lab.text() for lab in dlg.findChildren(QLabel))
        spins[0].setValue(25.0)
        spins[2].setValue(20.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.first_layer_test_size = 30.0
    win.settings.first_layer_test_height = 0.3
    win.settings.first_layer_test_margin = 15.0
    was = win.settings.bed_size
    win.settings.bed_size = (220.0, 220.0, 256.0)
    try:
        win.first_layer_test_dialog()
        assert seen["start"] == [pytest.approx(30.0), pytest.approx(0.3), pytest.approx(15.0)]
        assert "220" in seen["text"], "the dialog does not say which bed it uses"
        assert pump_until(lambda: len(win.doc.groups) == 5), "no first layer test"
        s = first_layer_test.sizes(bed_x=220.0, bed_y=220.0, size=25.0, margin=20.0)
        assert [g.name for g in win.doc.groups] == NAMES
        for g, (x, y) in zip(win.doc.groups, s["centres"]):
            m = g.ents.to_manifold()
            assert m.status() == mf.Error.NoError
            lo, hi = bbox(m)
            assert 0.5 * (lo[0] + hi[0]) == pytest.approx(x, abs=1e-3), "not there along x"
            assert 0.5 * (lo[1] + hi[1]) == pytest.approx(y, abs=1e-3), "not there along y"
            assert lo[2] == pytest.approx(0.0, abs=1e-6), "the square does not stand on the bed"
        assert [i[2] for i in win.doc.selection] == [g.id for g in win.doc.groups]
        assert pump_until(lambda: "squares" in win.hint.text()), win.hint.text()
        st = app_module.Settings()
        assert st.first_layer_test_size == pytest.approx(25.0)
        assert st.first_layer_test_height == pytest.approx(0.3)
        assert st.first_layer_test_margin == pytest.approx(20.0)
    finally:
        win.settings.bed_size = was
        forget_the_values(win.settings)


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_first_layer_test(30.0, 0.3, 15.0)
    assert pump_until(lambda: len(win.doc.groups) == 5)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("first_layer_test_size", "first_layer_test_height", "first_layer_test_margin")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (999.0, 999.0, 999.0)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.first_layer_test_size == 60.0
        assert s.first_layer_test_height == 1.0
        assert s.first_layer_test_margin == 50.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_stringing_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("First Layer Test") == labels.index("Stringing Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
