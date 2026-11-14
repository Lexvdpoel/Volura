"""Calibration cubes: the classic XYZ cube you print to check that the axes of a printer are to size.

`calibration_cube.calibration_cube` gives the cube as one closed volume standing on `z = 0`: a cube
of `size` millimetres, centred in x and y, with the axis of each direction engraved in the face it
runs along - Z in the top face, X in the face that looks to the front and Y in the face that looks
to the right - `depth` millimetres deep. The letters are cut in and never stood on, so the cube
stays `size` across in every direction and a caliper reads `size` off a good print. Without fonts
the cube is simply a cube.

These tests (Qt offscreen) weigh that: one volume on the bed and centred, `size` in x, y and z with
and without letters; a smaller volume than the cube once there are letters, with a letter on the
top, the front and the right face and nothing on the back, the left or the bottom; the plain cube
whole; every range reached by clamping and a number that is no number refused; and the place in the
interface (3D Printing, right after the overhang test, one group, remembered values).

Run with:  python -m pytest tests/test_calibration_cube.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox)  # noqa: E402

from volura import app as app_module, calibration_cube, fonts, ops  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the letters need Qt

BASE = dict(size=20.0, letters=True, depth=0.6)
#: the letters are polygons, so a part is at most this far off
ABS = 0.05


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
def noletters(monkeypatch):
    """No fonts: the cube is then exactly the cube."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def cube(**kw) -> mf.Manifold:
    return calibration_cube.calibration_cube(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return calibration_cube.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def slab(m, axis, lo, hi) -> float:
    """The volume of the part between the two planes that stand on `axis` at `lo` and `hi`."""
    thick = hi - lo
    size = [1e4, 1e4, 1e4]
    size[axis] = thick
    at = [-5e3, -5e3, -5e3]
    at[axis] = lo
    return (m ^ mf.Manifold.cube(size).translate(at)).volume()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the cube

@pytest.mark.parametrize("letters", (False, True))
@pytest.mark.parametrize("size", (10.0, 20.0, 50.0))
def test_the_cube_is_one_volume_that_stands_on_the_bed_and_is_centred(letters, size):
    m = cube(size=size, letters=letters)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the cube fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cube does not stand on the bed"
    assert np.allclose(lo, [-0.5 * size, -0.5 * size, 0.0], atol=ABS), "not centred in x and y"
    assert np.allclose(hi, [0.5 * size, 0.5 * size, size], atol=ABS), "not the size asked"


@pytest.mark.parametrize("letters", (False, True))
def test_the_cube_is_size_wide_deep_and_high(letters):
    """The letters are cut in, so they add nothing to what a caliper measures."""
    s, m = numbers(letters=letters), cube(letters=letters)
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["size"], abs=ABS), "not as wide as asked"
    assert hi[1] - lo[1] == pytest.approx(s["size"], abs=ABS), "not as deep as asked"
    assert hi[2] - lo[2] == pytest.approx(s["size"], abs=ABS), "not as high as asked"


def test_the_cube_without_letters_is_the_whole_cube():
    s, m = numbers(), cube(letters=False)
    assert m.volume() == pytest.approx(s["size"] ** 3, rel=1e-9)
    assert slab(m, 2, s["size"] - 0.5 * s["depth"], s["size"]) == pytest.approx(
        s["size"] ** 2 * 0.5 * s["depth"], rel=1e-9), "the top of a plain cube is not whole"


# ------------------------------------------------------------------ the letters

def block(size) -> mf.Manifold:
    """The same cube with nothing cut in it, to weigh the engraving against."""
    return mf.Manifold.cube([size, size, size]).translate((-0.5 * size, -0.5 * size, 0.0))


def face(axis, sign, size) -> float:
    """The plane of the face the cube has where `axis` and `sign` point; the cube stands on the
    bed, so its bottom is at 0 and its top at `size`, while x and y lie around the middle."""
    if axis == 2:
        return size if sign > 0 else 0.0
    return 0.5 * size * sign


def test_the_letters_make_a_smaller_volume_than_the_cube():
    need_fonts()
    made = cube(letters=True)
    assert made.volume() < block(numbers()["size"]).volume(), "nothing was engraved"
    assert len(made.decompose()) == 1, "the engraving cut the cube in two"


@pytest.mark.parametrize("axis, sign", ((2, 1.0), (1, -1.0), (0, 1.0)))
def test_a_letter_is_cut_in_the_face_it_belongs_to(axis, sign):
    """Just inside the top, the front and the right face the lettered cube holds less plastic than
    the plain cube: the letter of that face is engraved there."""
    need_fonts()
    s = numbers()
    edge, room = face(axis, sign, s["size"]), 0.5 * s["depth"]
    lo, hi = (edge - room, edge) if sign > 0 else (edge, edge + room)
    assert slab(cube(letters=True), axis, lo, hi) < slab(block(s["size"]), axis, lo, hi), \
        "there is no letter in the face it was asked for"


@pytest.mark.parametrize("axis, sign", ((2, -1.0), (1, 1.0), (0, -1.0)))
def test_the_other_faces_stay_whole(axis, sign):
    """The bottom, the back and the left face carry no letter, so they keep all their plastic."""
    need_fonts()
    s = numbers()
    edge, room = face(axis, sign, s["size"]), 0.5 * s["depth"]
    lo, hi = (edge - room, edge) if sign > 0 else (edge, edge + room)
    assert slab(cube(letters=True), axis, lo, hi) == pytest.approx(
        slab(block(s["size"]), axis, lo, hi), abs=1e-6), "a letter was cut in a face that has none"


@pytest.mark.parametrize("axis, sign", ((2, 1.0), (1, -1.0), (0, 1.0)))
def test_the_letters_do_not_cut_deeper_than_asked(axis, sign):
    """The slab under the engraving of every lettered face is whole, so the cut stops at `depth`."""
    need_fonts()
    s = numbers()
    edge, room = face(axis, sign, s["size"]), 0.2
    lo = edge - s["depth"] - room if sign > 0 else edge + s["depth"]
    assert slab(cube(letters=True), axis, lo, lo + room) == pytest.approx(
        slab(block(s["size"]), axis, lo, lo + room), abs=1e-6), "the engraving cuts too deep"


def test_a_cube_without_fonts_is_a_plain_cube(noletters):
    assert cube(letters=True).volume() == pytest.approx(numbers()["size"] ** 3, rel=1e-9)


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_cube_and_its_letters():
    s = calibration_cube.sizes(size=30.0, letters=False, depth=1.0)
    assert s == {"size": 30.0, "letters": False, "depth": 1.0}
    assert calibration_cube.sizes() == {"size": 20.0, "letters": True, "depth": 0.6}


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(size=0.0, depth=0.0)
    assert small["size"] == calibration_cube.SIZE_MIN
    assert small["depth"] == calibration_cube.DEPTH_MIN
    big = numbers(size=999.0, depth=999.0)
    assert big["size"] == calibration_cube.SIZE_MAX
    assert big["depth"] == calibration_cube.DEPTH_MAX
    lo, hi = bbox(cube(size=999.0, depth=999.0))
    assert hi[2] - lo[2] == pytest.approx(calibration_cube.SIZE_MAX, abs=ABS)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("size", "depth"):
        with pytest.raises(ValueError):
            calibration_cube.calibration_cube(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            calibration_cube.sizes(**dict(BASE, **{key: bad}))


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The calibration cube settings back at what the program starts with."""
    for key, value in (("calibration_cube_size", 20.0), ("calibration_cube_letters", True),
                       ("calibration_cube_depth", 0.6)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_builds_a_cube_and_remembers_it(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        cb = dlg.findChild(QCheckBox)
        seen["start"] = [spins[0].value(), cb.isChecked(), spins[1].value()]
        spins[0].setValue(30.0)
        assert cb.isChecked(), "the letters are off by default"
        spins[1].setValue(1.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.calibration_cube_size = 15.0
    win.settings.calibration_cube_letters = True
    win.settings.calibration_cube_depth = 0.5
    try:
        win.calibration_cube_dialog()
        assert seen["start"] == [pytest.approx(15.0), True, pytest.approx(0.5)]
        assert pump_until(lambda: len(win.doc.groups) == 1), "no cube"
        g = win.doc.groups[0]
        assert g.name == "Calibration cube"
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert [i[2] for i in win.doc.selection] == [g.id]
        lo, hi = bbox(g.ents.to_manifold())
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cube does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(30.0, abs=1e-3), "not as wide as asked"
        assert hi[2] - lo[2] == pytest.approx(30.0, abs=1e-3), "not as high as asked"
        assert pump_until(lambda: "caliper" in win.hint.text()), win.hint.text()
        s = app_module.Settings()
        assert s.calibration_cube_size == pytest.approx(30.0)
        assert s.calibration_cube_letters is True
        assert s.calibration_cube_depth == pytest.approx(1.0)
    finally:
        forget_the_values(win.settings)


def test_the_dialog_leaves_the_letters_off_when_they_are_off(win, monkeypatch):
    def accept(dlg):
        dlg.findChild(QCheckBox).setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.calibration_cube_size = 20.0
    try:
        win.calibration_cube_dialog()
        assert pump_until(lambda: len(win.doc.groups) == 1), "no cube"
        made = win.doc.groups[0].ents.to_manifold()
        assert made.volume() == pytest.approx(20.0 ** 3, rel=1e-6), "letters were cut anyway"
        assert app_module.Settings().calibration_cube_letters is False
    finally:
        forget_the_values(win.settings)


def test_one_undo_takes_the_whole_cube_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_calibration_cube(20.0, False, 0.6)
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the cube is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_cube_lands_next_to_what_is_already_in_the_model(win):
    ops.group_from_manifold(win.doc, mf.Manifold.cube([20.0, 20.0, 10.0]), "#cccccc", "block")
    win._apply_calibration_cube(20.0, True, 0.6)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no cube"
    cube_g = win.doc.groups[1]
    assert pump_until(lambda: cube_g.bbox() is not None and cube_g.bbox()[0][0] >= 29.9), \
        "the cube did not land 10 mm to the right of the model"
    assert cube_g.bbox()[0][2] == pytest.approx(0.0), "the cube does not stand on the bed"


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("calibration_cube_size", "calibration_cube_letters", "calibration_cube_depth")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (200.0, "no", 9.0)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.calibration_cube_size == 50.0
        assert s.calibration_cube_letters is False
        assert s.calibration_cube_depth == 1.2
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_overhang_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Calibration Cube") == labels.index("Overhang Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
