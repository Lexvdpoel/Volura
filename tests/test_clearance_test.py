"""Clearance Test: a plate with a row of holes and a loose pin that shows which clearance
this printer likes.

`threads.clearance_test` makes the two parts in one frame. The plate is a box of
``thickness`` mm high with a hole of Ø pin_d + c for every clearance, in a row along X,
``pin_d + 6`` mm from centre to centre and 5 mm from the edge; under every hole its clearance
stands on the top face in letters of 0.6 mm, so the plate is that much deeper at the front.
The pin is a cylinder of Ø pin_d and ``thickness`` + 4 mm high with a 0.4 mm chamfer on top,
standing on the bed 5 mm behind the plate. The user prints the plate, tries the pin in every
hole, and the clearance of the hole in which it fits best is the clearance to use in Screw
Hole, threads and Cut and Join. Without fonts the plate simply has no labels, so it is then
exactly the box with the holes out of it; a hole is a polygon of ``segments`` sides, whose
area is `n / 2 * r² * sin(2 pi / n)`, 0.08 % under a circle at 64 sides.

These tests (Qt offscreen) check those volumes, the order of the row, the pin, the refused
input, and the place in the interface (Tools > Print Tools, two groups, one undo).

Run with:  python -m pytest tests/test_clearance_test.py
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
from volura.brep import Entities  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

PIN = 5.0
CLEARANCES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
THICKNESS = 4.0
LABELS = 3.0
SEGMENTS = 64
SPACING = PIN + 6.0
RADII = [0.5 * (PIN + c) for c in CLEARANCES]
BACK = max(RADII) + 5.0                                   # mm of plate behind the holes
FRONT = BACK + threads.CLEARANCE_TEST_ROOM + LABELS       # room for the letters in front
LO_X = -RADII[0] - 5.0
HI_X = (len(CLEARANCES) - 1) * SPACING + RADII[-1] + 5.0
BOX = (HI_X - LO_X) * (BACK + FRONT) * THICKNESS


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
    """No fonts: the plate is then exactly the box with the holes cut out of it."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def polygon_area(radius, segments=SEGMENTS):
    """The area of the regular polygon a hole really is."""
    return 0.5 * segments * radius ** 2 * math.sin(2.0 * math.pi / segments)


def plate(**kw):
    m, _pin = threads.clearance_test(**kw)
    return m


def bbox(m):
    bb = np.asarray(m.bounding_box(), dtype=float)
    return bb[:3], bb[3:]


def openings(plate_man, z=THICKNESS):
    """(x, radius) of every round hole in the horizontal face of the plate at height `z`."""
    ents = Entities.from_manifold(plate_man)
    out = []
    for fid in ents.F:
        if abs(float(ents.F[fid].normal[2])) < 0.999:
            continue
        pts = np.vstack(ents.face_points(fid))
        if np.any(np.abs(pts[:, 2] - z) > 1e-6):
            continue
        for loop in ents.face_points(fid):
            c, r = threads.circle_loop(np.asarray(loop, dtype=float))
            if c is not None:
                out.append((float(c[0]), r))
    return sorted(out)


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the geometry

def test_the_plate_is_the_box_minus_the_holes(nolabels):
    m = plate(pin_d=PIN, clearances=CLEARANCES, thickness=THICKNESS, segments=SEGMENTS)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() == pytest.approx(BOX - sum(polygon_area(r) * THICKNESS for r in RADII),
                                       rel=1e-6)
    # the row it should be: every hole 0.08 % smaller than a circle at 64 sides
    assert m.volume() == pytest.approx(BOX - sum(math.pi * r ** 2 * THICKNESS for r in RADII),
                                       rel=0.005)


def test_the_plate_stands_on_the_bed_with_room_for_the_letters(nolabels):
    lo, hi = bbox(plate())
    assert lo == pytest.approx([LO_X, -FRONT, 0.0])
    assert hi == pytest.approx([HI_X, BACK, THICKNESS]), "no labels: nothing sticks up"


def test_the_letters_stand_under_the_row_and_add_volume(monkeypatch):
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")
    m = plate()
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])
    plain = plate()
    assert m.volume() > plain.volume(), "the letters add nothing"
    assert (plain - m).volume() == pytest.approx(0.0, abs=1e-3), "the letters cut into the plate"
    lo, hi = bbox(m)
    assert hi[2] == pytest.approx(THICKNESS + threads.CLEARANCE_TEST_LABEL)
    assert (lo[:2], hi[:2]) == ((pytest.approx([LO_X, -FRONT]), pytest.approx([HI_X, BACK]))), \
        "the letters hang outside the plate"
    # the letters of a hole stand under it, clear of the hole itself
    for (x, r), c in zip(openings(plain), CLEARANCES):
        letters = m ^ mf.Manifold.cube([2 * r, 2 * r, 2 * r]).translate((x - r, -r, THICKNESS))
        assert letters.volume() == pytest.approx(0.0, abs=1e-9), f"{c:.1f} lies in its hole"


def test_the_holes_grow_from_left_to_right(nolabels):
    got = openings(plate())
    assert [x for x, _r in got] == pytest.approx([i * SPACING for i in range(len(RADII))])
    assert [r for _x, r in got] == pytest.approx(RADII, abs=0.01)


def test_the_holes_go_all_the_way_through(nolabels):
    m = plate()
    for x, _r in openings(m):
        probe = mf.Manifold.cube([0.4, 0.4, THICKNESS]).translate((x - 0.2, -0.2, 0.0))
        assert (probe ^ m).volume() == pytest.approx(0.0, abs=1e-9), "the hole is not through"


def test_the_pin_is_a_cylinder_with_a_chamfered_top():
    _m, pin = threads.clearance_test(pin_d=PIN, clearances=CLEARANCES, thickness=THICKNESS,
                                     segments=SEGMENTS)
    assert pin.status() == mf.Error.NoError and not pin.is_empty()
    ch, h, r = threads.CLEARANCE_TEST_CHAMFER, THICKNESS + 4.0, 0.5 * PIN
    want = (math.pi * r ** 2 * (h - ch)
            + math.pi * ch / 3.0 * (r ** 2 + r * (r - ch) + (r - ch) ** 2))
    assert pin.volume() == pytest.approx(want, rel=0.01)
    lo, hi = bbox(pin)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(h), "the pin is not that tall"
    assert hi[0] - lo[0] == pytest.approx(PIN) and hi[1] - lo[1] == pytest.approx(PIN)


def test_the_pin_stands_on_the_bed_next_to_the_plate(nolabels):
    m, pin = threads.clearance_test()
    lo, _hi = bbox(pin)
    assert lo[2] == pytest.approx(0.0), "the pin does not stand on the bed"
    assert lo[1] - bbox(m)[1][1] == pytest.approx(5.0), "the pin is not 5 mm from the plate"
    assert (m ^ pin).volume() == pytest.approx(0.0, abs=1e-9), "plate and pin overlap"


def test_the_smallest_and_the_biggest_pin_both_work(nolabels):
    for pin_d in (2.0, 20.0):
        m, pin = threads.clearance_test(pin_d=pin_d, clearances=(0.0, 2.0))
        assert m.status() == mf.Error.NoError and pin.status() == mf.Error.NoError
        assert m.volume() > 0 and pin.volume() > 0


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="pin"):
        threads.clearance_test(pin_d=1.0)
    with pytest.raises(ValueError, match="pin"):
        threads.clearance_test(pin_d=25.0)
    with pytest.raises(ValueError, match="two holes"):
        threads.clearance_test(clearances=(0.2,))
    with pytest.raises(ValueError, match="clearance"):
        threads.clearance_test(clearances=(0.0, 2.5))
    with pytest.raises(ValueError, match="clearance"):
        threads.clearance_test(clearances=(-0.1, 0.0))


# -------------------------------------------------------------------- the window

def test_the_action_makes_two_groups_that_are_selected(win):
    win._apply_clearance_test(PIN, 0.0, 0.1, 6)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no plate and no pin"
    assert [g.name for g in win.doc.groups] == ["Clearance test", "Clearance test pin"]
    assert all(g.ents.to_manifold().status() == mf.Error.NoError for g in win.doc.groups)
    assert sorted(i[2] for i in win.doc.selection) == sorted(g.id for g in win.doc.groups)
    assert pump_until(lambda: "clearance" in win.hint.text()), win.hint.text()


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_clearance_test(PIN, 0.0, 0.1, 6)
    assert pump_until(lambda: len(win.doc.groups) == 2)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_test_lands_next_to_what_is_already_in_the_model(win):
    ops.group_from_manifold(win.doc, mf.Manifold.cube([20.0, 20.0, 10.0]), "#cccccc", "block")
    win._apply_clearance_test(PIN, 0.0, 0.1, 6)
    assert pump_until(lambda: len(win.doc.groups) == 3), "no plate and no pin"
    plate_g = next(g for g in win.doc.groups if g.name == "Clearance test")
    assert pump_until(lambda: plate_g.bbox() is not None and plate_g.bbox()[0][0] >= 29.9), \
        "the test did not land 10 mm to the right of the model"
    assert plate_g.bbox()[0][2] == pytest.approx(0.0), "the plate does not stand on the bed"


def test_the_row_stops_at_two_mm_of_clearance(win):
    """The spin boxes together allow more than 2 mm: then the step is made smaller."""
    win._apply_clearance_test(PIN, 1.0, 0.5, 6)
    assert pump_until(lambda: len(win.doc.groups) == 2)
    m = next(g for g in win.doc.groups if g.name == "Clearance test").ents.to_manifold()
    step = (2.0 - 1.0) / 5                       # the biggest step that keeps the row under 2 mm
    radii = [0.5 * (PIN + 1.0 + i * step) for i in range(6)]
    assert [r for _x, r in openings(m)] == pytest.approx(radii, abs=0.01)
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(5 * SPACING + radii[0] + radii[-1] + 10.0)


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        count = dlg.findChild(QSpinBox)
        seen["start"] = [spins[0].value(), spins[1].value(), spins[2].value(), count.value()]
        spins[0].setValue(6.0)
        spins[1].setValue(0.15)
        spins[2].setValue(0.25)
        count.setValue(8)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.clearance_test_pin = 4.0
    win.settings.clearance_test_first = 0.05
    win.settings.clearance_test_step = 0.2
    win.settings.clearance_test_count = 5
    win.clearance_test_dialog()
    assert seen["start"] == [4.0, pytest.approx(0.05), pytest.approx(0.2), 5]
    assert pump_until(lambda: len(win.doc.groups) == 2)
    s = app_module.Settings()
    assert (s.clearance_test_pin, s.clearance_test_first) == (6.0, pytest.approx(0.15))
    assert (s.clearance_test_step, s.clearance_test_count) == (pytest.approx(0.25), 8)


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("clearance_test_pin", "clearance_test_first", "clearance_test_step",
            "clearance_test_count")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (80.0, -3.0, "wide", 99)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.clearance_test_pin == 20.0
        assert s.clearance_test_first == 0.0
        assert s.clearance_test_step == 0.1
        assert s.clearance_test_count == 10
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_print_tools_menu_after_the_screw_hole(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.print_tools_menu)]
    at = labels.index("Thread")
    assert labels[at:at + 7] == ["Thread", "External Thread on Cylinder", "Internal Thread in Hole",
                                 "New Bolt", "New Nut or Sleeve", "Screw Hole", "Clearance Test"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
