"""Spur gear: an involute gear wheel that the laser or the mill cuts out of a sheet.

`gears.spur_gear` gives the wheel as closed contours: the outline, counter-clockwise, and
the bore, clockwise, when one is asked for. The size comes from the module, the pitch
diameter per tooth, so a wheel of 20 teeth of module 2 has a pitch diameter of 40 mm,
stands 2 mm above that circle with its teeth and has its root 2.5 mm below it, at 17.5 mm.
The flanks are involutes, sampled finely enough that every chord stays within `tol` of the
curve, and every tooth is thinned by the backlash, half on each flank, so a pair of wheels
has the play it needs. Two wheels of the same module at their centre distance
(`gears.center_distance`) then run along each other without pressing.

The same contours also make a wheel to print: extruded over its thickness, standing on the
bed, with the bore a hole through it.

These tests (Qt offscreen) measure the circles of the wheel, count its teeth, let a pair
of wheels run against each other, check the refused input and the place in the interface
(View menu and sheet toolbar next to Fit Test, the wheel on the cut layer in one undo
step, and the printed wheel in Print Tools next to the clearance test).

Run with:  python -m pytest tests/test_gears.py
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
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QMenu, QSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import gears  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TOL = 0.02
OPTIONS = {"teeth": 20, "module": 2.0, "angle": 20.0, "bore": 5.0, "backlash": 0.1}
THICKNESS = 8.0
SOLID = dict(OPTIONS, thickness=THICKNESS)


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def area(loop):
    """The shoelace area of a loop: positive counter-clockwise, negative clockwise."""
    p = np.asarray(loop, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def section(*loops):
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float)) for q in loops],
                           mf.FillRule.NonZero)


def radii(loop, center=(0.0, 0.0)):
    p = np.asarray(loop, dtype=float) - np.asarray(center, dtype=float)
    return np.hypot(p[:, 0], p[:, 1])


def rotate(loop, deg, at=(0.0, 0.0)):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(at, dtype=float)
    return (np.asarray(loop, dtype=float) - c) @ R.T + c


def meshed(z1=20, z2=31, backlash=0.0, module=2.0):
    """The two wheels at their centre distance, the second with a space to the first."""
    a = gears.spur_gear(z1, module=module, backlash=backlash)[0]
    b = gears.spur_gear(z2, module=module, backlash=backlash)[0]
    # half a tooth pitch: the tooth of the one wheel stands in the space of the other
    place = rotate(b, 180.0 + 180.0 / z2) + np.array([gears.center_distance(z1, z2, module),
                                                      0.0])
    return a, place


def gap(a, b):
    return float(np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])))


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


def menu_labels(win, title):
    for m in win.menuBar().findChildren(QMenu):
        if clean(m.title()) == title:
            return [clean(a.text()) for a in m.actions() if not a.isSeparator()]
    raise AssertionError(f"no menu {title}")


# ------------------------------------------------------------------ the wheel

def test_the_teeth_stand_between_the_root_and_the_tip_circle():
    outer = gears.spur_gear(20)[0]
    r = radii(outer)
    assert r.max() == pytest.approx(22.0, abs=TOL), "the tip circle is r + module"
    assert r.min() >= 17.5 - TOL, "the root circle is r − 1.25 × module"
    assert r.min() == pytest.approx(17.5, abs=TOL)


def test_twenty_teeth_stand_evenly_round_the_wheel():
    outer = gears.spur_gear(20)[0]
    high = radii(outer) > 0.5 * (22.0 + 17.5)
    tips = sum(1 for i in range(len(high)) if high[i] and not high[i - 1])
    assert tips == 20
    both = mf.CrossSection.batch_boolean([section(outer), section(rotate(outer, 360.0 / 20))],
                                         mf.OpType.Intersect)
    assert both.area() == pytest.approx(section(outer).area(), abs=1e-6), \
        "turning the wheel one tooth pitch gives the same wheel"


def test_the_outline_is_counter_clockwise_and_does_not_cross_itself():
    outer = gears.spur_gear(20, bore=5.0)[0]
    assert area(outer) > 0.0, "the outline of the wheel must be counter-clockwise"
    assert section(outer).area() == pytest.approx(area(outer), abs=1e-6)


def test_the_wheel_is_cut_where_the_tooth_becomes_pointed():
    outer = gears.spur_gear(300, module=0.3, pressure_angle=30.0)[0]
    assert section(outer).area() == pytest.approx(area(outer), abs=1e-6)
    assert radii(outer).max() <= 0.3 * 300 / 2.0 + 0.3 + TOL


def test_the_bore_is_a_hole_in_the_wheel():
    assert len(gears.spur_gear(20)) == 1, "without a bore the wheel is one loop"
    outer, bore = gears.spur_gear(20, bore=5.0)
    assert radii(bore).max() == pytest.approx(2.5, abs=TOL)
    assert area(bore) < 0.0, "the bore must be clockwise"
    # a polygon inside the circle: it misses at most its chord error along the rim
    assert area(bore) == pytest.approx(-math.pi * 2.5 ** 2, abs=2.0 * math.pi * 2.5 * TOL)
    assert section(outer, bore).area() == pytest.approx(section(outer).area() + area(bore),
                                                        abs=1e-6)


def test_the_wheel_stands_where_it_is_asked():
    outer = gears.spur_gear(20, center=(30.0, 40.0))[0]
    assert radii(outer, (30.0, 40.0)).max() == pytest.approx(22.0, abs=TOL)


def test_a_finer_tolerance_cuts_the_flanks_more_finely():
    assert len(gears.spur_gear(20, tol=0.002)[0]) > len(gears.spur_gear(20, tol=0.2)[0])


def test_two_wheels_at_the_centre_distance_run_along_each_other():
    a, b = meshed(backlash=0.0)
    inter = mf.CrossSection.batch_boolean([section(a), section(b)], mf.OpType.Intersect)
    assert inter.area() < 1e-3, "the teeth may not run into each other"
    assert gap(a, b) < 0.05, "without backlash the flanks must nearly touch"
    loose_a, loose_b = meshed(backlash=0.1)
    assert gap(loose_a, loose_b) > gap(a, b), "backlash makes room between the flanks"
    assert mf.CrossSection.batch_boolean([section(loose_a), section(loose_b)],
                                         mf.OpType.Intersect).area() < 1e-3


def test_the_diameters_a_gear_needs():
    assert gears.pitch_diameter(20, 2.0) == pytest.approx(40.0)
    assert gears.pitch_diameter(31, 2.0) == pytest.approx(62.0)
    assert gears.center_distance(20, 31, 2.0) == pytest.approx(51.0)
    assert gears.center_distance(20, 20, 2.0) == pytest.approx(gears.pitch_diameter(20, 2.0))


@pytest.mark.parametrize("kw, wrong", (
    ({"teeth": 5}, "teeth"), ({"teeth": 301}, "teeth"), ({"teeth": 20.5}, "teeth"),
    ({"module": 0.2}, "module"), ({"module": 21.0}, "module"),
    ({"pressure_angle": 14.0}, "pressure angle"), ({"pressure_angle": 31.0}, "pressure angle"),
    ({"bore": -1.0}, "bore"), ({"bore": 35.0}, "bore"),
    ({"backlash": -0.1}, "backlash"),
    ({"teeth": 6, "module": 0.3, "backlash": 1.0}, "backlash")))
def test_impossible_wheels_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        gears.spur_gear(**{"teeth": 20, "module": 2.0, **kw})


# ------------------------------------------------------------------ in the window

def test_the_action_adds_the_wheel_in_one_undo_step(win):
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    added = win.make_spur_gear_sheet(OPTIONS)
    assert added, "nothing on the worksheet"
    assert len(win.doc.undo_stack) == steps + 1, "the wheel is not one undo step"
    assert len(win.doc.sheet.paths) == len(before) + len(added)
    cut = win.doc.sheet.default_layer("cut")
    assert [p.layer for p in added] == [cut, cut], "the wheel and its bore on the cut layer"
    assert all(int(p.src) == int(added[0].src) for p in added), "the wheel is one part"
    assert all(p in win.sheet_view.selected_paths() for p in added)
    assert win.hint.text() == "Spur gear with 20 teeth added (pitch diameter 40.00 mm)."
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_the_wheel_lands_to_the_right_of_what_is_already_on_the_sheet(win):
    sheet = win.doc.sheet
    sheet.add_path([[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]], closed=True,
                   layer=sheet.default_layer("cut"))
    added = win.make_spur_gear_sheet(dict(OPTIONS, bore=0.0))
    pts = np.vstack([np.asarray(p.pts, dtype=float) for p in added])
    assert float(pts[:, 0].min()) == pytest.approx(40.0 + sheet.sheet_gap)
    assert float(pts[:, 1].min()) == pytest.approx(0.0)


def test_a_bore_that_eats_the_wheel_is_a_clear_text(win):
    before = list(win.doc.sheet.paths)
    assert win.make_spur_gear_sheet(dict(OPTIONS, bore=39.0)) is None
    assert win.doc.sheet.paths == before
    assert "bore" in win.hint.text()


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [dlg.findChild(QSpinBox).value()] + [round(sp.value(), 2)
                                                             for sp in spins]
        seen["title"] = dlg.windowTitle()
        seen["info"] = dlg.info.text()
        dlg.findChild(QSpinBox).setValue(30)
        spins[0].setValue(1.5)
        seen["wider"] = dlg.info.text()
        spins[2].setValue(8.0)
        spins[3].setValue(0.2)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.gear_teeth = 17
    win.settings.gear_module = 1.0
    win.settings.gear_angle = 14.5
    win.settings.gear_bore = 4.0
    win.settings.gear_backlash = 0.05
    win.spur_gear_dialog()
    assert seen["start"] == [17, 1.0, 14.5, 4.0, 0.05]
    assert seen["title"] == "Spur Gear"
    assert "17.00" in seen["info"] and "19.00" in seen["info"]
    assert "45.00" in seen["wider"] and "48.00" in seen["wider"]
    assert (win.settings.gear_teeth, win.settings.gear_module, win.settings.gear_angle,
            win.settings.gear_bore, win.settings.gear_backlash) == \
        (30, pytest.approx(1.5), pytest.approx(14.5), pytest.approx(8.0), pytest.approx(0.2))
    s = app_module.Settings()
    assert (s.gear_teeth, s.gear_module, s.gear_bore) == (30, pytest.approx(1.5),
                                                          pytest.approx(8.0))


def test_the_spur_gear_follows_the_fit_test_in_the_view_menu(win):
    labels = menu_labels(win, "View")
    assert labels.index("Fit Test") == labels.index("Vent Pattern") + 1
    assert labels.index("Spur Gear") == labels.index("Fit Test") + 1


def test_the_spur_gear_follows_the_fit_test_on_the_sheet_toolbar(win):
    labels = [clean(a.text()) for a in win.tb_sheet.actions() if not a.isSeparator()]
    assert labels.index("Spur Gear") == labels.index("Fit Test") + 1


# ---------------------------------------------------------------- the printed wheel

def test_the_printed_wheel_is_the_section_extruded(win):
    outer, bore = gears.spur_gear(20, module=2.0, bore=5.0)
    win.make_spur_gear_model(SOLID)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no wheel in the model"
    g = win.doc.groups[0]
    assert g.name == "Spur Gear"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() == pytest.approx(section(outer, bore).area() * THICKNESS, rel=1e-6)
    assert m.genus() == 1, "the bore must be a hole through the wheel"


def test_the_printed_wheel_stands_on_the_bed(win):
    win.make_spur_gear_model(SOLID)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no wheel in the model"
    g = win.doc.groups[0]
    assert pump_until(lambda: g.bbox() is not None)
    lo, hi = np.asarray(g.bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the wheel does not stand on the bed"
    assert hi[2] == pytest.approx(THICKNESS)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the wheel is not centred on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    assert pump_until(lambda: "thick" in win.hint.text()), win.hint.text()
    assert win.hint.text() == ("Spur gear with 20 teeth added (pitch diameter 40.00 mm, "
                               "8.00 mm thick).")


def test_one_undo_takes_the_printed_wheel_away(win):
    steps = len(win.doc.undo_stack)
    win.make_spur_gear_model(SOLID)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no wheel in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the wheel is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_a_bore_that_eats_the_printed_wheel_is_a_clear_text(win):
    win.make_spur_gear_model(dict(SOLID, bore=39.0))
    assert pump_until(lambda: "bore" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_printed_gear_follows_the_clearance_test_in_print_tools(win):
    labels = [clean(a.text()) for a in win.print_tools_menu.actions() if not a.isSeparator()]
    assert labels.index("Spur Gear") == labels.index("Clearance Test") + 1


def test_the_print_dialog_starts_at_the_last_thickness_and_remembers_it(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins]
        spins[-1].setValue(12.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.gear_module, win.settings.gear_angle = 1.0, 14.5
    win.settings.gear_bore, win.settings.gear_backlash = 4.0, 0.05
    win.settings.gear_thickness = 6.0
    win.spur_gear_dialog(solid=True)
    assert seen["start"] == [1.0, 14.5, 4.0, 0.05, 6.0]
    assert pump_until(lambda: len(win.doc.groups) == 1), "no wheel in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] - lo[2] == pytest.approx(12.0), "the wheel is not as thick as asked"
    assert win.settings.gear_thickness == pytest.approx(12.0)
    assert app_module.Settings().gear_thickness == pytest.approx(12.0)


def test_the_worksheet_dialog_asks_no_thickness(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        seen["options"] = dlg.options()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.spur_gear_dialog()
    assert seen["spins"] == 4, "the worksheet dialog has no thickness to ask"
    assert "thickness" not in seen["options"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
