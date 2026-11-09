"""Knob: a printable thumb knob that is screwed on a bolt or pushed on a shaft.

`knob.knob` gives the knob as one closed volume that stands on `z = 0` and is centred on the
origin, `height` millimetres high and `diameter` millimetres across. The rim is what you hold:
`knurl` straight ribs of 1 mm (about one rib per 2 mm of circumference), `lobes` rounded lobes
whose valleys stand at 0.7 of the diameter, or `smooth` a plain cylinder. A lobe and a rib
crest stand at 0 degrees, so a knob always reaches `diameter` on the x.

The mount is how the knob takes a part: `nut` cuts a clearance hole through the knob with a
hexagonal pocket for a nut in the bottom face, `head` a clearance hole with a round
counterbore for the head of a socket head bolt, and `shaft` a blind D hole of the shaft
diameter with one side flat. These tests weigh every grip against every mount: the plastic
that a pocket takes out is the plastic the pocket promises, the nut pocket is really
hexagonal, the shaft hole really stops short of the top, and the input that cannot be printed
is refused. The place in the interface is checked too (Print Tools after the project box, one
group in one undo step to the right of the model, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_knob.py
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QDoubleSpinBox, QSpinBox)

import volura.app as app_module  # noqa: E402
from volura import knob, threads  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

D, H = 30.0, 12.0
BASE = dict(diameter=D, height=H, grip="knurl", lobes=6, bore="M4", kind="nut",
            clearance=0.2, shaft_d=6.0, flat=0.5)
OPTIONS = dict(BASE)
#: the ribs, lobes and holes are polygons, so a knob is at most this far off
REL = 0.02


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


def knob_at(**kw):
    return knob.knob(**dict(BASE, **kw))


def polygon_area(radius, segments=knob.SEGMENTS):
    """The area of the inscribed polygon a round rim or hole is made of."""
    return 0.5 * segments * radius * radius * math.sin(2.0 * math.pi / segments)


def hex_area(across_flats):
    """The area of the hexagon a nut pocket is: the nut of `across_flats` millimetres."""
    return math.sqrt(3.0) / 2.0 * across_flats ** 2


def d_area(radius, flat):
    """The area of a D shaft hole: a circle of `radius` with a chord of `flat` off it."""
    d0 = radius - flat
    return math.pi * radius * radius - (radius * radius * math.acos(d0 / radius)
                                        - d0 * math.sqrt(radius * radius - d0 * d0))


def reference(diameter=D, height=H):
    """The knob without any hole: the smooth rim of `diameter` extruded `height` high."""
    return mf.Manifold.extrude(mf.CrossSection([knob.circle_points(0.5 * diameter,
                                                                   knob.SEGMENTS)]), height)


def removed_volume(kind, diameter=D, height=H, bore="M4", clearance=0.2, shaft_d=6.0,
                   flat=0.5):
    """The plastic the mount of `kind` cuts out of a smooth knob."""
    if kind == "shaft":
        return d_area(0.5 * (shaft_d + clearance), flat) * (height - knob.SHAFT_ROOM)
    hole_d, head_d, head_h, _cs = threads.SCREW_HOLES[bore]
    hole = polygon_area(0.5 * (hole_d + clearance))
    if kind == "nut":
        s, nut_h = threads.HEX_NUTS[bore]
        pocket, deep = hex_area(s + clearance), nut_h + knob.POCKET_ROOM
    else:
        pocket, deep = polygon_area(0.5 * (head_d + clearance)), head_h + knob.POCKET_ROOM
    # the straight hole runs through the pocket too, so that part is only counted once
    return hole * height + pocket * deep - hole * deep


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


# ------------------------------------------------------------------ the knob

@pytest.mark.parametrize("grip", knob.KNOB_GRIPS)
@pytest.mark.parametrize("kind", knob.KNOB_KINDS)
def test_every_grip_and_mount_is_a_valid_knob_on_the_bed(grip, kind):
    m = knob_at(grip=grip, kind=kind)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the knob does not stand on the bed"
    assert hi[2] == pytest.approx(H), "the knob is not as high as asked"
    assert hi[0] == pytest.approx(0.5 * D, abs=0.1), "a rib or a lobe does not reach the rim"
    assert hi[1] <= 0.5 * D + 0.1
    assert lo[1] >= -0.7 * D - 0.1, "the rim is thinner than the valleys of a lobed knob"
    assert m.volume() < reference().volume()


def test_a_smooth_knob_is_a_cylinder_of_the_size_asked():
    m = knob_at(grip="smooth", kind="shaft", shaft_d=3.0, flat=0.2)
    lo, hi = bbox(m)
    assert [hi[0] - lo[0], hi[1] - lo[1]] == pytest.approx([D, D])
    assert m.volume() == pytest.approx(reference().volume()
                                       - d_area(1.6, 0.2) * (H - knob.SHAFT_ROOM), rel=1e-3)
    assert m.genus() == 0, "a blind shaft hole leaves the knob solid"


def test_a_knurled_rim_is_a_circle_with_ribs_on_it():
    m = knob_at(grip="knurl", kind="shaft", shaft_d=3.0, flat=0.2)
    core = polygon_area(0.5 * D - knob.RIB_DEPTH) * (H - knob.SHAFT_ROOM) \
        - d_area(1.6, 0.2) * (H - knob.SHAFT_ROOM)
    smooth = reference().volume() - d_area(1.6, 0.2) * (H - knob.SHAFT_ROOM)
    assert core < m.volume() < smooth, "the ribs stand between the core and the rim"
    assert m.slice(0.5 * H).area() < polygon_area(0.5 * D)
    assert m.slice(0.5 * H).area() > polygon_area(0.5 * D - knob.RIB_DEPTH)


def test_a_lobed_rim_is_a_star_of_the_right_area():
    n = 6
    low = knob.VALLEY_OVER_RADIUS * 0.5 * D
    mid, amp = 0.5 * (0.5 * D + low), 0.5 * (0.5 * D - low)
    area = math.pi * (mid * mid + 0.5 * amp * amp)      # the cosine averages its square
    m = knob_at(grip="lobes", lobes=n, kind="shaft", shaft_d=3.0, flat=0.2)
    assert m.volume() == pytest.approx(area * H - d_area(1.6, 0.2) * (H - knob.SHAFT_ROOM),
                                       rel=0.01)
    lo, hi = bbox(m)
    assert hi[1] - lo[1] < D, "a valley of a lobed knob stands at 0.7 of the diameter"


def test_every_mount_takes_the_plastic_it_promises():
    ref = reference()
    for kind in knob.KNOB_KINDS:
        m = knob_at(grip="smooth", kind=kind)
        assert ref.volume() - m.volume() == pytest.approx(removed_volume(kind), rel=REL), kind


def test_the_nut_pocket_is_hexagonal_and_the_head_pocket_round():
    s, nut_h = threads.HEX_NUTS["M4"]
    hole_d, head_d, head_h, _cs = threads.SCREW_HOLES["M4"]
    z = 0.5 * min(nut_h, head_h)
    nut = knob_at(grip="smooth", kind="nut").slice(z)
    head = knob_at(grip="smooth", kind="head").slice(z)
    assert nut.area() == pytest.approx(polygon_area(0.5 * D) - hex_area(s + 0.2), rel=REL)
    assert head.area() == pytest.approx(polygon_area(0.5 * D)
                                        - polygon_area(0.5 * (head_d + 0.2)), rel=REL)


def test_the_bolt_hole_goes_through_and_the_shaft_hole_does_not():
    nut = knob_at(grip="smooth", kind="nut")
    shaft = knob_at(grip="smooth", kind="shaft")
    top = mf.Manifold.cylinder(1.0, 1.0, 1.0, 16).translate((0.0, 0.0, H - 1.0))
    assert (nut ^ top).volume() == pytest.approx(0.0, abs=1e-6), "the bolt cannot come out"
    assert (shaft ^ top).volume() > math.pi * 0.9, "the shaft hole is not blind"
    assert nut.genus() == 1 and shaft.genus() == 0
    assert shaft.slice(H - 0.5).area() == pytest.approx(polygon_area(0.5 * D), rel=REL), \
        "the shaft hole goes through the top"
    assert shaft.slice(0.5 * (H - knob.SHAFT_ROOM)).area() \
        == pytest.approx(polygon_area(0.5 * D) - d_area(3.1, 0.5), rel=REL)


def test_the_flat_is_on_one_side_of_the_shaft_hole():
    round_hole = knob_at(grip="smooth", kind="shaft", flat=0.0)
    flat_hole = knob_at(grip="smooth", kind="shaft", flat=0.5)
    assert flat_hole.volume() > round_hole.volume(), "a flat may not make the hole bigger"
    z = 0.5 * (H - knob.SHAFT_ROOM)
    assert flat_hole.slice(z).area() > round_hole.slice(z).area()


@pytest.mark.parametrize("kw, wrong", (
    ({"diameter": 6.0}, "diameter"), ({"diameter": 130.0}, "diameter"),
    ({"height": 2.0}, "height"), ({"height": 80.0}, "height"),
    ({"grip": "square"}, "grip"), ({"grip": ""}, "grip"),
    ({"kind": "thread"}, "mount"), ({"kind": None}, "mount"),
    ({"bore": "M9"}, "screw size"), ({"bore": "M10"}, "screw size"),
    ({"lobes": 2}, "lobes"), ({"lobes": 13}, "lobes"),
    ({"clearance": -0.1}, "clearance"), ({"clearance": 1.5}, "clearance"),
    ({"shaft_d": 0.0}, "shaft diameter"), ({"shaft_d": -3.0}, "shaft diameter"),
    ({"flat": -0.5}, "flat"), ({"kind": "shaft", "flat": 4.0}, "flat")))
def test_impossible_knobs_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        knob_at(**kw)


@pytest.mark.parametrize("grip", knob.KNOB_GRIPS)
@pytest.mark.parametrize("kind", knob.KNOB_KINDS)
@pytest.mark.parametrize("bore", sorted(threads.SCREW_HOLES, key=lambda s: float(s[1:])))
def test_every_mount_fits_in_a_knob_with_room(grip, kind, bore):
    m = knob_at(grip=grip, kind=kind, bore=bore, diameter=120.0, height=60.0)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.genus() == (0 if kind == "shaft" else 1)


@pytest.mark.parametrize("kw, wrong", (
    ({"grip": "smooth", "kind": "nut", "bore": "M8", "diameter": 10.0}, "hole does not fit"),
    ({"grip": "knurl", "kind": "nut", "bore": "M6", "diameter": 12.0}, "hole does not fit"),
    ({"grip": "lobes", "kind": "head", "bore": "M6", "diameter": 14.0}, "hole does not fit"),
    ({"grip": "smooth", "kind": "shaft", "shaft_d": 20.0, "diameter": 20.0},
     "hole does not fit"),
    ({"grip": "smooth", "kind": "nut", "bore": "M8", "height": 6.0}, "hole does not fit"),
    ({"grip": "lobes", "kind": "head", "bore": "M6", "height": 6.0}, "hole does not fit")))
def test_a_hole_that_leaves_too_little_wall_is_refused(kw, wrong):
    """The wall around the pocket and the plastic above it have to survive a hand."""
    with pytest.raises(ValueError, match=wrong):
        knob_at(**kw)


# ------------------------------------------------------------------ in the window

def test_the_knob_is_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_knob_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no knob in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the knob is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Knob"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(knob_at(**OPTIONS).volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_knob_stands_to_the_right_of_the_model(win):
    win.make_knob_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no knob in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the knob does not stand on the bed"
    assert hi[2] == pytest.approx(H)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the knob is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_knob_model(dict(OPTIONS, diameter=40.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second knob is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the knobs are not 10 mm apart"
    assert pump_until(lambda: "Knob of" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Knob of 40 mm added."


def test_a_knob_that_cannot_be_built_is_a_clear_text(win):
    win.make_knob_model(dict(OPTIONS, diameter=10.0))
    assert pump_until(lambda: "Knob failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_knob_follows_the_project_box_in_print_tools(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Knob") == labels.index("Project Box") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins]
        seen["lobes"] = dlg.findChild(QSpinBox).value()
        seen["combos"] = [cb.currentData() for cb in dlg.findChildren(QComboBox)]
        seen["title"] = dlg.windowTitle()
        spins[0].setValue(45.0)
        spins[1].setValue(20.0)
        dlg.findChild(QSpinBox).setValue(8)
        dlg.findChildren(QComboBox)[0].setCurrentIndex(
            dlg.findChildren(QComboBox)[0].findData("lobes"))
        dlg.findChildren(QComboBox)[1].setCurrentIndex(
            dlg.findChildren(QComboBox)[1].findData("shaft"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.knob_diameter, win.settings.knob_height = 25.0, 10.0
    win.settings.knob_grip, win.settings.knob_lobes = "smooth", 5
    win.settings.knob_kind, win.settings.knob_bore = "head", "M5"
    win.settings.knob_shaft_d, win.settings.knob_flat = 7.0, 0.6
    win.settings.knob_clearance = 0.3
    win.knob_dialog()
    assert seen["title"] == "Knob"
    assert seen["start"] == [25.0, 10.0, 7.0, 0.6, 0.3]
    assert seen["lobes"] == 5
    assert seen["combos"] == ["smooth", "head", "M5"]
    assert pump_until(lambda: len(win.doc.groups) == 1), "no knob in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(45.0), "the diameter is not as asked"
    assert hi[2] == pytest.approx(20.0)
    assert win.doc.groups[0].ents.to_manifold().genus() == 0, "the mount did not reach the knob"
    assert win.settings.knob_diameter == pytest.approx(45.0)
    assert win.settings.knob_lobes == 8
    assert win.settings.knob_grip == "lobes" and win.settings.knob_kind == "shaft"
    s = app_module.Settings()
    assert (s.knob_diameter, s.knob_height, s.knob_lobes, s.knob_grip, s.knob_kind,
            s.knob_bore) == (pytest.approx(45.0), pytest.approx(20.0), 8, "lobes", "shaft",
                             "M5")


def test_the_clearance_starts_at_the_one_of_the_screw_holes():
    """Without a clearance of its own the knob takes the one this printer already proved."""
    from volura.settings import qsettings
    q = qsettings()
    saved = {k: q.value(k) for k in ("knob_clearance", "screw_hole_clearance")}
    try:
        q.remove("knob_clearance")
        q.setValue("screw_hole_clearance", 0.45)
        q.sync()
        assert app_module.Settings().knob_clearance == pytest.approx(0.45)
    finally:
        for key, value in saved.items():
            q.setValue(key, value)
        q.sync()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
