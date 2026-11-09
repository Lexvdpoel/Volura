"""Standoff: a printable spacer that holds a PCB or a panel off the surface under it.

`standoff.standoff` gives the standoff as one closed volume that stands on `z = 0` and is
centred on the origin, `height` millimetres high and `outer` millimetres wide: `round` a
cylinder, `hex` a hexagon of `outer` across the flats. The mount is how the screw is held:
`through` a clearance hole all the way, `tap` a self-tapping pilot hole of 0.85 of the nominal
diameter and `insert` a hole for a heat-set insert from the top with the clearance hole under
it. `count` puts a row of them along X, `outer + 4` mm apart, as one part.

These tests weigh every shape against every mount: the plastic a hole takes out is the plastic
the hole promises, the insert hole really is wider at the top than below it, a row really is
that many bodies at the right distance, and the input that cannot be printed is refused. The
place in the interface is checked too (Print Tools right after the knob, one group in one undo
step to the right of the model, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_standoff.py
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
from volura import standoff, threads  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

H, O = 10.0, 8.0
BASE = dict(height=H, outer=O, shape="round", bore="M3", kind="through", clearance=0.2,
            count=1)
OPTIONS = dict(BASE)
#: the holes and the hexagon are polygons, so a standoff is at most this far off
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


def standoff_at(**kw):
    return standoff.standoff(**dict(BASE, **kw))


def polygon_area(radius, segments=standoff.SEGMENTS):
    """The area of the inscribed polygon a round body or a round hole is made of."""
    return 0.5 * segments * radius * radius * math.sin(2.0 * math.pi / segments)


def hex_area(across_flats):
    """The area of the hexagon a hexagonal body is: `across_flats` between the flats."""
    return math.sqrt(3.0) / 2.0 * across_flats ** 2


def body_area(shape, outer):
    """The area of the standoff from above, without any hole."""
    return hex_area(outer) if shape == "hex" else polygon_area(0.5 * outer)


def insert_depth(bore, height=H):
    """How deep the hole of a heat-set insert of `bore` goes from the top face."""
    return min(height - standoff.INSERT_ROOM,
               standoff.INSERT_DEPTH * float(bore[1:]) + standoff.INSERT_EXTRA)


def removed_volume(shape, kind, outer=O, height=H, bore="M3", clearance=0.2):
    """The plastic the mount of `kind` cuts out of a plain body."""
    hole_d = threads.SCREW_HOLES[bore][0]
    if kind == "tap":
        hole = polygon_area(0.5 * standoff.TAP_FACTOR * float(bore[1:]))
        return hole * height
    hole = polygon_area(0.5 * (hole_d + clearance))
    if kind == "insert":
        top = polygon_area(0.5 * standoff.INSERTS[bore])
        return hole * height + (top - hole) * insert_depth(bore, height)
    return hole * height


def slice_area(m, z, outer=O):
    """The plastic of a standoff in a thin slice at `z`, as a volume per mm of height."""
    side = 3.0 * outer
    slab = mf.Manifold.cube([side, side, 0.2]).translate((-0.5 * side, -0.5 * side, z - 0.1))
    return (m ^ slab).volume() / 0.2


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


# ------------------------------------------------------------------ the standoff

@pytest.mark.parametrize("shape", standoff.STANDOFF_SHAPES)
@pytest.mark.parametrize("kind", standoff.STANDOFF_KINDS)
def test_every_shape_and_mount_is_a_valid_standoff_on_the_bed(shape, kind):
    m = standoff_at(shape=shape, kind=kind)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the standoff does not stand on the bed"
    assert hi[2] == pytest.approx(H), "the standoff is not as high as asked"
    assert hi[0] == pytest.approx(0.5 * O / math.cos(math.pi / 6.0) if shape == "hex"
                                  else 0.5 * O, abs=0.05)
    assert hi[1] - lo[1] == pytest.approx(O, abs=0.05), "a hexagon is measured across flats"
    assert m.genus() == 1, "the screw cannot go through"


def test_a_round_standoff_is_a_cylinder_of_the_size_asked():
    m = standoff_at(kind="through")
    lo, hi = bbox(m)
    assert [hi[0] - lo[0], hi[1] - lo[1]] == pytest.approx([O, O])
    assert m.volume() == pytest.approx(body_area("round", O) * H
                                       - removed_volume("round", "through"), rel=1e-3)


def test_a_hexagonal_standoff_is_a_hexagon_across_the_flats():
    m = standoff_at(shape="hex", kind="tap")
    lo, hi = bbox(m)
    assert hi[1] - lo[1] == pytest.approx(O, abs=0.05)
    assert hi[0] - lo[0] == pytest.approx(O / math.cos(math.pi / 6.0), abs=0.05)
    assert m.volume() == pytest.approx(body_area("hex", O) * H
                                       - removed_volume("hex", "tap"), rel=1e-3)


@pytest.mark.parametrize("shape", standoff.STANDOFF_SHAPES)
def test_every_mount_takes_the_plastic_it_promises(shape):
    for kind in standoff.STANDOFF_KINDS:
        m = standoff_at(shape=shape, kind=kind)
        want = body_area(shape, O) * H - removed_volume(shape, kind)
        assert m.volume() == pytest.approx(want, rel=REL), kind


def test_the_through_hole_is_open_and_of_the_clearance_size():
    m = standoff_at(kind="through")
    hole_d = threads.SCREW_HOLES["M3"][0]
    assert m.slice(0.5 * H).area() == pytest.approx(polygon_area(0.5 * O)
                                                    - polygon_area(0.5 * (hole_d + 0.2)),
                                                    rel=REL)
    top = mf.Manifold.cylinder(1.0, 1.0, 1.0, 16).translate((0.0, 0.0, H - 1.0))
    assert (m ^ top).volume() == pytest.approx(0.0, abs=1e-6), "the screw cannot come out"


def test_the_self_tapping_pilot_is_smaller_than_the_clearance_hole():
    tap = standoff_at(kind="tap")
    through = standoff_at(kind="through")
    assert tap.volume() > through.volume(), "a pilot hole may not be the bigger hole"
    assert tap.slice(0.5 * H).area() == pytest.approx(
        polygon_area(0.5 * O) - polygon_area(0.5 * standoff.TAP_FACTOR * 3.0), rel=REL)


def test_the_insert_hole_is_wider_at_the_top_than_below_it():
    m = standoff_at(kind="insert")
    depth = insert_depth("M3")
    above = slice_area(m, H - 0.5 * depth)
    below = slice_area(m, 0.5 * (H - depth))
    assert above < below, "the insert does not fit in its hole"
    assert below == pytest.approx(slice_area(standoff_at(kind="through"), 0.5 * (H - depth)),
                                  rel=REL), "the hole under the insert is not the clearance hole"
    assert slice_area(m, H - depth - 0.2) == pytest.approx(below, rel=REL)


@pytest.mark.parametrize("bore", sorted(standoff.INSERTS, key=lambda s: float(s[1:])))
def test_an_insert_hole_is_as_deep_as_the_insert_is_long(bore):
    m = standoff_at(bore=bore, kind="insert", outer=12.0, height=20.0)
    depth = insert_depth(bore, 20.0)
    wide = standoff.INSERTS[bore]
    assert slice_area(m, 20.0 - 0.5 * depth, 12.0) < slice_area(m, 0.5 * (20.0 - depth), 12.0)
    assert slice_area(m, 20.0 - 0.5 * depth, 12.0) == pytest.approx(polygon_area(0.5 * 12.0)
                                                                    - polygon_area(0.5 * wide),
                                                                    rel=REL)


def test_a_row_is_that_many_standoffs_at_the_right_distance():
    n = 4
    m = standoff_at(count=n)
    parts = m.decompose()
    assert len(parts) == n, "the row is not one standoff per piece"
    spacing = O + standoff.ROW_ROOM
    xs = sorted(float(np.asarray(p.bounding_box())[0]) for p in parts)
    assert [x - xs[0] for x in xs] == pytest.approx([i * spacing for i in range(n)])
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(H)
    assert hi[0] - lo[0] == pytest.approx((n - 1) * spacing + O)
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the row is not centred"
    assert m.volume() == pytest.approx(n * standoff_at().volume(), rel=1e-3)


@pytest.mark.parametrize("kw, wrong", (
    ({"height": 1.0}, "height"), ({"height": 120.0}, "height"),
    ({"outer": 2.0}, "outer size"), ({"outer": 60.0}, "outer size"),
    ({"shape": "square"}, "shape"), ({"shape": ""}, "shape"),
    ({"kind": "thread"}, "mount"), ({"kind": None}, "mount"),
    ({"bore": "M9"}, "screw size"), ({"bore": "M10"}, "screw size"),
    ({"bore": "M6", "kind": "insert"}, "heat-set insert"),
    ({"count": 0}, "number of standoffs"), ({"count": 13}, "number of standoffs"),
    ({"clearance": -0.1}, "clearance"), ({"clearance": 1.5}, "clearance")))
def test_impossible_standoffs_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        standoff_at(**kw)


@pytest.mark.parametrize("kw, wrong", (
    ({"shape": "round", "kind": "through", "bore": "M4", "outer": 4.0}, "hole does not fit"),
    ({"shape": "round", "kind": "tap", "bore": "M5", "outer": 5.0}, "hole does not fit"),
    ({"shape": "hex", "kind": "insert", "bore": "M3", "outer": 5.0}, "hole does not fit"),
    ({"shape": "hex", "kind": "insert", "bore": "M5", "outer": 7.0}, "hole does not fit")))
def test_a_hole_that_leaves_too_little_wall_is_refused(kw, wrong):
    """The wall around the hole has to survive a screw."""
    with pytest.raises(ValueError, match=wrong):
        standoff_at(**kw)


@pytest.mark.parametrize("shape", standoff.STANDOFF_SHAPES)
@pytest.mark.parametrize("kind", standoff.STANDOFF_KINDS)
@pytest.mark.parametrize("bore", sorted(threads.SCREW_HOLES, key=lambda s: float(s[1:])))
def test_every_mount_fits_in_a_standoff_with_room(shape, kind, bore):
    if kind == "insert" and bore not in standoff.INSERTS:
        pytest.skip("no heat-set insert in that size")
    m = standoff_at(shape=shape, kind=kind, bore=bore, outer=40.0, height=100.0)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.genus() == 1


# ------------------------------------------------------------------ in the window

def test_the_standoffs_are_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_standoff_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no standoffs in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the standoffs are not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Standoff"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(standoff_at(**OPTIONS).volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_row_stands_to_the_right_of_the_model(win):
    win.make_standoff_model(dict(OPTIONS, count=4))
    assert pump_until(lambda: len(win.doc.groups) == 1), "no standoffs in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the standoffs do not stand on the bed"
    assert hi[2] == pytest.approx(H)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the row is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_standoff_model(dict(OPTIONS, count=4, height=12.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second row is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the rows are not 10 mm apart"
    assert pump_until(lambda: "standoffs of" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "4 standoffs of 12 mm added."


def test_a_row_that_cannot_be_built_is_a_clear_text(win):
    win.make_standoff_model(dict(OPTIONS, outer=4.0))
    assert pump_until(lambda: "Standoff failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_standoff_follows_the_knob_in_print_tools(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Standoff") == labels.index("Knob") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins]
        seen["count"] = dlg.findChild(QSpinBox).value()
        seen["combos"] = [cb.currentData() for cb in dlg.findChildren(QComboBox)]
        seen["title"] = dlg.windowTitle()
        spins[0].setValue(16.0)
        spins[1].setValue(10.0)
        dlg.findChild(QSpinBox).setValue(4)
        dlg.findChildren(QComboBox)[0].setCurrentIndex(
            dlg.findChildren(QComboBox)[0].findData("hex"))
        dlg.findChildren(QComboBox)[1].setCurrentIndex(
            dlg.findChildren(QComboBox)[1].findData("insert"))
        dlg.findChildren(QComboBox)[2].setCurrentIndex(
            dlg.findChildren(QComboBox)[2].findData("M4"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.standoff_height, win.settings.standoff_outer = 8.0, 5.0
    win.settings.standoff_shape, win.settings.standoff_kind = "hex", "tap"
    win.settings.standoff_bore, win.settings.standoff_count = "M2.5", 2
    win.settings.standoff_clearance = 0.3
    win.standoff_dialog()
    assert seen["title"] == "Standoff"
    assert seen["start"] == [8.0, 5.0, 0.3]
    assert seen["count"] == 2
    assert seen["combos"] == ["hex", "tap", "M2.5"]
    assert pump_until(lambda: len(win.doc.groups) == 1), "no standoffs in the model"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 4, \
        "the count did not reach the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] == pytest.approx(16.0), "the height is not as asked"
    assert hi[1] - lo[1] == pytest.approx(10.0, abs=0.05), "the shape is not as asked"
    assert win.settings.standoff_height == pytest.approx(16.0)
    assert win.settings.standoff_count == 4
    assert win.settings.standoff_shape == "hex" and win.settings.standoff_kind == "insert"
    s = app_module.Settings()
    assert (s.standoff_height, s.standoff_outer, s.standoff_shape, s.standoff_kind,
            s.standoff_bore, s.standoff_count) == (pytest.approx(16.0), pytest.approx(10.0),
                                                   "hex", "insert", "M4", 4)


def test_the_clearance_starts_at_the_one_of_the_screw_holes():
    """Without a clearance of its own the standoff takes the one this printer already proved."""
    from volura.settings import qsettings
    q = qsettings()
    saved = {k: q.value(k) for k in ("standoff_clearance", "screw_hole_clearance")}
    try:
        q.remove("standoff_clearance")
        q.setValue("screw_hole_clearance", 0.45)
        q.sync()
        assert app_module.Settings().standoff_clearance == pytest.approx(0.45)
    finally:
        for key, value in saved.items():
            q.setValue(key, value)
        q.sync()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

