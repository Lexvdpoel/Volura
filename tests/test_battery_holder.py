"""Battery holders: a printed block that keeps round cells side by side.

`battery_holder.battery_holder` gives the holder as one closed volume that stands on `z = 0` and is
centred on the origin in plan: a block in which `count` cradles of `d + 2 clearance` millimetres
across are cut out of the top, `L + 2 clearance` millimetres long, open at the top so a cell presses
in from above, with `wall` millimetres of plastic between the cradles, around them and under them.
A slot goes through the end wall at both ends of every cradle for a spring contact or a wire, and a
hole through the floor at each end of the holder.

These tests weigh a holder against what it promises: it is one volume that stands on the bed, the
box round it is the size asked for, and a cell of the type asked for fits in every cradle without
touching it. A size that cannot be printed becomes the nearest one that can, and a cell that is not
known is refused. The place in the interface is checked too (Parts right after the hinge, one group
in one undo step, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_battery_holder.py
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
from volura import battery_holder  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

CELL, D, L = "AA", 14.5, 50.5
BASE = dict(cell=CELL, count=2, wall=1.6, clearance=0.4)
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
    return battery_holder.battery_holder(**dict(BASE, **kw))


def cell(size=CELL, n=2, wall=1.6, clearance=0.4, cradle=0):
    """A bare cell of `size` as it lies in cradle `cradle` of a holder for `n` cells."""
    d, ln = battery_holder.CELLS[size]
    y = (cradle - 0.5 * (n - 1)) * (d + 2.0 * clearance + wall)
    return (mf.Manifold.cylinder(ln, 0.5 * d, 0.5 * d, 64).rotate((0.0, 90.0, 0.0))
            .translate((-0.5 * ln, y, wall + 0.5 * d + clearance)))


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


# ------------------------------------------------------------------ the holder

def test_the_default_holder_is_one_valid_volume_that_stands_on_the_bed():
    m = make()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() > 0.0, "the holder is not a volume"
    lo, _hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the holder does not stand on the bed"


def test_the_box_round_the_holder_is_the_size_asked():
    d, c, t, n = D, BASE["clearance"], BASE["wall"], BASE["count"]
    lo, hi = bbox(make())
    assert hi[0] - lo[0] == pytest.approx(L + 2 * c + 2 * t), "the cradle is not the length asked"
    assert hi[1] - lo[1] == pytest.approx(n * (d + 2 * c) + (n + 1) * t), \
        "the cradles are not side by side"
    assert hi[2] - lo[2] == pytest.approx(t + 0.5 * d + c), "the cradle is not open at the top"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the holder is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the holder is not centred"
    block = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    assert make().volume() < 0.7 * block, "the cradles are not cut out of the top"


def test_a_cell_presses_into_every_cradle_without_touching_it():
    for n in (1, 3):
        holder = make(count=n)
        for i in range(n):
            gap = mf.Manifold.batch_boolean([holder, cell(n=n, cradle=i)],
                                            mf.OpType.Intersect).volume()
            assert gap == pytest.approx(0.0, abs=1e-6), f"cell {i} of {n} does not fit"


def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can():
    d, c, t = D, battery_holder.CLEARANCE_MAX, battery_holder.WALL_MAX
    m = battery_holder.battery_holder(cell=CELL, count=99, wall=99.0, clearance=99.0)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(L + 2 * c + 2 * t)
    assert hi[1] - lo[1] == pytest.approx(battery_holder.COUNT_MAX * (d + 2 * c)
                                          + (battery_holder.COUNT_MAX + 1) * t)
    assert hi[2] - lo[2] == pytest.approx(t + 0.5 * d + c)
    one = bbox(battery_holder.battery_holder(cell=CELL, count=0, wall=0.1, clearance=0.0))
    lo, hi = one
    assert hi[1] - lo[1] == pytest.approx(d + 2 * battery_holder.CLEARANCE_MIN
                                          + 2 * battery_holder.WALL_MIN)
    assert hi[0] - lo[0] == pytest.approx(L + 2 * battery_holder.CLEARANCE_MIN
                                          + 2 * battery_holder.WALL_MIN)


def test_a_cell_that_is_not_known_is_refused():
    with pytest.raises(ValueError):
        battery_holder.battery_holder(cell="D")


# ------------------------------------------------------------------ in the window

def test_the_holder_is_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_battery_holder_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no holder in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the holder is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Battery holder"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(make().volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_holder_stands_on_the_bed_next_to_the_model(win):
    win.make_battery_holder_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no holder in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    bx, by, _bz = win.settings.bed_size
    assert lo[2] == pytest.approx(0.0), "the holder does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the holder is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    assert pump_until(lambda: "cells added" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Holder for 2 AA cells added."
    win.make_battery_holder_model(dict(OPTIONS, count=1))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second holder is missing"
    assert pump_until(lambda: "cell holder" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "1 cell holder for AA added."
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the holders are not 10 mm apart"


def test_the_holder_follows_the_hinge_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Battery Holder") == labels.index("Print-in-Place Hinge") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [dlg.cell.currentData(), int(dlg.count.value()),
                         round(dlg.wall.value(), 2), round(dlg.clearance.value(), 2)]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        dlg.cell.setCurrentIndex(dlg.cell.findData("18650"))
        dlg.count.setValue(3)
        dlg.wall.setValue(2.0)
        dlg.clearance.setValue(0.5)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.battery_cell, win.settings.battery_count = "AAA", 4
    win.settings.battery_wall, win.settings.battery_clearance = 1.4, 0.3
    win.battery_holder_dialog()
    assert seen["title"] == "Battery holder"
    assert seen["start"] == ["AAA", 4, 1.4, 0.3]
    assert seen["spins"] == 2
    assert pump_until(lambda: len(win.doc.groups) == 1), "no holder in the model"
    _d, ln = battery_holder.CELLS["18650"]
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(ln + 2 * 0.5 + 2 * 2.0), "the length is not as asked"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 1, "the holder is not one part"
    assert win.settings.battery_cell == "18650"
    s = app_module.Settings()
    assert (s.battery_cell, s.battery_count, s.battery_wall,
            s.battery_clearance) == ("18650", 3, pytest.approx(2.0), pytest.approx(0.5))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
