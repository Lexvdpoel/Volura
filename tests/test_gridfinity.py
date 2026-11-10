"""Gridfinity: the open modular storage standard of bins that stand on a plate and stack.

`gridfinity.gridfinity_bin` gives a bin as one closed volume that stands on `z = 0` and is centred
on the origin in plan: `cols` by `rows` squares of the grid of 42 mm and `units` height units of
7 mm, on a foot under every square, hollowed above a floor of `floor` with walls of `wall`, with
compartments made by straight dividers that stop under the top edge, four magnet pockets in the
bottom of every foot when asked for, and the stacking lip on top of the wall.
`gridfinity.gridfinity_baseplate` gives the plate that carries the grid: in every square a hole of
the same profile as the foot of a bin, upside down and open at the bottom.

These tests weigh the two against each other, which is what the standard really promises: a bin is
one volume of the size it says, a bin drops into a plate without touching it but cannot go lower,
a bin of two by two stands on a plate of three by three where the grid puts it and not half a
square beside it, a bin stands in the lip of the bin under it, and the plastic the magnet pockets
cut out is the plastic of four magnets per square. A size that cannot be printed becomes the
nearest one that can. A finger scoop stands in the corner between the floor and the front wall and
a label shelf under the top of the back wall, and a bin with both is still one volume of the size
it says, that stands in the lip of the bin under it.

Run with:  python -m pytest tests/test_gridfinity.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import gridfinity  # noqa: E402

#: mm³ that two parts may still share and count as apart: where a foot rests on a chamfer they
#: touch, and the mesh leaves a sliver of rounding behind
FIT = 1e-3
GRID, UNIT = gridfinity.GRID, gridfinity.HEIGHT_UNIT


def span(m):
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[3:] - b[:3]


def overlap(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


# ------------------------------------------------------------------ the bin

def test_a_bin_is_one_volume_of_the_size_it_says():
    m = gridfinity.gridfinity_bin(cols=2, rows=1, units=3)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert span(m) == pytest.approx([2 * GRID - 0.5, GRID - 0.5, 3 * UNIT + 4.4])
    assert np.asarray(m.bounding_box())[2] == pytest.approx(0.0)
    plain = gridfinity.gridfinity_bin(cols=2, rows=1, units=3, lip=False)
    assert span(plain)[2] == pytest.approx(3 * UNIT), "without the lip the bin is 21 high"


def test_a_bin_stands_in_a_baseplate_without_touching_it_and_no_lower():
    m, plate = gridfinity.gridfinity_bin(1, 1, 3), gridfinity.gridfinity_baseplate(1, 1)
    assert span(plate) == pytest.approx([GRID, GRID, 4.65])
    assert overlap(m, plate) < FIT
    assert overlap(m.translate((0.0, 0.0, -0.3)), plate) > 1.0, "the foot goes through the plate"


@pytest.mark.parametrize("corner", [(1, 1), (-1, -1), (1, -1)])
def test_a_bin_of_two_by_two_stands_on_a_plate_of_three_by_three(corner):
    m = gridfinity.gridfinity_bin(cols=2, rows=2, units=3)
    plate = gridfinity.gridfinity_baseplate(3, 3)
    # the plate has its holes on -42, 0, 42; the feet of the bin are 21 from its middle
    at = m.translate((corner[0] * 0.5 * GRID, corner[1] * 0.5 * GRID, 0.0))
    assert overlap(at, plate) < FIT
    # a quarter of a square beside the grid, the feet land on the rims between the holes
    assert overlap(at.translate((0.25 * GRID, 0.0, 0.0)), plate) > 1.0


def test_a_bin_stands_in_the_lip_of_the_bin_under_it():
    m = gridfinity.gridfinity_bin(1, 1, 3)
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT))) < FIT
    # the foot comes to rest on the chamfers of the lip, a little under the nominal height
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT - 0.5))) > 1.0, "nothing carries the bin"
    # and the lip holds it in place sideways
    assert overlap(m, m.translate((1.0, 0.0, 3 * UNIT - 0.3))) > 1.0, "the bin slides off"


@pytest.mark.parametrize("cols, rows", [(2, 1), (1, 2), (2, 2), (3, 2)])
def test_nothing_crosses_the_opening_of_a_bin_of_more_squares(cols, rows):
    m = gridfinity.gridfinity_bin(cols, rows, 3)
    room = 2.0 * (gridfinity.NECK_CHAMFER + 1.0)
    probe = mf.Manifold.cube([cols * GRID - 0.5 - room, rows * GRID - 0.5 - room,
                              gridfinity.LIP_HEIGHT + 2.0]).translate(
        (-0.5 * (cols * GRID - 0.5 - room), -0.5 * (rows * GRID - 0.5 - room), 3 * UNIT - 1.0))
    assert overlap(m, probe) < FIT, "a bar crosses the opening"


def test_a_bin_of_two_squares_stands_in_the_lip_of_one_like_it():
    m = gridfinity.gridfinity_bin(2, 1, 3)
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT))) < FIT
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT - 0.5))) > 1.0, "nothing carries the bin"
    assert overlap(m, m.translate((1.0, 0.0, 3 * UNIT - 0.3))) > 1.0, "the bin slides off"


def test_a_small_bin_stands_on_a_larger_one():
    big, small = gridfinity.gridfinity_bin(2, 1, 3), gridfinity.gridfinity_bin(1, 1, 3)
    for x in (-0.5 * GRID, 0.5 * GRID):
        assert overlap(big, small.translate((x, 0.0, 3 * UNIT))) < FIT


def test_magnet_pockets_cut_the_plastic_of_four_magnets():
    for cols, rows in ((1, 1), (2, 1)):
        plain = gridfinity.gridfinity_bin(cols, rows, 3, magnets=False)
        magnet = gridfinity.gridfinity_bin(cols, rows, 3, magnets=True)
        want = cols * rows * 4.0 * math.pi * (0.5 * gridfinity.MAGNET_DIA) ** 2
        want *= gridfinity.MAGNET_DEPTH
        assert plain.volume() - magnet.volume() == pytest.approx(want, rel=0.01)


def test_the_dividers_stand_between_the_compartments_and_under_the_edge():
    plain = gridfinity.gridfinity_bin(1, 1, 3, divisions=1)
    split = gridfinity.gridfinity_bin(1, 1, 3, divisions=3)
    inner = gridfinity.CELL - 2.0 * 1.2
    x = -0.5 * inner + inner / 3.0                       # where the first divider stands
    middle = mf.Manifold.cube([0.6, 10.0, 4.0]).translate((x - 0.3, -5.0, 10.0))
    assert overlap(plain, middle) == pytest.approx(0.0, abs=FIT)
    assert overlap(split, middle) == pytest.approx(middle.volume(), rel=1e-3)
    above = mf.Manifold.cube([0.6, 10.0, 0.5]).translate((x - 0.3, -5.0, 3 * UNIT - 0.6))
    assert overlap(split, above) == pytest.approx(0.0, abs=FIT), "the divider reaches the edge"


def test_sizes_that_cannot_be_printed_become_the_nearest_that_can():
    assert span(gridfinity.gridfinity_bin(cols=99, rows=0, units=3))[:2] == pytest.approx(
        [10 * GRID - 0.5, GRID - 0.5])
    assert span(gridfinity.gridfinity_bin(1, 1, units=0, lip=False))[2] == pytest.approx(2 * UNIT)
    assert span(gridfinity.gridfinity_bin(1, 1, units=99, lip=False))[2] == pytest.approx(20 * UNIT)
    m = gridfinity.gridfinity_bin(1, 1, 3, wall=-5, floor=-5)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert span(gridfinity.gridfinity_baseplate(cols=0, rows=99))[:2] == pytest.approx(
        [GRID, 10 * GRID])


def test_the_finger_scoop_fills_the_corner_between_the_floor_and_the_front_wall():
    plain = gridfinity.gridfinity_bin(1, 1, 3)
    scooped = gridfinity.gridfinity_bin(1, 1, 3, scoop=True)
    inner = gridfinity.CELL - 2.0 * 1.2
    floor = gridfinity.BASE_HEIGHT + 1.0
    corner = mf.Manifold.cube([1.0, 1.0, 1.0]).translate((-0.5, -0.5 * inner + 1.5, floor + 0.5))
    assert overlap(plain, corner) < FIT
    assert overlap(scooped, corner) == pytest.approx(corner.volume(), rel=1e-3), "no scoop"
    middle = mf.Manifold.cube([1.0, 1.0, 1.0]).translate((-0.5, -0.5, floor + 0.5))
    assert overlap(scooped, middle) < FIT, "the scoop runs over the whole floor"


def test_the_label_shelf_stands_under_the_top_of_the_back_wall():
    plain = gridfinity.gridfinity_bin(1, 1, 3)
    tabbed = gridfinity.gridfinity_bin(1, 1, 3, label=True)
    inner, H = gridfinity.CELL - 2.0 * 1.2, 3 * UNIT
    under = mf.Manifold.cube([1.0, 1.0, 1.0]).translate((-0.5, 0.5 * inner - 5.5, H - 2.5))
    assert overlap(plain, under) < FIT
    assert overlap(tabbed, under) == pytest.approx(under.volume(), rel=1e-3), "no label shelf"
    d = gridfinity.LABEL_DEPTH
    far = mf.Manifold.cube([inner - 6.0, inner - d - 5.5, d - 1.0]).translate(
        (-0.5 * (inner - 6.0), -0.5 * inner + 3.0, H - d))
    assert overlap(tabbed, far) < FIT, "the shelf hangs over the opening"


def test_a_bin_with_a_scoop_and_a_label_shelf_still_stands_in_the_lip():
    m = gridfinity.gridfinity_bin(2, 1, 3, divisions=2, scoop=True, label=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert span(m) == pytest.approx([2 * GRID - 0.5, GRID - 0.5, 3 * UNIT + 4.4])
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT))) < FIT
    assert overlap(m, m.translate((0.0, 0.0, 3 * UNIT - 0.5))) > 1.0, "nothing carries the bin"
    floor = gridfinity.BASE_HEIGHT + 1.0
    divider = mf.Manifold.cube([1.0, 1.0, 1.0]).translate((-0.5, -0.5, floor + 0.5))
    assert overlap(m, divider) == pytest.approx(divider.volume(), rel=1e-3), \
        "the divider does not reach the floor"


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_baseplate_and_remembers_it(monkeypatch, tmp_path):
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData("baseplate"))
        assert not dlg.units.isEnabled(), "a baseplate has no height units"
        dlg.cols.setValue(3)
        dlg.rows.setValue(2)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.gridfinity_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no baseplate in the model"
        assert span(w.doc.groups[0].ents.to_manifold())[:2] == pytest.approx([3 * GRID, 2 * GRID])
        s = app_module.Settings()
        assert (s.grid_kind, s.grid_cols, s.grid_rows) == ("baseplate", 3, 2)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Gridfinity") == labels.index("Cable Chain") + 1
    finally:
        w.settings.grid_kind = "bin"
        w.settings.save()
        w.doc.dirty = False
        w.close()


def test_the_dialog_greys_out_and_remembers_the_scoop_and_the_label_shelf(monkeypatch, tmp_path):
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData("baseplate"))
        assert not dlg.scoop.isEnabled(), "a baseplate has no finger scoop"
        assert not dlg.label.isEnabled(), "a baseplate has no label shelf"
        dlg.kind.setCurrentIndex(dlg.kind.findData("bin"))
        assert dlg.scoop.isEnabled() and dlg.label.isEnabled(), "a bin has both of them"
        dlg.scoop.setChecked(True)
        dlg.label.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.gridfinity_dialog()
        deadline = time.monotonic() + 20.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no bin in the model"
        s = app_module.Settings()
        assert (s.grid_scoop, s.grid_label) == (True, True)
    finally:
        w.settings.grid_scoop = w.settings.grid_label = False
        w.settings.save()
        w.doc.dirty = False
        w.close()
