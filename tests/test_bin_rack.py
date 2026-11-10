"""Bin wall rack: a plate on the wall with a rail in every row, and the hook a bin hangs on it with.

`bin_rack.bin_rack` gives a plate of `cols` by `rows` cells, lying on its back, with a rail at the
top of every row that leans 45 degrees up, so a groove opens to the top between the rail and the
plate, and countersunk screw holes. `storage_bin(hanger=True)` gives a bin a hook along the top of
its back wall, down at 45 degrees, and a rib along the bottom.

These tests hang a bin on the rack the way it hangs on the wall: the hook lies on the rail without
going into it, a bin that drops a little would go into the rail (so it rests on it), a bin that is
pulled away from the wall hits the rail (so it cannot fall off forwards), and two bins in rows
above each other do not touch. The place in the interface is checked too.

Run with:  python -m pytest tests/test_bin_rack.py
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
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import bin_rack, storage_bin  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

RACK = dict(cols=3, rows=2, cell_w=85.0, cell_h=50.0, plate=4.0, screw="M4")
BIN = dict(length=80.0, width=60.0, height=40.0)


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def hung(row=0, x=0.0, rack=RACK, bin_kw=BIN):
    """The rack, and a bin with the hook hung in `row` at `x`, in the coordinates of the rack.

    The rack lies on its back, so the wall is `-z` and up the wall is `+y`. The bin is turned so
    its back faces the plate, at the depth of a rail from it, with its top edge on the top of the
    row: that is where the hook says it hangs.
    """
    m = bin_rack.bin_rack(**rack)
    plate_top = bbox(m)[1][2] - storage_bin.CLEAT_OUT
    b = storage_bin.storage_bin(hanger=True, **bin_kw)
    h, n = rack["cell_h"], rack["rows"]
    M = np.array([[1.0, 0.0, 0.0, x],
                  [0.0, 0.0, 1.0, (row + 1) * h - bin_kw["height"] - 0.5 * n * h],
                  [0.0, -1.0, 0.0, plate_top + storage_bin.CLEAT_OUT + 0.5 * bin_kw["width"]]])
    return m, b.transform(M)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the rack

def test_the_rack_is_one_volume_of_the_grid_asked_for():
    m = bin_rack.bin_rack(**RACK)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the rails are not one piece with the plate"
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(3 * 85.0)
    assert hi[1] - lo[1] == pytest.approx(2 * 50.0)
    assert hi[2] - lo[2] == pytest.approx(4.0 + storage_bin.CLEAT_OUT)
    assert lo[2] == pytest.approx(0.0), "the rack does not lie on the bed"


def test_every_row_has_two_screw_holes_and_none_without_screws():
    assert bin_rack.bin_rack(**RACK).genus() == 2 * 2
    assert bin_rack.bin_rack(**dict(RACK, cols=1, rows=3)).genus() == 3
    assert bin_rack.bin_rack(**dict(RACK, screw="none")).genus() == 0


def test_the_plate_is_thick_enough_for_the_countersink():
    lo, hi = bbox(bin_rack.bin_rack(**dict(RACK, plate=2.0)))
    assert hi[2] - lo[2] - storage_bin.CLEAT_OUT >= 2.25 + bin_rack.PLATE_UNDER_SINK - 1e-9


def test_an_unknown_screw_is_refused():
    with pytest.raises(ValueError):
        bin_rack.bin_rack(**dict(RACK, screw="M7"))


def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can():
    lo, hi = bbox(bin_rack.bin_rack(cols=0, rows=99, cell_w=1.0, cell_h=999.0, screw="none"))
    assert hi[0] - lo[0] == pytest.approx(bin_rack.CELL_W_MIN)
    assert hi[1] - lo[1] == pytest.approx(bin_rack.ROWS_MAX * bin_rack.CELL_H_MAX)


# ------------------------------------------------------------------ the hook

def test_the_hook_is_only_there_with_the_hanger_and_is_one_piece_with_the_bin():
    plain = storage_bin.storage_bin(**BIN)
    hooked = storage_bin.storage_bin(hanger=True, **BIN)
    assert hooked.volume() > plain.volume()
    assert len(hooked.decompose()) == 1
    lo, hi = bbox(hooked)
    assert hi[1] - lo[1] == pytest.approx(BIN["width"] + storage_bin.CLEAT_OUT - storage_bin.CLEAT_PLAY)
    assert hi[2] - lo[2] == pytest.approx(BIN["height"]), "the hook sticks out over the top"
    assert lo[2] == pytest.approx(0.0)


def test_a_bin_hangs_on_the_rail_without_going_into_it():
    rack, b = hung()
    assert common(rack, b) == pytest.approx(0.0, abs=1e-6)


def test_a_bin_rests_on_the_rail():
    rack, b = hung()
    assert common(rack, b.translate((0.0, -0.1, 0.0))) > 1.0, "nothing holds the bin up"


def test_a_bin_cannot_be_pulled_off_forwards():
    rack, b = hung()
    assert common(rack, b.translate((0.0, 0.0, 1.0))) > 1.0, "the hook comes off the rail"


def test_the_bin_hangs_upright_against_the_plate():
    rack, b = hung()
    assert common(rack, b.translate((0.0, 0.0, -storage_bin.CLEAT_PLAY - 0.1))) > 0.0, \
        "the rib does not reach the plate"


def test_bins_in_two_rows_and_next_to_each_other_do_not_touch():
    _rack, low = hung(row=0, x=-85.0)
    _rack, high = hung(row=1, x=-85.0)
    _rack, side = hung(row=0, x=0.0)
    assert common(low, high) == pytest.approx(0.0, abs=1e-6)
    assert common(low, side) == pytest.approx(0.0, abs=1e-6)


def test_a_bin_as_high_as_a_row_still_fits():
    tall = dict(BIN, height=RACK["cell_h"])
    _rack, low = hung(row=0, bin_kw=tall)
    rack, high = hung(row=1, bin_kw=tall)
    assert common(low, high) == pytest.approx(0.0, abs=1e-6)
    assert common(rack, low) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ stops and panels

def test_a_bin_slid_to_the_end_of_a_rail_hits_the_stop():
    hook_end = 0.5 * BIN["length"] - BIN.get("radius", 4.0)      # the hook stops at the corner
    x = -(1.5 * RACK["cell_w"]) + hook_end - 1.0                  # 1 mm past the end of the rail
    rack, b = hung(x=x)
    assert common(rack, b) > 1.0, "the bin slides off the end of the rail"
    rack, b = hung(x=x, rack=dict(RACK, stops=False))
    assert common(rack, b) == pytest.approx(0.0, abs=1e-6)


def test_the_columns_are_shared_over_the_panels():
    assert bin_rack.panel_columns(5, 2) == [2, 2, 1]
    assert bin_rack.panel_columns(4, 2) == [2, 2]
    assert bin_rack.panel_columns(3, 0) == [3]
    assert bin_rack.panel_columns(3, 3) == [3]


def panels(cols=4, panel_cols=2):
    """The panels of a rack, placed next to each other the way they slide together."""
    kw = dict(RACK, cols=cols)
    kw.pop("stops", None)
    parts = bin_rack.bin_rack_panels(panel_cols=panel_cols, **kw)
    placed, x = [], 0.0
    for n, m in zip(bin_rack.panel_columns(cols, panel_cols), parts):
        placed.append(m.translate((x + 0.5 * n * RACK["cell_w"], 0.0, 0.0)))
        x += n * RACK["cell_w"]
    return placed


def test_every_panel_is_one_piece_with_its_screw_holes():
    parts = bin_rack.bin_rack_panels(panel_cols=2, **dict(RACK, cols=5))
    assert len(parts) == 3
    assert all(len(m.decompose()) == 1 for m in parts)
    assert [m.genus() for m in parts] == [4, 4, 2]


def test_the_panels_slide_together_without_going_into_each_other():
    a, b = panels()
    assert common(a, b) == pytest.approx(0.0, abs=1e-6)
    assert common(a, b.translate((0.0, 0.0, 1.0))) == pytest.approx(0.0, abs=1e-6), \
        "the panel does not slide in from the front"


def test_the_dovetails_hold_the_panels_together_along_the_wall():
    a, b = panels()
    assert common(a, b.translate((1.0, 0.0, 0.0))) > 1.0, "the panels pull apart sideways"
    assert common(a, b.translate((0.0, 1.0, 0.0))) > 1.0, "the panels slide apart up the wall"


def test_only_the_outer_ends_of_a_rack_in_panels_have_stops():
    a, b = panels()
    seam = 2 * RACK["cell_w"]
    y_top = RACK["cell_h"] - 0.5 * RACK["rows"] * RACK["cell_h"]
    top = bbox(a)[1][2]
    # just next to the seam, on the rail level where a stop would be: nothing in the way
    probe = mf.Manifold.cube([2.0 * bin_rack.STOP_LONG, 2.0, 1.0]).translate(
        (seam - bin_rack.STOP_LONG, y_top - 3.0, top - 1.5))
    assert common(a, probe) + common(b, probe) == pytest.approx(0.0, abs=1e-6)
    outer = probe.translate((-seam, 0.0, 0.0))
    assert common(a, outer) > 0.0, "the outer end has no stop"


# ------------------------------------------------------------------ the interface

def test_the_rack_follows_the_bin_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Bin Wall Rack") == labels.index("Storage Bin") + 1


def test_the_dialog_remembers_and_builds_one_part(win, monkeypatch):
    def fill(dlg):
        dlg.cols.setValue(2)
        dlg.rows.setValue(3)
        dlg.cell_w.setValue(70.0)
        dlg.cell_h.setValue(45.0)
        dlg.screw.setCurrentIndex(dlg.screw.findData("M3"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.bin_rack_dialog()
    assert pump_until(lambda: len(win.doc.groups) == 1), "no rack in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert (hi - lo)[:2] == pytest.approx([140.0, 135.0])
    assert len(win.doc.undo_stack) >= 1
    s = app_module.Settings()
    assert (s.rack_cols, s.rack_rows, s.rack_cell_w, s.rack_cell_h, s.rack_screw) == (
        2, 3, pytest.approx(70.0), pytest.approx(45.0), "M3")


def test_the_bin_dialog_has_the_wall_hook(win, monkeypatch):
    def fill(dlg):
        dlg.hanger.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.bin_hanger = False
    win.storage_bin_dialog()
    assert pump_until(lambda: len(win.doc.groups) == 1), "no bin in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[1] - lo[1] > win.settings.bin_width, "the bin has no hook"
    assert app_module.Settings().bin_hanger is True
    win.settings.bin_hanger = False           # the other tests make bins without a hook
    win.settings.save()


def test_a_rack_in_panels_is_a_part_per_panel(win, monkeypatch):
    def fill(dlg):
        dlg.cols.setValue(4)
        dlg.panel_cols.setValue(2)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.bin_rack_dialog()
    assert pump_until(lambda: len(win.doc.groups) == 2), "not a part per panel"
    assert len(win.doc.undo_stack) == 1, "the panels are not one undo step"
    win.settings.rack_panel_cols = 0          # the other tests make a rack in one piece
    win.settings.save()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
