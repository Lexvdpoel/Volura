"""Pegboard parts: the board of 1/4 inch holes, and the hooks, shelves and holders that hang on it.

`pegboard.pegboard_part` makes a part that hangs on two holes of `PEG_PITCH` apart: a straight upper
peg that rises through the board, so the part's own weight keeps it on, and a lower peg as deep as
the board that keeps it from swinging, with a back plate between them against the front of the
board. The part lies on its side on the bed with the pegs on the bed. `pegboard_board` is the board
itself, with a rim and screw bosses on its back. These tests weigh the parts against a real board:
every kind is one volume that stands on the bed and prints without support, the pegs pass the round
holes of a board of every thickness and stand on the pitch, `reach` makes a hook longer and a shelf
deeper, a hook rises, and the ring of a holder stands upright and takes the tool asked for.

Run with:  python -m pytest tests/test_pegboard.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, pegboard  # noqa: E402

HANGING = ("hook", "shelf", "holder")


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def block(x, y, dx, dy, z0, z1):
    """A probe: the box of `dx` by `dy`, centred on (`x`, `y`) in plan, from `z0` up to `z1`."""
    return mf.Manifold.cube([dx, dy, z1 - z0]).translate((x - 0.5 * dx, y - 0.5 * dy, z0))


def bounds(part):
    b = np.asarray(part.bounding_box(), dtype=float)
    return b[:3], b[3:]


def board_behind(s):
    """The board the part hangs on: a slab behind the back plate with round holes at the pegs."""
    b, r = s["board"], 0.5 * pegboard.PEG_HOLE
    slab = mf.Manifold.cube([b, 300.0, 300.0]).translate((s["x_face"] - b, -150.0, -150.0))
    for y in (s["peg_y"], s["lower_y"]):
        slab = slab - (mf.Manifold.cylinder(b + 2.0, r, r, 96).rotate((0.0, 90.0, 0.0))
                       .translate((s["x_face"] - b - 1.0, y, s["peg_z"])))
    return slab


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_every_kind_is_one_volume_that_stands_on_the_bed(kind):
    part = pegboard.pegboard_part(kind)
    assert part.status() == mf.Error.NoError
    assert len(part.decompose()) == 1, f"a {kind} is not one closed volume"
    lo, hi = bounds(part)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), f"a {kind} does not stand on the bed"
    assert hi[:2] == pytest.approx(-lo[:2], abs=1e-6), f"a {kind} is not centred in plan"


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_every_kind_prints_without_support(kind):
    v, t = geom.mesh_arrays(pegboard.pegboard_part(kind))
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    size = np.linalg.norm(n, axis=1)
    keep = size > 1e-12                       # a triangle of no area has no direction
    n, a, b, c = n[keep] / size[keep][:, None], a[keep], b[keep], c[keep]
    z = (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0
    hanging = (n[:, 2] < -math.sin(math.radians(45.0)) - 0.01) & (z > 1e-6)
    assert not hanging.any(), f"a {kind} has a face that overhangs more than 45 degrees"


@pytest.mark.parametrize("board", [3.0, 5.0, 7.0])
@pytest.mark.parametrize("kind", HANGING)
def test_the_pegs_pass_the_holes_of_a_board(kind, board):
    s = pegboard.sizes(kind, board=board)
    part = pegboard.pegboard_part(kind, board=board)
    assert common(part, board_behind(s)) == pytest.approx(0.0, abs=1e-6), \
        "a peg is too fat for a hole of the board"
    for y in (s["peg_y"], s["lower_y"]):
        inside = block(s["x_face"] - 0.5 * board, y, 0.5, 4.0, 0.0, s["peg_z"] * 2.0)
        assert common(part, inside) > 0.0, "a peg does not go into the board"


def test_the_upper_peg_rises_and_comes_out_behind_the_board():
    s = pegboard.sizes("hook")
    part = pegboard.pegboard_part("hook")
    assert 10.0 < s["peg_rise"] <= pegboard.PEG_RISE_MAX, "the peg does not rise enough to hold"
    behind = s["x_face"] - s["board"] - 0.75
    rise = (s["board"] * 0.5 + 0.75) * math.tan(math.radians(s["peg_rise"]))
    assert common(part, block(behind, s["peg_y"] + rise, 0.5, 0.5, 0.0, 1.0)) > 0.0, \
        "the upper peg does not come out behind the board, risen"
    assert common(part, block(behind, s["lower_y"], 0.5, 6.0, 0.0, 5.0)) == 0.0, \
        "the lower peg goes through the board and beyond it"


def test_the_two_pegs_stand_on_the_pitch():
    s = pegboard.sizes("shelf")
    assert s["peg_y"] - s["lower_y"] == pytest.approx(pegboard.PEG_PITCH)
    part = pegboard.pegboard_part("shelf")
    mid = s["x_face"] - 0.5 * s["board"]
    for y in (s["peg_y"], s["lower_y"]):
        lane = block(mid, y, 0.2, pegboard.PEG_HOLE, 0.0, 10.0)
        lo, hi = bounds(mf.Manifold.batch_boolean([part, lane], mf.OpType.Intersect))
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(y, abs=0.05), "a peg is off its hole"


@pytest.mark.parametrize("kind", ["hook", "shelf"])
def test_reach_makes_a_hook_longer_and_a_shelf_deeper(kind):
    short = pegboard.pegboard_part(kind, reach=20.0)
    long = pegboard.pegboard_part(kind, reach=60.0)
    (lo_s, hi_s), (lo_l, hi_l) = bounds(short), bounds(long)
    assert (hi_l[0] - lo_l[0]) - (hi_s[0] - lo_s[0]) == pytest.approx(40.0), \
        f"reach does not reach in a {kind}"
    s = pegboard.sizes(kind, reach=60.0)
    column = block(s["x_face"] + 58.0, 0.0, 1.0, 300.0, 0.0, 1.0)
    assert common(long, column) > 0.0, f"the {kind} stops short of its reach"


def test_a_hook_rises_to_its_tip():
    s = pegboard.sizes("hook")
    part = pegboard.pegboard_part("hook")
    root = block(s["x_face"] + 5.0, 0.0, 1.0, 300.0, 0.0, 1.0)
    tip = block(s["x_face"] + s["reach"], 0.0, 1.0, 300.0, 0.0, 1.0)
    low = bounds(mf.Manifold.batch_boolean([part, root], mf.OpType.Intersect))[0][1]
    high = bounds(mf.Manifold.batch_boolean([part, tip], mf.OpType.Intersect))[0][1]
    assert high - low == pytest.approx((s["reach"] - 5.0) * math.tan(math.radians(
        pegboard.HOOK_RISE)), abs=0.1), "the arm of a hook does not rise"


@pytest.mark.parametrize("size", [5.0, 12.0, 40.0])
def test_the_ring_stands_upright_and_takes_the_tool_asked_for(size):
    s = pegboard.sizes("holder", size=size)
    part = pegboard.pegboard_part("holder", size=size)

    def tool(d):
        """A tool of `d` across, standing up through the ring as it hangs: along y."""
        return (mf.Manifold.cylinder(60.0, 0.5 * d, 0.5 * d, pegboard.SEGMENTS)
                .rotate((-90.0, 0.0, 0.0)).translate((s["ring_x"], s["ring_y"] - 20.0,
                                                      s["ring_z"])))
    assert common(part, tool(size - 0.2)) == pytest.approx(0.0, abs=1e-6), \
        "the tool does not drop into the ring from above"
    assert common(part, tool(size + 0.5)) > 0.0, "the ring is wider than asked"


def test_a_number_that_does_not_print_becomes_the_nearest_one_that_does():
    s = pegboard.sizes("hook", width=0.0, reach=0.0, board=0.0, size=0.0)
    assert (s["width"], s["reach"], s["board"]) == (
        pegboard.WIDTH_MIN, pegboard.REACH_MIN, pegboard.BOARD_MIN)
    s = pegboard.sizes("holder", width=99.0, reach=999.0, board=99.0, size=99.0)
    assert (s["width"], s["reach"], s["board"], s["size"]) == (
        pegboard.WIDTH_MAX, pegboard.REACH_MAX, pegboard.BOARD_MAX, pegboard.SIZE_MAX)
    b = pegboard.board_sizes(cols=0, rows=99, standoff=0.0)
    assert (b["cols"], b["rows"], b["standoff"]) == (
        pegboard.COLS_MIN, pegboard.ROWS_MAX, pegboard.STANDOFF_MIN)


def test_a_kind_there_is_no_such_thing_of_is_refused():
    for bad in ("ladder", "", "hooks"):
        with pytest.raises(ValueError, match="pegboard"):
            pegboard.pegboard_part(bad)
        with pytest.raises(ValueError, match="pegboard"):
            pegboard.sizes(bad)


# ------------------------------------------------------------------ the board

def test_the_board_has_its_holes_on_the_pitch_and_four_screw_holes():
    board = pegboard.pegboard_board(cols=5, rows=3, board=5.0, standoff=15.0)
    assert board.genus() == 5 * 3 + 4, "not every hole and screw hole goes through"
    lo, hi = bounds(board)
    assert (hi - lo) == pytest.approx([5 * pegboard.PEG_PITCH, 3 * pegboard.PEG_PITCH, 20.0])
    r = 0.5 * pegboard.PEG_HOLE
    for i in range(5):
        for j in range(3):
            x, y = lo[0] + (i + 0.5) * pegboard.PEG_PITCH, lo[1] + (j + 0.5) * pegboard.PEG_PITCH
            fits = mf.Manifold.cylinder(5.0, r - 0.05, r - 0.05, 48).translate((x, y, 0.0))
            assert common(board, fits) == pytest.approx(0.0, abs=1e-6), "a hole is too small"


def test_the_board_stands_off_the_wall_with_room_for_the_pegs():
    s = pegboard.sizes("hook", board=5.0)
    board = pegboard.pegboard_board(board=5.0, standoff=pegboard.STANDOFF_MIN)
    lo, hi = bounds(board)
    # behind the board, from the plate of a hook to the tip of its peg, is less than the standoff
    assert s["x_face"] - s["x_tip"] - s["board"] < pegboard.STANDOFF_MIN
    # and that room is empty behind every hole that is not next to a boss or the rim
    x, y = lo[0] + 2.5 * pegboard.PEG_PITCH, lo[1] + 2.5 * pegboard.PEG_PITCH
    assert common(board, block(x, y, 8.0, 8.0, 5.0 + 0.01, hi[2])) == 0.0


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_holder_and_a_board_and_remembers_them(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def holder(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData("holder"))
        assert dlg.size.isEnabled() and not dlg.cols.isEnabled()
        dlg.size.setValue(20.0)
        dlg.width.setValue(10.0)
        return QDialog.Accepted

    def board(dlg):
        assert dlg.kind.currentData() == "holder", "the dialog starts at the last kind"
        dlg.kind.setCurrentIndex(dlg.kind.findData("board"))
        assert dlg.cols.isEnabled() and not dlg.reach.isEnabled() and not dlg.size.isEnabled()
        dlg.cols.setValue(4)
        dlg.rows.setValue(3)
        return QDialog.Accepted

    def wait(n):
        deadline = time.monotonic() + 15.0
        while len(w.doc.groups) < n and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == n, "no part in the model"

    try:
        monkeypatch.setattr(app_module.QDialog, "exec", holder)
        w.pegboard_dialog()
        wait(1)
        s = app_module.Settings()
        assert (s.peg_kind, s.peg_size, s.peg_width) == ("holder", pytest.approx(20.0),
                                                         pytest.approx(10.0))
        monkeypatch.setattr(app_module.QDialog, "exec", board)
        w.pegboard_dialog()
        wait(2)
        lo, hi = np.asarray(w.doc.groups[1].bbox(), dtype=float)
        assert hi[:2] - lo[:2] == pytest.approx([4 * pegboard.PEG_PITCH, 3 * pegboard.PEG_PITCH])
        s = app_module.Settings()
        assert (s.peg_kind, s.peg_cols, s.peg_rows) == ("board", 4, 3)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Pegboard") == labels.index("Gridfinity") + 1
    finally:
        w.settings.peg_kind = "hook"
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
