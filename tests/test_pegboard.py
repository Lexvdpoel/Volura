"""Pegboard parts: the board of 1/4 inch holes, and the hooks, shelves and holders that hang on it.

`pegboard.pegboard_part` makes a part that hangs on two holes of `PEG_PITCH` apart like a steel
pegboard hook: a round upper peg that goes through the board and bends up behind it into a tip,
and a straight round lower peg as deep as the board, with a back plate between them against the
front of the board. The part lies on its side on the bed, symmetric about its middle plane.
`pegboard_board` is the board itself, with chamfered holes and a rim and screw bosses on its back.
These tests weigh the parts against a real board: every kind is one symmetric volume on the bed,
the pegs are round and on the pitch and pass the holes of a board of every thickness, the tip
hooks behind the board with less than half a millimetre of play and less than two degrees of
tilt, the hook goes on by tilting it in, `reach` makes a hook longer and a shelf deeper, a hook
rises, and the ring of a holder is round, stands upright and takes the tool asked for.

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

from volura import pegboard  # noqa: E402

HANGING = ("hook", "shelf", "holder")
BOARDS = (3.0, 5.0, 7.0)
#: mm3 of overlap with the board that still counts as touching: the corners of the polygons
TOUCH = 0.5


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def block(x, y, dx, dy, z0, z1):
    """A probe: the box of `dx` by `dy`, centred on (`x`, `y`) in plan, from `z0` up to `z1`."""
    return mf.Manifold.cube([dx, dy, z1 - z0]).translate((x - 0.5 * dx, y - 0.5 * dy, z0))


def bounds(part):
    b = np.asarray(part.bounding_box(), dtype=float)
    return b[:3], b[3:]


def board_behind(s, lift=0.0):
    """The board the part hangs on: a slab behind the back plate with round holes, whose middles
    stand `lift` above the axes of the pegs."""
    b, r = s["board"], 0.5 * pegboard.PEG_HOLE
    slab = mf.Manifold.cube([b, 300.0, 300.0]).translate((s["x_face"] - b, -150.0, -150.0))
    for y in (s["peg_y"], s["lower_y"]):
        slab = slab - (mf.Manifold.cylinder(b + 2.0, r, r, 96).rotate((0.0, 90.0, 0.0))
                       .translate((s["x_face"] - b - 1.0, y + lift, s["peg_z"])))
    return slab


def hung(kind, board):
    """A part and its board as it hangs: the pegs just above the bottom of their holes."""
    s = pegboard.sizes(kind, board=board)
    part = pegboard.pegboard_part(kind, board=board)
    return s, part, board_behind(s, pegboard.hanging_drop() - 0.02)


def section_area(part, x, y, z, size=10.0, thick=0.02):
    """The area of the part cut square to x at `x`, near (`y`, `z`)."""
    return common(part, block(x, y, thick, size, z - 0.5 * size, z + 0.5 * size)) / thick


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_every_kind_is_one_volume_that_stands_on_the_bed(kind):
    part = pegboard.pegboard_part(kind)
    assert part.status() == mf.Error.NoError
    assert len(part.decompose()) == 1, f"a {kind} is not one closed volume"
    lo, hi = bounds(part)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), f"a {kind} does not stand on the bed"
    assert hi[:2] == pytest.approx(-lo[:2], abs=1e-6), f"a {kind} is not centred in plan"


@pytest.mark.parametrize("kw", [{}, {"board": 3.0, "width": 6.0},
                                {"board": 7.0, "width": 20.0, "size": 40.0}])
@pytest.mark.parametrize("kind", HANGING)
def test_a_part_is_symmetric_about_its_middle(kind, kw):
    s = pegboard.sizes(kind, **kw)
    part = pegboard.pegboard_part(kind, **kw)
    mirror = part.mirror((0.0, 0.0, 1.0)).translate((0.0, 0.0, 2.0 * s["peg_z"]))
    differ = (part - mirror).volume() + (mirror - part).volume()
    assert differ < 0.01 * part.volume(), f"the two halves of a {kind} differ"


@pytest.mark.parametrize("kind", HANGING)
def test_the_pegs_are_round_and_stand_on_the_pitch(kind):
    s = pegboard.sizes(kind)
    part = pegboard.pegboard_part(kind)
    assert s["peg_y"] - s["lower_y"] == pytest.approx(pegboard.PEG_PITCH)
    x = s["x_face"] - 0.5 * s["board"]                    # in the middle of the board
    for y in (s["peg_y"], s["lower_y"]):
        area = section_area(part, x, y, s["peg_z"])
        assert area == pytest.approx(math.pi * 0.25 * pegboard.PEG_D ** 2, rel=0.02), \
            "a peg is not a round rod of PEG_D"
        cut = mf.Manifold.batch_boolean([part, block(x, y, 0.02, 10.0, 0.0, 50.0)],
                                        mf.OpType.Intersect)
        lo, hi = bounds(cut)
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(y, abs=0.05), "a peg is off its hole"
        assert 0.5 * (lo[2] + hi[2]) == pytest.approx(s["peg_z"], abs=0.05), \
            "a peg is not in the middle of the part"
        assert hi[1] - lo[1] == pytest.approx(pegboard.PEG_D, abs=0.05)


@pytest.mark.parametrize("board", BOARDS)
@pytest.mark.parametrize("kind", HANGING)
def test_the_pegs_pass_the_holes_of_a_board(kind, board):
    s, part, slab = hung(kind, board)
    assert common(part, slab) < TOUCH, "a peg is too fat for a hole of the board"
    for y in (s["peg_y"], s["lower_y"]):
        inside = block(s["x_face"] - 0.5 * board, y, 0.5, 4.0, 0.0, s["peg_z"] * 2.0)
        assert common(part, inside) > 0.0, "a peg does not go into the board"
    beyond = block(s["x_face"] - board - 1.0, s["lower_y"], 2.0, 6.0, 0.0, 2.0 * s["peg_z"])
    assert common(part, beyond) == pytest.approx(0.0, abs=1e-6), \
        "the lower peg goes through the board and beyond it"


@pytest.mark.parametrize("board", BOARDS)
def test_the_tip_stands_up_behind_the_board(board):
    s, part, slab = hung("hook", board)
    x = s["x_tip"] + 0.5 * pegboard.PEG_D                  # the axis of the tip
    top = s["peg_y"] + pegboard.TIP
    assert common(part, block(x, top - 0.5, 1.0, 1.0, 0.0, 50.0)) > 0.0, "no tip at the top"
    assert common(part, block(x, top + 0.3, 1.0, 0.5, 0.0, 50.0)) == 0.0, "the tip is too high"
    # pulled 3 mm straight out of the board, the tip runs into the back of it
    assert common(part.translate((3.0, 0.0, 0.0)), slab) > 1.0, "the tip does not hook behind"


@pytest.mark.parametrize("board", BOARDS)
def test_the_part_has_little_play_on_the_board(board):
    s, part, slab = hung("hook", board)
    base = common(part, slab)

    def touches(p):
        return common(p, slab) > base + 0.005

    moved = 0.0
    while not touches(part.translate((moved + 0.05, 0.0, 0.0))) and moved < 5.0:
        moved += 0.05
    assert 0.05 < moved <= 0.5, f"it moves {moved:.2f} mm along the peg"
    # tilted top away from the board - about the bottom edge of the back plate, which stays on
    # the board while the lower peg has room in its hole - until the tip touches
    px, py = s["x_face"], s["y_bottom"]
    turned = 0.0
    while turned < 10.0:
        p = part.translate((-px, -py, 0.0)).rotate((0.0, 0.0, -(turned + 0.1))) \
            .translate((px, py, 0.0))
        if touches(p):
            break
        turned += 0.1
    assert 0.1 < turned < 2.0, f"it tilts {turned:.1f} degrees"


@pytest.mark.parametrize("board", BOARDS)
def test_the_hook_goes_on_by_tilting_it_in(board):
    path = pegboard.insertion_path(board)
    assert path is not None, f"the hook does not go on a board of {board} mm"
    assert path[0][0] == pytest.approx(90.0) and path[-1][0] == pytest.approx(0.0)
    steps = list(zip(path, path[1:]))
    assert all(math.hypot(b[1] - a[1], b[2] - a[2]) <= pegboard.INSERT_SHIFT + 1e-9
               for a, b in steps), "the part jumps"
    part, slab = pegboard.mount_section(board), pegboard.board_section(board)
    for deg, dx, dy in path:
        assert (pegboard.placed(part, board, deg, dx, dy) ^ slab).area() <= \
            pegboard.INSERT_OVERLAP


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
def test_the_ring_is_round_stands_upright_and_takes_the_tool_asked_for(size):
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
    # cut across its axis, the ring is a round ring: the area of the annulus, centred on the axis
    y = s["ring_y"] + 0.5 * pegboard.HOLDER_HEIGHT
    R, r = s["ring_r"], s["bore_r"]
    ring = mf.Manifold.batch_boolean(
        [part, mf.Manifold.cylinder(0.02, R + 0.5, R + 0.5, 96).rotate((-90.0, 0.0, 0.0))
         .translate((s["ring_x"], y, s["ring_z"]))], mf.OpType.Intersect)
    assert ring.volume() / 0.02 == pytest.approx(math.pi * (R * R - r * r), rel=0.02)
    lo, hi = bounds(ring)
    assert hi[2] - lo[2] == pytest.approx(2.0 * R, abs=0.05)
    assert lo[2] == pytest.approx(0.0, abs=1e-6) or s["width"] > 2.0 * R


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

def test_the_board_has_chamfered_holes_on_the_pitch_and_four_screw_holes():
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
            # the face of the board, where a peg comes in, is wider by the chamfer
            mouth = mf.Manifold.cylinder(0.1, r + 0.3, r + 0.3, 48).translate((x, y, 0.0))
            assert common(board, mouth) == pytest.approx(0.0, abs=1e-6), "no chamfer"
            deeper = mf.Manifold.cylinder(0.1, r + 0.3, r + 0.3, 48).translate((x, y, 1.0))
            assert common(board, deeper) > 0.0, "the chamfer runs too deep"


def test_the_board_stands_off_the_wall_with_room_for_the_tips():
    s = pegboard.sizes("hook", board=5.0)
    board = pegboard.pegboard_board(cols=6, rows=6, board=5.0, standoff=pegboard.STANDOFF_MIN)
    lo, _hi = bounds(board)
    behind = s["x_face"] - s["x_tip"] - s["board"]
    assert behind < pegboard.STANDOFF_MIN, "the tip reaches the wall"
    # behind every hole, where the tip of a hung part stands up, the rim and the bosses leave room
    rise = pegboard.TIP + pegboard.hanging_drop()
    tips = [block(lo[0] + (i + 0.5) * pegboard.PEG_PITCH,
                  lo[1] + (j + 0.5) * pegboard.PEG_PITCH + 0.5 * rise,
                  pegboard.PEG_D, rise, 5.0 + 0.01, 5.0 + behind)
            for i in range(6) for j in range(6)]
    assert common(board, mf.Manifold.batch_boolean(tips, mf.OpType.Add)) == 0.0


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
