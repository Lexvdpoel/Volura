"""Pegboard parts: hooks, shelves and holders that hang on a board of 1/4 inch holes.

`pegboard.pegboard_part` makes a part that hangs on two holes of `PEG_PITCH` apart: an upper peg
that goes through the board and bends down behind it, and a lower peg as deep as the board that
keeps the part from swinging, with a back plate between them against the front of the board. The
part lies on its side on the bed and is extruded `width` millimetres, so the pegs are as thick as
the print in one direction and `PEG_DIA` in the other. These tests weigh the parts against that:
every kind is one volume that stands on the bed, the pegs stand on the pitch and go into a board
hole, the upper peg bends down and not up, `reach` makes a hook longer and a shelf deeper, and the
ring of a holder takes the tool that was asked for.

Run with:  python -m pytest tests/test_pegboard.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import pegboard  # noqa: E402


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def block(x, y, dx, dy, w):
    """A probe: the box of `dx` by `dy`, centred on (`x`, `y`) in plan, from the bed up `w`."""
    return mf.Manifold.cube([dx, dy, w]).translate((x - 0.5 * dx, y - 0.5 * dy, 0.0))


def span(part):
    lo, hi = np.asarray(part.bounding_box()[:3]), np.asarray(part.bounding_box()[3:])
    return hi - lo


def peg_slab(s):
    """A slab around the straight part of the pegs: behind the board, in front of the bend."""
    x0 = s["x_back"] + 0.5 * pegboard.PEG_DIA + 0.1
    x1 = s["x_face"] - 0.1
    return block(0.5 * (x0 + x1), 0.0, x1 - x0, 4.0 * pegboard.PEG_PITCH, s["width"])


def peg_probe(s, y):
    """The straight part of one peg, from the back of the board up to the plate."""
    x0 = s["x_back"] + 0.5 * pegboard.PEG_DIA + 0.1
    x1 = s["x_face"] - 0.1
    return block(0.5 * (x0 + x1), y, x1 - x0, 2.0 * pegboard.PEG_DIA, s["width"])


def board_hole(s, y):
    """A board hole of `PEG_HOLE` about the axis of the peg at `y`, along the board."""
    r = pegboard.PEG_HOLE
    return (mf.Manifold.cylinder(2.0 * r, 2.0 * r, 2.0 * r, pegboard.SEGMENTS, True)
            .rotate((0.0, 90.0, 0.0))
            .translate((s["x_face"] - 0.5 * s["board"], y, 0.5 * s["width"])))


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_every_kind_is_one_volume_that_stands_on_the_bed(kind):
    part = pegboard.pegboard_part(kind)
    assert part.status() == mf.Error.NoError
    assert len(part.decompose()) == 1, f"a {kind} is not one closed volume"
    lo, hi = np.asarray(part.bounding_box()[:3]), np.asarray(part.bounding_box()[3:])
    assert lo[2] == pytest.approx(0.0, abs=1e-6), f"a {kind} does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(pegboard.WIDTH_DEFAULT)
    assert hi[:2] == pytest.approx(-lo[:2], abs=1e-6), f"a {kind} is not centred in plan"


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_the_two_pegs_stand_on_the_pitch(kind):
    s = pegboard.sizes(kind)
    pegs = mf.Manifold.batch_boolean([pegboard.pegboard_part(kind), peg_slab(s)],
                                     mf.OpType.Intersect).decompose()
    assert len(pegs) == 2, "the two pegs are not two pegs"
    axes = sorted(0.5 * (p.bounding_box()[1] + p.bounding_box()[4]) for p in pegs)
    assert axes[1] - axes[0] == pytest.approx(pegboard.PEG_PITCH, abs=1e-6)
    assert axes == [pytest.approx(s["lower_y"], abs=1e-6), pytest.approx(s["peg_y"], abs=1e-6)]
    for p in pegs:
        assert p.bounding_box()[4] - p.bounding_box()[1] == pytest.approx(pegboard.PEG_DIA,
                                                                          abs=1e-6)


@pytest.mark.parametrize("kind", pegboard.KINDS)
def test_a_peg_goes_into_a_hole_of_the_board(kind):
    s = pegboard.sizes(kind)
    part = pegboard.pegboard_part(kind)
    for y in (s["peg_y"], s["lower_y"]):
        peg = mf.Manifold.batch_boolean([part, peg_probe(s, y)], mf.OpType.Intersect)
        assert peg.volume() > 0.0, "a peg does not go through the board"
        assert peg.volume() - common(peg, board_hole(s, y)) < 1e-3, "a peg is too thick for a hole"


def test_the_upper_pegs_bends_down_behind_the_board():
    s = pegboard.sizes("hook")
    part = pegboard.pegboard_part("hook")
    behind = s["x_face"] - s["board"] - 2.0
    assert common(part, block(behind, s["peg_y"] - 4.0, 1.0, 1.0, s["width"])) > 0.0, \
        "the upper peg does not hang down behind the board"
    assert common(part, block(behind, s["peg_y"] + 4.0, 1.0, 1.0, s["width"])) == 0.0, \
        "the upper peg bends up instead of down"
    assert common(part, block(behind, s["lower_y"], 1.0, 1.0, s["width"])) == 0.0, \
        "the lower peg goes through the board and beyond it"


@pytest.mark.parametrize("kind", ["hook", "shelf"])
def test_reach_makes_a_hook_longer_and_a_shelf_deeper(kind):
    short = pegboard.pegboard_part(kind, reach=20.0)
    long = pegboard.pegboard_part(kind, reach=60.0)
    assert span(long)[0] - span(short)[0] == pytest.approx(40.0), f"reach does not reach in a {kind}"
    s = pegboard.sizes(kind, reach=60.0)
    assert common(long, block(s["x_face"] + 55.0, s["y_bottom"] + 0.5, 1.0, 1.0,
                              s["width"])) > 0.0, f"the {kind} stops short of its reach"
    assert common(long, block(s["x_face"] + 61.0, s["y_bottom"] + 0.5, 1.0, 1.0,
                              s["width"])) == 0.0, f"the {kind} stands past its reach"


@pytest.mark.parametrize("size", [5.0, 12.0, 40.0])
def test_the_ring_takes_the_tool_of_the_size_asked_for(size):
    s = pegboard.sizes("holder", size=size)
    part = pegboard.pegboard_part("holder", size=size)
    loose = (mf.Manifold.cylinder(s["width"], 0.5 * (size - 0.5), 0.5 * (size - 0.5),
                                  pegboard.SEGMENTS)
             .translate((s["ring_x"], s["ring_y"], 0.0)))
    tight = (mf.Manifold.cylinder(s["width"], 0.5 * (size + 0.5), 0.5 * (size + 0.5),
                                  pegboard.SEGMENTS)
             .translate((s["ring_x"], s["ring_y"], 0.0)))
    assert common(part, loose) == pytest.approx(0.0, abs=1e-6), "the ring is tighter than asked"
    assert common(part, tight) > 0.0, "the ring is wider than asked"


def test_a_number_that_does_not_print_becomes_the_nearest_one_that_does():
    s = pegboard.sizes("hook", width=0.0, reach=0.0, board=0.0, size=0.0)
    assert (s["width"], s["reach"], s["board"]) == (
        pegboard.WIDTH_MIN, pegboard.REACH_MIN, pegboard.BOARD_MIN)
    s = pegboard.sizes("holder", width=99.0, reach=999.0, board=99.0, size=99.0)
    assert (s["width"], s["reach"], s["board"], s["size"]) == (
        pegboard.WIDTH_MAX, pegboard.REACH_MAX, pegboard.BOARD_MAX, pegboard.SIZE_MAX)
    assert s["ring_x"] - s["x_face"] == pytest.approx(pegboard.REACH_MAX)
    part = pegboard.pegboard_part("shelf", width=99.0, reach=999.0, board=99.0)
    lo, hi = np.asarray(part.bounding_box()[:3]), np.asarray(part.bounding_box()[3:])
    assert hi[2] - lo[2] == pytest.approx(pegboard.WIDTH_MAX)
    assert hi[0] - lo[0] == pytest.approx(
        pegboard.REACH_MAX + pegboard.BOARD_MAX + pegboard.PEG_THROUGH + 0.5 * pegboard.PEG_DIA)


def test_a_kind_there_is_no_such_thing_of_is_refused():
    for bad in ("ladder", "", "hooks"):
        with pytest.raises(ValueError, match="pegboard"):
            pegboard.pegboard_part(bad)
        with pytest.raises(ValueError, match="pegboard"):
            pegboard.sizes(bad)


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_holder_and_remembers_it(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        assert not dlg.size.isEnabled() or dlg.kind.currentData() == "holder"
        dlg.kind.setCurrentIndex(dlg.kind.findData("holder"))
        assert dlg.size.isEnabled(), "a holder takes the size of its tool"
        dlg.size.setValue(20.0)
        dlg.width.setValue(10.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.pegboard_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no part in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert hi[2] - lo[2] == pytest.approx(10.0), "the part is not as high as its width"
        s = app_module.Settings()
        assert (s.peg_kind, s.peg_size, s.peg_width) == ("holder", pytest.approx(20.0),
                                                         pytest.approx(10.0))
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Pegboard Hook") == labels.index("Gridfinity") + 1
    finally:
        w.settings.peg_kind = "hook"
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
