"""Project box: standoffs on the floor under the mounting holes of a common circuit board.

`enclosure.project_box(board=...)` puts a round standoff with a pilot hole under every mounting
hole of a board from `enclosure.BOARDS`, `board_height` millimetres high on the floor, with the
board in the middle of the box, along its length or turned a quarter when only that fits. These
tests find the standoffs where the board's holes are: plastic around a hole and the pilot hole
empty, the right number of them, the turn, and a box too small for the board refused.

Run with:  python -m pytest tests/test_project_box_boards.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import enclosure  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BOX = dict(length=95.0, width=65.0, height=30.0, wall=2.0, floor=2.0, radius=3.0)


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def probe(x, y, z, r, h=1.0):
    return mf.Manifold.cylinder(h, r, r, 24).translate((x, y, z))


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


@pytest.mark.parametrize("board", list(enclosure.BOARDS))
def test_a_standoff_with_a_pilot_hole_under_every_mounting_hole(board):
    box = dict(BOX, length=110.0) if board == "Arduino Mega" else BOX
    base, _lid = enclosure.project_box(**box, board=board, board_height=6.0)
    nominal = float(enclosure.BOARDS[board][3][1:])
    r_post = 0.5 * enclosure.POST_OVER_SCREW * nominal
    r_pilot = 0.5 * enclosure.PILOT_OVER_SCREW * nominal
    z = box["floor"] + 3.0                    # halfway up the standoffs
    for x, y in enclosure.board_places(board, box["length"], box["width"]):
        ring = probe(x, y, z, 0.5 * (r_post + r_pilot)) - probe(x, y, z - 1.0, r_pilot + 0.05, 3.0)
        assert common(base, ring) == pytest.approx(ring.volume(), rel=1e-3), \
            f"no standoff at {x:.1f}, {y:.1f}"
        hole = probe(x, y, z, 0.8 * r_pilot)
        assert common(base, hole) == pytest.approx(0.0, abs=1e-6), "the pilot hole is not empty"
        above = probe(x, y, box["floor"] + 6.5, r_post)
        assert common(base, above) == pytest.approx(0.0, abs=1e-6), "the standoff is too high"


def test_without_a_board_the_floor_is_empty():
    plain, _ = enclosure.project_box(**BOX)
    with_board, _ = enclosure.project_box(**BOX, board="Raspberry Pi 4 / 5")
    assert with_board.volume() > plain.volume()
    assert with_board.genus() == plain.genus(), "a pilot hole goes through the floor"


def test_the_board_lies_in_the_middle_and_turns_when_it_only_fits_that_way():
    straight = enclosure.board_places("Raspberry Pi 4 / 5", 95.0, 65.0)
    xs, ys = [p[0] for p in straight], [p[1] for p in straight]
    assert max(xs) - min(xs) == pytest.approx(58.0) and max(ys) - min(ys) == pytest.approx(49.0)
    assert sum(xs) / 4 == pytest.approx(-42.5 + 32.5) and sum(ys) / 4 == pytest.approx(0.0)
    turned = enclosure.board_places("Raspberry Pi 4 / 5", 65.0, 95.0)
    xs, ys = [p[0] for p in turned], [p[1] for p in turned]
    assert max(xs) - min(xs) == pytest.approx(49.0) and max(ys) - min(ys) == pytest.approx(58.0)


def test_a_box_too_small_for_the_board_is_refused():
    with pytest.raises(ValueError):
        enclosure.project_box(**dict(BOX, length=60.0, width=40.0), board="Arduino Uno")
    with pytest.raises(ValueError):
        enclosure.project_box(**BOX, board="Commodore 64")
    with pytest.raises(ValueError):
        enclosure.project_box(**BOX, board="Arduino Uno", board_height=40.0)


def test_the_dialog_remembers_the_board(win, monkeypatch):
    def fill(dlg):
        dlg.length.setValue(95.0)
        dlg.width.setValue(65.0)
        dlg.board.setCurrentIndex(dlg.board.findData("Raspberry Pi 4 / 5"))
        dlg.board_height.setValue(7.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.project_box_dialog()
    s = app_module.Settings()
    assert (s.box3d_board, s.box3d_board_height) == ("Raspberry Pi 4 / 5", pytest.approx(7.0))
    win.settings.box3d_board, win.settings.box3d_board_height = "", 5.0   # the other tests: no board
    win.settings.save()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
