"""Reading sliced G-code back for the print preview (volura/gcode_preview.py).

Also the two colour palettes of the preview and the colour-blind preference behind them.

Run with:  python -m pytest tests/test_gcode_preview.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from volura import gcode_preview as gp  # noqa: E402
from volura.settings import qsettings  # noqa: E402

GCODE = """\
;FLAVOR:Marlin
G28 ; home, before the first layer: left out
G1 X50 Y50 F3000
;LAYER:0
G0 F6000 X10 Y10 Z0.2
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;TYPE:FILL
G1 X10 Y20 E3.0
G0 F6000 X10 Y10
;TIME_ELAPSED:10.0
;LAYER:1
G0 X10 Y10 Z0.4
G91
M83
;TYPE:SKIN
G1 F600 X10 E0.5
G1 Y10 E0.5
G90
G92 E0
;TIME_ELAPSED:25.0
"""


@pytest.fixture
def preview():
    return gp.parse(GCODE.splitlines())


def test_layers_and_heights(preview):
    assert [layer.number for layer in preview.layers] == [0, 1]
    assert [layer.z for layer in preview.layers] == pytest.approx([0.2, 0.4])


def test_kinds_and_travel(preview):
    first = preview.layers[0]
    assert [gp.KIND_NAMES[k] for k in first.kind] == \
        ["travel", "WALL-OUTER", "WALL-OUTER", "FILL", "travel"]
    assert first.seg[1].tolist() == pytest.approx([10, 10, 20, 10])


def test_relative_moves_and_extrusion(preview):
    second = preview.layers[1]
    assert [gp.KIND_NAMES[k] for k in second.kind] == ["SKIN", "SKIN"]
    assert second.seg.ravel().tolist() == pytest.approx([10, 10, 20, 10, 20, 10, 20, 20])


def test_times_follow_time_elapsed(preview):
    assert preview.layers[0].seconds == pytest.approx(10.0)
    assert preview.layers[1].seconds == pytest.approx(15.0)
    assert preview.seconds == pytest.approx(25.0)
    t = preview.layers[1].t
    assert t[0] == pytest.approx(7.5)            # two moves of the same length and speed


def test_position_along_the_layer(preview):
    layer = preview.layers[1]
    pos, done = layer.position(3.75)
    assert pos == pytest.approx((15.0, 10.0)) and done == 0
    pos, done = layer.position(100.0)
    assert pos == pytest.approx((20.0, 20.0)) and done == 2


def test_box_of_what_is_printed(preview):
    assert preview.lo == pytest.approx((10.0, 10.0))
    assert preview.hi == pytest.approx((20.0, 20.0))


def test_empty_gcode():
    p = gp.parse([])
    assert p.layers == [] and p.seconds == 0.0


def test_the_3d_view_turns_moves_zooms_and_draws(preview, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    from volura import print_preview as pp
    path = tmp_path / "x.gcode"
    path.write_text(GCODE, encoding="utf-8")
    dlg = pp.PrintPreviewDialog(None, str(path), preview, (0.0, 0.0, 100.0, 100.0), "")
    try:
        dlg.resize(800, 700)
        dlg.show()
        QApplication.processEvents()
        view = dlg.view
        assert view.layer == 1, "the preview starts at the whole print"
        # every printed line is there once, the lower layers dimmed
        assert view.lines.start == [0, 6, 10]
        assert view.lines.dim == pytest.approx(view.lines.col * pp.DIM)
        yaw, target, distance = view.cam.yaw, view.cam.target.copy(), view.cam.distance
        c = view.rect().center()
        QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, c)
        QTest.mouseMove(view, c + QPoint(60, 20))
        QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, c + QPoint(60, 20))
        assert view.cam.yaw != yaw, "dragging does not turn the view"
        QTest.mousePress(view, Qt.RightButton, Qt.NoModifier, c)
        QTest.mouseMove(view, c + QPoint(40, 0))
        QTest.mouseRelease(view, Qt.RightButton, Qt.NoModifier, c + QPoint(40, 0))
        assert (view.cam.target != target).any(), "right-dragging does not move the view"
        view.cam.zoom(0.5)
        assert view.cam.distance < distance
        dlg.top.click()
        assert view.cam.pitch > 89.0, "Top does not look down"
        dlg.iso.click()
        assert view.cam.pitch < 60.0
        dlg.only.setChecked(True)
        dlg.travel.setChecked(True)
        assert view.only_layer and view.show_travel
        view.at = 3.0
        dlg.grab()                                       # it draws, half a layer and the nozzle
    finally:
        dlg.close()


# ------------------------------------------------------------------- colours

def test_both_palettes_have_the_same_kinds():
    from volura import print_preview as pp
    assert [name for _color, name in pp.KIND_STYLE_CB] == \
        [name for _color, name in pp.KIND_STYLE]
    assert len({color for color, _name in pp.KIND_STYLE_CB}) == len(pp.KIND_STYLE_CB), \
        "the colour-blind palette has two of the same colour"
    assert pp.kind_style(True) is pp.KIND_STYLE_CB
    assert pp.kind_style(False) is pp.KIND_STYLE


class FakeWindow(QWidget):
    """A parent window for the preview dialog that only carries the colour-blind preference."""

    def __init__(self, colour_blind):
        super().__init__()
        self.settings = SimpleNamespace(colour_blind=colour_blind)


def fill_rows(view):
    """The rows of `lines.col` of the FILL move of the first layer, its two vertices."""
    printed = view.preview.layers[0].kind
    printed = printed[printed != gp.TRAVEL]
    first = int(np.flatnonzero(printed == gp.KIND["FILL"])[0])
    return slice(view.lines.start[0] + 2 * first, view.lines.start[0] + 2 * first + 2)


@pytest.mark.parametrize("colour_blind, palette", ((True, "KIND_STYLE_CB"),
                                                   (False, "KIND_STYLE")))
def test_the_preview_colours_the_lines_with_the_chosen_palette(preview, tmp_path,
                                                               colour_blind, palette):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    from volura import print_preview as pp
    path = tmp_path / "x.gcode"
    path.write_text(GCODE, encoding="utf-8")
    win = FakeWindow(colour_blind)
    dlg = pp.PrintPreviewDialog(win, str(path), preview, (0.0, 0.0, 100.0, 100.0), "")
    try:
        assert dlg.colour_blind is colour_blind
        colour = np.array(pp._rgb(getattr(pp, palette)[gp.KIND["FILL"]][0]), dtype=np.float32)
        assert dlg.view.lines.col[fill_rows(dlg.view)] == pytest.approx(np.tile(colour, (2, 1)))
    finally:
        dlg.close()
        win.close()


def test_the_colour_blind_preference_is_stored():
    q = qsettings()
    saved = q.value("colour_blind")
    try:
        from volura.app import Settings
        s = Settings()
        s.colour_blind = not s.colour_blind
        s.save()
        assert Settings().colour_blind is s.colour_blind
    finally:
        if saved is None:
            q.remove("colour_blind")
        else:
            q.setValue("colour_blind", saved)
        q.sync()
