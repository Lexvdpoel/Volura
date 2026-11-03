"""Het uiterlijk van de 3D-weergave zoals SketchUp 2017 en de muisaanwijzers.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import icons  # noqa: E402
from volura.camera import Camera, horizon_y  # noqa: E402


def make_cam(pitch, ortho=False, w=800, h=600):
    cam = Camera()
    cam.ortho, cam.pitch, cam.width, cam.height = ortho, pitch, w, h
    return cam


def test_horizon_is_in_the_middle_of_the_screen_at_pitch_0():
    """Bij pitch 0 kijgt de camera horizontaal: de horizon loopt precies door het midden."""
    assert abs(horizon_y(make_cam(0.0)) - 300.0) < 1e-6, "de horizon moet op de helft van het scherm liggen"


def test_horizon_falls_outside_the_screen_at_pitch_89():
    """Bijna recht naar beneden kijken: er is geen horizon meer op het scherm (alleen grond)."""
    y = horizon_y(make_cam(89.0))
    assert y < 0 or y > 600, f"bij pitch 89 ligt de horizon buiten het scherm, nu {y}"


def test_horizon_moves_up_and_down_with_the_pitch():
    """Omhoog kijken haalt de horizon uit beeld, naar beneden kijken steekt hem omhoog."""
    assert horizon_y(make_cam(20.0)) < 300.0, "bij een positieve pitch staat de horizon boven het midden"
    assert horizon_y(make_cam(-20.0)) > 300.0, "bij een negatieve pitch staat de horizon onder het midden"


@pytest.mark.parametrize("pitch, ground_only", ((89.99, True), (-89.99, False)))
def test_horizon_of_the_parallel_top_and_bottom_view(pitch, ground_only):
    """Parallel van bovenaf is alleen grond, van onderaf alleen lucht."""
    y = horizon_y(make_cam(pitch, ortho=True))
    assert (y < 0 if ground_only else y > 600), "de bovenaanzicht toont alleen grond, de onderaanzicht alleen lucht"


def test_tool_cursor_is_a_qcursor():
    """SketchUp toont het icoon van het gereedschap als muisaanwijzer."""
    for name in ("line", "eraser", "paint", "pushpull", "move", "rotate", "scale", "tape",
                 "orbit", "pan", "zoom", "select"):
        assert isinstance(icons.cursor(name), QCursor), f"icons.cursor('{name}') moet een QCursor zijn"


def test_every_tool_points_with_its_own_icon():
    """Elk gereedschap verwijst naar een bestaand icoon en geeft een QCursor."""
    from volura.tools import TOOL_CLASSES

    for cls in TOOL_CLASSES:
        assert cls.cursor_icon in icons.SVG, f"{cls.name} verwijst naar een onbekend icoon"
        tool = cls.__new__(cls)
        tool._cursor = None
        assert isinstance(tool.cursor, QCursor), f"{cls.name} geeft geen QCursor"
    assert [c for c in TOOL_CLASSES if c.name == "select"][0].cursor_icon == "select"
