"""The look of the 3D view as in SketchUp 2017, and the mouse cursors.

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
    """At pitch 0 the camera looks horizontally: the horizon runs exactly through the middle."""
    assert abs(horizon_y(make_cam(0.0)) - 300.0) < 1e-6, "the horizon must be halfway down the screen"


def test_horizon_falls_outside_the_screen_at_pitch_89():
    """Looking almost straight down: there is no horizon on the screen anymore (only ground)."""
    y = horizon_y(make_cam(89.0))
    assert y < 0 or y > 600, f"at pitch 89 the horizon is off the screen, now {y}"


def test_horizon_moves_up_and_down_with_the_pitch():
    """Looking up moves the horizon down the screen, looking down moves it up."""
    assert horizon_y(make_cam(20.0)) < 300.0, "at a positive pitch the horizon is above the middle"
    assert horizon_y(make_cam(-20.0)) > 300.0, "at a negative pitch the horizon is below the middle"


@pytest.mark.parametrize("pitch, ground_only", ((89.99, True), (-89.99, False)))
def test_horizon_of_the_parallel_top_and_bottom_view(pitch, ground_only):
    """Parallel from above is only ground, from below only sky."""
    y = horizon_y(make_cam(pitch, ortho=True))
    assert (y < 0 if ground_only else y > 600), "the top view shows only ground, the bottom view only sky"


def test_tool_cursor_is_a_qcursor():
    """SketchUp shows the tool icon as the mouse cursor."""
    for name in ("line", "eraser", "paint", "pushpull", "move", "rotate", "scale", "tape",
                 "orbit", "pan", "zoom", "select"):
        assert isinstance(icons.cursor(name), QCursor), f"icons.cursor('{name}') must be a QCursor"


def test_every_tool_points_with_its_own_icon():
    """Every tool refers to an existing icon and gives a QCursor."""
    from volura.tools import TOOL_CLASSES

    for cls in TOOL_CLASSES:
        assert cls.cursor_icon in icons.SVG, f"{cls.name} refers to an unknown icon"
        tool = cls.__new__(cls)
        tool._cursor = None
        assert isinstance(tool.cursor, QCursor), f"{cls.name} gives no QCursor"
    assert [c for c in TOOL_CLASSES if c.name == "select"][0].cursor_icon == "select"
