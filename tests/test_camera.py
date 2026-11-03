"""Camera > Previous/Next, Field of View and Zoom Window.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import (QCoreApplication, QEventLoop, QPoint, QPointF, Qt,  # noqa: E402
                            QTimer)
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = ((50.0, 50.0, 0.0), 400.0, -60.0, 30.0, 35.0, False)


# ------------------------------------------------------------------ Camera

def test_state_is_a_plain_tuple():
    from volura.camera import Camera
    s = Camera().state()
    assert isinstance(s, tuple) and len(s) == 6
    assert len(s[0]) == 3 and all(isinstance(v, float) for v in s[0])
    assert isinstance(s[5], bool)


def test_set_state_restores_the_whole_camera():
    from volura.camera import Camera
    cam = Camera()
    s = cam.state()
    cam.target = np.array([1.0, 2.0, 3.0])
    cam.distance, cam.yaw, cam.pitch, cam.fov, cam.ortho = 12.0, 44.0, -5.0, 70.0, True
    cam.set_state(s)
    assert cam.state() == s
    assert cam.ortho is False
    cam.target[0] = 99.0                        # the stored state may not be shared
    assert cam.state() != s
    cam.set_state(s)
    assert cam.state() == s


# ------------------------------------------------------------------ helpers

class Drag:
    """Enough of a mouse event for the zoom window tool while dragging."""

    def buttons(self):
        return Qt.LeftButton


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # run the deferred initial_view now
    set_size(w, 800, 600)
    w.view.cam.set_state(BASE)
    w.view.cam_hist[:] = [w.view.cam.state()]
    w.view.cam_idx = 0
    w.on_cam_history(False, False)
    yield w
    w.doc.dirty = False
    w.close()


def set_size(win, width, height):
    """The size the camera projects with; paintGL keeps this in sync when shown."""
    win.view.resize(width, height)
    win.view.cam.width, win.view.cam.height = width, height


def change(win, **kw):
    """Change the camera and end the movement, as releasing the mouse does."""
    cam = win.view.cam
    for key, value in kw.items():
        if key == "target":
            cam.target = np.array(value, dtype=float)
        else:
            setattr(cam, key, value)
    win.view.commit_camera()
    return cam.state()


def spin(ms):
    """Let Qt run for a while, so its timers fire."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wheel(win, x, y, steps):
    ev = QWheelEvent(QPointF(x, y), QPointF(x, y), QPoint(0, 0), QPoint(0, steps),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    win.view.wheelEvent(ev)


def drag(win, p0, p1):
    tool = win.view.tool
    tool.mouse_press(None, *p0)
    tool.mouse_move(Drag(), *p1)
    tool.mouse_release(None, *p1)


# ------------------------------------------------------------------ history

def test_starts_without_previous_or_next(win):
    assert len(win.view.cam_hist) == 1
    assert not win.a_prev_view.isEnabled()
    assert not win.a_next_view.isEnabled()
    win.a_prev_view.trigger()
    win.a_next_view.trigger()
    assert win.view.cam.state() == BASE


def test_each_ended_movement_adds_one_state(win):
    change(win, distance=300.0)
    assert win.a_prev_view.isEnabled() and not win.a_next_view.isEnabled()
    change(win, distance=200.0)
    change(win, yaw=15.0)
    assert len(win.view.cam_hist) == 4
    assert win.view.cam_idx == 3


def test_the_same_camera_is_not_recorded_twice(win):
    change(win, distance=300.0)
    win.view.commit_camera()
    win.view.commit_camera()
    assert len(win.view.cam_hist) == 2


def test_previous_goes_back_next_goes_forward(win):
    s1 = change(win, distance=300.0)
    s2 = change(win, yaw=15.0)
    s3 = change(win, target=(0.0, 0.0, 5.0))
    win.a_prev_view.trigger()
    assert win.view.cam.state() == s2
    win.a_prev_view.trigger()
    assert win.view.cam.state() == s1
    assert win.a_prev_view.isEnabled() and win.a_next_view.isEnabled()
    win.a_next_view.trigger()
    assert win.view.cam.state() == s2
    win.a_next_view.trigger()
    assert win.view.cam.state() == s3
    assert not win.a_next_view.isEnabled()
    win.a_next_view.trigger()
    assert win.view.cam.state() == s3


def test_previous_also_restores_field_of_view_and_projection(win):
    change(win, fov=60.0, ortho=True)
    assert win.view.cam.ortho and win.view.cam.fov == 60.0
    win.a_prev_view.trigger()
    assert not win.view.cam.ortho and win.view.cam.fov == 35.0


def test_new_movement_after_previous_clears_next(win):
    change(win, distance=300.0)
    change(win, distance=200.0)
    win.a_prev_view.trigger()
    assert win.a_next_view.isEnabled()
    change(win, distance=100.0)
    assert not win.a_next_view.isEnabled()
    assert len(win.view.cam_hist) == 3


def test_at_most_50_states(win):
    for i in range(60):
        change(win, distance=100.0 + i)
    assert len(win.view.cam_hist) == win.view.CAM_HISTORY_MAX == 50
    assert win.view.cam.state() == win.view.cam_hist[-1]
    for _ in range(60):
        win.a_prev_view.trigger()
    assert not win.a_prev_view.isEnabled()
    assert win.view.cam.state() == win.view.cam_hist[0]


def test_standard_view_zoom_and_projection_store_a_state(win):
    n = len(win.view.cam_hist)
    win.views["top"].trigger()
    assert len(win.view.cam_hist) == n + 1
    win.view.zoom_extents()
    assert len(win.view.cam_hist) == n + 2
    win.a_ortho.trigger()
    assert win.view.cam.ortho and len(win.view.cam_hist) == n + 3
    assert win.view.cam_hist[-1][5] is True


def test_scrolling_bundles_into_one_state(win):
    n = len(win.view.cam_hist)
    for _ in range(4):
        wheel(win, 400, 300, 120)
    assert len(win.view.cam_hist) == n              # nothing stored yet
    assert win.view._cam_settle.isActive()
    spin(win.view.CAM_SCROLL_SETTLE_MS + 300)
    assert len(win.view.cam_hist) == n + 1
    assert win.view.cam.state() == win.view.cam_hist[-1]
    assert win.view.cam.distance < BASE[1]


# ------------------------------------------------------------------ field of view

def test_field_of_view_sets_zoom_tool_and_measurements(win):
    win.a_fov.trigger()
    assert win.view.tool.name == "zoom"
    assert win.vcb_label.text() == "Field of View"
    assert win.vcb.text() == "35.0 deg"
    win.vcb.setText("60")
    win.on_vcb_return()
    assert win.view.cam.fov == pytest.approx(60.0)
    assert win.vcb.text() == "60.0 deg"
    assert win.view.cam_hist[-1][4] == pytest.approx(60.0)


def test_field_of_view_stays_within_5_to_120_degrees(win):
    win.a_fov.trigger()
    win.view.submit_vcb("500")
    assert win.view.cam.fov == 120.0
    win.view.submit_vcb("1")
    assert win.view.cam.fov == 5.0


# ------------------------------------------------------------------ zoom window

def test_zoom_window_returns_to_the_previous_tool(win):
    win.set_tool("select")
    win.a_zoom_window.trigger()
    assert win.view.tool.name == "zoomwindow"
    assert win.view.prev_tool == "select"
    drag(win, (200, 150), (600, 450))
    assert win.view.tool.name == "select"


def test_zoom_window_zooms_to_the_frame(win):
    win.set_tool("select")
    win.a_zoom_window.trigger()
    d0, t0 = win.view.cam.distance, win.view.cam.target.copy()
    drag(win, (400, 300), (800, 600))               # the right-bottom quarter
    cam = win.view.cam
    assert cam.distance == pytest.approx(d0 * 0.5)
    s, ok = cam.project(cam.target)
    assert ok[0] and np.allclose(s[0], [400, 300], atol=1e-6)   # the frame centre is centred
    assert np.linalg.norm(cam.target - t0) > 1.0                # it looked off-centre
    assert win.view.cam.state() == win.view.cam_hist[-1]


def test_zoom_window_smaller_than_two_pixels_does_nothing(win):
    win.set_tool("zoomwindow")
    drag(win, (300, 300), (301, 301))
    assert win.view.cam.state() == BASE
    assert win.view.tool.name == "select"


def test_zoom_window_escapes_with_esc(win):
    win.set_tool("select")
    win.a_zoom_window.trigger()
    tool = win.view.tool
    tool.mouse_press(None, 100, 100)
    tool.mouse_move(Drag(), 400, 400)
    assert tool.busy()
    tool.cancel()
    assert not tool.busy()
    assert win.view.tool.name == "select"
    assert win.view.cam.state() == BASE
