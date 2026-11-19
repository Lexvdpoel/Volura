"""The row at the bottom of the window: the workspace tabs and the text on the left, the
3D | 2D switch ending where the viewer ends, the measurements under the side bar."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QStatusBar  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    w.resize(1280, 800)
    w.show()
    for _ in range(5):
        QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def right(w, widget):
    return widget.mapTo(w, QPoint(widget.width(), 0)).x()


def settle():
    for _ in range(5):
        QCoreApplication.processEvents()


def test_everything_is_on_one_row_in_the_status_bar(win):
    sb = win.findChild(QStatusBar)
    for widget in (win.workspace_bar, win.hint, win.vcb, win.vcb_label):
        assert sb.isAncestorOf(widget)


def test_the_switch_ends_where_the_viewer_ends(win):
    settle()
    assert right(win, win.mode_button) == right(win, win.central)


def test_the_switch_follows_the_viewer_when_the_window_changes(win):
    win.resize(1100, 800)
    settle()
    assert right(win, win.mode_button) == right(win, win.central)


def test_the_measurements_stand_right_of_the_viewer(win):
    settle()
    assert win.vcb.mapTo(win, QPoint(0, 0)).x() >= right(win, win.central)
