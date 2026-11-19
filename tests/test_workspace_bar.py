"""The workspace bar lines its right end up with the viewer, and lets go when it is gone.

Run with:  python -m pytest tests/test_workspace_bar.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget  # noqa: E402

from volura.workspaces import WorkspaceBar  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _window():
    top = QWidget()
    lay = QVBoxLayout(top)
    row = QHBoxLayout()
    viewer = QWidget()
    viewer.setMinimumSize(400, 300)
    row.addWidget(viewer, 1)
    side = QWidget()
    side.setFixedWidth(150)
    row.addWidget(side)
    lay.addLayout(row)
    bar = WorkspaceBar(parent=top)
    lay.addWidget(bar)
    bar.follow(viewer)
    top.resize(800, 500)
    top.show()
    return top, viewer, bar


def test_the_bar_ends_where_the_viewer_ends():
    top, viewer, bar = _window()
    try:
        for _ in range(5):
            QCoreApplication.processEvents()
        right = viewer.mapTo(top, viewer.rect().topRight()).x() + 1
        assert bar.mapTo(top, bar.rect().topLeft()).x() + bar.width() == right
    finally:
        top.close()
        top.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_a_pending_alignment_does_nothing_once_the_bar_is_deleted(monkeypatch):
    """A resize schedules the alignment for later; when the window is deleted before
    that moment, the call must be dropped instead of reaching a deleted widget (it
    raised on every following event of the application before). An error in a Qt
    callback is only handed to sys.excepthook, so that is where we look."""
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc[1]))
    top, viewer, bar = _window()
    for _ in range(3):
        QCoreApplication.processEvents()
    top.resize(900, 500)                     # schedules _align on the next round
    top.close()
    top.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    for _ in range(5):
        QCoreApplication.processEvents()     # the scheduled call comes due: no error
    assert errors == []
