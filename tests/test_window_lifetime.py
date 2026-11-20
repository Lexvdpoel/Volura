"""A window that is deleted right after it was made does not leave work behind.

The main window schedules its first steps (the starting view, the welcome, the autosave
offer, the update check, the SpaceMouse) for just after it is built. Those calls belong
to the window: when it is deleted first, they must be dropped instead of reaching a
deleted window. An error in a Qt callback is only handed to sys.excepthook, so that is
where we look.

Run with:  python -m pytest tests/test_window_lifetime.py
"""
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def test_the_first_steps_of_a_deleted_window_are_dropped(monkeypatch):
    # what earlier tests of this process left waiting comes due first, before we listen:
    # only the steps of the window made here are what this test is about
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    for _ in range(5):
        QCoreApplication.processEvents()
    errors = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda *exc: errors.append("".join(traceback.format_exception(*exc))))
    w = app_module.MainWindow()          # schedules its first steps, not run yet
    w.doc.dirty = False
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    for _ in range(5):
        QCoreApplication.processEvents()  # the scheduled calls come due: no error
    assert errors == [], "\n".join(errors)
