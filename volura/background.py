"""Run a long operation in the background so the window keeps responding.

A task is a plain function ``task(report)`` that runs in a :class:`QThreadPool`
worker. The GUI thread stays free to repaint and to answer the window manager.
When the task takes longer than :data:`PROGRESS_DELAY_MS`, a modal
:class:`QProgressDialog` pops up (so a quick task never flashes a dialog); the
dialog blocks the canvas and the menus while it is up, and shows a Cancel button
where cancelling is safe. The result, the exception or the cancellation is
delivered back on the GUI thread through the ``finished`` / ``failed`` /
``cancelled`` signals, so the caller updates widgets exactly as it would after a
synchronous call.

``report(fraction, text)`` lets the task show progress (``fraction`` in 0..1, or
None for a busy bar) and notice a cancel: it raises :class:`BackgroundCancelled`
as soon as the user pressed Cancel, at the next safe point. A task that never
reports simply runs to its end.

Safety: the task runs off the GUI thread and must not touch Qt widgets, and it
must not read the document while the window could still change it. So from the
start of a job until its end the window takes no input at all (mouse, keys and
their shortcuts, wheel, drops: see :class:`_InputBlocker`), also in the delay
before the dialog shows and after Cancel, and ``MainWindow._busy`` pauses the
SpaceMouse. A cancelled job never delivers its result, also when the task ran to
its end because it had no ``report`` left to stop at.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import (QEvent, QEventLoop, QObject, QRunnable, Qt, QTimer,
                            QThreadPool, Signal, Slot)
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

from .i18n import tr

#: how long a task may run before the progress dialog is shown (milliseconds)
PROGRESS_DELAY_MS = 300

#: jobs that are still running, so Python keeps them (and their signals) alive
_ACTIVE: set["BackgroundJob"] = set()

#: the user input that could change the model while a job reads it
_INPUT = {QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick,
          QEvent.MouseMove, QEvent.Wheel, QEvent.KeyPress, QEvent.KeyRelease,
          QEvent.ShortcutOverride, QEvent.Shortcut, QEvent.DragEnter, QEvent.DragMove,
          QEvent.Drop, QEvent.TabletPress, QEvent.TouchBegin, QEvent.ContextMenu}


class _InputBlocker(QObject):
    """Swallows input for the windows that have a job running, except in its progress dialog."""

    def eventFilter(self, obj, event):
        if not _ACTIVE or event.type() not in _INPUT or not isinstance(obj, QWidget):
            return False
        top = obj.window()
        if isinstance(top, QProgressDialog):
            return False
        busy = {job._window for job in _ACTIVE}
        while top is not None:
            if top in busy:
                return True
            top = top.parentWidget()
        return False


_BLOCKER = None


def _block_input():
    global _BLOCKER
    app = QApplication.instance()
    if _BLOCKER is None and app is not None:
        _BLOCKER = _InputBlocker(app)
        app.installEventFilter(_BLOCKER)


class BackgroundCancelled(Exception):
    """Raised inside a task by ``report`` when the user asked to cancel."""


class _TaskSignals(QObject):
    """Signals of the worker; emitted from the pool thread, delivered on the GUI thread."""

    finished = Signal(object)         # the return value of the task
    failed = Signal(object)           # the exception the task raised
    cancelled = Signal()
    progress = Signal(int, str)       # permille (0..1000) or -1 for a busy bar, text


class _Worker(QRunnable):
    """Runs ``fn(report)`` in the pool and reports the outcome through ``signals``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()
        self.cancel_event = threading.Event()
        self.setAutoDelete(True)

    def report(self, fraction=None, text=None):
        """The progress callback handed to the task; raises on cancel."""
        if self.cancel_event.is_set():
            raise BackgroundCancelled()
        permille = -1 if fraction is None else max(0, min(1000, int(float(fraction) * 1000)))
        self.signals.progress.emit(permille, text or "")
        return self.cancel_event.is_set()

    @Slot()
    def run(self):
        try:
            result = self.fn(self.report)
        except BackgroundCancelled:
            self.signals.cancelled.emit()
        except BaseException as exc:            # a sync error would be shown; keep that behaviour
            self.signals.failed.emit(exc)
        else:
            if self.cancel_event.is_set():      # cancelled after its last report: drop the result
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)


class BackgroundJob(QObject):
    """One background task with an optional delayed progress dialog; lives on the GUI thread."""

    finished = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, window, task, text, can_cancel=True,
                 delay_ms=PROGRESS_DELAY_MS, show_dialog=True):
        super().__init__(window)
        self._window = window
        self._worker = _Worker(task)
        self._can_cancel = can_cancel
        self._delay_ms = delay_ms
        self._show_dialog = show_dialog
        self._text = text
        self._dialog = None
        self._done = False
        self._outcome = None                       # ("ok"|"error"|"cancel", value)
        s = self._worker.signals
        s.finished.connect(self._on_finished)
        s.failed.connect(self._on_failed)
        s.cancelled.connect(self._on_cancelled)
        s.progress.connect(self._on_progress)

    # ------------------------------------------------------------------ state
    @property
    def done(self):
        return self._done

    @property
    def result(self):
        return self._outcome[1] if self._outcome and self._outcome[0] == "ok" else None

    @property
    def error(self):
        return self._outcome[1] if self._outcome and self._outcome[0] == "error" else None

    @property
    def was_cancelled(self):
        return bool(self._outcome and self._outcome[0] == "cancel")

    # ---------------------------------------------------------------- running
    def start(self):
        _ACTIVE.add(self)
        _block_input()
        self._set_busy()
        if self._show_dialog:
            QTimer.singleShot(self._delay_ms, self._show_progress)
        QThreadPool.globalInstance().start(self._worker)
        return self

    def cancel(self):
        """Ask the task to stop; it stops at the next ``report`` (safe point)."""
        self._worker.cancel_event.set()

    def wait(self, timeout_ms=120000):
        """Spin the event loop until the job is over; True when it finished in time."""
        if self._done:
            return True
        loop = QEventLoop(self)
        for sig in (self.finished, self.failed, self.cancelled):
            sig.connect(loop.quit)
        timer = QTimer(loop)
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        loop.exec()
        return self._done

    # ------------------------------------------------------------- GUI slots
    def _show_progress(self):
        if self._done or self._dialog is not None:
            return
        dlg = QProgressDialog(self._text, tr("Cancel") if self._can_cancel else "",
                              0, 0, self._window)
        dlg.setWindowTitle(tr("Working…"))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)                  # we already waited out the delay
        if self._can_cancel:
            dlg.canceled.connect(self._cancel_pressed)
        dlg.show()
        self._dialog = dlg

    def _cancel_pressed(self):
        """Cancel hides a QProgressDialog: keep it up until the task has really stopped."""
        self.cancel()
        if self._dialog is not None and not self._done:
            self._dialog.setCancelButton(None)
            self._dialog.setLabelText(tr("Cancelling…"))
            self._dialog.show()

    def _on_progress(self, permille, text):
        if self._worker.cancel_event.is_set():     # the dialog says "Cancelling…" now
            return
        if text:
            self._text = text
        if self._dialog is None:
            return
        if text:
            self._dialog.setLabelText(text)
        if permille >= 0:
            if self._dialog.maximum() != 1000:
                self._dialog.setRange(0, 1000)
            self._dialog.setValue(permille)

    def _on_finished(self, result):
        self._finish("ok", result)
        self.finished.emit(result)

    def _on_failed(self, exc):
        self._finish("error", exc)
        self.failed.emit(exc)

    def _on_cancelled(self):
        self._finish("cancel", None)
        self.cancelled.emit()

    def _finish(self, kind, value):
        self._done = True
        self._outcome = (kind, value)
        if self._dialog is not None:
            self._dialog.reset()
            self._dialog.deleteLater()
            self._dialog = None
        _ACTIVE.discard(self)
        self._set_busy()
        # the window no longer keeps the job (and its result) alive: only whoever holds it
        self.setParent(None)

    def _set_busy(self):
        """The window is busy while any of its jobs runs (two exports can overlap)."""
        if self._window is not None:
            try:
                self._window._busy = any(job._window is self._window for job in _ACTIVE)
            except Exception:
                pass


def run(window, task, *, text, can_cancel=True, delay_ms=PROGRESS_DELAY_MS,
        on_done=None, on_error=None, on_cancel=None, show_dialog=True, wait=False):
    """Run ``task(report)`` in the background; return the :class:`BackgroundJob`.

    ``on_done(result)``, ``on_error(exc)`` and ``on_cancel()`` run on the GUI
    thread. With ``wait=True`` the call blocks (spinning the event loop) until the
    task is over, so synchronous callers and tests can use the same route.
    """
    job = BackgroundJob(window, task, text, can_cancel=can_cancel,
                        delay_ms=delay_ms, show_dialog=show_dialog)
    if on_done is not None:
        job.finished.connect(on_done)
    if on_error is not None:
        job.failed.connect(on_error)
    if on_cancel is not None:
        job.cancelled.connect(on_cancel)
    job.start()
    if wait:
        job.wait()
    return job
