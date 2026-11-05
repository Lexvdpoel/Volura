"""Long operations in the background: the window keeps responding, results stay correct.

volura/background.py runs a task in a worker thread and brings the result, the
error or a cancel back to the GUI thread. These tests (Qt offscreen) check that:

* an export through the background route writes exactly the same file as the
  direct, synchronous route;
* the event loop keeps running while a task is busy (a QTimer still fires);
* an error in the task is shown with the same message as the synchronous route;
* cancelling leaves no half-written file behind;
* the document is not changed by a background task, and the window is marked
  busy for the duration of the job (so the canvas and the SpaceMouse stay still).
"""
import hashlib
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QObject, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget  # noqa: E402

from volura import app as app_module, background, fileio, ops  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


class FakeBox:
    """Stands in for QMessageBox, so a test sees the messages without a dialog."""

    def __init__(self):
        self.asked = []

    def critical(self, *args, **kwargs):
        self.asked.append(("critical",) + args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(("information",) + args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def add_cube(win, size=(10.0, 10.0, 10.0)):
    """A closed volume, so there is something to export."""
    return ops.group_from_manifold(win.doc, mf.Manifold.cube(size), "#f2f2f2", "cube")


def digest(ents):
    h = hashlib.sha256()
    for k in sorted(ents.V):
        h.update(np.asarray(ents.V[k], dtype=float).tobytes())
    for k in sorted(ents.F):
        h.update(repr(ents.F[k].loops).encode())
    return h.hexdigest()


def snapshot(doc):
    """A structural fingerprint of the whole document."""
    groups = [(g.id, g.name, g.color, g.visible, g.locked, digest(g.ents)) for g in doc.groups]
    return (groups, digest(doc.root), doc.dirty, doc.path, doc.next_gid, doc.group_counter)


def pump_until(predicate, timeout_ms=8000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------- results

def test_background_export_writes_the_same_file_as_the_direct_route(win, tmp_path):
    add_cube(win)
    direct = tmp_path / "a" / "model.stl"
    bg = tmp_path / "b" / "model.stl"
    direct.parent.mkdir()
    bg.parent.mkdir()

    n_direct = fileio.export_stl(str(direct), ops.parts(win.doc))

    job = background.run(win, lambda report: fileio.export_stl(str(bg), ops.parts(win.doc)),
                         text="Exporting STL…", show_dialog=False, wait=True)
    assert job.error is None, job.error
    assert job.result == n_direct
    assert bg.read_bytes() == direct.read_bytes(), "the background file differs"


def test_result_and_error_come_back_on_the_gui_thread(win, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)

    def boom(report):
        return 1 / 0

    job = background.run(win, boom, text="Working…", show_dialog=False,
                         on_error=win._export_failed, wait=True)
    assert job.error is not None
    assert box.asked == [("critical", "Exporting Failed", "division by zero")], box.asked


# --------------------------------------------------------------- the loop runs

def test_event_loop_runs_while_a_task_is_busy(win):
    started = threading.Event()

    def slow(report):
        started.set()
        for i in range(60):
            report(i / 60)
            time.sleep(0.01)

    job = background.run(win, slow, text="Working…", show_dialog=False)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    assert job.wait(10000), "the task did not finish in time"
    timer.stop()
    assert ticks, "the event loop did not run while the task was busy"


def test_progress_dialog_appears_after_the_delay(win):
    def slow(report):
        for i in range(40):
            report(i / 40, "Exporting STL…")
            time.sleep(0.02)

    job = background.run(win, slow, text="Exporting STL…", delay_ms=20, show_dialog=True)
    seen = []
    timer = QTimer()
    timer.timeout.connect(lambda: seen.extend(win.findChildren(QProgressDialog)))
    timer.start(15)
    assert job.wait(10000)
    timer.stop()
    assert seen, "no progress dialog appeared for a slow task"
    assert any("Exporting STL" in d.labelText() for d in seen)


def test_no_dialog_for_a_quick_task(win):
    job = background.run(win, lambda report: 7, text="Working…", delay_ms=50,
                         show_dialog=True, wait=True)
    assert job.result == 7
    assert not win.findChildren(QProgressDialog), "a quick task must not flash a dialog"


# ------------------------------------------------------------------- cancel

def test_cancel_leaves_no_half_file(win, tmp_path):
    target = tmp_path / "out.stl"
    started = threading.Event()

    def slow(report):
        started.set()
        for i in range(200):
            report(i / 200)                 # raises BackgroundCancelled once cancelled
            time.sleep(0.01)
        target.write_bytes(b"a complete file")   # only reached when never cancelled

    job = background.run(win, slow, text="Working…", show_dialog=False)
    while not started.is_set():
        _app.processEvents()
    job.cancel()
    assert job.wait(10000)
    assert job.was_cancelled
    assert not target.exists(), "cancelling left a file behind"
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"


# ------------------------------------------------------- the document stays put

def test_document_is_unchanged_during_a_background_task(win):
    add_cube(win)
    win.doc.notify()
    win.doc.dirty = False
    before = snapshot(win.doc)
    busy = []

    def read_model(report):
        busy.append(win._busy)              # the window is busy while the task reads it
        return len(ops.parts(win.doc))

    job = background.run(win, read_model, text="Preparing…", show_dialog=False, wait=True)
    assert job.error is None, job.error
    assert job.result >= 1
    assert busy == [True], "the window was not busy while the task ran"
    assert not win._busy, "the busy flag was not cleared when the task finished"
    assert snapshot(win.doc) == before, "the background task changed the document"


def test_synchronous_export_objects_still_works(win):
    add_cube(win)
    parts = win.export_objects()
    assert parts and all(p.manifold is not None for p in parts)


# ------------------------------------------------------- the app uses the route

def test_import_stl_runs_in_the_background(win, tmp_path):
    stl = tmp_path / "cube.stl"
    fileio.write_stl(str(stl), mf.Manifold.cube((10.0, 10.0, 10.0)))
    before = len(win.doc.groups)
    job = win.import_stl(str(stl))                 # reads the file off the GUI thread
    assert pump_until(lambda: len(win.doc.groups) == before + 1), "the group never appeared"
    job.wait()
    assert job.error is None, job.error
    assert abs(win.doc.groups[-1].ents.to_manifold().volume() - 1000.0) < 1e-3


def test_interactive_vector_export_matches_the_sync_route(win, tmp_path, monkeypatch):
    add_cube(win)
    opts = {"source": "top"}
    sync_path = tmp_path / "sync.svg"
    assert win.export_vector("svg", path=str(sync_path), options=opts) == str(sync_path)

    bg_path = tmp_path / "bg.svg"

    class FakeDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return str(bg_path), ""

    monkeypatch.setattr(app_module, "QFileDialog", FakeDialog)
    assert win.export_vector("svg", options=opts) is None, "interactive export returns right away"
    assert pump_until(bg_path.exists), "the background vector export wrote nothing"
    assert bg_path.read_bytes() == sync_path.read_bytes(), "the background SVG differs"


# ------------------------------------------------- cancel, input and clean-up

def held_task(started, release):
    """A task that reports once and then waits, as a long read without safe points."""
    def task(report):
        report(0.0, "Working…")
        started.set()
        release.wait(10)
        return 42
    return task


def test_cancel_after_the_last_report_drops_the_result(win):
    started, release = threading.Event(), threading.Event()
    done = []
    job = background.run(win, held_task(started, release), text="Working…",
                         show_dialog=False, on_done=done.append)
    assert pump_until(started.is_set)
    job.cancel()
    release.set()
    assert job.wait(10000)
    assert job.was_cancelled
    assert job.result is None
    assert done == [], "a cancelled job still delivered its result"


def test_cancel_keeps_the_dialog_up_until_the_task_stops(win):
    started, release = threading.Event(), threading.Event()
    job = background.run(win, held_task(started, release), text="Working…", delay_ms=0)
    assert pump_until(lambda: bool(win.findChildren(QProgressDialog)))
    dlg = win.findChildren(QProgressDialog)[0]
    dlg.canceled.emit()                       # what the Cancel button does (it then hides the dialog)
    _app.processEvents()
    assert dlg.isVisible(), "the dialog went away while the task still ran"
    assert dlg.labelText() == "Cancelling…"
    release.set()
    assert job.wait(10000)
    assert job.was_cancelled
    assert pump_until(lambda: not win.findChildren(QProgressDialog)), "the dialog stayed"


class Recorder(QObject):
    """Counts the key presses that reach a widget."""

    def __init__(self):
        super().__init__()
        self.keys = 0

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.keys += 1
        return False


def test_the_window_takes_no_input_while_a_job_runs(win):
    target = QWidget(win)
    rec = Recorder()
    target.installEventFilter(rec)

    def press():
        QApplication.sendEvent(target, QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))

    started, release = threading.Event(), threading.Event()
    job = background.run(win, held_task(started, release), text="Working…", delay_ms=60000)
    assert pump_until(started.is_set)
    press()                                   # before the dialog shows: blocked as well
    assert rec.keys == 0, "a key reached the window while a job read the model"
    release.set()
    assert job.wait(10000)
    press()
    assert rec.keys == 1, "the window takes no input after the job"


def test_the_window_stays_busy_until_its_last_job_ends(win):
    s1, r1, s2, r2 = (threading.Event() for _ in range(4))
    j1 = background.run(win, held_task(s1, r1), text="Working…", show_dialog=False)
    j2 = background.run(win, held_task(s2, r2), text="Working…", show_dialog=False)
    assert pump_until(lambda: s1.is_set() and s2.is_set())
    r1.set()
    assert j1.wait(10000)
    assert win._busy, "the first job to end cleared the busy flag of the other"
    r2.set()
    assert j2.wait(10000)
    assert not win._busy


def test_a_finished_job_is_not_kept_by_the_window(win):
    job = background.run(win, lambda report: 7, text="Working…", show_dialog=False, wait=True)
    assert job.result == 7
    assert job.parent() is None
    assert job not in win.findChildren(background.BackgroundJob)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
