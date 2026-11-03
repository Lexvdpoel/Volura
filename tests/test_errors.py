"""Fault tolerance of the interface: a failing handler must never crash Volura.

A tool whose mouse press raises, a paint that fails, a save that runs into an
error: everything goes to volura.log with a line in the status bar, and the
document keeps the state it had before the half finished edit.

Runs without a display (Qt offscreen platform).
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, errors, fileio  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.tools import Tool  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

GROUND = Plane((0, 0, 0), (0, 0, 1))
RECT = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)


# ------------------------------------------------------------------- stand-ins

class Ev:
    """Stands in for a QMouseEvent."""

    def __init__(self, button=Qt.LeftButton):
        self._button = button

    def position(self):
        return QPointF(20.0, 20.0)

    def globalPosition(self):
        return QPointF(20.0, 20.0)

    def button(self):
        return self._button

    def modifiers(self):
        return Qt.NoModifier


class KeyEv:
    """Stands in for a QKeyEvent."""

    def key(self):
        return Qt.Key_X

    def text(self):
        return "x"


class WheelEv:
    """A scroll wheel that raises as soon as the handler reads it."""

    def angleDelta(self):
        raise RuntimeError("scrollwiel mislukt")

    def position(self):
        return QPointF(10.0, 10.0)


class BadEvent(QEvent):
    """An event that raises while the handler looks at its type."""

    def __init__(self):
        super().__init__(QEvent.ShortcutOverride)

    def type(self):
        raise RuntimeError("event mislukt")


class BoomTool(Tool):
    name = "boom"
    status = "Kapot gereedschap"

    def mouse_press(self, e, x, y):
        raise RuntimeError("muisdruk mislukt")


class HalfEditTool(Tool):
    """Draws a rectangle and then crashes: the edit is only half finished."""

    name = "half"
    status = "Halve bewerking"

    def mouse_press(self, e, x, y):
        self.doc.checkpoint()
        self.doc.root.add_shape(GROUND, [RECT])
        assert self.doc.root.F, "rechthoek gaf geen vlak"
        raise RuntimeError("mislukt halverwege")


class PaintTool(Tool):
    name = "paintboom"
    status = "Tekenen mislukt"

    def draw(self, view):
        raise RuntimeError("tekenen mislukt")


class MoveTool(Tool):
    name = "moveboom"
    status = "Bewegen mislukt"

    def mouse_move(self, e, x, y):
        raise RuntimeError("bewegen mislukt")

    def mouse_release(self, e, x, y):
        raise RuntimeError("loslaten mislukt")

    def double_click(self, e, x, y):
        raise RuntimeError("dubbelklik mislukt")

    def key_press(self, e):
        raise RuntimeError("toets mislukt")


class FakeBox:
    """Stands in for QMessageBox, so a test sees the messages without a dialog."""

    Yes, No = 1, 0

    def __init__(self):
        self.asked = []

    def question(self, *args, **kwargs):
        self.asked.append(("question",) + args[1:])
        return self.Yes

    def critical(self, *args, **kwargs):
        self.asked.append(("critical",) + args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(("information",) + args[1:])

    def warning(self, *args, **kwargs):
        self.asked.append(("warning",) + args[1:])


# --------------------------------------------------------------------- fixtures

@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test."""
    monkeypatch.setattr(errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def install(win, tool):
    """Put a tool on the viewport without going through the menus."""
    win.view.tools[tool.name] = tool
    win.view.tool = tool
    return tool


def draw_rect(doc):
    created = doc.root.add_shape(GROUND, [RECT])
    assert created, "rechthoek gaf geen vlak"
    return created[0]


def log_text():
    """What Volura wrote to volura.log so far."""
    with open(errors.log_path(), encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------- event handlers

def test_failing_mouse_press_does_not_raise(win):
    install(win, BoomTool(win.view))
    win.view.mousePressEvent(Ev())          # the app must simply go on
    assert win.hint.text() == errors.ERROR_STATUS


def test_failing_mouse_press_is_logged(win):
    install(win, BoomTool(win.view))
    win.view.mousePressEvent(Ev())
    text = log_text()
    assert "RuntimeError" in text and "muisdruk mislukt" in text


def test_half_finished_edit_is_undone(win):
    install(win, HalfEditTool(win.view))
    win.view.mousePressEvent(Ev())
    assert not win.doc.root.F, "de halve bewerking is achtergelaten"
    assert not win.doc.undo_stack, "het checkpoint van de mislukte bewerking is gebleven"


def test_tool_is_reset_after_a_failure(win):
    tool = install(win, HalfEditTool(win.view))
    tool.vcb_value = "25"
    win.view.mousePressEvent(Ev())
    assert tool.vcb_value == ""


def test_move_release_doubleclick_and_key_failures_are_caught(win):
    install(win, MoveTool(win.view))
    win.view.mouseMoveEvent(Ev())
    win.view.mouseReleaseEvent(Ev())
    win.view.mouseDoubleClickEvent(Ev())
    win.view.keyPressEvent(KeyEv())
    win.view.wheelEvent(WheelEv())
    assert win.hint.text() == errors.ERROR_STATUS


def test_failing_paint_does_not_raise(win):
    install(win, PaintTool(win.view))
    win.view.paintGL()                       # never an "Error calling ..." dialog
    assert "paintGL" in log_text()


def test_event_always_returns_a_bool(win):
    assert win.view.event(BadEvent()) is False
    assert isinstance(win.view.event(BadEvent()), bool)


# ------------------------------------------------------------------- the log

def test_excepthook_logs_and_asks_nothing_offscreen(win, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    try:
        raise ValueError("buiten de handlers mislukt")
    except ValueError:
        app_module.excepthook(*sys.exc_info())
    assert "buiten de handlers mislukt" in log_text()
    assert not box.asked, "offscreen hoort geen melding te tonen"


# ------------------------------------------------------------------ OpenGL

def fake_gl(monkeypatch, errors_to_return):
    """A stand-in for PyOpenGL that hands out the given error codes."""
    calls = []

    class FakeGL:
        GL_NO_ERROR = 0

        @staticmethod
        def glGetError():
            code = errors_to_return[len(calls)] if len(calls) < len(errors_to_return) else 0
            calls.append(code)
            return code

    stub = types.ModuleType("OpenGL")
    stub.GL = FakeGL
    monkeypatch.setitem(sys.modules, "OpenGL", stub)
    return calls


def test_initialize_gl_empties_the_error_queue(monkeypatch):
    calls = fake_gl(monkeypatch, [1282, 1280, 1286])
    assert errors.clear_gl_errors() == 3
    assert len(calls) == 4, "er wordt te veel of te weinig glGetError gedaan"


def test_initialize_gl_drains_at_most_ten_errors(monkeypatch):
    calls = fake_gl(monkeypatch, [1282] * 50)
    assert errors.clear_gl_errors() == 10
    assert len(calls) == 10


def test_clearing_gl_errors_survives_a_broken_gl(monkeypatch):
    calls = fake_gl(monkeypatch, [])

    class Boom:
        GL_NO_ERROR = 0

        @staticmethod
        def glGetError():
            calls.append(1)
            raise RuntimeError("geen GL-context")

    sys.modules["OpenGL"].GL = Boom
    assert errors.clear_gl_errors() == 0


# ---------------------------------------------------------------------- files

def test_failed_save_keeps_the_old_file(win, monkeypatch, tmp_path):
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    draw_rect(win.doc)
    win.doc.path = str(tmp_path / "model.volura")
    assert win.file_save() is True
    with open(win.doc.path, "rb") as f:
        before = f.read()

    def kapot(path, files):
        raise OSError("schijf vol")

    monkeypatch.setattr(fileio, "write_zip", kapot)
    assert win.file_save() is False
    with open(win.doc.path, "rb") as f:
        assert f.read() == before, "het bestaande model is beschadigd"
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")], "tijdelijk bestand achtergelaten"
    assert any(kind == "critical" for kind, *_ in box.asked)


def test_save_writes_a_readable_file(win, tmp_path):
    from volura.model import Document
    draw_rect(win.doc)
    win.doc.path = str(tmp_path / "model.volura")
    assert win.file_save() is True
    other = Document()
    fileio.load_document(other, win.doc.path)
    assert len(other.root.F) == len(win.doc.root.F)


# ------------------------------------------------------------------- autosave

def test_autosave_only_writes_a_changed_document(win, tmp_path):
    target = tmp_path / "autosave.volura"
    assert win.autosave() is False and not target.exists()
    win.doc.checkpoint()                 # something changed
    assert win.autosave() is True
    assert target.exists()


def test_autosave_is_gone_after_a_normal_exit(win, tmp_path):
    win.doc.checkpoint()
    assert win.autosave() is True
    win.close()
    assert not (tmp_path / "autosave.volura").exists()


def test_autosave_restore_is_not_asked_offscreen(win, tmp_path, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    (tmp_path / "autosave.volura").write_bytes(b"not a model")
    assert win.offer_autosave_restore() is False
    assert not box.asked


def test_restore_autosave_loads_the_model(win, tmp_path):
    draw_rect(win.doc)
    win.doc.checkpoint()
    assert win.autosave() is True
    win.doc.reset()
    win.doc.notify()
    assert not win.doc.root.F
    assert win.restore_autosave() is True
    assert win.doc.root.F, "het automatisch opgeslagen model is niet geladen"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
