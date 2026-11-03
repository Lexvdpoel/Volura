"""Status bar texts and the Measurements box label per tool, as in SketchUp 2017.

Every tool has a `status_text()` per step and a `vcb_label()` that fits the tool;
the viewport updates both after every mouse or key event.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # run the deferred initial_view now
    w.resize(800, 600)
    w.view.resize(800, 600)
    yield w
    w.doc.dirty = False
    w.close()


def tool(win, name):
    """The tool instance of the viewport, reset to its first step."""
    t = win.view.tools[name]
    t.reset()
    return t


def click(win, x, y):
    """A real left click through the event routing of the viewport."""
    p = QPointF(float(x), float(y))
    win.view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton,
                                         Qt.LeftButton, Qt.NoModifier))


def test_every_tool_has_a_status_text_and_a_label(win):
    from volura.tools import TOOL_CLASSES
    for cls in TOOL_CLASSES:
        t = win.view.tools[cls.name]
        assert isinstance(t.status_text(), str), f"{cls.name}: status_text() is not a string"
        assert t.status_text().strip(), f"{cls.name}: status_text() is empty"
        assert isinstance(t.vcb_label(), str), f"{cls.name}: vcb_label() is not a string"


def test_status_text_changes_after_the_first_line_click(win):
    win.set_tool("line")
    t = win.view.tools["line"]
    before = t.status_text()
    click(win, 400, 300)
    assert t.pts, "the click should have been a start point"
    after = t.status_text()
    assert after != before, "the status text did not change after the first click"
    assert "end point" in after.lower()
    assert win.hint.text() == after            # the status bar follows the step
    t.reset()
    assert t.status_text() == before


def test_line_step_text_without_a_click(win):
    assert tool(win, "line").status_text() == "Select start point."
    assert tool(win, "line").vcb_label() == "Length"


STEP_TEXTS = [
    ("line", {}, "Select start point."),
    ("line", {"pts": [np.zeros(3)]}, "end point"),
    ("rectangle", {}, "Select first corner."),
    ("rectangle", {"p0": np.zeros(3)}, "Select opposite corner or type values."),
    ("circle", {}, "Select center point."),
    ("circle", {"p0": np.zeros(3)}, "Select radius or type a value."),
    ("polygon", {}, "Select center point."),
    ("polygon", {"p0": np.zeros(3)}, "Select radius or type a value."),
    ("arc2", {}, "Select start point."),
    ("arc2", {"p0": np.zeros(3)}, "Select end point or type a length."),
    ("arc2", {"p0": np.zeros(3), "p1": np.array([10.0, 0, 0])}, "bulge"),
    ("pushpull", {}, "Select a face to push or pull."),
    ("pushpull", {"target": (0, 0, np.zeros(3), np.array([0.0, 0, 1]))},
     "Move to push/pull or type a distance. Ctrl = new face."),
    ("move", {}, "Select object to move. Ctrl = copy."),
    ("move", {"base": np.zeros(3)}, "Select target point or type a distance."),
    ("rotate", {}, "Select center point."),
    ("rotate", {"center": np.zeros(3)}, "Select start point of the angle."),
    ("rotate", {"center": np.zeros(3), "ref": np.array([1.0, 0, 0])},
     "Select end point of the angle or type an angle."),
    ("scale", {}, "Select a grip to scale. Ctrl = about center, Shift = uniform."),
    ("tape", {}, "Select start point to measure."),
    ("protractor", {}, "Select center point of the protractor."),
    ("eraser", {}, "Select objects to erase. Ctrl = soften, Shift = hide."),
    ("paint", {}, "Select an object to paint."),
    ("select", {}, "Select objects. Shift = add/remove. Ctrl = add. "
                   "Ctrl+Shift = remove."),
]


@pytest.mark.parametrize("name,state,expected", STEP_TEXTS)
def test_status_text_per_step(win, name, state, expected):
    t = tool(win, name)
    for key, value in state.items():
        setattr(t, key, value)
    assert expected in t.status_text(), f"{name}: {t.status_text()!r}"


LABELS = [
    ("select", {}, ""),
    ("line", {}, "Length"),
    ("rectangle", {}, "Measurements"),
    ("circle", {}, "Sides"),
    ("circle", {"p0": np.zeros(3)}, "Radius"),
    ("polygon", {}, "Sides"),
    ("polygon", {"p0": np.zeros(3)}, "Radius"),
    ("arc2", {}, "Length"),
    ("arc2", {"p0": np.zeros(3), "p1": np.array([10.0, 0, 0])}, "Bulge"),
    ("pushpull", {}, "Distance"),
    ("offset", {}, "Distance"),
    ("move", {}, "Distance"),
    ("rotate", {}, "Angle"),
    ("scale", {}, "Scale factor"),
    ("tape", {}, "Length"),
    ("protractor", {}, "Angle"),
    ("zoom", {}, "Field of View"),
]


@pytest.mark.parametrize("name,state,expected", LABELS)
def test_measurements_box_label(win, name, state, expected):
    t = tool(win, name)
    for key, value in state.items():
        setattr(t, key, value)
    assert t.vcb_label() == expected, f"{name}: {t.vcb_label()!r}"


def test_status_bar_and_measurements_box_follow_the_tool(win):
    win.set_tool("pushpull")
    assert win.hint.text() == win.view.tools["pushpull"].status_text()
    assert win.vcb_label.text() == "Distance"
    win.set_tool("rotate")
    assert win.hint.text() == "Select center point."
    assert win.vcb_label.text() == "Angle"


def test_status_bar_message_stays_until_the_next_event(win):
    """A message of the window is not overwritten until the next event."""
    win.view.set_status("Model cut with a plane")
    assert win.hint.text() == "Model cut with a plane"
    win.set_tool("select")
    assert win.hint.text() == win.view.tools["select"].status_text()
