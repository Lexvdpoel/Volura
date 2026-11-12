"""The line width of Window > Preferences > Display, for people who see thin lines badly.

The setting is one factor (`Settings.line_scale`) that multiplies every line width of the
3D view and of the worksheet: edges, guides, axes, the sheet and the outlines on it.
Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.

Run with:  python -m pytest tests/test_line_width.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from OpenGL import GL as real_gl  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

CHOICES = [("Normal", 1.0), ("Thick", 1.5), ("Thicker", 2.0), ("Thickest", 3.0)]


@pytest.fixture(scope="module")
def win():
    q = qsettings()
    saved = q.value("line_scale")
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()
    if saved is None:
        q.remove("line_scale")
    else:
        q.setValue("line_scale", saved)
    q.sync()


class Mock:
    """Stands in for OpenGL: remembers every call, hands out the real GL constants."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("GL_"):
            return getattr(real_gl, name)

        def call(*args):
            self.calls.append((name, args))
        return call


def asked(win, monkeypatch, line_scale, call, draw):
    """The arguments of `call` (for example "glLineWidth") while `draw` runs at that setting."""
    from volura import viewport as vp
    m = Mock()
    monkeypatch.setattr(vp, "gl", m)
    was = win.settings.line_scale
    win.settings.line_scale = line_scale
    try:
        draw()
    finally:
        win.settings.line_scale = was
    return [a[0] for n, a in m.calls if n == call]


# ------------------------------------------------------------------- the setting

def test_the_line_scale_is_stored_like_the_other_settings():
    """Normal by default, back from the settings after a save, and never outside 1..3."""
    q = qsettings()
    q.remove("line_scale")
    assert app_module.Settings().line_scale == pytest.approx(1.0), "Normal is the default"
    s = app_module.Settings()
    s.line_scale = 2.0
    s.save()
    assert app_module.Settings().line_scale == pytest.approx(2.0)
    for stored, want in ((9.0, 3.0), (0.2, 1.0), ("thick", 1.0)):
        q.setValue("line_scale", stored)
        assert app_module.Settings().line_scale == pytest.approx(want), f"{stored!r} in the settings"
    q.remove("line_scale")
    q.sync()


# ------------------------------------------------------------------- the 3D view

def test_every_line_of_the_3d_view_grows(win, monkeypatch):
    """Viewport._prim multiplies the width of every edge, guide and axis by the setting."""
    def draw():
        win.view.draw_lines([[0, 0, 0], [10, 10, 10]], (0.1, 0.1, 0.1, 1), 1.5)

    assert asked(win, monkeypatch, 1.0, "glLineWidth", draw) == [1.5]
    assert asked(win, monkeypatch, 2.0, "glLineWidth", draw) == [3.0]
    assert asked(win, monkeypatch, 3.0, "glLineWidth", draw) == [4.5]


def test_the_points_of_the_3d_view_grow_too(win, monkeypatch):
    """The markers of the inference engine are points; they grow along with the lines."""
    def draw():
        win.view.draw_points([[0, 0, 0]], (1.0, 0.0, 0.0, 1.0), 8.0)

    assert asked(win, monkeypatch, 1.0, "glPointSize", draw) == [8.0]
    assert asked(win, monkeypatch, 2.0, "glPointSize", draw) == [16.0]


# ------------------------------------------------------------------- the worksheet

def test_the_worksheet_draws_its_outlines_thicker(win):
    """The pens of the 2D worksheet multiply their fixed width by the same setting."""
    path = win.doc.sheet.add_path(np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]),
                                  closed=True)
    img = QImage(120, 120, QImage.Format_ARGB32)
    was = win.settings.line_scale

    def drawn(value):
        """The width of the pen the outlines are drawn with at that setting."""
        win.settings.line_scale = value
        p = QPainter(img)
        win.sheet_view._draw_paths(p)
        width = p.pen().widthF()
        p.end()
        return width

    try:
        assert drawn(1.0) == pytest.approx(1.4)
        assert drawn(2.0) == pytest.approx(2.8)
    finally:
        win.settings.line_scale = was
        win.doc.sheet.paths.remove(path)


# ------------------------------------------------------------------- preferences

def test_the_line_width_choice_is_on_the_display_page(win):
    """The combo box shows the current factor, writes the new one back and keeps it."""
    from volura.app import PreferencesDialog
    was = win.settings.line_scale
    try:
        win.settings.line_scale = 2.0
        dlg = PreferencesDialog(win, "Display")
        assert dlg.line_width.parent() is dlg.stack.widget(1), "the choice lives on Display"
        assert [dlg.line_width.itemText(i) for i in range(dlg.line_width.count())] == \
            [name for name, _scale in CHOICES]
        assert [dlg.line_width.itemData(i) for i in range(dlg.line_width.count())] == \
            [scale for _name, scale in CHOICES]
        assert dlg.line_width.currentIndex() == 2, "the box starts at the stored factor"
        assert "thin lines" in dlg.line_width.toolTip()
        dlg.line_width.setCurrentIndex(3)
        dlg.accept()
        assert win.settings.line_scale == pytest.approx(3.0)
        assert type(win.settings)().line_scale == pytest.approx(3.0)
        again = PreferencesDialog(win, "Display")
        assert again.line_width.currentIndex() == 3
    finally:
        win.settings.line_scale = was
        win.settings.save()


def test_a_factor_that_is_not_a_choice_starts_at_normal(win):
    """A factor from another program may not leave the combo box without a choice."""
    from volura.app import PreferencesDialog
    was = win.settings.line_scale
    try:
        win.settings.line_scale = 1.7
        dlg = PreferencesDialog(win, "Display")
        assert dlg.line_width.currentIndex() == 0
    finally:
        win.settings.line_scale = was


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
