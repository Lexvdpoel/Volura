"""View > display styles: face style, back edges, hidden geometry.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

STYLE_KEYS = ("face_style", "back_edges", "show_hidden", "hide_rest", "xray")
STYLES = {"a_wireframe": "wireframe", "a_hidden_lines": "hiddenline",
          "a_shaded": "shaded", "a_monochrome": "monochrome"}


@pytest.fixture(scope="module")
def win():
    q = QSettings("Volura", "Volura")
    saved = {k: q.value(k) for k in STYLE_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


def shortcut_of(action):
    return [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]


def test_face_style_actions(win):
    """Every View > Face Style choice sets Settings.face_style to its own value."""
    for attr, value in STYLES.items():
        action = getattr(win, attr)
        assert action.isCheckable(), f"'{action.text()}' must be checkable"
        action.trigger()
        assert win.settings.face_style == value, \
            f"{attr} must set face_style '{value}', got {win.settings.face_style!r}"


@pytest.mark.parametrize("value", sorted(STYLES.values()))
def test_face_style_exclusive(win, value):
    """The four face styles are one exclusive choice: exactly one action is checked."""
    getattr(win, [k for k, v in STYLES.items() if v == value][0]).trigger()
    checked = [v for k, v in STYLES.items() if getattr(win, k).isChecked()]
    assert checked == [value], f"exactly one face style must be checked, now {checked}"


def test_xray_combines_with_any_style(win):
    """X-ray is a separate switch: it stays out of the exclusive face style group."""
    win.a_shaded.trigger()
    win.a_xray.trigger()
    assert win.settings.xray is True
    assert win.settings.face_style == "shaded"
    assert win.a_shaded.isChecked()
    win.a_wireframe.trigger()
    assert win.settings.xray is True and win.settings.face_style == "wireframe"
    win.a_xray.trigger()
    assert win.settings.xray is False


def test_back_edges_shortcut_and_toggle(win):
    """View > Edge Style > Back Edges has shortcut K and toggles Settings.back_edges."""
    a = win.a_back_edges
    assert "K" in shortcut_of(a), f"'Back Edges' must have shortcut K, has {shortcut_of(a)}"
    assert a.isCheckable() and a.isEnabled()
    was = win.settings.back_edges
    a.trigger()
    assert win.settings.back_edges is not was
    a.trigger()
    assert win.settings.back_edges is was


def test_hidden_geometry_and_hide_rest(win):
    """View > Hidden Geometry and Component Edit > Hide Rest of Model toggle."""
    for action, attr in ((win.a_hidden_geometry, "show_hidden"), (win.a_hide_rest, "hide_rest")):
        assert action.isCheckable() and action.isEnabled(), action.text()
        was = getattr(win.settings, attr)
        action.trigger()
        assert getattr(win.settings, attr) is not was
        action.trigger()
        assert getattr(win.settings, attr) is was


def test_guides_toggle(win):
    """View > Guides is a working toggle now: it shows and hides the guides."""
    assert win.a_guides.isCheckable() and win.a_guides.isEnabled()
    was = win.settings.show_guides
    win.a_guides.trigger()
    assert win.settings.show_guides is not was
    win.a_guides.trigger()
    assert win.settings.show_guides is was


def test_settings_are_stored(win):
    """The view style choices come back from QSettings, like the other settings."""
    win.a_monochrome.trigger()
    win.a_back_edges.trigger()
    win.a_hidden_geometry.trigger()
    win.a_hide_rest.trigger()
    win.settings.save()
    again = type(win.settings)()
    assert again.face_style == "monochrome"
    assert again.back_edges is win.settings.back_edges
    assert again.show_hidden is win.settings.show_hidden
    assert again.hide_rest is win.settings.hide_rest


def test_viewport_uses_the_face_style(win):
    """The viewport draws what the style says: no faces in wireframe, no lighting in hidden line."""
    from OpenGL import GL as real_gl
    from volura import viewport as vp

    class Mock:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name.startswith("GL_"):
                return getattr(real_gl, name)

            def call(*args):
                self.calls.append((name, args))
            return call

    def drawn(**styles):
        m = Mock()
        old = {k: getattr(win.settings, k) for k in styles}
        for k, v in styles.items():
            setattr(win.settings, k, v)
        vp.gl = m
        try:
            win.view._draw_model()
        finally:
            vp.gl = real_gl
            for k, v in old.items():
                setattr(win.settings, k, v)
        tris = [a for n, a in m.calls if n == "glDrawArrays" and a[0] == real_gl.GL_TRIANGLES]
        lit = [a for n, a in m.calls if n == "glEnable" and a[0] == real_gl.GL_LIGHTING]
        return len(tris), len(lit)

    ents = win.doc.root
    if not len(ents.F):
        from volura.geom import Plane
        import numpy as np
        pl = Plane((0, 0, 0), (0, 0, 1))
        loop = pl.to_local(np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0],
                                     [40.0, 30.0, 0.0], [0.0, 30.0, 0.0]]))
        ents.add_shape(pl, [loop])
        ents.pushpull(list(ents.F)[0], 10.0)
    assert drawn(face_style="shaded")[0] > 0, "shaded draws faces"
    assert drawn(face_style="wireframe")[0] == 0, "wireframe draws no faces"
    assert drawn(face_style="hiddenline")[0] > 0, "hidden line draws faces (white)"
    assert drawn(face_style="hiddenline")[1] == 0, "hidden line has no lighting"
    assert drawn(face_style="monochrome")[0] > 0, "monochrome draws faces"
