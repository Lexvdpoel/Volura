"""The choice of keyboard and mouse scheme: SketchUp (what Volura does now) or Autodesk Fusion.

`keyschemes.scheme_defaults` makes the default shortcut of every command for a scheme out
of the shortcuts the actions are built with: the Fusion table replaces the keys of its
commands and takes those keys away from the commands that had them, so no key ends up on
two commands. The window puts such a list on its actions, keeps what the user stored in
Preferences > Shortcuts, and lets the middle mouse button of the 3D view pan when the
scheme does that. The 2D worksheet needs nothing of this: its keys live on the worksheet
itself (`sheetview.SHEET_KEYS`) and are the letters Fusion uses there too.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QKeySequence, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QTextBrowser  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import keyschemes, window_print  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

FUSION, SKETCHUP = keyschemes.FUSION_SCHEME, keyschemes.SKETCHUP


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.set_key_scheme(SKETCHUP)
    w.restore_shortcuts()
    w.doc.dirty = False
    w.close()


def defaults(w):
    """{command path: [key texts]} of what the commands of the window answer to now."""
    return {path: w.sequence_texts(a) for path, a in w.command_paths().items()}


def fusion_defaults(w):
    return keyschemes.scheme_defaults(FUSION, defaults(w))


def middle_drag(w, modifiers=Qt.NoModifier):
    """A middle-drag to the right through the handlers of the 3D view.

    Returns how far the camera turned and how far it moved: an orbit turns, a pan moves.
    """
    v = w.view
    v.cam.yaw, v.cam.pitch, v.cam.target = 0.0, 0.0, np.zeros(3)
    p, q = QPointF(400.0, 300.0), QPointF(460.0, 300.0)
    v.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.MiddleButton,
                                  Qt.MiddleButton, modifiers))
    drag = v._drag
    v.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, q, q, Qt.NoButton, Qt.MiddleButton,
                                 modifiers))
    turned, moved = abs(v.cam.yaw), float(np.linalg.norm(v.cam.target))
    v.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, q, q, Qt.MiddleButton,
                                    Qt.NoButton, modifiers))
    return drag, turned, moved


# ------------------------------------------------------------------ the scheme

def test_the_fusion_scheme_has_the_fusion_keys(win):
    f = fusion_defaults(win)
    assert f["Tools/Push/Pull"] == ["E", "Q"]
    assert f["Tools/Offset"] == ["O"]
    assert f["Tools/Fillet"] == ["F"]
    assert f["Tools/Tape Measure"] == ["I"]
    assert f["Tools/Eraser"] == [], "no key calls the eraser, Del still deletes"
    for path in ("Camera/Orbit", "Camera/Pan", "Camera/Zoom", "Tools/Rotate", "Tools/Scale"):
        assert f[path] == [], f"{path} is a mouse movement in Fusion"
    assert "F6" in f["Camera/Zoom Extents"], "Fusion zooms to extents with F6"
    assert "Shift+Z" in f["Camera/Zoom Extents"], "and keeps the key Volura had"


def test_every_command_of_the_fusion_table_is_a_real_command(win):
    paths = win.command_paths()
    assert not set(keyschemes.FUSION) - set(paths), "a command of the table is not in the menus"


def test_the_fusion_scheme_leaves_the_rest_alone(win):
    f = fusion_defaults(win)
    for path in ("Draw/Lines/Line", "Draw/Shapes/Rectangle", "Draw/Shapes/Circle"):
        assert f[path] == defaults(win)[path], "L, R and C are the same in both schemes"
    assert f["Tools/Select"] == ["Space"] and f["Tools/Move"] == ["M"]
    assert f["Tools/Paint Bucket"] == ["B"] and f["Edit/Make Component"] == ["G"]
    assert f["Tools/3D Text"] == ["Shift+T"] and f["View/Edge Style/Back Edges"] == ["K"]
    assert f["Help/Keyboard Shortcuts"] == ["F1"] and f["3D Printing/Print Check"] == ["F5"]
    assert f["Edit/Copy"] == ["Ctrl+C"] and f["View/2D Worksheet"] == ["Ctrl+Shift+2"]


def test_a_key_of_the_scheme_is_on_one_command_of_a_mode(win):
    """The 3D commands and the worksheet commands may share a letter (L is Line in both),
    but inside the mode you are in, nothing may be doubled."""
    f = fusion_defaults(win)
    for mode in ("3d", "2d"):
        win.set_mode(mode)
        seen = {}
        for path, a in win.command_paths().items():
            if not a.isVisible():
                continue
            for text in f.get(path, []):
                assert text not in seen, f"{text} is on '{seen[text]}' and '{path}' in {mode}"
                seen[text] = path
        assert seen, f"the {mode} mode has no shortcuts at all"
    win.set_mode("3d")


@pytest.mark.parametrize("choice", (SKETCHUP, "", "autocad", None))
def test_the_sketchup_scheme_is_what_volura_ships(win, choice):
    base = defaults(win)
    assert keyschemes.scheme_defaults(choice, base) == base


def test_an_unknown_choice_is_the_sketchup_scheme():
    assert keyschemes.SCHEMES == (SKETCHUP, FUSION)
    assert keyschemes.scheme_of(FUSION) == FUSION
    for unknown in ("", "Autodesk", "sketchup ", None):
        assert keyschemes.scheme_of(unknown) == SKETCHUP


# ------------------------------------------------------------------ the window

def test_switching_the_scheme_changes_the_actions(win):
    win.set_key_scheme(FUSION)
    T = win.tool_actions
    assert win.sequence_texts(T["pushpull"]) == ["E", "Q"]
    assert win.sequence_texts(T["offset"]) == ["O"]
    assert win.sequence_texts(T["fillet"]) == ["F"]
    assert win.sequence_texts(T["tape"]) == ["I"]
    assert win.sequence_texts(T["eraser"]) == [] and win.sequence_texts(T["orbit"]) == []
    assert "F6" in win.sequence_texts(win.a_extents)
    assert win.sequence_texts(T["line"]) == ["L"], "what is the same stays on the action"


def test_switching_back_restores_the_sketchup_keys_exactly(win):
    base = defaults(win)
    win.set_key_scheme(FUSION)
    assert defaults(win) != base
    win.set_key_scheme(SKETCHUP)
    assert defaults(win) == base
    assert not win.view.middle_pans


def test_a_shortcut_of_the_user_survives_the_switch(win):
    assert win.assign_shortcut("Tools/Chamfer", QKeySequence("Ctrl+Alt+K"), ask=False)
    win.set_key_scheme(FUSION)
    assert win.sequence_texts(win.a_chamfer) == ["Ctrl+Alt+K"]
    win.set_key_scheme(SKETCHUP)
    assert win.sequence_texts(win.a_chamfer) == ["Ctrl+Alt+K"]
    assert "shortcuts/Tools/Chamfer" in qsettings().allKeys()


def test_a_key_of_the_user_wins_over_the_new_defaults(win):
    """I is the Fusion key of Tape Measure; the user gave it to Chamfer, so that is it."""
    assert win.assign_shortcut("Tools/Chamfer", QKeySequence("I"), ask=False)
    win.set_key_scheme(FUSION)
    assert win.sequence_texts(win.a_chamfer) == ["I"]
    assert win.sequence_texts(win.tool_actions["tape"]) == []


def test_reset_goes_back_to_the_default_of_the_scheme_that_is_on(win):
    win.set_key_scheme(FUSION)
    win.change_shortcuts(win.tool_actions["pushpull"], ["Ctrl+Alt+P"])
    win.restore_shortcuts()
    assert win.sequence_texts(win.tool_actions["pushpull"]) == ["E", "Q"]
    assert not [k for k in qsettings().allKeys() if k.startswith("shortcuts/")]


def test_the_scheme_is_saved_and_loaded(win):
    win.set_key_scheme(FUSION)
    assert app_module.Settings().key_scheme == FUSION
    again = app_module.MainWindow()
    assert again.sequence_texts(again.tool_actions["pushpull"]) == ["E", "Q"]
    assert again.view.middle_pans
    again.doc.dirty = False
    again.close()


def test_a_bogus_scheme_in_the_settings_is_sketchup():
    q = qsettings()
    q.setValue("key_scheme", "autocad")
    q.sync()
    assert app_module.Settings().key_scheme == SKETCHUP
    q.remove("key_scheme")
    q.sync()


def test_preferences_switch_the_scheme_at_once(win):
    dlg = app_module.PreferencesDialog(win)
    dlg.scheme.setCurrentIndex(dlg.scheme.findData(FUSION))
    assert dlg.scheme.itemText(dlg.scheme.currentIndex()) == "Fusion"
    dlg.apply()
    assert win.sequence_texts(win.tool_actions["offset"]) == ["O"]
    assert win.settings.key_scheme == FUSION
    assert app_module.Settings().key_scheme == FUSION


# ------------------------------------------------------------------ the tips

def test_the_tool_tips_name_the_key_of_the_scheme(win):
    tip = win.tool_actions["pushpull"].toolTip()
    assert tip.startswith("Push/Pull") and tip.endswith("(P)")
    win.set_key_scheme(FUSION)
    assert win.tool_actions["pushpull"].toolTip().endswith("(E, Q)")
    assert "(" not in win.tool_actions["eraser"].toolTip(), "no key, no bracket"


def test_the_help_lists_the_keys_of_the_scheme(win, monkeypatch):
    seen = {}
    monkeypatch.setattr(QTextBrowser, "setHtml", lambda self, html: seen.setdefault("html", html))
    monkeypatch.setattr(QDialog, "exec", lambda self, *a: QDialog.Accepted)

    win.show_help()
    assert "<b>P</b></td><td>Push/Pull" in seen["html"]
    assert "Push/Pull (P) works on every face" in seen["html"]
    assert "Orbit, with Shift: pan" in seen["html"]

    seen.clear()
    win.set_key_scheme(FUSION)
    win.show_help()
    html = seen["html"]
    assert "<b>E, Q</b></td><td>Push/Pull" in html
    assert "<b>P</b></td>" not in html, "the old keys are not in the list"
    assert "<b>E</b></td><td>Eraser" not in html, "the eraser has no key to show"
    assert "Push/Pull (E, Q) works on every face" in html
    assert "Pan, with Shift: orbit" in html


# ------------------------------------------------------------------ the mouse

def test_the_middle_button_orbits_in_sketchup(win):
    assert not win.view.middle_pans
    drag, turned, moved = middle_drag(win)
    assert drag == "orbit" and turned and not moved


def test_the_middle_button_pans_in_fusion(win):
    win.set_key_scheme(FUSION)
    assert win.view.middle_pans
    drag, turned, moved = middle_drag(win)
    assert drag == "pan" and moved and not turned


def test_shift_swaps_orbit_and_pan_the_other_way_round(win):
    assert middle_drag(win, Qt.ShiftModifier)[0] == "pan"
    win.set_key_scheme(FUSION)
    assert middle_drag(win, Qt.ShiftModifier)[0] == "orbit"


@pytest.mark.parametrize("scheme", ("sketchup", "fusion"))
def test_the_worksheet_tools_name_their_key(win, scheme):
    from volura.app import SHEET_TOOL_KEYS
    from volura.sheetview import SHEET_KEYS
    win.set_key_scheme(scheme)
    assert win.sheet_tool_actions["select"].toolTip().endswith("(Space)")
    assert win.sheet_tool_actions["line"].toolTip().endswith("(L)")
    for key, name in SHEET_KEYS.items():
        assert QKeySequence(key).toString() in SHEET_TOOL_KEYS[name] or name == "offset"


def test_the_worksheet_keeps_its_own_keys(win):
    """The letters of the worksheet live on the worksheet, in both schemes."""
    win.set_mode("2d")
    win.set_key_scheme(FUSION)
    for key, name in ((Qt.Key_L, "line"), (Qt.Key_R, "rectangle"), (Qt.Key_C, "circle"),
                      (Qt.Key_O, "offset")):
        win.sheet_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
        assert win.sheet_view.tool.name == name, f"{name} does not answer to its letter"
    assert all(not a.shortcuts() for a in win.sheet_tool_actions.values()), \
        "the worksheet tools have no shortcuts of their own to clash with the model"
    win.set_mode("3d")


def test_the_help_text_takes_the_keys_of_the_scheme():
    """What is left of the placeholders is what the window fills with the scheme."""
    html = window_print.tr(window_print.HELP_HTML).format(rows="<tr/>", mouse="Pan",
                                                          eraser=" (E)", pushpull=" (P)")
    assert "{" not in html and "(E)" in html
