"""The user interface methods of the window live in `volura/window_ui.py`.

Step 3 of splitting `volura/app.py`: the shortcuts, the theme, the 3D mouse, the status bar,
the edit commands, hide and lock, the view and the right-click menu moved verbatim into
`UiMixin`, a plain class without a Qt base, and `MainWindow(QMainWindow, SheetMixin,
PrintMixin, UiMixin)` inherits it, so `win.load_shortcuts()` and the rest keep working for
the window and for the tests. The methods that ask something with a message box or an input
dialog stayed in the app, together with `closeEvent`, which the window must still find
before Qt does, so the tests that patch those names on `volura.app` keep seeing them. The
new module may not reach back into the app; that check runs in a fresh process, because the
rest of the suite has imported both modules long before.

Run with:  python -m pytest tests/test_window_ui.py
"""
import ast
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from volura import window_ui  # noqa: E402
from volura.window_ui import UiMixin  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# what moved, in the order of the sections of the app
MOVED = ["command_paths", "command_label", "path_of", "shortcut_text", "sequence_texts",
         "action_using", "read_shortcut_value", "load_shortcuts", "store_shortcuts",
         "change_shortcuts", "remove_shortcut", "restore_shortcuts", "apply_theme",
         "_system_scheme_changed", "spacemouse_status", "on_spacemouse_motion",
         "on_spacemouse_command", "set_tool", "on_tool_changed", "on_sheet_tool_changed",
         "on_vcb_changed", "on_vcb_return", "on_doc_changed", "on_selection_changed",
         "update_title", "show_entity_info", "select_color", "undo", "redo", "warn_locked",
         "unlocked", "delete", "select_all", "select_none", "erase_guides", "_hidden_items",
         "_show_items", "hide_selected", "unhide_selected", "unhide_last", "unhide_all",
         "lock_selected", "unlock_selected", "unlock_all", "copy", "cut", "paste",
         "paste_at_location", "_paste", "make_group", "explode_group", "edit_group",
         "close_group", "on_cam_history", "camera_prev", "camera_next", "camera_fov",
         "camera_zoom_window", "apply_projection", "toggle_grid", "toggle_axes",
         "toggle_edges", "toggle_xray", "toggle_overhangs", "toggle_back_edges",
         "toggle_show_hidden", "toggle_guides", "toggle_hide_rest", "set_face_style",
         "_selected_faces", "_clicked_face", "_lay_flat_face", "face_area",
         "select_face_edges", "select_connected_faces", "select_all_connected",
         "select_same_material", "align_view_to_face", "lay_flat_on_face",
         "reverse_selected_faces", "orient_selected_faces", "zoom_selection",
         "build_context_menu", "show_context_menu", "paint_selection"]

# what stayed behind: the methods that ask something, and the close of the window itself
KEPT = ["assign_shortcut", "model_info", "print_setup", "preferences", "start_spacemouse",
        "show_welcome", "show_welcome_on_startup", "_paste_svg_from_clipboard",
        "show_area_selection", "show_area_material", "closeEvent"]

CODE = ("import sys, volura.window_ui; print(sorted(m for m in sys.modules"
        " if m.startswith('volura')))")


def _loaded():
    """The names of the volura modules a fresh process has after the import."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                         check=True, env=env, cwd=ROOT)
    return ast.literal_eval(out.stdout.strip())


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def test_window_ui_does_not_import_the_app():
    """No circular import: the app takes the mixin, never the other way around."""
    loaded = _loaded()
    assert "volura.window_ui" in loaded
    assert "volura.app" not in loaded, loaded


def test_the_mixin_has_no_qt_base():
    """It is a plain class: the window brings Qt, the mixin only brings methods."""
    assert UiMixin.__bases__ == (object,)


@pytest.mark.parametrize("name", MOVED)
def test_the_window_still_offers_the_method(name):
    from volura.app import MainWindow
    assert getattr(MainWindow, name) is getattr(UiMixin, name)
    assert getattr(MainWindow, name).__module__ == "volura.window_ui"


@pytest.mark.parametrize("name", KEPT)
def test_the_method_that_asks_stayed_in_the_app(name):
    from volura.app import MainWindow
    assert name not in vars(UiMixin)
    assert getattr(MainWindow, name).__module__ == "volura.app"


def test_the_app_still_has_the_keys_of_the_shortcuts():
    """The menu paths and the panel title are read there too, so the app imports them back."""
    import volura.app as app_module
    for name in ("SHORTCUTS_KEY", "MOVED_COMMANDS", "OTHER_COMMANDS", "SOURCE_TEXT",
                 "PANEL_INFO"):
        assert getattr(app_module, name) is getattr(window_ui, name)
    for name in ("clean_text", "source_text_of", "translation_tables"):
        assert getattr(app_module, name) is getattr(window_ui, name)
    assert window_ui.MOVED_COMMANDS["File/Print Check"] == "3D Printing/Print Check"


def test_the_window_still_arranges_its_own_interface(win):
    paths = win.command_paths()
    assert "File/New" in paths and win.path_of(paths["File/New"]) == "File/New"
    assert win.sequence_texts(paths["File/New"]), "New keeps its shortcut"
    win.update_title()
    assert win.windowTitle().endswith("- Volura")
    win.select_all()
    win.select_none()
    assert win.doc.selection == []
    win.undo()
    win.redo()
    menu = win.build_context_menu()
    assert isinstance(menu, QMenu) and menu.actions()
    QCoreApplication.processEvents()
