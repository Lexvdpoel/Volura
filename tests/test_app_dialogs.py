"""The general dialogs of the window live in `volura/app_dialogs.py`.

Step 4 of splitting `volura/app.py`: the classes moved verbatim and the app imports them
back, so `volura.app.ModelInfoDialog` and the rest keep working for the window and for the
tests, which also patch those names on the app. The new module may not reach back into the
app; that check runs in a fresh process, because the rest of the suite has imported both
modules long before.

Run with:  python -m pytest tests/test_app_dialogs.py
"""
import ast
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, app_dialogs  # noqa: E402
from volura import ops  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIALOGS = ["SimplifyDialog", "PageDialog", "ModelInfoDialog", "PreferencesDialog",
           "ShortcutsPanel"]

CODE = ("import sys, volura.app_dialogs; print(sorted(m for m in sys.modules"
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


def test_app_dialogs_does_not_import_the_app():
    """No circular import: the app takes the dialogs, never the other way around."""
    loaded = _loaded()
    assert "volura.app_dialogs" in loaded
    assert "volura.app" not in loaded, loaded


@pytest.mark.parametrize("name", DIALOGS)
def test_the_app_still_offers_the_dialog(name):
    assert getattr(app_module, name) is getattr(app_dialogs, name)
    assert getattr(app_module, name).__module__ == "volura.app_dialogs"


def test_the_app_still_offers_the_printers_and_the_filaments():
    assert app_module.PRINTERS is app_dialogs.PRINTERS
    assert app_module.FILAMENT_MATERIALS is app_dialogs.FILAMENT_MATERIALS
    assert app_dialogs.PRINTERS[0][1] == (256, 256, 256)
    assert app_dialogs.FILAMENT_MATERIALS[0] == ("PLA", 1.24)


def test_the_welcome_dialog_stayed_with_the_window():
    """It reads welcome_enabled(), which the tests patch on the app."""
    assert app_module.WelcomeDialog.__module__ == "volura.app"


def test_model_info_still_reads_the_settings_of_the_window(win):
    dlg = app_module.ModelInfoDialog(win, "Print Bed")
    assert dlg.page_titles == ["Units", "Statistics", "Print Bed"]
    assert dlg.printer.itemData(1) == app_module.PRINTERS[0][1]
    assert [sp.value() for sp in dlg.bed_spins] == list(win.settings.bed_size)
    assert dlg.filament.currentData()[0] == win.settings.filament_material


def test_preferences_still_holds_the_shortcut_panel(win):
    dlg = app_module.PreferencesDialog(win)
    assert dlg.page_titles == ["General", "Display", "Drawing", "3D Mouse", "Shortcuts"]
    assert isinstance(dlg.shortcuts, app_module.ShortcutsPanel)
    assert dlg.shortcuts.select_command("File/New")


def test_the_simplify_dialog_still_lists_the_steps(win):
    dlg = app_module.SimplifyDialog(win, [])
    assert dlg.slider.maximum() == len(ops.SIMPLIFY_STEPS) - 1
    assert dlg.tolerance == ops.SIMPLIFY_DEFAULT_TOLERANCE
    dlg.deleteLater()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
