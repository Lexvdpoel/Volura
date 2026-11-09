"""The dialogs of the 2D worksheet live in `volura/sheet_dialogs.py`.

Step 2 of splitting `volura/app.py`: the classes moved verbatim and the app imports them
back, so `volura.app.BoxJointDialog` and the rest keep working for the window and for the
tests. The new module may not reach back into the app; that check runs in a fresh process,
because the rest of the suite has imported both modules long before.

Run with:  python -m pytest tests/test_sheet_dialogs.py
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

from volura import app as app_module, sheet_dialogs  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIALOGS = ["VectorPreview", "VectorExportDialog", "SheetExportDialog", "SheetSettingsDialog",
           "AutoArrangeDialog", "BoxJointDialog", "LivingHingeDialog", "VentPatternDialog",
           "FitTestDialog", "HoldingTabsDialog", "RoundCornersDialog", "DogboneCornersDialog",
           "JobEstimateDialog"]

CODE = ("import sys, volura.sheet_dialogs; print(sorted(m for m in sys.modules"
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


def test_sheet_dialogs_does_not_import_the_app():
    """No circular import: the app takes the dialogs, never the other way around."""
    loaded = _loaded()
    assert "volura.sheet_dialogs" in loaded
    assert "volura.app" not in loaded, loaded


@pytest.mark.parametrize("name", DIALOGS)
def test_the_app_still_offers_the_dialog(name):
    assert getattr(app_module, name) is getattr(sheet_dialogs, name)
    assert getattr(app_module, name).__module__ == "volura.sheet_dialogs"


def test_the_sheet_dialog_still_reads_and_writes_the_sheet(win):
    sheet = win.doc.sheet
    sheet.sheet_w, sheet.sheet_h, sheet.sheet_thickness = 420.0, 297.0, 6.0
    dlg = app_module.SheetSettingsDialog(win)
    assert dlg.preset.currentText().startswith("A3"), "the dialog starts at the sheet size"
    assert dlg.values()["thickness"] == 6.0
    dlg.w.setValue(500.0)
    assert dlg.values()["w"] == 500.0


def test_the_export_dialog_still_offers_the_machine_options(win):
    sheet = win.doc.sheet
    sheet.add_path([[0, 0], [40, 0], [40, 30], [0, 30]], closed=True, layer="Cut", src=1)
    svg = app_module.SheetExportDialog(win, "svg")
    assert svg.s_max is None and "Cut" in svg.options()["layers"]
    assert isinstance(svg.preview, app_module.VectorPreview), "the preview of the same module"
    gcode = app_module.SheetExportDialog(win, "gcode")
    assert gcode.s_max.value() > 0 and gcode.options()["dynamic"] is gcode.dynamic.isChecked()
