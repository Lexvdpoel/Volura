"""The worksheet methods of the window live in `volura/window_sheet.py`.

Step 1 of splitting `volura/app.py`: the methods of the 2D worksheet moved verbatim into
`SheetMixin`, a plain class without a Qt base, and `MainWindow(QMainWindow, SheetMixin)`
inherits it, so `win.vent_pattern_dialog()` and the rest keep working for the window and
for the tests. The methods that ask something with a message box, a file dialog or an
input dialog stayed in the app, together with the actions and the 3D side of the features,
so the tests that patch those names on `volura.app` keep seeing them. The new module may
not reach back into the app; that check runs in a fresh process, because the rest of the
suite has imported both modules long before.

Run with:  python -m pytest tests/test_window_sheet.py
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

from volura import window_sheet  # noqa: E402
from volura.window_sheet import SheetMixin  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the methods of the worksheet that moved, in the order of the sections of the app
MOVED = ["open_sheet_settings", "auto_arrange",
         "box_joint_dialog", "box_panels_from", "_box_failure", "make_box_sheet",
         "_nest_box_drawing", "make_box_model",
         "_hinge_rectangle", "living_hinge_dialog", "make_hinge_sheet",
         "vent_outline", "_vent_selection_hint", "vent_pattern_dialog", "apply_vent_pattern",
         "fit_test_dialog", "make_fit_test_sheet",
         "spur_gear_dialog", "make_spur_gear_sheet",
         "holding_tabs_dialog", "apply_holding_tabs",
         "round_corners_dialog", "dogbone_corners_dialog",
         "_laser_layers", "_add_laser_menu", "_face_plate_thickness", "_place_beside_sheet",
         "_link_ents", "_link_sheet_loops", "_shape_matches", "_link_sent_to_sheet",
         "_link_parts_to_sheet", "sync_links", "_replace_sheet_src", "update_links",
         "unlink",
         "sheet_layers_for_export", "default_sheet_export_options", "sheet_export_options",
         "sheet_export_drawing", "layer_file_name", "sheet_file_name", "_write_vector",
         "vector_export_options", "vector_export_drawing", "vector_export_empty_message",
         "export_vector", "_write_vector_async"]

# what stayed behind: the methods that ask something (the 3D side of the gear went with
# the print methods in step 2, see tests/test_window_print.py)
KEPT = ["apply_round_corners", "apply_dogbone_corners",
        "send_selection_to_sheet", "sheet_to_model", "export_sheet",
        "_export_vector_sync", "_export_vector_async", "_vector_export_failed"]

CODE = ("import sys, volura.window_sheet; print(sorted(m for m in sys.modules"
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


def test_window_sheet_does_not_import_the_app():
    """No circular import: the app takes the mixin, never the other way around."""
    loaded = _loaded()
    assert "volura.window_sheet" in loaded
    assert "volura.app" not in loaded, loaded


def test_the_mixin_has_no_qt_base():
    """It is a plain class: the window brings Qt, the mixin only brings methods."""
    assert SheetMixin.__bases__ == (object,)


@pytest.mark.parametrize("name", MOVED)
def test_the_window_still_offers_the_method(name):
    from volura.app import MainWindow
    assert getattr(MainWindow, name) is getattr(SheetMixin, name)
    assert getattr(MainWindow, name).__module__ == "volura.window_sheet"


@pytest.mark.parametrize("name", KEPT)
def test_the_method_that_asks_stayed_in_the_app(name):
    from volura.app import MainWindow
    assert name not in vars(SheetMixin)
    assert getattr(MainWindow, name).__module__ == "volura.app"


def test_the_limits_of_the_background_work_came_along():
    """The two constants are read through the window, wherever they live."""
    from volura.app import MainWindow
    assert "BACKGROUND_ROUND_POINTS" in vars(SheetMixin)
    assert "BACKGROUND_DOGBONE_POINTS" in vars(SheetMixin)
    assert MainWindow.BACKGROUND_ROUND_POINTS == SheetMixin.BACKGROUND_ROUND_POINTS
    assert MainWindow.BACKGROUND_DOGBONE_POINTS == SheetMixin.BACKGROUND_DOGBONE_POINTS


def test_the_window_is_still_the_owner_of_everything(win):
    assert win.mode == "3d"
    assert win.vent_outline() is None, "nothing is selected on the worksheet"
    assert win.sync_links("both", announce_empty=True) == 0
    assert win.unlink() == 0


def test_the_window_still_puts_a_box_on_the_worksheet(win):
    sheet = win.doc.sheet
    sheet.sheet_w, sheet.sheet_h, sheet.sheet_thickness = 420.0, 297.0, 6.0
    added = win.make_box_sheet({"w": 90.0, "d": 60.0, "h": 40.0, "t": 6.0, "labels": False})
    assert added and len(added) >= 5
    assert all(p.layer == sheet.default_layer("cut") for p in added)
    assert all(p.src for p in added), "every panel keeps its source number"


def test_the_window_still_exports_the_worksheet(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path([[0, 0], [40, 0], [40, 30], [0, 30]], closed=True, layer="Cut", src=1)
    path = str(tmp_path / "job.svg")
    assert win.export_sheet("svg", path=path) == path
    assert os.path.getsize(path) > 0
    assert "Cut" in win.default_sheet_export_options()["layers"]


def test_the_names_of_the_export_files(win):
    assert win.layer_file_name("/tmp/job.svg", "Cut/Deep: 1", "svg") == "/tmp/job_CutDeep 1.svg"
    assert win.sheet_file_name("/tmp/job.svg", 2, "svg").endswith("_sheet2.svg")
