"""The 3D printing methods of the window live in `volura/window_print.py`.

Step 2 of splitting `volura/app.py`: the methods that build or change a printed part moved
verbatim into `PrintMixin`, a plain class without a Qt base, and
`MainWindow(QMainWindow, SheetMixin, PrintMixin)` inherits it, so `win.knob_dialog()` and
the rest keep working for the window and for the tests. The methods that ask something with
a message box or an input dialog stayed in the app, together with the actions and the parts
of a feature that report a failure, so the tests that patch those names on `volura.app`
keep seeing them. The new module may not reach back into the app; that check runs in a
fresh process, because the rest of the suite has imported both modules long before.

Run with:  python -m pytest tests/test_window_print.py
"""
import ast
import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDoubleSpinBox  # noqa: E402

from volura import window_print  # noqa: E402
from volura.window_print import PrintMixin  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the printed parts and the print tools that moved, in the order of the sections of the app
MOVED = ["make_spur_gear_model",
         "project_box_dialog", "make_project_box_model",
         "knob_dialog", "make_knob_model",
         "standoff_dialog", "make_standoff_model",
         "label_plate_dialog", "make_label_plate_model",
         "cable_clip_dialog", "make_cable_clip_model",
         "ask_sheet_text",
         "_in_selection_order", "solid_op", "outer_shell",
         "simplify_model", "simplify_groups", "_run_simplify",
         "mouse_ears_dialog",
         "split_parts", "_target_items", "_target_groups",
         "drop_to_bed", "center_on_bed", "mirror",
         "_thread_target", "thread_dialog", "_mate_group", "_apply_thread",
         "_screw_hole_target", "screw_hole_dialog", "_apply_screw_hole",
         "clearance_test_dialog", "_apply_clearance_test",
         "overhang_test_dialog", "_apply_overhang_test",
         "_cut_model", "_spin", "cut_join_dialog", "_apply_cut_join",
         "_lay_part_flat", "_separate_on_bed", "ask_text",
         "print_check", "show_help"]

# what stayed behind: the methods that ask something, or report a failure with a message box
KEPT = ["apply_round_corners", "apply_dogbone_corners",
        "_sel_groups", "_failed", "trim_solids",
        "elephant_foot_chamfer", "_run_mouse_ears",
        "orient_for_printing", "_maybe_suggest_scale_up", "_ask_scale_up",
        "_maybe_suggest_simplify", "_ask_simplify_import",
        "intersect_faces", "arrange_on_bed", "insert_primitive"]

CODE = ("import sys, volura.window_print; print(sorted(m for m in sys.modules"
        " if m.startswith('volura')))")


def _loaded():
    """The names of the volura modules a fresh process has after the import."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                         check=True, env=env, cwd=ROOT)
    return ast.literal_eval(out.stdout.strip())


def _pump_until(predicate, timeout_ms=15000):
    """Wait for a background run to finish; the window keeps painting while it works."""
    end = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def test_window_print_does_not_import_the_app():
    """No circular import: the app takes the mixin, never the other way around."""
    loaded = _loaded()
    assert "volura.window_print" in loaded
    assert "volura.app" not in loaded, loaded


def test_the_mixin_has_no_qt_base():
    """It is a plain class: the window brings Qt, the mixin only brings methods."""
    assert PrintMixin.__bases__ == (object,)


@pytest.mark.parametrize("name", MOVED)
def test_the_window_still_offers_the_method(name):
    from volura.app import MainWindow
    assert getattr(MainWindow, name) is getattr(PrintMixin, name)
    assert getattr(MainWindow, name).__module__ == "volura.window_print"


@pytest.mark.parametrize("name", KEPT)
def test_the_method_that_asks_stayed_in_the_app(name):
    from volura.app import MainWindow
    assert name not in vars(PrintMixin)
    assert getattr(MainWindow, name).__module__ == "volura.app"


def test_the_lists_of_the_parts_came_along():
    """The kinds of the dialogs are read through the window, wherever they live."""
    from volura.app import MainWindow
    for name in ("THREAD_KINDS", "SCREW_HOLE_KINDS", "CUT_JOINTS"):
        assert name in vars(PrintMixin)
        assert getattr(MainWindow, name) is getattr(PrintMixin, name)


def test_the_app_still_has_the_tools_and_the_help_text():
    """The print check and the help show these, so they moved with the code."""
    import volura.app as app_module
    assert app_module.TOOL_INFO is window_print.TOOL_INFO
    assert app_module.PANEL_PRINT_CHECK is window_print.PANEL_PRINT_CHECK
    assert any(sc for _, _, _, sc in app_module.TOOL_INFO), "the help lists the shortcuts"
    assert "{rows}" in window_print.HELP_HTML


def test_the_window_is_still_the_owner_of_everything(win):
    assert win.mode == "3d"
    assert win._target_groups() == [], "an empty model has no part to work on"
    win.drop_to_bed()
    win.center_on_bed()
    win.mirror("x")
    assert win.doc.groups == []


def test_the_window_still_builds_a_printed_wheel(win):
    win.make_spur_gear_model({"teeth": 12, "module": 2.0, "thickness": 3.0,
                              "angle": 20.0, "bore": 0.0, "backlash": 0.0})
    assert _pump_until(lambda: len(win.doc.groups) == 1), "no wheel in the model"
    assert win.doc.groups[0].name == "Spur Gear"


def test_the_spin_box_of_the_print_dialogs(win):
    sp = win._spin(0.0, 10.0, 4.0)
    assert isinstance(sp, QDoubleSpinBox)
    assert sp.value() == 4.0 and sp.suffix() == " mm"
