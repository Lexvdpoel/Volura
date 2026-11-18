"""3D milling of the model in the window: the command, the dialog and the machine.

The CNC tab shows the model under the cutter, so it also has the command that turns the
solids into G-code for a mill (`volura/toolpath3d.py` through `volura/mill3d_dialog.py`).
These tests run the dialog on a box: it computes in the background, draws the paths over
the model, saves a .nc file that stays inside the stock and hands the job to the CNC
Controller - and the 4th axis is only there when the remembered machine drives one.

Run with:  python -m pytest tests/test_mill3d_dialog.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, mill3d_dialog, ops, workspaces  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


class FakeBox:
    """The message boxes of the dialog and of the window, collected instead of shown."""

    def __init__(self):
        self.asked = []

    def critical(self, *args, **kwargs):
        self.asked.append(("critical",) + args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(("information",) + args[1:])

    def warning(self, *args, **kwargs):
        self.asked.append(("warning",) + args[1:])


class FakeSave:
    """QFileDialog.getSaveFileName, answering with the path a test chose."""

    def __init__(self, path):
        self.path = str(path)
        self.asked = []

    def getSaveFileName(self, *args, **kwargs):
        self.asked.append(args[1:3])
        return self.path, "G-code for a mill (*.nc *.gcode *.gc)"


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    monkeypatch.setattr(mill3d_dialog, "QMessageBox", box)
    q = qsettings()
    for key in [k for k in q.allKeys() if k.startswith(mill3d_dialog.SETTING)]:
        q.remove(key)
    q.sync()
    w = app_module.MainWindow()
    w.box = box
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    if getattr(w, "_mill3d_dialog", None) is not None:
        w._mill3d_dialog.done(0)
    w.close()


def cnc_3d(w):
    """The window in the CNC tab, with the model on the screen."""
    w.set_workspace(workspaces.CNC)
    w.set_mode("3d")
    return w


def box_model(w, size=(10.0, 10.0, 5.0)):
    ops.group_from_manifold(w.doc, mf.Manifold.cube(size), "#88aacc", "Block")
    return w


def machine(w, key):
    """The mill the CNC Controller remembers."""
    panel = w.machine_panel(workspaces.CNC)
    panel.machine_box.setCurrentIndex(max(0, panel.machine_box.findData(key)))
    return panel


def computed_dialog(w):
    """The dialog of the window, with the toolpaths of the model computed.

    The stock stands wider than the block: a 6 mm end mill cannot go down next to a wall
    that is only 2 mm away, so this is where roughing has room to show what it does.
    """
    dlg = w.mill3d_job()
    assert dlg is not None
    dlg.show()
    dlg.res.setCurrentIndex(max(0, dlg.res.findData(0.5)))
    dlg.margin.setValue(12.0)
    job = dlg.compute()
    assert job.wait(), job.error
    assert job.error is None, job.error
    return dlg


# ------------------------------------------------------------------ the command

def test_the_command_is_only_there_in_the_cnc_tab_with_the_model(win):
    assert not win.a_mill3d.isVisible()
    cnc_3d(win)
    assert win.a_mill3d.isVisible()
    win.set_mode("2d")
    assert not win.a_mill3d.isVisible()
    win.set_mode("3d")
    win.set_workspace(workspaces.DESIGN)
    assert not win.a_mill3d.isVisible()


def test_the_command_is_also_under_file_export(win):
    def clean(text):
        return text.replace("&", "").replace("…", "").strip()

    file_menu = next(a.menu() for a in win.menuBar().actions()
                     if a.menu() is not None and clean(a.text()) == "File")
    export = next(a.menu() for a in file_menu.actions()
                  if a.menu() is not None and clean(a.text()) == "Export")
    labels = [clean(a.text()) for a in export.actions() if not a.isSeparator()]
    assert "G-code for a Mill from the Model (.nc)" in labels
    assert win.a_exp_mill3d.isVisible()


def test_an_empty_model_says_so_and_keeps_the_dialog_away(win):
    cnc_3d(win)
    assert win.mill3d_job() is None
    assert "no solid" in win.hint.text()
    assert win._mill3d_dialog is None


def test_only_the_selected_parts_go_under_the_cutter(win):
    box_model(win)
    other = ops.group_from_manifold(win.doc, mf.Manifold.cube((4.0, 4.0, 4.0)),
                                    "#cc8888", "Other")
    win.doc.selection = [("g", other.id, other.id)]
    dlg = computed_dialog(win)
    assert (dlg.job["stock"].x1 - dlg.job["stock"].x0) == pytest.approx(4.0 + 2 * 12.0)


# ------------------------------------------------------------------ the dialog

def test_the_dialog_computes_and_draws_the_paths_over_the_model(win):
    cnc_3d(win)
    box_model(win)
    dlg = computed_dialog(win)
    assert dlg.job["paths"] and len(dlg.job["kinds"]) == len(dlg.job["paths"])
    assert {"rough", "finish"} <= set(dlg.job["kinds"])
    assert len(win.view.overlay_lines) == len(dlg.job["paths"])
    assert "lines" in dlg.info.text() and ":" in dlg.info.text()
    assert dlg.b_save.isEnabled() and dlg.b_send.isEnabled()
    dlg.close()
    assert win.view.overlay_lines == []


def test_the_dialog_saves_a_nc_file_inside_the_stock(win, tmp_path, monkeypatch):
    cnc_3d(win)
    box_model(win)
    dlg = computed_dialog(win)
    path = tmp_path / "block.nc"
    monkeypatch.setattr(mill3d_dialog, "QFileDialog", FakeSave(path))
    dlg.save()
    text = path.read_text(encoding="utf-8")
    assert "G1 " in text and text.startswith("; Volura")
    assert "G21" in text and "M3 S10000" in text and "M5" in text
    stock = dlg.job["stock"]
    for line in text.splitlines():
        if not line.startswith("G1 X"):
            continue
        words = {w[0]: float(w[1:]) for w in line.split() if w[0] in "XYZ"}
        assert stock.x0 - 1e-6 <= words["X"] <= stock.x1 + 1e-6
        assert stock.y0 - 1e-6 <= words["Y"] <= stock.y1 + 1e-6
        assert words["Z"] <= 1e-9                       # nothing cuts above the stock
    assert str(path) in win.hint.text()


def test_the_dialog_remembers_its_numbers(win):
    cnc_3d(win)
    box_model(win)
    dlg = win.mill3d_job()
    dlg.show()
    dlg.step_down.setValue(1.5)
    dlg.b_finish.setChecked(False)
    dlg.done(0)
    again = win.mill3d_job()
    assert again.step_down.value() == 1.5
    assert not again.b_finish.isChecked()


def test_the_4th_axis_is_only_there_for_a_machine_that_drives_one(win):
    cnc_3d(win)
    box_model(win)
    machine(win, "grbl_mill")
    dlg = win.mill3d_job()
    dlg.show()
    assert not dlg.sides.model().item(1).isEnabled()
    assert not dlg.four_axis_possible()
    assert "axes" in dlg.fourth_hint.text()
    assert dlg.options()["fourth"] is False
    machine(win, "longmill_vortex")
    dlg._sides_changed()
    assert dlg.four_axis_possible()
    assert dlg.sides.model().item(1).isEnabled()
    dlg.sides.setCurrentIndex(1)
    assert dlg.options()["fourth"] is True


def test_the_fourth_axis_writes_the_angle_of_the_second_side(win):
    cnc_3d(win)
    box_model(win)
    machine(win, "longmill_vortex")
    dlg = win.mill3d_job()
    dlg.show()
    dlg.res.setCurrentIndex(max(0, dlg.res.findData(0.5)))
    dlg.margin.setValue(12.0)
    dlg.sides.setCurrentIndex(1)
    job = dlg.compute()
    assert job.wait() and job.error is None, job.error
    assert "G0 A180.000" in dlg.job["gcode"]
    assert set(dlg.job["angles"]) == {0.0, 180.0}


# ------------------------------------------------------------------ the machine

def test_sending_loads_the_job_in_the_cnc_controller(win):
    cnc_3d(win)
    box_model(win)
    dlg = computed_dialog(win)
    panel = win.machine_panel(workspaces.CNC)
    assert panel.job == []
    dlg.send()
    assert panel.job and panel.job[0] == "G21"
    assert panel.job_name == "the 3D model"
    assert any(line.startswith("G1 ") for line in panel.job)
    assert "CNC Controller" in win.hint.text()


def test_the_controller_is_the_one_of_the_cnc_tab(win):
    cnc_3d(win)
    assert win.machine_panel(workspaces.CNC).is_cnc
    assert win.machine_panel(workspaces.LASER) is not win.machine_panel(workspaces.CNC)
