"""File > Export > G-code for a GRBL Laser: the worksheet as the text of the machine.

`volura/gcode.py` writes the G-code (tests/test_gcode.py); here the worksheet of the
window goes through it. The export is the third kind next to SVG and DXF in the
"Export for Laser/Mill" dialog, with the same layer choice and compensation, and for
G-code also the highest S word and M4 or M3, which come back next time because they
belong to the machine and not to the drawing. A layer without a speed stops the export
with a message instead of writing G-code that GRBL throws away.

Run with:  python -m pytest tests/test_gcode_export.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from volura import __version__, app as app_module  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.vector2d import job_estimate  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def points(text) -> set:
    """The X/Y pairs of the moves of a G-code text, as {(x, y)}."""
    out = set()
    for line in text.splitlines():
        w = {word[0]: float(word[1:]) for word in line.split() if word[:1] in ("X", "Y")}
        if "X" in w and "Y" in w:
            out.add((w["X"], w["Y"]))
    return out


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def boxes(monkeypatch):
    """The message boxes of the export, without one on the screen."""

    class Box:
        def __init__(self):
            self.messages = []

        def information(self, parent, title, text, *a, **k):
            self.messages.append(text)
            return 0

        def critical(self, parent, title, text, *a, **k):
            self.messages.append(text)
            return 0

    box = Box()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    return box


@pytest.fixture(autouse=True)
def machine_settings():
    """Every test starts with a machine of its own: nothing remembered yet."""
    q = qsettings()
    q.remove(app_module.GCODE_S_MAX_KEY)
    q.remove(app_module.GCODE_DYNAMIC_KEY)
    q.sync()
    yield q


def export(win, path, layers=("Cut",), **options):
    """The export of the worksheet to `path`, with these layers and choices."""
    opt = {"layers": list(layers), "compensate": False, "per_layer": False}
    opt.update(options)
    return win.export_sheet("gcode", path=path, options=opt)


# ------------------------------------------------------------------ the file

def test_the_worksheet_comes_out_as_G_code(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(40, 30), closed=True, layer="Cut", src=1)
    sheet.add_path(rect(10, 10, 60), closed=True, layer="Engrave", src=2)
    path = str(tmp_path / "job.gcode")
    got = export(win, path, layers=("Cut", "Engrave"))
    assert got == path
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    assert "\r" not in text, "the machine reads plain \\n, also on Windows"
    lines = text.splitlines()
    assert lines[0] == f"; Volura {__version__}"
    assert "M4" in lines, "dynamic power by default"
    assert any(line.startswith("G1 ") for line in lines)
    assert lines[-2] == "M5"
    assert "; layer Cut, pass 1/1" in lines and "; layer Engrave, pass 1/1" in lines


def test_the_S_maximum_and_M3_come_into_the_file(win, tmp_path):
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Cut", src=1)
    path = str(tmp_path / "old-board.gcode")
    export(win, path, s_max=255, dynamic=False)
    lines = open(path, encoding="utf-8").read().splitlines()
    assert "M3" in lines and "M4" not in lines
    cut = [line for line in lines if line.startswith("G1 ")]
    assert all(line.endswith("S255") for line in cut), "100 % of an S maximum of 255"


def test_a_layer_left_out_gives_no_G_code_of_its_paths(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(40, 30), closed=True, layer="Cut", src=1)
    sheet.add_path(rect(10, 10, 60), closed=True, layer="Engrave", src=2)
    path = str(tmp_path / "cut.gcode")
    export(win, path, layers=("Cut",))
    text = open(path, encoding="utf-8").read()
    assert "; layer Cut, pass 1/1" in text and "; layer Engrave" not in text
    assert all(x < 60.0 and y < 60.0 for x, y in points(text)), "nothing of the engraving"


def test_the_status_bar_adds_the_time_of_the_job_estimate(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(100, 50), closed=True, layer="Cut", src=1)
    path = str(tmp_path / "job.gcode")
    export(win, path)
    want = app_module.fmt_time(job_estimate(sheet)["total"])
    assert want == "0:15"
    assert "estimated time 0:15" in win.hint.text()


def test_the_save_dialog_offers_gcode(win, monkeypatch, tmp_path):
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Cut", src=1)
    target = str(tmp_path / "job")
    asked = []

    def ask(parent, title, name, flt, *a, **k):
        asked.append((name, flt))
        return target, flt

    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", ask)
    got = win.export_sheet("gcode", options={"layers": ["Cut"], "compensate": False,
                                             "per_layer": False})
    assert got == target + ".gcode" and os.path.exists(got)
    assert asked[0][1] == "G-code (*.gcode *.nc *.gc)"
    assert asked[0][0].endswith(".gcode")


# ------------------------------------------------------------------ what stops it

def test_a_layer_without_speed_is_a_message_and_no_file(win, boxes, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(40, 30), closed=True, layer="Cut", src=1)
    sheet.layers["Cut"].params["speed"] = 0.0
    path = str(tmp_path / "job.gcode")
    assert export(win, path) is None
    assert not os.path.exists(path), "G-code with F0 is no result"
    assert boxes.messages and "layer Cut has no speed" in boxes.messages[0]


def test_a_layer_without_passes_is_a_message_and_no_file(win, boxes, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(40, 30), closed=True, layer="Cut", src=1)
    sheet.layers["Cut"].params["passes"] = 0
    path = str(tmp_path / "job.gcode")
    assert export(win, path) is None
    assert not os.path.exists(path)
    assert boxes.messages and "layer Cut has no passes" in boxes.messages[0]


# ------------------------------------------------------------------ the dialog

def test_only_the_G_code_dialog_asks_for_the_power(win):
    dialogs = [app_module.SheetExportDialog(win, kind) for kind in ("svg", "dxf", "gcode")]
    try:
        assert dialogs[0].s_max is None and dialogs[1].dynamic is None
        assert dialogs[2].s_max is not None and dialogs[2].dynamic is not None
        assert (dialogs[2].s_max.minimum(), dialogs[2].s_max.maximum()) == (1, 100000)
        assert dialogs[2].s_max.value() == 1000, "the $30 of a modern board"
        assert dialogs[2].dynamic.isChecked(), "M4 by default"
        assert "s_max" not in dialogs[0].options()
        assert dialogs[2].options()["s_max"] == 1000
        assert dialogs[2].options()["dynamic"] is True
    finally:
        for dlg in dialogs:
            dlg.deleteLater()


def test_the_power_of_the_machine_comes_back(win, monkeypatch):
    def ok(self, *args):
        self.s_max.setValue(255)
        self.dynamic.setChecked(False)
        self.accept()                       # the OK button, without the screen
        return QDialog.Accepted

    monkeypatch.setattr(app_module.SheetExportDialog, "exec", ok)
    opt = win.sheet_export_options("gcode")
    assert opt["s_max"] == 255 and opt["dynamic"] is False
    q = qsettings()
    assert int(q.value(app_module.GCODE_S_MAX_KEY)) == 255
    assert q.value(app_module.GCODE_DYNAMIC_KEY) in (False, "false", "0")
    again = app_module.SheetExportDialog(win, "gcode")
    try:
        assert again.s_max.value() == 255 and again.dynamic.isChecked() is False
    finally:
        again.deleteLater()


def test_an_svg_export_leaves_the_machine_settings_alone(win, monkeypatch):
    def ok(self, *args):
        self.accept()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.SheetExportDialog, "exec", ok)
    assert win.sheet_export_options("svg")["compensate"] is False
    q = qsettings()
    assert q.value(app_module.GCODE_S_MAX_KEY) is None
    assert q.value(app_module.GCODE_DYNAMIC_KEY) is None


# ------------------------------------------------------------------ the menu

def export_menu(win):
    """The Export submenu of the File menu."""
    file_menu = next(a.menu() for a in win.menuBar().actions()
                     if a.text().replace("&", "") == "File")
    return next(a.menu() for a in file_menu.actions()
                if a.menu() is not None and a.text().replace("&", "") == "Export")


def test_the_G_code_entry_stands_next_to_the_DXF_one(win):
    labels = [a.text() for a in export_menu(win).actions() if not a.isSeparator()]
    i = labels.index("2D Vector (&DXF)…")
    assert labels[i - 1] == "2D Vector (&SVG)…"
    assert labels[i + 1] == "G-code for a GRBL Laser (.gcode)…"


def test_the_menu_entry_exports_the_worksheet(win, monkeypatch):
    asked = []
    monkeypatch.setattr(win, "export_sheet",
                        lambda kind="svg", path=None, options=None: asked.append(kind))
    win.a_exp_gcode.trigger()
    assert asked == ["gcode"]
