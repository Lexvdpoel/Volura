"""Exporting the printable parts as a Wavefront OBJ with a .mtl: one object per part,
each in its own color. Every OBJ is read back with Volura's own importer, so a round
trip keeps the names, the colors, the volumes and the coordinates.

Run with:  python -m pytest tests
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, fileio, ops  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def two_cubes():
    """Two closed cubes of 10 mm next to each other, in red and blue."""
    return [ops.Part("Left Part", "#ff0000", mf.Manifold.cube((10, 10, 10))),
            ops.Part("Right", "#0000ff", mf.Manifold.cube((10, 10, 10)).translate((20, 0, 0)))]


def read(path):
    vols, _ = fileio.import_mesh_report(str(path))
    return vols


# ------------------------------------------------------------------ the file

def test_two_cubes_come_back_as_two_colored_objects(tmp_path):
    path = tmp_path / "two.obj"
    assert fileio.export_obj(str(path), two_cubes()) == 24
    vols = read(path)
    assert [(v.name, v.color) for v in vols] == [("Left_Part", "#ff0000"), ("Right", "#0000ff")]
    assert all(abs(v.manifold.volume() - 1000) < 1e-3 for v in vols)
    assert (tmp_path / "two.mtl").exists()


def test_the_mtl_holds_one_material_per_color(tmp_path):
    same = ops.Part("Third", "#ff0000", mf.Manifold.cube((10, 10, 10)).translate((40, 0, 0)))
    path = tmp_path / "two.obj"
    fileio.export_obj(str(path), two_cubes() + [same])
    mtl = (tmp_path / "two.mtl").read_text()
    assert mtl.count("newmtl ") == 2
    assert "newmtl color_ff0000" in mtl and "newmtl color_0000ff" in mtl
    assert all(line.startswith(("newmtl ", "Kd ", "Ka 0 0 0", "d 1", "illum 1"))
               for line in mtl.splitlines())
    assert [(v.name, v.color) for v in read(path)] == [("Left_Part", "#ff0000"),
                                                       ("Right", "#0000ff"), ("Third", "#ff0000")]


def test_a_file_name_with_spaces_gets_a_mtl_name_without(tmp_path):
    """Readers that take only the first word after mtllib must still find the colors."""
    path = tmp_path / "my model.obj"
    fileio.export_obj(str(path), two_cubes())
    assert "mtllib my_model.mtl" in path.read_text().splitlines()
    assert (tmp_path / "my_model.mtl").exists()
    assert [v.color for v in read(path)] == ["#ff0000", "#0000ff"]


def test_a_part_far_from_the_origin_keeps_its_coordinates(tmp_path):
    """Rounding the corners must not move a part: 250.123456 mm stays 250.123456 mm."""
    path = tmp_path / "far.obj"
    far = mf.Manifold.cube((10, 10, 10)).translate((250.123456, 0, 0))
    fileio.export_obj(str(path), [ops.Part("Far", "#f2f2f2", far)])
    (vol,) = read(path)
    assert list(vol.manifold.bounding_box()[:3]) == pytest.approx([250.123456, 0.0, 0.0], abs=1e-5)


def test_a_part_that_does_not_print_is_left_out(tmp_path):
    path = tmp_path / "mixed.obj"
    open_part = ops.Part("Open", "#0000ff", problem="not closed: 4 open or doubled edge(s)")
    assert fileio.export_obj(str(path), [two_cubes()[0], open_part]) == 12
    assert [v.name for v in read(path)] == ["Left_Part"]


def test_nothing_to_export_is_refused(tmp_path):
    path = tmp_path / "empty.obj"
    with pytest.raises(ValueError, match="nothing to export"):
        fileio.export_obj(str(path), [ops.Part("Open", "#0000ff", problem="not closed")])
    assert not path.exists() and not (tmp_path / "empty.mtl").exists()


# ------------------------------------------------------------------ the window

class FakeBox:
    """Stands in for QMessageBox, so a test sees the messages without a dialog."""

    def __init__(self):
        self.asked = []

    def critical(self, *args, **kwargs):
        self.asked.append(("critical",) + args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(("information",) + args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    w = app_module.MainWindow()
    w.box = box
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def clean(text):
    return text.replace("&", "").replace("…", "").strip()


def export_labels(win):
    file_menu = next(a.menu() for a in win.menuBar().actions()
                     if a.menu() is not None and clean(a.text()) == "File")
    export = next(a.menu() for a in file_menu.actions()
                  if a.menu() is not None and clean(a.text()) == "Export")
    return [clean(a.text()) for a in export.actions() if not a.isSeparator()]


def test_the_export_menu_offers_the_obj_after_stl_per_color(win):
    labels = export_labels(win)
    assert labels.index("STL per Color") < labels.index("3D Model with Colors (OBJ)") \
        < labels.index("2D Vector (SVG)")


def test_the_window_writes_the_obj_in_the_background(win, tmp_path):
    ops.group_from_manifold(win.doc, mf.Manifold.cube((10, 10, 10)), "#ff0000", "Left")
    ops.group_from_manifold(win.doc, mf.Manifold.cube((10, 10, 10)).translate((20, 0, 0)),
                            "#0000ff", "Right")
    path = tmp_path / "two.obj"
    win._write_export_async("obj", str(path), ops.parts(win.doc))
    assert pump_until(lambda: "OBJ exported" in win.hint.text()), win.box.asked
    assert "2 colors" in win.hint.text() and str(path) in win.hint.text()
    assert sorted(v.color for v in read(path)) == ["#0000ff", "#ff0000"]
    assert (tmp_path / "two.mtl").exists()


def test_export_obj_asks_for_an_obj_file_name(win, tmp_path, monkeypatch):
    asked = []

    class Files:
        @staticmethod
        def getSaveFileName(parent, title, suggested, selected):
            asked.append((title, suggested, selected))
            return str(tmp_path / "model.obj"), selected

    monkeypatch.setattr(app_module, "QFileDialog", Files)
    ops.group_from_manifold(win.doc, mf.Manifold.cube((10, 10, 10)), "#ff0000", "Left")
    win.export_obj()
    assert pump_until(lambda: (tmp_path / "model.obj").exists()), win.box.asked
    assert asked[0][0] == "Export as OBJ" and asked[0][2] == "OBJ (*.obj)"
    assert asked[0][1].endswith(".obj")
