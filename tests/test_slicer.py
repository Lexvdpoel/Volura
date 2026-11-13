"""Slicing with CuraEngine: File > Slice….

`slicer.slice_meshes` writes the parts as STL files and the settings of a `cura_profiles.Stacks`
as CuraEngine's resolved-settings file, runs CuraEngine and reads the print time and the filament
back from the G-code. The parts are moved to the middle of the bed, because CuraEngine counts from
there. These tests check the pieces on their own - the STL, the placement, reading the header in
both G-code flavours - and, when CuraEngine and the Cura profiles are there
(tools/fetch_curaengine.py), slice a real part for a real printer and check the G-code, and drive
the dialog in the window.

Run with:  python -m pytest tests/test_slicer.py
"""
import os
import re
import struct
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import cura_profiles, slicer  # noqa: E402

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")


def _cube(size=20.0, at=(0.0, 0.0, 0.0)):
    m = mf.Manifold.cube((size, size, size)).translate(at).to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def test_the_stl_is_a_binary_stl_of_the_mesh(tmp_path):
    v, t = _cube()
    path = tmp_path / "c.stl"
    slicer.write_stl(str(path), v, t)
    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    assert count == len(t) and len(data) == 84 + 50 * len(t)
    first = struct.unpack("<12f", data[84:132])
    assert np.allclose(first[3:6], v[t[0, 0]]), "the corners of the first triangle"


def test_the_parts_go_to_the_middle_of_the_bed_and_onto_it():
    a, ta = _cube(10.0, (100.0, 50.0, 5.0))
    b, tb = _cube(10.0, (130.0, 50.0, 5.0))
    placed = slicer.place_on_bed([(a, ta, 0), (b, tb, 0)], {})
    lo = np.min([p[0].min(axis=0) for p in placed], axis=0)
    hi = np.max([p[0].max(axis=0) for p in placed], axis=0)
    assert np.allclose(lo + hi, [0.0, 0.0, hi[2] + lo[2]]) and lo[2] == pytest.approx(0.0)
    assert placed[1][0][:, 0].min() - placed[0][0][:, 0].min() == pytest.approx(30.0), \
        "the parts keep their places relative to each other"


def test_the_header_of_a_marlin_g_code():
    head = ";FLAVOR:Marlin\n;TIME:1678\n;Filament used: 0.699145m, 0.1m\n;Layer height: 0.1\n" \
           ";LAYER_COUNT:98\n"
    info = slicer.parse_header(head)
    assert info["seconds"] == 1678 and info["layers"] == 98
    assert info["filament_m"] == pytest.approx([0.699145, 0.1])


def test_the_header_of_an_ultimaker_g_code():
    head = ";START_OF_HEADER\n;FLAVOR:Griffin\n;EXTRUDER_TRAIN.0.MATERIAL.VOLUME_USED:2130\n" \
           ";PRINT.TIME:611\n;END_OF_HEADER\n"
    info = slicer.parse_header(head)
    assert info["seconds"] == 611 and info["volume_mm3"] == [2130.0]


def test_without_an_engine_slicing_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(slicer, "find_engine", lambda: None)
    with pytest.raises(RuntimeError):
        slicer.slice_meshes(None, [_cube() + (0,)], str(tmp_path / "x.gcode"))


@needs_engine
def test_a_cube_slices_for_the_ender_3(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    st = p.stacks(v, m, "standard", {"infill_sparse_density": 20.0})
    out = tmp_path / "cube.gcode"
    seen = []
    r = slicer.slice_meshes(st, [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)], str(out),
                            on_progress=seen.append)
    text = out.read_text(encoding="utf-8")
    assert r.layers == 100, "20 mm at 0.2 mm layers"
    assert 300 < r.seconds < 3600 and 1.0 < r.grams < 6.0
    assert ";FLAVOR:Marlin" in text and "M104 S200" in text, "the PLA temperature"
    assert not re.search(r"\{[a-z_]+\}", text), "a start g-code token was not filled in"
    lo = float(re.search(r"^;MINX:([\d.]+)", text, re.M).group(1))
    hi = float(re.search(r"^;MAXX:([\d.]+)", text, re.M).group(1))
    assert 0.5 * (lo + hi) == pytest.approx(117.5, abs=1.0), "in the middle of the 235 mm bed"


def _fill(text):
    """The infill moves of a G-code: every G1 under ;TYPE:FILL."""
    moves, on = [], False
    for line in text.splitlines():
        if line.startswith(";TYPE:"):
            on = line == ";TYPE:FILL"
        elif on and line.startswith("G1"):
            moves.append(line)
    return moves


@needs_engine
def test_every_infill_pattern_of_the_slice_dialog_slices_its_own_way(tmp_path):
    from volura.slice_dialog import INFILL_PATTERNS
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    seen = {}
    for _label, pattern in INFILL_PATTERNS[1:]:
        st = p.stacks(v, m, "standard", {"infill_sparse_density": 20.0,
                                         "infill_pattern": pattern})
        out = tmp_path / f"{pattern}.gcode"
        slicer.slice_meshes(st, [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)], str(out))
        seen[pattern] = "\n".join(_fill(out.read_text(encoding="utf-8")))
    assert all(seen[k] for k in seen if k != "lightning"), "a pattern without infill"
    same = [(a, b) for a in seen for b in seen if a < b and seen[a] == seen[b]]
    assert not same, f"these patterns slice the same: {same}"


@needs_engine
def test_slice_from_the_file_menu(monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    import volura.slice_dialog as sd
    import volura.window_print as wp
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    v, t = _cube(15.0)
    from volura import ops
    ops.group_from_manifold(w.doc, mf.Manifold.cube((15.0, 15.0, 15.0)), (0.8, 0.8, 0.8), "cube")
    out = tmp_path / "menu.gcode"
    seen = {}

    def fill(dlg):
        dlg.maker.setCurrentText("Creality3D")
        i = dlg.printer_box.findData("creality_ender3")
        assert i >= 0
        dlg.printer_box.setCurrentIndex(i)
        assert dlg.nozzle.currentText() == "0.4mm Nozzle"
        assert dlg.material.currentData() == "generic_pla_175"
        assert dlg.brand.currentText() == "Generic" and dlg.brand.count() > 3
        dlg.brand.setCurrentText(next(dlg.brand.itemText(i) for i in range(dlg.brand.count())
                                      if dlg.brand.itemText(i) != "Generic"))
        assert dlg.material.count() > 0 and dlg.material.currentData() != "generic_pla_175"
        dlg.brand.setCurrentText("Generic")
        i = dlg.material.findData("generic_pla_175")
        dlg.material.setCurrentIndex(i)
        assert dlg.material.currentText() == "PLA"
        assert dlg.quality.currentData() == "standard"
        assert "235" in dlg.info.text(), "the build volume of the printer"
        dlg.infill.setValue(15)
        dlg.support.setCurrentIndex(1)
        dlg.pattern.setCurrentIndex([k for _l, k in sd.INFILL_PATTERNS].index("gyroid"))
        assert dlg.user_values()["infill_pattern"] == "gyroid"
        seen["opened"] = True
        return QDialog.Accepted

    monkeypatch.setattr(sd.SliceDialog, "exec", fill)
    errors = []
    monkeypatch.setattr(wp.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(wp.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    import volura.print_preview as pp

    def look(dlg):
        """The preview: layers to look at, a nozzle that plays, then Save G-code…"""
        seen["layers"] = len(dlg.preview.layers)
        dlg.slider.setValue(5)
        dlg.play.setChecked(True)
        dlg._tick()
        assert dlg.view.at is not None and dlg.view.layer == 5
        dlg.play.setChecked(False)
        dlg.grab()                                    # it draws
        dlg._save()
        return QDialog.Accepted

    monkeypatch.setattr(pp.PrintPreviewDialog, "exec", look)
    try:
        assert w.a_slice in w.flat_actions(w.menuBar())
        w.slice_dialog()
        deadline = time.monotonic() + 120.0
        while not out.exists() and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        while "Sliced" not in w.hint.text() and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert seen.get("opened") and out.exists()
        assert "Sliced" in w.hint.text() and "g filament" in w.hint.text()
        assert seen["layers"] > 10 and str(out) in w.hint.text()
        assert w.settings.slice_printer == "creality_ender3"
        assert w.settings.slice_infill == 15 and w.settings.slice_support == 1
        assert w.settings.slice_infill_pattern == "gyroid"
        text = out.read_text(encoding="utf-8")
        assert ";FLAVOR:Marlin" in text
    finally:
        w.doc.dirty = False
        w.close()
