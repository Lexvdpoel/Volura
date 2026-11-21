"""Temperature Tower: a stack of segments, one per nozzle temperature.

`temperature_tower.temperature_tower` gives the tower as one closed volume standing on `z = 0`: a
column of `segments` segments of `SEG_H` millimetres, hottest at the bottom, each with a 45°
overhang on one side, the lowest one carrying a small bridge in front and the column ending in a
thin cone, with the temperature of every segment stood on its front face. The whole thing prints on
a plate a little bigger than the column. `temperature_tower.steps` gives the plan the printer has to
follow, so a slice can change the nozzle temperature at every segment; the first temperature is the
print temperature and every other one is asked for with `M104` at its height - which is what the
generalised `slicer.with_commands` does, the same mechanism as a colour change by height.

These tests (Qt offscreen) weigh the geometry: one volume on the bed, as tall as `segments * SEG_H +
BASE`, with the numbers there; the presets of the dialog; the validation (a step of 0, an end equal
to the start, too few or too many segments, a number that is no number); the plan round-tripping
through a file; `with_commands` putting exactly the right `M104` lines at the right layers while the
filament-change wrapper keeps working; and, when CuraEngine is there, a real slice that asks for
every temperature in turn.

Run with:  python -m pytest tests/test_temperature_tower.py
"""
import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDoubleSpinBox  # noqa: E402

from volura import (app as app_module, cura_profiles, fileio, ops, slicer,  # noqa: E402
                    temperature_tower, window_print)
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

PLA = dict(start=220.0, end=190.0, step=5.0)
BASE = temperature_tower.BASE
SEG_H = temperature_tower.SEG_H
#: the tower is built from blocks and letters, so a part is at most this far off
ABS = 0.05


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    for key, value in (("temperature_tower_preset", "PLA"), ("temperature_tower_start", 220.0),
                       ("temperature_tower_end", 190.0), ("temperature_tower_step", 5.0)):
        setattr(w.settings, key, value)
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def made(**kw) -> mf.Manifold:
    return temperature_tower.temperature_tower(**dict(PLA, **kw))


def bbox(m):
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def probe(m, x, y, z, size=1.0) -> float:
    box = mf.Manifold.cube([size, size, size]).translate((x - 0.5 * size, y - 0.5 * size,
                                                          z - 0.5 * size))
    return (m ^ box).volume()


def pump_until(predicate, timeout_ms=20000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the tower

@pytest.mark.parametrize("labels", (False, True))
def test_the_tower_is_one_volume_that_stands_on_the_bed_and_is_centred(labels):
    s = temperature_tower.sizes(**PLA)
    m = made(labels=labels)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the tower fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the tower does not stand on the bed"
    n = s["segments"]
    assert hi[2] - lo[2] == pytest.approx(n * SEG_H + BASE, abs=1e-3), "not as tall as asked"


def test_the_tower_is_taller_for_every_segment_it_has():
    for step in (5.0, 10.0, 15.0):
        n = temperature_tower.sizes(220.0, 190.0, step)["segments"]
        lo, hi = bbox(temperature_tower.temperature_tower(220.0, 190.0, step))
        assert hi[2] - lo[2] == pytest.approx(n * SEG_H + BASE, abs=1e-3)


def test_a_segment_is_plastic_where_its_temperature_stands():
    """There is plastic in the middle of every band, so the numbers have something to stand on."""
    s = temperature_tower.sizes(**PLA)
    m = made()
    for i in range(s["segments"]):
        z = BASE + (i + 0.5) * SEG_H
        assert probe(m, 0.0, 0.0, z) > 0.0, f"segment {i} is not there"


def test_the_overhang_reaches_out_beyond_its_segment():
    """The 45° overhang reaches past the column on one side, above the width of a segment."""
    m = made(labels=False)
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(temperature_tower.FOOT + 2.0 * temperature_tower.EDGE,
                                          abs=ABS), "the plate is not as wide as it is"
    # near the top of a segment there is plastic out past the side of the column, on +x only
    z = BASE + 0.9 * SEG_H
    assert probe(m, 0.5 * temperature_tower.FOOT + 1.0, 0.0, z) > 0.0, "no overhang reached out"
    assert probe(m, -0.5 * temperature_tower.FOOT - 1.0, 0.0, z) == pytest.approx(0.0, abs=1e-6), \
        "something stands out on the side without the overhang"


def test_the_numbers_change_the_volume():
    """The raised temperatures are plastic on the front face, so a tower with them is bigger."""
    with_l = temperature_tower.temperature_tower(**PLA, labels=True).volume()
    without = temperature_tower.temperature_tower(**PLA, labels=False).volume()
    assert with_l > without, "the numbers are not stood on the tower"


# ------------------------------------------------------------------ the plan

def test_steps_lists_the_temperatures_hottest_first_with_their_heights():
    plan = temperature_tower.steps(**PLA)
    assert [t for _z, t in plan] == [220, 215, 210, 205, 200, 195, 190]
    assert [z for z, _t in plan] == [round(BASE + i * SEG_H, 6) for i in range(7)]


def test_a_cooling_preset_walks_up_just_the_same():
    plan = temperature_tower.steps(190.0, 220.0, 5.0)
    assert [t for _z, t in plan] == [190, 195, 200, 205, 210, 215, 220]


# ------------------------------------------------------------------ the presets

def test_every_preset_gives_a_sensible_tower():
    for name, (start, end, step) in temperature_tower.PRESETS.items():
        s = temperature_tower.sizes(start, end, step)
        assert 2 <= s["segments"] <= temperature_tower.SEGMENTS_MAX, name
        assert s["temps"][0] == int(start), name
        assert s["temps"][-1] == int(end), name
        m = temperature_tower.temperature_tower(start, end, step)
        assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, name


def test_sizes_reports_the_bands_and_the_size():
    s = temperature_tower.sizes(**PLA)
    assert s["segments"] == 7
    assert s["size"][2] == pytest.approx(7 * SEG_H + BASE)


# ------------------------------------------------------------------ the validation

@pytest.mark.parametrize("bad", (0.0, -5.0))
def test_a_step_of_zero_or_less_is_refused(bad):
    with pytest.raises(ValueError):
        temperature_tower.temperature_tower(220.0, 190.0, bad)


def test_an_end_equal_to_the_start_is_refused():
    with pytest.raises(ValueError):
        temperature_tower.temperature_tower(220.0, 220.0, 5.0)


def test_too_few_segments_is_refused():
    with pytest.raises(ValueError):
        temperature_tower.temperature_tower(220.0, 218.0, 5.0)      # one segment


def test_too_many_segments_is_refused():
    with pytest.raises(ValueError):
        temperature_tower.temperature_tower(300.0, 150.0, 5.0)      # 31 segments


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "hot"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("start", "end", "step"):
        with pytest.raises(ValueError):
            temperature_tower.temperature_tower(**dict(PLA, **{key: bad}))
        with pytest.raises(ValueError):
            temperature_tower.sizes(**dict(PLA, **{key: bad}))


# ------------------------------------------------------------------ the round-trip

def test_the_plan_round_trips_through_a_file(tmp_path):
    doc = Document()
    g = ops.group_from_manifold(doc, made(labels=False), "#f2f2f2", "Temperature tower")
    g.meta = {"temperature_steps": temperature_tower.steps(**PLA)}
    path = str(tmp_path / "tower.volura")
    fileio.save_document(doc, path)
    doc2 = Document()
    fileio.load_document(doc2, path)
    assert doc2.group(g.id).meta["temperature_steps"] == temperature_tower.steps(**PLA)


# ------------------------------------------------------------------ the commands

def _gcode(zs):
    """A small G-code with one layer per height in `zs`, each with a Z move and an extrusion."""
    return "".join(f";LAYER:{i}\nG1 Z{z:.2f} F900\nG1 X10 E{i + 1}\n" for i, z in enumerate(zs))


def test_with_commands_puts_an_M104_at_the_first_layer_at_or_above_its_height():
    text = _gcode([0.2, 0.4, 0.6, 0.8, 1.0])
    out = slicer.with_commands(text, [(0.5, "M104 S215")]).splitlines()
    i = out.index("M104 S215")
    assert out[i - 1] == ";LAYER:2", "the command is at the first layer at or above 0.5 mm"


def test_with_commands_inserts_one_line_per_height_in_order():
    text = _gcode([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    cmds = [(0.5, "M104 S215"), (0.9, "M104 S210")]
    out = slicer.with_commands(text, cmds)
    assert out.count("M104 S215") == 1 and out.count("M104 S210") == 1
    assert out.index("M104 S215") < out.index("M104 S210"), "the commands came in the wrong order"


def test_the_right_number_of_M104_lines_lands_on_the_right_layers():
    """A plan of 7 temperatures inserts 6 M104 lines, each on the first layer past its height."""
    zs = [0.2 * k for k in range(1, 401)]          # 0.2 mm layers up to 80 mm
    text = _gcode(zs)
    plan = temperature_tower.steps(**PLA)
    cmds = [(z, f"M104 S{t}") for z, t in plan[1:]]
    out = slicer.with_commands(text, cmds)
    assert out.count("M104 S") == len(cmds) == len(plan) - 1
    lines = out.splitlines()
    heights = slicer.layer_heights(out)
    for z_from, t in plan[1:]:
        line = f"M104 S{t}"
        assert line in lines, line
        k = lines.index(line)
        assert lines[k - 1].startswith(";LAYER:"), "the change is at a layer boundary"
        at = next(h for ln, h in heights if ln > k - 1)
        assert at >= z_from - 1e-6, f"{line} is before the layer that reaches {z_from} mm"
        prev = [h for ln, h in heights if ln < k - 1]
        assert not prev or prev[-1] < z_from, f"{line} came too early"


def test_two_commands_in_one_layer_are_both_written_but_not_the_same_twice():
    text = _gcode([0.2, 0.4, 0.6])
    assert slicer.with_commands(text, [(0.5, "M104 S210"), (0.5, "M140 S60")]).count("\n") > \
        text.count("\n")
    once = slicer.with_commands(text, [(0.5, "M104 S210"), (0.55, "M104 S210")])
    assert once.count("M104 S210") == 1, "the same command twice in one layer is asked for once"


def test_a_command_above_the_print_or_an_empty_plan_does_nothing():
    text = _gcode([0.2, 0.4, 0.6])
    assert slicer.with_commands(text, [(9.0, "M104 S200")]) == text
    assert slicer.with_commands(text, []) == text


def test_the_filament_change_wrapper_still_works():
    """The old function still asks for the filament on the first layer at or above its height."""
    L = (";LAYER:0\nG1 Z0.2 F900\nG1 X10 E1\n;LAYER:1\nG1 Z0.4\nG1 X10 E2\n"
         ";LAYER:2\nG1 Z0.6\nG1 X10 E3\n")
    out = slicer.with_filament_changes(L, [0.21, 0.6]).splitlines()
    assert out == [";LAYER:0", "G1 Z0.2 F900", "G1 X10 E1",
                   ";LAYER:1", "M600", "G1 Z0.4", "G1 X10 E2",
                   ";LAYER:2", "M600", "G1 Z0.6", "G1 X10 E3"]
    assert slicer.with_filament_changes(L, [7.0]) == L


def test_commands_are_written_into_the_g_code_file(tmp_path):
    path = tmp_path / "x.gcode"
    path.write_text(_gcode([0.2, 0.4, 0.6, 0.8, 1.0]), encoding="utf-8")
    slicer.insert_commands(str(path), [(0.5, "M104 S215")])
    assert path.read_text(encoding="utf-8").count("M104 S215") == 1


def test_set_print_temperature_sets_the_user_value():
    lib = cura_profiles.Library(RESOURCES) if RESOURCES else None
    if lib is None:
        pytest.skip("no Cura profiles here")
    p = lib.printer("creality_ender3")
    st = p.stacks(p.default_variant(), p.default_material(), "standard",
                  {"material_print_temperature": 200.0})
    slicer.set_print_temperature(st, 220)
    assert st.extruders[0].get("material_print_temperature") == 220
    assert st.glob.get("material_print_temperature") == 220
    assert slicer.resolved_settings(st)["extruder.0"]["material_print_temperature"] == 220


# -------------------------------------------------------------------- the real slice

@needs_engine
def test_a_temperature_tower_slices_and_changes_temperature_along_the_way(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    st = p.stacks(p.default_variant(), p.default_material(), "standard",
                  {"infill_sparse_density": 0.0})
    plan = temperature_tower.steps(**PLA)
    slicer.set_print_temperature(st, plan[0][1])
    assert slicer.resolved_settings(st)["extruder.0"]["material_print_temperature"] == 220

    mesh = temperature_tower.temperature_tower(**PLA).to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3] + np.array([40.0, 40.0, 0.0])
    tris = np.asarray(mesh.tri_verts)
    out = tmp_path / "tower.gcode"
    slicer.slice_meshes(st, [(verts, tris, 0)], str(out))
    slicer.insert_commands(str(out), [(z, f"M104 S{t}") for z, t in plan[1:]])

    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    want = [t for _z, t in plan[1:]]
    at = []
    for t in want:
        k = [i for i, ln in enumerate(lines) if ln == f"M104 S{t}"]
        assert k, f"M104 S{t} is not in the G-code"
        at.append(k[-1])
    assert at == sorted(at), "the temperatures did not come in order"
    heights = slicer.layer_heights(text)
    for (z_from, t), k in zip(plan[1:], at):
        first_after = next(h for ln, h in heights if ln >= k)
        assert first_after >= z_from - 0.21, f"M104 S{t} is before {z_from} mm"


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    for key, value in (("temperature_tower_preset", "PLA"), ("temperature_tower_start", 220.0),
                       ("temperature_tower_end", 190.0), ("temperature_tower_step", 5.0)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_fills_the_fields_from_a_preset(win, monkeypatch):
    seen = {}

    def accept(dlg):
        combo = dlg.findChild(QComboBox)
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["default"] = [combo.currentText(), spins[0].value(), spins[1].value(), spins[2].value()]
        combo.setCurrentText("PETG")
        seen["petg"] = [spins[0].value(), spins[1].value(), spins[2].value()]
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.temperature_tower_dialog()
        assert seen["default"] == ["PLA", pytest.approx(220.0), pytest.approx(190.0),
                                   pytest.approx(5.0)]
        assert seen["petg"] == [pytest.approx(250.0), pytest.approx(220.0), pytest.approx(5.0)]
        assert pump_until(lambda: len(win.doc.groups) == 1), "no temperature tower"
        g = win.doc.groups[0]
        assert g.name == "Temperature tower"
        assert g.meta["temperature_steps"] == temperature_tower.steps(250.0, 220.0, 5.0)
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert pump_until(lambda: "temperature" in win.hint.text().lower()), win.hint.text()
        st = app_module.Settings()
        assert st.temperature_tower_preset == "PETG"
        assert st.temperature_tower_start == pytest.approx(250.0)
    finally:
        forget_the_values(win.settings)


def test_the_dialog_refuses_an_end_equal_to_the_start(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.temperature_tower_dialog()
    assert win.doc.groups == [], "a tower with one segment was built anyway"
    assert any("differ" in text.lower() for _title, text in asked), asked


def test_one_undo_takes_the_whole_tower_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_temperature_tower(220.0, 190.0, 5.0, temperature_tower.steps(**PLA))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the tower is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_bed_temperature_steps_come_from_the_part(win):
    win._apply_temperature_tower(220.0, 190.0, 5.0, temperature_tower.steps(**PLA))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    result = win._bed_temperature_steps(only=False)
    assert result is not None
    print_temp, changes = result
    assert print_temp == 220
    assert changes == [(11.0, 215), (21.0, 210), (31.0, 205), (41.0, 200), (51.0, 195),
                       (61.0, 190)]


def test_the_bed_has_no_steps_without_a_tower(win):
    assert win._bed_temperature_steps(only=False) is None


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("temperature_tower_start", "temperature_tower_end", "temperature_tower_step",
            "temperature_tower_preset")
    saved = {k: q.value(k) for k in keys}
    try:
        q.setValue("temperature_tower_start", 999.0)
        q.setValue("temperature_tower_end", -999.0)
        q.setValue("temperature_tower_step", 999.0)
        q.setValue("temperature_tower_preset", "KAPUT")
        q.sync()
        s = app_module.Settings()
        assert s.temperature_tower_start == 300.0
        assert s.temperature_tower_end == 150.0
        assert s.temperature_tower_step == 15.0
        assert s.temperature_tower_preset == "PLA"
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_first_layer_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Temperature Tower") == labels.index("First Layer Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
