"""Slice settings of a single part: Entity Info's Slice Settings section and what follows from it.

A group can slice with CuraEngine settings of its own: they sit in `g.meta["mesh_settings"]`, which
`ops.parts` hands to the slice as the part's `mesh_settings` and `slicer.per_mesh_settings` keeps
only the `settable_per_mesh` ones of, so they reach CuraEngine as settings of that one mesh. In the
window they are made in the collapsible Slice Settings section of Entity Info - a row per setting, a
check box that says whether the part wants that setting at all - and the Outliner shows a gear
behind the name of every part that has any. The summary of a slice says how many parts sliced with
settings of their own.

These tests (Qt offscreen) check the section (which rows, what they store, undo, what stays when the
part carries something the panel does not show, what a field starts with), the Outliner (the gear,
renaming a part that has one) and, when CuraEngine is there, a slice that ends with that sentence.

Run with:  python -m pytest tests/test_part_settings.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QLineEdit)

import volura.app as app_module  # noqa: E402
from volura import ops, slicer, tray  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: the settings the window fixture hands every test, so no test follows what another left behind
SLICE_KEYS = ("slice_infill", "slice_walls")

#: the row labels of the section, in the order it shows them
ROWS = ("Infill", "Infill pattern", "Wall lines", "Top layers", "Bottom layers", "Flow")


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test, with the two settings the
    section takes its start values from set to values of the test's own."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    q = qsettings()
    saved = {k: q.value(k) for k in SLICE_KEYS}
    w = app_module.MainWindow()
    w.settings.slice_infill = 20
    w.settings.slice_walls = 0            # 0: the last slice left the walls to the profile
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


def box(win, name="cube", size=10.0, at=(0.0, 0.0, 0.0)):
    man = mf.Manifold.cube((size, size, size)).translate(at)
    return ops.group_from_manifold(win.doc, man, "#cccccc", name)


def select(win, *groups):
    win.doc.set_selection([("g", g.id, g.id) for g in groups])
    win.info.refresh()


def section_of(win):
    """The Slice Settings section of Entity Info, or None when the panel shows none."""
    return next((s for s in win.info.body.findChildren(tray.TraySection)
                 if s.title.startswith(tr("Slice Settings"))), None)


def slice_rows(win) -> dict:
    """The rows of the section as {check box text: (check box, field)}."""
    form = section_of(win).body.layout()
    widgets = [form.itemAt(i).widget() for i in range(form.count())]
    return {head.text(): (head, field) for head, field in zip(widgets[0::2], widgets[1::2])
            if isinstance(head, QCheckBox) and field is not None}


def pump_until(predicate, timeout_ms=60000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------- the section

def test_a_group_gets_a_section_with_the_settings_a_part_can_have_on_its_own(win):
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    assert tuple(rows) == ROWS, "the section shows another set of rows"
    assert all(not check.isChecked() for check, _field in rows.values()), "a plain part has settings"
    assert all(isinstance(field, (QDoubleSpinBox, QComboBox)) for _check, field in rows.values())


def test_the_section_is_closed_until_you_open_it(win):
    g = box(win)
    select(win, g)
    head = section_of(win)
    assert head.body.isHidden(), "the section starts open"
    head.header.click()                       # what the user does with the mouse
    assert not head.body.isHidden()
    win.info.refresh()                        # and it stays that way through a refresh
    assert not section_of(win).body.isHidden()


def test_the_panel_shows_no_section_for_a_face_or_nothing(win):
    g = box(win)
    select(win, g)
    win.doc.set_selection([])
    win.info.refresh()
    assert section_of(win) is None
    win.doc.set_selection([("g", g.id, g.id)])
    win.info.refresh()
    assert section_of(win) is not None


def test_a_row_on_stores_the_value_under_its_cura_key(win):
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    rows["Top layers"][0].setChecked(True)
    assert g.meta["mesh_settings"] == {"top_layers": 4}
    rows["Infill"][0].setChecked(True)
    assert g.meta["mesh_settings"] == {"top_layers": 4, "infill_sparse_density": 20.0}
    assert win.doc.undo_stack, "a setting was stored without a checkpoint"


def test_the_start_value_is_what_the_last_slice_used(win):
    win.settings.slice_infill = 35
    win.settings.slice_walls = 3
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    assert rows["Infill"][1].value() == 35
    assert rows["Wall lines"][1].value() == 3
    assert rows["Flow"][1].value() == 100.0
    # the walls were left to the profile there: 0 is no value, so the part starts with a plain two
    win.settings.slice_walls = 0
    win.info.refresh()
    assert slice_rows(win)["Wall lines"][1].value() == 2


def test_switching_the_last_row_off_leaves_no_empty_dict(win):
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    rows["Flow"][0].setChecked(True)
    assert g.meta["mesh_settings"] == {"material_flow": 100.0}
    rows = slice_rows(win)                    # the panel rebuilt itself
    rows["Flow"][0].setChecked(False)
    assert "mesh_settings" not in g.meta, "an empty dict is not 'no settings'"


def test_a_value_you_type_replaces_the_stored_one(win):
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    rows["Infill"][0].setChecked(True)
    field = slice_rows(win)["Infill"][1]
    field.setValue(62.0)
    field.editingFinished.emit()              # as pressing Enter does
    assert g.meta["mesh_settings"]["infill_sparse_density"] == pytest.approx(62.0)


def test_the_pattern_row_stores_curas_name(win):
    g = box(win)
    select(win, g)
    rows = slice_rows(win)
    field = rows["Infill pattern"][1]
    assert "" not in [field.itemData(i) for i in range(field.count())], \
        "the part chooses a pattern of its own or leaves it off"
    rows["Infill pattern"][0].setChecked(True)
    field = slice_rows(win)["Infill pattern"][1]
    field.setCurrentIndex(field.findData("gyroid"))
    assert g.meta["mesh_settings"] == {"infill_pattern": "gyroid"}


def test_the_rows_start_checked_with_what_the_part_already_carries(win):
    g = box(win)
    g.meta = {"mesh_settings": {"material_flow": 92.5, "wall_line_count": 3}}
    select(win, g)
    rows = slice_rows(win)
    assert rows["Flow"][0].isChecked() and rows["Flow"][1].value() == pytest.approx(92.5)
    assert rows["Wall lines"][0].isChecked() and rows["Wall lines"][1].value() == 3
    assert not rows["Infill"][0].isChecked()


def test_what_the_panel_does_not_show_stays_on_the_part(win):
    g = box(win)
    g.meta = {"mesh_settings": {"support_infill_rate": 8.0},
              "temperature_steps": [(0.2, 210)]}
    select(win, g)
    rows = slice_rows(win)
    rows["Top layers"][0].setChecked(True)
    assert g.meta["mesh_settings"] == {"support_infill_rate": 8.0, "top_layers": 4}
    assert g.meta["temperature_steps"] == [(0.2, 210)], "the section touched what it was not shown"


def test_a_stored_value_that_is_no_number_does_not_break_the_panel(win):
    g = box(win)
    g.meta = {"mesh_settings": {"material_flow": None, "top_layers": "plenty"}}
    select(win, g)
    rows = slice_rows(win)
    assert rows["Flow"][0].isChecked() and rows["Flow"][1].value() == 100.0
    assert rows["Top layers"][1].value() == 4
    assert g.meta["mesh_settings"] == {"material_flow": None, "top_layers": "plenty"}


def test_a_value_outside_the_field_stays_the_parts_own(win):
    """The field cannot show 500 %, but switching another row on must not quietly rewrite it."""
    g = box(win)
    g.meta = {"mesh_settings": {"material_flow": 500.0}}
    select(win, g)
    rows = slice_rows(win)
    assert rows["Flow"][1].value() == rows["Flow"][1].maximum()
    rows["Bottom layers"][0].setChecked(True)
    assert g.meta["mesh_settings"] == {"material_flow": 500.0, "bottom_layers": 4}


def test_one_undo_takes_one_row_back(win):
    g = box(win)
    select(win, g)
    steps = len(win.doc.undo_stack)
    slice_rows(win)["Infill"][0].setChecked(True)
    assert len(win.doc.undo_stack) == steps + 1, "one row is more than one undo step"
    win.undo()
    assert "mesh_settings" not in g.meta, "undo left the setting behind"
    select(win, g)                            # undo also took the selection away
    assert not slice_rows(win)["Infill"][0].isChecked()


def test_the_part_hands_its_settings_to_the_slice(win):
    g = box(win)
    select(win, g)
    slice_rows(win)["Infill"][0].setChecked(True)
    field = slice_rows(win)["Infill"][1]
    field.setValue(0.0)
    field.editingFinished.emit()
    part = ops.parts(win.doc)[0]
    assert part.mesh_settings == {"infill_sparse_density": 0.0}


# -------------------------------------------------------------------- the Outliner

def test_the_outliner_marks_a_part_with_its_own_settings(win):
    first, second = box(win, "first"), box(win, "second")
    win.outliner.refresh()
    assert [win.outliner.list.item(i).text() for i in range(2)] == ["first", "second"]
    select(win, first)
    slice_rows(win)["Top layers"][0].setChecked(True)
    win.outliner.refresh()
    marked = win.outliner.list.item(0)
    assert marked.text() == "first ⚙", "the gear is missing"
    assert marked.toolTip() == "Slices with settings of its own"
    assert win.outliner.list.item(1).text() == "second"
    select(win, first)
    slice_rows(win)["Top layers"][0].setChecked(False)
    win.outliner.refresh()
    assert win.outliner.list.item(0).text() == "first", "the gear stayed"


def test_renaming_a_marked_part_keeps_its_name_and_its_settings(win):
    g = box(win, "first")
    select(win, g)
    slice_rows(win)["Infill"][0].setChecked(True)
    win.outliner.refresh()
    item = win.outliner.list.item(0)
    assert item.text().endswith("⚙")
    item.setText("a better name")             # what typing in the list does
    assert g.name == "a better name", "the gear ended up in the name"
    assert g.meta["mesh_settings"] == {"infill_sparse_density": 20.0}
    win.outliner.refresh()
    assert win.outliner.list.item(0).text() == "a better name ⚙"


def test_a_part_renamed_in_entity_info_keeps_its_gear(win):
    g = box(win, "first")
    select(win, g)
    slice_rows(win)["Infill"][0].setChecked(True)
    field = next(w for w in win.info.body.findChildren(QLineEdit) if w.text() == "first")
    field.setText("a better name")
    field.editingFinished.emit()              # as pressing Enter does
    assert g.name == "a better name"
    assert g.meta["mesh_settings"] == {"infill_sparse_density": 20.0}
    win.outliner.refresh()
    assert win.outliner.list.item(0).text() == "a better name ⚙", \
        "the gear belongs to the part, not to the name"


# ----------------------------------------------------------------------- the slice

@needs_engine
def test_the_summary_says_which_parts_sliced_on_their_own(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QDialog
    import volura.print_preview as pp
    import volura.slice_dialog as sd
    import volura.window_print as wp
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    box(w, "plain", 15.0)
    lean = box(w, "lean", 15.0, at=(30.0, 0.0, 0.0))
    lean.meta = {"mesh_settings": {"infill_sparse_density": 0.0}}
    errors = []
    monkeypatch.setattr(wp.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))

    def fill(dlg):
        dlg.maker.setCurrentText("Creality3D")
        i = dlg.printer_box.findData("creality_ender3")
        assert i >= 0
        dlg.printer_box.setCurrentIndex(i)
        return QDialog.Accepted

    seen = {}

    def look(dlg):
        seen["summary"] = dlg.summary.text()
        return QDialog.Accepted

    monkeypatch.setattr(sd.SliceDialog, "exec", fill)
    monkeypatch.setattr(pp.PrintPreviewDialog, "exec", look)
    try:
        w.slice_dialog()
        assert pump_until(lambda: bool(seen) or bool(errors), 180000), "nothing came of the slice"
        assert not errors, errors
        assert "1 part slices with settings of its own" in seen["summary"], seen["summary"]
        lean.meta = {}                        # a plain bed says nothing about it
        seen.clear()
        w.slice_dialog()
        assert pump_until(lambda: bool(seen) or bool(errors), 180000), "nothing came of the slice"
        assert not errors, errors
        assert "settings of its own" not in seen["summary"], seen["summary"]
    finally:
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


@needs_engine
def test_a_part_with_no_infill_of_its_own_prints_hollow_next_to_one_that_follows_the_print(tmp_path):
    """Two equal cubes on a print with 40 % infill: the one that wants 0 % has no infill in its
    middle layers, the other one has; the own value stays on its own part."""
    import numpy as np
    from volura import cura_profiles
    printer = cura_profiles.Library(RESOURCES).printer("creality_ender3")
    st = printer.stacks(printer.default_variant(), printer.default_material(), "standard",
                        {"infill_sparse_density": 40.0})
    meshes = []
    for x, own in ((20.0, {"infill_sparse_density": 0}), (70.0, {})):
        mm = mf.Manifold.cube([20.0, 20.0, 20.0]).translate([x, 40.0, 0.0]).to_mesh()
        meshes.append((np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0,
                       slicer.per_mesh_settings(st, own)))
    out = tmp_path / "two-cubes.gcode"
    slicer.slice_meshes(st, meshes, str(out))
    text = out.read_text(encoding="utf-8", errors="replace")
    middle = text.split(";LAYER:50\n")[1].split(";LAYER:51\n")[0]
    parts = {}
    for chunk in middle.split(";MESH:")[1:]:
        name, _, body = chunk.partition("\n")
        parts[name.strip()] = parts.get(name.strip(), "") + body
    assert ";TYPE:FILL" not in parts.get("part0", ""), "the cube that wants 0 % has infill"
    assert ";TYPE:FILL" in parts.get("part1", ""), "the cube that follows the print has no infill"
