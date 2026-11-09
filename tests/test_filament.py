"""Filament weight, length and cost of a printable part.

volura/ops.filament_use() turns the solid volume of a part into grams, metres of filament
and a cost, and volura/ops.print_volume() estimates what a slicer lays down for it with
walls and sparse infill; the Printer Setup page stores the material, diameter, price, walls
and infill, and Entity Info shows the result under the volume of a closed group. These tests
(Qt offscreen) check the pure computation, the settings on the Print Bed page and the rows
in Entity Info.

Run with:  python -m pytest tests/test_filament.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import ops  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

FILAMENT_KEYS = ("filament_material", "filament_density", "filament_diameter", "filament_price",
                 "print_shell", "print_infill")


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test; keeps the filament and
    print settings of the user intact, whatever a test writes to QSettings."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    q = qsettings()
    saved = {k: q.value(k) for k in FILAMENT_KEYS}
    w = app_module.MainWindow()
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


def material_row(dlg, name):
    for i in range(dlg.filament.count()):
        if dlg.filament.itemData(i)[0] == name:
            return i
    raise AssertionError(f"no material {name} on the Print Bed page")


# ------------------------------------------------------------------ the computation

def test_a_cube_of_pla_is_its_volume_times_the_density():
    grams, metres, cost = ops.filament_use(1000.0, 1.24, 1.75, 20.0)
    assert grams == pytest.approx(1.24)
    # 1000 mm³ of 1.75 mm filament: 1000 / (pi * 0.875²) mm long
    assert metres == pytest.approx(1000.0 / (math.pi * 0.875 ** 2) / 1000.0)
    assert metres == pytest.approx(0.4158, abs=1e-3)
    assert cost == pytest.approx(1.24 / 1000.0 * 20.0)


def test_weight_and_length_scale_with_the_part_and_the_price():
    base = ops.filament_use(1000.0, 1.24, 1.75, 20.0)
    twice = ops.filament_use(2000.0, 1.24, 1.75, 20.0)
    assert twice[0] == pytest.approx(2 * base[0])
    assert twice[1] == pytest.approx(2 * base[1])
    assert twice[2] == pytest.approx(2 * base[2])
    # a thicker filament of the same volume is shorter, a dearer spool costs more
    thick = ops.filament_use(1000.0, 1.24, 2.85, 20.0)
    assert thick[0] == pytest.approx(base[0]) and thick[1] < base[1]
    dear = ops.filament_use(1000.0, 1.24, 1.75, 40.0)
    assert dear[2] == pytest.approx(2 * base[2])


def test_a_slicer_lays_walls_and_sparse_infill_inside():
    # a cube of 20 mm: a skin of 1.2 mm over 2400 mm² of surface, 15 % of what is left inside
    assert ops.print_volume(8000.0, 2400.0, 1.2, 0.15) == pytest.approx(
        2400.0 * 1.2 + 0.15 * (8000.0 - 2400.0 * 1.2))


def test_full_infill_and_a_thin_part_use_the_whole_volume():
    assert ops.print_volume(8000.0, 2400.0, 1.2, 1.0) == pytest.approx(8000.0)
    # a plate of 100 × 100 × 1 mm is thinner than twice the wall, so it prints solid
    assert ops.print_volume(10000.0, 20400.0, 1.2, 0.15) == pytest.approx(10000.0)
    # no infill leaves the walls alone; nonsense settings are clamped to what they mean
    assert ops.print_volume(8000.0, 2400.0, 1.2, 0.0) == pytest.approx(2400.0 * 1.2)
    assert ops.print_volume(8000.0, 2400.0, -1.0, 0.15) == pytest.approx(0.15 * 8000.0)
    assert ops.print_volume(8000.0, 2400.0, 1.2, 2.0) == pytest.approx(8000.0)


# ------------------------------------------------------------------- the settings page

def test_print_bed_page_shows_the_materials_and_their_density(win):
    from volura.app import ModelInfoDialog
    dlg = ModelInfoDialog(win, "Print Bed")
    names = [dlg.filament.itemData(i)[0] for i in range(dlg.filament.count())]
    assert names == ["PLA", "PETG", "ABS", "ASA", "TPU", "Nylon", "Custom"]
    assert dlg.filament.itemData(material_row(dlg, "PLA"))[1] == 1.24
    assert dlg.filament.itemData(material_row(dlg, "PETG"))[1] == 1.27


def test_choosing_a_material_fills_its_density_and_is_kept(win):
    from volura.app import ModelInfoDialog
    was = (win.settings.filament_material, win.settings.filament_density,
           win.settings.filament_diameter, win.settings.filament_price)
    dlg = ModelInfoDialog(win, "Print Bed")
    dlg.filament.setCurrentIndex(material_row(dlg, "PETG"))
    assert dlg.filament_density.value() == 1.27 and not dlg.filament_density.isEnabled()
    dlg.filament_diameter.setValue(2.85)
    dlg.filament_price.setValue(30.0)
    dlg.accept()
    try:
        assert win.settings.filament_material == "PETG"
        assert win.settings.filament_density == pytest.approx(1.27)
        assert win.settings.filament_diameter == pytest.approx(2.85)
        assert win.settings.filament_price == pytest.approx(30.0)
        again = type(win.settings)()
        assert again.filament_material == "PETG" and again.filament_diameter == pytest.approx(2.85)
        reopened = ModelInfoDialog(win, "Print Bed")
        assert reopened.filament.currentData()[0] == "PETG"
        assert reopened.filament_diameter.value() == pytest.approx(2.85)
    finally:
        (win.settings.filament_material, win.settings.filament_density,
         win.settings.filament_diameter, win.settings.filament_price) = was
        win.settings.save()


def test_a_custom_material_keeps_the_density_you_type(win):
    from volura.app import ModelInfoDialog
    was = (win.settings.filament_material, win.settings.filament_density)
    dlg = ModelInfoDialog(win, "Print Bed")
    dlg.filament.setCurrentIndex(material_row(dlg, "Custom"))
    assert dlg.filament_density.isEnabled()
    dlg.filament_density.setValue(0.95)
    dlg.accept()
    try:
        assert win.settings.filament_material == "Custom"
        assert win.settings.filament_density == pytest.approx(0.95)
        assert type(win.settings)().filament_density == pytest.approx(0.95)
    finally:
        win.settings.filament_material, win.settings.filament_density = was
        win.settings.save()


def test_the_walls_and_the_infill_start_at_a_slicer_default(win):
    from volura.app import ModelInfoDialog
    dlg = ModelInfoDialog(win, "Print Bed")
    assert dlg.print_shell.value() == pytest.approx(1.2)
    assert dlg.print_infill.value() == 15


def test_the_walls_and_the_infill_are_kept(win):
    from volura.app import ModelInfoDialog
    was = (win.settings.print_shell, win.settings.print_infill)
    dlg = ModelInfoDialog(win, "Print Bed")
    dlg.print_shell.setValue(2.4)
    dlg.print_infill.setValue(40)
    dlg.accept()
    try:
        assert win.settings.print_shell == pytest.approx(2.4)
        assert win.settings.print_infill == 40
        again = type(win.settings)()
        assert again.print_shell == pytest.approx(2.4) and again.print_infill == 40
        reopened = ModelInfoDialog(win, "Print Bed")
        assert reopened.print_shell.value() == pytest.approx(2.4)
        assert reopened.print_infill.value() == 40
    finally:
        win.settings.print_shell, win.settings.print_infill = was
        win.settings.save()


# ---------------------------------------------------------------------- Entity Info

def entity_info_text(win):
    win.info.refresh()
    return " ".join(lab.text() for lab in win.info.body.findChildren(QLabel))


def test_entity_info_shows_weight_and_filament_of_a_closed_group(win):
    win.settings.filament_material = "PLA"
    win.settings.filament_density = 1.24
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 0.0
    win.settings.print_infill = 100
    g = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "cube")
    win.doc.set_selection([("g", g.id, g.id)])
    grams, metres, cost = ops.filament_use(1000.0, 1.24, 1.75, 20.0)
    text = entity_info_text(win)
    assert "Weight" in text and f"≈ {grams:.1f} g (PLA, 100 % infill)" in text
    assert f"≈ {metres:.2f} m · {cost:.2f}" in text


def test_entity_info_weight_follows_the_settings(win):
    win.settings.filament_material = "PETG"
    win.settings.filament_density = 1.27
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 0.0
    win.settings.print_infill = 100
    g = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "cube")
    win.doc.set_selection([("g", g.id, g.id)])
    grams, metres, cost = ops.filament_use(1000.0, 1.27, 1.75, 20.0)
    text = entity_info_text(win)
    assert f"≈ {grams:.1f} g (PETG, 100 % infill)" in text
    assert f"≈ {metres:.2f} m · {cost:.2f}" in text


def test_entity_info_shows_the_walls_and_infill_of_the_settings(win):
    win.settings.filament_material = "PLA"
    win.settings.filament_density = 1.24
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 1.2
    win.settings.print_infill = 15
    g = ops.group_from_manifold(win.doc, ops.primitive("box", 20, 20, 20), "#cccccc", "cube")
    win.doc.set_selection([("g", g.id, g.id)])
    grams, metres, cost = ops.filament_use(ops.print_volume(8000.0, 2400.0, 1.2, 0.15),
                                           1.24, 1.75, 20.0)
    text = entity_info_text(win)
    assert f"≈ {grams:.1f} g (PLA, 15 % infill)" in text
    assert f"≈ {metres:.2f} m · {cost:.2f}" in text
    # what a solid print of the same part would use stays under it
    solid = ops.filament_use(8000.0, 1.24, 1.75, 20.0)[0]
    assert f"solid: {solid:.1f} g" in text
    assert grams < solid


def test_an_open_group_gets_no_weight(win):
    win.settings.filament_material = "PLA"
    g = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "lid")
    g.ents.erase_faces([sorted(g.ents.F)[-1]])
    win.doc.set_selection([("g", g.id, g.id)])
    text = entity_info_text(win)
    assert "Weight" not in text and "Volume" not in text


def test_several_groups_selected_give_the_totals_of_the_job(win):
    win.settings.filament_material = "PLA"
    win.settings.filament_density = 1.24
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 0.0
    win.settings.print_infill = 100
    a = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "a")
    b = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "b")
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    grams, metres, cost = ops.filament_use(2000.0, 1.24, 1.75, 20.0)
    text = entity_info_text(win)
    assert "Total volume" in text and "2.000 cm³" in text
    assert f"≈ {grams:.1f} g (PLA, 100 % infill)" in text
    assert f"≈ {metres:.2f} m · {cost:.2f}" in text
    assert "Not counted" not in text


def test_the_totals_take_the_walls_and_infill_of_every_part(win):
    win.settings.filament_material = "PLA"
    win.settings.filament_density = 1.24
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 1.2
    win.settings.print_infill = 15
    a = ops.group_from_manifold(win.doc, ops.primitive("box", 20, 20, 20), "#cccccc", "a")
    b = ops.group_from_manifold(win.doc, ops.primitive("box", 20, 20, 20), "#cccccc", "b")
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    one = ops.print_volume(8000.0, 2400.0, 1.2, 0.15)
    grams, metres, cost = ops.filament_use(2 * one, 1.24, 1.75, 20.0)
    text = entity_info_text(win)
    assert "Total volume" in text and "16.000 cm³" in text
    assert f"≈ {grams:.1f} g (PLA, 15 % infill)" in text
    assert f"≈ {metres:.2f} m · {cost:.2f}" in text


def test_groups_that_are_not_closed_are_left_out_of_the_totals(win):
    win.settings.filament_material = "PLA"
    win.settings.filament_density = 1.24
    win.settings.filament_diameter = 1.75
    win.settings.filament_price = 20.0
    win.settings.print_shell = 0.0
    win.settings.print_infill = 100
    cube = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "cube")
    lid = ops.group_from_manifold(win.doc, ops.primitive("box", 10, 10, 10), "#cccccc", "lid")
    lid.ents.erase_faces([sorted(lid.ents.F)[-1]])
    win.doc.set_selection([("g", cube.id, cube.id), ("g", lid.id, lid.id)])
    grams = ops.filament_use(1000.0, 1.24, 1.75, 20.0)[0]
    text = entity_info_text(win)
    assert "Total volume" in text and "1.000 cm³" in text
    assert f"≈ {grams:.1f} g (PLA, 100 % infill)" in text
    assert "Not counted" in text and "1 part is not closed" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
