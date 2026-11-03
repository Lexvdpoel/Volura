"""Werkblad en model werken samen: onderdelen als platen in 3D, vlakken als bewerking in 2D,
en de werkbalken overleven het heen en weer schakelen."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import sheet3d  # noqa: E402
from volura.vector2d import Drawing  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.show()
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def rect(x, y, w, h):
    return np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], float)


def plate_drawing():
    d = Drawing.new_sheet()
    d.add_path(rect(0, 0, 40, 30), closed=True, layer="Snijden")
    d.add_path(rect(10, 10, 10, 10), closed=True, layer="Snijden")       # gat
    d.add_path(rect(25, 5, 10, 20), closed=True, layer="Uitkameren")     # kamer van boven
    d.add_path(rect(2, 2, 5, 5), closed=True, layer="Graveren")          # blijft 2D
    d.add_path(rect(60, 0, 20, 20), closed=True, layer="Snijden")        # tweede deel
    d.add_path(rect(100, 0, 20, 20), closed=True, layer="Hulplijnen")    # geen onderdeel
    d.layers["Uitkameren"].params["depth"] = 1.0
    return d


def test_onderdelen_als_platen():
    solids = sheet3d.part_solids(plate_drawing(), 3.0)
    assert len(solids) == 2
    (_n1, a), (_n2, b) = solids
    # 40*30 - gat 10*10 = 1100 mm2 maal 3, min de kamer 10*20*1
    assert a.volume() == pytest.approx(1100 * 3 - 200 * 1, rel=1e-6)
    assert b.volume() == pytest.approx(400 * 3, rel=1e-6)
    lo, hi = np.asarray(a.bounding_box()).reshape(2, 3)
    assert lo[2] == pytest.approx(0) and hi[2] == pytest.approx(3)


def test_alleen_de_selectie():
    d = plate_drawing()
    solids = sheet3d.part_solids(d, 2.0, [d.paths[4]])
    assert len(solids) == 1 and solids[0][1].volume() == pytest.approx(800)


def test_werkblad_naar_model_en_vervangen(win):
    sheet = win.doc.sheet
    sheet.merge(plate_drawing())
    sheet.sheet_thickness = 3.0
    groups = win.sheet_to_model()
    assert len(groups) == 2 and win.mode == "3d"
    bb = groups[1].bbox()
    assert bb[1][2] - bb[0][2] == pytest.approx(3.0)
    assert bb[0][0] == pytest.approx(60) and bb[1][1] == pytest.approx(20)
    # nogmaals: de vorige onderdelen worden vervangen, niet verdubbeld
    sheet.sheet_thickness = 4.0
    again = win.sheet_to_model()
    assert len(win.doc.groups) == 2
    assert again[0].bbox()[1][2] == pytest.approx(4.0)
    win.doc.undo()
    assert len(win.doc.groups) == 2


def test_vlak_naar_laserlaag(win):
    ents = win.doc.root
    for a, b in zip(rect(0, 0, 30, 20), np.roll(rect(0, 0, 30, 20), -1, axis=0)):
        ents.add_line(np.array([*a, 0.0]), np.array([*b, 0.0]))
    fid = next(iter(ents.F))
    win.doc.set_selection([("f", win.doc.active, fid)])
    added = win.send_selection_to_sheet("Graveren")
    assert added and all(p.layer == "Graveren" for p in added)
    P = np.vstack([p.pts for p in added])
    assert np.ptp(P[:, 0]) == pytest.approx(30) and np.ptp(P[:, 1]) == pytest.approx(20)
    assert win.mode == "2d"
    # het rechtsklikmenu in 3D biedt de lagen aan
    win.set_mode("3d")
    win.doc.set_selection([("f", win.doc.active, fid)])
    menu = win.build_context_menu()
    labels = [a.text() for a in menu.actions()]
    assert "Laseren / frezen" in labels


def test_plaat_naar_werkblad_neemt_dikte_over(win):
    import manifold3d as mf
    from volura import ops
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube((50, 20, 6)), "#cccccc", "plaat")
    win.doc.set_selection([("g", g.id, g.id)])
    win.doc.sheet.sheet_thickness = None
    added = win.send_selection_to_sheet("Snijden")
    assert added
    assert win.doc.sheet.sheet_thickness == pytest.approx(6.0)
    # en terug: dezelfde plaat komt weer als 6 mm in het model
    win.sheet_view.set_selection([])
    back = win.sheet_to_model()
    assert len(back) == 1 and back[0].bbox()[1][2] == pytest.approx(6.0)


def test_werkbalken_terug_na_2d(win):
    bars = win._toolbars_3d()
    before = {b: b.isVisible() for b in bars}
    assert any(before.values())
    win.set_mode("2d")
    win.set_mode("2d")                  # tweede keer, zoals Plaat… of de doosgenerator doen
    win.open_sheet_settings = lambda: None
    win.set_mode("3d")
    QCoreApplication.processEvents()
    assert {b: b.isVisible() for b in bars} == before


def test_rechtsklik_op_zijvlak_neemt_dat_vlak(win):
    """Rechtsklik op een vlak van een plaat (groep): dat vlak, niet het bovenaanzicht."""
    import manifold3d as mf
    from volura import ops
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube((50, 20, 6)), "#cccccc", "plaat")
    side = next(f for f, F in g.ents.F.items() if F.normal[1] < -0.999)    # voorkant 50 x 6
    win.doc.set_selection([("g", g.id, g.id)])
    win._context_face = (g.id, side)
    menu = win.build_context_menu()
    win._context_face = None
    laser = next(a.menu() for a in menu.actions() if a.text() == "Laseren / frezen")
    next(a for a in laser.actions() if a.text().startswith("Snijden")).trigger()
    P = np.vstack([p.pts for p in win.doc.sheet.paths if p.layer == "Snijden"])
    assert sorted([np.ptp(P[:, 0]), np.ptp(P[:, 1])]) == pytest.approx([6, 50])
    assert not win.doc.sheet.sheet_thickness       # een staand vlak zegt niets over de dikte


def test_rechtsklik_op_bovenvlak_neemt_dikte_over(win):
    import manifold3d as mf
    from volura import ops
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube((50, 20, 6)), "#cccccc", "plaat")
    top = next(f for f, F in g.ents.F.items() if F.normal[2] > 0.999)
    win.doc.sheet.sheet_thickness = None
    added = win.send_selection_to_sheet("Snijden", (g.id, top))
    P = np.vstack([p.pts for p in added])
    assert np.ptp(P[:, 0]) == pytest.approx(50) and np.ptp(P[:, 1]) == pytest.approx(20)
    assert win.doc.sheet.sheet_thickness == pytest.approx(6.0)


def test_rechterkolom_toont_wat_bij_de_modus_hoort(win):
    shown = lambda: {t for t in win.tray.sections if win.tray.is_shown(t)}  # noqa: E731
    assert shown() == {"Entiteitsinfo", "Materialen", "Overzicht", "Printcontrole"}
    win.set_mode("2d")
    assert shown() == {"Entiteitsinfo", "Lagen"}
    assert not win.tray_actions["Materialen"].isVisible()
    win.set_mode("3d")
    assert "Lagen" not in shown() and "Materialen" in shown()


def test_entiteitsinfo_op_het_werkblad(win):
    from PySide6.QtWidgets import QLabel
    sheet = win.doc.sheet
    sheet.merge(plate_drawing())
    sheet.sheet_w, sheet.sheet_h, sheet.sheet_thickness = 300.0, 200.0, 3.0
    win.set_mode("2d")
    win.sheet_view.set_selection([sheet.paths[0]])
    text = " ".join(lab.text() for lab in win.info.body.findChildren(QLabel))
    assert "1 contour geselecteerd" in text and "40.00 × 30.00 mm" in text
    assert "300 × 200 mm" in text and "3 mm" in text
    assert "Filament" not in text and "Groepen" not in text


def test_modusknop_leesbaar_en_plaat_in_beeld(win):
    win.set_mode("2d")
    win.mode_button._anim.stop()
    assert win.mode_button.isChecked()
    win.set_mode("3d")
    QCoreApplication.processEvents()
    assert not win.mode_button.isChecked()
    sheet = win.doc.sheet
    sheet.sheet_w, sheet.sheet_h = 300.0, 200.0
    sheet.merge(plate_drawing())
    win.set_mode("2d")
    win.sheet_view.zoom_extents()
    v = win.sheet_view
    for corner in ((0, 0), (300, 200)):
        q = v.to_screen(corner)
        assert 0 <= q.x() <= v.width() and 0 <= q.y() <= v.height()
