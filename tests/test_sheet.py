"""Het 2D-werkblad naast het 3D-model.

Het werkblad (`Document.sheet`) gaat mee in het bestand en door undo en redo;
`SheetView` tekent het en de gereedschappen uit `volura.sheettools` bewerken het;
het venster schakelt tussen 3D en 2D en voegt er import, export en
"Naar 2D-werkblad kopiëren" aan toe.

Run with:  python -m pytest tests
"""
import json
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu, QToolBar  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import dxf, errors, fileio, svg  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.sheettools import SHEET_TOOLS  # noqa: E402
from volura.sheetview import SheetView, grid_step  # noqa: E402
from volura.vector2d import Drawing  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

GROUND = Plane((0, 0, 0), (0, 0, 1))
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


# ------------------------------------------------------------------ helpers

@pytest.fixture()
def win():
    from volura.app import MainWindow
    from PySide6.QtCore import QCoreApplication
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # de uitgestelde beginweergave nu afhandelen
    w.resize(1000, 700)
    w.view.resize(1000, 700)
    yield w
    w.doc.dirty = False
    w.close()


def sheet_doc():
    """Document met één vierkant en één lijn op het werkblad."""
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True, layer="0")
    doc.sheet.add_path([[30, 30], [60, 45]], layer="0")
    return doc


def view_of(doc, size=(900, 600)):
    v = SheetView(doc)
    v.resize(*size)
    v.show()
    v.zoom_extents()
    return v


def press(view, x, y, mods=Qt.NoModifier):
    p = QPointF(float(x), float(y))
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton,
                                     Qt.LeftButton, mods))


def release(view, x, y, mods=Qt.NoModifier):
    p = QPointF(float(x), float(y))
    view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, p, p, Qt.LeftButton,
                                       Qt.NoButton, mods))


def move(view, x, y, mods=Qt.NoModifier):
    p = QPointF(float(x), float(y))
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, p, p, Qt.LeftButton,
                                    Qt.LeftButton, mods))


def click(view, x, y, mods=Qt.NoModifier):
    press(view, x, y, mods)
    release(view, x, y, mods)


def drag(view, a, b, mods=Qt.NoModifier):
    press(view, a[0], a[1], mods)
    move(view, b[0], b[1], mods)
    release(view, b[0], b[1], mods)


def type_in(view, text):
    """Typen in het meetvak zoals in het venster: toetsen, daarna Enter."""
    for ch in text:
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_unknown, Qt.NoModifier, ch))
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))


def cube_in(win, w=10, d=10, h=10):
    """Kubus van losse geometrie in het 3D-model van het venster."""
    doc = win.doc
    doc.root.add_shape(GROUND, [np.array([[0, 0], [w, 0], [w, d], [0, d]], float)])
    doc.root.pushpull(next(iter(doc.root.F)), -h)
    doc.notify()
    return doc


def pts_of(path):
    return np.asarray(path.pts, dtype=float)


def total_area(drawing):
    from volura.vector2d import regions
    return sum(abs(sum(_signed(s) for s in shape)) for shape in regions(drawing.paths))


def _signed(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ------------------------------------------------------------------ 1. model en bestand

def test_reset_geeft_leeg_werkblad():
    doc = Document()
    assert isinstance(doc.sheet, Drawing)
    # een nieuw werkblad krijgt de standaardlagen; "0" blijft de eerste
    assert doc.sheet.paths == [] and list(doc.sheet.layers)[0] == "0"
    assert {"Snijden", "Graveren", "Hulplijnen"} <= set(doc.sheet.layers)
    doc.sheet.add_path(SQUARE, closed=True)
    doc.reset()
    assert doc.sheet.paths == []


def test_werkblad_gaat_mee_naar_het_bestand(tmp_path):
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True, layer="snij", filled=True, src=1)
    doc.sheet.add_path([[0, 0], [5, 5]], layer="graveer")
    lay = doc.sheet.ensure_layer("graveer")
    lay.color, lay.op, lay.visible = "#ff0000", "engrave", False
    doc.sheet.sheet_w, doc.sheet.sheet_h = 300.0, 200.0
    path = str(tmp_path / "model.volura")
    fileio.save_document(doc, path)

    back = Document()
    fileio.load_document(back, path)
    assert back.sheet.paths == doc.sheet.paths
    assert sorted(back.sheet.layers) == sorted(doc.sheet.layers)
    got = back.sheet.layers["graveer"]
    assert (got.color, got.op, got.visible, got.locked) == ("#ff0000", "engrave", False, False)
    assert (back.sheet.sheet_w, back.sheet.sheet_h) == (300.0, 200.0)


def test_bestand_zonder_werkblad_laadt_als_leeg(tmp_path):
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True)
    path = tmp_path / "oud.volura"
    fileio.save_document(doc, str(path))
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    assert "sheet" in data
    del data["sheet"]
    old = tmp_path / "oud2.volura"
    with zipfile.ZipFile(old, "w") as z:
        z.writestr("document.json", json.dumps(data))
    back = Document()
    fileio.load_document(back, str(old))           # een bestand van vóór het werkblad
    assert back.sheet.paths == [] and list(back.sheet.layers) == ["0"]


# ------------------------------------------------------------------ 2. undo en redo

def test_undo_en_redo_herstellen_het_werkblad():
    doc = Document()
    doc.root.add_shape(GROUND, [np.array(SQUARE, float)])
    faces = len(doc.root.F)
    doc.checkpoint()
    doc.sheet.add_path(SQUARE, closed=True)
    doc.sheet.add_path([[50, 0], [60, 0]])
    assert len(doc.sheet.paths) == 2
    doc.checkpoint()
    doc.sheet.paths[0].pts = pts_of(doc.sheet.paths[0]) + np.array([5.0, 0.0])
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float) + [5.0, 0.0])

    doc.undo()
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float))
    doc.undo()
    assert doc.sheet.paths == []
    assert len(doc.root.F) == faces, "het 3D-model is niet veranderd"
    doc.redo()
    assert len(doc.sheet.paths) == 2
    doc.redo()
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float) + [5.0, 0.0])
    assert len(doc.root.F) == faces


def test_3d_undo_laadt_het_werkblad_ongemoeid():
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True)
    doc.root.add_shape(GROUND, [np.array(SQUARE, float)])
    doc.checkpoint()
    doc.root.erase_faces(list(doc.root.F)[:1])
    assert len(doc.root.F) == 0
    doc.undo()
    assert len(doc.root.F) == 1
    assert len(doc.sheet.paths) == 1 and np.allclose(doc.sheet.bounds(), (0, 0, 10, 10))


def test_2D_wijziging_maakt_het_document_dirty():
    doc = Document()
    seen = []
    doc.changed.connect(lambda: seen.append(1))
    assert not doc.dirty
    doc.checkpoint()
    doc.sheet.add_path(SQUARE, closed=True)
    doc.notify()
    assert doc.dirty and seen


# ------------------------------------------------------------------ 3. het werkblad zelf

def test_rondreis_door_het_scherm():
    v = view_of(sheet_doc())
    for point in ((0.0, 0.0), (12.5, -7.25), (-40.0, 33.5)):
        for _ in range(3):
            p = np.array(point)
            assert np.allclose(v.to_world(v.to_screen(p)), p, atol=1e-9)
            v.zoom_by(1.7)
        v.zoom_extents()


def test_wiel_zoomt_rond_de_cursor():
    from PySide6.QtGui import QWheelEvent
    v = view_of(sheet_doc())
    screen = QPointF(300.0, 220.0)
    before = v.to_world(screen)
    ev = QWheelEvent(screen, screen, QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                     Qt.NoModifier, Qt.NoScrollPhase, False)
    v.wheelEvent(ev)
    assert v.scale > 0
    assert np.allclose(v.to_world(screen), before, atol=1e-6), "het punt onder de cursor blijft"


def test_zoom_passend_zet_alles_in_beeld():
    v = view_of(sheet_doc())
    v.scale = 0.02
    v.zoom_extents()
    box = v.world_rect()
    b = v.doc.sheet.bounds()
    assert box[0] <= b[0] and box[1] <= b[1] and box[2] >= b[2] and box[3] >= b[3]


def test_raster_stapt_1_5_10_50_100():
    steps = [grid_step(s) for s in (8.0, 2.0, 1.0, 0.5, 0.1)]
    assert steps == [1.0, 5.0, 10.0, 50.0, 100.0]
    assert grid_step(0.01) > 100.0 and grid_step(100.0) < 1.0


def test_klik_kiest_het_dichtstbijzijnde_pad():
    v = view_of(sheet_doc())
    sq, ln = v.doc.sheet.paths
    s = v.to_screen((0.0, 5.0))
    assert v.path_at(s.x(), s.y()) is sq
    s2 = v.to_screen((45.0, 37.5))
    assert v.path_at(s2.x(), s2.y()) is ln
    s3 = v.to_screen((5.0, 5.0))                      # binnen het vierkant, maar nergens langs
    assert v.path_at(s3.x(), s3.y()) is None


def test_kaderselectie_links_naar_rechts_en_rechts_naar_links():
    doc = Document()
    inside = doc.sheet.add_path([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)
    crossing = doc.sheet.add_path([[5, 5], [100, 5]])
    far = doc.sheet.add_path([[200, 200], [210, 210]])
    v = view_of(doc)
    v.scale = 4.0
    v.origin = QPointF(60.0, 540.0)
    a = v.to_screen((-5.0, -5.0))
    b = v.to_screen((20.0, 20.0))
    left = (a.x(), a.y())
    right = (b.x(), b.y())

    window = v.paths_in_band(left, right)             # links naar rechts: alles erin
    assert any(p is inside for p in window)
    assert not any(p is crossing for p in window)
    assert not any(p is far for p in window)

    crossing_sel = v.paths_in_band(right, left)       # rechts naar links: ook wat het raakt
    assert any(p is inside for p in crossing_sel)
    assert any(p is crossing for p in crossing_sel)
    assert not any(p is far for p in crossing_sel)


def test_kaderselectie_door_het_gereedschap():
    v = view_of(sheet_doc())
    sq, ln = v.doc.sheet.paths
    a = v.to_screen((-5.0, -5.0))
    b = v.to_screen((20.0, 20.0))
    drag(v, (a.x(), a.y()), (b.x(), b.y()))
    assert v.selected == [sq], "links naar rechts: alleen wat er helemaal in ligt"
    c = v.to_screen((50.0, 50.0))
    d = v.to_screen((8.0, 8.0))
    drag(v, (c.x(), c.y()), (d.x(), d.y()))
    assert {id(p) for p in v.selected} == {id(sq), id(ln)}, "rechts naar links: ook wat het raakt"
    drag(v, (d.x(), d.y()), (c.x(), c.y()))
    assert v.selected == [], "links naar rechts ligt hier niets helemaal in"


def test_shift_voegt_toe_en_haalt_weg():
    v = view_of(sheet_doc())
    sq, ln = v.doc.sheet.paths
    s = v.to_screen((0.0, 5.0))
    click(v, s.x(), s.y())
    assert v.selected == [sq]
    s2 = v.to_screen((45.0, 37.5))
    click(v, s2.x(), s2.y(), Qt.ShiftModifier)
    assert {id(p) for p in v.selected} == {id(sq), id(ln)}
    click(v, s2.x(), s2.y(), Qt.ShiftModifier)
    assert v.selected == [sq]


def test_verborgen_laag_is_niet_te_zien_of_te_kiezen():
    doc = sheet_doc()
    doc.sheet.layers["0"].visible = False
    v = view_of(doc)
    s = v.to_screen((0.0, 5.0))
    assert v.path_at(s.x(), s.y()) is None
    assert v._selectable_paths() == []
    v.select_all()
    assert v.selected == []


def test_muiscoordinaten_in_de_statusbalk():
    v = view_of(sheet_doc())
    got = []
    v.coords_changed.connect(got.append)
    s = v.to_screen((12.0, 8.0))
    move(v, s.x(), s.y())
    assert got and "12.00" in got[-1] and "8.00" in got[-1]


def test_werkblad_tekent_zichzelf(tmp_path):
    from PySide6.QtGui import QImage
    v = view_of(sheet_doc())
    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    v.render(image)                                   # raster, asen, paden en gereedschap

    def dark():
        return sum(1 for y in range(0, 600, 2) for x in range(0, 900, 2)
                   if image.pixelColor(x, y).lightness() < 200)

    assert dark() > 50, "het werkblad tekent niets"


# ------------------------------------------------------------------ 4. gereedschappen

def test_SHEET_TOOLS_register():
    names = [cls.name for cls in SHEET_TOOLS]
    # de vier gereedschappen eronder, daarna de tekengereedschappen
    assert names[:4] == ["select", "move", "rotate", "scale"]
    assert {"line", "rectangle", "roundrect", "circle", "ellipse", "arc", "centerarc",
            "polygon", "text", "offset"} <= set(names)
    v = view_of(sheet_doc())
    assert set(v.tools) == set(names)


def test_verplaatsen_met_getypte_afstand():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("move")
    base = v.to_screen((0.0, 0.0))
    click(v, base.x(), base.y())
    type_in(v, "10;5")
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([10.0, 5.0]))


def test_verplaatsen_met_losse_afstand_langs_de_muis():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("move")
    a = v.to_screen((0.0, 0.0))
    b = v.to_screen((20.0, 0.0))
    click(v, a.x(), a.y())
    move(v, b.x(), b.y())
    v.submit_vcb("5")
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([5.0, 0.0]))


def test_verplaatsen_klik_klik_en_kopie_met_ctrl():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("move")
    a = v.to_screen((0.0, 0.0))
    b = v.to_screen((25.0, 0.0))
    click(v, a.x(), a.y())
    click(v, b.x(), b.y())
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([25.0, 0.0]))

    n = len(doc.sheet.paths)
    c = v.to_screen((0.0, 0.0))
    d = v.to_screen((0.0, 40.0))
    click(v, c.x(), c.y(), Qt.ControlModifier)
    click(v, d.x(), d.y(), Qt.ControlModifier)
    assert len(doc.sheet.paths) == n + 1, "met Ctrl wordt er een kopie bij gezet"
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([25.0, 0.0]))
    copy = doc.sheet.paths[-1]
    assert np.allclose(pts_of(copy), np.array(SQUARE, float) + np.array([25.0, 40.0]))


def test_draaien_90_graden():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("rotate")
    c = v.to_screen((0.0, 0.0))
    click(v, c.x(), c.y())
    v.submit_vcb("90")
    assert np.allclose(pts_of(sq), np.array([[0, 0], [0, 10], [-10, 10], [-10, 0]], float))


def test_draaien_door_te_klikken():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("rotate")
    c = v.to_screen((5.0, 5.0))                       # middelpunt van het vierkant
    click(v, c.x(), c.y())
    p = v.to_screen((5.0, 25.0))                      # 90° linksom vanaf de eerste klik
    click(v, p.x(), p.y())
    assert np.allclose(np.sort(pts_of(sq), axis=0), np.sort(np.array(SQUARE, float), axis=0),
                       atol=1e-6)


def test_schalen_2x():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("scale")
    a = v.to_screen((5.0, 5.0))
    click(v, a.x(), a.y())
    v.submit_vcb("2")
    assert np.allclose(pts_of(sq), np.array([[-5, -5], [15, -5], [15, 15], [-5, 15]], float))


def test_schalen_door_te_slepen():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("scale")
    a = v.to_screen((15.0, 5.0))                      # afstand 10 mm vanaf het midden
    b = v.to_screen((25.0, 5.0))                      # nu 20 mm: factor 2
    drag(v, (a.x(), a.y()), (b.x(), b.y()))
    assert np.allclose(pts_of(sq), np.array([[-5, -5], [15, -5], [15, 15], [-5, 15]], float))


def test_spiegelen_rond_het_midden_van_de_selectie():
    doc = Document()
    p = doc.sheet.add_path([[0, 0], [10, 0], [10, 4]], closed=False)
    v = view_of(doc)
    v.set_selection([p])
    assert v.mirror(True)
    assert np.allclose(pts_of(p), np.array([[10, 0], [0, 0], [0, 4]], float))
    assert v.mirror(False)
    assert np.allclose(pts_of(p), np.array([[10, 4], [0, 4], [0, 0]], float))


def test_alles_selecteren_en_wissen():
    doc = sheet_doc()
    v = view_of(doc)
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.ControlModifier, "a"))
    assert len(v.selected) == 2
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier, ""))
    assert doc.sheet.paths == [] and v.selected == []
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier, "\x1b"))


def test_esc_en_gereedschapstoetsen():
    v = view_of(sheet_doc())
    v.set_selection(list(v.doc.sheet.paths))
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier, "\x1b"))
    assert v.selected == []
    for key, name in ((Qt.Key_M, "move"), (Qt.Key_Q, "rotate"), (Qt.Key_S, "scale"),
                      (Qt.Key_Space, "select")):
        v.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, ""))
        assert v.tool.name == name


def test_elke_bewerking_is_one_undo_stap():
    doc = sheet_doc()
    v = view_of(doc)
    sq, ln = doc.sheet.paths
    before = [pts_of(p).copy() for p in doc.sheet.paths]
    steps = len(doc.undo_stack)
    v.set_selection([sq, ln])
    v.set_tool("move")
    a = v.to_screen((0.0, 0.0))
    b = v.to_screen((7.0, 3.0))
    click(v, a.x(), a.y())
    click(v, b.x(), b.y())
    assert len(doc.undo_stack) == steps + 1, "verplaatsen is één undo-stap"
    doc.undo()
    assert len(doc.sheet.paths) == len(before)
    assert all(np.allclose(pts_of(a), b) for a, b in zip(doc.sheet.paths, before))


# ------------------------------------------------------------------ 5. 3D en 2D in het venster

def sheet_toolbar(win):
    return next((b for b in win.findChildren(QToolBar) if b.objectName() == "SheetToolbar"), None)


def test_werkbalk_sheettoolbar_is_only_in_2d(win):
    bar = sheet_toolbar(win)
    assert bar is not None, "werkbalk met objectName 'SheetToolbar' ontbreekt"
    assert bar.isHidden(), "in 3D is de werkbalk van het werkblad niet te zien"
    win.set_mode("2d")
    assert not bar.isHidden()
    win.set_mode("3d")
    assert bar.isHidden()


def test_set_mode_verandert_het_3d_model_niet(win):
    cube_in(win)
    win.view.set_view("iso")
    win.view.zoom_extents()
    model = win.doc.root.state()
    cam = win.view.cam.state()
    win.set_mode("2d")
    assert win.mode == "2d"
    assert win.central.currentWidget() is win.sheet_view
    assert win.doc.root.state() == model and win.view.cam.state() == cam
    win.set_mode("3d")
    assert win.mode == "3d"
    assert win.central.currentWidget() is win.view
    assert win.doc.root.state() == model and win.view.cam.state() == cam


def test_in_2d_wijken_de_3d_werkbalken_en_gereedschappen(win):
    shown = [b for b in (win.tb_main, win.tb_tools, win.tb_print) if not b.isHidden()]
    win.set_mode("2d")
    assert all(b.isHidden() for b in shown), "de 3D-werkbalken zijn verborgen in 2D"
    assert not win.tool_actions["pushpull"].isEnabled()
    assert not win.a_delete.isEnabled()
    win.set_mode("3d")
    assert [b for b in (win.tb_main, win.tb_tools, win.tb_print) if not b.isHidden()] == shown
    assert win.tool_actions["pushpull"].isEnabled()


def test_beeld_menu_en_statusbalk_schakelen(win):
    beeld = next(m for m in win.menuBar().findChildren(QMenu)
                 if m.title().replace("&", "") == "Beeld")
    labels = [a.text().replace("&", "").replace("…", "")
              for a in beeld.actions() if not a.isSeparator()]
    assert "2D-werkblad" in labels and "Werkbalken" in labels
    assert labels.index("Werkbalken") < labels.index("2D-werkblad")
    assert win.a_sheet.text().replace("&", "") == "2D-werkblad"
    assert win.a_sheet.isCheckable()
    assert "Ctrl+Shift+2" in [s.toString() for s in win.a_sheet.shortcuts()]
    assert win.mode_button.text() == "3D | 2D", "de knop rechtsonder schakelt tussen 3D en 2D"
    win.toggle_sheet_mode()
    assert win.mode == "2d" and win.a_sheet.isChecked()
    assert "2D-werkblad" in win.mode_button.toolTip()
    win.mode_button.click()
    assert win.mode == "3d" and not win.a_sheet.isChecked()
    assert "3D-model" in win.mode_button.toolTip()


def test_modus_hoort_niet_bij_het_bestand(win, tmp_path):
    win.set_mode("2d")
    win.doc.sheet.add_path(SQUARE, closed=True)
    path = str(tmp_path / "m.volura")
    win.doc.path = path
    assert win.file_save()
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    assert "sheet" in data and "mode" not in data
    win.file_open(path)
    assert win.mode == "2d", "laden verandert de gekozen modus niet"
    assert len(win.doc.sheet.paths) == 1


# ------------------------------------------------------------------ 6. import en export

def test_importeren_in_2d_naar_het_werkblad(win, tmp_path):
    src = Drawing()
    src.add_path(SQUARE, closed=True, layer="snij", filled=True, src=1)
    src.add_path([[40, 0], [50, 0], [50, 10]], closed=False, layer="graveer")
    path = str(tmp_path / "tekening.svg")
    svg.write_svg(src, path)

    win.set_mode("2d")
    added = win.import_vector_to_sheet(path)
    assert added and len(added) == len(win.doc.sheet.paths)
    assert np.allclose(win.doc.sheet.bounds(), src.bounds(), atol=1e-3), "posities blijven"
    assert {"snij", "graveer"} <= set(win.doc.sheet.layers)
    assert win.mode == "2d"
    assert set(id(p) for p in win.sheet_view.selected) == {id(p) for p in added}

    again = win.import_vector_to_sheet(path)
    # lagen worden op naam samengevoegd: er komen er twee bij, ook bij de tweede keer
    before = len(win.doc.sheet.layers)
    assert len(again) == 2
    win.import_vector_to_sheet(path)
    assert len(win.doc.sheet.layers) == before, "lagen worden op naam samengevoegd"


def test_importeren_in_3d_blijft_naar_het_model_gaan(win, tmp_path):
    src = Drawing()
    src.add_path(SQUARE, closed=True, layer="0", filled=True, src=1)
    path = str(tmp_path / "blok.svg")
    svg.write_svg(src, path)
    assert win.mode == "3d"
    win.import_vector(path, options={"anchor": "center", "open_edges": True})
    assert win.doc.sheet.paths == [], "in 3D gaat importeren naar het model"
    assert win.doc.root.F or win.doc.root.E


def test_exporteren_in_2d_schrijft_zichtbare_lagen(win, tmp_path):
    doc = win.doc
    doc.sheet.add_path(SQUARE, closed=True, layer="snij", filled=True, src=1)
    doc.sheet.add_path([[50, 0], [60, 0]], layer="graveer")
    doc.sheet.layers["graveer"].visible = False
    win.set_mode("2d")

    a = str(tmp_path / "blad.svg")
    assert win.export_vector("svg", path=a) == a
    back = svg.read_svg(a)
    assert {p.layer for p in back.paths} == {"snij"}
    assert abs(total_area(back) - 100.0) < 1e-3

    b = str(tmp_path / "blad.dxf")
    assert win.export_vector("dxf", path=b) == b
    assert {p.layer for p in dxf.read_dxf(b).paths} == {"snij"}
    assert "contouren geëxporteerd naar" in win.hint.text()


def test_naar_2d_werkblad_kopieren_met_bovenaanzicht(win):
    cube_in(win, 10, 10, 10)
    model = win.doc.root.state()
    added = win.copy_to_sheet({"source": "top"})
    assert added and len(added) == 1
    assert win.mode == "2d"
    assert win.doc.root.state() == model, "het 3D-model blijft onveranderd"
    p = added[0]
    assert p.closed and np.allclose(np.sort(pts_of(p), axis=0),
                                    np.sort(np.array(SQUARE, float), axis=0), atol=1e-6)
    assert abs(total_area(win.doc.sheet) - 100.0) < 1e-3
    assert win.sheet_view.selected == added


def test_naar_2d_werkblad_kopieren_van_lege_opbrengst(win, monkeypatch):
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
    assert win.copy_to_sheet({"source": "top"}) is None
    assert win.doc.sheet.paths == []
    assert box.messages and "volume" in box.messages[0]
    assert win.hint.text() == "Er is niets gekopieerd."


def test_kopieren_kan_ook_doorsnede_en_vlakken(win):
    cube_in(win, 10, 10, 10)
    for opts in ({"source": "section", "z": 5.0}, {"source": "faces"}):
        win.doc.set_selection([("f", 0, f) for f in list(win.doc.root.F)[:1]])
        added = win.copy_to_sheet(dict(opts))
        assert added and win.doc.sheet.paths, opts
        win.doc.undo()
    assert win.doc.sheet.paths == [], "undo haalt beide kopieen weer weg"


def test_pannen_met_de_middelste_knop_en_spatie():
    v = view_of(sheet_doc())
    before = QPointF(v.origin)
    p = QPointF(400.0, 300.0)
    q = QPointF(460.0, 340.0)
    v.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.MiddleButton,
                                  Qt.MiddleButton, Qt.NoModifier))
    v.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, q, q, Qt.MiddleButton,
                                 Qt.MiddleButton, Qt.NoModifier))
    v.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, q, q, Qt.MiddleButton,
                                    Qt.NoButton, Qt.NoModifier))
    assert (v.origin.x(), v.origin.y()) == pytest.approx((before.x() + 60, before.y() + 40))

    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))
    start = QPointF(v.origin)
    drag(v, (400.0, 300.0), (350.0, 300.0))
    assert v.origin.x() == pytest.approx(start.x() - 50), "spatie + slepen pant"
    v.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier, " "))
    start = QPointF(v.origin)
    drag(v, (400.0, 300.0), (420.0, 300.0))
    assert v.origin.x() == pytest.approx(start.x()), "zonder spatie slepen is geen pannen"


def test_plaatrechthoek_en_laagkleuren_worden_getekend():
    from PySide6.QtGui import QImage
    doc = Document()
    doc.sheet.sheet_w, doc.sheet.sheet_h = 300.0, 200.0
    doc.sheet.add_path(SQUARE, closed=True, layer="snij")
    doc.sheet.layers["snij"].color = "#0000ff"
    v = view_of(doc)
    v.zoom_extents()
    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    v.render(image)

    def blues():
        return sum(1 for y in range(0, 600, 2) for x in range(0, 900, 2)
                   if image.pixelColor(x, y).blue() > 150
                   and image.pixelColor(x, y).red() < 120)

    assert blues() > 20, "de laagkleur wordt niet getekend"
    v.set_selection(list(doc.sheet.paths))
    v.render(image)
    assert blues() > 20, "een geselecteerd pad is ook blauw"
    doc.sheet.layers["snij"].visible = False
    v.render(image)
    assert blues() == 0, "een verborgen laag is niet te zien"


def test_ongeldige_invoer_in_het_meetvak():
    doc = sheet_doc()
    v = view_of(doc)
    v.set_selection([doc.sheet.paths[0]])
    v.set_tool("move")
    a = v.to_screen((0.0, 0.0))
    click(v, a.x(), a.y())
    v.submit_vcb("nee")
    assert "Ongeldige invoer" in v._status_shown
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float))


def test_een_interne_fout_in_2d_geeft_alleen_een_melding(monkeypatch):
    doc = sheet_doc()
    v = view_of(doc)
    logged = []
    monkeypatch.setattr(errors, "log_exception", lambda *a, **k: logged.append(k))
    v.tools["select"].mouse_press = lambda e, x, y: (_ for _ in ()).throw(RuntimeError("kapot"))
    click(v, 100.0, 100.0)                     # de app mag gewoon doorgaan
    assert logged, "de fout wordt gelogd"
    assert errors.ERROR_STATUS in v._status_shown


def test_meetvak_in_de_statusbalk_werkt_in_2d(win):
    cube_in(win)
    win.copy_to_sheet({"source": "top"})
    sv = win.sheet_view
    sq = win.doc.sheet.paths[0]
    sv.set_selection([sq])
    sv.set_tool("move")
    a = sv.to_screen((0.0, 0.0))
    click(sv, a.x(), a.y())
    win.vcb.setText("3;4")
    win.on_vcb_return()
    assert np.allclose(np.sort(pts_of(sq), axis=0),
                       np.sort(np.array(SQUARE, float) + [3.0, 4.0], axis=0))
    assert win.mode == "2d"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and not fn.__code__.co_argcount:
            fn()
            print("ok", name)
