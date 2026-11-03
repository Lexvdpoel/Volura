"""Tekengereedschappen en bewerkingen op het 2D-werkblad, met snappen.

De gereedschappen uit `volura.sheettools` worden aangestuurd via hun muis- en
meetvakfuncties op een offscreen `SheetView`, met een hulpje dat wereldmillimeters
naar muisgebeurtenissen omzet. `volura.sheetsnap` laat de cursor naar eindpunten,
middens, middelpunten, snijpunten, paden en het raster trekken en lijnt horizontaal
of verticaal uit; de bewerkingen (samenvoegen, aftrekken, snijpunt, aaneenrijgen,
opsplitsen, uitlijnen, gelijk verdelen, dupliceren) zijn steeds één undo-stap.

Run with:  python -m pytest tests
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QCoreApplication, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu, QToolBar  # noqa: E402

from volura import fonts  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.sheettools import (SHEET_TOOLS, circle_loop, distribute_deltas, grid_offsets,
                               rounded_rect_loop)  # noqa: E402
from volura.sheetview import SheetView  # noqa: E402
from volura.vector2d import polygon_area, regions  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

SCALE = 4.0                 # pixels per mm
ORIGIN = (200.0, 400.0)     # de schermplek van wereld (0, 0)


# ------------------------------------------------------------------ helpers

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(1000, 700)
    w.sheet_view.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def view_of(doc, size=(900, 600)):
    """Een werkblad met bekende schaal, zodat millimeters precies op pixels vallen."""
    v = SheetView(doc)
    v.resize(*size)
    v.show()
    v.scale = SCALE
    v.origin = QPointF(*ORIGIN)
    return v


def sheet_view(doc=None, grid=True, **kw):
    doc = doc or Document()
    v = view_of(doc)
    v.snap_to_grid = grid
    for key, value in kw.items():
        setattr(v, key, value)
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


def dblclick(view, x, y):
    p = QPointF(float(x), float(y))
    view.mouseDoubleClickEvent(QMouseEvent(QEvent.MouseButtonDblClick, p, p, Qt.LeftButton,
                                           Qt.LeftButton, Qt.NoModifier))


def type_in(view, text):
    """Typen in het meetvak zoals in het venster: toetsen, daarna Enter."""
    for ch in text:
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_unknown, Qt.NoModifier, ch))
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))


def W(view, x, y):
    """Wereldmillimeters naar het schermpunt dat eronder staat."""
    s = view.to_screen((float(x), float(y)))
    return s.x(), s.y()


def wclick(view, x, y, mods=Qt.NoModifier):
    click(view, *W(view, x, y), mods)


def wmove(view, x, y, mods=Qt.NoModifier):
    move(view, *W(view, x, y), mods)


def wtype(view, text):
    type_in(view, text)


def pts_of(path):
    return np.asarray(path.pts, dtype=float)


def area_of(path) -> float:
    return polygon_area(path.pts) if path.closed else 0.0


def total_area(paths) -> float:
    return sum(abs(sum(polygon_area(loop) for loop in shape)) for shape in regions(paths))


def new_paths(doc, before):
    """De paden die er sinds `before` bij zijn gekomen."""
    return doc.sheet.paths[before:]


# ------------------------------------------------------------------ 1. gereedschappen

def test_rechthoek_30_20_is_gesloten_met_oppervlakte_600():
    v = sheet_view()
    doc = v.doc
    v.set_tool("rectangle")
    wclick(v, 0.0, 0.0)
    wmove(v, 12.0, 7.0)
    assert v.tool.p0 is not None
    wtype(v, "30;20")
    made = new_paths(doc, 0)
    assert len(made) == 1 and made[0].closed
    assert abs(area_of(made[0]) - 600.0) < 1e-6
    assert np.allclose(np.sort(pts_of(made[0]), axis=0),
                       np.sort(np.array([[0, 0], [30, 0], [30, 20], [0, 20]], float), axis=0))


def test_rechthoek_door_twee_hoeken_te_klikken():
    v = sheet_view()
    v.set_tool("rectangle")
    wclick(v, 4.0, 4.0)
    wclick(v, 24.0, 14.0)
    assert len(v.doc.sheet.paths) == 1
    assert abs(area_of(v.doc.sheet.paths[0]) - 200.0) < 1e-6


def test_afgeronde_rechthoek_r5():
    v = sheet_view()
    doc = v.doc
    v.set_tool("roundrect")
    wclick(v, 0.0, 0.0)
    wclick(v, 30.0, 20.0)
    assert v.tool.box is not None
    wtype(v, "5")
    path = new_paths(doc, 0)[-1]
    want = 600.0 - (4.0 - math.pi) * 25.0
    assert path.closed
    assert abs(area_of(path) - want) / want < 0.005
    assert len(path.pts) > 8, "de hoeken zijn afgevlakt"


def test_cirkel_r10_met_middelpunt_en_straal_in_meta():
    v = sheet_view()
    doc = v.doc
    v.set_tool("circle")
    wclick(v, 40.0, 10.0)
    wtype(v, "10")
    path = new_paths(doc, 0)[-1]
    assert path.closed and path.filled
    assert abs(area_of(path) - math.pi * 100.0) / (math.pi * 100.0) < 0.001
    assert path.meta.get("kind") == "circle"
    assert np.allclose(path.meta["center"], [40.0, 10.0])
    assert abs(path.meta["radius"] - 10.0) < 1e-9


def test_cirkel_zijden_met_12s():
    v = sheet_view()
    v.set_tool("circle")
    wtype(v, "12s")
    assert v.tool.segments == 12
    wclick(v, 0.0, 0.0)
    wtype(v, "10")
    path = v.doc.sheet.paths[-1]
    assert len(path.pts) == 12
    assert abs(total_area([path]) - 300.0) < 1.0, "twaalfhoek met straal 10"


def test_ellips_met_halvematen():
    v = sheet_view()
    v.set_tool("ellipse")
    wclick(v, 0.0, 0.0)
    wtype(v, "20;10")
    path = v.doc.sheet.paths[-1]
    want = math.pi * 20.0 * 10.0
    assert abs(area_of(path) - want) / want < 0.001
    assert path.meta["rx"] == 20.0 and path.meta["ry"] == 10.0


def test_veelhoek_6_zijden():
    v = sheet_view()
    v.set_tool("polygon")
    assert v.tool.segments == 6
    wtype(v, "5s")
    assert v.tool.segments == 5
    wclick(v, 0.0, 0.0)
    wtype(v, "10")
    path = v.doc.sheet.paths[-1]
    assert len(path.pts) == 5 and path.closed
    want = 0.5 * 5 * 100.0 * math.sin(2 * math.pi / 5)
    assert abs(area_of(path) - want) < 1e-6


def test_boog_door_drie_punten_krijgt_middelpunt_en_straal():
    v = sheet_view()
    v.set_tool("arc")
    wclick(v, 0.0, 0.0)
    wclick(v, 10.0, 10.0)
    wclick(v, 20.0, 0.0)
    path = v.doc.sheet.paths[-1]
    assert not path.closed and len(path.pts) > 3
    assert path.meta["kind"] == "arc"
    assert np.allclose(path.meta["center"], [10.0, 0.0])
    assert abs(path.meta["radius"] - 10.0) < 1e-6
    assert np.allclose(pts_of(path)[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(pts_of(path)[-1], [20.0, 0.0], atol=1e-6)


def test_boog_om_middelpunt_met_hoek_uit_het_meetvak():
    v = sheet_view()
    v.set_tool("centerarc")
    wclick(v, 0.0, 0.0)
    wclick(v, 10.0, 0.0)
    wtype(v, "90")
    path = v.doc.sheet.paths[-1]
    assert not path.closed
    assert abs(path.meta["radius"] - 10.0) < 1e-6
    assert np.allclose(pts_of(path)[0], [10.0, 0.0], atol=1e-6)
    assert np.allclose(pts_of(path)[-1], [0.0, 10.0], atol=1e-6)
    quarter = math.radians(90.0) * 10.0
    length = float(np.sum(np.hypot(*np.diff(pts_of(path), axis=0).T)))
    assert abs(length - quarter) / quarter < 0.01


def test_polylijn_sluit_door_op_het_startpunt_te_klikken():
    v = sheet_view()
    doc = v.doc
    v.set_tool("line")
    wclick(v, 0.0, 0.0)
    wclick(v, 20.0, 0.0)
    wclick(v, 20.0, 10.0)
    assert len(v.tool.pts) == 3
    wclick(v, 0.0, 0.0)
    made = new_paths(doc, 0)
    assert len(made) == 1 and made[0].closed
    assert len(made[0].pts) == 3
    assert abs(area_of(made[0]) - 100.0) < 1e-6, "de driehoek is gesloten"


def test_polylijn_met_enter_en_met_dubbelklik():
    v = sheet_view()
    doc = v.doc
    v.set_tool("line")
    wclick(v, 0.0, 0.0)
    wclick(v, 10.0, 0.0)
    wclick(v, 10.0, 10.0)
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))
    assert len(doc.sheet.paths) == 1 and not doc.sheet.paths[0].closed
    wclick(v, 40.0, 0.0)
    wclick(v, 50.0, 0.0)
    dblclick(v, *W(v, 50.0, 0.0))
    assert len(doc.sheet.paths) == 2
    assert len(doc.sheet.paths[1].pts) == 2, "de dubbelklik telt niet als extra punt"


def test_lijn_lengte_en_hoek_uit_het_meetvak():
    v = sheet_view()
    v.set_tool("line")
    wclick(v, 0.0, 0.0)
    wmove(v, 10.0, 0.0)
    wtype(v, "25")
    assert np.allclose(v.tool.pts[-1], [25.0, 0.0], atol=1e-9)
    wtype(v, "10<90")
    assert np.allclose(v.tool.pts[-1], [25.0, 10.0], atol=1e-9)
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))
    path = v.doc.sheet.paths[-1]
    assert np.allclose(pts_of(path), [[0, 0], [25, 0], [25, 10]], atol=1e-9)


def test_verschuiving_met_afgeronde_hoeken():
    v = sheet_view()
    doc = v.doc
    doc.sheet.add_path([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)
    v.set_tool("offset")
    wclick(v, 0.0, 5.0)
    assert v.tool.target is doc.sheet.paths[0]
    wmove(v, -3.0, 5.0)
    assert v.tool.distance == pytest.approx(3.0)
    wtype(v, "2")
    made = new_paths(doc, 1)
    assert len(made) >= 1 and all(p.closed for p in made)
    want = 14.0 * 14.0 - (4.0 - math.pi) * 4.0
    assert abs(total_area(made) - want) / want < 0.01


def test_verschuiving_scherp_en_naar_binnen():
    v = sheet_view()
    v.offset_round = False
    doc = v.doc
    doc.sheet.add_path([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)
    v.set_tool("offset")
    wclick(v, 5.0, 10.0)
    wmove(v, 5.0, 8.0)
    assert v.tool.distance < 0.0, "binnen de contour is de afstand negatief"
    wtype(v, "2")
    made = new_paths(doc, 1)
    assert abs(total_area(made) - 36.0) < 1e-6


# ------------------------------------------------------------------ 2. snappen

def test_snappen_naar_eindpunt_en_midden_van_een_lijnstuk():
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    v = sheet_view(doc)
    got = v.snap_engine.snap(*W(v, 20.6, 0.4))
    assert got.kind == "endpoint" and got.label == "Eindpunt"
    assert np.allclose(got.point, [20.0, 0.0])
    got = v.snap_engine.snap(*W(v, 9.6, 0.5))
    assert got.kind == "midpoint" and got.label == "Middelpunt"
    assert np.allclose(got.point, [10.0, 0.0])
    got = v.snap_engine.snap(*W(v, 4.0, 1.2))
    assert got.kind == "onpath" and got.label == "Op pad"
    assert abs(got.point[1]) < 1e-9


def test_snappen_naar_middelpunt_van_cirkel_en_boog():
    doc = Document()
    loop = circle_loop([40.0, 20.0], 10.0)
    doc.sheet.add_path(loop, closed=True, meta={"kind": "circle",
                                                "center": [40.0, 20.0], "radius": 10.0})
    v = sheet_view(doc)
    got = v.snap_engine.snap(*W(v, 40.7, 19.4))
    assert got.kind == "center" and got.label == "Middelpunt"
    assert np.allclose(got.point, [40.0, 20.0])


def test_snijpunt_van_twee_lijnen_wordt_gevonden():
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    doc.sheet.add_path([[10, -10], [10, 10]])
    v = sheet_view(doc)
    got = v.snap_engine.snap(*W(v, 10.4, 0.6))
    assert got.kind == "intersection" and got.label == "Snijpunt"
    assert np.allclose(got.point, [10.0, 0.0])


def test_raster_aan_uit_op_de_werkbalk():
    doc = Document()
    v = sheet_view(doc, grid=True)
    got = v.snap_engine.snap(*W(v, 50.7, 49.6))
    assert got.kind == "grid" and got.label == "Raster"
    assert np.allclose(got.point, [50.0, 50.0])
    v.set_grid_snap(False)
    got = v.snap_engine.snap(*W(v, 50.7, 49.6))
    assert got.kind == "" and np.allclose(got.point, [50.7, 49.6], atol=1e-9)


def test_uitlijnen_met_hulplijn_hoek_aan_hoek():
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    v = sheet_view(doc)
    got = v.snap_engine.snap(*W(v, 30.0, 0.6), anchor=np.array([25.0, 0.0]))
    assert got.kind == "align"
    assert abs(got.point[1]) < 1e-9, "horizontaal op één lijn met de vorige klik"
    assert got.guide and got.guide[0] == "h"
    got = v.snap_engine.snap(*W(v, 20.4, 8.0))
    assert got.kind == "align" and got.guide[0] == "v"
    assert abs(got.point[0] - 20.0) < 1e-9, "verticaal op één lijn met het eindpunt"


def test_het_gereedschap_gebruikt_het_snap_punt():
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    v = sheet_view(doc)
    v.set_tool("line")
    wclick(v, 20.4, 0.5)                       # naast het eindpunt
    assert np.allclose(v.tool.pts[0], [20.0, 0.0])
    wclick(v, 0.6, -0.4)
    assert np.allclose(v.tool.pts[1], [0.0, 0.0])


def test_markering_wordt_getekend():
    from PySide6.QtGui import QImage
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    v = sheet_view(doc)
    v.set_tool("line")
    wmove(v, 20.3, 0.3)
    assert v.tool.snap.kind == "endpoint"
    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    v.render(image)
    reds = sum(1 for y in range(0, 600, 2) for x in range(0, 900, 2)
               if image.pixelColor(x, y).red() > 150 and image.pixelColor(x, y).green() < 130)
    assert reds > 4, "de snap-markering is niet te zien"


# ------------------------------------------------------------------ 3. tekst

def test_tekst_Ao_met_minstens_twee_gaten():
    if not fonts.has_fonts():
        pytest.skip("dit systeem heeft geen lettertypen")
    v = sheet_view()
    doc = v.doc
    tool = v.tools["text"]
    assert tool.set_text("Ao", fonts.default_family(), 10.0, False)
    v.set_tool("text")
    wclick(v, 0.0, 0.0)
    made = new_paths(doc, 0)
    assert len(made) >= 4, "buitenranden en gaten"
    assert all(p.closed for p in made)
    assert len({int(p.src) for p in made}) == 1, "één bronelement"
    holes = [p for p in made if area_of(p) < 0.0]
    assert len(holes) >= 2, "de gaten van A en o"
    outsides = [p for p in made if area_of(p) > 0.0]
    a = min(outsides, key=lambda p: float(pts_of(p)[:, 0].min()))
    height = float(pts_of(a)[:, 1].max() - pts_of(a)[:, 1].min())
    assert abs(height - 10.0) / 10.0 < 0.10
    filled = sum(abs(area_of(p)) for p in outsides) - sum(abs(area_of(p)) for p in holes)
    assert abs(total_area(made) - filled) < 1e-6


def test_tekst_hoogte_wijzigen_uit_het_meetvak():
    if not fonts.has_fonts():
        pytest.skip("dit systeem heeft geen lettertypen")
    v = sheet_view()
    tool = v.tools["text"]
    tool.set_text("A", fonts.default_family(), 10.0, False)
    v.set_tool("text")
    wtype(v, "20")
    assert tool.height == pytest.approx(20.0)
    top = max(float(p[:, 1].max()) for p in tool.loops)
    assert abs(top - 20.0) / 20.0 < 0.10, "de hoofdletterhoogte volgt het meetvak"
    assert min(float(p[:, 1].min()) for p in tool.loops) > -1.0, "de tekst staat op de basislijn"


# ------------------------------------------------------------------ 4. bewerkingen

def two_squares():
    doc = Document()
    a = doc.sheet.add_path([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True,
                           filled=True, src=1)
    b = doc.sheet.add_path([[5, 0], [15, 0], [15, 10], [5, 10]], closed=True,
                           filled=True, src=2)
    return doc, a, b


def test_union_aftrekken_en_intersect_van_twee_vierkanten():
    doc, a, b = two_squares()
    v = sheet_view(doc)
    v.set_selection([a, b])
    assert v.sheet_boolean("union")
    assert abs(total_area(doc.sheet.paths) - 150.0) < 1e-6
    doc.undo()
    v.set_selection(list(doc.sheet.paths))
    assert v.sheet_boolean("subtract")
    assert abs(total_area(doc.sheet.paths) - 50.0) < 1e-6
    doc.undo()
    v.set_selection(list(doc.sheet.paths))
    assert v.sheet_boolean("intersect")
    assert abs(total_area(doc.sheet.paths) - 50.0) < 1e-6
    doc.undo()
    assert len(doc.sheet.paths) == 2


def test_booleaanse_bewerking_zonder_selectie_geeft_melding():
    doc, a, _b = two_squares()
    v = sheet_view(doc)
    v.clear_selection()
    assert not v.sheet_boolean("union")
    assert "selecteer" in v._status_shown.lower()
    assert len(doc.sheet.paths) == 2
    v.set_selection([a])
    assert not v.sheet_boolean("subtract")
    assert len(doc.sheet.paths) == 2


def test_bewerking_met_gaten_blijft_een_vorm():
    doc = Document()
    outer = doc.sheet.add_path([[0, 0], [20, 0], [20, 20], [0, 20]], closed=True,
                               filled=True, src=1)
    inner = doc.sheet.add_path([[5, 5], [15, 5], [15, 15], [5, 15]], closed=True,
                               filled=True, src=2)
    v = sheet_view(doc)
    v.set_selection([outer, inner])
    assert v.sheet_boolean("subtract")
    paths = doc.sheet.paths
    assert len(paths) == 2 and sum(1 for p in paths if area_of(p) < 0) == 1
    assert abs(total_area(paths) - 300.0) < 1e-6


def test_aaneenrijgen_en_opsplitsen():
    doc = Document()
    for seg in ([[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]], [[0, 10], [0, 0]]):
        doc.sheet.add_path(seg)
    v = sheet_view(doc)
    v.set_selection(list(doc.sheet.paths))
    assert v.join_selected() == 1
    assert doc.sheet.paths[0].closed and len(doc.sheet.paths[0].pts) == 4
    v.set_selection(list(doc.sheet.paths))
    assert v.split_selected() == 4
    assert all(len(p.pts) == 2 and not p.closed for p in doc.sheet.paths)


def test_uitlijnen_links_midden_en_rechts():
    doc = Document()
    a = doc.sheet.add_path([[0, 0], [4, 0], [4, 2], [0, 2]], closed=True)
    b = doc.sheet.add_path([[10, 5], [14, 5], [14, 7], [10, 7]], closed=True)
    v = sheet_view(doc)
    v.set_selection([a, b])
    assert v.align_selected("left")
    assert pts_of(a)[:, 0].min() == pytest.approx(0.0)
    assert pts_of(b)[:, 0].min() == pytest.approx(0.0)
    doc.undo()
    v.set_selection(list(doc.sheet.paths))
    assert v.align_selected("right")
    assert pts_of(doc.sheet.paths[0])[:, 0].max() == pytest.approx(14.0)
    assert pts_of(doc.sheet.paths[1])[:, 0].max() == pytest.approx(14.0)
    doc.undo()
    v.set_selection(list(doc.sheet.paths))
    assert v.align_selected("center")
    assert np.allclose(np.mean(pts_of(doc.sheet.paths[0]), axis=0),
                       np.mean(pts_of(doc.sheet.paths[1]), axis=0), atol=1e-9)


def test_gelijk_verdelen():
    doc = Document()
    for x in (0.0, 11.0, 30.0):
        doc.sheet.add_path([[x, 0], [x + 2, 0], [x + 2, 2], [x, 2]], closed=True)
    v = sheet_view(doc)
    v.set_selection(list(doc.sheet.paths))
    assert v.distribute_selected("x")
    mids = sorted(float(pts_of(p)[:, 0].mean()) for p in doc.sheet.paths)
    assert mids[0] == pytest.approx(1.0) and mids[-1] == pytest.approx(31.0)
    assert mids[1] == pytest.approx(16.0), "het middenpunt gaat precies in het midden"
    assert np.allclose(distribute_deltas([doc.sheet.paths[0]], "x")[0], [0.0, 0.0])


def test_dupliceren_in_raster_3_bij_2_met_15_mm():
    doc = Document()
    doc.sheet.add_path([[0, 0], [4, 0], [4, 2], [0, 2]], closed=True)
    v = sheet_view(doc)
    v.set_selection(list(doc.sheet.paths))
    made = v.duplicate_grid(3, 2, 15.0, 15.0)
    assert len(made) == 5
    mins = sorted(tuple(np.round(pts_of(p).min(axis=0), 6)) for p in doc.sheet.paths)
    assert mins == [(0.0, 0.0), (0.0, 15.0), (15.0, 0.0), (15.0, 15.0),
                    (30.0, 0.0), (30.0, 15.0)]
    assert grid_offsets(1, 1, 10.0, 10.0) == []


def test_dupliceren_in_rij_met_cirkel_houdt_het_middelpunt():
    doc = Document()
    doc.sheet.add_path(circle_loop([0.0, 0.0], 5.0), closed=True, filled=True,
                       meta={"kind": "circle", "center": [0.0, 0.0], "radius": 5.0})
    v = sheet_view(doc)
    v.set_selection(list(doc.sheet.paths))
    made = v.duplicate_grid(3, 1, 20.0, 0.0)
    assert len(made) == 2
    centers = sorted(tuple(np.round(p.meta["center"], 6)) for p in doc.sheet.paths)
    assert centers == [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)]
    assert all(p.meta["radius"] == pytest.approx(5.0) for p in doc.sheet.paths)


# ------------------------------------------------------------------ 5. undo

def test_elke_gereedschap_bewerking_is_one_undo_stap():
    v = sheet_view()
    doc = v.doc
    steps = len(doc.undo_stack)
    v.set_tool("rectangle")
    wclick(v, 0.0, 0.0)
    wtype(v, "30;20")
    assert len(doc.undo_stack) == steps + 1
    v.set_tool("circle")
    wclick(v, 50.0, 0.0)
    wtype(v, "5")
    assert len(doc.undo_stack) == steps + 2
    assert len(doc.sheet.paths) == 2
    doc.undo()
    assert len(doc.sheet.paths) == 1
    doc.undo()
    assert doc.sheet.paths == []
    doc.redo()
    doc.redo()
    assert len(doc.sheet.paths) == 2


def test_bewerkingen_zijn_ongedaan_te_maken():
    doc, a, b = two_squares()
    v = sheet_view(doc)
    v.set_selection([a, b])
    v.sheet_boolean("union")
    doc.undo()
    assert len(doc.sheet.paths) == 2
    assert abs(total_area(doc.sheet.paths) - 150.0) < 1e-6

    doc2 = Document()
    doc2.sheet.add_path([[0, 0], [4, 0], [4, 2], [0, 2]], closed=True)
    v2 = sheet_view(doc2)
    v2.set_selection(list(doc2.sheet.paths))
    v2.duplicate_grid(3, 2, 15.0, 15.0)
    doc2.undo()
    assert len(doc2.sheet.paths) == 1
    v2.set_selection(list(doc2.sheet.paths))
    doc2.checkpoint()
    v2.split_selected()
    doc2.undo()
    assert len(doc2.sheet.paths) == 1 and len(doc2.sheet.paths[0].pts) == 4


def test_snappen_en_gereedschap_veranderen_het_document_niet():
    doc = Document()
    doc.sheet.add_path([[0, 0], [20, 0]])
    v = sheet_view(doc)
    steps = len(doc.undo_stack)
    v.set_tool("line")
    wmove(v, 20.2, 0.2)
    wmove(v, 10.1, 0.1)
    v.set_tool("select")
    assert len(doc.undo_stack) == steps
    assert len(doc.sheet.paths) == 1


# ------------------------------------------------------------------ 6. bediening

def test_SHEET_TOOLS_heeft_de_tekengereedschappen():
    names = [cls.name for cls in SHEET_TOOLS]
    for want in ("line", "rectangle", "roundrect", "circle", "ellipse", "arc", "centerarc",
                 "polygon", "text", "offset"):
        assert want in names
    v = sheet_view()
    assert set(v.tools) == set(names)
    for name in names:
        tool = v.tools[name]
        assert tool.status_text().strip(), f"{name}: geen statustekst"
        assert isinstance(tool.vcb_label(), str)


def test_sneltoetsen_zijn_dezelfde_als_in_3d():
    v = sheet_view()
    for key, name in ((Qt.Key_L, "line"), (Qt.Key_R, "rectangle"), (Qt.Key_C, "circle"),
                      (Qt.Key_A, "arc"), (Qt.Key_F, "offset"), (Qt.Key_T, "text"),
                      (Qt.Key_E, "ellipse"), (Qt.Key_P, "polygon"), (Qt.Key_N, "roundrect"),
                      (Qt.Key_M, "move"), (Qt.Key_Q, "rotate"), (Qt.Key_S, "scale")):
        v.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, ""))
        assert v.tool.name == name, f"toets {key} geeft {v.tool.name}"


def test_ongeldige_invoer_in_het_meetvak_geeft_melding():
    v = sheet_view()
    v.set_tool("rectangle")
    wtype(v, "30;20")
    assert "klik eerst" in v._status_shown
    wclick(v, 0.0, 0.0)
    v.submit_vcb("nee")
    assert "Ongeldige invoer" in v._status_shown
    assert v.doc.sheet.paths == []


def test_rechtsklikmenu_met_de_bewerkingen():
    doc, a, b = two_squares()
    v = sheet_view(doc)
    menu = v.build_context_menu()
    assert isinstance(menu, QMenu)
    labels = [a.text().replace("&", "") for a in menu.actions()]
    assert {"Samenvoegen", "Aftrekken", "Snijpunt", "Aaneenrijgen",
            "Opsplitsen in lijnstukken"} <= set(labels)
    assert "Dupliceren in rij of raster…" in labels
    submenus = [a.menu() for a in menu.actions() if a.menu() is not None]
    align = [sorted(x.text() for x in m.actions()) for m in submenus if m.title() == "Uitlijnen"]
    assert align and set(align[0]) == {"Links", "Midden", "Rechts", "Boven", "Onder"}


def test_werkbalk_sheettoolbar_heeft_de_nieuwe_knoppen(win):
    bar = next((b for b in win.findChildren(QToolBar)
                if b.objectName() == "SheetToolbar"), None)
    assert bar is not None
    labels = [a.text().replace("&", "") for a in bar.actions() if not a.isSeparator()]
    for want in ("Lijn", "Rechthoek", "Afgeronde rechthoek", "Cirkel", "Ellips", "Boog",
                 "Veelhoek", "Tekst", "Verschuiving", "Contouren samenvoegen",
                 "Contouren aftrekken", "Snijpunt van contouren", "Lijnen aaneenrijgen",
                 "Opsplitsen in lijnstukken", "Dupliceren in rij of raster…",
                 "Raster vastzetten"):
        assert want in labels, f"knop {want} ontbreekt op de werkbalk"
    assert bar.isHidden(), "in 3D is de werkbalk van het werkblad niet te zien"
    win.set_mode("2d")
    assert not bar.isHidden()


def test_werkbalkknoppen_werken_in_het_venster(win):
    win.set_mode("2d")
    sv = win.sheet_view
    win.sheet_tool_actions["circle"].trigger()
    assert sv.tool.name == "circle"
    win.a_sheet_grid.trigger()
    assert not sv.snap_to_grid and not win.a_sheet_grid.isChecked()
    win.a_sheet_round.trigger()
    assert not sv.offset_round
    doc = win.doc
    doc.sheet.add_path([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True, filled=True, src=1)
    doc.sheet.add_path([[5, 0], [15, 0], [15, 10], [5, 10]], closed=True, filled=True, src=2)
    sv.set_selection(list(doc.sheet.paths))
    win.a_sheet_union.trigger()
    assert abs(total_area(doc.sheet.paths) - 150.0) < 1e-6
    assert win.mode == "2d"


def test_tekst_gereedschap_vraagt_om_tekst_als_die_er_nog_niet_is(win, monkeypatch):
    asked = []
    monkeypatch.setattr(win, "ask_sheet_text", lambda: asked.append(1) or False)
    win.set_mode("2d")
    sv = win.sheet_view
    sv.set_tool("text")
    wclick(sv, 0.0, 0.0)
    assert asked, "zonder contouren vraagt het gereedschap eerst om tekst"
    assert sv.doc.sheet.paths == []


def test_hulpmiddelen_staan_aan():
    v = sheet_view()
    assert v.snap_to_grid and v.offset_round
    assert v.join_type_name() == "Afgerond"
    v.set_offset_round(False)
    assert v.join_type_name() == "Scherp"
    assert v.draw_layer() == "Snijden", "een nieuw werkblad tekent op de snijlaag"
    v.doc.sheet.ensure_layer("snij")
    v.doc.sheet.layers["snij"].locked = True
    v.current_layer = "snij"
    assert v.draw_layer() == "0", "op een vergrendelde laag teken je niet"


def test_nieuwe_paden_krijgen_de_actieve_laag():
    v = sheet_view()
    v.doc.sheet.ensure_layer("snij")
    v.current_layer = "snij"
    v.set_tool("rectangle")
    wclick(v, 0.0, 0.0)
    wtype(v, "10;10")
    assert v.doc.sheet.paths[0].layer == "snij"


def test_gebouwde_vormen_zelf():
    loop = rounded_rect_loop([0, 0], [30, 20], 5.0)
    want = 600.0 - (4.0 - math.pi) * 25.0
    assert abs(polygon_area(loop) - want) / want < 0.005
    circle = circle_loop([0, 0], 10.0)
    assert abs(polygon_area(circle) - math.pi * 100.0) / (math.pi * 100.0) < 0.001


# ------------------------------------------------------------------ 7. meta van krommen

def test_meta_cirkel_mee_naar_het_bestand_en_terug(tmp_path):
    from volura import fileio
    v = sheet_view()
    doc = v.doc
    v.set_tool("circle")
    wclick(v, 10.0, 10.0)
    wtype(v, "6")
    circle = doc.sheet.paths[0]
    path = str(tmp_path / "blad.volura")
    fileio.save_document(doc, path)
    back = Document()
    fileio.load_document(back, path)
    assert back.sheet.paths[0].meta == circle.meta
    assert back.sheet.paths[0].meta["kind"] == "circle"
    assert np.allclose(back.sheet.paths[0].pts, circle.pts)


def test_meta_blijft_bij_undo_redo_en_kopieeren():
    v = sheet_view()
    doc = v.doc
    v.set_tool("circle")
    wclick(v, 0.0, 0.0)
    wtype(v, "5")
    meta = dict(doc.sheet.paths[0].meta)
    doc.undo()
    assert doc.sheet.paths == []
    doc.redo()
    assert doc.sheet.paths[0].meta == meta
    v.set_selection(list(doc.sheet.paths))
    v.set_tool("move")
    wclick(v, 0.0, 0.0)
    wclick(v, 30.0, 0.0)
    assert np.allclose(doc.sheet.paths[0].meta["center"], [30.0, 0.0], atol=1e-9)
    assert doc.sheet.paths[0].meta["radius"] == pytest.approx(meta["radius"])


def test_meta_gaat_mee_als_de_tekening_verplaatst_wordt():
    from volura.vector2d import Drawing
    d = Drawing()
    d.add_path(circle_loop([0.0, 0.0], 5.0), closed=True, filled=True,
               meta={"kind": "circle", "center": [0.0, 0.0], "radius": 5.0})
    moved = d.transformed(np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]))
    assert np.allclose(moved.paths[0].meta["center"], [12.0, -3.0])
    assert moved.paths[0].meta["radius"] == pytest.approx(5.0)
    assert d.copy().paths[0].meta == d.paths[0].meta
    assert d.paths[0].meta["center"] == [0.0, 0.0], "het origineel blijft"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and not fn.__code__.co_argcount:
            fn()
            print("ok", name)
