"""Bewerkingslagen voor laser en frees, kerf-/freescorrectie en dogbones.

`volura/cam.py` is de rekenkern zonder Qt: `nesting_depth` zegt hoe diep een contour
in de andere ligt, `compensate` schuift de contouren opzij met de halve kerf of de
freesradius, en `dogbones` geeft een frees ruimte in de binnenhoeken. Daarbovenop:
de standaardlagen van een nieuw werkblad (de kleuren van LightBurn), hun instellingen,
het lagenpaneel in de tray en "Exporteren voor laser/frees".

Run with:  python -m pytest tests
"""
import math
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import cam, dxf, svg  # noqa: E402
from volura.vector2d import (DEFAULT_LAYERS, Drawing, _point_in_poly,  # noqa: E402
                             polygon_area)

_app = QApplication.instance() or QApplication(sys.argv[:1])


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def circle(cx, cy, r, n=180):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def box(pts):
    p = np.asarray(pts, dtype=float)
    return p.min(axis=0), p.max(axis=0)


def span(pts):
    lo, hi = box(pts)
    return hi - lo


def edge_distance(pts, q):
    """Kleinste afstand van punt `q` tot de rand van `pts`."""
    p = np.asarray(pts, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
    t = np.clip(np.sum((q - a) * d, axis=1) / L2, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - q, axis=1)))


def inside_or_on(pts, q, tol=1e-6):
    return _point_in_poly(np.asarray(pts, float), np.asarray(q, float)) or \
        edge_distance(pts, q) <= tol


def sheet_with(layer, *shapes, **params):
    """Een leeg werkblad met de standaardlagen, met de vormen op `layer`."""
    d = Drawing.new_sheet()
    d.layers[layer].params.update(params)
    for pts in shapes:
        d.add_path(pts, closed=True, layer=layer, src=1)
    return d


# ------------------------------------------------------------------ 1. nestingdiepte

def test_nesting_depth_plaat_met_gat_met_eiland():
    d = sheet_with("Snijden", rect(100, 100), circle(50, 50, 20), circle(50, 50, 5))
    assert cam.nesting_depth(d.paths) == [0, 1, 2]


def test_nesting_depth_telt_niet_over_lagen_heen():
    d = Drawing.new_sheet()
    d.add_path(rect(100, 100), closed=True, layer="Snijden", src=1)
    d.add_path(circle(50, 50, 20), closed=True, layer="Boren", src=2)
    assert cam.nesting_depth([p for p in d.paths if p.layer == "Snijden"]) == [0]
    assert cam.nesting_depth([p for p in d.paths if p.layer == "Boren"]) == [0]


def test_nesting_depth_twee_onderdelen_naast_elkaar():
    d = sheet_with("Snijden", rect(20, 20), rect(20, 20, 40.0, 0.0))
    assert cam.nesting_depth(d.paths) == [0, 0]


def test_nesting_depth_open_lijnen_krijgen_nul():
    d = Drawing.new_sheet()
    d.add_path([[0, 0], [10, 0]], layer="Graveren")
    d.add_path(rect(10, 10), closed=True, layer="Graveren", src=1)
    assert cam.nesting_depth(d.paths) == [0, 0]


# ------------------------------------------------------------------ 2. kerfcorrectie

def test_kerf_buitenrand_naar_buiten_en_gat_kleiner():
    d = sheet_with("Snijden", rect(50, 50), circle(25, 25, 10), kerf=0.2)
    outer, hole = cam.compensate(d).paths
    lo, hi = box(outer.pts)
    assert lo == pytest.approx((-0.1, -0.1), abs=0.01)
    assert span(outer.pts) == pytest.approx((50.2, 50.2), abs=0.01)
    r = np.linalg.norm(np.asarray(hole.pts, float) - np.array([25.0, 25.0]), axis=1)
    assert np.max(np.abs(r - 9.9)) < 0.01, "een gat wordt kleiner getekend"


def test_kerf_eiland_in_een_gat_groeit_weer():
    d = sheet_with("Snijden", rect(100, 100), circle(50, 50, 20), circle(50, 50, 5), kerf=0.2)
    outer, gat, eiland = cam.compensate(d).paths
    assert span(outer.pts) == pytest.approx((100.2, 100.2), abs=0.01)
    r = np.linalg.norm(np.asarray(gat.pts, float) - np.array([50.0, 50.0]), axis=1)
    assert np.max(np.abs(r - 19.9)) < 0.01
    r = np.linalg.norm(np.asarray(eiland.pts, float) - np.array([50.0, 50.0]), axis=1)
    assert np.max(np.abs(r - 5.1)) < 0.01


def test_graveren_en_boren_blijven_ongewijzigd():
    d = Drawing.new_sheet()
    d.add_path(rect(20, 10), closed=True, layer="Graveren", src=1)
    d.add_path(circle(50, 5, 2), closed=True, layer="Boren", src=2)
    d.add_path([[0, 40], [10, 40]], layer="Snijden")
    out = cam.compensate(d)
    for before, after in zip(d.paths, out.paths):
        assert np.array_equal(np.asarray(before.pts, float), np.asarray(after.pts, float))
        assert after.layer == before.layer


def test_correctie_laat_de_tekening_zelf_metruste():
    d = sheet_with("Snijden", rect(50, 50), kerf=0.2)
    out = cam.compensate(d)
    assert np.array_equal(np.asarray(d.paths[0].pts, float), rect(50, 50))
    assert out.layers["Snijden"].params["kerf"] == 0.2


def test_gat_kleiner_dan_de_kerf_verdwijnt():
    d = sheet_with("Snijden", rect(20, 20), circle(10, 10, 0.04), kerf=0.2)
    out = cam.compensate(d)
    assert len(out.paths) == 1
    assert any("te klein" in w for w in out.warnings)


# ------------------------------------------------------------------ 3. frezen

def test_freesradius_buiten_en_binnen():
    out = cam.compensate(sheet_with("Frezen buiten", rect(20, 20), tool_d=3.0))
    assert span(out.paths[0].pts) == pytest.approx((23.0, 23.0), abs=1e-6)
    out = cam.compensate(sheet_with("Frezen binnen", rect(20, 20), tool_d=3.0))
    assert span(out.paths[0].pts) == pytest.approx((17.0, 17.0), abs=1e-6)


def test_uitkameren_blijft_binnen_hetgebied_en_vult_het():
    d = sheet_with("Uitkameren", rect(20, 20), tool_d=3.0, stepover=50.0)
    paths = cam.compensate(d).paths
    assert len(paths) >= 4, "concentrische paden tot het gebied op is"
    for p in paths:
        lo, hi = box(p.pts)
        assert lo[0] >= 1.5 - 1e-9 and lo[1] >= 1.5 - 1e-9
        assert hi[0] <= 18.5 + 1e-9 and hi[1] <= 18.5 + 1e-9
    # een ronde frees vult een scherpe hoek nooit: tot r*wortel(2) is dat de grens
    reach = 1.5 * math.sqrt(2.0) + 0.01
    steps = np.arange(0.1, 19.95, 0.4)
    for x in steps:
        for y in steps:
            q = np.array([x, y])
            assert min(edge_distance(p.pts, q) for p in paths) <= reach, (x, y)


def test_uitkameren_met_gaten_in_het_gebied():
    d = sheet_with("Uitkameren", rect(30, 20), circle(15, 10, 4), tool_d=3.0, stepover=50.0)
    paths = cam.compensate(d).paths
    assert paths
    for p in paths:                      # niets van de paden mag in het gat komen
        for q in p.pts:
            assert np.hypot(q[0] - 15.0, q[1] - 10.0) >= 4.0 - 1e-6


# ------------------------------------------------------------------ 4. dogbones

def test_dogbones_vier_cirkels_in_de_hoeken():
    d = sheet_with("Frezen binnen", rect(30, 20), tool_d=6.0)
    made = cam.dogbones(d.paths, 6.0, "dogbone")
    assert len(made) == 1 and made[0].closed and made[0].layer == "Frezen binnen"
    loop = made[0].pts
    assert len(cam.corner_spots(rect(30, 20), 6.0, "dogbone")) == 4
    # elke hoek levert een cirkeldeel buiten het gat: r^2 (pi/2 - 1)
    want = 30.0 * 20.0 + 4.0 * 3.0 ** 2 * (math.pi / 2.0 - 1.0)
    assert abs(abs(polygon_area(loop)) - want) < 0.2
    for c in ((0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)):
        assert inside_or_on(loop, c), f"hoek {c} ligt buiten de nieuwe contour"


def test_t_bone_cirkels_langs_de_langste_rand():
    d = sheet_with("Frezen binnen", rect(30, 20), tool_d=6.0)
    loop = cam.dogbones(d.paths, 6.0, "tbone")[0].pts
    lo, hi = box(loop)
    assert hi[0] - lo[0] == pytest.approx(30.0, abs=1e-6), "geen uitsparing langs de korte rand"
    assert hi[1] - lo[1] == pytest.approx(26.0, abs=0.02), "wel 3 mm buiten de lange rand"
    want = 30.0 * 20.0 + 4.0 * (math.pi * 3.0 ** 2 / 2.0)
    assert abs(abs(polygon_area(loop)) - want) < 0.2
    for c in ((0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)):
        assert inside_or_on(loop, c)


def test_gestrekte_hoeken_krijgen_geen_uitsparing():
    assert cam.corner_spots(circle(0.0, 0.0, 20.0, n=64), 6.0, "dogbone") == []
    assert cam.corner_spots(circle(0.0, 0.0, 3.0, n=40), 6.0, "tbone") == []
    # in een L-vormige uitsparing komt de frees bij vijf hoeken niet in de zesde
    L = np.array([[0, 0], [40, 0], [40, 40], [20, 40], [20, 20], [0, 20]], float)
    assert len(cam.corner_spots(L, 6.0, "dogbone")) == 5


def test_hoekuitsparing_in_de_correctie():
    kaal = cam.compensate(sheet_with("Frezen binnen", rect(30, 20), tool_d=6.0))
    met = cam.compensate(sheet_with("Frezen binnen", rect(30, 20), tool_d=6.0,
                                    corner="dogbone"))
    assert abs(polygon_area(met.paths[0].pts)) > abs(polygon_area(kaal.paths[0].pts)) + 1.0


# ------------------------------------------------------------------ 5. standaardlagen

KLEUREN = {"Snijden": "#ff0000", "Graveren": "#0000ff", "Rasteren": "#000000",
           "Frezen buiten": "#00e000", "Frezen binnen": "#ff8000", "Uitkameren": "#00e0e0",
           "Boren": "#ff00ff", "Hulplijnen": "#a0a0a0"}
BEWERKINGEN = {"Snijden": "cut", "Graveren": "engrave", "Rasteren": "raster",
               "Frezen buiten": "mill_outside", "Frezen binnen": "mill_inside",
               "Uitkameren": "pocket", "Boren": "drill", "Hulplijnen": "none"}


def test_nieuwe_werkbladen_hebben_de_standaardlagen():
    d = Drawing.new_sheet()
    assert [name for name, _color, _op in DEFAULT_LAYERS] == list(KLEUREN)
    assert list(d.layers) == ["0"] + list(KLEUREN)
    for name, color in KLEUREN.items():
        assert d.layers[name].color == color, name
        assert d.layers[name].op == BEWERKINGEN[name], name
    assert d.layers["0"].visible and d.layers["0"].op == "cut"
    assert d.layers["Hulplijnen"].op == "none"


def test_instellingen_per_laag():
    d = Drawing.new_sheet()
    for name in ("Snijden", "Graveren", "Rasteren", "Boren"):
        p = d.layers[name].params
        assert p["kerf"] == pytest.approx(0.15) and p["passes"] == 1
        assert "power" in p and "speed" in p
    for name in ("Frezen buiten", "Frezen binnen", "Uitkameren"):
        p = d.layers[name].params
        assert p["tool_d"] == pytest.approx(3.175)
        assert {"depth", "step_depth", "stepover", "corner"} <= set(p)
        assert p["corner"] == "none"
    assert d.layers["Hulplijnen"].params == {}


def test_nieuw_document_krijgt_ook_de_standaardlagen():
    from volura.model import Document
    doc = Document()
    assert list(doc.sheet.layers) == ["0"] + list(KLEUREN)
    assert doc.sheet.paths == []


# ------------------------------------------------------------------ 6. export

def test_svg_export_en_teruglezen_geeft_zelfde_lagen(tmp_path):
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 150.0
    d.add_path(rect(30, 20), closed=True, layer="Snijden", src=1)
    d.add_path(circle(60.0, 10.0, 5.0), closed=True, layer="Boren", src=2)
    d.add_path(rect(20, 10, 0.0, 40.0), closed=True, layer="Rasteren", filled=True, src=3)
    d.layers["Snijden"].params.update(power=42.0, speed=17.5, passes=2, kerf=0.18)
    d.layers["Uitkameren"].params.update(tool_d=4.0, depth=2.5, step_depth=0.7,
                                         stepover=45.0, corner="dogbone")
    text = svg.write_svg(d, fill_layers=("Rasteren",))
    back = svg.read_svg(text.encode())
    assert list(back.layers) == list(d.layers)
    for name, lay in d.layers.items():
        got = back.layers[name]
        assert (got.color, got.op) == (lay.color, lay.op), name
        assert got.params == lay.params, name
    assert 'data-volura-op="cut"' in text
    assert 'data-volura-param-kerf="0.18"' in text
    assert 'data-volura-param-corner="dogbone"' in text


def test_rasterlaag_gaat_gevuld_de_rest_als_haarlijn():
    d = Drawing.new_sheet()
    d.add_path(rect(20, 10), closed=True, layer="Rasteren", filled=True, src=1)
    d.add_path(rect(20, 10, 0.0, 30.0), closed=True, layer="Snijden", src=2)
    text = svg.write_svg(d, fill_layers=("Rasteren",))
    groep = {m.group(1): m.group(0)
            for m in re.finditer(r'<g [^>]*inkscape:label="([^"]+)"[^>]*>', text)}
    assert 'fill="#000000"' in groep["Rasteren"]
    assert 'fill-rule="evenodd"' in groep["Rasteren"]
    assert 'fill="none"' in groep["Snijden"]
    assert 'stroke="#ff0000"' in groep["Snijden"]


def test_dxf_export_met_lagen_en_dichtstbijzijnde_aci_kleur(tmp_path):
    d = Drawing.new_sheet()
    d.add_path(rect(30, 20), closed=True, layer="Snijden", src=1)
    d.add_path(circle(60.0, 10.0, 5.0), closed=True, layer="Boren", src=2)
    path = str(tmp_path / "plaat.dxf")
    dxf.write_dxf(d, path, layers=["0", "Snijden", "Boren"])
    back = dxf.read_dxf(path)
    assert {p.layer for p in back.paths} == {"Snijden", "Boren"}
    assert back.layers["Snijden"].color == "#ff0000"
    assert back.layers["Boren"].color == "#ff00ff"


# ------------------------------------------------------------------ 7. het venster

@pytest.fixture()
def win():
    from PySide6.QtCore import QCoreApplication

    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(1200, 800)
    yield w
    w.doc.dirty = False
    w.close()


def test_lagenpaneel_in_de_tray(win):
    assert "Lagen" in win.tray.extra_titles and "Lagen" not in win.tray.panel_titles
    assert win.tray.is_open("Lagen") is False, "in 3D is het lagenpaneel samengeklapt"
    win.set_mode("2d")
    assert win.tray.is_open("Lagen"), "in 2D gaat het lagenpaneel open"
    table = win.layers_panel.table
    names = list(win.doc.sheet.layers)
    assert table.rowCount() == len(names)
    assert [table.item(r, 3).text() for r in range(table.rowCount())] == names
    win.set_mode("3d")
    assert not win.tray.is_open("Lagen")


def test_lagenpaneel_toont_bewerking_en_kleur(win):
    table = win.layers_panel.table
    row = list(win.doc.sheet.layers).index("Uitkameren")
    assert table.cellWidget(row, 4).currentData() == "pocket"
    assert "Actieve laag" in win.layers_panel.info.text()
    row = list(win.doc.sheet.layers).index("Snijden")
    assert table.cellWidget(row, 4).currentData() == "cut"


def test_oogje_verbergt_de_laag(win):
    win.set_mode("2d")
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Graveren", src=1)
    win.layers_panel.refresh()
    row = list(win.doc.sheet.layers).index("Graveren")
    win.layers_panel.table.cellWidget(row, 0).click()
    assert win.doc.sheet.layers["Graveren"].visible is False
    assert "verborgen" in win.hint.text()
    win.doc.undo()
    assert win.doc.sheet.layers["Graveren"].visible is True, "een bewerking is één undo-stap"


def test_slotje_houdt_de_laag_op_slot(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="Boren", src=1)
    win.layers_panel.refresh()
    row = list(sheet.layers).index("Boren")
    spot = win.sheet_view.to_screen(path.pts[0])
    assert win.sheet_view.path_at(spot.x(), spot.y()) is path
    win.layers_panel.table.cellWidget(row, 1).click()
    assert sheet.layers["Boren"].locked
    assert win.sheet_view.path_at(spot.x(), spot.y()) is None, "op slot kun je niet selecteren"
    win.layers_panel.table.cellWidget(row, 1).click()
    assert not sheet.layers["Boren"].locked


def test_dubbelklik_hernoemt_de_laag(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    sheet.add_path(rect(10, 10), closed=True, layer="Graveren", src=1)
    win.layers_panel.refresh()
    row = list(sheet.layers).index("Graveren")
    win.layers_panel.table.item(row, 3).setText("Graveren ondiep")
    win.layers_panel.on_rename(win.layers_panel.table.item(row, 3))
    assert "Graveren ondiep" in sheet.layers and "Graveren" not in sheet.layers
    assert sheet.paths[0].layer == "Graveren ondiep", "de paden volgen de naam"


def test_laag_toevoegen_en_verwijderen(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    win.layers_panel.add_layer()
    assert len(sheet.layers) == 10
    new = list(sheet.layers)[-1]
    win.layers_panel.refresh()
    win.layers_panel.table.selectRow(list(sheet.layers).index(new))
    win.layers_panel.remove_layer()               # een lege laag vraagt niets
    assert new not in sheet.layers
    assert list(sheet.layers)[0] == "0"


def test_verwijderen_vraagt_wat_er_met_de_paden_gebeurt():
    d = Drawing.new_sheet()
    d.add_path(rect(5, 5), closed=True, layer="Snijden", src=1)
    assert d.remove_layer("Snijden")
    assert "Snijden" not in d.layers and d.paths[0].layer == "0"
    d.add_path(rect(5, 5), closed=True, layer="Boren", src=1)
    assert d.remove_layer("Boren", move_paths=False)
    assert "Boren" not in d.layers and all(p.layer != "Boren" for p in d.paths)


def test_naar_laag_op_de_werkbalk(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="0", src=1)
    win.sheet_view.set_selection([path])
    assert win.a_sheet_to_layer.text().replace("&", "") == "Naar laag"
    assert win.move_selection_to_layer()
    assert sheet.paths[0].layer == "Snijden", "de actieve laag is de snijlaag"
    win.doc.undo()
    assert sheet.paths[0].layer == "0"
    win.sheet_view.set_selection([sheet.paths[0]])
    assert win.move_selection_to_layer("Boren")
    assert sheet.paths[0].layer == "Boren"


def test_naar_laag_in_het_rechtsklikmenu(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="0", src=1)
    win.sheet_view.set_selection([path])
    menu = win.sheet_view.build_context_menu()
    sub = next(a.menu() for a in menu.actions() if a.menu() is not None
               and a.menu().title() == "Naar laag")
    assert any("Uitkameren" in a.text() for a in sub.actions())
    for a in sub.actions():
        if "Uitkameren" in a.text():
            a.trigger()
    assert sheet.paths[0].layer == "Uitkameren"


def test_nieuwe_paden_komen_op_de_snijlaag(win):
    from PySide6.QtCore import QPointF, QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    win.set_mode("2d")
    v = win.sheet_view
    v.set_tool("rectangle")
    p = QPointF(200.0, 200.0)
    v.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton,
                                  Qt.LeftButton, Qt.NoModifier))
    v.submit_vcb("10;10")
    assert v.doc.sheet.paths and v.doc.sheet.paths[-1].layer == "Snijden"


def test_export_keuzes_standaard(win):
    win.set_mode("2d")
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Snijden", src=1)
    opt = win.default_sheet_export_options()
    assert "Snijden" in opt["layers"] and "Hulplijnen" not in opt["layers"]
    assert opt["compensate"] is False, "lasersoftware doet de kerf zelf"
    win.doc.sheet.add_path(rect(10, 10, 0.0, 30.0), closed=True, layer="Uitkameren", src=2)
    assert win.default_sheet_export_options()["compensate"] is True, "frees heeft een diameter"


def test_export_dialoog_toont_de_lagen(win):
    from volura.app import SheetExportDialog
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Snijden", src=1)
    dlg = SheetExportDialog(win, "svg")
    opt = dlg.options()
    assert "Snijden" in opt["layers"] and "Hulplijnen" not in opt["layers"]
    assert opt["compensate"] is False
    assert dlg.preview.drawing is not None and dlg.preview.drawing.paths
    dlg.deleteLater()


def test_export_sheet_met_correctie_en_per_laag(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(50, 50), closed=True, layer="Snijden", src=1)
    sheet.add_path(circle(25.0, 25.0, 10.0), closed=True, layer="Snijden", src=1)
    sheet.add_path(rect(10, 10, 80.0, 0.0), closed=True, layer="Rasteren", filled=True, src=2)
    sheet.layers["Snijden"].params["kerf"] = 0.2
    svg_path = str(tmp_path / "plaat.svg")
    got = win.export_sheet("svg", path=svg_path,
                           options={"layers": ["Snijden", "Rasteren"],
                                    "compensate": True, "per_layer": False})
    assert got == svg_path
    text = open(svg_path, encoding="utf-8").read()
    back = svg.read_svg(svg_path)
    snij = [p for p in back.paths if p.layer == "Snijden"]
    assert len(snij) == 2
    buiten = max(snij, key=lambda p: abs(polygon_area(p.pts)))
    assert span(buiten.pts) == pytest.approx((50.2, 50.2), abs=0.01), "met kerfcorrectie"
    gat = min(snij, key=lambda p: abs(polygon_area(p.pts)))
    r = np.linalg.norm(np.asarray(gat.pts, float) - np.array([25.0, 25.0]), axis=1)
    assert np.max(np.abs(r - 9.9)) < 0.01
    assert 'fill="#000000"' in text, "rasterlaag gevuld"
    assert 'data-volura-param-kerf="0.2"' in text, "instellingen gaan mee"
    assert "contouren geëxporteerd naar" in win.hint.text()

    per = str(tmp_path / "laag.svg")
    win.export_sheet("svg", path=per,
                     options={"layers": ["Snijden", "Rasteren"],
                              "compensate": False, "per_layer": True})
    snij_pad = str(tmp_path / "laag_Snijden.svg")
    assert os.path.exists(snij_pad) and os.path.exists(str(tmp_path / "laag_Rasteren.svg"))
    snij = svg.read_svg(snij_pad)
    assert {p.layer for p in snij.paths} == {"Snijden"}
    buiten = max((p for p in snij.paths if p.closed), key=lambda p: abs(polygon_area(p.pts)))
    assert span(buiten.pts) == pytest.approx((50.0, 50.0), abs=0.01), "nu zonder correctie"


def test_export_sheet_zonder_lagen_geeft_melding(win, monkeypatch):
    import volura.app as app_module

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
    win.set_mode("2d")
    assert win.export_sheet("svg", path="x.svg",
                            options={"layers": [], "compensate": False,
                                     "per_layer": False}) is None
    assert box.messages and "bewerking" in box.messages[0]
    assert "Er is niets geëxporteerd" in win.hint.text()
