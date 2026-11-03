"""Machining layers for laser and mill, kerf/cutter compensation and dogbones.

`volura/cam.py` is the computing core without Qt: `nesting_depth` says how deep a contour
lies inside the others, `compensate` offsets the contours by half the kerf or the
cutter radius, and `dogbones` gives a cutter room in the inside corners. On top of that:
the default layers of a new worksheet (the colours of LightBurn), their settings,
the Layers panel in the tray and "Export for laser/mill".

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
    """Smallest distance from point `q` to the edge of `pts`."""
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
    """An empty worksheet with the default layers, with the shapes on `layer`."""
    d = Drawing.new_sheet()
    d.layers[layer].params.update(params)
    for pts in shapes:
        d.add_path(pts, closed=True, layer=layer, src=1)
    return d


# ------------------------------------------------------------------ 1. nesting depth

def test_nesting_depth_sheet_with_hole_with_island():
    d = sheet_with("Cut", rect(100, 100), circle(50, 50, 20), circle(50, 50, 5))
    assert cam.nesting_depth(d.paths) == [0, 1, 2]


def test_nesting_depth_does_not_count_across_layers():
    d = Drawing.new_sheet()
    d.add_path(rect(100, 100), closed=True, layer="Cut", src=1)
    d.add_path(circle(50, 50, 20), closed=True, layer="Drill", src=2)
    assert cam.nesting_depth([p for p in d.paths if p.layer == "Cut"]) == [0]
    assert cam.nesting_depth([p for p in d.paths if p.layer == "Drill"]) == [0]


def test_nesting_depth_two_parts_side_by_side():
    d = sheet_with("Cut", rect(20, 20), rect(20, 20, 40.0, 0.0))
    assert cam.nesting_depth(d.paths) == [0, 0]


def test_nesting_depth_open_lines_get_zero():
    d = Drawing.new_sheet()
    d.add_path([[0, 0], [10, 0]], layer="Engrave")
    d.add_path(rect(10, 10), closed=True, layer="Engrave", src=1)
    assert cam.nesting_depth(d.paths) == [0, 0]


# ------------------------------------------------------------------ 2. kerf compensation

def test_kerf_outer_edge_outward_and_hole_smaller():
    d = sheet_with("Cut", rect(50, 50), circle(25, 25, 10), kerf=0.2)
    outer, hole = cam.compensate(d).paths
    lo, hi = box(outer.pts)
    assert lo == pytest.approx((-0.1, -0.1), abs=0.01)
    assert span(outer.pts) == pytest.approx((50.2, 50.2), abs=0.01)
    r = np.linalg.norm(np.asarray(hole.pts, float) - np.array([25.0, 25.0]), axis=1)
    assert np.max(np.abs(r - 9.9)) < 0.01, "a hole is drawn smaller"


def test_kerf_island_in_a_hole_grows_again():
    d = sheet_with("Cut", rect(100, 100), circle(50, 50, 20), circle(50, 50, 5), kerf=0.2)
    outer, hole, island = cam.compensate(d).paths
    assert span(outer.pts) == pytest.approx((100.2, 100.2), abs=0.01)
    r = np.linalg.norm(np.asarray(hole.pts, float) - np.array([50.0, 50.0]), axis=1)
    assert np.max(np.abs(r - 19.9)) < 0.01
    r = np.linalg.norm(np.asarray(island.pts, float) - np.array([50.0, 50.0]), axis=1)
    assert np.max(np.abs(r - 5.1)) < 0.01


def test_engrave_and_drill_stay_unchanged():
    d = Drawing.new_sheet()
    d.add_path(rect(20, 10), closed=True, layer="Engrave", src=1)
    d.add_path(circle(50, 5, 2), closed=True, layer="Drill", src=2)
    d.add_path([[0, 40], [10, 40]], layer="Cut")
    out = cam.compensate(d)
    for before, after in zip(d.paths, out.paths):
        assert np.array_equal(np.asarray(before.pts, float), np.asarray(after.pts, float))
        assert after.layer == before.layer


def test_compensation_leaves_the_drawing_itself_alone():
    d = sheet_with("Cut", rect(50, 50), kerf=0.2)
    out = cam.compensate(d)
    assert np.array_equal(np.asarray(d.paths[0].pts, float), rect(50, 50))
    assert out.layers["Cut"].params["kerf"] == 0.2


def test_hole_smaller_than_the_kerf_disappears():
    d = sheet_with("Cut", rect(20, 20), circle(10, 10, 0.04), kerf=0.2)
    out = cam.compensate(d)
    assert len(out.paths) == 1
    assert any("too small" in w for w in out.warnings)


# ------------------------------------------------------------------ 3. milling

def test_cutter_radius_outside_and_inside():
    out = cam.compensate(sheet_with("Mill outside", rect(20, 20), tool_d=3.0))
    assert span(out.paths[0].pts) == pytest.approx((23.0, 23.0), abs=1e-6)
    out = cam.compensate(sheet_with("Mill inside", rect(20, 20), tool_d=3.0))
    assert span(out.paths[0].pts) == pytest.approx((17.0, 17.0), abs=1e-6)


def test_pocket_stays_inside_the_area_and_fills_it():
    d = sheet_with("Pocket", rect(20, 20), tool_d=3.0, stepover=50.0)
    paths = cam.compensate(d).paths
    assert len(paths) >= 4, "concentric paths until the area is used up"
    for p in paths:
        lo, hi = box(p.pts)
        assert lo[0] >= 1.5 - 1e-9 and lo[1] >= 1.5 - 1e-9
        assert hi[0] <= 18.5 + 1e-9 and hi[1] <= 18.5 + 1e-9
    # a round cutter never fills a sharp corner: up to r*sqrt(2) is the limit
    reach = 1.5 * math.sqrt(2.0) + 0.01
    steps = np.arange(0.1, 19.95, 0.4)
    for x in steps:
        for y in steps:
            q = np.array([x, y])
            assert min(edge_distance(p.pts, q) for p in paths) <= reach, (x, y)


def test_pocket_with_holes_in_the_area():
    d = sheet_with("Pocket", rect(30, 20), circle(15, 10, 4), tool_d=3.0, stepover=50.0)
    paths = cam.compensate(d).paths
    assert paths
    for p in paths:                      # none of the paths may enter the hole
        for q in p.pts:
            assert np.hypot(q[0] - 15.0, q[1] - 10.0) >= 4.0 - 1e-6


# ------------------------------------------------------------------ 4. dogbones

def test_dogbones_four_circles_in_the_corners():
    d = sheet_with("Mill inside", rect(30, 20), tool_d=6.0)
    made = cam.dogbones(d.paths, 6.0, "dogbone")
    assert len(made) == 1 and made[0].closed and made[0].layer == "Mill inside"
    loop = made[0].pts
    assert len(cam.corner_spots(rect(30, 20), 6.0, "dogbone")) == 4
    # every corner adds a circle segment outside the hole: r^2 (pi/2 - 1)
    want = 30.0 * 20.0 + 4.0 * 3.0 ** 2 * (math.pi / 2.0 - 1.0)
    assert abs(abs(polygon_area(loop)) - want) < 0.2
    for c in ((0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)):
        assert inside_or_on(loop, c), f"corner {c} lies outside the new contour"


def test_t_bone_circles_along_the_longest_edge():
    d = sheet_with("Mill inside", rect(30, 20), tool_d=6.0)
    loop = cam.dogbones(d.paths, 6.0, "tbone")[0].pts
    lo, hi = box(loop)
    assert hi[0] - lo[0] == pytest.approx(30.0, abs=1e-6), "no relief along the short edge"
    assert hi[1] - lo[1] == pytest.approx(26.0, abs=0.02), "but 3 mm beyond the long edge"
    want = 30.0 * 20.0 + 4.0 * (math.pi * 3.0 ** 2 / 2.0)
    assert abs(abs(polygon_area(loop)) - want) < 0.2
    for c in ((0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)):
        assert inside_or_on(loop, c)


def test_obtuse_corners_get_no_relief():
    assert cam.corner_spots(circle(0.0, 0.0, 20.0, n=64), 6.0, "dogbone") == []
    assert cam.corner_spots(circle(0.0, 0.0, 3.0, n=40), 6.0, "tbone") == []
    # in an L-shaped pocket the cutter misses five corners, not the sixth
    L = np.array([[0, 0], [40, 0], [40, 40], [20, 40], [20, 20], [0, 20]], float)
    assert len(cam.corner_spots(L, 6.0, "dogbone")) == 5


def test_corner_relief_in_the_compensation():
    bare = cam.compensate(sheet_with("Mill inside", rect(30, 20), tool_d=6.0))
    relieved = cam.compensate(sheet_with("Mill inside", rect(30, 20), tool_d=6.0,
                                         corner="dogbone"))
    assert abs(polygon_area(relieved.paths[0].pts)) > abs(polygon_area(bare.paths[0].pts)) + 1.0


# ------------------------------------------------------------------ 5. default layers

COLORS = {"Cut": "#ff0000", "Engrave": "#0000ff", "Raster": "#000000",
          "Mill outside": "#00e000", "Mill inside": "#ff8000", "Pocket": "#00e0e0",
          "Drill": "#ff00ff", "Guides": "#a0a0a0"}
OPERATIONS = {"Cut": "cut", "Engrave": "engrave", "Raster": "raster",
              "Mill outside": "mill_outside", "Mill inside": "mill_inside",
              "Pocket": "pocket", "Drill": "drill", "Guides": "none"}
#: The default layer names of Volura 0.1.0, with their operations.
DUTCH = {"Snijden": "cut", "Graveren": "engrave", "Rasteren": "raster",
         "Frezen buiten": "mill_outside", "Frezen binnen": "mill_inside",
         "Uitkameren": "pocket", "Boren": "drill", "Hulplijnen": "none"}


def test_new_sheets_have_the_default_layers():
    d = Drawing.new_sheet()
    assert [name for name, _color, _op in DEFAULT_LAYERS] == list(COLORS)
    assert list(d.layers) == ["0"] + list(COLORS)
    for name, color in COLORS.items():
        assert d.layers[name].color == color, name
        assert d.layers[name].op == OPERATIONS[name], name
    assert d.layers["0"].visible and d.layers["0"].op == "cut"
    assert d.layers["Guides"].op == "none"


def test_old_file_with_dutch_layer_names_keeps_its_operations():
    """A sheet from Volura 0.1.0 has Dutch default layer names; they stay valid."""
    old = Drawing.from_json({
        "layers": {"0": {"name": "0", "op": "cut"},
                   "Snijden": {"name": "Snijden", "color": "#ff0000", "op": "cut"},
                   "Uitkameren": {"name": "Uitkameren", "color": "#00e0e0", "op": "pocket"}},
        "paths": [{"pts": [[0, 0], [10, 0], [10, 10]], "closed": True, "layer": name}
                  for name in DUTCH]})
    for name, op in DUTCH.items():
        assert old.layers[name].op == op, name
        assert old.default_layer(op) == name, name
    assert old.layers["Boren"].color == "#ff00ff", "a Dutch name gets the default colour"
    assert "Cut" not in old.layers, "an old sheet does not get the English layers"


def test_settings_per_layer():
    d = Drawing.new_sheet()
    for name in ("Cut", "Engrave", "Raster", "Drill"):
        p = d.layers[name].params
        assert p["kerf"] == pytest.approx(0.15) and p["passes"] == 1
        assert "power" in p and "speed" in p
    for name in ("Mill outside", "Mill inside", "Pocket"):
        p = d.layers[name].params
        assert p["tool_d"] == pytest.approx(3.175)
        assert {"depth", "step_depth", "stepover", "corner"} <= set(p)
        assert p["corner"] == "none"
    assert d.layers["Guides"].params == {}


def test_new_document_also_gets_the_default_layers():
    from volura.model import Document
    doc = Document()
    assert list(doc.sheet.layers) == ["0"] + list(COLORS)
    assert doc.sheet.paths == []


# ------------------------------------------------------------------ 6. export

def test_svg_export_and_reading_back_gives_same_layers(tmp_path):
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 150.0
    d.add_path(rect(30, 20), closed=True, layer="Cut", src=1)
    d.add_path(circle(60.0, 10.0, 5.0), closed=True, layer="Drill", src=2)
    d.add_path(rect(20, 10, 0.0, 40.0), closed=True, layer="Raster", filled=True, src=3)
    d.layers["Cut"].params.update(power=42.0, speed=17.5, passes=2, kerf=0.18)
    d.layers["Pocket"].params.update(tool_d=4.0, depth=2.5, step_depth=0.7,
                                     stepover=45.0, corner="dogbone")
    text = svg.write_svg(d, fill_layers=("Raster",))
    back = svg.read_svg(text.encode())
    assert list(back.layers) == list(d.layers)
    for name, lay in d.layers.items():
        got = back.layers[name]
        assert (got.color, got.op) == (lay.color, lay.op), name
        assert got.params == lay.params, name
    assert 'data-volura-op="cut"' in text
    assert 'data-volura-param-kerf="0.18"' in text
    assert 'data-volura-param-corner="dogbone"' in text


def test_raster_layer_goes_filled_the_rest_as_hairline():
    d = Drawing.new_sheet()
    d.add_path(rect(20, 10), closed=True, layer="Raster", filled=True, src=1)
    d.add_path(rect(20, 10, 0.0, 30.0), closed=True, layer="Cut", src=2)
    text = svg.write_svg(d, fill_layers=("Raster",))
    group = {m.group(1): m.group(0)
             for m in re.finditer(r'<g [^>]*inkscape:label="([^"]+)"[^>]*>', text)}
    assert 'fill="#000000"' in group["Raster"]
    assert 'fill-rule="evenodd"' in group["Raster"]
    assert 'fill="none"' in group["Cut"]
    assert 'stroke="#ff0000"' in group["Cut"]


def test_dxf_export_with_layers_and_nearest_aci_color(tmp_path):
    d = Drawing.new_sheet()
    d.add_path(rect(30, 20), closed=True, layer="Cut", src=1)
    d.add_path(circle(60.0, 10.0, 5.0), closed=True, layer="Drill", src=2)
    path = str(tmp_path / "sheet.dxf")
    dxf.write_dxf(d, path, layers=["0", "Cut", "Drill"])
    back = dxf.read_dxf(path)
    assert {p.layer for p in back.paths} == {"Cut", "Drill"}
    assert back.layers["Cut"].color == "#ff0000"
    assert back.layers["Drill"].color == "#ff00ff"


# ------------------------------------------------------------------ 7. the window

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


def test_layers_panel_in_the_tray(win):
    assert "Layers" in win.tray.extra_titles and "Layers" not in win.tray.panel_titles
    assert win.tray.is_open("Layers") is False, "in 3D the Layers panel is collapsed"
    win.set_mode("2d")
    assert win.tray.is_open("Layers"), "in 2D the Layers panel opens"
    table = win.layers_panel.table
    names = list(win.doc.sheet.layers)
    assert table.rowCount() == len(names)
    assert [table.item(r, 3).text() for r in range(table.rowCount())] == names
    win.set_mode("3d")
    assert not win.tray.is_open("Layers")


def test_layers_panel_shows_operation_and_color(win):
    table = win.layers_panel.table
    row = list(win.doc.sheet.layers).index("Pocket")
    assert table.cellWidget(row, 4).currentData() == "pocket"
    assert "Active layer" in win.layers_panel.info.text()
    row = list(win.doc.sheet.layers).index("Cut")
    assert table.cellWidget(row, 4).currentData() == "cut"


def test_eye_hides_the_layer(win):
    win.set_mode("2d")
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Engrave", src=1)
    win.layers_panel.refresh()
    row = list(win.doc.sheet.layers).index("Engrave")
    win.layers_panel.table.cellWidget(row, 0).click()
    assert win.doc.sheet.layers["Engrave"].visible is False
    assert "hidden" in win.hint.text()
    win.doc.undo()
    assert win.doc.sheet.layers["Engrave"].visible is True, "a change is one undo step"


def test_lock_keeps_the_layer_locked(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="Drill", src=1)
    win.layers_panel.refresh()
    row = list(sheet.layers).index("Drill")
    spot = win.sheet_view.to_screen(path.pts[0])
    assert win.sheet_view.path_at(spot.x(), spot.y()) is path
    win.layers_panel.table.cellWidget(row, 1).click()
    assert sheet.layers["Drill"].locked
    assert win.sheet_view.path_at(spot.x(), spot.y()) is None, "locked cannot be selected"
    win.layers_panel.table.cellWidget(row, 1).click()
    assert not sheet.layers["Drill"].locked


def test_double_click_renames_the_layer(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    sheet.add_path(rect(10, 10), closed=True, layer="Engrave", src=1)
    win.layers_panel.refresh()
    row = list(sheet.layers).index("Engrave")
    win.layers_panel.table.item(row, 3).setText("Engrave shallow")
    win.layers_panel.on_rename(win.layers_panel.table.item(row, 3))
    assert "Engrave shallow" in sheet.layers and "Engrave" not in sheet.layers
    assert sheet.paths[0].layer == "Engrave shallow", "the paths follow the name"


def test_add_and_remove_layer(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    win.layers_panel.add_layer()
    assert len(sheet.layers) == 10
    new = list(sheet.layers)[-1]
    win.layers_panel.refresh()
    win.layers_panel.table.selectRow(list(sheet.layers).index(new))
    win.layers_panel.remove_layer()               # an empty layer asks nothing
    assert new not in sheet.layers
    assert list(sheet.layers)[0] == "0"


def test_removing_asks_what_happens_to_the_paths():
    d = Drawing.new_sheet()
    d.add_path(rect(5, 5), closed=True, layer="Cut", src=1)
    assert d.remove_layer("Cut")
    assert "Cut" not in d.layers and d.paths[0].layer == "0"
    d.add_path(rect(5, 5), closed=True, layer="Drill", src=1)
    assert d.remove_layer("Drill", move_paths=False)
    assert "Drill" not in d.layers and all(p.layer != "Drill" for p in d.paths)


def test_move_to_layer_on_the_toolbar(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="0", src=1)
    win.sheet_view.set_selection([path])
    assert win.a_sheet_to_layer.text().replace("&", "") == "To Layer"
    assert win.move_selection_to_layer()
    assert sheet.paths[0].layer == "Cut", "the active layer is the cut layer"
    win.doc.undo()
    assert sheet.paths[0].layer == "0"
    win.sheet_view.set_selection([sheet.paths[0]])
    assert win.move_selection_to_layer("Drill")
    assert sheet.paths[0].layer == "Drill"


def test_move_to_layer_in_the_context_menu(win):
    win.set_mode("2d")
    sheet = win.doc.sheet
    path = sheet.add_path(rect(10, 10), closed=True, layer="0", src=1)
    win.sheet_view.set_selection([path])
    menu = win.sheet_view.build_context_menu()
    sub = next(a.menu() for a in menu.actions() if a.menu() is not None
               and a.menu().title() == "Move to Layer")
    assert any("Pocket" in a.text() for a in sub.actions())
    for a in sub.actions():
        if "Pocket" in a.text():
            a.trigger()
    assert sheet.paths[0].layer == "Pocket"


def test_new_paths_go_on_the_cut_layer(win):
    from PySide6.QtCore import QPointF, QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    win.set_mode("2d")
    v = win.sheet_view
    v.set_tool("rectangle")
    p = QPointF(200.0, 200.0)
    v.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton,
                                  Qt.LeftButton, Qt.NoModifier))
    v.submit_vcb("10;10")
    assert v.doc.sheet.paths and v.doc.sheet.paths[-1].layer == "Cut"


def test_export_default_choices(win):
    win.set_mode("2d")
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Cut", src=1)
    opt = win.default_sheet_export_options()
    assert "Cut" in opt["layers"] and "Guides" not in opt["layers"]
    assert opt["compensate"] is False, "laser software does the kerf itself"
    win.doc.sheet.add_path(rect(10, 10, 0.0, 30.0), closed=True, layer="Pocket", src=2)
    assert win.default_sheet_export_options()["compensate"] is True, "a cutter has a diameter"


def test_export_dialog_shows_the_layers(win):
    from volura.app import SheetExportDialog
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Cut", src=1)
    dlg = SheetExportDialog(win, "svg")
    opt = dlg.options()
    assert "Cut" in opt["layers"] and "Guides" not in opt["layers"]
    assert opt["compensate"] is False
    assert dlg.preview.drawing is not None and dlg.preview.drawing.paths
    dlg.deleteLater()


def test_export_sheet_with_compensation_and_per_layer(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(rect(50, 50), closed=True, layer="Cut", src=1)
    sheet.add_path(circle(25.0, 25.0, 10.0), closed=True, layer="Cut", src=1)
    sheet.add_path(rect(10, 10, 80.0, 0.0), closed=True, layer="Raster", filled=True, src=2)
    sheet.layers["Cut"].params["kerf"] = 0.2
    svg_path = str(tmp_path / "sheet.svg")
    got = win.export_sheet("svg", path=svg_path,
                           options={"layers": ["Cut", "Raster"],
                                    "compensate": True, "per_layer": False})
    assert got == svg_path
    text = open(svg_path, encoding="utf-8").read()
    back = svg.read_svg(svg_path)
    cuts = [p for p in back.paths if p.layer == "Cut"]
    assert len(cuts) == 2
    outer = max(cuts, key=lambda p: abs(polygon_area(p.pts)))
    assert span(outer.pts) == pytest.approx((50.2, 50.2), abs=0.01), "with kerf compensation"
    hole = min(cuts, key=lambda p: abs(polygon_area(p.pts)))
    r = np.linalg.norm(np.asarray(hole.pts, float) - np.array([25.0, 25.0]), axis=1)
    assert np.max(np.abs(r - 9.9)) < 0.01
    assert 'fill="#000000"' in text, "raster layer filled"
    assert 'data-volura-param-kerf="0.2"' in text, "the settings come along"
    assert "outlines exported to" in win.hint.text()

    per = str(tmp_path / "layer.svg")
    win.export_sheet("svg", path=per,
                     options={"layers": ["Cut", "Raster"],
                              "compensate": False, "per_layer": True})
    cut_path = str(tmp_path / "layer_Cut.svg")
    assert os.path.exists(cut_path) and os.path.exists(str(tmp_path / "layer_Raster.svg"))
    cuts = svg.read_svg(cut_path)
    assert {p.layer for p in cuts.paths} == {"Cut"}
    outer = max((p for p in cuts.paths if p.closed), key=lambda p: abs(polygon_area(p.pts)))
    assert span(outer.pts) == pytest.approx((50.0, 50.0), abs=0.01), "now without compensation"


def test_export_sheet_without_layers_gives_message(win, monkeypatch):
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
    assert box.messages and "operation" in box.messages[0]
    assert "Nothing was exported" in win.hint.text()
