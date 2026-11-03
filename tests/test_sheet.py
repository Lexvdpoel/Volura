"""The 2D worksheet next to the 3D model.

The worksheet (`Document.sheet`) goes into the file and through undo and redo;
`SheetView` draws it and the tools from `volura.sheettools` edit it;
the window switches between 3D and 2D and adds import, export and
"Copy to 2D Worksheet".

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
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(1000, 700)
    w.view.resize(1000, 700)
    yield w
    w.doc.dirty = False
    w.close()


def sheet_doc():
    """Document with one square and one line on the worksheet."""
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
    """Type in the Measurements box as in the window: keys, then Enter."""
    for ch in text:
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_unknown, Qt.NoModifier, ch))
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))


def cube_in(win, w=10, d=10, h=10):
    """Cube of loose geometry in the 3D model of the window."""
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


# ------------------------------------------------------------------ 1. model and file

def test_reset_gives_empty_worksheet():
    doc = Document()
    assert isinstance(doc.sheet, Drawing)
    # a new worksheet gets the default layers; "0" stays the first
    assert doc.sheet.paths == [] and list(doc.sheet.layers)[0] == "0"
    assert {"Cut", "Engrave", "Guides"} <= set(doc.sheet.layers)
    doc.sheet.add_path(SQUARE, closed=True)
    doc.reset()
    assert doc.sheet.paths == []


def test_worksheet_goes_into_the_file(tmp_path):
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True, layer="cutting", filled=True, src=1)
    doc.sheet.add_path([[0, 0], [5, 5]], layer="engraving")
    lay = doc.sheet.ensure_layer("engraving")
    lay.color, lay.op, lay.visible = "#ff0000", "engrave", False
    doc.sheet.sheet_w, doc.sheet.sheet_h = 300.0, 200.0
    path = str(tmp_path / "model.volura")
    fileio.save_document(doc, path)

    back = Document()
    fileio.load_document(back, path)
    assert back.sheet.paths == doc.sheet.paths
    assert sorted(back.sheet.layers) == sorted(doc.sheet.layers)
    got = back.sheet.layers["engraving"]
    assert (got.color, got.op, got.visible, got.locked) == ("#ff0000", "engrave", False, False)
    assert (back.sheet.sheet_w, back.sheet.sheet_h) == (300.0, 200.0)


def test_file_without_worksheet_loads_as_empty(tmp_path):
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True)
    path = tmp_path / "old.volura"
    fileio.save_document(doc, str(path))
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    assert "sheet" in data
    del data["sheet"]
    old = tmp_path / "old2.volura"
    with zipfile.ZipFile(old, "w") as z:
        z.writestr("document.json", json.dumps(data))
    back = Document()
    fileio.load_document(back, str(old))           # a file from before the worksheet
    assert back.sheet.paths == [] and list(back.sheet.layers) == ["0"]


# ------------------------------------------------------------------ 2. undo and redo

def test_undo_and_redo_restore_the_worksheet():
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
    assert len(doc.root.F) == faces, "the 3D model has not changed"
    doc.redo()
    assert len(doc.sheet.paths) == 2
    doc.redo()
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float) + [5.0, 0.0])
    assert len(doc.root.F) == faces


def test_3d_undo_leaves_the_worksheet_alone():
    doc = Document()
    doc.sheet.add_path(SQUARE, closed=True)
    doc.root.add_shape(GROUND, [np.array(SQUARE, float)])
    doc.checkpoint()
    doc.root.erase_faces(list(doc.root.F)[:1])
    assert len(doc.root.F) == 0
    doc.undo()
    assert len(doc.root.F) == 1
    assert len(doc.sheet.paths) == 1 and np.allclose(doc.sheet.bounds(), (0, 0, 10, 10))


def test_2D_change_makes_the_document_dirty():
    doc = Document()
    seen = []
    doc.changed.connect(lambda: seen.append(1))
    assert not doc.dirty
    doc.checkpoint()
    doc.sheet.add_path(SQUARE, closed=True)
    doc.notify()
    assert doc.dirty and seen


# ------------------------------------------------------------------ 3. the worksheet itself

def test_round_trip_through_the_screen():
    v = view_of(sheet_doc())
    for point in ((0.0, 0.0), (12.5, -7.25), (-40.0, 33.5)):
        for _ in range(3):
            p = np.array(point)
            assert np.allclose(v.to_world(v.to_screen(p)), p, atol=1e-9)
            v.zoom_by(1.7)
        v.zoom_extents()


def test_wheel_zooms_around_the_cursor():
    from PySide6.QtGui import QWheelEvent
    v = view_of(sheet_doc())
    screen = QPointF(300.0, 220.0)
    before = v.to_world(screen)
    ev = QWheelEvent(screen, screen, QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                     Qt.NoModifier, Qt.NoScrollPhase, False)
    v.wheelEvent(ev)
    assert v.scale > 0
    assert np.allclose(v.to_world(screen), before, atol=1e-6), "the point under the cursor stays"


def test_zoom_extents_brings_everything_into_view():
    v = view_of(sheet_doc())
    v.scale = 0.02
    v.zoom_extents()
    box = v.world_rect()
    b = v.doc.sheet.bounds()
    assert box[0] <= b[0] and box[1] <= b[1] and box[2] >= b[2] and box[3] >= b[3]


def test_grid_steps_1_5_10_50_100():
    steps = [grid_step(s) for s in (8.0, 2.0, 1.0, 0.5, 0.1)]
    assert steps == [1.0, 5.0, 10.0, 50.0, 100.0]
    assert grid_step(0.01) > 100.0 and grid_step(100.0) < 1.0


def test_click_picks_the_nearest_path():
    v = view_of(sheet_doc())
    sq, ln = v.doc.sheet.paths
    s = v.to_screen((0.0, 5.0))
    assert v.path_at(s.x(), s.y()) is sq
    s2 = v.to_screen((45.0, 37.5))
    assert v.path_at(s2.x(), s2.y()) is ln
    s3 = v.to_screen((5.0, 5.0))                      # inside the square, but near no edge
    assert v.path_at(s3.x(), s3.y()) is None


def test_box_selection_left_to_right_and_right_to_left():
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

    window = v.paths_in_band(left, right)             # left to right: entirely inside
    assert any(p is inside for p in window)
    assert not any(p is crossing for p in window)
    assert not any(p is far for p in window)

    crossing_sel = v.paths_in_band(right, left)       # right to left: also what it touches
    assert any(p is inside for p in crossing_sel)
    assert any(p is crossing for p in crossing_sel)
    assert not any(p is far for p in crossing_sel)


def test_box_selection_by_the_tool():
    v = view_of(sheet_doc())
    sq, ln = v.doc.sheet.paths
    a = v.to_screen((-5.0, -5.0))
    b = v.to_screen((20.0, 20.0))
    drag(v, (a.x(), a.y()), (b.x(), b.y()))
    assert v.selected == [sq], "left to right: only what lies entirely inside"
    c = v.to_screen((50.0, 50.0))
    d = v.to_screen((8.0, 8.0))
    drag(v, (c.x(), c.y()), (d.x(), d.y()))
    assert {id(p) for p in v.selected} == {id(sq), id(ln)}, "right to left: also what it touches"
    drag(v, (d.x(), d.y()), (c.x(), c.y()))
    assert v.selected == [], "left to right nothing lies entirely inside here"


def test_shift_adds_and_removes():
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


def test_hidden_layer_cannot_be_seen_or_picked():
    doc = sheet_doc()
    doc.sheet.layers["0"].visible = False
    v = view_of(doc)
    s = v.to_screen((0.0, 5.0))
    assert v.path_at(s.x(), s.y()) is None
    assert v._selectable_paths() == []
    v.select_all()
    assert v.selected == []


def test_mouse_coordinates_in_the_status_bar():
    v = view_of(sheet_doc())
    got = []
    v.coords_changed.connect(got.append)
    s = v.to_screen((12.0, 8.0))
    move(v, s.x(), s.y())
    assert got and "12.00" in got[-1] and "8.00" in got[-1]


def test_worksheet_draws_itself(tmp_path):
    from PySide6.QtGui import QImage
    v = view_of(sheet_doc())
    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    v.render(image)                                   # grid, axes, paths and tool

    def dark():
        return sum(1 for y in range(0, 600, 2) for x in range(0, 900, 2)
                   if image.pixelColor(x, y).lightness() < 200)

    assert dark() > 50, "the worksheet draws nothing"


# ------------------------------------------------------------------ 4. tools

def test_SHEET_TOOLS_register():
    names = [cls.name for cls in SHEET_TOOLS]
    # the four basic tools first, then the drawing tools
    assert names[:4] == ["select", "move", "rotate", "scale"]
    assert {"line", "rectangle", "roundrect", "circle", "ellipse", "arc", "centerarc",
            "polygon", "text", "offset"} <= set(names)
    v = view_of(sheet_doc())
    assert set(v.tools) == set(names)


def test_move_with_typed_distance():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("move")
    base = v.to_screen((0.0, 0.0))
    click(v, base.x(), base.y())
    type_in(v, "10;5")
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([10.0, 5.0]))


def test_move_with_single_distance_along_the_mouse():
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


def test_move_click_click_and_copy_with_ctrl():
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
    assert len(doc.sheet.paths) == n + 1, "with Ctrl a copy is added"
    assert np.allclose(pts_of(sq), np.array(SQUARE, float) + np.array([25.0, 0.0]))
    copy = doc.sheet.paths[-1]
    assert np.allclose(pts_of(copy), np.array(SQUARE, float) + np.array([25.0, 40.0]))


def test_rotate_90_degrees():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("rotate")
    c = v.to_screen((0.0, 0.0))
    click(v, c.x(), c.y())
    v.submit_vcb("90")
    assert np.allclose(pts_of(sq), np.array([[0, 0], [0, 10], [-10, 10], [-10, 0]], float))


def test_rotate_by_clicking():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("rotate")
    c = v.to_screen((5.0, 5.0))                       # centre of the square
    click(v, c.x(), c.y())
    p = v.to_screen((5.0, 25.0))                      # 90° counterclockwise from the first click
    click(v, p.x(), p.y())
    assert np.allclose(np.sort(pts_of(sq), axis=0), np.sort(np.array(SQUARE, float), axis=0),
                       atol=1e-6)


def test_scale_2x():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("scale")
    a = v.to_screen((5.0, 5.0))
    click(v, a.x(), a.y())
    v.submit_vcb("2")
    assert np.allclose(pts_of(sq), np.array([[-5, -5], [15, -5], [15, 15], [-5, 15]], float))


def test_scale_by_dragging():
    doc = sheet_doc()
    v = view_of(doc)
    sq = doc.sheet.paths[0]
    v.set_selection([sq])
    v.set_tool("scale")
    a = v.to_screen((15.0, 5.0))                      # distance 10 mm from the centre
    b = v.to_screen((25.0, 5.0))                      # now 20 mm: factor 2
    drag(v, (a.x(), a.y()), (b.x(), b.y()))
    assert np.allclose(pts_of(sq), np.array([[-5, -5], [15, -5], [15, 15], [-5, 15]], float))


def test_mirror_around_the_center_of_the_selection():
    doc = Document()
    p = doc.sheet.add_path([[0, 0], [10, 0], [10, 4]], closed=False)
    v = view_of(doc)
    v.set_selection([p])
    assert v.mirror(True)
    assert np.allclose(pts_of(p), np.array([[10, 0], [0, 0], [0, 4]], float))
    assert v.mirror(False)
    assert np.allclose(pts_of(p), np.array([[10, 4], [0, 4], [0, 0]], float))


def test_select_all_and_delete():
    doc = sheet_doc()
    v = view_of(doc)
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.ControlModifier, "a"))
    assert len(v.selected) == 2
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier, ""))
    assert doc.sheet.paths == [] and v.selected == []
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier, "\x1b"))


def test_esc_and_tool_keys():
    v = view_of(sheet_doc())
    v.set_selection(list(v.doc.sheet.paths))
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier, "\x1b"))
    assert v.selected == []
    for key, name in ((Qt.Key_M, "move"), (Qt.Key_Q, "rotate"), (Qt.Key_S, "scale"),
                      (Qt.Key_Space, "select")):
        v.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, ""))
        assert v.tool.name == name


def test_every_operation_is_one_undo_step():
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
    assert len(doc.undo_stack) == steps + 1, "moving is one undo step"
    doc.undo()
    assert len(doc.sheet.paths) == len(before)
    assert all(np.allclose(pts_of(a), b) for a, b in zip(doc.sheet.paths, before))


# ------------------------------------------------------------------ 5. 3D and 2D in the window

def sheet_toolbar(win):
    return next((b for b in win.findChildren(QToolBar) if b.objectName() == "SheetToolbar"), None)


def test_sheettoolbar_is_only_in_2d(win):
    bar = sheet_toolbar(win)
    assert bar is not None, "toolbar with objectName 'SheetToolbar' is missing"
    assert bar.isHidden(), "in 3D the worksheet toolbar is not visible"
    win.set_mode("2d")
    assert not bar.isHidden()
    win.set_mode("3d")
    assert bar.isHidden()


def test_set_mode_does_not_change_the_3d_model(win):
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


def test_in_2d_the_3d_toolbars_and_tools_step_aside(win):
    shown = [b for b in (win.tb_main, win.tb_tools, win.tb_print) if not b.isHidden()]
    win.set_mode("2d")
    assert all(b.isHidden() for b in shown), "the 3D toolbars are hidden in 2D"
    assert not win.tool_actions["pushpull"].isEnabled()
    assert not win.a_delete.isEnabled()
    win.set_mode("3d")
    assert [b for b in (win.tb_main, win.tb_tools, win.tb_print) if not b.isHidden()] == shown
    assert win.tool_actions["pushpull"].isEnabled()


def test_view_menu_and_status_bar_switch(win):
    view = next(m for m in win.menuBar().findChildren(QMenu)
                 if m.title().replace("&", "") == "View")
    labels = [a.text().replace("&", "").replace("…", "")
              for a in view.actions() if not a.isSeparator()]
    assert "2D Worksheet" in labels and "Toolbars" in labels
    assert labels.index("Toolbars") < labels.index("2D Worksheet")
    assert win.a_sheet.text().replace("&", "") == "2D Worksheet"
    assert win.a_sheet.isCheckable()
    assert "Ctrl+Shift+2" in [s.toString() for s in win.a_sheet.shortcuts()]
    assert win.mode_button.text() == "3D | 2D", "the button at the bottom right switches between 3D and 2D"
    win.toggle_sheet_mode()
    assert win.mode == "2d" and win.a_sheet.isChecked()
    assert "2D worksheet" in win.mode_button.toolTip()
    win.mode_button.click()
    assert win.mode == "3d" and not win.a_sheet.isChecked()
    assert "3D model" in win.mode_button.toolTip()


def test_mode_does_not_belong_to_the_file(win, tmp_path):
    win.set_mode("2d")
    win.doc.sheet.add_path(SQUARE, closed=True)
    path = str(tmp_path / "m.volura")
    win.doc.path = path
    assert win.file_save()
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    assert "sheet" in data and "mode" not in data
    win.file_open(path)
    assert win.mode == "2d", "loading does not change the chosen mode"
    assert len(win.doc.sheet.paths) == 1


# ------------------------------------------------------------------ 6. import and export

def test_import_in_2d_goes_to_the_worksheet(win, tmp_path):
    src = Drawing()
    src.add_path(SQUARE, closed=True, layer="cutting", filled=True, src=1)
    src.add_path([[40, 0], [50, 0], [50, 10]], closed=False, layer="engraving")
    path = str(tmp_path / "drawing.svg")
    svg.write_svg(src, path)

    win.set_mode("2d")
    added = win.import_vector_to_sheet(path)
    assert added and len(added) == len(win.doc.sheet.paths)
    assert np.allclose(win.doc.sheet.bounds(), src.bounds(), atol=1e-3), "positions stay"
    assert {"cutting", "engraving"} <= set(win.doc.sheet.layers)
    assert win.mode == "2d"
    assert set(id(p) for p in win.sheet_view.selected) == {id(p) for p in added}

    again = win.import_vector_to_sheet(path)
    # layers are merged by name: two are added, also the second time
    before = len(win.doc.sheet.layers)
    assert len(again) == 2
    win.import_vector_to_sheet(path)
    assert len(win.doc.sheet.layers) == before, "layers are merged by name"


def test_import_in_3d_still_goes_to_the_model(win, tmp_path):
    src = Drawing()
    src.add_path(SQUARE, closed=True, layer="0", filled=True, src=1)
    path = str(tmp_path / "block.svg")
    svg.write_svg(src, path)
    assert win.mode == "3d"
    win.import_vector(path, options={"anchor": "center", "open_edges": True})
    assert win.doc.sheet.paths == [], "in 3D importing goes to the model"
    assert win.doc.root.F or win.doc.root.E


def test_export_in_2d_writes_visible_layers(win, tmp_path):
    doc = win.doc
    doc.sheet.add_path(SQUARE, closed=True, layer="cutting", filled=True, src=1)
    doc.sheet.add_path([[50, 0], [60, 0]], layer="engraving")
    doc.sheet.layers["engraving"].visible = False
    win.set_mode("2d")

    a = str(tmp_path / "sheet.svg")
    assert win.export_vector("svg", path=a) == a
    back = svg.read_svg(a)
    assert {p.layer for p in back.paths} == {"cutting"}
    assert abs(total_area(back) - 100.0) < 1e-3

    b = str(tmp_path / "sheet.dxf")
    assert win.export_vector("dxf", path=b) == b
    assert {p.layer for p in dxf.read_dxf(b).paths} == {"cutting"}
    assert "outlines exported to" in win.hint.text()


def test_copy_to_2d_worksheet_with_top_view(win):
    cube_in(win, 10, 10, 10)
    model = win.doc.root.state()
    added = win.copy_to_sheet({"source": "top"})
    assert added and len(added) == 1
    assert win.mode == "2d"
    assert win.doc.root.state() == model, "the 3D model stays unchanged"
    p = added[0]
    assert p.closed and np.allclose(np.sort(pts_of(p), axis=0),
                                    np.sort(np.array(SQUARE, float), axis=0), atol=1e-6)
    assert abs(total_area(win.doc.sheet) - 100.0) < 1e-3
    assert win.sheet_view.selected == added


def test_copy_to_2d_worksheet_with_empty_result(win, monkeypatch):
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
    assert win.hint.text() == "Nothing was copied."


def test_copy_can_also_do_section_and_faces(win):
    cube_in(win, 10, 10, 10)
    for opts in ({"source": "section", "z": 5.0}, {"source": "faces"}):
        win.doc.set_selection([("f", 0, f) for f in list(win.doc.root.F)[:1]])
        added = win.copy_to_sheet(dict(opts))
        assert added and win.doc.sheet.paths, opts
        win.doc.undo()
    assert win.doc.sheet.paths == [], "undo removes both copies again"


def test_pan_with_the_middle_button_and_space():
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
    assert v.origin.x() == pytest.approx(start.x() - 50), "space + drag pans"
    v.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier, " "))
    start = QPointF(v.origin)
    drag(v, (400.0, 300.0), (420.0, 300.0))
    assert v.origin.x() == pytest.approx(start.x()), "dragging without space does not pan"


def test_sheet_rectangle_and_layer_colors_are_drawn():
    from PySide6.QtGui import QImage
    doc = Document()
    doc.sheet.sheet_w, doc.sheet.sheet_h = 300.0, 200.0
    doc.sheet.add_path(SQUARE, closed=True, layer="cutting")
    doc.sheet.layers["cutting"].color = "#0000ff"
    v = view_of(doc)
    v.zoom_extents()
    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    v.render(image)

    def blues():
        return sum(1 for y in range(0, 600, 2) for x in range(0, 900, 2)
                   if image.pixelColor(x, y).blue() > 150
                   and image.pixelColor(x, y).red() < 120)

    assert blues() > 20, "the layer color is not drawn"
    v.set_selection(list(doc.sheet.paths))
    v.render(image)
    assert blues() > 20, "a selected path is blue too"
    doc.sheet.layers["cutting"].visible = False
    v.render(image)
    assert blues() == 0, "a hidden layer is not visible"


def test_invalid_input_in_the_measurements_box():
    doc = sheet_doc()
    v = view_of(doc)
    v.set_selection([doc.sheet.paths[0]])
    v.set_tool("move")
    a = v.to_screen((0.0, 0.0))
    click(v, a.x(), a.y())
    v.submit_vcb("no")
    assert "Invalid input" in v._status_shown
    assert np.allclose(pts_of(doc.sheet.paths[0]), np.array(SQUARE, float))


def test_an_internal_error_in_2d_only_gives_a_message(monkeypatch):
    doc = sheet_doc()
    v = view_of(doc)
    logged = []
    monkeypatch.setattr(errors, "log_exception", lambda *a, **k: logged.append(k))
    v.tools["select"].mouse_press = lambda e, x, y: (_ for _ in ()).throw(RuntimeError("broken"))
    click(v, 100.0, 100.0)                     # the app may simply carry on
    assert logged, "the error is logged"
    assert errors.ERROR_STATUS in v._status_shown


def test_measurements_box_in_the_status_bar_works_in_2d(win):
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
