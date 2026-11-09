"""2D vector export from the 3D model: selected faces, top view, section.

The pure logic (`volura.vector_export`) can be tested without a window; the last
block checks the menu items, the preview in the dialog and writing a
file through the window.

Run with:  python -m pytest tests
"""
import math
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import dxf, ops, svg  # noqa: E402
from volura import vector_export as vx  # noqa: E402
from volura.geom import Plane, rotation_matrix  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.vector2d import polygon_area, regions  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

GROUND = Plane((0, 0, 0), (0, 0, 1))


# ------------------------------------------------------------------ helpers

def box_doc(w=10, d=10, h=10):
    """Cube of w x d x h mm on the ground plane, as loose geometry."""
    doc = Document()
    doc.root.add_shape(GROUND, [np.array([[0, 0], [w, 0], [w, d], [0, d]], float)])
    doc.root.pushpull(next(iter(doc.root.F)), -h)
    return doc


def cylinder_doc(r=5.0, h=10.0, segments=64):
    doc = Document()
    ops.group_from_manifold(doc, ops.primitive("cylinder", r, h, segments=segments),
                            "#f2f2f2", "Cylinder")
    return doc


def plate_doc(w=20, d=10, h=4, hole=(8.0, 4.0, 10.0, 6.0)):
    """Plate with a through hole: outer edge 20 x 10, hole 2 x 2 mm."""
    doc = Document()
    outer = np.array([[0, 0], [w, 0], [w, d], [0, d]], float)
    x0, y0, x1, y1 = hole
    hole_loop = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)[::-1].copy()
    doc.root.add_shape(GROUND, [outer, hole_loop])
    doc.root.pushpull(next(iter(doc.root.F)), -h)
    return doc


def slanted_plane(angle=35.0):
    """The plane of a small plate turned about the x axis."""
    R = rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(angle))
    return Plane((0, 0, 0), R @ np.array([0.0, 0.0, 1.0]), R @ np.array([1.0, 0.0, 0.0]))


def slanted_face_doc(angle=35.0):
    """Face of 10 x 20 mm, turned about the x axis; gives (document, face id)."""
    doc = Document()
    fid = doc.root.add_shape(slanted_plane(angle),
                             [np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)])[0]
    return doc, fid


def hole_face(ents):
    """The face with the most loops: for the plate that is the face with the hole."""
    return max(ents.F, key=lambda f: (len(ents.F[f].loops), -f))


def shapes_of(drawing):
    """The drawing as filled shapes: [outer edge, holes]."""
    return regions(drawing.paths)


def shape_area(shape):
    return abs(sum(polygon_area(loop) for loop in shape))


def total_area(drawing):
    return sum(shape_area(s) for s in shapes_of(drawing))


def bbox_of_src(drawing, src):
    pts = np.concatenate([p.pts for p in drawing.paths if p.src == src])
    return tuple(pts.min(axis=0)) + tuple(pts.max(axis=0))


def separation(a, b):
    """Smallest distance between two bounding boxes (0 if they touch or intersect)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


# ------------------------------------------------------------------ top view

def test_top_view_of_cube_is_one_closed_square_of_100():
    d = vx.drawing_top(box_doc())
    assert len(d.paths) == 1
    assert d.paths[0].closed and d.paths[0].layer == "0"
    sh = shapes_of(d)
    assert len(sh) == 1 and len(sh[0]) == 1
    assert abs(shape_area(sh[0]) - 100.0) < 1e-6
    assert np.allclose(d.bounds(), (0.0, 0.0, 10.0, 10.0))
    assert d.warnings == []


def test_top_view_keeps_the_model_origin():
    doc = box_doc()
    doc.root.transform(np.eye(3), np.array([12.0, -7.0, 0.0]))
    assert np.allclose(vx.drawing_top(doc).bounds(), (12.0, -7.0, 22.0, 3.0))


def test_top_view_merges_the_parts_of_the_model():
    doc = box_doc()
    ops.group_from_manifold(doc, ops.primitive("box", 4, 4, 4).translate((3, 3, 2)),
                            "#d23b3b", "Block")
    d = vx.drawing_top(doc)
    assert len(shapes_of(d)) == 1
    assert abs(total_area(d) - 100.0) < 1e-6          # the block lies inside the cube


def test_top_view_of_only_selected_groups():
    doc = box_doc()
    g = ops.group_from_manifold(doc, ops.primitive("box", 4, 4, 4).translate((3, 3, 2)),
                                "#d23b3b", "Block")
    doc.set_selection([("g", g.id, g.id)])
    assert abs(total_area(vx.drawing_top(doc, True)) - 16.0) < 1e-6


def test_top_view_of_plate_with_hole_has_outer_loop_and_hole():
    d = vx.drawing_top(plate_doc())
    assert all(p.closed and p.layer == "0" for p in d.paths)
    sh = shapes_of(d)
    assert len(sh) == 1 and len(sh[0]) == 2
    outer, hole = sh[0]
    assert polygon_area(outer) > 0 and polygon_area(hole) < 0
    assert abs(polygon_area(outer) - 200.0) < 1e-6
    assert abs(polygon_area(hole) + 4.0) < 1e-6
    assert abs(shape_area(sh[0]) - 196.0) < 1e-6
    # outer edge and hole belong to one source element, otherwise the hole disappears
    assert len(d.paths) == 2 and len({p.src for p in d.paths}) == 1


# ------------------------------------------------------------------ section

def test_section_of_cube_at_mid_height_is_the_same_square():
    d = vx.drawing_section(box_doc(), 5.0)
    sh = shapes_of(d)
    assert len(sh) == 1 and len(sh[0]) == 1
    assert abs(shape_area(sh[0]) - 100.0) < 1e-6
    assert np.allclose(d.bounds(), (0.0, 0.0, 10.0, 10.0))


def test_section_above_the_model_is_empty():
    assert vx.drawing_section(box_doc(), 12.0).paths == []


def test_section_of_cylinder_is_a_closed_circle_of_about_pi_r_squared():
    d = vx.drawing_section(cylinder_doc(), 5.0)
    assert len(d.paths) == 1 and d.paths[0].closed and d.paths[0].layer == "0"
    pts = d.paths[0].pts
    assert len(pts) >= 16
    area = abs(polygon_area(pts))
    assert abs(area - math.pi * 25.0) <= 0.02 * math.pi * 25.0
    r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    assert r.max() - r.min() < 1e-6                    # one circle, no double points


def test_section_of_plate_with_hole_is_a_ring():
    sh = shapes_of(vx.drawing_section(plate_doc(), 2.0))
    assert len(sh) == 1 and len(sh[0]) == 2
    assert abs(shape_area(sh[0]) - 196.0) < 1e-6


def test_section_z_default_is_the_middle_of_the_model():
    assert abs(vx.section_z_default(box_doc()) - 5.0) < 1e-9
    assert vx.section_z_default(Document()) == 0.0


# ------------------------------------------------------------------ selected faces

def test_selected_face_is_exported_at_true_size():
    doc, fid = slanted_face_doc()
    d = vx.drawing_from_faces(doc, [("f", 0, fid)])
    assert len(d.paths) == 1 and d.paths[0].closed and d.paths[0].layer == "0"
    b = d.bounds()
    assert abs((b[2] - b[0]) - 10.0) < 1e-9
    assert abs((b[3] - b[1]) - 20.0) < 1e-9
    pts = d.paths[0].pts
    edges = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert sorted(np.round(edges, 6)) == [10.0, 10.0, 20.0, 20.0]


def test_selected_face_with_a_hole_keeps_its_hole():
    doc = plate_doc()
    d = vx.drawing_from_faces(doc, [("f", 0, hole_face(doc.root))])
    sh = shapes_of(d)
    assert len(sh) == 1 and len(sh[0]) == 2
    assert abs(shape_area(sh[0]) - 196.0) < 1e-6


def test_faces_in_the_same_plane_keep_their_place():
    doc = Document()
    doc.root.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])
    doc.root.add_shape(GROUND, [np.array([[15, 0], [25, 0], [25, 10], [15, 10]], float)])
    d = vx.drawing_from_faces(doc, [("f", 0, f) for f in doc.root.F])
    assert len(d.paths) == 2
    assert np.allclose(d.bounds(), (0.0, 0.0, 25.0, 10.0))
    assert len(shapes_of(d)) == 2
    assert abs(total_area(d) - 200.0) < 1e-6


def test_faces_in_different_planes_are_laid_out_side_by_side():
    doc = Document()
    doc.root.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])
    doc.root.add_shape(Plane((0, 0, 0), (0, -1.0, 0), (1.0, 0, 0)),
                       [np.array([[0, 0], [30, 0], [30, 5], [0, 5]], float)])
    d = vx.drawing_from_faces(doc, [("f", 0, f) for f in doc.root.F])
    boxes = [bbox_of_src(d, s) for s in sorted({p.src for p in d.paths})]
    assert len(boxes) == 2
    assert separation(boxes[0], boxes[1]) >= vx.GAP - 1e-9
    assert abs(total_area(d) - 250.0) < 1e-6


def test_faces_export_ignores_edges_and_unknown_items():
    doc = box_doc()
    items = [("e", 0, e) for e in list(doc.root.E)[:3]] + [("g", 999, 999), ("f", 0, 999)]
    assert vx.drawing_from_faces(doc, items).paths == []


def test_export_without_selection_uses_the_document_selection():
    doc, fid = slanted_face_doc()
    doc.set_selection([("f", 0, fid)])
    d = vx.drawing_from_faces(doc)
    assert len(d.paths) == 1
    assert abs((d.bounds()[2] - d.bounds()[0]) - 10.0) < 1e-9


# ------------------------------------------------------------------ reading back from SVG and DXF

def make_cases():
    plate = plate_doc()
    return {
        "top":vx.drawing_top(plate),
        "section":vx.drawing_section(plate, 2.0),
        "faces":vx.drawing_from_faces(plate, [("f", 0, f) for f in plate.root.F]),
        "cylinder":vx.drawing_section(cylinder_doc(), 5.0),
    }


CASES = make_cases()


@pytest.mark.parametrize("name", list(CASES))
def test_svg_round_trip(name):
    drawing = CASES[name]
    back = svg.read_svg(svg.write_svg(drawing).encode())
    assert len(back.paths) == len(drawing.paths)
    assert all(p.closed for p in back.paths)
    assert np.allclose(back.bounds(), drawing.bounds(), atol=1e-3)
    assert abs(total_area(back) - total_area(drawing)) < 1e-3 * max(total_area(drawing), 1.0)
    assert [len(s) for s in shapes_of(back)] == [len(s) for s in shapes_of(drawing)]


@pytest.mark.parametrize("name", list(CASES))
def test_dxf_round_trip(name):
    drawing = CASES[name]
    back = dxf.read_dxf(dxf.write_dxf(drawing))
    assert back.paths and all(p.closed and p.layer == "0" for p in back.paths)
    # a circle stays a CIRCLE; it is read back with the arc tolerance of 0.01 mm
    assert np.allclose(back.bounds(), drawing.bounds(), atol=0.02)
    assert abs(total_area(back) - total_area(drawing)) < 0.01 * max(total_area(drawing), 1.0)
    assert [len(s) for s in shapes_of(back)] == [len(s) for s in shapes_of(drawing)]


def test_svg_and_dxf_files_are_written_to_disk():
    drawing = vx.drawing_top(plate_doc())
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "model.svg")
        b = os.path.join(tmp, "model.dxf")
        assert svg.write_svg(drawing, a) and os.path.getsize(a) > 0
        assert dxf.write_dxf(drawing, b) and os.path.getsize(b) > 0
        assert np.allclose(svg.read_svg(a).bounds(), drawing.bounds(), atol=1e-3)
        assert np.allclose(dxf.read_dxf(b).bounds(), drawing.bounds(), atol=1e-3)


# ------------------------------------------------------------------ the window

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


class MessageBox:
    """Stands in for QMessageBox, so a test reads the messages instead of clicking."""

    def __init__(self):
        self.messages = []

    def information(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return 0

    def critical(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return 0

    def warning(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return 0


def export_menu(win):
    files = next(m for m in win.menuBar().findChildren(QMenu)
                 if m.title().replace("&", "") == "File")
    return next(m for m in files.findChildren(QMenu)
                if m.title().replace("&", "") == "Export")


def menu_labels(menu):
    return [a.text().replace("&", "").replace("…", "")
            for a in menu.actions() if not a.isSeparator()]


def test_export_menu_has_the_two_vector_items(win):
    labels = menu_labels(export_menu(win))
    assert "2D Vector (SVG)" in labels and "2D Vector (DXF)" in labels
    assert labels.index("2D Vector (SVG)") < labels.index("2D Vector (DXF)")
    assert labels.index("STL per Color") < labels.index("2D Vector (SVG)")
    assert win.a_exp_svg2d.text().replace("&", "").replace("…", "") == "2D Vector (SVG)"
    assert win.a_exp_dxf2d.text().replace("&", "").replace("…", "") == "2D Vector (DXF)"


def test_window_writes_svg_and_dxf(win, tmp_path):
    doc = win.doc
    doc.root.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])
    doc.root.pushpull(next(iter(doc.root.F)), -10)
    doc.notify()

    a = str(tmp_path / "model.svg")
    assert win.export_vector("svg", path=a, options={"source": "top"}) == a
    assert abs(total_area(svg.read_svg(a)) - 100.0) < 1e-3
    assert "1 outline exported to" in win.hint.text() and a in win.hint.text()

    b = str(tmp_path / "section.dxf")
    assert win.export_vector("dxf", path=b, options={"source": "section", "z": 5.0}) == b
    assert abs(total_area(dxf.read_dxf(b)) - 100.0) < 1e-3
    assert b in win.hint.text()


def test_window_exports_the_selected_faces(win, tmp_path):
    doc = win.doc
    fid = doc.root.add_shape(slanted_plane(),
                             [np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)])[0]
    doc.set_selection([("f", 0, fid)])
    doc.notify()
    p = str(tmp_path / "faces.svg")
    assert win.export_vector("svg", path=p, options={"source": "faces"}) == p
    b = svg.read_svg(p).bounds()
    assert abs((b[2] - b[0]) - 10.0) < 1e-3 and abs((b[3] - b[1]) - 20.0) < 1e-3


def test_window_refuses_an_empty_result(win, tmp_path, monkeypatch):
    box = MessageBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    p = str(tmp_path / "nothing.svg")
    assert win.export_vector("svg", path=p, options={"source": "top"}) is None
    assert not os.path.exists(p)
    assert "volume" in " ".join(box.messages)
    assert win.hint.text() == "Nothing was exported."

    assert win.export_vector("dxf", path=str(tmp_path / "nothing.dxf"),
                             options={"source": "section", "z": 99.0}) is None


def test_window_reports_a_part_that_is_not_a_volume(win, tmp_path, monkeypatch):
    box = MessageBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    win.doc.root.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])
    win.doc.notify()
    assert win.export_vector("svg", path=str(tmp_path / "face.svg"),
                             options={"source": "top"}) is None
    assert "open or doubled edge" in " ".join(box.messages)


def test_vector_export_dialog_offers_the_three_sources(win):
    dlg = app_module.VectorExportDialog(win)
    assert dlg.windowTitle() == "Export 2D Vector"
    assert not dlg.b_faces.isEnabled()                 # nothing is selected
    assert dlg.options()["source"] == "top"
    assert not dlg.z_edit.isEnabled()
    assert dlg.preview.drawing is not None             # the preview draws right away

    doc = win.doc
    doc.root.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)])
    doc.root.pushpull(next(iter(doc.root.F)), -10)
    doc.notify()

    dlg2 = app_module.VectorExportDialog(win)
    assert abs(dlg2.options()["z"] - 5.0) < 1e-9       # halfway up the model

    doc.set_selection([("f", 0, list(doc.root.F)[0])])
    dlg3 = app_module.VectorExportDialog(win)
    assert dlg3.b_faces.isEnabled() and dlg3.b_faces.isChecked()
    assert dlg3.options()["source"] == "faces"
    assert not dlg3.only.isEnabled()                   # does not apply to faces

    dlg3.b_section.setChecked(True)
    assert dlg3.z_edit.isEnabled()
    dlg3.z_edit.setText("2.5 in")
    dlg3.refresh()
    assert abs(dlg3.options()["z"] - 63.5) < 1e-9      # a number with a unit
    assert dlg3.z_edit.styleSheet() == ""
    dlg3.z_edit.setText("no")
    dlg3.refresh()
    assert abs(dlg3.options()["z"] - 5.0) < 1e-9       # fell back to the middle
    assert dlg3.z_edit.styleSheet()                    # red border on invalid text


def test_vector_export_dialog_only_selected(win):
    doc = win.doc
    g = ops.group_from_manifold(doc, ops.primitive("box", 4, 4, 4), "#d23b3b", "Block")
    doc.set_selection([("g", g.id, g.id)])
    dlg = app_module.VectorExportDialog(win)
    assert dlg.only.isEnabled() and not dlg.only.isChecked()
    dlg.only.setChecked(True)
    assert dlg.options()["only_selected"] is True
    assert abs(total_area(dlg.preview.drawing) - 16.0) < 1e-6


def test_vector_preview_paints_the_drawing(win):
    preview = app_module.VectorPreview(win)
    preview.resize(240, 180)
    image = QImage(240, 180, QImage.Format_ARGB32)
    image.fill(0)

    def dark_pixels():
        return sum(1 for y in range(0, 180, 3) for x in range(0, 240, 3)
                   if image.pixelColor(x, y).lightness() < 128)

    preview.render(image)                                  # no drawing yet
    assert dark_pixels() == 0
    preview.set_drawing(vx.drawing_top(plate_doc()))
    preview.render(image)
    assert dark_pixels() > 20, "the preview draws no outlines"
    preview.set_drawing(None)
    preview.render(image)                                  # must not crash then


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and not fn.__code__.co_argcount:
            fn()
            print("ok", name)
