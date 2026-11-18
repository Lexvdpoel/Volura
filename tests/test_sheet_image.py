"""A picture on the worksheet: the PNG in the meta of a path and the rectangle around it.

`volura/sheet_image.py` is the small Qt half of a picture: grey values in, base64 PNG out,
and a rectangle of four points that carries them. What is tested here is that nothing is
lost on the way - the round trip of the pixels, the size of the rectangle the picture gets,
and that a huge photograph comes back as a picture that still fits in a file.

Run with:  python -m pytest tests/test_sheet_image.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import sheet_image as si  # noqa: E402
from volura.vector2d import Drawing  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def photo(rows=6, cols=8):
    """A picture with every grey in it, so a lost value shows up right away."""
    rng = np.random.default_rng(7)
    return rng.random((rows, cols))


# ------------------------------------------------------------------ the pixels
def test_the_pixels_come_back_as_they_went():
    gray = photo()
    out = si.decode_gray(si.encode_gray(gray))
    assert out.shape == gray.shape
    assert np.max(np.abs(out - gray)) <= 1.0 / 255.0


def test_the_picture_is_a_plain_base64_text():
    text = si.encode_gray(photo())
    assert isinstance(text, str) and text.isascii()
    assert text[:4] == "iVBO"                      # the PNG signature, in base64


def test_black_burns_and_white_does_not_in_both_directions():
    gray = np.array([[0.0, 1.0]])
    assert np.allclose(si.decode_gray(si.encode_gray(gray)), gray, atol=1e-9)


def test_a_picture_that_is_no_picture_says_so():
    with pytest.raises(ValueError):
        si.decode_gray("bm90IGEgcGljdHVyZQ==")     # "not a picture"


def test_a_picture_of_many_pixels_keeps_only_what_the_laser_can_burn():
    gray = np.zeros((400, 2000))
    out = si.fit(gray)
    assert max(out.shape) == si.MAX_SIDE
    assert out.shape[0] < gray.shape[0]            # both sides follow the longest one


def test_a_picture_that_already_fits_is_left_alone():
    gray = photo()
    assert si.fit(gray) is gray


# ------------------------------------------------------------------ the rectangle
def test_the_picture_becomes_a_closed_rectangle_of_four_points():
    path = si.picture_path(photo(6, 8), 100.0, "Raster")
    assert path.closed and path.layer == "Raster"
    assert len(path.pts) == 4
    assert np.allclose(path.pts, [[0.0, 0.0], [100.0, 0.0], [100.0, 75.0], [0.0, 75.0]])


def test_the_picture_stands_where_it_was_put():
    path = si.picture_path(photo(4, 8), 20.0, "Raster", at=(30.0, 12.0))
    assert np.allclose(path.pts[0], [30.0, 12.0])          # bottom left
    assert np.allclose(path.pts[3], [30.0, 12.0 + 10.0])   # top left


def test_the_pixels_travel_in_the_meta_of_the_path():
    gray = photo(6, 8)
    path = si.picture_path(gray, 50.0, "Raster")
    assert path.meta["image_w"] == 8 and path.meta["image_h"] == 6
    assert np.max(np.abs(si.decode_gray(path.meta["image"]) - gray)) <= 1.0 / 255.0


def test_a_huge_picture_is_stored_small():
    path = si.picture_path(np.zeros((300, 3000)), 600.0, "Raster")
    assert path.meta["image_w"] == si.MAX_SIDE
    assert path.meta["image_h"] == pytest.approx(150, abs=2)
    assert len(path.meta["image"]) < 400000


# ------------------------------------------------------------------ to the laser
def test_only_the_pictures_of_the_visible_raster_layers_go_to_the_laser():
    d = Drawing.new_sheet()
    d.add_path(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
               closed=True, layer="Raster", meta={"image": si.encode_gray(photo())})
    made = si.pictures_of(d)
    assert len(made) == 1
    gray, pts, lay = made[0]
    assert gray.shape == (6, 8) and len(pts) == 4 and lay.op == "raster"
    d.layers["Raster"].visible = False
    assert si.pictures_of(d) == []


def test_a_path_without_a_picture_is_no_picture():
    d = Drawing.new_sheet()
    d.add_path(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
               closed=True, layer="Raster")
    assert si.pictures_of(d) == []


# ------------------------------------------------------------------ the layer settings
def test_a_raster_layer_asks_how_a_pixel_becomes_a_dot():
    from volura.panels import LayerParamsDialog
    from volura.vector2d import Layer
    values = LayerParamsDialog(Layer("Raster", "#000000", "raster",
                                     params={"speed": 20.0})).values()
    assert values["image_mode"] == "dither"
    assert values["interval"] == pytest.approx(0.1)
    assert values["speed"] == pytest.approx(20.0)


def test_the_line_interval_and_the_mode_of_a_layer_are_kept():
    from volura.panels import LayerParamsDialog
    from volura.vector2d import Layer
    values = LayerParamsDialog(Layer("Raster", "#000000", "raster",
                                     params={"interval": 0.25, "image_mode": "halftone"})).values()
    assert values["interval"] == pytest.approx(0.25)
    assert values["image_mode"] == "halftone"


def test_a_layer_without_pictures_is_left_as_it_was():
    from volura.panels import LayerParamsDialog
    from volura.vector2d import Layer
    values = LayerParamsDialog(Layer("Cut", "#ff0000", "cut", params={"speed": 20.0})).values()
    assert "image_mode" not in values and "interval" not in values


# ------------------------------------------------------------------ through the window
@pytest.fixture()
def win(monkeypatch, tmp_path):
    from volura import app as app_module
    from volura.app import MainWindow
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def test_the_picture_lies_at_the_margin_of_the_sheet_and_is_selected(win):
    sheet = win.doc.sheet
    sheet.sheet_w, sheet.sheet_h, sheet.sheet_margin = 60.0, 60.0, 5.0
    path = win.add_picture_to_sheet(photo())
    assert np.allclose(path.pts[0], [5.0, 5.0])
    assert path.pts[1][0] - path.pts[0][0] == pytest.approx(50.0), "the sheet is narrower than 100 mm"
    assert path.layer == sheet.default_layer("raster")
    assert win.sheet_view.selected_paths() == [path]
    assert win.mode == "2d"


def test_a_wide_sheet_gives_the_picture_its_hundred_millimetres(win):
    win.doc.sheet.sheet_w, win.doc.sheet.sheet_h = 600.0, 400.0
    path = win.add_picture_to_sheet(photo())
    assert path.pts[1][0] - path.pts[0][0] == pytest.approx(100.0)


def test_a_worksheet_without_a_raster_layer_gets_one(win):
    sheet = win.doc.sheet
    sheet.layers.pop("Raster")
    path = win.add_picture_to_sheet(photo(), width_mm=30.0)
    assert sheet.layers[path.layer].op == "raster"


def test_a_picture_on_the_worksheet_is_one_undo_step(win):
    sheet = win.doc.sheet
    before = len(sheet.paths)
    win.add_picture_to_sheet(photo(), width_mm=30.0)
    assert len(sheet.paths) == before + 1
    win.doc.undo()
    assert len(sheet.paths) == before


def test_a_click_inside_the_picture_selects_it(win):
    path = win.add_picture_to_sheet(photo(), width_mm=40.0)
    view = win.sheet_view
    middle = 0.5 * (path.pts[0] + path.pts[2])
    here = view.to_screen(middle)
    assert view.path_at(here.x(), here.y()) is path
    there = view.to_screen(middle + np.array([500.0, 500.0]))
    assert view.path_at(there.x(), there.y()) is not path


def test_the_worksheet_paints_the_picture_under_its_outline(win):
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import Qt
    path = win.add_picture_to_sheet(np.ones((10, 10)), width_mm=40.0)   # only dark pixels
    view = win.sheet_view
    view.resize(400, 300)
    view.zoom_rect((0.0, 0.0, 60.0, 60.0))
    image = QImage(400, 300, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    view._draw_paths(painter)
    painter.end()
    middle = view.to_screen(0.5 * (path.pts[0] + path.pts[2]))
    assert image.pixelColor(int(middle.x()), int(middle.y())).lightness() < 128


def test_a_hidden_layer_paints_no_picture(win):
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import Qt
    path = win.add_picture_to_sheet(np.ones((10, 10)), width_mm=40.0)
    win.doc.sheet.layers[path.layer].visible = False
    view = win.sheet_view
    view.resize(400, 300)
    view.zoom_rect((0.0, 0.0, 60.0, 60.0))
    image = QImage(400, 300, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    view._draw_paths(painter)
    painter.end()
    middle = view.to_screen(0.5 * (path.pts[0] + path.pts[2]))
    assert image.pixelColor(int(middle.x()), int(middle.y())).lightness() > 200


def test_the_gcode_of_the_worksheet_engraves_the_picture_before_it_cuts(win, tmp_path):
    sheet = win.doc.sheet
    sheet.add_path(np.array([[200.0, 200.0], [240.0, 200.0], [240.0, 230.0], [200.0, 230.0]]),
                   closed=True, layer="Cut")
    path = win.add_picture_to_sheet(np.ones((10, 10)), width_mm=20.0)
    target = str(tmp_path / "job.gcode")
    assert win.export_sheet("gcode", path=target) == target
    with open(target, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    burns = [n for n, line in enumerate(lines) if line.startswith("G1")]
    assert burns, "the picture engraves"
    assert burns[0] < lines.index("; layer Cut, pass 1/1"), "engrave first, cut last"
    for line in lines[burns[0]:lines.index("; layer Cut, pass 1/1")]:
        if line.startswith("G1"):
            x = float(line.split("X")[1].split()[0])
            y = float(line.split("Y")[1].split()[0])
            assert 5.0 - 1e-3 <= x <= 25.0 + 1e-3 and 5.0 - 1e-3 <= y <= 25.0 + 1e-3
    assert path.layer == sheet.default_layer("raster")


def test_file_import_picture_puts_the_file_on_the_worksheet(win, monkeypatch, tmp_path):
    from PySide6.QtGui import QColor, QImage
    from volura import app as app_module
    png = tmp_path / "logo.png"
    image = QImage(8, 6, QImage.Format_RGB32)
    image.fill(QColor(255, 255, 255))
    assert image.save(str(png))
    monkeypatch.setattr(app_module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(png), "")))
    win.a_import_picture.trigger()
    path = win.doc.sheet.paths[-1]
    assert path.meta["image_w"] == 8 and path.meta["image_h"] == 6
    assert path.layer == win.doc.sheet.default_layer("raster")
    assert win.mode == "2d"


def test_the_picture_action_is_a_worksheet_command(win):
    from volura import app as app_module
    assert win.a_import_picture.text() == "Picture…"
    assert app_module.tr("Put a photograph or a logo on the worksheet to engrave it") \
        in win.a_import_picture.toolTip()
    assert win._menu_modes[id(win.a_import_picture)][1] == "2d"
