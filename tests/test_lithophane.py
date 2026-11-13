"""Lithophanes: a plate whose thickness is the brightness of a photo, so a light behind it shows it.

`lithophane.lithophane` gives the plate as one closed volume that stands in the x-z plane: the
picture is read on a grid of square cells of `pixel` millimetres, `width / pixel` of them across,
and every cell is as thick as its brightness says - `thick` for black and `thin` for white - with a
flat frame of `thick` plastic `border` millimetres wide round it. The flat back is on `y = 0`, the
relief grows towards `-y`, the top of the picture is at the top and the plate stands on `z = 0`.

These tests weigh the plate against that: one volume on the bed and centred in x; the outside the
picture plus the frame and as deep as the black; a black picture `thick` and a white one `thin`,
and the halves of a picture lying where the halves of the image lie; inverting swapping them; the
frame as thick as the black; the grid keeping the aspect of the image and staying under its cap;
every range reached by clamping, the black by the white plus a quarter millimetre; a picture that
is no picture refused; a photo building in well under the time a window waits; and the dialog that
reads a png, builds one group and remembers what was filled in.

Run with:  python -m pytest tests/test_lithophane.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox)  # noqa: E402

from volura import lithophane  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(width=100.0, thin=0.8, thick=3.0, border=3.0, invert=False, pixel=0.2)
#: the plate is made of flat surfaces, so a part is at most this far off
ABS = 1e-4


def part(gray, **kw) -> mf.Manifold:
    return lithophane.lithophane(gray, **dict(BASE, **kw))


def numbers(shape, **kw) -> dict:
    return lithophane.sizes(shape, **dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def thickness(m, x, z, r=0.3) -> float:
    """How thick the plate is at (`x`, `z`): what a little post through it meets."""
    post = mf.Manifold.cube([2.0 * r, 40.0, 2.0 * r]).translate((x - r, -20.0, z - r))
    cut = m ^ post
    assert not cut.is_empty(), "there is no plate at that place"
    lo, hi = bbox(cut)
    return hi[1] - lo[1]


def gradient(rows, columns) -> np.ndarray:
    """A picture that goes from black on the left to white on the right."""
    return np.tile(np.linspace(0.0, 1.0, columns), (rows, 1))


def half(side) -> np.ndarray:
    """A picture that is black on one half and white on the other, row 0 at the top."""
    pic = np.ones((40, 40))
    if side == "left":
        pic[:, :20] = 0.0
    elif side == "right":
        pic[:, 20:] = 0.0
    elif side == "top":
        pic[:20] = 0.0
    else:
        pic[20:] = 0.0
    return pic


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("border", (0.0, 3.0, 10.0))
@pytest.mark.parametrize("invert", (False, True))
def test_the_plate_is_one_volume_standing_up_and_centred_in_x(border, invert):
    m = part(gradient(120, 160), width=60.0, border=border, invert=invert, pixel=0.5)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the plate fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the plate does not stand on the bed"
    assert lo[0] == pytest.approx(-hi[0], abs=ABS), "the plate is not centred in x"
    assert hi[1] == pytest.approx(0.0, abs=ABS), "the back is not on y = 0"
    assert lo[1] < -1e-9, "the relief grows towards the back instead of the front"


def test_the_outside_is_the_picture_with_the_frame_round_it():
    pic = gradient(90, 120)
    s = numbers(pic.shape, width=80.0, border=4.0, pixel=0.4)
    lo, hi = bbox(part(pic, width=80.0, border=4.0, pixel=0.4))
    assert hi[0] - lo[0] == pytest.approx(s["outside_width"], abs=ABS), "not as wide as asked"
    assert hi[2] - lo[2] == pytest.approx(s["outside_height"], abs=ABS), "not as high as asked"
    assert hi[1] - lo[1] == pytest.approx(s["thick"], abs=ABS), "not as deep as the black"
    assert s["height"] == pytest.approx(80.0 * 90 / 120), "the height does not follow the image"


# ------------------------------------------------------------------ the thickness

def test_a_black_picture_is_thick_and_a_white_one_thin():
    s = numbers((40, 40))
    black = part(np.zeros((40, 40)))
    white = part(np.ones((40, 40)))
    assert thickness(black, 0.0, s["border"] + 0.5 * s["height"]) == pytest.approx(
        s["thick"], abs=ABS)
    assert thickness(white, 0.0, s["border"] + 0.5 * s["height"]) == pytest.approx(
        s["thin"], abs=ABS)


@pytest.mark.parametrize("side", ("left", "right", "top", "bottom"))
def test_the_dark_half_of_the_image_is_the_thick_half_of_the_plate(side):
    """The top of the image is at the top of the plate and its left at -x, so a probe in the black
    half of the picture meets `thick` plastic and one in the white half meets `thin`."""
    s = numbers((40, 40))
    m = part(half(side))
    mid = s["border"] + 0.5 * s["height"]
    where = {"left": (-0.25 * s["width"], mid), "right": (0.25 * s["width"], mid),
             "top": (0.0, s["border"] + 0.75 * s["height"]),
             "bottom": (0.0, s["border"] + 0.25 * s["height"])}
    other = {"left": (0.25 * s["width"], mid), "right": (-0.25 * s["width"], mid),
             "top": (0.0, s["border"] + 0.25 * s["height"]),
             "bottom": (0.0, s["border"] + 0.75 * s["height"])}
    assert thickness(m, *where[side]) == pytest.approx(s["thick"], abs=ABS), \
        "the black half is not the thick half"
    assert thickness(m, *other[side]) == pytest.approx(s["thin"], abs=ABS), \
        "the white half is not the thin half"


@pytest.mark.parametrize("side", ("left", "top"))
def test_inverting_swaps_light_and_dark(side):
    s = numbers((40, 40))
    plain = part(half(side))
    turned = part(half(side), invert=True)
    mid = s["border"] + 0.5 * s["height"]
    x = -0.25 * s["width"] if side == "left" else 0.0
    z = mid if side == "left" else s["border"] + 0.75 * s["height"]
    assert thickness(plain, x, z) == pytest.approx(s["thick"], abs=ABS)
    assert thickness(turned, x, z) == pytest.approx(s["thin"], abs=ABS)


def test_the_frame_is_as_thick_as_the_black():
    """A white picture on a frame: the frame stands as high as a black picture would, so the thin
    edge of the photo is not where the plate breaks."""
    s = numbers((40, 40), border=5.0)
    m = part(np.ones((40, 40)), border=5.0)
    edge = 0.5 * s["outside_width"] - 2.5           # in the middle of the frame
    for x in (edge, -edge):
        assert thickness(m, x, 0.5 * s["outside_height"]) == pytest.approx(s["thick"], abs=ABS), \
            "the frame is not as thick as the black"
    assert thickness(m, 0.0, 0.5 * s["outside_height"]) == pytest.approx(s["thin"], abs=ABS)


# ------------------------------------------------------------------ the grid

def test_the_grid_keeps_the_aspect_of_the_picture():
    for shape in ((20, 40), (37, 53), (200, 150)):
        cols, rows = lithophane.grid_size(shape, 100.0, 0.5)
        assert cols / rows == pytest.approx(shape[1] / shape[0], rel=0.01), \
            "the grid is not the shape of the picture"
    assert lithophane.grid_size((100, 100), 100.0, 0.5) == (200, 200)


def test_the_grid_never_grows_past_its_cap():
    """A big picture on fine cells is read with coarser ones, so the grid stays printable."""
    cols, rows = lithophane.grid_size((2000, 3000), 250.0, 0.1)
    assert cols <= lithophane.GRID_MAX and rows <= lithophane.GRID_MAX
    assert cols == lithophane.GRID_MAX
    assert cols / rows == pytest.approx(1.5, rel=0.01)
    s = numbers((2000, 3000), width=250.0, pixel=0.1)
    assert s["pixel"] > 0.1, "the cell was not made coarser"
    assert s["cols"] <= lithophane.GRID_MAX and s["rows"] <= lithophane.GRID_MAX


def test_a_picture_of_any_size_gives_the_same_plate():
    """The same picture read on more cells is the same plate: the resampling averages, it does not
    sample, so a photo does not turn into stripes."""
    small = np.kron(gradient(20, 30), np.ones((4, 4)))
    big = np.kron(gradient(20, 30), np.ones((12, 12)))
    a = part(small, width=60.0, pixel=0.5)
    b = part(big, width=60.0, pixel=0.5)
    assert a.volume() == pytest.approx(b.volume(), rel=1e-6)


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_plate_and_the_grid():
    s = lithophane.sizes((150, 200), width=120.0, thin=1.0, thick=4.0, border=2.0, invert=True,
                         pixel=0.4)
    assert s["width"] == 120.0 and s["thin"] == 1.0 and s["thick"] == 4.0
    assert s["border"] == 2.0 and s["invert"] is True and s["pixel"] == pytest.approx(0.4)
    assert s["height"] == pytest.approx(90.0)
    assert s["cols"] == 300 and s["rows"] == 225
    assert s["outside_width"] == pytest.approx(124.0)
    assert s["outside_height"] == pytest.approx(94.0)
    assert set(s) == {"width", "thin", "thick", "border", "invert", "pixel", "height", "cols",
                      "rows", "outside_width", "outside_height"}


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers((40, 40), width=0.0, thin=0.0, thick=0.0, border=-5.0, pixel=0.0)
    assert small["width"] == lithophane.WIDTH_MIN and small["thin"] == lithophane.THIN_MIN
    assert small["thick"] == lithophane.THICK_MIN
    assert small["border"] == lithophane.BORDER_MIN
    assert small["pixel"] == lithophane.PIXEL_MIN
    big = numbers((40, 40), width=999.0, thin=999.0, thick=999.0, border=999.0, pixel=9.0)
    assert big["width"] == lithophane.WIDTH_MAX and big["thin"] == lithophane.THIN_MAX
    assert big["thick"] == lithophane.THICK_MAX and big["border"] == lithophane.BORDER_MAX
    assert big["pixel"] == lithophane.PIXEL_MAX
    lo, hi = bbox(part(np.zeros((40, 40)), width=0.0, thin=0.0, thick=0.0, border=999.0))
    assert hi[1] - lo[1] == pytest.approx(lithophane.THICK_MIN, abs=ABS)


def test_the_black_always_stands_out_above_the_white():
    """A black that is barely thicker than the white is made thicker: without that difference
    there is no photo to see."""
    near = numbers((40, 40), thin=2.0, thick=2.1)
    assert near["thick"] == pytest.approx(2.0 + lithophane.THICK_ROOM)
    small = numbers((40, 40), thin=0.4, thick=0.5)
    assert small["thick"] == lithophane.THICK_MIN
    assert small["thick"] - small["thin"] >= lithophane.THICK_ROOM - 1e-9


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("width", "thin", "thick", "border", "pixel"):
        with pytest.raises(ValueError):
            lithophane.lithophane(np.zeros((8, 8)), **dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            lithophane.sizes((8, 8), **dict(BASE, **{key: bad}))


@pytest.mark.parametrize("bad", (np.zeros(8), np.zeros((4, 4, 3)), [[], []], np.full((4, 4), np.nan),
                                 "photo"))
def test_a_picture_that_is_no_picture_is_refused(bad):
    with pytest.raises(ValueError):
        lithophane.lithophane(bad, **BASE)
    with pytest.raises(ValueError):
        lithophane.sizes(bad, **BASE)


# ------------------------------------------------------------------ the size of the mesh

def test_a_photo_builds_quickly():
    """A 300 by 200 picture on the default cells is the middle of what people load; the window
    waits for it in the background, so it has to be ready in seconds, not minutes."""
    started = time.perf_counter()
    m = part(np.random.default_rng(7).random((200, 300)))
    assert time.perf_counter() - started < 5.0, "a photo took more than five seconds"
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1


# ------------------------------------------------------------------ in the window

def write_photo(path):
    """A small png: black on the left, white on the right, written the way Qt writes images."""
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtCore import Qt
    img = QImage(40, 24, QImage.Format_Grayscale8)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255))
    p.drawRect(20, 0, 20, 24)
    p.end()
    assert img.save(str(path))
    return path


def test_the_dialog_reads_a_picture_builds_it_and_remembers_it(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    photo = write_photo(tmp_path / "holiday.png")

    def fill(dlg):
        assert not dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled(), \
            "the plate can be made before there is a picture"
        dlg.image.setText(str(photo))
        assert dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled(), \
            "the picture on disk is not accepted"
        dlg.width.setValue(80.0)
        dlg.thin.setValue(1.0)
        dlg.thick.setValue(4.0)
        dlg.border.setValue(2.0)
        dlg.detail.setCurrentIndex(dlg.detail.findData(0.4))
        dlg.invert.setChecked(True)
        assert "84.0" in dlg.info.text(), "the dialog does not show the size of the plate"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.lithophane_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no lithophane in the model"
        assert [g.name for g in w.doc.groups] == ["holiday"], "the group is not the picture's name"
        s = lithophane.sizes((24, 40), width=80.0, thin=1.0, thick=4.0, border=2.0, invert=True,
                             pixel=0.4)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the plate does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["outside_width"], abs=1e-3), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["outside_height"], abs=1e-3), \
            "not as high as computed"
        settings = app_module.Settings()
        assert (settings.lithophane_width, settings.lithophane_thin, settings.lithophane_thick,
                settings.lithophane_border, settings.lithophane_pixel,
                settings.lithophane_invert) == (80.0, 1.0, 4.0, 2.0, 0.4, True)
        assert settings.lithophane_image == str(photo)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Lithophane") == labels.index("Cable Winder") + 1
    finally:
        for key, value in (("lithophane_width", 100.0), ("lithophane_thin", 0.8),
                           ("lithophane_thick", 3.0), ("lithophane_border", 3.0),
                           ("lithophane_pixel", 0.2), ("lithophane_invert", False),
                           ("lithophane_image", "")):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
