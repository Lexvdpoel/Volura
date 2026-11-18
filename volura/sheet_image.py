"""A picture that lies on the 2D worksheet: how it is stored and how it is placed.

A picture on the worksheet is no new kind of object but a normal closed path of four
points - bottom left, bottom right, top right, top left, in millimetres - on a layer whose
operation is "raster". Its `meta` carries the picture itself: ``image`` is a PNG of
eight-bit grey values in base64, ``image_w`` and ``image_h`` the number of pixels. Because
it is an ordinary path, select, move, rotate, scale, copy, undo, save and load all already
work; a picture that is rotated on the sheet is a rotated rectangle with the same pixels in
it, and `volura.laser_image.sample_on_grid` puts it back on the rows of the laser.

The pixels are stored the way `volura.laser_image.to_gray` gives them: **1.0 is dark** and
burns, 0.0 is white and does not. A PNG stores the other way round, so the two directions
of :func:`encode_gray` turn the scale over.

`volura/laser_image.py` stays free of Qt; the Qt here is only the PNG, because QImage
reads every format a user can drop on the window without another package in the build.
"""
from __future__ import annotations

import base64

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QImage

from .i18n import tr
from .laser_image import resample
from .vector2d import VPath

#: the longest side of a picture that is kept, in pixels: more than this engraves no
#: finer than the dot of the laser does and only makes the file big
MAX_SIDE = 1500

#: the keys a picture leaves in the meta of its path
META_IMAGE = "image"
META_WIDTH = "image_w"
META_HEIGHT = "image_h"


def encode_gray(gray) -> str:
    """The grey values (1.0 = dark) as the base64 of an eight-bit grayscale PNG."""
    gray = np.atleast_2d(np.asarray(gray, dtype=float))
    rows, cols = gray.shape
    data = np.ascontiguousarray(np.rint((1.0 - np.clip(gray, 0.0, 1.0)) * 255.0),
                                dtype=np.uint8)
    line = (cols + 3) // 4 * 4           # QImage wants every row on four bytes
    if line > cols:
        padded = np.zeros((rows, line), dtype=np.uint8)
        padded[:, :cols] = data
        data = padded
    image = QImage(data, cols, rows, line, QImage.Format_Grayscale8)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    image.save(buf, "PNG")
    return base64.b64encode(bytes(buf.data())).decode("ascii")


def decode_gray(text) -> np.ndarray:
    """The base64 of :func:`encode_gray` back to grey values (1.0 = dark)."""
    image = QImage()
    if not image.loadFromData(base64.b64decode(str(text))):
        raise ValueError(tr("The picture in this shape could not be read."))
    image = image.convertToFormat(QImage.Format_Grayscale8)
    rows, cols, line = image.height(), image.width(), image.bytesPerLine()
    data = np.frombuffer(image.constBits(), dtype=np.uint8,
                         count=line * rows).reshape(rows, line)[:, :cols]
    return 1.0 - data.astype(float) / 255.0


def fit(gray):
    """The picture small enough to store: at most `MAX_SIDE` pixels on its longest side.

    Shrinking averages the pixels that fall together (`laser_image.resample`), so a photo
    keeps its grey instead of gaining a moiré.
    """
    gray = np.atleast_2d(np.asarray(gray, dtype=float))
    rows, cols = gray.shape
    if max(rows, cols) <= MAX_SIDE:
        return gray
    if cols >= rows:
        return resample(gray, float(cols), cols / float(MAX_SIDE))
    return resample(gray.T, float(rows), rows / float(MAX_SIDE)).T


def picture_image(text) -> QImage:
    """The base64 PNG of a picture as a QImage, to paint on the worksheet."""
    image = QImage()
    image.loadFromData(base64.b64decode(str(text)))
    return image


def pictures_of(drawing) -> list:
    """The pictures of a drawing as `(grey values, points, layer)`, ready for the laser.

    Only the paths on a visible raster layer that carry a picture; hidden layers go to no
    machine at all. `gcode.grbl_laser` takes these, because that module knows no Qt and so
    cannot read the PNG of a path itself.
    """
    out = []
    for path in drawing.paths:
        text = dict(path.meta or {}).get(META_IMAGE)
        lay = drawing.layers.get(path.layer)
        if not text or lay is None or not getattr(lay, "visible", True) or lay.op != "raster":
            continue
        out.append((decode_gray(text), np.asarray(path.pts, dtype=float), lay))
    return out


def picture_path(gray, width_mm, layer, at=(0.0, 0.0)) -> VPath:
    """The picture as a rectangle on the worksheet: `width_mm` wide, at `at` (bottom left).

    The height follows from the shape of the picture, and the pixels go in the meta, so the
    path can be moved, turned, copied and saved like any other outline.
    """
    gray = np.atleast_2d(fit(gray))
    rows, cols = gray.shape
    width_mm = float(width_mm)
    height_mm = width_mm * rows / float(cols)
    x, y = float(at[0]), float(at[1])
    pts = np.array([[x, y], [x + width_mm, y], [x + width_mm, y + height_mm], [x, y + height_mm]])
    return VPath(pts, True, layer,
                 meta={META_IMAGE: encode_gray(gray), META_WIDTH: int(cols),
                       META_HEIGHT: int(rows)})
