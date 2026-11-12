"""Slicing for resin printers: from parts to the layer images the printer exposes.

A resin (MSLA) printer cures one whole layer at a time with an image on its screen, so a slice is
a stack of grayscale images. This cuts the parts - with their supports and pad, see
`volura.sla_supports` - at the middle of every layer, fills the cross-sections into an image of
the printer's screen with smooth (anti-aliased) edges, and writes them as an SL1 archive: the
open format of PrusaSlicer, a zip with one PNG per layer, config.ini and the profile as
prusaslicer.ini. UVtools' command line then converts that archive into the format the printer
reads (.ctb, .goo, .pwmx, ...), with the lift and light values of the printer's profile.

The image follows PrusaSlicer's mapping, which the profiles are made for: millimetres on the
build plate become pixels with the origin in the top left corner of the image, then the image is
mirrored as the profile says, and a portrait screen swaps x and y.

Qt is only used to draw and encode the images (QImage, QPainter); no widgets.
"""
from __future__ import annotations

import io
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import manifold3d as mf
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QImage, QPainter, QPainterPath, QPolygonF, QTransform

from .sla_profiles import (ResinPrinter, ResinPrint, ResinSettings, converter,
                           converter_environment)

#: the name of the layers inside the archive
JOB = "volura"
#: seconds a layer takes besides its light: the plate lifts out of the resin and comes back
LIFT_SECONDS = 6.0


@dataclass
class ResinResult:
    """What a resin slice made: the file, its layers, the time and the resin it takes."""

    path: str
    layers: int
    seconds: float
    millilitres: float
    archive: str = ""            # the SL1 archive the printer's file was made from


def image_transform(printer: ResinPrinter) -> QTransform:
    """Millimetres on the build plate to pixels of the screen, as PrusaSlicer maps them."""
    pw, ph = printer.pixel
    w, h = printer.pixels_x, printer.pixels_y
    sx, sy = 1.0 / pw, 1.0 / ph
    flip_x = printer.mirror_x
    flip_y = not printer.mirror_y                 # the origin of an image is its top left
    t = QTransform()
    if printer.portrait:
        # a portrait screen: x and y swap, and the swap itself mirrors x once
        flip_x = not flip_x
        t = QTransform(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        sx, sy = 1.0 / ph, 1.0 / pw
    scale = QTransform.fromScale(-sx if flip_x else sx, -sy if flip_y else sy)
    shift = QTransform.fromTranslate(w if flip_x else 0.0, h if flip_y else 0.0)
    return t * scale * shift


def layer_image(polygons, printer: ResinPrinter, transform: QTransform) -> QImage:
    """The image of one layer: its polygons (in mm on the plate) white on black."""
    img = QImage(printer.pixels_x, printer.pixels_y, QImage.Format_Grayscale8)
    img.fill(0)
    if polygons:
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for poly in polygons:
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
            path.closeSubpath()
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setTransform(transform)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(Qt.white))
        p.drawPath(path)
        p.end()
    return img


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png_bytes(img: QImage) -> bytes:
    """A grayscale image as PNG, quickly.

    A layer of a 12K screen is some 60 million pixels, mostly black; Qt's own PNG writer takes
    more than a second for it. This writes the rows unfiltered and compresses them with zlib at
    its fastest level, which takes a fraction of that and lets several layers encode at once.
    """
    return _png(img.width(), img.height(), _pixels(img), 0, 0)


def _pixels(img: QImage) -> np.ndarray:
    """The pixels of a grayscale image as rows of bytes, without the padding at their ends."""
    rows = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.bytesPerLine() * img.height())
    return rows.reshape(img.height(), img.bytesPerLine())[:, :img.width()]


#: black rows are compressed this many at a time, once per row width, and reused
_ZERO_ROWS = 256
_zero_blocks: dict[int, bytes] = {}
_ADLER = 65521


def _deflate(data: bytes) -> bytes:
    """`data` as raw deflate blocks that end on a full flush: blocks like that can be put one
    after the other in any order, as they never refer back to what came before them."""
    z = zlib.compressobj(1, zlib.DEFLATED, -15)
    return z.compress(data) + z.flush(zlib.Z_FULL_FLUSH)


def _zeros(row: int, count: int) -> bytes:
    """`count` black rows of `row` bytes, deflated; whole runs of them come from a cache."""
    block = _zero_blocks.get(row)
    if block is None:
        block = _zero_blocks.setdefault(row, _deflate(bytes(row * _ZERO_ROWS)))
    whole, rest = divmod(count, _ZERO_ROWS)
    return block * whole + (_deflate(bytes(row * rest)) if rest else b"")


def _png(w, h, block, x0, y0) -> bytes:
    """A black PNG of `w` by `h` with the pixels of `block` at (`x0`, `y0`).

    The rows are unfiltered (filter byte 0). A layer is mostly black, so the zlib stream is put
    together by hand: the black rows above and below the block are deflated blocks from a cache,
    only the rows with something in them are compressed, and the checksum of the black bytes
    follows from a formula (adler32 of zeros leaves its first sum as it is and adds that sum to
    the second once per byte). No image of the whole screen is ever made in memory.
    """
    bh, bw = block.shape
    row = w + 1
    a, b = 1, 0                                  # the two sums of adler32
    parts = [b"\x78\x01"]                        # zlib header: deflate, fastest

    def black(count):
        nonlocal b
        if count > 0:
            parts.append(_zeros(row, count))
            b = (b + count * row * a) % _ADLER

    black(y0)
    if bh:
        rows = np.zeros((bh, row), dtype=np.uint8)
        rows[:, 1 + x0:1 + x0 + bw] = block
        data = memoryview(rows).cast("B")              # no copy: zlib reads the array
        parts.append(_deflate(data))
        s = zlib.adler32(data, (b << 16) | a)
        a, b = s & 0xFFFF, s >> 16
    black(h - y0 - bh)
    parts.append(b"\x03\x00")                    # an empty final block
    parts.append(struct.pack(">I", (b << 16) | a))
    header = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8 bit grayscale
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", b"".join(parts)) + _chunk(b"IEND", b""))


def layer_png(polygons, printer: ResinPrinter, transform: QTransform) -> bytes:
    """The PNG of one layer, drawn only where it has something: the rest of the screen is black."""
    w, h = printer.pixels_x, printer.pixels_y
    pts = [np.asarray(p, dtype=float) for p in polygons if len(p)]
    if not pts:
        return _png(w, h, np.zeros((0, 0), dtype=np.uint8), 0, 0)
    allp = np.concatenate(pts)
    corners = [transform.map(QPointF(float(x), float(y)))
               for x in (allp[:, 0].min(), allp[:, 0].max())
               for y in (allp[:, 1].min(), allp[:, 1].max())]
    x0 = max(0, int(math.floor(min(c.x() for c in corners))) - 2)
    x1 = min(w, int(math.ceil(max(c.x() for c in corners))) + 2)
    y0 = max(0, int(math.floor(min(c.y() for c in corners))) - 2)
    y1 = min(h, int(math.ceil(max(c.y() for c in corners))) + 2)
    if x1 <= x0 or y1 <= y0:
        return _png(w, h, np.zeros((0, 0), dtype=np.uint8), 0, 0)
    img = QImage(x1 - x0, y1 - y0, QImage.Format_Grayscale8)
    img.fill(0)
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    for poly in pts:
        path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
        path.closeSubpath()
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setTransform(transform * QTransform.fromTranslate(-x0, -y0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(Qt.white))
    p.drawPath(path)
    p.end()
    return _png(w, h, _pixels(img).copy(), x0, y0)


def layer_heights(top: float, layer_height: float) -> list[float]:
    """The middle of every layer up to `top`: where the parts are cut."""
    n = max(1, int(math.ceil(top / layer_height - 1e-9)))
    return [(i + 0.5) * layer_height for i in range(n)]


def place_on_plate(parts, printer: ResinPrinter):
    """The parts moved so their common centre is the middle of the plate, the lowest on z = 0."""
    boxes = [np.asarray(m.bounding_box(), dtype=float) for m in parts]
    lo = np.min([b[:3] for b in boxes], axis=0)
    hi = np.max([b[3:] for b in boxes], axis=0)
    shift = (0.5 * printer.width - 0.5 * (lo[0] + hi[0]),
             0.5 * printer.depth - 0.5 * (lo[1] + hi[1]), -lo[2])
    return [m.translate(shift) for m in parts]


def _ini(values: dict) -> str:
    return "".join(f"{k} = {v}\n" for k, v in sorted(values.items()))


def print_seconds(layers: int, settings: ResinSettings) -> float:
    bottom = min(layers, settings.bottom_layers)
    return (bottom * settings.bottom_exposure + (layers - bottom) * settings.exposure
            + layers * LIFT_SECONDS)


def write_sl1(path, printer: ResinPrinter, profile: ResinPrint | None, settings: ResinSettings,
              parts, on_progress=None, workers=None) -> ResinResult:
    """Slice `parts` (manifolds in mm on the plate, already placed) into an SL1 archive."""
    solid = mf.Manifold.batch_boolean(list(parts), mf.OpType.Add) if len(parts) > 1 else parts[0]
    top = float(solid.bounding_box()[5])
    if top > printer.height + 1e-6:
        raise ValueError(f"the print is {top:.1f} mm high; "
                         f"the printer prints {printer.height:g} mm")
    zs = layer_heights(top, settings.layer_height)
    transform = image_transform(printer)

    def one(z):
        return layer_png(solid.slice(z).to_polygons(), printer, transform)

    seconds = print_seconds(len(zs), settings)
    ml = solid.volume() / 1000.0
    values = dict(profile.values if profile else {})
    values.update(printer.values)
    values.update({"layer_height": f"{settings.layer_height:g}",
                   "initial_layer_height": f"{settings.layer_height:g}",
                   "exposure_time": f"{settings.exposure:g}",
                   "initial_exposure_time": f"{settings.bottom_exposure:g}",
                   "faded_layers": str(settings.bottom_layers)})
    config = {"action": "print", "jobDir": JOB, "expTime": f"{settings.exposure:g}",
              "expTimeFirst": f"{settings.bottom_exposure:g}", "expUserProfile": "0",
              "fileCreationTimestamp": time.strftime("%Y-%m-%d at %H:%M:%S UTC", time.gmtime()),
              "hollow": "0", "layerHeight": f"{settings.layer_height:g}",
              "materialName": "Volura resin", "numFade": str(settings.bottom_layers),
              "numFast": str(len(zs)), "numSlow": "0",
              "printProfile": profile.id if profile else "Volura",
              "printTime": f"{seconds:.0f}", "printerModel": "SL1",
              "printerProfile": printer.name, "printerVariant": "default",
              "prusaSlicerVersion": "Volura", "usedMaterial": f"{ml:.3f}"}
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("config.ini", _ini(config))
        z.writestr("prusaslicer.ini", _ini(values))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, data in enumerate(pool.map(one, zs)):
                # PNG is compressed already: stored as it is
                z.writestr(zipfile.ZipInfo(f"{JOB}{i:05d}.png"), data,
                           compress_type=zipfile.ZIP_STORED)
                if on_progress:
                    on_progress((i + 1) / len(zs))
    return ResinResult(path, len(zs), seconds, ml, path)


def convert(archive: str, printer: ResinPrinter, out_path: str, uvtools: str,
            timeout=None) -> str:
    """The SL1 `archive` converted by UVtools into the format of `printer` at `out_path`."""
    if printer.file_format == "sl1":
        with open(archive, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        return out_path
    cmd = converter(uvtools)
    if cmd is None:
        raise RuntimeError("UVtools was not found")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    # UVtools picks the format by the extension too: it writes next to the result under the
    # printer's own extension, and the file is moved to its place afterwards
    made = out_path + ".volura-tmp." + printer.extension

    def run():
        return subprocess.run([cmd, "convert", archive, printer.file_format, made],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              creationflags=flags, timeout=timeout, env=converter_environment())

    proc = run()
    text = proc.stderr + proc.stdout
    if proc.returncode != 0 and "VersionNotFound" in text:
        # a profile that names a version of the format UVtools does not know: without it,
        # UVtools writes its own default version of that format
        _drop_file_version(archive)
        proc = run()
        text = proc.stderr + proc.stdout
    if proc.returncode != 0 or not os.path.isfile(made):
        if os.path.exists(made):
            os.remove(made)
        lines = [ln.strip() for ln in text.splitlines()
                 if ln.strip() and not ln.strip().startswith("at ") and "[" not in ln]
        raise RuntimeError(lines[-1] if lines else f"UVtools stopped with code {proc.returncode}")
    os.replace(made, out_path)
    return out_path


def _drop_file_version(archive: str):
    """The archive again, with the FILEVERSION_ keyword taken out of the printer notes."""
    with zipfile.ZipFile(archive) as z:
        items = [(info, z.read(info.filename)) for info in z.infolist()]
    with zipfile.ZipFile(archive, "w") as z:
        for info, data in items:
            if info.filename == "prusaslicer.ini":
                text = data.decode("utf-8", errors="replace")
                data = re.sub(r"\\nFILEVERSION_\d+", "", text).encode("utf-8")
            z.writestr(info, data)


def slice_resin(parts, printer: ResinPrinter, profile: ResinPrint | None,
                settings: ResinSettings, out_path: str, uvtools: str | None,
                on_progress=None) -> ResinResult:
    """Slice `parts` (manifolds) for `printer` into the printer's own file at `out_path`.

    The parts keep their places to each other and land in the middle of the plate. The SL1
    archive is made next to the result and removed again once it is converted.
    """
    if not parts:
        raise RuntimeError("there is nothing to slice")
    placed = place_on_plate(parts, printer)
    fd, archive = tempfile.mkstemp(suffix=".sl1", prefix="volura-")
    os.close(fd)
    try:
        result = write_sl1(archive, printer, profile, settings, placed, on_progress)
        convert(archive, printer, out_path, uvtools)
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass
    result.path, result.archive = out_path, ""
    return result


def read_layer(archive_bytes: bytes, index: int) -> QImage:
    """One layer image back out of SL1 archive bytes (for the preview and the tests)."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        return QImage.fromData(z.read(f"{JOB}{index:05d}.png"))
