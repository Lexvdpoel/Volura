"""Makes the icon files of Volura.

    python packaging/make_icons.py                 # icon from volura/icons.py
    python packaging/make_icons.py --from assets/volura.ico
    python packaging/make_icons.py --size 512 --out /tmp

This makes `assets/volura.png` of 256 x 256 pixels (the icon that goes into
`~/.local/share/icons/hicolor/256x256/apps` on Linux) and `assets/volura.ico` for
the Windows build. `--from` takes an existing icon or image and scales it to the
right size; without `--from` the script draws the icon from `volura/icons.py`.
The script works without a display (Qt offscreen).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QByteArray, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

DEFAULT_SIZE = 256


def from_svg(size):
    """The app icon as it is also drawn in the toolbar."""
    from volura.icons import SVG

    renderer = QSvgRenderer(QByteArray(f'<svg xmlns="http://www.w3.org/2000/svg" '
                                       f'viewBox="0 0 24 24">{SVG["app"]}</svg>'.encode()))
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def from_file(path, size):
    """Scale an existing icon (volura.ico) or image to the right size."""
    image = QImage(str(path))
    if image.isNull():
        raise SystemExit(f"{path} is not a readable image")
    if image.width() != size or image.height() != size:
        image = image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return image


def write(image, out, size):
    out.mkdir(parents=True, exist_ok=True)
    png = out / "volura.png"
    ico = out / "volura.ico"
    if not image.save(str(png), "PNG"):
        raise SystemExit(f"{png} not written")
    if not image.save(str(ico), "ICO"):
        raise SystemExit(f"{ico} not written")
    check = QImage(str(png))
    print(f"{png}  ({check.width()} x {check.height()})")
    print(f"{ico}  ({check.width()} x {check.height()})")
    if (check.width(), check.height()) != (size, size):
        raise SystemExit(f"{png} is {check.width()} x {check.height()}, not {size} x {size}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", help="existing icon or image, for example assets/volura.ico")
    ap.add_argument("--out", default=str(ROOT / "assets"), help="folder for volura.png and volura.ico")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="square size of the PNG (256)")
    args = ap.parse_args(argv)

    size = max(16, args.size)
    if QGuiApplication.instance() is None:
        QGuiApplication(sys.argv[:1])
    image = from_file(Path(args.source), size) if args.source else from_svg(size)
    write(image, Path(args.out), size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
