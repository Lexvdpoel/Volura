"""Fonts for text in 2D and 3D.

Qt gets its fonts from the system; on some systems (a bare Linux, a locked-down
Windows installation, or the offscreen platform without a font library) the
database is empty and `QPainterPath.addText` draws boxes instead of letters.
`ensure_fonts()` then registers the system's font files itself, so text on the
worksheet works the same everywhere.

`text_loops()` turns a text into its outlines: closed loops in millimetres with
Y up, the baseline at y = 0 and the outer boundary counter clockwise, so the
holes of o, a and e come along as negative loops.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import manifold3d as mf

#: what the worksheet prefers to draw with, in order of preference
PREFERRED = ("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Noto Sans",
             "FreeSans", "Segoe UI", "Verdana", "Tahoma")

#: files tried first when the font database is empty
PREFERRED_FILES = ("arial", "helvetica", "dejavusans", "liberationsans", "notosans",
                   "freesans", "segoeui", "verdana", "tahoma")

FONT_SUFFIXES = (".ttf", ".otf")
MAX_FILES = 16          # don't register the whole font collection, that takes too long


def font_dirs() -> list[Path]:
    """The folders that hold fonts on this platform."""
    if os.name == "nt":
        root = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or ""
        dirs = [Path(root) / "Fonts"] if root else []
    elif sys.platform == "darwin":
        dirs = [Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts"]
    else:
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts"]
    return [d for d in dirs if d.is_dir()]


def _font_files() -> list[Path]:
    files = []
    for folder in font_dirs():
        try:
            files += [p for p in sorted(folder.iterdir())
                      if p.suffix.lower() in FONT_SUFFIXES]
        except OSError:
            continue
    def rank(p: Path):
        stem = p.stem.lower().replace(" ", "")
        for i, want in enumerate(PREFERRED_FILES):
            if stem.startswith(want):
                return (0, i, stem)
        return (1, 0, stem)
    files.sort(key=rank)
    return files[:MAX_FILES]


def families() -> list[str]:
    """The fonts Qt knows now (empty without a QApplication or without fonts)."""
    from PySide6.QtGui import QFontDatabase, QGuiApplication
    if QGuiApplication.instance() is None:
        return []
    try:
        return list(QFontDatabase.families())
    except Exception:
        return []


def ensure_fonts() -> list[str]:
    """The available fonts; registers the system's files when there are none."""
    got = families()
    if got:
        return got
    from PySide6.QtGui import QFontDatabase, QGuiApplication
    if QGuiApplication.instance() is None:
        return []
    for path in _font_files():
        try:
            if QFontDatabase.addApplicationFont(str(path)) >= 0 and families():
                break
        except Exception:
            continue
    return families()


def default_family(preferred=PREFERRED) -> str:
    """The first available font name from `preferred`, or whatever there is."""
    got = ensure_fonts()
    if not got:
        return ""
    low = {f.lower(): f for f in got}
    for want in preferred:
        if want.lower() in low:
            return low[want.lower()]
    return got[0]


def has_fonts() -> bool:
    return bool(ensure_fonts())


def text_loops(text, family=None, height=10.0, bold=False) -> list[np.ndarray]:
    """The outlines of `text` in millimetres: cap height `height`, Y up.

    One loop per outer boundary or hole; the outer boundary runs counter clockwise,
    holes clockwise (the same convention as `vector2d.regions`). Empty when the text
    or the font yields nothing.
    """
    from PySide6.QtGui import QFont, QFontMetricsF, QPainterPath
    text = str(text or "")
    if not text.strip() or float(height) <= 0.0:
        return []
    font = QFont(family or default_family())
    font.setPixelSize(200)
    font.setBold(bool(bold))
    path = QPainterPath()
    path.addText(0, 0, font, text)
    cap = QFontMetricsF(font).capHeight() or 140.0
    s = float(height) / float(cap)
    raw = []
    for poly in path.toSubpathPolygons():
        pts = np.array([[pt.x() * s, -pt.y() * s] for pt in poly], dtype=float)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) >= 3:
            raw.append(np.ascontiguousarray(pts))
    if not raw:
        return []
    cs = mf.CrossSection(raw, mf.FillRule.NonZero)
    loops = [np.asarray(q, dtype=float) for q in cs.to_polygons()]
    loops = [L for L in loops if len(L) >= 3 and abs(_area(L)) > 1e-9]
    if not loops:
        return []
    allp = np.concatenate(loops)
    x0 = float(allp[:, 0].min())
    out = [np.ascontiguousarray(L - np.array([x0, 0.0]), dtype=float) for L in loops]
    if sum(_area(L) for L in out) < 0.0:        # all reversed: flip the whole text
        out = [np.ascontiguousarray(L[::-1].copy(), dtype=float) for L in out]
    return out


def _area(loop) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
