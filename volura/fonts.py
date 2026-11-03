"""Lettertypen voor tekst in 2D en 3D.

Qt haalt zijn lettertypen van het systeem; op sommige systemen (een kaal Linux,
een afgeschermde Windows-installatie, of het offscreen-platform zonder
lettertype-bibliotheek) is de databank leeg en tekent `QPainterPath.addText`
vierhoeken in plaats van letters. `ensure_fonts()` meldt dan de lettertypebestanden
van het systeem zelf aan, zodat tekst op het werkblad overal hetzelfde werkt.

`text_loops()` maakt van een tekst de contouren: gesloten lussen in millimeters met
Y omhoog, met de basislijn op y = 0 en de buitenrand tegen de klok in, zodat de
gaten van o, a en e als negatieve lussen meegaan.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import manifold3d as mf

#: wat het werkblad het liefst tekent, in volgorde van voorkeur
PREFERRED = ("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Noto Sans",
             "FreeSans", "Segoe UI", "Verdana", "Tahoma")

#: bestanden die het eerst geprobeerd worden als de lettertype-databank leeg is
PREFERRED_FILES = ("arial", "helvetica", "dejavusans", "liberationsans", "notosans",
                   "freesans", "segoeui", "verdana", "tahoma")

FONT_SUFFIXES = (".ttf", ".otf")
MAX_FILES = 16          # niet het hele lettertypeboek aanmelden, dat duurt te lang


def font_dirs() -> list[Path]:
    """De mappen waar op dit platform lettertypen staan."""
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
    """De lettertypen die Qt nu kent (leeg zonder QApplication of zonder lettertypen)."""
    from PySide6.QtGui import QFontDatabase, QGuiApplication
    if QGuiApplication.instance() is None:
        return []
    try:
        return list(QFontDatabase.families())
    except Exception:
        return []


def ensure_fonts() -> list[str]:
    """De beschikbare lettertypen; meldt bestanden van het systeem aan als er geen zijn."""
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
    """De eerste beschikbare lettertype-naam uit `preferred`, of wat er wél is."""
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
    """De contouren van `text` in millimeters: hoofdletterhoogte `height`, Y omhoog.

    Eén lus per buitenrand of gat; de buitenrand loopt tegen de klok in, gaten met
    de klok mee (dezelfde afspraak als `vector2d.regions`). Leeg als de tekst of het
    lettertype niets oplevert.
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
    if sum(_area(L) for L in out) < 0.0:        # alles omgekeerd: spiegel de whole tekst
        out = [np.ascontiguousarray(L[::-1].copy(), dtype=float) for L in out]
    return out


def _area(loop) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
