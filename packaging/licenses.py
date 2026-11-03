"""Schrijft LICENSE.txt en THIRD_PARTY_LICENSES.txt naast de zelfstandige versie.

    python packaging/licenses.py dist/Volura

Wordt door build.py aangeroepen. De licenties van wat PyInstaller meelevert (Qt en
PySide6, manifold3d, numpy, PyOpenGL en Python zelf) moeten bij de verspreiding zitten.
De teksten komen uit de geïnstalleerde pakketten; wat daar niet in zit (LGPL/GPL voor
Qt, de licentie van PyOpenGL) staat in packaging/licenses/.
"""
from __future__ import annotations

import importlib.metadata as md
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent / "licenses"
RULE = "=" * 78

QT_NOTICE = """\
Volura gebruikt Qt 6 via PySide6 en shiboken6 onder de GNU Lesser General Public
License versie 3 (LGPL v3). Qt is (c) The Qt Company Ltd. en andere bijdragers.

- De Qt-, PySide6- en shiboken6-bibliotheken staan als losse gedeelde bibliotheken
  (.dll / .so) in de map _internal en zijn niet statisch gekoppeld. Je mag ze
  vervangen door een eigen (gewijzigde) versie van dezelfde Qt-versie.
- Broncode van Qt:          https://code.qt.io  en  https://download.qt.io
- Broncode van PySide6:     https://code.qt.io/cgit/pyside/pyside-setup.git
- Qt bevat zelf onderdelen van derden; het overzicht met hun licenties staat op
  https://doc.qt.io/qt-6/licenses-used-in-qt.html

Hieronder volgt de tekst van de LGPL v3, en daarna de GPL v3 waarop de LGPL v3
voortbouwt."""


def version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def dist_license_files(dist):
    """License/notice files that a wheel ships in its dist-info."""
    try:
        files = md.distribution(dist).files or []
    except md.PackageNotFoundError:
        return []
    out = []
    for f in files:
        name = str(f).replace("\\", "/")
        if ".dist-info/licenses/" in name or any(k in name.upper().rsplit("/", 1)[-1]
                                                  for k in ("LICENSE", "COPYING", "NOTICE", "AUTHORS")):
            path = Path(f.locate())
            if path.is_file() and "LicenseRef-Qt-Commercial" not in name:
                out.append((name, path))
    return out


def section(title, body):
    return f"\n\n{RULE}\n{title}\n{RULE}\n\n{body.strip()}\n"


def files_section(title, files):
    parts = [f"--- {name} ---\n\n{path.read_text(encoding='utf-8', errors='replace').strip()}"
             for name, path in files]
    return section(title, "\n\n".join(parts)) if parts else ""


def python_license():
    base = Path(sys.base_prefix)
    for p in (base / "LICENSE.txt",
              base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "LICENSE.txt"):
        if p.is_file():
            return p
    return None


def build_text():
    qt = version("PySide6") or version("PySide6_Essentials") or "?"
    parts = [
        "Volura bevat of gebruikt de volgende open source software van derden.\n"
        "Volura zelf valt onder de MIT-licentie, zie LICENSE.txt.\n\n",
        "Onderdeel                  Versie      Licentie\n"
        f"Qt 6 / PySide6 / shiboken6 {qt:<11} LGPL-3.0-only\n"
        f"manifold3d                 {version('manifold3d') or '?':<11} Apache-2.0\n"
        f"NumPy                      {version('numpy') or '?':<11} BSD-3-Clause (met onderdelen onder 0BSD, MIT, Zlib, CC0)\n"
        f"PyOpenGL                   {version('PyOpenGL') or '?':<11} BSD-stijl\n"
        f"Python                     {sys.version.split()[0]:<11} PSF-2.0\n",
    ]
    parts.append(section(f"Qt 6 / PySide6 {qt}", QT_NOTICE))
    parts.append(section("GNU LESSER GENERAL PUBLIC LICENSE, versie 3",
                         (HERE / "LGPL-3.0.txt").read_text(encoding="utf-8")))
    parts.append(section("GNU GENERAL PUBLIC LICENSE, versie 3",
                         (HERE / "GPL-3.0.txt").read_text(encoding="utf-8")))
    parts.append(files_section(f"manifold3d {version('manifold3d') or ''}", dist_license_files("manifold3d")))
    parts.append(files_section(f"NumPy {version('numpy') or ''}", dist_license_files("numpy")))
    ogl = [("PyOpenGL license.txt", HERE / "PyOpenGL-license.txt")]
    try:
        import OpenGL
        dlls = Path(OpenGL.__file__).parent / "DLLS"
        ogl += [(f"OpenGL/DLLS/{p.name}", p) for p in sorted(dlls.glob("*COPYING*")) if p.is_file()]
    except ImportError:
        pass
    parts.append(files_section(f"PyOpenGL {version('PyOpenGL') or ''}", ogl))
    py = python_license()
    if py is not None:
        parts.append(files_section(f"Python {sys.version.split()[0]}", [("LICENSE.txt", py)]))
    return "".join(parts).lstrip() + "\n"


def write(app_dir):
    app_dir = Path(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", app_dir / "LICENSE.txt")
    out = app_dir / "THIRD_PARTY_LICENSES.txt"
    out.write_text(build_text(), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist" / "Volura"))
