"""Writes LICENSE.txt and THIRD_PARTY_LICENSES.txt next to the standalone version.

    python packaging/licenses.py dist/Volura

Called by build.py. The licenses of what PyInstaller bundles (Qt and PySide6,
manifold3d, numpy, PyOpenGL and Python itself) and of the slicer next to it (CuraEngine
and the Cura printer profiles) must ship with the distribution. The texts come from the
installed packages; what is not in there (LGPL/GPL for Qt, AGPL for CuraEngine, the
PyOpenGL license) is in packaging/licenses/.
"""
from __future__ import annotations

import importlib.metadata as md
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent / "licenses"
#: the Cura release the slicer comes from (tools/fetch_curaengine.py)
CURA_VERSION = "5.14.0"
RULE = "=" * 78

QT_NOTICE = """\
Volura uses Qt 6 through PySide6 and shiboken6 under the GNU Lesser General Public
License version 3 (LGPL v3). Qt is (c) The Qt Company Ltd. and other contributors.

- The Qt, PySide6 and shiboken6 libraries are separate shared libraries
  (.dll / .so) in the _internal folder and are not statically linked. You may
  replace them with your own (modified) build of the same Qt version.
- Qt source code:           https://code.qt.io  and  https://download.qt.io
- PySide6 source code:      https://code.qt.io/cgit/pyside/pyside-setup.git
- Qt itself contains third-party components; the overview with their licenses is at
  https://doc.qt.io/qt-6/licenses-used-in-qt.html

Below follows the text of the LGPL v3, and after that the GPL v3 that the LGPL v3
builds on."""

CURA_NOTICE = """\
File > Slice uses CuraEngine and the printer profiles of UltiMaker Cura {version}.

- CuraEngine is (c) UltiMaker and contributors, under the GNU Affero General Public
  License version 3 (AGPL v3). It is a separate program in the curaengine folder,
  started by Volura; Volura is not linked to it. It is the unchanged CuraEngine of the
  official Cura {version} release, with the libraries it loads (Arcus, polyclipping,
  cura-formulae-engine, oneTBB and the Microsoft C++ runtime on Windows).
  Source code: https://github.com/Ultimaker/CuraEngine/tree/{version}
- The printer definitions, nozzle variants and quality profiles in
  curaengine/resources are (c) UltiMaker and the printer makers who contributed them,
  under the GNU Lesser General Public License version 3 (LGPL v3, text above).
  Source: https://github.com/Ultimaker/Cura/tree/{version}/resources
- The material profiles in curaengine/resources/materials come from
  https://github.com/Ultimaker/fdm_materials and are in the public domain (CC0 1.0).

Below follows the text of the AGPL v3."""


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
        "Volura contains or uses the following third-party open source software.\n"
        "Volura itself is under the MIT license, see LICENSE.txt.\n\n",
        "Component                  Version     License\n"
        f"Qt 6 / PySide6 / shiboken6 {qt:<11} LGPL-3.0-only\n"
        f"manifold3d                 {version('manifold3d') or '?':<11} Apache-2.0\n"
        f"NumPy                      {version('numpy') or '?':<11} BSD-3-Clause (with parts under 0BSD, MIT, Zlib, CC0)\n"
        f"PyOpenGL                   {version('PyOpenGL') or '?':<11} BSD-style\n"
        f"Python                     {sys.version.split()[0]:<11} PSF-2.0\n",
        f"CuraEngine                 {CURA_VERSION:<11} AGPL-3.0 (a separate program)\n"
        f"Cura printer profiles      {CURA_VERSION:<11} LGPL-3.0 (materials CC0-1.0)\n",
    ]
    parts.append(section(f"Qt 6 / PySide6 {qt}", QT_NOTICE))
    parts.append(section("GNU LESSER GENERAL PUBLIC LICENSE, version 3",
                         (HERE / "LGPL-3.0.txt").read_text(encoding="utf-8")))
    parts.append(section("GNU GENERAL PUBLIC LICENSE, version 3",
                         (HERE / "GPL-3.0.txt").read_text(encoding="utf-8")))
    parts.append(section(f"CuraEngine and the Cura printer profiles {CURA_VERSION}",
                         CURA_NOTICE.format(version=CURA_VERSION)))
    parts.append(section("GNU AFFERO GENERAL PUBLIC LICENSE, version 3",
                         (HERE / "AGPL-3.0.txt").read_text(encoding="utf-8")))
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
