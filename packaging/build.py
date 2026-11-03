"""Builds a standalone version of Volura with PyInstaller.

    python packaging/build.py

    Windows -> dist/Volura/Volura.exe  + dist/Volura-windows-x86_64.zip
               + dist/Volura-windows-setup.exe (when Inno Setup 6 is present)
    Linux   -> dist/Volura/Volura      + dist/Volura-linux-x86_64.tar.gz
    macOS   -> dist/Volura/Volura.app  + dist/Volura-macos-<arch>.zip

PyInstaller cannot cross-compile: build on the platform you are building for.
The workflow in .github/workflows/build.yml does that for Windows and Linux.
The licenses (LICENSE.txt and THIRD_PARTY_LICENSES.txt, see licenses.py) are always
included. The Linux build also includes `volura.desktop` and `install.sh`; that script
puts Volura in ~/.local/opt/volura and the menu entry in ~/.local/share/applications.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
ASSETS = ROOT / "assets"
LINUX_DIR = ROOT / "packaging" / "linux"
LINUX_EXTRA = ("volura.desktop", "install.sh")
INSTALLER = ROOT / "packaging" / "windows" / "volura.iss"

SYSTEMS = {"win32": "windows", "linux": "linux", "darwin": "macos"}
ARCHES = {"amd64": "x86_64", "x64": "x86_64", "x86_64": "x86_64", "i386": "x86",
          "i686": "x86", "aarch64": "aarch64", "arm64": "aarch64"}


def target():
    """('windows', 'x86_64') / ('linux', 'x86_64') / ..."""
    system = SYSTEMS.get(sys.platform)
    if system is None:
        raise SystemExit(f"Volura does not build on {sys.platform}; build on Windows, Linux or macOS.")
    machine = (platform.machine() or "unknown").lower()
    return system, ARCHES.get(machine, machine)


def icon(system):
    """The icon PyInstaller can use; on Linux the desktop file sets the icon."""
    name = {"windows": "volura.ico", "macos": "volura.icns"}.get(system)
    path = ASSETS / name if name else None
    return path if path is not None and path.is_file() else None


def pyinstaller(system):
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed"]
    ic = icon(system)
    if ic is not None:
        cmd += ["--icon", str(ic)]
    cmd += [
        "--name", "Volura",
        "--add-data", f"{ASSETS / 'volura.png'}{os.pathsep}assets",
        "--collect-submodules", "OpenGL.platform",
        "--collect-submodules", "OpenGL.arrays",
        "--collect-submodules", "volura.locale",      # loaded by name (i18n.table)
        "--collect-binaries", "manifold3d",
        "main.py",
    ]
    print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def executable(system):
    """What PyInstaller produces as the program; on macOS that is an .app bundle."""
    if system == "windows":
        return DIST / "Volura" / "Volura.exe"
    if system == "macos":
        return DIST / "Volura" / "Volura.app"
    return DIST / "Volura" / "Volura"


def add_linux_files():
    """The build script itself is platform independent; only Linux gets a menu entry and icon."""
    app = DIST / "Volura"
    for name in LINUX_EXTRA:
        src = LINUX_DIR / name
        if src.is_file():
            shutil.copy2(src, app / name)
    png = ASSETS / "volura.png"
    if png.is_file():
        shutil.copy2(png, app / "volura.png")
    for f in (app / "install.sh", app / "volura.desktop"):
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o755)


def archive(system, arch):
    name = f"Volura-{system}-{arch}"
    app = DIST / "Volura"
    binary = executable(system)
    if not binary.exists():
        raise SystemExit(f"PyInstaller did not create {binary}; see the message above.")
    binary.chmod(binary.stat().st_mode | 0o755)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import licenses
    licenses.write(app)            # LICENSE.txt and THIRD_PARTY_LICENSES.txt next to the program
    if system == "linux":
        add_linux_files()
        fmt = "gztar"
    else:
        fmt = "zip"
    made = Path(shutil.make_archive(str(DIST / name), fmt, str(DIST), "Volura"))
    print(f"Done: {made}  ({round(made.stat().st_size / 1e6, 1)} MB)")
    return made


def find_iscc():
    """The Inno Setup compiler: on the PATH or in one of the usual places."""
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return Path(found)
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            path = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if path.is_file():
                return path
    return None


def version():
    text = (ROOT / "volura" / "__init__.py").read_text(encoding="utf-8")
    return text.split("__version__ = ", 1)[1].split()[0].strip("\"'")


def installer():
    """dist/Volura-windows-setup.exe: the installer, also used by the automatic update."""
    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup 6 not found: no installer made (only the zip).")
        return None
    subprocess.run([str(iscc), "/Q", f"/DAppVersion={version()}", str(INSTALLER)], check=True)
    made = DIST / "Volura-windows-setup.exe"
    print(f"Done: {made}  ({round(made.stat().st_size / 1e6, 1)} MB)")
    return made


def main():
    system, arch = target()
    print(f"Building Volura for {system}-{arch} (Python {platform.python_version()})")
    pyinstaller(system)
    archive(system, arch)
    if system == "windows":
        installer()
    print("Now unpack and start it"
          + (", then run ./install.sh for the menu entry" if system == "linux" else "."))


if __name__ == "__main__":
    main()
