"""Bouwt een zelfstandige versie van Volura met PyInstaller.

    python packaging/build.py

    Windows -> dist/Volura/Volura.exe  + dist/Volura-windows-x86_64.zip
               + dist/Volura-windows-setup.exe (als Inno Setup 6 er is)
    Linux   -> dist/Volura/Volura      + dist/Volura-linux-x86_64.tar.gz
    macOS   -> dist/Volura/Volura.app  + dist/Volura-macos-<arch>.zip

PyInstaller kan niet cross-compileren: bouw op het platform waarvoor je bouwt.
De workflow in .github/workflows/build.yml doet dat voor Windows en Linux.
De licenties (LICENSE.txt en THIRD_PARTY_LICENSES.txt, zie licenses.py) gaan altijd
mee. Bij de Linux-build gaan ook `volura.desktop` en `install.sh` mee; dat script zet
Volura in ~/.local/opt/volura en het menu-item in ~/.local/share/applications.
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
        raise SystemExit(f"Volura bouwt niet op {sys.platform}; bouw op Windows, Linux of macOS.")
    machine = (platform.machine() or "unknown").lower()
    return system, ARCHES.get(machine, machine)


def icon(system):
    """Het icoon dat PyInstaller kan gebruiken; op Linux zet het desktop-bestand het icoon."""
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
        "--collect-binaries", "manifold3d",
        "main.py",
    ]
    print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def executable(system):
    """Wat PyInstaller als programma neerzet; op macOS is dat een .app-bundel."""
    if system == "windows":
        return DIST / "Volura" / "Volura.exe"
    if system == "macos":
        return DIST / "Volura" / "Volura.app"
    return DIST / "Volura" / "Volura"


def add_linux_files():
    """Het build-script zelf is platformonafhankelijk; alleen Linux krijgt menu-item en icoon."""
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
        raise SystemExit(f"PyInstaller heeft {binary} niet gemaakt; kijk in de melding hierboven.")
    binary.chmod(binary.stat().st_mode | 0o755)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import licenses
    licenses.write(app)            # LICENSE.txt en THIRD_PARTY_LICENSES.txt naast het programma
    if system == "linux":
        add_linux_files()
        fmt = "gztar"
    else:
        fmt = "zip"
    made = Path(shutil.make_archive(str(DIST / name), fmt, str(DIST), "Volura"))
    print(f"Klaar: {made}  ({round(made.stat().st_size / 1e6, 1)} MB)")
    return made


def find_iscc():
    """De Inno Setup-compiler: op het PATH of op een van de gebruikelijke plekken."""
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
    """dist/Volura-windows-setup.exe: de installatie, ook gebruikt door de automatische update."""
    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup 6 niet gevonden: geen installatie gemaakt (alleen de zip).")
        return None
    subprocess.run([str(iscc), "/Q", f"/DAppVersion={version()}", str(INSTALLER)], check=True)
    made = DIST / "Volura-windows-setup.exe"
    print(f"Klaar: {made}  ({round(made.stat().st_size / 1e6, 1)} MB)")
    return made


def main():
    system, arch = target()
    print(f"Volura bouwen voor {system}-{arch} (Python {platform.python_version()})")
    pyinstaller(system)
    archive(system, arch)
    if system == "windows":
        installer()
    print("Nu uitpakken en starten"
          + (", daarna ./install.sh voor het menu" if system == "linux" else "."))


if __name__ == "__main__":
    main()
