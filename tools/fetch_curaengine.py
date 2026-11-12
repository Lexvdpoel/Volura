"""Fetches CuraEngine and the Cura printer profiles into ./curaengine for slicing.

    python tools/fetch_curaengine.py [--package FILE]

UltiMaker publishes CuraEngine only inside the Cura download, so this takes the official Cura
release for this platform (the .msi on Windows, the AppImage on Linux), unpacks it into a
temporary folder without installing anything, and keeps what Volura slices with:

    curaengine/CuraEngine(.exe)       the engine (AGPL-3.0), with the libraries it needs
    curaengine/lib/                   on Linux: the shared libraries next to it
    curaengine/resources/             definitions, extruders, variants, quality, materials and
                                      texts/product_to_id.json (LGPL-3.0, materials CC0)

`--package` uses a Cura package you downloaded already instead of downloading it. The folder is
not part of the source tree (.gitignore); packaging/build.py puts it next to Volura, and the
tests use it when it is there.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CURA_VERSION = "5.14.0"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "curaengine"
RELEASES = f"https://github.com/Ultimaker/Cura/releases/download/{CURA_VERSION}/"
PACKAGES = {"win32": f"UltiMaker-Cura-{CURA_VERSION}-win64-X64.msi",
            "linux": f"UltiMaker-Cura-{CURA_VERSION}-linux-X64.AppImage"}
#: the libraries CuraEngine.exe loads from its own folder on Windows
WINDOWS_LIBS = ("tbb12.dll", "cura-formulae-engine.dll", "polyclipping.dll", "Arcus.dll",
                "MSVCP140.dll", "MSVCP140_ATOMIC_WAIT.dll", "VCRUNTIME140.dll",
                "VCRUNTIME140_1.dll")
#: the parts of Cura's resources folder that slicing needs
RESOURCE_DIRS = ("definitions", "extruders", "variants", "quality", "materials")


def download(name: str, folder: Path) -> Path:
    path = folder / name
    print(f"downloading {RELEASES + name}")
    with urllib.request.urlopen(RELEASES + name) as resp, open(path, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)
    return path


def unpack_msi(msi: Path, folder: Path) -> Path:
    """An administrative install: the files of the .msi, unpacked, nothing installed."""
    out = folder / "x"
    out.mkdir()
    rc = subprocess.call(["msiexec", "/a", str(msi), "/qn", f"TARGETDIR={out}"])
    if rc != 0:
        raise SystemExit(f"msiexec could not unpack {msi.name} (code {rc}); "
                         "use a folder with a short path for TEMP")
    return next(p.parent for p in out.rglob("CuraEngine.exe"))


def unpack_appimage(image: Path, folder: Path) -> Path:
    image.chmod(0o755)
    subprocess.check_call([str(image), "--appimage-extract"], cwd=folder,
                          stdout=subprocess.DEVNULL)
    return next(p.parent for p in (folder / "squashfs-root").rglob("CuraEngine")
                if p.is_file())


def linux_libs(engine: Path, root: Path) -> list[Path]:
    """The shared libraries of the engine that come from the Cura package, not from the system."""
    env = dict(os.environ)
    dirs = sorted({str(p.parent) for p in root.rglob("*.so*")})
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs)
    out = subprocess.run(["ldd", str(engine)], env=env, capture_output=True, text=True).stdout
    libs = []
    for line in out.splitlines():
        if "=>" in line:
            path = line.split("=>")[1].split("(")[0].strip()
            if path and str(root) in path:
                libs.append(Path(path))
    return libs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", help="a Cura .msi or AppImage that was downloaded already")
    args = ap.parse_args()
    name = PACKAGES.get(sys.platform)
    if name is None:
        raise SystemExit(f"no CuraEngine for {sys.platform}")
    with tempfile.TemporaryDirectory(prefix="cura-") as tmp:
        tmp = Path(tmp)
        package = Path(args.package) if args.package else download(name, tmp)
        if sys.platform == "win32":
            base = unpack_msi(package, tmp)
            engine, libs = base / "CuraEngine.exe", [base / n for n in WINDOWS_LIBS]
            resources = base / "share" / "cura" / "resources"
        else:
            base = unpack_appimage(package, tmp)
            engine = base / "CuraEngine"
            root = tmp / "squashfs-root"
            libs = linux_libs(engine, root)
            resources = next(p for p in root.rglob("resources") if (p / "definitions").is_dir())
        if TARGET.exists():
            shutil.rmtree(TARGET)
        TARGET.mkdir()
        shutil.copy2(engine, TARGET)
        lib_dir = TARGET if sys.platform == "win32" else TARGET / "lib"
        lib_dir.mkdir(exist_ok=True)
        for lib in libs:
            if lib.is_file():
                shutil.copy2(lib, lib_dir / lib.name)
        for d in RESOURCE_DIRS:
            shutil.copytree(resources / d, TARGET / "resources" / d)
        (TARGET / "resources" / "texts").mkdir()
        shutil.copy2(resources / "texts" / "product_to_id.json", TARGET / "resources" / "texts")
        (TARGET / "VERSION").write_text(CURA_VERSION + "\n", encoding="utf-8")
    print(f"CuraEngine {CURA_VERSION} is in {TARGET}")


if __name__ == "__main__":
    main()
