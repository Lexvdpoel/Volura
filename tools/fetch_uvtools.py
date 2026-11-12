"""Fetches UVtools' command line and its resin printer profiles into ./uvtools for resin slicing.

    python tools/fetch_uvtools.py [--package FILE]

Volura slices for resin (MSLA) printers itself and writes the open Prusa SL1 format; the printers
want their own formats (.ctb, .goo, .pwmx, ...), and UVtools converts between all of them. This
takes the official UVtools release for this platform, keeps only what its command line program
`UVtoolsCmd` needs (not the program with its window), and the printer profiles that come with it:

    uvtools/UVtoolsCmd(.exe)           the converter (AGPL-3.0), with the libraries it loads
    uvtools/profiles/printer/*.ini     the resin printers, as PrusaSlicer printer profiles
    uvtools/profiles/sla_print/*.ini   print profiles: layer height and supports
    uvtools/VERSION

`--package` uses a UVtools release zip you downloaded already. The folder is not part of the
source tree (.gitignore); packaging/build.py puts it next to Volura, and the tests use it when it
is there.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UVTOOLS_VERSION = "7.1.2"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "uvtools"
RELEASES = f"https://github.com/sn4k3/UVtools/releases/download/v{UVTOOLS_VERSION}/"
PACKAGES = {"win32": f"UVtools_win-x64_v{UVTOOLS_VERSION}.zip",
            "linux": f"UVtools_linux-x64_v{UVTOOLS_VERSION}.zip"}
#: the program of the command line, by platform
COMMAND = "UVtoolsCmd.exe" if sys.platform == "win32" else "UVtoolsCmd"


def download(name: str, folder: Path) -> Path:
    path = folder / name
    print(f"downloading {RELEASES + name}")
    with urllib.request.urlopen(RELEASES + name) as resp, open(path, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)
    return path


def command_files(base: Path) -> list[str]:
    """The files `UVtoolsCmd` loads: what its .deps.json lists for this platform, and itself.

    The release is published self-contained, so every assembly, native library and the .NET
    runtime lie flat in one folder; the window of UVtools (Avalonia and the like) is left out.
    """
    deps = json.loads((base / "UVtoolsCmd.deps.json").read_text(encoding="utf-8"))
    target = next(v for k, v in deps["targets"].items() if "/" in k)    # the one with a runtime
    names = {os.path.basename(path) for info in target.values()
             for key in ("runtime", "native") for path in info.get(key, {})}
    names |= {p.name for p in base.iterdir() if p.name.startswith("UVtoolsCmd")}
    # OpenCV's native library, which reads and writes the layer images: on Linux the release
    # does not list it in the .deps.json, but loads it all the same
    names |= {p.name for p in base.iterdir() if p.name.startswith(("cvextern", "libcvextern"))}
    return sorted(n for n in names if (base / n).is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", help="a UVtools release zip that was downloaded already")
    args = ap.parse_args()
    name = PACKAGES.get(sys.platform)
    if name is None:
        raise SystemExit(f"no UVtools for {sys.platform}")
    with tempfile.TemporaryDirectory(prefix="uvtools-", ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        package = Path(args.package) if args.package else download(name, tmp)
        base = tmp / "x"
        with zipfile.ZipFile(package) as z:
            z.extractall(base)
        if not (base / "UVtoolsCmd.deps.json").is_file():
            base = next(p.parent for p in base.rglob("UVtoolsCmd.deps.json"))
        if TARGET.exists():
            shutil.rmtree(TARGET)
        TARGET.mkdir()
        for n in command_files(base):
            shutil.copy2(base / n, TARGET / n)
        if sys.platform != "win32":
            (TARGET / COMMAND).chmod(0o755)
        profiles = base / "Assets" / "PrusaSlicer"
        for kind in ("printer", "sla_print"):
            shutil.copytree(profiles / kind, TARGET / "profiles" / kind)
        shutil.copy2(base / "LICENSE", TARGET / "LICENSE")
        (TARGET / "VERSION").write_text(UVTOOLS_VERSION + "\n", encoding="utf-8")
    print(f"UVtools {UVTOOLS_VERSION} is in {TARGET}")


if __name__ == "__main__":
    main()
