"""Resin (MSLA) printers and print settings for File > Slice for Resin.

No Qt in this file. The profiles are PrusaSlicer's .ini profiles for resin printers that come with
UVtools (tools/fetch_uvtools.py puts them in uvtools/profiles): a printer profile says how large
the screen of the printer is in millimetres and in pixels, whether the image is mirrored, how high
it prints and - in its notes - the file format the printer reads and the lift and light values
UVtools sets in that file. A print profile ("Universal 0.05 - Medium Supports") gives the layer
height and the sizes of the supports and the pad.

The exposure is not in these profiles: it depends on the resin. `ResinSettings` holds what you
choose for it, with defaults that suit a standard resin on a mono screen.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

#: an environment variable that points at a uvtools folder instead of the bundled one
UVTOOLS_ENV = "VOLURA_UVTOOLS"
#: the program that converts between the resin formats
COMMAND = "UVtoolsCmd.exe" if sys.platform == "win32" else "UVtoolsCmd"


def find_uvtools() -> str | None:
    """The uvtools folder with the converter and the profiles: next to Volura, or in the source."""
    env = os.environ.get(UVTOOLS_ENV)
    candidates = [env] if env else []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), "uvtools"))
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "uvtools"))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "uvtools"))
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "profiles", "printer")):
            return c
    return None


def converter(folder: str | None) -> str | None:
    """The UVtools command line in `folder`, when it is there."""
    if not folder:
        return None
    path = os.path.join(folder, COMMAND)
    return path if os.path.isfile(path) else None


def read_ini(path: str) -> dict:
    """A PrusaSlicer .ini profile: `key = value` lines, comments and empty lines left out."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _float(value, default=0.0) -> float:
    try:
        return float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return default


def _bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


@dataclass
class ResinPrinter:
    """One resin printer: its screen, how high it prints, and the file it reads."""

    id: str
    name: str
    maker: str
    width: float                 # mm of screen along x
    depth: float                 # mm of screen along y
    pixels_x: int
    pixels_y: int
    height: float                # mm it prints at most
    mirror_x: bool
    mirror_y: bool
    portrait: bool
    file_format: str             # what UVtools writes: "ctb", "encrypted.ctb", "goo", ... or "sl1"
    values: dict = field(default_factory=dict)      # the whole profile, for the archive

    @property
    def pixel(self) -> tuple[float, float]:
        """mm of one pixel along x and y."""
        return self.width / self.pixels_x, self.depth / self.pixels_y

    @property
    def extension(self) -> str:
        """The file extension of what the printer reads."""
        fmt = self.file_format.lower()
        return fmt.split(".")[-1] if fmt != "encrypted.ctb" else "ctb"

    @property
    def custom_values(self) -> dict:
        """The values UVtools sets in the printer's file (lift heights and speeds, light)."""
        notes = self.values.get("printer_notes", "").replace("\\n", "\n")
        m = re.search(r"START_CUSTOM_VALUES\n(.*?)\nEND_CUSTOM_VALUES", notes, re.S)
        out = {}
        for line in (m.group(1).splitlines() if m else []):
            key, _, value = line.strip().partition("_")
            if key and value:
                out[key] = value
        return out


def _notes(values) -> str:
    return values.get("printer_notes", "").replace("\\n", "\n")


def read_printer(path: str) -> ResinPrinter | None:
    """The printer of a printer profile, or None for one that is not a resin printer."""
    v = read_ini(path)
    if v.get("printer_technology", "SLA") != "SLA":
        return None
    name = os.path.splitext(os.path.basename(path))[0]
    words = name.split()
    maker = words[1] if words[0] == "UVtools" and len(words) > 1 else words[0]
    notes = _notes(v)
    fmt = re.search(r"FILEFORMAT_([A-Za-z0-9.]+)", notes)
    try:
        px, py = int(v["display_pixels_x"]), int(v["display_pixels_y"])
        w, d = float(v["display_width"]), float(v["display_height"])
    except (KeyError, ValueError):
        return None
    return ResinPrinter(
        id=name, name=name, maker=maker, width=w, depth=d, pixels_x=px, pixels_y=py,
        height=_float(v.get("max_print_height"), 150.0),
        mirror_x=_bool(v.get("display_mirror_x")), mirror_y=_bool(v.get("display_mirror_y")),
        portrait=v.get("display_orientation", "landscape") == "portrait",
        file_format=(fmt.group(1).lower() if fmt else "sl1"), values=v)


@dataclass
class ResinPrint:
    """A print profile: the layer height and the sizes of the supports and the pad."""

    id: str
    layer_height: float
    level: str                   # "Light", "Medium" or "Heavy" supports
    values: dict = field(default_factory=dict)

    def number(self, key, default=0.0) -> float:
        return _float(self.values.get(key), default)


def read_print(path: str) -> ResinPrint:
    v = read_ini(path)
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(Light|Medium|Heavy)", name)
    return ResinPrint(name, _float(v.get("layer_height"), 0.05), m.group(1) if m else "Medium", v)


class ResinLibrary:
    """The resin printers and print profiles in a uvtools folder."""

    def __init__(self, folder: str):
        self.folder = folder
        self.printers = []
        pdir = os.path.join(folder, "profiles", "printer")
        for name in sorted(os.listdir(pdir)):
            if name.endswith(".ini"):
                p = read_printer(os.path.join(pdir, name))
                if p is not None:
                    self.printers.append(p)
        self.printers.sort(key=lambda p: (p.maker.lower(), p.name.lower()))
        sdir = os.path.join(folder, "profiles", "sla_print")
        self.prints = sorted((read_print(os.path.join(sdir, n)) for n in os.listdir(sdir)
                              if n.endswith(".ini")) if os.path.isdir(sdir) else [],
                             key=lambda p: (p.layer_height, p.level))

    def printer(self, pid: str) -> ResinPrinter | None:
        return next((p for p in self.printers if p.id == pid), None)

    @property
    def makers(self) -> list[str]:
        return sorted({p.maker for p in self.printers}, key=str.lower)

    def layer_heights(self) -> list[float]:
        return sorted({p.layer_height for p in self.prints})

    def print_profile(self, layer_height: float, level: str = "Medium") -> ResinPrint | None:
        """The print profile of that layer height and support level, or the nearest one."""
        if not self.prints:
            return None
        return min(self.prints, key=lambda p: (abs(p.layer_height - layer_height),
                                               p.level != level))


@dataclass
class ResinSettings:
    """What you choose for a resin print besides the printer: layers, light, supports, shell."""

    layer_height: float = 0.05
    exposure: float = 2.5                # s per layer
    bottom_exposure: float = 25.0        # s for each of the bottom layers
    bottom_layers: int = 4
    transition_layers: int = 0           # layers that fade from the bottom to the normal time
    supports: bool = True
    support_level: str = "Medium"        # Light, Medium or Heavy
    density: float = 100.0               # % of the support points of the profile
    elevation: float = 5.0               # mm the part stands above the pad on its supports
    pad: bool = True
    hollow: bool = False                 # the part as a shell, with drain holes
    wall: float = 2.5                    # mm thick that shell
    drain_holes: bool = True             # holes near the bottom, so the resin inside drains


def converter_environment() -> dict:
    """The environment UVtools runs in: without the ICU libraries of the system, which a .NET
    program otherwise needs on Linux and which not every system has."""
    env = dict(os.environ)
    env["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"
    return env


_problems: dict[str, str | None] = {}


def converter_problem(folder: str | None) -> str | None:
    """Why UVtools does not run here, or None when it does (asked once per folder)."""
    cmd = converter(folder)
    if cmd is None:
        return "UVtools was not found"
    if cmd not in _problems:
        import subprocess
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            proc = subprocess.run([cmd, "--version"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60,
                                  creationflags=flags, env=converter_environment())
            _problems[cmd] = None if proc.returncode == 0 else \
                ((proc.stderr or proc.stdout).strip().splitlines() or ["?"])[-1]
        except (OSError, subprocess.SubprocessError) as e:
            _problems[cmd] = str(e)
    return _problems[cmd]
