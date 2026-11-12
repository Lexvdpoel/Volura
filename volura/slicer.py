"""Slicing with CuraEngine: from parts and a printer profile to G-code.

No Qt in this file. It writes the parts as STL files, the worked-out settings of
`cura_profiles.Stacks` as CuraEngine's resolved-settings file (`-r`), runs CuraEngine and reads
back what the G-code says about the print: how long it takes and how much filament it uses.

CuraEngine places a model relative to the middle of the bed, unless the printer has its origin in
the middle already (`machine_center_is_zero`), so `slice_parts` moves the parts there: they stay
where they stand relative to each other and their centre lands in the centre of the bed.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

#: the name of the engine program
ENGINE = "CuraEngine.exe" if sys.platform == "win32" else "CuraEngine"
#: an environment variable that points at a CuraEngine to use instead of the bundled one
ENGINE_ENV = "VOLURA_CURAENGINE"
#: an environment variable that points at a Cura `resources` folder instead of the bundled one
RESOURCES_ENV = "VOLURA_CURA_RESOURCES"


def _bundle_dirs():
    """Where the bundled slicer can be: next to the program, or in the source tree's folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.dirname(sys.executable))
        if hasattr(sys, "_MEIPASS"):
            roots.append(sys._MEIPASS)
    roots.append(os.path.dirname(here))
    return [os.path.join(r, "curaengine") for r in roots]


def find_engine() -> str | None:
    """The CuraEngine program to slice with, or None when there is none."""
    own = os.environ.get(ENGINE_ENV)
    if own and os.path.isfile(own):
        return own
    for d in _bundle_dirs():
        path = os.path.join(d, ENGINE)
        if os.path.isfile(path):
            return path
    return shutil.which(ENGINE)


#: the system's dynamic loader on Linux: the CuraEngine of the Cura AppImage names a loader
#: relative to the AppImage, so it is started through this one, with its own libraries in `lib`
#: next to it. It needs glibc 2.38 or newer (Ubuntu 24.04 and later); see `engine_problem`
LINUX_LOADER = "/lib64/ld-linux-x86-64.so.2"
_PROBLEMS: dict = {}


def engine_command(engine: str) -> list:
    """How to start `engine`: on its own, or on Linux through the loader with its own libraries."""
    lib = os.path.join(os.path.dirname(engine), "lib")
    if sys.platform.startswith("linux") and os.path.isdir(lib) and os.path.exists(LINUX_LOADER):
        return [LINUX_LOADER, "--library-path", lib, engine]
    return [engine]


def engine_problem(engine: str) -> str | None:
    """Why `engine` does not start on this computer, or None when it does (asked once)."""
    if engine not in _PROBLEMS:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            run = subprocess.run(engine_command(engine) + ["help"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=30,
                                 env=engine_environment(engine), creationflags=flags)
            text = run.stdout + run.stderr
            ok = "CuraEngine slice" in text
            problem = None if ok else (text.strip().splitlines() or ["CuraEngine did not start"])[-1]
            if problem and "GLIBC" in problem:
                problem = ("CuraEngine needs a newer Linux (glibc 2.38 or later, such as "
                           "Ubuntu 24.04): " + problem)
        except (OSError, subprocess.SubprocessError) as exc:
            problem = str(exc)
        _PROBLEMS[engine] = problem
    return _PROBLEMS[engine]


def engine_environment(engine: str) -> dict:
    """The environment CuraEngine runs in: on Linux its own libraries in `lib` next to it."""
    env = dict(os.environ)
    lib = os.path.join(os.path.dirname(engine), "lib")
    if sys.platform != "win32" and os.path.isdir(lib):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (lib, env.get("LD_LIBRARY_PATH")) if p)
    return env


def find_resources() -> str | None:
    """The Cura profile library (a `resources` folder with `definitions`), or None."""
    own = os.environ.get(RESOURCES_ENV)
    if own and os.path.isdir(os.path.join(own, "definitions")):
        return own
    for d in _bundle_dirs():
        path = os.path.join(d, "resources")
        if os.path.isdir(os.path.join(path, "definitions")):
            return path
    return None


@dataclass
class SliceResult:
    """What a slice made: the G-code file and what it says about the print."""

    path: str
    seconds: float = 0.0
    filament_m: list = field(default_factory=list)     # per extruder
    grams: float = 0.0
    layers: int = 0
    log: str = ""


def _json_value(v):
    """A value as CuraEngine reads it from the resolved-settings file."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) if isinstance(v, float) else int(v)
    if isinstance(v, (list, tuple)):
        return [_json_value(x) for x in v]
    if v is None:
        return ""
    return str(v)


def write_stl(path, verts, tris):
    """A binary STL of one mesh."""
    v = np.asarray(verts, dtype=np.float32)
    t = np.asarray(tris, dtype=np.int64)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln > 0, ln, 1.0)[:, None]
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3),
                                  ("c", "<f4", 3), ("attr", "<u2")])
    rec["n"], rec["a"], rec["b"], rec["c"] = n, a, b, c
    with open(path, "wb") as fh:
        fh.write(b"Volura".ljust(80, b" "))
        fh.write(np.uint32(len(t)).tobytes())
        fh.write(rec.tobytes())


def place_on_bed(meshes, settings) -> list:
    """The meshes moved so their common centre is the centre of the bed, the lowest on z = 0.

    CuraEngine adds half the bed to every coordinate unless the origin is in the middle of the bed,
    so the coordinates it gets are relative to that middle.
    """
    lo = np.min([np.asarray(v).min(axis=0) for v, _t, _e in meshes], axis=0)
    hi = np.max([np.asarray(v).max(axis=0) for v, _t, _e in meshes], axis=0)
    shift = np.array([-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2]])
    return [(np.asarray(v, dtype=float) + shift, t, e) for v, t, e in meshes]


def resolved_settings(stacks) -> dict:
    """The resolved-settings document for CuraEngine's `-r`: the global stack and each extruder."""
    doc = {"global": {k: _json_value(v) for k, v in stacks.global_settings().items()}}
    for i in range(len(stacks.extruders)):
        doc[f"extruder.{i}"] = {k: _json_value(v) for k, v in stacks.extruder_settings(i).items()}
    return doc


def parse_header(text: str) -> dict:
    """The print time, filament and layer count CuraEngine writes at the top of its G-code."""
    out = {"seconds": 0.0, "filament_m": [], "layers": 0}
    m = re.search(r"^;TIME:(\d+(?:\.\d+)?)", text, re.M)
    if m:
        out["seconds"] = float(m.group(1))
    m = re.search(r"^;Filament used:\s*(.+)$", text, re.M)
    if m:
        out["filament_m"] = [float(x.rstrip("m")) for x in re.findall(r"[\d.]+m", m.group(1))]
    m = re.search(r"^;PRINT\.TIME:(\d+(?:\.\d+)?)", text, re.M)        # UltiMaker's Griffin header
    if m:
        out["seconds"] = float(m.group(1))
    used = re.findall(r"^;EXTRUDER_TRAIN\.(\d+)\.MATERIAL\.VOLUME_USED:([\d.]+)", text, re.M)
    if used and not out["filament_m"]:
        out["volume_mm3"] = [float(v) for _i, v in used]
    m = re.search(r"^;LAYER_COUNT:(\d+)", text, re.M)
    if m:
        out["layers"] = int(m.group(1))
    return out


def slice_meshes(stacks, meshes, out_path, engine=None, timeout=None, on_progress=None) -> SliceResult:
    """Slice `meshes` - a list of (verts, tris, extruder_nr) in bed millimetres - to `out_path`.

    `stacks` is the `cura_profiles.Stacks` of the printer, nozzle, material and quality to slice
    with. `on_progress(fraction)` is called while CuraEngine works. A RuntimeError says why a slice
    failed.
    """
    engine = engine or find_engine()
    if not engine:
        raise RuntimeError("CuraEngine was not found")
    if not meshes:
        raise RuntimeError("there is nothing to slice")
    doc = resolved_settings(stacks)
    meshes = place_on_bed(meshes, doc["global"])
    with tempfile.TemporaryDirectory(prefix="volura-slice-") as tmp:
        for i, (v, t, e) in enumerate(meshes):
            name = f"part{i}.stl"
            write_stl(os.path.join(tmp, name), v, t)
            doc[name] = {"extruder_nr": int(e)}
        settings_path = os.path.join(tmp, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        cmd = engine_command(engine) + ["slice", "-p", "-r", settings_path, "-o", out_path]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=flags,
                                env=engine_environment(engine))
        log = []
        for line in proc.stdout:
            log.append(line)
            m = re.search(r"Progress:\s*\w+[^\d]*([\d.]+)", line)
            if m and on_progress:
                try:
                    on_progress(min(1.0, float(m.group(1))))
                except ValueError:
                    pass
        rc = proc.wait(timeout=timeout)
    text = "".join(log)
    if rc != 0 or not os.path.isfile(out_path):
        errors = [ln.strip() for ln in log if "[error]" in ln or "[critical]" in ln]
        raise RuntimeError(errors[-1] if errors else f"CuraEngine stopped with code {rc}")
    with open(out_path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    info = parse_header(head)
    density = stacks.material.metadata["density"] if stacks.material is not None else 1.24
    diameter = float(stacks.extruders[0].get("material_diameter") or 1.75)
    area = math.pi * (0.5 * diameter) ** 2                      # mm2
    if info.get("volume_mm3"):
        info["filament_m"] = [v / area / 1000.0 for v in info["volume_mm3"]]
    grams = sum(info["filament_m"]) * 1000.0 * area * density / 1000.0
    return SliceResult(out_path, info["seconds"], info["filament_m"], grams, info["layers"], text)
