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
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

from . import cura_profiles

_log = logging.getLogger(__name__)

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
    volume_mm3: list = field(default_factory=list)     # per extruder


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
    so the coordinates it gets are relative to that middle. A mesh is ``(verts, tris, extruder)``
    with, for a part that carries its own per-mesh settings, a fourth element that is kept as is.
    """
    lo = np.min([np.asarray(m[0]).min(axis=0) for m in meshes], axis=0)
    hi = np.max([np.asarray(m[0]).max(axis=0) for m in meshes], axis=0)
    shift = np.array([-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2]])
    return [(np.asarray(m[0], dtype=float) + shift, m[1], m[2], *m[3:]) for m in meshes]


def _formula_names(text) -> set:
    """The settings a formula of a printer definition reads: the names in `=line_width * 2`."""
    try:
        return set(cura_profiles.Formula(str(text)).code.co_names)
    except (AttributeError, TypeError, ValueError, SyntaxError):
        return set()           # a formula that does not compile reads nothing


def worked_out_from(definition, keys) -> set:
    """Every setting of `definition` whose value is worked out from one of `keys`.

    A setting's value in a definition is often a formula over other settings, and that formula can
    read a setting that is itself worked out from one of `keys`, so this walks the formulas until
    nothing new turns up; a formula that reads itself does not make it turn forever."""
    needs = {k: _formula_names(p["value"]) for k, p in definition.items()
             if isinstance(p.get("value"), str)}
    found, todo = set(), list(keys)
    while todo:
        want = todo.pop()
        for key, names in needs.items():
            if key not in found and want in names:
                found.add(key)
                todo.append(key)
    return found


def _mesh_stack(stacks, settings):
    """The global stack of `stacks` with a part's own values on top of the layers it works out from.

    It is a `cura_profiles.Stack` of the same definition and the same layers - your own values, the
    quality - with the part's values as the topmost layer, so every formula is worked out exactly
    as it is for the whole print and only the part's values change the outcome. A setting that comes
    from an extruder or a `resolve` still comes from there: those values are the print's."""
    return cura_profiles.Stack(stacks, [dict(settings), *stacks.glob.layers],
                               stacks.printer.definition.settings, None)


def derived_mesh_settings(stacks, settings, keys) -> dict:
    """What a part's own values work out to for the settings CuraEngine only reads the result of.

    Cura works the formulas of a printer definition out in its interface and hands CuraEngine the
    results, so CuraEngine spaces the infill by `infill_line_distance` and does not know
    `infill_sparse_density` at all. A part that overrides the density therefore has to hand over the
    line distance that follows from it too - and the overlap, the pocket size of the cross patterns
    and the top and bottom layers with it - or the engine lays the infill of the whole print. Which
    settings follow from which is what `worked_out_from` reads from the definition, so this keeps
    working for a printer whose definitions derive more than these.

    Only a value that comes out different from the print's own is handed over, so a part that
    overrides nothing that matters sends nothing extra."""
    definition = stacks.printer.definition.settings
    want = sorted(k for k in worked_out_from(definition, keys)
                  if (definition.get(k) or {}).get("settable_per_mesh"))
    if not want:
        return {}
    glob = _mesh_stack(stacks, settings)
    whole = stacks.global_settings()
    out = {}
    for key in want:
        value, print_value = glob.get(key), whole.get(key)
        if (value is not None and _fits_the_setting(value, definition.get(key))
                and _is_different(value, print_value)):
            out[key] = _json_value(value)
    return out


def _is_different(value, print_value) -> bool:
    """Whether what a part works out is something else than what the print as a whole uses.

    Two formulas that come out the same but in another order can differ by a rounding error, so
    numbers only count as different when they are apart by more than a thousandth of a
    micrometre."""
    if (isinstance(value, (int, float)) and isinstance(print_value, (int, float))
            and not isinstance(value, bool) and not isinstance(print_value, bool)):
        return not math.isclose(float(value), float(print_value), rel_tol=1e-9, abs_tol=1e-6)
    return _json_value(value) != _json_value(print_value)


def _is_setting_value(value) -> bool:
    """Whether a value can travel to CuraEngine at all.

    A setting of one mesh travels as text on the command line, so a number has to be a real number
    - the engine turns text into a number and makes nothing of `nan` - and nothing deeper than a
    list of numbers and words can be handed over."""
    if isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_is_setting_value(one) for one in value)
    return False


def _fits_the_setting(value, props) -> bool:
    """Whether a part's value is what this setting holds, by the type its definition gives it: a
    number for a measure or a count, True or False for a switch, a word for a pattern, and for the
    settings that hold a list or an extruder anything that can be handed over at all.

    CuraEngine turns the text it is given into what the setting wants, so a value of another kind
    ends the whole slice with an engine error; a value that does not fit its setting is dropped like
    a key that cannot be per-mesh, because one wrong value in a file is no reason to print nothing."""
    kind = (props or {}).get("type") or ""
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in ("float", "int"):
        return number and math.isfinite(float(value))
    if kind == "bool":
        return isinstance(value, bool)
    if kind in ("enum", "str"):
        return isinstance(value, str) and value != ""
    return _is_setting_value(value)


def per_mesh_settings(stacks, settings) -> dict:
    """The part of a part's `settings` that CuraEngine accepts for a single mesh.

    A mesh may carry its own values for the settings its printer definition marks
    `settable_per_mesh` - `infill_sparse_density` and `material_flow` among them - and those go to
    the engine as settings of that one mesh. A key the definition does not mark so (a global-only or
    an extruder setting), or one the definition does not know at all, cannot be honoured per mesh: it
    is dropped with a warning rather than refused, because a setting the engine ignores is no reason
    to stop a print - which is also what happens to a value that does not fit its setting, a `nan` or
    a word where a measure belongs. Values are made JSON-safe the way every setting is.

    The values that follow from these - the line distance a density works out to - are added by
    `derived_mesh_settings`, because CuraEngine reads results instead of working formulas out."""
    out = {}
    if not settings:
        return out
    definition = stacks.printer.definition.settings
    for key, value in settings.items():
        props = definition.get(key)
        value = _json_value(value)
        if not props or not props.get("settable_per_mesh"):
            _log.warning("not a per-mesh setting, ignored for this part: %s", key)
        elif not _fits_the_setting(value, props):
            _log.warning("not a value CuraEngine can read for %s, ignored for this part", key)
        else:
            out[key] = value
    out.update(derived_mesh_settings(stacks, out, list(out)))
    return out


def resolved_settings(stacks) -> dict:
    """The resolved-settings document for CuraEngine's `-r`: the global stack and each extruder."""
    doc = {"global": {k: _json_value(v) for k, v in stacks.global_settings().items()}}
    for i in range(len(stacks.extruders)):
        doc[f"extruder.{i}"] = {k: _json_value(v) for k, v in stacks.extruder_settings(i).items()}
    return doc


def parse_header(text: str) -> dict:
    """What CuraEngine writes at the top of its G-code: the print time, the filament of every
    extruder and the layer count."""
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
    used = {int(i): float(v) for i, v in
            re.findall(r"^;EXTRUDER_TRAIN\.(\d+)\.MATERIAL\.VOLUME_USED:([\d.]+)", text, re.M)}
    if used:            # what every extruder uses, also when the header gives the lengths too
        out["volume_mm3"] = [used.get(i, 0.0) for i in range(1 + max(used))]
    m = re.search(r"^;LAYER_COUNT:(\d+)", text, re.M)
    if m:
        out["layers"] = int(m.group(1))
    return out


#: the code that asks the printer to change its filament by hand; firmware may know another
#: one, which is why the slice dialog lets you fill in your own
FILAMENT_CHANGE = "M600"


def layer_heights(text: str) -> list:
    """Every layer of the G-code CuraEngine writes, as ``[(line, height in mm)]``.

    CuraEngine writes `;LAYER:n` before every layer, but the move up to that layer often comes
    before it, at the end of the layer below, and a Z hop lifts the nozzle in between. So the
    height of a layer is the height the nozzle is at when the layer first extrudes.
    """
    out, want, z = [], False, None
    for k, line in enumerate(text.splitlines()):
        code = line.split(";", 1)[0].strip().upper()
        if line.startswith(";LAYER:"):
            out.append([k, None])
            want = True
        elif code and code.split(None, 1)[0] in ("G0", "G1", "G00", "G01"):
            m = re.search(r"\bZ(-?\d*\.?\d+)", code)
            if m:
                z = float(m.group(1))
            if (want and z is not None and re.search(r"\bE-?\d*\.?\d", code)
                    and re.search(r"\b[XY]-?\d", code)):
                out[-1][1] = z               # the first line this layer lays
                want = False
    return [row for row in out if row[1] is not None]


def with_commands(text: str, commands) -> str:
    """The G-code with every command at the start of the first layer at or above its height in mm.

    `commands` is an iterable of ``(height in mm, command line)``. A change by height is something
    the printer does halfway, so it is asked for at a layer boundary, where the nozzle has just
    moved up. Several commands that belong to one layer are all written, in the order they came,
    while the same command twice in one layer is asked for only once; a height above the print asks
    for nothing.
    """
    lines = text.splitlines(keepends=True)
    layers = layer_heights(text)
    where: dict = {}
    for h, command in commands:
        hit = next((k for k, z in layers if z >= float(h) - 1e-6), None)
        if hit is None:
            continue
        made = where.setdefault(hit, [])
        if command not in made:
            made.append(command)
    if not where:
        return text
    break_line = "\r\n" if "\r\n" in text else "\n"
    for k in sorted(where, reverse=True):
        lines[k + 1:k + 1] = [c + break_line for c in where[k]]
    return "".join(lines)


def with_filament_changes(text: str, heights, command=FILAMENT_CHANGE) -> str:
    """The G-code with `command` at the start of the first layer at or above every height in mm.

    A colour change by height is something the printer does halfway, so it is asked for at a
    layer boundary, where the nozzle has just moved up. Two heights that land in one layer ask
    for it once; a height above the print asks for nothing.
    """
    return with_commands(text, [(h, command) for h in heights])


def insert_commands(path, commands) -> None:
    """Put those ``(height mm, command)`` pairs into the G-code file at `path`, overwriting it."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    changed = with_commands(text, commands)
    if changed == text:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(changed)


def insert_filament_changes(path, heights, command=FILAMENT_CHANGE) -> None:
    """Put `command` at those heights in the G-code file at `path`, overwriting it."""
    insert_commands(path, [(h, command) for h in heights])


#: the words of a move line, the way `gcode_preview` reads them too
_WORD = re.compile(r"([A-Z])\s*(-?\d*\.?\d+)")
#: the E word of a move line, so it can be written again with another number
_E_WORD = re.compile(r"(?<![A-Za-z])E\s*-?\d*\.?\d+", re.IGNORECASE)
#: a file that lets its firmware retract: G10 pulls the filament back and G11 pushes it back, and
#: neither of them carries a length Volura can change
_FIRMWARE_RETRACTION = re.compile(r"^\s*G1[01]\b", re.M)
#: the moves that lay a line or travel, and the codes that say what is absolute and what is not
_MOVES = ("G0", "G1", "G00", "G01")


def _retraction_bands(bands) -> list:
    """The bands of a retraction test as ``[(z_from, z_to, length)]``, in the order they came.

    A band is asked for by height and not by part, so every test on the bed hands over its bands and
    they are kept in the order they came; where two of them cover one height, the first wins. An
    entry that is no band of three numbers is left out with a warning, like a meta of a
    hand-written file that carries no plan.
    """
    out = []
    for one in (bands or []):
        try:
            z_from, z_to, length = (float(v) for v in one)
        except (TypeError, ValueError):
            _log.warning("not a band of a retraction test, ignored: %s", one)
            continue
        if (math.isfinite(z_from) and math.isfinite(z_to) and math.isfinite(length)
                and z_to > z_from):
            out.append((z_from, z_to, length))
        else:
            _log.warning("not a band of a retraction test, ignored: %s", one)
    return out


def _band_length(z, bands):
    """The retraction length of the band that stands at `z`, or None when no band covers it.

    A band runs from its `z_from` up to but not including its `z_to`; the tolerance keeps a layer
    that lands a hair under a boundary in the band above, where the rounding of a change by height
    also puts it."""
    for z_from, z_to, length in bands:
        if z_from - 1e-6 <= z < z_to - 1e-6:
            return length
    return None


def _number_text(value) -> str:
    """A number the way a G-code word holds it: five decimals at most, no noise of the addition
    behind it and never a `-0`."""
    v = round(float(value), 5) + 0.0
    text = f"{v:.5f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _with_e(line: str, value: float) -> str:
    """The line with its E word written again as `value`; everything else, the comment included,
    stays exactly as it was."""
    body = line.rstrip("\r\n")
    eol = line[len(body):]
    cut = body.find(";")
    code = body[:cut] if cut >= 0 else body
    note = body[cut:] if cut >= 0 else ""
    made, n = _E_WORD.subn(lambda _m: f"E{_number_text(value)}", code, count=1)
    if not n:
        return line
    return made + note + eol


def rewrite_retractions(text: str, bands) -> str:
    """The G-code with, in every layer of a band, the retractions of the length of that band.

    `bands` is ``[(z_from mm, z_to mm, length mm), ...]``, the plan of a retraction test. CuraEngine
    writes no command for a retraction but a move that only turns the extruder: an E that goes down
    with no X and no Y in it, followed by the E that goes back up again once the nozzle has reached
    where it wants to lay plastic. So a band gets its length by rewriting those two moves: every
    retraction in the layers of the band becomes `length` millimetres, and the prime that answers it
    the same length back, so the filament stands exactly where it stood and only the travel is
    printed with another length. Nothing else is touched; the plastic a print lays down stays
    exactly as much as it was.

    `G1 E-0.8` and `G1 E486.1304` say the same thing in two languages, and both are handled. With
    absolute extrusion (M82, what CuraEngine does by default) an E names a position, so from the
    first changed length every E that follows carries the difference with it - and a `G92 E0`, which
    starts the count of filament again, brings the writing back to what the file said. With relative
    extrusion (M83) an E is already a difference, so only the retraction and its prime are rewritten.

    Only the moves of a layer are asked for: CuraEngine writes `;LAYER:` before one and
    `;TIME_ELAPSED:` after it, and between those two the nozzle prints; the filament the start code
    pushes out and the retraction of the end code are left to the printer. A G-code that carries no
    `;LAYER:` comments is filtered on its height alone. A file whose firmware retracts itself with
    `G10` and `G11` holds no length to change, so it comes back as it stood with a note in the log.
    """
    made = _retraction_bands(bands)
    if not made:
        return text
    if _FIRMWARE_RETRACTION.search(text):
        _log.warning("this G-code retracts with G10 and G11, which carry no length to change: the "
                     "retraction test prints with the retraction of the profile")
        return text
    has_layers = ";LAYER:" in text
    lines = text.splitlines(keepends=True)
    pos = 0.0                    # mm of filament behind the nozzle, as the printer reads this file
    shift = 0.0                  # mm added to every E word that follows, so the position stays right
    z = 0.0                      # mm high the nozzle stands
    abs_pos = abs_e = True       # G90/G91 for x, y and z; M82/M83 for the filament
    inside = not has_layers      # are the moves here part of a layer?
    want = None                  # the length of the retraction that has not come back yet
    changed = False
    for k, line in enumerate(lines):
        body = line.rstrip("\r\n")
        cut = body.find(";")
        code = body[:cut] if cut >= 0 else body
        if not code.strip():
            if body.startswith(";LAYER:"):
                inside = True
            elif body.startswith(";TIME_ELAPSED:"):
                inside = False
            continue
        head = code.split(None, 1)[0].upper()
        if head in _MOVES:
            rest = code[len(head):].upper()
            if "E" not in rest and "Z" not in rest:
                continue                    # a move that neither turns the extruder nor climbs
            words = {key: float(v) for key, v in _WORD.findall(rest)}
            if "Z" in words:
                z = words["Z"] if abs_pos else z + words["Z"]
            if "E" not in words:
                continue
            value = words["E"]
            delta = value + shift - pos if abs_e else value
            only_filament = not ("X" in words or "Y" in words)
            length = _band_length(z, made) if only_filament and inside else None
            new = None                                  # what this E has to say about the filament
            if delta < -1e-9 and length is not None:
                new, want = -length, length             # the retraction of a layer of its band
            elif delta > 1e-9 and only_filament and want is not None:
                new, want = want, None                  # and the prime that answers it
            if abs_e:
                shift += 0.0 if new is None else new - delta
                word = value + shift
                pos = word
            else:
                word = value if new is None else new
                pos += word
            if abs(word - value) > 1e-9:
                lines[k] = _with_e(line, word)
                changed = True
        elif head == "G90":
            abs_pos = True
        elif head == "G91":
            abs_pos = False
        elif head == "M82":
            abs_e = True
        elif head == "M83":
            abs_e = False
        elif head == "G92":
            words = {key: float(v) for key, v in _WORD.findall(code[len(head):].upper())}
            if "Z" in words:
                z = words["Z"]
            if "E" in words:
                # the printer starts counting filament again, so the difference the rewrite built up
                # is nothing more than the length of the retraction that is still behind the nozzle
                shift = 0.0
                pos = words["E"]
    return "".join(lines) if changed else text


def insert_retractions(path, bands) -> None:
    """Rewrite the retractions of the G-code file at `path` per band, overwriting it."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    changed = rewrite_retractions(text, bands)
    if changed == text:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(changed)


def set_print_temperature(stacks, temp) -> None:
    """Set `temp` as the print temperature the print starts at, as a user value on `stacks`.

    The value goes on the top layer of the global stack and of every extruder - the same place a
    change in the slice dialog goes - and the caches are cleared so the engine reads the new
    temperature. A temperature tower needs this for its bottom segment, so the first layers print
    at its temperature before any `M104` comes along.
    """
    for stack in [stacks.glob] + list(stacks.extruders):
        if stack.layers:
            stack.layers[0]["material_print_temperature"] = float(temp)
        stack.cache.clear()
        stack.raw_cache.clear()


def orca_gcode_for(stacks, meshes, glob: dict):
    """The start and end G-code of a printer from OrcaSlicer's profiles, worked out for this print.

    Those printers keep their G-code in OrcaSlicer's language (see `volura.orca_gcode`), which
    CuraEngine does not read; None for a printer of Cura's own.
    """
    from . import orca_gcode
    meta = stacks.printer.definition.metadata
    if "orca_start_gcode" not in meta:
        return None
    e0 = stacks.extruders[0]

    def num(stack, key, default=0.0):
        try:
            return float(stack.get(key))
        except (TypeError, ValueError):
            return default
    lo = np.min([np.asarray(m[0]).min(axis=0) for m in meshes], axis=0)
    hi = np.max([np.asarray(m[0]).max(axis=0) for m in meshes], axis=0)
    if not glob.get("machine_center_is_zero"):
        half = np.array([0.5 * float(glob.get("machine_width", 0)),
                         0.5 * float(glob.get("machine_depth", 0)), 0.0])
        lo, hi = lo + half, hi + half
    first = num(stacks.glob, "layer_height_0", 0.2)
    layer = num(stacks.glob, "layer_height", 0.2) or 0.2
    top = float(hi[2])
    material = stacks.material.metadata.get("material", "PLA") if stacks.material else "PLA"
    values = orca_gcode.variables(
        meta.get("orca_config", {}),
        nozzle_temp0=num(e0, "material_print_temperature_layer_0", 200),
        nozzle_temp=num(e0, "material_print_temperature", 200),
        bed_temp0=num(e0, "material_bed_temperature_layer_0", 60),
        bed_temp=num(e0, "material_bed_temperature", 60),
        material=str(material).upper(), layer_height=layer, first_layer_height=first,
        nozzle=num(e0, "machine_nozzle_size", 0.4), retraction=num(e0, "retraction_amount", 0.8),
        travel_speed=num(e0, "speed_travel", 150), print_min=(lo[0], lo[1]),
        print_max=(hi[0], hi[1]), top=top,
        layers=1 + max(0, math.ceil((top - first) / layer - 1e-6)),
        printer=stacks.printer.name, quality=stacks.quality_name,
        filament=stacks.material.name if stacks.material else "")
    try:
        return (orca_gcode.render(meta["orca_start_gcode"], values),
                orca_gcode.render(meta.get("orca_end_gcode", ""), values))
    except orca_gcode.OrcaGcodeError as e:
        raise RuntimeError(f"the start G-code of this printer cannot be worked out: {e}") from None


def command_value(value) -> str:
    """A setting value the way CuraEngine's `-s` wants it, as one word.

    The engine reads every value as text and turns it into what its call site wants, so a list
    becomes `[a,b]` - the same text the resolved-settings file holds - and a bool one of the words
    it compares against."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(command_value(v) for v in value) + "]"
    return str(value)


def slice_meshes(stacks, meshes, out_path, engine=None, timeout=None, on_progress=None) -> SliceResult:
    """Slice `meshes` - a list of (verts, tris, extruder_nr) in bed millimetres - to `out_path`.

    `stacks` is the `cura_profiles.Stacks` of the printer, nozzle, material and quality to slice
    with. `on_progress(fraction)` is called while CuraEngine works. A RuntimeError says why a slice
    failed.

    A mesh may carry a fourth element: a dict of CuraEngine settings for that one mesh, as
    `per_mesh_settings` kept only the `settable_per_mesh` ones of them. Those follow the mesh's own
    file on the command line, which is where CuraEngine takes a setting for that mesh alone, so a
    single pad of a flow rate test prints with its own `material_flow` while the rest of the bed
    keeps the global value. A mesh that asks for nothing is loaded from the settings document
    instead, so a bed of a thousand plain parts stays one short command.
    """
    engine = engine or find_engine()
    if not engine:
        raise RuntimeError("CuraEngine was not found")
    if not meshes:
        raise RuntimeError("there is nothing to slice")
    doc = resolved_settings(stacks)
    meshes = place_on_bed(meshes, doc["global"])
    offset = stacks.printer.definition.metadata.get("orca_offset")
    if offset and any(offset):
        # the middle of what this printer can print is not the middle of Cura's bed
        meshes = [(m[0] + np.array([offset[0], offset[1], 0.0]), m[1], m[2], *m[3:])
                  for m in meshes]
    gcode = orca_gcode_for(stacks, meshes, doc["global"])
    if gcode is not None:
        start, end = gcode
        doc["global"]["machine_start_gcode"], doc["global"]["machine_end_gcode"] = start, end
        # what Cura's own front end decides: heat up first only when the start G-code does not
        codes = set(re.findall(r"^\s*(M1(?:04|09|40|90))(?![0-9])", start, re.M))
        doc["global"]["material_bed_temp_prepend"] = not codes & {"M140", "M190"}
        doc["global"]["material_print_temp_prepend"] = not codes & {"M104", "M109"}
    with tempfile.TemporaryDirectory(prefix="volura-slice-") as tmp:
        for i, m in enumerate(meshes):
            write_stl(os.path.join(tmp, f"part{i}.stl"), m[0], m[1])
        own = [m[3] if len(m) > 3 and m[3] else {} for m in meshes]
        # a part that wants settings of its own is named on the command line, which is where
        # CuraEngine takes a setting for one mesh alone; the model keys of the document load every
        # part that asks for nothing, so a bed of a thousand plain parts is still a short command
        for i, settings in enumerate(own):
            if not settings:
                doc[f"part{i}.stl"] = {"extruder_nr": int(meshes[i][2])}
        settings_path = os.path.join(tmp, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        # right behind the mesh its own -s settings, which beat the values of the whole print for
        # that mesh; the parts are named from the working folder of the engine, so a long row of
        # parts with settings of their own is still one command of a length Windows accepts
        cmd = engine_command(engine) + ["slice", "-p", "-r", settings_path,
                                        "-o", os.path.abspath(out_path)]
        for i, settings in enumerate(own):
            if not settings:
                continue
            cmd += ["-e" + str(int(meshes[i][2])), "-l", f"part{i}.stl"]
            for key, value in settings.items():
                cmd += ["-s", f"{key}={command_value(value)}"]
        # Windows hands a program its arguments as one string of at most 32767 characters, and the
        # settings of a part are worth a line of those; ask before the engine chokes on it silently
        if sum(len(one) + 1 for one in cmd) > 30000:
            raise RuntimeError("too many parts slice with settings of their own: take some of them "
                               "back in Entity Info, or slice the model in pieces")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=flags,
                                cwd=tmp, env=engine_environment(engine))
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
    else:
        # a Marlin header only says how long every filament ran: the volume comes from its section
        info["volume_mm3"] = [m * 1000.0 * area for m in info["filament_m"]]
    grams = sum(info["filament_m"]) * 1000.0 * area * density / 1000.0
    return SliceResult(out_path, info["seconds"], info["filament_m"], grams, info["layers"], text,
                       info["volume_mm3"])
