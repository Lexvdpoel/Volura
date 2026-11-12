"""Adds the printers of OrcaSlicer's profiles that Cura does not have to ./curaengine for slicing.

    python tools/fetch_orca_profiles.py [--profiles DIR]

Run it after tools/fetch_curaengine.py. It takes the printer profiles of an OrcaSlicer release
(only the resources/profiles folder of its source, with git), and makes a Cura printer of every
printer model that the Cura profiles do not have already:

    curaengine/resources/definitions/orca_*.def.json    the printer: bed, height, firmware,
                                                        retraction and speeds of its profile
    curaengine/resources/extruders/orca_*.def.json      its extruder: nozzle and filament
    curaengine/resources/variants/orca_*.inst.cfg       one per nozzle OrcaSlicer has for it
    curaengine/resources/ORCA_VERSION                   the OrcaSlicer release they came from

The start and end G-code stay in OrcaSlicer's own language; volura/orca_gcode.py works them out
when you slice. A printer whose G-code that cannot do is left out. The materials and qualities
are Cura's common ones. `--profiles` uses a profiles folder you have already (from an OrcaSlicer
checkout or installation) instead of fetching it. The profiles are AGPL-3.0, like OrcaSlicer.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from volura import orca_gcode  # noqa: E402

ORCA_VERSION = "2.5.0"
REPO = "https://github.com/OrcaSlicer/OrcaSlicer.git"
TARGET = ROOT / "curaengine" / "resources"
#: vendors of the profiles that are no printer maker
SKIP_VENDORS = {"Custom", "OrcaArena", "OrcaFilamentLibrary"}
#: OrcaSlicer's firmware flavors as Cura names them
FLAVORS = {"marlin": "RepRap (Marlin/Sprinter)", "marlin2": "RepRap (Marlin/Sprinter)",
           "klipper": "RepRap (Marlin/Sprinter)", "smoothie": "RepRap (Marlin/Sprinter)",
           "reprapfirmware": "RepRap (RepRap)", "repetier": "Repetier"}
#: the speeds of a print profile that become Cura settings: Cura key -> OrcaSlicer key; the ones
#: that print are held to what the hot end melts (see `speeds`)
SPEEDS = {"speed_wall_0": ("outer_wall_speed", "wall_line_width_0"),
          "speed_wall_x": ("inner_wall_speed", "wall_line_width_x"),
          "speed_print": ("inner_wall_speed", "line_width"),
          "speed_infill": ("sparse_infill_speed", "infill_line_width"),
          "speed_topbottom": ("top_surface_speed", "skin_line_width"),
          "speed_support": ("support_speed", "support_line_width"),
          "speed_travel": ("travel_speed", None),
          "speed_layer_0": ("initial_layer_speed", None)}
#: what OrcaSlicer's plastics melt at most, in mm3/s, when the filament profile does not say
MAX_FLOW = 12.0


def fetch(folder: Path) -> Path:
    """resources/profiles of the OrcaSlicer release, without the rest of its source."""
    repo = folder / "OrcaSlicer"
    subprocess.check_call(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1",
                           "--branch", f"v{ORCA_VERSION}", "--filter=blob:none", "--sparse",
                           REPO, str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone",
                           "/resources/profiles/*.json", "/resources/profiles/*/machine/",
                           "/resources/profiles/*/process/", "/resources/profiles/*/filament/"])
    return repo / "resources" / "profiles"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Profiles:
    """The presets of an OrcaSlicer profiles folder, with what they inherit."""

    def __init__(self, root):
        self.root = Path(root)
        self.vendors = {}                       # folder -> the name of the maker
        self.presets = {}                       # (kind, name) -> (vendor, raw json)
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "name" in data:
                self.vendors[path.stem] = data["name"]
        for vendor in self.vendors:
            for kind in ("machine", "process", "filament"):
                for p in glob.glob(str(self.root / vendor / kind / "**" / "*.json"), recursive=True):
                    try:
                        data = json.loads(Path(p).read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if "name" in data:
                        self.presets.setdefault((kind, data["name"]), (vendor, data))

    def flat(self, kind: str, name: str, depth=0) -> dict:
        """The preset `name` with what it inherits merged in."""
        if (kind, name) not in self.presets or depth > 20:
            return {}
        data = dict(self.presets[(kind, name)][1])
        parent = data.get("inherits")
        if parent:
            base = self.flat(kind, parent, depth + 1)
            base.update(data)
            data = base
        return data

    def machines(self):
        """(vendor, model, [preset names]) of every printer model that can be chosen."""
        models = {}
        for (kind, name), (vendor, data) in self.presets.items():
            if kind != "machine" or vendor in SKIP_VENDORS or data.get("instantiation") != "true":
                continue
            model = self.flat(kind, name).get("printer_model") or name
            models.setdefault((vendor, model), []).append(name)
        return [(v, m, sorted(n)) for (v, m), n in sorted(models.items())]

    def process_for(self, machine: dict, preset: str) -> dict:
        name = machine.get("default_print_profile")
        if name and ("process", name) in self.presets:
            return self.flat("process", name)
        for (kind, pname), (_v, data) in self.presets.items():
            if kind == "process" and preset in (data.get("compatible_printers") or []):
                return self.flat("process", pname)
        return {}

    def filament_for(self, machine: dict) -> dict:
        names = machine.get("default_filament_profile") or []
        names = [names] if isinstance(names, str) else names
        for name in names:
            if ("filament", name) in self.presets:
                return self.flat("filament", name)
        return {}


def _first(value, default=None):
    """The first of a setting that OrcaSlicer keeps per extruder ("0.4" or "0.4,0.4" or a list)."""
    if isinstance(value, list):
        value = value[0] if value else default
    if isinstance(value, str) and "," in value:
        value = value.split(",")[0].strip()
    v = orca_gcode.typed(value) if isinstance(value, str) else value
    return default if v is None or v == "" else v


def _text(value) -> str:
    return "\n".join(value) if isinstance(value, list) else (value or "")


def nozzle_label(preset: str, model: str, nozzle: float) -> str:
    """The name of a nozzle as Cura shows it: "0.4 mm", "0.6 mm HF"."""
    rest = preset[len(model):].strip() if preset.startswith(model) else ""
    rest = re.sub(r"(?i)\b\d+(\.\d+)?\s*(mm)?\s*", "", rest, count=1)
    rest = re.sub(r"(?i)\bnozzle\b", "", rest).strip(" -_()")
    return f"{nozzle:g} mm" + (f" {rest}" if rest else "")


def bed(machine: dict):
    """(width, depth, centre is zero, round, offset) of the printable area of a machine preset.

    Cura's bed runs from 0 to its width, or is centred on 0; a bed of OrcaSlicer that starts a
    little away from 0 keeps the part of it from 0 on, and `offset` is how far the middle of what
    is printable is from the middle of Cura's bed.
    """
    pts = orca_gcode.points(machine.get("printable_area", []))
    if len(pts) < 3:
        return None
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    round_bed = len(pts) > 8
    if x0 < 0 and abs(x0 + x1) < 1.0 and y0 < 0 and abs(y0 + y1) < 1.0:
        return x1 - x0, y1 - y0, True, round_bed, (0.0, 0.0)
    if x1 <= 0 or y1 <= 0:
        return None
    ox, oy = max(x0, 0.0), max(y0, 0.0)
    return x1, y1, False, round_bed, (0.5 * ox, 0.5 * oy)


def speeds(process: dict, flow: float) -> dict:
    """The speeds of an OrcaSlicer print profile as Cura settings, held to what the hot end melts:
    a line of `line width` by the layer height may go no faster than `flow` mm3/s."""
    out = {}
    for cura, (orca, width) in SPEEDS.items():
        v = _first(process.get(orca))
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            continue
        if width:
            out[cura] = {"value": f"min({v:g}, {flow:g} / max(0.01, layer_height * {width}))"}
        else:
            out[cura] = {"value": f"{v:g}"}
    return out


def check_gcode(config: dict, start: str, end: str, model: str):
    """Why the start and end G-code of a printer cannot be worked out, or None when they can."""
    values = orca_gcode.variables(
        config, nozzle_temp0=210, nozzle_temp=205, bed_temp0=60, bed_temp=60, material="PLA",
        layer_height=0.2, first_layer_height=0.2, nozzle=0.4, retraction=0.8, travel_speed=150,
        print_min=(10.0, 10.0), print_max=(50.0, 50.0), top=20.0, layers=100, printer=model)
    try:
        unknown = sorted(n for n in orca_gcode.names_used(start + "\n" + end) if n not in values)
        if unknown:
            return "unknown " + ", ".join(unknown[:4])
        text = orca_gcode.render(start, values) + orca_gcode.render(end, values)
    except orca_gcode.OrcaGcodeError as e:
        return str(e)
    if "{" in text or "}" in text:
        return "braces left in the G-code"
    return None


def cura_printers(resources: Path):
    """The names of Cura's printers, and its makers by their key."""
    names, makers = set(), {}
    for path in (resources / "definitions").glob("*.def.json"):
        if path.name.startswith("orca_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names.add(_key(data.get("name", "")))
        maker = data.get("metadata", {}).get("manufacturer")
        if maker:
            makers.setdefault(_key(maker), maker)
    return names, makers


def maker_name(vendor_name: str, makers: dict) -> str:
    """The maker as Cura spells it when Cura has it ("Creality" is Cura's "Creality3D")."""
    k = _key(vendor_name)
    for mk, name in makers.items():
        if mk == k:
            return name
    for mk, name in sorted(makers.items()):
        if len(k) >= 4 and (mk.startswith(k) or k.startswith(mk)):
            return name
    return {"Bambulab": "Bambu Lab"}.get(vendor_name, vendor_name)


def slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


def convert(profiles_dir, resources: Path, log=print) -> list[str]:
    """Writes a Cura printer for every OrcaSlicer printer model Cura does not have; their ids."""
    profiles = Profiles(profiles_dir)
    names, makers = cura_printers(resources)
    for folder in ("definitions", "extruders", "variants"):
        (resources / folder).mkdir(parents=True, exist_ok=True)
        for old in (resources / folder).glob("orca_*"):
            old.unlink()
    made, skipped = [], []
    for vendor, model, presets in profiles.machines():
        maker = maker_name(profiles.vendors[vendor], makers)
        if _key(model) in names or _key(maker + " " + model) in names:
            continue
        flats = {p: profiles.flat("machine", p) for p in presets}
        nozzles = {}
        for p in presets:
            n = _first(flats[p].get("nozzle_diameter"), 0.4)
            nozzles[p] = float(n) if isinstance(n, (int, float)) and 0.05 < n < 3 else 0.4
        base = min(presets, key=lambda p: (abs(float(nozzles[p]) - 0.4), p))
        machine = flats[base]
        shape = bed(machine)
        if shape is None:
            skipped.append((model, "no printable area"))
            continue
        width, depth, centred, round_bed, offset = shape
        process = profiles.process_for(machine, base)
        filament = profiles.filament_for(machine)
        start = _text(machine.get("machine_start_gcode"))
        end = _text(machine.get("machine_end_gcode"))
        config = {}
        for source in (filament, process, machine):
            config.update({k: v for k, v in source.items()
                           if k not in ("inherits", "name", "from", "instantiation", "type",
                                        "setting_id", "compatible_printers", "filament_id")})
        problem = check_gcode(config, start, end, model)
        if problem:
            skipped.append((model, problem))
            continue
        used = orca_gcode.names_used(start + "\n" + end) | {"printable_area", "nozzle_diameter",
                                                           "printer_model"}
        config = {k: v for k, v in config.items() if k in used}
        did = "orca_" + slug(model)
        ext = did + "_extruder_0"
        flow = _first(filament.get("filament_max_volumetric_speed"), MAX_FLOW)
        diameter = _first(filament.get("filament_diameter"), 1.75)
        height = _first(machine.get("printable_height"), 200)
        flavor = FLAVORS.get(str(_first(machine.get("gcode_flavor"), "marlin")).lower(),
                             "RepRap (Marlin/Sprinter)")
        retract = _first(machine.get("retraction_length"), 0.8)
        retract_speed = _first(machine.get("retraction_speed"), 35)
        hop = _first(machine.get("z_hop"), 0)
        relative = bool(_first(process.get("use_relative_e_distances"), 1))
        labels = {}
        for p in sorted(presets, key=lambda p: (float(nozzles[p]), p)):
            label = nozzle_label(p, model, float(nozzles[p]))
            labels.setdefault(label, p)
        overrides = {
            "machine_name": {"default_value": model},
            "machine_width": {"default_value": round(width, 3)},
            "machine_depth": {"default_value": round(depth, 3)},
            "machine_height": {"default_value": float(height)},
            "machine_center_is_zero": {"default_value": centred},
            "machine_shape": {"default_value": "elliptic" if round_bed else "rectangular"},
            "machine_heated_bed": {"default_value": True},
            "machine_gcode_flavor": {"default_value": flavor},
            "machine_extruder_count": {"default_value": 1},
            "machine_start_gcode": {"default_value": ""},
            "machine_end_gcode": {"default_value": ""},
            "relative_extrusion": {"value": "True" if relative else "False"},
            "retraction_amount": {"default_value": float(retract)},
            "retraction_speed": {"default_value": float(retract_speed)},
            "retraction_hop_enabled": {"default_value": float(hop) > 0},
            "retraction_hop": {"default_value": float(hop) if float(hop) > 0 else 0.2},
        }
        overrides.update(speeds(process, float(flow)))
        definition = {
            "version": 2, "name": model, "inherits": "fdmprinter",
            "metadata": {
                "visible": True, "author": "OrcaSlicer", "manufacturer": maker,
                "file_formats": "text/x-gcode", "has_materials": True, "has_variants": True,
                "preferred_variant_name": nozzle_label(base, model, float(nozzles[base])),
                "preferred_quality_type": "draft",          # 0.2 mm, OrcaSlicer's standard
                "machine_extruder_trains": {"0": ext},
                "orca_preset": base, "orca_start_gcode": start, "orca_end_gcode": end,
                "orca_config": config, "orca_offset": list(offset)},
            "overrides": overrides}
        extruder = {
            "version": 2, "name": "Extruder 1", "inherits": "fdmextruder",
            "metadata": {"machine": did, "position": "0"},
            "overrides": {"extruder_nr": {"default_value": 0},
                          "machine_nozzle_size": {"default_value": float(nozzles[base])},
                          "material_diameter": {"default_value": float(diameter)}}}
        (resources / "definitions" / f"{did}.def.json").write_text(
            json.dumps(definition, indent=1, ensure_ascii=False), encoding="utf-8")
        (resources / "extruders" / f"{ext}.def.json").write_text(
            json.dumps(extruder, indent=1, ensure_ascii=False), encoding="utf-8")
        for label, p in labels.items():
            (resources / "variants" / f"{did}_{slug(label)}.inst.cfg").write_text(
                "[general]\n"
                f"name = {label}\n"
                "version = 4\n"
                f"definition = {did}\n\n"
                "[metadata]\n"
                "setting_version = 25\n"
                "type = variant\n"
                "hardware_type = nozzle\n\n"
                "[values]\n"
                f"machine_nozzle_size = {float(nozzles[p]):g}\n", encoding="utf-8")
        made.append(did)
    for model, why in skipped:
        log(f"left out {model}: {why}")
    log(f"{len(made)} printers from OrcaSlicer {ORCA_VERSION}, {len(skipped)} left out")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", help="an OrcaSlicer resources/profiles folder to use")
    args = ap.parse_args()
    if not (TARGET / "definitions" / "fdmprinter.def.json").is_file():
        raise SystemExit("run tools/fetch_curaengine.py first: the Cura profiles are not there")
    if args.profiles:
        convert(args.profiles, TARGET)
    else:
        with tempfile.TemporaryDirectory(prefix="orca-", ignore_cleanup_errors=True) as tmp:
            convert(fetch(Path(tmp)), TARGET)
    (TARGET / "ORCA_VERSION").write_text(ORCA_VERSION + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
