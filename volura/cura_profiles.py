"""Printer profiles for slicing: the Cura definitions, nozzles, qualities and materials.

No Qt in this file. It reads the profile library Cura ships with (its `resources` folder) and turns
one choice of printer, nozzle, material and quality into the flat lists of settings CuraEngine
slices with, so the slicer can be tested on its own.

Cura keeps a setting in layers. A **definition** (`definitions/*.def.json`) describes every setting
of a printer and inherits from a more general one, down to `fdmprinter`; an **extruder** definition
does the same for one nozzle. On top of that lie a **variant** (the nozzle size), a **material** (an
`.xml.fdm_material` file) and a **quality** (`.inst.cfg` files with a layer height and what goes
with it), and the topmost value wins. Many values are not numbers but small Python formulas over
other settings, such as `wall_thickness = line_width * 2`; Cura works them out in its interface and
hands CuraEngine only the results, so this file does that too, the way Cura does it:

- a stack per extruder and one for the whole printer (the global stack); an extruder stack looks in
  its own layers first and then in the global stack, and a formula is worked out from the stack
  that asked for it, so `speed_wall` follows the speed of the extruder it is asked for;
- a setting that is not settable per extruder always comes from the global stack, a setting with
  `limit_to_extruder` comes from that extruder, and the global stack works out a setting with a
  `resolve` formula over all extruders;
- `extruderValue`, `extruderValues`, `resolveOrValue` and the other helpers Cura gives a formula,
  and a formula that fails counts as 0, as in Cura.

The library is read lazily: the list of printers reads only the definitions, and the qualities and
materials are read the first time a printer needs them. `Library.load` reads it all at once
instead: with many threads, because the more than ten thousand small files make a first read wait
on the disk (and the virus scanner) far longer than on Python, and into a cache file, so the next
time it takes a moment instead of a minute.
"""
from __future__ import annotations

import configparser
import json
import math
import os
import pickle
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property

#: the namespace of Cura's material files
_UM = "{http://www.ultimaker.com/material}"
#: material file settings and the Cura settings they set (from Cura's XmlMaterialProfile)
MATERIAL_SETTINGS = {
    "print temperature": "default_material_print_temperature",
    "heated bed temperature": "default_material_bed_temperature",
    "standby temperature": "material_standby_temperature",
    "processing temperature graph": "material_flow_temp_graph",
    "print cooling": "cool_fan_speed",
    "retraction amount": "retraction_amount",
    "retraction speed": "retraction_speed",
    "adhesion tendency": "material_adhesion_tendency",
    "surface energy": "material_surface_energy",
    "build volume temperature": "build_volume_temperature",
    "anti ooze retract position": "material_anti_ooze_retracted_position",
    "anti ooze retract speed": "material_anti_ooze_retraction_speed",
    "break preparation position": "material_break_preparation_retracted_position",
    "break preparation speed": "material_break_preparation_speed",
    "break preparation temperature": "material_break_preparation_temperature",
    "break position": "material_break_retracted_position",
    "flush purge speed": "material_flush_purge_speed",
    "flush purge length": "material_flush_purge_length",
    "end of filament purge speed": "material_end_of_filament_purge_speed",
    "end of filament purge length": "material_end_of_filament_purge_length",
    "maximum park duration": "material_maximum_park_duration",
    "no load move factor": "material_no_load_move_factor",
    "break speed": "material_break_speed",
    "break temperature": "material_break_temperature",
}


class Formula:
    """A setting value that is worked out from other settings: `=line_width * 2` in a profile."""

    __slots__ = ("text", "code")

    def __init__(self, text: str):
        self.text = text.strip()
        try:
            self.code = compile(self.text, "<setting>", "eval")
        except SyntaxError:
            self.code = None

    def __repr__(self):
        return f"Formula({self.text!r})"


def _parse_bool(text) -> bool:
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in ("true", "1", "yes")


def _literal(text: str, kind: str | None):
    """A value as a profile writes it, turned into what its setting type holds."""
    text = text.strip()
    if text.startswith("="):
        return Formula(text[1:])
    if kind in ("float",):
        try:
            return float(text)
        except ValueError:
            return Formula(text)
    if kind in ("int",):
        try:
            return int(float(text))
        except ValueError:
            return Formula(text)
    if kind == "bool":
        return _parse_bool(text)
    if kind in ("[int]", "polygon", "polygons", "vec3"):
        try:
            return json.loads(text)
        except ValueError:
            return Formula(text)
    return text


def _definition_value(props: dict):
    """The value a definition gives a setting: its formula, or else its default value."""
    if "value" in props:
        v = props["value"]
        return Formula(v) if isinstance(v, str) and props.get("type") not in ("str",) else (
            Formula(v) if isinstance(v, str) else v)
    return props.get("default_value", 0)


# ------------------------------------------------------------------ containers

@dataclass
class Definition:
    """A printer or extruder definition with everything it inherits merged into it."""

    id: str
    name: str
    metadata: dict
    settings: dict            # key -> merged properties (type, default_value, value, ...)


@dataclass
class Profile:
    """A variant, quality or material: a name, its metadata and the values it sets."""

    id: str
    name: str
    metadata: dict
    values: dict = field(default_factory=dict)


#: the layout of the cache `Library.load` writes; a cache of another layout is read again
CACHE_FORMAT = 1
#: how many files `Library.load` reads at the same time
READERS = 8
#: the folders of a library whose times tell whether a cache is still good
_FOLDERS = ("definitions", "extruders", "quality", "variants", "materials", "texts")


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Library:
    """Cura's profile library in `resources`: the printers and what they print with."""

    def __init__(self, resources: str):
        self.root = resources
        self._defs: dict[str, dict] = {}            # id -> the raw json, read once
        self._merged: dict[str, Definition] = {}
        self._qualities: list[Profile] | None = None
        self._materials: list[Profile] | None = None
        self._variants: list[Profile] | None = None

    # ---- definitions
    def _raw(self, did: str) -> dict:
        if did not in self._defs:
            for folder in ("definitions", "extruders"):
                path = os.path.join(self.root, folder, did + ".def.json")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        self._defs[did] = json.load(fh)
                    break
            else:
                raise KeyError(did)
        return self._defs[did]

    def definition(self, did: str) -> Definition:
        """The definition `did` with its parents merged in: metadata and setting properties."""
        if did in self._merged:
            return self._merged[did]
        raw = self._raw(did)
        if "inherits" in raw:
            parent = self.definition(raw["inherits"])
            metadata = dict(parent.metadata)
            settings = {k: dict(v) for k, v in parent.settings.items()}
        else:
            metadata, settings = {}, {}
        metadata.update(raw.get("metadata", {}))
        if "machine_extruder_trains" in raw.get("metadata", {}):
            metadata["machine_extruder_trains"] = dict(raw["metadata"]["machine_extruder_trains"])

        def walk(tree):
            for key, s in tree.items():
                if s.get("type") != "category":
                    settings[key] = {k: v for k, v in s.items() if k not in ("children",)}
                walk(s.get("children", {}))
        walk(raw.get("settings", {}))
        for key, o in raw.get("overrides", {}).items():
            settings.setdefault(key, {}).update(o)
        d = Definition(did, raw.get("name", did), metadata, settings)
        self._merged[did] = d
        return d

    @cached_property
    def machines(self) -> list[tuple[str, str, str]]:
        """Every printer you can choose: (id, name, manufacturer), sorted by maker and name."""
        found = []
        folder = os.path.join(self.root, "definitions")
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".def.json"):
                continue
            did = name[:-len(".def.json")]
            try:
                d = self.definition(did)
            except (KeyError, ValueError, OSError):
                continue
            meta = d.metadata
            if not _parse_bool(meta.get("visible", True)) or did in ("fdmprinter", "fdmextruder"):
                continue
            if meta.get("machine_extruder_trains") is None:
                continue
            found.append((did, d.name, str(meta.get("manufacturer", "Unknown"))))
        found.sort(key=lambda m: (m[2].lower(), m[1].lower()))
        return found

    # ---- instance profiles
    def _inst_files(self, folder):
        top = os.path.join(self.root, folder)
        for base, _dirs, files in os.walk(top):
            for name in files:
                if name.endswith(".inst.cfg"):
                    yield os.path.join(base, name)

    def _read_inst(self, path) -> Profile:
        cp = configparser.ConfigParser(interpolation=None)
        cp.optionxform = str
        with open(path, encoding="utf-8") as fh:
            cp.read_file(fh)
        meta = dict(cp["metadata"]) if cp.has_section("metadata") else {}
        general = dict(cp["general"]) if cp.has_section("general") else {}
        meta["definition"] = general.get("definition", "fdmprinter")
        values = dict(cp["values"]) if cp.has_section("values") else {}
        pid = os.path.basename(path)[:-len(".inst.cfg")]
        return Profile(pid, general.get("name", pid), meta, values)

    @property
    def qualities(self) -> list[Profile]:
        if self._qualities is None:
            self._qualities = [self._read_inst(p) for p in self._inst_files("quality")]
        return self._qualities

    @property
    def variants(self) -> list[Profile]:
        if self._variants is None:
            self._variants = [self._read_inst(p) for p in self._inst_files("variants")]
        return self._variants

    @property
    def materials(self) -> list[Profile]:
        """Every material file: the common values, and per printer in `metadata["machines"]`."""
        if self._materials is None:
            self._materials = []
            folder = os.path.join(self.root, "materials")
            products = self._product_ids()
            for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
                if name.endswith(".xml.fdm_material"):
                    try:
                        self._materials.append(_read_material(os.path.join(folder, name), products))
                    except (ET.ParseError, OSError, ValueError):
                        continue
        return self._materials

    # ---- everything at once
    def load(self, cache: str | None = None, stamp: str = "") -> bool:
        """Read the printers, nozzles, qualities and materials now, so choosing is quick.

        The files are read `READERS` at a time. With `cache` (a file path) they come from there
        instead when it was written for this library - the same folder, the same `stamp` (the
        version of the program, say) and the same times of its folders - and the cache is
        written after reading them otherwise. A cache that cannot be read or written is left
        alone. Returns whether the cache was used.
        """
        key = self._cache_key(stamp)
        if cache:
            got = _read_cache(cache, key)
            if got is not None:
                self.__dict__["machines"] = got[0]
                self._qualities, self._variants, self._materials = got[1:]
                return True
        with ThreadPoolExecutor(READERS) as pool:
            paths = []
            for folder in ("definitions", "extruders"):
                top = os.path.join(self.root, folder)
                names = sorted(os.listdir(top)) if os.path.isdir(top) else []
                paths += [(n[:-len(".def.json")], os.path.join(top, n))
                          for n in names if n.endswith(".def.json")]
            for (did, _path), raw in zip(paths, pool.map(self._try_json, [p for _d, p in paths])):
                if raw is not None:
                    self._defs.setdefault(did, raw)     # a printer before an extruder of that id
            self.machines
            if self._qualities is None:
                self._qualities = list(pool.map(self._read_inst, list(self._inst_files("quality"))))
            if self._variants is None:
                self._variants = list(pool.map(self._read_inst, list(self._inst_files("variants"))))
            if self._materials is None:
                folder = os.path.join(self.root, "materials")
                names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
                products = self._product_ids()
                read = pool.map(lambda n: _try_material(os.path.join(folder, n), products),
                                [n for n in names if n.endswith(".xml.fdm_material")])
                self._materials = [m for m in read if m is not None]
        if cache:
            _write_cache(cache, key, (self.machines, self._qualities, self._variants,
                                      self._materials))
        return False

    @staticmethod
    def _try_json(path):
        try:
            return _read_json(path)
        except (OSError, ValueError):
            return None

    def _cache_key(self, stamp: str) -> tuple:
        """What a cache must have been written for: this folder, `stamp` and its folder times."""
        times = []
        for folder in _FOLDERS:
            try:
                times.append(os.stat(os.path.join(self.root, folder)).st_mtime_ns)
            except OSError:
                times.append(0)
        return (CACHE_FORMAT, os.path.normcase(os.path.realpath(self.root)), str(stamp),
                tuple(times))

    def _product_ids(self) -> dict:
        path = os.path.join(self.root, "texts", "product_to_id.json")
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def printer(self, did: str) -> "Printer":
        return Printer(self, did)


def _product_to_ids(product: str, table: dict) -> list[str]:
    """The definition ids a product name in a material file stands for (Cura's product_to_id)."""
    if product in table:
        return [table[product]]
    for pattern, did in table.items():
        if "#" in pattern:
            rx = "^" + re.escape(pattern).replace("\\#", "([0-9]+)") + "$"
            m = re.match(rx, product)
            if m:
                return [did.replace("#", m.group(1))]
    return [product.lower().replace(" ", "_").replace("+", "_plus")]


def _try_material(path: str, products: dict) -> Profile | None:
    try:
        return _read_material(path, products)
    except (ET.ParseError, OSError, ValueError):
        return None


def _read_cache(path: str, key) -> tuple | None:
    """What `Library.load` cached at `path` for `key`, or None when it is not there or not that."""
    try:
        with open(path, "rb") as fh:
            got = pickle.load(fh)
    except Exception:                       # missing, cut short or from another version
        return None
    if not isinstance(got, tuple) or len(got) != 2 or got[0] != key:
        return None
    return got[1]


def _write_cache(path: str, key, payload) -> None:
    """Write `payload` for `key` to `path` whole or not at all; a folder we cannot write is fine."""
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "wb") as fh:
            pickle.dump((key, payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _read_material(path: str, products: dict) -> Profile:
    root = ET.parse(path).getroot()
    meta_el = root.find(_UM + "metadata")
    name_el = meta_el.find(_UM + "name")

    def text(el, tag, default=""):
        e = el.find(_UM + tag) if el is not None else None
        return e.text.strip() if e is not None and e.text else default
    brand, kind, color = text(name_el, "brand"), text(name_el, "material"), text(name_el, "color")
    props = root.find(_UM + "properties")
    diameter = float(text(props, "diameter", "2.85") or 2.85)
    density = float(text(props, "density", "1.24") or 1.24)
    base = os.path.basename(path)[:-len(".xml.fdm_material")]
    meta = {"base_file": base, "brand": brand, "material": kind, "color_name": color,
            "GUID": text(meta_el, "GUID"), "color_code": text(meta_el, "color_code", "#ffffff"),
            "approximate_diameter": int(round(diameter)), "diameter": diameter,
            "density": density, "machines": {}, "excluded": set()}
    values = {"material_diameter": diameter, "material_guid": meta["GUID"],
              "material_type": kind, "material_brand": brand}
    settings = root.find(_UM + "settings")
    if settings is not None:
        for s in settings.findall(_UM + "setting"):
            key = MATERIAL_SETTINGS.get(s.get("key", ""))
            if key and s.text:
                values[key] = s.text.strip()
        for machine in settings.findall(_UM + "machine"):
            own, hotends, compatible = {}, {}, True
            for s in machine.findall(_UM + "setting"):
                if s.get("key") == "hardware compatible":
                    compatible = _parse_bool(s.text)
                key = MATERIAL_SETTINGS.get(s.get("key", ""))
                if key and s.text:
                    own[key] = s.text.strip()
            for h in machine.findall(_UM + "hotend"):
                hv = {}
                for s in h.findall(_UM + "setting"):
                    key = MATERIAL_SETTINGS.get(s.get("key", ""))
                    if key and s.text:
                        hv[key] = s.text.strip()
                hotends[h.get("id", "")] = hv
            for ident in machine.findall(_UM + "machine_identifier"):
                for did in _product_to_ids(ident.get("product", ""), products):
                    if compatible:
                        meta["machines"][did] = (own, hotends)
                    else:
                        meta["excluded"].add(did)
    label = " ".join(p for p in (brand, kind, color) if p and p != "Generic") or kind
    if brand == "Generic":
        label = f"Generic {kind}"
    return Profile(base, label, meta, values)


# ------------------------------------------------------------------ a printer and its choices

class Printer:
    """One printer of the library and what you can choose for it.

    `variants` are its nozzles, `materials(variant)` the materials that fit its filament, and
    `qualities(variant, material)` the quality types every extruder has a profile for, as
    (quality_type, name, layer_height). `stacks(...)` builds the settings for one choice.
    """

    def __init__(self, library: Library, did: str):
        self.library, self.id = library, did
        self.definition = library.definition(did)
        meta = self.definition.metadata
        self.has_materials = _parse_bool(meta.get("has_materials", True))
        self.has_variants = _parse_bool(meta.get("has_variants", False))
        self.has_machine_quality = _parse_bool(meta.get("has_machine_quality", False))
        self.quality_definition = (meta.get("quality_definition", did) if self.has_machine_quality
                                   else "fdmprinter")
        self.exclude_materials = set(meta.get("exclude_materials", []))
        self.preferred_quality_type = meta.get("preferred_quality_type", "normal")
        self.preferred_variant_name = meta.get("preferred_variant_name", "")
        self.preferred_material = meta.get("preferred_material", "")
        trains = meta.get("machine_extruder_trains", {"0": "fdmextruder"})
        count = self._definition_number("machine_extruder_count", 1)
        self.extruders = [library.definition(trains[str(i)]) for i in range(int(count))
                          if str(i) in trains]

    def _definition_number(self, key, default):
        v = _definition_value(self.definition.settings.get(key, {}))
        return v if isinstance(v, (int, float)) else default

    @property
    def name(self) -> str:
        return self.definition.name

    # ---- nozzles
    @cached_property
    def variants(self) -> list[Profile]:
        """The nozzles of this printer (none when it has no variants)."""
        if not self.has_variants:
            return []
        found = [v for v in self.library.variants
                 if v.metadata.get("definition") == self.id
                 and v.metadata.get("hardware_type", "nozzle") == "nozzle"]
        return sorted(found, key=lambda v: v.name)

    def default_variant(self) -> Profile | None:
        for v in self.variants:
            if v.name == self.preferred_variant_name:
                return v
        return self.variants[0] if self.variants else None

    # ---- materials
    def filament_diameter(self) -> float:
        """The filament this printer takes: 1.75 or 2.85, from its first extruder."""
        stacks = Stacks(self, self.default_variant(), None, None)
        try:
            return float(stacks.extruders[0].get("material_diameter"))
        except (TypeError, ValueError, IndexError):
            return 2.85

    def materials(self, variant: Profile | None = None) -> list[Profile]:
        """The materials for this printer: of its filament diameter, not excluded, one per file."""
        if not self.has_materials:
            return []
        approx = int(round(self.filament_diameter()))
        found = []
        for m in self.library.materials:
            if m.metadata["approximate_diameter"] != approx:
                continue
            if m.id in self.exclude_materials or self.id in m.metadata["excluded"]:
                continue
            found.append(m)
        found.sort(key=lambda m: (m.metadata["brand"] != "Generic", m.name.lower()))
        return found

    def default_material(self, variant: Profile | None = None) -> Profile | None:
        mats = self.materials(variant)
        for m in mats:
            if m.id == self.preferred_material:
                return m
        for m in mats:
            if m.metadata["brand"] == "Generic" and m.metadata["material"] == "PLA":
                return m
        return mats[0] if mats else None

    # ---- qualities
    def _global_qualities(self) -> list[Profile]:
        qs = [q for q in self.library.qualities
              if q.metadata.get("definition") == self.quality_definition
              and _parse_bool(q.metadata.get("global_quality", False))]
        if not qs:
            qs = [q for q in self.library.qualities
                  if q.metadata.get("definition") == "fdmprinter"
                  and _parse_bool(q.metadata.get("global_quality", False))]
        return qs

    def _extruder_qualities(self, variant: Profile | None, material: Profile | None) -> list:
        """The extruder quality profiles for one nozzle and material, the way Cura picks them."""
        lib, qd = self.library, self.quality_definition
        if not self.has_machine_quality:
            return [q for q in lib.qualities if q.metadata.get("definition") == "fdmprinter"]
        pool = [q for q in lib.qualities if q.metadata.get("definition") == qd]
        if self.has_variants and variant is not None:
            pool_v = [q for q in pool if q.metadata.get("variant") == variant.name]
        else:
            pool_v = pool
        if not self.has_materials or material is None:
            return pool
        exact = [q for q in pool_v if q.metadata.get("material") == material.id]
        if exact:
            return exact
        kind, brand = material.metadata["material"], material.metadata["brand"]
        same_brand = {m.id for m in lib.materials
                      if m.metadata["material"] == kind and m.metadata["brand"] == brand}
        found = [q for q in pool_v if q.metadata.get("material") in same_brand]
        if not found:
            same_kind = {m.id for m in lib.materials if m.metadata["material"] == kind}
            found = [q for q in pool_v if q.metadata.get("material") in same_kind]
        if not found:
            found = [q for q in pool_v if _parse_bool(q.metadata.get("global_quality", False))]
        return found

    def qualities(self, variant: Profile | None = None,
                  material: Profile | None = None) -> list[tuple[str, str, float]]:
        """The quality types you can print with: (quality_type, name, layer_height), finest first."""
        ext = {q.metadata.get("quality_type") for q in self._extruder_qualities(variant, material)}
        out, seen = [], set()
        for q in self._global_qualities():
            qt = q.metadata.get("quality_type")
            if qt in seen or (self.has_machine_quality and self.has_materials and qt not in ext):
                continue
            seen.add(qt)
            try:
                lh = float(q.values.get("layer_height", "nan"))
            except ValueError:
                lh = math.nan
            out.append((qt, q.name, lh))
        if not out:                          # Cura shows "Not supported"; fall back to all
            for q in self._global_qualities():
                qt = q.metadata.get("quality_type")
                if qt not in seen:
                    seen.add(qt)
                    out.append((qt, q.name, float(q.values.get("layer_height", "nan") or "nan")))
        out.sort(key=lambda t: (math.isnan(t[2]), t[2]))
        return out

    def default_quality(self, variant=None, material=None) -> str | None:
        qs = self.qualities(variant, material)
        for q in qs:
            if q[0] == self.preferred_quality_type:
                return q[0]
        return qs[len(qs) // 2][0] if qs else None

    def stacks(self, variant=None, material=None, quality_type=None, user=None) -> "Stacks":
        """The settings for one choice; `user` is a dict of your own changes on top of all."""
        return Stacks(self, variant, material, quality_type, user or {})


# ------------------------------------------------------------------ evaluating the settings

class _Names(dict):
    """The names a formula sees: the settings of the stack that asked for it."""

    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def __getitem__(self, key):
        v = self.stack.get(key)
        if v is None:
            raise KeyError(key)
        return v


class Stack:
    """The layers of one extruder, or of the whole printer, and the values they give."""

    def __init__(self, owner: "Stacks", layers: list[dict], definition: dict, position: int | None):
        self.owner, self.layers, self.definition, self.position = owner, layers, definition, position
        self.cache: dict = {}
        self.busy: set = set()
        self.resolving: set = set()
        self.raw_cache: dict = {}

    @property
    def is_global(self):
        return self.position is None

    def props(self, key) -> dict:
        """The definition properties of a setting: from this stack's definition or the global one."""
        p = self.definition.get(key)
        if p is None and not self.is_global:
            p = self.owner.glob.definition.get(key)
        return p or {}

    def keys(self):
        ks = set(self.definition)
        for layer in self.layers:
            ks.update(layer)
        return ks

    def raw(self, key):
        """The raw value of a setting: the top layer that sets it, else the definition."""
        for layer in self.layers:
            if key in layer:
                return layer[key]
        if key in self.definition:
            return _definition_value(self.definition[key])
        if not self.is_global:
            return self.owner.glob.raw(key)
        return None

    def has_user_value(self, key) -> bool:
        return bool(self.layers) and key in self.layers[0]

    def get(self, key):
        """The value of a setting as this stack sees it, with every formula worked out."""
        if key in self.cache:
            return self.cache[key]
        if key in self.resolving:        # asked again while its resolve runs: the plain value
            return self._plain(key, self.props(key))
        if key in self.busy:             # a formula that needs itself: Cura gives up there too
            return None
        self.busy.add(key)
        try:
            v = self._get(key)
        finally:
            self.busy.discard(key)
        self.cache[key] = v
        return v

    def raw_get(self, key):
        """The value this stack's own layers give, worked out here: what `extruderValue` reads.

        Unlike `get` it does not hand a setting that is not per extruder to the global stack:
        Cura's extruderValue and extruderValues read the raw value of the extruder stack.
        """
        if key in self.raw_cache:
            return self.raw_cache[key]
        if ("raw", key) in self.busy:
            return None
        self.busy.add(("raw", key))
        try:
            v = self.value_of(self.raw(key), self.props(key).get("type"))
        finally:
            self.busy.discard(("raw", key))
        self.raw_cache[key] = v
        return v

    def _get(self, key):
        props = self.props(key)
        if not props and key not in self.keys() and (self.is_global or key not in
                                                      self.owner.glob.keys()):
            return None
        g = self.owner.glob
        if not self.is_global:
            if not _parse_bool(props.get("settable_per_extruder", True)):
                return g.get(key)
            limit = self._limit(props)
            if limit is not None and limit != self.position and 0 <= limit < len(self.owner.extruders):
                return self.owner.extruders[limit].get(key)
        else:
            if "resolve" in props and not self.has_user_value(key) and self.owner.extruders:
                self.resolving.add(key)
                try:
                    v = self.evaluate(Formula(props["resolve"]))
                finally:
                    self.resolving.discard(key)
                if v is not None:
                    return v
        return self._plain(key, props)

    def _plain(self, key, props):
        """The value without a resolve: from the extruder it is limited to, or from the layers."""
        if self.is_global:
            limit = self._limit(props)
            if (limit is not None and 0 <= limit < len(self.owner.extruders)
                    and _parse_bool(props.get("settable_per_extruder", True))):
                return self.owner.extruders[limit].get(key)
        return self.value_of(self.raw(key), props.get("type"))

    def _limit(self, props):
        lim = props.get("limit_to_extruder")
        if lim is None:
            return None
        v = self.evaluate(Formula(str(lim))) if isinstance(lim, str) else lim
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return v if v >= 0 else None

    def value_of(self, raw, kind=None):
        if isinstance(raw, Formula):
            return self.evaluate(raw)
        if isinstance(raw, str) and kind not in ("str", "enum", "extruder", "optional_extruder"):
            return _literal(raw, kind) if not raw.startswith("=") else self.evaluate(Formula(raw[1:]))
        if isinstance(raw, str) and raw.startswith("="):
            return self.evaluate(Formula(raw[1:]))
        return raw

    def evaluate(self, formula: Formula):
        if formula.code is None:
            return 0
        try:
            return eval(formula.code, self.owner.functions, _Names(self))   # noqa: S307
        except Exception:                    # a formula that fails is 0, as in Cura
            return 0


class Stacks:
    """The global stack and one stack per extruder for one choice of nozzle, material and quality."""

    def __init__(self, printer: Printer, variant, material, quality_type, user=None):
        self.printer = printer
        user = user or {}
        lib = printer.library
        gdef = printer.definition.settings
        self.quality_name = ""
        gquality = {}
        if quality_type is not None:
            for q in printer._global_qualities():
                if q.metadata.get("quality_type") == quality_type:
                    gquality, self.quality_name = q.values, q.name
                    break
        equality = {}
        if quality_type is not None:
            for q in printer._extruder_qualities(variant, material):
                if (q.metadata.get("quality_type") == quality_type
                        and not _parse_bool(q.metadata.get("global_quality", False))):
                    equality = q.values
                    break
        mvalues = {}
        if material is not None:
            mvalues = dict(material.values)
            own = material.metadata["machines"].get(printer.id)
            if own:
                mvalues.update(own[0])
                if variant is not None:
                    for hotend, hv in own[1].items():
                        if hotend and hotend.replace(" ", "") in variant.name.replace(" ", ""):
                            mvalues.update(hv)
        vvalues = dict(variant.values) if variant is not None else {}
        guser = {k: v for k, v in user.items()}
        self.glob = Stack(self, [guser, gquality], gdef, None)
        self.extruders = []
        for i, edef in enumerate(printer.extruders):
            layers = [dict(user), equality, mvalues, vvalues]
            settings = dict(edef.settings)
            self.extruders.append(Stack(self, layers, settings, i))
        self.material, self.variant = material, variant
        self.jobname = "Volura"
        self.functions = self._functions()

    def _functions(self) -> dict:
        def extruder_value(position, key):
            try:
                pos = int(position)
            except (TypeError, ValueError):
                pos = 0
            if pos < 0:
                pos = 0
            if pos >= len(self.extruders):
                pos = 0
            v = self.extruders[pos].raw_get(key) if self.extruders else self.glob.get(key)
            return v.lower() if isinstance(v, str) else v

        def extruder_values(key):
            out = [e.raw_get(key) for e in self.extruders]
            out = [v for v in out if v is not None]
            return out or [self.glob.get(key)]

        def any_with_or_default(key):
            for i, e in enumerate(self.extruders):
                if e.raw_get(key):
                    return str(i)
            return "0"

        def any_with_material(key):
            if self.material is not None and key in self.material.values:
                return "0"
            return "0"

        funcs = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        funcs.update({"math": math, "extruderValue": extruder_value,
                      "extruderValues": extruder_values,
                      "resolveOrValue": lambda key: self.glob.get(key),
                      "defaultExtruderPosition": lambda: "0",
                      "anyExtruderWithMaterial": any_with_material,
                      "anyExtruderNrWithOrDefault": any_with_or_default,
                      "valueFromContainer": lambda *a: None,
                      "extruderValueFromContainer": lambda *a: None,
                      "debug": lambda *a: None})
        return funcs

    # ---- what CuraEngine gets
    def global_settings(self) -> dict:
        """Every setting of the global stack, worked out, as CuraEngine's global settings."""
        keys = set(self.glob.keys())
        for e in self.extruders:
            keys.update(e.definition)
        out = {}
        for k in sorted(keys):
            v = self.glob.get(k)
            if v is None and self.extruders:
                v = self.extruders[0].get(k)
            if v is not None:
                out[k] = v
        out["print_bed_temperature"] = out.get("material_bed_temperature")
        out["quality_name"] = self.quality_name
        out["quality_changes_name"] = ""
        # what a start g-code may ask for besides the settings (Cura's replacement tokens)
        now = time.localtime()
        out["time"] = time.strftime("%H:%M:%S", now)
        out["date"] = time.strftime("%d-%m-%Y", now)
        out["day"] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][int(time.strftime("%w", now))]
        out["jobname"] = self.jobname
        return out

    def extruder_settings(self, i: int) -> dict:
        """Every setting extruder `i` may set, worked out from its own stack."""
        e = self.extruders[i]
        keys = set(e.definition) | set(self.glob.keys())
        for layer in e.layers:
            keys.update(layer)
        out = {}
        for k in sorted(keys):
            props = e.props(k)
            if props and not _parse_bool(props.get("settable_per_extruder", True)):
                continue
            v = e.get(k)
            if v is not None:
                out[k] = v
        out["print_temperature"] = out.get("material_print_temperature")
        out["travel_speed"] = out.get("speed_travel")
        if self.material is not None:
            out["material_id"] = self.material.id
            out["material_type"] = self.material.metadata["material"]
            out["material_name"] = self.material.name
            out["material_brand"] = self.material.metadata["brand"]
            out["material_density"] = self.material.metadata["density"]
        return out


def engine_text(value) -> str:
    """A value the way Cura hands it to CuraEngine: `str()` of the Python value."""
    return str(value)
