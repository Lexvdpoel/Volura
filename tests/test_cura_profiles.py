"""Printer profiles for slicing: Cura's definitions, nozzles, materials and qualities.

`cura_profiles.Library` reads a Cura `resources` folder and `Printer.stacks` turns one choice of
nozzle, material and quality into the worked-out settings CuraEngine slices with. These tests
build a small library of their own, so every rule can be seen at work: a formula is worked out
from the extruder that asks for it, the topmost layer wins (your own value, the quality, the
material, the nozzle, the definition), a setting that is not per extruder comes from the global
stack, `resolve` and `limit_to_extruder` behave as in Cura, a formula that fails counts as 0, and
the materials and qualities a printer offers are the ones Cura would offer.

At the end, when the Cura library is there (tools/fetch_curaengine.py), the same is checked
against the real profiles.

Run with:  python -m pytest tests/test_cura_profiles.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import cura_profiles, slicer  # noqa: E402

FDMPRINTER = {
    "name": "FFF Printer", "version": 2,
    "metadata": {"visible": False, "has_materials": True, "preferred_material": "generic_pla"},
    "settings": {
        "machine_settings": {"type": "category", "children": {
            "machine_width": {"type": "float", "default_value": 100,
                              "settable_per_extruder": False},
            "machine_extruder_count": {"type": "int", "default_value": 1,
                                       "settable_per_extruder": False},
            "machine_nozzle_size": {"type": "float", "default_value": 0.4},
            "material_diameter": {"type": "float", "default_value": 2.85},
            "machine_start_gcode": {"type": "str", "default_value": "G28",
                                    "settable_per_extruder": False}}},
        "resolution": {"type": "category", "children": {
            "layer_height": {"type": "float", "default_value": 0.1,
                             "settable_per_extruder": False},
            "line_width": {"type": "float", "default_value": 0.4, "value": "machine_nozzle_size"},
            "wall_thickness": {"type": "float", "default_value": 0.8,
                               "value": "line_width * 2"},
            "broken": {"type": "float", "default_value": 5, "value": "no_such_setting * 2"}}},
        "material": {"type": "category", "children": {
            "default_material_print_temperature": {"type": "float", "default_value": 210},
            "material_print_temperature": {"type": "float", "default_value": 210,
                                           "value": "default_material_print_temperature"},
            "default_material_bed_temperature": {"type": "float", "default_value": 60,
                                                 "settable_per_extruder": False},
            "material_bed_temperature": {"type": "float", "default_value": 60,
                                         "value": "default_material_bed_temperature",
                                         "settable_per_extruder": False}}},
        "speed": {"type": "category", "children": {
            "speed_print": {"type": "float", "default_value": 50},
            "speed_wall": {"type": "float", "default_value": 25, "value": "speed_print / 2"},
            "speed_travel": {"type": "float", "default_value": 150}}},
        "support": {"type": "category", "children": {
            "support_extruder_nr": {"type": "extruder", "default_value": "0",
                                    "settable_per_extruder": False},
            "support_line_width": {"type": "float", "default_value": 0.4,
                                   "value": "line_width",
                                   "limit_to_extruder": "support_extruder_nr"}}},
        "adhesion": {"type": "category", "children": {
            "adhesion_type": {"type": "enum", "default_value": "brim",
                              "resolve": "min(extruderValues('adhesion_type'), "
                                         "key=('raft', 'brim', 'skirt', 'none').index)",
                              "settable_per_extruder": False}}},
    },
}
FDMEXTRUDER = {"name": "Extruder", "version": 2, "metadata": {"visible": False},
               "settings": {"machine_settings": {"type": "category", "children": {
                   "extruder_nr": {"type": "extruder", "default_value": "0"}}}}}
PRINTER = {"name": "Test Printer", "version": 2, "inherits": "fdmprinter",
           "metadata": {"visible": True, "manufacturer": "Volura", "has_variants": True,
                        "has_machine_quality": True, "preferred_variant_name": "0.4mm Nozzle",
                        "preferred_quality_type": "standard",
                        "machine_extruder_trains": {"0": "testprinter_extruder_0"}},
           "overrides": {"machine_width": {"default_value": 220},
                         "material_diameter": {"default_value": 1.75}}}
EXTRUDER = {"name": "Extruder 1", "version": 2, "inherits": "fdmextruder",
            "metadata": {"machine": "testprinter", "position": "0"},
            "overrides": {"extruder_nr": {"default_value": 0}}}
DUAL = {"name": "Dual Printer", "version": 2, "inherits": "fdmprinter",
        "metadata": {"visible": True, "manufacturer": "Volura",
                     "machine_extruder_trains": {"0": "dual_extruder_0",
                                                 "1": "dual_extruder_1"}},
        "overrides": {"machine_extruder_count": {"default_value": 2},
                      "support_extruder_nr": {"default_value": "1"},
                      "material_diameter": {"default_value": 1.75}}}


def _inst(path, general, metadata, values):
    lines = ["[general]"] + [f"{k} = {v}" for k, v in general.items()] + ["", "[metadata]"]
    lines += [f"{k} = {v}" for k, v in metadata.items()] + ["", "[values]"]
    lines += [f"{k} = {v}" for k, v in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _material(path, base, kind, diameter, temperature, machine_temperature=None):
    machine = ""
    if machine_temperature is not None:
        machine = (f'<machine><machine_identifier manufacturer="Volura" product="testprinter"/>'
                   f'<setting key="print temperature">{machine_temperature}</setting></machine>')
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<fdmmaterial xmlns="http://www.ultimaker.com/material" version="1.3">
  <metadata><name><brand>Generic</brand><material>{kind}</material><color>Generic</color></name>
    <GUID>{base}-guid</GUID><version>1</version><color_code>#ffffff</color_code></metadata>
  <properties><density>1.24</density><diameter>{diameter}</diameter></properties>
  <settings><setting key="print temperature">{temperature}</setting>
    <setting key="heated bed temperature">55</setting>{machine}</settings>
</fdmmaterial>
""", encoding="utf-8")


@pytest.fixture
def library(tmp_path):
    d = tmp_path / "definitions"
    e = tmp_path / "extruders"
    d.mkdir()
    e.mkdir()
    for name, doc in (("fdmprinter", FDMPRINTER), ("fdmextruder", FDMEXTRUDER),
                      ("testprinter", PRINTER), ("dual", DUAL)):
        (d / f"{name}.def.json").write_text(json.dumps(doc), encoding="utf-8")
    (e / "testprinter_extruder_0.def.json").write_text(json.dumps(EXTRUDER), encoding="utf-8")
    for i in (0, 1):
        doc = dict(EXTRUDER, metadata={"machine": "dual", "position": str(i)},
                   overrides={"extruder_nr": {"default_value": i}})
        (e / f"dual_extruder_{i}.def.json").write_text(json.dumps(doc), encoding="utf-8")
    for size in ("0.4", "0.6"):
        _inst(tmp_path / "variants" / "test" / f"testprinter_{size}.inst.cfg",
              {"definition": "testprinter", "name": f"{size}mm Nozzle", "version": 4},
              {"hardware_type": "nozzle", "type": "variant"}, {"machine_nozzle_size": size})
    q = tmp_path / "quality"
    _inst(q / "test" / "global_standard.inst.cfg",
          {"definition": "testprinter", "name": "Standard Quality", "version": 4},
          {"global_quality": "True", "quality_type": "standard", "type": "quality"},
          {"layer_height": "0.2", "wall_thickness": "=line_width * 3"})
    _inst(q / "test" / "global_draft.inst.cfg",
          {"definition": "testprinter", "name": "Draft Quality", "version": 4},
          {"global_quality": "True", "quality_type": "draft", "type": "quality"},
          {"layer_height": "0.3"})
    _inst(q / "test" / "global_fine.inst.cfg",
          {"definition": "testprinter", "name": "Fine Quality", "version": 4},
          {"global_quality": "True", "quality_type": "fine", "type": "quality"},
          {"layer_height": "0.1"})
    for qt in ("standard", "draft"):
        _inst(q / "test" / f"pla_{qt}_0.4.inst.cfg",
              {"definition": "testprinter", "name": qt, "version": 4},
              {"quality_type": qt, "material": "generic_pla_175", "variant": "0.4mm Nozzle",
               "type": "quality"}, {"speed_print": "40"})
    _inst(q / "draft.inst.cfg", {"definition": "fdmprinter", "name": "Draft", "version": 4},
          {"global_quality": "True", "quality_type": "draft", "type": "quality"},
          {"layer_height": "0.25"})
    m = tmp_path / "materials"
    m.mkdir()
    _material(m / "generic_pla_175.xml.fdm_material", "generic_pla_175", "PLA", 1.75, 200, 215)
    _material(m / "generic_petg_175.xml.fdm_material", "generic_petg_175", "PETG", 1.75, 240)
    _material(m / "generic_pla.xml.fdm_material", "generic_pla", "PLA", 2.85, 200)
    return cura_profiles.Library(str(tmp_path))


def _choose(lib, did="testprinter", nozzle="0.4mm Nozzle", material="generic_pla_175",
            quality="standard", user=None):
    p = lib.printer(did)
    v = next((x for x in p.variants if x.name == nozzle), None)
    mat = next((x for x in p.materials(v) if x.id == material), None)
    return p.stacks(v, mat, quality, user)


# ------------------------------------------------------------------ the library

def test_only_printers_you_can_choose_are_listed(library):
    ids = [m[0] for m in library.machines]
    assert ids == ["dual", "testprinter"], "base definitions are not printers"
    assert ("testprinter", "Test Printer", "Volura") in library.machines


def test_a_printer_offers_its_own_nozzles_and_prefers_the_one_it_names(library):
    p = library.printer("testprinter")
    assert [v.name for v in p.variants] == ["0.4mm Nozzle", "0.6mm Nozzle"]
    assert p.default_variant().name == "0.4mm Nozzle"


def test_a_printer_offers_the_materials_of_its_filament_only(library):
    p = library.printer("testprinter")
    ids = [m.id for m in p.materials(p.default_variant())]
    assert set(ids) == {"generic_pla_175", "generic_petg_175"}, "a 2.85 mm material for 1.75"
    assert p.default_material().id == "generic_pla_175"


def test_a_quality_is_offered_only_where_the_material_has_a_profile_for_it(library):
    p = library.printer("testprinter")
    v = p.default_variant()
    pla = next(m for m in p.materials(v) if m.id == "generic_pla_175")
    types = [q[0] for q in p.qualities(v, pla)]
    assert types == ["standard", "draft"], "sorted by layer height, without 'fine'"
    assert p.qualities(v, pla)[0] == ("standard", "Standard Quality", 0.2)
    assert p.default_quality(v, pla) == "standard"


def test_a_material_without_profiles_borrows_those_of_its_kind(library):
    """PETG has no quality profiles here; Cura then falls back to the global qualities."""
    p = library.printer("testprinter")
    v = p.default_variant()
    petg = next(m for m in p.materials(v) if m.id == "generic_petg_175")
    assert {q[0] for q in p.qualities(v, petg)} >= {"standard", "draft"}


def test_a_printer_without_own_qualities_uses_the_general_ones(library):
    p = library.printer("dual")
    assert [q[0] for q in p.qualities()] == ["draft"]
    st = p.stacks(None, p.default_material(), "draft")
    assert st.glob.get("layer_height") == pytest.approx(0.25)


# ------------------------------------------------------------------ the stacks

def test_the_quality_sets_the_layer_height(library):
    assert _choose(library).glob.get("layer_height") == pytest.approx(0.2)
    assert _choose(library, quality="draft").glob.get("layer_height") == pytest.approx(0.3)


def test_a_formula_is_worked_out_from_the_extruder_that_asks(library):
    """The quality says wall_thickness = line_width * 3, and line_width is the nozzle size."""
    st = _choose(library, nozzle="0.6mm Nozzle")
    e = st.extruders[0]
    assert e.get("line_width") == pytest.approx(0.6)
    assert e.get("wall_thickness") == pytest.approx(1.8)


def test_the_extruder_quality_and_formulas_on_top_of_it(library):
    e = _choose(library).extruders[0]
    assert e.get("speed_print") == pytest.approx(40.0), "the extruder quality sets the speed"
    assert e.get("speed_wall") == pytest.approx(20.0), "speed_wall follows the speed"


def test_the_material_sets_its_temperatures_and_the_printer_its_own(library):
    st = _choose(library)
    e = st.extruders[0]
    assert e.get("default_material_print_temperature") == pytest.approx(215), \
        "the material file has its own temperature for this printer"
    assert e.get("material_print_temperature") == pytest.approx(215)
    assert st.glob.get("material_bed_temperature") == pytest.approx(60), \
        "not per extruder: from the global stack, which has no material"


def test_your_own_values_win(library):
    st = _choose(library, user={"layer_height": 0.12, "speed_print": 70.0})
    assert st.glob.get("layer_height") == pytest.approx(0.12)
    assert st.extruders[0].get("speed_wall") == pytest.approx(35.0)


def test_a_setting_not_per_extruder_comes_from_the_global_stack(library):
    st = _choose(library, user={})
    st.extruders[0].layers[0]["machine_width"] = 999       # an extruder may not say it
    assert st.extruders[0].get("machine_width") == pytest.approx(220)


def test_resolve_picks_over_all_extruders(library):
    p = library.printer("dual")
    st = p.stacks(None, None, "draft")
    st.extruders[1].layers[0]["adhesion_type"] = "raft"
    assert st.glob.get("adhesion_type") == "raft", "raft wins over brim in the resolve"
    st = p.stacks(None, None, "draft", {"adhesion_type": "skirt"})
    assert st.glob.get("adhesion_type") == "skirt", "your own value is not resolved"


def test_limit_to_extruder_takes_the_value_of_that_extruder(library):
    p = library.printer("dual")
    st = p.stacks(None, None, "draft")
    st.extruders[1].layers[0]["line_width"] = 0.8
    assert st.extruders[0].get("support_line_width") == pytest.approx(0.8), \
        "support is printed by extruder 1 here"
    assert st.glob.get("support_line_width") == pytest.approx(0.8)


def test_a_formula_that_fails_counts_as_zero(library):
    assert _choose(library).extruders[0].get("broken") == 0


def test_what_curaengine_gets(library):
    st = _choose(library)
    g = st.global_settings()
    e = st.extruder_settings(0)
    assert g["layer_height"] == pytest.approx(0.2) and g["machine_width"] == pytest.approx(220)
    assert {"date", "time", "day", "jobname", "print_bed_temperature"} <= set(g), \
        "the tokens a start g-code may use"
    assert "machine_width" not in e, "an extruder gets only what it may set"
    assert e["material_print_temperature"] == pytest.approx(215)
    assert e["print_temperature"] == pytest.approx(215)
    assert e["material_type"] == "PLA"
    doc = slicer.resolved_settings(st)
    assert set(doc) == {"global", "extruder.0"}
    json.dumps(doc)                                         # all of it is plain JSON


# ------------------------------------------------------------------ reading it all at once

def _everything(lib):
    return (lib.machines, [(p.id, p.name, p.metadata, p.values) for p in lib.qualities],
            [(p.id, p.metadata, p.values) for p in lib.variants],
            [(p.id, p.metadata, p.values) for p in lib.materials])


def test_load_reads_what_the_lazy_library_reads(library):
    lazy = _everything(cura_profiles.Library(library.root))
    assert library.load() is False
    assert _everything(library) == lazy


def test_load_keeps_a_cache_and_reads_it_back(library, tmp_path):
    cache = str(tmp_path / "cache" / "profiles.cache")
    assert library.load(cache, "1.0") is False, "there was no cache yet"
    again = cura_profiles.Library(library.root)
    assert again.load(cache, "1.0") is True, "the cache was not used"
    assert _everything(again) == _everything(library)
    st = _choose(again)                                     # and it slices the same
    assert st.global_settings()["layer_height"] == pytest.approx(0.2)


def test_a_cache_of_another_version_or_a_broken_one_is_read_again(library, tmp_path):
    cache = str(tmp_path / "profiles.cache")
    library.load(cache, "1.0")
    assert cura_profiles.Library(library.root).load(cache, "1.1") is False
    assert cura_profiles.Library(library.root).load(cache, "1.1") is True, "not written anew"
    with open(cache, "wb") as fh:
        fh.write(b"not a cache")
    fresh = cura_profiles.Library(library.root)
    assert fresh.load(cache, "1.1") is False
    assert _everything(fresh) == _everything(library)


def test_a_cache_is_read_again_when_the_profiles_change(library, tmp_path):
    cache = str(tmp_path / "profiles.cache")
    library.load(cache)
    folder = os.path.join(library.root, "quality")
    st = os.stat(folder)
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert cura_profiles.Library(library.root).load(cache) is False


def test_a_cache_that_cannot_be_written_is_no_problem(library, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert library.load(str(blocker / "sub" / "profiles.cache")) is False
    assert library.machines


# ------------------------------------------------------------------ the real profiles

REAL = slicer.find_resources()
needs_real = pytest.mark.skipif(REAL is None, reason="no Cura profiles (tools/fetch_curaengine.py)")


@pytest.fixture(scope="module")
def real():
    return cura_profiles.Library(REAL)


@needs_real
def test_the_real_library_has_hundreds_of_printers(real):
    assert len(real.machines) > 400
    makers = {m[2] for m in real.machines}
    assert {"Creality3D", "Prusa3D", "Ultimaker B.V."} & makers


@needs_real
def test_the_ender_3_slices_at_standard_quality_with_its_own_settings(real):
    p = real.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    assert v.name == "0.4mm Nozzle" and m.id == "generic_pla_175"
    q = p.default_quality(v, m)
    assert q == "standard"
    st = p.stacks(v, m, q)
    g, e = st.global_settings(), st.extruder_settings(0)
    assert g["layer_height"] == pytest.approx(0.2)
    assert g["machine_width"] == pytest.approx(235)
    assert e["line_width"] == pytest.approx(0.4)
    assert e["wall_thickness"] == pytest.approx(0.8)
    assert e["material_diameter"] == pytest.approx(1.75)
    assert g["adhesion_type"] in ("skirt", "brim", "raft", "none")
