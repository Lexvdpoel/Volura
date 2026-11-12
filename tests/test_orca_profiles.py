"""The printers of OrcaSlicer's profiles: their G-code language and how they become Cura printers.

`volura.orca_gcode` works out OrcaSlicer's start and end G-code (placeholders, expressions,
conditions); tools/fetch_orca_profiles.py turns OrcaSlicer's printer presets into Cura printer
definitions with a nozzle per preset. The tests build a small profiles folder of their own; at
the end, when the real profiles are there, the Flashforge printers are sliced.

Run with:  python -m pytest tests/test_orca_profiles.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fetch_orca_profiles as fop  # noqa: E402
from volura import cura_profiles, slicer  # noqa: E402
from volura.orca_gcode import OrcaGcodeError, names_used, points, render, typed, variables  # noqa: E402

V = {"bed": 60, "temps": [210, 220], "name": "PLA", "lh": 0.2, "area": [-110.0, 110.0],
     "on": True, "n": 7}


# ------------------------------------------------------------------ the G-code language

@pytest.mark.parametrize("template, out", [
    ("M190 S[bed]", "M190 S60"),
    ("M109 S[temps]", "M109 S210"),                       # a list without index: its first
    ("M109 S[temps_1]", "M109 S220"),                     # [name_1] is [name[1]]
    ("M109 S{temps[1]}", "M109 S220"),
    ("G1 Z{lh + 0.1}", "G1 Z0.3"),
    ("{bed / 7}", "8"),                                   # whole numbers divide as in C++
    ("{bed / 8.0}", "7.5"),
    ("{max(bed, 70)} {min(lh, 1)}", "70 0.2"),
    ("{int(2.7)} {round(2.5)}", "2 3"),
    ("{if bed > 50}hot{else}cold{endif}", "hot"),
    ("{if bed > 70}a{elsif bed > 50}b{else}c{endif}", "b"),
    ("{if name == \"PLA\" and on}pla{endif}", "pla"),
    ("{if !on || n % 2 == 0}x{else}y{endif}", "y"),
    ("{if name =~ /P.A/}match{endif}", "match"),
    ("{(bed > 50 ? 1 : 2)}", "1"),
    ("{local x = bed * 2}X{x}", "X120"),
    ("{if bed > 50}{if n > 9}a{else}b{endif}{endif}", "b"),
    ("SET [unknown] here", "SET [unknown] here"),         # not ours: plain G-code
    ("{digits(lh, 4, 2)}", "0.20"),
    ("{area[0]}x{area[1]}", "-110x110"),
    ("{\"a\" + name}", "aPLA"),
])
def test_render(template, out):
    assert render(template, V) == out


@pytest.mark.parametrize("template", ["{nobody}", "{if bed}x", "{endif}", "{bed +}", "{bed",
                                      "{foo(1)}", "{bed / 0}"])
def test_what_cannot_be_worked_out_says_so(template):
    with pytest.raises(OrcaGcodeError):
        render(template, V)


def test_names_used():
    assert names_used("M190 S[bed]\n{if temps[0] > 1}G1 Z{lh}{endif}[x_0]") == \
        {"bed", "temps", "lh", "x"}


def test_values_of_a_profile_are_typed():
    assert typed(["210", "0.4", "klipper", "true"]) == [210, 0.4, "klipper", True]
    assert points(["-110x-110", "110x110", "", "bad"]) == [(-110.0, -110.0), (110.0, 110.0)]


def test_variables_put_the_print_over_the_profile():
    v = variables({"nozzle_diameter": ["0.4", "0.4"], "printable_area": ["0x0", "256x256"],
                   "printable_height": "250", "z_offset": "0.1"},
                  nozzle_temp0=215, nozzle_temp=210, bed_temp0=65, bed_temp=60, material="PETG",
                  layer_height=0.2, first_layer_height=0.25, nozzle=0.6, retraction=0.8,
                  travel_speed=200, print_min=(10, 20), print_max=(30, 50), top=12.0,
                  layers=60, now=time.localtime(0))
    assert v["nozzle_temperature_initial_layer"] == [215, 215]
    assert v["bed_temperature_initial_layer_single"] == 65
    assert v["filament_type"] == ["PETG", "PETG"]
    assert v["first_layer_print_size"] == [20.0, 30.0]
    assert v["print_bed_max"] == [256.0, 256.0]
    assert v["max_print_height"] == 250 and v["z_offset"] == 0.1
    assert v["total_layer_count"] == 60 and v["max_layer_z"] == 12.0


# ------------------------------------------------------------------ the converter

START = ("M190 S[bed_temperature_initial_layer_single]\n"
         "M109 S[nozzle_temperature_initial_layer]\n"
         "{if curr_bed_type == \"Textured PEI Plate\"}G1 Z{0.2 + z_offset}{endif}\n"
         "G1 X{first_layer_print_min[0]} F{travel_speed * 60}")


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def converted(tmp_path):
    prof = tmp_path / "profiles"
    _write(prof / "Acme.json", {"name": "Acme"})
    _write(prof / "Custom.json", {"name": "Custom Printer"})
    m = prof / "Acme" / "machine"
    _write(m / "fdm_acme_common.json", {
        "type": "machine", "name": "fdm_acme_common", "instantiation": "false",
        "printable_area": ["-110x-110", "110x-110", "110x110", "-110x110"],
        "printable_height": "220", "gcode_flavor": "klipper", "retraction_length": ["0.8"],
        "retraction_speed": ["35"], "z_hop": ["0.4"], "machine_start_gcode": START,
        "machine_end_gcode": "M104 S0", "default_print_profile": "0.20mm Standard @Acme",
        "default_filament_profile": ["Acme PLA"]})
    for nozzle in ("0.4", "0.6"):
        _write(m / f"Acme One {nozzle} Nozzle.json", {
            "type": "machine", "name": f"Acme One {nozzle} Nozzle", "inherits": "fdm_acme_common",
            "instantiation": "true", "printer_model": "Acme One", "nozzle_diameter": [nozzle]})
    _write(m / "Acme Odd 0.4 Nozzle.json", {
        "type": "machine", "name": "Acme Odd 0.4 Nozzle", "inherits": "fdm_acme_common",
        "instantiation": "true", "printer_model": "Acme Odd", "nozzle_diameter": ["0.4"],
        "machine_start_gcode": "M109 S{not_a_setting}"})
    _write(m / "Acme Dup 0.4 Nozzle.json", {
        "type": "machine", "name": "Acme Dup 0.4 Nozzle", "inherits": "fdm_acme_common",
        "instantiation": "true", "printer_model": "Acme Dup", "nozzle_diameter": ["0.4"]})
    _write(m / "Acme Edge 0.4 Nozzle.json", {
        "type": "machine", "name": "Acme Edge 0.4 Nozzle", "inherits": "fdm_acme_common",
        "instantiation": "true", "printer_model": "Acme Edge", "nozzle_diameter": ["0.4"],
        "printable_area": ["4x6", "250x6", "250x210", "4x210"]})
    _write(prof / "Custom" / "machine" / "Custom 0.4.json", {
        "type": "machine", "name": "Custom 0.4", "instantiation": "true",
        "printer_model": "Custom", "nozzle_diameter": ["0.4"],
        "printable_area": ["0x0", "200x0", "200x200", "0x200"]})
    _write(prof / "Acme" / "process" / "standard.json", {
        "type": "process", "name": "0.20mm Standard @Acme", "outer_wall_speed": "200",
        "inner_wall_speed": "300", "travel_speed": "500", "use_relative_e_distances": "1"})
    _write(prof / "Acme" / "filament" / "pla.json", {
        "type": "filament", "name": "Acme PLA", "filament_diameter": ["1.75"],
        "filament_max_volumetric_speed": ["20"]})
    res = tmp_path / "resources"
    _write(res / "definitions" / "acme_dup.def.json",
           {"name": "Acme Dup", "metadata": {"manufacturer": "Acme Corp"}})
    log = []
    made = fop.convert(prof, res, log=log.append)
    return res, made, log


def test_every_new_printer_model_becomes_one_cura_printer(converted):
    res, made, log = converted
    assert made == ["orca_acme_edge", "orca_acme_one"]
    assert any("Acme Odd" in line and "not_a_setting" in line for line in log)


def test_the_printer_definition(converted):
    res, _made, _log = converted
    d = json.loads((res / "definitions" / "orca_acme_one.def.json").read_text(encoding="utf-8"))
    o, meta = d["overrides"], d["metadata"]
    assert d["name"] == "Acme One" and d["inherits"] == "fdmprinter"
    assert meta["manufacturer"] == "Acme Corp"               # Cura's spelling of the maker
    assert o["machine_width"]["default_value"] == 220 and o["machine_depth"]["default_value"] == 220
    assert o["machine_center_is_zero"]["default_value"] is True
    assert o["machine_height"]["default_value"] == 220
    assert o["machine_gcode_flavor"]["default_value"] == "RepRap (Marlin/Sprinter)"
    assert o["retraction_hop_enabled"]["default_value"] is True
    assert o["relative_extrusion"]["value"] == "True"
    assert o["speed_travel"]["value"] == "500"
    assert o["speed_wall_0"]["value"].startswith("min(200, 20 /")
    assert meta["orca_start_gcode"] == START
    assert meta["preferred_variant_name"] == "0.4 mm"
    assert "printable_area" in meta["orca_config"]          # what the G-code may ask for
    assert "outer_wall_speed" not in meta["orca_config"]    # ... and not the rest
    e = json.loads((res / "extruders" / "orca_acme_one_extruder_0.def.json")
                   .read_text(encoding="utf-8"))
    assert e["metadata"]["machine"] == "orca_acme_one"
    assert e["overrides"]["material_diameter"]["default_value"] == 1.75


def test_a_nozzle_per_preset(converted):
    res, _made, _log = converted
    names = sorted(p.name for p in (res / "variants").glob("orca_acme_one_*.inst.cfg"))
    assert names == ["orca_acme_one_0_4_mm.inst.cfg", "orca_acme_one_0_6_mm.inst.cfg"]
    text = (res / "variants" / "orca_acme_one_0_6_mm.inst.cfg").read_text(encoding="utf-8")
    assert "name = 0.6 mm" in text and "machine_nozzle_size = 0.6" in text


def test_a_bed_that_starts_away_from_zero(converted):
    res, _made, _log = converted
    d = json.loads((res / "definitions" / "orca_acme_edge.def.json").read_text(encoding="utf-8"))
    assert d["overrides"]["machine_width"]["default_value"] == 250
    assert d["overrides"]["machine_center_is_zero"]["default_value"] is False
    assert d["metadata"]["orca_offset"] == [2.0, 3.0]


@pytest.mark.parametrize("preset, label", [("Acme One 0.4 Nozzle", "0.4 mm"),
                                           ("Acme One 0.6 HF nozzle", "0.6 mm HF"),
                                           ("Something else", "0.4 mm")])
def test_nozzle_label(preset, label):
    assert fop.nozzle_label(preset, "Acme One", 0.4 if "0.6" not in preset else 0.6) == label


# ------------------------------------------------------------------ the real profiles

RES = slicer.find_resources()
HAS_ORCA = RES is not None and os.path.isfile(os.path.join(RES, "ORCA_VERSION"))
ENGINE = slicer.find_engine()
needs_orca = pytest.mark.skipif(not HAS_ORCA, reason="no OrcaSlicer printers "
                                                     "(tools/fetch_orca_profiles.py)")


@needs_orca
def test_the_versions_agree():
    sys.path.insert(0, os.path.join(ROOT, "packaging"))
    import licenses
    assert licenses.ORCA_VERSION == fop.ORCA_VERSION
    with open(os.path.join(RES, "ORCA_VERSION"), encoding="utf-8") as fh:
        assert fh.read().strip() == fop.ORCA_VERSION


@needs_orca
@pytest.mark.skipif(ENGINE is None or (ENGINE and slicer.engine_problem(ENGINE) is not None),
                    reason="no CuraEngine that runs here")
@pytest.mark.parametrize("did, start", [
    ("orca_flashforge_adventurer_5m_pro", "G1 X110 Y-110 F6000"),
    ("orca_flashforge_creator_5_pro", "M109 S200 T0")])
def test_the_flashforge_printers_slice(tmp_path, did, start):
    lib = cura_profiles.Library(RES)
    assert did in {m[0] for m in lib.machines}
    p = lib.printer(did)
    v = p.default_variant()
    m = p.default_material(v)
    assert v.name == "0.4 mm" and m.metadata["material"] == "PLA"
    st = p.stacks(v, m, p.default_quality(v, m))
    assert st.glob.get("layer_height") == pytest.approx(0.2)
    cube = np.array([[x, y, z] for x in (0, 20) for y in (0, 20) for z in (0, 20)], float)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    out = tmp_path / "cube.gcode"
    r = slicer.slice_meshes(st, [(cube, tris, 0)], str(out), engine=ENGINE)
    text = out.read_text(encoding="utf-8")
    head = text[:text.index(";LAYER:0")]
    assert start in head
    assert "{" not in head and "[nozzle" not in head and "[bed" not in head
    assert head.count("M109 S200") == 1                       # heated once, by the printer's own
    assert 95 <= r.layers <= 101 and r.seconds > 0
