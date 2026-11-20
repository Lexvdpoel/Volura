"""Slicing with CuraEngine: File > Slice….

`slicer.slice_meshes` writes the parts as STL files and the settings of a `cura_profiles.Stacks`
as CuraEngine's resolved-settings file, runs CuraEngine and reads the print time and the filament
back from the G-code. The parts are moved to the middle of the bed, because CuraEngine counts from
there. These tests check the pieces on their own - the STL, the placement, reading the header in
both G-code flavours - and, when CuraEngine and the Cura profiles are there
(tools/fetch_curaengine.py), slice a real part for a real printer and check the G-code and the
picture of the print it carries for the screen of the printer, and drive the dialog in the window.

Run with:  python -m pytest tests/test_slicer.py
"""
import base64
import os
import re
import struct
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import cura_profiles, slicer  # noqa: E402

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")


def _cube(size=20.0, at=(0.0, 0.0, 0.0)):
    m = mf.Manifold.cube((size, size, size)).translate(at).to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def _pictures(text):
    """The pictures in the comments of a G-code, as the screen of a printer reads them: a list of
    (width, height, the PNG, the length the file itself says that base64 has)."""
    got, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        word = line.split()
        if word[:3] != [";", "thumbnail", "begin"]:
            continue
        body = ""
        for follow in lines[i + 1:]:
            if follow.startswith("; thumbnail end"):
                break
            body += follow[2:]
        wide, high = word[3].split("x")
        got.append((int(wide), int(high), base64.b64decode(body), int(word[4])))
    return got


def test_the_stl_is_a_binary_stl_of_the_mesh(tmp_path):
    v, t = _cube()
    path = tmp_path / "c.stl"
    slicer.write_stl(str(path), v, t)
    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    assert count == len(t) and len(data) == 84 + 50 * len(t)
    first = struct.unpack("<12f", data[84:132])
    assert np.allclose(first[3:6], v[t[0, 0]]), "the corners of the first triangle"


def test_the_parts_go_to_the_middle_of_the_bed_and_onto_it():
    a, ta = _cube(10.0, (100.0, 50.0, 5.0))
    b, tb = _cube(10.0, (130.0, 50.0, 5.0))
    placed = slicer.place_on_bed([(a, ta, 0), (b, tb, 0)], {})
    lo = np.min([p[0].min(axis=0) for p in placed], axis=0)
    hi = np.max([p[0].max(axis=0) for p in placed], axis=0)
    assert np.allclose(lo + hi, [0.0, 0.0, hi[2] + lo[2]]) and lo[2] == pytest.approx(0.0)
    assert placed[1][0][:, 0].min() - placed[0][0][:, 0].min() == pytest.approx(30.0), \
        "the parts keep their places relative to each other"


def test_the_header_of_a_marlin_g_code():
    head = ";FLAVOR:Marlin\n;TIME:1678\n;Filament used: 0.699145m, 0.1m\n;Layer height: 0.1\n" \
           ";LAYER_COUNT:98\n"
    info = slicer.parse_header(head)
    assert info["seconds"] == 1678 and info["layers"] == 98
    assert info["filament_m"] == pytest.approx([0.699145, 0.1])


def test_the_header_of_an_ultimaker_g_code():
    head = ";START_OF_HEADER\n;FLAVOR:Griffin\n;EXTRUDER_TRAIN.0.MATERIAL.VOLUME_USED:2130\n" \
           ";PRINT.TIME:611\n;END_OF_HEADER\n"
    info = slicer.parse_header(head)
    assert info["seconds"] == 611 and info["volume_mm3"] == [2130.0]


def test_without_an_engine_slicing_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(slicer, "find_engine", lambda: None)
    with pytest.raises(RuntimeError):
        slicer.slice_meshes(None, [_cube() + (0,)], str(tmp_path / "x.gcode"))


# ---------------------------------------------------------------- the colors of the print

MATERIALS = [["Red PLA", "#d23b3b"], ["Blue PLA", "#2f6fd6"], ["Green PLA", "#3aa655"]]


def _tagged(color, painted=None, size=10.0):
    """A cube in `color`, the faces at the far end of the z axis painted `painted`, with the
    face tags a printed part carries (see `volura.ops`)."""
    from volura import brep, ops
    man = mf.Manifold.cube((size, size, size))
    mesh = man.to_mesh()
    corner = np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts).reshape(-1, 3)]
    top = np.all(corner[:, :, 2] > size - 1e-6, axis=1) if painted else np.zeros(len(corner), bool)
    colors = [painted if hit else color for hit in top]
    tags = np.array([ops.color_id(c) for c in colors], dtype=np.uint64)
    return brep.manifold_from_tris(np.asarray(mesh.vert_properties, dtype=np.float64),
                                   np.asarray(mesh.tri_verts, dtype=np.uint64), tags)


def _colored_parts():
    """One part per color of MATERIALS."""
    from volura import ops
    return [ops.Part(name, color, _tagged(color)) for name, color in MATERIALS]


def _model_colors():
    from volura import fileio
    return fileio.model_colors(_colored_parts(), MATERIALS)


def test_the_slice_takes_the_filaments_of_the_multicolor_3mf():
    """What the 3MF numbers the colors with is what the print extrudes with."""
    from volura.slice_dialog import Multicolor
    colors = _model_colors()
    assert [color for _n, color, _e in colors] == [c for _n, c in MATERIALS]
    assert [e for _n, _c, e in colors] == [1, 2, 3], "the filament order of the Materials panel"
    plan = Multicolor(3, colors=colors)
    assert [plan.extruder(color) for _n, color, _e in colors] == [0, 1, 2]
    assert plan.filaments() == [(color, name) for name, color in MATERIALS]


def test_a_painted_face_joins_the_filaments_of_the_model():
    """A face painted in another color takes its own filament, and a color the model does not
    use does not burn one."""
    from volura import fileio, ops
    from volura.slice_dialog import Multicolor
    red = ops.Part("Red", "#d23b3b", _tagged("#d23b3b", painted="#3aa655"))
    colors = fileio.model_colors([red], MATERIALS)
    assert [(name, color, e) for name, color, e in colors] == \
        [("Red PLA", "#d23b3b", 1), ("Green PLA", "#3aa655", 2)], "blue is not in the model"
    plan = Multicolor(2, colors=colors)
    assert plan.extruder("#d23b3b") == 0 and plan.extruder("#3aa655") == 1
    assert plan.filaments() == [("#d23b3b", "Red PLA"), ("#3aa655", "Green PLA")]


def test_the_color_table_of_the_dialog_reroutes_a_color():
    """Two colors on one extruder: that extruder shows the first of them."""
    from volura.slice_dialog import Multicolor
    plan = Multicolor(2, mapping={"#d23b3b": 0, "#2f6fd6": 0, "#3aa655": 1}, colors=_model_colors())
    assert [plan.extruder(c) for c in ("#d23b3b", "#2f6fd6", "#3aa655")] == [0, 0, 1]
    assert plan.filaments() == [("#d23b3b", "Red PLA"), ("#3aa655", "Green PLA")]


def test_colors_that_do_not_fit_the_extruders_wrap_around():
    from volura.slice_dialog import Multicolor
    plan = Multicolor(2, colors=_model_colors())        # three colors, two extruders
    assert [plan.extruder(c) for c in ("#d23b3b", "#2f6fd6", "#3aa655")] == [0, 1, 0]


def test_a_print_of_one_extruder_needs_no_tower_and_two_do():
    from volura.slice_dialog import Multicolor
    assert Multicolor(1).values(235.0, 235.0) == {}
    assert Multicolor(2, tower=False).values(235.0, 235.0) == \
        {"prime_tower_enable": False, "ooze_shield_enabled": False}
    both = Multicolor(2, tower=True, ooze=True).values(235.0, 235.0)
    assert both["ooze_shield_enabled"] is True
    assert both["prime_tower_size"] == pytest.approx(35.2), "15% of the shortest side of the bed"
    assert Multicolor(2).values(200.0, 100.0)["prime_tower_size"] == pytest.approx(15.0)
    assert Multicolor(2).values()["prime_tower_size"] == pytest.approx(15.0), "no bed given"


def test_the_heights_of_the_changes_come_and_go_as_one_settings_line():
    from volura.slice_dialog import heights_text, parse_heights
    assert parse_heights("5,12.5, 5 ; 3") == [3.0, 5.0, 12.5]
    assert parse_heights("") == [] and parse_heights(None) == []
    assert heights_text([12.5, 5.0, 5.0]) == "5,12.5"
    assert parse_heights(heights_text([0.2, 7.35])) == [0.2, 7.35]


LAYERS = (";LAYER:0\nG1 Z0.2 F900\nG1 X10 E1\n"
          ";LAYER:1\nG1 Z0.4\nG1 X10 E2\n"
          ";LAYER:2\nG1 Z0.6\nG1 X10 E3\n")


def test_the_layers_of_the_g_code_and_their_heights():
    assert [round(z, 2) for _line, z in slicer.layer_heights(LAYERS)] == [0.2, 0.4, 0.6]


def test_a_filament_change_lands_on_the_first_layer_at_or_above_its_height():
    lines = slicer.with_filament_changes(LAYERS, [0.21, 0.6]).splitlines()
    assert lines == [";LAYER:0", "G1 Z0.2 F900", "G1 X10 E1",
                     ";LAYER:1", "M600", "G1 Z0.4", "G1 X10 E2",
                     ";LAYER:2", "M600", "G1 Z0.6", "G1 X10 E3"], \
        "the change comes where the print is past the height"


def test_two_changes_in_one_layer_ask_for_the_filament_once():
    assert slicer.with_filament_changes(LAYERS, [0.5, 0.55]).count("M600") == 1


def test_a_change_above_the_print_or_below_the_first_layer_asks_for_nothing():
    assert slicer.with_filament_changes(LAYERS, [7.0]) == LAYERS
    assert slicer.with_filament_changes(LAYERS, []) == LAYERS


def test_the_command_of_the_change_is_yours_and_keeps_the_line_endings():
    got = slicer.with_filament_changes(LAYERS, [0.5], "M600 P3")
    assert "\nM600 P3\n" in got
    crlf = slicer.with_filament_changes(LAYERS.replace("\n", "\r\n"), [0.5])
    assert "M600\r\n" in crlf and crlf.count("M600") == 1


def test_the_changes_are_written_into_the_g_code(tmp_path):
    path = tmp_path / "x.gcode"
    path.write_text(LAYERS, encoding="utf-8")
    slicer.insert_filament_changes(str(path), [0.21])
    assert path.read_text(encoding="utf-8").splitlines()[4] == "M600"
    before = path.read_bytes()
    slicer.insert_filament_changes(str(path), [99.0])
    assert path.read_bytes() == before, "a file that needs nothing is left alone"


@needs_engine
def test_a_cube_slices_for_the_ender_3(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    st = p.stacks(v, m, "standard", {"infill_sparse_density": 20.0})
    out = tmp_path / "cube.gcode"
    seen = []
    r = slicer.slice_meshes(st, [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)], str(out),
                            on_progress=seen.append)
    text = out.read_text(encoding="utf-8")
    assert r.layers == 100, "20 mm at 0.2 mm layers"
    assert 300 < r.seconds < 3600 and 1.0 < r.grams < 6.0
    assert ";FLAVOR:Marlin" in text and "M104 S200" in text, "the PLA temperature"
    assert not re.search(r"\{[a-z_]+\}", text), "a start g-code token was not filled in"
    lo = float(re.search(r"^;MINX:([\d.]+)", text, re.M).group(1))
    hi = float(re.search(r"^;MAXX:([\d.]+)", text, re.M).group(1))
    assert 0.5 * (lo + hi) == pytest.approx(117.5, abs=1.0), "in the middle of the 235 mm bed"


@needs_engine
def test_two_colours_on_the_one_nozzle_of_a_changer(tmp_path):
    """A filament changer in front of the one nozzle of an Ender 3 asks for both filaments."""
    from volura.slice_dialog import Multicolor
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v, m = p.default_variant(), p.default_material()
    st = p.stacks(v, m, "standard", {"infill_sparse_density": 0.0})
    cura_profiles.filament_changer(st, 2)
    colors = [("Red PLA", "#d23b3b", 1), ("Blue PLA", "#2f6fd6", 2)]
    st.glob.layers[0].update(Multicolor(2, colors=colors).values(235.0, 235.0))
    out = tmp_path / "two-colour.gcode"
    r = slicer.slice_meshes(st, [_cube(10.0, (30.0, 30.0, 0.0)) + (0,),
                                 _cube(10.0, (50.0, 30.0, 0.0)) + (1,)], str(out))
    text = out.read_text(encoding="utf-8")
    assert re.search(r"^T0\b", text, re.M) and re.search(r"^T1\b", text, re.M), \
        "the print asks the changer for both filaments"
    assert len(r.volume_mm3) == 2 and all(v > 0 for v in r.volume_mm3), \
        "both filaments are in the header of the G-code"


@needs_engine
def test_a_colour_change_by_height_lands_in_the_g_code(tmp_path):
    """A printer with one filament and no changer is asked to change halfway."""
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v, m = p.default_variant(), p.default_material()
    st = p.stacks(v, m, "standard", {"infill_sparse_density": 0.0})
    out = tmp_path / "change.gcode"
    slicer.slice_meshes(st, [_cube(10.0, (30.0, 30.0, 0.0)) + (0,)], str(out))
    text = out.read_text(encoding="utf-8")
    assert slicer.FILAMENT_CHANGE not in text, "a plain print does not change filament"
    slicer.insert_filament_changes(str(out), [5.0])
    changed = out.read_text(encoding="utf-8").splitlines()
    at = [k for k, line in enumerate(changed) if line == slicer.FILAMENT_CHANGE]
    assert len(at) == 1, "one change, at a layer boundary"
    z = [zz for _line, zz in slicer.layer_heights("\n".join(changed[at[0] - 1:]))]
    assert z and z[0] == pytest.approx(5.0, abs=0.21), "the first layer at or above 5 mm"
    assert changed[at[0] - 1].startswith(";LAYER:"), "the change comes where a layer starts"
    heights = _tools_by_layer("\n".join(changed[:at[0]]))
    before = [line for line in changed[:at[0]] if not line.startswith(";LAYER:")]
    after = _tools_by_layer("\n".join(before + [";LAYER:x"] + changed[at[0]:]))
    assert max(heights) < 5.0 <= min(after), \
        f"printed up to {max(heights)} before the change and from {min(after)} after it"


def _two_extruders(plan):
    """The stacks of a printer with two extruders of its own (Creality CR-X), set up for `plan`."""
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_cr-x")
    assert p.extruder_count == 2
    st = p.stacks(p.default_variant(), p.default_material(), "standard",
                  {"infill_sparse_density": 0.0})
    st.glob.layers[0].update(plan.values(200.0, 200.0))
    return st


def _tools_by_layer(text):
    """{height: the extruders that lay lines at that height} of the layers of a G-code (the
    start G-code, before the first layer, primes the nozzles and is left out)."""
    out, z, tool, started = {}, None, 0, False
    for line in text.splitlines():
        code = line.split(";", 1)[0].strip()
        if line.startswith(";LAYER:"):
            started = True
        elif re.match(r"^T\d+$", code):
            tool = int(code[1:])
        elif code.startswith(("G0", "G1")):
            m = re.search(r"\bZ(-?[\d.]+)", code)
            if m:
                z = float(m.group(1))
            if started and re.search(r"\bE[\d.]", code) and re.search(r"\b[XY]", code):
                out.setdefault(round(z, 3), set()).add(tool)
    return out


@needs_engine
def test_two_colours_on_two_extruders_print_with_both_and_a_prime_tower(tmp_path):
    from volura.slice_dialog import Multicolor
    colors = [("Red PLA", "#d23b3b", 1), ("Blue PLA", "#2f6fd6", 2)]
    plan = Multicolor(2, colors=colors)
    out = tmp_path / "two.gcode"
    slicer.slice_meshes(_two_extruders(plan),
                        [_cube(10.0, (60.0, 60.0, 0.0)) + (plan.extruder("#d23b3b"),),
                         _cube(10.0, (90.0, 60.0, 0.0)) + (plan.extruder("#2f6fd6"),)], str(out))
    text = out.read_text(encoding="utf-8")
    assert re.search(r"^T0\b", text, re.M) and re.search(r"^T1\b", text, re.M)
    assert ";TYPE:PRIME-TOWER" in text, "a prime tower to wipe the nozzle on"


@needs_engine
def test_a_colour_routed_to_the_first_extruder_never_calls_the_second(tmp_path):
    from volura.slice_dialog import Multicolor
    colors = [("Red PLA", "#d23b3b", 1), ("Blue PLA", "#2f6fd6", 2)]
    plan = Multicolor(2, mapping={"#d23b3b": 0, "#2f6fd6": 0}, tower=False, colors=colors)
    out = tmp_path / "one.gcode"
    slicer.slice_meshes(_two_extruders(plan),
                        [_cube(10.0, (60.0, 60.0, 0.0)) + (plan.extruder("#d23b3b"),),
                         _cube(10.0, (90.0, 60.0, 0.0)) + (plan.extruder("#2f6fd6"),)], str(out))
    tools = set().union(*_tools_by_layer(out.read_text(encoding="utf-8")).values())
    assert tools == {0}, "both colours print with the first extruder"


@needs_engine
def test_a_painted_top_prints_with_the_second_extruder_only_at_the_top(tmp_path):
    """A 10 mm cube with its top painted green: the green skin of 1 mm is the only thing the
    second extruder prints, so it only works in the top millimetre of layers."""
    from volura import fileio, ops, paint_split
    from volura.slice_dialog import Multicolor
    man = _tagged("#d23b3b", painted="#3aa655").translate((60.0, 60.0, 0.0))
    part = ops.Part("Red", "#d23b3b", man)
    plan = Multicolor(2, tower=False, colors=fileio.model_colors([part], MATERIALS))
    pieces = paint_split.split_painted(man, fileio.triangle_colors(man), part.color, 1.0)
    meshes = []
    for piece, color in pieces:
        m = piece.to_mesh()
        meshes.append((np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts),
                       plan.extruder(color)))
    out = tmp_path / "painted.gcode"
    slicer.slice_meshes(_two_extruders(plan), meshes, str(out))
    by_layer = _tools_by_layer(out.read_text(encoding="utf-8"))
    first = min(by_layer)                    # every extruder primes with a skirt there
    second = [z for z, tools in by_layer.items() if 1 in tools and z > first]
    assert second, "the painted top prints with the second extruder"
    assert min(second) > 10.0 - 1.0 - 0.3, f"the second extruder starts too low: {min(second)}"
    assert any(0 in tools for z, tools in by_layer.items() if z < 5.0), "the rest is red"


def _fill(text):
    """The infill moves of a G-code: every G1 under ;TYPE:FILL."""
    moves, on = [], False
    for line in text.splitlines():
        if line.startswith(";TYPE:"):
            on = line == ";TYPE:FILL"
        elif on and line.startswith("G1"):
            moves.append(line)
    return moves


@needs_engine
def test_every_infill_pattern_of_the_slice_dialog_slices_its_own_way(tmp_path):
    from volura.slice_dialog import INFILL_PATTERNS
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    seen = {}
    for _label, pattern in INFILL_PATTERNS[1:]:
        st = p.stacks(v, m, "standard", {"infill_sparse_density": 20.0,
                                         "infill_pattern": pattern})
        out = tmp_path / f"{pattern}.gcode"
        slicer.slice_meshes(st, [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)], str(out))
        seen[pattern] = "\n".join(_fill(out.read_text(encoding="utf-8")))
    assert all(seen[k] for k in seen if k != "lightning"), "a pattern without infill"
    same = [(a, b) for a in seen for b in seen if a < b and seen[a] == seen[b]]
    assert not same, f"these patterns slice the same: {same}"


def _moves_of_type(text, type_name):
    """The G1 extrusion moves of one kind of line in the G-code, as the `;TYPE:` marks it."""
    moves, on = 0, False
    for line in text.splitlines():
        if line.startswith(";TYPE:"):
            on = line == f";TYPE:{type_name}"
        elif on and line.startswith("G1") and " E" in line:
            moves += 1
    return moves


def _top_layer_moves(text):
    """The G1 extrusion moves of the top layer of a G-code (Cura writes ironing with its top
    skin, without a `;TYPE:` of its own, so the count there is what shows the extra pass)."""
    top = text.rsplit(";LAYER:", 1)[-1]
    return sum(1 for line in top.splitlines() if line.startswith("G1") and " E" in line)


def _ender3():
    """The Ender 3 with its default nozzle and filament, as the slice dialog would set it up."""
    p = cura_profiles.Library(RESOURCES).printer("creality_ender3")
    v = p.default_variant()
    return p, v, p.default_material(v)


@needs_engine
def test_ironing_lays_an_extra_pass_over_the_flat_top(tmp_path):
    """A 20 mm cube with a flat top: ironing goes over that top once more, so the top layer
    holds more moves than without it."""
    p, v, m = _ender3()
    meshes = [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)]
    plain = tmp_path / "plain.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 0.0}),
                        meshes, str(plain))
    without = _top_layer_moves(plain.read_text(encoding="utf-8"))
    ironed = tmp_path / "ironed.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 0.0,
                                                    "ironing_enabled": True,
                                                    "ironing_only_highest_layer": True}),
                        meshes, str(ironed))
    with_ironing = _top_layer_moves(ironed.read_text(encoding="utf-8"))
    assert without > 0 and with_ironing > without, \
        f"ironed top layer {with_ironing} moves, plain top {without}"


@needs_engine
def test_fuzzy_skin_roughens_the_outer_wall(tmp_path):
    """The bumpy wall is cut into far more little lines than the straight one."""
    p, v, m = _ender3()
    meshes = [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)]
    plain = tmp_path / "smooth.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 0.0}),
                        meshes, str(plain))
    smooth = _moves_of_type(plain.read_text(encoding="utf-8"), "WALL-OUTER")
    fuzzy = tmp_path / "fuzzy.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 0.0,
                                                    "magic_fuzzy_skin_enabled": True,
                                                    "magic_fuzzy_skin_thickness": 0.3}),
                        meshes, str(fuzzy))
    rough = _moves_of_type(fuzzy.read_text(encoding="utf-8"), "WALL-OUTER")
    assert 0 < smooth < rough, f"outer wall: {smooth} smooth moves, {rough} fuzzy"


def _sphere(size=20.0, at=(40.0, 40.0)):
    m = mf.Manifold.sphere(size / 2, 48).translate((*at, size / 2)).to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def _cylinder(radius=15.0, height=30.0, at=(40.0, 40.0)):
    m = mf.Manifold.cylinder(height, radius, radius, 64).translate((*at, height / 2)).to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def _zs_within_layers(text):
    """The heights the nozzle extrudes at inside every layer of a G-code: an ordinary print keeps
    one height to a layer, a spiral rises through the layer it is in."""
    out = []
    for part in re.split(r"^;LAYER:", text, flags=re.M)[1:]:
        heights = set()
        for line in part.splitlines():
            code = line.split(";", 1)[0].strip()
            if code.startswith(("G1", "G0")) and "Z" in code and "E" in code:
                heights.add(round(float(re.search(r"\bZ(-?\d*\.?\d+)", code).group(1)), 3))
        out.append(sorted(heights))
    return out


@needs_engine
def test_adaptive_layer_height_gives_a_sphere_layers_of_more_than_one_height(tmp_path):
    """Where the sphere curves the layers come thinner than 0.2 mm, elsewhere thicker, so the
    print runs faster without losing the smoothness of the curves."""
    p, v, m = _ender3()
    meshes = [_sphere() + (0,)]

    def steps(user):
        """The thickness of the layers, as the difference between one layer height and the next."""
        out = tmp_path / "sphere.gcode"
        slicer.slice_meshes(p.stacks(v, m, "standard", user), meshes, str(out))
        zs = sorted({round(z, 3) for _line, z in slicer.layer_heights(
            out.read_text(encoding="utf-8"))})
        return {round(b - a, 3) for a, b in zip(zs, zs[1:])}

    plain = steps({})
    adaptive = steps({"adaptive_layer_height_enabled": True})
    assert plain == {0.2}, "an ordinary print stacks every layer of this quality on 0.2 mm"
    assert len(adaptive) > 1, f"adaptive layers all take the same step: {sorted(adaptive)}"
    assert max(adaptive) > 0.2, "the steep lower walls are not printed with thicker layers"


@needs_engine
def test_four_walls_print_more_inner_wall_lines_than_two(tmp_path):
    p, v, m = _ender3()
    meshes = [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)]
    inner = {}
    for walls in (2, 4):
        out = tmp_path / f"walls{walls}.gcode"
        slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 0.0,
                                                        "wall_line_count": walls}),
                            meshes, str(out))
        inner[walls] = _moves_of_type(out.read_text(encoding="utf-8"), "WALL-INNER")
    assert inner[2] > 0 and inner[4] > inner[2], \
        f"2 walls: {inner[2]} inner moves, 4 walls: {inner[4]}"


@needs_engine
def test_vase_mode_spirals_one_wall_up_with_a_bottom_under_it(tmp_path):
    """A cylinder in vase mode: the wall rises in one line through every layer and nothing is
    filled, where the same cylinder without it fills its inside and keeps every layer flat."""
    p, v, m = _ender3()
    meshes = [_cylinder() + (0,)]
    plain = tmp_path / "plain.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 20.0}),
                        meshes, str(plain))
    text = plain.read_text(encoding="utf-8")
    assert _moves_of_type(text, "FILL") > 0 and not [zs for zs in _zs_within_layers(text)
                                                     if len(zs) > 1], \
        "an ordinary print fills the inside and keeps every layer flat"
    vase = tmp_path / "vase.gcode"
    slicer.slice_meshes(p.stacks(v, m, "standard", {"infill_sparse_density": 20.0,
                                                    "magic_spiralize": True}),
                        meshes, str(vase))
    spiral = vase.read_text(encoding="utf-8")
    layers = _zs_within_layers(spiral)
    rising = [zs for zs in layers if len(zs) > 1]
    assert len(layers) > 20 and ";TYPE:FILL" not in spiral, "a vase has no infill"
    assert _moves_of_type(spiral, "WALL-OUTER") > 0
    assert len(rising) > len(layers) // 2, \
        f"only {len(rising)} of {len(layers)} layers rise as a spiral does"
    assert rising[-1][-1] > rising[0][0], "and the wall climbs the whole way up"


@needs_engine
def test_every_seam_of_the_slice_dialog_reaches_the_engine(tmp_path):
    """File > Slice… > Extras > Seam: the names and the values are Cura's own, so CuraEngine
    takes each of them and still prints a wall after wall."""
    import volura.slice_dialog as slice_dialog
    p, v, m = _ender3()
    meshes = [_cube(20.0, (40.0, 40.0, 0.0)) + (0,)]
    for _label, kind, extra in slice_dialog.SEAMS[1:]:
        st = p.stacks(v, m, "standard", dict(extra, z_seam_type=kind))
        assert st.global_settings()["z_seam_type"] == kind, "CuraEngine reads it from there"
        out = tmp_path / f"seam_{kind}.gcode"
        slicer.slice_meshes(st, meshes, str(out))
        text = out.read_text(encoding="utf-8")
        assert ";LAYER:0" in text and _moves_of_type(text, "WALL-OUTER") > 0, f"{kind} slices"


@needs_engine
def test_slice_from_the_file_menu(monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    import volura.slice_dialog as sd
    import volura.window_print as wp
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    v, t = _cube(15.0)
    from volura import ops
    ops.group_from_manifold(w.doc, mf.Manifold.cube((15.0, 15.0, 15.0)), "#cccccc", "cube")
    out = tmp_path / "menu.gcode"
    seen = {}

    def fill(dlg):
        dlg.maker.setCurrentText("Creality3D")
        i = dlg.printer_box.findData("creality_ender3")
        assert i >= 0
        dlg.printer_box.setCurrentIndex(i)
        assert dlg.nozzle.currentText() == "0.4mm Nozzle"
        assert dlg.material.currentData() == "generic_pla_175"
        assert dlg.brand.currentText() == "Generic" and dlg.brand.count() > 3
        dlg.brand.setCurrentText(next(dlg.brand.itemText(i) for i in range(dlg.brand.count())
                                      if dlg.brand.itemText(i) != "Generic"))
        assert dlg.material.count() > 0 and dlg.material.currentData() != "generic_pla_175"
        dlg.brand.setCurrentText("Generic")
        i = dlg.material.findData("generic_pla_175")
        dlg.material.setCurrentIndex(i)
        assert dlg.material.currentText() == "PLA"
        assert dlg.quality.currentData() == "standard"
        assert "235" in dlg.info.text(), "the build volume of the printer"
        dlg.infill.setValue(15)
        dlg.support.setCurrentIndex(1)
        dlg.pattern.setCurrentIndex([k for _l, k in sd.INFILL_PATTERNS].index("gyroid"))
        assert dlg.user_values()["infill_pattern"] == "gyroid"
        dlg.thumbnail.setCurrentIndex(dlg.thumbnail.findData("32x32/300x300"))
        assert dlg.thumbnail.currentData() == "32x32/300x300"
        seen["opened"] = True
        return QDialog.Accepted

    monkeypatch.setattr(sd.SliceDialog, "exec", fill)
    errors = []
    monkeypatch.setattr(wp.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(wp.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    import volura.print_preview as pp

    def look(dlg):
        """The preview: layers to look at, a nozzle that plays, then Save G-code…"""
        seen["layers"] = len(dlg.preview.layers)
        assert " g filament (" in dlg.summary.text() and "≈ " in dlg.summary.text(), \
            "the summary does not say what the filament costs"
        dlg.slider.setValue(5)
        dlg.play.setChecked(True)
        dlg._tick()
        assert dlg.view.at is not None and dlg.view.layer == 5
        dlg.play.setChecked(False)
        dlg.grab()                                    # it draws
        dlg._save()
        return QDialog.Accepted

    monkeypatch.setattr(pp.PrintPreviewDialog, "exec", look)
    try:
        assert w.a_slice in w.flat_actions(w.menuBar())
        w.slice_dialog()
        deadline = time.monotonic() + 120.0
        while not out.exists() and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        while "Sliced" not in w.hint.text() and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert seen.get("opened") and out.exists()
        assert "Sliced" in w.hint.text() and "g filament" in w.hint.text()
        assert seen["layers"] > 10 and str(out) in w.hint.text()
        assert w.settings.slice_printer == "creality_ender3"
        assert w.settings.slice_infill == 15 and w.settings.slice_support == 1
        assert w.settings.slice_infill_pattern == "gyroid"
        assert w.settings.slice_thumbnail == "32x32/300x300", "the next slice draws what you chose"
        text = out.read_text(encoding="utf-8")
        assert ";FLAVOR:Marlin" in text
        shots = _pictures(text)
        assert [(w, h) for w, h, _png, _n in shots] == [(32, 32), (300, 300)], \
            "the two sizes of the choice, the little one first"
        for wide, high, png, number in shots:
            assert png[:8] == b"\x89PNG\r\n\x1a\n", "the picture in the G-code is no PNG"
            assert len(base64.b64encode(png)) == number, "the block says another length"
            img = QImage.fromData(png)
            assert (img.width(), img.height()) == (wide, high), "another picture than its size says"
    finally:
        w.doc.dirty = False
        w.close()
