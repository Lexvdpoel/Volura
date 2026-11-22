"""Flow Rate Test: a row of pads, one per material flow, so the flow that lays a smooth, closed
top surface can be read off the print.

`flow_rate_test.flow_rate_test` gives the pads as separate closed volumes standing on `z = 0`:
`pads` square pads of `PAD` by `PAD` millimetres, `HEIGHT` tall, `GAP` apart, in a row or in two
rows, every one printed with its own flow from `start` to `end` by `step` percent, with that flow
engraved in its top surface. `flow_rate_test.flows` gives the flow of every pad, and the model hands
each pad its own `material_flow` through the part's `mesh_settings`, which the slice carries to
CuraEngine as a setting of that one mesh: `slicer.slice_meshes` loads a mesh that wants settings of
its own with `-l` and puts those settings right behind it, where they beat the values of the whole
print for that mesh alone, while the parts that ask for nothing load from the settings document as
they always did.

These tests (Qt offscreen) weigh the geometry: the pads are valid and separate, the layout folds
into two rows when a single row would be too wide, the engraving takes plastic out of the top, and a
flow reads 97.5 rather than 97.50000000000001; the validation (a step of 0, too few or too many
pads, a number that is no number); the `mesh_settings` of a pad round-tripping through a file; the
general per-mesh mechanism (only keys the definition marks `settable_per_mesh` reach the engine, a
global-only key is dropped with a warning, and what a value works out to for the settings
CuraEngine only reads the result of comes along); the window (a row of named, self-configuring
groups as one undo step); and, when CuraEngine is there, that the per-pad flow reaches the engine as
that pad's own setting, that extrusion really scales with the flow and that the flow of one pad
leaves the pad next to it alone.

Run with:  python -m pytest tests/test_flow_rate_test.py
"""
import json
import logging
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

from volura import (app as app_module, cura_profiles, fileio, flow_rate_test, ops, slicer,  # noqa: E402
                    window_print)
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: the dialog's default: nine pads from 90 to 110 in steps of 2.5
PLA = dict(start=90.0, end=110.0, step=2.5)


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    for key, value in (("flow_rate_test_start", 90.0), ("flow_rate_test_end", 110.0),
                       ("flow_rate_test_step", 2.5)):
        setattr(w.settings, key, value)
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def made(**kw) -> list:
    return flow_rate_test.flow_rate_test(**dict(PLA, **kw))


def pump_until(predicate, timeout_ms=30000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the pads

@pytest.mark.parametrize("labels", (False, True))
def test_the_test_is_a_set_of_valid_separate_pads(labels):
    s = flow_rate_test.sizes(**PLA)
    pads = made(labels=labels)
    assert len(pads) == s["pads"] == 9
    for m in pads:
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert len(m.decompose()) == 1, "a pad fell apart"


def test_the_pads_are_three_millimetres_tall_and_stand_on_the_bed():
    for m in made():
        b = np.asarray(m.bounding_box(), dtype=float)
        assert b[2] == pytest.approx(0.0, abs=1e-6), "a pad does not stand on the bed"
        assert b[5] - b[2] == pytest.approx(flow_rate_test.HEIGHT, abs=1e-3)


def test_a_single_row_is_kept_until_it_grows_wider_than_the_bed():
    # five pads of 30 with 3 mm between them fit one row; six would be wider than MAX_ROW
    assert flow_rate_test.sizes(90.0, 110.0, 2.5)["rows"] == 2       # nine pads
    assert flow_rate_test.sizes(90.0, 100.0, 2.5)["rows"] == 1       # five pads
    assert flow_rate_test.sizes(90.0, 112.5, 2.5)["rows"] == 2       # ten pads
    assert flow_rate_test.sizes(90.0, 112.5, 2.5)["size"][0] <= flow_rate_test.MAX_ROW + 1e-6


def test_the_pads_lie_in_the_order_of_their_flows():
    s = flow_rate_test.sizes(**PLA)
    pads = made(labels=False)
    # left to right, top to bottom: the flows keep rising
    order = sorted(range(len(pads)), key=lambda i: (
        np.asarray(pads[i].bounding_box())[1], np.asarray(pads[i].bounding_box())[0]))
    seen = [s["flows"][i] for i in order]
    assert seen == sorted(seen), "the pads do not walk up in flow from left to right"


def test_two_pads_never_touch():
    pads = made(labels=False)
    boxes = [np.asarray(m.bounding_box(), dtype=float) for m in pads]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            apart = (a[3] + 1e-6 <= b[0]) or (b[3] + 1e-6 <= a[0]) or \
                    (a[4] + 1e-6 <= b[1]) or (b[4] + 1e-6 <= a[1])
            assert apart, "pads {} and {} overlap".format(i, j)


def test_the_engraved_flow_takes_plastic_out_of_the_top():
    """The number is cut into the top, so a pad with a label holds less plastic than one without."""
    with_l = sum(m.volume() for m in made(labels=True))
    without = sum(m.volume() for m in made(labels=False))
    assert with_l < without, "the flow is not engraved in the top"


def test_the_engraving_is_deep_but_does_not_cut_through():
    pads = made(labels=True)
    for m in pads:
        assert m.volume() < flow_rate_test.PAD ** 2 * flow_rate_test.HEIGHT + 1e-6
        # the bottom is still solid where the number was cut: a thin probe there is all plastic
        b = np.asarray(m.bounding_box(), dtype=float)
        cx, cy = 0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4])
        probe = mf.Manifold.cube([1.0, 1.0, 1.0]).translate((cx - 0.5, cy - 0.5, 0.0))
        assert (m ^ probe).volume() == pytest.approx(1.0, abs=1e-3), "the number cut through"


# ------------------------------------------------------------------ the flows

def test_the_default_walk_is_nine_pads_and_reads_without_float_noise():
    fl = flow_rate_test.flows(**PLA)
    assert fl == [90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0]
    assert [flow_rate_test.flow_text(v) for v in fl] == \
        ["90", "92.5", "95", "97.5", "100", "102.5", "105", "107.5", "110"]


def test_a_walk_down_gives_the_flows_highest_first():
    fl = flow_rate_test.flows(110.0, 90.0, 5.0)
    assert fl == [110.0, 105.0, 100.0, 95.0, 90.0]


@pytest.mark.parametrize("value, text", ((97.5, "97.5"), (100.0, "100"), (90.0, "90"),
                                         (97.50000000000001, "97.5"), (-0.0, "0")))
def test_flow_text_formats_the_way_a_group_and_an_engraving_write_it(value, text):
    assert flow_rate_test.flow_text(value) == text


# ------------------------------------------------------------------ the validation

@pytest.mark.parametrize("bad", (0.0, -2.5))
def test_a_step_of_zero_or_less_is_refused(bad):
    with pytest.raises(ValueError):
        flow_rate_test.flow_rate_test(90.0, 110.0, bad)


def test_a_walk_that_gives_one_pad_is_refused():
    with pytest.raises(ValueError):
        flow_rate_test.flow_rate_test(90.0, 90.0, 2.5)      # one pad
    with pytest.raises(ValueError):
        flow_rate_test.flow_rate_test(90.0, 91.0, 2.5)      # still one pad


def test_too_many_pads_is_refused():
    with pytest.raises(ValueError):
        flow_rate_test.flow_rate_test(50.0, 150.0, 5.0)     # 21 pads


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "rich"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("start", "end", "step"):
        with pytest.raises(ValueError):
            flow_rate_test.flow_rate_test(**dict(PLA, **{key: bad}))
        with pytest.raises(ValueError):
            flow_rate_test.flows(**dict(PLA, **{key: bad}))


def test_the_flows_clamp_to_what_a_printer_prints_at():
    assert flow_rate_test.flows(10.0, 400.0, 25.0) == [50.0, 75.0, 100.0, 125.0, 150.0]


# ------------------------------------------------------------------ the round-trip

def test_a_pad_carries_its_flow_on_itself_through_a_file(tmp_path):
    doc = Document()
    pad = made(labels=False)[0]
    g = ops.group_from_manifold(doc, pad, "#f2f2f2", "Flow 90 %")
    g.meta = {"mesh_settings": {"material_flow": 90.0}}
    path = str(tmp_path / "pads.volura")
    fileio.save_document(doc, path)
    doc2 = Document()
    fileio.load_document(doc2, path)
    assert doc2.group(g.id).meta["mesh_settings"] == {"material_flow": 90.0}


# ----------------------------------------------------------- the per-mesh mechanism

def _printer():
    return cura_profiles.Library(RESOURCES).printer("creality_ender3")


def _stacks():
    p = _printer()
    return p.stacks(p.default_variant(), p.default_material(), "standard", {})


def _layer_moves(block) -> list:
    """The extruding XY moves of one layer block, as (middle x, middle y, mm of filament, mm of
    travel). A move without coordinates carries the coordinates before it, as G-code means it."""
    moves = []
    x = y = e = None
    for line in block.splitlines():
        code = line.split(";", 1)[0].strip()
        if code[:2] not in ("G0", "G1", "G00", "G01"):
            continue
        gx = re.search(r"\bX(-?\d*\.?\d+)", code)
        gy = re.search(r"\bY(-?\d*\.?\d+)", code)
        ge = re.search(r"\bE(-?\d*\.?\d+)", code)
        nx = float(gx.group(1)) if gx else x
        ny = float(gy.group(1)) if gy else y
        ne = float(ge.group(1)) if ge else None
        if (ne is not None and x is not None and y is not None and e is not None
                and (nx != x or ny != y) and ne - e > 1e-9):
            moves.append((0.5 * (x + nx), 0.5 * (y + ny), ne - e,
                          ((nx - x) ** 2 + (ny - y) ** 2) ** 0.5))
        x, y = nx, ny
        if ne is not None:
            e = ne
    return moves


def _top_layer(text):
    """The top layer block that extrudes."""
    for block in reversed(text.split(";LAYER:")[1:]):
        if sum(m[3] for m in _layer_moves(block)) > 10:
            return block
    raise AssertionError("no top layer with an extrusion was found")


def _top_layer_per_mm(text):
    """The extrusion (mm of filament) per millimetre of XY travel on the top layer that extrudes."""
    moves = _layer_moves(_top_layer(text))
    return sum(m[2] for m in moves) / sum(m[3] for m in moves)


def _top_layer_per_mm_per_island(text) -> list:
    """The same per island of that layer, left to right. Parts that stand apart are separate
    islands there, so the moves fall apart on the gap between them and every group says what its
    own part printed with."""
    moves = _layer_moves(_top_layer(text))
    xs = sorted({round(m[0], 3) for m in moves})
    assert len(xs) > 1, "this layer is one point wide"
    cut = max(range(len(xs) - 1), key=lambda i: xs[i + 1] - xs[i])
    assert xs[cut + 1] - xs[cut] > 1.0, "the parts of this layer do not lie apart"
    middle = 0.5 * (xs[cut] + xs[cut + 1])
    out = []
    for left in (True, False):
        keep = [m for m in moves if (m[0] < middle) is left]
        if keep:
            out.append(sum(m[2] for m in keep) / sum(m[3] for m in keep))
    return out


@needs_engine
def test_extrusion_scales_with_the_flow_on_the_real_engine(tmp_path):
    """The same pad printed at 110 % lays 110/90 times the plastic per millimetre it travels as at
    90 %: the flow really drives the extrusion, so a row of pads each printed with its own flow
    reads as a walk from lean to rich."""
    printer = _printer()
    mm = made(labels=False)[0].to_mesh()
    verts, tris = np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts)
    per = {}
    for flow in (90.0, 110.0):
        st = printer.stacks(printer.default_variant(), printer.default_material(), "standard",
                            {"infill_sparse_density": 0.0, "material_flow": flow})
        out = tmp_path / f"flow{flow:g}.gcode"
        slicer.slice_meshes(st, [(verts, tris, 0)], str(out))
        per[flow] = _top_layer_per_mm(out.read_text(encoding="utf-8", errors="replace"))
    ratio = per[110.0] / per[90.0]
    assert ratio == pytest.approx(110.0 / 90.0, rel=0.03), per


def test_only_per_mesh_settings_pass():
    st = _stacks()
    got = slicer.per_mesh_settings(st, {"material_flow": 97.5})
    assert got["material_flow"] == 97.5, "a per-mesh setting was dropped"
    # what CuraEngine reads for a wall is what material_flow works out to, so that comes along
    assert got["wall_material_flow"] == 97.5 and got["infill_material_flow"] == 97.5
    derived = slicer.worked_out_from(st.printer.definition.settings, ["material_flow"])
    assert set(got) <= derived | {"material_flow"}, "a setting that has nothing to do with it"


def test_a_value_the_print_uses_anyway_adds_nothing():
    """A part that overrides a setting with the value the print has anyway needs no derived values
    either: the whole print already works out to them."""
    st = _stacks()
    assert slicer.per_mesh_settings(st, {"material_flow": 100.0}) == {"material_flow": 100.0}
    assert slicer.per_mesh_settings(st, {"infill_sparse_density": 20}) == \
        {"infill_sparse_density": 20}


def test_the_line_distance_a_density_works_out_to_goes_with_it():
    """CuraEngine spaces the infill by `infill_line_distance` and does not look at the density, so
    a part with its own density has to hand over the distance that follows from it."""
    st = _stacks()
    assert st.global_settings()["infill_line_distance"] == pytest.approx(6.0)     # 20 % infill
    got = slicer.per_mesh_settings(st, {"infill_sparse_density": 40.0})
    assert got["infill_line_distance"] == pytest.approx(3.0)
    assert got["infill_sparse_density"] == 40.0


def test_a_pattern_takes_what_follows_from_it_along():
    st = _stacks()
    got = slicer.per_mesh_settings(st, {"infill_pattern": "gyroid"})
    assert got["infill_pattern"] == "gyroid"
    assert got["infill_line_distance"] == pytest.approx(2.0), "the gyroid needs more lines per mm"


def test_a_count_that_changes_nothing_derives_nothing():
    """A formula can work out differently by a rounding error alone; that is no reason to hand the
    engine a setting of its own."""
    st = _stacks()
    assert slicer.per_mesh_settings(st, {"top_layers": 4}) == {"top_layers": 4}


def test_worked_out_from_walks_the_formulas():
    st = _stacks()
    definition = st.printer.definition.settings
    assert "infill_line_distance" in slicer.worked_out_from(definition,
                                                            ["infill_sparse_density"])
    assert "cross_infill_pocket_size" in slicer.worked_out_from(definition,
                                                                ["infill_sparse_density"]), \
        "the pocket size reads the line distance, which reads the density"
    assert slicer.worked_out_from(definition, ["no_such_setting"]) == set()
    assert slicer.worked_out_from(definition, []) == set()


def test_a_setting_that_is_no_number_and_a_list_reach_the_engine_as_words():
    assert slicer.command_value(True) == "true" and slicer.command_value(False) == "false"
    assert slicer.command_value(12) == "12" and slicer.command_value(2.5) == "2.5"
    assert slicer.command_value([0, 90]) == "[0,90]", "CuraEngine reads a list between brackets"
    assert slicer.command_value("gyroid") == "gyroid"


def test_a_global_only_key_is_dropped_with_a_warning(caplog):
    st = _stacks()
    with caplog.at_level(logging.WARNING, logger="volura.slicer"):
        got = slicer.per_mesh_settings(st, {"material_flow": 100.0, "infill_extruder_nr": 0})
    assert got == {"material_flow": 100.0}, "a global-only setting reached the engine"
    assert any("infill_extruder_nr" in r.getMessage() for r in caplog.records)


def test_a_key_the_definition_does_not_know_is_dropped(caplog):
    st = _stacks()
    with caplog.at_level(logging.WARNING, logger="volura.slicer"):
        assert slicer.per_mesh_settings(st, {"not_a_cura_setting": 1}) == {}
    assert any("not_a_cura_setting" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "plenty", "90", {}, [1, 2], True))
def test_a_value_that_fits_no_setting_is_dropped(caplog, bad):
    """A file can hold anything under a key, and CuraEngine ends a whole print on a value it cannot
    turn into what its setting wants: such a value goes, with a warning, like a key that is not
    per-mesh, so one wrong value in a file prints the print instead of nothing."""
    st = _stacks()
    with caplog.at_level(logging.WARNING, logger="volura.slicer"):
        assert slicer.per_mesh_settings(st, {"infill_sparse_density": bad}) == {}
    assert any("infill_sparse_density" in r.getMessage() for r in caplog.records)


def test_every_kind_of_setting_wants_its_own_kind_of_value():
    st = _stacks()
    assert slicer.per_mesh_settings(st, {"infill_pattern": 42}) == {}, "a number is no pattern"
    assert slicer.per_mesh_settings(st, {"wall_line_count": "two"}) == {}, "a word is no count"
    assert slicer.per_mesh_settings(st, {"top_layers": None}) == {}
    assert slicer.per_mesh_settings(st, {"infill_angles": [0, 90]}) == {"infill_angles": [0, 90]}


def test_no_settings_is_nothing_to_do():
    st = _stacks()
    assert slicer.per_mesh_settings(st, None) == {}
    assert slicer.per_mesh_settings(st, {}) == {}


def test_the_part_reads_its_flow_from_the_group_meta():
    doc = Document()
    g = ops.group_from_manifold(doc, made(labels=False)[0], "#f2f2f2", "Flow 90 %")
    g.meta = {"mesh_settings": {"material_flow": 92.5}}
    part = ops.parts(doc)[0]
    assert part.mesh_settings == {"material_flow": 92.5}


def test_a_group_without_meta_carries_no_mesh_settings():
    doc = Document()
    ops.group_from_manifold(doc, made(labels=False)[0], "#f2f2f2", "plain")
    part = ops.parts(doc)[0]
    assert part.mesh_settings == {}


class EngineSettings:
    """`subprocess` with its `Popen` wrapped, so a test reads what CuraEngine was asked to slice
    with before its temporary folder is gone again: the settings file `-r` names, and the command
    line that hands every mesh the settings it wants for itself."""

    def __init__(self, docs, cmds=None):
        self.docs = docs
        self.cmds = [] if cmds is None else cmds

    def Popen(self, cmd, *args, **kwargs):
        with open(cmd[cmd.index("-r") + 1], encoding="utf-8") as fh:
            self.docs.append(json.load(fh))
        self.cmds.append(list(cmd))
        return subprocess.Popen(cmd, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(subprocess, name)


def _own_of_every_mesh(cmd) -> dict:
    """The command line of the slice as {mesh file: {setting: value}}: every `-l` names a mesh and
    the `-s` settings behind it belong to that mesh alone."""
    got, i = {}, 0
    while i < len(cmd):
        if cmd[i] == "-l":
            own, walk = {}, i + 2
            while walk + 1 < len(cmd) and cmd[walk] == "-s":
                key, _, value = cmd[walk + 1].partition("=")
                own[key] = value
                walk += 2
            got[cmd[i + 1]] = own
            i = walk
        else:
            i += 1
    return got


@needs_engine
def test_the_slice_hands_every_pad_its_flow_as_that_pad_alone(tmp_path, monkeypatch):
    docs, cmds = [], []
    monkeypatch.setattr(slicer, "subprocess", EngineSettings(docs, cmds))
    st = _stacks()
    pads, fl = made(), flow_rate_test.flows(**PLA)
    meshes = []
    for m, f in zip(pads, fl):
        mm = m.to_mesh()
        meshes.append((np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0,
                       slicer.per_mesh_settings(st, {"material_flow": f})))
    out = tmp_path / "flow.gcode"
    result = slicer.slice_meshes(st, meshes, str(out))
    assert out.is_file() and result.layers > 0, "the pads did not slice"
    own = _own_of_every_mesh(cmds[0])
    doc = docs[0]
    assert len(own) == len(fl), "every pad is its own mesh"
    for i, f in enumerate(fl):
        got = own[f"part{i}.stl"]
        assert float(got["material_flow"]) == pytest.approx(f), f"pad {i} lost its flow"
        if f != doc["global"]["material_flow"]:
            assert float(got["wall_material_flow"]) == pytest.approx(f), \
                "and what that flow works out to for a wall"
        else:
            # the pad in the middle of the walk wants what the print uses anyway: nothing to add
            assert "wall_material_flow" not in got, "a value the print has anyway needs no setting"
    # the settings of one pad are not slipped into the document of the whole print
    assert not [name for name in doc if name.endswith(".stl")], \
        "a pad with settings of its own also got a model key"
    assert doc["global"]["material_flow"] == pytest.approx(100.0), "the print keeps its own flow"


@needs_engine
def test_a_bed_that_asks_for_nothing_still_loads_from_the_settings_document(tmp_path, monkeypatch):
    """Only a part with settings of its own is named on the command line; every other part loads the
    way it always did, so a big bed stays one short command and the engine still gets all of them."""
    docs, cmds = [], []
    monkeypatch.setattr(slicer, "subprocess", EngineSettings(docs, cmds))
    st = _stacks()
    meshes = []
    for i, pad in enumerate(made(labels=False)[:4]):
        mm = pad.to_mesh()
        own = slicer.per_mesh_settings(st, {"material_flow": 95.0}) if i == 1 else {}
        meshes.append((np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0, own))
    out = tmp_path / "mixed.gcode"
    result = slicer.slice_meshes(st, meshes, str(out))
    assert out.is_file() and result.layers > 0 and result.grams > 0, "the pads did not all slice"
    doc, cmd = docs[0], cmds[0]
    assert sorted(k for k in doc if k.endswith(".stl")) == ["part0.stl", "part2.stl", "part3.stl"], \
        "the parts that ask for nothing come from the document"
    assert doc["part0.stl"] == {"extruder_nr": 0}, "a plain part only says who prints it"
    own = _own_of_every_mesh(cmd)
    assert list(own) == ["part1.stl"], "more than the part with settings stands on the command line"
    assert float(own["part1.stl"]["material_flow"]) == pytest.approx(95.0)


@needs_engine
def test_a_bed_whose_parts_all_want_settings_says_so_before_the_engine_chokes(tmp_path,
                                                                              monkeypatch):
    """Windows hands a program its arguments as one string of at most 32767 characters, and the
    settings of a part cost a line of those: a bed that asks for too much is refused with words the
    user can act on, rather than an engine that stops without a word."""
    monkeypatch.setattr(slicer, "subprocess", EngineSettings([], []))
    st = _stacks()
    own = slicer.per_mesh_settings(st, {"material_flow": 95.0})
    meshes = []
    for i in range(200):
        cube = mf.Manifold.cube((6.0, 6.0, 6.0)).translate((9.0 * (i % 14), 9.0 * (i // 14), 0.0))
        mm = cube.to_mesh()
        meshes.append((np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0,
                       dict(own)))
    with pytest.raises(RuntimeError, match="settings of their own"):
        slicer.slice_meshes(st, meshes, str(tmp_path / "never.gcode"))
    assert not (tmp_path / "never.gcode").exists()


@needs_engine
def test_the_flow_of_one_pad_leaves_the_pad_next_to_it_alone(tmp_path):
    """Two identical pads standing apart: the left one asks for 80 % flow, the right one takes what
    the print uses. Measured per pad on the top layer, the left one extrudes about 0.8 times what it
    did before and the right one exactly the same: a setting of one mesh stays on that mesh."""
    st = _stacks()
    pad = made(labels=False)[0]
    b = np.asarray(pad.bounding_box(), dtype=float)
    apart = 0.5 * (b[3] - b[0]) + 2.0        # every pad half a pad and 2 mm from the middle
    meshes = []
    for side in (-1.0, 1.0):
        one = pad.translate((-0.5 * (b[0] + b[3]) + side * apart, -0.5 * (b[1] + b[4]), 0.0))
        mm = one.to_mesh()
        meshes.append((np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0, {}))
    plain = tmp_path / "both-plain.gcode"
    slicer.slice_meshes(st, meshes, str(plain))
    was = _top_layer_per_mm_per_island(plain.read_text(encoding="utf-8", errors="replace"))
    lean = slicer.per_mesh_settings(st, {"material_flow": 80.0})
    own = [(meshes[0][0], meshes[0][1], 0, lean), (meshes[1][0], meshes[1][1], 0, {})]
    mixed = tmp_path / "left-lean.gcode"
    slicer.slice_meshes(st, own, str(mixed))
    now = _top_layer_per_mm_per_island(mixed.read_text(encoding="utf-8", errors="replace"))
    assert len(was) == len(now) == 2, "the two pads did not stay two islands"
    assert now[0] == pytest.approx(0.8 * was[0], rel=0.04), \
        f"the lean pad extrudes {now[0]} per mm, it was {was[0]}"
    assert now[1] == pytest.approx(was[1], rel=0.02), "the flow of one pad reached the other"


@needs_engine
def test_a_two_nozzle_printer_keeps_both_the_nozzle_and_the_settings_of_a_part(tmp_path):
    """Which nozzle prints a mesh and what that mesh wants for itself are handed over at the same
    place on the command line: a printer with two extruders still prints with both of them, and the
    part on the second one prints with four walls where the plain part prints with the print's."""
    dual = cura_profiles.Library(RESOURCES).printer("creality_cr-x")
    variant = dual.default_variant()
    st = dual.stacks(variant, dual.default_material(variant), "standard",
                     {"infill_sparse_density": 0.0})
    assert len(st.extruders) == 2, "this printer has two nozzles"

    def cube(size, at):
        mm = mf.Manifold.cube((size, size, size)).translate(at).to_mesh()
        return np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts)

    more = slicer.per_mesh_settings(st, {"wall_line_count": 4})
    meshes = [cube(20.0, (-30.0, 0.0, 0.0)) + (0, {}), cube(20.0, (30.0, 0.0, 0.0)) + (1, more)]
    out = tmp_path / "two-nozzles.gcode"
    result = slicer.slice_meshes(st, meshes, str(out))
    volumes = list(result.volume_mm3)
    assert len(volumes) == 2 and all(v > 0 for v in volumes), f"both nozzles have to print: {volumes}"
    assert volumes[1] > volumes[0], \
        f"the part with four walls printed {volumes[1]}, the plain one {volumes[0]}"


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    for key, value in (("flow_rate_test_start", 90.0), ("flow_rate_test_end", 110.0),
                       ("flow_rate_test_step", 2.5)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_builds_a_named_row_and_keeps_the_values(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["default"] = [spins[0].value(), spins[1].value(), spins[2].value()]
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.flow_rate_test_dialog()
        assert seen["default"] == [pytest.approx(90.0), pytest.approx(110.0), pytest.approx(2.5)]
        assert pump_until(lambda: len(win.doc.groups) == 9), win.doc.groups
        fl = flow_rate_test.flows(90.0, 110.0, 2.5)
        for g, f in zip(win.doc.groups, fl):
            assert g.name == "Flow %s %%" % flow_rate_test.flow_text(f)
            assert g.meta["mesh_settings"] == {"material_flow": f}
            assert g.ents.to_manifold().status() == mf.Error.NoError
        assert pump_until(lambda: "flow" in win.hint.text().lower()), win.hint.text()
        st = app_module.Settings()
        assert st.flow_rate_test_start == pytest.approx(90.0)
        assert st.flow_rate_test_end == pytest.approx(110.0)
        assert st.flow_rate_test_step == pytest.approx(2.5)
    finally:
        forget_the_values(win.settings)


def test_the_dialog_refuses_too_many_pads(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(50.0)
        spins[1].setValue(150.0)
        spins[2].setValue(5.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.flow_rate_test_dialog()
    assert win.doc.groups == [], "a test with twenty-one pads was built anyway"
    assert any("at most" in text.lower() for _title, text in asked), asked


def test_one_undo_takes_the_whole_row_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_flow_rate_test(90.0, 110.0, 2.5, flow_rate_test.flows(**PLA))
    assert pump_until(lambda: len(win.doc.groups) == 9)
    assert len(win.doc.undo_stack) == steps + 1, "the row is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("flow_rate_test_start", "flow_rate_test_end", "flow_rate_test_step")
    saved = {k: q.value(k) for k in keys}
    try:
        q.setValue("flow_rate_test_start", 999.0)
        q.setValue("flow_rate_test_end", -999.0)
        q.setValue("flow_rate_test_step", 999.0)
        q.sync()
        s = app_module.Settings()
        assert s.flow_rate_test_start == 150.0
        assert s.flow_rate_test_end == 50.0
        assert s.flow_rate_test_step == 25.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_pressure_advance_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Flow Rate Test") == labels.index("Pressure Advance Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
