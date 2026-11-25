"""What every part of a print costs: the time and the filament of every part of a slice.

CuraEngine writes a `;MESH:` comment whenever it moves on to another mesh, and `;MESH:NONMESH` for
everything that lays no plastic of a part of the model - the skirt and brim, the prime tower, the
support it builds for the print as a whole and the travel from one part to the next.
`volura/gcode_preview.py` follows those comments for every move it already measures, and its
`per_part` weighs them per part, so the print preview can say which part takes three of the five
hours and how many grams every part costs. The meshes go to the engine as `part0.stl`,
`part1.stl`, ... (see `volura/slicer.py`) and the slice hands those names back with its result,
which is what makes a row point at the group its part is in the model. File > Slice shows the rows
in a table under the preview - one plate at a time, because that is what a slice is - and clicking
a row lights that part up in the print and selects it in the model.

These tests (Qt offscreen) check the arithmetic of a hand-written G-code with two meshes, a
NONMESH block and known feed rates - the rows add up to the print to the second and to the
milligram -, the parts of that file that print nothing, the meshes no part owns and a print of one
part (no table), what a click or Enter on a row does, and, with the real CuraEngine, that two cubes
of a different size really divide the time and the grams of the print over their parts. The helpers
`add_cube` and `pump_until` follow tests/test_plates.py.

Run with:  python -m pytest tests/test_part_times.py
"""
import math
import os
import sys
import time
from types import SimpleNamespace

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QWidget  # noqa: E402

import volura.app as app_module  # noqa: E402
import volura.print_preview as pp  # noqa: E402
import volura.window_print as wp  # noqa: E402
from volura import cura_profiles, gcode_preview as gp, ops, slicer  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.slice_dialog import Multicolor  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: two parts on one bed and, under `;MESH:NONMESH`, the travel at the end of the first layer and
#: the support of the second: CuraEngine builds that support for the print as a whole and writes
#: it before the mesh of the part under it, so it is no part's own. The skirt starts the print
#: before any mesh is named. Every move has a known length and feed; the first layer ends with a
#: `;TIME_ELAPSED:` of twice what its feeds work out to - which is how the preview scales its own
#: times to the ones the engine measured - and the second with its own time as it stands.
TWO = """\
;FLAVOR:Marlin
;TIME:6
;Filament used: 0.9m
G28 ; home, before the first layer: left out
;LAYER:0
G0 F6000 X0 Y0 Z0.2
;TYPE:SKIRT
G1 F6000 X10 Y0 E50
;MESH:part0
;TYPE:WALL-OUTER
G1 F1200 X30 Y0 E250
;MESH:part1
G0 F6000 X50 Y0
;TYPE:FILL
G1 F1200 X50 Y20 E450
;MESH:NONMESH
;TYPE:FILL
G1 F1200 X50 Y30 E500
;TIME_ELAPSED:5.6
;LAYER:1
G0 F6000 X50 Y30 Z0.4
;TYPE:SUPPORT
G1 F2400 X50 Y50 E800
;MESH:part0
;TYPE:WALL-INNER
G1 F2400 X50 Y60 E900
;TIME_ELAPSED:6.35
"""

#: what the slice knows of its meshes: {mesh name in the G-code: (group id, name in the model)}
PARTS = {"part0": (11, "left"), "part1": (12, "right")}

#: the times of `TWO` per part, and the millimetres of filament they push out (the E words of a
#: hand-written file are bigger than a real print's, so that the grams land in a real range)
LEFT_SECONDS, RIGHT_SECONDS, OTHER_SECONDS = 2.25, 2.4, 1.7
PRINT_SECONDS = LEFT_SECONDS + RIGHT_SECONDS + OTHER_SECONDS     # what ;TIME_ELAPSED adds up to

#: the measure this file weighs its filament with: an area of pi mm2, so one mm of filament is
#: pi mm3 and, at a density of 2 g/cm3, 2 pi mg
DENSITY, FILAMENT = 2.0, 2.0
MILLIGRAM = math.pi * 1.0 ** 2 * DENSITY / 1000.0     # grams of one mm of filament


def rows_of(text=TWO, parts=PARTS, **kw):
    """The preview and the rows of `per_part` of hand-written G-code."""
    preview = gp.parse(text.splitlines(), filament=FILAMENT)
    return preview, gp.per_part(preview, parts, density=DENSITY, filament=FILAMENT, **kw)


def named(one, preview):
    """The name of the mesh of a move; "?" for a move before the first `;MESH:` of the file."""
    return "?" if int(one) == gp.NO_MESH else preview.mesh_names[int(one)]


def by_name(rows):
    """The rows keyed by what they are named, the row of no part under "other"."""
    return {one.name or "other": one for one in rows}


# ------------------------------------------------------------------ which mesh lays what

def test_every_move_of_a_layer_knows_the_mesh_it_belongs_to():
    preview = gp.parse(TWO.splitlines())
    assert preview.mesh_names == ["part0", "part1", "NONMESH"]
    first, second = preview.layers
    assert [named(one, preview) for one in first.mesh] == \
        ["?", "part0", "part1", "part1", "NONMESH"]
    # the last `;MESH:` of a layer carries over into the next one, as CuraEngine writes it: the
    # support at the head of the second layer prints under the NONMESH that ended the first
    assert [named(one, preview) for one in second.mesh] == ["NONMESH", "part0"]


def test_moves_before_the_first_mesh_comment_belong_to_no_part():
    preview = gp.parse([";LAYER:0", "G1 F1200 X10 E1", ";MESH:part0", "G1 X20 E2"])
    assert preview.mesh_names == ["part0"]
    assert [int(code) for code in preview.layers[0].mesh] == [gp.NO_MESH, 0]


# ---------------------------------------------------------------- the rows of the parts

def test_the_time_of_every_part_adds_up_to_its_own_moves():
    _preview, rows = rows_of()
    got = {key: one.seconds for key, one in by_name(rows).items()}
    assert got == pytest.approx({"left": LEFT_SECONDS, "right": RIGHT_SECONDS,
                                 "other": OTHER_SECONDS})
    assert [one.name for one in rows] == ["right", "left", ""], "the rows come longest first"


def test_every_part_is_weighed_with_the_density_and_the_diameter_of_the_print():
    _preview, rows = rows_of()
    got = by_name(rows)
    assert got["left"].metres == pytest.approx(0.3)          # 300 mm of filament
    assert got["left"].grams == pytest.approx(300 * MILLIGRAM)
    assert got["right"].grams == pytest.approx(200 * MILLIGRAM)
    assert got["other"].grams == pytest.approx(400 * MILLIGRAM)


def test_the_travel_the_skirt_and_the_support_are_no_part():
    """The skirt comes before the first `;MESH:`, the travel and the fill at the end of the first
    layer and the support of the second under `;MESH:NONMESH`: CuraEngine builds the support of
    the print together, so it is no part's own, and the costs of all those moves come together in
    the row of "Other"."""
    _preview, rows = rows_of()
    other = by_name(rows)["other"]
    assert other.other and other.name == "" and other.group_id is None
    # the first layer is scaled by 5.6 / 2.8: the skirt of 0.1 s weighs 0.2, the fill at its end
    # 1.0; the support of the second layer keeps its own 0.5 s
    assert other.seconds == pytest.approx(0.2 + 1.0 + 0.5)
    assert other.metres == pytest.approx(0.4)                # 50 mm prime + 50 mm + 300 mm support
    assert other.codes == (2, gp.NO_MESH)                    # NONMESH and the unnamed prime


def test_the_rows_add_up_to_the_whole_print_the_dialog_shows():
    preview, rows = rows_of()
    assert sum(one.seconds for one in rows) == pytest.approx(preview.seconds)
    assert sum(one.metres for one in rows) * 1000.0 == pytest.approx(
        float(preview.filament_mm().sum()))
    assert sum(one.grams for one in rows) == pytest.approx(900 * MILLIGRAM)
    assert sum(one.share for one in rows) == pytest.approx(1.0)
    assert by_name(rows)["left"].share == pytest.approx(LEFT_SECONDS / PRINT_SECONDS)


def test_a_support_volume_of_the_model_lays_no_plastic_of_a_part():
    """A support blocker or enforcer goes to the engine behind the parts, so its mesh comes after
    them under a name that belongs to no part; what it turns into belongs to no part either."""
    text = (";LAYER:0\n;MESH:part0\nG1 F1200 X10 E1\n"
            ";MESH:part1\nG1 X20 E2\n;MESH:part2\nG1 X30 E4\n")
    preview, rows = rows_of(text=text)
    assert preview.mesh_names == ["part0", "part1", "part2"]
    assert [one.name for one in rows if not one.other] == ["left", "right"]
    other = by_name(rows)["other"]
    assert other.other and other.codes == (2,)
    assert other.seconds == pytest.approx(preview.layers[0].seconds / 3.0)


def test_a_part_is_weighed_with_the_filament_of_its_print():
    """The grams of a part mean nothing without the filament they were weighed with: the diameter
    the extruder pushes and the density of the material of the slice, and the generic values where
    a stack says nothing about its filament."""
    assert slicer.filament_measure(object()) == (1.75, 1.24)
    nothing = SimpleNamespace(extruders=[], material=None)
    assert slicer.filament_measure(nothing) == (1.75, 1.24)
    empty = SimpleNamespace(extruders=[{"material_diameter": ""}],
                            material=SimpleNamespace(metadata={}))
    assert slicer.filament_measure(empty) == (1.75, 1.24)
    petg = SimpleNamespace(extruders=[{"material_diameter": 2.85}],
                           material=SimpleNamespace(metadata={"density": 1.27}))
    assert slicer.filament_measure(petg) == (2.85, 1.27)


def test_a_part_that_printed_nothing_keeps_a_row_of_nothing():
    """A part the slice handed the engine but that printed nothing is still a part."""
    _preview, rows = rows_of(parts={**PARTS, "part9": (13, "phantom")})
    phantom = next(one for one in rows if one.group_id == 13)
    assert (phantom.seconds, phantom.grams, phantom.metres, phantom.share) == (0.0, 0.0, 0.0, 0.0)
    assert phantom.codes == ()
    assert sum(one.share for one in rows) == pytest.approx(1.0)


def test_the_meshes_of_one_part_add_up_to_one_row():
    """A part painted in a colour that comes off another extruder prints with two meshes."""
    _preview, rows = rows_of(parts={**PARTS, "part1": (11, "left")})
    left = [one for one in rows if one.group_id == 11]
    assert len(left) == 1 and len(rows) == 2
    assert left[0].seconds == pytest.approx(LEFT_SECONDS + RIGHT_SECONDS)
    assert sorted(left[0].codes) == [0, 1]


def test_every_mesh_the_slice_does_not_name_is_no_part():
    preview, rows = rows_of(parts={})
    assert len(rows) == 1 and rows[0].other and rows[0].share == pytest.approx(1.0)
    assert rows[0].seconds == pytest.approx(preview.seconds)


def test_a_g_code_that_names_no_mesh_is_one_row_of_no_part():
    text = ";LAYER:0\nG1 F1200 X10 E1\nG1 X20 E2\n;TIME_ELAPSED:2.0\n"
    preview, rows = rows_of(text=text, parts={})
    assert len(rows) == 1 and rows[0].other and rows[0].seconds == pytest.approx(2.0)


def test_nothing_at_all_is_a_print_of_no_parts_and_of_no_time():
    """A G-code without a single move has no costs to divide; the parts it was sliced from keep
    their row, of nothing, so it is to see that nothing printed."""
    preview = gp.parse([])
    assert gp.per_part(preview) == []
    rows = gp.per_part(preview, PARTS)
    assert [one.name for one in rows] == ["left", "right"]
    assert all(one.seconds == 0.0 and one.share == 0.0 for one in rows)


def test_a_print_without_a_skirt_or_support_needs_no_row_of_other():
    """Every mesh of the G-code belongs to a part and nothing prints beside them: nothing left to
    put in a row of "Other"."""
    text = ";LAYER:0\n;MESH:part0\nG1 F1200 X10 E1\n;MESH:part1\nG1 X20 E2\n"
    _preview, rows = rows_of(text=text)
    assert [one.name for one in rows] == ["left", "right"]
    assert sum(one.share for one in rows) == pytest.approx(1.0)


def test_a_layer_that_says_nothing_about_its_meshes_is_all_no_part():
    """A preview of moves without a mesh of their own - an old reader, a hand-made layer - does
    not crash on missing information and knows no parts, so everything is "Other"."""
    layer = gp.Layer(0, 0.2, np.array([[0.0, 0.0, 10.0, 0.0]], dtype=np.float32),
                     np.array([gp.KIND["FILL"]], dtype=np.uint8), np.array([1.0]),
                     pushed=np.array([2.0]), material=np.array([0], dtype=np.uint8))
    rows = gp.per_part(gp.Preview([layer]), PARTS, density=DENSITY, filament=FILAMENT)
    other = by_name(rows)["other"]
    assert other.other and other.codes == (gp.NO_MESH,)
    assert other.seconds == pytest.approx(1.0)
    assert other.grams == pytest.approx(2.0 * MILLIGRAM)
    assert other.share == pytest.approx(1.0), "there is nothing else the time could mean here"


# --------------------------------------------------------------------------- the engine

def _cube(size, at):
    m = mf.Manifold.cube((size, size, size)).translate(at).to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def _stacks(infill=20.0):
    printer = cura_profiles.Library(RESOURCES).printer("creality_ender3")
    return printer.stacks(printer.default_variant(), printer.default_material(), "standard",
                          {"infill_sparse_density": infill})


@needs_engine
def test_two_cubes_of_a_different_size_divide_the_print_between_them(tmp_path):
    """The real engine on two real cubes: the small one on part0, the big one on part1."""
    stacks = _stacks()
    out = tmp_path / "two-cubes.gcode"
    result = slicer.slice_meshes(stacks, [_cube(10.0, (20.0, 40.0, 0.0)) + (0,),
                                          _cube(20.0, (60.0, 40.0, 0.0)) + (0,)], str(out))
    assert result.mesh_names == [slicer.mesh_name(0), slicer.mesh_name(1)]
    diameter, density = slicer.filament_measure(stacks)
    preview = gp.read_gcode(str(out), diameter)
    rows = gp.per_part(preview, {"part0": (1, "small"), "part1": (2, "big")},
                       density=density, filament=diameter)
    small, big = by_name(rows)["small"], by_name(rows)["big"]
    other = [one for one in rows if one.other]
    assert len(rows) == 3 and len(other) == 1, rows
    assert rows == sorted(rows, key=lambda one: -one.seconds) and rows[0].name == "big"
    assert big.seconds > small.seconds, "the cube of twice the side takes less time"
    # the volume ratio of the two cubes is 8; walls and the skin of the small one eat into that,
    # but the big one has to print clearly more plastic
    assert big.grams / small.grams >= 3.0, "the bigger cube prints less than three times as much"
    assert big.metres / small.metres >= 3.0
    # the skirt and the travel from one part to the other are no part, so they have their own row
    assert other[0].seconds > 0.0 and other[0].grams > 0.0
    # and everything together is what the summary above the table says
    assert sum(one.seconds for one in rows) == pytest.approx(result.seconds, abs=1.0)
    assert sum(one.grams for one in rows) == pytest.approx(result.grams, abs=0.01)
    assert sum(one.metres for one in rows) == pytest.approx(sum(result.filament_m), abs=0.01)
    assert sum(one.share for one in rows) == pytest.approx(1.0, abs=1e-6)


@needs_engine
def test_a_print_of_one_part_has_one_part_row_and_the_row_of_nothing(tmp_path):
    stacks = _stacks(infill=0.0)
    out = tmp_path / "one-cube.gcode"
    result = slicer.slice_meshes(stacks, [_cube(10.0, (30.0, 40.0, 0.0)) + (0,)], str(out))
    preview = gp.read_gcode(str(out), slicer.filament_measure(stacks)[0])
    rows = gp.per_part(preview, dict(zip(result.mesh_names, [(7, "only")])))
    assert len([one for one in rows if not one.other]) == 1
    assert by_name(rows)["only"].seconds > 0.0
    assert sum(one.seconds for one in rows) == pytest.approx(preview.seconds)


# ------------------------------------------------------------------------ the slice flow

class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])

    def critical(self, *args, **kwargs):
        self.asked.append(args[1:])


class FakeStacks:
    """Just what the slice reads out of the stacks of a printer: the size of its machine."""

    def __init__(self, bed=(200.0, 200.0, 200.0)):
        self.glob = {"machine_width": bed[0], "machine_depth": bed[1], "machine_height": bed[2]}
        self.extruders = []                    # no filament of its own: 1.75 mm is what counts


class FakeSlicer:
    """Stands in for `slicer.slice_meshes`: writes the G-code of the meshes it was handed, named
    the way the engine names them, and says so in its result."""

    def __init__(self, text=TWO):
        self.text = text
        self.calls = []

    def __call__(self, stacks, meshes, out_path, engine=None, timeout=None, on_progress=None,
                 helpers=None):
        self.calls.append((len(meshes), len(helpers or ())))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(self.text)
        return slicer.SliceResult(path=out_path, seconds=PRINT_SECONDS, filament_m=[0.9],
                                  grams=900 * MILLIGRAM, layers=2,
                                  volume_mm3=[900 * math.pi * (FILAMENT / 2.0) ** 2],
                                  mesh_names=[slicer.mesh_name(i) for i in range(len(meshes))])


class SeeingDialog:
    """Stands in for the print preview: keeps the rows it was shown and who wants the clicks."""

    sent = saved_to = None

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, win, path, preview, bed, summary, ask=None, filaments=(), grams=0.0,
                 minutes=0.0, parts=(), on_select=None):
        self.seen.update(preview=preview, rows=list(parts), select=on_select)
        return self

    def exec(self):
        return QDialog.Accepted


SETTING_KEYS = ("bed_size", "slice_thumbnail", "filament_price", "print_queue", "printers",
                "spool_for_slot")


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs into the folder of the test, on a bed big enough for two cubes."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    monkeypatch.setattr(wp, "QMessageBox", box)      # the slice dialogs live in window_print
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    w = app_module.MainWindow()
    w.box = box
    w.settings.bed_size = (200.0, 200.0, 200.0)
    w.settings.slice_thumbnail = "off"
    w.settings.filament_price = 20.0
    w.settings.print_queue = []
    w.settings.printers = []
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


def add_cube(doc, at, size=20.0, name="cube"):
    """A closed cube of `size` mm with its corner on `at`, as its own group."""
    m = mf.Manifold.cube((float(size), float(size), float(size))).translate(at)
    return ops.group_from_manifold(doc, m, "#cccccc", name)


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    got = predicate()
    for _ in range(30):
        _app.processEvents()
        time.sleep(0.005)
    return got


def sliced(win, monkeypatch):
    """What the preview of File > Slice is shown for the parts of the model: its rows, and the
    thing it calls when a row of them is clicked."""
    monkeypatch.setattr(slicer, "slice_meshes", FakeSlicer())
    seen = {}
    monkeypatch.setattr(wp, "PrintPreviewDialog", SeeingDialog(seen))
    win._slice_async(FakeStacks(), ops.parts(win.doc), "engine", "Test printer", Multicolor(1))
    assert pump_until(lambda: bool(seen)), "the slice never reached the preview"
    return seen


def test_the_rows_of_a_slice_point_at_the_groups_they_measured(win, monkeypatch):
    left = add_cube(win.doc, (10.0, 10.0, 0.0), name="left")
    right = add_cube(win.doc, (60.0, 10.0, 0.0), name="right")
    rows = sliced(win, monkeypatch)["rows"]
    assert {one.name for one in rows if not one.other} == {"left", "right"}
    assert {one.group_id for one in rows if not one.other} == {left.id, right.id}
    assert by_name(rows)["left"].seconds == pytest.approx(LEFT_SECONDS)
    assert [one for one in rows if one.other], "the travel and the skirt have no row of their own"
    assert not [one for one in rows if one.name not in ("left", "right", "")], rows


def test_a_click_on_a_row_of_the_slice_selects_its_part_in_the_model(win, monkeypatch):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="left")
    right = add_cube(win.doc, (60.0, 10.0, 0.0), name="right")
    seen = sliced(win, monkeypatch)
    seen["select"](by_name(seen["rows"])["right"].group_id)
    assert [g.id for g in win.doc.selected_groups()] == [right.id]
    seen["select"](None)                    # the row of "Other" has no group to point at
    assert [g.id for g in win.doc.selected_groups()] == [right.id], "it selected nothing at all"
    seen["select"]("a group that is no longer there")     # the model changed since the slice
    assert [g.id for g in win.doc.selected_groups()] == [right.id], "it selected a ghost"


# ------------------------------------------------------------------- the table in the preview

class Parent(QWidget):
    """A parent window for the preview that only carries the settings the preview reads."""

    def __init__(self):
        super().__init__()
        self.settings = SimpleNamespace(colour_blind=False, printers=[], spools=[],
                                        spool_for_slot=[])


def preview_dialog(preview, parts, chosen):
    """The real preview of a preview of `parts`, shown so its table has a size."""
    win = Parent()
    dlg = pp.PrintPreviewDialog(win, "x.gcode", preview, (0.0, 0.0, 100.0, 100.0), "the summary",
                                None, (), grams=1.0, minutes=2.0, parts=parts, on_select=chosen)
    dlg.resize(800, 700)
    dlg.show()
    QApplication.processEvents()
    return win, dlg


def click_row(dlg, row):
    """A click on a row of the "Per part" table, the way a person gives one."""
    rect = dlg.part_table.visualRect(dlg.part_table.model().index(row, 0))
    QTest.mouseClick(dlg.part_table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    QApplication.processEvents()


def test_two_parts_come_with_a_table_of_what_each_cost():
    preview, rows = rows_of()
    asked = []
    win, dlg = preview_dialog(preview, rows, asked.append)
    try:
        table = dlg.part_table
        assert table is not None and table.rowCount() == 3
        assert [table.item(r, 0).text() for r in range(3)] == ["right", "left", tr("Other")]
        assert table.item(0, 1).text() == "0:02"          # 2.4 s, the longest part first
        assert table.item(1, 2).text() == tr("{g:.1f} g, {m:.2f} m").format(
            g=300 * MILLIGRAM, m=0.3)
        assert table.item(0, 3).text() == tr("{pct} %").format(pct=100.0 * RIGHT_SECONDS
                                                               / PRINT_SECONDS)
        assert [one.name for one in dlg.parts] == ["right", "left", ""]
    finally:
        dlg.close()
        win.close()


def test_clicking_a_row_lights_its_part_up_in_the_print_and_selects_it():
    preview, rows = rows_of()
    asked = []
    win, dlg = preview_dialog(preview, rows, asked.append)
    try:
        click_row(dlg, 0)                                  # the row of "right", which is part1
        assert dlg.view.part == (1,), "the view does not follow the row you clicked"
        assert asked == [12], "the group of the row did not reach the window"
        own = np.repeat(np.isin(dlg.view.lines.part, by_name(rows)["right"].codes), pp.VERTS)
        bright = dlg.view.lines.focus(dlg.view.part)
        assert own.any(), "the print has no line of the part you clicked"
        assert (bright[own] == dlg.view.lines.col[own]).all(), "its own lines are not bright"
        assert (bright[~own] == dlg.view.lines.dim[~own]).all(), "the rest of the print is not dim"
        click_row(dlg, 0)                                  # the same row twice: the whole print
        assert dlg.view.part is None and dlg.view._focus is None
        assert len(asked) == 1, "the row you clicked twice selected something new"
        click_row(dlg, 1)
        assert dlg.view.part == (0,) and asked == [12, 11]
    finally:
        dlg.close()
        win.close()


def test_enter_on_a_row_of_the_table_does_what_a_click_does():
    """The keyboard reaches the table too: the arrow keys walk to a row, Enter chose it."""
    preview, rows = rows_of()
    asked = []
    win, dlg = preview_dialog(preview, rows, asked.append)
    try:
        dlg.part_table.setCurrentCell(1, 0)                  # the row of "left", part0
        QApplication.processEvents()
        QTest.keyClick(dlg.part_table, Qt.Key_Return)
        assert dlg.view.part == (0,), "Enter did not light the row the cursor stands on"
        assert asked == [11]
        QTest.keyClick(dlg.part_table, Qt.Key_Return)
        assert dlg.view.part is None and len(asked) == 1, "Enter twice showed a part again"
    finally:
        dlg.close()
        win.close()


def test_a_print_of_one_part_shows_no_table_and_keeps_its_totals():
    """Nothing to choose between: one part and the moves that print no part are no table."""
    preview, rows = rows_of(parts={"part0": (11, "left"), "part1": (11, "left")})
    assert len([one for one in rows if not one.other]) == 1
    win, dlg = preview_dialog(preview, rows, lambda group_id: None)
    try:
        assert dlg.part_table is None
        assert dlg.summary.text() == "the summary"
    finally:
        dlg.close()
        win.close()


def test_a_preview_of_a_g_code_that_names_no_parts_shows_no_table():
    """The preview of a G-code from the print history has no rows to show."""
    preview = gp.parse(TWO.splitlines(), filament=FILAMENT)
    win, dlg = preview_dialog(preview, (), lambda group_id: None)
    try:
        assert dlg.part_table is None and dlg.parts == []
    finally:
        dlg.close()
        win.close()
