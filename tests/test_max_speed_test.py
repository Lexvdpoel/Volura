"""Max Speed Test: a single-walled tube whose segments print at a rising volumetric flow.

`max_speed_test.max_speed_test` gives the tower as one closed volume standing on `z = 0`: a
cylinder of `DIAMETER` millimetres with a wall of one line of the nozzle, `segments` segments of
`SEG_H` millimetres high, with a rib of `RIB_H` millimetres on the wall at every boundary between
two segments and the flow of the first and last segment engraved in a tab at the foot.
`max_speed_test.flows` gives the flow of every segment, `speeds` the mm/s those flows mean at the
layer height and line width of the print, and `commands` the `M220 S<percent>` line every segment
is worth relative to the base speed - the percent depends on the slice, so the part keeps the
plan of flows in `g.meta["speed_steps"]` and the slice turns it into commands with its own stacks,
while `g.meta["mesh_settings"]` hands the part its single wall and its base speed through the
general per-mesh mechanism (`slicer.per_mesh_settings`, `-l` and `-s`).

These tests (Qt offscreen) weigh that: one closed volume on the bed, `segments * SEG_H` tall,
`DIAMETER` across with a wall of one line of the nozzle; a rib at every boundary between two
segments, `RIB_H` tall, and none between them or above the last one; the tab at the foot with the
two flows engraved shallow, and gone when this system has no fonts; the speed formula and its
percents, the presets and the validation (a step of 0, too few or too many segments, a flow that
is no number, a speed past 500 mm/s named by its segment); the plan and the mesh settings
round-tripping through a file; `with_commands` writing the M220 lines in order at the right
layers, next to the other tests' commands; the window (the summary with every segment's mm/s, the
speed limit refused, the bed turning flows into percents with the slice's stacks) and, when
CuraEngine is there, a real slice that speeds up at every segment and lays its walls at the base
speed that M220 scales.

Run with:  python -m pytest tests/test_max_speed_test.py
"""
import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QDoubleSpinBox, QLabel)

from volura import (app as app_module, cura_profiles, fileio, fonts,  # noqa: E402
                    max_speed_test, ops, pressure_advance_test, slicer, temperature_tower,
                    window_print)
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: the numbers the tests build with: three segments of 5, 10 and 15 mm³/s
SMALL = dict(start=5.0, end=15.0, step=5.0)
SEG_H = max_speed_test.SEG_H
D = max_speed_test.DIAMETER
ABS = 1e-4
#: the layer height and line width the tests convert flows with, the ones the core assumes
LH, LW = max_speed_test.DEFAULT_LAYER_HEIGHT, max_speed_test.DEFAULT_LINE_WIDTH
#: the settings of the dialog, back at what the program starts with
SETTINGS = (("max_speed_test_preset", ""), ("max_speed_test_start", 5.0),
            ("max_speed_test_end", 25.0), ("max_speed_test_step", 2.5))


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


class StubStacks:
    """Stands in for `cura_profiles.Stacks`: global settings the slice would use."""

    def __init__(self, layer_height=LH, line_width=LW):
        self.values = {"layer_height": layer_height, "line_width": line_width}

    def global_settings(self):
        return dict(self.values)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    for key, value in SETTINGS:
        setattr(w.settings, key, value)
    # the preview converts the flows with the printer of the last slice: here it must fall back
    # on the measures the test assumes, whatever an earlier test left in the same QSettings
    for key in ("slice_printer", "slice_variant", "slice_material", "slice_quality"):
        setattr(w.settings, key, "")
    w.settings.printers = []          # what stands under Preferences is up to the test
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def nofonts(monkeypatch):
    """No fonts: the tower is then exactly the tube with its ribs and tab, no numbers."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no numbers to test")


def made(**kw) -> mf.Manifold:
    return max_speed_test.max_speed_test(**dict(SMALL, **kw))


def numbers(**kw) -> dict:
    return max_speed_test.sizes(**dict(SMALL, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def probe(m, x, y, z, size=0.1) -> float:
    """How much of a little cube around `x, y, z` is plastic."""
    box = mf.Manifold.cube([size, size, size]).translate((x - 0.5 * size, y - 0.5 * size,
                                                          z - 0.5 * size))
    return (m ^ box).volume()


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the tube

@pytest.mark.parametrize("labels", (False, True))
def test_the_tower_is_one_volume_that_stands_on_the_bed_and_is_the_tube(labels):
    s = numbers()
    m = made(labels=labels)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the tower fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the tower does not stand on the bed"
    assert lo[0] == pytest.approx(-0.5 * D, abs=ABS)
    assert hi[0] == pytest.approx(0.5 * D + max_speed_test.TAB_OUT, abs=ABS), \
        "the tab at the foot is not part of the tower"
    assert lo[1] == pytest.approx(-0.5 * D, abs=ABS) and hi[1] == pytest.approx(0.5 * D, abs=ABS)
    assert hi[2] - lo[2] == pytest.approx(s["segments"] * SEG_H, abs=1e-3), "not as tall as asked"


def test_the_tower_is_taller_for_every_segment_it_has():
    for step in (5.0, 10.0):
        n = max_speed_test.sizes(5.0, 60.0, step)["segments"]
        lo, hi = bbox(max_speed_test.max_speed_test(5.0, 60.0, step))
        assert hi[2] - lo[2] == pytest.approx(n * SEG_H, abs=1e-3)


def test_the_wall_is_one_line_of_the_nozzle():
    """The wall where the slice prints its single line is exactly one nozzle wide, hollow inside
    and empty outside: a vase, not a solid rod."""
    m = made(labels=False)
    r = 0.5 * D
    assert probe(m, r - 0.5 * max_speed_test.NOZZLE, 0.0, 2.5) > 0.0, "the wall is not there"
    # 0.15 in, because the sixty-four flat faces of the mesh sag a hair under 19.6 mm between
    # their corners - a probe closer grazes the wall
    assert probe(m, r - max_speed_test.NOZZLE - 0.15, 0.0, 2.5) == \
        pytest.approx(0.0, abs=1e-9), "the inside of the wall is not hollow"
    assert probe(m, r + 0.05, 0.0, 2.5) == pytest.approx(0.0, abs=1e-9), \
        "something stands out of the wall where there is no boundary"
    assert probe(m, 0.0, 0.0, 2.5) == pytest.approx(0.0, abs=1e-9), "the tube is not open inside"


@pytest.mark.parametrize("z", (5.0, 10.0))
def test_there_is_a_rib_at_every_boundary_between_two_segments(z):
    """The three-segment tower has a rib at its two inner boundaries and none above the top."""
    m = made(labels=False)
    r = 0.5 * D + 0.5 * max_speed_test.RIB_OUT
    assert probe(m, r, 0.0, z, size=0.3) > 0.0, f"the boundary at {z} mm has no rib"
    assert probe(m, r, 0.0, 15.0, size=0.3) == pytest.approx(0.0, abs=1e-9), \
        "the top got a rib too - there is no segment above it"


@pytest.mark.parametrize("z", (2.5, 7.5, 12.5))
def test_between_the_boundaries_the_wall_stands_plain(z):
    m = made(labels=False)
    r = 0.5 * D + 0.5 * max_speed_test.RIB_OUT
    assert probe(m, r, 0.0, z, size=0.3) == pytest.approx(0.0, abs=1e-9), \
        f"something stands out at {z} mm, where no boundary is"


def test_the_rib_is_a_step_of_RIB_H_tall():
    m = made(labels=False)
    r = 0.5 * D + 0.5 * max_speed_test.RIB_OUT
    half = 0.5 * max_speed_test.RIB_H
    assert probe(m, r, 0.0, 5.0 - half + 0.05, size=0.08) > 0.0
    assert probe(m, r, 0.0, 5.0 + half - 0.05, size=0.08) > 0.0
    assert probe(m, r, 0.0, 5.0 - half - 0.05, size=0.08) == pytest.approx(0.0, abs=1e-9)
    assert probe(m, r, 0.0, 5.0 + half + 0.05, size=0.08) == pytest.approx(0.0, abs=1e-9)


def test_the_tab_stands_at_the_foot_and_only_there():
    m = made(labels=False)
    t = max_speed_test.TAB_T
    assert probe(m, 0.5 * D + 0.5 * max_speed_test.TAB_OUT, 0.0, 0.5 * t) > 0.0, \
        "the tab at the foot is not there"
    assert probe(m, 0.5 * D + 3.0, 0.0, t + 0.3, size=0.3) == pytest.approx(0.0, abs=1e-9), \
        "the tab stands higher than it is thick"


@pytest.mark.parametrize("labels", (False, True))
def test_the_default_walk_makes_nine_segments_of_forty_five_millimetres(labels):
    s = max_speed_test.sizes()
    assert s["segments"] == 9
    lo, hi = bbox(max_speed_test.max_speed_test(labels=labels))
    assert hi[2] - lo[2] == pytest.approx(45.0, abs=1e-3)
    assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=1e-3)
    assert hi[1] - lo[1] == pytest.approx(s["size"][1], abs=1e-3)


def test_the_tab_label_takes_plastic_out_of_the_tab():
    need_fonts()
    plain, made_m = made(labels=False), made(labels=True)
    assert made_m.volume() < plain.volume(), "the flows are not engraved in the tab"
    assert made_m.volume() > 0.98 * plain.volume(), "the numbers ate too much of the tab"


def test_the_tab_carries_the_first_and_the_last_flow():
    need_fonts()
    grooves = made(labels=False) - made(labels=True)
    lo, hi = bbox(grooves)
    assert lo[2] == pytest.approx(max_speed_test.TAB_T - max_speed_test.ENGRAVE, abs=1e-6)
    assert hi[2] == pytest.approx(max_speed_test.TAB_T, abs=1e-6), \
        "the engraving is not as deep as it is"
    assert lo[1] < -1.0 and hi[1] > 1.0, "both numbers should stand on the tab, apart from each "
    "other"
    # between the two numbers nothing is engraved: only the two end flows stand on the tab
    band = mf.Manifold.cube([20.0, 3.0, 2.0]).translate((0.5 * D + 0.2, -1.5, 0.0))
    assert (grooves ^ band).volume() == pytest.approx(0.0, abs=1e-9), \
        "a flow between the first and the last was engraved too"


def test_without_fonts_the_tab_carries_no_numbers(nofonts):
    m = made(labels=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.volume() == pytest.approx(made(labels=False).volume(), rel=1e-9)


# ------------------------------------------------------------------ the flows and the speeds

def test_the_default_walk_is_nine_flows_and_prints_both_ends():
    fl = max_speed_test.flows()
    assert fl == [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
    assert max_speed_test.sizes()["segments"] == 9
    assert max_speed_test.sizes()["size"] == (D + max_speed_test.TAB_OUT, D, 45.0)


def test_a_walk_down_gives_the_flows_highest_first():
    assert max_speed_test.flows(15.0, 5.0, 5.0) == [15.0, 10.0, 5.0]


@pytest.mark.parametrize("value, text", ((7.5, "7.5"), (5.0, "5"), (22.500000000000004, "22.5"),
                                         (-0.0, "0"), (12.25, "12.25")))
def test_a_flow_is_written_without_trailing_noise(value, text):
    assert max_speed_test.flow_text(value) == text


def test_the_speed_divides_the_flow_by_the_cross_section_of_the_wall():
    assert max_speed_test.speed(5.0, 0.2, 0.4) == pytest.approx(62.5)
    assert max_speed_test.speed(10.0, 0.2, 0.45) == pytest.approx(10.0 / 0.09)
    assert max_speed_test.speed(25.0, 0.3, 0.42) == pytest.approx(25.0 / (0.3 * 0.42))


def test_a_thick_wall_needs_less_speed_for_the_same_flow():
    """The same flow at a bigger cross-section is a slower line - that is the whole formula."""
    thin = max_speed_test.speed(12.0, 0.2, 0.4)
    thick = max_speed_test.speed(12.0, 0.3, 0.6)
    assert thin == pytest.approx(150.0) and thick == pytest.approx(66.6666667, rel=1e-6)
    assert thin > thick


def test_the_speeds_are_every_segment_in_order():
    assert max_speed_test.speeds([5.0, 10.0, 15.0], LH, LW) == \
        [pytest.approx(62.5), pytest.approx(125.0), pytest.approx(187.5)]
    assert max_speed_test.speeds([], LH, LW) == []


def test_the_percent_is_the_speed_as_a_fraction_of_the_base_speed():
    assert max_speed_test.BASE_SPEED == 100.0
    assert max_speed_test.percent(10.0, 0.2, 0.4) == 125       # 125 mm/s on a base of 100 mm/s
    assert max_speed_test.percent(2.0, 0.2, 0.4) == 25         # 24.999... that walks to 25
    assert max_speed_test.percents([5.0, 10.0], 0.2, 0.4) == [63, 125]


def test_a_half_is_rounded_up_and_noise_of_the_walk_cannot_underride_it():
    assert max_speed_test.speed(5.0, 0.2, 0.4) < 62.5, \
        "0.2 times 0.4 sits a hair over a full twentieth - that is the noise"
    assert max_speed_test.percent(5.0, 0.2, 0.4) == 63, "a half did not round up"
    assert max_speed_test.percent(4.0, 0.2, 0.4) == 50


def test_the_percent_never_stops_the_print():
    assert max_speed_test.percent(1.0, 0.8, 1.6) == 1, "M220 S0 would stop the extruder"


def test_the_command_is_an_M220_with_the_percent():
    assert max_speed_test.command(10.0, 0.2, 0.4) == "M220 S125"
    assert max_speed_test.command(5.0, 0.2, 0.4) == "M220 S63"


# ------------------------------------------------------------------ the plan

def test_steps_lists_the_flows_lowest_first_with_their_heights():
    plan = max_speed_test.steps(**SMALL)
    assert plan == [[0.0, 5.0], [5.0, 10.0], [10.0, 15.0]]


def test_the_default_plan_walks_up_to_the_end():
    plan = max_speed_test.steps()
    assert plan[0] == [0.0, 5.0] and plan[-1] == [40.0, 25.0]
    assert [z for z, _f in plan] == [i * SEG_H for i in range(9)]


def test_the_flows_of_the_plan_carry_no_noise_of_the_walk():
    for _z, f in max_speed_test.steps(5.0, 25.0, 2.5):
        assert f == round(f, 3)


def test_the_commands_become_the_percent_of_every_segment_including_the_lowest():
    cmds = max_speed_test.commands(max_speed_test.steps(**SMALL), 0.2, 0.4)
    assert cmds == [[0.0, "M220 S63"], [5.0, "M220 S125"], [10.0, "M220 S188"]]
    assert max_speed_test.commands([], 0.2, 0.4) == []


# ------------------------------------------------------------------ the presets

def test_every_preset_gives_a_sensible_tower():
    for name, (start, end, step) in max_speed_test.PRESETS.items():
        s = max_speed_test.sizes(start, end, step)
        assert max_speed_test.SEGMENTS_MIN <= s["segments"] <= max_speed_test.SEGMENTS_MAX, name
        assert max_speed_test.FLOW_MIN <= s["flows"][0] <= max_speed_test.FLOW_MAX, name
        assert max_speed_test.FLOW_MIN <= s["flows"][-1] <= max_speed_test.FLOW_MAX, name
        assert s["flows"][0] == start and s["flows"][-1] <= end, name
        m = max_speed_test.max_speed_test(start, end, step, labels=False)
        assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, name


def test_a_preset_walks_the_whole_range_it_names():
    for name, (start, end, step) in max_speed_test.PRESETS.items():
        s = max_speed_test.sizes(start, end, step)
        assert s["segments"] == int(round(abs(end - start) / step)) + 1, name
        assert s["size"][2] == s["segments"] * SEG_H, name


def test_a_preset_prints_within_the_speed_limit_at_the_assumed_slice():
    for name, (start, end, step) in max_speed_test.PRESETS.items():
        spouts = max_speed_test.speeds(max_speed_test.flows(start, end, step), LH, LW)
        assert max(spouts) <= max_speed_test.SPEED_MAX + 1e-9, name


def test_the_dialog_names_the_hotend_and_the_flow_it_melts():
    assert max_speed_test.preset_label("standard_hotend") == "Standard hotend 5-20"
    assert max_speed_test.preset_label("high_flow") == "High flow 10-40"
    assert max_speed_test.preset_label("unknown") == "unknown"
    assert max_speed_test.default_preset() == next(iter(max_speed_test.PRESETS))


# ------------------------------------------------------------------ the validation

@pytest.mark.parametrize("bad", (0.0, -2.5))
def test_a_step_of_zero_or_less_is_refused(bad):
    with pytest.raises(ValueError):
        max_speed_test.max_speed_test(5.0, 25.0, bad)
    with pytest.raises(ValueError):
        max_speed_test.flows(5.0, 25.0, bad)


def test_a_walk_that_gives_one_segment_is_refused():
    with pytest.raises(ValueError):
        max_speed_test.max_speed_test(10.0, 10.0, 2.5)         # one segment
    with pytest.raises(ValueError):
        max_speed_test.flows(10.0, 12.0, 2.5)                  # still one segment


def test_too_many_segments_is_refused():
    with pytest.raises(ValueError):
        max_speed_test.flows(1.0, 60.0, 2.8)                   # 22 segments
    with pytest.raises(ValueError):
        max_speed_test.sizes(1.0, 60.0, 2.8)
    with pytest.raises(ValueError):
        max_speed_test.max_speed_test(1.0, 60.0, 2.8)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "fast"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("start", "end", "step"):
        with pytest.raises(ValueError):
            max_speed_test.max_speed_test(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            max_speed_test.flows(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            max_speed_test.steps(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            max_speed_test.sizes(**dict(SMALL, **{key: bad}))


def test_the_flows_clamp_to_what_prints():
    assert max_speed_test.flows(0.0, 60.0, 30.0) == [1.0, 31.0]
    assert max_speed_test.flows(-5.0, 999.0, 25.0) == [1.0, 26.0, 51.0]
    m = max_speed_test.max_speed_test(0.0, 999.0, 29.5)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1


def test_a_speed_past_the_limit_is_refused_with_its_segment_named():
    # 45 mm³/s through 0.2 by 0.4 mm is 562.5 mm/s, past the 500 mm/s a tower may chase
    with pytest.raises(ValueError) as gone:
        max_speed_test.speeds([5.0, 45.0], 0.2, 0.4)
    assert "segment 2" in str(gone.value) and "562.5" in str(gone.value)
    # a plan that passes the limit on its way up names the first segment past it; 20 mm³/s
    # through 0.1 by 0.4 mm is exactly 500 mm/s, which is still allowed
    with pytest.raises(ValueError) as gone:
        max_speed_test.commands(max_speed_test.steps(5.0, 25.0, 2.5), 0.1, 0.4)
    assert "segment 8" in str(gone.value)
    with pytest.raises(ValueError):
        max_speed_test.percent(45.0, 0.2, 0.4)
    with pytest.raises(ValueError):
        max_speed_test.command(45.0, 0.2, 0.4)


def test_the_limit_is_the_last_speed_still_allowed():
    assert max_speed_test.speed(40.0, 0.2, 0.4) <= max_speed_test.SPEED_MAX + 1e-9
    assert max_speed_test.percent(40.0, 0.2, 0.4) == 500
    assert max_speed_test.speeds([40.0], 0.2, 0.4) == [pytest.approx(500.0)]


@pytest.mark.parametrize("layer, width", ((0, 0.4), (0.2, 0), (None, 0.4), (0.2, "wide"),
                                          (-0.2, 0.4)))
def test_a_measure_that_is_no_measure_is_refused(layer, width):
    with pytest.raises(ValueError):
        max_speed_test.speed(5.0, layer, width)
    with pytest.raises(ValueError):
        max_speed_test.commands([[0.0, 5.0]], layer, width)


# ------------------------------------------------------------------ the mesh settings

def _printer():
    return cura_profiles.Library(RESOURCES).printer("creality_ender3")


def _stacks():
    p = _printer()
    return p.stacks(p.default_variant(), p.default_material(), "standard", {})


def test_the_part_carries_its_single_wall_and_base_speed_through_a_file(tmp_path):
    doc = Document()
    g = ops.group_from_manifold(doc, made(labels=False), "#f2f2f2", "Max speed test")
    g.meta = {"speed_steps": max_speed_test.steps(**SMALL),
              "mesh_settings": max_speed_test.mesh_settings()}
    path = str(tmp_path / "ms.volura")
    fileio.save_document(doc, path)
    doc2 = Document()
    fileio.load_document(doc2, path)
    assert doc2.group(g.id).meta["speed_steps"] == max_speed_test.steps(**SMALL)
    assert doc2.group(g.id).meta["mesh_settings"] == max_speed_test.mesh_settings()


def test_a_part_reads_its_plan_and_settings_back_from_its_meta():
    doc = Document()
    g = ops.group_from_manifold(doc, made(labels=False), "#f2f2f2", "Max speed test")
    g.meta = {"mesh_settings": max_speed_test.mesh_settings()}
    parts = ops.parts(doc)
    assert parts[0].mesh_settings == max_speed_test.mesh_settings()


def test_the_single_wall_and_the_base_speed_reach_the_engine_as_mesh_settings():
    st = _stacks()
    got = slicer.per_mesh_settings(st, max_speed_test.mesh_settings())
    assert got["speed_print"] == max_speed_test.BASE_SPEED
    assert got["speed_wall"] == max_speed_test.BASE_SPEED
    assert got["speed_wall_0"] == max_speed_test.BASE_SPEED
    assert got["speed_topbottom"] == max_speed_test.BASE_SPEED
    assert got["wall_line_count"] == 1
    assert got["infill_sparse_density"] == 0.0
    assert got["top_layers"] == 0


def test_vase_mode_is_no_setting_a_mesh_can_ask_for():
    """`magic_spiralize` is global, so the test spirals by other means; a part that asks for it
    anyway is dropped like any other global-only key."""
    st = _stacks()
    assert not st.printer.definition.settings["magic_spiralize"].get("settable_per_mesh")
    assert slicer.per_mesh_settings(st, {"magic_spiralize": True}) == {}


# ------------------------------------------------------------------ the commands in the G-code

def _gcode(zs):
    """A small G-code with one layer per height in `zs`, each with a Z move and an extrusion."""
    return "".join(f";LAYER:{i}\nG1 Z{z:.2f} F900\nG1 X10 E{i + 1}\n" for i, z in enumerate(zs))


def test_with_commands_writes_every_percent_of_the_default_plan():
    text = _gcode([0.2 * k for k in range(1, 301)])
    plan = max_speed_test.commands(max_speed_test.steps(), LH, LW)
    out = slicer.with_commands(text, plan)
    lines = out.splitlines()
    for _z, line in plan:
        assert lines.count(line) == 1, line
    assert out.count("M220 S") == len(plan) == 9


def test_the_right_number_of_percents_lands_on_the_right_layers():
    zs = [0.2 * k for k in range(1, 301)]
    text = _gcode(zs)
    plan = max_speed_test.commands(max_speed_test.steps(5.0, 25.0, 2.5), LH, LW)
    out = slicer.with_commands(text, plan)
    lines = out.splitlines()
    heights = slicer.layer_heights(out)
    for z_from, line in plan:
        assert line in lines, line
        k = lines.index(line)
        assert lines[k - 1].startswith(";LAYER:"), "the change is at a layer boundary"
        at = next(h for ln, h in heights if ln > k - 1)
        assert at >= z_from - 1e-6, f"{line} is before the layer that reaches {z_from} mm"
        prev = [h for ln, h in heights if ln < k - 1]
        assert not prev or prev[-1] < z_from, f"{line} came too early"


def test_the_percent_of_the_lowest_segment_goes_in_at_the_first_layer():
    """The printer may sit on another speed factor, so the first layer already sets the start."""
    text = _gcode([0.2, 0.4, 0.6])
    plan = max_speed_test.commands(max_speed_test.steps(**SMALL), LH, LW)
    out = slicer.with_commands(text, plan).splitlines()
    assert out[1] == "M220 S63", "the first layer does not set the flow of the lowest segment"


def test_a_command_above_the_print_and_an_empty_plan_do_nothing():
    text = _gcode([0.2, 0.4, 0.6])
    assert slicer.with_commands(text, [[9.0, "M220 S100"]]) == text
    assert slicer.with_commands(text, []) == text


def test_the_other_tests_still_work_alongside_the_percents():
    """M104, SET_PRESSURE_ADVANCE and M220 on one bed all come in height order."""
    text = _gcode([0.2 * k for k in range(1, 401)])
    both = [(z, f"M104 S{t}") for z, t in temperature_tower.steps(220.0, 190.0, 5.0)[1:]]
    both += pressure_advance_test.commands("marlin", pressure_advance_test.steps(0.0, 0.03, 0.01))
    both += max_speed_test.commands(max_speed_test.steps(**SMALL), LH, LW)
    out = slicer.with_commands(text, sorted(both, key=lambda one: one[0]))
    assert out.count("M104 S") == 6 and out.count("M900 K") == 3 and out.count("M220 S") == 3
    lines = out.splitlines()
    want = ["M900 K0", "M220 S63",              # at 0 mm: both tests set their lowest segment
            "M900 K0.01", "M220 S125",          # at 5 mm
            "M900 K0.02", "M220 S188",          # at 10 mm
            "M104 S215", "M104 S190"]           # at 20 and 70 mm: the tower, whose first switch
    # stands a BASE plate above the bed, ends the order
    at = [k for k, ln in enumerate(lines) if ln in want]
    assert [lines[k] for k in at] == want, "the three kinds did not come in height order"


def test_commands_are_written_into_the_g_code_file(tmp_path):
    path = tmp_path / "x.gcode"
    path.write_text(_gcode([0.2, 0.4, 0.6, 0.8, 1.0]), encoding="utf-8")
    slicer.insert_commands(str(path), [[0.5, "M220 S125"]])
    assert path.read_text(encoding="utf-8").count("M220 S125") == 1


# -------------------------------------------------------------------- the real slice

def _wall_feedrates(text, below) -> list:
    """The F words (mm/min) of the wall extrusions below `below` mm, skipping the first layer:
    CuraEngine scales a move's F down for corners and short lines, so the nominal speed is what
    the fastest wall moves run at - and they must run at the part's own base speed."""
    out, t, z = [], None, None
    for line in text.splitlines():
        if line.startswith(";TYPE:"):
            t = line[6:].strip().upper()
            continue
        if line.startswith(";") or not line.strip() or line[0] not in "G":
            continue
        mz = re.search(r"\bZ(-?\d*\.?\d+)", line)
        if mz:
            z = float(mz.group(1))
        if not re.search(r"\bE-?\d*\.?\d", line) or not re.search(r"\b[XY]", line):
            continue
        mf_ = re.search(r"\bF(\d*\.?\d+)", line)
        if mf_ and z is not None and 0.21 < z < below and t and t.startswith("WALL"):
            out.append(float(mf_.group(1)))
    return out


@needs_engine
def test_a_max_speed_test_slices_and_speeds_up_along_the_way(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    st = p.stacks(p.default_variant(), p.default_material(), "standard", {})
    g = st.global_settings()
    lh, lw = float(g["layer_height"]), float(g["line_width"])
    assert float(g["speed_wall"]) < 0.9 * max_speed_test.BASE_SPEED, \
        "the print's own wall speed should be far from the part's, or this proves nothing"
    plan = max_speed_test.steps(5.0, 15.0, 5.0)
    cmds = max_speed_test.commands(plan, lh, lw)
    mesh = max_speed_test.max_speed_test(5.0, 15.0, 5.0, labels=False).to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3] + np.array([40.0, 40.0, 0.0])
    tris = np.asarray(mesh.tri_verts)
    own = slicer.per_mesh_settings(st, max_speed_test.mesh_settings())
    out = tmp_path / "ms.gcode"
    slicer.slice_meshes(st, [(verts, tris, 0, own)], str(out))
    slicer.insert_commands(str(out), cmds)

    text = out.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    want = [line for _z, line in cmds]
    assert want == ["M220 S63", "M220 S125", "M220 S188"]
    at = []
    for line in want:
        k = [i for i, ln in enumerate(lines) if ln == line]
        assert k, f"{line} is not in the G-code"
        at.append(k[-1])
    assert at == sorted(at), "the percents did not come in order"
    heights = slicer.layer_heights(text)
    for (z_from, _line), k in zip(cmds, at):
        first_after = next(h for ln, h in heights if ln >= k)
        assert first_after >= z_from - 0.21, f"the change at line {k} is before {z_from} mm"

    # one wall of one line and nothing inside: the part really sliced as a single-walled vase
    assert "WALL-INNER" not in text, "the part did not slice with a single wall line"
    assert not re.search(r"^;TYPE:.*INFILL.*$", text, re.M), "the part did not slice without infill"
    # the first segment's walls run at the part's own base speed: F = speed * 60, scaled down by
    # the engine only for corners and short lines - the fastest wall moves say the nominal
    fps = _wall_feedrates(text, below=SEG_H - 0.01)
    assert len(fps) > 50, "too few wall moves to read the speed from"
    assert max(fps) == pytest.approx(max_speed_test.BASE_SPEED * 60.0, rel=0.02), \
        f"the base speed did not reach the engine (fastest wall F: {max(fps):g})"


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The max speed test settings back at what the program starts with."""
    for key, value in SETTINGS:
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_fills_the_fields_and_remembers_them(win, monkeypatch):
    seen = {}

    def accept(dlg):
        preset_cb = dlg.findChildren(QComboBox)[0]
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["default"] = [preset_cb.currentData(), spins[0].value(), spins[1].value(),
                           spins[2].value()]
        preset_cb.setCurrentIndex(preset_cb.findData("high_flow"))
        seen["preset"] = [spins[0].value(), spins[1].value(), spins[2].value()]
        spins[1].setValue(15.0)
        spins[2].setValue(5.0)
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.max_speed_test_dialog()
        assert seen["default"] == ["standard_hotend", pytest.approx(5.0), pytest.approx(25.0),
                                   pytest.approx(2.5)]
        assert seen["preset"] == [pytest.approx(10.0), pytest.approx(40.0), pytest.approx(5.0)]
        assert "Segment 1" in seen["text"] and "Segment 2" in seen["text"], "the dialog does not \
            say what prints"
        assert pump_until(lambda: len(win.doc.groups) == 1), "no max speed test"
        g = win.doc.groups[0]
        assert g.name == "Max speed test"
        assert g.meta["speed_steps"] == max_speed_test.steps(10.0, 15.0, 5.0)
        assert g.meta["mesh_settings"] == max_speed_test.mesh_settings()
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert pump_until(lambda: "ribs" in win.hint.text().lower()), win.hint.text()
        st = app_module.Settings()
        assert st.max_speed_test_preset == "high_flow"
        assert st.max_speed_test_start == pytest.approx(10.0)
        assert st.max_speed_test_end == pytest.approx(15.0)
        assert st.max_speed_test_step == pytest.approx(5.0)
    finally:
        forget_the_values(win.settings)


def test_the_summary_names_every_segment_with_its_flow_and_speed(win, monkeypatch):
    seen = {}

    def accept(dlg):
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.max_speed_test_dialog()
    text = seen["text"]
    assert text.count("Segment ") == 9, text
    assert "Segment 1: 5 mm³/s = 62 mm/s" in text, text
    assert "Segment 9" in text and "25 mm³/s" in text, text
    assert "Speeds are for 0.2 mm layers and 0.4 mm lines" in text, \
        "the dialog should say which slice the speeds belong to"


def test_the_dialog_refuses_a_speed_past_the_limit(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(55.0)
        spins[1].setValue(60.0)
        spins[2].setValue(5.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.max_speed_test_dialog()
    assert win.doc.groups == [], "a tower past the speed limit was built anyway"
    assert any("segment 1" in text.lower() for _title, text in asked), asked


def test_the_dialog_says_what_is_wrong_while_the_values_change(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))       # an end equal to the start
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.max_speed_test_dialog()
    assert "at least" in seen["text"].lower(), seen["text"]


def test_the_preview_falls_back_on_the_assumed_measures_without_a_printer(win):
    assert win._preview_layer_and_width() == (max_speed_test.DEFAULT_LAYER_HEIGHT,
                                              max_speed_test.DEFAULT_LINE_WIDTH)


def test_the_preview_reads_the_last_slice_choice(win, monkeypatch):
    real = (0.2, 0.4)
    win.settings.slice_printer = "creality_ender3"
    win.settings.slice_quality = "standard"
    win.settings.slice_variant = ""
    win.settings.slice_material = ""

    class FakeLib:
        def printer(self, did):
            p = _printer()
            assert did == "creality_ender3"
            return p

    monkeypatch.setattr("volura.slice_dialog.is_loaded", lambda res: True)
    monkeypatch.setattr("volura.slice_dialog.library", lambda res: FakeLib())
    assert win._preview_layer_and_width() == pytest.approx(real)


def test_the_preview_falls_back_when_Settings_remember_no_printer(win, monkeypatch):
    win.settings.slice_printer = "de_vliegende_theepot"
    monkeypatch.setattr("volura.slice_dialog.is_loaded", lambda res: True)
    monkeypatch.setattr("volura.slice_dialog.library", lambda res: (
        cura_profiles.Library(RESOURCES)))
    assert win._preview_layer_and_width() == (max_speed_test.DEFAULT_LAYER_HEIGHT,
                                              max_speed_test.DEFAULT_LINE_WIDTH)


def test_one_undo_takes_the_whole_tower_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the tower is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_slice_turns_the_flows_into_percents_with_its_stacks(win):
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    want = max_speed_test.steps(**SMALL)
    assert win.doc.groups[0].meta["speed_steps"] == want
    # a slice of 0.3 by 0.5 mm: 5, 10 and 15 mm³/s mean 33.3, 66.7 and 100 mm/s
    result = win._bed_height_commands(only=False, stacks=StubStacks(0.3, 0.5))
    assert result is not None
    print_temp, changes = result
    assert print_temp is None, "a max speed test says nothing about the temperature"
    assert changes == [(0.0, "M220 S33"), (5.0, "M220 S67"), (10.0, "M220 S100")]


def test_the_bed_uses_the_assumed_measures_when_no_stacks_are_known(win):
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    changes = win._bed_height_commands(only=False)[1]
    assert changes == [(0.0, "M220 S63"), (5.0, "M220 S125"), (10.0, "M220 S188")]


def test_a_slice_that_would_run_past_the_limit_is_refused(win):
    """A tower made for a thin slice cannot be sent through a thick one at the same flows."""
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    with pytest.raises(ValueError) as gone:
        # through a line 0.09 mm wide, already 10 mm³/s would mean 556 mm/s
        win._bed_height_commands(only=False, stacks=StubStacks(0.2, 0.09))
    assert "segment 2" in str(gone.value), gone.value


def test_the_bed_has_no_commands_without_a_test(win):
    assert win._bed_height_commands(only=False) is None


def test_the_bed_writes_every_kind_of_command(win):
    """A temperature tower, a pressure advance test and a max speed test on one bed give all
    three kinds of changes, in height order."""
    win._apply_temperature_tower(220.0, 190.0, 5.0, temperature_tower.steps(220.0, 190.0, 5.0))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    win._apply_pressure_advance_test("klipper", 0.0, 0.03, 0.01,
                                     pressure_advance_test.steps(0.0, 0.03, 0.01))
    assert pump_until(lambda: len(win.doc.groups) == 2)
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 3)
    print_temp, changes = win._bed_height_commands(only=False, stacks=StubStacks())
    assert print_temp == 220
    temps = [line for _z, line in changes if line.startswith("M104")]
    advance = [line for _z, line in changes if line.startswith("SET_PRESSURE_ADVANCE")]
    percents = [(z, line) for z, line in changes if line.startswith("M220")]
    assert len(temps) == 6 and len(advance) == 3 and len(percents) == 3
    assert percents == [(0.0, "M220 S63"), (5.0, "M220 S125"), (10.0, "M220 S188")]
    assert changes == sorted(changes, key=lambda one: one[0]), "the changes are not in height order"


def test_a_meta_that_is_no_flow_is_ignored(win):
    win._apply_max_speed_test(5.0, 15.0, 5.0, max_speed_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    g = win.doc.groups[0]
    g.meta["speed_steps"] = [["geen flow"], {"niet": "dit"}, 5, None, [1.0, "x"],
                             [15.0, 20.0]]
    changes = win._bed_height_commands(only=False, stacks=StubStacks())[1]
    assert changes == [(15.0, "M220 S250")], "a junk entry was written along with the real ones"


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("max_speed_test_preset", "max_speed_test_start", "max_speed_test_end",
            "max_speed_test_step")
    saved = {k: q.value(k) for k in keys}
    try:
        q.setValue("max_speed_test_preset", "KAPUT")
        q.setValue("max_speed_test_start", 999.0)
        q.setValue("max_speed_test_end", -999.0)
        q.setValue("max_speed_test_step", 999.0)
        q.sync()
        s = app_module.Settings()
        assert s.max_speed_test_preset == "", "a preset Volura does not know comes back"
        assert s.max_speed_test_start == 60.0
        assert s.max_speed_test_end == 1.0
        assert s.max_speed_test_step == 30.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_flow_rate_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Max Speed Test") == labels.index("Flow Rate Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
