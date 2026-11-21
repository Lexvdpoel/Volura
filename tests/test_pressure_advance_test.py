"""Pressure Advance Test: a hollow tower whose segments print with a rising pressure advance value.

`pressure_advance_test.pressure_advance_test` gives the tower as one closed volume standing on `z = 0`:
a square box of `FOOT` by `FOOT` millimetres with sharp corners and walls of `WALL_LINES` lines of the
nozzle, hollow above a floor of `FLOOR` millimetres and without a top, `segments` segments of `SEG_H`
millimetres high with the value of every segment engraved `ENGRAVE` millimetres deep in the middle of
its own front wall. `pressure_advance_test.steps` gives the plan of values the printer has to follow and
`commands` the command line every one of them means: `SET_PRESSURE_ADVANCE ADVANCE=…` for Klipper,
`M900 K…` for Marlin and `M572 D0 S…` for RepRapFirmware. Those commands go on the part itself, in
`g.meta["height_commands"]`, and the generalised `slicer.with_commands` writes them at the first layer
at or above their height - the same mechanism that asks a temperature tower's `M104` lines, so a bed
with both writes both.

These tests (Qt offscreen) weigh that: one closed volume on the bed, `segments * SEG_H` tall, `FOOT`
across with sharp corners, hollow above its floor with walls of two lines of the nozzle and no top; the
numbers engraved shallow and not through the wall, in the middle of their own band, and gone when this
system has no fonts; the plan of values, the presets per firmware and the syntax of every firmware; the
validation (a step of 0, an end equal to the start, too few or too many segments, a number that is no
number); the commands round-tripping through a file; `with_commands` writing exactly `segments`
commands in order at the right layers; the window (the firmware a Klipper printer on the network asks
for, the preset that fills the values, both kinds of commands on a bed that carries both tests) and,
when CuraEngine is there, a real slice that sets every value in turn.

Run with:  python -m pytest tests/test_pressure_advance_test.py
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
                    ops, pressure_advance_test, printer_link, slicer, temperature_tower,
                    window_print)
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: the numbers the tests build with: three segments of 0.01, the lowest printing without advance
SMALL = dict(start=0.0, end=0.03, step=0.01)
BASE = pressure_advance_test.FLOOR
SEG_H = pressure_advance_test.SEG_H
FOOT = pressure_advance_test.FOOT
WALL = pressure_advance_test.WALL
#: the box is exact polygons, so a part is at most this far off
ABS = 1e-4
#: the firmware with the command it writes, in the order of the dialog's box
SYNTAX = {"klipper": "SET_PRESSURE_ADVANCE ADVANCE={v}", "marlin": "M900 K{v}",
          "rrf": "M572 D0 S{v}"}
#: the settings of the dialog, back at what the program starts with
SETTINGS = (("pressure_advance_firmware", ""), ("pressure_advance_preset", ""),
            ("pressure_advance_start", 0.0), ("pressure_advance_end", 0.1),
            ("pressure_advance_step", 0.005))


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
    for key, value in SETTINGS:
        setattr(w.settings, key, value)
    w.settings.printers = []          # what stands under Preferences is up to the test
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def nofonts(monkeypatch):
    """No fonts: the tower is then exactly the box, with no numbers in its walls."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no numbers to test")


def made(**kw) -> mf.Manifold:
    return pressure_advance_test.pressure_advance_test(**dict(SMALL, **kw))


def numbers(**kw) -> dict:
    return pressure_advance_test.sizes(**dict(SMALL, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def probe(m, x, y, z, size=0.1) -> float:
    """How much of a little cube around `x, y, z` is plastic."""
    box = mf.Manifold.cube([size, size, size]).translate((x - 0.5 * size, y - 0.5 * size,
                                                          z - 0.5 * size))
    return (m ^ box).volume()


def front_band(m, z_from, z_to) -> float:
    """The plastic of the front wall between two heights, over the whole width of the tower."""
    box = mf.Manifold.cube([FOOT + 2.0, WALL + 1.0, z_to - z_from]) \
        .translate((-0.5 * (FOOT + 2.0), -0.5 * FOOT - 1.0, z_from))
    return (m ^ box).volume()


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the tower

@pytest.mark.parametrize("labels", (False, True))
def test_the_tower_is_one_volume_that_stands_on_the_bed_and_is_centred(labels):
    s = numbers()
    m = made(labels=labels)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the tower fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the tower does not stand on the bed"
    assert lo[0] == pytest.approx(-0.5 * FOOT, abs=ABS) and hi[0] == pytest.approx(0.5 * FOOT,
                                                                                   abs=ABS)
    assert lo[1] == pytest.approx(-0.5 * FOOT, abs=ABS) and hi[1] == pytest.approx(0.5 * FOOT,
                                                                                   abs=ABS)
    assert hi[2] - lo[2] == pytest.approx(s["segments"] * SEG_H, abs=1e-3), "not as tall as asked"


def test_the_tower_is_taller_for_every_segment_it_has():
    for step in (0.01, 0.02):
        n = pressure_advance_test.sizes(0.0, 0.1, step)["segments"]
        lo, hi = bbox(pressure_advance_test.pressure_advance_test(0.0, 0.1, step))
        assert hi[2] - lo[2] == pytest.approx(n * SEG_H, abs=1e-3)


def test_the_sharp_corners_make_a_square_of_FOOT_by_FOOT():
    """Nothing leans out or stands out: the corners are corners and the box is a square."""
    lo, hi = bbox(made(labels=False))
    assert hi[0] - lo[0] == pytest.approx(FOOT, abs=ABS)
    assert hi[1] - lo[1] == pytest.approx(FOOT, abs=ABS)
    # a corner of a wall is plastic, the air beside the corner is empty
    assert probe(made(labels=False), -0.5 * FOOT + 0.1, -0.5 * FOOT + 0.1, 2.5) > 0.0
    assert probe(made(labels=False), -0.5 * FOOT - 0.2, -0.5 * FOOT - 0.2, 2.5) == \
        pytest.approx(0.0, abs=1e-9), "something stands out beside a corner"


def test_the_box_is_hollow_above_its_floor_and_has_no_top():
    m = made(labels=False)
    top = numbers()["size"][2]
    assert probe(m, 0.0, 0.0, 0.5 * BASE) > 0.0, "the floor is not there"
    assert probe(m, 0.0, 0.0, BASE + 0.5) == pytest.approx(0.0, abs=1e-9), "the box is not hollow"
    assert probe(m, 0.0, 0.0, top - 0.1) == pytest.approx(0.0, abs=1e-9), "the box got a top"


def test_the_wall_is_two_lines_of_the_nozzle_wide():
    """The plastic is what a wall of `WALL_LINES` lines and a floor of `FLOOR` makes, no more."""
    s, m = numbers(), made(labels=False)
    top = s["segments"] * SEG_H
    want = FOOT * FOOT * top - (FOOT - 2.0 * WALL) ** 2 * (top - BASE)
    assert m.volume() == pytest.approx(want, rel=1e-6), "the wall is not two lines of the nozzle"
    y0 = -0.5 * FOOT                                  # the outside of the front wall
    assert probe(m, 10.0, y0 + 0.5 * WALL, 2.5) > 0.0, "the wall is not where it should be"
    assert probe(m, 10.0, y0 + WALL + 0.3, 2.5) == pytest.approx(0.0, abs=1e-9), \
        "the inside of the wall is not where the hollow starts"


def test_the_wall_behind_the_numbers_stays_whole():
    """A wall of two lines keeps more than one line behind an engraving of 0.3 mm."""
    need_fonts()
    m, n = made(), numbers()["segments"]
    worst = min(probe(m, x, -0.5 * FOOT + WALL - 0.15, i * SEG_H + 0.5 * SEG_H, size=0.3)
                for i in range(n) for x in np.arange(-8.0, 8.01, 0.5))
    assert worst == pytest.approx(0.3 ** 3, abs=1e-6), "an engraving cut the wall open"


def test_the_numbers_are_engraved_shallow_in_stead_of_stood_on():
    need_fonts()
    plain, made_m = made(labels=False), made(labels=True)
    assert made_m.volume() < plain.volume(), "the numbers are not cut into the wall"
    assert made_m.volume() > 0.99 * plain.volume(), "the numbers ate too much of the wall"
    grooves = plain - made_m
    lo, hi = bbox(grooves)
    assert lo[1] == pytest.approx(-0.5 * FOOT, abs=1e-6), "the engraving starts behind the face"
    assert hi[1] == pytest.approx(-0.5 * FOOT + pressure_advance_test.ENGRAVE, abs=1e-6), \
        "the engraving is not as deep as it is"
    assert hi[2] - lo[2] <= SEG_H * numbers()["segments"], "the numbers run past the tower"


def test_every_segment_carries_its_own_number():
    need_fonts()
    m, plain, n = made(), made(labels=False), numbers()["segments"]
    for i in range(n):
        z = i * SEG_H
        assert front_band(m, z + 0.5 * SEG_H - 0.4, z + 0.5 * SEG_H + 0.4) < \
            front_band(plain, z + 0.5 * SEG_H - 0.4, z + 0.5 * SEG_H + 0.4), \
            f"segment {i} carries no number"
    for i in range(1, n):
        assert front_band(m, i * SEG_H - 0.4, i * SEG_H + 0.4) == \
            pytest.approx(front_band(plain, i * SEG_H - 0.4, i * SEG_H + 0.4), abs=1e-6), \
            f"the numbers of segment {i} ran into the wall above it"


def test_the_engraving_leaves_more_than_one_line_of_wall():
    """The numbers are engraved in a wall of two lines: that only works when it leaves a wall."""
    assert pressure_advance_test.ENGRAVE + pressure_advance_test.ROOM <= WALL
    assert pressure_advance_test.CAP < SEG_H


def test_without_fonts_the_walls_carry_no_numbers(nofonts):
    m = made(labels=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.volume() == pytest.approx(made(labels=False).volume(), rel=1e-9)


# ------------------------------------------------------------------ the plan

def test_steps_lists_the_values_lowest_first_with_their_heights():
    plan = pressure_advance_test.steps(**SMALL)
    assert [v for _z, v in plan] == [0.0, 0.01, 0.02]
    assert [z for z, _v in plan] == [0.0, SEG_H, 2.0 * SEG_H]


def test_a_range_that_walks_down_works_just_the_same():
    plan = pressure_advance_test.steps(0.02, 0.0, 0.01)
    assert [v for _z, v in plan] == [0.02, 0.01]


def test_the_values_carry_no_noise_of_the_walk():
    plan = pressure_advance_test.steps(0.0, 0.1, 0.005)
    assert [v for _z, v in plan][:4] == [0.0, 0.005, 0.01, 0.015]
    assert plan[-1][1] == 0.095
    for _z, v in plan:
        assert v == round(v, 6)


def test_the_lowest_value_is_the_start_and_the_top_one_stays_under_the_end():
    s = numbers()
    assert s["values"][0] == SMALL["start"]
    assert s["values"][-1] < SMALL["end"]
    assert s["size"] == (FOOT, FOOT, s["segments"] * SEG_H)


# ------------------------------------------------------- the firmware and its commands

@pytest.mark.parametrize("firmware", tuple(SYNTAX))
def test_every_firmware_writes_its_own_command(firmware):
    assert pressure_advance_test.command(firmware, 0.005) == SYNTAX[firmware].format(v="0.005")
    assert pressure_advance_test.command(firmware, 0.0) == SYNTAX[firmware].format(v="0")


@pytest.mark.parametrize("value, text", ((0.0, "0"), (1.0, "1"), (0.035000000000000003, "0.035"),
                                        (0.005, "0.005"), (1.25, "1.25"), (-0.0, "0"),
                                        (-1e-17, "0"), (5.0, "5")))
def test_a_value_is_written_without_trailing_noise(value, text):
    assert pressure_advance_test.value_text(value) == text
    assert pressure_advance_test.command("klipper", value) == f"SET_PRESSURE_ADVANCE ADVANCE={text}"


def test_a_firmware_volura_does_not_know_is_refused():
    with pytest.raises(ValueError):
        pressure_advance_test.command("repetier", 0.01)
    with pytest.raises(ValueError):
        pressure_advance_test.commands("repetier", pressure_advance_test.steps(**SMALL))
    assert pressure_advance_test.commands("repetier", []) == [], "an empty plan needs no firmware"


def test_the_commands_name_the_value_of_every_segment_including_the_lowest():
    cmds = pressure_advance_test.commands("klipper", pressure_advance_test.steps(**SMALL))
    assert cmds == [[0.0, "SET_PRESSURE_ADVANCE ADVANCE=0"],
                    [5.0, "SET_PRESSURE_ADVANCE ADVANCE=0.01"],
                    [10.0, "SET_PRESSURE_ADVANCE ADVANCE=0.02"]]
    assert pressure_advance_test.commands("marlin", []) == []


def test_a_klipper_printer_on_the_network_says_klipper():
    assert pressure_advance_test.default_firmware() == "marlin"
    assert pressure_advance_test.default_firmware([]) == "marlin"
    assert pressure_advance_test.default_firmware(None) == "marlin"
    klipper = printer_link.Connection("Zero", "moonraker", "http://zero.local")
    octoprint = printer_link.Connection("Ender", "octoprint", "http://ender.local")
    assert pressure_advance_test.default_firmware([octoprint]) == "marlin"
    assert pressure_advance_test.default_firmware([octoprint, klipper]) == "klipper"
    assert pressure_advance_test.default_firmware([None, "zero"]) == "marlin", \
        "a list of things that are no printers should not stop the dialog"


def test_the_preset_of_a_firmware_is_its_direct_drive_one():
    assert pressure_advance_test.default_preset("klipper") == "klipper_direct"
    assert pressure_advance_test.default_preset("marlin") == "marlin_direct"
    assert pressure_advance_test.default_preset("rrf") == "rrf"
    assert pressure_advance_test.default_preset("unknown") == next(iter(pressure_advance_test.PRESETS))


def test_the_dialog_names_a_firmware_and_the_drive_of_its_extruder():
    assert pressure_advance_test.firmware_label("klipper") == "Klipper"
    assert pressure_advance_test.firmware_label("unknown") == "unknown"
    assert pressure_advance_test.preset_label("klipper_bowden").startswith("Klipper")
    assert pressure_advance_test.preset_label("unknown") == "unknown"
    for name, (fw, _a, _b, _d) in pressure_advance_test.PRESETS.items():
        assert pressure_advance_test.preset_label(name).startswith(
            pressure_advance_test.firmware_label(fw)), name


# ------------------------------------------------------------------ the presets

def test_every_preset_gives_a_sensible_tower():
    for name, (fw, start, end, step) in pressure_advance_test.PRESETS.items():
        s = pressure_advance_test.sizes(start, end, step)
        assert pressure_advance_test.SEGMENTS_MIN <= s["segments"] <= \
            pressure_advance_test.SEGMENTS_MAX, name
        assert s["values"][0] == start, name
        assert s["values"][-1] <= end, name
        assert pressure_advance_test.command(fw, s["values"][-1]).startswith(
            pressure_advance_test.FIRMWARES[fw][0].split("{")[0]), name
        m = pressure_advance_test.pressure_advance_test(start, end, step, labels=False)
        assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, name


def test_a_preset_walks_the_whole_range_it_names():
    for name, (_fw, start, end, step) in pressure_advance_test.PRESETS.items():
        s = pressure_advance_test.sizes(start, end, step)
        assert s["segments"] == int(round(abs(end - start) / step)), name
        assert s["size"][2] == s["segments"] * SEG_H, name


# ------------------------------------------------------------------ the validation

@pytest.mark.parametrize("bad", (0.0, -0.005))
def test_a_step_of_zero_or_less_is_refused(bad):
    with pytest.raises(ValueError):
        pressure_advance_test.pressure_advance_test(0.0, 0.1, bad)


def test_an_end_equal_to_the_start_is_refused():
    with pytest.raises(ValueError):
        pressure_advance_test.pressure_advance_test(0.05, 0.05, 0.005)


def test_too_few_segments_is_refused():
    with pytest.raises(ValueError):
        pressure_advance_test.pressure_advance_test(0.0, 0.005, 0.005)      # one segment


def test_too_many_segments_is_refused():
    with pytest.raises(ValueError):
        pressure_advance_test.pressure_advance_test(0.0, 0.1, 0.004)        # 25 segments
    with pytest.raises(ValueError):
        pressure_advance_test.sizes(0.0, 5.0, 0.001)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "fast"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("start", "end", "step"):
        with pytest.raises(ValueError):
            pressure_advance_test.pressure_advance_test(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            pressure_advance_test.sizes(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            pressure_advance_test.steps(**dict(SMALL, **{key: bad}))


def test_the_values_are_clamped_to_what_a_firmware_takes():
    s = pressure_advance_test.sizes(-5.0, 99.0, 1.0)
    assert s["values"][0] == pressure_advance_test.VALUE_MIN
    assert s["values"][-1] == 4.0 and s["segments"] == 5
    m = pressure_advance_test.pressure_advance_test(0.0, 99.0, 1.0)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1


# ------------------------------------------------------------------ the round-trip

def test_the_commands_round_trip_through_a_file(tmp_path):
    doc = Document()
    cmds = pressure_advance_test.commands("klipper", pressure_advance_test.steps(**SMALL))
    g = ops.group_from_manifold(doc, made(labels=False), "#f2f2f2", "Pressure advance test")
    g.meta = {"height_commands": cmds}
    path = str(tmp_path / "pa.volura")
    fileio.save_document(doc, path)
    doc2 = Document()
    fileio.load_document(doc2, path)
    assert doc2.group(g.id).meta["height_commands"] == cmds


# ------------------------------------------------------------------ the commands in the G-code

def _gcode(zs):
    """A small G-code with one layer per height in `zs`, each with a Z move and an extrusion."""
    return "".join(f";LAYER:{i}\nG1 Z{z:.2f} F900\nG1 X10 E{i + 1}\n" for i, z in enumerate(zs))


@pytest.mark.parametrize("firmware", tuple(SYNTAX))
def test_with_commands_writes_the_syntax_of_a_firmware(firmware):
    text = _gcode([0.2 * k for k in range(1, 101)])
    plan = pressure_advance_test.commands(firmware, pressure_advance_test.steps(**SMALL))
    out = slicer.with_commands(text, plan)
    lines = out.splitlines()
    for _z, line in plan:
        assert lines.count(line) == 1, line    # "=0" starts "=0.01", so count whole lines only
    assert out.count("\n") - text.count("\n") == len(plan)


def test_the_right_number_of_commands_lands_on_the_right_layers():
    """A plan of twenty segments inserts twenty commands, each on the first layer past its height."""
    zs = [0.2 * k for k in range(1, 601)]          # 0.2 mm layers up to 120 mm
    text = _gcode(zs)
    fw, start, end, step = pressure_advance_test.PRESETS["klipper_direct"]
    plan = pressure_advance_test.commands(fw, pressure_advance_test.steps(start, end, step))
    out = slicer.with_commands(text, plan)
    assert out.count("SET_PRESSURE_ADVANCE") == len(plan) == pressure_advance_test.SEGMENTS_MAX
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


def test_the_value_of_the_lowest_segment_goes_in_at_the_first_layer():
    """The printer may sit on another value, so the first layer already gets the start of the test."""
    text = _gcode([0.2, 0.4, 0.6])
    plan = pressure_advance_test.commands("marlin", pressure_advance_test.steps(**SMALL))
    out = slicer.with_commands(text, plan).splitlines()
    assert out[1] == "M900 K0", "the first layer does not set the value of the lowest segment"


def test_a_command_above_the_print_and_an_empty_plan_do_nothing():
    text = _gcode([0.2, 0.4, 0.6])
    assert slicer.with_commands(text, [[9.0, "M900 K0.2"]]) == text
    assert slicer.with_commands(text, []) == text


def test_the_temperature_tower_machinery_still_works_with_them():
    """The M104 lines of a tower and the M900 of a PA test both come into one G-code, in order."""
    text = _gcode([0.2 * k for k in range(1, 401)])        # 0.2 mm layers up to 80 mm
    both = [(z, f"M104 S{t}") for z, t in temperature_tower.steps(220.0, 190.0, 5.0)[1:]]
    both += pressure_advance_test.commands("marlin", pressure_advance_test.steps(**SMALL))
    out = slicer.with_commands(text, sorted(both, key=lambda one: one[0]))
    assert out.count("M104 S") == 6 and out.count("M900 K") == 3
    lines = out.splitlines()
    want = ["M900 K0", "M900 K0.02", "M104 S215", "M104 S190"]     # by height: 0, 10, 11, 61 mm
    at = [k for k, ln in enumerate(lines) if ln in want]
    assert [lines[k] for k in at] == want, "the two kinds of change did not come in height order"


def test_commands_are_written_into_the_g_code_file(tmp_path):
    path = tmp_path / "x.gcode"
    path.write_text(_gcode([0.2, 0.4, 0.6, 0.8, 1.0]), encoding="utf-8")
    slicer.insert_commands(str(path), [[0.5, "M900 K0.02"]])
    assert path.read_text(encoding="utf-8").count("M900 K0.02") == 1


# -------------------------------------------------------------------- the real slice

@needs_engine
def test_a_pressure_advance_test_slices_and_sets_a_value_along_the_way(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    st = p.stacks(p.default_variant(), p.default_material(), "standard",
                  {"infill_sparse_density": 0.0})
    plan = pressure_advance_test.commands("klipper", pressure_advance_test.steps(0.0, 0.02, 0.005))
    mesh = pressure_advance_test.pressure_advance_test(0.0, 0.02, 0.005, labels=False).to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3] + np.array([40.0, 40.0, 0.0])
    tris = np.asarray(mesh.tri_verts)
    out = tmp_path / "pa.gcode"
    slicer.slice_meshes(st, [(verts, tris, 0)], str(out))
    slicer.insert_commands(str(out), plan)

    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    want = [line for _z, line in plan]
    assert want == ["SET_PRESSURE_ADVANCE ADVANCE=0", "SET_PRESSURE_ADVANCE ADVANCE=0.005",
                    "SET_PRESSURE_ADVANCE ADVANCE=0.01", "SET_PRESSURE_ADVANCE ADVANCE=0.015"]
    at = []
    for line in want:
        k = [i for i, ln in enumerate(lines) if ln == line]
        assert k, f"{line} is not in the G-code"
        at.append(k[-1])
    assert at == sorted(at), "the values did not come in order"
    heights = slicer.layer_heights(text)
    for (z_from, _line), k in zip(plan, at):
        first_after = next(h for ln, h in heights if ln >= k)
        assert first_after >= z_from - 0.21, f"the change at line {k} is before {z_from} mm"
    assert re.search(r"^SET_PRESSURE_ADVANCE ADVANCE=0$", text, re.M)


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The pressure advance settings back at what the program starts with."""
    for key, value in SETTINGS:
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_fills_the_fields_and_remembers_them(win, monkeypatch):
    seen = {}

    def accept(dlg):
        firmware_cb, preset_cb = dlg.findChildren(QComboBox)
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["default"] = [firmware_cb.currentData(), preset_cb.currentData(),
                           spins[0].value(), spins[1].value(), spins[2].value()]
        firmware_cb.setCurrentIndex(firmware_cb.findData("klipper"))
        seen["firmware"] = [preset_cb.currentData(), spins[0].value(), spins[1].value(),
                            spins[2].value()]
        preset_cb.setCurrentIndex(preset_cb.findData("klipper_bowden"))
        seen["bowden"] = [spins[0].value(), spins[1].value(), spins[2].value()]
        spins[1].setValue(0.03)
        spins[2].setValue(0.01)
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.pressure_advance_test_dialog()
        assert seen["default"] == ["marlin", "marlin_direct", pytest.approx(0.0),
                                   pytest.approx(0.1), pytest.approx(0.005)]
        assert seen["firmware"] == ["klipper_direct", pytest.approx(0.0), pytest.approx(0.1),
                                    pytest.approx(0.005)]
        assert seen["bowden"] == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(0.05)]
        assert "3" in seen["text"] and "15" in seen["text"], "the dialog does not say what prints"
        assert pump_until(lambda: len(win.doc.groups) == 1), "no pressure advance test"
        g = win.doc.groups[0]
        assert g.name == "Pressure advance test"
        assert g.meta["height_commands"] == pressure_advance_test.commands(
            "klipper", pressure_advance_test.steps(0.0, 0.03, 0.01))
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert pump_until(lambda: "corners" in win.hint.text().lower()), win.hint.text()
        st = app_module.Settings()
        assert st.pressure_advance_firmware == "klipper"
        assert st.pressure_advance_preset == "klipper_bowden"
        assert st.pressure_advance_start == pytest.approx(0.0)
        assert st.pressure_advance_end == pytest.approx(0.03)
        assert st.pressure_advance_step == pytest.approx(0.01)
    finally:
        forget_the_values(win.settings)


def test_a_klipper_printer_on_the_network_fills_the_firmware(win, monkeypatch):
    seen = {}

    def accept(dlg):
        firmware_cb, preset_cb = dlg.findChildren(QComboBox)
        seen["pair"] = [firmware_cb.currentData(), preset_cb.currentData()]
        return QDialog.Rejected

    win.settings.printers = [printer_link.Connection("Zero", "moonraker", "http://zero.local")]
    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.pressure_advance_test_dialog()
        assert seen["pair"] == ["klipper", "klipper_direct"]
    finally:
        win.settings.printers = []


def test_a_saved_firmware_comes_back_before_the_printers_do(win, monkeypatch):
    seen = {}

    def accept(dlg):
        seen["firmware"] = dlg.findChildren(QComboBox)[0].currentData()
        return QDialog.Rejected

    win.settings.printers = [printer_link.Connection("Zero", "moonraker", "http://zero.local")]
    win.settings.pressure_advance_firmware = "rrf"
    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.pressure_advance_test_dialog()
        assert seen["firmware"] == "rrf", "what the user chose last time is not good enough"
    finally:
        win.settings.printers = []
        forget_the_values(win.settings)


def test_the_dialog_refuses_an_end_equal_to_the_start(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.pressure_advance_test_dialog()
    assert win.doc.groups == [], "a test of one value was built anyway"
    assert any("differ" in text.lower() for _title, text in asked), asked


def test_the_dialog_refuses_too_many_segments(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(0.1)
        spins[2].setValue(0.001)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.pressure_advance_test_dialog()
    assert win.doc.groups == [], "a tower of a hundred segments was built anyway"
    assert any("at most" in text.lower() for _title, text in asked), asked


def test_the_dialog_says_what_is_wrong_while_the_values_change(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))       # an end equal to the start
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.pressure_advance_test_dialog()
    assert "differ" in seen["text"].lower(), seen["text"]


def test_one_undo_takes_the_whole_tower_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_pressure_advance_test("klipper", 0.0, 0.03, 0.01,
                                     pressure_advance_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the tower is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_height_commands_come_from_the_part(win):
    win._apply_pressure_advance_test("marlin", 0.0, 0.03, 0.01,
                                     pressure_advance_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    want = pressure_advance_test.commands("marlin", pressure_advance_test.steps(**SMALL))
    assert win.doc.groups[0].meta["height_commands"] == want
    result = win._bed_height_commands(only=False)
    assert result is not None
    print_temp, changes = result
    assert print_temp is None, "a pressure advance test says nothing about the temperature"
    assert changes == [(0.0, "M900 K0"), (5.0, "M900 K0.01"), (10.0, "M900 K0.02")]


def test_the_bed_has_no_commands_without_a_test(win):
    assert win._bed_height_commands(only=False) is None


def test_the_bed_writes_both_kinds_of_command(win):
    """A temperature tower and a pressure advance test on one bed give both kinds of commands."""
    win._apply_temperature_tower(220.0, 190.0, 5.0, temperature_tower.steps(220.0, 190.0, 5.0))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    win._apply_pressure_advance_test("klipper", 0.0, 0.03, 0.01,
                                     pressure_advance_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 2)
    print_temp, changes = win._bed_height_commands(only=False)
    assert print_temp == 220
    temps = [line for _z, line in changes if line.startswith("M104")]
    advance = [line for z, line in changes if line.startswith("SET_PRESSURE_ADVANCE")]
    assert len(temps) == 6, "the tower asks for every temperature past its first"
    assert advance == ["SET_PRESSURE_ADVANCE ADVANCE=0", "SET_PRESSURE_ADVANCE ADVANCE=0.01",
                       "SET_PRESSURE_ADVANCE ADVANCE=0.02"]
    assert changes == sorted(changes, key=lambda one: one[0]), "the changes are not in height order"
    assert (11.0, "M104 S215") in changes and (5.0, "SET_PRESSURE_ADVANCE ADVANCE=0.01") in changes
    assert win._bed_temperature_steps(only=False)[1] == [
        (11.0, 215), (21.0, 210), (31.0, 205), (41.0, 200), (51.0, 195), (61.0, 190)]


def test_a_meta_that_is_no_command_is_ignored(win):
    win._apply_pressure_advance_test("marlin", 0.0, 0.03, 0.01,
                                     pressure_advance_test.steps(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    g = win.doc.groups[0]
    g.meta["height_commands"] = [["geen command"], {"niet": "dit"}, 5, None, [1.0],
                                 [15.0, "M900 K0.03"]]
    changes = win._bed_height_commands(only=False)[1]
    assert changes == [(15.0, "M900 K0.03")], "a junk entry was written along with the real ones"


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("pressure_advance_firmware", "pressure_advance_preset", "pressure_advance_start",
            "pressure_advance_end", "pressure_advance_step")
    saved = {k: q.value(k) for k in keys}
    try:
        q.setValue("pressure_advance_firmware", "KAPUT")
        q.setValue("pressure_advance_preset", "KAPUT")
        q.setValue("pressure_advance_start", 999.0)
        q.setValue("pressure_advance_end", -999.0)
        q.setValue("pressure_advance_step", 999.0)
        q.sync()
        s = app_module.Settings()
        assert s.pressure_advance_firmware == "", "a firmware Volura does not know comes back"
        assert s.pressure_advance_preset == ""
        assert s.pressure_advance_start == 5.0
        assert s.pressure_advance_end == 0.0
        assert s.pressure_advance_step == 1.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_temperature_tower(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Pressure Advance Test") == labels.index("Temperature Tower") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
