"""Retraction Test: two thin towers the nozzle travels between, every band with a longer retraction.

`retraction_test.retraction_test` gives the test as one closed volume standing on `z = 0`: two
towers of `DIAMETER` millimetres `DISTANCE` apart between their centres on a base of `BASE`
millimetres, one band of `BAND_H` millimetres per length from `start` by `step` towards `end`, a
notch of `RING` millimetres cut all round one tower at every boundary between two bands and the
lengths of the lowest and the highest band raised on a tab at the foot of the base.
`retraction_test.lengths` gives the length of every band and `retraction_test.bands` the plan of
``[z_from, z_to, length]`` the part keeps in `g.meta["retraction_bands"]`.

A retraction is no command the printer can be told halfway but a length CuraEngine hides in the
extrusion moves around a travel, so `slicer.rewrite_retractions` rewrites those moves after the
slice: in every layer of a band the move that pulls filament back with nothing moved in x or y gets
the length of the band and the prime that answers it the same length back, in absolute (M82) as well
as relative (M83) extrusion, with the E positions after it following the running difference and a
`G92 E0` starting the count again. The moves of the start and the end code and a firmware that
retracts with `G10`/`G11` are left alone.

These tests (Qt offscreen) weigh that: one closed volume on the bed, `bands * BAND_H` tall, notches
at every inner boundary and nowhere else, the tab at the foot with both numbers standing above it
(and gone without fonts); the walk of lengths, the presets and the validation (a step of 0, too few
or too many bands, a length that is no number, lengths clamped to what a printer retracts); the plan
round-tripping through a file; `rewrite_retractions` on hand-written G-code in both languages of E -
the length per band, the filament where it was, nothing else changed, layers outside the bands and
the start and end code untouched, `G92 E0` and `G10`/`G11` handled; the window (its summary, its
presets, its bands for the slice, one undo) and, when CuraEngine is there, a real slice of the test
whose retractions change at every band while the plastic it lays stays the same.

Run with:  python -m pytest tests/test_retraction_test.py
"""
import logging
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
                    gcode_preview, ops, retraction_test, slicer, window_print)
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: the numbers the tests build with: three bands of 0.5, 1.0 and 1.5 mm
SMALL = dict(start=0.5, end=1.5, step=0.5)
BAND_H = retraction_test.BAND_H
D = retraction_test.DIAMETER
RING = retraction_test.RING
TAB_T = retraction_test.TAB_T
TAB_OUT = retraction_test.TAB_OUT
TAB_W = retraction_test.TAB_W
ABS = 1e-4
#: the settings of the dialog, back at what the program starts with
SETTINGS = (("retraction_test_preset", ""), ("retraction_test_start", 0.5),
            ("retraction_test_end", 3.0), ("retraction_test_step", 0.5))
#: three hand-written bands: the lowest is already the length the G-code comes with
PLAN = [(0.0, 5.0, 1.0), (5.0, 10.0, 2.0), (10.0, 15.0, 3.0)]

_WORD = re.compile(r"([A-Z])\s*(-?\d*\.?\d+)")
_E_WORD = re.compile(r"(?<![A-Za-z])E\s*-?\d*\.?\d+", re.IGNORECASE)


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
    for key in ("slice_printer", "slice_variant", "slice_material", "slice_quality"):
        setattr(w.settings, key, "")
    w.settings.printers = []          # what stands under Preferences is up to the test
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def nofonts(monkeypatch):
    """No fonts: the test is then exactly the towers with their base, tab and notches."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no numbers to test")


def made(**kw) -> mf.Manifold:
    return retraction_test.retraction_test(**dict(SMALL, **kw))


def numbers(**kw) -> dict:
    return retraction_test.sizes(**dict(SMALL, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def probe(m, x, y, z, size=0.1) -> float:
    """How much of a little cube around `x, y, z` is plastic."""
    box = mf.Manifold.cube([size, size, size]).translate((x - 0.5 * size, y - 0.5 * size,
                                                          z - 0.5 * size))
    return (m ^ box).volume()


def row(m):
    """Where the towers stand in a centred test: ``(y of the row, x of both towers)``.

    The tab sticks out of the base on the -y side, so the row is not the middle of the box but
    `TAB_OUT` millimetres further from that side."""
    lo, hi = bbox(m)
    y = lo[1] + TAB_OUT + 0.5 * retraction_test.DEPTH
    return y, (lo[0] + 0.5 * D, hi[0] - 0.5 * D)


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def _gcode(zs, abs_e=True, markers=True, start_code=False, end_code=False,
           retraction=1.0, prime=None) -> str:
    """A small G-code with one layer per height, a retraction and a prime around a travel."""
    prime = retraction if prime is None else prime
    e = 20.0 if start_code and abs_e else (100.0 if abs_e else 0.0)
    lines = ["M82" if abs_e else "M83", "G90"]
    if start_code:
        if abs_e:
            lines += ["G92 E0", f"G1 F300 E{e:.5f}"]
        else:
            lines += ["G1 F300 E20.00000"]
    for i, z in enumerate(zs):
        if markers:
            lines.append(f";LAYER:{i}")
        lines.append(f"G1 Z{z:.3f} F900")
        if abs_e:
            lines.append(f"G1 F1200 X10 Y10 E{e + 0.2:.5f}")
            e += 0.2
            lines.append(f"G1 F2700 E{e - retraction:.5f}")
            e -= retraction
        else:
            lines.append("G1 F1200 X10 Y10 E0.20000")
            lines.append(f"G1 F2700 E-{retraction:.5f}")
        lines.append("G0 F6000 X20 Y10")
        if abs_e:
            lines.append(f"G1 F2700 E{e + prime:.5f}")
            e += prime
        else:
            lines.append(f"G1 F2700 E{prime:.5f}")
        if markers:
            lines.append(f";TIME_ELAPSED:{i * 10}")
    if end_code:
        lines.append(f"G1 F2700 E{e - 5.0:.5f}" if abs_e else "G1 F2700 E-5.00000")
        lines.append("G0 F6000 X30 Y10")
    return "\n".join(lines) + "\n"


def _filament_moves(text):
    """Every move that touches filament as ``(z, delta, only_filament, inside_a_layer)``.

    It reads the same languages of E as the slice does: absolute E words are positions, relative
    E words are differences, and `G92 E0` starts the position again. This is deliberately written
    like the code it checks, so the test says what the printer would really do."""
    out, pos, shift, z = [], 0.0, 0.0, 0.0
    abs_e, inside = True, ";LAYER:" not in text
    for line in text.splitlines():
        if line.startswith(";LAYER:"):
            inside = True
            continue
        if line.startswith(";TIME_ELAPSED:"):
            inside = False
            continue
        body = line.split(";", 1)[0].strip()
        if not body:
            continue
        code = body.upper()
        head = code.split(None, 1)[0]
        if head in ("G0", "G1", "G00", "G01"):
            words = {k: float(v) for k, v in _WORD.findall(code[len(head):])}
            if "Z" in words:
                z = words["Z"]
            if "E" not in words:
                continue
            value = words["E"]
            delta = value + shift - pos if abs_e else value
            out.append((z, delta, not ("X" in words or "Y" in words), inside))
            pos = value + shift if abs_e else pos + value
        elif head == "M82":
            abs_e = True
        elif head == "M83":
            abs_e = False
        elif head == "G92":
            words = {k: float(v) for k, v in _WORD.findall(code[3:])}
            if "Z" in words:
                z = words["Z"]
            if "E" in words:
                pos, shift = words["E"], 0.0
    return out


def _only_values(text, negative=True):
    """The lengths of the moves that only turn the extruder, inside the layers, with their sign."""
    return [round(d, 6) for _z, d, only, inside in _filament_moves(text)
            if only and inside and (d < -1e-9 if negative else d > 1e-9)]


def _band_values(text, plan, negative=True):
    """The lengths of every band in the plan: the retractions (or primes) that stand in it."""
    out = [[] for _ in plan]
    for z, d, only, inside in _filament_moves(text):
        if only and inside and ((d < -1e-9) == negative and abs(d) > 1e-9):
            for i, (z_from, z_to, _want) in enumerate(plan):
                if z_from - 1e-6 <= z < z_to - 1e-6:
                    out[i].append(abs(d))
                    break
    return out


def _plastic(text) -> float:
    """How much filament every move that lays a line pushes out, primes aside."""
    return sum(d for _z, d, only, _inside in _filament_moves(text) if d > 1e-9 and not only)


def _total_filament(text) -> float:
    """The net movement of the filament through the whole file, the moves of the primes included."""
    return sum(d for _z, d, _only, _inside in _filament_moves(text))


def _without_e(text):
    """The G-code with every E word emptied: only the numbers on the filament may differ."""
    return _E_WORD.sub("E", text)

# ------------------------------------------------------------------ the towers and the base

@pytest.mark.parametrize("labels", (False, True))
def test_the_test_is_one_volume_that_stands_on_the_bed(labels):
    s = numbers()
    m = made(labels=labels)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the test fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the test does not stand on the bed"
    assert hi[0] - lo[0] == pytest.approx(retraction_test.LENGTH, abs=ABS)
    assert hi[1] - lo[1] == pytest.approx(retraction_test.DEPTH + TAB_OUT, abs=ABS), \
        "the tab at the foot is not part of the test"
    assert hi[2] - lo[2] == pytest.approx(s["bands"] * BAND_H, abs=1e-3), "not as tall as asked"


def test_the_test_is_taller_for_every_band_it_has():
    for start, end, step in ((0.5, 5.0, 0.5), (0.5, 9.5, 1.0)):
        n = retraction_test.sizes(start, end, step)["bands"]
        lo, hi = bbox(retraction_test.retraction_test(start, end, step, labels=False))
        assert hi[2] - lo[2] == pytest.approx(n * BAND_H, abs=1e-3)


def test_two_thin_towers_stand_at_both_ends_of_the_base():
    """Both towers are there, `DISTANCE` apart, and nothing stands between them."""
    m = made(labels=False)
    y, xs = row(m)
    assert xs[1] - xs[0] == pytest.approx(retraction_test.DISTANCE, abs=ABS)
    for x in xs:
        assert probe(m, x, y, 12.0) > 0.0, "a tower is missing"
        assert probe(m, x, y, 0.5) > 0.0, "a tower does not stand on the base"
        assert probe(m, x + 0.5 * D - 0.06, y, 12.0) > 0.0, "the tower is not as wide as it is"
        assert probe(m, x + 0.5 * D + 0.06, y, 12.0) == pytest.approx(0.0, abs=1e-9), \
            "the tower is thicker than it is"
    assert probe(m, 0.5 * (xs[0] + xs[1]), y, 12.0) == pytest.approx(0.0, abs=1e-9), \
        "something connects the towers, so the nozzle would not have to travel"


@pytest.mark.parametrize("z", (5.0, 10.0))
def test_there_is_a_notch_at_every_boundary_between_two_bands(z):
    m = made(labels=False)
    y, xs = row(m)
    groove = 0.5 * D - 0.5 * RING
    assert probe(m, xs[0] + groove, y, z, size=0.2) == pytest.approx(0.0, abs=1e-9), \
        f"the boundary at {z} mm has no notch"


def test_the_boundaries_on_the_bed_and_on_the_top_get_no_notch():
    m = made(labels=False)
    y, xs = row(m)
    groove = 0.5 * D - 0.5 * RING
    solid = 0.5 * D - RING - 0.15
    assert probe(m, xs[0] + solid, y, 0.3, size=0.2) > 0.0, "the bed was notched too"
    assert probe(m, xs[0] + groove, y, 14.7, size=0.2) > 0.0, "the top was notched too"


def test_only_one_tower_carries_the_notches():
    m = made(labels=False)
    y, xs = row(m)
    groove = 0.5 * D - 0.5 * RING
    assert probe(m, xs[1] + groove, y, 5.0, size=0.2) > 0.0, \
        "the second tower was notched too, where it cannot be read apart from the base"


def test_between_the_boundaries_the_tower_stands_plain():
    m = made(labels=False)
    y, xs = row(m)
    groove = 0.5 * D - 0.5 * RING
    for z in (2.5, 7.5, 12.5):
        assert probe(m, xs[0] + groove, y, z, size=0.2) > 0.0, f"something is cut at {z} mm"


def test_the_tab_stands_at_the_foot_and_only_there():
    m = made(labels=False)
    lo, hi = bbox(m)
    y = lo[1] + 0.5 * TAB_OUT
    assert probe(m, 0.0, y, 0.5 * TAB_T) > 0.0, "the tab at the foot is not there"
    assert probe(m, 0.0, y, TAB_T + 0.3, size=0.3) == pytest.approx(0.0, abs=1e-9), \
        "the tab stands higher than it is thick"
    assert hi[2] > TAB_T + 10.0, "the tab is not at the foot of the towers"


@pytest.mark.parametrize("labels", (False, True))
def test_the_default_walk_makes_six_bands_of_thirty_millimetres(labels):
    s = retraction_test.sizes()
    assert s["bands"] == 6
    lo, hi = bbox(retraction_test.retraction_test(labels=labels))
    assert hi[2] - lo[2] == pytest.approx(30.0, abs=1e-3)
    assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=1e-3)
    assert hi[1] - lo[1] == pytest.approx(s["size"][1], abs=1e-3)


def test_the_tab_label_stands_above_the_tab_at_both_ends():
    need_fonts()
    plain, with_labels = made(labels=False), made(labels=True)
    raised = with_labels - plain
    assert raised.volume() > 0.0, "the lengths are not raised on the tab"
    assert with_labels.volume() > plain.volume(), "the raised numbers take plastic away"
    assert len(with_labels.decompose()) == 1, "the numbers made the test fall apart"
    lo, hi = bbox(raised)
    assert lo[2] == pytest.approx(TAB_T, abs=1e-6), "the numbers do not stand on the tab"
    assert hi[2] == pytest.approx(TAB_T + retraction_test.LABEL_PROTR, abs=1e-6), \
        "the numbers are not as high as they are"
    for x in (-0.25 * TAB_W, 0.25 * TAB_W):
        band = mf.Manifold.cube([8.0, 8.0, 2.0]).translate((x - 4.0, -7.5, TAB_T))
        assert (raised ^ band).volume() > 0.0, f"no length stands at x = {x}"


def test_without_fonts_the_tab_carries_no_numbers(nofonts):
    m = made(labels=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.volume() == pytest.approx(made(labels=False).volume(), rel=1e-9)


# ------------------------------------------------------------------ the lengths and the plan

def test_the_default_walk_is_six_lengths_and_prints_both_ends():
    assert retraction_test.lengths() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert retraction_test.sizes()["bands"] == 6
    assert retraction_test.sizes()["size"] == (retraction_test.LENGTH,
                                               retraction_test.DEPTH + TAB_OUT, 30.0)


def test_a_walk_down_gives_the_lengths_highest_first():
    assert retraction_test.lengths(1.5, 0.5, 0.5) == [1.5, 1.0, 0.5]


@pytest.mark.parametrize("value, text", ((7.5, "7.5"), (5.0, "5"), (0.30000000000000004, "0.3"),
                                         (-0.0, "0"), (12.25, "12.25")))
def test_a_length_is_written_without_trailing_noise(value, text):
    assert retraction_test.length_text(value) == text


def test_the_plan_lists_every_band_with_its_height_and_length():
    assert retraction_test.bands(**SMALL) == [[0.0, 5.0, 0.5], [5.0, 10.0, 1.0],
                                              [10.0, 15.0, 1.5]]
    plan = retraction_test.bands()
    assert [z for z, _z_to, _v in plan] == [i * BAND_H for i in range(6)]
    assert plan[-1][2] == 3.0


def test_the_plan_carries_no_noise_of_the_walk():
    for _z_from, _z_to, v in retraction_test.bands():
        assert v == round(v, 3)


def test_every_preset_gives_a_sensible_test():
    for name, (start, end, step) in retraction_test.PRESETS.items():
        s = retraction_test.sizes(start, end, step)
        assert retraction_test.BANDS_MIN <= s["bands"] <= retraction_test.BANDS_MAX, name
        assert retraction_test.LENGTH_MIN <= s["lengths"][0] <= retraction_test.LENGTH_MAX, name
        assert retraction_test.LENGTH_MIN <= s["lengths"][-1] <= retraction_test.LENGTH_MAX, name
        assert s["lengths"][0] == pytest.approx(start), name
        assert s["lengths"][-1] <= end, name
        assert s["size"][2] == s["bands"] * BAND_H, name
        m = retraction_test.retraction_test(start, end, step, labels=False)
        assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, name


def test_the_dialog_names_the_hotend_and_the_walk_it_makes():
    assert retraction_test.preset_label("direct_drive") == "Direct drive 0.5-3"
    assert retraction_test.preset_label("bowden") == "Bowden 2-7"
    assert retraction_test.preset_label("unknown") == "unknown"
    assert retraction_test.default_preset() == next(iter(retraction_test.PRESETS))


# ------------------------------------------------------------------ the validation

@pytest.mark.parametrize("bad", (0.0, -2.5))
def test_a_step_of_zero_or_less_is_refused(bad):
    with pytest.raises(ValueError):
        retraction_test.retraction_test(0.5, 3.0, bad)
    with pytest.raises(ValueError):
        retraction_test.lengths(0.5, 3.0, bad)


def test_a_walk_that_gives_one_band_is_refused():
    with pytest.raises(ValueError) as gone:
        retraction_test.retraction_test(1.0, 1.0, 0.5)
    assert "at least 2" in str(gone.value)
    with pytest.raises(ValueError):
        retraction_test.lengths(1.0, 1.25, 0.5)


def test_too_many_bands_is_refused():
    with pytest.raises(ValueError) as gone:
        retraction_test.lengths(0.1, 10.0, 0.05)
    assert "at most 15" in str(gone.value)
    with pytest.raises(ValueError):
        retraction_test.sizes(0.1, 10.0, 0.05)
    with pytest.raises(ValueError):
        retraction_test.retraction_test(0.1, 10.0, 0.05)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "slow"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("start", "end", "step"):
        with pytest.raises(ValueError):
            retraction_test.retraction_test(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            retraction_test.lengths(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            retraction_test.bands(**dict(SMALL, **{key: bad}))
        with pytest.raises(ValueError):
            retraction_test.sizes(**dict(SMALL, **{key: bad}))


def test_the_lengths_clamp_to_what_a_printer_retracts():
    ls = retraction_test.lengths(-5.0, 12.0, 4.7)
    assert ls[0] == pytest.approx(retraction_test.LENGTH_MIN)
    assert ls[-1] <= retraction_test.LENGTH_MAX
    assert all(retraction_test.LENGTH_MIN <= v <= retraction_test.LENGTH_MAX for v in ls)
    m = retraction_test.retraction_test(-5.0, 12.0, 4.7, labels=False)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1


# -------------------------------------------------------------------- the plan in a file

def test_the_bands_round_trip_through_a_file(tmp_path):
    doc = Document()
    g = ops.group_from_manifold(doc, made(labels=False), "#f2f2f2", "Retraction test")
    plan = retraction_test.bands(**SMALL)
    g.meta = {"retraction_bands": plan}
    path = str(tmp_path / "rt.volura")
    fileio.save_document(doc, path)
    doc2 = Document()
    fileio.load_document(doc2, path)
    assert doc2.group(g.id).meta["retraction_bands"] == plan

# -------------------------------------------------------------- the retractions in the G-code

def test_absolute_retractions_get_the_length_of_their_band():
    text = _gcode([1.0, 3.0, 6.0, 9.0, 12.0, 14.0], abs_e=True)
    out = slicer.rewrite_retractions(text, PLAN)
    got = _band_values(out, PLAN)
    assert [len(one) for one in got] == [2, 2, 2], got
    for (z_from, _z_to, want), values in zip(PLAN, got):
        assert values, f"the band from {z_from} mm has no retractions"
        assert all(v == pytest.approx(want, abs=1e-4) for v in values), (z_from, values)


def test_relative_retractions_get_the_length_of_their_band():
    text = _gcode([1.0, 3.0, 6.0, 9.0, 12.0, 14.0], abs_e=False)
    out = slicer.rewrite_retractions(text, PLAN)
    got = _band_values(out, PLAN)
    assert [len(one) for one in got] == [2, 2, 2], got
    for (z_from, _z_to, want), values in zip(PLAN, got):
        assert values, f"the band from {z_from} mm has no retractions"
        assert all(v == pytest.approx(want, abs=1e-4) for v in values), (z_from, values)


@pytest.mark.parametrize("abs_e", (True, False))
def test_primes_answer_their_retractions_and_leave_the_filament_where_it_was(abs_e):
    text = _gcode([1.0, 3.0, 6.0, 9.0, 12.0, 14.0], abs_e=abs_e)
    out = slicer.rewrite_retractions(text, PLAN)
    only = [round(d, 6) for _z, d, one, inside in _filament_moves(out) if one and inside]
    want = [-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 2.0, -3.0, 3.0, -3.0, 3.0]
    assert only == want, "a retraction or its prime did not get the length of its band"
    assert [round(a + b, 6) for a, b in zip(only[::2], only[1::2])] == [0.0] * 6


@pytest.mark.parametrize("abs_e", (True, False))
def test_nothing_but_the_retraction_moves_changes(abs_e):
    text = _gcode([1.0, 3.0, 6.0, 9.0, 12.0, 14.0], abs_e=abs_e)
    out = slicer.rewrite_retractions(text, PLAN)
    assert out != text, "the bands did not change the G-code"
    assert _without_e(out) == _without_e(text), "something other than an E word changed"
    assert _plastic(out) == pytest.approx(_plastic(text), rel=1e-4), "the test lays other plastic"
    assert _total_filament(out) == pytest.approx(_total_filament(text), abs=1e-3), \
        "the filament did not end where it was"


def test_a_g92_starts_the_count_of_filament_again():
    text = """M82
G90
;LAYER:0
G1 Z1.000 F900
G1 F1200 X10 Y10 E11
G1 F2700 E10
G0 F6000 X20 Y10
G1 F2700 E11
;TIME_ELAPSED:0
G92 E0
;LAYER:1
G1 Z6.000 F900
G1 F1200 X10 Y10 E1
G1 F2700 E-1
G0 F6000 X20 Y10
G1 F2700 E1
;TIME_ELAPSED:10
"""
    out = slicer.rewrite_retractions(text, [(0.0, 5.0, 2.0), (5.0, 10.0, 3.0)])
    assert _without_e(out) == _without_e(text)
    only = [round(d, 6) for _z, d, one, inside in _filament_moves(out) if one and inside]
    assert only == [-2.0, 2.0, -3.0, 3.0], only
    assert _plastic(out) == pytest.approx(_plastic(text), rel=1e-4)


def test_layers_outside_the_bands_keep_their_retraction():
    text = _gcode([2.5, 7.5, 12.5])
    out = slicer.rewrite_retractions(text, [(5.0, 10.0, 2.5)])
    assert [round(abs(v), 6) for v in _only_values(out)] == [1.0, 2.5, 1.0]
    assert _without_e(out) == _without_e(text)


def test_a_file_without_layer_markers_is_filtered_on_height_alone():
    text = _gcode([2.5, 7.5, 12.5], markers=False)
    out = slicer.rewrite_retractions(text, [(5.0, 10.0, 2.5)])
    assert [round(abs(v), 6) for v in _only_values(out)] == [1.0, 2.5, 1.0]


def test_the_start_and_end_code_keep_their_moves():
    text = _gcode([1.0, 6.0], abs_e=True, start_code=True, end_code=True)
    out = slicer.rewrite_retractions(text, [(0.0, 5.0, 2.0)])
    before = [round(d, 6) for _z, d, one, inside in _filament_moves(text) if one and not inside]
    after = [round(d, 6) for _z, d, one, inside in _filament_moves(out) if one and not inside]
    assert after == before, "the start or the end code was rewritten too"
    inside = [round(d, 6) for _z, d, one, inside in _filament_moves(out) if one and inside]
    assert inside == [-2.0, 2.0, -1.0, 1.0], inside
    assert _without_e(out) == _without_e(text)


def test_a_retraction_and_its_prime_keep_their_pair_at_a_band_boundary():
    text = _gcode([4.8, 5.2])
    out = slicer.rewrite_retractions(text, [(0.0, 5.0, 1.0), (5.0, 10.0, 2.0)])
    only = [round(d, 6) for _z, d, one, inside in _filament_moves(out) if one and inside]
    assert only == [-1.0, 1.0, -2.0, 2.0], only


def test_firmware_retraction_is_left_alone(caplog):
    text = """M82
G90
;LAYER:0
G1 Z1.000 F900
G1 F1200 X10 Y10 E1
G10
G0 F6000 X20 Y10
G11
G1 F1200 X30 Y20 E2
;TIME_ELAPSED:0
"""
    caplog.set_level(logging.WARNING, logger="volura.slicer")
    out = slicer.rewrite_retractions(text, [(0.0, 5.0, 2.0)])
    assert out == text, "G10 and G11 carry no length and must be left to the printer"
    assert "G10" in caplog.text and "G11" in caplog.text


def test_junk_bands_are_left_out(caplog):
    text = _gcode([1.0, 6.0])
    caplog.set_level(logging.WARNING, logger="volura.slicer")
    assert slicer.rewrite_retractions(text, [None, [1.0], [0.0, -1.0, 2.0],
                                             [0.0, 5.0, "x"]]) == text
    out = slicer.rewrite_retractions(text, [None, [1.0], [0.0, -1.0, 2.0],
                                            [0.0, 5.0, "x"], [0.0, 5.0, 2.0]])
    assert [round(abs(v), 6) for v in _only_values(out)] == [2.0, 1.0]
    assert caplog.records, "a band that is no band was ignored without a word"


def test_the_first_band_wins_at_an_overlap():
    text = _gcode([1.0, 4.0])
    out = slicer.rewrite_retractions(text, [(0.0, 5.0, 2.0), (2.0, 7.0, 9.0)])
    assert [round(abs(v), 6) for v in _only_values(out)] == [2.0, 2.0]


def test_the_top_of_a_band_is_not_inside_the_band():
    text = _gcode([0.0, 5.0, 10.0, 15.0])
    out = slicer.rewrite_retractions(text, PLAN[:2])
    assert [round(abs(v), 6) for v in _only_values(out)] == [1.0, 2.0, 1.0, 1.0]


def test_the_retractions_are_rewritten_in_the_g_code_file(tmp_path):
    path = tmp_path / "rt.gcode"
    path.write_text(_gcode([1.0]), encoding="utf-8")
    slicer.insert_retractions(str(path), [(0.0, 5.0, 2.0)])
    made = path.read_text(encoding="utf-8")
    assert _only_values(made) == [-2.0]
    assert _only_values(made, negative=False) == [2.0]


def test_no_bands_and_an_empty_file_change_nothing():
    assert slicer.rewrite_retractions(_gcode([1.0]), []) == _gcode([1.0])
    assert slicer.rewrite_retractions("", PLAN) == ""


def test_empty_bands_leave_the_g_code_file_alone(tmp_path):
    path = tmp_path / "rt.gcode"
    path.write_text(_gcode([1.0]), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    slicer.insert_retractions(str(path), [])
    assert path.read_text(encoding="utf-8") == before


# -------------------------------------------------------------------- the real slice

@needs_engine
def test_a_retraction_test_slices_with_its_own_lengths_per_band(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    st = p.stacks(p.default_variant(), p.default_material(), "standard", {})
    mesh = made(labels=False).to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3] + np.array([60.0, 60.0, 0.0])
    tris = np.asarray(mesh.tri_verts)
    out = tmp_path / "rt.gcode"
    slicer.slice_meshes(st, [(verts, tris, 0, {})], str(out))

    before = out.read_text(encoding="utf-8", errors="replace")
    before_preview = gcode_preview.read_gcode(str(out))
    plan = retraction_test.bands(**SMALL)
    slicer.insert_retractions(str(out), plan)
    after = out.read_text(encoding="utf-8", errors="replace")
    after_preview = gcode_preview.read_gcode(str(out))

    assert _without_e(after) == _without_e(before), "the rewrite changed something but the E words"
    got = _band_values(after, plan)
    for (z_from, z_to, want), values in zip(plan, got):
        assert values, f"no retraction stands between {z_from} and {z_to} mm"
        assert all(v == pytest.approx(want, abs=1e-3) for v in values), (z_from, values)
    assert _plastic(after) == pytest.approx(_plastic(before), rel=1e-3), \
        "the rewrite changed how much plastic the test lays"
    assert _total_filament(after) == pytest.approx(_total_filament(before), abs=1e-2), \
        "the filament no longer ends where it did"
    assert len(after_preview.layers) == len(before_preview.layers)
    assert np.allclose(after_preview.filament_mm(), before_preview.filament_mm(),
                       rtol=1e-3, atol=1e-3), "the preview reads other plastic than before"

# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The retraction test settings back at what the program starts with."""
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
        preset_cb.setCurrentIndex(preset_cb.findData("bowden"))
        seen["preset"] = [spins[0].value(), spins[1].value(), spins[2].value()]
        spins[0].setValue(0.5)
        spins[1].setValue(1.5)
        spins[2].setValue(0.5)
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    try:
        win.retraction_test_dialog()
        assert seen["default"] == ["direct_drive", pytest.approx(0.5), pytest.approx(3.0),
                                   pytest.approx(0.5)]
        assert seen["preset"] == [pytest.approx(2.0), pytest.approx(7.0), pytest.approx(1.0)]
        assert "3 bands: 0.5 to 1.5 mm" in seen["text"], seen["text"]
        assert "54 × 16 × 15 mm" in seen["text"], seen["text"]
        assert "whole bed" in seen["text"], seen["text"]
        assert pump_until(lambda: len(win.doc.groups) == 1), "no retraction test"
        g = win.doc.groups[0]
        assert g.name == "Retraction test"
        assert g.meta["retraction_bands"] == retraction_test.bands(**SMALL)
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert [i[2] for i in win.doc.selection] == [g.id]
        assert pump_until(lambda: "notches" in win.hint.text().lower()), win.hint.text()
        st = app_module.Settings()
        assert st.retraction_test_preset == "bowden"
        assert st.retraction_test_start == pytest.approx(0.5)
        assert st.retraction_test_end == pytest.approx(1.5)
        assert st.retraction_test_step == pytest.approx(0.5)
    finally:
        forget_the_values(win.settings)


def test_the_summary_says_the_walk_the_size_and_the_whole_bed(win, monkeypatch):
    seen = {}

    def accept(dlg):
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.retraction_test_dialog()
    text = seen["text"]
    assert "6 bands: 0.5 to 3 mm" in text, text
    assert "54 × 16 × 30 mm" in text, text
    assert "whole bed" in text, text
    assert "band" in text and "hold" in text, text


def test_the_dialog_says_what_is_wrong_while_the_values_change(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))
        seen["text"] = dlg.findChildren(QLabel)[-1].text()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.retraction_test_dialog()
    assert "at least" in seen["text"].lower(), seen["text"]


def test_the_dialog_refuses_a_walk_that_gives_one_band(win, monkeypatch):
    asked = []
    monkeypatch.setattr(window_print.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: asked.append((title, text)))

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[1].setValue(float(spins[0].value()))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.retraction_test_dialog()
    assert win.doc.groups == [], "a test with one band was built anyway"
    assert any("at least" in text.lower() for _title, text in asked), asked


def test_a_band_is_made_without_fonts_too(nofonts, win):
    steps = len(win.doc.undo_stack)
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1), "the test did not come without fonts"
    assert win.doc.groups[0].ents.to_manifold().status() == mf.Error.NoError
    assert len(win.doc.undo_stack) == steps + 1


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_bed_gives_the_slice_the_bands_of_its_test(win):
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    plan = retraction_test.bands(**SMALL)
    assert win._bed_retraction_bands(only=False) == [tuple(one) for one in plan]
    assert win._bed_retraction_bands(only=True) == [tuple(one) for one in plan]


def test_the_bed_only_uses_selected_groups(win):
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    win.doc.set_selection([])
    assert win._bed_retraction_bands(only=True) == []
    assert win._bed_retraction_bands(only=False)


def test_two_tests_on_one_bed_hand_over_every_band(win):
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    win._apply_retraction_test(2.0, 4.0, 2.0, retraction_test.bands(2.0, 4.0, 2.0))
    assert pump_until(lambda: len(win.doc.groups) == 2)
    bands = win._bed_retraction_bands(only=False)
    assert [z for z, _z_to, _length in bands] == [0.0, 0.0, 5.0, 5.0, 10.0]
    assert bands == sorted(bands, key=lambda one: one[0]), "the bands are not in height order"


def test_a_meta_that_is_no_band_is_ignored(win):
    win._apply_retraction_test(0.5, 1.5, 0.5, retraction_test.bands(**SMALL))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    g = win.doc.groups[0]
    g.meta["retraction_bands"] = ["geen band", {"niet": "dit"}, 5, None, [1.0, 2.0],
                                  [1.0, 2.0, 0.75]]
    assert win._bed_retraction_bands(only=False) == [(1.0, 2.0, 0.75)]


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("retraction_test_preset", "retraction_test_start", "retraction_test_end",
            "retraction_test_step")
    saved = {k: q.value(k) for k in keys}
    try:
        q.setValue("retraction_test_preset", "KAPUT")
        q.setValue("retraction_test_start", 999.0)
        q.setValue("retraction_test_end", -999.0)
        q.setValue("retraction_test_step", 999.0)
        q.sync()
        s = app_module.Settings()
        assert s.retraction_test_preset == "", "a preset Volura does not know comes back"
        assert s.retraction_test_start == 10.0
        assert s.retraction_test_end == 0.1
        assert s.retraction_test_step == 5.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_max_speed_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Retraction Test") == labels.index("Max Speed Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
