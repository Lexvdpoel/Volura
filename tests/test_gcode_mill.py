"""G-code for a 2.5D mill: the levels it goes down in, the order of the layers, the text.

`volura/gcode_mill.py` is the core without Qt. `grbl_mill` lets `cam.compensate` move the
drawn contours to the path of the centre of the cutter and writes the G-code on top of
that: Z = 0 is the top of the material, every layer goes down to its depth in steps, and
the drilled holes come before the pockets, those before the inside contours and the
outside contours that free the part last. `mill_estimate` reads the same text back the way
a controller follows it, so the time stands for the very job that is written.

Run with:  python -m pytest tests/test_gcode_mill.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import __version__  # noqa: E402
from volura.gcode_mill import grbl_mill, mill_estimate  # noqa: E402
from volura.vector2d import Drawing  # noqa: E402


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def circle(cx, cy, r, n=72):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def sheet(*shapes, layer="Mill outside", **params) -> Drawing:
    """An empty worksheet whose default layers carry `params`; `shapes` go on `layer`."""
    d = Drawing.new_sheet()
    for lay in d.layers.values():
        lay.params.update(params)
    for pts in shapes:
        d.add_path(pts, closed=True, layer=layer, src=1)
    return d


def words(line) -> dict:
    """The words of a G-code line, as {letter: value}; anything without a number is left out."""
    out = {}
    for word in line.split():
        try:
            out[word[0]] = float(word[1:])
        except (IndexError, ValueError):
            continue
    return out


def plunges(text) -> list[float]:
    """The Z of every `G1 Z` line, in order: the levels the cutter goes down to."""
    return [w["Z"] for w in (words(line.split(";")[0]) for line in text.splitlines())
            if "Z" in w and w.get("G") == 1.0]


def cut_points(text) -> set[tuple[float, float]]:
    """Every point the cutter cuts, so the XY of the G1 moves that move in XY."""
    out, mode = set(), "G1"
    for raw in text.splitlines():
        w = words(raw.split(";")[0])
        if not w:
            continue
        if "G" in w:
            mode = int(w["G"])
        if mode == 1 and "X" in w and "Y" in w:
            out.add((w["X"], w["Y"]))
    return out


def drill_spots(text) -> list[tuple[float, float]]:
    """Where the mill goes straight down: the XY of every `G1 Z` line."""
    out, pos = [], (0.0, 0.0)
    for raw in text.splitlines():
        w = words(raw.split(";")[0])
        if "X" in w and "Y" in w:
            pos = (w["X"], w["Y"])
        elif "Z" in w and w.get("G") == 1.0:
            out.append(pos)
    return out


def distance_to_rect(q, w=40.0, h=30.0) -> float:
    """How far a point lies outside a rectangle of `w` × `h` at the origin."""
    dx = max(abs(q[0] - w / 2.0) - w / 2.0, 0.0)
    dy = max(abs(q[1] - h / 2.0) - h / 2.0, 0.0)
    return math.hypot(dx, dy)


# ------------------------------------------------------------------ the levels

def test_a_layer_of_3_mm_with_steps_of_1_mm_cuts_at_every_level():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0))
    assert plunges(text) == [-1.0, -2.0, -3.0]


def test_the_levels_are_written_with_three_decimals():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0))
    assert "G1 Z-1.000 F300" in text.splitlines()


def test_a_step_of_zero_cuts_the_depth_in_one_pass():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=0.0))
    assert plunges(text) == [-3.0]


def test_the_last_level_lands_on_the_depth_also_when_the_step_does_not_divide_it():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=2.5, step_depth=1.0))
    assert plunges(text) == [-1.0, -2.0, -2.5]


def test_nothing_is_cut_deeper_than_the_depth_of_the_layer():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=2.5, step_depth=1.0))
    assert all(w.get("Z", 0.0) >= -2.5 for w in map(words, text.splitlines()))


# ------------------------------------------------------------------ the compensation

def test_the_cutter_runs_1_5_mm_outside_the_drawn_rectangle():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0))
    assert cut_points(text)
    assert all(distance_to_rect(q) == pytest.approx(1.5, abs=1e-3)
               for q in cut_points(text))


# ------------------------------------------------------------------ the text around it

def test_the_text_starts_with_the_header_and_ends_at_the_origin():
    text = grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0))
    lines = text.splitlines()
    assert lines[:5] == [f"; Volura {__version__}", "G21", "G90", "G17", "G0 Z5.000"]
    assert lines[-3:] == ["G0 Z5.000", "M5", "G0 X0 Y0"]
    assert "M3 S10000" in lines and "G4 P2" in lines, "the spindle and the time to spin up"


def test_the_spindle_is_started_once_when_two_layers_want_the_same_rpm():
    d = sheet(rect(40, 30), tool_d=3.0, depth=1.0, step_depth=1.0)
    d.add_path(rect(6, 6, 20, 12), closed=True, layer="Mill inside", src=2)
    assert grbl_mill(d).count("M3 S") == 1


# ------------------------------------------------------------------ the order of the layers

def test_a_hole_is_milled_before_the_part_is_cut_loose():
    d = sheet(rect(40, 30), tool_d=3.0, depth=1.0, step_depth=1.0)
    d.add_path(circle(20, 15, 4), closed=True, layer="Mill inside", src=2)
    lines = grbl_mill(d).splitlines()
    inside = next(i for i, line in enumerate(lines) if line.startswith("; layer Mill inside"))
    outside = next(i for i, line in enumerate(lines) if line.startswith("; layer Mill outside"))
    assert inside < outside


# ------------------------------------------------------------------ what is not milled

def test_a_hidden_layer_goes_to_no_machine_at_all():
    d = sheet(rect(40, 30), tool_d=3.0, depth=1.0, step_depth=1.0)
    d.layers["Mill outside"].visible = False
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_mill(d)


def test_a_worksheet_that_is_only_for_the_laser_is_no_milling_job():
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_mill(sheet(rect(40, 30), layer="Cut", kerf=0.15))


def test_a_layer_without_feed_names_the_layer():
    with pytest.raises(ValueError, match="layer Mill outside has no feed"):
        grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=1.0, feed=0.0))


def test_a_layer_without_plunge_feed_names_the_layer():
    with pytest.raises(ValueError, match="layer Mill outside has no plunge feed"):
        grbl_mill(sheet(rect(40, 30), tool_d=3.0, depth=1.0, plunge=0.0))


# ------------------------------------------------------------------ drilling

def test_a_drill_layer_makes_one_hole_per_circle_at_its_centre():
    d = sheet(circle(10, 10, 2), circle(60, 25, 3), layer="Drill", depth=4.0)
    text = grbl_mill(d)
    assert drill_spots(text) == [(10.0, 10.0), (60.0, 25.0)]
    assert "G1 Z-4.000 F300" in text.splitlines()


def test_a_drill_layer_without_mill_settings_uses_the_defaults():
    """A drill layer carries the settings of a laser, so the depth comes from the mill."""
    d = Drawing.new_sheet()
    d.add_path(circle(10, 10, 2), closed=True, layer="Drill", src=1)
    assert "G1 Z-3.000 F300" in grbl_mill(d).splitlines()


# ------------------------------------------------------------------ the estimate

def test_the_estimate_counts_the_millimetres_and_the_seconds():
    d = sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0)
    info = mill_estimate(d)
    assert info["cut_mm"] > 0.0 and info["travel_mm"] > 0.0 and info["seconds"] > 0.0


def test_the_estimate_grows_with_the_depth():
    shallow = mill_estimate(sheet(rect(40, 30), tool_d=3.0, depth=1.0, step_depth=1.0))
    deep = mill_estimate(sheet(rect(40, 30), tool_d=3.0, depth=3.0, step_depth=1.0))
    assert deep["cut_mm"] > shallow["cut_mm"] and deep["seconds"] > shallow["seconds"]
