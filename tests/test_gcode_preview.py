"""Reading sliced G-code back for the print preview (volura/gcode_preview.py).

Run with:  python -m pytest tests/test_gcode_preview.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import gcode_preview as gp  # noqa: E402

GCODE = """\
;FLAVOR:Marlin
G28 ; home, before the first layer: left out
G1 X50 Y50 F3000
;LAYER:0
G0 F6000 X10 Y10 Z0.2
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;TYPE:FILL
G1 X10 Y20 E3.0
G0 F6000 X10 Y10
;TIME_ELAPSED:10.0
;LAYER:1
G0 X10 Y10 Z0.4
G91
M83
;TYPE:SKIN
G1 F600 X10 E0.5
G1 Y10 E0.5
G90
G92 E0
;TIME_ELAPSED:25.0
"""


@pytest.fixture
def preview():
    return gp.parse(GCODE.splitlines())


def test_layers_and_heights(preview):
    assert [layer.number for layer in preview.layers] == [0, 1]
    assert [layer.z for layer in preview.layers] == pytest.approx([0.2, 0.4])


def test_kinds_and_travel(preview):
    first = preview.layers[0]
    assert [gp.KIND_NAMES[k] for k in first.kind] == \
        ["travel", "WALL-OUTER", "WALL-OUTER", "FILL", "travel"]
    assert first.seg[1].tolist() == pytest.approx([10, 10, 20, 10])


def test_relative_moves_and_extrusion(preview):
    second = preview.layers[1]
    assert [gp.KIND_NAMES[k] for k in second.kind] == ["SKIN", "SKIN"]
    assert second.seg.ravel().tolist() == pytest.approx([10, 10, 20, 10, 20, 10, 20, 20])


def test_times_follow_time_elapsed(preview):
    assert preview.layers[0].seconds == pytest.approx(10.0)
    assert preview.layers[1].seconds == pytest.approx(15.0)
    assert preview.seconds == pytest.approx(25.0)
    t = preview.layers[1].t
    assert t[0] == pytest.approx(7.5)            # two moves of the same length and speed


def test_position_along_the_layer(preview):
    layer = preview.layers[1]
    pos, done = layer.position(3.75)
    assert pos == pytest.approx((15.0, 10.0)) and done == 0
    pos, done = layer.position(100.0)
    assert pos == pytest.approx((20.0, 20.0)) and done == 2


def test_box_of_what_is_printed(preview):
    assert preview.lo == pytest.approx((10.0, 10.0))
    assert preview.hi == pytest.approx((20.0, 20.0))


def test_empty_gcode():
    p = gp.parse([])
    assert p.layers == [] and p.seconds == 0.0
