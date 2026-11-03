"""The computing core of "Cut and Join" (volura/joints.py).

A 40 x 20 x 20 mm bar is cut in the middle (x = 20, normal +X); then the joints
are checked. Expected volumes: one half is 8000 mm³, the whole block 16000 mm³.
The joint kinds ("geen", "pennen", ...) are internal keys of joints.KINDS.
"""
import os
import sys

import manifold3d as mf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import joints  # noqa: E402


def block():
    return mf.Manifold.cube((40, 20, 20))


ORIGIN = (20.0, 0.0, 0.0)
NORMAL = (1.0, 0.0, 0.0)


def volume(m):
    assert m.status() == mf.Error.NoError, m.status()
    return m.volume()


# ------------------------------------------------------------------ cutting

def test_cut_without_joint_keeps_the_total_volume():
    a, b, extra = joints.split_with_joint(block(), ORIGIN, NORMAL, "geen")
    assert extra == []
    assert abs(volume(a) - 8000) < 1e-6
    assert abs(volume(b) - 8000) < 1e-6
    assert abs(volume(a) + volume(b) - 16000) < 1e-6


def test_cut_puts_part_a_at_the_normal_side():
    a, b = joints.cut(block(), ORIGIN, NORMAL)
    assert a.bounding_box()[0] >= 20.0 - 1e-6      # part_a lies at the +X side
    assert b.bounding_box()[3] <= 20.0 + 1e-6


# ------------------------------------------------------------------ pins

def test_pins_adds_a_pin_to_a_and_a_hole_to_b():
    params = {"pin_diameter": 5.0, "pin_length": 10.0, "count": "auto", "wall": 2.0}
    a, b, extra = joints.split_with_joint(block(), ORIGIN, NORMAL, "pennen", params)
    assert extra == []
    assert volume(a) > 8000                        # pin added
    assert volume(b) < 8000                        # hole removed
    assert (a ^ b).volume() < 1e-6                 # the pin fits the hole


# ------------------------------------------------------------------ loose pins

def test_loose_pins_make_extra_parts_and_holes_in_both():
    params = {"pin_diameter": 5.0, "pin_length": 10.0, "count": "auto", "wall": 2.0}
    a, b, extra = joints.split_with_joint(block(), ORIGIN, NORMAL, "losse pennen", params)
    assert len(extra) >= 1                         # loose pins are added
    assert volume(a) < 8000 and volume(b) < 8000   # both parts get a hole
    for pin in extra:
        assert pin.status() == mf.Error.NoError
        assert pin.bounding_box()[2] >= -1e-6      # the pin lies on the bed


# ------------------------------------------------------------------ dovetail

def test_dovetail_parts_are_valid_and_do_not_overlap():
    params = {"dovetail_width": 8.0, "dovetail_depth": 5.0, "clearance": 0.2}
    a, b, extra = joints.split_with_joint(block(), ORIGIN, NORMAL, "zwaluwstaart", params)
    assert extra == []
    assert a.status() == mf.Error.NoError and b.status() == mf.Error.NoError
    assert (a ^ b).volume() < 1e-6                 # the parts slide into each other
    gap = (block() - joints.union_all([a, b])).volume()
    assert gap > 0                                  # there is clearance
    assert abs(volume(a) + volume(b) + gap - 16000) < 1.0


# ------------------------------------------------------------------ pin_positions

def _section_of(y, z):
    box = mf.Manifold.cube((5, y, z))
    return joints.section(box, (2.5, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_pin_positions_narrow_section_has_no_room():
    pts, warn = joints.pin_positions(_section_of(2, 20), radius=2.5, wall=2.0)
    assert pts == []
    assert warn


def test_pin_positions_square_gets_one_pin():
    pts, warn = joints.pin_positions(_section_of(20, 20), radius=2.5, wall=2.0)
    assert len(pts) == 1
    assert not warn


def test_pin_positions_long_section_gets_two_pins():
    pts, warn = joints.pin_positions(_section_of(20, 60), radius=2.5, wall=2.0)
    assert len(pts) == 2
    assert not warn


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
