"""Print Check also says what needs support.

`ops.check_parts()` gets an `overhang_limit` (the degrees of *Window > Preferences >
Drawing*); with it every valid part is measured as it stands, straight down, and a part whose
triangles hang over more steeply than the limit gets an info message with the area that a
printer has to support. When Orient for Printing can bring that area down to at most 75 % of
what it is now, the same message says so. Without the limit nothing is said about support, so
every caller that does not ask for it hears nothing new. These tests (pure ops, no window)
check:

* a cube on the bed: nothing about support;
* a T (a plate of 30 × 30 × 5 on a pillar of 10 × 10 × 20) that stands on its pillar:
  8.0 cm² of support, the underside of the plate, and the advice to orient it;
* the same T with the plate on the bed: nothing hangs over, so no message;
* an I-beam on its side: 2.0 cm² of support that no turning rids it of, so no advice;
* a part that is not a closed volume: nothing about support;
* `check_parts` without a limit: nothing about support at all.

Run with:  python -m pytest tests/test_print_check_support.py
"""
import math
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import manifold_from_mesh, mesh_arrays, rotation_matrix  # noqa: E402
from volura.model import Document  # noqa: E402

LIMIT = 45


def t_shape(plate_on_bed=False):
    """A plate of 30 × 30 × 5 mm centred on a pillar of 10 × 10 × 20 mm: a T.

    It stands on its pillar, so the underside of the plate hangs over; `plate_on_bed` puts
    the plate on the bed with the pillar on top, the way Orient for Printing would lay it."""
    plate = mf.Manifold.cube((30.0, 30.0, 5.0)).translate((0.0, 0.0, 0.0 if plate_on_bed else 20.0))
    pillar = mf.Manifold.cube((10.0, 10.0, 20.0)).translate((10.0, 10.0, 5.0 if plate_on_bed else 0.0))
    return ops.union_all([plate, pillar])


def i_beam_on_its_side():
    """Two plates of 30 × 30 × 5 with a web between them, laid on its side.

    Standing upright it hangs over 8.0 cm² and lying on its side 2.0 cm², and that is the
    least it can do in every way it can stand: its support cannot be turned away."""
    lo = mf.Manifold.cube((30.0, 30.0, 5.0))
    web = mf.Manifold.cube((10.0, 10.0, 20.0)).translate((10.0, 10.0, 5.0))
    hi = mf.Manifold.cube((30.0, 30.0, 5.0)).translate((0.0, 0.0, 25.0))
    V, T = mesh_arrays(ops.union_all([lo, web, hi]))
    V = V @ rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi / 2).T
    V -= V.min(axis=0)                       # drop it on the bed
    return manifold_from_mesh(V, T)


def parts_of(man, name="Part"):
    """One group holding `man`, in an empty model, as a print check sees it."""
    doc = Document()
    ops.group_from_manifold(doc, man, "#f2f2f2", name)
    return ops.parts(doc)


def open_box_parts():
    """The parts of a model with a box of 20 mm that lost one face, so it is no volume."""
    doc = Document()
    ents = Entities.from_manifold(mf.Manifold.cube((20.0, 20.0, 20.0)))
    ents.remove([next(iter(ents.F))], [])
    doc.add_group(ents, "#f2f2f2", "Open")
    return ops.parts(doc)


def check(man, name="Part", limit=LIMIT):
    """The messages of a print check on one group holding `man`."""
    return ops.check_parts(parts_of(man, name), None, limit)


def support_msgs(msgs):
    """The messages that talk about support."""
    return [m for _level, m in msgs if "hangs over" in m]


def supported_area(text):
    """The cm² of a support message."""
    return float(re.search(r"([\d.]+) cm²", text).group(1))


def test_a_cube_on_the_bed_is_told_nothing_about_support():
    assert support_msgs(check(mf.Manifold.cube((20.0, 20.0, 20.0)), "Cube")) == []


def test_a_t_that_stands_on_its_pillar_reports_the_plate_it_has_to_support():
    man = t_shape()
    msgs = support_msgs(check(man, "T"))
    assert len(msgs) == 1
    assert msgs[0].startswith("T: ") and "more than 45°" in msgs[0]
    V, T = mesh_arrays(man)
    want = ops.overhang_area(V, T, ops.DOWN_DIR, LIMIT) / 100.0
    assert supported_area(msgs[0]) == pytest.approx(want, rel=0.01)
    assert supported_area(msgs[0]) == pytest.approx(8.0, abs=0.05), "the underside of the plate"


def test_a_t_whose_plate_lies_on_the_bed_hangs_over_nothing():
    assert support_msgs(check(t_shape(plate_on_bed=True), "T")) == []


def test_a_t_that_prints_far_better_on_its_plate_is_told_to_orient_it():
    assert "Orient for Printing can bring that down to 0.0 cm²." in support_msgs(check(t_shape(), "T"))[0]


def test_a_part_that_no_turning_rids_of_its_support_gets_no_advice():
    msgs = support_msgs(check(i_beam_on_its_side(), "Beam"))
    assert len(msgs) == 1
    assert supported_area(msgs[0]) == pytest.approx(2.0, abs=0.05)
    assert "Orient for Printing" not in msgs[0], "turning it gains less than a quarter"


def test_a_part_that_is_not_a_closed_volume_is_told_nothing_about_support():
    parts = open_box_parts()
    assert parts[0].problem, "the box is meant to be open"
    assert support_msgs(ops.check_parts(parts, None, LIMIT)) == []


def test_without_a_limit_nothing_is_said_about_support():
    parts = parts_of(t_shape(), "T")
    assert support_msgs(check(t_shape(), "T", limit=None)) == []
    assert support_msgs(ops.check_parts(parts)) == []
