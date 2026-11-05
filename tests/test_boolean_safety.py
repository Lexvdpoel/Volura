"""Safety net for the boolean operations (union, subtract, intersect).

Two cubes 20x20x20, the second shifted by (10, 0, 0), in two colors,
and where meaningful a sphere.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.model import Document  # noqa: E402


def cube_20():
    """A 20x20x20 cube."""
    return mf.Manifold.cube((20, 20, 20))


def cube_20_shifted():
    """A 20x20x20 cube shifted by (10, 0, 0)."""
    return cube_20().translate((10, 0, 0))


def doc_two_cubes():
    """Create a document with two overlapping cubes in different colors."""
    doc = Document()
    a_ents = Entities.from_manifold(cube_20(), color="#f2f2f2")
    a = doc.add_group(a_ents, "#f2f2f2", "A")
    b_ents = Entities.from_manifold(cube_20_shifted(), color="#2f6fd6")
    b = doc.add_group(b_ents, "#2f6fd6", "B")
    return doc, a, b


def volume(ents):
    """Get the volume of an Entities object."""
    m = ents.to_manifold()
    assert m.status().name == "NoError", m.status()
    return round(m.volume(), 2)


def bbox(ents):
    """Get bounding box as (min, max) of (x, y, z)."""
    return ents.bbox()


def test_union_two_overlapping_cubes_gives_correct_volume():
    """union: volume 20*20*20 + 10*20*20 = 12000."""
    doc, a, b = doc_two_cubes()
    assert volume(a.ents) == 8000.0
    assert volume(b.ents) == 8000.0
    result = ops.boolean(doc, [a, b], "union")
    assert result is a
    assert len(doc.groups) == 1
    assert volume(result.ents) == pytest.approx(12000.0, rel=1e-6)


def test_union_keeps_name_and_color_of_first_group():
    """After union, the result has the name and color of group a."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "union")
    assert result.name == "A"
    assert result.color == "#f2f2f2"


def test_union_bounding_box():
    """Union bounding box: (0,0,0)-(30,20,20)."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "union")
    min_b, max_b = bbox(result.ents)
    assert min_b[0] == pytest.approx(0.0, abs=1e-6)
    assert min_b[1] == pytest.approx(0.0, abs=1e-6)
    assert min_b[2] == pytest.approx(0.0, abs=1e-6)
    assert max_b[0] == pytest.approx(30.0, abs=1e-6)
    assert max_b[1] == pytest.approx(20.0, abs=1e-6)
    assert max_b[2] == pytest.approx(20.0, abs=1e-6)


def test_union_parts_gives_one_part_without_problems():
    """ops.parts(doc) gives one part without problem."""
    doc, a, b = doc_two_cubes()
    ops.boolean(doc, [a, b], "union")
    parts = ops.parts(doc)
    assert len(parts) == 1
    assert parts[0].problem is None


def test_subtract_two_overlapping_cubes_gives_correct_volume():
    """subtract: volume 4000 and bounding box (0,0,0)-(10,20,20)."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "subtract")
    assert result is a
    assert len(doc.groups) == 1
    assert volume(result.ents) == pytest.approx(4000.0, rel=1e-6)


def test_subtract_bounding_box():
    """Subtract bounding box: (0,0,0)-(10,20,20)."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "subtract")
    min_b, max_b = bbox(result.ents)
    assert min_b[0] == pytest.approx(0.0, abs=1e-6)
    assert min_b[1] == pytest.approx(0.0, abs=1e-6)
    assert min_b[2] == pytest.approx(0.0, abs=1e-6)
    assert max_b[0] == pytest.approx(10.0, abs=1e-6)
    assert max_b[1] == pytest.approx(20.0, abs=1e-6)
    assert max_b[2] == pytest.approx(20.0, abs=1e-6)


def test_intersect_two_overlapping_cubes_gives_correct_volume():
    """intersect: volume 4000 and bounding box (10,0,0)-(20,20,20)."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "intersect")
    assert result is a
    assert len(doc.groups) == 1
    assert volume(result.ents) == pytest.approx(4000.0, rel=1e-6)


def test_intersect_bounding_box():
    """Intersect bounding box: (10,0,0)-(20,20,20)."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "intersect")
    min_b, max_b = bbox(result.ents)
    assert min_b[0] == pytest.approx(10.0, abs=1e-6)
    assert min_b[1] == pytest.approx(0.0, abs=1e-6)
    assert min_b[2] == pytest.approx(0.0, abs=1e-6)
    assert max_b[0] == pytest.approx(20.0, abs=1e-6)
    assert max_b[1] == pytest.approx(20.0, abs=1e-6)
    assert max_b[2] == pytest.approx(20.0, abs=1e-6)


def test_subtract_full_overlap_gives_none_and_removes_group():
    """A subtract that removes everything gives None and leaves no group of a behind."""
    doc = Document()
    # Create two identical cubes (b completely covers a)
    a_ents = Entities.from_manifold(cube_20(), color="#f2f2f2")
    a = doc.add_group(a_ents, "#f2f2f2", "A")
    b_ents = Entities.from_manifold(cube_20(), color="#2f6fd6")
    b = doc.add_group(b_ents, "#2f6fd6", "B")
    assert len(doc.groups) == 2
    result = ops.boolean(doc, [a, b], "subtract")
    assert result is None
    assert len(doc.groups) == 0


def test_undo_after_boolean_restores_two_groups():
    """Undo: with checkpoint() before the operation, undo gives two groups again."""
    doc, a, b = doc_two_cubes()
    before_count = len(doc.groups)
    a_vol_before = volume(a.ents)
    b_vol_before = volume(b.ents)
    a_name_before = a.name
    a_color_before = a.color
    b_name_before = b.name
    b_color_before = b.color
    doc.checkpoint()
    ops.boolean(doc, [a, b], "union")
    assert len(doc.groups) == 1
    assert volume(doc.groups[0].ents) == pytest.approx(12000.0, rel=1e-6)
    doc.undo()
    assert len(doc.groups) == before_count
    groups_by_id = {g.id: g for g in doc.groups}
    a_after = groups_by_id[a.id]
    b_after = groups_by_id[b.id]
    assert volume(a_after.ents) == pytest.approx(a_vol_before, rel=1e-6)
    assert volume(b_after.ents) == pytest.approx(b_vol_before, rel=1e-6)
    assert a_after.name == a_name_before
    assert a_after.color == a_color_before
    assert b_after.name == b_name_before
    assert b_after.color == b_color_before


def test_redo_after_boolean_restores_union():
    """Redo: after undo, redo gives the union result again."""
    doc, a, b = doc_two_cubes()
    doc.checkpoint()
    ops.boolean(doc, [a, b], "union")
    doc.undo()
    assert len(doc.groups) == 2
    doc.redo()
    assert len(doc.groups) == 1
    assert volume(doc.groups[0].ents) == pytest.approx(12000.0, rel=1e-6)


def test_subtract_from_open_volume_raises_value_error():
    """A group that is not a closed volume gives a ValueError with the group name."""
    doc = Document()
    # Create a cube and remove one face
    ents = Entities.from_manifold(cube_20(), color="#f2f2f2")
    # Get the first face and remove it
    first_face = list(ents.F.keys())[0]
    ents.remove([first_face], [])
    g = doc.add_group(ents, "#f2f2f2", "open_cube")
    # Add another cube for the boolean operation
    b_ents = Entities.from_manifold(cube_20_shifted(), color="#2f6fd6")
    b = doc.add_group(b_ents, "#2f6fd6", "B")
    before_count = len(doc.groups)
    b_vol_before = volume(b.ents)
    with pytest.raises(ValueError) as exc_info:
        ops.boolean(doc, [g, b], "union")
    assert "open_cube" in str(exc_info.value)
    assert len(doc.groups) == before_count
    assert volume(b.ents) == pytest.approx(b_vol_before, rel=1e-6)


def test_union_result_can_be_pushpulled():
    """After a union, pushpull on the top face: volume 12000 + 30*20*5."""
    doc, a, b = doc_two_cubes()
    result = ops.boolean(doc, [a, b], "union")
    initial_volume = volume(result.ents)
    assert initial_volume == pytest.approx(12000.0, rel=1e-6)
    # Get the top face (highest z, positive normal in z)
    top_face = None
    max_z = -float("inf")
    for fid, face in result.ents.F.items():
        normal = face.normal
        if normal[2] > 0.9:
            pts = result.ents.face_points(fid)
            z_max = np.max([p[2] for p in pts])
            if z_max > max_z:
                max_z = z_max
                top_face = fid
    assert top_face is not None
    # Pushpull the top face by 5
    new_face_id = result.ents.pushpull(top_face, 5)
    assert new_face_id is not None
    final_volume = volume(result.ents)
    expected_volume = 12000.0 + 30 * 20 * 5
    assert final_volume == pytest.approx(expected_volume, rel=1e-6)
