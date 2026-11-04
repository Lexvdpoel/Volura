"""Safety net for geometry after edits: after every operation the entities must equal a
copy rebuilt from their saved state, and every face's triangulation must cover exactly
the face. This catches a stale cache in faster code (triangulation, bulk building)."""
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


def block():
    doc = Document()
    g = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), "#f2f2f2", "Block")
    return doc, g


def fresh(ents):
    return Entities.from_state(ents.state())


def loop_area(ents, loop, n):
    p = np.array([ents.V[v] for v in loop])
    return abs(float(np.dot(n, np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))) / 2


def face_area(ents, fid):
    """Area of a face from its loops: the outer loop minus the holes."""
    face = ents.F[fid]
    areas = [loop_area(ents, loop, face.normal) for loop in face.loops]
    return areas[0] - sum(areas[1:])


def tri_area(ents, fid):
    t = np.asarray(ents.face_triangles(fid)[0], dtype=float).reshape(-1, 3, 3)
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()) / 2


def total_area(ents):
    return sum(tri_area(ents, f) for f in ents.F)


def check(ents):
    """Every face triangulates to its own area, and a rebuilt copy is the same solid."""
    for f in ents.F:
        assert tri_area(ents, f) == pytest.approx(face_area(ents, f), rel=1e-6, abs=1e-9), f
    copy = fresh(ents)
    assert (len(copy.F), len(copy.E), len(copy.V)) == (len(ents.F), len(ents.E), len(ents.V))
    m, mc = ents.to_manifold(), copy.to_manifold()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert mc.volume() == pytest.approx(m.volume(), rel=1e-9)
    assert total_area(copy) == pytest.approx(total_area(ents), rel=1e-9)
    assert copy.bbox() == pytest.approx(ents.bbox(), abs=1e-9)
    return m.volume()


def top_faces(ents, z):
    return [f for f, face in ents.F.items() if face.normal[2] > 0.9
            and abs(ents.V[face.loops[0][0]][2] - z) < 1e-9]


def half(ents, z, front):
    """The top face at height z on the front (y < 15) or back side of the block."""
    for f in top_faces(ents, z):
        y = np.mean([ents.V[v][1] for v in ents.F[f].loops[0]])
        if (y < 15) == front:
            return f
    raise AssertionError("no such face")


def split_top(ents):
    ents.add_line((0, 15, 20), (40, 15, 20))


def test_a_line_splits_the_top_face():
    _, g = block()
    assert len(top_faces(g.ents, 20)) == 1
    split_top(g.ents)
    assert len(top_faces(g.ents, 20)) == 2
    assert check(g.ents) == pytest.approx(24000.0)
    assert total_area(g.ents) == pytest.approx(2 * (40 * 30 + 40 * 20 + 30 * 20))


def test_pushpull_up_and_down():
    _, g = block()
    split_top(g.ents)
    cap = g.ents.pushpull(half(g.ents, 20, front=True), 10.0)
    assert check(g.ents) == pytest.approx(24000.0 + 40 * 15 * 10)
    g.ents.pushpull(cap, -5.0)
    assert check(g.ents) == pytest.approx(24000.0 + 40 * 15 * 5)
    g.ents.pushpull(half(g.ents, 20, front=False), -5.0)
    assert check(g.ents) == pytest.approx(24000.0)


def test_transform_moves_and_turns():
    _, g = block()
    split_top(g.ents)
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    g.ents.transform(rz, (100.0, 5.0, 0.0))
    assert check(g.ents) == pytest.approx(24000.0)
    assert g.ents.bbox() == pytest.approx(np.array([[70.0, 5.0, 0.0], [100.0, 45.0, 20.0]]))
    # a pushpull after the transform uses the moved geometry
    g.ents.pushpull(top_faces(g.ents, 20)[0], 10.0)
    assert check(g.ents) == pytest.approx(24000.0 + 40 * 15 * 10)


def test_undo_and_redo_restore_volume_and_area():
    doc, g = block()
    states = [(check(g.ents), total_area(g.ents))]
    doc.checkpoint()
    split_top(g.ents)
    states.append((check(g.ents), total_area(g.ents)))
    doc.checkpoint()
    g.ents.pushpull(half(g.ents, 20, front=True), 10.0)
    states.append((check(g.ents), total_area(g.ents)))
    assert states[2][0] == pytest.approx(30000.0)

    for want in (states[1], states[0]):
        assert doc.undo()
        ents = doc.groups[0].ents
        assert (check(ents), total_area(ents)) == pytest.approx(want)
    for want in (states[1], states[2]):
        assert doc.redo()
        ents = doc.groups[0].ents
        assert (check(ents), total_area(ents)) == pytest.approx(want)


def test_twenty_edits_in_a_row():
    _, g = block()
    volume = 24000.0
    for i in range(10):                          # ten strips across the top, 3 mm apart
        y = 1.5 + 3 * i
        g.ents.add_line((0, y, 20), (40, y, 20))
    check(g.ents)
    for i, f in enumerate(sorted(top_faces(g.ents, 20),
                                 key=lambda f: np.mean([g.ents.V[v][1] for v in g.ents.F[f].loops[0]]))):
        if i % 2 or i >= 20:
            continue
        a = face_area(g.ents, f)
        g.ents.pushpull(f, 2.0 + i)
        volume += a * (2.0 + i)
    assert check(g.ents) == pytest.approx(volume)


def test_a_cylinder_keeps_its_soft_side():
    doc = Document()
    g = ops.group_from_manifold(doc, mf.Manifold.cylinder(20, 10, 10, 48), "#f2f2f2", "Pin")
    v0 = check(g.ents)
    top = top_faces(g.ents, 20)[0]
    g.ents.pushpull(top, 5.0)
    assert check(g.ents) == pytest.approx(v0 * 25 / 20, rel=1e-9)
