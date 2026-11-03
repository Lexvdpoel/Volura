"""Entities.reverse_faces and Entities.orient_faces (SketchUp right-click on a face).

The box is built exactly like box() in tests/test_brep.py (40 x 30 x 10 mm,
volume 12000 mm3).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_brep import box, face_at  # noqa: E402


def volume_or_status(ents):
    m = ents.to_manifold()
    if m.status().name != "NoError":
        return m.status().name
    return round(m.volume(), 2)


def test_reverse_faces_flips_the_normal():
    e = box()
    top = face_at(e, 20, 15, 10)
    assert e.F[top].normal[2] > 0
    e.reverse_faces([top])
    assert e.F[top].normal[2] < 0
    assert np.allclose(e.F[top].normal, [0, 0, -1])


def test_reverse_faces_reverses_the_loop_winding():
    e = box()
    top = face_at(e, 20, 15, 10)
    before = list(e.F[top].loops[0])
    e.reverse_faces([top])
    assert list(e.F[top].loops[0]) == list(reversed(before))


def test_reverse_faces_makes_the_box_volume_invalid_or_negative():
    e = box()
    assert volume_or_status(e) == 12000.0
    e.reverse_faces([face_at(e, 20, 15, 10)])
    v = volume_or_status(e)
    assert v != 12000.0 and (isinstance(v, str) or v < 0), f"verwacht ongeldig/negatief, kreeg {v}"


def test_orient_faces_repairs_a_single_flipped_face():
    e = box()
    e.reverse_faces([face_at(e, 20, 15, 10)])
    bottom = face_at(e, 20, 15, 0)
    e.orient_faces(bottom)
    m = e.to_manifold()
    assert m.status().name == "NoError", m.status()
    assert round(m.volume(), 2) == 12000.0


def test_orient_faces_keeps_a_consistent_box_unchanged():
    e = box()
    normals = {f: e.F[f].normal.copy() for f in e.F}
    e.orient_faces(face_at(e, 20, 15, 0))
    assert all(np.allclose(e.F[f].normal, n) for f, n in normals.items())
    assert volume_or_status(e) == 12000.0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as ex:
                fails += 1
                print("FAIL", name, type(ex).__name__, ex)
    print("failures:", fails)
