"""The exact result of ``Entities.from_manifold``, kept bit for bit the same.

``from_manifold`` numbers the vertices, edges and faces of an imported mesh in a
particular order: a region first takes the ids of the vertices it is the first to
touch, then those of the edges its loops add, then its own face id. Tools and tests
refer to those ids, so a faster implementation may change how it gets there but not
what comes out. The whole result is therefore fingerprinted here - every vertex
position, every edge with its end points and its soft and hidden flags, every face
with its loops and its normal - and the hash of that dump is stored per mesh.
"""
import hashlib
import os
import sys

import manifold3d as mf
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities  # noqa: E402


def sphere():
    return mf.Manifold.sphere(30, 64)


def cube_with_hole():
    """A block with a square tunnel through it: faces with a hole loop."""
    return (mf.Manifold.cube([40.0, 30.0, 20.0], True) -
            mf.Manifold.cube([10.0, 40.0, 10.0], True))


def refined_cube():
    """A cube whose faces are split into a grid: few regions, long loops."""
    return mf.Manifold.cube([25.0, 25.0, 25.0], True).refine(4)


def offset_cubes():
    """Fifty cubes pushed into each other: many regions that share vertices."""
    out = None
    for i in range(50):
        c = mf.Manifold.cube([8.0, 8.0, 8.0], True).translate(
            [5.0 * (i % 7), 5.0 * (i // 7), 2.0 * (i % 3)])
        out = c if out is None else out + c
    return out


def cylinder():
    """Soft seams and a rounded wall next to flat caps."""
    return mf.Manifold.cylinder(30.0, 12.0, 12.0, 24)


def dump(ents):
    """A canonical text of everything ``from_manifold`` decides about a mesh."""
    lines = []
    for v in sorted(ents.V):
        p = np.round(ents.V[v], 9)
        lines.append("V %d %.9f %.9f %.9f" % (v, p[0], p[1], p[2]))
    for e in sorted(ents.E):
        a, b = ents.E[e]
        lines.append("E %d %d %d %d %d" % (e, a, b, int(e in ents.soft), int(e in ents.hidden_e)))
    for f in sorted(ents.F):
        face = ents.F[f]
        loops = " ".join("(" + ",".join(str(v) for v in loop) + ")" for loop in face.loops)
        n = np.round(face.normal, 9)
        lines.append("F %d %s %.9f %.9f %.9f %d"
                     % (f, loops, n[0], n[1], n[2], int(f in ents.hidden_f)))
    lines.append("next %d" % ents.next_id)
    return "\n".join(lines)


def fingerprint(ents):
    return hashlib.sha256(dump(ents).encode()).hexdigest()


EXPECTED = {
    "sphere": "3f2ddb6d3fc3ca08d662deff2f46cb793e67c510902d8839d9af3b7472363224",   # 1026/3072/2048
    "cube with a hole": "536f9f9b211c940e066c90154fb742dc9ce74c413e90e50b2f00b53556e87024",
    "refined cube": "b5a6d1b7e8665a2b3551fbd9af13b32e49c061b23f4c1775bf7556efb2f95461",
    "fifty offset cubes": "9efc92c1a988726a23cf511c28fbf2d2a56600711a0363363399976715def0c6",
    "cylinder": "6fe5964592118055f6803d4c58c9dca88c365e867e352522618495ed5e709526",
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_from_manifold_fingerprint(name):
    man = {"sphere": sphere, "cube with a hole": cube_with_hole, "refined cube": refined_cube,
           "fifty offset cubes": offset_cubes, "cylinder": cylinder}[name]()
    ents = Entities.from_manifold(man)
    assert ents.F, "nothing came out of %s" % name
    assert fingerprint(ents) == EXPECTED[name], "the result of %s changed" % name


if __name__ == "__main__":
    fails = 0
    for name in sorted(EXPECTED):
        man = {"sphere": sphere, "cube with a hole": cube_with_hole,
               "refined cube": refined_cube, "fifty offset cubes": offset_cubes,
               "cylinder": cylinder}[name]()
        try:
            got = fingerprint(Entities.from_manifold(man))
            assert got == EXPECTED[name], "the result of %s changed" % name
            print("ok  ", name)
        except Exception as ex:
            fails += 1
            print("FAIL", name, type(ex).__name__, ex)
    print("failures:", fails)
