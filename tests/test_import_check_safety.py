"""Safety net for STL import and the print check: pins the current results, so that faster
code (caches, bulk building, background work) cannot change what an import or a check gives."""
import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.geom import mesh_arrays  # noqa: E402
from volura.model import Document  # noqa: E402

SHAPES = {
    "cube": lambda: mf.Manifold.cube((20, 20, 20)),
    "sphere": lambda: mf.Manifold.sphere(15, 48).translate((0, 0, 15)),
    "tube": lambda: mf.Manifold.cylinder(20, 8, 8, 32) - mf.Manifold.cylinder(20, 4, 4, 32),
}


def soup(man):
    """The triangles of a Manifold as an (n, 3, 3) array, as an STL holds them."""
    v, t = mesh_arrays(man)
    return v[t]


def write_ascii(path, tris, fmt="{:.9f}"):
    lines = ["solid test"]
    for a, b, c in tris:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += ["      vertex " + " ".join(fmt.format(x) for x in p) for p in (a, b, c)]
        lines += ["    endloop", "  endfacet"]
    lines.append("endsolid test")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def write_binary(path, tris):
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = np.asarray(tris, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)) + rec.tobytes())
    return str(path)


def import_as_group(doc, path):
    """What the app does on File > Import STL."""
    return ops.group_from_manifold(doc, fileio.import_stl(path), "#f2f2f2", "Imported")


def kinds(msgs):
    return [k for k, _ in msgs]


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_export_then_import_keeps_the_model(shape, tmp_path):
    doc = Document()
    man = SHAPES[shape]()
    ops.group_from_manifold(doc, man, "#f2f2f2", "Shape")
    path = str(tmp_path / "shape.stl")
    fileio.export_stl(path, ops.parts(doc))

    doc2 = Document()
    g = import_as_group(doc2, path)
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() == pytest.approx(man.volume(), rel=1e-5)
    assert np.asarray(m.bounding_box()) == pytest.approx(np.asarray(man.bounding_box()), abs=1e-4)
    assert kinds(ops.check_parts(ops.parts(doc2))) == ["ok"]


def test_ascii_and_binary_stl_read_the_same(tmp_path):
    tris = soup(SHAPES["tube"]())
    va, ta = fileio.read_stl(write_ascii(tmp_path / "a.stl", tris))
    vb, tb = fileio.read_stl(write_binary(tmp_path / "b.stl", tris))
    assert len(ta) == len(tb) == len(tris)
    assert va == pytest.approx(vb, abs=1e-5)
    ma = fileio.import_stl(str(tmp_path / "a.stl"))
    mb = fileio.import_stl(str(tmp_path / "b.stl"))
    assert ma.volume() == pytest.approx(mb.volume(), rel=1e-6)
    assert ma.num_vert() == mb.num_vert()


def test_corners_that_differ_by_1e7_are_merged(tmp_path):
    tris = soup(mf.Manifold.cube((20, 20, 20))).copy()
    tris[::2, 0] += 1e-7                       # every other triangle: one corner nudged
    m = fileio.import_stl(write_ascii(tmp_path / "near.stl", tris, "{:.10f}"))
    assert m.num_vert() == 8
    assert m.volume() == pytest.approx(8000.0, rel=1e-6)


def test_triangles_listed_twice_become_two_shells(tmp_path):
    # current behaviour, not necessarily the wanted one: manifold3d accepts the doubled
    # triangles as two coincident cubes, so the volume counts twice
    tris = soup(mf.Manifold.cube((20, 20, 20)))
    m = fileio.import_stl(write_ascii(tmp_path / "twice.stl", np.concatenate([tris, tris])))
    assert m.num_vert() == 16 and m.num_tri() == 24
    assert m.volume() == pytest.approx(16000.0, rel=1e-6)


def test_an_open_stl_is_refused(tmp_path):
    tris = soup(mf.Manifold.cube((20, 20, 20)))[:-1]
    with pytest.raises(ValueError, match="not watertight"):
        fileio.import_stl(write_ascii(tmp_path / "open.stl", tris))
    with pytest.raises(ValueError, match="not watertight"):
        fileio.import_stl(write_binary(tmp_path / "open_bin.stl", tris))


def test_an_empty_stl_is_refused(tmp_path):
    path = write_ascii(tmp_path / "empty.stl", [])
    with pytest.raises(ValueError, match="no triangles"):
        fileio.import_stl(path)


def test_check_on_a_good_model_reports_only_ok():
    doc = Document()
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)), "#f2f2f2", "Cube")
    ops.group_from_manifold(doc, SHAPES["sphere"]().translate((50, 0, 0)), "#d23b3b", "Ball")
    msgs = ops.check_parts(ops.parts(doc))
    assert kinds(msgs) == ["ok", "ok"]
    assert "Cube" in msgs[0][1] and "8.00 cm³" in msgs[0][1]
    assert "Ball" in msgs[1][1]


def test_check_reports_overlap_of_different_colors_only():
    doc = Document()
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)), "#f2f2f2", "A")
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)).translate((10, 0, 0)), "#d23b3b", "B")
    msgs = ops.check_parts(ops.parts(doc))
    info = [m for k, m in msgs if k == "info"]
    assert len(info) == 1
    assert "A and B overlap (4000.00 mm³)" in info[0]

    doc = Document()
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)), "#f2f2f2", "A")
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)).translate((10, 0, 0)), "#f2f2f2", "B")
    assert "info" not in kinds(ops.check_parts(ops.parts(doc)))


def test_check_follows_a_moved_imported_group(tmp_path):
    path = write_binary(tmp_path / "cube.stl", soup(mf.Manifold.cube((20, 20, 20))))
    doc = Document()
    g = import_as_group(doc, path)
    sel = [("g", g.id, g.id)]
    bed = (100, 100, 100)
    assert kinds(ops.check_parts(ops.parts(doc), bed)) == ["ok"]

    doc.checkpoint()
    ops.transform_selection(doc, np.eye(3), (0.0, 0.0, -5.0), items=sel)
    msgs = ops.check_parts(ops.parts(doc), bed)
    assert any(k == "waarschuwing" and "5.00 mm below the print bed" in m for k, m in msgs)

    ops.transform_selection(doc, np.eye(3), (0.0, 0.0, 15.0), items=sel)
    msgs = ops.check_parts(ops.parts(doc), bed)
    assert any(k == "waarschuwing" and "floats 10.00 mm above the bed" in m for k, m in msgs)

    ops.transform_selection(doc, np.eye(3), (90.0, 0.0, -10.0), items=sel)
    msgs = ops.check_parts(ops.parts(doc), bed)
    assert any(k == "waarschuwing" and "outside the print volume" in m for k, m in msgs)

    doc.undo()
    assert kinds(ops.check_parts(ops.parts(doc), bed)) == ["ok"]
