"""Safety net for STL and 3MF export: pins the current results, so that faster code
(caches, bulk building, background work) cannot change what ends up in the files."""
import collections
import os
import re
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.model import Document  # noqa: E402

MATERIALS = [("White", "#f2f2f2"), ("Black", "#2b2b2b"), ("Red", "#d23b3b"), ("Blue", "#2f6fd6")]
SPHERE = mf.Manifold.sphere(15, 48).translate((60, 0, 15))
TUBE = (mf.Manifold.cylinder(20, 8, 8, 32) - mf.Manifold.cylinder(20, 4, 4, 32)).translate((0, 50, 0))


def single_cube():
    doc = Document()
    doc.materials = list(MATERIALS)
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)), "#f2f2f2", "Cube")
    return doc


def four_colors():
    """Cube, sphere, tube and a cube that overlaps the first one, in four colors."""
    doc = Document()
    doc.materials = list(MATERIALS)
    ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)), "#f2f2f2", "Cube")
    ops.group_from_manifold(doc, SPHERE, "#2b2b2b", "Sphere")
    ops.group_from_manifold(doc, TUBE, "#d23b3b", "Tube")
    ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)).translate((15, 15, 15)), "#2f6fd6", "Corner")
    return doc


def painted_block():
    doc = Document()
    doc.materials = list(MATERIALS)
    g = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), "#f2f2f2", "Block")
    for f, face in g.ents.F.items():
        if face.normal[2] > 0.9:
            g.ents.set_face_color(f, "#d23b3b")
    return doc


MODELS = {"single_cube": single_cube, "four_colors": four_colors, "painted_block": painted_block}


@pytest.fixture(params=sorted(MODELS))
def doc(request):
    return MODELS[request.param]()


def union_volume(doc):
    return ops.union_all([p.manifold for p in ops.parts(doc) if p.manifold is not None]).volume()


def mesh_volume(verts, tris):
    v = np.asarray(verts, dtype=float)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return float(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0)


def open_edges(verts, tris):
    """Edges not shared by exactly two triangles; STL repeats the corners of every
    triangle, so corners are matched on their position."""
    v = np.round(np.asarray(verts, dtype=float), 5)
    ids = {}
    key = [ids.setdefault(tuple(p), len(ids)) for p in v]
    count = collections.Counter()
    for a, b, c in np.asarray(tris, dtype=np.int64).reshape(-1, 3):
        for u, w in ((a, b), (b, c), (c, a)):
            count[tuple(sorted((key[u], key[w])))] += 1
    return sum(1 for n in count.values() if n != 2)


def stl_body(path):
    """The STL without its 80-byte header (the header holds the file name)."""
    with open(path, "rb") as f:
        return f.read()[80:]


def stl_bbox(path):
    v, _ = fileio.read_stl(path)
    v = np.asarray(v, dtype=float)
    return v.min(axis=0), v.max(axis=0)


def model_3mf(path):
    with zipfile.ZipFile(path) as z:
        return z.read("3D/3dmodel.model").decode(), z.read("Metadata/model_settings.config").decode(), z.namelist()


def objects_3mf(xml):
    """(vertices, triangles) of every mesh object in a 3MF model."""
    out = []
    for body in re.findall(r"<mesh>(.*?)</mesh>", xml, re.S):
        v = [tuple(float(x) for x in m) for m in
             re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', body)]
        t = [tuple(int(x) for x in m) for m in re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', body)]
        out.append((v, t))
    return out


def test_stl_is_closed_and_has_the_volume_of_the_model(doc, tmp_path):
    path = str(tmp_path / "model.stl")
    fileio.export_stl(path, ops.parts(doc))
    v, t = fileio.read_stl(path)
    assert len(t) > 0
    assert open_edges(v, t) == 0
    assert mesh_volume(v, t) == pytest.approx(union_volume(doc), rel=1e-6)


def test_3mf_parts_add_up_to_the_model(doc, tmp_path):
    path = str(tmp_path / "model.3mf")
    used = fileio.export_3mf(path, ops.parts(doc), doc.materials)
    xml, config, names = model_3mf(path)
    assert "3D/3dmodel.model" in names and "Metadata/model_settings.config" in names
    objects = objects_3mf(xml)
    assert len(objects) == len(ops.print_parts(ops.parts(doc)))
    for v, t in objects:
        assert open_edges(v, t) == 0
    assert sum(mesh_volume(v, t) for v, t in objects) == pytest.approx(union_volume(doc), rel=1e-6)
    # the colors in use are numbered 1..n without gaps
    assert sorted(e for _, e in used) == list(range(1, len(used) + 1))
    assert {int(x) for x in re.findall(r'<metadata key="extruder" value="(\d+)"', config)} \
        <= set(range(1, len(used) + 1))


def test_stl_per_color_adds_up_to_the_model(doc, tmp_path):
    files = fileio.export_stl_per_color(str(tmp_path / "model.stl"), ops.parts(doc), doc.materials)
    assert len(files) == len(ops.print_parts(ops.parts(doc)))
    total = 0.0
    for f in files:
        v, t = fileio.read_stl(f)
        assert open_edges(v, t) == 0
        total += mesh_volume(v, t)
    assert total == pytest.approx(union_volume(doc), rel=1e-6)


@pytest.mark.parametrize("fmt", ("stl", "3mf"))
def test_export_is_deterministic(doc, tmp_path, fmt):
    def export(name):
        path = str(tmp_path / f"{name}.{fmt}")
        if fmt == "stl":
            fileio.export_stl(path, ops.parts(doc))
            return stl_body(path)
        fileio.export_3mf(path, ops.parts(doc), doc.materials)
        with zipfile.ZipFile(path) as z:
            return {n: z.read(n) for n in z.namelist()}

    first = export("a")
    ops.parts(doc)
    assert export("b") == first


def test_export_follows_a_moved_group(tmp_path):
    doc = single_cube()
    g = doc.groups[0]
    doc.checkpoint()
    ops.transform_selection(doc, np.eye(3), (5.0, -3.0, 2.0), items=[("g", g.id, g.id)])
    path = str(tmp_path / "moved.stl")
    fileio.export_stl(path, ops.parts(doc))
    lo, hi = stl_bbox(path)
    assert lo == pytest.approx((5.0, -3.0, 2.0), abs=1e-6)
    assert hi == pytest.approx((25.0, 17.0, 22.0), abs=1e-6)
    v, t = fileio.read_stl(path)
    assert mesh_volume(v, t) == pytest.approx(8000.0, rel=1e-6)


def test_export_follows_pushpull_and_undo(tmp_path):
    doc = single_cube()
    g = doc.groups[0]
    top = next(f for f, face in g.ents.F.items() if face.normal[2] > 0.9)
    before = str(tmp_path / "before.stl")
    fileio.export_stl(before, ops.parts(doc))

    doc.checkpoint()
    g.ents.pushpull(top, 10.0)
    after = str(tmp_path / "after.stl")
    fileio.export_stl(after, ops.parts(doc))
    v, t = fileio.read_stl(after)
    assert mesh_volume(v, t) == pytest.approx(20 * 20 * 30, rel=1e-6)
    assert stl_bbox(after)[1][2] == pytest.approx(30.0)

    assert doc.undo()
    undone = str(tmp_path / "undone.stl")
    fileio.export_stl(undone, ops.parts(doc))
    assert stl_body(undone) == stl_body(before)

    assert doc.redo()
    redone = str(tmp_path / "redone.stl")
    fileio.export_stl(redone, ops.parts(doc))
    v, t = fileio.read_stl(redone)
    assert mesh_volume(v, t) == pytest.approx(20 * 20 * 30, rel=1e-6)


def test_3mf_follows_a_repainted_face(tmp_path):
    doc = painted_block()
    g = doc.groups[0]
    path = str(tmp_path / "a.3mf")
    fileio.export_3mf(path, ops.parts(doc), doc.materials)
    assert model_3mf(path)[0].count('paint_color="') == 2           # the red top: two triangles

    doc.checkpoint()
    for f, face in g.ents.F.items():
        if face.normal[2] > 0.9:
            g.ents.set_face_color(f, None)
    fileio.export_3mf(path, ops.parts(doc), doc.materials)
    assert model_3mf(path)[0].count('paint_color="') == 0

    doc.undo()
    fileio.export_3mf(path, ops.parts(doc), doc.materials)
    assert model_3mf(path)[0].count('paint_color="') == 2
