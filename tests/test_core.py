"""Document, export and file tests (no GUI). Run with:  python -m pytest tests"""
import os
import sys
import tempfile
import zipfile

import manifold3d as mf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import fileio, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


def box_doc(w=20, d=20, h=10):
    doc = Document()
    doc.root.add_shape(GROUND, [np.array([[0, 0], [w, 0], [w, d], [0, d]], float)])
    f = next(iter(doc.root.F))
    doc.root.pushpull(f, -h)
    return doc


def top(ents):
    return max(ents.F, key=lambda f: (ents.F[f].normal[2], ents.V[ents.F[f].loops[0][0]][2]))


def test_loose_box_is_printable_part():
    doc = box_doc()
    parts = ops.parts(doc)
    assert len(parts) == 1 and parts[0].manifold is not None
    assert abs(parts[0].manifold.volume() - 4000) < 1e-6


def test_open_box_is_reported():
    doc = box_doc()
    doc.root.erase_faces([top(doc.root)])
    parts = ops.parts(doc)
    assert parts[0].manifold is None and "niet gesloten" in parts[0].problem


def test_group_and_inlay_export():
    doc = box_doc()
    g = ops.make_group(doc, [("f", ROOT, f) for f in doc.root.F])
    assert not doc.root.F and len(doc.groups) == 1
    ents = g.ents
    t = top(ents)
    pl = ents.face_plane(t)
    c = pl.to_local(np.array([7.0, 7.0, 10.0]))
    ents.add_shape(pl, [np.array([[0, 0], [6, 0], [6, 6], [0, 6]], float) + c])
    small = min((f for f in ents.faces_in_plane(pl)), key=lambda f: len(ents.F[f].loops))
    inner = [f for f in ents.faces_in_plane(pl) if len(ents.F[f].loops) == 1 and f != t]
    target = inner[0] if inner else small
    red = ops.inlay(doc, ents, target, -2, "#d23b3b")
    parts = ops.parts(doc)
    vols = sorted(round(p.manifold.volume(), 3) for p in parts)
    assert vols == [72.0, 4000 - 72.0]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.volura")
        fileio.save_document(doc, p)
        d2 = Document()
        fileio.load_document(d2, p)
        assert len(d2.groups) == 2
        assert sorted(round(x.manifold.volume(), 3) for x in ops.parts(d2)) == vols
        p3 = os.path.join(tmp, "t.3mf")
        used = fileio.export_3mf(p3, parts, doc.materials)
        assert sorted(u[1] for u in used) == [1, 3]
        with zipfile.ZipFile(p3) as z:
            prusa = z.read("Metadata/Slic3r_PE_model.config").decode()
            bambu = z.read("Metadata/model_settings.config").decode()
        assert prusa.count("<volume ") == 2
        assert bambu.count("<part ") == 2 and 'key="extruder" value="3"' in bambu
        ps = os.path.join(tmp, "t.stl")
        fileio.export_stl(ps, parts)
        assert abs(fileio.import_stl(ps).volume() - 4000) < 1e-3
        assert len(fileio.export_stl_per_color(ps, parts, doc.materials)) == 2
    assert red.color == "#d23b3b"


def test_overlap_resolution():
    doc = Document()
    ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), "#ffffff")
    ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)).translate((5, 0, 0)), "#ff0000")
    res = dict(ops.print_parts(ops.parts(doc)))
    assert abs(res["#ffffff"].volume() - 500) < 1e-6
    assert abs(res["#ff0000"].volume() - 1000) < 1e-6


def test_undo_redo():
    doc = box_doc()
    doc.checkpoint()
    doc.root.erase_edges(list(doc.root.E)[:1])
    assert len(doc.root.F) == 4
    doc.undo()
    assert len(doc.root.F) == 6
    doc.redo()
    assert len(doc.root.F) == 4


def test_move_group_and_copy():
    doc = box_doc()
    g = ops.make_group(doc, [("f", ROOT, f) for f in doc.root.F])
    ops.transform_selection(doc, np.eye(3), np.array([5.0, 0, 0]), items=[("g", g.id, g.id)])
    assert np.allclose(g.bbox()[0], [5, 0, 0])
    ops.transform_selection(doc, np.eye(3), np.array([30.0, 0, 0]), copy=True, items=[("g", g.id, g.id)])
    assert len(doc.groups) == 2


def test_move_face_stretches_box():
    doc = box_doc()
    t = top(doc.root)
    vids = {v for loop in doc.root.F[t].loops for v in loop}
    doc.root.transform(np.eye(3), np.array([0, 0, 5.0]), vids)
    assert abs(doc.root.to_manifold().volume() - 6000) < 1e-6


def test_from_manifold_primitive_is_solid():
    ents = Entities.from_manifold(ops.primitive("cylinder", 10, 20, segments=48))
    m = ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert len(ents.F) == 50  # top, bottom and 48 sides
    assert len([e for e in ents.E if e not in ents.soft]) == 96


def test_plane_frame_right_handed():
    for n in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0], [1, 2, 3]):
        p = Plane((0, 0, 0), n)
        assert np.allclose(np.cross(p.u, p.v), p.n)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
