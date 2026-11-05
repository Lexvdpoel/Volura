"""Safety net for what goes to the printer: which parts ops.parts offers, which solids
ops.print_parts builds from them, and what ops.check_parts reports about them, so that
faster code (caches, bulk building, background work) cannot quietly change a print."""
import json
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

WHITE = "#f2f2f2"
RED = "#d23b3b"
BOX = mf.Manifold.cube((40, 30, 20))        # 24000 mm³, six faces, bbox 0..40 / 0..30 / 0..20
BOX_FACES = 6
BOX_BB = [[0.0, 0.0, 0.0], [40.0, 30.0, 20.0]]
CUBE = mf.Manifold.cube((20, 20, 20))       # 8000 mm³


def box_doc(name="Box", color=WHITE):
    """A document with one 40 x 30 x 20 mm group."""
    doc = Document()
    g = ops.group_from_manifold(doc, BOX, color, name)
    return doc, g


def part_info(doc):
    """(name, color, volume) of every printable part, or the problem text instead of a volume."""
    return [(p.name, p.color, p.manifold.volume() if p.manifold is not None else p.problem)
            for p in ops.parts(doc)]


def printed(doc):
    """(color, volume, triangles) of every solid ops.print_parts would hand to the exporter."""
    return [(c, m.volume(), m.num_tri()) for c, m in ops.print_parts(ops.parts(doc))]


def saved_json(doc, tmp_path):
    path = tmp_path / "model.vlua"
    fileio.save_document(doc, str(path))
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("document.json"))


def test_hidden_group_is_not_printed():
    doc, g = box_doc()
    assert part_info(doc) == [("Box", WHITE, 24000.0)]
    assert printed(doc) == [(WHITE, 24000.0, 12)]

    g.visible = False
    assert ops.parts(doc) == []
    assert ops.print_parts(ops.parts(doc)) == []
    # hiding leaves the group and its geometry in the document
    assert [x.name for x in doc.groups] == ["Box"]
    assert len(g.ents.F) == BOX_FACES
    assert g.ents.bbox().tolist() == BOX_BB

    g.visible = True
    assert part_info(doc) == [("Box", WHITE, 24000.0)]
    assert printed(doc) == [(WHITE, 24000.0, 12)]


def test_locked_group_is_printed_like_an_unlocked_one():
    doc, g = box_doc()
    before = printed(doc)
    g.locked = True
    assert printed(doc) == before == [(WHITE, 24000.0, 12)]
    assert part_info(doc) == [("Box", WHITE, 24000.0)]
    assert len(g.ents.F) == BOX_FACES
    assert g.ents.bbox().tolist() == BOX_BB


def test_explode_does_nothing_to_a_locked_group():
    doc, g = box_doc()
    g.locked = True
    ops.explode(doc, g)
    # the group stays a group and no loose geometry appears
    assert [x.id for x in doc.groups] == [g.id]
    assert doc.root.F == {}
    assert len(g.ents.F) == BOX_FACES
    assert g.ents.bbox().tolist() == BOX_BB
    assert part_info(doc) == [("Box", WHITE, 24000.0)]
    assert printed(doc) == [(WHITE, 24000.0, 12)]


def test_explode_makes_loose_geometry_with_the_group_color():
    doc, g = box_doc()
    ops.explode(doc, g)
    assert doc.groups == []
    assert len(doc.root.F) == BOX_FACES
    assert {doc.root.F[f].color for f in doc.root.F} == {WHITE}
    assert doc.root.bbox().tolist() == BOX_BB
    assert part_info(doc) == [("Loose geometry 1", WHITE, 24000.0)]
    assert printed(doc) == [(WHITE, 24000.0, 12)]


def test_make_group_takes_loose_geometry_back_into_one_group(tmp_path):
    doc, g = box_doc()
    ops.explode(doc, g)
    new = ops.make_group(doc, [("f", ROOT, f) for f in doc.root.F])
    assert [x.id for x in doc.groups] == [new.id]
    assert new.color == WHITE
    assert len(new.ents.F) == BOX_FACES
    assert new.ents.to_manifold().volume() == pytest.approx(24000.0, rel=1e-6)
    assert new.ents.bbox().tolist() == BOX_BB
    # the loose geometry is gone: the only part left is the new group
    assert doc.root.F == {}
    assert part_info(doc) == [(new.name, WHITE, 24000.0)]
    assert printed(doc) == [(WHITE, 24000.0, 12)]

    data = saved_json(doc, tmp_path)
    assert [gr["color"] for gr in data["groups"]] == [WHITE]
    assert len(data["groups"][0]["ents"]["F"]) == BOX_FACES
    assert data["root"]["F"] == {}


def test_print_parts_merges_per_color_and_the_later_color_wins():
    doc = Document()
    ops.group_from_manifold(doc, CUBE, WHITE, "W1")
    ops.group_from_manifold(doc, CUBE.translate((30, 0, 0)), WHITE, "W2")
    ops.group_from_manifold(doc, CUBE.translate((10, 0, 0)), RED, "R")
    assert part_info(doc) == [("W1", WHITE, 8000.0), ("W2", WHITE, 8000.0), ("R", RED, 8000.0)]

    solids = ops.print_parts(ops.parts(doc))
    assert [c for c, _ in solids] == [WHITE, RED]
    white, red = solids[0][1], solids[1][1]
    # white: 8000 + 8000, minus the 10 x 20 x 20 = 4000 mm³ that red covers
    assert white.volume() == pytest.approx(12000.0, rel=1e-6)
    assert red.volume() == pytest.approx(8000.0, rel=1e-6)
    assert white.num_tri() == 24 and red.num_tri() == 12
    assert tuple(white.bounding_box()) == (0.0, 0.0, 0.0, 50.0, 20.0, 20.0)
    assert tuple(red.bounding_box()) == (10.0, 0.0, 0.0, 30.0, 20.0, 20.0)


def test_open_group_is_reported_and_never_printed():
    doc, g = box_doc(name="Open")
    g.ents.remove([list(g.ents.F)[0]], [])
    assert len(g.ents.F) == BOX_FACES - 1

    parts = ops.parts(doc)
    assert len(parts) == 1
    assert parts[0].name == "Open" and parts[0].color == WHITE
    assert parts[0].manifold is None
    assert "not closed" in parts[0].problem
    msgs = ops.check_parts(parts)
    assert [kind for kind, _ in msgs] == ["fout"]
    assert msgs[0][1].startswith("Open: not closed")
    assert ops.print_parts(parts) == []


def test_painted_face_keeps_volume_parts_and_part_color():
    doc, g = box_doc()
    top = max(g.ents.F, key=lambda f: g.ents.F[f].normal[2])
    assert g.ents.F[top].color is None
    before = ops.parts(doc)
    assert len(before) == 1
    assert before[0].manifold.volume() == pytest.approx(24000.0, rel=1e-6)

    g.ents.set_face_color(top, RED)
    assert g.ents.F[top].color == RED
    assert len(g.ents.F) == BOX_FACES
    assert g.ents.bbox().tolist() == BOX_BB

    after = ops.parts(doc)
    assert len(after) == 1
    assert after[0].manifold.volume() == pytest.approx(24000.0, rel=1e-6)
    # the part keeps the group color; only its triangles carry the face color now
    assert after[0].color == WHITE
    assert printed(doc) == [(WHITE, 24000.0, 12)]
