"""Safety net for long undo/redo series.

After every single undo and every single redo the document has to be exactly what it
was at that point of the series: the same groups in the same order, the same mesh, the
same colours, the same places and the same saved file. Faster code (cached states, bulk
geometry, background work) may not quietly change any of that, so every step is checked
on more than its volume.
"""
import json
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.model import Document  # noqa: E402

WHITE, BLACK, RED, BLUE, GREEN = "#f2f2f2", "#2b2b2b", "#d23b3b", "#2f6fd6", "#3fa34d"
NO_TURN = np.eye(3)
TURN_LEFT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # a quarter turn about z


# ------------------------------------------------------------------ what is compared
def snapshot(doc, tmp_path):
    """The document as Volura writes it: the JSON inside the .volura file."""
    path = str(tmp_path / "state.volura")
    fileio.save_document(doc, path)
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("document.json"))


def volumes(doc):
    """(name, colour, volume) of every group, in the order of doc.groups."""
    return [(g.name, g.color, round(g.ents.to_manifold().volume(), 6)) for g in doc.groups]


def shape(doc):
    """What a volume cannot show: the size of the mesh and the colour of every face."""
    return [(g.name, g.color, len(g.ents.F), len(g.ents.E), len(g.ents.V),
             tuple(sorted(str(f.color) for f in g.ents.F.values()))) for g in doc.groups]


def bboxes(doc):
    """Where every group is."""
    return [g.ents.bbox() for g in doc.groups if g.ents.bbox() is not None]


def record(doc, tmp_path):
    """Everything a step is remembered by."""
    return volumes(doc), shape(doc), bboxes(doc), snapshot(doc, tmp_path)


def check(doc, tmp_path, want):
    """The document must be exactly the recorded state `want`."""
    got_vols, got_shape, got_boxes, got_snap = record(doc, tmp_path)
    want_vols, want_shape, want_boxes, want_snap = want
    assert [(n, c) for n, c, _ in got_vols] == [(n, c) for n, c, _ in want_vols]
    for (name, _, got), (_, _, want_v) in zip(got_vols, want_vols):
        assert got == pytest.approx(want_v, rel=1e-6), name
    assert got_shape == want_shape
    assert len(got_boxes) == len(want_boxes)
    for got_box, want_box in zip(got_boxes, want_boxes):
        assert got_box == pytest.approx(np.asarray(want_box, dtype=float), abs=1e-6)
    assert got_snap == want_snap


# ------------------------------------------------------------------ reading the mesh
def top_faces(ents, z):
    """Every face that lies flat at height z."""
    return [f for f, face in ents.F.items() if face.normal[2] > 0.9
            and all(abs(ents.V[v][2] - z) < 1e-9 for v in face.loops[0])]


def top_face(ents, z):
    faces = top_faces(ents, z)
    assert len(faces) == 1, f"expected one top face at z={z}, found {faces}"
    return faces[0]


def face_center(ents, fid):
    return np.mean([ents.V[v] for v in ents.F[fid].loops[0]], axis=0)


def painted(ents, color):
    return sum(1 for face in ents.F.values() if face.color == color)


def assert_volume(group, want):
    assert group.ents.to_manifold().volume() == pytest.approx(want, rel=1e-6)


def assert_bbox(group, want):
    assert group.ents.bbox() == pytest.approx(np.asarray(want, dtype=float), abs=1e-6)


# ------------------------------------------------------------------ the fifteen edits
def s01_make_the_block(doc, env):
    env["block"] = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), WHITE, "Block")
    assert len(doc.groups) == 1
    assert len(env["block"].ents.F) == 6
    assert_volume(env["block"], 40 * 30 * 20)
    assert_bbox(env["block"], [[0, 0, 0], [40, 30, 20]])


def s02_move_the_block(doc, env):
    ops.transform_selection(doc, NO_TURN, (10.0, 0.0, 0.0), items=[("g", env["block"].id, env["block"].id)])
    assert_volume(env["block"], 40 * 30 * 20)
    assert_bbox(env["block"], [[10, 0, 0], [50, 30, 20]])


def s03_push_the_top_up(doc, env):
    top = top_face(env["block"].ents, 20)
    env["cap"] = env["block"].ents.pushpull(top, 5.0)
    assert env["cap"] is not None and env["cap"] in env["block"].ents.F
    assert env["cap"] != top
    assert len(env["block"].ents.F) == 6
    assert_volume(env["block"], 40 * 30 * 25)
    assert_bbox(env["block"], [[10, 0, 0], [50, 30, 25]])


def s04_draw_a_line_across_the_top(doc, env):
    ents = env["block"].ents
    ents.add_line((10.0, 15.0, 25.0), (50.0, 15.0, 25.0))
    assert len(top_faces(ents, 25)) == 2                       # the line split the top in two
    assert (len(ents.F), len(ents.E), len(ents.V)) == (7, 15, 10)
    assert_volume(env["block"], 40 * 30 * 25)                  # a line on a face adds no material
    assert_bbox(env["block"], [[10, 0, 0], [50, 30, 25]])


def s05_paint_the_front_half(doc, env):
    ents = env["block"].ents
    halves = [f for f in top_faces(ents, 25) if face_center(ents, f)[1] < 15]
    assert len(halves) == 1
    env["front"] = halves[0]
    ents.set_face_color(env["front"], RED)
    assert ents.F[env["front"]].color == RED
    assert painted(ents, RED) == 1 and painted(ents, None) == 6
    assert (len(ents.F), len(ents.E), len(ents.V)) == (7, 15, 10)
    assert_volume(env["block"], 40 * 30 * 25)


def s06_make_the_post(doc, env):
    env["post"] = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 30)).translate((60, 0, 0)),
                                          BLACK, "Post")
    assert [g.name for g in doc.groups] == ["Block", "Post"]
    assert_volume(env["post"], 10 * 10 * 30)
    assert_bbox(env["post"], [[60, 0, 0], [70, 10, 30]])
    assert_volume(env["block"], 40 * 30 * 25)


def s07_move_the_post_onto_the_block(doc, env):
    ops.transform_selection(doc, NO_TURN, (-25.0, 10.0, 0.0), items=[("g", env["post"].id, env["post"].id)])
    assert_bbox(env["post"], [[35, 10, 0], [45, 20, 30]])
    assert_volume(env["post"], 10 * 10 * 30)


def s08_union_the_two(doc, env):
    """The union of 30000 + 3000 mm³ that overlap over 10 * 10 * 25 = 2500 mm³."""
    ops.boolean(doc, [env["block"], env["post"]], "union")
    assert [g.name for g in doc.groups] == ["Block"]
    assert doc.groups[0].color == WHITE
    assert len(env["block"].ents.F) == 11
    assert_volume(env["block"], 30000 + 3000 - 2500)
    assert_bbox(env["block"], [[10, 0, 0], [50, 30, 30]])


def s09_paint_the_post_top(doc, env):
    ents = env["block"].ents
    env["post_top"] = top_face(ents, 30)
    assert face_center(ents, env["post_top"]) == pytest.approx(np.array([40.0, 15.0, 30.0]), abs=1e-6)
    ents.set_face_color(env["post_top"], BLUE)
    assert painted(ents, BLUE) == 1
    assert len(ents.F) == 11
    assert_volume(env["block"], 30500)


def s10_turn_the_solid(doc, env):
    ops.transform_selection(doc, TURN_LEFT, (100.0, 0.0, 0.0), items=[("g", env["block"].id, env["block"].id)])
    assert face_center(env["block"].ents, env["post_top"]) == pytest.approx(np.array([85.0, 40.0, 30.0]), abs=1e-6)
    assert painted(env["block"].ents, BLUE) == 1
    assert_volume(env["block"], 30500)
    assert_bbox(env["block"], [[70, 10, 0], [100, 50, 30]])


def s11_push_the_painted_top_up(doc, env):
    ents = env["block"].ents
    env["cap2"] = ents.pushpull(env["post_top"], 5.0)
    assert env["cap2"] is not None and env["cap2"] in ents.F
    assert ents.F[env["cap2"]].color == BLUE                   # push/pull paints along with the face
    assert painted(ents, BLUE) == 5
    assert len(ents.F) == 11
    assert_volume(env["block"], 30500 + 10 * 10 * 5)
    assert_bbox(env["block"], [[70, 10, 0], [100, 50, 35]])


def s12_make_a_cutting_cube(doc, env):
    env["cube"] = ops.group_from_manifold(doc, mf.Manifold.cube((20, 20, 20)).translate((60, 20, 0)),
                                          RED, "Cutter")
    assert [g.name for g in doc.groups] == ["Block", "Cutter"]
    assert_volume(env["cube"], 20 * 20 * 20)
    assert_bbox(env["cube"], [[60, 20, 0], [80, 40, 20]])


def s13_cut_the_cube_out_of_the_solid(doc, env):
    """What is left: 31000 - 10 * 20 * 20 = 27000 mm³. A boolean drops the face colours
    of the groups it merges, so after this cut nothing is painted any more."""
    env["block"] = ops.boolean(doc, [env["block"], env["cube"]], "subtract")
    assert [g.name for g in doc.groups] == ["Block"]
    assert doc.groups[0].color == WHITE
    assert len(env["block"].ents.F) == 15
    assert painted(env["block"].ents, BLUE) == 0
    assert_volume(env["block"], 31000 - 10 * 20 * 20)
    assert_bbox(env["block"], [[70, 10, 0], [100, 50, 35]])


def s14_make_a_stud(doc, env):
    env["stud"] = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)).translate((75, 15, 35)),
                                          GREEN, "Stud")
    assert [g.name for g in doc.groups] == ["Block", "Stud"]
    assert_volume(env["stud"], 10 * 10 * 10)
    assert_bbox(env["stud"], [[75, 15, 35], [85, 25, 45]])


def s15_remove_the_stud(doc, env):
    doc.remove_group(env["stud"].id)
    assert doc.group(env["stud"].id) is None
    assert [g.name for g in doc.groups] == ["Block"]
    assert len(env["block"].ents.F) == 15
    assert_volume(env["block"], 27000)


STEPS = [s01_make_the_block, s02_move_the_block, s03_push_the_top_up,
         s04_draw_a_line_across_the_top, s05_paint_the_front_half, s06_make_the_post,
         s07_move_the_post_onto_the_block, s08_union_the_two, s09_paint_the_post_top,
         s10_turn_the_solid, s11_push_the_painted_top_up, s12_make_a_cutting_cube,
         s13_cut_the_cube_out_of_the_solid, s14_make_a_stud, s15_remove_the_stud]


def build(doc, tmp_path):
    """The fifteen edits, each behind its own checkpoint, with the state after every one."""
    history = [record(doc, tmp_path)]
    env = {}
    for step in STEPS:
        doc.checkpoint()
        step(doc, env)
        history.append(record(doc, tmp_path))
    return history


def test_undo_and_redo_a_series_of_fifteen_edits(tmp_path):
    doc = Document()
    history = build(doc, tmp_path)

    for i in range(len(STEPS), 0, -1):
        assert doc.undo() is True
        check(doc, tmp_path, history[i - 1])       # back to the state after edit i - 1
    assert doc.undo() is False                     # the undo stack held exactly fifteen edits
    assert doc.groups == [] and doc.root.F == {}   # and the document is the empty one again

    for i in range(1, len(STEPS) + 1):
        assert doc.redo() is True
        check(doc, tmp_path, history[i])           # the last redo is the state before undoing
    assert doc.redo() is False                     # the redo stack is empty

    for i in range(len(STEPS), 0, -1):             # redo put every edit back on the undo stack
        assert doc.undo() is True
        check(doc, tmp_path, history[i - 1])


def test_a_new_edit_after_an_undo_forgets_what_was_undone(tmp_path):
    """An edit in Volura begins with checkpoint(), and that is what clears the redo stack."""
    doc = Document()
    doc.checkpoint()
    g = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), WHITE, "Cube")
    created = record(doc, tmp_path)

    doc.checkpoint()
    ops.transform_selection(doc, NO_TURN, (50.0, 0.0, 0.0), items=[("g", g.id, g.id)])
    assert_bbox(g, [[50, 0, 0], [60, 10, 10]])
    assert doc.undo() is True
    check(doc, tmp_path, created)
    assert len(doc.redo_stack) == 1                # the move could still be redone

    doc.checkpoint()                               # a new edit, like every edit in Volura
    cap = g.ents.pushpull(top_face(g.ents, 10), 5.0)
    assert cap in g.ents.F
    assert doc.redo_stack == []
    assert doc.redo() is False                     # current behaviour: redo says it did nothing
    assert_volume(g, 10 * 10 * 15)
    assert_bbox(g, [[0, 0, 0], [10, 10, 15]])      # the undone move did not come back

    assert doc.undo() is True                      # the new edit itself is still undoable
    check(doc, tmp_path, created)
    assert doc.redo() is True
    assert_volume(g, 10 * 10 * 15)


def test_undo_on_an_empty_document_and_redo_without_undo(tmp_path):
    doc = Document()
    empty = record(doc, tmp_path)
    assert doc.undo() is False                     # current behaviour: nothing to undo
    assert doc.redo() is False                     # current behaviour: nothing to redo
    assert doc.undo() is False
    assert doc.redo() is False
    check(doc, tmp_path, empty)
    assert doc.undo_stack == [] and doc.redo_stack == []

    doc.checkpoint()
    g = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), WHITE, "Cube")
    assert doc.undo() is True
    assert doc.undo() is False                     # the series was one edit long
    assert doc.redo() is True
    assert_volume(g, 10 * 10 * 10)
    assert doc.redo() is False
