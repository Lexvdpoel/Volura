"""Hide/unhide and lock/unlock as in SketchUp 2017.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import json
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import fileio, ops  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))
LOCKED = "This group is locked."


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_hidden = False
    w.view.cam.width, w.view.cam.height = 800, 600
    w.view.cam.set_view("iso")
    yield w
    w.doc.dirty = False
    w.close()


def draw_rect(win, size=(40.0, 30.0)):
    """A loose rectangle in the ground plane; returns (face id, edge ids)."""
    loop = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=float)
    created = win.doc.root.add_shape(GROUND, [loop])
    assert created, "rectangle gave no face"
    ents = win.doc.root
    return created[0], ents.face_edges(ents.F[created[0]])


def box_group(win):
    """A closed box as a group; returns the group."""
    win.doc.root.add_shape(GROUND, [np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float)])
    f = next(iter(win.doc.root.F))
    win.doc.root.pushpull(f, -10)
    return ops.make_group(win.doc, [("f", ROOT, f) for f in win.doc.root.F])


def clean(text):
    return text.replace("&", "").replace("…", "").strip()


def centre_of(win, item):
    """Screen position of the middle of a face or edge."""
    ents = win.doc.ents(ROOT)
    if item[0] == "f":
        p = np.concatenate(ents.face_points(item[1])).mean(axis=0)
    else:
        p = np.mean(ents.edge_points(item[1]), axis=0)
    s, ok = win.view.cam.project(p[None])
    assert ok[0], "point is off screen"
    return s[0]


# ------------------------------------------------------------------- hiding

def test_hide_face_removes_it_from_picking(win):
    """A hidden face is not drawn, not picked and not used for inference."""
    fid, _ = draw_rect(win)
    win.view.zoom_extents()
    x, y = centre_of(win, ("f", fid))
    o, d = win.view.cam.ray(x, y)
    hit = win.view.engine.pick_face(o, d)
    assert hit is not None and hit[2] == fid
    win.doc.root.hide([fid])
    assert win.view.engine.pick_face(o, d) is None
    assert win.view.engine.infer(x, y).kind != "face"
    rc = win.view._render_data(win.doc.root, None)
    assert not len(rc["pos"]), "a hidden face is not drawn"
    assert len(rc["hidden"]["pos"]), "the hidden face stays in the model"


def test_show_hidden_makes_faces_pickable_again(win):
    fid, _ = draw_rect(win)
    win.view.zoom_extents()
    x, y = centre_of(win, ("f", fid))
    o, d = win.view.cam.ray(x, y)
    win.doc.root.hide([fid])
    assert win.view.engine.pick_face(o, d) is None
    win.settings.show_hidden = True
    hit = win.view.engine.pick_face(o, d)
    assert hit is not None and hit[2] == fid, "with Hidden Geometry you can pick hidden faces"


def test_hide_edge_removes_it_from_picking_and_snapping(win):
    _, eids = draw_rect(win)
    win.view.zoom_extents()
    eid = eids[0]
    x, y = centre_of(win, ("e", eid))
    assert win.view.engine.pick_edge(x, y, only={ROOT}, px=8)[2] == eid
    win.doc.root.hide([], [eid])
    assert win.view.engine.pick_edge(x, y, only={ROOT}, px=8) is None
    assert win.view.engine.infer(x, y).kind not in ("edge", "midpoint")
    rc = win.view._render_data(win.doc.root, None)
    assert len(rc["hedges"]) == 2 and len(rc["edges"]) == 2 * (len(eids) - 1)
    win.settings.show_hidden = True
    assert win.view.engine.pick_edge(x, y, only={ROOT}, px=8)[2] == eid


def test_hidden_geometry_stays_part_of_the_model(win):
    """Hidden geometry still counts for export and the print check."""
    g = box_group(win)
    assert g.ents.to_manifold().volume() == pytest.approx(4000.0)
    g.ents.hide(list(g.ents.F)[:1])
    parts = ops.parts(win.doc)
    assert len(parts) == 1 and parts[0].manifold is not None
    assert parts[0].manifold.volume() == pytest.approx(4000.0)


# ------------------------------------------------------------------- menu

def test_hide_and_unhide_a_face_through_the_menu(win):
    fid, _ = draw_rect(win)
    win.doc.set_selection([("f", ROOT, fid)])
    win.a_hide.trigger()
    assert win.doc.root.hidden_f == {fid}, "Edit > Hide hides the selected face"
    assert win.doc.last_hidden == [("f", ROOT, fid)], "the last hidden items are remembered"
    assert not win.doc.selection

    win.a_unhide_last.trigger()
    assert win.doc.root.hidden_f == set()
    assert win.doc.last_hidden == []

    win.doc.set_selection([("f", ROOT, fid)])
    win.a_hide.trigger()
    win.doc.set_selection([("f", ROOT, fid)])   # selected while Hidden Geometry is on
    win.a_unhide_sel.trigger()
    assert win.doc.root.hidden_f == set(), "Unhide > Selected"

    win.doc.set_selection([("f", ROOT, fid)])
    win.a_hide.trigger()
    win.a_unhide.trigger()
    assert win.doc.root.hidden_f == set(), "Unhide > All"


def test_hide_and_unhide_a_group_through_the_menu(win):
    g = box_group(win)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_hide.trigger()
    assert not g.visible
    assert g.id not in [ctx for ctx, _, _ in win.doc.contexts()], "a hidden group is not shown"

    win.a_unhide.trigger()
    assert g.visible, "Unhide > All also unhides groups"


def test_unhide_all_shows_hidden_faces_edges_and_groups(win):
    g = box_group(win)
    fid, eids = draw_rect(win)
    win.doc.root.hide([fid], [eids[0]])
    g.visible = False
    win.a_unhide.trigger()
    assert win.doc.root.hidden_f == set() and win.doc.root.hidden_e == set()
    assert all(gr.visible for gr in win.doc.groups)


def test_hide_and_unhide_are_one_undo_step(win):
    fid, eids = draw_rect(win)
    win.doc.set_selection([("f", ROOT, fid), ("e", ROOT, eids[0])])
    win.a_hide.trigger()
    assert win.doc.root.hidden_f == {fid} and win.doc.root.hidden_e == {eids[0]}
    win.doc.undo()
    assert win.doc.root.hidden_f == set() and win.doc.root.hidden_e == set()
    win.doc.redo()
    assert win.doc.root.hidden_f == {fid} and win.doc.root.hidden_e == {eids[0]}

    win.a_unhide.trigger()
    assert win.doc.root.hidden_f == set() and win.doc.root.hidden_e == set()
    win.doc.undo()
    assert win.doc.root.hidden_f == {fid} and win.doc.root.hidden_e == {eids[0]}


def test_select_all_skips_hidden_geometry(win):
    fid, eids = draw_rect(win)
    win.doc.root.hide([fid], eids[:1])
    win.a_sel_all.trigger()
    assert ("f", ROOT, fid) not in win.doc.selection
    assert ("e", ROOT, eids[0]) not in win.doc.selection
    assert len(win.doc.selection) == len(win.doc.root.E) - 1
    win.settings.show_hidden = True
    win.a_sel_all.trigger()
    assert ("f", ROOT, fid) in win.doc.selection and ("e", ROOT, eids[0]) in win.doc.selection


def test_hide_lock_actions_are_no_longer_placeholders(win):
    for attr, text in (("a_unhide_sel", "Selected"), ("a_unhide_last", "Last"),
                       ("a_unhide", "All"), ("a_lock", "Lock"),
                       ("a_unlock_sel", "Selected"), ("a_unlock_all", "All")):
        a = getattr(win, attr)
        assert a.isEnabled(), f"Edit > {text} is still a placeholder"
        assert text in clean(a.text())


# ------------------------------------------------------------------- locking

def test_lock_and_unlock_through_the_menu(win):
    g = box_group(win)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_lock.trigger()
    assert g.locked
    win.a_unlock_sel.trigger()
    assert not g.locked
    win.a_lock.trigger()
    win.a_unlock_all.trigger()
    assert not any(gr.locked for gr in win.doc.groups)
    win.doc.undo()
    assert win.doc.group(g.id).locked, "unlocking is one undo step"
    win.doc.redo()
    assert not win.doc.group(g.id).locked


def test_transform_selection_leaves_a_locked_group_alone(win):
    g = box_group(win)
    g.locked = True
    before = g.bbox().copy()
    ops.transform_selection(win.doc, np.eye(3), np.array([15.0, 0, 0]), items=[("g", g.id, g.id)])
    assert np.allclose(g.bbox(), before), "a locked group cannot be moved"
    ops.transform_selection(win.doc, np.eye(3), np.array([15.0, 0, 0]), copy=True,
                            items=[("g", g.id, g.id)])
    assert len(win.doc.groups) == 1, "a locked group cannot be copied"
    ops.resize_group(win.doc, g, np.array([40.0, 40.0, 20.0]))
    assert np.allclose(g.bbox(), before), "a locked group cannot be scaled"


def test_transform_selection_still_moves_the_others(win):
    g1 = box_group(win)
    g2 = box_group(win)
    g1.locked = True
    ops.transform_selection(win.doc, np.eye(3), np.array([15.0, 0, 0]),
                            items=[("g", g1.id, g1.id), ("g", g2.id, g2.id)])
    assert np.allclose(g2.bbox()[0], [15, 0, 0])


def test_locked_group_cannot_be_edited_painted_or_deleted(win):
    g = box_group(win)
    g.locked = True
    win.doc.set_selection([("g", g.id, g.id)])

    assert win.doc.enter_group(g.id) is False, "double-click does not open a locked group"
    win.edit_group()
    assert win.doc.active == ROOT
    assert win.hint.text() == LOCKED

    win.paint_selection("#3fa34d")
    assert g.color != "#3fa34d", "a locked group cannot be painted"

    win.delete()
    assert win.doc.group(g.id) is not None, "a locked group cannot be deleted"
    assert win.hint.text() == LOCKED

    ops.explode(win.doc, g)
    assert win.doc.group(g.id) is not None, "a locked group cannot be exploded"


def test_unlocked_group_still_works(win):
    g = box_group(win)
    win.doc.set_selection([("g", g.id, g.id)])
    win.edit_group()
    assert win.doc.active == g.id
    win.close_group()
    ops.transform_selection(win.doc, np.eye(3), np.array([15.0, 0, 0]), items=[("g", g.id, g.id)])
    assert np.allclose(g.bbox()[0], [15, 0, 0])


def test_hidden_geometry_is_drawn_dashed_and_translucent(win):
    """View > Hidden Geometry: edges dashed, faces translucent."""
    from OpenGL import GL as real_gl
    from volura import viewport as vp
    fid, eids = draw_rect(win)
    win.doc.root.hide([fid], [eids[0]])
    win.view.zoom_extents()

    class Mock:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name.startswith("GL_"):
                return getattr(real_gl, name)

            def call(*args):
                self.calls.append((name, args))
            return call

    def run(show):
        m = Mock()
        win.settings.show_hidden = show
        win.settings.face_style = "shaded"
        vp.gl = m
        try:
            win.view._draw_model()
        finally:
            vp.gl = real_gl
            win.settings.show_hidden = False
        return m

    def count(m, mode):
        return sum(1 for n, a in m.calls if n == "glDrawArrays" and a[0] == mode)

    off, on = run(False), run(True)
    assert count(off, real_gl.GL_TRIANGLES) == 0, "a hidden face is not drawn"
    assert count(on, real_gl.GL_TRIANGLES) > 0, "with Hidden Geometry the faces are drawn"
    assert count(on, real_gl.GL_LINES) > count(off, real_gl.GL_LINES)
    assert any(n == "glLineStipple" for n, _ in on.calls), "hidden edges are dashed"
    assert not any(n == "glLineStipple" for n, _ in off.calls)


def test_locked_group_is_drawn_with_a_red_frame(win):
    """A selected locked group gets red instead of blue."""
    from OpenGL import GL as real_gl
    from volura import viewport as vp
    from volura.viewport import LOCK_COLOR, SELECT_COLOR
    g = box_group(win)
    win.view.zoom_extents()
    win.doc.set_selection([("g", g.id, g.id)])
    assert [gr.id for gr in win.doc.selected_groups()] == [g.id], "a locked group can still be selected"

    class Mock:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name.startswith("GL_"):
                return getattr(real_gl, name)

            def call(*args):
                self.calls.append((name, args))
            return call

    def colors():
        m = Mock()
        vp.gl = m
        try:
            win.view._draw_model()
        finally:
            vp.gl = real_gl
        return [a for n, a in m.calls if n == "glColor4f" and len(a) == 4]

    win.doc.set_selection([("g", g.id, g.id)])
    assert any(c[:3] == SELECT_COLOR for c in colors())
    g.locked = True
    got = colors()
    assert any(c[:3] == LOCK_COLOR for c in got), "the selection frame of a locked group is red"
    assert not any(c[:3] == SELECT_COLOR for c in got)


def test_move_tool_refuses_a_locked_group(win):
    g = box_group(win)
    g.locked = True
    win.view.zoom_extents()
    win.doc.set_selection([])
    s, ok = win.view.cam.project(g.bbox().mean(axis=0)[None])
    assert ok[0]
    assert win.view.tools["move"].grab_selection(s[0][0], s[0][1]) == []
    assert win.hint.text() == LOCKED, "Move says that the group is locked"


def test_context_menu_locks_and_unlocks_groups(win):
    g = box_group(win)
    win.doc.set_selection([("g", g.id, g.id)])
    items = [clean(a.text()) for a in win.build_context_menu().actions()]
    assert "Lock" in items, items
    g.locked = True
    items = [clean(a.text()) for a in win.build_context_menu().actions()]
    assert "Unlock" in items and "Lock" not in items, items


def test_face_context_menu_is_unchanged(win):
    """The right-click menu of a face still starts as in the spec."""
    import ui_spec
    fid, _ = draw_rect(win)
    win.doc.set_selection([("f", ROOT, fid)])
    items = ["---" if a.isSeparator() else clean(a.text()) + (" >" if a.menu() is not None else "")
             for a in win.build_context_menu().actions()]
    assert items[:len(ui_spec.CONTEXT_FACE)] == ui_spec.CONTEXT_FACE


# ------------------------------------------------------------------- files

def test_save_and_load_keeps_hidden_and_locked(tmp_path, win):
    g = box_group(win)
    fid, eids = draw_rect(win)
    win.doc.root.hide([fid], eids[:1])
    g.locked = True
    g.ents.hide(list(g.ents.F)[:2])
    path = str(tmp_path / "model.volura")
    fileio.save_document(win.doc, path)

    other = Document()
    fileio.load_document(other, path)
    assert other.root.hidden_f == {fid} and other.root.hidden_e == set(eids[:1])
    g2 = other.group(g.id)
    assert g2 is not None and g2.locked
    assert g2.ents.hidden_f == set(list(g.ents.F)[:2])


def test_old_files_without_these_fields_still_load(tmp_path, win):
    box_group(win)
    draw_rect(win)
    path = str(tmp_path / "old.volura")
    fileio.save_document(win.doc, path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    for key in ("hidden_f", "hidden_e"):
        data["root"].pop(key)
        data["groups"][0]["ents"].pop(key)
    data["groups"][0].pop("locked")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("document.json", json.dumps(data))

    other = Document()
    fileio.load_document(other, path)
    assert other.root.hidden_f == set() and other.root.hidden_e == set()
    assert other.root.F and other.groups and not other.groups[0].locked
    assert other.root.F.keys() == win.doc.root.F.keys()


def test_locked_group_survives_undo_of_another_change(win):
    g = box_group(win)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_lock.trigger()
    win.doc.checkpoint()
    g.ents.set_face_color(list(g.ents.F)[0], "#3fa34d")
    win.doc.undo()
    assert win.doc.group(g.id).locked, "stays locked after undo of another change"


# ------------------------------------------------------------------- settings

def test_show_hidden_setting_is_stored():
    q = QSettings("Volura", "Volura")
    saved = q.value("show_hidden")
    try:
        from volura.app import Settings
        s = Settings()
        s.show_hidden = not s.show_hidden
        s.save()
        assert Settings().show_hidden is s.show_hidden
    finally:
        if saved is None:
            q.remove("show_hidden")
        else:
            q.setValue("show_hidden", saved)
        q.sync()
