"""Support blockers and support enforcers: volumes in the model that shape the support of a print.

A group can take another part in a print than being a part: `g.meta["role"]` says which. A blocker
is a volume the support of the print may not enter (CuraEngine's `anti_overhang_mesh`) and an
enforcer is a volume that becomes support, also when the print has support switched off
(`support_mesh` with `support_mesh_drop_down`). `ops.parts` leaves those volumes out and
`ops.role_parts` gives them with the settings of their role on their own mesh (see
`ops.ROLE_SETTINGS`), so no export carries them, no print check counts them and
`slicer.slice_meshes(..., helpers=...)` builds them into the support of the print instead of into
the print itself. In the 3D view they are see-through red and green, Entity Info picks the role and
3D Printing has two commands that put a cube of 20 mm against the selected face - or on the middle
of the bed when nothing is selected.

These tests (Qt offscreen) check the role in the model (set, read, junk, undo, saved file), the two
commands, the Role row of Entity Info, what the view and the print checks make of a volume and,
when CuraEngine is there, that the settings reach the engine on the mesh of the volume alone and
that the support of a T really follows the volumes. The helpers `face_at` and `rect` are copied
from tests/test_overhangs.py.

Run with:  python -m pytest tests/test_support_roles.py
"""
import os
import re
import subprocess
import sys
import time

import manifold3d as mf
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QLabel,  # noqa: E402
                               QMessageBox)

import volura.app as app_module  # noqa: E402
from volura import fileio, ops, slicer, theme  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, point_in_polys  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.viewport import ROLE_ALPHA  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: every setting and checkable action a test of this file relies on, so no test follows what
#: another one left behind in the settings
SETTING_KEYS = ("bed_size", "current_color", "show_overhangs", "show_thin_walls",
                "show_tight_fits", "overhang_limit", "min_wall", "screw_hole_clearance")

#: the cube the two commands put in the model, and the colours the view draws it in
CUBE = ops.SUPPORT_VOLUME_MM
ROLE_THEME = {ops.ROLE_BLOCKER: "blocker", ops.ROLE_ENFORCER: "enforcer"}


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs into the folder of the test, on a bed of its own, with the views off."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    w = app_module.MainWindow()
    w.settings.bed_size = (200.0, 200.0, 250.0)
    w.settings.current_color = "#cccccc"
    w.settings.show_overhangs = False
    w.settings.show_thin_walls = False
    w.settings.show_tight_fits = False
    w.settings.overhang_limit = 45
    w.settings.min_wall = 0.8
    w.settings.screw_hole_clearance = 0.2
    for a in (w.a_overhangs, w.a_thin_walls, w.a_tight_fits):
        a.setChecked(False)                 # trigger() would otherwise switch them off
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


# ------------------------------------------------------------------------ the model

def P(x, y, z=0):
    return np.array([x, y, z], dtype=float)


def face_at(ents, x, y, z):
    p = P(x, y, z)
    for f in ents.F:
        pl = ents.face_plane(f)
        if abs(pl.distance(p)) > 1e-6:
            continue
        polys = [pl.to_local(L) for L in ents.face_points(f)]
        if point_in_polys(polys, pl.to_local(p)):
            return f
    raise AssertionError(f"no face at {p}")


def rect(ents, x0, y0, x1, y1, z=0):
    pl = Plane((0, 0, z), (0, 0, 1))
    hit = [f for f in ents.faces_in_plane(pl)]
    if hit:
        pl = ents.face_plane(hit[0])
        pl = Plane((0, 0, z), pl.n)
    loop = pl.to_local(np.array([P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)]))
    return ents.add_shape(pl, [loop])


def block(x0, y0, z0, x1, y1, z1):
    """A closed block as a geometry: every face a plain polygon of four corners."""
    ents = Entities()
    rect(ents, x0, y0, x1, y1, z=z0)
    f = face_at(ents, (x0 + x1) / 2, (y0 + y1) / 2, z0)
    ents.pushpull(f, z1 - z0 if ents.F[f].normal[2] > 0 else -(z1 - z0))
    assert np.allclose(ents.bbox(), [[x0, y0, z0], [x1, y1, z1]])
    return ents


def cube_man(size=20.0, at=(0.0, 0.0, 0.0)):
    return mf.Manifold.cube((size, size, size)).translate(at)


def box_man(size=(20.0, 20.0, 20.0), at=(0.0, 0.0, 0.0)):
    return mf.Manifold.cube(size).translate(at)


def add(doc, shape, name, role=None, color="#cccccc"):
    """A group of a manifold (or of a raw geometry) with a role when it is no printed part."""
    man = shape if isinstance(shape, mf.Manifold) else None
    g = (ops.group_from_manifold(doc, man, color, name) if man is not None
         else doc.add_group(shape, color, name))
    if role:
        ops.set_group_role(g, role)
    doc.notify()
    return g


def t_shape():
    """A column in the middle of a bar: both ends of that bar hang in the air (the same shape as
    `_t_shape` of tests/test_slicer.py, but standing symmetric about its column)."""
    return (mf.Manifold.cube((10.0, 10.0, 30.0))
            + mf.Manifold.cube((40.0, 10.0, 10.0)).translate((-15.0, 0.0, 30.0)))


def model_doc():
    """A printed part, a blocker, an enforcer and a group whose role key holds junk."""
    doc = Document()
    add(doc, cube_man(), "part")
    add(doc, cube_man(at=(100.0, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    add(doc, cube_man(at=(0.0, 100.0, 0.0)), "foam", ops.ROLE_ENFORCER)
    junk = add(doc, cube_man(at=(100.0, 100.0, 0.0)), "junk")
    junk.meta["role"] = "no such role"
    return doc


# ------------------------------------------------------------------------- the window

def select(win, *items):
    win.doc.set_selection(list(items))
    win.info.refresh()


def pump_until(predicate, timeout_ms=60000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def clean(text):
    return text.replace("&", "").replace("…", "").strip()


def role_combo(win):
    """The Role row of Entity Info."""
    for box in win.info.body.findChildren(QComboBox):
        if {box.itemData(i) for i in range(box.count())} == set(ops.ROLES):
            return box
    raise AssertionError("Entity Info shows no Role row")


def title_label(win):
    return next(lab.text() for lab in win.info.body.findChildren(QLabel)
                if lab.text().startswith("<b>Group"))


def drawn(win, ents):
    """The render data the model pass makes for `ents`, with the role of its group applied."""
    win.view._overhang_ctx = win.view._overhang_context()
    for _ctx, e, _g, rc, _mats in win.view._draw_contexts(False):
        if e is ents:
            return rc
    raise AssertionError("the view draws no data for this geometry")


def painted_as(win, ents, colour):
    """One bool per face-vertex whether the view draws it in `colour` (a name of theme.COLORS)."""
    rgb = np.array(theme.rgb(colour), dtype=np.float32)
    rc = drawn(win, ents)
    return np.all(np.isclose(rc["front"][:, :3], rgb), axis=1)


def painted_colour(win, ents, colour):
    """True when the view draws every face of `ents` in `colour`, at both sides."""
    rgb = np.array(theme.rgb(colour), dtype=np.float32)
    rc = drawn(win, ents)
    return bool(np.all(painted_as(win, ents, colour))
                and np.all(np.isclose(rc["back"][:, :3], rgb)))


def role_colour(win, ents, role):
    return painted_colour(win, ents, ROLE_THEME[role])


# ---------------------------------------------------------------------- with the engine

def fill(dlg):
    dlg.maker.setCurrentText("Creality3D")
    i = dlg.printer_box.findData("creality_ender3")
    assert i >= 0
    dlg.printer_box.setCurrentIndex(i)
    return QDialog.Accepted


def ender3(user=None):
    from volura import cura_profiles
    printer = cura_profiles.Library(RESOURCES).printer("creality_ender3")
    return printer.stacks(printer.default_variant(), printer.default_material(), "standard",
                          dict(user or {}))


def mesh_of(p, stacks):
    """(verts, tris, extruder, own settings) of a part, the way the window builds them."""
    mm = p.manifold.to_mesh()
    return (np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), 0,
            slicer.per_mesh_settings(stacks, p.mesh_settings))


def slice_with_roles(doc, stacks, out_path, capture=None):
    """The G-code of `doc` sliced the way the window does: its parts, its support volumes beside
    them on the command line."""
    swap = capture is not None
    if swap:
        real, slicer.subprocess = slicer.subprocess, capture
    try:
        slicer.slice_meshes(stacks, [mesh_of(p, stacks) for p in ops.parts(doc)], str(out_path),
                            helpers=[mesh_of(p, stacks) for p in ops.role_parts(doc)])
    finally:
        if swap:
            slicer.subprocess = real
    with open(str(out_path), encoding="utf-8", errors="replace") as fh:
        return fh.read()


class CommandLine:
    """`subprocess` with its `Popen` wrapped, so a test reads the arguments CuraEngine was called
    with, and what `-r` told it, before that temporary folder is gone again."""

    def __init__(self):
        self.cmds = []
        self.globals = []

    def Popen(self, cmd, *args, **kwargs):
        import json
        self.cmds.append(list(cmd))
        with open(cmd[cmd.index("-r") + 1], encoding="utf-8") as fh:
            self.globals.append(json.load(fh)["global"])
        return subprocess.Popen(cmd, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(subprocess, name)


def per_mesh(cmd):
    """The settings the command hands to each mesh of its own: {mesh file: {key: value}}."""
    out, name = {}, None
    for i, token in enumerate(cmd):
        if token == "-l":
            name = cmd[i + 1]
            out[name] = {}
        elif token == "-s" and name is not None:
            key, _, value = cmd[i + 1].partition("=")
            out[name][key] = value
    return out


#: the plastic of these G-code types is no plastic of the part itself
NON_PART = ("SUPPORT", "SKIRT", "BRIM", "RAFT", "BUILD", "ADAPTIVE", "SKEW")


def extrusions(text):
    """The (x, y, z, type) of every move that lays something, in the coordinates of the bed."""
    out, pos, kind = [], [0.0, 0.0, 0.0], ""
    for line in text.splitlines():
        if line.startswith(";TYPE:"):
            kind = line[len(";TYPE:"):].strip()
            continue
        if not line.startswith(("G0 ", "G1 ")) or " E" not in line:
            continue
        before = pos[0], pos[1]
        for word in line.split()[1:]:
            if len(word) > 1 and word[0] in "XYZ":
                try:
                    pos["XYZ".index(word[0])] = float(word[1:])
                except ValueError:
                    pass        # a reset of the extruder or a formula is no coordinate
        if (pos[0], pos[1]) != before:      # a retraction lays nothing
            out.append((pos[0], pos[1], pos[2], kind))
    return out


def arms(text):
    """(left, right, part moves): the support extrusions beside the column of the T, and how often
    the print lays plastic of itself.

    The first layer is the column alone, so the plastic it lays says where the middle of the print
    is; every support extrusion beside that stands under one of the two ends of the bar."""
    first = text.split(";LAYER:0\n", 1)[1].split(";LAYER:", 1)[0]
    base = [p for p in extrusions(first) if p[3] and not p[3].startswith(NON_PART)]
    lo = min(p[0] for p in base)
    hi = max(p[0] for p in base)
    left = right = part = 0
    for x, _y, _z, kind in extrusions(text):
        if kind.startswith("SUPPORT"):
            if x < lo:
                left += 1
            elif x > hi:
                right += 1
        elif kind and not kind.startswith(NON_PART):
            part += 1
    return left, right, part


# --------------------------------------------------------------- the role in the model

def test_a_group_that_says_nothing_is_a_printed_part():
    doc = Document()
    g = add(doc, cube_man(), "part")
    assert ops.group_role(g) == ops.ROLE_PART
    assert "role" not in g.meta, "a plain part says nothing about roles"


@pytest.mark.parametrize("junk", ["", "blockers", None, 7, True, ["blocker"], {"enforcer": 1}])
def test_junk_under_the_role_key_reads_as_a_printed_part(junk):
    doc = model_doc_without_roles()
    g = doc.groups[0]
    g.meta["role"] = junk
    assert ops.group_role(g) == ops.ROLE_PART, "printing what you were shown is the safe thing"
    assert [p.name for p in ops.parts(doc)] == ["part"]
    assert ops.role_parts(doc) == []


def model_doc_without_roles():
    doc = Document()
    add(doc, cube_man(), "part")
    return doc


def test_a_junk_role_is_written_back_as_a_role_of_the_three():
    doc = model_doc_without_roles()
    g = doc.groups[0]
    g.meta["role"] = "banana"
    ops.set_group_role(g, ops.ROLE_ENFORCER)
    assert g.meta["role"] == "enforcer"
    ops.set_group_role(g, ops.ROLE_PART)
    assert "role" not in g.meta, "a printed part leaves no key behind"
    ops.set_group_role(g, "banana")
    assert "role" not in g.meta, "a role that is no role is no key either"


def test_a_group_without_a_meta_dict_reads_as_a_printed_part():
    doc = model_doc_without_roles()
    g = doc.groups[0]
    g.meta = None
    assert ops.group_role(g) == ops.ROLE_PART
    assert ops.parts(doc)[0].mesh_settings == {}


def test_the_role_travels_through_a_saved_file(tmp_path):
    doc = model_doc()
    path = str(tmp_path / "roles.volura")
    fileio.save_document(doc, path)
    other = Document()
    fileio.load_document(other, path)
    assert {g.name: ops.group_role(g) for g in other.groups} == {
        "part": ops.ROLE_PART, "wall": ops.ROLE_BLOCKER,
        "foam": ops.ROLE_ENFORCER, "junk": ops.ROLE_PART}
    plain = next(g for g in other.groups if g.name == "part")
    assert "role" not in plain.meta
    assert [p.name for p in ops.parts(other)] == ["part", "junk"]


def test_a_new_role_is_one_undo_step():
    doc = model_doc_without_roles()
    g = doc.groups[0]
    doc.checkpoint()
    ops.set_group_role(g, ops.ROLE_ENFORCER)
    assert ops.group_role(g) == ops.ROLE_ENFORCER
    doc.undo()
    assert ops.group_role(g) == ops.ROLE_PART
    doc.redo()
    assert ops.group_role(g) == ops.ROLE_ENFORCER


# ------------------------------------------------------------- parts and role_parts

def test_parts_leave_the_support_volumes_out():
    doc = model_doc()
    assert [p.name for p in ops.parts(doc)] == ["part", "junk"], "a support volume printed itself"
    assert sorted(p.name for p in ops.role_parts(doc)) == ["foam", "wall"]


def test_only_selected_groups_are_part_of_a_slice():
    doc = model_doc()
    wall = next(g for g in doc.groups if g.name == "wall")
    doc.set_selection([("g", wall.id, wall.id)])
    assert ops.parts(doc, only_selected=True) == []
    assert [p.name for p in ops.role_parts(doc, only_selected=True)] == ["wall"]


def test_role_parts_carry_the_settings_of_their_role():
    doc = model_doc()
    got = {p.name: p.mesh_settings for p in ops.role_parts(doc)}
    assert got["wall"] == {"anti_overhang_mesh": True}
    assert got["foam"] == {"support_mesh": True, "support_mesh_drop_down": True}
    assert {p.role for p in ops.role_parts(doc)} == {ops.ROLE_BLOCKER, ops.ROLE_ENFORCER}


def test_a_support_volume_keeps_its_own_settings_below_those_of_its_role():
    doc = model_doc_without_roles()
    g = doc.groups[0]
    ops.set_group_role(g, ops.ROLE_BLOCKER)
    g.meta["mesh_settings"] = {"material_flow": 92.5}
    assert ops.role_parts(doc)[0].mesh_settings == {"material_flow": 92.5,
                                                    "anti_overhang_mesh": True}


def test_a_part_carries_its_own_settings_and_the_role_of_a_part():
    doc = model_doc_without_roles()
    g = doc.groups[0]
    g.meta["mesh_settings"] = {"material_flow": 92.5}
    part = ops.parts(doc)[0]
    assert part.mesh_settings == {"material_flow": 92.5}
    assert part.role == ops.ROLE_PART


@needs_engine
def test_the_settings_of_both_roles_are_ones_a_single_mesh_accepts():
    st = ender3()
    for role in (ops.ROLE_BLOCKER, ops.ROLE_ENFORCER):
        settings = ops.ROLE_SETTINGS[role]
        assert slicer.per_mesh_settings(st, settings) == settings, f"{role} is no per-mesh role"


# ------------------------------------------------------------- the two commands

def labels_of(menu):
    return [clean(a.text()) for a in menu.actions() if not a.isSeparator()]


def test_the_two_commands_are_in_the_print_tools_menu(win):
    labels = labels_of(win.print_tools_menu)
    assert labels[labels.index("Add Support Blocker") + 1] == "Add Support Enforcer"
    assert labels[labels.index("Add Support Enforcer") + 1] == "Elephant Foot Chamfer"


def test_add_support_blocker_stands_a_cube_against_the_selected_face(win):
    part = add(win.doc, block(0.0, 0.0, 0.0, 20.0, 20.0, 20.0), "part")
    top = face_at(part.ents, 10.0, 10.0, 20.0)
    select(win, ("f", part.id, top))
    win.a_support_blocker.trigger()
    volumes = ops.role_parts(win.doc)
    assert len(volumes) == 1 and volumes[0].role == ops.ROLE_BLOCKER
    g = win.doc.groups[-1]
    assert np.allclose(g.bbox(), [[0.0, 0.0, 20.0], [20.0, 20.0, 40.0]]), "not against the face"
    assert g.name.startswith(f"{tr('Support blocker')} "), g.name
    assert [grp.id for grp in win.doc.selected_groups()] == [g.id], "it cannot be moved at once"
    assert tr("Support blocker") in win.hint.text()
    assert win.tool_actions["move"].isChecked()
    # the volume stands against the part and is no part of it: nothing of it prints inside it
    assert volumes[0].manifold.volume() == pytest.approx(CUBE ** 3)
    common = part.ents.to_manifold() ^ volumes[0].manifold
    assert common.volume() == pytest.approx(0.0, abs=1e-6)


def test_add_support_enforcer_stands_on_the_middle_of_the_bed_without_a_selection(win):
    win.a_support_enforcer.trigger()
    g = win.doc.groups[-1]
    assert np.allclose(g.bbox(), [[90.0, 90.0, 0.0], [110.0, 110.0, CUBE]])
    assert ops.group_role(g) == ops.ROLE_ENFORCER
    assert g.name == f"{tr('Support enforcer')} 1"


def test_a_volume_against_a_vertical_face_stands_beside_the_part(win):
    part = add(win.doc, block(0.0, 0.0, 0.0, 20.0, 20.0, 20.0), "part")
    side = face_at(part.ents, 20.0, 10.0, 10.0)
    select(win, ("f", part.id, side))
    win.a_support_enforcer.trigger()
    assert np.allclose(win.doc.groups[-1].bbox(), [[20.0, 0.0, 0.0], [40.0, 20.0, 20.0]])


def test_a_selected_group_is_no_face_to_stand_against(win):
    """Only a face says where the volume goes; a part that is selected says the middle of the bed."""
    part = add(win.doc, block(0.0, 0.0, 0.0, 20.0, 20.0, 20.0), "part")
    select(win, ("g", part.id, part.id))
    win.a_support_blocker.trigger()
    assert np.allclose(win.doc.groups[-1].bbox(), [[90.0, 90.0, 0.0], [110.0, 110.0, CUBE]])


def test_a_face_that_is_gone_leaves_the_place_of_the_volume_to_the_bed():
    doc = Document()
    g = add(doc, cube_man(), "part")
    doc.set_selection([("f", g.id, 987654)])
    assert ops.support_volume_center(doc) is None
    doc.set_selection([("g", g.id, g.id), ("f", g.id, next(iter(g.ents.F)))])
    assert ops.support_volume_center(doc) is not None    # the face that is there wins


def test_a_support_command_is_one_undo_step(win):
    groups = len(win.doc.groups)
    win.a_support_blocker.trigger()
    assert len(win.doc.groups) == groups + 1
    win.doc.undo()
    assert len(win.doc.groups) == groups
    assert ops.role_parts(win.doc) == []


def test_a_second_volume_keeps_the_role_of_the_first(win):
    win.a_support_blocker.trigger()
    win.a_support_enforcer.trigger()
    assert [ops.group_role(g) for g in win.doc.groups] == [ops.ROLE_BLOCKER, ops.ROLE_ENFORCER]
    assert len({g.name for g in win.doc.groups}) == 2, "the two volumes carry the same name"


def test_the_command_leaves_the_part_a_part(win):
    part = add(win.doc, cube_man(), "part")
    select(win, ("g", part.id, part.id))
    win.a_support_blocker.trigger()
    assert [g.name for g in win.doc.selected_groups()] == [win.doc.groups[-1].name]
    assert ops.group_role(part) == ops.ROLE_PART, "the part took the role of the volume"
    assert len(ops.parts(win.doc)) == 1


def test_the_view_of_a_volume_you_add_is_see_through(win):
    """The cube of the command is drawn as a volume, not as a part: red or green and translucent."""
    win.a_support_blocker.trigger()
    g = win.doc.groups[-1]
    assert role_colour(win, g.ents, ops.ROLE_BLOCKER)
    assert 0 < ROLE_ALPHA < 1, "a support volume is never solid"


# ----------------------------------------------------------------------- Entity Info

def test_entity_info_shows_the_role_of_the_group(win):
    part = add(win.doc, cube_man(), "part")
    select(win, ("g", part.id, part.id))
    box = role_combo(win)
    assert box.currentData() == ops.ROLE_PART
    assert [box.itemData(i) for i in range(box.count())] == list(ops.ROLES)
    assert tr("Part") in [box.itemText(i) for i in range(box.count())]


def test_picking_another_role_in_entity_info_lands_in_the_model(win):
    part = add(win.doc, cube_man(), "part")
    select(win, ("g", part.id, part.id))
    steps = len(win.doc.undo_stack)
    box = role_combo(win)
    box.setCurrentIndex(box.findData(ops.ROLE_ENFORCER))
    _app.processEvents()
    assert ops.group_role(part) == ops.ROLE_ENFORCER
    assert len(win.doc.undo_stack) == steps + 1, "not one undo step"
    win.doc.undo()
    assert ops.group_role(part) == ops.ROLE_PART


def test_entity_info_keeps_the_role_of_the_group_across_a_refresh(win):
    wall = add(win.doc, cube_man(), "wall", ops.ROLE_BLOCKER)
    select(win, ("g", wall.id, wall.id))
    assert role_combo(win).currentData() == ops.ROLE_BLOCKER
    win.info.refresh()
    assert role_combo(win).currentData() == ops.ROLE_BLOCKER


def test_entity_info_reads_a_junk_role_as_a_part_and_writes_a_role_back_cleanly(win):
    part = add(win.doc, cube_man(), "part")
    part.meta["role"] = "banana"
    select(win, ("g", part.id, part.id))
    assert role_combo(win).currentData() == ops.ROLE_PART, "junk may not paint itself as a role"
    box = role_combo(win)
    box.setCurrentIndex(box.findData(ops.ROLE_ENFORCER))
    assert part.meta["role"] == "enforcer"
    win.info.refresh()
    box = role_combo(win)
    box.setCurrentIndex(box.findData(ops.ROLE_PART))
    assert "role" not in part.meta, "back to a printed part means no key at all"


def test_entity_info_says_what_a_support_volume_is(win):
    wall = add(win.doc, cube_man(), "wall", ops.ROLE_BLOCKER)
    select(win, ("g", wall.id, wall.id))
    assert tr("Support blocker") in title_label(win)
    assert "printable part" not in title_label(win), "a support volume is no part"
    part = add(win.doc, cube_man(at=(100.0, 0.0, 0.0)), "part")
    select(win, ("g", part.id, part.id))
    assert "printable part" in title_label(win)


# ---------------------------------------------------------------------------- the view

def test_the_view_draws_both_support_volumes_in_their_own_colour(win):
    wall = add(win.doc, cube_man(), "wall", ops.ROLE_BLOCKER)
    foam = add(win.doc, cube_man(at=(100.0, 0.0, 0.0)), "foam", ops.ROLE_ENFORCER)
    part = add(win.doc, cube_man(at=(0.0, 100.0, 0.0)), "part")
    assert role_colour(win, wall.ents, ops.ROLE_BLOCKER)
    assert role_colour(win, foam.ents, ops.ROLE_ENFORCER)
    assert not role_colour(win, part.ents, ops.ROLE_BLOCKER), "the part was painted too"


def test_a_support_volume_keeps_the_colours_of_the_troubles_of_a_print_to_itself(win):
    """Overhangs, thin walls and tight fits are troubles of a part; a volume only shows its role."""
    win.settings.show_overhangs = True
    part = add(win.doc, t_shape(), "T")
    assert painted_as(win, part.ents, "overhang").any(), \
        "the ends of the bar hang in the air and the view paints them"
    wall = add(win.doc, t_shape().translate((60.0, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    assert not painted_as(win, wall.ents, "overhang").any(), \
        "the volume took the colour of a face that needs support"
    assert role_colour(win, wall.ents, ops.ROLE_BLOCKER)


def test_the_overhang_view_counts_no_faces_of_a_support_volume(win):
    win.settings.show_overhangs = True
    add(win.doc, t_shape(), "T")
    _bed_z, with_part = win.view._overhang_faces()
    assert with_part > 0, "the ends of the bar hang in the air"
    add(win.doc, t_shape().translate((60.0, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    win.view._overhang_cache = None         # a later frame asks the whole model again
    assert win.view._overhang_faces()[1] == with_part, "a support volume is no part to support"


def test_a_support_volume_below_the_model_does_not_move_the_bed_of_the_view(win):
    win.settings.show_overhangs = True
    add(win.doc, cube_man(), "part")
    add(win.doc, cube_man(at=(0.0, 0.0, -30.0)), "wall", ops.ROLE_BLOCKER)
    win.view._overhang_cache = None
    assert win.view._overhang_faces()[0] == pytest.approx(0.0), "the print stands on z = 0"


def test_a_thin_support_volume_is_no_thin_wall_of_the_print(win):
    win.settings.show_thin_walls = True
    thin = box_man(size=(20.0, 0.5, 20.0))
    part = add(win.doc, thin, "wall")
    assert pump_until(lambda: win.view._thin_faces() > 0), "the cut of the wall never landed"
    win.doc.remove_group(part.id)
    win.doc.notify()
    add(win.doc, thin, "blocker", ops.ROLE_BLOCKER)
    assert win.view._thin_faces() == 0, "a support volume is no wall of the print"


def test_a_support_volume_touching_a_part_is_no_tight_fit(win):
    win.settings.show_tight_fits = True
    win.settings.screw_hole_clearance = 0.4
    part = add(win.doc, cube_man(), "part")
    wall = add(win.doc, cube_man(at=(19.8, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    assert pump_until(lambda: win.view._tight_mask_of(part.ents) is not None), \
        "the check of the parts never landed"
    assert win.view._tight_faces() == 0, "the volume stands against the part on purpose"
    assert win.view._tight_mask_of(wall.ents) is None, "a support volume is never checked"


# ------------------------------------------------------------- exports and print checks

def test_no_export_carries_a_support_volume(win, tmp_path):
    add(win.doc, cube_man(), "part")
    add(win.doc, cube_man(at=(100.0, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    parts = win.export_objects()
    assert [p.name for p in parts] == ["part"]
    obj = str(tmp_path / "model.obj")
    fileio.export_obj(obj, parts)
    with open(obj, encoding="utf-8") as fh:
        text = fh.read()
    assert len(re.findall(r"^v ", text, re.M)) == 8, "the volume travelled with the part"
    assert len(re.findall(r"^f ", text, re.M)) == 12, "the volume travelled with the part"


def test_the_3mf_holds_no_support_volume(win, tmp_path):
    import zipfile
    add(win.doc, cube_man(), "part")
    add(win.doc, cube_man(at=(100.0, 0.0, 0.0)), "wall", ops.ROLE_BLOCKER)
    path = str(tmp_path / "model.3mf")
    fileio.export_3mf(path, win.export_objects(), win.doc.materials)
    with zipfile.ZipFile(path) as zf:
        model = zf.read("3D/3dmodel.model").decode("utf-8")
    assert model.count("<triangle ") == 12, "the volume travelled with the parts"


def test_the_print_check_makes_no_problem_of_a_support_volume(win):
    add(win.doc, cube_man(), "part")
    ents = Entities()
    rect(ents, 0.0, 0.0, 30.0, 30.0, z=200.0)            # one open face is no closed volume
    add(win.doc, ents, "wall", ops.ROLE_BLOCKER)
    parts = ops.parts(win.doc)
    assert [p.name for p in parts] == ["part"], "the open volume asked to be checked"
    msgs = ops.check_parts(parts, win.settings.bed_size, win.settings.overhang_limit)
    assert [m for m in msgs if m[0] in ("error", "warning")] == [], msgs


# --------------------------------------------------------------------- the slice dialog

@needs_engine
def test_the_slice_of_the_window_takes_the_support_volumes_along(monkeypatch, tmp_path):
    """The window hands the volumes of the model to the slice as helpers, and the parts without."""
    import volura.slice_dialog as sd
    import volura.slicer as slicer_module
    import volura.window_print as wp
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    add(w.doc, t_shape(), "T")
    add(w.doc, box_man((20.0, 20.0, 41.0), (-17.5, -5.0, -1.0)), "wall", ops.ROLE_BLOCKER)
    seen, errors = {}, []
    monkeypatch.setattr(wp.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))

    def fake(stacks, meshes, out_path, engine=None, timeout=None, on_progress=None, helpers=None):
        seen["printed"] = len(meshes)
        seen["helpers"] = [h[3] for h in (helpers or [])]
        raise RuntimeError("the test stops the slice here")

    monkeypatch.setattr(slicer_module, "slice_meshes", fake)
    monkeypatch.setattr(sd.SliceDialog, "exec", fill)
    try:
        w.slice_dialog()
        assert pump_until(lambda: bool(errors), 180000), "the slice never started"
        assert "stops the slice here" in errors[0], errors
        assert seen["printed"] == 1, "the model sliced more than its part"
        assert seen["helpers"] == [{"anti_overhang_mesh": True}], seen["helpers"]
    finally:
        w.doc.dirty = False
        w.close()


@needs_engine
def test_an_open_support_volume_is_said_before_the_slice(monkeypatch, tmp_path):
    """A volume that is no closed volume blocks nothing and becomes no support, so the window asks
    instead of letting the print come out with support where the user meant to have none."""
    import volura.slice_dialog as sd
    import volura.slicer as slicer_module
    import volura.window_print as wp
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    add(w.doc, t_shape(), "T")
    ents = Entities()
    rect(ents, -17.5, -5.0, 2.5, 15.0, z=10.0)           # an open face is no volume
    add(w.doc, ents, "wall", ops.ROLE_BLOCKER)
    asked, sliced, errors = [], [], []
    monkeypatch.setattr(wp.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    monkeypatch.setattr(wp.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.No)

    def fake(*args, **kwargs):
        sliced.append(args)
        return None

    monkeypatch.setattr(slicer_module, "slice_meshes", fake)
    monkeypatch.setattr(sd.SliceDialog, "exec", fill)
    try:
        w.slice_dialog()
        assert pump_until(lambda: bool(asked), 180000), "nothing was asked"
        assert "wall" in asked[0], asked[0]
        assert not sliced and not errors, "it sliced while you said no"
    finally:
        w.doc.dirty = False
        w.close()


# ------------------------------------------------------------------- with CuraEngine

@needs_engine
def test_a_blocker_keeps_the_support_off_the_arm_it_hangs_over(tmp_path, monkeypatch):
    doc = model_doc_with("blocker")
    plain = model_doc_plain()
    user = {"infill_sparse_density": 0.0, "support_enable": True, "support_type": "buildplate"}
    capture = CommandLine()
    monkeypatch.setattr(slicer, "subprocess", capture)
    both = arms(slice_with_roles(plain, ender3(user), tmp_path / "both.gcode"))
    blocked = arms(slice_with_roles(doc, ender3(user), tmp_path / "blocked.gcode", capture))
    assert both[0] > 0 and both[1] > 0, "both ends of the bar need support"
    assert blocked[0] == 0, "support still stands under the arm the volume hangs over"
    assert blocked[1] > 0, "the other arm lost its support too"
    assert blocked[2] == both[2], "the volume printed itself along with the part"
    assert per_mesh(capture.cmds[-1]) == {"part1.stl": {"anti_overhang_mesh": "true"}}
    assert capture.globals[-1]["support_enable"] is True


@needs_engine
def test_an_enforcer_makes_support_where_the_print_has_none(tmp_path, monkeypatch):
    doc = model_doc_with("enforcer")
    plain = model_doc_plain()
    user = {"infill_sparse_density": 0.0, "support_enable": False}
    capture = CommandLine()
    monkeypatch.setattr(slicer, "subprocess", capture)
    nothing = arms(slice_with_roles(plain, ender3(user), tmp_path / "none.gcode"))
    made = arms(slice_with_roles(doc, ender3(user), tmp_path / "made.gcode", capture))
    assert nothing[0] == 0 and nothing[1] == 0, "a print with support off has no support"
    assert made[0] > 0, "the enforcer made no support of its own"
    assert made[1] == 0, "support turned up where no volume stands"
    assert made[2] == nothing[2], "the enforcer printed itself along with the part"
    assert per_mesh(capture.cmds[-1]) == {"part1.stl": {"support_mesh": "true",
                                                        "support_mesh_drop_down": "true"}}
    assert capture.globals[-1]["support_enable"] is False, \
        "the support came from a global setting, not from the volume"


def model_doc_plain():
    doc = Document()
    add(doc, t_shape(), "T")
    return doc


def model_doc_with(role):
    """The T with a support volume over (blocker) or under (enforcer) the arm at the left."""
    doc = Document()
    add(doc, t_shape(), "T")
    if role == ops.ROLE_BLOCKER:
        add(doc, box_man((20.0, 20.0, 41.0), (-17.5, -5.0, -1.0)), "wall", role)
    else:
        add(doc, box_man(at=(-17.5, -5.0, 10.0)), "foam", role)
    return doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
