"""The thin wall view: which faces are thinner than the printer can lay down.

`thin_walls.thin_mask` is the rule on its own; the rest is View > Face Style > Thin
Walls in the window, which runs without a display (Qt offscreen platform), like
tests/test_overhangs.py. The helpers `rect` and `face_at` are copied from
tests/test_brep.py. The cut of the model runs in the background, so the tests wait for
its answer with `pump_until`, like the jobs of tests/test_simplify.py do.
"""
import os
import sys
import time

import numpy as np
import manifold3d as mf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import thin_walls, theme  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, point_in_polys  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.inference import GeoCache  # noqa: E402
from volura.settings import qsettings  # noqa: E402

SETTING_KEYS = ("show_thin_walls", "min_wall", "overhang_limit")
HINT = "Thin walls: {n} faces under {mm} mm"
HINT_ONE = "Thin walls: {n} face under {mm} mm"


def hint(n, mm):
    """The status text of the thin wall view for `n` faces."""
    return (HINT_ONE if n == 1 else HINT).format(n=n, mm=mm)


def P(x, y, z=0):
    return np.array([x, y, z], dtype=float)


def rect(ents, x0, y0, x1, y1, z=0):
    pl = Plane((0, 0, z), (0, 0, 1))
    hit = [f for f in ents.faces_in_plane(pl)]
    if hit:
        pl = ents.face_plane(hit[0])
        pl = Plane((0, 0, z), pl.n)
    loop = pl.to_local(np.array([P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)]))
    return ents.add_shape(pl, [loop])


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


def box(x0, y0, z0, x1, y1, z1):
    """A closed block from (x0, y0, z0) to (x1, y1, z1)."""
    ents = Entities()
    rect(ents, x0, y0, x1, y1, z=z0)
    f = face_at(ents, (x0 + x1) / 2, (y0 + y1) / 2, z0)
    ents.pushpull(f, z1 - z0 if ents.F[f].normal[2] > 0 else -(z1 - z0))
    assert np.allclose(ents.bbox(), [[x0, y0, z0], [x1, y1, z1]])
    return ents


def shell(inner=19.6, outer=20.0, segments=244):
    """A hollow sphere as a raw Manifold: ~60.000 triangles at 244 segments."""
    return mf.Manifold.sphere(outer, segments) - mf.Manifold.sphere(inner, segments)


def points_of(manifold):
    """The centre of every triangle of a Manifold, as the view would hand them over."""
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=float).reshape(-1, 3)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    return v[t].mean(axis=1)


def thin_faces(ents, min_wall=0.8):
    """The faces of `ents` that lie in a wall thinner than `min_wall`, as their centres."""
    c = GeoCache.build(ents)
    ctr = c["T"].mean(axis=1)                     # the centre of every triangle
    mask = thin_walls.thin_mask(ents.to_manifold(), ctr, min_wall)
    out = {}
    for f in sorted(set(int(i) for i in c["tri_fid"])):
        rows = c["tri_fid"] == f
        out[f] = (tuple(np.round(ctr[rows].mean(axis=0), 3)), bool(mask[rows].any()))
    return out


def thin_at(ents, height, min_wall=0.8):
    """Which of the faces whose centre lies at `height` are thin."""
    return {f: thin for f, (ctr, thin) in thin_faces(ents, min_wall).items()
            if abs(ctr[2] - height) < 1e-6}


# ------------------------------------------------------------------ the rule

def test_a_wall_of_half_a_millimetre_is_thin():
    """Every face of a wall of 0.5 mm: nowhere a circle of 0.8 mm fits in it."""
    assert all(thin for _ctr, thin in thin_faces(box(0, 0, 0, 100, 0.5, 10)).values())


def test_a_wall_of_two_millimetres_may_stay():
    assert not any(thin for _ctr, thin in thin_faces(box(0, 0, 0, 100, 2.0, 10)).values())


def test_a_solid_part_is_never_thin():
    assert not any(thin for _ctr, thin in thin_faces(box(50, 50, 0, 70, 70, 20)).values())


def test_the_wall_is_measured_across_every_axis():
    """A plate of 0.4 mm lying flat is thin, but no horizontal cut can see that: it is
    40 by 40 millimetres wide in every cut. The axes on its side are what find it."""
    plate = box(0, 0, 0, 40, 40, 0.4)
    assert all(thin for _ctr, thin in thin_faces(plate).values())
    c = GeoCache.build(plate)
    thin = np.zeros(len(c["T"]), dtype=bool)
    thin_walls._one_axis(plate.to_manifold(), c["T"].mean(axis=1), thin, 0.8, 0.4)
    assert not thin.any(), "the z axis alone sees a plate of 40 by 40 millimetres"


def test_a_floor_under_a_block_is_only_thin_where_it_is_free():
    """A floor of 0.4 mm under a block of 10 mm: under the block the material is 10.4 mm
    thick, so nothing is thin there; around it the floor is a wall of 0.4 mm."""
    ents = box(0, 0, 0, 40, 40, 0.4)
    rect(ents, 10, 10, 30, 30, z=0.4)
    ents.pushpull(face_at(ents, 20, 20, 0.4), 10.0)
    assert list(thin_at(ents, 0.2).values()) == [True] * 4, "the four edges of the floor"
    assert list(thin_at(ents, 0.4).values()) == [True], "the top of the floor, around the block"
    assert list(thin_at(ents, 0.0).values()) == [False], "under the block the material is thick"
    assert not any(thin_at(ents, 5.4).values()), "the walls of the block are 10 mm thick"
    assert list(thin_at(ents, 10.4).values()) == [False], "and so is its top"


def test_a_wide_part_is_cut_with_fewer_planes():
    """`MAX_PLANES` per axis is the most a part ever asks for, however tall it is; the
    step grows with it, and the axis along the wall still sees its 0.5 mm thickness."""
    assert 400 / (0.6 / 2) > thin_walls.MAX_PLANES, "this wall asks for more planes than there are"
    assert all(thin for _ctr, thin in thin_faces(box(0, 0, 0, 400, 0.5, 400), 0.6).values())


def test_a_hollow_sphere_is_thin_over_its_whole_shell():
    """0.8 mm of shell on a sphere of 40 mm: no face may be left out, however it stands,
    and a model of 60.000 triangles may not take minutes.

    A cut runs in about three seconds on a normal machine; the ceiling here only guards
    against a way of working that grows with the number of triangles instead of the number
    of cut planes - which on a slow test machine would still be far below it."""
    shell = mf.Manifold.sphere(20.0, 244) - mf.Manifold.sphere(19.6, 244)
    points = points_of(shell)
    assert len(points) > 55000, "a shell this thick asks for a real mesh"
    started = time.perf_counter()
    mask = thin_walls.thin_mask(shell, points, 0.8)
    took = time.perf_counter() - started
    assert mask.all(), f"{len(points) - int(mask.sum())} faces of the shell were left out"
    assert took < 20.0, f"the cut took {took:.1f} s"


# ------------------------------------------------------------------ edge cases

def test_an_empty_part_says_nothing():
    assert thin_walls.thin_mask(mf.Manifold(), [[0.0, 0.0, 0.0]], 0.8).tolist() == [False]
    assert thin_walls.thin_mask(shell(20, 20, 24), [[0.0, 0.0, 0.0]], 0.8).tolist() == [False]


def test_no_faces_nothing_marked():
    assert thin_walls.thin_mask(shell(), [], 0.8).tolist() == []


def test_a_part_that_is_only_a_face_is_not_a_thin_wall():
    """A loose face has no volume; a cut of it is empty, which is not the same as thin."""
    ents = Entities()
    rect(ents, 0, 0, 10, 10)
    assert not any(thin for _ctr, thin in thin_faces(ents).values())


def test_a_point_beside_the_part_is_not_thin():
    ents = box(0, 0, 0, 100, 0.5, 10)
    wall = ents.to_manifold()
    assert thin_walls.thin_mask(wall, [[500.0, 500.0, 5.0]], 0.8).tolist() == [False]
    assert thin_walls.thin_mask(wall, [[50.0, 25.0, -20.0]], 0.8).tolist() == [False]
    assert thin_walls.thin_mask(wall, [[50.0, 0.25, 5.0]], 0.8).tolist() == [True]


def test_a_wall_of_zero_is_no_wall():
    for too_small in (0.0, -0.4):
        with pytest.raises(ValueError):
            thin_walls.thin_mask(shell(), [[0.0, 0.0, 0.0]], too_small)


@pytest.mark.parametrize("step", (None, 0.0, 50.0, 1e-4))
def test_the_width_of_the_cuts_changes_the_work_but_not_the_answer(step):
    """The default is half the wall; a wider step saves cuts and a narrower one costs
    more, but a wall of 0.5 mm is thin whatever the width of the cuts."""
    ents = box(0, 0, 0, 100, 0.5, 10)
    points = GeoCache.build(ents)["T"].mean(axis=1)
    assert thin_walls.thin_mask(ents.to_manifold(), points, 0.8, step).all()


# ------------------------------------------------------------------ the window

@pytest.fixture()
def win():
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_thin_walls = False
    w.settings.min_wall = 0.8
    w.settings.overhang_limit = 45
    w.a_thin_walls.setChecked(False)
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


def add(win, ents, name):
    g = win.doc.add_group(ents, "#f2f2f2", name)
    win.doc.notify()
    return g


def cut(win, *ents_list):
    """Ask for the cut of these geometries and wait until every answer has landed.

    The first ask starts the work in the background and says nothing yet - which is how
    the model keeps its own colours until the answer is there."""
    for e in ents_list:
        assert win.view._thin_mask_of(e) is None, "an answer that is not there says nothing"
    assert pump_until(lambda: all(win.view._thin_mask_of(e) is not None for e in ents_list)), \
        "the cut never came to an end"


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def tinted(win, ents, colour):
    """The vertices the view paints in `colour` (a name of theme.COLORS), as one bool per
    vertex. A frame sets the overhang context itself; a test that goes straight to the
    render data has to do that too (see tests/test_overhangs.py)."""
    win.view._overhang_ctx = win.view._overhang_context()
    rc = win.view._render_data(ents, None)
    rgb = np.array(theme.rgb(colour), dtype=np.float32)
    return np.all(np.isclose(rc["front"][:, :3], rgb), axis=1)


# ------------------------------------------------------------------ the action

def test_the_action_is_a_checkable_view_style(win):
    """View > Face Style > Thin Walls is a switch like Overhangs, and it is stored."""
    a = win.a_thin_walls
    assert a.isCheckable() and a.isEnabled()
    a.trigger()
    assert win.settings.show_thin_walls is True and a.isChecked()
    assert type(win.settings)().show_thin_walls is True
    a.trigger()
    assert win.settings.show_thin_walls is False
    assert type(win.settings)().show_thin_walls is False


def test_the_thinnest_wall_is_stored(win):
    assert win.settings.min_wall == 0.8
    win.settings.min_wall = 1.2
    win.settings.save()
    assert type(win.settings)().min_wall == 1.2
    for strange in (0.0, -1.0, 99.0, float("nan")):
        win.settings.min_wall = strange
        win.settings.save()
        assert type(win.settings)().min_wall == 0.8, f"{strange} is no wall width"


# ------------------------------------------------------------------ the view

def test_a_thick_part_says_nothing(win):
    add(win, box(100, 100, 0, 110, 110, 10), "solid")
    win.a_thin_walls.trigger()
    cut(win, win.doc.groups[0].ents)
    assert "Thin wall" not in win.hint.text()
    assert not tinted(win, win.doc.groups[0].ents, "thin").any()


def test_a_thin_wall_is_tinted_and_counted(win):
    """A wall of 0.5 mm under a wall of 2 mm: only the thin one is marked, and the status
    bar counts faces - all six of the thin wall, none of the thick one."""
    add(win, box(0, 0, 0, 100, 0.5, 10), "thin")
    add(win, box(0, 20, 0, 100, 22, 10), "thick")
    win.a_thin_walls.trigger()
    cut(win, win.doc.groups[0].ents, win.doc.groups[1].ents)
    assert win.hint.text() == hint(6, "0.8")
    assert int(tinted(win, win.doc.groups[0].ents, "thin").sum()) == 36, "6 faces of 2 triangles"
    assert not tinted(win, win.doc.groups[1].ents, "thin").any()
    win.a_thin_walls.trigger()
    assert "Thin wall" not in win.hint.text(), "the count must not stay in the bar"
    assert win.hint.text() == win.view.tool.status_text()
    assert not tinted(win, win.doc.groups[0].ents, "thin").any(), "off: nothing is tinted"


def test_the_thinness_of_the_wall_you_ask_for(win):
    """The two wide faces of a wall of 0.5 mm are 0.5 mm apart: a circle of 0.4 mm fits
    between them, so under 0.4 mm nothing of it is thin - not even the faces where the wall
    ends, whose square corners a round circle cannot reach. Another width of the wall is
    another question, so it needs a new cut, and the bar no longer says it is waiting."""
    ents = add(win, box(0, 0, 0, 100, 0.5, 10), "thin").ents
    win.a_thin_walls.trigger()
    cut(win, ents)
    assert win.hint.text() == hint(6, "0.8")
    assert int(tinted(win, ents, "thin").sum()) == 36
    win.settings.min_wall = 0.4
    cut(win, ents)
    assert win.view._thin_mask_of(ents).sum() == 0
    assert not tinted(win, ents, "thin").any()
    assert win.hint.text() != tr("Checking wall thickness…")
    answer = win.view._thin_mask_of(ents)
    assert win.view._thin_mask_of(ents) is answer, "and the answer is kept without a new cut"


def test_the_bar_says_what_a_cut_waits_for(win):
    """While the model is cut the status bar says so - a big model takes seconds and the
    window takes no input meanwhile - and the count takes its place as soon as the answer
    has landed."""
    ents = add(win, box(0, 0, 0, 100, 0.5, 10), "thin").ents
    win.a_thin_walls.trigger()
    assert win.view._thin_mask_of(ents) is None
    assert win.hint.text() == tr("Checking wall thickness…")
    cut(win, ents)
    assert win.hint.text() == hint(6, "0.8")


def test_a_running_cut_is_not_asked_twice(win, monkeypatch):
    """While the cut of one geometry runs, asking again must not start a second one; a
    different wall width is a different question and does."""
    asked = []
    monkeypatch.setattr(win.view, "_run_thin_job", lambda job: asked.append(job[0]))
    ents = add(win, box(0, 0, 0, 100, 0.5, 10), "thin").ents
    win.a_thin_walls.trigger()
    assert win.view._thin_mask_of(ents) is None
    assert win.view._thin_mask_of(ents) is None
    assert asked.count(ents) == 1, "one cut for one geometry"
    win.settings.min_wall = 0.4
    assert win.view._thin_mask_of(ents) is None
    assert asked.count(ents) == 2, "another wall width is another cut"


def test_an_edit_asks_for_a_new_cut(win):
    """A wall of 2 mm made from the thin one by pushing its face out: the answer of the
    old version may not be shown for the new one."""
    ents = add(win, box(0, 0, 0, 100, 0.5, 10), "thin").ents
    win.a_thin_walls.trigger()
    cut(win, ents)
    assert tinted(win, ents, "thin").any()
    win.doc.checkpoint()
    ents.pushpull(face_at(ents, 50, 0, 5), 1.5)
    win.doc.notify()
    assert win.view._thin_mask_of(ents) is None, "the model moved on; the answer did not"
    cut(win, ents)
    assert not tinted(win, ents, "thin").any(), "now the wall is 2 mm thick"


def test_the_overhang_colour_wins(win):
    """The bottom of a floating thin wall both hangs over and is thin: it stays red,
    because support is the worse of the two problems."""
    add(win, box(100, 100, 0, 110, 110, 10), "on the bed")
    add(win, box(0, 0, 20, 10, 0.5, 30), "floating")
    win.a_overhangs.trigger()
    win.a_thin_walls.trigger()
    cut(win, win.doc.groups[0].ents, win.doc.groups[1].ents)
    over = "Overhangs: 1 face needs support (over {limit}°)".format(
        limit=int(win.settings.overhang_limit))
    assert win.hint.text() == over + " · " + hint(6, "0.8")
    red = tinted(win, win.doc.groups[1].ents, "overhang")
    magenta = tinted(win, win.doc.groups[1].ents, "thin")
    assert int(red.sum()) == 6, "the bottom face: two triangles, red"
    assert int(magenta.sum()) == 30, "the other five faces of the wall, magenta"
    assert not (red & magenta).any()


def test_a_part_that_is_no_volume_is_skipped(win):
    """A loose face on the sheet is not a part: the cut of it is empty and nothing says
    it is thin."""
    ents = Entities()
    rect(ents, 0, 0, 10, 10)
    add(win, ents, "face only")
    win.a_thin_walls.trigger()
    cut(win, win.doc.groups[0].ents)
    assert "Thin wall" not in win.hint.text()
    assert not tinted(win, ents, "thin").any()


def test_the_thinnest_wall_is_a_preference(win):
    """Window > Preferences > Drawing: Thinnest wall, 0.1..10 mm, stored like the rest."""
    from volura.app import PreferencesDialog
    was = win.settings.min_wall
    try:
        dlg = PreferencesDialog(win, "Drawing")
        assert dlg.min_wall.value() == was
        assert dlg.min_wall.minimum() == pytest.approx(0.1)
        assert dlg.min_wall.maximum() == pytest.approx(10.0)
        dlg.min_wall.setValue(1.6)
        dlg.accept()
        assert win.settings.min_wall == pytest.approx(1.6)
        assert type(win.settings)().min_wall == pytest.approx(1.6)
    finally:
        win.settings.min_wall = was
        win.settings.save()


def test_a_deleted_part_leaves_no_answer_behind(win):
    """The answer belongs to a geometry; when that leaves the document its cut and its
    answer go with it, and a late answer of a part that is gone is dropped."""
    thin = add(win, box(0, 0, 0, 100, 0.5, 10), "thin")
    other = add(win, box(0, 20, 0, 100, 22, 10), "thick")
    win.a_thin_walls.trigger()
    cut(win, thin.ents, other.ents)
    assert win.hint.text() == hint(6, "0.8")
    gid = thin.id
    win.doc.remove_group(gid)
    win.doc.notify()
    win.view._prune_rcache()
    assert id(thin.ents) not in win.view._thin_cache
    assert "Thin wall" not in win.hint.text()
    assert win.view._thin_mask_of(other.ents) is not None, "the rest keeps its answer"


def _centres(solid):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3]
    return verts[np.asarray(mesh.tri_verts)].mean(axis=1)


def test_the_rim_of_a_one_millimetre_tray_is_not_thin():
    """A circle of 0.8 mm cannot reach into a square corner, but a corner is no wall: the
    top of the 1 mm wall of an open tray and its outer edges stay as they are."""
    import manifold3d as mf
    tray = (mf.Manifold.cube([150, 150, 60])
            - mf.Manifold.cube([148, 148, 59]).translate([1, 1, 1.5]))
    assert not thin_walls.thin_mask(tray, _centres(tray), 0.8).any()


def test_a_cut_that_grazes_a_round_pin_does_not_make_it_thin():
    """A plane beside the axis of a pin cuts a narrow strip of it, which is one axis on its
    own; a solid pin of Ø 6 mm has no thin face from any side."""
    import manifold3d as mf
    pins = mf.Manifold.cube([60, 60, 1.5])
    for i in range(4):
        for j in range(4):
            pins = pins + mf.Manifold.cylinder(40, 3, 3, 64).translate(
                [10 + 13 * i, 10 + 13 * j, 0])
    assert not thin_walls.thin_mask(pins, _centres(pins), 0.8).any()
