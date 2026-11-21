"""The tight fit view: which faces come closer to another part than the printer's clearance.

`clearance.tight_masks` is the rule on its own; the rest is View > Face Style > Tight
Fits in the window, which runs without a display (Qt offscreen platform), like
tests/test_thin_walls.py. The helpers `rect` and `face_at` are copied from
tests/test_brep.py. The check covers the whole model at once and runs in the
background, so the tests wait for its answer with `pump_until`, like the jobs of
tests/test_simplify.py do.
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

from volura import clearance, theme  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, point_in_polys  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.inference import GeoCache  # noqa: E402
from volura.settings import qsettings  # noqa: E402

SETTING_KEYS = ("show_tight_fits", "show_thin_walls", "show_overhangs", "min_wall",
                "overhang_limit", "screw_hole_clearance")
HINT = "Tight fits: {n} faces closer than {mm} mm to another part"
HINT_ONE = "Tight fits: {n} face closer than {mm} mm to another part"
CHECKING = "Checking clearances…"


def hint(n, mm):
    """The status text of the tight fit view for `n` faces."""
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


def solid(x0, y0, z0, x1, y1, z1):
    """The same block as a Manifold, for the tests of the rule itself."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def centres(part):
    """The centre of every triangle of a Manifold, as the view would hand them over."""
    mesh = part.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=float).reshape(-1, 3)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    return v[t].mean(axis=1)


def tight(ents, room=0.3, others=()):
    """The faces of `ents` that come closer than `room` to one of `others`, as a bool per
    face together with its centre - the way the view reads its answer back."""
    c = GeoCache.build(ents)
    parts = [(ents.to_manifold(), c["T"].mean(axis=1))]
    parts += [(e.to_manifold(), GeoCache.build(e)["T"].mean(axis=1)) for e in others]
    masks = clearance.tight_masks(parts, room)
    ctr, out = c["T"].mean(axis=1), {}
    for f in sorted(set(int(i) for i in c["tri_fid"])):
        rows = c["tri_fid"] == f
        out[f] = (tuple(np.round(ctr[rows].mean(axis=0), 3)), bool(masks[0][rows].any()))
    return out


def near_faces(ents, others, room=0.3):
    """Which faces of `ents` are tight, as {face: tight}."""
    return {f: was for f, (_ctr, was) in tight(ents, room, others).items()}


# ------------------------------------------------------------------ the rule

def test_two_parts_a_tenth_apart_are_tight_where_they_face():
    """Two blocks of 10 mm with 0.1 mm between them and a clearance of 0.3 mm: the two
    faces that look at each other are tight, the four far ones are not."""
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10)
    pa, pb = centres(a), centres(b)
    ma, mb = clearance.tight_masks([(a, pa), (b, pb)], 0.3)
    assert ma[pa[:, 0] > 9.9].all(), "the face of A that looks at B"
    assert not ma[pa[:, 0] < 0.1].any(), "and nothing of the face behind it"
    assert mb[pb[:, 0] < 10.2].all(), "B answers A in the same place"
    assert not mb[pb[:, 0] > 20.0].any()


def test_a_millimetre_of_room_is_enough():
    a, b = solid(0, 0, 0, 10, 10, 10), solid(11.0, 0, 0, 21.0, 10, 10)
    ma, mb = clearance.tight_masks([(a, centres(a)), (b, centres(b))], 0.3)
    assert not ma.any() and not mb.any()


def test_parts_that_run_into_each_other_are_tight():
    """Where two parts share a place there is no room left for any plastic at all, so
    those faces are the tightest the view can find - and the part buried in the other is
    tight over its whole side, not only where it pokes through."""
    a, b = solid(0, 0, 0, 10, 10, 10), solid(9, 0, 0, 19, 10, 10)
    pa = centres(a)
    ma, mb = clearance.tight_masks([(a, pa), (b, centres(b))], 0.3)
    buried = (pa[:, 0] > 9.5) & (pa[:, 1] > 1.0) & (pa[:, 1] < 9.0)
    assert buried.any() and ma[buried].all(), "the faces of A inside B"
    assert not ma[pa[:, 0] < 0.1].any(), "the far side of A is 9 mm from B"
    assert mb.any(), "B is tight too, where A runs into it"


def test_two_parts_in_exactly_the_same_place_are_tight_whole():
    """Two blocks in the same place: every face of each lies exactly on the surface of the
    other, so there is no distance to measure and nothing to call inside - and every face
    is tight all the same."""
    a, b = solid(0, 0, 0, 20, 20, 20), solid(0, 0, 0, 20, 20, 20)
    pa, pb = centres(a), centres(b)
    ma, mb = clearance.tight_masks([(a, pa), (b, pb)], 0.3)
    assert ma.all() and mb.all(), f"{int((~ma).sum())} + {int((~mb).sum())} faces were left out"


def test_a_part_buried_in_another_is_tight_whole():
    """A ball deep inside a block touches nothing and is nowhere nearer than 47 mm, and
    still every one of its faces is tight: it has no room left at all."""
    ball = mf.Manifold.sphere(3.0, 40).translate([50.0, 50.0, 50.0])
    block = solid(0, 0, 0, 100, 100, 100)
    mb, mk = clearance.tight_masks([(ball, centres(ball)), (block, centres(block))], 0.3)
    assert mb.all(), f"{len(mb) - int(mb.sum())} faces of the ball were left out"
    assert not mk.any(), "the block is 47 mm from the ball"


def test_a_part_buried_in_a_curved_part_is_tight_whole():
    """The block above hides its even-odd test behind two big faces per side; here the
    triangles of the outer ball are small enough to stand sorted in the cells, so a point
    deep inside one is asked about the triangles its ray runs through - and has to count
    every one of them once, however many cells it fills along the way."""
    inner = mf.Manifold.sphere(3.0, 60).translate([40.0, 40.0, 40.0])
    outer = mf.Manifold.sphere(12.0, 60).translate([40.0, 40.0, 40.0])
    pi, po = centres(inner), centres(outer)
    mi, mo = clearance.tight_masks([(inner, pi), (outer, po)], 0.3)
    assert mi.all(), f"{len(mi) - int(mi.sum())} faces of the inner ball were left out"
    assert not mo.any(), "the skin of the outer ball stands 9 mm from the inner one"


def test_a_ball_hanging_in_a_hollow_box_is_tight_where_it_neares_the_wall():
    """A ball hanging 0.15 mm clear inside a hollow box: it touches nothing, so the ball is
    only tight where it almost reaches one of the six walls it hangs between, and the box
    only where the ball comes near its inside - the two answers come from different parts
    of the rule, the ball from the even-odd test and the box from the measured distance."""
    room = 0.3
    centre = P(10.5, 10.5, 10.5)              # the hollow of the box runs 0.2 .. 20.8
    ball = mf.Manifold.sphere(10.15, 60).translate(centre)
    shell = (mf.Manifold.cube([21, 21, 21])
             - mf.Manifold.cube([20.6, 20.6, 20.6]).translate([0.2, 0.2, 0.2])
             ).refine_to_length(2.5)
    pb, ps = centres(ball), centres(shell)
    mb, ms = clearance.tight_masks([(ball, pb), (shell, ps)], room)
    gap = (10.3 - np.abs(pb - centre)).min(axis=1)     # the wall of the hollow, mm
    assert mb.shape == gap.shape and (mb == (gap < room)).all(), \
        f"{int((mb != (gap < room)).sum())} faces of the ball answered wrong"
    reach = np.linalg.norm(ps - centre, axis=1) - 10.15
    sure = np.abs(reach - room) > 0.02                 # a polyhedron is not a sphere
    assert (ms == (reach < room))[sure].all(), \
        f"{int((ms != (reach < room))[sure].sum())} faces of the box answered wrong"
    outside = np.any((ps < 0.2) | (ps > 20.8), axis=1)
    assert mb.any() and ms.any() and not ms[outside].any(), "the outside of the box is free"


def test_one_part_alone_says_nothing():
    a = solid(0, 0, 0, 10, 10, 10)
    assert not clearance.tight_masks([(a, centres(a))], 0.3)[0].any()


def test_three_parts_in_a_row_are_tight_on_both_sides():
    """A touches B and B touches C; A and C stand 10 mm apart: B is tight on both of its
    faces, A and C only on the one that looks at B."""
    a, b, c = (solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10),
               solid(20.2, 0, 0, 30.2, 10, 10))
    pa, pb, pc = centres(a), centres(b), centres(c)
    ma, mb, mc = clearance.tight_masks([(a, pa), (b, pb), (c, pc)], 0.3)
    assert ma[pa[:, 0] > 9.9].all() and not ma[pa[:, 0] < 0.1].any()
    assert mb[pb[:, 0] < 10.2].all() and mb[pb[:, 0] > 20.0].all()
    assert mc[pc[:, 0] < 20.4].all() and not mc[pc[:, 0] > 30.0].any()


def test_the_middle_of_a_wall_between_two_parts_is_not_tight():
    """The faces of B that look away from A and C are 10 mm from both; a part is only
    tight where it is close, not all over."""
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10)
    pb = centres(b)
    _ma, mb = clearance.tight_masks([(a, centres(a)), (b, pb)], 0.3)
    middle = (pb[:, 0] > 10.5) & (pb[:, 0] < 19.7)
    assert middle.any() and not mb[middle].any()


# ------------------------------------------------------------------ edge cases

def test_no_parts_and_a_part_without_faces_say_nothing():
    assert clearance.tight_masks([], 0.3) == []
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10)
    assert clearance.tight_masks([(a, []), (b, centres(b))], 0.3)[0].tolist() == []
    assert not clearance.tight_masks([(mf.Manifold(), [[5.0, 5.0, 5.0]]),
                                      (b, centres(b))], 0.3)[0].any()


def test_a_part_that_is_only_a_face_is_not_a_part():
    """A loose face has no volume, so it is not a part to keep room for: the block next to
    it stays as it is."""
    ents = Entities()
    rect(ents, 0, 0, 10, 10)
    near = box(10.1, 0, 0, 20.1, 10, 10)
    assert not any(near_faces(near, [ents]).values())


def test_a_point_outside_every_part_is_not_tight():
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10)
    far = [[-500.0, 500.0, 5.0]]
    assert clearance.tight_masks([(a, far), (b, centres(b))], 0.3)[0].tolist() == [False]


def test_a_point_above_the_edge_two_faces_share_is_still_inside():
    """A point in the middle of a block stands exactly above the diagonal that splits its
    top face; a ray straight up crosses that edge twice and the even-odd test would call
    the point outside. Every point of a row along that diagonal has to be found inside."""
    twin = solid(0, 0, 0, 40, 40, 40)
    inside = [clearance.tight_masks([(solid(20.0 + k * 1e-7, 20.0 - k * 1e-7, 20.0,
                                            20.4, 20.4, 20.4),
                                      [[20.0 + k * 1e-7, 20.0 - k * 1e-7, 20.0]]),
                                     (twin, centres(twin))], 0.3)[0].all()
                for k in range(60)]
    assert all(inside), f"{inside.count(False)} of {len(inside)} points were called outside"


def test_a_room_of_zero_is_no_question():
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10.1, 0, 0, 20.1, 10, 10)
    for too_small in (0.0, -0.4):
        with pytest.raises(ValueError):
            clearance.tight_masks([(a, centres(a)), (b, centres(b))], too_small)


def test_many_parts_in_a_row_cost_no_more_than_the_pairs_they_make():
    """Two hundred blocks of 2 mm in a row with 0.1 mm between them: every part is tight
    with the two beside it and with nobody further off, and the check costs what the pairs
    of neighbours cost - not what the number of parts is, and not what a part that stands
    far from everything costs in triangles."""
    blocks = [solid(2.1 * i, 0, 0, 2.1 * i + 2, 4, 4) for i in range(200)]
    parts = [(b, centres(b)) for b in blocks]
    started = time.perf_counter()
    masks = clearance.tight_masks(parts, 0.3)
    took = time.perf_counter() - started
    assert took < 10.0, f"the row took {took:.1f} s"
    for i, (mask, (_part, points)) in enumerate(zip(masks, parts)):
        near = points[:, 0] < 2.1 * i + 0.05
        far = (points[:, 0] > 2.1 * i + 0.05) & (points[:, 0] < 2.1 * i + 1.95)
        assert far.any() and not mask[far].any(), f"part {i}: the faces along the row"
        if i:
            assert mask[near].all(), f"part {i} is tight with the one before it"
        else:
            assert not mask[near].any(), "the first block has nothing before it"
    assert not masks[-1][parts[-1][1][:, 0] > 2.1 * 199 + 1.95].any(), "and so has the last"


# ------------------------------------------------------------------ the cost

def test_a_box_in_a_hollow_box_is_checked_in_seconds():
    """A block of 40 mm with 0.2 mm of room in a hollow box of 40.8 mm, both of about
    20.000 triangles: the inner block is tight over its whole surface and the box only on
    the inside of its walls, and the check may not take longer than the thin wall cut of a
    sphere of 60.000 triangles (tests/test_thin_walls.py)."""
    inner = solid(0.4, 0.4, 0.4, 40.4, 40.4, 40.4).refine_to_length(1.1)
    shell = (mf.Manifold.cube([40.8, 40.8, 40.8])
             - mf.Manifold.cube([40.4, 40.4, 40.4]).translate([0.2, 0.2, 0.2])
             ).refine_to_length(1.5)
    pi, po = centres(inner), centres(shell)
    assert 15000 < len(pi) < 25000 and 15000 < len(po) < 25000, "two real meshes"
    started = time.perf_counter()
    mi, mo = clearance.tight_masks([(inner, pi), (shell, po)], 0.3)
    took = time.perf_counter() - started
    # the centre of a face of the block stands 0.2 mm from the wall around it, and the
    # centre of a face of the box as far from the block as its own distance to it
    room = np.minimum(pi - 0.2, 40.6 - pi).min(axis=1)
    off = np.maximum(np.maximum(0.4 - po, po - 40.4), 0.0)
    assert (mi == (room < 0.3)).all(), "the block is tight wherever the box is close"
    assert (mo == (np.sqrt((off * off).sum(axis=1)) < 0.3)).all(), "and so is the box"
    assert mo.any() and mo.sum() < len(mo), "the inside of the walls, and not the outside"
    assert took < 5.0, f"the check took {took:.1f} s"


# ------------------------------------------------------------------ the window

@pytest.fixture()
def win():
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_tight_fits = False
    w.settings.show_thin_walls = False
    w.settings.show_overhangs = False
    w.settings.min_wall = 0.8
    w.settings.overhang_limit = 45
    w.settings.screw_hole_clearance = 0.2
    w.a_tight_fits.setChecked(False)
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


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def wait_for(win, *ents_list):
    """Wait until the check of the model has answered for these parts."""
    assert pump_until(lambda: all(win.view._tight_mask_of(e) is not None for e in ents_list)), \
        "the check never came to an end"


def checked(win, *ents_list):
    """Ask for the check of the model and wait until every answer has landed.

    The first ask starts the work in the background and says nothing yet - which is how
    the model keeps its own colours until the answer is there."""
    for e in ents_list:
        assert win.view._tight_mask_of(e) is None, "an answer that is not there says nothing"
    wait_for(win, *ents_list)


def tinted(win, ents, colour):
    """The vertices the view paints in `colour` (a name of theme.COLORS), as one bool per
    vertex. A frame sets the overhang context itself; a test that goes straight to the
    render data has to do that too (see tests/test_overhangs.py)."""
    win.view._overhang_ctx = win.view._overhang_context()
    rc = win.view._render_data(ents, None)
    rgb = np.array(theme.rgb(colour), dtype=np.float32)
    return np.all(np.isclose(rc["front"][:, :3], rgb), axis=1)


def painted(win, ents, colour):
    """How many vertices of `ents` the view paints in `colour`; a box face is two
    triangles, so a painted face of a box shows as six of them."""
    return int(tinted(win, ents, colour).sum())


# ------------------------------------------------------------------ the action

def test_the_action_is_a_checkable_view_style(win):
    """View > Face Style > Tight Fits is a switch like Thin Walls, and it is stored."""
    a = win.a_tight_fits
    assert a.isCheckable() and a.isEnabled()
    assert a.toolTip().startswith("Mark the faces")
    assert "Clearance Test" in a.toolTip(), "the tooltip says where the clearance comes from"
    a.trigger()
    assert win.settings.show_tight_fits is True and a.isChecked()
    assert type(win.settings)().show_tight_fits is True
    a.trigger()
    assert win.settings.show_tight_fits is False
    assert type(win.settings)().show_tight_fits is False


def test_the_room_is_the_clearance_of_the_print_menu(win):
    """The view asks about the clearance of 3D Printing > Clearance Test, which is
    Settings.screw_hole_clearance; the view brings no clearance setting of its own, only
    the switch of View > Face Style."""
    assert win.settings.screw_hole_clearance == pytest.approx(0.2)
    own = [k for k in dir(win.settings) if "clearance" in k and "tight" in k]
    assert own == [], f"the view brought its own clearance: {own}"


# ------------------------------------------------------------------ the view

def test_two_parts_a_tenth_apart_are_tinted_and_counted(win):
    """Two blocks of 10 mm with 0.1 mm between them and a clearance of 0.2 mm: the face of
    each that looks at the other is cyan and counted - two faces, six vertices each."""
    a = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    b = add(win, box(10.1, 0, 0, 20.1, 10, 10), "right").ents
    win.a_tight_fits.trigger()
    checked(win, a, b)
    assert win.hint.text() == hint(2, "0.2")
    assert painted(win, a, "tight") == 6, "the one face that looks at the other"
    assert painted(win, b, "tight") == 6
    win.a_tight_fits.trigger()
    assert "Tight fit" not in win.hint.text(), "the count must not stay in the bar"
    assert win.hint.text() == win.view.tool.status_text()
    assert not tinted(win, a, "tight").any(), "off: nothing is tinted"


def test_two_parts_a_millimetre_apart_say_nothing(win):
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    add(win, box(11.0, 0, 0, 21.0, 10, 10), "right")
    win.a_tight_fits.trigger()
    checked(win, win.doc.groups[0].ents, win.doc.groups[1].ents)
    assert "Tight fit" not in win.hint.text()
    assert not tinted(win, win.doc.groups[0].ents, "tight").any()


def test_the_room_you_ask_for_is_the_room_you_get(win):
    """The same two blocks are tight under a clearance of 0.2 mm and free under 0.05 mm;
    another clearance is another question, so it needs a new check, and the bar no longer
    says it is waiting."""
    a = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    b = add(win, box(10.1, 0, 0, 20.1, 10, 10), "right").ents
    win.a_tight_fits.trigger()
    checked(win, a, b)
    assert win.hint.text() == hint(2, "0.2")
    win.settings.screw_hole_clearance = 0.05
    checked(win, a, b)
    assert win.view._tight_mask_of(a).sum() == 0
    assert not tinted(win, a, "tight").any()
    assert win.hint.text() != tr(CHECKING)
    answer = win.view._tight_mask_of(a)
    assert win.view._tight_mask_of(a) is answer, "and the answer is kept without a new check"


def test_no_clearance_no_check(win, monkeypatch):
    """A clearance of nothing says the printer lays nothing between two parts; there is no
    distance under that to look for, so the view asks for no check at all and the bar does
    not start to wait for one."""
    a = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    add(win, box(10.1, 0, 0, 20.1, 10, 10), "right")
    win.settings.screw_hole_clearance = 0.0
    asked = []
    monkeypatch.setattr(win.view, "_run_tight_job", lambda key: asked.append(key))
    win.a_tight_fits.trigger()
    assert win.view._tight_mask_of(a) is None
    assert win.view._tight_faces() == 0
    assert asked == [], "a clearance of nothing is no question"
    assert win.hint.text() != tr(CHECKING)
    assert not tinted(win, a, "tight").any()


def test_the_bar_says_what_the_check_waits_for(win):
    """While the model is checked the status bar says so - a big model takes seconds and
    the window takes no input meanwhile - and the count takes its place as soon as the
    answer has landed."""
    a = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    add(win, box(10.1, 0, 0, 20.1, 10, 10), "right")
    win.a_tight_fits.trigger()
    assert win.view._tight_mask_of(a) is None
    assert win.hint.text() == tr(CHECKING)
    checked(win, a)
    assert win.hint.text() == hint(2, "0.2")


def test_a_running_check_is_not_asked_twice(win, monkeypatch):
    """While the model is being checked, asking again must not start a second check;
    another clearance is another question and does."""
    asked = []
    monkeypatch.setattr(win.view, "_run_tight_job", lambda key: asked.append(key))
    a = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    add(win, box(10.1, 0, 0, 20.1, 10, 10), "right")
    win.a_tight_fits.trigger()
    assert win.view._tight_mask_of(a) is None
    assert win.view._tight_mask_of(a) is None
    assert len(asked) == 1, "one check for one model"
    win.settings.screw_hole_clearance = 0.4
    assert win.view._tight_mask_of(a) is None
    assert len(asked) == 2, "another clearance is another check"


def test_an_edit_asks_for_a_new_check(win):
    """The right block moved 2 mm away: the answer of the old model may not be shown for
    the new one, and one part that moved asks for the whole model again."""
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    right = add(win, box(10.1, 0, 0, 20.1, 10, 10), "right").ents
    win.a_tight_fits.trigger()
    checked(win, right)
    assert tinted(win, right, "tight").any()
    win.doc.checkpoint()
    right.pushpull(face_at(right, 10.1, 5, 5), -2.0)      # 2 mm of room
    win.doc.notify()
    assert win.view._tight_mask_of(right) is None, "the model moved on; the answer did not"
    checked(win, right)
    assert not tinted(win, right, "tight").any(), "now there is 2 mm of room"


def test_the_other_colours_win(win):
    """Five parts: a block on the bed, a thin wall and a thick one 0.1 mm apart, and two
    plates stacked with 0.1 mm between them, the last four of them off the bed. The thin
    wall stays magenta where it is tight and the bottom of a plate stays red - a nozzle
    and support are the worse troubles - so only a face that is nothing but close turns
    cyan."""
    add(win, box(0, 0, 0, 10, 10, 2), "on the bed")
    thin = add(win, box(0, 20, 20, 10, 20.5, 30), "thin wall").ents
    thick = add(win, box(0, 20.6, 20, 10, 22.6, 30), "thick wall").ents
    under = add(win, box(30, 0, 20, 50, 10, 22), "plate").ents
    over = add(win, box(30, 0, 22.1, 50, 10, 24.1), "plate above").ents
    win.a_overhangs.trigger()
    win.a_thin_walls.trigger()
    win.a_tight_fits.trigger()
    checked(win, thin, thick, under, over)
    assert win.hint.text() == ("Overhangs: 4 faces need support (over {limit}°) · "
                               "Thin walls: {t} faces under {w} mm · "
                               "Tight fits: {f} faces closer than {mm} mm to another part"
                               ).format(limit=int(win.settings.overhang_limit), t=6, w="0.8",
                                        f=4, mm="0.2")
    # every one of the four parts has a face that is tight; only two of them show it, the
    # other two show the trouble that is worse than a close neighbour
    assert [tuple(painted(win, e, c) for c in ("tight", "thin", "overhang"))
            for e in (thin, thick, under, over)] == [
        (0, 30, 6),       # the thin wall: magenta everywhere but on its red bottom
        (6, 0, 6),        # the thick wall: its face against the thin one is cyan
        (6, 0, 6),        # the plate below: its top is cyan
        (0, 0, 6),        # the plate above: that same face is red, tight though it is
    ]


def test_a_part_that_is_no_volume_is_skipped(win):
    """A loose face 0.1 mm under a block is not a part: there is no solid to keep room
    for, and the block above it says nothing."""
    ents = Entities()
    rect(ents, 0, 0, 10, 10)
    add(win, ents, "face only")
    near = add(win, box(0, 0, 0.1, 10, 10, 10.1), "block").ents
    win.a_tight_fits.trigger()
    checked(win, near)
    assert "Tight fit" not in win.hint.text()
    assert not tinted(win, near, "tight").any()


def test_a_deleted_part_leaves_no_answer_behind(win):
    """The answer belongs to the model it was made of; when a part leaves the document its
    answer goes with it, and what is left asks for a check of its own - it may well be
    free now that its neighbour is gone."""
    left = add(win, box(0, 0, 0, 10, 10, 10), "left")
    right = add(win, box(10.1, 0, 0, 20.1, 10, 10), "right")
    win.a_tight_fits.trigger()
    checked(win, left.ents, right.ents)
    assert win.hint.text() == hint(2, "0.2")
    gid = left.id
    win.doc.remove_group(gid)
    win.doc.notify()
    win.view._prune_rcache()
    assert id(left.ents) not in win.view._tight_landed
    assert win.view._tight_answer is None, "an answer of a model that no longer exists"
    assert win.view._tight_mask_of(right.ents) is None, "what is left asks for its own check"
    assert "Tight fit" not in win.hint.text()
    wait_for(win, right.ents)
    assert not tinted(win, right.ents, "tight").any(), "nothing is left to be close to"
