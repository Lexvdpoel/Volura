"""Which parts of a model stand in one another, and the red shapes the view draws there.

`collisions.collisions` is the rule on its own; the rest is View > Face Style > Collisions in
the window, which runs without a display (Qt offscreen platform), like tests/test_tight_fits.py.
The helpers `rect` and `face_at` are copied from tests/test_brep.py. The check covers the model
in the background and only starts after it has been left alone, so the tests wait for its answer
with `pump_until`, like the jobs of tests/test_simplify.py do.
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

from volura import collisions, ops, theme  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, point_in_polys  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.settings import qsettings  # noqa: E402

SETTING_KEYS = ("show_collisions", "show_tight_fits", "show_thin_walls", "show_overhangs",
                "min_wall", "overhang_limit", "screw_hole_clearance")
CHECKING = "Checking for collisions…"
NONE = "No collisions"
ONE = "{n} collision: {list}"
MANY = "{n} collisions: {list}"
LISTED = "{a} and {b} ({vol} cm³)"
NEAR_ONE = "(only the {n} part near the selection was checked)"
NEAR_MANY = "(only the {n} parts near the selection were checked)"


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


# ------------------------------------------------------------------ the rule

def test_two_parts_that_stand_in_one_answer_with_the_solid_they_share():
    """Two blocks of 10 mm with 5 mm of them in one another: the answer is one collision of
    500 mm³, and its mesh is the block of 5 × 10 × 10 mm that has to move."""
    a, b = solid(0, 0, 0, 10, 10, 10), solid(5, 0, 0, 15, 10, 10)
    hits = collisions.collisions([("a", a), ("b", b)])
    assert len(hits) == 1
    assert hits[0]["a"] == "a" and hits[0]["b"] == "b"
    assert hits[0]["volume"] == pytest.approx(500.0)
    verts, tris = hits[0]["mesh"]
    assert len(verts) and len(tris)
    assert np.max(tris) < len(verts)
    assert np.allclose(verts.min(axis=0), [5.0, 0.0, 0.0])
    assert np.allclose(verts.max(axis=0), [10.0, 10.0, 10.0])


def test_two_parts_that_only_touch_at_a_face_are_a_fit():
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10, 0, 0, 20, 10, 10)
    assert collisions.collisions([("a", a), ("b", b)]) == []


def test_parts_whose_boxes_are_apart_are_never_asked(monkeypatch):
    """No boolean, no mesh, nothing: a box that cannot reach another one answers with its box
    alone, which is what keeps a model of many parts that stand apart cheap to look at."""
    asked = []

    def count_boolean(a, b):
        asked.append((id(a), id(b)))
        return a ^ b

    monkeypatch.setattr(collisions, "_intersect", count_boolean)
    a, b = solid(0, 0, 0, 10, 10, 10), solid(20, 0, 0, 30, 10, 10)
    assert collisions.collisions([("a", a), ("b", b)]) == []
    assert asked == []


def test_boxes_that_touch_at_a_face_are_asked(monkeypatch):
    """The box test only throws away pairs that cannot overlap; a face-touch has to reach the
    boolean, because there the difference between a fit and a collision is made."""
    asked = []

    def count_boolean(a, b):
        asked.append(True)
        return a ^ b

    monkeypatch.setattr(collisions, "_intersect", count_boolean)
    a, b = solid(0, 0, 0, 10, 10, 10), solid(10, 0, 0, 20, 10, 10)
    assert collisions.collisions([("a", a), ("b", b)]) == []
    assert asked == [True]


def test_the_same_part_pasted_twice_is_a_collision_whole():
    a, b = solid(0, 0, 0, 10, 10, 10), solid(0, 0, 0, 10, 10, 10)
    hits = collisions.collisions([("copy", a), ("original", b)])
    assert len(hits) == 1
    assert hits[0]["a"] == "copy" and hits[0]["b"] == "original"
    assert hits[0]["volume"] == pytest.approx(1000.0)
    verts, tris = hits[0]["mesh"]
    assert np.allclose(verts.min(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(verts.max(axis=0), [10.0, 10.0, 10.0])
    assert len(tris) >= 1


def test_the_biggest_overlap_comes_first():
    whole = solid(0, 0, 0, 10, 10, 10)
    small = solid(1, 0, 0, 2, 10, 10)
    middle = solid(3, 0, 0, 5, 10, 10)
    large = solid(6, 0, 0, 9, 10, 10)
    hits = collisions.collisions([("small", small), ("middle", middle),
                                  ("large", large), ("whole", whole)])
    assert [(h["a"], h["b"]) for h in hits] == [("large", "whole"),
                                                ("middle", "whole"),
                                                ("small", "whole")]
    assert [h["volume"] for h in hits] == [pytest.approx(300.0), pytest.approx(200.0),
                                           pytest.approx(100.0)]


def test_equal_overlaps_keep_the_order_of_the_parts():
    whole = solid(0, 0, 0, 10, 10, 10)
    first = solid(1, 0, 0, 2, 10, 10)
    second = solid(3, 0, 0, 4, 10, 10)
    hits = collisions.collisions([("first", first), ("second", second), ("whole", whole)])
    assert [(h["a"], h["b"]) for h in hits] == [("first", "whole"), ("second", "whole")]


def test_a_floor_above_the_overlap_says_nothing():
    whole = solid(0, 0, 0, 10, 10, 10)
    other = solid(9.9, 0, 0, 10, 10, 10)
    assert collisions.collisions([("a", whole), ("b", other)], min_volume=20.0) == []
    assert len(collisions.collisions([("a", whole), ("b", other)], min_volume=0.0)) == 1


def test_a_sliver_the_boolean_leaves_behind_is_no_collision(monkeypatch):
    whole = solid(0, 0, 0, 10, 10, 10)
    other = solid(5, 0, 0, 15, 10, 10)
    monkeypatch.setattr(collisions, "_volume", lambda _man: 5e-4)
    assert collisions.collisions([("a", whole), ("b", other)]) == []
    assert len(collisions.collisions([("a", whole), ("b", other)], min_volume=0.0)) == 1
    assert collisions.collisions([("a", whole), ("b", other)], min_volume=None) == []
    assert collisions.collisions([("a", whole), ("b", other)], min_volume="not a floor") == []
    assert len(collisions.collisions([("a", whole), ("b", other)], min_volume=-1.0)) == 1


def test_no_parts_and_parts_that_are_no_solids_answer_nothing():
    assert collisions.collisions([]) == []
    assert collisions.collisions([("none", None), ("empty", mf.Manifold())]) == []
    only = solid(0, 0, 0, 10, 10, 10)
    assert collisions.collisions([("only", only)]) == []
    assert collisions.collisions([("none", None), ("only", only)]) == []


def test_a_boolean_the_kernel_refuses_is_answered_as_no_overlap(monkeypatch):
    monkeypatch.setattr(collisions, "_intersect", lambda _a, _b: None)
    assert collisions.collisions([("a", solid(0, 0, 0, 10, 10, 10)),
                                  ("b", solid(5, 0, 0, 15, 10, 10))]) == []


# ------------------------------------------------------------------ the window

@pytest.fixture()
def win():
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_collisions = False
    w.settings.show_tight_fits = False
    w.settings.show_thin_walls = False
    w.settings.show_overhangs = False
    w.settings.min_wall = 0.8
    w.settings.overhang_limit = 45
    w.settings.screw_hole_clearance = 0.2
    for a in (w.a_collisions, w.a_tight_fits, w.a_thin_walls, w.a_overhangs):
        a.setChecked(False)
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


def checked(win):
    """Wait until the collision check of the model has answered for the model as it is now."""
    assert pump_until(lambda: (win.view._collision_answer is not None
                               and win.view._collision_answer[0] == win.view._collision_key())), \
        "the collision check never came to an end"


# ------------------------------------------------------------------ the action

def test_the_action_is_a_checkable_view_style(win):
    """View > Face Style > Collisions is a switch like Tight Fits, and it is stored."""
    a = win.a_collisions
    assert a.isCheckable() and a.isEnabled()
    assert a.toolTip().startswith("Draw in red")
    assert "have in common" in a.toolTip()
    a.trigger()
    assert win.settings.show_collisions is True and a.isChecked()
    assert type(win.settings)().show_collisions is True
    a.trigger()
    assert win.settings.show_collisions is False
    assert type(win.settings)().show_collisions is False


# ------------------------------------------------------------------ the view

def test_two_parts_that_stand_in_one_are_counted_and_drawn_in_red(win):
    """Two blocks with 5 mm in one another: one red shape of 500 mm³, and the bar names the
    two parts and says their overlap in cm³."""
    left = add(win, box(0, 0, 0, 10, 10, 10), "left").ents
    right = add(win, box(5, 0, 0, 15, 10, 10), "right").ents
    win.a_collisions.trigger()
    checked(win)

    hits = win.view._collision_hits()
    assert len(hits) == 1
    assert {hits[0]["a"], hits[0]["b"]} == {id(left), id(right)}
    assert hits[0]["volume"] == pytest.approx(500.0)
    pts = hits[0]["pts"]
    assert pts.shape == (36, 3)
    assert np.allclose(pts.min(axis=0), [5.0, 0.0, 0.0])
    assert np.allclose(pts.max(axis=0), [10.0, 10.0, 10.0])
    r, g, b = theme.rgb("collision")
    assert r > max(g, b) + 0.2, "the overlap is red in both themes"

    listed = tr(LISTED).format(a="right", b="left", vol="0.50")
    expected = tr(ONE).format(n=1, list=listed)
    reverse = tr(LISTED).format(a="left", b="right", vol="0.50")
    assert win.hint.text() in (expected, tr(ONE).format(n=1, list=reverse))


def test_a_model_of_separated_parts_says_no_collisions(win):
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    add(win, box(11.0, 0, 0, 21.0, 10, 10), "right")
    win.a_collisions.trigger()
    checked(win)
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(NONE)


def test_two_parts_that_only_touch_are_a_fit(win):
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    add(win, box(10.0, 0, 0, 20.0, 10, 10), "right")
    win.a_collisions.trigger()
    checked(win)
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(NONE)


def test_the_bar_says_what_the_check_waits_for(win, monkeypatch):
    """A big model takes seconds; while the check runs the bar says so, and the red shapes
    and the count only arrive with the answer."""
    asked = []

    def wait_forever(key, parts, measured):
        asked.append((key, len(parts), measured))
        win.view.set_status(tr(CHECKING))

    monkeypatch.setattr(win.view, "_run_collision_job", wait_forever)
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    add(win, box(5, 0, 0, 15, 10, 10), "right")
    win.a_collisions.trigger()
    assert pump_until(lambda: asked), "the check never started"
    assert len(asked) == 1
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(CHECKING)


def test_a_running_check_is_not_asked_twice(win, monkeypatch):
    """A click in the model changes the selection without changing a part; the answer of the
    running check still applies to the same model and is not asked again."""
    asked = []
    monkeypatch.setattr(win.view, "_run_collision_job",
                        lambda key, parts, measured: asked.append(key))
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    right = add(win, box(5, 0, 0, 15, 10, 10), "right")
    win.a_collisions.trigger()
    assert pump_until(lambda: asked), "the check never started"
    assert len(asked) == 1
    win.doc.selection = [("g", right.id, right.id)]
    win.view.collisions_changed()
    win.view._check_collisions()
    assert len(asked) == 1, "the same model is not measured twice"


def test_an_edit_asks_for_a_new_check(win):
    """The right block moved clear of the left one: the answer of the old model may not be
    shown for the new one, and the bar then says there is no collision left."""
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    right = add(win, box(5, 0, 0, 15, 10, 10), "right")
    win.a_collisions.trigger()
    checked(win)
    assert len(win.view._collision_hits()) == 1

    # moved the way the Move tool moves a part
    win.doc.checkpoint()
    ops.transform_selection(win.doc, np.eye(3), np.array([10.0, 0.0, 0.0]),
                            items=[("g", right.id, right.id)])
    win.doc.notify()
    assert win.view._collision_answer[0] != win.view._collision_key()
    assert win.view._collision_hits() == []
    checked(win)
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(NONE)


def test_turning_the_view_off_clears_answer_and_status(win):
    add(win, box(0, 0, 0, 10, 10, 10), "left")
    add(win, box(5, 0, 0, 15, 10, 10), "right")
    win.a_collisions.trigger()
    checked(win)
    assert len(win.view._collision_hits()) == 1
    assert "collision" in win.hint.text().lower()

    win.a_collisions.trigger()
    assert win.view._collision_answer is None
    assert win.view._collision_hits() == []
    assert "collision" not in win.hint.text().lower()


def test_a_part_that_is_no_volume_is_not_a_collision(win):
    face_only = Entities()
    rect(face_only, 0, 0, 10, 10)
    add(win, face_only, "face only")
    add(win, box(0, 0, 0, 10, 10, 10), "block")
    win.a_collisions.trigger()
    checked(win)
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(NONE)


def test_a_hidden_part_is_not_measured(win):
    right = add(win, box(0, 0, 0, 10, 10, 10), "left")
    hidden = add(win, box(5, 0, 0, 15, 10, 10), "right")
    win.a_collisions.trigger()
    checked(win)
    assert len(win.view._collision_hits()) == 1

    hidden.visible = False
    win.doc.notify()
    assert win.view._collision_answer[0] != win.view._collision_key()
    checked(win)
    assert win.view._collision_hits() == []
    assert win.hint.text() == tr(NONE)
    assert right.visible is True


def test_a_big_model_is_measured_near_the_selection(win, monkeypatch):
    """Sixty parts plus the model's own geometry is more than the view wants to measure whole.
    With one part selected, only that part is measured, and the bar says how much of the model
    the answer is about; with nothing selected, the whole model is measured again."""
    asked = []
    monkeypatch.setattr(win.view, "_run_collision_job",
                        lambda key, parts, measured: asked.append((len(parts), measured)))
    groups = [add(win, box(100.0 * i, 0, 0, 100.0 * i + 1, 1, 1), f"part {i}")
              for i in range(60)]
    assert len(win.view._collision_parts()) > win.view.COLLISION_PARTS

    win.settings.show_collisions = True
    win.doc.selection = [("g", groups[0].id, groups[0].id)]
    win.view._check_collisions()
    assert asked == [(1, 1)]
    win.view._collision_answer = (win.view._collision_key(), [], 1)
    assert win.view._collision_hint() == f"{tr(NONE)} {tr(NEAR_ONE).format(n=1)}"
    win.view._collision_answer = (win.view._collision_key(), [], 2)
    assert win.view._collision_hint() == f"{tr(NONE)} {tr(NEAR_MANY).format(n=2)}"

    asked.clear()
    win.view._collision_busy = None
    win.doc.selection = []
    win.view._check_collisions()
    assert asked == [(len(win.view._collision_parts()), None)]
