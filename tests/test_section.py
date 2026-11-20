"""The section view: cut the 3D view open with a plane you can move.

`section.plane`, `keeps`, `ray_span`, `keep_span`, `cut_of` and `cap_triangles` are the rule
on its own; the rest is View > Section in the window, in its panel and in the picture the
view makes of it. Those run without a display (Qt offscreen platform) and with the stand-in
for the GL driver of tests/test_render_fast.py, like tests/test_thin_walls.py; the helpers
`rect`, `face_at` and `box` and the `pump_until` that waits for a job are from there too.
"""
import math
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

from volura import ops, section, theme, viewport  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, point_in_polys  # noqa: E402
from volura.i18n import tr, trn  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.viewport import WIREFRAME  # noqa: E402
from test_render_fast import start  # noqa: E402
from test_thin_walls import pump_until  # noqa: E402

#: what these tests change in the settings that Volura keeps between runs
SETTING_KEYS = ("face_style",)


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


def cube(side=20.0, at=(0.0, 0.0, 0.0)):
    """A solid cube of `side` mm standing on the corner `at`."""
    return mf.Manifold.cube([side, side, side]).translate(at)


def tray(outer=20.0, wall=2.0, height=20.0, floor=2.0):
    """A box without a top, open upwards: a floor and four walls of `wall` mm."""
    inside = outer - 2 * wall
    deep = height - floor
    return (cube(outer, (0.0, 0.0, 0.0))
            - mf.Manifold.cube([inside, inside, deep]).translate([wall, wall, floor]))


def area(triangles):
    """The area of the cut surface the triangles cover."""
    t = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    if not len(t):
        return 0.0
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


# ---------------------------------------------------------------- the plane

@pytest.mark.parametrize("axis", section.AXES)
def test_the_plane_keeps_the_half_below_it(axis):
    eq = section.plane(axis, 5.0)
    below, above = np.zeros(3), np.zeros(3)
    below[section.AXES.index(axis)], above[section.AXES.index(axis)] = 4.0, 6.0
    assert section.keeps(eq, below)[0], "the half below the plane is what stays"
    assert not section.keeps(eq, above)[0], "the half above the plane is cut away"


@pytest.mark.parametrize("axis", section.AXES)
def test_flip_keeps_the_other_half(axis):
    k = section.AXES.index(axis)
    eq = section.plane(axis, 5.0, flip=True)
    below, above = np.zeros(3), np.zeros(3)
    below[k], above[k] = 4.0, 6.0
    assert not section.keeps(eq, below)[0]
    assert section.keeps(eq, above)[0]
    assert eq == tuple(-v for v in section.plane(axis, 5.0))


def test_the_plane_on_the_origin_keeps_a_zero_equation():
    assert section.plane("z", 0.0) == (0.0, 0.0, -1.0, 0.0)
    assert section.plane("x", 0.0) == (-1.0, 0.0, 0.0, 0.0)
    assert section.plane("y", 0.0) == (0.0, -1.0, 0.0, 0.0)


def test_the_plane_of_an_unknown_axis_is_an_error():
    for bad in ("q", "", None, 0):
        with pytest.raises(ValueError):
            section.plane(bad, 1.0)
        with pytest.raises(ValueError):
            section.cap_triangles(cube(), bad, 1.0)


def test_keeps_holds_every_axis_and_a_point_on_the_plane():
    eq = section.plane("z", 5.0)
    points = [[0, 0, 4.9], [10, 2, 5.0], [-3, 0, 5.1]]
    assert section.keeps(eq, points).tolist() == [True, True, False]
    eq = section.plane("x", -2.0, flip=True)
    assert section.keeps(eq, points).tolist() == [True, True, False]


def test_keeps_of_nothing_and_of_one_point():
    eq = section.plane("z", 5.0)
    assert section.keeps(eq, []).tolist() == []
    assert section.keeps(eq, [0.0, 0.0, 4.0]).tolist() == [True]
    assert len(section.keeps(eq, np.zeros((7, 3)))) == 7


# ---------------------------------------------------------------- the position

def test_the_panel_range_follows_the_model_of_that_axis():
    box_ = np.array([[10.0, -5.0, 2.0], [30.0, 5.0, 8.0]])
    assert section.span(box_, "x") == (10.0, 30.0)
    assert section.span(box_, "y") == (-5.0, 5.0)
    assert section.span(box_, "z") == (2.0, 8.0)
    assert section.span(None, "z") is None
    assert section.span(box_, "q") is None


@pytest.mark.parametrize("axis,want", (("x", 20.0), ("y", 0.0), ("z", 5.0)))
def test_no_position_is_the_middle_of_the_model(axis, want):
    box_ = np.array([[10.0, -5.0, 2.0], [30.0, 5.0, 8.0]])
    assert section.view_plane(axis, None, False, box_) == section.plane(axis, want)


@pytest.mark.parametrize("pos,want", ((None, 5.0), (-100.0, 2.0), (99.0, 8.0), (4.5, 4.5)))
def test_a_position_beside_the_model_is_moved_to_its_nearest_edge(pos, want):
    box_ = np.array([[0.0, 0.0, 2.0], [10.0, 10.0, 8.0]])
    eq = section.view_plane("z", pos, False, box_)
    assert eq == section.plane("z", want)


@pytest.mark.parametrize("pos", (float("nan"), float("inf")))
def test_a_position_that_is_no_number_is_the_middle(pos):
    box_ = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 20.0]])
    assert section.view_plane("z", pos, False, box_) == section.plane("z", 10.0)


def test_without_a_model_there_is_nothing_to_cut():
    assert section.view_plane("z", None, False, None) is None


@pytest.mark.parametrize("axis", ("q", "", None))
def test_an_axis_that_does_not_exist_cuts_nothing(axis):
    box_ = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 20.0]])
    assert section.view_plane(axis, None, False, box_) is None


# ---------------------------------------------------------------- the settings of the view

class Settings:
    """A settings object with only the section view in it, as :func:`section.cut_of` reads it."""

    def __init__(self, **kw):
        values = dict(section_on=True, section_axis="z", section_pos=None, section_flip=False)
        values.update(kw)
        self.__dict__.update(values)


MODEL = np.array([[0.0, 0.0, 0.0], [40.0, 20.0, 20.0]])


def test_the_view_cuts_only_when_the_section_view_is_on():
    assert section.cut_of(Settings(section_on=False), MODEL) is None
    assert section.cut_of(None, MODEL) is None
    assert section.cut_of(object(), MODEL) is None      # settings without a section in them


def test_the_view_cuts_nothing_when_there_is_no_model():
    assert section.cut_of(Settings(), None) is None
    assert section.cut_of(Settings(section_axis="q"), MODEL) is None


def test_the_view_cuts_the_middle_of_the_model_along_z():
    assert section.cut_of(Settings(), MODEL) == ("z", 10.0, False, section.plane("z", 10.0))


@pytest.mark.parametrize("axis,pos", (("x", 20.0), ("y", 10.0), ("z", 10.0)))
def test_the_view_cuts_the_middle_of_whatever_axis_it_stands_on(axis, pos):
    assert section.cut_of(Settings(section_axis=axis), MODEL)[:3] == (axis, pos, False)


@pytest.mark.parametrize("pos,want", ((6.0, 6.0), (-10.0, 0.0), (99.0, 20.0), (None, 10.0)))
def test_the_view_cuts_where_the_panel_puts_the_plane(pos, want):
    assert section.cut_of(Settings(section_pos=pos), MODEL)[:3] == ("z", want, False)


def test_a_flipped_view_cuts_the_other_half():
    axis, pos, flip, eq = section.cut_of(Settings(section_pos=6.0, section_flip=True), MODEL)
    assert (axis, pos, flip) == ("z", 6.0, True)
    assert eq == section.plane("z", 6.0, flip=True)


# ------------------------------------------------- what of a ray or an edge the plane leaves

def test_a_ray_that_starts_in_what_stays_ends_where_it_runs_into_the_cut():
    eq = section.plane("z", 5.0)                 # the half below 5 is what stays
    assert section.ray_span(eq, (0, 0, 0), (0, 0, 1)) == (0.0, 5.0)
    assert section.ray_span(eq, (0, 0, 2), (0, 0, 1)) == (0.0, 3.0)
    assert section.ray_span(eq, (0, 0, 4), (0, 0, 2)) == (0.0, 0.5)


def test_a_ray_that_starts_in_the_cut_sees_only_what_lies_beyond_the_plane():
    eq = section.plane("z", 5.0)
    assert section.ray_span(eq, (0, 0, 10), (0, 0, -1)) == (5.0, math.inf)
    assert section.ray_span(eq, (0, 0, 10), (0, 0, -2)) == (2.5, math.inf)
    eq = section.plane("z", 5.0, flip=True)      # now the half above 5 stays
    assert section.ray_span(eq, (0, 0, 0), (0, 0, 1)) == (5.0, math.inf)


def test_a_ray_that_runs_along_the_plane_keeps_everything_it_touches():
    eq = section.plane("z", 5.0)
    assert section.ray_span(eq, (0, 0, 2), (1, 0, 0)) == (0.0, math.inf)
    assert section.ray_span(eq, (0, 0, 5), (1, 1, 0)) == (0.0, math.inf)
    assert section.ray_span(eq, (0, 0, 5), (1, 0, -1)) == (0.0, math.inf)
    assert section.ray_span(eq, (0, 0, 5), (1, 0, 1)) == (0.0, 0.0)


def test_a_ray_that_hangs_entirely_in_the_cut_hits_nothing():
    eq = section.plane("z", 5.0)
    assert section.ray_span(eq, (0, 0, 10), (1, 0, 0)) is None
    assert section.ray_span(eq, (0, 0, 10), (1, 0, 1)) is None
    assert section.ray_span(eq, (0, 0, 10), (0, 0, 1)) is None


def test_the_span_of_a_ray_and_the_points_on_it_agree():
    """What a ray keeps is what `keeps` says about the points on it: one rule, not two."""
    eq = section.plane("y", 3.0, flip=True)
    for o in ((-4, 0, 2), (2, 8, -1), (1, 3, 0)):
        for raw in ((1, 0.25, 0.5), (0, -1, 0), (0.1, 1, -0.3), (0, 0, -1)):
            d = np.asarray(raw, dtype=float)
            d = d / np.linalg.norm(d)
            span = section.ray_span(eq, o, d)
            for t in np.linspace(0.0, 12.0, 25):
                p = np.asarray(o, dtype=float) + d * t
                at = section.keeps(eq, p)[0]
                inside = span is not None and span[0] - 1e-9 <= t <= span[1] + 1e-9
                assert at == inside, f"{o} + {d} t = {t} is kept: {at}, span says {span}"


def test_the_span_of_an_edge_that_crosses_the_plane_is_the_part_that_stays():
    eq = section.plane("z", 5.0)
    lo, hi = section.keep_span(eq, [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
                               [[0, 0, 10], [0, 0, 4], [0, 0, 2]])
    assert lo.tolist() == [0.0, 0.0, 0.0]
    assert hi[0] == pytest.approx(0.5) and hi[1] == 1.0 and hi[2] == 1.0


def test_the_span_of_an_edge_in_the_cut_is_empty():
    eq = section.plane("x", 0.0, flip=True)          # everything with x < 0 is cut away
    lo, hi = section.keep_span(eq, [[-1, 0, 0], [5, 0, 0]], [[-2, 3, 0], [6, 0, 1]])
    assert lo[0] > hi[0], "an edge that hangs entirely in the cut keeps nothing"
    assert lo[1] == 0.0 and hi[1] == 1.0


def test_an_edge_standing_on_the_plane_is_not_cut_away_itself():
    eq = section.plane("y", 2.0)
    lo, hi = section.keep_span(eq, [[0, 2, 0], [0, 8, 0]], [[1, 2, 0], [1, 2, 0]])
    assert lo.tolist() == [0.0, 1.0] and hi.tolist() == [1.0, 1.0]
    lo, hi = section.keep_span(eq, [[0, 2, 0], [0, 0, 0]], [[1, 2, 0], [1, 4, 0]])
    assert lo[0] == 0.0 and hi[0] == 1.0, "an edge on the plane is kept, cut open and all"
    assert hi[1] == pytest.approx(0.5)


def test_the_span_of_an_edge_and_the_points_on_it_agree():
    """What of an edge stays is what `keeps` says about its points: one rule, not two."""
    eq = section.plane("z", 5.0)
    a = np.array([[0, 0, -3], [0, 0, 6], [0, 0, 5], [0, 0, 7], [0, 0, 1]], dtype=float)
    b = np.array([[0, 0, 9], [0, 0, 4], [0, 0, 5], [0, 0, 12], [0, 0, 9]], dtype=float)
    lo, hi = section.keep_span(eq, a, b)
    u = np.linspace(0.0, 1.0, 21)
    for k in range(len(a)):
        for f in u:
            p = a[k] + (b[k] - a[k]) * f
            at = section.keeps(eq, p)[0]
            assert at == (lo[k] - 1e-9 <= f <= hi[k] + 1e-9), f"edge {k} at u = {f}"


def test_the_span_of_no_edges_is_nothing_at_all():
    lo, hi = section.keep_span(section.plane("z", 5.0), np.zeros((0, 3)), np.zeros((0, 3)))
    assert len(lo) == 0 and len(hi) == 0


# ---------------------------------------------------------------- the caps

def test_a_cube_cut_halfway_is_a_square_of_its_own_section():
    at = section.cap_triangles(cube(20.0, (0.0, 0.0, 0.0)), "z", 5.0)
    assert area(at) == pytest.approx(400.0)
    assert np.allclose(at[:, :, 2], 5.0)
    assert at.dtype == np.float32 and at.shape[1:] == (3, 3)


@pytest.mark.parametrize("axis", section.AXES)
def test_a_cube_cut_along_any_axis_is_a_square_on_that_axis(axis):
    at = section.cap_triangles(cube(20.0, (0.0, 0.0, 0.0)), axis, 5.0)
    assert area(at) == pytest.approx(400.0)
    k = section.AXES.index(axis)
    every = at.reshape(-1, 3)[:, k]
    assert np.allclose(every, 5.0), f"the cap does not lie in the plane at {axis} = 5"


def test_a_hollow_box_cuts_to_a_ring():
    """The cap of an open box is the wall that was cut: outside minus inside, hole and all."""
    at = section.cap_triangles(tray(), "z", 10.0)
    assert area(at) == pytest.approx(20 * 20 - 16 * 16)
    assert np.allclose(at.reshape(-1, 3)[:, 2], 10.0)
    inside = point_in_polys([t[:, :2] for t in at.reshape(-1, 3, 3)], (10.0, 10.0))
    assert not inside, "the middle of the box is empty, so no cap covers it"


def test_a_cut_that_lands_in_the_floor_of_a_hollow_box_is_a_ring_too():
    """The plane stands on the inside floor of the tray: the wall of 2 mm is all there is."""
    at = section.cap_triangles(tray(), "z", 2.0)
    assert area(at) == pytest.approx(20 * 20 - 16 * 16, rel=0.02)


def test_a_cut_in_the_bottom_or_the_top_face_does_not_raise():
    part = cube(20.0, (0.0, 0.0, 0.0))
    assert area(section.cap_triangles(part, "z", 0.0)) == pytest.approx(400.0, rel=0.02)
    assert area(section.cap_triangles(part, "x", 20.0)) == pytest.approx(400.0, rel=0.02)


@pytest.mark.parametrize("axis", section.AXES)
@pytest.mark.parametrize("pos", (-1.0, 21.0, -1000.0, 1e6))
def test_a_cut_beside_the_part_is_no_cap(axis, pos):
    at = section.cap_triangles(cube(20.0, (0.0, 0.0, 0.0)), axis, pos)
    assert at.shape == (0, 3, 3) and at.dtype == np.float32
    assert area(at) == 0.0


def test_a_cut_in_the_gap_between_two_parts_is_no_cap():
    """Two cubes with 4 mm between them: the plane at z = 22 crosses neither."""
    part = cube(20.0, (0.0, 0.0, 0.0)) + cube(20.0, (0.0, 0.0, 26.0))
    assert section.cap_triangles(part, "z", 22.0).shape == (0, 3, 3)
    assert area(section.cap_triangles(part, "z", 28.0)) == pytest.approx(400.0)


@pytest.mark.parametrize("part", (None, mf.Manifold()))
def test_no_part_is_no_cap(part):
    assert section.cap_triangles(part, "z", 5.0).shape == (0, 3, 3)


def test_a_part_that_is_only_a_face_is_no_cap():
    """A loose face is no volume: manifold3d has no inside to cut open."""
    ents = Entities()
    rect(ents, 0, 0, 10, 10)
    assert section.cap_triangles(ents.to_manifold(), "z", 5.0).shape == (0, 3, 3)


def test_a_solid_of_a_thousand_parts_is_cut_in_one_go():
    """The caps of a busy model come from one solid, however many pieces it holds."""
    part = cube(4.0, (0.0, 0.0, 0.0))
    for i in range(1, 12):
        part = part + cube(4.0, (6.0 * i, 0.0, 0.0))
    assert area(section.cap_triangles(part, "z", 2.0)) == pytest.approx(12 * 16.0)


def test_the_cap_of_a_part_that_is_not_a_closed_volume_is_something_sensible():
    """An open shell gives whatever manifold3d makes of it; it may not crash or hang."""
    ents = box(0, 0, 0, 10, 10, 10)
    for f in list(ents.F):           # take the top away: the box is a bowl now
        if abs(ents.F[f].normal[2] - 1.0) < 1e-9 and ents.face_plane(f).distance(P(5, 5, 10)) < 1e-6:
            ents.erase_faces([f])
    at = section.cap_triangles(ents.to_manifold(), "z", 5.0)
    assert at.shape[1:] == (3, 3)
    assert area(at) >= 0.0


# ---------------------------------------------------------------- the window

@pytest.fixture()
def win():
    """A window of its own, with a view that is still closed: every test opens it itself.

    The face style is kept in the settings, so a test that draws a wireframe gives the old one
    back; without that the next window would open in wireframe and draw no cut surfaces."""
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    w.view.cam.width, w.view.cam.height = 800, 600
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)


def block(win, x1=20.0, y1=20.0, z1=40.0):
    """A closed block of `x1` x `y1` x `z1` mm standing on the origin, drawn in the model."""
    ents = win.doc.root
    rect(ents, 0.0, 0.0, x1, y1)
    f = face_at(ents, x1 / 2, y1 / 2, 0.0)
    ents.pushpull(f, z1 if ents.F[f].normal[2] > 0 else -z1)
    assert np.allclose(ents.bbox(), [[0.0, 0.0, 0.0], [x1, y1, z1]])
    return ents


def part(win, solid, name="part", color="#cc3333"):
    """A solid part of the model, as a group: the plane cuts it open like the view shows it."""
    return ops.group_from_manifold(win.doc, solid, color, name)


def opened(win, axis="z", pos=None, flip=False):
    """View > Section on, with the plane of these settings, as the menu and panel leave it.

    A test that puts the settings in directly has to tell the panel and the bar as well, just
    like the panel does itself after every change (see :meth:`SectionPanel.apply`)."""
    win.settings.section_axis = axis
    win.settings.section_pos = pos
    win.settings.section_flip = flip
    if not win.a_section.isChecked():
        win.a_section.trigger()
    win.section_panel.refresh()
    win.view.update_view_status()
    return win.view.engine.section_cut()


def closed(win):
    """View > Section off again, the way the menu does it."""
    if win.a_section.isChecked():
        win.a_section.trigger()


def picture(win, monkeypatch, caps=False):
    """One model pass with the stand-in for the GL driver installed.

    Returns ``(the stand-in, the draws of that pass)``. With `caps` the view first waits for
    the solids of the parts: the pass that opens the view has no cut surfaces yet, because
    they are made in the background (see :meth:`Viewport._cap_solid_of`)."""
    fake = start(monkeypatch, win)
    win.view._draw_model()
    if not caps:
        return fake, list(fake.draws)
    assert pump_until(lambda: not win.view._cap_busy), "the solid of a part is never made"
    first = len(fake.draws)
    win.view._draw_model()
    return fake, fake.draws[first:]


def tri(draws):
    """The draws of faces among the draws of one pass."""
    return [d for d in draws if d["mode"] == viewport.gl.GL_TRIANGLES]


def cut_of_the_frame(draws):
    """The draws of the pass that are not cut by the plane: the cut surfaces and nothing else."""
    return [d for d in tri(draws) if not clipped(d)]


def clipped(draw):
    """Whether this draw of the model was made with the section plane switched on."""
    return bool(draw["caps"].get(viewport.gl.GL_CLIP_PLANE0))


def hint_of(win):
    win.view.update_view_status()
    return win.hint.text()


# ---------------------------------------------------------------- the action and the panel

def test_the_view_starts_closed_and_the_model_is_whole(win):
    """Nobody wants a model that is cut in half at start-up, and no file holds a section."""
    assert win.settings.section_on is False
    assert win.a_section.isChecked() is False
    assert win.view.engine.section_cut() is None


def test_the_menu_switches_the_view_and_opens_the_panel_with_it(win):
    win.a_section.trigger()
    assert win.settings.section_on is True
    assert win.tray.is_open(tr("Section")), "the plane is what to move, so the panel opens"
    win.a_section.trigger()
    assert win.settings.section_on is False
    assert win.view.engine.section_cut() is None


def test_the_panel_is_offered_wherever_the_three_dimensional_view_is():
    from volura.app import MainWindow
    from volura.window_ui import PANEL_SECTION
    for (workspace, mode), panels in MainWindow.WORKSPACE_PANELS.items():
        assert (PANEL_SECTION in panels) == (mode == "3d"), f"{workspace} {mode}"


def test_the_panel_shows_the_plane_in_the_middle_of_the_model(win):
    block(win, 20.0, 20.0, 40.0)
    opened(win)
    assert win.settings.section_pos is None, "no position of its own is the middle"
    assert win.view.engine.section_cut()[1] == pytest.approx(20.0)
    assert win.section_panel.pos.value() == pytest.approx(20.0)
    assert win.section_panel.slider.value() == pytest.approx(win.section_panel.STEPS / 2)


def test_the_panel_runs_over_the_model_of_the_axis_it_chose(win):
    block(win, 20.0, 30.0, 40.0)
    opened(win)
    assert (win.section_panel.pos.minimum(), win.section_panel.pos.maximum()) == (0.0, 40.0)
    win.section_panel.axis.setCurrentIndex(section.AXES.index("y"))
    assert (win.section_panel.pos.minimum(), win.section_panel.pos.maximum()) == (0.0, 30.0)


def test_sliding_the_plane_moves_it_and_the_field_follows(win):
    block(win, 20.0, 20.0, 40.0)
    opened(win)
    win.section_panel.slider.setValue(int(win.section_panel.STEPS * 0.25))
    assert win.settings.section_pos == pytest.approx(10.0)
    assert win.section_panel.pos.value() == pytest.approx(10.0)
    assert win.view.engine.section_cut()[1] == pytest.approx(10.0)


def test_typing_a_position_moves_the_plane_and_the_slider_follows(win):
    block(win, 20.0, 20.0, 40.0)
    opened(win)
    win.section_panel.pos.setValue(34.5)
    assert win.settings.section_pos == pytest.approx(34.5)
    assert win.section_panel.slider.value() == pytest.approx(win.section_panel.STEPS * 34.5 / 40,
                                                            abs=1)
    assert win.view.engine.section_cut()[1] == pytest.approx(34.5)


def test_a_position_beside_the_model_is_moved_back_on_to_it(win):
    """The field cannot go past the model, but the settings can: a model that shrank left the
    plane standing in the air, and the view has to cut something sensible then."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=900.0)
    assert win.view.engine.section_cut()[1] == pytest.approx(40.0)
    assert win.section_panel.pos.value() == pytest.approx(40.0)
    opened(win, pos=-900.0)
    assert win.view.engine.section_cut()[1] == pytest.approx(0.0)
    assert win.section_panel.pos.value() == pytest.approx(0.0)


def test_the_panel_moves_the_plane_back_to_the_middle(win):
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=34.0)
    win.section_panel.centre.click()
    assert win.settings.section_pos is None
    assert win.view.engine.section_cut()[1] == pytest.approx(20.0)
    assert win.section_panel.slider.value() == pytest.approx(win.section_panel.STEPS / 2)


def test_another_axis_moves_the_plane_back_to_the_middle_of_that_one(win):
    """A position along z says nothing about x, so the plane starts again in the middle."""
    block(win, 40.0, 20.0, 40.0)
    opened(win, axis="z", pos=38.0)
    win.section_panel.axis.setCurrentIndex(section.AXES.index("x"))
    assert win.settings.section_pos is None
    assert win.view.engine.section_cut()[0] == "x"
    assert win.view.engine.section_cut()[1] == pytest.approx(20.0)


def test_the_flip_of_the_panel_keeps_the_other_half(win):
    """The box of the panel is clicked, not called: what the view cuts has to follow the click.
    """
    block(win, 20.0, 20.0, 40.0)
    cut = opened(win, pos=10.0)
    assert section.keeps(cut[3], P(0, 0, 5.0))[0]
    assert not section.keeps(cut[3], P(0, 0, 15.0))[0]
    win.section_panel.flip.click()
    assert win.settings.section_flip is True
    cut = win.view.engine.section_cut()
    assert not section.keeps(cut[3], P(0, 0, 5.0))[0]
    assert section.keeps(cut[3], P(0, 0, 15.0))[0]
    assert win.section_panel.flip.isChecked() is True
    win.section_panel.flip.click()
    assert win.settings.section_flip is False
    assert section.keeps(win.view.engine.section_cut()[3], P(0, 0, 5.0))[0]


def test_the_panel_says_the_view_is_closed_while_it_is(win):
    """The panel is always in the tray, so it has to explain why it does nothing."""
    block(win)
    assert win.section_panel.off.isHidden() is False
    assert win.section_panel.pos.isEnabled() is False
    assert win.section_panel.slider.isEnabled() is False
    assert win.section_panel.centre.isEnabled() is False
    opened(win)
    assert win.section_panel.off.isHidden() is True
    assert win.section_panel.pos.isEnabled() is True


def test_an_empty_model_has_no_plane_to_move(win):
    """Without a model there is nothing to cut and no range to slide over: no plane at all,
    and the panel keeps its words instead of a field of nothing."""
    assert win.doc.model_bbox() is None
    assert opened(win) is None, "an empty model is not cut"
    win.section_panel.set_from_slider(win.section_panel.STEPS)
    win.section_panel.go_middle()
    assert win.settings.section_pos is None
    assert win.section_panel.slider.isEnabled() is False
    assert win.section_panel.pos.isEnabled() is False


def test_a_flat_model_slides_over_a_thousand_places_instead_of_none(win):
    """A sheet of 0.1 mm is thin but not nothing: the plane has to be able to move over it."""
    block(win, 20.0, 20.0, 0.1)
    opened(win)
    assert win.section_panel.slider.isEnabled() is True
    win.section_panel.slider.setValue(int(win.section_panel.STEPS * 0.75))
    assert win.view.engine.section_cut()[1] == pytest.approx(0.075, abs=1e-3)


def test_the_bar_says_where_the_plane_stands(win):
    block(win, 20.0, 20.0, 40.0)
    assert "Section" not in hint_of(win)
    opened(win, axis="z", pos=12.0)
    assert hint_of(win) == tr("Section: {axis} at {mm} mm").format(axis="z", mm="12")
    opened(win, axis="x", pos=12.5)
    assert hint_of(win) == tr("Section: {axis} at {mm} mm").format(axis="x", mm="12.5")
    closed(win)
    assert "Section" not in hint_of(win), "the bar does not keep a plane that is gone"


def test_the_bar_keeps_the_other_hints_next_to_it(win, monkeypatch):
    """The section is the last of the hints of the view; it does not push the overhang count
    away, and when it goes that count simply stands alone again."""
    block(win, 20.0, 20.0, 40.0)
    monkeypatch.setattr(win.view, "_overhang_faces", lambda: (0.0, 3))
    over = trn("Overhangs: {n} face needs support (over {limit}°)",
               "Overhangs: {n} faces need support (over {limit}°)", 3).format(
        n=3, limit=int(win.settings.overhang_limit))
    opened(win, pos=12.0)
    assert hint_of(win) == over + " · " + tr("Section: {axis} at {mm} mm").format(
        axis="z", mm="12")
    closed(win)
    assert hint_of(win) == over


# ---------------------------------------------------------------- the picture

def test_the_plane_is_handed_to_the_driver_before_the_first_face(win, monkeypatch):
    """A clip plane is read once, when it is handed over; everything after that is cut with
    it, so the plane has to be there before the model starts and stand for every face."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    fake, draws = picture(win, monkeypatch)
    given = [c for c in fake.calls if c[0] == "glClipPlane"]
    assert given == [("glClipPlane", viewport.gl.GL_CLIP_PLANE0, (0.0, 0.0, -1.0, 20.0))], \
        "the plane is handed over more than once, or not at all"
    assert tri(draws), "the model was not drawn"
    assert all(clipped(d) for d in tri(draws)), "a face was drawn without the plane on"


def test_the_frame_gives_the_plane_back_and_nothing_else(win, monkeypatch):
    """The overlays and the next frame draw whole again: a plane or an array left switched on
    would eat the rest of the picture."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    fake, draws = picture(win, monkeypatch, caps=True)
    assert not fake.is_on(viewport.gl.GL_CLIP_PLANE0), "the plane is left on after the frame"
    left = [cap for cap in viewport._CAPS + viewport._ARRAYS if fake.is_on(cap)]
    assert not left, f"the frame left {left} switched on"
    assert cut_of_the_frame(draws), "without a cut surface there was nothing to test here"


def test_a_view_that_is_closed_draws_the_same_faces(win, monkeypatch):
    """Switching the view on adds the plane and the cut surface; the faces of the model are
    drawn exactly as often as before, so the section does not quietly draw anything twice."""
    block(win, 20.0, 20.0, 40.0)
    fake, draws = picture(win, monkeypatch)
    assert tri(draws), "nothing was drawn"
    assert not any(clipped(d) for d in tri(draws)), "the plane cuts while the view is closed"
    before = len(tri(draws))
    opened(win, pos=20.0)
    _, draws2 = picture(win, monkeypatch)
    assert len(tri(draws2)) == before, "the model itself is drawn another number of times"
    assert all(clipped(d) for d in tri(draws2))
    assert not cut_of_the_frame(draws2), "the first pass has no cut surface to draw yet"


def test_the_cut_surface_is_drawn_where_the_plane_cut_the_material(win, monkeypatch):
    """The cap is drawn lit, lifted towards the eye with the polygon offset, without the plane
    it stands in, and after the faces of the model - so it wins from the faces around it."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    fake, draws = picture(win, monkeypatch, caps=True)
    n = sum(len(t) for t in win.view._cap_cache.values())
    assert n == len(section.cap_triangles(win.doc.root.to_manifold(), "z", 20.0))
    cut = cut_of_the_frame(draws)
    assert len(cut) == 1, f"one part, one cut surface: {cut}"
    assert cut[0]["count"] == 3 * n, "the cut surface is drawn incomplete"
    assert tri(draws).index(cut[0]) == len(tri(draws)) - 1, "drawn before the model"
    for cap in ("GL_LIGHTING", "GL_COLOR_MATERIAL", "GL_POLYGON_OFFSET_FILL"):
        assert cut[0]["caps"].get(getattr(viewport.gl, cap)), f"{cap} is off: the cap is flat"
    for array in ("GL_VERTEX_ARRAY", "GL_NORMAL_ARRAY"):
        assert cut[0]["arrays"].get(getattr(viewport.gl, array)), f"{array} is off"
    assert not clipped(cut[0]), "a cap in the plane is clipped away with the rest"
    colour = ("glColor4f", *theme.rgb("section"), 1.0)
    assert colour in fake.calls, "the cut surface is not drawn in the colour of the section"
    clip = viewport.gl.GL_CLIP_PLANE0
    states = ["enable" if c == ("enable", clip) else "disable"
              for c in fake.calls if c in (("enable", clip), ("disable", clip))]
    assert states[-1] == "disable", "the frame leaves the plane switched on"
    assert all(a != b for a, b in zip(states, states[1:])), \
        f"the plane is switched without being used between it: {states}"
    assert states.count("enable") >= 2, "the plane is not switched back on after the cut surface"


def test_the_cut_surface_covers_no_more_than_the_wall_that_was_cut(win, monkeypatch):
    """A hollow part is drawn hollow: the cap is the ring of wall the plane cut, so looking
    into the box one sees the inside of its walls and not a closed bottom."""
    part(win, tray())
    opened(win, pos=10.0)
    fake, draws = picture(win, monkeypatch, caps=True)
    cut = cut_of_the_frame(draws)
    assert len(cut) == 1
    got = np.concatenate([t for t in win.view._cap_cache.values()])
    assert cut[0]["count"] == 3 * len(got)
    assert area(got) == pytest.approx(20 * 20 - 16 * 16)


def test_a_part_that_is_hidden_gets_no_cut_surface(win, monkeypatch):
    """A part that is not drawn may not leave its cut surface standing in the picture; the part
    that is still there simply keeps its own."""
    part(win, cube(20.0, (0.0, 0.0, 0.0)), name="near")
    far = part(win, cube(20.0, (60.0, 0.0, 0.0)), name="far")
    opened(win, pos=10.0)
    _, draws = picture(win, monkeypatch, caps=True)
    assert len(cut_of_the_frame(draws)) == 2, "both parts stand cut open"
    far.visible = False
    fake, draws = picture(win, monkeypatch, caps=True)
    assert len(cut_of_the_frame(draws)) == 1, "a hidden part closes itself open"


def test_a_plane_that_cannot_stand_there_cuts_nothing(win, monkeypatch):
    """An axis that does not exist is no plane: nothing is cut and nothing is drawn wrong, just
    as if the view were closed."""
    block(win, 20.0, 20.0, 40.0)
    assert opened(win, axis="q") is None
    fake, draws = picture(win, monkeypatch, caps=True)
    assert tri(draws) and not any(clipped(d) for d in tri(draws)), \
        "a plane that cannot stand still cuts"
    assert not win.view._cap_solids and not win.view._cap_cache, "cut surfaces made for no plane"


def test_a_part_that_is_no_volume_stands_there_without_a_cut_surface(win, monkeypatch):
    """A loose face is no material: manifold3d has no inside to cut open. The model is simply
    drawn cut, without a cap and without a word about it."""
    ents = win.doc.root
    rect(ents, 0.0, 0.0, 20.0, 20.0)
    opened(win, pos=10.0)
    fake, draws = picture(win, monkeypatch, caps=True)
    assert tri(draws), "the face itself is still drawn"
    assert all(clipped(d) for d in tri(draws))
    assert not cut_of_the_frame(draws)
    assert not [t for t in win.view._cap_cache.values() if len(t)]


def test_a_wireframe_does_not_even_ask_for_the_solid_of_a_part(win, monkeypatch):
    """Cut surfaces are not drawn in a wireframe, so making the solid of every part for them
    would be work nobody sees; the view waits until faces are drawn again."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    win.settings.face_style = WIREFRAME
    fake, draws = picture(win, monkeypatch, caps=True)
    assert not tri(draws)
    assert not win.view._cap_solids and not win.view._cap_busy


# ---------------------------------------------------------------- the waiting

def test_the_solid_of_a_part_is_made_while_the_bar_says_what_it_waits_for(win, monkeypatch):
    """The solid a cut surface comes from takes seconds on a big model and may not stop the
    view, so it is made in the background; until it is there the model is drawn cut open
    without cut surfaces, and the bar says what it is waiting for."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    fake = start(monkeypatch, win)
    win.view._draw_model()
    assert win.view._cap_busy, "the solid was made on the spot, on the GUI thread"
    assert win.hint.text() == tr("Preparing the section…"), "the bar is silent about the wait"
    assert not cut_of_the_frame(list(fake.draws)), "a cut surface was drawn from nothing yet"
    assert pump_until(lambda: not win.view._cap_busy), "the solid is never made"
    assert len(win.view._cap_solids) == 1
    assert tr("Section: {axis} at {mm} mm").format(axis="z", mm="20") in hint_of(win), \
        "the bar keeps the wait instead of going back to the plane"
    first = len(fake.draws)
    win.view._draw_model()
    assert cut_of_the_frame(fake.draws[first:]), "after the wait there is nothing to see"


def test_a_solid_of_geometry_that_moved_on_is_not_used(win, monkeypatch):
    """A solid belongs to one version of a part. The cut of a block that grew since would
    close it open in the wrong place, so a new solid is asked for."""
    ents = block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    picture(win, monkeypatch, caps=True)
    assert win.view._cap_solids[id(ents)][1] == ents.version
    ents.pushpull(face_at(ents, 10.0, 10.0, 40.0), 20.0)       # the block grows to 60 mm
    assert win.view._cap_solid_of(ents) is None, "an old solid answers new geometry"
    assert win.view._cap_busy[id(ents)] == (ents, ents.version)
    assert pump_until(lambda: win.view._cap_solids.get(id(ents), (0, -1))[1] == ents.version), \
        "the new solid never lands"
    assert win.view._cap_solid_of(ents) is not None


def test_the_cut_surface_goes_with_the_part_it_was_cut_from(win, monkeypatch):
    """A part that is gone may not leave a cut surface behind: its id can even come back with
    another part, and then the wrong model would be closed open."""
    g = part(win, cube(20.0))
    opened(win, pos=10.0)
    picture(win, monkeypatch, caps=True)
    assert win.view._cap_cache and win.view._cap_solids
    win.doc.remove_group(g.id)
    fake = start(monkeypatch, win)
    win.view._draw_model()
    assert not win.view._cap_cache and not win.view._cap_solids and not win.view._cap_busy


def test_the_cut_surface_of_a_part_behind_the_plane_is_no_cap(win, monkeypatch):
    """A part that hangs entirely in the cut-away half is gone from the view; the plane passes
    beside it, so there is nothing to cap - and the part that is cut still gets its cap."""
    part(win, cube(20.0, (0.0, 0.0, 0.0)), name="under")
    part(win, cube(20.0, (0.0, 0.0, 60.0)), name="behind")
    opened(win, pos=20.0)
    fake, draws = picture(win, monkeypatch, caps=True)
    assert len([t for t in win.view._cap_cache.values() if len(t)]) == 1
    assert len(cut_of_the_frame(draws)) == 1


def test_only_the_newest_place_of_the_plane_is_kept(win, monkeypatch):
    """A plane stands in one place at a time, so the cut of the place before it is worth
    nothing: sliding it through a big model may not fill the memory with old cuts."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    picture(win, monkeypatch, caps=True)
    assert len(win.view._cap_cache) == 1
    win.section_panel.pos.setValue(30.0)
    picture(win, monkeypatch, caps=True)
    keys = list(win.view._cap_cache)
    assert len(keys) == 1 and keys[0][2:] == ("z", 30.0), f"old cuts are kept: {keys}"


def test_the_cut_surfaces_go_when_the_view_closes(win, monkeypatch):
    """Closing the view gives the model back as it was: nothing of the cut is kept, so opening
    it again on another model cannot start from a leftover."""
    block(win, 20.0, 20.0, 40.0)
    opened(win, pos=20.0)
    picture(win, monkeypatch, caps=True)
    assert win.view._cap_cache and win.view._cap_solids
    closed(win)
    fake = start(monkeypatch, win)
    win.view._draw_model()
    assert not win.view._cap_cache and not win.view._cap_solids and not win.view._cap_busy
    assert not any(clipped(d) for d in tri(fake.draws)), \
        "the plane still cuts a view that is closed"
    assert not fake.is_on(viewport.gl.GL_CLIP_PLANE0), "a closed view leaves the plane switched on"


# ---------------------------------------------------------------- the picking

def screen(win, point):
    """Where the model point `point` lands on the screen of the view."""
    s, ok = win.view.cam.project(np.array([point], dtype=float))
    assert ok[0], f"{point} does not stand on the screen"
    return float(s[0][0]), float(s[0][1])


def face_under(win, point):
    """What a click on the model point `point` picks, as a face id (or None on nothing)."""
    o, d = win.view.cam.ray(*screen(win, point))
    hit = win.view.engine.pick_face(o, d)
    return None if hit is None else hit[2]


def edge_under(win, point):
    """The id of the edge a click near the model point `point` picks, or None."""
    hit = win.view.engine.pick_edge(*screen(win, point), px=6.0)
    return None if hit is None else hit[2]


def test_a_face_behind_the_plane_is_no_longer_there_to_pick(win):
    """What you cannot see you cannot click: looking down on a block that is cut halfway, the
    click goes through the open top to the bottom of it instead of stopping on its top face."""
    ents = block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("top")
    win.view.zoom_extents()
    top, bottom = face_at(ents, 20.0, 20.0, 40.0), face_at(ents, 20.0, 20.0, 0.0)
    assert face_under(win, (20.0, 20.0, 40.0)) == top
    opened(win, pos=20.0)
    assert face_under(win, (20.0, 20.0, 40.0)) == bottom, "the click stops on a face that is gone"
    closed(win)
    assert face_under(win, (20.0, 20.0, 40.0)) == top, "the model is not whole again after it"


def test_the_half_the_plane_keeps_is_the_half_that_picks(win):
    """Looking up at the same block from below, the flip decides whether the bottom face is
    there to pick or the click goes through to the top of the block."""
    ents = block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("bottom")
    win.view.zoom_extents()
    top, bottom = face_at(ents, 20.0, 20.0, 40.0), face_at(ents, 20.0, 20.0, 0.0)
    assert face_under(win, (20.0, 20.0, 0.0)) == bottom
    opened(win, pos=20.0)
    assert face_under(win, (20.0, 20.0, 0.0)) == bottom, "the half that stays was cut away too"
    opened(win, pos=20.0, flip=True)
    assert face_under(win, (20.0, 20.0, 0.0)) == top, "the flip did not reach the picking"


def test_a_ray_that_runs_along_the_cut_hits_nothing_at_all(win):
    """A ray that only travels through the half that was cut away has nothing left to hit: not
    even the face that closes the cut off, because that cap is no geometry to select."""
    block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("iso")
    win.view.zoom_extents()
    o, d = np.array([20.0, 20.0, 30.0]), np.array([1.0, 0.0, 0.0])
    assert win.view.engine.pick_face(o, d) is not None
    opened(win, pos=20.0)
    assert win.view.engine.pick_face(o, d) is None, "a wall behind the plane was picked through"


def test_a_plane_standing_on_the_model_cuts_nothing_from_the_picking(win):
    """The plane on the bottom face with the flip on keeps all of the model: every point that
    stands on the plane is kept, so the picking is simply what it always was."""
    block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("iso")
    win.view.zoom_extents()
    face, edge = face_under(win, (20.0, 20.0, 40.0)), edge_under(win, (40.0, 0.0, 30.0))
    opened(win, pos=0.0, flip=True)
    assert face_under(win, (20.0, 20.0, 40.0)) == face
    assert edge_under(win, (40.0, 0.0, 30.0)) == edge


def test_an_edge_hanging_in_the_cut_is_not_pickable(win):
    """An edge that was cut away with the face around it is gone from the picking too; the
    same place lower down, where the model is still there, still picks the edge that stands it."""
    block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("iso")
    win.view.zoom_extents()
    above = edge_under(win, (20.0, 0.0, 40.0))       # the middle of a top edge
    assert above is not None
    opened(win, pos=20.0)
    assert edge_under(win, (20.0, 0.0, 40.0)) is None, "an edge in the cut can still be picked"
    assert edge_under(win, (20.0, 0.0, 0.0)) is not None, "an edge that stands there is gone too"
    closed(win)
    assert edge_under(win, (20.0, 0.0, 40.0)) == above


def test_an_edge_that_crosses_the_plane_is_pickable_below_it(win):
    """A corner edge that goes through the plane is there below it and gone above it, so the
    picking follows the plane along the edge instead of all or nothing."""
    block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("iso")
    win.view.zoom_extents()
    corner = (40.0, 0.0)
    opened(win, pos=20.0)
    assert edge_under(win, corner + (34.0,)) is None, "the corner above the plane can be picked"
    hit = edge_under(win, corner + (6.0,))
    assert hit is not None, "the corner below the plane is gone too"
    closed(win)
    assert edge_under(win, corner + (34.0,)) == hit, "the edge is another one than the one below"


def test_the_snap_points_of_the_model_follow_the_plane_too(win):
    """The points a tool snaps to are the points of the geometry: behind the plane they are cut
    away with it, so a snap cannot drag a new edge through the cut to a point that is not there.
    """
    block(win, 40.0, 40.0, 40.0)
    win.view.cam.set_view("iso")
    win.view.zoom_extents()
    pts = [[20.0, 0.0, 34.0], [20.0, 0.0, 6.0]]
    assert win.view.engine.keep_mask(pts).tolist() == [True, True]
    opened(win, pos=20.0)
    assert win.view.engine.keep_mask(pts).tolist() == [False, True]
    assert win.view.engine.keep_mask([]).tolist() == [], "no points is not an error"


def test_the_picking_does_not_ask_for_the_model_box_again_every_time(win):
    """The plane stands on the model, so its position follows the model; on a busy model that
    box is not free, so it is only looked at again when the model has moved on."""
    ents = block(win, 40.0, 40.0, 40.0)
    opened(win, pos=20.0)
    first = win.view.engine.section_cut()
    assert win.view.engine.section_cut() is first, "the cut is made again for every click"
    ents.pushpull(face_at(ents, 20.0, 20.0, 40.0), 10.0)
    assert win.view.engine.section_cut() is not first, "a model that grew left the plane alone"
    assert win.view.engine.section_cut()[1] == pytest.approx(20.0)

