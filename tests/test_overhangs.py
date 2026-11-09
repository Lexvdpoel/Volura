"""The overhang view: which faces need support when printed.

`ops.overhang_mask` is the rule on its own; the rest is View > Face Style > Overhangs
in the window, which runs without a display (Qt offscreen platform), like
tests/test_view_styles.py. The helpers `rect` and `face_at` are copied from
tests/test_brep.py.
"""
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import ops, theme  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.inference import GeoCache  # noqa: E402
from volura.settings import qsettings  # noqa: E402

SETTING_KEYS = ("show_overhangs", "overhang_limit")
HINT = "Overhangs: {n} faces need support (over {limit}°)"
HINT_ONE = "Overhangs: {n} face needs support (over {limit}°)"


def hint(n, limit):
    """The status text of the overhang view for `n` faces."""
    return (HINT_ONE if n == 1 else HINT).format(n=n, limit=limit)


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
    from volura.geom import point_in_polys
    for f in ents.F:
        pl = ents.face_plane(f)
        if abs(pl.distance(p)) > 1e-6:
            continue
        polys = [pl.to_local(L) for L in ents.face_points(f)]
        if point_in_polys(polys, pl.to_local(p)):
            return f
    raise AssertionError(f"no face at {p}")


def cube(x0, y0, z0, size):
    """A closed cube from (x0, y0, z0) to (x0 + size, y0 + size, z0 + size)."""
    ents = Entities()
    rect(ents, x0, y0, x0 + size, y0 + size, z=z0)
    f = face_at(ents, x0 + size / 2, y0 + size / 2, z0)
    ents.pushpull(f, size if ents.F[f].normal[2] > 0 else -size)
    assert np.allclose(ents.bbox(), [[x0, y0, z0], [x0 + size, y0 + size, z0 + size]])
    return ents


def underside(slope_deg):
    """The outward normal of the underside of a ledge sloping `slope_deg` from horizontal."""
    a = np.radians(slope_deg)
    return np.array([0.0, np.sin(a), -np.cos(a)])


@pytest.fixture()
def win():
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_overhangs = False
    w.settings.overhang_limit = 45
    w.a_overhangs.setChecked(False)
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


def tinted(win, ents):
    """The vertices the overhang view paints in the warning colour, as a bool per vertex."""
    win.view._overhang_ctx = win.view._overhang_context()   # what a frame would tint with
    rc = win.view._render_data(ents, None)
    warn = np.array(theme.rgb("overhang"), dtype=np.float32)
    return np.all(np.isclose(rc["front"][:, :3], warn), axis=1)


# ------------------------------------------------------------------ the rule

def test_vertical_wall_may_stay_as_it_is():
    """A vertical wall is 0° from the vertical: no support, however high it stands."""
    marked = ops.overhang_mask([[0, 1, 0], [1, 0, 0]], [10.0, 250.0], 0.0, 45)
    assert marked.tolist() == [False, False]


def test_face_pointing_straight_down():
    """The bottom of a part needs support, unless it lies on the print bed."""
    down = [0, 0, -1]
    assert ops.overhang_mask([down], [20.0], 0.0, 45).tolist() == [True]
    assert ops.overhang_mask([down], [0.0], 0.0, 45).tolist() == [False]
    assert ops.overhang_mask([down], [0.005], 0.0, 45).tolist() == [False]
    assert ops.overhang_mask([down], [0.02], 0.0, 45).tolist() == [True]


def test_face_pointing_up_is_never_marked():
    assert ops.overhang_mask([[0, 0, 1], [0.3, 0.3, 0.9]], [5.0, 5.0], 0.0, 45).tolist() == [False, False]


@pytest.mark.parametrize("slope, marked", ((30, True), (60, False), (45, False)))
def test_underside_of_a_ledge(slope, marked):
    """30° from horizontal hangs over 60° from the vertical and needs support; 60° does not.

    Exactly the limit is still allowed: a face is marked when it hangs over further."""
    assert ops.overhang_mask([underside(slope)], [10.0], 0.0, 45).tolist() == [marked]


def test_normals_may_be_too_short_or_too_long():
    """The rule normalises: what counts is the direction of the normal, not its length."""
    marked = ops.overhang_mask([[0, 0, -7], [0, 0, -0.001]], [10.0, 10.0], 0.0, 45)
    assert marked.tolist() == [True, True]


def test_no_faces_nothing_marked():
    assert ops.overhang_mask(np.zeros((0, 3)), np.zeros(0), 0.0, 45).tolist() == []
    assert ops.overhang_mask([], [], 0.0, 45).tolist() == []


def marked_faces(ents, bed_z, limit=45):
    """The faces of `ents` that need support, read off the triangles of the render cache."""
    c = GeoCache.build(ents)
    over = ops.overhang_mask(c["tri_nrm"], c["T"].mean(axis=1)[:, 2], bed_z, limit)
    return sorted(set(int(f) for f in c["tri_fid"][over]))


def test_a_cube_on_the_bed_needs_no_support():
    """Every face of a cube that stands on the bed: the walls are vertical, the bottom lies flat."""
    assert marked_faces(cube(100, 100, 0, 10), 0.0) == []


def test_a_floating_cube_marks_its_bottom():
    ents = cube(140, 100, 20, 10)
    assert len(marked_faces(ents, 0.0)) == 1
    assert marked_faces(ents, 20.0) == [], "on its own bed there is nothing to support"


# ------------------------------------------------------------------ the window

def test_the_action_is_a_checkable_view_style(win):
    """View > Face Style > Overhangs is a switch like X-ray, and it is stored."""
    a = win.a_overhangs
    assert a.isCheckable() and a.isEnabled()
    a.trigger()
    assert win.settings.show_overhangs is True and a.isChecked()
    assert type(win.settings)().show_overhangs is True
    a.trigger()
    assert win.settings.show_overhangs is False
    assert type(win.settings)().show_overhangs is False


def test_a_cube_on_the_bed_says_nothing(win):
    """Also not at the origin: the bed is the lowest z of the model, not z = 0 of a group."""
    add(win, cube(100, 100, 0, 10), "on the bed")
    win.a_overhangs.trigger()
    assert win.hint.text() != hint(1, 45)
    assert "Overhang" not in win.hint.text()
    assert not tinted(win, win.doc.groups[0].ents).any(), "the bottom on the bed stays its own colour"


def test_a_floating_cube_counts_its_bottom(win):
    """The bed is the lowest z of all visible geometry: the cube at z = 20 hangs over."""
    add(win, cube(100, 100, 0, 10), "on the bed")
    win.a_overhangs.trigger()
    add(win, cube(140, 100, 20, 10), "floating")
    assert win.hint.text() == hint(1, 45)
    assert int(tinted(win, win.doc.groups[1].ents).sum()) == 6, "the bottom face: two triangles"
    assert not tinted(win, win.doc.groups[0].ents).any()
    win.a_overhangs.trigger()
    assert "Overhang" not in win.hint.text(), "the count must not stay in the bar"
    assert win.hint.text() == win.view.tool.status_text()
    assert not tinted(win, win.doc.groups[1].ents).any(), "off: nothing is tinted"


def test_the_count_follows_the_model(win):
    """The hint is renewed on every change of the model, and disappears with the last overhang."""
    add(win, cube(100, 100, 0, 10), "on the bed")
    add(win, cube(140, 100, 20, 10), "floating")
    win.a_overhangs.trigger()
    assert win.hint.text() == hint(1, 45)
    win.doc.remove_group(win.doc.groups[1].id)
    win.doc.notify()
    assert "Overhang" not in win.hint.text()


def test_the_limit_of_the_preferences_is_used(win):
    """The hint follows the limit of the Preferences. A flat underside points straight down
    (90°), so it needs support at every limit the Preferences allow."""
    add(win, cube(100, 100, 0, 10), "on the bed")
    add(win, cube(140, 100, 20, 10), "floating")
    win.a_overhangs.trigger()
    assert win.hint.text() == hint(1, 45)
    win.settings.overhang_limit = 89
    win.view.update_overhang_status()
    assert win.hint.text() == hint(1, 89)
    win.settings.overhang_limit = 10
    win.view.update_overhang_status()
    assert win.hint.text() == hint(1, 10)


def test_the_overhang_limit_is_a_preference(win):
    """Window > Preferences > Drawing: Overhang limit, 0..89°, stored like the rest."""
    from volura.app import PreferencesDialog
    was = win.settings.overhang_limit
    try:
        dlg = PreferencesDialog(win, "Drawing")
        assert dlg.overhang_limit.value() == was
        assert dlg.overhang_limit.minimum() == 0 and dlg.overhang_limit.maximum() == 89
        dlg.overhang_limit.setValue(30)
        dlg.accept()
        assert win.settings.overhang_limit == 30
        assert type(win.settings)().overhang_limit == 30
    finally:
        win.settings.overhang_limit = was
        win.settings.save()
