"""Print-in-place ball joints: a ball in its socket that prints in one piece and turns at once.

`ball_joint.ball_joint` gives the joint as two closed volumes that are already assembled and stand
on `z = 0`: the socket — a floor, a wall, a neck and the bowl the ball lies in — and the ball with
its stem, which stands on a slender foot that is broken off after printing. The two keep
`clearance` millimetres between them, the ball is `capture` millimetres behind the neck so it
cannot fall out, and the stem tilts `swing` degrees to either side before the joint stops.

These tests weigh a joint against what it promises: two volumes that are not fused and keep their
play, a ball that is caught, a joint that turns to the swing and no further, every face of it
printable without support, slits that leave springs and one wall, and a foot that is an island of
its own on the bed. A size that cannot be printed becomes the nearest one that can. The place in
the interface is checked too (Parts between the knob and the hinge, two groups in one undo step,
and the dialog that remembers its values), and a real slice says the joint needs no support.

Run with:  python -m pytest tests/test_ball_joint.py
"""
import math
import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox)  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import ball_joint, cura_profiles, slicer  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BALL, CLEAR, STEM, SWING, WALL, BASE_FLOOR, LENGTH, GRIP = (
    12.0, 0.3, 3.0, 35.0, 2.5, 3.0, 12.0, 4)
BASE = dict(ball_d=BALL, clearance=CLEAR, stem_d=STEM, swing=SWING, wall=WALL, base=BASE_FLOOR,
            stem_len=LENGTH, grip=GRIP)
OPTIONS = dict(BASE)

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def make(**kw):
    return ball_joint.ball_joint(**dict(BASE, **kw))


def about(m, angle, axis_azimuth, cz):
    """The joint turned `angle` degrees around the ball's centre, about the axis through it that
    points in the direction `axis_azimuth`: the stem leans towards that azimuth."""
    t = math.radians(angle)
    a = math.radians(axis_azimuth)
    ux, uy = math.cos(a), math.sin(a)
    c, s = math.cos(t), math.sin(t)
    C = 1.0 - c
    rot = np.array([[c + ux * ux * C, ux * uy * C, uy * s],
                    [ux * uy * C, c + uy * uy * C, -ux * s],
                    [-uy * s, ux * s, c]])
    pivot = np.array([[0.0], [0.0], [cz]])
    return m.transform(np.hstack([rot, pivot - rot @ pivot]))


def facets(m):
    """The unit normals, the centroids and the areas of the triangles of a manifold."""
    mesh = m.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-12
    return n[ok] / length[ok][:, None], (a + b + c)[ok] / 3.0, length[ok] / 2.0


def overhangs(m, min_area=0.02):
    """The faces that lean steeper than the printable limit and are too big to be a hair of the
    boolean: a joint that prints without support has none of them above the bed."""
    n, centre, area = facets(m)
    steep = n[:, 2] < -math.sin(math.pi / 4.0) - 0.01
    return int((steep & (centre[:, 2] > 1e-6) & (area > min_area)).sum())


def islands(m, z=0.01):
    """The pieces a manifold touches the bed with: the widest reach of every island of the thin
    slice at `z`."""
    slab = mf.Manifold.cylinder(z, 60.0, 60.0, 64).translate((0.0, 0.0, z))
    out = []
    for part in (m ^ slab).decompose():
        b = np.asarray(part.bounding_box(), dtype=float)
        out.append(max(abs(b[0]), abs(b[1]), b[3], b[4]))
    return sorted(out)


def arcs(m, z):
    """How many pieces a thin horizontal slice of a manifold falls apart into."""
    slab = mf.Manifold.cylinder(0.1, 60.0, 60.0, 64).translate((0.0, 0.0, z))
    return len((m ^ slab).decompose())


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the joint

def test_the_default_joint_is_two_valid_volumes_that_stand_on_the_bed():
    socket, moving = make()
    for m in (socket, moving):
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert m.volume() > 0.0, "the joint is not a volume"
        assert bbox(m)[0][2] == pytest.approx(0.0), "the part does not stand on the bed"
    assert bbox(socket)[0][2] == 0.0 and bbox(moving)[0][2] == 0.0


def test_the_two_parts_are_not_fused_and_keep_their_play():
    socket, moving = make()
    assert len(socket.decompose()) == 1, "the socket is not one piece"
    assert len(moving.decompose()) == 1, "the ball is not one piece"
    assert len(ball_joint.joint(**BASE).decompose()) == 2, "the two parts printed into one"
    assert moving.min_gap(socket, 1.0) >= CLEAR - 0.01, "the ball rubs its socket"
    grown = mf.Manifold.sphere(max(0.02, CLEAR - 0.02), 16)
    assert (moving.minkowski_sum(grown) ^ socket).volume() == pytest.approx(0.0, abs=1e-6)


def test_the_ball_is_caught_behind_the_neck():
    n = ball_joint.sizes(**BASE)
    assert n["capture"] >= ball_joint.CAPTURE_MIN, "the neck holds nothing of the ball"
    assert n["opening"] < n["ball"], "the ball falls out of the socket"
    assert n["stem"] <= n["opening"] - 2.0 * CLEAR + 1e-6, "the stem rubs the neck"
    assert n["neck_z"] > n["centre"], "the neck stands below the middle of the ball"


def test_the_box_round_the_joint_is_the_size_asked():
    n = ball_joint.sizes(**BASE)
    lo, hi = bbox(ball_joint.joint(**BASE))
    assert hi[0] - lo[0] == pytest.approx(n["base_d"]), "the socket is not as wide as it says"
    assert hi[1] - lo[1] == pytest.approx(n["base_d"]), "the socket is not as wide as it says"
    assert hi[2] - lo[2] == pytest.approx(n["height"]), "the stem does not reach as high as it says"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the joint is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the joint is not centred"


def test_the_joint_turns_to_the_swing_it_promises_and_no_further():
    socket, moving = make()
    n = ball_joint.sizes(**BASE)
    free = (about(moving, n["swing"] - 0.5, 90.0, n["centre"]) ^ socket).volume()
    assert free == pytest.approx(0.0, abs=1e-6), "the joint rubs before it reaches its swing"
    held = (about(moving, n["swing"] + 5.0, 90.0, n["centre"]) ^ socket).volume()
    assert held > 0.01, "the neck does not stop the ball past its swing"


def test_the_joint_turns_whatever_it_was_asked():
    for kw in (dict(swing=5.0), dict(swing=20.0), dict(swing=60.0), dict(stem_d=6.0),
               dict(ball_d=8.0), dict(ball_d=20.0, stem_d=5.0), dict(clearance=0.15),
               dict(clearance=0.6), dict(base=1.0), dict(base=10.0), dict(grip=0), dict(grip=8),
               dict(wall=8.0), dict(stem_len=4.0)):
        socket, moving = make(**kw)
        n = ball_joint.sizes(**dict(BASE, **kw))
        free = (about(moving, n["swing"] - 0.5, 90.0, n["centre"]) ^ socket).volume()
        assert free == pytest.approx(0.0, abs=1e-6), f"{kw}: the parts touch at rest or at the tilt"


def test_every_face_of_the_joint_prints_without_support():
    socket, moving = make()
    assert overhangs(moving) == 0, "the ball leans steeper than the bed allows"
    assert overhangs(socket) == 0, "the socket leans steeper than the bed allows"
    for kw in (dict(ball_d=8.0), dict(ball_d=20.0, stem_d=5.0), dict(ball_d=40.0),
               dict(swing=5.0), dict(clearance=0.6), dict(base=1.0), dict(base=10.0)):
        socket, moving = make(**kw)
        assert overhangs(moving) == 0, f"{kw}: the ball needs support"
        assert overhangs(socket) == 0, f"{kw}: the socket needs support"


def test_the_grip_slits_leave_springs_and_one_wall():
    for grip in (1, 4, 8):
        socket, moving = make(grip=grip)
        n = ball_joint.sizes(**dict(BASE, grip=grip))
        assert arcs(socket, n["top_z"] - 0.3) == grip, f"{grip} slits are not {grip} springs"
        assert len(socket.decompose()) == 1, f"{grip} slits saw the wall off"
    loose, _moving = make(grip=0)
    assert arcs(loose, ball_joint.sizes(**dict(BASE, grip=0))["top_z"] - 0.3) == 1, \
        "a joint without slits is sawn anyway"


def test_the_breakaway_foot_is_an_island_of_its_own_on_the_bed():
    socket, moving = make()
    assert islands(moving) == [pytest.approx(ball_joint.FOOT, abs=0.05)], \
        "the ball does not stand on one slender foot"
    reach = islands(socket)
    assert len(reach) == 1, "the socket does not stand on one ring of its own"
    assert reach[0] == pytest.approx(ball_joint.sizes(**BASE)["base_d"] / 2.0, abs=0.05)
    assert islands(moving)[0] < reach[0] / 4.0, "the foot is too thick to break off"


def test_the_two_volumes_become_the_joint_of_the_module():
    socket, moving = make()
    both = ball_joint.joint(**BASE)
    assert both.status() == mf.Error.NoError
    assert both.volume() == pytest.approx(socket.volume() + moving.volume(), rel=1e-3)


# ------------------------------------------------------------------ sizes that have to give

@pytest.mark.parametrize("kw", [
    dict(ball_d=4000.0, stem_d=4000.0, swing=200.0, clearance=9.0, wall=90.0, base=90.0,
         stem_len=900.0, grip=90),
    dict(ball_d=0.1, stem_d=0.1, swing=0.0, clearance=0.0, wall=0.0, base=0.0, stem_len=0.0,
         grip=-4),
    dict(ball_d=8.0, stem_d=16.0, swing=60.0, clearance=0.6),
    dict(ball_d=40.0, stem_d=16.0, swing=60.0, clearance=0.15),
    dict(ball_d=8.1, clearance=0.15, stem_d=1.6),
    dict(ball_d=40.0, clearance=0.6, base=1.0),
])
def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can(kw):
    size = dict(BASE, **kw)
    n = ball_joint.sizes(**size)
    assert n["ball"] == pytest.approx(max(ball_joint.BALL_MIN, min(ball_joint.BALL_MAX,
                                                                   kw.get("ball_d", BALL))))
    assert n["swing"] >= ball_joint.SWING_MIN, "the swing was taken away completely"
    assert n["capture"] >= ball_joint.CAPTURE_MIN - 1e-6, "nothing holds the ball any more"
    assert n["opening"] < n["ball"], "the ball falls out of the socket"
    assert n["stem"] <= n["opening"] - 2.0 * n["clearance"] + 1e-6, "the stem rubs the neck"
    assert n["stem"] < n["ball"], "the stem swallows the ball"
    assert ball_joint.BASE_MIN <= n["floor"] <= ball_joint.BASE_MAX
    assert ball_joint.GRIP_MIN <= n["grip"] <= ball_joint.GRIP_MAX
    socket, moving = make(**kw)
    for m in (socket, moving):
        assert m.status() == mf.Error.NoError and m.volume() > 0.0, f"{kw}: nothing left to print"
    assert moving.min_gap(socket, 1.0) >= n["clearance"] - 0.05, f"{kw}: the parts rub"
    free = (about(moving, n["swing"] - 0.5, 90.0, n["centre"]) ^ socket).volume()
    assert free == pytest.approx(0.0, abs=1e-6), f"{kw}: the clamped joint rubs"


def test_a_swing_that_does_not_fit_becomes_the_widest_one_that_holds():
    n = ball_joint.sizes(**dict(BASE, swing=60.0))
    assert n["swing_asked"] == 60.0, "what was asked for is not told back"
    assert ball_joint.SWING_MIN <= n["swing"] < 60.0, "the swing was not taken down at all"
    asked = ball_joint.sizes(**dict(BASE, swing=500.0))
    assert asked["swing_asked"] == ball_joint.SWING_MAX
    assert asked["swing"] <= ball_joint.SWING_MAX


def test_a_stem_that_leaves_nothing_to_hold_becomes_a_thinner_one():
    n = ball_joint.sizes(**dict(BASE, ball_d=8.0, stem_d=ball_joint.STEM_MAX))
    assert ball_joint.STEM_MIN <= n["stem"] < 8.0, "the stem is still wider than the ball"
    assert n["capture"] >= ball_joint.CAPTURE_MIN - 1e-6
    assert n["swing"] >= ball_joint.SWING_MIN


def test_a_big_ball_stands_on_a_floor_thick_enough_to_be_printable():
    n = ball_joint.sizes(**dict(BASE, ball_d=20.0, base=1.0))
    assert n["floor"] > ball_joint.BASE_MIN + 0.5, "the floor stayed as thin as asked"
    assert n["tip_slope"] <= ball_joint.TIP_LIMIT + 1e-6, "the tip leans further than it may"
    assert n["centre"] > n["floor"] + 0.5 * n["ball"], "the ball is sunk in its floor"
    assert n["foot"] <= ball_joint.FOOT + 1e-9, "the foot is thicker than a breakaway foot may be"


# ------------------------------------------------------------------ a real slice

def _mesh(m, at=(0.0, 0.0)):
    placed = m.translate((at[0], at[1], 0.0)).to_mesh()
    return np.asarray(placed.vert_properties)[:, :3], np.asarray(placed.tri_verts), 0


def _first_layer_moves(text):
    """The (x, y) of every extrusion move of the first layer, as the printer lays them down."""
    body = text.split(";LAYER:1", 1)[0].split(";LAYER:0", 1)[-1]
    x = y = 0.0
    points = []
    for line in body.splitlines():
        code = line.split(";", 1)[0].strip()
        if not code.startswith(("G0", "G1")):
            continue
        got = {a: re.search(rf"\b{a}(-?\d*\.?\d+)", code) for a in "XYZE"}
        for axis in ("X", "Y"):
            if got[axis] is not None:
                value = float(got[axis].group(1))
                if axis == "X":
                    x = value
                else:
                    y = value
        if got["E"] is not None:
            points.append((x, y))
    return points


@needs_engine
def test_the_joint_slices_without_support_and_both_parts_stand_on_the_first_layer(tmp_path):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v, m = p.default_variant(), p.default_material(p.default_variant())
    stacks = p.stacks(v, m, "standard", {"infill_sparse_density": 0.0, "support_enabled": False})
    socket, moving = make()
    out = tmp_path / "ball_joint.gcode"
    slicer.slice_meshes(stacks, [_mesh(socket), _mesh(moving)], str(out))
    text = out.read_text(encoding="utf-8")
    assert ";TYPE:SUPPORT" not in text, "the joint needs support it is not meant to have"
    moves = _first_layer_moves(text)
    assert moves, "nothing is extruded on the first layer"
    xs = [x for x, _y in moves]
    ys = [y for _x, y in moves]
    middle = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    machine = slicer.resolved_settings(stacks)["global"]
    assert middle[0] == pytest.approx(0.5 * float(machine["machine_width"]), abs=2.0), \
        "the joint is not printed in the middle of the bed"
    assert middle[1] == pytest.approx(0.5 * float(machine["machine_depth"]), abs=2.0)
    reach = [math.hypot(x - middle[0], y - middle[1]) for x, y in moves]
    assert any(r <= ball_joint.FOOT + 0.4 for r in reach), "the foot of the ball is not on the bed"
    assert any(r >= 4.0 for r in reach), "the floor of the socket is not on the bed"
    assert not [r for r in reach if 1.2 <= r <= 2.2], "the foot prints into the floor it stands on"


# ------------------------------------------------------------------ in the window

def test_the_joint_is_two_groups_that_one_undo_takes_away(win):
    socket_out, ball_out = make()
    steps = len(win.doc.undo_stack)
    win.make_ball_joint_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no ball joint in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the joint is not one undo step"
    socket, moving = win.doc.groups
    assert [g.name for g in (socket, moving)] == ["Ball socket", "Ball and stem"]
    assert pump_until(lambda: socket.ents.to_manifold().volume() > 0.0)
    assert socket.ents.to_manifold().volume() == pytest.approx(socket_out.volume(), rel=1e-3)
    assert moving.ents.to_manifold().volume() == pytest.approx(ball_out.volume(), rel=1e-3)
    assert moving.ents.to_manifold().min_gap(socket.ents.to_manifold(), 1.0) >= CLEAR - 0.05, \
        "the two groups were moved apart from each other"
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_two_parts_keep_where_they_print_and_move_on_the_bed_together(win):
    win.make_ball_joint_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no ball joint in the model"
    assert pump_until(lambda: all(g.bbox() is not None for g in win.doc.groups))
    bx, by, _bz = win.settings.bed_size
    boxes = [np.asarray(g.bbox(), dtype=float) for g in win.doc.groups]
    for lo, _hi in boxes:
        assert lo[2] == pytest.approx(0.0), "the part does not stand on the bed"
    for lo, hi in boxes:
        assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the joint is not on the bed"
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by), "the joint is not on the bed"
    assert pump_until(lambda: "ball joint" in win.hint.text().lower()), win.hint.text()
    win.make_ball_joint_model(dict(OPTIONS, ball_d=16.0))
    assert pump_until(lambda: len(win.doc.groups) == 4), "the second joint is missing"
    assert pump_until(lambda: all(g.bbox() is not None for g in win.doc.groups))
    first, _first_ball, second, _second_ball = (np.asarray(g.bbox(), dtype=float)
                                               for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the joints are not 10 mm apart"


def test_the_ball_joint_follows_the_knob_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.parts_category("Mechanical"))]
    assert labels.index("Print-in-Place Ball Joint") == labels.index("Knob") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [round(dlg.ball.value(), 2), round(dlg.clearance.value(), 2),
                        round(dlg.stem.value(), 2), round(dlg.swing.value(), 1),
                        round(dlg.wall.value(), 2), round(dlg.base.value(), 2),
                        round(dlg.stem_len.value(), 2), int(dlg.grip.value())]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        seen["info"] = dlg.info.text()
        seen["calm"] = dlg.info.styleSheet()
        dlg.ball.setValue(16.0)
        dlg.stem.setValue(16.0)
        dlg.grip.setValue(6)
        seen["warn"] = dlg.info.styleSheet()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.ball_joint_ball, win.settings.ball_joint_clearance = 10.0, 0.35
    win.settings.ball_joint_stem, win.settings.ball_joint_swing = 4.0, 25.0
    win.settings.ball_joint_wall, win.settings.ball_joint_base = 2.0, 4.0
    win.settings.ball_joint_stem_len, win.settings.ball_joint_grip = 10.0, 2
    win.ball_joint_dialog()
    assert seen["title"] == "Print-in-place ball joint"
    assert seen["start"] == [10.0, 0.35, 4.0, 25.0, 2.0, 4.0, 10.0, 2]
    assert seen["spins"] == 7, "the swing is the only field without a millimetre behind it"
    assert seen["info"].startswith("<i>") and "mm" in seen["info"]
    assert seen["calm"] == "", "a joint that fits is marked as a problem"
    assert seen["warn"], "a stem that leaves nothing to hold is not marked"
    assert pump_until(lambda: len(win.doc.groups) == 2), "no ball joint in the model"
    assert int(win.settings.ball_joint_grip) == 6, "the slits are not remembered"
    s = app_module.Settings()
    assert (s.ball_joint_ball, s.ball_joint_stem, s.ball_joint_grip) == (pytest.approx(16.0),
                                                                        pytest.approx(16.0), 6)
    assert s.ball_joint_clearance == pytest.approx(0.35)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
