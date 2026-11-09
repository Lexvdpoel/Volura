"""Round corners on the worksheet: a radius in the sharp corners of the selected paths.

`sheettools.round_corners` puts a tangent arc of `radius` in every vertex where the
direction turns by more than `min_angle` degrees, so a laser-cut panel has no sharp corners
left. A corner that sticks out and a corner that bites in both get one, the radius of a
corner is limited to the middle of the segment next to it so two corners on a short edge
cannot overlap, and a corner without room stays sharp. The two ends of an open path never
move, and a contour that repeats its first point is treated as the closed shape it is.

In the window (Qt offscreen) every selected path with a corner is replaced by its rounded
copy on the same layer and with the same source, in one undo step; nothing selected or
nothing to round leaves the undo stack untouched and says so in the status bar.

Run with:  python -m pytest tests/test_round_corners.py
"""
import hashlib
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura.sheettools import circle_loop, round_corners, seg_distance  # noqa: E402
from volura.vector2d import _point_in_poly, next_src, polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

R = 3.0
RECT = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
L = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 40.0], [0.0, 40.0]])
TRI = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 20.0]])


def area(pts):
    return abs(polygon_area(pts))


def has_point(pts, p, tol=1e-6):
    return bool(np.all(np.abs(np.asarray(pts, float) - np.asarray(p, float)) <= tol,
                       axis=1).any())


def crosses(a, b, c, d) -> bool:
    """Whether the segment `a`-`b` cuts `c`-`d`, with a shared end point allowed."""
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b)


def simple(pts) -> bool:
    """Whether the closed polyline `pts` cuts itself nowhere."""
    q = np.asarray(pts, dtype=float)
    segs = [(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            if i == 0 and j == len(segs) - 1:
                continue                        # the two segments that end in the start
            if crosses(*segs[i], *segs[j]):
                return False
    return True


def on_edge(pts, a, b):
    """The positions of the points of `pts` that lie on the segment `a`-`b`, along it."""
    d = b - a
    L = float(np.hypot(*d))
    d = d / L
    return [float((p - a) @ d) for p in np.asarray(pts, float)
            if seg_distance(p, a, b) <= 1e-9]


def zigzag(count=200, r_out=40.0, r_in=25.0) -> np.ndarray:
    """A closed star of `count` points, its radius out and in from point to point."""
    t = np.linspace(0.0, 2.0 * math.pi, count + 1)[:-1]
    rad = np.where(np.arange(count) % 2 == 0, r_out, r_in)
    return np.column_stack([rad * np.cos(t), rad * np.sin(t)])


def digest(q) -> str:
    """The fingerprint of a contour: its shape and every point rounded to a micrometre (
    the last digits differ between platforms; + 0.0 turns -0.0 into 0.0)."""
    q = np.ascontiguousarray(np.asarray(q, dtype=float).reshape(-1, 2), dtype=float)
    return hashlib.sha256((np.round(q, 6) + 0.0).tobytes() + str(q.shape).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the function

def test_a_rectangle_loses_exactly_its_four_corners():
    q = round_corners(RECT, R)
    assert area(q) == pytest.approx(40.0 * 20.0 - (4.0 - math.pi) * R * R, abs=0.05)


def test_every_point_of_a_rounded_rectangle_stays_inside_it():
    q = round_corners(RECT, R)
    assert (q[:, 0] >= -1e-9).all() and (q[:, 0] <= 40.0 + 1e-9).all()
    assert (q[:, 1] >= -1e-9).all() and (q[:, 1] <= 20.0 + 1e-9).all()
    on_edge = [min(seg_distance(p, RECT[i], RECT[(i + 1) % 4]) for i in range(4)) for p in q]
    assert all(_point_in_poly(RECT, p) or d <= 1e-9 for p, d in zip(q, on_edge)), \
        "an arc cuts the corner off, it never leaves the rectangle"


def test_the_straight_runs_of_a_rectangle_stay_exactly():
    extra = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
    q = round_corners(extra, R)
    assert has_point(q, (20.0, 0.0)), "a vertex that is no corner does not move"
    bottom = on_edge(q, RECT[0], RECT[1])
    assert min(bottom) == pytest.approx(R) and max(bottom) == pytest.approx(40.0 - R), \
        "the straight part stops where the arc begins"


def test_a_concave_corner_gives_material_back():
    q = round_corners(L, R)
    corner = R * R * (1.0 - math.pi / 4.0)      # what one square corner costs or gains
    sharp = 40.0 * 40.0 - 20.0 * 20.0
    assert area(q) == pytest.approx(sharp - 4.0 * corner, abs=0.05), \
        "five corners out of the material, the concave one back into it"
    assert area(q) > sharp - 5.0 * corner
    assert not _point_in_poly(L, (20.2, 20.2)) and _point_in_poly(q, (20.2, 20.2)), \
        "the arc of the inside corner fills the notch"


def test_a_radius_too_large_is_limited_to_the_segments():
    q = round_corners(TRI, 1000.0)
    assert simple(q), "the arcs of the three corners may not cross one another"
    assert area(q) < area(TRI)
    for i, a in enumerate(TRI):
        b = TRI[(i + 1) % 3]
        pos = on_edge(q, a, b)
        L = float(np.hypot(*(b - a)))
        assert pos == sorted(pos), "the two tangent points of an edge may not pass each other"
        assert min(pos) >= -1e-9 and max(pos) <= L + 1e-9
        assert not has_point(q, a), "a corner this round is no corner left"


def test_the_ends_of_an_open_path_stay_as_they_are():
    line = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 20.0]])
    q = round_corners(line, 4.0, closed=False)
    assert np.allclose(q[0], line[0]) and np.allclose(q[-1], line[-1])
    assert not has_point(q, (30.0, 0.0)), "the corner in the middle is rounded"
    assert len(q) > len(line)


def test_a_closed_path_rounds_its_start_point_too():
    q = round_corners(RECT, R)
    assert not has_point(q, RECT[0])


def test_a_repeated_closing_point_gives_the_same_result():
    closed = np.vstack([RECT, RECT[:1]])        # imports often close a contour by hand
    assert np.allclose(round_corners(closed, R), round_corners(RECT, R))


def test_points_on_top_of_each_other_are_dropped():
    pinched = np.vstack([RECT[:1], RECT])       # the start point twice
    assert np.allclose(round_corners(pinched, R), round_corners(RECT, R))


def test_no_radius_changes_nothing():
    for radius in (0.0, -1.0):
        assert np.array_equal(round_corners(RECT, radius), RECT)


def test_a_contour_without_sharp_corners_stays_as_it_is():
    loop = circle_loop((0.0, 0.0), 20.0, segments=64)
    assert np.array_equal(round_corners(loop, 2.0), loop), "5.6 degrees is no corner"


# ------------------------------------------------------------------ the exact contour

OPEN = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [60.0, 20.0], [60.0, 45.0]])

#: the fingerprint of the rounded contour for a few contours; a rewrite of the arithmetic
#: may not move a single point, so a corner that comes out elsewhere fails here
PINNED = [
    ("rectangle", RECT, dict(radius=R), "0909296e0028c825"),
    ("L-shape", L, dict(radius=R), "e7ada0af48e043dd"),
    ("circle-64", circle_loop((0.0, 0.0), 20.0, segments=64), dict(radius=2.0),
     "41671a2d1f1c66d8"),
    ("star-200", zigzag(), dict(radius=3.0), "e6bd27e3a2691d72"),
    ("open-polyline", OPEN, dict(radius=4.0, closed=False), "c13827d18d0c3ab0"),
]


@pytest.mark.parametrize("name, pts, kw, want", PINNED, ids=[p[0] for p in PINNED])
def test_the_rounded_contour_is_pinned(name, pts, kw, want):
    assert digest(round_corners(pts, **kw)) == want


# ------------------------------------------------------------------ in the window

@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


def path(win, pts, closed=True):
    """A path on the cutting layer of the worksheet, selected."""
    sheet = win.doc.sheet
    layer = sheet.default_layer("cut") or sheet.free_layer_name("cut")
    p = sheet.add_path(pts, closed=closed, layer=layer, src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.add_to_selection([p])
    return p


def test_a_rectangle_is_rounded_in_one_undo_step(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.corner_radius = 3.0
    p = path(win, [[0, 0], [60, 0], [60, 40], [0, 40]])
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    made = win.round_corners_dialog()
    assert len(win.doc.undo_stack) == steps + 1, "one undo step for all the paths"
    assert made is not None and len(made) == 1
    assert made[0].closed == p.closed, "a shape stays a shape"
    assert made[0].layer == p.layer and int(made[0].src) == int(p.src), \
        "same layer and source, so the part still moves and nests as one"
    assert len(made[0]) > len(p), "the arcs are new points"
    assert win.hint.text() == "1 path rounded."
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_open_and_closed_paths_are_rounded_together(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.corner_radius = 2.0
    closed = path(win, [[0, 0], [40, 0], [40, 20], [0, 20]])
    opened = path(win, [[60, 0], [80, 0], [80, 20]], closed=False)
    made = win.round_corners_dialog()
    assert made is not None and len(made) == 2
    assert [q.closed for q in made] == [closed.closed, opened.closed]
    assert np.allclose(made[1].pts[0], np.asarray(opened.pts, float)[0]), "the end stays"
    assert win.hint.text() == "2 paths rounded."


def test_a_path_without_corners_changes_nothing(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.corner_radius = 2.0
    line = path(win, [[0, 0], [40, 0], [80, 0]], closed=False)
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.round_corners_dialog() is None
    assert len(win.doc.undo_stack) == steps, "nothing changed, so no undo step"
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "1 path has no corners to round."


def test_nothing_selected_is_no_undo_step(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing selected: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    win.set_mode("2d")
    win.sheet_view.clear_selection()
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.round_corners_dialog() is None
    assert len(win.doc.undo_stack) == steps
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select the paths to round first."


def test_the_dialog_starts_at_the_last_radius_and_remembers_the_new_one(win, monkeypatch):
    win.settings.corner_radius = 5.0
    path(win, [[0, 0], [40, 0], [40, 20], [0, 20]])
    seen = {}

    def fill(dlg):
        seen["start"] = round(dlg.findChild(QDoubleSpinBox).value(), 2)
        seen["range"] = (dlg.findChild(QDoubleSpinBox).minimum(),
                         dlg.findChild(QDoubleSpinBox).maximum())
        dlg.findChild(QDoubleSpinBox).setValue(1.5)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.round_corners_dialog()
    assert seen["start"] == 5.0
    assert seen["range"] == (0.1, 100.0)
    assert win.settings.corner_radius == pytest.approx(1.5)
