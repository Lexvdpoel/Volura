"""The two free-form shapes of the vent pattern: Voronoi cells and slots.

`boxgen.vent_pattern` with `shape="voronoi"` scatters seeds on a jittered grid - the
draw is decided by `seed` alone, so the same seed gives the same cells - and cuts each
cell back half a web from every shared bisector, so two neighbours leave `web` of
material between them. Cells smaller than a tenth of `size` squared go, and so does
everything the edge margin cuts off. With `shape="slots"` the holes are stadiums: rows
of slots of `slot_length` (four times the hole width by default) and `size` wide, a web
between the rows and half a slot of offset, and `angle` turns the rows by 0, 45 or 90
degrees. The web of both shapes is checked by the shortest vertex-to-edge distance, the
margin by subtracting the contour shrunk by the margin from every hole.

In the window the new options travel with the pattern: the dialog remembers them
(Settings), `Document.vents` keeps them, a refit makes the same grille again for the
new outline, a saved file brings them back, and an old file without them gets the
defaults. On a linked wall the grille is cut through the wall, as the hexagons are.

Run with:  python -m pytest tests/test_vent_pattern_extra.py
"""
import itertools
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QPushButton  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import boxgen, cam, fileio, linking, ops  # noqa: E402
from volura.sheet_dialogs import VentPatternDialog  # noqa: E402
from volura.sheettools import circle_loop  # noqa: E402
from volura.vector2d import next_src, polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

RECT = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]])
SQUARE = np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [0.0, 60.0]])
L_SHAPED = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [55.0, 60.0],
                     [55.0, 30.0], [0.0, 30.0]])
VOR = {"shape": "voronoi", "size": 10.0, "web": 2.0, "margin": 3.0, "seed": 1}
SLOT = {"shape": "slots", "size": 6.0, "web": 2.0, "margin": 3.0}
SIZE, WEB, MARGIN = 6.0, 2.0, 3.0

X, Y, Z = 60.0, 40.0, 30.0     # mm, the outside of the box for the linked-wall test
WALL = 2.0                     # mm, the thickness of its walls


def as_sets(holes):
    """The holes as a sorted list of rounded points, so two patterns can be compared."""
    return sorted(np.round(q, 9).tolist() for q in holes)


def section(*loops):
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float)) for q in loops],
                           mf.FillRule.NonZero)


def min_gap(holes, limit=None):
    """The shortest distance between the edges of any two holes.

    The closest points of two hole boundaries include a corner of one seen from an edge
    of the other, and `_pair_dists` measures that exactly for a whole list of pairs -
    with `limit`, only the pairs whose corner circles can still reach that far.
    """
    c = np.array([np.asarray(q).mean(axis=0) for q in holes])
    r = np.array([float(np.max(np.hypot(*(np.asarray(q) - ci).T)))
                  for q, ci in zip(holes, c)])
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    pairs = [(i, j) for i, j in itertools.combinations(range(len(holes)), 2)
             if limit is None or d[i, j] <= limit + r[i] + r[j]]
    return min(boxgen._pair_dists(holes, pairs), default=float("inf"))


def inside_room(holes, outline, margin):
    """No hole stands outside the contour shrunk by the margin (the fuzz standard)."""
    room = cam.offset_loop(outline, -margin)
    if not room:
        assert not holes, "the margin ate the whole contour, so no hole may be left"
        return
    outside = mf.CrossSection.batch_boolean([section(*holes), section(*room)],
                                            mf.OpType.Subtract)
    assert outside.area() < 1e-6, "a hole stands outside the contour shrunk by the margin"


def no_overlap(holes):
    """Two holes never lie over each other: the area of all of them is their sum."""
    assert sum(abs(polygon_area(q)) for q in holes) - section(*holes).area() < 1e-6


# ------------------------------------------------------------------ Voronoi

def test_the_voronoi_mosaic_fills_the_rectangle():
    holes = boxgen.vent_pattern(RECT, **VOR)
    assert 40 <= len(holes) <= 75
    inside_room(holes, RECT, MARGIN)
    no_overlap(holes)


def test_two_neighbouring_cells_keep_the_web():
    holes = boxgen.vent_pattern(RECT, **VOR)
    tight = min_gap(holes)
    assert tight >= WEB - 1e-3, "a pair of cells stands closer than the web"
    assert tight <= WEB + 0.25, "the cells are not packed up to the web"


def test_cells_smaller_than_a_tenth_of_the_hole_area_are_gone():
    holes = boxgen.vent_pattern(RECT, **VOR)
    for q in holes:
        assert abs(polygon_area(q)) >= 0.10 * VOR["size"] ** 2 - 1e-6


def test_the_same_seed_makes_the_same_cells_and_another_seed_other_cells():
    a = boxgen.vent_pattern(RECT, **dict(VOR, seed=7))
    b = boxgen.vent_pattern(RECT, **dict(VOR, seed=7))
    other = boxgen.vent_pattern(RECT, **dict(VOR, seed=8))
    assert as_sets(a) == as_sets(b)
    assert as_sets(a) != as_sets(other)


def test_a_circle_and_an_L_shape_take_the_mosaic_too():
    circle = boxgen.vent_pattern(circle_loop((0.0, 0.0), 50.0), **VOR)
    assert 45 <= len(circle) <= 95
    inside_room(circle, circle_loop((0.0, 0.0), 50.0), MARGIN)
    corners = boxgen.vent_pattern(L_SHAPED, **dict(VOR, seed=3))
    assert corners
    assert min_gap(corners, limit=WEB + 0.1) >= WEB - 1e-3, \
        "the notch of the L pinches two cells together"


def test_a_closed_or_clockwise_contour_changes_nothing():
    base = boxgen.vent_pattern(RECT, **VOR)
    assert as_sets(boxgen.vent_pattern(np.vstack([RECT, RECT[:1]]), **VOR)) == as_sets(base)
    assert as_sets(boxgen.vent_pattern(RECT[::-1], **VOR)) == as_sets(base)


def test_a_contour_too_small_for_a_cell_holds_none():
    assert boxgen.vent_pattern([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]], **VOR) == []
    assert boxgen.vent_pattern(RECT, **dict(VOR, margin=40.0)) == []


def test_too_many_cells_ask_for_bigger_ones():
    with pytest.raises(ValueError, match="too many cells"):
        boxgen.vent_pattern([[0.0, 0.0], [800.0, 0.0], [800.0, 400.0], [0.0, 400.0]],
                            shape="voronoi", size=0.5, web=1.0, margin=2.0, seed=1)


# ------------------------------------------------------------------ slots

def test_a_slot_is_a_stadium_of_four_times_the_hole_width():
    holes = boxgen.vent_pattern(RECT, **SLOT)
    assert len(holes) == 17
    for q in holes:
        w = float(q[:, 0].max() - q[:, 0].min())
        h = float(q[:, 1].max() - q[:, 1].min())
        assert w == pytest.approx(4.0 * SIZE, abs=0.05)
        assert h == pytest.approx(SIZE, abs=1e-6)
        assert abs(polygon_area(q)) == pytest.approx(18.0 * 6.0 + math.pi * 3.0 ** 2, abs=0.6)


def test_the_rows_alternate_half_a_slot_and_stand_one_web_apart():
    holes = boxgen.vent_pattern(RECT, **SLOT)
    rows = {}
    for q in holes:
        rows.setdefault(round(float(q.mean(axis=0)[1]), 6), []).append(float(q.mean(axis=0)[0]))
    ys = sorted(rows)
    assert len(ys) >= 3, "the room of the rectangle fits several rows of slots"
    for a, b in zip(ys, ys[1:]):
        assert b - a == pytest.approx(SIZE + WEB, abs=1e-6)
    step = 4.0 * SIZE + WEB
    for xs in rows.values():
        assert len(xs) > 1
        assert min(b - a for a, b in itertools.combinations(sorted(xs), 2)) \
            == pytest.approx(step, abs=1e-6)
    for ya, yb in zip(ys, ys[1:]):                # each next row is shifted half a slot
        for xa in rows[ya]:
            assert min(abs(xa - xb) % step for xb in rows[yb]) == pytest.approx(step / 2.0,
                                                                                abs=1e-6)


def test_the_slot_web_is_web_wide_and_the_slots_stay_inside_the_margin():
    holes = boxgen.vent_pattern(RECT, **SLOT)
    tight = min_gap(holes)
    assert WEB - 1e-3 <= tight <= WEB + 0.1
    inside_room(holes, RECT, MARGIN)
    no_overlap(holes)


def test_the_angle_option_turns_the_rows():
    straight = boxgen.vent_pattern(RECT, **SLOT)
    upright = boxgen.vent_pattern(RECT, **dict(SLOT, angle=90.0))
    assert len(upright) >= 10
    for q in upright:
        assert float(q[:, 0].max() - q[:, 0].min()) == pytest.approx(SIZE, abs=1e-6)
        assert float(q[:, 1].max() - q[:, 1].min()) == pytest.approx(4.0 * SIZE, abs=0.05)
    turned = boxgen.vent_pattern(RECT, **dict(SLOT, angle=45.0))
    assert turned and len(turned) < len(straight)
    for q in turned:
        w = float(q[:, 0].max() - q[:, 0].min())
        h = float(q[:, 1].max() - q[:, 1].min())
        assert w == pytest.approx(h, abs=1e-3)               # symmetric box at 45 degrees
        # the straight part turns, the round ends always add the full width
        assert w == pytest.approx((4.0 * SIZE - SIZE) / math.sqrt(2.0) + SIZE, abs=0.05)
    inside_room(turned, RECT, MARGIN)
    inside_room(upright, RECT, MARGIN)
    assert min_gap(turned, limit=WEB + 0.1) >= WEB - 1e-3
    assert min_gap(upright, limit=WEB + 0.1) >= WEB - 1e-3


def test_slot_length_makes_longer_slots_and_a_shorter_grille():
    holes = boxgen.vent_pattern(RECT, **dict(SLOT, slot_length=40.0))
    assert holes
    assert len(holes) < 17
    for q in holes:
        assert float(q[:, 0].max() - q[:, 0].min()) == pytest.approx(40.0, abs=0.05)
    assert min_gap(holes, limit=WEB + 0.1) >= WEB - 1e-3


def test_a_slot_shorter_than_it_is_wide_is_refused():
    with pytest.raises(ValueError, match="at least as long"):
        boxgen.vent_pattern(RECT, **dict(SLOT, slot_length=4.0))


def test_slots_are_made_always_the_same_and_need_room_too():
    assert as_sets(boxgen.vent_pattern(RECT, **dict(SLOT, angle=45.0))) \
        == as_sets(boxgen.vent_pattern(RECT, **dict(SLOT, angle=45.0)))
    assert boxgen.vent_pattern([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]], **SLOT) == []
    assert boxgen.vent_pattern(RECT, **dict(SLOT, margin=40.0)) == []


# ------------------------------------------------------------------ remembered

def test_an_old_file_gives_the_new_options_their_defaults():
    old = linking.Vent.from_dict({"src": 5, "shape": "voronoi", "size": 8.0})
    assert old.seed == 1 and old.slot_length == 0.0 and old.angle == 0.0
    back = linking.Vent.from_dict(linking.Vent(5, "slots", 6.0, 2.0, 3.0,
                                               seed=42, slot_length=30.0, angle=90.0).to_dict())
    assert (back.seed, back.slot_length, back.angle) == (42, 30.0, 90.0)


# ------------------------------------------------------------------ mini fuzz

def outline(rng):
    """A random closed contour: a star (often non-convex), a rectangle or a circle."""
    kind = int(rng.integers(0, 3))
    if kind == 1:
        w, h = rng.uniform(40.0, 120.0, 2)
        x0, y0 = rng.uniform(-40.0, 40.0, 2)
        return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    if kind == 2:
        return circle_loop(rng.uniform(-30.0, 30.0, 2), float(rng.uniform(30.0, 60.0)))
    n = int(rng.integers(5, 13))
    r_out = float(rng.uniform(45.0, 90.0))
    r_in = r_out * float(rng.uniform(0.55, 0.95))
    a = np.linspace(0.0, 2.0 * math.pi, 2 * n, endpoint=False) + rng.uniform(0.0, math.pi)
    rad = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.column_stack([rad * np.cos(a), rad * np.sin(a)])


@pytest.mark.parametrize("seed", range(6))
def test_voronoi_and_slots_of_random_outlines(seed):
    rng = np.random.default_rng(3000 + seed)
    pts = outline(rng)
    vor = dict(shape="voronoi", size=float(rng.uniform(7.0, 10.0)), web=float(rng.uniform(1.5, 3.0)),
               margin=float(rng.uniform(0.5, 2.5)), seed=int(seed) + 1)
    holes = boxgen.vent_pattern(pts, **vor)
    inside_room(holes, pts, vor["margin"])
    no_overlap(holes)
    assert min_gap(holes, limit=vor["web"] + 0.1) >= vor["web"] - 1e-3, \
        f"seed {seed}: two cells stand closer than the web"
    slot = dict(shape="slots", size=float(rng.uniform(4.0, 6.0)), web=float(rng.uniform(1.5, 3.0)),
                margin=float(rng.uniform(0.5, 2.5)), angle=float(45 * (seed % 3)))
    holes = boxgen.vent_pattern(pts, **slot)
    inside_room(holes, pts, slot["margin"])
    no_overlap(holes)
    assert min_gap(holes, limit=slot["web"] + 0.1) >= slot["web"] - 1e-3, \
        f"seed {seed}: two slots stand closer than the web"
    assert holes, f"seed {seed}: neither shape made a single hole"


# ------------------------------------------------------------------ in the window

@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.settings.vent_shape, w.settings.vent_size = "voronoi", 10.0
    w.settings.vent_web, w.settings.vent_margin = 2.0, 3.0
    w.settings.vent_seed, w.settings.vent_slot_length, w.settings.vent_angle = 7, 0.0, 0.0
    yield w
    w.doc.dirty = False
    w.close()


def contour(win, pts=RECT):
    sheet = win.doc.sheet
    p = sheet.add_path(pts, closed=True, layer=sheet.default_layer("cut"),
                       src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.set_selection([p])
    return p


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


def holes_of(win, outline_path):
    return [p for p in win.doc.sheet.paths
            if int(p.src) == int(outline_path.src) and p is not outline_path and p.closed]


def test_the_dialog_shows_only_the_options_of_the_chosen_shape(win):
    win.settings.vent_shape = "hex"
    dlg = VentPatternDialog(win)
    assert not dlg.slotlen.isVisibleTo(dlg) and not dlg.angle.isVisibleTo(dlg)
    assert not dlg.seed.isVisibleTo(dlg)
    dlg.shape.setCurrentIndex(dlg.shape.findData("voronoi"))
    assert dlg.seed.isVisibleTo(dlg) and not dlg.slotlen.isVisibleTo(dlg)
    dlg.shape.setCurrentIndex(dlg.shape.findData("slots"))
    assert dlg.slotlen.isVisibleTo(dlg) and dlg.angle.isVisibleTo(dlg)
    assert not dlg.seed.isVisibleTo(dlg)
    assert dlg.slotlen.value() == 0, "0 mm reads as automatic, four times the hole width"


def test_shuffle_draws_a_fresh_seed(win, monkeypatch):
    from volura import sheet_dialogs
    asked = []

    def fake_randint(lo, hi):
        asked.append((lo, hi))
        return 4242

    monkeypatch.setattr(sheet_dialogs.random, "randint", fake_randint)
    dlg = VentPatternDialog(win)
    dlg.shape.setCurrentIndex(dlg.shape.findData("voronoi"))
    shuffle = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Shuffle")
    shuffle.click()
    assert dlg.seed.value() == 4242
    assert dlg.options()["seed"] == 4242
    assert asked, "the shuffle asked for one number in the spin's range"


def test_the_action_cuts_the_mosaic_of_the_remembered_seed(win, monkeypatch):
    accept_dialog(monkeypatch)
    outline = contour(win)
    made = win.vent_pattern_dialog()
    expected = boxgen.vent_pattern(RECT, shape="voronoi", size=10.0, web=2.0,
                                   margin=3.0, seed=7)
    assert made and len(made) == len(expected)
    assert all(int(p.src) == int(outline.src) for p in made)
    vent = win.doc.vents[0]
    assert vent.shape == "voronoi" and int(vent.seed) == 7
    assert win.settings.vent_seed == 7


def test_the_dialog_hands_the_new_options_to_the_pattern(win):
    win.settings.vent_shape = "slots"
    outline = contour(win)
    made = win.apply_vent_pattern({"shape": "slots", "size": 6.0, "web": 2.0, "margin": 3.0,
                                   "seed": 3, "slot_length": 30.0, "angle": 90.0})
    assert made and len(made) == len(boxgen.vent_pattern(RECT, shape="slots", size=6.0,
                                                         web=2.0, margin=3.0, seed=3,
                                                         slot_length=30.0, angle=90.0))
    vent = win.doc.vents[0]
    assert (vent.shape, int(vent.seed), vent.slot_length, vent.angle) == ("slots", 3, 30.0, 90.0)
    assert all(p in made for p in holes_of(win, outline))


def test_a_refit_makes_the_same_grille_again_for_the_new_outline(win):
    outline = contour(win)
    win.apply_vent_pattern(dict(shape="voronoi", size=10.0, web=2.0, margin=3.0, seed=7))
    outline.pts = SQUARE.copy()
    assert win.refit_vents() == 1
    expected = boxgen.vent_pattern(SQUARE, shape="voronoi", size=10.0, web=2.0,
                                   margin=3.0, seed=7)
    assert as_sets([p.pts for p in holes_of(win, outline)]) == as_sets(expected)
    assert int(win.doc.vents[0].seed) == 7


def test_a_refit_keeps_the_slots_as_the_user_set_them(win):
    outline = contour(win)
    win.apply_vent_pattern({"shape": "slots", "size": 6.0, "web": 2.0, "margin": 3.0,
                            "seed": 1, "slot_length": 30.0, "angle": 90.0})
    outline.pts = SQUARE.copy()
    assert win.refit_vents() == 1
    expected = boxgen.vent_pattern(SQUARE, shape="slots", size=6.0, web=2.0, margin=3.0,
                                   slot_length=30.0, angle=90.0)
    assert as_sets([p.pts for p in holes_of(win, outline)]) == as_sets(expected)
    vent = win.doc.vents[0]
    assert vent.slot_length == 30.0 and vent.angle == 90.0


def test_the_new_options_survive_the_file(win, tmp_path):
    contour(win)
    win.apply_vent_pattern(dict(shape="voronoi", size=10.0, web=2.0, margin=3.0, seed=7))
    path = str(tmp_path / "vent_extra.volura")
    fileio.save_document(win.doc, path)
    win.doc.vents = []
    fileio.load_document(win.doc, path)
    vent = win.doc.vents[0]
    assert vent.shape == "voronoi" and int(vent.seed) == 7
    assert vent.slot_length == 0.0 and vent.angle == 0.0


# ------------------------------------------------------------------ linked wall

def hollow_box():
    return mf.Manifold.cube([X, Y, Z]) - mf.Manifold.cube(
        [X - 2 * WALL, Y - 2 * WALL, Z]).translate((WALL, WALL, WALL))


def test_a_voronoi_grille_in_a_linked_wall_cuts_through_it(win):
    g = ops.group_from_manifold(win.doc, hollow_box(), "#3b7dd8", "Box")
    win.doc.notify()
    win.set_mode("3d")
    fid = next(f for f, F in g.ents.F.items()
               if F.normal[1] > 0.999 and F.loops
               and abs(g.ents.V[F.loops[0][0]][1] - Y) < 1e-6)
    win.doc.set_selection([("f", g.id, fid)])
    added = win.send_selection_to_sheet(layer=win.doc.sheet.default_layer("cut"))
    src = int(added[0].src)
    win.set_mode("2d")
    outer = max((p for p in win.doc.sheet.paths if int(p.src) == src and p.closed),
                key=lambda p: abs(polygon_area(p.pts)))
    win.sheet_view.set_selection([outer])
    made = win.apply_vent_pattern({"shape": "voronoi", "size": 8.0, "web": 1.5,
                                   "margin": 2.5, "seed": 4})
    assert made, "the mosaic found room in the wall"
    was = win.doc.group(g.id).ents.to_manifold().volume()
    win.set_mode("3d")
    solid = win.doc.group(g.id).ents.to_manifold()
    assert solid.genus() == len(made), "one handle per cell of the grille"
    cut_out = sum(abs(polygon_area(p.pts)) for p in made) * WALL
    assert was - solid.volume() == pytest.approx(cut_out, rel=0.03), \
        "the wall lost exactly the cells, through its thickness"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
