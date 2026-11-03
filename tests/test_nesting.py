"""Sheet size and automatic nesting.

`volura/nesting.py` (without Qt) groups paths into parts (`parts_of`) and arranges
them with bottom-left fill as economically as possible over one or more sheets (`nest`), with
rotations, gap and margin. The sheet settings live in the `Drawing` and go
through file and undo; the worksheet warns about paths outside the sheet and the
export can write a separate file per sheet.

Run with:  python -m pytest tests
"""
import math
import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import nesting  # noqa: E402
from volura.vector2d import Drawing  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def circle(cx, cy, r, n=120):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def section(loops):
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def inflate(shape, delta):
    return section([shape]).offset(delta, mf.JoinType.Round, 2.0)


def overlap_area(a, b):
    return mf.CrossSection.batch_boolean([a, b], mf.OpType.Intersect).area()


def placed_shapes(placements):
    """The outline of every part as it lands on its sheet (local coordinates)."""
    out = []
    for (part, si, angle, dx, dy) in placements:
        M = nesting.place_transform(part, angle, dx, dy)
        out.append((si, part.shape @ M[:2, :2].T + M[:2, 2]))
    return out


def assert_within_sheet(shapes, sheet_w, sheet_h, margin):
    for shp in shapes:
        assert shp[:, 0].min() >= margin - 1e-6
        assert shp[:, 1].min() >= margin - 1e-6
        assert shp[:, 0].max() <= sheet_w - margin + 1e-6
        assert shp[:, 1].max() <= sheet_h - margin + 1e-6


def assert_no_overlap(shapes, gap):
    """No two parts overlap, not even after growing them by gap/2 - 1e-6."""
    if gap <= 0:
        return
    grown = [inflate(s, gap / 2.0 - 1e-6) for s in shapes]
    for i in range(len(grown)):
        for j in range(i + 1, len(grown)):
            assert overlap_area(grown[i], grown[j]) <= 1e-6


def grid_rects(d, w, h, cols, rows, ox=1000.0, oy=1000.0, layer="Cut"):
    """`cols` × `rows` rectangles of `w` × `h`, far apart as separate parts."""
    for r in range(rows):
        for c in range(cols):
            d.add_path(rect(w, h, ox + c * (w + 500), oy + r * (h + 500)),
                       closed=True, layer=layer, src=len(d.paths) + 1)


# ------------------------------------------------------------------ sheet settings

def test_sheet_settings_go_through_json():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 400.0, 300.0
    d.sheet_thickness, d.sheet_margin = 6.0, 7.5
    d.sheet_gap, d.sheet_count = 3.0, 4
    back = Drawing.from_json(d.to_json())
    assert back.sheet_w == 400.0 and back.sheet_h == 300.0
    assert back.sheet_thickness == 6.0
    assert back.sheet_margin == 7.5 and back.sheet_gap == 3.0
    assert back.sheet_count == 4


def test_sheet_settings_backward_compatible():
    # an old file only has [width, height] in "sheet"
    old = {"paths": [], "layers": {}, "sheet": [297.0, 210.0]}
    d = Drawing.from_json(old)
    assert d.sheet_w == 297.0 and d.sheet_h == 210.0
    assert d.sheet_margin == 5.0 and d.sheet_gap == 2.0     # defaults
    assert d.sheet_count == 1 and d.sheet_thickness is None


def test_sheet_settings_survive_undo_copy():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_gap, d.sheet_count = 200.0, 100.0, 1.5, 3
    c = d.copy()
    assert (c.sheet_gap, c.sheet_count) == (1.5, 3)
    t = d.transformed(np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float))
    assert (t.sheet_gap, t.sheet_count) == (1.5, 3)


def test_multiple_sheets_lie_side_by_side():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_count = 200.0, 100.0, 3
    origins = nesting.sheet_origins(d)
    assert origins == [(0.0, 0.0), (250.0, 0.0), (500.0, 0.0)]   # i * (200 + 50)


# ------------------------------------------------------------------ nesting

def test_twelve_rectangles_on_1_sheet():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 48.0, 28.0, 4, 3)
    parts = nesting.parts_of(d)
    assert len(parts) == 12
    placements = nesting.nest(parts, 200.0, 100.0, 0.0, 2.0)
    assert len(set(si for (_p, si, *_r) in placements)) == 1
    assert len(placements) == 12


def test_no_overlap_and_inside_the_sheet():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 48.0, 28.0, 4, 3)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 200.0, 100.0, 0.0, 2.0)
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 200.0, 100.0, 0.0)
    assert_no_overlap(shapes, 2.0)


def test_margin_keeps_parts_off_the_edge():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 40.0, 30.0, 3, 2)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 200.0, 100.0, 10.0, 2.0)
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 200.0, 100.0, 10.0)


def test_holes_and_engravings_inside_an_outer_edge_stay_together():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 200.0
    d.add_path(rect(40, 30, 0, 0), closed=True, layer="Cut", src=1)
    d.add_path(circle(20, 15, 5), closed=True, layer="Cut", src=1)     # hole
    d.add_path(circle(10, 8, 2), closed=True, layer="Engrave", src=2)     # engraving
    d.add_path([[6, 6], [14, 24]], layer="Guides")                     # open, inside
    parts = nesting.parts_of(d)
    assert len(parts) == 1
    assert len(parts[0].paths) == 4


def test_too_many_parts_gives_multiple_sheets():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 100.0, 100.0
    for i in range(6):
        d.add_path(rect(60, 60, 1000 + i * 200, 0), closed=True, layer="Cut", src=i + 1)
    parts = nesting.parts_of(d)
    assert len(parts) == 6
    placements = nesting.nest(parts, 100.0, 100.0, 0.0, 2.0)
    assert len(set(si for (_p, si, *_r) in placements)) == 6      # 1 per sheet


def test_part_larger_than_sheet_gives_error():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 50.0, 50.0
    d.add_path(rect(200, 200, 0, 0), closed=True, layer="Cut", src=1)
    parts = nesting.parts_of(d)
    with pytest.raises(ValueError) as e:
        nesting.nest(parts, 50.0, 50.0, 5.0, 2.0)
    assert "does not fit" in str(e.value).lower()


def test_max_sheets_limit_gives_error():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 50.0, 50.0
    for i in range(5):
        d.add_path(rect(40, 40, 1000 + i * 100, 0), closed=True, layer="Cut", src=i + 1)
    parts = nesting.parts_of(d)
    with pytest.raises(ValueError):
        nesting.nest(parts, 50.0, 50.0, 0.0, 2.0, max_sheets=2)


def test_L_shape_with_rotations_fits_on_1_sheet():
    # two L shapes (40 × 40 with a 20 × 20 corner removed) together fill a 60 × 40 rectangle;
    # only by turning one 180° do they get that room on a sheet of 62 × 42.
    L = np.array([[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]], dtype=float)
    gap = 2.0
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 60.0 + gap, 40.0 + gap
    d.add_path(L.copy(), closed=True, layer="Cut", src=1)
    d.add_path(L.copy() + np.array([500.0, 0.0]), closed=True, layer="Cut", src=2)
    parts = nesting.parts_of(d)
    assert len(parts) == 2
    placements = nesting.nest(parts, 60.0 + gap, 40.0 + gap, 0.0, gap,
                              rotations=(0, 90, 180, 270))
    assert len(set(si for (_p, si, *_r) in placements)) == 1
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 60.0 + gap, 40.0 + gap, 0.0)
    assert_no_overlap(shapes, gap)
    assert any(abs(angle - 180.0) < 1e-6 for (_p, _si, angle, _dx, _dy) in placements)


def test_without_rotation_the_L_shape_does_not_fit():
    L = np.array([[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]], dtype=float)
    gap = 2.0
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 60.0 + gap, 40.0 + gap
    d.add_path(L.copy(), closed=True, layer="Cut", src=1)
    d.add_path(L.copy() + np.array([500.0, 0.0]), closed=True, layer="Cut", src=2)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 60.0 + gap, 40.0 + gap, 0.0, gap, rotations=(0,))
    assert len(set(si for (_p, si, *_r) in placements)) == 2      # rotation is needed


def test_utilization():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 100.0, 100.0
    d.add_path(rect(50, 50, 0, 0), closed=True, layer="Cut", src=1)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 100.0, 100.0, 0.0, 2.0)
    assert abs(nesting.utilization(placements, 100.0, 100.0) - 0.25) < 1e-6


def test_50_rectangles_within_15_seconds():
    import random
    rng = random.Random(1234)
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 1000.0, 1000.0
    for i in range(50):
        w, h = rng.uniform(20, 90), rng.uniform(20, 90)
        d.add_path(rect(w, h, 5000 + i * 300, 0), closed=True, layer="Cut", src=i + 1)
    parts = nesting.parts_of(d)
    assert len(parts) == 50
    start = time.time()
    placements = nesting.nest(parts, 1000.0, 1000.0, 5.0, 2.0)
    assert time.time() - start < 15.0
    assert len(placements) == 50
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 1000.0, 1000.0, 5.0)
    assert_no_overlap(shapes, 2.0)


# ------------------------------------------------------------------ outside the sheet

def test_path_outside_the_sheet_is_reported():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_margin = 200.0, 100.0, 5.0
    d.add_path(rect(40, 30, 20, 20), closed=True, layer="Cut", src=1)   # inside
    d.add_path(rect(40, 30, 1, 1), closed=True, layer="Cut", src=2)     # in the margin
    d.add_path(rect(40, 30, 400, 10), closed=True, layer="Cut", src=3)  # beside it
    outside = nesting.outside_paths(d)
    assert {int(p.src) for p in outside} == {2, 3}


def test_no_sheet_gives_no_warning():
    d = Drawing.new_sheet()
    d.add_path(rect(40, 30, 500, 500), closed=True, layer="Cut", src=1)
    assert nesting.outside_paths(d) == []


# ------------------------------------------------------------------ interface and export

@pytest.fixture()
def win():
    from volura.app import MainWindow
    from PySide6.QtCore import QCoreApplication
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def test_auto_arrange_through_the_window(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap, s.sheet_count = 200.0, 100.0, 0.0, 2.0, 1
    grid_rects(s, 48.0, 28.0, 4, 3)
    assert win.sheet_view.auto_arrange({"rotations": (0, 90, 180, 270), "gap": 2.0,
                                        "margin": 0.0, "add_sheets": True})
    assert "12 parts" in win.hint.text()
    assert "utilization" in win.hint.text()
    assert nesting.outside_paths(s) == []                       # everything neatly on the sheet


def test_part_moves_as_a_whole(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap = 200.0, 200.0, 5.0, 2.0
    outer = s.add_path(rect(40, 30, 0, 0), closed=True, layer="Cut", src=1)
    hole = s.add_path(circle(20, 15, 5), closed=True, layer="Cut", src=1)
    engr = s.add_path(circle(10, 8, 2), closed=True, layer="Engrave", src=2)
    before = [np.linalg.norm(p.pts.mean(axis=0) - q.pts.mean(axis=0))
              for p, q in ((outer, hole), (outer, engr), (hole, engr))]
    win.sheet_view.auto_arrange({"rotations": (0, 90, 180, 270), "gap": 2.0,
                                 "margin": 5.0, "add_sheets": True})
    after = [np.linalg.norm(p.pts.mean(axis=0) - q.pts.mean(axis=0))
             for p, q in ((outer, hole), (outer, engr), (hole, engr))]
    assert np.allclose(before, after, atol=1e-6)                # moved rigidly


def test_auto_arrange_larger_than_sheet_reports_without_crash(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap = 50.0, 50.0, 5.0, 2.0
    s.add_path(rect(200, 200, 0, 0), closed=True, layer="Cut", src=1)
    assert not win.sheet_view.auto_arrange({"rotations": (0,), "gap": 2.0,
                                            "margin": 5.0, "add_sheets": True})
    assert "does not fit" in win.hint.text().lower()


def test_export_per_sheet_writes_one_file_per_sheet(win, tmp_path):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap, s.sheet_count = 100.0, 100.0, 0.0, 2.0, 1
    s.add_path(rect(80, 80, 500, 0), closed=True, layer="Cut", src=1)
    s.add_path(rect(80, 80, 900, 0), closed=True, layer="Cut", src=2)
    win.sheet_view.auto_arrange({"rotations": (0,), "gap": 2.0, "margin": 0.0,
                                 "add_sheets": True})
    assert s.sheet_count == 2
    path = str(tmp_path / "cut.svg")
    first = win.export_sheet("svg", path, options={"layers": ["Cut"], "compensate": False,
                                                   "per_layer": False, "per_sheet": True})
    p1 = str(tmp_path / "cut_sheet1.svg")
    p2 = str(tmp_path / "cut_sheet2.svg")
    assert os.path.exists(p1) and os.path.exists(p2)
    assert os.path.basename(first) == "cut_sheet1.svg"
    for one in (p1, p2):
        text = open(one, encoding="utf-8").read()
        assert re.search(r'viewBox="0\.0+ 0\.0+ 100\.0+ 100\.0+"', text)   # sheet as page
        nums = [float(v) for pair in re.findall(r"([0-9.]+),([0-9.]+)", text) for v in pair]
        assert nums and max(nums) <= 100.0 + 1e-6                          # inside the sheet
