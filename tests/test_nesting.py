"""Plaatformaat en automatisch nesten.

`volura/nesting.py` (zonder Qt) groepeert paden in onderdelen (`parts_of`) en schikt
ze met bottom-left fill zo zuinig mogelijk over één of meer platen (`nest`), met
rotaties, tussenruimte en marge. De plaatinstellingen leven in de `Drawing` en gaan
mee door bestand en undo; het werkblad waarschuwt voor paden buiten de plaat en de
export kan per plaat een apart bestand schrijven.

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
    """De omtrek van elk onderdeel zoals het op zijn plaat belandt (lokale coordinaten)."""
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
    """Twee onderdelen overlappen niet, ook niet na opblazen met gap/2 - 1e-6."""
    if gap <= 0:
        return
    grown = [inflate(s, gap / 2.0 - 1e-6) for s in shapes]
    for i in range(len(grown)):
        for j in range(i + 1, len(grown)):
            assert overlap_area(grown[i], grown[j]) <= 1e-6


def grid_rects(d, w, h, cols, rows, ox=1000.0, oy=1000.0, layer="Snijden"):
    """`cols` × `rows` rechthoeken van `w` × `h`, ver uit elkaar als losse onderdelen."""
    for r in range(rows):
        for c in range(cols):
            d.add_path(rect(w, h, ox + c * (w + 500), oy + r * (h + 500)),
                       closed=True, layer=layer, src=len(d.paths) + 1)


# ------------------------------------------------------------------ plaatinstellingen

def test_plaatinstellingen_gaan_mee_door_json():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 400.0, 300.0
    d.sheet_thickness, d.sheet_margin = 6.0, 7.5
    d.sheet_gap, d.sheet_count = 3.0, 4
    back = Drawing.from_json(d.to_json())
    assert back.sheet_w == 400.0 and back.sheet_h == 300.0
    assert back.sheet_thickness == 6.0
    assert back.sheet_margin == 7.5 and back.sheet_gap == 3.0
    assert back.sheet_count == 4


def test_plaatinstellingen_achterwaarts_compatibel():
    # een oud bestand kent alleen [breedte, hoogte] in "sheet"
    old = {"paths": [], "layers": {}, "sheet": [297.0, 210.0]}
    d = Drawing.from_json(old)
    assert d.sheet_w == 297.0 and d.sheet_h == 210.0
    assert d.sheet_margin == 5.0 and d.sheet_gap == 2.0     # standaarden
    assert d.sheet_count == 1 and d.sheet_thickness is None


def test_plaatinstellingen_overleven_undo_kopie():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_gap, d.sheet_count = 200.0, 100.0, 1.5, 3
    c = d.copy()
    assert (c.sheet_gap, c.sheet_count) == (1.5, 3)
    t = d.transformed(np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float))
    assert (t.sheet_gap, t.sheet_count) == (1.5, 3)


def test_meerdere_platen_liggen_naast_elkaar():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_count = 200.0, 100.0, 3
    origins = nesting.sheet_origins(d)
    assert origins == [(0.0, 0.0), (250.0, 0.0), (500.0, 0.0)]   # i * (200 + 50)


# ------------------------------------------------------------------ nesten

def test_twaalf_rechthoeken_op_1_plaat():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 48.0, 28.0, 4, 3)
    parts = nesting.parts_of(d)
    assert len(parts) == 12
    placements = nesting.nest(parts, 200.0, 100.0, 0.0, 2.0)
    assert len(set(si for (_p, si, *_r) in placements)) == 1
    assert len(placements) == 12


def test_geen_overlap_en_binnen_de_plaat():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 48.0, 28.0, 4, 3)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 200.0, 100.0, 0.0, 2.0)
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 200.0, 100.0, 0.0)
    assert_no_overlap(shapes, 2.0)


def test_marge_houdt_onderdelen_van_de_rand():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    grid_rects(d, 40.0, 30.0, 3, 2)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 200.0, 100.0, 10.0, 2.0)
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 200.0, 100.0, 10.0)


def test_gaten_en_gravures_binnen_een_buitenrand_bij_elkaar():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 200.0, 200.0
    d.add_path(rect(40, 30, 0, 0), closed=True, layer="Snijden", src=1)
    d.add_path(circle(20, 15, 5), closed=True, layer="Snijden", src=1)     # gat
    d.add_path(circle(10, 8, 2), closed=True, layer="Graveren", src=2)     # gravure
    d.add_path([[6, 6], [14, 24]], layer="Hulplijnen")                     # open, binnen
    parts = nesting.parts_of(d)
    assert len(parts) == 1
    assert len(parts[0].paths) == 4


def test_te_veel_onderdelen_geeft_meerdere_platen():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 100.0, 100.0
    for i in range(6):
        d.add_path(rect(60, 60, 1000 + i * 200, 0), closed=True, layer="Snijden", src=i + 1)
    parts = nesting.parts_of(d)
    assert len(parts) == 6
    placements = nesting.nest(parts, 100.0, 100.0, 0.0, 2.0)
    assert len(set(si for (_p, si, *_r) in placements)) == 6      # 1 per plaat


def test_onderdeel_groter_dan_plaat_geeft_fout():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 50.0, 50.0
    d.add_path(rect(200, 200, 0, 0), closed=True, layer="Snijden", src=1)
    parts = nesting.parts_of(d)
    with pytest.raises(ValueError) as e:
        nesting.nest(parts, 50.0, 50.0, 5.0, 2.0)
    assert "past niet" in str(e.value).lower()


def test_max_platen_limiet_geeft_fout():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 50.0, 50.0
    for i in range(5):
        d.add_path(rect(40, 40, 1000 + i * 100, 0), closed=True, layer="Snijden", src=i + 1)
    parts = nesting.parts_of(d)
    with pytest.raises(ValueError):
        nesting.nest(parts, 50.0, 50.0, 0.0, 2.0, max_sheets=2)


def test_L_vorm_met_rotaties_past_op_1_plaat():
    # twee L-vormen (40 × 40 met een 20 × 20 hoek weg) vullen samen een 60 × 40 rechthoek;
    # alleen door er een 180° te draaien krijgen ze die ruimte op een plaat van 62 × 42.
    L = np.array([[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]], dtype=float)
    gap = 2.0
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 60.0 + gap, 40.0 + gap
    d.add_path(L.copy(), closed=True, layer="Snijden", src=1)
    d.add_path(L.copy() + np.array([500.0, 0.0]), closed=True, layer="Snijden", src=2)
    parts = nesting.parts_of(d)
    assert len(parts) == 2
    placements = nesting.nest(parts, 60.0 + gap, 40.0 + gap, 0.0, gap,
                              rotations=(0, 90, 180, 270))
    assert len(set(si for (_p, si, *_r) in placements)) == 1
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 60.0 + gap, 40.0 + gap, 0.0)
    assert_no_overlap(shapes, gap)
    assert any(abs(angle - 180.0) < 1e-6 for (_p, _si, angle, _dx, _dy) in placements)


def test_zonder_rotatie_lukt_de_L_vorm_niet():
    L = np.array([[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]], dtype=float)
    gap = 2.0
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 60.0 + gap, 40.0 + gap
    d.add_path(L.copy(), closed=True, layer="Snijden", src=1)
    d.add_path(L.copy() + np.array([500.0, 0.0]), closed=True, layer="Snijden", src=2)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 60.0 + gap, 40.0 + gap, 0.0, gap, rotations=(0,))
    assert len(set(si for (_p, si, *_r) in placements)) == 2      # rotatie is nodig


def test_benuttingsgraad():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 100.0, 100.0
    d.add_path(rect(50, 50, 0, 0), closed=True, layer="Snijden", src=1)
    parts = nesting.parts_of(d)
    placements = nesting.nest(parts, 100.0, 100.0, 0.0, 2.0)
    assert abs(nesting.utilization(placements, 100.0, 100.0) - 0.25) < 1e-6


def test_50_rechthoeken_binnen_15_seconde():
    import random
    rng = random.Random(1234)
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h = 1000.0, 1000.0
    for i in range(50):
        w, h = rng.uniform(20, 90), rng.uniform(20, 90)
        d.add_path(rect(w, h, 5000 + i * 300, 0), closed=True, layer="Snijden", src=i + 1)
    parts = nesting.parts_of(d)
    assert len(parts) == 50
    start = time.time()
    placements = nesting.nest(parts, 1000.0, 1000.0, 5.0, 2.0)
    assert time.time() - start < 15.0
    assert len(placements) == 50
    shapes = [s for (_si, s) in placed_shapes(placements)]
    assert_within_sheet(shapes, 1000.0, 1000.0, 5.0)
    assert_no_overlap(shapes, 2.0)


# ------------------------------------------------------------------ buiten de plaat

def test_pad_buiten_de_plaat_wordt_gemeld():
    d = Drawing.new_sheet()
    d.sheet_w, d.sheet_h, d.sheet_margin = 200.0, 100.0, 5.0
    d.add_path(rect(40, 30, 20, 20), closed=True, layer="Snijden", src=1)   # binnen
    d.add_path(rect(40, 30, 1, 1), closed=True, layer="Snijden", src=2)     # in de marge
    d.add_path(rect(40, 30, 400, 10), closed=True, layer="Snijden", src=3)  # ernaast
    outside = nesting.outside_paths(d)
    assert {int(p.src) for p in outside} == {2, 3}


def test_geen_plaat_geeft_geen_waarschuwing():
    d = Drawing.new_sheet()
    d.add_path(rect(40, 30, 500, 500), closed=True, layer="Snijden", src=1)
    assert nesting.outside_paths(d) == []


# ------------------------------------------------------------------ interface en export

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


def test_auto_schikken_via_het_venster(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap, s.sheet_count = 200.0, 100.0, 0.0, 2.0, 1
    grid_rects(s, 48.0, 28.0, 4, 3)
    assert win.sheet_view.auto_arrange({"rotations": (0, 90, 180, 270), "gap": 2.0,
                                        "margin": 0.0, "add_sheets": True})
    assert "12 onderdelen" in win.hint.text()
    assert "benutting" in win.hint.text()
    assert nesting.outside_paths(s) == []                       # alles netjes op de plaat


def test_onderdeel_beweegt_als_geheel(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap = 200.0, 200.0, 5.0, 2.0
    outer = s.add_path(rect(40, 30, 0, 0), closed=True, layer="Snijden", src=1)
    hole = s.add_path(circle(20, 15, 5), closed=True, layer="Snijden", src=1)
    engr = s.add_path(circle(10, 8, 2), closed=True, layer="Graveren", src=2)
    before = [np.linalg.norm(p.pts.mean(axis=0) - q.pts.mean(axis=0))
              for p, q in ((outer, hole), (outer, engr), (hole, engr))]
    win.sheet_view.auto_arrange({"rotations": (0, 90, 180, 270), "gap": 2.0,
                                 "margin": 5.0, "add_sheets": True})
    after = [np.linalg.norm(p.pts.mean(axis=0) - q.pts.mean(axis=0))
             for p, q in ((outer, hole), (outer, engr), (hole, engr))]
    assert np.allclose(before, after, atol=1e-6)                # star verplaatst


def test_auto_schikken_groter_dan_plaat_meldt_zonder_crash(win):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap = 50.0, 50.0, 5.0, 2.0
    s.add_path(rect(200, 200, 0, 0), closed=True, layer="Snijden", src=1)
    assert not win.sheet_view.auto_arrange({"rotations": (0,), "gap": 2.0,
                                            "margin": 5.0, "add_sheets": True})
    assert "past niet" in win.hint.text().lower()


def test_export_per_plaat_schrijft_een_bestand_per_plaat(win, tmp_path):
    win.set_mode("2d")
    s = win.doc.sheet
    s.sheet_w, s.sheet_h, s.sheet_margin, s.sheet_gap, s.sheet_count = 100.0, 100.0, 0.0, 2.0, 1
    s.add_path(rect(80, 80, 500, 0), closed=True, layer="Snijden", src=1)
    s.add_path(rect(80, 80, 900, 0), closed=True, layer="Snijden", src=2)
    win.sheet_view.auto_arrange({"rotations": (0,), "gap": 2.0, "margin": 0.0,
                                 "add_sheets": True})
    assert s.sheet_count == 2
    path = str(tmp_path / "snede.svg")
    first = win.export_sheet("svg", path, options={"layers": ["Snijden"], "compensate": False,
                                                   "per_layer": False, "per_sheet": True})
    p1 = str(tmp_path / "snede_plaat1.svg")
    p2 = str(tmp_path / "snede_plaat2.svg")
    assert os.path.exists(p1) and os.path.exists(p2)
    assert os.path.basename(first) == "snede_plaat1.svg"
    for one in (p1, p2):
        text = open(one, encoding="utf-8").read()
        assert re.search(r'viewBox="0\.0+ 0\.0+ 100\.0+ 100\.0+"', text)   # plaat als pagina
        nums = [float(v) for pair in re.findall(r"([0-9.]+),([0-9.]+)", text) for v in pair]
        assert nums and max(nums) <= 100.0 + 1e-6                          # binnen de plaat
