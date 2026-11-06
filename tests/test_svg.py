"""SVG import and export and the 2D model (without GUI). Run with:  python -m pytest tests"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import svg  # noqa: E402
from volura.vector2d import (Drawing, VPath, arc_points, ellipse_arc_points,  # noqa: E402
                             join_open_paths, polygon_area, regions, svg_arc)

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ')
TAIL = "</svg>"


def doc(body, attrs='width="100mm" height="100mm" viewBox="0 0 100 100"'):
    return svg.read_svg((HEAD + attrs + ">" + body + TAIL).encode())


def pts_of(d, i=0):
    return np.asarray(d.paths[i].pts, dtype=float)


def corners(d, i=0):
    return sorted((round(float(x), 6), round(float(y), 6)) for x, y in pts_of(d, i))


def area(shape):
    return sum(polygon_area(loop) for loop in shape)


def seg_dist(poly, q):
    """Distance from point `q` to the polyline `poly`."""
    a, b = poly[:-1], poly[1:]
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
    t = np.clip(np.sum((q - a) * d, axis=1) / L2, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * d - q, axis=1)))


# ------------------------------------------------------------------ units

def test_rect_millimeters_y_up():
    d = doc('<rect x="10" y="10" width="20" height="20" fill="none" stroke="#ff0000"/>',
            'width="100mm" height="50mm" viewBox="0 0 100 50"')
    assert len(d.paths) == 1
    assert d.paths[0].closed
    assert corners(d) == [(10.0, 20.0), (10.0, 40.0), (30.0, 20.0), (30.0, 40.0)]
    assert d.sheet_w == 100.0 and d.sheet_h == 50.0


def test_viewbox_scaling_and_pixels():
    # 1000 viewBox units on 100 mm: everything a tenth, the height follows the ratio.
    d = doc('<rect x="100" y="100" width="200" height="200" fill="none" stroke="#000"/>',
            'width="100mm" viewBox="0 0 1000 500"')
    assert corners(d) == [(10.0, 20.0), (10.0, 40.0), (30.0, 20.0), (30.0, 40.0)]
    assert d.sheet_w == 100.0 and abs(d.sheet_h - 50.0) < 1e-9
    # 96 px is 25.4 mm at 96 dpi
    d2 = svg.read_svg((HEAD + 'width="96" height="96">'
                       '<rect width="96" height="96" fill="none" stroke="#000"/>'
                       + TAIL).encode())
    assert abs(d2.sheet_w - 25.4) < 1e-9 and abs(d2.sheet_h - 25.4) < 1e-9
    assert corners(d2) == [(0.0, 0.0), (0.0, 25.4), (25.4, 0.0), (25.4, 25.4)]
    # inch and centimetre
    d3 = svg.read_svg((HEAD + 'width="1in" height="2cm">'
                       '<rect width="10" height="10" fill="none" stroke="#000"/>'
                       + TAIL).encode())
    assert abs(d3.sheet_w - 25.4) < 1e-9 and abs(d3.sheet_h - 20.0) < 1e-9


def test_viewbox_only_uses_px():
    d = svg.read_svg((HEAD + 'viewBox="0 0 100 50">'
                      '<rect width="100" height="50" fill="none" stroke="#000"/>'
                      + TAIL).encode())
    assert abs(d.sheet_w - 100 * 25.4 / 96) < 1e-9
    assert abs(d.sheet_h - 50 * 25.4 / 96) < 1e-9


# ------------------------------------------------------------------ curves

def test_circle_from_two_relative_arcs():
    d = doc('<path d="M30 50 a20 20 0 1 0 40 0 a20 20 0 1 0 -40 0 z" fill="#000000"/>')
    assert len(d.paths) == 1 and d.paths[0].closed and d.paths[0].filled
    reg = regions(d.paths)
    assert len(reg) == 1 and len(reg[0]) == 1
    assert abs(abs(area(reg[0])) - math.pi * 20 ** 2) < 0.005 * math.pi * 20 ** 2


def test_cubic_bezier_stays_within_tolerance():
    p0, p1, p2, p3 = (np.array(a, float) for a in ((0, 0), (10, 20), (30, -10), (40, 15)))
    tol = 0.01
    d = svg.read_svg((HEAD + 'width="100mm" height="100mm" viewBox="0 0 100 100">'
                      '<path d="M0 100C10 80 30 110 40 85" fill="none" stroke="#0000ff"/>'
                      + TAIL).encode(), tol=tol)
    poly = pts_of(d)
    t = np.linspace(0.0, 1.0, 2001)[:, None]
    curve = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2
             + t ** 3 * p3)
    worst = max(seg_dist(poly, q) for q in curve)
    assert worst <= tol
    assert len(poly) > 4


def test_a_scaled_down_curve_is_flattened_to_the_tolerance_in_millimetres():
    # a tracer writes its paths ten times too big and scales them down (as potrace does):
    # the tolerance holds for the drawing in mm, not for the units of the path, so the
    # curve gets about as many points as the same curve drawn at its real size
    real = svg.read_svg((HEAD + 'width="100mm" height="100mm" viewBox="0 0 100 100">'
                         '<path d="M0 100C10 80 30 110 40 85" fill="none" stroke="#0000ff"/>'
                         + TAIL).encode(), tol=0.01)
    scaled = svg.read_svg((HEAD + 'width="100mm" height="100mm" viewBox="0 0 100 100">'
                           '<g transform="scale(0.1)"><path d="M0 1000C100 800 300 1100 400 850"'
                           ' fill="none" stroke="#0000ff"/></g>' + TAIL).encode(), tol=0.01)
    assert abs(len(pts_of(scaled)) - len(pts_of(real))) <= 2
    p0, p1, p2, p3 = (np.array(a, float) for a in ((0, 0), (10, 20), (30, -10), (40, 15)))
    t = np.linspace(0.0, 1.0, 2001)[:, None]
    curve = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2
             + t ** 3 * p3)
    assert max(seg_dist(pts_of(scaled), q) for q in curve) <= 0.01


def test_quadratic_and_smooth_commands():
    d = doc('<path d="M0 0 Q10 20 20 0 T40 0" fill="none" stroke="#0000ff"/>')
    p = pts_of(d)
    assert len(p) > 6
    assert np.allclose(p[0], [0, 100]) and np.allclose(p[-1], [40, 100])
    assert abs(p[:, 1].min() - 90.0) < 0.02       # first arc bends 10 mm down
    assert abs(p[:, 1].max() - 110.0) < 0.02      # T mirrors the control point: upward


# ------------------------------------------------------------------ transformations

def test_nested_transforms():
    d = doc('<g transform="rotate(90 50 50)"><g transform="translate(10 0)">'
            '<line x1="60" y1="50" x2="61" y2="50" stroke="#000000"/></g></g>')
    p = pts_of(d)
    assert np.allclose(p[0], [50.0, 30.0], atol=1e-9)
    assert np.allclose(p[1], [50.0, 29.0], atol=1e-9)


def test_matrix_scale_and_skew():
    d = doc('<g transform="matrix(2 0 0 2 5 5)"><line x1="0" y1="0" x2="10" y2="0" '
            'stroke="#000000"/></g>')
    assert np.allclose(pts_of(d), [[5.0, 95.0], [25.0, 95.0]], atol=1e-9)
    d2 = doc('<g transform="skewX(45)"><line x1="0" y1="10" x2="0" y2="20" '
             'stroke="#000000"/></g>')
    assert np.allclose(pts_of(d2), [[10.0, 90.0], [20.0, 80.0]], atol=1e-9)


# ------------------------------------------------------------------ fill rules

def test_evenodd_ring_has_a_hole():
    d = doc('<path fill="#000000" fill-rule="evenodd" d="M50 50 m-30 0 a30 30 0 1 0 60 0 '
            'a30 30 0 1 0 -60 0 z M50 50 m-10 0 a10 10 0 1 0 20 0 a10 10 0 1 0 -20 0 z"/>')
    reg = regions(d.paths)
    assert len(reg) == 1 and len(reg[0]) == 2
    assert abs(abs(area(reg[0])) - math.pi * (900 - 100)) < 0.005 * math.pi * 800


def test_nonzero_overlap_is_union():
    d = doc('<path fill="#ff0000" d="M10 10h20v20h-20z M20 20h20v20h-20z"/>')
    reg = regions(d.paths)
    assert len(reg) == 1 and len(reg[0]) == 1
    assert abs(area(reg[0]) - 700.0) < 1e-6


def test_two_elements_are_unioned_too():
    d = doc('<rect x="0" y="0" width="10" height="10" fill="#ff0000"/>'
            '<rect x="5" y="5" width="10" height="10" fill="#ff0000"/>')
    reg = regions(d.paths)
    assert len(reg) == 1
    assert abs(abs(area(reg[0])) - 175.0) < 1e-6


# ------------------------------------------------------------------ notation

def test_number_notation_and_implicit_repeat():
    d = doc('<path d="M1.5.5L-.5e1,2" fill="none" stroke="#000000"/>')
    assert np.allclose(pts_of(d), [[1.5, 99.5], [-5.0, 98.0]], atol=1e-9)
    d2 = doc('<path d="M0 0 10 0 10 10z" fill="none" stroke="#000000"/>')
    assert d2.paths[0].closed
    assert corners(d2) == [(0.0, 100.0), (10.0, 90.0), (10.0, 100.0)]
    d3 = doc('<path d="M0 0h10v10h-10z" fill="none" stroke="#000000"/>')
    assert corners(d3) == [(0.0, 90.0), (0.0, 100.0), (10.0, 90.0), (10.0, 100.0)]
    d4 = doc('<path d="M1e-3 0L.5e1 1E-2" fill="none" stroke="#000000"/>')
    assert np.allclose(pts_of(d4), [[0.001, 100.0], [5.0, 99.99]], atol=1e-9)


def test_arc_flags_without_separators():
    d = doc('<path d="M30 50a20 20 0 0140 0" fill="none" stroke="#000000"/>')
    p = pts_of(d)
    assert np.allclose(p[0], [30.0, 50.0]) and np.allclose(p[-1], [70.0, 50.0])
    # sweep=1 runs over the top in SVG (small y), so towards large y in our model
    assert abs(p[:, 1].max() - 70.0) < 0.02
    d2 = doc('<path d="M30 50a20 20 0 0 0 40 0" fill="none" stroke="#000000"/>')
    assert abs(pts_of(d2)[:, 1].min() - 30.0) < 0.02
    assert abs(pts_of(d2)[0, 1] - 50.0) < 1e-9


# ------------------------------------------------------------------ layers

def test_inkscape_layers():
    d = doc('<g inkscape:groupmode="layer" inkscape:label="Cut">'
            '<rect width="10" height="10" fill="none" stroke="#ff0000"/></g>'
            '<g inkscape:groupmode="layer" inkscape:label="Engrave" stroke="#00ff00">'
            '<rect x="20" width="10" height="10" fill="none"/></g>')
    assert "Cut" in d.layers and "Engrave" in d.layers
    assert d.paths[0].layer == "Cut" and d.paths[1].layer == "Engrave"
    assert d.layers["Engrave"].color == "#00ff00"


def test_color_layers_without_inkscape():
    d = doc('<rect width="10" height="10" fill="none" stroke="#ff0000"/>'
            '<rect x="20" width="10" height="10" fill="none" stroke="#0000ff"/>'
            '<rect x="40" width="10" height="10" fill="#00ff00" stroke="none"/>')
    assert d.paths[0].layer == "#ff0000" and d.layers["#ff0000"].color == "#ff0000"
    assert d.paths[1].layer == "#0000ff"
    assert d.paths[2].layer == "#00ff00" and d.paths[2].filled
    assert "0" in d.layers


def test_style_attribute_class_and_inheritance():
    d = svg.read_svg((HEAD + 'width="100mm" height="100mm" viewBox="0 0 100 100">'
                      '<style>.cut { stroke: #FF0000; fill: none }</style>'
                      '<g style="stroke:#00ff00">'
                      '<rect class="cut" width="10" height="10"/>'
                      '<rect x="20" width="10" height="10"/>'
                      '<rect x="40" width="10" height="10" style="stroke:#0000ff"/>'
                      '<rect x="60" width="10" height="10" display="none"/>'
                      '<rect x="80" width="10" height="10" visibility="hidden"/></g>'
                      + TAIL).encode())
    assert [p.layer for p in d.paths] == ["#ff0000", "#00ff00", "#0000ff"]


def test_hidden_and_empty_are_skipped():
    d = doc('<rect width="10" height="10" fill="none" stroke="none"/>'
            '<rect x="20" width="10" height="10" fill="none" stroke="#ff0000"/>')
    assert len(d.paths) == 1


# ------------------------------------------------------------------ use and defs

def test_use_with_offset():
    d = doc('<defs><rect id="r" width="10" height="10" fill="none" stroke="#000000"/></defs>'
            '<use xlink:href="#r" x="5" y="5"/>')
    assert len(d.paths) == 1
    assert corners(d) == [(5.0, 85.0), (5.0, 95.0), (15.0, 85.0), (15.0, 95.0)]


def test_defs_are_not_drawn_and_text_warns():
    d = doc('<defs><circle id="c" r="5" fill="#ff0000"/></defs>'
            '<text x="10" y="10">hello</text><image x="0" y="0" width="5" height="5"/>')
    assert d.paths == []
    assert any("text" in w.lower() for w in d.warnings)
    assert any("image" in w.lower() for w in d.warnings)


# ------------------------------------------------------------------ safety

def test_entity_is_refused():
    bad = ('<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY x "M0 0L10 10">]>'
           + HEAD + '><path d="&x;"/>' + TAIL)
    try:
        svg.read_svg(bad.encode())
        assert False, "a file with entities should be refused"
    except ValueError as exc:
        assert "entities" in str(exc).lower()


def test_not_svg():
    try:
        svg.read_svg(b"<html><body>no</body></html>")
        assert False, "not an SVG should give an error"
    except ValueError:
        pass


# ------------------------------------------------------------------ writing

def make_drawing():
    d = Drawing()
    d.ensure_layer("#ff0000", "#ff0000", "cut")
    d.ensure_layer("Engrave", "#00ff00", "engrave")
    outer = ellipse_arc_points((30, 20), 10, 10, 0.0, 2 * math.pi, 0.005)[:-1]
    hole = ellipse_arc_points((30, 20), 3, 3, 0.0, 2 * math.pi, 0.005)[:-1]
    d.add_path(outer, closed=True, layer="#ff0000", filled=True, rule="evenodd", src=1)
    d.add_path(hole, closed=True, layer="#ff0000", filled=True, rule="evenodd", src=1)
    d.add_path(np.array([[5, 5], [15, 5], [15, 12]], float), closed=False, layer="Engrave")
    d.add_path(np.array([[40, 40], [50, 40], [50, 50], [40, 50]], float), closed=True,
               layer="Engrave", src=3)
    return d


def test_round_trip():
    d = make_drawing()
    text = svg.write_svg(d)
    back = svg.read_svg(text.encode())
    assert sorted(back.layers) == sorted(d.layers)
    for name, lay in d.layers.items():
        assert back.layers[name].color == lay.color
        assert back.layers[name].op == lay.op
    assert len(back.paths) == len(d.paths)
    for orig, got in zip(d.paths, back.paths):
        assert got.layer == orig.layer
        assert got.closed == orig.closed
        assert got.filled == orig.filled
        assert got.rule == orig.rule
        assert got.src == orig.src
        assert len(got.pts) == len(orig.pts)
        assert np.max(np.abs(got.pts - orig.pts)) < 1e-3
    # the holes stay a hole
    reg = regions(back.paths)
    ring = [s for s in reg if len(s) == 2]
    assert len(reg) == 2 and len(ring) == 1
    assert abs(abs(area(ring[0])) - math.pi * (100 - 9)) < 0.35


def test_round_trip_with_sheet_and_file():
    d = make_drawing()
    d.sheet_w, d.sheet_h = 200.0, 100.0
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "drawing.svg")
        text = svg.write_svg(d, p)
        assert os.path.exists(p)
        with open(p, encoding="utf-8") as f:
            assert f.read() == text
        back = svg.read_svg(p)
    assert back.sheet_w == 200.0 and back.sheet_h == 100.0
    for orig, got in zip(d.paths, back.paths):
        assert np.max(np.abs(got.pts - orig.pts)) < 1e-3


def test_write_svg_page_is_bbox_plus_margin():
    d = Drawing()
    d.add_path(np.array([[10, 20], [30, 20], [30, 25]], float), closed=True)
    text = svg.write_svg(d)
    assert 'width="22.0000mm"' in text and 'height="7.0000mm"' in text
    assert 'viewBox="9.0000 19.0000 22.0000 7.0000"' in text
    assert 'stroke-width="0.0100"' in text
    assert 'inkscape:groupmode="layer"' in text
    assert 'M10.0000,6.0000L30.0000,6.0000L30.0000,1.0000Z' in text
    back = svg.read_svg(text.encode())
    assert np.max(np.abs(back.paths[0].pts - np.array([[10, 20], [30, 20], [30, 25]], float))) < 1e-4


def test_write_svg_layers_and_fill():
    d = make_drawing()
    d.ensure_layer("Hidden", "#0000ff").visible = False
    d.add_path(np.array([[1, 1], [2, 1]], float), layer="Hidden")
    text = svg.write_svg(d, layers=["Engrave"], fill_layers=("Engrave",))
    assert 'inkscape:label="Engrave"' in text
    assert 'data-volura-op="engrave"' in text
    assert 'inkscape:label="Cut"' not in text and "Hidden" not in text
    assert 'fill="#00ff00"' in text and 'fill-rule="evenodd"' in text
    back = svg.read_svg(text.encode())
    assert [p.layer for p in back.paths] == ["Engrave", "Engrave"]


# ------------------------------------------------------------------ helpers

def test_arc_helpers():
    a = arc_points((0, 0), 30.0, 0.0, math.pi, 0.01)
    assert np.allclose(np.linalg.norm(a, axis=1), 30.0)
    assert np.allclose(a[0], [30, 0]) and np.allclose(a[-1], [-30, 0])
    step = np.max(np.linalg.norm(a[1:] - a[:-1], axis=1))
    assert 30.0 * (1.0 - math.cos(step / 60.0)) < 0.01 + 1e-9
    # radii that are too small are scaled up: the arc still runs from point to point
    p0, p1 = np.array([0.0, 0.0]), np.array([40.0, 0.0])
    arc = svg_arc(p0, 5.0, 5.0, 0.0, 0, 1, p1, 0.01)
    assert np.allclose(arc[-1], [40.0, 0.0])       # points after p0 up to and including p1
    assert np.max(np.abs(arc[:, 1])) > 10.0


def test_rounded_rect_and_ellipse():
    d = doc('<rect x="0" y="0" width="20" height="10" rx="5" ry="5" fill="none" '
            'stroke="#000000"/><ellipse cx="50" cy="20" rx="10" ry="5" fill="none" '
            'stroke="#000000"/>')
    outer = regions([d.paths[0]])
    want = 20 * 10 - (4.0 - math.pi) * 25.0        # minus the four quarter circles
    assert abs(abs(area(outer[0])) - want) < 0.02 * want
    reg = regions([d.paths[1]])
    assert abs(abs(area(reg[0])) - math.pi * 10 * 5) < 0.005 * math.pi * 50


def test_preserve_aspect_ratio_none():
    d = doc('<rect width="100" height="100" fill="none" stroke="#000"/>',
            'width="100mm" height="50mm" viewBox="0 0 100 100" '
            'preserveAspectRatio="none"')
    assert corners(d) == [(0.0, 0.0), (0.0, 50.0), (100.0, 0.0), (100.0, 50.0)]


def test_broken_path_warns_instead_of_crashing():
    d = doc('<path d="M5 5L15 5WAT" stroke="#ff0000" fill="none"/>'
            '<rect width="4" height="4" fill="none" stroke="#00ff00"/>')
    assert len(d.paths) == 1 and d.paths[0].layer == "#00ff00"
    assert any("skipped" in w for w in d.warnings)


def test_copy_is_independent():
    d = make_drawing()
    c = d.copy()
    c.paths[0].pts[0] += 5.0
    c.layers["#ff0000"].color = "#000000"
    assert d.paths[0].pts[0][0] == 40.0
    assert d.layers["#ff0000"].color == "#ff0000"


def test_join_open_paths_makes_a_square():
    segs = [VPath(np.array([[10, 0], [10, 10]], float), False),
            VPath(np.array([[0, 0], [10, 0]], float), False),
            VPath(np.array([[0, 10], [0, 0]], float), False),
            VPath(np.array([[10, 10], [0, 10]], float), False)]
    out = join_open_paths(segs)
    assert len(out) == 1 and out[0].closed
    assert sorted(map(tuple, out[0].pts)) == [(0, 0), (0, 10), (10, 0), (10, 10)]
    assert len(regions(out)) == 1


def test_join_open_paths_keeps_open_and_closed():
    segs = [VPath(np.array([[0, 0], [10, 0]], float), False),
            VPath(np.array([[20, 0], [30, 5]], float), False),
            VPath(np.array([[0, 0], [0, 5], [5, 0]], float), True)]
    out = join_open_paths(segs)
    assert len(out) == 3
    assert sum(p.closed for p in out) == 1


def test_json_round_trip():
    d = make_drawing()
    d.sheet_w, d.sheet_h = 320.0, 200.5
    d.warnings.append("something")
    back = Drawing.from_json(d.to_json())
    assert len(back.paths) == len(d.paths)
    for got, orig in zip(back.paths, d.paths):
        assert (got.closed, got.layer, got.filled, got.rule, got.src) == \
               (orig.closed, orig.layer, orig.filled, orig.rule, orig.src)
        assert np.max(np.abs(got.pts - orig.pts)) <= 1e-6
    assert back.layers == d.layers
    assert back.sheet_w == 320.0 and back.sheet_h == 200.5
    assert back.warnings == ["something"]
    assert abs(back.bounds()[2] - d.bounds()[2]) <= 1e-6


def test_transformed_and_bounds():
    d = make_drawing()
    m = np.array([[0, -1, 100], [1, 0, 0], [0, 0, 1]], float)
    t = d.transformed(m)
    assert t.bounds()[0] > 0
    assert np.allclose(t.paths[2].pts[0], [95.0, 5.0])
    assert d.bounds() == (5.0, 5.0, 50.0, 50.0)
    assert t is not d and t.paths[2] is not d.paths[2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
