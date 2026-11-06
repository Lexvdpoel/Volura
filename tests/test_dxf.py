"""DXF import and export to the 2D model (without GUI).

Run with:  python -m pytest tests
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import dxf  # noqa: E402
from volura.vector2d import (Drawing, arc_points, ellipse_arc_points,  # noqa: E402
                             polygon_area, regions)


def dxf_doc(entities, header="", tables=""):
    """A minimal but valid ASCII DXF around the given ENTITIES."""
    parts = ["0", "SECTION", "2", "HEADER"]
    parts += header.split()
    parts += ["0", "ENDSEC"]
    if tables:
        parts += ["0", "SECTION", "2", "TABLES", tables, "0", "ENDSEC"]
    parts += ["0", "SECTION", "2", "ENTITIES", entities.strip(), "0", "ENDSEC",
              "0", "EOF"]
    return "\n".join(parts) + "\n"


def ent(etype, **codes):
    """One entity from keywords; a list for a code gives several values."""
    lines = ["0", etype]
    for key, val in codes.items():
        code = int(key.split("_")[0])
        vals = val if isinstance(val, (list, tuple)) else [val]
        for v in vals:
            lines += [str(code), str(v)]
    return "\n".join(lines)


def ent_pairs(etype, seq):
    """One entity from an ordered list of (code, value).

    Real DXF writes the coordinates of (LW)POLYLINE and SPLINE interleaved per point
    (10, 20, 42, 10, 20, ...), so that is how we build them too.
    """
    lines = ["0", etype]
    for code, val in seq:
        lines += [str(code), str(val)]
    return "\n".join(lines)


def lwpolyline(layer, verts, bulges=None, closed=True):
    bulges = bulges or [0] * len(verts)
    seq = [(8, layer), (90, len(verts)), (70, 1 if closed else 0)]
    for (x, y), b in zip(verts, bulges):
        seq += [(10, x), (20, y), (42, b)]
    return ent_pairs("LWPOLYLINE", seq)


def corners(d, i=0):
    return sorted((round(float(x), 6), round(float(y), 6)) for x, y in d.paths[i].pts)


def area(shape):
    return sum(polygon_area(loop) for loop in shape)


def radius_of(pts):
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    return float(np.mean(np.linalg.norm(p - c, axis=1)))


# ------------------------------------------------------------------ arc from bulge

def test_lwpolyline_bulge_half_circle():
    e = lwpolyline("0", [(0, 0), (10, 0)], bulges=[1, 0], closed=True)
    d = dxf.read_dxf(dxf_doc(e))
    assert len(d.paths) == 1 and d.paths[0].closed
    reg = regions(d.paths)
    got = abs(area(reg[0]))
    assert abs(got - math.pi * 25 / 2) < 0.005 * (math.pi * 25 / 2)


# ------------------------------------------------------------------ units

def test_insunits_inch():
    header = "9 $INSUNITS 70 1"
    e = ent("LINE", **{"8": "0", "10": 0, "20": 0, "11": 1, "21": 0})
    d = dxf.read_dxf(dxf_doc(e, header=header))
    p = np.asarray(d.paths[0].pts, float)
    assert abs(np.linalg.norm(p[1] - p[0]) - 25.4) < 1e-6


# ------------------------------------------------------------------ blocks and INSERT

def test_block_insert_rotation_scale():
    square = lwpolyline("0", [(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)
    blocks = "\n".join(["0", "BLOCK", "8", "0", "2", "SQUARE", "10", "0", "20", "0",
                        "30", "0", square, "0", "ENDBLK"])
    ins_a = ent("INSERT", **{"8": "0", "2": "SQUARE"})
    ins_b = ent("INSERT", **{"8": "0", "2": "SQUARE", "10": 100, "20": 0,
                             "41": 2, "42": 2, "50": 90})
    full = dxf_doc(ins_a + "\n" + ins_b)
    full = full.replace("2\nENTITIES",
                        "2\nBLOCKS\n" + blocks + "\n0\nENDSEC\n0\nSECTION\n2\nENTITIES")
    d = dxf.read_dxf(full)
    assert len(d.paths) == 2
    assert corners(d, 0) == [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
    assert corners(d, 1) == [(80.0, 0.0), (80.0, 20.0), (100.0, 0.0), (100.0, 20.0)]


# ------------------------------------------------------------------ extrusion

def test_circle_extrusion_mirrors_x():
    e = ent("CIRCLE", **{"8": "0", "10": 10, "20": 5, "40": 3,
                         "210": 0, "220": 0, "230": -1})
    d = dxf.read_dxf(dxf_doc(e))
    c = np.asarray(d.paths[0].pts, float).mean(axis=0)
    assert abs(c[0] + 10.0) < 1e-3 and abs(c[1] - 5.0) < 1e-3
    assert abs(radius_of(d.paths[0].pts) - 3.0) < 1e-2


# ------------------------------------------------------------------ ellipse

def test_ellipse_area():
    e = ent("ELLIPSE", **{"8": "0", "10": 0, "20": 0, "11": 10, "21": 0,
                          "40": 0.5, "41": 0.0, "42": 6.283185307179586})
    d = dxf.read_dxf(dxf_doc(e))
    assert d.paths[0].closed
    reg = regions(d.paths)
    got = abs(area(reg[0]))
    assert abs(got - math.pi * 10 * 5) < 0.005 * math.pi * 50


# ------------------------------------------------------------------ spline

def test_clamped_cubic_spline_endpoints():
    seq = [(8, "0"), (71, 3)]
    seq += [(40, k) for k in (0, 0, 0, 0, 1, 1, 1, 1)]
    for x, y in ((0, 0), (10, 10), (20, 0), (30, 15)):
        seq += [(10, x), (20, y)]
    d = dxf.read_dxf(dxf_doc(ent_pairs("SPLINE", seq)))
    p = np.asarray(d.paths[0].pts, float)
    assert np.allclose(p[0], [0, 0], atol=1e-6)
    assert np.allclose(p[-1], [30, 15], atol=1e-6)


def test_rational_quarter_circle_spline():
    w = math.cos(math.pi / 4)
    seq = [(8, "0"), (71, 2)]
    seq += [(40, k) for k in (0, 0, 0, 1, 1, 1)]
    seq += [(42, wt) for wt in (1, w, 1)]
    for x, y in ((1, 0), (1, 1), (0, 1)):
        seq += [(10, x), (20, y)]
    d = dxf.read_dxf(dxf_doc(ent_pairs("SPLINE", seq)))
    p = np.asarray(d.paths[0].pts, float)
    dev = np.max(np.abs(np.linalg.norm(p, axis=1) - 1.0))
    assert dev < 0.01


# ------------------------------------------------------------------ joining

def test_four_lines_join_to_square():
    lines = []
    for (x0, y0, x1, y1) in ((0, 0, 10, 0), (10, 0, 10, 10), (10, 10, 0, 10), (0, 10, 0, 0)):
        lines.append(ent("LINE", **{"8": "0", "10": x0, "20": y0, "11": x1, "21": y1}))
    d = dxf.read_dxf(dxf_doc("\n".join(lines)))
    assert len(d.paths) == 1 and d.paths[0].closed
    assert corners(d) == [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
    assert len(regions(d.paths)) == 1


# ------------------------------------------------------------------ layers and colors

def test_layers_and_aci_colors_bylayer():
    tables = ("0\nTABLE\n2\nLAYER\n70\n3\n"
              "0\nLAYER\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n"
              "0\nLAYER\n2\nCUT\n70\n0\n62\n1\n6\nCONTINUOUS\n"
              "0\nLAYER\n2\nGRAV\n70\n0\n62\n3\n6\nCONTINUOUS\n"
              "0\nENDTAB")
    e1 = ent("LINE", **{"8": "CUT", "10": 0, "20": 0, "11": 10, "21": 0})
    e2 = ent("LINE", **{"8": "GRAV", "10": 0, "20": 5, "11": 10, "21": 5})
    d = dxf.read_dxf(dxf_doc(e1 + "\n" + e2, tables=tables))
    assert d.layers["CUT"].color == "#ff0000"
    assert d.layers["GRAV"].color == "#00ff00"
    assert d.paths[0].layer == "CUT" and d.paths[1].layer == "GRAV"


def test_true_color_and_explicit_aci():
    e = ent("LINE", **{"8": "COLOR", "10": 0, "20": 0, "11": 5, "21": 0, "420": 0x0000ff})
    d = dxf.read_dxf(dxf_doc(e))
    assert d.layers["COLOR"].color == "#0000ff"
    assert d.paths[0].layer == "COLOR"


# ------------------------------------------------------------------ binary DXF

def test_binary_dxf_refused():
    data = b"AutoCAD Binary DXF\r\n\x00\x00\x01\x00"
    try:
        dxf.read_dxf(data)
        assert False, "binary DXF should be refused"
    except ValueError as exc:
        assert "binary" in str(exc).lower()


# ------------------------------------------------------------------ writing

def make_drawing():
    d = Drawing()
    d.ensure_layer("0", "#000000")
    d.ensure_layer("Red", "#ff0000", "cut")
    d.ensure_layer("Green", "#00ff00", "engrave")
    d.add_path(np.array([[0, 0], [20, 0], [20, 15], [0, 15]], float), closed=True,
               layer="Red")
    circle = arc_points((50, 30), 8.0, 0.0, 2 * math.pi, 0.005)[:-1]
    d.add_path(circle, closed=True, layer="Red")
    d.add_path(np.array([[0, 40], [30, 40]], float), closed=False, layer="Green")
    return d


def test_round_trip_contours_and_layers():
    d = make_drawing()
    text = dxf.write_dxf(d)
    back = dxf.read_dxf(text)
    assert sorted(back.layers) == sorted(d.layers)
    assert back.layers["Red"].color == "#ff0000"
    assert back.layers["Green"].color == "#00ff00"
    # the square stays a square with the same corners
    sq = [p for p in back.paths if p.closed and len(p.pts) == 4]
    assert len(sq) == 1
    assert sorted((round(float(x), 4), round(float(y), 4)) for x, y in sq[0].pts) == \
        [(0.0, 0.0), (0.0, 15.0), (20.0, 0.0), (20.0, 15.0)]
    # the open line stays open
    op = [p for p in back.paths if not p.closed]
    assert len(op) == 1 and np.max(np.abs(op[0].pts - np.array([[0, 40], [30, 40]], float))) < 1e-4


def test_circle_detected_as_circle_entity():
    d = Drawing()
    d.ensure_layer("0", "#000000")
    circle = ellipse_arc_points((10, 10), 5, 5, 0.0, 2 * math.pi, 0.005)[:-1]
    assert len(circle) >= 16
    d.add_path(circle, closed=True, layer="0")
    text = dxf.write_dxf(d)
    assert "\nCIRCLE\n" in text
    back = dxf.read_dxf(text)
    circ = [p for p in back.paths if p.closed]
    assert len(circ) == 1
    assert abs(radius_of(circ[0].pts) - 5.0) < 0.01
    c = np.asarray(circ[0].pts, float).mean(axis=0)
    assert abs(c[0] - 10.0) < 0.01 and abs(c[1] - 10.0) < 0.01


def test_write_skips_hidden_and_filters_layers():
    d = make_drawing()
    d.ensure_layer("Hidden", "#0000ff").visible = False
    d.add_path(np.array([[1, 1], [2, 1]], float), layer="Hidden")
    text = dxf.write_dxf(d, layers=["Green"])
    assert "Hidden" not in text and "Red" not in text
    assert "Green" in text
    assert text.rstrip().endswith("EOF")


def test_write_header_and_format():
    text = dxf.write_dxf(make_drawing())
    assert "AC1009" in text
    assert "$INSUNITS" in text and "$MEASUREMENT" in text
    assert "\nPOLYLINE\n" in text and "\nVERTEX\n" in text and "\nSEQEND\n" in text
    assert "\nLINE\n" in text


def test_an_empty_value_keeps_the_pairs_in_step():
    """An empty text value (a custom property without a value in the header, as some CAD
    programs write) is a line of its own: it must not shift every later pair."""
    header = "\n".join(["9", "$CUSTOMPROPERTYTAG", "1", "Manager",
                        "9", "$CUSTOMPROPERTY", "1", "",
                        "9", "$INSUNITS", "70", "4"])
    text = dxf_doc(ent("LINE", **{"8": "0", "10": 0, "20": 0, "11": 40, "21": 0}) + "\n"
                   + ent("LINE", **{"8": "0", "10": 40, "20": 0, "11": 40, "21": 25}), header="")
    text = text.replace("HEADER\n", "HEADER\n" + header + "\n", 1)
    pairs = dxf._pairs(text)
    assert (1, "") in pairs
    assert (70, "4") in pairs
    d = dxf.read_dxf(text)
    assert len(d.paths) == 1                       # the two lines chained into one path
    assert np.allclose(d.paths[0].pts, [[0, 0], [40, 0], [40, 25]])


def test_a_stray_blank_line_where_a_code_belongs_is_skipped():
    text = dxf_doc(ent("LINE", **{"8": "0", "10": 0, "20": 0, "11": 10, "21": 0}))
    text = text.replace("ENTITIES\n", "ENTITIES\n\n", 1)
    d = dxf.read_dxf(text)
    assert len(d.paths) == 1
    assert np.allclose(d.paths[0].pts, [[0, 0], [10, 0]])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
