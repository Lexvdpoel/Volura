"""Importing SVG/DXF into 3D, placing on a face and extruding.

The pure logic (`volura.vector_import`) can be tested without a window; the
last block checks that the window has the tool and, given a path
(dialog skipped), places something in the document.

Run with:  python -m pytest tests
"""
import math
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import dxf, vector_import as vi  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.svg import read_svg  # noqa: E402
from volura.vector2d import arc_points, polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

GROUND = Plane((0, 0, 0), (0, 0, 1))
SVG_HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" '
            'width="20mm" height="10mm" viewBox="0 0 20 10">')


# ------------------------------------------------------------------ helpers

def svg_rect_with_hole():
    """Rectangle 20x10 mm with a round hole r=2, as one element with the evenodd fill rule."""
    body = ('<path fill="#000000" fill-rule="evenodd" '
            'd="M0 0 H20 V10 H0 Z M10 5 m-2 0 a2 2 0 1 0 4 0 a2 2 0 1 0 -4 0 Z"/>')
    return read_svg((SVG_HEAD + body + "</svg>").encode())


def dxf_rect_with_hole():
    """The same rectangle in DXF: outer edge counterclockwise, hole clockwise."""
    outer = [(0, 0), (20, 0), (20, 10), (0, 10)]
    circle = arc_points((10, 5), 2.0, 2.0 * math.pi, 0.0, 0.005)[:-1]   # clockwise
    hole = [(float(x), float(y)) for x, y in circle]

    def lwp(verts):
        out = ["0", "LWPOLYLINE", "8", "0", "90", str(len(verts)), "70", "1"]
        for x, y in verts:
            out += ["10", repr(x), "20", repr(y), "42", "0"]
        return out

    parts = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
             "0", "SECTION", "2", "ENTITIES"]
    parts += lwp(outer) + lwp(hole)
    parts += ["0", "ENDSEC", "0", "EOF"]
    return dxf.read_dxf("\n".join(parts) + "\n")


def region_area(shape):
    return abs(sum(polygon_area(loop) for loop in shape))


def face_area(ents, fid):
    tri, _ = ents.face_triangles(fid)
    if not len(tri):
        return 0.0
    return float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1).sum() / 2.0)


def box_doc(w=20, d=20, h=10):
    doc = Document()
    doc.root.add_shape(GROUND, [np.array([[0, 0], [w, 0], [w, d], [0, d]], float)])
    f = next(iter(doc.root.F))
    doc.root.pushpull(f, -h)
    return doc


def top_plane(ents):
    fid = max(ents.F, key=lambda f: (ents.F[f].normal[2], ents.V[ents.F[f].loops[0][0]][2]))
    return ents.face_plane(fid)


# ------------------------------------------------------------------ SVG: face with hole

def test_svg_face_with_hole_on_ground():
    drawing = svg_rect_with_hole()
    regions, opens = vi.prepare(drawing)
    assert len(regions) == 1 and len(regions[0]) == 2 and opens == []

    ents = Entities()
    created = vi.place_faces(ents, GROUND, regions, opens, np.zeros(2))
    assert len(ents.F) == 1 and len(created) == 1
    assert len(ents.F[created[0]].loops) == 2

    outer, hole = regions[0]
    expected = abs(polygon_area(outer)) - abs(polygon_area(hole))
    assert abs(face_area(ents, created[0]) - expected) < 1e-6 * 200


# ------------------------------------------------------------------ SVG: extruded volume

def test_svg_extruded_group_volume():
    drawing = svg_rect_with_hole()
    regions, opens = vi.prepare(drawing)
    area = region_area(regions[0])

    doc = Document()
    g = vi.place_extruded(doc, GROUND, regions, np.zeros(2), 5.0, "#ff0000", "Test")
    assert g is not None and len(doc.groups) == 1
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert abs(m.volume() - 5.0 * area) <= 1e-3 * 5.0 * area


# ------------------------------------------------------------------ on the top face of a cube

def test_place_on_top_face_of_cube():
    drawing = svg_rect_with_hole()
    regions, _ = vi.prepare(drawing)

    doc = box_doc()
    pl = top_plane(doc.root)
    assert abs(pl.origin[2] - 10.0) < 1e-6

    ents = Entities()
    vi.place_faces(ents, pl, regions, [], np.zeros(2))
    zs = np.array([ents.V[v][2] for v in ents.V])
    assert np.allclose(zs, 10.0, atol=1e-6)

    g = vi.place_extruded(doc, pl, regions, np.zeros(2), 5.0, "#00ff00", "On")
    zc = np.array([g.ents.V[v][2] for v in g.ents.V])
    assert zc.min() >= 10.0 - 1e-6 and zc.max() <= 15.0 + 1e-6
    assert abs(zc.min() - 10.0) < 1e-6 and abs(zc.max() - 15.0) < 1e-6


# ------------------------------------------------------------------ target width scales proportionally

def test_target_width_scales_proportionally():
    drawing = svg_rect_with_hole()          # 20 wide, 10 high
    regions, _ = vi.prepare(drawing, target_width=50.0)
    pts = np.concatenate(regions[0])
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    assert abs(w - 50.0) < 1e-6
    assert abs(h - 25.0) < 1e-6


# ------------------------------------------------------------------ open path: edges, no faces

def test_open_path_makes_edges_only():
    drawing = read_svg((SVG_HEAD + '<polyline points="0,0 10,0 10,5" '
                        'fill="none" stroke="#000000"/></svg>').encode())
    regions, opens = vi.prepare(drawing)
    assert regions == [] and len(opens) == 1

    ents = Entities()
    created = vi.place_faces(ents, GROUND, regions, opens, np.zeros(2))
    assert created == [] and len(ents.F) == 0
    assert len(ents.E) == 2


# ------------------------------------------------------------------ DXF gives the same result

def test_dxf_same_result_as_svg():
    drawing = dxf_rect_with_hole()
    regions, opens = vi.prepare(drawing)
    assert len(regions) == 1 and len(regions[0]) == 2 and opens == []

    ents = Entities()
    created = vi.place_faces(ents, GROUND, regions, opens, np.zeros(2))
    assert len(ents.F) == 1 and len(ents.F[created[0]].loops) == 2
    outer, hole = regions[0]
    expected = abs(polygon_area(outer)) - abs(polygon_area(hole))
    assert abs(face_area(ents, created[0]) - expected) < 1e-6 * 200

    doc = Document()
    g = vi.place_extruded(doc, GROUND, regions, np.zeros(2), 5.0, "#ff0000", "Dxf")
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert abs(m.volume() - 5.0 * region_area(regions[0])) <= 1e-3 * 5.0 * region_area(regions[0])


# ------------------------------------------------------------------ the window

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def test_window_has_vectorplace_tool(win):
    assert "vectorplace" in win.view.tools
    assert win.view.tools["vectorplace"].name == "vectorplace"


def test_window_import_via_path_places_something(win):
    svg = (SVG_HEAD + '<rect x="0" y="0" width="20" height="10" fill="#000000"/></svg>')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "drawing.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)

        opts = {"scale": 1.0, "target_width": None, "anchor": "center",
                "open_edges": True, "extrude": False, "height": 3.0,
                "color": "#ff0000", "name": "Face"}
        win.import_vector(path, opts)
        assert len(win.doc.root.F) == 1

        opts2 = dict(opts, extrude=True, height=4.0, name="Volume")
        g = win.import_vector(path, opts2)
        assert g is not None and len(win.doc.groups) == 1
        assert abs(g.ents.to_manifold().volume() - 20 * 10 * 4) < 1e-3 * 800


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and not fn.__code__.co_argcount:
            fn()
            print("ok", name)
