"""Painted faces in the multi-color 3MF: slicers read them as their own color painting."""
import os
import re
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402

from volura import fileio, ops  # noqa: E402
from volura.model import Document  # noqa: E402

MATERIALS = [("White", "#f2f2f2"), ("Black", "#2b2b2b"), ("Red", "#d23b3b"), ("Blue", "#2f6fd6")]


def triangles(path):
    model = zipfile.ZipFile(path).read("3D/3dmodel.model").decode()
    assert 'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06"' in model
    return re.findall(r"<triangle [^>]*>", model)


def painted(tris):
    codes = {}
    for t in tris:
        m = re.search(r'paint_color="([0-9A-F]+)"', t)
        if m:
            assert f'slic3rpe:mmu_segmentation="{m.group(1)}"' in t     # PrusaSlicer too
            codes[m.group(1)] = codes.get(m.group(1), 0) + 1
    return codes


def block(doc, top=None, front=None, color="#f2f2f2"):
    g = ops.group_from_manifold(doc, mf.Manifold.cube((40, 30, 20)), color, "Block")
    for f, face in g.ents.F.items():
        if top and face.normal[2] > 0.9:
            g.ents.set_face_color(f, top)
        if front and face.normal[1] < -0.9:
            g.ents.set_face_color(f, front)
    return g


@pytest.mark.parametrize("extruder, code", ((1, "4"), (2, "8"), (3, "0C"), (4, "1C"), (5, "2C"),
                                            (16, "DC"), (0, ""), (40, "")))
def test_paint_code_is_the_slicers_brush_code(extruder, code):
    assert fileio.paint_code(extruder) == code


def test_painted_faces_become_painted_triangles(tmp_path):
    doc = Document()
    doc.materials = list(MATERIALS)
    block(doc, top="#d23b3b", front="#2f6fd6")
    path = tmp_path / "paint.3mf"
    used = fileio.export_3mf(str(path), ops.parts(doc), doc.materials)
    tris = triangles(path)
    assert len(tris) == 12
    # only the used colors are numbered, in list order: white 1, red 2, blue 3
    assert painted(tris) == {"8": 2, "0C": 2}
    assert used == [("White", 1), ("Red", 2), ("Blue", 3)]


def test_unpainted_model_has_no_paint(tmp_path):
    doc = Document()
    doc.materials = list(MATERIALS)
    block(doc)
    path = tmp_path / "plain.3mf"
    fileio.export_3mf(str(path), ops.parts(doc), doc.materials)
    assert painted(triangles(path)) == {}


def test_black_paint_is_not_lost(tmp_path):
    """#000000 must not be mistaken for "no color"."""
    doc = Document()
    doc.materials = [("White", "#f2f2f2"), ("Pure black", "#000000")]
    block(doc, top="#000000")
    path = tmp_path / "black.3mf"
    fileio.export_3mf(str(path), ops.parts(doc), doc.materials)
    assert painted(triangles(path)) == {"8": 2}


def test_face_color_survives_the_color_split(tmp_path):
    """Overlapping parts are cut apart by color; painted faces keep their color."""
    doc = Document()
    doc.materials = list(MATERIALS)
    block(doc, top="#d23b3b")
    ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)).translate((35, 25, 15)), "#2f6fd6", "Corner")
    path = tmp_path / "split.3mf"
    fileio.export_3mf(str(path), ops.parts(doc), doc.materials)
    codes = painted(triangles(path))
    assert codes.get("8", 0) >= 2                         # the red top (filament 2), with a notch
    assert ops.id_color(ops.color_id("#000000")) == "#000000"
    assert ops.id_color(0) is None


def test_only_used_colors_get_a_filament(tmp_path):
    """Yellow (5th in the list) with black and red painted faces: filaments 1, 2 and 3, so the
    file fits a slicer with four filaments."""
    doc = Document()
    doc.materials = [("White", "#f2f2f2"), ("Black", "#2b2b2b"), ("Red", "#d23b3b"),
                     ("Blue", "#2f6fd6"), ("Yellow", "#f2c230")]
    block(doc, top="#d23b3b", front="#2b2b2b", color="#f2c230")
    path = tmp_path / "three.3mf"
    used = fileio.export_3mf(str(path), ops.parts(doc), doc.materials)
    assert used == [("Black", 1), ("Red", 2), ("Yellow", 3)]
    assert painted(triangles(path)) == {"4": 2, "8": 2}
    config = zipfile.ZipFile(path).read("Metadata/model_settings.config").decode()
    assert '<metadata key="extruder" value="3"/>' in config
    assert 'value="5"' not in config
