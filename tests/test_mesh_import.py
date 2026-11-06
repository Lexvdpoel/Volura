"""Importing 3D models: STL, OBJ, 3MF, PLY, OFF, AMF, glTF/GLB and COLLADA.

Every format is written here by hand (a cube, mostly as quads, the way other programs
write them) and read back through `volura.mesh_formats` and `fileio.import_mesh_report`;
the last block imports each one through the window, as File > Import does, and checks
the progress bar and the placing of a big drawing in the background.

Run with:  python -m pytest tests
"""
import base64
import json
import os
import struct
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, background, fileio, mesh_formats, vector_import  # noqa: E402
from volura.geom import Plane  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

# a cube of corners x + 2y + 4z, its faces counter-clockwise seen from outside
CORNERS = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
TRIS = [t for q in QUADS for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]


def cube(size=10.0, at=(0.0, 0.0, 0.0)):
    return CORNERS * size + np.asarray(at, dtype=float)


def volumes(path):
    vols, notes = fileio.import_mesh_report(str(path))
    return vols, notes


def bounds(vol):
    lo, hi = vol.manifold.bounding_box()[:3], vol.manifold.bounding_box()[3:]
    return np.round(lo, 6), np.round(hi, 6)


# ------------------------------------------------------------------ the formats

def test_every_extension_is_known_also_in_upper_case():
    for ext in (".stl", ".obj", ".3mf", ".ply", ".off", ".amf", ".glb", ".gltf", ".dae"):
        assert mesh_formats.is_mesh("model" + ext) and mesh_formats.is_mesh("MODEL" + ext.upper())
    assert not mesh_formats.is_mesh("drawing.svg")
    with pytest.raises(ValueError, match="cannot import"):
        mesh_formats.read_mesh("model.xyz")


def test_stl(tmp_path):
    path = tmp_path / "cube.stl"
    fileio.write_stl(str(path), mf.Manifold.cube((10.0, 10.0, 10.0)))
    (vol,), notes = volumes(path)
    assert vol.name == "cube" and abs(vol.manifold.volume() - 1000) < 1e-6 and not notes


def write_obj(path, objects, mtl=None):
    lines = [f"mtllib {os.path.basename(mtl)}"] if mtl else []
    for name, verts, material in objects:
        lines.append(f"o {name}")
        lines += [f"v {x} {y} {z}" for x, y, z in verts]
        lines += ["vt 0 0", "vn 0 0 1"]
        if material:
            lines.append(f"usemtl {material}")
        n = len(verts)
        for q in QUADS:              # relative (negative) indices with texture and normal
            lines.append("f " + " ".join(f"{k - n}/-1/-1" for k in q))
    path.write_text("\n".join(lines) + "\n")


def test_obj_objects_become_parts_with_their_material_color(tmp_path):
    mtl = tmp_path / "parts.mtl"
    mtl.write_text("newmtl Red\nKd 1 0 0\nnewmtl Blue\nKd 0 0 1.0\n")
    path = tmp_path / "parts.obj"
    write_obj(path, [("Left", cube(), "Red"), ("Right", cube(at=(20, 0, 0)), "Blue")], mtl)
    vols, _ = volumes(path)
    assert [(v.name, v.color) for v in vols] == [("Left", "#ff0000"), ("Right", "#0000ff")]
    assert all(abs(v.manifold.volume() - 1000) < 1e-6 for v in vols)
    assert bounds(vols[1])[0].tolist() == [20, 0, 0]


def test_obj_without_objects_is_one_part_with_positive_indices(tmp_path):
    path = tmp_path / "plain.obj"
    path.write_text("# a cube\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in cube())
                    + "g faces\n" + "".join("f " + " ".join(str(k + 1) for k in q) + "\n"
                                            for q in QUADS))
    (vol,), _ = volumes(path)
    assert vol.name == "plain" and abs(vol.manifold.volume() - 1000) < 1e-6


def test_obj_with_a_bad_index_is_refused(tmp_path):
    path = tmp_path / "bad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
    with pytest.raises(ValueError, match="does not exist"):
        mesh_formats.read_mesh(str(path))


def write_ply(path, fmt, verts, faces):
    head = (f"ply\nformat {fmt} 1.0\ncomment made by a test\nelement vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
            "end_header\n")
    if fmt == "ascii":
        body = "".join(f"{x} {y} {z} 200\n" for x, y, z in verts)
        body += "".join(f"{len(f)} " + " ".join(map(str, f)) + "\n" for f in faces)
        path.write_bytes(head.encode() + body.encode())
        return
    end = "<" if fmt == "binary_little_endian" else ">"
    body = b"".join(struct.pack(end + "fffB", *v, 200) for v in verts)
    body += b"".join(struct.pack(end + "B" + "i" * len(f), len(f), *f) for f in faces)
    path.write_bytes(head.encode() + body)


@pytest.mark.parametrize("fmt", ["ascii", "binary_little_endian", "binary_big_endian"])
@pytest.mark.parametrize("faces", [QUADS, TRIS], ids=["quads", "triangles"])
def test_ply(tmp_path, fmt, faces):
    path = tmp_path / "cube.ply"
    write_ply(path, fmt, cube(), faces)
    (vol,), _ = volumes(path)
    assert abs(vol.manifold.volume() - 1000) < 1e-4


def test_off_with_comments_and_quads(tmp_path):
    path = tmp_path / "cube.off"
    path.write_text("OFF\n# a comment\n8 6 0\n" + "".join(f"{x} {y} {z}\n" for x, y, z in cube())
                    + "".join(f"4 {' '.join(map(str, q))} 255 0 0\n" for q in QUADS))
    (vol,), _ = volumes(path)
    assert abs(vol.manifold.volume() - 1000) < 1e-6


def test_a_file_that_is_not_off_is_refused(tmp_path):
    path = tmp_path / "wrong.off"
    path.write_text("solid nothing\n")
    with pytest.raises(ValueError, match="not an OFF"):
        mesh_formats.read_mesh(str(path))


def model_xml(objects, items, unit="millimeter", extra_ns=""):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="{unit}" '
            f'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"{extra_ns}>'
            f'<resources>{objects}</resources><build>{items}</build></model>')


def mesh_object(oid, verts, name="", tris=TRIS, attrs=""):
    v = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in verts)
    t = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tris)
    return (f'<object id="{oid}" type="model" name="{name}"{attrs}><mesh><vertices>{v}'
            f'</vertices><triangles>{t}</triangles></mesh></object>')


def write_3mf(path, files):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.'
                   'openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/'
                   '3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing'
                   '/2013/01/3dmodel"/></Relationships>')
        for name, data in files.items():
            z.writestr(name, data)


def test_3mf_from_volura_comes_back_with_its_colors(tmp_path):
    from volura.ops import Part
    parts = [Part("Red", "#ff0000", mf.Manifold.cube((10, 10, 10))),
             Part("Blue", "#0000ff", mf.Manifold.cube((10, 10, 10)).translate((20, 0, 0)))]
    path = tmp_path / "two.3mf"
    fileio.export_3mf(str(path), parts, [["Red", "#ff0000"], ["Blue", "#0000ff"]])
    vols, _ = volumes(path)
    assert sorted(v.color for v in vols) == ["#0000ff", "#ff0000"]
    assert all(abs(v.manifold.volume() - 1000) < 1e-4 for v in vols)


def test_3mf_units_components_build_transforms_and_production_paths(tmp_path):
    prod = ' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
    sub = model_xml(mesh_object(1, cube(1.0), "Inner"), "", unit="centimeter")
    objects = (mesh_object(1, cube(1.0), "Plain")
               + '<object id="2" type="model" name="Moved"><components>'
               '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/></components></object>'
               '<object id="3" type="model" name="Elsewhere"><components>'
               '<component objectid="1" p:path="/3D/Objects/inner.model"/></components></object>'
               + mesh_object(4, cube(1.0), "Support").replace('type="model"', 'type="support"'))
    items = ('<item objectid="1"/><item objectid="2" transform="2 0 0 0 2 0 0 0 2 0 0 10"/>'
             '<item objectid="3" transform="1 0 0 0 1 0 0 0 1 0 0 30"/><item objectid="4"/>')
    path = tmp_path / "cm.3mf"
    write_3mf(path, {"3D/3dmodel.model": model_xml(objects, items, "centimeter", prod),
                     "3D/Objects/inner.model": sub})
    vols, _ = volumes(path)
    assert [v.name for v in vols] == ["Plain", "Plain 2", "Inner"]
    assert abs(vols[0].manifold.volume() - 1000) < 1e-4          # 1 cm cube = 10 mm
    assert bounds(vols[1]) [0].tolist() == [100, 0, 100]         # (5 cm, then x2) and z 10 cm
    assert abs(vols[1].manifold.volume() - 8000) < 1e-3
    assert bounds(vols[2])[0].tolist() == [0, 0, 300]


PRUSA_CONFIG = """<?xml version="1.0" encoding="UTF-8"?>
<config><object id="1" instances_count="1">
 <metadata type="object" key="extruder" value="1"/>
 <volume firstid="0" lastid="11"><metadata type="volume" key="name" value="Body"/>
  <metadata type="volume" key="volume_type" value="ModelPart"/></volume>
 <volume firstid="12" lastid="23"><metadata type="volume" key="name" value="Eye"/>
  <metadata type="volume" key="modifier" value="1"/>
  <metadata type="volume" key="volume_type" value="ParameterModifier"/>
  <metadata type="volume" key="extruder" value="2"/></volume>
 <volume firstid="24" lastid="35"><metadata type="volume" key="name" value="Infill"/>
  <metadata type="volume" key="volume_type" value="ParameterModifier"/></volume>
</object></config>"""


def test_3mf_from_prusaslicer_splits_parts_and_colors_the_color_modifiers(tmp_path):
    verts = np.concatenate([cube(10.0), cube(4.0, at=(8, 0, 0)), cube(2.0, at=(1, 1, 1))])
    tris = TRIS + [tuple(k + 8 for k in t) for t in TRIS] + [tuple(k + 16 for k in t) for t in TRIS]
    path = tmp_path / "prusa.3mf"
    write_3mf(path, {
        "3D/3dmodel.model": model_xml(mesh_object(1, verts, tris=tris), '<item objectid="1"/>'),
        "Metadata/Slic3r_PE_model.config": PRUSA_CONFIG,
        "Metadata/Slic3r_PE.config": "; extruder_colour = #FF0000;\n"
                                     "; filament_colour = #00FF00;#0000FF\n"})
    vols, notes = volumes(path)
    assert [(v.name, v.color) for v in vols] == [("Body", "#ff0000"), ("Eye", "#0000ff")]
    # the eye is only the part of the modifier inside the body, cut out of the body
    assert abs(vols[1].manifold.volume() - 2 * 4 * 4) < 1e-4
    assert abs(vols[0].manifold.volume() - (1000 - 32)) < 1e-4
    assert any("Infill" in n for n in notes)            # a modifier without a filament


def test_3mf_from_bambu_studio_leaves_out_modifier_parts(tmp_path):
    objects = (mesh_object(1, cube(10.0)) + mesh_object(2, cube(2.0))
               + '<object id="3" type="model"><components><component objectid="1"/>'
               '<component objectid="2"/></components></object>')
    config = ('<?xml version="1.0"?><config><object id="3"><metadata key="name" value="Box"/>'
              '<metadata key="extruder" value="2"/>'
              '<part id="1" subtype="normal_part"><metadata key="name" value="Box"/></part>'
              '<part id="2" subtype="negative_part"><metadata key="name" value="Hole"/></part>'
              '</object></config>')
    path = tmp_path / "bambu.3mf"
    write_3mf(path, {"3D/3dmodel.model": model_xml(objects, '<item objectid="3"/>'),
                     "Metadata/model_settings.config": config,
                     "Metadata/project_settings.config": json.dumps(
                         {"filament_colour": ["#FFFFFF", "#123456"]})})
    vols, notes = volumes(path)
    assert [(v.name, v.color) for v in vols] == [("Box", "#123456")]
    assert any("Hole" in n for n in notes)


def test_a_damaged_3mf_is_refused(tmp_path):
    path = tmp_path / "broken.3mf"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="damaged"):
        mesh_formats.read_mesh(str(path))


def amf_xml(unit="millimeter"):
    v = "".join(f"<vertex><coordinates><x>{x}</x><y>{y}</y><z>{z}</z></coordinates></vertex>"
                for x, y, z in cube(1.0))
    t = "".join(f"<triangle><v1>{a}</v1><v2>{b}</v2><v3>{c}</v3></triangle>" for a, b, c in TRIS)
    return (f'<?xml version="1.0"?><amf unit="{unit}"><material id="1"><color><r>0</r><g>1</g>'
            f'<b>0</b></color></material><object id="1"><metadata type="name">Box</metadata>'
            f'<mesh><vertices>{v}</vertices><volume materialid="1">{t}</volume></mesh></object>'
            '</amf>')


@pytest.mark.parametrize("zipped", [False, True])
def test_amf_with_units_and_material_color(tmp_path, zipped):
    path = tmp_path / "box.amf"
    if zipped:
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("box.amf", amf_xml("inch"))
    else:
        path.write_text(amf_xml("inch"))
    (vol,), _ = volumes(path)
    assert vol.name == "Box" and vol.color == "#00ff00"
    assert abs(vol.manifold.volume() - 25.4 ** 3) < 1e-2


def gltf_doc(binary_uri=None):
    verts = cube(0.01).astype("<f4")             # 1 cm, in metres
    idx = np.array(TRIS, dtype="<u2").ravel()
    data = idx.tobytes() + verts.tobytes()
    doc = {"asset": {"version": "2.0"}, "scene": 0,
           "scenes": [{"nodes": [0]}],
           "nodes": [{"name": "Holder", "translation": [0, 0.05, 0], "children": [1]},
                     {"name": "Cube", "mesh": 0, "scale": [2, 1, 1]}],
           "meshes": [{"name": "CubeMesh", "primitives": [
               {"attributes": {"POSITION": 1}, "indices": 0, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 0.5, 0, 1]}}],
           "buffers": [{"byteLength": len(data)} if binary_uri is None else
                       {"byteLength": len(data), "uri": binary_uri(data)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(idx) * 2},
                           {"buffer": 0, "byteOffset": len(idx) * 2,
                            "byteLength": verts.nbytes}],
           "accessors": [{"bufferView": 0, "componentType": 5123, "count": len(idx),
                          "type": "SCALAR"},
                         {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC3"}]}
    return doc, data


def check_gltf_cube(vol):
    # metres to millimetres, Y up to Z up: the 5 cm shift along Y is now 50 mm up
    assert vol.name == "Cube" and vol.color == "#ff8000"
    assert abs(vol.manifold.volume() - 20 * 10 * 10) < 1e-2
    lo, hi = bounds(vol)
    assert lo[2] == pytest.approx(50, abs=1e-3) and hi[2] == pytest.approx(60, abs=1e-3)
    assert hi[0] - lo[0] == pytest.approx(20, abs=1e-3)


def test_glb(tmp_path):
    doc, data = gltf_doc()
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    data += b"\0" * (-len(data) % 4)
    body = (struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(data), 0x004E4942) + data)
    path = tmp_path / "cube.glb"
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    (vol,), _ = volumes(path)
    check_gltf_cube(vol)


def test_gltf_with_an_embedded_and_with_a_separate_buffer(tmp_path):
    doc, _ = gltf_doc(lambda d: "data:application/octet-stream;base64,"
                      + base64.b64encode(d).decode())
    path = tmp_path / "embedded.gltf"
    path.write_text(json.dumps(doc))
    check_gltf_cube(volumes(path)[0][0])

    def separate(d):
        (tmp_path / "cube data.bin").write_bytes(d)
        return "cube%20data.bin"
    doc, _ = gltf_doc(separate)
    path = tmp_path / "separate.gltf"
    path.write_text(json.dumps(doc))
    check_gltf_cube(volumes(path)[0][0])


DAE = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit meter="0.01" name="centimeter"/><up_axis>Y_UP</up_axis></asset>
 <library_effects><effect id="fx"><profile_COMMON><technique sid="t"><lambert>
  <diffuse><color>0 0 1 1</color></diffuse></lambert></technique></profile_COMMON></effect>
 </library_effects>
 <library_materials><material id="blue"><instance_effect url="#fx"/></material></library_materials>
 <library_geometries><geometry id="box" name="BoxMesh"><mesh>
  <source id="pos"><float_array id="pos-a" count="24">{coords}</float_array>
   <technique_common><accessor source="#pos-a" count="8" stride="3"/></technique_common></source>
  <vertices id="verts"><input semantic="POSITION" source="#pos"/></vertices>
  <polylist count="6" material="mat"><input semantic="VERTEX" source="#verts" offset="0"/>
   <input semantic="NORMAL" source="#pos" offset="1"/>
   <vcount>4 4 4 4 4 4</vcount><p>{quads}</p></polylist>
 </mesh></geometry></library_geometries>
 <library_nodes><node id="part" name="Part"><instance_geometry url="#box">
  <bind_material><technique_common><instance_material symbol="mat" target="#blue"/>
  </technique_common></bind_material></instance_geometry></node></library_nodes>
 <library_visual_scenes><visual_scene id="scene">
  <node name="Moved"><translate>0 3 0</translate><instance_node url="#part"/></node>
  <node name="Twice"><matrix>2 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
   <instance_node url="#part"/></node>
 </visual_scene></library_visual_scenes>
 <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>"""


def test_dae_with_units_y_up_shared_nodes_and_material_color(tmp_path):
    path = tmp_path / "box.dae"
    path.write_text(DAE.format(coords=" ".join(str(c) for c in cube(1.0).ravel()),
                               quads=" ".join(f"{k} 0" for q in QUADS for k in q)))
    vols, _ = volumes(path)
    assert [v.name for v in vols] == ["Part", "Part 2"]
    assert all(v.color == "#0000ff" for v in vols)
    assert abs(vols[0].manifold.volume() - 1000) < 1e-3          # 1 cm = 10 mm
    assert bounds(vols[0])[0][2] == pytest.approx(30)           # 3 cm along Y is up
    assert abs(vols[1].manifold.volume() - 2000) < 1e-3
    assert bounds(vols[1])[0][0] == pytest.approx(100)


# ------------------------------------------------------------------ closed or not

def test_a_loose_fin_is_dropped_so_the_rest_is_closed():
    # a stray triangle hinged on the edge 4-5 of the cube, sticking up: that edge is used
    # three times, the two other edges of the stray triangle once each
    verts = np.concatenate([cube(), [[5.0, 0.0, 15.0]]])
    vol = fileio.closed_volume(verts, np.array(TRIS + [(4, 5, 8)]))
    assert vol.manifold is not None and vol.loose == 1
    assert abs(vol.manifold.volume() - 1000) < 1e-6
    # a cube without flaws drops nothing
    assert fileio.closed_volume(cube(), np.array(TRIS)).loose == 0


def test_an_open_mesh_is_refused_with_the_reason(tmp_path):
    path = tmp_path / "open.off"
    path.write_text("OFF\n8 5 0\n" + "".join(f"{x} {y} {z}\n" for x, y, z in cube())
                    + "".join(f"4 {' '.join(map(str, q))}\n" for q in QUADS[1:]))
    with pytest.raises(ValueError, match="not watertight"):
        volumes(path)


def test_open_parts_are_tried_together(tmp_path):
    # one cube split over two objects (as a program does per material): open apart,
    # closed together
    path = tmp_path / "split.obj"
    verts = cube()
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += ["o Sides"] + ["f " + " ".join(str(k + 1) for k in q) for q in QUADS[2:]]
    lines += ["o Caps"] + ["f " + " ".join(str(k + 1) for k in q) for q in QUADS[:2]]
    path.write_text("\n".join(lines) + "\n")
    (vol,), notes = volumes(path)
    assert vol.name == "split" and abs(vol.manifold.volume() - 1000) < 1e-6 and not notes


def test_an_open_part_next_to_a_closed_one_is_left_out_with_a_note(tmp_path):
    path = tmp_path / "mixed.obj"
    lines = [f"v {x} {y} {z}" for x, y, z in cube()]
    lines += [f"v {x} {y} {z}" for x, y, z in cube(at=(20, 0, 0))]
    lines += ["o Good"] + ["f " + " ".join(str(k + 1) for k in q) for q in QUADS]
    lines += ["o Leaky"] + ["f " + " ".join(str(k + 9) for k in q) for q in QUADS[1:]]
    path.write_text("\n".join(lines) + "\n")
    (vol,), notes = volumes(path)
    assert vol.name == "Good" and any("Leaky" in n for n in notes)


def test_progress_only_goes_forward_and_ends_at_one(tmp_path):
    path = tmp_path / "parts.obj"
    write_obj(path, [("A", cube(), None), ("B", cube(at=(20, 0, 0)), None)])
    seen = []
    fileio.import_mesh_report(str(path), seen.append)
    assert seen and seen[-1] == 1.0
    assert all(b >= a for a, b in zip(seen, seen[1:]))


# ------------------------------------------------------------------ in the window

class FakeBox:
    def __init__(self):
        self.asked = []

    def critical(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    w = app_module.MainWindow()
    w.box = box
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def every_format(tmp_path):
    """One closed cube of 10 mm in every 3D format."""
    files = {}
    p = tmp_path / "f.stl"
    fileio.write_stl(str(p), mf.Manifold.cube((10.0, 10.0, 10.0)))
    files["stl"] = p
    p = tmp_path / "f.obj"
    write_obj(p, [("Cube", cube(), None)])
    files["obj"] = p
    p = tmp_path / "f.3mf"
    write_3mf(p, {"3D/3dmodel.model": model_xml(mesh_object(1, cube()), '<item objectid="1"/>')})
    files["3mf"] = p
    p = tmp_path / "f.ply"
    write_ply(p, "binary_little_endian", cube(), QUADS)
    files["ply"] = p
    p = tmp_path / "f.off"
    p.write_text("OFF 8 6 0\n" + "".join(f"{x} {y} {z}\n" for x, y, z in cube())
                 + "".join(f"4 {' '.join(map(str, q))}\n" for q in QUADS))
    files["off"] = p
    p = tmp_path / "f.amf"
    p.write_text(amf_xml("centimeter"))
    files["amf"] = p
    doc, _ = gltf_doc(lambda d: "data:application/octet-stream;base64,"
                      + base64.b64encode(d).decode())
    doc["nodes"][1]["scale"] = [1, 1, 1]
    p = tmp_path / "f.gltf"
    p.write_text(json.dumps(doc))
    files["gltf"] = p
    p = tmp_path / "f.dae"
    p.write_text(DAE.format(coords=" ".join(str(c) for c in cube(1.0).ravel()),
                            quads=" ".join(f"{k} 0" for q in QUADS for k in q))
                 .replace('<node name="Twice"><matrix>2 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1</matrix>\n'
                          '   <instance_node url="#part"/></node>', ""))
    files["dae"] = p
    return files


@pytest.mark.parametrize("kind", ["stl", "obj", "3mf", "ply", "off", "amf", "gltf", "dae"])
def test_the_window_imports_every_format_in_the_background(win, tmp_path, kind):
    path = every_format(tmp_path)[kind]
    job = win.import_file(str(path))
    job.wait()
    assert job.error is None, job.error
    assert win.box.asked == []
    (g,) = win.doc.groups
    assert g.name == "f"
    assert abs(g.ents.to_manifold().volume() - 1000) < 1e-2
    assert win.doc.selection == [("g", g.id, g.id)]


def test_import_reports_progress_with_text(win, tmp_path, monkeypatch):
    path = tmp_path / "parts.obj"
    write_obj(path, [("A", cube(), None), ("B", cube(at=(20, 0, 0)), None)])
    seen = []
    real = background._Worker.report

    def spy(self, fraction=None, text=None):
        seen.append((fraction, text))
        return real(self, fraction, text)
    monkeypatch.setattr(background._Worker, "report", spy)
    job = win.import_file(str(path))
    job.wait()
    fractions = [f for f, _ in seen if f is not None]
    assert fractions[-1] == 1.0 and all(b >= a - 1e-9 for a, b in zip(fractions, fractions[1:]))
    texts = [t for _, t in seen if t]
    assert any("parts.obj" in t for t in texts) and any("Building A" in t for t in texts)


def test_a_multi_part_file_becomes_selected_colored_groups_in_one_undo_step(win, tmp_path):
    mtl = tmp_path / "parts.mtl"
    mtl.write_text("newmtl Red\nKd 1 0 0\n")
    path = tmp_path / "parts.obj"
    # (a material holds until the next usemtl, so the part without one comes first)
    write_obj(path, [("Left", cube(), None), ("Right", cube(at=(20, 0, 0)), "Red")], mtl)
    win.import_file(str(path)).wait()
    left, right = win.doc.groups
    assert (left.name, left.color) == ("Left", win.settings.current_color)
    assert (right.name, right.color) == ("Right", "#ff0000")
    assert len(win.doc.selection) == 2 and "2 parts imported" in win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_a_broken_file_shows_the_error(win, tmp_path):
    path = tmp_path / "broken.3mf"
    path.write_bytes(b"nope")
    win.import_file(str(path)).wait()
    assert win.doc.groups == [] and "damaged" in str(win.box.asked)


def test_open_and_drop_accept_every_format(win, tmp_path):
    files = every_format(tmp_path)
    win.file_open(str(files["ply"]))
    assert len(win.doc.groups) == 1 or pump_jobs(win, 1)
    win.import_dropped_file(str(files["dae"]))
    assert pump_jobs(win, 2)


def pump_jobs(win, n):
    import time
    end = time.time() + 30
    while len(win.doc.groups) < n and time.time() < end:
        QApplication.processEvents()
    return len(win.doc.groups) == n


def big_drawing(tmp_path):
    """An SVG with enough points that it is placed in the background."""
    circles = "".join(f'<circle cx="{(i % 10) * 30}" cy="{(i // 10) * 30}" r="10"/>'
                      for i in range(60))
    path = tmp_path / "many.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="400mm" '
                    f'viewBox="0 0 400 400">{circles}</svg>')
    return path


def test_a_big_drawing_is_placed_in_the_background_like_a_small_one(win, tmp_path):
    path = big_drawing(tmp_path)
    drawing = vector_import.load_vector(str(path))
    regs, opens = vector_import.prepare(drawing)
    points = sum(len(loop) for shape in regs for loop in shape)
    assert points >= win.BACKGROUND_PLACE_POINTS
    from volura.model import Document
    direct = Document()
    vector_import.place_faces(direct.root, Plane((0, 0, 0), (0, 0, 1)), regs, opens, (0, 0))
    win.import_vector(str(path), {"anchor": "center", "open_edges": True})
    assert len(win.doc.root.F) == len(direct.root.F) == 60
    assert len(win.doc.root.E) == len(direct.root.E)
    win.doc.undo()
    assert not win.doc.root.F


def test_cancelling_the_placing_leaves_the_model_as_it_was(win, tmp_path, monkeypatch):
    path = big_drawing(tmp_path)

    def cancel_at_once(window, task, **kw):
        job = background.BackgroundJob(window, task, kw.get("text", ""))
        job.cancel()
        job.start()
        while not job.done:
            job.wait()
        return job
    loaded = win.load_vector(str(path))
    monkeypatch.setattr(app_module.background, "run", cancel_at_once)
    before = len(win.doc.undo_stack)
    assert win.place_vector(loaded, {"anchor": "center"}) is None
    assert not win.doc.root.F and len(win.doc.undo_stack) == before


def test_a_vector_file_that_fails_to_read_shows_the_error(win, tmp_path):
    path = tmp_path / "bad.svg"
    path.write_text("<svg")
    assert win.import_vector(str(path), {"anchor": "center"}) is None
    assert win.box.asked
