"""Reading and writing Volura documents, STL and 3MF files."""
from __future__ import annotations

import json
import os
import re
import struct
import tempfile
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import manifold3d as mf

from . import __version__
from .brep import Entities, Face
from .geom import Plane, manifold_from_mesh, mesh_arrays
from .i18n import tr
from .model import ROOT, Document, Group, Guide
from .ops import id_color, print_parts, union_all
from .vector2d import Drawing

FORMAT_VERSION = 2


# ------------------------------------------------------------------ writing safely

def atomic_write(path, write):
    """Write to a temporary file in the same folder and only then replace `path`.

    `write` gets the name of the temporary file; if it fails, the existing file stays
    intact and the temporary file is removed.
    """
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def write_zip(path, files):
    """Write the given names and contents as a zip file (a .volura or .3mf)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


# ------------------------------------------------------------------ native

def _ents_to_json(ents):
    return {
        "V": {str(k): [round(float(c), 9) for c in p] for k, p in ents.V.items()},
        "E": {str(k): list(v) for k, v in ents.E.items()},
        "F": {str(k): {"loops": [list(l) for l in f.loops], "n": f.normal.tolist(), "color": f.color}
              for k, f in ents.F.items()},
        "soft": sorted(ents.soft),
        "hidden_f": sorted(ents.hidden_f),
        "hidden_e": sorted(ents.hidden_e),
        "next": ents.next_id,
    }


def _ents_from_json(d):
    V = {int(k): np.array(p, dtype=float) for k, p in d["V"].items()}
    E = {int(k): tuple(v) for k, v in d["E"].items()}
    F = {int(k): Face(f["loops"], f["n"], f.get("color")) for k, f in d["F"].items()}
    hidden = (set(d.get("hidden_f", [])), set(d.get("hidden_e", [])))
    return Entities.from_state((V, E, F, set(d.get("soft", [])), d.get("next", 1), hidden))


def save_document(doc: Document, path):
    data = {
        "format": "volura", "version": FORMAT_VERSION, "app": __version__,
        "materials": doc.materials, "next_gid": doc.next_gid, "group_counter": doc.group_counter,
        "root": _ents_to_json(doc.root),
        "groups": [{"id": g.id, "name": g.name, "color": g.color, "visible": g.visible,
                    "locked": g.locked, "ents": _ents_to_json(g.ents)} for g in doc.groups],
        "guides": [{"id": g.id, "p": [round(float(c), 9) for c in g.point],
                    "d": None if g.direction is None else [round(float(c), 9) for c in g.direction]}
                   for g in doc.guides],
        "next_guide_id": doc.next_guide_id,
        "sheet": doc.sheet.to_json(),
    }
    files = {"document.json": json.dumps(data)}
    atomic_write(path, lambda tmp: write_zip(tmp, files))


def load_document(doc: Document, path):
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
        if data.get("format") != "volura":
            raise ValueError(tr("Not a Volura file."))
        doc.reset()
        doc.materials = [list(m) for m in data.get("materials", doc.materials)]
        if data.get("version", 1) < 2:
            _load_v1(doc, z, data)
        else:
            doc.root = _ents_from_json(data["root"])
            for gd in data["groups"]:
                doc.groups.append(Group(gd["id"], gd["name"], gd["color"], _ents_from_json(gd["ents"]),
                                        gd.get("visible", True), gd.get("locked", False)))
            doc.next_gid = data.get("next_gid", 1 + max([0] + [g.id for g in doc.groups]))
            doc.group_counter = data.get("group_counter", len(doc.groups) + 1)
            for gd in data.get("guides", []):      # files from before the guides have no guides
                doc.guides.append(Guide(gd["id"], gd["p"], gd.get("d")))
            doc.next_guide_id = data.get("next_guide_id",
                                         1 + max([0] + [g.id for g in doc.guides]))
        doc.load_sheet(Drawing.from_json(data.get("sheet") or {}))   # old files have no sheet
    doc.path = path
    doc.dirty = False


def _load_v1(doc, z, data):
    """Files of Volura 0.1: solids become groups, sketch faces become loose faces."""
    import io
    for od in data["objects"]:
        arr = np.load(io.BytesIO(z.read(od["mesh"])))
        m = manifold_from_mesh(arr["verts"], arr["tris"].astype(np.int64))
        g = doc.add_group(Entities.from_manifold(m), od["color"], od["name"])
        g.visible = od.get("visible", True)
    for rd in data["regions"]:
        pl = Plane.from_dict(rd["plane"])
        doc.root.add_shape(pl, [np.asarray(p, float) for p in rd["polys"]])
    for ld in data["lines"]:
        pts = np.asarray(ld["points"], float)
        for a, b in zip(pts[:-1], pts[1:]):
            doc.root.add_line(a, b)


# ------------------------------------------------------------------ STL

def write_stl(path, manifold, name="Volura"):
    v, t = mesh_arrays(manifold)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n / np.maximum(ln, 1e-30)[:, None]
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri

    def write(tmp):
        with open(tmp, "wb") as f:
            header = f"{name} - exported by Volura {__version__}".encode("ascii", "replace")[:80]
            f.write(header.ljust(80, b" "))
            f.write(struct.pack("<I", len(t)))
            f.write(rec.tobytes())

    atomic_write(path, write)
    return len(t)


def read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + count * 50 == len(data):
            rec = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            verts = rec["v"].reshape(-1, 3).astype(float)
            return verts, np.arange(len(verts)).reshape(-1, 3)
    text = data.decode("ascii", "ignore")
    nums = re.findall(r"vertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)", text)
    verts = np.array(nums, dtype=float).reshape(-1, 3)
    verts = verts[: len(verts) // 3 * 3]
    return verts, np.arange(len(verts)).reshape(-1, 3)


def import_stl(path):
    """Return a Manifold, raising ValueError when the mesh is not watertight."""
    v, t = read_stl(path)
    if not len(t):
        raise ValueError(tr("The STL file contains no triangles."))
    m = manifold_from_mesh(v, t)
    if m.status() != mf.Error.NoError or m.is_empty():
        raise ValueError(tr("The STL file is not watertight ({status}). "
                            "Only closed volumes can be imported.").format(status=m.status()))
    return m


def export_stl(path, part_list):
    m = union_all([p.manifold for p in part_list if p.manifold is not None])
    if m.is_empty():
        raise ValueError(tr("There is nothing to export."))
    return write_stl(path, m, os.path.splitext(os.path.basename(path))[0])


def export_stl_per_color(path, part_list, materials):
    """Write one STL per color: <name>_<color>.stl. Returns the written files."""
    base, _ = os.path.splitext(path)
    names = {c.lower(): n for n, c in materials}
    files = []
    for i, (color, m) in enumerate(print_parts(part_list), 1):
        cname = names.get(color.lower(), color.lstrip("#"))
        safe = re.sub(r"[^\w-]+", "_", cname)
        fn = f"{base}_{i}_{safe}.stl"
        write_stl(fn, m, cname)
        files.append(fn)
    return files


# ------------------------------------------------------------------ 3MF

def paint_code(extruder):
    """A whole triangle painted with this extruder, in the code of the slicers' color brush.

    OrcaSlicer / Bambu Studio (paint_color) and PrusaSlicer (slic3rpe:mmu_segmentation)
    store painted triangles as a hex string of bits: two bits "not split", then the state;
    states 1 and 2 fit in two bits, from 3 on the bits are 11 plus four bits of state - 3.
    The nibbles are written last first: extruder 1 = "4", 2 = "8", 3 = "0C", 4 = "1C".
    """
    k = int(extruder)
    if k == 1:
        return "4"
    if k == 2:
        return "8"
    if 3 <= k <= 18:
        return f"{k - 3:X}C"
    return ""


def triangle_colors(man):
    """The face color of every triangle of a Manifold (None where it has none)."""
    mesh = man.to_mesh64()
    ids = np.asarray(mesh.face_id, dtype=np.int64) if len(mesh.face_id) else None
    n = len(np.asarray(mesh.tri_verts).reshape(-1, 3))
    if ids is None or len(ids) != n:
        return [None] * n
    return [id_color(i) for i in ids]


def export_3mf(path, part_list, materials, name="Volura"):
    """Multi-color 3MF: one object, one part per color, painted faces as painted triangles.

    Colors are stored as 3MF base materials. The part/extruder assignment is written
    both in the PrusaSlicer layout and in the Bambu Studio / OrcaSlicer layout.
    Faces painted in another color than their part become painted triangles, which
    OrcaSlicer, Bambu Studio and PrusaSlicer read as their own color painting.

    Only the colors that are used get an extruder: 1, 2, 3, ... in the order of the
    material list (colors that are not in the list come last). The file then fits a
    slicer with as many filaments as the model has colors. Returns [(name, extruder)].
    """
    parts = print_parts(part_list)
    if not parts:
        raise ValueError(tr("There is nothing to export."))
    names = {c.lower(): n for n, c in materials}
    tri_colors = [triangle_colors(m) for _, m in parts]
    present = {color.lower() for color, _ in parts}
    for colors in tri_colors:
        present.update(c.lower() for c in colors if c is not None)
    order = [c.lower() for _, c in materials]
    order += sorted(present - set(order))
    numbers = {c: k + 1 for k, c in enumerate(c for c in order if c in present)}

    def extruder(color):
        return numbers[color.lower()]

    paired = sorted(zip(parts, tri_colors), key=lambda pt: extruder(pt[0][0]))
    parts = [pt[0] for pt in paired]
    tri_colors = [pt[1] for pt in paired]
    used = [p[0] for p in parts]
    painted = []

    # One mesh object per color, joined as components of a single parent object.
    # Every slicer then sees one object made of several parts.
    parent = len(parts) + 2
    mesh_xml, comp_xml, volumes = [], [], []
    toff = 0
    for k, (color, m) in enumerate(parts):
        v, t = mesh_arrays(m)
        oid = k + 2
        cname = escape(names.get(color.lower(), color))
        tris = []
        for (a, b, c), tc in zip(t, tri_colors[k]):
            paint = ""
            if tc is not None and tc.lower() != color.lower():
                code = paint_code(extruder(tc))
                if code:
                    paint = f' paint_color="{code}" slic3rpe:mmu_segmentation="{code}"'
                    if tc.lower() not in painted and tc.lower() not in used:
                        painted.append(tc.lower())
            tris.append(f'<triangle v1="{a}" v2="{b}" v3="{c}"{paint}/>')
        mesh_xml.append(
            f'<object id="{oid}" type="model" name="{cname}" pid="1" pindex="{k}"><mesh><vertices>\n'
            + "\n".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in v)
            + '\n</vertices><triangles>\n'
            + "\n".join(tris)
            + '\n</triangles></mesh></object>\n')
        comp_xml.append(f'<component objectid="{oid}"/>')
        volumes.append((toff, toff + len(t) - 1, color, oid))
        toff += len(t)

    bases = "".join(
        f'<base name="{escape(names.get(c.lower(), c))}" displaycolor="{c.upper()}FF"/>'
        for c in used + painted)
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
        f'<metadata name="Application">Volura-{__version__}</metadata>\n'
        f'<metadata name="Title">{escape(name)}</metadata>\n'
        '<resources>\n'
        f'<basematerials id="1">{bases}</basematerials>\n'
        + "".join(mesh_xml)
        + f'<object id="{parent}" type="model" name="{escape(name)}"><components>'
        + "".join(comp_xml) + '</components></object>\n</resources>\n'
        f'<build><item objectid="{parent}"/></build>\n</model>\n')

    # PrusaSlicer: parts as triangle ranges of the combined object.
    vol_xml = []
    for first, last, color, _ in volumes:
        vol_xml.append(
            f' <volume firstid="{first}" lastid="{last}">\n'
            f'  <metadata type="volume" key="name" value="{escape(names.get(color.lower(), color))}"/>\n'
            f'  <metadata type="volume" key="volume_type" value="ModelPart"/>\n'
            f'  <metadata type="volume" key="extruder" value="{extruder(color)}"/>\n'
            ' </volume>\n')
    config = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
              f'<object id="{parent}" instances_count="1">\n'
              f' <metadata type="object" key="name" value="{escape(name)}"/>\n'
              + "".join(vol_xml) + '</object>\n</config>\n')

    # OrcaSlicer / Bambu Studio: parts refer to the component objects.
    part_xml = []
    for _, _, color, oid in volumes:
        part_xml.append(
            f'    <part id="{oid}" subtype="normal_part">\n'
            f'      <metadata key="name" value="{escape(names.get(color.lower(), color))}"/>\n'
            f'      <metadata key="extruder" value="{extruder(color)}"/>\n'
            '    </part>\n')
    bambu_config = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                    f'  <object id="{parent}">\n'
                    f'    <metadata key="name" value="{escape(name)}"/>\n'
                    f'    <metadata key="extruder" value="{extruder(parts[0][0])}"/>\n'
                    + "".join(part_xml) + '  </object>\n</config>\n')

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="config" ContentType="text/xml"/>'
        '</Types>')
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>')
    files = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": rels,
        "3D/3dmodel.model": model,
        "Metadata/Slic3r_PE_model.config": config,
        "Metadata/model_settings.config": bambu_config,
    }
    atomic_write(path, lambda tmp: write_zip(tmp, files))
    return sorted(((names.get(c.lower(), c), extruder(c)) for c in used + painted),
                  key=lambda item: item[1])
