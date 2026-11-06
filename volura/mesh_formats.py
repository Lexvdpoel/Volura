"""Reading 3D mesh files: STL, OBJ, 3MF, PLY, OFF, AMF, glTF/GLB and COLLADA (DAE).

Every reader returns a :class:`MeshFile`: the parts of the file -- a name, a triangle
soup and a color when the file has one -- in millimetres with Z up, the way Volura
models, plus notes about what was left out. Nothing here knows about the document or
Qt: `fileio.import_mesh_report` turns the parts into closed volumes and the window
imports those as groups.

`progress(fraction)` (0..1) is called now and then while a big file is read, so the
import can show how far it is; it may raise to stop the reading (a cancelled import).
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np

from .i18n import tr

#: the extensions `read_mesh` understands, with the reader for each
MESH_EXTENSIONS = (".stl", ".obj", ".3mf", ".ply", ".off", ".amf", ".glb", ".gltf", ".dae")

#: millimetres per unit of the unit names of 3MF and AMF
UNIT_MM = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "meter": 1000.0,
           "inch": 25.4, "foot": 304.8}

#: how many lines or elements a reader handles between two progress calls
PROGRESS_EVERY = 100_000


@dataclass
class MeshPart:
    """One piece of a mesh file: a triangle soup, in millimetres with Z up."""

    name: str
    verts: np.ndarray                 # (N, 3) float
    tris: np.ndarray                  # (M, 3) int64 indices into verts
    color: str | None = None          # "#rrggbb", or None when the file gives none
    #: a slicer's color modifier: only where it overlaps the printed parts with the same
    #: `owner` does it count, as a region in its own color
    modifier: bool = False
    owner: str | None = None


@dataclass
class MeshFile:
    parts: list = field(default_factory=list)
    notes: list = field(default_factory=list)    # translated remarks on what was skipped


def is_mesh(path) -> bool:
    """True if `path` is a 3D mesh file Volura can import (also in upper case)."""
    return os.path.splitext(str(path))[1].lower() in MESH_EXTENSIONS


def read_mesh(path, progress=None) -> MeshFile:
    """Read any supported mesh file; the extension decides how."""
    ext = os.path.splitext(str(path))[1].lower()
    reader = {".stl": read_stl_file, ".obj": read_obj, ".3mf": read_3mf, ".ply": read_ply,
              ".off": read_off, ".amf": read_amf, ".glb": read_gltf, ".gltf": read_gltf,
              ".dae": read_dae}.get(ext)
    if reader is None:
        raise ValueError(tr("Volura cannot import {ext} files.").format(ext=ext or path))
    out = reader(path, progress or _no_progress)
    out.parts = [p for p in out.parts if len(p.tris)]
    return out


def _no_progress(fraction):
    pass


def _base(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _soup(verts, tris):
    return (np.asarray(verts, dtype=float).reshape(-1, 3),
            np.asarray(tris, dtype=np.int64).reshape(-1, 3))


def _hex(rgb):
    """0..1 floats as "#rrggbb"."""
    r, g, b = (max(0, min(255, round(float(c) * 255))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_color(text):
    """A color written as #RGB, #RRGGBB or #RRGGBBAA as "#rrggbb"; None if it is not one."""
    t = (text or "").strip().lstrip("#")
    if len(t) in (3, 4):
        t = "".join(c * 2 for c in t[:3])
    if len(t) >= 6 and re.fullmatch(r"[0-9a-fA-F]+", t[:6]):
        return "#" + t[:6].lower()
    return None


def _fan(poly):
    """A polygon (a list of vertex indices) as triangles fanned from its first corner."""
    return [(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1)]


def _transform(verts, M):
    """`verts` (N, 3) through the 4x4 matrix `M` (column vectors)."""
    return verts @ M[:3, :3].T + M[:3, 3]


#: Y up (glTF, most COLLADA) to Z up: a quarter turn about X
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


# ------------------------------------------------------------------ STL

STL_RECORD = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_stl(path):
    """The triangle soup of a binary or ASCII STL: ``(verts, tris)``."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + count * 50 == len(data):
            rec = np.frombuffer(data[84:], dtype=STL_RECORD)
            verts = rec["v"].reshape(-1, 3).astype(float)
            return verts, np.arange(len(verts)).reshape(-1, 3)
    text = data.decode("ascii", "ignore")
    nums = re.findall(r"vertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)", text)
    verts = np.array(nums, dtype=float).reshape(-1, 3)
    verts = verts[: len(verts) // 3 * 3]
    return verts, np.arange(len(verts)).reshape(-1, 3)


def read_stl_file(path, progress):
    v, t = read_stl(path)
    return MeshFile([MeshPart(_base(path), v, t)])


# ------------------------------------------------------------------ OBJ

def _obj_materials(folder, names):
    """The diffuse colors (Kd) of the materials in the .mtl files next to an OBJ."""
    colors = {}
    for name in names:
        try:
            with open(os.path.join(folder, name), encoding="utf-8", errors="ignore") as f:
                current = None
                for line in f:
                    w = line.split()
                    if len(w) >= 2 and w[0] == "newmtl":
                        current = " ".join(w[1:])
                    elif len(w) >= 4 and w[0] == "Kd" and current is not None:
                        colors[current] = _hex([float(c) for c in w[1:4]])
        except (OSError, ValueError):
            pass
    return colors


def read_obj(path, progress):
    """Wavefront OBJ: every ``o`` object becomes a part; polygons are fanned."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    coords = []
    objects = []                       # [name, flat triangle indices, {material: faces}]
    current = None
    material = None
    libs = []
    n = len(lines)
    for i, raw in enumerate(lines):
        if i % PROGRESS_EVERY == 0:
            progress(0.9 * i / max(n, 1))
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        head, _, rest = line.partition(" ")
        if head == "v":
            w = rest.split()
            coords.append(w[:3])
        elif head == "f":
            if current is None:
                current = [_base(path), [], {}]
                objects.append(current)
            nv = len(coords)
            poly = []
            for tok in rest.split():
                k = int(tok.split("/", 1)[0])
                poly.append(k - 1 if k > 0 else nv + k)
            if len(poly) >= 3:
                current[1].extend(_fan(poly))
                current[2][material] = current[2].get(material, 0) + len(poly) - 2
        elif head == "o":
            # a new object (groups, "g", are often only material sections: they stay)
            name = rest.strip() or _base(path)
            if current is not None and not current[1]:
                current[0] = name
            else:
                current = [name, [], {}]
                objects.append(current)
        elif head == "usemtl":
            material = rest.strip()
        elif head == "mtllib":
            libs.append(rest.strip())
    verts = np.array(coords, dtype=float).reshape(-1, 3) if coords else np.zeros((0, 3))
    colors = _obj_materials(os.path.dirname(os.path.abspath(path)), libs) if libs else {}
    out = MeshFile()
    for name, tris, used in objects:
        if not tris:
            continue
        t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if t.min() < 0 or t.max() >= len(verts):
            raise ValueError(tr("The OBJ file refers to a vertex that does not exist."))
        main = max(used, key=used.get) if used else None
        out.parts.append(MeshPart(name, verts, t, colors.get(main)))
    progress(1.0)
    return out


# ------------------------------------------------------------------ OFF

def read_off(path, progress):
    """Object File Format (OFF, COFF, NOFF ...): one part, polygons fanned."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [ln.split("#", 1)[0].strip() for ln in f]
    lines = [ln for ln in lines if ln]
    if not lines or not lines[0].split()[0].upper().endswith("OFF"):
        raise ValueError(tr("This is not an OFF file."))
    first = lines[0].split()
    head_rest = first[1:]
    body = lines[1:]
    if len(head_rest) < 2:
        head_rest, body = body[0].split(), body[1:]
    nv, nf = int(head_rest[0]), int(head_rest[1])
    verts = np.array([ln.split()[:3] for ln in body[:nv]], dtype=float).reshape(-1, 3)
    tris = []
    for i, ln in enumerate(body[nv:nv + nf]):
        if i % PROGRESS_EVERY == 0:
            progress(i / max(nf, 1))
        w = ln.split()
        k = int(w[0])
        tris.extend(_fan([int(x) for x in w[1:1 + k]]))
    v, t = _soup(verts, tris)
    return MeshFile([MeshPart(_base(path), v, t)])


# ------------------------------------------------------------------ PLY

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2",
             "int16": "i2", "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4",
             "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8",
             "float64": "f8"}


def _ply_header(data):
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(tr("This is not a PLY file."))
    body = data.index(b"\n", end) + 1
    fmt, elements = None, []
    for line in data[:end].decode("ascii", "ignore").splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            elements.append((w[1], int(w[2]), []))
        elif w[0] == "property" and elements:
            if w[1] == "list":
                elements[-1][2].append((w[4], PLY_TYPES[w[2]], PLY_TYPES[w[3]]))
            else:
                elements[-1][2].append((w[2], PLY_TYPES[w[1]], None))
    return fmt, elements, body


def read_ply(path, progress):
    """Stanford PLY, ASCII or binary: the vertex and face elements as one part."""
    with open(path, "rb") as f:
        data = f.read()
    fmt, elements, at = _ply_header(data)
    verts, tris = np.zeros((0, 3)), []
    if fmt == "ascii":
        tokens = data[at:].split()
        pos = 0
        for name, count, props in elements:
            if all(lt is None for _, _, lt in props):
                rows = np.array(tokens[pos:pos + count * len(props)], dtype=float)
                rows = rows.reshape(count, len(props))
                pos += count * len(props)
                if name == "vertex":
                    cols = [p[0] for p in props]
                    verts = rows[:, [cols.index(c) for c in "xyz"]]
                continue
            for i in range(count):
                if name == "face" and i % PROGRESS_EVERY == 0:
                    progress(i / max(count, 1))
                for pname, _, lt in props:
                    if lt is None:
                        pos += 1
                        continue
                    k = int(tokens[pos])
                    idx = [int(x) for x in tokens[pos + 1:pos + 1 + k]]
                    pos += 1 + k
                    if name == "face" and pname in ("vertex_indices", "vertex_index"):
                        tris.extend(_fan(idx))
    else:
        end = "<" if fmt == "binary_little_endian" else ">"
        for name, count, props in elements:
            if all(lt is None for _, _, lt in props):
                dt = np.dtype([(p, end + t) for p, t, _ in props])
                rows = np.frombuffer(data, dtype=dt, count=count, offset=at)
                at += dt.itemsize * count
                if name == "vertex":
                    verts = np.stack([rows[c].astype(float) for c in "xyz"], axis=1)
                continue
            at, faces = _ply_binary_lists(data, at, end, count, props, name == "face", progress)
            tris.extend(faces)
    v, t = _soup(verts, tris)
    return MeshFile([MeshPart(_base(path), v, t)])


def _ply_binary_lists(data, at, end, count, props, faces, progress):
    """A binary element with list properties; returns the new offset and the triangles."""
    # the common case first: one list of three indices and nothing else per face
    if faces and len(props) == 1:
        _, ct, it = props[0]
        dt = np.dtype([("n", end + ct), ("i", end + it, 3)])
        if at + dt.itemsize * count <= len(data):
            rows = np.frombuffer(data, dtype=dt, count=count, offset=at)
            if count == 0 or (rows["n"] == 3).all():
                return at + dt.itemsize * count, rows["i"].astype(np.int64).tolist()
    tris = []
    for i in range(count):
        if faces and i % PROGRESS_EVERY == 0:
            progress(i / max(count, 1))
        for pname, t, lt in props:
            if lt is None:
                at += np.dtype(t).itemsize
                continue
            k = int(np.frombuffer(data, end + t, 1, at)[0])
            at += np.dtype(t).itemsize
            idx = np.frombuffer(data, end + lt, k, at).tolist()
            at += np.dtype(lt).itemsize * k
            if faces and pname in ("vertex_indices", "vertex_index"):
                tris.extend(_fan(idx))
    return at, tris


# ------------------------------------------------------------------ 3MF

NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
_MESH = re.compile(r"<((?:\w+:)?)mesh\b[^>]*>(.*?)</\1mesh>", re.S)
_VERT_FAST = re.compile(r'<(?:\w+:)?vertex\s+x="([^"]*)"\s+y="([^"]*)"\s+z="([^"]*)"')
_TRI_FAST = re.compile(r'<(?:\w+:)?triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"')
_VERT_ANY = re.compile(r"<(?:\w+:)?vertex\b([^>]*)>")
_TRI_ANY = re.compile(r"<(?:\w+:)?triangle\b([^>]*)>")
_ATTR = re.compile(r'([\w:]+)\s*=\s*"([^"]*)"')
#: the parts a slicer prints; modifiers, negative volumes and support blockers are not
SLICER_PRINTED = {"ModelPart", "normal_part", ""}
#: modifiers, which give the region they cover another filament when they set an extruder
SLICER_MODIFIERS = {"ParameterModifier", "modifier_part"}


def _3mf_matrix(text):
    """A 3MF transform "m00 m01 m02 m10 ... m32" (row vectors) as a 4x4 column matrix."""
    M = np.eye(4)
    if text:
        m = [float(x) for x in text.split()]
        if len(m) == 12:
            M[:3, :3] = np.array(m[:9]).reshape(3, 3).T
            M[:3, 3] = m[9:12]
    return M


def _3mf_mesh(text):
    """The vertices and triangles inside one <mesh> of a 3MF model."""
    v = _VERT_FAST.findall(text)
    if len(v) != len(_VERT_ANY.findall(text)):
        v = [tuple(dict(_ATTR.findall(a)).get(k, "0") for k in "xyz")
             for a in _VERT_ANY.findall(text)]
    t = _TRI_FAST.findall(text)
    if len(t) != len(_TRI_ANY.findall(text)):
        t = [tuple(dict(_ATTR.findall(a)).get(k, "0") for k in ("v1", "v2", "v3"))
             for a in _TRI_ANY.findall(text)]
    return (np.array(v, dtype=float).reshape(-1, 3),
            np.array(t, dtype=np.int64).reshape(-1, 3))


class _3mfModel:
    """One model file of a 3MF package: its objects, materials, build and unit."""

    def __init__(self, text):
        meshes = []

        def keep(m):
            meshes.append(m.group(2))
            return f'<{m.group(1)}mesh ref="{len(meshes) - 1}"/>'
        root = ET.fromstring(_MESH.sub(keep, text))
        self.unit = UNIT_MM.get(root.get("unit", "millimeter"), 1.0)
        self.meshes = meshes
        self.objects = {}
        self.materials = {}
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "basematerials":
                self.materials[el.get("id")] = [
                    _hex_color(b.get("displaycolor")) for b in el
                    if b.tag.rsplit("}", 1)[-1] == "base"]
            elif tag == "object":
                self.objects[el.get("id")] = el
        self.items = [el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "item"]


class _3mfReader:
    def __init__(self, path, progress):
        self.z = zipfile.ZipFile(path)
        self.progress = progress
        self.models = {}
        self.total = sum(i.file_size for i in self.z.infolist() if i.filename.endswith(".model"))
        self.done = 0
        self.parts = []

    def model(self, name):
        name = name.lstrip("/")
        if name not in self.models:
            text = self.z.read(name).decode("utf-8", "ignore")
            self.models[name] = _3mfModel(text)
            self.done += len(text)
            self.progress(0.8 * min(1.0, self.done / max(self.total, 1)))
        return self.models[name]

    def root_name(self):
        try:
            rels = ET.fromstring(self.z.read("_rels/.rels"))
            for r in rels:
                if r.get("Type", "").endswith("/3dmodel"):
                    return r.get("Target", "").lstrip("/")
        except (KeyError, ET.ParseError):
            pass
        return "3D/3dmodel.model"

    def walk(self, model, oid, M, top, names):
        """Collect the meshes under object `oid` of `model` as parts, through transform M."""
        obj = model.objects.get(oid)
        if obj is None or obj.get("type", "model") not in ("model", "solidsupport"):
            return
        name = obj.get("name") or ""
        names = names + [name] if name else names
        color = None
        if obj.get("pid") in model.materials:
            mats = model.materials[obj.get("pid")]
            k = int(obj.get("pindex", 0) or 0)
            color = mats[k] if 0 <= k < len(mats) else None
        for el in obj:
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "mesh":
                v, t = _3mf_mesh(model.meshes[int(el.get("ref"))])
                v = _transform(v * model.unit, M)
                self.parts.append(dict(top=top, oid=oid, model=model, verts=v, tris=t,
                                       name=names[-1] if names else "", color=color))
            elif tag == "components":
                for c in el:
                    if c.tag.rsplit("}", 1)[-1] != "component":
                        continue
                    sub = c.get(f"{{{NS_PROD}}}path")
                    sub_model = self.model(sub) if sub else model
                    C = _3mf_matrix(c.get("transform"))
                    C[:3, 3] *= model.unit
                    self.walk(sub_model, c.get("objectid"), M @ C, top, names)


def _prusa_volumes(z):
    """PrusaSlicer's parts: {object id: [(first, last, name, kind, extruder)]} and its object
    extruders, from Metadata/Slic3r_PE_model.config."""
    try:
        root = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
    except (KeyError, ET.ParseError):
        return {}, {}
    vols, obj_ext = {}, {}
    for o in root.iter("object"):
        meta = {m.get("key"): m.get("value") for m in o.findall("metadata")}
        obj_ext[o.get("id")] = meta.get("extruder")
        for v in o.iter("volume"):
            vm = {m.get("key"): m.get("value") for m in v.findall("metadata")}
            kind = vm.get("volume_type") or ("ParameterModifier" if vm.get("modifier") == "1"
                                             else "ModelPart")
            vols.setdefault(o.get("id"), []).append(
                (int(v.get("firstid", 0)), int(v.get("lastid", -1)), vm.get("name", ""),
                 kind, vm.get("extruder")))
    return vols, obj_ext


def _bambu_parts(z):
    """Bambu Studio / OrcaSlicer: {(object id, part id): (name, subtype, extruder)} and the
    object extruders, from Metadata/model_settings.config."""
    try:
        root = ET.fromstring(z.read("Metadata/model_settings.config"))
    except (KeyError, ET.ParseError):
        return {}, {}
    parts, obj_ext = {}, {}
    for o in root.iter("object"):
        meta = {m.get("key"): m.get("value") for m in o.findall("metadata")}
        obj_ext[o.get("id")] = meta.get("extruder")
        for p in o.iter("part"):
            pm = {m.get("key"): m.get("value") for m in p.findall("metadata")}
            parts[(o.get("id"), p.get("id"))] = (pm.get("name", ""),
                                                 p.get("subtype", "normal_part"),
                                                 pm.get("extruder"))
    return parts, obj_ext


def _slicer_colors(z):
    """The filament colors of a slicer project, extruder 1 first."""
    try:
        cfg = json.loads(z.read("Metadata/project_settings.config"))
        return [_hex_color(c) for c in cfg.get("filament_colour", [])]
    except (KeyError, ValueError, AttributeError):
        pass
    try:
        text = z.read("Metadata/Slic3r_PE.config").decode("utf-8", "ignore")
    except KeyError:
        return []
    found = {}
    for key in ("extruder_colour", "filament_colour"):
        m = re.search(rf"^;\s*{key}\s*=\s*(.*)$", text, re.M)
        found[key] = [_hex_color(c) for c in m.group(1).split(";")] if m else []
    ext, fil = found["extruder_colour"], found["filament_colour"]
    return [ext[i] if i < len(ext) and ext[i] else (fil[i] if i < len(fil) else None)
            for i in range(max(len(ext), len(fil)))]


def read_3mf(path, progress):
    """3MF: every mesh of every build item becomes a part, with the slicer's parts split
    out and its modifiers left out; colors from the filaments or the base materials."""
    out = MeshFile()
    try:
        r = _3mfReader(path, progress)
    except zipfile.BadZipFile:
        raise ValueError(tr("This 3MF file is damaged (it is not a valid zip file)."))
    with r.z:
        root = r.model(r.root_name())
        for item in root.items:
            sub = item.get(f"{{{NS_PROD}}}path")
            model = r.model(sub) if sub else root
            M = _3mf_matrix(item.get("transform"))
            M[:3, 3] *= model.unit
            r.walk(model, item.get("objectid"), M, item.get("objectid"), [])
        prusa, prusa_obj = _prusa_volumes(r.z)
        bambu, bambu_obj = _bambu_parts(r.z)
        colors = _slicer_colors(r.z)
    progress(0.9)

    def filament(*extruders):
        for e in extruders:
            try:
                k = int(e)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= len(colors) and colors[k - 1]:
                return colors[k - 1]
        return None

    def add(name, verts, tris, kind, ext, obj_ext, color, owner):
        if kind in SLICER_PRINTED:
            out.parts.append(MeshPart(name, verts, tris, filament(ext, obj_ext) or color,
                                      owner=owner))
        elif kind in SLICER_MODIFIERS and filament(ext):
            out.parts.append(MeshPart(name, verts, tris, filament(ext), modifier=True,
                                      owner=owner))
        else:
            skipped.add(name or kind)

    base = _base(path)
    skipped = set()
    for p in r.parts:
        name = p["name"] or base
        owner = f"{id(p['model'])}:{p['top']}"
        if p["model"] is root and p["oid"] in prusa:
            for first, last, vname, kind, ext in prusa[p["oid"]]:
                add(vname or name, p["verts"], p["tris"][first:last + 1], kind, ext,
                    prusa_obj.get(p["oid"]), p["color"], owner)
            continue
        pname, subtype, ext = bambu.get((p["top"], p["oid"]), (None, "normal_part", None))
        add(pname or name, p["verts"], p["tris"], subtype, ext, bambu_obj.get(p["top"]),
            p["color"], owner)
    if skipped:
        out.notes.append(tr("Left out (slicer modifiers): {names}").format(
            names=", ".join(sorted(skipped))))
    _number_duplicates(out.parts)
    return out


def _number_duplicates(parts):
    """Give parts with the same name a number: "Leg", "Leg 2", "Leg 3"."""
    seen = {}
    for p in parts:
        k = seen.get(p.name, 0) + 1
        seen[p.name] = k
        if k > 1:
            p.name = f"{p.name} {k}"


# ------------------------------------------------------------------ AMF

def _local(el):
    return el.tag.rsplit("}", 1)[-1]


def _child(el, name):
    for c in el:
        if _local(c) == name:
            return c
    return None


def _children(el, name):
    return [c for c in el if _local(c) == name]


def _amf_color(el):
    c = _child(el, "color") if el is not None else None
    if c is None:
        return None
    try:
        return _hex([float(_child(c, k).text) for k in ("r", "g", "b")])
    except (AttributeError, TypeError, ValueError):
        return None


def read_amf(path, progress):
    """Additive Manufacturing File (also zipped): every volume of every object is a part."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            data = z.read(z.namelist()[0])
    root = ET.fromstring(data)
    unit = UNIT_MM.get(root.get("unit", "millimeter"), 1.0)
    materials = {m.get("id"): _amf_color(m) for m in _children(root, "material")}
    out = MeshFile()
    objects = _children(root, "object")
    for i, obj in enumerate(objects):
        progress(i / max(len(objects), 1))
        mesh = _child(obj, "mesh")
        if mesh is None:
            continue
        coords = []
        for v in _children(_child(mesh, "vertices"), "vertex"):
            c = _child(v, "coordinates")
            coords.append([float(_child(c, k).text) for k in "xyz"])
        verts = np.array(coords, dtype=float).reshape(-1, 3) * unit
        name = _amf_name(obj) or f"{_base(path)} {obj.get('id', i + 1)}"
        volumes = _children(mesh, "volume")
        for vol in volumes:
            tris = [[int(_child(t, k).text) for k in ("v1", "v2", "v3")]
                    for t in _children(vol, "triangle")]
            vname = _amf_name(vol) if len(volumes) > 1 else None
            color = _amf_color(vol) or materials.get(vol.get("materialid")) or _amf_color(obj)
            out.parts.append(MeshPart(vname or name, verts, _soup(verts, tris)[1], color))
    _number_duplicates(out.parts)
    return out


def _amf_name(el):
    for m in _children(el, "metadata"):
        if m.get("type") == "name" and m.text:
            return m.text.strip()
    return None


# ------------------------------------------------------------------ glTF / GLB

GLTF_TYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
GLTF_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _gltf_load(path):
    with open(path, "rb") as f:
        data = f.read()
    folder = os.path.dirname(os.path.abspath(path))
    binary = None
    if data[:4] == b"glTF":
        length = struct.unpack("<I", data[8:12])[0]
        at, doc = 12, None
        while at + 8 <= min(length, len(data)):
            size, kind = struct.unpack("<II", data[at:at + 8])
            chunk = data[at + 8:at + 8 + size]
            if kind == 0x4E4F534A:
                doc = json.loads(chunk)
            elif kind == 0x004E4942:
                binary = chunk
            at += 8 + size
        if doc is None:
            raise ValueError(tr("This GLB file has no scene description."))
    else:
        doc = json.loads(data.decode("utf-8"))
    buffers = []
    for b in doc.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            buffers.append(binary or b"")
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            from urllib.parse import unquote
            with open(os.path.join(folder, unquote(uri)), "rb") as f:
                buffers.append(f.read())
    return doc, buffers


def _gltf_accessor(doc, buffers, k):
    acc = doc["accessors"][k]
    dt = np.dtype(GLTF_TYPES[acc["componentType"]])
    n = GLTF_SIZES[acc["type"]]
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, n), dtype=dt)
    view = doc["bufferViews"][acc["bufferView"]]
    buf = buffers[view["buffer"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or dt.itemsize * n
    arr = np.ndarray((count, n), dtype=dt, buffer=buf, offset=start,
                     strides=(stride, dt.itemsize))
    return np.array(arr)


def _gltf_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    T, R, S = np.eye(4), np.eye(4), np.eye(4)
    T[:3, 3] = node.get("translation", (0, 0, 0))
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    S[[0, 1, 2], [0, 1, 2]] = node.get("scale", (1, 1, 1))
    return T @ R @ S


def _strip(idx):
    t = [(idx[i], idx[i + 1], idx[i + 2]) if i % 2 == 0 else (idx[i + 1], idx[i], idx[i + 2])
         for i in range(len(idx) - 2)]
    return np.array(t, dtype=np.int64).reshape(-1, 3)


def read_gltf(path, progress):
    """glTF 2.0 (.gltf with its buffers, or .glb): every mesh instance in the scene is a
    part. glTF works in metres with Y up; Volura in millimetres with Z up."""
    doc, buffers = _gltf_load(path)
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    top = Y_UP_TO_Z_UP @ np.diag([1000.0, 1000.0, 1000.0, 1.0])
    out = MeshFile()
    stack = [(i, top) for i in reversed(roots)]
    seen = 0
    while stack:
        i, parent = stack.pop()
        node = nodes[i]
        M = parent @ _gltf_matrix(node)
        if "mesh" in node:
            seen += 1
            progress(seen / max(len(nodes), 1))
            mesh = doc["meshes"][node["mesh"]]
            vs, ts, color, skipped = [], [], None, False
            for prim in mesh.get("primitives", []):
                mode = prim.get("mode", 4)
                if mode not in (4, 5, 6) or "POSITION" not in prim.get("attributes", {}):
                    skipped = True
                    continue
                v = _gltf_accessor(doc, buffers, prim["attributes"]["POSITION"]).astype(float)
                idx = (_gltf_accessor(doc, buffers, prim["indices"]).ravel().astype(np.int64)
                       if "indices" in prim else np.arange(len(v)))
                if mode == 4:
                    t = idx[: len(idx) // 3 * 3].reshape(-1, 3)
                elif mode == 5:
                    t = _strip(idx)
                else:
                    t = np.array(_fan(idx.tolist()), dtype=np.int64).reshape(-1, 3)
                ts.append(t + sum(len(x) for x in vs))
                vs.append(v)
                if color is None and "material" in prim:
                    pbr = doc["materials"][prim["material"]].get("pbrMetallicRoughness", {})
                    if "baseColorFactor" in pbr:
                        color = _hex(pbr["baseColorFactor"])
            if vs:
                verts = _transform(np.concatenate(vs), M)
                tris = np.concatenate(ts)
                if np.linalg.det(M[:3, :3]) < 0:        # a mirroring node flips the winding
                    tris = tris[:, [0, 2, 1]]
                name = node.get("name") or mesh.get("name") or _base(path)
                out.parts.append(MeshPart(name, verts, tris, color))
        stack.extend((c, M) for c in reversed(node.get("children", [])))
    _number_duplicates(out.parts)
    return out


# ------------------------------------------------------------------ COLLADA

def read_dae(path, progress):
    """COLLADA (.dae, e.g. from SketchUp): every geometry instance in the scene is a part."""
    root = ET.parse(path).getroot()
    ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""

    def q(name):
        return f"{ns}{name}"

    unit_el = root.find(f"{q('asset')}/{q('unit')}")
    scale = float(unit_el.get("meter", 1.0)) * 1000.0 if unit_el is not None else 1000.0
    up = root.find(f"{q('asset')}/{q('up_axis')}")
    top = np.diag([scale, scale, scale, 1.0])
    if up is not None and (up.text or "").strip() == "Y_UP":
        top = Y_UP_TO_Z_UP @ top
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}

    def ref(url):
        return by_id.get((url or "").lstrip("#"))

    def material_color(mid):
        mat = ref("#" + mid) if mid else None
        eff = ref(mat.find(q("instance_effect")).get("url")) if (
            mat is not None and mat.find(q("instance_effect")) is not None) else None
        if eff is None:
            return None
        for d in eff.iter(q("diffuse")):
            c = d.find(q("color"))
            if c is not None and c.text:
                return _hex([float(x) for x in c.text.split()[:3]])
        return None

    def floats(src):
        arr = src.find(q("float_array"))
        vals = np.array(arr.text.split(), dtype=float) if arr is not None and arr.text else \
            np.zeros(0)
        acc = src.find(f"{q('technique_common')}/{q('accessor')}")
        stride = int(acc.get("stride", 3)) if acc is not None else 3
        return vals[: len(vals) // stride * stride].reshape(-1, stride)[:, :3]

    def geometry(geo):
        """(verts, tris, first material symbol) of a <geometry>; None without a mesh."""
        mesh = geo.find(q("mesh"))
        if mesh is None:
            return None
        vs, ts, symbol = [], [], None
        vert_src = {}
        for v in mesh.findall(q("vertices")):
            for inp in v.findall(q("input")):
                if inp.get("semantic") == "POSITION":
                    vert_src[v.get("id")] = inp.get("source")
        for prim in mesh:
            kind = _local(prim)
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = prim.findall(q("input"))
            stride = 1 + max((int(i.get("offset", 0)) for i in inputs), default=0)
            vin = next((i for i in inputs if i.get("semantic") == "VERTEX"), None)
            if vin is None:
                continue
            src = ref(vert_src.get(vin.get("source", "").lstrip("#"), vin.get("source")))
            if src is None:
                continue
            pos = floats(src)
            off = int(vin.get("offset", 0))
            if symbol is None:
                symbol = prim.get("material")
            polys = []
            if kind == "triangles":
                p = prim.find(q("p"))
                idx = np.array(p.text.split(), dtype=np.int64) if p is not None and p.text \
                    else np.zeros(0, np.int64)
                idx = idx[: len(idx) // stride * stride].reshape(-1, stride)[:, off]
                t = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            else:
                if kind == "polylist":
                    vc = prim.find(q("vcount"))
                    counts = [int(x) for x in vc.text.split()] if vc is not None and vc.text \
                        else []
                    p = prim.find(q("p"))
                    idx = np.array(p.text.split(), dtype=np.int64).reshape(-1, stride)[:, off]
                    at = 0
                    for c in counts:
                        polys.append(idx[at:at + c].tolist())
                        at += c
                else:
                    for p in prim.findall(q("p")):
                        polys.append(np.array(p.text.split(), dtype=np.int64)
                                     .reshape(-1, stride)[:, off].tolist())
                t = np.array([tri for poly in polys for tri in _fan(poly)],
                             dtype=np.int64).reshape(-1, 3)
            ts.append(t + sum(len(x) for x in vs))
            vs.append(pos)
        if not vs:
            return None
        return np.concatenate(vs), np.concatenate(ts), symbol

    scene_ref = root.find(f"{q('scene')}/{q('instance_visual_scene')}")
    scene = ref(scene_ref.get("url")) if scene_ref is not None else \
        root.find(f"{q('library_visual_scenes')}/{q('visual_scene')}")
    out = MeshFile()
    if scene is None:
        return out
    cache = {}
    total = max(1, len(list(root.iter(q("instance_geometry")))))

    def node_matrix(node):
        M = np.eye(4)
        for el in node:
            kind = _local(el)
            vals = [float(x) for x in (el.text or "").split()]
            T = np.eye(4)
            if kind == "matrix" and len(vals) == 16:
                T = np.array(vals).reshape(4, 4)
            elif kind == "translate" and len(vals) == 3:
                T[:3, 3] = vals
            elif kind == "scale" and len(vals) == 3:
                T = np.diag(vals + [1.0])
            elif kind == "rotate" and len(vals) == 4:
                axis = np.array(vals[:3])
                axis = axis / (np.linalg.norm(axis) or 1.0)
                a = np.radians(vals[3])
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                T[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
            else:
                continue
            M = M @ T
        return M

    def visit(node, parent, names, depth=0):
        if depth > 64:
            return
        M = parent @ node_matrix(node)
        name = node.get("name") or ""
        names = names + [name] if name else names
        for el in node:
            kind = _local(el)
            if kind == "instance_geometry":
                geo = ref(el.get("url"))
                if geo is None:
                    continue
                if geo.get("id") not in cache:
                    cache[geo.get("id")] = geometry(geo)
                g = cache[geo.get("id")]
                progress(len(out.parts) / total)
                if g is None:
                    continue
                verts, tris, symbol = g
                color = None
                for im in el.iter(q("instance_material")):
                    if im.get("symbol") == symbol:
                        color = material_color(im.get("target", "").lstrip("#"))
                if np.linalg.det(M[:3, :3]) < 0:
                    tris = tris[:, [0, 2, 1]]
                name = names[-1] if names else (geo.get("name") or _base(path))
                out.parts.append(MeshPart(name, _transform(verts, M), tris, color))
            elif kind == "node":
                visit(el, M, names, depth + 1)
            elif kind == "instance_node":
                target = ref(el.get("url"))
                if target is not None:
                    visit(target, M, names, depth + 1)

    for node in scene.findall(q("node")):
        visit(node, top, [])
    _number_duplicates(out.parts)
    return out
