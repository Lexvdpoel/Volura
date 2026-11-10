"""Reading and writing Volura documents, importing meshes, writing STL, OBJ and 3MF."""
from __future__ import annotations

import json
import os
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass

import numpy as np
import manifold3d as mf

from . import __version__
from .brep import Entities, Face
from .geom import Plane, manifold_from_mesh, merge_mesh, mesh_arrays
from .i18n import tr
from .linking import Link, Vent
from .model import ROOT, Document, Group, Guide
from .mesh_formats import STL_RECORD, read_mesh, read_stl  # noqa: F401 (read_stl: API)
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
        "links": [l.to_dict() for l in doc.links],
        "vents": [v.to_dict() for v in doc.vents],
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
    doc.links = [Link.from_dict(d) for d in data.get("links") or []]   # nor links
    doc.vents = [Vent.from_dict(d) for d in data.get("vents") or []]   # nor vent patterns
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
    rec = np.zeros(len(t), dtype=STL_RECORD)
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


def _drop_duplicate_triangles(verts, tris):
    """Drop triangles that repeat an earlier one; return ``(tris, removed)``.

    Two triangles are duplicates when they have the same three merged corners in the same
    cyclic order: ``(a, b, c)``, ``(b, c, a)`` and ``(c, a, b)`` are the same triangle, while
    the reversed ``(a, c, b)`` is not and stays. Exactly one of every group of duplicates
    remains. Fully vectorised with numpy, so it keeps up with the millions of triangles a
    large STL can hold.
    """
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    _, merged = merge_mesh(verts, tris)
    a, b = merged[:, 0], merged[:, 1]
    low = merged.min(axis=1, keepdims=True)          # rotate each triangle so its smallest
    canon = np.where(a[:, None] == low, merged,      # corner comes first: that is the same
                     np.where(b[:, None] == low,     # tuple for every cyclic rotation and a
                              merged[:, [1, 2, 0]],  # different one for a reversed triangle
                              merged[:, [2, 0, 1]]))
    order = np.lexsort((canon[:, 2], canon[:, 1], canon[:, 0]))
    s = canon[order]
    first = np.ones(len(s), dtype=bool)
    first[1:] = np.any(s[1:] != s[:-1], axis=1)      # the first of every group of equals
    keep = np.zeros(len(tris), dtype=bool)
    keep[order[first]] = True
    return tris[keep], int((~keep).sum())


def import_stl_report(path):
    """Return ``(Manifold, removed)``: the mesh and how many duplicate triangles were dropped.

    Raises ValueError when the mesh is not watertight.
    """
    v, t = read_stl(path)
    if not len(t):
        raise ValueError(tr("The STL file contains no triangles."))
    vol = closed_volume(v, t)
    if vol.manifold is None:
        raise ValueError(tr("The STL file is not watertight ({status}). "
                            "Only closed volumes can be imported.").format(status=vol.status))
    return vol.manifold, vol.duplicates


def import_stl(path):
    """Return a Manifold, raising ValueError when the mesh is not watertight."""
    return import_stl_report(path)[0]


@dataclass
class Volume:
    """A closed volume read from a mesh file; `manifold` is None when the part is not closed."""

    name: str
    manifold: object
    color: str | None = None
    duplicates: int = 0          # duplicate triangles dropped
    loose: int = 0               # loose triangles dropped (see _drop_fins)
    status: object = None
    repair: object = None        # a RepairReport when the part had to be repaired


def _edge_use(tris):
    """How many triangles use each edge of every triangle: an (M, 3) array."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    _, inverse, count = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    return count[inverse.ravel()].reshape(3, -1).T


def _drop_fins(tris, rounds=8):
    """Drop loose "fins": triangles with an open edge whose other edges are used by three or
    more triangles. That is a stray triangle folded against a closed surface, the most common
    flaw in a mesh a slicer still prints; without it the surface is closed again. Returns
    ``(tris, dropped)``; a mesh without such triangles comes back as it was."""
    dropped = 0
    for _ in range(rounds):
        use = _edge_use(tris)
        bad = use != 2
        if not bad.any():
            break
        fin = (use == 1).any(axis=1) & ((use == 1) | (use >= 3)).all(axis=1)
        if not fin.any():
            break
        dropped += int(fin.sum())
        tris = tris[~fin]
    return tris, dropped


def closed_volume(verts, tris, name=""):
    """The triangle soup as a :class:`Volume`; its manifold is None when it is not one
    closed volume (``status`` then says why), also after dropping duplicate triangles and
    loose fins."""
    if not len(tris):
        return Volume(name, None, status="Empty")
    tris, duplicates = _drop_duplicate_triangles(verts, tris)
    m = manifold_from_mesh(verts, tris)
    loose = 0
    if m.status() == mf.Error.NotManifold:
        v, t = merge_mesh(verts, tris)
        t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
        t, loose = _drop_fins(t)
        if loose:
            m = manifold_from_mesh(v, t)
    if m.status() != mf.Error.NoError or m.is_empty():
        return Volume(name, None, duplicates=duplicates, status=m.status())
    return Volume(name, m, duplicates=duplicates, loose=loose, status=m.status())


def repaired_volume(verts, tris, name="", progress=None):
    """The triangle soup as a :class:`Volume` after `repair.repair_mesh`; its manifold is
    None when even the repair could not close it."""
    from .repair import repair_mesh
    m, report = repair_mesh(verts, tris, progress)
    return Volume(name, m, status=report.status, repair=report)


def import_mesh_report(path, progress=None, repair=True):
    """Every closed volume in a mesh file (STL, OBJ, 3MF, ...): ``(volumes, notes)``.

    ``volumes`` is a list of :class:`Volume` (color None where the file gives none);
    ``notes`` are translated remarks on what was left out. A part that is not closed on its
    own is tried together with the other open parts (an object can be split over several
    meshes, e.g. one per material). With `repair` what still is not closed is repaired
    (see `volura.repair`): all open parts together, else one by one. What then still is
    not closed is left out. Raises ValueError when nothing closed is left.
    """
    report = progress or (lambda fraction: None)
    meshes = read_mesh(path, lambda f: report(0.7 * f))
    if not meshes.parts:
        raise ValueError(tr("The file contains no triangles."))
    notes = list(meshes.notes)
    volumes, open_parts, status = [], [], None
    modifiers, owned = [], {}
    total = sum(len(p.tris) for p in meshes.parts)
    done = 0
    for p in meshes.parts:
        report(0.7 + 0.3 * done / max(total, 1))
        done += len(p.tris)
        vol = closed_volume(p.verts, p.tris, p.name)
        vol.color = p.color
        if p.modifier:
            modifiers.append((p, vol))
        elif vol.manifold is None:
            open_parts.append(p)
            status = vol.status
        else:
            volumes.append(vol)
            owned.setdefault(p.owner, []).append(vol)
    if len(open_parts) > 1:
        verts = np.concatenate([p.verts for p in open_parts])
        offsets = np.cumsum([0] + [len(p.verts) for p in open_parts[:-1]])
        tris = np.concatenate([p.tris + o for p, o in zip(open_parts, offsets)])
        vol = closed_volume(verts, tris, os.path.splitext(os.path.basename(path))[0])
        if vol.manifold is not None:
            colors = {p.color for p in open_parts}
            vol.color = colors.pop() if len(colors) == 1 else None
            volumes.append(vol)
            open_parts = []
    if open_parts and repair:
        report(0.85)
        whole = os.path.splitext(os.path.basename(path))[0]
        groups = [open_parts] + ([[p] for p in open_parts] if len(open_parts) > 1 else [])
        for k, group in enumerate(groups):
            if not any(p in open_parts for p in group):
                continue
            verts = np.concatenate([p.verts for p in group])
            offsets = np.cumsum([0] + [len(p.verts) for p in group[:-1]])
            tris = np.concatenate([p.tris + o for p, o in zip(group, offsets)])
            vol = repaired_volume(verts, tris, whole if len(group) > 1 else group[0].name,
                                  lambda f, k=k: report(0.85 + 0.15 * (k + f) / len(groups)))
            if vol.manifold is None:
                status = vol.status
                continue
            colors = {p.color for p in group}
            vol.color = colors.pop() if len(colors) == 1 else None
            volumes.append(vol)
            open_parts = [p for p in open_parts if p not in group]
    # a color modifier of a slicer becomes the part of its object it covers, in its color,
    # cut out of the parts below it; as in the slicer a later modifier wins where they overlap
    claimed = {}
    regions = []
    for p, vol in reversed(modifiers):
        inside = None
        if vol.manifold is not None and owned.get(p.owner):
            inside = union_all([v.manifold for v in owned[p.owner]]) ^ vol.manifold
            if p.owner in claimed:
                inside = inside - claimed[p.owner]
        if inside is None or inside.is_empty():
            notes.append(tr("Color modifier left out: {name}").format(name=p.name))
            continue
        claimed[p.owner] = inside if p.owner not in claimed else claimed[p.owner] + inside
        vol.manifold = inside
        regions.append(vol)
    for owner, cut in claimed.items():
        for v in owned[owner]:
            v.manifold = v.manifold - cut
    volumes = [v for v in volumes if not v.manifold.is_empty()] + regions[::-1]
    if not volumes:
        raise ValueError(tr("The file is not watertight ({status}). "
                            "Only closed volumes can be imported.").format(status=status))
    if open_parts:
        notes.append(tr("Not closed, left out: {names}").format(
            names=", ".join(p.name for p in open_parts)))
    report(1.0)
    return volumes, notes


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


# ------------------------------------------------------------------ OBJ

def obj_object_name(name):
    """A part name as an OBJ object name: one word, whitespace becomes underscores."""
    return "_".join(str(name or "").split()) or "Volura"


def obj_material(color):
    """'#rrggbb' as its OBJ material: ``("color_d23b3b", (0.824, 0.231, 0.231))``.

    The name holds no spaces, so every reader finds it back in the .mtl; a part without
    a color gets the grey of the viewport.
    """
    t = re.sub(r"[^0-9a-f]", "", str(color or "").lower())
    t = t[:6] if len(t) >= 6 else "c8c8c8"
    return f"color_{t}", tuple(int(t[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


def export_obj(path, part_list):
    """Wavefront OBJ with a .mtl next to it: one object per part, each in its own color.

    The vertex indices run over the whole file (1-based, as OBJ wants it), so every
    object refers to its own corners; the .mtl has one material per color in use.
    Millimetres, like every Volura export. Returns the number of triangles.
    """
    parts = [p for p in part_list if p.manifold is not None]
    if not parts:
        raise ValueError(tr("There is nothing to export."))
    # mtllib takes the rest of the line as one name in some readers and only the first word
    # in others: a name without spaces works in both
    mtl_name = obj_object_name(os.path.splitext(os.path.basename(path))[0]) + ".mtl"
    colors = {}                                   # material name -> (r, g, b)
    lines = [f"# {os.path.basename(path)} - exported by Volura {__version__}",
             f"mtllib {mtl_name}"]
    first = 1                                     # the first vertex index of this object
    tris = 0
    for p in parts:
        v, t = mesh_arrays(p.manifold)
        name, rgb = obj_material(p.color)
        colors.setdefault(name, rgb)
        lines.append(f"o {obj_object_name(p.name)}")
        lines.append(f"usemtl {name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
        lines += [f"f {a + first} {b + first} {c + first}" for a, b, c in t]
        first += len(v)
        tris += len(t)
    mtl = "".join(f"newmtl {name}\nKd {r:.6f} {g:.6f} {b:.6f}\nKa 0 0 0\nd 1\nillum 1\n"
                  for name, (r, g, b) in colors.items())

    def write_obj(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def write_mtl(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(mtl)

    atomic_write(path, write_obj)
    atomic_write(os.path.join(os.path.dirname(path), mtl_name), write_mtl)
    return tris


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
    from xml.sax.saxutils import escape   # xml.sax is heavy: only this writer needs it
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
