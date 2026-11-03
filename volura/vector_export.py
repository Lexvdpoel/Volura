"""2D-vectorexport uit het 3D-model: geselecteerde vlakken, bovenaanzicht en doorsnede.

De logica hier is bewust vrij van Qt: uit het document worden tekeningen
(`volura.vector2d.Drawing`) gehaald -- de geselecteerde vlakken op ware grootte,
het bovenaanzicht van de volumes en een doorsnede op hoogte Z -- en die gaan naar
`volura.svg` of `volura.dxf`. De interface in `volura.app` bouwt hierop voort.

Alle maten zijn millimeters met Y omhoog. Bij het bovenaanzicht en de doorsnede
blijven x en y van het model behouden, zodat de tekening op de oorsprong van het
model uitgelijnd blijft; geselecteerde vlakken krijgen het lokale stelsel van hun
eigen vlak, en vlakken die niet in hetzelfde vlak liggen worden naast elkaar
gelegd met `GAP` ertussen.
"""
from __future__ import annotations

import numpy as np
import manifold3d as mf

from . import ops
from .geom import default_u, normalize
from .vector2d import Drawing, VPath, regions

GAP = 5.0            # mm tussen vormen die niet in hetzelfde vlak liggen
CLEAN_TOL = 1e-6     # mm: punten die dichter bij elkaar liggen, vallen samen
LAYER = "0"          # alles komt op laag "0"


# --------------------------------------------------------------------------
#  2D-punten opschonen
# --------------------------------------------------------------------------

def _dist_to_line(p, a, b):
    """Afstand van `p` tot de lijn door `a` en `b`."""
    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L <= 1e-12:
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))
    return abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / L


def clean_loop(pts, tol=CLEAN_TOL):
    """Een gesloten polylijn oplonen: dubbele punten en punten op een rechte lijn weg.

    `Manifold.slice` en `Manifold.project` geven een hoek vaak twee keer terug
    (eens per aangrenzende driehoek) en een punt midden op een rechte lijn is ook
    overbodig; zulke punten vallen weg. Een punt valt weg als het dichter dan
    `tol` bij de lijn van zijn buren ligt, of dichter dan `tol` bij het vorige
    punt. Er blijven minimaal drie punten over.
    """
    p = [np.asarray(q, dtype=float) for q in np.asarray(pts, dtype=float).reshape(-1, 2)]
    changed = True
    while changed and len(p) > 3:
        changed = False
        keep = []
        for i, cur in enumerate(p):
            prev = p[i - 1]
            nxt = p[(i + 1) % len(p)]
            if float(np.hypot(cur[0] - prev[0], cur[1] - prev[1])) <= tol:
                changed = True
                continue
            if _dist_to_line(cur, prev, nxt) <= tol:
                changed = True
                continue
            keep.append(cur)
        p = keep
    return np.ascontiguousarray(np.asarray(p, dtype=float).reshape(-1, 2))


# --------------------------------------------------------------------------
#  geselecteerde vlakken
# --------------------------------------------------------------------------

def plane_frame(n):
    """Rechtshandig lokaal stelsel `(n, u, v)` dat alleen van de normaal afhangt.

    Twee coplanaire vlakken krijgen zo hetzelfde assenstelsel en blijven dus op
    hun onderlinge plek staan. Een normaal die naar achteren wijst wordt omge-
    draaid (hetzelfde adres als `brep.plane_key`), zodat een vlak altijd van voren
    bekeken wordt: een platte tekening op de grond blijft boven boven liggen.
    """
    n = normalize(np.asarray(n, dtype=float))
    for c in n:
        if abs(c) > 1e-9:
            if c < 0.0:
                n = -n
            break
    u = np.asarray(default_u(n), dtype=float)
    u = u - n * float(u @ n)
    if float(np.linalg.norm(u)) < 1e-9:
        u = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    u = normalize(u)
    return n, u, np.cross(n, u)


def _plane_key(n, origin):
    """Sleutel van het vlak: dezelfde normaal en dezelfde afstand tot de oorsprong."""
    return tuple(round(float(c), 6) for c in n) + (round(float(np.dot(n, origin)), 4),)


def _face_loops(ents, fid):
    """(normaal, oorsprong, lussen) van een vlak, de lussen in zijn eigen 2D-stelsel."""
    face = ents.F[fid]
    n, u, v = plane_frame(face.normal)
    origin = n * float(np.dot(n, ents.V[face.loops[0][0]]))
    M = np.stack([u, v])
    loops = []
    for loop in face.loops:
        pts = np.array([ents.V[v_] for v_ in loop], dtype=float)
        q = clean_loop((pts - origin) @ M.T)
        if len(q) >= 3:
            loops.append(q)
    return n, origin, loops


def _shapes(loops, rule="nonzero"):
    """Lussen als gevulde vormen [buitenrand, gaten], via `vector2d.regions`."""
    paths = [VPath(np.ascontiguousarray(l, float), True, LAYER, True, rule, 1)
             for l in loops if len(l) >= 3]
    return regions(paths) if paths else []


def _bbox(loops):
    """(xmin, ymin, xmax, ymax) van een lijst lussen, of None als er geen zijn."""
    pts = [np.asarray(l, dtype=float).reshape(-1, 2) for l in loops if len(l)]
    if not pts:
        return None
    P = np.concatenate(pts)
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max()), float(P[:, 1].max()))


def _layout(bboxes, gap=GAP):
    """Verschuiving per groep: de eerste blijft liggen, de rest schuift eronder.

    De eerste groep bepaalt de plek van de tekening; de andere groepen gaan in
    rijen onder elkaar, met `gap` mm tussenruimte en om de beurt naast elkaar.
    """
    if len(bboxes) < 2 or any(b is None for b in bboxes):
        return [(0.0, 0.0) for _ in bboxes]
    shifts = [(0.0, 0.0)]
    left, bottom, right, _top = bboxes[0]
    wrap = max(right - left, max(b[2] - b[0] for b in bboxes[1:]))
    x, y = right + gap, bottom
    for b in bboxes[1:]:
        w, h = b[2] - b[0], b[3] - b[1]
        if x + w > left + wrap:                       # nieuwe rij, lager
            x, y = left, bottom - gap - h
            bottom = y
        shifts.append((x - b[0], y - b[1]))
        x += w + gap
    return shifts


def drawing_from_faces(doc, selection=None, gap=GAP) -> Drawing:
    """Elk geselecteerd vlak op ware grootte in zijn eigen vlak.

    `selection` is een lijst selectie-items zoals `Document.selection` het doet,
    `("f", ctx, fid)`; zonder argument is de actieve selectie het uitgangspunt.
    Vlakken in hetzelfde vlak houden hun onderlinge plek, vlakken in verschillende
    vlakken worden naast elkaar gelegd met `gap` mm ertussen. Een vlak met gaten
    geeft één vorm: de buitenrand en de gaten horen bij één bronelement (`src`).
    """
    items = doc.selection if selection is None else list(selection)
    order, by_plane = [], {}
    seen = set()
    for item in items:
        try:
            kind, ctx, fid = item
        except (TypeError, ValueError):
            continue
        if kind != "f" or (ctx, fid) in seen:
            continue
        ents = doc.ents(ctx)
        if ents is None or fid not in ents.F:
            continue
        seen.add((ctx, fid))
        n, origin, loops = _face_loops(ents, fid)
        if not loops:
            continue
        key = _plane_key(n, origin)
        if key not in by_plane:
            by_plane[key] = []
            order.append(key)
        by_plane[key].extend(loops)

    drawing = Drawing()
    groups = [g for g in (_shapes(by_plane[k]) for k in order) if g]
    boxes = [_bbox([l for shape in g for l in shape]) for g in groups]
    src = 0
    for group, (dx, dy) in zip(groups, _layout(boxes, gap)):
        for shape in group:
            src += 1
            for loop in shape:
                drawing.add_path(np.asarray(loop, dtype=float) + np.array([dx, dy]),
                                 closed=True, layer=LAYER, filled=True, rule="nonzero", src=src)
    return drawing


# --------------------------------------------------------------------------
#  bovenaanzicht en doorsnede
# --------------------------------------------------------------------------

def _sections(doc, only_selected, what, z=0.0):
    """(doorsnedes per onderdeel, waarschuwingen) voor `project` of `slice`."""
    out, warn = [], []
    for part in ops.parts(doc, only_selected=bool(only_selected)):
        if part.manifold is None:
            warn.append(f"{part.name} overgeslagen: {part.problem}.")
            continue
        try:
            cs = part.manifold.project() if what == "top" else part.manifold.slice(float(z))
        except Exception as e:                          # een kapot volume mag nooit crashen
            warn.append(f"{part.name} overgeslagen: {e}.")
            continue
        if cs.is_empty():
            continue
        polys = [clean_loop(q) for q in cs.to_polygons()]
        polys = [np.ascontiguousarray(q, dtype=float) for q in polys if len(q) >= 3]
        if not polys:
            continue
        if what == "top":
            # een projectie is vaak zelfkruisend; de positieve vulregel maakt er
            # pas een normale vorm van (zoals Manifold.project aanbeveelt)
            cs = mf.CrossSection(polys, mf.FillRule.NonZero)
            if cs.is_empty():
                continue
        out.append(cs)
    return out, warn


def _drawing_from_sections(sections, warn) -> Drawing:
    """De vereniging van de doorsnedes als gesloten paden op laag "0"."""
    drawing = Drawing()
    for text in warn:
        drawing.warn(text)
    if not sections:
        return drawing
    whole = sections[0] if len(sections) == 1 else mf.CrossSection.batch_boolean(
        sections, mf.OpType.Add)
    if whole.is_empty():
        return drawing
    polys = [clean_loop(q) for q in whole.to_polygons()]
    for i, shape in enumerate(_shapes(polys, "evenodd"), 1):
        for loop in shape:
            drawing.add_path(np.asarray(loop, dtype=float), closed=True, layer=LAYER,
                             filled=True, rule="nonzero", src=i)
    return drawing


def drawing_top(doc, only_selected=False) -> Drawing:
    """Bovenaanzicht (omtrek) van de volumes: `Manifold.project()` per onderdeel."""
    sections, warn = _sections(doc, only_selected, "top")
    return _drawing_from_sections(sections, warn)


def drawing_section(doc, z=0.0, only_selected=False) -> Drawing:
    """Doorsnede op hoogte `z`: `Manifold.slice(z)` per onderdeel."""
    sections, warn = _sections(doc, only_selected, "section", z)
    return _drawing_from_sections(sections, warn)


def section_z_default(doc):
    """Het midden van het model in de hoogte: de standaardwaarde voor een doorsnede."""
    box = doc.model_bbox()
    if box is None:
        return 0.0
    return 0.5 * (float(box[0][2]) + float(box[1][2]))
