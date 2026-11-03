"""Vectorbestanden (SVG, DXF) importeren in de 3D-omgeving.

De logica hier is bewust vrij van Qt: ze leest een tekening (via `volura.svg` of
`volura.dxf`), schaalt en verankert haar, en zet haar om naar vlakken, randen of
een geextrudeerd volume in het B-rep-model. Zo is alles zonder venster te testen.
De interface in `volura.app` en het gereedschap in `volura.tools` bouwen hierop.

Alle 2D-maten zijn millimeters met Y omhoog (het gedeelde model van
`volura.vector2d`); `plane` is het vlak waarop geplaatst wordt (grond of een vlak
van een model) met een rechtshandig lokaal stelsel `(u, v, n)`.
"""
from __future__ import annotations

import os

import numpy as np
import manifold3d as mf

from . import dxf, svg
from .geom import extrude_cs
from .vector2d import regions

VECTOR_EXTENSIONS = (".svg", ".dxf")


def is_vector(path) -> bool:
    """True als `path` een vectorbestand is (SVG of DXF, ook met hoofdletters)."""
    return os.path.splitext(str(path))[1].lower() in VECTOR_EXTENSIONS


def load_vector(path):
    """Een SVG- of DXF-bestand als 2D-tekening; de keuze volgt de extensie."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".svg":
        return svg.read_svg(path)
    if ext == ".dxf":
        return dxf.read_dxf(path)
    raise ValueError("Alleen SVG- en DXF-bestanden kunnen als vector worden geïmporteerd.")


def load_svg_text(text):
    """SVG-tekst (bijv. van het klembord) als 2D-tekening."""
    return svg.read_svg(text)


def prepare(drawing, scale=1.0, target_width=None, anchor="center"):
    """Schaal en veranker een tekening; geef (regios, open_padenvoort.

    `target_width` (in mm) wint van `scale`: de breedte van de omhullende wordt
    dan exact `target_width` en de hoogte schaalt in verhouding mee. Daarna wordt
    de tekening verschoven zodat het ankerpunt -- het midden ("center") of de
    linkerbenedenhoek ("lower_left") van de omhullende -- in de oorsprong ligt.

    De gesloten vormen komen terug als regios (via `vector2d.regions`, gaten
    volgens de vulregel van de bron) en de open paden als een lijst `(N, 2)`-arrays.
    """
    b = drawing.bounds()
    if b is None:
        return [], []
    xmin, ymin, xmax, ymax = b
    w, h = xmax - xmin, ymax - ymin
    factor = float(scale) if scale else 1.0
    if target_width is not None and w > 0.0:
        factor = float(target_width) / w
    if str(anchor) == "lower_left":
        ax, ay = xmin, ymin
    else:
        ax, ay = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    M = np.array([[factor, 0.0, -factor * ax],
                  [0.0, factor, -factor * ay],
                  [0.0, 0.0, 1.0]])
    scaled = drawing.transformed(M)
    closed = [p for p in scaled.paths if p.closed and len(p.pts) >= 3]
    open_paths = [np.asarray(p.pts, dtype=float) for p in scaled.paths
                  if not p.closed and len(p.pts) >= 2]
    return regions(closed), open_paths


def place_faces(ents, plane, regios, open_paden, offset):
    """Maak van de regios vlakken en van de open paden randen op `plane`.

    `regios` is de lijst vormen uit `prepare` (eerste lus buitenrand, rest gaten);
    elk wordt één vlak met `ents.add_shape`. `open_paden` worden randen met
    `ents.add_segment`. `offset` is de 2D-verschuiving in het lokale stelsel van
    `plane` (waar het ankerpunt terechtkomt). Geeft de nieuwe vlak-ids terug.
    """
    offset = np.asarray(offset, dtype=float).reshape(2)
    created = []
    for shape in regios:
        loops = [np.asarray(loop, dtype=float) + offset for loop in shape if len(loop) >= 3]
        if loops:
            created += list(ents.add_shape(plane, loops) or [])
    for path in open_paden or ():
        pts = np.asarray(path, dtype=float) + offset
        if len(pts) < 2:
            continue
        W = plane.to_world(pts)
        for i in range(len(W) - 1):
            ents.add_segment(W[i], W[i + 1])
    return created


def cross_section(regios, offset=(0.0, 0.0)):
    """Eén doorsnede van alle regios (gaten blijven gaten), verschoven met `offset`."""
    offset = np.asarray(offset, dtype=float).reshape(2)
    sections = []
    for shape in regios:
        loops = [np.ascontiguousarray(np.asarray(loop, dtype=float) + offset, float)
                 for loop in shape if len(loop) >= 3]
        if not loops:
            continue
        cs = mf.CrossSection(loops, mf.FillRule.NonZero)
        if not cs.is_empty():
            sections.append(cs)
    if not sections:
        return mf.CrossSection()
    if len(sections) == 1:
        return sections[0]
    return mf.CrossSection.batch_boolean(sections, mf.OpType.Add)


def place_extruded(doc, plane, regios, offset, height, color, name):
    """Extrudeer de regios tot één groep loodrecht op `plane`, langs de normaal.

    De hoogte gaat langs de normaal van `plane` (naar buiten bij een modelvlak);
    een negatieve hoogte drukt in (aan de andere kant van het vlak). Het resultaat
    is altijd één losse groep: het onderliggende model blijft onveranderd. Geeft de
    nieuwe groep terug, of None als er niets te extruderen viel.
    """
    from . import ops
    cs = cross_section(regios, offset)
    if cs.is_empty():
        return None
    height = float(height)
    if abs(height) < 1e-9:
        return None
    man = extrude_cs(cs, plane, min(0.0, height), max(0.0, height))
    if man.is_empty():
        return None
    return ops.group_from_manifold(doc, man, color, name)
