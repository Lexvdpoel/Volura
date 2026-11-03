"""Dozen met vingerlassen: de rekenkern voor het lasersnijden van dozen.

Geen Qt in dit bestand. Het werkt op het gedeelde 2D-model uit `volura/vector2d.py`
en de plaatinstellingen van het werkblad, en is daarom los te testen en in het werkblad,
de export en het 3D-model weer te gebruiken.

Een **vingerlas** is een verlijmde hoek van twee panelen van dezelfde dikte: het ene
paneel heeft uitsteeksels (de mannelijke kant), het andere evenveel uitsparingen
(de vrouwelijke kant) op dezelfde plaatsen. `finger_edge` geeft die tandenrand als
polylijn in een eigen vlak: `x` langs de rand van 0 tot `length`, `y` van 0 (de
nominaal rand) tot `t` (de diepte van de tand). Bij de mannelijke kant staat de tand
omhoog (diepte `t`), bij de vrouwelijke kant staat daar de uitsparing, zodat de twee
in elkaar passen. Een laser haalt langs de snijlijn de halve kerf weg; de vingers gaan
daarom `kerf / 2` breder en de uitsparingen `kerf / 2` smaller, zodat de las strak past.

Een **paneel** (`Panel`) is de snijcontour van één vlak van de doos, plus de matrix
die het op zijn plek in de doos zet: `position3d` voert het lokale vlak `(u, v)` met
`u` naar rechts en `v` omhoog over in de wereld van het 3D-model, en de plaats van het
paneel is `position3d @ (u, v, 0..thickness)`. `panels_to_drawing` legt de contouren op
de snijlaag van een werkblad (met de paneelnaam klein gegraveerd in het midden),
`panels_to_manifolds` zet ze om in geëxtrudeerde volumes voor in het 3D-model.

De hoeken: waar drie panelen samenkomen mag geen gat en geen overlap zitten. Daarom
bezit één paneel elke hoek. De bodem bezit de onderste laag van de doos (haar randen
zijn overal vrouwelijk en de zijkanten hebben aan hun onderkant vingers), de voor- en
achterkant bezitten de verticale hoekkolommen (mannelijk op hun zijkanten, de linker-
en rechterkant zijn daar vrouwelijk) en het deksel bezit de bovenste laag (mannelijk
op zijn vier randen, de zijkanten zijn daar vrouwelijk). De vingers van de linker- en
rechterkant aan hun onderkant beginnen pas voorbij de hoekkolommen, want daar heeft de
bodem al het alleenrecht.

Aan de onderkant van de wanden staat bovendien `ends = SLOT`: het eerste en het laatste
vak van die las is geen vinger maar materiaal van de bodem. Zo houdt de bodem op elke
hoek een plein van `t` × `t` dat met twee volle randen aan de plaat vastzit. Zonder die
fase raken de uitsparingen van twee loodrechte randen elkaar in precies één punt,
kruist de omtrek van de bodem zichzelf en hangt het hoekplein aan niets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from .vector2d import Drawing, VPath

MALE = "male"
FEMALE = "female"
TOOTH = "tooth"         # het eerste en het laatste vak van een las is een vinger
SLOT = "slot"           # ... en is het tegengestelde, waardoor de hoek een plein houdt

#: vingerbreedte als veelvoud van de materiaaldikte (standaard van de dialoog)
FINGER_FACTOR = 2.5
#: dikte om aan te nemen als de plaatinstellingen geen materiaaldikte kennen (mm)
DEFAULT_THICKNESS = 3.0
MIN_FINGERS = 3          # een las zonder vingers houdt niet
LID_KINDS = ("none", "closed")
LID_LABELS = {"none": "Geen deksel", "closed": "Vast deksel"}

#: de Nederlandse namen van de panelen, zoals het 3D-model ze toont
PANEL_LABELS = {"bodem": "bodem", "voor": "voor", "achter": "achter",
                "links": "links", "rechts": "rechts", "deksel": "deksel"}


# --------------------------------------------------------------------------
#  de tandenrand
# --------------------------------------------------------------------------

def finger_count(length, finger_w) -> int:
    """Het aantal vingers: het grootste oneven getal n met length / n >= finger_w.

    Een oneven aantal zorgt dat beide einden van de rand een vinger (of een
    uitsparing) hebben; dan sluit de las van een paneel aan op de hoek. Is de rand te
    kort voor drie vingers, dan worden het er toch drie: een las met één vinger houdt
    niets.
    """
    L = abs(float(length))
    fw = abs(float(finger_w))
    if L <= 0.0 or fw <= 0.0:
        return MIN_FINGERS
    n = int(math.floor(L / fw))
    if n % 2 == 0:
        n -= 1
    return max(MIN_FINGERS, n)


def _step_line(length, depth, spans) -> np.ndarray:
    """De polylijn van een stapfunctie: `depth` hoog binnen `spans`, daarbuiten 0."""
    L, h = float(length), float(depth)
    keep = sorted((max(0.0, min(L, float(a))), max(0.0, min(L, float(b))))
                  for a, b in spans if float(b) > float(a))
    merged: list[list[float]] = []
    for a, b in keep:
        if merged and a <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    pts: list[tuple[float, float]] = []
    x = 0.0
    for a, b in merged:
        if a > x:
            pts += [(x, 0.0), (a, 0.0)]
        pts += [(a, h), (b, h)]
        x = b
    if x < L:
        pts += [(x, 0.0), (L, 0.0)]
    if not pts:
        pts = [(0.0, 0.0), (L, 0.0)]
    return np.ascontiguousarray(np.asarray(pts, dtype=float))


def finger_edge(length, t, finger_w, male, kerf=0.0, ends="tooth") -> np.ndarray:
    """De 2D-zaagtandrand van een lasverbinding, als polylijn ``(N, 2)``.

    De rand loopt langs de x-as van 0 tot `length`; `y` is 0 of `t` (de diepte van de
    tand). Het getal vingers komt uit `finger_count` en is oneven, dus beide einden van
    de rand zijn gelijk en passen de twee kanten van dezelfde lengte precies in elkaar:
    het gebied tussen de mannelijke rand en y = 0 en het gebied tussen de vrouwelijke
    rand en y = 0 vullen elkaar.

    `ends` is de fase aan de einden van de rand. Met "tooth" is het eerste en het laatste
    vak een vinger (mannelijk) of een uitsparing (vrouwelijk); met "slot" is dat juist
    het tegengestelde, zodat de mannelijke kant daar materiaal overlaat voor de hoek.
    Twee kanten die in elkaar moeten passen, gebruiken dezelfde `ends`.

    `kerf` is de breedte van de snedespleet: de vingers worden er `kerf / 2` breder op
    elke kant en de uitsparingen `kerf / 2` smaller, zodat de las na het snijden strak
    past. Bij `kerf = 0` zijn de twee kanten elkaars spiegelbeeld in de tandbreedte.
    """
    L = abs(float(length))
    h = abs(float(t))
    k = max(0.0, float(kerf or 0.0))
    if L <= 0.0:
        return np.array([[0.0, 0.0]], dtype=float)
    n = finger_count(L, finger_w)
    step = L / n
    first = 0 if str(ends or "tooth").strip().lower() != "slot" else 1
    teeth = [(i * step, (i + 1) * step) for i in range(first, n, 2)]   # vinger of uitsparing
    if male:
        spans = [(a - 0.5 * k, b + 0.5 * k) for a, b in teeth]        # vingers breder
    else:
        spans, x = [], 0.0                                            # uitsparingen smaller
        for a, b in teeth:
            a, b = a + 0.5 * k, b - 0.5 * k
            if a > x:
                spans.append((x, a))
            x = max(x, b)
        if x < L:
            spans.append((x, L))
    return _step_line(L, h, spans)


def tooth_area(edge, length) -> float:
    """Het gebied tussen de tandenrand `edge` en de nominaal rand y = 0 (mm²)."""
    L = abs(float(length))
    pts = _close_teeth(edge, L)
    return float(abs(_area(pts)))


def _close_teeth(edge, length) -> np.ndarray:
    """De tandenrand gesloten met de nominaal rand: het gebied van de tanden."""
    pts = np.asarray(edge, dtype=float)
    if not len(pts):
        return np.zeros((0, 2), dtype=float)
    out = [pts[0]]
    for q in pts[1:]:
        if abs(float(q[0]) - float(out[-1][0])) > 1e-12 or \
           abs(float(q[1]) - float(out[-1][1])) > 1e-12:
            out.append(q)
    out.append([float(out[-1][0]), 0.0])
    out.append([float(out[0][0]), 0.0])
    out.append([float(out[0][0]), float(out[0][1])])
    return np.asarray(out, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------
#  panelen
# --------------------------------------------------------------------------

@dataclass(eq=False)
class Panel:
    """Eén vlak van de doos: de snijcontour en de plek die hij in het model krijgt."""

    name: str
    outline: VPath
    position3d: np.ndarray
    thickness: float
    edges: dict = field(default_factory=dict)     # de las per rand, voor nazicht

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.outline.pts, dtype=float)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.position3d, dtype=float)

    def placed(self, pts=None) -> np.ndarray:
        """De contour (of punten) in de wereld van het 3D-model, op zijn plek in de doos."""
        p = self.points if pts is None else np.asarray(pts, dtype=float)
        M = self.matrix()
        q = np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p)), np.ones(len(p))])
        return (q @ M.T)[:, :3]


def _area(pts) -> float:
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean(pts) -> np.ndarray:
    """Opeenvolgende gelijke punten weg, ook over de sluiting, en naalden eruit.

    Een naald is een hoekpunt waar het pad terugloopt over zichzelf: `a → b → a`. Die
    ontstaan op een hoek van `links` en `rechts`, waar de uitsparing van de verticale las
    precies tot op de onderkant doorloopt en het plein van `t` × `t` daar wegsnijdt. Zo'n
    naald snijdt een spleet van nul breed en is voor een laser zinloos werk; de hoek en
    zijn dubbeling vallen daarom weg.
    """
    out: list[np.ndarray] = []
    for q in np.asarray(pts, dtype=float).reshape(-1, 2):
        if out and np.allclose(out[-1], q, rtol=0.0, atol=1e-9):
            continue
        out.append(np.asarray(q, dtype=float))
    while len(out) > 1 and np.allclose(out[0], out[-1], rtol=0.0, atol=1e-9):
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        n = len(out)
        for i in range(n):
            j = (i + 1) % n
            a, b, c = out[i - 1], out[i], out[j]
            if not np.allclose(a, c, rtol=0.0, atol=1e-9):
                continue
            if float((b - a) @ (c - b)) >= 0.0:
                continue
            for k in sorted({i, j}, reverse=True):
                out.pop(k)
            changed = True
            break
    return np.ascontiguousarray(np.asarray(out, dtype=float).reshape(-1, 2))


def _toothed(p0, p1, inward, q0, q1, gender, line, depth) -> list[np.ndarray]:
    """De punten van een getande rand, in looprichting van `p0` naar `p1`.

    De tanden staan op de nominaal rand tussen `p0` en `p1`; `line` is de polylijn uit
    `finger_edge`, waarvan de x-coördinaat de afstand langs de las is, gemeten vanaf
    `q0` in de richting van `q1`. De tand wijst bij de mannelijke kant uit het paneel
    (tegen `inward` in), bij de vrouwelijke kant snijdt hij `depth` diep naar binnen.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    u = np.asarray(q1, float) - np.asarray(q0, float)
    if float(np.hypot(*u)) <= 1e-12:
        return []
    u = u / float(np.hypot(*u))
    dd = float(d @ u)
    if abs(dd) <= 1e-12:
        return []
    pts = line if dd > 0.0 else np.asarray(line, dtype=float)[::-1]
    base = float((p0 - np.asarray(q0, float)) @ u)
    h = abs(float(depth))
    nrm = np.asarray(inward, float)

    def point(q, off):
        s = (float(q) - base) / dd
        return p0 + s * d + off * nrm

    if not len(pts):
        return []
    # de wand van de eerste en de laatste tand: op de nominaal rand, op dezelfde plek
    out: list[np.ndarray] = [point(pts[0][0], 0.0)]
    for q, y in pts:
        off = -float(y) if gender == MALE else h - float(y)
        out.append(point(q, off))
    out.append(point(pts[-1][0], 0.0))
    return out


def _rect_outline(a, b, edges, depth, finger_w, kerf) -> tuple[np.ndarray, dict]:
    """De gesloten omtrek van een rechthoek `a` × `b` met per rand een las.

    `edges` is een lijst van vier keer `None` of `(gender, lengte, q0, q1[, ends])`, in
    tegen klokzin rond de rechthoek: onder, rechts, boven, links. `q0` en `q1` zijn de
    einden van de las in de coördinaten van het paneel; `q0` is waar de tandenrand van
    `finger_edge` begint. `ends` is de fase aan de einden van die rand.
    """
    corners = [np.array([0.0, 0.0]), np.array([a, 0.0]), np.array([a, b]), np.array([0.0, b])]
    pts: list[np.ndarray] = []
    kinds: dict = {}
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        spec = edges[i] if i < len(edges) else None
        d = p1 - p0
        inward = np.array([-float(d[1]), float(d[0])], dtype=float)   # links van de looprichting
        n = float(np.hypot(*inward))
        if n > 1e-12:
            inward = inward / n
        pts.append(p0)
        if spec is not None:
            gender, length, q0, q1 = spec[:4]
            ends = spec[4] if len(spec) > 4 else "tooth"
            line = finger_edge(length, depth, finger_w, gender == MALE, kerf, ends)
            pts += _toothed(p0, p1, inward, q0, q1, gender, line, depth)
            kinds[RANDEN[i]] = gender
        pts.append(p1)
    return _clean(pts), kinds


RANDEN = ("onder", "rechts", "boven", "links")


def _matrix(x_col, y_col, z_col, off) -> np.ndarray:
    """4×4-matrix die (u, v, dikte) op de plek van het paneel zet (echte draaiing)."""
    M = np.eye(4, dtype=float)
    M[:3, :3] = np.column_stack([np.asarray(x_col, float), np.asarray(y_col, float),
                                 np.asarray(z_col, float)])
    M[:3, 3] = np.asarray(off, dtype=float)
    return M


X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def box_panels(w, d, h, t, finger_w=None, inner=True, lid="none", kerf=0.0) -> list[Panel]:
    """De panelen van een doos met vingerlassen, in de volgorde van de doos.

    `w`, `d` en `h` zijn de binnenmaten (`inner=True`) of de buitenmaten van de doos in
    millimeters, `t` is de materiaaldikte en `finger_w` de gewenste vingerbreedte
    (standaard `2.5 × t`). `lid` is "none" (open doos, de wanden lopen tot de bovenrand)
    of "closed" (vast deksel, een zesde paneel dat in de bovenrand van de wanden valt).
    `kerf` corrigeert de snedespleet van de laser, alleen in de vingerlassen. De getekende
    panelen overlappen elkaar dan met precies die kerf, want de snede haalt die weer weg.

    Elk paneel draagt `position3d`: de matrix die zijn vlak op zijn plek in de doos zet,
    zodat `panels_to_manifolds` er direct een geëxtrudeerd volume van kan maken.
    """
    w, d, h, t = float(w), float(d), float(h), float(t)
    lid = str(lid or "none").strip().lower()
    if lid not in LID_KINDS:
        raise ValueError('Deksel is "none" (geen) of "closed" (vast deksel).')
    if min(w, d, h) <= 0.0:
        raise ValueError("Breedte, diepte en hoogte moeten groter dan nul zijn.")
    if t <= 0.0:
        raise ValueError("De materiaaldikte moet groter dan nul zijn.")
    fw = FINGER_FACTOR * t if finger_w is None else float(finger_w)
    if fw <= 0.0:
        raise ValueError("De vingerbreedte moet groter dan nul zijn.")
    k = max(0.0, float(kerf or 0.0))

    if inner:
        W, D = w + 2.0 * t, d + 2.0 * t
        H = h + 2.0 * t if lid == "closed" else h + t
    else:
        W, D, H = w, d, h
    iw, id_ = W - 2.0 * t, D - 2.0 * t          # binnenmaat = wandlengte van voor en achter
    if min(iw, id_) <= 0.0:
        raise ValueError("De doos is kleiner dan twee keer de materiaaldikte.")
    if lid == "closed" and H - 2.0 * t <= 0.0:
        raise ValueError("De doos is te laag voor bodem, wanden en deksel.")
    wall = H - t                                # hoogte van een wand (de bodem is de laag eronder)

    def spec(gender, length, q0, q1, ends="tooth"):
        return (gender, length, np.asarray(q0, float), np.asarray(q1, float), ends)

    # -- de bodem: volle buitenmaat, overal vrouwelijk (hij bezit de onderste laag).
    #    De laden van de wanden beginnen pas voorbij het eerste vak, dus de bodem houdt
    #    aan elke hoek een plein van `t` × `t` dat met een volle rand aan de plaat vast
    #    zit; anders raakt het hoekplein de rest van de bodem in slechts één punt.
    bottom_edges = [
        spec(FEMALE, iw, (t, 0.0), (W - t, 0.0), SLOT),        # voor
        spec(FEMALE, id_, (W, D - t), (W, t), SLOT),           # rechts
        spec(FEMALE, iw, (W - t, D), (t, D), SLOT),            # achter
        spec(FEMALE, id_, (0.0, t), (0.0, D - t), SLOT),       # links
    ]
    # -- voor en achter: mannelijk onderaan en op beide verticale randen
    side_edges = [
        spec(MALE, iw, (0.0, 0.0), (iw, 0.0), SLOT),           # vingers in de bodem
        spec(MALE, wall, (iw, 0.0), (iw, wall)),
        spec(FEMALE, iw, (0.0, wall), (iw, wall)) if lid == "closed" else None,
        spec(MALE, wall, (0.0, 0.0), (0.0, wall)),
    ]
    # -- links: vrouwelijk op de verticale randen, onderaan voorbij de hoekkolommen
    left_edges = [
        spec(MALE, id_, (t, 0.0), (D - t, 0.0), SLOT),
        spec(FEMALE, wall, (D, 0.0), (D, wall)),
        spec(FEMALE, id_, (t, wall), (D - t, wall)) if lid == "closed" else None,
        spec(FEMALE, wall, (0.0, 0.0), (0.0, wall)),
    ]
    # -- rechts: hetzelfde, maar de las van het deksel loopt er in omgekeerde richting
    right_edges = [left_edges[0], left_edges[1],
                   spec(FEMALE, id_, (D - t, wall), (t, wall)) if lid == "closed" else None,
                   left_edges[3]]
    # -- deksel: binnenmaat, overal mannelijk (het bezit de bovenste laag)
    lid_edges = [
        spec(MALE, iw, (0.0, 0.0), (iw, 0.0)),
        spec(MALE, id_, (iw, 0.0), (iw, id_)),
        spec(MALE, iw, (iw, id_), (0.0, id_)),
        spec(MALE, id_, (0.0, 0.0), (0.0, id_)),
    ]

    def panel(name, a, b, edges, M):
        pts, kinds = _rect_outline(a, b, edges, t, fw, k)
        return Panel(name, VPath(pts, True, filled=True), M, t, kinds)

    panels = [
        panel("bodem", W, D, bottom_edges, _matrix(X, Y, Z, (0.0, 0.0, 0.0))),
        panel("voor", iw, wall, side_edges, _matrix(X, Z, -Y, (t, t, t))),
        panel("achter", iw, wall, side_edges, _matrix(-X, Z, Y, (W - t, D - t, t))),
        panel("links", D, wall, left_edges, _matrix(Y, Z, X, (0.0, 0.0, t))),
        panel("rechts", D, wall, right_edges, _matrix(-Y, Z, -X, (W, D, t))),
    ]
    if lid == "closed":
        panels.append(panel("deksel", iw, id_, lid_edges, _matrix(X, Y, Z, (t, t, H - t))))
    return panels


# --------------------------------------------------------------------------
#  naar werkblad en 3D-model
# --------------------------------------------------------------------------

def panels_to_drawing(panels, layer="Snijden", label_layer="Graveren",
                      labels=True, height=4.0, gap=2.0) -> Drawing:
    """De contouren van de panelen op een tekening, elk paneel als één bron.

    De snijcontouren komen op `layer` (snijden) en de paneelnamen — als kleine
    tekstcontouren in het midden van het paneel — op `label_layer` (graveren). Paden die
    bij elkaar horen krijgen hetzelfde bronnummer, zodat het automatisch schikken een
    paneel met zijn gravure als één geheel verplaatst.

    De panelen zijn elk in hun eigen vlak getekend en liggen daarom in de tekening
    naast elkaar (van links naar rechts, met `gap` ertussen), zodat ze elkaar niet in de
    weg zitten en het nesten ze als losse onderdelen herkent.
    """
    dr = Drawing()
    dr.ensure_layer(layer, "#ff0000", "cut")
    if labels and label_layer:
        dr.ensure_layer(label_layer, "#0000ff", "engrave")
    x = 0.0
    for i, p in enumerate(panels):
        src = i + 1
        pts = p.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        off = np.array([x - float(lo[0]), -float(lo[1])], dtype=float)
        dr.add_path(pts + off, closed=True, layer=layer, filled=True, src=src)
        if labels and label_layer:
            for loop in label_loops(p, height):
                dr.add_path(loop + off, closed=True, layer=label_layer, filled=True, src=src)
        x += float(hi[0] - lo[0]) + max(0.0, float(gap))
    return dr


def label_loops(panel, height=4.0) -> list[np.ndarray]:
    """De contouren van de paneelnaam, gecentreerd in het midden van het paneel."""
    pts = panel.points
    if len(pts) < 3:
        return []
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    if min(span) <= 0.0:
        return []
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters hebben een Qt-app nodig; deze kern is dat niet
    from . import fonts                   # Qt: pas hier, deze kern is Qt-vrij
    try:
        h = min(float(height), 0.4 * float(min(span)))
        loops = fonts.text_loops(panel.name, height=max(0.5, h))
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    fit = 0.6 * float(span[0])
    scale = fit / wide if wide > 1e-9 and wide > fit else 1.0
    out = []
    for L in loops:
        q = np.asarray(L, dtype=float) - allp.min(axis=0)
        out.append(np.ascontiguousarray(q * scale))
    got = np.concatenate(out) if out else np.zeros((0, 2))
    if not len(got):
        return []
    shift = 0.5 * (lo + hi) - 0.5 * (got.min(axis=0) + got.max(axis=0))
    return [np.ascontiguousarray(L + shift) for L in out]


def panels_to_manifolds(panels) -> list:
    """Elk paneel geëxtrudeerd met zijn dikte en getransformeerd naar zijn plek.

    Met `kerf` overlappen de panelen hier precies de kerf van de laser: de getekende
    vinger is aan elke kant `kerf / 2` breder en de gleuf evenveel smaller, omdat de
    snedespleet dat er later afhaalt. Het model toont daarom de panelen zoals ze van de
    laser komen en niet de doos na de montage; de vereniging van alle panelen is wel
    precies de wand van de doos.
    """
    out = []
    for p in panels:
        cs = mf.CrossSection([np.ascontiguousarray(p.points, dtype=float)], mf.FillRule.NonZero)
        man = cs.extrude(float(p.thickness))
        out.append(man.transform(np.ascontiguousarray(p.matrix()[:3], dtype=np.float64)))
    return out


def box_size(w, d, h, t, inner=True, lid="none"):
    """(breedte, diepte, hoogte) van de afgewerkte doos, voor de melding in de statusbalk."""
    w, d, h, t = float(w), float(d), float(h), float(t)
    closed = str(lid or "none").strip().lower() == "closed"
    if inner:
        return (w + 2.0 * t, d + 2.0 * t, h + (2.0 if closed else 1.0) * t)
    return (w, d, h)
