"""Afronden (fillet) en afschuinen (chamfer) van randen tussen vlakke vlakken.

Kijk loodrecht op een rand tussen twee vlakken. De twee vlakken snijden daar
een hoek uit van alpha = 180° - gamma, met gamma de hoek tussen de buiten-
normalen van de vlakken. Die hoek is het "hoekpunt" van de rand:

* bij een **bolle** rand (denk aan de bovenrand van een kubus) staat in die hoek
  het materiaal van het lichaam: afronden haalt er een wig van af;
* bij een **holle** rand (de binnenhoek van een L-profiel) staat in die hoek
  lucht: afronden vult die hoek op met een bolle rand.

Beide gevallen gebruiken dezelfde wig en dezelfde formules; alleen de richting
van de bissectrice en of de wig afgetrokken of opgeteld wordt verschillen.

Een keten van randen (de gladde bovenrand van een cilinder is uit 48 randjes
opgebouwd) wordt per segment afgeronden. De cutter van een segment steekt bij
een hoekpunt door dat het met een buursegment deelt, en de twee profielen die
daar in elkaar uitkomen worden met een loft (de convexe omhulling van de twee
profielvlakken, langs elkaar gelegd) aan elkaar verbonden zodat de afronding
doorloopt en er in de hoek geen gat of richel overblijft.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import manifold3d as mf

from .geom import Plane, extrude_cs, manifold_from_mesh, rotation_matrix

MARGIN = 0.01            # mm: cutters steken voorbij de uiteinden door, geen naden
MIN_ANGLE = math.radians(0.4)     # vlakker dan dit is de rand al glad
MAX_ANGLE = math.radians(179.6)   # vrijwel dubbelgevouwen: niets te ronden
RAY = np.array([1237.13, 917.71, 553.37])   # een richting die nooit op een as ligt


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def _union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    if not ms:
        return mf.Manifold()
    if len(ms) == 1:
        return ms[0]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)


def arc_segments(segments, sweep):
    """Aantal lijnstukken voor een boog van `sweep` radialen bij een cirkel van
    `segments` stukken: de cilinder van een afronding krijgt er 16 tot 32."""
    full = max(16, int(segments))
    return max(2, int(round(full * abs(float(sweep)) / (2 * math.pi))))


# ------------------------------------------------------- het hoekpunt van een rand

def corner_frame(p0, p1, n1, n2, convex=True):
    """(d, L, r1, r2, b, alpha) van het hoekpunt van een rand, of None.

    r1 en r2 zijn de halve lijnen waarop de twee vlakken van de rand af wijzen en
    b de bissectrice ertussen; alpha is de hoek tussen r1 en r2 (altijd kleiner
    dan 180°). Bij een bolle rand (convex) staat binnen die hoek het materiaal,
    bij een holle rand de lucht."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = _unit(p1 - p0)
    if d is None:
        return None
    L = float(np.linalg.norm(p1 - p0))
    u1 = np.asarray(n1, dtype=float) - d * float(np.dot(n1, d))
    u2 = np.asarray(n2, dtype=float) - d * float(np.dot(n2, d))
    u1, u2 = _unit(u1), _unit(u2)
    if u1 is None or u2 is None:
        return None
    gamma = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
    alpha = math.pi - gamma
    if not MIN_ANGLE < alpha < MAX_ANGLE:
        return None
    b = _unit((-1.0 if convex else 1.0) * (u1 + u2))
    if b is None:
        return None
    r1, r2 = _unit(np.cross(d, u1)), _unit(np.cross(d, u2))
    if r1 is None or r2 is None:
        return None
    if float(np.dot(r1, b)) < 0:
        r1 = -r1
    if float(np.dot(r2, b)) < 0:
        r2 = -r2
    if float(np.dot(r1, r2)) > math.cos(alpha) + 1e-6:
        return None
    return d, L, r1, r2, b, alpha


def corner_profile(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True, arc=0):
    """Het profiel van de cutter in p0, loodrecht op de rand, als 3D punten, en
    het aantal boogstukken dat daarbij gebruikt is. None als er niets te doen is.

    Bij "chamfer" is dat de driehoek tussen de rand en de twee punten op afstand
    `size` in de vlakken; bij "fillet" de wig tussen de rand, de twee raakpunten
    op afstand size/tan(alpha/2) en de cirkelboog met straal `size`."""
    fr = corner_frame(p0, p1, n1, n2, convex)
    if fr is None:
        return None
    d, L, r1, r2, b, alpha = fr
    p0 = np.asarray(p0, dtype=float)
    if mode == "chamfer":
        return np.array([p0, p0 + r1 * size, p0 + r2 * size], dtype=float), 0
    T = size / math.tan(alpha / 2.0)              # rand -> raakpunt
    t1, t2 = p0 + r1 * T, p0 + r2 * T
    c = p0 + b * (size / math.sin(alpha / 2.0))   # middelpunt van de cirkelboog
    a1, a2 = t1 - c, t2 - c
    sweep = math.acos(float(np.clip(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)),
                                    -1.0, 1.0)))
    axis = d if float(np.dot(np.cross(a1, a2), d)) > 0 else -d
    n = int(arc) if arc else arc_segments(segments, sweep)
    R = rotation_matrix(axis, sweep / n)
    pts, v = [p0, t1], a1
    for _ in range(n - 1):
        v = R @ v
        pts.append(c + v)
    pts.append(t2)
    return np.array(pts, dtype=float), n


def corner_wedge(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True):
    """Het vlakke doorsnede-polygoon van het hoekpunt van een rand (3D punten)."""
    pr = corner_profile(p0, p1, n1, n2, size, mode, segments, convex)
    return None if pr is None else pr[0]


def extrude_profile(p0, d, L, pts, e0=0.0, e1=0.0):
    """Het prisma van een profiel langs de randrichting, van -e0 tot L + e1."""
    if pts is None or len(pts) < 3:
        return None
    pl = Plane(p0, d, u=pts[1] - pts[0])
    cs = mf.CrossSection([pl.to_local(pts)], mf.FillRule.EvenOdd)
    cut = extrude_cs(cs, pl, -MARGIN - float(e0), L + MARGIN + float(e1))
    if cut.is_empty() or cut.status() != mf.Error.NoError:
        return None
    return cut


def edge_cutter(p0, p1, n1, n2, size, mode="fillet", segments=64, convex=True, extend=0.0):
    """Het lichaam dat bij een rand van het volume af moet (bolle rand) of erbij
    opgeteld moet worden (holle rand). None als er niets te doen is.

    Het lichaam steekt aan beide uiteinden MARGIN mm uit de rand, zodat er bij
    het snijden geen naden overblijven; `extend` (getal of (voor, na)) steekt
    verder uit, zodat cutters van een keten van randen elkaar overlappen."""
    pr = corner_profile(p0, p1, n1, n2, size, mode, segments, convex)
    if pr is None:
        return None
    pts, _arc = pr
    d = _unit(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    if d is None:
        return None
    L = float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))
    e0, e1 = (extend, extend) if np.isscalar(extend) else tuple(extend)
    return extrude_profile(p0, d, L, pts, e0, e1)


# ------------------------------------------------- twee segmenten aan elkaar zetten

def profile_at(pts, d, L, at_start):
    """Hetzelfde profiel, maar in het andere uiteinde van de rand."""
    return np.asarray(pts, dtype=float) if at_start else np.asarray(pts, dtype=float) + d * L


def loft(pa, pb):
    """Het lichaam tussen twee profielen die in hetzelfde hoekpunt staan: de
    convexe omhulling van de twee profielvlakken, punt voor punt langs elkaar
    gelegd. Dat vult de hoek tussen twee opeenvolgende segmenten van een keten
    zonder er meer af te halen dan de afronding zelf."""
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    n = len(pa)
    if n < 3 or len(pb) != n:
        return None
    verts = np.vstack([pa, pb])
    tris = []
    for i in range(1, n - 1):                     # de twee eindvlakken (waaier vanuit het hoekpunt)
        tris.append((0, i + 1, i))
        tris.append((n, n + i, n + i + 1))
    for i in range(n):                            # de mantel
        j = (i + 1) % n
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    tris = np.array(tris, dtype=np.int64)
    try:
        m = manifold_from_mesh(verts, tris)
        if m.is_empty() or m.status() != mf.Error.NoError or m.volume() < 0:
            m = manifold_from_mesh(verts, tris[:, ::-1])     # omgekeerde winding
    except Exception:
        return None
    if m.is_empty() or m.status() != mf.Error.NoError or m.volume() <= 0:
        return None
    return m


# ------------------------------------------------------- bol of hol bepalen

def _inside(man, q, far):
    hits = man.ray_cast(tuple(np.asarray(q, dtype=float)),
                        tuple(np.asarray(q, dtype=float) + far))
    return len(hits) % 2 == 1


def is_convex(man, p0, p1, n1, n2, size=1.0):
    """Bol of hol: de twee "zijkanten" van een rand zijn bij een bolle rand lucht,
    bij een holle rand materiaal. De vlakken van de rand raken die zijkanten niet,
    dus twee steekproeven zeggen iets; zijn het er maar één, dan is het bol."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = _unit(p1 - p0)
    if d is None:
        return True
    w = np.asarray(n1, dtype=float) - np.asarray(n2, dtype=float)
    w = _unit(w - d * float(np.dot(w, d)))
    if w is None:
        return True
    bb = np.asarray(man.bounding_box(), dtype=float).reshape(-1)
    span = bb[3:] - bb[:3] if bb.size == 6 else np.zeros(3)
    far = RAY * max(1.0, float(np.linalg.norm(span)))
    mid = 0.5 * (p0 + p1)
    step = float(np.clip(min(size, float(np.linalg.norm(p1 - p0))) * 0.1, 0.02, 2.0))
    inside = int(_inside(man, mid + w * step, far)) + int(_inside(man, mid - w * step, far))
    return inside != 2


# ------------------------------------------------------------------ toepassen

def apply(man, edges, size, mode="fillet", segments=64, join=True):
    """Rondd of schuin de randen `edges` = [(p0, p1, n1, n2)] af; geeft een nieuwe
    manifold. Bolle randen worden afgetrokken, holle opgeteld.

    Randen die een hoekpunt delen vormen een keten. Bij zo'n hoekpunt wordt het
    lichaam tussen de twee profielen toegevoegd, zodat de afronding over de hele
    keten doorloopt en er in de hoek geen richel staat (`join=False` laat dat
    achterwege)."""
    size = float(size)
    if size <= 0:
        raise ValueError("de maat moet groter dan 0 zijn")
    items = []
    for (p0, p1, n1, n2) in edges:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        n1 = np.asarray(n1, dtype=float)
        n2 = np.asarray(n2, dtype=float)
        convex = is_convex(man, p0, p1, n1, n2, size)
        fr = corner_frame(p0, p1, n1, n2, convex)
        if fr is None:
            continue
        _d, _L, _r1, _r2, _b, alpha = fr
        items.append({"p0": p0, "p1": p1, "n1": n1, "n2": n2, "convex": convex,
                      "d": fr[0], "L": fr[1], "alpha": alpha})
    if not items:
        return man
    at = defaultdict(list)
    for i, it in enumerate(items):
        at[_vkey(it["p0"])].append(i)
        at[_vkey(it["p1"])].append(i)
    cutters = []
    for it in items:
        cut = edge_cutter(it["p0"], it["p1"], it["n1"], it["n2"], size, mode,
                          segments, it["convex"])
        cutters.append(None if cut is None or cut.volume() <= 0 else (cut, it["convex"]))
    if not any(c is not None for c in cutters):
        return man
    if join:
        for key, idxs in at.items():
            for i, j in combinations(sorted(set(idxs)), 2):
                if cutters[i] is None or cutters[j] is None:
                    continue
                if items[i]["convex"] != items[j]["convex"]:
                    continue
                jnt = joint(items[i], items[j], key, mode, size, segments)
                if jnt is not None:
                    cutters.append((jnt, items[i]["convex"]))
    sub = [c for c, convex in cutters if convex]
    add = [c for c, convex in cutters if not convex]
    res = man
    if sub:
        res = res - _union(sub)
    if add:
        res = res + _union(add)
    return res


def joint(a, b, key, mode, size, segments):
    """Het lichaam dat de cutters van twee randen die in hetzelfde hoekpunt aan
    elkaar komen verbindt, zodat de afronding er doorloopt."""
    pa, na = corner_profile(a["p0"], a["p1"], a["n1"], a["n2"], size, mode,
                            segments, a["convex"])
    pb, nb = corner_profile(b["p0"], b["p1"], b["n1"], b["n2"], size, mode,
                            segments, b["convex"])
    if pa is None or pb is None:
        return None
    if na != nb:                                  # dezelfde bemonstering nodig om te verbinden
        n = min(na, nb)
        pa, _ = corner_profile(a["p0"], a["p1"], a["n1"], a["n2"], size, mode, segments,
                               a["convex"], n)
        pb, _ = corner_profile(b["p0"], b["p1"], b["n1"], b["n2"], size, mode, segments,
                               b["convex"], n)
    pa = profile_at(pa, a["d"], a["L"], _vkey(a["p0"]) == key)
    pb = profile_at(pb, b["d"], b["L"], _vkey(b["p0"]) == key)
    return loft(pa, pb)


def _vkey(p):
    return tuple(np.round(np.asarray(p, dtype=float), 3))


# --------------------------------------------------------- randen van een model

def edge_normals(ents, eid):
    """(p0, p1, n1, n2) van een rand met precies twee vlakken, of None."""
    if eid not in ents.E:
        return None
    fs = sorted(f for f in ents.e2f.get(eid, ()) if f in ents.F)
    if len(fs) != 2:
        return None
    p0, p1 = ents.edge_points(eid)
    return (np.asarray(p0, dtype=float), np.asarray(p1, dtype=float),
            ents.F[fs[0]].normal.copy(), ents.F[fs[1]].normal.copy())


def edge_data(ents, eids):
    """(rij met (eid, p0, p1, n1, n2), rij met randen die overgeslagen zijn)."""
    ok, bad = [], []
    for eid in eids:
        d = edge_normals(ents, eid)
        if d is None:
            if eid in ents.E:
                bad.append(eid)
            continue
        ok.append((eid,) + d)
    return ok, bad


def _continues(ents, e, w, shared, cos_lim):
    """Gaat rand `w` glad door waar rand `e` ophoudt: ze delen vlak `shared` en de
    twee vlakken daarachter zijn bijna hetzelfde vlak (de randen van één ronding)."""
    if e in ents.soft and w in ents.soft:
        return True
    ne = [ents.F[g].normal for g in ents.e2f.get(e, ()) if g != shared and g in ents.F]
    nw = [ents.F[g].normal for g in ents.e2f.get(w, ()) if g != shared and g in ents.F]
    if not ne or not nw:
        return False
    return any(abs(float(np.dot(a, b))) > cos_lim for a in ne for b in nw)


def soft_chain(ents, eid, angle=20.0):
    """Alle aaneengesloten gladde randen die één keten vormen met `eid`: de wanden
    van een cilinder zijn bijvoorbeeld uit 48 randjes opgebouwd die samen één
    ronding zijn. Een rechte hoek (kubus) stopt de keten.

    Twee gladde randen hoeven elkaar niet in een hoekpunt te raken: bij een cilinder
    lopen de twee naden van een wand vlak tegenover elkaar, en toch zijn ze dezelfde
    ronding. Randen die niet glad zijn worden alleen aan elkaar gereken als ze
    elkaar in een hoekpunt raken en de vlakken achter hen hetzelfde vlak zijn."""
    if eid not in ents.E:
        return []
    cos_lim = math.cos(math.radians(float(angle)))
    seen, stack = {eid}, [eid]
    while stack:
        e = stack.pop()
        for f in ents.e2f.get(e, ()):
            soft_face = e in ents.soft
            for w in ents.face_edges(ents.F[f]):
                if w in seen or w not in ents.E:
                    continue
                if not (soft_face and w in ents.soft) and not set(ents.E[w]) & set(ents.E[e]):
                    continue
                if _continues(ents, e, w, f, cos_lim):
                    seen.add(w)
                    stack.append(w)
    return sorted(seen)


smooth_chain = soft_chain        # dezelfde keten, andere naam
