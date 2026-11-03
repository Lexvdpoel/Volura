"""Schroefdraad (ISO metrisch) voor 3D-printen.

Een schroefdraad wordt als "schroefoppervlak" gebouwd: een rooster van ringen in
z (stap hooguit pitch/16) en hoeken (segments), waarbij elk punt de straal
r(theta, z) krijgt die het ISO-profiel heeft op positie

    u = (z - starts * pitch * theta / (2 pi)) mod pitch

langs de schroeflijn. Boven en onder sluit een vlakke deksel (driehoekswaaier
naar het middelpunt) het lichaam af, zodat het resultaat altijd een gesloten
manifold is: geschikt voor boolean-operaties en STL/3MF-export.

ISO-profiel: flankhoek 60 graden, H = 0.866 * pitch, buitendiameter major_d,
kerndiameter major_d - 1.0825 * pitch, afgeplatte top (H/8) en bodem (H/4).
``internal=True`` geeft het lichaam dat uit een bus gesneden wordt om binnendraad
te maken: dezelfde vorm met elke straal vergroot om de speling te geven.
"""
from __future__ import annotations

import math

import numpy as np
import manifold3d as mf

from .geom import affine, circle_points, manifold_from_mesh, normalize, rotation_matrix

TAN30 = math.tan(math.pi / 6)

# grove ISO-spoed per gangbare metrische maat (mm)
ISO_PITCH = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 16: 2.0, 20: 2.5}

DEFAULT_CLEARANCE = 0.15    # mm radiale speling: bout en moer passen na het printen


def iso_pitch(diameter):
    """Standaard grove spoed bij een metrische diameter (dichtstbijzijnde maat)."""
    sizes = sorted(ISO_PITCH)
    best = min(sizes, key=lambda d: abs(math.log(d / max(diameter, 1e-6))))
    return ISO_PITCH[best]


# ------------------------------------------------------------------ rekern

def _profile_radius(u, pitch, r_major, r_minor):
    """Straal van het ISO-profiel op positie u in [0, pitch) (0 = kruin)."""
    H = math.sqrt(3.0) / 2.0 * pitch
    u_c = H / 8.0 * TAN30          # halve breedte van de afgeplatte kruin
    q = np.minimum(u, pitch - u)   # afstand tot de dichtstbijzijnde kruin
    r = r_major - np.maximum(q - u_c, 0.0) / TAN30
    return np.maximum(r, r_minor)


def thread_solid(major_d, pitch, length, internal=False, clearance=0.0, starts=1,
                 left=False, segments=64, chamfer=True):
    """Schroefdraad-lichaam langs de Z-as van z=0 tot z=length (manifold3d).

    ``internal=True`` is het snijlichaam voor binnendraad: dezelfde vorm met elke
    straal + clearance. ``chamfer`` schuint de eerste en laatste halve spoed af
    onder 45 graden, zodat de draad makkelijk aangrijpt.
    """
    major_d, pitch, length = float(major_d), float(pitch), float(length)
    if major_d <= 0 or pitch <= 0 or length <= 0:
        raise ValueError("major_d, pitch en length moeten groter dan 0 zijn")
    starts = max(1, int(starts))
    segments = max(12, int(segments))
    r_major = major_d / 2.0 + clearance
    r_minor = (major_d - 1.0825 * pitch) / 2.0 + clearance
    if r_minor <= 0:
        raise ValueError("te grove spoed voor deze diameter")
    sign = -1.0 if left else 1.0

    n = max(2, int(math.ceil(length / (pitch / 16.0))))
    zs = np.linspace(0.0, length, n + 1)
    th = np.arange(segments) * (2.0 * math.pi / segments)
    u = np.mod(zs[:, None] - sign * starts * pitch * th[None, :] / (2.0 * math.pi), pitch)
    r = _profile_radius(u, pitch, r_major, r_minor)

    if chamfer:
        half = pitch / 2.0
        cut = np.maximum(half - zs, 0.0)[:, None] + np.maximum(zs - (length - half), 0.0)[:, None]
        if internal:
            r = r + cut          # het snijlichaam wordt aan de uiteinden wijder: schuine instap
        else:
            r = np.maximum(r - cut, max(0.02, 0.1 * r_minor))

    cos_t, sin_t = np.cos(th), np.sin(th)
    verts = np.stack([r * cos_t[None, :], r * sin_t[None, :],
                      np.broadcast_to(zs[:, None], r.shape)], axis=-1).reshape(-1, 3)
    c0, ct = len(verts), len(verts) + 1
    verts = np.vstack([verts, [[0.0, 0.0, 0.0], [0.0, 0.0, length]]])

    idx = np.arange((n + 1) * segments).reshape(n + 1, segments)
    a = idx[:-1].ravel()
    b = np.roll(idx[:-1], -1, axis=1).ravel()
    c = np.roll(idx[1:], -1, axis=1).ravel()
    d = idx[1:].ravel()
    tris = [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    ring0, ringN = idx[0], idx[-1]
    tris.append(np.stack([np.full(segments, c0), np.roll(ring0, -1), ring0], axis=1))
    tris.append(np.stack([np.full(segments, ct), ringN, np.roll(ringN, -1)], axis=1))
    man = manifold_from_mesh(verts, np.concatenate(tris))
    if man.status() == mf.Error.NoError and man.volume() < 0:
        man = manifold_from_mesh(verts, np.concatenate(tris)[:, ::-1])
    return man


def _place(man, center, axis):
    """Transformeer een lichaam dat langs +Z van 0 ligt naar (center, axis)."""
    a = normalize(np.asarray(axis, dtype=float))
    center = np.asarray(center, dtype=float)
    z = np.array([0.0, 0.0, 1.0])
    w = np.cross(z, a)
    if np.linalg.norm(w) < 1e-12:
        M3 = np.eye(3) if float(z @ a) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        M3 = rotation_matrix(w, math.atan2(float(np.linalg.norm(w)), float(z @ a)))
    return man.transform(affine(M3, center))


def external_on(manifold, center, axis, radius, length, pitch=1.5, clearance=0.0,
                starts=1, left=False, segments=64):
    """Vervang een cilinder in het lichaam door buitendraad met buitendiameter 2*radius."""
    cyl = _place(mf.Manifold.cylinder(length, radius, radius, max(16, int(segments))),
                 center, axis)
    thread = _place(thread_solid(2.0 * radius, pitch, length, clearance=clearance,
                                 starts=starts, left=left, segments=segments), center, axis)
    return ((manifold - cyl) + thread) ^ (manifold + cyl)


def internal_in(manifold, center, axis, radius, length, clearance=DEFAULT_CLEARANCE,
                pitch=1.5, starts=1, left=False, segments=64):
    """Snijd binnendraad in een gat met straal radius (major_d = 2*radius)."""
    thread = _place(thread_solid(2.0 * radius, pitch, length, internal=True,
                                 clearance=clearance, starts=starts, left=left,
                                 segments=segments), center, axis)
    return manifold - thread


def nut(major_d, pitch, height, outer_d=None, clearance=DEFAULT_CLEARANCE, hex=True,
        starts=1, left=False, segments=64):
    """Losse moer (zeskant) of bus (rond) met binnendraad, van z=0 tot z=height."""
    if outer_d is None:
        outer_d = major_d * 1.7
    if outer_d <= major_d:
        raise ValueError("de buitenmaat van een moer moet groter zijn dan de draad")
    if hex:
        pts = circle_points(outer_d / 2.0 / math.cos(math.pi / 6.0), 6)
        body = mf.Manifold.extrude(mf.CrossSection([pts]), height)
    else:
        body = mf.Manifold.cylinder(height, outer_d / 2.0, outer_d / 2.0,
                                    max(24, int(segments)))
    hole = thread_solid(major_d, pitch, height, internal=True, clearance=clearance,
                        starts=starts, left=left, segments=segments)
    return body - hole


# ------------------------------------------------- selectie -> draadcontext

def circle_loop(points, min_points=12):
    """(middelpunt, straal) als de lus een volledige cirkel is, anders (None, 0).

    Een cirkel is hier: minstens ``min_points`` punten met dezelfde afstand tot
    het gemiddelde en een gesloten hoekdekking.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < min_points:
        return None, 0.0
    c = P.mean(axis=0)
    d = np.linalg.norm(P - c, axis=1)
    r = float(d.mean())
    if r < 1e-6 or float(d.max() - d.min()) > 0.02 * r:
        return None, 0.0
    n = np.zeros(3)
    for a, b in zip(P - c, np.roll(P - c, -1, axis=0)):
        n += np.cross(a, b)
    if np.linalg.norm(n) < 1e-9:
        return None, 0.0
    n = normalize(n)
    u = normalize(P[0] - c)
    v = np.cross(n, u)
    ang = np.sort(np.arctan2((P - c) @ v, (P - c) @ u))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]]))
    if float(gaps.max()) > 2.5 * 2.0 * math.pi / len(P):
        return None, 0.0        # een boog, geen volledige cirkel
    return c, r


def _loop_area(P):
    """Oppervlakte van een vlakke lus (Newell), ongevoelig voor de volgorde."""
    c = P.mean(axis=0)
    n = np.zeros(3)
    for a, b in zip(P - c, np.roll(P - c, -1, axis=0)):
        n += np.cross(a, b)
    return float(np.linalg.norm(n)) / 2.0


def face_thread_context(ents, fid, want="outer"):
    """(center, axis, radius, length) van een cirkelvormig vlak of gat in een lichaam.

    ``want``: "outer" voor het eindvlak van een cilinder (buitendraad), "hole"
    voor de opening van een gat (binnendraad). De as wijst het lichaam in.
    from_manifold garandeert niet dat loops[0] de buitenlus is, dus de lussen
    worden op grootte vergeleken.
    """
    if fid not in ents.F or not ents.V:
        return None
    plane = ents.face_plane(fid)
    loops = ents.face_points(fid)
    if not loops:
        return None
    biggest = int(np.argmax([_loop_area(np.asarray(L, dtype=float)) for L in loops]))
    P = np.array(list(ents.V.values()))
    dist = (P - plane.origin) @ plane.n
    axis = -plane.n if -float(dist.min()) >= float(dist.max()) else plane.n
    best = None
    for k, L in enumerate(loops):
        if want == "outer" and k != biggest:
            continue
        if want == "hole" and k == biggest:
            continue
        c, r = circle_loop(L)
        if c is None or (best is not None and best[2] >= r):
            continue
        along = (P - c) @ axis
        best = (c, axis, r, float(along.max() - along.min()))
    return best


def group_thread_context(ents, want="outer"):
    """Context uit het grootste cirkelvormige vlak van een lichaam (bijv. een cilinder)."""
    best = None
    for fid in ents.visible_faces() if hasattr(ents, "visible_faces") else list(ents.F):
        tc = face_thread_context(ents, fid, want)
        if tc is not None and (best is None or tc[2] > best[2]):
            best = tc
    return best
