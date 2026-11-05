"""SketchUp style inference: snapping to endpoints, midpoints, edges, faces and axes."""
from __future__ import annotations

import numpy as np

from . import workers
from .brep import render_payload, render_triangulate_job
from .geom import AXES, GROUND, Plane, closest_on_line_to_ray
from .i18n import tr
from .model import ROOT

PICK_PX = 10.0
EDGE_PX = 7.0
AXIS_PX = 9.0          # the drawing axes: a little wider than an edge, as in SketchUp

# Warming the render cache of many groups at once only pays once there are enough
# faces to beat the trip to the workers. Measured on the benchmark model (24 groups,
# 16k faces) building every group's cache falls from ~1.8 s to ~0.5 s with the pool;
# below that the pickling costs as much as it saves, so small models build lazily.
RENDER_POOL_MIN_GROUPS = 2
RENDER_POOL_MIN_FACES = 1200


_AXES_ON = (tr("On Red Axis"), tr("On Green Axis"), tr("On Blue Axis"))
LABELS = {
    "endpoint": tr("Endpoint"),
    "origin": tr("Origin"),
    "midpoint": tr("Midpoint"),
    "edge": tr("On Edge"),
    "edge_axis": (tr("On Edge / On Red Axis"), tr("On Edge / On Green Axis"),
                  tr("On Edge / On Blue Axis")),
    "face": tr("On Face"),
    "guide": tr("On Guide"),
    "guidepoint": tr("Guide Point"),
    "axis": _AXES_ON,
    "axisline": _AXES_ON,
    "lock": tr("Locked"),
}


class Inference:
    def __init__(self, kind, point, **kw):
        self.kind = kind
        self.point = np.asarray(point, dtype=float)
        self.ctx = kw.get("ctx")
        self.fid = kw.get("fid")
        self.eid = kw.get("eid")
        self.axis = kw.get("axis")
        self.edge = kw.get("edge")
        self.plane = kw.get("plane")
        self.guide = kw.get("guide")

    @property
    def label(self):
        if self.kind == "edge" and self.axis is not None:
            return LABELS["edge_axis"][self.axis]
        lab = LABELS.get(self.kind, "")
        if isinstance(lab, tuple):
            return lab[self.axis]
        return lab


class GeoCache:
    """Per Entities: triangles, visible edges and snap points (rebuilt on change)."""

    def __init__(self):
        self._c = {}

    def get(self, ents):
        key = id(ents)
        c = self._c.get(key)
        if c is not None and c["version"] == ents.version and c["ents"] is ents:
            return c
        c = self._build(ents)
        self._c[key] = c
        self._trim()
        return c

    def warm(self, ents_list):
        """Triangulate many groups over the cores, so the first frame after opening a
        file does not build every group's cache one by one.

        A no-op when the pool is off or the model is too small to pay for the trip
        to the workers; the groups that are warmed are stored exactly as ``get``
        would build them, so a later change still rebuilds them (version check)."""
        ents_list = [e for e in ents_list if e is not None and len(e.F)]
        if workers.pool() is None or len(ents_list) < RENDER_POOL_MIN_GROUPS:
            return
        if sum(len(e.F) for e in ents_list) < RENDER_POOL_MIN_FACES:
            return
        try:
            results = workers.parallel_map(render_triangulate_job,
                                           [render_payload(e) for e in ents_list],
                                           min_items=RENDER_POOL_MIN_GROUPS)
        except Exception:
            return
        for e, (T, tri_fid, tri_nrm) in zip(ents_list, results):
            self._c[id(e)] = self._assemble(e, (T, tri_fid, tri_nrm, self._face_tris(T, tri_fid)))
            self._trim()

    def _trim(self):
        if len(self._c) > 64:
            for k in list(self._c)[:-32]:
                del self._c[k]

    @staticmethod
    def _face_tris(T, tri_fid):
        """The triangles of each face, split out of the concatenated ``T`` (one run per face)."""
        face_tris = {}
        if len(T):
            b = np.flatnonzero(np.r_[True, tri_fid[1:] != tri_fid[:-1], True])
            for s, e in zip(b[:-1], b[1:]):
                face_tris[int(tri_fid[s])] = T[s:e]
        return face_tris

    @staticmethod
    def _triangulate(ents):
        """``(T, tri_fid, tri_nrm, face_tris)``: the world triangles of every face, in the
        order the faces come in - the heavy part of the cache, done here in the main process."""
        tris, tri_fid, tri_nrm, face_tris = [], [], [], {}
        for f, face in ents.F.items():
            t, _ = ents.face_triangles(f)
            if not len(t):
                continue
            face_tris[f] = t
            tris.append(t)
            tri_fid.extend([f] * len(t))
            tri_nrm.append(np.repeat(face.normal[None], len(t), axis=0))
        return (np.concatenate(tris) if tris else np.zeros((0, 3, 3)),
                np.array(tri_fid, dtype=np.int64),
                np.concatenate(tri_nrm) if tri_nrm else np.zeros((0, 3)),
                face_tris)

    @staticmethod
    def _assemble(ents, tri):
        """The cache entry built around a group's triangulation ``(T, tri_fid, tri_nrm, face_tris)``."""
        T, tri_fid, tri_nrm, face_tris = tri
        hf, he = ents.hidden_f, ents.hidden_e
        tri_hidden = np.fromiter((f in hf for f in tri_fid), dtype=bool, count=len(tri_fid))
        hard = [e for e in ents.E if e not in ents.soft]
        ehidden = np.fromiter((e in he for e in hard), dtype=bool, count=len(hard))
        shown = [e for e, h in zip(hard, ~ehidden) if h]
        A = np.array([ents.V[ents.E[e][0]] for e in hard]) if hard else np.zeros((0, 3))
        B = np.array([ents.V[ents.E[e][1]] for e in hard]) if hard else np.zeros((0, 3))
        soft = [e for e in ents.E if e in ents.soft]
        SA = np.array([ents.V[ents.E[e][0]] for e in soft]) if soft else np.zeros((0, 3))
        SB = np.array([ents.V[ents.E[e][1]] for e in soft]) if soft else np.zeros((0, 3))
        vids = sorted({v for e in shown for v in ents.E[e]})
        corners = np.array([ents.V[v] for v in vids]) if vids else np.zeros((0, 3))
        return {
            "ents": ents, "version": ents.version,
            "T": T, "tri_fid": tri_fid,
            "tri_nrm": tri_nrm,
            "tri_hidden": tri_hidden, "nhidden": int(tri_hidden.sum()),
            "face_tris": face_tris,
            "p0": T[:, 0] if len(T) else np.zeros((0, 3)),
            "e1": (T[:, 1] - T[:, 0]) if len(T) else np.zeros((0, 3)),
            "e2": (T[:, 2] - T[:, 0]) if len(T) else np.zeros((0, 3)),
            "A": A, "B": B, "eids": hard, "ehidden": ehidden,
            "SA": SA, "SB": SB, "sids": soft,
            "corners": corners, "mid": (A + B) / 2 if len(A) else np.zeros((0, 3)),
            "bbox": ents.bbox(),
        }

    @staticmethod
    def _build(ents):
        return GeoCache._assemble(ents, GeoCache._triangulate(ents))


def ray_tris(c, o, d, allow_hidden=False):
    if not len(c["T"]):
        return None
    e1, e2 = c["e1"], c["e2"]
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tvec = o - c["p0"]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = (qvec @ d) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
    ok = (np.abs(det) > 1e-14) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    if not allow_hidden and c.get("nhidden"):
        ok &= ~c["tri_hidden"]
    if not ok.any():
        return None
    cand = np.nonzero(ok)[0]
    k = cand[np.argmin(t[cand])]
    return float(t[k]), int(c["tri_fid"][k])


def _closest_line_params(p1, d1, p2, d2):
    """(s, u) minimising |p1 + s d1 - (p2 + u d2)|, or None when the lines are parallel."""
    dd, ee, de = float(d1 @ d1), float(d2 @ d2), float(d1 @ d2)
    den = dd * ee - de * de
    if abs(den) < 1e-12:
        return None
    w = p1 - p2
    b1, b2 = -float(d1 @ w), -float(d2 @ w)
    return (b1 * ee - de * b2) / den, (b1 * de - dd * b2) / den


def line_segment_hit(p, d, a, b, tol=1e-4):
    """Where the infinite line (p, d) crosses the segment a-b, or None when it misses."""
    p, d, a, b = (np.asarray(v, dtype=float) for v in (p, d, a, b))
    r = b - a
    if float(r @ r) < 1e-18:
        return None
    su = _closest_line_params(p, d, a, r)
    if su is None:
        return None
    s, u = su
    if u < -1e-9 or u > 1 + 1e-9:
        return None
    if float(np.linalg.norm(p + d * s - (a + r * u))) > tol:
        return None                      # the two lines pass beside each other
    return a + r * u


def line_line_hit(p1, d1, p2, d2, tol=1e-4):
    """The crossing point of two infinite lines, or None when they do not meet."""
    p1, d1, p2, d2 = (np.asarray(v, dtype=float) for v in (p1, d1, p2, d2))
    su = _closest_line_params(p1, d1, p2, d2)
    if su is None:
        return None
    s, u = su
    if float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * u))) > tol:
        return None
    return p1 + d1 * s


class InferenceEngine:
    def __init__(self, doc, camera, settings=None):
        self.doc = doc
        self.cam = camera
        self.settings = settings
        self.cache = GeoCache()

    @property
    def show_hidden(self):
        """View > Hidden Geometry: hidden faces and edges can be picked too."""
        return bool(getattr(self.settings, "show_hidden", False))

    @property
    def show_guides(self):
        """View > Guides: hidden guides are not used for inference either."""
        return bool(getattr(self.settings, "show_guides", True))

    def invalidate(self):
        pass  # caches are keyed on entity versions

    # --------------------------------------------------------- guides
    def guide_lines(self):
        """The visible guide lines (infinite lines) of the document."""
        if not self.show_guides:
            return []
        return [g for g in self.doc.guides if g.is_line]

    def guide_points(self):
        """The visible guide points of the document."""
        if not self.show_guides:
            return []
        return [g for g in self.doc.guides if not g.is_line]

    def _guide_crossings(self, guides, A, B):
        """The points where guide lines cross edges or other guide lines."""
        if not guides:
            return np.zeros((0, 3))
        pts = []
        if len(A):
            r = B - A
            ee = np.einsum("ij,ij->i", r, r)
            for g in guides:
                d = g.direction
                w = g.point - A
                dd = float(d @ d)
                de = r @ d
                den = dd * ee - de * de
                b1 = -(w @ d)
                b2 = -np.einsum("ij,ij->i", r, w)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (b1 * ee - de * b2) / den
                    u = (b1 * de - dd * b2) / den
                    gap = np.linalg.norm(w + d[None] * s[:, None] - r * u[:, None], axis=1)
                ok = (np.abs(den) > 1e-12) & (u > -1e-9) & (u < 1 + 1e-9) & (gap < 1e-4) & (ee > 1e-18)
                pts.extend(A[k] + r[k] * u[k] for k in np.nonzero(ok)[0])
        for i, gi in enumerate(guides):
            for gj in guides[i + 1:]:
                p = line_line_hit(gi.point, gi.direction, gj.point, gj.direction)
                if p is not None:
                    pts.append(p)
        return np.array(pts, dtype=float).reshape(-1, 3) if pts else np.zeros((0, 3))

    def _on_guide(self, g, o, d, m):
        """The point of guide line g that the cursor ray comes closest to."""
        s = closest_on_line_to_ray(g.point, g.direction, o, d)
        if s is None:
            return None
        p = g.point + g.direction * s
        sp, ok = self.cam.project(p)
        if not ok[0]:
            return None
        return float(np.linalg.norm(sp[0] - m)), p

    def pick_guide(self, x, y, px=6.0, hit_t=None):
        """(pixel distance, guide id, world point) of the guide under the cursor, or None."""
        lines, points = self.guide_lines(), self.guide_points()
        if not lines and not points:
            return None
        o, d = self.cam.ray(x, y)
        m = np.array([x, y], dtype=float)
        if hit_t is None:
            fh = self.pick_face(o, d)
            hit_t = fh[0] if fh else np.inf
        occl = self.cam.pixel_size(self.cam.target) * 4 + 1e-3
        best = None
        if points:
            S, ok = self.cam.project(np.array([g.point for g in points]))
            dist = np.where(ok, np.linalg.norm(S - m, axis=1), np.inf)
            k = int(np.argmin(dist))
            if dist[k] < px:
                best = (float(dist[k]), points[k].id, points[k].point)
        for g in lines:
            hit = self._on_guide(g, o, d, m)
            if hit is None or hit[0] >= px:
                continue
            depth = float(np.dot(hit[1] - o, d))
            if depth > hit_t + occl * (1 + depth / max(self.cam.distance, 1)):
                continue                 # behind geometry
            if best is None or hit[0] < best[0]:
                best = (hit[0], g.id, hit[1])
        return best

    # --------------------------------------------------------- picking
    def pick_face(self, o, d, exclude=(), only=None, allow_hidden=None):
        allow = self.show_hidden if allow_hidden is None else allow_hidden
        best = None
        for ctx, ents, _ in self.doc.contexts():
            if ctx in exclude or (only is not None and ctx not in only):
                continue
            c = self.cache.get(ents)
            if c["bbox"] is None or not _ray_box(o, d, c["bbox"][0] - 1e-3, c["bbox"][1] + 1e-3):
                continue
            h = ray_tris(c, o, d, allow)
            if h and (best is None or h[0] < best[0]):
                best = (h[0], ctx, h[1], o + d * h[0])
        return best

    def pick_edge(self, x, y, exclude=(), only=None, px=EDGE_PX, allow_hidden=None):
        allow = self.show_hidden if allow_hidden is None else allow_hidden
        o, d = self.cam.ray(x, y)
        best = None
        for ctx, ents, _ in self.doc.contexts():
            if ctx in exclude or (only is not None and ctx not in only):
                continue
            c = self.cache.get(ents)
            mask = None if allow else ~c["ehidden"]
            dist, k, p = self._nearest_segment(c["A"], c["B"], x, y, o, d, mask)
            if dist is not None and dist < px and (best is None or dist < best[0]):
                best = (dist, ctx, c["eids"][k], p, (c["A"][k], c["B"][k]))
        return best

    def pick_entity(self, x, y):
        """What a click selects: ('edge', ctx, eid) | ('face', ctx, fid, p) | ('group', gid, p) |
        ('guide', ROOT, gid) | ('outside', None) | (None, None)."""
        o, d = self.cam.ray(x, y)
        active = self.doc.active
        fh = self.pick_face(o, d)
        hit_t = fh[0] if fh else np.inf
        eh = self.pick_edge(x, y, only={active}, px=5.0)
        gh = self.pick_guide(x, y, px=4.0, hit_t=hit_t)
        tol = self.cam.pixel_size(self.cam.target) * 4 + 1e-3
        if gh is not None and (eh is None or gh[0] <= eh[0]):
            return ("guide", ROOT, gh[1])
        if eh is not None:
            depth = float(np.dot(eh[3] - o, d))
            if fh is None or depth <= fh[0] + tol * (1 + depth / max(self.cam.distance, 1)):
                return ("edge", eh[1], eh[2])
        if fh is None:
            return (None, None)
        t, ctx, fid, p = fh
        if ctx == active:
            return ("face", ctx, fid, p)
        if active == ROOT:
            return ("group", ctx, p)
        return ("outside", ctx)

    def _nearest_segment(self, A, B, x, y, o, d, mask=None):
        if not len(A):
            return None, None, None
        SA, okA = self.cam.project(A)
        SB, okB = self.cam.project(B)
        ok = okA & okB
        m = np.array([x, y], dtype=float)
        ab = SB - SA
        L2 = np.einsum("ij,ij->i", ab, ab)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.clip(np.einsum("ij,ij->i", m - SA, ab) / L2, 0, 1)
        s = np.where(L2 > 1e-12, s, 0)
        q = SA + ab * s[:, None]
        dist = np.linalg.norm(q - m, axis=1)
        dist = np.where(ok, dist, np.inf)
        if mask is not None:
            dist = np.where(mask, dist, np.inf)
        k = int(np.argmin(dist))
        if not np.isfinite(dist[k]):
            return None, None, None
        a, b = A[k], B[k]
        dvec = b - a
        t = closest_on_line_to_ray(a, dvec, o, d)
        t = 0.5 if t is None else min(1.0, max(0.0, t))
        return float(dist[k]), k, a + dvec * t

    # --------------------------------------------------------- inference
    def infer(self, x, y, anchor=None, plane: Plane | None = None, lock=None,
              exclude=(), extra=None):
        """Best 3D point for the cursor (see SketchUp's inference engine).

        anchor: previous point (enables axis inference)
        plane:  constrain the result to this plane
        lock:   (origin, direction) line the result must lie on
        exclude: contexts (group ids) to ignore
        extra:  additional (n,3) snap points supplied by the tool
        """
        o, d = self.cam.ray(x, y)
        m = np.array([x, y], dtype=float)
        fh = self.pick_face(o, d, exclude)
        hit_t = fh[0] if fh else np.inf
        occl = self.cam.pixel_size(self.cam.target) * 4 + 1e-3

        P, K, A, B, EC = [], [], [], [], []
        for ctx, ents, _ in self.doc.contexts():
            if ctx in exclude:
                continue
            c = self.cache.get(ents)
            use = ~c["ehidden"] if not self.show_hidden else np.ones(len(c["eids"]), dtype=bool)
            if len(c["corners"]):
                P.append(c["corners"])
                K += ["endpoint"] * len(c["corners"])
            if len(c["mid"]):
                P.append(c["mid"][use])
                K += ["midpoint"] * int(use.sum())
            if len(c["A"]):
                A.append(c["A"][use])
                B.append(c["B"][use])
                EC += [(ctx, e) for e, ok in zip(c["eids"], use) if ok]
        P.append(np.zeros((1, 3)))
        K.append("origin")
        if extra is not None and len(extra):
            P.append(np.asarray(extra, float).reshape(-1, 3))
            K += ["endpoint"] * len(np.asarray(extra).reshape(-1, 3))
        A = np.concatenate(A) if A else np.zeros((0, 3))
        B = np.concatenate(B) if B else np.zeros((0, 3))

        guides = self.guide_lines()
        gpts = self.guide_points()
        if gpts:
            P.append(np.array([g.point for g in gpts]))
            K += ["guidepoint"] * len(gpts)
        cross = self._guide_crossings(guides, A, B)
        if len(cross):
            P.append(cross)
            K += ["guide"] * len(cross)
        P = np.concatenate(P)

        def visible(pts):
            depth = np.linalg.norm(pts - o, axis=1) if not self.cam.ortho else (pts - o) @ d
            return depth <= hit_t + occl * (1 + depth / max(self.cam.distance, 1))

        snap = None
        S, ok = self.cam.project(P)
        dist = np.linalg.norm(S - m, axis=1)
        dist = dist + np.array([2.0 if k == "midpoint" else 0.0 for k in K])
        dist = np.where(ok & visible(P), dist, np.inf)
        k = int(np.argmin(dist))
        if dist[k] < PICK_PX:
            snap = Inference(K[k], P[k])

        if lock is not None:
            lo_, ld = lock
            if snap is not None:
                s = float(np.dot(snap.point - lo_, ld) / np.dot(ld, ld))
            else:
                s = closest_on_line_to_ray(lo_, ld, o, d)
                if s is None:
                    s = 0.0
            return Inference("lock", lo_ + ld * s)

        if snap is not None:
            if plane is not None:
                snap.point = plane.project(snap.point)
            return snap

        # axis inference from the anchor
        axis_c = None
        if anchor is not None:
            for i in range(3):
                ax = AXES[i]
                if plane is not None and abs(np.dot(ax, plane.n)) > 1e-6:
                    continue
                s = closest_on_line_to_ray(anchor, ax, o, d)
                if s is None:
                    continue
                p = anchor + ax * s
                sp, okp = self.cam.project(p)
                dd = float(np.linalg.norm(sp[0] - m))
                if okp[0] and dd < PICK_PX and (axis_c is None or dd < axis_c[0]):
                    axis_c = (dd, i, p)

        edge_c = None
        if len(A):
            dist_e, ke, pe = self._nearest_segment(A, B, x, y, o, d)
            if dist_e is not None and dist_e < EDGE_PX and visible(pe[None])[0]:
                if plane is None or abs(plane.distance(pe)) < 1e-6:
                    edge_c = (dist_e, Inference("edge", pe, edge=(A[ke], B[ke]),
                                                ctx=EC[ke][0], eid=EC[ke][1]))

        # the drawing axes behave like edges, as in SketchUp: "On Red Axis" etc.
        axisline_c = None
        for i in range(3):
            if plane is not None and (abs(np.dot(AXES[i], plane.n)) > 1e-6
                                      or abs(plane.distance(np.zeros(3))) > 1e-6):
                continue
            s_ = closest_on_line_to_ray(np.zeros(3), AXES[i], o, d)
            if s_ is None:
                continue
            p = AXES[i] * s_
            sp, okp = self.cam.project(p)
            dd = float(np.linalg.norm(sp[0] - m))
            if okp[0] and dd < AXIS_PX and visible(p[None])[0]                     and (axisline_c is None or dd < axisline_c[0]):
                axisline_c = (dd, Inference("axisline", p, axis=i))

        guide_c = None
        for g in guides:
            hit = self._on_guide(g, o, d, m)
            if hit is None or hit[0] >= EDGE_PX or not visible(hit[1][None])[0]:
                continue
            if plane is not None and abs(plane.distance(hit[1])) > 1e-6:
                continue
            if guide_c is None or hit[0] < guide_c[0]:
                guide_c = (hit[0], Inference("guide", hit[1], guide=g.id))

        # SketchUp combines "on axis" with "on edge": the exact crossing point
        if axis_c is not None and len(A):
            ax = AXES[axis_c[1]]
            best = None
            for kk in range(len(A)):
                a, b = A[kk], B[kk]
                r = b - a
                w = anchor - a
                aa, bb, cc = ax @ ax, ax @ r, r @ r
                dd_, ee = ax @ w, r @ w
                den = aa * cc - bb * bb
                if abs(den) < 1e-12:
                    continue
                s = (bb * ee - cc * dd_) / den
                u = (aa * ee - bb * dd_) / den
                if u < -1e-9 or u > 1 + 1e-9:
                    continue
                p1 = anchor + ax * s
                p2 = a + r * u
                if np.linalg.norm(p1 - p2) > 1e-6:
                    continue
                sp, okp = self.cam.project(p2)
                ddd = float(np.linalg.norm(sp[0] - m))
                if okp[0] and ddd < PICK_PX and visible(p2[None])[0] and (best is None or ddd < best[0]):
                    best = (ddd, p2, kk)
            if best is not None:
                kk = best[2]
                return Inference("edge", best[1], edge=(A[kk], B[kk]), axis=axis_c[1],
                                 ctx=EC[kk][0], eid=EC[kk][1])

        # parallel to an axis from the anchor and on another drawing axis: the exact crossing
        if axis_c is not None and axisline_c is not None:
            j = axisline_c[1].axis
            if j != axis_c[1]:
                hit = line_line_hit(anchor, AXES[axis_c[1]], np.zeros(3), AXES[j])
                if hit is not None:
                    sp, okp = self.cam.project(hit)
                    if okp[0] and float(np.linalg.norm(sp[0] - m)) < PICK_PX:
                        return Inference("axisline", hit, axis=j)

        if axis_c is not None and (edge_c is None or axis_c[0] <= edge_c[0] + 3)                 and (axisline_c is None or axis_c[0] <= axisline_c[0] + 3):
            return Inference("axis", axis_c[2], axis=axis_c[1])
        if axisline_c is not None and (edge_c is None or axisline_c[0] < edge_c[0]):
            return axisline_c[1]
        if edge_c is not None:
            return edge_c[1]
        if guide_c is not None:
            return guide_c[1]

        if plane is not None:
            p = plane.intersect_ray(o, d, allow_behind=True)
            if p is None:
                p = plane.project(o + d * self.cam.distance)
            return Inference("plane", p, plane=plane)

        if fh:
            t, ctx, fid, p = fh
            ents = self.doc.ents(ctx)
            return Inference("face", p, ctx=ctx, fid=fid, plane=ents.face_plane(fid))

        p = GROUND.intersect_ray(o, d)
        if p is not None and np.linalg.norm(p - o) < self.cam.distance * 40:
            return Inference("ground", p, plane=GROUND)
        return Inference("free", o + d * self.cam.distance)


def _ray_box(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    return tmax >= max(tmin, 0.0)
