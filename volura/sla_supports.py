"""Automatic supports and a pad for resin prints.

No Qt in this file. A resin printer prints upside down: every layer is cured against the
bottom of the vat and pulled off it, so every part of the print that starts in mid-air - the
lowest point of an island, an overhang, a sloping underside - needs a support under it, and the
whole print stands on a **pad** on the build plate. This follows the approach of PrusaSlicer's
resin supports, written anew for Volura:

**Support points.** The model is cut into thin slices from the bottom up. A slice that does not
rest on the one below is an *island*; a slice that reaches out further than the one below has an
*overhang*. Every point that is placed supports a circle around it, and the circle grows with
the height above the point (`influence`): a point under a flat underside holds about 3 mm round
it, and the layers above it a little more. Islands get points spread over them on a triangular
grid; the edges of overhangs are sampled and every sample that no point covers yet gets one.

**Heads.** A head is a thin cone from a small sphere whose tip goes `penetration` into the model
to a sphere as thick as the pillar, pointing away from the surface but never more than the
critical angle away from straight down.

**Routing.** Every head first tries its own pillar straight down to the pad; a head whose way
down is blocked by the model bridges sideways to a pillar near it, or walks out at the critical
angle until it finds a free way down, or - when nothing else works - ends on the model itself in
an anchor (a small inverted head). Pillars that stand close together and high are braced with
zig-zag struts, so they do not sway. Every strut is checked against the model with rays: a ring of
rays around the strut, a safety gap wider than it.

**Pad.** The bases of all pillars are joined into one plate under the print, `pad_wall_thickness`
thick, with a brim round it; the model stands `support_object_elevation` above it.

`support(parts, profile, settings)` returns the parts lifted on their supports with the pad and
the supports as manifolds, all standing on z = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import manifold3d as mf
import numpy as np

#: the radius a support point holds against the height above it: (radius mm, height mm)
INFLUENCE = ((3.2, 0.0), (4.0, 3.9), (5.0, 15.0), (6.0, 40.0))
#: islands smaller than this (radius of a circle they fit in) are left out: too small to print
DEBRIS = 0.2
#: mm between the slices the support points are looked for in, at most
DETECT_STEP = 0.1
#: mm of overhang a layer may have over the one below without a support under it
SELF_SUPPORT = 0.1
#: mm kept free between a strut and the model
SAFETY = 0.5
#: rays round a strut when it is checked against the model
RING = 8
#: mm of influence a point of another part can have: points further away are not counted
REACH = 5.0
#: pillars taller than this are braced to a neighbour
BRACE_HEIGHT = 15.0
#: pieces a round strut is made of
SEGMENTS = 16
#: at least this pad, whatever the profile says: a thinner one breaks when it comes off
PAD_MIN = 1.0


@dataclass
class SupportConfig:
    """The sizes of supports and pad, from a resin print profile (PrusaSlicer's names)."""

    head_front: float = 0.8           # mm across the tip of a head
    penetration: float = 0.4          # mm the tip goes into the model
    head_width: float = 3.0           # mm of cone between tip and pillar
    pillar: float = 1.2               # mm across a pillar
    small_pillar: float = 0.6         # part of the pillar a small head and an anchor have
    base_diameter: float = 3.0
    base_height: float = 1.0
    critical_angle: float = 45.0      # degrees from straight down a strut may lean
    max_bridge: float = 10.0          # mm
    max_bridges_on_pillar: int = 3
    max_link: float = 10.0            # mm between pillars that are braced
    elevation: float = 5.0            # mm the model stands above the pad
    density: float = 100.0            # % of support points
    pad: bool = True
    pad_thickness: float = 1.0
    pad_brim: float = 1.6
    buildplate_only: bool = False

    @classmethod
    def from_profile(cls, values: dict, **override) -> "SupportConfig":
        def num(key, default):
            try:
                return float(str(values.get(key, default)).rstrip("%"))
            except ValueError:
                return default
        c = cls(head_front=num("support_head_front_diameter", 0.8),
                penetration=num("support_head_penetration", 0.4),
                head_width=num("support_head_width", 3.0),
                pillar=num("support_pillar_diameter", 1.2),
                small_pillar=num("support_small_pillar_diameter_percent", 60.0) / 100.0,
                base_diameter=num("support_base_diameter", 3.0),
                base_height=num("support_base_height", 1.0),
                critical_angle=num("support_critical_angle", 45.0),
                max_bridge=num("support_max_bridge_length", 10.0),
                max_bridges_on_pillar=int(num("support_max_bridges_on_pillar", 3)),
                max_link=num("support_max_pillar_link_distance", 10.0),
                elevation=num("support_object_elevation", 5.0),
                density=num("support_points_density_relative", 100.0),
                pad=str(values.get("pad_enable", "1")) not in ("0", "false"),
                pad_thickness=max(PAD_MIN, num("pad_wall_thickness", 1.0)),
                pad_brim=num("pad_brim_size", 1.6),
                buildplate_only=str(values.get("support_buildplate_only", "0")) in ("1", "true"))
        for k, v in override.items():
            setattr(c, k, v)
        return c


def influence(height: float, density: float = 100.0) -> float:
    """The radius a support point holds `height` mm above it."""
    pts = INFLUENCE
    if height <= pts[0][1]:
        r = pts[0][0]
    elif height >= pts[-1][1]:
        r = pts[-1][0]
    else:
        for (r0, h0), (r1, h1) in zip(pts, pts[1:]):
            if h0 <= height <= h1:
                r = r0 + (r1 - r0) * (height - h0) / (h1 - h0)
                break
    return r / math.sqrt(max(0.1, density / 100.0))


# ---- the model as triangles, for rays
class Mesh:
    """The triangles of a manifold with a grid over them in plan, for fast rays."""

    def __init__(self, solid: mf.Manifold, cell: float = 2.0):
        m = solid.to_mesh()
        v = np.asarray(m.vert_properties, dtype=float)[:, :3]
        t = np.asarray(m.tri_verts, dtype=np.int64)
        self.a, self.b, self.c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        n = np.cross(self.b - self.a, self.c - self.a)
        ln = np.linalg.norm(n, axis=1)
        self.normal = n / np.where(ln > 0, ln, 1.0)[:, None]
        self.lo = np.minimum(np.minimum(self.a, self.b), self.c)
        self.hi = np.maximum(np.maximum(self.a, self.b), self.c)
        self.cell = cell
        self.grid: dict[tuple, np.ndarray] = {}
        i0 = np.floor(self.lo[:, :2] / cell).astype(int)
        i1 = np.floor(self.hi[:, :2] / cell).astype(int)
        cells: dict[tuple, list] = {}
        for k in range(len(t)):
            for gx in range(i0[k, 0], i1[k, 0] + 1):
                for gy in range(i0[k, 1], i1[k, 1] + 1):
                    cells.setdefault((gx, gy), []).append(k)
        self.grid = {key: np.asarray(val, dtype=np.int64) for key, val in cells.items()}
        self.zmin = float(v[:, 2].min()) if len(v) else 0.0

    def _near(self, xmin, ymin, xmax, ymax) -> np.ndarray:
        gx0, gy0 = int(math.floor(xmin / self.cell)), int(math.floor(ymin / self.cell))
        gx1, gy1 = int(math.floor(xmax / self.cell)), int(math.floor(ymax / self.cell))
        found = [self.grid[(gx, gy)] for gx in range(gx0, gx1 + 1) for gy in range(gy0, gy1 + 1)
                 if (gx, gy) in self.grid]
        return np.unique(np.concatenate(found)) if found else np.zeros(0, dtype=np.int64)

    def vertical(self, x, y):
        """Every place the vertical line through (x, y) meets the model: (z, triangle) sorted."""
        idx = self._near(x, y, x, y)
        if not len(idx):
            return []
        a, b, c = self.a[idx], self.b[idx], self.c[idx]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-12
        d = np.where(ok, d, 1.0)
        l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
        l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
        l3 = 1.0 - l1 - l2
        inside = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2]
        return sorted(zip(z[inside].tolist(), idx[inside].tolist()))

    def segment_hit(self, p0, p1) -> float | None:
        """The fraction along p0 -> p1 where it first meets the model, or None."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        idx = self._near(lo[0], lo[1], hi[0], hi[1])
        if not len(idx):
            return None
        keep = idx[np.all(self.hi[idx] >= lo - 1e-9, axis=1) & np.all(self.lo[idx] <= hi + 1e-9,
                                                                      axis=1)]
        if not len(keep):
            return None
        a, b, c = self.a[keep], self.b[keep], self.c[keep]
        d = p1 - p0
        e1, e2 = b - a, c - a
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = p0 - a
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ d)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        return float(t[hit].min()) if hit.any() else None

    def beam_clear(self, p0, p1, radius) -> bool:
        """Whether a strut of `radius` from p0 to p1, with the safety gap round it, misses the
        model: a ray along its axis and a ring of rays round it."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        if ln < 1e-9:
            return True
        d /= ln
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, ref)
        u /= np.linalg.norm(u)
        w = np.cross(d, u)
        rr = radius + SAFETY
        offsets = [np.zeros(3)] + [rr * (math.cos(k * 2 * math.pi / RING) * u
                                         + math.sin(k * 2 * math.pi / RING) * w)
                                   for k in range(RING)]
        return all(self.segment_hit(p0 + o, p1 + o) is None for o in offsets)


# ---- support points
@dataclass
class Point:
    pos: np.ndarray                   # on the surface of the model
    normal: np.ndarray                # of that surface, pointing out (down-ish)
    island: bool = False


def _sample_grid(cs: mf.CrossSection, spacing: float) -> list[tuple[float, float]]:
    """Points of a triangular grid inside a cross-section."""
    if cs.is_empty():
        return []
    x0, y0, x1, y1 = cs.bounds()
    out = []
    dy = spacing * math.sqrt(3.0) / 2.0
    row = 0
    y = y0 + 0.5 * dy
    while y <= y1:
        x = x0 + (0.5 * spacing if row % 2 else 0.25 * spacing)
        while x <= x1:
            out.append((x, y))
            x += spacing
        y += dy
        row += 1
    if not out:
        return []
    pts = np.asarray(out)
    return [tuple(p) for p in pts[_inside(cs, pts)]]


def _inside(cs: mf.CrossSection, pts) -> np.ndarray:
    """Which of the points (n, 2) lie inside a cross-section (even-odd over its outlines)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    inside = np.zeros(len(pts), dtype=bool)
    for poly in cs.to_polygons():
        poly = np.asarray(poly, dtype=float)
        xs, ys = poly[:, 0][None, :], poly[:, 1][None, :]
        xs2, ys2 = np.roll(xs, -1, axis=1), np.roll(ys, -1, axis=1)
        span = np.where(ys2 != ys, ys2 - ys, 1.0)
        cross = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / span + xs)
        inside ^= (np.count_nonzero(cross, axis=1) % 2).astype(bool)
    return inside


def _contains(cs: mf.CrossSection, p) -> bool:
    """Whether a point lies inside a cross-section."""
    return bool(_inside(cs, [p])[0])


def _edge_samples(cs: mf.CrossSection, spacing: float) -> list[tuple[float, float]]:
    """Points along the outlines of a cross-section, `spacing` apart, just inside it."""
    inner = cs.offset(-0.05, mf.JoinType.Round)
    out = []
    for poly in (inner if not inner.is_empty() else cs).to_polygons():
        poly = np.asarray(poly)
        if len(poly) < 2:
            continue
        seg = np.roll(poly, -1, axis=0) - poly
        lengths = np.hypot(seg[:, 0], seg[:, 1])
        total = float(lengths.sum())
        n = max(1, int(total / spacing))
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        for k in range(n):
            s = (k + 0.5) * total / n
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(poly) - 1)
            f = (s - cum[i]) / lengths[i] if lengths[i] > 0 else 0.0
            out.append(tuple(poly[i] + f * seg[i]))
    return out


def support_points(solid: mf.Manifold, mesh: Mesh, cfg: SupportConfig, step: float):
    """The points the model needs a support under, on its surface, from the bottom up."""
    lo = np.asarray(solid.bounding_box()[:3])
    hi = np.asarray(solid.bounding_box()[3:])
    step = max(step, DETECT_STEP)
    r0 = influence(0.0, cfg.density)
    head_r = 0.5 * cfg.head_front
    points: list[Point] = []
    placed = np.zeros((0, 3))
    below = mf.CrossSection()
    z = lo[2] + 0.5 * step
    while z < hi[2]:
        layer = solid.slice(z)
        grown = below.offset(SELF_SUPPORT * step / DETECT_STEP, mf.JoinType.Round) \
            if not below.is_empty() else below
        for island in layer.decompose():
            rests = not (island ^ below).is_empty() if not below.is_empty() else False
            if not rests:
                # an island: spread points over it, at least one. A tiny one - the tip of a V,
                # the point of a cone - gets one in its middle: it is where a whole part starts
                x0, y0, x1, y1 = island.bounds()
                middle = [(0.5 * (x0 + x1), 0.5 * (y0 + y1))]
                if island.area() < math.pi * DEBRIS ** 2:
                    cand = middle
                else:
                    inner = island.offset(-head_r, mf.JoinType.Round)
                    area = inner if not inner.is_empty() else island
                    cand = (_sample_grid(area, r0 * 1.2) + _edge_samples(area, r0 * 1.2)
                            or middle)
                fresh = True
            else:
                over = island - grown
                if over.is_empty() or over.area() < 0.01:
                    continue
                cand = _edge_samples(over, 1.0) + _sample_grid(over, r0)
                fresh = False
            near = island.offset(REACH, mf.JoinType.Round)
            before = len(points)
            for x, y in cand:
                if len(placed):
                    dz = z - placed[:, 2]
                    dxy = np.hypot(placed[:, 0] - x, placed[:, 1] - y)
                    covered = (dz >= -1e-9) & (dxy <= _influences(dz, cfg.density))
                    holders = placed[np.nonzero(covered)[0][:4], :2]
                    if covered.any() and _inside(near, holders).any():
                        continue
                p = _on_surface(mesh, x, y, z, step)
                if p is None:
                    continue
                points.append(Point(p[0], p[1], fresh))
                placed = np.vstack([placed, p[0][None, :]])
            if fresh and len(points) == before and not _held(placed, island, z, cfg):
                # an island whose underside the rays missed (a sharp edge, a point): it still
                # needs one, straight under its middle, pointing down
                x0, y0, x1, y1 = island.bounds()
                p = _lowest(mesh, 0.5 * (x0 + x1), 0.5 * (y0 + y1), z, step)
                if p is not None:
                    points.append(Point(p, np.array([0.0, 0.0, -1.0]), True))
                    placed = np.vstack([placed, p[None, :]])
        below = layer
        z += step
    return points


_INF_H = np.array([h for _r, h in INFLUENCE])
_INF_R = np.array([r for r, _h in INFLUENCE])


def _influences(heights, density) -> np.ndarray:
    """`influence` for many heights at once."""
    return np.interp(heights, _INF_H, _INF_R) / math.sqrt(max(0.1, density / 100.0))


def _held(placed, island, z, cfg) -> bool:
    """Whether a point already placed holds the middle of an island."""
    if not len(placed):
        return False
    x0, y0, x1, y1 = island.bounds()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    dz = z - placed[:, 2]
    dxy = np.hypot(placed[:, 0] - cx, placed[:, 1] - cy)
    return bool(((dz >= -1e-9) & (dxy <= _influences(dz, cfg.density))).any())


def _lowest(mesh: Mesh, x, y, z, step):
    """The model under or at (x, y) near height z, whatever way its surface faces there."""
    hits = [hz for hz, _tri in mesh.vertical(x, y) if abs(hz - z) <= 2.0 * step + 1e-6]
    if not hits:
        return None
    return np.array([x, y, min(hits)])


def _on_surface(mesh: Mesh, x, y, z, step):
    """The underside of the model at (x, y) near height z, and its normal: (pos, normal)."""
    best = None
    for hz, tri in mesh.vertical(x, y):
        n = mesh.normal[tri]
        if n[2] >= -0.05:                      # not an underside
            continue
        if abs(hz - z) <= 1.5 * step + 1e-6 and (best is None or abs(hz - z) < abs(best[0] - z)):
            best = (hz, tri)
    if best is None:
        return None
    return np.array([x, y, best[0]]), mesh.normal[best[1]].copy()


# ---- the tree
@dataclass
class Head:
    tip: np.ndarray                   # the support point
    dir: np.ndarray                   # from the model out
    r_pin: float
    r_back: float
    junction: np.ndarray              # where the pillar or the bridge attaches

    def solid(self, penetration) -> mf.Manifold:
        c1 = self.tip + self.dir * (self.r_pin - penetration)
        return mf.Manifold.batch_hull([_sphere(c1, self.r_pin), _sphere(self.junction,
                                                                         self.r_back)])


@dataclass
class Pillar:
    x: float
    y: float
    top: float
    bottom: float
    r: float
    bridges: int = 0
    base: bool = True
    parts: list = field(default_factory=list)


def _sphere(c, r) -> mf.Manifold:
    return mf.Manifold.sphere(r, SEGMENTS).translate(tuple(float(v) for v in c))


def _strut(p0, p1, r0, r1=None) -> mf.Manifold:
    """A round strut (cylinder or cone) from p0 to p1 with balls at its ends."""
    r1 = r0 if r1 is None else r1
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ln = float(np.linalg.norm(d))
    if ln < 1e-6:
        return _sphere(p0, max(r0, r1))
    return mf.Manifold.batch_hull([_sphere(p0, r0), _sphere(p1, r1)])


def _head_dir(normal, critical) -> np.ndarray:
    """The direction of a head: the normal, but never further than `critical` from straight down."""
    down = np.array([0.0, 0.0, -1.0])
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    ang = math.acos(max(-1.0, min(1.0, float(n @ down))))
    lim = math.radians(critical)
    if ang <= lim:
        return n
    side = n - (n @ down) * down
    sl = np.linalg.norm(side)
    if sl < 1e-9:
        return down
    side /= sl
    return math.cos(lim) * down + math.sin(lim) * side


def build_tree(points, mesh: Mesh, cfg: SupportConfig, ground: float):
    """Heads, pillars, bridges and anchors for the support points: (struts, pillars, dropped)."""
    struts: list[mf.Manifold] = []
    pillars: list[Pillar] = []
    dropped = 0
    r_back = 0.5 * cfg.pillar
    r_pin = 0.5 * cfg.head_front
    tan_c = math.tan(math.radians(cfg.critical_angle))

    def make_head(pt, rb):
        for d in (_head_dir(pt.normal, cfg.critical_angle), np.array([0.0, 0.0, -1.0])):
            c1 = pt.pos + d * (r_pin - cfg.penetration)
            junction = c1 + d * (r_pin + cfg.head_width + rb)
            if junction[2] <= ground + cfg.base_height:
                continue
            start = pt.pos + d * (r_pin + 0.2)
            if mesh.beam_clear(start, junction, rb * 0.5):
                return Head(pt.pos, d, r_pin, rb, junction)
        return None

    def drop_clear(x, y, z_top, r):
        return mesh.beam_clear((x, y, z_top), (x, y, ground), r)

    def to_pillar(head: Head, limit: float):
        """Bridge the head to a pillar near it, going down at least the critical angle."""
        j = head.junction
        best = None
        for p in pillars:
            if p.bridges >= cfg.max_bridges_on_pillar:
                continue
            dist = math.hypot(p.x - j[0], p.y - j[1])
            if dist < 1e-6 or dist > limit:
                continue
            tz = j[2] - dist * tan_c
            if tz < ground + 4 * p.r:
                continue
            if best is None or dist < best[0]:
                best = (dist, p, tz)
        if best is None:
            return False
        _dist, p, tz = best
        target = np.array([p.x, p.y, tz])
        if tz > p.top:
            if not mesh.beam_clear((p.x, p.y, p.top), target, p.r):
                return False
        if not mesh.beam_clear(j, target, head.r_back):
            return False
        struts.append(_strut(j, target, head.r_back, p.r))
        if tz > p.top:
            p.top = tz
        p.bridges += 1
        return True

    def own_pillar(x, y, top, r):
        if not drop_clear(x, y, top, r):
            return False
        pillars.append(Pillar(x, y, top, ground, r))
        return True

    def escape(head: Head):
        """Walk out at the critical angle until there is a free way down."""
        j = head.junction
        horizontal = head.dir[:2] if np.linalg.norm(head.dir[:2]) > 1e-6 else np.array([1.0, 0])
        dirs = []
        base_ang = math.atan2(horizontal[1], horizontal[0])
        for k in range(8):
            a = base_ang + k * math.pi / 4.0
            dirs.append(np.array([math.cos(a), math.sin(a)]))
        step = max(head.r_back * 2, 1.0)
        for d2 in dirs:
            dist = step
            while dist <= cfg.max_bridge:
                end = np.array([j[0] + d2[0] * dist, j[1] + d2[1] * dist, j[2] - dist * tan_c])
                if end[2] < ground + 4 * head.r_back:
                    break
                if not mesh.beam_clear(j, end, head.r_back):
                    break
                if drop_clear(end[0], end[1], end[2], head.r_back):
                    struts.append(_strut(j, end, head.r_back))
                    pillars.append(Pillar(end[0], end[1], end[2], ground, head.r_back))
                    return True
                dist += step
        return False

    def anchor(head: Head):
        """A pillar down onto the model below, ending in a small inverted head."""
        if cfg.buildplate_only:
            return False
        j = head.junction
        hits = [z for z, tri in mesh.vertical(j[0], j[1]) if z < j[2] - 1.0
                and mesh.normal[tri][2] > 0.3]
        if not hits:
            return False
        surface = max(hits)
        small = max(0.2, cfg.small_pillar * r_back)
        end = np.array([j[0], j[1], surface + r_pin + cfg.head_width])
        if end[2] >= j[2] or not mesh.beam_clear(j, end, small):
            return False
        tip = np.array([j[0], j[1], surface - cfg.penetration + r_pin])
        struts.append(_strut(j, end, head.r_back, small))
        struts.append(_strut(end, tip, small, r_pin))
        return True

    heads = []
    for pt in points:
        h = make_head(pt, r_back)
        if h is None:
            h = make_head(pt, max(0.2, cfg.small_pillar * r_back))
        if h is None:
            dropped += 1
            continue
        heads.append(h)
    # the lowest heads first, so the pillars stand before the heads above bridge to them
    heads.sort(key=lambda h: h.junction[2])
    for h in heads:
        struts.append(h.solid(cfg.penetration))
        j = h.junction
        if to_pillar(h, min(cfg.max_bridge, 2.0 * 0.5 * cfg.base_diameter + 2.0)):
            continue
        if own_pillar(j[0], j[1], j[2], h.r_back):
            continue
        if to_pillar(h, cfg.max_bridge) or escape(h) or anchor(h):
            continue
        struts.pop()
        dropped += 1
    _brace(pillars, mesh, cfg, ground, struts, tan_c)
    for p in pillars:
        struts.append(_strut((p.x, p.y, p.bottom), (p.x, p.y, p.top), p.r))
        if p.base:
            struts.append(mf.Manifold.cylinder(cfg.base_height, 0.5 * cfg.base_diameter, p.r,
                                               SEGMENTS).translate((p.x, p.y, ground)))
    return struts, pillars, dropped


def _brace(pillars, mesh, cfg, ground, struts, tan_c):
    """Zig-zag struts between tall pillars close to each other."""
    linked = set()
    for i, a in enumerate(pillars):
        if a.top - ground < BRACE_HEIGHT:
            continue
        near = sorted((math.hypot(b.x - a.x, b.y - a.y), j) for j, b in enumerate(pillars)
                      if j != i)
        count = 0
        for dist, j in near:
            if dist > cfg.max_link or count >= 2:
                break
            if dist < 2 * a.r or (min(i, j), max(i, j)) in linked:
                continue
            b = pillars[j]
            lo = ground + cfg.base_height + 1.0
            hi = min(a.top, b.top) - 1.0
            rise = dist * tan_c
            if hi - lo < rise:
                continue
            linked.add((min(i, j), max(i, j)))
            count += 1
            z, here, there = lo, a, b
            while z + rise <= hi:
                p0 = (here.x, here.y, z)
                p1 = (there.x, there.y, z + rise)
                if mesh.beam_clear(p0, p1, 0.5 * min(a.r, b.r)):
                    struts.append(_strut(p0, p1, 0.5 * min(a.r, b.r)))
                z += rise
                here, there = there, here


def _pad(pillars, footprint: mf.CrossSection | None, cfg: SupportConfig, ground: float):
    """The plate under the print: the bases of the pillars (and the footprint of a model that
    stands on it) joined, with a brim, closed over small gaps, `pad_thickness` thick."""
    shapes = [mf.CrossSection.circle(0.5 * cfg.base_diameter, SEGMENTS).translate((p.x, p.y))
              for p in pillars if p.base]
    if footprint is not None and not footprint.is_empty():
        shapes.append(footprint)
    if not shapes:
        return None
    area = mf.CrossSection.batch_boolean(shapes, mf.OpType.Add)
    merge = 3.0
    area = area.offset(cfg.pad_brim + merge, mf.JoinType.Round).offset(-merge, mf.JoinType.Round)
    outer = [poly for poly in area.to_polygons()]
    area = mf.CrossSection(outer, mf.FillRule.Positive)       # no holes in the pad
    return area.extrude(cfg.pad_thickness).translate((0.0, 0.0, ground - cfg.pad_thickness))


@dataclass
class Supported:
    """The print as it stands on the plate: the parts, the supports, the pad, and the numbers."""

    parts: list
    supports: mf.Manifold | None
    pad: mf.Manifold | None
    points: int = 0
    pillars: int = 0
    dropped: int = 0

    @property
    def all(self) -> list:
        out = list(self.parts)
        if self.supports is not None and not self.supports.is_empty():
            out.append(self.supports)
        if self.pad is not None and not self.pad.is_empty():
            out.append(self.pad)
        return out


def support(parts, cfg: SupportConfig, supports_on: bool = True, step: float = 0.1) -> Supported:
    """The parts on supports and a pad, standing on z = 0.

    Without supports the parts stand on the plate, on the pad when there is one.
    """
    solid = mf.Manifold.batch_boolean(list(parts), mf.OpType.Add) if len(parts) > 1 else parts[0]
    zmin = float(solid.bounding_box()[2])
    if not supports_on:
        lift = cfg.pad_thickness if cfg.pad else 0.0
        moved = [p.translate((0.0, 0.0, lift - zmin)) for p in parts]
        pad = None
        if cfg.pad:
            foot = solid.translate((0.0, 0.0, -zmin)).slice(0.01)
            pad = _pad([], foot, cfg, cfg.pad_thickness)
        return Supported(moved, None, pad)
    ground = (cfg.pad_thickness if cfg.pad else 0.0)
    lifted = solid.translate((0.0, 0.0, ground + cfg.elevation - zmin))
    moved = [p.translate((0.0, 0.0, ground + cfg.elevation - zmin)) for p in parts]
    mesh = Mesh(lifted)
    pts = support_points(lifted, mesh, cfg, step)
    struts, pillars, dropped = build_tree(pts, mesh, cfg, ground)
    supports = mf.Manifold.batch_boolean(struts, mf.OpType.Add) if struts else None
    if supports is not None:
        # the supports stop at the plate
        supports = supports ^ mf.Manifold.cube([1e4, 1e4, 1e4]).translate((-5e3, -5e3, 0.0))
        if supports.is_empty():
            supports = None
    pad = _pad(pillars, None, cfg, ground) if cfg.pad else None
    return Supported(moved, supports, pad, len(pts), len(pillars), dropped)
