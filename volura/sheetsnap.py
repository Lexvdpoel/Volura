"""Snapping on the 2D worksheet, the way inference does it in 3D.

The worksheet has the same help as drawing in 3D: the cursor is pulled to a point
that really matters — the endpoint of a line, the middle of a segment, the centre of
a circle or arc, the intersection of two segments, the nearest point on a path or a
grid crossing. In addition the cursor is aligned horizontally or vertically with an
endpoint, with a dotted guide line along with it.

Everything works in millimetres with Y up; the thresholds are in pixels, so snapping
feels the same at every zoom (`distance in mm = pixels / view.scale`).

`SnapEngine` keeps the points of the drawing in a cache that is dropped as soon as
the document changes; `snap()` returns a `Snap` with the point, the kind, the label
for the tooltip and optionally the guide line.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainterPath, QPen

from .i18n import tr
from .vector2d import path_segments

GRID_MIN_PX = 8.0            # grid lines must be at least this far apart

ENDPOINT_PX = 9.0
INTERSECT_PX = 9.0
CENTER_PX = 9.0
MIDPOINT_PX = 7.0
ONPATH_PX = 6.0
GRID_PX = 5.0
ALIGN_PX = 4.0

#: which kind wins when more are within reach, as in SketchUp
PRIORITY = ("endpoint", "intersection", "center", "midpoint", "onpath")

#: where the points of that kind are in the cache of `SnapEngine`
FEATURE_OF = {"endpoint": "ends", "center": "centers", "midpoint": "mids"}

LABELS = {
    "endpoint": tr("Endpoint"),
    "midpoint": tr("Midpoint"),
    "center": tr("Center"),
    "intersection": tr("Intersection"),
    "onpath": tr("On Path"),
    "grid": tr("Grid"),
    "align": tr("Aligned"),
}

COLORS = {
    "endpoint": "#e2574c",
    "intersection": "#d23b3b",
    "center": "#2f6fd6",
    "midpoint": "#8e44ad",
    "onpath": "#2e7d32",
    "grid": "#7a8794",
    "align": "#f08a24",
}

SHAPES = {
    "endpoint": "square",
    "intersection": "cross",
    "center": "ring",
    "midpoint": "diamond",
    "onpath": "dot",
    "grid": "dot",
    "align": "dot",
}


def grid_step(scale: float) -> float:
    """The grid spacing in mm at this zoom: 1, 5, 10, 50, 100, 500, ... mm."""
    for e in range(-4, 8):
        for m in (1.0, 5.0):
            step = m * 10 ** e
            if step * scale >= GRID_MIN_PX:
                return step
    return 10.0 ** 8


class Snap:
    """The result of one snap attempt: a point, the kind and a guide line."""

    __slots__ = ("point", "kind", "label", "guide", "raw")

    def __init__(self, point, kind="", guide=None, raw=None):
        self.point = np.asarray(point, dtype=float)
        self.kind = kind
        self.label = LABELS.get(kind, "")
        self.guide = guide
        self.raw = np.asarray(point, dtype=float) if raw is None else np.asarray(raw, float)

    def __bool__(self):
        return bool(self.kind)

    def __repr__(self):
        return f"Snap({self.kind}, {self.point.round(3).tolist()})"


def _closest_on_segment(p, a, b):
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 <= 1e-30 else max(0.0, min(1.0, float((p - a) @ d) / L2))
    return a + t * d, t


def seg_intersection(p1, p2, p3, p4):
    """The point where segment `p1`-`p2` crosses `p3`-`p4`, or None."""
    r, s = p2 - p1, p4 - p3
    den = float(r[0] * s[1] - r[1] * s[0])
    if abs(den) <= 1e-12:
        return None
    q = p3 - p1
    t = float(q[0] * s[1] - q[1] * s[0]) / den
    u = float(q[0] * r[1] - q[1] * r[0]) / den
    if not (-1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9):
        return None
    return p1 + t * r


class SnapEngine:
    """Finds the snap point at the cursor: geometry first, then grid and alignment."""

    def __init__(self, view):
        self.view = view
        self._features = None
        try:
            view.doc.changed.connect(self.invalidate)
        except Exception:                       # a document without a signal (tests)
            pass

    # -- the points of the drawing ---------------------------------------

    def invalidate(self):
        self._features = None

    def _paths(self):
        """Visible paths you may snap to (you may follow a locked layer)."""
        out = []
        for p in self.view.doc.sheet.paths:
            if len(p.pts) < 2 or not self.view._visible(p):
                continue
            out.append(p)
        return out

    def features(self):
        """Endpoints, midpoints, centres and segments, once per drawing."""
        if self._features is not None:
            return self._features
        ends, mids, centers, segs = [], [], [], []
        for path in self._paths():
            pts = np.asarray(path.pts, dtype=float)
            if len(pts) < 2:
                continue
            ends.extend(pts.tolist())
            for a, b in path_segments(pts, path.closed):
                segs.append((a, b))
                mids.append((a + b) * 0.5)
            c = (path.meta or {}).get("center")
            if c is not None and len(c) == 2:
                centers.append(np.asarray(c, dtype=float).tolist())
        self._features = {
            "ends": np.asarray(ends, dtype=float).reshape(-1, 2),
            "mids": np.asarray(mids, dtype=float).reshape(-1, 2),
            "centers": np.asarray(centers, dtype=float).reshape(-1, 2),
            "segs": np.asarray(segs, dtype=float).reshape(-1, 2, 2),
        }
        return self._features

    def intersections(self, point=None, tol=None):
        """Intersections of the segments with each other.

        With `point` and `tol` only those in the square around it: an intersection
        within `tol` of the cursor lies on two segments whose bounding box touches that
        square, so the rest need not be checked.
        """
        S = self.features()["segs"]
        if len(S) < 2:
            return np.zeros((0, 2))
        lo, hi = np.minimum(S[:, 0], S[:, 1]), np.maximum(S[:, 0], S[:, 1])
        if point is None or tol is None:
            idx = np.arange(len(S))
        else:
            near = ((lo[:, 0] <= point[0] + tol) & (hi[:, 0] >= point[0] - tol)
                    & (lo[:, 1] <= point[1] + tol) & (hi[:, 1] >= point[1] - tol))
            idx = np.nonzero(near)[0]
        pts = []
        for k in range(len(idx) - 1):
            a1, a2 = S[int(idx[k])]
            for m in range(k + 1, len(idx)):
                b1, b2 = S[int(idx[m])]
                if _shares(a1, a2, b1, b2):
                    continue
                q = seg_intersection(a1, a2, b1, b2)
                if q is not None and np.isfinite(q).all():
                    pts.append(q.tolist())
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    # -- snapping ---------------------------------------------------------

    def snap(self, x=None, y=None, point=None, anchor=None, allow_grid=True,
             allow_align=True) -> Snap:
        """The snap point at screen point (`x`, `y`) or world point `point`."""
        raw = (np.asarray(point, dtype=float) if point is not None
               else self.view.to_world_xy(x, y))
        scale = max(float(self.view.scale), 1e-9)
        f = self.features()
        for kind in PRIORITY:
            tol = {"endpoint": ENDPOINT_PX, "intersection": INTERSECT_PX,
                   "center": CENTER_PX, "midpoint": MIDPOINT_PX,
                   "onpath": ONPATH_PX}[kind] / scale
            if kind == "onpath":
                got = self._on_path(raw, f["segs"], tol)
            elif kind == "intersection":
                got = _nearest(raw, self.intersections(raw, tol), tol)
            else:
                got = _nearest(raw, f[FEATURE_OF[kind]], tol)
            if got is not None:
                return Snap(got, kind, raw=raw)
        if allow_align:
            got = self._aligned(raw, anchor, f["ends"], ALIGN_PX / scale)
            if got is not None:
                point, guide = got
                return Snap(point, "align", guide=guide, raw=raw)
        if allow_grid and getattr(self.view, "snap_to_grid", True):
            step = grid_step(self.view.scale)
            q = np.round(raw / step) * step
            if float(np.hypot(*(q - raw))) * scale <= GRID_PX:
                return Snap(q, "grid", raw=raw)
        return Snap(raw, "", raw=raw)

    @staticmethod
    def _on_path(p, segs, tol):
        if not len(segs):
            return None
        a, b = segs[:, 0], segs[:, 1]
        d = b - a
        L2 = np.maximum(d[:, 0] ** 2 + d[:, 1] ** 2, 1e-18)
        t = np.clip(((p[0] - a[:, 0]) * d[:, 0] + (p[1] - a[:, 1]) * d[:, 1]) / L2, 0.0, 1.0)
        proj = a + d * t[:, None]
        dist = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
        i = int(np.argmin(dist))
        return proj[i] if float(dist[i]) <= tol else None

    @staticmethod
    def _aligned(raw, anchor, ends, tol):
        """Horizontally or vertically in line with the previous click or an endpoint."""
        refs = [np.asarray(anchor, dtype=float)] if anchor is not None else []
        if len(ends):
            near_y = ends[np.abs(ends[:, 1] - raw[1]) <= tol]
            near_x = ends[np.abs(ends[:, 0] - raw[0]) <= tol]
            if len(near_y):
                refs.append(near_y[int(np.argmin(np.abs(near_y[:, 1] - raw[1])))])
            if len(near_x):
                refs.append(near_x[int(np.argmin(np.abs(near_x[:, 0] - raw[0])))])
        for ref in refs:
            if abs(raw[1] - ref[1]) <= tol:
                return np.array([raw[0], ref[1]]), ("h", ref)
            if abs(raw[0] - ref[0]) <= tol:
                return np.array([ref[0], raw[1]]), ("v", ref)
        return None


def _nearest(p, pts, tol):
    if pts is None or not len(pts):
        return None
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    i = int(np.argmin(d))
    return pts[i] if float(d[i]) <= tol else None


def _shares(a1, a2, b1, b2) -> bool:
    """Do the segments touch in a shared point? Then there is nothing to snap to."""
    for p in (a1, a2):
        for q in (b1, b2):
            if abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9:
                return True
    return False


# --------------------------------------------------------------------------
#  the marker at the snap point
# --------------------------------------------------------------------------

def paint_snap(p, view, snap):
    """The coloured marker with tooltip and the dotted guide line, as in 3D."""
    if snap is None or not getattr(snap, "kind", ""):
        return
    if snap.guide:
        _paint_guide(p, view, snap.guide, snap.point)
    pos = view.to_screen(snap.point)
    color = QColor(COLORS.get(snap.kind, "#2b2b2b"))
    p.setPen(QPen(QColor(35, 35, 35), 1.2))
    p.setBrush(color)
    x, y = float(pos.x()), float(pos.y())
    shape = SHAPES.get(snap.kind, "dot")
    if shape == "square":
        p.drawRect(x - 3.5, y - 3.5, 7, 7)
    elif shape == "diamond":
        path = QPainterPath(QPointF(x, y - 4.5))
        for px, py in ((x + 4.5, y), (x, y + 4.5), (x - 4.5, y)):
            path.lineTo(px, py)
        path.closeSubpath()
        p.drawPath(path)
    elif shape == "ring":
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(x, y), 5.0, 5.0)
    elif shape == "cross":
        pen = QPen(color, 2.2)
        p.setPen(pen)
        p.drawLine(QPointF(x - 5, y - 5), QPointF(x + 5, y + 5))
        p.drawLine(QPointF(x - 5, y + 5), QPointF(x + 5, y - 5))
    else:
        p.drawEllipse(QPointF(x, y), 3.5, 3.5)
    p.setBrush(Qt.NoBrush)
    if snap.label:
        _paint_tip(p, pos, snap.label)


def _paint_guide(p, view, guide, point):
    direction, ref = guide
    a = view.to_screen(ref)
    b = view.to_screen(point)
    pen = QPen(QColor(COLORS["align"]))
    pen.setStyle(Qt.DashLine)
    pen.setWidthF(1.2)
    p.setPen(pen)
    if direction == "h":
        p.drawLine(QPointF(0.0, a.y()), QPointF(float(view.width()), a.y()))
        p.drawLine(a, b)
    else:
        p.drawLine(QPointF(a.x(), 0.0), QPointF(a.x(), float(view.height())))
        p.drawLine(a, b)


def _paint_tip(p, pos, text):
    """The small label next to the snap point, like the tooltip of inference."""
    font = QFont(p.font())
    font.setPointSizeF(max(7.5, font.pointSizeF() * 0.9))
    p.setFont(font)
    w = 8.0 * max(len(text), 1) + 10.0
    x = float(pos.x()) + 14.0
    y = float(pos.y()) - 26.0
    if x + w > p.viewport().width() - 4:
        x = float(pos.x()) - w - 14.0
    if y < 2:
        y = float(pos.y()) + 12.0
    p.setPen(QPen(QColor("#5a6472")))
    p.setBrush(QColor(255, 255, 230, 235))
    p.drawRoundedRect(x, y, w, 18.0, 3.0, 3.0)
    p.setPen(QPen(QColor("#2b2b2b")))
    p.drawText(QPointF(x + 5.0, y + 13.0), text)
    p.setBrush(Qt.NoBrush)
