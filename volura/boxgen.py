"""Boxes with finger joints: the computational core for laser cutting boxes.

No Qt in this file. It works on the shared 2D model from `volura/vector2d.py` and the
sheet settings of the worksheet, and can therefore be tested on its own and reused in the
worksheet, the export and the 3D model.

A **finger joint** is a glued corner of two panels of the same thickness: one panel
has protrusions (the male side), the other as many recesses (the female side) at the
same places. `finger_edge` gives that toothed edge as a polyline in its own plane: `x`
along the edge from 0 to `length`, `y` from 0 (the nominal edge) to `t` (the depth of
the tooth). On the male side the tooth points up (depth `t`), on the female side the
recess sits there, so the two fit together. A laser removes half the kerf along the
cut line; the fingers are therefore `kerf / 2` wider and the recesses `kerf / 2`
narrower, so the joint fits tightly.

A **panel** (`Panel`) is the cut contour of one face of the box, plus the matrix that
puts it in its place in the box: `position3d` maps the local plane `(u, v)` with `u` to
the right and `v` up into the world of the 3D model, and the place of the panel is
`position3d @ (u, v, 0..thickness)`. `panels_to_drawing` puts the contours on the
cutting layer of a worksheet (with the panel name engraved small in the middle),
`panels_to_manifolds` turns them into extruded volumes for the 3D model.

The corners: where three panels meet there must be no hole and no overlap. Therefore
one panel owns each corner. The bottom owns the lowest layer of the box (its edges are
female everywhere and the sides have fingers at their bottom), the front and back own
the vertical corner columns (male on their sides, the left and right side are female
there) and the lid owns the top layer (male on its four edges, the sides are female
there). The fingers of the left and right side at their bottom only start beyond the
corner columns, because there the bottom has the sole right.

At the bottom of the walls there is moreover `ends = SLOT`: the first and the last
cell of that joint is not a finger but material of the bottom. That way the bottom
keeps a square of `t` × `t` at every corner that is attached to the plate with two full
edges. Without that phase the recesses of two perpendicular edges touch in exactly one
point, the outline of the bottom crosses itself and the corner square hangs on nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from .i18n import tr
from .vector2d import DEFAULT_LAYERS, Drawing, VPath

MALE = "male"
FEMALE = "female"
TOOTH = "tooth"         # the first and the last cell of a joint is a finger
SLOT = "slot"           # ... or the opposite, so the corner keeps a square

#: finger width as a multiple of the material thickness (default of the dialog)
FINGER_FACTOR = 2.5
#: thickness to assume when the sheet settings know no material thickness (mm)
DEFAULT_THICKNESS = 3.0
MIN_FINGERS = 3          # a joint without fingers does not hold
LID_KINDS = ("none", "closed")
LID_LABELS = {"none": tr("No lid"), "closed": tr("Fixed lid")}

#: the names of the panels, as the 3D model shows them and as they are engraved
PANEL_LABELS = {"bottom": tr("base"), "front": tr("front"), "back": tr("back"),
                "left": tr("left"), "right": tr("right"), "lid": tr("lid")}

#: the default layers for the contours and the panel names (in the current language)
_LAYER_BY_OP = {op: name for name, _color, op in DEFAULT_LAYERS}


# --------------------------------------------------------------------------
#  the toothed edge
# --------------------------------------------------------------------------

def finger_count(length, finger_w) -> int:
    """The number of fingers: the largest odd number n with length / n >= finger_w.

    An odd number makes both ends of the edge have a finger (or a recess); then the
    joint of a panel connects to the corner. If the edge is too short for three
    fingers, there are three anyway: a joint with one finger holds nothing.
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
    """The polyline of a step function: `depth` high within `spans`, 0 outside."""
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
    """The 2D toothed edge of a joint, as a polyline ``(N, 2)``.

    The edge runs along the x axis from 0 to `length`; `y` is 0 or `t` (the depth of the
    tooth). The number of fingers comes from `finger_count` and is odd, so both ends of
    the edge are equal and the two sides of the same length fit exactly together:
    the area between the male edge and y = 0 and the area between the female edge and
    y = 0 complement each other.

    `ends` is the phase at the ends of the edge. With "tooth" the first and the last cell
    is a finger (male) or a recess (female); with "slot" it is the opposite, so the male
    side leaves material there for the corner. Two sides that must fit together use the
    same `ends`.

    `kerf` is the width of the cut gap: the fingers become `kerf / 2` wider on each side
    and the recesses `kerf / 2` narrower, so the joint fits tightly after cutting. With
    `kerf = 0` the two sides are each other's mirror image in tooth width.
    """
    L = abs(float(length))
    h = abs(float(t))
    k = max(0.0, float(kerf or 0.0))
    if L <= 0.0:
        return np.array([[0.0, 0.0]], dtype=float)
    n = finger_count(L, finger_w)
    step = L / n
    first = 0 if str(ends or "tooth").strip().lower() != "slot" else 1
    teeth = [(i * step, (i + 1) * step) for i in range(first, n, 2)]   # finger or recess
    if male:
        spans = [(a - 0.5 * k, b + 0.5 * k) for a, b in teeth]        # fingers wider
    else:
        spans, x = [], 0.0                                            # recesses narrower
        for a, b in teeth:
            a, b = a + 0.5 * k, b - 0.5 * k
            if a > x:
                spans.append((x, a))
            x = max(x, b)
        if x < L:
            spans.append((x, L))
    return _step_line(L, h, spans)


def tooth_area(edge, length) -> float:
    """The area between the toothed edge `edge` and the nominal edge y = 0 (mm²)."""
    L = abs(float(length))
    pts = _close_teeth(edge, L)
    return float(abs(_area(pts)))


def _close_teeth(edge, length) -> np.ndarray:
    """The toothed edge closed with the nominal edge: the area of the teeth."""
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
#  panels
# --------------------------------------------------------------------------

@dataclass(eq=False)
class Panel:
    """One face of the box: the cut contour and the place it gets in the model."""

    name: str
    outline: VPath
    position3d: np.ndarray
    thickness: float
    edges: dict = field(default_factory=dict)     # the joint per edge, for inspection

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.outline.pts, dtype=float)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.position3d, dtype=float)

    def placed(self, pts=None) -> np.ndarray:
        """The contour (or points) in the world of the 3D model, in its place in the box."""
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
    """Remove consecutive equal points, also across the closure, and remove needles.

    A needle is a vertex where the path runs back over itself: `a → b → a`. They
    arise at a corner of `left` and `right`, where the recess of the vertical joint
    runs exactly down to the bottom and cuts away the square of `t` × `t` there. Such a
    needle cuts a gap of zero width and is pointless work for a laser; the vertex and
    its duplicate are therefore dropped.
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
    """The points of a toothed edge, in the direction of travel from `p0` to `p1`.

    The teeth sit on the nominal edge between `p0` and `p1`; `line` is the polyline from
    `finger_edge`, whose x coordinate is the distance along the joint, measured from
    `q0` in the direction of `q1`. On the male side the tooth points out of the panel
    (against `inward`), on the female side it cuts `depth` deep inwards.
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
    # the wall of the first and the last tooth: on the nominal edge, at the same place
    out: list[np.ndarray] = [point(pts[0][0], 0.0)]
    for q, y in pts:
        off = -float(y) if gender == MALE else h - float(y)
        out.append(point(q, off))
    out.append(point(pts[-1][0], 0.0))
    return out


def _rect_outline(a, b, edges, depth, finger_w, kerf) -> tuple[np.ndarray, dict]:
    """The closed outline of a rectangle `a` × `b` with a joint per edge.

    `edges` is a list of four times `None` or `(gender, length, q0, q1[, ends])`,
    counter-clockwise around the rectangle: bottom, right, top, left. `q0` and `q1` are
    the ends of the joint in the coordinates of the panel; `q0` is where the toothed
    edge of `finger_edge` starts. `ends` is the phase at the ends of that edge.
    """
    corners = [np.array([0.0, 0.0]), np.array([a, 0.0]), np.array([a, b]), np.array([0.0, b])]
    pts: list[np.ndarray] = []
    kinds: dict = {}
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        spec = edges[i] if i < len(edges) else None
        d = p1 - p0
        inward = np.array([-float(d[1]), float(d[0])], dtype=float)   # left of the direction
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


RANDEN = ("onder", "rechts", "boven", "links")     # keys of Panel.edges (data, not shown)


def _matrix(x_col, y_col, z_col, off) -> np.ndarray:
    """4×4 matrix that puts (u, v, thickness) at the place of the panel (a true rotation)."""
    M = np.eye(4, dtype=float)
    M[:3, :3] = np.column_stack([np.asarray(x_col, float), np.asarray(y_col, float),
                                 np.asarray(z_col, float)])
    M[:3, 3] = np.asarray(off, dtype=float)
    return M


X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def box_panels(w, d, h, t, finger_w=None, inner=True, lid="none", kerf=0.0) -> list[Panel]:
    """The panels of a box with finger joints, in the order of the box.

    `w`, `d` and `h` are the inner dimensions (`inner=True`) or the outer dimensions of
    the box in millimetres, `t` is the material thickness and `finger_w` the desired
    finger width (default `2.5 × t`). `lid` is "none" (open box, the walls run up to the
    top edge) or "closed" (fixed lid, a sixth panel that drops into the top edge of the
    walls). `kerf` compensates the cut gap of the laser, only in the finger joints. The
    drawn panels then overlap each other by exactly that kerf, because the cut removes it.

    Every panel carries `position3d`: the matrix that puts its plane in its place in the
    box, so `panels_to_manifolds` can make an extruded volume of it directly.
    """
    w, d, h, t = float(w), float(d), float(h), float(t)
    lid = str(lid or "none").strip().lower()
    if lid not in LID_KINDS:
        raise ValueError(tr('Lid is "none" (no lid) or "closed" (fixed lid).'))
    if min(w, d, h) <= 0.0:
        raise ValueError(tr("Width, depth and height must be greater than zero."))
    if t <= 0.0:
        raise ValueError(tr("The material thickness must be greater than zero."))
    fw = FINGER_FACTOR * t if finger_w is None else float(finger_w)
    if fw <= 0.0:
        raise ValueError(tr("The finger width must be greater than zero."))
    k = max(0.0, float(kerf or 0.0))

    if inner:
        W, D = w + 2.0 * t, d + 2.0 * t
        H = h + 2.0 * t if lid == "closed" else h + t
    else:
        W, D, H = w, d, h
    iw, id_ = W - 2.0 * t, D - 2.0 * t          # inner size = wall length of front and back
    if min(iw, id_) <= 0.0:
        raise ValueError(tr("The box is smaller than twice the material thickness."))
    if lid == "closed" and H - 2.0 * t <= 0.0:
        raise ValueError(tr("The box is too low for bottom, walls and lid."))
    wall = H - t                                # height of a wall (the bottom is the layer below)

    def spec(gender, length, q0, q1, ends="tooth"):
        return (gender, length, np.asarray(q0, float), np.asarray(q1, float), ends)

    # -- the bottom: full outer size, female everywhere (it owns the lowest layer).
    #    The slots of the walls only start beyond the first cell, so the bottom keeps
    #    a square of `t` × `t` at every corner that is attached to the plate with a full
    #    edge; otherwise the corner square touches the rest of the bottom in only one point.
    bottom_edges = [
        spec(FEMALE, iw, (t, 0.0), (W - t, 0.0), SLOT),        # front
        spec(FEMALE, id_, (W, D - t), (W, t), SLOT),           # right
        spec(FEMALE, iw, (W - t, D), (t, D), SLOT),            # back
        spec(FEMALE, id_, (0.0, t), (0.0, D - t), SLOT),       # left
    ]
    # -- front and back: male at the bottom and on both vertical edges
    side_edges = [
        spec(MALE, iw, (0.0, 0.0), (iw, 0.0), SLOT),           # fingers into the bottom
        spec(MALE, wall, (iw, 0.0), (iw, wall)),
        spec(FEMALE, iw, (0.0, wall), (iw, wall)) if lid == "closed" else None,
        spec(MALE, wall, (0.0, 0.0), (0.0, wall)),
    ]
    # -- left: female on the vertical edges, at the bottom beyond the corner columns
    left_edges = [
        spec(MALE, id_, (t, 0.0), (D - t, 0.0), SLOT),
        spec(FEMALE, wall, (D, 0.0), (D, wall)),
        spec(FEMALE, id_, (t, wall), (D - t, wall)) if lid == "closed" else None,
        spec(FEMALE, wall, (0.0, 0.0), (0.0, wall)),
    ]
    # -- right: the same, but the joint of the lid runs in the opposite direction there
    right_edges = [left_edges[0], left_edges[1],
                   spec(FEMALE, id_, (D - t, wall), (t, wall)) if lid == "closed" else None,
                   left_edges[3]]
    # -- lid: inner size, male everywhere (it owns the top layer)
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
        panel(PANEL_LABELS["bottom"], W, D, bottom_edges, _matrix(X, Y, Z, (0.0, 0.0, 0.0))),
        panel(PANEL_LABELS["front"], iw, wall, side_edges, _matrix(X, Z, -Y, (t, t, t))),
        panel(PANEL_LABELS["back"], iw, wall, side_edges, _matrix(-X, Z, Y, (W - t, D - t, t))),
        panel(PANEL_LABELS["left"], D, wall, left_edges, _matrix(Y, Z, X, (0.0, 0.0, t))),
        panel(PANEL_LABELS["right"], D, wall, right_edges, _matrix(-Y, Z, -X, (W, D, t))),
    ]
    if lid == "closed":
        panels.append(panel(PANEL_LABELS["lid"], iw, id_, lid_edges,
                            _matrix(X, Y, Z, (t, t, H - t))))
    return panels


# --------------------------------------------------------------------------
#  to worksheet and 3D model
# --------------------------------------------------------------------------

def panels_to_drawing(panels, layer=None, label_layer=None,
                      labels=True, height=4.0, gap=2.0) -> Drawing:
    """The contours of the panels on a drawing, each panel as one source.

    The cut contours go on `layer` (cut) and the panel names — as small text contours
    in the middle of the panel — on `label_layer` (engrave). Without a name these are
    the default Cut and Engrave layers in the current language; a caller that merges into
    an existing sheet passes `sheet.default_layer("cut")` and `("engrave")`, so an old
    sheet with Dutch layer names gets no second cut layer. Paths that belong together get
    the same source number, so automatic arranging moves a panel with its engraving as
    one whole.

    The panels are each drawn in their own plane and therefore lie side by side in the
    drawing (from left to right, with `gap` in between), so they do not get in each
    other's way and nesting recognises them as separate parts.
    """
    layer = layer or _LAYER_BY_OP["cut"]
    if label_layer is None:
        label_layer = _LAYER_BY_OP["engrave"]
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
    """The contours of the panel name, centred in the middle of the panel."""
    pts = panel.points
    if len(pts) < 3:
        return []
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    if min(span) <= 0.0:
        return []
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
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
    """Every panel extruded with its thickness and transformed to its place.

    With `kerf` the panels overlap here by exactly the kerf of the laser: the drawn
    finger is `kerf / 2` wider on each side and the slot as much narrower, because the
    cut gap removes that later. The model therefore shows the panels as they come off
    the laser and not the box after assembly; the union of all panels is exactly the
    wall of the box, though.
    """
    out = []
    for p in panels:
        cs = mf.CrossSection([np.ascontiguousarray(p.points, dtype=float)], mf.FillRule.NonZero)
        man = cs.extrude(float(p.thickness))
        out.append(man.transform(np.ascontiguousarray(p.matrix()[:3], dtype=np.float64)))
    return out


def box_size(w, d, h, t, inner=True, lid="none"):
    """(width, depth, height) of the finished box, for the message in the status bar."""
    w, d, h, t = float(w), float(d), float(h), float(t)
    closed = str(lid or "none").strip().lower() == "closed"
    if inner:
        return (w + 2.0 * t, d + 2.0 * t, h + (2.0 if closed else 1.0) * t)
    return (w, d, h)
