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

A **living hinge** (`living_hinge`) is the other classic cut in plywood: a field of
straight slits next to each other, close enough together that the panel bends there
instead of breaking. It is the same kind of work as a finger joint — rows of cuts that
have to stay in step with each other — so it lives in this file too.

A **fit test** (`fit_test`) is the third classic cut: a comb of slots of `t + offset`
wide, cut from the sheet that is on the table, so the user learns which offset gives a
good press fit. Again a row of cuts that have to stay in step, so it lives here too.

A **vent pattern** (`vent_pattern`) is a field of holes in one closed contour: a grille
or a speaker cover. Here the grid is a plane instead of a row, but the work is the same —
cuts that stay in step with each other and that never touch the edge of the panel. The
honeycomb of hexagons and the field of circles stand on a triangular grid; the Voronoi
cells grow around a jittered grid of seed points, so the walls follow the bisectors
between the neighbours, and the slots lie in brick-offset rows. None of them needs a
library beyond numpy and manifold3d: the cells are cut with straight-line clipping
(Sutherland-Hodgman along the bisectors) and the Boolean intersection with the room.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import manifold3d as mf

from . import cam
from .i18n import tr
from .vector2d import DEFAULT_LAYERS, TOL, Drawing, VPath, arc_points, polygon_area

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
            # only where a finger of the other side comes in: a recess that runs to the end of
            # the edge stays there, or a sliver of kerf / 2 is left over at the corner
            a = a + 0.5 * k if a > 1e-9 else a
            b = b - 0.5 * k if b < L - 1e-9 else b
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


# --------------------------------------------------------------------------
#  living hinge
# --------------------------------------------------------------------------

def living_hinge(x0, y0, x1, y1, slit=15.0, gap=3.0, pitch=2.0, margin=3.0,
                 vertical=True) -> list[np.ndarray]:
    """The slits of a living hinge in the rectangle (x0, y0) to (x1, y1), each as (2, 2).

    A **living hinge** is a field of straight slits, cut full through plywood or MDF:
    the panel bends there, so a laser cutter makes a curved part out of flat stock. The
    slits stand in columns `pitch` apart; in a column the slits are `slit` long with a
    bridge of `gap` between them. Every second column is shifted by half the period
    `slit + gap`, so the bridges of neighbouring columns never lie in a row: that
    zigzag of material is what holds the panel together while it bends.

    `margin` stays free of slits along all four edges, so the pattern cannot cut the
    panel loose; the columns start `margin` from the left edge and stop `margin` before
    the right one. Even columns begin with a full slit at the bottom margin; the shifted
    columns get a piece of a slit at both margins, so they bend all the way along too — a
    piece shorter than `gap` is dropped, and so is a slit that would run from the bottom
    margin to the top one, because that would cut the panel in two.

    With `vertical=True` the slits run along Y and the sheet bends around a vertical
    axis; `vertical=False` is the same pattern turned 90°.
    """
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    slit, gap, pitch = float(slit), float(gap), float(pitch)
    margin = max(0.0, float(margin))
    if slit <= gap:
        raise ValueError(tr("The slits must be longer than the bridges between them."))
    if pitch <= 0.0:
        raise ValueError(tr("The distance between the columns must be greater than zero."))
    along, across = ((y0, y1), (x0, x1)) if vertical else ((x0, x1), (y0, y1))
    inner = (along[1] - along[0]) - 2.0 * margin
    cols = int(math.floor((across[1] - across[0] - 2.0 * margin) / pitch)) + 1
    if cols < 2 or inner <= 0.0:
        raise ValueError(tr("That rectangle is too small for a living hinge."))
    period = slit + gap
    lo, hi = along[0] + margin, along[1] - margin
    out = []
    for k in range(cols):
        c = across[0] + margin + k * pitch
        # a shifted column starts one period lower, so it also has a piece of a slit at the
        # bottom margin, just like at the top: otherwise its bottom stays stiff
        j = -(k % 2)
        while True:
            a = lo + 0.5 * period * (k % 2) + j * period      # every other column, half a step
            if a >= hi:
                break
            b = min(a + slit, hi)
            a = max(a, lo)
            if gap <= b - a < inner:
                out.append(np.array([[c, a], [c, b]] if vertical else [[a, c], [b, c]],
                                    dtype=float))
            j += 1
    return out


# --------------------------------------------------------------------------
#  vent pattern
# --------------------------------------------------------------------------

#: the corners of a hexagon whose centre-to-corner distance is 1, counter-clockwise: the
#: flats face up and down and to both sides, which is how a honeycomb stands
_HEX = np.array([[math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)]
                 for k in range(6)], dtype=float)

#: the shapes of the holes of a vent pattern
VENT_SHAPES = ("hex", "round", "voronoi", "slots")

#: how many grid points the Voronoi pattern may jitter at most: a full sheet of cells of
#: one millimetre would take minutes to clip one by one, and a grille that needs so many
#: cells is a raster job and not a cut job anyway
VENT_MAX_CELLS = 20_000

#: the seed points of the Voronoi cells jitter at most a quarter of `size` off their grid
#: spot: enough to look organic, little enough for the walls between the cells to stay
#: near the bisectors of neighbours a grid step apart
VENT_JITTER = 0.25

#: the grid spots a Voronoi neighbour can stand on for a cell: everything within two
#: steps of the seed's spot - with a jitter of at most a quarter of a step, a cell never
#: touches the territory of a seed further away than that
_VENT_NEIGHBOURS = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]


def _inside_points(loops, q) -> np.ndarray:
    """Which points of `q` (m, 2) lie in the area enclosed by `loops`?

    The loops are the edge of that area and the holes in it, so a point lies inside when a
    ray from it to the left crosses them an odd number of times. The points go in chunks,
    so a big panel does not build one huge table of every point against every edge.
    """
    q = np.asarray(q, dtype=float)
    inside = np.zeros(len(q), dtype=bool)
    for p in loops:
        a = np.asarray(p, dtype=float)
        b = np.roll(a, -1, axis=0)
        dy = np.where(b[:, 1] == a[:, 1], 1.0, b[:, 1] - a[:, 1])
        step = max(1, 4_000_000 // max(1, len(a)))
        for k in range(0, len(q), step):
            y = q[k:k + step, 1, None]
            band = (a[:, 1] > y) != (b[:, 1] > y)
            x = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
            inside[k:k + step] ^= np.count_nonzero(band & (x < q[k:k + step, 0, None]),
                                                   axis=1) % 2 == 1
    return inside


def _edge_distances(loops, q) -> np.ndarray:
    """The distance from every point of `q` (m, 2) to the nearest edge of `loops`."""
    q = np.asarray(q, dtype=float)
    out = np.full(len(q), np.inf)
    for p in loops:
        a = np.asarray(p, dtype=float)
        d = np.roll(a, -1, axis=0) - a
        L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
        step = max(1, 2_000_000 // max(1, len(a)))
        for k in range(0, len(q), step):
            w = q[k:k + step, None, :] - a[None, :, :]
            t = np.clip(np.einsum("mnj,nj->mn", w, d) / L2, 0.0, 1.0)
            r = w - t[:, :, None] * d[None, :, :]
            out[k:k + step] = np.minimum(out[k:k + step],
                                         np.sqrt(np.einsum("mnj,mnj->mn", r, r)).min(axis=1))
    return out


def _clip_halfplane(poly, p0, u):
    """Keep the side of the line through `p0` with normal `u` where `(p - p0) · u <= 0`.

    A convex polygon stays convex when a straight line cuts it, so one pass along the
    edges is enough: a point on the wrong side goes and is replaced by the crossing of
    the edge it lies between (Sutherland-Hodgman against one half-plane). The polygon is
    a plain list of (x, y) tuples: these cells have half a dozen corners and clipping
    them one line at a time is a lot of calls, so nothing in here may cost numpy
    overhead.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    ux, uy = float(u[0]), float(u[1])
    d = [(px - x0) * ux + (py - y0) * uy for px, py in poly]
    out = []
    n = len(poly)
    for h in range(n):
        dp, dq = d[h], d[(h + 1) % n]
        if dp <= 0.0:
            out.append(poly[h])
        if dp * dq < 0.0:
            t = dp / (dp - dq)
            px, py = poly[h]
            qx, qy = poly[(h + 1) % n]
            out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def _web_shrink(poly, seeds, web) -> list:
    """Cut `poly` (the cell of `seeds[0]`) by every neighbour, holding back half a web.

    The Voronoi cell of a seed is what a straight cut shares fairly between it and every
    neighbour: the set of points closer to the seed than to the neighbour. Moving that
    cut `web / 2` towards the seed is what keeps that much material on its side, so two
    neighbours that both do this leave `web` of material between their holes. A cell that
    every neighbour takes away comes back with fewer than three points.
    """
    x, y = seeds[0]
    for point in seeds[1:]:
        vx, vy = point[0] - x, point[1] - y
        length = math.hypot(vx, vy)
        if length < 1e-9:
            continue
        ux, uy = vx / length, vy / length
        poly = _clip_halfplane(poly, (x + (0.5 * length - 0.5 * web) * ux,
                                      y + (0.5 * length - 0.5 * web) * uy), (ux, uy))
        if len(poly) < 3:
            return poly
    return poly


def _pair_dists(holes, pairs, budget=4_000_000) -> list:
    """The least distance between the edges of each pair of holes in `pairs`.

    The closest points of two hole boundaries always include a corner of one hole seen
    from an edge of the other, so both ways round are measured and the smaller kept -
    the same law as `_edge_distances`, but for every pair in one sweep: the two point
    sets of a pair are laid side by side, their edges crossed, and the two self-
    comparisons set to a distance too big to mind. Chunks stay under `budget` crossings
    so a big pattern never allocates one unmanageable table.
    """
    out = []
    start = 0
    while start < len(pairs):
        big = len(holes[pairs[start][0]]) + len(holes[pairs[start][1]])
        stop = start + 1
        while stop < len(pairs):
            s = len(holes[pairs[stop][0]]) + len(holes[pairs[stop][1]])
            if (stop + 1 - start) * max(big, s) ** 2 > budget:
                break
            big, stop = max(big, s), stop + 1
        out.extend(_chunk_dists(holes, pairs[start:stop], big))
        start = stop
    return out


def _chunk_dists(holes, pairs, span) -> list:
    """The distances of one chunk of pairs, every point set padded to `span` points."""
    n = len(pairs)
    pts = np.zeros((n, span, 2))
    eds = np.zeros((n, span, 2))
    dvec = np.zeros((n, span, 2))
    L2 = np.full((n, span), 1e-30)
    ni = np.zeros(n, dtype=np.int64)
    nj = np.zeros(n, dtype=np.int64)
    for m, (i, j) in enumerate(pairs):
        vi, vj = np.asarray(holes[i], dtype=float), np.asarray(holes[j], dtype=float)
        ni[m], nj[m] = len(vi), len(vj)
        pts[m, :len(vi)] = vi
        pts[m, len(vi):len(vi) + len(vj)] = vj
        for q, base in ((vj, 0), (vi, len(vj))):     # j's edges first, then i's
            d = np.roll(q, -1, axis=0) - q
            eds[m, base:base + len(q)] = q
            dvec[m, base:base + len(q)] = d
            L2[m, base:base + len(q)] = np.maximum(np.sum(d * d, axis=1), 1e-30)
    w = pts[:, None, :, :] - eds[:, :, None, :]
    t = np.clip(np.einsum("mepd,med->mep", w, dvec) / L2[:, :, None], 0.0, 1.0)
    r = w - t[..., None] * dvec[:, :, None, :]
    dd = np.sqrt(np.einsum("mepd,mepd->mep", r, r))
    p, e = np.arange(span)[None, None, :], np.arange(span)[None, :, None]
    ok = (p < (ni + nj)[:, None, None]) & (e < (nj + ni)[:, None, None]) \
        & ((p < ni[:, None, None]) == (e < nj[:, None, None]))   # never hole against itself
    return [float(q) for q in np.where(ok, dd, np.inf).min(axis=(1, 2))]


def _chord_reach(qs, k) -> np.ndarray:
    """How far, per hole of the stack `qs`, a corner dropped by taking every k-th point
    reaches past the chord its two kept neighbours form.

    Every point the copy drops sits on an edge between two kept corners or between two
    dropped ones, and a straight edge between two points no further than that from a
    chord is nowhere further from it - so this one number bounds how far the original
    edge can be from the copy's, which is what the close-pair scan pays back as slack.
    """
    L = qs.shape[1]
    j = np.arange(L)
    jm = (j // k) * k
    last = ((L - 1) // k) * k                       # last kept corner; the run past it wraps
    a, b = qs[:, jm], qs[:, np.where(jm >= last, 0, jm + k)]
    w = qs - a
    dv = b - a
    t = np.clip(np.sum(w * dv, axis=2) / np.maximum(np.sum(dv * dv, axis=2), 1e-30), 0.0, 1.0)
    r = w - t[:, :, None] * dv
    d = np.sqrt(np.sum(r * r, axis=2))
    d[:, j % k == 0] = 0.0                          # kept corners sit on the chord anyway
    return d.max(axis=1)


def _web_proxies(holes, cap=0.25) -> tuple[list, list]:
    """A thinned copy of every hole for the close-pair scan, with how far it may be.

    A slot with round ends is mostly arc, and its arcs are points of a circle that a
    chord replaces to within a hair: taking every third point of a smooth hole costs
    little distance and saves much measuring. Where the corners are real - a Voronoi
    corner a chord would cut away by more than `cap` - the hole keeps all its points
    and owes nothing, so the scan never reports a distance the real holes cannot have.
    """
    holes = [np.asarray(q, dtype=float) for q in holes]
    prox, reach = list(holes), [0.0] * len(holes)
    groups: dict = {}
    for m, q in enumerate(holes):
        k = max(1, round(len(q) / 21.0))
        if k > 1 and len(q[::k]) >= 3:
            groups.setdefault((k, len(q)), []).append(m)
    for (k, _), ids in groups.items():
        heads = np.stack([holes[m] for m in ids])
        for m, e in zip(ids, _chord_reach(heads, k)):
            if float(e) <= cap:
                prox[m], reach[m] = holes[m][::k], float(e)
    return prox, reach


def _keep_whole_webs(holes, web, rounds=200) -> list[np.ndarray]:
    """The holes with the web kept everywhere: of a pair that stands too close, the
    smaller goes.

    On a contour with deep notches the room can cut so sharply that a piece of one cell
    ends up nearer to another hole than the web, without their bisector having said so.
    The laser may not meet less material than the user asked for, so those holes are
    dropped - the smaller of the pair first, because it has the least room to be right -
    and this repeats until every pair keeps the web. A normal pattern changes nothing
    here, and stops after the first scan.

    Holes that can come within the web of each other share a bin of a grid, so finding
    the pairs that truly need measuring is a handful of bins and not the square of the
    number of holes; the measuring itself happens on the thinned copies first, and only
    the few pairs standing closer than their copies' slack get the exact second look.
    """
    d = max(0.0, float(web) - 0.05)
    holes = [np.asarray(q, dtype=float) for q in holes]
    prox, reach = _web_proxies(holes)
    e = d + (2 * max(reach) + 0.01 if reach else 0.0)   # slack any pair of copies can carry
    for _ in range(rounds):
        if len(holes) < 2:
            return holes
        box = np.array([[q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()]
                       for q in prox])
        area = np.array([abs(polygon_area(q)) for q in holes])
        step = max(d + 0.01, float(np.median(box[:, 1] - box[:, 0] + box[:, 3] - box[:, 2])))
        bins: dict = {}
        for m in range(len(prox)):
            for bi in range(int((box[m, 0] - e) // step), int((box[m, 1] + e) // step) + 1):
                for bj in range(int((box[m, 2] - e) // step), int((box[m, 3] + e) // step) + 1):
                    bins.setdefault((bi, bj), []).append(m)
        seen = set()
        pairs = []
        for group in bins.values():
            for a_ in range(len(group) - 1):
                for b_ in range(a_ + 1, len(group)):
                    i, j = group[a_], group[b_]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    gap = d + reach[i] + reach[j]
                    if (box[i, 1] < box[j, 0] - gap or box[j, 1] < box[i, 0] - gap
                            or box[i, 3] < box[j, 2] - gap or box[j, 3] < box[i, 2] - gap):
                        continue
                    pairs.append((i, j))
        worst = np.zeros(len(prox))
        hit = np.zeros(len(prox), dtype=bool)
        near = [(i, j) for (i, j), c in zip(pairs, _pair_dists(prox, pairs))
                if c < d + reach[i] + reach[j]]
        for (i, j), m in zip(near, _pair_dists(holes, near)):
            if m >= d:
                continue
            worst[i] = max(worst[i], area[j])         # the larger of the pair stays
            worst[j] = max(worst[j], area[i])
            hit[i] = hit[j] = True
        if not hit.any():
            return holes
        drop = hit & (area <= worst + 1e-12)
        if not drop.any():
            return holes                                  # cannot resolve, leave them
        holes = [q for q, gone in zip(holes, drop) if not gone]
        prox = [q for q, gone in zip(prox, drop) if not gone]
        reach = [q for q, gone in zip(reach, drop) if not gone]
    return holes


def _section_of(loops) -> mf.CrossSection:
    """The filled area of closed loops as one cross-section, wound as manifold likes it."""
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                            for q in loops], mf.FillRule.NonZero)


def _jittered_lattice(loop, room, size, margin, seed):
    """The seed points of the Voronoi cells and the grid indices they stand on.

    A lattice of `size` millimetres covers the contour (with the margin and a cell of
    room on every side), and every point shifts a random but repeatable step of at most
    `VENT_JITTER` away: `seed` is the number that repeats the pattern. Points that end up
    in the edge margin are no seeds at all; `ii, jj` number the ones that stay so the
    neighbours of a cell can be looked up by grid spot.
    """
    n = int((loop[:, 0].max() - loop[:, 0].min()) / size + 1.0) \
        + 2 * int(math.ceil(margin / size + 1.0)) + 2
    m = int((loop[:, 1].max() - loop[:, 1].min()) / size + 1.0) \
        + 2 * int(math.ceil(margin / size + 1.0)) + 2
    if n * m > VENT_MAX_CELLS:
        raise ValueError(tr("There are too many cells for one pattern: make the cells bigger."))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    ii0, jj0 = np.meshgrid(np.arange(n), np.arange(m))
    ii, jj = ii0.ravel(), jj0.ravel()
    jitter = rng.uniform(-VENT_JITTER, VENT_JITTER, (len(ii), 2))
    x = loop[:, 0].min() - margin - size + (ii + 0.5 + jitter[:, 0]) * size
    y = loop[:, 1].min() - margin - size + (jj + 0.5 + jitter[:, 1]) * size
    spot = np.column_stack([x, y])
    keep = _inside_points(room, spot)
    return spot[keep], ii[keep], jj[keep]


def _voronoi_holes(loop, room, size, web, margin, seed) -> list[np.ndarray]:
    """Organic cells around the seeds: every cell shrunk by half a web, whole in the room.

    The cell of a seed is clipped against its neighbours on the grid, a step `size`
    away; two cells that both hold back half a web from their shared bisector leave the
    whole web between them. A cell at the edge has no neighbours there, so it runs on
    into the margin, and it is the edge of the margin itself that closes it: the material
    between such a hole and the outline is exactly the margin. What survives the room is
    one hole; thin slivers of a cell (less than a tenth of a cell) do not print and do
    not laser well, so they go.
    """
    seeds, ii, jj = _jittered_lattice(loop, room, size, margin, seed)
    if len(seeds) < 2:
        return []
    grid = -np.ones((int(jj.max()) + 5, int(ii.max()) + 5), dtype=np.int64)
    grid[jj + 2, ii + 2] = np.arange(len(seeds))
    near_j = np.array([a for a, _b in _VENT_NEIGHBOURS])
    near_i = np.array([b for _a, b in _VENT_NEIGHBOURS])
    allp = np.concatenate(room)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    box = [(lo[0] - web, lo[1] - web), (hi[0] + web, lo[1] - web),
           (hi[0] + web, hi[1] + web), (lo[0] - web, hi[1] + web)]
    spots = seeds.tolist()
    cells = []
    for k in range(len(seeds)):
        near = grid[jj[k] + 2 + near_j, ii[k] + 2 + near_i]
        cell = _web_shrink(box, [spots[k]] + [spots[h] for h in near if h >= 0], web)
        if len(cell) >= 3:
            cells.append(np.ascontiguousarray(np.array(cell, dtype=float)))
    if not cells:
        return []
    # The cells no longer share a wall with each other, so the room they all share is
    # taken from all of them in one go: a union of the cells and one intersection.
    got = mf.CrossSection.batch_boolean([_section_of([c]) for c in cells], mf.OpType.Add)
    got = mf.CrossSection.batch_boolean([got, _section_of(room)], mf.OpType.Intersect)
    floor = 0.10 * size * size
    holes = []
    for q in got.to_polygons():
        q = np.asarray(q, dtype=float)
        if len(q) < 3 or abs(polygon_area(q)) < floor:
            continue
        # a piece that lies only against the margin is no cell but the whole room: a
        # seed without a single living neighbour would open a frame in the panel
        # instead of grilling it
        mid = 0.5 * (q + np.roll(q, -1, axis=0))
        if float(_edge_distances(room, mid).max()) <= 1e-4:
            continue
        holes.append(np.ascontiguousarray(q))
    return _keep_whole_webs(holes, web)


def _slot_loop(size, length) -> np.ndarray:
    """One rounded slot of `length` by `size`, centred on the origin along +x: a stadium.

    The ends are the arcs of a circle of `size` across, flattening the same way a drawn
    circle does; `length` is the whole length, so a slot as long as it is wide is a
    circle.
    """
    r = 0.5 * size
    a = 0.5 * max(0.0, float(length) - size)
    right = arc_points(np.array([a, 0.0]), r, -0.5 * math.pi, 0.5 * math.pi, TOL / 4.0)
    left = arc_points(np.array([-a, 0.0]), r, 0.5 * math.pi, 1.5 * math.pi, TOL / 4.0)
    return np.ascontiguousarray(np.concatenate([right[:-1], left[:-1]]))


def _slots_holes(loop, room, size, web, slot_length, angle) -> list[np.ndarray]:
    """Rows of rounded slots, every other row half a pitch on, like a brick wall.

    The slots stand `web` apart in both directions: a row is `size + web` higher than the
    one below it and two slots in a row are `length + web` centres apart. What is turned
    on the worksheet is the whole grid, so `angle` only turns the grid and the slots
    around it, and a slot still counts only when it lies whole inside the margin.
    """
    length = 4.0 * size if slot_length is None or float(slot_length) <= 0.0 \
        else float(slot_length)
    if length + 1e-12 < size:
        raise ValueError(tr("A slot must be at least as long as it is wide."))
    unit = _slot_loop(size, max(length, size))
    pitch_x, pitch_y = max(length, size) + web, size + web
    allp = np.concatenate(room)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    c = 0.5 * (loop.min(axis=0) + loop.max(axis=0))
    rot = math.radians(float(angle) % 180.0)
    co, si = math.cos(rot), math.sin(rot)
    R = np.array([[co, -si], [si, co]])
    # the grid works in its own straight frame and is turned with the slots around the
    # centre of the contour: the room box, seen from that frame, is how much grid to make
    corner = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    back = c + (corner - c) @ R
    lo2, hi2 = back.min(axis=0), back.max(axis=0)
    rows = []
    for j in range(int(math.floor((lo2[1] - c[1]) / pitch_y)) - 1,
                   int(math.ceil((hi2[1] - c[1]) / pitch_y)) + 2):
        y = c[1] + j * pitch_y
        x0 = c[0] + (0.5 if j % 2 else 0.0) * pitch_x    # every other row, half a slot on
        i = np.arange(int(math.ceil((lo2[0] - x0) / pitch_x)) - 1,
                      int(math.floor((hi2[0] - x0) / pitch_x)) + 2)
        rows.append(np.column_stack([x0 + i * pitch_x, np.full(len(i), y)]))
    centres = c + (np.concatenate(rows) - c) @ R.T       # turned with the grid
    # as with the honeycomb, the centre decides most slots at once: one whose centre
    # lies deeper in the room than the tip of its end arc is in; one whose centre lies
    # outside, or nearer to the edge than the flat of the row, is out
    reach, flat = 0.5 * max(length, size), 0.5 * size
    gap = 1e-9
    hole_rot = unit @ R.T
    deep = _edge_distances(room, centres)
    deep[~_inside_points(room, centres)] = -1.0          # a centre outside the room is no centre
    keep = deep >= reach - gap
    for h in np.flatnonzero(~keep & (deep >= flat - gap)):
        hole = centres[h] + hole_rot
        ins = _inside_points(room, hole)
        # an edge exactly on the edge of the margin counts as inside: the edge comes
        # from an offset, and a slot may stand with its flat right on it
        keep[h] = bool(np.all(ins | (_edge_distances(room, hole) <= gap)))
    holes = [np.ascontiguousarray(centres[h] + hole_rot) for h in np.flatnonzero(keep)]
    return _keep_whole_webs(holes, web)


def vent_pattern(outline, shape="hex", size=6.0, web=2.0, margin=3.0, seed=1,
                 slot_length=None, angle=0.0) -> list[np.ndarray]:
    """The holes of a grille or a speaker cover, filling the closed contour `outline`.

    `outline` is a closed contour (n, 2) of any shape: the laser cuts these holes out of
    the panel that the contour encloses. `shape` is "hex" — regular hexagons of `size`
    millimetres across the flats, standing in a honeycomb — or "round": circles of `size`
    millimetres diameter. Both stand on the same triangular grid, a step `size + web` from
    the next hole, so the web of material between two neighbours is `web` wide everywhere.
    Hexagons face each other with a flat on that grid, which is why a honeycomb is the
    tightest way to fill a panel with holes.

    The other two shapes break that grid on purpose. "voronoi" fills the panel with
    organic cells: seed points on a jittered grid of step `size`, shaken at most a
    quarter of `size` off their spot by the random number series of `seed` — the same
    seed always gives the same pattern — and every cell cut by the bisectors with its
    neighbours and shrunk by half a web, so the walls between the cells are `web` wide
    while the wall along the margin is the margin itself. Cells of which less than a
    tenth survives the margin are no holes but slivers and go. "slots" lays rows of
    rounded slots of `size` wide and `slot_length` long (by default four times as long
    as wide) in a brick bond: rows a pitch `size + web` apart, every other row half a
    slot pitch on, the whole grid turned by `angle` degrees. Only whole slots count, as
    with the other shapes.

    The grid is centred on the bounding box of the contour, so the pattern of a rectangle
    or a circle is symmetric on every side. A hole only stays when it lies entirely inside
    the contour shrunk by `margin`: that material along the edge keeps the grille in the
    panel and leaves no half holes against the edge. A contour that is too small or too
    narrow holds no hole at all, and then the answer is an empty list.
    """
    loop = np.asarray(outline, dtype=float)
    if len(loop) > 1 and np.array_equal(loop[0], loop[-1]):
        loop = loop[:-1]                  # an imported contour often ends on its start
    if len(loop) < 3:
        return []
    shape = str(shape or "hex").strip().lower()
    if shape not in VENT_SHAPES:
        raise ValueError(tr("That shape of hole is unknown."))
    size, web = float(size), float(web)
    margin = max(0.0, float(margin))
    if size <= 0.0:
        raise ValueError(tr("The holes must be wider than zero."))
    if web <= 0.0:
        raise ValueError(tr("The material between the holes must be wider than zero."))
    room = cam.offset_loop(loop, -margin)
    if not room:
        return []
    if shape == "voronoi":
        return _voronoi_holes(loop, room, size, web, margin, seed)
    if shape == "slots":
        return _slots_holes(loop, room, size, web, slot_length, angle)
    if shape == "hex":
        unit = (size / math.sqrt(3.0)) * _HEX      # half a flat is size / 2, so this is the corner
    else:
        # the same flattening as a drawn circle (sheettools.circle_loop, which needs Qt)
        unit = arc_points(np.zeros(2), 0.5 * size, 0.0, 2.0 * math.pi, TOL / 4.0)[:-1]
    allp = np.concatenate(room)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    c = 0.5 * (loop.min(axis=0) + loop.max(axis=0))
    d = size + web
    col = 0.5 * math.sqrt(3.0) * d                 # the columns of the honeycomb
    rx, ry = float(unit[:, 0].max()), float(unit[:, 1].max())
    centres = []
    for j in range(int(math.ceil((lo[0] - c[0] - rx) / col)),
                   int(math.floor((hi[0] - c[0] + rx) / col)) + 1):
        y = c[1] + 0.5 * j * d                     # every column stands half a step higher
        i = np.arange(int(math.ceil((lo[1] - y - ry) / d)),
                      int(math.floor((hi[1] - y + ry) / d)) + 1)
        centres.append(np.column_stack([np.full(len(i), c[0] + j * col), y + i * d]))
    if not centres:
        return []
    centres = np.concatenate(centres)
    # The centre decides most holes at once: one whose centre lies deeper in the room than
    # its corners reach is in; one whose centre lies outside, or nearer to the edge than
    # its flats, is out. Only the holes in between, along the edge, are tested corner by
    # corner.
    gap = 1e-9
    reach = float(np.max(np.hypot(unit[:, 0], unit[:, 1])))
    flat = 0.5 * size * (1.0 if shape == "hex" else math.cos(math.pi / len(unit)))
    deep = _edge_distances(room, centres)
    deep[~_inside_points(room, centres)] = -1.0
    keep = deep >= reach - gap
    for h in np.flatnonzero(~keep & (deep >= flat - gap)):
        hole = centres[h] + unit
        ins = _inside_points(room, hole)
        # a corner exactly on the edge of the margin counts as inside: the edge comes from
        # an offset, and a hole may stand with its flat right on it
        keep[h] = bool(np.all(ins | (_edge_distances(room, hole) <= gap)))
    return [np.ascontiguousarray(centres[h] + unit) for h in np.flatnonzero(keep)]


# --------------------------------------------------------------------------
#  fit test
# --------------------------------------------------------------------------

#: mm of material at both ends of a fit test
FIT_TEST_END = 6.0
#: mm of material between two slots of a fit test
FIT_TEST_TOOTH = 10.0
#: mm of material under the slots of a fit test, beside the height of the letters
FIT_TEST_ROOM = 6.0


def _offset_text(offset) -> str:
    """How the offset of a fit test stands under its slot: "-0.1", "0", "+0.1"."""
    o = float(offset)
    return "0" if abs(o) < 1e-9 else f"{o:+g}"


def _fit_label(text, height, max_width) -> list[np.ndarray]:
    """The outlines of `text`, centred on the origin and at most `max_width` mm wide.

    Empty when this core runs without a Qt app, without fonts, or when the text yields
    nothing: a comb without labels is not an error, its slots are in order anyway.
    """
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return []                         # letters need a Qt app; this core is not one
    from . import fonts                   # Qt: only here, this core is Qt-free
    if not fonts.has_fonts():
        return []
    try:
        loops = fonts.text_loops(text, height=max(0.5, float(height)))
    except Exception:
        return []
    if not loops:
        return []
    allp = np.concatenate([np.asarray(L, dtype=float) for L in loops])
    wide = float(allp[:, 0].max() - allp[:, 0].min())
    s = min(1.0, float(max_width) / wide) if wide > 1e-9 else 1.0
    c = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    return [np.ascontiguousarray((np.asarray(L, dtype=float) - c) * s) for L in loops]


def fit_test(t, offsets=(-0.2, -0.1, 0.0, 0.1, 0.2), slot_depth=None, label_height=4.0,
             layer=None, label_layer=None) -> Drawing:
    """A comb of slots that shows which offset presses best on the sheet in use.

    Returns a `Drawing` with one part: a strip whose top edge carries one slot per offset,
    `t + offset` wide and `slot_depth` deep (by default `2 × t`, never shallower than 5 mm,
    so an offcut has something to grip). The slots stand in cells of `10 + t` mm, so about
    10 mm of material always stays between two of them, and 6 mm stays free at both ends
    and under the slots — under the slots there is `label_height` more, the room the
    letters need. Under every slot its offset stands engraved, so the user knows later
    what the number meant.

    The user cuts the comb from the sheet on the table, pushes an offcut of the same sheet
    into every slot and keeps the offset of the slot that holds it best: that is the kerf
    to fill or the finger width to use. The offsets run from small on the left to large on
    the right, so the slots grow from left to right; without fonts the comb simply has no
    labels and the order still tells the slots apart.

    `layer` and `label_layer` are the cut and the engrave layer to draw on, by default the
    default layers in the current language, as `panels_to_drawing` uses them. Outline and
    labels share one source number, so arranging the sheet moves the comb with its
    engravings as one whole.
    """
    t = float(t)
    if t <= 0.0:
        raise ValueError(tr("The material thickness must be greater than zero."))
    offs = [float(o) for o in offsets]
    if len(offs) < 2:
        raise ValueError(tr("A fit test needs at least two slots."))
    if any(t + o <= 0.0 for o in offs):
        raise ValueError(tr("A slot may not be zero or less wide."))
    depth = max(2.0 * t, 5.0) if slot_depth is None else float(slot_depth)
    if depth <= 0.0:
        raise ValueError(tr("The slot depth must be greater than zero."))
    lh = max(0.0, float(label_height))
    pitch = FIT_TEST_TOOTH + t
    width = 2.0 * FIT_TEST_END + len(offs) * pitch
    top = depth + FIT_TEST_ROOM + lh          # the top edge of the strip
    floor = top - depth                       # the bottom of the slots
    xs = [FIT_TEST_END + (i + 0.5) * pitch for i in range(len(offs))]

    pts = [(0.0, 0.0), (width, 0.0), (width, top)]
    for cx, o in reversed(list(zip(xs, offs))):
        half = 0.5 * (t + o)
        pts += [(cx + half, top), (cx + half, floor), (cx - half, floor), (cx - half, top)]
    pts.append((0.0, top))

    layer = layer or _LAYER_BY_OP["cut"]
    if label_layer is None:
        label_layer = _LAYER_BY_OP["engrave"]
    dr = Drawing()
    dr.ensure_layer(layer, "#ff0000", "cut")
    src = 1
    dr.add_path(_clean(pts), closed=True, layer=layer, filled=True, src=src)
    labels = []
    for cx, o in zip(xs, offs):
        place = np.array([cx, 0.5 * (FIT_TEST_ROOM + lh)], dtype=float)
        labels += [loop + place for loop in _fit_label(_offset_text(o), lh, pitch - 2.0)]
    if labels and label_layer:
        dr.ensure_layer(label_layer, "#0000ff", "engrave")
        for loop in labels:
            dr.add_path(loop, closed=True, layer=label_layer, filled=True, src=src)
    return dr
