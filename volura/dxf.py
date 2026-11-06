"""Read and write DXF to and from the shared 2D model (`volura.vector2d`).

DXF is the exchange format of milling and laser software (Fusion, Estlcam,
VCarve). This reads and writes ASCII DXF using only the standard library and
numpy (no ezdxf).

When reading, units from `$INSUNITS` are converted to millimetres, entities in
`ENTITIES` and in blocks via `INSERT` are followed (insertion point, scale,
rotation, nested blocks), and arcs from bulges, circles, ellipses and NURBS
splines (evaluated with de Boor) are flattened adaptively. The extrusion
direction (210/220/230) defines the object coordinate system; an extrusion of
(0, 0, -1) -- common in CAD files -- mirrors X. Layers come from the LAYER table
and the per-entity layer (8), colours from ACI (62) or true colour (420), with
BYLAYER and BYBLOCK following the layer colour. Loose line and arc pieces are
then chained into closed contours.

Writing produces DXF R12 (AC1009) for maximum compatibility:
`POLYLINE`/`VERTEX`/`SEQEND` (with the closed flag), `LINE` for two-point paths
and `CIRCLE` for closed paths that lie on a circle (handy for drill holes).
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import fileio
from .i18n import tr
from .vector2d import Drawing, Layer, TOL, arc_points, ellipse_arc_points, join_open_paths

# Units from $INSUNITS, converted to millimetres (0 or missing = mm).
INSUNITS_MM = {1: 25.4, 2: 304.8, 3: 3048.006096, 4: 1.0, 5: 10.0, 6: 1000.0,
               7: 100000.0, 8: 10000000.0, 9: 10000.0, 10: 1.0, 11: 0.0254,
               12: 0.254, 13: 0.0254, 14: 25400.0, 15: 254000.0, 16: 25.4,
               17: 2540.0, 18: 0.1}

# Standard ACI colours (index -> rgb); other indexes give grey.
ACI_RGB = {1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 255, 0), 4: (0, 255, 255),
           5: (0, 0, 255), 6: (255, 0, 255), 7: (0, 0, 0), 8: (128, 128, 128),
           9: (192, 192, 192)}

BYLAYER = 256
BYBLOCK = 0

# Entities we skip on purpose, with the matching message.
_TEXT_SKIPPED = tr("Text in the DXF was skipped; convert text to paths to mill it.")
SKIP_MESSAGES = {
    "TEXT": _TEXT_SKIPPED,
    "MTEXT": _TEXT_SKIPPED,
    "ATTDEF": _TEXT_SKIPPED,
    "ATTRIB": _TEXT_SKIPPED,
    "HATCH": tr("Hatch (HATCH) in the DXF was skipped."),
    "SOLID": tr("Face (SOLID) in the DXF was skipped."),
    "TRACE": tr("Face (TRACE) in the DXF was skipped."),
    "DIMENSION": tr("Dimension (DIMENSION) in the DXF was skipped."),
    "LEADER": tr("Leader (LEADER) in the DXF was skipped."),
    "TOLERANCE": tr("Tolerance (TOLERANCE) in the DXF was skipped."),
}

SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE"}


# --------------------------------------------------------------------------
#  source and group code pairs
# --------------------------------------------------------------------------

BINARY_MAGIC = b"AutoCAD Binary DXF"


def _data(source) -> bytes:
    """The source (path, bytes or text) as bytes; binary DXF is refused."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
    elif hasattr(source, "__fspath__"):
        data = _read_file(os.fspath(source))
    elif isinstance(source, str):
        if "\n" in source:
            data = source.encode("utf-8")
        else:
            data = _read_file(source)
    else:
        data = _read_file(str(source))
    if data[:len(BINARY_MAGIC)] == BINARY_MAGIC:
        raise ValueError(tr("Binary DXF is not supported; save the file as ASCII DXF."))
    return data


def _read_file(path) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as exc:
        raise ValueError(tr("Cannot open the DXF file: {error}").format(error=exc)) from exc


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace")


def _pairs(text: str):
    """The group code/value pairs as a list of (code, value).

    Every pair is two lines, and a value may be empty: an empty text, such as a custom
    property without a value in the header, is still a line of its own. Spaces around
    code and value and \\r\\n are ignored, and a value may itself contain spaces. Where a
    code is expected, a blank line or a line that is not a number is skipped, so the
    reading gets back in step instead of pairing every later code with the wrong value.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    i, n = 0, len(lines)
    while i < n - 1:
        try:
            code = int(lines[i])
        except ValueError:
            i += 1
            continue
        out.append((code, lines[i + 1]))
        i += 2
    return out


# --------------------------------------------------------------------------
#  colours
# --------------------------------------------------------------------------

def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c))) for c in rgb)


def aci_to_hex(aci) -> str:
    """ACI colour as #rrggbb; unknown indexes become grey."""
    try:
        aci = int(aci)
    except (TypeError, ValueError):
        return "#808080"
    rgb = ACI_RGB.get(aci)
    return _hex(rgb) if rgb else "#808080"


def hex_to_aci(color) -> int:
    """The ACI colour closest to `color` (#rrggbb)."""
    s = str(color or "#000000").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return 7
    try:
        r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 7
    best, dist = 7, None
    for aci, (cr, cg, cb) in ACI_RGB.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist is None or d < dist:
            best, dist = aci, d
    return best


def _true_color(value):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return _hex(((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF))


# --------------------------------------------------------------------------
#  object coordinate system
# --------------------------------------------------------------------------

def _ocs_matrix(extrusion) -> np.ndarray:
    """2x2 matrix that maps OCS points (x, y) to WCS for this extrusion direction.

    An extrusion of (0, 0, 1) is the identity; (0, 0, -1) mirrors X (the object
    coordinate system is then "upside down"). Other directions follow the DXF
    arbitrary axis algorithm.
    """
    N = np.asarray(extrusion, dtype=float)
    if N.shape != (3,):
        N = np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(N))
    if n < 1e-9:
        return np.eye(2)
    N = N / n
    if N[2] < 0.0 and abs(N[0]) < 1e-6 and abs(N[1]) < 1e-6:
        return np.array([[-1.0, 0.0], [0.0, 1.0]])
    if abs(N[0]) < 1.0 / 6.0 and abs(N[1]) < 1.0 / 6.0:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 0.0, 1.0])
    Ay = np.cross(N, temp)
    na = float(np.linalg.norm(Ay))
    if na < 1e-12:
        return np.eye(2)
    Ay = Ay / na
    Ax = np.cross(Ay, N)
    return np.column_stack([Ax[:2], Ay[:2]])


# --------------------------------------------------------------------------
#  arcs from bulges
# --------------------------------------------------------------------------

def bulge_arc(p0, p1, bulge, tol=TOL) -> np.ndarray:
    """Arc between `p0` and `p1` from the bulge; the points after `p0` up to and including `p1`."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if abs(bulge) < 1e-10:
        return np.array([p1])
    theta = 4.0 * math.atan(float(bulge))
    d = p1 - p0
    c = float(math.hypot(d[0], d[1]))
    if c < 1e-12 or abs(math.sin(theta / 2.0)) < 1e-12:
        return np.array([p1])
    r = c / (2.0 * math.sin(theta / 2.0))
    u = d / c
    nrm = np.array([-u[1], u[0]])
    mid = 0.5 * (p0 + p1)
    cot = math.cos(theta / 2.0) / math.sin(theta / 2.0)
    center = mid - nrm * (0.5 * c * cot)
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    arc = arc_points(center, abs(r), a0, a0 + theta, tol)
    return arc[1:]


def _vertices_and_bulges(pairs):
    """Points and bulges of an (LW)POLYLINE, in order, with bulge 0 when missing."""
    verts, bulges = [], []
    x = y = None
    bulge = None

    def commit():
        nonlocal x, y, bulge
        if x is not None and y is not None:
            verts.append((x, y))
            bulges.append(bulge if bulge is not None else 0.0)
        x = y = None
        bulge = None

    for c, v in pairs:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if c == 10:
            if x is not None and y is not None:
                commit()
            x = f
        elif c == 20:
            y = f
        elif c == 42:
            bulge = f
    commit()
    return verts, bulges


def _polyline_from_vertices(verts, bulges, closed, tol=TOL):
    """Points and bulges of an (LW)POLYLINE as a polyline with arcs."""
    n = len(verts)
    if n == 0:
        return np.zeros((0, 2)), False
    pts = [np.asarray(verts[0], dtype=float)]
    for i in range(n - 1):
        b = bulges[i] if i < len(bulges) else 0.0
        if abs(b) > 1e-10:
            pts.extend(bulge_arc(verts[i], verts[i + 1], b, tol))
        elif not np.allclose(verts[i + 1], pts[-1], rtol=0.0, atol=1e-12):
            pts.append(np.asarray(verts[i + 1], dtype=float))
    if closed and n >= 2:
        b = bulges[n - 1] if len(bulges) >= n else 0.0
        if abs(b) > 1e-10:
            pts.extend(bulge_arc(verts[n - 1], verts[0], b, tol))
    return np.asarray(pts, dtype=float).reshape(-1, 2), bool(closed)


# --------------------------------------------------------------------------
#  NURBS (de Boor)
# --------------------------------------------------------------------------

def _find_span(knots, degree, u, n):
    if u >= knots[n + 1]:
        return n
    if u <= knots[degree]:
        return degree
    lo, hi = degree, n + 1
    mid = (lo + hi) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _de_boor(degree, knots, pw, span, u):
    d = [pw[span - degree + k].copy() for k in range(degree + 1)]
    for r in range(1, degree + 1):
        for k in range(degree, r - 1, -1):
            ki = span - degree + k
            den = knots[ki + degree - r + 1] - knots[ki]
            a = 0.0 if abs(den) < 1e-15 else (u - knots[ki]) / den
            d[k] = (1.0 - a) * d[k - 1] + a * d[k]
    return d[degree]


def nurbs_points(degree, knots, ctrl, weights=None, samples_per_span=24):
    """NURBS curve as a polyline, evaluated with de Boor (rational with weights)."""
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1, 2)
    n = len(ctrl) - 1
    if n < 1 or degree < 1 or len(knots) < n + degree + 2:
        return ctrl.copy()
    if weights is None or len(weights) != n + 1:
        weights = [1.0] * (n + 1)
    pw = [np.array([ctrl[i, 0] * weights[i], ctrl[i, 1] * weights[i], weights[i]])
          for i in range(n + 1)]
    u0, u1 = knots[degree], knots[n + 1]
    if not u1 > u0:
        return ctrl.copy()
    interior = sorted({k for k in knots if u0 < k < u1})
    bounds = [u0] + interior + [u1]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = max(2, int(samples_per_span))
        for j in range(m + 1):
            u = min(a + (b - a) * j / m, u1)
            span = _find_span(knots, degree, u, n)
            h = _de_boor(degree, knots, pw, span, u)
            w = h[2]
            pt = h[:2] / w if abs(w) > 1e-15 else h[:2]
            if not out or float(np.linalg.norm(pt - out[-1])) > 1e-12:
                out.append(np.asarray(pt, dtype=float))
    return np.asarray(out, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------
#  entities
# --------------------------------------------------------------------------

class _Ent:
    """One entity: a type and the group code/value pairs under it."""

    __slots__ = ("type", "pairs")

    def __init__(self, etype, pairs):
        self.type = etype
        self.pairs = pairs

    def get(self, code, default=None):
        for c, v in self.pairs:
            if c == code:
                return v
        return default

    def num(self, code, default=0.0):
        v = self.get(code)
        if v is None:
            return float(default)
        try:
            return float(v)
        except ValueError:
            return float(default)

    def floats(self, code):
        out = []
        for c, v in self.pairs:
            if c == code:
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out


def _iter_entities(tokens):
    """Turn (code, value) groups into entities; POLYLINE/VERTEX/SEQEND together."""
    ents = []
    cur = None

    def flush():
        nonlocal cur
        if cur is not None and cur[0]:
            ents.append(_Ent(cur[0], cur[1]))
        cur = None

    i, n = 0, len(tokens)
    while i < n:
        code, value = tokens[i]
        if code == 0:
            t = value.strip().upper()
            if cur is not None and cur[0] == "POLYLINE" and t == "VERTEX":
                i += 1
                while i < n and tokens[i][0] != 0:
                    cur[1].append(tokens[i])
                    i += 1
                continue
            if cur is not None and cur[0] == "POLYLINE" and t == "SEQEND":
                i += 1
                while i < n and tokens[i][0] != 0:
                    i += 1
                flush()
                continue
            flush()
            cur = [t, []]
            i += 1
            continue
        if cur is not None:
            cur[1].append((code, value))
        i += 1
    flush()
    return ents


def _sections(pairs):
    """The pairs per section name (HEADER, TABLES, BLOCKS, ENTITIES)."""
    secs = {}
    i, n = 0, len(pairs)
    while i < n:
        code, value = pairs[i]
        if code == 0 and value.strip().upper() == "SECTION" and i + 1 < n \
                and pairs[i + 1][0] == 2:
            name = pairs[i + 1][1].strip().upper()
            body = []
            i += 2
            while i < n and not (pairs[i][0] == 0 and
                                 pairs[i][1].strip().upper() == "ENDSEC"):
                body.append(pairs[i])
                i += 1
            secs.setdefault(name, []).extend(body)
        i += 1
    return secs


# --------------------------------------------------------------------------
#  reading
# --------------------------------------------------------------------------

def read_dxf(source, tol=TOL) -> Drawing:
    """An ASCII DXF (path, bytes or text) as a 2D drawing in millimetres.

    Units from `$INSUNITS` are converted; entities in `ENTITIES` and in blocks via
    `INSERT` are followed; arcs, circles, ellipses and NURBS are flattened;
    extrusion (0, 0, -1) mirrors X; layers and colours come from the LAYER table
    and the entity. Loose line segments are chained together. Content that cannot
    be followed (text, hatches, dimensions) gives a message in
    `Drawing.warnings`.
    """
    data = _data(source)
    pairs = _pairs(_decode(data))
    if not pairs:
        raise ValueError(tr("This is not a DXF file: no group code pairs were found."))
    return _Reader(pairs, float(tol)).run()


class _Reader:
    def __init__(self, pairs, tol):
        self.pairs = pairs
        self.tol = tol
        self.drawing = Drawing()
        self.factor = 1.0
        self.layer_color = {}
        self.layer_visible = {}
        self.blocks = {}
        self.noticed = set()

    # -- pre-reading -------------------------------------------------------

    def run(self):
        secs = _sections(self.pairs)
        self._header(secs.get("HEADER", []))
        self._tables(secs.get("TABLES", []))
        self._blocks(secs.get("BLOCKS", []))
        for ent in _iter_entities(secs.get("ENTITIES", [])):
            self._emit(ent, np.eye(2), 0, ())
        for name, vis in self.layer_visible.items():
            self.drawing.ensure_layer(name, self.layer_color.get(name, "#000000")).visible = vis
        self.drawing.paths = join_open_paths(self.drawing.paths, max(self.tol, 1e-6))
        return self.drawing

    def _header(self, tokens):
        for i, (code, value) in enumerate(tokens):
            if code == 9 and value.strip().upper() == "$INSUNITS":
                for c2, v2 in tokens[i + 1:i + 4]:
                    if c2 == 70:
                        try:
                            units = int(float(v2))
                        except ValueError:
                            units = 0
                        self.factor = INSUNITS_MM.get(units, 1.0)
                        break

    def _tables(self, tokens):
        in_layer = False
        for ent in _iter_entities(tokens):
            if ent.type == "TABLE":
                in_layer = (ent.get(2, "").strip().upper() == "LAYER")
            elif ent.type == "ENDTAB":
                in_layer = False
            elif ent.type == "LAYER" and in_layer:
                name = ent.get(2, "0")
                aci = int(ent.num(62, 7))
                self.layer_visible[name] = aci >= 0
                self.layer_color[name] = aci_to_hex(abs(aci))

    def _blocks(self, tokens):
        i, n = 0, len(tokens)
        while i < n:
            code, value = tokens[i]
            if code == 0 and value.strip().upper() == "BLOCK":
                name = None
                head, body = [], []
                i += 1
                while i < n and tokens[i][0] != 0:
                    head.append(tokens[i])
                    i += 1
                while i < n and not (tokens[i][0] == 0 and
                                     tokens[i][1].strip().upper() == "ENDBLK"):
                    body.append(tokens[i])
                    i += 1
                for c, v in head:
                    if c == 2:
                        name = v
                if name:
                    self.blocks[name] = _iter_entities(body)
            i += 1

    # -- layers ------------------------------------------------------------

    def _notice(self, kind, text):
        if kind not in self.noticed:
            self.noticed.add(kind)
            self.drawing.warn(text)

    def _layer_of(self, ent):
        return ent.get(8, "0") or "0"

    def _color_of(self, ent, layer):
        tc = ent.get(420)
        if tc is not None:
            return _true_color(tc) or self.layer_color.get(layer, "#000000")
        aci = ent.get(62)
        if aci is None:
            return self.layer_color.get(layer, "#000000")
        try:
            aci = int(float(aci))
        except ValueError:
            return self.layer_color.get(layer, "#000000")
        if aci in (BYLAYER, BYBLOCK):
            return self.layer_color.get(layer, "#000000")
        return aci_to_hex(aci)

    def _open_layer(self, layer, color):
        if layer not in self.drawing.layers:
            self.drawing.ensure_layer(layer, self.layer_color.get(layer, color))

    # -- entities ----------------------------------------------------------

    def _emit(self, ent, M, depth, seen):
        t = ent.type
        if t == "INSERT":
            self._insert(ent, M, depth, seen)
            return
        if t == "POINT":
            return
        if t in SKIP_MESSAGES:
            self._notice(t, SKIP_MESSAGES[t])
            return
        if t not in SUPPORTED:
            return
        layer = self._layer_of(ent)
        color = self._color_of(ent, layer)
        self._open_layer(layer, color)
        ext = (ent.num(210, 0.0), ent.num(220, 0.0), ent.num(230, 1.0))
        Mo = _ocs_matrix(ext) @ M
        for pts, closed in self._geom(ent, t):
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(pts) < (3 if closed else 2):
                continue
            q = np.ascontiguousarray(pts @ Mo.T * self.factor, dtype=float)
            self.drawing.add_path(q, closed, layer)

    def _insert(self, ent, M, depth, seen):
        name = ent.get(2, "")
        if name in seen:
            self._notice("recur", tr("Nested copy (INSERT) with a loop was stopped."))
            return
        block = self.blocks.get(name)
        if block is None:
            self._notice("missing", tr("Block {name} of INSERT is not in the file.").format(name=name))
            return
        if depth > 16:
            self._notice("deep", tr("Copies (INSERT) nested too deeply; stopping."))
            return
        rows = int(ent.num(70, 1))
        cols = int(ent.num(71, 1))
        if rows > 1 or cols > 1:
            self._notice("array", tr("Rows and columns of INSERT were skipped; "
                                     "only the first instance was read."))
        ip = np.array([ent.num(10, 0.0), ent.num(20, 0.0)]) * self.factor
        sx, sy = ent.num(41, 1.0), ent.num(42, 1.0)
        rot = math.radians(ent.num(50, 0.0))
        co, si = math.cos(rot), math.sin(rot)
        lin = np.array([[co * sx, -si * sy], [si * sx, co * sy]])
        ext = (ent.num(210, 0.0), ent.num(220, 0.0), ent.num(230, 1.0))
        Mi = _ocs_matrix(ext) @ M
        layer = self._layer_of(ent)
        self._open_layer(layer, self.layer_color.get(layer, "#000000"))
        for cent in block:
            if cent.type == "POINT":
                continue
            if cent.type == "INSERT":
                self._insert(cent, Mi @ lin, depth + 1, seen + (name,))
                continue
            if cent.type not in SUPPORTED:
                continue
            sub = _Ent(cent.type, cent.pairs)
            sub_layer = cent.get(8, "") or ""
            if sub_layer in ("", "0"):
                sub_layer = layer
            self._open_layer(sub_layer, self.layer_color.get(sub_layer, "#000000"))
            sext = (cent.num(210, 0.0), cent.num(220, 0.0), cent.num(230, 1.0))
            Ms = _ocs_matrix(sext)
            for pts, closed in self._geom(sub, sub.type):
                pts = np.asarray(pts, dtype=float).reshape(-1, 2)
                if len(pts) < (3 if closed else 2):
                    continue
                q = pts @ Ms.T
                q = q @ lin.T + ip
                q = np.ascontiguousarray(q @ Mi.T, dtype=float)
                self.drawing.add_path(q, closed, sub_layer)

    def _geom(self, ent, t):
        tol = self.tol
        if t == "LINE":
            p0 = [ent.num(10), ent.num(20)]
            p1 = [ent.num(11), ent.num(21)]
            return [(np.array([p0, p1], float), False)]
        if t == "CIRCLE":
            c = [ent.num(10), ent.num(20)]
            r = ent.num(40)
            if r <= 0:
                return []
            pts = arc_points(c, r, 0.0, 2.0 * math.pi, tol)
            return [(np.asarray(pts[:-1], float), True)]
        if t == "ARC":
            c = [ent.num(10), ent.num(20)]
            r = ent.num(40)
            a0 = math.radians(ent.num(50))
            a1 = math.radians(ent.num(51))
            if r <= 0:
                return []
            span = (a1 - a0) % (2.0 * math.pi)
            if span <= 0:
                return []
            return [(arc_points(c, r, a0, a0 + span, tol), False)]
        if t == "ELLIPSE":
            c = [ent.num(10), ent.num(20)]
            ax = [ent.num(11), ent.num(21)]
            ratio = ent.num(40, 1.0)
            t0 = ent.num(41, 0.0)
            t1 = ent.num(42, 2.0 * math.pi)
            a = math.hypot(ax[0], ax[1])
            if a <= 0 or ratio <= 0:
                return []
            b = a * ratio
            rot = math.atan2(ax[1], ax[0])
            full = abs((t1 - t0) - 2.0 * math.pi) < 1e-6
            pts = ellipse_arc_points(c, a, b, t0, t1, tol, rot)
            if full:
                return [(np.asarray(pts[:-1], float), True)]
            return [(np.asarray(pts, float), False)]
        if t in ("LWPOLYLINE", "POLYLINE"):
            verts, bulges = _vertices_and_bulges(ent.pairs)
            closed = bool(int(ent.num(70, 0)) & 1)
            pts, cl = _polyline_from_vertices(verts, bulges, closed, tol)
            return [(pts, cl)] if len(pts) >= 2 else []
        if t == "SPLINE":
            return self._spline(ent)
        return []

    def _spline(self, ent):
        degree = int(ent.num(71, 3))
        knots = ent.floats(40)
        weights = ent.floats(42)
        verts, _ = _vertices_and_bulges(ent.pairs)
        if len(verts) >= 2 and degree >= 1 and len(knots) >= len(verts) + degree + 1:
            pts = nurbs_points(degree, knots, verts, weights or None)
            return [(pts, False)] if len(pts) >= 2 else []
        fx, fy = ent.floats(11), ent.floats(21)
        m = min(len(fx), len(fy))
        if m >= 2:
            return [(np.array([(fx[i], fy[i]) for i in range(m)], float), False)]
        return []


# --------------------------------------------------------------------------
#  writing
# --------------------------------------------------------------------------

def _fmt(v) -> str:
    s = f"{float(v):.6f}"
    return "0" if s in ("-0.000000", "0.000000") else s


def _fit_circle(pts):
    p = np.asarray(pts, dtype=float)
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    b = p[:, 0] ** 2 + p[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _is_circle(pts, tol=0.01):
    p = np.asarray(pts, dtype=float)
    if len(p) < 16:
        return None
    fit = _fit_circle(p)
    if fit is None:
        return None
    cx, cy, r = fit
    dev = float(np.max(np.abs(np.linalg.norm(p - np.array([cx, cy]), axis=1) - r)))
    return (cx, cy, r) if dev <= tol else None


def write_dxf(drawing, path=None, layers=None, detect_circles=True) -> str:
    """The drawing as DXF R12 text; also written to `path` when given.

    `POLYLINE`/`VERTEX`/`SEQEND` for closed and long paths, `LINE` for two-point
    paths and `CIRCLE` for closed paths of at least 16 points that lie on a circle
    within 0.01 mm (with `detect_circles`). Hidden layers are skipped; layers in
    `layers` limit the output. Ends with `EOF`.
    """
    wanted = set(layers) if layers is not None else None
    out = []

    def rec(code, value):
        out.append(str(code))
        out.append(str(value))

    rec(0, "SECTION")
    rec(2, "HEADER")
    rec(9, "$ACADVER")
    rec(1, "AC1009")
    rec(9, "$INSUNITS")
    rec(70, 4)
    rec(9, "$MEASUREMENT")
    rec(70, 1)
    rec(0, "ENDSEC")

    keep = []
    for name, lay in drawing.layers.items():
        if (wanted is not None and name not in wanted) or not lay.visible:
            continue
        keep.append((name, lay))
    if "0" not in {n for n, _ in keep}:
        keep.insert(0, ("0", Layer("0", "#000000")))
    rec(0, "SECTION")
    rec(2, "TABLES")
    rec(0, "TABLE")
    rec(2, "LAYER")
    rec(70, len(keep))
    for name, lay in keep:
        rec(0, "LAYER")
        rec(2, name)
        rec(70, 0)
        rec(62, hex_to_aci(lay.color))
        rec(6, "CONTINUOUS")
    rec(0, "ENDTAB")
    rec(0, "ENDSEC")

    rec(0, "SECTION")
    rec(2, "ENTITIES")
    kept = {n for n, _ in keep}
    for p in drawing.paths:
        if p.layer not in kept or len(p.pts) < 2:
            continue
        pts = np.asarray(p.pts, dtype=float)
        circ = _is_circle(pts, 0.01) if (detect_circles and p.closed) else None
        if circ is not None:
            cx, cy, r = circ
            rec(0, "CIRCLE")
            rec(8, p.layer)
            rec(10, _fmt(cx))
            rec(20, _fmt(cy))
            rec(30, 0)
            rec(40, _fmt(r))
        elif len(pts) == 2 and not p.closed:
            rec(0, "LINE")
            rec(8, p.layer)
            rec(10, _fmt(pts[0, 0]))
            rec(20, _fmt(pts[0, 1]))
            rec(30, 0)
            rec(11, _fmt(pts[1, 0]))
            rec(21, _fmt(pts[1, 1]))
            rec(31, 0)
        else:
            rec(0, "POLYLINE")
            rec(8, p.layer)
            rec(66, 1)
            rec(70, 1 if p.closed else 0)
            for x, y in pts:
                rec(0, "VERTEX")
                rec(8, p.layer)
                rec(10, _fmt(x))
                rec(20, _fmt(y))
                rec(30, 0)
            rec(0, "SEQEND")
    rec(0, "ENDSEC")
    rec(0, "EOF")

    text = "\n".join(out) + "\n"
    if path is not None:
        def write(tmp):
            with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
                f.write(text)
        fileio.atomic_write(path, write)
    return text
