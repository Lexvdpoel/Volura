"""SVG lezen en schrijven naar het gedeelde 2D-model (`volura.vector2d`).

Bij het lezen wordt de pagina in millimeters gezet met Y omhoog en de onderkant
van de pagina als y = 0 (SVG zelf heeft Y omlaag), blijven de Inkscape-lagen of
de lijnkleuren als lagen bewaard en onthoudt elk pad de vulregel van zijn
bronelement. Bij het schrijven krijgt elke laag een `<g>` met
`inkscape:groupmode="layer"`, de laagkleur als lijnkleur en de bewerking in
`data-volura-op` met de instellingen in `data-volura-param-*`, zodat Volura de
lagen van een laser of frees terugvindt, en krijgt elk bronelement één `<path>` met
al zijn deelwegen, zodat gaten (een cirkel in een cirkel) ook teruglezen als één vorm.
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from . import fileio
from .vector2d import (Drawing, TOL, ellipse_arc_points, flatten_cubic, flatten_quad,
                       svg_arc)

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Eenheden uit de SVG-specificatie, omgerekend naar millimeters.
UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0, "pc": 25.4 / 6.0}
EM_MM = 25.4 / 72.0 * 16.0        # 1 em = 16 px, px bij 96 dpi

# Elementen zonder zichtbare vorm (inhoud van defs, alleen bereikbaar via use).
SKIP_TAGS = {"style", "metadata", "title", "desc", "defs", "clippath", "mask", "pattern",
             "filter", "marker", "symbol", "solidcolor", "lineargradient", "radialgradient",
             "stop", "link", "font", "font-face", "namedview", "editorinfo", "page",
             "grid", "guide", "interface", "menu", "foreignobject"}
TEXT_TAGS = {"text", "tspan", "textpath", "altglyph", "tref", "textarea"}
RASTER_TAGS = {"image", "video", "audio", "canvas"}
GROUP_TAGS = {"g", "a", "switch", "svg"}

# De gangbare SVG-kleurnamen; onbekende namen leveren geen kleur op.
COLORS = {
    "black": "#000000", "silver": "#c0c0c0", "gray": "#808080", "grey": "#808080",
    "white": "#ffffff", "maroon": "#800000", "red": "#ff0000", "purple": "#800080",
    "fuchsia": "#ff00ff", "magenta": "#ff00ff", "green": "#008000", "lime": "#00ff00",
    "olive": "#808000", "yellow": "#ffff00", "navy": "#000080", "blue": "#0000ff",
    "teal": "#008080", "aqua": "#00ffff", "orange": "#ffa500", "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7", "aquamarine": "#7fffd4", "azure": "#f0ffff",
    "beige": "#f5f5dc", "bisque": "#ffe4c4", "blanchedalmond": "#ffebcd",
    "blueviolet": "#8a2be2", "brown": "#a52a2a", "burlywood": "#deb887",
    "cadetblue": "#5f9ea0", "chartreuse": "#7fff00", "chocolate": "#d2691e",
    "coral": "#ff7f50", "cornflowerblue": "#6495ed", "cornsilk": "#fff8dc",
    "crimson": "#dc143c", "cyan": "#00ffff", "darkblue": "#00008b", "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b", "darkgray": "#a9a9a9", "darkgreen": "#006400",
    "darkgrey": "#a9a9a9", "darkkhaki": "#bdb76b", "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f", "darkorange": "#ff8c00", "darkorchid": "#9932cc",
    "darkred": "#8b0000", "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f", "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3", "deeppink": "#ff1493", "deepskyblue": "#00bfff",
    "dimgray": "#696969", "dodgerblue": "#1e90ff", "firebrick": "#b22222",
    "floralwhite": "#fffaf0", "forestgreen": "#228b22", "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff", "gold": "#ffd700", "goldenrod": "#daa520",
    "greenyellow": "#adff2f", "honeydew": "#f0fff0", "hotpink": "#ff69b4",
    "indianred": "#cd5c5c", "indigo": "#4b0082", "ivory": "#fffff0", "khaki": "#f0e68c",
    "lavender": "#e6e6fa", "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff", "lightgoldenrodyellow": "#fafad2",
    "lightgray": "#d3d3d3", "lightgreen": "#90ee90", "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1", "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa", "lightslategray": "#778899", "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0", "limegreen": "#32cd32", "linen": "#faf0e6",
    "mediumaquamarine": "#66cdaa", "mediumblue": "#0000cd", "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db", "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585", "midnightblue": "#191970", "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5", "navajowhite": "#ffdead",
    "oldlace": "#fdf5e6", "olivedrab": "#6b8e23", "orangered": "#ff4500",
    "orchid": "#da70d6", "palegoldenrod": "#eee8aa", "palegreen": "#98fb98",
    "paleturquoise": "#afeeee", "palevioletred": "#db7093", "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9", "peru": "#cd853f", "pink": "#ffc0cb", "plum": "#dda0dd",
    "powderblue": "#b0e0e6", "rosybrown": "#bc8f8f", "royalblue": "#4169e1",
    "saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460",
    "seagreen": "#2e8b57", "seashell": "#fff5ee", "sienna": "#a0522d",
    "skyblue": "#87ceeb", "slateblue": "#6a5acd", "slategray": "#708090",
    "snow": "#fffafa", "springgreen": "#00ff7f", "steelblue": "#4682b4",
    "tan": "#d2b48c", "thistle": "#d8bfd8", "tomato": "#ff6347",
    "turquoise": "#40e0d0", "violet": "#ee82ee", "wheat": "#f5deb3",
    "whitesmoke": "#f5f5f5", "yellowgreen": "#9acd32",
    "none": None, "transparent": None, "currentcolor": None,
}

_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
_STYLE_PROPS = ("fill", "stroke", "fill-rule", "display", "visibility", "stroke-width")


# --------------------------------------------------------------------------
#  getallen, paden en matrices
# --------------------------------------------------------------------------

class _Scanner:
    """Leest een padopgave: commando's en getallen, ook zonder scheidingsteken."""

    def __init__(self, text):
        self.s = text or ""
        self.i = 0

    def skip(self):
        s = self.s
        while self.i < len(s) and (s[self.i].isspace() or s[self.i] == ","):
            self.i += 1

    def command(self):
        self.skip()
        if self.i < len(self.s) and self.s[self.i].isalpha():
            c = self.s[self.i]
            self.i += 1
            return c
        return None

    def has_number(self):
        self.skip()
        m = _NUM.match(self.s, self.i)
        return m is not None and m.end() > m.start()

    def number(self):
        self.skip()
        m = _NUM.match(self.s, self.i)
        if m is None or m.end() == m.start():
            raise ValueError("Onjuist getal in de padopgave van een SVG-element.")
        self.i = m.end()
        return float(m.group())

    def flag(self):
        """Een boogvlag is één teken 0 of 1, ook direct achter een ander getal."""
        self.skip()
        if self.i < len(self.s) and self.s[self.i] in "01":
            f = self.s[self.i] == "1"
            self.i += 1
            return f
        return bool(self.number())


def _numbers(text):
    return [float(x) for x in _NUM.findall(text or "")]


def path_subpaths(d, tol=TOL):
    """De `d`-opgave van een <path> als polylijnen: lijst van (punten, gesloten).

    Alle commando's van SVG 1.1 (M L H V C S Q T A Z, absoluut en relatief, met
    impliciete herhaling van het laatste commando) worden ondersteund. Krommen en
    bogen worden adaptief afgevlakt tot maximaal `tol` millimeter afwijking.
    Deelwegen van één element vormen samen één vorm (zie `vector2d.regions`).
    """
    sc = _Scanner(d)
    out = []
    pts: list[np.ndarray] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    prev_c = prev_q = None

    def close_open(closed=False):
        if len(pts) >= 2:
            out.append((np.asarray(pts, dtype=float), closed))

    cmd = sc.command()
    while cmd is not None:
        rel = cmd.islower()
        c = cmd.upper()
        if c not in "MLHVCSQTAZ":
            raise ValueError(f"onbekend commando '{cmd}' in de padopgave")
        first = True
        while True:
            if c == "Z":
                close_open(True)
                pts = []
                cur = start.copy()
                prev_c = prev_q = None
                break
            if not sc.has_number():
                break
            if c == "A":
                rx, ry, phi = sc.number(), sc.number(), sc.number()
                large, sweep = sc.flag(), sc.flag()
                end = _rel(cur, [sc.number(), sc.number()], rel)
                pts.extend(svg_arc(cur, rx, ry, phi, large, sweep, end, tol))
                cur, prev_c, prev_q = end, None, None
            elif c in ("C", "S", "Q", "T"):
                if c == "C":
                    c1 = _rel(cur, [sc.number(), sc.number()], rel)
                    c2 = _rel(cur, [sc.number(), sc.number()], rel)
                elif c == "S":
                    c1 = cur if prev_c is None else 2.0 * cur - prev_c
                    c2 = _rel(cur, [sc.number(), sc.number()], rel)
                elif c == "Q":
                    c1 = _rel(cur, [sc.number(), sc.number()], rel)
                    c2 = None
                else:
                    c1 = cur if prev_q is None else 2.0 * cur - prev_q
                    c2 = None
                end = _rel(cur, [sc.number(), sc.number()], rel)
                pts.extend(flatten_cubic(cur, c1, c2, end, tol) if c2 is not None
                           else flatten_quad(cur, c1, end, tol))
                cur = end
                prev_c = c2
                prev_q = None if c2 is not None else c1
            else:
                if c == "H":
                    v = sc.number()
                    step = np.array([v if rel else v - cur[0], 0.0])
                elif c == "V":
                    v = sc.number()
                    step = np.array([0.0, v if rel else v - cur[1]])
                else:
                    step = _rel(cur, [sc.number(), sc.number()], rel) - cur
                if c == "M":
                    close_open()
                    start = cur + step
                    pts = [start.copy()]
                else:
                    pts.append(cur + step)
                cur = cur + step
                prev_c = prev_q = None
            # een moveto met meerdere coordparen herhaalt zich als lijnstuk
            if first and c == "M":
                c = "l" if rel else "L"
            first = False
            if not sc.has_number():
                break
        cmd = sc.command()
    close_open()
    return out


def _rel(cur, args, rel):
    a = np.asarray(args, dtype=float)
    return cur + a if rel else a


def _transform_matrix(attr) -> np.ndarray:
    """De `transform`-opgave als 3x3 matrix (matrix, translate, scale, rotate, skew)."""
    M = np.eye(3)
    for name, args in re.findall(r"([A-Za-z]+)\s*\(([^()]*)\)", attr or ""):
        v = _numbers(args)
        n = name.lower()
        A = None
        if n == "matrix" and len(v) >= 6:
            A = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0.0, 0.0, 1.0]])
        elif n == "translate":
            A = np.array([[1.0, 0.0, v[0] if v else 0.0],
                          [0.0, 1.0, v[1] if len(v) > 1 else 0.0], [0.0, 0.0, 1.0]])
        elif n == "scale":
            sx = v[0] if v else 1.0
            sy = v[1] if len(v) > 1 else sx
            A = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        elif n == "rotate" and v:
            a = math.radians(v[0])
            A = np.array([[math.cos(a), -math.sin(a), 0.0],
                          [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
            if len(v) > 2:
                cx, cy = v[1], v[2]
                off = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
                back = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
                A = off @ A @ back
        elif n == "skewx" and v:
            A = np.array([[1.0, math.tan(math.radians(v[0])), 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        elif n == "skewy" and v:
            A = np.array([[1.0, 0.0, 0.0],
                          [math.tan(math.radians(v[0])), 1.0, 0.0], [0.0, 0.0, 1.0]])
        if A is not None:
            M = M @ A
    return M


# --------------------------------------------------------------------------
#  stijlen
# --------------------------------------------------------------------------

def _color(value):
    """Kleur als kleine letters "#rrggbb", of None (none, transparent, onbekend)."""
    if not value:
        return None
    s = str(value).strip().lower()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
            return "#" + s
        return None
    if s in COLORS:
        return COLORS[s]
    m = re.fullmatch(r"rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)\s*(?:[,/]\s*[\d.]+%?\s*)?\)", s)
    if m:
        return "#" + "".join(f"{min(255, int(c)):02x}" for c in m.groups())
    return None


def _declarations(text):
    """`style="..."` of het lichaam van een css-regel, als woordenboek."""
    out = {}
    for part in (text or "").split(";"):
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        val = re.sub(r"!important.*$", "", val).strip()
        key = key.strip().lower()
        if key in _STYLE_PROPS and val:
            out[key] = val.lower()
    return out


def _css_rules(text):
    """Eenvoudige css-regels: `.klas`, `#id` en `element`, ook gecombineerd."""
    rules = []
    text = re.sub(r"/\*.*?\*/", "", text or "", flags=re.S)
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text, flags=re.S):
        props = _declarations(body)
        if not props:
            continue
        for one in sel.split(","):
            one = one.strip()
            if one and not re.search(r"[ >+~:\[]", one):
                rules.append((one, props))
    return rules


def _matches(sel, tag, classes, ident):
    m = re.fullmatch(r"([A-Za-z][\w-]*)?((?:[.#][\w-]+)*)", sel)
    if not m:
        return False
    if m.group(1) and m.group(1).lower() != tag:
        return False
    for part in re.findall(r"[.#][\w-]+", m.group(2) or ""):
        if part[0] == "." and part[1:] not in classes:
            return False
        if part[0] == "#" and part[1:] != ident:
            return False
    return True


# --------------------------------------------------------------------------
#  eenvoudige vormen
# --------------------------------------------------------------------------

def _points_attr(text):
    v = _numbers(text)
    if len(v) < 4:
        return None
    return np.asarray(v[: len(v) // 2 * 2], dtype=float).reshape(-1, 2)


def rect_subpaths(x, y, w, h, rx=None, ry=None, tol=TOL):
    """`<rect>`, ook met afgeronde hoeken (rx/ry), als één gesloten polylijn."""
    if w <= 0.0 or h <= 0.0:
        return []
    if rx is None and ry is None:
        rx = ry = 0.0
    rx = min(float(rx or 0.0), w / 2.0)
    ry = min(float(ry or 0.0), h / 2.0)
    if rx <= 0.0 or ry <= 0.0:
        return [(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float), True)]
    pts = [np.array([x + rx, y], float)]
    for cx, cy, a in ((x + w - rx, y + ry, -90.0), (x + w - rx, y + h - ry, 0.0),
                      (x + rx, y + h - ry, 90.0), (x + rx, y + ry, 180.0)):
        a0, a1 = math.radians(a), math.radians(a + 90.0)
        arc = ellipse_arc_points((cx, cy), rx, ry, a0, a1, tol)
        pts.extend(arc[1:-1])
        pts.append(np.array([cx + rx * math.cos(a1), cy + ry * math.sin(a1)], float))
    return [(np.asarray(pts, dtype=float), True)]


def ellipse_subpaths(cx, cy, rx, ry, tol=TOL):
    """`<circle>` en `<ellipse>` als één gesloten polylijn."""
    if rx <= 0.0 or ry <= 0.0:
        return []
    pts = ellipse_arc_points((cx, cy), rx, ry, 0.0, 2.0 * math.pi, tol)
    return [(np.asarray(pts[:-1], dtype=float), True)]


# --------------------------------------------------------------------------
#  lezen
# --------------------------------------------------------------------------

def _local(tag):
    return str(tag).rsplit("}", 1)[-1].lower()


def _length(value, dpi):
    """Een SVG-lengte in millimeters; None bij percentage, ontbreken of onjuist."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s.endswith("%"):
        return None
    m = re.fullmatch(r"([-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?)"
                     r"(mm|cm|in|pt|pc|px|em|ex)?", s)
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2) or "px"
    if unit == "px":
        return v * 25.4 / float(dpi)
    if unit in ("em", "ex"):
        return v * EM_MM
    return v * UNIT_MM[unit]


def _view_box(elem):
    v = _numbers(elem.get("viewBox"))
    if len(v) >= 4 and v[2] > 0 and v[3] > 0:
        return v[0], v[1], v[2], v[3]
    return None


def _data(source) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        return source.read()
    s = os.fspath(source) if hasattr(source, "__fspath__") else str(source)
    if s.lstrip()[:1] == "<":
        return s.encode("utf-8")
    try:
        with open(s, "rb") as f:
            return f.read()
    except OSError as exc:
        raise ValueError(f"Kan het SVG-bestand niet openen: {exc}") from exc


def read_svg(source, tol=TOL, dpi=96) -> Drawing:
    """Een SVG-bestand (pad, bytes of tekst) als 2D-tekening in millimeters.

    Y wijst omhoog met de onderkant van de pagina als y = 0. Inkscape-lagen
    worden lagen; zonder lagen krijgt elke lijnkleur een eigen laag, zodat de
    kleurconventie van lasersoftware (rood = snijden) bewaard blijft. Onbekende
    of niet-navolgbare inhoud (tekst, afbeeldingen) geeft een melding in
    `Drawing.warnings`.
    """
    data = _data(source)
    if b"<!ENTITY" in data:
        raise ValueError("Een SVG met entiteiten (<!ENTITY) wordt niet gelezen. "
                         "Sla het bestand opnieuw op als platte SVG.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"De SVG kan niet worden gelezen: {exc}") from exc
    if _local(root.tag) != "svg":
        raise ValueError("Dit is geen SVG-bestand: er is geen <svg>-element gevonden.")
    return _Reader(root, tol, dpi).run()


class _Reader:
    """Loop door de SVG-boom en zet vormen om naar millimeters met Y omhoog."""

    def __init__(self, root, tol, dpi):
        self.root = root
        self.tol = float(tol)
        self.dpi = float(dpi)
        self.drawing = Drawing()
        self.rules = []
        self.byId = {}
        self.src = 0
        self.noticed = set()
        for elem in self.root.iter():
            ident = elem.get("id")
            if ident and ident not in self.byId:
                self.byId[ident] = elem
            if _local(elem.tag) == "style" and elem.text:
                self.rules.extend(_css_rules(elem.text))

    # -- pagina ------------------------------------------------------------

    def page(self):
        """Matrix naar millimeters en de onderkant van de pagina (voor de Y-omkering).

        De matige houding van `preserveAspectRatio` wordt toegepast, maar de
        oorsprong van de gebruikerseenheden blijft op zijn plek: een tekening die
        in het bestand op 50 mm staat, komt op 50 mm binnen. Zo blijft een
        rondreis door `write_svg` en `read_svg` exact.
        """
        root = self.root
        px = 25.4 / self.dpi
        vb = _view_box(root)
        w = _length(root.get("width"), self.dpi)
        h = _length(root.get("height"), self.dpi)
        base = np.array([[px, 0.0, 0.0], [0.0, px, 0.0], [0.0, 0.0, 1.0]])
        if vb is None:
            if w is None and h is None:
                return base, None, None, None
            return base, (h if h is not None else w), w, h
        x0, y0, vw, vh = vb
        if w is None and h is None:            # geen pagina: viewBox-eenheden zijn px
            return base, px * (y0 + vh), vw * px, vh * px
        if h is None:
            h = w * vh / vw
        elif w is None:
            w = h * vw / vh
        par = (root.get("preserveAspectRatio") or "xMidYMid meet").strip().lower()
        sx, sy = w / vw, h / vh
        if par.startswith("none"):
            M = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
            bottom = sy * (y0 + vh)
        else:
            s = max(sx, sy) if "slice" in par else min(sx, sy)
            tx = 0.0 if "xmin" in par else (w - s * vw if "xmax" in par else (w - s * vw) / 2.0)
            ty = 0.0 if "ymin" in par else (h - s * vh if "ymax" in par else (h - s * vh) / 2.0)
            M = np.array([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]])
            bottom = s * (y0 + vh) + ty
        return M, bottom, w, h

    # -- stijlen -----------------------------------------------------------

    def style(self, elem, inherited):
        """Geërfde stijl, presentatie-attributen, css en dan de inline-opgave."""
        st = dict(inherited)
        for key, val in elem.attrib.items():
            if _local(key) in _STYLE_PROPS and val:
                st[_local(key)] = val.strip().lower()
        tag = _local(elem.tag)
        classes = (elem.get("class") or "").split()
        ident = elem.get("id")
        for sel, props in self.rules:
            if _matches(sel, tag, classes, ident):
                st.update(props)
        st.update(_declarations(elem.get("style")))
        return st

    # -- omlopen -----------------------------------------------------------

    def run(self):
        base, bottom, pw, ph = self.page()
        self.walk(self.root, base, {}, None, frozenset())
        d = self.drawing
        flip = bottom
        if flip is None:
            b = d.bounds()
            flip = b[3] if b else 0.0
        out = d.transformed(np.array([[1.0, 0.0, 0.0], [0.0, -1.0, flip], [0.0, 0.0, 1.0]]))
        out.sheet_w, out.sheet_h = pw, ph
        return out

    def notice(self, kind, text):
        if kind not in self.noticed:
            self.noticed.add(kind)
            self.drawing.warn(text)

    def walk(self, elem, M, style, ink_layer, seen):
        tag = _local(elem.tag)
        st = self.style(elem, style)
        if st.get("display") == "none" or tag in SKIP_TAGS:
            return
        if tag in TEXT_TAGS:
            self.notice("text", "Tekst in de SVG is overgeslagen; zet tekst in Inkscape om "
                                "naar paden (Pad > Object naar pad) om mee te snijden.")
            return
        if tag in RASTER_TAGS:
            self.notice("image", "Afbeelding in de SVG is overgeslagen; een laser of frees "
                                 "kan geen plaatje volgen.")
            return
        M = M @ _transform_matrix(elem.get("transform"))
        if tag in GROUP_TAGS:
            if elem.get(f"{{{INK_NS}}}groupmode") == "layer":
                name = elem.get(f"{{{INK_NS}}}label") or "laag"
                lay = self.drawing.ensure_layer(name, _color(st.get("stroke")) or "#000000",
                                                elem.get("data-volura-op"))
                params = layer_params_of(elem)
                if params:
                    lay.params = {**dict(lay.params or {}), **params}
                ink_layer = name
            for child in elem:
                self.walk(child, M, st, ink_layer, seen)
            return
        if tag == "use":
            ref = (elem.get(f"{{{XLINK_NS}}}href") or elem.get("href") or "").strip()
            if not ref.startswith("#") or ref[1:] in seen:
                return
            target = self.byId.get(ref[1:])
            if target is None:
                self.drawing.warn(f"De verwijzing {ref} van <use> staat niet in het bestand.")
                return
            x = (_numbers(elem.get("x")) + [0.0])[0]
            y = (_numbers(elem.get("y")) + [0.0])[0]
            shift = np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])
            self.walk(target, M @ shift, st, ink_layer, seen | {ref[1:]})
            return
        if st.get("visibility") == "hidden":
            return
        try:
            self.emit(elem, tag, M, st, ink_layer)
        except ValueError as exc:
            self.drawing.warn(f"Element <{tag}> is overgeslagen: {exc}")

    def emit(self, elem, tag, M, st, ink_layer):
        def num(name, dflt=0.0):
            v = _numbers(elem.get(name))
            return v[0] if v else dflt

        tol = self.tol
        if tag == "path":
            subs = path_subpaths(elem.get("d"), tol)
        elif tag == "rect":
            subs = rect_subpaths(num("x"), num("y"), num("width"), num("height"),
                                 num("rx", -1.0) if elem.get("rx") is not None else None,
                                 num("ry", -1.0) if elem.get("ry") is not None else None, tol)
        elif tag == "circle":
            r = num("r")
            subs = ellipse_subpaths(num("cx"), num("cy"), r, r, tol)
        elif tag == "ellipse":
            subs = ellipse_subpaths(num("cx"), num("cy"), num("rx"), num("ry"), tol)
        elif tag == "line":
            p = [num("x1"), num("y1"), num("x2"), num("y2")]
            subs = [(np.array([[p[0], p[1]], [p[2], p[3]]], float), False)]
        elif tag == "polyline":
            pts = _points_attr(elem.get("points"))
            subs = [(pts, False)] if pts is not None else []
        elif tag == "polygon":
            pts = _points_attr(elem.get("points"))
            subs = [(pts, True)] if pts is not None else []
        else:
            return
        fill = _color(st.get("fill", "#000000"))
        stroke = _color(st.get("stroke"))
        layer = ink_layer or stroke or fill
        if layer is None:
            return                      # geen lijn en geen vulling: niets te zien
        if ink_layer is None:
            self.drawing.ensure_layer(layer, layer)
        rule = "evenodd" if st.get("fill-rule") == "evenodd" else "nonzero"
        if elem.get("data-volura-rule") in ("nonzero", "evenodd"):
            rule = elem.get("data-volura-rule")
        marked = elem.get("data-volura-filled")
        self.src += 1
        src = self.src
        if str(elem.get("data-volura-src", "")).lstrip("-").isdigit():
            src = int(elem.get("data-volura-src"))
            self.src = max(self.src, src)
        lin, off = M[:2, :2], M[:2, 2]
        for pts, closed in subs:
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            if closed and len(pts) > 1 and np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-9):
                pts = pts[:-1]
            if len(pts) < (3 if closed else 2):
                continue
            q = np.ascontiguousarray(pts @ lin.T + off, dtype=float)
            filled = bool(closed and fill) if marked is None else (closed and marked == "1")
            self.drawing.add_path(q, closed, layer, filled=filled, rule=rule, src=src)


# --------------------------------------------------------------------------
#  schrijven
# --------------------------------------------------------------------------

def _fmt(v) -> str:
    s = f"{float(v):.4f}"
    return "0.0000" if s == "-0.0000" else s


def _safe_id(name):
    s = re.sub(r"[^\w.:-]", "_", str(name))
    return ("_" + s) if s[:1].isdigit() else (s or "laag")


def _attr(value):
    return escape(str(value), {'"': "&quot;"})


#: De instellingen van een laag gaan mee in `data-volura-param-<sleutel>`-attributen.
PARAM_PREFIX = "data-volura-param-"


def _param_text(value) -> str:
    """Een instelling als tekst: getallen zo dat ze weer als getal terugkomen."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return repr(round(float(value), 6))
    return str(value)


def _param_value(text):
    """De tekst van een instelling terug als getal of als woord."""
    s = str(text).strip()
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def layer_param_attrs(layer) -> list:
    """`data-volura-param-*-`-attributen bij de instellingen van een laag."""
    out = []
    for key, value in dict(getattr(layer, "params", None) or {}).items():
        name = re.sub(r"[^A-Za-z0-9_-]", "-", str(key))
        if not name or value is None:
            continue
        out.append(f'{PARAM_PREFIX}{name}="{_attr(_param_text(value))}"')
    return out


def layer_params_of(elem) -> dict:
    """De instellingen die in een `<g>` van een laag staan, terug als woordenboek."""
    out = {}
    for key, value in elem.attrib.items():
        name = _local(key)
        if name.startswith(PARAM_PREFIX):
            key = name[len(PARAM_PREFIX):]
            if key:
                out[key] = _param_value(value)
    return out


def page_of(drawing):
    """(x, y, breedte, hoogte) van de pagina: de plaat, of de omtrek plus 1 mm marge."""
    w, h = drawing.sheet_w, drawing.sheet_h
    if w is not None and h is not None and w > 0 and h > 0:
        return 0.0, 0.0, float(w), float(h)
    b = drawing.bounds()
    if b is None:
        return 0.0, 0.0, 100.0, 100.0
    return b[0] - 1.0, b[1] - 1.0, (b[2] - b[0]) + 2.0, (b[3] - b[1]) + 2.0


def write_svg(drawing, path=None, stroke_mm=0.01, layers=None, fill_layers=()):
    """De tekening als SVG-tekst; ook naar `path` geschreven als die gegeven is.

    De pagina is de plaat als `sheet_w`/`sheet_h` gezet zijn, anders de omtrek
    plus 1 mm marge. Elke laag wordt een `<g>` met `inkscape:groupmode="layer"`,
    de laagkleur als lijnkleur, de bewerking in `data-volura-op` en de instellingen
    van de laag in `data-volura-param-*` (kerf, vermogen, freesdiameter, ...); lagen
    in `fill_layers` krijgen een vulling met `fill-rule="evenodd"` (om te rasteren).
    Paden die bij één bronelement horen (`src`) komen in één `<path>`, zodat
    gaten en vulregels behouden blijven. Coördinaten: millimeters, Y omlaag,
    vier decimalen.
    """
    ox, oy, page_w, page_h = page_of(drawing)
    top = oy + page_h
    wanted = set(layers) if layers is not None else None
    filled_layers = set(fill_layers or ())
    body = []
    for name, lay in drawing.layers.items():
        if (wanted is not None and name not in wanted) or not lay.visible:
            continue
        paths = [p for p in drawing.paths if p.layer == name]
        color = lay.color or "#000000"
        attrs = [f'id="{_safe_id(name)}"', 'inkscape:groupmode="layer"',
                 f'inkscape:label="{_attr(name)}"', f'stroke="{color}"',
                 f'stroke-width="{_fmt(stroke_mm)}"', f'data-volura-op="{lay.op}"']
        attrs += layer_param_attrs(lay)
        if name in filled_layers:
            attrs += [f'fill="{color}"', 'fill-rule="evenodd"']
        else:
            attrs.append('fill="none"')
        els = []
        for group in _group_by_src(paths):
            parts = []
            for p in group:
                if len(p.pts) < 2:
                    continue
                parts.append("M" + "L".join(f"{_fmt(x)},{_fmt(top - y)}"
                                            for x, y in p.pts) + ("Z" if p.closed else ""))
            if not parts:
                continue
            extra = f' data-volura-src="{group[0].src}"'
            if group[0].filled:
                extra += ' data-volura-filled="1"'
            if group[0].rule != "nonzero":
                extra += f' data-volura-rule="{group[0].rule}"'
            els.append(f'    <path d="{" ".join(parts)}"{extra}/>')
        body.append("  <g " + " ".join(attrs) + ">"
                    + ("\n" + "\n".join(els) + "\n  " if els else "") + "</g>")
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="{SVG_NS}" xmlns:inkscape="{INK_NS}" xmlns:sodipodi="{SODIPODI_NS}"'
           f' xmlns:xlink="{XLINK_NS}" version="1.1"'
           f' width="{_fmt(page_w)}mm" height="{_fmt(page_h)}mm"'
           f' viewBox="{_fmt(ox)} {_fmt(oy)} {_fmt(page_w)} {_fmt(page_h)}">',
           "  <metadata>Volura</metadata>"]
    out.extend(body)
    out.append("</svg>")
    text = "\n".join(out) + "\n"
    if path is not None:
        def write(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
        fileio.atomic_write(path, write)
    return text


def _group_by_src(paths):
    """Opeenvolgende paden die bij één bronelement horen samen in één lijst."""
    groups = []
    for p in paths:
        if groups and groups[-1][0] == (p.src, p.filled, p.rule):
            groups[-1][1].append(p)
        else:
            groups.append(((p.src, p.filled, p.rule), [p]))
    return [g[1] for g in groups]
