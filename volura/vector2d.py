"""Het gedeelde 2D-datamodel voor vectorwerk: lagen, paden en tekeningen.

Alle 2D-coördinaten zijn millimeters met Y omhoog (wiskundige richting) en staan
als numpy-arrays ``(N, 2)`` float64. SVG en DXF, het 2D-werkblad, het nesten op
een plaat en de bewerkingen voor lasersnijder en CNC-frees delen dit model; er is
bewust geen tweede model.

Een tekening is een lijst paden met een laagnaam. De laag draagt de bedoeling
(snijden, graveren, frezen) en de kleur, zodat de kleurconventie van lasersoftware
(rood = snijden) bewaard blijft. Paden met hetzelfde volgnummer ``src`` komen uit
hetzelfde bronelement en vormen samen een gevulde vorm; dat is nodig om gaten
(een cirkel in een cirkel) goed te kunnen vullen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import manifold3d as mf

# Wat een laag voor de machine betekent. "none" is alleen ter informatie.
OPS = ("cut", "engrave", "raster", "mill_outside", "mill_inside", "pocket", "drill", "none")

# De Nederlandse namen van die bewerkingen, zoals het lagenpaneel ze toont.
OP_LABELS = {"cut": "Snijden", "engrave": "Graveren", "raster": "Rasteren",
             "mill_outside": "Frezen buiten", "mill_inside": "Frezen binnen",
             "pocket": "Uitkameren", "drill": "Boren", "none": "Geen (niet exporteren)"}

LASER_OPS = ("cut", "engrave", "raster", "drill")
MILL_OPS = ("mill_outside", "mill_inside", "pocket")

TOL = 0.01          # standaard afwijking van de afgevlakte kromme, in mm
MAX_DEPTH = 48      # diepte van de adaptieve verfijning
JSON_DECIMALS = 6   # verliesvrij op 1e-6 mm

# Het plaatformaat: marge langs de rand, tussenruimte tussen onderdelen,
# aantal platen naast elkaar en de afstand tussen twee platen op het werkblad.
DEFAULT_SHEET_MARGIN = 5.0
DEFAULT_SHEET_GAP = 2.0
DEFAULT_SHEET_COUNT = 1
SHEET_SPACING = 50.0

# De instellingen per bewerking, met de standaardwaarde erbij.
DEFAULT_KERF = 0.15        # breedte van de snedespleet van een laser, in mm
DEFAULT_TOOL_D = 3.175     # freesdiameter (1/8 inch), in mm
CORNER_STYLES = ("none", "dogbone", "tbone")
CORNER_LABELS = {"none": "Geen", "dogbone": "Dogbone", "tbone": "T-bone"}
LASER_PARAMS = (("power", 100.0), ("speed", 20.0), ("passes", 1), ("kerf", DEFAULT_KERF))
MILL_PARAMS = (("tool_d", DEFAULT_TOOL_D), ("depth", 3.0), ("step_depth", 1.0),
               ("stepover", 50.0), ("corner", "none"))
PARAM_LABELS = {"power": "Vermogen (%)", "speed": "Snelheid (mm/s)",
                "passes": "Aantal doorgangen", "kerf": "Kerf (mm)",
                "tool_d": "Freesdiameter (mm)", "depth": "Diepte (mm)",
                "step_depth": "Stapdiepte (mm)", "stepover": "Stapover (%)",
                "corner": "Hoekuitsparing"}


def default_params(op) -> dict:
    """De instellingen die bij een bewerking horen, met de standaardwaarden."""
    if op in MILL_OPS:
        return {key: value for key, value in MILL_PARAMS}
    if op in LASER_OPS:
        return {key: value for key, value in LASER_PARAMS}
    return {}


def layer_param(layer, key, default=None):
    """Een instelling van een laag; ontbrekend of onbruikbaar geeft de standaardwaarde."""
    if default is None:
        default = default_params(layer.op if layer is not None else "").get(key)
    value = dict(layer.params or {}).get(key, None) if layer is not None else None
    if isinstance(value, bool) or value is None:
        return default
    if key == "corner":
        return value if str(value) in CORNER_STYLES else "none"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if key == "passes":
        return max(1, int(round(value)))
    return value


# De standaardlagen van een nieuw werkblad, in de kleurconventie van LightBurn:
# rood snijdt, blauw graveert, zwart rastert, groen en oranje frezen, cyaan kameren uit.
DEFAULT_LAYERS = (("Snijden", "#ff0000", "cut"),
                  ("Graveren", "#0000ff", "engrave"),
                  ("Rasteren", "#000000", "raster"),
                  ("Frezen buiten", "#00e000", "mill_outside"),
                  ("Frezen binnen", "#ff8000", "mill_inside"),
                  ("Uitkameren", "#00e0e0", "pocket"),
                  ("Boren", "#ff00ff", "drill"),
                  ("Hulplijnen", "#a0a0a0", "none"))


def default_layers() -> dict:
    """De lagen van een leeg werkblad: "0" plus de standaardlagen (in die volgorde)."""
    out = {"0": Layer("0", "#000000", "cut", params=default_params("cut"))}
    for name, color, op in DEFAULT_LAYERS:
        out[name] = Layer(name, color, op, params=default_params(op))
    return out


#: Een laag die zo heet krijgt die kleur en bewerking alvast mee (LightBurn-conventie).
DEFAULT_BY_NAME = {name: (color, op) for name, color, op in DEFAULT_LAYERS}


def _arr(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        a = a.reshape(-1, 2)
    return np.ascontiguousarray(a, dtype=float)


# --------------------------------------------------------------------------
#  het model
# --------------------------------------------------------------------------

@dataclass
class Layer:
    """Een laag: een kleur, een bedoeling en de vrije instellingen daarbij."""

    name: str
    color: str
    op: str = "cut"
    visible: bool = True
    locked: bool = False
    params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "color": self.color, "op": self.op,
                "visible": bool(self.visible), "locked": bool(self.locked),
                "params": dict(self.params)}

    @staticmethod
    def from_json(d) -> "Layer":
        op = d.get("op", "cut")
        return Layer(d.get("name", "0"), str(d.get("color", "#000000")).lower(),
                     op if op in OPS else "cut", bool(d.get("visible", True)),
                     bool(d.get("locked", False)), dict(d.get("params") or {}))


def _clean_meta(meta) -> dict:
    """De extra gegevens van een pad (middelpunt en straal van een cirkel of boog),
    netjes als gewone waarden zodat ze in een bestand kunnen."""
    out = {}
    for k, v in dict(meta or {}).items():
        if isinstance(v, (list, tuple)):
            out[k] = [round(float(x), JSON_DECIMALS) for x in v]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = [round(float(x), JSON_DECIMALS) for x in v.reshape(-1).tolist()]
    return out


META_POINTS = ("center",)        # sleutels met een punt in de meta van een pad
META_LENGTHS = ("radius", "r", "rx", "ry")


def _transform_meta(meta, lin, off):
    """Meta mee met een transformatie: punten erdoorheen, stralen met de maat mee."""
    if not meta:
        return {}
    out = dict(meta)
    k = abs(float(np.linalg.det(lin)))
    scale = math.sqrt(k) if k > 0 else 1.0
    for name in META_POINTS:
        v = out.get(name)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2:
            out[name] = (np.asarray(v, float) @ lin.T + off).tolist()
    for name in META_LENGTHS:
        if isinstance(out.get(name), (int, float)):
            out[name] = float(out[name]) * scale
    if "rot" in out and isinstance(out["rot"], (int, float)) and k > 0:
        out["rot"] = float(out["rot"]) + math.atan2(float(lin[1, 0]), float(lin[0, 0]))
    return out


@dataclass
class VPath:
    """Een polylijn: open of gesloten, met de laag en de herkomst uit de bron.

    `meta` bewaart wat de afgevlakte punten kwijt zijn: bij een cirkel of boog het
    middelpunt en de straal, zodat het werkblad ze als cirkel kan snappen en de
    gereedschappen ermee kunnen rekenen.
    """

    pts: np.ndarray
    closed: bool
    layer: str = "0"
    filled: bool = False
    rule: str = "nonzero"
    src: int = 0
    meta: dict = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, VPath):
            return NotImplemented
        return (self.closed == other.closed and self.layer == other.layer
                and self.filled == other.filled and self.rule == other.rule
                and self.src == other.src and np.array_equal(self.pts, other.pts))

    def __len__(self):
        return len(self.pts)

    def reversed(self) -> "VPath":
        return VPath(np.ascontiguousarray(self.pts[::-1]), self.closed, self.layer,
                     self.filled, self.rule, self.src, dict(self.meta))

    def to_json(self) -> dict:
        d = {"pts": [[round(float(x), JSON_DECIMALS), round(float(y), JSON_DECIMALS)]
                     for x, y in self.pts],
             "closed": bool(self.closed), "layer": self.layer,
             "filled": bool(self.filled), "rule": self.rule, "src": int(self.src)}
        meta = _clean_meta(self.meta)
        if meta:
            d["meta"] = meta
        return d

    @staticmethod
    def from_json(d) -> "VPath":
        rule = d.get("rule", "nonzero")
        meta = d.get("meta")
        return VPath(_arr(d.get("pts", [])), bool(d.get("closed", False)),
                     d.get("layer", "0"), bool(d.get("filled", False)),
                     rule if rule in ("nonzero", "evenodd") else "nonzero",
                     int(d.get("src", 0)), dict(meta) if isinstance(meta, dict) else {})


class Drawing:
    """Een 2D-tekening: paden, de lagen die ze gebruiken en eventueel een plaat."""

    def __init__(self, paths=None, layers=None, sheet_w=None, sheet_h=None, warnings=None,
                 thickness=None, margin=None, gap=None, sheet_count=None):
        self.paths: list[VPath] = list(paths) if paths else []
        self.layers: dict[str, Layer] = dict(layers) if layers else {}
        self.sheet_w = None if sheet_w is None else float(sheet_w)
        self.sheet_h = None if sheet_h is None else float(sheet_h)
        self.warnings: list[str] = list(warnings) if warnings else []
        # plaatinstellingen: materiaaldikte, marge, tussenruimte en aantal platen
        self.sheet_thickness = None if thickness is None else float(thickness)
        self.sheet_margin = DEFAULT_SHEET_MARGIN if margin is None else float(margin)
        self.sheet_gap = DEFAULT_SHEET_GAP if gap is None else float(gap)
        self.sheet_count = DEFAULT_SHEET_COUNT if sheet_count is None else max(1, int(sheet_count))
        self.ensure_layer("0")

    # -- opbouwen ----------------------------------------------------------

    def add_path(self, pts, closed=False, layer="0", filled=False, rule="nonzero",
                 src=0, meta=None) -> VPath:
        """Voeg een polylijn toe; de laag wordt zo nodig aangelegd."""
        p = VPath(_arr(pts), bool(closed), layer, bool(filled),
                  rule if rule in ("nonzero", "evenodd") else "nonzero", int(src),
                  dict(meta or {}))
        self.paths.append(p)
        self.ensure_layer(layer)
        return p

    def ensure_layer(self, name, color="#000000", op=None) -> Layer:
        """De laag met `name`, bestaand of nieuw (bestaanden laten we zo).

        Een laag met de naam van een standaardlaag ("Snijden", "Boren", ...) krijgt
        diens kleur en bewerking mee, tenzij de aanroeper die zelf geeft.
        """
        lay = self.layers.get(name)
        if lay is None:
            basis = DEFAULT_BY_NAME.get(str(name))
            want = op if op in OPS else (basis[1] if basis else "cut")
            tint = color if (basis is None or (color and _norm_color(color) != "#000000")) \
                else basis[0]
            lay = Layer(name, _norm_color(tint), want, params=default_params(want))
            self.layers[name] = lay
        elif op in OPS and lay.op == "cut" and op != "cut":
            lay.op = op
        return lay

    def warn(self, text):
        if text and text not in self.warnings:
            self.warnings.append(text)
        return text

    def _sheet_kwargs(self) -> dict:
        """De plaatinstellingen als trefwoorden, zodat een kopie ze meekrijgt."""
        return {"sheet_w": self.sheet_w, "sheet_h": self.sheet_h,
                "thickness": self.sheet_thickness, "margin": self.sheet_margin,
                "gap": self.sheet_gap, "sheet_count": self.sheet_count}

    # -- lagen bewerken ------------------------------------------------------

    def free_layer_name(self, base="Laag") -> str:
        """Een laagnaam die nog niet in gebruik is: "Laag 2", "Laag 3", ..."""
        name = f"{base} 2"
        while name in self.layers:
            n = int("".join(c for c in name if c.isdigit()) or 0)
            name = f"{base} {n + 1}"
        return name

    def rename_layer(self, name, new_name) -> bool:
        """Een laag hernoemen; de paden en de kleur blijven, de volgorde ook."""
        new_name = str(new_name or "").strip()
        if not new_name or new_name == name or name not in self.layers:
            return False
        if new_name in self.layers:
            return False
        layers = {}
        for old, lay in self.layers.items():
            layers[new_name if old == name else old] = lay
        self.layers = layers
        self.layers[new_name].name = new_name
        for p in self.paths:
            if p.layer == name:
                p.layer = new_name
        return True

    def remove_layer(self, name, move_paths=True, target="0") -> bool:
        """Een laag verwijderen; de paden naar `target` zetten of mee wissen."""
        if name not in self.layers or name == target:
            return False
        if move_paths:
            self.ensure_layer(target)
            for p in self.paths:
                if p.layer == name:
                    p.layer = target
        else:
            self.paths = [p for p in self.paths if p.layer != name]
        self.layers.pop(name, None)
        self.ensure_layer(target)
        return True

    # -- meten -------------------------------------------------------------

    def bounds(self):
        """(xmin, ymin, xmax, ymax) of None als er niets ligt."""
        if not self.paths:
            return None
        lo = np.full(2, np.inf)
        hi = np.full(2, -np.inf)
        for p in self.paths:
            if len(p.pts):
                lo = np.minimum(lo, p.pts.min(axis=0))
                hi = np.maximum(hi, p.pts.max(axis=0))
        if not np.isfinite(lo).all():
            return None
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def point_count(self) -> int:
        return int(sum(len(p.pts) for p in self.paths))

    def layer_names(self):
        return list(self.layers)

    # -- omzetten ----------------------------------------------------------

    def transformed(self, M) -> "Drawing":
        """Een nieuwe tekening met de affine 3x3 matrix `M` erop."""
        M = np.asarray(M, dtype=float)
        if M.shape != (3, 3):
            raise ValueError("transformatie moet een 3x3 matrix zijn")
        out = Drawing(layers={n: replace(l, params=dict(l.params))
                              for n, l in self.layers.items()},
                      warnings=list(self.warnings), **self._sheet_kwargs())
        lin, off = M[:2, :2], M[:2, 2]
        for p in self.paths:
            q = p.pts @ lin.T + off if len(p.pts) else p.pts.copy()
            out.paths.append(VPath(np.ascontiguousarray(q, dtype=float), p.closed, p.layer,
                                   p.filled, p.rule, p.src, _transform_meta(p.meta, lin, off)))
        return out

    def copy(self) -> "Drawing":
        out = Drawing(layers={n: replace(l, params=dict(l.params))
                              for n, l in self.layers.items()},
                      warnings=list(self.warnings), **self._sheet_kwargs())
        out.paths = [VPath(p.pts.copy(), p.closed, p.layer, p.filled, p.rule, p.src,
                           dict(p.meta)) for p in self.paths]
        return out

    def merge(self, other: "Drawing") -> list[VPath]:
        """De paden van `other` hierbij zetten; lagen samenvoegen op naam.

        Bestanden die naast elkaar gelegd of in een werkblad geïmporteerd worden,
        gebruiken allebei `src`-nummers vanaf 1; die nummers gaan hier boven het
        hoogste nummer van deze tekening, zodat vormen uit verschillende bronnen
        niet aan elkaar geplakt worden. De laagkleur van een laag die er al is
        blijft zoals hij was. Geeft de nieuw toegevoegde paden terug.
        """
        for name, lay in other.layers.items():
            if name in self.layers:
                self.ensure_layer(name, lay.color, lay.op)
            else:
                self.layers[name] = replace(lay, params=dict(lay.params))
        base = max([int(p.src) for p in self.paths], default=0)
        added = []
        for p in other.paths:
            q = VPath(p.pts.copy(), p.closed, p.layer, p.filled, p.rule, int(p.src) + base,
                      dict(p.meta))
            self.paths.append(q)
            self.ensure_layer(q.layer)
            added.append(q)
        for text in other.warnings:
            self.warn(text)
        return added

    def visible_drawing(self) -> "Drawing":
        """Een kopie met alleen de paden van de zichtbare lagen (voor export)."""
        out = Drawing(layers={n: replace(l, params=dict(l.params))
                              for n, l in self.layers.items() if l.visible},
                      warnings=list(self.warnings), **self._sheet_kwargs())
        out.paths = [VPath(p.pts.copy(), p.closed, p.layer, p.filled, p.rule, p.src,
                           dict(p.meta))
                     for p in self.paths if self.layers.get(p.layer, None) is None
                     or self.layers[p.layer].visible]
        for p in out.paths:
            out.ensure_layer(p.layer)
        return out

    def to_json(self) -> dict:
        sheet = [None if self.sheet_w is None else round(float(self.sheet_w), 6),
                 None if self.sheet_h is None else round(float(self.sheet_h), 6),
                 None if self.sheet_thickness is None else round(float(self.sheet_thickness), 6),
                 round(float(self.sheet_margin), 6), round(float(self.sheet_gap), 6),
                 int(self.sheet_count)]
        return {"paths": [p.to_json() for p in self.paths],
                "layers": {n: l.to_json() for n, l in self.layers.items()},
                "sheet": sheet,
                "warnings": list(self.warnings)}

    @staticmethod
    def from_json(d) -> "Drawing":
        layers = {n: Layer.from_json(ld) for n, ld in (d.get("layers") or {}).items()}
        sheet = list(d.get("sheet") or [])
        sheet += [None] * (6 - len(sheet))          # oudere bestanden hebben minder velden
        dr = Drawing(layers=layers, sheet_w=sheet[0], sheet_h=sheet[1],
                     thickness=sheet[2], margin=sheet[3], gap=sheet[4], sheet_count=sheet[5],
                     warnings=list(d.get("warnings") or []))
        dr.paths = [VPath.from_json(pd) for pd in (d.get("paths") or [])]
        for p in dr.paths:
            dr.ensure_layer(p.layer)
        return dr

    @staticmethod
    def new_sheet() -> "Drawing":
        """Een leeg 2D-werkblad: laag "0" plus de standaardlagen met hun kleuren."""
        return Drawing(layers=default_layers())


def _norm_color(c) -> str:
    """Kleur als kleine letters "#rrggbb"; andere teksten blijven zoals ze zijn."""
    s = str(c or "#000000").strip().lower()
    if len(s) == 4 and s.startswith("#"):
        s = "#" + "".join(ch * 2 for ch in s[1:])
    return s


# --------------------------------------------------------------------------
#  krommen afvlakken
# --------------------------------------------------------------------------

def _subdivide(fn, a, b, tol, out, depth):
    """Verfijn tot de middenspunt-afwijking van de koorde onder `tol` is."""
    pa, pb = fn(a), fn(b)
    if depth <= 0:
        out.append(pb)
        return
    m = 0.5 * (a + b)
    pm = fn(m)
    d, v = pb - pa, pm - pa
    L2 = float(d @ d)
    t = 0.0 if L2 <= 1e-30 else max(0.0, min(1.0, float(v @ d) / L2))
    if math.hypot(*(v - t * d)) <= tol:
        out.append(pb)
        return
    _subdivide(fn, a, m, tol, out, depth - 1)
    _subdivide(fn, m, b, tol, out, depth - 1)


def flatten_quad(p0, p1, p2, tol=TOL) -> np.ndarray:
    """De kwadratische Bezier: punten na `p0` tot en met `p2`.

    De afwijking van de koorden tot de echte kromme blijft onder `tol`:
    |B(t) - lijn(t)| <= halve afwijking van het tussenliggende controlepunt.
    """
    p0, p1, p2 = (np.asarray(p0, float), np.asarray(p1, float), np.asarray(p2, float))
    out: list[np.ndarray] = []

    def rec(a, b, c, depth):
        pa, pc = np.asarray(a, float), np.asarray(c, float)
        mid = 0.5 * (pa + pc)
        if depth <= 0 or 0.5 * math.hypot(*(np.asarray(b, float) - mid)) <= tol:
            out.append(pc)
            return
        ab = 0.5 * (pa + b)
        bc = 0.5 * (b + pc)
        abbc = 0.5 * (ab + bc)
        rec(pa, ab, abbc, depth - 1)
        rec(abbc, bc, pc, depth - 1)

    rec(p0, p1, p2, MAX_DEPTH)
    if not out:
        out.append(p2)
    return np.asarray(out, dtype=float)


def flatten_cubic(p0, p1, p2, p3, tol=TOL) -> np.ndarray:
    """De kubieke Bezier: punten na `p0` tot en met `p3`, afwijking <= `tol`.

    |B(t) - lijn(t)| <= 3t(1-t)((1-t)|d1| + t|d2|) <= 0.75 max(|d1|, |d2|),
    met d1 en d2 de controlepunten ten opzichte van de koorde.
    """
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    out: list[np.ndarray] = []

    def rec(a, b, c, d, depth):
        pa, pd = a, d
        if depth <= 0:
            out.append(pd)
            return
        d1 = np.asarray(b, float) - (2.0 * pa + pd) / 3.0
        d2 = np.asarray(c, float) - (pa + 2.0 * pd) / 3.0
        if 0.75 * max(float(np.hypot(*d1)), float(np.hypot(*d2))) <= tol:
            out.append(pd)
            return
        ab = 0.5 * (a + b)
        bc = 0.5 * (b + c)
        cd = 0.5 * (c + d)
        abc = 0.5 * (ab + bc)
        bcd = 0.5 * (bc + cd)
        mid = 0.5 * (abc + bcd)
        rec(pa, ab, abc, mid, depth - 1)
        rec(mid, bcd, cd, pd, depth - 1)

    rec(p0, p1, p2, p3, MAX_DEPTH)
    if not out:
        out.append(p3)
    return np.asarray(out, dtype=float)


def arc_points(center, r, a0, a1, tol=TOL) -> np.ndarray:
    """Cirkelboog van `a0` naar `a1` radialen, beide einden inbegrepen.

    De koordehoogte r(1-cos(theta/2)) blijft onder `tol`.
    """
    c = np.asarray(center, float)
    r = abs(float(r))
    span = float(a1) - float(a0)
    if r <= 0.0 or span == 0.0:
        return np.array([c + r * np.array([math.cos(a0), math.sin(a0)])])
    ratio = min(1.0, max(-1.0, 1.0 - tol / r))
    step = 2.0 * math.acos(ratio) if ratio > -1.0 else math.pi
    n = 1 if step <= 0 else max(1, int(math.ceil(abs(span) / step)))
    t = np.linspace(float(a0), float(a1), n + 1)
    return c + np.column_stack([r * np.cos(t), r * np.sin(t)])


def ellipse_arc_points(center, rx, ry, a0, a1, tol=TOL, rot=0.0) -> np.ndarray:
    """Boog van een ellips, van `a0` naar `a1` radialen, beide einden inbegrepen.

    `rot` is de hoek van de ellips in radialen. Voor een gesloten ellips lopen
    eerste en laatste punt op elkaar; wie dat niet wil neemt `pts[:-1]`.
    """
    c = np.asarray(center, dtype=float)
    rx, ry = abs(float(rx)), abs(float(ry))
    a0, a1 = float(a0), float(a1)
    co, si = math.cos(float(rot)), math.sin(float(rot))
    R = np.array([[co, -si], [si, co]])

    def point(t):
        return c + R @ np.array([rx * math.cos(t), ry * math.sin(t)])

    if a1 == a0 or rx <= 0.0 or ry <= 0.0:
        return np.array([point(a0)])
    out = [point(a0)]
    _subdivide(point, a0, a1, tol, out, MAX_DEPTH)
    return np.asarray(out, dtype=float)


def svg_arc(p0, rx, ry, phi, large, sweep, p1, tol=TOL) -> np.ndarray:
    """Ellipsboog uit de SVG-eindpuntnotatie (SVG 1.1, bijlage F.6.5).

    Te kleine stralen worden opgeschaald, zoals de specificatie voorschrijft.
    Geeft de punten na `p0` tot en met `p1` terug, in het coördinatenstelsel van de
    bron (dus met Y omlaag voor SVG).
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    rx, ry = abs(float(rx)), abs(float(ry))
    if rx < 1e-12 or ry < 1e-12 or np.allclose(p0, p1, rtol=0.0, atol=1e-12):
        return np.array([p1])
    hoek = math.radians(float(phi))
    co, si = math.cos(hoek), math.sin(hoek)
    dx, dy = 0.5 * (p0[0] - p1[0]), 0.5 * (p0[1] - p1[1])
    # stap 2: coördinaten in het stelsel van de ellips (F.6.5.2)
    x1 = co * dx + si * dy
    y1 = -si * dx + co * dy
    # stap 3: stralen die te klein zijn voor dit eindpuntenpaar opschalen
    lam = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if lam > 1.0:
        k = math.sqrt(lam)
        rx, ry = rx * k, ry * k
    # stap 4: het midden van de ellips
    rx2, ry2 = rx * rx, ry * ry
    noemer = rx2 * y1 * y1 + ry2 * x1 * x1
    if noemer <= 0.0:
        return np.array([p1])
    factor = math.sqrt(max(0.0, (rx2 * ry2 - noemer) / noemer))
    f = factor if bool(large) != bool(sweep) else -factor
    cxp = f * rx * y1 / ry
    cyp = f * -ry * x1 / rx
    ctr = np.array([co * cxp - si * cyp + 0.5 * (p0[0] + p1[0]),
                    si * cxp + co * cyp + 0.5 * (p0[1] + p1[1])])

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        ln = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / ln if ln else 0.0)))
        if ux * vy - uy * vx < 0.0:
            a = -a
        return a

    ux, uy = (x1 - cxp) / rx, (y1 - cyp) / ry
    vx, vy = (-x1 - cxp) / rx, (-y1 - cyp) / ry
    t0 = angle(1.0, 0.0, ux, uy)
    dt = angle(ux, uy, vx, vy)
    if not sweep and dt > 0.0:
        dt -= 2.0 * math.pi
    if sweep and dt < 0.0:
        dt += 2.0 * math.pi
    if dt == 0.0:
        return np.array([p1])
    rot = np.array([[co, -si], [si, co]])

    def point(t):
        return ctr + rot @ np.array([rx * math.cos(t), ry * math.sin(t)])

    out: list[np.ndarray] = []
    _subdivide(point, t0, t0 + dt, tol, out, MAX_DEPTH)
    pts = np.asarray(out, dtype=float)
    pts[-1] = p1
    return pts


# --------------------------------------------------------------------------
#  vlakken
# --------------------------------------------------------------------------

def polygon_area(pts) -> float:
    """Getekende oppervlakte van een gesloten polylijn (positief met de klok tegenin)."""
    p = _arr(pts)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_poly(poly, q) -> bool:
    x, y = float(q[0]), float(q[1])
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if (yi > y) != (yj > y):
            if x < xi + (y - yi) * (xj - xi) / (yj - yi):
                inside = not inside
    return inside


def regions(paths):
    """Gesloten paden als vlakken: per vorm een lijst lussen [buitenrand, gaten].

    Paden met hetzelfde `src`-nummer horen bij één bronelement en worden samen
    geïnterpreteerd met de vulregel van dat element (nonzero of evenodd); de
    groepen worden verenigd. De buitenrand is tegen de klok in, gaten met de
    klok mee, zodat `sum(polygon_area(l) for l in vorm)` de oppervlakte is.
    Dit is precies wat `Entities.add_shape(plane, lussen)` verwacht.
    """
    groups: dict[int, list[VPath]] = {}
    for p in paths:
        if p.closed and len(p.pts) >= 3:
            groups.setdefault(p.src, []).append(p)
    sections = []
    for _src, ps in groups.items():
        loops = [p.pts for p in ps]
        rule = mf.FillRule.EvenOdd if ps[0].rule == "evenodd" else mf.FillRule.NonZero
        cs = mf.CrossSection([np.ascontiguousarray(l, float) for l in loops], rule)
        if not cs.is_empty():
            sections.append(cs)
    if not sections:
        return []
    whole = sections[0] if len(sections) == 1 else mf.CrossSection.batch_boolean(
        sections, mf.OpType.Add)
    out = []
    for comp in whole.decompose():
        polys = [np.asarray(q, dtype=float) for q in comp.to_polygons()]
        polys = [p for p in polys if len(p) >= 3 and abs(polygon_area(p)) > 1e-12]
        if not polys:
            continue
        depth = []
        for i, poly in enumerate(polys):
            q = poly[0]
            d = 0
            for j, other in enumerate(polys):
                if i != j and _point_in_poly(other, q):
                    d += 1
            depth.append(d)
        shape = []
        for i, poly in enumerate(polys):
            if depth[i] % 2 == 0:
                shape.append(poly if polygon_area(poly) > 0 else poly[::-1].copy())
            else:
                shape.append(poly if polygon_area(poly) < 0 else poly[::-1].copy())
        if shape:
            out.append(shape)
    return out


# --------------------------------------------------------------------------
#  open lijnen aaneenrijgen
# --------------------------------------------------------------------------

def next_src(paths) -> int:
    """Het eerst vrije bronnummer: paden die bij elkaar horen krijgen er één."""
    return int(max([int(p.src) for p in paths] + [0])) + 1


def path_segments(pts, closed=False):
    """De lijnstukken van een polylijn als (a, b)-paren."""
    p = _arr(pts)
    if len(p) < 2:
        return []
    segs = list(zip(p[:-1], p[1:]))
    if closed:
        segs.append((p[-1], p[0]))
    return segs


def join_open_paths(paths, tol=0.01):
    """Open polylijnen met gelijke eindpunten aaneenrijgen (eventueel omgekeerd).

    Bestanden bevatten vaak losse lijnstukken die samen een vorm geven; hier
    worden ze één pad, en als het eindpunt op het startpunt uitkomt een gesloten
    vorm. Paden worden alleen binnen dezelfde laag verbonden.
    """
    paths = list(paths)
    used = [False] * len(paths)
    grid: dict[tuple, list[tuple[int, int]]] = {}

    def cell(pt):
        return (int(round(pt[0] / tol)), int(round(pt[1] / tol)))

    def endpoint(i, end):
        p = paths[i].pts
        return p[0] if end == 0 else p[-1]

    for i, p in enumerate(paths):
        if p.closed or len(p.pts) < 2:
            used[i] = True
            continue
        for end in (0, 1):
            grid.setdefault((p.layer, cell(endpoint(i, end))), []).append((i, end))

    def take(pt, layer, skip):
        kx, ky = cell(pt)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ref in grid.get((layer, (kx + dx, ky + dy)), ()):
                    j, end = ref
                    if used[j] or j in skip:
                        continue
                    q = endpoint(j, end)
                    d = math.hypot(q[0] - pt[0], q[1] - pt[1])
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, j, end)
        if best is None:
            return None
        used[best[1]] = True
        return best[1], best[2]

    def clean(pts):
        keep = [pts[0]]
        for q in pts[1:]:
            if math.hypot(q[0] - keep[-1][0], q[1] - keep[-1][1]) > tol:
                keep.append(q)
        return keep

    out: list[VPath] = []
    for i, p in enumerate(paths):
        if p.closed or len(p.pts) < 2:
            out.append(VPath(p.pts.copy(), p.closed, p.layer, p.filled, p.rule, p.src,
                             dict(p.meta)))
            continue
        if used[i]:
            continue
        used[i] = True
        chain = {i}
        pts = [np.asarray(q, float) for q in p.pts]
        while True:
            ref = take(pts[-1], p.layer, chain)
            if ref is None:
                break
            j, end = ref
            q = [np.asarray(w, float) for w in paths[j].pts]
            pts += q if end == 0 else q[::-1]
            chain.add(j)
        while True:
            ref = take(pts[0], p.layer, chain)
            if ref is None:
                break
            j, end = ref
            q = [np.asarray(w, float) for w in paths[j].pts]
            pts = (q[::-1] if end == 0 else q) + pts
            chain.add(j)
        pts = clean(pts)
        closed = len(pts) >= 4 and math.hypot(pts[0][0] - pts[-1][0],
                                             pts[0][1] - pts[-1][1]) <= tol
        if closed:
            pts = pts[:-1]
        meta = dict(p.meta) if len(chain) == 1 else {}
        out.append(VPath(np.asarray(pts, dtype=float).reshape(-1, 2), closed, p.layer,
                         p.filled, p.rule, p.src, meta))
    return out
