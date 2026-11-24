"""Cookie cutters: the computational core of a part that is an outline you press.

No Qt in this file except the letters, which come from `text3d` the way the 3D text part gets
them. It gives the cutter as one closed volume, so it can be tested on its own and reused in the
model and the export.

A **cookie cutter** is an outline you press: dough, marzipan, clay or soap. The outline comes from
the worksheet - the closed shapes that are selected there - from an SVG or DXF drawing or from
letters, and it becomes a wall of `wall` millimetres that follows it, `height` millimetres high.
The outline is not the centre of that wall but its inside: the dough inside the outline is the
cookie, so it has to stay free.

The wall is the outline offset outward by its own thickness, with **round joins**. A round join
rolls a corner open instead of letting the two sides of the offset run on until they cross, so a
sharp corner of the outline stays a corner the printer can build instead of a needle of plastic
that bends away while the cutter is pressed. The **grip flange** is the outline offset by the wall
plus the width of the flange, minus the outline, so it lies outward of the wall and the wall stands
on its inner edge. The flange is at `z = 0`: the cutter is printed **flange down on the bed**, which
is what holds it while you press, and its cutting edge points up.

The **cutting edge** is the top of the wall, not the bottom - the bottom only rests on the flange.
Over its top `EDGE` millimetres the wall thins to half its thickness, which is what cuts through
dough instead of wedging it apart. The step between the two is a shelf that faces up and out, so
nothing overhangs and the cutter prints without support, edge up, on its own flange.

**Stamping** turns the lines inside the outline - the holes of the drawing and the details within
it - into ridges of half a wall, whose top stands `EDGE` millimetres under the cutting edge so the
dough is marked and not cut. Without the stamp such a line is left alone: a wall that stands in the
middle of an opening and is held by nothing is not something a cutter can carry, and a cookie that
comes out whole is honest work while one with a ridge broken off in it is not. With the stamp on, a
ridge hangs in the middle of the opening, so the flange bridges to it with spokes of its own
thickness: a straight web from the ridge to the wall, welded into both. A ridge that no spoke can
reach is left out, because a cutter whose stamp breaks off in the dough is no cutter at all.

Several outlines in one source make **several cutters**, standing side by side where they stood and
joined into one volume: `width` is the width the whole source becomes, so a set of cutters keeps
its sizes to one another instead of every outline growing to the same width. Everything is scaled
to that width, aspect and layout kept, and the result lies centred on the origin with its flange
on `z = 0`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import manifold3d as mf
import numpy as np

from . import text3d, vector_import
from .i18n import tr
from .vector2d import VPath, path_length, regions

#: mm the narrowest outline that is still a cutter and the widest that lies on a bed
WIDTH_MIN, WIDTH_MAX = 10.0, 300.0
#: mm the thinnest wall that survives being pressed and the thickest that still cuts clean
WALL_MIN, WALL_MAX = 0.6, 3.0
#: mm the shortest cutter you can get a grip on and the tallest that prints without wobbling
HEIGHT_MIN, HEIGHT_MAX = 8.0, 40.0
#: mm the flange reaches past the wall; zero leaves a cutter of nothing but its wall
FLANGE_MIN, FLANGE_MAX = 0.0, 10.0
#: mm thick the flange is: thick enough to bridge to a ridge, thin enough to press through
RIM_MIN, RIM_MAX = 1.0, 5.0
#: mm at the top of the wall that thin to half the wall, which is the cutting edge
EDGE = 2.0
#: the ridge of a stamped line is this part of the wall wide
RIDGE = 0.5
#: mm wide the spoke is that bridges a ridge to the flange
SPOKE = 2.0
#: mm between two spokes along a ridge, and the most spokes one ridge gets
SPOKE_PITCH, SPOKE_MAX = 25.0, 8
#: mm past the outline a spoke reaches, so it is welded into the flange and not only touching it
SPOKE_REACH = 0.5
#: the points of a contour a nearest search walks at most, however big the import is
SAMPLE_MAX = 400
#: mm2: below this a loop has no area left and is no outline to cut with
AREA_EPS = 1e-6
#: the straight pieces a rounded join is approximated with
SEGMENTS = 64
#: where an outline comes from: the worksheet, a drawing or letters
SOURCES = ("sheet", "file", "text")
#: the source that offers something whatever is selected on the worksheet or read from disk
DEFAULT_SOURCE = "text"
#: the cap height the letters are taken at, because the cutter scales them to its own width
TEXT_HEIGHT = 10.0


@dataclass
class Spec:
    """Where the outline comes from and how the cutter that is made of it stands.

    `source` is one of `SOURCES`: the closed shapes that are selected on the worksheet (`sheet`,
    which hands them over as the paths themselves in `sheet`), an SVG or DXF file (`file`) or
    letters (`text`, with the `family` and `bold` of 3D text). `width` is the width the whole
    source becomes in millimetres, the height and the places of the outlines scaling along.

    `wall` is the thickness of the cutting wall, `height` how high the cutter stands and `edge`
    whether its top `EDGE` millimetres thin to half the wall. The flange is `flange` millimetres
    wide - 0 leaves nothing but the wall - and `flange_thickness` millimetres thick, and it is the
    only part of the cutter that touches the bed. With `stamp` the lines inside the outlines become
    ridges that the flange bridges to.
    """
    source: str = DEFAULT_SOURCE
    text: str = "Volura"
    family: str = ""
    bold: bool = False
    file: str = ""
    sheet: tuple = ()
    width: float = 70.0
    wall: float = 0.8
    height: float = 15.0
    flange: float = 4.0
    flange_thickness: float = 2.0
    edge: bool = True
    stamp: bool = False


def _number(value, lo, hi) -> float:
    """A number of the cutter, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a cookie cutter must be numbers"))
    return max(lo, min(hi, v))


def numbers(spec: Spec) -> dict:
    """The clamped numbers of the cutter, and the source its outlines come from.

    A source that is not one of `SOURCES` is refused here, so that it says what was asked for
    instead of quietly building a cutter out of letters.
    """
    source = str(spec.source)
    if source not in SOURCES:
        raise ValueError(tr("unknown cookie cutter source {source}").format(source=source))
    return {"source": source,
            "text": str(spec.text or ""), "family": str(spec.family or ""),
            "bold": bool(spec.bold), "file": str(spec.file or "").strip(), "sheet": spec.sheet,
            "width": _number(spec.width, WIDTH_MIN, WIDTH_MAX),
            "wall": _number(spec.wall, WALL_MIN, WALL_MAX),
            "height": _number(spec.height, HEIGHT_MIN, HEIGHT_MAX),
            "flange": _number(spec.flange, FLANGE_MIN, FLANGE_MAX),
            "flange_thickness": _number(spec.flange_thickness, RIM_MIN, RIM_MAX),
            "edge": bool(spec.edge), "stamp": bool(spec.stamp)}


def sizes(spec: Spec) -> dict:
    """The figures of the cutter this source makes, so a dialog can say them out loud.

    `cutters` is how many outlines stand in the source and `size_x` by `size_y` is the place they
    take on the bed together: the outlines scaled to `width`, with the wall and the flange added on
    every side. `height` is how high the cutter stands. A source with nothing closed in it is
    refused here already, so a dialog can say that before OK is pressed; so is a file that is not
    there and text that draws nothing.
    """
    s = numbers(spec)
    made = _cutters(spec, s)
    allp = np.concatenate([outer for outer, _inner in made])
    room = s["wall"] + s["flange"]
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    return {"cutters": len(made), "size_x": float(hi[0] - lo[0]) + 2.0 * room,
            "size_y": float(hi[1] - lo[1]) + 2.0 * room, "height": s["height"]}


# ------------------------------------------------------------------ the outlines


def _sheet_paths(spec: Spec) -> list:
    """The closed shapes of the worksheet selection; open lines and points are no outlines."""
    out = [p for p in (spec.sheet or ())
           if getattr(p, "closed", True)
           and len(np.asarray(getattr(p, "pts", p), dtype=float)) >= 3]
    if not out:
        raise ValueError(tr("the cookie cutter needs a closed outline"))
    return out


def _drawing(path: str):
    """The SVG or DXF the cutter is made of, refused the way an import refuses it."""
    if not path:
        raise ValueError(tr("Pick an SVG or DXF file."))
    try:
        return vector_import.load_vector(path)
    except OSError as exc:
        raise ValueError(tr("The drawing could not be read: {error}")
                         .format(error=exc.strerror or exc)) from None


def _source_paths(spec: Spec, s: dict) -> list:
    """The closed outlines of the source, as paths, so one fill rule reads all of them alike.

    The worksheet says per shape whether its overlaps are filled by the even-odd rule or by the
    non-zero one, a drawing carries the rule of its file and letters are always the filled shapes of
    the glyphs. `vector2d.regions` reads that off the paths and gives the holes back, so a logo with
    a hole in it stays a logo with a hole in it here too. `numbers` refused anything that is not one
    of these three sources already.
    """
    if s["source"] == "sheet":
        return _sheet_paths(spec)
    if s["source"] == "file":
        drawing = _drawing(s["file"])
        closed = [p for p in drawing.paths if p.closed and len(p.pts) >= 3]
        if not regions(closed):
            raise ValueError(tr("the cookie cutter needs a closed outline"))
        return closed
    loops = text3d.plan(s["text"], s["family"] or None, TEXT_HEIGHT, s["bold"], align="centre")
    return [VPath(np.asarray(loop, dtype=float), True, rule="nonzero", src=1) for loop in loops]


def _area(loop) -> float:
    """The signed area of a closed polygon: positive the way an outline has to run."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _placed(loop, factor, mid) -> np.ndarray | None:
    """One outline of the source in the millimetres of the bed, or None when nothing is left.

    `factor` is the scale of the whole source and `mid` the middle of its bounding box, which is
    what ends up in the origin. A path that closes itself by naming its first point again is still
    one outline of the same points, and an outline that scales down to nothing has no area left to
    cut with; both are on their way out. What is left runs counter-clockwise, so which way the
    source wound it stops mattering.
    """
    pts = np.asarray(loop, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=1e-9):
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    pts = (pts - np.asarray(mid, dtype=float)) * factor
    if abs(_area(pts)) < AREA_EPS:
        return None
    return np.ascontiguousarray(pts if _area(pts) > 0 else pts[::-1].copy(), dtype=float)


def _loops(spec: Spec, s: dict) -> list:
    """Every outline of the source, scaled to the asked width and centred on the origin.

    The whole source is scaled together - the bounding box of all its outlines together becomes
    `width` millimetres wide - so several cutters keep their sizes and their places to one another.
    Outlines that lie exactly on each other are one, because two of them would make the nesting
    beneath them a hole instead of a hole.
    """
    shapes = regions(_source_paths(spec, s))
    loops = [np.asarray(loop, dtype=float) for shape in shapes for loop in shape
             if len(loop) >= 3]
    if not loops:
        raise ValueError(tr("the cookie cutter needs a closed outline"))
    allp = np.concatenate(loops)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = float(hi[0] - lo[0])
    factor = (s["width"] / span) if span > 1e-9 else 1.0
    out, seen = [], set()
    for loop in loops:
        pts = _placed(loop, factor, 0.5 * (lo + hi))
        if pts is None:
            continue
        key = tuple(map(tuple, np.round(pts, 6)))
        if key not in seen:
            seen.add(key)
            out.append(pts)
    if not out:
        raise ValueError(tr("the cookie cutter needs a closed outline"))
    return out


def _boxes(loops):
    """The bounding box of every outline as four arrays, so nesting can be rejected cheaply."""
    flat = [np.asarray(loop, dtype=float) for loop in loops]
    return (np.array([f[:, 0].min() for f in flat]), np.array([f[:, 1].min() for f in flat]),
            np.array([f[:, 0].max() for f in flat]), np.array([f[:, 1].max() for f in flat]))


def _inside(loop, p) -> bool:
    """Whether a point lies within a closed outline: the crossings of a ray to the left of it.

    The same ray rule as `geom.point_in_polys`, but over a whole outline at once: an outline off a
    drawing has tens of thousands of points and this is asked of every pair of outlines in the
    source, which the point-by-point walk of the general helper is far too slow for here.
    """
    pts = np.asarray(loop, dtype=float)
    a, b = pts, np.roll(pts, -1, axis=0)
    over = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
    if not over.any():
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a[:, 0] + (p[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(over & (p[0] < x)) % 2)


def _nesting(loops) -> tuple:
    """How the outlines of the source lie in each other, as the two lists that belong together.

    `within` names for every outline the outlines it lies in, outermost first, so their number is
    its depth: an even depth bounds material and an odd one is a hole. `lines` is that matrix
    turned round: the lines that hang within an outline, nearest level first, which is what a stamp
    makes ridges of.

    An outline can only lie within another when its bounding box fits inside that other box, which
    rejects nearly every pair of a drawing full of separate shapes before the ray is ever walked.
    """
    x0, y0, x1, y1 = _boxes(loops)
    whole = ((x0[None, :] <= x0[:, None]) & (y0[None, :] <= y0[:, None])
             & (x1[None, :] >= x1[:, None]) & (y1[None, :] >= y1[:, None]))
    np.fill_diagonal(whole, False)
    within = []
    for i, loop in enumerate(loops):
        around = [int(j) for j in np.flatnonzero(whole[i]) if _inside(loops[j], loop[0])]
        within.append(sorted(around, key=lambda j: -float(x1[j] - x0[j])))
    lines = [[] for _ in loops]
    for j, around in enumerate(within):
        for i in around:
            lines[i].append(j)
    for inside in lines:
        inside.sort(key=lambda j: len(within[j]))
    return within, lines


def _cutters(spec: Spec, s: dict) -> list:
    """The cutters of the source as (outline, the lines within it): one per outer outline.

    An outline that lies in an even number of other outlines bounds material and is cut along; one
    in an odd number of them is a hole. Every outline therefore makes a cutter of its own standing
    where it stood, and the lines within it only matter to a stamp.

    With the stamp on, an outline that hangs inside another outline is not a cutter of its own but a
    line of that cutter's stamp, so the outlines that lie in nothing are the cutters and everything
    within them is a ridge. Without it, a small shape in the middle of a big one is a small cutter
    of its own - which is how a set of nested circles is a set of cutters.
    """
    loops = _loops(spec, s)
    within, lines = _nesting(loops)
    out = []
    for i, loop in enumerate(loops):
        if len(within[i]) % 2:
            continue                                    # a hole is not an outline to cut along
        if s["stamp"] and any(len(within[j]) % 2 == 0 for j in within[i]):
            continue                                    # a stamp swallows what hangs inside it
        out.append((loop, [loops[j] for j in lines[i]] if s["stamp"] else []))
    if not out:
        raise ValueError(tr("the cookie cutter needs a closed outline"))
    return out


# ------------------------------------------------------------------ the cross-sections


def _cs(loop) -> mf.CrossSection:
    """One outline as the area it bounds."""
    return mf.CrossSection([np.ascontiguousarray(loop, dtype=float)], mf.FillRule.NonZero)


def _offset(cs: mf.CrossSection, d: float) -> mf.CrossSection:
    """The area grown or shrunk by `d` millimetres, with round joins.

    A round join rolls the corner open instead of letting the two offset sides run on until they
    cross, which is what keeps a sharp corner of the outline a corner a printer can build: plastic
    that comes to a point in its last millimetres bends or breaks off while the cutter is pressed.
    Shrinking an area past nothing leaves nothing, which is an empty cross-section and not an error.
    """
    if abs(float(d)) < 1e-9:
        return cs
    out = cs.offset(float(d), mf.JoinType.Round, circular_segments=SEGMENTS)
    return mf.CrossSection() if out.is_empty() else out


def _band(cs: mf.CrossSection, d: float) -> mf.CrossSection:
    """The band of `d` millimetres outside a closed outline: the wall of the cutter."""
    grown = _offset(cs, d)
    return mf.CrossSection() if grown.is_empty() else grown - cs


def _ring(cs: mf.CrossSection, d: float) -> mf.CrossSection:
    """The band of `d` millimetres on both sides of a closed outline: the ridge of a stamp.

    A line has no side to wear, so a ridge stands on the line and not beside it. An outline smaller
    than the ridge has nothing left of its inside to take away and becomes a filled spot, which is
    what a dot of a stamp is.
    """
    grown = _offset(cs, d)
    if grown.is_empty():
        return mf.CrossSection()
    shrunk = _offset(cs, -d)
    return grown if shrunk.is_empty() else grown - shrunk


def _strip(a, b, half: float) -> mf.CrossSection:
    """The band of `half` millimetres on both sides of the straight piece from `a` to `b`."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return mf.CrossSection()
    v = np.array([-d[1], d[0]]) / n * half
    return mf.CrossSection([np.ascontiguousarray(
        np.array([a - v, b - v, b + v, a + v], dtype=float), dtype=float)])


def _spread(loop, n: int) -> np.ndarray:
    """At most `n` points of an outline, spread evenly over it, so a big import stays quick."""
    pts = np.asarray(loop, dtype=float)
    n = max(1, min(int(n), len(pts)))
    return pts[np.unique(np.linspace(0, len(pts) - 1, n).astype(int))]


def _closest(p, loop):
    """The point of an outline nearest to `p` and how far it is, off a spread of that outline."""
    pts = _spread(loop, SAMPLE_MAX)
    d = (pts[:, 0] - float(p[0])) ** 2 + (pts[:, 1] - float(p[1])) ** 2
    i = int(np.argmin(d))
    return pts[i], float(math.sqrt(float(d[i])))


def _nearest(a, b):
    """The two points of two outlines that lie closest together, and how far apart they are."""
    a, b = _spread(a, SAMPLE_MAX), _spread(b, SAMPLE_MAX)
    d = ((a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return a[i], b[j], float(math.sqrt(float(d[i, j])))


def _apart(loop, n: int, clear) -> list:
    """`n` points of an outline, spread over it and clear of the point `clear`.

    Two spokes next to each other hold as little as one, so a point too near to one that was taken
    already is passed over; an outline that has nothing left to give simply gives fewer spokes.
    """
    room = max(SPOKE, path_length(loop) / (2.0 * (n + 1)))
    taken, clear = [], np.asarray(clear, dtype=float)
    for p in _spread(loop, 4 * n):
        if np.linalg.norm(p - clear) < room:
            continue
        if any(np.linalg.norm(p - q) < room for q in taken):
            continue
        taken.append(p)
        if len(taken) >= n:
            break
    return taken


def _bridges(loop, outline, s: dict) -> list:
    """The straight webs that hold one stamped line: from a point on it to the outline.

    The shortest way between the line and the outline is always among them, because it is the way
    least likely to lead through a corner of the outline and out of the cutter; the others lie
    spread along the line, so a long line is held in more than one place. Every one of them starts
    on the line - so it is welded into the ridge - and runs past the outline into the flange the
    cutter stands on, which is why a spoke that only reaches the outline is not enough.
    """
    n = int(max(1, min(SPOKE_MAX, round(path_length(loop) / SPOKE_PITCH))))
    reach = s["wall"] + s["flange"] + SPOKE_REACH
    p0, _q0, _d0 = _nearest(loop, outline)
    out = []
    for p in [p0] + _apart(loop, n - 1, p0):
        q, d = _closest(p, outline)
        if d < 1e-6:
            continue                                    # the line touches the outline already
        u = (np.asarray(q, dtype=float) - np.asarray(p, dtype=float)) / d
        start = np.asarray(p, dtype=float)
        out.append((start, start + u * (d + reach)))
    return out


# ------------------------------------------------------------------ the pieces


def _grow(cs: mf.CrossSection, height: float) -> mf.Manifold:
    """The cross-section pulled `height` millimetres straight up from `z = 0`."""
    if cs.is_empty() or height <= 1e-9:
        return mf.Manifold()
    return mf.Manifold.extrude(cs, float(height))


def _cutter(outline, inner, s: dict) -> list:
    """One cutter as the solids it is made of: the wall, its edge, the flange and the stamp.

    The wall is the band outside the outline, from the bed to `EDGE` under the top; the top of it is
    the cutting edge and stands at half the thickness, so the wall steps in there and the step faces
    up and out. The flange is that band widened by the width of the flange and only as thick as the
    flange is, so it lies under the wall and outward of it: the cutter stands on it and presses
    through on a ring as wide as the wall and the flange together.
    """
    o = _cs(outline)
    wall = s["wall"]
    band = _band(o, wall)
    if band.is_empty() or band.area() < AREA_EPS:
        return []
    grown = _offset(o, wall + s["flange"])
    foot = band if grown.is_empty() else grown - o
    top = s["height"] - EDGE if s["edge"] else s["height"]
    parts = [_grow(band, top)]
    if s["edge"]:
        tip = _band(o, 0.5 * wall)
        if not tip.is_empty():
            parts.append(_grow(tip, EDGE).translate((0.0, 0.0, float(top))))
    if s["flange"] > 0.0 and not foot.is_empty():
        parts.append(_grow(foot, s["flange_thickness"]))
    if s["stamp"] and inner and not foot.is_empty():
        parts += _stamp(outline, inner, grown, foot, s)
    return parts


def _stamp(outline, inner, grown, foot, s: dict) -> list:
    """The ridges of the lines within one outline, and the spokes the flange bridges to them.

    A ridge stands from the bed to `EDGE` millimetres under the cutting edge, so it marks the dough
    instead of cutting it, and it is half the wall wide, which is thin enough to show a detail and
    thick enough to be the last thing that breaks. The spoke that holds it is a web in the thickness
    of the flange, from a point on the ridge to the outline and on into the flange beyond it, cut
    back to the place the cutter takes on the bed so that no spoke sticks out beside the flange.

    Whether that holds is then asked of the drawing and not of the intent: outline, spokes and ridge
    together have to make one piece. A ridge that does not - because the outline bends away from it
    and every spoke of it leads out of the cutter - is left out. A stamp that is one detail short
    is still a stamp, and a ridge that hangs loose is a piece of plastic in the dough.
    """
    wide = 0.5 * RIDGE * s["wall"]
    high = max(0.0, s["height"] - EDGE)
    parts = []
    for loop in inner:
        ridge = _ring(_cs(loop), wide)
        if ridge.is_empty() or ridge.area() < AREA_EPS:
            continue
        webs = [w for w in (_strip(a, b, 0.5 * SPOKE) ^ grown
                            for a, b in _bridges(loop, np.asarray(outline, dtype=float), s))
                if not w.is_empty()]
        whole = foot
        for web in webs:
            whole = whole + web
        whole = whole + ridge
        if whole.is_empty() or len(whole.decompose()) > 1:
            continue                                    # nothing holds this ridge, so no ridge
        parts.append(_grow(ridge, high))
        parts += [_grow(web, s["flange_thickness"]) for web in webs]
    return parts


def _centred(m: mf.Manifold) -> mf.Manifold:
    """The cutter on `z = 0` and centred in x and y, so it lands in the middle of the bed."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return m.translate((-0.5 * (b[0] + b[3]), -0.5 * (b[1] + b[4]), -b[2]))


def cookie_cutter(spec: Spec) -> mf.Manifold:
    """One printable cutter - or several, standing as the source stands - on `z = 0`.

    The outlines of the source are scaled to `width` millimetres, the whole source together so the
    cutters keep their sizes to one another, and every outline becomes a wall of `wall` millimetres
    that follows it with round joins and stands `height` millimetres high. Its top `EDGE`
    millimetres thin to half the wall when `edge` is on: that is the cutting edge, and it points up
    because the cutter stands on its flange - `flange` millimetres wide outward of the wall and
    `flange_thickness` millimetres thick - at `z = 0`.

    With `stamp` the lines within the outlines, holes and details, become ridges of half a wall that
    stand 2 millimetres under the edge and are bridged to the flange by spokes; a ridge that no
    spoke reaches is left out rather than left loose. Everything together is one closed volume,
    centred in x and y and on its flange. A source with nothing closed in it, a file that is not
    there, text that draws nothing and anything but a number for a size are all refused.
    """
    s = numbers(spec)
    parts = []
    for outline, inner in _cutters(spec, s):
        parts += _cutter(outline, inner, s)
    if not parts:
        raise ValueError(tr("the cookie cutter needs a closed outline"))
    m = parts[0] if len(parts) == 1 else mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if m.status() != mf.Error.NoError or m.is_empty() or m.volume() <= 0.0:
        raise ValueError(tr("the outline does not make a valid cutter"))
    return _centred(m)
