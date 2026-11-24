"""Text or a drawing on a face: the computational core for letters or a logo on one face of a part.

No Qt in this file except the letters, which come from `text3d` the way the 3D text part gets
them. It gives the new geometry of the part, so it can be tested on its own and reused by the
window that shows it.

A **3D text** is its own part; this is the other way round: the letters or the filled shapes of a
drawing go on a face that is already there. A name on the side of a box, a logo in the lid of a
print, the number of a drawer on its front: the part stays one part, and what is raised or cut
follows the face it stands on. `flat` says whether that face can carry something at all, which is
also what the action in the menu asks before it lets the dialog open.

The **face** is one flat face of one group: the shape stands on its plane, in the middle of its
area. Its reading direction is the direction the text runs: over a horizontal face along the world
x, like text on the sheet does; over a wall along the longest edge of the face, so text on the
front of a box follows that box. The direction up the face always points up in the world, so the
letters are never upside down and never mirrored: what stands on the front of a box prints as the
same word, not as its reflection.

The **shape** is the text, or an SVG or DXF drawing scaled to the width that is asked for. Holes
stay holes: the counter of an o and the hole of a logo are left out of the shape, so a raised logo
stands round its own holes too.

The **color** of a raised shape is picked in the dialog and is the color of the part by default, so
a name in another filament prints as one part with two colors. Manifold remembers per triangle from
which side of the join it came, the way the 3MF export remembers the colors of a print, so the
faces of the letters wear the picked color and the faces of the part keep the color they had - a
painted face included. The wall and the floor of an engraving wear the color of the part, because
that is the material they are cut from.

The **fit** is whether the shape stands on the face: a logo half over the edge of a lid prints
hanging in the air, so the dialog says it, without forbidding it. The **depth** of an engraving is
not free: an engraving deeper than the part is thick under that face comes out at the back, so it
is refused; the part itself is asked, not the distance to the other side, because a pocket under a
rib has nothing above it to hold it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import manifold3d as mf
import numpy as np

from . import geom, ops, text3d, vector_import
from .brep import Entities, manifold_from_tris
from .i18n import tr

#: mm past the face an engraving is cut, so a face that ends in its own plane leaves no skin
#: behind, exactly as on a label plate
CUT_MARGIN = 0.5
#: mm2 of a shape outside the face that still counts as "on the face": the rounding of the booleans
FIT_TOL = 0.01
#: mm2: below this a face is too small to carry anything
FACE_MIN = 1e-6
#: mm a point of the face may stand off its plane and still be that face
FLAT_TOL = 1e-4
#: mm3 and part of it: what a boolean may miss when the depth of an engraving is checked
DEPTH_EPS, DEPTH_REL = 1e-4, 1e-6
#: how wide a drawing is scaled to: a logo of a hair is a smudge, one of a metre is a wall
WIDTH_MIN, WIDTH_MAX = 1.0, 2000.0
#: the width the dialog starts with: a lid, not a sign
WIDTH_START = 40.0
#: how high a relief stands and how deep an engraving goes, in millimetres
RELIEF_MIN, RELIEF_MAX = 0.1, 20.0
#: what can go on a face, and how
MODES = ("raised", "engraved")
SOURCES = ("text", "drawing")
#: the rotation of the dialog: a quarter turn is how text turns on a wall
ROTATION_MIN, ROTATION_MAX = -360.0, 360.0


@dataclass
class Spec:
    """What goes on the face: where the shape comes from, how it stands and how high it is.

    `mode` is one of `MODES` and `source` one of `SOURCES`. For text the `family`, `height` and
    `bold` are those of 3D text; for a drawing the `file` is an SVG or a DXF and `width` the width
    it is scaled to, the height along. `relief` is how high the shape stands above the face, or -
    for an engraving - how deep it goes in; `rotation` turns it about the normal of the face, in
    degrees, and `offset_x` with `offset_y` move it over the face along its reading direction and
    up that face. `color` is the color of the raised shape, empty for the color of the part itself;
    an engraving always wears the color of the part, because a cut surface is the material it was
    cut from.
    """
    source: str = "text"
    mode: str = "raised"
    text: str = "Volura"
    family: str = ""
    height: float = 20.0
    bold: bool = False
    file: str = ""
    width: float = WIDTH_START
    relief: float = 0.8
    rotation: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    color: str = ""


@dataclass
class Check:
    """What the dialog shows while the fields are touched: the shape on the face, and whether all
    of it still stands on the face."""
    shape: mf.Manifold
    on_face: bool = True


def _relief(spec: Spec) -> float:
    """How high the relief stands, or how deep the engraving goes, within what is a part."""
    return min(RELIEF_MAX, max(RELIEF_MIN, float(spec.relief)))


def _width(spec: Spec) -> float:
    """The width a drawing is scaled to, within what is a part."""
    return min(WIDTH_MAX, max(WIDTH_MIN, float(spec.width)))


def _loop_area(p) -> float:
    """The signed area of one closed polygon: the points that run round it, positive for one way.

    The loops of a face run in the order they are stored, so the sign tells which way the face was
    built; it is also all the shoelace formula has to say about a loop.
    """
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _loop_centroid(p, area) -> tuple[float, float]:
    """The middle of the area of one closed polygon; a loop of no area gives its middle point."""
    if abs(area) < 1e-12:
        return float(p[:, 0].mean()), float(p[:, 1].mean())
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return (float(np.dot(x + x1, cr) / (6.0 * area)),
            float(np.dot(y + y1, cr) / (6.0 * area)))


def _areas(polys) -> list[float]:
    """The area of every loop of a face: the outer boundary positive, the holes off it.

    A face does not keep its outer boundary first - `Entities.from_manifold` walks the mesh and
    takes the loops of a region in the order it meets them - so the biggest loop is the outer one:
    every hole lies inside it and is therefore smaller. The sign is what the stored winding says,
    and that is nothing to build on, so it is replaced by what the loops mean.
    """
    a = [abs(_loop_area(p)) for p in polys]
    if not a:
        return []
    big = int(np.argmax(a))
    return [a[i] if i == big else -a[i] for i in range(len(a))]


def _area(polys) -> float:
    """How much face the loops of `polys` make together: the outer one with the holes off it."""
    return max(0.0, sum(_areas(polys)))


def flat(ents, fid) -> bool:
    """True when the face `fid` is one flat face big enough to carry a shape.

    The loops are walked, not triangulated: whether something can stand on the face is asked for
    every selection, and a big imported face is not worth a triangulation for that. A face of
    another shape (a cylinder's facet that is not a plane, a face that is bent because its
    vertices drifted out of it) is refused, and so is one of no area, because its middle and its
    reading direction would be whatever the numbers say.
    """
    face = ents.F.get(fid)
    if face is None or not np.isfinite(face.normal).all():
        return False
    if np.linalg.norm(face.normal) < 1e-9:
        return False
    pts = ents.face_points(fid)
    if not pts or min(len(p) for p in pts) < 3:
        return False
    pl = ents.face_plane(fid)
    for p in pts:
        if np.abs((p - pl.origin) @ pl.n).max() > FLAT_TOL:
            return False
    return _area([pl.to_local(p) for p in pts]) > FACE_MIN


def _outer(pl, pts):
    """The loop that bounds a face from outside: the biggest of its loops, because a hole lies
    inside the boundary and is therefore smaller. A face does not store that loop first, so which
    one it is has to be asked of the loops themselves.
    """
    return max(pts, key=lambda p: abs(_loop_area(pl.to_local(np.asarray(p, dtype=float)))))


def reading(pl, pts) -> np.ndarray:
    """The direction text reads over the face, as someone looking at it upright sees it.

    A horizontal face reads along the world x, like the text of the sheet does. On a wall the
    longest edge of the face says the way, so text on the front of a box follows the box; when two
    edges are just as long the level one wins, because text hangs level. Which way along that edge
    is chosen by what is left: the direction up the face (the normal turned by a quarter) points up
    in the world, so the letters hang right way and nothing is mirrored. A face of no edges to
    follow - a full circle - takes the direction the plane already prefers.
    """
    if abs(float(pl.n[2])) > 0.99:
        u = np.array([1.0, 0.0, 0.0])
    else:
        outer = np.asarray(_outer(pl, pts), dtype=float)
        d = np.roll(outer, -1, axis=0) - outer
        d -= np.outer(d @ pl.n, pl.n)             # the edge in the plane of the face
        ln = np.linalg.norm(d, axis=1)
        keep = np.nonzero(ln > 1e-9)[0]
        if not len(keep):
            return geom.default_u(pl.n)
        level = np.abs(d @ np.array([0.0, 0.0, 1.0])) / np.maximum(ln, 1e-12)
        best = max(keep.tolist(), key=lambda i: (round(float(ln[i]), 6), float(level[i])))
        u = d[best] / ln[best]
    v = np.cross(pl.n, u)
    if v[2] < -1e-9 or (abs(v[2]) <= 1e-9 and u[2] < 0.0):
        u = -u                                    # up the face goes up, not down
    return u


def centre(ents, fid) -> np.ndarray:
    """The middle of a face: the centre of its area, so a shape lands where the face is.

    A hole is weighed off, so a shape on a face with a hole in it centres on the material that is
    left and not on the hole. Which loop is the outer one is decided by `._areas`, because the
    middle of an area may not depend on how the face was built or which way its loops run.
    """
    pts = ents.face_points(fid)
    pl = ents.face_plane(fid)
    polys = [pl.to_local(p) for p in pts]
    total, sx, sy = 0.0, 0.0, 0.0
    for p, area in zip(polys, _areas(polys)):
        if abs(area) < 1e-12:
            continue
        cx, cy = _loop_centroid(p, _loop_area(p))
        total += area
        sx += area * cx
        sy += area * cy
    if abs(total) < 1e-12:
        return np.asarray(pl.origin, dtype=float)
    return pl.to_world(np.array([sx / total, sy / total]))


def frame(ents, fid, spec: Spec) -> geom.Plane:
    """The frame the shape stands in on the face `fid`: its x along the reading direction, its y
    up the face and its z along the normal of the face, with the origin where the shape lands.

    `offset_x` and `offset_y` move over the face before `rotation` turns the shape, so a rotation
    turns the shape about the place it got - the millimetres of the dialog stay the millimetres of
    the face, and a turn does not move a shape into another corner.
    """
    pl = ents.face_plane(fid)
    u = reading(pl, ents.face_points(fid))
    v = np.cross(pl.n, u)
    o = centre(ents, fid) + float(spec.offset_x) * u + float(spec.offset_y) * v
    a = math.radians(float(np.clip(spec.rotation, ROTATION_MIN, ROTATION_MAX)))
    return geom.Plane(o, pl.n, math.cos(a) * u + math.sin(a) * v)


def face_cs(ents, fid, pl: geom.Plane) -> mf.CrossSection:
    """The face `fid` as a cross-section in the frame `pl`, for the dialog to test the fit against.

    The even-odd rule is what makes the holes of the face holes here, whichever way its loops
    happen to run: the loops of a face never cross each other, so nesting is all there is to tell
    the material from the hole.
    """
    return geom.cs_from_loops([pl.to_local(p) for p in ents.face_points(fid)], mf.FillRule.EvenOdd)


def cross_section(spec: Spec) -> mf.CrossSection:
    """The shape as it lies flat, centred on the origin of the face: the letters, or the drawing.

    Text takes the outlines of 3D text, centred as a block is centred, so the middle of the face
    also carries the middle of the word. A drawing is read the way an import reads one and scaled
    so that its widest point becomes `width` millimetres; the height and the holes of the source
    go along with it. Nothing closed in the file, a file of another
    kind and text that draws nothing are all refused, because there would be nothing to see.
    """
    if spec.source == "text":
        cs = mf.CrossSection(text3d.plan(spec.text, spec.family or None, spec.height,
                                         bool(spec.bold), align="centre"), mf.FillRule.NonZero)
    elif spec.source == "drawing":
        path = str(spec.file or "").strip()
        if not path:
            raise ValueError(tr("Pick an SVG or DXF file."))
        try:
            drawing = vector_import.load_vector(path)
        except OSError as exc:
            raise ValueError(tr("The drawing could not be read: {error}")
                             .format(error=exc.strerror or exc)) from None
        regios, _open = vector_import.prepare(drawing, target_width=_width(spec))
        cs = vector_import.cross_section(regios)
    else:
        raise ValueError(tr("unknown face text source {source}").format(source=spec.source))
    if cs.is_empty() or cs.area() <= FACE_MIN:
        raise ValueError(tr("There is nothing to put on the face."))
    return cs


def solid(cs, pl: geom.Plane, mode, relief) -> mf.Manifold:
    """The volume the shape makes on the plane `pl`: `relief` high above the face, or as deep in
    it.

    An engraving is cut `CUT_MARGIN` millimetres past the face, so a face that ends exactly in its
    own plane leaves no skin behind, like the engraving of a label plate.
    """
    r = min(RELIEF_MAX, max(RELIEF_MIN, float(relief)))
    if mode == "engraved":
        return geom.extrude_cs(cs, pl, -r, CUT_MARGIN)
    return geom.extrude_cs(cs, pl, 0.0, r)


def too_deep(part: mf.Manifold, cs, pl: geom.Plane, relief) -> bool:
    """True when an engraving of `relief` millimetres does not fit in the part under the face.

    The pocket is asked of the part itself: everything of it the part does not hold would come out
    at the back, so a pocket that is not completely filled by material is too deep. A face with
    nothing under it - a wall of one layer, a hole - is refused by the same rule, which is right:
    there is nothing to engrave there.
    """
    pocket = geom.extrude_cs(cs, pl, -min(RELIEF_MAX, max(RELIEF_MIN, float(relief))), 0.0)
    if pocket.is_empty():
        return False
    return (pocket - part).volume() > DEPTH_EPS + DEPTH_REL * pocket.volume()


def fits(face_cs_, shape_cs) -> bool:
    """True when the shape stands on the face with nothing over its edge.

    What is left of the shape once the face is taken off it is what hangs over; a hair of it is
    what the booleans round off, so `FIT_TOL` is what nobody sees in the print.
    """
    if face_cs_.is_empty():
        return False
    return (shape_cs - face_cs_).area() <= FIT_TOL


def _face_tags(g):
    """{face: the number of its color} of the part, for the triangles to carry through the boolean.

    A face without a color of its own becomes the number 0, which is no color: that is the color of
    the group, and those faces keep following the group afterwards.
    """
    return {f: ops.color_id(g.ents.F[f].color) for f in g.ents.F}


def _tagged(man: mf.Manifold, color) -> mf.Manifold:
    """The solid `man` with the number of `color` on every triangle of it.

    A boolean keeps the tag of every triangle with the triangle, so after the part and the shape
    are joined each face of the result still says what color it came from; that is how the 3MF
    export keeps the colors of a print (see `Entities.to_manifold` and `volura.ops`). No color is
    the number 0, which is what `ops.id_color` reads back as no color.
    """
    mesh = man.to_mesh64()
    # the arrays of a mesh are views that may not be written to; the new mesh wants copies
    verts = np.array(mesh.vert_properties, dtype=np.float64, order="C")
    tris = np.array(mesh.tri_verts, dtype=np.int64, order="C")
    return manifold_from_tris(verts, tris,
                              np.full(len(tris), ops.color_id(color), dtype=np.uint64))


def _shape_color(g, spec: Spec) -> str:
    """The color the raised shape wears: what was picked, or nothing when that is the part's own.

    Nothing means no tag, so those faces stay without a color of their own and follow the color of
    the group, as they do when nothing is picked. An engraving wears the color of the part always:
    the wall of a pocket is the material the pocket is cut from, whatever was picked for a relief.
    """
    if spec.mode == "engraved":
        return ""
    color = str(spec.color or "").strip().lower()
    return "" if not color or color == str(g.color or "").strip().lower() else color


def _region_color(md, tris, tags) -> str | None:
    """The color of one coplanar region of the result mesh: the color of its triangles.

    Mostly all the triangles of a region carry the same tag. A relief whose wall stands flush with
    a wall of the part makes one face of the two, and then the color of the largest part of it is
    the one to wear, because that is what the eye sees.
    """
    ids = tags[tris]
    if not ids.any():
        return None
    if ids.min() == ids.max():
        return ops.id_color(int(ids[0]))
    area = np.linalg.norm(np.cross(md.e1[tris], md.e2[tris]), axis=1)
    return ops.id_color(int(np.argmax(np.bincount(ids, weights=area))))


def _paint(ents, man: mf.Manifold):
    """The colors the triangles of `man` came back with, put on the faces of `ents`.

    `Entities.from_manifold` makes one face per coplanar region of the mesh, in the order of those
    regions, and takes the normal of a face from the first triangle of its region, so face and
    region line up when they agree face for face. That is checked and not trusted: a face with the
    color of the wrong side of the part is worse than a face that wears the color of the group, so
    when anything is off here nothing is painted at all.
    """
    tags = np.asarray(man.to_mesh64().face_id, dtype=np.int64)
    if len(tags) != man.num_tri() or not tags.any():
        return                             # nothing was tagged; the part keeps the color it has
    md = geom.MeshData(man)
    faces = list(zip(sorted(ents.F), md.region_tris))
    if len(faces) != md.region_count:
        return
    for fid, tris in faces:
        if not np.allclose(ents.F[fid].normal, md.normals[tris[0]], atol=1e-6):
            return
    for fid, tris in faces:
        color = _region_color(md, tris, tags)
        if color:
            ents.set_face_color(fid, color)


def preview(g, fid, spec: Spec) -> Check:
    """The shape on the face `fid` of `g` as it would stand, and whether all of it stands on the
    face.

    The face is read and the shape is built, which is what a dialog repeats at every click in its
    fields; the part itself is not asked for. How deep an engraving may go is a question about the
    part and not about the face, so it is not answered here - `change` refuses it, and the window
    shows that in the status bar, because a part can be big enough for that to be work.

    A shape that hangs over the edge of the face comes back with `on_face` off: a logo that is cut
    by the edge of a lid is sometimes what was meant, so that is something to say, not to forbid.
    """
    cs = cross_section(spec)
    pl = frame(g.ents, fid, spec)
    return Check(solid(cs, pl, spec.mode, spec.relief), fits(face_cs(g.ents, fid, pl), cs))


def change(g, fid, spec: Spec) -> Entities:
    """The new geometry of `g`: the shape raised on the face `fid`, or engraved into it.

    Nothing of `g` changes here - the group is read and new `Entities` come back, so this runs in a
    background task like Simplify Model does, and the window puts the result in the group as one
    undo step. Name, color and meta of the group stay with it.

    The colors stay where they were: every triangle of the part carries the color of the face it
    lies on before the boolean and Manifold carries that tag along, so a face of the part that
    still exists keeps its color - a painted face too - and the faces of a raised shape wear the
    color that was picked for them. The wall and the floor of a pocket wear the color of the part,
    because that is the material they are cut from.
    """
    cs = cross_section(spec)
    pl = frame(g.ents, fid, spec)
    part = ops.closed_manifold(g, _face_tags(g))
    if spec.mode == "engraved" and too_deep(part, cs, pl, _relief(spec)):
        raise ValueError(tr("The engraving is deeper than the part is thick under that face."))
    shape = solid(cs, pl, spec.mode, spec.relief)
    if shape.is_empty():
        raise ValueError(tr("There is nothing to put on the face."))
    shape = _tagged(shape, _shape_color(g, spec))
    res = (part + shape) if spec.mode != "engraved" else (part - shape)
    if res.status() != mf.Error.NoError:
        raise ValueError(tr("The shape does not become a solid on that face."))
    if res.is_empty() or res.volume() <= DEPTH_EPS:
        raise ValueError(tr("That would leave nothing of the part."))
    ents = Entities.from_manifold(res)
    _paint(ents, res)
    return ents
