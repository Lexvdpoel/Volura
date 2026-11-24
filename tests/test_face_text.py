"""Text or a logo on a face: raise or engrave letters or a drawing on one flat face of a part.

`face_text.change()` reads a group and hands back new `Entities`: "raised" stands the shape
`relief` mm on the chosen face, "engraved" cuts it `relief` mm into it. The shape goes where the
face says it goes: in the middle of its area, reading along the face's own reading direction, which
is the way the text of the sheet reads over a horizontal face and the way a box reads over its
front, and never mirrored or upside down. A drawing comes from an SVG or a DXF file, scaled to the
width that is asked for, with its holes.

These tests weigh the part against what the feature promises: a raised text adds exactly the ink of
the letters times the relief and grows the part outwards only, an engraving takes the same volume
off and leaves the outside where it was, the hole of an o stays a hole, a shape that hangs over the
edge of the face is said but not forbidden, and what cannot be printed is refused: a pocket deeper
than the part is thick under the face, a shape that would leave nothing of the part, a file that is
not there and a drawing with nothing closed in it. The colors go with it: a picked color lands on
the faces of the raised shape and nowhere else, a face of the part that was painted keeps its paint
through the boolean, an engraving wears the color of the part, and a part with a relief in another
color is a part of two colors for the slicer. The place in the interface is checked too: one
flat face of one part has to be selected, the command is one undo step and keeps the name, the
color and what the part remembers, and the dialog draws its ghost over the model, says what hangs
over and remembers its values.

Run with:  python -m pytest tests/test_face_text.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QMenu)  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import face_text, fileio, fonts, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.face_text import Spec  # noqa: E402
from volura.geom import GROUND  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402
from volura.part_dialogs import ENGRAVED_COLOR, RAISED_COLOR, FaceTextDialog  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ACTION = "Text or Logo on Face…"
MODIFY = "Modify"
TEXT = dict(text="ab", height=8.0)
RED, GREEN, GREY = "#d23b3b", "#3aa655", "#cccccc"
#: the filaments the slice is asked about, red first, so the order of the answer is fixed
MATERIALS = [["Red PLA", RED], ["Grey PLA", GREY]]
#: a square 80 x 80 with a square hole 40 x 40 inside, as one even-odd path and as two DXF loops
RING_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
            'viewBox="0 0 100 100"><path fill-rule="evenodd" fill="#000" d="M10 10 H90 V90 H10 Z '
            'M30 30 H70 V70 H30 Z"/></svg>')
SQUARE_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
              'viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#000"/></svg>')
LINE_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" '
            'viewBox="0 0 10 10"><line x1="0" y1="5" x2="9" y2="5" stroke="#000"/></svg>')


def need_fonts():
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


@pytest.fixture()
def win():
    """A window that starts at the values these tests read, whatever another test left behind."""
    from volura.app import MainWindow
    w = MainWindow()
    s = w.settings
    s.face_text_source = "text"
    s.face_text_mode = "raised"
    s.face_text_file = ""
    s.face_text_width = 40.0
    s.face_text_relief = 0.8
    s.face_text_rotation = 0.0
    s.face_text_offset_x = 0.0
    s.face_text_offset_y = 0.0
    s.face_text_color = ""
    s.text3d_text = "ab"
    s.text3d_family = ""
    s.text3d_height = 8.0
    s.text3d_bold = False
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the parts and the faces

def box(doc, size=(40.0, 40.0, 10.0), name="Box", color="#cccccc"):
    """A closed box of `size` mm with its corner in the origin, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(size), color, name)


def plate_with_a_hole(doc, name="Plate"):
    """A plate of 40 x 40 x 6 mm with a round hole of 8 mm through it, left of the middle."""
    g = box(doc, (40.0, 40.0, 6.0), name)
    g.ents = Entities.from_manifold(g.ents.to_manifold() - mf.Manifold.cylinder(
        20.0, 4.0, 4.0, 40).translate((10.0, 20.0, -5.0)))
    return g


def face_of(g, n):
    """The id of the face of `g` whose outward normal points along `n`."""
    n = np.asarray(n, dtype=float)
    for fid, face in g.ents.F.items():
        if np.allclose(np.asarray(face.normal, dtype=float), n, atol=1e-6):
            return fid
    raise AssertionError(f"no face pointing along {n}")


def volume(g):
    return g.ents.to_manifold().volume()


def vertices(g):
    """The vertices of a group in a fixed order, so undo can be compared."""
    P = np.array(list(g.ents.V.values()), dtype=float)
    return P[np.lexsort(P.T)]


def ink_area(spec):
    """How much area the letters or the shapes of `spec` cover, in mm2."""
    return face_text.cross_section(spec).area()


def extents(man):
    """How far a shape reaches in x, y and z, in millimetres."""
    v = np.asarray(man.to_mesh64().vert_properties, dtype=float)[:, :3]
    return v.max(axis=0) - v.min(axis=0)


def middle(man):
    """The middle of the box around a shape."""
    v = np.asarray(man.to_mesh64().vert_properties, dtype=float)[:, :3]
    return 0.5 * (v.max(axis=0) + v.min(axis=0))


def ink_of(man):
    """The corners of a shape as (n, 3) millimetres."""
    return np.asarray(man.to_mesh64().vert_properties, dtype=float)[:, :3]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def dxf_square_with_hole():
    """The 80 x 80 square with a 40 x 40 hole in DXF: the outer loop and the hole as polylines."""
    outer = [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)]
    hole = [(30.0, 30.0), (30.0, 70.0), (70.0, 70.0), (70.0, 30.0)]

    def lwp(verts):
        out = ["0", "LWPOLYLINE", "8", "0", "90", str(len(verts)), "70", "1"]
        for x, y in verts:
            out += ["10", repr(x), "20", repr(y), "42", "0"]
        return out

    parts = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
             "0", "SECTION", "2", "ENTITIES"]
    parts += lwp(outer) + lwp(hole)
    parts += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------ the shape on a face

def test_the_text_lands_in_the_middle_of_the_face():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    pl = face_text.frame(g.ents, top, Spec(**TEXT))
    assert np.allclose(pl.origin, [20.0, 20.0, 10.0], atol=1e-9)
    assert np.allclose(pl.n, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(pl.u, [1.0, 0.0, 0.0], atol=1e-9), "text on a lid reads along x"


def test_a_relief_adds_exactly_the_ink_of_the_letters():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(**TEXT, relief=0.8)
    before = volume(g)
    g.ents = face_text.change(g, top, spec)
    assert volume(g) - before == pytest.approx(ink_area(spec) * 0.8, abs=1e-4)
    lo, hi = g.ents.bbox()
    assert hi[2] == pytest.approx(10.8), "the letters do not stand 0.8 mm above the lid"
    assert lo[2] == pytest.approx(0.0), "the part got thinner at the bottom"
    assert g.ents.to_manifold().status() == mf.Error.NoError


def test_an_engraving_takes_the_same_volume_off_and_leaves_the_outside():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(**TEXT, relief=0.6, mode="engraved")
    before = volume(g)
    g.ents = face_text.change(g, top, spec)
    assert volume(g) - before == pytest.approx(-ink_area(spec) * 0.6, abs=1e-4)
    lo, hi = g.ents.bbox()
    assert hi[2] == pytest.approx(10.0) and lo[2] == pytest.approx(0.0), "the box changed height"
    assert g.ents.to_manifold().status() == mf.Error.NoError


def test_the_hole_of_a_letter_stays_a_hole():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    ghost = face_text.preview(g, top, Spec(text="o", height=8.0, relief=0.8)).shape
    assert ghost.genus() == 1, "the o did not come out as a ring"


def test_a_shape_engraved_in_the_underside_of_a_part_stays_in_the_part():
    need_fonts()
    doc = Document()
    g = box(doc, (40.0, 40.0, 4.0))
    bottom = face_of(g, (0.0, 0.0, -1.0))
    spec = Spec(**TEXT, relief=0.5, mode="engraved")
    assert face_text.preview(g, bottom, spec).on_face
    before = volume(g)
    g.ents = face_text.change(g, bottom, spec)
    assert volume(g) - before == pytest.approx(-ink_area(spec) * 0.5, abs=1e-4)
    lo, hi = g.ents.bbox()
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(4.0), "the pocket cut the top"


def test_the_reading_direction_follows_the_wall_and_hangs_upright():
    need_fonts()
    doc = Document()
    g = box(doc, (60.0, 40.0, 10.0))
    front = face_of(g, (0.0, -1.0, 0.0))
    pl = face_text.frame(g.ents, front, Spec(**TEXT))
    assert np.allclose(pl.n, [0.0, -1.0, 0.0], atol=1e-9)
    assert pl.u @ np.array([1.0, 0.0, 0.0]) > 0.999, "text on a wall reads along its long edge"
    assert pl.v[2] > 0.999, "the letters would stand on their head"


def test_the_letters_are_neither_mirrored_upside_down_nor_tilted():
    need_fonts()
    doc = Document()
    g = box(doc)
    front = face_of(g, (0.0, -1.0, 0.0))
    ink = ink_of(face_text.preview(g, front, Spec(text="L", height=8.0, relief=0.5)).shape)
    centre = face_text.centre(g.ents, front)
    # the ink of an L sits left of the middle of its room (the stem) and under it (the foot); seen
    # from the front of the box, which looks along +y with +z up, that is -x and -z
    assert ink[:, 0].mean() < centre[0], "the L came out mirrored"
    assert ink[:, 2].mean() < centre[2], "the L hangs upside down"


def test_a_vertical_relief_grows_the_part_outwards_only():
    need_fonts()
    doc = Document()
    g = box(doc)
    front = face_of(g, (0.0, -1.0, 0.0))
    spec = Spec(**TEXT, relief=0.8)
    assert face_text.preview(g, front, spec).on_face, "this text does not fit on that wall"
    lo0, hi0 = g.ents.bbox()
    g.ents = face_text.change(g, front, spec)
    lo, hi = g.ents.bbox()
    assert lo[1] == pytest.approx(lo0[1] - 0.8), "the relief does not stand out of the wall"
    assert hi[1] == pytest.approx(hi0[1]), "the relief cut through the part"
    assert np.allclose(lo[[0, 2]], lo0[[0, 2]]) and np.allclose(hi[[0, 2]], hi0[[0, 2]])


def test_the_rotation_turns_the_shape_in_the_face():
    need_fonts()
    doc = Document()
    g = box(doc)
    front = face_of(g, (0.0, -1.0, 0.0))
    spec = Spec(**TEXT, relief=0.5)
    cs = face_text.cross_section(spec)
    straight = face_text.solid(cs, face_text.frame(g.ents, front, spec), "raised", 0.5)
    turned = face_text.solid(cs, face_text.frame(g.ents, front,
                                                 Spec(**TEXT, relief=0.5, rotation=90.0)),
                             "raised", 0.5)
    e0, e90 = extents(straight), extents(turned)
    assert e0[0] > e0[2] and e90[2] > e90[0], "the wide text did not become a tall one"
    assert e0[0] == pytest.approx(e90[2]) and e0[2] == pytest.approx(e90[0])
    assert e0[1] == pytest.approx(0.5) and e90[1] == pytest.approx(0.5), "the relief turned too"


def test_the_offsets_move_the_shape_over_the_face_and_not_through_it():
    need_fonts()
    doc = Document()
    g = box(doc)
    front = face_of(g, (0.0, -1.0, 0.0))
    spec = Spec(**TEXT, relief=0.5)
    cs = face_text.cross_section(spec)
    here = face_text.solid(cs, face_text.frame(g.ents, front, spec), "raised", 0.5)
    along = face_text.solid(cs, face_text.frame(g.ents, front,
                                                Spec(**TEXT, relief=0.5, offset_x=5.0)),
                            "raised", 0.5)
    up = face_text.solid(cs, face_text.frame(g.ents, front,
                                             Spec(**TEXT, relief=0.5, offset_y=-3.0)),
                         "raised", 0.5)
    assert middle(along) - middle(here) == pytest.approx(np.array([5.0, 0.0, 0.0]), abs=1e-6)
    assert middle(up) - middle(here) == pytest.approx(np.array([0.0, 0.0, -3.0]), abs=1e-6)


# ------------------------------------------------------------------ the colors of the shape

def mid_z(g, face):
    """How high the middle of a face stands: the height of the points its loops walk."""
    return float(np.mean([g.ents.V[v][2] for loop in face.loops for v in loop]))


def colors_in_plane(g, n, at):
    """The colors of the faces of `g` that lie in the plane through `at` and look along `n`.

    The walls of letters point the same ways as the walls of the part they stand on, so which way a
    face looks is not enough to follow a painted wall through the boolean; the plane says it.
    """
    n = np.asarray(n, dtype=float)
    at = np.asarray(at, dtype=float)
    out = []
    for f in g.ents.F.values():
        if not np.allclose(np.asarray(f.normal, dtype=float), n, atol=1e-6):
            continue
        p = np.array([g.ents.V[v] for loop in f.loops for v in loop], dtype=float)
        if np.abs((p - at) @ n).max() < 1e-6:
            out.append(f.color)
    return out


def test_a_picked_color_lands_on_the_shape_and_nowhere_else():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    g.ents = face_text.change(g, top, Spec(**TEXT, relief=0.8, color=RED))
    above = [f.color for f in g.ents.F.values() if mid_z(g, f) > 10.0]
    assert above and all(c == RED for c in above), above
    on_the_part = [f.color for f in g.ents.F.values() if mid_z(g, f) <= 10.0]
    assert on_the_part and all(c is None for c in on_the_part), \
        "a face of the part wears a color of its own now"


def test_an_engraving_wears_the_color_of_the_part_it_is_cut_in():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(**TEXT, relief=0.6, mode="engraved", color=RED)
    g.ents = face_text.change(g, top, spec)
    assert [c for c in (f.color for f in g.ents.F.values()) if c is not None] == [], \
        "a pocket is the material of the part, whatever was picked for a relief"


def test_a_painted_face_keeps_its_paint_after_a_relief_and_after_an_engraving():
    need_fonts()
    for mode in ("raised", "engraved"):
        doc = Document()
        g = box(doc)
        g.ents.set_face_color(face_of(g, (0.0, -1.0, 0.0)), GREEN)
        top = face_of(g, (0.0, 0.0, 1.0))
        g.ents = face_text.change(g, top, Spec(**TEXT, relief=0.8, mode=mode, color=RED))
        assert colors_in_plane(g, (0.0, -1.0, 0.0), (20.0, 0.0, 5.0)) == [GREEN], mode
        other = [c for c in (f.color for f in g.ents.F.values()) if c not in (None, GREEN)]
        if mode == "engraved":
            assert other == [], "an engraving painted a face of its own"
        else:
            assert set(other) == {RED} and other, other


def test_a_relief_that_stands_flush_with_a_wall_of_the_part_leaves_that_wall_alone(tmp_path):
    """A ring as wide as the lid makes one face of its wall and the wall of the part.

    Manifold says which triangles came from where, but the two are coplanar and stand in one face,
    and a face wears one color: the largest part of it decides, which is the wall of the part.
    """
    path = write(tmp_path, "ring.svg", RING_SVG)
    doc = Document()
    g = box(doc)
    g.ents.set_face_color(face_of(g, (0.0, -1.0, 0.0)), GREEN)
    top = face_of(g, (0.0, 0.0, 1.0))
    g.ents = face_text.change(g, top, Spec(source="drawing", file=path, width=40.0, relief=0.8,
                                           color=RED))
    assert colors_in_plane(g, (0.0, -1.0, 0.0), (20.0, 0.0, 5.0)) == [GREEN]
    assert colors_in_plane(g, (0.0, 1.0, 0.0), (20.0, 40.0, 5.0)) == [None]
    tops = [f.color for f in g.ents.F.values() if mid_z(g, f) > 10.0]
    assert tops and all(c == RED for c in tops), tops


def test_the_other_part_of_the_same_group_keeps_its_colors():
    """Two shells in one group: what stands on one of them leaves the other as it was."""
    need_fonts()
    doc = Document()
    two = (mf.Manifold.cube((40.0, 40.0, 10.0))
           + mf.Manifold.cube((20.0, 20.0, 10.0)).translate((0.0, 50.0, 0.0)))
    g = ops.group_from_manifold(doc, two, GREY, "Two")
    lids = {f: np.mean(np.array([g.ents.V[v] for v in fa.loops[0]], dtype=float), axis=0)
            for f, fa in g.ents.F.items() if np.allclose(fa.normal, (0.0, 0.0, 1.0))}
    near = [f for f, c in lids.items() if c[1] <= 40.0]
    for f in [f for f, c in lids.items() if c[1] > 40.0]:
        g.ents.set_face_color(f, GREEN)
    before = g.ents.to_manifold().volume()
    spec = Spec(**TEXT, relief=0.8, color=RED)
    g.ents = face_text.change(g, near[0], spec)
    lid_plane = colors_in_plane(g, (0.0, 0.0, 1.0), (10.0, 60.0, 10.0))
    assert GREEN in lid_plane, "the shell that was left alone lost its paint"
    assert RED not in lid_plane, lid_plane
    assert sum(1 for f in g.ents.F.values() if f.color == RED) > 0
    assert g.ents.to_manifold().volume() - before == pytest.approx(ink_area(spec) * 0.8, abs=1e-4)


def test_a_shape_in_the_color_of_the_part_wears_no_color_of_its_own():
    need_fonts()
    doc = Document()
    g = box(doc, color="#CCCCCC")           # the same color, written another way
    top = face_of(g, (0.0, 0.0, 1.0))
    g.ents = face_text.change(g, top, Spec(**TEXT, relief=0.8, color=GREY))
    assert all(f.color is None for f in g.ents.F.values()), \
        "a shape in the color of the part pins that color on its faces"


def test_a_part_without_a_color_of_its_own_still_gives_the_shape_the_picked_one():
    need_fonts()
    doc = Document()
    g = box(doc)
    g.color = None
    top = face_of(g, (0.0, 0.0, 1.0))
    g.ents = face_text.change(g, top, Spec(**TEXT, relief=0.8, color=RED))
    assert any(f.color == RED for f in g.ents.F.values()), "the picked color was thrown away"


def test_a_raised_text_in_another_color_is_a_part_of_two_colors():
    """What the model shows as two colors is what the slicer gets two filaments for."""
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    g.ents = face_text.change(g, top, Spec(**TEXT, relief=0.8, color=RED))
    colors = [color for _n, color, _e in fileio.model_colors(ops.parts(doc), MATERIALS)]
    assert colors == [RED, GREY], colors


# ------------------------------------------------------------------ the drawings

def test_a_drawing_is_scaled_to_the_width_and_keeps_its_hole(tmp_path):
    path = write(tmp_path, "ring.svg", RING_SVG)
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(source="drawing", file=path, width=20.0, relief=0.8)
    shape = face_text.preview(g, top, spec).shape
    assert extents(shape)[0] == pytest.approx(20.0), "the logo did not become 20 mm wide"
    assert extents(shape)[1] == pytest.approx(20.0), "the height did not scale along"
    assert shape.genus() == 1, "the hole of the logo filled up"


def test_a_dxf_drawing_works_too(tmp_path):
    path = write(tmp_path, "ring.dxf", dxf_square_with_hole())
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(source="drawing", file=path, width=20.0, relief=0.8)
    before = volume(g)
    g.ents = face_text.change(g, top, spec)
    assert volume(g) - before == pytest.approx((20.0 * 20.0 - 10.0 * 10.0) * 0.8, abs=1e-3)
    assert face_text.preview(g, top, spec).shape.genus() == 1


def test_the_fit_says_what_hangs_over_the_edge_without_forbidding_it(tmp_path):
    path = write(tmp_path, "ring.svg", RING_SVG)
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    small = Spec(source="drawing", file=path, width=20.0, relief=0.5)
    big = Spec(source="drawing", file=path, width=58.0, relief=0.5)
    assert face_text.preview(g, top, small).on_face
    assert not face_text.preview(g, top, big).on_face, "a 58 mm logo on a 40 mm face fits?"
    before = volume(g)
    g.ents = face_text.change(g, top, big)          # said, not forbidden
    assert volume(g) > before


# ------------------------------------------------------------------ what is refused

def test_an_engraving_deeper_than_the_part_is_thick_is_refused():
    need_fonts()
    doc = Document()
    g = box(doc, (40.0, 40.0, 1.5))
    top = face_of(g, (0.0, 0.0, 1.0))
    before = volume(g)
    with pytest.raises(ValueError) as exc:
        face_text.change(g, top, Spec(**TEXT, relief=2.0, mode="engraved"))
    assert "deeper" in str(exc.value)
    assert volume(g) == pytest.approx(before), "the part was cut anyway"


def test_a_pocket_over_a_hole_through_the_part_has_nothing_under_it(tmp_path):
    path = write(tmp_path, "square.svg", SQUARE_SVG)
    doc = Document()
    g = plate_with_a_hole(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    assert len(g.ents.F[top].loops) == 2, "the hole did not come through the lid"
    # the hole is 10 mm left of the middle of the lid, so a 12 mm logo there is cut in air
    spec = Spec(source="drawing", file=path, width=12.0, relief=1.0, mode="engraved",
                offset_x=-10.0)
    assert not face_text.preview(g, top, spec).on_face, "the logo stands over the hole"
    with pytest.raises(ValueError) as exc:
        face_text.change(g, top, spec)
    assert "deeper" in str(exc.value)


def test_an_engraving_of_the_whole_top_leaves_nothing_of_the_part(tmp_path):
    path = write(tmp_path, "square.svg", SQUARE_SVG)
    doc = Document()
    g = box(doc, (20.0, 20.0, 10.0))
    top = face_of(g, (0.0, 0.0, 1.0))
    spec = Spec(source="drawing", file=path, width=20.0, relief=10.0, mode="engraved")
    before = volume(g)
    with pytest.raises(ValueError) as exc:
        face_text.change(g, top, spec)
    assert "nothing" in str(exc.value)
    assert volume(g) == pytest.approx(before)


def test_a_face_with_a_hole_is_flat_and_centres_on_the_material():
    doc = Document()
    g = plate_with_a_hole(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    assert face_text.flat(g.ents, top)
    centre = face_text.centre(g.ents, top)
    assert centre[0] > 20.0, "the middle of the lid did not move away from the hole"
    assert centre[1] == pytest.approx(20.0, abs=1e-6)


def test_a_face_that_is_not_a_plane_and_a_face_that_is_gone_are_not_faces():
    doc = Document()
    g = box(doc, (30.0, 30.0, 6.0))
    top = face_of(g, (0.0, 0.0, 1.0))
    assert face_text.flat(g.ents, top)
    v = g.ents.F[top].loops[0][0]
    g.ents.V[v] = g.ents.V[v] + np.array([0.0, 0.0, 1.0])
    assert not face_text.flat(g.ents, top), "a face that is bent is nothing to stand on"
    assert not face_text.flat(g.ents, 10 ** 6), "a face that is not there is not a face either"


def test_the_source_that_draws_nothing_is_refused(tmp_path):
    need_fonts()
    missing = str(tmp_path / "gone.svg")
    other = write(tmp_path, "notes.txt", "not a drawing")
    open_path = write(tmp_path, "line.svg", LINE_SVG)
    for spec, want in ((Spec(text="   "), "text"),
                       (Spec(source="drawing", file=""), "svg"),
                       (Spec(source="drawing", file=other), "vector"),
                       (Spec(source="drawing", file=missing), "open"),
                       (Spec(source="drawing", file=open_path), "nothing"),
                       (Spec(source="strange"), "strange")):
        with pytest.raises(ValueError) as exc:
            face_text.cross_section(spec)
        assert want.lower() in str(exc.value).lower(), f"{want}: {exc.value}"


def test_the_values_of_a_hand_written_settings_file_stay_within_bounds():
    need_fonts()
    doc = Document()
    g = box(doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    cs = face_text.cross_section(Spec(**TEXT))
    pl = face_text.frame(g.ents, top, Spec(**TEXT))
    assert face_text.solid(cs, pl, "raised", 0.0).volume() == pytest.approx(
        cs.area() * face_text.RELIEF_MIN)
    assert face_text.solid(cs, pl, "raised", 10 ** 6).volume() == pytest.approx(
        cs.area() * face_text.RELIEF_MAX)
    assert face_text.solid(cs, pl, "engraved", -5.0).volume() == pytest.approx(
        cs.area() * (face_text.RELIEF_MIN + face_text.CUT_MARGIN))


# ------------------------------------------------------------------ the command in the window

def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def clean(text):
    return text.replace("&", "").replace("…", "")


def menu_labels(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


def submenu(win, title):
    """The menu of the window, however deep, that carries this title."""
    for menu in win.menuBar().findChildren(QMenu):
        if clean(menu.title()) == title:
            return menu
    raise AssertionError(f"no menu named {title}")


def select_face(win, g, fid):
    win.doc.set_selection([("f", g.id, fid)])
    win.doc.notify()


def test_the_command_wants_one_flat_face_of_one_part(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    win.doc.selection = []
    assert win.face_text_dialog() is None
    assert "flat face" in win.hint.text()
    win.doc.set_selection([("g", g.id, g.id)])
    assert win.face_text_dialog() is None, "a whole group is not a face"
    other = box(win.doc, (20.0, 20.0, 5.0), "Other")
    select_face(win, g, top)
    win.doc.selection.append(("f", other.id, face_of(other, (0.0, 0.0, 1.0))))
    assert win.face_text_dialog() is None, "two faces do not say which one was meant"
    win.doc.root.add_shape(GROUND, [np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0],
                                             [0.0, 30.0]])])
    loose = max(win.doc.root.F)
    win.doc.set_selection([("f", ROOT, loose)])
    assert win.face_text_dialog() is None, "loose geometry has no part to carry the shape"


def test_a_locked_part_is_left_alone(win):
    need_fonts()
    g = box(win.doc)
    g.locked = True
    before = volume(g)
    select_face(win, g, face_of(g, (0.0, 0.0, 1.0)))
    assert win.face_text_dialog() is None
    assert "locked" in win.hint.text().lower()
    assert volume(g) == pytest.approx(before)


def test_raising_a_text_on_a_face_is_one_undo_step_that_keeps_the_part(win):
    need_fonts()
    g = box(win.doc, name="Lid", color="#3aa655")
    g.meta["note"] = "kept"
    top = face_of(g, (0.0, 0.0, 1.0))
    select_face(win, g, top)
    before, old = volume(g), vertices(g)
    spec = Spec(**TEXT, relief=0.8)
    assert win._apply_face_text(g, top, spec) is g
    assert pump_until(lambda: volume(g) > before), "nothing appeared on the lid"
    assert volume(g) - before == pytest.approx(ink_area(spec) * 0.8, abs=1e-4)
    assert (g.name, g.color, g.meta["note"]) == ("Lid", "#3aa655", "kept")
    assert win.doc.selection == [("g", g.id, g.id)], "the part is not selected again"
    assert "raised on Lid" in win.hint.text(), win.hint.text()
    win.undo()
    assert volume(g) == pytest.approx(before)
    assert np.allclose(vertices(g), old), "one undo did not bring the lid back as it was"


def test_an_engraving_that_is_too_deep_leaves_the_part_as_it_was(win):
    need_fonts()
    g = box(win.doc, (40.0, 40.0, 1.5))
    top = face_of(g, (0.0, 0.0, 1.0))
    select_face(win, g, top)
    before, old = volume(g), vertices(g)
    win._apply_face_text(g, top, Spec(**TEXT, relief=2.0, mode="engraved"))
    assert pump_until(lambda: "deeper" in win.hint.text()), win.hint.text()
    assert volume(g) == pytest.approx(before)
    assert np.allclose(vertices(g), old)
    assert not win.doc.undo_stack, "a refusal should not cost an undo step"


def test_the_action_is_under_tools_modify_and_only_for_the_3d_model(win):
    modify = submenu(win, MODIFY)
    assert any(a.menu() is modify for a in submenu(win, "Tools").actions()), \
        "Modify does not sit under Tools"
    assert ACTION in menu_labels(modify), menu_labels(modify)
    assert win._menu_modes[id(modify.menuAction())][1] == "3d", \
        "engraving a part is no worksheet work"
    assert "flat face" in win.a_face_text.toolTip()


# ------------------------------------------------------------------ the dialog

def test_the_dialog_draws_its_ghost_over_the_model_and_takes_it_away(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    select_face(win, g, top)
    dlg = FaceTextDialog(win, g, top)
    assert win.view.overlay_solid is not None, "the model shows nothing of what will be made"
    assert win.view.overlay_solid[1] == RAISED_COLOR
    dlg.mode.setCurrentIndex(dlg.mode.findData("engraved"))
    assert win.view.overlay_solid[1] == ENGRAVED_COLOR
    assert win.view.overlay_solid[0].bbox[0][2] == pytest.approx(10.0 - win.settings.face_text_relief)
    assert win.view.overlay_solid[0].bbox[1][2] == pytest.approx(
        10.0 + face_text.CUT_MARGIN), "a pocket may only reach past the face by the cut margin"
    dlg.close()
    assert win.view.overlay_solid is None, "the ghost stayed after the dialog"


def test_the_dialog_shows_only_the_fields_of_the_source_that_is_chosen(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    assert not dlg.text.isHidden() and dlg.drawing_row.isHidden(), "the fields of both sources are on"
    dlg.source.setCurrentIndex(dlg.source.findData("drawing"))
    assert not dlg.drawing_row.isHidden() and dlg.text.isHidden()
    dlg.close()


def test_the_dialog_says_what_hangs_over_the_edge_of_the_face(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    ok = dlg.bb.button(QDialogButtonBox.Ok)
    assert dlg.info.text() == "" and ok.isEnabled()
    dlg.height.setValue(30.0)
    assert "edge" in dlg.info.text(), dlg.info.text()
    assert ok.isEnabled(), "hanging over is nothing to wait for"
    dlg.close()


def test_the_dialog_refuses_a_drawing_that_is_not_there(win):
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    win.settings.face_text_source = "drawing"
    win.settings.face_text_file = ""
    dlg = FaceTextDialog(win, g, top)
    assert dlg.info.text()
    assert not dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
    assert win.view.overlay_solid is None
    dlg.close()


def test_the_field_of_the_relief_is_called_depth_for_an_engraving(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    assert clean(dlg.layout().labelForField(dlg.relief).text()) == "Relief"
    dlg.mode.setCurrentIndex(dlg.mode.findData("engraved"))
    assert clean(dlg.layout().labelForField(dlg.relief).text()) == "Depth"
    dlg.close()


def test_the_color_of_the_shape_starts_at_the_color_of_the_part(win):
    need_fonts()
    g = box(win.doc, color=GREY)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    assert dlg.spec().color == GREY, "the shape starts out wearing the color of the part"
    assert dlg.color.text() == GREY and not dlg.color.icon().isNull()
    dlg.close()


def test_a_color_that_is_not_a_color_in_the_settings_is_left_alone(win):
    need_fonts()
    win.settings.face_text_color = "not a color"
    g = box(win.doc, color=GREEN)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    assert dlg.spec().color == GREEN, "a broken setting should not hide the color of the part"
    dlg.close()


def test_the_color_field_is_only_there_for_a_relief(win):
    need_fonts()
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    dlg = FaceTextDialog(win, g, top)
    assert not dlg.color.isHidden()
    dlg.mode.setCurrentIndex(dlg.mode.findData("engraved"))
    assert dlg.color.isHidden(), "a pocket wears the color of the part, so nothing to pick"
    dlg.close()


def test_the_picked_color_of_the_shape_is_remembered(win, monkeypatch):
    need_fonts()
    g = box(win.doc, color=GREY)
    top = face_of(g, (0.0, 0.0, 1.0))
    select_face(win, g, top)
    seen = {}

    def accept(dlg):
        seen["start"] = dlg.spec().color
        dlg.pick_color(RED)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    assert win.face_text_dialog() is g
    assert seen["start"] == GREY
    assert win.settings.face_text_color == RED
    assert pump_until(lambda: any(f.color == RED for f in g.ents.F.values())), \
        "OK did not give the part the color that was picked"
    other = box(win.doc, color=GREEN, name="Other")
    dlg = FaceTextDialog(win, other, face_of(other, (0.0, 0.0, 1.0)))
    assert dlg.spec().color == RED, "the picked color is not remembered for the next part"
    dlg.close()


def test_the_dialog_starts_at_the_remembered_values_and_remembers_the_new_ones(win, tmp_path,
                                                                              monkeypatch):
    need_fonts()
    path = write(tmp_path, "ring.svg", RING_SVG)
    s = win.settings
    s.face_text_source = "drawing"
    s.face_text_mode = "engraved"
    s.face_text_file = path
    s.face_text_width = 25.0
    s.face_text_relief = 1.2
    s.face_text_rotation = 30.0
    s.face_text_offset_x = 2.0
    s.face_text_offset_y = -1.0
    s.text3d_text = "Hi"
    s.text3d_height = 12.0
    s.text3d_bold = True
    g = box(win.doc)
    top = face_of(g, (0.0, 0.0, 1.0))
    select_face(win, g, top)
    seen = {}

    def accept(dlg):
        seen["start"] = (str(dlg.source.currentData()), str(dlg.mode.currentData()),
                         dlg.file.text(), dlg.width.value(), dlg.relief.value(),
                         dlg.rotation.value(), dlg.offset_x.value(), dlg.offset_y.value(),
                         dlg.text.toPlainText(), dlg.height.value(), dlg.bold.isChecked())
        dlg.source.setCurrentIndex(dlg.source.findData("text"))
        dlg.mode.setCurrentIndex(dlg.mode.findData("raised"))
        dlg.relief.setValue(0.4)
        dlg.rotation.setValue(-90.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    before = volume(g)
    assert win.face_text_dialog() is g
    assert seen["start"] == ("drawing", "engraved", path, pytest.approx(25.0),
                             pytest.approx(1.2), pytest.approx(30.0), pytest.approx(2.0),
                             pytest.approx(-1.0), "Hi", pytest.approx(12.0), True)
    assert (s.face_text_source, s.face_text_mode) == ("text", "raised")
    assert s.face_text_relief == pytest.approx(0.4)
    assert s.face_text_rotation == pytest.approx(-90.0)
    assert s.face_text_file == path, "the drawing that stood there is gone from the settings"
    assert s.face_text_color == "", "the color of the part itself is nothing to remember"
    assert s.text3d_text == "Hi" and s.text3d_bold is True
    assert pump_until(lambda: volume(g) > before), "OK did not put the shape on the part"
    assert "raised on Box" in win.hint.text(), win.hint.text()
