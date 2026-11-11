"""Exporting the parts as STEP: solids with flat faces for other CAD programs.

Every file is read back with a small reader of the "#n=TYPE(...);" lines, and checked the
way a CAD program would: every reference leads somewhere, every edge is used once in each
direction (so the shell is closed), a flat face is one face with its holes as inner bounds,
the counts of vertices, edges and faces fit the shape, and every corner lies on the plane
of its face.

Run with:  python -m pytest tests
"""
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, fileio, ops, step  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ENTITY = re.compile(r"^#(\d+)=(.*);$")
REF = re.compile(r"#(\d+)")


def cube():
    return mf.Manifold.cube((20.0, 30.0, 10.0))


def cube_with_hole():
    return cube() - mf.Manifold.cylinder(20.0, 5.0, 5.0, 32).translate((10.0, 15.0, -5.0))


def hollow_cube():
    return cube() - mf.Manifold.cube((10.0, 10.0, 4.0)).translate((5.0, 10.0, 3.0))


class Step:
    """The entities of a STEP file: `kind[n]` the type of #n, `args[n]` the text between
    its outer brackets, `refs[n]` the numbers it refers to."""

    def __init__(self, text):
        self.text = text
        head, _, data = text.partition("DATA;")
        self.head = head
        self.kind, self.args, self.refs = {}, {}, {}
        for line in data.split("\n"):
            m = ENTITY.match(line.strip())
            if not m:
                continue
            n, body = int(m.group(1)), m.group(2)
            self.kind[n] = body[:body.index("(")].strip() or "COMPLEX"
            self.args[n] = body[body.index("(") + 1:body.rindex(")")]
            self.refs[n] = [int(r) for r in REF.findall(body)]

    def of(self, kind):
        return [n for n, k in self.kind.items() if k == kind]

    def point(self, n):
        """The coordinates of a CARTESIAN_POINT."""
        inner = self.args[n][self.args[n].index("(") + 1:self.args[n].rindex(")")]
        return np.array([float(x) for x in inner.split(",")])

    def direction(self, n):
        return self.point(n)


def write(tmp_path, solids, name="part"):
    path = tmp_path / f"{name}.step"
    faces = step.write_step(str(path), solids)
    return Step(path.read_text(encoding="ascii")), faces


def edge_use(s):
    """{EDGE_CURVE: Counter of the senses its ORIENTED_EDGEs use it with}."""
    use = {e: Counter() for e in s.of("EDGE_CURVE")}
    for oe in s.of("ORIENTED_EDGE"):
        e = s.refs[oe][0]
        use[e][s.args[oe].rsplit(",", 1)[1].strip()] += 1
    return use


# ------------------------------------------------------------------ the file

def test_the_header_names_the_ap214_schema_and_millimetres(tmp_path):
    s, _ = write(tmp_path, [("box", cube())])
    assert s.text.startswith("ISO-10303-21;") and s.text.rstrip().endswith("END-ISO-10303-21;")
    assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));" in s.head
    assert "SI_UNIT(.MILLI.,.METRE.)" in s.text
    assert s.text.isascii()


def test_every_reference_leads_to_an_entity(tmp_path):
    s, _ = write(tmp_path, [("box", cube_with_hole())])
    for n, refs in s.refs.items():
        assert all(r in s.kind for r in refs), f"#{n} refers to an entity that is not there"


def test_numbers_are_written_with_a_decimal_point():
    assert step._real(25) == "25." and step._real(-0.0) == "0." and step._real(0.5) == "0.5"
    assert step._real(1e-7) == "1.E-07" and step._real(-2.5e12) == "-2.5E+12"


def test_names_are_quoted_for_step():
    assert step._text("Bob's part") == "'Bob''s part'"
    assert step._text("Kabeldoorvoer é") == "'Kabeldoorvoer ?'"


# ------------------------------------------------------------------ the solids

def test_a_box_is_six_faces_and_a_closed_shell(tmp_path):
    s, faces = write(tmp_path, [("box", cube())])
    assert faces == 6 and len(s.of("ADVANCED_FACE")) == 6
    assert len(s.of("MANIFOLD_SOLID_BREP")) == 1 and len(s.of("CLOSED_SHELL")) == 1
    assert len(s.of("PLANE")) == 6
    use = edge_use(s)
    assert all(c == Counter({".T.": 1, ".F.": 1}) for c in use.values()), \
        "an edge is not used once in each direction: the shell is not closed"
    corners = np.array([s.point(s.refs[v][0]) for v in s.of("VERTEX_POINT")])
    assert np.allclose(corners.min(0), (0, 0, 0)) and np.allclose(corners.max(0), (20, 30, 10))


def test_a_hole_is_a_hole_in_its_face(tmp_path):
    s, faces = write(tmp_path, [("plate", cube_with_hole())])
    assert faces == 6 + 32, "the box faces and the 32 facets of the hole"
    holes = [f for f in s.of("ADVANCED_FACE")
             if any(s.kind[r] == "FACE_BOUND" for r in s.refs[f])]
    assert len(holes) == 2, "the top and the bottom have the hole as an inner bound"
    assert len(s.of("FACE_OUTER_BOUND")) == faces
    assert all(c == Counter({".T.": 1, ".F.": 1}) for c in edge_use(s).values())


@pytest.mark.parametrize("shape, genus", ((cube, 0), (cube_with_hole, 1)))
def test_vertices_edges_and_faces_fit_the_shape(tmp_path, shape, genus):
    """Euler for faces with holes: V - E + F - H = 2 - 2 * genus (H: the inner bounds)."""
    s, _ = write(tmp_path, [("x", shape())])
    v, e = len(s.of("VERTEX_POINT")), len(s.of("EDGE_CURVE"))
    f, h = len(s.of("ADVANCED_FACE")), len(s.of("FACE_BOUND"))
    assert v - e + f - h == 2 - 2 * genus


def test_every_corner_lies_on_the_plane_of_its_face(tmp_path):
    s, _ = write(tmp_path, [("x", cube_with_hole())])
    for f in s.of("ADVANCED_FACE"):
        plane = s.refs[f][-1]
        axis = s.refs[plane][0]
        origin, normal = s.point(s.refs[axis][0]), s.direction(s.refs[axis][1])
        for bound in s.refs[f][:-1]:
            loop = s.refs[bound][0]
            for oe in s.refs[loop]:
                curve = s.refs[oe][0]
                for vtx in s.refs[curve][:2]:
                    p = s.point(s.refs[vtx][0])
                    assert abs(np.dot(p - origin, normal)) < 1e-6


def test_the_faces_point_out_of_the_solid(tmp_path):
    """The outer bound runs counter-clockwise round the normal of its plane."""
    s, _ = write(tmp_path, [("box", cube())])
    for f in s.of("ADVANCED_FACE"):
        axis = s.refs[s.refs[f][-1]][0]
        normal = s.direction(s.refs[axis][1])
        loop = s.refs[s.refs[f][0]][0]
        pts = []
        for oe in s.refs[loop]:
            curve = s.refs[oe][0]
            a, b = s.refs[curve][:2]
            start = a if s.args[oe].endswith(".T.") else b
            pts.append(s.point(s.refs[start][0]))
        pts = np.array(pts)
        area = sum(np.cross(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
        assert np.dot(area, normal) > 0
        centre = pts.mean(0)
        assert np.dot(centre - np.array([10.0, 15.0, 5.0]), normal) > 0, "the face points in"


def test_two_parts_are_two_solids_with_their_names(tmp_path):
    s, _ = write(tmp_path, [("Left", cube()), ("Right", cube().translate((40.0, 0.0, 0.0)))])
    breps = s.of("MANIFOLD_SOLID_BREP")
    assert sorted(s.args[b].split(",")[0] for b in breps) == ["'Left'", "'Right'"]
    rep = s.of("ADVANCED_BREP_SHAPE_REPRESENTATION")[0]
    assert set(breps) <= set(s.refs[rep])


def test_loose_pieces_of_one_part_are_solids_of_their_own(tmp_path):
    two = cube() + cube().translate((40.0, 0.0, 0.0))
    s, _ = write(tmp_path, [("Pair", two)])
    assert sorted(s.args[b].split(",")[0] for b in s.of("MANIFOLD_SOLID_BREP")) == [
        "'Pair 1'", "'Pair 2'"]


def test_a_hollow_part_is_a_solid_with_a_void(tmp_path):
    s, _ = write(tmp_path, [("Hollow", hollow_cube())])
    assert not s.of("MANIFOLD_SOLID_BREP")
    (brep,) = s.of("BREP_WITH_VOIDS")
    (void,) = s.of("ORIENTED_CLOSED_SHELL")
    assert void in s.refs[brep] and s.args[void].endswith(".F.")
    assert len(s.of("CLOSED_SHELL")) == 2 and len(s.of("ADVANCED_FACE")) == 12


def test_a_round_part_keeps_its_facets(tmp_path):
    s, faces = write(tmp_path, [("rod", mf.Manifold.cylinder(15.0, 8.0, 8.0, 48))])
    assert faces == 48 + 2
    assert all(c == Counter({".T.": 1, ".F.": 1}) for c in edge_use(s).values())


def test_nothing_to_write_is_refused(tmp_path):
    with pytest.raises(ValueError):
        step.write_step(str(tmp_path / "x.step"), [])
    with pytest.raises(ValueError):
        fileio.export_step(str(tmp_path / "x.step"), [ops.Part("x", "#ff0000", None)])
    assert not (tmp_path / "x.step").exists()


def test_flat_faces_merges_only_what_lies_in_one_plane():
    from volura.geom import mesh_arrays
    verts, tris = mesh_arrays(cube())
    group, normals, offsets, _ = step.flat_faces(verts, tris)
    assert len(normals) == 6 and len(tris) == 12
    assert sorted(Counter(group.tolist()).values()) == [2] * 6
    assert np.allclose(np.abs(normals).sum(1), 1.0)


# ------------------------------------------------------------------ the window

class FakeBox:
    def __init__(self):
        self.asked = []

    def critical(self, *args, **kwargs):
        self.asked.append(("critical",) + args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(("information",) + args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    w = app_module.MainWindow()
    w.box = box
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def pump_until(predicate, timeout_ms=15000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def clean(text):
    return text.replace("&", "").replace("…", "").strip()


def test_the_export_menu_offers_step_after_the_obj(win):
    file_menu = next(a.menu() for a in win.menuBar().actions()
                     if a.menu() is not None and clean(a.text()) == "File")
    export = next(a.menu() for a in file_menu.actions()
                  if a.menu() is not None and clean(a.text()) == "Export")
    labels = [clean(a.text()) for a in export.actions() if not a.isSeparator()]
    assert labels.index("3D Model with Colors (OBJ)") + 1 == labels.index("CAD Model (STEP)")


def test_export_step_asks_for_a_step_file_and_writes_it(win, tmp_path, monkeypatch):
    asked = []

    class Files:
        @staticmethod
        def getSaveFileName(parent, title, suggested, selected):
            asked.append((title, suggested, selected))
            return str(tmp_path / "model.step"), selected

    monkeypatch.setattr(app_module, "QFileDialog", Files)
    ops.group_from_manifold(win.doc, cube(), "#ff0000", "Box")
    ops.group_from_manifold(win.doc, cube().translate((40.0, 0.0, 0.0)), "#0000ff", "Other")
    win.export_step()
    assert pump_until(lambda: "STEP exported" in win.hint.text()), win.box.asked
    assert asked[0][0] == "Export as STEP" and asked[0][2] == "STEP (*.step *.stp)"
    assert asked[0][1].endswith(".step")
    s = Step((tmp_path / "model.step").read_text(encoding="ascii"))
    assert len(s.of("MANIFOLD_SOLID_BREP")) == 2 and len(s.of("ADVANCED_FACE")) == 12
    assert "12 faces" in win.hint.text()
