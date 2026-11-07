"""The edges of the three worksheet generators: Box with Finger Joints, Living Hinge, Fit Test.

The generators are easy to like on the happy path: a box of 100 × 60 × 40 mm in plywood of
3 mm on an empty worksheet. These tests are about what happens around that path. A sheet of
Volura 0.1.0 carries Dutch layer names ("Snijden", "Graveren"), and a generator must then cut
into the layer that is already there instead of opening a second cut layer. A worksheet with
a sheet size sends the panels of a box through the auto arrange and one without them keeps
them side by side; neither may raise, and with a sheet size everything must stay on the
sheet. One undo must take away exactly what a generator added — the paths and the layers —
and a generator that adds nothing (impossible sizes, no room for slits) may not leave an
empty undo step behind, only the text in the status bar that says why. The living hinge wants
a rectangle that lies straight: a rotated one gets the clear message, while a closed path
that repeats its first point at the end (as imports store a rectangle) is still a rectangle.
The offsets of a fit test come from `first + i * step`, which is never exact in floating
point, so the label under a slot must still read "-0.3" and not "-0.30000000000000004". And
plywood of 0.5 mm as well as of 20 mm must give an outline that does not cross itself, which
is checked by comparing the area of the outline with the area manifold3d makes of it.

Run with:  python -m pytest tests/test_sheet_generators_edges.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import boxgen, fonts  # noqa: E402
from volura.vector2d import Drawing, next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: The default layer names of Volura 0.1.0, with their operations.
DUTCH = {"0": "cut", "Snijden": "cut", "Graveren": "engrave", "Rasteren": "raster",
         "Frezen buiten": "mill_outside", "Uitkameren": "pocket", "Boren": "drill",
         "Hulplijnen": "none"}
BOX = {"w": 100.0, "d": 60.0, "h": 40.0, "t": 3.0, "inner": True, "lid": "none",
       "kerf": 0.0, "labels": True, "mode": "sheet"}
HINGE = {"slit": 15.0, "gap": 3.0, "pitch": 2.0, "margin": 3.0, "vertical": True}
FIT = {"t": 3.0, "first": -0.2, "step": 0.1, "count": 5}
RECT = (0.0, 0.0, 60.0, 40.0)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    w.set_mode("2d")
    yield w
    w.doc.dirty = False
    w.close()


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


def old_sheet(win):
    """The worksheet of a file from Volura 0.1.0: the default layers, but in Dutch."""
    sheet = Drawing.from_json({"layers": {n: {"name": n, "color": "#ff0000", "op": op}
                                          for n, op in DUTCH.items()},
                               "paths": [], "sheet": [300.0, 200.0, 3.0, 5.0, 2.0, 1]})
    win.doc.load_sheet(sheet)
    return sheet


def cut_layers(sheet):
    return [name for name, lay in sheet.layers.items() if lay.op == "cut"]


def rectangle(win, rect=RECT, pts=None):
    """One closed rectangle on the worksheet, selected."""
    sheet = win.doc.sheet
    if pts is None:
        x0, y0, x1, y1 = rect
        pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    p = sheet.add_path(pts, closed=True, layer=sheet.default_layer("cut"),
                       src=next_src(sheet.paths))
    win.sheet_view.set_selection([p])
    return p


def outline_area(pts):
    """The area of a closed outline by shoelace."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def solid_area(pts):
    """The same outline as manifold3d fills it: equal when it does not cross itself."""
    cs = mf.CrossSection([np.ascontiguousarray(pts, dtype=float)], mf.FillRule.NonZero)
    return cs.extrude(1.0).volume()


def assert_whole_outline(pts):
    a, b = outline_area(pts), solid_area(pts)
    assert a == pytest.approx(b, rel=1e-9), \
        f"the outline crosses itself: shoelace {a:g}, filled {b:g}"


# ------------------------------------------------------- 1. Dutch layer names of an old file

def test_the_box_cuts_into_the_dutch_layer_of_an_old_sheet(win):
    sheet = old_sheet(win)
    added = win.make_box_sheet(dict(BOX))
    assert added, "nothing on the worksheet"
    assert cut_layers(sheet) == ["0", "Snijden"], "no second cut layer"
    assert all(p.layer in ("Snijden", "Graveren") for p in added)
    assert any(p.layer == "Snijden" for p in added), "the contours must be cut"


def test_the_living_hinge_cuts_into_the_dutch_layer_of_an_old_sheet(win):
    sheet = old_sheet(win)
    p = sheet.add_path([[0, 0], [60, 0], [60, 40], [0, 40]], closed=True, layer="Snijden",
                       src=next_src(sheet.paths))
    added = win.make_hinge_sheet(RECT, HINGE, src=int(p.src))
    assert added, "nothing on the worksheet"
    assert cut_layers(sheet) == ["0", "Snijden"], "no second cut layer"
    assert all(q.layer == "Snijden" for q in added)


def test_the_fit_test_cuts_into_the_dutch_layer_of_an_old_sheet(win):
    sheet = old_sheet(win)
    added = win.make_fit_test_sheet(FIT)
    assert added, "nothing on the worksheet"
    assert cut_layers(sheet) == ["0", "Snijden"], "no second cut layer"
    assert all(p.layer in ("Snijden", "Graveren") for p in added)
    assert any(p.layer == "Snijden" for p in added)


def test_the_box_on_an_old_sheet_keeps_its_layers_while_arranging(win):
    """The auto arrange route may not lose the Dutch names on its way back to the sheet."""
    sheet = old_sheet(win)                      # 300 × 200 mm, so the panels are nested
    added = win.make_box_sheet(dict(BOX))
    assert added
    assert cut_layers(sheet) == ["0", "Snijden"], "no second cut layer"
    assert all(p.layer in ("Snijden", "Graveren") for p in added)


# ------------------------------------------------------------------- 2. with and without a sheet

def test_the_box_without_a_sheet_size_stays_side_by_side(win):
    sheet = win.doc.sheet
    sheet.sheet_w = sheet.sheet_h = None
    added = win.make_box_sheet(dict(BOX))
    assert added, "a worksheet without a sheet size still gets its panels"
    assert "no sheet size" in win.hint.text()


def test_the_box_with_a_sheet_size_lies_on_the_sheet(win):
    sheet = win.doc.sheet
    sheet.sheet_w, sheet.sheet_h = 300.0, 200.0
    added = win.make_box_sheet(dict(BOX))
    pts = np.vstack([np.asarray(p.pts, dtype=float) for p in added])
    assert float(pts[:, 0].min()) >= 0.0 and float(pts[:, 1].min()) >= 0.0
    assert float(pts[:, 0].max()) <= 300.0 + 1e-9, "a panel hangs off the sheet"
    assert float(pts[:, 1].max()) <= 200.0 + 1e-9, "a panel hangs off the sheet"


def test_the_living_hinge_works_with_and_without_a_sheet_size(win):
    for size in (None, (300.0, 200.0)):
        sheet = win.doc.sheet
        sheet.sheet_w, sheet.sheet_h = (None, None) if size is None else size
        rectangle(win, (10.0, 10.0, 70.0, 50.0))
        added = win.make_hinge_sheet((10.0, 10.0, 70.0, 50.0), HINGE)
        assert added, f"no slits with the sheet size {size}"
        pts = np.vstack([np.asarray(p.pts, dtype=float) for p in added])
        assert float(pts[:, 0].min()) >= 10.0 and float(pts[:, 0].max()) <= 70.0
        assert float(pts[:, 1].min()) >= 10.0 and float(pts[:, 1].max()) <= 50.0


def test_the_fit_test_works_with_and_without_a_sheet_size(win):
    for size in (None, (300.0, 200.0)):
        sheet = win.doc.sheet
        sheet.sheet_w, sheet.sheet_h = (None, None) if size is None else size
        added = win.make_fit_test_sheet(FIT)
        assert added, f"no comb with the sheet size {size}"
        pts = np.vstack([np.asarray(p.pts, dtype=float) for p in added])
        assert float(pts[:, 0].min()) >= 0.0 and float(pts[:, 1].min()) >= 0.0
        if size is not None:
            assert float(pts[:, 0].max()) <= 300.0 + 1e-9, "the comb hangs off the sheet"
            assert float(pts[:, 1].max()) <= 200.0 + 1e-9, "the comb hangs off the sheet"


# ---------------------------------------------------------------------------- 3. undo
#
# (the tests below compare the number of paths and the layers before and after one undo,
#  and check that a generator that adds nothing leaves no undo step behind)

def undo_roundtrip(win, make):
    """One step more, one undo, and everything back as it was: paths, layers and steps."""
    sheet = win.doc.sheet
    paths, layers, steps = len(sheet.paths), list(sheet.layers), len(win.doc.undo_stack)
    added = make()
    assert added, "nothing was added"
    assert len(win.doc.undo_stack) == steps + 1, "not one undo step"
    assert len(sheet.paths) == paths + len(added)
    win.doc.undo()
    assert len(sheet.paths) == paths, "the undo left paths behind"
    assert list(sheet.layers) == layers, "the undo left layers behind"
    assert len(win.doc.undo_stack) == steps


def test_one_undo_removes_exactly_the_box(win):
    undo_roundtrip(win, lambda: win.make_box_sheet(dict(BOX)))


def test_one_undo_removes_exactly_the_box_on_a_sheet(win):
    win.doc.sheet.sheet_w, win.doc.sheet.sheet_h = 300.0, 200.0
    undo_roundtrip(win, lambda: win.make_box_sheet(dict(BOX)))


def test_one_undo_removes_exactly_the_slits(win, monkeypatch):
    accept_dialog(monkeypatch)
    rectangle(win)
    undo_roundtrip(win, lambda: win.living_hinge_dialog())


def test_one_undo_removes_exactly_the_fit_test(win):
    undo_roundtrip(win, lambda: win.make_fit_test_sheet(FIT))


def test_a_box_of_impossible_size_leaves_no_undo_step(win):
    sheet = win.doc.sheet
    paths, steps = len(sheet.paths), len(win.doc.undo_stack)
    assert win.make_box_sheet(dict(BOX, w=0.0)) is None
    assert len(sheet.paths) == paths and len(win.doc.undo_stack) == steps
    assert win.hint.text() == "Width, depth and height must be greater than zero."
    assert win.make_box_sheet(dict(BOX, inner=False, w=4.0, t=3.0)) is None
    assert len(win.doc.undo_stack) == steps
    assert win.hint.text() == "The box is smaller than twice the material thickness."


def test_a_hinge_without_room_leaves_no_undo_step(win):
    sheet = win.doc.sheet
    rectangle(win, (0.0, 0.0, 60.0, 8.0))
    paths, steps = len(sheet.paths), len(win.doc.undo_stack)
    assert win.make_hinge_sheet((0.0, 0.0, 60.0, 8.0), HINGE) is None
    assert len(sheet.paths) == paths and len(win.doc.undo_stack) == steps
    assert win.hint.text() == "There is no room left for slits in this rectangle."
    assert win.make_hinge_sheet((0.0, 0.0, 4.0, 8.0), HINGE) is None
    assert len(win.doc.undo_stack) == steps
    assert win.hint.text() == "That rectangle is too small for a living hinge."


def test_a_fit_test_without_slots_leaves_no_undo_step(win):
    sheet = win.doc.sheet
    paths, steps = len(sheet.paths), len(win.doc.undo_stack)
    assert win.make_fit_test_sheet({"t": 0.5, "first": -1.0, "step": 0.1, "count": 5}) is None
    assert len(sheet.paths) == paths and len(win.doc.undo_stack) == steps
    assert win.hint.text() == "A slot may not be zero or less wide."


# ------------------------------------------------------------------- 4. the rectangle of a hinge

def test_a_rotated_rectangle_is_not_a_rectangle_for_a_hinge(win, monkeypatch):
    """The slits run along the axes of the sheet, so a rotated rectangle gets the message."""
    def no_dialog(dlg):
        raise AssertionError("a rotated rectangle has no hinge: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    a = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    rot = np.array([[a[0], -a[1]], [a[1], a[0]]])
    corners = np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 40.0], [0.0, 40.0]]) @ rot.T
    rectangle(win, pts=corners + np.array([40.0, 10.0]))
    paths = len(win.doc.sheet.paths)
    assert win.living_hinge_dialog() is None
    assert len(win.doc.sheet.paths) == paths
    assert win.hint.text() == "Select one rectangle on the worksheet first."


def test_a_rectangle_that_repeats_its_first_point_is_still_a_rectangle(win):
    """A closed path may end on its first point (imports store rectangles that way)."""
    pts = [[0.0, 0.0], [60.0, 0.0], [60.0, 40.0], [0.0, 40.0], [0.0, 0.0]]
    rectangle(win, pts=pts)
    assert win._hinge_rectangle() == (0.0, 0.0, 60.0, 40.0)


# ---------------------------------------------------------------------- 5. the labels of a fit test

def test_the_labels_of_a_fit_test_read_cleanly(monkeypatch):
    """`first + i * step` is never exact: the text under a slot must still be short."""
    seen = []

    def loops(text, height=10.0, **kw):
        seen.append(text)
        return []

    monkeypatch.setattr(fonts, "has_fonts", lambda: True)
    monkeypatch.setattr(fonts, "text_loops", loops)
    offs = tuple(-0.3 + i * 0.1 for i in range(7))
    boxgen.fit_test(3.0, offs)
    assert seen == ["-0.3", "-0.2", "-0.1", "0", "+0.1", "+0.2", "+0.3"]


def test_no_offset_of_the_dialog_ever_gives_a_long_label(monkeypatch):
    """Every start value, step and number of slots of the dialog gives a short label."""
    seen = []

    def loops(text, height=10.0, **kw):
        seen.append(text)
        return []

    monkeypatch.setattr(fonts, "has_fonts", lambda: True)
    monkeypatch.setattr(fonts, "text_loops", loops)
    for first in (-1.0, -0.75, -0.3, -0.05, 0.0):
        for step in (0.02, 0.05, 0.1, 0.25, 0.5):
            for count in (2, 3, 7, 10):
                offs = tuple(first + i * step for i in range(count))
                if min(3.0 + o for o in offs) <= 0.0:
                    continue
                seen.clear()
                boxgen.fit_test(3.0, offs)
                for text in seen:
                    assert "e" not in text and "E" not in text, text
                    assert text != "-0", text
                    assert len(text.lstrip("+-").replace(".", "")) <= 4, text


# ----------------------------------------------------------- 6. very thin and very thick material

@pytest.mark.parametrize("t", (0.5, 20.0))
@pytest.mark.parametrize("lid", ("none", "closed"))
def test_thin_and_thick_material_give_the_box_a_whole_outline(t, lid):
    for p in boxgen.box_panels(100.0, 60.0, 40.0, t, lid=lid):
        assert_whole_outline(p.points)


@pytest.mark.parametrize("t", (0.5, 20.0))
def test_thin_and_thick_material_give_the_fit_test_a_whole_comb(t):
    offs = tuple(-0.3 + i * 0.1 for i in range(7))
    assert_whole_outline(boxgen.fit_test(t, offs).paths[0].pts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
