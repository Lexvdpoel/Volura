"""Braille labels: a label of raised dots that is read with a fingertip.

`braille.cells` gives the dots of every cell of a text: uncontracted Braille with the capital,
number and letter signs a reader needs. `braille.braille_label` stands those cells as dots of 1.5
mm across and 0.6 mm high on a plate that stands on `z = 0` and is centred on the origin, and puts
the same text in letters above the dots when asked. The sizes are the Marburg Medium ones: dots
2.5 mm apart in a cell, cells 6 mm apart from left dot column to left dot column and lines 10 mm
apart, and the lines are left aligned, because a reader's hand starts at the left edge of a line.

These tests read a label the way a reader and a printer do: the cells of the alphabet and of the
signs, one piece of plastic for every dot, the distances between them in millimetres, a dot that
is a cap of 1.5 mm wide and 0.6 mm high, no face that leans down (so no support), and a plate that
grows for the letters, the margin and the hole. The place in the interface is checked too (Parts
after the bag clip, one group in one undo step, and the dialog that remembers its values and
refuses a character that has no cell).

Run with:  python -m pytest tests/test_braille.py
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
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,  # noqa: E402
                               QPlainTextEdit)

import volura.app as app_module  # noqa: E402
from volura import braille, enclosure, fonts  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TEXT = "Volura"
T, M, RAD = 2.0, 4.0, 2.0
OPTIONS = dict(text="Braille label", margin=M, thickness=T, radius=RAD, hole=0.0, letters=True)
#: the label the geometry tests measure: dots only, so a dot is the only thing above the plate
BASE = dict(OPTIONS, text=TEXT, letters=False)
#: mm: a distance a fingertip reads may not be off by more than a hair
ABS = 0.01
#: mm: the dot is a 48-piece sphere cut in a box, so its edge is a polygon, not a circle
DOT_ABS = 0.05
#: mm: the widest a dot is, so the letters above the Braille are told apart from the dots
DOT_WIDE = 1.6


def need_fonts():
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def part(**kw):
    return braille.braille_label(**dict(BASE, **kw))


def plate_prism(m):
    """The plate of `m` without its dots: the same rounded rectangle, `T` mm thick."""
    lo, hi = bbox(m)
    return mf.Manifold.extrude(enclosure.rounded_rect(hi[0] - lo[0], hi[1] - lo[1], RAD), T)


def islands(m):
    """What stands above the plate, every piece on its own, as (x0, y0, x1, y1, top)."""
    out = []
    for piece in (m - plate_prism(m)).decompose():
        b = np.asarray(piece.bounding_box(), dtype=float)
        out.append((b[0], b[1], b[3], b[4], b[5]))
    return out


def dots(m):
    """The Braille dots above the plate: the pieces that are no wider than a dot."""
    return [p for p in islands(m) if max(p[2] - p[0], p[3] - p[1]) <= DOT_WIDE]


def letters(m):
    """The letters above the plate: the pieces that carry the text for people who read by eye."""
    return [p for p in islands(m) if max(p[2] - p[0], p[3] - p[1]) > DOT_WIDE]


def centres(points):
    """The middle of every box, with the leftmost one on x = 0 and the lowest one on y = 0."""
    pts = [(0.5 * (p[0] + p[2]), 0.5 * (p[1] + p[3])) for p in points]
    left, low = min(p[0] for p in pts), min(p[1] for p in pts)
    return sorted((x - left, y - low) for x, y in pts)


def corner(points):
    """The same points, moved so that the leftmost stands on x = 0 and the lowest on y = 0."""
    left, low = min(p[0] for p in points), min(p[1] for p in points)
    return sorted((x - left, y - low) for x, y in points)


def facets(m):
    """The unit normals, the centroids and the areas of the triangles of a manifold."""
    mesh = m.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-9
    return n[ok] / length[ok][:, None], (a + b + c)[ok] / 3.0, length[ok] / 2.0


def faces_that_lean_down(m):
    """The faces that look down while they stand above the bed: those need support."""
    n, centre, area = facets(m)
    return int(((n[:, 2] < -1e-6) & (centre[:, 2] > 1e-6) & (area > 1e-6)).sum())


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the cells

@pytest.mark.parametrize("text, want", (
    ("a", [(1,)]),
    ("Ab", [(6,), (1,), (1, 2)]),
    ("12", [(3, 4, 5, 6), (1,), (1, 2)]),
    ("1a", [(3, 4, 5, 6), (1,), (5, 6), (1,)]),
    ("a b", [(1,), (), (1, 2)]),
    ("k", [(1, 3)]),
    ("w", [(2, 4, 5, 6)]),
    ("10", [(3, 4, 5, 6), (1,), (2, 4, 5)]),
    ("2a1", [(3, 4, 5, 6), (1, 2), (5, 6), (1,), (3, 4, 5, 6), (1,)]),
    ("1A", [(3, 4, 5, 6), (1,), (6,), (1,)]),
    (".", [(2, 5, 6)]),
    ("-", [(3, 6)]),
    ("'a-b'", [(3,), (1,), (3, 6), (1, 2), (3,)]),
    ("a\nb", [(1,), (), (1, 2)]),
))
def test_a_text_gives_the_cells_a_reader_finds(text, want):
    assert braille.cells(text) == want


def test_a_capital_is_announced_and_a_capital_after_a_digit_needs_no_letter_sign():
    assert braille.cells("AbC") == [(6,), (1,), (1, 2), (6,), (1, 4)]
    assert braille.cells("1a2") == [(3, 4, 5, 6), (1,), (5, 6), (1,), (3, 4, 5, 6), (1, 2)]
    assert braille.cells("1A") == [(3, 4, 5, 6), (1,), (6,), (1,)]
    assert braille.line_cells("Ab")[0][0] == braille.CAPITAL_SIGN


def test_a_space_is_a_cell_without_dots_and_a_new_line_a_row_of_its_own():
    assert braille.line_cells("a b") == [[(1,), (), (1, 2)]]
    assert braille.cells("a b")[1] == ()
    assert braille.cells("ab\ncd") == braille.cells("ab") + [()] + braille.cells("cd")
    assert braille.line_cells("ab\ncd") == [[(1,), (1, 2)], [(1, 4), (1, 4, 5)]]
    with pytest.raises(ValueError, match="needs text"):
        braille.cells(" ")


def test_every_letter_has_a_cell_of_its_own():
    got = [braille.cells(ch)[0] for ch in "abcdefghijklmnopqrstuvwxyz"]
    assert len(set(got)) == 26, "two letters share a cell, so a label reads wrong"
    assert all(0 < len(cell) <= 6 for cell in got)
    assert all(all(1 <= dot <= 6 for dot in cell) for cell in got)


def test_the_letters_k_to_t_are_a_to_j_with_dot_three():
    for low, high in zip("abcdefghij", "klmnopqrst"):
        assert set(braille.cells(high)[0]) == set(braille.cells(low)[0]) | {3}


def test_the_digits_are_the_letters_a_to_j_behind_a_number_sign():
    assert braille.cells("5") == [braille.NUMBER_SIGN, braille.cells("e")[0]]
    assert braille.NUMBER_SIGN == (3, 4, 5, 6)
    assert braille.CAPITAL_SIGN == (6,) and braille.LETTER_SIGN == (5, 6)


def test_the_dots_of_a_cell_run_down_the_left_column_first():
    assert [braille.dot_xy(n) for n in (1, 2, 3)] == [(0.0, 0.0), (0.0, -2.5), (0.0, -5.0)]
    assert [braille.dot_xy(n) for n in (4, 5, 6)] == [(2.5, 0.0), (2.5, -2.5), (2.5, -5.0)]


@pytest.mark.parametrize("text, bad", (("€", "€"), ("aéb", "é"), ("đ", "đ"), ("Я", "Я"),
                                       ("a\U0001f600", "\U0001f600")))
def test_a_character_without_a_cell_is_named(text, bad):
    with pytest.raises(ValueError) as exc:
        braille.cells(text)
    assert bad in str(exc.value)


@pytest.mark.parametrize("kw, wrong", (
    ({"text": ""}, "needs text"), ({"text": "   "}, "needs text"),
    ({"text": None}, "needs text"), ({"text": "a\nb\nc\nd\ne"}, "at most"),
    ({"text": "x" * 41}, "at most"), ({"text": "a€"}, "€"),
    ({"margin": 1.0}, "margin"), ({"margin": 20.0}, "margin"),
    ({"thickness": 0.5}, "thickness"), ({"thickness": 12.0}, "thickness"),
    ({"radius": -0.5}, "corner radius"),
    ({"hole": -1.0}, "hole"), ({"hole": 20.0}, "hole")))
def test_impossible_labels_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        part(**kw)


def test_a_dot_number_that_is_not_in_a_cell_is_said():
    for bad in (0, 7, -2):
        with pytest.raises(ValueError, match=str(bad)):
            braille.dot_xy(bad)


def test_a_font_that_is_not_there_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="No font found"):
        braille.braille_label(**dict(BASE, letters=True))


def test_a_font_that_draws_nothing_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "Nowhere Sans")
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="No font found"):
        braille.braille_label(**dict(BASE, letters=True))


def test_the_font_is_not_asked_for_a_label_that_wants_no_letters(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("letters were asked of the font")

    monkeypatch.setattr(fonts, "text_loops", boom)
    assert part(letters=False).volume() > plate_prism(part(letters=False)).volume()


# ------------------------------------------------------------------ the label

@pytest.mark.parametrize("text", ("Volura", "Drawer 2a", "a\nb\nc", "Hello, World! 42"))
def test_a_label_is_one_valid_volume_that_stands_on_the_bed(text):
    m = part(text=text)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the dots are not fused to the plate"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the label does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the plate is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the plate is not centred"
    assert m.genus() == 0


def test_every_dot_of_the_text_is_one_piece_of_plastic():
    for text in ("Volura", "Drawer 2a", "a\nb", "A B.", "1a"):
        m = part(text=text)
        want = sum(len(cell) for cell in braille.cells(text))
        assert want > 0, f"{text!r} holds no dots at all"
        assert len(dots(m)) == want, text


def test_the_cells_stand_six_apart_and_their_dots_two_and_a_half():
    got = centres(dots(part(text="ab")))
    want = corner([(0.0, 0.0), (braille.CELL_PITCH, 0.0), (braille.CELL_PITCH,
                                                          -braille.DOT_PITCH)])
    assert len(got) == 3, got
    for (x, y), (wx, wy) in zip(got, want):
        assert x == pytest.approx(wx, abs=ABS) and y == pytest.approx(wy, abs=ABS), \
            f"{got} is not {want}"


def test_the_lines_stand_ten_apart_and_start_at_the_same_left_edge():
    got = centres(dots(part(text="a\nb")))
    assert len(got) == 3
    ys = sorted((y for _x, y in got), reverse=True)
    assert ys[0] - ys[1] == pytest.approx(braille.LINE_PITCH, abs=ABS), "the lines are not 10 mm"
    assert ys[1] - ys[2] == pytest.approx(braille.DOT_PITCH, abs=ABS), "dot 2 sits elsewhere"
    assert max(x for x, _y in got) < ABS, "the second line moved to the right"


def test_two_lines_are_left_aligned_and_not_centred():
    """A reader's hand starts at the left edge of every line, so the short line does not move."""
    got = [(0.5 * (p[0] + p[2]), 0.5 * (p[1] + p[3])) for p in islands(part(text="ab\nv"))]
    top = max(y for _x, y in got)
    first = min(x for x, y in got if y > top - braille.LINE_PITCH / 2.0)
    second = min(x for x, y in got if y <= top - braille.LINE_PITCH / 2.0)
    assert first == pytest.approx(second, abs=ABS), "the short line was centred"


def test_a_dot_stands_six_of_a_millimetre_high_and_is_a_dot_and_a_half_across():
    m = part(text="a")
    pieces = dots(m)
    assert len(pieces) == 1
    b = np.asarray((m - plate_prism(m)).decompose()[0].bounding_box(), dtype=float)
    assert b[2] == pytest.approx(T, abs=1e-6), "the dot is sunk into the plate"
    assert b[5] - T == pytest.approx(braille.DOT_H, abs=DOT_ABS)
    assert b[3] - b[0] == pytest.approx(braille.DOT_D, abs=DOT_ABS)
    assert b[4] - b[1] == pytest.approx(braille.DOT_D, abs=DOT_ABS)
    assert m.volume() > plate_prism(m).volume(), "a dot cuts away instead of standing up"


def test_the_plate_is_the_dots_with_margin_on_every_side():
    for margin in (braille.MARGIN_MIN, 4.0, braille.MARGIN_MAX):
        m = part(text="Volura", margin=margin)
        pts = braille.dot_centres("Volura")
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(max(p[0] for p in pts) - min(p[0] for p in pts)
                                              + braille.DOT_D + 2.0 * margin)
        assert hi[1] - lo[1] == pytest.approx(max(p[1] for p in pts) - min(p[1] for p in pts)
                                              + braille.DOT_D + 2.0 * margin)
        assert hi[2] == pytest.approx(T + braille.DOT_H)


def test_a_hole_makes_a_longer_plate_and_a_handle():
    d = 4.0
    plain, hung = part(), part(hole=d)
    plo, phi = bbox(plain)
    lo, hi = bbox(hung)
    assert hung.genus() == plain.genus() + 1, "the screw cannot go through"
    assert hi[0] - lo[0] == pytest.approx(phi[0] - plo[0] + d + M)
    assert hi[1] - lo[1] == pytest.approx(phi[1] - plo[1]), "the plate grew in the wrong way"
    # the hole stands `margin` left of the dots, in the middle of the plate depth
    pts = braille.dot_centres(TEXT)
    wide = (max(p[0] for p in pts) + 0.5 * braille.DOT_D
            - (min(p[0] for p in pts) - 0.5 * braille.DOT_D))
    probe = mf.Manifold.cylinder(T + 2.0, 0.5 * d - 0.05, 0.5 * d - 0.05, braille.SEGMENTS) \
        .translate((-0.5 * (wide + M), 0.0, -1.0))
    assert (hung ^ probe).volume() == pytest.approx(0.0, abs=1e-6), "the hole is elsewhere"


def test_a_hole_wider_than_the_plate_is_deep_makes_the_plate_deeper_for_it():
    m = part(hole=braille.HOLE_MAX)
    lo, hi = bbox(m)
    assert hi[1] - lo[1] >= braille.HOLE_MAX + 2.0 * M - 1e-9, "the hole cuts the plate open"
    assert m.genus() == 1
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1


def test_a_square_plate_of_the_thinnest_with_the_widest_hole_still_stands_up():
    need_fonts()
    m = part(radius=0.0, margin=braille.MARGIN_MIN, thickness=braille.THICKNESS_MIN,
             hole=braille.HOLE_MAX, letters=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.genus() == 1, "the hole is not a hole anymore"
    assert bbox(m)[0][2] == pytest.approx(0.0) and bbox(m)[1][2] == pytest.approx(
        braille.THICKNESS_MIN + braille.DOT_H)
    assert faces_that_lean_down(m) == 0


def test_an_empty_line_keeps_its_place_between_the_lines():
    one, blank = part(text="a\nb"), part(text="a\n\nb")
    lo1, hi1 = bbox(one)
    lo2, hi2 = bbox(blank)
    assert hi2[1] - lo2[1] == pytest.approx(hi1[1] - lo1[1] + braille.LINE_PITCH)
    assert len(dots(blank)) == len(dots(one)), "an empty line lost or gained dots"


def test_the_biggest_label_still_comes_out_as_one_volume():
    """Four lines of forty characters is the most a label is; that is a lot of dots."""
    text = "\n".join(word * braille.CHARS_MAX for word in "xyzw")
    lo_t, hi_t = bbox(part(text=text))
    m = part(text=text, hole=4.0)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.genus() == 1, "the hole went away with so many dots"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0) and 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0)
    assert hi[0] - lo[0] == pytest.approx(hi_t[0] - lo_t[0] + 4.0 + M)


def test_nothing_of_the_label_leans_down_so_it_prints_without_support():
    for kw in ({}, {"hole": 4.0}, {"margin": braille.MARGIN_MIN}, {"text": "a\nb"},
               {"text": "1a", "hole": 6.0}):
        assert faces_that_lean_down(part(**kw)) == 0, f"{kw}: the label needs support"


def test_the_only_face_that_looks_down_is_the_bottom_on_the_bed():
    n, centre, area = facets(part())
    down = (n[:, 2] < -1e-6) & (area > 1e-6)
    assert down.any(), "a plate without a bottom face is not a plate"
    assert centre[down][:, 2].max() == pytest.approx(0.0, abs=1e-6)


def test_the_letters_above_make_the_plate_taller_and_stay_one_volume():
    need_fonts()
    dots_only, both = part(), part(letters=True)
    dlo, dhi = bbox(dots_only)
    lo, hi = bbox(both)
    assert both.status() == mf.Error.NoError
    assert len(both.decompose()) == 1, "the letters are not fused to the plate"
    assert hi[1] - lo[1] == pytest.approx(dhi[1] - dlo[1] + braille.LETTER_HEIGHT + M, abs=0.5)
    assert hi[2] == pytest.approx(dhi[2]), "the letters stand higher than the dots"
    assert hi[0] - lo[0] >= dhi[0] - dlo[0], "the plate lost width for the letters"
    assert (both - plate_prism(both)).volume() > (dots_only - plate_prism(dots_only)).volume()


def test_the_letters_stand_above_the_dots_and_read_from_their_left_edge():
    need_fonts()
    m = part(text="Volura", letters=True)
    ink, got = letters(m), dots(m)
    assert ink, "there are no letters above the dots"
    assert min(p[0] for p in ink) == pytest.approx(min(p[0] for p in got), abs=0.05), \
        "the letters are not left aligned with the dots"
    assert min(p[1] for p in ink) - max(p[3] for p in got) == pytest.approx(M, abs=0.5), \
        "the letters do not stand margin clear of the dots"
    assert max(p[4] for p in ink) == pytest.approx(T + braille.LETTER_RELIEF), \
        "the letters do not stand on the plate"


def test_the_letters_above_print_without_support_too():
    need_fonts()
    assert faces_that_lean_down(part(letters=True)) == 0
    assert faces_that_lean_down(part(text="Typing 4 g", letters=True)) == 0


def test_several_lines_of_dots_stand_above_each_other():
    one, two = part(text="Volura"), part(text="Volura\nTools")
    lo1, hi1 = bbox(one)
    lo2, hi2 = bbox(two)
    assert hi2[1] - lo2[1] == pytest.approx(hi1[1] - lo1[1] + braille.LINE_PITCH)
    assert hi2[0] - lo2[0] == pytest.approx(hi1[0] - lo1[0]), "the lines are not the same width"
    assert two.genus() == 0 and len(two.decompose()) == 1


# ------------------------------------------------------------------ in the window

def test_the_braille_label_is_one_group_that_one_undo_takes_away(win):
    need_fonts()
    steps = len(win.doc.undo_stack)
    win.make_braille_label_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no Braille label in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the label is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Braille label", "the group is not named after the first line"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(braille.braille_label(**OPTIONS).volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    assert pump_until(lambda: win.hint.text() == "Braille label added."), win.hint.text()
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo = np.asarray(win.doc.groups[0].bbox(), dtype=float)[0]
    assert lo[2] == pytest.approx(0.0), "the label does not stand on the bed"
    win.doc.undo()
    assert win.doc.groups == []


def test_a_label_that_cannot_be_built_is_a_clear_text(win):
    win.make_braille_label_model(dict(OPTIONS, text="€"))
    assert pump_until(lambda: "Braille label failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_braille_label_follows_the_bag_clip_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.parts_category("Home and Desk"))]
    assert labels.index("Braille Label") == labels.index("Bag Clip") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    need_fonts()
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["text"] = dlg.text.toPlainText()
        seen["start"] = [round(dlg.margin.value(), 2), round(dlg.thickness.value(), 2),
                         round(dlg.radius.value(), 2), round(dlg.hole.value(), 2)]
        seen["letters"] = dlg.letters.isChecked()
        seen["edit"] = isinstance(dlg.findChild(QPlainTextEdit), QPlainTextEdit)
        dlg.text.setPlainText("Kruiden 2")
        dlg.margin.setValue(5.0)
        dlg.thickness.setValue(3.0)
        dlg.radius.setValue(1.0)
        dlg.hole.setValue(4.0)
        dlg.letters.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.braille_text, win.settings.braille_margin = "Oud", 3.0
    win.settings.braille_thickness, win.settings.braille_radius = 1.5, 3.0
    win.settings.braille_hole, win.settings.braille_letters = 0.0, True
    win.settings.save()
    win.braille_label_dialog()
    assert seen["title"] == "Braille Label"
    assert seen["text"] == "Oud"
    assert seen["start"] == [3.0, 1.5, 3.0, 0.0]
    assert seen["letters"] is True
    assert seen["edit"], "the text is not written in a multi-line box"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no Braille label in the model"
    assert win.doc.groups[0].name == "Kruiden 2", "the group is not named after the first line"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] == pytest.approx(3.0 + braille.DOT_H), "the thickness is not as asked"
    assert hi[0] - lo[0] > hi[1] - lo[1], "the hole did not make the plate longer"
    assert win.settings.braille_text == "Kruiden 2"
    assert win.settings.braille_margin == pytest.approx(5.0)
    s = app_module.Settings()
    assert (s.braille_text, s.braille_margin, s.braille_thickness, s.braille_radius,
            s.braille_hole, s.braille_letters) == (
        "Kruiden 2", pytest.approx(5.0), pytest.approx(3.0), pytest.approx(1.0),
        pytest.approx(4.0), False)


def test_a_character_without_a_cell_is_said_and_ok_waits(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["ok_at_first"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        dlg.text.setPlainText("Volura €")
        seen["info"] = dlg.info.text()
        seen["ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        seen["marked"] = bool(dlg.text.styleSheet())
        dlg.text.setPlainText("Volura")
        seen["ok_again"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        seen["clean"] = dlg.text.styleSheet()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.braille_text = "Volura"
    win.settings.save()
    assert win.braille_label_dialog() is None
    assert seen["ok_at_first"], "OK waits for a text that reads fine"
    assert "€" in seen["info"], seen["info"]
    assert not seen["ok"], "OK still works with a character that has no cell"
    assert seen["marked"], "the field is not marked"
    assert seen["ok_again"], "OK did not come back for a text that reads fine"
    assert seen["clean"] == "", "the field stayed marked"
    assert win.doc.groups == [], "a refused label still came into the model"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
