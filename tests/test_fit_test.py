"""Fit Test: a comb of slots that shows which offset presses best on the sheet in use.

`boxgen.fit_test` makes the comb as one closed outline: a strip with one slot per offset
along its top edge, `t + offset` wide and `slot_depth` deep (2 × t, never shallower than
5 mm). The slots stand in cells of `10 + t` mm, so about 10 mm of material stays between
two of them, and 6 mm stays free at both ends and under the slots — under the slots
`label_height` more, the room the letters need. Under every slot its offset stands
engraved, "-0.1", "0", "+0.1", and the slots grow from narrow on the left to wide on the
right, so the user cuts the comb from the sheet on the table, pushes an offcut in every
slot and keeps the offset of the slot that holds best: that is the kerf or the finger
width to use. Without fonts the comb simply has no labels.

These tests (Qt offscreen) measure the slots on the outline, check the labels, the
refused input and the place in the interface (View menu and sheet toolbar next to Living
Hinge, the comb on the worksheet in one undo step).

Run with:  python -m pytest tests/test_fit_test.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QMenu, QSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import boxgen, fonts  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

T = 3.0
OFFSETS = (-0.2, -0.1, 0.0, 0.1, 0.2)
DEPTH = max(2.0 * T, 5.0)
LABEL = 4.0
PITCH = boxgen.FIT_TEST_TOOTH + T
WIDTH = 2.0 * boxgen.FIT_TEST_END + len(OFFSETS) * PITCH
OPTIONS = {"t": T, "first": -0.2, "step": 0.1, "count": 5}


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def comb(**kw):
    o = {"t": T, "offsets": OFFSETS, "label_height": LABEL}
    o.update(kw)
    return boxgen.fit_test(**o)


def bottoms(drawing):
    """(x_left, x_right, y) of every notch in the top edge, from left to right."""
    pts = np.asarray(drawing.paths[0].pts, dtype=float)
    top, base = float(pts[:, 1].max()), float(pts[:, 1].min())
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        if abs(float(a[1]) - float(b[1])) > 1e-9:
            continue                          # not a horizontal segment
        if not base + 1e-9 < float(a[1]) < top - 1e-9:
            continue                          # the bottom edge or a piece of the top edge
        out.append((min(float(a[0]), float(b[0])), max(float(a[0]), float(b[0])),
                    float(a[1])))
    return sorted(out)


def labels_of(drawing, op="engrave"):
    return [p for p in drawing.paths if drawing.layers[p.layer].op == op]


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


def menu_labels(win, title):
    for m in win.menuBar().findChildren(QMenu):
        if clean(m.title()) == title:
            return [clean(a.text()) for a in m.actions() if not a.isSeparator()]
    raise AssertionError(f"no menu {title}")


# ------------------------------------------------------------------ the comb

def test_the_outline_is_one_closed_part_on_the_cut_layer():
    d = comb()
    assert len(d.paths) >= 1
    part = d.paths[0]
    assert part.closed and part.filled
    assert d.layers[part.layer].op == "cut"
    pts = np.asarray(part.pts, dtype=float)
    assert pts[:, 0].min() == pytest.approx(0.0) and pts[:, 1].min() == pytest.approx(0.0)
    assert pts[:, 0].max() == pytest.approx(WIDTH)
    assert pts[:, 1].max() == pytest.approx(DEPTH + boxgen.FIT_TEST_ROOM + LABEL)


def test_the_slots_are_t_plus_the_offset_and_grow_to_the_right():
    got = bottoms(comb())
    assert len(got) == len(OFFSETS)
    assert [round(r - l, 6) for l, r, _y in got] == pytest.approx([T + o for o in OFFSETS])
    widths = [r - l for l, r, _y in got]
    assert widths == sorted(widths), "the slots must go from narrow to wide"


def test_the_slots_are_a_pitch_apart_with_six_mm_of_material_all_around():
    got = bottoms(comb())
    centres = [0.5 * (l + r) for l, r, _y in got]
    assert [round(c - b, 6) for c, b in zip(centres[1:], centres[:-1])] == \
        pytest.approx([PITCH] * (len(OFFSETS) - 1))
    assert got[0][0] >= boxgen.FIT_TEST_END - 1e-9
    assert got[-1][1] <= WIDTH - boxgen.FIT_TEST_END + 1e-9


def test_the_depth_of_the_slots_and_the_material_under_them():
    got = bottoms(comb())
    ys = {round(y, 6) for _l, _r, y in got}
    assert len(ys) == 1, "all slots are cut to the same depth"
    assert next(iter(ys)) == pytest.approx(DEPTH + boxgen.FIT_TEST_ROOM + LABEL - DEPTH)
    # a deeper cut keeps the same room for the letters under the slots
    ys = {round(y, 6) for _l, _r, y in bottoms(comb(slot_depth=9.0))}
    assert next(iter(ys)) == pytest.approx(boxgen.FIT_TEST_ROOM + LABEL)


def test_the_default_depth_is_twice_the_thickness_but_never_under_five_mm():
    for t in (0.5, 2.0, 6.0):
        d = comb(t=t, offsets=(-0.1, 0.1))
        top = float(np.asarray(d.paths[0].pts, dtype=float)[:, 1].max())
        assert top - bottoms(d)[0][2] == pytest.approx(max(2.0 * t, 5.0))


def test_the_offsets_stand_engraved_under_their_slot():
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no labels to test")
    d = comb()
    labels = labels_of(d)
    assert labels, "the offsets should stand under the slots"
    assert all(p.closed and p.filled for p in labels)
    assert all(int(p.src) == int(d.paths[0].src) for p in labels), "labels move with the comb"
    centres = [0.5 * (l + r) for l, r, _y in bottoms(d)]
    floor = bottoms(d)[0][2]
    for c in centres:
        mine = [p for p in labels
                if abs(0.5 * (np.asarray(p.pts)[:, 0].min() + np.asarray(p.pts)[:, 0].max())
                       - c) <= 0.5 * PITCH]
        assert mine, f"no offset under the slot at x = {c:g}"
    for p in labels:
        q = np.asarray(p.pts, dtype=float)
        assert float(q[:, 1].max()) < floor, "a label cuts into a slot"
        assert float(q[:, 0].min()) >= 0.0 and float(q[:, 0].max()) <= WIDTH


def test_without_fonts_the_comb_has_no_labels(monkeypatch):
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])
    d = comb()
    assert len(d.paths) == 1
    assert not labels_of(d)


def test_impossible_sizes_raise():
    with pytest.raises(ValueError, match="thickness"):
        comb(t=0.0)
    with pytest.raises(ValueError, match="thickness"):
        comb(t=-2.0)
    with pytest.raises(ValueError, match="two slots"):
        comb(offsets=(0.1,))
    with pytest.raises(ValueError, match="two slots"):
        comb(offsets=())
    with pytest.raises(ValueError, match="wide"):
        comb(t=1.0, offsets=(-1.0, 0.0))
    with pytest.raises(ValueError, match="wide"):
        comb(t=1.0, offsets=(-1.5, 0.2))
    with pytest.raises(ValueError, match="depth"):
        comb(slot_depth=0.0)


# ------------------------------------------------------------------ in the window

def test_the_action_adds_the_comb_in_one_undo_step(win):
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    added = win.make_fit_test_sheet(OPTIONS)
    assert added, "nothing on the worksheet"
    assert len(win.doc.undo_stack) == steps + 1, "the comb is not one undo step"
    assert len(win.doc.sheet.paths) == len(before) + len(added)
    cut = win.doc.sheet.default_layer("cut")
    assert len([p for p in added if p.layer == cut]) == 1, "one outline on the cut layer"
    assert all(int(p.src) == int(added[0].src) for p in added), "the comb is one part"
    assert all(p in win.sheet_view.selected_paths() for p in added)
    assert win.hint.text() == ("Cut it and use the slot that holds the sheet best: its "
                               "offset goes into the kerf or the finger width.")
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_the_comb_lands_to_the_right_of_what_is_already_on_the_sheet(win):
    sheet = win.doc.sheet
    sheet.add_path([[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]], closed=True,
                   layer=sheet.default_layer("cut"))
    added = win.make_fit_test_sheet(OPTIONS)
    pts = np.vstack([np.asarray(p.pts, dtype=float) for p in added])
    assert float(pts[:, 0].min()) == pytest.approx(40.0 + sheet.sheet_gap)
    assert float(pts[:, 1].min()) == pytest.approx(0.0)


def test_a_thickness_that_leaves_no_slot_is_a_clear_text(win):
    before = list(win.doc.sheet.paths)
    assert win.make_fit_test_sheet({"t": 0.5, "first": -1.0, "step": 0.1, "count": 5}) is None
    assert win.doc.sheet.paths == before
    assert "wide" in win.hint.text()


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        count = dlg.findChild(QSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins] + [count.value()]
        seen["thickness"] = dlg.windowTitle()
        spins[1].setValue(-0.3)
        spins[2].setValue(0.2)
        count.setValue(7)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.fit_test_first = -0.15
    win.settings.fit_test_step = 0.05
    win.settings.fit_test_count = 4
    win.fit_test_dialog()
    want_t = win.doc.sheet.sheet_thickness or boxgen.DEFAULT_THICKNESS
    assert seen["start"] == [round(want_t, 2), pytest.approx(-0.15), pytest.approx(0.05), 4]
    assert seen["thickness"] == "Fit Test"
    assert win.settings.fit_test_first == pytest.approx(-0.3)
    assert win.settings.fit_test_step == pytest.approx(0.2)
    assert win.settings.fit_test_count == 7
    s = app_module.Settings()
    assert (s.fit_test_first, s.fit_test_step, s.fit_test_count) == \
        (pytest.approx(-0.3), pytest.approx(0.2), 7)


def test_the_fit_test_follows_the_living_hinge_in_the_view_menu(win):
    labels = menu_labels(win, "View")
    assert labels.index("Vent Pattern") == labels.index("Living Hinge") + 1
    assert labels.index("Fit Test") == labels.index("Vent Pattern") + 1


def test_the_fit_test_follows_the_living_hinge_on_the_sheet_toolbar(win):
    labels = [clean(a.text()) for a in win.tb_sheet.actions() if not a.isSeparator()]
    assert labels.index("Vent Pattern") == labels.index("Living Hinge") + 1
    assert labels.index("Fit Test") == labels.index("Vent Pattern") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
