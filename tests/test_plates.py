"""More than one build plate: a big project spread over several beds of the same printer.

The plate a group belongs to is *where it stands*: the bed under the centre of its bounding box
(`volura.plates`), so a move, a scale or an undo can never leave a part on the wrong plate -
there is nothing to keep in step. `g.meta["plate"]` is only a cache of that, written by
`plates.sync_meta` so a saved file says what the screen showed. Tools > Arrange on Plates
(`ops.arrange_on_plates`) fills one bed and sends what does not fit to the next, the Print tab
shows one plate at a time through the plate chooser, File > Slice slices the plate you are
looking at as if it stood on the real bed, and File > Slice all Plates… writes one G-code per
plate with a table of what each took.

These tests (Qt offscreen) check the plate of a group and the junk a hand-written file can
contain, the plate in the saved file and what undo does to it, the spread over several beds,
what the plate chooser leaves on the screen, and - with the slicer faked - that a plate goes to
the engine where plate 1 would have put it and that every plate gets its own file. The helpers
`add_cube`, `FakeBox` and `pump_until` follow tests/test_arrange.py.

Run with:  python -m pytest tests/test_plates.py
"""
import itertools
import os
import sys
import time

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import volura.app as app_module  # noqa: E402
import volura.window_print as wp  # noqa: E402
from volura import (cura_profiles, fileio, ops, plates, print_history,  # noqa: E402
                    print_queue, printer_link, slicer)
from volura.i18n import tr  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.slice_dialog import Multicolor  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BED = (200.0, 200.0, 200.0)
#: a bed of the same width but deeper: three cubes of 55 mm side by side and four rows deep, so a
#: plate holds twelve of them and thirty of them need three plates
DEEP = (200.0, 300.0, 200.0)
CUBE = 50.0
MARGIN = ops.BED_MARGIN
GAP = ops.BED_GAP

#: every setting a test of this file relies on, so no test follows what another one left behind
#: in the settings
SETTING_KEYS = ("bed_size", "slice_thumbnail", "filament_price", "print_queue", "printers")

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])

    def critical(self, *args, **kwargs):
        self.asked.append(args[1:])


class FakeStacks:
    """Just what the slice reads out of the stacks of a printer: the size of its machine."""

    def __init__(self, bed=BED):
        self.glob = {"machine_width": bed[0], "machine_depth": bed[1],
                     "machine_height": bed[2]}
        self.extruders = []                    # no filament of its own: 1.75 mm is what counts


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs into the folder of the test, on a plate-sized bed of its own."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    copies = tmp_path / "copies"
    copies.mkdir()
    monkeypatch.setattr(print_history, "gcode_dir", lambda create=True: str(copies))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    monkeypatch.setattr(wp, "QMessageBox", box)      # the slice dialogs live in window_print
    monkeypatch.setattr(wp, "PrintPreviewDialog", FakePreview)   # a slice opens no real preview
    monkeypatch.setattr("volura.slice_dialog.PlateSummaryDialog", FakeSummary)
    q = qsettings()
    saved = {k: q.value(k) for k in SETTING_KEYS}
    w = app_module.MainWindow()
    w.box = box
    w.settings.bed_size = BED
    w.settings.slice_thumbnail = "off"
    w.settings.filament_price = 20.0
    w.settings.print_queue = []
    w.settings.printers = []
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()


def add_cube(doc, at, size=CUBE, name="cube"):
    """A closed cube of `size` mm with its corner on `at`, as its own group."""
    m = mf.Manifold.cube((float(size), float(size), float(size))).translate(at)
    return ops.group_from_manifold(doc, m, "#cccccc", name)


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value.

    A background job hands its result to the window through a signal that the loop picks up one
    turn later, so a few more turns follow: otherwise the window finishes a job in the middle of
    the next test, with the dialogs that test replaced already back in place."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    got = predicate()
    for _ in range(30):
        _app.processEvents()
        time.sleep(0.005)
    return got


def gap_between(a, b):
    """The free space between two boxes on the axis that separates them, -1 if they overlap."""
    for axis in (0, 1):
        d = max(a[0][axis] - b[1][axis], b[0][axis] - a[1][axis])
        if d > 1e-6:
            return d
    return -1.0


def assert_beds_free(groups, bed=BED):
    """Every group inside the bed it stands on, on that bed, with the gap free from the others."""
    for g in groups:
        lo, hi = g.ents.bbox()
        at = plates.origin(plates.plate_of_group(g, bed), bed)
        assert abs(lo[2]) < 1e-6, f"{g.name} does not stand on its bed"
        assert (lo[:2] >= at[:2] + np.array([MARGIN - 1e-6] * 2)).all(), \
            f"{g.name} outside the margin of its bed: {lo}"
        assert (hi[:2] <= at[:2] + np.array(bed[:2]) - MARGIN + 1e-6).all(), \
            f"{g.name} outside its bed: {hi}"
    for a, b in itertools.combinations(groups, 2):
        if plates.plate_of_group(a, bed) != plates.plate_of_group(b, bed):
            continue                             # the parts of two beds never touch
        assert gap_between(a.ents.bbox(), b.ents.bbox()) >= GAP - 1e-6, \
            f"parts overlap or touch: {a.name} vs {b.name}"


def names(groups):
    return sorted(g.name for g in groups)


# ------------------------------------------------------------------ the plate of a group

def test_the_plate_is_the_bed_under_the_centre_of_the_part():
    doc = Document()
    near = add_cube(doc, (10.0, 10.0, 0.0), name="near")
    far = add_cube(doc, (300.0, 10.0, 0.0), name="far")      # plate 2 starts at 240
    assert plates.plate_of_group(near, BED) == plates.FIRST
    assert plates.plate_of_group(far, BED) == 2
    assert plates.plates_used(doc, BED) == [1, 2]
    assert plates.plate_count(doc, BED) == 2
    assert names(plates.groups_on_plate(doc, 2, BED)) == ["far"]
    assert names(plates.groups_on_plate(doc, plates.ALL, BED)) == ["far", "near"]
    assert plates.plates_used(Document(), BED) == [1], "an empty model is one plate"


def test_a_part_across_two_beds_follows_its_centre():
    doc = Document()
    g = add_cube(doc, (215.0, 10.0, 0.0), name="across")     # 215..265: centre on the second bed
    assert g.ents.bbox()[0][0] < plates.pitch(BED) < g.ents.bbox()[1][0]
    assert plates.plate_of_group(g, BED) == 2


def test_a_bed_without_a_width_or_junk_puts_everything_on_the_first_plate():
    assert plates.plate_of_x(500.0, (0.0, 200.0, 200.0)) == plates.FIRST
    assert plates.origin(3, (0.0, 200.0, 200.0)) == pytest.approx([0.0, 0.0, 0.0])
    assert plates.plate_of_x(500.0, "wide") == plates.FIRST
    assert plates.plate_of_x(None, BED) == plates.FIRST
    assert plates.plate_of_x(float("nan"), BED) == plates.FIRST
    assert plates.plate_of_x(-500.0, BED) == plates.FIRST


def test_a_plate_number_that_says_nothing_is_the_first_plate():
    for junk in (None, "", "three", {}, [], "3 plates", 0, -2, -0.5, float("nan"),
                 float("inf"), float("-inf"), object()):
        assert plates.plate_number(junk) == 1, f"{junk!r} is no plate number"
    for good, want in (("3", 3), (3.0, 3), (3, 3), ("1", 1), (2.9, 2)):
        assert plates.plate_number(good) == want, f"{good!r} is not {want}"
    assert plates.plate_choice(plates.ALL) == plates.ALL
    assert plates.plate_choice("0") == plates.ALL
    assert plates.plate_choice(None) == plates.ALL
    assert plates.plate_choice(3) == 3
    assert plates.plate_choice("junk") == 1


def test_a_group_that_says_nothing_about_its_plate_is_on_the_first():
    def group(meta, box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)):
        return type("G", (), {"meta": meta,
                              "bbox": lambda s, b=box: np.array([b[:3], b[3:]])})()
    assert plates.stored_plate(group(None)) == 1
    assert plates.stored_plate(group({})) == 1
    assert plates.stored_plate(group({"plate": "junk"})) == 1
    assert plates.stored_plate(group({"plate": 3})) == 3
    assert plates.plate_of_group(group({"plate": 3}), BED) == 1     # where it stands wins
    assert plates.plate_of_group(type("G", (), {"meta": {"plate": 3},
                                                "bbox": lambda s: None})(), BED) == 3


# ------------------------------------------------------------------- the plate in the file

def test_the_plate_of_a_group_is_saved_and_read_back(win, tmp_path):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    assert plates.sync_meta(win.doc, BED) == 1          # the far cube says nothing yet
    near = next(g for g in win.doc.groups if g.name == "near")
    far = next(g for g in win.doc.groups if g.name == "far")
    assert far.meta["plate"] == 2, "the plate of a part on the second bed is not in its meta"
    assert "plate" not in near.meta, "the first plate should not need a key"
    path = str(tmp_path / "two plates.volura")
    win.doc.path = path
    fileio.save_document(win.doc, path)
    other = Document()
    fileio.load_document(other, path)
    assert {g.name: plates.stored_plate(g) for g in other.groups} == {"near": 1, "far": 2}
    assert plates.plates_used(other, BED) == [1, 2]


def test_sync_meta_leaves_alone_what_already_says_the_right_thing(win):
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    assert plates.sync_meta(win.doc, BED) == 1
    assert plates.sync_meta(win.doc, BED) == 0
    win.doc.groups[0].meta["plate"] = 7                 # a number that disagrees with the bed
    assert plates.sync_meta(win.doc, BED) == 1
    assert win.doc.groups[0].meta["plate"] == 2, "where the part stands wins"


def test_a_part_that_moves_by_undo_takes_its_plate_number_back(win):
    g = add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.doc.checkpoint()
    ops.transform_selection(win.doc, np.eye(3), np.array([-200.0, 0.0, 0.0]),
                            items=[("g", g.id, g.id)])
    win.sync_plates()
    assert plates.stored_plate(g) == 1 and "plate" not in g.meta
    win.doc.undo()
    assert g.ents.bbox()[0][0] == pytest.approx(300.0)
    win.sync_plates()
    assert g.meta["plate"] == 2


def test_the_window_keeps_the_plate_cache_in_step(win):
    g = add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.doc.notify()                            # what every command does after it changed the model
    assert g.meta.get("plate") == 2, "the window did not write the plate of the part"
    ops.transform_selection(win.doc, np.eye(3), np.array([-200.0, 0.0, 0.0]),
                            items=[("g", g.id, g.id)])
    win.on_doc_changed()
    assert "plate" not in g.meta


def test_a_saved_file_carries_the_plate_even_without_a_change_notice(win, tmp_path):
    g = add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    path = str(tmp_path / "far.volura")
    win.doc.path = path
    win.file_save()
    assert g.meta.get("plate") == 2, "saving did not write down where the part stands"
    other = Document()
    fileio.load_document(other, path)
    assert plates.stored_plate(other.groups[0]) == 2


# ------------------------------------------------------------------- Arrange on Plates

def thirty_cubes(doc, at=(10.0, 10.0, 0.0), size=55.0):
    """Thirty cubes of 55 mm on one and the same spot, one on top of the other."""
    return [add_cube(doc, (at[0], at[1], k * size), size=size, name=f"cube {k}")
            for k in range(30)]


def test_thirty_cubes_spread_over_three_beds(win):
    """A plate of 200 × 300 mm holds twelve cubes of 55 mm; thirty of them need three plates."""
    win.settings.bed_size = DEEP
    groups = thirty_cubes(win.doc)
    moved, used = ops.arrange_on_plates(win.doc, [("g", g.id, g.id) for g in groups], DEEP)
    assert (moved, used) == (30, 3)
    assert plates.plates_used(win.doc, DEEP) == [1, 2, 3]
    assert_beds_free(groups, DEEP)
    for plate in (1, 2, 3):
        assert len(plates.groups_on_plate(win.doc, plate, DEEP)) > 0, f"plate {plate} stayed empty"


def test_arrange_on_plates_is_one_undo_step(win):
    win.settings.bed_size = DEEP
    groups = thirty_cubes(win.doc)
    spots = [g.ents.bbox()[0].copy() for g in groups]
    win.doc.checkpoint()
    ops.arrange_on_plates(win.doc, [("g", g.id, g.id) for g in groups], DEEP)
    assert any(not np.allclose(g.ents.bbox()[0], at) for g, at in zip(groups, spots))
    win.doc.undo()
    for g, at in zip(groups, spots):
        assert g.ents.bbox()[0] == pytest.approx(at), f"undo did not bring {g.name} back"
    assert plates.plates_used(win.doc, DEEP) == [1]


def test_a_part_that_does_not_fit_a_plate_raises_and_moves_nothing(win):
    groups = [add_cube(win.doc, (10.0, 10.0, 0.0), size=400.0, name="big"),
              add_cube(win.doc, (500.0, 10.0, 0.0), name="small")]
    spots = [g.ents.bbox()[0].copy() for g in groups]
    with pytest.raises(ValueError) as exc:
        ops.arrange_on_plates(win.doc, [("g", g.id, g.id) for g in groups], BED)
    assert "big" in str(exc.value)
    for g, at in zip(groups, spots):
        assert g.ents.bbox()[0] == pytest.approx(at)


def test_arrange_on_bed_leaves_the_other_plates_where_they_are(win):
    near = add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    far = add_cube(win.doc, (300.0, 100.0, 0.0), name="far")
    at = near.ents.bbox()[0].copy()
    ops.arrange_on_bed(win.doc, [("g", g.id, g.id) for g in (near, far)],
                       win.settings.bed_size, plate=2)
    assert np.allclose(near.ents.bbox()[0], at), "arrange moved a part of another plate"
    assert plates.plate_of_group(far, BED) == 2
    assert_beds_free([far])


def test_arrange_on_plates_looks_at_every_plate(win):
    near = add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    far = add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.settings.bed_size = DEEP
    win.set_plate(1)
    groups = [add_cube(win.doc, (10.0, 10.0, k * CUBE), name=f"pile {k}") for k in range(30)]
    win.a_arrange_plates.trigger()
    everything = [near, far] + groups
    assert pump_until(lambda: plates.plate_count(win.doc, DEEP) > 2), "nothing was spread out"
    assert_beds_free(everything, DEEP)
    assert plates.plates_used(win.doc, DEEP) == [1, 2, 3]


def test_parts_and_role_parts_follow_the_plate(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    far = add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    ops.set_group_role(far, ops.ROLE_BLOCKER)
    assert names(ops.parts(win.doc, plate=1, bed=BED)) == ["near"]
    assert names(ops.parts(win.doc, plate=2, bed=BED)) == []
    assert names(ops.role_parts(win.doc, plate=2, bed=BED)) == ["far"]
    assert names(ops.parts(win.doc, bed=BED)) == ["near"]
    assert names(ops.role_parts(win.doc, bed=BED)) == ["far"]


def test_the_two_new_commands_are_in_the_menus(win):
    file_menu = win.menuBar().actions()[0].menu()
    assert any(a is win.a_slice_all for a in file_menu.actions())
    assert any(a is win.a_arrange_plates for a in win.print_tools_menu.actions())
    assert tr("Arrange on Plates") in win.a_arrange_plates.text()


# ------------------------------------------------------------------- the plate chooser

def test_the_plate_chooser_filters_what_is_drawn(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.set_mode("3d")

    def drawn():
        return sorted(g.name for _c, _e, g in win.view._plate_contexts(False) if g is not None)

    assert drawn() == ["far", "near"]
    win.set_plate(2)
    assert drawn() == ["far"]
    win.set_plate(1)
    assert drawn() == ["near"]
    win.set_plate(plates.ALL)
    assert drawn() == ["far", "near"]
    assert win.view._plate_shown(None) is True
    win.set_plate(3)
    assert drawn() == [], "a plate that is still empty shows the parts of the others"


def test_the_camera_frames_the_plate_you_are_looking_at(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.set_mode("3d")
    win.set_plate(2)
    lo, hi = win.view.visible_bbox()
    assert lo[0] >= plates.pitch(BED) - 1e-6, "the view still frames the first plate"
    assert hi[0] <= 2 * plates.pitch(BED) + 1e-6
    win.set_plate(9)                                  # a plate that is still empty
    assert win.view.visible_bbox() is None
    assert win.view._plate_beds() == list(range(2, 11))


def test_the_plate_chooser_lists_every_plate_the_model_needs(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.set_workspace("print")
    box = win.plate_box
    assert [box.itemData(k) for k in range(box.count())] == [1, 2, plates.ALL]
    assert box.itemText(box.count() - 1) == tr("All plates")
    box.setCurrentIndex(0)                         # plate 1
    assert win.plate_choice == 1 and win.view.plate == 1
    box.setCurrentIndex(box.count() - 1)
    assert win.plate_choice == plates.ALL and win.current_plate() == plates.FIRST
    assert win.view.plate == plates.ALL, "the view still shows one plate"


def test_the_plate_chooser_hides_itself_for_a_model_of_one_plate(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    win.set_workspace("print")
    assert not win.plate_box.isVisibleTo(win)
    assert win.plate_choice == plates.ALL


def test_a_plate_that_is_still_empty_becomes_a_bed_on_the_screen(win):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    win.set_mode("3d")
    assert win.view._plate_beds() == []
    win.set_plate(3)
    assert win.view._plate_beds() == [2, 3, 4], "an empty plate needs a bed to stand on"


def test_leaving_the_print_tab_shows_every_plate_again(win):
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.set_workspace("print")
    win.set_plate(2)
    win.set_workspace("design")
    assert win.plate_choice == plates.ALL and win.view.plate == plates.ALL


# --------------------------------------------------------------------------- the slice

class FakeSlicer:
    """Stands in for `slicer.slice_meshes`: keeps the meshes it was handed, writes a file."""

    def __init__(self):
        self.calls = []

    def __call__(self, stacks, meshes, out_path, engine=None, timeout=None,
                 on_progress=None, helpers=None):
        self.calls.append(([(v.copy(), t.copy(), e) for v, t, e, _s in meshes],
                           [(v.copy(), t.copy()) for v, t in (helpers or [])]))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("; fake gcode\nM104 S200\n")
        return slicer.SliceResult(path=out_path, seconds=120.0, grams=8.0, layers=6)


class FakePreview:
    """Stands in for the print preview: here the slice is what counts, not the layers."""

    def __init__(self, *args, **kwargs):
        pass

    def exec(self):
        return QDialog.Accepted


def sliced_meshes(win, monkeypatch, plate, parts, bed=BED):
    """The meshes File > Slice hands to the engine for `parts` of plate `plate`."""
    fake = FakeSlicer()
    monkeypatch.setattr(slicer, "slice_meshes", fake)
    monkeypatch.setattr(wp, "PrintPreviewDialog", FakePreview)
    win._slice_async(FakeStacks(), parts, "engine", "Test printer", Multicolor(1),
                     plate=plate, bed=bed)
    assert pump_until(lambda: bool(fake.calls)), "the slice never reached the engine"
    return fake.calls[0][0]


def test_a_multicolor_3mf_of_several_plates_says_what_it_cannot_carry(win, monkeypatch, tmp_path):
    """A 3MF has no word for a second bed, so the export says where the other plates went."""
    from volura import app as app_mod
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    path = str(tmp_path / "two plates.3mf")
    monkeypatch.setattr(app_mod, "QFileDialog", type("D", (), {
        "getSaveFileName": staticmethod(lambda *a, **kw: (path, "3MF (*.3mf)"))}))
    win.export_3mf()
    assert pump_until(lambda: bool(win.box.asked)), "nothing was said about the export"
    said = str(win.box.asked[-1])
    assert "2 plates" in said, win.box.asked
    assert os.path.exists(path), "the 3MF was not written"


def test_a_multicolor_3mf_of_one_plate_says_nothing_about_plates(win, monkeypatch, tmp_path):
    from volura import app as app_mod
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    path = str(tmp_path / "one plate.3mf")
    monkeypatch.setattr(app_mod, "QFileDialog", type("D", (), {
        "getSaveFileName": staticmethod(lambda *a, **kw: (path, "3MF (*.3mf)"))}))
    win.export_3mf()
    assert pump_until(lambda: bool(win.box.asked)), "nothing was said about the export"
    assert "plates" not in str(win.box.asked[-1]), win.box.asked


def test_a_plate_goes_to_the_engine_where_plate_1_would_have_put_it(win, monkeypatch):
    """Plate 2 sliced is exactly the print the same parts would give on the real bed."""
    near = add_cube(win.doc, (10.0, 20.0, 0.0), name="near")
    first = sliced_meshes(win, monkeypatch, plates.FIRST, ops.parts(win.doc, plate=1, bed=BED))
    win.doc.remove_group(near.id)
    add_cube(win.doc, (10.0 + plates.pitch(BED), 20.0, 0.0), name="far")
    second = sliced_meshes(win, monkeypatch, 2, ops.parts(win.doc, plate=2, bed=BED))
    assert [m[2] for m in first] == [m[2] for m in second]
    for (a, at, _e), (b, bt, _e) in zip(first, second):
        assert np.allclose(a, b), "plate 2 went to the slicer on its own bed"
        assert np.allclose(at, bt), "the triangles are not the same ones"
        assert np.allclose(b.min(axis=0), [10.0, 20.0, 0.0])
        assert np.allclose(b.max(axis=0) - b.min(axis=0), [CUBE] * 3)


def test_the_slice_of_one_plate_leaves_the_other_plates_alone(win, monkeypatch):
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    meshes = sliced_meshes(win, monkeypatch, 2, ops.parts(win.doc, plate=2, bed=BED))
    assert len(meshes) == 1, "the plate that was not chosen was sliced too"
    assert np.allclose(np.min(meshes[0][0], axis=0), [60.0, 10.0, 0.0])


def test_a_part_that_is_too_big_for_the_machine_names_its_plate(win, monkeypatch):
    add_cube(win.doc, (250.0, 10.0, 0.0), size=400.0, name="big")
    fake = FakeSlicer()
    monkeypatch.setattr(slicer, "slice_meshes", fake)
    box = win.box
    win._slice_async(FakeStacks(bed=(100.0, 100.0, 100.0)), ops.parts(win.doc, plate=2, bed=BED),
                     "engine", "Test printer", Multicolor(1), plate=2, bed=BED)
    assert pump_until(lambda: bool(box.asked)), "nothing was said about the part that does not fit"
    assert any("build volume" in str(one) for one in box.asked), box.asked
    assert not fake.calls, "it sliced anyway"


class FakeSliceDialog:
    """Stands in for the slice dialog: one printer, the default choice, one extruder."""

    def __init__(self, win, lib, colors, count, **kwargs):
        self.printer = type("P", (), {"name": "Test printer"})()
        self.count = count

    def exec(self):
        return QDialog.Accepted

    def choice(self):
        return {}

    def stacks(self):
        return FakeStacks()

    def multicolor(self):
        return Multicolor(1)


class FakeSummary:
    """Stands in for the table at the end of Slice all Plates…: keeps the rows it was shown."""

    rows = None

    def __init__(self, win, rows, price=0.0):
        FakeSummary.rows = rows
        self.queued = False

    def exec(self):
        return QDialog.Accepted


def slice_all(win, monkeypatch, sd, folder, stacks=None, real=False):
    """File > Slice all Plates… with everything around the slice faked.

    `real` leaves `slicer.slice_meshes` alone, so the engine of this computer does the work; then
    there is nothing to watch and the helper returns None."""
    fake = FakeSlicer()
    FakeSummary.rows = None

    class Dialog(FakeSliceDialog):
        def stacks(self):
            return stacks or FakeStacks()

    monkeypatch.setattr(slicer, "find_engine", lambda: ENGINE or "engine")
    monkeypatch.setattr(slicer, "find_resources", lambda: RESOURCES or "resources")
    monkeypatch.setattr(slicer, "engine_problem", lambda engine: None)
    if not real:
        monkeypatch.setattr(slicer, "slice_meshes", fake)
    monkeypatch.setattr(sd, "SliceDialog", Dialog)
    monkeypatch.setattr(sd, "PlateSummaryDialog", FakeSummary)
    monkeypatch.setattr(wp, "QFileDialog", type("D", (), {
        "getExistingDirectory": staticmethod(lambda *a, **k: str(folder))}))
    win.slice_all_dialog()
    return fake


def test_slice_all_plates_writes_one_file_per_plate(win, monkeypatch, tmp_path):
    import volura.slice_dialog as sd
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), name="far")
    win.doc.path = os.path.join(str(tmp_path), "model.volura")
    folder = tmp_path / "gcode"
    folder.mkdir()
    fake = slice_all(win, monkeypatch, sd, folder)
    assert pump_until(lambda: FakeSummary.rows is not None), "no summary at the end"
    assert sorted(os.listdir(folder)) == ["model_plate1.gcode", "model_plate2.gcode"]
    assert [one[0] for one in FakeSummary.rows] == [1, 2]
    assert [one[1] for one in FakeSummary.rows] == [1, 1], "a plate counted the parts of another"
    assert [one[2] for one in FakeSummary.rows] == [str(folder / f"model_plate{k}.gcode")
                                                    for k in (1, 2)]
    assert all(one[3] == pytest.approx(120.0) and one[4] == pytest.approx(8.0)
               for one in FakeSummary.rows)
    assert len(fake.calls) == 2
    for one, call in zip(FakeSummary.rows, fake.calls):
        # every plate goes to the engine on the real bed: the cube 60 mm into the bed of plate 2
        # arrives 60 mm into plate 1
        at = 10.0 if one[0] == plates.FIRST else 60.0
        assert np.allclose(np.min(call[0][0][0], axis=0), [at, 10.0, 0.0]), \
            "a plate went to the engine on its own bed"


def test_slice_all_plates_skips_a_plate_without_parts(win, monkeypatch, tmp_path):
    import volura.slice_dialog as sd
    add_cube(win.doc, (10.0, 10.0, 0.0), name="near")
    add_cube(win.doc, (550.0, 10.0, 0.0), name="far")        # plate 3: plate 2 stays empty
    win.doc.path = os.path.join(str(tmp_path), "model.volura")
    folder = tmp_path / "gcode"
    folder.mkdir()
    fake = slice_all(win, monkeypatch, sd, folder)
    assert pump_until(lambda: FakeSummary.rows is not None), "no summary at the end"
    assert sorted(os.listdir(folder)) == ["model_plate1.gcode", "model_plate3.gcode"]
    assert [one[0] for one in FakeSummary.rows] == [1, 3], "the empty plate became a print"
    assert len(fake.calls) == 2


class FakeQueueDialog:
    """Stands in for the dialog that asks which printer the sliced plates are meant for."""

    def __init__(self, parent, printers):
        self.printers = list(printers)

    def exec(self):
        return QDialog.Accepted

    def printer_name(self):
        return "Test printer"


def test_add_all_to_the_print_queue_queues_every_plate(win, monkeypatch, tmp_path):
    """All plates in the queue: one printer for all of them, plate 1 first, with its own G-code."""
    import volura.print_preview as pp
    paths = []
    for plate in (1, 2):
        one = tmp_path / f"model_plate{plate}.gcode"
        one.write_text("; fake gcode\n", encoding="utf-8")
        paths.append(str(one))
    win.settings.printers = [printer_link.Connection("Test printer", "octoprint",
                                                     "192.168.1.20", "k")]
    monkeypatch.setattr(pp, "QueueDialog", FakeQueueDialog)
    win._queue_plates([(1, 1, paths[0], 120.0, 8.0), (2, 1, paths[1], 60.0, 3.0)])
    waiting = print_queue.records(win.settings.print_queue)
    assert [one.printer for one in waiting] == ["Test printer", "Test printer"]
    assert [one.name for one in waiting] == [tr("{name} — plate {n}").format(name="model", n=n)
                                             for n in (1, 2)]
    assert [one.minutes for one in waiting] == [pytest.approx(2.0), pytest.approx(1.0)]
    assert [one.grams for one in waiting] == [pytest.approx(8.0), pytest.approx(3.0)]
    for one in waiting:
        assert one.gcode and os.path.exists(one.gcode), "the queue lost the G-code of a plate"
        assert one.gcode.startswith(str(tmp_path / "copies")), "the copy went elsewhere"
    assert "2 waiting" in win.hint.text()


def test_add_all_without_a_printer_says_so_and_queues_nothing(win, monkeypatch, tmp_path):
    import volura.print_preview as pp
    path = tmp_path / "model_plate1.gcode"
    path.write_text("; fake gcode\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(pp, "QueueDialog", lambda *a, **kw: called.append(1))
    win._queue_plates([(1, 1, str(path), 60.0, 3.0)])
    assert not called, "it asked for a printer that is not there"
    assert win.settings.print_queue in ([], None), print_queue.records(win.settings.print_queue)
    assert any("Printers" in str(one) for one in win.box.asked), win.box.asked


@needs_engine
def test_a_part_of_every_plate_really_becomes_two_gcodes(win, monkeypatch, tmp_path):
    """The real engine: both plates slice, each one on the bed it was sliced for."""
    import volura.slice_dialog as sd
    add_cube(win.doc, (10.0, 10.0, 0.0), size=20.0, name="near")
    add_cube(win.doc, (300.0, 10.0, 0.0), size=20.0, name="far")
    win.doc.path = os.path.join(str(tmp_path), "model.volura")
    folder = tmp_path / "gcode"
    folder.mkdir()
    lib = cura_profiles.Library(RESOURCES)
    printer = lib.printer("creality_ender3")
    stacks = printer.stacks(printer.default_variant(), printer.default_material(), "standard",
                            {"infill_sparse_density": 0.0, "layer_height": 0.3})
    slice_all(win, monkeypatch, sd, folder, stacks=stacks, real=True)
    assert pump_until(lambda: FakeSummary.rows is not None, timeout_ms=240000), "no summary"
    assert sorted(os.listdir(folder)) == ["model_plate1.gcode", "model_plate2.gcode"]
    assert [one[0] for one in FakeSummary.rows] == [1, 2]
    for one in FakeSummary.rows:
        assert one[2].endswith(f"model_plate{one[0]}.gcode")
        assert one[3] > 0 and one[4] > 0, f"plate {one[0]} measured nothing"
        with open(one[2], encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        assert "G1 " in text, f"{one[2]} lays nothing down"
        # the cube of each plate is printed in the middle of the real bed, not 300 mm beside it
        xs, ys = [], []
        second = text.split(";LAYER:1\n", 1)[1].split(";LAYER:2\n", 1)[0]
        kind = ""
        for line in second.splitlines():
            if line.startswith(";TYPE:"):
                kind = line[len(";TYPE:"):].strip()      # the skirt is no part of the cube
            elif line.startswith("G1 ") and " E" in line and kind.startswith(("WALL", "SKIN")):
                for word in line.split()[1:]:
                    if word[0] == "X":
                        xs.append(float(word[1:]))
                    elif word[0] == "Y":
                        ys.append(float(word[1:]))
        assert xs and ys, f"{one[2]} lays nothing on its second layer"
        assert max(xs) - min(xs) == pytest.approx(20.0, abs=1.5), f"plate {one[0]}: not one cube"
        machine = slicer.resolved_settings(stacks)["global"]
        middle = (0.5 * float(machine["machine_width"]), 0.5 * float(machine["machine_depth"]))
        assert 0.5 * (max(xs) + min(xs)) == pytest.approx(middle[0], abs=2.0), \
            f"plate {one[0]} is not printed in the middle of the bed"
        assert 0.5 * (max(ys) + min(ys)) == pytest.approx(middle[1], abs=2.0), \
            f"plate {one[0]} is not printed in the middle of the bed"
