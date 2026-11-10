"""Storage bins: a printed open-top box with compartments and a ledge for a label.

`storage_bin.storage_bin` gives the bin as one closed volume that stands on `z = 0` and is centred
on the origin in plan: walls of `wall` and a floor of `floor`, rounded vertical corners, and
`cols` by `rows` compartments made by straight dividers that stop just under the top edge. Along
the inside of the front wall a ledge of twelve millimetres runs at the top, whose underside stands
at 45 degrees, so a label lies under it readable and the ledge prints without support.

These tests weigh a bin against what it promises: it is one volume that stands on the bed, the box
round it is the size asked for, a compartment is empty above the floor, a divider stands where two
compartments meet, and the ledge is plastic that is not there without it. A size that cannot be
printed becomes the nearest one that can. The place in the interface is checked too (Parts right
after the battery holder, one part in one undo step, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_storage_bin.py
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
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import storage_bin  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(length=80.0, width=60.0, height=40.0, wall=1.6, floor=1.2, radius=4.0,
            cols=1, rows=1, label=False)
OPTIONS = dict(BASE, cols=3, rows=2, label=True)
INSET = 2.0                             # mm a test box stays from the wall around it


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
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


def make(**kw):
    return storage_bin.storage_bin(**dict(BASE, **kw))


def inside(n_x=1, n_y=1, cols=1, rows=1, i=0, j=0):
    """A box that fills `n_x` by `n_y` compartments of the compartment at `i`, `j`, `INSET`
    millimetres inside the walls around it and one millimetre above the floor."""
    inner_l, inner_w = BASE["length"] - 2.0 * BASE["wall"], BASE["width"] - 2.0 * BASE["wall"]
    wide = n_x * inner_l / cols - 2.0 * INSET
    deep = n_y * inner_w / rows - 2.0 * INSET
    x = (i - 0.5 * (cols - 1)) * inner_l / cols
    y = (j - 0.5 * (rows - 1)) * inner_w / rows
    high = BASE["height"] - BASE["floor"] - 2.0 * INSET
    return mf.Manifold.cube([wide, deep, high]).translate(
        (x - 0.5 * wide, y - 0.5 * deep, BASE["floor"] + INSET - 1.0))


def gap(m, box):
    """How much of `box` is inside the bin `m`."""
    return mf.Manifold.batch_boolean([m, box], mf.OpType.Intersect).volume()


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


# ------------------------------------------------------------------ the bin

def test_the_default_bin_is_one_valid_volume_that_stands_on_the_bed():
    m = make()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() > 0.0, "the bin is not a volume"
    lo, _hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the bin does not stand on the bed"


def test_the_box_round_the_bin_is_the_size_asked():
    for kw in ({}, dict(cols=3, rows=2, label=True), dict(radius=0.0), dict(wall=5.0, floor=5.0)):
        lo, hi = bbox(make(**kw))
        assert hi[0] - lo[0] == pytest.approx(BASE["length"]), f"not {kw}"
        assert hi[1] - lo[1] == pytest.approx(BASE["width"]), f"not {kw}"
        assert hi[2] - lo[2] == pytest.approx(BASE["height"]), f"not {kw}"
        assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the bin is not centred"
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the bin is not centred"
    block = BASE["length"] * BASE["width"] * BASE["height"]
    assert make().volume() < 0.6 * block, "the inside is not cut out of the block"


def test_a_compartment_is_empty_above_the_floor():
    m = make()
    assert gap(m, inside()) == pytest.approx(0.0, abs=1e-6), "the inside is not empty"
    for cols, rows in ((3, 2), (2, 4)):
        m = make(cols=cols, rows=rows)
        for i in range(cols):
            for j in range(rows):
                assert gap(m, inside(cols=cols, rows=rows, i=i, j=j)) == pytest.approx(
                    0.0, abs=1e-6), f"compartment {i},{j} of {cols},{rows} is not empty"


def test_a_divider_stands_where_two_compartments_meet():
    box = inside()                       # the box that fills the inside of a bin without dividers
    assert gap(make(cols=3), box) > 1000.0, "there is no divider between the compartments"
    assert gap(make(rows=3), box) > 1000.0, "there is no divider between the rows"


def test_the_label_ledge_is_plastic_that_is_only_there_with_the_label():
    plain, ledged = make(), make(label=True)
    assert ledged.volume() > plain.volume(), "the ledge adds nothing"
    # against the front wall, a centimetre below the top edge: empty without the ledge
    place = mf.Manifold.cube([20.0, 2.5, 4.0]).translate((-10.0, -27.5, 34.0))
    assert gap(plain, place) == pytest.approx(0.0, abs=1e-6), "the plain bin has a ledge"
    assert gap(ledged, place) > 100.0, "the ledge does not lie along the inside of the front wall"


def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can():
    big = storage_bin.storage_bin(length=999, width=999, height=999, wall=99, floor=99,
                                  radius=99, cols=99, rows=99)
    assert big.status() == mf.Error.NoError and not big.is_empty()
    lo, hi = bbox(big)
    assert hi[0] - lo[0] == pytest.approx(storage_bin.LENGTH_MAX)
    assert hi[1] - lo[1] == pytest.approx(storage_bin.WIDTH_MAX)
    assert hi[2] - lo[2] == pytest.approx(storage_bin.HEIGHT_MAX)
    assert big.volume() == pytest.approx(make(length=300, width=300, height=200, wall=5.0,
                                              floor=5.0, radius=20, cols=10, rows=10,
                                              label=True).volume(), rel=1e-9)
    small = storage_bin.storage_bin(length=1, width=1, height=1, wall=0.01, floor=0.01, radius=-5)
    assert small.status() == mf.Error.NoError and not small.is_empty()
    lo, hi = bbox(small)
    assert hi[0] - lo[0] == pytest.approx(storage_bin.LENGTH_MIN)
    assert hi[1] - lo[1] == pytest.approx(storage_bin.WIDTH_MIN)
    assert hi[2] - lo[2] == pytest.approx(storage_bin.HEIGHT_MIN)


# ------------------------------------------------------------------ in the window

def test_the_bin_is_one_part_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_storage_bin_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no bin in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the bin is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Storage bin"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert len(m.decompose()) == 1, "the bin is not one part"
    assert m.volume() == pytest.approx(make(**OPTIONS).volume(), rel=1e-6)
    assert pump_until(lambda: "bin added" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Storage bin added."
    win.doc.undo()
    assert win.doc.groups == []


def test_the_bin_stands_on_the_bed_next_to_the_model(win):
    win.make_storage_bin_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no bin in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    bx, by, _bz = win.settings.bed_size
    assert lo[2] == pytest.approx(0.0), "the bin does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the bin is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_storage_bin_model(dict(OPTIONS, length=40.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second bin is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the bins are not 10 mm apart"


def test_the_bin_follows_the_holder_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Storage Bin") == labels.index("Battery Holder") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [round(dlg.length.value(), 2), int(dlg.cols.value()),
                         dlg.label.isChecked()]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        dlg.length.setValue(100.0)
        dlg.width.setValue(50.0)
        dlg.cols.setValue(3)
        dlg.rows.setValue(2)
        dlg.label.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.bin_length, win.settings.bin_cols, win.settings.bin_label = 70.0, 2, True
    win.settings.bin_hanger, win.settings.bin_label_text = False, ""
    win.storage_bin_dialog()
    assert seen["title"] == "Storage bin"
    assert seen["start"] == [70.0, 2, True]
    assert seen["spins"] == 6, "the dialog does not ask the six sizes in millimetres"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no bin in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(100.0), "the length is not as asked"
    assert hi[1] - lo[1] == pytest.approx(50.0), "the width is not as asked"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 1, "the bin is not one part"
    s = app_module.Settings()
    assert (s.bin_length, s.bin_width, s.bin_cols, s.bin_rows, s.bin_label) == (
        pytest.approx(100.0), pytest.approx(50.0), 3, 2, False)


# ------------------------------------------------------------------ the label

def in_pocket(label, kw=BASE):
    """The label laid in the pocket of the ledge, against the front wall."""
    inner_w = kw["width"] - 2.0 * kw["wall"]
    _lx, ly = storage_bin.label_size(kw["length"], kw["width"], kw["height"], kw["wall"],
                                     kw["floor"])
    return label.translate((0.0, -0.5 * inner_w + storage_bin.LABEL_PLAY + 0.5 * ly,
                            kw["height"] - storage_bin.LABEL_POCKET))


def label_of(text="Screws M3"):
    return storage_bin.bin_label(text, **{k: BASE[k] for k in (
        "length", "width", "height", "wall", "floor")})


def test_the_ledge_has_a_pocket_for_the_label():
    lx, ly = storage_bin.label_size()
    assert lx == pytest.approx(BASE["length"] - 2.0 * BASE["wall"] - 2.0 * storage_bin.LABEL_PLAY)
    assert ly == pytest.approx(storage_bin.LABEL_DEPTH - storage_bin.LABEL_LIP
                               - 2.0 * storage_bin.LABEL_PLAY)
    plate = mf.Manifold.cube([lx, ly, storage_bin.LABEL_POCKET]).translate(
        (-0.5 * lx, -0.5 * ly, 0.0))
    assert gap(make(label=True), in_pocket(plate)) == pytest.approx(0.0, abs=1e-6), \
        "the pocket is not there"


def test_the_label_lies_flush_in_the_pocket_and_the_lip_holds_it():
    label = label_of()
    assert len(label.decompose()) == 1, "the letters are not one piece with the plate"
    lo, hi = bbox(label)
    assert hi[2] - lo[2] == pytest.approx(storage_bin.LABEL_POCKET + storage_bin.LABEL_RELIEF)
    placed = in_pocket(label)
    m = make(label=True)
    assert gap(m, placed) == pytest.approx(0.0, abs=1e-6), "the label does not fit"
    assert gap(m, placed.translate((0.0, 0.5, 0.0))) > 1.0, "the lip does not hold the label"


def test_a_long_text_stays_on_the_label():
    lx, ly = storage_bin.label_size()
    lo, hi = bbox(label_of("A very long text that never fits on one small label"))
    assert hi[0] - lo[0] == pytest.approx(lx) and hi[1] - lo[1] == pytest.approx(ly)


def test_a_bin_without_room_for_a_pocket_has_no_label():
    narrow = dict(length=80.0, width=20.0, height=40.0, wall=5.0, floor=1.2)
    assert storage_bin.label_size(**narrow) is None
    with pytest.raises(ValueError):
        storage_bin.bin_label("Bits", **narrow)


def test_the_dialog_makes_the_label_with_the_bin(win, monkeypatch):
    def fill(dlg):
        dlg.label.setChecked(True)
        dlg.label_text.setText("Screws")
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.bin_hanger = False
    win.storage_bin_dialog()
    assert pump_until(lambda: len(win.doc.groups) == 2), "no label next to the bin"
    assert app_module.Settings().bin_label_text == "Screws"
    win.settings.bin_label_text = ""             # the other tests make a bin without a label
    win.settings.save()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
