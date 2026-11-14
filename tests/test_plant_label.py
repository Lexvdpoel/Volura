"""Plant labels: a stake for a pot whose flat head carries the name of the plant.

`plant_label.plant_label` gives the label as one closed volume on `z = 0`: a head of `width` by
`head` millimetres whose four corners round with `CORNER`, an 8 millimetre wide `stake` hanging
below it that ends in a point, and the name standing `raised` millimetres out of the top face of
the head in letters half as high as the head and shrunk to fit it.

These tests weigh the label against that: one volume on the bed and centred, `head + stake` by
`width` outside; the stake full width in its middle and narrower near its end; letters that raise
the top of the head to `thickness + raised` and that a probe over one of them finds, inside the
head; no text, no plate higher than `thickness`; a long name shrunk until it fits; no face that is
not vertical or horizontal, so nothing overhangs; every range reached by clamping and every number
that is no number refused; and the dialog that builds the label and remembers it.

Run with:  python -m pytest tests/test_plant_label.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import fonts, geom, plant_label  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the letters need it

BASE = dict(text="Basil", width=60.0, head=20.0, stake=80.0, thickness=2.4, raised=0.8)
#: the rounded corners and the letters are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4


def part(**kw) -> mf.Manifold:
    return plant_label.plant_label(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return plant_label.sizes(**dict(BASE, **kw))


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def slab(m, z0, z1) -> float:
    """The volume of the part between the heights `z0` and `z1`."""
    return (m ^ mf.Manifold.cube([1e4, 1e4, z1 - z0]).translate((-5e3, -5e3, z0))).volume()


def ink(m, s) -> mf.Manifold:
    """The letters only: everything of the label that stands above the plate."""
    return m ^ mf.Manifold.cube([1e4, 1e4, 10.0]).translate(
        (-5e3, -5e3, s["thickness"] + 1e-3))


def normals(m):
    """The unit normals and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


def frame(s) -> dict:
    """The faces of the label, in the millimetres the outline is drawn in, and how it lies.

    The head is drawn from `y = 0` up to `y = head` and the stake hangs from it to `y = -stake`,
    and the finished label is centred, so `off` is what moves such a point onto the bed.
    """
    return {"off": (0.0, 0.5 * (s["stake"] - s["head"])),
            "tip": -s["stake"], "half": 0.5 * plant_label.STAKE_WIDTH,
            "middle": 0.5 * s["head"] + 0.5 * (s["stake"] - s["head"])}


def on_bed(p, x, y):
    """A point of the outline, moved onto the bed."""
    return x + p["off"][0], y + p["off"][1]


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("text", ("", "Basil"))
@pytest.mark.parametrize("size", ((30.0, 12.0, 30.0), (60.0, 20.0, 80.0), (120.0, 40.0, 200.0)))
def test_the_label_is_one_volume_that_lies_on_the_bed_and_is_centred(text, size):
    width, head, stake = size
    kw = dict(text=text, width=width, head=head, stake=stake)
    m = part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the label fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the label does not lie on the bed"
    assert np.allclose(lo[:2], -0.5 * np.array([width, head + stake]), atol=ABS), \
        "not centred in x and y"
    assert np.allclose(hi[:2], 0.5 * np.array([width, head + stake]), atol=ABS), \
        "not the size asked"


def test_the_outside_is_its_head_plus_its_stake_by_its_width():
    s, m = numbers(text=""), part(text="")
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["width"], abs=ABS), "not as wide as asked"
    assert hi[1] - lo[1] == pytest.approx(s["length"], abs=ABS), "not as long as head plus stake"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "not as thick as asked"


# ------------------------------------------------------------------ the stake

def test_the_stake_ends_in_a_point():
    """The strip is full width in the middle of the stake and half as wide halfway down its point,
    so it is easy to push into the soil."""
    s, m = numbers(), part()
    p = frame(s)
    shaft = 0.5 * (plant_label.POINT - s["stake"])       # halfway down the straight part
    assert touches(m, post(*on_bed(p, 0.9 * p["half"], shaft), 0.5, s["thickness"] - 0.5)), \
        "the stake is narrower than it is wide"
    point = p["tip"] + 0.5 * plant_label.POINT           # halfway down the point
    assert touches(m, post(*on_bed(p, 0.4 * p["half"], point), 0.5, s["thickness"] - 0.5)), \
        "nothing is left of the stake on its point"
    assert not touches(m, post(*on_bed(p, 0.8 * p["half"], point), 0.5, s["thickness"] - 0.5)), \
        "the stake is not pointed: it is full width halfway down its point"


# ------------------------------------------------------------------ the letters

def test_the_name_stands_raised_on_the_head():
    need_fonts()
    s, plain = numbers(), part(text="")
    made = part()
    assert len(made.decompose()) == 1, "the letters cut the label in two"
    lo, hi = bbox(made)
    assert hi[2] - lo[2] == pytest.approx(s["thickness"] + s["raised"], abs=ABS), \
        "the letters do not stand on the head"
    assert slab(made, s["thickness"] + 0.5 * s["raised"], hi[2]) > 0.0, "nothing stands on the head"
    assert slab(plain, s["thickness"] + 0.05, s["thickness"] + s["raised"]) == pytest.approx(
        0.0, abs=1e-6), "a plate without letters has plastic above it"


def test_a_probe_over_a_letter_meets_the_name():
    """A post on the bed of the letters, where one of them stands, is stopped by the name."""
    need_fonts()
    s, m = numbers(), part()
    hi = bbox(m)[1]
    v, _t = geom.mesh_arrays(m)
    top = v[np.abs(v[:, 2] - hi[2]) < 1e-6]
    assert len(top), "nothing stands on top of the head"
    x, y = float(top[0][0]), float(top[0][1])
    assert touches(m, post(x, y, s["thickness"] + 0.5 * s["raised"], hi[2] + 1.0)), \
        "no letter stands where the name should be"


def test_the_letters_stay_inside_the_head():
    """The name fits the head less its margins and is centred on it, so no letter runs over the
    edge of the plate."""
    need_fonts()
    s, m = numbers(), part()
    p = frame(s)
    lo, hi = bbox(ink(m, s))
    room = 0.5 * np.array([s["width"] - plant_label.MARGIN_X, s["head"] - plant_label.MARGIN_Y])
    mid = np.array([0.0, p["middle"]])
    assert np.all(lo[:2] >= mid - room - 1e-3) and np.all(hi[:2] <= mid + room + 1e-3), \
        "the letters run past the head"
    assert np.allclose(0.5 * (lo[:2] + hi[:2]), mid, atol=ABS), "the name is not centred on the head"


@pytest.mark.parametrize("text", ("", "   ", "\n"))
def test_no_text_leaves_a_plain_plate(text):
    s, m = numbers(text=text), part(text=text)
    lo, hi = bbox(m)
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "an empty plate stands higher"
    assert hi[1] - lo[1] == pytest.approx(s["length"], abs=ABS), "an empty plate is another size"
    assert slab(m, s["thickness"] + 0.05, s["thickness"] + plant_label.RAISED_MAX) \
        == pytest.approx(0.0, abs=1e-6), "something stands on an empty plate"


def test_a_long_name_is_shrunk_until_it_fits_the_head():
    need_fonts()
    s, m = numbers(text="ROSEMARY FOR THE KITCHEN WINDOW"), part(
        text="ROSEMARY FOR THE KITCHEN WINDOW")
    made = ink(m, s)
    assert not made.is_empty(), "nothing was raised"
    lo, hi = bbox(made)
    assert hi[0] - lo[0] <= s["width"] - plant_label.MARGIN_X + 1e-3, "the name is too wide"
    assert hi[1] - lo[1] <= s["head"] - plant_label.MARGIN_Y + 1e-3, "the name is too tall"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(text=""), dict(text="VOLURA IN BLOOM"),
                                dict(width=30.0, head=12.0, stake=30.0),
                                dict(width=120.0, head=40.0, stake=200.0, thickness=4.0)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The label is a plate pulled straight up with letters on it, so no face leans and nothing
    needs support."""
    n, area = normals(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the label would overhang"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_label_it_makes():
    s = plant_label.sizes(text="Mint", width=80.0, head=30.0, stake=150.0, thickness=3.0,
                          raised=1.2)
    assert s["text"] == "Mint" and s["width"] == 80.0 and s["head"] == 30.0
    assert s["stake"] == 150.0 and s["thickness"] == 3.0 and s["raised"] == 1.2
    assert s["length"] == pytest.approx(180.0)
    assert set(s) == {"text", "width", "head", "stake", "thickness", "raised", "length"}


def test_the_default_label_is_a_basil_stake():
    s = plant_label.sizes()
    assert s["text"] == "Basil" and s["width"] == 60.0 and s["head"] == 20.0
    assert s["stake"] == 80.0 and s["thickness"] == 2.4 and s["raised"] == 0.8
    assert s["length"] == pytest.approx(100.0)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(width=0.0, head=0.0, stake=0.0, thickness=0.0, raised=0.0)
    assert small["width"] == plant_label.WIDTH_MIN and small["head"] == plant_label.HEAD_MIN
    assert small["stake"] == plant_label.STAKE_MIN
    assert small["thickness"] == plant_label.THICKNESS_MIN
    assert small["raised"] == plant_label.RAISED_MIN
    big = numbers(width=999.0, head=999.0, stake=999.0, thickness=999.0, raised=999.0)
    assert big["width"] == plant_label.WIDTH_MAX and big["head"] == plant_label.HEAD_MAX
    assert big["stake"] == plant_label.STAKE_MAX
    assert big["thickness"] == plant_label.THICKNESS_MAX
    assert big["raised"] == plant_label.RAISED_MAX
    for kw in (dict(width=0.0, head=0.0, stake=0.0, thickness=0.0, raised=0.0, text=""),
               dict(width=999.0, head=999.0, stake=999.0, thickness=999.0, raised=999.0,
                    text="")):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["width"], abs=ABS), "a clamped width was lost"
        assert hi[1] - lo[1] == pytest.approx(s["length"], abs=ABS), "a clamped length was lost"
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "a clamped height was lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("width", "head", "stake", "thickness", "raised"):
        with pytest.raises(ValueError):
            plant_label.plant_label(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            plant_label.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_label_and_remembers_it(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.text.setText("Mint")
        dlg.width.setValue(80.0)
        dlg.head.setValue(30.0)
        dlg.stake.setValue(150.0)
        dlg.thickness.setValue(3.0)
        dlg.raised.setValue(1.2)
        assert "180.0" in dlg.info.text(), "the dialog does not show how long the label is"
        assert "80.0" in dlg.info.text(), "the dialog does not show how wide the head is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.plant_label_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no label in the model"
        assert [g.name for g in w.doc.groups] == ["Mint"], "the group is not named after the text"
        s = plant_label.sizes(text="Mint", width=80.0, head=30.0, stake=150.0, thickness=3.0,
                              raised=1.2)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the label does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["width"], abs=ABS), "not as wide as computed"
        assert hi[1] - lo[1] == pytest.approx(s["length"], abs=ABS), "not as long as computed"
        if fonts.has_fonts():
            assert hi[2] - lo[2] == pytest.approx(s["thickness"] + s["raised"],
                                                  abs=1e-3), "not as high as computed"
        settings = app_module.Settings()
        assert (settings.plant_label_text, settings.plant_label_width,
                settings.plant_label_head, settings.plant_label_stake,
                settings.plant_label_thickness, settings.plant_label_raised) == (
            "Mint", 80.0, 30.0, 150.0, 3.0, 1.2)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Plant Label") == labels.index("Filament Clip") + 1
    finally:
        for key, value in (("plant_label_text", "Basil"), ("plant_label_width", 60.0),
                           ("plant_label_head", 20.0), ("plant_label_stake", 80.0),
                           ("plant_label_thickness", 2.4), ("plant_label_raised", 0.8)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
