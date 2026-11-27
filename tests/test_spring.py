"""Zigzag and spiral springs: a strip folded flat that gives under compression, and a strip wound
flat that gives when its hub is turned.

`spring.zigzag` builds the folded strip as one closed volume that lies flat on `z = 0` and is
centred in x and y: `folds` straight beams of `length` by `thickness` millimetres, stacked with a
gap `g`, joined alternately right and left by half rounds, and finished with a screw pad at each
end. `spring.spiral` winds the same kind of strip round a hub, gives the hub a D-shaped bore and
the outer end a square lug. Both springs are printed lying flat, so the only faces that look down
are their bottom faces on the bed.

These tests check that both springs come out as one printable volume, that the numbers in
`sizes` are the numbers of that volume, and that the stiffness follows the beam formula. A size
outside the range of a printable spring becomes the nearest size that prints; a spiral whose strip
would reach its own middle is refused. The place in the interface is checked too: the action stands
between Spool Adapter and Spur Gear, one part is one undo step, two parts are placed ten
millimetres apart, the dialog starts at the old values and remembers the new ones, and a real slice
says that the spring needs no support.

Run with:  python -m pytest tests/test_spring.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import cura_profiles, slicer, spring  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: the fields of the dialog, as the window remembers them
BASE = dict(kind="zigzag", material="PLA", folds=6, length=20.0, thickness=1.2, gap=2.5,
            height=5.0, hole=3.2, turns=4, inner=4.0, bore=5.0, lug=3.2)
OPTIONS = dict(BASE)
ZIGZAG_OPTIONS = dict(BASE)
SPIRAL_OPTIONS = dict(BASE, kind="spiral")

#: the fields of each builder, without the fields that belong to the other kind
ZIGZAG = {k: BASE[k] for k in ("folds", "length", "thickness", "gap", "height", "hole")}
SPIRAL = {k: BASE[k] for k in ("turns", "inner", "thickness", "gap", "height", "bore", "lug")}

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: a probe small enough to stand inside a spring hole and large enough to be worth a test
PROBE = mf.Manifold.cylinder(0.06, 0.25, 0.25, 8)
PROBE_VOLUME = PROBE.volume()


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=30000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def zigzag(**kw):
    """A zigzag spring with the fields of the dialog, with these answers over the defaults."""
    return spring.zigzag(**dict(ZIGZAG, **kw))


def spiral(**kw):
    """A spiral spring with the fields of the dialog, with these answers over the defaults."""
    return spring.spiral(**dict(SPIRAL, **kw))


def numbers(**kw):
    """The numbers that belong to those fields, as the dialog shows them under them."""
    return spring.sizes(**dict(OPTIONS, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def solid(m, x, y, z):
    """Whether a small probe at (x, y, z) falls in material."""
    return ((PROBE.translate((x, y, z))) - m).volume() < 0.02 * PROBE_VOLUME


def facets(m):
    """The unit normals, the centroids and the areas of the triangles of a manifold."""
    mesh = m.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-12
    return n[ok] / length[ok][:, None], (a + b + c)[ok] / 3.0, length[ok] / 2.0


def overhang_area(m):
    """The area of the faces above the bed that look down; a flat spring has none of those."""
    n, centre, area = facets(m)
    keep = (n[:, 2] < -0.1) & (centre[:, 2] > 1e-5)
    return float(area[keep].sum())


def _mesh(m):
    placed = m.to_mesh()
    return np.asarray(placed.vert_properties)[:, :3], np.asarray(placed.tri_verts), 0


def spiral_length_by_integral(inner, pitch, turns, steps=2000):
    """The length of the middle line, found by adding up the little pieces of a long spiral."""
    b = pitch / (2.0 * math.pi)
    theta = 2.0 * math.pi * turns

    def integrand(a):
        r = inner + b * a
        return math.sqrt(r * r + b * b)

    total = 0.0
    for i in range(steps):
        a0 = theta * i / steps
        a1 = theta * (i + 1) / steps
        total += (a1 - a0) * 0.5 * (integrand(a0) + integrand(a1))
    return total


def set_settings(win, **over):
    """Every value the spring dialog keeps, at the defaults of this file with these answers over
    it: no test may depend on what another test left in the settings."""
    for key, value in dict(OPTIONS, **over).items():
        setattr(win.settings, "spring_" + key, value)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    set_settings(w)
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the folded strip

def test_the_default_zigzag_is_one_valid_volume_that_lies_on_the_bed():
    m, n = zigzag(), numbers()
    assert m.status() == mf.Error.NoError, "the spring is not a valid volume"
    assert len(m.decompose()) == 1, "the spring fell apart"
    assert m.genus() == 2, "the spring is not bored through at both ends"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(n["h"])
    assert hi[0] - lo[0] == pytest.approx(n["width"], abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(n["height"], abs=1e-6)
    assert overhang_area(m) < 1e-6, "the folded strip prints with an overhang"


def test_the_zigzag_holes_go_through_the_plastic_at_both_ends():
    m, n = zigzag(), numbers(kind="zigzag")
    centre = 0.5 * (n["strip"] + n["pad"])
    for y in (-centre, centre):
        assert not solid(m, 0.0, y, 0.5 * n["h"]), "the screw hole is not open"
    inside = 0.5 * n["strip"] + 0.5 * spring.PAD_WALL
    assert solid(m, 0.0, -inside, 0.5 * n["h"]), "the end pad is too small for its hole"
    assert solid(m, 0.0, inside, 0.5 * n["h"]), "the end pad is too small for its hole"


def test_a_zigzag_without_a_screw_hole_is_one_flat_piece_of_plastic():
    m = zigzag(hole=0.0)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.genus() == 0, "a spring without holes still has a hole"
    assert overhang_area(m) < 1e-6


@pytest.mark.parametrize("kw", [
    dict(folds=2, length=5.0, thickness=0.6, gap=1.0, height=2.0, hole=0.0),
    dict(folds=8, length=35.0, thickness=2.0, gap=4.0, height=8.0, hole=6.0),
])
def test_the_zigzag_keeps_its_beams_and_its_holes_at_every_size(kw):
    m, n = zigzag(**kw), numbers(**kw)
    assert m.status() == mf.Error.NoError and m.volume() > 0.0
    assert len(m.decompose()) == 1, f"{kw}: the spring fell apart"
    assert m.genus() == (2 if n["hole"] > 0.0 else 0)
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(n["h"])
    assert hi[0] - lo[0] == pytest.approx(n["width"], abs=0.05)
    assert hi[1] - lo[1] == pytest.approx(n["height"], abs=0.05)
    assert overhang_area(m) < 1e-6


# ------------------------------------------------------------------ the wound strip

def test_the_default_spiral_is_one_valid_volume_that_lies_on_the_bed():
    m, n = spiral(), numbers(kind="spiral")
    assert m.status() == mf.Error.NoError, "the spiral is not a valid volume"
    assert len(m.decompose()) == 1, "the spiral fell apart"
    assert m.genus() == 2, "the spiral is not bored at the hub and at the lug"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(n["h"])
    assert hi[0] - lo[0] == pytest.approx(n["width"], abs=0.05)
    assert hi[1] - lo[1] == pytest.approx(n["height"], abs=0.05)
    assert overhang_area(m) < 1e-6, "the spiral prints with an overhang"


@pytest.mark.parametrize("kw,genus", [
    (dict(bore=0.0, lug=0.0), 0),           # a solid hub and no lug: one piece without a hole
    (dict(bore=3.0, lug=0.0), 1),           # only the D bore goes through
    (dict(bore=0.0, lug=2.5), 1),           # only the screw hole in the lug goes through
])
def test_the_spiral_has_exactly_the_holes_it_was_asked_for(kw, genus):
    m = spiral(**kw)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.genus() == genus, f"{kw}: the spring has the wrong number of holes"
    assert overhang_area(m) < 1e-6


@pytest.mark.parametrize("kw", [
    dict(turns=1, inner=1.0, thickness=0.6, gap=1.0, height=2.0, bore=0.0, lug=0.0),
    dict(turns=6, inner=8.0, thickness=1.5, gap=3.0, height=7.0, bore=4.0, lug=2.0),
    dict(turns=2, inner=3.0, thickness=0.8, gap=1.5, height=3.0, bore=0.0, lug=0.0),
])
def test_the_spiral_keeps_its_shape_and_its_numbers_at_every_size(kw):
    m, n = spiral(**kw), numbers(kind="spiral", **kw)
    assert m.status() == mf.Error.NoError and m.volume() > 0.0
    assert len(m.decompose()) == 1, f"{kw}: the spring fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0)
    assert hi[2] == pytest.approx(n["h"])
    assert hi[0] - lo[0] == pytest.approx(n["width"], abs=0.05)
    assert hi[1] - lo[1] == pytest.approx(n["height"], abs=0.05)
    assert overhang_area(m) < 1e-6


# ------------------------------------------------------------------ the numbers under the fields

def test_the_zigzag_stiffness_follows_the_cantilever_formula():
    n, n2, n4 = numbers(), numbers(thickness=2.4), numbers(folds=12)
    expected = 3500.0 * n["h"] * n["t"] ** 3 / (n["folds"] * n["L"] ** 3)
    travel = n["folds"] * spring.STRAIN * n["L"] ** 2 / (3.0 * n["t"])
    assert n["k"] == pytest.approx(expected)
    assert n["travel"] == pytest.approx(travel)
    assert n["force"] == pytest.approx(n["k"] * n["travel"])
    assert n2["k"] == pytest.approx(8.0 * n["k"]), "thickness does not count to the third power"
    assert n4["k"] == pytest.approx(0.5 * n["k"]), "doubling the beams does not halve the force"


def test_the_spiral_stiffness_follows_the_length_of_the_strip():
    n, n2, n3 = numbers(kind="spiral"), numbers(kind="spiral", thickness=2.4, gap=1.3), \
        numbers(kind="spiral", height=10.0)
    expected = 3500.0 * n["h"] * n["t"] ** 3 / (12.0 * n["length"])
    assert n["kt"] == pytest.approx(expected)
    assert n["kt_deg"] == pytest.approx(expected * math.pi / 180.0)
    assert n2["kt"] == pytest.approx(8.0 * n["kt"])
    assert n3["kt"] == pytest.approx(2.0 * n["kt"])


def test_the_length_of_the_spiral_is_the_length_of_its_middle_line():
    got = spring.spiral_length(4.0, 3.7, 4)
    assert got == pytest.approx(spiral_length_by_integral(4.0, 3.7, 4), rel=1e-3)
    assert spring.spiral_length(4.0, 3.7, 5) > got
    assert spring.spiral_length(6.0, 3.7, 4) > got


def test_a_spring_kind_and_a_material_are_checked_before_anything_is_built():
    with pytest.raises(ValueError) as exc:
        spring.sizes(kind="coil")
    assert "kind" in str(exc.value).lower()
    with pytest.raises(ValueError) as exc:
        spring.sizes(material="wood")
    assert "material" in str(exc.value).lower()
    with pytest.raises(ValueError):
        spring.spring(kind="coil")


@pytest.mark.parametrize("bad", [None, "", "many", math.nan, math.inf])
def test_a_count_that_is_not_a_whole_number_is_refused(bad):
    with pytest.raises(ValueError) as exc:
        spring.sizes(folds=bad)
    assert "whole" in str(exc.value).lower()


@pytest.mark.parametrize("bad", [None, "", "thick", math.nan, math.inf])
def test_a_size_that_is_not_a_number_is_refused(bad):
    with pytest.raises(ValueError) as exc:
        spring.sizes(thickness=bad)
    assert "number" in str(exc.value).lower()


def test_an_outside_size_becomes_the_nearest_size_that_prints():
    small = spring.sizes(kind="zigzag", folds=-10, length=-5.0, thickness=0.1, gap=0.2,
                         height=0.5, hole=-1.0)
    assert small["folds"] == spring.FOLDS_MIN
    assert small["L"] == pytest.approx(spring.LENGTH_MIN)
    assert small["t"] == pytest.approx(spring.THICKNESS_MIN)
    assert small["g"] == pytest.approx(spring.GAP_MIN)
    assert small["h"] == pytest.approx(spring.HEIGHT_MIN)
    assert small["hole"] == 0.0

    big = spring.sizes(kind="zigzag", folds=1000, length=1000.0, thickness=100.0, gap=100.0,
                       height=100.0, hole=100.0)
    assert big["folds"] == spring.FOLDS_MAX
    assert big["L"] == pytest.approx(spring.LENGTH_MAX)
    assert big["t"] == pytest.approx(spring.THICKNESS_MAX)
    assert big["g"] == pytest.approx(spring.GAP_MAX)
    assert big["h"] == pytest.approx(spring.HEIGHT_MAX)
    assert big["hole"] == pytest.approx(spring.HOLE_MAX)

    spiral = spring.sizes(kind="spiral", turns=1000, inner=-10.0, thickness=spring.THICKNESS_MIN,
                          gap=spring.GAP_MIN, bore=100.0, lug=100.0)
    assert spiral["turns"] == spring.TURNS_MAX
    assert spiral["inner"] == pytest.approx(spring.INNER_MIN)
    assert spiral["lug"] == pytest.approx(spring.LUG - 2.0 * spring.LUG_WALL)


def test_a_count_that_is_almost_whole_becomes_the_whole_number_near_it():
    assert spring.sizes(folds=2.6)["folds"] == 3
    assert spring.sizes(folds=4.4)["folds"] == 4
    assert spring.sizes(kind="spiral", turns=4.6)["turns"] == 5


def test_the_spiral_refuses_a_strip_that_would_reach_its_own_middle():
    with pytest.raises(ValueError) as exc:
        spiral(inner=1.0)
    assert "middle" in str(exc.value).lower()
    with pytest.raises(ValueError):
        numbers(kind="spiral", inner=1.0)


def test_a_large_bore_and_a_large_lug_hole_become_the_largest_ones_that_keep_a_wall():
    n = spring.sizes(kind="spiral", bore=20.0, lug=6.0)
    assert n["bore"] == pytest.approx(2.0 * (n["hub"] - spring.HUB_WALL))
    assert n["lug"] == pytest.approx(spring.LUG - 2.0 * spring.LUG_WALL)
    tiny = spring.sizes(kind="spiral", inner=1.0, thickness=0.6, gap=1.0, bore=5.0)
    assert tiny["bore"] == pytest.approx(2.0 * (tiny["hub"] - spring.HUB_WALL))
    assert 0.0 < tiny["bore"] <= 0.25


def test_spring_dispatches_to_the_right_builder_and_leaves_out_the_other_fields():
    folded = spring.spring(**ZIGZAG_OPTIONS)
    wound = spring.spring(**SPIRAL_OPTIONS)
    assert folded.volume() == pytest.approx(zigzag().volume())
    assert wound.volume() == pytest.approx(spiral().volume())


# ------------------------------------------------------------------ a real slice

@needs_engine
@pytest.mark.parametrize("part", [zigzag, spiral])
def test_a_spring_slices_without_support(tmp_path, part):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v, m = p.default_variant(), p.default_material(p.default_variant())
    stacks = p.stacks(v, m, "standard", {"infill_sparse_density": 0.0,
                                         "support_enabled": False})
    out = tmp_path / "spring.gcode"
    slicer.slice_meshes(stacks, [_mesh(part())], str(out))
    text = out.read_text(encoding="utf-8")
    assert ";TYPE:SUPPORT" not in text, "the spring needs support it is not meant to have"
    assert text.count(";LAYER:") > 5, "barely anything was sliced"


# ------------------------------------------------------------------ in the window

def test_the_spring_comes_into_the_model_as_one_undo_step(win):
    wanted = spring.spring(**ZIGZAG_OPTIONS)
    steps = len(win.doc.undo_stack)
    win.make_spring_model(ZIGZAG_OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no spring in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the spring is not one undo step"
    group = win.doc.groups[0]
    assert group.name == "Zigzag spring"
    assert pump_until(lambda: group.ents.to_manifold().volume() > 0.0)
    got = group.ents.to_manifold()
    assert got.status() == mf.Error.NoError
    assert got.volume() == pytest.approx(wanted.volume(), rel=1e-3)
    lo, hi = bbox(got)
    assert lo[2] == pytest.approx(0.0), "the spring does not lie on the bed"
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the spring is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by), "the spring is not on the bed"
    assert pump_until(lambda: "zigzag spring" in win.hint.text().lower()), win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_the_spiral_comes_in_next_to_the_first_spring(win):
    win.make_spring_model(ZIGZAG_OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no zigzag spring in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    first = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    win.make_spring_model(SPIRAL_OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "the spiral spring is missing"
    assert pump_until(lambda: all(g.bbox() is not None for g in win.doc.groups))
    second = np.asarray(win.doc.groups[1].bbox(), dtype=float)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the parts are not 10 mm apart"
    assert [g.name for g in win.doc.groups] == ["Zigzag spring", "Spiral spring"]
    assert pump_until(lambda: win.doc.groups[1].ents.to_manifold().genus() == 2)


def test_the_spring_stands_between_the_spool_and_the_gear_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.parts_category("Mechanical"))]
    assert labels.index("Spring") == labels.index("Spool Adapter") + 1
    assert labels.index("Spring") == labels.index("Spur Gear") - 1
    assert win.a_spring in win.parts_category("Mechanical").actions()


def test_a_spring_that_cannot_be_made_leaves_the_model_as_it_was(win):
    win.make_spring_model(dict(BASE, kind="coil"))
    assert pump_until(lambda: "spring failed" in win.hint.text().lower()), win.hint.text()
    assert win.doc.groups == []
    win.view.set_status("")
    win.make_spring_model(dict(SPIRAL_OPTIONS, inner=1.0))
    assert pump_until(lambda: "spiral spring failed" in win.hint.text().lower()), win.hint.text()
    assert "inner" in win.hint.text().lower()
    assert win.doc.groups == [], "a spring that could not be made still came into the model"


def test_the_dialog_starts_at_a_spring_when_the_stored_choice_is_gone(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["start"] = [dlg.kind.currentData(), dlg.material.currentData(),
                         int(dlg.folds.value()), round(dlg.length.value(), 2),
                         round(dlg.thickness.value(), 2), round(dlg.gap.value(), 2),
                         round(dlg.height.value(), 2), round(dlg.hole.value(), 2),
                         int(dlg.turns.value()), round(dlg.inner.value(), 2),
                         round(dlg.bore.value(), 2), round(dlg.lug.value(), 2)]
        seen["info"] = (dlg.info.text(), dlg.info.styleSheet())
        seen["ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        return QDialog.Rejected

    set_settings(win, kind="coil", material="wood", folds=999, length=999.0, thickness=99.0,
                 gap=99.0, height=99.0, hole=99.0, turns=999, inner=-10.0, bore=100.0, lug=100.0)
    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    before = len(win.doc.groups)
    assert win.spring_dialog() is None
    assert len(win.doc.groups) == before, "a cancelled dialog still built a spring"
    assert seen["start"] == ["zigzag", "PLA", spring.FOLDS_MAX, spring.LENGTH_MAX,
                             spring.THICKNESS_MAX, spring.GAP_MAX, spring.HEIGHT_MAX,
                             spring.HOLE_MAX, spring.TURNS_MAX, spring.INNER_MIN, spring.BORE_MAX,
                             spring.LUG]
    assert seen["info"][0].startswith("<i>") and seen["info"][1] == "", seen["info"]
    assert seen["ok"], "a spring that prints was blocked: " + seen["info"][0]


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [dlg.kind.currentData(), dlg.material.currentData(),
                         int(dlg.folds.value()), round(dlg.length.value(), 2),
                         round(dlg.thickness.value(), 2), round(dlg.gap.value(), 2),
                         round(dlg.height.value(), 2), round(dlg.hole.value(), 2),
                         int(dlg.turns.value()), round(dlg.inner.value(), 2),
                         round(dlg.bore.value(), 2), round(dlg.lug.value(), 2)]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        seen["info"] = (dlg.info.text(), dlg.info.styleSheet())
        seen["ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        seen["spiral_fields"] = [dlg.folds.isEnabled(), dlg.length.isEnabled(),
                                 dlg.hole.isEnabled(), dlg.turns.isEnabled(),
                                 dlg.inner.isEnabled(), dlg.bore.isEnabled(),
                                 dlg.lug.isEnabled()]
        dlg.kind.setCurrentIndex(dlg.kind.findData("zigzag"))
        seen["zigzag_fields"] = [dlg.folds.isEnabled(), dlg.length.isEnabled(),
                                 dlg.hole.isEnabled(), dlg.turns.isEnabled(),
                                 dlg.inner.isEnabled(), dlg.bore.isEnabled(),
                                 dlg.lug.isEnabled()]
        dlg.kind.setCurrentIndex(dlg.kind.findData("spiral"))
        dlg.material.setCurrentIndex(dlg.material.findData("TPU 95A"))
        dlg.turns.setValue(4)
        dlg.inner.setValue(5.5)
        dlg.thickness.setValue(1.8)
        dlg.gap.setValue(2.0)
        dlg.height.setValue(4.0)
        dlg.bore.setValue(2.0)
        dlg.lug.setValue(1.5)
        seen["new"] = dlg.options()
        seen["new_ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        return QDialog.Accepted

    set_settings(win, kind="spiral", material="Nylon", folds=5, length=15.0, thickness=1.5,
                 gap=2.0, height=4.0, hole=0.0, turns=3, inner=5.0, bore=3.0, lug=2.0)
    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.spring_dialog()
    assert seen["title"] == "Spring"
    assert seen["start"] == ["spiral", "Nylon", 5, 15.0, 1.5, 2.0, 4.0, 0.0, 3, 5.0, 3.0, 2.0]
    assert seen["spins"] == 8, "every length field but the counts carries its millimetres"
    assert seen["info"][0].startswith("<i>") and seen["info"][1] == "", seen["info"]
    assert "Nylon" in seen["info"][0], seen["info"]
    assert seen["ok"] and seen["new_ok"], "a spring that can be made was blocked"
    assert seen["spiral_fields"] == [False, False, False, True, True, True, True]
    assert seen["zigzag_fields"] == [True, True, True, False, False, False, False]
    assert seen["new"]["kind"] == "spiral" and seen["new"]["material"] == "TPU 95A"
    assert seen["new"]["turns"] == 4 and seen["new"]["inner"] == pytest.approx(5.5)
    assert seen["new"]["bore"] == pytest.approx(2.0)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no spring in the model"
    assert win.doc.groups[0].name == "Spiral spring"
    assert pump_until(lambda: win.doc.groups[0].ents.to_manifold().volume() > 0.0)
    made = win.doc.groups[0].ents.to_manifold()
    assert made.volume() == pytest.approx(spiral(turns=4, inner=5.5, thickness=1.8, gap=2.0,
                                                 height=4.0, bore=2.0, lug=1.5).volume(),
                                          rel=1e-3)
    back = app_module.Settings()
    assert (back.spring_kind, back.spring_material) == ("spiral", "TPU 95A")
    assert back.spring_turns == 4
    assert back.spring_inner == pytest.approx(5.5)
    assert back.spring_thickness == pytest.approx(1.8)
    assert back.spring_gap == pytest.approx(2.0)
    assert back.spring_height == pytest.approx(4.0)
    assert back.spring_bore == pytest.approx(2.0)
    assert back.spring_lug == pytest.approx(1.5)


def test_the_dialog_refuses_a_spiral_that_reaches_its_middle_and_grows_a_bore_too_far(win,
                                                                                     monkeypatch):
    seen = {}

    def fill(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData("spiral"))
        seen["calm"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.inner.setValue(spring.INNER_MIN)
        seen["refused"] = (dlg.bb.button(QDialogButtonBox.Ok).isEnabled(), dlg.info.text())
        dlg.inner.setValue(4.0)
        dlg.bore.setValue(10.0)
        seen["bore"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.bore.setValue(0.0)
        dlg.lug.setValue(5.0)
        seen["lug"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.lug.setValue(0.0)
        seen["quiet"] = (dlg.info.text(), dlg.info.styleSheet())
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    before = len(win.doc.groups)
    assert win.spring_dialog() is None
    assert len(win.doc.groups) == before, "a cancelled dialog still built a spring"
    assert seen["calm"][1] == "", seen["calm"]
    assert not seen["refused"][0], "a spiral that reaches its middle was not refused"
    assert "inner radius" in seen["refused"][1].lower(), seen["refused"][1]
    assert "Ø10.00" in seen["bore"][0] and "6.80" in seen["bore"][0], seen["bore"]
    assert seen["bore"][1], "a bore that was made smaller is not shown as a warning"
    assert "3.60" in seen["lug"][0] and seen["lug"][1], seen["lug"]
    assert seen["quiet"][1] == "" and seen["quiet"][0].startswith("<i>"), seen["quiet"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
