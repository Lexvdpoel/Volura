"""Spool adapters: a hub that fills the hole of a filament spool, printed flat on its flange.

`spool_adapter.spool_adapter` gives the adapter as one closed volume that stands on its flange on
`z = 0` and is centred in x and y: a hub of `hub` millimetres that goes `depth` millimetres into
the hole of the spool, on a flange of `hub + 2 * lip` millimetres that keeps it from sliding in,
with a bore for a rod of `rod` millimetres or a pocket for a 608 bearing and an axle hole under
it. The hub is 0.4 mm under the hole and its top edge is chamfered, and `spokes`
cuts the plastic out of the hub between the ring around the bore and the ring inside its outside.

These tests weigh the part against that: it is one piece that stands on the bed and is as wide as
its flange, the hub fits a hole of `hub` and a hole of a millimetre smaller does not fit it, the
flange is wider than the hub and does not go into the hole, a rod of `rod` turns in the bore and a
thicker one does not, a 608 bearing presses into its pocket with the axle through below it, the
spokes cut plastic away and leave the hub in one piece, a bore too big for the hole makes the hub
larger instead of thin, every range is reached by clamping and nothing hangs over, because the
adapter stands on its flange and every wall goes straight up from it.

Run with:  python -m pytest tests/test_spool_adapter.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, spool_adapter  # noqa: E402
from volura.spool_adapter import FLANGE  # noqa: E402

BASE = dict(hub=53.0, depth=12.0, lip=4.0, rod=8.0, bearing=False, spokes=5)
#: the hub and the pockets are polygons, so a part is at most this far off
ABS = 0.05
#: mm of ring that a probe of the wall of a hole is thick
PROBE = 0.1


def part(**kw) -> mf.Manifold:
    return spool_adapter.spool_adapter(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return spool_adapter.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a rod that does not fit is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def hole_probe(r, z0, z1) -> mf.Manifold:
    """A thin ring at the radius `r`, between the heights `z0` and `z1`: the wall of a hole."""
    cs = (mf.CrossSection.circle(r + PROBE, spool_adapter.SEGMENTS)
          - mf.CrossSection.circle(max(0.0, r - PROBE), spool_adapter.SEGMENTS))
    return mf.Manifold.extrude(cs, z1 - z0).translate((0.0, 0.0, z0))


def rod_probe(d, z0, z1) -> mf.Manifold:
    """A cylinder of `d` millimetres along the axis of the adapter, from `z0` to `z1`."""
    return mf.Manifold.cylinder(z1 - z0, 0.5 * d, 0.5 * d, spool_adapter.SEGMENTS) \
        .translate((0.0, 0.0, z0))


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("hub, depth, lip", ((25.0, 5.0, 2.0), (53.0, 12.0, 4.0),
                                             (90.0, 40.0, 10.0)))
@pytest.mark.parametrize("rod, bearing, spokes", ((5.0, False, 3), (8.0, False, 0),
                                                  (30.0, True, 8)))
def test_the_adapter_is_one_piece_that_stands_on_its_flange(hub, depth, lip, rod, bearing,
                                                            spokes):
    kw = dict(hub=hub, depth=depth, lip=lip, rod=rod, bearing=bearing, spokes=spokes)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the adapter fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the adapter does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "the adapter is not as high"
    assert hi[0] - lo[0] == pytest.approx(s["flange"], abs=ABS), "the flange is not as wide"
    assert hi[1] - lo[1] == pytest.approx(s["flange"], abs=ABS), "the flange is not round"
    assert lo[0] == pytest.approx(-0.5 * s["flange"], abs=ABS), "the adapter is not centred in x"
    assert lo[1] == pytest.approx(-0.5 * s["flange"], abs=ABS), "the adapter is not centred in y"


def test_the_flange_is_wider_than_the_hub_and_only_two_millimetres_high():
    """The flange is what keeps the adapter from sliding into the spool: it reaches `lip` past the
    hub all round and stands `FLANGE` millimetres high, and above it there is nothing but hub."""
    s, m = numbers(), part()
    assert s["flange"] == pytest.approx(s["hub"] + 2.0 * s["lip"]), "the flange is not lipped"
    assert s["flange"] > s["hub"], "the flange is not wider than the hub"
    r = 0.25 * (s["hub"] + s["flange"])
    assert touches(m, hole_probe(r, 0.2, FLANGE - 0.2)), "the flange is not there"
    assert not touches(m, hole_probe(r, FLANGE + 0.2, s["height"])), \
        "something stands beside the hub above the flange"


# ------------------------------------------------------------------ the hub

@pytest.mark.parametrize("hub", (25.0, 53.0, 90.0))
def test_the_hub_fits_the_hole_of_the_spool(hub):
    """The hub is `HUB_PLAY` under the hole, so it goes in by hand, and a hole a millimetre
    smaller is not wide enough for it anywhere along the hub."""
    s, m = numbers(hub=hub), part(hub=hub)
    assert s["hub"] == pytest.approx(hub - spool_adapter.HUB_PLAY)
    assert not touches(m, hole_probe(0.5 * hub, FLANGE + 0.5, s["height"] - 0.2)), \
        "the hub does not fit a hole of that size"
    assert touches(m, hole_probe(0.5 * hub - 1.0, FLANGE + 0.5, s["height"] - 0.2)), \
        "the hub is more than its play under the hole"


@pytest.mark.parametrize("depth", (5.0, 12.0, 40.0))
def test_the_hub_is_as_long_as_asked_and_the_flange_lies_under_it(depth):
    s, m = numbers(depth=depth), part(depth=depth)
    assert s["height"] == pytest.approx(FLANGE + depth), "the hub does not stand on the flange"
    lo, hi = bbox(m)
    assert hi[2] - lo[2] == pytest.approx(depth + FLANGE, abs=ABS)
    assert touches(m, hole_probe(0.5 * s["hub"] - 0.5, FLANGE + 0.2, s["height"] - 0.2)), \
        "the hub is not there"
    assert not touches(m, hole_probe(0.5 * s["hub"] - 0.5, s["height"] + 0.2,
                                     s["height"] + 1.0)), "the hub stands higher than its depth"


def test_the_top_edge_of_the_hub_is_chamfered():
    """The hub is led into the hole by a chamfer of a millimetre: right under the top there is no
    plastic against the wall of the hole, but a millimetre lower there is."""
    s, m = numbers(), part()
    r = 0.5 * s["hub"] - 0.2
    assert not touches(m, hole_probe(r, s["height"] - 0.1, s["height"])), \
        "the hub is not chamfered at its top"
    assert touches(m, hole_probe(r, s["height"] - 1.4, s["height"] - 1.0)), \
        "the chamfer eats deeper than it should"


# ------------------------------------------------------------------ the bore

@pytest.mark.parametrize("rod", (5.0, 8.0, 30.0))
def test_the_bore_takes_the_rod_it_is_made_for(rod):
    """The bore is `ROD_PLAY` wider than the rod, so the rod turns in it and a rod that is half a
    millimetre fatter than asked for does not go through."""
    s, m = numbers(rod=rod), part(rod=rod)
    assert s["bore"] == pytest.approx(rod + spool_adapter.ROD_PLAY)
    assert not touches(m, rod_probe(rod, -1.0, s["height"] + 1.0)), "the rod does not fit"
    assert touches(m, rod_probe(rod + 0.5, -1.0, s["height"] + 1.0)), \
        "the bore is wider than its play"


def test_a_bearing_pocket_sits_on_top_of_the_axle_hole():
    """With `bearing` the top of the hub is a pocket of `BEARING_DEPTH` for a 608 bearing and all
    that is left through the middle is the hole the axle goes through."""
    s, m = numbers(bearing=True), part(bearing=True)
    assert s["pocket"] == pytest.approx(spool_adapter.BEARING_D + spool_adapter.BEARING_PLAY)
    assert s["bore"] == pytest.approx(spool_adapter.AXLE_D)
    top = s["height"] - spool_adapter.BEARING_DEPTH
    assert not touches(m, rod_probe(spool_adapter.BEARING_D, top, s["height"] + 1.0)), \
        "a 608 bearing does not fit the pocket"
    assert not touches(m, rod_probe(spool_adapter.AXLE_D, -1.0, s["height"] + 1.0)), \
        "the axle does not go through"
    assert touches(m, rod_probe(spool_adapter.AXLE_D + 0.4, -1.0, top)), \
        "the axle hole is wider than the bearing's axle"


def test_an_adapter_for_a_rod_has_no_bearing_pocket():
    s, m = numbers(), part()
    assert s["pocket"] == 0.0
    top = s["height"] - spool_adapter.BEARING_DEPTH
    assert touches(m, rod_probe(spool_adapter.BEARING_D, top, s["height"] - 0.2)), \
        "the hub is bored out without being asked for it"


@pytest.mark.parametrize("hub, rod, bearing", ((25.0, 30.0, False), (25.0, 8.0, True),
                                               (35.0, 30.0, False)))
def test_a_bore_too_big_for_the_hole_makes_the_hub_larger(hub, rod, bearing):
    """A rod or a bearing that leaves less than `MIN_WALL` to the outside of the hub grows the hub
    instead of the wall, so the adapter is never a shell that splits on the spool."""
    s, m = numbers(hub=hub, rod=rod, bearing=bearing), part(hub=hub, rod=rod, bearing=bearing)
    assert s["hub"] >= hub - spool_adapter.HUB_PLAY, "the hub was made smaller"
    assert s["wall"] == pytest.approx(spool_adapter.MIN_WALL), "the wall is not what it promises"
    assert len(m.decompose()) == 1, "the adapter fell apart"
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["hub"] + 2.0 * s["lip"], abs=ABS)


# ------------------------------------------------------------------ the spokes

@pytest.mark.parametrize("spokes", (3, 5, 8))
def test_the_spokes_cut_plastic_out_of_the_hub(spokes):
    """The pockets between the spokes stand on the flange and are open at the top, so a spoked hub
    is a lot less plastic and still one piece that carries the spool round."""
    s, m = numbers(spokes=spokes), part(spokes=spokes)
    assert s["hollow"] is True
    assert m.volume() < part(spokes=0).volume(), "the spokes did not save any plastic"
    assert len(m.decompose()) == 1, f"the hub with {spokes} spokes fell apart"
    assert m.genus() == 1, "the pockets cut the hub open"


def test_a_hub_without_spokes_is_solid():
    """0 spokes is a solid hub; 1 and 2 are too few to hold anything, so they become 3."""
    assert numbers(spokes=0)["spokes"] == 0 and numbers(spokes=0)["hollow"] is False
    assert numbers(spokes=1)["spokes"] == spool_adapter.SPOKES_MIN
    assert numbers(spokes=2)["spokes"] == spool_adapter.SPOKES_MIN
    assert numbers(spokes=0)["wall"] == numbers(spokes=5)["wall"]
    assert len(part(spokes=0).decompose()) == 1


@pytest.mark.parametrize("hub, rod, bearing", ((25.0, 30.0, False), (25.0, 8.0, True)))
def test_a_hub_without_room_between_its_rings_stays_solid(hub, rod, bearing):
    """Where the bore comes within `RING_OUT` of the outside of the hub there is no annulus left
    to cut away, so the hub stays solid instead of falling to pieces."""
    s = numbers(hub=hub, rod=rod, bearing=bearing)
    assert s["wall"] <= 0.5 * (spool_adapter.RING_IN + spool_adapter.RING_OUT)
    assert s["hollow"] is False
    assert part(hub=hub, rod=rod, bearing=bearing).volume() == \
        part(hub=hub, rod=rod, bearing=bearing, spokes=0).volume()


# ------------------------------------------------------------------ printing

def test_nothing_hangs_over_because_the_adapter_stands_on_its_flange():
    """The flange lies on the bed and hub, rings and spokes go straight up from it, so the only
    face that looks down is the one that lies on the bed."""
    m = part(hub=90.0, depth=40.0, lip=10.0, rod=30.0, spokes=8)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -1e-6) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face hangs over, so the adapter would need support"


def test_a_bearing_pocket_is_open_at_the_top():
    """The pocket looks up out of the hub instead of hanging under it, so the adapter prints flat
    on its flange with nothing to support."""
    m = part(bearing=True)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -1e-6) & (c[:, 2] > 1e-6)
    assert not down.any(), "the pocket hangs over inside the hub"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_adapter_and_its_bore():
    s = spool_adapter.sizes(hub=60.0, depth=20.0, lip=6.0, rod=10.0, bearing=False, spokes=6)
    assert s["hub"] == pytest.approx(59.6) and s["depth"] == 20.0 and s["lip"] == 6.0
    assert s["rod"] == 10.0 and s["bearing"] is False and s["spokes"] == 6
    assert s["bore"] == pytest.approx(10.3) and s["pocket"] == 0.0
    assert s["flange"] == pytest.approx(71.6) and s["height"] == pytest.approx(22.0)
    assert s["wall"] == pytest.approx(24.65) and s["hollow"] is True
    assert set(s) == {"hub", "depth", "lip", "rod", "bearing", "spokes", "bore", "pocket",
                      "flange", "height", "wall", "hollow"}


def test_the_default_adapter_fits_a_five_centimetre_spool_hole():
    s = spool_adapter.sizes()
    assert s["hub"] == pytest.approx(52.6) and s["depth"] == 12.0 and s["lip"] == 4.0
    assert s["rod"] == 8.0 and s["bearing"] is False and s["spokes"] == 5
    assert s["bore"] == pytest.approx(8.3) and s["flange"] == pytest.approx(60.6)
    assert s["height"] == pytest.approx(14.0) and s["wall"] > spool_adapter.MIN_WALL


@pytest.mark.parametrize("spokes, want", ((0, 0), (1, 3), (3, 3), (5, 5), (8, 8), (99, 8)))
def test_the_number_of_spokes_follows_what_the_hub_holds(spokes, want):
    assert spool_adapter.sizes(spokes=spokes)["spokes"] == want


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(hub=0.0, depth=0.0, lip=0.0, rod=0.0, spokes=-5)
    assert small["hub"] == pytest.approx(spool_adapter.HUB_MIN - spool_adapter.HUB_PLAY)
    assert small["depth"] == spool_adapter.DEPTH_MIN and small["lip"] == spool_adapter.LIP_MIN
    assert small["rod"] == spool_adapter.ROD_MIN and small["spokes"] == 0
    big = numbers(hub=999.0, depth=999.0, lip=999.0, rod=999.0, spokes=999)
    assert big["hub"] == pytest.approx(spool_adapter.HUB_MAX - spool_adapter.HUB_PLAY)
    assert big["depth"] == spool_adapter.DEPTH_MAX and big["lip"] == spool_adapter.LIP_MAX
    assert big["rod"] == spool_adapter.ROD_MAX and big["spokes"] == spool_adapter.SPOKES_MAX
    lo, hi = bbox(part(hub=0.0, depth=0.0, lip=0.0, rod=0.0, spokes=0))
    assert hi[2] - lo[2] == pytest.approx(FLANGE + spool_adapter.DEPTH_MIN, abs=ABS), \
        "the smallest adapter is not as high as its hub and its flange"
    assert len(part(hub=999.0, depth=999.0, lip=999.0, rod=999.0, spokes=999).decompose()) == 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("hub", "depth", "lip", "rod", "spokes"):
        with pytest.raises(ValueError):
            spool_adapter.spool_adapter(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            spool_adapter.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_an_adapter_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.hub.setValue(60.0)
        dlg.depth.setValue(20.0)
        dlg.lip.setValue(6.0)
        dlg.rod.setValue(10.0)
        dlg.spokes.setValue(6)
        assert not dlg.bearing.isChecked(), "a rod is what an adapter is bored for by default"
        assert "59.6" in dlg.info.text(), "the dialog does not show the hub it makes"
        dlg.bearing.setChecked(True)
        assert not dlg.rod.isEnabled(), "the rod still counts with a bearing in the way"
        assert "22.1" in dlg.info.text(), "the dialog does not show the bearing pocket"
        dlg.bearing.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.spool_adapter_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no adapter in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = spool_adapter.sizes(hub=60.0, depth=20.0, lip=6.0, rod=10.0, spokes=6)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the adapter does not stand on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(s["flange"], abs=1e-3), "the flange is not as computed"
        settings = app_module.Settings()
        assert (settings.spool_adapter_hub, settings.spool_adapter_depth,
                settings.spool_adapter_lip, settings.spool_adapter_rod,
                settings.spool_adapter_bearing, settings.spool_adapter_spokes) == (
            60.0, 20.0, 6.0, 10.0, False, 6)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Spool Adapter") == labels.index("Phone Stand") + 1
    finally:
        w.settings.spool_adapter_hub, w.settings.spool_adapter_depth = 53.0, 12.0
        w.settings.spool_adapter_lip, w.settings.spool_adapter_rod = 4.0, 8.0
        w.settings.spool_adapter_bearing, w.settings.spool_adapter_spokes = False, 5
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
