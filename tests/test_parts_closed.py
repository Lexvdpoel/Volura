"""Every ready-made part must come back out of the model as the closed volume that went in.

Draw > Parts builds a manifold and hands it to ``ops.group_from_manifold``; the export takes the
group back through ``ops.parts`` and calls it "not closed" when that round trip does not give a
valid volume. A part can therefore be a perfect manifold and still fail in the model: where two
of its faces touch exactly - a countersink cone that ends on the wall of its hole, a screw post
that only touches the wall of the box, a rim that is stood on the back of a board - the boolean
leaves a sliver, and the vertices at its ends merge into one when the faces are made. The edge
that is left then belongs to three faces, which is not a closed volume.

This builds every part of the menu with its default sizes and with a few other typical ones, and
every screw and choice a part offers, puts the result in a fresh document and asks the export for
it again: the same number of parts, no problem, and the same volume within a tenth of a percent.

Run with:  python -m pytest tests/test_parts_closed.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import (battery_holder, bin_rack, bit_holder, bracket, cable_chain,  # noqa: E402
                    cable_clip, cable_winder, card_holder, drawer_organizer, enclosure, fan_guard,
                    fonts, gears,
                    gridfinity,
                    grommet, hex_key_holder, hinge, hose_adapter, knob, label_plate, lithophane,
                    pegboard,
                    phone_stand,
                    plant_pot, screw_jar, spool_adapter, standoff, storage_bin, text3d, tie_mount, wall_hook)
from volura import ops  # noqa: E402
from volura.model import Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])     # the letters need Qt

COLOR = "#f2f2f2"


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def gear(teeth, module=2.0, thickness=6.0, **kw):
    """A spur gear as the window makes it: the tooth profile extruded `thickness` millimetres."""
    sec = mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                           for q in gears.spur_gear(teeth, module=module, **kw)],
                          mf.FillRule.EvenOdd)
    return [mf.Manifold.extrude(sec, thickness)]


def box(**kw):
    """A project box and its lid, as ``enclosure.project_box`` gives them."""
    return list(enclosure.project_box(**kw))


def rack(**kw):
    """A bin wall rack as the window takes it: one manifold per panel."""
    return bin_rack.bin_rack_panels(**kw)


def plate(**kw):
    """A label plate, whose text needs the fonts of this system."""
    need_fonts()
    return [label_plate.label_plate(**kw)]


def letters(*args, **kw):
    """3D text, whose letters need the fonts of this system."""
    need_fonts()
    return [text3d.text3d(*args, **kw)]


def bin_label(**kw):
    """A storage bin, with its label when it has one, as the window puts both in the model."""
    text = kw.pop("text", "")
    made = [storage_bin.storage_bin(**kw)]
    if text:
        need_fonts()
        made.append(storage_bin.bin_label(text, **{k: kw[k] for k in
                                                   ("length", "width", "height", "wall", "floor",
                                                    "radius", "scoop", "stack")}))
    return made


def winder(**kw):
    """A cable winder, whose engraved label needs the fonts of this system when it has one."""
    if str(kw.get("label", "")).strip():
        need_fonts()
    return [cable_winder.cable_winder(**kw)]


def photo(**kw):
    """A lithophane of a small picture: a gradient, so every thickness between is in it."""
    rows, cols = int(kw.pop("rows", 12)), int(kw.pop("cols", 16))
    gray = np.linspace(0.0, 1.0, rows * cols).reshape(rows, cols)
    return [lithophane.lithophane(gray, **kw)]


def holder(**kw):
    """A hex key holder, whose engraved sizes need the fonts of this system when it has them."""
    if kw.get("labels", True):
        need_fonts()
    return [hex_key_holder.hex_key_holder(**kw)]


CASES = [
    ("project box default", lambda: box()),
    ("project box small", lambda: box(length=40, width=30, height=25, wall=1.6, floor=1.6,
                                      radius=2.0, lid=1.6, lip=2.0, clearance=0.1)),
    ("project box vented with posts", lambda: box(length=200, width=120, height=80, wall=2.5,
                                                  floor=3.0, radius=4.0, lid=2.5, lip=4.0,
                                                  clearance=0.3, screws="M3", vents=4)),
    ("project box with a board", lambda: box(length=120, width=65, height=45, wall=2.0,
                                             screws="M2.5", board="Arduino Uno")),

    ("knob default", lambda: [knob.knob()]),
    ("knob small smooth head", lambda: [knob.knob(diameter=15.0, height=6.0, grip="smooth",
                                                  bore="M3", kind="head")]),
    ("knob large lobed shaft", lambda: [knob.knob(diameter=60.0, height=25.0, grip="lobes",
                                                  lobes=8, bore="M5", kind="shaft",
                                                  shaft_d=10.0, flat=2.5)]),

    ("standoff default", lambda: [standoff.standoff()]),
    ("standoff small hex tapped", lambda: [standoff.standoff(height=4.0, outer=4.0, shape="hex",
                                                             bore="M2", kind="tap")]),
    ("standoff row of inserts", lambda: [standoff.standoff(height=40.0, outer=12.0, shape="hex",
                                                           bore="M5", kind="insert", count=6)]),

    ("label plate default", lambda: plate()),
    ("label plate engraved with holes", lambda: plate(
        text="Volura\n2026", height=5.0, thickness=1.2, relief=0.4, mode="engraved",
        margin=1.5, radius=1.0, hole=3.0)),
    ("label plate large", lambda: plate(
        text="HELLO WORLD", height=20.0, thickness=4.0, relief=1.5, margin=6.0, radius=4.0,
        hole=5.0)),

    ("cable clip default", lambda: [cable_clip.cable_clip()]),
    ("cable clip small row", lambda: [cable_clip.cable_clip(cable_d=2.5, width=5.0, wall=1.0,
                                                            opening=0.4, base=1.5, bore="M2",
                                                            count=4)]),
    ("cable clip large", lambda: [cable_clip.cable_clip(cable_d=30.0, width=35.0, wall=4.0,
                                                        opening=0.9, base=6.0, bore="M5",
                                                        count=2)]),

    ("hinge default", lambda: [hinge.hinge()]),
    ("hinge small", lambda: [hinge.hinge(length=15.0, width=6.0, thickness=1.0, knuckles=3,
                                         clearance=0.1, pin_d=1.5)]),
    ("hinge large", lambda: [hinge.hinge(length=200.0, width=50.0, thickness=6.0, knuckles=9,
                                         clearance=0.8, pin_d=10.0)]),

    ("battery holder default", lambda: [battery_holder.battery_holder()]),
    ("battery holder one AAA", lambda: [battery_holder.battery_holder(
        cell="AAA", count=1, wall=1.2, clearance=0.2, grip=0.0, screw="none")]),
    ("battery holder four 18650", lambda: [battery_holder.battery_holder(
        cell="18650", count=4, wall=2.5, clearance=0.6, grip=0.1, screw="M4")]),

    ("storage bin default", lambda: [storage_bin.storage_bin()]),
    ("storage bin small plain", lambda: [storage_bin.storage_bin(
        length=20, width=20, height=10, wall=1.0, floor=0.8, radius=0.0, label=False)]),
    ("storage bin with label", lambda: bin_label(length=120, width=80, height=50, wall=2.0,
                                                 floor=1.6, radius=6.0, cols=3, label=True,
                                                 hanger=True, stack=True, scoop=True,
                                                 text="Volura")),

    ("bin rack default", lambda: rack()),
    ("bin rack one cell no screws", lambda: rack(cols=1, rows=1, cell_w=40.0, cell_h=35.0,
                                                 plate=2.0, screw="none")),
    ("bin rack in panels", lambda: rack(cols=6, rows=3, cell_w=100.0, cell_h=60.0, plate=6.0,
                                        screw="M3", panel_cols=2)),

    ("cable chain default", lambda: [cable_chain.cable_chain()]),
    ("cable chain small", lambda: [cable_chain.cable_chain(width=8.0, height=6.0, wall=1.2,
                                                           links=3, bend=20.0)]),
    ("cable chain large", lambda: [cable_chain.cable_chain(width=60.0, height=40.0, wall=3.5,
                                                           links=12, bend=90.0)]),

    ("gridfinity bin default", lambda: [gridfinity.gridfinity_bin()]),
    ("gridfinity bin with magnets", lambda: [gridfinity.gridfinity_bin(
        cols=2, rows=2, units=5, wall=0.8, floor=0.6, divisions=4, lip=True, magnets=True,
        scoop=True, label=True)]),
    ("gridfinity bin low no lip", lambda: [gridfinity.gridfinity_bin(
        cols=3, rows=1, units=2, wall=2.0, floor=2.0, divisions=1, lip=False)]),
    ("gridfinity baseplate default", lambda: [gridfinity.gridfinity_baseplate()]),
    ("gridfinity baseplate one cell", lambda: [gridfinity.gridfinity_baseplate(1, 1)]),
    ("gridfinity baseplate large", lambda: [gridfinity.gridfinity_baseplate(6, 4)]),

    ("pegboard hook default", lambda: [pegboard.pegboard_part()]),
    ("pegboard hook thin board", lambda: [pegboard.pegboard_part(
        kind="hook", width=6.0, reach=15.0, board=3.0)]),
    ("pegboard shelf", lambda: [pegboard.pegboard_part(kind="shelf", width=12.0, reach=80.0)]),
    ("pegboard shelf thick board", lambda: [pegboard.pegboard_part(
        kind="shelf", width=20.0, reach=100.0, board=7.0)]),
    ("pegboard holder", lambda: [pegboard.pegboard_part(kind="holder", size=12.0)]),
    ("pegboard holder large", lambda: [pegboard.pegboard_part(
        kind="holder", width=20.0, reach=100.0, board=7.0, size=40.0)]),
    ("pegboard board default", lambda: [pegboard.pegboard_part(kind="board")]),
    ("pegboard board small", lambda: [pegboard.pegboard_part(
        kind="board", cols=2, rows=2, board=3.0, standoff=10.0)]),
    ("pegboard board large", lambda: [pegboard.pegboard_part(
        kind="board", cols=12, rows=12, board=7.0, standoff=40.0)]),

    ("hose adapter default", lambda: [hose_adapter.hose_adapter()]),
    ("hose adapter small", lambda: [hose_adapter.hose_adapter(d1=4.0, d2=3.0, wall=0.8,
                                                              barbs=1)]),
    ("hose adapter straight large", lambda: [hose_adapter.hose_adapter(
        d1=50.0, d2=32.0, wall=4.0, barbs=6, straight=True)]),

    ("fan guard default", lambda: [fan_guard.fan_guard()]),
    ("fan guard thin spokes", lambda: [fan_guard.fan_guard(
        fan=92, thickness=1.0, pattern="spokes", bar=0.8, gap=2.0)]),
    ("fan guard grid", lambda: [fan_guard.fan_guard(
        fan=120, thickness=4.0, pattern="grid", bar=3.0, gap=8.0)]),

    ("desk grommet default", lambda: [grommet.grommet()]),
    ("desk grommet small no lid", lambda: [grommet.grommet(hole=20.0, desk=10.0, flange=4.0,
                                                           wall=2.0, slot=0.0, lid=False)]),
    ("desk grommet large", lambda: [grommet.grommet(hole=120.0, desk=60.0, flange=20.0,
                                                    wall=4.0, slot=60.0, lid=True)]),

    ("corner bracket default", lambda: [bracket.bracket()]),
    ("corner bracket small plain", lambda: [bracket.bracket(
        leg_a=15.0, leg_b=15.0, width=8.0, thickness=2.0, holes=0, screw="M3",
        countersink=False, gussets=False, fillet=0.0)]),
    ("corner bracket large", lambda: [bracket.bracket(
        leg_a=150.0, leg_b=100.0, width=60.0, thickness=10.0, holes=3, screw="M6",
        gussets=True, fillet=3.0)]),

    ("zip-tie mount default", lambda: [tie_mount.tie_mount()]),
    ("zip-tie mount small one way", lambda: [tie_mount.tie_mount(
        base=12.0, tie_width=2.5, tie_thick=1.0, height=2.0, screw="M3", countersink=False,
        ways=1)]),
    ("zip-tie mount large", lambda: [tie_mount.tie_mount(
        base=40.0, tie_width=9.0, tie_thick=2.5, height=15.0, screw="M5", ways=2)]),

    ("bit holder default", lambda: [bit_holder.bit_holder()]),
    ("bit holder one bit no label", lambda: [bit_holder.bit_holder(
        cols=1, rows=1, depth=6.0, fit=0.0, pitch=8.5, angle=0.0, label=False)]),
    ("bit holder slanted full", lambda: [bit_holder.bit_holder(
        cols=12, rows=4, depth=20.0, fit=0.4, pitch=15.0, angle=30.0, label=True)]),

    ("memory card holder default", lambda: [card_holder.card_holder()]),
    ("memory card holder one card", lambda: [card_holder.card_holder(
        sd=1, micro=0, usb=0, rows=1, fit=0.1, depth=6.0)]),
    ("memory card holder full", lambda: [card_holder.card_holder(
        sd=6, micro=6, usb=6, rows=2, fit=0.6, depth=20.0)]),

    ("wall hook default", lambda: [wall_hook.wall_hook()]),
    ("wall hook small one hole", lambda: [wall_hook.wall_hook(
        reach=10.0, rise=5.0, width=8.0, thickness=3.0, plate=20.0, screw="M3",
        countersink=False, holes=1)]),
    ("wall hook large", lambda: [wall_hook.wall_hook(
        reach=80.0, rise=50.0, width=40.0, thickness=10.0, plate=150.0, screw="M5",
        countersink=True, holes=2)]),

    ("phone stand default", lambda: [phone_stand.phone_stand()]),
    ("phone stand small no cable", lambda: [phone_stand.phone_stand(
        width=30.0, angle=50.0, back=40.0, device=5.0, lip=4.0, cable=False)]),
    ("phone stand large", lambda: [phone_stand.phone_stand(
        width=200.0, angle=80.0, back=160.0, device=16.0, lip=20.0, cable=True)]),

    ("spool adapter default", lambda: [spool_adapter.spool_adapter()]),
    ("spool adapter small bearing solid hub", lambda: [spool_adapter.spool_adapter(
        hub=25.0, depth=5.0, lip=2.0, bearing=True, spokes=0)]),
    ("spool adapter large rod eight spokes", lambda: [spool_adapter.spool_adapter(
        hub=90.0, depth=40.0, lip=10.0, rod=30.0, spokes=8)]),

    ("drawer organizer default", lambda: [drawer_organizer.drawer_organizer()]),
    ("drawer organizer small two by two", lambda: [drawer_organizer.drawer_organizer(
        length=60.0, width=45.0, height=20.0, clearance=1.0, cols=2, rows=2, wall=1.6, floor=1.0,
        radius=3.0, finger=True)]),
    ("drawer organizer large no clearance", lambda: [drawer_organizer.drawer_organizer(
        length=300.0, width=240.0, height=100.0, clearance=0.0, cols=6, rows=4, wall=3.0,
        floor=2.0, radius=10.0, finger=True)]),

    ("plant pot default", lambda: plant_pot.plant_pot()),
    ("plant pot small hexagon without saucer", lambda: plant_pot.plant_pot(
        diameter=40.0, height=30.0, taper=50.0, wall=1.2, holes=1, shape="hexagon", saucer=False)),
    ("plant pot large nine drain holes", lambda: plant_pot.plant_pot(
        diameter=250.0, height=250.0, taper=90.0, wall=4.0, holes=9, shape="round", saucer=True)),

    ("cable winder default", lambda: winder()),
    ("cable winder small no clamp", lambda: winder(
        length=40.0, width=20.0, thickness=2.0, slot=8.0, depth=5.0, keyhole=False)),
    ("cable winder large labelled", lambda: winder(
        length=200.0, width=100.0, thickness=6.0, slot=30.0, depth=60.0, label="VOLURA")),

    ("lithophane default", lambda: photo()),
    ("lithophane small without border", lambda: photo(
        width=40.0, thin=0.4, thick=1.0, border=0.0, pixel=0.1)),
    ("lithophane large inverted coarse", lambda: photo(
        rows=40, cols=60, width=250.0, thin=2.0, thick=6.0, border=10.0, invert=True, pixel=1.0)),

    ("hex key holder default", lambda: holder()),
    ("hex key holder one key flat top", lambda: holder(sizes_mm="6", steps=False, labels=False)),
    ("hex key holder full set deep holes", lambda: holder(
        sizes_mm="0.7, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19", clearance=0.6,
        depth=30.0, gap=10.0, margin=10.0)),

    ("screw-top jar default", lambda: screw_jar.screw_jar()),
    ("screw-top jar small coarse smooth lid", lambda: screw_jar.screw_jar(
        diameter=20.0, height=15.0, wall=1.6, pitch=5.0, thread_len=6.0, clearance=0.1, grip=0)),
    ("screw-top jar large fine many ribs", lambda: screw_jar.screw_jar(
        diameter=150.0, height=200.0, wall=4.0, pitch=2.0, thread_len=20.0, clearance=0.6,
        grip=48)),

    ("3D text default", lambda: letters("Volura")),
    ("3D text small", lambda: letters("A", height=2.0, depth=0.4)),
    ("3D text bold centred two lines", lambda: letters(
        "VOLURA\n3D", height=40.0, depth=8.0, bold=True, align="centre")),

    ("spur gear default", lambda: gear(12)),
    ("spur gear with bore", lambda: gear(20, module=1.5, thickness=6.0, bore=8.0)),
    ("spur gear many teeth", lambda: gear(48, module=1.0, thickness=10.0, backlash=0.2)),
]


def round_trip(made):
    """The parts the export finds after `made` went into a fresh document as groups."""
    doc = Document()
    for i, m in enumerate(made, 1):
        ops.group_from_manifold(doc, m, COLOR, "Part %d" % i)
    return doc, ops.parts(doc)


def assert_closed(name, made):
    """The round trip of one part: the same closed volumes, the same volume, no problem."""
    assert made, "%s: the part built nothing" % name
    for m in made:
        assert m.status() == mf.Error.NoError and not m.is_empty(), \
            "%s: the module itself did not give a closed volume" % name
    doc, found = round_trip(made)
    assert [p.name for p in found] == [g.name for g in doc.groups]
    assert len(found) == len(made), \
        "%s: the model gives %d part(s) instead of %d" % (name, len(found), len(made))
    for m, p in zip(made, found):
        assert p.problem is None, "%s: %s" % (name, p.problem)
        assert p.manifold is not None, "%s: no manifold came back" % name
        assert p.manifold.volume() == pytest.approx(m.volume(), rel=1e-3), \
            "%s: the volume changed in the round trip" % name


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_part_is_closed_after_the_round_trip(case):
    """Every part of Draw > Parts is a closed volume again once the export asks for it."""
    name, build = case
    assert_closed(name, build())


def options():
    """The choices a part offers, each at least once: a hole for every screw, a rim on every
    board. A choice can hide a face that touches itself, so the round trip runs for all of them."""
    out = []
    for size in wall_hook.SCREWS:
        for sink in (True, False):
            for width in (8.0, 40.0):
                out.append(("wall hook %s %s, %g mm wide"
                            % (size, "countersunk" if sink else "plain", width),
                            lambda size=size, sink=sink, width=width:
                            [wall_hook.wall_hook(width=width, screw=size, countersink=sink)]))
    for size in bracket.SCREWS:
        for sink in (True, False):
            out.append(("corner bracket %s %s" % (size, "countersunk" if sink else "plain"),
                        lambda size=size, sink=sink:
                        [bracket.bracket(screw=size, countersink=sink)]))
    for size in tie_mount.SCREWS:
        for sink in (True, False):
            out.append(("zip-tie mount %s %s" % (size, "countersunk" if sink else "plain"),
                        lambda size=size, sink=sink:
                        [tie_mount.tie_mount(screw=size, countersink=sink)]))
    for size in ("M2", "M2.5", "M3", "M4"):
        for radius in (0.0, 3.0):
            out.append(("project box posts %s, corner radius %g" % (size, radius),
                        lambda size=size, radius=radius:
                        list(enclosure.project_box(screws=size, radius=radius))))
    for board in (3.0, 7.0):
        for cols, rows in ((2, 2), (6, 6)):
            out.append(("pegboard board %g mm, %d by %d holes" % (board, cols, rows),
                        lambda board=board, cols=cols, rows=rows:
                        [pegboard.pegboard_part(kind="board", cols=cols, rows=rows,
                                                board=board)]))
    for shape in standoff.STANDOFF_SHAPES:
        for kind in standoff.STANDOFF_KINDS:
            out.append(("standoff %s %s" % (shape, kind),
                        lambda shape=shape, kind=kind:
                        [standoff.standoff(shape=shape, kind=kind, bore="M3")]))
    for pattern in fan_guard.PATTERNS:
        out.append(("fan guard %s" % pattern,
                    lambda pattern=pattern: [fan_guard.fan_guard(pattern=pattern)]))
    for lid in (True, False):
        out.append(("desk grommet with%s lid" % ("" if lid else "out"),
                    lambda lid=lid: [grommet.grommet(lid=lid)]))
    return out


OPTIONS = options()


@pytest.mark.parametrize("case", OPTIONS, ids=[c[0] for c in OPTIONS])
def test_every_option_is_closed_after_the_round_trip(case):
    """Every screw and choice a part offers comes back closed from the model as well."""
    name, build = case
    assert_closed(name, build())
