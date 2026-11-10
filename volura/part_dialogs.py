"""The dialogs of the printable parts: the gear, the box and the small hardware.

The spur gear, the project box, the knob, the standoff, the label plate, the 3D text, the cable
clip, the print-in-place hinge, the battery holder and the storage bin. Each one starts at the
values of Settings and remembers the new ones; the actions that open them and the models they
build live in `volura/app.py`.
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFontComboBox, QFormLayout, QLabel, QPlainTextEdit, QSpinBox)

from . import (battery_holder, cable_clip, enclosure, fonts, gears, hinge, knob, label_plate,
               standoff, storage_bin, text3d, threads)
from .i18n import tr


class SpurGearDialog(QDialog):
    """The "Spur Gear…" dialog: an involute gear wheel for the worksheet or the model.

    The teeth and the module together say how big the wheel is, the pressure angle how
    steep the flanks stand and the backlash how much play two wheels have against each
    other; those values come from the axle, the pinion and the material, so the dialog
    starts where the user left them (Settings) and remembers the new ones. Under the
    fields the wheel stands in figures, because that is what the user checks it against:
    the place in the box and the distance to the other axle.

    With `solid` the wheel is not cut from a sheet but printed, so the dialog also asks
    how thick the wheel has to be.
    """

    def __init__(self, win, solid=False):
        super().__init__(win)
        s = win.settings
        self.solid = bool(solid)
        self.setWindowTitle(tr("Spur Gear"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5, suffix=" mm"):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setValue(float(value))
            sp.valueChanged.connect(self._measure)
            return sp

        self.teeth = QSpinBox()
        self.teeth.setRange(gears.TEETH_MIN, gears.TEETH_MAX)
        self.teeth.setValue(s.gear_teeth)
        self.teeth.setToolTip(tr("Number of teeth on the wheel"))
        self.teeth.valueChanged.connect(self._measure)
        self.module = spin(gears.MODULE_MIN, gears.MODULE_MAX, s.gear_module, 0.5)
        self.module.setToolTip(tr("The pitch diameter per tooth: two wheels of the same "
                                  "module run against each other"))
        self.angle = spin(gears.ANGLE_MIN, gears.ANGLE_MAX, s.gear_angle, 0.5, "°")
        self.angle.setToolTip(tr("How steep the flanks stand; 20 degrees is the usual choice"))
        self.bore = spin(0.0, 200.0, s.gear_bore, 1.0)
        self.bore.setToolTip(tr("Diameter of the hole for the axle; zero leaves a solid wheel"))
        self.backlash = spin(0.0, 1.0, s.gear_backlash, 0.05)
        self.backlash.setToolTip(tr("The play of a pair of wheels on the pitch circle, so "
                                    "the flanks do not press on each other"))
        if self.solid:
            self.thickness = spin(1.0, 100.0, s.gear_thickness, 0.5)
            self.thickness.setToolTip(tr("How thick the printed wheel is"))
        self.info = QLabel()
        lay.addRow(tr("Teeth"), self.teeth)
        lay.addRow(tr("Module"), self.module)
        lay.addRow(tr("Pressure angle"), self.angle)
        lay.addRow(tr("Bore"), self.bore)
        lay.addRow(tr("Backlash"), self.backlash)
        if self.solid:
            lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._measure()

    def _measure(self):
        """The pitch and the outside diameter of the wheel as it stands in the fields."""
        m = float(self.module.value())
        d = gears.pitch_diameter(int(self.teeth.value()), m)
        self.info.setText(tr("<i>Pitch diameter {d} mm, outside diameter {o} mm</i>")
                          .format(d=f"{d:.2f}", o=f"{d + 2.0 * m:.2f}"))

    def options(self) -> dict:
        o = {"teeth": int(self.teeth.value()), "module": round(self.module.value(), 6),
             "angle": round(self.angle.value(), 6), "bore": round(self.bore.value(), 6),
             "backlash": round(self.backlash.value(), 6)}
        if self.solid:
            o["thickness"] = round(self.thickness.value(), 6)
        return o


class ProjectBoxDialog(QDialog):
    """The "Project Box…" dialog: an enclosure with a lid that slides into it.

    The user fills in the **inside** of the box, because that is what has to fit: the
    electronics. The wall, the floor and the lid are the plastic around it and the clearance
    is the room the lip of the lid needs to slide into the box, which every printer likes a
    little differently, so the dialog starts at the clearance of the screw holes, the one
    this printer already proved. With a screw size the base also gets a post in every corner
    with a pilot hole, and the lid a hole above it; with vent slots the two long walls get
    slots through them, so the air inside can change. The other values come from Settings and
    are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Project Box"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.length = spin(1.0, 500.0, s.box3d_length)
        self.length.setToolTip(tr("The inside length: what has to fit in the box"))
        self.width = spin(1.0, 500.0, s.box3d_width)
        self.width.setToolTip(tr("The inside width: what has to fit in the box"))
        self.height = spin(1.0, 500.0, s.box3d_height)
        self.height.setToolTip(tr("The inside height, from the floor to the open top"))
        self.wall = spin(enclosure.WALL_MIN, 20.0, s.box3d_wall, 0.2)
        self.wall.setToolTip(tr("The thickness of the sides and of the lip of the lid"))
        self.floor = spin(enclosure.FLOOR_MIN, 20.0, s.box3d_floor, 0.2)
        self.floor.setToolTip(tr("The thickness of the bottom"))
        self.radius = spin(0.0, 100.0, s.box3d_radius)
        self.radius.setToolTip(tr("The rounding of the vertical corners; the inside is one "
                                  "wall thinner"))
        self.lid = spin(enclosure.LID_MIN, 20.0, s.box3d_lid, 0.2)
        self.lid.setToolTip(tr("The thickness of the lid"))
        self.lip = spin(0.0, 50.0, s.box3d_lip)
        self.lip.setToolTip(tr("How deep the lid slides into the box"))
        self.clearance = spin(0.0, enclosure.CLEARANCE_MAX, s.box3d_clearance, 0.05)
        self.clearance.setToolTip(tr("The room between the lip and the wall, so the lid "
                                     "slides in"))
        self.screws = QComboBox()
        self.screws.addItem(tr("None"), "")
        for size in enclosure.SCREW_SIZES:
            self.screws.addItem(size, size)
        self.screws.setCurrentIndex(max(0, self.screws.findData(s.box3d_screws)))
        self.screws.setToolTip(tr("Round posts in the corners with a pilot hole, to screw "
                                  "the lid on"))
        self.vents = QSpinBox()
        self.vents.setRange(0, enclosure.VENTS_MAX)
        self.vents.setValue(int(s.box3d_vents))
        self.vents.setToolTip(tr("Vertical slots through both long walls, so the air inside "
                                 "the box can change"))
        lay.addRow(tr("Inside length"), self.length)
        lay.addRow(tr("Inside width"), self.width)
        lay.addRow(tr("Inside height"), self.height)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Floor"), self.floor)
        lay.addRow(tr("Corner radius"), self.radius)
        lay.addRow(tr("Lid"), self.lid)
        lay.addRow(tr("Lip"), self.lip)
        lay.addRow(tr("Clearance"), self.clearance)
        lay.addRow(tr("Screws"), self.screws)
        lay.addRow(tr("Vent slots"), self.vents)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"length": round(self.length.value(), 6),
                "width": round(self.width.value(), 6),
                "height": round(self.height.value(), 6),
                "wall": round(self.wall.value(), 6),
                "floor": round(self.floor.value(), 6),
                "radius": round(self.radius.value(), 6),
                "lid": round(self.lid.value(), 6),
                "lip": round(self.lip.value(), 6),
                "clearance": round(self.clearance.value(), 6),
                "screws": str(self.screws.currentData() or ""),
                "vents": int(self.vents.value())}


class KnobDialog(QDialog):
    """The "Knob…" dialog: a printable thumb knob that turns on a bolt or a shaft.

    The knob is the round part you turn with two fingers, so the dialog starts with what the
    hand has to fit around: the diameter and the height. The grip is the rim you hold: ribs
    stay grippy with oily fingers, lobes are easier to print than fine ribs and still turn
    under a thumb, and a smooth rim is a plain cylinder. The mount says what the knob turns
    on: a hexagonal pocket for a nut, a counterbore for the head of a socket head bolt, or a
    blind D hole for the shaft of a potentiometer, whose flat keeps the knob from spinning on
    the axle. The clearance is added to every diameter, because a printed hole comes out
    smaller than drawn, so the dialog starts at the clearance of the screw holes, the one
    this printer already proved. The other values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Knob"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.diameter = spin(knob.DIAMETER_MIN, knob.DIAMETER_MAX, s.knob_diameter)
        self.diameter.setToolTip(tr("Across the highest points of the rim: what the hand "
                                    "wraps around"))
        self.height = spin(knob.HEIGHT_MIN, knob.HEIGHT_MAX, s.knob_height)
        self.height.setToolTip(tr("How thick the knob is; a knob of 4 mm is already hard to "
                                  "hold"))
        self.grip = QComboBox()
        for label, key in ((tr("Knurled"), "knurl"), (tr("Lobed"), "lobes"),
                           (tr("Smooth"), "smooth")):
            self.grip.addItem(label, key)
        self.grip.setCurrentIndex(max(0, self.grip.findData(s.knob_grip)))
        self.grip.setToolTip(tr("Straight ribs stay grippy with oily fingers, lobes print "
                                "easier and a smooth rim is plain"))
        self.lobes = QSpinBox()
        self.lobes.setRange(knob.LOBES_MIN, knob.LOBES_MAX)
        self.lobes.setValue(s.knob_lobes)
        self.lobes.setToolTip(tr("Number of lobes of a lobed rim"))
        self.mount = QComboBox()
        for label, key in ((tr("Nut"), "nut"), (tr("Bolt head"), "head"),
                           (tr("D-shaft"), "shaft")):
            self.mount.addItem(label, key)
        self.mount.setCurrentIndex(max(0, self.mount.findData(s.knob_kind)))
        self.mount.setToolTip(tr("A pocket for a nut, a counterbore for the head of a bolt, "
                                 "or a blind hole for a shaft"))
        self.size = QComboBox()
        for size in sorted(threads.SCREW_HOLES, key=lambda m: float(m[1:])):
            self.size.addItem(size, size)
        self.size.setCurrentIndex(max(0, self.size.findData(s.knob_bore)))
        self.size.setToolTip(tr("The thread of the bolt the knob screws onto, which sets the "
                                "hole and the pocket"))
        self.shaft_d = spin(0.5, 50.0, s.knob_shaft_d)
        self.shaft_d.setToolTip(tr("The diameter of the shaft the knob is pushed on"))
        self.flat = spin(0.0, 10.0, s.knob_flat, 0.1)
        self.flat.setToolTip(tr("How much of the shaft hole is cut away, so the knob cannot "
                                "spin on the axle"))
        self.clearance = spin(0.0, knob.CLEARANCE_MAX, s.knob_clearance, 0.05)
        self.clearance.setToolTip(tr("The room on every diameter, because a printed hole "
                                     "comes out smaller than drawn"))
        lay.addRow(tr("Diameter"), self.diameter)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Grip"), self.grip)
        lay.addRow(tr("Lobes"), self.lobes)
        lay.addRow(tr("Mount"), self.mount)
        lay.addRow(tr("Size"), self.size)
        lay.addRow(tr("Shaft diameter"), self.shaft_d)
        lay.addRow(tr("Flat"), self.flat)
        lay.addRow(tr("Clearance"), self.clearance)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"diameter": round(self.diameter.value(), 6),
                "height": round(self.height.value(), 6),
                "grip": str(self.grip.currentData()),
                "lobes": int(self.lobes.value()),
                "kind": str(self.mount.currentData()),
                "bore": str(self.size.currentData()),
                "shaft_d": round(self.shaft_d.value(), 6),
                "flat": round(self.flat.value(), 6),
                "clearance": round(self.clearance.value(), 6)}


class StandoffDialog(QDialog):
    """The "Standoff…" dialog: a printable spacer that holds a PCB or a panel off a surface.

    A standoff is the spacer under a board, so the dialog starts with what it has to hold the
    board off: the height. The outer size and the shape say how thick the spacer is and whether
    it is a cylinder or a hexagon, which stands still on a tilted bed. The mount says how the
    screw is held: a clearance hole the screw goes through, a pilot hole a self-tapping screw
    cuts its own thread in, or a hole a heat-set insert is melted into, which takes a bolt as
    often as you like. The clearance is added to the clearance holes, because a printed hole
    comes out smaller than drawn, so the dialog starts at the clearance of the screw holes, the
    one this printer already proved. The count puts that many standoffs in a row, so the corners
    of a PCB print in one go. The other values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Standoff"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.height = spin(standoff.HEIGHT_MIN, standoff.HEIGHT_MAX, s.standoff_height)
        self.height.setToolTip(tr("How far the board is held off the surface under it"))
        self.outer = spin(standoff.OUTER_MIN, standoff.OUTER_MAX, s.standoff_outer)
        self.outer.setToolTip(tr("How thick the standoff is; a hexagon is measured across "
                                 "the flats"))
        self.shape = QComboBox()
        for label, key in ((tr("Round"), "round"), (tr("Hexagonal"), "hex")):
            self.shape.addItem(label, key)
        self.shape.setCurrentIndex(max(0, self.shape.findData(s.standoff_shape)))
        self.shape.setToolTip(tr("A hexagon stands still on a tilted bed and grips in a slot"))
        self.mount = QComboBox()
        for label, key in ((tr("Through hole"), "through"), (tr("Self-tapping"), "tap"),
                           (tr("Heat-set insert"), "insert")):
            self.mount.addItem(label, key)
        self.mount.setCurrentIndex(max(0, self.mount.findData(s.standoff_kind)))
        self.mount.setToolTip(tr("A hole the screw goes through, a pilot hole a self-tapping "
                                 "screw cuts its thread in, or a hole for a heat-set insert"))
        self.size = QComboBox()
        for size in sorted(threads.SCREW_HOLES, key=lambda m: float(m[1:])):
            self.size.addItem(size, size)
        self.size.setCurrentIndex(max(0, self.size.findData(s.standoff_bore)))
        self.size.setToolTip(tr("The thread of the screw, which sets the hole and the size "
                                "of the insert"))
        self.count = QSpinBox()
        self.count.setRange(standoff.COUNT_MIN, standoff.COUNT_MAX)
        self.count.setValue(s.standoff_count)
        self.count.setToolTip(tr("How many standoffs in a row, so the corners of a board "
                                 "print in one go"))
        self.clearance = spin(0.0, standoff.CLEARANCE_MAX, s.standoff_clearance, 0.05)
        self.clearance.setToolTip(tr("The room on every clearance hole, because a printed hole "
                                     "comes out smaller than drawn"))
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Outer size"), self.outer)
        lay.addRow(tr("Shape"), self.shape)
        lay.addRow(tr("Mount"), self.mount)
        lay.addRow(tr("Size"), self.size)
        lay.addRow(tr("Count"), self.count)
        lay.addRow(tr("Clearance"), self.clearance)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"height": round(self.height.value(), 6),
                "outer": round(self.outer.value(), 6),
                "shape": str(self.shape.currentData()),
                "kind": str(self.mount.currentData()),
                "bore": str(self.size.currentData()),
                "count": int(self.count.value()),
                "clearance": round(self.clearance.value(), 6)}


class LabelPlateDialog(QDialog):
    """The "Label Plate…" dialog: a printable name tag or drawer label with text on it.

    A label is read from a distance, so the dialog starts with the text and how tall its letters
    are; the height is the height of a capital, which is what a label is read by. The plate is
    that text with a band of margin all round, so the margin and the corner radius say how much
    plate there is and how it feels in the fingers. Raised letters are what a printer does best
    and print in one go with the plate; engraved ones still read when the plate is rubbed or
    painted, but may not go deeper than the plate is thick. A hole hangs the tag on a hook or a
    screw. The values come from Settings and are remembered, the text too.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Label Plate"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.text = QPlainTextEdit()
        self.text.setPlainText(s.label_text)
        self.text.setTabChangesFocus(True)
        self.text.setToolTip(tr("The text on the plate; Enter starts the next line"))
        self.height = spin(label_plate.HEIGHT_MIN, label_plate.HEIGHT_MAX, s.label_height)
        self.height.setToolTip(tr("The height of a capital letter, which is what a label is "
                                  "read by"))
        self.thickness = spin(label_plate.THICKNESS_MIN, label_plate.THICKNESS_MAX,
                              s.label_thickness)
        self.thickness.setToolTip(tr("How thick the plate is"))
        self.relief = spin(label_plate.RELIEF_MIN, label_plate.RELIEF_MAX, s.label_relief, 0.1)
        self.relief.setToolTip(tr("How high the letters stand above the plate, or how deep they "
                                  "are cut in"))
        self.mode = QComboBox()
        for label, key in ((tr("Raised"), "raised"), (tr("Engraved"), "engraved")):
            self.mode.addItem(label, key)
        self.mode.setCurrentIndex(max(0, self.mode.findData(s.label_mode)))
        self.mode.setToolTip(tr("Raised letters print easiest, engraved ones still read after "
                                "the paint"))
        self.margin = spin(label_plate.MARGIN_MIN, label_plate.MARGIN_MAX, s.label_margin)
        self.margin.setToolTip(tr("The band of plate around the text"))
        self.radius = spin(0.0, 20.0, s.label_radius)
        self.radius.setToolTip(tr("The rounding of the corners of the plate"))
        self.hole = spin(0.0, label_plate.HOLE_MAX, s.label_hole)
        self.hole.setToolTip(tr("A round hole through the plate, left of the text, to hang the "
                                "tag up; 0 leaves it out"))
        lay.addRow(tr("Text"), self.text)
        lay.addRow(tr("Text height"), self.height)
        lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(tr("Relief"), self.relief)
        lay.addRow(tr("Style"), self.mode)
        lay.addRow(tr("Margin"), self.margin)
        lay.addRow(tr("Corner radius"), self.radius)
        lay.addRow(tr("Hole"), self.hole)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"text": self.text.toPlainText(),
                "height": round(self.height.value(), 6),
                "thickness": round(self.thickness.value(), 6),
                "relief": round(self.relief.value(), 6),
                "mode": str(self.mode.currentData()),
                "margin": round(self.margin.value(), 6),
                "radius": round(self.radius.value(), 6),
                "hole": round(self.hole.value(), 6)}


class Text3DDialog(QDialog):
    """The "3D Text…" dialog: a part that is made of letters.

    Here the text is the part itself, so the dialog starts with the text and how tall its letters
    are; the height is the height of a capital, which is what text is read by. The depth is how
    thick the letters are, and since they stand on that side, a shallow text is a plate with a
    name on it and a deep one is a knob you can grab. Bold strokes read from further away and wear
    less quickly. The alignment says how the lines of several lines stand under each other; the
    whole block is centred, because that is where a part is placed. The values come from Settings
    and are remembered, the text too.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("3D Text"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.text = QPlainTextEdit()
        self.text.setPlainText(s.text3d_text)
        self.text.setTabChangesFocus(True)
        self.text.setToolTip(tr("The text of the part; Enter starts the next line"))
        self.family = QFontComboBox()
        self.family.setCurrentFont(QFont(s.text3d_family or fonts.default_family()))
        self.family.setToolTip(tr("The font of the letters"))
        self.height = spin(text3d.HEIGHT_MIN, text3d.HEIGHT_MAX, s.text3d_height)
        self.height.setToolTip(tr("The height of a capital letter, which is what text is read by"))
        self.depth = spin(text3d.DEPTH_MIN, text3d.DEPTH_MAX, s.text3d_depth)
        self.depth.setToolTip(tr("How thick the letters are, which is how the part stands "
                                 "on the bed"))
        self.bold = QCheckBox(tr("Bold"))
        self.bold.setChecked(bool(s.text3d_bold))
        self.bold.setToolTip(tr("Thicker strokes, which read from further away and wear less"))
        self.align = QComboBox()
        for label, key in ((tr("Left"), "left"), (tr("Centre"), "centre"),
                           (tr("Right"), "right")):
            self.align.addItem(label, key)
        self.align.setCurrentIndex(max(0, self.align.findData(s.text3d_align)))
        self.align.setToolTip(tr("How the lines stand under each other"))
        lay.addRow(tr("Text"), self.text)
        lay.addRow(tr("Font"), self.family)
        lay.addRow(tr("Text height"), self.height)
        lay.addRow(tr("Depth"), self.depth)
        lay.addRow("", self.bold)
        lay.addRow(tr("Alignment"), self.align)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"text": self.text.toPlainText(),
                "family": str(self.family.currentFont().family()),
                "height": round(self.height.value(), 6),
                "depth": round(self.depth.value(), 6),
                "bold": bool(self.bold.isChecked()),
                "align": str(self.align.currentData())}


class CableClipDialog(QDialog):
    """The "Cable Clip…" dialog: a printable clip that holds a cable against a wall or a desk.

    A clip is made for one cable, so the dialog starts with what it has to hold: the diameter of
    the cable or the bundle. The width is how high the clip stands, since it prints lying on its
    side, and the wall is the ring that springs back behind the cable. The opening is the mouth as
    a percentage of the cable: smaller than the cable, or the cable falls out, and wide enough to
    snap the cable in without tearing the ring. The foot is the tab that is screwed to the wall and
    the size is the screw that goes through it. The count puts that many clips in a row, so a run
    along a desk prints in one go. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Cable clip"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.cable_d = spin(cable_clip.CABLE_MIN, cable_clip.CABLE_MAX, s.clip_cable_d)
        self.cable_d.setToolTip(tr("Across the cable or the bundle the clip has to hold"))
        self.width = spin(cable_clip.WIDTH_MIN, cable_clip.WIDTH_MAX, s.clip_width)
        self.width.setToolTip(tr("How high the clip stands, which is how wide it is: the cable "
                                 "runs through it like a sleeve"))
        self.wall = spin(cable_clip.WALL_MIN, cable_clip.WALL_MAX, s.clip_wall, 0.2)
        self.wall.setToolTip(tr("The ring around the cable, which springs back behind it"))
        self.opening = spin(100.0 * cable_clip.OPENING_MIN, 100.0 * cable_clip.OPENING_MAX,
                            100.0 * s.clip_opening, 1.0)
        self.opening.setDecimals(0)
        self.opening.setSuffix(" %")
        self.opening.setToolTip(tr("The mouth as a part of the cable: less than 100 % holds the "
                                   "cable, more lets it fall out"))
        self.base = spin(cable_clip.BASE_MIN, cable_clip.BASE_MAX, s.clip_base)
        self.base.setToolTip(tr("How thick the foot is that the screw goes through"))
        self.size = QComboBox()
        for size in sorted(threads.SCREW_HOLES, key=lambda m: float(m[1:])):
            self.size.addItem(size, size)
        self.size.setCurrentIndex(max(0, self.size.findData(s.clip_bore)))
        self.size.setToolTip(tr("The screw that fixes the clip to the wall or the desk"))
        self.count = QSpinBox()
        self.count.setRange(cable_clip.COUNT_MIN, cable_clip.COUNT_MAX)
        self.count.setValue(s.clip_count)
        self.count.setToolTip(tr("How many clips in a row, so a run of them prints in one go"))
        lay.addRow(tr("Cable diameter"), self.cable_d)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Opening"), self.opening)
        lay.addRow(tr("Base"), self.base)
        lay.addRow(tr("Size"), self.size)
        lay.addRow(tr("Count"), self.count)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"cable_d": round(self.cable_d.value(), 6),
                "width": round(self.width.value(), 6),
                "wall": round(self.wall.value(), 6),
                "opening": round(self.opening.value() / 100.0, 6),
                "base": round(self.base.value(), 6),
                "bore": str(self.size.currentData()),
                "count": int(self.count.value())}


class HingeDialog(QDialog):
    """The "Print-in-Place Hinge…" dialog: a hinge that prints as one piece and turns at once.

    A hinge is made for something that has to open, so the dialog starts with the leaves: how long
    the hinge is along the opening and how far and how thick each leaf is. The knuckles are the
    barrels that turn around one pin; their number is odd, so both ends of the hinge carry the pin
    and the pin cannot walk out. The clearance is the play between the pin and its hole and between
    two neighbouring barrels, which is what makes the hinge move once it is off the printer, and
    the pin is the rod the knuckles turn on. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Print-in-place hinge"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.length = spin(hinge.LENGTH_MIN, hinge.LENGTH_MAX, s.hinge_length)
        self.length.setToolTip(tr("Along the pin, which is how wide the door, the lid or the "
                                  "case is"))
        self.width = spin(hinge.WIDTH_MIN, hinge.WIDTH_MAX, s.hinge_width)
        self.width.setToolTip(tr("How far one leaf reaches out from the pin, on each side of it"))
        self.thickness = spin(hinge.THICKNESS_MIN, hinge.THICKNESS_MAX, s.hinge_thickness, 0.2)
        self.thickness.setToolTip(tr("How thick each leaf is, which is also the wall of its "
                                     "knuckles"))
        self.knuckles = QSpinBox()
        self.knuckles.setRange(hinge.KNUCKLES_MIN, hinge.KNUCKLES_MAX)
        self.knuckles.setSingleStep(2)
        self.knuckles.setValue(s.hinge_knuckles if s.hinge_knuckles % 2 else s.hinge_knuckles + 1)
        self.knuckles.setToolTip(tr("How many barrels there are on the pin: an odd number, so the "
                                    "pin is held at both ends"))
        self.clearance = spin(hinge.CLEARANCE_MIN, hinge.CLEARANCE_MAX, s.hinge_clearance, 0.05)
        self.clearance.setToolTip(tr("The play between the pin and its hole and between two "
                                     "knuckles: too little freezes the hinge, too much makes it "
                                     "rattle"))
        self.pin_d = spin(hinge.PIN_MIN, hinge.PIN_MAX, s.hinge_pin_d)
        self.pin_d.setToolTip(tr("The rod the knuckles turn on, which prints as one piece with "
                                 "the leaf that carries it"))
        lay.addRow(tr("Length"), self.length)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(tr("Knuckles"), self.knuckles)
        lay.addRow(tr("Clearance"), self.clearance)
        lay.addRow(tr("Pin diameter"), self.pin_d)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"length": round(self.length.value(), 6),
                "width": round(self.width.value(), 6),
                "thickness": round(self.thickness.value(), 6),
                "knuckles": int(self.knuckles.value()),
                "clearance": round(self.clearance.value(), 6),
                "pin_d": round(self.pin_d.value(), 6)}


class BatteryHolderDialog(QDialog):
    """The "Battery Holder…" dialog: a block that keeps round cells side by side.

    A holder is made for one kind of cell, so the dialog starts with the cell: the list gives the
    usual types and the cradles are built around the diameter and the length that belong to it. The
    count is how many cells stand side by side, the wall is the plastic between the cradles, around
    them and under them, and the clearance is the play a cell presses in with: too little and the
    cell does not go in, too much and it rattles in the holder. The grip is how much narrower the
    opening over a cradle is than the cell, so the cell snaps in instead of only lying in its
    cradle. The mounting screws are the flat head screws that fix the holder to a base through the
    floor, sunk under the cells, or none at all. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Battery holder"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.cell = QComboBox()
        for name in battery_holder.CELLS:
            self.cell.addItem(name, name)
        self.cell.setCurrentIndex(max(0, self.cell.findData(s.battery_cell)))
        self.cell.setToolTip(tr("The cell the cradles are made for"))
        self.count = QSpinBox()
        self.count.setRange(battery_holder.COUNT_MIN, battery_holder.COUNT_MAX)
        self.count.setValue(s.battery_count)
        self.count.setToolTip(tr("How many cells stand side by side in the holder"))
        self.wall = spin(battery_holder.WALL_MIN, battery_holder.WALL_MAX, s.battery_wall, 0.2)
        self.wall.setToolTip(tr("The plastic between the cradles, around them and under them"))
        self.clearance = spin(battery_holder.CLEARANCE_MIN, battery_holder.CLEARANCE_MAX,
                              s.battery_clearance, 0.05)
        self.clearance.setToolTip(tr("The play around a cell: too little and it does not press in, "
                                     "too much and it rattles in the holder"))
        self.grip = spin(100.0 * battery_holder.GRIP_MIN, 100.0 * battery_holder.GRIP_MAX,
                         100.0 * s.battery_grip, 1.0)
        self.grip.setDecimals(0)
        self.grip.setSuffix(" %")
        self.grip.setToolTip(tr("How much narrower the opening over a cradle is than the cell: the "
                                "cell snaps in behind the lips and cannot fall out"))
        self.screw = QComboBox()
        self.screw.addItem(tr("None"), "none")
        for size in battery_holder.SCREWS[1:]:
            self.screw.addItem(size, size)
        self.screw.setCurrentIndex(max(0, self.screw.findData(s.battery_screw)))
        self.screw.setToolTip(tr("The flat head screws that fix the holder to a base through the "
                                 "floor, sunk under the cells"))
        lay.addRow(tr("Cell"), self.cell)
        lay.addRow(tr("Count"), self.count)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Clearance"), self.clearance)
        lay.addRow(tr("Grip %"), self.grip)
        lay.addRow(tr("Mounting screws"), self.screw)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"cell": str(self.cell.currentData()),
                "count": int(self.count.value()),
                "wall": round(self.wall.value(), 6),
                "clearance": round(self.clearance.value(), 6),
                "grip": round(self.grip.value() / 100.0, 6),
                "screw": str(self.screw.currentData())}


class StorageBinDialog(QDialog):
    """The "Storage Bin…" dialog: an open-top box with compartments and a ledge for a label.

    The three sizes are the outside of the bin, because that is what has to fit on the shelf, and
    the wall and the floor are the plastic around the inside. The radius rounds the vertical
    corners, the columns and the rows say how many compartments the dividers make, and the ledge
    is the sloped piece along the inside of the front wall on which a label lies readable. The
    values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Storage bin"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.length = spin(storage_bin.LENGTH_MIN, storage_bin.LENGTH_MAX, s.bin_length, 5.0)
        self.length.setToolTip(tr("The outside length of the bin, along the front wall"))
        self.width = spin(storage_bin.WIDTH_MIN, storage_bin.WIDTH_MAX, s.bin_width, 5.0)
        self.width.setToolTip(tr("The outside width of the bin, from the front wall to the back"))
        self.height = spin(storage_bin.HEIGHT_MIN, storage_bin.HEIGHT_MAX, s.bin_height, 5.0)
        self.height.setToolTip(tr("The outside height of the bin, floor included"))
        self.wall = spin(storage_bin.WALL_MIN, storage_bin.WALL_MAX, s.bin_wall, 0.2)
        self.wall.setToolTip(tr("The plastic of the walls and of the dividers"))
        self.floor = spin(storage_bin.FLOOR_MIN, storage_bin.FLOOR_MAX, s.bin_floor, 0.2)
        self.floor.setToolTip(tr("The thickness of the bottom, so nothing walks out under the wall"))
        self.radius = spin(storage_bin.RADIUS_MIN, storage_bin.RADIUS_MAX, s.bin_radius, 0.5)
        self.radius.setToolTip(tr("The rounding of the vertical corners outside"))
        self.cols = QSpinBox()
        self.cols.setRange(storage_bin.COLS_MIN, storage_bin.COLS_MAX)
        self.cols.setValue(s.bin_cols)
        self.cols.setToolTip(tr("How many compartments lie next to each other along the length"))
        self.rows = QSpinBox()
        self.rows.setRange(storage_bin.COLS_MIN, storage_bin.COLS_MAX)
        self.rows.setValue(s.bin_rows)
        self.rows.setToolTip(tr("How many compartments lie next to each other along the width"))
        self.label = QCheckBox(tr("Label ledge"))
        self.label.setChecked(bool(s.bin_label))
        self.label.setToolTip(tr("A sloped ledge along the inside of the front wall, on which a "
                                 "label lies readable; it prints without support"))
        lay.addRow(tr("Length"), self.length)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Floor"), self.floor)
        lay.addRow(tr("Radius"), self.radius)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow("", self.label)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"length": round(self.length.value(), 6),
                "width": round(self.width.value(), 6),
                "height": round(self.height.value(), 6),
                "wall": round(self.wall.value(), 6),
                "floor": round(self.floor.value(), 6),
                "radius": round(self.radius.value(), 6),
                "cols": int(self.cols.value()),
                "rows": int(self.rows.value()),
                "label": bool(self.label.isChecked())}
