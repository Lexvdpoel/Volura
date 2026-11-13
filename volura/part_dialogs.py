"""The dialogs of the printable parts: the gear, the box and the small hardware.

The spur gear, the project box, the knob, the standoff, the label plate, the 3D text, the cable
clip, the print-in-place hinge, the battery holder, the storage bin and its wall rack. Each one
starts at the values of Settings and remembers the new ones; the actions that open them and the models they
build live in `volura/app.py`.
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFontComboBox, QFormLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QSpinBox)

from . import (battery_holder, bin_rack, bit_holder, bracket, cable_chain, cable_clip,
               card_holder, drawer_organizer, enclosure, fan_guard, fonts, gears,
               gridfinity, grommet, hinge, hose_adapter, knob, label_plate, pegboard, phone_stand,
               plant_pot, spool_adapter, standoff, storage_bin, text3d, threads, tie_mount,
               wall_hook)
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
        self.board = QComboBox()
        self.board.addItem(tr("None"), "")
        for name in enclosure.BOARDS:
            self.board.addItem(name, name)
        self.board.setCurrentIndex(max(0, self.board.findData(s.box3d_board)))
        self.board.setToolTip(tr("Standoffs on the floor under the mounting holes of this "
                                 "board, so it is screwed down"))
        self.board_height = spin(enclosure.BOARD_HEIGHT_MIN, enclosure.BOARD_HEIGHT_MAX,
                                 s.box3d_board_height)
        self.board_height.setToolTip(tr("How high the board stands above the floor"))
        lay.addRow(tr("Circuit board"), self.board)
        lay.addRow(tr("Board standoffs"), self.board_height)
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
                "vents": int(self.vents.value()),
                "board": str(self.board.currentData() or ""),
                "board_height": round(self.board_height.value(), 6)}


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
        self.scoop = QCheckBox(tr("Open front"))
        self.scoop.setChecked(bool(s.bin_scoop))
        self.scoop.setToolTip(tr("From half the height up the front is open and the side walls "
                                 "run back at 45 degrees, so you reach in also when bins stand "
                                 "on each other"))
        self.label = QCheckBox(tr("Label slot"))
        self.label.setChecked(bool(s.bin_label))
        self.label.setToolTip(tr("A slot in the front wall into which a printed label slides "
                                 "from above, facing forward"))
        lay.addRow(tr("Length"), self.length)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Floor"), self.floor)
        lay.addRow(tr("Radius"), self.radius)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow("", self.scoop)
        lay.addRow("", self.label)
        self.label_text = QLineEdit(s.bin_label_text)
        self.label_text.setPlaceholderText(tr("No label"))
        self.label_text.setToolTip(tr("Text for a printed label that slides into the slot in "
                                      "the front; leave it empty for no label"))
        self.label_text.setEnabled(self.label.isChecked())
        self.label.toggled.connect(self.label_text.setEnabled)
        lay.addRow(tr("Label text"), self.label_text)
        self.hanger = QCheckBox(tr("Wall hook"))
        self.hanger.setChecked(bool(s.bin_hanger))
        self.hanger.setToolTip(tr("A hook along the top of the back wall and a rib along its "
                                  "bottom, so the bin hangs on a rail of the bin wall rack"))
        lay.addRow("", self.hanger)
        self.stack = QCheckBox(tr("Stackable"))
        self.stack.setChecked(bool(s.bin_stack))
        self.stack.setToolTip(tr("A foot under the bin that drops into a bin of the same size, so "
                                 "the bins stand on each other without sliding off"))
        lay.addRow("", self.stack)
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
                "label": bool(self.label.isChecked()),
                "hanger": bool(self.hanger.isChecked()),
                "stack": bool(self.stack.isChecked()),
                "scoop": bool(self.scoop.isChecked()),
                "label_text": self.label_text.text().strip()}


class PegboardDialog(QDialog):
    """The "Pegboard…" dialog: the board itself, or a hook, a shelf or a tool holder for it.

    The kind says which part. For a part that hangs, the width is how thick it prints, the reach
    how far it stands out of the board and the board how thick the pegboard is; a holder also
    takes the size of the tool it holds. The board itself takes its holes across and down, its
    thickness and how far it stands off the wall. What a kind does not use is greyed out. The
    values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Pegboard"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.kind = QComboBox()
        for kind, label in (("hook", tr("Hook")), ("shelf", tr("Shelf")),
                            ("holder", tr("Tool holder")), ("board", tr("Board"))):
            self.kind.addItem(label, kind)
        self.kind.setCurrentIndex(max(0, self.kind.findData(s.peg_kind)))
        self.width = spin(pegboard.WIDTH_MIN, pegboard.WIDTH_MAX, s.peg_width, 1.0)
        self.width.setToolTip(tr("How thick the part is, which is the height it prints"))
        self.reach = spin(pegboard.REACH_MIN, pegboard.REACH_MAX, s.peg_reach, 5.0)
        self.reach.setToolTip(tr("How far the part stands out of the board"))
        self.board = spin(pegboard.BOARD_MIN, pegboard.BOARD_MAX, s.peg_board, 0.5)
        self.board.setToolTip(tr("The thickness of the pegboard: the hook behind it fits that board "
                                 "with less than half a millimetre of play"))
        self.size = spin(pegboard.SIZE_MIN, pegboard.SIZE_MAX, s.peg_size, 1.0)
        self.size.setToolTip(tr("The tool the holder takes, across"))

        def count(lo, hi, value):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(int(value))
            return sp

        self.cols = count(pegboard.COLS_MIN, pegboard.COLS_MAX, s.peg_cols)
        self.cols.setToolTip(tr("Holes across the board, one inch apart"))
        self.rows = count(pegboard.ROWS_MIN, pegboard.ROWS_MAX, s.peg_rows)
        self.rows.setToolTip(tr("Holes down the board, one inch apart"))
        self.standoff = spin(pegboard.STANDOFF_MIN, pegboard.STANDOFF_MAX, s.peg_standoff, 1.0)
        self.standoff.setToolTip(tr("How far the board stands off the wall, room for the pegs "
                                    "behind it"))
        lay.addRow(tr("Part"), self.kind)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Reach"), self.reach)
        lay.addRow(tr("Board"), self.board)
        lay.addRow(tr("Tool size"), self.size)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Off the wall"), self.standoff)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _kind_changed(self, *_):
        kind = self.kind.currentData()
        for w in (self.width, self.reach):
            w.setEnabled(kind != "board")
        self.size.setEnabled(kind == "holder")
        for w in (self.cols, self.rows, self.standoff):
            w.setEnabled(kind == "board")

    def options(self) -> dict:
        return {"kind": str(self.kind.currentData()),
                "width": round(self.width.value(), 6),
                "reach": round(self.reach.value(), 6),
                "board": round(self.board.value(), 6),
                "size": round(self.size.value(), 6),
                "cols": int(self.cols.value()), "rows": int(self.rows.value()),
                "standoff": round(self.standoff.value(), 6)}


class FanGuardDialog(QDialog):
    """The "Fan Guard…" dialog: a grille for a square fan of a standard size.

    The fan size gives the plate and the spacing of the screw holes; the grille is rings,
    spokes or a grid, of bars as wide as `bar` with openings of `gap`, on a plate as thick as
    asked. Under the fields how much of the blade circle stays open. The values come from
    Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Fan guard"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.2)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.fan = QComboBox()
        for size in sorted(fan_guard.FANS):
            self.fan.addItem(f"{size} mm", size)
        self.fan.setCurrentIndex(max(0, self.fan.findData(int(s.fan_size))))
        self.pattern = QComboBox()
        for name, label in (("rings", tr("Rings")), ("spokes", tr("Spokes")),
                            ("grid", tr("Grid"))):
            self.pattern.addItem(label, name)
        self.pattern.setCurrentIndex(max(0, self.pattern.findData(s.fan_pattern)))
        self.thickness = spin(fan_guard.THICK_MIN, fan_guard.THICK_MAX, s.fan_thickness)
        self.bar = spin(fan_guard.BAR_MIN, fan_guard.BAR_MAX, s.fan_bar)
        self.bar.setToolTip(tr("The width of every bar of the grille"))
        self.gap = spin(fan_guard.GAP_MIN, fan_guard.GAP_MAX, s.fan_gap)
        self.gap.setToolTip(tr("The widest opening of rings or a grid: what keeps fingers out"))
        self.info = QLabel()
        lay.addRow(tr("Fan"), self.fan)
        lay.addRow(tr("Grille"), self.pattern)
        lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(tr("Bar"), self.bar)
        lay.addRow(tr("Opening"), self.gap)
        lay.addRow("", self.info)
        self.fan.currentIndexChanged.connect(self._changed)
        self.pattern.currentIndexChanged.connect(self._changed)
        for w in (self.thickness, self.bar, self.gap):
            w.valueChanged.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = fan_guard.sizes(**self.options())
        whole = 3.141592653589793 * (0.5 * s["opening"]) ** 2
        self.info.setText(tr("Screw holes {d:g} mm apart, {p:.0f} % open")
                          .format(d=s["hole_spacing"], p=100.0 * s["open_area"] / whole))

    def options(self) -> dict:
        return {"fan": int(self.fan.currentData()), "pattern": str(self.pattern.currentData()),
                "thickness": round(self.thickness.value(), 6),
                "bar": round(self.bar.value(), 6), "gap": round(self.gap.value(), 6)}


class GrommetDialog(QDialog):
    """The "Desk Grommet…" dialog: a cable pass-through for a hole in a desk top.

    The hole in the desk and the thickness of the desk top give the sleeve that goes into it; the
    flange says how far the grommet comes out past the hole and the wall how thick that sleeve is.
    The lid closes the hole off and its slot is how far in from the rim the cables reach. Under
    the fields the sleeve, the bore and the lid the grommet will have. The values come from
    Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Desk grommet"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.hole = spin(grommet.HOLE_MIN, grommet.HOLE_MAX, s.grommet_hole)
        self.hole.setSingleStep(5.0)
        self.hole.setToolTip(tr("The diameter of the hole in the desk"))
        self.desk = spin(grommet.DESK_MIN, grommet.DESK_MAX, s.grommet_desk)
        self.desk.setToolTip(tr("How thick the desk top is"))
        self.flange = spin(grommet.FLANGE_MIN, grommet.FLANGE_MAX, s.grommet_flange)
        self.flange.setToolTip(tr("How far the flange comes out past the hole"))
        self.wall = spin(grommet.WALL_MIN, grommet.WALL_MAX, s.grommet_wall)
        self.wall.setToolTip(tr("The plastic of the sleeve"))
        self.slot = spin(0.0, 0.5 * grommet.HOLE_MAX, s.grommet_slot)
        self.slot.setToolTip(tr("How far in from the rim of the lid the cables reach"))
        self.lid = QCheckBox(tr("A lid with a slot for the cables"))
        self.lid.setChecked(bool(s.grommet_lid))
        self.info = QLabel()
        lay.addRow(tr("Hole"), self.hole)
        lay.addRow(tr("Desk top"), self.desk)
        lay.addRow(tr("Flange"), self.flange)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Slot"), self.slot)
        lay.addRow("", self.lid)
        lay.addRow("", self.info)
        for w in (self.hole, self.desk, self.flange, self.wall, self.slot):
            w.valueChanged.connect(self._changed)
        self.lid.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = grommet.sizes(**self.options())
        if s["lid"]:
            self.info.setText(tr("Sleeve {d:g} by {l:g} mm, bore {b:g} mm, lid {c:g} mm")
                              .format(d=s["sleeve_d"], l=s["sleeve_len"], b=s["bore_d"],
                                      c=s["lid_d"]))
        else:
            self.info.setText(tr("Sleeve {d:g} by {l:g} mm, bore {b:g} mm")
                              .format(d=s["sleeve_d"], l=s["sleeve_len"], b=s["bore_d"]))

    def options(self) -> dict:
        return {"hole": round(self.hole.value(), 6), "desk": round(self.desk.value(), 6),
                "flange": round(self.flange.value(), 6), "wall": round(self.wall.value(), 6),
                "slot": round(self.slot.value(), 6), "lid": self.lid.isChecked()}


class BracketDialog(QDialog):
    """The "Corner Bracket…" dialog: an L-shaped angle bracket with screw holes in both legs.

    The two legs and the width say how big the bracket is and the thickness how thick each leg
    is; the holes in each leg, the screw and whether its head sinks say how it is fastened, and
    the inner radius and the gussets how much stronger the bend is made. Under the fields the
    hole it drills and the number of holes each leg really gets, because a short leg takes fewer.
    The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Corner bracket"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.leg_a = spin(bracket.LEG_MIN, bracket.LEG_MAX, s.bracket_leg_a)
        self.leg_a.setSingleStep(5.0)
        self.leg_a.setToolTip(tr("How long the first leg is, measured on the outside"))
        self.leg_b = spin(bracket.LEG_MIN, bracket.LEG_MAX, s.bracket_leg_b)
        self.leg_b.setSingleStep(5.0)
        self.leg_b.setToolTip(tr("How long the other leg is, measured on the outside"))
        self.width = spin(bracket.WIDTH_MIN, bracket.WIDTH_MAX, s.bracket_width)
        self.width.setSingleStep(5.0)
        self.width.setToolTip(tr("How wide the bracket is, which is how high it stands on the bed"))
        self.thickness = spin(bracket.THICK_MIN, bracket.THICK_MAX, s.bracket_thickness)
        self.thickness.setToolTip(tr("How thick each of the two legs is"))
        self.holes = QSpinBox()
        self.holes.setRange(bracket.HOLES_MIN, bracket.HOLES_MAX)
        self.holes.setValue(int(s.bracket_holes))
        self.holes.setToolTip(tr("How many screw holes there are in each leg"))
        self.screw = QComboBox()
        for size in bracket.SCREWS:
            self.screw.addItem(size, size)
        self.screw.setCurrentIndex(max(0, self.screw.findData(s.bracket_screw)))
        self.screw.setToolTip(tr("The screw that goes through the holes"))
        self.countersink = QCheckBox(tr("The head sinks into the inside of the leg"))
        self.countersink.setChecked(bool(s.bracket_countersink))
        self.gussets = QCheckBox(tr("Gussets"))
        self.gussets.setToolTip(tr("A rib in the inner corner at both ends of the width"))
        self.gussets.setChecked(bool(s.bracket_gussets))
        self.fillet = spin(0.0, bracket.FILLET_MAX * bracket.THICK_MAX, s.bracket_fillet)
        self.fillet.setToolTip(tr("The radius of the inner corner, which is what makes it strong"))
        self.info = QLabel()
        lay.addRow(tr("Leg A"), self.leg_a)
        lay.addRow(tr("Leg B"), self.leg_b)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(tr("Holes"), self.holes)
        lay.addRow(tr("Screw"), self.screw)
        lay.addRow("", self.countersink)
        lay.addRow("", self.gussets)
        lay.addRow(tr("Inner radius"), self.fillet)
        lay.addRow("", self.info)
        for w in (self.leg_a, self.leg_b, self.width, self.thickness, self.fillet):
            w.valueChanged.connect(self._changed)
        self.holes.valueChanged.connect(self._changed)
        self.screw.currentIndexChanged.connect(self._changed)
        self.countersink.toggled.connect(self._changed)
        self.gussets.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = bracket.sizes(**self.options())
        self.info.setText(tr("Hole {d:g} mm, {a} in leg A, {b} in leg B, gusset {g:g} mm")
                          .format(d=s["hole_d"], a=len(s["holes_a"]), b=len(s["holes_b"]),
                                  g=s["gusset"]))

    def options(self) -> dict:
        return {"leg_a": round(self.leg_a.value(), 6), "leg_b": round(self.leg_b.value(), 6),
                "width": round(self.width.value(), 6),
                "thickness": round(self.thickness.value(), 6),
                "holes": int(self.holes.value()), "screw": str(self.screw.currentData()),
                "countersink": self.countersink.isChecked(),
                "gussets": self.gussets.isChecked(),
                "fillet": round(self.fillet.value(), 6)}


class TieMountDialog(QDialog):
    """The "Zip-Tie Mount…" dialog: a cable-tie anchor that is screwed onto a surface.

    The base plate says how big the part that lies against the surface is, and the two sizes of the
    tie how big the tunnel is that it slides through; the screw and whether its head sinks say how
    the mount is fastened, and the ways whether the tie can go either way. Under the fields the
    tunnel, the bridge and the height the mount really stands, because a low one is made as tall as
    the roof over the tunnel needs. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Zip-tie mount"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.base = spin(tie_mount.BASE_MIN, tie_mount.BASE_MAX, s.tie_mount_base)
        self.base.setSingleStep(2.0)
        self.base.setToolTip(tr("How big the plate is that lies against the surface"))
        self.tie_width = spin(tie_mount.TIE_W_MIN, tie_mount.TIE_W_MAX, s.tie_mount_tie_width)
        self.tie_width.setToolTip(tr("How wide the cable tie that slides through it is"))
        self.tie_thick = spin(tie_mount.TIE_T_MIN, tie_mount.TIE_T_MAX, s.tie_mount_tie_thick)
        self.tie_thick.setToolTip(tr("How thick the cable tie that slides through it is"))
        self.height = spin(2.0, tie_mount.HEIGHT_MAX, s.tie_mount_height)
        self.height.setToolTip(tr("How high the mount stands on the bed, roof over the tunnel"))
        self.screw = QComboBox()
        for size in tie_mount.SCREWS:
            self.screw.addItem(size, size)
        self.screw.setCurrentIndex(max(0, self.screw.findData(s.tie_mount_screw)))
        self.screw.setToolTip(tr("The screw that fastens the mount to the surface"))
        self.countersink = QCheckBox(tr("The head sinks into the floor of the tunnel"))
        self.countersink.setChecked(bool(s.tie_mount_countersink))
        self.ways = QSpinBox()
        self.ways.setRange(tie_mount.WAYS_MIN, tie_mount.WAYS_MAX)
        self.ways.setValue(int(s.tie_mount_ways))
        self.ways.setToolTip(tr("One tunnel for the tie, or two that cross so it can go either way"))
        self.info = QLabel()
        lay.addRow(tr("Base plate"), self.base)
        lay.addRow(tr("Tie width"), self.tie_width)
        lay.addRow(tr("Tie thickness"), self.tie_thick)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Screw"), self.screw)
        lay.addRow("", self.countersink)
        lay.addRow(tr("Ways"), self.ways)
        lay.addRow("", self.info)
        for w in (self.base, self.tie_width, self.tie_thick, self.height):
            w.valueChanged.connect(self._changed)
        self.screw.currentIndexChanged.connect(self._changed)
        self.countersink.toggled.connect(self._changed)
        self.ways.valueChanged.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = tie_mount.sizes(**self.options())
        self.info.setText(tr("Tunnel {w:g} × {h:g} mm, bridge {b:g} mm, {z:g} mm high")
                          .format(w=s["slot_w"], h=s["slot_h"], b=s["bridge"], z=s["height"]))

    def options(self) -> dict:
        return {"base": round(self.base.value(), 6),
                "tie_width": round(self.tie_width.value(), 6),
                "tie_thick": round(self.tie_thick.value(), 6),
                "height": round(self.height.value(), 6),
                "screw": str(self.screw.currentData()),
                "countersink": self.countersink.isChecked(), "ways": int(self.ways.value())}


class BitHolderDialog(QDialog):
    """The "Bit Holder…" dialog: a block that stands screwdriver bits upright in rows.

    The columns and rows say how many bits there are, the depth how far they stand in the block and
    the fit how much wider a socket is than a bit; the pitch is the distance between the sockets and
    the tilt how far the rows lean towards you. Under the fields the socket as it prints and the
    block around it, because a pitch that is too small is raised to keep the wall between the
    sockets. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Bit holder"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        def count(lo, hi, value, tip):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(int(value))
            sp.setToolTip(tip)
            return sp

        self.cols = count(bit_holder.COLS_MIN, bit_holder.COLS_MAX, s.bit_holder_cols,
                          tr("How many bits there are in a row"))
        self.rows = count(bit_holder.ROWS_MIN, bit_holder.ROWS_MAX, s.bit_holder_rows,
                          tr("How many rows of bits there are"))
        self.depth = spin(bit_holder.DEPTH_MIN, bit_holder.DEPTH_MAX, s.bit_holder_depth, 1.0)
        self.depth.setToolTip(tr("How far the bit stands in the block"))
        self.fit = spin(bit_holder.FIT_MIN, bit_holder.FIT_MAX, s.bit_holder_fit, 0.05)
        self.fit.setDecimals(2)
        self.fit.setToolTip(tr("How much wider the socket is than the bit, on every flat"))
        self.pitch = spin(bit_holder.PITCH_MIN, bit_holder.PITCH_MAX, s.bit_holder_pitch, 0.5)
        self.pitch.setToolTip(tr("The distance between the centres of two sockets next to each "
                                 "other"))
        self.angle = QDoubleSpinBox()
        self.angle.setRange(bit_holder.ANGLE_MIN, bit_holder.ANGLE_MAX)
        self.angle.setDecimals(1)
        self.angle.setSingleStep(5.0)
        self.angle.setSuffix(" °")
        self.angle.setValue(float(s.bit_holder_angle))
        self.angle.setToolTip(tr("How far the rows lean backwards, so the bits lean towards you"))
        self.label = QCheckBox(tr("Label ledge"))
        self.label.setChecked(bool(s.bit_holder_label))
        self.label.setToolTip(tr("A flat along the front, below the top, for a printed or written "
                                 "label"))
        self.info = QLabel()
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Socket depth"), self.depth)
        lay.addRow(tr("Fit"), self.fit)
        lay.addRow(tr("Pitch"), self.pitch)
        lay.addRow(tr("Back tilt"), self.angle)
        lay.addRow("", self.label)
        lay.addRow("", self.info)
        for w in (self.depth, self.fit, self.pitch, self.angle):
            w.valueChanged.connect(self._changed)
        for w in (self.cols, self.rows):
            w.valueChanged.connect(self._changed)
        self.label.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = bit_holder.sizes(**self.options())
        self.info.setText(tr("{c} × {r} sockets of {f:g} mm, block {w:g} × {l:g} × {h:g} mm")
                          .format(c=s["cols"], r=s["rows"], f=s["flats"], w=s["width"],
                                  l=s["length"], h=s["height"]))

    def options(self) -> dict:
        return {"cols": int(self.cols.value()), "rows": int(self.rows.value()),
                "depth": round(self.depth.value(), 6), "fit": round(self.fit.value(), 6),
                "pitch": round(self.pitch.value(), 6), "angle": round(self.angle.value(), 6),
                "label": self.label.isChecked()}


class CardHolderDialog(QDialog):
    """The "Memory Card Holder…" dialog: a block that stands cards and USB sticks upright.

    The three counts say how many cards of every kind stand in the block, the rows how many rows of
    columns there are, the depth how far the cards go in and the fit how much wider a slot is than
    its card. Under the fields the block the holder will have, because a kind without cards is left
    out and the block is narrower for it; with not a single card there is nothing to print and OK
    waits. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Memory card holder"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step, tip):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            sp.setToolTip(tip)
            return sp

        def count(lo, hi, value, tip):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(int(value))
            sp.setToolTip(tip)
            return sp

        self.sd = count(card_holder.SD_MIN, card_holder.SD_MAX, s.card_holder_sd,
                        tr("How many SD cards stand in the block"))
        self.micro = count(card_holder.MICRO_MIN, card_holder.MICRO_MAX, s.card_holder_micro,
                           tr("How many microSD cards stand in the block"))
        self.usb = count(card_holder.USB_MIN, card_holder.USB_MAX, s.card_holder_usb,
                         tr("How many USB sticks stand in the block"))
        self.rows = count(card_holder.ROWS_MIN, card_holder.ROWS_MAX, s.card_holder_rows,
                          tr("How many rows of cards there are"))
        self.depth = spin(card_holder.DEPTH_MIN, card_holder.DEPTH_MAX, s.card_holder_depth, 1.0,
                          tr("How far a card goes into the block"))
        self.fit = spin(card_holder.FIT_MIN, card_holder.FIT_MAX, s.card_holder_fit, 0.05,
                        tr("How much wider a slot is than its card, on every side"))
        self.fit.setDecimals(2)
        self.info = QLabel()
        lay.addRow(tr("SD cards"), self.sd)
        lay.addRow(tr("microSD cards"), self.micro)
        lay.addRow(tr("USB sticks"), self.usb)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Slot depth"), self.depth)
        lay.addRow(tr("Fit"), self.fit)
        lay.addRow("", self.info)
        for w in (self.sd, self.micro, self.usb, self.rows, self.depth, self.fit):
            w.valueChanged.connect(self._changed)
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addRow(self.bb)
        self._changed()

    def _changed(self, *_):
        o = self.options()
        try:
            s = card_holder.sizes(**o)
        except ValueError as e:
            self.info.setText(str(e))
            self.bb.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        self.bb.button(QDialogButtonBox.Ok).setEnabled(True)
        self.info.setText(tr("{s} SD, {m} microSD, {u} USB, block {w:g} × {l:g} × {h:g} mm")
                          .format(s=s["sd"], m=s["micro"], u=s["usb"], w=s["width"],
                                  l=s["length"], h=s["height"]))

    def options(self) -> dict:
        return {"sd": int(self.sd.value()), "micro": int(self.micro.value()),
                "usb": int(self.usb.value()), "rows": int(self.rows.value()),
                "depth": round(self.depth.value(), 6), "fit": round(self.fit.value(), 6)}


class WallHookDialog(QDialog):
    """The "Wall Hook…" dialog: a J that is screwed to a wall, for a coat, a bag or a cable.

    The reach says how far the hook stands out from the wall and the rise how high its tip comes
    above the arm; the plate is what lies against the wall and the thickness how thick plate, arm
    and tip are, which is also how strong the hook is. The holes, the screw and whether its head
    sinks say how it is fastened. Under the fields the hook it will make, because a plate that is
    too short for its holes is made taller and a thick hook is never too short to bend. The values
    come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Wall hook"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.reach = spin(wall_hook.REACH_MIN, wall_hook.REACH_MAX, s.wall_hook_reach)
        self.reach.setSingleStep(5.0)
        self.reach.setToolTip(tr("How far out from the wall the hook reaches"))
        self.rise = spin(wall_hook.RISE_MIN, wall_hook.RISE_MAX, s.wall_hook_rise)
        self.rise.setSingleStep(5.0)
        self.rise.setToolTip(tr("How high the tip rises above the arm"))
        self.width = spin(wall_hook.WIDTH_MIN, wall_hook.WIDTH_MAX, s.wall_hook_width)
        self.width.setSingleStep(5.0)
        self.width.setToolTip(tr("How wide the hook is, which is how high it stands on the bed"))
        self.thickness = spin(wall_hook.THICK_MIN, wall_hook.THICK_MAX, s.wall_hook_thickness)
        self.thickness.setToolTip(tr("How thick the plate, the arm and the tip are"))
        self.plate = spin(0.0, wall_hook.PLATE_MAX, s.wall_hook_plate)
        self.plate.setSingleStep(5.0)
        self.plate.setToolTip(tr("How tall the plate that lies against the wall is"))
        self.holes = QSpinBox()
        self.holes.setRange(wall_hook.HOLES_MIN, wall_hook.HOLES_MAX)
        self.holes.setValue(int(s.wall_hook_holes))
        self.holes.setToolTip(tr("How many screw holes there are in the plate"))
        self.screw = QComboBox()
        for size in wall_hook.SCREWS:
            self.screw.addItem(size, size)
        self.screw.setCurrentIndex(max(0, self.screw.findData(s.wall_hook_screw)))
        self.screw.setToolTip(tr("The screw that goes through the holes"))
        self.countersink = QCheckBox(tr("The head sinks into the front of the plate"))
        self.countersink.setChecked(bool(s.wall_hook_countersink))
        self.info = QLabel()
        lay.addRow(tr("Reach"), self.reach)
        lay.addRow(tr("Rise"), self.rise)
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Thickness"), self.thickness)
        lay.addRow(tr("Plate"), self.plate)
        lay.addRow(tr("Holes"), self.holes)
        lay.addRow(tr("Screw"), self.screw)
        lay.addRow("", self.countersink)
        lay.addRow("", self.info)
        for w in (self.reach, self.rise, self.width, self.thickness, self.plate):
            w.valueChanged.connect(self._changed)
        self.holes.valueChanged.connect(self._changed)
        self.screw.currentIndexChanged.connect(self._changed)
        self.countersink.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = wall_hook.sizes(**self.options())
        self.info.setText(tr("Hook {r:g} × {i:g} mm, hole {d:g} mm, {n} in the plate")
                          .format(r=s["reach"], i=s["rise"], d=s["hole_d"], n=len(s["holes_y"])))

    def options(self) -> dict:
        return {"reach": round(self.reach.value(), 6), "rise": round(self.rise.value(), 6),
                "width": round(self.width.value(), 6),
                "thickness": round(self.thickness.value(), 6),
                "plate": round(self.plate.value(), 6), "holes": int(self.holes.value()),
                "screw": str(self.screw.currentData()), "countersink": self.countersink.isChecked()}


class PhoneStandDialog(QDialog):
    """The "Phone Stand…" dialog: a stand a phone or a tablet leans in.

    The width is how wide the stand is, which is also how high it stands on the bed because it
    prints on its side; the angle is how far the back support leans back from the bed and the back
    height how far it rises up that lean. The device thickness is the phone or tablet with its case
    in it, and the lip how far the front lip comes up above the ledge it stands on. Under the
    fields the base the stand is computed with, because a device that leans further over its foot
    needs a longer one. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Phone stand"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(5.0)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.width = spin(phone_stand.WIDTH_MIN, phone_stand.WIDTH_MAX, s.phone_stand_width)
        self.width.setToolTip(tr("How wide the stand is, which is how high it stands on the bed"))
        self.angle = QDoubleSpinBox()
        self.angle.setRange(phone_stand.ANGLE_MIN, phone_stand.ANGLE_MAX)
        self.angle.setDecimals(1)
        self.angle.setSingleStep(5.0)
        self.angle.setSuffix(" °")
        self.angle.setValue(float(s.phone_stand_angle))
        self.angle.setToolTip(tr("How far the back support leans back from the bed"))
        self.back = spin(phone_stand.BACK_MIN, phone_stand.BACK_MAX, s.phone_stand_back)
        self.back.setToolTip(tr("How far the support rises up the lean"))
        self.device = spin(phone_stand.DEVICE_MIN, phone_stand.DEVICE_MAX, s.phone_stand_device)
        self.device.setSingleStep(0.5)
        self.device.setToolTip(tr("How thick the phone or tablet is, its case included"))
        self.lip = spin(phone_stand.LIP_MIN, phone_stand.LIP_MAX, s.phone_stand_lip)
        self.lip.setSingleStep(1.0)
        self.lip.setToolTip(tr("How far the lip comes up above the ledge the device stands on"))
        self.cable = QCheckBox(tr("Cable slot"))
        self.cable.setChecked(bool(s.phone_stand_cable))
        self.cable.setToolTip(tr("A gap in the middle of the width for a charging cable, through "
                                 "the ledge and the base under it"))
        self.info = QLabel()
        lay.addRow(tr("Width"), self.width)
        lay.addRow(tr("Angle"), self.angle)
        lay.addRow(tr("Back height"), self.back)
        lay.addRow(tr("Device thickness"), self.device)
        lay.addRow(tr("Lip height"), self.lip)
        lay.addRow("", self.cable)
        lay.addRow("", self.info)
        for w in (self.width, self.angle, self.back, self.device, self.lip):
            w.valueChanged.connect(self._changed)
        self.cable.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = phone_stand.sizes(**self.options())
        self.info.setText(tr("Base {d:.1f} mm long, stand {h:.1f} mm high")
                          .format(d=s["depth"], h=s["height"]))

    def options(self) -> dict:
        return {"width": round(self.width.value(), 6), "angle": round(self.angle.value(), 6),
                "back": round(self.back.value(), 6), "device": round(self.device.value(), 6),
                "lip": round(self.lip.value(), 6), "cable": self.cable.isChecked()}


class SpoolAdapterDialog(QDialog):
    """The "Spool Adapter…" dialog: a hub that fills the hole of a filament spool.

    The hub is the hole in the middle of the spool, the depth how far the adapter goes into it and
    the lip how far the flange reaches past the hub to keep the adapter from sliding in. The rod is
    what turns in the middle; a 608 bearing makes a pocket for the bearing of a spool holder
    instead and greys out the rod. The spokes hold the two rings of the hub together where there is
    plastic to save. Under the fields the hub and the bore the adapter is made with, because a bore
    that is too fat for the hole makes the hub larger. The values come from Settings and are
    remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Spool adapter"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(1.0)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.hub = spin(spool_adapter.HUB_MIN, spool_adapter.HUB_MAX, s.spool_adapter_hub)
        self.hub.setSingleStep(5.0)
        self.hub.setToolTip(tr("The hole in the middle of the spool that the adapter fills"))
        self.depth = spin(spool_adapter.DEPTH_MIN, spool_adapter.DEPTH_MAX, s.spool_adapter_depth)
        self.depth.setSingleStep(5.0)
        self.depth.setToolTip(tr("How far the hub goes into the spool"))
        self.lip = spin(spool_adapter.LIP_MIN, spool_adapter.LIP_MAX, s.spool_adapter_lip)
        self.lip.setToolTip(tr("How far the flange reaches past the hub all round"))
        self.rod = spin(spool_adapter.ROD_MIN, spool_adapter.ROD_MAX, s.spool_adapter_rod)
        self.rod.setSingleStep(0.5)
        self.rod.setToolTip(tr("The rod that turns in the middle of the adapter"))
        self.bearing = QCheckBox(tr("608 bearing"))
        self.bearing.setChecked(bool(s.spool_adapter_bearing))
        self.bearing.setToolTip(tr("A pocket for a 608 bearing instead of a hole for a rod, with "
                                   "the axle through under it"))
        self.spokes = QSpinBox()
        self.spokes.setRange(0, spool_adapter.SPOKES_MAX)
        self.spokes.setValue(int(s.spool_adapter_spokes))
        self.spokes.setToolTip(tr("How many spokes hold the hub together; 0 makes the hub solid"))
        self.info = QLabel()
        lay.addRow(tr("Hub (spool hole)"), self.hub)
        lay.addRow(tr("Depth"), self.depth)
        lay.addRow(tr("Lip"), self.lip)
        lay.addRow(tr("Rod"), self.rod)
        lay.addRow("", self.bearing)
        lay.addRow(tr("Spokes"), self.spokes)
        lay.addRow("", self.info)
        for w in (self.hub, self.depth, self.lip, self.rod, self.spokes):
            w.valueChanged.connect(self._changed)
        self.bearing.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        self.rod.setEnabled(not self.bearing.isChecked())
        s = spool_adapter.sizes(**self.options())
        self.info.setText(tr("Hub {h:.1f} mm, bore {b:.1f} mm, {n} spokes")
                          .format(h=s["hub"], b=max(s["bore"], s["pocket"]), n=s["spokes"]))

    def options(self) -> dict:
        return {"hub": round(self.hub.value(), 6), "depth": round(self.depth.value(), 6),
                "lip": round(self.lip.value(), 6), "rod": round(self.rod.value(), 6),
                "bearing": self.bearing.isChecked(), "spokes": int(self.spokes.value())}


class DrawerOrganizerDialog(QDialog):
    """The "Drawer Organizer…" dialog: a tray that drops into a drawer and divides it.

    The length and the width are the inside of the drawer and the clearance how much smaller the
    tray is made on every side so it drops in; the height is how high it stands. The columns and
    rows say into how many equal compartments it is divided, the wall and the floor how thick its
    plastic is and the radius how round the inside corners are. The finger notches cut the top of
    every inner wall so you get a finger down beside an item. Under the fields the size of one
    compartment, because that is what you measure the things you sort against. The values come from
    Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Drawer organizer"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=1.0):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.length = spin(drawer_organizer.LENGTH_MIN, drawer_organizer.LENGTH_MAX,
                           s.drawer_organizer_length, 5.0)
        self.length.setToolTip(tr("The inside length of the drawer the tray drops into"))
        self.width = spin(drawer_organizer.WIDTH_MIN, drawer_organizer.WIDTH_MAX,
                          s.drawer_organizer_width, 5.0)
        self.width.setToolTip(tr("The inside width of the drawer the tray drops into"))
        self.height = spin(drawer_organizer.HEIGHT_MIN, drawer_organizer.HEIGHT_MAX,
                           s.drawer_organizer_height, 5.0)
        self.height.setToolTip(tr("How high the tray stands in the drawer"))
        self.clearance = spin(drawer_organizer.CLEAR_MIN, drawer_organizer.CLEAR_MAX,
                              s.drawer_organizer_clearance, 0.1)
        self.clearance.setToolTip(tr("How much smaller the tray is made on every side, so it drops in"))
        self.cols = QSpinBox()
        self.cols.setRange(drawer_organizer.COLS_MIN, drawer_organizer.COLS_MAX)
        self.cols.setValue(int(s.drawer_organizer_cols))
        self.cols.setToolTip(tr("How many compartments along the length of the drawer"))
        self.rows = QSpinBox()
        self.rows.setRange(drawer_organizer.ROWS_MIN, drawer_organizer.ROWS_MAX)
        self.rows.setValue(int(s.drawer_organizer_rows))
        self.rows.setToolTip(tr("How many compartments along the width of the drawer"))
        self.wall = spin(drawer_organizer.WALL_MIN, drawer_organizer.WALL_MAX,
                         s.drawer_organizer_wall, 0.2)
        self.wall.setToolTip(tr("How thick the walls and the dividers are"))
        self.floor = spin(drawer_organizer.FLOOR_MIN, drawer_organizer.FLOOR_MAX,
                          s.drawer_organizer_floor, 0.2)
        self.floor.setToolTip(tr("How thick the floor the tray stands on is"))
        self.radius = spin(drawer_organizer.RADIUS_MIN, drawer_organizer.RADIUS_MAX,
                           s.drawer_organizer_radius, 0.5)
        self.radius.setToolTip(tr("How round the inside corners of the compartments are"))
        self.finger = QCheckBox(tr("Finger notches"))
        self.finger.setChecked(bool(s.drawer_organizer_finger))
        self.finger.setToolTip(tr("A half-round notch in the top of every inner wall, so you can "
                                  "lift an item out from between two dividers"))
        self.info = QLabel()
        lay.addRow(tr("Length (drawer inside)"), self.length)
        lay.addRow(tr("Width (drawer inside)"), self.width)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Clearance"), self.clearance)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Floor"), self.floor)
        lay.addRow(tr("Radius"), self.radius)
        lay.addRow("", self.finger)
        lay.addRow("", self.info)
        for w in (self.length, self.width, self.height, self.clearance, self.cols, self.rows,
                  self.wall, self.floor, self.radius):
            w.valueChanged.connect(self._changed)
        self.finger.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = drawer_organizer.sizes(**self.options())
        self.info.setText(tr("Compartment {l:.1f} × {w:.1f} mm")
                          .format(l=s["compartment_length"], w=s["compartment_width"]))

    def options(self) -> dict:
        return {"length": round(self.length.value(), 6), "width": round(self.width.value(), 6),
                "height": round(self.height.value(), 6),
                "clearance": round(self.clearance.value(), 6), "cols": int(self.cols.value()),
                "rows": int(self.rows.value()), "wall": round(self.wall.value(), 6),
                "floor": round(self.floor.value(), 6), "radius": round(self.radius.value(), 6),
                "finger": self.finger.isChecked()}


class PlantPotDialog(QDialog):
    """The "Plant Pot…" dialog: a flower pot that drains and the saucer that catches the water.

    The diameter is how wide the pot is at its top and the height how high it stands; the taper is
    how wide the bottom is as a percentage of the top, the wall how thick the plastic is and the
    drain holes how many the water runs out through. The shape is round or a hexagon, which is
    measured across its flats. Under the fields the bottom the pot stands on and what it holds,
    because that is what a plant is bought against. The values come from Settings and are
    remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Plant Pot"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=1.0, suffix=" mm"):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setValue(float(value))
            return sp

        self.shape = QComboBox()
        for label, key in ((tr("Round"), "round"), (tr("Hexagonal"), "hexagon")):
            self.shape.addItem(label, key)
        self.shape.setCurrentIndex(max(0, self.shape.findData(s.plant_pot_shape)))
        self.shape.setToolTip(tr("A hexagon is measured across its flats and does not roll off "
                                 "the table while you press the soil in"))
        self.diameter = spin(plant_pot.D_MIN, plant_pot.D_MAX, s.plant_pot_diameter, 5.0)
        self.diameter.setToolTip(tr("How wide the pot is across at its top"))
        self.height = spin(plant_pot.HEIGHT_MIN, plant_pot.HEIGHT_MAX, s.plant_pot_height, 5.0)
        self.height.setToolTip(tr("How high the pot stands"))
        self.taper = spin(plant_pot.TAPER_MIN, plant_pot.TAPER_MAX, s.plant_pot_taper, 1.0, " %")
        self.taper.setToolTip(tr("How wide the bottom is as a percentage of the top; the smaller "
                                 "the bottom, the better the pot prints without support"))
        self.wall = spin(plant_pot.WALL_MIN, plant_pot.WALL_MAX, s.plant_pot_wall, 0.2)
        self.wall.setToolTip(tr("How thick the plastic of the pot is"))
        self.holes = QSpinBox()
        self.holes.setRange(plant_pot.HOLES_MIN, plant_pot.HOLES_MAX)
        self.holes.setValue(int(s.plant_pot_holes))
        self.holes.setToolTip(tr("How many holes the water runs out through; one stands in the "
                                 "middle of the floor and the rest in a circle round it"))
        self.saucer = QCheckBox(tr("Saucer"))
        self.saucer.setChecked(bool(s.plant_pot_saucer))
        self.saucer.setToolTip(tr("Print the saucer that catches the water beside the pot, so the "
                                  "two stand side by side on the bed"))
        self.info = QLabel()
        lay.addRow(tr("Shape"), self.shape)
        lay.addRow(tr("Top diameter"), self.diameter)
        lay.addRow(tr("Height"), self.height)
        lay.addRow(tr("Bottom taper"), self.taper)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Drain holes"), self.holes)
        lay.addRow("", self.saucer)
        lay.addRow("", self.info)
        for w in (self.diameter, self.height, self.taper, self.wall):
            w.valueChanged.connect(self._changed)
        self.holes.valueChanged.connect(self._changed)
        self.shape.currentIndexChanged.connect(self._changed)
        self.saucer.toggled.connect(self._changed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._changed()

    def _changed(self, *_):
        s = plant_pot.sizes(**self.options())
        self.info.setText(tr("Bottom {d:.1f} mm · holds {v:.2f} l")
                          .format(d=s["bottom"], v=s["capacity"]))

    def options(self) -> dict:
        return {"diameter": round(self.diameter.value(), 6),
                "height": round(self.height.value(), 6), "taper": round(self.taper.value(), 6),
                "wall": round(self.wall.value(), 6), "holes": int(self.holes.value()),
                "shape": self.shape.currentData(), "saucer": self.saucer.isChecked()}


class HoseAdapterDialog(QDialog):
    """The "Hose Adapter…" dialog: a reducer between two hoses.

    The two diameters are the inside of the hoses, the bottom one and the top one as it prints;
    the wall is the plastic around the water way and the barbs how many grip each hose. Plain
    ends are for tube that is pushed on, and grey out the barbs. Under the fields the height and
    the collar the adapter will have. The values come from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Hose adapter"))
        lay = QFormLayout(self)

        def spin(lo, hi, value):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.d1 = spin(hose_adapter.D_MIN, hose_adapter.D_MAX, s.hose_d1)
        self.d1.setToolTip(tr("The inside of the hose that goes over the bottom end"))
        self.d2 = spin(hose_adapter.D_MIN, hose_adapter.D_MAX, s.hose_d2)
        self.d2.setToolTip(tr("The inside of the hose that goes over the top end"))
        self.wall = spin(hose_adapter.WALL_MIN, hose_adapter.WALL_MAX, s.hose_wall)
        self.wall.setToolTip(tr("The plastic around the water way"))
        self.barbs = QSpinBox()
        self.barbs.setRange(hose_adapter.BARBS_MIN, hose_adapter.BARBS_MAX)
        self.barbs.setValue(int(s.hose_barbs))
        self.barbs.setToolTip(tr("How many barbs grip each hose"))
        self.straight = QCheckBox(tr("Plain ends (push-on tube)"))
        self.straight.setChecked(bool(s.hose_straight))
        self.info = QLabel()
        lay.addRow(tr("Hose bottom"), self.d1)
        lay.addRow(tr("Hose top"), self.d2)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Barbs"), self.barbs)
        lay.addRow("", self.straight)
        lay.addRow("", self.info)
        for w in (self.d1, self.d2, self.wall, self.barbs):
            w.valueChanged.connect(self._changed)
        self.straight.toggled.connect(self._changed)
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addRow(self.bb)
        self._changed()

    def _changed(self, *_):
        self.barbs.setEnabled(not self.straight.isChecked())
        o = self.options()
        try:
            s = hose_adapter.sizes(**o)
        except ValueError as e:
            self.info.setText(str(e))
            self.bb.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        self.bb.button(QDialogButtonBox.Ok).setEnabled(True)
        self.info.setText(tr("Height {h:.1f} mm, collar {c:.1f} mm, water way {b:.1f} mm")
                          .format(h=s["length"], c=s["collar_d"],
                                  b=min(s["bore1"], s["bore2"])))

    def options(self) -> dict:
        return {"d1": round(self.d1.value(), 6), "d2": round(self.d2.value(), 6),
                "wall": round(self.wall.value(), 6), "barbs": int(self.barbs.value()),
                "straight": self.straight.isChecked()}


class GridfinityDialog(QDialog):
    """The "Gridfinity…" dialog: a bin or a baseplate of the open storage grid of 42 mm.

    The kind says which part, the columns and rows how many squares of the grid it covers; a bin
    also has its height in units of 7 mm, its compartments, the stacking lip, the magnet pockets,
    the finger scoop in the front and the shelf for a label in the back. What only a bin has is
    greyed out for a baseplate. The values come from Settings and
    are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Gridfinity"))
        lay = QFormLayout(self)

        def count(lo, hi, value, tip):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(int(value))
            sp.setToolTip(tip)
            return sp

        self.kind = QComboBox()
        self.kind.addItem(tr("Bin"), "bin")
        self.kind.addItem(tr("Baseplate"), "baseplate")
        self.kind.setCurrentIndex(max(0, self.kind.findData(s.grid_kind)))
        self.cols = count(gridfinity.COLS_MIN, gridfinity.COLS_MAX, s.grid_cols,
                          tr("How many squares of 42 mm across"))
        self.rows = count(gridfinity.COLS_MIN, gridfinity.COLS_MAX, s.grid_rows,
                          tr("How many squares of 42 mm deep"))
        self.units = count(gridfinity.UNITS_MIN, gridfinity.UNITS_MAX, s.grid_units,
                           tr("The height of the bin in units of 7 mm, the foot included"))
        self.divisions = count(gridfinity.DIV_MIN, gridfinity.DIV_MAX, s.grid_divisions,
                               tr("How many compartments the bin has along its length"))
        self.lip = QCheckBox(tr("Stacking lip"))
        self.lip.setChecked(bool(s.grid_lip))
        self.lip.setToolTip(tr("A lip on top of the wall, in which the foot of a bin above stands"))
        self.magnets = QCheckBox(tr("Magnet holes"))
        self.magnets.setChecked(bool(s.grid_magnets))
        self.magnets.setToolTip(tr("Four pockets of 6.5 by 2.4 mm in the bottom of every foot, for "
                                   "magnets that hold the bin on a steel plate"))
        self.label = QCheckBox(tr("Label shelf"))
        self.label.setChecked(bool(s.grid_label))
        self.label.setToolTip(tr("A shelf under the top of the inside of the back wall, at "
                                 "45 degrees, so a printed label stands in the bin and you read it "
                                 "from the front"))
        self.scoop = QCheckBox(tr("Finger scoop"))
        self.scoop.setChecked(bool(s.grid_scoop))
        self.scoop.setToolTip(tr("A rounding between the floor and the inside of the front wall, so "
                                 "you slide small parts out with a finger"))
        lay.addRow(tr("Part"), self.kind)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Height units"), self.units)
        lay.addRow(tr("Compartments"), self.divisions)
        lay.addRow("", self.lip)
        lay.addRow("", self.magnets)
        lay.addRow("", self.scoop)
        lay.addRow("", self.label)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _kind_changed(self, *_):
        is_bin = self.kind.currentData() == "bin"
        for w in (self.units, self.divisions, self.lip, self.magnets, self.scoop, self.label):
            w.setEnabled(is_bin)

    def options(self) -> dict:
        return {"kind": str(self.kind.currentData()),
                "cols": int(self.cols.value()),
                "rows": int(self.rows.value()),
                "units": int(self.units.value()),
                "divisions": int(self.divisions.value()),
                "lip": bool(self.lip.isChecked()),
                "magnets": bool(self.magnets.isChecked()),
                "scoop": bool(self.scoop.isChecked()),
                "label": bool(self.label.isChecked())}


class CableChainDialog(QDialog):
    """The "Cable Chain…" dialog: a drag chain printed in one piece.

    The width and the height are the inside, the room for the cables; the wall is the plastic of
    the plates and the bars, and the links say how long the chain is. The values come from
    Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Cable chain"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.width = spin(cable_chain.WIDTH_MIN, cable_chain.WIDTH_MAX, s.chain_width, 1.0)
        self.width.setToolTip(tr("The room for the cables, across the chain"))
        self.height = spin(cable_chain.HEIGHT_MIN, cable_chain.HEIGHT_MAX, s.chain_height, 1.0)
        self.height.setToolTip(tr("The room for the cables, from bar to bar"))
        self.wall = spin(cable_chain.WALL_MIN, cable_chain.WALL_MAX, s.chain_wall, 0.2)
        self.wall.setToolTip(tr("The plastic of the side plates and the cross bars"))
        self.links = QSpinBox()
        self.links.setRange(cable_chain.LINKS_MIN, cable_chain.LINKS_MAX)
        self.links.setValue(s.chain_links)
        self.links.setToolTip(tr("How many links the chain has"))
        self.bend = QDoubleSpinBox()
        self.bend.setRange(cable_chain.BEND_MIN, cable_chain.BEND_MAX)
        self.bend.setDecimals(0)
        self.bend.setSingleStep(5.0)
        self.bend.setSuffix(" °")
        self.bend.setValue(float(s.chain_bend))
        self.bend.setToolTip(tr("How far one link turns on the next, one way; the other way the "
                                "chain stays straight"))
        lay.addRow(tr("Inside width"), self.width)
        lay.addRow(tr("Inside height"), self.height)
        lay.addRow(tr("Wall"), self.wall)
        lay.addRow(tr("Links"), self.links)
        lay.addRow(tr("Bend"), self.bend)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"width": round(self.width.value(), 6),
                "height": round(self.height.value(), 6),
                "wall": round(self.wall.value(), 6),
                "links": int(self.links.value()),
                "bend": round(self.bend.value(), 6)}


class BinRackDialog(QDialog):
    """The "Bin Wall Rack…" dialog: a plate on the wall with a rail at the top of every row.

    The columns and rows make the grid and the cell sizes say how much room a bin has in it: a
    bin with the wall hook that is as high as a row or lower hangs in it without touching the one
    under it. The plate thickness and the countersunk screws hold it on the wall. The values come
    from Settings and are remembered.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Bin wall rack"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(1)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.cols = QSpinBox()
        self.cols.setRange(bin_rack.COLS_MIN, bin_rack.COLS_MAX)
        self.cols.setValue(s.rack_cols)
        self.cols.setToolTip(tr("How many cells lie next to each other across the rack"))
        self.rows = QSpinBox()
        self.rows.setRange(bin_rack.ROWS_MIN, bin_rack.ROWS_MAX)
        self.rows.setValue(s.rack_rows)
        self.rows.setToolTip(tr("How many rows of bins hang above each other, each on its own rail"))
        self.cell_w = spin(bin_rack.CELL_W_MIN, bin_rack.CELL_W_MAX, s.rack_cell_w, 5.0)
        self.cell_w.setToolTip(tr("The width of a cell: the length of a bin with some room next "
                                  "to it"))
        self.cell_h = spin(bin_rack.CELL_H_MIN, bin_rack.CELL_H_MAX, s.rack_cell_h, 5.0)
        self.cell_h.setToolTip(tr("The height of a row: a bin up to this height hangs in it "
                                  "without touching the bin under it"))
        self.plate = spin(bin_rack.PLATE_MIN, bin_rack.PLATE_MAX, s.rack_plate, 0.5)
        self.plate.setToolTip(tr("The thickness of the plate against the wall"))
        self.screw = QComboBox()
        self.screw.addItem(tr("None"), "none")
        for size in bin_rack.SCREWS[1:]:
            self.screw.addItem(size, size)
        self.screw.setCurrentIndex(max(0, self.screw.findData(s.rack_screw)))
        self.screw.setToolTip(tr("The flat head screws that hold the rack on the wall, two in "
                                 "every row"))
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Cell width"), self.cell_w)
        lay.addRow(tr("Row height"), self.cell_h)
        lay.addRow(tr("Plate"), self.plate)
        lay.addRow(tr("Screws"), self.screw)
        self.panel_cols = QSpinBox()
        self.panel_cols.setRange(0, bin_rack.COLS_MAX)
        self.panel_cols.setSpecialValueText(tr("One piece"))
        self.panel_cols.setValue(s.rack_panel_cols)
        self.panel_cols.setToolTip(tr("For a rack wider than the bed: panels of this many columns "
                                      "that slide together with dovetails"))
        lay.addRow(tr("Columns per panel"), self.panel_cols)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"cols": int(self.cols.value()),
                "panel_cols": int(self.panel_cols.value()),
                "rows": int(self.rows.value()),
                "cell_w": round(self.cell_w.value(), 6),
                "cell_h": round(self.cell_h.value(), 6),
                "plate": round(self.plate.value(), 6),
                "screw": str(self.screw.currentData())}
