"""The 3D printing side of the main window: the printed parts and the print tools of
Volura.

Step 2 of splitting `volura/app.py`: the methods that build or change a printed part moved
here verbatim, into a plain mixin that `MainWindow` inherits. The spur gear, the project
box, the knob, the standoff, the label plate, the 3D text, the cable clip, the solid tools
and the simplify, the mouse ears, the parts on the bed, the screw thread and the screw hole,
the clearance and the overhang test, the cut & join, the print check and the help.

The methods that ask something with a message box or an input dialog stay in
`volura/app.py`, together with the actions that call all of this; the window still owns
everything, this module only holds the code. The list of the tools, the title of the print
check panel and the help text live here too, because the print check and the help show
them; the app imports them back.
"""
from __future__ import annotations

import manifold3d as mf
import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFontComboBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox, QTextBrowser,
                               QVBoxLayout)

from . import (background, battery_holder, bin_rack, cable_chain, cable_clip, enclosure,
               fan_guard, gears,
               gridfinity, grommet, hinge, hose_adapter, joints, knob, label_plate, ops, pegboard,
               standoff, storage_bin, text3d, theme, threads)
from .app_dialogs import SimplifyDialog
from .brep import Entities
from .i18n import tr, tr_noop, trn
from .inference import GeoCache
from .model import ROOT
from .part_dialogs import (BatteryHolderDialog, BinRackDialog, CableChainDialog,
                           CableClipDialog, FanGuardDialog, GridfinityDialog, GrommetDialog,
                           HingeDialog, HoseAdapterDialog,
                           KnobDialog, LabelPlateDialog, PegboardDialog, ProjectBoxDialog,
                           StandoffDialog, StorageBinDialog, Text3DDialog)


TOOL_INFO = [
    # name, label (English, translated at use), icon, shortcut
    ("select", tr_noop("Select"), "select", "Space"),
    ("eraser", tr_noop("Eraser"), "eraser", "E"),
    ("paint", tr_noop("Paint Bucket"), "paint", "B"),
    ("line", tr_noop("Line"), "line", "L"),
    ("rectangle", tr_noop("Rectangle"), "rectangle", "R"),
    ("circle", tr_noop("Circle"), "circle", "C"),
    ("polygon", tr_noop("Polygon"), "polygon", None),
    ("text", tr_noop("3D Text"), "text", "Shift+T"),
    ("pushpull", tr_noop("Push/Pull"), "pushpull", "P"),
    ("offset", tr_noop("Offset"), "offset", "F"),
    ("move", tr_noop("Move"), "move", "M"),
    ("rotate", tr_noop("Rotate"), "rotate", "Q"),
    ("scale", tr_noop("Scale"), "scale", "S"),
    ("tape", tr_noop("Tape Measure"), "tape", "T"),
    ("orbit", tr_noop("Orbit"), "orbit", "O"),
    ("pan", tr_noop("Pan"), "pan", "H"),
    ("zoom", tr_noop("Zoom"), "zoom", "Z"),
]


PANEL_PRINT_CHECK = tr_noop("Print Check")


#: Help > Keyboard Shortcuts; {rows} are the tools with a shortcut, {mouse}, {eraser} and
#: {pushpull} say what the scheme that is on now does with those
HELP_HTML = tr_noop("""
<h2>Volura - how it works</h2>
<p>Volura works like SketchUp: draw a flat face and make a volume of it with <b>Push/Pull</b>.</p>
<ol>
<li>Draw edges with Line (L), Rectangle (R), Circle (C) or Polygon. A closed shape becomes a face.</li>
<li>Edges split each other, a line across a face splits the face; with the Eraser{eraser} you erase edges
(or parts of them) and faces are merged again.</li>
<li>Push/Pull{pushpull} works on every face or part of a face: up = add, down = recess or hole.
Type a distance and Enter; the direction follows your mouse.</li>
<li>Select a shape (triple-click) and make a <b>group</b> of it (G): every group is a separate
printable part with its own color. Double-click a group to edit it.</li>
<li><b>Multicolor</b>: hold <b>Alt</b> while pushing/pulling: inward makes an inlay (recess +
filling as a separate group), outward a separate part. Give it a color with the Paint Bucket (B) and
export as 3MF.</li>
</ol>
<h3>Keyboard shortcuts</h3>
<table cellpadding="3">{rows}
<tr><td><b>A</b></td><td>2 Point Arc (in Draw &gt; Arcs also Arc, 3 Point Arc and Pie)</td></tr>
<tr><td><b>Shift+Z</b></td><td>Zoom Extents</td></tr>
<tr><td><b>Middle mouse button</b></td><td>{mouse}. Scroll wheel: zoom</td></tr>
<tr><td><b>Arrow keys</b></td><td>Lock an axis (right = red/X, left = green/Y, up = blue/Z)</td></tr>
<tr><td><b>Shift</b> (hold)</td><td>Lock the current direction</td></tr>
<tr><td><b>Ctrl</b></td><td>Move/rotate: copy. Push/Pull: new start face</td></tr>
<tr><td><b>Alt</b></td><td>Push/Pull: separate part / inlay</td></tr>
<tr><td><b>G</b></td><td>Make a component or group of the selection</td></tr>
<tr><td><b>Esc</b></td><td>Cancel the current operation</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>Undo / redo</td></tr>
</table>
<h3>Measurements (bottom right)</h3>
<p>Just start typing during an operation. Examples: <code>25</code>, <code>2.5cm</code>,
<code>40;20</code> (rectangle), <code>64s</code> (segments), <code>10;0;5</code> (move),
<code>5x</code> or <code>/5</code> after a copy (array), <code>45</code> (degrees when rotating).</p>
""")


class PrintMixin:
    """The 3D printing methods of `MainWindow`; the window itself lives in the app."""

    # ------------------------------------------------------------ spur gear
    def make_spur_gear_model(self, options):
        """The wheel as a printed part in the model: one undo step, standing on the bed.

        A wheel of many teeth is a contour of tens of thousands of points, so the section
        and the extrusion run in the background, like the clearance test, and the window
        keeps responding. The wheel stands on the bed and takes the middle of the free
        bed, or the free place 10 mm to the right of what is already in the model.
        """
        self.set_mode("3d")
        doc = self.doc
        teeth, module = int(options["teeth"]), float(options["module"])
        thickness = float(options["thickness"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the spur gear…"))
            loops = gears.spur_gear(teeth, module=module, pressure_angle=options["angle"],
                                    bore=options["bore"], backlash=options["backlash"])
            # the outline runs counter-clockwise and the bore clockwise: EvenOdd makes the
            # bore a hole, whatever the way round the two loops are
            sec = mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float))
                                   for q in loops], mf.FillRule.EvenOdd)
            m = mf.Manifold.extrude(sec, thickness)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the wheel away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Spur Gear"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Spur gear with {n} teeth added (pitch diameter {d} mm, "
                                    "{t} mm thick).")
                                 .format(n=teeth, d=f"{gears.pitch_diameter(teeth, module):.2f}",
                                         t=f"{thickness:.2f}"))

        def failed(exc):
            self.view.set_status(tr("Spur gear failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the spur gear…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ project box
    def project_box_dialog(self):
        """Tools > Print Tools > Project Box…: an enclosure with a lid that fits into it."""
        dlg = ProjectBoxDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.box3d_length = float(options["length"])
        self.settings.box3d_width = float(options["width"])
        self.settings.box3d_height = float(options["height"])
        self.settings.box3d_wall = float(options["wall"])
        self.settings.box3d_floor = float(options["floor"])
        self.settings.box3d_radius = float(options["radius"])
        self.settings.box3d_lid = float(options["lid"])
        self.settings.box3d_lip = float(options["lip"])
        self.settings.box3d_clearance = float(options["clearance"])
        self.settings.box3d_screws = str(options.get("screws", "") or "")
        self.settings.box3d_vents = int(options.get("vents", 0) or 0)
        self.settings.box3d_board = str(options.get("board", "") or "")
        self.settings.box3d_board_height = float(options.get("board_height", 5.0))
        self.settings.save()
        made = self.make_project_box_model(options)
        self.setFocus()
        return made

    def make_project_box_model(self, options):
        """The box and its lid as printed parts: one undo step, side by side on the bed.

        The two rounded contours and the booleans that make them run in the background, like
        the spur gear, so the window keeps responding on a big box. The pair takes the free
        place on the bed with the lid 10 mm to the right of the base; the lid is already
        flipped, so its flat top lies on the bed and its lip stands up in the air and
        neither part needs support.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: float(options[k]) for k in ("length", "width", "height", "wall", "floor",
                                               "radius", "lid", "lip", "clearance")}
        screws = str(options.get("screws", "") or "")
        vents = int(options.get("vents", 0) or 0)
        board = str(options.get("board", "") or "")
        board_height = float(options.get("board_height", 5.0))
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size
        gap = 10.0                                  # mm of bed between the two parts
        wide = size["length"] + 2.0 * size["wall"]  # the base and the lid are equally wide
        span = 2.0 * wide + gap

        def work(report):
            report(0.1, tr("Building the project box…"))
            base, lid = enclosure.project_box(**size, screws=screws, vents=vents, board=board,
                                              board_height=board_height)
            for m in (base, lid):
                if m.status() != mf.Error.NoError or m.is_empty():
                    raise ValueError(tr("the result is not a valid volume"))
            # both parts are centred on the origin: put the pair in the free place on the bed
            x = 0.5 * bx if bb is None else float(bb[1][0]) + gap + 0.5 * span
            return (base.translate((x - 0.5 * span + 0.5 * wide, 0.5 * by, 0.0)),
                    lid.translate((x + 0.5 * span - 0.5 * wide, 0.5 * by, 0.0)))

        def done(parts):
            doc.checkpoint()                # one undo takes the box and the lid away
            color = self.settings.current_color
            new = [ops.group_from_manifold(doc, parts[0], color, tr("Box")),
                   ops.group_from_manifold(doc, parts[1], color, tr("Lid"))]
            doc.set_selection([("g", g.id, g.id) for g in new])
            doc.notify()
            self.view.set_status(tr("Project box {l} × {w} × {h} mm added (inside).")
                                 .format(l=f"{size['length']:g}", w=f"{size['width']:g}",
                                         h=f"{size['height']:g}"))

        def failed(exc):
            self.view.set_status(tr("Project box failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the project box…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ knob
    def knob_dialog(self):
        """Tools > Print Tools > Knob…: a thumb knob that turns on a bolt or a shaft."""
        dlg = KnobDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.knob_diameter = float(options["diameter"])
        self.settings.knob_height = float(options["height"])
        self.settings.knob_grip = str(options["grip"])
        self.settings.knob_lobes = int(options["lobes"])
        self.settings.knob_kind = str(options["kind"])
        self.settings.knob_bore = str(options["bore"])
        self.settings.knob_shaft_d = float(options["shaft_d"])
        self.settings.knob_flat = float(options["flat"])
        self.settings.knob_clearance = float(options["clearance"])
        self.settings.save()
        made = self.make_knob_model(options)
        self.setFocus()
        return made

    def make_knob_model(self, options):
        """The knob as a printed part: one undo step, standing on the bed next to the model.

        The rim is a contour of hundreds of points and the pocket is a boolean, so the knob
        is built in the background, like the spur gear, and the window keeps responding. The
        knob stands on the bed in the middle of the free bed, or on the free place 10 mm to
        the right of what is already in the model, so a second knob does not land in the
        first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("diameter", "height", "grip", "lobes", "kind",
                                        "bore", "shaft_d", "flat", "clearance")}
        diameter = float(options["diameter"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the knob…"))
            m = knob.knob(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the knob away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Knob"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Knob of {d} mm added.").format(d=f"{diameter:g}"))

        def failed(exc):
            self.view.set_status(tr("Knob failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the knob…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ standoff
    def standoff_dialog(self):
        """Tools > Print Tools > Standoff…: a spacer that holds a PCB or a panel off a surface."""
        dlg = StandoffDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.standoff_height = float(options["height"])
        self.settings.standoff_outer = float(options["outer"])
        self.settings.standoff_shape = str(options["shape"])
        self.settings.standoff_kind = str(options["kind"])
        self.settings.standoff_bore = str(options["bore"])
        self.settings.standoff_count = int(options["count"])
        self.settings.standoff_clearance = float(options["clearance"])
        self.settings.save()
        made = self.make_standoff_model(options)
        self.setFocus()
        return made

    def make_standoff_model(self, options):
        """The standoffs as printed parts: one undo step, on the bed next to the model.

        A row is up to twelve bodies and every hole is a boolean, so the row is built in the
        background, like the knob, and the window keeps responding. It stands on the bed in the
        middle of the free bed, or on the free place 10 mm to the right of what is already in
        the model, so a second row does not land in the first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("height", "outer", "shape", "bore", "kind",
                                        "clearance", "count")}
        height, n = float(options["height"]), int(options["count"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the standoffs…"))
            m = standoff.standoff(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the standoffs away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Standoff"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(trn("{n} standoff of {h} mm added.",
                                     "{n} standoffs of {h} mm added.", n)
                                 .format(n=n, h=f"{height:g}"))

        def failed(exc):
            self.view.set_status(tr("Standoff failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the standoffs…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ label plate
    def label_plate_dialog(self):
        """Tools > Print Tools > Label Plate…: a printable name tag or drawer label."""
        dlg = LabelPlateDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.label_text = str(options["text"]).strip()
        self.settings.label_height = float(options["height"])
        self.settings.label_thickness = float(options["thickness"])
        self.settings.label_relief = float(options["relief"])
        self.settings.label_mode = str(options["mode"])
        self.settings.label_margin = float(options["margin"])
        self.settings.label_radius = float(options["radius"])
        self.settings.label_hole = float(options["hole"])
        self.settings.save()
        made = self.make_label_plate_model(options)
        self.setFocus()
        return made

    def make_label_plate_model(self, options):
        """The label plate as a printed part: one undo step, on the bed next to the model.

        The text is hundreds of contours and the letters are a boolean with the plate, so the
        label is built in the background, like the standoffs, and the window keeps responding. It
        stands on the bed in the middle of the free bed, or on the free place 10 mm to the right
        of what is already in the model, so a second label does not land in the first one. The
        group is named after the first line of the text, because that is what the user typed.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("text", "height", "thickness", "relief", "mode",
                                        "margin", "radius", "hole")}
        name = str(options["text"]).split("\n")[0].strip() or tr("Label Plate")
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the label plate…"))
            m = label_plate.label_plate(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the label away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, name)
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Label plate added."))

        def failed(exc):
            self.view.set_status(tr("Label plate failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the label plate…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ 3D text
    def text3d_dialog(self):
        """Draw > 3D Text…: a part that is made of letters."""
        dlg = Text3DDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.text3d_text = str(options["text"]).strip()
        self.settings.text3d_family = str(options["family"])
        self.settings.text3d_height = float(options["height"])
        self.settings.text3d_depth = float(options["depth"])
        self.settings.text3d_bold = bool(options["bold"])
        self.settings.text3d_align = str(options["align"])
        self.settings.save()
        made = self.make_text3d_model(options)
        self.setFocus()
        return made

    def make_text3d_model(self, options):
        """The 3D text as a printed part: one undo step, on the bed next to the model.

        The letters are hundreds of contours and several lines are a block of them, so the text
        is built in the background, like the label plate, and the window keeps responding. It
        stands on the bed in the middle of the free bed, or on the free place 10 mm to the right
        of what is already in the model, so a second text does not land in the first one. The
        group is named after the first line of the text, because that is what the user typed.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("text", "family", "height", "depth", "bold", "align")}
        name = str(options["text"]).split("\n")[0].strip() or tr("3D Text")
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the 3D text…"))
            m = text3d.text3d(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the text away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, name)
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("3D text added."))

        def failed(exc):
            self.view.set_status(tr("3D text failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the 3D text…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ cable clip
    def cable_clip_dialog(self):
        """Tools > Print Tools > Cable Clip…: a clip that holds a cable against a wall or a desk."""
        dlg = CableClipDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.clip_cable_d = float(options["cable_d"])
        self.settings.clip_width = float(options["width"])
        self.settings.clip_wall = float(options["wall"])
        self.settings.clip_opening = float(options["opening"])
        self.settings.clip_base = float(options["base"])
        self.settings.clip_bore = str(options["bore"])
        self.settings.clip_count = int(options["count"])
        self.settings.save()
        made = self.make_cable_clip_model(options)
        self.setFocus()
        return made

    def make_cable_clip_model(self, options):
        """The cable clips as printed parts: one undo step, on the bed next to the model.

        A row is up to twelve clips and the mouth and the screw hole are booleans, so the row is
        built in the background, like the standoffs, and the window keeps responding. It stands on
        the bed in the middle of the free bed, or on the free place 10 mm to the right of what is
        already in the model, so a second row does not land in the first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("cable_d", "width", "wall", "opening", "base", "bore",
                                        "count")}
        cable_d, n = float(options["cable_d"]), int(options["count"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the cable clips…"))
            m = cable_clip.cable_clip(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the clips away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Cable clip"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(trn("{n} cable clip for {d} mm cable added.",
                                     "{n} cable clips for {d} mm cable added.", n)
                                 .format(n=n, d=f"{cable_d:g}"))

        def failed(exc):
            self.view.set_status(tr("Cable clip failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the cable clips…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ print-in-place hinge
    def hinge_dialog(self):
        """Draw > Parts > Print-in-Place Hinge…: a hinge that prints as one piece and turns."""
        dlg = HingeDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.hinge_length = float(options["length"])
        self.settings.hinge_width = float(options["width"])
        self.settings.hinge_thickness = float(options["thickness"])
        self.settings.hinge_knuckles = int(options["knuckles"])
        self.settings.hinge_clearance = float(options["clearance"])
        self.settings.hinge_pin_d = float(options["pin_d"])
        self.settings.save()
        made = self.make_hinge_model(options)
        self.setFocus()
        return made

    def make_hinge_model(self, options):
        """The hinge as a printed part: one undo step, on the bed next to the model.

        The knuckles and the hole in them are a dozen booleans, so the hinge is built in the
        background, like the cable clips, and the window keeps responding. It lies flat on the bed
        in the middle of the free bed, or on the free place 10 mm to the right of what is already
        in the model, so a second hinge does not land in the first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("length", "width", "thickness", "knuckles", "clearance",
                                        "pin_d")}
        length = float(options["length"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the hinge…"))
            m = hinge.hinge(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the hinge away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color,
                                        tr("Print-in-place hinge"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Print-in-place hinge of {l} mm added.").format(
                l=f"{length:g}"))

        def failed(exc):
            self.view.set_status(tr("Print-in-place hinge failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the hinge…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ battery holder
    def battery_holder_dialog(self):
        """Draw > Parts > Battery Holder…: a block that keeps round cells side by side."""
        dlg = BatteryHolderDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.battery_cell = str(options["cell"])
        self.settings.battery_count = int(options["count"])
        self.settings.battery_wall = float(options["wall"])
        self.settings.battery_clearance = float(options["clearance"])
        self.settings.battery_grip = float(options["grip"])
        self.settings.battery_screw = str(options["screw"])
        self.settings.save()
        made = self.make_battery_holder_model(options)
        self.setFocus()
        return made

    def make_battery_holder_model(self, options):
        """The holder as a printed part: one undo step, on the bed next to the model.

        The cradles, the contact slots and the mounting holes are a dozen booleans, so the holder is
        built in the background, like the hinge, and the window keeps responding. It stands on the
        bed in the middle of the free bed, or on the free place 10 mm to the right of what is
        already in the model, so a second holder does not land in the first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("cell", "count", "wall", "clearance", "grip", "screw")}
        cell, n = str(options["cell"]), int(options["count"])
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the battery holder…"))
            m = battery_holder.battery_holder(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the holder away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Battery holder"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(trn("{n} cell holder for {cell} added.",
                                     "Holder for {n} {cell} cells added.", n)
                                 .format(n=n, cell=cell))

        def failed(exc):
            self.view.set_status(tr("Battery holder failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the battery holder…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ storage bin
    def storage_bin_dialog(self):
        """Draw > Parts > Storage Bin…: an open-top box with compartments and a label ledge."""
        dlg = StorageBinDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.bin_length = float(options["length"])
        self.settings.bin_width = float(options["width"])
        self.settings.bin_height = float(options["height"])
        self.settings.bin_wall = float(options["wall"])
        self.settings.bin_floor = float(options["floor"])
        self.settings.bin_radius = float(options["radius"])
        self.settings.bin_cols = int(options["cols"])
        self.settings.bin_rows = int(options["rows"])
        self.settings.bin_label = bool(options["label"])
        self.settings.bin_hanger = bool(options.get("hanger", False))
        self.settings.bin_stack = bool(options.get("stack", False))
        self.settings.bin_scoop = bool(options.get("scoop", False))
        self.settings.bin_label_text = str(options.get("label_text", ""))
        self.settings.save()
        made = self.make_storage_bin_model(options)
        self.setFocus()
        return made

    def make_storage_bin_model(self, options):
        """The bin as a printed part: one undo step, on the bed next to the model.

        The cavity, the dividers and the ledge are a handful of booleans, so the bin is built in
        the background, like the holder, and the window keeps responding. It stands in the middle
        of the free bed, or on the free place 10 mm to the right of what is already in the model,
        so a second bin does not land in the first one.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("length", "width", "height", "wall", "floor", "radius",
                                        "cols", "rows", "label")}
        size["hanger"] = bool(options.get("hanger", False))
        size["stack"] = bool(options.get("stack", False))
        size["scoop"] = bool(options.get("scoop", False))
        text = str(options.get("label_text", "") or "").strip() if size["label"] else ""
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the storage bin…"))
            m = storage_bin.storage_bin(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            m = m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))
            if not text:
                return m, None
            # the label lies on the bed in front of the bin, along its length
            report(0.7, tr("Building the label…"))
            label = storage_bin.bin_label(text, **{k: size[k] for k in (
                "length", "width", "height", "wall", "floor", "radius", "scoop", "stack")})
            b = np.asarray(m.bounding_box(), dtype=float)
            c = np.asarray(label.bounding_box(), dtype=float)
            return m, label.translate((0.5 * (b[0] + b[3]) - 0.5 * (c[0] + c[3]),
                                       b[1] - 10.0 - c[4], -c[2]))

        def done(made):
            m, label = made
            doc.checkpoint()                # one undo takes the bin (and its label) away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Storage bin"))
            sel = [("g", g.id, g.id)]
            if label is not None:
                lg = ops.group_from_manifold(doc, label, self.settings.current_color, tr("Label"))
                sel.append(("g", lg.id, lg.id))
            doc.set_selection(sel)
            doc.notify()
            self.view.set_status(tr("Storage bin added.") if label is None
                                 else tr("Storage bin and its label added."))

        def failed(exc):
            self.view.set_status(tr("Storage bin failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the storage bin…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ pegboard
    def pegboard_dialog(self):
        """Draw > Parts > Pegboard…: the board, or a hook, a shelf or a tool holder for it."""
        dlg = PegboardDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        for key in ("kind", "width", "reach", "board", "size", "cols", "rows", "standoff"):
            setattr(self.settings, "peg_" + key, options[key])
        self.settings.save()
        made = self.make_pegboard_model(options)
        self.setFocus()
        return made

    def make_pegboard_model(self, options):
        """The part as printed - on its side, or a board face down: one undo step, next to the
        model."""
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("kind", "width", "reach", "board", "size", "cols", "rows",
                                        "standoff") if k in options}
        name = tr("Pegboard") if options["kind"] == "board" else tr("Pegboard hook")
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the pegboard part…"))
            m = pegboard.pegboard_part(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the part away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, name)
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Pegboard part added."))

        def failed(exc):
            self.view.set_status(tr("Pegboard part failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the pegboard part…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ fan guard
    def fan_guard_dialog(self):
        """Draw > Parts > Fan Guard…: a grille for a square fan."""
        dlg = FanGuardDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.fan_size = options["fan"]
        for key in ("pattern", "thickness", "bar", "gap"):
            setattr(self.settings, "fan_" + key, options[key])
        self.settings.save()
        made = self.make_fan_guard_model(options)
        self.setFocus()
        return made

    def make_fan_guard_model(self, options):
        """The guard lying flat as it prints: one undo step, next to the model."""
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("fan", "pattern", "thickness", "bar", "gap")}
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the fan guard…"))
            m = fan_guard.fan_guard(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the guard away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Fan guard"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Fan guard added."))

        def failed(exc):
            self.view.set_status(tr("Fan guard failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the fan guard…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ desk grommet
    def grommet_dialog(self):
        """Draw > Parts > Desk Grommet…: a cable pass-through for a hole in a desk top."""
        dlg = GrommetDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        for key in ("hole", "desk", "flange", "wall", "slot", "lid"):
            setattr(self.settings, "grommet_" + key, options[key])
        self.settings.save()
        made = self.make_grommet_model(options)
        self.setFocus()
        return made

    def make_grommet_model(self, options):
        """The grommet with its lid on the bed as it prints: one undo step, next to the model."""
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("hole", "desk", "flange", "wall", "slot", "lid")}
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the desk grommet…"))
            m = grommet.grommet(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the grommet away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Desk grommet"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Desk grommet added."))

        def failed(exc):
            self.view.set_status(tr("Desk grommet failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the desk grommet…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ hose adapter
    def hose_adapter_dialog(self):
        """Draw > Parts > Hose Adapter…: a reducer between two hoses."""
        dlg = HoseAdapterDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        for key in ("d1", "d2", "wall", "barbs", "straight"):
            setattr(self.settings, "hose_" + key, options[key])
        self.settings.save()
        made = self.make_hose_adapter_model(options)
        self.setFocus()
        return made

    def make_hose_adapter_model(self, options):
        """The adapter standing upright as it prints: one undo step, next to the model."""
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("d1", "d2", "wall", "barbs", "straight")}
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the hose adapter…"))
            m = hose_adapter.hose_adapter(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the adapter away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Hose adapter"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Hose adapter added."))

        def failed(exc):
            self.view.set_status(tr("Hose adapter failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the hose adapter…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ gridfinity
    def gridfinity_dialog(self):
        """Draw > Parts > Gridfinity…: a bin or a baseplate of the open storage grid."""
        dlg = GridfinityDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        for key in ("kind", "cols", "rows", "units", "divisions", "lip", "magnets", "scoop",
                    "label"):
            setattr(self.settings, "grid_" + key, options[key])
        self.settings.save()
        made = self.make_gridfinity_model(options)
        self.setFocus()
        return made

    def make_gridfinity_model(self, options):
        """The bin or the baseplate as a printed part: one undo step, next to the model."""
        self.set_mode("3d")
        doc = self.doc
        kind = options["kind"]
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the Gridfinity part…"))
            if kind == "baseplate":
                m = gridfinity.gridfinity_baseplate(options["cols"], options["rows"])
            else:
                m = gridfinity.gridfinity_bin(options["cols"], options["rows"], options["units"],
                                              divisions=options["divisions"], lip=options["lip"],
                                              magnets=options["magnets"],
                                              scoop=options["scoop"], label=options["label"])
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the part away again
            name = tr("Gridfinity baseplate") if kind == "baseplate" else tr("Gridfinity bin")
            g = ops.group_from_manifold(doc, m, self.settings.current_color, name)
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("{name} added.").format(name=name))

        def failed(exc):
            self.view.set_status(tr("Gridfinity failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the Gridfinity part…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ cable chain
    def cable_chain_dialog(self):
        """Draw > Parts > Cable Chain…: a drag chain printed in one piece."""
        dlg = CableChainDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.chain_width = float(options["width"])
        self.settings.chain_height = float(options["height"])
        self.settings.chain_wall = float(options["wall"])
        self.settings.chain_links = int(options["links"])
        self.settings.chain_bend = float(options.get("bend", 60.0))
        self.settings.save()
        made = self.make_cable_chain_model(options)
        self.setFocus()
        return made

    def make_cable_chain_model(self, options):
        """The chain as one printed part of loose links: one undo step, next to the model."""
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("width", "height", "wall", "links")}
        size["bend"] = float(options.get("bend", 60.0))
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the cable chain…"))
            m = cable_chain.cable_chain(**size)
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
            mid = 0.5 * (lo + hi)
            x = 0.5 * bx if bb is None else float(bb[1][0]) + 10.0 + 0.5 * (hi[0] - lo[0])
            return m.translate((x - mid[0], 0.5 * by - mid[1], -lo[2]))

        def done(m):
            doc.checkpoint()                # one undo takes the chain away again
            g = ops.group_from_manifold(doc, m, self.settings.current_color, tr("Cable chain"))
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("Cable chain of {n} links added.").format(n=size["links"]))

        def failed(exc):
            self.view.set_status(tr("Cable chain failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the cable chain…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ bin wall rack
    def bin_rack_dialog(self):
        """Draw > Parts > Bin Wall Rack…: a plate on the wall with rails the bins hang on."""
        dlg = BinRackDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.rack_cols = int(options["cols"])
        self.settings.rack_rows = int(options["rows"])
        self.settings.rack_cell_w = float(options["cell_w"])
        self.settings.rack_cell_h = float(options["cell_h"])
        self.settings.rack_plate = float(options["plate"])
        self.settings.rack_screw = str(options["screw"])
        self.settings.rack_panel_cols = int(options.get("panel_cols", 0))
        self.settings.save()
        made = self.make_bin_rack_model(options)
        self.setFocus()
        return made

    def make_bin_rack_model(self, options):
        """The rack as printed parts, lying on their backs: one undo step, next to the model.

        Like the bin it is built in the background. A rack in panels gives one part per panel,
        10 mm apart from left to right; the row of them lies in the middle of the free bed or
        10 mm to the right of what is already in the model.
        """
        self.set_mode("3d")
        doc = self.doc
        size = {k: options[k] for k in ("cols", "rows", "cell_w", "cell_h", "plate", "screw")}
        size["panel_cols"] = int(options.get("panel_cols", 0))
        bb = doc.model_bbox()
        bx, by, _bz = self.settings.bed_size

        def work(report):
            report(0.1, tr("Building the bin wall rack…"))
            panels = bin_rack.bin_rack_panels(**size)
            for m in panels:
                if m.status() != mf.Error.NoError or m.is_empty():
                    raise ValueError(tr("the result is not a valid volume"))
            boxes = [np.asarray(m.bounding_box(), dtype=float) for m in panels]
            wide = sum(b[3] - b[0] for b in boxes) + 10.0 * (len(boxes) - 1)
            x = (0.5 * bx - 0.5 * wide) if bb is None else float(bb[1][0]) + 10.0
            placed = []
            for m, b in zip(panels, boxes):
                placed.append(m.translate((x - b[0], 0.5 * by - 0.5 * (b[1] + b[4]), -b[2])))
                x += b[3] - b[0] + 10.0
            return placed

        def done(panels):
            doc.checkpoint()                # one undo takes the rack away again
            groups = []
            for i, m in enumerate(panels):
                name = (tr("Bin wall rack") if len(panels) == 1
                        else tr("Bin wall rack {n}").format(n=i + 1))
                groups.append(ops.group_from_manifold(doc, m, self.settings.current_color, name))
            doc.set_selection([("g", g.id, g.id) for g in groups])
            doc.notify()
            if len(panels) == 1:
                self.view.set_status(tr("Bin wall rack added. Bins hang on it with Wall hook on."))
            else:
                self.view.set_status(tr("Bin wall rack added in {n} panels that slide together. "
                                        "Bins hang on it with Wall hook on.").format(n=len(panels)))

        def failed(exc):
            self.view.set_status(tr("Bin wall rack failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the bin wall rack…"), on_done=done,
                       on_error=failed)

    # ------------------------------------------------------------ dogbone corners
    def ask_sheet_text(self):
        """Dialog of the 2D text tool: text, font, height and bold."""
        from .fonts import default_family, PREFERRED
        tool = self.sheet_view.tools["text"]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Text on the Worksheet"))
        form = QFormLayout(dlg)
        ed = QLineEdit(tool.text or getattr(self, "_last_sheet_text", "Volura"))
        family = QComboBox()
        family.addItems(PREFERRED)
        if tool.family:
            family.setCurrentText(tool.family)
        elif default_family() in PREFERRED:
            family.setCurrentText(default_family())
        bold = QCheckBox(tr("Bold"))
        bold.setChecked(bool(tool.bold))
        h = QDoubleSpinBox()
        h.setRange(0.5, 2000)
        h.setDecimals(2)
        h.setSuffix(" mm")
        h.setValue(float(tool.height) if tool.height else 10.0)
        form.addRow(tr("Text"), ed)
        form.addRow(tr("Font"), family)
        form.addRow("", bold)
        form.addRow(tr("Cap height"), h)
        form.addRow(QLabel(tr("The letters are placed as closed outlines, with the holes\n"
                              "of o, a and e in them. Type a height in the Measurements box to "
                              "change it.")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not ed.text().strip():
            return False
        self._last_sheet_text = ed.text()
        if not tool.set_text(ed.text(), family.currentText(), h.value(), bold.isChecked()):
            self.hint.setText(tr("This text gave no outlines; choose another font."))
            return False
        self.sheet_view.set_tool("text")
        self.sheet_view.setFocus()
        return True
    # ------------------------------------------------------------ solids
    def _in_selection_order(self, groups):
        """Groups in the order in which they were selected (SketchUp solid tools)."""
        order = {it: k for k, it in enumerate(self.doc.selection)}
        return sorted(groups, key=lambda g: order.get(("g", g.id, g.id), 0))

    def solid_op(self, op):
        groups = self._sel_groups(2)
        if not groups:
            return
        groups = self.unlocked(groups)
        if len(groups) < 2:
            return
        groups = self._in_selection_order(groups)
        self.doc.checkpoint()
        try:
            res = ops.boolean(self.doc, groups, op)
        except ValueError as e:
            self._failed(e)
            return
        self.doc.set_selection([("g", res.id, res.id)] if res else [])
        self.doc.notify()

    def outer_shell(self):
        """Tools > Outer Shell: only the outside of the selected groups."""
        groups = self.unlocked(self._sel_groups() or [])
        if not groups:
            return
        self.doc.checkpoint()
        try:
            res = ops.outer_shell(self.doc, groups)
        except ValueError as e:
            self._failed(e)
            return
        self.doc.set_selection([("g", res.id, res.id)] if res else [])
        self.doc.notify()
    # ------------------------------------------------------------ simplify model

    def simplify_model(self):
        """Tools > Simplify Model: fewer triangles on the selected groups."""
        groups = self.doc.selected_groups()
        if not groups:
            self.hint.setText(tr("Select the group(s) to simplify first."))
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        self.simplify_groups(groups)

    def simplify_groups(self, groups):
        """Show the Simplify Model dialog for `groups` and apply the chosen quality."""
        dlg = SimplifyDialog(self, groups)
        if dlg.exec() != QDialog.Accepted:
            return None
        return self._run_simplify(groups, dlg.tolerance, dlg.manifolds)

    def _run_simplify(self, groups, tolerance, manifolds=None):
        """Simplify every group in the background, then swap the new geometry in with one
        checkpoint, so a single undo restores them all. `manifolds` ({group id: Manifold})
        are those the dialog already made for its estimate."""
        groups = [g for g in groups if not g.locked]
        manifolds = dict(manifolds or {})

        def work(report):
            # building the new faces and their render cache is the slow part: seconds
            # for a big model, so it happens here and the window keeps responding
            out = []
            for i, g in enumerate(groups):
                report(i / len(groups), tr("Simplifying {name}…").format(name=g.name))
                before, after, ents = ops.simplified(g, tolerance, manifolds.get(g.id))
                out.append((g, before, after, ents, GeoCache.build(ents),
                            ops.part_summary(g.name, g.color, ents, pool=True)))
            return out

        def done(out):
            self.doc.checkpoint()
            for g, _b, _a, ents, cache, summary in out:
                g.ents = ents
                self.view.engine.cache.put(cache)
                self.info.remember_part(g, summary)
            self.doc.notify()
            before = sum(o[1] for o in out)
            after = sum(o[2] for o in out)
            self.hint.setText(tr("Simplified: {n} fewer triangles ({before} -> {after}).")
                              .format(n=before - after, before=before, after=after))

        if not groups:
            return None
        return background.run(self, work, text=tr("Simplifying…"),
                              on_done=done, on_error=self._failed)
    # ------------------------------------------------------------ mouse ears

    def mouse_ears_dialog(self):
        """Tools > Print Tools > Mouse Ears…: thin discs on the bed under the sharp bottom
        corners of the selected groups, so those corners do not lift (warp) while printing."""
        gids = self._target_groups()
        if not gids:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Mouse Ears"))
        form = QFormLayout(dlg)
        dia_sp = self._spin(5.0, 30.0, self.settings.mouse_ear_diameter, 1, " mm", 0.5)
        hgt_sp = self._spin(0.1, 2.0, self.settings.mouse_ear_height, 2, " mm", 0.1)
        form.addRow(tr("Diameter"), dia_sp)
        form.addRow(tr("Height"), hgt_sp)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        diameter, height = float(dia_sp.value()), float(hgt_sp.value())
        self.settings.mouse_ear_diameter = diameter
        self.settings.mouse_ear_height = height
        self.settings.save()
        self._run_mouse_ears(gids, diameter, height)
    # ------------------------------------------------------------ orient for printing

    def split_parts(self):
        groups = self._sel_groups()
        if not groups:
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        self.doc.checkpoint()
        out = []
        for g in groups:
            out += [("g", p.id, p.id) for p in ops.split_parts(self.doc, g)]
        self.doc.set_selection(out)
        self.doc.notify()

    def _target_items(self):
        return list(self.doc.selection) or ops.whole_items(self.doc)

    def _target_groups(self):
        """The groups the target items refer to: the selected groups, plus the group that a
        selected face or edge belongs to, so that clicking a face of a part means the part.
        Loose geometry has no group and is left out; with nothing selected this is every
        visible group (see `whole_items`), so hidden groups are never touched."""
        gids, seen = [], set()
        for k, ctx, i in self._target_items():
            gid = ctx if k in ("f", "e") else (i if k == "g" else None)
            if gid is None or gid == ROOT or gid in seen or self.doc.group(gid) is None:
                continue
            seen.add(gid)
            gids.append(gid)
        return gids

    def drop_to_bed(self):
        items = self._target_items()
        if items:
            self.doc.checkpoint()
            ops.drop_to_bed(self.doc, items)
            self.doc.notify()

    def center_on_bed(self):
        items = self._target_items()
        if items:
            self.doc.checkpoint()
            ops.center_on_bed(self.doc, items, self.settings.bed_size)
            self.doc.notify()

    def mirror(self, axis):
        items = list(self.doc.selection)
        if items:
            self.doc.checkpoint()
            ops.mirror(self.doc, items, axis)
            self.doc.notify()
    # ------------------------------------------------------------ screw thread
    THREAD_KINDS = [(tr("External thread on cylinder"), "external"),
                    (tr("Internal thread in hole"), "internal"),
                    (tr("New bolt"), "bolt"), (tr("New nut or sleeve"), "nut")]

    def _thread_target(self, kind):
        """(ents, group, center, axis, radius, length) from the selection, or None.

        Valid: a selected circular end face (external thread) or the opening of a hole
        (internal thread), or a group that is a cylinder/sleeve."""
        want = "outer" if kind == "external" else "hole"
        doc = self.doc
        for k, ctx, i in list(doc.selection):
            if k != "f":
                continue
            ents = doc.ents(ctx)
            if ents is None:
                continue
            tc = threads.face_thread_context(ents, i, want)
            if tc is not None:
                return (ents, doc.group(ctx)) + tc
        for g in doc.selected_groups():
            tc = threads.group_thread_context(g.ents, want)
            if tc is not None:
                return (g.ents, g) + tc
        return None

    def thread_dialog(self, preset=None):
        kinds = self.THREAD_KINDS
        idx = next((i for i, (_, k) in enumerate(kinds) if k == preset), 0)
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Screw Thread"))
        form = QFormLayout(dlg)
        kind_cb = QComboBox()
        for label, _ in kinds:
            kind_cb.addItem(label)
        kind_cb.setCurrentIndex(idx)
        dia = QDoubleSpinBox()
        dia.setRange(1.0, 500.0)
        dia.setDecimals(2)
        dia.setSuffix(" mm")
        dia.setValue(8.0)
        pitch_cb = QComboBox()
        pitch_cb.setEditable(True)
        for p in sorted(threads.ISO_PITCH.values()):
            pitch_cb.addItem(f"{p:g}")
        len_sp = QDoubleSpinBox()
        len_sp.setRange(0.5, 2000.0)
        len_sp.setDecimals(2)
        len_sp.setSuffix(" mm")
        len_sp.setValue(20.0)
        clr_sp = QDoubleSpinBox()
        clr_sp.setRange(0.0, 2.0)
        clr_sp.setDecimals(2)
        clr_sp.setSingleStep(0.05)
        clr_sp.setSuffix(" mm")
        clr_sp.setValue(threads.DEFAULT_CLEARANCE)
        starts_sp = QSpinBox()
        starts_sp.setRange(1, 4)
        left_cb = QCheckBox(tr("Left-hand thread"))
        outer_sp = QDoubleSpinBox()
        outer_sp.setRange(2.0, 500.0)
        outer_sp.setDecimals(1)
        outer_sp.setSuffix(" mm")
        outer_sp.setValue(13.6)
        hex_cb = QCheckBox(tr("Hexagon (uncheck for a round sleeve)"))
        hex_cb.setChecked(True)
        mate_cb = QCheckBox(tr("Also make a matching nut/bolt"))
        form.addRow(tr("Kind"), kind_cb)
        form.addRow(tr("Diameter"), dia)
        form.addRow(tr("Pitch (ISO coarse, free to enter)"), pitch_cb)
        form.addRow(tr("Length / height"), len_sp)
        form.addRow(tr("Clearance"), clr_sp)
        form.addRow(tr("Number of starts"), starts_sp)
        form.addRow("", left_cb)
        form.addRow(tr("Outer size of nut/sleeve"), outer_sp)
        form.addRow("", hex_cb)
        form.addRow("", mate_cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        def defaults(*_):
            kind = kinds[kind_cb.currentIndex()][1]
            tc = self._thread_target(kind) if kind in ("external", "internal") else None
            if tc is not None:
                dia.setValue(max(dia.minimum(), min(dia.maximum(), 2 * tc[4])))
                len_sp.setValue(max(len_sp.minimum(), min(len_sp.maximum(), tc[5])))
            pitch_cb.setEditText(f"{threads.iso_pitch(dia.value()):g}")
            outer_sp.setValue(round(min(outer_sp.maximum(), dia.value() * 1.7), 1))

        kind_cb.currentIndexChanged.connect(defaults)
        dia.valueChanged.connect(lambda v: outer_sp.setValue(
            round(min(outer_sp.maximum(), v * 1.7), 1)))
        defaults()
        if dlg.exec() != QDialog.Accepted:
            return
        kind = kinds[kind_cb.currentIndex()][1]
        try:
            pitch = float(pitch_cb.currentText().replace(",", "."))
        except ValueError:
            pitch = 0.0
        if pitch <= 0:
            pitch = threads.iso_pitch(dia.value())
        try:
            self._apply_thread(kind, dia.value(), pitch, len_sp.value(), clr_sp.value(),
                               starts_sp.value(), left_cb.isChecked(), outer_sp.value(),
                               hex_cb.isChecked(), mate_cb.isChecked())
        except ValueError as e:
            self.view.set_status(tr("Screw thread failed: {error}").format(error=e))

    def _mate_group(self, man, label, near_bb, dx):
        """Put a matching part next to the edited part and make a group of it."""
        lo, hi = np.asarray(near_bb[:3]), np.asarray(near_bb[3:])
        man = man.translate((hi[0] + dx, (lo[1] + hi[1]) / 2.0, lo[2]))
        return ops.group_from_manifold(self.doc, man, self.settings.current_color, label)

    def _apply_thread(self, kind, d, pitch, length, clearance, starts, left, outer_d,
                      hexagon, mate):
        doc = self.doc
        if kind in ("external", "internal"):
            t = self._thread_target(kind)
            if t is None:
                self.view.set_status(
                    tr("First select a group (or its circular face) with a cylinder.")
                    if kind == "external" else
                    tr("First select a group (or the face with the opening) with a hole."))
                return
            ents, g, center, axis, radius, depth = t
            if g is not None and g.locked:
                self.warn_locked()
                return
            m = ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                self.view.set_status(tr("This part is not a closed volume."))
                return
            near = g.name if g else tr("Loose geometry")
            if kind == "external":
                res = threads.external_on(m, center, axis, radius, length, pitch=pitch,
                                          starts=starts, left=left)
                message = (tr("External thread M{d:g}x{p:g} put on {near} including a matching "
                              "part.") if mate else
                           tr("External thread M{d:g}x{p:g} put on {near}."))
            else:
                res = threads.internal_in(m, center, axis, radius, length, clearance,
                                          pitch=pitch, starts=starts, left=left)
                message = (tr("Internal thread M{d:g}x{p:g} put on {near} including a matching "
                              "part.") if mate else
                           tr("Internal thread M{d:g}x{p:g} put on {near}."))
            if res.status() != mf.Error.NoError or res.is_empty():
                self.view.set_status(tr("Screw thread failed: the result is not a valid volume."))
                return
            doc.checkpoint()
            new_ents = Entities.from_manifold(res)
            if g is not None:
                g.ents = new_ents
            else:
                doc.root = new_ents
            sel = [] if g is None else [("g", g.id, g.id)]
            if mate:
                bb = res.bounding_box()
                if kind == "external":
                    nm = threads.nut(d, pitch, max(0.8 * d, 4 * pitch), outer_d, clearance,
                                     hex=hexagon, starts=starts, left=left)
                    mg = self._mate_group(nm, tr("Nut M{d:g}x{p:g}").format(d=d, p=pitch), bb,
                                          outer_d / 2 + 5)
                else:
                    nm = threads.thread_solid(d, pitch, max(length, 2 * d), starts=starts,
                                              left=left)
                    mg = self._mate_group(nm, tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch), bb,
                                          d + 5)
                if mg is not None:
                    sel.append(("g", mg.id, mg.id))
            doc.set_selection(sel)
            doc.notify()
            self.view.set_status(message.format(d=d, p=pitch, near=near))
            return

        if kind == "bolt":
            man = threads.thread_solid(d, pitch, length, starts=starts, left=left)
            label = tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch)
        else:
            man = threads.nut(d, pitch, length, outer_d, clearance, hex=hexagon,
                              starts=starts, left=left)
            label = (tr("Nut M{d:g}x{p:g}") if hexagon else tr("Sleeve M{d:g}x{p:g}")).format(
                d=d, p=pitch)
        if man.status() != mf.Error.NoError or man.is_empty():
            self.view.set_status(tr("Screw thread failed: invalid dimensions."))
            return
        doc.checkpoint()
        bx, by, _ = self.settings.bed_size
        lo, hi = np.asarray(man.bounding_box()[:3]), np.asarray(man.bounding_box()[3:])
        man = man.translate((bx / 2 - (lo[0] + hi[0]) / 2, by / 2 - (lo[1] + hi[1]) / 2, -lo[2]))
        g = ops.group_from_manifold(doc, man, self.settings.current_color, label)
        sel = [("g", g.id, g.id)]
        if mate:
            if kind == "bolt":
                nm = threads.nut(d, pitch, max(0.8 * d, 4 * pitch), outer_d, clearance,
                                 hex=hexagon, starts=starts, left=left)
                mate_label = (tr("Nut M{d:g}x{p:g}") if hexagon
                              else tr("Sleeve M{d:g}x{p:g}")).format(d=d, p=pitch)
                mg = self._mate_group(nm, mate_label, man.bounding_box(), outer_d / 2 + 5)
            else:
                nm = threads.thread_solid(d, pitch, max(2 * d, length), starts=starts, left=left)
                mg = self._mate_group(nm, tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch),
                                      man.bounding_box(), d + 5)
            if mg is not None:
                sel.append(("g", mg.id, mg.id))
        doc.active = ROOT
        doc.set_selection(sel)
        doc.notify()
        self.view.set_status((tr("{label} placed with a matching part.") if mate
                              else tr("{label} placed.")).format(label=label))
    # ------------------------------------------------------------ screw hole
    # (label, threads.SCREW_HOLE_KINDS identifier)
    SCREW_HOLE_KINDS = [(tr("Through"), "through"),
                        (tr("Counterbore"), "counterbore"),
                        (tr("Countersunk"), "countersink"),
                        (tr("Nut pocket"), "nutpocket")]

    def _screw_hole_target(self):
        """(ents, group, center, axis, length) of the selected circular face, or None.

        The size of the circle does not matter, only its centre and the direction of the
        face: the user can draw any circle where the screw has to go."""
        doc = self.doc
        for k, ctx, i in list(doc.selection):
            if k != "f":
                continue
            ents, g = doc.ents(ctx), doc.group(ctx)
            if ents is None or g is None:
                continue
            tc = threads.face_thread_context(ents, i, "outer")
            if tc is not None:
                return (ents, g) + tc[:2] + tc[3:]
        return None

    def screw_hole_dialog(self):
        """Tools > Print Tools > Screw Hole…: a printable hole for a metric screw on the
        circle that is selected on a part: straight, or with room for the head."""
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Screw Hole"))
        form = QFormLayout(dlg)
        size_cb = QComboBox()
        for size in sorted(threads.SCREW_HOLES, key=lambda s: float(s[1:])):
            size_cb.addItem(size)
        size_cb.setCurrentText(self.settings.screw_hole_size)
        kind_cb = QComboBox()
        for label, _ in self.SCREW_HOLE_KINDS:
            kind_cb.addItem(label)
        kind_cb.setCurrentIndex(next((i for i, (_, k) in enumerate(self.SCREW_HOLE_KINDS)
                                      if k == self.settings.screw_hole_kind), 0))
        clr_sp = self._spin(0.0, 1.0, self.settings.screw_hole_clearance, 2, " mm", 0.05)
        tear_cb = QCheckBox(tr("Teardrop for sideways holes"))
        tear_cb.setToolTip(tr("A 45° roof on a horizontal hole, so it prints without support"))
        tear_cb.setChecked(self.settings.screw_hole_teardrop)
        form.addRow(tr("Screw size"), size_cb)
        form.addRow(tr("Kind"), kind_cb)
        form.addRow("", tear_cb)
        form.addRow(tr("Clearance"), clr_sp)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        size = size_cb.currentText()
        label, kind = self.SCREW_HOLE_KINDS[kind_cb.currentIndex()]
        clearance = clr_sp.value()
        teardrop = bool(tear_cb.isChecked())
        self.settings.screw_hole_size = size
        self.settings.screw_hole_kind = kind
        self.settings.screw_hole_clearance = float(clearance)
        self.settings.screw_hole_teardrop = teardrop
        self.settings.save()
        self._apply_screw_hole(size, kind, label, clearance, teardrop=teardrop)

    def _apply_screw_hole(self, size, kind, label, clearance, teardrop=False):
        doc = self.doc
        t = self._screw_hole_target()
        if t is None:
            self.view.set_status(tr("First select the circle you drew where the screw goes."))
            return
        ents, g, center, axis, depth = t
        if g.locked:
            self.warn_locked()
            return

        def work(report):
            # cutting the hole in a big part is the slow part: seconds, so it happens here
            # and the window keeps responding
            report(0.1, tr("Cutting the hole…"))
            m = ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                raise ValueError(tr("This part is not a closed volume."))
            res = threads.screw_hole(m, center, axis, depth, size, kind, clearance=clearance,
                                     segments=int(self.settings.circle_segments),
                                     teardrop=teardrop)
            if res.status() != mf.Error.NoError or res.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            return res

        def done(res):
            doc.checkpoint()              # one undo brings the part back as it was
            g.ents = Entities.from_manifold(res)
            doc.set_selection([("g", g.id, g.id)])
            doc.notify()
            self.view.set_status(tr("{size} {kind} hole cut in {name}.").format(
                size=size, kind=label.lower(), name=g.name))

        def failed(exc):
            self.view.set_status(tr("Screw hole failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Cutting the hole…"), on_done=done, on_error=failed)
    # ------------------------------------------------------------ clearance test

    def clearance_test_dialog(self):
        """Tools > Print Tools > Clearance Test…: a plate with a row of holes that grow by a
        step and a loose pin, so the user learns which clearance the printer likes."""
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Clearance Test"))
        form = QFormLayout(dlg)
        pin_sp = self._spin(2.0, 20.0, self.settings.clearance_test_pin, 1, " mm", 0.5)
        first_sp = self._spin(0.0, 1.0, self.settings.clearance_test_first, 2, " mm", 0.05)
        step_sp = self._spin(0.05, 0.5, self.settings.clearance_test_step, 2, " mm", 0.05)
        count_sp = QSpinBox()
        count_sp.setRange(2, 10)
        count_sp.setValue(self.settings.clearance_test_count)
        form.addRow(tr("Pin diameter"), pin_sp)
        form.addRow(tr("First clearance"), first_sp)
        form.addRow(tr("Step"), step_sp)
        form.addRow(tr("Number of holes"), count_sp)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        pin_d, first = float(pin_sp.value()), float(first_sp.value())
        step, count = float(step_sp.value()), int(count_sp.value())
        self.settings.clearance_test_pin = pin_d
        self.settings.clearance_test_first = first
        self.settings.clearance_test_step = step
        self.settings.clearance_test_count = count
        self.settings.save()
        self._apply_clearance_test(pin_d, first, step, count)

    def _apply_clearance_test(self, pin_d, first, step, count):
        doc = self.doc
        # a clearance never grows past 2 mm, so the step is made smaller when the row would
        step = min(step, (2.0 - first) / max(1, count - 1))
        clearances = tuple(first + i * step for i in range(count))
        bb = doc.model_bbox()
        x0 = 0.0 if bb is None else float(bb[1][0]) + 10.0   # mm to the right of the model
        segments = int(self.settings.circle_segments)

        def work(report):
            # the letters of the labels are the slow part of a plate with many holes, so it
            # happens here and the window keeps responding
            report(0.1, tr("Building the clearance test…"))
            plate, pin = threads.clearance_test(pin_d=pin_d, clearances=clearances,
                                                segments=segments)
            for m in (plate, pin):
                if m.status() != mf.Error.NoError or m.is_empty():
                    raise ValueError(tr("the result is not a valid volume"))
            # the plate is drawn around its first hole: move the test to the free spot on the bed
            dx = x0 - float(np.asarray(plate.bounding_box())[0])
            return plate.translate((dx, 0.0, 0.0)), pin.translate((dx, 0.0, 0.0))

        def done(result):
            plate, pin = result
            doc.checkpoint()                # one undo takes the plate and the pin away
            color = self.settings.current_color
            new = [ops.group_from_manifold(doc, plate, color, tr("Clearance test")),
                   ops.group_from_manifold(doc, pin, color, tr("Clearance test pin"))]
            doc.set_selection([("g", g.id, g.id) for g in new])
            doc.notify()
            self.view.set_status(tr("Print it and use the clearance of the hole the pin "
                                    "fits best."))

        def failed(exc):
            self.view.set_status(tr("Clearance test failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the clearance test…"),
                       on_done=done, on_error=failed)
    # -------------------------------------------------------------- overhang test

    def overhang_test_dialog(self):
        """Tools > Print Tools > Overhang Test…: a plate of fins that lean further over, so the
        user learns up to which angle the printer prints an overhang without support."""
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Overhang Test"))
        form = QFormLayout(dlg)
        first_sp = self._spin(0.0, 85.0, self.settings.overhang_test_first, 0, " °", 5.0)
        step_sp = self._spin(1.0, 45.0, self.settings.overhang_test_step, 0, " °", 5.0)
        count_sp = QSpinBox()
        count_sp.setRange(1, 10)
        count_sp.setValue(self.settings.overhang_test_count)
        form.addRow(tr("First angle"), first_sp)
        form.addRow(tr("Step"), step_sp)
        form.addRow(tr("Number of fins"), count_sp)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        first, step = float(first_sp.value()), float(step_sp.value())
        count = int(count_sp.value())
        self.settings.overhang_test_first = first
        self.settings.overhang_test_step = step
        self.settings.overhang_test_count = count
        self.settings.save()
        self._apply_overhang_test(first, step, count)

    def _apply_overhang_test(self, first, step, count):
        doc = self.doc
        # an overhang never leans past 85 degrees, so the step is made smaller when the row would
        step = min(step, (85.0 - first) / max(1, count - 1))
        angles = tuple(first + i * step for i in range(count))
        bb = doc.model_bbox()
        x0 = 0.0 if bb is None else float(bb[1][0]) + 10.0   # mm to the right of the model

        def work(report):
            # the letters of the labels are the slow part of a plate with many fins, so it
            # happens here and the window keeps responding
            report(0.1, tr("Building the overhang test…"))
            plate = threads.overhang_test(angles=angles)
            if plate.status() != mf.Error.NoError or plate.is_empty():
                raise ValueError(tr("the result is not a valid volume"))
            # the plate is drawn around its first fin: move the test to the free spot on the bed
            dx = x0 - float(np.asarray(plate.bounding_box())[0])
            return plate.translate((dx, 0.0, 0.0))

        def done(result):
            doc.checkpoint()                # one undo takes the plate away again
            new = ops.group_from_manifold(doc, result, self.settings.current_color,
                                          tr("Overhang test"))
            doc.set_selection([("g", new.id, new.id)])
            doc.notify()
            self.view.set_status(tr("Print it and read the largest angle that still came out "
                                    "clean."))

        def failed(exc):
            self.view.set_status(tr("Overhang test failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Building the overhang test…"),
                       on_done=done, on_error=failed)
    # ------------------------------------------------------------ cut & join
    # (label, joints.KINDS identifier; the identifiers stay Dutch)
    CUT_JOINTS = [(tr("No joint (cut only)"), "geen"),
                  (tr("Pins (on the upper part)"), "pennen"),
                  (tr("Holes for loose pins"), "losse pennen"),
                  (tr("Dovetail"), "zwaluwstaart")]

    def _cut_model(self, groups):
        """One solid manifold of the selected groups, or None when they are not solid."""
        ms = []
        for g in groups:
            m = g.ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                return None
            ms.append(m)
        return ops.union_all(ms) if len(ms) > 1 else ms[0]

    def _spin(self, lo, hi, value, decimals=2, suffix=" mm", step=None):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSuffix(suffix)
        if step is not None:
            sp.setSingleStep(step)
        sp.setValue(value)
        return sp

    def cut_join_dialog(self, origin=None, normal=None):
        groups = self._sel_groups()
        if not groups:
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        model = self._cut_model(groups)
        if model is None:
            self.view.set_status(tr("Cut: the selection is not a closed volume."))
            return
        if origin is None:
            bb = np.asarray(model.bounding_box())
            origin = tuple((bb[:3] + bb[3:]) / 2)
            normal = (0.0, 0.0, 1.0)

        d = joints.DEFAULTS
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Cut and Join"))
        form = QFormLayout(dlg)
        kind_cb = QComboBox()
        for label, _ in self.CUT_JOINTS:
            kind_cb.addItem(label)
        dia = self._spin(1.0, 100.0, d["pin_diameter"])
        length = self._spin(1.0, 200.0, d["pin_length"])
        count_cb = QComboBox()
        count_cb.setEditable(True)
        for c in ("auto", "1", "2", "3", "4"):
            count_cb.addItem(c)
        wall = self._spin(0.5, 20.0, d["wall"])
        tw = self._spin(1.0, 200.0, d["dovetail_width"])
        td = self._spin(0.5, 100.0, d["dovetail_depth"])
        ta = self._spin(1.0, 60.0, d["dovetail_angle"], 1, "°")
        clr = self._spin(0.0, 2.0, d["clearance"], 2, " mm", 0.05)
        sep = QCheckBox(tr("Move the parts apart (10 mm apart)"))
        flat = QCheckBox(tr("Lay the parts flat on the bed"))
        form.addRow(tr("Joint"), kind_cb)
        form.addRow(tr("Pin diameter"), dia)
        form.addRow(tr("Pin length"), length)
        form.addRow(tr("Number of pins"), count_cb)
        form.addRow(tr("Wall thickness around hole"), wall)
        form.addRow(tr("Tail width"), tw)
        form.addRow(tr("Tail depth"), td)
        form.addRow(tr("Tail angle"), ta)
        form.addRow(tr("Clearance"), clr)
        form.addRow("", sep)
        form.addRow("", flat)

        def update_enabled():
            kind = self.CUT_JOINTS[kind_cb.currentIndex()][1]
            pin = kind in ("pennen", "losse pennen")
            dovetail = kind == "zwaluwstaart"
            for w in (dia, length, count_cb, wall):
                w.setEnabled(pin)
            for w in (tw, td, ta):
                w.setEnabled(dovetail)
            clr.setEnabled(pin or dovetail)
        kind_cb.currentIndexChanged.connect(update_enabled)
        update_enabled()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        kind = self.CUT_JOINTS[kind_cb.currentIndex()][1]
        txt = count_cb.currentText().strip().lower()
        count = int(txt) if txt.isdigit() else "auto"
        params = {"pin_diameter": dia.value(), "pin_length": length.value(), "count": count,
                  "wall": wall.value(), "dovetail_width": tw.value(), "dovetail_depth": td.value(),
                  "dovetail_angle": ta.value(), "clearance": clr.value(),
                  "segments": self.settings.circle_segments}
        self._apply_cut_join(groups, model, origin, normal, kind, params,
                             sep.isChecked(), flat.isChecked())

    def _apply_cut_join(self, groups, model, origin, normal, kind, params, separate, flat):
        try:
            a, b, extras = joints.split_with_joint(model, origin, normal, kind, params)
        except Exception as e:
            self.view.set_status(tr("Cut failed: {error}").format(error=e))
            return
        if (a.is_empty() or b.is_empty() or a.status() != mf.Error.NoError
                or b.status() != mf.Error.NoError):
            self.view.set_status(tr("The cut did not give two valid parts."))
            return
        doc = self.doc
        color, name = groups[0].color, groups[0].name
        doc.checkpoint()
        first = groups[0]
        first.ents = Entities.from_manifold(a)
        for g in groups[1:]:
            doc.remove_group(g.id)
        part_b = doc.add_group(Entities.from_manifold(b), color,
                               tr("{name} (part B)").format(name=name))
        sel = [("g", first.id, first.id), ("g", part_b.id, part_b.id)]
        for i, pin in enumerate(extras):
            pg = ops.group_from_manifold(doc, pin, self.settings.current_color,
                                         tr("Pin {n}").format(n=i + 1))
            sel.append(("g", pg.id, pg.id))
        a_item, b_item = [("g", first.id, first.id)], [("g", part_b.id, part_b.id)]
        if flat:
            self._lay_part_flat(first, normal, +1)
            self._lay_part_flat(part_b, normal, -1)
            self._separate_on_bed(first, part_b)
        elif separate:
            n = joints.normalize(np.asarray(normal, dtype=float))
            ops.transform_selection(doc, np.eye(3), n * 10.0, items=a_item)
            ops.transform_selection(doc, np.eye(3), -n * 10.0, items=b_item)
        doc.active = ROOT
        doc.set_selection(sel)
        doc.notify()
        label = dict((k, l) for l, k in self.CUT_JOINTS)[kind].lower()
        self.view.set_status((trn("Model cut with {joint} and {n} loose pin.",
                                  "Model cut with {joint} and {n} loose pins.", len(extras))
                              if extras else tr("Model cut with {joint}."))
                             .format(joint=label, n=len(extras)))

    def _lay_part_flat(self, g, normal, sign):
        """Rotate a part about its centre so the cut face lies down, then drop it to the bed."""
        bb = g.bbox()
        if bb is None:
            return
        c = 0.5 * (np.asarray(bb[0], dtype=float) + np.asarray(bb[1], dtype=float))
        R = joints.align_rotation(np.asarray(normal, dtype=float), np.array([0.0, 0.0, float(sign)]))
        item = [("g", g.id, g.id)]
        ops.transform_selection(self.doc, R, c - R @ c, items=item)
        ops.drop_to_bed(self.doc, item)

    def _separate_on_bed(self, ga, gb):
        """Slide part B in +X next to part A so the two do not overlap on the bed."""
        ba = ga.bbox()
        if ba is None:
            return
        gap = float(ba[1][0] - ba[0][0]) + 10.0
        ops.transform_selection(self.doc, np.eye(3), np.array([gap, 0.0, 0.0]),
                                items=[("g", gb.id, gb.id)])

    def ask_text(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("3D Text"))
        form = QFormLayout(dlg)
        ed = QLineEdit(getattr(self, "_last_text", "Volura"))
        font = QFontComboBox()
        font.setCurrentFont(self.font())
        bold = QCheckBox(tr("Bold"))
        bold.setChecked(True)
        h = QDoubleSpinBox()
        h.setRange(0.5, 1000)
        h.setValue(10)
        h.setSuffix(" mm")
        form.addRow(tr("Text"), ed)
        form.addRow(tr("Font"), font)
        form.addRow("", bold)
        form.addRow(tr("Cap height"), h)
        form.addRow(QLabel(tr("After placing: Push/Pull. Alt held = separate part\n"
                              "(e.g. inlaid text in another color).")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not ed.text().strip():
            if not self.view.tools["text"].loops:
                self.set_tool("select")
            return
        self._last_text = ed.text()
        self.view.tools["text"].set_text(ed.text(), font.currentFont().family(), h.value(), bold.isChecked())
        if self.view.tool is not self.view.tools["text"]:
            self.view.set_tool("text")
    # ------------------------------------------------------------ printing
    def print_check(self):
        part_list = ops.parts(self.doc)
        msgs = ops.check_parts(part_list, self.settings.bed_size, self.settings.overhang_limit)
        if not part_list:
            msgs = [("info", tr("The model has no volumes yet."))]
        loose = len(self.doc.root.loose_edges())
        if loose:
            msgs.append(("info", trn("{n} loose edge without a face is not printed.",
                                     "{n} loose edges without a face are not printed.",
                                     loose).format(n=loose)))
        # the levels are the identifiers of ops.check_parts (Dutch, not shown)
        icons = {"ok": "✔", "info": "ℹ", "waarschuwing": "⚠", "fout": "✖"}
        colors = {"ok": theme.color("ok"), "info": theme.color("info"),
                  "waarschuwing": theme.color("warn"), "fout": theme.color("error")}
        html = tr("<h3>Print Check</h3><p>Every part (group or loose shape) must be a closed "
                  "volume: no holes, no double or loose faces. Only closed volumes are "
                  "exported.</p>") + "<ul>"
        for level, text in msgs:
            html += f'<li style="color:{colors[level]}">{icons[level]} {text}</li>'
        html += "</ul>"
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(PANEL_PRINT_CHECK))
        dlg.resize(620, 420)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        row = QHBoxLayout()
        b1 = QPushButton(tr("Drop to Bed"))
        b1.clicked.connect(lambda: (self.drop_to_bed(), dlg.accept()))
        b2 = QPushButton(tr("Close"))
        b2.clicked.connect(dlg.accept)
        row.addWidget(b1)
        row.addStretch(1)
        row.addWidget(b2)
        lay.addLayout(row)
        dlg.exec()

    def show_help(self):
        """Help > Keyboard Shortcuts: the keys of the scheme that is on now."""
        rows = "".join(
            f"<tr><td><b>{self.shortcut_text(self.tool_actions[name])}</b></td>"
            f"<td>{tr(label)}</td></tr>"
            for name, label, _icon, _key in TOOL_INFO
            if self.sequence_texts(self.tool_actions[name]))
        eraser = self.shortcut_text(self.tool_actions["eraser"])
        pushpull = self.shortcut_text(self.tool_actions["pushpull"])
        html = tr(HELP_HTML).format(
            rows=rows,
            mouse=tr("Pan, with Shift: orbit") if self.view.middle_pans
            else tr("Orbit, with Shift: pan"),
            eraser=f" ({eraser})" if eraser else "",
            pushpull=f" ({pushpull})" if pushpull else "")
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Volura Help"))
        dlg.resize(700, 640)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        dlg.exec()
