"""The "3D Mill Job…" dialog: the cutters, the passes, the stock and the sides.

The model becomes G-code for a mill: `volura/toolpath3d.py` reads the solids into a height
map and lays roughing and finishing lines over it, with **Z = 0 on the top of the stock**,
because a mill is zeroed on the material. This dialog only collects what to ask for and
remembers it; the feed, the plunge and the spindle start at the settings of the 2.5D mill
(`volura/gcode_mill.py`), since the same little router is usually both.

The work runs in the background (`volura/background.py`): a model of 50 000 triangles on a
0.5 mm grid is seconds of computing, and the window keeps answering while it runs. When the
paths are there they are drawn over the model - **roughing gray, finishing blue** - so the
user sees what the machine will do before saving anything, and they go away with the
dialog.

A part that is milled on both sides wants the **4th axis**: the A axis of a 3+1 mill lies
along X and is turned by hand between the sides. That is only offered when the machine the
CNC Controller remembers drives four axes; with three the choice stays grey with the
reason under it.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QMessageBox, QPushButton,
                               QSpinBox, QVBoxLayout)

from . import background, geom, gcode_mill, grbl, ops, panels, toolpath3d, workspaces
from .i18n import tr, tr_noop
from .settings import flag, qsettings
from .sheet_dialogs import EXPORT_FILTERS, fmt_time

#: where the dialog remembers its numbers
SETTING = "mill3d/"

#: the rapid of a hobby mill, for what the job costs (millimetres per minute)
RAPID = 3000.0

#: the colours of the preview in the 3D view: roughing gray, finishing blue
ROUGH_COLOR = (0.55, 0.55, 0.55, 1.0)
FINISH_COLOR = (0.15, 0.45, 0.95, 1.0)

#: the cutters to choose from
TOOLS = (("flat", tr_noop("Flat end mill")), ("ball", tr_noop("Ball nose")))

#: the resolutions of the height map, in millimetres per cell
RESOLUTIONS = ((0.2, tr_noop("Fine (0.2 mm)")), (0.5, tr_noop("Normal (0.5 mm)")),
               (1.0, tr_noop("Coarse (1.0 mm)")))

#: the direction of the finishing lines
DIRECTIONS = ((0, tr_noop("In X")), (90, tr_noop("In Y")))

#: how many sides of the part the mill goes to
SIDES = ((0, tr_noop("Top only (3 axes)")),
         (1, tr_noop("Top and bottom, turned on A (4th axis)")))

#: the number boxes: field, default, lowest, highest and the step of the wheel
FIELDS = {"rough_d": (6.0, 0.1, 32.0, 0.1),
          "finish_d": (3.0, 0.1, 32.0, 0.1),
          "step_down": (2.0, 0.05, 50.0, 0.1),
          "rough_stepover": (40.0, 1.0, 100.0, 5.0),
          "leave": (0.3, 0.0, 5.0, 0.05),
          "finish_stepover": (0.3, 0.02, 20.0, 0.05),
          "feed": (800.0, 10.0, 20000.0, 50.0),
          "plunge": (300.0, 10.0, 5000.0, 50.0),
          "rpm": (10000, 0, 60000, 500),
          "safe_z": (5.0, 0.5, 100.0, 0.5),
          "margin": (2.0, 0.0, 50.0, 0.5)}


def defaults() -> dict:
    """The starting point of the dialog: the numbers of the 2.5D mill."""
    params = {key: value for key, value in gcode_mill.MILL_PARAMS}
    out = {key: value[0] for key, value in FIELDS.items()}
    out["feed"] = float(params["feed"])
    out["plunge"] = float(params["plunge"])
    out["rpm"] = int(params["rpm"])
    return out


def _number(field) -> float:
    """A remembered number of the dialog, or the default that belongs to the field."""
    default = defaults()[field]
    try:
        return type(default)(float(qsettings().value(SETTING + field, default)))
    except (TypeError, ValueError):
        return default


def _choice(field, default, table):
    """The data of a remembered combo box choice; the default when it names nothing."""
    want = str(qsettings().value(SETTING + field, "") or "")
    for data, _label in table:
        if str(data) == want:
            return data
    return default


def mill_parts(parts, o: dict, report=None) -> dict:
    """The toolpaths and the G-code of a part list, with the choices of the dialog.

    This runs off the GUI thread: the parts are already there and nothing of the window is
    touched. `report(fraction)` follows the sides on the progress dialog.

    {"gcode", "paths", "kinds", "angles", "stock", "info", "triangles"}
    """
    if report is not None:
        report(0.0)
    mesh = ops.union_all([p.manifold for p in parts])
    vertices, triangles = geom.mesh_arrays(mesh)
    rough = toolpath3d.Tool(o["rough_kind"], o["rough_d"]) if o["rough"] else None
    finish = toolpath3d.Tool(o["finish_kind"], o["finish_d"]) if o["finish"] else None
    if rough is None and finish is None:
        raise ValueError(tr("Choose at least one operation: roughing or finishing."))
    paths, angles, stock, kinds = toolpath3d.indexed(
        vertices, triangles, (0.0, 180.0) if o["fourth"] else (0.0,),
        rough_tool=rough, finish_tool=finish, res=o["res"], step_down=o["step_down"],
        stepover=o["rough_stepover"], stock_to_leave=o["leave"],
        finish_stepover=o["finish_stepover"], finish_angle=o["finish_angle"],
        margin=o["margin"], report=report)
    if not paths:
        raise ValueError(tr("The cutters are too big for the stock; nothing is left to mill."))
    text = toolpath3d.to_gcode(paths, finish or rough, o["feed"], o["plunge"], o["rpm"],
                               o["safe_z"], stock, angles=angles if o["fourth"] else None)
    return {"gcode": text, "paths": paths, "kinds": kinds, "angles": angles,
            "stock": stock, "triangles": len(triangles),
            "info": toolpath3d.estimate(paths, o["feed"], o["plunge"], RAPID)}


class Mill3DDialog(QDialog):
    """CNC > 3D Mill Job…: the cutters, the passes, the stock, the sides and the result."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.job = None
        self._closed = False
        self.setWindowTitle(tr("3D Mill Job"))
        lay = QVBoxLayout(self)
        lay.addWidget(panels.note(tr("The cutter follows the surface of the model. Z = 0 is "
                                     "the top of the stock, so zero the mill on the "
                                     "material.")))
        lay.addWidget(self._operations_box())
        lay.addWidget(self._passes_box())
        lay.addWidget(self._sides_box())
        self.info = panels.note(tr("Nothing is computed yet."))
        lay.addWidget(self.info)
        lay.addLayout(self._buttons_box())
        self._restore()
        self._sides_changed()
        self._toggled()

    # ------------------------------------------------------------------ the fields
    def _spin(self, field, decimals=2, suffix=" mm"):
        """A number box with the range and the default of its field."""
        default, lo, hi, step = FIELDS[field]
        box = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
        if isinstance(box, QDoubleSpinBox):
            box.setDecimals(decimals)
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setSuffix(suffix)
        box.setValue(default)
        box.valueChanged.connect(self._changed)
        return box

    def _combo(self, table, tip=""):
        box = QComboBox()
        for data, label in table:
            box.addItem(tr(label), data)
        if tip:
            box.setToolTip(tr(tip))
        box.currentIndexChanged.connect(self._changed)
        return box

    def _operations_box(self):
        """The two operations, each with a cutter of its own."""
        box = QGroupBox(tr("Operations"))
        lay = QFormLayout(box)
        self.b_rough = QCheckBox(tr("Roughing"))
        self.rough_kind = self._combo(TOOLS)
        self.rough_d = self._spin("rough_d")
        self.rough_d.setToolTip(tr("The end mill that takes the stock away"))
        self.b_finish = QCheckBox(tr("Finishing"))
        self.finish_kind = self._combo(TOOLS)
        self.finish_d = self._spin("finish_d")
        self.finish_d.setToolTip(tr("The ball nose that follows the shape of the model"))
        lay.addRow(self.b_rough)
        lay.addRow(tr("Cutter"), self.rough_kind)
        lay.addRow(tr("Diameter"), self.rough_d)
        lay.addRow(self.b_finish)
        lay.addRow(tr("Cutter"), self.finish_kind)
        lay.addRow(tr("Diameter"), self.finish_d)
        self.b_rough.toggled.connect(self._toggled)
        self.b_finish.toggled.connect(self._toggled)
        return box

    def _passes_box(self):
        """How deep and how far apart, the spindle, the safe height and the stock."""
        box = QGroupBox(tr("Passes and stock"))
        lay = QFormLayout(box)
        self.step_down = self._spin("step_down")
        self.step_down.setToolTip(tr("How deep roughing goes in one pass"))
        self.rough_stepover = self._spin("rough_stepover", suffix=" %")
        self.rough_stepover.setToolTip(tr("How far the next line of roughing stands from "
                                          "this one, as a part of the diameter"))
        self.leave = self._spin("leave")
        self.leave.setToolTip(tr("The skin that roughing leaves for the finishing pass"))
        self.finish_stepover = self._spin("finish_stepover")
        self.finish_stepover.setToolTip(tr("How far the finishing lines stand apart; a ball "
                                           "nose leaves a scallop of about a tenth of its "
                                           "diameter"))
        self.finish_dir = self._combo(DIRECTIONS)
        self.finish_dir.setToolTip(tr("The direction of the finishing lines"))
        self.feed = self._spin("feed", decimals=0, suffix=" mm/min")
        self.plunge = self._spin("plunge", decimals=0, suffix=" mm/min")
        self.rpm = self._spin("rpm", decimals=0, suffix=" rpm")
        self.safe_z = self._spin("safe_z")
        self.safe_z.setToolTip(tr("How high above the stock the cutter travels"))
        self.margin = self._spin("margin")
        self.margin.setToolTip(tr("How much wider than the model the stock is"))
        self.res = self._combo(RESOLUTIONS, tip="The size of a cell of the height map: "
                                                "0.5 mm is normal, 0.2 mm is fine and takes "
                                                "longer")
        lay.addRow(tr("Step down"), self.step_down)
        lay.addRow(tr("Roughing stepover"), self.rough_stepover)
        lay.addRow(tr("Stock to leave"), self.leave)
        lay.addRow(tr("Finishing stepover"), self.finish_stepover)
        lay.addRow(tr("Finishing lines"), self.finish_dir)
        lay.addRow(tr("Feed"), self.feed)
        lay.addRow(tr("Plunge feed"), self.plunge)
        lay.addRow(tr("Spindle"), self.rpm)
        lay.addRow(tr("Safe height"), self.safe_z)
        lay.addRow(tr("Stock margin"), self.margin)
        lay.addRow(tr("Resolution"), self.res)
        return box

    def _sides_box(self):
        """The sides of the part; the 4th axis only for a machine that has one."""
        box = QGroupBox(tr("Sides"))
        lay = QFormLayout(box)
        self.sides = self._combo(SIDES)
        self.fourth_hint = panels.note("")
        lay.addRow(tr("Milling"), self.sides)
        lay.addRow(self.fourth_hint)
        self.sides.currentIndexChanged.connect(self._sides_changed)
        return box

    def _buttons_box(self):
        """Compute, and only afterwards: save the file or hand the job to the machine."""
        row = QHBoxLayout()
        self.b_compute = QPushButton(tr("Compute"))
        self.b_compute.setToolTip(tr("Follow the model with the cutters"))
        self.b_save = QPushButton(tr("Save G-code…"))
        self.b_save.setToolTip(tr("Write the job to a .nc file"))
        self.b_send = QPushButton(tr("Send to the CNC Controller"))
        self.b_send.setToolTip(tr("Load the job in the CNC Controller next to the view"))
        self.b_close = QPushButton(tr("Close"))
        for b in (self.b_compute, self.b_save, self.b_send, self.b_close):
            row.addWidget(b)
        self.b_save.setEnabled(False)
        self.b_send.setEnabled(False)
        self.b_compute.clicked.connect(self.compute)
        self.b_save.clicked.connect(self.save)
        self.b_send.clicked.connect(self.send)
        self.b_close.clicked.connect(self.close)
        return row

    # ------------------------------------------------------------------ the machine
    def machine_axes(self) -> int:
        """How many axes the mill of the CNC Controller drives; 3 when nothing is known."""
        try:
            return int(self.win.machine_panel(workspaces.CNC).machine.axes)
        except Exception:
            return 3

    def four_axis_possible(self) -> bool:
        """Whether that mill has an A axis that can be turned to the second side."""
        return self.machine_axes() >= 4

    def _sides_changed(self, *_):
        """The 4th axis is only there for a machine that drives one."""
        can = self.four_axis_possible()
        self.sides.model().item(1).setEnabled(can)
        if not can:
            self.sides.setCurrentIndex(0)
            machine = self.win.machine_panel(workspaces.CNC).machine
            self.fourth_hint.setText(tr("{name} has {n} axes; the second side needs a 4th "
                                        "axis.").format(name=machine.label, n=machine.axes))
        else:
            self.fourth_hint.setText(tr("The mill stops, the part is turned to A 180° by "
                                        "hand, and the other side is milled from the top."))
        self._changed()

    # ------------------------------------------------------------------ the choices
    def _toggled(self, *_):
        """Only what the chosen operations need is open for the hand."""
        for on, widgets in ((self.b_rough.isChecked(),
                             (self.rough_kind, self.rough_d, self.step_down,
                              self.rough_stepover)),
                            (self.b_finish.isChecked(),
                             (self.finish_kind, self.finish_d, self.finish_stepover,
                              self.finish_dir))):
            for w in widgets:
                w.setEnabled(on)

    def _changed(self, *_):
        """A choice is not the job yet: the paths have to be computed again."""
        if self.job is not None:
            self.info.setText(tr("The choices changed; compute the toolpaths again."))

    def options(self) -> dict:
        """What the user asked for, as the numbers `mill_parts` works with."""
        return {"rough": self.b_rough.isChecked(),
                "rough_kind": self.rough_kind.currentData(),
                "rough_d": self.rough_d.value(),
                "step_down": self.step_down.value(),
                "rough_stepover": self.rough_stepover.value() / 100.0,
                "leave": self.leave.value(),
                "finish": self.b_finish.isChecked(),
                "finish_kind": self.finish_kind.currentData(),
                "finish_d": self.finish_d.value(),
                "finish_stepover": self.finish_stepover.value(),
                "finish_angle": int(self.finish_dir.currentData()),
                "feed": self.feed.value(),
                "plunge": self.plunge.value(),
                "rpm": int(self.rpm.value()),
                "safe_z": self.safe_z.value(),
                "margin": self.margin.value(),
                "res": float(self.res.currentData()),
                "fourth": int(self.sides.currentData()) == 1 and self.four_axis_possible()}

    def _restore(self):
        """The numbers of the last time this dialog was open."""
        self.b_rough.setChecked(flag(SETTING + "rough", True))
        self.b_finish.setChecked(flag(SETTING + "finish", True))
        for box, (field, default, table) in (
                (self.rough_kind, ("rough_kind", "flat", TOOLS)),
                (self.finish_kind, ("finish_kind", "ball", TOOLS)),
                (self.finish_dir, ("finish_angle", 0, DIRECTIONS)),
                (self.res, ("res", 0.5, RESOLUTIONS)),
                (self.sides, ("side", 0, SIDES))):
            box.setCurrentIndex(max(0, box.findData(_choice(field, default, table))))
        for field, box in (("rough_d", self.rough_d), ("finish_d", self.finish_d),
                           ("step_down", self.step_down),
                           ("rough_stepover", self.rough_stepover), ("leave", self.leave),
                           ("finish_stepover", self.finish_stepover), ("feed", self.feed),
                           ("plunge", self.plunge), ("rpm", self.rpm),
                           ("safe_z", self.safe_z), ("margin", self.margin)):
            box.setValue(_number(field))

    def remember(self):
        """Keep what was chosen, so the next job starts where this one stopped."""
        o = self.options()
        q = qsettings()
        q.setValue(SETTING + "rough", o["rough"])
        q.setValue(SETTING + "finish", o["finish"])
        q.setValue(SETTING + "side", "1" if o["fourth"] else "0")
        for field, value in (("rough_kind", str(o["rough_kind"])),
                             ("finish_kind", str(o["finish_kind"])),
                             ("finish_angle", str(o["finish_angle"])),
                             ("res", str(o["res"]))):
            q.setValue(SETTING + field, value)
        for field in FIELDS:
            q.setValue(SETTING + field, o[field] * (100.0 if field == "rough_stepover"
                                                    else 1.0))
        q.sync()

    # ------------------------------------------------------------------ the work
    def compute(self):
        """The toolpaths of the model, in the background; the buttons wait for them."""
        only = bool(self.win.doc.selected_groups())
        o = self.options()

        def prepare(report):
            report(0.0, tr("Reading the model…"))
            parts = [p for p in ops.parts(self.win.doc, only_selected=only)
                     if p.manifold is not None]
            if not parts:
                raise ValueError(tr("There is no solid in the model to mill."))
            report(0.1, tr("Computing the toolpaths…"))
            return mill_parts(parts, o,
                              report=lambda part: report(0.1 + 0.85 * part))

        self.b_compute.setEnabled(False)
        return background.run(self.win, prepare, text=tr("Computing the toolpaths…"),
                              on_done=self._computed, on_error=self._failed)

    def _computed(self, job):
        """The paths are there: say what they cost and show them over the model."""
        self.b_compute.setEnabled(True)
        self.job = job
        lines = job["gcode"].count("\n")
        info = job["info"]
        self.info.setText(tr("{n} lines, {mm} mm of cutting, about {t}").format(
            n=lines, mm=f"{info['cut_mm'] / 1000.0:.2f}", t=fmt_time(info["seconds"])))
        self.b_save.setEnabled(True)
        self.b_send.setEnabled(True)
        self.show_preview()
        self.win.hint.setText(tr("{n} lines of toolpath for the mill, drawn over the "
                                 "model.").format(n=lines))

    def _failed(self, exc):
        self.b_compute.setEnabled(True)
        QMessageBox.critical(self, tr("3D Mill Job"), str(exc))

    # ------------------------------------------------------------------ the preview
    def show_preview(self):
        """The paths over the model: roughing gray, finishing blue."""
        if self.job is None:
            return
        self.win.view.set_overlay_lines(
            [(path, ROUGH_COLOR if kind == "rough" else FINISH_COLOR, 1.2)
             for path, kind in zip(self.job["paths"], self.job["kinds"])])

    def clear_preview(self):
        """What this dialog drew over the model is gone again."""
        self.win.view.clear_overlay_lines()

    # ------------------------------------------------------------------ the result
    def save(self):
        """Save G-code…: a .nc file, named after the model like the export of the sheet."""
        if self.job is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save G-code"),
                                              self.win.export_name(".nc"),
                                              EXPORT_FILTERS["mill"])
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".nc"
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(self.job["gcode"])
        except OSError as e:
            QMessageBox.critical(self, tr("3D Mill Job"), str(e))
            return
        self.win.hint.setText(tr("G-code for the mill written to {path}").format(path=path))

    def send(self):
        """The job to the CNC Controller, the same way that one takes the worksheet."""
        if self.job is None:
            return
        panel = self.win.machine_panel(workspaces.CNC)
        self.win.set_workspace(workspaces.CNC)
        panel.set_job(grbl.job_lines(self.job["gcode"]), tr("the 3D model"))
        self.win.hint.setText(tr("The 3D milling job is in the CNC Controller."))

    # ------------------------------------------------------------------ closing
    def done(self, result):
        self._leave()
        super().done(result)

    def closeEvent(self, event):
        self._leave()
        super().closeEvent(event)

    def _leave(self):
        """Once, by any road: the preview goes away and the numbers stay."""
        if self._closed:
            return
        self._closed = True
        self.clear_preview()
        self.remember()
        if getattr(self.win, "_mill3d_dialog", None) is self:
            self.win._mill3d_dialog = None
