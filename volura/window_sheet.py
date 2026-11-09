"""The worksheet side of the main window: the 2D methods of Volura.

Step 1 of splitting `volura/app.py`: the methods that work on the drawing of the
worksheet moved here verbatim, into a plain mixin that `MainWindow` inherits. The sheet
and the nesting, the box with finger joints, the living hinge, the vent pattern, the fit
test and the spur gear sheet, the holding tabs and the round and dogbone corners, the
exports for laser and mill and as 2D vector, putting 3D on the worksheet and the live
links between the two sides.

The methods that ask something with a message box, a file dialog or an input dialog stay
in `volura/app.py`, together with the actions that call all of this; the window still
owns everything, this module only holds the code.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QDialog

from . import (background, boxgen, cam, dxf, errors, gcode, gears, linking, nesting, ops,
               sheet3d, sheettools, svg, vector_export)
from .brep import Entities
from .i18n import tr, trn
from .icons import icon
from .model import ROOT
from .panels import swatch
from .part_dialogs import SpurGearDialog
from .sheet_dialogs import (AutoArrangeDialog, BoxJointDialog, DogboneCornersDialog,
                            FitTestDialog, GCODE_S_MAX_DEFAULT, HoldingTabsDialog,
                            LivingHingeDialog, RoundCornersDialog, SheetExportDialog,
                            SheetSettingsDialog, VectorExportDialog, VentPatternDialog)
from .sheettools import joined_paths
from .tools import format_length
from .vector2d import (SHEET_SPACING, Drawing, MILL_OPS, OP_LABELS, VPath, _transform_meta,
                       default_layer_names, polygon_area)


class SheetMixin:
    """The worksheet methods of `MainWindow`; the window itself lives in the app."""

    # ------------------------------------------------------------ sheet and nesting
    def open_sheet_settings(self):
        """View > Sheet…: size, thickness, margin, gap and number of sheets."""
        self.set_mode("2d")
        dlg = SheetSettingsDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return
        values = dlg.values()
        sheet = self.doc.sheet

        def change():
            sheet.sheet_w = values["w"]
            sheet.sheet_h = values["h"]
            sheet.sheet_thickness = values["thickness"]
            sheet.sheet_margin = values["margin"]
            sheet.sheet_gap = values["gap"]
            sheet.sheet_count = values["count"]
        self.doc.checkpoint()
        change()
        self.doc.notify()
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        message = (tr("Sheet {w:g} × {h:g} mm, {count} sheet, margin {margin:g} mm, gap {gap:g} mm")
                   if values["count"] == 1 else
                   tr("Sheet {w:g} × {h:g} mm, {count} sheets, margin {margin:g} mm, gap {gap:g} mm"))
        self.hint.setText(message.format(w=values["w"], h=values["h"], count=values["count"],
                                         margin=values["margin"], gap=values["gap"]))
        self.setFocus()

    def auto_arrange(self, options=None):
        """View > Auto Arrange…: arrange parts economically over the sheets."""
        self.set_mode("2d")
        if not self.doc.sheet.paths:
            self.hint.setText(tr("There is nothing on the worksheet to arrange."))
            return
        if options is None:
            dlg = AutoArrangeDialog(self)
            if dlg.exec() != QDialog.Accepted:
                self.setFocus()
                return
            options = dlg.options()
        self.sheet_view.auto_arrange(options)
        self.setFocus()

    # ------------------------------------------------------------ box with finger joints
    def box_joint_dialog(self):
        """View > Box with Finger Joints…: the panels on the worksheet or as a 3D preview."""
        dlg = BoxJointDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        if options["mode"] == BoxJointDialog.MODEL:
            made = self.make_box_model(options)
        else:
            made = self.make_box_sheet(options)
        self.setFocus()
        return made

    def box_panels_from(self, options):
        """The panels of the box from an options dict; `ValueError` for impossible sizes."""
        o = dict(options or {})
        sheet = self.doc.sheet
        t = float(o.get("t") or sheet.sheet_thickness or boxgen.DEFAULT_THICKNESS)
        return boxgen.box_panels(float(o.get("w", 100.0)), float(o.get("d", 60.0)),
                                 float(o.get("h", 40.0)), t, o.get("finger_w"),
                                 bool(o.get("inner", True)), str(o.get("lid", "none")),
                                 float(o.get("kerf", 0.0) or 0.0))

    def _box_failure(self, error, note, message):
        """A failed box: a clear text in the status bar, never a crash.

        `note` is the English text for the log, `message` the translated template with
        {error} for the status bar."""
        if isinstance(error, ValueError):
            self.hint.setText(str(error))
        else:
            errors.log_exception(note=f"{note}: {error}")
            self.hint.setText(message.format(error=error))
        return None

    def make_box_sheet(self, options):
        """"On Worksheet": the cutting outlines on the worksheet, auto arranged on the sheet.

        One undo step: the panels go on in the same step and their place comes straight
        from the auto arrange. Without a sheet size they stay side by side.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        try:
            panels = self.box_panels_from(options)
            drawing = boxgen.panels_to_drawing(panels, layer=sheet.default_layer("cut") or None,
                                               label_layer=sheet.default_layer("engrave") or None,
                                               labels=bool(options.get("labels", True)))
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the box", tr("The box failed: {error}"))
        note = ""
        used = 0
        if sheet.sheet_w and sheet.sheet_h:
            try:
                placements = nesting.nest(nesting.parts_of(drawing), float(sheet.sheet_w),
                                          float(sheet.sheet_h), float(sheet.sheet_margin),
                                          float(sheet.sheet_gap))
            except ValueError as e:
                note = f" ({e})"
            except Exception as e:
                return self._box_failure(e, "arranging", tr("The arranging failed: {error}"))
            else:
                drawing = self._nest_box_drawing(drawing, placements, sheet)
                used = max((si for (_p, si, *_r) in placements), default=0) + 1
        else:
            note = tr(" (no sheet size: choose View > Sheet… to arrange)")
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        if used:
            sheet.sheet_count = max(int(sheet.sheet_count or 1), used)
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        W, D, H = boxgen.box_size(float(options.get("w", 100.0)), float(options.get("d", 60.0)),
                                  float(options.get("h", 40.0)),
                                  float(options.get("t") or sheet.sheet_thickness
                                          or boxgen.DEFAULT_THICKNESS),
                                  bool(options.get("inner", True)),
                                  str(options.get("lid", "none")))
        if not used:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm){note}")
        elif used == 1:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm), "
                         "arranged on {used} sheet{note}")
        else:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm), "
                         "arranged on {used} sheets{note}")
        self.hint.setText(message.format(n=len(panels), w=W, d=D, h=H, used=used, note=note))
        return added

    def _nest_box_drawing(self, drawing, placements, sheet):
        """Put the arranged places on the drawing: every panel with its engraving."""
        spacing = float(sheet.sheet_w) + SHEET_SPACING
        out = Drawing(layers={n: replace(l, params=dict(l.params))
                              for n, l in drawing.layers.items()},
                      **sheet._sheet_kwargs())
        for (part, si, angle, dx, dy) in placements:
            M = nesting.place_transform(part, angle, dx, dy, origin=(si * spacing, 0.0))
            lin, off = M[:2, :2], M[:2, 2]
            for p in part.paths:
                q = np.asarray(p.pts, dtype=float) @ lin.T + off
                out.paths.append(VPath(np.ascontiguousarray(q), p.closed, p.layer, p.filled,
                                       p.rule, p.src, _transform_meta(p.meta, lin, off)))
                out.ensure_layer(p.layer)
        return out

    def make_box_model(self, options):
        """"Preview in 3D": every panel as its own group, in its place in the box."""
        self.set_mode("3d")
        try:
            panels = self.box_panels_from(options)
            solids = boxgen.panels_to_manifolds(panels)
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the box", tr("The box failed: {error}"))
        t = float(options.get("t") or self.doc.sheet.sheet_thickness
                  or boxgen.DEFAULT_THICKNESS)
        W, D, H = boxgen.box_size(float(options.get("w", 100.0)), float(options.get("d", 60.0)),
                                  float(options.get("h", 40.0)), t,
                                  bool(options.get("inner", True)),
                                  str(options.get("lid", "none")))
        bx, by, _bz = self.settings.bed_size
        shift = (bx / 2.0 - W / 2.0, by / 2.0 - D / 2.0, 0.0)
        self.doc.checkpoint()
        groups = []
        for p, man in zip(panels, solids):
            g = ops.group_from_manifold(self.doc, man.translate(shift),
                                        self.settings.current_color,
                                        tr("Box - {name}").format(name=p.name))
            groups.append(g)
        self.doc.set_selection([("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.view.zoom_extents()
        self.hint.setText(tr("{n} panels of the box placed in the model ({w:g} × {d:g} × {h:g} mm)")
                          .format(n=len(groups), w=W, d=D, h=H))
        return groups

    # ------------------------------------------------------------ living hinge
    def _hinge_rectangle(self, paths=None, tol=0.01):
        """(x0, y0, x1, y1) when the selection is one rectangle lying straight, else None.

        A living hinge only fits in a rectangle of the drawing: four points on two
        vertical and two horizontal lines, within `tol` mm. A rotated rectangle is not
        one; the slits of a hinge run along the axes of the sheet.
        """
        sel = self.sheet_view.selected_paths() if paths is None else list(paths)
        if len(sel) != 1 or not sel[0].closed:
            return None
        pts = np.asarray(sel[0].pts, dtype=float)
        # Imported rectangles often end on their first point again; that is still four.
        if len(pts) == 5 and np.all(np.abs(pts[0] - pts[-1]) <= tol):
            pts = pts[:-1]
        if len(pts) != 4:
            return None
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        if x1 - x0 <= tol or y1 - y0 <= tol:
            return None
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for p in pts:
            if not any(abs(p[0] - cx) <= tol and abs(p[1] - cy) <= tol for cx, cy in corners):
                return None
        for cx, cy in corners:
            if not any(abs(p[0] - cx) <= tol and abs(p[1] - cy) <= tol for p in pts):
                return None
        return (x0, y0, x1, y1)

    def living_hinge_dialog(self):
        """View > Living Hinge…: the slits of a living hinge in the selected rectangle."""
        sel = self.sheet_view.selected_paths()
        rect = self._hinge_rectangle(sel)
        if rect is None:
            self.hint.setText(tr("Select one rectangle on the worksheet first."))
            self.setFocus()
            return None
        dlg = LivingHingeDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.hinge_slit = float(options["slit"])
        self.settings.hinge_gap = float(options["gap"])
        self.settings.hinge_pitch = float(options["pitch"])
        self.settings.hinge_margin = float(options["margin"])
        self.settings.hinge_vertical = bool(options["vertical"])
        self.settings.save()
        made = self.make_hinge_sheet(rect, options, src=int(sel[0].src))
        self.setFocus()
        return made

    def make_hinge_sheet(self, rect, options, src=None):
        """The slits on the cut layer of the worksheet: one undo step, all of them selected.

        Every slit gets the source number of the rectangle, so clicking the panel,
        moving it and arranging the sheet take the pattern along with the panel it is
        cut in.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        try:
            slits = boxgen.living_hinge(rect[0], rect[1], rect[2], rect[3],
                                        slit=options["slit"], gap=options["gap"],
                                        pitch=options["pitch"], margin=options["margin"],
                                        vertical=options["vertical"])
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the living hinge",
                                     tr("The living hinge failed: {error}"))
        if not slits:
            self.hint.setText(tr("There is no room left for slits in this rectangle."))
            return None
        layer = sheet.default_layer("cut") or default_layer_names("cut")[0]
        drawing = Drawing()
        for pts in slits:
            drawing.add_path(pts, closed=False, layer=layer, src=1)
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        if src:
            for p in added:
                p.src = int(src)
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(trn("{n} slit added: the panel bends there.",
                              "{n} slits added: the panel bends there.", len(added))
                          .format(n=len(added)))
        return added

    # ------------------------------------------------------------ vent pattern
    def vent_outline(self):
        """The contour the vent pattern fills: (points, source number) or None.

        One selected closed path, or loose lines that together close into exactly one
        contour (a drawing from a DXF, or a shape drawn line by line); the lines themselves
        stay as they are."""
        sel = self.sheet_view.selected_paths()
        if len(sel) == 1 and sel[0].closed:
            return np.asarray(sel[0].pts, dtype=float), int(sel[0].src or 0)
        if sel and not any(p.closed for p in sel):
            joined = joined_paths(sel)
            if len(joined) == 1 and joined[0].closed:
                return np.asarray(joined[0].pts, dtype=float), int(sel[0].src or 0)
        return None

    def _vent_selection_hint(self):
        self.hint.setText(tr("Select one closed contour on the worksheet first (a rectangle, "
                             "a circle or lines that close into one shape)."))

    def vent_pattern_dialog(self):
        """Laser and Mill > Vent Pattern…: the holes of a grille in the selected contour."""
        if self.vent_outline() is None:
            self._vent_selection_hint()
            self.setFocus()
            return None
        dlg = VentPatternDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.vent_shape = str(options["shape"])
        self.settings.vent_size = float(options["size"])
        self.settings.vent_web = float(options["web"])
        self.settings.vent_margin = float(options["margin"])
        self.settings.save()
        made = self.apply_vent_pattern(options)
        self.setFocus()
        return made

    def apply_vent_pattern(self, options):
        """The holes on the cut layer of the worksheet: one undo step, all of them selected.

        Every hole gets the source number of the contour, so clicking the panel, moving it
        and arranging the sheet take the grille along with the panel it is cut in.
        """
        found = self.vent_outline()
        if found is None:
            self._vent_selection_hint()
            return None
        outline_pts, outline_src = found
        self.set_mode("2d")
        sheet = self.doc.sheet
        try:
            holes = boxgen.vent_pattern(outline_pts, shape=options["shape"],
                                        size=options["size"], web=options["web"],
                                        margin=options["margin"])
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the vent pattern",
                                     tr("The vent pattern failed: {error}"))
        if not holes:
            self.hint.setText(tr("No hole fits inside this contour: make the holes, the web "
                                 "or the edge margin smaller."))
            return None
        layer = sheet.default_layer("cut") or default_layer_names("cut")[0]
        drawing = Drawing()
        for pts in holes:
            drawing.add_path(pts, closed=True, layer=layer, src=1)
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        if outline_src:
            for p in added:
                p.src = outline_src
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(trn("{n} hole added.", "{n} holes added.", len(added))
                          .format(n=len(added)))
        return added

    # ------------------------------------------------------------ fit test
    def fit_test_dialog(self):
        """View > Fit Test…: a comb of slots that shows which offset presses best."""
        dlg = FitTestDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.fit_test_first = float(options["first"])
        self.settings.fit_test_step = float(options["step"])
        self.settings.fit_test_count = int(options["count"])
        self.settings.save()
        made = self.make_fit_test_sheet(options)
        self.setFocus()
        return made

    def make_fit_test_sheet(self, options):
        """The comb on the cut layer: one undo step, next to what is already on the sheet.

        The test is a small part that lives beside the real parts, so it is not nested on
        the sheets like the panels of a box: it takes the free place to the right of the
        drawing, one gap further on.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        offs = tuple(float(options["first"]) + i * float(options["step"])
                     for i in range(int(options["count"])))
        try:
            drawing = boxgen.fit_test(float(options["t"]), offs,
                                      layer=sheet.default_layer("cut") or None,
                                      label_layer=sheet.default_layer("engrave") or None)
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the fit test", tr("The fit test failed: {error}"))
        b = sheet.bounds()
        if b is not None:
            dx = float(b[2]) + max(0.0, float(sheet.sheet_gap or 0.0))
            drawing = drawing.transformed(np.array([[1.0, 0.0, dx], [0.0, 1.0, float(b[1])],
                                                    [0.0, 0.0, 1.0]]))
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(tr("Cut it and use the slot that holds the sheet best: its offset "
                             "goes into the kerf or the finger width."))
        return added

    # ------------------------------------------------------------ spur gear
    def spur_gear_dialog(self, solid=False):
        """View > Spur Gear…: the wheel on the worksheet; Print Tools: a printable wheel."""
        dlg = SpurGearDialog(self, solid=solid)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.gear_teeth = int(options["teeth"])
        self.settings.gear_module = float(options["module"])
        self.settings.gear_angle = float(options["angle"])
        self.settings.gear_bore = float(options["bore"])
        self.settings.gear_backlash = float(options["backlash"])
        if solid:
            self.settings.gear_thickness = float(options["thickness"])
        self.settings.save()
        made = (self.make_spur_gear_model(options) if solid
                else self.make_spur_gear_sheet(options))
        self.setFocus()
        return made

    def make_spur_gear_sheet(self, options):
        """The wheel on the cut layer: one undo step, next to what is already on the sheet.

        A gear is a part like any other, so it takes the free place to the right of the
        drawing, one gap further on, and moves and nests with its own source number.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        teeth, module = int(options["teeth"]), float(options["module"])
        try:
            loops = gears.spur_gear(teeth, module=module, pressure_angle=options["angle"],
                                    bore=options["bore"], backlash=options["backlash"])
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the spur gear", tr("The spur gear failed: {error}"))
        layer = sheet.default_layer("cut") or default_layer_names("cut")[0]
        low = np.vstack(loops).min(axis=0)          # the wheel starts in the corner (0, 0)
        drawing = Drawing()
        for loop in loops:
            drawing.add_path(np.asarray(loop) - low, closed=True, layer=layer, filled=True,
                             src=1)
        b = sheet.bounds()
        if b is not None:
            dx = float(b[2]) + max(0.0, float(sheet.sheet_gap or 0.0))
            drawing = drawing.transformed(np.array([[1.0, 0.0, dx], [0.0, 1.0, float(b[1])],
                                                    [0.0, 0.0, 1.0]]))
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(tr("Spur gear with {n} teeth added (pitch diameter {d} mm).")
                          .format(n=teeth, d=f"{gears.pitch_diameter(teeth, module):.2f}"))
        return added

    # ------------------------------------------------------------ holding tabs
    def holding_tabs_dialog(self):
        """View > Holding Tabs…: small uncut bridges that keep the parts in the sheet."""
        if not self.sheet_view.selected_paths():
            self.hint.setText(tr("Select the closed contours to tab first."))
            self.setFocus()
            return None
        dlg = HoldingTabsDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.tab_count = int(options["count"])
        self.settings.tab_width = float(options["width"])
        self.settings.tab_outer_only = bool(options["outer_only"])
        self.settings.save()
        made = self.apply_holding_tabs(options)
        self.setFocus()
        return made

    def apply_holding_tabs(self, options):
        """The tabs on the selected closed contours of the cutting layers: one undo step.

        Only the contours that are cut clean through the sheet get tabs (layer Cut, Mill
        or Drill); open lines and engravings are skipped and counted. Each contour is
        replaced by its open pieces on the same layer with the same source number, so a
        part and its tabs still move and nest together.

        With `outer_only` (the default) a hole is left without tabs: its scrap has nothing
        to hold on to and falls out on its own. Whether a contour is a hole comes from its
        nesting depth over all the closed paths of its own layer on the sheet, not only the
        selection, so a hole counts as a hole even when its outer contour is not selected.
        """
        sel = self.sheet_view.selected_paths()
        self.set_mode("2d")
        sheet = self.doc.sheet
        count, width = int(options["count"]), float(options["width"])
        outer_only = bool(options.get("outer_only", True))
        depths = {}
        if outer_only:
            by_layer = {}
            for p in sheet.paths:
                if p.closed and len(p.pts) >= 3:
                    by_layer.setdefault(p.layer, []).append(p)
            for group in by_layer.values():
                for p, d in zip(group, cam.nesting_depth(group)):
                    depths[id(p)] = d
        replacements, tabbed, too_small, skipped, holes = {}, 0, 0, 0, 0
        for p in sel:
            lay = sheet.layers.get(p.layer)
            op = getattr(lay, "op", "cut") if lay is not None else "cut"
            if not (p.closed and len(p.pts) >= 3 and op in sheet3d.THROUGH_OPS):
                skipped += 1
                continue
            if outer_only and depths.get(id(p), 0) % 2:
                holes += 1
                continue
            pieces = sheettools.holding_tabs(p.pts, count, width)
            if len(pieces) == 1 and np.array_equal(pieces[0], np.asarray(p.pts, float)):
                too_small += 1
                continue
            replacements[id(p)] = pieces
            tabbed += 1
        notes = []
        if tabbed:
            notes.append(trn("{n} contour got {k} tabs each.",
                             "{n} contours got {k} tabs each.", tabbed).format(n=tabbed, k=count))
        if holes:
            notes.append(trn("{n} hole was left without tabs.",
                             "{n} holes were left without tabs.", holes).format(n=holes))
        if skipped:
            notes.append(trn("{n} path was skipped (open or not a cutting layer).",
                             "{n} paths were skipped (open or not a cutting layer).", skipped)
                         .format(n=skipped))
        if too_small:
            notes.append(trn("{n} contour is too small for these tabs.",
                             "{n} contours are too small for these tabs.", too_small)
                         .format(n=too_small))
        if not tabbed:
            self.hint.setText(" ".join(notes))
            return None
        self.doc.checkpoint()
        live = set(replacements)
        keep, fresh = [], []
        for p in sheet.paths:
            if id(p) not in live:
                keep.append(p)
                continue
            for piece in replacements[id(p)]:
                q = VPath(np.asarray(piece, dtype=float), False, p.layer, False, p.rule,
                          p.src, {})
                keep.append(q)
                fresh.append(q)
        sheet.paths = keep
        self.doc.notify()
        self.sheet_view.set_selection(fresh)
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(" ".join(notes))
        return fresh

    # ------------------------------------------------------------ round corners
    def round_corners_dialog(self):
        """View > Round Corners…: a radius in the sharp corners of the selected paths."""
        if not self.sheet_view.selected_paths():
            self.hint.setText(tr("Select the paths to round first."))
            self.setFocus()
            return None
        dlg = RoundCornersDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.corner_radius = float(options["radius"])
        self.settings.save()
        made = self.apply_round_corners(options)
        self.setFocus()
        return made

    #: from this many points on, the corners are rounded off the GUI thread
    BACKGROUND_ROUND_POINTS = 2000

    # ------------------------------------------------------------ dogbone corners
    def dogbone_corners_dialog(self):
        """View > Dogbone Corners…: a relief in every inside corner the mill cannot reach."""
        if not self.sheet_view.selected_paths():
            self.hint.setText(tr("Select the closed contours for the dogbones first."))
            self.setFocus()
            return None
        dlg = DogboneCornersDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        self.settings.dogbone_tool_d = float(options["tool_d"])
        self.settings.save()
        made = self.apply_dogbone_corners(options)
        self.setFocus()
        return made

    #: from this many points on, the reliefs are cut off the GUI thread
    BACKGROUND_DOGBONE_POINTS = 2000

    # ------------------------------------------------------------ 3D and worksheet together
    def _laser_layers(self):
        """The layers of the worksheet that have an operation (not the guides)."""
        return [(n, lay) for n, lay in self.doc.sheet.layers.items()
                if lay.op != "none" and not lay.locked]

    def _add_laser_menu(self, menu):
        """Right click in 3D: "Laser / Mill" with a line per layer of the worksheet."""
        sm = menu.addMenu(icon("to_laser"), tr("Laser / Mill"))
        face = getattr(self, "_context_face", None)
        for name, lay in self._laser_layers():
            a = sm.addAction(swatch(lay.color or "#000000", 14),
                             f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            a.triggered.connect(lambda checked=False, n=name, f=face:
                                self.send_selection_to_sheet(n, f))
        return sm

    def _face_plate_thickness(self, faces):
        """The sheet thickness for horizontal faces in a group: the height of that group."""
        groups = []
        for c, i in faces:
            g = next((g for g in self.doc.groups if g.id == c), None)
            if g is None or abs(g.ents.F[i].normal[2]) < 0.999:
                return None
            groups.append(g)
        return sheet3d.solid_thickness(groups)

    def _place_beside_sheet(self, drawing):
        """Move a new drawing to the right of the worksheet's content (or into the corner
        of the sheet when it is still empty), so nothing overlaps."""
        pts = [p.pts for p in drawing.paths if len(p.pts)]
        if not pts:
            return
        P = np.vstack(pts)
        lo = P.min(axis=0)
        sheet = self.doc.sheet
        have = [p.pts for p in sheet.paths if len(p.pts)]
        if have:
            Q = np.vstack(have)
            target = np.array([Q[:, 0].max() + float(sheet.sheet_gap) + 5.0, Q[:, 1].min()])
        else:
            m = float(sheet.sheet_margin) if sheet.sheet_w else 0.0
            target = np.array([m, m])
        shift = target - lo
        for p in drawing.paths:
            p.pts = np.ascontiguousarray(p.pts + shift)

    # ------------------------------------------------------------ the live links
    def _link_ents(self, link):
        """The geometry a link reads in 3D: the group, or the loose geometry for ROOT."""
        gid = int(link.group)
        if gid == ROOT:
            return self.doc.root
        g = self.doc.group(gid)
        return g.ents if g else None

    def _link_sheet_loops(self, src):
        """The closed paths of one source element on the worksheet, in link order."""
        loops = [p.pts for p in self.doc.sheet.paths
                 if int(p.src) == int(src) and p.closed and len(p.pts) >= 3]
        return linking.sorted_loops(loops)

    @staticmethod
    def _shape_matches(a, b) -> bool:
        """Whether the outline of a source element on the sheet is the shape of a part.

        The rigid comparison of `same_shape` first; when the two were cut up differently
        (a plate read by `plate_of`, the sheet by a projection) the areas of the loops say
        the same thing, so a part is still recognised as the shape that went to the sheet.
        """
        if linking.same_shape(a, b):
            return True
        aa = sorted(abs(polygon_area(l)) for l in a)
        bb = sorted(abs(polygon_area(l)) for l in b)
        return bool(aa) and len(aa) == len(bb) and all(
            abs(x - y) <= 1e-3 * max(1.0, y) for x, y in zip(aa, bb))

    def _link_sent_to_sheet(self, added, groups, faces):
        """Live links for the outlines that just went to the worksheet.

        Every source element that came from a selected part or face is matched back to it
        by shape, so the link knows which group and which `src` it joins. A plate becomes a
        "plate" link (its shape may also travel back); a face, and a group that is not a
        plate, becomes a "face" link that only carries the shape from 3D to the sheet.
        """
        by_src: dict = {}
        for p in added:
            by_src.setdefault(int(p.src), []).append(p.pts)
        srcs = {s: linking.sorted_loops(l) for s, l in by_src.items()}
        candidates = []
        for ctx, fid in faces:
            ents = self.doc.ents(ctx)
            if ents is None or fid not in ents.F:
                continue
            F = ents.F[fid]
            frame = linking.face_frame(F.normal, ents.V[F.loops[0][0]])
            shape = linking.sorted_loops([linking.to_plane([ents.V[v] for v in loop], frame)
                                          for loop in F.loops])
            if shape:
                candidates.append({"group": int(ctx), "kind": "face", "shape": shape,
                                   "thickness": None, "frame": linking.frame_to_list(frame)})
        if not candidates:
            for g in groups:
                m = g.ents.to_manifold()
                if m is None or m.is_empty():
                    continue
                pl = linking.plate_of(m)
                if pl is not None:
                    candidates.append({"group": int(g.id), "kind": "plate",
                                       "shape": linking.sorted_loops(pl["loops"]),
                                       "thickness": float(pl["thickness"]), "frame": None})
                    continue
                bb = g.bbox()
                if bb is None:
                    continue
                frame = linking.face_frame((0.0, 0.0, 1.0), 0.5 * (bb[0] + bb[1]))
                shape = linking.face_shape(m, frame)
                if shape:
                    candidates.append({"group": int(g.id), "kind": "face",
                                       "shape": linking.sorted_loops(shape), "thickness": None,
                                       "frame": linking.frame_to_list(frame)})
        used = set()
        for src, loops in srcs.items():
            for i, cand in enumerate(candidates):
                if i in used or not self._shape_matches(loops, cand["shape"]):
                    continue
                used.add(i)
                self.doc.links.append(linking.Link(src=src, group=cand["group"],
                                                   kind=cand["kind"], shape=cand["shape"],
                                                   thickness=cand["thickness"],
                                                   frame=cand["frame"]))
                break

    def _link_parts_to_sheet(self, groups, plates):
        """A "plate" link for every part that came from the worksheet into the model."""
        for g, (_name, man, src) in zip(groups, plates):
            pl = linking.plate_of(man)
            if pl is None:
                continue
            self.doc.links.append(linking.Link(src=int(src), group=int(g.id), kind="plate",
                                               shape=linking.sorted_loops(pl["loops"]),
                                               thickness=float(pl["thickness"]), frame=None))

    def sync_links(self, direction="both", announce_empty=False) -> int:
        """Carry a shape that changed on one side to the other; returns how many moved.

        `direction` is "to_sheet" (entering 2D: the model tells the worksheet), "to_model"
        (entering 3D: the worksheet tells the model) or "both" (the Update Links command).
        Only the shape travels and every side keeps its own placement: the new shape is put
        where the old one lay by the rigid fit of `linking.placement`. A plate is the only
        kind that may change in 3D; a change of a face outline on the sheet is left where it
        is with a word about it. When both sides changed, the model wins. A link whose group
        or paths are gone is dropped. One undo step, only when something really changed.
        """
        doc = self.doc
        if not doc.links:
            if announce_empty and direction == "both":
                self.hint.setText(tr("Nothing to update."))
            return 0
        live = {int(g.id) for g in doc.groups} | {ROOT}
        present = {int(p.src) for p in doc.sheet.paths if p.closed and len(p.pts) >= 3}
        alive = [l for l in doc.links if int(l.group) in live and int(l.src) in present]
        dropped = len(doc.links) - len(alive)
        to_sheet, to_model, face_only, both = [], [], 0, 0
        for l in alive:
            sheet_loops = self._link_sheet_loops(l.src)
            ents = self._link_ents(l)
            man = ents.to_manifold() if ents is not None else None
            plate = linking.plate_of(man) if (man is not None and not man.is_empty()) else None
            three = None
            if plate is not None and l.kind == "plate":
                three = linking.sorted_loops(plate["loops"])
            elif l.kind == "face":
                frame = linking.frame_of(l.frame)
                if frame is not None and man is not None and not man.is_empty():
                    three = linking.sorted_loops(linking.face_shape(man, frame) or [])
            ch3 = three is not None and linking.shape_changed(l.shape, three)
            ch2 = linking.shape_changed(l.shape, sheet_loops)
            push_sheet = direction in ("to_sheet", "both") and ch3
            push_model = direction in ("to_model", "both") and ch2 and l.kind == "plate" \
                and plate is not None
            if push_sheet and push_model:
                push_model = False
                both += 1
            elif direction in ("to_model", "both") and ch2 and l.kind == "face":
                face_only += 1
            if push_sheet:
                to_sheet.append((l, three, sheet_loops))
            elif push_model:
                to_model.append((l, sheet_loops, plate))
        changed = bool(to_sheet or to_model or dropped)
        if changed:
            doc.checkpoint()
            if dropped:
                doc.links = alive
            for l, three, sheet_loops in to_sheet:
                self._replace_sheet_src(int(l.src),
                                        linking.place(three, linking.placement(l.shape, sheet_loops)))
                l.shape = three
            for l, sheet_loops, plate in to_model:
                new = linking.place(linking.place(sheet_loops,
                                                  linking.invert(linking.placement(l.shape, sheet_loops))),
                                    linking.placement(l.shape, linking.sorted_loops(plate["loops"])))
                solid = linking.plate_solid(new, plate["frame"],
                                            l.thickness if l.thickness else plate["thickness"])
                g = doc.group(int(l.group))
                if g is not None:
                    g.ents = Entities.from_manifold(solid)
                l.shape = linking.sorted_loops(new)
            doc.notify()
            if self.mode == "2d":
                self.sheet_view._refresh_outside(announce=False)
            else:
                self.view.update()
        updated = len(to_sheet) + len(to_model)
        if both:
            self.hint.setText(trn("Both sides changed: the model won for {n} link.",
                                  "Both sides changed: the model won for {n} links.", both)
                              .format(n=both))
        elif updated:
            self.hint.setText(trn("{n} link updated.", "{n} links updated.", updated)
                              .format(n=updated))
        elif face_only:
            self.hint.setText(tr("This link only goes from 3D to 2D: a change on the worksheet "
                                 "is not put back in the model."))
        elif announce_empty and direction == "both":
            self.hint.setText(tr("Nothing to update."))
        return updated

    def _replace_sheet_src(self, src, loops):
        """The paths of one source element replaced by `loops`, on the same layer."""
        sheet = self.doc.sheet
        old = [p for p in sheet.paths if int(p.src) == src]
        if not old:
            return
        layer, filled, rule = old[0].layer, old[0].filled, old[0].rule
        sheet.paths = [p for p in sheet.paths if int(p.src) != src]
        for l in loops:
            sheet.add_path(np.asarray(l, dtype=float), closed=True, layer=layer,
                           filled=filled, rule=rule, src=src)

    def update_links(self, checked=False):
        """Tools > Update Links: sync both directions now."""
        self.sync_links("both", announce_empty=True)

    def unlink(self, checked=False):
        """Tools > Unlink: drop the links of the selected parts and outlines."""
        doc = self.doc
        links = doc.links
        if not links:
            self.hint.setText(tr("There are no links to remove."))
            return 0
        groups = {int(g.id) for g in doc.selected_groups()}
        srcs = {int(p.src) for p in self.sheet_view.selected_paths()} if self.mode == "2d" else set()
        if not groups and not srcs:
            self.hint.setText(tr("First select the parts or the outlines that are linked."))
            return 0
        keep = [l for l in links if int(l.group) not in groups and int(l.src) not in srcs]
        removed = len(links) - len(keep)
        if not removed:
            self.hint.setText(tr("Nothing to update."))
            return 0
        doc.checkpoint()
        doc.links = keep
        doc.notify()
        self.hint.setText(trn("{n} link removed.", "{n} links removed.", removed)
                          .format(n=removed))
        return removed

    # ------------------------------------------------------------ export for laser and mill
    def sheet_layers_for_export(self) -> list:
        """The layers of the worksheet that mean something to the machine."""
        return [n for n, lay in self.doc.sheet.layers.items()
                if lay.visible and lay.op != "none"]

    def default_sheet_export_options(self) -> dict:
        """The default choices: all visible work layers, compensation only for milling."""
        names = self.sheet_layers_for_export()
        mill = any(self.doc.sheet.layers[n].op in MILL_OPS
                   and any(p.layer == n for p in self.doc.sheet.paths) for n in names)
        return {"layers": names, "compensate": mill, "per_layer": False, "per_sheet": False}

    def sheet_export_options(self, kind="svg"):
        """The "Export for Laser/Mill" dialog; None on cancel, else an options dict."""
        dlg = SheetExportDialog(self, kind)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def sheet_export_drawing(self, options) -> "Drawing":
        """The drawing of the chosen layers, with the compensation when it is on."""
        sheet = self.doc.sheet
        names = [n for n in (options.get("layers") or ()) if n in sheet.layers]
        out = Drawing(layers={n: replace(sheet.layers[n], params=dict(sheet.layers[n].params))
                              for n in names},
                      **sheet._sheet_kwargs())
        for p in sheet.paths:
            if p.layer in names:
                out.paths.append(VPath(np.asarray(p.pts, dtype=float).copy(), p.closed, p.layer,
                                       p.filled, p.rule, p.src, dict(p.meta)))
        for n in names:
            out.ensure_layer(n)
        if options.get("compensate"):
            out = cam.compensate(out)
        return out

    @staticmethod
    def layer_file_name(path, name, kind) -> str:
        """`filename_layername.svg`: the layer name made fit for a file name."""
        base = os.path.splitext(path)[0]
        safe = ("".join(ch for ch in str(name) if ch not in '<>:"/\\|?*\n').strip(" .")
                or tr("layer"))
        return f"{base}_{safe}.{kind}"

    @staticmethod
    def sheet_file_name(path, index, kind) -> str:
        """`filename_sheet1.svg`: the file of one sheet (the word in the user's language)."""
        base = os.path.splitext(path)[0]
        return tr("{base}_sheet{index}.{ext}").format(base=base, index=int(index), ext=kind)

    @staticmethod
    def _write_vector(drawing, path, kind, layers, fill_layers=(),
                      s_max=GCODE_S_MAX_DEFAULT, dynamic=True):
        """Write one drawing to an SVG, DXF or G-code file, with the chosen layers."""
        if kind == "dxf":
            dxf.write_dxf(drawing, path, layers=layers)
        elif kind == "gcode":
            text = gcode.grbl_laser(drawing, s_max=s_max, dynamic=dynamic)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        else:
            svg.write_svg(drawing, path, layers=layers, fill_layers=fill_layers)

    # ------------------------------------------------------------ 2D vector export
    def vector_export_options(self):
        """The 'Export 2D Vector' dialog; None on cancel, else an options dict."""
        dlg = VectorExportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def vector_export_drawing(self, options):
        """The drawing for these choices: selected faces, top view or section."""
        doc = self.doc
        source = options.get("source", "top")
        if source == "faces":
            return vector_export.drawing_from_faces(doc)
        only = bool(options.get("only_selected"))
        if source == "section":
            return vector_export.drawing_section(doc, float(options.get("z", 0.0)), only)
        return vector_export.drawing_top(doc, only)

    def vector_export_empty_message(self, options, drawing):
        """Why there is nothing to write, in words that help the user."""
        source = options.get("source", "top")
        if source == "faces":
            return tr("No face is selected. Click a face (or draw a rectangle) and try again.")
        only = bool(options.get("only_selected"))
        if source == "section" and ops.parts(self.doc, only_selected=only):
            return tr("At height {z} the model cuts nothing: there is no volume there.").format(
                z=format_length(float(options.get('z', 0.0))))
        return tr("There are no closed volumes to export. Draw a face and use Push/Pull (P), "
                  "or close the group with Print Check.")

    def export_vector(self, kind="svg", path=None, options=None):
        """File > Export > 2D Vector: write the drawing to SVG or DXF.

        Without `options` the dialog asks what should go; without `path` Qt asks for a
        file name. On the 2D worksheet the dialog is about the worksheet itself and only
        the visible layers are written. Returns the written path, or None.

        With a `path` the whole export runs synchronously (and returns the path); the
        interactive export without one runs in the background, so a big model does not
        freeze the window, and returns None right away.
        """
        kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        if self.mode == "2d":
            return self.export_sheet(kind, path)
        opts = self.vector_export_options() if options is None else dict(options)
        if opts is None:
            return None
        if path is not None:
            return self._export_vector_sync(kind, path, opts)
        return self._export_vector_async(kind, opts)

    def _write_vector_async(self, drawing, path, kind):
        """Write an already built drawing to `path` in the background."""
        def write(report):
            report(0.0, tr("Exporting…"))
            (dxf.write_dxf if kind == "dxf" else svg.write_svg)(drawing, path)
            return len(drawing.paths)

        def written(n):
            self.hint.setText(trn("{n} outline exported to {where}",
                                  "{n} outlines exported to {where}", n)
                              .format(n=n, where=path))

        background.run(self, write, text=tr("Exporting…"), can_cancel=False, on_done=written,
                       on_error=self._vector_export_failed)
