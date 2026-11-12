"""The panels of the Default Tray: what sits under the collapsible headers.

Entity Info, Materials, Outliner, Print Check and the Layers of the 2D worksheet.
Every panel takes the main window, reads and changes `win.doc` and is asked to
`refresh()` when the document changes; `volura/tray.py` builds the dock that holds
them.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import background, ops, theme
from .i18n import tr, trn
from .icons import icon
from .model import ROOT
from .vector2d import (CORNER_LABELS, CORNER_STYLES, LASER_OPS, LASER_PARAMS, MILL_OPS,
                       MILL_PARAMS, OP_LABELS, OPS, PARAM_LABELS, layer_param)


def note(text):
    """An explanatory line under the fields of a dialog page."""
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def swatch(color, size=16):
    """A color square with a gray border: black and white stay visible in both themes."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    p = QPainter(pm)
    p.setPen(QPen(QColor("#808080"), 1.0))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------- panels

class MaterialsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        head = QLabel(tr("Filaments (the colors you use get filament 1, 2, 3… in this order):"))
        head.setWordWrap(True)
        lay.addWidget(head)
        self.list = QListWidget()
        self.list.setAccessibleName(tr("Filaments"))
        self.list.setIconSize(QSize(22, 22))
        self.list.itemClicked.connect(self.pick)
        self.list.itemDoubleClicked.connect(lambda it: self.edit())
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, fn in ((tr("New"), self.add), (tr("Edit"), self.edit), (tr("Remove"), self.remove)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        doc = self.win.doc
        self.list.clear()
        cur = self.win.settings.current_color.lower()
        for i, (name, color) in enumerate(doc.materials, 1):
            it = QListWidgetItem(swatch(color, 22), f"{i}: {name}")
            it.setData(Qt.UserRole, color)
            self.list.addItem(it)
            if color.lower() == cur:
                self.list.setCurrentItem(it)

    def pick(self, it):
        self.win.settings.current_color = it.data(Qt.UserRole)
        self.win.set_tool("paint")

    def add(self):
        c = QColorDialog.getColor(QColor("#888888"), self, tr("New filament color"))
        if not c.isValid():
            return
        doc = self.win.doc
        doc.checkpoint()
        doc.materials.append([tr("Color {n}").format(n=len(doc.materials) + 1), c.name()])
        self.win.settings.current_color = c.name()
        doc.notify()

    def edit(self):
        row = self.list.currentRow()
        doc = self.win.doc
        if row < 0:
            return
        name, color = doc.materials[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Edit Filament"))
        form = QFormLayout(dlg)
        ed = QLineEdit(name)
        btn = QPushButton()
        state = {"color": color}
        btn.setIcon(swatch(color, 20))
        btn.setText(color)

        def choose():
            c = QColorDialog.getColor(QColor(state["color"]), dlg)
            if c.isValid():
                state["color"] = c.name()
                btn.setIcon(swatch(c.name(), 20))
                btn.setText(c.name())
        btn.clicked.connect(choose)
        form.addRow(tr("Name"), ed)
        form.addRow(tr("Color"), btn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        doc.checkpoint()
        new = state["color"]
        doc.materials[row] = [ed.text() or name, new]
        if new.lower() != color.lower():
            for g in doc.groups:
                if g.color.lower() == color.lower():
                    g.color = new
            for ents in [doc.root] + [g.ents for g in doc.groups]:
                for f, face in list(ents.F.items()):
                    if face.color and face.color.lower() == color.lower():
                        ents.set_face_color(f, new)
            if self.win.settings.current_color.lower() == color.lower():
                self.win.settings.current_color = new
        doc.notify()

    def remove(self):
        row = self.list.currentRow()
        doc = self.win.doc
        if row < 0 or len(doc.materials) <= 1:
            return
        doc.checkpoint()
        doc.materials.pop(row)
        doc.notify()


# Entity Info checks a group with fewer faces than this right away; a bigger one (an
# imported scan) in the background, so selecting it does not freeze the window.
PART_INFO_BACKGROUND_FACES = 5000


class EntityInfoPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.addStretch(1)
        self.body = None
        self._part_cache = {}           # id(ents) -> (ents, version, (name, color), summary)
        self._part_pending = set()      # (id(ents), version, (name, color)) being checked

    def refresh(self):
        if self.body is not None:
            self.lay.removeWidget(self.body)
            self.body.hide()
            self.body.deleteLater()
        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        doc = self.win.doc
        if getattr(self.win, "mode", "3d") == "2d":
            self._sheet_form(form)
            self.lay.insertWidget(0, self.body)
            return
        sel = doc.selection
        groups = doc.selected_groups()
        faces = [(c, i) for k, c, i in sel if k == "f"]
        edges = [(c, i) for k, c, i in sel if k == "e"]
        guides = [(c, i) for k, c, i in sel if k == "guide"]
        if doc.active != ROOT and doc.group(doc.active):
            form.addRow(QLabel(tr("<i>Editing: {name}</i>").format(name=doc.group(doc.active).name)))
        if not sel:
            form.addRow(QLabel(tr("<b>No selection</b>")))
            form.addRow(tr("Groups"), QLabel(str(len(doc.groups))))
            form.addRow(tr("Loose faces"), QLabel(str(len(doc.root.F))))
            form.addRow(tr("Loose edges"), QLabel(str(len(doc.root.E))))
            form.addRow(tr("Guides"), QLabel(str(len(doc.guides))))
        elif len(sel) == 1 and groups:
            self._group_form(form, groups[0])
        elif len(sel) == 1 and guides:
            g = doc.guide(guides[0][1])
            if g is not None:
                form.addRow(QLabel(tr("<b>Guide</b>") if g.is_line else tr("<b>Guide Point</b>")))
                form.addRow(tr("Point"), QLabel(f"{g.point[0]:.2f}, {g.point[1]:.2f}, {g.point[2]:.2f} mm"))
                if g.is_line:
                    d = g.direction
                    form.addRow(tr("Direction"), QLabel(f"{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}"))
        elif len(sel) == 1 and faces:
            c, f = faces[0]
            ents = doc.ents(c)
            tri, _ = ents.face_triangles(f)
            area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
            form.addRow(QLabel(tr("<b>Face</b>")))
            form.addRow(tr("Area"), QLabel(f"{area:.2f} mm²"))
            form.addRow(tr("Color"), QLabel(ents.F[f].color or tr("default")))
        elif len(sel) == 1 and edges:
            c, e = edges[0]
            ents = doc.ents(c)
            A, B = ents.edge_points(e)
            form.addRow(QLabel(tr("<b>Edge</b>")))
            form.addRow(tr("Length"), QLabel(f"{np.linalg.norm(B - A):.3f} mm"))
            form.addRow(tr("Faces"), QLabel(str(len(ents.e2f.get(e, ())))))
            form.addRow(tr("Softened"), QLabel(tr("yes") if e in ents.soft else tr("no")))
        else:
            form.addRow(QLabel(tr("<b>{count} items selected</b>").format(count=len(sel))))
            form.addRow(tr("Groups"), QLabel(str(len(groups))))
            form.addRow(tr("Faces"), QLabel(str(len(faces))))
            form.addRow(tr("Edges"), QLabel(str(len(edges))))
            if guides:
                form.addRow(tr("Guides"), QLabel(str(len(guides))))
            bb = ops.selection_bbox(doc)
            if bb is not None:
                d = bb[1] - bb[0]
                form.addRow(tr("Dimensions"), QLabel(f"{d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} mm"))
            if len(groups) > 1:
                self._totals_form(form, groups)
        self.lay.insertWidget(0, self.body)

    def _sheet_form(self, form):
        """Entity Info on the worksheet: the selected outlines and the sheet."""
        sheet = self.win.doc.sheet
        sel = self.win.sheet_view.selected_paths()
        paths = sel or sheet.paths
        if sel:
            form.addRow(QLabel(tr("<b>1 outline selected</b>") if len(sel) == 1
                               else tr("<b>{count} outlines selected</b>").format(count=len(sel))))
        else:
            form.addRow(QLabel(tr("<b>No selection</b>")))
            form.addRow(tr("Outlines"), QLabel(str(len(sheet.paths))))
        layers = sorted({p.layer for p in paths})
        if sel and len(layers) == 1:
            lay = sheet.layers.get(layers[0])
            op = OP_LABELS.get(lay.op, lay.op) if lay else ""
            form.addRow(tr("Layer"), QLabel(f"{layers[0]} ({op})" if op else layers[0]))
        elif layers:
            form.addRow(tr("Layers"), QLabel(str(len(layers))))
        if sel:
            closed = sum(1 for p in sel if p.closed)
            form.addRow(tr("Closed"), QLabel(tr("{closed} of {total}").format(closed=closed,
                                                                            total=len(sel))))
            pts = [p.pts for p in sel if len(p.pts)]
            if pts:
                P = np.vstack(pts)
                lo, hi = P.min(axis=0), P.max(axis=0)
                form.addRow(tr("Dimensions"), QLabel(f"{hi[0] - lo[0]:.2f} × {hi[1] - lo[1]:.2f} mm"))
                form.addRow(tr("Position"), QLabel(f"{lo[0]:.2f}, {lo[1]:.2f} mm"))
        form.addRow(QLabel(tr("<b>Sheet</b>")))
        if sheet.sheet_w and sheet.sheet_h:
            form.addRow(tr("Size"), QLabel(f"{sheet.sheet_w:g} × {sheet.sheet_h:g} mm"))
            form.addRow(tr("Count"), QLabel(str(int(sheet.sheet_count or 1))))
        else:
            form.addRow(tr("Size"), QLabel(tr("not set")))
        form.addRow(tr("Thickness"), QLabel(f"{sheet.sheet_thickness:g} mm" if sheet.sheet_thickness
                                            else tr("not set")))

    def _group_form(self, form, g):
        doc = self.win.doc
        form.addRow(QLabel(tr("<b>Group (printable part)</b>")))
        name = QLineEdit(g.name)

        def rename():
            if name.text() and name.text() != g.name:
                self._apply(lambda: setattr(doc.group(g.id), "name", name.text()))
        name.editingFinished.connect(rename)
        form.addRow(tr("Name"), name)
        combo = QComboBox()
        idx = -1
        for i, (mname, color) in enumerate(doc.materials):
            combo.addItem(swatch(color), f"{i + 1}: {mname}", color)
            if color.lower() == g.color.lower():
                idx = i
        if idx < 0:
            combo.addItem(swatch(g.color), g.color, g.color)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

        def recolor(i):
            c = combo.itemData(i)
            if c and c.lower() != g.color.lower():
                self._apply(lambda: setattr(doc.group(g.id), "color", c))
        combo.currentIndexChanged.connect(recolor)
        form.addRow(tr("Filament"), combo)
        bb = g.bbox()
        if bb is None:
            return
        lo, hi = bb
        size = hi - lo
        spins = []
        for label, v in zip((tr("Width (X)"), tr("Depth (Y)"), tr("Height (Z)")), size):
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 100000)
            sp.setDecimals(2)
            sp.setSuffix(" mm")
            sp.setValue(v)
            spins.append(sp)
            form.addRow(label, sp)
        lock = QCheckBox(tr("Keep proportions"))
        lock.setChecked(True)
        form.addRow(lock)

        def resize(changed):
            new = np.array([s.value() for s in spins])
            if np.allclose(new, size, atol=1e-3):
                return
            if lock.isChecked() and size[changed] > 1e-9:
                new = size * (new[changed] / size[changed])
            self._apply(lambda: ops.resize_group(doc, doc.group(g.id), new))
        for i, sp in enumerate(spins):
            sp.editingFinished.connect(lambda i=i: resize(i))
        form.addRow(tr("Position"), QLabel(", ".join(f"{round(v, 2) + 0.0:.2f}" for v in lo) + " mm"))
        summary = self._summary(g)
        if summary is None:
            form.addRow(tr("Printable"), QLabel(tr("Calculating…")))
            return
        self._part_rows(form, summary)

    def _summary(self, g):
        """``(volume, surface, problem)`` of the group `g`, from the cache or checked now.

        None means the group is too big to check on the GUI thread: the check runs in the
        background and refreshes the panel when it is done."""
        ents, version, key = g.ents, g.ents.version, (g.name, g.color)
        cached = self._part_cache.get(id(ents))
        if cached is not None and cached[0] is ents and cached[1:3] == (version, key):
            return cached[3]
        if len(ents.F) < PART_INFO_BACKGROUND_FACES:
            summary = ops.part_summary(g.name, g.color, ents)
            self._remember_part(ents, version, key, summary)
            return summary
        # a big group: the check triangulates every face, seconds on an imported scan, so
        # it runs in a worker process and the panel is refreshed when it is done
        if (id(ents), version, key) in self._part_pending:
            return None                    # already on its way
        self._part_pending.add((id(ents), version, key))
        body, name, color = self.body, g.name, g.color

        def task(report):
            return ops.part_summary(name, color, ents, pool=True)

        def finish():
            self._part_pending.discard((id(ents), version, key))

        def done(summary):
            finish()
            if ents.version != version:
                return                   # edited meanwhile: a newer refresh follows
            self._remember_part(ents, version, key, summary)
            if self.body is not None and not self.body.isHidden():
                self.refresh()           # shows it now, if this group is still selected

        background.run(None, task, text=tr("Calculating…"), can_cancel=False,
                       show_dialog=False, on_done=done, on_error=lambda e: finish())
        return None

    def remember_part(self, g, summary):
        """The Entity Info check of `g` as a background task already made it (an import,
        a simplify), so selecting the group needs no second check."""
        self._remember_part(g.ents, g.ents.version, (g.name, g.color), summary)

    def _remember_part(self, ents, version, key, summary):
        self._part_cache[id(ents)] = (ents, version, key, summary)
        while len(self._part_cache) > 16:          # it holds the geometry: only a few
            del self._part_cache[next(iter(self._part_cache))]

    def _part_rows(self, form, summary):
        """Volume, surface, filament and whether the group prints, as `summary` gives them."""
        vol, area, problem = summary
        if problem is None:
            form.addRow(tr("Volume"), QLabel(f"{vol / 1000:.3f} cm³"))
            form.addRow(tr("Surface"), QLabel(f"{area / 100:.2f} cm²"))
            self._filament_rows(form, [(vol, area)])
            ok = QLabel(tr("✔ closed volume (solid)"))
            ok.setStyleSheet(f"color: {theme.color('ok')}; font-weight: bold;")
        else:
            ok = QLabel(f"✖ {problem}")
            ok.setStyleSheet(f"color: {theme.color('error')}; font-weight: bold;")
            ok.setWordWrap(True)
        form.addRow(tr("Printable"), ok)

    def _filament_rows(self, form, parts):
        """The weight and the filament a print of ``[(volume, area)]`` parts takes here.

        The rows show what a slicer lays down with the walls and the infill of the
        Printer Setup page; every part gets its own skin. The weight of a solid print
        stays under it, small, because that is the most these parts can use."""
        s = self.win.settings
        vol = sum(v for v, _ in parts)
        print_vol = sum(ops.print_volume(v, a, s.print_shell, s.print_infill / 100.0)
                        for v, a in parts)
        solid = ops.filament_use(vol, s.filament_density, s.filament_diameter, s.filament_price)
        grams, metres, cost = ops.filament_use(print_vol, s.filament_density,
                                               s.filament_diameter, s.filament_price)
        weight = QLabel(f"≈ {grams:.1f} g ({tr(s.filament_material)}, "
                        f"{tr('{n} % infill').format(n=s.print_infill)})"
                        f"<br><span style='font-size: small; color:{theme.color('muted')}'>"
                        f"{tr('solid')}: {solid[0]:.1f} g</span>")
        weight.setTextFormat(Qt.RichText)
        form.addRow(tr("Weight"), weight)
        form.addRow(tr("Filament"), QLabel(f"≈ {metres:.2f} m · {cost:.2f}"))

    def _totals_form(self, form, groups):
        """What a whole print job takes: the volume, weight, filament and cost of all the
        closed groups of the selection together."""
        summaries = [self._summary(g) for g in groups]
        if any(s is None for s in summaries):
            form.addRow(tr("Total volume"), QLabel(tr("Calculating…")))
            return
        closed = [s for s in summaries if s[2] is None]
        skipped = len(summaries) - len(closed)
        vol = sum(s[0] for s in closed)
        form.addRow(tr("Total volume"), QLabel(f"{vol / 1000:.3f} cm³"))
        if closed:
            self._filament_rows(form, [(s[0], s[1]) for s in closed])
        if skipped:
            text = trn("{n} part is not closed", "{n} parts are not closed", skipped)
            form.addRow(tr("Not counted"), QLabel(text.format(n=skipped)))

    def _apply(self, fn):
        doc = self.win.doc
        doc.checkpoint()
        fn()
        QTimer.singleShot(0, doc.notify)


class OutlinerPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QListWidget()
        self.list.setAccessibleName(tr("Parts of the model"))
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self.on_select)
        self.list.itemChanged.connect(self.on_changed)
        self.list.itemDoubleClicked.connect(self.on_double)
        lay.addWidget(self.list)
        self._updating = False

    def refresh(self):
        self._updating = True
        doc = self.win.doc
        self.list.clear()
        sel = {i for k, _, i in doc.selection if k == "g"}
        for g in doc.groups:
            it = QListWidgetItem(swatch(g.color), g.name)
            it.setData(Qt.UserRole, g.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if g.visible else Qt.Unchecked)
            self.list.addItem(it)
            it.setSelected(g.id in sel)
        if doc.root.F or doc.root.E:
            it = QListWidgetItem(tr("Loose geometry ({n} faces)").format(n=len(doc.root.F)))
            it.setData(Qt.UserRole, None)
            it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
            self.list.addItem(it)
        self._updating = False

    def on_select(self):
        if self._updating:
            return
        doc = self.win.doc
        if doc.active != ROOT:
            return
        ids = [it.data(Qt.UserRole) for it in self.list.selectedItems() if it.data(Qt.UserRole)]
        keep = [s for s in doc.selection if s[0] != "g"]
        doc.set_selection(keep + [("g", i, i) for i in ids])

    def on_double(self, it):
        gid = it.data(Qt.UserRole)
        if gid and not self.win.doc.enter_group(gid):
            self.win.warn_locked()      # a locked group does not open

    def on_changed(self, it):
        if self._updating:
            return
        doc = self.win.doc
        g = doc.group(it.data(Qt.UserRole)) if it.data(Qt.UserRole) else None
        if g is None:
            return
        vis = it.checkState() == Qt.Checked
        if vis != g.visible or it.text() != g.name:
            doc.checkpoint()
            g.visible = vis
            g.name = it.text() or g.name
            QTimer.singleShot(0, doc.notify)


class PrintCheckPanel(QWidget):
    """Tray panel "Print Check": why something is not a valid volume (Solid Inspector).

    One row per part with its problems underneath. Clicking a problem selects the
    geometry involved and marks it red in the viewport; double clicking zooms on it."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.problems = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.tree = QTreeWidget()
        self.tree.setAccessibleName(tr("Problems of the parts"))
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self.on_current)
        self.tree.itemDoubleClicked.connect(self.on_double)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.b_check = QPushButton(tr("Check"))
        self.b_check.clicked.connect(self.refresh)
        self.b_fix = QPushButton(tr("Repair All"))
        self.b_fix.clicked.connect(self.repair)
        row.addWidget(self.b_check)
        row.addWidget(self.b_fix)
        lay.addLayout(row)
        self.summary = QLabel(tr("Click Check to test every part for a closed volume."))
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

    # ------------------------------------------------------------ checking
    def refresh(self, checked=False):
        self.problems = ops.inspect(self.win.doc)
        by_part, order = {}, []
        for n, p in enumerate(self.problems):
            key = (p.ctx, p.part)
            if key not in by_part:
                by_part[key] = []
                order.append(key)
            by_part[key].append(n)
        self.tree.blockSignals(True)
        self.tree.clear()
        for key in order:
            head = QTreeWidgetItem([f"{key[1]}  ({len(by_part[key])})"])
            head.setForeground(0, theme.qcolor("error"))
            self.tree.addTopLevelItem(head)
            for n in by_part[key]:
                p = self.problems[n]
                it = QTreeWidgetItem([p.text()])
                it.setData(0, Qt.UserRole, n)
                it.setToolTip(0, tr("{kind}: {detail}\n{faces} faces, {edges} edges").format(
                    kind=p.kind, detail=p.detail, faces=len(p.faces), edges=len(p.edges)))
                head.addChild(it)
            head.setExpanded(True)
        if not order:
            ok = QTreeWidgetItem([tr("✔ every part is a closed volume")])
            ok.setForeground(0, theme.qcolor("ok"))
            self.tree.addTopLevelItem(ok)
        self.tree.blockSignals(False)
        if order:
            bad = len({p.part for p in self.problems})
            self.summary.setText(trn("{count} part with problems. Click a problem to see it; "
                                     "double-click zooms to it.",
                                     "{count} parts with problems. Click a problem to see it; "
                                     "double-click zooms to it.", bad).format(count=bad))
        else:
            self.summary.setText(tr("All parts are closed volumes and can be printed."))
        self.win.view.problem_items = []
        self.win.view.update()

    def repair(self, checked=False):
        """Repair All: one undo step that closes and orients the parts."""
        fixed = ops.repair_all(self.win.doc)
        self.refresh()
        self.win.hint.setText(tr("Print Check: parts repaired (Undo reverts it).")
                              if fixed else tr("Print Check: there was nothing to repair."))

    # ------------------------------------------------------------ picking
    def _problem(self, it):
        n = None if it is None else it.data(0, Qt.UserRole)
        if n is None or n >= len(self.problems):
            return None
        return self.problems[n]

    def on_current(self, cur, prev=None):
        p = self._problem(cur)
        if p is None:
            self.win.view.problem_items = []
            self.win.view.update()
            return
        items = p.items()
        self.win.view.problem_items = items
        self.win.doc.set_selection(items)
        self.win.view.update()

    def on_double(self, it, column=0):
        p = self._problem(it)
        if p is None:
            return
        items = p.items()
        self.win.view.problem_items = items
        self.win.doc.set_selection(items)
        self.win.view.zoom_items(items)

    def clear_marks(self):
        self.win.view.problem_items = []


# ---------------------------------------------------------------- layers of the 2D worksheet

class LayerParamsDialog(QDialog):
    """The "Layer settings" dialog: what the machine should do with that layer.

    A laser layer gets power, speed, number of passes and the kerf; a mill layer the
    tool diameter, the depth, the step depth, the stepover and the corner relief for
    the inside corners. The values are stored in `Layer.params` and so travel along in
    the file and in the SVG export.
    """

    SUFFIX = {"power": " %", "stepover": " %", "speed": " mm/s"}

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.op = getattr(layer, "op", "cut")
        self.fields = {}
        self.setWindowTitle(tr("Settings of Layer {name}").format(name=layer.name))
        form = QFormLayout(self)
        form.addRow(QLabel(tr("Operation: {op}").format(op=OP_LABELS.get(self.op, self.op))))
        keys = (LASER_PARAMS if self.op in LASER_OPS
                else MILL_PARAMS if self.op in MILL_OPS else ())
        for key, default in keys:
            value = layer_param(layer, key, default)
            if key == "passes":
                w = QSpinBox()
                w.setRange(1, 50)
                w.setSuffix("×")
                w.setValue(int(value))
            elif key == "corner":
                w = QComboBox()
                for style in CORNER_STYLES:
                    w.addItem(CORNER_LABELS[style], style)
                w.setCurrentIndex(max(0, list(CORNER_STYLES).index(
                    value if value in CORNER_STYLES else "none")))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(0.0, 100000.0)
                w.setSingleStep(0.05 if key in ("kerf", "tool_d", "step_depth", "depth") else 1.0)
                w.setSuffix(self.SUFFIX.get(key, " mm"))
                w.setValue(float(value))
            w.setToolTip(PARAM_LABELS.get(key, key))
            form.addRow(PARAM_LABELS.get(key, key), w)
            self.fields[key] = w
        if not keys:
            form.addRow(note(tr("This layer is for information only and is not exported.")))
        elif self.op in MILL_OPS:
            form.addRow(note(tr("The compensation shifts the outline sideways by half the tool "
                                "diameter. A pocket runs concentric paths with the stepover until "
                                "the area is cleared. A corner relief gives the cutter room in the "
                                "inside corners.")))
        else:
            form.addRow(note(tr("The kerf compensation moves outer edges out by half the kerf and "
                                "makes holes smaller by the same amount, so the part has the drawn "
                                "size after cutting.")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        out = {}
        for key, w in self.fields.items():
            if isinstance(w, QSpinBox):
                out[key] = int(w.value())
            elif isinstance(w, QComboBox):
                out[key] = str(w.currentData())
            else:
                out[key] = float(w.value())
        return out


class LayersPanel(QWidget):
    """Tray panel "Layers": color, name, operation, visibility and settings per layer.

    One row per layer of the 2D worksheet. Click the color box to choose a color (red
    cuts, blue engraves, as laser software expects), double-click the name to rename,
    choose the operation in the combo box, and turn a layer off with the eye or lock it
    with the padlock. New paths go on the active layer; you choose it by clicking a row.
    """

    COLS = 6
    WIDTHS = (26, 26, 30, 90, 116, 36)

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.names: list[str] = []
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(QLabel(tr("Layers of the 2D worksheet (click a row for the active layer):")))
        self.table = QTableWidget(0, self.COLS)
        self.table.setAccessibleName(tr("Layers of the worksheet"))
        self.table.setHorizontalHeaderLabels(["", "", "", tr("Layer"), tr("Operation"), ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.setIconSize(QSize(16, 16))
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.on_row)
        self.table.itemChanged.connect(self.on_rename)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_add = QPushButton(tr("Add"))
        self.b_add.clicked.connect(self.add_layer)
        self.b_remove = QPushButton(tr("Delete Layer"))
        self.b_remove.clicked.connect(self.remove_layer)
        row.addWidget(self.b_add)
        row.addWidget(self.b_remove)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.b_active = QPushButton(tr("Active Layer"))
        self.b_active.clicked.connect(self.pick_active)
        self.b_params = QPushButton(tr("Settings…"))
        self.b_params.clicked.connect(self.edit_params)
        row.addWidget(self.b_active)
        row.addWidget(self.b_params)
        lay.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.refresh()

    # ------------------------------------------------------------ building

    @contextmanager
    def _quiet(self):
        """No reaction to our own changes in the table (they come from the model)."""
        saved, self._busy = self._busy, True
        try:
            yield
        finally:
            self._busy = saved

    def refresh(self, checked=False):
        sheet = self.win.doc.sheet
        active = self.win.sheet_view.current_layer
        with self._quiet():
            self.names = list(sheet.layers)
            self.table.clearContents()
            self.table.setRowCount(len(self.names))
            for row, name in enumerate(self.names):
                lay = sheet.layers[name]
                it = QTableWidgetItem(name)
                if name == active:
                    font = QFont()
                    font.setBold(True)
                    it.setFont(font)
                it.setToolTip(tr("Double-click to rename"))
                self.table.setItem(row, 3, it)
                self.table.setCellWidget(row, 0, self._button(
                    row, "eye" if lay.visible else "eye_off",
                    tr("Hide layer (is not exported)") if lay.visible
                    else tr("Show layer"), self.toggle_visible))
                self.table.setCellWidget(row, 1, self._button(
                    row, "lock" if lay.locked else "unlock",
                    tr("Lock layer (can no longer be selected)") if not lay.locked
                    else tr("Unlock layer"), self.toggle_locked))
                self.table.setCellWidget(row, 2, self._button(
                    row, "paint", tr("Layer color ({color})").format(color=lay.color), self.pick_color))
                self.table.setCellWidget(row, 4, self._op_box(row, lay))
                self.table.setCellWidget(row, 5, self._button(
                    row, "check", tr("Settings of this layer"), self.edit_params))
            self.table.selectRow(max(0, self.names.index(active)) if active in self.names else 0)
        self.info.setText(self._summary(sheet, active))

    def _summary(self, sheet, active) -> str:
        lay = sheet.layers.get(active)
        if lay is None:
            return ""
        n = sum(1 for p in sheet.paths if p.layer == active)
        if lay.op in MILL_OPS:
            details = tr("cutter Ø{tool:g} mm, depth {depth:g} mm").format(
                tool=layer_param(lay, 'tool_d'), depth=layer_param(lay, 'depth'))
        elif lay.op in LASER_OPS:
            details = tr("{power:g} %, {speed:g} mm/s, {passes}×, kerf {kerf:g} mm").format(
                power=layer_param(lay, 'power'), speed=layer_param(lay, 'speed'),
                passes=layer_param(lay, 'passes'), kerf=layer_param(lay, 'kerf'))
        else:
            details = tr("not exported")
        return tr("Active layer: {name} — {op}; {n} outlines; {details}.").format(
            name=active, op=OP_LABELS.get(lay.op, lay.op), n=n, details=details)

    def _button(self, row, name, tip, slot):
        b = QToolButton()
        b.setIcon(icon(name, 16))
        b.setIconSize(QSize(16, 16))
        b.setAutoRaise(True)
        b.setToolTip(tip)
        b.clicked.connect(lambda checked=False, r=row: slot(r))
        return b

    def _op_box(self, row, lay):
        box = QComboBox()
        for op in OPS:
            box.addItem(OP_LABELS[op], op)
        box.setCurrentIndex(max(0, OPS.index(lay.op) if lay.op in OPS else 0))
        box.setToolTip(tr("What the machine does with this layer"))
        box.currentIndexChanged.connect(lambda _=0, r=row, b=box: self.set_op(r, b))
        return box

    def _layer(self, row):
        try:
            return self.win.doc.sheet.layers[self.names[row]]
        except (IndexError, KeyError):
            return None

    def _edit(self, change, message):
        """One undo step on the worksheet, then update everything."""
        self.win.doc.checkpoint()
        change()
        self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(message)

    # ------------------------------------------------------------ buttons

    def on_row(self):
        if self._busy:
            return
        row = self.table.currentRow()
        if 0 <= row < len(self.names):
            self.win.sheet_view.set_current_layer(self.names[row])
            self.refresh()

    def pick_active(self):
        row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        if lay.locked:
            self.win.hint.setText(tr("Layer {name} is locked; it cannot be the active layer now.")
                                  .format(name=lay.name))
            return
        self.win.sheet_view.set_current_layer(lay.name)
        self.refresh()
        self.win.hint.setText(tr("New outlines go to layer {name}.").format(name=lay.name))

    def on_rename(self, item):
        if self._busy or item is None or item.column() != 3:
            return
        sheet = self.win.doc.sheet
        old = self.names[item.row()] if item.row() < len(self.names) else None
        new = item.text().strip()
        if old is None or not new or new == old:
            self.refresh()
            return
        if new in sheet.layers:
            self.win.hint.setText(tr("There is already a layer named {name}.").format(name=new))
            self.refresh()
            return
        with self._quiet():
            self.win.doc.checkpoint()
            sheet.rename_layer(old, new)
            if self.win.sheet_view.current_layer == old:
                self.win.sheet_view.current_layer = new
            self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(tr("Layer {old} is now called {new}.").format(old=old, new=new))

    def add_layer(self):
        sheet = self.win.doc.sheet
        name = sheet.free_layer_name()
        color = ["#00a0a0", "#a000a0", "#0000a0", "#a0a000", "#00a000"][len(sheet.layers) % 5]

        def change():
            sheet.ensure_layer(name, color, "cut")
        self._edit(change, tr("Layer {name} added; draw or move outlines onto it.").format(name=name))
        if name in self.names:
            self.table.selectRow(self.names.index(name))
            self.win.sheet_view.set_current_layer(name)
            self.refresh()

    def remove_layer(self):
        row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        if lay.name == "0":
            self.win.hint.setText(tr("Layer 0 is the base layer and cannot be removed."))
            return
        sheet = self.win.doc.sheet
        n = sum(1 for p in sheet.paths if p.layer == lay.name)
        if n:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(tr("Delete Layer {name}").format(name=lay.name))
            box.setText(tr("Layer {name} has {n} outlines.").format(name=lay.name, n=n))
            b_move = box.addButton(tr("Paths to layer 0"), QMessageBox.AcceptRole)
            b_del = box.addButton(tr("Delete paths too"), QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked not in (b_move, b_del):
                return
            move = clicked is b_move
        else:
            move = True
        active = self.win.sheet_view.current_layer

        def change():
            sheet.remove_layer(lay.name, move_paths=move)
            if active == lay.name:
                self.win.sheet_view.current_layer = "0"
        if move and n:
            message = tr("Layer {name} deleted ({n} outlines to layer 0).")
        elif n:
            message = tr("Layer {name} deleted with {n} outlines.")
        else:
            message = tr("Layer {name} deleted.")
        self._edit(change, message.format(name=lay.name, n=n))

    def toggle_visible(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.visible

        def change():
            lay.visible = new
        self._edit(change, (tr("Layer {name} visible again.") if new
                            else tr("Layer {name} hidden; it does not go to the machine."))
                   .format(name=lay.name))

    def toggle_locked(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.locked

        def change():
            lay.locked = new
        self._edit(change, (tr("Layer {name} locked: you can no longer click its outlines.") if new
                            else tr("Layer {name} unlocked.")).format(name=lay.name))

    def pick_color(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        color = QColorDialog.getColor(QColor(lay.color or "#000000"), self,
                                      tr("Color of Layer {name}").format(name=lay.name))
        if not color.isValid():
            return

        new = color.name().lower()

        def change():
            lay.color = new
        self._edit(change, tr("Layer {name} is now {color}.").format(name=lay.name, color=new))

    def set_op(self, row, box):
        lay = self._layer(row)
        if lay is None or self._busy:
            return
        op = box.currentData()
        if op == lay.op:
            return

        def change():
            lay.op = op
        self._edit(change, tr("Layer {name}: {op}.").format(name=lay.name, op=OP_LABELS.get(op, op)))

    def edit_params(self, row=None):
        if row is None:
            row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        dlg = LayerParamsDialog(lay, self)
        if dlg.exec() != QDialog.Accepted:
            self.refresh()
            return
        values = dlg.values()

        def change():
            lay.params = {**dict(lay.params or {}), **values}
        self._edit(change, tr("Settings of layer {name} changed.").format(name=lay.name))
