"""The general dialogs of the window: Simplify Model, Model Info and Preferences.

The quality slider of Tools > Simplify Model, the pages of Window > Model Info (units,
statistics and print bed) and of Window > Preferences (general, display, drawing, 3D
mouse and the editable shortcuts). The window they talk to and the actions that open
them live in `volura/app.py`.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QKeySequenceEdit,
                               QListWidget, QListWidgetItem, QPushButton, QSlider,
                               QSpinBox, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from . import background, i18n, keyschemes, ops, theme
from .i18n import tr, tr_noop
from .panels import note
from .tools import format_length, set_length_precision


PRINTERS = [
    ("Bambu Lab X1 / P1 / A1", (256, 256, 256)),
    ("Bambu Lab A1 mini", (180, 180, 180)),
    ("Prusa MK4 / MK3S", (250, 210, 220)),
    ("Prusa MINI", (180, 180, 180)),
    ("Prusa XL", (360, 360, 360)),
    ("Creality Ender 3", (220, 220, 250)),
    ("Creality K1", (220, 220, 250)),
    ("Voron 2.4 (350)", (350, 350, 340)),
]

# The materials on the Printer Setup page: name and density in g/cm³. The dialog adds a
# "Custom" choice with an editable density, so an unusual spool is not left out.
FILAMENT_MATERIALS = [
    ("PLA", 1.24),
    ("PETG", 1.27),
    ("ABS", 1.04),
    ("ASA", 1.07),
    ("TPU", 1.21),
    ("Nylon", 1.14),
]


class SimplifyDialog(QDialog):
    """Tools > Simplify Model: a quality slider and a live estimate of the result.

    The slider has the fixed steps of ops.SIMPLIFY_STEPS; under it the maximum deviation
    in mm and what that step does. The "N triangles -> M triangles" estimate runs in the
    background (no progress dialog, so the slider never stutters on a big model) and only
    the last calculation counts: an older one that finishes later is ignored. OK returns
    the chosen tolerance; the window does the actual simplification."""

    def __init__(self, win, groups):
        super().__init__(win)
        self.win = win
        self.groups = list(groups)
        self.tolerance = ops.SIMPLIFY_DEFAULT_TOLERANCE
        self.preview_before = self.preview_after = None
        self.manifolds = {}             # group id -> its closed manifold (None: not closed)
        self._manifold_lock = threading.Lock()
        self._token = 0
        self._job = None
        self.setWindowTitle(tr("Simplify Model"))
        lay = QVBoxLayout(self)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(ops.SIMPLIFY_STEPS) - 1)
        self.slider.setPageStep(1)
        self.slider.setValue(self._default_index())
        lay.addWidget(self.slider)

        self.tol_label = QLabel()
        lay.addWidget(self.tol_label)
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        lay.addWidget(self.desc_label)
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        lay.addWidget(self.preview_label)
        if any(ops.has_painted_faces(g) for g in self.groups):
            warn = QLabel(tr("Painted face colors are lost; the parts keep the group color."))
            warn.setWordWrap(True)
            lay.addWidget(warn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.slider.valueChanged.connect(self._step_changed)
        self._step_changed(self.slider.value())

    @staticmethod
    def _default_index():
        for i, (tol, _d) in enumerate(ops.SIMPLIFY_STEPS):
            if tol == ops.SIMPLIFY_DEFAULT_TOLERANCE:
                return i
        return len(ops.SIMPLIFY_STEPS) // 2

    def _step_changed(self, index):
        tol, desc = ops.SIMPLIFY_STEPS[index]
        self.tolerance = tol
        self.tol_label.setText(tr("Maximum deviation: {tol} mm").format(tol=tol))
        self.desc_label.setText(tr(desc))
        self._estimate()

    def _estimate(self):
        """Estimate the triangle count after simplifying, off the GUI thread."""
        if self._job is not None and not self._job.done:
            self._job.cancel()
        self._token += 1
        token = self._token
        self.preview_before = self.preview_after = None
        self.preview_label.setText(tr("Calculating…"))
        groups, tol = list(self.groups), self.tolerance
        manifolds, lock = self.manifolds, self._manifold_lock

        def task(report):
            report(0.0)
            before = after = 0
            for g in groups:
                # the manifold of a group is made once (seconds on a big model); every
                # step of the slider after that is only manifold's own simplify
                with lock:              # a quick second step waits instead of doing it again
                    if g.id not in manifolds:
                        try:
                            manifolds[g.id] = ops.closed_manifold(g)
                        except ValueError:
                            manifolds[g.id] = None
                m = manifolds[g.id]
                if m is None:
                    continue
                before += m.num_tri()
                after += m.simplify(tol).num_tri()
            return before, after

        def done(result):
            if token == self._token:
                self._show_estimate(*result)

        self._job = background.run(None, task, text=tr("Calculating…"), can_cancel=False,
                                   show_dialog=False, on_done=done)

    def _show_estimate(self, before, after):
        self.preview_before, self.preview_after = before, after
        if before <= 0:
            self.preview_label.setText(tr("Nothing to simplify."))
            return
        pct = round(100 * (before - after) / before)
        self.preview_label.setText(
            tr("{before} triangles -> {after} triangles ({pct} % fewer)")
            .format(before=before, after=after, pct=pct))


class PageDialog(QDialog):
    """A dialog with the pages on the left and their content on the right, like SketchUp."""
    SIZE = (640, 440)

    def __init__(self, win, title, pages, page=0):
        super().__init__(win)
        self.win = win
        self.page_titles = list(pages)
        self.setWindowTitle(title)
        self.resize(*self.SIZE)
        lay = QVBoxLayout(self)
        body = QHBoxLayout()
        self.pages_list = QListWidget()
        self.pages_list.setFixedWidth(140)
        for name in pages:
            QListWidgetItem(name, self.pages_list)
        self.stack = QStackedWidget()
        body.addWidget(self.pages_list)
        body.addWidget(self.stack, 1)
        lay.addLayout(body)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.pages_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.select_page(page)

    def add_page(self, widget):
        self.stack.addWidget(widget)
        return widget

    def select_page(self, page):
        if isinstance(page, str):
            if page not in self.page_titles:     # the English title, or one in another language
                english = [i18n.source_text(t) for t in self.page_titles]
                page = (tr(page) if tr(page) in self.page_titles
                        else self.page_titles[english.index(i18n.source_text(page))])
            page = self.page_titles.index(page)
        page = max(0, min(len(self.page_titles) - 1, int(page)))
        self.pages_list.setCurrentRow(page)
        self.stack.setCurrentIndex(page)

    def accept(self):
        self.apply()
        super().accept()

    def apply(self):
        """Subclasses store what is on their pages here."""


class ModelInfoDialog(PageDialog):
    """Window > Model Info: Units, Statistics and Print Bed (SketchUp 2017)."""

    PAGES = (tr("Units"), tr("Statistics"), tr("Print Bed"))
    SIZE = (660, 430)

    def __init__(self, win, page=0):
        super().__init__(win, tr("Model Info"), self.PAGES, page)
        self._build_units()
        self._build_statistics()
        self._build_bed()
        self.refresh_statistics()
        self.pages_list.currentRowChanged.connect(self._page_shown)
        self._page_shown(self.pages_list.currentRow())

    def _page_shown(self, row):
        if row == 1:
            self.refresh_statistics()

    def _build_units(self):
        page = QWidget()
        form = QFormLayout(page)
        snap = self.win.settings.length_snap
        box = QGroupBox(tr("Length"))
        inner = QFormLayout(box)
        self.snap_on = QCheckBox(tr("Enable length snapping"))
        self.snap_on.setChecked(snap > 0)
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.01, 100)
        self.snap.setDecimals(2)
        self.snap.setSingleStep(0.5)
        self.snap.setSuffix(" mm")
        self.snap.setValue(snap or 1.0)
        self.snap.setEnabled(snap > 0)
        self.snap_on.toggled.connect(self.snap.setEnabled)
        self.snap.setToolTip(tr("Lengths that do not snap to a point or edge are rounded to this, "
                                "like 'Length Snapping' in SketchUp."))
        inner.addRow(self.snap_on)
        inner.addRow(tr("Round to"), self.snap)
        self.precision = QComboBox()
        self.precision.addItems(["0", "0.0", "0.00"])
        self.precision.setCurrentIndex(min(self.win.settings.length_precision, 2))
        self.sample = QLabel()
        self.precision.currentIndexChanged.connect(self._show_sample)
        self._show_sample()
        inner.addRow(tr("Display precision"), self.precision)
        inner.addRow(tr("Sample"), self.sample)
        form.addRow(box)
        form.addRow(note(tr("The display precision is the number of decimals of the Measurements "
                            "input and of the dimensions in the drawing window, like 'Precision' "
                            "in SketchUp.")))
        form.addRow(note(tr("The print bed and the printer are on the Print Bed page.")))
        self.add_page(page)

    def _show_sample(self):
        self.sample.setText(format_length(123.456, self.precision.currentIndex()))

    def _build_statistics(self):
        page = QWidget()
        form = QFormLayout(page)
        self.stats = {}
        # keyed by the English label, shown translated
        for label in (tr_noop("Groups"), tr_noop("Loose faces"), tr_noop("Loose edges"),
                      tr_noop("Total volume")):
            value = QLabel()
            form.addRow(tr(label), value)
            self.stats[label] = value
        form.addRow(note(tr("The volume is the sum of all closed parts: every group and every "
                            "loose shape that forms a closed volume.")))
        self.add_page(page)

    def refresh_statistics(self):
        doc = self.win.doc
        self.stats["Groups"].setText(str(len(doc.groups)))
        self.stats["Loose faces"].setText(str(len(doc.root.F)))
        self.stats["Loose edges"].setText(str(len(doc.root.E)))
        volume = sum(p.manifold.volume() for p in ops.parts(doc) if p.manifold is not None)
        self.stats["Total volume"].setText(f"{volume / 1000:.3f} cm³")

    def _build_bed(self):
        page = QWidget()
        form = QFormLayout(page)
        self.printer = QComboBox()
        self.printer.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.printer.addItem(tr("Custom"), None)
        for name, size in PRINTERS:
            self.printer.addItem(f"{name}  ({size[0]}×{size[1]}×{size[2]})", size)
        self.bed_spins = []
        for label, v in zip((tr("Bed X"), tr("Bed Y"), tr("Height Z")), self.win.settings.bed_size):
            sp = QDoubleSpinBox()
            sp.setRange(10, 2000)
            sp.setSuffix(" mm")
            sp.setValue(v)
            self.bed_spins.append(sp)
            form.addRow(label, sp)

        def preset(i):
            size = self.printer.itemData(i)
            if size:
                for sp, v in zip(self.bed_spins, size):
                    sp.setValue(v)
        self.printer.currentIndexChanged.connect(preset)
        form.insertRow(0, tr("Printer"), self.printer)
        form.addRow(note(tr("The bed sets the grid of the drawing area and the limits of the "
                            "print check.")))
        box = QGroupBox(tr("Filament"))
        inner = QFormLayout(box)
        self.filament = QComboBox()
        for name, density in FILAMENT_MATERIALS:
            self.filament.addItem(f"{name} ({density} g/cm³)", (name, density))
        self.filament.addItem(tr("Custom"), ("Custom", None))
        self.filament_density = QDoubleSpinBox()
        self.filament_density.setRange(0.1, 20.0)
        self.filament_density.setDecimals(2)
        self.filament_density.setSuffix(" g/cm³")
        self.filament_diameter = QDoubleSpinBox()
        self.filament_diameter.setRange(0.5, 5.0)
        self.filament_diameter.setDecimals(2)
        self.filament_diameter.setSingleStep(0.05)
        self.filament_diameter.setSuffix(" mm")
        self.filament_price = QDoubleSpinBox()
        self.filament_price.setRange(0.0, 100000.0)
        self.filament_price.setDecimals(2)
        self.filament_price.setSuffix(" " + tr("per kg"))
        self.print_shell = QDoubleSpinBox()
        self.print_shell.setRange(0.0, 5.0)
        self.print_shell.setDecimals(2)
        self.print_shell.setSingleStep(0.2)
        self.print_shell.setSuffix(" mm")
        self.print_shell.setToolTip(tr("Thickness of the walls and of the top and bottom "
                                       "of a print, as your slicer prints them."))
        self.print_infill = QSpinBox()
        self.print_infill.setRange(0, 100)
        self.print_infill.setSuffix(" %")
        self.print_infill.setToolTip(tr("How much of the inside of a print is filled; the "
                                        "rest is empty space."))
        s = self.win.settings
        self.filament.setCurrentIndex(max(
            (i for i in range(self.filament.count())
             if self.filament.itemData(i)[0] == s.filament_material),
            default=self.filament.count() - 1))
        self.filament_density.setValue(s.filament_density)
        self.filament_diameter.setValue(s.filament_diameter)
        self.filament_price.setValue(s.filament_price)
        self.print_shell.setValue(s.print_shell)
        self.print_infill.setValue(s.print_infill)

        def pick_material(i):
            density = self.filament.itemData(i)[1]
            self.filament_density.setEnabled(density is None)
            if density is not None:
                self.filament_density.setValue(density)
        self.filament.currentIndexChanged.connect(pick_material)
        pick_material(self.filament.currentIndex())
        inner.addRow(tr("Material"), self.filament)
        inner.addRow(tr("Density"), self.filament_density)
        inner.addRow(tr("Filament diameter"), self.filament_diameter)
        inner.addRow(tr("Price"), self.filament_price)
        inner.addRow(tr("Walls and top/bottom"), self.print_shell)
        inner.addRow(tr("Infill"), self.print_infill)
        form.addRow(box)
        form.addRow(note(tr("Entity Info shows the weight and cost of a selected part with "
                            "these walls and this infill; a print that is solid uses the most.")))
        self.add_page(page)

    def apply(self):
        s = self.win.settings
        s.length_snap = self.snap.value() if self.snap_on.isChecked() else 0.0
        s.length_precision = self.precision.currentIndex()
        s.bed_size = tuple(sp.value() for sp in self.bed_spins)
        name, density = self.filament.currentData()
        s.filament_material = name
        s.filament_density = density if density is not None else self.filament_density.value()
        s.filament_diameter = self.filament_diameter.value()
        s.filament_price = self.filament_price.value()
        s.print_shell = self.print_shell.value()
        s.print_infill = self.print_infill.value()
        s.save()
        set_length_precision(s.length_precision)
        self.win.view.update()


class PreferencesDialog(PageDialog):
    """Window > Preferences: General, Display, Drawing, 3D Mouse and Shortcuts (SketchUp 2017)."""

    PAGES = (tr("General"), tr("Display"), tr("Drawing"), tr("3D Mouse"), tr("Shortcuts"))
    SIZE = (680, 470)
    SPEED_STEPS = 20                       # slider 1..40 = speed 0.05..2 times

    def __init__(self, win, page=0):
        super().__init__(win, tr("Preferences"), self.PAGES, page)
        self._build_general()
        self._build_looks()
        self._build_drawing()
        self._build_spacemouse()
        self.shortcuts = ShortcutsPanel(win)
        self.add_page(self.shortcuts)

    def _build_general(self):
        page = QWidget()
        form = QFormLayout(page)
        self.segments = QSpinBox()
        self.segments.setRange(8, 512)
        self.segments.setValue(self.win.settings.circle_segments)
        self.segments.setToolTip(tr("Number of segments of a circle, arc or polygon. While "
                                    "drawing, type for example 32s to change it for that shape."))
        form.addRow(tr("Circle segments"), self.segments)
        form.addRow(note(tr("The more segments, the rounder a circle looks and the longer "
                            "drawing and exporting take.")))
        self.updates = QCheckBox(tr("Check for updates at startup"))
        self.updates.setChecked(self.win.settings.check_updates)
        self.updates.setToolTip(tr("Once a day Volura checks on GitHub whether there is a newer "
                                   "version."))
        form.addRow(self.updates)
        self.ask_simplify = QCheckBox(tr("Ask to simplify a large import"))
        self.ask_simplify.setChecked(self.win.settings.ask_simplify_import)
        self.ask_simplify.setToolTip(tr("After importing a model with many triangles, ask "
                                        "whether to simplify it to keep Volura fast."))
        form.addRow(self.ask_simplify)
        self.language = QComboBox()
        self.language.addItem(tr("System"), i18n.SYSTEM)
        for code, name in i18n.LANGUAGES:       # language names stay in their own language
            self.language.addItem(name, code)
        self.language.setCurrentIndex(max(0, self.language.findData(i18n.saved_choice())))
        self.language.setToolTip(tr("System uses the language of Windows (English if Volura "
                                    "does not have it)."))
        form.addRow(tr("Language"), self.language)
        form.addRow(note(tr("Takes effect after restarting Volura.")))
        self.scheme = QComboBox()
        self.scheme.addItem(tr("SketchUp (default)"), keyschemes.SKETCHUP)
        self.scheme.addItem(tr("Fusion"), keyschemes.FUSION_SCHEME)
        self.scheme.setCurrentIndex(max(0, self.scheme.findData(self.win.settings.key_scheme)))
        self.scheme.setToolTip(tr("The keys of the commands and what the middle mouse button "
                                  "does in the 3D view."))
        form.addRow(tr("Keyboard and mouse:"), self.scheme)
        form.addRow(note(tr("Fusion has the keys and the middle mouse button of Autodesk "
                            "Fusion; the shortcuts you changed yourself stay as they are.")))
        self.add_page(page)

    def _build_looks(self):
        page = QWidget()
        form = QFormLayout(page)
        self.theme = QComboBox()
        for key, label in theme.CHOICES:
            self.theme.addItem(label, key)
        self.theme.setCurrentIndex(max(0, self.theme.findData(self.win.settings.theme)))
        self.theme.setToolTip(tr("System follows the Windows setting (light or dark)."))
        form.addRow(tr("Theme"), self.theme)
        form.addRow(note(tr("In the dark theme the sheet on the 2D worksheet stays light, so the "
                            "layer colors stay readable, and the 3D window gets a muted gray "
                            "background.")))
        self.add_page(page)

    def _build_spacemouse(self):
        page = QWidget()
        form = QFormLayout(page)
        s = self.win.settings
        self.sm_on = QCheckBox(tr("Use a 3Dconnexion 3D mouse"))
        self.sm_on.setChecked(s.spacemouse)
        form.addRow(self.sm_on)
        self.sm_speed = QSlider(Qt.Horizontal)
        self.sm_speed.setRange(1, 2 * self.SPEED_STEPS)
        self.sm_speed.setValue(round(s.spacemouse_speed * self.SPEED_STEPS))
        self.sm_speed.setToolTip(tr("How fast the model moves with the cap."))
        form.addRow(tr("Speed"), self.sm_speed)
        form.addRow(tr("Connection"), QLabel(self.win.spacemouse_status()))
        self.sm_live = QLabel(tr("move the cap to test"))
        form.addRow(tr("Motion"), self.sm_live)
        self.win.spacemouse.motion.connect(self._show_motion)
        self.win.spacemouse.command.connect(self._show_command)
        self.finished.connect(self._stop_live)
        form.addRow(note(tr("Works with the SpaceMouse (Wireless, Pro, Compact, Enterprise) and "
                            "older devices such as SpaceNavigator, SpaceExplorer and SpacePilot. "
                            "With 3DxWare installed you set up buttons and axes in the Volura "
                            "profile in 3DxWare; without a driver Volura reads the device itself.\n"
                            "Sliding the cap pans the model, pulling or pushing zooms, twisting and "
                            "tilting rotates the model. In 2D the cap pans and zooms the "
                            "worksheet.")))
        self.add_page(page)

    def _show_motion(self, m):
        self.sm_live.setText(tr("pan {tx:+.2f} {ty:+.2f} {tz:+.2f}   rotate {rx:+.2f} {ry:+.2f} "
                                "{rz:+.2f}").format(tx=m.tx, ty=m.ty, tz=m.tz,
                                                    rx=m.rx, ry=m.ry, rz=m.rz))

    def _show_command(self, name):
        self.sm_live.setText(tr("button: {name}").format(name=name))

    def _stop_live(self, *args):
        for signal, slot in ((self.win.spacemouse.motion, self._show_motion),
                             (self.win.spacemouse.command, self._show_command)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _build_drawing(self):
        page = QWidget()
        form = QFormLayout(page)
        self.split_lines = QCheckBox(tr("Split lines automatically"))
        self.split_lines.setChecked(True)
        self.split_lines.setEnabled(False)
        self.split_lines.setToolTip(tr("As in SketchUp: edges that cross always split each other, "
                                       "and a line across a face splits that face."))
        form.addRow(self.split_lines)
        form.addRow(note(tr("Volura always does this, like SketchUp; it is only shown here.")))
        self.overhang_limit = QSpinBox()
        self.overhang_limit.setRange(0, 89)
        self.overhang_limit.setSuffix("°")
        self.overhang_limit.setValue(self.win.settings.overhang_limit)
        self.overhang_limit.setToolTip(tr("Faces that hang over more steeply than this from the "
                                          "vertical need support; View > Face Style > Overhangs "
                                          "marks them."))
        form.addRow(tr("Overhang limit"), self.overhang_limit)
        self.add_page(page)

    def apply(self):
        s = self.win.settings
        s.circle_segments = self.segments.value()
        s.check_updates = self.updates.isChecked()
        s.ask_simplify_import = self.ask_simplify.isChecked()
        s.overhang_limit = self.overhang_limit.value()
        s.spacemouse_speed = self.sm_speed.value() / self.SPEED_STEPS
        sm_changed = s.spacemouse != self.sm_on.isChecked()
        s.spacemouse = self.sm_on.isChecked()
        theme_changed = s.theme != self.theme.currentData()
        s.theme = self.theme.currentData()
        scheme = keyschemes.scheme_of(str(self.scheme.currentData() or keyschemes.SKETCHUP))
        scheme_changed = s.key_scheme != scheme
        # the language is read once at startup (i18n.saved_choice), so it applies after a restart
        s.q.setValue(i18n.SETTINGS_KEY, str(self.language.currentData() or i18n.SYSTEM))
        s.save()
        # also when the choice did not change: the windows can still show another theme
        if theme_changed or theme.wants_dark(s.theme) != theme.is_dark():
            self.win.apply_theme()
        if sm_changed:
            self.win.start_spacemouse()
        if scheme_changed:
            self.win.set_key_scheme(scheme)
        self.win.view.tools["circle"].segments = s.circle_segments
        self.win.view.update()
        self.win.view.update_overhang_status()


class ShortcutsPanel(QWidget):
    """Preferences > Shortcuts: filter, the commands per menu, and add/remove a shortcut."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.items = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("Filter")))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("Type to search"))
        self.filter.textChanged.connect(self._filter)
        top.addWidget(self.filter, 1)
        lay.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([tr("Command"), tr("Shortcut")])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setColumnWidth(1, 150)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Add shortcut")))
        self.key_edit = QKeySequenceEdit()
        row.addWidget(self.key_edit, 1)
        self.add_button = QPushButton("+")
        self.add_button.setToolTip(tr("Puts the typed shortcut on the chosen command"))
        self.add_button.clicked.connect(self.add_shortcut)
        self.remove_button = QPushButton("−")
        self.remove_button.setToolTip(tr("Removes the shortcut from the chosen command"))
        self.remove_button.clicked.connect(self.remove_shortcut)
        self.restore_button = QPushButton(tr("Reset All"))
        self.restore_button.setToolTip(tr("Puts all shortcuts back to their defaults"))
        self.restore_button.clicked.connect(self.restore_all)
        for b in (self.add_button, self.remove_button, self.restore_button):
            b.setMinimumWidth(40)
            row.addWidget(b)
        lay.addLayout(row)
        self.reload()

    def reload(self, keep=None):
        """Every command of the menus, grouped per menu, with the shortcut it has now.

        The paths are the English ones (the settings keys); the tree shows the menu texts."""
        keep = keep or self.selected_path()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.items = {}
        self.labels = {}
        groups = {}
        for path, action in self.win.command_paths().items():
            label = self.win.command_label(path)
            self.labels[path] = label
            menu, _, name = label.partition("/")
            groups.setdefault(menu, []).append((name, path, action))
        for menu, rows in groups.items():
            head = QTreeWidgetItem([menu, ""])
            head.setFlags(head.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(head)
            for name, path, action in sorted(rows, key=lambda r: r[0].lower()):
                it = QTreeWidgetItem([name, self.win.shortcut_text(action)])
                it.setData(0, Qt.UserRole, path)
                head.addChild(it)
                self.items[path] = it
            head.setExpanded(True)
        self.tree.blockSignals(False)
        if keep in self.items:
            self.tree.setCurrentItem(self.items[keep])

    def selected_path(self):
        it = self.tree.currentItem()
        if it is None:
            return None
        return it.data(0, Qt.UserRole) or None

    def select_command(self, path):
        it = self.items.get(path)
        if it is None:
            return False
        self.tree.setCurrentItem(it)
        return True

    def add_shortcut(self, checked=False):
        """The "+" button: the typed shortcut goes to the chosen command."""
        path = self.selected_path()
        seq = QKeySequence(self.key_edit.keySequence())
        if not path or seq.isEmpty() or not self.win.assign_shortcut(path, seq):
            return False
        self.key_edit.clear()
        self._show(path)
        return True

    def remove_shortcut(self, checked=False):
        """The "−" button: the typed shortcut, or all shortcuts, off the chosen command."""
        path = self.selected_path()
        if not path:
            return False
        seq = QKeySequence(self.key_edit.keySequence())
        if not self.win.remove_shortcut(path, None if seq.isEmpty() else seq):
            return False
        self._show(path)
        return True

    def restore_all(self, checked=False):
        """Reset All: back to the defaults of the scheme that is on."""
        self.win.restore_shortcuts()
        self.reload()

    def _show(self, path):
        action = self.win.command_paths().get(path)
        it = self.items.get(path)
        if action is not None and it is not None:
            it.setText(1, self.win.shortcut_text(action))

    def _filter(self, text):
        text = text.strip().lower()
        for path, it in self.items.items():
            label = self.labels.get(path, path)
            it.setHidden(bool(text) and text not in label.lower() and text not in path.lower())
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            found = any(not head.child(j).isHidden() for j in range(head.childCount()))
            head.setHidden(bool(text) and not found)
