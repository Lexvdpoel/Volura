"""The user interface of the main window: the shortcuts, the theme, the 3D mouse, the
status bar, the edit commands, hide and lock, the view and the right-click menu of Volura.

Step 3 of splitting `volura/app.py`: the methods that arrange the interface moved here
verbatim, into a plain mixin that `MainWindow` inherits. The keys of the stored shortcuts,
the commands that changed of menu and the title of the Entity Info panel live here too,
because these methods read them; the app imports them back.

The methods that ask something with a message box or an input dialog stay in
`volura/app.py`, together with the actions that call all of this; the window still owns
everything, this module only holds the code.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMenu

from . import background, i18n, ops, theme
from .i18n import tr, tr_noop, trn
from .model import LOCKED_MSG, ROOT
from .panels import swatch
from .settings import qsettings
from .tools import fmt_deg


SHORTCUTS_KEY = "shortcuts/"

# The commands that moved when the menus were arranged by mode (3D model or 2D worksheet):
# their old path, under which a shortcut of the user may still be stored, and the new one.
MOVED_COMMANDS = {
    'Edit/Copy to 2D Worksheet': 'Tools/Copy to 2D Worksheet',
    'Edit/Intersect Faces/With Model': 'Tools/Intersect Faces/With Model',
    'Edit/Intersect Faces/With Selection': 'Tools/Intersect Faces/With Selection',
    'Edit/Job Estimate': 'Laser and Mill/Job Estimate',
    'Edit/To Laser or Mill': 'Tools/To Laser or Mill',
    'File/Print Check': '3D Printing/Print Check',
    'File/Printer Setup': '3D Printing/Printer Setup',
    'Other/Align Bottom': 'Tools/Align/Align Bottom',
    'Other/Align Center': 'Tools/Align/Align Center',
    'Other/Align Left': 'Tools/Align/Align Left',
    'Other/Align Right': 'Tools/Align/Align Right',
    'Other/Align Top': 'Tools/Align/Align Top',
    'Other/Arc': 'Draw/Arc',
    'Other/Arc Around Center': 'Draw/Arc Around Center',
    'Other/Center on Bed': '3D Printing/Center on Bed',
    'Other/Circle': 'Draw/Circle',
    'Other/Distribute Horizontally': 'Tools/Distribute/Distribute Horizontally',
    'Other/Distribute Vertically': 'Tools/Distribute/Distribute Vertically',
    'Other/Drop to Bed': '3D Printing/Drop to Bed',
    'Other/Duplicate in Row or Grid': 'Tools/Duplicate in Row or Grid',
    'Other/Duplicate in a Circle': 'Tools/Duplicate in a Circle',
    'Other/Ellipse': 'Draw/Ellipse',
    'Other/Flip Horizontally': 'Tools/Flip Horizontally',
    'Other/Flip Vertically': 'Tools/Flip Vertically',
    'Other/Intersect Outlines': 'Tools/Intersect Outlines',
    'Other/Join Lines': 'Tools/Join Lines',
    'Other/Line': 'Draw/Line',
    'Other/Polygon': 'Draw/Polygon',
    'Other/Rectangle': 'Draw/Rectangle',
    'Other/Rounded Corners on Offset': 'Tools/Rounded Corners on Offset',
    'Other/Rounded Rectangle': 'Draw/Rounded Rectangle',
    'Other/Snap to Grid': 'View/Snap to Grid',
    'Other/Split into Segments': 'Tools/Split into Segments',
    'Other/Subtract Outlines': 'Tools/Subtract Outlines',
    'Other/Text': 'Draw/Text',
    'Other/To Layer': 'Tools/To Layer',
    'Other/Turn': 'Tools/Turn',
    'Other/Union Outlines': 'Tools/Union Outlines',
    'Other/Worksheet Layers': 'View/Worksheet Layers',
    'Tools/Arrange on Bed': '3D Printing/Arrange on Bed',
    'Tools/Print Tools/Cable Clip': 'Draw/Parts/Cable Clip',
    'Tools/Print Tools/Chamfer': 'Tools/Chamfer',
    'Tools/Print Tools/Clearance Test': '3D Printing/Clearance Test',
    'Tools/Print Tools/Cut': '3D Printing/Cut',
    'Tools/Print Tools/Cut and Join': '3D Printing/Cut and Join',
    'Tools/Print Tools/Elephant Foot Chamfer': '3D Printing/Elephant Foot Chamfer',
    'Tools/Print Tools/External Thread on Cylinder': 'Tools/Screw Threads/External Thread on Cylinder',
    'Tools/Print Tools/Fillet': 'Tools/Fillet',
    'Tools/Print Tools/Internal Thread in Hole': 'Tools/Screw Threads/Internal Thread in Hole',
    'Tools/Print Tools/Knob': 'Draw/Parts/Knob',
    'Tools/Print Tools/Label Plate': 'Draw/Parts/Label Plate',
    'Tools/Print Tools/Mouse Ears': '3D Printing/Mouse Ears',
    'Tools/Print Tools/New Bolt': 'Tools/Screw Threads/New Bolt',
    'Tools/Print Tools/New Nut or Sleeve': 'Tools/Screw Threads/New Nut or Sleeve',
    'Tools/Print Tools/Orient for Printing': '3D Printing/Orient for Printing',
    'Tools/Print Tools/Overhang Test': '3D Printing/Overhang Test',
    'Tools/Print Tools/Project Box': 'Draw/Parts/Project Box',
    'Tools/Print Tools/Screw Hole': 'Tools/Screw Hole',
    'Tools/Print Tools/Spur Gear': 'Draw/Parts/Spur Gear',
    'Tools/Print Tools/Standoff': 'Draw/Parts/Standoff',
    'Tools/Print Tools/Thread': 'Tools/Screw Threads/Thread',
    'View/Auto Arrange': 'Laser and Mill/Auto Arrange',
    'View/Box with Finger Joints': 'Draw/Box with Finger Joints',
    'View/Dogbone Corners': 'Laser and Mill/Dogbone Corners',
    'View/Fit Test': 'Laser and Mill/Fit Test',
    'View/Holding Tabs': 'Laser and Mill/Holding Tabs',
    'View/Living Hinge': 'Tools/Living Hinge',
    'View/Parts to 3D Model': 'Tools/Parts to 3D Model',
    'View/Round Corners': 'Tools/Round Corners',
    'View/Sheet': 'Laser and Mill/Sheet',
    'View/Spur Gear': 'Draw/Spur Gear',
    'View/Vent Pattern': 'Tools/Vent Pattern',
}


# the group of the commands that are only in the right-click menu; English, as it is part
# of the settings keys (shown translated)
OTHER_COMMANDS = tr_noop("Other")
SOURCE_TEXT = "source_text"   # QAction property: the English text, when tr() can't tell


# The Entity Info panel of the Default Tray; the Tray knows it by its translated title.
PANEL_INFO = tr_noop("Entity Info")


def clean_text(text):
    """Menu text without the & accelerator and without a trailing ellipsis."""
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def source_text_of(action):
    """The English text of an action or menu (with & and …), for language-independent keys."""
    text = action.property(SOURCE_TEXT)
    if text:
        return str(text)
    text = action.text()
    return text if i18n.language() == "en" else i18n.source_text(text)


_TABLES = None


def translation_tables():
    """The translation tables of the other languages, to recognise their old settings keys."""
    global _TABLES
    if _TABLES is None:
        _TABLES = [t for t in (i18n.table(code) for code, _name in i18n.LANGUAGES if code != "en")
                   if t]
    return _TABLES


class UiMixin:
    """The user interface methods of `MainWindow`; the window itself lives in the app."""

    # ------------------------------------------------------------ shortcuts
    def command_paths(self):
        """{'File/New': QAction} for every command in the menus; the first menu wins.

        The paths are made of the English menu texts, so the shortcuts stored under them
        (QSettings "shortcuts/<path>") work in every language. `command_label` gives the
        path as the menus show it; `_legacy_paths` maps the path the same command had in
        another language (the Dutch Volura 0.1.0 stored those) to the English one."""
        out, labels, legacy = {}, {}, {}
        tables = translation_tables()

        def segments(a):
            """The path segment of a menu or command: English, shown, per other language."""
            src = source_text_of(a)
            return (clean_text(src), clean_text(a.text()),
                    *(clean_text(t.get(src) or t.get(clean_text(src)) or src) for t in tables))

        def add(paths, a, overwrite=False):
            if paths[0] in out and not overwrite:
                return
            out[paths[0]] = a
            labels[paths[0]] = paths[1]
            for old in paths[2:]:
                legacy.setdefault(old, paths[0])

        def walk(menu, prefixes):
            for a in menu.actions():
                if a.isSeparator():
                    continue
                sub = a.menu()
                if sub is not None:
                    if sub is not getattr(self, "recent_menu", None):
                        walk(sub, tuple(f"{p}/{s}" for p, s in zip(prefixes, segments(a))))
                    continue
                segs = segments(a)
                if segs[0]:
                    add(tuple(f"{p}/{s}" for p, s in zip(prefixes, segs)), a)

        for a in self.menuBar().actions():
            menu = a.menu()
            if menu is not None:
                walk(menu, segments(a))
        seen = {id(a) for a in out.values()}
        other = (OTHER_COMMANDS, tr(OTHER_COMMANDS),
                 *(t.get(OTHER_COMMANDS, OTHER_COMMANDS) for t in tables))
        for a in self.actions():          # the commands that only live in the right-click menu
            segs = segments(a)
            if segs[0] and id(a) not in seen:
                add(tuple(f"{p}/{s}" for p, s in zip(other, segs)), a, overwrite=True)
        for old_path, new_path in MOVED_COMMANDS.items():
            if new_path in out:
                legacy.setdefault(old_path, new_path)
        self._command_labels = labels
        self._legacy_paths = legacy
        return out

    def command_label(self, path):
        """The path of a command as the menus show it (in the current language)."""
        return getattr(self, "_command_labels", {}).get(path, path)

    def path_of(self, action):
        """The menu path of an action, the key its shortcuts are stored under."""
        for path, a in self.command_paths().items():
            if a is action:
                return path
        return None

    @staticmethod
    def shortcut_text(action):
        """The shortcuts of an action as shown in the list, e.g. 'Ctrl+Shift+E'."""
        return ", ".join(s.toString(QKeySequence.NativeText) for s in action.shortcuts())

    @staticmethod
    def sequence_texts(action):
        return [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]

    def action_using(self, text, skip=None):
        """The command that now has this shortcut, if any."""
        for a in self.command_paths().values():
            if a is not skip and text in self.sequence_texts(a):
                return a
        return None

    @staticmethod
    def read_shortcut_value(value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in value if str(v).strip()]

    def load_shortcuts(self):
        """Remember the defaults, then put the shortcuts from QSettings on the actions."""
        paths = self.command_paths()
        self._default_shortcuts = {p: self.sequence_texts(a) for p, a in paths.items()}
        taken = {}
        for a in paths.values():
            for text in self.sequence_texts(a):
                taken.setdefault(text, a)
        stored = {}
        q = qsettings()
        keys = [k for k in q.allKeys() if k.startswith(SHORTCUTS_KEY)]
        for key in keys:
            path = key[len(SHORTCUTS_KEY):]
            if path in paths:
                stored[path] = self.read_shortcut_value(q.value(key))
        legacy = getattr(self, "_legacy_paths", {})
        for key in keys:                      # paths in another language, e.g. 'Bestand/Nieuw'
            path = key[len(SHORTCUTS_KEY):]
            new = legacy.get(path)
            if path in paths or new is None:
                continue
            value = self.read_shortcut_value(q.value(key))
            q.remove(key)                     # move it to the English path for good
            if new not in stored:
                stored[new] = value
                q.setValue(SHORTCUTS_KEY + new, value)
        for path in stored:                   # first let go of what those commands use now
            a = paths[path]
            for text in self.sequence_texts(a):
                if taken.get(text) is a:
                    del taken[text]
            a.setShortcuts([])
        for path, texts in stored.items():    # then hand out the stored ones
            a = paths[path]
            keep = [t for t in texts if taken.get(t) in (None, a)]
            for t in keep:
                taken[t] = a
            a.setShortcuts([QKeySequence(t) for t in keep])

    def store_shortcuts(self, action, texts):
        """Keep the user's shortcut list in QSettings; a default list is not stored."""
        path = self.path_of(action)
        if path is None:
            return
        q = qsettings()
        if texts == list(getattr(self, "_default_shortcuts", {}).get(path, [])):
            q.remove(SHORTCUTS_KEY + path)
        else:
            q.setValue(SHORTCUTS_KEY + path, texts)
        q.sync()

    def change_shortcuts(self, action, texts):
        action.setShortcuts([QKeySequence(t) for t in texts])
        self.store_shortcuts(action, texts)

    def remove_shortcut(self, path, sequence=None):
        """Preferences > Shortcuts > '−': this shortcut, or all of them, off the command."""
        action = self.command_paths().get(path)
        if action is None:
            return False
        current = self.sequence_texts(action)
        if sequence is None:
            if not current:
                return False
            texts = []
        else:
            text = QKeySequence(sequence).toString(QKeySequence.PortableText)
            if text not in current:
                return False
            texts = [t for t in current if t != text]
        self.change_shortcuts(action, texts)
        return True

    def restore_shortcuts(self):
        """Preferences > Shortcuts > Reset All: back to the shortcuts Volura ships with."""
        q = qsettings()
        for key in q.allKeys():
            if key.startswith(SHORTCUTS_KEY):
                q.remove(key)
        q.sync()
        defaults = getattr(self, "_default_shortcuts", {})
        for path, a in self.command_paths().items():
            a.setShortcuts([QKeySequence(t) for t in defaults.get(path, [])])

    # ------------------------------------------------------------ theme
    def apply_theme(self):
        """Preferences > Display > Theme: the new theme everywhere at once."""
        dark = theme.apply(self.settings.theme)
        self.tray.apply_theme()
        self.info.refresh()
        for w in (self.view, self.sheet_view, self.mode_button):
            w.update()
        return dark

    def _system_scheme_changed(self, scheme=None):
        """Windows switches between light and dark: follow it when the theme is System."""
        if self.settings.theme == theme.SYSTEM and theme.wants_dark(theme.SYSTEM) != theme.is_dark():
            self.apply_theme()

    # ------------------------------------------------------------ 3D mouse
    def spacemouse_status(self):
        if not self.settings.spacemouse:
            return tr("off")
        if self.spacemouse.name:
            return tr("connected through {name}").format(name=self.spacemouse.name)
        return tr("no 3D mouse found")

    def on_spacemouse_motion(self, m):
        if self._busy or not self.isActiveWindow():
            return
        self.active_view().spacemouse_move(m, self.settings.spacemouse_speed)

    def on_spacemouse_command(self, name):
        """A button of the 3D mouse: fit to view, or a standard view in 3D."""
        if self._busy:
            return
        if name == "fit":
            self.zoom_extents()
        elif self.mode == "3d":
            self.view.set_view(name)

    # ------------------------------------------------------------ status
    def set_tool(self, name):
        self.view.set_tool(name)
        if name == "text":
            QTimer.singleShot(0, self.ask_text)

    def on_tool_changed(self, name):
        for n, a in self.tool_actions.items():
            a.setChecked(n == name)
        icons = getattr(self, "_toolbar_tool_icons", {})
        for act, btn in getattr(self, "_dropdown_buttons", {}).items():
            ic = icons.get(act)
            if ic is not None:
                btn.setIcon(ic)

    def on_sheet_tool_changed(self, name):
        for n, a in self.sheet_tool_actions.items():
            a.setChecked(n == name)

    def on_vcb_changed(self, label, value):
        self.vcb_label.setText(label)
        if not self.vcb.hasFocus():
            self.vcb.setText(value)

    def on_vcb_return(self):
        text = self.vcb.text()
        view = self.sheet_view if self.mode == "2d" else self.view
        view.setFocus()
        view.submit_vcb(text)

    def on_doc_changed(self):
        self.update_title()
        self.materials.refresh()
        self.outliner.refresh()
        self.info.refresh()
        if hasattr(self, "layers_panel"):
            self.layers_panel.refresh()
        if self.view.problem_items:
            self.print_panel.clear_marks()      # the marked geometry may be gone now

    def on_selection_changed(self):
        self.outliner.refresh()
        self.info.refresh()

    def update_title(self):
        name = os.path.basename(self.doc.path) if self.doc.path else tr("Untitled")
        star = "*" if self.doc.dirty else ""
        self.setWindowTitle(f"{name}{star} - Volura")

    def show_entity_info(self):
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel(tr(PANEL_INFO))

    def select_color(self, color):
        self.settings.current_color = color
        if not any(c.lower() == color.lower() for _, c in self.doc.materials):
            self.doc.checkpoint()
            self.doc.materials.append([tr("Color {n}").format(n=len(self.doc.materials) + 1), color])
            self.doc.notify()
        self.materials.refresh()
        self.view.set_status(tr("Current color: {color}").format(color=color))

    # ------------------------------------------------------------ edit
    def undo(self):
        self.view.tool.reset()
        self.sheet_view.tool.reset()
        self.view.preview_transform = None
        self.doc.undo()

    def redo(self):
        self.view.tool.reset()
        self.sheet_view.tool.reset()
        self.doc.redo()

    def warn_locked(self):
        """Edit > Lock: say in the status bar why nothing happened."""
        self.view.set_status(LOCKED_MSG)

    def unlocked(self, groups):
        """The groups that may be changed; warns about the locked ones."""
        locked = [g for g in groups if g.locked]
        if locked:
            self.warn_locked()
        return [g for g in groups if not g.locked]

    def delete(self):
        sel = list(self.doc.selection)
        if not sel:
            return
        locked = {g.id for g in self.doc.selected_groups() if g.locked}
        if locked:
            self.warn_locked()
            sel = [it for it in sel if not (it[0] == "g" and it[2] in locked)]
            if not sel:
                return
        self.doc.checkpoint()
        for k, c, i in sel:
            if k == "g":
                self.doc.remove_group(i)
            elif k == "guide":
                self.doc.remove_guides([i])
        by_ctx = {}
        for k, c, i in sel:
            if k in ("f", "e"):
                by_ctx.setdefault(c, ([], []))[0 if k == "f" else 1].append(i)
        for c, (fs, es) in by_ctx.items():
            ents = self.doc.ents(c)
            if ents is None:
                continue
            ents.erase_faces(fs)
            ents.erase_edges(es)
        self.doc.selection = []
        self.doc.notify()

    def select_all(self):
        ctx = self.doc.active
        ents = self.doc.active_ents()
        if self.settings.show_hidden:
            faces, edges = ents.F, ents.E
        else:
            faces, edges = ents.visible_faces(), ents.visible_edges()
        items = [("f", ctx, f) for f in faces] + [("e", ctx, e) for e in edges]
        if ctx == ROOT:
            items += [("g", g.id, g.id) for g in self.doc.visible_groups()]
        self.doc.set_selection(items)

    def select_none(self):
        self.doc.set_selection([])

    def erase_guides(self):
        """Edit > Delete Guides: every guide of the document, in one undo step."""
        if not self.doc.guides:
            return
        self.doc.checkpoint()
        self.doc.clear_guides()
        self.doc.notify()

    # ------------------------------------------------------------ hide / lock
    def _hidden_items(self, items):
        """The subset of the given selection items that is now hidden."""
        out = []
        for k, ctx, i in items:
            if k == "g":
                g = self.doc.group(i)
                if g is not None and not g.visible:
                    out.append(("g", ctx, i))
                continue
            ents = self.doc.ents(ctx)
            if ents is not None and i in (ents.hidden_f if k == "f" else ents.hidden_e):
                out.append((k, ctx, i))
        return out

    def _show_items(self, items):
        """Make the given items visible again (groups through `visible`, the rest through hidden_*)."""
        by_ctx = {}
        for k, ctx, i in items:
            if k == "g":
                g = self.doc.group(i)
                if g is not None:
                    g.visible = True
                continue
            by_ctx.setdefault(ctx, ([], []))[0 if k == "f" else 1].append(i)
        for ctx, (fs, es) in by_ctx.items():
            ents = self.doc.ents(ctx)
            if ents is not None:
                ents.unhide(fs, es)

    def hide_selected(self):
        """Edit > Hide: hide the selection and remember what it was."""
        items = [("g", g.id, g.id) for g in self.doc.selected_groups() if g.visible]
        items += [it for it in self.doc.selection if it[0] in ("f", "e")]
        if not items:
            return
        self.doc.checkpoint()
        by_ctx = {}
        for k, ctx, i in items:
            if k == "g":
                self.doc.group(i).visible = False
            else:
                by_ctx.setdefault(ctx, ([], []))[0 if k == "f" else 1].append(i)
        for ctx, (fs, es) in by_ctx.items():
            ents = self.doc.ents(ctx)
            if ents is not None:
                ents.hide(fs, es)
        self.doc.last_hidden = items
        self.doc.selection = []
        self.doc.notify()

    def unhide_selected(self):
        """Edit > Unhide > Selected (useful with Hidden Geometry on)."""
        items = self._hidden_items(self.doc.selection)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.notify()

    def unhide_last(self):
        """Edit > Unhide > Last: what Edit > Hide hid last."""
        items = self._hidden_items(self.doc.last_hidden)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.last_hidden = []
        self.doc.notify()

    def unhide_all(self):
        """Edit > Unhide > All: every hidden face, edge and group."""
        hidden = any(not g.visible for g in self.doc.groups) or \
            any(e.hidden_f or e.hidden_e for e in self.doc.all_ents())
        if not hidden:
            return
        self.doc.checkpoint()
        for g in self.doc.groups:
            g.visible = True
        for ents in self.doc.all_ents():
            ents.unhide_all()
        self.doc.last_hidden = []
        self.doc.notify()

    def lock_selected(self):
        """Edit > Lock: the selected groups can no longer be changed."""
        groups = [g for g in self.doc.selected_groups() if not g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = True
        self.doc.notify()

    def unlock_selected(self):
        """Edit > Unlock > Selected."""
        groups = [g for g in self.doc.selected_groups() if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def unlock_all(self):
        """Edit > Unlock > All."""
        groups = [g for g in self.doc.groups if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def copy(self):
        self.clipboard = list(self.doc.selection)
        self.hint.setText(trn("{n} item copied.", "{n} items copied.", len(self.clipboard))
                          .format(n=len(self.clipboard)))

    def cut(self):
        if not self.doc.selection:
            return
        self.copy()
        self.delete()

    def paste(self):
        if self._paste_svg_from_clipboard():
            return
        self._paste(np.array([10.0, 10.0, 0.0]), move_tool=True)

    def paste_at_location(self):
        self._paste(np.zeros(3), move_tool=False)

    def _paste(self, offset, move_tool):
        self.doc.validate_selection()
        items = [it for it in self.clipboard if it[0] == "g" and self.doc.group(it[1])] + \
                [it for it in self.clipboard if it[0] != "g" and self.doc.ents(it[1]) is not None
                 and (it[2] in self.doc.ents(it[1]).F or it[2] in self.doc.ents(it[1]).E)]
        if not items:
            return
        self.doc.checkpoint()
        new = ops.transform_selection(self.doc, np.eye(3), offset, copy=True, items=items)
        self.doc.set_selection(new)
        self.doc.notify()
        if move_tool:
            self.set_tool("move")

    def make_group(self):
        if not any(k in ("f", "e") for k, _, _ in self.doc.selection):
            return
        self.doc.checkpoint()
        g = ops.make_group(self.doc)
        self.doc.active = ROOT
        self.doc.set_selection([("g", g.id, g.id)] if g else [])
        self.doc.notify()

    def explode_group(self, wait=False):
        """Edit > Explode Group: the selected groups become loose geometry, one undo step.

        Merging a large solid into the loose geometry takes a while, so it runs in the
        background with a progress bar that can be cancelled; the document only changes when
        it is done.
        """
        groups = self.unlocked(self.doc.selected_groups())
        if not groups:
            return
        doc = self.doc
        ids = [g.id for g in groups]

        def work(report):
            return ops.exploded_root(doc, groups, report)

        def done(root):
            doc.checkpoint()
            doc.root.load_state(root.state())
            for gid in ids:
                doc.remove_group(gid)
            doc.selection = []
            doc.notify()

        def failed(exc):
            self.hint.setText(tr("Explode failed: {error}").format(error=exc))

        background.run(self, work, text=tr("Exploding…"), on_done=done, on_error=failed,
                       wait=wait)

    def edit_group(self):
        groups = self.doc.selected_groups()
        for g in groups:
            if self.doc.enter_group(g.id):
                return
        if groups:
            self.warn_locked()      # a locked group does not open

    def close_group(self):
        self.doc.exit_group()

    # ------------------------------------------------------------ view
    def on_cam_history(self, can_prev, can_next):
        self.a_prev_view.setEnabled(can_prev)
        self.a_next_view.setEnabled(can_next)

    def camera_prev(self, checked=False):
        self.view.prev_camera()

    def camera_next(self, checked=False):
        self.view.next_camera()

    def camera_fov(self, checked=False):
        """Camera > Field of View: the Zoom tool with the field of view in the Measurements box."""
        self.set_tool("zoom")
        self.vcb.setText(fmt_deg(self.view.cam.fov))
        self.vcb.setFocus()
        self.vcb.selectAll()

    def camera_zoom_window(self, checked=False):
        self.set_tool("zoomwindow")

    def apply_projection(self):
        self.view.set_projection(self.a_ortho.isChecked())

    def toggle_grid(self, on):
        self.settings.show_grid = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_axes(self, on):
        self.settings.show_axes = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_edges(self, on):
        self.settings.show_edges = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_xray(self, on):
        self.settings.xray = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_overhangs(self, on):
        """View > Face Style > Overhangs: mark the faces that need support when printed."""
        self.settings.show_overhangs = bool(on)
        self.settings.save()
        self.view.update()
        self.view.update_overhang_status()

    def toggle_back_edges(self, on):
        self.settings.back_edges = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_show_hidden(self, on):
        self.settings.show_hidden = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_guides(self, on):
        """View > Guides: show or hide the construction guides."""
        self.settings.show_guides = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_hide_rest(self, on):
        self.settings.hide_rest = bool(on)
        self.settings.save()
        self.view.update()

    def set_face_style(self, value):
        self.settings.face_style = value
        self.settings.save()
        self.view.update()

    # ------------------------------------------------------------ context menu
    def _selected_faces(self):
        """(ctx, fid) of every selected face, the clicked one first."""
        out = []
        for k, c, i in self.doc.selection:
            ents = self.doc.ents(c)
            if k == "f" and ents is not None and i in ents.F:
                out.append((c, i))
        return out

    def _clicked_face(self):
        faces = self._selected_faces()
        return faces[0] if faces else None

    def _lay_flat_face(self):
        """(group, fid) of the right-clicked face, when that face lies in a group."""
        face = getattr(self, "_context_face", None) or self._clicked_face()
        g = self.doc.group(face[0]) if face is not None else None
        return None if g is None else (g, face[1])

    def face_area(self, ents, fid):
        tri, _ = ents.face_triangles(fid)
        if not len(tri):
            return 0.0
        return float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                    axis=1).sum() / 2)

    def select_face_edges(self):
        items = []
        for c, i in self._selected_faces():
            ents = self.doc.ents(c)
            items += [("e", c, e) for e in ents.face_edges(ents.F[i])]
        self.doc.set_selection(items)

    def select_connected_faces(self):
        by_ctx = {}
        for c, i in self._selected_faces():
            by_ctx.setdefault(c, set()).add(i)
        items = []
        for c, fs in by_ctx.items():
            ents = self.doc.ents(c)
            nb = set(fs)
            for f in fs:
                for e in ents.face_edges(ents.F[f]):
                    nb |= ents.e2f.get(e, set())
            items += [("f", c, f) for f in sorted(nb)]
        self.doc.set_selection(items)

    def select_all_connected(self):
        by_ctx = {}
        for k, c, i in self.doc.selection:
            if k in ("f", "e"):
                by_ctx.setdefault(c, ([], []))[0 if k == "f" else 1].append(i)
        items = []
        for c, (fs, es) in by_ctx.items():
            ents = self.doc.ents(c)
            if ents is None:
                continue
            f2, e2 = ents.connected(fs, es)
            items += [("f", c, f) for f in sorted(f2)] + [("e", c, e) for e in sorted(e2)]
        self.doc.set_selection(items)

    def select_same_material(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        color = ents.F[fid].color
        self.doc.set_selection([("f", c, f) for f in sorted(ents.F) if ents.F[f].color == color])

    def align_view_to_face(self):
        """Align View: look straight at the face (view direction = -normal)."""
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        pts = np.concatenate(ents.face_points(fid))
        n = np.asarray(ents.F[fid].normal, dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        cam = self.view.cam
        cam.pitch = max(-89.99, min(89.99, math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))))
        cam.yaw = math.degrees(math.atan2(n[1], n[0]))
        cam.fit((pts.min(axis=0), pts.max(axis=0)))
        cam.target = pts.mean(axis=0)
        self.view.commit_camera()
        self.view.update()

    def lay_flat_on_face(self):
        """Right click on a face: turn its group until that face lies on the print bed."""
        face = self._lay_flat_face()
        if face is None:
            return
        g, fid = face
        if g.locked:
            self.warn_locked()
            return
        # a face that already points straight down on a part that already stands on the bed
        # does not move: no reason to turn it, and no reason to take an undo step either
        bb = g.bbox()
        if (fid in g.ents.F and np.allclose(np.asarray(g.ents.F[fid].normal, dtype=float),
                                            ops.DOWN_DIR, atol=1e-9)
                and bb is not None and abs(float(bb[0][2])) <= 1e-6):
            return
        self.doc.checkpoint()
        ops.lay_flat(self.doc, g.id, fid)
        self.doc.notify()

    def reverse_selected_faces(self):
        by_ctx = {}
        for c, i in self._selected_faces():
            by_ctx.setdefault(c, []).append(i)
        if not by_ctx:
            return
        self.doc.checkpoint()
        for c, fs in by_ctx.items():
            self.doc.ents(c).reverse_faces(fs)
        self.doc.notify()

    def orient_selected_faces(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        self.doc.checkpoint()
        self.doc.ents(c).orient_faces(fid)
        self.doc.notify()

    def zoom_selection(self):
        bb = ops.selection_bbox(self.doc)
        if bb is None:
            return
        self.view.cam.fit((bb[0], bb[1]))
        self.view.commit_camera()
        self.view.update()

    def build_context_menu(self):
        """The right-click menu for the current selection (SketchUp 2017 order)."""
        m = QMenu(self)
        sel = self.doc.selection
        if sel:
            m.addAction(tr(PANEL_INFO), self.show_entity_info)
            m.addAction(self.a_delete)
            m.addAction(self.a_hide)
            sel_groups = self.doc.selected_groups()
            if sel_groups:
                # SketchUp: right click > Lock / Unlock
                m.addAction(self.a_unlock if all(g.locked for g in sel_groups) else self.a_lock)
            if self._selected_faces():
                m.addSeparator()
                sm = m.addMenu(tr("Select"))
                sm.addAction(tr("Bounding Edges"), self.select_face_edges)
                sm.addAction(tr("Connected Faces"), self.select_connected_faces)
                sm.addAction(tr("All Connected"), self.select_all_connected)
                sm.addAction(tr("All with Same Material"), self.select_same_material)
                om = m.addMenu(tr("Area"))
                om.addAction(tr("Selection"), self.show_area_selection)
                om.addAction(tr("Material"), self.show_area_material)
                fm = m.addMenu(tr("Intersect Faces"))
                fm.addAction(self.a_cut_face_model)
                fm.addAction(self.a_cut_face_sel)
                m.addAction(tr("Align View"), self.align_view_to_face)
                m.addAction(tr("Reverse Faces"), self.reverse_selected_faces)
                m.addAction(tr("Orient Faces"), self.orient_selected_faces)
                m.addAction(tr("Zoom Selection"), self.zoom_selection)
            if self._selected_faces() or self.doc.selected_groups():
                m.addSeparator()
                self._add_laser_menu(m)
            m.addSeparator()
            if any(k in ("f", "g") for k, _, _ in sel):
                cm = m.addMenu(tr("Filament Color"))
                for i, (name, color) in enumerate(self.doc.materials, 1):
                    a = cm.addAction(swatch(color), f"{i}: {name}")
                    a.triggered.connect(lambda checked=False, c=color: self.paint_selection(c))
            if any(k in ("f", "e") for k, _, _ in sel):
                m.addAction(self.a_group)
            groups = self.doc.selected_groups()
            if groups:
                m.addAction(self.a_edit_group)
                m.addAction(self.a_explode)
                if len(groups) >= 2:
                    for a in (self.a_union, self.a_subtract, self.a_intersect):
                        m.addAction(a)
                if len(groups) == 2:
                    m.addAction(self.a_trim)
                m.addAction(self.a_shell)
                m.addAction(self.a_split)
            mm = m.addMenu(tr("Flip Along"))
            for i, naam in enumerate((tr("Red Axis"), tr("Green Axis"), tr("Blue Axis"))):
                mm.addAction(naam, lambda checked=False, ax=i: self.mirror(ax))
            if self._lay_flat_face() is not None:
                m.addAction(tr("Lay Flat on This Face"), self.lay_flat_on_face)
            m.addAction(self.a_drop)
            m.addAction(self.a_center)
            m.addAction(self.a_arrange)
        else:
            m.addAction(self.a_sel_all)
            m.addAction(tr("Unhide All"), self.unhide_all)
        link = self._context_link()
        if link is not None:
            m.addSeparator()
            m.addAction(tr("Go to Linked Outline"),
                        lambda checked=False, s=int(link.src): self.go_to_linked_outline(s))
        if self.doc.active != ROOT:
            m.addSeparator()
            m.addAction(self.a_close_group)
        m.addSeparator()
        m.addAction(self.a_extents)
        return m

    def show_context_menu(self, gpos):
        local = self.view.mapFromGlobal(gpos)
        pick = self.view.engine.pick_entity(local.x(), local.y())
        from .tools import pick_item
        item = pick_item(pick) if pick[0] else None
        if item is not None and item not in self.doc.selection:
            if item[0] == "f":
                ents = self.doc.ents(item[1])
                self.doc.set_selection([item] + [("e", item[1], e) for e in ents.face_edges(ents.F[item[2]])])
            else:
                self.doc.set_selection([item])
        # the face under the mouse, also in a group: "Laser / Mill" takes that very face
        o, d = self.view.cam.ray(local.x(), local.y())
        fh = self.view.engine.pick_face(o, d) if pick[0] in ("face", "group") else None
        self._context_face = (fh[1], fh[2]) if fh else None
        try:
            self.build_context_menu().exec(gpos)
        finally:
            self._context_face = None

    def paint_selection(self, color):
        items = [it for it in self.doc.selection if it[0] in ("f", "g")]
        locked = [it for it in items if it[0] == "g" and getattr(self.doc.group(it[2]), "locked", False)]
        if locked:
            self.warn_locked()
            items = [it for it in items if it not in locked]
        if not items:
            return
        self.doc.checkpoint()
        for k, c, i in items:
            if k == "g":
                self.doc.group(i).color = color
            else:
                self.doc.ents(c).set_face_color(i, color)
        self.doc.notify()
