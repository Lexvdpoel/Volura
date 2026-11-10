"""The Volura document: loose geometry, groups, materials, guides, selection and undo.

Like SketchUp, the model is loose geometry (edges and faces) plus groups. A
group is a separate piece of geometry with its own color; for 3D printing every
group (and every closed shell of loose geometry) is one printable part.

Construction geometry (guide lines and guide points) lives next to it: it helps you
draw, but is never printed or exported.

Next to the 3D model every document has one 2D sheet (`Document.sheet`, a
`vector2d.Drawing`): vector work for the laser cutter and the mill. Working in 2D
never changes the 3D geometry, and undo, `dirty` and the file cover both.

Between the two sit the live links (`Document.links`, see `volura/linking.py`): one per
part that went from the model to the worksheet or came back from it, so a shape that
changes on one side is carried to the other when you switch between them. A vent pattern
that was cut into one of those outlines is remembered next to them (`Document.vents`), so
it is made again when the outline of its panel changes.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from .brep import Entities
from .geom import normalize
from .i18n import tr, tr_noop
from .linking import Link, Vent
from .vector2d import Drawing

# Names are translated when a new document gets them (see Document.reset); a saved file
# keeps the names it was saved with. Materials are matched by color, never by name.
DEFAULT_MATERIALS = [
    (tr_noop("White"), "#f2f2f2"),
    (tr_noop("Black"), "#2b2b2b"),
    (tr_noop("Red"), "#d23b3b"),
    (tr_noop("Blue"), "#2f6fd6"),
    (tr_noop("Yellow"), "#f2c230"),
    (tr_noop("Green"), "#3fa34d"),
    (tr_noop("Orange"), "#f08a24"),
    (tr_noop("Gray"), "#9a9a9a"),
]

ROOT = 0

LOCKED_MSG = tr("This group is locked.")


class Group:
    def __init__(self, gid, name, color, ents, visible=True, locked=False):
        self.id = gid
        self.name = name
        self.color = color
        self.ents = ents
        self.visible = visible
        self.locked = locked      # Edit > Lock: cannot be moved, painted or edited

    def bbox(self):
        return self.ents.bbox()


GUIDE_LENGTH = 100000.0     # mm each way: long enough for a guide to look infinite


class Guide:
    """A construction guide: an infinite line (point + direction) or a point."""

    def __init__(self, gid, point, direction=None):
        self.id = gid
        self.point = np.asarray(point, dtype=float).reshape(3).copy()
        d = None if direction is None else normalize(np.asarray(direction, dtype=float).reshape(3))
        self.direction = None if d is None or np.linalg.norm(d) < 1e-9 else d

    @property
    def is_line(self):
        return self.direction is not None

    def endpoints(self):
        """The two ends of the drawn segment, ±GUIDE_LENGTH from the point."""
        if not self.is_line:
            return self.point.copy(), self.point.copy()
        return (self.point - self.direction * GUIDE_LENGTH,
                self.point + self.direction * GUIDE_LENGTH)

    def state(self):
        return (self.id, self.point.tolist(), None if self.direction is None else self.direction.tolist())

    @staticmethod
    def from_state(s):
        return Guide(s[0], s[1], s[2])


class Document(QObject):
    changed = Signal()
    selection_changed = Signal()

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.root = Entities()
        self.groups: list[Group] = []
        self.guides: list[Guide] = []
        self.materials = [[tr(n), c] for n, c in DEFAULT_MATERIALS]
        self.active = ROOT
        self.next_gid = 1
        self.group_counter = 1
        self.next_guide_id = 1
        self.selection = []          # ("f"|"e", ctx, id) or ("g", gid, gid) or ("guide", ROOT, gid)
        self.last_hidden = []        # what Edit > Hide hid last
        self.undo_stack = []
        self.redo_stack = []
        self.path = None
        self.dirty = False
        # the 2D worksheet: one drawing next to the 3D model, with the default layers
        self.sheet = Drawing.new_sheet()
        # the live links between a group in the model and the outlines of one src on that sheet
        self.links: list = []
        # the vent patterns cut into an outline of that sheet, so they fit a new outline again
        self.vents: list = []

    # ------------------------------------------------------------ guides
    def guide(self, gid):
        for g in self.guides:
            if g.id == gid:
                return g
        return None

    def add_guide(self, point, direction=None):
        """A new guide line (with direction) or guide point (without)."""
        g = Guide(self.next_guide_id, point, direction)
        self.next_guide_id += 1
        self.guides.append(g)
        return g

    def remove_guides(self, gids):
        gids = set(gids)
        self.guides = [g for g in self.guides if g.id not in gids]

    def clear_guides(self):
        self.guides = []

    # ------------------------------------------------------------ contexts
    def ents(self, ctx):
        if ctx == ROOT:
            return self.root
        g = self.group(ctx)
        return g.ents if g else None

    def active_ents(self):
        return self.ents(self.active) or self.root

    def group(self, gid):
        for g in self.groups:
            if g.id == gid:
                return g
        return None

    def visible_groups(self):
        return [g for g in self.groups if g.visible]

    def contexts(self):
        """(ctx, entities, group or None) for everything that is drawn."""
        out = [(ROOT, self.root, None)]
        out += [(g.id, g.ents, g) for g in self.groups if g.visible]
        return out

    def all_ents(self):
        """The geometry of the model and of every group, hidden ones included."""
        return [self.root] + [g.ents for g in self.groups]

    def enter_group(self, gid):
        g = self.group(gid)
        if g is None:
            return False
        if g.locked:
            return False
        self.active = gid
        self.selection = []
        self.notify()
        return True

    def exit_group(self):
        if self.active != ROOT:
            gid = self.active
            self.active = ROOT
            self.selection = [("g", gid, gid)]
            self.notify()

    # ------------------------------------------------------------ undo
    def _state(self):
        return (self.root.state(),
                [(g.id, g.name, g.color, g.visible, g.locked, g.ents.state()) for g in self.groups],
                [g.state() for g in self.guides],
                [list(m) for m in self.materials], self.active, self.next_gid, self.group_counter,
                self.next_guide_id, self.sheet.copy(), [l.to_dict() for l in self.links],
                [v.to_dict() for v in self.vents])

    def _restore(self, st):
        (root, groups, guides, mats, self.active, self.next_gid, self.group_counter,
         self.next_guide_id, sheet, links, vents) = st
        self.root.load_state(root)
        old = {g.id: g for g in self.groups}        # undo keeps the Group objects you have
        self.groups = []
        for i, n, c, v, lk, s in groups:
            g = old.pop(i, None)
            if g is None:
                g = Group(i, n, c, Entities.from_state(s), v, lk)
            else:
                g.name, g.color, g.visible, g.locked = n, c, v, lk
                g.ents.load_state(s)
            self.groups.append(g)
        self.guides = [Guide.from_state(s) for s in guides]
        self.materials = [list(m) for m in mats]
        if self.active != ROOT and self.group(self.active) is None:
            self.active = ROOT
        self.selection = []
        self.links = [Link.from_dict(d) for d in links]
        self.vents = [Vent.from_dict(d) for d in vents]
        self.load_sheet(sheet)

    def load_sheet(self, sheet: "Drawing"):
        """Copy the contents of `sheet` onto this document's worksheet, without a new object."""
        src = sheet.copy()            # never share paths with an undo state
        self.sheet.paths = src.paths
        self.sheet.layers = src.layers
        self.sheet.sheet_w = src.sheet_w
        self.sheet.sheet_h = src.sheet_h
        self.sheet.sheet_thickness = src.sheet_thickness
        self.sheet.sheet_margin = src.sheet_margin
        self.sheet.sheet_gap = src.sheet_gap
        self.sheet.sheet_count = src.sheet_count
        self.sheet.warnings = src.warnings
        self.sheet.ensure_layer("0")

    def checkpoint(self):
        self.push_state(self._state())

    def push_state(self, st):
        """Undo: remember a state taken before a change, as checkpoint() does."""
        self.undo_stack.append(st)
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._state())
        self._restore(self.undo_stack.pop())
        self.dirty = True
        self.notify()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._state())
        self._restore(self.redo_stack.pop())
        self.dirty = True
        self.notify()
        return True

    def notify(self):
        self.validate_selection()
        self.changed.emit()
        self.selection_changed.emit()

    # ------------------------------------------------------------ selection
    def set_selection(self, items):
        self.selection = list(dict.fromkeys(items))
        self.validate_selection()
        self.selection_changed.emit()

    def validate_selection(self):
        ok = []
        for kind, ctx, i in self.selection:
            if kind == "g":
                if self.group(i) is not None:
                    ok.append((kind, ctx, i))
                continue
            if kind == "guide":
                if self.guide(i) is not None:
                    ok.append((kind, ctx, i))
                continue
            ents = self.ents(ctx)
            if ents is None:
                continue
            if (kind == "f" and i in ents.F) or (kind == "e" and i in ents.E):
                ok.append((kind, ctx, i))
        self.selection = ok

    def selected_groups(self):
        return [self.group(i) for k, _, i in self.selection if k == "g" and self.group(i)]

    def selected_geometry(self, ctx=None):
        """(faces, edges) selected in a context (default: the active one)."""
        ctx = self.active if ctx is None else ctx
        fs = [i for k, c, i in self.selection if k == "f" and c == ctx]
        es = [i for k, c, i in self.selection if k == "e" and c == ctx]
        return fs, es

    def selected_vertices(self, ctx=None):
        ents = self.ents(self.active if ctx is None else ctx)
        fs, es = self.selected_geometry(ctx)
        vids = set()
        for f in fs:
            for loop in ents.F[f].loops:
                vids |= set(loop)
        for e in es:
            vids |= set(ents.E[e])
        return vids

    # ------------------------------------------------------------ groups
    def add_group(self, ents, color, name=None):
        if name is None:
            name = tr("Part {n}").format(n=self.group_counter)
            self.group_counter += 1
        g = Group(self.next_gid, name, color, ents)
        self.next_gid += 1
        self.groups.append(g)
        return g

    def remove_group(self, gid):
        self.groups = [g for g in self.groups if g.id != gid]
        if self.active == gid:
            self.active = ROOT

    def replace_group_ents(self, gid, ents):
        g = self.group(gid)
        if g is not None:
            g.ents = ents

    def model_bbox(self):
        boxes = [b for b in (self.root.bbox(), *[g.bbox() for g in self.visible_groups()]) if b is not None]
        if not boxes:
            return None
        return np.array([np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)])
