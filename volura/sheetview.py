"""Het 2D-werkblad: een vlakke tekening naast het 3D-model.

`SheetView` is een gewoon QWidget met QPainter (geen OpenGL): de wereld is in
millimeters met Y omhoog, het raster past zich aan het zoomniveau aan, en de paden
van `doc.sheet` worden in hun laagkleur getekend. Het venster is de spelgeleider van
de gereedschappen uit `volura.sheettools`: het geeft muis, toetsen en meetvak door,
houdt de selectie bij en doet elke bewerking als één undo-stap op het document.

De 3D-weergave (`volura.viewport.Viewport`) en dit werkblad staan samen in het
centrale widget van het venster; schakelen doet `MainWindow.set_mode`.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QMenu, QSpinBox, QWidget)

from . import errors, nesting
from .sheettools import (ALIGN_LABELS, MEASURE_LABEL, PICK_PX, SHEET_TOOLS, align_deltas,
                         boolean_loops, copy_of, distribute_deltas, grid_offsets,
                         joined_paths, mirror_matrix, next_src, path_distance, path_in_rect,
                         path_touches_rect, paths_bbox, split_into_segments, translation,
                         transformed_points)
from .sheetsnap import SnapEngine, grid_step
from .vector2d import (SHEET_SPACING, OP_LABELS, Layer, VPath, _transform_meta)

_NO_LAYER = Layer("0", "#000000")   # voor een pad waarvan de laag weg is

GRID_MIN_PX = 8.0            # een rasterlijn moet minstens zo ver uit elkaar liggen
SELECT_COLOR = "#2f6fd6"     # geselecteerde paden: blauw en dikker
HOVER_COLOR = "#f08a24"      # het pad onder de cursor
GRID_COLOR = "#dfe3e8"
GRID_MAJOR = "#b9c0c9"
AXIS_X = "#d23b3b"
AXIS_Y = "#3fa34d"
SHEET_COLOR = "#4b5563"         # rand van de plaat: doorgetrokken en donker
SHEET_FILL = "#ffffff"          # de plaat zelf licht, met het raster erop
SHEET_SHADOW = "#c4cad3"
OUTSIDE_SHEET = "#e8ebef"       # naast de plaat iets donkerder, zodat de plaat opvalt
MARGIN_COLOR = "#8fa3bf"
OUTSIDE_COLOR = "#e02020"     # paden buiten de plaat of in de marge: rode rand
SHEET_LABEL_COLOR = "#6b7280"
VCB_CHARS = set("0123456789.,;-+=\"' ")

#: sneltoetsen op het werkblad, dezelfde letters als in 3D (L, R, C, A, F, ...)
SHEET_KEYS = {
    Qt.Key_L: "line", Qt.Key_R: "rectangle", Qt.Key_N: "roundrect", Qt.Key_C: "circle",
    Qt.Key_E: "ellipse", Qt.Key_A: "arc", Qt.Key_B: "centerarc", Qt.Key_P: "polygon",
    Qt.Key_T: "text", Qt.Key_F: "offset", Qt.Key_O: "offset",
}


class SheetView(QWidget):
    """De 2D-omgeving: raster, plaat, paden, selectie en de gereedschappen erbij."""

    status_changed = Signal(str)
    vcb_changed = Signal(str, str)        # label, waarde
    tool_changed = Signal(str)
    coords_changed = Signal(str)          # de cursorplaats in de statusbalk
    selection_changed = Signal()
    sheet_settings_requested = Signal()   # "Plaat…" vraagt de plaatinstellingen
    auto_arrange_requested = Signal()     # "Automatisch schikken" vraagt de opties
    to_model_requested = Signal()         # "Onderdelen naar 3D-model": platen in het model

    MIN_SCALE = 0.005                      # pixels per mm
    MAX_SCALE = 500.0

    def __init__(self, doc, settings=None, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.settings = settings
        self.tools = {cls.name: cls(self) for cls in SHEET_TOOLS}
        self.tool = None
        self.scale = 4.0                   # pixels per mm
        self.origin = QPointF(0.0, 0.0)    # de schermplek van wereld (0, 0)
        self.selected: list = []           # de geselecteerde VPath-objecten
        self.hover = None
        self.preview_transform = None      # voor errors.recover
        self.vcb_text = ""
        self.snap_to_grid = True           # raster vastzetten, aan of uit op de werkbalk
        self.offset_round = True           # afgeronde of scherpe hoeken bij Verschuiving
        self.current_layer = "Snijden"     # de laag van nieuw getekende paden
        self.snap_engine = SnapEngine(self)
        self._status_note = None
        self._status_shown = ""
        self._vcb_shown = None
        self._pan_from = None
        self._space = False
        self._placed_origin = False
        self._outside_ids: set = set()     # id van paden buiten de plaat of in de marge
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        doc.changed.connect(self._on_doc_changed)
        self.set_tool("select")

    # ------------------------------------------------------------ instellingen van het werkblad
    def join_type_name(self) -> str:
        return "Afgerond" if self.offset_round else "Scherp"

    def set_offset_round(self, round_corners: bool):
        self.offset_round = bool(round_corners)
        self.update()

    def set_grid_snap(self, on: bool):
        """Raster vastzetten: de cursor naar rasterkruisingen trekken of niet."""
        self.snap_to_grid = bool(on)
        self.update()

    def draw_layer(self) -> str:
        """De laag van nieuw getekende paden; een vergrendelde laag is dat niet."""
        lay = self.doc.sheet.layers.get(self.current_layer)
        if lay is None or lay.locked:
            return "0"
        return self.current_layer

    def set_current_layer(self, name) -> bool:
        """De actieve laag: nieuwe paden komen hier terecht (paneel "Lagen")."""
        name = str(name)
        if name not in self.doc.sheet.layers:
            return False
        self.current_layer = name
        self.update()
        return True

    def move_to_layer(self, name) -> bool:
        """De selectie naar een andere laag zetten, als één undo-stap."""
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst de contouren die naar een laag moeten.")
            return False
        name = str(name)
        lay = self.doc.sheet.layers.get(name)
        if lay is None:
            self.set_status(f"Laag {name} bestaat niet.")
            return False
        if lay.locked:
            self.set_status(f"Laag {name} staat op slot; maak hem eerst open.")
            return False
        moved = [p for p in sel if p.layer != name]
        if not moved:
            self.set_status(f"Die contouren staan al op laag {name}.")
            return False

        def change():
            for p in moved:
                p.layer = name
        self.edit(change)
        self.set_status(f"{len(moved)} contouren naar laag {name}")
        return True

    # ------------------------------------------------------------ wereld en scherm
    def to_screen(self, pt) -> QPointF:
        """Wereldpunt (mm) naar schermkoordinaten (pixels, Y omlaag)."""
        return QPointF(self.origin.x() + float(pt[0]) * self.scale,
                       self.origin.y() - float(pt[1]) * self.scale)

    def to_world(self, pos) -> np.ndarray:
        """Schermpunt (QPoint of QPointF) naar wereldkoordinaten (mm, Y omhoog)."""
        x = float(pos.x()) if hasattr(pos, "x") else float(pos[0])
        y = float(pos.y()) if hasattr(pos, "y") else float(pos[1])
        return self.to_world_xy(x, y)

    def to_world_xy(self, x, y) -> np.ndarray:
        x, y = float(x), float(y)
        return np.array([(x - self.origin.x()) / self.scale,
                         (self.origin.y() - y) / self.scale])

    def world_rect(self):
        """(xmin, ymin, xmax, ymax) van wat er nu te zien is."""
        a = self.to_world_xy(0.0, float(self.height()))
        b = self.to_world_xy(float(self.width()), 0.0)
        return float(a[0]), float(a[1]), float(b[0]), float(b[1])

    def _clamp_scale(self, value):
        return max(self.MIN_SCALE, min(self.MAX_SCALE, float(value)))

    def zoom_extents(self):
        """Alles op het werkblad (of de plaat) zo groot mogelijk in beeld."""
        s = self.doc.sheet
        box = s.bounds()
        if s.sheet_w and s.sheet_h:          # de platen horen er altijd helemaal bij
            for ox, oy in nesting.sheet_origins(s) or [(0.0, 0.0)]:
                r = (ox, oy, ox + float(s.sheet_w), oy + float(s.sheet_h))
                box = r if box is None else (min(box[0], r[0]), min(box[1], r[1]),
                                             max(box[2], r[2]), max(box[3], r[3]))
        w, h = max(self.width(), 1), max(self.height(), 1)
        if box is None:
            self.scale = self._clamp_scale(min(w, h) / 200.0)
            self.origin = QPointF(0.35 * w, 0.65 * h)
        else:
            pad, label = 24.0, 20.0          # ruimte rondom, en boven voor "Plaat · maat"
            dw = max(box[2] - box[0], 1e-6)
            dh = max(box[3] - box[1], 1e-6)
            self.scale = self._clamp_scale(min((w - 2 * pad) / dw, (h - 2 * pad - label) / dh))
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            self.origin = QPointF(0.5 * w - cx * self.scale,
                                  0.5 * (h + label) + cy * self.scale)
        self._placed_origin = True
        self.update()

    def zoom_by(self, factor, about: QPointF | None = None):
        """Zoomen rond een schermpunt: dat punt blijft op zijn plek staan."""
        about = QPointF(self.width() / 2.0, self.height() / 2.0) if about is None else about
        before = self.to_world(about)
        self.scale = self._clamp_scale(self.scale * float(factor))
        after = self.to_world(about)
        self.origin = QPointF(self.origin.x() + (after[0] - before[0]) * self.scale,
                              self.origin.y() - (after[1] - before[1]) * self.scale)
        self.update()

    # ------------------------------------------------------------ gereedschap en status
    def set_tool(self, name):
        if name not in self.tools:
            return
        if self.tool is not None:
            self.tool.deactivate()
        self.tool = self.tools[name]
        self.vcb_text = ""
        self.hover = None
        self._vcb_shown = None
        self._status_note = None
        self.tool.activate()
        self.setCursor(self.tool.cursor_shape)
        self.tool_changed.emit(name)
        self.refresh_status()
        self.update()

    def set_status(self, text):
        self._status_note = text
        self.show_status(text)

    def show_status(self, text):
        if text and text != self._status_shown:
            self._status_shown = text
            self.status_changed.emit(text)

    def refresh_status(self):
        if self.tool is None:
            return
        note, self._status_note = self._status_note, None
        self.show_status(note or self.tool.status_text())
        self.update_vcb()

    def announce(self):
        """Zorg dat de statusbalk opnieuw zegt wat het gereedschap nu wil (na moduswissel)."""
        self._status_shown = ""
        self._vcb_shown = None
        self.refresh_status()

    def update_vcb(self):
        if self.tool is None:
            return
        label = self.tool.vcb_label() or MEASURE_LABEL
        value = self.vcb_text if self.vcb_text else self.tool.vcb_value
        if (label, value) == self._vcb_shown:
            return
        self._vcb_shown = (label, value)
        self.vcb_changed.emit(label, value)

    def submit_vcb(self, text):
        self.vcb_text = ""
        if self.tool is not None and text.strip():
            try:
                self.tool.vcb_enter(text.strip())
            except ValueError as e:
                self.set_status(f"Ongeldige invoer: {e}")
            except Exception as e:                       # een gereedschap mag nooit crashen
                errors.log_exception(note=f"2D-gereedschap: {e}")
                self.set_status(f"Invoer mislukt: {e}")
        self._vcb_shown = None
        self.refresh_status()
        self.update()

    def set_vcb(self, value):
        self.tool.vcb_value = value
        self.update_vcb()

    # ------------------------------------------------------------ selectie
    def _on_doc_changed(self):
        self._prune_selection()
        self._refresh_outside()
        self.update()

    def _refresh_outside(self, announce=True) -> int:
        """Paden buiten de plaat of in de marge markeren en het aantal melden."""
        try:
            outside = nesting.outside_paths(self.doc.sheet)
        except Exception as e:                       # een waarschuwing mag nooit crashen
            errors.log_exception(note=f"plaatcontrole: {e}")
            outside = []
        self._outside_ids = {id(p) for p in outside}
        n = len(outside)
        if n and announce:
            self.show_status(f"{n} paden liggen buiten de plaat of in de marge")
        return n

    def _prune_selection(self):
        live = {id(p) for p in self.doc.sheet.paths}
        keep = [p for p in self.selected if id(p) in live]
        if len(keep) != len(self.selected):
            self.selected = keep

    def selected_paths(self) -> list:
        self._prune_selection()
        return list(self.selected)

    def set_selection(self, paths):
        out, seen = [], set()
        for p in list(paths or ()):
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        self.selected = out
        self.selection_changed.emit()
        self.update()

    def add_to_selection(self, paths):
        self.set_selection(list(self.selected) + list(paths or ()))

    def toggle_selection(self, path):
        if any(p is path for p in self.selected):
            self.selected = [p for p in self.selected if p is not path]
        else:
            self.selected = list(self.selected) + [path]
        self.selection_changed.emit()
        self.update()

    def clear_selection(self):
        if self.selected:
            self.selected = []
            self.selection_changed.emit()
        self.update()

    def select_all(self):
        self.set_selection(self._selectable_paths())

    def _layer(self, path) -> Layer:
        """De laag van een pad; een pad zonder laag gedraagt zich als laag "0"."""
        return self.doc.sheet.layers.get(path.layer) or _NO_LAYER

    def _visible(self, path) -> bool:
        return self._layer(path).visible

    def _selectable_paths(self):
        return [p for p in self.doc.sheet.paths if self._visible(p)
                and not self._layer(p).locked and len(p.pts)]

    def selection_center(self):
        box = paths_bbox(self.selected_paths())
        if box is None:
            box = self.doc.sheet.bounds()
        if box is None:
            return None
        return np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])

    def path_at(self, x, y):
        """Het pad dat het dichtst bij de cursor ligt, binnen een paar pixels."""
        tol = PICK_PX / self.scale
        p = self.to_world_xy(x, y)
        best, dist = None, tol
        for path in self._selectable_paths():
            d = path_distance(path, p)
            if d <= dist:
                best, dist = path, d
        return best

    def paths_in_band(self, a, b):
        """Kaderselectie: links naar rechts alleen wat erin ligt, rechts naar links ook wat het raakt."""
        w0 = self.to_world_xy(min(a[0], b[0]), max(a[1], b[1]))
        w1 = self.to_world_xy(max(a[0], b[0]), min(a[1], b[1]))
        rect = (float(w0[0]), float(w0[1]), float(w1[0]), float(w1[1]))
        window = b[0] >= a[0]
        out = []
        for path in self._selectable_paths():
            if path_in_rect(path, rect) if window else path_touches_rect(path, rect):
                out.append(path)
        return out

    # ------------------------------------------------------------ bewerken
    def edit(self, change):
        """Eén undo-stap: checkpoint, de paden aanpassen, het document melden."""
        self.doc.checkpoint()
        change()
        self.doc.notify()
        self.update()

    def transform_paths(self, paths, M, copy=False):
        """Matrix `M` op de paden; met `copy` blijven de originele paden liggen."""
        M = np.asarray(M, dtype=float)
        lin, off = M[:2, :2], M[:2, 2]
        if copy:
            fresh = []
            for p in paths:
                q = copy_of(p)
                q.pts = transformed_points(q.pts, M)
                q.meta = _transform_meta(p.meta, lin, off)
                fresh.append(q)
            self.doc.sheet.paths = list(self.doc.sheet.paths) + fresh
            self.set_selection(fresh)
            return fresh
        for p in paths:
            p.pts = transformed_points(p.pts, M)
            p.meta = _transform_meta(p.meta, lin, off)
        return list(paths)

    def delete_selected(self):
        sel = self.selected_paths()
        if not sel:
            self.set_status("Er is niets geselecteerd om te wissen.")
            return 0
        live = {id(p) for p in sel}
        sheet = self.doc.sheet

        def drop():
            sheet.paths = [p for p in sheet.paths if id(p) not in live]

        self.edit(drop)
        self.clear_selection()
        self.set_status(f"{len(sel)} contouren gewist")
        return len(sel)

    def mirror(self, horizontal=True):
        """Spiegelen rond het midden van de selectie: horizontaal = links en rechts om."""
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst wat u wilt spiegelen.")
            return False
        center = self.selection_center()
        if center is None:
            return False
        self.edit(lambda: self.transform_paths(sel, mirror_matrix(bool(horizontal), center)))
        self.set_status("Contouren gespiegeld")
        return True

    # ------------------------------------------------------------ bewerkingen op de selectie
    def _replace_selected(self, shapes, message):
        """De selectie vervangen door nieuwe paden, als één undo-stap.

        `shapes` is een lijst `(punten, gesloten, meta, gevuld)`; alle paden die erbij
        horen krijgen één bronnummer, zodat gaten als één vorm gelezen worden.
        """
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst de contouren voor deze bewerking.")
            return []
        sheet = self.doc.sheet
        src = next_src(sheet.paths)
        live = {id(p) for p in sel}
        fresh: list = []

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed, meta, filled in shapes:
                q = VPath(np.asarray(pts, dtype=float), bool(closed), sel[0].layer,
                          bool(filled), "nonzero", src, dict(meta or {}))
                keep.append(q)
                fresh.append(q)
            sheet.paths = keep
        self.edit(change)
        self.set_selection(fresh)
        self.set_status(message)
        return fresh

    def sheet_boolean(self, op) -> bool:
        """Samenvoegen, aftrekken of snijden van de geselecteerde gesloten paden."""
        labels = {"union": "samengevoegd", "subtract": "afgetrokken",
                  "intersect": "in snijpunt gezet"}
        sel = self.selected_paths()
        if len([p for p in sel if p.closed and len(p.pts) >= 3]) < (1 if op == "union" else 2):
            self.set_status("Selecteer eerst twee gesloten contouren voor deze bewerking.")
            return False
        try:
            loops = boolean_loops(sel, op)
        except ValueError as e:
            self.set_status(f"Bewerking lukt niet: {e}")
            return False
        except Exception as e:                       # manifold3d mag hier niets laten crashen
            errors.log_exception(note=f"2D-booleaanse bewerking: {e}")
            self.set_status(f"Bewerking lukt niet: {e}")
            return False
        if not loops:
            self.set_status("Er blijft niets over van deze contouren.")
            return False
        shapes = [(loop, True, {}, True) for loop in loops]
        self._replace_selected(shapes, f"{len(sel)} contouren {labels[op]}")
        return True

    def join_selected(self) -> int:
        """Open lijnstukken aaneenrijgen tot polylijnen en gesloten vormen."""
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst de lijnen die u wilt aaneenrijgen.")
            return 0
        joined = joined_paths(sel)
        src = next_src(self.doc.sheet.paths)
        out = []
        for p in joined:
            p = VPath(p.pts, p.closed, p.layer, p.filled, p.rule, src, dict(p.meta))
            out.append((p.pts, p.closed, p.meta, p.filled))
        live = {id(p) for p in sel}
        sheet = self.doc.sheet

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed, meta, filled in out:
                keep.append(VPath(np.asarray(pts, dtype=float), closed, sel[0].layer,
                                  filled, "nonzero", src, meta))
            sheet.paths = keep
        self.edit(change)
        self.set_selection([p for p in sheet.paths if int(p.src) == src])
        made = len(out)
        self.set_status(f"{made} paden uit {len(sel)} lijnen")
        return made

    def split_selected(self) -> int:
        """Elk geselecteerd pad in losse lijnstukken opsplitsen."""
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst de contouren die u wilt opsplitsen.")
            return 0
        segs = split_into_segments(sel)
        src = next_src(self.doc.sheet.paths)
        sheet = self.doc.sheet
        live = {id(p) for p in sel}

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed in segs:
                keep.append(VPath(np.asarray(pts, dtype=float), closed, sel[0].layer,
                                  False, "nonzero", src))
            sheet.paths = keep
        self.edit(change)
        self.set_selection([p for p in sheet.paths if int(p.src) == src])
        self.set_status(f"{len(segs)} lijnstukken")
        return len(segs)

    def align_selected(self, mode) -> bool:
        """Uitlijnen: links, midden, rechts, boven of onder, over de hele selectie."""
        sel = self.selected_paths()
        if len(sel) < 2:
            self.set_status("Selecteer minstens twee contouren om uit te lijnen.")
            return False
        if mode not in ALIGN_LABELS:
            return False
        deltas = align_deltas(sel, mode)
        self.edit(lambda: [self.transform_paths([p], translation(d))
                           for p, d in zip(sel, deltas)])
        self.set_status(f"contouren uitgelijnd ({ALIGN_LABELS[mode]})")
        return True

    def distribute_selected(self, axis="x") -> bool:
        """Gelijk verdelen: overal dezelfde afstand tussen de middenpunten."""
        sel = self.selected_paths()
        if len(sel) < 3:
            self.set_status("Selecteer minstens drie contouren om gelijk te verdelen.")
            return False
        deltas = distribute_deltas(sel, axis)
        self.edit(lambda: [self.transform_paths([p], translation(d))
                           for p, d in zip(sel, deltas)])
        self.set_status("contouren gelijk verdeeld")
        return True

    def duplicate_grid(self, cols=2, rows=1, dx=10.0, dy=10.0) -> list:
        """Dupliceren in een rij of raster: kopieën naast en boven het origineel."""
        sel = self.selected_paths()
        if not sel:
            self.set_status("Selecteer eerst wat u wilt dupliceren.")
            return []
        offsets = grid_offsets(cols, rows, dx, dy)
        if not offsets:
            self.set_status("Een rij van één kolom en één rij is geen duplicatie.")
            return []
        fresh: list = []

        def change():
            src = next_src(self.doc.sheet.paths)
            for off in offsets:
                for p in sel:
                    q = copy_of(p)
                    q.pts = transformed_points(q.pts, translation(off))
                    q.meta = _transform_meta(p.meta, np.eye(3)[:2, :2], off)
                    q.src = src
                    self.doc.sheet.paths.append(q)
                    fresh.append(q)
        self.edit(change)
        self.set_selection(fresh)
        self.set_status(f"{len(fresh)} kopieën geplaatst")
        return fresh

    # ------------------------------------------------------------ automatisch schikken
    def arrange_source(self) -> list:
        """De paden die meedoen: de selectie, of alles zichtbare als niets gekozen is."""
        sel = self.selected_paths()
        if sel:
            return sel
        return [p for p in self.doc.sheet.paths
                if self._visible(p) and not self._layer(p).locked and len(p.pts)]

    def auto_arrange(self, options=None) -> bool:
        """Schik de onderdelen zo zuinig mogelijk over de platen, als één undo-stap."""
        options = dict(options or {})
        sheet = self.doc.sheet
        if not sheet.sheet_w or not sheet.sheet_h:
            self.set_status("Kies eerst een plaatformaat (Plaat…) voor je schikt.")
            return False
        source = self.arrange_source()
        if not source:
            self.set_status("Er is niets om te schikken.")
            return False
        gap = float(options.get("gap", sheet.sheet_gap))
        margin = float(options.get("margin", sheet.sheet_margin))
        rotations = options.get("rotations", (0, 90, 180, 270))
        add_sheets = bool(options.get("add_sheets", True))
        max_sheets = int(sheet.sheet_count or 1) if not add_sheets else int(options.get("max_sheets", 20))
        parts = nesting.parts_of(sheet, source)
        try:
            placements = nesting.nest(parts, sheet.sheet_w, sheet.sheet_h, margin, gap,
                                      rotations=rotations, max_sheets=max_sheets)
        except ValueError as e:
            self.set_status(str(e))
            return False
        except Exception as e:                       # nesten mag nooit crashen
            errors.log_exception(note=f"automatisch schikken: {e}")
            self.set_status(f"Schikken lukt niet: {e}")
            return False
        spacing = float(sheet.sheet_w) + SHEET_SPACING
        used = max((si for (_p, si, *_r) in placements), default=0) + 1

        def change():
            for (part, si, angle, dx, dy) in placements:
                M = nesting.place_transform(part, angle, dx, dy, origin=(si * spacing, 0.0))
                lin, off = M[:2, :2], M[:2, 2]
                for p in part.paths:
                    p.pts = transformed_points(p.pts, M)
                    p.meta = _transform_meta(p.meta, lin, off)
            if add_sheets:
                sheet.sheet_count = max(int(sheet.sheet_count or 1), used)
        self.edit(change)
        util = nesting.utilization(placements, sheet.sheet_w, sheet.sheet_h)
        self._refresh_outside(announce=False)
        self.set_status(f"{len(parts)} onderdelen op {used} platen, "
                        f"benutting {round(util * 100)}%")
        return True

    # ------------------------------------------------------------ rechtsklikmenu
    @errors.safe_handler
    def contextMenuEvent(self, event):
        pos = event.pos()
        path = self.path_at(pos.x(), pos.y())
        if path is not None and path not in self.selected:
            self.set_selection([path])
        menu = self.build_context_menu()
        menu.exec(event.globalPos())

    def build_context_menu(self) -> QMenu:
        """Het rechtsklikmenu van het werkblad: gereedschap en de bewerkingen."""
        menu = QMenu(self)
        for name, label in (("select", "Selecteren"), ("line", "Lijn"),
                            ("rectangle", "Rechthoek"), ("circle", "Cirkel"),
                            ("text", "Tekst…"), ("offset", "Verschuiving")):
            act = menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, n=name: self._menu_tool(n))
        menu.addSeparator()
        for label, slot in (("Samenvoegen", lambda: self.sheet_boolean("union")),
                            ("Aftrekken", lambda: self.sheet_boolean("subtract")),
                            ("Snijpunt", lambda: self.sheet_boolean("intersect")),
                            ("Aaneenrijgen", self.join_selected),
                            ("Opsplitsen in lijnstukken", self.split_selected)):
            menu.addAction(label, slot)
        align = menu.addMenu("Uitlijnen")
        for mode in ALIGN_LABELS:
            align.addAction(ALIGN_LABELS[mode].capitalize(),
                            lambda checked=False, m=mode: self.align_selected(m))
        dist = menu.addMenu("Gelijk verdelen")
        dist.addAction("Horizontaal", lambda checked=False: self.distribute_selected("x"))
        dist.addAction("Verticaal", lambda checked=False: self.distribute_selected("y"))
        layers = menu.addMenu("Naar laag")
        for name, lay in self.doc.sheet.layers.items():
            act = layers.addAction(f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            act.setEnabled(not lay.locked)
            if lay.locked:
                act.setToolTip("Deze laag staat op slot")
            act.triggered.connect(lambda checked=False, n=name: self.move_to_layer(n))
        menu.addSeparator()
        menu.addAction("Dupliceren in rij of raster…", self.duplicate_dialog)
        menu.addAction("Wissen", self.delete_selected)
        menu.addAction("Alles selecteren", self.select_all)
        menu.addSeparator()
        menu.addAction("Automatisch schikken…", lambda: self.auto_arrange_requested.emit())
        menu.addAction("Plaat…", lambda: self.sheet_settings_requested.emit())
        menu.addAction("Onderdelen naar 3D-model", lambda: self.to_model_requested.emit())
        return menu

    def _menu_tool(self, name):
        if name == "text":
            self.set_tool("text")
            self.tools["text"].ask_text()
            self.setFocus()
            return
        self.set_tool(name)

    def duplicate_dialog(self):
        """Dialoog 'Dupliceren in een rij of raster' (aantal en afstand)."""
        dlg = DuplicateSheetDialog(self)
        if dlg.exec() == QDialog.Accepted:
            c, r, dx, dy = dlg.values()
            self.duplicate_grid(c, r, dx, dy)
        self.setFocus()

    # ------------------------------------------------------------ tekenen
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw_background(p)
        self._draw_grid(p)
        self._draw_sheet_rect(p)
        self._draw_margin(p)
        self._draw_axes(p)
        self._draw_paths(p)
        if self.tool is not None:
            try:
                self.tool.paint(p)
            except Exception as e:                       # een voorbeeld mag nooit crashen
                errors.log_exception(note=f"2D-voorbeeld: {e}")
        p.end()

    def _draw_grid(self, p):
        step = grid_step(self.scale)
        xmin, ymin, xmax, ymax = self.world_rect()
        if (xmax - xmin) / step > 4000 or (ymax - ymin) / step > 4000:
            return
        minor = QPen(QColor(GRID_COLOR))
        minor.setWidthF(1.0)
        major = QPen(QColor(GRID_MAJOR))
        major.setWidthF(1.0)
        start = math.floor(xmin / step) * step
        x = start
        while x <= xmax:
            i = round(x / step)
            p.setPen(major if i % 10 == 0 else minor)
            a = self.to_screen((x, ymin))
            b = self.to_screen((x, ymax))
            p.drawLine(a, b)
            x += step
        start = math.floor(ymin / step) * step
        y = start
        while y <= ymax:
            i = round(y / step)
            p.setPen(major if i % 10 == 0 else minor)
            a = self.to_screen((xmin, y))
            b = self.to_screen((xmax, y))
            p.drawLine(a, b)
            y += step

    def _sheet_screen_rects(self):
        """De platen als schermrechthoeken; leeg zonder plaatformaat."""
        s = self.doc.sheet
        if not s.sheet_w or not s.sheet_h:
            return []
        out = []
        for (ox, oy) in nesting.sheet_origins(s) or [(0.0, 0.0)]:
            a = self.to_screen((ox, oy))
            b = self.to_screen((ox + float(s.sheet_w), oy + float(s.sheet_h)))
            out.append(QRectF(a, b).normalized())
        return out

    def _draw_background(self, p):
        """Wit zonder plaat; met een plaat is alles ernaast grijs en de plaat licht met
        een schaduw, zodat duidelijk is waar het materiaal ligt."""
        rects = self._sheet_screen_rects()
        if not rects:
            p.fillRect(self.rect(), QColor("#ffffff"))
            return
        p.fillRect(self.rect(), QColor(OUTSIDE_SHEET))
        for r in rects:
            p.fillRect(r.translated(3.0, 3.0), QColor(SHEET_SHADOW))
        for r in rects:
            p.fillRect(r, QColor(SHEET_FILL))

    def _draw_sheet_rect(self, p):
        s = self.doc.sheet
        rects = self._sheet_screen_rects()
        if not rects:
            return
        pen = QPen(QColor(SHEET_COLOR))
        pen.setWidthF(1.6)
        pen.setJoinStyle(Qt.MiterJoin)
        p.setBrush(Qt.NoBrush)
        font = p.font()
        font.setPointSizeF(10.0)
        font.setBold(True)
        size = f"{float(s.sheet_w):g} × {float(s.sheet_h):g}" +             (f" × {float(s.sheet_thickness):g}" if s.sheet_thickness else "") + " mm"
        for i, r in enumerate(rects):
            p.setPen(pen)
            p.drawRect(r)
            p.setPen(QPen(QColor(SHEET_LABEL_COLOR)))
            p.setFont(font)
            label = (f"Plaat {i + 1}" if len(rects) > 1 else "Plaat") + f"  ·  {size}"
            p.drawText(QPointF(r.left() + 2.0, r.top() - 6.0), label)

    def _draw_margin(self, p):
        """De marge langs de rand stippellijn, zodat zichtbaar is wat erop mag."""
        s = self.doc.sheet
        m = float(s.sheet_margin or 0.0)
        if not s.sheet_w or not s.sheet_h or m <= 0:
            return
        pen = QPen(QColor(MARGIN_COLOR))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for (x0, y0, x1, y1) in nesting.usable_rects(s):
            a = self.to_screen((x0, y0))
            b = self.to_screen((x1, y1))
            p.drawRect(QRectF(a, b).normalized())

    def _draw_axes(self, p):
        xmin, ymin, xmax, ymax = self.world_rect()
        pen = QPen(QColor(AXIS_X))
        pen.setWidthF(1.2)
        p.setPen(pen)
        o = self.to_screen((0.0, 0.0))
        p.drawLine(self.to_screen((xmin, 0.0)), self.to_screen((xmax, 0.0)))
        pen = QPen(QColor(AXIS_Y))
        pen.setWidthF(1.2)
        p.setPen(pen)
        p.drawLine(self.to_screen((0.0, ymin)), self.to_screen((0.0, ymax)))
        p.setPen(QPen(QColor("#2b2b2b")))
        p.drawLine(QPointF(o.x() - 6, o.y()), QPointF(o.x() + 6, o.y()))
        p.drawLine(QPointF(o.x(), o.y() - 6), QPointF(o.x(), o.y() + 6))

    def _draw_paths(self, p):
        sheet = self.doc.sheet
        chosen = {id(path) for path in self.selected}
        for path in sheet.paths:
            if not self._visible(path) or len(path.pts) < 1:
                continue
            color = self._layer(path).color or "#000000"
            if id(path) in self._outside_ids:      # buiten de plaat of in de marge
                warn = QPen(QColor(OUTSIDE_COLOR))
                warn.setWidthF(4.2)
                p.setPen(warn)
                p.drawPath(self._path_of(path))
            if path is self.hover:
                pen = QPen(QColor(HOVER_COLOR))
                pen.setWidthF(3.0)
                p.setPen(pen)
                p.drawPath(self._path_of(path))
            pen = QPen(QColor(SELECT_COLOR if id(path) in chosen else color))
            pen.setWidthF(2.6 if id(path) in chosen else 1.4)
            p.setPen(pen)
            p.drawPath(self._path_of(path))

    def _path_of(self, path) -> QPainterPath:
        pts = path.pts
        out = QPainterPath()
        if not len(pts):
            return out
        out.moveTo(self.to_screen(pts[0]))
        for q in pts[1:]:
            out.lineTo(self.to_screen(q))
        if path.closed:
            out.closeSubpath()
        return out

    # ------------------------------------------------------------ invoer
    def event(self, e):
        """Cijfers in het meetvak mogen nooit een sneltoets worden, zoals in de 3D-weergave."""
        from PySide6.QtCore import QEvent
        if e.type() == QEvent.ShortcutOverride:
            t = e.text()
            if self.vcb_text and t and (t in VCB_CHARS or t.isalnum() or t == " "):
                e.accept()
                return True
            if t and t in "0123456789.-,;=":
                e.accept()
                return True
            if e.key() in (Qt.Key_Escape, Qt.Key_Backspace, Qt.Key_Return, Qt.Key_Enter):
                e.accept()
                return True
        return super().event(e)

    @errors.safe_handler
    def keyPressEvent(self, e):
        k = e.key()
        t = e.text()
        mods = e.modifiers()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self.vcb_text:
                self.submit_vcb(self.vcb_text)
            elif self.tool is not None:
                self.tool.finish_input()          # Enter zonder tekst: klaar met tekenen
                self.refresh_status()
                self.update()
            return
        if k == Qt.Key_Backspace:
            if self.vcb_text:
                self.vcb_text = self.vcb_text[:-1]
                self.update_vcb()
            return
        if k == Qt.Key_Escape:
            if self.vcb_text:
                self.vcb_text = ""
                self.update_vcb()
                return
            if self.tool is not None and self.tool.name != "select":
                self.set_tool("select")
                return
            self.clear_selection()
            return
        if k == Qt.Key_Space:
            self._space = True
            if self.tool is not None and self.tool.name != "select":
                self.set_tool("select")
            return
        if mods & Qt.ControlModifier and k == Qt.Key_A:
            self.select_all()
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace) and not self.vcb_text:
            self.delete_selected()
            return
        if not mods and k in (Qt.Key_M, Qt.Key_Q, Qt.Key_S):
            self.set_tool({Qt.Key_M: "move", Qt.Key_Q: "rotate", Qt.Key_S: "scale"}[k])
            return
        if not mods and k in SHEET_KEYS:
            self.set_tool(SHEET_KEYS[k])          # L, R, C, A, F … zoals in 3D
            return
        if t and (t in "0123456789.-,;="
                  or (self.vcb_text and (t in VCB_CHARS or t.isalnum() or t == "<"))):
            self.vcb_text += t
            self.update_vcb()
            return
        if self.tool is not None and self.tool.key_press(e):
            self.refresh_status()
            self.update()
            return
        super().keyPressEvent(e)

    @errors.safe_handler
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = False
            return
        super().keyReleaseEvent(e)

    @errors.safe_handler
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        self._pan_from = None
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._pan_from = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton:
            if self.tool is not None:
                self.tool.cancel()
            self.refresh_status()
            self.update()
            return
        if e.button() == Qt.LeftButton and self.tool is not None:
            self.tool.mouse_press(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseDoubleClickEvent(self, e):
        """Dubbelklik beëindigt een open pad, zoals in 3D."""
        self.setFocus()
        if e.button() != Qt.LeftButton or self.tool is None:
            return
        self._pan_from = None
        pos = e.position()
        self.tool.double_click(e, pos.x(), pos.y())
        self.refresh_status()
        self.update()

    @errors.safe_handler
    def mouseMoveEvent(self, e):
        pos = e.position()
        self.show_coords(pos)
        if self._pan_from is not None:
            d = pos - self._pan_from
            self.origin = QPointF(self.origin.x() + d.x(), self.origin.y() + d.y())
            self._pan_from = pos
            self.update()
            return
        if self.tool is not None:
            self.tool.mouse_move(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.LeftButton) and self._pan_from is not None:
            self._pan_from = None
            self.setCursor(self.tool.cursor_shape if self.tool else Qt.ArrowCursor)
            return
        if e.button() == Qt.LeftButton and self.tool is not None:
            pos = e.position()
            self.tool.mouse_release(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.zoom_by(1.25 ** steps, e.position())

    def show_coords(self, pos):
        w = self.to_world(pos)
        self.coords_changed.emit(f"X {w[0]:.2f} mm   Y {w[1]:.2f} mm")

    def resizeEvent(self, event):
        if not self._placed_origin and self.width() > 0 and self.height() > 0:
            self._placed_origin = True
            self.origin = QPointF(0.35 * self.width(), 0.65 * self.height())
        super().resizeEvent(event)

    def showEvent(self, event):
        """De eerste keer dat het werkblad in beeld komt: meteen alles zichtbaar."""
        super().showEvent(event)
        if not self._placed_origin:
            self.zoom_extents()


class DuplicateSheetDialog(QDialog):
    """Dupliceren in een rij of raster: aantal kolommen en rijen, en de afstand ertussen."""

    def __init__(self, view: SheetView, parent=None):
        super().__init__(parent or view)
        self.setWindowTitle("Dupliceren in een rij of raster")
        box = self.selected_box(view)
        lay = QFormLayout(self)
        self.cols = QSpinBox()
        self.cols.setRange(1, 100)
        self.cols.setValue(3)
        self.cols.setToolTip("Aantal exemplaren naast elkaar")
        self.rows = QSpinBox()
        self.rows.setRange(1, 100)
        self.rows.setValue(1)
        self.rows.setToolTip("Aantal exemplaren boven elkaar")
        self.dx = QDoubleSpinBox()
        self.dy = QDoubleSpinBox()
        for spin in (self.dx, self.dy):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
        self.dx.setValue(max(10.0, box[2] - box[0]) if box else 10.0)
        self.dy.setValue(max(10.0, box[3] - box[1]) if box else 10.0)
        lay.addRow("Kolommen", self.cols)
        lay.addRow("Rijen", self.rows)
        lay.addRow("Afstand horizontaal", self.dx)
        lay.addRow("Afstand verticaal", self.dy)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)

    @staticmethod
    def selected_box(view: SheetView):
        return paths_bbox(view.selected_paths())

    def values(self):
        return (int(self.cols.value()), int(self.rows.value()),
                float(self.dx.value()), float(self.dy.value()))
