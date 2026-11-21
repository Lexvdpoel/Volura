"""Window > Bed Mesh…: how flat the bed of a Klipper printer is, point by point.

A bed mesh is what the probe of a printer touched: a grid of z heights over the bed, which
Klipper interpolates to a finer grid so a print follows the bed everywhere. This window paints
that grid as a heat map - blue under 0, red above it, white where the bed is at 0 - and writes
every probed value in the cell it belongs to, so the low corner of a bed is seen at a glance and
read where it stands. `volura/bed_mesh.py` says what a value looks like and what the window says
of a mesh; `volura/printer_link.py` asks the printer what it has.

Only Klipper (Moonraker) has a bed mesh, so only those printers stand in the box at the top:
OctoPrint has one only through a plugin and PrusaLink not at all. A mesh is asked from the
printer and never made here, so with no Klipper printer under Window > Preferences > Printers
the window says that and asks nothing at all.

The printer is asked once when the window comes on the screen, and again only when *Refresh* is
pressed: a bed mesh does not change by itself, so there is nothing to poll. The asking runs in
the background (volura/background.py) and without a progress dialog, and a printer whose answer
has not come back yet is not asked a second time. What it got wrong stands muted under the
picture, and the picture of the last good answer stays on the screen; *Refresh* tries again.
Moving the mouse over the picture says which point of the bed stands under it.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPalette, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import background, bed_mesh, printer_link, theme
from .i18n import tr

#: the space around the heat map, and the room the colour bar keeps for its numbers (pixels)
MARGIN = 12
#: how little a cell may be before its number is left out (pixels): a number that barely fits
#: is a mess of ink, and the colour of that cell says as much
CELL_TEXT = 30
#: what the window says when there is no printer with a bed mesh to ask
NEEDS_KLIPPER = tr("Bed mesh needs a Klipper (Moonraker) printer; add one under Window > "
                   "Preferences > Printers")
#: what the window says of a printer that answers no mesh: a bed nobody probed is no error
NO_MESH = tr("This printer has no bed mesh loaded. Run a bed mesh calibration on the printer "
             "and press Refresh.")


def key_of(conn) -> tuple:
    """What a bed mesh belongs to in this window: one printer, one mesh. The printer that is not
    there has no key at all."""
    if conn is None:
        return ()
    return (conn.name, conn.kind, printer_link.normalise_url(conn.url))


def _ink(back) -> QColor:
    """Black or white ink on a cell of that colour: whichever of the two is read easiest."""
    return QColor(Qt.black if back.lightnessF() > 0.55 else Qt.white)


def _mm(value) -> str:
    """A z of the mesh as the window says it: mm with three decimals, for a bed is flat to
    hundredths."""
    return f"{float(value):.3f}"


class BedView(QWidget):
    """The mesh as a heat map, in the proportions of the bed itself: x to the right and y up, so
    the mesh lies on the screen the way the bed lies under the nozzle, seen from above.

    The first row of a mesh is the lowest y of the bed - its front - and is drawn at the bottom.
    The map is painted with the grid the printer interpolated when it has one and with the probed
    points when it has none, while the numbers are always the probed ones: those are what the
    probe really touched. Moving the mouse says which probed cell stands under it (`hovered`);
    the window turns that into the x, y and z of that point of the bed.
    """

    hovered = Signal(int, int)              # the row and column under the mouse, -1 for neither

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.lo = self.hi = 0.0             # what the colours span, in mm
        self._over = None                   # the cell the mouse stood over last
        self.setMinimumSize(380, 300)
        self.setMouseTracking(True)
        self.setAccessibleName(tr("The mesh of the bed as a heat map"))

    # ---------------------------------------------------------------------- what to paint
    def set_mesh(self, mesh, lo=0.0, hi=0.0):
        """The mesh to paint and what its colours span; None takes the picture away."""
        self.mesh = mesh
        self.lo, self.hi = (0.0, 0.0) if mesh is None else (float(lo), float(hi))
        self._over = None
        self.update()

    def _grid(self):
        """What the map is painted with; None when there is nothing to paint."""
        grid = None if self.mesh is None else self.mesh.picture()
        if grid is None or getattr(grid, "ndim", 0) != 2 or not grid.size:
            return None
        return grid

    def _probed(self):
        """The grid the numbers come from; None when the mesh has no such grid."""
        grid = None if self.mesh is None else self.mesh.probed
        if grid is None or getattr(grid, "ndim", 0) != 2 or not grid.size:
            return None
        return grid

    def place(self) -> QRect:
        """Where the bed is painted: as much of the widget as fits, in the proportions of the bed
        itself. An empty rect when there is nothing to paint."""
        if self.mesh is None or self.width() <= 2 * MARGIN or self.height() <= 2 * MARGIN:
            return QRect()
        wide, high = self.mesh.x_max - self.mesh.x_min, self.mesh.y_max - self.mesh.y_min
        if wide <= 0 or high <= 0:
            return QRect()                  # a bed that bounds nothing has nothing to lay out
        scale = min((self.width() - 2 * MARGIN) / wide, (self.height() - 2 * MARGIN) / high)
        bed_w, bed_h = int(wide * scale), int(high * scale)
        if bed_w < 1 or bed_h < 1:
            return QRect()                  # a bed that thin has no cells left to draw
        return QRect((self.width() - bed_w) // 2, (self.height() - bed_h) // 2, bed_w, bed_h)

    @staticmethod
    def _cell(where: QRect, rows, cols, row, col) -> QRect:
        """The cell of one row and column of a grid of `rows` by `cols` on `where`, with row 0 at
        the bottom - the front of the bed - and no seam between two neighbouring cells."""
        back = where.y() + where.height()
        left = where.x() + where.width() * col // cols
        right = where.x() + where.width() * (col + 1) // cols
        top = back - where.height() * (row + 1) // rows
        bottom = back - where.height() * row // rows
        return QRect(left, top, right - left, bottom - top)

    @staticmethod
    def _under(x, y, where: QRect, grid) -> tuple:
        """Which cell of `grid` stands under this pixel: (row, col), or None when the pixel
        stands outside the bed."""
        if grid is None or where.isNull() or not where.contains(int(x), int(y)):
            return None
        rows, cols = grid.shape
        col = min(cols - 1, max(0, int((x - where.x()) * cols / where.width())))
        row = min(rows - 1, max(0, int((where.y() + where.height() - y) * rows / where.height())))
        return int(row), int(col)

    def rect_of(self, row, col) -> QRect:
        """The cell that shows this row and column of the painted map, in pixels; an empty rect
        when the map has no such cell."""
        where, grid = self.place(), self._grid()
        if where.isNull() or grid is None:
            return QRect()
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return QRect()
        return self._cell(where, grid.shape[0], grid.shape[1], row, col)

    def cell_at(self, x, y):
        """Which probed cell stands under this pixel: (row, col), or None when nothing stands
        there. The numbers on the map come from the probed grid, so the mouse is answered on that
        same grid and says what stands written there."""
        return self._under(x, y, self.place(), self._probed())

    # ------------------------------------------------------------------------- the painting
    def paintEvent(self, event):
        p = QPainter(self)
        where, grid = self.place(), self._grid()
        if grid is None or where.isNull():
            self._say_empty(p)
            p.end()
            return
        rows, cols = grid.shape
        p.setPen(Qt.NoPen)
        for row in range(rows):
            for col in range(cols):
                colour = bed_mesh.colour(float(grid[row][col]), self.lo, self.hi)
                p.setBrush(QColor(*colour))
                p.drawRect(self._cell(where, rows, cols, row, col))
        self._say_values(p, where)
        p.setPen(QPen(QColor(theme.color("muted"))))
        p.setBrush(Qt.NoBrush)
        p.drawRect(where.adjusted(0, 0, -1, -1))          # the edge of the bed
        p.end()

    def _say_empty(self, p):
        """What stands in the middle of a bed nothing was asked about yet."""
        p.setPen(QPen(QColor(theme.color("muted"))))
        p.drawText(self.rect(), int(Qt.AlignCenter),
                   tr("Press Refresh to ask this printer how flat its bed is"))

    def _say_values(self, p, where: QRect):
        """Every probed value in the middle of the part of the bed it stands for, in black or
        white ink: whichever of the two stands out against the colour that is painted there."""
        probed = self._probed()
        if probed is None:
            return
        rows, cols = probed.shape
        for row in range(rows):
            for col in range(cols):
                cell = self._cell(where, rows, cols, row, col)
                if cell.width() < CELL_TEXT or cell.height() < CELL_TEXT // 2:
                    continue                # the colour of this point says more than ink would
                p.setPen(QPen(self._back_at(cell)))
                p.drawText(cell, int(Qt.AlignCenter), f"{float(probed[row][col]):.2f}")

    def _back_at(self, cell: QRect) -> QColor:
        """The ink to write a value with at this place of the map."""
        where, grid = self.place(), self._grid()
        at = self._under(cell.center().x(), cell.center().y(), where, grid)
        if at is None:
            return _ink(QColor(*bed_mesh.MID))
        return _ink(QColor(*bed_mesh.colour(float(grid[at[0]][at[1]]), self.lo, self.hi)))

    # ------------------------------------------------------------------------- the mouse
    def mouseMoveEvent(self, event):
        self._show_where(event.position().x(), event.position().y())

    def leaveEvent(self, event):
        """The mouse is gone: whatever it stood over is said no longer."""
        self._show_where(None, None)
        super().leaveEvent(event)

    def _show_where(self, x, y):
        """Say which cell the mouse stands over - and only when that changed, because one move
        over one cell is a hundred events that all say the same."""
        at = None if x is None else self.cell_at(x, y)
        if at == self._over:
            return
        self._over = at
        self.hovered.emit(-1 if at is None else at[0], -1 if at is None else at[1])


class Legend(QWidget):
    """The bar under the picture: which colour stands for which z.

    The bar spans exactly what the map spans, so the white of 0 mm stands where it belongs
    between the lowest and the highest point of the bed; under the bar stand both ends and 0."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lo = self.hi = 0.0
        self.setFixedHeight(38)
        self.setAccessibleName(tr("The colours of the heat map"))

    def set_range(self, lo, hi):
        """What the colours of the picture span; a bar with nothing to colour has nothing to say
        and hides itself."""
        self.lo, self.hi = float(lo), float(hi)
        self.setVisible(self.hi > self.lo)
        self.update()

    def paintEvent(self, event):
        if self.hi <= self.lo:
            return
        metrics = self.fontMetrics()
        bar = QRect(MARGIN, 1, max(1, self.width() - 2 * MARGIN),
                    max(1, self.height() - metrics.height() - 4))
        shade = QLinearGradient(bar.left(), 0, bar.right(), 0)
        shade.setColorAt(0.0, QColor(*bed_mesh.colour(self.lo, self.lo, self.hi)))
        middle = min(1.0, max(0.0, (0.0 - self.lo) / (self.hi - self.lo)))
        shade.setColorAt(middle, QColor(*bed_mesh.MID))
        shade.setColorAt(1.0, QColor(*bed_mesh.colour(self.hi, self.lo, self.hi)))
        p = QPainter(self)
        p.setPen(QPen(QColor(theme.color("muted"))))
        p.setBrush(QBrush(shade))
        p.drawRect(bar)
        self._say_over(p, bar, metrics)
        p.end()

    def _say_over(self, p, bar: QRect, metrics):
        """The three numbers under the bar, each over the place it stands for: the lowest and the
        highest point at both ends, and 0 where 0 falls on the bar."""
        p.setPen(QPen(self.palette().color(QPalette.WindowText)))
        wide = metrics.horizontalAdvance("-0.000") + 6
        zero = bar.left() + int(bar.width() * (0.0 - self.lo) / (self.hi - self.lo))
        says = [(self.lo, bar.left(), 1), (self.hi, bar.right(), -1)]
        if bar.width() > 3 * wide:              # three numbers on a narrow bar are one mess
            says.insert(1, (0.0, zero, 0))
        for value, at, edge in says:
            left = at if edge > 0 else (at - wide if edge < 0 else at - wide // 2)
            p.drawText(QRect(int(left), bar.bottom() + 2, wide, metrics.height()),
                       int(Qt.AlignLeft | Qt.AlignTop), _mm(value))


class BedMeshWindow(QWidget):
    """Window > Bed Mesh…: the mesh of a Klipper printer's bed, painted and read point by point.

    Modeless, like Window > Printer Status: looking at a bed may not stop you from working on the
    model. Volura keeps one of these; calling it again brings the window that is already there to
    the front. A mesh belongs to the printer it came from, so it goes when that printer is gone
    from Preferences or another one is chosen; a window that comes back to the printer it had,
    shows the mesh it has and asks nothing again.
    """

    def __init__(self, win):
        super().__init__(win, Qt.Window)
        self.win = win
        self.conns = []                   # the Klipper printers under Preferences
        self.mesh = None                  # what the printer in the box said about its bed
        self._job = None                  # the asking on its way, so never two at once
        self._seen = ()                   # the printer the screen speaks of
        self.setWindowTitle(tr("Bed Mesh"))
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.printer = QComboBox()
        self.printer.setToolTip(tr("The Klipper (Moonraker) printers under Window > Preferences "
                                   "> Printers"))
        self.printer.currentIndexChanged.connect(self._chosen)
        top.addWidget(self.printer, 1)
        self.refresh = QPushButton(tr("Refresh"))
        self.refresh.setToolTip(tr("Ask this printer for the mesh of its bed again"))
        self.refresh.clicked.connect(self.ask)
        top.addWidget(self.refresh)
        lay.addLayout(top)
        self.needs = QLabel(NEEDS_KLIPPER)
        self.needs.setWordWrap(True)
        self.needs.setEnabled(False)
        self.needs.hide()
        lay.addWidget(self.needs)
        self.view = BedView(self)
        lay.addWidget(self.view, 1)
        self.legend = Legend(self)
        self.legend.hide()                # comes up with the first mesh that spans something
        lay.addWidget(self.legend)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.color('muted')};")
        self.error.hide()
        lay.addWidget(self.error)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setEnabled(False)
        lay.addWidget(self.status)
        self.view.hovered.connect(self._hovered)
        self.reload()

    # -------------------------------------------------------------------------- the printers
    def klipper_printers(self) -> list:
        """The printers under Preferences that can have a bed mesh: the Moonraker ones."""
        return [conn for conn in list(self.win.settings.printers) if conn.kind == "moonraker"]

    def connection(self):
        """The printer in the box; None when there is no Klipper printer to choose from."""
        return self.printer.currentData()

    def reload(self):
        """The printers under Preferences can have changed: the box and what is shown follow.

        Nothing is asked here - the window asks when it comes on the screen and when *Refresh* is
        pressed - and a mesh of another printer than the one in the box is taken away, because it
        says something about another bed than the one being looked at."""
        conns = self.klipper_printers()
        if conns != self.conns:
            was = key_of(self.connection())
            self.conns = conns
            self.printer.blockSignals(True)
            self.printer.clear()
            for conn in conns:
                self.printer.addItem(conn.name, conn)
                self.printer.setItemData(self.printer.count() - 1,
                                         printer_link.normalise_url(conn.url), Qt.ToolTipRole)
            back = next((one for one, conn in enumerate(conns) if key_of(conn) == was), 0)
            self.printer.setCurrentIndex(back if conns else -1)
            self.printer.blockSignals(False)
        self.needs.setVisible(not conns)
        for widget in (self.printer, self.refresh, self.view, self.summary):
            widget.setVisible(bool(conns))
        if self._seen != key_of(self.connection()):
            self._clear()

    def _chosen(self):
        """Another printer in the box: what the previous one said is nothing to see here, and
        this one has not been asked yet."""
        if key_of(self.connection()) == self._seen:
            return
        self._clear()
        self.ask()

    # ---------------------------------------------------------------------------- the asking
    def ask(self):
        """Ask this printer for the mesh of its bed, in the background and without a dialog of
        its own.

        A printer whose answer has not come back yet is not asked a second time, and a printer
        that is not in the box - or no printer at all, when none is to be chosen from - is asked
        nothing."""
        conn = self.connection()
        if conn is None or (self._job is not None and not self._job.done):
            return
        self._seen = key_of(conn)           # what an answer on its way will be the mesh of

        def work(report):
            return printer_link.bed_mesh(conn)

        self._job = background.run(None, work, text=tr("Speaking to the printer…"),
                                   can_cancel=False, show_dialog=False,
                                   on_done=self.answered, on_error=self.failed)

    def stop(self):
        """What is on its way may come back to nothing: the window is out of use."""
        if self._job is not None and not self._job.done:
            self._job.cancel()
        self._job = None

    def answered(self, mesh):
        """What the printer says about its bed: painted, summed up, with nothing left of
        whatever went wrong the last time."""
        if self._job is None:
            return                       # stopped while it was on its way: nothing to show
        self._job = None
        self._forget()
        if mesh is None:
            self._say(NO_MESH)            # a bed nobody probed is nothing to paint, not an error
            return
        self.mesh = mesh
        lo, hi = bed_mesh.limits(mesh.picture())
        self.view.set_mesh(mesh, lo, hi)
        self.legend.set_range(lo, hi)
        self.summary.setText(bed_mesh.summary(mesh))
        self._say("")

    def failed(self, exc):
        """What went wrong stands muted under the picture; the picture of the last good answer
        stays, because a bed does not change between two asking."""
        if self._job is None:
            return
        self._job = None
        self._say(str(exc))

    def _say(self, text=""):
        """The last thing that went wrong, muted under the picture; nothing has nothing to say."""
        self.error.setText(str(text or ""))
        self.error.setVisible(bool(text))

    def _forget(self):
        """Nothing of a mesh on the screen: the picture, its bar and what was said of it."""
        self.mesh = None
        self.view.set_mesh(None)
        self.legend.set_range(0.0, 0.0)
        self.summary.setText("")
        self.status.setText("")

    def _clear(self):
        """Nothing at all is spoken of: another printer is in the box, or none, and what was on
        its way to the printer of before has nothing to come back to."""
        self.stop()
        self._seen = ()
        self._forget()
        self._say("")

    # ---------------------------------------------------------------------------- the mouse
    def _hovered(self, row, col):
        """Where the mouse stands on the bed, in the line at the bottom; -1 is no mouse at all."""
        point = None if row < 0 or self.mesh is None else self.mesh.point(row, col)
        if point is None:
            self.status.setText("")
            return
        x, y, z = point
        self.status.setText(tr("x {x} mm, y {y} mm, z {z} mm").format(
            x=f"{x:.1f}", y=f"{y:.1f}", z=_mm(z)))

    # ------------------------------------------------------------------ on and off the screen
    def showEvent(self, event):
        """Back on the screen: the box follows Preferences, and a bed that was not asked about
        yet is asked now - and only now, because a bed mesh does not change by itself."""
        super().showEvent(event)
        self.reload()
        if self.mesh is None and self._job is None:
            self.ask()

    def hideEvent(self, event):
        """Off the screen or closed: nothing waits for an answer any more."""
        self.stop()
        super().hideEvent(event)
