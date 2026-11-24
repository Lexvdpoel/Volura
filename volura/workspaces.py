"""The workspaces at the bottom of the viewer: Design, Print, Laser and CNC.

A workspace says what the window is for right now. **Design** is drawing and modelling,
with the 3D model and the 2D worksheet behind the 3D | 2D switch. **Print** is the model
on the bed of a 3D printer, filament (FDM) or resin (SLA): only 3D, with the commands of
the printer in a toolbar of their own. **Laser** is the
worksheet and the laser on its cable next to it; a laser cuts flat, so it is only 2D and
the switch goes away. **CNC** is a mill: a gantry router cuts the worksheet like a laser
does, but a mill with more axes cuts the model itself, so CNC has both views again, with
the machine on its cable next to them.

This file only holds the bar with the tabs; what a workspace shows is decided by the main
window (`MainWindow.set_workspace`).
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QTabBar, QWidget

from .i18n import tr

DESIGN = "design"
PRINT = "print"
LASER = "laser"
CNC = "cnc"

#: the tabs, in their order: key, name, what the tab is for
WORKSPACES = (
    (DESIGN, tr("Design"), tr("Model and draw: the 3D model or the 2D worksheet")),
    (PRINT, tr("Print"), tr("The model on the bed of a 3D printer, filament or resin: "
                           "check, orient and slice; only 3D")),
    (LASER, tr("Laser"), tr("The worksheet and the laser on its cable; a laser cuts flat, "
                           "so only 2D")),
    (CNC, tr("CNC"), tr("A gantry router or a multi-axis mill: the worksheet or the model, "
                       "and the machine on its cable")),
)

#: the views of a workspace, the first one where it starts; the 3D | 2D switch is only
#: there for a workspace with both
VIEWS = {DESIGN: ("3d", "2d"), PRINT: ("3d",), LASER: ("2d",), CNC: ("3d", "2d")}

#: the workspaces with a machine on a cable next to the view
MACHINES = (LASER, CNC)


def normal(key) -> str:
    """A known workspace; anything else is Design."""
    key = str(key or "").lower()
    return key if key in VIEWS else DESIGN


def has_switch(key) -> bool:
    """Whether a workspace shows the 3D | 2D switch: only when it has both views."""
    return len(VIEWS[normal(key)]) > 1


class WorkspaceBar(QWidget):
    """The row under the viewer: the tabs of the workspaces and the 3D | 2D switch."""

    #: the user chose another tab; the key of the workspace
    changed = Signal(str)

    def __init__(self, mode_switch=None, parent=None, hint=None, extras=(), before_switch=()):
        """`hint` is the text of the window, beside the tabs and taking the free room;
        `before_switch` stand just left of the 3D | 2D switch and take their room from the hint,
        so they can never cover the switch (the cursor position of the worksheet);
        `extras` stand at the right end, after the 3D | 2D switch, all on the one row."""
        super().__init__(parent)
        self.setObjectName("workspaceBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 2)
        lay.setSpacing(8)
        self.tabs = QTabBar()
        self.tabs.setObjectName("workspaceTabs")
        self.tabs.setShape(QTabBar.RoundedSouth)
        self.tabs.setDocumentMode(True)
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setAccessibleName(tr("Workspace"))
        for key, label, tip in WORKSPACES:
            i = self.tabs.addTab(label)
            self.tabs.setTabData(i, key)
            self.tabs.setTabToolTip(i, tip)
        lay.addWidget(self.tabs)
        if hint is not None:
            lay.addWidget(hint, 1)
        else:
            lay.addStretch(1)
        for w in before_switch:
            lay.addWidget(w)
        self.mode_switch = mode_switch
        if mode_switch is not None:
            lay.addWidget(mode_switch)
        for w in extras:
            lay.addWidget(w)
        self.tabs.currentChanged.connect(lambda _i: self.changed.emit(self.key()))
        self._viewer = None

    def follow(self, viewer) -> None:
        """End where `viewer` ends, so the 3D | 2D switch lines up with its right edge
        also when a side bar or the machine panel next to it is made wider."""
        self._viewer = viewer
        viewer.installEventFilter(self)
        QTimer.singleShot(0, self, self._align)   # not after self is gone

    def eventFilter(self, obj, event):
        if obj is self._viewer and event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            QTimer.singleShot(0, self, self._align)   # not after self is gone
        return False

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self, self._align)   # not after self is gone

    def _align(self) -> None:
        viewer = self._viewer
        if viewer is None or not viewer.isVisible():
            return
        top = self.window()
        right = viewer.mapTo(top, QPoint(viewer.width(), 0)).x()
        left = self.mapTo(top, QPoint(0, 0)).x()
        if right - left > self.minimumSizeHint().width() and right - left != self.width():
            self.setFixedWidth(right - left)

    def key(self) -> str:
        return normal(self.tabs.tabData(self.tabs.currentIndex()))

    def set_key(self, key) -> None:
        """Show `key` as the chosen tab without saying so: the window already knows."""
        key = normal(key)
        for i in range(self.tabs.count()):
            if self.tabs.tabData(i) == key:
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(i)
                self.tabs.blockSignals(False)
                return
