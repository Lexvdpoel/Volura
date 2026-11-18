"""The workspaces at the bottom of the viewer: Design, Laser and CNC.

A workspace says what the window is for right now. **Design** is drawing and modelling,
with the 3D model and the 2D worksheet behind the 3D | 2D switch. **Laser** is the
worksheet and the laser on its cable next to it; a laser cuts flat, so it is only 2D and
the switch goes away. **CNC** is a mill: a gantry router cuts the worksheet like a laser
does, but a mill with more axes cuts the model itself, so CNC has both views again, with
the machine on its cable next to them.

This file only holds the bar with the tabs; what a workspace shows is decided by the main
window (`MainWindow.set_workspace`).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QTabBar, QWidget

from .i18n import tr

DESIGN = "design"
LASER = "laser"
CNC = "cnc"

#: the tabs, in their order: key, name, what the tab is for
WORKSPACES = (
    (DESIGN, tr("Design"), tr("Model and draw: the 3D model or the 2D worksheet")),
    (LASER, tr("Laser"), tr("The worksheet and the laser on its cable; a laser cuts flat, "
                           "so only 2D")),
    (CNC, tr("CNC"), tr("A gantry router or a multi-axis mill: the worksheet or the model, "
                       "and the machine on its cable")),
)

#: whether the 3D | 2D switch belongs to a workspace
HAS_3D = {DESIGN: True, LASER: False, CNC: True}

#: the workspaces with a machine on a cable next to the view
MACHINES = (LASER, CNC)


def normal(key) -> str:
    """A known workspace; anything else is Design."""
    key = str(key or "").lower()
    return key if key in HAS_3D else DESIGN


class WorkspaceBar(QWidget):
    """The row under the viewer: the tabs of the workspaces and the 3D | 2D switch."""

    #: the user chose another tab; the key of the workspace
    changed = Signal(str)

    def __init__(self, mode_switch=None, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 6, 2)
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
        lay.addStretch(1)
        self.mode_switch = mode_switch
        if mode_switch is not None:
            lay.addWidget(mode_switch)
        self.tabs.currentChanged.connect(lambda _i: self.changed.emit(self.key()))

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
