"""The SketchUp 2017 "Default Tray": one dock on the right with collapsible panels.

Every panel is a :class:`TraySection`: a full width header bar with a ``>``/``v``
arrow and the content below it. Clicking the header bar collapses or expands that
panel; several panels can be open at the same time.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDockWidget, QSizePolicy, QToolButton, QVBoxLayout, QWidget

TRAY_TITLE = "Standaardvenster"
TRAY_WIDTH = 300

HEADER_STYLE = """
QToolButton#trayHeader {
    text-align: left;
    padding: 5px 4px;
    border: none;
    border-top: 1px solid #d8d8d8;
    border-bottom: 1px solid #c4c4c4;
    background: #ededed;
    font-weight: bold;
}
QToolButton#trayHeader:hover { background: #e2e2e2; }
"""


class TraySection(QWidget):
    """One collapsible panel of the Default Tray."""

    toggled = Signal(str, bool)

    def __init__(self, title, widget, is_open=True, parent=None):
        super().__init__(parent)
        self.title = title
        self.header = QToolButton(self)
        self.header.setObjectName("trayHeader")
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setAutoRaise(True)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.clicked.connect(self.toggle)

        self.body = widget
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self._apply(is_open)

    def is_open(self):
        return not self.body.isHidden()

    def set_open(self, is_open):
        is_open = bool(is_open)
        if is_open == self.is_open():
            return
        self._apply(is_open)
        self.toggled.emit(self.title, is_open)

    def toggle(self):
        self.set_open(not self.is_open())

    def _apply(self, is_open):
        self.header.setChecked(is_open)
        self.header.setArrowType(Qt.DownArrow if is_open else Qt.RightArrow)
        self.body.setVisible(is_open)


class Tray(QDockWidget):
    """The Default Tray: a single dock with the collapsible panels in SketchUp order.

    `extra_panels` are Volura panels that are not part of the SketchUp tray: they are
    created closed and are not listed in `panel_titles`, so the SketchUp tray stays as
    it is recorded in tests/ui_spec.py."""

    panel_toggled = Signal(str, bool)

    def __init__(self, panels, open_panels=None, parent=None, extra_panels=()):
        super().__init__(TRAY_TITLE, parent)
        self.setObjectName("defaulttray")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
                         | QDockWidget.DockWidgetFloatable)
        self.panel_titles = [title for title, _ in panels]
        self.extra_titles = [title for title, _ in extra_panels]
        default_open = self.panel_titles if open_panels is None else list(open_panels)

        content = QWidget()
        content.setObjectName("trayContent")
        content.setStyleSheet(HEADER_STYLE)
        content.setMinimumWidth(TRAY_WIDTH - 40)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.sections = {}
        for title, widget in list(panels) + list(extra_panels):
            start_open = title in default_open and title not in self.extra_titles
            section = TraySection(title, widget, start_open)
            section.toggled.connect(self.panel_toggled.emit)
            self.sections[title] = section
            lay.addWidget(section)
        lay.addStretch(1)
        self.setWidget(content)

    def section(self, title):
        return self.sections[title]

    def is_open(self, title):
        return self.section(title).is_open()

    def open_panel(self, title):
        self.section(title).set_open(True)

    def close_panel(self, title):
        self.section(title).set_open(False)

    def set_panel_open(self, title, is_open):
        self.section(title).set_open(is_open)

    def set_panel_shown(self, title, shown):
        """Een paneel helemaal weglaten (kop en inhoud), bijvoorbeeld in de andere modus."""
        self.section(title).setVisible(bool(shown))

    def is_shown(self, title):
        return not self.section(title).isHidden()

    def toggle_panel(self, title):
        self.section(title).toggle()
