"""Every control a keyboard or a screen reader can reach has a name.

Qt reads a widget's text out loud, so a button that shows only an icon, a colour swatch, a
spin box without a label next to it or a widget that paints itself has nothing to read. Those
get an `accessibleName`, and this test keeps it that way: it walks the window with its docks
and the dialogs of Draw > Parts and File > Slice, and reports every reachable control that has
neither an accessible name nor a text, a placeholder, a label, a buddy or a tooltip.

Run with:  python -m pytest tests/test_accessibility.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QDialog, QDockWidget, QFormLayout, QLabel,
                               QLineEdit, QApplication, QMenu, QStackedWidget, QToolButton,
                               QWidget)  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import ops, slicer  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])
ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def widgets_of(root):
    """Every widget under `root`, also the ones inside its dock widgets."""
    found = list(root.findChildren(QWidget))
    for dock in root.findChildren(QDockWidget):
        found += dock.findChildren(QWidget)
    seen, out = set(), []
    for w in found:
        if id(w) not in seen:
            seen.add(id(w))
            out.append(w)
    return out


def shown(w, root):
    """Would `w` be seen when its window is shown? A page of a stacked widget that is not the
    current one counts: it is hidden only because another page stands on top of it."""
    while w is not None and w is not root:
        if w.isHidden() and not isinstance(w.parentWidget(), QStackedWidget):
            return False
        w = w.parentWidget()
    return w is root


def reachable(w, root):
    """Can a keyboard or a screen reader reach this control?"""
    if isinstance(w, (QMenu, QDockWidget)):
        return False
    if not (w.focusPolicy() != Qt.NoFocus or isinstance(w, QAbstractButton)):
        return False
    return shown(w, root)


def item_widgets(item):
    """The widgets of a layout item, also those of the layouts inside it."""
    if item is None:
        return []
    if item.widget() is not None:
        return [item.widget()] + item.widget().findChildren(QWidget)
    if item.layout() is None:
        return []
    out = []
    for i in range(item.layout().count()):
        out += item_widgets(item.layout().itemAt(i))
    return out


def form_fields(root):
    """The widgets that a QFormLayout gives a label: the field of every row with a label."""
    out = []
    for form in root.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            if label is None or label.widget() is None or not clean(label.widget().text()):
                continue
            out += item_widgets(form.itemAt(row, QFormLayout.FieldRole))
    return out


def named(w, widgets):
    """Does a screen reader have anything to read for this control?"""
    if w.accessibleName().strip() or w.toolTip().strip():
        return True
    if isinstance(w, QAbstractButton) and clean(w.text()):
        return True
    if isinstance(w, QToolButton) and w.defaultAction() is not None:
        if clean(w.defaultAction().text()) or w.defaultAction().toolTip().strip():
            return True
    if isinstance(w, QLineEdit) and w.placeholderText().strip():
        return True
    if isinstance(w, QLabel) and clean(w.text()):
        return True
    return any(isinstance(other, QLabel) and other.buddy() is w and clean(other.text())
               for other in widgets)


def unnamed(root):
    """The reachable controls of `root` that have no name for a screen reader."""
    widgets = widgets_of(root)
    labelled = {id(w) for w in form_fields(root)}
    return [w for w in widgets
            if reachable(w, root) and id(w) not in labelled and not named(w, widgets)]


def describe(w):
    """Class, object name and parent: enough to find the control again in the code."""
    parent = w.parentWidget()
    where = type(parent).__name__ if parent is not None else "nothing"
    if parent is not None and parent.objectName():
        where += f" '{parent.objectName()}'"
    return f"{type(w).__name__} '{w.objectName()}' in {where}"


def report(root):
    """What this test has to complain about, one line per control."""
    return "\n".join(describe(w) for w in unnamed(root))


def open_tray(win):
    """Every panel of the Default Tray open, so its controls can be reached too."""
    for title in list(win.tray.sections):
        win.tray.open_panel(title)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """The window as the user sees it: shown, with every panel of the Default Tray open."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    w.show()
    open_tray(w)
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def menu_named(parent, title):
    """The submenu of `parent` that is called `title`, or None."""
    for a in parent.actions():
        if a.menu() is not None and clean(a.text()) == title:
            return a.menu()
    return None


@pytest.mark.parametrize("mode", ("3d", "2d"))
def test_the_window_and_its_docks_have_named_controls(win, mode):
    """Every button, field and view tells a screen reader what it is, in the model as well
    as on the worksheet."""
    win.set_mode(mode)
    open_tray(win)
    QCoreApplication.processEvents()
    problems = report(win)
    assert not problems, problems


def test_the_dialogs_of_draw_parts_have_named_controls(win, monkeypatch):
    """Every dialog of Draw > Parts, opened the way the window opens it."""
    parts = menu_named(menu_named(win.menuBar(), "Draw"), "Parts")
    assert parts is not None and len(parts.actions()) > 10, "Draw > Parts is not there"
    actions = [a for a in parts.actions() if not a.isSeparator()]
    opened, problems = [], []

    def inspect(dlg):
        opened.append(dlg)
        what = report(dlg)
        if what:
            problems.append(f"{clean(dlg.windowTitle())}:\n{what}")
        dlg.close()
        return QDialog.Rejected

    monkeypatch.setattr(app_module.QDialog, "exec", inspect)
    for action in actions:
        action.trigger()
    QCoreApplication.processEvents()
    assert len(opened) == len(actions), "not every part opened a dialog"
    assert not problems, "\n".join(problems)


@needs_engine
def test_the_slice_dialog_has_named_controls(win, monkeypatch):
    """File > Slice…, opened the way tests/test_slicer.py opens it."""
    import volura.slice_dialog as sd
    ops.group_from_manifold(win.doc, mf.Manifold.cube((15.0, 15.0, 15.0)), (0.8, 0.8, 0.8),
                            "cube")
    found = []

    def inspect(dlg):
        found.append(report(dlg))
        dlg.close()
        return QDialog.Rejected

    monkeypatch.setattr(sd.SliceDialog, "exec", inspect)
    win.slice_dialog()
    deadline = time.monotonic() + 120.0
    while not found and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert found, "the slice dialog did not open"
    assert not found[0], found[0]



def test_the_resin_dialog_has_named_controls(win, monkeypatch):
    """File > Slice for Resin…, with the resin profiles when they are there."""
    from volura import sla_dialog, sla_profiles
    folder = sla_profiles.find_uvtools()
    if folder is None:
        pytest.skip("no resin profiles (tools/fetch_uvtools.py)")
    dlg = sla_dialog.ResinDialog(win, sla_profiles.ResinLibrary(folder))
    try:
        dlg.show()
        _app.processEvents()
        problems = report(dlg)
        assert not problems, problems
    finally:
        dlg.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
