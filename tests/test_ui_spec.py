"""Checks that Volura's menus, shortcuts, toolbar and tray follow tests/ui_spec.py.

Runs without a display (Qt offscreen platform).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QToolBar, QToolButton  # noqa: E402

import ui_spec  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def shortcut_of(action):
    seqs = [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]
    return seqs


@pytest.fixture(scope="module")
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def menu_items(menu):
    out = []
    for a in menu.actions():
        if a.isSeparator():
            out.append("---")
        elif a.menu() is not None:
            out.append((clean(a.text()) + " >", a))
        else:
            out.append((clean(a.text()), a))
    # collapse double separators / leading separators (Qt hides them anyway)
    res = []
    for it in out:
        if it == "---" and (not res or res[-1] == "---"):
            continue
        res.append(it)
    while res and res[-1] == "---":
        res.pop()
    return res


def top_menus(win):
    return {clean(a.text()): a.menu() for a in win.menuBar().actions() if a.menu() is not None}


def check_items(menu, spec, where):
    items = menu_items(menu)
    labels = [i if i == "---" else i[0] for i in items]
    want = [i if i == "---" else i[0] for i in spec]
    # every spec item must appear, in order (extra Volura items are allowed in between)
    pos = 0
    for w in want:
        if w == "---":
            continue
        assert w in labels[pos:], f"{where}: '{w}' ontbreekt of staat niet op de juiste plek (na positie {pos}); " \
                                  f"menu is nu {labels}"
        pos = labels.index(w, pos) + 1
    for s in spec:
        if s == "---" or s[1] is None:
            continue
        act = next(i[1] for i in items if i != "---" and i[0] == s[0])
        assert s[1] in shortcut_of(act), f"{where}: '{s[0]}' moet sneltoets {s[1]} hebben, heeft {shortcut_of(act)}"


def test_menu_bar_order(win):
    titles = [clean(a.text()) for a in win.menuBar().actions() if a.menu() is not None]
    assert titles == [m for m, _ in ui_spec.MENUS]


@pytest.mark.parametrize("title", [m for m, _ in ui_spec.MENUS])
def test_menu(win, title):
    spec = dict(ui_spec.MENUS)[title]
    check_items(top_menus(win)[title], spec, title)


@pytest.mark.parametrize("key", list(ui_spec.SUBMENUS))
def test_submenu(win, key):
    title, sub = key
    menu = top_menus(win)[title]
    act = next((a for a in menu.actions() if a.menu() is not None and clean(a.text()) == sub), None)
    assert act is not None, f"submenu {title} > {sub} ontbreekt"
    check_items(act.menu(), ui_spec.SUBMENUS[key], f"{title} > {sub}")


def test_extra_sketchup_shortcuts(win):
    """SketchUp also accepts Ctrl+Z (undo) and Shift+Z (zoom extents)."""
    all_actions = win.findChildren(type(win.menuBar().actions()[0]))
    shortcuts = {s for a in all_actions for s in shortcut_of(a)}
    for s in ("Ctrl+Z", "Shift+Z", "Ctrl+X"):
        assert s in shortcuts, f"sneltoets {s} ontbreekt"


def test_no_duplicate_shortcuts(win):
    seen = {}
    for a in win.findChildren(type(win.menuBar().actions()[0])):
        for s in shortcut_of(a):
            if s in seen and seen[s] is not a:
                pytest.fail(f"sneltoets {s} zit op '{clean(seen[s].text())}' en '{clean(a.text())}'")
            seen[s] = a


def test_getting_started_toolbar(win):
    bars = [b for b in win.findChildren(QToolBar) if b.isVisibleTo(win) or not b.isHidden()]
    main = [b for b in bars if win.toolBarArea(b).name == "TopToolBarArea"]
    assert main, "geen werkbalk bovenaan"
    tb = main[0]
    labels = []
    for a in tb.actions():
        if a.isSeparator():
            labels.append("|")
        else:
            labels.append(clean(a.text()))
    while labels and labels[0] == "|":
        labels.pop(0)
    assert labels == ui_spec.TOOLBAR, f"werkbalk is {labels}"


def test_toolbar_dropdowns(win):
    tb = [b for b in win.findChildren(QToolBar) if win.toolBarArea(b).name == "TopToolBarArea"][0]
    for label, tools in ui_spec.TOOLBAR_DROPDOWNS.items():
        act = next(a for a in tb.actions() if clean(a.text()) == label)
        btn = tb.widgetForAction(act)
        assert isinstance(btn, QToolButton) and btn.menu() is not None, f"'{label}' heeft geen uitklapmenu"
        got = [clean(a.text()) for a in btn.menu().actions() if not a.isSeparator()]
        assert got == tools, f"uitklapmenu '{label}' is {got}"


def test_no_left_toolbar_by_default(win):
    for b in win.findChildren(QToolBar):
        if win.toolBarArea(b).name == "LeftToolBarArea":
            assert b.isHidden(), "SketchUp 2017 toont standaard geen gereedschapsbalk links"


def test_default_tray(win):
    docks = [d for d in win.findChildren(QDockWidget) if not d.isHidden()]
    assert [clean(d.windowTitle()) for d in docks] == [ui_spec.TRAY_TITLE], \
        f"er moet precies één paneel '{ui_spec.TRAY_TITLE}' zijn, nu: {[d.windowTitle() for d in docks]}"
    tray = docks[0]
    titles = getattr(tray, "panel_titles", None)
    assert titles is not None, "het standaardvenster moet een attribuut panel_titles hebben (lijst met titels)"
    assert list(titles) == ui_spec.TRAY_PANELS


def test_measurements_box(win):
    assert clean(win.vcb_label.text()) == ui_spec.MEASUREMENTS_LABEL_DEFAULT
