"""Modelinfo, Voorkeuren (met aanpasbare sneltoetsen) en het welkomstscherm.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton, QApplication  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import tools  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

KEYS = ("length_snap", "length_precision", "circle_segments", "bed_size", "show_welcome")


def settings():
    return QSettings("Volura", "Volura")


def shortcut_keys():
    return [k for k in settings().allKeys() if k.startswith("shortcuts/")]


def shortcuts_of(action):
    return [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


@pytest.fixture(scope="module", autouse=True)
def restore_settings():
    """QSettings and the length precision belong to the user, not to the tests."""
    q = settings()
    saved = {k: q.value(k) for k in KEYS}
    yield
    for k in shortcut_keys():
        q.remove(k)
    for k, v in saved.items():
        if v is None:
            q.remove(k)
        else:
            q.setValue(k, v)
    q.sync()
    tools.set_length_precision(int(saved["length_precision"] or 2))


@pytest.fixture(autouse=True)
def clean_shortcuts(win):
    """No half-assigned shortcut may leak into another test (or another run)."""
    win.restore_shortcuts()
    yield
    win.restore_shortcuts()


@pytest.fixture(scope="module")
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture
def fresh():
    """Makes the window the user sees after a restart; closed after the test."""
    from volura.app import MainWindow
    made = []

    def make():
        w = MainWindow()
        made.append(w)
        return w
    yield make
    for w in made:
        w.doc.dirty = False
        w.close()


class FakeBox:
    """Stands in for QMessageBox, so a test can answer a question without a dialog."""

    Yes, No = 1, 0

    def __init__(self, answer=1):
        self.answer = answer
        self.asked = []

    def question(self, *args, **kwargs):
        self.asked.append(args)
        return self.answer


class FakeDialog:
    """Stands in for one of the dialogs, so a test can see what is opened on which page."""

    def __init__(self, opened, name):
        self.opened = opened
        self.name = name

    def __call__(self, window, page=0):
        self.opened.append((self.name, page))
        return self

    def exec(self):
        return 0


# ---------------------------------------------------------------- the dialogs themselves

def test_dialogs_can_be_made_without_showing(win):
    from volura.app import ModelInfoDialog, PreferencesDialog, WelcomeDialog
    dialogs = (ModelInfoDialog(win), PreferencesDialog(win), WelcomeDialog(win))
    for dlg in dialogs:
        assert not dlg.isVisible(), f"{dlg.windowTitle()} staat op zichzelf, maar wordt niet getoond"
        assert dlg.windowTitle()
    assert ModelInfoDialog(win).page_titles == ["Eenheden", "Statistieken", "Printbed"]
    assert PreferencesDialog(win).page_titles == ["Algemeen", "Tekenen", "Sneltoetsen"]


def test_model_info_has_a_page_list_and_a_page_per_title(win):
    from volura.app import ModelInfoDialog
    dlg = ModelInfoDialog(win)
    assert [dlg.pages_list.item(i).text() for i in range(dlg.pages_list.count())] == dlg.page_titles
    assert dlg.stack.count() == len(dlg.page_titles)
    for i, title in enumerate(dlg.page_titles):
        dlg.select_page(title)
        assert dlg.pages_list.currentRow() == i and dlg.stack.currentIndex() == i


def test_model_info_statistics(win):
    from volura import ops
    from volura.app import ModelInfoDialog
    doc = win.doc
    before = len(doc.groups)
    g = ops.group_from_manifold(doc, ops.primitive("box", 20, 20, 20), "#d23b3b", "Blok")
    try:
        dlg = ModelInfoDialog(win)
        assert dlg.stats["Groepen"].text() == str(before + 1)
        assert dlg.stats["Losse vlakken"].text() == str(len(doc.root.F))
        assert dlg.stats["Losse randen"].text() == str(len(doc.root.E))
        assert dlg.stats["Volume in totaal"].text().startswith("8.000"), \
            f"een blok van 20×20×20 mm is 8 cm³, nu {dlg.stats['Volume in totaal'].text()}"
    finally:
        doc.remove_group(g.id)


def test_model_info_printbed_preset_sets_the_bed(win):
    from volura.app import ModelInfoDialog
    was = win.settings.bed_size
    dlg = ModelInfoDialog(win)
    dlg.printer.setCurrentIndex(1)
    size = tuple(float(v) for v in dlg.printer.itemData(1))
    assert tuple(sp.value() for sp in dlg.bed_spins) == size
    dlg.accept()
    try:
        assert win.settings.bed_size == size
        assert tuple(type(win.settings)().bed_size) == size
    finally:
        win.settings.bed_size = was
        win.settings.save()


def test_preferences_general_and_drawing(win):
    from volura.app import PreferencesDialog
    was = win.settings.circle_segments
    dlg = PreferencesDialog(win)
    assert dlg.segments.value() == was
    dlg.segments.setValue(24)
    dlg.accept()
    try:
        assert win.settings.circle_segments == 24
        assert win.view.tools["circle"].segments == 24
        assert type(win.settings)().circle_segments == 24
        again = PreferencesDialog(win)
        assert again.segments.value() == 24
        assert again.split_lines.isChecked() and not again.split_lines.isEnabled(), \
            "Lijnen automatisch splitsen staat vast aan en is uitgeschakeld"
    finally:
        win.settings.circle_segments = was
        win.view.tools["circle"].segments = was
        win.settings.save()


def test_welcome_screen_looks_like_sketchup(win):
    from volura.app import WelcomeDialog
    dlg = WelcomeDialog(win)
    text = " ".join(l.text() for l in dlg.findChildren(QLabel))
    assert "Volura" in text
    for step in ("Teken een vorm", "Duwen/trekken", "Exporteer"):
        assert step in text, f"stap {step} ontbreekt in het welkomstscherm"
    assert "Begin met Volura" in [b.text() for b in dlg.findChildren(QPushButton)]
    assert "Altijd tonen bij opstarten" in dlg.show_at_start.text()


# ---------------------------------------------------------------- units and fmt_len

def test_fmt_len_respects_the_precision():
    was = tools.LENGTH_PRECISION
    try:
        for decimals, want in ((0, "12 mm"), (1, "12.3 mm"), (2, "12.35 mm"), (3, "12.346 mm")):
            tools.set_length_precision(decimals)
            assert tools.fmt_len(12.3456) == want, f"{decimals} decimalen"
    finally:
        tools.set_length_precision(was)


def test_model_info_units_are_kept_and_used(win):
    from volura.app import ModelInfoDialog
    dlg = ModelInfoDialog(win)
    dlg.precision.setCurrentIndex(0)
    dlg.snap_on.setChecked(False)
    dlg.accept()
    try:
        assert win.settings.length_precision == 0
        assert win.settings.length_snap == 0.0, "uit vinken betekent geen lengte-afronding"
        assert tools.fmt_len(12.3456) == "12 mm"
        again = type(win.settings)()
        assert again.length_precision == 0 and again.length_snap == 0.0
        reopened = ModelInfoDialog(win)
        assert reopened.precision.currentIndex() == 0 and not reopened.snap_on.isChecked()
        assert reopened.sample.text() == "123 mm"
    finally:
        back = ModelInfoDialog(win)
        back.precision.setCurrentIndex(2)
        back.snap_on.setChecked(True)
        back.accept()
    assert tools.fmt_len(12.3456) == "12.35 mm"
    assert win.settings.length_snap == 1.0


def test_saved_precision_is_used_by_a_new_window(win, fresh):
    from volura.app import ModelInfoDialog
    dlg = ModelInfoDialog(win)
    dlg.precision.setCurrentIndex(1)
    dlg.snap_on.setChecked(True)
    dlg.accept()
    try:
        again = fresh()
        assert again.settings.length_precision == 1
        assert tools.fmt_len(12.3456) == "12.3 mm"
    finally:
        back = ModelInfoDialog(win)
        back.precision.setCurrentIndex(2)
        back.accept()


# ---------------------------------------------------------------- shortcuts

def test_shortcut_page_lists_the_commands_grouped_per_menu(win):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win)
    tree = dlg.shortcuts.tree
    groups = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    menus = [clean(a.text()) for a in win.menuBar().actions() if a.menu() is not None]
    assert groups == menus + [app_module.OTHER_COMMANDS], "één groep per menu, in dezelfde volgorde"
    paths = win.command_paths()
    for path in ("Bestand/Nieuw", "Venster/Modelinfo", "Venster/Voorkeuren", "Help/Welkom bij Volura",
                 "Overig/Op bed plaatsen"):
        assert path in paths, f"{path} ontbreekt in de lijst"
    assert paths["Bestand/Nieuw"] is win.a_new
    assert len(paths) == len(set(id(a) for a in paths.values())), "elke opdracht staat er precies één keer in"
    item = dlg.shortcuts.items["Bestand/Nieuw"]
    assert item.text(0) == "Nieuw" and "Ctrl+N" in item.text(1)
    assert "Recente bestanden" not in " ".join(paths), "recente bestanden zijn geen opdracht"


def test_shortcut_filter(win):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win)
    dlg.shortcuts.filter.setText("opslaan")
    assert not dlg.shortcuts.items["Bestand/Opslaan"].isHidden()
    assert dlg.shortcuts.items["Bestand/Nieuw"].isHidden()
    dlg.shortcuts.filter.setText("")
    assert not dlg.shortcuts.items["Bestand/Nieuw"].isHidden()


def test_assigned_shortcut_comes_back_in_a_new_window(win, fresh):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win)
    assert dlg.shortcuts.select_command("Bestand/Nieuw")
    dlg.shortcuts.key_edit.setKeySequence(QKeySequence("Ctrl+Alt+J"))
    assert dlg.shortcuts.add_shortcut(), "de '+'-knop wijst de sneltoets toe"
    assert shortcuts_of(win.a_new) == ["Ctrl+N", "Ctrl+Alt+J"]
    assert shortcut_keys() == ["shortcuts/Bestand/Nieuw"]
    assert "Ctrl+Alt+J" in dlg.shortcuts.items["Bestand/Nieuw"].text(1)
    again = fresh()
    assert shortcuts_of(again.a_new) == ["Ctrl+N", "Ctrl+Alt+J"]


def test_shortcut_that_is_in_use_moves_when_confirmed(win, fresh, monkeypatch):
    from volura.app import PreferencesDialog
    box = FakeBox(FakeBox.Yes)
    monkeypatch.setattr(app_module, "QMessageBox", box)
    dlg = PreferencesDialog(win)
    assert dlg.shortcuts.select_command("Bestand/Nieuw")
    dlg.shortcuts.key_edit.setKeySequence(QKeySequence("Ctrl+S"))
    assert dlg.shortcuts.add_shortcut()
    assert box.asked, "er wordt gevraagd of de sneltoets overgezet mag worden"
    assert "Ctrl+S" in shortcuts_of(win.a_new)
    assert "Ctrl+S" not in shortcuts_of(win.a_save)
    again = fresh()
    assert "Ctrl+S" in shortcuts_of(again.a_new)
    assert "Ctrl+S" not in shortcuts_of(again.a_save), "Opslaan houdt zijn lijst zonder Ctrl+S"


def test_shortcut_that_is_in_use_stays_when_refused(win, monkeypatch):
    from volura.app import PreferencesDialog
    box = FakeBox(FakeBox.No)
    monkeypatch.setattr(app_module, "QMessageBox", box)
    dlg = PreferencesDialog(win)
    assert dlg.shortcuts.select_command("Bestand/Nieuw")
    dlg.shortcuts.key_edit.setKeySequence(QKeySequence("Ctrl+S"))
    assert not dlg.shortcuts.add_shortcut()
    assert box.asked
    assert shortcuts_of(win.a_new) == ["Ctrl+N"]
    assert "Ctrl+S" in shortcuts_of(win.a_save)
    assert shortcut_keys() == []


def test_assigned_shortcut_works_in_the_window(win):
    """Not only stored: the window really reacts to the key that was just assigned."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from volura.app import PreferencesDialog
    win.set_tool("select")
    dlg = PreferencesDialog(win)
    assert dlg.shortcuts.select_command("Gereedschap/Gum")
    dlg.shortcuts.key_edit.setKeySequence(QKeySequence("Ctrl+Alt+J"))
    assert dlg.shortcuts.add_shortcut()
    win.show()
    try:
        QApplication.processEvents()
        QTest.keyClick(win, Qt.Key_J, Qt.AltModifier | Qt.ControlModifier)
        assert win.view.tool.name == "eraser", "de nieuwe sneltoets doet iets"
    finally:
        win.set_tool("select")
        win.hide()


def test_remove_shortcut_and_restore_everything(win):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win)
    assert dlg.shortcuts.select_command("Bestand/Nieuw")
    dlg.shortcuts.key_edit.setKeySequence(QKeySequence("Ctrl+N"))
    assert dlg.shortcuts.remove_shortcut(), "de '−'-knop haalt de sneltoets af"
    assert shortcuts_of(win.a_new) == []
    assert shortcut_keys() == ["shortcuts/Bestand/Nieuw"]
    dlg.shortcuts.restore_all()
    assert "Ctrl+N" in shortcuts_of(win.a_new)
    assert shortcut_keys() == [], "Alles herstellen wist de eigen sneltoetsen ook uit QSettings"


# ---------------------------------------------------------------- what opens what

def test_print_setup_opens_model_info_on_the_bed_page(win, monkeypatch):
    opened = []
    monkeypatch.setattr(app_module, "ModelInfoDialog", FakeDialog(opened, "modelinfo"))
    win.a_print_setup.trigger()
    assert opened == [("modelinfo", "Printbed")], \
        "Bestand > Printer instellen opent Modelinfo op de pagina Printbed"
    win.a_model_info.trigger()
    assert opened == [("modelinfo", "Printbed"), ("modelinfo", 0)]


def test_venster_and_help_open_the_new_dialogs(win, monkeypatch):
    opened = []
    monkeypatch.setattr(app_module, "ModelInfoDialog", FakeDialog(opened, "modelinfo"))
    monkeypatch.setattr(app_module, "PreferencesDialog", FakeDialog(opened, "voorkeuren"))
    monkeypatch.setattr(app_module, "WelcomeDialog", FakeDialog(opened, "welkom"))
    win.a_model_info.trigger()
    win.a_prefs.trigger()
    win.a_welcome.trigger()
    assert opened == [("modelinfo", 0), ("voorkeuren", 0), ("welkom", 0)]


# ---------------------------------------------------------------- the welcome screen at startup

def test_welcome_checkbox_is_kept(win):
    from volura.app import WelcomeDialog, welcome_enabled
    q = settings()
    q.setValue("show_welcome", True)
    q.sync()
    assert welcome_enabled(), "standaard staat het welkomstscherm aan"
    off = WelcomeDialog(win)
    off.show_at_start.setChecked(False)
    try:
        assert not welcome_enabled()
        again = WelcomeDialog(win)
        assert not again.show_at_start.isChecked()
    finally:
        on = WelcomeDialog(win)
        on.show_at_start.setChecked(True)
    assert welcome_enabled()


def test_welcome_is_not_shown_when_running_offscreen(win, monkeypatch):
    calls = []
    monkeypatch.setattr(type(win), "show_welcome", lambda self, checked=False: calls.append(1))
    monkeypatch.setattr(app_module, "welcome_enabled", lambda: True)
    win.show_welcome_on_startup()
    assert calls == [], "tests draaien offscreen: geen welkomstscherm"


def test_welcome_is_shown_at_startup_when_it_is_on(win, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "is_offscreen", lambda: False)
    monkeypatch.setattr(app_module, "welcome_enabled", lambda: True)
    monkeypatch.setattr(type(win), "show_welcome", lambda self, checked=False: calls.append(1))
    win.show_welcome_on_startup()
    assert calls == [1]


def test_welcome_stays_hidden_when_the_box_is_off(win, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "is_offscreen", lambda: False)
    monkeypatch.setattr(app_module, "welcome_enabled", lambda: False)
    monkeypatch.setattr(type(win), "show_welcome", lambda self, checked=False: calls.append(1))
    win.show_welcome_on_startup()
    assert calls == []
