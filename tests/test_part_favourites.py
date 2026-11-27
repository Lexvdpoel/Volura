"""Part favourites: a part with the settings it was made with, kept for later.

Draw > Parts > Favourites stands above the categories. `Add Last Part to Favourites…` keeps the
part that was made last - which part it was, under which name in the menu and with which numbers -
and every favourite on the list places that part again with exactly those numbers, in one undo
step. `Manage Favourites…` renames the parts, takes some off and puts the others in another order.
A favourite is one JSON string in the settings (volura/part_favourites.py), so a part of a version
that is newer, or a hand-edited ini file, is dropped instead of breaking the menu.

These tests check what a favourite is and what survives the settings, the remembering (which part
was made last, with which settings and under which menu name), the menu (empty, filled, escaped,
the thirty-first refused), placing one (the same part again, and a favourite that does not work
leaving the model alone) and the manage dialog.

Run with:  python -m pytest tests/test_part_favourites.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QInputDialog,  # noqa: E402
                               QMessageBox)

import volura.app as app_module  # noqa: E402
from volura import part_favourites as pf  # noqa: E402
from volura.app_dialogs import FavouritesDialog  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: the options of a cable clip, the part these favourites are made of
OPTIONS = dict(cable_d=6.0, width=8.0, wall=1.6, opening=0.8, base=3.0, bore="M3", count=1)


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def labels(menu):
    """The commands of a menu, in the order it shows them, without & and …."""
    return [clean(a.text().replace("&&", "&")) for a in menu.actions() if not a.isSeparator()]


def lines(dlg):
    """The lines of the list of the manage dialog, as they stand there."""
    return [dlg.list.item(i).text() for i in range(dlg.list.count())]


def favourite(name, **options):
    """A favourite of a cable clip, with options of its own for the tests that look at them."""
    return dict(name=name, title="Cable Clip", method="make_cable_clip_model",
                options=dict(OPTIONS, **options))


def action_of(menu, name):
    """The command of the favourite called `name`, None when the menu does not show it.

    A name with an & stands twice in the text of the command, as in every Qt menu."""
    return next((a for a in menu.actions() if a.text().replace("&&", "&") == name), None)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.part_favourites = []           # what another window kept is not this window's
    w._last_part, w._part_made, w._part_titled = None, 0, -1
    w.update_favourites_menu()
    w.doc.dirty = False
    QCoreApplication.processEvents()          # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def make_a_clip(win, groups=1, **options):
    """A cable clip on the bed, so there is a last part to add to the favourites."""
    win.make_cable_clip_model(dict(OPTIONS, **options))
    assert pump_until(lambda: len(win.doc.groups) == groups), f"{groups} cable clips expected"
    win.update_favourites_menu()


def typed(name):
    """The answer a maker gives at the question of `Add Last Part to Favourites…`."""
    return staticmethod(lambda *a, **kw: (name, True))


# ------------------------------------------------------------------ what a favourite is

def test_only_the_names_of_part_methods_pass():
    assert pf.is_part_method("make_cable_clip_model")
    for wrong in ("make_", "make_model", "make_cable clip_model", "cable_clip", "open", "",
                  None, 7, "make_cable_clip", " make_cable_clip_model "):
        assert not pf.is_part_method(wrong), wrong


def test_the_options_are_kept_as_json_holds_them():
    got = pf.jsonable(dict(a=(1, 2), b=[1.5, "M3"], c=None, d=True))
    assert got == {"a": [1, 2], "b": [1.5, "M3"], "c": None, "d": True}
    assert isinstance(pf.jsonable({"key": object})["key"], str), "what JSON cannot hold is lost"


def test_a_favourite_needs_a_name_a_method_and_options():
    assert pf.is_favourite(favourite("Clip"))
    for wrong in ({}, "Clip", None, [],
                  {"name": "  ", "method": "make_cable_clip_model", "options": {}},
                  {"name": "Clip", "method": 4, "options": {}},
                  {"name": "Clip", "method": "make_cable_clip_model"},
                  {"name": "Clip", "method": "make_cable_clip_model", "options": "nope"}):
        assert not pf.is_favourite(wrong), wrong


def test_a_favourite_of_a_part_that_is_gone_keeps_its_place():
    """An unknown method is no reason to drop a favourite: it has to stay on the list of
    Manage Favourites…, where it can be taken off."""
    assert pf.is_favourite({"name": "Old part", "method": "open_a_file", "options": {}})


def test_the_name_of_the_part_comes_from_the_method_that_makes_it():
    assert pf.part_name("make_cable_clip_model") == "Cable Clip"
    assert pf.part_name("make_a_bracket_model") == "A Bracket"
    assert pf.part_name("nothing") == ""


def test_nothing_that_is_not_a_list_of_favourites_survives_the_settings():
    assert pf.load("") == [] and pf.load(None) == []
    assert pf.load("this is no json") == []
    assert pf.load('{"name": "Clip"}') == []                   # a list is asked, not one item
    assert pf.load('[{"name": "Clip"}, "nope", 4]') == []       # a favourite needs more
    assert pf.load('[{"name": "  ", "method": "make_cable_clip_model", "options": {}}]') == []


def test_a_string_the_settings_split_in_parts_is_read_back_as_one():
    items = [favourite("Clip"), favourite("Other", cable_d=9.0)]
    kept = pf.save(items)
    assert pf.load(kept) == items
    assert pf.load(kept.split(",")) == items                   # what QSettings can hand back
    assert pf.load([4, 5]) == []                               # numbers were never favourites


def test_the_settings_hold_the_favourites_once_and_in_order():
    items = [favourite("Second"), favourite("First", cable_d=9.0)]
    again = pf.load(pf.save(items))
    assert isinstance(pf.save(items), str)
    assert [i["name"] for i in again] == ["Second", "First"]
    assert again == items


def test_thirty_favourites_is_the_most_and_the_rest_is_left_out():
    items = [favourite(f"Clip {i}") for i in range(pf.MAX_FAVOURITES + 7)]
    kept = pf.load(pf.save(items))
    assert len(kept) == pf.MAX_FAVOURITES
    assert [i["name"] for i in kept][:2] == ["Clip 0", "Clip 1"]


def test_a_favourite_of_a_part_volura_does_not_know_still_reads_back():
    """Such a favourite cannot be placed but stays on the list, so it can be removed."""
    odd = {"name": "Old part", "title": "", "method": "make_no_such_model", "options": {}}
    assert pf.load(pf.save([odd])) == [pf.copy(odd)]


# ------------------------------------------------------------------ the menu

def test_the_favourites_menu_stands_above_the_categories(win):
    assert labels(win.parts_menu)[0] == "Favourites"
    assert win.favourites_menu in [a.menu() for a in win.parts_menu.actions()]


def test_an_empty_list_says_so_and_can_add_nothing(win):
    assert labels(win.favourites_menu) == ["Add Last Part to Favourites", "No favourites yet",
                                           "Manage Favourites"]
    assert not win.a_fav_add.isEnabled(), "a part can be added before one was made"
    assert not action_of(win.favourites_menu, "No favourites yet").isEnabled()
    assert win.a_fav_manage.isEnabled()


def test_making_a_part_remembers_it_with_its_settings_and_its_menu_name(win, monkeypatch):
    def fill(dlg):
        dlg.cable_d.setValue(9.0)
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", fill)
    win.a_cable_clip.trigger()
    assert pump_until(lambda: len(win.doc.groups) == 1), "no cable clips in the model"
    part = win._last_part
    assert part["method"] == "make_cable_clip_model"
    assert part["title"] == "Cable Clip", "the name in the menu did not reach the part"
    assert part["options"]["cable_d"] == pytest.approx(9.0)
    assert win.a_fav_add.isEnabled(), "the part just made cannot be added"


def test_the_part_that_was_asked_for_last_is_the_last_part(win):
    make_a_clip(win)
    assert win._last_part["method"] == "make_cable_clip_model"
    assert win._last_part["title"] == "", "a part that never stood in a menu has no menu name"
    make_a_clip(win, groups=2, cable_d=9.0)
    assert win._last_part["options"]["cable_d"] == pytest.approx(9.0), "the oldest part is kept"


def test_making_a_part_of_another_kind_replaces_the_last_part(win, monkeypatch):
    make_a_clip(win)
    monkeypatch.setattr(QDialog, "exec", lambda dlg: QDialog.Rejected)
    win.a_knob.trigger()
    assert win._last_part["method"] == "make_cable_clip_model", "a refused dialog changed the part"
    win.make_standoff_model(dict(height=10.0, outer=8.0, shape="round", bore="M3",
                                 kind="through", clearance=0.2, count=1))
    assert pump_until(lambda: len(win.doc.groups) == 2)
    assert win._last_part["method"] == "make_standoff_model"
    assert win._last_part["options"]["bore"] == "M3"


def test_a_favourite_stands_in_the_menu_under_its_name(win, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", typed("My clip"))
    make_a_clip(win)
    win.a_fav_add.trigger()
    assert labels(win.favourites_menu) == ["Add Last Part to Favourites", "My clip",
                                           "Manage Favourites"]
    assert "My clip is added to the favourites." in win.hint.text(), win.hint.text()


def test_a_name_with_an_ampersand_is_a_menu_item_of_its_own(win, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", typed("Desk & wall hook"))
    make_a_clip(win)
    win.a_fav_add.trigger()
    shown = [a.text() for a in win.favourites_menu.actions() if not a.isSeparator()]
    assert "Desk && wall hook" in shown, f"the & of the name is taken for a mnemonic: {shown}"
    assert action_of(win.favourites_menu, "Desk & wall hook") is not None


def test_a_name_that_is_only_spaces_adds_nothing(win, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", typed("   "))
    make_a_clip(win)
    win.a_fav_add.trigger()
    assert win.settings.part_favourites == []
    assert labels(win.favourites_menu)[1] == "No favourites yet"


def test_a_very_long_name_is_cut_to_a_name_that_fits_the_menu(win, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", typed("k" * 400))
    make_a_clip(win)
    win.a_fav_add.trigger()
    assert len(win.settings.part_favourites[0]["name"]) == pf.MAX_NAME


def test_the_name_is_asked_with_the_menu_name_of_the_part_as_default(win, monkeypatch):
    asked = {}

    def ask(_parent, _title, _label, **kw):
        asked["default"] = kw.get("text")
        return ("", False)                                      # Cancel

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(ask))
    monkeypatch.setattr(QDialog, "exec", lambda dlg: QDialog.Accepted)
    win.a_cable_clip.trigger()
    assert pump_until(lambda: len(win.doc.groups) == 1)
    win.update_favourites_menu()
    win.a_fav_add.trigger()
    assert asked["default"] == "Cable Clip"
    assert win.settings.part_favourites == [], "cancelling added a favourite"


def test_a_part_that_never_stood_in_a_menu_is_asked_by_the_name_of_its_method(win, monkeypatch):
    asked = {}

    def ask(_parent, _title, _label, **kw):
        asked["default"] = kw.get("text")
        return ("", False)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(ask))
    make_a_clip(win)
    win.a_fav_add.trigger()
    assert asked["default"] == "Cable Clip"


def test_a_favourite_is_kept_for_the_next_time_volura_starts(win, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", typed("My clip"))
    make_a_clip(win)
    win.a_fav_add.trigger()
    again = app_module.Settings()
    assert [i["name"] for i in again.part_favourites] == ["My clip"]
    assert again.part_favourites[0]["options"] == pf.jsonable(OPTIONS)
    assert again.part_favourites[0]["method"] == "make_cable_clip_model"


def test_the_thirty_first_favourite_is_refused_with_a_word(win, monkeypatch):
    win.settings.part_favourites = [favourite(f"Clip {i}") for i in range(pf.MAX_FAVOURITES)]
    asked, told = [], []
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **kw: asked.append(1) or ("Name", True)))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **kw: told.append(a[2])))
    make_a_clip(win)
    win.a_fav_add.trigger()
    assert not asked, "a name was asked for a favourite that cannot be added"
    assert len(win.settings.part_favourites) == pf.MAX_FAVOURITES
    assert str(pf.MAX_FAVOURITES) in told[0] and "Manage Favourites" in told[0]


# ------------------------------------------------------------------ placing one

def test_a_favourite_is_the_same_part_again_in_one_undo_step(win):
    make_a_clip(win, cable_d=9.0)
    volume = win.doc.groups[0].ents.to_manifold().volume()
    win.settings.part_favourites = [favourite("My clip", cable_d=9.0)]
    win.update_favourites_menu()
    steps = len(win.doc.undo_stack)
    action_of(win.favourites_menu, "My clip").trigger()
    assert pump_until(lambda: len(win.doc.groups) == 2), "the favourite did not come back"
    assert len(win.doc.undo_stack) == steps + 1, "the favourite is more than one undo step"
    assert win.doc.groups[1].ents.to_manifold().volume() == pytest.approx(volume, rel=1e-6)
    win.doc.undo()
    assert len(win.doc.groups) == 1


def test_a_favourite_places_the_part_with_the_settings_it_was_saved_with(win):
    win.settings.part_favourites = [favourite("Wide", cable_d=9.0, count=2)]
    win.update_favourites_menu()
    action_of(win.favourites_menu, "Wide").trigger()
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert pump_until(lambda: "2 cable clips for 9 mm cable" in win.hint.text()), win.hint.text()
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 2


def test_the_part_that_was_placed_becomes_the_last_part_too(win):
    win.settings.part_favourites = [favourite("My clip")]
    win.update_favourites_menu()
    action_of(win.favourites_menu, "My clip").trigger()
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert win._last_part["method"] == "make_cable_clip_model"
    assert win._last_part["title"] == "Cable Clip"


def test_a_favourite_of_a_part_volura_does_not_know_leaves_the_model_alone(win):
    broken = {"name": "Old part", "title": "", "method": "make_no_such_model", "options": {}}
    win.settings.part_favourites = [broken]
    win.update_favourites_menu()
    action_of(win.favourites_menu, "Old part").trigger()
    assert "Old part is no longer a part that Volura knows." in win.hint.text(), win.hint.text()
    assert win.doc.groups == []
    assert win.settings.part_favourites == [broken], "it stays so it can be taken off"


def test_a_favourite_that_is_no_favourite_adds_nothing(win):
    win.settings.part_favourites = [{"name": "Clip", "title": "",
                                     "method": "not_a_method", "options": {}}]
    win.update_favourites_menu()
    assert win.place_part_favourite(win.settings.part_favourites[0]) is False
    assert win.place_part_favourite({}) is False
    assert win.place_part_favourite(None) is False
    assert win.doc.groups == []


def test_a_favourite_whose_settings_do_not_fit_says_so_and_adds_nothing(win):
    win.settings.part_favourites = [{"name": "Empty clip", "title": "Cable Clip",
                                     "method": "make_cable_clip_model", "options": {}}]
    win.update_favourites_menu()
    action_of(win.favourites_menu, "Empty clip").trigger()
    assert pump_until(lambda: "failed" in win.hint.text()), win.hint.text()
    assert "Empty clip" in win.hint.text()
    assert win.doc.groups == []
    assert len(win.settings.part_favourites) == 1


# ------------------------------------------------------------------ managing them

def test_manage_renames_moves_and_removes(win, monkeypatch):
    win.settings.part_favourites = [favourite("One"), favourite("Two"), favourite("Three")]
    monkeypatch.setattr(QInputDialog, "getText", typed("Second"))
    seen = {}

    def edit(dlg):
        seen["start"] = lines(dlg)
        dlg.list.setCurrentRow(1)
        dlg.btn_rename.click()
        seen["renamed"] = lines(dlg)
        dlg.list.setCurrentRow(0)
        dlg.btn_down.click()
        dlg.list.setCurrentRow(2)
        dlg.btn_remove.click()

    monkeypatch.setattr(FavouritesDialog, "exec", lambda dlg: edit(dlg) or QDialog.Accepted)
    win.a_fav_manage.trigger()
    assert seen["start"] == ["One  (Cable Clip)", "Two  (Cable Clip)", "Three  (Cable Clip)"]
    assert seen["renamed"][1] == "Second  (Cable Clip)"
    assert [i["name"] for i in win.settings.part_favourites] == ["Second", "One"]
    assert labels(win.favourites_menu) == ["Add Last Part to Favourites", "Second", "One",
                                           "Manage Favourites"]
    again = app_module.Settings()
    assert [i["name"] for i in again.part_favourites] == ["Second", "One"]


def test_manage_that_is_cancelled_changes_nothing(win, monkeypatch):
    win.settings.part_favourites = [favourite("One")]
    monkeypatch.setattr(FavouritesDialog, "exec", lambda dlg: QDialog.Rejected)
    win.a_fav_manage.trigger()
    assert [i["name"] for i in win.settings.part_favourites] == ["One"]


def test_the_last_favourite_off_leaves_an_empty_list_that_says_so(win, monkeypatch):
    win.settings.part_favourites = [favourite("One")]

    def edit(dlg):
        dlg.list.setCurrentRow(0)
        dlg.btn_remove.click()
        assert dlg.list.count() == 0
        assert not dlg.btn_remove.isEnabled(), "nothing chosen, yet something to remove"
        assert not dlg.btn_up.isEnabled() and not dlg.btn_down.isEnabled()

    monkeypatch.setattr(FavouritesDialog, "exec", lambda dlg: edit(dlg) or QDialog.Accepted)
    win.a_fav_manage.trigger()
    assert win.settings.part_favourites == []
    assert labels(win.favourites_menu) == ["Add Last Part to Favourites", "No favourites yet",
                                           "Manage Favourites"]


def test_a_favourite_without_a_menu_name_still_shows_what_it_is(win, monkeypatch):
    """A favourite of a part that was never made through the Parts menu: the manage dialog
    says which part it is by the name of the method that makes it."""
    win.settings.part_favourites = [{"name": "Clip", "title": "",
                                     "method": "make_cable_clip_model", "options": {}}]
    seen = {}

    def look(dlg):
        seen["lines"] = lines(dlg)

    monkeypatch.setattr(FavouritesDialog, "exec", lambda dlg: look(dlg) or QDialog.Rejected)
    win.a_fav_manage.trigger()
    assert seen["lines"] == ["Clip  (Cable Clip)"]
    assert len(win.settings.part_favourites) == 1, "cancelling changed the list"


def test_the_commands_of_the_manage_dialog_have_names(win):
    dlg = FavouritesDialog(win)
    assert dlg.list.accessibleName()
    for button in (dlg.btn_rename, dlg.btn_up, dlg.btn_down, dlg.btn_remove):
        assert button.text() and button.toolTip()
    assert not dlg.btn_rename.isEnabled(), "nothing is chosen yet"
    assert not dlg.btn_up.isEnabled() and not dlg.btn_down.isEnabled()
    dlg.deleteLater()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
