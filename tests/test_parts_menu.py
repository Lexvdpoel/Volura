"""Draw > Parts: 42 parts in four category submenus.

`app.PART_CATEGORIES` is the one place that says in which category a part stands: Mechanical,
Cables, Storage and Home and Desk, in that order, each sorted by the English name of its parts.
The menu is built from that table, so a part that is added to the window without being added to
the table is missing from the menu - which is exactly what these tests catch, together with a
part that stands in two categories or in none.

Also checked: the mnemonics of a menu do not clash, the Parts menu is only there in 3D, and the
command search of Preferences > Shortcuts still finds every part by its name.

Run with:  python -m pytest tests/test_parts_menu.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura.app import PART_CATEGORIES, MainWindow  # noqa: E402
from volura.window_ui import clean_text  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: every part of Draw > Parts, by the name of its action; a new part is added here too, so
#: the test below says so when it was forgotten in the categories
PARTS = ("a_bracket", "a_grommet", "a_fan_guard", "a_filament_clip", "a_hose_adapter", "a_knob",
         "a_print_ball_joint", "a_print_hinge", "a_snap_fit", "a_spool_adapter",
         "a_spur_gear_3d", "a_standoff", "a_timing_pulley",
         "a_cable_chain", "a_cable_clip", "a_cable_comb", "a_cable_winder", "a_tie_mount",
         "a_battery_holder", "a_bin_rack", "a_bit_holder", "a_drawer_organizer", "a_gridfinity",
         "a_hex_key_holder", "a_card_holder", "a_pegboard", "a_project_box", "a_screw_jar",
         "a_storage_bin",
         "a_bag_clip", "a_coaster", "a_cookie_cutter", "a_desk_hook", "a_door_wedge",
         "a_label_plate",
         "a_lithophane", "a_pen_cup", "a_phone_stand", "a_plant_label", "a_plant_pot",
         "a_soap_dish", "a_wall_hook")
CATEGORIES = ("&Mechanical", "&Cables", "&Storage", "&Home and Desk")


@pytest.fixture(scope="module")
def win():
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def labels(menu):
    """The commands of a menu, in the order it shows them, without & and …."""
    return [clean_text(a.text()) for a in menu.actions() if not a.isSeparator()]


def test_the_parts_menu_has_the_four_categories_in_order(win):
    assert labels(win.parts_menu) == [clean_text(c) for c in CATEGORIES]


@pytest.mark.parametrize("title", CATEGORIES)
def test_a_category_stands_in_the_parts_menu(win, title):
    menu = win.parts_category(clean_text(title))
    assert menu is not None, f"{title} has no submenu"
    assert menu in [a.menu() for a in win.parts_menu.actions()], "the category is not in Parts"


@pytest.mark.parametrize("title", CATEGORIES)
def test_a_category_is_sorted_by_the_name_of_its_parts(win, title):
    got = labels(win.parts_category(clean_text(title)))
    assert got == sorted(got, key=str.lower), got


def test_every_part_stands_in_exactly_one_category(win):
    """The table holds every part once, and the menu shows the table."""
    named = [name for _title, names in PART_CATEGORIES for name in names]
    assert sorted(named) == sorted(PARTS), "a part is missing from a category or in two of them"
    assert len(named) == len(set(named)) == len(PARTS) == 42
    for title, names in PART_CATEGORIES:
        actions = win.parts_category(clean_text(title)).actions()
        assert [getattr(win, name) for name in names] == actions, f"{title} is not the table"


def test_the_parts_menu_is_only_there_in_3d(win):
    draw = next(m.menu() for m in win.menuBar().actions()
                if m.menu() is not None and clean_text(m.text()) == "Draw")
    assert win.parts_menu in [a.menu() for a in draw.actions()]
    win.set_mode("3d")
    assert win.parts_menu.menuAction().isVisible(), "the model has no parts"
    win.set_mode("2d")
    assert not win.parts_menu.menuAction().isVisible(), "the worksheet builds no parts"
    win.set_mode("3d")


def test_the_mnemonics_of_the_parts_menus_do_not_clash(win):
    """Two commands with the same & in one menu take turns underlining it."""
    for menu in ([win.parts_menu] + [win.parts_category(clean_text(t)) for t in CATEGORIES]):
        keys = []
        for a in menu.actions():
            text = a.text()
            for i, ch in enumerate(text):
                if ch == "&" and text[i + 1:i + 2] != "&":
                    keys.append(text[i + 1].lower())
        assert len(keys) == len(set(keys)), f"{clean_text(menu.title())}: {keys}"


def test_the_command_search_still_finds_every_part(win):
    """Preferences > Shortcuts: typing the name of a part leaves it in the list."""
    from volura.app import PreferencesDialog
    actions = [getattr(win, name) for name in PARTS]
    paths = {path: a for path, a in win.command_paths().items() if a in actions}
    assert len(paths) == len(PARTS), "a part is not in the list of commands"
    dlg = PreferencesDialog(win)
    for path in sorted(paths):
        name = clean_text(path.rsplit("/", 1)[1]).lower()
        dlg.shortcuts.filter.setText(name)
        assert not dlg.shortcuts.items[path].isHidden(), f"'{name}' does not find {path}"
    dlg.shortcuts.filter.setText("")

