"""Draw > Parts > Favourites: a part with the settings it was made with, kept for later.

A favourite is a small dict: the name it stands under in the menu (`name`), the title of the
part as Draw > Parts calls it (`title`), the name of the `make_..._model` method of the window
that builds it (`method`) and the options that method takes (`options`). The whole list travels
as one JSON string in the settings (`SETTINGS_KEY`), so every value has to be something JSON
knows; `jsonable` turns the options a part dialog handed out into that.

No Qt in this file: the window (volura/app.py) makes the menu and the dialog of Manage
Favourites…, this file only says what a favourite is and how it is kept. That is also why the
settings are read here and not in the window: a string that is no longer understood is dropped
by `load`, so a favourite from a version that is newer, or a hand-edited ini file, can never
keep Volura from starting.
"""
from __future__ import annotations

import json

#: the key of the settings string that holds all favourites (volura.app.Settings)
SETTINGS_KEY = "part_favourites"

#: how many favourites there can be: the parts of Volura fit in that, and a menu that long
#: is not a favourites list anymore but a second Parts menu
MAX_FAVOURITES = 30

#: how long the name of a favourite may be, so the menu of Favourites stays readable
MAX_NAME = 60


def is_part_method(name) -> bool:
    """Is this the name of a method that makes a part, like `make_cable_clip_model`?

    The window places a favourite by calling the method under this name, so only the plain
    name of such a method passes; what a hand-edited ini file holds does not."""
    if not isinstance(name, str) or not name.isidentifier():
        return False
    return (name.startswith("make_") and name.endswith("_model")
            and len(name) > len("make_") + len("_model"))


def jsonable(value):
    """`value` as JSON can keep it: a tuple becomes a list, what JSON cannot hold becomes text."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def is_favourite(item) -> bool:
    """Is this something the list and the menu can show?

    The method is not checked against the parts there are: a favourite of a part that Volura
    does not know anymore stays on the list, where it can be taken off."""
    if not isinstance(item, dict):
        return False
    for key in ("name", "method"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return isinstance(item.get("options"), dict)


def text_of(item, key) -> str:
    """The text of one field of a favourite, empty when it is missing or not a text.

    Also empty for something that is no dict at all: the menu and the window ask a favourite
    for its name and its method before they look at it."""
    value = item.get(key) if isinstance(item, dict) else None
    return value.strip() if isinstance(value, str) else ""


def copy(item) -> dict:
    """A favourite of its own, so the manage dialog can rename and move without changing
    what the menu shows before its OK."""
    return {"name": text_of(item, "name"), "title": text_of(item, "title"),
            "method": text_of(item, "method"), "options": jsonable(item["options"])}


def load(text) -> list:
    """The favourites of the settings string; everything it does not understand is dropped."""
    if not text:
        return []
    if isinstance(text, (list, tuple)):       # what QSettings makes of a string when it takes
        text = ",".join(str(part) for part in text)   # it for a comma-separated list
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if is_favourite(item):
            out.append(copy(item))
        if len(out) >= MAX_FAVOURITES:
            break
    return out


def save(items) -> str:
    """The favourites as the string that goes back in the settings."""
    return json.dumps([copy(i) for i in items if is_favourite(i)][:MAX_FAVOURITES],
                      ensure_ascii=False, sort_keys=True)


def part_name(method) -> str:
    """The name of a part from the method that makes it: `make_cable_clip_model` becomes
    'Cable Clip'. For a favourite that was never made through the Parts menu."""
    if not is_part_method(method):
        return ""
    return " ".join(word.capitalize() for word in method[len("make_"):-len("_model")].split("_"))
