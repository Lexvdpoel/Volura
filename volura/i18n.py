"""Translations. English is the source language; other languages are lookup tables.

Every text the user sees goes through :func:`tr` (not ``_``: the code uses ``_`` as a
throwaway variable, as in ``_, r, u = cam.basis()``)::

    from .i18n import tr
    label = tr("Push/Pull")
    status = tr("{count} faces selected").format(count=n)

Placeholders are named (``{count}``), never f-strings inside ``tr()``, so a translation
can put them in another order. A table lives in ``volura/locale/<code>.py`` as
``TRANSLATIONS = {"English text": "translated text"}``; a missing entry falls back to
the English text.

The language is chosen once, before the first window is built: the environment
variable ``VOLURA_LANG``, else Preferences > General > Language (QSettings key
``language``), else the language of the system. Changing it takes effect after a
restart, because menus and module-level labels are made with the language of that
moment.
"""
from __future__ import annotations

import importlib
import os

SYSTEM = "system"
LANGUAGES = (("en", "English"), ("nl", "Nederlands"))
SETTINGS_KEY = "language"

_language = "en"
_table: dict[str, str] = {}
_reverse: dict[str, str] | None = None


def tr(text: str) -> str:
    """The text in the current language (the English text when there is no translation)."""
    return _table.get(text, text)


def tr_noop(text: str) -> str:
    """Mark a text for translation without translating it now (``tr`` is applied later)."""
    return text


def language() -> str:
    return _language


def table(code: str) -> dict:
    if code == "en":
        return {}
    try:
        return dict(importlib.import_module(f"volura.locale.{code}").TRANSLATIONS)
    except (ImportError, AttributeError):
        return {}


def system_language() -> str:
    try:
        from PySide6.QtCore import QLocale
        code = QLocale.system().name().split("_")[0].lower()
    except Exception:
        code = "en"
    return code if code in dict(LANGUAGES) else "en"


def saved_choice() -> str:
    """Preferences > General > Language: "system" or a language code."""
    try:
        from PySide6.QtCore import QSettings
        value = str(QSettings("Volura", "Volura").value(SETTINGS_KEY, SYSTEM) or SYSTEM)
    except Exception:
        value = SYSTEM
    return value if value == SYSTEM or value in dict(LANGUAGES) else SYSTEM


def resolve(choice: str | None = None) -> str:
    env = os.environ.get("VOLURA_LANG", "").strip().lower()
    if env in dict(LANGUAGES):
        return env
    choice = saved_choice() if choice is None else choice
    return system_language() if choice == SYSTEM else choice


def set_language(code: str | None = None) -> str:
    """Use this language from now on; None picks it as described in the module docstring."""
    global _language, _table, _reverse
    code = resolve() if code is None else code
    _language = code if code in dict(LANGUAGES) else "en"
    _table = table(_language)
    _reverse = None
    return _language


def source_text(text: str) -> str:
    """The English text of a translated one (or the text itself): for stable settings keys."""
    global _reverse
    if _reverse is None:
        _reverse = {}
        for code, _name in LANGUAGES:
            for en, other in table(code).items():
                _reverse.setdefault(other, en)
    return _reverse.get(text, text)


set_language()
