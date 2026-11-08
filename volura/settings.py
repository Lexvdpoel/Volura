"""Where Volura keeps its settings.

Normally ``QSettings("Volura", "Volura")``: the registry of the user on Windows, an ini
file in the configuration folder on Linux. That store cannot be redirected with
``QSettings.setDefaultFormat``, because the (organization, application) constructor
ignores that setting, so :func:`qsettings` looks at the environment variable
``VOLURA_SETTINGS_FILE`` first: when it is set, the whole program reads and writes that
ini file and nothing of the user is touched. ``tests/conftest.py`` sets it to a file in a
temporary folder, so a test run cannot leave test values behind in the settings of the
user, also not when it is killed halfway.
"""
from __future__ import annotations

import os

SETTINGS_FILE_ENV = "VOLURA_SETTINGS_FILE"


def settings_file() -> str:
    """The ini file of ``VOLURA_SETTINGS_FILE``, or "" when it is not set."""
    return os.environ.get(SETTINGS_FILE_ENV, "").strip()


def qsettings():
    """The settings of Volura: the file of ``VOLURA_SETTINGS_FILE``, else the user's own."""
    # Qt is imported here and not at the top: volura.i18n needs the settings before the
    # first QApplication exists, and must keep working when PySide6 is missing.
    from PySide6.QtCore import QSettings
    path = settings_file()
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings("Volura", "Volura")
