"""The tests check the English texts, whatever the language of this computer.

Volura keeps its settings in the user's own QSettings (the registry on Windows), and some
tests save files or change preferences. Everything is put back as it was after the run,
so the tests never leave recent files or changed settings behind for the user.
"""
import os

os.environ["VOLURA_LANG"] = "en"

_saved = None


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings("Volura", "Volura")


def pytest_sessionstart(session):
    global _saved
    q = _settings()
    _saved = {key: q.value(key) for key in q.allKeys()}


def pytest_sessionfinish(session, exitstatus):
    if _saved is None:
        return
    q = _settings()
    q.clear()
    for key, value in _saved.items():
        q.setValue(key, value)
    q.sync()
