"""The tests check the English texts, whatever the language of this computer.

Volura normally keeps its settings in the user's own QSettings (the registry on Windows),
and the tests change preferences. `VOLURA_SETTINGS_FILE` is set here, before anything
imports volura, to an ini file in a fresh temporary folder, so every test reads and writes
that file and the settings of the user are never touched - also not when the run is killed
halfway and the hooks below never get to restore anything. They still put the keys back
after a normal run, so the file of this run stays as it was too.
"""
import os
import tempfile

os.environ["VOLURA_LANG"] = "en"
# the tests check results, not the process pool: run every job in the main process
# (tests/test_workers.py turns the pool on for the few tests that need it)
os.environ.setdefault("VOLURA_WORKERS", "0")
# the settings of this test run, instead of the registry of the user (volura.settings)
os.environ["VOLURA_SETTINGS_FILE"] = os.path.join(
    tempfile.mkdtemp(prefix="volura-settings-"), "volura.ini")

import pytest  # noqa: E402

_saved = None


@pytest.fixture(scope="module", autouse=True)
def _delete_the_windows_of_a_test_file():
    """Delete the windows a test file leaves open once it is done.

    Left alone they all stay until the end of the run, and Qt then takes them down one
    by one in its own exit code, in a time that grows faster than their number: with
    the whole suite in one process that took half an hour after the summary.

    They are hidden and deleted, never closed: closing a main window saves its settings,
    and a window a test left behind still holds the settings of that test (the Fusion key
    scheme, say), which the next window of the run would then start with."""
    yield
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    for w in app.topLevelWidgets():
        w.hide()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settings():
    from volura.settings import qsettings
    return qsettings()


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
