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

_saved = None


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
