"""The tests never touch the settings of the user.

`volura.settings.qsettings()` follows the environment variable `VOLURA_SETTINGS_FILE`,
which `tests/conftest.py` sets to an ini file in a temporary folder before volura is
imported. This checks that the store of the tests really is that file, and that what is
written there stays there: the settings of this computer (the registry on Windows) know
nothing of the key written below. Of the real store only that one key is read, never
written; the key is removed from the temporary file afterwards.

Run with:  python -m pytest tests/test_settings_isolation.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QSettings  # noqa: E402

from volura.settings import qsettings, settings_file  # noqa: E402

KEY = "isolation_test_marker_2f6c"


def test_the_tests_read_and_write_the_temporary_ini_file():
    """The helper points at the ini file of VOLURA_SETTINGS_FILE, not at the registry."""
    assert settings_file(), "conftest.py must set VOLURA_SETTINGS_FILE before volura is imported"
    was = os.path.normcase(os.path.abspath(settings_file()))
    now = os.path.normcase(os.path.abspath(qsettings().fileName()))
    assert now == was, f"{now} is not the settings file of this test run"


def test_what_a_test_writes_stays_in_the_temporary_ini_file():
    q = qsettings()
    q.setValue(KEY, "volura test")
    q.sync()
    try:
        assert QSettings("Volura", "Volura").value(KEY) is None, \
            "a test value ended up in the settings of the user"
    finally:
        q.remove(KEY)
        q.sync()
