"""Help > Sponsor Volura… and the GitHub Sponsor button point at the same Sponsors page.

Run with:  python -m pytest tests/test_sponsor.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from volura import app as app_module  # noqa: E402
from volura.i18n import tr  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def win():
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def _help_menu(w):
    for a in w.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == "Help":
            return a.menu()
    raise AssertionError("no Help menu")


def test_the_sponsor_page_is_the_github_sponsors_page():
    assert app_module.SPONSOR_URL == "https://github.com/sponsors/Lexvdpoel"


def test_help_has_sponsor_volura_just_before_about(win):
    acts = [a for a in _help_menu(win).actions() if not a.isSeparator()]
    assert win.a_sponsor in acts
    assert acts.index(win.a_sponsor) + 1 == acts.index(win.a_about)
    assert win.a_sponsor.text() == tr("&Sponsor Volura…")


def test_sponsor_volura_opens_the_sponsors_page(win, monkeypatch):
    opened = []
    monkeypatch.setattr(app_module.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toString()))
    win.a_sponsor.trigger()
    assert opened == [app_module.SPONSOR_URL]


def test_the_about_box_asks_for_sponsors(win, monkeypatch):
    shown = []
    monkeypatch.setattr(app_module.QMessageBox, "about",
                        lambda parent, title, text: shown.append(text))
    win.about()
    assert shown and app_module.SPONSOR_URL in shown[0]
    assert "open source" in shown[0]


def test_the_repository_has_a_sponsor_button():
    with open(os.path.join(ROOT, ".github", "FUNDING.yml"), encoding="utf-8") as f:
        text = f.read()
    assert "github: Lexvdpoel" in text


def test_the_dutch_texts_are_there():
    from volura.locale.nl import TRANSLATIONS
    assert TRANSLATIONS["&Sponsor Volura…"] == "Volura &sponsoren…"
    assert any("Word sponsor" in v for v in TRANSLATIONS.values())
