"""The splash screen: it can be built and follows the steps, and it can be turned off."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import __version__, splash  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def test_splash_shows_the_version_and_follows_the_steps():
    s = splash.Splash()
    s.show()
    assert s.version_label.text() == __version__
    s.step("Loading 3D view…", 40)
    assert s.step_label.text() == "Loading 3D view…"
    assert s.percent == 40
    s.step("Building the interface…")
    assert s.percent == 40, "a step without a percentage keeps the bar where it is"
    s.close()


def test_no_splash_on_request_or_without_a_display(monkeypatch):
    monkeypatch.delenv("VOLURA_NO_SPLASH", raising=False)
    assert splash.no_splash(["volura", "--no-splash"])
    monkeypatch.setenv("VOLURA_NO_SPLASH", "1")
    assert splash.no_splash(["volura"])
    monkeypatch.delenv("VOLURA_NO_SPLASH")
    assert splash.no_splash(["volura"]), "the tests run offscreen: no splash there"
