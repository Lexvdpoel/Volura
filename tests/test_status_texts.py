"""Statusbalkteksten en het label van het meetvak per gereedschap, zoals SketchUp 2017.

Elk gereedschap heeft een `status_text()` per stap en een `vcb_label()` die bij het
gereedschap past; de viewport werkt beide na elke muis- of toetsgebeurtenis bij.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # run the deferred initial_view now
    w.resize(800, 600)
    w.view.resize(800, 600)
    yield w
    w.doc.dirty = False
    w.close()


def tool(win, name):
    """The tool instance of the viewport, reset to its first step."""
    t = win.view.tools[name]
    t.reset()
    return t


def click(win, x, y):
    """A real left click through the event routing of the viewport."""
    p = QPointF(float(x), float(y))
    win.view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton,
                                         Qt.LeftButton, Qt.NoModifier))


def test_elk_gereedschap_heeft_een_statustekst_en_een_label(win):
    from volura.tools import TOOL_CLASSES
    for cls in TOOL_CLASSES:
        t = win.view.tools[cls.name]
        assert isinstance(t.status_text(), str), f"{cls.name}: status_text() is geen tekst"
        assert t.status_text().strip(), f"{cls.name}: status_text() is leeg"
        assert isinstance(t.vcb_label(), str), f"{cls.name}: vcb_label() is geen tekst"


def test_statustekst_verandert_na_de_eerste_klik_van_lijn(win):
    win.set_tool("line")
    t = win.view.tools["line"]
    before = t.status_text()
    click(win, 400, 300)
    assert t.pts, "de klik had een beginpunt moeten zijn"
    after = t.status_text()
    assert after != before, "de statustekst veranderde niet na de eerste klik"
    assert "eindpunt" in after.lower()
    assert win.hint.text() == after            # de statusbalk volgt de stap
    t.reset()
    assert t.status_text() == before


def test_staptekst_van_lijn_zonder_klik(win):
    assert tool(win, "line").status_text() == "Selecteer beginpunt."
    assert tool(win, "line").vcb_label() == "Lengte"


STEP_TEXTS = [
    ("line", {}, "Selecteer beginpunt."),
    ("line", {"pts": [np.zeros(3)]}, "eindpunt"),
    ("rectangle", {}, "Selecteer eerste hoek."),
    ("rectangle", {"p0": np.zeros(3)}, "Selecteer tegenoverliggende hoek of typ waarden."),
    ("circle", {}, "Selecteer middelpunt."),
    ("circle", {"p0": np.zeros(3)}, "Selecteer straal of typ een waarde."),
    ("polygon", {}, "Selecteer middelpunt."),
    ("polygon", {"p0": np.zeros(3)}, "Selecteer straal of typ een waarde."),
    ("arc2", {}, "Selecteer beginpunt."),
    ("arc2", {"p0": np.zeros(3)}, "Selecteer eindpunt of typ een lengte."),
    ("arc2", {"p0": np.zeros(3), "p1": np.array([10.0, 0, 0])}, "uitstulping"),
    ("pushpull", {}, "Selecteer een vlak om te duwen of te trekken."),
    ("pushpull", {"target": (0, 0, np.zeros(3), np.array([0.0, 0, 1]))},
     "Beweeg om te duwen/trekken of typ een afstand. Ctrl = nieuw vlak."),
    ("move", {}, "Selecteer object om te verplaatsen. Ctrl = kopieren."),
    ("move", {"base": np.zeros(3)}, "Selecteer doelpunt of typ een afstand."),
    ("rotate", {}, "Selecteer middelpunt."),
    ("rotate", {"center": np.zeros(3)}, "Selecteer beginpunt van de hoek."),
    ("rotate", {"center": np.zeros(3), "ref": np.array([1.0, 0, 0])},
     "Selecteer eindpunt van de hoek of typ een hoek."),
    ("scale", {}, "Selecteer een greep om te schalen. Ctrl = rond middelpunt, Shift = uniform."),
    ("tape", {}, "Selecteer beginpunt om te meten."),
    ("protractor", {}, "Selecteer middelpunt van de gradenboog."),
    ("eraser", {}, "Selecteer objecten om te wissen. Ctrl = verzachten, Shift = verbergen."),
    ("paint", {}, "Selecteer een object om te verven."),
    ("select", {}, "Selecteer objecten. Shift = toevoegen/verwijderen. Ctrl = toevoegen. "
                   "Ctrl+Shift = verwijderen."),
]


@pytest.mark.parametrize("name,state,expected", STEP_TEXTS)
def test_statustekst_per_stap(win, name, state, expected):
    t = tool(win, name)
    for key, value in state.items():
        setattr(t, key, value)
    assert expected in t.status_text(), f"{name}: {t.status_text()!r}"


LABELS = [
    ("select", {}, ""),
    ("line", {}, "Lengte"),
    ("rectangle", {}, "Afmetingen"),
    ("circle", {}, "Zijden"),
    ("circle", {"p0": np.zeros(3)}, "Straal"),
    ("polygon", {}, "Zijden"),
    ("polygon", {"p0": np.zeros(3)}, "Straal"),
    ("arc2", {}, "Lengte"),
    ("arc2", {"p0": np.zeros(3), "p1": np.array([10.0, 0, 0])}, "Uitstulping"),
    ("pushpull", {}, "Afstand"),
    ("offset", {}, "Afstand"),
    ("move", {}, "Afstand"),
    ("rotate", {}, "Hoek"),
    ("scale", {}, "Schaal"),
    ("tape", {}, "Lengte"),
    ("protractor", {}, "Hoek"),
    ("zoom", {}, "Gezichtsveld"),
]


@pytest.mark.parametrize("name,state,expected", LABELS)
def test_label_van_het_meetvak(win, name, state, expected):
    t = tool(win, name)
    for key, value in state.items():
        setattr(t, key, value)
    assert t.vcb_label() == expected, f"{name}: {t.vcb_label()!r}"


def test_statusbalk_en_meetvak_volgen_het_gereedschap(win):
    win.set_tool("pushpull")
    assert win.hint.text() == win.view.tools["pushpull"].status_text()
    assert win.vcb_label.text() == "Afstand"
    win.set_tool("rotate")
    assert win.hint.text() == "Selecteer middelpunt."
    assert win.vcb_label.text() == "Hoek"


def test_statusbalk_blijft_staan_na_een_klik_zonder_stap(win):
    """A message of the window is not overwritten until the next event."""
    win.view.set_status("Model gesneden met een vlak")
    assert win.hint.text() == "Model gesneden met een vlak"
    win.set_tool("select")
    assert win.hint.text() == win.view.tools["select"].status_text()
