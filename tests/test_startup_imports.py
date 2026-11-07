"""Starting fast: `import volura.app` may not drag in what only one feature needs.

The updater reaches GitHub through `urllib.request`, and the 3MF and SVG writers
escape their text with `xml.sax.saxutils`. Both chains together cost about a
tenth of a second, and neither belongs to the main window, so both are imported
where they are used. The check runs in a fresh process: the rest of the suite has
imported everything long before, so `sys.modules` here would say nothing.
"""
import ast
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = ("import sys, volura.app; print(sorted(m for m in sys.modules"
        " if m.startswith(('urllib', 'xml.sax', 'volura.updater'))))")


def _loaded():
    """The names of the modules a fresh process has after `import volura.app`."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                         check=True, env=env, cwd=ROOT)
    return ast.literal_eval(out.stdout.strip())


def test_the_app_does_not_import_the_updater_or_its_xml_writers():
    loaded = _loaded()
    assert "volura.updater" not in loaded and "urllib.request" not in loaded
    assert not [m for m in loaded if m.startswith("xml.sax")], loaded
