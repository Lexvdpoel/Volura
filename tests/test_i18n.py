"""English is the source language; every text has a Dutch translation with the same placeholders."""
import ast
import glob
import os
import string
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from volura import i18n  # noqa: E402
from volura.locale import nl  # noqa: E402


def tr_literals():
    """Every literal passed to tr() or tr_noop() in the volura package: (text, file:line)."""
    out = []
    for path in glob.glob(os.path.join(ROOT, "volura", "*.py")):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("tr", "tr_noop") \
                    and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                out.append((node.args[0].value, f"{os.path.basename(path)}:{node.lineno}"))
    return out


def fields(text):
    return sorted(name for _, name, _, _ in string.Formatter().parse(text) if name is not None)


def test_tests_run_in_english():
    assert i18n.language() == "en"
    assert i18n.tr("File") == "File"


def test_every_text_has_a_dutch_translation():
    missing = sorted({f"{where}: {text!r}" for text, where in tr_literals()
                      if text not in nl.TRANSLATIONS})
    assert not missing, "\n".join(missing[:40])


def test_placeholders_match():
    wrong = [en for en, text in nl.TRANSLATIONS.items() if fields(en) != fields(text)]
    assert not wrong, wrong[:20]


def test_only_strings_that_are_used():
    used = {text for text, _ in tr_literals()}
    unused = sorted(set(nl.TRANSLATIONS) - used)
    assert len(unused) <= len(nl.TRANSLATIONS) // 20, unused[:40]   # a little slack for tr(variable)


def test_switching_language():
    try:
        assert i18n.set_language("nl") == "nl"
        assert i18n.tr("File") == nl.TRANSLATIONS.get("File", "File")
        assert i18n.source_text(nl.TRANSLATIONS.get("File", "File")) == "File"
        assert i18n.tr("a text without translation") == "a text without translation"
    finally:
        i18n.set_language("en")


@pytest.mark.parametrize("choice, expected", (("en", "en"), ("nl", "nl")))
def test_choice_wins_over_system(monkeypatch, choice, expected):
    monkeypatch.delenv("VOLURA_LANG", raising=False)
    assert i18n.resolve(choice) == expected
