"""The changelog has an entry for the version being released.

The release on GitHub takes its notes from the CHANGELOG.md section of the tagged version,
so a version without a section would be released without notes.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from volura import __version__  # noqa: E402


def sections():
    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
        text = f.read()
    return re.findall(r"^## (\d+\.\d+\.\d+) — (\d{4}-\d{2}-\d{2})\n(.*?)(?=^## |\Z)",
                      text, flags=re.M | re.S)


def test_the_current_version_has_a_section_with_content():
    found = {version: body for version, _date, body in sections()}
    assert __version__ in found
    assert re.search(r"^- ", found[__version__], flags=re.M)


def test_versions_are_listed_newest_first():
    versions = [tuple(int(p) for p in v.split(".")) for v, _d, _b in sections()]
    assert versions == sorted(versions, reverse=True)
    assert versions[0] == tuple(int(p) for p in __version__.split("."))
