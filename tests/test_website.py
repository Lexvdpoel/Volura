"""The texts of volura.dev in docs/website/: the feature cards and the manual.

The site loads these files from the latest release, so a broken file is broken online the
moment it is released. Every file must be well-formed HTML without scripts, every text must be
there in English and in Dutch, the headings of the manual need an id (the table of contents is
made from them) and every link to a section must land on one.

Run with:  python -m pytest tests/test_website.py
"""
import os
import re
from html.parser import HTMLParser

import pytest

WEBSITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "website")
FILES = ("features.html", "manual.html")
VOID = {"br", "hr", "img", "path", "rect", "circle", "ellipse", "input", "wbr"}


class Page(HTMLParser):
    """Tags, ids, section links and the language of every element with its own text."""

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.ids, self.anchors, self.tags = [], [], set(), [], []
        self.langs = {"en": 0, "nl": 0}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if "id" in a:
            self.ids.add(a["id"])
        if a.get("href", "").startswith("#"):
            self.anchors.append(a["href"][1:])
        if a.get("lang") in self.langs:
            self.langs[a["lang"]] += 1
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append(f"</{tag}> at line {self.getpos()[0]}, open: {self.stack[-1:]}")
            return
        self.stack.pop()


def page(name):
    p = Page()
    with open(os.path.join(WEBSITE, name), encoding="utf-8") as f:
        p.feed(f.read())
    p.close()
    return p


@pytest.mark.parametrize("name", FILES)
def test_the_file_is_well_formed(name):
    p = page(name)
    assert not p.errors and not p.stack, (p.errors, p.stack)


@pytest.mark.parametrize("name", FILES)
def test_no_scripts_and_no_inline_code(name):
    p = page(name)
    for tag, attrs in p.tags:
        assert tag not in ("script", "iframe", "object", "embed", "form"), tag
        assert not [k for k in attrs if k.startswith("on")], (tag, attrs)
        assert not str(attrs.get("href", "")).lower().startswith("javascript:"), attrs


@pytest.mark.parametrize("name", FILES)
def test_every_text_is_in_english_and_dutch(name):
    p = page(name)
    assert p.langs["en"] == p.langs["nl"] > 0, p.langs


def test_every_section_of_the_manual_has_an_id():
    with open(os.path.join(WEBSITE, "manual.html"), encoding="utf-8") as f:
        text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
    heads = re.findall(r"<h2\b[^>]*>", text)
    assert heads and all(re.search(r'\sid="[^"]+"', h) for h in heads), heads


def test_links_to_a_section_land_on_one():
    p = page("manual.html")
    assert set(p.anchors) <= p.ids, set(p.anchors) - p.ids


def test_every_feature_card_has_a_title_in_both_languages():
    with open(os.path.join(WEBSITE, "features.html"), encoding="utf-8") as f:
        cards = f.read().split('<div class="card">')[1:]
    assert len(cards) >= 6
    for card in cards:
        h3 = re.findall(r"<h3\b.*?</h3>", card, re.S)
        joined = " ".join(h3)
        assert 'lang="en"' in joined and 'lang="nl"' in joined, card[:200]
