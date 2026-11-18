"""Bake the features and the manual of a Volura release into the volura.dev pages, so a
search engine sees the text without running site.js (which still swaps in the newest
release in the browser). Usage: python bake_site.py SITE_DIR TAG"""
import re
import sys
import urllib.request

site, tag = sys.argv[1], sys.argv[2]
RAW = "https://raw.githubusercontent.com/Lexvdpoel/Volura/{tag}/docs/website/{name}.html"


def fragment(name):
    with urllib.request.urlopen(RAW.format(tag=tag, name=name), timeout=30) as r:
        html = r.read().decode("utf-8")
    html = re.sub(r"^\s*<!--.*?-->\s*", "", html, count=1, flags=re.S)   # the note on top
    assert "<script" not in html.lower()
    return html.strip()


def bake(page, name, opening):
    path = f"{site}/{page}"
    s = open(path, encoding="utf-8").read()
    start = s.index(opening) + len(opening)
    end_tag = "</article>" if opening.startswith("<article") else "\n      </div>"
    if "<!--baked:" in s[start:start + 200]:
        end = s.index("<!--/baked-->", start) + len("<!--/baked-->")
    else:
        end = s.index(end_tag, start)
    body = f"\n<!--baked:{tag}-->\n{fragment(name)}\n<!--/baked-->"
    s = s[:start] + body + s[end:] if "<!--baked:" in s[start:start + 200] else \
        s[:start] + body + "\n" + s[end:]
    open(path, "w", encoding="utf-8", newline="").write(s)
    print(page, "baked", tag)


bake("index.html", "features", '<div class="grid" data-remote="features">')
bake("manual.html", "manual", '<article class="doc" data-remote="manual" data-has-toc>')
