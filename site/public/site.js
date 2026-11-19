// Language switch (English / Nederlands), the feature cards and the manual from GitHub, and the
// table of contents of the manual.
// Loaded in <head> without defer: the language is set before the page is drawn, so the
// header does not jump from English to Dutch after loading.
//
// The feature cards and the manual come from docs/website/ in the Volura repository, at the
// tag of the latest release: a new version brings its own texts, without touching this site.
// The pages themselves only hold a loading message; when GitHub cannot be reached it becomes
// a message with a link to the texts on GitHub.
(function () {
  var root = document.documentElement;
  var REPO = "Lexvdpoel/Volura";
  var RELEASE_TTL = 60 * 60 * 1000;     // ask GitHub for the latest release at most once an hour
  var FALLBACK = "https://github.com/" + REPO + "/tree/main/docs/website";

  function stored(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }

  function initial() {
    var q = new URLSearchParams(location.search).get("lang");
    if (q === "en" || q === "nl") return q;
    var s = stored("volura-lang");
    if (s === "en" || s === "nl") return s;
    // English, unless the browser's preferred language is Dutch
    var prefs = navigator.languages && navigator.languages.length ? navigator.languages
                                                                  : [navigator.language || "en"];
    return String(prefs[0]).toLowerCase().indexOf("nl") === 0 ? "nl" : "en";
  }

  function setLang(lang, remember) {
    root.setAttribute("data-lang", lang);
    root.setAttribute("lang", lang);
    var t = root.getAttribute("data-title-" + lang);
    if (t) document.title = t;
    document.querySelectorAll(".lang button").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.lang === lang ? "true" : "false");
    });
    // only a click on the switch is remembered; the automatic choice follows the browser
    if (remember) store("volura-lang", lang);
    document.querySelectorAll("[data-placeholder-" + lang + "]").forEach(function (el) {
      el.placeholder = el.getAttribute("data-placeholder-" + lang);
    });
  }

  var lang = initial();
  setLang(lang);

  // ---------------------------------------------------------------- texts from GitHub

  // The latest release as {tag, version, date}, from the cache when it is recent enough: the
  // GitHub API allows 60 requests an hour per visitor.
  function latestRelease(fresh) {
    try {
      var c = JSON.parse(stored("volura-release") || "null");
      if (!fresh && c && c.tag && Date.now() - c.at < RELEASE_TTL) {
        c.cached = true;
        return Promise.resolve(c);
      }
    } catch (e) { /* a damaged cache is asked again */ }
    return fetch("https://api.github.com/repos/" + REPO + "/releases/latest",
                 { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (j) {
        var rel = { tag: j.tag_name, version: String(j.tag_name).replace(/^v/, ""),
                    date: (j.published_at || "").slice(0, 10), at: Date.now() };
        store("volura-release", JSON.stringify(rel));
        return rel;
      });
  }

  function fetchText(tag, name) {
    return fetch("https://raw.githubusercontent.com/" + REPO + "/" + encodeURIComponent(tag) +
                 "/docs/website/" + name + ".html")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); });
  }

  // The fragment as DOM nodes, without anything that could run code: it comes from the
  // repository, but a page should never execute what it loads as text.
  function clean(html) {
    var t = document.createElement("template");
    t.innerHTML = html;
    t.content.querySelectorAll("script, iframe, object, embed, link, meta, base, form")
      .forEach(function (el) { el.remove(); });
    t.content.querySelectorAll("*").forEach(function (el) {
      Array.prototype.slice.call(el.attributes).forEach(function (a) {
        var v = a.value.replace(/\s+/g, "").toLowerCase();
        if (/^on/i.test(a.name) || ((a.name === "href" || a.name === "src") &&
                                     v.indexOf("javascript:") === 0)) {
          el.removeAttribute(a.name);
        }
      });
    });
    return t.content;
  }

  function replace(el, html) {
    var nodes = clean(html);
    // a fragment without content (a broken download) counts as not loaded
    if (!nodes.querySelector("*")) return false;
    el.replaceChildren(nodes);
    return true;
  }

  function showVersion(rel) {
    document.querySelectorAll("[data-version]").forEach(function (el) {
      el.textContent = rel.version + (rel.date ? " · " + rel.date : "");
      el.closest("[data-version-row]") && el.closest("[data-version-row]").removeAttribute("hidden");
    });
  }

  // The loading message of `el` becomes a message with a link to the texts on GitHub.
  function failed(el) {
    var msg = el.querySelector(".remote-msg");
    if (!msg) return;
    msg.replaceChildren();
    [["en", "This text could not be loaded from GitHub. Try again later, or read it ",
      "on GitHub", "."],
     ["nl", "Deze tekst kon niet van GitHub geladen worden. Probeer het later opnieuw, of lees hem ",
      "op GitHub", "."]].forEach(function (t) {
      var span = document.createElement("span");
      span.setAttribute("lang", t[0]);
      var a = document.createElement("a");
      a.href = FALLBACK;
      a.textContent = t[2];
      span.append(t[1], a, t[3]);
      msg.appendChild(span);
    });
  }

  function loadRemote() {
    var parts = document.querySelectorAll("[data-remote]");
    var wantsVersion = document.querySelector("[data-version]");
    if (!parts.length && !wantsVersion) return;
    if (!("fetch" in window) || !("content" in document.createElement("template"))) {
      parts.forEach(failed);
      return;
    }
    // a remembered release that has meanwhile been replaced may have lost its texts: ask
    // GitHub once more before giving up
    function load(el, rel) {
      return fetchText(rel.tag, el.getAttribute("data-remote")).catch(function (err) {
        if (!rel.cached) throw err;
        return latestRelease(true).then(function (now) {
          showVersion(now);
          return fetchText(now.tag, el.getAttribute("data-remote"));
        });
      });
    }

    latestRelease().then(function (rel) {
      showVersion(rel);
      parts.forEach(function (el) {
        load(el, rel).then(function (html) {
          if (!replace(el, html)) { failed(el); return; }
          if (el.hasAttribute("data-has-toc")) {
            buildToc();
            // the page may have been opened on a section that has only now arrived
            if (location.hash) {
              var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
              if (target) target.scrollIntoView();
            }
          }
        }).catch(function () { failed(el); });
      });
    }).catch(function () { parts.forEach(failed); });   // GitHub unreachable or rate limited
  }

  // ---------------------------------------------------------------- manual: contents

  var observer = null;

  // A short id for a heading that has none, from its English text: "Project box" -> "project-box".
  function slug(h, prefix) {
    var en = h.querySelector("span[lang='en']");
    var text = (en ? en.textContent : h.textContent).toLowerCase();
    var base = prefix + "-" + text.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    var id = base, n = 2;
    while (document.getElementById(id)) id = base + "-" + n++;
    return id;
  }

  // The text of a heading in both languages, as a link in the contents.
  function link(h) {
    var a = document.createElement("a");
    a.href = "#" + h.id;
    var spans = h.querySelectorAll("span[lang]");
    if (spans.length) {
      spans.forEach(function (s) {
        var c = document.createElement("span");
        c.setAttribute("lang", s.getAttribute("lang"));
        c.textContent = s.textContent;
        a.appendChild(c);
      });
    } else {
      a.textContent = h.textContent;
    }
    return a;
  }

  function setOpen(li, open) {
    li.classList.toggle("open", open);
    var b = li.querySelector(":scope > .fold");
    if (b) b.setAttribute("aria-expanded", open ? "true" : "false");
  }

  // The table of contents: every chapter (<h2 id>) with its topics (<h3>) folded under it.
  // A click on a chapter opens it; the chapter you are reading opens by itself.
  function buildToc() {
    var list = document.querySelector(".toc ol");
    var doc = document.querySelector("[data-has-toc]");
    if (!list || !doc) return;
    var items = [], chapter = null, sub = null;
    doc.querySelectorAll("h2[id], h3").forEach(function (h) {
      if (h.tagName === "H2") {
        chapter = document.createElement("li");
        chapter.className = "chapter";
        chapter.appendChild(link(h));
        sub = null;
        items.push(chapter);
        return;
      }
      if (!chapter) return;
      if (!h.id) h.id = slug(h, chapter.querySelector("a").getAttribute("href").slice(1));
      if (!sub) {
        sub = document.createElement("ol");
        var fold = document.createElement("button");
        fold.type = "button";
        fold.className = "fold";
        fold.setAttribute("aria-expanded", "false");
        fold.setAttribute("aria-label", "Open / close");
        chapter.insertBefore(fold, chapter.firstChild);
        chapter.appendChild(sub);
      }
      var li = document.createElement("li");
      li.appendChild(link(h));
      sub.appendChild(li);
    });
    if (!items.length) return;
    list.replaceChildren.apply(list, items);
    if (!list.dataset.wired) {
      list.dataset.wired = "1";
      list.addEventListener("click", function (e) {
        var fold = e.target.closest(".fold");
        var li = e.target.closest("li.chapter");
        if (!li) return;
        if (fold) setOpen(li, !li.classList.contains("open"));
        else if (e.target.closest("a") && e.target.closest("a").parentNode === li) setOpen(li, true);
      });
    }
    buildSearch();
    watchToc();
  }

  // A search field above the contents: only the chapters and topics that match stay.
  function buildSearch() {
    var toc = document.querySelector(".toc");
    if (!toc || toc.querySelector(".toc-search")) { filterToc(); return; }
    var input = document.createElement("input");
    input.type = "search";
    input.className = "toc-search";
    input.setAttribute("data-placeholder-en", "Search the manual");
    input.setAttribute("data-placeholder-nl", "Zoek in de handleiding");
    input.placeholder = input.getAttribute("data-placeholder-" + root.getAttribute("data-lang"));
    input.addEventListener("input", filterToc);
    toc.insertBefore(input, toc.firstChild);
  }

  function filterToc() {
    var input = document.querySelector(".toc-search");
    var q = input ? input.value.trim().toLowerCase() : "";
    document.querySelectorAll(".toc li.chapter").forEach(function (ch) {
      var own = ch.querySelector(":scope > a").textContent.toLowerCase().indexOf(q) >= 0;
      var any = false;
      ch.querySelectorAll("ol li").forEach(function (li) {
        var hit = !q || own || li.textContent.toLowerCase().indexOf(q) >= 0;
        li.hidden = !hit;
        any = any || (q && li.textContent.toLowerCase().indexOf(q) >= 0);
      });
      ch.hidden = !!q && !own && !any;
      if (q) setOpen(ch, any);
    });
  }

  // highlight the chapter and the topic on screen, and open that chapter
  function watchToc() {
    if (observer) observer.disconnect();
    var links = document.querySelectorAll(".toc a[href^='#']");
    if (!links.length || !("IntersectionObserver" in window)) return;
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var a = byId[en.target.id];
        if (!a) return;
        links.forEach(function (x) { x.classList.remove("on"); });
        a.classList.add("on");
        var ch = a.closest("li.chapter");
        if (ch) {
          ch.querySelector(":scope > a").classList.add("on");
          var searching = document.querySelector(".toc-search");
          if (!(searching && searching.value.trim())) {
            document.querySelectorAll(".toc li.chapter.open").forEach(function (o) {
              if (o !== ch) setOpen(o, false);
            });
            setOpen(ch, true);
          }
        }
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

  function ready() {
    setLang(root.getAttribute("data-lang") || lang);     // now the buttons exist
    document.addEventListener("click", function (e) {
      var b = e.target.closest(".lang button");
      if (b) setLang(b.dataset.lang, true);
    });
    buildToc();
    loadRemote();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready);
  else ready();
})();
