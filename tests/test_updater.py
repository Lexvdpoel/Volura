"""Controleren op updates en de nieuwe versie zelf installeren, zonder netwerk getest."""
import hashlib
import io
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import __version__, updater  # noqa: E402

PAYLOAD = b"MZ" + b"\0" * 5000


def release_json(tag="v9.0.0", **extra):
    data = {"tag_name": tag, "html_url": "https://github.com/Lexvdpoel/Volura/releases/tag/" + tag,
            "body": "Nieuw: van alles", "draft": False, "prerelease": False,
            "assets": [{"name": "Volura-windows-x86_64.zip", "browser_download_url": "https://x/zip",
                        "size": 10},
                       {"name": updater.SETUP_ASSET, "browser_download_url": "https://x/setup",
                        "size": len(PAYLOAD),
                        "digest": "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()}]}
    data.update(extra)
    return data


class FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_versies_vergelijken():
    assert updater.parse_version("v1.2") == (1, 2, 0)
    assert updater.parse_version("0.10.1-beta") == (0, 10, 1)
    assert updater.parse_version("onzin") == ()
    assert updater.is_newer("v0.10.0", "0.9.9")
    assert not updater.is_newer("v0.1.0", "0.1.0")
    assert not updater.is_newer("", "0.1.0")


def test_release_uit_de_api(monkeypatch):
    monkeypatch.setattr(updater.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResponse(json.dumps(release_json()).encode()))
    rel = updater.latest_release()
    assert rel.version == "9.0.0" and rel.setup_url == "https://x/setup"
    assert rel.setup_sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert updater.release_from_json(release_json(prerelease=True)) is None


def test_download_controleert_grootte_en_controlegetal(monkeypatch):
    rel = updater.release_from_json(release_json())
    monkeypatch.setattr(updater.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResponse(PAYLOAD))
    seen = []
    path = updater.download(rel, lambda d, t: seen.append((d, t)))
    assert path.read_bytes() == PAYLOAD and seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    bad = PAYLOAD[:-1] + b"X"
    monkeypatch.setattr(updater.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResponse(bad))
    with pytest.raises(ValueError):
        updater.download(rel)
    with pytest.raises(InterruptedError):
        updater.download(rel, cancelled=lambda: True)


def test_alleen_een_geinstalleerde_versie_werkt_zichzelf_bij(tmp_path, monkeypatch):
    exe = tmp_path / "Volura.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert not updater.can_self_install(exe)            # uit een zip: geen uninstaller
    (tmp_path / "unins000.exe").write_bytes(b"")
    assert updater.can_self_install(exe)


def test_geen_update_check_in_tests_of_vanaf_broncode():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    from volura.app import MainWindow
    w = MainWindow()
    try:
        assert w.check_for_updates_on_startup() is None
        assert "Controleren op updates…" in [a.text().replace("&", "") for a in w.actions()]
    finally:
        w.doc.dirty = False
        w.close()


def test_melding_bij_een_nieuwere_versie(monkeypatch):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    from volura import app as vapp
    w = vapp.MainWindow()
    offered = []
    monkeypatch.setattr(w, "offer_update", lambda rel: offered.append(rel.version))
    try:
        rel = updater.release_from_json(release_json())
        w.on_update_result(rel, None, automatic=True)
        assert offered == ["9.0.0"]
        same = updater.release_from_json(release_json(tag="v" + __version__))
        w.on_update_result(same, None, automatic=True)
        assert offered == ["9.0.0"]                     # geen melding voor dezelfde versie
    finally:
        w.doc.dirty = False
        w.close()
