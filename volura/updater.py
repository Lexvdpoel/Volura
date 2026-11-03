"""Controleren op een nieuwere versie op GitHub en die zelf installeren.

Geen Qt in dit bestand: het hoofdvenster roept `latest_release` in een achtergrond-
draad aan en toont de uitkomst. Een release op GitHub heeft een tag `vX.Y.Z` en de
Windows-installatie als bestand `Volura-windows-setup.exe`. Is Volura met die
installatie geïnstalleerd (er staat dan een `unins000.exe` naast `Volura.exe`), dan
kan het de nieuwe installatie zelf downloaden en stil laten draaien; die start
Volura daarna opnieuw. Anders (zip, Linux, broncode) wijst het naar de downloadpagina.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__

REPO = "Lexvdpoel/Volura"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
SETUP_ASSET = "Volura-windows-setup.exe"
TIMEOUT = 10.0
CHUNK = 256 * 1024
# De installatie met deze opties: geen vragen, programma's die het bestand open hebben
# netjes sluiten; de installatie start Volura daarna zelf weer.
SETUP_ARGS = ("/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS")


@dataclass
class Release:
    version: str               # "0.2.0"
    page: str                  # de releasepagina op GitHub
    notes: str = ""            # de tekst van de release
    setup_url: str | None = None
    setup_size: int = 0
    setup_sha256: str | None = None


def parse_version(text) -> tuple:
    """'v1.2.3' / '1.2' / '1.2.3-beta' -> (1, 2, 3); onleesbaar -> ()."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(text or ""))
    if not m:
        return ()
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_newer(latest, current=__version__) -> bool:
    a, b = parse_version(latest), parse_version(current)
    return bool(a) and bool(b) and a > b


def release_from_json(data) -> Release | None:
    """Een Release uit het antwoord van de GitHub-API; None bij een concept of voorversie."""
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    version = ".".join(str(n) for n in parse_version(data.get("tag_name")))
    if not version:
        return None
    rel = Release(version, str(data.get("html_url") or RELEASES_PAGE), str(data.get("body") or ""))
    for asset in data.get("assets") or ():
        if asset.get("name") == SETUP_ASSET:
            rel.setup_url = asset.get("browser_download_url")
            rel.setup_size = int(asset.get("size") or 0)
            digest = str(asset.get("digest") or "")
            if digest.startswith("sha256:"):
                rel.setup_sha256 = digest.split(":", 1)[1].lower()
    return rel


def _request(url):
    return urllib.request.Request(url, headers={
        "User-Agent": f"Volura/{__version__}",
        "Accept": "application/vnd.github+json"})


def latest_release(timeout=TIMEOUT) -> Release | None:
    """De nieuwste release op GitHub; een netwerkfout geeft een OSError/ValueError."""
    with urllib.request.urlopen(_request(API_LATEST), timeout=timeout) as resp:
        return release_from_json(json.loads(resp.read().decode("utf-8")))


def can_self_install(exe=None) -> bool:
    """Alleen een met de installatie geïnstalleerde Windows-versie kan zichzelf bijwerken."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    folder = Path(exe or sys.executable).resolve().parent
    return any(folder.glob("unins*.exe"))


def download(release: Release, progress=None, cancelled=None) -> Path:
    """Download de installatie naar een tijdelijke map en controleer de grootte en SHA-256.

    `progress(gedaan, totaal)` wordt per blok aangeroepen; `cancelled()` mag True geven
    om te stoppen (dan volgt InterruptedError).
    """
    if not release.setup_url:
        raise ValueError("Deze release heeft geen Windows-installatie.")
    folder = Path(tempfile.mkdtemp(prefix="volura-update-"))
    path = folder / f"Volura-{release.version}-setup.exe"
    sha = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(_request(release.setup_url), timeout=TIMEOUT) as resp, \
            open(path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or release.setup_size or 0)
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("Download afgebroken")
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            sha.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    if release.setup_size and done != release.setup_size:
        raise ValueError("De download is onvolledig; probeer het later opnieuw.")
    if release.setup_sha256 and sha.hexdigest() != release.setup_sha256:
        raise ValueError("De download klopt niet (controlegetal wijkt af); probeer het opnieuw.")
    return path


def run_installer(path) -> subprocess.Popen:
    """Start de installatie los van Volura; het venster moet daarna zelf afsluiten."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([os.fspath(path), *SETUP_ARGS], close_fds=True, creationflags=flags)
