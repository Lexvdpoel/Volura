"""Check GitHub for a newer version and install it.

No Qt in this file: the main window calls `latest_release` in a background thread
and shows the outcome. A release on GitHub has a tag `vX.Y.Z` and the Windows
installer as the file `Volura-windows-setup.exe`. When Volura was installed with that
installer (there is then an `unins000.exe` next to `Volura.exe`), it can download the
new installer itself and run it silently; the installer then starts Volura again.
Otherwise (zip, Linux, source code) it points to the download page.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import ssl
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .i18n import tr

REPO = "Lexvdpoel/Volura"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
SETUP_ASSET = "Volura-windows-setup.exe"
TIMEOUT = 10.0
CHUNK = 256 * 1024
# The installer with these options: no questions, programs that have the files open are
# closed cleanly; the installer then starts Volura again itself.
SETUP_ARGS = ("/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS")


@dataclass
class Release:
    version: str               # "0.2.0"
    page: str                  # the release page on GitHub
    notes: str = ""            # the text of the release
    setup_url: str | None = None
    setup_size: int = 0
    setup_sha256: str | None = None


def parse_version(text) -> tuple:
    """'v1.2.3' / '1.2' / '1.2.3-beta' -> (1, 2, 3); unreadable -> ()."""
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
    """A Release from the GitHub API response; None for a draft or a pre-release."""
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


def ssl_contexts():
    """The ways to check GitHub's certificate, best first.

    Python on Windows reads only the root certificates that are already in the Windows
    store; Windows itself fetches a missing root the first time it needs one. On a computer
    that never needed GitHub's root, Python then fails with "unable to get issuer
    certificate". truststore lets Windows (or macOS) do the check, as a browser would;
    certifi brings Mozilla's own list of root certificates as a fallback.
    """
    out = []
    try:
        import truststore
        out.append(truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    except Exception:                   # not installed, or the system offers no verifier
        pass
    try:
        import certifi
        out.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    out.append(ssl.create_default_context())
    return out


def _open(url, timeout):
    """urlopen with the first certificate check that works; other errors come out as they are."""
    error = None
    for context in ssl_contexts():
        try:
            return urllib.request.urlopen(_request(url), timeout=timeout, context=context)
        except urllib.error.URLError as e:
            if not isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise
            error = e
    raise error


def latest_release(timeout=TIMEOUT) -> Release | None:
    """The newest release on GitHub; a network error raises OSError/ValueError."""
    with _open(API_LATEST, timeout) as resp:
        return release_from_json(json.loads(resp.read().decode("utf-8")))


def can_self_install(exe=None) -> bool:
    """Only a Windows version installed with the installer can update itself."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    folder = Path(exe or sys.executable).resolve().parent
    return any(folder.glob("unins*.exe"))


def download(release: Release, progress=None, cancelled=None) -> Path:
    """Download the installer to a temporary folder and check its size and SHA-256.

    `progress(done, total)` is called per block; `cancelled()` may return True to stop
    (InterruptedError follows).
    """
    if not release.setup_url:
        raise ValueError(tr("This release has no Windows installer."))
    folder = Path(tempfile.mkdtemp(prefix="volura-update-"))
    path = folder / f"Volura-{release.version}-setup.exe"
    sha = hashlib.sha256()
    done = 0
    with _open(release.setup_url, TIMEOUT) as resp, \
            open(path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or release.setup_size or 0)
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError(tr("Download cancelled"))
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            sha.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    if release.setup_size and done != release.setup_size:
        raise ValueError(tr("The download is incomplete; please try again later."))
    if release.setup_sha256 and sha.hexdigest() != release.setup_sha256:
        raise ValueError(tr("The download is not correct (checksum differs); please try again."))
    return path


def run_installer(path) -> subprocess.Popen:
    """Start the installer detached from Volura; the window must then close itself."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([os.fspath(path), *SETUP_ARGS], close_fds=True, creationflags=flags)
