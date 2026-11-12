"""The standalone build ships the licences of everything PyInstaller bundles."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("licenses", ROOT / "packaging" / "licenses.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_third_party_licenses(tmp_path):
    out = load().write(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert (tmp_path / "LICENSE.txt").read_text(encoding="utf-8").startswith("MIT License")
    for needle in ("GNU LESSER GENERAL PUBLIC LICENSE", "GNU GENERAL PUBLIC LICENSE",
                   "https://code.qt.io", "Apache License", "Michael C. Fletcher",
                   "NumPy", "Python", "GNU AFFERO GENERAL PUBLIC LICENSE",
                   "https://github.com/Ultimaker/CuraEngine/tree/", "CC0"):
        assert needle in text, needle
    # the Qt commercial licence of the wheels is not ours to ship
    assert "LicenseRef-Qt-Commercial" not in text


def test_build_writes_licenses():
    src = (ROOT / "packaging" / "build.py").read_text(encoding="utf-8")
    assert "licenses.write(app)" in src
    assert os.path.isfile(ROOT / "packaging" / "licenses" / "LGPL-3.0.txt")
