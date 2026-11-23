"""The version information of Volura.exe and of the installer.

Right-click > Properties > Details of the built program has to show a product name, a product
version and a publisher: the antivirus reputation systems and the code-signing services look at
them. packaging/build.py writes the resource of Volura.exe from the version in volura/__init__.py,
in the text format of PyInstaller's --version-file, and packaging/windows/volura.iss repeats the
same fields on the installer.
"""
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = "volura_version_info.txt"
VERSION_TEXT = "__version__ = \"0.2.80.dev1\""


def load():
    spec = importlib.util.spec_from_file_location("packaging_build", ROOT / "packaging" / "build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def string_structs(text):
    """{'ProductName': 'Volura', ...} of the StringStruct lines of the version file."""
    return dict(re.findall(r"StringStruct\('(\w+)', '([^']*)'\)", text))


def captured_runs(tmp_path, monkeypatch, build):
    """build.py working in tmp_path instead of the tree, with the PyInstaller call captured;
    returns the list that collects the commands (nothing is built)."""
    (tmp_path / "volura").mkdir()
    (tmp_path / "volura" / "__init__.py").write_text(VERSION_TEXT + "\n", encoding="utf-8")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(build.subprocess, "run",
                        lambda cmd, **kwargs: commands.append([str(c) for c in cmd]))
    return commands


def test_every_field_of_the_resource_is_set():
    found = string_structs(load().version_info_text("0.2.79"))
    assert sorted(found) == ["CompanyName", "FileDescription", "FileVersion", "InternalName",
                             "LegalCopyright", "OriginalFilename", "ProductName", "ProductVersion"]
    for name in ("CompanyName", "FileDescription", "InternalName", "ProductName"):
        assert found[name] == "Volura", name
    assert found["OriginalFilename"] == "Volura.exe"
    assert found["FileVersion"] == "0.2.79"
    assert found["ProductVersion"] == "0.2.79"
    assert found["LegalCopyright"] == "(c) 2026 L. van der Poel, MIT License"


def test_the_binary_version_is_a_four_tuple():
    text = load().version_info_text("0.2.79")
    assert "filevers=(0, 2, 79, 0)" in text
    assert "prodvers=(0, 2, 79, 0)" in text


@pytest.mark.parametrize("version, numbers", [
    ("0.2.79", (0, 2, 79, 0)),
    ("0.2.80.dev1", (0, 2, 80, 0)),      # a development version before a release
    ("1.0rc1", (1, 0, 0, 0)),
    ("2", (2, 0, 0, 0)),
    ("1.2.3.4.5", (1, 2, 3, 4)),
    ("", (0, 0, 0, 0)),                  # no version at all
    ("nightly", (0, 0, 0, 0)),           # nothing to count
])
def test_odd_versions_give_sane_tuples(version, numbers):
    build = load()
    assert build.version_numbers(version) == numbers
    text = build.version_info_text(version)
    assert f"filevers={numbers!r}" in text
    assert f"prodvers={numbers!r}" in text
    # FileVersion keeps what is in volura/__init__.py, so the dev suffix is not lost
    assert string_structs(text)["FileVersion"] == version


def test_the_version_of_this_tree_is_understood():
    build = load()
    version = build.version()
    numbers = build.version_numbers(version)
    assert len(numbers) == 4
    assert numbers[0] == int(version.split(".")[0])
    assert string_structs(build.version_info_text(version))["ProductVersion"] == version


def test_the_text_is_a_single_expression():
    # PyInstaller reads --version-file with eval(), so the whole file has to be one expression
    compile(load().version_info_text("0.2.79"), VERSION_FILE, "eval")


def parse_with_pyinstaller(tmp_path, text):
    """The VSVersionInfo of the text, as PyInstaller reads --version-file; None when this
    Python has no PyInstaller to read it with."""
    try:
        from PyInstaller.utils.win32 import versioninfo
    except Exception:      # not installed, or not usable on this platform
        return None
    path = tmp_path / VERSION_FILE
    path.write_text(text, encoding="utf-8")
    if hasattr(versioninfo, "load_version_info_from_text_file"):      # PyInstaller 6
        return versioninfo.load_version_info_from_text_file(str(path))
    info = versioninfo.VSVersionInfo()                                 # before that
    info.load(str(path))
    return info


def four_tuple(high, low):
    return (high >> 16, high & 0xFFFF, low >> 16, low & 0xFFFF)


@pytest.mark.parametrize("version, numbers", [
    ("0.2.79", (0, 2, 79, 0)),
    ("0.2.80.dev1", (0, 2, 80, 0)),
    ("1.0rc1", (1, 0, 0, 0)),
])
def test_pyinstaller_reads_the_text(tmp_path, version, numbers):
    info = parse_with_pyinstaller(tmp_path, load().version_info_text(version))
    if info is None:
        pytest.skip("PyInstaller is not installed")
    ffi = info.ffi
    assert four_tuple(ffi.fileVersionMS, ffi.fileVersionLS) == numbers
    assert four_tuple(ffi.productVersionMS, ffi.productVersionLS) == numbers
    found = {}
    for kid in info.kids:
        if getattr(kid, "name", "") != "StringFileInfo":
            continue
        for table in kid.kids:
            for struct in table.kids:
                found[struct.name] = struct.val
    assert found["ProductName"] == "Volura"
    assert found["ProductVersion"] == version
    assert found["CompanyName"] == "Volura"
    assert found["OriginalFilename"] == "Volura.exe"
    assert info.toRaw()                    # what PyInstaller writes into Volura.exe


def test_the_windows_build_passes_the_version_file(tmp_path, monkeypatch):
    build = load()
    commands = captured_runs(tmp_path, monkeypatch, build)
    build.pyinstaller("windows")
    cmd = commands[0]
    path = Path(cmd[cmd.index("--version-file") + 1])
    assert path.name == VERSION_FILE
    assert path.parent == tmp_path / "build"
    text = path.read_text(encoding="utf-8")
    assert string_structs(text)["ProductVersion"] == "0.2.80.dev1"
    assert "filevers=(0, 2, 80, 0)" in text


@pytest.mark.parametrize("system", ["linux", "macos"])
def test_the_other_builds_stay_as_they_were(tmp_path, monkeypatch, system):
    build = load()
    commands = captured_runs(tmp_path, monkeypatch, build)
    build.pyinstaller(system)
    assert "--version-file" not in commands[0]
    assert "main.py" in commands[0]         # the command was really made, not broken
    assert not (tmp_path / "build").exists()     # and no version file was written


def test_the_installer_shows_the_same_as_the_program():
    iss = (ROOT / "packaging" / "windows" / "volura.iss").read_text(encoding="utf-8")
    assert '#define AppName "Volura"' in iss
    assert "VersionInfoCompany={#AppName}" in iss
    assert "VersionInfoDescription={#AppName}" in iss
    assert f"VersionInfoCopyright={load().LEGAL_COPYRIGHT}" in iss
