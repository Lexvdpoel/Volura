"""Volura is built for Windows and Linux, so nothing in `volura/` may assume Windows.

Every module of the package is imported (a module level `import winreg`, or a
name that only resolves because Windows ignores letter case, fails here), and
the sources are scanned for the things that only exist on Windows: `os.startfile`
and hard coded paths such as "C:\\Users\\...".

Runs without a display (Qt offscreen platform).
"""
import importlib
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "volura"

MODULES = sorted(p.stem for p in PACKAGE.glob("*.py"))

# A single drive letter, a colon and a backslash: "C:\Users", "D:\\data", ...
DRIVE_PATH = re.compile(r"\b[A-Za-z]:\\")
STARTFILE = re.compile(r"\bos\.startfile\b")
WINDOWS_ONLY = re.compile(
    r"\b(?:import|from)\s+(?:winreg|win32api|win32con|win32gui|win32process|pywintypes|pythoncom)\b"
    r"|\bctypes\.windll\b|\bmsvcrt\b"
)
RELATIVE_IMPORT = re.compile(r"^\s*from\s+\.([A-Za-z_][\w.]*)\s+import\b", re.MULTILINE)


def sources():
    return sorted(PACKAGE.glob("*.py"))


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def test_package_has_modules():
    assert len(MODULES) > 10, MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    """An import that only works on Windows, or a circular import, shows up here."""
    module = importlib.import_module(f"volura.{name}")
    assert module is not None


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_no_os_startfile(path):
    """`os.startfile` is Windows only; use QDesktopServices.openUrl instead."""
    text = path.read_text(encoding="utf-8")
    for m in STARTFILE.finditer(text):
        pytest.fail(f"{path.name}:{line_of(text, m.start())}: os.startfile is Windows only")


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_no_hardcoded_drive_paths(path):
    """A path with a drive letter does not exist on Linux; use QStandardPaths or Path.home()."""
    text = path.read_text(encoding="utf-8")
    for m in DRIVE_PATH.finditer(text):
        pytest.fail(f"{path.name}:{line_of(text, m.start())}: schijfletter-pad {m.group(0)!r}")


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_no_windows_only_modules(path):
    text = path.read_text(encoding="utf-8")
    for m in WINDOWS_ONLY.finditer(text):
        pytest.fail(f"{path.name}:{line_of(text, m.start())}: {m.group(0)!r} is Windows only")


def test_file_names_are_lowercase():
    """Linux file names are case sensitive, so a module file must be spelled exactly as it is imported."""
    bad = [p.name for p in sources() if p.name != p.name.lower()]
    assert not bad, f"hoofdletters in bestandsnaam: {bad}"


def test_relative_imports_match_a_file_exactly():
    """`from .Viewport import ...` works on Windows and breaks on Linux."""
    names = {p.stem for p in sources()}
    problems = []
    for path in sources():
        text = path.read_text(encoding="utf-8")
        for m in RELATIVE_IMPORT.finditer(text):
            top = m.group(1).split(".")[0]
            if top not in names:
                problems.append(f"{path.name}:{line_of(text, m.start())}: .{top}")
    assert not problems, f"relatieve import zonder bestand: {problems}"


def test_settings_and_logs_do_not_need_an_appdata_folder(tmp_path, monkeypatch):
    """QStandardPaths picks the right folder per platform; the log dir must simply be writable."""
    from PySide6.QtCore import QStandardPaths

    from volura import errors

    monkeypatch.setattr(QStandardPaths, "writableLocation", staticmethod(lambda t: str(tmp_path)))
    folder = errors.log_dir()
    assert Path(folder) == tmp_path
    assert Path(folder).is_dir()


def packaging_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("volura_build", ROOT / "packaging" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_script_knows_the_current_platform():
    build = packaging_module()
    system, arch = build.target()
    assert system in {"windows", "linux", "macos"}
    assert arch == arch.lower() and " " not in arch


def test_build_names_the_archive_per_platform(tmp_path, monkeypatch):
    """Linux must end up with dist/Volura-linux-x86_64.tar.gz, Windows and macOS with a zip."""
    build = packaging_module()
    monkeypatch.setattr(build, "DIST", tmp_path)
    made = {}
    for system, exe in (("linux", "Volura"), ("windows", "Volura.exe"), ("macos", "Volura.app")):
        app = tmp_path / "Volura"
        if app.exists():
            shutil.rmtree(app)
        app.mkdir(parents=True)
        built = app / exe
        if exe.endswith(".app"):
            built.mkdir()
        else:
            built.write_bytes(b"placeholder")
        made[system] = build.archive(system, "x86_64").name
    assert made["linux"] == "Volura-linux-x86_64.tar.gz"
    assert made["windows"] == "Volura-windows-x86_64.zip"
    assert made["macos"] == "Volura-macos-x86_64.zip"


def test_linux_desktop_entry_and_installer():
    """The menu item belongs in the graphics category and the app in ~/.local/opt/volura."""
    desktop_path = ROOT / "packaging" / "linux" / "volura.desktop"
    installer_path = ROOT / "packaging" / "linux" / "install.sh"
    desktop = desktop_path.read_text(encoding="utf-8")
    installer = installer_path.read_text(encoding="utf-8")

    assert desktop.startswith("[Desktop Entry]")
    assert "Categories=Graphics;3DGraphics;Engineering;" in desktop
    assert "$HOME/.local/opt/volura" in installer
    assert "$HOME/.local/share/applications" in installer
    for path in (desktop_path, installer_path):
        # shell and desktop files run with LF, also when the repository is checked out on Windows
        assert b"\r\n" not in path.read_bytes(), f"{path.name} bevat Windows-regeleinden"
