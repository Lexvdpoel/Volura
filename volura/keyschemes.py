"""The keyboard and mouse schemes of Volura: SketchUp and Autodesk Fusion.

People who learned Fusion keep their hands where they learned them: Push/Pull on E,
Offset on O, Fillet on F, and the middle mouse button pans instead of orbiting. The
table here holds only what is different in that scheme, so the shortcuts the actions
are built with stay the only place the SketchUp keys are written down.

A command is the English menu path under which its shortcuts are stored (the keys of
QSettings "shortcuts/<path>", see `window_ui`), a key is the text of
`QKeySequence.PortableText`. The 2D worksheet needs nothing of this: its keys live on
the worksheet itself (`sheetview.SHEET_KEYS`) and are the letters Fusion uses there
too, with O for Offset.
"""
from __future__ import annotations

#: what Preferences > General > Keyboard and mouse offers
SCHEMES = ("sketchup", "fusion")
SKETCHUP, FUSION_SCHEME = SCHEMES

#: the commands whose keys differ in the Fusion scheme; a command that is not here keeps
#: its default keys, an empty list takes them away
FUSION = {
    "Tools/Push/Pull": ["E", "Q"],
    "Tools/Offset": ["O"],
    "Tools/Fillet": ["F"],
    "Tools/Tape Measure": ["I"],
    "Tools/Eraser": [],                   # Del still deletes the selection
    "Tools/Rotate": [],
    "Tools/Scale": [],
    "Camera/Orbit": [],                   # Fusion steers the view with the mouse
    "Camera/Pan": [],
    "Camera/Zoom": [],
    "Camera/Zoom Extents": ["Ctrl+Shift+E", "Shift+Z", "F6"],
}


def scheme_of(choice) -> str:
    """The scheme a setting names; anything unknown is the scheme Volura does now."""
    return choice if choice in SCHEMES else SKETCHUP


def scheme_defaults(scheme, defaults):
    """{command path: [key texts]} of every command, for one scheme.

    `defaults` are the shortcuts Volura ships with, which are the SketchUp ones. The
    keys of the scheme replace the list of their command, and every key the scheme
    gives to a command is taken away from the command that had it, so no key ends up
    on two commands.
    """
    out = {path: list(texts) for path, texts in defaults.items()}
    if scheme_of(scheme) != FUSION_SCHEME:
        return out
    for path, texts in FUSION.items():
        out[path] = list(texts)
    given = {text for texts in FUSION.values() for text in texts}
    for path, texts in out.items():
        if path not in FUSION:
            out[path] = [t for t in texts if t not in given]
    return out
