"""The user interface of Volura (English labels), grown from SketchUp 2017.

Recorded from SketchUp Make 2017 (menus, submenus, default shortcuts, the
"Getting Started" toolbar and the Default Tray) and since then arranged for Volura:
the 3D printing and the laser/mill commands have a menu of their own, and the menus
show the commands of the 3D model or of the 2D worksheet, depending on the mode.
Each menu lists its items in order as (label, shortcut). A label ending in ">" is a
submenu, listed in SUBMENUS. "---" is a separator. Volura specific items are marked
with a comment.

The menu titles and item labels must match exactly (ignoring "&" accelerators
and a trailing "…"). Shortcuts are the SketchUp defaults.
"""

MENUS = [
    ("File", [
        ("New", "Ctrl+N"),
        ("Open", "Ctrl+O"),
        "---",
        ("Save", "Ctrl+S"),
        ("Save As", None),
        ("Save a Copy As", None),
        "---",
        ("Import", None),
        ("Export >", None),
        "---",
        ("Print", "Ctrl+P"),                # SketchUp: Print -> export 3MF for the slicer
        "---",
        ("Exit", None),
    ]),
    ("Edit", [
        ("Undo", "Alt+Backspace"),
        ("Redo", "Ctrl+Y"),
        "---",
        ("Cut", "Shift+Del"),
        ("Copy", "Ctrl+C"),
        ("Paste", "Ctrl+V"),
        ("Paste In Place", None),
        ("Delete", "Del"),
        ("Delete Guides", None),
        "---",
        ("Select All", "Ctrl+A"),
        ("Select None", "Ctrl+T"),
        "---",
        ("Hide", None),
        ("Unhide >", None),
        "---",
        ("Lock", None),
        ("Unlock >", None),
        "---",
        ("Make Component", "G"),
        ("Make Group", None),
        ("Close Group/Component", None),
    ]),
    ("View", [
        ("Toolbars", None),
        "---",
        ("Hidden Geometry", None),
        ("Axes", None),
        ("Guides", None),
        ("Print Bed", None),                # Volura
        "---",
        ("Edge Style >", None),
        ("Face Style >", None),
        ("Component Edit >", None),
    ]),
    ("Camera", [
        ("Previous", None),
        ("Next", None),
        ("Standard Views >", None),
        "---",
        ("Parallel Projection", None),
        ("Perspective", None),
        "---",
        ("Orbit", "O"),
        ("Pan", "H"),
        ("Zoom", "Z"),
        ("Field of View", None),
        ("Zoom Window", "Ctrl+Shift+W"),
        ("Zoom Extents", "Ctrl+Shift+E"),
    ]),
    ("Draw", [
        ("Lines >", None),
        ("Arcs >", None),
        ("Shapes >", None),
        "---",
        ("Basic Shapes >", None),           # Volura (SketchUp: Sandbox position)
        ("Parts >", None),                  # Volura: the parts it builds for you
    ]),
    ("Tools", [
        ("Select", "Space"),
        ("Eraser", "E"),
        ("Paint Bucket", "B"),
        "---",
        ("Move", "M"),
        ("Rotate", "Q"),
        ("Scale", "S"),
        "---",
        ("Push/Pull", "P"),
        ("Follow Me", None),
        ("Offset", "F"),
        "---",
        ("Solid Tools >", None),            # Volura: grouped with the other solid commands
        ("Intersect Faces >", None),        # SketchUp: in Edit
        ("Outer Shell", None),
        "---",
        ("Screw Threads >", None),          # Volura
        ("Screw Hole", None),               # Volura
        "---",
        ("Tape Measure", "T"),
        ("Protractor", None),
        "---",
        ("3D Text", None),
    ]),
    ("3D Printing", [                       # Volura: only with the 3D model
        ("Printer Setup", None),            # SketchUp: Print Setup  -> printbed / printer
        ("Print Check", "F5"),              # SketchUp: Print Preview -> print check
        "---",
        ("Orient for Printing", None),
        ("Drop to Bed", "Ctrl+B"),
        ("Center on Bed", None),
        ("Arrange on Bed", None),
        "---",
        ("Cut", None),
        ("Cut and Join", None),
        "---",
        ("Elephant Foot Chamfer", None),
        ("Mouse Ears", None),
        "---",
        ("Clearance Test", None),
        ("Overhang Test", None),
    ]),
    ("Laser and Mill", [                    # Volura: only on the 2D worksheet
        ("Sheet", None),
        ("Auto Arrange", None),
        "---",
        ("Fit Test", None),
        "---",
        ("Holding Tabs", None),
        ("Dogbone Corners", None),
        "---",
        ("Job Estimate", None),
    ]),
    ("Window", [
        ("Default Tray >", None),
        "---",
        ("Model Info", None),
        ("Preferences", None),
    ]),
    ("Help", [
        ("Welcome to Volura", None),
        ("Keyboard Shortcuts", None),
        "---",
        ("About Volura", None),
    ]),
]

SUBMENUS = {
    ("File", "Export"): [("3D Model (STL)", None), ("Multicolor 3MF", None), ("STL per Color", None)],
    ("Edit", "Unhide"): [("Selected", None), ("Last", None), ("All", None)],
    ("Edit", "Unlock"): [("Selected", None), ("All", None)],
    ("Tools", "Intersect Faces"): [("With Model", None), ("With Selection", None)],
    ("Tools", "Screw Threads"): [("Thread", None), ("External Thread on Cylinder", None),
                                       ("Internal Thread in Hole", None), ("New Bolt", None),
                                       ("New Nut or Sleeve", None)],
    ("Draw", "Parts"): [("Spur Gear", None), ("Project Box", None), ("Knob", None),
                               ("Standoff", None), ("Label Plate", None), ("Cable Clip", None)],
    ("View", "Edge Style"): [("Edges", None), ("Back Edges", "K")],
    ("View", "Face Style"): [("X-ray", None), "---", ("Wireframe", None), ("Hidden Line", None),
                             ("Shaded", None), ("Monochrome", None)],
    ("View", "Component Edit"): [("Hide Rest of Model", None)],
    ("Camera", "Standard Views"): [("Top", None), ("Bottom", None), ("Front", None), ("Back", None),
                                   ("Left", None), ("Right", None), ("Iso", None)],
    ("Draw", "Lines"): [("Line", "L"), ("Freehand", None)],
    ("Draw", "Arcs"): [("Arc", None), ("2 Point Arc", "A"), ("3 Point Arc", None), ("Pie", None)],
    ("Draw", "Shapes"): [("Rectangle", "R"), ("Rotated Rectangle", None), ("Circle", "C"), ("Polygon", None)],
    ("Draw", "Basic Shapes"): [("Box", None), ("Cylinder", None), ("Cone", None), ("Sphere", None),
                               ("Torus", None)],
    ("Tools", "Solid Tools"): [("Intersect", None), ("Union", None), ("Subtract", None),
                               ("Trim", None), ("Split", None)],
    ("Window", "Default Tray"): [("Entity Info", None), ("Materials", None), ("Outliner", None)],
}

# The "Getting Started" toolbar, left to right ("|" = separator). Volura replaces the
# 3D Warehouse / Extension buttons at the end with its print buttons.
TOOLBAR = ["Select", "Eraser", "Line", "2 Point Arc", "Rectangle", "Push/Pull", "Offset",
           "Move", "Rotate", "Scale", "|", "Tape Measure", "3D Text", "Paint Bucket", "|", "Orbit",
           "Pan", "Zoom", "Zoom Extents", "|", "Print Check", "Print"]

# Buttons with a drop-down arrow in SketchUp: toolbar label -> the drawing tools it offers
TOOLBAR_DROPDOWNS = {
    "Line": ["Line", "Freehand"],
    "2 Point Arc": ["Arc", "2 Point Arc", "3 Point Arc", "Pie"],
    "Rectangle": ["Rectangle", "Rotated Rectangle", "Circle", "Polygon"],
}

# Right click on a face (SketchUp order). Items for groups are added after "Delete"/"Hide".
CONTEXT_FACE = ["Entity Info", "Delete", "Hide", "---", "Select >", "Area >",
                "Intersect Faces >", "Align View", "Reverse Faces", "Orient Faces",
                "Zoom Selection"]

# The Default Tray: one dock on the right with collapsible panels in this order
TRAY_TITLE = "Default Tray"
TRAY_PANELS = ["Entity Info", "Materials", "Outliner"]

# The Measurements box in the status bar (bottom right)
MEASUREMENTS_LABEL_DEFAULT = "Measurements"
