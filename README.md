# Volura

**Volura** is an open source 3D modelling program for 3D printing, laser cutting and milling, for
Windows and Linux. It works like SketchUp 2017: you draw a flat face and turn it into a volume with
**Push/Pull**.

Website and manual: [volura.dev](https://volura.dev) · Downloads: [Releases](https://github.com/Lexvdpoel/Volura/releases/latest) · What changed: [Changelog](CHANGELOG.md)

Just like in SketchUp, a model consists of **edges and faces**: edges split each other where they
cross, a closed shape becomes a face, a line across a face splits it, and the Eraser removes (parts
of) edges, after which faces merge again. Push/Pull works on any face or part of a face and extends
or cuts adjacent faces exactly like SketchUp.

For 3D printing, Volura checks on export that every part is a closed volume (no holes, open or
doubled edges) and writes it flawlessly as STL or 3MF through the Manifold kernel.

The interface is available in English and Dutch (*Window > Preferences > General > Language*).

## Features

- SketchUp workflow: Line, Rectangle, Circle, Polygon, Arcs (2 Point Arc, Arc, 3 Point Arc and
  Pie), 3D Text, Push/Pull, Offset, Move, Rotate, Scale, Tape Measure, Paint Bucket, Eraser,
  Orbit/Pan/Zoom
- Inference like in SketchUp: endpoint, midpoint, origin, on edge, on face, red/green/blue axis,
  and axis + edge combined. Arrow keys or Shift lock a direction
- Length snapping to 1 mm (adjustable), angles snap to 15°
- Measurements box at the bottom right (just type): `25`, `2,5cm`, `40;20`, `64s`, `10;0;5`, `5x`, `/5`.
  The direction follows your mouse, like in SketchUp
- **Arrays**: after a Ctrl-copy, type `5x` (or `x5`, `5*`) to repeat it into 5 copies and `/5` to
  divide the step into 5. With Move this makes a row, with Rotate a circle (rotate-copy a spoke
  60°, type `5x` for 6 spokes); one undo removes the whole array
- Selecting: click, double-click (face + edges), triple-click (everything connected), selection box
- **Groups** (G): every group is a separate printable part with its own filament color. Double-click
  to edit a group
- **Multicolor**: hold **Alt** while pushing/pulling. Inwards makes an *inlay* (pocket + filling as a
  separate group), outwards a separate part. Ctrl works like in SketchUp (new starting face)
- Solid tools on groups: union, subtract, intersect, trim, split, outer shell, mirror, place on bed
- **Intersect Faces** (Edit or right-click): *With Model* adds the cut lines with the rest of the
  model as edges so faces split, *With Selection* only between the selected items
- Primitives: box, cylinder, cone, sphere, torus
- Display styles under **View**: Wireframe, Hidden Line, Shaded and Monochrome (View > Face Style),
  X-ray (Alt+X), Back Edges (K), Hidden Geometry (softened edges) and hiding the rest of the model
  while editing a group
- **Dark theme**: *Window > Preferences > Display > Theme*: System, Light or Dark
- **3Dconnexion SpaceMouse** support (*Window > Preferences > 3D Mouse*): through 3DxWare 10 or the
  older 3DxSoftware (siappdll), without a driver through Windows Raw Input, and on Linux through
  spacenavd
- Print bed of your printer (Bambu, Prusa, Creality, Voron or a custom size) and a print check
- **Arrange on Bed** (Tools or right-click): lay the selected groups, or all groups when nothing is
  selected, side by side on the print bed without overlapping, like a slicer does; parts are turned
  0, 90, 180 or 270 degrees and stand on the bed, 5 mm from the edge and from each other
- **Lay Flat on This Face** (right-click a face of a group): turn that part about its own centre until
  the face you clicked lies on the print bed, like "place on face" of a slicer
- **Screw Hole** (Tools > Print Tools): turn a circle you drew on a part into a printable hole
  for a metric screw (M2 up to M8): straight, with a counterbore for a socket head, a 90°
  countersink for a flat head or a hexagonal pocket that holds a nut; the clearance (0.2 mm
  by default) is added to every diameter, because a printed hole comes out smaller than drawn
- **Elephant Foot Chamfer** (Tools > Print Tools): a small chamfer along the bottom edges of the
  selected groups, or of all visible groups when nothing is selected, so the squashed first layer of
  the print no longer sticks out ("elephant's foot"); the size is 0.4 mm by default and is
  remembered, and parts that stand on a point or a tilted edge are left alone
- **Overhang view** (View > Face Style > Overhangs): marks every face that hangs over more steeply
  than the limit (45° from the vertical by default, under *Window > Preferences > Drawing*) in a
  warning colour, so you see at a glance what needs support; faces that lie on the print bed are
  left alone and the status bar counts the faces that need it
- **Model Info** (Window) with the pages Units (length rounding and the number of decimals of the
  Measurements box), Statistics (groups, loose edges and faces, total volume) and Print Bed
- **Filament** (File > Printer Setup > Print Bed): the material (PLA, PETG, ABS, ASA, TPU, Nylon
  or a custom density), the diameter (1.75 or 2.85 mm) and the price per kg; **Entity Info** then
  shows the weight, length and cost of a selected part as a solid print (with infill it is less).
  Select several groups and Entity Info also shows the totals of the whole job: the volume,
  weight, filament length and cost of all the closed parts together
- **Preferences** (Window) with circle segments and a Shortcuts page like in SketchUp: search,
  commands per menu, add or remove a shortcut and reset everything
- A welcome screen at startup (Help > Welcome to Volura), which can be turned off
- Export:
  - **STL**: all (or the selected) parts merged into one watertight object
  - **Multicolor 3MF**: one object with one part per filament color, already assigned to the right
    extruder (PrusaSlicer, OrcaSlicer, Bambu Studio); the colors you use are numbered 1, 2, 3, ...
    in the order of the filament panel. Faces painted with the Paint Bucket in another
    color are exported as painted triangles, which these slicers show and print as their own color
    painting
  - **STL per color**
  - **3D model with colors (OBJ)**: one object per part, each in its own color from the `.mtl`
    that is written next to it
  - Where parts with different colors overlap, the part made later wins
- Native file format `.volura` (files from version 0.1 are converted), unlimited undo
- **3D import**: STL, OBJ, 3MF, PLY, OFF, AMF, glTF/GLB and COLLADA (DAE), with a progress
  bar. Every closed part becomes a group in its own color; slicer projects (PrusaSlicer,
  Bambu Studio, OrcaSlicer) keep their parts and filament colors. A model that is not closed
  is repaired on import where possible: faces on both sides or the wrong way round, holes,
  parts touching along an edge, small knots and the inner walls of SketchUp models. A model
  that comes in tiny (made in inches, centimetres or metres) can be scaled up right away

### Laser and mill

- **2D worksheet** next to the 3D model (Ctrl+Shift+2 or the 3D | 2D switch at the bottom right), saved
  in the same file: lines, rectangles, circles, arcs, polygons and text; offset, union, subtract,
  align and distribute
- **SVG and DXF** import and export; place a drawing on a face and extrude it
- **Operation layers** in the LightBurn colors: cut, engrave, raster, mill outside/inside, pocket and
  drill, with kerf and tool compensation and dogbones or T-bones
- **Sheet size and automatic arrange** (nesting) over one or more sheets
- **Boxes with finger joints**, on the worksheet or as a 3D preview
- **Model and worksheet together**: right-click a face or sheet > *Laser / Mill* puts it on a layer of
  the worksheet; *Parts to 3D Model* places everything that is cut as a sheet with the sheet thickness
  in 3D

## Installation

### Windows

Download [`Volura-windows-setup.exe`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-windows-setup.exe)
and run it. The installer needs no administrator rights (it installs for you only, in
`%LOCALAPPDATA%\Programs\Volura`), creates a Start menu shortcut and opens `.volura` files with
Volura. Uninstall through *Settings > Apps*. The installer is available in English and Dutch.

Prefer no installation? Unpack [`Volura-windows-x86_64.zip`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-windows-x86_64.zip)
and start `Volura\Volura.exe`. Your settings and your unsaved model live in the app's own data folder
(`%APPDATA%\Volura`).

### Updates

At startup Volura checks once a day whether there is a newer version on GitHub (can be turned off in
*Window > Preferences*, or check yourself with *Help > Check for Updates…*). A Volura installed with
the installer downloads the new version, verifies it, installs it and restarts. The zip, Linux and
source versions open the download page.

### Linux

Download [`Volura-linux-x86_64.tar.gz`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-linux-x86_64.tar.gz) and:

```sh
tar xzf Volura-linux-x86_64.tar.gz
cd Volura
./install.sh      # to ~/.local/opt/volura, with a menu entry in ~/.local/share/applications
volura            # or from the menu: Graphics > Volura
```

`install.sh` also creates `~/.local/bin/volura` and puts the icon in
`~/.local/share/icons/hicolor/256x256/apps`. `./install.sh --remove` removes everything again and
`VOLURA_PREFIX=/your/path ./install.sh` picks another folder. Qt needs `libegl1`, `libgl1`,
`libxkbcommon0`, `libfontconfig1` and `libdbus-1-3`:

```sh
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3
```

OpenGL is also required (Mesa is fine). Under Wayland, Volura starts through XWayland automatically.
For a SpaceMouse on Linux, install and start `spacenavd`.

## Running from source

Python 3.10 or newer is required.

**Windows:** double-click `run.bat`
**Linux:** `./run.sh`

Or by hand:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python main.py                       # Windows: .venv\Scripts\python main.py
```

Besides PySide6, PyOpenGL, numpy and manifold3d, `requirements.txt` also contains pytest and
PyInstaller, so one installation is enough for tests and a build.

The language normally follows the preference or the system; the environment variable `VOLURA_LANG`
(`en` or `nl`) overrides it.

## Building a standalone version

```sh
pip install -r requirements.txt
python packaging/build.py
```

| Platform | Build | Archive |
|---|---|---|
| Windows | `dist/Volura/Volura.exe` | `dist/Volura-windows-x86_64.zip` and `dist/Volura-windows-setup.exe` |
| Linux | `dist/Volura/Volura` + `volura.desktop` and `install.sh` | `dist/Volura-linux-x86_64.tar.gz` |

`build.py` makes the Windows installer with [Inno Setup 6](https://jrsoftware.org/isinfo.php) when it
is installed (`packaging/windows/volura.iss`). PyInstaller cannot cross-compile: build on the
platform you are building for. The GitHub Actions workflow in `.github/workflows/build.yml` builds
and tests both platforms automatically (Python 3.12, tests with `QT_QPA_PLATFORM=offscreen`) and
attaches the installer, the zip and the tar.gz to a release for a `v*` tag.

Releasing a new version: raise `__version__` in `volura/__init__.py`, add a `## x.y.z — date` section
to `CHANGELOG.md` (the release notes are taken from it), commit, and push a tag with the
same number (`git tag v0.2.0 && git push origin v0.2.0`). The update check in Volura looks at the
newest release.
`python packaging/make_icons.py` regenerates the icon: `assets/volura.png` and `assets/volura.ico`
at 256 x 256, also from an existing icon with `--from assets/volura.ico`.

## Tests

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q tests    # Windows: $env:QT_QPA_PLATFORM = "offscreen"
```

The tests need no display. `tests/test_portability.py` keeps the code platform independent: it
imports every module in `volura/` and allows no `os.startfile` and no paths with a drive letter.

## Keyboard shortcuts

The complete list is in the program itself, under **Window > Preferences > Shortcuts**; there you
can also add your own shortcuts, remove them and reset everything. The help window (F1) lists them
too.

| Key | Tool | Key | Tool |
|---|---|---|---|
| Space | Select | P | Push/Pull |
| E | Eraser | F | Offset |
| B | Paint Bucket | M | Move (Ctrl = copy) |
| L | Line | Q | Rotate |
| R | Rectangle | S | Scale |
| C | Circle | T | Tape Measure |
| G | Make Group (Polygon: through the menu) | O / H / Z | Orbit / Pan / Zoom |
| A | 2 Point Arc (Arc, 3 Point Arc and Pie: Draw > Arcs) | Shift+T | 3D Text |
| Shift+Z | Zoom Extents | K | Back Edges (View) |
| Alt+X | X-ray | | |

Middle mouse button = orbit, Shift + middle mouse button = pan, scroll wheel = zoom towards the cursor.

## Multicolor print in 4 steps

1. Make the base object, for example a rectangle of `60;40` and Push/Pull `8`.
2. Draw a shape on the top or put 3D text on it (Shift+T).
3. Pick a color in the filament panel and push the shape 1 mm inwards while holding **Alt**.
4. *File > Export as 3MF (multicolor)* and open the file in your slicer.

Loose geometry that forms a closed volume also counts as a part; make it a group (triple-click, G)
to give it a color of its own.

Only the surface needs a color? Paint faces with the Paint Bucket (B): in the slicer those faces get
the filament of their color, the rest of the part keeps the part's filament. The slicer shows its own
filament colors. Only the colors you use get a filament: 1, 2, 3, ... in the order of the
filament panel, so the file fits a slicer with as many filaments as your model has colors.

## Project structure

```
volura/
  brep.py       SketchUp geometry: edges, faces, splitting, healing, push/pull
  geom.py       planes, 2D cross sections, mesh analysis
  model.py      document: loose geometry, groups, filaments, selection, undo/redo
  ops.py        push/pull, booleans, transformations, print preparation
  fileio.py     .volura, STL, OBJ, 3MF; closed volumes from imported meshes
  mesh_formats.py  reading STL, OBJ, 3MF, PLY, OFF, AMF, glTF/GLB, DAE
  repair.py     repairing meshes that are not closed
  camera.py     orbit camera (Z up)
  inference.py  SketchUp inference
  viewport.py   OpenGL view and input
  tools.py      all tools
  app.py        main window, menus, panels, Model Info, Preferences and welcome screen
  vector2d.py   the 2D model of the worksheet; svg.py and dxf.py read and write it
  sheetview.py  the 2D worksheet; sheettools.py its drawing tools
  cam.py        kerf and tool compensation; nesting.py arranging; boxgen.py boxes with finger joints
  sheet3d.py    worksheet and model together: parts as sheets in 3D
  theme.py      light and dark theme
  spacemouse.py 3Dconnexion 3D mice (3DxWare, Raw Input, spacenavd)
  i18n.py       translations; locale/ holds the Dutch tables
  updater.py    checking for a newer version and installing it
tests/          tests without a GUI (pytest); test_brep.py compares with results from SketchUp 2017
                test_portability.py keeps Windows assumptions out of volura/
docs/website/   the feature cards and the manual of volura.dev, in English and Dutch; the site
                loads them from the latest release, so they go online with each version
packaging/      build.py (PyInstaller), make_icons.py (icon), windows/ (Inno Setup),
                linux/ (volura.desktop, install.sh)
.github/        build.yml: tests and builds for Windows and Linux
```

## Technology

Python, Qt 6 (PySide6), OpenGL and the [Manifold](https://github.com/elalish/manifold) geometry kernel
for robust boolean operations with guaranteed closed results.

Volura is an independent implementation that mimics the workflow of SketchUp 2017. It contains no
SketchUp code and is not affiliated with SketchUp.

## License

MIT, see [LICENSE](LICENSE). © 2026 L. van der Poel.

Volura uses third-party open source software: Qt 6 and PySide6 (LGPL v3), Manifold (Apache 2.0),
NumPy (BSD), PyOpenGL (BSD-style) and Python (PSF). The standalone versions contain their full
license texts in `THIRD_PARTY_LICENSES.txt`; `packaging/licenses.py` assembles that file during the
build. The Qt libraries are separate files in the program folder and may be replaced.
