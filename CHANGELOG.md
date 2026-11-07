# Changelog

What changed in each version of Volura. Downloads are on the
[releases page](https://github.com/Lexvdpoel/Volura/releases).

## 0.2.13 — 2026-11-07

### New
- **Living Hinge** (worksheet, View > Living Hinge…): select a rectangle and it is filled
  with staggered slits, so plywood or MDF cut on a laser bends there. Slit length, bridge,
  column spacing, edge margin and direction are adjustable.

### Fixed
- A rare crash when a background job (export, import, print tools) ended at the same
  moment as its worker thread.

## 0.2.12 — 2026-11-07

### New
- **Orient for Printing** (Tools > Print Tools): turns every selected part so that it stands
  on the face that needs the least support, like the auto-orient of a slicer; parts that
  already stand right stay where they are.
- **Clearance Test** (Tools > Print Tools): a small test print with a row of holes that grow
  by a step, each labelled with its clearance, and a loose pin. Print it, try the pin in
  every hole and use the clearance of the best fit in Screw Hole, threads and Cut and Join.

## 0.2.11 — 2026-11-07

### New
- **Mouse Ears** (Tools > Print Tools): thin round discs on the bed under the sharp bottom
  corners of the selected parts, so those corners do not lift (warp) while printing; cut them
  off after printing.
- **Screw Hole: nut pocket**: a fourth kind that makes a hexagonal pocket in which a hex nut
  (ISO 4032) sits, with the straight hole for the screw through the rest.
- **Screw Hole: teardrop for sideways holes**: a hole that runs sideways gets a 45° roof, so
  it prints without support.

## 0.2.10 — 2026-11-07

### New
- **Screw Hole** (Tools > Print Tools): draw a circle where the screw goes, select it and
  get a printable hole for a metric screw from M2 up to M8: straight, with a counterbore for
  a socket head or a 90° countersink for a flat head. The clearance (0.2 mm by default) is
  added to every diameter, because printed holes come out smaller than drawn.
- **Elephant Foot Chamfer** (Tools > Print Tools): a small chamfer along the bottom edges of
  the selected parts, so the squashed first layer of the print no longer sticks out.

### Improved
- Volura starts faster: the update check and some file-format code are only loaded when
  they are needed.

## 0.2.9 — 2026-11-07

### New
- **Export as OBJ with colors** (File > Export): every printable part becomes its own object
  in its own color, with a .mtl file next to it, for other 3D programs and renderers.
- **Lay Flat on This Face**: right-click a face of a part and it turns, the shortest way
  round, until that face lies on the print bed, like "place on face" in a slicer.
- **Totals in Entity Info**: with several groups selected, Entity Info shows the total volume,
  weight, filament length and cost, and how many parts are left out because they are not
  closed.

## 0.2.8 — 2026-11-07

### New
- **Filament use** in Entity Info: a closed part shows the weight, length and cost of the
  filament it takes as a solid print. Choose the material (PLA, PETG, ABS, ASA, TPU, Nylon or
  your own density), the filament diameter and the price per kg under File > Printer Setup.
- **Overhang view** (View > Face Style > Overhangs): faces that hang over too far to print
  without support are coloured, and the status bar counts them. The limit is 45° from the
  vertical, adjustable under Window > Preferences > Drawing. Faces on the print bed are left
  alone.
- **Arrange on Bed** (Tools, or right-click): lays the selected groups, or all groups, side by
  side on the print bed without overlapping, turned where that helps. One undo puts
  everything back.
- **Circular arrays**: after a rotate copy, type `5x` for 5 more copies around the same axis,
  or `/5` to divide the angle into 5 steps, just like Move makes a row.

### Improved
- The website now takes its feature list and manual from each release, so they always match
  the latest version.

## 0.2.7 — 2026-11-06

### New
- **More 3D formats**: besides STL, Volura now imports OBJ, 3MF, PLY, OFF, AMF, glTF/GLB
  and COLLADA (DAE), with a progress bar. Every closed part becomes a group in its own
  color. Slicer projects from PrusaSlicer, Bambu Studio and OrcaSlicer keep their parts
  and filament colors, including color modifiers.
- **Repair on import**: a model that is not closed is repaired where possible. Faces on
  both sides or the wrong way round, holes, parts that touch along an edge, small knots
  and the inner walls of SketchUp models are fixed, and the status bar says what was done.
  A model that cannot be repaired reliably is still refused, with the reason.
- A model that comes in tiny (under 5 mm) was probably made in other units. Volura asks
  whether to scale it up from inches, centimetres or metres, as an undo step of its own.

### Improved
- Big imports and Simplify Model no longer freeze the window; the work runs in the
  background. A 69,000-triangle scan used to freeze it for 25 s.
- Placing a large SVG or DXF drawing runs in the background with a progress bar and can
  be cancelled. Curves in a scaled SVG are flattened to the real size, so a detailed
  drawing places in seconds instead of minutes.
- DXF import of a drawing full of splines: from minutes to seconds.

### Fixed
- A DXF file with an empty value (for instance a custom property without a value in its
  header) imported nothing at all.

## 0.2.6 — 2026-11-06

### New
- **Simplify Model** (Tools menu): fewer triangles on the selected groups, with a slider
  from "Lossless" (0.005 mm) to "Half the nozzle" (0.2 mm). Every step explains what to
  expect in the print; the default (0.02 mm) stays far below a 0.1 mm layer. A live
  estimate shows how many triangles are left, and one undo restores everything.
- After importing a model with more than 20,000 triangles Volura offers to simplify it.
  "Don't ask again" turns this off; Preferences turns it back on.

### Improved
- Drawing the 3D view is much faster: the model is kept on the graphics card instead of
  being sent every frame. On a heavy model (234,000 triangles) the first frame went from
  26 s to 5 s, and orbiting is more than four times smoother.
- Export and opening files use several processor cores: STL export is about three times
  faster, 3MF export about twice.

### Fixed
- An STL that lists triangles twice (some exporters write an object twice) was imported
  as two shells on top of each other, with double the volume. The duplicates are now
  dropped, and the status bar says how many.

## 0.2.5 — 2026-11-05

### New
- A splash screen with a progress bar while Volura starts.

### Improved
- Long operations (import, export, booleans) run in the background, so the window keeps
  responding and they can be cancelled.
- Building geometry, snapping and hovering are two to three times faster on large models.
### Fixed
- Preferences are written to disk right away, and pressing OK in Preferences always puts
  the windows in the chosen theme, also when the choice itself did not change.

## 0.2.4 — 2026-11-03

### Fixed
- Multicolor 3MF: only the colors that are used get a filament, numbered 1, 2, 3, … So a
  model in three colors needs three filaments in the slicer, instead of falling back to
  filament 1. The export message says which color is on which filament.

## 0.2.3 — 2026-11-03

### New
- Multicolor 3MF: faces painted with the Paint Bucket are exported as slicer color
  painting (OrcaSlicer, Bambu Studio and PrusaSlicer). Before, every group became one
  color.

## 0.2.2 — 2026-11-03

### Fixed
- Fillet and Chamfer: the edge under the cursor broke the drawing of the 3D view while
  the tool was active.

## 0.2.1 — 2026-11-03

### Improved
- The red, green and blue axes snap like edges, as in SketchUp: from a little further
  away, before a face under the cursor, and a line parallel to one axis ends exactly on
  another.

### Fixed
- The update check failed with "unable to get issuer certificate" on computers that never
  needed GitHub's root certificate. Windows now checks the certificate itself.

## 0.2.0 — 2026-11-03

### New
- Light and dark theme (Preferences > Display > Theme: System, Light, Dark).
- 3Dconnexion SpaceMouse support: through 3DxWare, without a driver on Windows (Raw
  Input), and through spacenavd on Linux.
- The interface is in English; Dutch stays available under Preferences > General >
  Language. Older files, shortcuts and Dutch layer names keep working.

## 0.1.0 — 2026-11-03

- First public version: SketchUp-style modelling for 3D printing, laser cutting and CNC,
  with push/pull, groups as printable parts, a print check, STL and 3MF export, and
  SVG/DXF import and export.
