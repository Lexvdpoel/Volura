# Changelog

What changed in each version of Volura. Downloads are on the
[releases page](https://github.com/Lexvdpoel/Volura/releases).

## 0.2.44 — 2026-11-12

### New
- **Corner bracket** (Draw > Parts): an L-shaped angle bracket with countersunk screw holes in
  both legs, a rounded inner corner and gussets. It prints on its side, so the layer lines run
  round the bend and nothing needs support.
- **Zip-tie mount** (Draw > Parts): a cable-tie anchor with one screw through its base. The tie
  slides through a tunnel in the bridge over it, in one direction or two that cross, and the
  head of the screw sinks into the floor of that tunnel so the tie lies flat over it.
- **Bit holder** (Draw > Parts): a block with rows of hexagonal sockets for 1/4 inch screwdriver
  bits, with a mouth chamfer so a bit finds its way in, a fit you can set, rows that can lean
  towards you by up to 30 degrees and a ledge along the front for a label.

### Website
- The manual describes the corner bracket, the zip-tie mount and the bit holder.

## 0.2.43 — 2026-11-11

### New
- **Export to STEP** (File > Export > CAD Model (STEP)): the parts as solids for Fusion,
  FreeCAD, SolidWorks or Onshape, with real flat faces: a box is six faces and a hole is a hole
  in its face, not a fan of triangles. Every part is its own named body; a hollow part keeps its
  hollow. (GitHub #2)
- **Fusion keys and mouse** (Window > Preferences > General > Keyboard and mouse): next to the
  SketchUp keys there is now a Fusion scheme, with Push/Pull on E and Q, Offset on O, Fillet on F,
  the Tape Measure on I, Zoom Extents also on F6, and the middle mouse button panning (with Shift
  it orbits). Shortcuts you changed yourself stay as they are. (GitHub #1)
- **Hose adapter** (Draw > Parts): a reducer between two hoses or a hose and a fitting, with
  barbs, that prints upright without support.
- **Fan guard** (Draw > Parts): a grille for a square fan of 40 to 140 mm with the screw holes
  at their standard spacing, as rings, spokes or a grid.
- **Desk grommet** (Draw > Parts): a cable pass-through for a hole of 20 to 120 mm in a desk top,
  with crush ribs on the sleeve and a lid with a cable slot; it prints flange down without
  support.

### Improved
- **Pegboard**: the board itself can be printed too (holes, a back rim and screw bosses), the
  pegs are narrow enough for the holes and rise so they lock behind the board, and the tool
  holder is an upright ring that is open at the top.

### Website
- The manual describes the STEP export, the keyboard schemes, the hose adapter, the fan guard,
  the desk grommet and the pegboard board.

## 0.2.42 — 2026-11-11

### New
- **Pegboard hooks, shelves and tool holders** (Draw > Parts): parts that hang on a standard
  pegboard with 1/4 inch holes at a 1 inch pitch, with the width, reach and size you choose.
- **Linked faces work both ways**: a hole or a vent pattern drawn on the worksheet inside the
  outline of a linked face is cut through the wall under that face, and a hole deleted on the
  worksheet closes again in the model. A corner or notch cut off the outline is cut out of the
  wall, and an edge moved outwards builds the wall on, as thick as the wall beside it. It all
  feels like working on one part: one undo step brings everything back.
- **Go to Linked Part / Go to Linked Outline**: a linked outline shows a small chain mark on the
  worksheet; right-click it to select its part (or face) in 3D, or right-click the part or its
  linked face to find the outline on the worksheet again.

### Improved
- **Gridfinity bins** can have a finger scoop at the front and a label shelf at the back.

### Fixed
- The finger-joint box no longer shows short protruding lines at the corners of its side
  panels: the kerf narrowing at the ends of an edge left thin slivers there.
- A face whose outline is a rectangle is no longer read mirrored between the worksheet and the
  model, so holes drawn on the worksheet land on the right side of the wall.

### Website
- The manual describes the pegboard parts, the Gridfinity scoop and label shelf, how linked
  faces carry holes and outline changes back to the model, and the link mark and its menus.

## 0.2.41 — 2026-11-10

### Improved
- **Cable chain with stops**: a small tab behind every pin runs in an arc-shaped window of the
  link before it, so a link turns from straight to the bend you enter (20 to 90 degrees) and no
  further; the other way the chain stays straight and carries itself over a span.

### Fixed
- Gridfinity bins of more than one square no longer have thin bars across the top of the
  opening: the stacking lip now runs around the whole bin.

### Website
- The manual describes the stops of the cable chain and shows the storage bins with their open
  front and labels on the wall rack.

## 0.2.40 — 2026-11-10

### New
- **Gridfinity** (Draw > Parts): bins and baseplates of the open Gridfinity storage grid of
  42 mm. A bin covers the columns and rows you choose, is a number of 7 mm units high and can
  have compartments, a stacking lip and magnet holes; a baseplate carries a grid of the size you
  choose. Bins drop into the plate and stand on each other.

### Improved
- **Storage bins open at the front**: with Open front the front wall stops at half the height and
  the side walls run back at 45 degrees, so you reach into a bin also when another one stands on
  it.
- **The label of a storage bin faces forward**: it slides from above into a dovetail slot in the
  front wall, its letters flush with the front, and cannot fall out forwards. This replaces the
  label ledge on top.

### Website
- The manual describes Gridfinity, the open front and the label slot.

## 0.2.39 — 2026-11-10

### New
- **Storage Bin** (Draw > Parts): an open-top box of the outside size you enter, with rounded
  corners, compartments made by straight dividers and a sloped label ledge along the front wall
  that prints without support. With a label text the bin comes with a thin label that lies flush
  in a pocket in the ledge. A bin can get a wall hook, and it can be stackable: a foot drops into
  a bin of the same size, so bins stand on each other without sliding off.
- **Bin Wall Rack** (Draw > Parts): a plate for the wall with a grid of rails that bins with the
  wall hook hang on, anywhere along a row. You choose the columns, rows and cell size; two
  countersunk holes per row hold it on the wall, and a stop closes the ends of every rail. A
  rack wider than the bed comes in panels that slide together with dovetails.
- **Cable Chain** (Draw > Parts): a drag chain printed in one piece, the links already in each
  other on their pins, for the inside width and height your cables need.
- **Circuit boards in the project box**: pick an Arduino Uno or Mega or a Raspberry Pi 4 / 5,
  Zero or Pico, and the floor gets a standoff with a pilot hole under every mounting hole.

### Improved
- **Battery holder**: two countersunk mounting holes (M2.5, M3, M4 or none) in the middle of the
  first and the last cradle, diagonally across the holder, also for an even number of cells.
- **Vent patterns follow their outline**: a vent pattern is made again for its outline when that
  outline changes shape, also on a plate that is linked to the model.
- **Linked faces follow what is cut into them**: a pocket, a blind hole, a notch or a cut-off
  corner made in 3D now comes to the linked outline on the worksheet.
- **Explode Group** is much faster for a group that touches no loose geometry (a sphere of 4608
  faces in a fifth of a second instead of 49 seconds), and otherwise shows a progress bar with
  Cancel.

### Fixed
- The Rotate tool shows a dotted line from the centre to the point you are about to pick.
- A linked face keeps its own face when the part is turned, instead of the face that came to lie
  where it was.

### Website
- The manual describes the storage bin, its label and stacking, the bin wall rack, the cable
  chain, the board standoffs and the new mounting holes of the battery holder, with a picture of
  bins hanging on the rack.

## 0.2.38 — 2026-11-10

### Fixed
- The Linux download is back: version 0.2.37 was only built for Windows. Nothing else changed.

## 0.2.37 — 2026-11-10

### New
- **3D Text** (Draw > 3D Text…): type a text, on one or more lines, and it becomes a part of
  letters in the font, height and depth you choose, bold or not, aligned left, centred or right,
  standing on the bed.

### Improved
- **The battery holder grips its cells**: the opening over every cradle is a little narrower than
  the cell (5 % by default, adjustable under Grip), so the cells snap in and stay in when you turn
  the holder over.

### Website
- The manual describes 3D Text and the grip of the battery holder.

## 0.2.36 — 2026-11-10

### New
- **Print-in-Place Hinge** (Draw > Parts): a hinge that comes off the printer already
  assembled and turns at once: two leaves flat on the bed, joined by knuckles around one pin
  with the play you enter. The pin and hole ends are chamfered, so nothing needs support.
- **Battery Holder** (Draw > Parts): one to eight AA, AAA, 18650, 21700 or CR123A cells side by
  side in open cradles, with slots in the end walls for spring contacts or wires and holes in
  the floor for the leads.

### Website
- The manual and the feature overview describe the hinge and the battery holder.

## 0.2.35 — 2026-11-10

### Improved
- **The singular when there is one**: messages such as "1 cable clip added" no longer say
  "1 cable clips", in English and in Dutch.
- The program's code is split into smaller parts (the dialogs, the panels, the worksheet,
  3D printing and the controls each in their own module), which makes it easier to
  maintain. Nothing changes in how Volura works.

## 0.2.34 — 2026-11-09

### New
- **Model and worksheet stay linked**: a part that goes to the worksheet with To Laser or
  Mill, or comes from it with Parts to 3D Model, stays linked to its outline. Change the
  shape on one side and the other follows when you switch between 3D and 2D (or with
  Tools > Update Links), while each side keeps its own place: arranging the sheet never
  moves the model, and moving a plate in 3D never moves the sheet. A plate also changes
  back from the sheet, rebuilt with its thickness where it stands; a face of another solid
  only goes from the model to the sheet. Links are saved with the file, every update is
  one undo step, and Tools > Unlink lets go.

### Website
- The manual explains the links.

## 0.2.33 — 2026-11-09

### Website
- **A clearer feature overview**: every card is one or two sentences, in full rows of three,
  and the pictures sit in three wide cards between them.
- **A new manual**: the chapters follow the menus, there is a quick start, every function
  has its own topic with one sentence, where it is in the menus, its options as a short
  list and its pictures, and the manual starts with what is new in this version.

### Improved
- The groundwork for keeping a part in the model and its outline on the worksheet linked
  (not in use yet).

## 0.2.32 — 2026-11-09

### Improved
- **Filament estimate like a slicer**: Print Check and Entity Info now show the weight,
  length and cost of a real print, with a skin of the wall thickness and the infill of the
  rest (1.2 mm and 15 % by default, under 3D Printing > Printer Setup). The weight of a
  solid print stays below it, as the most a part can use.

### Website
- Screenshots in the feature overview and the manual, in English and Dutch: the printable
  parts, the overhang view, the test prints, the project box, the knobs, the worksheet,
  the vent pattern and the new menus. They are made by tools/screenshots.py and loaded
  from GitHub, so they stay up to date.

## 0.2.31 — 2026-11-09

### Fixed
- The toolbar of the 2D worksheet was far too wide and kept folding away: it held more
  buttons than fit in the window, and six of them had no icon and showed their text. The
  worksheet now has three narrow toolbars side by side (drawing, editing, and laser and
  mill), and every button has an icon.

## 0.2.30 — 2026-11-09

### Improved
- **Menus by mode**: the menus show only what works where you are. With the 3D model you
  get the new **3D Printing** menu (printer setup, print check, orienting and placing on the
  bed, cut and join, elephant foot, mouse ears and the test prints); on the 2D worksheet
  the new **Laser and Mill** menu (sheet, nesting, fit test, holding tabs, dogbone corners
  and the job estimate). Edit, View, Camera, Draw, Tools and Export follow the mode too:
  on the worksheet Draw holds the 2D drawing tools, the finger joint box and the spur gear,
  and Tools the 2D commands that were only on the toolbar before (union, subtract,
  overlap, join, flip, align, distribute, duplicate, round corners, living hinge, vent
  pattern). With the 3D model the printable parts are under **Draw > Parts** and the
  screw threads and the screw hole under **Tools**. Shortcuts you set yourself move along
  with their command.
- **Vent Pattern** also fills loose lines that close into one contour (a DXF, or a shape
  drawn line by line), says in its dialog what it does, and tells you what to make smaller
  when no hole fits.
- **Dutch translation** reviewed: wrong terms corrected (for example "Overlap" for
  intersect, "Op raster snappen", "Halve assen", "Drukhoek", "Flankspeling"), "je"
  throughout, one word per concept, and clearer names such as Gatenrooster, Bruggetjes and
  Tijdsschatting.

### Website
- The manual and the feature overview follow the new menus and the new Dutch names.

## 0.2.29 — 2026-11-09

### New
- **Cable Clip** (Tools > Print Tools > Cable Clip…): a printable clip that holds a cable
  or a bundle against a wall or a desk. The cable snaps in through a mouth narrower than
  the cable, and the foot has a hole for a screw; a whole row prints in one go, lying on
  its side so it needs no support.

### Improved
- **Project Box** can have vent slots: up to twenty vertical slots through each long wall,
  below the lip of the lid and clear of the screw posts.

### Website
- The manual describes the cable clip and the vent slots.

## 0.2.28 — 2026-11-09

### Fixed
- The Linux download of 0.2.27 was missing: two tests compared contours to the last digit,
  and those digits differ between Windows and Linux. The tests now compare at a micrometre;
  the program itself is the same as 0.2.27.

## 0.2.27 — 2026-11-09

### New
- **Label Plate** (Tools > Print Tools > Label Plate…): a printable tag for a drawer, a box
  or a keyring, with up to four lines of text raised on the plate or engraved into it,
  rounded corners and, if you like, a hole for a ring or a screw.

### Improved
- Round Corners and Dogbone Corners are much faster on big imported contours: a contour of
  20,000 points without corners now takes a hundredth of a second instead of one and a
  half, with exactly the same result.

### Website
- The manual describes the label plate.

## 0.2.26 — 2026-11-09

### New
- **Knob** (Tools > Print Tools > Knob…): a printable thumb knob, knurled, lobed or smooth,
  with a pocket for a hex nut, a counterbore for the head of a socket head bolt or a D hole
  for a potentiometer or motor shaft, with the clearance your printer needs.
- **Standoff** (Tools > Print Tools > Standoff…): printable spacers for a circuit board or a
  panel, round or hexagonal, with a through hole, a pilot hole for a self-tapping screw or
  a hole for a heat-set insert; a whole set prints in one row.

### Website
- The feature overview describes the knob, the standoff, Round Corners and Dogbone
  Corners; the manual describes the knob and the standoff.

## 0.2.25 — 2026-11-08

### New
- **Overhang Test** (Tools > Print Tools > Overhang Test…): a small calibration print with a
  row of fins that lean further over from left to right, each with its angle in raised
  letters. Print it once and read the largest angle your printer still prints cleanly
  without support.
- **Dogbone Corners** (worksheet, View > Dogbone Corners…): for milling, a round relief of
  the cutter diameter in every inside corner of the material, so a square tab fits into a
  milled square slot. Holes and outer contours are told apart by how they lie in each
  other on the layer.

### Website
- The feature overview describes the Overhang Test; the manual describes both.

## 0.2.24 — 2026-11-08

### New
- **Round Corners** (worksheet, View > Round Corners…): a radius of your choice in the
  sharp corners of the selected paths, open or closed, so a cut panel has no sharp edges
  left. A corner whose segments are too short for the radius is rounded as far as they
  allow; the ends of an open line stay where they are.

### Improved
- **Project Box** can screw its lid on: pick a screw size (M2 to M4) and the box gets a
  post with a blind pilot hole in every inside corner and the lid a clearance hole above
  it; the lip of the lid steps around the posts, so it still slides in.

### Website
- The manual describes the screw posts and Round Corners, the feature overview the screw posts.

## 0.2.23 — 2026-11-08

### New
- **G-code for a GRBL laser** (worksheet, File > Export > G-code for a GRBL Laser): hobby
  diode lasers get the job straight from Volura, with the same layer choice and kerf
  compensation as the SVG and DXF export. Set the highest S value of your controller and
  whether the power follows the speed (M4). Holes and islands are cut before the part
  around them, so nothing drops out early, and the head goes to the nearest path next.
  A layer without a speed gets a clear message instead of a file the laser refuses.
- **Job Estimate** (worksheet): per laser layer the metres cut, the number of pierces, the
  passes and the time, plus the travel between the paths and the total, before you send
  the job to the laser.

### Improved
- The test suite keeps its settings in a temporary file and never touches your own.

### Website
- The feature overview and the manual describe the job estimate and the G-code export.

## 0.2.22 — 2026-11-08

### New
- **Project Box** (Tools > Print Tools > Project Box…): a printable enclosure from the inside
  length, width and height your electronics need, with wall, floor and corner radius, and
  a lid whose lip slides into the box with the clearance you choose. The box and the lid
  land side by side on the bed, the lid already upside down, so neither needs support.
- **Duplicate in a Circle** (worksheet, right-click menu or sheet toolbar): copies of the
  selection spread over a full circle or an arc, around the centre of the selection, the
  origin or a point you type, turning along with the circle or keeping their orientation.

### Website
- The feature overview and the manual describe the project box and the circular duplicate.

## 0.2.21 — 2026-11-08

### New
- **Spur Gear**: an involute gear from the number of teeth, the module and the pressure
  angle, with a bore for the axle and the backlash you choose; the dialog shows the pitch
  and outside diameter as you type. On the worksheet (View > Spur Gear…) it lands on the
  cut layer next to the drawing; in the model (Tools > Print Tools > Spur Gear…) it stands
  on the bed as a printable part of the thickness you enter. Two gears of the same module
  and pressure angle mesh at half the sum of their pitch diameters.

### Website
- The feature overview and the manual describe the spur gear.

## 0.2.20 — 2026-11-08

### New
- **Vent Pattern** (worksheet, View > Vent Pattern…): fill one selected closed contour of
  any shape with the holes of a ventilation grille or a speaker cover — a honeycomb of
  hexagons or a field of circles. Set the hole size, the web of material between two holes
  and the margin along the edge; only whole holes are cut, on the cut layer, and they move
  and nest with the panel they are cut in.

### Website
- The feature overview mentions holding tabs, the Print Check advice and the vent pattern.

## 0.2.19 — 2026-11-08

### New
- **Holding Tabs** (worksheet, View > Holding Tabs…): leave a number of small, evenly
  spread uncut bridges in the selected closed contours on a cutting layer, so a part stays
  in the sheet until the laser or mill is done; snap it out afterwards. Set the number of
  tabs per contour and their width. By default only the outer contours get tabs, since
  the scrap of a hole falls out on its own.

## 0.2.18 — 2026-11-07

### Improved
- Big meshes are faster to work with: importing a mesh of 200,000 triangles turns it into
  faces about a quarter faster, and Entity Info, export and Print Check triangulate such a
  part about eight times faster.
- Print Check only checks two parts for overlap when they can touch at all.

## 0.2.17 — 2026-11-07

### Fixed
- Print Check could freeze Volura for minutes on a smooth part (a sphere or a scanned
  figure), and Orient for Printing took just as long on it. Both are fast again.

## 0.2.16 — 2026-11-07

### Improved
- **Print Check** says what to do with a part outside the print volume: it only lies in the
  wrong place (Arrange on Bed), it fits turned a quarter turn, or it is larger than the
  print volume (Cut and Join splits it into parts that fit).
- **Print Check** reports how much of each part hangs over more steeply than the overhang
  limit and needs support, and suggests Orient for Printing when turning the part brings
  that down by at least a quarter.

## 0.2.15 — 2026-11-07

### Fixed
- Elephant Foot Chamfer, Mouse Ears, Orient for Printing and Arrange on Bed now act on the
  part when a face or edge of it is selected, instead of doing nothing; with nothing
  selected they leave hidden parts alone.
- Orient for Printing, Arrange on Bed and Lay Flat on This Face no longer leave an empty
  undo step when nothing moves.
- Living Hinge recognises a rectangle whose outline ends on its first point again, as
  imported rectangles often do.

## 0.2.14 — 2026-11-07

### New
- **Fit Test** (worksheet, View > Fit Test…): a comb of slots that grow by a step, each
  with its offset engraved under it. Cut it from the sheet you use, push an offcut into
  every slot and use the offset of the slot that holds it best.

### Website
- The feature overview now also shows the print tools of the last versions (elephant foot
  chamfer, mouse ears, lay flat, arrange on bed), the test prints and the living hinge.

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
