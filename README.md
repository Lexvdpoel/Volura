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
- **Intersect Faces** (Tools or right-click): *With Model* adds the cut lines with the rest of the
  model as edges so faces split, *With Selection* only between the selected items
- Primitives: box, cylinder, cone, sphere, torus
- Display styles under **View**: Wireframe, Hidden Line, Shaded and Monochrome (View > Face Style),
  X-ray (Alt+X), Back Edges (K), Hidden Geometry (softened edges) and hiding the rest of the model
  while editing a group
- **Themes**: *Window > Preferences > Display > Theme*: System, Light, Dark or High contrast
- **Accessible**: every command has a menu entry and a shortcut you can change, and every button,
  field and list has a name a screen reader can read, also the ones that only show an icon. The
  print preview can colour the kinds of line with a colour-blind friendly palette, and the edges,
  guides and outlines of the 3D view and the worksheet can be drawn up to three times as thick
  (*Window > Preferences > Display*) ([ACCESSIBILITY.md](ACCESSIBILITY.md))
- **3Dconnexion SpaceMouse** support (*Window > Preferences > 3D Mouse*): through 3DxWare 10 or the
  older 3DxSoftware (siappdll), without a driver through Windows Raw Input, and on Linux through
  spacenavd
- Print bed of your printer (Bambu, Prusa, Creality, Voron or a custom size) and a print check
  that also says how much of a part hangs over more steeply than the overhang limit and needs
  support, and how much less Orient for Printing can make that
- **Arrange on Bed** (Tools or right-click): lay the selected groups, or all groups when nothing is
  selected, side by side on the print bed without overlapping, like a slicer does; parts are turned
  0, 90, 180 or 270 degrees and stand on the bed, 5 mm from the edge and from each other
- **Lay Flat on This Face** (right-click a face of a group): turn that part about its own centre until
  the face you clicked lies on the print bed, like "place on face" of a slicer
- **Screw Hole** (Tools): turn a circle you drew on a part into a printable hole
  for a metric screw (M2 up to M8): straight, with a counterbore for a socket head, a 90°
  countersink for a flat head or a hexagonal pocket that holds a nut; a hole that runs
  sideways can get a 45° teardrop roof so it prints without support; the clearance (0.2 mm by
  default) is added to every diameter, because a printed hole comes out smaller than drawn
- **Clearance Test** (3D Printing): a small test print that shows which clearance your
  printer likes: a plate with a row of holes that grow by a step, each with its clearance in
  raised letters, and a loose pin next to it; print it, try the pin in every hole and use the
  clearance of the hole it fits best in Screw Hole, threads and Cut and Join
- **Overhang Test** (3D Printing): a small test print that shows up to which angle your
  printer prints an overhang without support: a plate with a row of fins that lean further over
  from left to right, each with its angle in raised letters; print it and read the largest angle
  that still came out clean
- **Project Box** (Draw > Parts): an enclosure with a lid that fits into it: enter the
  inside length, width and height that your electronics need, with the wall, the floor, the
  rounding of the corners and the clearance between the lip of the lid and the wall. With a
  screw size the base also gets a post in every corner with a pilot hole and the lid a
  clearance hole above it, so the lid screws on, and with vent slots the two long walls get
  vertical slots through them, evenly spaced and centred, so the air inside the box can
  change. Pick a circuit board (Arduino Uno or Mega, Raspberry Pi 4 / 5, Zero or Pico) and the
  floor gets a standoff with a pilot hole under every mounting hole of it, the board in the
  middle of the box. Base and lid land side by side on the bed, the lid flipped over so both print
  without support
- **Knob** (Draw > Parts): a printable thumb knob that turns on a bolt or a shaft: the
  diameter and the height, the grip of the rim (knurled ribs, rounded lobes or smooth) and the
  mount (a hexagonal pocket for a nut, a counterbore for the head of a socket head bolt, or a
  blind D hole for the shaft of a potentiometer), with the screw size and the clearance that
  printing needs. It stands on the bed to the right of your model as one undo step
- **Standoff** (Draw > Parts): printable spacers that hold a PCB or a panel off the
  surface under it: the height and the outer size, round or hexagonal across the flats, with a
  through hole a screw goes through, a self-tapping pilot hole or a hole for a heat-set insert,
  in the screw size you choose. Up to twelve standoffs print in one row and land on the bed to
  the right of your model as one undo step
- **Label Plate** (Draw > Parts): a printable tag for a drawer, a box or a keyring:
  up to four lines of text, raised on the plate or engraved into it, on a plate with rounded
  corners and, if you like, a hole for a ring or a screw
- **3D Text** (Draw): a part made of letters: your text in any font of your computer, several
  lines, as thick as you like, standing on the bed with the holes of o, a and 8 in it
- **Cable Clip** (Draw > Parts): a printable clip that keeps a cable, a hose or a bundle
  where it belongs: a ring of the cable diameter with a mouth narrower than the cable, so you push
  the cable in and it stays, on a foot you screw to a wall or under a desk. The clip prints lying
  on its side, so the ring needs no support. Up to twelve clips print in one row that you break
  apart afterwards, on the bed to the right of your model as one undo step
- **Print-in-Place Hinge** (Draw > Parts): a printable hinge that comes off the printer
  already assembled: two leaves that lie flat on the bed beside each other, joined by a row of
  barrels around one pin. Half of the barrels carry the pin, the other half have a hole around it
  with the play you enter, so the hinge turns as soon as it is off the bed and there is nothing to
  screw or to glue. The pin ends and the holes end in a chamfer, so nothing needs support, and the
  number of barrels is odd, so the pin cannot walk out
- **Battery Holder** (Draw > Parts): a printable block that keeps one to eight round cells
  side by side, of the cell you choose: AA, AAA, 18650, 21700 or CR123A. Every cell lies in a
  cradle that is open at the top, with the play you enter all round it, so the cell lies in
  from above, and the opening over a cradle is narrower than the cell by the grip you enter, so the
  cell snaps in and cannot fall out. A slot through the end wall at both ends of every cradle takes
  a spring contact or a wire, and two holes with a countersunk head of the screw you choose (M2.5,
  M3, M4 or none) mount the holder: one under the axis of the first cradle near one end and one
  under the axis of the last cradle near the other, so the screws lie diagonally in the holder and
  the head sinks below the cell
- **Storage Bin** (Draw > Parts): a printable open-top box whose inside is divided into
  compartments by straight dividers of the wall thickness, which stop a millimetre under the top
  edge so you can still reach over them. With *Open front* the front is open from half the height
  up and the side walls run back at 45 degrees, so you reach in also when bins stand on each
  other. A dovetail slot in the front wall takes a printed label that slides in from above, facing
  forward; with a *Label text* the bin comes with that label, its letters flush with the front.
  With *Wall hook* the back wall gets a hook along the top and a rib along the bottom, so the bin
  hangs on the wall rack. A *Stackable* bin has a foot that drops into a bin of the same size, so
  bins stand on each other without sliding off
- **Bin Wall Rack** (Draw > Parts): a plate for the wall with a grid of cells whose number and
  size you choose; every row has a rail that leans 45 degrees up, so the hook of a bin drops behind
  it and the bin hangs anywhere along the row and slides sideways. A bin as high as a row or lower
  hangs without touching the bin under it, and countersunk holes (M3, M4 or none) hold the rack on
  the wall, and a stop closes both ends of every rail. It prints lying on its back without support;
  a rack wider than the bed comes in panels of the columns you choose, which slide together with
  dovetails
- **Cable Chain** (Draw > Parts): a drag chain for the cables of a moving axis, printed in one
  piece: the links lie in a row, already in each other, with 0.3 mm of room around every pin and
  between the plates, so the chain comes off the bed and bends. Enter the inside width and height
  for the cables, the wall, the number of links and how far one link may bend on the next: stops
  keep the chain straight the other way, so it carries itself over a span
- **Gridfinity** (Draw > Parts): bins and baseplates of the open Gridfinity storage grid of
  42 mm, so they fit what others print for it. A bin covers the columns and rows you choose, is a
  number of 7 mm units high, has compartments, a stacking lip and magnet holes if you want; a
  baseplate carries a grid of the size you choose. Bins drop into the plate and stand on each
  other. A bin can have a finger scoop, the rounding between the floor and the inside of the front
  wall so you slide small parts out with a finger, and a label shelf under the top of the inside of
  the back wall, at 45 degrees, in which a printed label stands
- **Pegboard** (Draw > Parts): a pegboard of 1/4 inch holes at 1 inch with a rim and screw bosses
  that hold it off the wall, and hooks, shelves with a brace and round tool holders for it. A part
  hangs like a steel pegboard hook: a round upper peg bends up behind the board into a tip, with
  less than half a millimetre of play, so it hangs straight and cannot be pulled off, and a straight
  lower peg keeps it from swinging. Tilt it, push the tip through and swing it down. The parts
  print on their side with support under the pegs
- **Hose Adapter** (Draw > Parts): a reducer between two hoses of the inside diameters you enter,
  with barbs that grip each hose or plain ends for push-on tube, and a collar the hoses stop
  against. It prints upright without support and keeps its wall all the way round the water way
- **Fan Guard** (Draw > Parts): a guard for a square fan of 40 to 140 mm, with the screw holes of
  the fan at their standard spacing and a grille of rings, spokes or a grid; rings and a grid
  leave no opening wider than you enter, so fingers stay out
- **Desk Grommet** (Draw > Parts): a cable pass-through for a hole of 20 to 120 mm in a desk top:
  a flange that lies on the desk, a sleeve with crush ribs that stays in the hole and a lid with a
  slot that drops into the flange and prints next to it. It prints flange down without support
- **Corner Bracket** (Draw > Parts): an L-shaped angle bracket with legs of 15 to 150 mm and up to
  three screw holes in each of them, driven in from the inside of the L so a flat head sinks into
  the inner face, a rounded inner corner and a gusset rib over 40 % of the shorter leg at both
  ends of the width; it prints on its side, so the layer lines run round the bend and nothing
  needs support
- **Zip-Tie Mount** (Draw > Parts): a cable-tie anchor with a base plate of 12 to 40 mm that lies
  against the surface and one screw through its middle; the tie slides through a tunnel in the
  bridge over it, one tunnel or two that cross so the tie can go either way, and the head of the
  screw ends in the floor of that tunnel so the tie lies flat over it. The tunnel closes on a roof
  of 45 degrees, so the mount prints without support
- **Bit Holder** (Draw > Parts): a block that stands 1/4 inch screwdriver bits upright in rows of 1
  to 12; every bit is held by the flats of its hexagonal socket, a little wider than the bit, and
  the rows lean up to 30 degrees backwards so the bits lean towards you. The mouth of a socket is
  chamfered so a bit finds its way in, and a ledge along the front takes a label. It lies on its
  block on the bed and prints without support
- **Memory Card Holder** (Draw > Parts): a block that stands up to 12 SD cards, 12 microSD cards
  and 12 USB sticks upright, every kind in a column of slots of its own size and a second row of
  columns behind the first when you want it. The slots are open at the top and chamfered at their
  mouth, and in front of every column the top dips down so you can pinch the card out there. The
  holder lies on its block on the bed and prints without support
- **Wall Hook** (Draw > Parts): a J that you screw to a wall or the side of a cupboard, for a
  coat, a bag, headphones or a cable: the plate lies against the wall, the arm reaches 10 to 80 mm
  out and the tip stops what hangs on it from sliding off. Both inside bends are rounded, because
  a sharp inside corner is where a hook snaps, and the plate takes one or two countersunk screws.
  The hook prints on its side, so the layer lines run along the J, and its holes are teardrops
- **Phone Stand** (Draw > Parts): a stand a phone or a tablet leans in: the back support holds it
  at 50 to 80 degrees from the bed and a front lip keeps the foot of the device in a slot that is a
  millimetre wider than the device, its case included. The length of the base is computed from the
  lean, so the centre of a leaning tablet stays 10 mm in front of the back edge and the stand does
  not tip over backwards, and a gap of 12 mm in the middle of the width lets a charging cable run
  in behind the stand and up to the phone. The stand prints lying on its side, so every wall is
  pulled straight up from the bed and nothing needs support
- **Spool Adapter** (Draw > Parts): a hub that fills the big hole in the middle of a filament
  spool, so the spool turns true on a rod that is much thinner than that hole or on the 608 bearing
  of a spool holder. The hub is 5 to 40 mm long and 0.4 mm under the hole, so it
  goes in by hand, and it stands on a flange that keeps it from sliding into the spool. Bore it for
  a rod of 5 to 30 mm or ask for a pocket that a 608 bearing presses into with the axle through
  under it; spokes hold the hub together where there is plastic to save. It prints flat on its
  flange, so the pockets are open at the top and nothing needs support
- **Drawer Organizer** (Draw > Parts): a tray that drops into a drawer and divides it into
  `cols` by `rows` equal compartments. You enter the inside of the drawer (40 to 300 mm each way)
  and how high the tray stands (15 to 120 mm); it is made a little smaller on every side so it drops
  in without forcing. The walls and dividers are 1.2 to 4 mm and the floor 0.8 to 3 mm, the inside
  corners of the compartments are rounded so you scoop things out and the tray cleans easily, and a
  half-round finger notch in the top of every inner wall lets you lift an item out from between two
  dividers. It prints upright on its floor, so every wall is pulled straight up and nothing needs
  support
- **Plant Pot** (Draw > Parts): a flower pot with the saucer that catches its water. You enter the
  diameter across its top (40 to 250 mm) and how high it stands (30 to 250 mm), round or as a
  hexagon measured across its flats. The bottom is `taper` percent of the top (50 to 100), so the
  wall leans outwards and every layer is supported by the one under it: the pot prints upright
  without support, and the taper is made milder when it would lean past 45 degrees. The wall is 1.2
  to 4 mm and the floor 2.5 mm, the rim rounds out by 2 mm so the pot is comfortable to hold, and
  the dialog says how wide the bottom is and what the pot holds in litres. 0 to 9 drain holes of
  8 mm let the water out - one in the middle of the floor and the rest in a circle halfway to the
  wall - and channels under the floor lead the water out from under the pot. The
  saucer is 15 mm high and 10 mm all round larger than the bottom of the pot, and prints beside it
  with 5 mm in between, so one print gives you both
- **Cable Winder** (Draw > Parts): an H-shaped plate you wind a loose cable round and hang up. You
  enter the length of the plate (40 to 200 mm), its width (20 to 100 mm) and its thickness (2 to
  6 mm), and how wide (4 to 30 mm) and how deep (5 to 60 mm) a slot is cut from the middle of each
  short end. The two ends are the flanges that keep the wound cable from sliding off and the middle
  part between the two slots is the barrel it winds on - the dialog says how long that barrel is.
  The outside corners round with 3 mm and the closed end of every slot with half the slot width, so
  nothing snags the cable, and an optional keyhole pinches the end of the cable while you wind. An
  optional label engraves itself 0.6 mm deep in the top face of the middle part. It prints lying
  flat, so every wall is pulled straight up and nothing overhangs or needs support
- **Lithophane** (Draw > Parts): a plate whose thickness is the brightness of a photo, so with a
  light behind it the photo appears. Pick any png, jpg, bmp, gif or webp and enter how wide the
  picture is printed (40 to 250 mm) - how high it gets follows the picture itself - the thickness
  where the picture is white (0.4 to 2 mm) and where it is black (1 to 6 mm), the flat frame of
  thick plastic round it (0 to 10 mm) and how big one cell of the grid is (0.1 to 1 mm). Dark
  places come further out and let less light through, the back stays flat and inverting the whole
  thing gives a negative. It prints standing up on its border, which is how a lithophane gets its
  quality: the layers go through the thickness of the picture instead of across it, so the photo
  shows no layer lines, and it needs no support
- **Screw-Top Jar** (Draw > Parts): a round jar and the lid that screws onto it. Enter the inside
  diameter (20 to 150 mm) and depth (15 to 200 mm), the wall (1.6 to 4 mm), the thread pitch (2 to
  5 mm) and length (6 to 20 mm), the play between the threads (0.1 to 0.6 mm) and how many grip
  ribs the lid gets; the dialog says how big the closed jar is and how much it holds. The thread
  is the ISO profile of the thread tools, bevelled at both ends; the jar prints standing on its
  floor and the lid on its top, both without support
- **Elephant Foot Chamfer** (3D Printing): a small chamfer along the bottom edges of the
  selected groups, or of all visible groups when nothing is selected, so the squashed first layer of
  the print no longer sticks out ("elephant's foot"); the size is 0.4 mm by default and is
  remembered, and parts that stand on a point or a tilted edge are left alone
- **Mouse Ears** (3D Printing): thin round discs on the bed under the sharp bottom corners
  of the selected groups (or all visible ones), so those corners do not lift (warp) while printing -
  a classic trick for ABS and ASA and for large flat parts; a corner is sharp when its angle is at
  most 100°, so a box gets an ear on each corner and a cylinder gets none, and the discs are cut off
  after printing
- **Orient for Printing** (3D Printing): turn the selected groups, or all visible ones, so
  that each stands on the face that needs the least support, like the auto-orient of a slicer; the
  flat sides of the convex hull are tried at the overhang limit, the largest face wins a tie, a part
  that already stands right is only dropped on the bed, and locked or open groups are left alone
- **Overhang view** (View > Face Style > Overhangs): marks every face that hangs over more steeply
  than the limit (45° from the vertical by default, under *Window > Preferences > Drawing*) in a
  warning colour, so you see at a glance what needs support; faces that lie on the print bed are
  left alone and the status bar counts the faces that need it
- **Model Info** (Window) with the pages Units (length rounding and the number of decimals of the
  Measurements box), Statistics (groups, loose edges and faces, total volume) and Print Bed
- **Filament** (3D Printing > Printer Setup > Print Bed): the material (PLA, PETG, ABS, ASA, TPU, Nylon
  or a custom density), the diameter (1.75 or 2.85 mm), the price per kg and what a slicer prints
  with: walls of 1.2 mm and 15 % infill by default; **Entity Info** then shows the weight, length and
  cost of a selected part as a print with those walls and that infill, with the weight of a solid
  print under it.
  Select several groups and Entity Info also shows the totals of the whole job: the volume,
  weight, filament length and cost of all the closed parts together
- **Preferences** (Window) with circle segments and a Shortcuts page like in SketchUp: search,
  commands per menu, add or remove a shortcut and reset everything
- **Keyboard and mouse scheme** (*Window > Preferences > General*): **SketchUp** (what Volura does
  now) or **Fusion**, for people who know Autodesk Fusion. There Push/Pull answers to E and Q,
  Offset to O, Fillet to F and the Tape Measure to I, Zoom Extents also to F6, Orbit, Pan, Zoom,
  Rotate and Scale have no letter, and the middle mouse button pans (with Shift it orbits). The
  shortcuts you changed yourself stay as they are and Reset All goes back to the keys of the
  scheme you chose
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
  - **CAD model (STEP)**: one solid per part for Fusion, FreeCAD, SolidWorks or Onshape, with
    real flat faces (a box is six faces, a hole is a hole in its face, not a fan of triangles);
    round surfaces keep the facets they have in Volura
  - Where parts with different colors overlap, the part made later wins
- **Slice** (File > Slice): G-code for your printer without a separate slicer. Choose the maker
  and the printer from the printer profiles of Cura and OrcaSlicer (nearly 1000 printers), then
  the nozzle, the filament brand and material and the quality, and set infill and its pattern,
  support and bed adhesion; Volura works out the profile the way Cura does and slices with CuraEngine, which comes
  with the download. The print preview then shows the G-code in 3D, layer by layer, every line as
  wide as it prints (turn, move and zoom it with the mouse), with a virtual printer that plays each layer as the printer will
  print it, the print time and the filament; save the G-code from there
- **Slice for Resin** (File > Slice for Resin): layer images for more than 150 resin (MSLA)
  printers of Anycubic, Elegoo, Phrozen, Creality, EPAX and others, in the printer's own file
  (.ctb, .goo, .pwmx, ...). Volura puts the parts on automatic supports - heads, pillars that
  bridge and brace each other, anchors on the model where nothing else fits - and a pad, shows
  them in 3D and layer by layer, and slices; UVtools, which comes with the download, writes the
  printer's format. **Hollowing** makes a large part a shell 1.5 to 6 mm thick with two drain
  holes near the bottom, so the resin inside can run out and the print uses far less resin
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
- **Duplicates in a row, a grid or a circle**: copies at a fixed spacing, or spread over an
  arc around the centre of the selection, the origin or a point, turning along or not
- **SVG and DXF** import and export; place a drawing on a face and extrude it
- **Operation layers** in the LightBurn colors: cut, engrave, raster, mill outside/inside, pocket and
  drill, with kerf and tool compensation and dogbones or T-bones
- **Sheet size and automatic arrange** (nesting) over one or more sheets
- **Boxes with finger joints**, on the worksheet or as a 3D preview
- **Living hinges**: fill a selected rectangle with staggered slits, so plywood or MDF
  bends there when it is cut
- **Vent pattern**: fill a selected closed contour with a honeycomb of hexagons or a field
  of circles, for a ventilation grille or a speaker cover, with the web and the edge margin
  you choose
- **Spur gear**: an involute gear for the laser or the printer, from the number of teeth,
  the module and the pressure angle, with a bore and the backlash you choose; in the 3D
  model (Draw > Parts) the same wheel comes out as a printable part of the
  thickness you choose
- **Fit test**: a small comb of slots that shows which offset gives a good press fit for
  the sheet you cut, with that offset engraved under every slot
- **Holding tabs**: leave a few small uncut bridges in the selected closed contours, so a
  part stays in the sheet until the job is done; by default only the outer contours, since
  the scrap of a hole falls out on its own
- **Round corners**: put a radius in the sharp corners of the selected paths, open or
  closed, so a cut panel has no sharp edges left to cut into your hand; a corner whose
  segments are too short for the radius is rounded as far as they allow
- **Dogbone corners**: for a milled part, a round relief of the cutter diameter in every
  inside corner of the material, so a square tab fits into a milled square slot; holes and
  outer contours are told apart by their nesting on the layer
- **Job estimate**: how many metres of contour each layer cuts, how often the laser pierces
  and how long the whole job takes, with the rapid moves between the paths along
- **G-code for a GRBL laser**: File > Export > *G-code for a GRBL Laser* writes the laser
  layers as plain G-code for the machine, with the S maximum and M3 or M4 of the controller
  and the time of the job in the status bar
- **Model and worksheet together**: right-click a face or sheet > *Laser / Mill* puts it on a layer of
  the worksheet; *Parts to 3D Model* places everything that is cut as a sheet with the sheet thickness
  in 3D. Every outline stays linked to its part: a linked face keeps its face when you turn the part
  over, and a pocket, a blind hole, a notch or a corner you cut into that face comes to the worksheet.
  The holes go back too: a hole or a vent pattern drawn in a linked face is cut through that wall,
  and a hole deleted on the worksheet closes again in the model. A corner or notch cut off the
  outline on the worksheet is cut out of the wall, and an edge moved outwards builds the wall on.
  A linked outline carries a small chain at its corner, and right-click it > *Go to Linked Part*
  takes you to the part in 3D; right-click the part or its linked face > *Go to Linked Outline*
  takes you back to the contour on the worksheet

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
QT_QPA_PLATFORM=offscreen python -m pytest -q -n auto tests    # Windows: $env:QT_QPA_PLATFORM = "offscreen"
```

`-n auto` (pytest-xdist) spreads the tests over all processor cores, which takes the run from about
twelve minutes to two or three; leave it out to run them one after the other.

The tests need no display. `tests/test_portability.py` keeps the code platform independent: it
imports every module in `volura/` and allows no `os.startfile` and no paths with a drive letter.

A test run never changes the settings of the user: `tests/conftest.py` sets the environment
variable `VOLURA_SETTINGS_FILE`, and with it Volura reads and writes that ini file instead of the
registry. `python -m pytest -q tests/test_settings_isolation.py` checks that.

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

The table is the **SketchUp** scheme. Under *Window > Preferences > General > Keyboard and mouse*
there is also the **Fusion** scheme, for people who know Autodesk Fusion: Push/Pull on E and Q,
Offset on O, Fillet on F, Tape Measure on I and Zoom Extents also on F6, no letter for Orbit, Pan,
Zoom, Rotate and Scale, and the middle mouse button pans (Shift + middle = orbit).

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
  theme.py      light, dark and high contrast theme
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

## Community

Questions, ideas and prints go to [Discussions](https://github.com/Lexvdpoel/Volura/discussions):

- **Q&A**: how do I ...? Ask about modelling, the ready-made parts, exporting and slicing
- **Ideas**: suggest a feature or a new part, and upvote the ones you want
- **Show and tell**: show what you made with Volura

Something that is clearly broken goes to [Issues](https://github.com/Lexvdpoel/Volura/issues/new/choose)
as a bug report.

Want to help with code, translations or documentation? Read [CONTRIBUTING.md](CONTRIBUTING.md) and
the [Code of Conduct](CODE_OF_CONDUCT.md). [ACCESSIBILITY.md](ACCESSIBILITY.md) says what Volura offers for
keyboard, sight and other needs, what does not work well yet, and how to report a barrier.

## License

MIT, see [LICENSE](LICENSE). © 2026 L. van der Poel.

Volura uses third-party open source software: Qt 6 and PySide6 (LGPL v3), Manifold (Apache 2.0),
NumPy (BSD), PyOpenGL (BSD-style) and Python (PSF). The standalone versions contain their full
license texts in `THIRD_PARTY_LICENSES.txt`; `packaging/licenses.py` assembles that file during the
build. The Qt libraries are separate files in the program folder and may be replaced.

File > Slice uses CuraEngine (AGPL v3) as a separate program in the `curaengine` folder, with the
printer profiles of Cura (LGPL v3) and printers made from the profiles of OrcaSlicer (AGPL v3) by
`tools/fetch_orca_profiles.py`. Volura starts CuraEngine and reads the profiles as data; it does not
contain or link to their code, so Volura itself stays under the MIT license. File > Slice for
Resin uses the command line of UVtools (AGPL v3) the same way, from the `uvtools` folder, with
the resin printer profiles that come with it. The sources of CuraEngine and UVtools are attached
to every release as `CuraEngine-<version>-source.tar.gz` and `UVtools-<version>-source.tar.gz`.
