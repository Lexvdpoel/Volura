# Accessibility

Volura should be usable by as many makers as possible. This page says honestly what works today,
what does not work well yet, and how to tell us about a barrier you run into.

## What Volura offers

**Keyboard**
- Every tool has a single-key shortcut (Space, L, R, P, M and so on; see the table in the
  [README](README.md#keyboard-shortcuts)), and the menus work with Alt and the arrow keys.
- All shortcuts can be changed, added or removed under *Window > Preferences > Shortcuts*. There
  are two ready-made schemes, SketchUp and Fusion, under *Window > Preferences > General >
  Keyboard and mouse*.
- The **Measurements box** at the bottom right takes typed values while you draw (`25`, `2,5cm`,
  `40;20`, `5x`), so sizes never depend on precise mouse movement.
- F1 opens the help with the full list of shortcuts.

**Mouse and other input**
- Orbit, Pan and Zoom are also tools (O, H, Z), so the view can be turned with a plain left-button
  drag instead of the middle mouse button.
- 3Dconnexion SpaceMouse devices are supported (*Window > Preferences > 3D Mouse*).
- Ready-made parts (Draw > Parts) are made by filling in numbers in a dialog, without drawing.

**Seeing**
- A light and a dark theme, or the theme of the system (*Window > Preferences*).
- A colour-blind friendly palette for the print preview, with colours that people with red-green
  colour blindness can tell apart (*Window > Preferences > Display*).
- Volura follows the display scaling of the operating system, so text and controls grow with it.
- Every part is checked for a closed volume before export, and problems are reported in words,
  not only with colour.

**Language**
- English and Dutch; the language follows the system, or can be set in the preferences.

## Known limitations

We prefer to be clear about these:

- **Screen readers**: the menus, dialogs and the Measurements box are standard Qt controls and
  are read out, but the 3D view itself cannot be explored with a screen reader. Modelling in 3D
  is visual by nature; the typed values and the Parts dialogs are the most usable route today.
  Every button, field and list now has a descriptive accessible name, also the ones that only
  show an icon; `tests/test_accessibility.py` walks the window and the dialogs and keeps it
  that way.
- **Colour**: the print preview has a colour-blind friendly palette (*Window > Preferences >
  Display*) and its legend names every colour, so the kinds of line are not colour alone any
  more. Other views still tell things apart by colour only, such as the red, green and blue axes;
  no view offers patterns.
- **Contrast and size**: there is no high-contrast theme of Volura's own, and the line width in
  the 3D view cannot be set yet.
- **Motion**: the print preview plays an animation only when you press Play; nothing moves by
  itself.

## Tell us about a barrier

If something in Volura is hard or impossible for you to use, please tell us. That is the most
useful way to improve it.

- Open an [issue](https://github.com/Lexvdpoel/Volura/issues/new/choose) and mention
  "accessibility" in the title; it gets the `accessibility` label and priority.
- Or start a discussion in [Ideas](https://github.com/Lexvdpoel/Volura/discussions/categories/ideas)
  if you have a suggestion.

Describe what you tried to do, what got in the way, and which assistive technology you use (for
example a screen reader, magnifier, switch access or voice control) with your operating system.

## For contributors

When you change the interface, please keep it accessible:

- Give new commands a menu entry, so they can be reached and given a shortcut.
- Give buttons and fields a text or tooltip that says what they do, and use
  `setAccessibleName` where a control has only an icon.
- Never use colour as the only way to tell something apart; add text, a shape or a legend.
- Let dialogs work with the keyboard alone: a sensible tab order, Enter for OK, Escape to cancel.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rest of the guidelines.
