"""The SketchUp 2017 user interface that Volura mimics (Dutch labels).

Recorded from SketchUp Make 2017 (menus, submenus, default shortcuts, the
"Getting Started" toolbar and the Default Tray). Each menu lists its items in
order as (label, shortcut). A label ending in ">" is a submenu, listed in SUBMENUS.
"---" is a separator. Volura specific items are marked with a comment.

The menu titles and item labels must match exactly (ignoring "&" accelerators
and a trailing "…"). Shortcuts are the SketchUp defaults.
"""

MENUS = [
    ("Bestand", [
        ("Nieuw", "Ctrl+N"),
        ("Openen", "Ctrl+O"),
        "---",
        ("Opslaan", "Ctrl+S"),
        ("Opslaan als", None),
        ("Kopie opslaan als", None),
        "---",
        ("Importeren", None),
        ("Exporteren >", None),
        "---",
        ("Printer instellen", None),        # SketchUp: Print Setup  -> printbed / printer
        ("Printcontrole", None),            # SketchUp: Print Preview -> print check
        ("Afdrukken", "Ctrl+P"),            # SketchUp: Print -> export 3MF for the slicer
        "---",
        ("Afsluiten", None),
    ]),
    ("Bewerken", [
        ("Ongedaan maken", "Alt+Backspace"),
        ("Opnieuw", "Ctrl+Y"),
        "---",
        ("Knippen", "Shift+Del"),
        ("Kopiëren", "Ctrl+C"),
        ("Plakken", "Ctrl+V"),
        ("Plakken op locatie", None),
        ("Wissen", "Del"),
        ("Hulplijnen wissen", None),
        "---",
        ("Alles selecteren", "Ctrl+A"),
        ("Niets selecteren", "Ctrl+T"),
        "---",
        ("Verbergen", None),
        ("Zichtbaar maken >", None),
        "---",
        ("Vergrendelen", None),
        ("Ontgrendelen >", None),
        "---",
        ("Component maken", "G"),
        ("Groep maken", None),
        ("Groep/component sluiten", None),
        "---",
        ("Vlakken snijden >", None),
    ]),
    ("Beeld", [
        ("Werkbalken", None),
        "---",
        ("Verborgen geometrie", None),
        ("Assen", None),
        ("Hulplijnen", None),
        ("Printbed", None),                 # Volura
        "---",
        ("Randstijl >", None),
        ("Vlakstijl >", None),
        ("Componentbewerking >", None),
    ]),
    ("Camera", [
        ("Vorige", None),
        ("Volgende", None),
        ("Standaardweergaven >", None),
        "---",
        ("Parallelle projectie", None),
        ("Perspectief", None),
        "---",
        ("Draaien", "O"),
        ("Pannen", "H"),
        ("Zoomen", "Z"),
        ("Gezichtsveld", None),
        ("Zoomvenster", "Ctrl+Shift+W"),
        ("Zoomen tot alles", "Ctrl+Shift+E"),
    ]),
    ("Tekenen", [
        ("Lijnen >", None),
        ("Bogen >", None),
        ("Vormen >", None),
        "---",
        ("Basisvormen >", None),            # Volura (SketchUp: Sandbox position)
    ]),
    ("Gereedschap", [
        ("Selecteren", "Space"),
        ("Gum", "E"),
        ("Verfemmer", "B"),
        "---",
        ("Verplaatsen", "M"),
        ("Roteren", "Q"),
        ("Schalen", "S"),
        "---",
        ("Duwen/trekken", "P"),
        ("Volg mij", None),
        ("Verschuiving", "F"),
        "---",
        ("Buitenschil", None),
        ("Volumegereedschap >", None),
        "---",
        ("Meetlint", "T"),
        ("Gradenboog", None),
        "---",
        ("3D-tekst", None),
    ]),
    ("Venster", [
        ("Standaardvenster >", None),
        "---",
        ("Modelinfo", None),
        ("Voorkeuren", None),
    ]),
    ("Help", [
        ("Welkom bij Volura", None),
        ("Sneltoetsen", None),
        "---",
        ("Over Volura", None),
    ]),
]

SUBMENUS = {
    ("Bestand", "Exporteren"): [("3D-model (STL)", None), ("Meerkleuren 3MF", None), ("STL per kleur", None)],
    ("Bewerken", "Zichtbaar maken"): [("Geselecteerd", None), ("Laatste", None), ("Alles", None)],
    ("Bewerken", "Ontgrendelen"): [("Geselecteerd", None), ("Alles", None)],
    ("Bewerken", "Vlakken snijden"): [("Met model", None), ("Met selectie", None)],
    ("Beeld", "Randstijl"): [("Randen", None), ("Achterliggende randen", "K")],
    ("Beeld", "Vlakstijl"): [("Röntgen", None), "---", ("Draadmodel", None), ("Verborgen lijnen", None),
                             ("Gearceerd", None), ("Monochroom", None)],
    ("Beeld", "Componentbewerking"): [("Rest van model verbergen", None)],
    ("Camera", "Standaardweergaven"): [("Boven", None), ("Onder", None), ("Voor", None), ("Achter", None),
                                       ("Links", None), ("Rechts", None), ("Iso", None)],
    ("Tekenen", "Lijnen"): [("Lijn", "L"), ("Uit de vrije hand", None)],
    ("Tekenen", "Bogen"): [("Boog", None), ("Boog door 2 punten", "A"), ("Boog door 3 punten", None),
                           ("Taartpunt", None)],
    ("Tekenen", "Vormen"): [("Rechthoek", "R"), ("Gedraaide rechthoek", None), ("Cirkel", "C"),
                            ("Veelhoek", None)],
    ("Tekenen", "Basisvormen"): [("Kubus / balk", None), ("Cilinder", None), ("Kegel", None), ("Bol", None),
                                 ("Torus", None)],
    ("Gereedschap", "Volumegereedschap"): [("Snijpunt", None), ("Samenvoegen", None), ("Aftrekken", None),
                                           ("Bijsnijden", None), ("Splitsen", None)],
    ("Venster", "Standaardvenster"): [("Entiteitsinfo", None), ("Materialen", None), ("Overzicht", None)],
}

# The "Getting Started" toolbar, left to right ("|" = separator). Volura replaces the
# 3D Warehouse / Extension buttons at the end with its print buttons.
TOOLBAR = ["Selecteren", "Gum", "Lijn", "Boog door 2 punten", "Rechthoek", "Duwen/trekken", "Verschuiving",
           "Verplaatsen", "Roteren", "Schalen", "|", "Meetlint", "3D-tekst", "Verfemmer", "|", "Draaien",
           "Pannen", "Zoomen", "Zoomen tot alles", "|", "Printcontrole", "Afdrukken"]

# Buttons with a drop-down arrow in SketchUp: toolbar label -> the drawing tools it offers
TOOLBAR_DROPDOWNS = {
    "Lijn": ["Lijn", "Uit de vrije hand"],
    "Boog door 2 punten": ["Boog", "Boog door 2 punten", "Boog door 3 punten", "Taartpunt"],
    "Rechthoek": ["Rechthoek", "Gedraaide rechthoek", "Cirkel", "Veelhoek"],
}

# Right click on a face (SketchUp order). Items for groups are added after "Wissen"/"Verbergen".
CONTEXT_FACE = ["Entiteitsinfo", "Wissen", "Verbergen", "---", "Selecteren >", "Oppervlakte >",
                "Vlakken snijden >", "Weergave uitlijnen", "Vlakken omkeren", "Vlakken oriënteren",
                "Selectie zoomen"]

# The Default Tray: one dock on the right with collapsible panels in this order
TRAY_TITLE = "Standaardvenster"
TRAY_PANELS = ["Entiteitsinfo", "Materialen", "Overzicht"]

# The Measurements box in the status bar (bottom right)
MEASUREMENTS_LABEL_DEFAULT = "Afmetingen"
