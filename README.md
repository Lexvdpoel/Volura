# Volura

**Volura** is een open source 3D-modelleerprogramma voor 3D-printen, lasersnijden en frezen, voor
Windows en Linux. Het werkt zoals SketchUp 2017: je tekent een plat vlak en maakt er met
**Duwen/trekken** een volume van.

Website en handleiding: [volura.dev](https://volura.dev) · Downloads: [Releases](https://github.com/Lexvdpoel/Volura/releases/latest)

Net als in SketchUp bestaat een model uit **randen en vlakken**: randen splitsen elkaar waar ze
kruisen, een gesloten vorm wordt een vlak, een lijn over een vlak splitst het, en met de gum wis je
(delen van) randen waarna vlakken weer samenvoegen. Duwen/trekken werkt op elk vlak of deel van een
vlak en verlengt of snijdt aangrenzende vlakken precies zoals SketchUp.

Voor 3D-printen controleert Volura bij het exporteren of elk onderdeel een gesloten volume is
(geen gaten, open of dubbele randen) en schrijft het via de Manifold-kern foutloos als STL of 3MF.

## Functies

- SketchUp-werkwijze: Lijn, Rechthoek, Cirkel, Veelhoek, Bogen (Boog door 2 punten, Boog, Boog door 3 punten
  en Taartpunt), 3D-tekst, Duwen/trekken, Verschuiving (offset),
  Verplaatsen, Roteren, Schalen, Meetlint, Verfemmer, Gum, Orbit/Pan/Zoom
- Inferentie zoals in SketchUp: eindpunt, middelpunt, oorsprong, op rand, op vlak, rode/groene/blauwe as,
  en as + rand gecombineerd. Pijltjestoetsen of Shift vergrendelen een richting
- Lengte-snapping op 1 mm (instelbaar), hoeken snappen op 15°
- Maatinvoer rechtsonder (gewoon typen): `25`, `2,5cm`, `40;20`, `64s`, `10;0;5`, `5x`, `/5`.
  De richting volgt je muis, zoals in SketchUp
- Selecteren: klik, dubbelklik (vlak + randen), driemaal klikken (alles wat verbonden is), selectiekader
- **Groepen** (G): elke groep is een apart printbaar onderdeel met een eigen filamentkleur. Dubbelklik om
  een groep te bewerken
- **Meerkleuren**: houd **Alt** vast bij duwen/trekken. Naar binnen maakt een *inleg* (uitsparing + vulling als
  aparte groep), naar buiten een apart onderdeel. Ctrl werkt zoals in SketchUp (nieuw beginvlak)
- Volume-gereedschap op groepen: samenvoegen, aftrekken, snijpunt, bijsnijden, splitsen, buitenschil,
  spiegelen, op bed plaatsen
- **Vlakken snijden** (Bewerken of rechtsklik): *Met model* legt de snijlijnen met de rest van het model
  als randen neer waardoor vlakken splitsen, *Met selectie* alleen tussen de geselecteerde items
- Primitieven: balk, cilinder, kegel, bol, torus
- Weergavestijlen onder **Beeld**: Draadmodel, Verborgen lijnen, Gearceerd en Monochroom (Beeld > Vlakstijl),
  Röntgen (Alt+X), achterliggende randen (K), verborgen geometrie (verzachte randen) en het rest van het model
  verbergen tijdens het bewerken van een groep
- Printbed van je printer (Bambu, Prusa, Creality, Voron of eigen maat) en een printcontrole
- **Modelinfo** (Venster) met de pagina's Eenheden (lengte-afronding en het aantal decimalen van de
  maatinvoer), Statistieken (groepen, losse randen en vlakken, totaal volume) en Printbed
- **Voorkeuren** (Venster) met cirkelsegmenten en een pagina Sneltoetsen zoals in SketchUp: zoeken,
  opdrachten per menu, een sneltoets toevoegen of afhalen en alles terugzetten
- Een welkomstscherm bij het opstarten (Help > Welkom bij Volura), uit te vinken
- Export:
  - **STL**: alle (of de geselecteerde) onderdelen samengevoegd tot één waterdicht object
  - **3MF meerkleuren**: één object met één onderdeel per filamentkleur, al gekoppeld aan de juiste
    extruder (PrusaSlicer, OrcaSlicer, Bambu Studio)
  - **STL per kleur**
  - Waar onderdelen met verschillende kleuren overlappen, wint het later gemaakte onderdeel
- Eigen bestandsformaat `.volura` (bestanden van versie 0.1 worden omgezet), STL-import, onbeperkt ongedaan maken

### Laser en frees

- **2D-werkblad** naast het 3D-model (Ctrl+Shift+2 of de schuifknop 3D | 2D rechtsonder), opgeslagen in
  hetzelfde bestand: lijnen, rechthoeken, cirkels, bogen, veelhoeken en tekst; verschuiven, samenvoegen,
  aftrekken, uitlijnen en verdelen
- **SVG en DXF** importeren en exporteren; een tekening op een vlak plaatsen en extruderen
- **Bewerkingslagen** in de kleuren van LightBurn: snijden, graveren, rasteren, frezen buiten/binnen,
  uitkameren en boren, met kerf- en freescorrectie en dogbones of T-bones
- **Plaatformaat en automatisch schikken** (nesten) over een of meer platen
- **Dozen met vingerlassen**, op het werkblad of als 3D-voorbeeld
- **Model en werkblad samen**: rechtsklik op een vlak of plaat > *Laseren / frezen* zet het op een laag van
  het werkblad; *Onderdelen naar 3D-model* zet alles wat gesneden wordt als plaat met de plaatdikte in 3D

## Installeren

### Windows

Download [`Volura-windows-setup.exe`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-windows-setup.exe)
en start het. De installatie heeft geen beheerdersrechten nodig (ze installeert voor jou alleen in
`%LOCALAPPDATA%\Programs\Volura`), maakt een snelkoppeling in het Startmenu en opent `.volura`-bestanden
met Volura. Verwijderen gaat via *Instellingen > Apps*.

Liever zonder installatie: pak [`Volura-windows-x86_64.zip`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-windows-x86_64.zip)
uit en start `Volura\Volura.exe`. Je instellingen en je onopgeslagen model staan in de eigen gegevensmap
van de app (`%APPDATA%\Volura`).

### Updates

Volura kijkt bij het opstarten één keer per dag of er op GitHub een nieuwere versie is (uit te zetten in
*Venster > Voorkeuren*, of zelf via *Help > Controleren op updates…*). Een met de installatie
geïnstalleerde Volura downloadt de nieuwe versie, controleert die, installeert hem en start opnieuw.
De zip-, Linux- en broncodeversie openen de downloadpagina.

### Linux

Download [`Volura-linux-x86_64.tar.gz`](https://github.com/Lexvdpoel/Volura/releases/latest/download/Volura-linux-x86_64.tar.gz) en:

```sh
tar xzf Volura-linux-x86_64.tar.gz
cd Volura
./install.sh      # naar ~/.local/opt/volura, met menu-item in ~/.local/share/applications
volura            # of vanuit het menu: Grafisch > Volura
```

`install.sh` maakt bovendien `~/.local/bin/volura` en zet het icoon in
`~/.local/share/icons/hicolor/256x256/apps`. `./install.sh --remove` haalt alles weer weg en
`VOLURA_PREFIX=/eigen/pad ./install.sh` kiest een andere map. Qt heeft `libegl1`, `libgl1`,
`libxkbcommon0`, `libfontconfig1` en `libdbus-1-3` nodig:

```sh
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3
```

Verder is OpenGL nodig (Mesa is voldoende). Onder Wayland start Volura automatisch via XWayland.

## Starten vanaf de broncode

Python 3.10 of nieuwer is nodig.

**Windows:** dubbelklik `run.bat`
**Linux:** `./run.sh`

Of handmatig:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python main.py                       # Windows: .venv\Scripts\python main.py
```

`requirements.txt` bevat naast PySide6, PyOpenGL, numpy en manifold3d ook pytest en PyInstaller,
één installatie is genoeg voor tests én een build.

## Zelfstandige versie bouwen

```sh
pip install -r requirements.txt
python packaging/build.py
```

| Platform | Build | Archief |
|---|---|---|
| Windows | `dist/Volura/Volura.exe` | `dist/Volura-windows-x86_64.zip` en `dist/Volura-windows-setup.exe` |
| Linux | `dist/Volura/Volura` + `volura.desktop` en `install.sh` | `dist/Volura-linux-x86_64.tar.gz` |

De Windows-installatie maakt `build.py` met [Inno Setup 6](https://jrsoftware.org/isinfo.php) als dat
geïnstalleerd is (`packaging/windows/volura.iss`). PyInstaller kan niet cross-compileren: bouw op het
platform waarvoor je bouwt. De GitHub Actions-workflow in `.github/workflows/build.yml` bouwt en test
beide platformen automatisch (Python 3.12, tests met `QT_QPA_PLATFORM=offscreen`) en hangt de
installatie, de zip en de tar.gz bij een tag `v*` aan een release.

Een nieuwe versie uitbrengen: zet `__version__` in `volura/__init__.py` hoger, commit, en push een tag
met hetzelfde nummer (`git tag v0.2.0 && git push origin v0.2.0`). De update-controle in Volura kijkt
naar de nieuwste release.
`python packaging/make_icons.py` maakt het icoon opnieuw: `assets/volura.png` en `assets/volura.ico`
van 256 x 256, ook uit een bestaand icoon met `--from assets/volura.ico`.

## Tests

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q tests    # Windows: $env:QT_QPA_PLATFORM = "offscreen"
```

De tests hebben geen beeldscherm nodig. `tests/test_portability.py` houdt de code platformonafhankelijk:
het importeert elke module in `volura/` en wil geen `os.startfile` en geen paden met een schijfletter zien.

## Sneltoetsen

Het volledige overzicht staat in het programma zelf, onder **Venster > Voorkeuren > Sneltoetsen**; daar
maak je ook je eigen toetsen, haal je ze af en zet je alles terug. Het helpvenster (F1) noemt ze ook.

| Toets | Gereedschap | Toets | Gereedschap |
|---|---|---|---|
| Spatie | Selecteren | P | Duwen/trekken |
| E | Gum | F | Verschuiving |
| B | Verfemmer | M | Verplaatsen (Ctrl = kopie) |
| L | Lijn | Q | Roteren |
| R | Rechthoek | S | Schalen |
| C | Cirkel | T | Meetlint |
| G | Groep maken (Veelhoek: via menu) | O / H / Z | Orbit / Pan / Zoom |
| A | Boog door 2 punten (Boog, 3 punten en Taartpunt: Tekenen > Bogen) | Shift+T | 3D-tekst |
| Shift+Z | Zoomen tot alles | K | Achterliggende randen (Beeld) |
| Alt+X | Röntgen | | |

Middelste muisknop = draaien, Shift + middelste muisknop = pannen, scrollwiel = zoomen naar de cursor.

## Meerkleurenprint in 4 stappen

1. Maak het basisobject, bijvoorbeeld een rechthoek van `60;40` en Duwen/trekken `8`.
2. Teken op de bovenkant een vorm of zet er 3D-tekst op (Shift+T).
3. Kies in het filamentpaneel een kleur en duw de vorm met **Alt** ingedrukt 1 mm naar binnen.
4. *Bestand > Exporteren als 3MF (meerkleuren)* en open het bestand in je slicer.

Losse geometrie die een gesloten volume vormt telt ook als onderdeel; maak er een groep van (driemaal
klikken, G) om het een eigen kleur te geven.

## Projectstructuur

```
volura/
  brep.py       SketchUp-geometrie: randen, vlakken, splitsen, helen, duwen/trekken
  geom.py       vlakken, 2D-doorsneden, mesh-analyse
  model.py      document: losse geometrie, groepen, filamenten, selectie, undo/redo
  ops.py        push/pull, booleans, transformaties, printvoorbereiding
  fileio.py     .volura, STL, 3MF
  camera.py     orbit-camera (Z omhoog)
  inference.py  SketchUp-inferentie
  viewport.py   OpenGL-weergave en invoer
  tools.py      alle gereedschappen
  app.py        hoofdvenster, menu's, panelen, Modelinfo, Voorkeuren en welkomstscherm
  vector2d.py   het 2D-model van het werkblad; svg.py en dxf.py lezen en schrijven het
  sheetview.py  het 2D-werkblad; sheettools.py de tekengereedschappen ervan
  cam.py        kerf- en freescorrectie; nesting.py schikken; boxgen.py dozen met vingerlassen
  sheet3d.py    werkblad en model samen: onderdelen als platen in 3D
  updater.py    controleren op een nieuwere versie en die installeren
tests/          tests zonder GUI (pytest); test_brep.py vergelijkt met uitkomsten van SketchUp 2017
                test_portability.py houdt Windows-aannames buiten volura/
packaging/      build.py (PyInstaller), make_icons.py (icoon), windows/ (Inno Setup),
                linux/ (volura.desktop, install.sh)
.github/        build.yml: tests en builds voor Windows en Linux
```

## Techniek

Python, Qt 6 (PySide6), OpenGL en de [Manifold](https://github.com/elalish/manifold)-geometriekern
voor robuuste booleaanse bewerkingen met gegarandeerd gesloten resultaten.

Volura is een eigen implementatie die de werkwijze van SketchUp 2017 nabootst. Het bevat geen code
van SketchUp en is daar niet aan gelieerd.

## Licentie

MIT, zie [LICENSE](LICENSE). © 2026 L. van der Poel.

Volura gebruikt open source software van derden: Qt 6 en PySide6 (LGPL v3), Manifold (Apache 2.0),
NumPy (BSD), PyOpenGL (BSD-stijl) en Python (PSF). De zelfstandige versies bevatten hun volledige
licentieteksten in `THIRD_PARTY_LICENSES.txt`; `packaging/licenses.py` stelt dat bestand samen bij
het bouwen. De Qt-bibliotheken staan als losse bestanden in de programmamap en mogen vervangen worden.
