#!/bin/sh
# Zet een uitgepakte Volura-build in ~/.local/opt/volura en maakt een menu-item.
#   tar xzf Volura-linux-x86_64.tar.gz
#   cd Volura && ./install.sh
# Met VOLURA_PREFIX kies je een andere plek; `./install.sh --remove` haalt het weer weg.
set -e

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${VOLURA_PREFIX:-"$HOME/.local/opt/volura"}
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/256x256/apps"

case "$DEST" in
    ""|"/"|"$HOME"|"$HOME/.local")
        echo "fout: VOLURA_PREFIX wijst naar $DEST, dat is te gevaarlijk om te vervangen" >&2
        exit 1 ;;
esac

if [ "$1" = "--remove" ]; then
    rm -rf "$DEST" "$BIN/volura" "$APPS/volura.desktop" "$ICONS/volura.png"
    echo "Volura is verwijderd uit $DEST"
    exit 0
fi

if [ ! -f "$HERE/Volura" ]; then
    echo "fout: in $HERE ligt geen Volura; pak eerst Volura-linux-x86_64.tar.gz uit" >&2
    exit 1
fi
if [ ! -f "$HERE/volura.desktop" ]; then
    echo "fout: volura.desktop ontbreekt naast dit script" >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST" "$BIN" "$APPS" "$ICONS"
cp -R "$HERE"/. "$DEST"/
chmod 755 "$DEST/Volura"

ln -sf "$DEST/Volura" "$BIN/volura"
if [ -f "$HERE/volura.png" ]; then
    cp "$HERE/volura.png" "$ICONS/volura.png"
fi
sed "s|^Exec=.*|Exec=$DEST/Volura %f|" "$HERE/volura.desktop" > "$APPS/volura.desktop"

update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo "Volura staat in $DEST en is te starten uit het menu (Grafisch > Volura)."
case ":$PATH:" in
    *":$BIN:"*) echo "Of gewoon met: volura" ;;
    *) echo "Voeg $BIN nog toe aan je PATH om 'volura' te kunnen typen." ;;
esac
