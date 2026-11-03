#!/bin/sh
# Installs an unpacked Volura build in ~/.local/opt/volura and adds a menu entry.
#   tar xzf Volura-linux-x86_64.tar.gz
#   cd Volura && ./install.sh
# VOLURA_PREFIX picks another location; `./install.sh --remove` removes it again.
set -e

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${VOLURA_PREFIX:-"$HOME/.local/opt/volura"}
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/256x256/apps"

case "$DEST" in
    ""|"/"|"$HOME"|"$HOME/.local")
        echo "error: VOLURA_PREFIX points to $DEST, which is too dangerous to replace" >&2
        exit 1 ;;
esac

if [ "$1" = "--remove" ]; then
    rm -rf "$DEST" "$BIN/volura" "$APPS/volura.desktop" "$ICONS/volura.png"
    echo "Volura has been removed from $DEST"
    exit 0
fi

if [ ! -f "$HERE/Volura" ]; then
    echo "error: there is no Volura in $HERE; unpack Volura-linux-x86_64.tar.gz first" >&2
    exit 1
fi
if [ ! -f "$HERE/volura.desktop" ]; then
    echo "error: volura.desktop is missing next to this script" >&2
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

echo "Volura is installed in $DEST and can be started from the menu (Graphics > Volura)."
case ":$PATH:" in
    *":$BIN:"*) echo "Or simply with: volura" ;;
    *) echo "Add $BIN to your PATH to be able to type 'volura'." ;;
esac
