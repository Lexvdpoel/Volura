"""Built-in toolbar icons as small SVG drawings (no external image files)."""
from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QCursor, QGuiApplication, QIcon, QIconEngine, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from . import theme

def S(color="#2b2b2b", w=1.6):
    return f'stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"'


F = 'fill="#cfe0f5"'
N = 'fill="none"'

SVG = {
    "select": f'<path d="M5 3 L5 19 L9 15 L12 21 L14.5 20 L11.5 14 L17 14 Z" fill="#fff" {S()}/>',
    # eraser: pink tip with a blue sleeve, lying at an angle, with a smudge below it
    "eraser": f'<path d="M11.5 22.6 L18 22.6" {N} stroke="#9a9a9a" stroke-width="1.4" stroke-linecap="round"/>'
              f'<g transform="rotate(45 12 11.5)">'
              f'<rect x="6.5" y="1.5" width="11" height="20" rx="2.2" fill="#2f6fd6" {S()}/>'
              f'<path d="M6.5 13.5 L17.5 13.5 L17.5 19.3 A2.2 2.2 0 0 1 15.3 21.5 L8.7 21.5 A2.2 2.2 0 0 1 6.5 19.3 Z" fill="#f5a3b8" {S()}/>'
              f'<path d="M9 5 L15 5 M9 8.5 L15 8.5" {N} stroke="#fff" stroke-width="1.4" stroke-linecap="round"/></g>',
    # paint bucket pouring to the left, with a drop (hotspot) at the bottom left
    "paint": f'<g transform="matrix(-1 0 0 1 24 0)"><g transform="rotate(55 10 12)">'
             f'<path d="M3.5 8 L5 18.5 Q10 21 15 18.5 L16.5 8 Z" fill="#b8c9de" {S()}/>'
             f'<ellipse cx="10" cy="8" rx="6.5" ry="2.2" fill="#fff" {S()}/>'
             f'<path d="M4.6 8.6 Q10 11.2 15.4 8.6 Q10 7.6 4.6 8.6 Z" fill="#2f6fd6"/>'
             f'<path d="M5 18.5 C6 21.5 9 22.5 10 20.2" {N} {S(w=1.3)}/></g>'
             f'<path d="M16.8 14.8 C18.6 15.3 19.6 16.2 19.8 17.4" {N} stroke="#2f6fd6" stroke-width="2" stroke-linecap="round"/>'
             f'<path d="M19.8 16.6 C21.3 18.6 21.8 19.6 21.8 20.5 A2 2 0 0 1 17.8 20.5 C17.8 19.6 18.3 18.6 19.8 16.6 Z" fill="#2f6fd6" {S(w=1.1)}/></g>',
    # pencil: graphite tip at the bottom left (hotspot), wooden point, yellow shaft, ferrule and eraser
    "line": f'<path d="M3 21 L5.4 14.9 L9.1 18.6 Z" fill="#f3d9b1" {S(w=1.2)}/>'
            f'<path d="M3 21 L3.9 18.8 L5.2 20.1 Z" fill="#2b2b2b" stroke="#2b2b2b" stroke-width="0.8" stroke-linejoin="round"/>'
            f'<path d="M5.4 14.9 L14.6 5.7 L18.3 9.4 L9.1 18.6 Z" fill="#f2c230" {S(w=1.2)}/>'
            f'<path d="M7.2 16.8 L16.4 7.6" {N} stroke="#c99a12" stroke-width="1"/>'
            f'<path d="M14.6 5.7 L16.4 4 L20 7.6 L18.3 9.4 Z" fill="#c4c9cf" {S(w=1.2)}/>'
            f'<path d="M16.4 4 L17.8 2.5 Q20.4 3.6 21.5 6.2 L20 7.6 Z" fill="#f5a3b8" {S(w=1.2)}/>',
    "arc": f'<path d="M4 20 A16 16 0 0 1 20 4" {N} {S()}/>'
           f'<circle cx="4" cy="20" r="1.7" fill="#2b2b2b"/><circle cx="20" cy="4" r="1.7" fill="#2b2b2b"/>',
    "arc2": f'<path d="M3 18 L21 18" {N} {S()} stroke-dasharray="3 2"/>'
            f'<path d="M3 18 A9 9 0 0 1 21 18" {N} {S()}/>'
            f'<path d="M12 9 L12 18" {N} stroke="#d23b3b" stroke-width="1.4"/>'
            f'<circle cx="3" cy="18" r="1.7" fill="#2b2b2b"/><circle cx="21" cy="18" r="1.7" fill="#2b2b2b"/>'
            f'<circle cx="12" cy="9" r="1.7" fill="#d23b3b"/>',
    "arc3": f'<path d="M3 19 A11 11 0 0 1 21 19" {N} {S()}/>'
            f'<circle cx="3" cy="19" r="1.7" fill="#2b2b2b"/><circle cx="12" cy="8" r="1.7" fill="#2b2b2b"/>'
            f'<circle cx="21" cy="19" r="1.7" fill="#2b2b2b"/>',
    "pie": f'<path d="M12 12 L20 12 A8 8 0 0 0 12 4 Z" {F} {S()}/>'
           f'<path d="M12 12 L20 12" {N} stroke="#d23b3b" stroke-width="1.4"/>'
           f'<circle cx="12" cy="12" r="1.5" fill="#2b2b2b"/>',
    "freehand": f'<path d="M3 18 C6 9 8 20 11 12 C13 6 15 16 18 9 C19 6 20.5 7 21 6" {N} {S()}/>',
    "rectangle": f'<rect x="3.5" y="6" width="17" height="12" {F} {S()}/>',
    "rotrect": f'<path d="M4 14 L15 6 L19 11.5 L8 19.5 Z" {F} {S()}/>'
               f'<path d="M4 14 L15 6" {N} stroke="#d23b3b" stroke-width="2.2" stroke-linecap="round"/>'
               f'<circle cx="4" cy="14" r="1.7" fill="#2b2b2b"/>',
    "circle": f'<circle cx="12" cy="12" r="8.5" {F} {S()}/><circle cx="12" cy="12" r="1" fill="#2b2b2b"/>',
    # rounded rectangle: the same corners as Rectangle, but with a radius in them
    "roundrect": f'<path d="M7 6 L17 6 A2 2 0 0 1 19 8 L19 16 A2 2 0 0 1 17 18 L7 18 A2 2 0 0 1 5 16 '
                 f'L5 8 A2 2 0 0 1 7 6 Z" {F} {S()}/>'
                 f'<path d="M5 12 L7.6 12 A1.4 1.4 0 0 0 9 10.6 L9 8" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.5"/>',
    "ellipse": f'<ellipse cx="12" cy="12" rx="9" ry="5.5" {F} {S()}/>'
               f'<path d="M3 12 L21 12 M12 6.5 L12 17.5" {N} stroke="#d23b3b" stroke-width="1.1" stroke-dasharray="2 1.6"/>'
               f'<circle cx="12" cy="12" r="1" fill="#2b2b2b"/>',
    # arc around a center: radius and angle are visible
    "centerarc": f'<path d="M12 12 L19 12" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>'
                 f'<path d="M12 12 L6.5 5.5" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>'
                 f'<path d="M19 12 A7 7 0 0 1 6.5 5.5" {N} {S()}/>'
                 f'<circle cx="12" cy="12" r="1.6" fill="#2b2b2b"/>',
    "polygon": f'<path d="M12 3 L20 7.5 L20 16.5 L12 21 L4 16.5 L4 7.5 Z" {F} {S()}/>',
    "text": f'<path d="M5 20 L12 4 L19 20 M8 14 L16 14" {N} {S(w=2.2)}/>',
    "pushpull": f'<path d="M4 13 L12 17 L20 13 L20 18 L12 22 L4 18 Z" fill="#b8c9de" {S()}/>'
                f'<path d="M4 13 L12 9 L20 13 L12 17 Z" fill="#e8eef7" {S()}/>'
                f'<path d="M12 13 L12 2 M8.5 5.5 L12 2 L15.5 5.5" {N} stroke="#d23b3b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "followme": f'<path d="M4 18 L4 10 A6 6 0 0 1 10 4 L17 4" {N} {S()}/>'
                f'<path d="M8 18 L8 12 A2 2 0 0 1 10 10 L17 10" {N} {S(w=1.1)}/>'
                f'<rect x="4" y="15" width="4" height="4" {F} {S()}/>'
                f'<path d="M17.5 1 L21.5 4.5 L17.5 8" {N} stroke="#d23b3b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
    "offset": f'<rect x="3" y="4" width="18" height="16" {N} {S()}/><rect x="7" y="8" width="10" height="8" {F} {S()}/>'
              f'<path d="M3 4 L7 8 M21 20 L17 16" stroke="#d23b3b" stroke-width="1.4"/>',
    "move": f'<path d="M12 2 L12 22 M2 12 L22 12 M9 5 L12 2 L15 5 M9 19 L12 22 L15 19 M5 9 L2 12 L5 15 M19 9 L22 12 L19 15" {N} {S("#d23b3b", 1.8)}/>',
    # rotate: block in the old position (dotted), rotated block and a curved arrow around the pivot
    "rotate": f'<rect x="6" y="9" width="10" height="10" {N} {S("#8a8a8a", 1.1)} stroke-dasharray="2 1.6"/>'
              f'<rect x="6" y="9" width="10" height="10" transform="rotate(-28 11 14)" {F} {S()}/>'
              f'<path d="M1.8 11.5 A9.5 9.5 0 0 1 18.5 8.3" {N} stroke="#d23b3b" stroke-width="2" stroke-linecap="round"/>'
              f'<path d="M20.4 10.6 L20.2 6.7 L16.8 9.3 Z" fill="#d23b3b" stroke="#d23b3b" stroke-width="1" stroke-linejoin="round"/>'
              f'<circle cx="11" cy="14" r="1.5" fill="#2b2b2b"/>',
    "scale": f'<rect x="3" y="9" width="12" height="12" {F} {S()}/><rect x="3" y="3" width="18" height="18" {N} {S()} stroke-dasharray="2 2"/>'
             f'<path d="M12 12 L20 4 M15 4 L20 4 L20 9" {N} stroke="#d23b3b" stroke-width="1.8" stroke-linecap="round"/>',
    "tape": f'<circle cx="10" cy="12" r="7" fill="#f2c230" {S()}/><circle cx="10" cy="12" r="2.2" fill="#fff" {S()}/>'
            f'<path d="M10 19 L22 19 L22 16 L17 16" fill="#f2e5a0" {S()}/>',
    # orbit (camera): globe with a red orbit around it
    "orbit": f'<g transform="rotate(-18 12 12)">'
             f'<path d="M22.5 12 A10.5 4.2 0 0 0 1.5 12" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="1.8 1.6"/></g>'
             f'<circle cx="12" cy="12" r="7.2" fill="#cfe0f5" {S()}/>'
             f'<ellipse cx="12" cy="12" rx="3.2" ry="7.2" {N} {S(w=1)}/>'
             f'<path d="M4.8 12 L19.2 12 M6 8.2 L18 8.2 M6 15.8 L18 15.8" {N} {S(w=1)}/>'
             f'<g transform="rotate(-18 12 12)">'
             f'<path d="M1.5 12 A10.5 4.2 0 0 0 21.6 13.6" {N} stroke="#d23b3b" stroke-width="2" stroke-linecap="round"/>'
             f'<path d="M23.4 9.4 L19.4 12.6 L23.6 14 Z" fill="#d23b3b" stroke="#d23b3b" stroke-width="1" stroke-linejoin="round"/></g>',
    "pan": f'<path d="M8 12 L8 5 A1.5 1.5 0 0 1 11 5 L11 11 L11 3.5 A1.5 1.5 0 0 1 14 3.5 L14 11 L14 5 A1.5 1.5 0 0 1 17 5 '
           f'L17 14 C17 18 15 21 12 21 C9 21 7.5 19.5 6 17 L3.5 13 A1.4 1.4 0 0 1 6 11.8 Z" fill="#fff" {S()}/>',
    "zoom": f'<circle cx="10" cy="10" r="6.5" {F} {S()}/><path d="M15 15 L21 21" {S(w=3)}/>',
    "zoom_extents": f'<circle cx="11" cy="11" r="5" {F} {S()}/><path d="M15 15 L19 19" {S(w=2.6)}/>'
                    f'<path d="M2 7 L2 2 L7 2 M17 2 L22 2 L22 7 M2 17 L2 22 L7 22" {N} stroke="#d23b3b" stroke-width="1.6"/>',
    "new": f'<path d="M6 2.5 L14 2.5 L19 7.5 L19 21.5 L6 21.5 Z" fill="#fff" {S()}/><path d="M14 2.5 L14 7.5 L19 7.5" {N} {S()}/>',
    "open": f'<path d="M2.5 6 L9 6 L11 8 L21 8 L21 19 L2.5 19 Z" fill="#f2c230" {S()}/>',
    "save": f'<path d="M3 3 L18 3 L21 6 L21 21 L3 21 Z" fill="#2f6fd6" {S()}/><rect x="7" y="3" width="9" height="6" fill="#fff" {S()}/>'
            f'<rect x="6" y="13" width="12" height="8" fill="#fff" {S()}/>',
    "undo": f'<path d="M9 5 L4 10 L9 15 M4 10 L15 10 A5 5 0 0 1 15 20 L10 20" {N} {S(w=2)}/>',
    "redo": f'<path d="M15 5 L20 10 L15 15 M20 10 L9 10 A5 5 0 0 0 9 20 L14 20" {N} {S(w=2)}/>',
    "stl": f'<path d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z" fill="#9a9a9a" {S()}/><path d="M3 7 L12 12 L21 7 M12 12 L12 22" {N} {S()}/>',
    "3mf": f'<path d="M12 2 L21 7 L12 12 L3 7 Z" fill="#d23b3b" {S()}/><path d="M3 7 L12 12 L12 22 L3 17 Z" fill="#2f6fd6" {S()}/>'
           f'<path d="M21 7 L12 12 L12 22 L21 17 Z" fill="#f2c230" {S()}/>',
    "union": f'<circle cx="9" cy="12" r="6" fill="#9fd29b" {S()}/><circle cx="15" cy="12" r="6" fill="#9fd29b" {S()}/>'
             f'<path d="M12 6.8 A6 6 0 0 1 12 17.2 A6 6 0 0 1 12 6.8" fill="#9fd29b"/>',
    "subtract": f'<circle cx="9" cy="12" r="6" fill="#9fd29b" {S()}/><circle cx="15" cy="12" r="6" fill="#fff" {S()}/>',
    "intersect": f'<circle cx="9" cy="12" r="6" {N} {S()}/><circle cx="15" cy="12" r="6" {N} {S()}/>'
                 f'<path d="M12 6.8 A6 6 0 0 1 12 17.2 A6 6 0 0 1 12 6.8" fill="#9fd29b" {S()}/>',
    "trim": (f'<circle cx="9" cy="12" r="6" {N} {S()} stroke-dasharray="3 2"/>'
             f'<path d="M15 6 A6 6 0 1 1 15 18 A6 6 0 1 1 15 6 Z M12 6.8 A6 6 0 0 1 12 17.2 '
             f'A6 6 0 0 1 12 6.8 Z" fill-rule="evenodd" fill="#9fd29b" {S()}/>'),
    "shell": f'<path d="M12 3 L20 7.5 L12 12 L4 7.5 Z" fill="#e8eef7" {S()}/>'
             f'<path d="M4 7.5 L12 12 L12 21 L4 16.5 Z" fill="#b8c9de" {S()}/>'
             f'<path d="M20 7.5 L12 12 L12 21 L20 16.5 Z" fill="#8fa6c3" {S()}/>'
             f'<path d="M12 8.5 L15 10 L12 11.5 L9 10 Z" fill="#fff" {S(w=1.1)}/>',
    "cut_faces": f'<path d="M3 16 L12 21 L21 16 L21 18.5 L12 23.5 L3 18.5 Z" fill="#b8c9de" {S()}/>'
                 f'<path d="M3 8 L12 13 L21 8 L12 3 Z" fill="#e8eef7" {S()}/>'
                 f'<path d="M12 13 L12 21" {N} stroke="#d23b3b" stroke-width="1.6" stroke-dasharray="2.5 2"/>',
    "check": f'<circle cx="12" cy="12" r="9" fill="#3fa34d" {S()}/><path d="M7 12 L11 16 L17 8" {N} stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
    "bed": f'<path d="M2 15 L12 20 L22 15 L12 10 Z" fill="#cfe0c5" {S()}/><path d="M12 4 L12 12 M9 9 L12 12 L15 9" {N} stroke="#d23b3b" stroke-width="1.8"/>',
    "view_iso": f'<path d="M12 3 L20 7.5 L12 12 L4 7.5 Z" fill="#e8eef7" {S()}/><path d="M4 7.5 L12 12 L12 21 L4 16.5 Z" fill="#b8c9de" {S()}/>'
                f'<path d="M20 7.5 L12 12 L12 21 L20 16.5 Z" fill="#8fa6c3" {S()}/>',
    "view_top": f'<rect x="4" y="4" width="16" height="16" fill="#e8eef7" {S()}/><path d="M12 8 L12 16 M8 12 L16 12" stroke="#2f6fd6" stroke-width="1.6"/>',
    "view_front": f'<rect x="4" y="4" width="16" height="16" fill="#b8c9de" {S()}/><path d="M4 20 L20 20" stroke="#d23b3b" stroke-width="2.4"/>',
    "view_right": f'<rect x="4" y="4" width="16" height="16" fill="#8fa6c3" {S()}/><path d="M4 20 L20 20" stroke="#3fa34d" stroke-width="2.4"/>',
    "view_back": f'<rect x="4" y="4" width="16" height="16" fill="#b8c9de" {S()}/><path d="M4 4 L20 20" stroke="#888" stroke-width="1"/>',
    "view_left": f'<rect x="4" y="4" width="16" height="16" fill="#8fa6c3" {S()}/><path d="M20 4 L4 20" stroke="#888" stroke-width="1"/>',
    "xray": f'<rect x="4" y="4" width="16" height="16" fill="#cfe0f5" fill-opacity="0.5" {S()}/><rect x="8" y="8" width="8" height="8" {N} {S()} stroke-dasharray="2 1.5"/>',
    "thread": f'<path d="M10 2.5 L14 2.5 L14 21.5 L10 21.5 Z" {F} {S()}/>'
              f'<path d="M10 5.5 L14 7.5 M10 9 L14 11 M10 12.5 L14 14.5 M10 16 L14 18" {N} stroke="#d23b3b" stroke-width="1.5"/>',
    "thread_ext": f'<path d="M4 9 L20 9 L20 15 L4 15 Z" {F} {S()}/>'
                  f'<path d="M6.5 9 L9 15 M10 9 L12.5 15 M13.5 9 L16 15" {N} stroke="#d23b3b" stroke-width="1.5"/>',
    "thread_int": f'<path d="M12 3 L20 7.5 L20 16.5 L12 21 L4 16.5 L4 7.5 Z" {F} {S()}/>'
                  f'<circle cx="12" cy="12" r="5" {N} {S()} stroke-dasharray="2 1.6"/>',
    "bolt": f'<path d="M6 3.5 L18 3.5 L18 8 L15 8 L15 20.5 L9 20.5 L9 8 L6 8 Z" {F} {S()}/>'
            f'<path d="M9 10.5 L15 12.5 M9 13.5 L15 15.5 M9 16.5 L15 18.5" {N} stroke="#d23b3b" stroke-width="1.3"/>',
    "nut": f'<path d="M12 3 L20 7.5 L20 16.5 L12 21 L4 16.5 L4 7.5 Z" {F} {S()}/>'
           f'<circle cx="12" cy="12" r="4.5" fill="#fff" {S()}/>',
    # cut: block in two halves, the cutting plane (red dotted) and the red cut surface
    "cutplane": f'<path d="M4 13 L12 17 L12 22 L4 18 Z" fill="#b8c9de" {S()}/>'
                f'<path d="M12 17 L20 13 L20 18 L12 22 Z" fill="#8fa6c3" {S()}/>'
                f'<path d="M0.8 13 L12 18.6 L23.2 13 L12 7.4 Z" fill="#d23b3b" fill-opacity="0.12" stroke="#d23b3b" stroke-width="1" stroke-dasharray="2 1.4"/>'
                f'<path d="M4 13 L12 17 L20 13 L12 9 Z" fill="#f2a6a0" {S()}/>'
                f'<path d="M4 5.5 L12 9.5 L12 12.5 L4 8.5 Z" fill="#b8c9de" {S()}/>'
                f'<path d="M12 9.5 L20 5.5 L20 8.5 L12 12.5 Z" fill="#8fa6c3" {S()}/>'
                f'<path d="M4 5.5 L12 9.5 L20 5.5 L12 1.5 Z" fill="#e8eef7" {S()}/>',
    # cut and join: two halves with a dovetail and matching recess, the cut in red
    "cut_join": f'<path d="M2 4 L10 4 L10 10 L14.5 8 L14.5 16 L10 14 L10 20 L2 20 Z" {F} {S()}/>'
                f'<path d="M15.5 4 L22 4 L22 20 L15.5 20 L15.5 14 L20 16 L20 8 L15.5 10 Z" fill="#9fd29b" {S()}/>'
                f'<path d="M15 1.5 L15 22.5" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>',
    "fillet": f'<path d="M3 3 L21 3 M3 3 L3 21" {N} {S()} stroke-dasharray="3 2"/>'
              f'<path d="M3 21 L3 12 A9 9 0 0 1 12 3 L21 3" {N} stroke="#d23b3b" stroke-width="2.2" stroke-linecap="round"/>'
              f'<circle cx="3" cy="3" r="1.6" fill="#2b2b2b"/>',
    "chamfer": f'<path d="M3 3 L21 3 M3 3 L3 21" {N} {S()} stroke-dasharray="3 2"/>'
               f'<path d="M3 21 L3 12 L12 3 L21 3" {N} stroke="#d23b3b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
               f'<circle cx="3" cy="3" r="1.6" fill="#2b2b2b"/>',
    "mirror_h": f'<path d="M12 2 L12 22" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>'
                f'<path d="M3 7 L9.5 12 L3 17 Z" {F} {S()}/>'
                f'<path d="M21 7 L14.5 12 L21 17 Z" fill="#e8eef7" stroke="#8a8a8a" stroke-width="1.2"/>',
    "mirror_v": f'<path d="M2 12 L22 12" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>'
                f'<path d="M7 3 L12 9.5 L17 3 Z" {F} {S()}/>'
                f'<path d="M7 21 L12 14.5 L17 21 Z" fill="#e8eef7" stroke="#8a8a8a" stroke-width="1.2"/>',
    # 2D worksheet: join lines, split, align, distribute, duplicate, grid
    "join": f'<path d="M3 18 L9 18 L9 6" {N} {S(w=2.2)}/>'
            f'<path d="M13 6 L21 6" {N} {S(w=2.2)} stroke-dasharray="3 2"/>'
            f'<circle cx="9" cy="6" r="1.8" fill="#d23b3b"/><circle cx="13" cy="6" r="1.8" fill="#fff" {S()}/>',
    "split": f'<path d="M3 12 L9 12" {N} {S(w=2.2)}/><path d="M12.5 12 L18 12" {N} {S(w=2.2)}/>'
             f'<path d="M21 12 L22.5 12" {N} {S(w=2.2)}/>'
             f'<path d="M10.6 6 L10.6 18 M14.4 6 L14.4 18" {N} stroke="#d23b3b" stroke-width="1.2" stroke-dasharray="2 1.6"/>',
    "align": f'<path d="M3 3 L3 21" {N} stroke="#d23b3b" stroke-width="1.4"/>'
             f'<rect x="5.5" y="5" width="13" height="4" {F} {S(w=1.2)}/>'
             f'<rect x="5.5" y="11" width="8" height="4" {F} {S(w=1.2)}/>'
             f'<rect x="5.5" y="17" width="11" height="3" {F} {S(w=1.2)}/>',
    "distribute": f'<rect x="3" y="9" width="4" height="6" {F} {S(w=1.2)}/>'
                  f'<rect x="10" y="9" width="4" height="6" {F} {S(w=1.2)}/>'
                  f'<rect x="17" y="9" width="4" height="6" {F} {S(w=1.2)}/>'
                  f'<path d="M7.4 12 L9.6 12 M14.4 12 L16.6 12" {N} stroke="#d23b3b" stroke-width="1.4"/>',
    "duplicate": f'<rect x="3" y="3" width="11" height="11" {F} {S()}/>'
                 f'<rect x="10" y="10" width="11" height="11" fill="#fff" {S()}/>'
                 f'<path d="M15.5 12.5 L15.5 18.5 M12.5 15.5 L18.5 15.5" {N} stroke="#d23b3b" stroke-width="1.6"/>',
    "grid": f'<path d="M3 3 L21 3 L21 21 L3 21 Z M9 3 L9 21 M15 3 L15 21 M3 9 L21 9 M3 15 L21 15" {N} {S(w=1.1)}/>'
            f'<circle cx="9" cy="9" r="2" fill="#d23b3b"/>',
    "round_join": f'<path d="M3 21 L3 12 A9 9 0 0 1 12 3 L21 3" {N} stroke="#d23b3b" stroke-width="2.2" stroke-linecap="round"/>'
                  f'<path d="M3 3 L21 3 M3 3 L3 21" {N} {S()} stroke-dasharray="3 2"/>',
    "sharp_join": f'<path d="M3 21 L3 3 L21 3" {N} stroke="#d23b3b" stroke-width="2.2" stroke-linejoin="miter"/>'
                  f'<path d="M7 17 L7 7 L17 7" {N} {S()} stroke-dasharray="3 2"/>',
    "ortho": f'<path d="M4 8 L14 8 L14 20 L4 20 Z M4 8 L10 4 L20 4 L14 8 M14 20 L20 16 L20 4" {N} {S()}/>',
    # eye and lock of the layers panel
    "eye": f'<path d="M2 12 C6 5.5 18 5.5 22 12 C18 18.5 6 18.5 2 12 Z" fill="#e8eef7" {S()}/>'
           '<circle cx="12" cy="12" r="3.4" fill="#2f6fd6" stroke="#2b2b2b" stroke-width="1.2"/>',
    "eye_off": f'<path d="M2 12 C6 5.5 18 5.5 22 12 C18 18.5 6 18.5 2 12 Z" fill="#f0f0f0" '
               f'{S("#9a9a9a")}/><path d="M4 20 L20 4" {N} stroke="#d23b3b" stroke-width="2.2" '
               'stroke-linecap="round"/>',
    "lock": f'<rect x="5" y="10" width="14" height="11" rx="2" fill="#f2c230" {S()}/>'
            f'<path d="M8 10 L8 7 A4 4 0 0 1 16 7 L16 10" {N} {S()}/>',
    "unlock": f'<rect x="5" y="10" width="14" height="11" rx="2" fill="#f2c230" {S()}/>'
              f'<path d="M8 10 L8 7 A4 4 0 0 1 15.5 6" {N} {S()}/>',
    "layers": '<path d="M12 3 L21 8 L12 13 L3 8 Z" fill="#e8eef7" stroke="#2b2b2b" stroke-width="1.4" '
              'stroke-linejoin="round"/><path d="M3 12.5 L12 17.5 L21 12.5" fill="none" stroke="#2b2b2b" '
              'stroke-width="1.4" stroke-linejoin="round"/><path d="M3 17 L12 22 L21 17" fill="none" '
              'stroke="#2b2b2b" stroke-width="1.4" stroke-linejoin="round"/>',
    "sheet": f'<rect x="2.5" y="5" width="19" height="14" rx="1" fill="#eef2f7" {S()}/>'
             f'<rect x="5.5" y="8" width="13" height="8" {N} stroke="#8a8a8a" stroke-width="1" '
             f'stroke-dasharray="2 1.6"/>',
    "nest": f'<rect x="2.5" y="4" width="19" height="16" rx="1" fill="#eef2f7" {S()}/>'
            f'<rect x="4.5" y="6" width="8" height="6" fill="#9fd29b" {S(w=1.2)}/>'
            f'<rect x="13.5" y="6" width="6" height="12" fill="#cfe0f5" {S(w=1.2)}/>'
            f'<rect x="4.5" y="13" width="8" height="5" fill="#f2c230" {S(w=1.2)}/>',
    # finger-joint box: an open box with teeth above the rim
    "boxjoint": f'<path d="M4 9 L12 4.5 L20 9 L12 13.5 Z" fill="#cfe0f5" {S()}/>'
                f'<path d="M4 9 L4 18 L12 22.5 L12 13.5 Z" fill="#9fd29b" {S()}/>'
                f'<path d="M20 9 L20 18 L12 22.5 L12 13.5 Z" fill="#e6ecf3" {S()}/>'
                f'<path d="M4 9 L4 6 M7 6 L7 9 M10 6 L10 9" {N} stroke="#d23b3b" stroke-width="1.5" '
                f'stroke-linecap="round"/>'
                f'<path d="M20 9 L20 6 M17 6 L17 9 M14 6 L14 9" {N} stroke="#d23b3b" '
                f'stroke-width="1.5" stroke-linecap="round"/>',
    # living hinge: staggered slits in a plate of plywood
    "hinge": f'<rect x="3" y="5" width="18" height="14" rx="1" fill="#e8c897" {S()}/>'
             f'<path d="M7.5 7 L7.5 11.5 M7.5 13.5 L7.5 17 M12 7 L12 9.5 M12 11.5 L12 14.5 '
             f'M12 16.5 L12 17 M16.5 7 L16.5 11.5 M16.5 13.5 L16.5 17" {N} stroke="#d23b3b" '
             f'stroke-width="1.5" stroke-linecap="round"/>',
    # worksheet to model: a red cut contour rising up as a thick plate
    "to_model": f'<path d="M3 20 L11 16 L21 19 L13 23 Z" {N} stroke="#d23b3b" stroke-width="1.4" '
                f'stroke-dasharray="2 1.4" stroke-linejoin="round"/>'
                f'<path d="M3 9 L11 5 L21 8 L13 12 Z" fill="#e8c897" {S()}/>'
                f'<path d="M3 9 L3 12 L13 15 L13 12 Z" fill="#c9a26a" {S()}/>'
                f'<path d="M13 12 L13 15 L21 11 L21 8 Z" fill="#b08850" {S()}/>',
    # model to worksheet: a face with a red laser beam on it
    "to_laser": f'<path d="M2 17 L10 13 L22 16 L14 20.5 Z" fill="#cfe0f5" {S()}/>'
                f'<path d="M7 16 L11 14.2 L17 15.8 L13 17.8 Z" {N} stroke="#d23b3b" '
                f'stroke-width="1.5" stroke-linejoin="round"/>'
                f'<rect x="10" y="2.5" width="6" height="5" rx="1" fill="#6b7685" {S(w=1.2)}/>'
                f'<path d="M13 7.5 L13 15" {N} stroke="#d23b3b" stroke-width="1.6"/>',
    "app": '<rect x="1" y="1" width="22" height="22" rx="5" fill="#1f3b63"/>'
           '<path d="M12 4 L20 8.5 L12 13 L4 8.5 Z" fill="#f2c230"/>'
           '<path d="M4 8.5 L12 13 L12 21 L4 16.5 Z" fill="#d23b3b"/>'
           '<path d="M20 8.5 L12 13 L12 21 L20 16.5 Z" fill="#2f6fd6"/>',
}

_cache = {}
_pixmaps = {}


def _dark_body(body):
    """Make the dark lines of an icon light, for the dark theme."""
    for dark, light in theme.ICON_INK.items():
        body = body.replace(f'"{dark}"', f'"{light}"')
    return body


def pixmap(name, size=24, dark=False):
    """The SVG drawing of `name` as a pixmap of `size` x `size` pixels."""
    body = SVG.get(name, SVG["select"])
    if dark:
        body = _dark_body(body)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">{body}</svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return QPixmap.fromImage(img)


class _ThemedIcon(QIconEngine):
    """Draws the icon in the current theme on every request: a theme switch then
    needs no new icons on all the buttons."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def pixmap(self, size, mode, state):
        side = max(1, min(size.width(), size.height()))
        dark = theme.is_dark()
        key = (self.name, side, dark, int(mode.value) if hasattr(mode, "value") else int(mode))
        pm = _pixmaps.get(key)
        if pm is None:
            pm = pixmap(self.name, side, dark)
            if mode == QIcon.Disabled:
                app = QGuiApplication.instance()
                style = app.style() if app is not None and hasattr(app, "style") else None
                if style is not None:
                    from PySide6.QtWidgets import QStyleOption
                    pm = style.generatedIconPixmap(QIcon.Disabled, pm, QStyleOption())
            _pixmaps[key] = pm
        return pm

    def scaledPixmap(self, size, mode, state, scale):
        """Sharp on a scaled screen: draw in real pixels."""
        scale = max(1.0, float(scale))
        pm = QPixmap(self.pixmap(QSize(round(size.width() * scale), round(size.height() * scale)),
                                 mode, state))
        pm.setDevicePixelRatio(scale)
        return pm

    def paint(self, painter, rect, mode, state):
        scale = painter.device().devicePixelRatioF() if painter.device() is not None else 1.0
        painter.drawPixmap(rect, self.scaledPixmap(rect.size(), mode, state, scale))

    def actualSize(self, size, mode, state):
        side = min(size.width(), size.height())
        return size.scaled(side, side, Qt.KeepAspectRatio)

    def availableSizes(self, mode=QIcon.Normal, state=QIcon.Off):
        return [QSize(s, s) for s in (16, 24, 32, 48, 64)]

    def clone(self):
        return _ThemedIcon(self.name)


def icon(name, size=32):
    key = (name, size)
    if key in _cache:
        return _cache[key][0]
    engine = _ThemedIcon(name)
    ic = QIcon(engine)
    _cache[key] = (ic, engine)          # the engine must stay alive as long as the icon exists
    return ic


# Where the tool "touches" the point under the cursor: pencil tip, eraser corner, paint drop.
CURSOR_HOTSPOTS = {"line": (3, 21), "freehand": (3, 18), "eraser": (5, 19), "paint": (4, 22)}
_cursors = {}


def cursor(name, size=24):
    """The tool icon as mouse pointer, like SketchUp shows it (Select keeps the arrow).

    Made on demand and cached: a QCursor needs a QApplication that already exists.
    """
    if not name or name == "select":
        return QCursor(Qt.ArrowCursor)
    if QGuiApplication.instance() is None:
        return QCursor(Qt.ArrowCursor)
    key = (name, size)
    if key not in _cursors:
        hx, hy = CURSOR_HOTSPOTS.get(name, (size // 2, size // 2))
        _cursors[key] = QCursor(pixmap(name, size), hx, hy)
    return _cursors[key]
