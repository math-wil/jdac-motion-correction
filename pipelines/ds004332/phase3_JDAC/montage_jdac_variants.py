#!/usr/bin/env python3
"""
Montage de la figure B (variantes JDAC) a partir des captures FSLeyes 01..08.

- recadre chaque capture sur le cerveau (retire le noir autour),
- met toutes les vignettes a la meme hauteur,
- assemble une grille 2 lignes x 4 colonnes serree (petit interstice noir),
- ajoute les titres de colonnes et les labels de lignes.

Ordre attendu des captures (dans IN_DIR) :
  01 entree, 02 normal, 03 sans débruitage 4x, 04 anti-artefact 1x   (sub-01 immobile)
  05 entree, 06 normal, 07 sans débruitage 4x, 08 anti-artefact 1x   (sub-19 sévère)

Sortie : OUT (ecrase la figure du meme nom dans le vault).
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IN_DIR = Path.home() / "Pictures/jdac_images"
OUT = Path.home() / "Documents/research-notes/99_Attachements/jdac_variants_rigid.png"

COLS = ["entrée (cerveau rigide)", "JDAC normal", "sans débruitage (4×)", "anti-artefact (1×)"]
ROWS = ["sub-01_run-01\nAgit. 0.20\nimmobile", "sub-19_run-03\nAgit. 3.15\nsévère"]

ROW_H = 460        # hauteur cible des vignettes
GAP = 6            # interstice entre vignettes
TITLE_H = 40       # bandeau titres colonnes
LABEL_W = 168      # colonne gauche pour les labels de lignes
MARGIN = 10
BG = (0, 0, 0)
FG = (235, 235, 235)


def crop_brain(im, thr=12, pad=6):
    """Recadre sur les pixels non noirs."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return im
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(im.width, x1 + pad); y1 = min(im.height, y1 + pad)
    return im.crop((x0, y0, x1, y1))


def to_height(im, h):
    w = int(round(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    # charger, recadrer, normaliser la hauteur
    tiles = []
    for i in range(1, 9):
        im = Image.open(IN_DIR / f"{i:02d}.png")
        tiles.append(to_height(crop_brain(im), ROW_H))

    cell_w = max(t.width for t in tiles)          # largeur de cellule commune
    grid_w = LABEL_W + 4 * cell_w + 3 * GAP
    grid_h = TITLE_H + 2 * ROW_H + GAP
    canvas = Image.new("RGB", (grid_w + 2 * MARGIN, grid_h + 2 * MARGIN), BG)
    draw = ImageDraw.Draw(canvas)
    f_title = get_font(22)
    f_label = get_font(18)

    # titres de colonnes
    for c, t in enumerate(COLS):
        cx = MARGIN + LABEL_W + c * (cell_w + GAP) + cell_w // 2
        bb = draw.textbbox((0, 0), t, font=f_title)
        draw.text((cx - (bb[2] - bb[0]) // 2, MARGIN + 6), t, font=f_title, fill=FG)

    # vignettes + labels de lignes
    for r in range(2):
        y = MARGIN + TITLE_H + r * (ROW_H + GAP)
        # label ligne (multi-lignes, centre vertical)
        lab = ROWS[r]
        bb = draw.multiline_textbbox((0, 0), lab, font=f_label, spacing=4)
        draw.multiline_text((MARGIN + 6, y + ROW_H // 2 - (bb[3] - bb[1]) // 2),
                            lab, font=f_label, fill=FG, spacing=4)
        for c in range(4):
            t = tiles[r * 4 + c]
            x = MARGIN + LABEL_W + c * (cell_w + GAP) + (cell_w - t.width) // 2
            canvas.paste(t, (x, y))

    canvas.save(str(OUT))
    print(f"Montage -> {OUT}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
