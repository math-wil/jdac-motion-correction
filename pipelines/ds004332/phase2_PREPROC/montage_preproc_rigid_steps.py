#!/usr/bin/env python3
"""
Montage de la figure A (etapes du preprocessing rigide) a partir des captures
FSLeyes 09..16 (dossier ~/Pictures/preproc_images).

Grille 2 lignes x 4 colonnes :
  colonnes = brut | N4 | recalage rigide | cerveau (SynthStrip)
  ligne 1  = sub-01_run-01 (immobile)   : 09 10 11 12
  ligne 2  = sub-19_run-03 (sévère)     : 13 14 15 16

Recadrage sur le cerveau, hauteur uniforme, interstice serre, titres + labels.
Sortie : 99_Attachements/preproc_rigid_steps.png
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IN_DIR = Path.home() / "Pictures/preproc_images"
OUT = Path.home() / "Documents/research-notes/99_Attachements/preproc_rigid_steps.png"

COLS = ["brut", "N4 (biais corrigé)", "recalage rigide MNI", "cerveau (SynthStrip)"]
ROWS = ["sub-01_run-01\nAgit. 0.20\nimmobile", "sub-19_run-03\nAgit. 3.15\nsévère"]
IDX = [9, 10, 11, 12, 13, 14, 15, 16]   # ordre des captures

ROW_H = 460
GAP = 6
TITLE_H = 40
LABEL_W = 168
MARGIN = 10
BG = (0, 0, 0)
FG = (235, 235, 235)


def crop_brain(im, thr=12, pad=6):
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return im
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(im.width, x1 + pad); y1 = min(im.height, y1 + pad)
    return im.crop((x0, y0, x1, y1))


def to_height(im, h):
    return im.resize((int(round(im.width * h / im.height)), h), Image.LANCZOS)


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    tiles = [to_height(crop_brain(Image.open(IN_DIR / f"{i:02d}.png")), ROW_H) for i in IDX]
    cell_w = max(t.width for t in tiles)
    grid_w = LABEL_W + 4 * cell_w + 3 * GAP
    grid_h = TITLE_H + 2 * ROW_H + GAP
    canvas = Image.new("RGB", (grid_w + 2 * MARGIN, grid_h + 2 * MARGIN), BG)
    draw = ImageDraw.Draw(canvas)
    f_title = get_font(22)
    f_label = get_font(18)

    for c, t in enumerate(COLS):
        cx = MARGIN + LABEL_W + c * (cell_w + GAP) + cell_w // 2
        bb = draw.textbbox((0, 0), t, font=f_title)
        draw.text((cx - (bb[2] - bb[0]) // 2, MARGIN + 6), t, font=f_title, fill=FG)

    for r in range(2):
        y = MARGIN + TITLE_H + r * (ROW_H + GAP)
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
