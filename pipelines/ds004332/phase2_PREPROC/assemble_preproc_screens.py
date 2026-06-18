#!/usr/bin/env python3
"""
Assemble les captures FSLeyes manuelles (coupe axiale) en un montage propre pour
la reunion : grille 2 lignes (run-01, run-03) x 3 colonnes (brut / n4 / n4 + masque).
Chaque capture est recadree sur le cerveau (bounding box du non-noir).

Entree : ~/Pictures/preproc_images/ (captures manuelles)
Sortie : results/ds004332/phase2_PREPROC/preproc_steps_sub01.png
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path.home() / "Pictures/preproc_images"
OUT = (Path.home() / "Documents/jdac-motion-correction/results/ds004332/"
       "phase2_PREPROC/preproc_steps_sub01.png")

# (run, agitation, {col: fichier})
ROWS = [
    ("run-01", 0.20, {"brut": "sub01_run01",
                      "n4": "sub01_run01_n4.png",
                      "mask": "sub01_run01_n4mask.png"}),
    ("run-03", 3.16, {"brut": "sub01_run03.png",
                      "n4": "sub01_run03_n4.png",
                      "mask": "sub01_run03_n4mask.png"}),
]
COLS = [("brut", "brut"), ("n4", "n4 (biais corrigé)"), ("mask", "n4 + masque skull-strip")]


def autocrop(fname, thr=12, margin=10):
    im = Image.open(SRC / fname).convert("RGB")
    a = np.asarray(im)
    m = a.max(2) > thr                       # tout pixel non-noir (gris ou rouge)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min() - margin), min(a.shape[1], xs.max() + margin)
    y0, y1 = max(0, ys.min() - margin), min(a.shape[0], ys.max() + margin)
    return im.crop((x0, y0, x1, y1))


fig, axes = plt.subplots(len(ROWS), 3, figsize=(13, 4.4 * len(ROWS)),
                         facecolor="black")
axes = np.atleast_2d(axes)
for r, (run, agit, files) in enumerate(ROWS):
    for c, (key, _) in enumerate(COLS):
        ax = axes[r, c]
        ax.imshow(autocrop(files[key]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("black")
    axes[r, 0].set_ylabel(f"{run}\nAgitation {agit}", color="white", fontsize=12)

for c, (_, title) in enumerate(COLS):
    axes[0, c].set_title(title, color="white", fontsize=13)

fig.suptitle("Preprocessing 2 étapes (N4 + SynthStrip) — sub-01, coupe axiale",
             color="white", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.subplots_adjust(wspace=0.02, hspace=0.06)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=150, facecolor="black")
print(f"Figure -> {OUT}")
