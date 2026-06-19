#!/usr/bin/env python3
"""
Assemble la figure JDAC a partir des captures FSLeyes manuelles (~/Pictures/jdac_images).
Meme mise en page que fig_preproc_steps.py :
    lignes   = sub-01 run-01 (Agitation 0.20) et run-03 (3.16)
    colonnes = brut | cerveau skull-strippe (N4 + SynthStrip) | post-JDAC

Sortie : results/ds004332/phase3_JDAC/brut_preproc_jdac_sub01.png

Usage (env cortical-motion) :
    python assemble_jdac_fig.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HOME = Path.home()
SHOTS = HOME / "Pictures/jdac_images"
OUT = HOME / "Documents/jdac-motion-correction/results/ds004332/phase3_JDAC/brut_preproc_jdac_sub01.png"

# (run, score Agitation, prefixe des captures)
RUNS = [("run-01", 0.20, "sub01_run01"), ("run-03", 3.16, "sub01_run03")]
# (suffixe fichier, titre colonne)
COLS = [("", "brut"),
        ("_brain", "cerveau skull-strippé (N4 + SynthStrip)"),
        ("_jdac", "post-JDAC")]


def main():
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(14, 4.6 * len(RUNS)))

    for row, (run, agit, prefix) in enumerate(RUNS):
        for col, (suffix, _) in enumerate(COLS):
            ax = axes[row, col]
            path = SHOTS / f"{prefix}{suffix}.png"
            if path.is_file():
                ax.imshow(mpimg.imread(str(path)))
            else:
                ax.text(0.5, 0.5, f"MANQUE\n{path.name}", color="red",
                        ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{run}\nAgitation {agit}", fontsize=10)

    for col, (_, title) in enumerate(COLS):
        axes[0, col].set_title(title, fontsize=11)

    fig.suptitle("Effet JDAC — sub-01, coupe axiale (brut / cerveau / JDAC)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
