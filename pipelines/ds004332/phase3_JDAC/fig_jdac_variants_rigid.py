#!/usr/bin/env python3
"""
Figure de comparaison des variantes de JDAC sur le pipeline RIGIDE, reunion 2026-07-02.

Pour deux acquisitions (faible et fort mouvement), on aligne :
    entree (cerveau rigide) | JDAC normal | JDAC sans débruitage 4x | anti-artefact 1x

Sert a la fois de figure "effet JDAC" (2 premieres colonnes) et de figure des
deux nouvelles branches sans debruitage.

- Toutes les images partagent la meme grille rigide (meme affine) -> meme coupe.
- Coupe axiale, RAS canonique, fenetrage percentile 99.9 par panneau (l'entree est
  en intensite brute, les sorties JDAC en [0,1]).

Sortie : results/ds004332/phase3_JDAC/jdac_variants_rigid.png

Usage (env cortical-motion) :
    python fig_jdac_variants_rigid.py
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
D = HOME / "Documents/derivatives/ds004332"
PRE = D / "preproc_rigid"
JN = D / "jdac_rigid"               # normal
JL = D / "jdac_rigid_nodenoise"     # sans débruitage, 4 passes
JO = D / "jdac_rigid_antiartonly"   # anti-artefact, 1 passe
OUT = HOME / "Documents/jdac-motion-correction/results/ds004332/phase3_JDAC/jdac_variants_rigid.png"

# (id, score Agitation) : un scan immobile et un scan tres bouge
CASES = [("sub-01_run-01", 0.20), ("sub-19_run-03", 3.15)]


def load_canonical(path):
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata()


def axial(vol, frac=0.52):
    k = int(round(vol.shape[2] * frac))
    return vol[:, :, k].T


def window(img):
    return 0.0, float(np.percentile(img, 99.9))


def main():
    cols = ["entrée (cerveau rigide)", "JDAC normal",
            "sans débruitage (4×)", "anti-artefact (1×)"]
    fig, axes = plt.subplots(len(CASES), 4, figsize=(18, 4.6 * len(CASES)))
    axes = np.atleast_2d(axes)

    for row, (sid, agit) in enumerate(CASES):
        vols = [
            load_canonical(PRE / sid / f"{sid}_brain.nii.gz"),
            load_canonical(JN / sid / f"{sid}_T1w_jdac.nii.gz"),
            load_canonical(JL / sid / f"{sid}_T1w_jdac_nodenoise.nii.gz"),
            load_canonical(JO / sid / f"{sid}_T1w_jdac_antiartonly.nii.gz"),
        ]
        for col, vol in enumerate(vols):
            sl = axial(vol)
            lo, hi = window(sl)
            axes[row, col].imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{sid}\nAgitation {agit}", fontsize=10)

    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle("JDAC et variantes sans débruitage — pipeline rigide, coupe axiale", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
