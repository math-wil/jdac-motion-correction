#!/usr/bin/env python3
"""
Figure des 2 etapes du preprocessing (abp_n4 + mri_synthstrip) pour la reunion.

Grille 2 lignes x 3 colonnes :
    lignes   = sub-01 run-01 (Agitation 0.20, immobile) et run-03 (3.16, tres bouge)
    colonnes = brut | _n4 (biais corrige) | _n4 + masque skull-strip (rouge, rempli)

Garantie de position :
- DANS un run, brut/_n4/_mask partagent la meme grille voxel -> meme point anatomique.
- Orientation : images remises en RAS canonique (nib.as_closest_canonical) avant la coupe,
  pour un affichage anatomique correct (superieur en haut), comme FSLeyes.
- Coupe sagittale (proche de la ligne mediane).

Sortie : results/ds004332/phase2_PREPROC/preproc_steps_sub01.png

Usage (env cortical-motion) :
    python fig_preproc_steps.py
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HOME = Path.home()
RAW = HOME / "Documents/raw_datasets/ds004332"
PRE = HOME / "Documents/derivatives/ds004332/preproc_natif"
OUT = HOME / "Documents/jdac-motion-correction/results/ds004332/phase2_PREPROC/preproc_steps_sub01.png"

RUNS = [("run-01", 0.20), ("run-03", 3.16)]   # (run, score Agitation)
SUB = "sub-01"
RED = ListedColormap(["red"])


def load_canonical(path):
    """Charge en RAS canonique (orientation anatomique standard)."""
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata()


def axial(vol, frac=0.52):
    """Coupe axiale (axe 2 = I-S en RAS) a la position fractionnaire frac
    (frac~0.52 = niveau ventricules/noyaux gris).

    En RAS : axe0 = L->R, axe1 = P->A. On affiche l'avant (A) en haut.
    """
    k = int(round(vol.shape[2] * frac))
    sl = vol[:, :, k]   # (L->R, P->A)
    return sl.T         # avec origin='lower' : avant (A) en haut


def window(img):
    # Plage quasi complete (comme FSLeyes, robuste off) : montre la difference
    # d'intensite brut vs n4 (le brut a un max bien plus haut -> apparait plus sombre).
    return 0.0, float(np.percentile(img, 99.9))


def main():
    argparse.ArgumentParser().parse_args()

    fig, axes = plt.subplots(len(RUNS), 3, figsize=(14, 4.6 * len(RUNS)))
    axes = np.atleast_2d(axes)

    for row, (run, agit) in enumerate(RUNS):
        sid = f"{SUB}_{run}"
        raw = load_canonical(RAW / SUB / "anat"
                             / f"{SUB}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii")
        n4 = load_canonical(PRE / sid / f"{sid}_n4.nii.gz")
        mask = load_canonical(PRE / sid / f"{sid}_mask.nii.gz")

        s_raw, s_n4, s_mk = axial(raw), axial(n4), axial(mask)

        # col 0 : brut
        lo, hi = window(s_raw)
        axes[row, 0].imshow(s_raw, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        # col 1 : n4
        lo, hi = window(s_n4)
        axes[row, 1].imshow(s_n4, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        # col 2 : n4 + masque rempli (rouge semi-transparent)
        axes[row, 2].imshow(s_n4, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        m = np.ma.masked_where(s_mk <= 0, s_mk)
        axes[row, 2].imshow(m, cmap=RED, origin="lower", alpha=0.35)

        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
        axes[row, 0].set_ylabel(f"{run}\nAgitation {agit}", fontsize=10)

    for c, t in enumerate(["brut", "n4 (biais corrigé)", "n4 + masque skull-strip"]):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle(f"Preprocessing 2 étapes (N4 + SynthStrip) — {SUB}, coupe axiale",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
