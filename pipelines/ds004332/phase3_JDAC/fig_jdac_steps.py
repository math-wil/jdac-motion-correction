#!/usr/bin/env python3
"""
Figure de l'effet JDAC sur sub-01, meme mise en page que fig_preproc_steps.py.

Grille 2 lignes x 3 colonnes :
    lignes   = sub-01 run-01 (Agitation 0.20, immobile) et run-03 (3.16, tres bouge)
    colonnes = brut | cerveau skull-strippe (N4 + SynthStrip) | post-JDAC

- Coupe axiale (axe 2 = I-S en RAS), meme position fractionnaire que fig_preproc_steps.
- Orientation RAS canonique (nib.as_closest_canonical) avant la coupe (superieur en haut).
- cerveau preproc (_brain) et post-JDAC (jdac_fixed) partagent la meme grille -> meme coupe.

Sortie : results/ds004332/phase3_JDAC/brut_preproc_jdac_sub01.png

Usage (env cortical-motion) :
    python fig_jdac_steps.py
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
RAW = HOME / "Documents/raw_datasets/ds004332"
PRE = HOME / "Documents/derivatives/ds004332/preproc"
JD = HOME / "Documents/derivatives/ds004332/jdac_fixed"
OUT = HOME / "Documents/jdac-motion-correction/results/ds004332/phase3_JDAC/brut_preproc_jdac_sub01.png"

RUNS = [("run-01", 0.20), ("run-03", 3.16)]   # (run, score Agitation)
SUB = "sub-01"


def load_canonical(path):
    """Charge en RAS canonique (orientation anatomique standard)."""
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata()


def axial(vol, frac=0.52):
    """Coupe axiale (axe 2 = I-S en RAS) a la position fractionnaire frac.
    En RAS : axe0 = L->R, axe1 = P->A. Avant (A) en haut avec origin='lower'."""
    k = int(round(vol.shape[2] * frac))
    return vol[:, :, k].T


def window(img):
    return 0.0, float(np.percentile(img, 99.9))


def main():
    argparse.ArgumentParser().parse_args()

    fig, axes = plt.subplots(len(RUNS), 3, figsize=(14, 4.6 * len(RUNS)))
    axes = np.atleast_2d(axes)

    for row, (run, agit) in enumerate(RUNS):
        sid = f"{SUB}_{run}"
        raw = load_canonical(RAW / SUB / "anat"
                             / f"{SUB}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii")
        brain = load_canonical(PRE / sid / f"{sid}_brain.nii.gz")
        jdac = load_canonical(JD / sid / f"{sid}_T1w_jdac.nii.gz")

        for col, vol in enumerate([raw, brain, jdac]):
            sl = axial(vol)
            lo, hi = window(sl)
            axes[row, col].imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{run}\nAgitation {agit}", fontsize=10)

    for c, t in enumerate(["brut", "cerveau skull-strippé (N4 + SynthStrip)", "post-JDAC"]):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle(f"Effet JDAC — {SUB}, coupe axiale (brut / preproc / JDAC)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
