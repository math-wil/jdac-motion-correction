#!/usr/bin/env python3
"""
Figure des etapes du preprocessing RIGIDE (N4 + recalage rigide + SynthStrip),
pour la reunion 2026-07-02. Remplace l'ancienne figure non rigide.

Grille 2 lignes x 4 colonnes :
    lignes   = sub-01 run-01 (Agitation 0.20, immobile) et run-03 (3.16, tres bouge)
    colonnes = brut | _n4 (biais corrige) | _rigid (recale rigide vers MNI) | _brain (skull-strip)

- Coupe axiale (axe 2 = I-S en RAS), orientation RAS canonique avant la coupe.
- brut et _n4 partagent la grille d'origine ; _rigid et _brain la grille MNI elargie.
  Chaque panneau est fenetre independamment (percentile 99.9).

Sortie : results/ds004332/phase2_PREPROC/preproc_rigid_steps_sub01.png

Usage (env cortical-motion) :
    python fig_preproc_rigid_steps.py
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
RAW = HOME / "Documents/raw_datasets/ds004332"
PRE = HOME / "Documents/derivatives/ds004332/preproc_rigid"
OUT = HOME / "Documents/jdac-motion-correction/results/ds004332/phase2_PREPROC/preproc_rigid_steps_sub01.png"

SUB = "sub-01"
RUNS = [("run-01", 0.20), ("run-03", 3.16)]   # (run, score Agitation)


def load_canonical(path):
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata()


def axial(vol, frac=0.52):
    k = int(round(vol.shape[2] * frac))
    return vol[:, :, k].T   # origin='lower' -> avant (A) en haut


def window(img):
    return 0.0, float(np.percentile(img, 99.9))


def main():
    fig, axes = plt.subplots(len(RUNS), 4, figsize=(18, 4.6 * len(RUNS)))
    axes = np.atleast_2d(axes)

    for row, (run, agit) in enumerate(RUNS):
        sid = f"{SUB}_{run}"
        raw = load_canonical(RAW / SUB / "anat"
                             / f"{SUB}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii")
        n4 = load_canonical(PRE / sid / f"{sid}_n4.nii.gz")
        rigid = load_canonical(PRE / sid / f"{sid}_rigid.nii.gz")
        brain = load_canonical(PRE / sid / f"{sid}_brain.nii.gz")

        for col, vol in enumerate([raw, n4, rigid, brain]):
            sl = axial(vol)
            lo, hi = window(sl)
            axes[row, col].imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{run}\nAgitation {agit}", fontsize=10)

    for c, t in enumerate(["brut", "N4 (biais corrigé)", "recalage rigide MNI", "cerveau (SynthStrip)"]):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle(f"Preprocessing rigide — {SUB}, coupe axiale", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
