#!/usr/bin/env python3
"""
Montage QC JDAC sur le MEME echantillon de 8 que le preproc (view_preproc_sample.sh) :
sub-01 (3 runs, ancrage M1) + les 4 plus forts mouvements + 1 controle median.

Une ligne par sujet, deux colonnes :
  - cerveau preproc (N4 + SynthStrip) = ENTREE de JDAC
  - sortie JDAC (jdac_fixed)           = APRES correction
Coupe sagittale mediane. On voit directement le geste de JDAC, sujet par sujet,
du plus fort mouvement au plus faible.

Sortie : results/ds004332/phase3_JDAC/montage_jdac.png

Usage (env conda cortical-motion) :
    python montage_jdac.py
"""
from pathlib import Path

import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
PRE_DIR = HOME / "Documents/derivatives/ds004332/preproc"
JDAC_DIR = HOME / "Documents/derivatives/ds004332/jdac_fixed"
OUT_PNG = (HOME / "Documents/jdac-motion-correction/results/ds004332/"
           "phase3_JDAC/montage_jdac.png")

# Identique a view_preproc_sample.sh : id:motion, du plus fort mouvement au plus faible
SAMPLE = [
    ("sub-11_run-03", 3.29),
    ("sub-03_run-02", 3.25),
    ("sub-14_run-03", 3.23),
    ("sub-07_run-03", 3.18),
    ("sub-01_run-03", 3.16),
    ("sub-20_run-03", 0.71),
    ("sub-01_run-02", 0.26),
    ("sub-01_run-01", 0.20),
]


def sagittal(path):
    """Coupe sagittale mediane, en RAS canonique (S en haut, A a gauche)."""
    vol = nib.as_closest_canonical(nib.load(str(path))).get_fdata()
    i = vol.shape[0] // 2
    return vol[i, :, :].T   # origin='lower'


def main():
    n = len(SAMPLE)
    fig, axes = plt.subplots(n, 2, figsize=(2 * 2.6, n * 2.6))

    for row, (sid, mv) in enumerate(SAMPLE):
        brain = PRE_DIR / sid / f"{sid}_brain.nii.gz"
        jdac = JDAC_DIR / sid / f"{sid}_T1w_jdac.nii.gz"
        for col, (path, tag) in enumerate([(brain, "preproc"), (jdac, "JDAC")]):
            ax = axes[row, col]
            if path.is_file():
                ax.imshow(sagittal(path), cmap="gray", origin="lower")
            else:
                ax.text(0.5, 0.5, "MANQUE", color="red", ha="center", va="center")
            if col == 0:
                ax.set_ylabel(f"{sid}\nmotion={mv:.2f}", fontsize=8)
            if row == 0:
                ax.set_title(tag, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("QC JDAC : preproc (entree) vs JDAC (sortie) "
                 "-- echantillon des 8, mouvement decroissant", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=120)
    print(f"Montage ({n} sujets) -> {OUT_PNG}")


if __name__ == "__main__":
    main()
