#!/usr/bin/env python3
"""
Montage QC du skull-strip sur les 66 images preprocessees.

Pour chaque image : coupe mediane du _n4 (niveaux de gris) avec le CONTOUR du
masque _mask superpose en rouge. Tuiles triees par mouvement DECROISSANT
(les cas les plus a risque -- forts mouvements -- en premier) pour reperer
d'un coup d'oeil un skull-strip rate (cortex rogne ou crane/dure-mere garde).

Sortie : results/ds004332/phase2_PREPROC/montage_preproc_mask.png

Usage (env conda cortical-motion) :
    python montage_preproc_mask.py
    python montage_preproc_mask.py --axis 2 --cols 6
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path.home()
PRE_DIR = HOME / "Documents/derivatives/ds004332/preproc"
AGIT_CSV = (HOME / "Documents/jdac-motion-correction/results/ds004332/"
            "agitation/ds004332_agitation_clinica.csv")
OUT_PNG = (HOME / "Documents/jdac-motion-correction/results/ds004332/"
           "phase2_PREPROC/montage_preproc_mask.png")


def load_motion():
    """Retourne {id: motion} a partir du CSV d'agitation (id = sub-XX_run-0Y)."""
    m = {}
    with open(AGIT_CSV) as f:
        for r in csv.DictReader(f):
            m[f"{r['sub']}_{r['condition']}"] = float(r["motion"])
    return m


def sagittal(path):
    """Coupe sagittale mediane, en RAS canonique (S en haut, A a gauche)."""
    vol = nib.as_closest_canonical(nib.load(str(path))).get_fdata()
    i = vol.shape[0] // 2
    return vol[i, :, :].T   # origin='lower' : S en haut, avant a droite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, default=2, help="axe de coupe (0/1/2)")
    ap.add_argument("--cols", type=int, default=6, help="nb de colonnes")
    args = ap.parse_args()

    motion = load_motion()
    ids = sorted(p.name for p in PRE_DIR.iterdir() if p.is_dir())
    # tri par mouvement decroissant (cas a risque en premier) ; inconnus a la fin
    ids.sort(key=lambda i: motion.get(i, -1.0), reverse=True)

    n = len(ids)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    n_missing = 0
    for ax, sid in zip(axes, ids):
        n4_path = PRE_DIR / sid / f"{sid}_n4.nii.gz"
        mask_path = PRE_DIR / sid / f"{sid}_mask.nii.gz"
        if not n4_path.is_file() or not mask_path.is_file():
            ax.set_title(f"{sid}\nMANQUE", fontsize=6, color="red")
            ax.axis("off")
            n_missing += 1
            continue
        n4 = sagittal(n4_path)
        mk = sagittal(mask_path)
        ax.imshow(n4, cmap="gray", origin="lower")
        ax.contour(mk > 0, levels=[0.5], colors="red", linewidths=0.6)
        mv = motion.get(sid)
        lbl = f"{sid}\nmotion={mv:.2f}" if mv is not None else f"{sid}\nmotion=?"
        ax.set_title(lbl, fontsize=6)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("QC skull-strip : contour du masque (rouge) sur _n4 "
                 "-- trie par mouvement decroissant", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=110)
    print(f"Montage ({n} images, {n_missing} manquantes) -> {OUT_PNG}")


if __name__ == "__main__":
    main()
