#!/usr/bin/env python3
"""Compare les trois runs de sub-19 sur les quatre conditions d'image (rigide).

Lignes = même cerveau, mouvement croissant. Colonnes = entrée preproc puis les
trois sorties JDAC. Toutes les images partagent la grille rigide.

Réglages de rendu (montage type réunion) :
- coupe axiale choisie sur l'ÉTENDUE du cerveau (pas sur la grille paddée), pour
  tomber au milieu du cerveau et pas trop haut ;
- recadrage sur la boîte englobante commune du cerveau (images resserrées) ;
- panneaux grands, fenêtrage percentile par panneau.
Les intensités affichées sont normalisées par panneau pour la visualisation ;
les comparaisons quantitatives restent en Phase 4.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


HOME = Path.home()
DERIV = HOME / "Documents/derivatives/ds004332"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/ds004332/phase3_JDAC/sub19_three_runs_four_conditions.png"

RUNS = [
    ("run-01", "consigne immobile", 0.20573646),
    ("run-02", "nodding léger", 0.44703093),
    ("run-03", "shaking sévère", 3.152689),
]
CONDITIONS = [
    ("preproc", "entrée preproc", "preproc_rigid/{sid}/{sid}_brain.nii.gz"),
    ("jdac", "JDAC complet", "jdac_rigid/{sid}/{sid}_T1w_jdac.nii.gz"),
    ("antiartonly", "anti-artefact 1×", "jdac_rigid_antiartonly/{sid}/{sid}_T1w_jdac_antiartonly.nii.gz"),
    ("nodenoise", "sans débruiteur 4×", "jdac_rigid_nodenoise/{sid}/{sid}_T1w_jdac_nodenoise.nii.gz"),
]

Z_FRACTION = 0.50   # position de la coupe le long de l'étendue du cerveau
MARGIN = 6          # voxels de marge autour du cerveau pour le recadrage


def load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata().astype(np.float32)


def common_slice_and_box(volumes):
    """Coupe axiale (milieu du cerveau) et boîte englobante 2D communes à tous les volumes."""
    mask = np.zeros(volumes[0].shape, dtype=bool)
    for v in volumes:
        mask |= v > 0
    zs = np.where(mask.any(axis=(0, 1)))[0]
    k = int(round(zs.min() + Z_FRACTION * (zs.max() - zs.min())))
    sl = mask[:, :, k]
    rows = np.where(sl.any(axis=1))[0]
    cols = np.where(sl.any(axis=0))[0]
    r0, r1 = max(rows.min() - MARGIN, 0), min(rows.max() + MARGIN, sl.shape[0] - 1)
    c0, c1 = max(cols.min() - MARGIN, 0), min(cols.max() + MARGIN, sl.shape[1] - 1)
    return k, (r0, r1, c0, c1)


def panel(volume, k, box):
    """Coupe recadrée, orientation FSLeyes : antérieur en haut, droite du patient à gauche (radiologique)."""
    r0, r1, c0, c1 = box
    return volume[r0:r1 + 1, c0:c1 + 1, k].T[:, ::-1]


def main() -> None:
    grid = {}
    for run, _, _ in RUNS:
        sid = f"sub-19_{run}"
        for cond, _, template in CONDITIONS:
            grid[(run, cond)] = load(DERIV / template.format(sid=sid))
    k, box = common_slice_and_box(list(grid.values()))

    nrow, ncol = len(RUNS), len(CONDITIONS)
    r0, r1, c0, c1 = box
    # aspect d'AFFICHAGE (après .T) = hauteur/largeur = étendue A / étendue R
    aspect = (c1 - c0) / (r1 - r0)
    panel_w = 3.6
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_w * aspect * nrow))
    for r, (run, label, agitation) in enumerate(RUNS):
        for c, (cond, title, _) in enumerate(CONDITIONS):
            vol = grid[(run, cond)]
            axes[r, c].imshow(panel(vol, k, box), cmap="gray", origin="lower",
                              vmin=0.0, vmax=float(vol.max()))   # rendu FSLeyes : 0 -> max
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=13)
        axes[r, 0].set_ylabel(f"sub-19 {run}\n{label}\nAgitation {agitation:.2f}", fontsize=11)

    # panneaux collés (aucun espace), marge à gauche pour les libellés de ligne
    fig.subplots_adjust(left=0.055, right=0.999, top=0.965, bottom=0.005, wspace=0.0, hspace=0.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Figure -> {OUT}  (coupe z={k})")


if __name__ == "__main__":
    main()
