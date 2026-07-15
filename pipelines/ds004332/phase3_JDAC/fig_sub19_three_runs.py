#!/usr/bin/env python3
"""Compare the three sub-19 runs across the four rigid image conditions.

Rows are the same brain with increasing instructed motion. Columns are the
preprocessed JDAC input and the three JDAC outputs. All panels share the rigid
grid and use the same axial slice fraction. Display intensities are normalized
per panel only for visualization; quantitative comparisons remain in Phase 4.
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
    (
        "antiartonly",
        "anti-artefact 1×",
        "jdac_rigid_antiartonly/{sid}/{sid}_T1w_jdac_antiartonly.nii.gz",
    ),
    (
        "nodenoise",
        "sans débruiteur 4×",
        "jdac_rigid_nodenoise/{sid}/{sid}_T1w_jdac_nodenoise.nii.gz",
    ),
]


def load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata().astype(np.float32)


def normalized_slice(volume: np.ndarray, fraction: float = 0.52) -> np.ndarray:
    index = int(round(volume.shape[2] * fraction))
    image = volume[:, :, index].T
    mask = image > 0
    if not mask.any():
        return image
    low, high = np.percentile(image[mask], [1, 99.5])
    scaled = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    scaled[~mask] = 0
    return scaled


def main() -> None:
    fig, axes = plt.subplots(len(RUNS), len(CONDITIONS), figsize=(15, 11))
    for row, (run, label, agitation) in enumerate(RUNS):
        sid = f"sub-19_{run}"
        for col, (_, title, template) in enumerate(CONDITIONS):
            path = DERIV / template.format(sid=sid)
            axes[row, col].imshow(normalized_slice(load(path)), cmap="gray", origin="lower")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
        axes[row, 0].set_ylabel(
            f"sub-19 {run}\n{label}\nAgitation {agitation:.3f}", fontsize=10
        )
    fig.suptitle(
        "Même cerveau, mouvement croissant : entrée et variantes JDAC",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Figure -> {OUT}")


if __name__ == "__main__":
    main()
