#!/usr/bin/env python3
"""Montage FreeSurfer à fenêtre fixe et contours aseg pour la zone centrale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_ROOTS = {
    "preproc": "~/projects/ctb-sbouix/mathw/freesurfer_preproc_rigid_ds004332",
    "jdac": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_ds004332",
    "jdac_antiartonly": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_antiartonly_ds004332",
    "jdac_nodenoise": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_nodenoise_ds004332",
}
CENTRAL_LABELS = {
    4: "Left-Lateral-Ventricle",
    10: "Left-Thalamus-Proper",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    43: "Right-Lateral-Ventricle",
    49: "Right-Thalamus-Proper",
}
COLORS = {4:"#00bcd4", 10:"#ff9800", 14:"#e91e63", 15:"#9c27b0", 43:"#00bcd4", 49:"#ff9800"}


def root_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("format: condition=/chemin/SUBJECTS_DIR")
    condition, path = value.split("=", 1)
    if condition not in DEFAULT_ROOTS:
        raise argparse.ArgumentTypeError(f"condition inconnue: {condition}")
    return condition, Path(path).expanduser()


def load(path: Path) -> np.ndarray:
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-19_run-01")
    parser.add_argument("--root", action="append", default=[], type=root_arg)
    parser.add_argument("--slice-fraction", type=float, default=0.50)
    parser.add_argument("--voxel", nargs=3, type=int, metavar=("I", "J", "K"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/ds004332/phase5_fidelity/central_structure_sub19_run01.png"))
    args = parser.parse_args()

    roots = {name: Path(path).expanduser() for name, path in DEFAULT_ROOTS.items()}
    roots.update(dict(args.root))
    volumes: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for condition, root in roots.items():
        mri = root / args.subject / "mri"
        t1, aseg = mri / "T1.mgz", mri / "aseg.mgz"
        if not t1.exists() or not aseg.exists():
            raise FileNotFoundError(f"{condition}: fichiers absents dans {mri}")
        volumes[condition] = load(t1)
        labels[condition] = load(aseg).astype(np.int16)

    shapes = {v.shape for v in volumes.values()} | {v.shape for v in labels.values()}
    if len(shapes) != 1:
        raise ValueError(f"grilles incompatibles: {sorted(shapes)}")

    mask = np.logical_or.reduce([v > 0 for v in volumes.values()])
    zs = np.where(mask.any(axis=(0, 1)))[0]
    k = int(round(zs.min() + args.slice_fraction * (zs.max() - zs.min())))
    if args.voxel:
        k = args.voxel[2]

    rows = np.where(mask[:, :, k].any(axis=1))[0]
    cols = np.where(mask[:, :, k].any(axis=0))[0]
    margin = 8
    r0, r1 = max(rows.min()-margin, 0), min(rows.max()+margin, mask.shape[0]-1)
    c0, c1 = max(cols.min()-margin, 0), min(cols.max()+margin, mask.shape[1]-1)

    samples = np.concatenate([v[v > 0][::100] for v in volumes.values()])
    vmin, vmax = np.percentile(samples, [1.0, 99.5])
    fig, axes = plt.subplots(1, len(volumes), figsize=(4.2*len(volumes), 4.8), squeeze=False)
    for ax, (condition, volume) in zip(axes[0], volumes.items()):
        image = volume[r0:r1+1, c0:c1+1, k].T[:, ::-1]
        seg = labels[condition][r0:r1+1, c0:c1+1, k].T[:, ::-1]
        ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        for label_id, color in COLORS.items():
            ax.contour(seg == label_id, levels=[0.5], colors=[color], linewidths=0.8)
        if args.voxel:
            i, j, _ = args.voxel
            ax.plot((r1-i), (j-c0), marker="+", color="lime", ms=12, mew=2)
            found = int(labels[condition][i, j, k])
            print(f"{condition}: voxel {i},{j},{k} -> {found} {CENTRAL_LABELS.get(found, 'autre label')}")
        ax.set_title(condition)
        ax.axis("off")

    legend = "orange: thalamus | rose: 3e ventricule | cyan: ventricules latéraux"
    fig.suptitle(f"{args.subject}, coupe axiale k={k}, fenêtre commune [{vmin:.1f}, {vmax:.1f}]\n{legend}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Figure: {args.out}")


if __name__ == "__main__":
    main()
