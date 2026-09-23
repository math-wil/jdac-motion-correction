#!/usr/bin/env python3
"""Pilote séparé : SynthStrip seul, puis JDAC, sur neuf T1w ds004332.

Sans --stage strip ou --stage jdac, aucun traitement n'est lancé. Les sorties
historiques ne sont jamais modifiées. Voir le README de ce dossier avant usage.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SUBJECTS = ("sub-17", "sub-13", "sub-14")
RUNS = ("run-01", "run-02", "run-03")
WEIGHTS = (
    "Pretrained_Denoiser_l1loss_epoch15.pth",
    "Pretrained_AntiArtNet_l1loss_epoch150.pth",
)
REPO = Path(__file__).resolve().parents[3]


def acquisitions(raw_root):
    for sub in SUBJECTS:
        for run in RUNS:
            sid = f"{sub}_{run}"
            raw = raw_root / sub / "anat" / f"{sub}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii"
            yield sid, raw


def paths(out_root, sid):
    pre = out_root / "preproc_minimal" / sid
    corrected = out_root / "jdac_minimal" / sid
    return (
        pre / f"{sid}_brain.nii.gz",
        pre / f"{sid}_mask.nii.gz",
        corrected / f"{sid}_T1w_jdac_minimal.nii.gz",
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_grid(first, second):
    import nibabel as nib
    import numpy as np

    a = nib.load(str(first))
    b = nib.load(str(second))
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-5, rtol=0):
        raise RuntimeError(f"Grilles différentes : {first} / {second}")
    return {"shape": list(a.shape), "affine": a.affine.tolist()}


def make_mask_qc(items, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    fig, axes = plt.subplots(len(items), 5, figsize=(19, 29))
    records = []
    for row, (sid, raw) in enumerate(items):
        brain, mask_path, _ = paths(out_root, sid)
        if not brain.is_file() or not mask_path.is_file():
            raise RuntimeError(f"SynthStrip incomplet pour {sid}")
        same_grid(raw, brain)
        same_grid(raw, mask_path)
        image = nib.as_closest_canonical(nib.load(str(raw)))
        mask_image = nib.as_closest_canonical(nib.load(str(mask_path)))
        data = image.get_fdata(dtype=np.float32)
        mask = mask_image.get_fdata() > 0.5
        if not mask.any():
            raise RuntimeError(f"Masque vide pour {sid}")
        low, high = np.percentile(data[mask], (1, 99))
        if high <= low:
            raise RuntimeError(f"Intensités constantes pour {sid}")
        volume_mm3 = float(mask.sum() * abs(np.linalg.det(image.affine[:3, :3])))
        extent = np.array(np.nonzero(mask))
        lo, hi = extent.min(axis=1), extent.max(axis=1)
        xs = [int(lo[0] + f * (hi[0] - lo[0])) for f in (0.35, 0.65)]
        ys = [int((lo[1] + hi[1]) / 2)]
        zs = [int(lo[2] + f * (hi[2] - lo[2])) for f in (0.25, 0.65)]
        slices = [(0, x, f"sag {x}") for x in xs]
        slices += [(1, y, f"cor {y}") for y in ys]
        slices += [(2, z, f"ax {z}") for z in zs]
        for col, (axis, index, label) in enumerate(slices):
            raw_slice = np.rot90(np.take(data, index, axis=axis))
            mask_slice = np.rot90(np.take(mask, index, axis=axis))
            ax = axes[row, col]
            ax.imshow(np.clip((raw_slice - low) / (high - low), 0, 1),
                      cmap="gray", vmin=0, vmax=1)
            if mask_slice.any() and not mask_slice.all():
                ax.contour(mask_slice.astype(float), levels=[0.5], colors="red", linewidths=0.7)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        axes[row, 0].text(-0.07, 0.5, sid, va="center", ha="right",
                          rotation=90, transform=axes[row, 0].transAxes, fontsize=9)
        records.append({"id": sid, "raw": str(raw), "mask": str(mask_path),
                        "mask_volume_mm3": round(volume_mm3, 1)})
        print(f"{sid} : volume du masque = {volume_mm3:.0f} mm³")
    fig.tight_layout()
    output = out_root / "mask_qc.png"
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"Montage à inspecter : {output}")
    return records


def make_jdac_qc(items, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    fig, axes = plt.subplots(len(items), 6, figsize=(22, 29))
    records = []
    for row, (sid, _) in enumerate(items):
        brain, mask_path, corrected_path = paths(out_root, sid)
        if not all(path.is_file() for path in (brain, mask_path, corrected_path)):
            raise RuntimeError(f"Paire JDAC incomplète pour {sid}")
        same_grid(brain, corrected_path)
        same_grid(brain, mask_path)
        before = nib.as_closest_canonical(nib.load(str(brain))).get_fdata(dtype=np.float32)
        after = nib.as_closest_canonical(nib.load(str(corrected_path))).get_fdata(dtype=np.float32)
        mask = nib.as_closest_canonical(nib.load(str(mask_path))).get_fdata() > 0.5
        if not np.isfinite(after).all():
            raise RuntimeError(f"Valeurs non finies dans la sortie JDAC pour {sid}")
        foreground = np.array(np.nonzero(before > 0.01))
        lo, hi = foreground.min(axis=1), foreground.max(axis=1) + 1
        cropped = before[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        p0, p98 = np.percentile(cropped, (0, 98))
        if p98 <= p0:
            raise RuntimeError(f"Plage d'intensité invalide pour {sid}")
        before_scaled = np.clip((before - p0) / (p98 - p0), 0, 1)
        center = [int((a + b) / 2) for a, b in zip(lo, hi)]
        for col, (volume, name) in enumerate(((before_scaled, "entrée"), (after, "JDAC"))):
            for axis, plane in enumerate(("sag", "cor", "ax")):
                ax = axes[row, 3 * col + axis]
                ax.imshow(np.rot90(np.take(volume, center[axis], axis=axis)),
                          cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{name} {plane}", fontsize=8)
                ax.axis("off")
        axes[row, 0].text(-0.07, 0.5, sid, va="center", ha="right",
                          rotation=90, transform=axes[row, 0].transAxes, fontsize=9)
        mean_abs_change = float(np.mean(np.abs(after[mask] - before_scaled[mask])))
        records.append({"id": sid, "brain": str(brain), "corrected": str(corrected_path),
                        "mean_abs_change_0_1": round(mean_abs_change, 6),
                        "corrected_min": float(after.min()), "corrected_max": float(after.max())})
        print(f"{sid} : variation absolue moyenne = {mean_abs_change:.4f} sur [0,1]")
    fig.tight_layout()
    output = out_root / "jdac_qc.png"
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"Montage à inspecter : {output}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "strip", "qc", "jdac", "qc-jdac"), default="check")
    parser.add_argument("--raw-root", type=Path, default=Path.home() / "Documents/raw_datasets/ds004332")
    parser.add_argument("--out-root", type=Path, default=Path.home() / "Documents/derivatives/ds004332/jdac_minimal_pilot")
    parser.add_argument("--jdac-root", type=Path, default=Path.home() / "Documents/jdac")
    parser.add_argument("--freesurfer-home", type=Path, default=Path(os.environ.get(
        "FREESURFER_HOME", "/project/hippocampus/common/softwares/freesurfer")))
    parser.add_argument("--mask-reviewed", action="store_true",
                        help="Requis pour JDAC après inspection visuelle des neuf masques")
    args = parser.parse_args()

    items = list(acquisitions(args.raw_root))
    missing_raw = [str(raw) for _, raw in items if not raw.is_file()]
    if missing_raw:
        parser.error("T1w absents :\n" + "\n".join(missing_raw))

    synthstrip = args.freesurfer_home / "bin/mri_synthstrip"
    if args.stage in ("check", "strip") and not synthstrip.is_file():
        parser.error(f"SynthStrip absent : {synthstrip}")

    weight_paths = [args.jdac_root / "PretrainedModels" / name for name in WEIGHTS]
    if args.stage in ("check", "jdac"):
        missing_weights = [str(path) for path in weight_paths if not path.is_file()]
        if missing_weights:
            parser.error("Poids JDAC absents :\n" + "\n".join(missing_weights))
        if not (args.jdac_root / "models.py").is_file():
            parser.error(f"Code JDAC absent : {args.jdac_root / 'models.py'}")

    print(f"{len(items)} T1w trouvés : {', '.join(sid for sid, _ in items)}")
    print(f"Sorties isolées : {args.out_root}")
    if args.stage == "check":
        old_pre = Path.home() / "Documents/derivatives/ds004332/preproc_rigid"
        old_jdac = Path.home() / "Documents/derivatives/ds004332/jdac_rigid"
        old_pairs = sum(
            (old_pre / sid / f"{sid}_brain.nii.gz").is_file()
            and (old_jdac / sid / f"{sid}_T1w_jdac.nii.gz").is_file()
            for sid, _ in items
        )
        print(f"Anciennes paires d'images disponibles : {old_pairs}/{len(items)}")
        print("Contrôle en lecture seule terminé ; aucun traitement lancé.")
        return

    records = []
    if args.stage == "strip":
        for sid, raw in items:
            brain, mask, _ = paths(args.out_root, sid)
            if brain.exists() or mask.exists():
                if not (brain.exists() and mask.exists()):
                    raise RuntimeError(f"Sortie SynthStrip incomplète pour {sid} ; vérifier avant reprise")
                status = "déjà présent"
            else:
                brain.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([str(synthstrip), "-i", str(raw), "-o", str(brain),
                                "-m", str(mask)], check=True,
                               env={**os.environ, "FREESURFER_HOME": str(args.freesurfer_home)})
                status = "créé"
            grid = same_grid(raw, brain)
            same_grid(raw, mask)
            records.append({"id": sid, "raw": str(raw), "brain": str(brain),
                            "mask": str(mask), "status": status, "grid": grid})
            print(f"{sid} : masque et cerveau {status}, grille native vérifiée")
        print("ARRÊT : inspecter visuellement les neuf masques avant --stage jdac.")

    if args.stage == "qc":
        records = make_mask_qc(items, args.out_root)
        print("Ce montage ne valide pas automatiquement les masques ; l'inspecter avant JDAC.")

    if args.stage == "qc-jdac":
        records = make_jdac_qc(items, args.out_root)
        print("Ce montage sert au contrôle technique, pas à juger la récupération anatomique.")

    if args.stage == "jdac":
        if not args.mask_reviewed:
            parser.error("JDAC exige --mask-reviewed après inspection visuelle des neuf masques")
        for sid, raw in items:
            brain, mask, _ = paths(args.out_root, sid)
            if not brain.is_file() or not mask.is_file():
                parser.error(f"SynthStrip incomplet pour {sid} ; exécuter --stage strip")
            same_grid(raw, brain)
            same_grid(raw, mask)

        sys.path.insert(0, str(REPO / "pipelines/ds004332/phase3_JDAC"))
        os.chdir(args.jdac_root)  # Le script historique charge ./PretrainedModels.
        from run_jdac import load_models, process_subject

        denoiser, antiart = load_models()
        for sid, raw in items:
            brain, mask, corrected = paths(args.out_root, sid)
            result = process_subject(sid, brain, args.out_root / "jdac_minimal",
                                     denoiser, antiart, output_tag="jdac_minimal")
            if result["status"] not in ("ok", "skipped") or not corrected.is_file():
                raise RuntimeError(f"Inférence incomplète pour {sid} : {result}")
            grid = same_grid(brain, corrected)
            records.append({"id": sid, "raw": str(raw), "brain": str(brain),
                            "corrected": str(corrected), "status": result["status"],
                            "grid": grid})
            print(f"{sid} : JDAC {result['status']}, grille native vérifiée")

    manifest = {
        "stage": args.stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": list(SUBJECTS),
        "runs": list(RUNS),
        "records": records,
        "source_script": str(Path(__file__).resolve()),
    }
    if args.stage == "jdac":
        manifest["weight_sha256"] = {path.name: sha256(path) for path in weight_paths}
    path = args.out_root / f"{args.stage}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest : {path}")


if __name__ == "__main__":
    main()
