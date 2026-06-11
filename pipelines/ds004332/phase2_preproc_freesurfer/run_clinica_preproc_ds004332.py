"""
Préprocessing équivalent à Clinica t1-linear pour ds004332.
Pipeline : N4BiasFieldCorrection → registration affine vers MNI → crop 160×192×160 → normalisation [0,1].

Cible : acq-mpragepmcoff_rec-wore run-01/02/03 (22 sujets).
Sortie : ~/Documents/derivatives/ds004332/clinica_preproc/<sub>/anat/<sub>_run-XX_clinica.nii.gz

Usage :
    conda run -n cortical-motion python3 \
        ~/Documents/motion-analysis/pipelines/ds004332/run_clinica_preproc_ds004332.py
"""

import os
import numpy as np
import nibabel as nib
import ants
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DIR   = Path.home() / "Documents/raw/ds004332"
OUT_DIR   = Path.home() / "Documents/derivatives/ds004332/clinica_preproc"
MNI_PATH  = str(Path.home() / ".antspy/mni.nii.gz")
N_JOBS    = min(6, os.cpu_count())

SUBS = sorted([d.name for d in RAW_DIR.glob("sub-*/")])
RUNS = ["run-01", "run-02", "run-03"]
ACQ  = "acq-mpragepmcoff_rec-wore"

# Clinica t1-linear default crop : 182×218×182 → 160×192×160
CROP = (slice(11, 171), slice(13, 205), slice(11, 171))


def _cropped_affine(mni_affine: np.ndarray, crop_start=(11, 13, 11)) -> np.ndarray:
    """Affine correct pour l'image croppée : origin += R @ crop_start."""
    aff = mni_affine.copy()
    aff[:3, 3] += aff[:3, :3] @ np.array(crop_start, dtype=float)
    return aff


def preprocess_one(sub, run):
    raw_file = RAW_DIR / sub / "anat" / f"{sub}_{ACQ}_{run}_T1w.nii"
    if not raw_file.exists():
        return sub, run, "missing"

    out_anat = OUT_DIR / sub / "anat"
    out_anat.mkdir(parents=True, exist_ok=True)
    out_file = out_anat / f"{sub}_{run}_clinica.nii.gz"

    if out_file.exists():
        return sub, run, "skip"

    try:
        mni = ants.image_read(MNI_PATH)
        img = ants.image_read(str(raw_file))

        # N4 bias field correction
        img_n4 = ants.n4_bias_field_correction(img)

        # Affine registration vers MNI
        reg = ants.registration(
            fixed=mni,
            moving=img_n4,
            type_of_transform="Affine",
            random_seed=42,
        )
        warped = reg["warpedmovout"]  # 182×218×182, 1mm isotrope

        # Numpy : crop + normalisation
        data = warped.numpy().astype(np.float32)
        data = data[CROP]  # 160×192×160

        data = np.clip(data, 0, None)
        brain = data[data > 0]
        vmax = float(np.percentile(brain, 99.9)) if brain.size > 0 else float(data.max())
        if vmax > 1e-6:
            data = np.clip(data / vmax, 0.0, 1.0)

        # Affine croppé (MNI affine + offset du crop)
        mni_nib = nib.load(MNI_PATH)
        aff_crop = _cropped_affine(mni_nib.affine)

        nib.save(nib.Nifti1Image(data, affine=aff_crop), str(out_file))
        return sub, run, "ok"

    except Exception as e:
        return sub, run, f"error: {e}"


def _worker(args):
    return preprocess_one(*args)


if __name__ == "__main__":
    tasks = [(sub, run) for sub in SUBS for run in RUNS]
    print(f"Preprocessing {len(tasks)} images ({len(SUBS)} sujets × {len(RUNS)} runs)")
    print(f"N4 + affine MNI (antspyx) + crop 160×192×160 + normalisation [0,1]")
    print(f"Jobs parallèles : {N_JOBS}")

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=N_JOBS) as exe:
        futures = {exe.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            sub, run, status = fut.result()
            if status == "ok":
                done += 1
                print(f"  ✓ {sub} {run}")
            elif status == "skip":
                skipped += 1
                print(f"  ~ {sub} {run} (existe déjà)")
            elif status == "missing":
                print(f"  - {sub} {run} (fichier absent)")
            else:
                errors += 1
                print(f"  ✗ {sub} {run} : {status}")

    print(f"\nRésumé : {done} ok, {skipped} skip, {errors} erreurs")
    print(f"Sortie : {OUT_DIR}")
