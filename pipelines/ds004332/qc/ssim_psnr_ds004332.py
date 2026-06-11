"""
SSIM et PSNR avant/après correction JDAC pour ds004332.

Référence   : run-01 (sans mouvement, preprocessé FLIRT)
Avant JDAC  : run-02 / run-03 preprocessés FLIRT (images avec mouvement)
Après JDAC  : sorties JDAC pour run-02 / run-03

Sortie : ~/Documents/derivatives/ds004332/results/ssim_psnr_ds004332.csv
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

JDAC_READY = Path.home() / "Documents/derivatives/ds004332/jdac_ready"
JDAC_OUT   = Path.home() / "Documents/derivatives/ds004332/jdac_outputs"
OUTPUT_CSV = Path.home() / "Documents/derivatives/ds004332/results/ssim_psnr_ds004332.csv"

SUBS = sorted([d.name for d in JDAC_READY.glob("sub-*/")])

records = []
for sub in SUBS:
    ref_path = JDAC_READY / sub / "anat" / f"{sub}_run-01_brain_norm01.nii.gz"
    if not ref_path.exists():
        print(f"[SKIP] {sub} : référence run-01 absente")
        continue
    ref = nib.load(ref_path).get_fdata(dtype=np.float32)

    for run in ("run-02", "run-03"):
        before_path = JDAC_READY / sub / "anat" / f"{sub}_{run}_brain_norm01.nii.gz"
        after_path  = JDAC_OUT / run / sub / f"{sub}_{run}_jdac.nii.gz"

        if not before_path.exists():
            print(f"[SKIP] {sub} {run} : image avant JDAC absente")
            continue
        if not after_path.exists():
            print(f"[SKIP] {sub} {run} : sortie JDAC absente")
            continue

        before = nib.load(before_path).get_fdata(dtype=np.float32)
        after  = nib.load(after_path).get_fdata(dtype=np.float32)

        if ref.shape != before.shape or ref.shape != after.shape:
            print(f"[SKIP] {sub} {run} : shapes incompatibles")
            continue

        ssim_before = ssim(ref, before, data_range=1.0)
        psnr_before = psnr(ref, before, data_range=1.0)
        ssim_after  = ssim(ref, after,  data_range=1.0)
        psnr_after  = psnr(ref, after,  data_range=1.0)

        print(f"  {sub} {run} : SSIM {ssim_before:.4f} -> {ssim_after:.4f}  |  PSNR {psnr_before:.2f} -> {psnr_after:.2f} dB")
        records.append({
            "sub": sub, "run": run,
            "ssim_before": round(ssim_before, 4), "ssim_after": round(ssim_after, 4),
            "psnr_before": round(psnr_before, 3), "psnr_after": round(psnr_after, 3),
        })

df = pd.DataFrame(records)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n=== Résumé ===")
print(df.groupby("run")[["ssim_before", "ssim_after", "psnr_before", "psnr_after"]].mean().round(3).to_string())
print(f"\nCSV : {OUTPUT_CSV}")
