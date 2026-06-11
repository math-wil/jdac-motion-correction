"""
Dénormalise les sorties JDAC ds004332 run-02/03, lance Agitation sur
les 5 conditions et produit un CSV de comparaison.

Conditions :
  raw_run-01      image brute run-01 (référence)
  raw_run-02      image brute run-02 (avec mouvement)
  raw_run-03      image brute run-03 (avec mouvement)
  jdac_run-02     sortie JDAC run-02 dénormalisée
  jdac_run-03     sortie JDAC run-03 dénormalisée

Usage :
    cd ~/Documents/agitation
    conda run -n cortical-motion python3 \
        ~/Documents/motion-analysis/pipelines/ds004332/run_agitation_jdac_comparison.py
"""

import subprocess
import tempfile
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

AGITATION_CLI = Path.home() / "Documents/agitation/cli.py"
RAW_DIR       = Path.home() / "Documents/raw/ds004332"
JDAC_READY    = Path.home() / "Documents/derivatives/ds004332/jdac_ready"
JDAC_OUT      = Path.home() / "Documents/derivatives/ds004332/jdac_outputs"
DENORM_DIR    = Path.home() / "Documents/derivatives/ds004332/jdac_outputs_denorm"
OUTPUT_CSV    = Path.home() / "Documents/motion-analysis/datasets/ds004332/results/ds004332_agitation_jdac_comparison.csv"

SUBS = sorted([d.name for d in JDAC_READY.iterdir() if d.name.startswith("sub-")])
RAW_PATTERN = "{sub}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii"


def compute_vmax(raw_path, mask_path):
    raw  = nib.load(raw_path).get_fdata(dtype=np.float32)
    mask = nib.load(mask_path).get_fdata(dtype=np.float32)
    # Le masque est dans l'espace run-01 (FLIRT), approximation acceptable
    # pour des mouvements rigides de faible amplitude
    brain = raw[mask > 0.5]
    brain = brain[brain > 0]
    if brain.size == 0:
        return float(raw.max())
    return float(np.percentile(brain, 99.9))


def denormalize(sub, run):
    raw_path  = RAW_DIR / sub / "anat" / RAW_PATTERN.format(sub=sub, run=run)
    mask_path = JDAC_READY / sub / "anat" / f"{sub}_brain_mask.nii.gz"
    jdac_path = JDAC_OUT / f"run-0{run[-1]}" / sub / f"{sub}_run-0{run[-1]}_jdac.nii.gz"
    out_path  = DENORM_DIR / sub / f"{sub}_{run}_jdac_denorm.nii.gz"

    if not raw_path.exists():
        print(f"  MANQUANT brut : {raw_path.name}")
        return None
    if not jdac_path.exists():
        print(f"  MANQUANT JDAC : {jdac_path.name}")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmax = compute_vmax(raw_path, mask_path)

    jdac_img = nib.load(jdac_path)
    data = jdac_img.get_fdata(dtype=np.float32) * vmax
    nib.save(nib.Nifti1Image(data, jdac_img.affine, jdac_img.header), out_path)
    return out_path


def run_agitation(condition, entries):
    print(f"\n── {condition} ({len(entries)} sujets) ──")
    entries = [(sub, path) for sub, path in entries if Path(path).exists()]
    if not entries:
        print("  Aucun fichier trouvé.")
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sub,motion,stratum,data\n")
        for sub, path in entries:
            f.write(f"{sub}_{condition},0.0,{condition},{path}\n")
        input_csv = f.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    result = subprocess.run(
        [sys.executable, str(AGITATION_CLI), "dataset", "-f", input_csv, "-g", "-o", output_csv],
        capture_output=True, text=True,
        cwd=str(AGITATION_CLI.parent)
    )
    if result.returncode != 0:
        print(f"  ERREUR : {result.stderr.strip()}")
        return None

    df = pd.read_csv(output_csv)
    df["condition"] = condition
    df["sub"] = df["sub"].str.replace(f"_{condition}$", "", regex=True)
    print(f"  ✓ mean={df['motion'].mean():.3f}  std={df['motion'].std():.3f}  "
          f"min={df['motion'].min():.3f}  max={df['motion'].max():.3f}")
    return df


# --- Dénormalisation ---
print("=== Dénormalisation JDAC run-02 et run-03 ===")
DENORM_DIR.mkdir(parents=True, exist_ok=True)

denorm_paths = {}
for run_label, run_key in [("run-02", "run-02"), ("run-03", "run-03")]:
    denorm_paths[run_label] = {}
    for sub in SUBS:
        path = denormalize(sub, run_label)
        if path:
            denorm_paths[run_label][sub] = str(path)
    print(f"  {run_label} : {len(denorm_paths[run_label])}/22 dénormalisés")

# --- Agitation ---
print("\n=== Agitation ===")

conditions = {
    "raw_run-01": [
        (sub, str(RAW_DIR / sub / "anat" / RAW_PATTERN.format(sub=sub, run="run-01")))
        for sub in SUBS
    ],
    "raw_run-02": [
        (sub, str(RAW_DIR / sub / "anat" / RAW_PATTERN.format(sub=sub, run="run-02")))
        for sub in SUBS
    ],
    "raw_run-03": [
        (sub, str(RAW_DIR / sub / "anat" / RAW_PATTERN.format(sub=sub, run="run-03")))
        for sub in SUBS
    ],
    "jdac_run-02": [
        (sub, denorm_paths["run-02"].get(sub, ""))
        for sub in SUBS
    ],
    "jdac_run-03": [
        (sub, denorm_paths["run-03"].get(sub, ""))
        for sub in SUBS
    ],
}

results = []
for condition, entries in conditions.items():
    df = run_agitation(condition, entries)
    if df is not None:
        results.append(df)

if not results:
    print("Aucun résultat.")
    sys.exit(1)

combined = pd.concat(results, ignore_index=True)[["condition", "sub", "motion"]]
combined = combined.sort_values(["sub", "condition"]).reset_index(drop=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_CSV, index=False)

print(f"\n=== Résumé par condition ===")
print(combined.groupby("condition")["motion"].agg(["mean","std","min","max"]).round(3).to_string())
print(f"\nCSV : {OUTPUT_CSV}")
