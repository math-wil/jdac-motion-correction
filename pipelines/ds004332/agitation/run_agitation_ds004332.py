"""
Lance Agitation sur les 5 conditions ds004332 et produit un CSV unique.

Conditions :
  run-01       jdac_ready run-01 (référence, sans mouvement)
  run-02       jdac_ready run-02 (avant JDAC)
  run-03       jdac_ready run-03 (avant JDAC)
  run-02_jdac  jdac_outputs run-02 (après JDAC)
  run-03_jdac  jdac_outputs run-03 (après JDAC)

Usage :
    cd ~/Documents/agitation
    conda run -n cortical-motion python3 \
        ~/Documents/motion-analysis/pipelines/ds004332/run_agitation_ds004332.py
"""

import subprocess
import tempfile
import sys
from pathlib import Path
import pandas as pd

AGITATION_CLI = Path.home() / "Documents/agitation/cli.py"
JDAC_READY    = Path.home() / "Documents/derivatives/ds004332/jdac_ready"
JDAC_OUT      = Path.home() / "Documents/derivatives/ds004332/jdac_outputs"
OUTPUT_CSV    = Path.home() / "Documents/motion-analysis/datasets/ds004332/results/ds004332_agitation_5conditions.csv"

SUBS = sorted([d.name for d in JDAC_READY.iterdir() if d.name.startswith("sub-")])

CONDITIONS = {
    "run-01": [
        (sub, str(JDAC_READY / sub / "anat" / f"{sub}_run-01_brain_norm01.nii.gz"))
        for sub in SUBS
    ],
    "run-02": [
        (sub, str(JDAC_READY / sub / "anat" / f"{sub}_run-02_brain_norm01.nii.gz"))
        for sub in SUBS
    ],
    "run-03": [
        (sub, str(JDAC_READY / sub / "anat" / f"{sub}_run-03_brain_norm01.nii.gz"))
        for sub in SUBS
    ],
    "run-02_jdac": [
        (sub, str(JDAC_OUT / "run-02" / sub / f"{sub}_run-02_jdac.nii.gz"))
        for sub in SUBS
    ],
    "run-03_jdac": [
        (sub, str(JDAC_OUT / "run-03" / sub / f"{sub}_run-03_jdac.nii.gz"))
        for sub in SUBS
    ],
}


def run_condition(condition, entries):
    print(f"\n── {condition} ({len(entries)} sujets) ──")

    missing = [path for _, path in entries if not Path(path).exists()]
    if missing:
        print(f"  AVERTISSEMENT : {len(missing)} fichiers manquants, ignorés.")
        entries = [(sub, path) for sub, path in entries if Path(path).exists()]

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
        print(f"  ERREUR :\n{result.stderr.strip()}")
        return None

    df = pd.read_csv(output_csv)
    df["condition"] = condition
    df["sub_clean"] = df["sub"].str.replace(f"_{condition}$", "", regex=True)
    print(f"  ✓ {len(df)} scores (mean={df['motion'].mean():.3f})")
    return df


all_results = []
for condition, entries in CONDITIONS.items():
    df = run_condition(condition, entries)
    if df is not None:
        all_results.append(df)

if not all_results:
    print("Aucun résultat produit.")
    sys.exit(1)

combined = pd.concat(all_results, ignore_index=True)
combined = combined.rename(columns={"sub_clean": "sub_id"})[
    ["condition", "sub_id", "motion"]
]
combined = combined.sort_values(["sub_id", "condition"]).reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV combiné : {OUTPUT_CSV}")
print(combined.groupby("condition")["motion"].describe().round(3))
