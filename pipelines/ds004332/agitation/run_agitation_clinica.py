"""
Agitation sur images ds004332 préprocessées avec le pipeline équivalent Clinica t1-linear
(N4 + affine MNI + crop 160×192×160 + normalisation [0,1]).

Conditions : run-01 (référence), run-02 (mouvement modéré), run-03 (mouvement sévère).

Usage :
    conda run -n cortical-motion python3 \
        ~/Documents/motion-analysis/pipelines/ds004332/run_agitation_clinica.py
"""

import subprocess
import tempfile
import sys
from pathlib import Path
import pandas as pd

AGITATION_CLI = Path.home() / "Documents/agitation/cli.py"
CLINICA_DIR   = Path.home() / "Documents/derivatives/ds004332/clinica_preproc"
OUTPUT_CSV    = Path.home() / "Documents/motion-analysis/datasets/ds004332/results/ds004332_agitation_clinica.csv"

SUBS = sorted([d.name for d in CLINICA_DIR.glob("sub-*/")])

CONDITIONS = {
    "run-01": [(sub, str(CLINICA_DIR / sub / "anat" / f"{sub}_run-01_clinica.nii.gz")) for sub in SUBS],
    "run-02": [(sub, str(CLINICA_DIR / sub / "anat" / f"{sub}_run-02_clinica.nii.gz")) for sub in SUBS],
    "run-03": [(sub, str(CLINICA_DIR / sub / "anat" / f"{sub}_run-03_clinica.nii.gz")) for sub in SUBS],
}


def run_agitation(condition, entries):
    entries = [(sub, path) for sub, path in entries if Path(path).exists()]
    print(f"\n── {condition} ({len(entries)} sujets) ──")
    if not entries:
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


results = []
for condition, entries in CONDITIONS.items():
    df = run_agitation(condition, entries)
    if df is not None:
        results.append(df)

if not results:
    print("Aucun résultat — vérifie que le preprocessing est terminé.")
    sys.exit(1)

combined = pd.concat(results, ignore_index=True)[["condition", "sub", "motion"]]
combined = combined.sort_values(["sub", "condition"]).reset_index(drop=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_CSV, index=False)

print(f"\n=== Résumé ===")
print(combined.groupby("condition")["motion"].agg(["mean", "std", "min", "max"]).round(3).to_string())
print(f"\nCSV : {OUTPUT_CSV}")
