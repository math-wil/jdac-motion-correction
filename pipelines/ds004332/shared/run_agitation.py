"""
run_agitation.py
---------------------
Wrapper autour de la CLI agitation pour n'importe quel dataset.
Gère automatiquement les deux cas :
  - Dataset BIDS standard (dossier anat/ + dataset_description.json) → appel direct
  - Dataset non-BIDS ou structure plate → création d'un tmp_bids avec symlinks

Usage :
    python run_agitation.py \
        --bids_root /chemin/vers/dataset \
        --out       /chemin/vers/scores.csv

    # Si la structure n'est pas BIDS standard (ex. preprocessed_mni) :
    python run_agitation.py \
        --bids_root /chemin/vers/preprocessed_mni \
        --out       /chemin/vers/scores.csv \
        --pattern   "{sub}/{sub}_t1w_mni.nii.gz"

Flags optionnels :
    --pattern     Pattern des fichiers T1w dans les sous-dossiers sujets
                  Par défaut : "{sub}/anat/{sub}_T1w.nii.gz" (BIDS standard)
                  Pour preprocessed_mni : "{sub}/{sub}_t1w_mni.nii.gz"
    --tmp_dir     Dossier pour le tmp_bids temporaire (défaut : /tmp/tmp_bids_agitation)
    --keep_tmp    Ne pas supprimer le tmp_bids après le run
"""

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path


BIDS_PATTERN     = "{sub}/anat/{sub}_T1w.nii.gz"
BIDS_DESCRIPTION = {"Name": "tmp_bids", "BIDSVersion": "1.0.0"}


def is_bids(bids_root: Path) -> bool:
    """Vérifie si le dossier est un dataset BIDS valide (dataset_description.json présent)."""
    return (bids_root / "dataset_description.json").exists()


def create_tmp_bids(bids_root: Path, tmp_dir: Path, pattern: str) -> Path:
    """
    Crée une structure BIDS temporaire avec symlinks vers les fichiers T1w.
    pattern : ex. "{sub}/{sub}_t1w_mni.nii.gz"
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # dataset_description.json
    desc_file = tmp_dir / "dataset_description.json"
    with open(desc_file, "w") as f:
        json.dump(BIDS_DESCRIPTION, f)

    n_ok = 0
    n_missing = 0
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name
        src = bids_root / pattern.format(sub=sub)
        if not src.exists():
            n_missing += 1
            continue
        anat_dir = tmp_dir / sub / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        dst = anat_dir / f"{sub}_T1w.nii.gz"
        if not dst.exists():
            os.symlink(str(src.resolve()), str(dst))
        n_ok += 1

    print(f"  tmp_bids : {n_ok} sujets liés, {n_missing} fichiers manquants")
    return tmp_dir


def run_agitation(bids_dir: Path, out_csv: Path) -> int:
    """Lance agitation dataset et retourne le code de retour."""
    cmd = ["agitation", "dataset", "-d", str(bids_dir), "-o", str(out_csv)]
    print(f"  Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, help="Racine du dataset")
    parser.add_argument("--out",       required=True, help="Chemin du CSV de sortie")
    parser.add_argument("--pattern",   default=BIDS_PATTERN,
                        help="Pattern fichiers T1w (défaut BIDS standard)")
    parser.add_argument("--tmp_dir",   default="/tmp/tmp_bids_agitation",
                        help="Dossier tmp_bids temporaire")
    parser.add_argument("--keep_tmp",  action="store_true",
                        help="Conserver le tmp_bids après le run")
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    out_csv   = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Pipeline Agitation ===")
    print(f"  Dataset : {bids_root}")
    print(f"  Sortie  : {out_csv}\n")

    # Détermine si on doit créer un tmp_bids
    if is_bids(bids_root) and args.pattern == BIDS_PATTERN:
        print("Structure BIDS standard détectée — appel direct.")
        target = bids_root
        tmp_created = False
    else:
        print("Structure non-BIDS ou pattern custom — création du tmp_bids...")
        tmp_dir = Path(args.tmp_dir)
        target  = create_tmp_bids(bids_root, tmp_dir, args.pattern)
        tmp_created = True

    # Lance agitation
    print("\nLancement d'Agitation...")
    rc = run_agitation(target, out_csv)

    # Nettoyage
    if tmp_created and not args.keep_tmp:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        print(f"  tmp_bids supprimé.")

    if rc == 0:
        print(f"\nTerminé. Scores → {out_csv}")
    else:
        print(f"\nErreur Agitation (code {rc}). Vérifie le message ci-dessus.")

    return rc


if __name__ == "__main__":
    main()
