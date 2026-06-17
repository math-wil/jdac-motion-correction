"""
preproc.py — Preprocessing ds004332 (bras PREPROC et JDAC).

2 étapes, espace natif (pas de recalage, pas de crop) :
  1. abp_n4        (ANTsPy)     : correction de biais d'intensité
  2. mri_synthstrip (FreeSurfer): extraction du cerveau

Le cerveau produit sert aux DEUX bras :
  - PREPROC : cerveau -> FreeSurfer
  - JDAC    : cerveau -> JDAC -> FreeSurfer

Usage :
  conda run -n cortical-motion python preproc.py
"""
import ants
import subprocess
import os
from pathlib import Path

FS = "/project/hippocampus/common/softwares/freesurfer"


def preproc_une_image(entree, sortie_dir, sid):
    out = Path(sortie_dir) / sid
    out.mkdir(parents=True, exist_ok=True)
    n4_path    = out / f"{sid}_n4.nii.gz"
    brain_path = out / f"{sid}_brain.nii.gz"
    mask_path  = out / f"{sid}_mask.nii.gz"

    if brain_path.exists():
        print(f"[{sid}] déjà fait, on saute.")
        return

    # 1. correction de biais
    img = ants.image_read(str(entree))
    n4 = ants.abp_n4(img)
    ants.image_write(n4, str(n4_path))

    # 2. skull stripping
    subprocess.run(
        [f"{FS}/bin/mri_synthstrip",
         "-i", str(n4_path), "-o", str(brain_path), "-m", str(mask_path)],
        check=True,
        env={**os.environ, "FREESURFER_HOME": FS},
    )
    print(f"[{sid}] OK -> {brain_path}")


if __name__ == "__main__":
    raw_root = Path("/home/av62870@ens.ad.etsmtl.ca/Documents/raw_datasets/ds004332")
    out_dir  = "/home/av62870@ens.ad.etsmtl.ca/Documents/derivatives/ds004332/preproc"

    fichiers = sorted(raw_root.glob("sub-*/anat/sub-*_acq-mpragepmcoff_rec-wore_run-*_T1w.nii"))
    print(f"{len(fichiers)} images à traiter")
    for f in fichiers:
        sid = f.parent.parent.name + "_" + f.stem.split("_")[-2]   # ex : sub-01_run-01
        preproc_une_image(f, out_dir, sid)
