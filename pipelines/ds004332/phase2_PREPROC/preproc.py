"""
preproc.py — Preprocessing ds004332 (bras PREPROC et JDAC).

3 étapes (recalage rigide ajouté le 22/06, décision réunion 19/06) :
  1. abp_n4              (ANTsPy)     : correction de biais d'intensité
  2. ants.registration Rigid          : recalage RIGIDE vers MNI (6 DOF)
  3. mri_synthstrip     (FreeSurfer)  : extraction du cerveau

Pourquoi le rigide : il tourne et translate l'image sans la redimensionner, donc
l'épaisseur reste comparable au natif. (L'affine de Clinica redimensionnait le
cerveau et décalait l'épaisseur de +0.14 mm.) Le rigide donne une orientation
commune sans toucher à l'échelle. Pas de crop : FreeSurfer reconforme, JDAC crope
lui-même.

Le cerveau produit (`<sid>_brain.nii.gz`) sert aux DEUX bras :
  - PREPROC : cerveau -> FreeSurfer
  - JDAC    : cerveau -> JDAC -> FreeSurfer

Sortie : derivatives/ds004332/preproc_rigid/<sid>/
  <sid>_n4.nii.gz, <sid>_rigid.nii.gz, <sid>_brain.nii.gz, <sid>_mask.nii.gz,
  <sid>_orig2mni_rigid_*.mat (transfo rigide, gardée au cas où).
Le pipeline non-rigide précédent reste dans preproc/ (intact, pour comparaison).

Usage :
  conda run -n cortical-motion python preproc.py                      # batch des 66
  conda run -n cortical-motion python preproc.py --one sub-01_run-01  # une image
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import ants

FS = "/project/hippocampus/common/softwares/freesurfer"
MNI = str(Path.home() / ".antspy/mni.nii.gz")
MNI_PAD = 48  # marge (voxels) ajoutee autour du MNI : le FOV MNI standard est trop
              # court en bas et coupait le cervelet. On depose le rigide sur une
              # grille MNI elargie pour garder tout le cerveau.
RAW_ROOT = Path("/home/av62870@ens.ad.etsmtl.ca/Documents/raw_datasets/ds004332")
OUT_DIR = "/home/av62870@ens.ad.etsmtl.ca/Documents/derivatives/ds004332/preproc_rigid"


def preproc_une_image(entree, sortie_dir, sid):
    out = Path(sortie_dir) / sid
    out.mkdir(parents=True, exist_ok=True)
    n4_path    = out / f"{sid}_n4.nii.gz"
    rigid_path = out / f"{sid}_rigid.nii.gz"
    brain_path = out / f"{sid}_brain.nii.gz"
    mask_path  = out / f"{sid}_mask.nii.gz"

    if brain_path.exists():
        print(f"[{sid}] déjà fait, on saute.")
        return

    # 1. correction de biais
    img = ants.image_read(str(entree))
    n4 = ants.abp_n4(img)
    ants.image_write(n4, str(n4_path))

    # 2. recalage RIGIDE (6 DOF : rotation + translation, AUCUNE mise a l'echelle,
    #    contrairement a l'affine de Clinica). On calcule la transfo avec le MNI
    #    standard, puis on depose le resultat sur une grille MNI ELARGIE (MNI_PAD
    #    voxels de marge tout autour) pour ne pas couper le bas du cerveau.
    mni = ants.image_read(MNI)
    reg = ants.registration(fixed=mni, moving=n4, type_of_transform="Rigid")
    mni_big = ants.pad_image(mni, pad_width=[(MNI_PAD, MNI_PAD)] * 3)
    rigid = ants.apply_transforms(fixed=mni_big, moving=n4, transformlist=reg["fwdtransforms"])
    ants.image_write(rigid, str(rigid_path))
    for i, t in enumerate(reg["fwdtransforms"]):
        shutil.copy(t, str(out / f"{sid}_orig2mni_rigid_{i}{Path(t).suffix}"))

    # 3. skull stripping (sur l'image rigide)
    subprocess.run(
        [f"{FS}/bin/mri_synthstrip",
         "-i", str(rigid_path), "-o", str(brain_path), "-m", str(mask_path)],
        check=True,
        env={**os.environ, "FREESURFER_HOME": FS},
    )
    print(f"[{sid}] OK -> {brain_path}")


def chemin_image(sid):
    """sub-01_run-01 -> chemin BIDS de l'image brute correspondante."""
    sub, run = sid.split("_")
    return RAW_ROOT / sub / "anat" / f"{sub}_acq-mpragepmcoff_rec-wore_{run}_T1w.nii"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help="traiter une seule image, ex. sub-01_run-01")
    ap.add_argument("--out_dir", default=OUT_DIR)
    a = ap.parse_args()

    if a.one:
        f = chemin_image(a.one)
        if not f.exists():
            raise SystemExit(f"introuvable : {f}")
        preproc_une_image(f, a.out_dir, a.one)
    else:
        fichiers = sorted(RAW_ROOT.glob("sub-*/anat/sub-*_acq-mpragepmcoff_rec-wore_run-*_T1w.nii"))
        print(f"{len(fichiers)} images à traiter")
        for f in fichiers:
            sid = f.parent.parent.name + "_" + f.stem.split("_")[-2]   # ex : sub-01_run-01
            preproc_une_image(f, a.out_dir, sid)
