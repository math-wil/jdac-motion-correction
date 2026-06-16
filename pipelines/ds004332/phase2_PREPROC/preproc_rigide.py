"""
preproc_rigide.py — Preprocessing RIGIDE pour ds004332 (bras PREPROC et JDAC).

But : préparer le cerveau sans changer sa TAILLE (le recalage affine de Clinica
redimensionnait le cerveau et faussait l'épaisseur de +0.14 mm ; le rigide ne fait
que tourner/translater, donc l'épaisseur reste comparable au natif).

Pipeline (UNIQUEMENT de vrais outils) :
  1. ants.abp_n4              : correction de biais d'intensité
  2. ants.registration Rigid : recalage RIGIDE vers MNI (6 DOF, ne redimensionne pas)
  3. mri_synthstrip          : extraction du cerveau (FreeSurfer)

Choix registration vs apply_transforms :
  On utilise ants.registration, qui CALCULE et APPLIQUE la transfo rigide en une fois
  (sortie = reg["warpedmovout"]). On NE PASSE PAS par ants.apply_transforms : celle-ci
  ne sert qu'à ré-appliquer une transfo existante (ex. aller-retour MNI de l'Option 1,
  abandonnée). On sauve quand même la transfo (.mat) au cas où on voudrait reprojeter
  une autre image plus tard.

Pas de crop fixe : FreeSurfer reconforme de toute façon, et JDAC fait son propre crop.

Usage :
  # une image
  conda run -n cortical-motion python preproc_rigide.py \
      --in raw_datasets/ds004332/sub-01/anat/sub-01_acq-mpragepmcoff_rec-wore_run-01_T1w.nii \
      --id sub-01_run-01 --out_dir derivatives/ds004332/rigid_preproc

  # toutes les images (acq-mpragepmcoff_rec-wore, 3 runs/sujet)
  conda run -n cortical-motion python preproc_rigide.py --batch \
      --raw_root raw_datasets/ds004332 --out_dir derivatives/ds004332/rigid_preproc
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path
import ants

# Cible d'orientation pour le recalage rigide. Le rigide ne redimensionne pas,
# donc la variante exacte de MNI importe peu (sert juste de repère d'orientation).
MNI = str(Path.home() / ".antspy/mni.nii.gz")
FREESURFER_HOME = "/project/hippocampus/common/softwares/freesurfer"


def preproc_one(in_path, sid, out_dir):
    out = Path(out_dir) / sid
    out.mkdir(parents=True, exist_ok=True)
    brain = out / f"{sid}_rigid_brain.nii.gz"
    if brain.exists():
        print(f"[{sid}] déjà fait, on saute.")
        return
    if not Path(in_path).exists():
        print(f"[{sid}] ENTRÉE INTROUVABLE : {in_path}")
        return

    # 1. correction de biais
    img = ants.image_read(str(in_path))
    n4 = ants.abp_n4(img)

    # 2. recalage RIGIDE vers MNI (calcule + applique -> warpedmovout)
    reg = ants.registration(fixed=ants.image_read(MNI), moving=n4, type_of_transform="Rigid")
    rigid_path = out / f"{sid}_rigid.nii.gz"
    ants.image_write(reg["warpedmovout"], str(rigid_path))
    for i, t in enumerate(reg["fwdtransforms"]):           # on sauve la transfo
        shutil.copy(t, str(out / f"{sid}_orig2mni_rigid_{i}{Path(t).suffix}"))

    # 3. skull stripping (vrai mri_synthstrip)
    env = {**os.environ, "FREESURFER_HOME": FREESURFER_HOME}
    subprocess.run(
        [f"{FREESURFER_HOME}/bin/mri_synthstrip",
         "-i", str(rigid_path), "-o", str(brain), "-m", str(out / f"{sid}_rigid_mask.nii.gz")],
        check=True, env=env,
    )
    print(f"[{sid}] OK -> {brain}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="traiter toutes les images mpragepmcoff_rec-wore")
    ap.add_argument("--raw_root", help="racine BIDS (mode batch)")
    ap.add_argument("--in", dest="inp", help="une image (mode simple)")
    ap.add_argument("--id", help="identifiant sub-XX_run-YY (mode simple)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.batch:
        root = Path(a.raw_root)
        imgs = sorted(root.glob("sub-*/anat/sub-*_acq-mpragepmcoff_rec-wore_run-*_T1w.nii"))
        print(f"{len(imgs)} images trouvées.")
        for f in imgs:
            # sub-01_acq-..._run-01_T1w.nii -> id = sub-01_run-01
            name = f.name
            sub = name.split("_")[0]
            run = [p for p in name.split("_") if p.startswith("run-")][0]
            preproc_one(f, f"{sub}_{run}", a.out_dir)
    else:
        if not (a.inp and a.id):
            sys.exit("mode simple : --in et --id requis")
        preproc_one(a.inp, a.id, a.out_dir)


if __name__ == "__main__":
    main()
