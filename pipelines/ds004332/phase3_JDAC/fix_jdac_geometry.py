#!/usr/bin/env python3
"""
Correctif geometrie des sorties JDAC.

Bug : run_jdac.py applique CropForeground + DivisiblePad (l'image passe ex. de
192x256x256 a 160x192x160) mais sauvegarde avec l'affine d'origine inchange ->
les donnees sont recadrees alors que l'affine dit "meme origine", donc le cerveau
est decale dans l'espace monde (ne se superpose plus a l'entree, FOV different).

Ce script replace chaque sortie JDAC dans la grille EXACTE de l'entree preproc
(meme shape + meme affine), SANS relancer le modele : le crop est deterministe
(foreground > 0.01) et le DivisiblePad(k=16) est symetrique (before = total//2),
ce qui a ete verifie voxel a voxel sur sub-01.

Sortie : derivatives/ds004332/jdac_fixed/<id>/<id>_T1w_jdac.nii.gz (NOUVEAU dossier,
les originaux decales sont conserves pour comparaison/QC).

Usage :
    python fix_jdac_geometry.py            # ecrit les 66 corriges dans jdac_fixed/
    python fix_jdac_geometry.py --check    # verifie seulement (Dice), n'ecrit rien
    # bras rigide :
    python fix_jdac_geometry.py \
        --pre  ~/Documents/derivatives/ds004332/preproc_rigid \
        --jd   ~/Documents/derivatives/ds004332/jdac_rigid \
        --out  ~/Documents/derivatives/ds004332/jdac_rigid_fixed
"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

HOME = Path.home()
PRE = HOME / "Documents/derivatives/ds004332/preproc"
JD = HOME / "Documents/derivatives/ds004332/jdac"
OUT_DIR = HOME / "Documents/derivatives/ds004332/jdac_fixed"


def fix_one(jd_path, check=False):
    sid = jd_path.parent.name                       # ex : sub-01_run-01
    brain_img = nib.load(str(PRE / sid / f"{sid}_brain.nii.gz"))
    brain = brain_img.get_fdata()
    jd_img = nib.load(str(jd_path))
    jd = jd_img.get_fdata()

    if jd.shape == brain.shape:
        return sid, "deja_ok", None

    # bbox CropForeground (select_fn = x > 0.01, margin = 0)
    c = np.array(np.nonzero(brain > 0.01))
    lo, hi = c.min(1), c.max(1) + 1
    cropped = hi - lo
    # DivisiblePad(k=16) symetrique -> offset 'before' par axe
    before = [((16 - s % 16) % 16) // 2 for s in cropped]

    pad_shape = tuple(int(np.ceil(s / 16) * 16) for s in cropped)
    if pad_shape != jd.shape:
        return sid, f"INCOHERENT pad={pad_shape} jdac={jd.shape}", None

    # un-pad -> bloc cropped, puis replacer dans la grille d'origine
    sl = tuple(slice(b, b + s) for b, s in zip(before, cropped))
    block = jd[sl]
    out = np.zeros(brain.shape, dtype=np.float32)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = block

    # QC1 : aucune valeur ajoutee/perdue vs le jdac d'origine (juste deplacees)
    same = (np.count_nonzero(out) == np.count_nonzero(jd)
            and np.isclose(out.sum(), jd.sum(), rtol=1e-5))
    # QC2 : recouvrement avec le cerveau preproc
    a, b = out > 0, brain > 0.01
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())

    if not check:
        out_sub = OUT_DIR / sid
        out_sub.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(out, brain_img.affine, brain_img.header),
                 str(out_sub / f"{sid}_T1w_jdac.nii.gz"))
    flag = "OK" if same else "!! VALEURS MODIFIEES"
    return sid, f"dice={dice:.3f} valeurs={flag}", out.shape


def main():
    global PRE, JD, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pre", help="dossier cerveaux preproc (entree JDAC)")
    ap.add_argument("--jd", help="dossier sorties JDAC a corriger")
    ap.add_argument("--out", help="dossier de sortie corrige")
    args = ap.parse_args()
    if args.pre: PRE = Path(args.pre).expanduser()
    if args.jd:  JD = Path(args.jd).expanduser()
    if args.out: OUT_DIR = Path(args.out).expanduser()
    paths = sorted(JD.glob("*/*_T1w_jdac.nii.gz"))
    print(f"{len(paths)} fichiers JDAC -> {OUT_DIR}")
    for p in paths:
        sid, msg, shape = fix_one(p, check=args.check)
        print(f"  {sid:18s} {msg}  {shape if shape else ''}")


if __name__ == "__main__":
    main()
