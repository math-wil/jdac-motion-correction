"""
preprocess_jdac.py
------------------
Preprocessing générique pour entrée JDAC (no_internal_preproc).

Mode simple (--paired absent) — un seul T1w par sujet :
  1. Skull stripping (BET -R -m) → masque cérébral
  2. Application du masque
  3. Normalisation [0,1] + DivisiblePad k=16

Mode paired (--paired) — trois images par sujet (ref, mov1, mov2) :
  1. FLIRT rigid 6 DOF : mov1 + mov2 → ref
  2. Skull stripping de ref uniquement → masque partagé
  3. Application du masque aux 3 images
  4. Normalisation [0,1] + DivisiblePad k=16

Usage :
    # Mode simple
    python pipelines/generic/preprocess_jdac.py \
        --input_dir  ~/Documents/Datasets/ds000115 \
        --output_dir ~/Documents/Results/ds000115/jdac_ready

    # Mode paired — MR-ART (défauts, pas besoin de spécifier les patterns)
    python pipelines/generic/preprocess_jdac.py \
        --input_dir  ~/Documents/Datasets/MRART \
        --output_dir ~/Documents/Results/mrart/jdac_ready \
        --paired

    # Mode paired — ds004332 (noms et extension différents)
    python pipelines/generic/preprocess_jdac.py \
        --input_dir  ~/Documents/Datasets/ds004332 \
        --output_dir ~/Documents/Results/ds004332/jdac_ready \
        --paired \
        --ref_pattern  "{sub}_acq-mpragepmcoff_rec-wore_run-01_T1w.nii" \
        --mov1_pattern "{sub}_acq-mpragepmcoff_rec-wore_run-02_T1w.nii" \
        --mov2_pattern "{sub}_acq-mpragepmcoff_rec-wore_run-03_T1w.nii" \
        --ext .nii

Pré-requis : FSL (bet, flirt, fslmaths), nibabel + numpy dans l'env conda actif.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import nibabel as nib
import numpy as np

# Défauts MR-ART (rétrocompatibilité totale)
DEFAULT_REF_PATTERN  = "{sub}_acq-standard_T1w.nii.gz"
DEFAULT_MOV1_PATTERN = "{sub}_acq-headmotion1_T1w.nii.gz"
DEFAULT_MOV2_PATTERN = "{sub}_acq-headmotion2_T1w.nii.gz"


def run(cmd, desc):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} a échoué :\n{result.stderr.strip()}")


def normalize_and_pad(in_path, out_path, k=16):
    img  = nib.load(in_path)
    data = img.get_fdata(dtype=np.float32)
    data = np.clip(data, 0, None)  # supprime artefacts négatifs FLIRT spline
    brain = data[data > 0]
    if brain.size > 0:
        vmax = float(np.percentile(brain, 99.9))  # robuste aux outliers FLIRT
    else:
        vmax = float(data.max())
    if vmax > 1e-6:
        data = np.clip(data / vmax, 0.0, 1.0)
    for axis in range(3):
        r = data.shape[axis] % k
        if r != 0:
            pad_total  = k - r
            pad_before = pad_total // 2
            pad_after  = pad_total - pad_before
            pad_width  = [(0, 0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            data = np.pad(data, pad_width, mode="constant", constant_values=0)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)


def stem(pattern):
    """Extrait un label court depuis un pattern de nom de fichier pour nommer les sorties.
    Ex: '{sub}_acq-mpragepmcoff_rec-wore_run-01_T1w.nii' → 'run-01'
        '{sub}_acq-standard_T1w.nii.gz'                  → 'standard'
    """
    p = pattern.replace("{sub}_", "").replace("_T1w.nii.gz", "").replace("_T1w.nii", "")
    # Garde uniquement la dernière composante acq/rec/run trouvée
    parts = p.split("_")
    # Priorité : run- > rec- > acq- > tout
    for prefix in ("run-", "rec-", "acq-"):
        for part in reversed(parts):
            if part.startswith(prefix):
                return part
    return parts[-1]


def process_simple(sub, raw_anat, out_anat):
    out = out_anat / f"{sub}_T1w_brain_norm01.nii.gz"
    if out.exists():
        print(f"  Déjà traité, ignoré.")
        return

    t1w = raw_anat / f"{sub}_T1w.nii.gz"
    if not t1w.exists():
        # Essai sans .gz
        t1w_nii = raw_anat / f"{sub}_T1w.nii"
        if t1w_nii.exists():
            t1w = t1w_nii
        else:
            print(f"  AVERTISSEMENT : T1w introuvable, ignoré.")
            return

    sk      = out_anat / "tmp_sk.nii.gz"
    sk_mask = out_anat / "tmp_sk_mask.nii.gz"
    mask    = out_anat / f"{sub}_brain_mask.nii.gz"
    masked  = out_anat / "tmp_masked.nii.gz"

    run(["bet", str(t1w), str(sk), "-R", "-m"], "BET skull strip")
    sk_mask.rename(mask)
    run(["fslmaths", str(t1w), "-mas", str(mask), str(masked)], "Masque")
    print(f"  → Normalisation [0,1] + DivisiblePad k=16")
    normalize_and_pad(masked, out)

    for tmp in out_anat.glob("tmp_*.nii.gz"):
        tmp.unlink()


def process_paired(sub, raw_anat, out_anat, ref_pattern, mov1_pattern, mov2_pattern):
    # Labels courts pour nommer les sorties (ex: "standard", "run-01", "run-02")
    label_ref  = stem(ref_pattern)
    label_mov1 = stem(mov1_pattern)
    label_mov2 = stem(mov2_pattern)

    out_ref  = out_anat / f"{sub}_{label_ref}_brain_norm01.nii.gz"
    out_mov1 = out_anat / f"{sub}_{label_mov1}_brain_norm01.nii.gz"
    out_mov2 = out_anat / f"{sub}_{label_mov2}_brain_norm01.nii.gz"

    if out_ref.exists() and out_mov1.exists() and out_mov2.exists():
        print(f"  Déjà traité, ignoré.")
        return

    ref  = raw_anat / ref_pattern.format(sub=sub)
    mov1 = raw_anat / mov1_pattern.format(sub=sub)
    mov2 = raw_anat / mov2_pattern.format(sub=sub)

    missing = [f.name for f in [ref, mov1, mov2] if not f.exists()]
    if missing:
        print(f"  AVERTISSEMENT : fichiers manquants {missing}, ignoré.")
        return

    mov1_reg = out_anat / "tmp_mov1_reg.nii.gz"
    mov2_reg = out_anat / "tmp_mov2_reg.nii.gz"
    ref_sk   = out_anat / "tmp_ref_sk.nii.gz"
    sk_mask  = out_anat / "tmp_ref_sk_mask.nii.gz"
    mask     = out_anat / f"{sub}_brain_mask.nii.gz"
    ref_msk  = out_anat / "tmp_ref_masked.nii.gz"
    mov1_msk = out_anat / "tmp_mov1_masked.nii.gz"
    mov2_msk = out_anat / "tmp_mov2_masked.nii.gz"

    run(["flirt", "-in", str(mov1), "-ref", str(ref),
         "-out", str(mov1_reg), "-dof", "6", "-interp", "spline"],
        f"FLIRT : {label_mov1} → {label_ref}")
    run(["flirt", "-in", str(mov2), "-ref", str(ref),
         "-out", str(mov2_reg), "-dof", "6", "-interp", "spline"],
        f"FLIRT : {label_mov2} → {label_ref}")
    run(["bet", str(ref), str(ref_sk), "-R", "-m"],
        f"BET : masque depuis {label_ref}")
    sk_mask.rename(mask)

    for src, dst, label in [(ref,      ref_msk,  label_ref),
                            (mov1_reg, mov1_msk, label_mov1),
                            (mov2_reg, mov2_msk, label_mov2)]:
        run(["fslmaths", str(src), "-mas", str(mask), str(dst)],
            f"Masque → {label}")

    for src, dst, label in [(ref_msk,  out_ref,  label_ref),
                            (mov1_msk, out_mov1, label_mov1),
                            (mov2_msk, out_mov2, label_mov2)]:
        normalize_and_pad(src, dst)

    for tmp in out_anat.glob("tmp_*.nii.gz"):
        tmp.unlink()


def _worker(args_tuple):
    sub, input_dir, output_dir, paired, ref_pattern, mov1_pattern, mov2_pattern = args_tuple
    raw_anat = Path(input_dir) / sub / "anat"
    out_anat = Path(output_dir) / sub / "anat"
    out_anat.mkdir(parents=True, exist_ok=True)
    try:
        if paired:
            process_paired(sub, raw_anat, out_anat, ref_pattern, mov1_pattern, mov2_pattern)
        else:
            process_simple(sub, raw_anat, out_anat)
        return sub, None
    except Exception as e:
        return sub, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",    required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--paired",       action="store_true")
    parser.add_argument("--ref_pattern",  default=DEFAULT_REF_PATTERN,
                        help="Pattern nom fichier référence (défaut MR-ART)")
    parser.add_argument("--mov1_pattern", default=DEFAULT_MOV1_PATTERN,
                        help="Pattern nom fichier condition 1 (défaut MR-ART)")
    parser.add_argument("--mov2_pattern", default=DEFAULT_MOV2_PATTERN,
                        help="Pattern nom fichier condition 2 (défaut MR-ART)")
    parser.add_argument("--ext",          default=".nii.gz",
                        help="Extension des fichiers en entrée (défaut: .nii.gz)")
    parser.add_argument("--n_jobs",       type=int, default=min(8, os.cpu_count()),
                        help="Nombre de sujets traités en parallèle (défaut: 8)")
    args = parser.parse_args()

    if args.ext != ".nii.gz":
        def fix_ext(pattern, ext):
            return pattern.replace(".nii.gz", ext).replace(".nii", ext) \
                          if not pattern.endswith(ext) else pattern
        args.ref_pattern  = fix_ext(args.ref_pattern,  args.ext)
        args.mov1_pattern = fix_ext(args.mov1_pattern, args.ext)
        args.mov2_pattern = fix_ext(args.mov2_pattern, args.ext)

    for tool in ["bet", "fslmaths"] + (["flirt"] if args.paired else []):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            print(f"ERREUR : {tool} introuvable dans le PATH.")
            sys.exit(1)

    input_dir  = str(Path(args.input_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())

    subjects = sorted([d.name for d in Path(input_dir).iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])
    mode = "paired" if args.paired else "simple"
    print(f"\n{len(subjects)} sujets — mode {mode} — {args.n_jobs} workers\n")

    tasks = [(sub, input_dir, output_dir, args.paired,
              args.ref_pattern, args.mov1_pattern, args.mov2_pattern)
             for sub in subjects]

    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sub, err = future.result()
            done += 1
            if err:
                print(f"  [{done}/{len(subjects)}] ERREUR {sub} : {err}")
                errors.append(sub)
            else:
                print(f"  [{done}/{len(subjects)}] ✓ {sub}")

    if errors:
        print(f"\nÉchecs ({len(errors)}) : {', '.join(errors)}")
    else:
        print(f"\nPreprocessing terminé : {output_dir}")


if __name__ == "__main__":
    main()
