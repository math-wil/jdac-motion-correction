"""
run_jdac_nodenoise.py
---------------------
Variante de JDAC SANS la phase de débruitage, pour tester la piste de la
réunion du 19/06 : « JDAC lisse trop, essayer sans le débruitage ».

JDAC normal = deux U-Net pré-entraînés enchaînés dans une boucle itérative :
un débruiteur (lissage, entraîné sur bruit gaussien) et un modèle anti-artefact
(correction du mouvement). Ici on garde le modèle anti-artefact pré-entraîné et
on retire le débruiteur. Aucun réentraînement, aucun GPU obligatoire.

Deux variantes (option --variant) :
  - loop : on garde la boucle itérative de JDAC (max_iter, early-stop) mais
           l'étape débruiteur est remplacée par l'identité. C'est « JDAC sans
           sa phase de débruitage », au plus proche de la mécanique d'origine.
  - once : on applique le modèle anti-artefact une seule fois, sans boucle.
           Isole l'apport brut de la correction de mouvement.

Entrée = les MÊMES cerveaux que le JDAC rigide actuel : N4 + recalage rigide
MNI + SynthStrip, listés dans all66_subjects_rigid.csv (colonne t1w_path pointe
sur preproc_rigid/.../_brain.nii.gz).

IMPORTANT : lancer depuis ~/Documents/jdac/ (les poids sont dans ./PretrainedModels/,
env conda cortical-motion).

Usage :
    cd ~/Documents/jdac
    python ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/run_jdac_nodenoise.py \
        --subjects ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/all66_subjects_rigid.csv \
        --variant  loop \
        --out_dir  ~/Documents/derivatives/ds004332/jdac_rigid_nodenoise/

Sortie par sujet (suffixe selon la variante, pour ne pas confondre les versions) :
    variant loop  -> out_dir/<id>/<id>_T1w_jdac_nodenoise.nii.gz
    variant once  -> out_dir/<id>/<id>_T1w_jdac_antiartonly.nii.gz
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import pandas as pd

import monai.transforms as mt
from monai.utils import set_determinism

# Lancement depuis ~/Documents/jdac/ pour importer le modèle anti-artefact.
sys.path.insert(0, str(Path.home() / "Documents/jdac"))
from models import AntiART_UNet

torch.manual_seed(0)
set_determinism(seed=0)

device = 'cpu'
PRETRAINED_DIR = Path('./PretrainedModels')


# ---------------------------------------------------------------------------
# Modèle anti-artefact (on ne charge PAS le débruiteur)
# ---------------------------------------------------------------------------

def load_antiart():
    AntiArt = AntiART_UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(16, 32, 64, 128, 128, 16),
        norm='batch', upsample='nontrainable'
    ).to(device)
    model_path = PRETRAINED_DIR / 'Pretrained_AntiArtNet_l1loss_epoch150.pth'
    states = torch.load(str(model_path), map_location=lambda storage, loc: storage)
    AntiArt.load_state_dict(states['weights'])
    AntiArt.eval()
    print(f"  AntiArt chargé : {model_path}")
    return AntiArt


# ---------------------------------------------------------------------------
# Fonctions JDAC (identiques au script run_jdac.py, débruiteur exclu)
# ---------------------------------------------------------------------------

def torch_std_estimate(img):
    dh = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]
    dw = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    gra_map = (dh[:, :, 1:, 1:, :] + dw[:, :, 1:, :, 1:] + dz[:, :, :, 1:, 1:]) / 3.
    return torch.std(gra_map, dim=(-1, -2, -3), keepdim=True)


def anti_artifacts3d(antiartNet, perturbed_samples, step_lr=1.0):
    mt_scale = mt.ScaleIntensity(minv=0., maxv=1.0)
    perturbed_samples = mt_scale(perturbed_samples)
    perturbed_samples = perturbed_samples * 255
    scores, scores_down = antiartNet(perturbed_samples)
    scores = scores / 255.0
    perturbed_samples = perturbed_samples / 255.0
    res = (perturbed_samples - scores)
    res = (res - torch.mean(res)).clip(-0.04, 0.04)
    denoised = perturbed_samples - step_lr * res
    return denoised.clip(0, 1)


# ---------------------------------------------------------------------------
# Les deux variantes sans débruitage
# ---------------------------------------------------------------------------

def antiart_loop(AntiArt, noise_patch, learning_rate=0.2, max_iter=4,
                 threshold_std=0.028, earlystop=True):
    """Boucle JDAC d'origine, étape débruiteur remplacée par l'identité."""
    v = noise_patch.clone()
    x = v.clone()
    u = torch.zeros_like(v)
    for idx in range(max_iter):
        x_old = x.clone()
        x = x_old * (1 - learning_rate) + v * learning_rate
        xu = (x - u)
        v = xu                              # débruiteur retiré : identité
        v_sub_u = (v + u).clip(0, 1)
        x = anti_artifacts3d(AntiArt, v_sub_u)
        u = (v_sub_u - x)
        est_sigma = torch_std_estimate(x).item()
        if earlystop and est_sigma < threshold_std:
            print(f"    Early stop à l'itération {idx+1}")
            break
    return x, torch_std_estimate(x).item()


def antiart_once(AntiArt, noise_patch):
    """Une seule application du modèle anti-artefact, sans boucle."""
    x = anti_artifacts3d(AntiArt, noise_patch)
    return x, torch_std_estimate(x).item()


# ---------------------------------------------------------------------------
# Traitement d'un sujet (preprocessing interne + inversion géométrie,
# identiques à run_jdac.py)
# ---------------------------------------------------------------------------

def process_subject(sub, t1w_path, out_dir, AntiArt, variant):
    t1w_path = Path(t1w_path)
    sub_out = Path(out_dir) / sub
    sub_out.mkdir(parents=True, exist_ok=True)
    # Suffixe explicite dans le nom de fichier pour distinguer les versions :
    #   loop -> _jdac_nodenoise   |   once -> _jdac_antiartonly
    suffix = "nodenoise" if variant == "loop" else "antiartonly"
    out_path = sub_out / f"{sub}_T1w_jdac_{suffix}.nii.gz"

    if out_path.exists():
        print(f"  [{sub}] Déjà traité, on passe.")
        return {"sub": sub, "status": "skipped", "duration_s": 0}

    if not t1w_path.exists():
        print(f"  [{sub}] Fichier introuvable : {t1w_path}")
        return {"sub": sub, "status": "missing_input", "duration_s": 0}

    t0 = time.time()

    img = nib.load(str(t1w_path))
    image_data = img.get_fdata()
    orig_shape = image_data.shape

    # Preprocessing interne (CropForeground >0.01, percentiles 0-98 -> [0,1],
    # pad divisible par 16), appliqué à la main pour pouvoir inverser la géométrie.
    fg = image_data > 0.01
    if not fg.any():
        print(f"  [{sub}] image vide")
        return {"sub": sub, "status": "empty", "duration_s": 0}
    coords = np.array(np.nonzero(fg))
    lo, hi = coords.min(1), coords.max(1) + 1
    cropped = image_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    cropped = mt.ScaleIntensityRangePercentiles(0, 98, 0.0, 1.0, True)(
        torch.tensor(cropped).unsqueeze(0))[0].numpy()
    pads = [(((16 - s % 16) % 16) // 2, (16 - s % 16) % 16 - ((16 - s % 16) % 16) // 2)
            for s in cropped.shape]
    padded = np.pad(cropped, pads, mode="constant")

    img_tensor = torch.tensor(padded).unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W,D)
    mask = img_tensor > 0

    print(f"  [{sub}] Inférence anti-artefact (variant={variant})...")
    with torch.no_grad():
        if variant == "loop":
            out, grad_std = antiart_loop(AntiArt, img_tensor, max_iter=4, earlystop=True)
        else:
            out, grad_std = antiart_once(AntiArt, img_tensor)
    out = (out * mask)[0][0].numpy()                                     # forme padded

    # Inversion géométrie : un-pad puis replacer dans la grille d'origine
    # -> même shape + même affine que l'entrée (utilisable direct par recon-all).
    sl = tuple(slice(b, b + s) for (b, _), s in zip(pads, cropped.shape))
    out_full = np.zeros(orig_shape, dtype=np.float32)
    out_full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = out[sl]

    out_nii = nib.Nifti1Image(out_full, img.affine, img.header)
    nib.save(out_nii, str(out_path))

    duration = time.time() - t0
    print(f"  [{sub}] ✓ {out_path} ({duration:.0f}s, grad_std={grad_std:.4f})")
    return {"sub": sub, "status": "ok", "duration_s": duration, "grad_std": grad_std}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", required=True, help="CSV sujets (all66_subjects_rigid.csv)")
    parser.add_argument("--variant", choices=["loop", "once"], required=True,
                        help="loop = boucle JDAC sans débruiteur ; once = anti-artefact une passe")
    parser.add_argument("--out_dir", required=True, help="Dossier de sortie")
    args = parser.parse_args()

    if not Path('./PretrainedModels').exists():
        print("ERREUR : lance ce script depuis ~/Documents/jdac/")
        sys.exit(1)

    print(f"Variante : {args.variant}")
    AntiArt = load_antiart()

    subjects = pd.read_csv(args.subjects)
    print(f"\n{len(subjects)} sujets à traiter\n")

    logs = []
    for _, row in subjects.iterrows():
        print(f"--- {row['sub']} | motion={row['motion']:.3f} ---")
        logs.append(process_subject(row['sub'], row['t1w_path'], args.out_dir, AntiArt, args.variant))

    log_df = pd.DataFrame(logs)
    log_path = Path(args.out_dir) / "jdac_log.csv"
    log_df.to_csv(log_path, index=False)

    n_ok = (log_df["status"] == "ok").sum()
    print(f"\n{'='*50}")
    print(f"Terminé ({args.variant}) : {n_ok}/{len(log_df)} sujets")
    print(f"Log : {log_path}")


if __name__ == "__main__":
    main()
