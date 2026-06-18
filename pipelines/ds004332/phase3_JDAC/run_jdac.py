"""
run_jdac.py
-----------
Inférence JDAC avec preprocessing interne MONAI (reproduit JDAC_Application.ipynb des auteurs).
S'applique via un CSV de sujets (colonnes : sub, t1w_path, motion, stratum).

Preprocessing interne appliqué :
    - CropForeground
    - ScaleIntensityRangePercentiles (percentiles 0-98, sortie [0,1])
    - DivisiblePad (k=16)

IMPORTANT : doit être lancé depuis ~/Documents/jdac/
car JDAC charge les poids depuis ./PretrainedModels/ (env conda cortical-motion).

Usage :
    cd ~/Documents/jdac
    python ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/run_jdac.py \
        --subjects ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/m1_sub01_subjects.csv \
        --out_dir  ~/Documents/derivatives/ds004332/jdac_m1_test/

Sortie par sujet :
    out_dir/sub-XXXX/sub-XXXX_T1w_jdac.nii.gz
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import pandas as pd

import monai.transforms as mt
from monai.transforms import Compose
from monai.utils import set_determinism

# Doit être lancé depuis ~/Documents/jdac/ pour que l'import fonctionne
sys.path.insert(0, str(Path.home() / "Documents/jdac"))
from models import AntiART_UNet, Denoiser_CondUNet

torch.manual_seed(0)
set_determinism(seed=0)

device = 'cpu'
PRETRAINED_DIR = Path('./PretrainedModels')


# ---------------------------------------------------------------------------
# Chargement des modèles (identique au notebook)
# ---------------------------------------------------------------------------

def load_models():
    print("Chargement des modèles...")

    Denoiser = Denoiser_CondUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(16, 32, 64, 128, 128, 16),
        norm='batch', upsample='nontrainable'
    ).to(device)
    model_path = PRETRAINED_DIR / 'Pretrained_Denoiser_l1loss_epoch15.pth'
    states = torch.load(str(model_path), map_location=lambda storage, loc: storage)
    Denoiser.load_state_dict(states['weights'])
    Denoiser.eval()
    print(f"  Denoiser chargé : {model_path}")

    AntiArt = AntiART_UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(16, 32, 64, 128, 128, 16),
        norm='batch', upsample='nontrainable'
    ).to(device)
    model_path = PRETRAINED_DIR / 'Pretrained_AntiArtNet_l1loss_epoch150.pth'
    states = torch.load(str(model_path), map_location=lambda storage, loc: storage)
    AntiArt.load_state_dict(states['weights'])
    AntiArt.eval()
    print(f"  AntiArt chargé  : {model_path}")

    return Denoiser, AntiArt


# ---------------------------------------------------------------------------
# Fonctions JDAC (copiées exactement du notebook)
# ---------------------------------------------------------------------------

def torch_std_estimate(img):
    dh = img[:,:,:,:,1:] - img[:,:,:,:,:-1]
    dw = img[:,:,:,1:,:] - img[:,:,:,:-1,:]
    dz = img[:,:,1:,:,:] - img[:,:,:-1,:,:]
    gra_map = (dh[:,:,1:,1:,:] + dw[:,:,1:,:,1:] + dz[:,:,:,1:,1:]) / 3.
    return torch.std(gra_map, dim=(-1,-2,-3), keepdim=True)


def torch_gradmap(img):
    dh = img[:,:,:,:,1:] - img[:,:,:,:,:-1]
    dw = img[:,:,:,1:,:] - img[:,:,:,:-1,:]
    dz = img[:,:,1:,:,:] - img[:,:,:-1,:,:]
    gra_map = (dh[:,:,1:,1:,:] + dw[:,:,1:,:,1:] + dz[:,:,:,1:,1:]) / 3.
    return torch.nn.functional.pad(gra_map, (1,0,1,0,1,0), "constant", 0)


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


def denoiser3d(denoiserNet, v_var, sigma=None):
    mt_scale = mt.ScaleIntensity(minv=0., maxv=1.0)
    v_var = mt_scale(v_var)
    if sigma is None:
        sigma = torch.abs(torch_std_estimate(v_var) - 0.028)
    predict, _, _ = denoiserNet(v_var, sigma)
    used_sigmas2 = sigma ** 2
    predict_noise = predict * used_sigmas2
    denoised = v_var - predict_noise
    return denoised.clip(0, 1)


def DenoiseAndAntiArt(Denoiser, AntiArt, noise_patch,
                      learning_rate=0.2, max_iter=4,
                      threshold_std=0.028, earlystop=True):
    v = noise_patch.clone()
    x = v.clone()
    u = torch.zeros_like(v)

    for idx in range(max_iter):
        x_old = x.clone()
        x = x_old * (1 - learning_rate) + v * learning_rate
        xu = (x - u)
        sigma = torch_std_estimate(xu).item()
        v = denoiser3d(Denoiser, xu, torch.tensor(sigma))
        v_sub_u = (v + u).clip(0, 1)
        x = anti_artifacts3d(AntiArt, v_sub_u)
        u = (v_sub_u - x)
        est_sigma = torch_std_estimate(x).item()
        if earlystop and est_sigma < threshold_std:
            print(f"    Early stop à l'itération {idx+1}")
            break

    return x, torch_std_estimate(x).item()


# ---------------------------------------------------------------------------
# Preprocessing (identique au notebook NBOLD)
# ---------------------------------------------------------------------------

def threshold_at_one(x):
    return x > 0.01

apply_trans = Compose([
    mt.CropForeground(select_fn=threshold_at_one, margin=0),
    mt.ScaleIntensityRangePercentiles(0, 98, 0.0, 1.0, True),
    mt.DivisiblePad(k=16),
])


# ---------------------------------------------------------------------------
# Traitement d'un sujet
# ---------------------------------------------------------------------------

def process_subject(sub, t1w_path, out_dir, Denoiser, AntiArt):
    t1w_path = Path(t1w_path)
    sub_out  = Path(out_dir) / sub
    sub_out.mkdir(parents=True, exist_ok=True)
    out_path = sub_out / f"{sub}_T1w_jdac.nii.gz"

    if out_path.exists():
        print(f"  [{sub}] Déjà traité, on passe.")
        return {"sub": sub, "status": "skipped", "duration_s": 0}

    if not t1w_path.exists():
        print(f"  [{sub}] Fichier introuvable : {t1w_path}")
        return {"sub": sub, "status": "missing_input", "duration_s": 0}

    t0 = time.time()

    # Chargement
    img        = nib.load(str(t1w_path))
    image_data = img.get_fdata()
    orig_shape = image_data.shape

    # Preprocessing interne, applique MANUELLEMENT pour pouvoir INVERSER la geometrie
    # (reproduit apply_trans : CropForeground select>0.01 margin 0, ScaleIntensity 0-98,
    #  DivisiblePad k=16 symetrique).
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

    # Inférence JDAC
    print(f"  [{sub}] Inférence JDAC (CPU, ~5-15 min)...")
    with torch.no_grad():
        denoised, grad_std = DenoiseAndAntiArt(
            Denoiser, AntiArt, img_tensor,
            max_iter=4, earlystop=True
        )
    denoised = (denoised * mask)[0][0].numpy()                            # forme padded

    # Inversion geometrie : un-pad puis replacer dans la grille d'origine.
    # -> sortie = MEME shape + MEME affine que l'entree (utilisable direct par recon-all,
    #    se superpose au cerveau preproc). Corrige le bug du crop+pad sauvegarde avec
    #    l'affine d'origine.
    sl = tuple(slice(b, b + s) for (b, _), s in zip(pads, cropped.shape))
    out_full = np.zeros(orig_shape, dtype=np.float32)
    out_full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = denoised[sl]

    # Sauvegarde
    out_nii = nib.Nifti1Image(out_full, img.affine, img.header)
    nib.save(out_nii, str(out_path))

    duration = time.time() - t0
    print(f"  [{sub}] ✓ Sauvegardé : {out_path} ({duration:.0f}s, grad_std={grad_std:.4f})")
    return {"sub": sub, "status": "ok", "duration_s": duration, "grad_std": grad_std}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", required=True, help="CSV sujets sélectionnés")
    parser.add_argument("--out_dir",  required=True, help="Dossier de sortie")
    args = parser.parse_args()

    # Vérification qu'on est dans le bon répertoire
    if not Path('./PretrainedModels').exists():
        print("ERREUR : lance ce script depuis ~/Documents/jdac/")
        print("  cd ~/Documents/jdac && python jdac_infer.py ...")
        sys.exit(1)

    Denoiser, AntiArt = load_models()

    subjects = pd.read_csv(args.subjects)
    print(f"\n{len(subjects)} sujets à traiter\n")

    logs = []
    for _, row in subjects.iterrows():
        print(f"--- {row['sub']} | motion={row['motion']:.3f} | {row['stratum']} ---")
        log = process_subject(row['sub'], row['t1w_path'], args.out_dir, Denoiser, AntiArt)
        logs.append(log)

    log_df = pd.DataFrame(logs)
    log_path = Path(args.out_dir) / "jdac_log.csv"
    log_df.to_csv(log_path, index=False)

    n_ok = (log_df["status"] == "ok").sum()
    print(f"\n{'='*50}")
    print(f"Terminé : {n_ok}/{len(log_df)} sujets traités")
    print(f"Log : {log_path}")


if __name__ == "__main__":
    main()
