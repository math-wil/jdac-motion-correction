"""
run_jdac_preprocessed.py
------------------------
Lance JDAC sur des images déjà preprocessées (skull-strippées, normalisées [0,1],
DivisiblePad k=16). Le preprocessing interne JDAC/MONAI est désactivé.

Usage :
    # ds000115
    python pipelines/generic/run_jdac_preprocessed.py \
        --input_dir  ~/Documents/Results/ds000115/jdac_ready \
        --output_dir ~/Documents/Results/ds000115/jdac_outputs \
        --pattern    "{sub}/anat/{sub}_T1w_brain_norm01.nii.gz"

    # MR-ART headmotion1
    python pipelines/generic/run_jdac_preprocessed.py \
        --input_dir  ~/Documents/Results/mrart/jdac_ready \
        --output_dir ~/Documents/Results/mrart/jdac_outputs \
        --pattern    "{sub}/anat/{sub}_acq-headmotion1_T1w_brain_norm01.nii.gz"

Pré-requis : lancer depuis ~/Documents/jdac/
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import monai.transforms as mt
from monai.utils import set_determinism

sys.path.insert(0, str(Path.home() / "Documents/jdac"))
from models import AntiART_UNet, Denoiser_CondUNet

torch.manual_seed(0)
set_determinism(seed=0)

device = 'cuda'
PRETRAINED_DIR = Path('./PretrainedModels')


def load_models():
    print("Chargement des modèles...")
    Denoiser = Denoiser_CondUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(16, 32, 64, 128, 128, 16),
        norm='batch', upsample='nontrainable'
    ).to(device)
    states = torch.load(str(PRETRAINED_DIR / 'Pretrained_Denoiser_l1loss_epoch15.pth'),
                        map_location=lambda storage, loc: storage)
    Denoiser.load_state_dict(states['weights'])
    Denoiser.eval()
    print("  Denoiser chargé")

    AntiArt = AntiART_UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(16, 32, 64, 128, 128, 16),
        norm='batch', upsample='nontrainable'
    ).to(device)
    states = torch.load(str(PRETRAINED_DIR / 'Pretrained_AntiArtNet_l1loss_epoch150.pth'),
                        map_location=lambda storage, loc: storage)
    AntiArt.load_state_dict(states['weights'])
    AntiArt.eval()
    print("  AntiArt chargé")
    return Denoiser, AntiArt


def torch_std_estimate(img):
    dh = img[:,:,:,:,1:] - img[:,:,:,:,:-1]
    dw = img[:,:,:,1:,:] - img[:,:,:,:-1,:]
    dz = img[:,:,1:,:,:] - img[:,:,:-1,:,:]
    gra_map = (dh[:,:,1:,1:,:] + dw[:,:,1:,:,1:] + dz[:,:,:,1:,1:]) / 3.
    return torch.std(gra_map, dim=(-1,-2,-3), keepdim=True)


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
        v = denoiser3d(Denoiser, xu,
                       torch.tensor(sigma, device=device, dtype=torch.float32))
        v_sub_u = (v + u).clip(0, 1)
        x = anti_artifacts3d(AntiArt, v_sub_u)
        u = (v_sub_u - x)
        est_sigma = torch_std_estimate(x).item()
        if earlystop and est_sigma < threshold_std:
            print(f"    Early stop à l'itération {idx+1}")
            break
    return x, torch_std_estimate(x).item()


def process_subject(sub, t1w_path, out_dir, Denoiser, AntiArt):
    t1w_path = Path(t1w_path)
    sub_out  = Path(out_dir) / sub
    sub_out.mkdir(parents=True, exist_ok=True)
    out_path = sub_out / f"{sub}_T1w_jdac.nii.gz"

    if out_path.exists():
        print(f"  [{sub}] Déjà traité, ignoré.")
        return {"sub": sub, "status": "skipped", "duration_s": 0}

    if not t1w_path.exists():
        print(f"  [{sub}] Introuvable : {t1w_path}")
        return {"sub": sub, "status": "missing_input", "duration_s": 0}

    t0 = time.time()
    img        = nib.load(str(t1w_path))
    image_data = img.get_fdata()
    img_tensor = torch.tensor(image_data).unsqueeze(0).unsqueeze(0).float().to(device)
    mask       = img_tensor > 0

    print(f"  [{sub}] Inférence JDAC (GPU)...")
    with torch.no_grad():
        denoised, grad_std = DenoiseAndAntiArt(Denoiser, AntiArt, img_tensor,
                                               max_iter=4, earlystop=True)
    denoised_img = denoised * mask
    nib.save(nib.Nifti1Image(denoised_img[0][0].cpu().numpy(), img.affine, img.header),
             str(out_path))
    duration = time.time() - t0
    print(f"  [{sub}] ✓ {out_path.name} ({duration:.0f}s, grad_std={grad_std:.4f})")
    return {"sub": sub, "status": "done", "duration_s": round(duration, 1),
            "grad_std": round(grad_std, 4)}


def main():
    if not Path('./PretrainedModels').exists():
        print("ERREUR : lance ce script depuis ~/Documents/jdac/")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern",    required=True)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d.name for d in input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])
    print(f"\n{len(subjects)} sujets à traiter\n")

    Denoiser, AntiArt = load_models()

    logs = []
    for sub in subjects:
        t1w_path = input_dir / args.pattern.format(sub=sub)
        log = process_subject(sub, str(t1w_path), output_dir, Denoiser, AntiArt)
        logs.append(log)

    log_csv = output_dir / "jdac_log.csv"
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    done = sum(1 for l in logs if l["status"] == "done")
    print(f"\n{'='*50}")
    print(f"Terminé : {done}/{len(subjects)} sujets traités")
    print(f"Log : {log_csv}")


if __name__ == "__main__":
    main()
