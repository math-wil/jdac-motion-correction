"""Évaluation image façon JDAC : similarité du scan bougé (avant/après correction) au scan propre.

Reprend le protocole d'évaluation de l'article JDAC (PSNR/SSIM sur l'image ET sur les cartes de
gradient, en pleine référence contre un scan propre). Ici, toutes les images sont déjà sur la même
grille rigide (vérifié), donc pas de recalage. On compare, pour chaque sujet, ses deux scans bougés
(run-02 nodding, run-03 shaking) au scan propre du même sujet, avant correction (preproc) et après
chaque correction (jdac, antiartonly, nodenoise).

Deux références propres :
- clean = preproc run-01 (le brut adapté à l'espace rigide, non corrigé) -> mesure la proximité à
  l'anatomie propre. Confond correction du mouvement et changement d'intensité propre à la condition.
- intra = run-01 de la MÊME condition -> enlève le changement d'intensité de la condition, mesure la
  cohérence interne (mais la référence est elle-même traitée).

Intensités ramenées à [0,1] dans le cerveau (percentiles 1–99) pour comparer preproc (échelle N4) et
sorties JDAC ([0,1]). Métriques calculées dans la boîte englobante du cerveau commun.

Sortie : results/ds004332/phase4_compare_3bras/image_metrics.csv
Usage : python compute_image_metrics.py [--subjects sub-19 sub-10]
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

HOME = Path.home()
REPO = Path(__file__).resolve().parents[3]
DERIV = HOME / "Documents/derivatives/ds004332"
OUT = REPO / "results/ds004332/phase4_compare_3bras/image_metrics.csv"

# fichier image pour (condition, id) ; preproc = baseline avant correction
FILE = {
    "preproc":          "preproc_rigid/{id}/{id}_brain.nii.gz",
    "jdac":             "jdac_rigid/{id}/{id}_T1w_jdac.nii.gz",
    "jdac_antiartonly": "jdac_rigid_antiartonly/{id}/{id}_T1w_jdac_antiartonly.nii.gz",
    "jdac_nodenoise":   "jdac_rigid_nodenoise/{id}/{id}_T1w_jdac_nodenoise.nii.gz",
}
CONDITIONS = ["preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
MOVED = {"run-02": "nodding", "run-03": "shaking"}


def load(cond, sid):
    p = DERIV / FILE[cond].format(id=sid)
    return nib.load(p).get_fdata().astype(np.float32) if p.exists() else None


def norm(vol):
    """Échelle robuste [0,1] dans le cerveau (voxels > 0), fond à 0."""
    m = vol > 0
    lo, hi = np.percentile(vol[m], [1, 99])
    hi = hi if hi > lo else lo + 1e-6
    out = np.clip((vol - lo) / (hi - lo), 0, 1)
    out[~m] = 0
    return out


def gradmag(vol):
    gx, gy, gz = np.gradient(vol)
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def compare(ref, test):
    """ref, test normalisés [0,1]. Métriques dans la boîte du cerveau commun."""
    mask = (ref > 0) & (test > 0)
    if mask.sum() < 1000:
        return None
    idx = np.where(mask)
    bb = tuple(slice(i.min(), i.max() + 1) for i in idx)
    r, t = ref[bb], test[bb]
    rg, tg = gradmag(ref)[bb], gradmag(test)[bb]
    gmax = max(rg.max(), tg.max(), 1e-6)
    return {
        "psnr_img": peak_signal_noise_ratio(r, t, data_range=1.0),
        "ssim_img": structural_similarity(r, t, data_range=1.0),
        "ssim_grad": structural_similarity(rg / gmax, tg / gmax, data_range=1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=[f"sub-{i:02d}" for i in range(1, 23)])
    args = ap.parse_args()

    rows = []
    for s in args.subjects:
        clean = load("preproc", f"{s}_run-01")            # référence propre (non corrigée)
        clean_n = norm(clean) if clean is not None else None
        within_ref = {c: load(c, f"{s}_run-01") for c in CONDITIONS}
        within_n = {c: (norm(v) if v is not None else None) for c, v in within_ref.items()}
        for run, cons in MOVED.items():
            for c in CONDITIONS:
                mv = load(c, f"{s}_{run}")
                if mv is None:
                    continue
                mv_n = norm(mv)
                rec = {"subject": s, "run": run, "consigne": cons, "condition": c}
                if clean_n is not None:
                    r = compare(clean_n, mv_n)
                    if r:
                        rec.update({f"clean_{k}": v for k, v in r.items()})
                if within_n[c] is not None:
                    r = compare(within_n[c], mv_n)
                    if r:
                        rec.update({f"intra_{k}": v for k, v in r.items()})
                rows.append(rec)
                print(f"  {s} {run} {c}: "
                      + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if isinstance(v, float)))
    import pandas as pd
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n{len(df)} lignes -> {OUT}")


if __name__ == "__main__":
    main()
