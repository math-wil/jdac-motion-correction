"""
compare_datasets.py
-------------------
Compare les profils ΔAIC entre deux datasets (ex. OASIS1 vs ds001907).
Produit :
- Scatter plot ΔAIC OASIS1 vs ΔAIC ds001907 par région APARC
- Barplot côte à côte des top régions
- CSV des régions concordantes (significatives dans les deux)

Usage :
    python compare_datasets.py \
        --glm_a  datasets/oasis1/results_mni/oasis1_glm_aparc_mni.csv \
        --label_a OASIS1 \
        --glm_b  datasets/ds001907/results/ds001907_glm_aparc.csv \
        --label_b ds001907 \
        --out_dir datasets/comparisons
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_glm(path, label):
    df = pd.read_csv(path)[["region", "delta_aic", "coef_motion", "p_fdr", "sig_fdr"]]
    df.columns = ["region", f"delta_aic_{label}", f"coef_{label}", f"p_fdr_{label}", f"sig_{label}"]
    return df


def plot_scatter(merged, label_a, label_b, out_path):
    """Scatter ΔAIC_A vs ΔAIC_B par région."""
    x = merged[f"delta_aic_{label_a}"]
    y = merged[f"delta_aic_{label_b}"]
    r, p = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = []
    for _, row in merged.iterrows():
        if row.get(f"sig_{label_a}", False) and row.get(f"sig_{label_b}", False):
            colors.append("#d73027")   # significatif dans les deux
        elif row.get(f"sig_{label_a}", False) or row.get(f"sig_{label_b}", False):
            colors.append("#fc8d59")   # significatif dans un seul
        else:
            colors.append("#91bfdb")   # non significatif

    ax.scatter(x, y, c=colors, alpha=0.8, s=50)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.axhline(4, color="gray", lw=0.4, ls=":")
    ax.axvline(4, color="gray", lw=0.4, ls=":")

    # Annoter les régions concordantes (rouge)
    concordant = merged[
        merged.get(f"sig_{label_a}", False) & merged.get(f"sig_{label_b}", False)
    ] if f"sig_{label_a}" in merged.columns else pd.DataFrame()
    for _, row in concordant.iterrows():
        ax.annotate(row["region"],
                    (row[f"delta_aic_{label_a}"], row[f"delta_aic_{label_b}"]),
                    fontsize=6, alpha=0.8)

    ax.set_xlabel(f"ΔAIC — {label_a}")
    ax.set_ylabel(f"ΔAIC — {label_b}")
    ax.set_title(f"Comparaison ΔAIC par région APARC\n{label_a} vs {label_b} | Spearman r={r:.3f}, p={p:.3e}")

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#d73027", label=f"Significatif dans les deux (FDR<0.05)"),
        Patch(color="#fc8d59", label="Significatif dans un seul"),
        Patch(color="#91bfdb", label="Non significatif"),
    ]
    ax.legend(handles=legend, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Scatter : {out_path}")


def plot_barplot_comparison(merged, label_a, label_b, out_path, top_n=20):
    """Barplot côte à côte ΔAIC des top régions."""
    # Top régions selon ΔAIC moyen des deux datasets
    merged["delta_aic_mean"] = (
        merged[f"delta_aic_{label_a}"] + merged[f"delta_aic_{label_b}"]
    ) / 2
    top = merged.nlargest(top_n, "delta_aic_mean").sort_values("delta_aic_mean")

    x = np.arange(len(top))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(x - width/2, top[f"delta_aic_{label_a}"], width, label=label_a, color="#4575b4", alpha=0.8)
    ax.barh(x + width/2, top[f"delta_aic_{label_b}"], width, label=label_b, color="#d73027", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(top["region"], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(4, color="gray", lw=0.6, ls="--", alpha=0.7, label="ΔAIC = 4")
    ax.set_xlabel("ΔAIC")
    ax.set_title(f"Top {top_n} régions APARC — ΔAIC {label_a} vs {label_b}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Barplot comparaison : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm_a",   required=True)
    parser.add_argument("--label_a", required=True)
    parser.add_argument("--glm_b",   required=True)
    parser.add_argument("--label_b", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = load_glm(args.glm_a, args.label_a)
    b = load_glm(args.glm_b, args.label_b)

    merged = a.merge(b, on="region", how="inner")
    print(f"  {len(merged)} régions en commun")

    # Régions concordantes
    sig_a = f"sig_{args.label_a}"
    sig_b = f"sig_{args.label_b}"
    if sig_a in merged.columns and sig_b in merged.columns:
        concordant = merged[merged[sig_a] & merged[sig_b]]
        print(f"\n  Régions significatives dans les deux datasets : {len(concordant)}")
        if len(concordant):
            print(concordant[["region",
                               f"delta_aic_{args.label_a}",
                               f"delta_aic_{args.label_b}"]].to_string(index=False))
        concordant.to_csv(out / f"concordant_{args.label_a}_{args.label_b}.csv", index=False)

    merged.to_csv(out / f"comparison_{args.label_a}_{args.label_b}.csv", index=False)

    plot_scatter(merged, args.label_a, args.label_b,
                 out / f"scatter_{args.label_a}_{args.label_b}.png")
    plot_barplot_comparison(merged, args.label_a, args.label_b,
                            out / f"barplot_{args.label_a}_{args.label_b}.png")

    print(f"\nTerminé. Fichiers dans {out}")


if __name__ == "__main__":
    main()
