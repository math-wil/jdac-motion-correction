"""
pipeline_glm_aparc.py
---------------------
Pipeline GLM APARC générique — fonctionne sur n'importe quel dataset BIDS.
Pour chaque région APARC (34 lh + 34 rh) :
    base   : thickness ~ age + sex [+ group]
    motion : thickness ~ age + sex [+ group] + motion
Calcule ΔAIC, coefficients motion, correction FDR.
Sélectionne ~25 sujets stratifiés par score Agitation pour JDAC.

Usage :
    python pipeline_glm_aparc.py \
        --dataset      oasis1 \
        --scores       /chemin/vers/scores.csv \
        --fs_dir       /chemin/vers/freesurfer \
        --participants /chemin/vers/participants.tsv \
        --bids_root    /chemin/vers/bids \
        --out_dir      /chemin/vers/resultats

Flags optionnels :
    --id_format oasis1   Convertit sub-OASIS1XXXX → sub-XXXX (spécifique OASIS1)
    --n_low     9        Nombre de sujets stratum bas pour JDAC
    --n_mid     7        Nombre de sujets stratum moyen pour JDAC
    --n_high    9        Nombre de sujets stratum haut pour JDAC
    --no_jdac            Ne génère pas la sélection de sujets JDAC
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Lecture aparc.stats
# ---------------------------------------------------------------------------

def parse_aparc_stats(filepath):
    regions = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                regions[parts[0]] = float(parts[4])
            except ValueError:
                continue
    return regions


def load_all_aparc(fs_dir, subjects):
    fs_dir = Path(fs_dir)
    rows = []
    n_missing = 0
    first_missing = None
    for sub in subjects:
        row = {"sub": sub}
        ok = True
        for hemi in ["lh", "rh"]:
            path = fs_dir / sub / "stats" / f"{hemi}.aparc.stats"
            if not path.exists():
                n_missing += 1
                if first_missing is None:
                    first_missing = str(path)
                ok = False
                break
            for region, thick in parse_aparc_stats(path).items():
                row[f"{hemi}_{region}"] = thick
        if ok:
            rows.append(row)
    if n_missing:
        print(f"  aparc.stats manquants pour {n_missing} sujets. Premier : {first_missing}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chargement scores Agitation
# ---------------------------------------------------------------------------

def load_scores(scores_path):
    df = pd.read_csv(scores_path)
    # Format Agitation CLI : première colonne = index, colonnes sub + motion
    if "sub" not in df.columns:
        df = df.iloc[:, 1:3]
        df.columns = ["sub", "motion"]
    return df[["sub", "motion"]].dropna()


# ---------------------------------------------------------------------------
# Chargement démographie
# ---------------------------------------------------------------------------

def load_demo(participants_path, id_format=None):
    demo = pd.read_csv(participants_path, sep="\t")

    if "participant_id" not in demo.columns:
        raise ValueError("Colonne participant_id introuvable dans participants.tsv")

    if id_format == "oasis1":
        # sub-OASIS10001 → sub-0001
        demo["sub"] = (
            demo["participant_id"]
            .astype(str)
            .str.extract(r"sub-OASIS1(\d+)", expand=False)
            .apply(lambda x: f"sub-{x}" if pd.notna(x) else np.nan)
        )
    else:
        demo["sub"] = demo["participant_id"].astype(str)

    # Sexe
    sex_col = next((c for c in demo.columns if c.lower() in ["sex", "gender"]), None)
    if sex_col is None:
        raise ValueError("Colonne sex/gender introuvable dans participants.tsv")
    demo["sex"]     = demo[sex_col]
    demo["sex_num"] = (demo["sex"].astype(str).str.upper() == "M").astype(int)

    # Âge
    age_col = next((c for c in demo.columns if "age" in c.lower()), None)
    if age_col is None:
        raise ValueError("Colonne age introuvable dans participants.tsv")
    demo["age"] = pd.to_numeric(demo[age_col], errors="coerce")

    # Groupe (Parkinson, pathologie, etc.) — optionnel
    group_col = next(
        (c for c in demo.columns if c.lower() in ["group", "diagnosis", "pathology"]),
        None
    )
    has_group = group_col is not None
    if has_group:
        demo["group"]     = demo[group_col]
        demo["group_num"] = (
            demo[group_col].astype(str).str.lower()
            .str.contains("park|patient|case|disease")
        ).astype(int)
        print(f"  Colonne groupe : '{group_col}'")
        print(f"  Patients : {demo['group_num'].sum()} | Contrôles : {(demo['group_num']==0).sum()}")
    else:
        print("  Aucune colonne groupe — GLM sans covariable group")

    keep = ["sub", "age", "sex", "sex_num"] + (["group", "group_num"] if has_group else [])
    return demo[keep], has_group


# ---------------------------------------------------------------------------
# GLM par région
# ---------------------------------------------------------------------------

def fit_region(df, col, has_group):
    base_covars   = ["age", "sex_num"] + (["group_num"] if has_group else [])
    motion_covars = base_covars + ["motion"]
    sub = df[motion_covars + [col]].dropna()
    if len(sub) < 20:
        return None

    y  = sub[col]
    Xb = sm.add_constant(sub[base_covars])
    Xm = sm.add_constant(sub[motion_covars])

    try:
        mb = sm.GLM(y, Xb, family=sm.families.Gaussian()).fit()
        mm = sm.GLM(y, Xm, family=sm.families.Gaussian()).fit()
    except Exception as e:
        print(f"  Erreur GLM sur {col} : {e}")
        return None

    return {
        "region":      col,
        "n":           len(sub),
        "coef_motion": mm.params.get("motion", np.nan),
        "se_motion":   mm.bse.get("motion", np.nan),
        "p_motion":    mm.pvalues.get("motion", np.nan),
        "aic_base":    mb.aic,
        "aic_motion":  mm.aic,
        "delta_aic":   mb.aic - mm.aic,
    }


# ---------------------------------------------------------------------------
# Figure ΔAIC
# ---------------------------------------------------------------------------

def plot_delta_aic(df, out_path, dataset_name):
    df = df.sort_values("delta_aic")
    colors = ["#d73027" if v < 0 else "#4575b4" for v in df["delta_aic"]]
    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.20)))
    ax.barh(df["region"], df["delta_aic"], color=colors)
    ax.axvline(0,  color="black", lw=0.8)
    ax.axvline(4,  color="gray", lw=0.6, ls="--", alpha=0.7, label="ΔAIC = ±4")
    ax.axvline(-4, color="gray", lw=0.6, ls="--", alpha=0.7)
    ax.set_xlabel("ΔAIC  (AIC_base − AIC_motion)\nPositif = le modèle avec mouvement est meilleur")
    ax.set_title(f"Impact du score Agitation sur l'épaisseur corticale\npar région APARC — {dataset_name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Figure ΔAIC : {out_path}")


# ---------------------------------------------------------------------------
# Sélection stratifiée ~25 sujets pour JDAC
# ---------------------------------------------------------------------------

def select_subjects(df, bids_root, out_path, n_low, n_mid, n_high):
    df = (df[["sub", "motion", "age", "sex"]]
          .dropna()
          .sort_values("motion")
          .reset_index(drop=True))
    n = len(df)

    idx_low  = list(range(n_low))
    idx_high = list(range(n - n_high, n))
    mid_c    = n // 2
    idx_mid  = list(range(mid_c - n_mid // 2, mid_c - n_mid // 2 + n_mid))
    all_idx  = sorted(set(idx_low + idx_mid + idx_high))

    sel = df.loc[all_idx].copy()
    sel["stratum"] = "mid"
    sel.loc[sel.index.isin(df.index[idx_low]),  "stratum"] = "low"
    sel.loc[sel.index.isin(df.index[idx_high]), "stratum"] = "high"

    bids_root = Path(bids_root)
    sel["t1w_path"] = sel["sub"].apply(
        lambda s: str(bids_root / s / "anat" / f"{s}_T1w.nii.gz")
    )
    sel.to_csv(out_path, index=False)
    print(f"\n  {len(sel)} sujets sélectionnés → {out_path}")
    print(sel[["sub", "motion", "stratum"]].to_string(index=False))
    return sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True, help="Nom du dataset (ex. oasis1, ds001907)")
    parser.add_argument("--scores",       required=True, help="CSV scores Agitation")
    parser.add_argument("--fs_dir",       required=True, help="Répertoire FreeSurfer")
    parser.add_argument("--participants", required=True, help="participants.tsv BIDS")
    parser.add_argument("--bids_root",    required=True, help="Racine du dataset BIDS")
    parser.add_argument("--out_dir",      required=True, help="Dossier de sortie")
    parser.add_argument("--id_format",    default=None,  help="Format IDs : 'oasis1' pour conversion sub-OASIS1XXXX")
    parser.add_argument("--n_low",        type=int, default=9)
    parser.add_argument("--n_mid",        type=int, default=7)
    parser.add_argument("--n_high",       type=int, default=9)
    parser.add_argument("--no_jdac",      action="store_true", help="Ne pas générer la sélection JDAC")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Scores ----
    print(f"=== Pipeline GLM APARC — {args.dataset} ===\n")
    print("Chargement des scores Agitation...")
    scores = load_scores(args.scores)
    print(f"  {len(scores)} scores chargés")

    # ---- Démographie ----
    print("Chargement des données démographiques...")
    demo, has_group = load_demo(args.participants, args.id_format)
    merged = scores.merge(demo, on="sub", how="inner")
    print(f"  {len(merged)} sujets après fusion")

    # ---- APARC ----
    print("\nLecture des fichiers aparc.stats...")
    aparc = load_all_aparc(args.fs_dir, merged["sub"].tolist())
    print(f"  {len(aparc)} sujets avec aparc.stats")

    full = merged.merge(aparc, on="sub", how="inner")
    print(f"  {len(full)} sujets dans le DataFrame final")

    # ---- GLM ----
    region_cols = [c for c in aparc.columns if c != "sub"]
    print(f"\nAjustement GLM sur {len(region_cols)} régions...")
    results = [r for col in region_cols if (r := fit_region(full, col, has_group)) is not None]
    res_df  = pd.DataFrame(results)

    _, p_fdr, _, _ = multipletests(res_df["p_motion"].fillna(1), method="fdr_bh")
    res_df["p_fdr"]   = p_fdr
    res_df["sig_fdr"] = p_fdr < 0.05

    res_csv = out / f"{args.dataset}_glm_aparc.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"  Résultats → {res_csv}")

    print(f"\n  Régions significatives (FDR p<0.05) : {res_df['sig_fdr'].sum()} / {len(res_df)}")
    print(f"  ΔAIC > 4 : {(res_df['delta_aic'] > 4).sum()} / {len(res_df)}")
    print(f"  ΔAIC < -4 : {(res_df['delta_aic'] < -4).sum()} / {len(res_df)}")
    print("\n  Top 10 régions par ΔAIC :")
    print(res_df.nlargest(10, "delta_aic")[["region", "delta_aic", "coef_motion", "p_fdr"]].to_string(index=False))

    plot_delta_aic(res_df, out / f"{args.dataset}_aic_delta.png", args.dataset)

    if not args.no_jdac:
        print("\nSélection des sujets pour JDAC...")
        select_subjects(full, args.bids_root, out / f"{args.dataset}_jdac_subjects.csv",
                        args.n_low, args.n_mid, args.n_high)

    print(f"\nTerminé. Fichiers dans : {out}")


if __name__ == "__main__":
    main()
