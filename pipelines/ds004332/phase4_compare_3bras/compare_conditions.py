"""Comparaison de l'épaisseur corticale entre 5 conditions de traitement (ds004332, pipeline rigide).

Équivalent scriptable du notebook `explore_epaisseur_rigide.ipynb`, même logique et mêmes
modèles. Cinq conditions : brut, preproc, jdac, jdac_antiartonly (anti-artefact x1),
jdac_nodenoise (boucle x4). Suit le plan du directeur (réunion 2026-07-02) :
  1. épaisseur sur les scans immobiles (isole le lissage / offset) ;
  2. pentes épaisseur ~ Agitation par condition (globale + intra-sujet) ;
  3. modèles emboîtés M0 (âge+sexe) vs M1 (+Agitation), méthode de Charles ;
  3b. forme non-linéaire (Agitation², splines, par strate).
Analyses descriptives conservées : par consigne, interaction condition × niveau, Wilcoxon par niveau.

Vocabulaire : `condition` = les 5 traitements ; `consigne` = still/nodding/shaking.

ENTRÉES
- brut     : results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv (empilé, ThickAvg)
- preproc/jdac : derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv
- variantes    : derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/…_{lh,rh}.csv
- Agitation : results/ds004332/agitation/ds004332_agitation_clinica.csv
- démographie : raw_datasets/ds004332/participants.tsv

SORTIES (results/ds004332/phase4_compare_3bras/) : immobiles_par_condition.csv,
consigne_par_condition.csv, pentes_par_condition.csv, strate_par_condition.csv,
m0_vs_m1.csv, nonlineaire_par_condition.csv.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HOME  = Path.home()
REPO  = Path(__file__).resolve().parents[3]
DERIV = HOME / "Documents/derivatives/ds004332"
BRUTE_CSV    = REPO / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv"
AGIT_CSV     = REPO / "results/ds004332/agitation/ds004332_agitation_clinica.csv"
PARTICIPANTS = HOME / "Documents/raw_datasets/ds004332/participants.tsv"
OUTDIR = REPO / "results/ds004332/phase4_compare_3bras"

CONDITIONS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
CONSIGNE = {"run-01": "still", "run-02": "nodding", "run-03": "shaking"}
WIDE_PATH = {
    "preproc":          "thickness/thickness_preproc_{h}.csv",
    "jdac":             "thickness/thickness_jdac_{h}.csv",
    "jdac_antiartonly": "thickness/thickness_jdac_antiartonly_{h}.csv",
    "jdac_nodenoise":   "thickness/thickness_jdac_nodenoise_{h}.csv",
}


# ------------------------------------------------------------------------------ chargement
def load_brut():
    d = pd.read_csv(BRUTE_CSV)
    d = d[d["ThickAvg"] > 0].copy()                      # retire sub-01_run-03 (=0)
    d["run"] = d["subject"].str.split("_").str[1]
    d["subject"] = d["subject"].str.split("_").str[0]
    d = d.rename(columns={"ThickAvg": "thickness"})
    d["condition"] = "brut"
    return d[["subject", "run", "hemi", "region", "thickness", "condition"]]


def load_wide(condition):
    frames = []
    for hemi in ["lh", "rh"]:
        w = pd.read_csv(DERIV / WIDE_PATH[condition].format(h=hemi), sep="\t")
        w = w.rename(columns={w.columns[0]: "id"})
        cols = [c for c in w.columns if c.endswith("_thickness") and "MeanThickness" not in c]
        long = w.melt(id_vars="id", value_vars=cols, var_name="rr", value_name="thickness")
        long["hemi"] = hemi
        long["region"] = (long["rr"].str.replace(f"{hemi}_", "", regex=False)
                                     .str.replace("_thickness", "", regex=False))
        frames.append(long)
    d = pd.concat(frames, ignore_index=True)
    d["subject"] = d["id"].str.split("_").str[0]
    d["run"] = d["id"].str.split("_").str[1]
    d["condition"] = condition
    return d[["subject", "run", "hemi", "region", "thickness", "condition"]]


def load_data():
    thick = pd.concat([load_brut()] + [load_wide(c) for c in CONDITIONS if c != "brut"],
                      ignore_index=True)
    thick = thick[thick["thickness"] > 0].copy()

    agit = pd.read_csv(AGIT_CSV).rename(columns={"condition": "run", "sub": "subject", "motion": "agitation"})
    demog = pd.read_csv(PARTICIPANTS, sep="\t").rename(columns={"participant_id": "subject"})
    demog["sex_bin"] = (demog["sex"] == "F").astype(int)

    df = thick.merge(agit[["subject", "run", "agitation"]], on=["subject", "run"], how="inner")
    df = df.merge(demog[["subject", "age", "sex_bin"]], on="subject", how="left")
    df["consigne"] = df["run"].map(CONSIGNE)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS)

    print("Acquisitions par condition :")
    print(df.groupby("condition", observed=True)
            .apply(lambda x: x.groupby(["subject", "run"]).ngroups).to_string())
    return df


def aggregate(df):
    g = (df.groupby(["subject", "run", "consigne", "condition"], observed=True)
           .agg(thickness=("thickness", "mean"), agitation=("agitation", "first"),
                age=("age", "first"), sex_bin=("sex_bin", "first"))
           .reset_index())
    scans = g[["subject", "run", "agitation"]].drop_duplicates().copy()
    scans["niveau"] = pd.cut(scans["agitation"], [0, 0.3, 1.0, 2.0, np.inf],
                             labels=["faible", "leger", "modere", "severe"], include_lowest=True)
    g = g.merge(scans[["subject", "run", "niveau"]], on=["subject", "run"])
    return g


# ------------------------------------------------------------------ étape 1 : immobiles
def immobiles(g):
    imm = g[g["consigne"] == "still"].copy()
    tbl = imm.pivot_table(index="condition", values="thickness",
                          aggfunc=["mean", "std", "count"], observed=True)
    tbl.columns = ["epaisseur_moy", "ecart_type", "n"]
    tbl = tbl.reindex(CONDITIONS)
    ref = tbl.loc["brut", "epaisseur_moy"]
    tbl["ecart_brut_mm"] = tbl["epaisseur_moy"] - ref
    tbl["ecart_brut_pct"] = 100 * (tbl["epaisseur_moy"] - ref) / ref
    tbl.round(4).to_csv(OUTDIR / "immobiles_par_condition.csv")
    print("\n===== Étape 1 : épaisseur sur les scans immobiles (offset / lissage) =====")
    print(tbl.round(3).to_string())
    return tbl


# ------------------------------------------------------------------ descriptif par consigne
def consigne_desc(g):
    piv = g.pivot_table(index="consigne", columns="condition", values="thickness",
                        aggfunc="mean", observed=True).reindex(["still", "nodding", "shaking"])[CONDITIONS]
    var = 100 * (piv - piv.loc["still"]) / piv.loc["still"]
    piv.round(3).to_csv(OUTDIR / "consigne_par_condition.csv")
    print("\n===== Descriptif : épaisseur (mm) par consigne x condition =====")
    print(piv.round(3).to_string())
    print("\nVariation relative au still de la même condition (%) :")
    print(var.round(2).to_string())
    return piv


# ------------------------------------------------------------------ étape 2 : pentes
def pentes(g):
    print("\n===== Étape 2 : pentes épaisseur ~ Agitation =====")
    print("Pente globale par condition (mm par unité) :")
    for c in CONDITIONS:
        sub = g[g["condition"] == c]
        sl, _, _, p, _ = stats.linregress(sub["agitation"], sub["thickness"])
        print(f"  {c:16s}: {sl:+.4f}  (p={p:.2g})")

    rows = []
    for (subj, cond), sub in g.groupby(["subject", "condition"], observed=True):
        if len(sub) < 2 or sub["agitation"].std() == 0:
            continue
        rows.append(dict(subject=subj, condition=cond,
                         slope=stats.linregress(sub["agitation"], sub["thickness"]).slope))
    pen = pd.DataFrame(rows)
    med = pen.groupby("condition", observed=True)["slope"].median().reindex(CONDITIONS)
    print("\nPente médiane intra-sujet par condition :")
    print(med.round(4).to_string())

    wide = pen.pivot(index="subject", columns="condition", values="slope")
    rec = {}
    print("\nWilcoxon apparié des pentes intra-sujet (chaque condition vs brut) :")
    for c in CONDITIONS:
        if c == "brut":
            continue
        pair = wide[["brut", c]].dropna()
        p = stats.wilcoxon(pair["brut"], pair[c]).pvalue if len(pair) >= 3 else float("nan")
        rec[c] = p
        print(f"  brut vs {c:16s} (n={len(pair)}): p={p:.3g}")
    out = pd.DataFrame({"pente_mediane": med})
    out["p_vs_brut"] = pd.Series(rec)
    out.round(4).to_csv(OUTDIR / "pentes_par_condition.csv")
    return pen


# ------------------------------------------------------------------ strates : interaction + wilcoxon
def strates(g):
    gi = g.copy()
    gi["condition"] = pd.Categorical(gi["condition"], categories=CONDITIONS)
    fit = smf.mixedlm("thickness ~ C(condition, Treatment('brut')) * C(niveau, Treatment('faible'))",
                      gi, groups=gi["subject"]).fit(reml=True, method="powell", disp=False)
    print("\n===== Interaction condition x niveau (écart au brut selon le niveau de mouvement) =====")
    for k in fit.params.index:
        if ":" in k:
            cond = k.split("T.")[1].split("]")[0]
            niv = k.split("T.")[-1].rstrip("]")
            print(f"  {cond:16s} x {niv:7s}: {fit.params[k]:+.4f}  (p={fit.pvalues[k]:.2g})")

    paire = (g.pivot_table(index=["subject", "run", "niveau"], columns="condition",
                           values="thickness", observed=True).dropna(subset=["brut"]).reset_index())
    rows = []
    for niv in ["faible", "leger", "modere", "severe"]:
        s = paire[paire["niveau"] == niv]
        rec = {"niveau": niv, "n": len(s)}
        for c in CONDITIONS:
            if c == "brut":
                continue
            ss = s.dropna(subset=[c])
            rec[f"{c}_pct"] = (100 * (ss[c] - ss["brut"]) / ss["brut"]).mean() if len(ss) else float("nan")
            try:
                rec[f"p_{c}"] = stats.wilcoxon(ss[c], ss["brut"]).pvalue if len(ss) >= 3 else float("nan")
            except Exception:
                rec[f"p_{c}"] = float("nan")
        rows.append(rec)
    strat = pd.DataFrame(rows).set_index("niveau")
    strat.round(4).to_csv(OUTDIR / "strate_par_condition.csv")
    print("\nÉcart au brut par niveau (%) et Wilcoxon apparié :")
    print(strat.round(3).to_string())
    return strat


# ------------------------------------------------------------------ étape 3 : M0 vs M1
def _fit_ml(formula, data):
    return smf.mixedlm(formula, data, groups=data["subject"]).fit(reml=False, method="powell", disp=False)


def _aic(res):
    # structure aléatoire identique entre modèles (intercept sujet + variance résiduelle)
    return -2 * res.llf + 2 * (len(res.fe_params) + 2)


def m0_vs_m1(g):
    rows = []
    for c in CONDITIONS:
        sub = g[g["condition"] == c].dropna(subset=["age", "sex_bin", "agitation", "thickness"])
        m0 = _fit_ml("thickness ~ age + sex_bin", sub)
        m1 = _fit_ml("thickness ~ age + sex_bin + agitation", sub)
        lr = 2 * (m1.llf - m0.llf)
        rows.append(dict(condition=c, n=sub.groupby(["subject", "run"]).ngroups,
                         coef_agitation=m1.params["agitation"], p_agitation=m1.pvalues["agitation"],
                         LRT_chi2=lr, p_LRT=stats.chi2.sf(lr, 1), dAIC=_aic(m0) - _aic(m1)))
    tab = pd.DataFrame(rows).set_index("condition").reindex(CONDITIONS)
    tab.round(4).to_csv(OUTDIR / "m0_vs_m1.csv")
    print("\n===== Étape 3 : M0 (âge+sexe) vs M1 (+Agitation), apport du mouvement =====")
    print("dAIC>0 et p_LRT<0.05 : Agitation améliore le modèle (le mouvement prédit l'épaisseur).")
    print(tab.round(4).to_string())
    return tab


# ------------------------------------------------------------------ étape 3b : non-linéaire
def nonlineaire(g):
    formes = {
        "lineaire":    "thickness ~ age + sex_bin + agitation",
        "quadratique": "thickness ~ age + sex_bin + agitation + I(agitation**2)",
        "splines":     "thickness ~ age + sex_bin + bs(agitation, df=3)",
        "par_strate":  "thickness ~ age + sex_bin + C(niveau)",
    }
    rows = []
    for c in CONDITIONS:
        sub = g[g["condition"] == c].dropna(subset=["age", "sex_bin", "agitation", "thickness"])
        aics = {}
        for nom, f in formes.items():
            try:
                aics[nom] = _aic(smf.mixedlm(f, sub, groups=sub["subject"])
                                 .fit(reml=False, method="powell", disp=False))
            except Exception:
                aics[nom] = float("nan")
        base = aics["lineaire"]
        rows.append(dict(condition=c, **{f"dAIC_{n}": aics[n] - base for n in formes}))
    tab = pd.DataFrame(rows).set_index("condition").reindex(CONDITIONS)
    tab.round(3).to_csv(OUTDIR / "nonlineaire_par_condition.csv")
    print("\n===== Étape 3b : formes non-linéaires vs linéaire (ΔAIC, par strate incluse) =====")
    print("ΔAIC<0 : forme préférée au linéaire (écart > 2 = appui notable).")
    print(tab.round(2).to_string())
    return tab


def main():
    ap = argparse.ArgumentParser(description="Comparaison de l'épaisseur entre 5 conditions (ds004332, rigide).")
    ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Sorties : {OUTDIR}\n")
    df = load_data()
    g = aggregate(df)
    immobiles(g)
    consigne_desc(g)
    pentes(g)
    strates(g)
    m0_vs_m1(g)
    nonlineaire(g)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
