"""
GLM Pipeline A vs B -- ds004332 -- effet de JDAC sur le biais de mouvement.

================================================================================
QUESTION
================================================================================
En Phase 1 (images brutes -> FreeSurfer) : plus le mouvement (Agitation) est fort,
plus l'epaisseur corticale mesuree est faible (pente negative).

Ici on compare deux bras, sur les MEMES scans :
    PREPROC : cerveau (N4 + SynthStrip) -> FreeSurfer            (= reference, "B")
    JDAC    : cerveau -> JDAC -> FreeSurfer                       (= corrige,   "A")

Deux choses a distinguer :
  - OFFSET   : JDAC abaisse-t-il l'epaisseur de facon uniforme ? (effet "arm")
  - PENTE    : JDAC change-t-il le LIEN mouvement -> epaisseur ?  (interaction
               agitation x arm)

Verdict :
  - interaction ~ 0, non significative  -> JDAC ne corrige pas le mouvement,
    il ne fait que decaler/lisser uniformement.
  - interaction positive et significative (pente jdac moins negative que preproc)
    -> JDAC aplatit la pente = vraie correction du mouvement.

================================================================================
MODELES
================================================================================
Par bras (comme Pipeline B), pour chaque region :
    thickness ~ age + sexe + agitation + (1|sujet)        -> beta_agitation par bras

Compare (les deux bras empiles, scans apparies) :
    thickness ~ age + sexe + agitation * arm + (1|sujet)
      - agitation              : pente du bras preproc
      - arm[T.jdac]            : OFFSET uniforme de JDAC
      - agitation:arm[T.jdac]  : CHANGEMENT de pente du a JDAC  <-- le test clef
Au niveau global (epaisseur moyenne par run) et par region (+ FDR sur l'interaction).

================================================================================
ENTREES
================================================================================
- Epaisseur : aparcstats2table (format large), 4 fichiers :
      derivatives/ds004332/thickness_{preproc,jdac}_{lh,rh}.csv
- Agitation : results/ds004332/agitation/ds004332_agitation_clinica.csv
- Demographics : raw_datasets/ds004332/participants.tsv

SORTIES
- Par region par bras   : results/ds004332/phase3_JDAC/glm_AvsB_par_bras.csv
- Interaction par region : results/ds004332/phase3_JDAC/glm_AvsB_interaction.csv
- Resume global imprime en fin.

Usage :
    conda run -n cortical-motion python3 \
        ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/glm_pipeline_AvsB.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

HOME = Path("/home/av62870@ens.ad.etsmtl.ca")
REPO = HOME / "Documents/jdac-motion-correction"
THICK_DIR = HOME / "Documents/derivatives/ds004332"        # thickness_{arm}_{hemi}.csv
AGITATION_CSV = REPO / "results/ds004332/agitation/ds004332_agitation_clinica.csv"
PARTICIPANTS = HOME / "Documents/raw_datasets/ds004332/participants.tsv"
OUT_BRAS = REPO / "results/ds004332/phase3_JDAC/glm_AvsB_par_bras.csv"
OUT_INTER = REPO / "results/ds004332/phase3_JDAC/glm_AvsB_interaction.csv"

ARMS = ["preproc", "jdac"]


# ------------------------------------------------------------------------------
# Chargement : aparcstats2table (large) -> long (subject, run, hemi, region, ...)
# ------------------------------------------------------------------------------
def load_arm_thickness(arm):
    frames = []
    for hemi in ["lh", "rh"]:
        w = pd.read_csv(THICK_DIR / f"thickness_{arm}_{hemi}.csv", sep="\t")
        w = w.rename(columns={w.columns[0]: "id"})
        region_cols = [c for c in w.columns
                       if c.endswith("_thickness") and "MeanThickness" not in c]
        long = w.melt(id_vars="id", value_vars=region_cols,
                      var_name="region_raw", value_name="thickness_mean")
        long["hemi"] = hemi
        long["region"] = (long["region_raw"]
                          .str.replace(f"{hemi}_", "", regex=False)
                          .str.replace("_thickness", "", regex=False))
        frames.append(long)
    df = pd.concat(frames, ignore_index=True)
    df["subject"] = df["id"].str.split("_").str[0]
    df["run"] = df["id"].str.split("_").str[1]
    df["arm"] = arm
    return df[["subject", "run", "hemi", "region", "thickness_mean", "arm"]]


def load_data():
    thick = pd.concat([load_arm_thickness(a) for a in ARMS], ignore_index=True)
    n_before = len(thick)
    thick = thick[thick["thickness_mean"] > 0].copy()      # echecs silencieux = 0

    agit = pd.read_csv(AGITATION_CSV).rename(
        columns={"condition": "run", "sub": "subject", "motion": "agitation"})
    demog = pd.read_csv(PARTICIPANTS, sep="\t").rename(
        columns={"participant_id": "subject"})
    demog["sex_bin"] = (demog["sex"] == "F").astype(int)

    df = thick.merge(agit[["subject", "run", "agitation"]], on=["subject", "run"], how="inner")
    df = df.merge(demog[["subject", "age", "sex_bin"]], on="subject", how="left")

    for a in ARMS:
        sub = df[df["arm"] == a]
        print(f"  {a:8s}: {sub.groupby(['subject','run']).ngroups} runs, {len(sub)} obs region")
    print(f"Retire {n_before - len(thick)} obs a 0 (echec silencieux). Total obs = {len(df)}")
    return df


# ------------------------------------------------------------------------------
# Modeles par bras (pente Agitation, comme Pipeline B)
# ------------------------------------------------------------------------------
def fit_region_single(sub_df):
    """thickness ~ age + sexe + agitation + (1|sujet). Fallback OLS effets fixes."""
    try:
        fit = mixedlm("thickness_mean ~ age + sex_bin + agitation",
                      sub_df, groups=sub_df["subject"]).fit(reml=True, method="powell", disp=False)
        if fit.converged:
            return fit.params["agitation"], fit.bse["agitation"], fit.pvalues["agitation"], "mixedlm"
    except Exception:
        pass
    fit = ols("thickness_mean ~ agitation + C(subject)", sub_df).fit()
    return fit.params["agitation"], fit.bse["agitation"], fit.pvalues["agitation"], "ols_fe"


def glm_par_bras(df):
    rows = []
    for arm in ARMS:
        d = df[df["arm"] == arm]
        for (hemi, region), sub in d.groupby(["hemi", "region"]):
            if sub["subject"].nunique() < 5 or sub["agitation"].std() == 0:
                continue
            beta, se, p, mtype = fit_region_single(sub.copy())
            rows.append(dict(arm=arm, hemi=hemi, region=region,
                             beta_agitation=beta, se=se, p_value=p, model=mtype,
                             n_obs=len(sub)))
    res = pd.DataFrame(rows)
    for arm in ARMS:
        m = res["arm"] == arm
        res.loc[m, "p_fdr"] = multipletests(res.loc[m, "p_value"], method="fdr_bh")[1]
    return res


# ------------------------------------------------------------------------------
# Modele compare : interaction agitation x arm (le test clef)
# ------------------------------------------------------------------------------
def fit_region_interaction(sub_df):
    """thickness ~ age + sexe + agitation * arm + (1|sujet).
    Renvoie offset (arm jdac), delta_pente (interaction) et p de l'interaction."""
    sub_df = sub_df.copy()
    sub_df["arm"] = pd.Categorical(sub_df["arm"], categories=["preproc", "jdac"])
    try:
        fit = mixedlm("thickness_mean ~ age + sex_bin + agitation * arm",
                      sub_df, groups=sub_df["subject"]).fit(reml=True, method="powell", disp=False)
        if fit.converged:
            inter = [k for k in fit.params.index if "agitation:" in k][0]
            armk = [k for k in fit.params.index if k.startswith("arm[") and ":" not in k][0]
            return (fit.params["agitation"], fit.params[armk],
                    fit.params[inter], fit.pvalues[inter], "mixedlm")
    except Exception:
        pass
    fit = ols("thickness_mean ~ agitation * arm + C(subject)", sub_df).fit()
    inter = [k for k in fit.params.index if "agitation:" in k][0]
    armk = [k for k in fit.params.index if k.startswith("arm[") and ":" not in k][0]
    return (fit.params["agitation"], fit.params[armk],
            fit.params[inter], fit.pvalues[inter], "ols_fe")


def glm_interaction(df):
    rows = []
    for (hemi, region), sub in df.groupby(["hemi", "region"]):
        if sub["arm"].nunique() < 2 or sub["agitation"].std() == 0:
            continue
        pente_pre, offset, delta, p, mtype = fit_region_interaction(sub)
        rows.append(dict(hemi=hemi, region=region, pente_preproc=pente_pre,
                         offset_jdac=offset, delta_pente=delta, p_inter=p,
                         model=mtype, n_obs=len(sub)))
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p_inter"], method="fdr_bh")[1]
    return res.sort_values("p_inter")


# ------------------------------------------------------------------------------
# Resume global (epaisseur moyenne par run, par bras)
# ------------------------------------------------------------------------------
def resume_global(df):
    g = (df.groupby(["subject", "run", "arm"])
           .agg(thickness_mean=("thickness_mean", "mean"),
                age=("age", "first"), sex_bin=("sex_bin", "first"),
                agitation=("agitation", "first"))
           .reset_index())
    g["arm"] = pd.Categorical(g["arm"], categories=["preproc", "jdac"])

    print("\n========== RESUME GLOBAL ==========")
    # pente par bras
    for arm in ARMS:
        sub = g[g["arm"] == arm]
        f = ols("thickness_mean ~ age + sex_bin + agitation", sub).fit()
        print(f"  {arm:8s} (n={len(sub)}) : pente agitation = {f.params['agitation']:+.4f} mm/mm "
              f"(p={f.pvalues['agitation']:.3g}), epaisseur moy = {sub['thickness_mean'].mean():.3f} mm")

    # modele compare
    fit = ols("thickness_mean ~ age + sex_bin + agitation * arm", g).fit()
    inter = [k for k in fit.params.index if "agitation:" in k][0]
    armk = [k for k in fit.params.index if k.startswith("arm[") and ":" not in k][0]
    print("\n  -- Modele compare : thickness ~ age + sexe + agitation * arm --")
    print(f"    pente preproc       : {fit.params['agitation']:+.4f} mm/mm (p={fit.pvalues['agitation']:.3g})")
    print(f"    OFFSET jdac (arm)   : {fit.params[armk]:+.4f} mm        (p={fit.pvalues[armk]:.3g})")
    print(f"    DELTA pente (inter) : {fit.params[inter]:+.4f} mm/mm    (p={fit.pvalues[inter]:.3g})  <-- test JDAC")
    pente_jdac = fit.params["agitation"] + fit.params[inter]
    print(f"    => pente jdac = {pente_jdac:+.4f} mm/mm (preproc {fit.params['agitation']:+.4f})")
    if fit.pvalues[inter] < 0.05 and abs(pente_jdac) < abs(fit.params["agitation"]):
        print("    INTERPRETATION : pente aplatie significativement -> JDAC corrige (partiellement) le mouvement.")
    else:
        print("    INTERPRETATION : pente inchangee -> JDAC decale/lisse uniformement, ne corrige pas le mouvement.")
    return g


def main():
    print("Chargement...")
    df = load_data()

    print("\nGLM par bras (pente Agitation par region)...")
    par_bras = glm_par_bras(df)
    OUT_BRAS.parent.mkdir(parents=True, exist_ok=True)
    par_bras.to_csv(OUT_BRAS, index=False)
    for arm in ARMS:
        m = par_bras["arm"] == arm
        print(f"  {arm:8s}: {(par_bras.loc[m,'p_fdr']<0.05).sum()}/{m.sum()} regions FDR<0.05, "
              f"{(par_bras.loc[m,'beta_agitation']<0).sum()}/{m.sum()} pentes negatives, "
              f"pente mediane = {par_bras.loc[m,'beta_agitation'].median():+.4f}")

    print("\nGLM interaction (changement de pente du a JDAC, par region)...")
    inter = glm_interaction(df)
    inter.to_csv(OUT_INTER, index=False)
    print(f"  Regions interaction FDR<0.05 : {(inter['p_fdr']<0.05).sum()}/{len(inter)}")
    print(f"  delta_pente median = {inter['delta_pente'].median():+.4f} mm/mm")

    resume_global(df)
    print(f"\nCSV : {OUT_BRAS}\n      {OUT_INTER}")


if __name__ == "__main__":
    main()
