"""
GLM Pipeline B -- ds004332 -- sorties FreeSurfer recon-all (et non FastSurfer).

================================================================================
BUT
================================================================================
Quantifier l'effet du mouvement (score Agitation) sur l'epaisseur corticale
mesuree par FreeSurfer, en controlant age et sexe. C'est le "Pipeline B" :

    images brutes -> FreeSurfer recon-all -> epaisseur ~ age + sexe + Agitation + (1|sujet)

Pipeline B sert de reference (mouvement non corrige). Il sera compare a Pipeline A
(raw -> JDAC -> FreeSurfer) : si JDAC corrige le biais de mouvement, les betas
age/sexe doivent etre identiques entre A et B.

Ce script reprend la METHODE de run_glm_pipeline_b.py (par region + modele mixte
+ FDR), mais lit les sorties FreeSurfer recon-all (ThickAvg_phase1_complete.csv)
au lieu des sorties FastSurfer (cortical_thickness.csv, non retenues).

================================================================================
CONSTRUCTION DU MODELE (general)
================================================================================
Pour CHAQUE region corticale (34 DKT x 2 hemispheres) on ajuste :

    thickness_mean ~ age + sexe + agitation   avec un intercept aleatoire par sujet

- "thickness_mean" : epaisseur moyenne de la region pour un (sujet, run) donne.
- "agitation"      : score de mouvement Agitation (mm), mesure sur l'image, varie
                     par run (still / nodding / shaking).
- "age", "sexe"    : constantes par sujet.
- "(1|sujet)"      : INTERCEPT ALEATOIRE par sujet (voir note ci-dessous).

Note "(1|sujet)" : chaque sujet a 3 runs (mesures repetees). Les 3 runs d'un meme
sujet sont correles (un cerveau naturellement epais l'est dans ses 3 runs). Le
terme (1|sujet) donne a chaque sujet sa propre ligne de base et estime l'effet
du mouvement A L'INTERIEUR des sujets, sans gonfler la confiance en traitant les
66 mesures comme independantes. La variance de ces lignes de base = "Group Var".
Si Group Var ~ 0, il ne reste plus de difference inter-sujets une fois age/sexe/
agitation pris en compte : le modele mixte se reduit alors a une regression simple
(les betas restent valides). Quand le modele mixte ne converge pas, on bascule sur
un OLS a effets fixes sujet (fallback), comme dans le script FastSurfer original.

Correction FDR (Benjamini-Hochberg) : on teste 62 regions, donc on corrige les
p-values pour le test multiple.

================================================================================
SORTIES
================================================================================
- CSV par region : results/ds004332/results/glm_pipeline_b_freesurfer_results.csv
- Resume global "style Charles" (epaisseur moyenne globale ~ age+sexe+agitation,
  effets fixes) imprime en fin, pour le narratif "gain statistique d'Agitation".

Usage :
    conda run -n cortical-motion python3 \
        ~/Documents/motion-analysis/pipelines/ds004332/run_glm_pipeline_b_freesurfer.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")  # masque les ConvergenceWarning (geres par le fallback)

# ------------------------------------------------------------------------------
# CHEMINS (memes sources que ta version FastSurfer, sauf l'epaisseur = FreeSurfer)
# ------------------------------------------------------------------------------
HOME = Path("/home/av62870@ens.ad.etsmtl.ca")
# Epaisseur : sorties FreeSurfer recon-all (ThickAvg par region, 66 runs)
THICKNESS_CSV = HOME / "Documents/Results/freesurfer_ds004332/ThickAvg_phase1_complete.csv"
# Agitation Clinica : score de mouvement par sujet x run (identique a la version FastSurfer)
AGITATION_CSV = HOME / "Documents/motion-analysis/results/ds004332/results/ds004332_agitation_clinica.csv"
# Demographics BIDS
PARTICIPANTS = HOME / "Documents/Datasets/ds004332/participants.tsv"
# Sortie, dans TON dossier de resultats ds004332
OUTPUT_CSV = HOME / "Documents/motion-analysis/results/ds004332/results/glm_pipeline_b_freesurfer_results.csv"


def load_data():
    """Charge et fusionne epaisseur (FreeSurfer) + Agitation + demographics."""
    # --- Epaisseur FreeSurfer : colonnes subject(=sub-01_run-01), hemi, region, ThickAvg ---
    thickness = pd.read_csv(THICKNESS_CSV)
    # Le champ "subject" combine sujet et run : on le decoupe.
    thickness["run"] = thickness["subject"].str.split("_").str[1]       # run-01 / run-02 / run-03
    thickness["subject"] = thickness["subject"].str.split("_").str[0]   # sub-01
    # On aligne le nom de colonne sur la methode FastSurfer.
    thickness = thickness.rename(columns={"ThickAvg": "thickness_mean"})

    # Echecs silencieux FreeSurfer : epaisseur = 0 partout (ex. sub-01_run-03, CortexVol=0).
    # On retire ces lignes (sinon des 0 fausseraient les regressions).
    n_before = len(thickness)
    thickness = thickness[thickness["thickness_mean"] > 0].copy()
    n_drop = n_before - len(thickness)

    # --- Agitation : condition->run, sub->subject, motion->agitation ---
    agitation = pd.read_csv(AGITATION_CSV).rename(
        columns={"condition": "run", "sub": "subject", "motion": "agitation"}
    )

    # --- Demographics : sexe binaire (F=1) comme dans ta version ---
    demog = pd.read_csv(PARTICIPANTS, sep="\t").rename(columns={"participant_id": "subject"})
    demog["sex_bin"] = (demog["sex"] == "F").astype(int)

    # --- Fusions ---
    df = thickness.merge(agitation, on=["subject", "run"], how="inner")
    df = df.merge(demog[["subject", "age", "sex_bin"]], on="subject", how="left")

    print(f"Lignes epaisseur retenues : {len(thickness)} (retire {n_drop} lignes a 0 = echec silencieux)")
    print(f"Observations apres merge : {len(df)}")
    print(f"Sujets : {df['subject'].nunique()} | Runs : {sorted(df['run'].unique())}")
    print(f"Agitation -- min={df['agitation'].min():.3f}  max={df['agitation'].max():.3f}  "
          f"mean={df['agitation'].mean():.3f} mm")
    return df


def fit_region(sub_df):
    """Ajuste, pour une region, thickness_mean ~ age + sexe + agitation + (1|sujet).

    Modele mixte (REML, optimiseur powell). Si pas de convergence, fallback OLS a
    effets fixes sujet (age/sexe deviennent colineaires avec sujet, on ne garde
    donc qu'agitation + C(subject)). Renvoie (beta_agitation, se, p, type_modele).
    """
    try:
        model = mixedlm("thickness_mean ~ age + sex_bin + agitation",
                        sub_df, groups=sub_df["subject"])
        fit = model.fit(reml=True, method="powell", disp=False)
        if fit.converged:
            return (fit.params["agitation"], fit.bse["agitation"],
                    fit.pvalues["agitation"], "mixedlm")
    except Exception:
        pass
    # Fallback : OLS within-subject (effets fixes sujet)
    fit = ols("thickness_mean ~ agitation + C(subject)", sub_df).fit()
    return (fit.params["agitation"], fit.bse["agitation"],
            fit.pvalues["agitation"], "ols_fe")


def run_glm(df):
    """Ajuste un modele par region et renvoie un tableau de resultats."""
    regions = df[["hemi", "region"]].drop_duplicates().sort_values(["hemi", "region"])
    results = []
    for _, row in regions.iterrows():
        hemi, region = row["hemi"], row["region"]
        sub_df = df[(df["hemi"] == hemi) & (df["region"] == region)].copy()
        n_obs, n_subs = len(sub_df), sub_df["subject"].nunique()

        # On saute les regions sans variabilite exploitable.
        if n_subs < 5 or sub_df["agitation"].std() == 0:
            results.append(dict(hemi=hemi, region=region, beta=np.nan, se=np.nan,
                                p_value=np.nan, model="skip", n_obs=n_obs, n_subjects=n_subs))
            continue
        try:
            beta, se, pval, mtype = fit_region(sub_df)
            results.append(dict(hemi=hemi, region=region, beta=beta, se=se,
                                p_value=pval, model=mtype, n_obs=n_obs, n_subjects=n_subs))
        except Exception as e:
            print(f"  ERREUR {hemi} {region}: {e}")
            results.append(dict(hemi=hemi, region=region, beta=np.nan, se=np.nan,
                                p_value=np.nan, model="error", n_obs=n_obs, n_subjects=n_subs))
    return pd.DataFrame(results)


def global_summary(df):
    """Resume "style Charles" : epaisseur moyenne globale par run ~ age+sexe+agitation,
    effets fixes (OLS). Donne le coefficient agitation, sa p-value et le R2 -> gain."""
    g = (df.groupby(["subject", "run"])
           .agg(thickness_mean=("thickness_mean", "mean"),
                age=("age", "first"), sex_bin=("sex_bin", "first"),
                agitation=("agitation", "first"))
           .reset_index())
    fit = ols("thickness_mean ~ age + sex_bin + agitation", g).fit()
    return g, fit


def main():
    df = load_data()

    # --- GLM par region + FDR ---
    n_regions = df[["hemi", "region"]].drop_duplicates().shape[0]
    print(f"\nAjustement de {n_regions} regions...")
    results = run_glm(df)
    print(f"Modeles utilises : {results['model'].value_counts().to_dict()}")

    valid = results["p_value"].notna()
    results.loc[valid, "p_fdr"] = multipletests(results.loc[valid, "p_value"],
                                                 method="fdr_bh")[1]
    results = results.sort_values("p_value")

    n_valid = int(valid.sum())
    print(f"Regions p<0.05 non corrige : {(results['p_value'] < 0.05).sum()}/{n_valid}")
    print(f"Regions p_FDR<0.05         : {(results['p_fdr'] < 0.05).sum()}/{n_valid}")
    neg = (results["beta"] < 0).sum()
    print(f"Beta negatif (mouvement -> epaisseur reduite) : {neg}/{n_valid}")

    print("\n-- Top 12 regions (p non corrige) --")
    cols = ["hemi", "region", "beta", "se", "p_value", "p_fdr", "model", "n_obs"]
    print(results[cols].head(12).to_string(index=False, float_format="{:.4g}".format))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultats par region : {OUTPUT_CSV}")

    # --- Resume global style Charles ---
    g, gfit = global_summary(df)
    print("\n========== RESUME GLOBAL (style Charles, effets fixes) ==========")
    print(f"epaisseur moyenne globale ~ age + sexe + agitation   (n={len(g)} runs, OLS)")
    print(f"  agitation : beta = {gfit.params['agitation']:+.4f} mm/mm  "
          f"(p = {gfit.pvalues['agitation']:.3g})")
    print(f"  age       : beta = {gfit.params['age']:+.4f}  (p = {gfit.pvalues['age']:.3g})")
    print(f"  sex_bin(F): beta = {gfit.params['sex_bin']:+.4f}  (p = {gfit.pvalues['sex_bin']:.3g})")
    print(f"  R2 = {gfit.rsquared:.3f}  |  log-vraisemblance = {gfit.llf:.1f}")


if __name__ == "__main__":
    main()
