"""
Comparaison 3 bras (brute / preproc / jdac) -- ds004332 -- effet de JDAC sur le
biais de mouvement en epaisseur corticale.

================================================================================
QUESTION (bien posee)
================================================================================
Pour chaque sujet s, run r (01 still / 02 nodding / 03 shaking), bras a :
  - brute   : image brute -> FreeSurfer                       (reference non corrigee)
  - preproc : cerveau (N4 + SynthStrip) -> FreeSurfer
  - jdac    : cerveau -> JDAC -> FreeSurfer
Epaisseur T(s,r,a). Agitation m(s,r) (mouvement mesure, partage entre bras).

Deux effets a distinguer :
  - OFFSET : JDAC (ou preproc) decale-t-il l'epaisseur de facon uniforme ?
  - PENTE  : change-t-il dT/dm, le LIEN mouvement -> epaisseur ?
JDAC "corrige le mouvement" seulement si la PENTE s'aplatit (moins negative),
independamment de l'offset.

================================================================================
EXPERIENCES
================================================================================
E1 -- Descriptif par bras x condition : epaisseur moyenne et % brute->preproc/jdac.
E2 -- Pente intra-sujet : pente T~m par sujet et par bras (3 runs), comparee
      entre bras (Wilcoxon apparie). Test intra-sujet direct.
E3 -- Modele lineaire mixte (LMM), test formel :
      T ~ age + sexe + agitation * arm + (1|sujet), reference = brute.
      arm = offset ; agitation:arm = changement de pente (le verdict). Par region + FDR.

================================================================================
ENTREES
================================================================================
- brute   : results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv (long ; ThickAvg)
            sub-01_run-03 = echec silencieux (ThickAvg=0) -> retire.
- preproc/jdac : derivatives/ds004332/thickness_{arm}_{lh,rh}.csv (aparcstats2table large)
- agitation : results/ds004332/agitation/ds004332_agitation_clinica.csv
- demographics : raw_datasets/ds004332/participants.tsv

Epaisseur globale par run = moyenne NON ponderee des 68 regions (meme methode pour
les 3 bras, donc comparable ; differe legerement de la MeanThickness ponderee de FS).

SORTIES (results/ds004332/phase4_compare_3bras/)
- e1_par_condition.csv, e2_pentes_sujet.csv, e3_interaction_regions.csv
- resume imprime.

Usage : conda run -n cortical-motion python3 compare_3bras.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

HOME = Path("/home/av62870@ens.ad.etsmtl.ca")
REPO = HOME / "Documents/jdac-motion-correction"
DERIV = HOME / "Documents/derivatives/ds004332"
BRUTE_CSV = REPO / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv"
AGIT_CSV = REPO / "results/ds004332/agitation/ds004332_agitation_clinica.csv"
PARTICIPANTS = HOME / "Documents/raw_datasets/ds004332/participants.tsv"
OUTDIR = REPO / "results/ds004332/phase4_compare_3bras"

COND = {"run-01": "still", "run-02": "nodding", "run-03": "shaking"}


# ------------------------------------------------------------------------------
# Chargement : 3 bras -> long (subject, run, hemi, region, thickness, arm)
# ------------------------------------------------------------------------------
def load_brute():
    d = pd.read_csv(BRUTE_CSV)
    d = d[d["ThickAvg"] > 0].copy()                      # retire sub-01_run-03 (=0)
    d["run"] = d["subject"].str.split("_").str[1]
    d["subject"] = d["subject"].str.split("_").str[0]
    d = d.rename(columns={"ThickAvg": "thickness"})
    d["arm"] = "brute"
    return d[["subject", "run", "hemi", "region", "thickness", "arm"]]


def load_wide(arm):
    frames = []
    for hemi in ["lh", "rh"]:
        w = pd.read_csv(DERIV / f"thickness_{arm}_{hemi}.csv", sep="\t")
        w = w.rename(columns={w.columns[0]: "id"})
        cols = [c for c in w.columns if c.endswith("_thickness") and "MeanThickness" not in c]
        long = w.melt(id_vars="id", value_vars=cols, var_name="rr", value_name="thickness")
        long["hemi"] = hemi
        long["region"] = long["rr"].str.replace(f"{hemi}_", "", regex=False).str.replace("_thickness", "", regex=False)
        frames.append(long)
    d = pd.concat(frames, ignore_index=True)
    d["subject"] = d["id"].str.split("_").str[0]
    d["run"] = d["id"].str.split("_").str[1]
    d["arm"] = arm
    return d[["subject", "run", "hemi", "region", "thickness", "arm"]]


def load_data():
    thick = pd.concat([load_brute(), load_wide("preproc"), load_wide("jdac")], ignore_index=True)
    thick = thick[thick["thickness"] > 0].copy()

    agit = pd.read_csv(AGIT_CSV).rename(columns={"condition": "run", "sub": "subject", "motion": "agitation"})
    demog = pd.read_csv(PARTICIPANTS, sep="\t").rename(columns={"participant_id": "subject"})
    demog["sex_bin"] = (demog["sex"] == "F").astype(int)

    df = thick.merge(agit[["subject", "run", "agitation"]], on=["subject", "run"], how="inner")
    df = df.merge(demog[["subject", "age", "sex_bin"]], on="subject", how="left")
    df["condition"] = df["run"].map(COND)

    print("Runs par bras :")
    for a in ["brute", "preproc", "jdac"]:
        sub = df[df["arm"] == a]
        print(f"  {a:8s}: {sub.groupby(['subject','run']).ngroups} runs, "
              f"{sub.groupby('region').ngroups} regions")
    return df


def global_par_run(df):
    """Epaisseur globale (moyenne non ponderee des regions) par sujet x run x bras."""
    g = (df.groupby(["subject", "run", "condition", "arm"])
           .agg(thickness=("thickness", "mean"),
                agitation=("agitation", "first"),
                age=("age", "first"), sex_bin=("sex_bin", "first"))
           .reset_index())
    return g


# ------------------------------------------------------------------------------
# E1 -- descriptif par bras x condition
# ------------------------------------------------------------------------------
def e1(g):
    piv = g.pivot_table(index="condition", columns="arm", values="thickness", aggfunc="mean")
    piv = piv.reindex(["still", "nodding", "shaking"])
    piv["pct_brute_preproc"] = 100 * (piv["preproc"] - piv["brute"]) / piv["brute"]
    piv["pct_brute_jdac"] = 100 * (piv["jdac"] - piv["brute"]) / piv["brute"]
    piv.round(3).to_csv(OUTDIR / "e1_par_condition.csv")
    print("\n===== E1 : epaisseur moyenne par bras x condition (mm) =====")
    print(piv.round(3).to_string())
    print("\nMoyenne globale par bras :")
    print(g.groupby("arm")["thickness"].mean().round(3).to_string())
    return piv


# ------------------------------------------------------------------------------
# E2 -- pente intra-sujet (T ~ agitation sur les 3 runs), par bras
# ------------------------------------------------------------------------------
def e2(g):
    rows = []
    for (subj, arm), sub in g.groupby(["subject", "arm"]):
        if len(sub) < 2 or sub["agitation"].std() == 0:
            continue
        sl, inter, r, p, se = stats.linregress(sub["agitation"], sub["thickness"])
        rows.append(dict(subject=subj, arm=arm, slope=sl, n_runs=len(sub)))
    pentes = pd.DataFrame(rows)
    pentes.round(4).to_csv(OUTDIR / "e2_pentes_sujet.csv", index=False)

    print("\n===== E2 : pente intra-sujet T~agitation (mm par mm), par bras =====")
    print(pentes.groupby("arm")["slope"].agg(["count", "mean", "median"]).round(4).to_string())

    # tests apparies (sujets presents dans les 2 bras compares)
    wide = pentes.pivot(index="subject", columns="arm", values="slope")
    print("\nTests apparies (Wilcoxon) sur les pentes par sujet :")
    for a, b in [("brute", "jdac"), ("preproc", "jdac"), ("brute", "preproc")]:
        pair = wide[[a, b]].dropna()
        if len(pair) >= 5:
            try:
                w, p = stats.wilcoxon(pair[a], pair[b])
                print(f"  {a} vs {b} (n={len(pair)}) : pente med {a}={pair[a].median():+.4f}, "
                      f"{b}={pair[b].median():+.4f}, p={p:.3g}")
            except Exception as e:
                print(f"  {a} vs {b} : {e}")
    return pentes


# ------------------------------------------------------------------------------
# E3 -- LMM interaction agitation x arm (reference = brute)
# ------------------------------------------------------------------------------
def e3_global(g):
    g = g.copy()
    g["arm"] = pd.Categorical(g["arm"], categories=["brute", "preproc", "jdac"])
    print("\n===== E3 : modele global  T ~ age + sexe + agitation * arm + (1|sujet) =====")
    try:
        fit = mixedlm("thickness ~ age + sex_bin + agitation * C(arm, Treatment('brute'))",
                      g, groups=g["subject"]).fit(reml=True, method="powell", disp=False)
    except Exception:
        fit = ols("thickness ~ age + sex_bin + agitation * C(arm, Treatment('brute'))", g).fit()

    def get(key):
        for k in fit.params.index:
            if key in k:
                return fit.params[k], fit.pvalues[k]
        return np.nan, np.nan

    sl_brute, p_sl = fit.params.get("agitation", np.nan), fit.pvalues.get("agitation", np.nan)
    off_pre, p_off_pre = get("T.preproc]") if False else get("arm, Treatment('brute'))[T.preproc]")
    off_jd, p_off_jd = get("arm, Treatment('brute'))[T.jdac]")
    int_pre, p_int_pre = get("agitation:C(arm, Treatment('brute'))[T.preproc]")
    int_jd, p_int_jd = get("agitation:C(arm, Treatment('brute'))[T.jdac]")

    print(f"  pente brute            : {sl_brute:+.4f} mm/mm (p={p_sl:.3g})")
    print(f"  offset preproc         : {off_pre:+.4f} mm    (p={p_off_pre:.3g})")
    print(f"  offset jdac            : {off_jd:+.4f} mm    (p={p_off_jd:.3g})")
    print(f"  delta pente preproc    : {int_pre:+.4f} mm/mm (p={p_int_pre:.3g})")
    print(f"  delta pente jdac       : {int_jd:+.4f} mm/mm (p={p_int_jd:.3g})  <-- test JDAC")
    print(f"  => pentes : brute {sl_brute:+.4f} | preproc {sl_brute+int_pre:+.4f} | jdac {sl_brute+int_jd:+.4f}")
    if p_int_jd < 0.05 and abs(sl_brute + int_jd) < abs(sl_brute):
        print("  INTERPRETATION : pente jdac aplatie significativement -> JDAC reduit l'impact du mouvement.")
    else:
        print("  INTERPRETATION : pente jdac non distincte de brute -> JDAC ne corrige pas le mouvement (offset/lissage).")


def e3_regions(df):
    rows = []
    for (hemi, region), sub in df.groupby(["hemi", "region"]):
        if sub["arm"].nunique() < 3 or sub["agitation"].std() == 0:
            continue
        sub = sub.copy()
        sub["arm"] = pd.Categorical(sub["arm"], categories=["brute", "preproc", "jdac"])
        try:
            fit = mixedlm("thickness ~ age + sex_bin + agitation * C(arm, Treatment('brute'))",
                          sub, groups=sub["subject"]).fit(reml=True, method="powell", disp=False)
        except Exception:
            continue
        key = "agitation:C(arm, Treatment('brute'))[T.jdac]"
        if key in fit.params.index:
            rows.append(dict(hemi=hemi, region=region, delta_pente_jdac=fit.params[key],
                             p_inter=fit.pvalues[key]))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_fdr"] = multipletests(res["p_inter"], method="fdr_bh")[1]
        res = res.sort_values("p_inter")
        res.round(4).to_csv(OUTDIR / "e3_interaction_regions.csv", index=False)
        print(f"\n  Par region (interaction jdac vs brute) : {(res['p_fdr']<0.05).sum()}/{len(res)} regions FDR<0.05, "
              f"delta_pente median = {res['delta_pente_jdac'].median():+.4f}")
    return res


def figures(g, pentes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["brute", "preproc", "jdac"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) epaisseur par condition, 3 bras
    for arm in order:
        m = g[g["arm"] == arm].groupby("condition")["thickness"].mean().reindex(["still", "nodding", "shaking"])
        ax[0].plot(["still", "nodding", "shaking"], m.values, marker="o", label=arm)
    ax[0].set_ylabel("épaisseur moyenne (mm)"); ax[0].set_title("E1 : épaisseur par condition")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # (b) distribution des pentes par sujet, par bras
    data = [pentes[pentes["arm"] == a]["slope"].values for a in order]
    ax[1].boxplot(data, labels=order, showmeans=True)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylim(-0.8, 0.3)   # borne : pentes/sujet bruitees (3 pts), outliers hors champ
    ax[1].set_ylabel("pente T ~ agitation (mm/mm)")
    ax[1].set_title("E2 : pente mouvement→épaisseur par sujet\n(axe borné, outliers masqués)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Comparaison 3 bras (brute / preproc / jdac) — préliminaire", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(OUTDIR / "compare_3bras.png"), dpi=140)
    print(f"Figure -> {OUTDIR / 'compare_3bras.png'}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    g = global_par_run(df)
    e1(g)
    pentes = e2(g)
    e3_global(g)
    e3_regions(df)
    figures(g, pentes)
    print(f"\nSorties dans {OUTDIR}")


if __name__ == "__main__":
    main()
