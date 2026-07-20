#!/usr/bin/env python3
"""Génère le notebook volumétrique FreeSurfer consacré à aseg.stats."""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("explore_aseg_rigide.ipynb")


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))

    md("""# Volumes sous-corticaux après JDAC et ses variantes

Ce notebook répond uniquement à trois questions :

1. Les traitements modifient-ils le `run-01` alors qu'il n'y a presque aucun mouvement à corriger ?
2. Les scans bougés se rapprochent-ils du `raw/run-01` du même sujet ?
3. Le volume mesuré reste-t-il associé au score Agitation ?

Toutes les mesures sont des **volumes en mm³** extraits de `aseg.stats`. Le `raw/run-01` est une référence opérationnelle, pas une vérité anatomique parfaite.""")

    code("""from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

CONDITIONS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
SHORT = {"brut":"brut", "preproc":"prep", "jdac":"jdac",
         "jdac_antiartonly":"aa×1", "jdac_nodenoise":"aa×4"}
REPO = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "results/ds004332").exists()), Path.home() / "Documents/GitHub/jdac-motion-correction")
ASEG_DIR = Path.home() / "Documents/derivatives/ds004332/aseg_stats"
ASEG_FILES = {
    "brut": "aseg_brut.csv",
    "preproc": "aseg_preproc.csv",
    "jdac": "aseg_jdac.csv",
    "jdac_antiartonly": "aseg_antiartonly.csv",
    "jdac_nodenoise": "aseg_nodenoise.csv",
}
AGIT = REPO / "results/ds004332/agitation/ds004332_agitation_clinica.csv"

missing = [str(ASEG_DIR / name) for name in ASEG_FILES.values()
           if not (ASEG_DIR / name).is_file()]
if missing:
    raise FileNotFoundError("Tables aseg absentes :\\n" + "\\n".join(missing))

def load_aseg_table(condition, filename):
    # Les fichiers portent l'extension .csv mais sont séparés par tabulations.
    wide = pd.read_csv(ASEG_DIR / filename, sep="\t")
    id_col = "Measure:volume" if "Measure:volume" in wide.columns else wide.columns[0]
    identifiers = wide[id_col].astype(str).str.extract(
        r"(sub-[^_/\\s]+)_(run-\\d+)", expand=True
    )
    if identifiers.isna().any().any():
        bad = wide.loc[identifiers.isna().any(axis=1), id_col].head().tolist()
        raise ValueError(f"Identifiants sujet/run non reconnus dans {filename}: {bad}")
    long = wide.drop(columns=id_col).assign(
        subject=identifiers[0].values,
        run=identifiers[1].values,
    ).melt(id_vars=["subject", "run"], var_name="measure", value_name="value_mm3")
    long["condition"] = condition
    long["source_file"] = str(ASEG_DIR / filename)
    return long

d = pd.concat(
    [load_aseg_table(condition, filename)
     for condition, filename in ASEG_FILES.items()],
    ignore_index=True,
)
d["value_mm3"] = pd.to_numeric(d["value_mm3"], errors="coerce")
d["condition"] = pd.Categorical(d["condition"], CONDITIONS, ordered=True)
agit = pd.read_csv(AGIT).rename(columns={"sub":"subject", "condition":"run", "motion":"agitation"})
d = d.merge(agit[["subject", "run", "agitation"]], on=["subject", "run"], how="left")
print(f"Dossier source : {ASEG_DIR}")
print(f"{len(d):,} mesures | {d['measure'].nunique()} volumes | {d[['subject','run','condition']].drop_duplicates().shape[0]} acquisitions")""")

    md("""## 0. Complétude et contrôle des données

Une acquisition n'est analysée que si son `aseg.stats` existe. Les valeurs manquantes, les doublons et l'eTIV sont contrôlés avant les modèles. L'eTIV est conservé comme résultat : une variation de l'eTIV entre conditions peut elle-même révéler une instabilité de reconstruction.""")

    code("""qc = pd.DataFrame({
    "n acquisitions": d.groupby("condition", observed=True).apply(
        lambda x: x[["subject","run"]].drop_duplicates().shape[0]),
    "n mesures": d.groupby("condition", observed=True)["measure"].nunique(),
    "valeurs manquantes": d.groupby("condition", observed=True)["value_mm3"].apply(lambda x: x.isna().sum()),
    "doublons": d.groupby("condition", observed=True).apply(
        lambda x: x.duplicated(["subject","run","measure"]).sum()),
})
display(qc)
etiv = d[d["measure"].isin(["eTIV", "EstimatedTotalIntraCranialVol"])]
if len(etiv):
    display(etiv.groupby(["condition","run"], observed=True)["value_mm3"].agg(["count","mean","std"]))""")

    md("""## A. Effet propre du traitement sur le scan immobile

Pour chaque structure et chaque sujet :

[
\\Delta_{s,c}=V_{s,run01,c}-V_{s,run01,brut}
]

Une différence non nulle indique que le traitement ou la reconstruction FreeSurfer modifie le volume mesuré sur la référence immobile. Les tests sont appariés et les p-values sont corrigées par FDR sur l'ensemble des structures et conditions.""")

    code("""wide01 = d[d["run"]=="run-01"].pivot_table(
    index=["subject","measure"], columns="condition", values="value_mm3", observed=True)
rows = []
for condition in CONDITIONS[1:]:
    pairs = wide01[["brut", condition]].dropna()
    for measure, x in pairs.groupby(level="measure"):
        delta = x[condition] - x["brut"]
        try:
            p = stats.wilcoxon(delta).pvalue if len(delta) >= 3 and np.any(delta != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append({"condition":condition, "measure":measure, "n":len(delta),
                     "delta_mean_mm3":delta.mean(), "delta_median_mm3":delta.median(),
                     "delta_median_pct":100*np.median(delta/x["brut"]), "p":p})
a = pd.DataFrame(rows)
valid = a["p"].notna()
a.loc[valid, "p_fdr"] = multipletests(a.loc[valid, "p"], method="fdr_bh")[1]
display(a.sort_values("delta_median_pct", key=abs, ascending=False).head(25))

heat = a.pivot(index="measure", columns="condition", values="delta_median_pct")
top = heat.abs().max(axis=1).nlargest(30).index
plt.figure(figsize=(10, 11))
sns.heatmap(heat.loc[top], center=0, cmap="vlag", annot=True, fmt=".1f",
            cbar_kws={"label":"variation médiane vs raw/run-01 (%)"})
plt.title("Structures les plus modifiées sur le run-01")
plt.xlabel(""); plt.ylabel("")
plt.tight_layout(); plt.show()""")

    md("""## B. Récupération des volumes des scans bougés

Pour chaque structure, le traitement est comparé au `raw/run-01` du même sujet.

[
G=|V_{raw,run}-V_{raw,run01}|-|V_{condition,run}-V_{raw,run01}|
]

- (G>0) : le traitement rapproche le volume de la référence ;
- (G<0) : il l'en éloigne.

Une amélioration moyenne ne suffit pas : la carte par structure permet de détecter les erreurs qui s'annulent entre régions.""")

    code("""ref = (d[(d["condition"]=="brut") & (d["run"]=="run-01")]
       [["subject","measure","value_mm3"]].rename(columns={"value_mm3":"reference_mm3"}))
moved = d[d["run"].isin(["run-02","run-03"])].merge(ref, on=["subject","measure"], how="inner")
moved["error_mm3"] = (moved["value_mm3"]-moved["reference_mm3"]).abs()
raw_error = (moved[moved["condition"]=="brut"]
             [["subject","run","measure","error_mm3"]].rename(columns={"error_mm3":"raw_error_mm3"}))
moved = moved.merge(raw_error, on=["subject","run","measure"], how="inner")
moved["gain_mm3"] = moved["raw_error_mm3"] - moved["error_mm3"]

summary_b = moved.groupby("condition", observed=True).agg(
    n=("gain_mm3","size"), gain_median_mm3=("gain_mm3","median"),
    gain_mean_mm3=("gain_mm3","mean"), fraction_amelioree=("gain_mm3",lambda x:(x>0).mean()))
display(summary_b)

b = (moved[moved["condition"]!="brut"].groupby(["condition","measure"], observed=True)["gain_mm3"]
     .agg(["count","mean","median"]).reset_index())
display(b.sort_values("median", ascending=False).groupby("condition", observed=True).head(8))""")

    md("""## C. Le mouvement prédit-il encore chaque volume ?

Pour chaque condition et chaque mesure, le modèle principal est :

[
Volume sim Agitation + (1|sujet)
]

L'intercept aléatoire absorbe les différences stables de taille entre sujets. Le coefficient Agitation est exprimé en **mm³ par point Agitation**. Une correction FDR est appliquée séparément dans chaque condition.""")

    code("""rows = []
for condition in CONDITIONS:
    dc = d[d["condition"]==condition].dropna(subset=["agitation","value_mm3"])
    for measure, x in dc.groupby("measure", observed=True):
        if len(x) < 20 or x["subject"].nunique() < 8:
            continue
        try:
            fit = smf.mixedlm("value_mm3 ~ agitation", x, groups=x["subject"]).fit(
                reml=False, method="lbfgs", disp=False)
            coef, p = fit.params["agitation"], fit.pvalues["agitation"]
            lo, hi = fit.conf_int().loc["agitation"]
            method = "mixedlm"
        except Exception:
            fit = smf.ols("value_mm3 ~ agitation + C(subject)", x).fit()
            coef, p = fit.params["agitation"], fit.pvalues["agitation"]
            lo, hi = fit.conf_int().loc["agitation"]
            method = "ols_sujet_fixe"
        rows.append({"condition":condition, "measure":measure, "n":len(x),
                     "coef_mm3_par_point":coef, "ic95_bas":lo, "ic95_haut":hi,
                     "p":p, "methode":method})
c = pd.DataFrame(rows)
c["p_fdr"] = c.groupby("condition", observed=True)["p"].transform(
    lambda p: multipletests(p, method="fdr_bh")[1])
display(c.sort_values(["condition","p_fdr"]).groupby("condition", observed=True).head(12))""")

    md("""## Structures centrales à examiner visuellement

La zone entourée se situe autour du troisième ventricule et des thalamus médiaux. `aseg` ne possède pas nécessairement un label autonome pour l'adhérence interthalamique. La disparition visuelle doit donc être séparée d'une variation du label ou du volume segmenté.""")

    code("""central = [
    "3rd-Ventricle", "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent", "CSF"
]
central_a = a[a["measure"].isin(central)].copy()
display(central_a.sort_values(["measure","condition"]))
if len(central_a):
    plt.figure(figsize=(11, 5))
    sns.pointplot(data=central_a, x="measure", y="delta_median_pct",
                  hue="condition", dodge=True)
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.ylabel("variation médiane du run-01 (%)")
    plt.xlabel(""); plt.xticks(rotation=35, ha="right")
    plt.title("Structures centrales : effet propre du traitement")
    plt.tight_layout(); plt.show()""")

    md("""## Conclusion à remplir après exécution

- **Stabilité du run-01 :** quelles structures sont modifiées sans mouvement ?
- **Récupération :** quelles conditions rapprochent réellement les runs bougés de la référence ?
- **Mouvement résiduel :** quelles mesures restent associées à Agitation après FDR ?
- **Zone centrale :** le changement est-il un contraste visuel, un volume différent ou un label différent ?

Aucune amélioration globale ne sera appelée « restauration anatomique » sans cohérence entre ces quatre lectures.""")

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name":"Python (cortical-motion)",
                                      "language":"python", "name":"python3"}
    nbf.write(nb, OUT)
    print(f"Notebook créé: {OUT}")


if __name__ == "__main__":
    build()
