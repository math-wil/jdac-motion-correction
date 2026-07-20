#!/usr/bin/env python3
"""Génère le notebook volumétrique FreeSurfer consacré à aseg.stats.

Corrections de lisibilité par rapport à la première version :
- les mesures de qualité (trous de surface, hypointensités, ratios -to-eTIV) sont
  traitées à part des volumes anatomiques, car leur base est quasi nulle et les
  pourcentages y explosent sans signification ;
- les structures dégénérées (volume médian nul : 5th-Ventricle, vessel,
  choroid-plexus, Optic-Chiasm, hypointensités latéralisées) sont retirées ;
- les variations sont exprimées en mm³ et en pourcentage calculé en ratio des
  médianes (robuste aux dénominateurs individuels proches de zéro), avec une
  normalisation eTIV pour comparer entre structures ;
- la correction FDR est calculée partout, y compris en section C.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("explore_aseg_rigide.ipynb")


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))

    md("""# Volumes sous-corticaux après JDAC et ses variantes

Ce notebook répond à trois questions :

1. Les traitements modifient-ils le `run-01` alors qu'il n'y a presque aucun mouvement à corriger ?
2. Les scans bougés se rapprochent-ils du `raw/run-01` du même sujet ?
3. Le volume mesuré reste-t-il associé au score Agitation ?

Les mesures sont des **volumes en mm³** extraits de `aseg.stats`. Le `raw/run-01` est une référence opérationnelle, pas une vérité anatomique parfaite.

Trois précautions de lecture, décidées après une première version illisible :

- **Mesures de qualité séparées.** Les trous de surface (`SurfaceHoles`), les hypointensités et les ratios `-to-eTIV` ne sont pas des volumes anatomiques et ont une base quasi nulle. Exprimés en pourcentage ils explosaient et masquaient le signal réel. Ils sont analysés à part, en unités natives.
- **Structures dégénérées retirées.** Toute mesure dont le volume médian sur `raw/run-01` est nul (`5th-Ventricle`, `vessel`, `choroid-plexus`, `Optic-Chiasm`, hypointensités latéralisées) est écartée : aucune variation exploitable.
- **Pourcentages robustes.** La variation relative est le ratio des médianes (médiane des Δ sur médiane de la base), pas la médiane des rapports, pour éviter qu'un sujet à volume proche de zéro fasse diverger le pourcentage. Une normalisation par l'eTIV du sujet complète la lecture.""")

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
print(f"{len(d):,} mesures | {d['measure'].nunique()} colonnes | {d[['subject','run','condition']].drop_duplicates().shape[0]} acquisitions")""")

    md("""## Classification des mesures

Chaque colonne de `aseg.stats` est rangée dans une catégorie avant toute analyse :

- **Volumes anatomiques** (globaux et régionaux) : seuls concernés par les variations en % ;
- **Mesures de qualité** : trous de surface, hypointensités, ratios `-to-eTIV`, analysées à part ;
- **Dégénérées** : volume médian nul sur `raw/run-01`, retirées.

L'eTIV du `raw/run-01` de chaque sujet sert de normalisateur commun.""")

    code("""ETIV_NAME = "EstimatedTotalIntraCranialVol"
QC_MEASURES = {"SurfaceHoles", "lhSurfaceHoles", "rhSurfaceHoles", "WM-hypointensities"}
RATIO_MEASURES = {"BrainSegVol-to-eTIV", "MaskVol-to-eTIV"}
GLOBAL_MEASURES = {
    "BrainSegVol", "BrainSegVolNotVent", "lhCortexVol", "rhCortexVol", "CortexVol",
    "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol", "CerebralWhiteMatterVol",
    "SubCortGrayVol", "TotalGrayVol", "SupraTentorialVol", "SupraTentorialVolNotVent",
    "MaskVol", ETIV_NAME,
}
MIN_SIZE_MM3 = 100.0   # seuil sous lequel une base est jugée trop petite / dégénérée

# base = volume médian sur le scan de référence (brut, run-01)
baseline = (d[(d["condition"]=="brut") & (d["run"]=="run-01")]
            .groupby("measure")["value_mm3"].median())

# Les mesures de qualité (trous, hypointensités, ratios) restent en catégorie QC
# quelle que soit leur base : le seuil de taille ne trie que les vrais volumes.
NON_ANAT = QC_MEASURES | RATIO_MEASURES | {ETIV_NAME}
qc = sorted((QC_MEASURES | RATIO_MEASURES) & set(baseline.index))
degenerate = sorted([m for m in baseline.index
                     if m not in NON_ANAT and baseline[m] <= MIN_SIZE_MM3])
anat = [m for m in baseline.index
        if m not in NON_ANAT and baseline[m] > MIN_SIZE_MM3]
anat_global = [m for m in anat if m in GLOBAL_MEASURES]
anat_regional = [m for m in anat if m not in GLOBAL_MEASURES]

# eTIV de référence par sujet (brut, run-01) pour normaliser les écarts
etiv_ref = (d[(d["condition"]=="brut") & (d["run"]=="run-01") & (d["measure"]==ETIV_NAME)]
            [["subject", "value_mm3"]].rename(columns={"value_mm3":"etiv_ref"}))
d = d.merge(etiv_ref, on="subject", how="left")

print(f"Volumes anatomiques : {len(anat)}  ({len(anat_global)} globaux, {len(anat_regional)} régionaux)")
print(f"Mesures de qualité analysées à part : {qc}")
print(f"Dégénérées retirées (base médiane ≤ {MIN_SIZE_MM3:.0f} mm³) : {degenerate}")
print(f"eTIV de référence disponible pour {etiv_ref['subject'].nunique()} sujets "
      f"(médiane {etiv_ref['etiv_ref'].median():,.0f} mm³)")""")

    md("""## 0. Complétude et contrôle des données

Une acquisition n'est analysée que si son `aseg.stats` existe. Les valeurs manquantes, les doublons et l'eTIV sont contrôlés avant les modèles. Une variation de l'eTIV entre conditions révélerait déjà une instabilité de reconstruction.""")

    code("""qc_tab = pd.DataFrame({
    "n acquisitions": d.groupby("condition", observed=True).apply(
        lambda x: x[["subject","run"]].drop_duplicates().shape[0]),
    "n mesures": d.groupby("condition", observed=True)["measure"].nunique(),
    "valeurs manquantes": d.groupby("condition", observed=True)["value_mm3"].apply(lambda x: x.isna().sum()),
    "doublons": d.groupby("condition", observed=True).apply(
        lambda x: x.duplicated(["subject","run","measure"]).sum()),
})
display(qc_tab)
etiv = d[d["measure"]==ETIV_NAME]
if len(etiv):
    print("eTIV (mm³) par condition et run :")
    display(etiv.groupby(["condition","run"], observed=True)["value_mm3"].agg(["count","mean","std"]))""")

    md("""## A. Effet propre du traitement sur le scan immobile

Pour chaque structure anatomique et chaque sujet, sur `run-01` :

Δ = V(condition) − V(brut)

Une différence non nulle indique que le traitement ou la reconstruction FreeSurfer modifie le volume mesuré sur la référence immobile. Le pourcentage est le **ratio des médianes** (médiane des Δ divisée par la médiane de la base), robuste aux dénominateurs proches de zéro. La colonne `delta_pct_etiv` rapporte le même Δ à l'eTIV du sujet, pour comparer des structures de tailles très différentes. Les tests sont appariés (Wilcoxon) et corrigés par FDR sur l'ensemble des structures et conditions.""")

    code("""wide01 = d[d["run"]=="run-01"].pivot_table(
    index=["subject","measure"], columns="condition", values="value_mm3", observed=True)
etiv01 = d[d["run"]=="run-01"].drop_duplicates("subject").set_index("subject")["etiv_ref"]
rows = []
for condition in CONDITIONS[1:]:
    pairs = wide01[["brut", condition]].dropna()
    for measure, x in pairs.groupby(level="measure"):
        if measure not in anat:
            continue
        delta = x[condition] - x["brut"]
        base = x["brut"].median()
        subj = x.index.get_level_values("subject")
        etiv_sub = etiv01.reindex(subj).values
        try:
            p = stats.wilcoxon(delta).pvalue if len(delta) >= 3 and np.any(delta != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append({"condition":condition, "measure":measure,
                     "groupe":"global" if measure in anat_global else "régional",
                     "n":len(delta), "baseline_mm3":base,
                     "delta_median_mm3":delta.median(),
                     "delta_pct":100*delta.median()/base,
                     "delta_pct_etiv":100*np.nanmedian(delta.values/etiv_sub), "p":p})
a = pd.DataFrame(rows)
valid = a["p"].notna()
a.loc[valid, "p_fdr"] = multipletests(a.loc[valid, "p"], method="fdr_bh")[1]

print("Volumes GLOBAUX (cortex, matière blanche, volumes totaux) — variation vs brut sur run-01 :")
display(a[a["groupe"]=="global"].sort_values("condition")
        .loc[:, ["condition","measure","baseline_mm3","delta_median_mm3","delta_pct","delta_pct_etiv","p_fdr"]]
        .round({"baseline_mm3":0,"delta_median_mm3":0,"delta_pct":2,"delta_pct_etiv":3,"p_fdr":4}))
print("\\nStructures RÉGIONALES les plus modifiées (|%| décroissant) :")
display(a[a["groupe"]=="régional"].sort_values("delta_pct", key=abs, ascending=False)
        .loc[:, ["condition","measure","baseline_mm3","delta_median_mm3","delta_pct","p_fdr"]]
        .head(25).round({"baseline_mm3":0,"delta_median_mm3":0,"delta_pct":2,"p_fdr":4}))

heat = a[a["groupe"]=="régional"].pivot(index="measure", columns="condition", values="delta_pct")
top = heat.abs().max(axis=1).nlargest(25).index
plt.figure(figsize=(9, 9))
sns.heatmap(heat.loc[top], center=0, cmap="vlag", annot=True, fmt=".1f",
            cbar_kws={"label":"variation médiane vs raw/run-01 (%)"})
plt.title("Structures régionales les plus modifiées sur le run-01")
plt.xlabel(""); plt.ylabel("")
plt.tight_layout(); plt.show()""")

    md("""### A-bis. Mesures de qualité (hors volumes anatomiques)

Trous de surface, hypointensités et ratios `-to-eTIV` ont une base quasi nulle : un pourcentage y est trompeur. Ils sont donc lus en **variation absolue** (unités natives : nombre de trous, mm³ d'hypointensités, ratio sans unité). Un traitement qui crée des trous de surface ou des hypointensités dégrade la reconstruction sans que cela soit une correction du mouvement.""")

    code("""rows = []
for condition in CONDITIONS[1:]:
    pairs = wide01[["brut", condition]].dropna()
    for measure, x in pairs.groupby(level="measure"):
        if measure not in qc:
            continue
        delta = x[condition] - x["brut"]
        try:
            p = stats.wilcoxon(delta).pvalue if len(delta) >= 3 and np.any(delta != 0) else 1.0
        except ValueError:
            p = np.nan
        rows.append({"condition":condition, "measure":measure, "n":len(delta),
                     "brut_median":x["brut"].median(), "cond_median":x[condition].median(),
                     "delta_median":delta.median(), "p":p})
aq = pd.DataFrame(rows)
if len(aq):
    valid = aq["p"].notna()
    aq.loc[valid, "p_fdr"] = multipletests(aq.loc[valid, "p"], method="fdr_bh")[1]
    display(aq.sort_values(["measure","condition"]).round(3))""")

    md("""## B. Récupération des volumes des scans bougés

Pour chaque structure anatomique, le traitement est comparé au `raw/run-01` du même sujet :

G = |V(raw, run) − V(raw, run01)| − |V(condition, run) − V(raw, run01)|

- G > 0 : le traitement rapproche le volume de la référence ;
- G < 0 : il l'en éloigne.

Le gain médian est aussi exprimé en % de la base de la structure (`gain_pct`), pour que le classement ne soit plus dominé par les gros volumes totaux. La fraction de cas améliorés est déjà sans échelle.""")

    code("""ref = (d[(d["condition"]=="brut") & (d["run"]=="run-01")]
       [["subject","measure","value_mm3"]].rename(columns={"value_mm3":"reference_mm3"}))
moved = d[d["run"].isin(["run-02","run-03"]) & d["measure"].isin(anat)].merge(
    ref, on=["subject","measure"], how="inner")
moved["error_mm3"] = (moved["value_mm3"]-moved["reference_mm3"]).abs()
raw_error = (moved[moved["condition"]=="brut"]
             [["subject","run","measure","error_mm3"]].rename(columns={"error_mm3":"raw_error_mm3"}))
moved = moved.merge(raw_error, on=["subject","run","measure"], how="inner")
moved["gain_mm3"] = moved["raw_error_mm3"] - moved["error_mm3"]

summary_b = moved[moved["condition"]!="brut"].groupby("condition", observed=True).agg(
    n=("gain_mm3","size"), gain_median_mm3=("gain_mm3","median"),
    gain_mean_mm3=("gain_mm3","mean"), fraction_amelioree=("gain_mm3",lambda x:(x>0).mean()))
print("Bilan par condition (toutes structures anatomiques confondues) :")
display(summary_b.round({"gain_median_mm3":1,"gain_mean_mm3":1,"fraction_amelioree":3}))

base_ser = baseline.reindex(anat)
b = (moved[moved["condition"]!="brut"].groupby(["condition","measure"], observed=True)["gain_mm3"]
     .median().reset_index(name="gain_median_mm3"))
b["gain_pct"] = 100 * b["gain_median_mm3"] / b["measure"].map(base_ser)
print("\\nStructures les mieux récupérées par condition (gain en % de la base) :")
display(b.sort_values("gain_pct", ascending=False).groupby("condition", observed=True).head(6)
        .round({"gain_median_mm3":1,"gain_pct":2}))
print("\\nStructures les plus dégradées par condition :")
display(b.sort_values("gain_pct").groupby("condition", observed=True).head(6)
        .round({"gain_median_mm3":1,"gain_pct":2}))""")

    md("""## C. Le mouvement prédit-il encore chaque volume ?

Pour chaque condition et chaque volume anatomique :

Volume ~ Agitation + (1 | sujet)

L'intercept aléatoire absorbe les différences stables de taille entre sujets. Le coefficient Agitation est exprimé en **mm³ par point Agitation**, et `coef_pct_par_point` le rapporte à la base de la structure pour comparer les effets. La correction FDR est appliquée séparément dans chaque condition, sur les modèles convergés.""")

    code("""rows = []
for condition in CONDITIONS:
    dc = d[(d["condition"]==condition) & d["measure"].isin(anat)].dropna(subset=["agitation","value_mm3"])
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
                     "coef_mm3_par_point":coef, "coef_pct_par_point":100*coef/baseline.get(measure, np.nan),
                     "ic95_bas":lo, "ic95_haut":hi, "p":p, "methode":method})
c = pd.DataFrame(rows)
c["p_fdr"] = np.nan
for cond, grp in c.groupby("condition", observed=True):
    v = grp["p"].notna()
    if v.any():
        c.loc[grp.index[v], "p_fdr"] = multipletests(grp.loc[v, "p"], method="fdr_bh")[1]

n_sig = (c.assign(sig=c["p_fdr"] < 0.05).groupby("condition", observed=True)["sig"]
         .agg(["sum","size"]).rename(columns={"sum":"mesures liées à Agitation (FDR<0.05)","size":"mesures testées"}))
print("Nombre de volumes encore prédits par Agitation après FDR :")
display(n_sig)
print("\\nDouze mesures les plus liées à Agitation par condition :")
display(c.sort_values(["condition","p_fdr"]).groupby("condition", observed=True).head(12)
        .loc[:, ["condition","measure","n","coef_mm3_par_point","coef_pct_par_point","p","p_fdr","methode"]]
        .round({"coef_mm3_par_point":1,"coef_pct_par_point":3,"p":5,"p_fdr":4}))""")

    md("""## Structures centrales à examiner visuellement

La zone entourée se situe autour du troisième ventricule et des thalamus médiaux. `aseg` ne possède pas nécessairement un label autonome pour l'adhérence interthalamique. Une disparition visuelle doit donc être séparée d'une variation de label ou de volume segmenté.""")

    code("""central = [
    "3rd-Ventricle", "Left-Thalamus", "Right-Thalamus",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
    "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent", "CSF"
]
central = [m for m in central if m in anat]
central_a = a[a["measure"].isin(central)].copy()
display(central_a.sort_values(["measure","condition"])
        .loc[:, ["condition","measure","baseline_mm3","delta_median_mm3","delta_pct","p_fdr"]]
        .round({"baseline_mm3":0,"delta_median_mm3":0,"delta_pct":2,"p_fdr":4}))
if len(central_a):
    plt.figure(figsize=(11, 5))
    sns.pointplot(data=central_a, x="measure", y="delta_pct",
                  hue="condition", dodge=True)
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.ylabel("variation médiane du run-01 (%)")
    plt.xlabel(""); plt.xticks(rotation=35, ha="right")
    plt.title("Structures centrales : effet propre du traitement")
    plt.tight_layout(); plt.show()""")

    md("""## Synthèse automatique

Résumé chiffré tiré directement des tables ci-dessus, pour relire le notebook sans recalcul manuel. Aucune valeur n'est saisie à la main.""")

    code("""lignes = []
for condition in CONDITIONS[1:]:
    aa = a[a["condition"]==condition]
    top = aa.loc[aa["delta_pct"].abs().idxmax()] if len(aa) else None
    frac = summary_b.loc[condition, "fraction_amelioree"] if condition in summary_b.index else np.nan
    nsig = int((c[c["condition"]==condition]["p_fdr"] < 0.05).sum())
    lignes.append({
        "condition": condition,
        "run-01 : structure la plus déplacée": f"{top['measure']} ({top['delta_pct']:+.1f} %)" if top is not None else "",
        "récupération : fraction améliorée": round(frac, 3) if frac == frac else np.nan,
        "mvt résiduel : n mesures liées à Agitation (FDR<0.05)": nsig,
    })
synth = pd.DataFrame(lignes).set_index("condition")
display(synth)
nsig_brut = int((c[c["condition"]=="brut"]["p_fdr"] < 0.05).sum())
print(f"Référence : sur le brut, {nsig_brut} volumes anatomiques sont liés à Agitation (FDR<0.05).")""")

    md("""## Conclusion à remplir après exécution

- **Stabilité du run-01 :** quelles structures anatomiques sont déplacées sans mouvement, et de combien en mm³ / % ?
- **Qualité :** un traitement crée-t-il des trous de surface ou des hypointensités (section A-bis) ?
- **Récupération :** quelles conditions rapprochent réellement les runs bougés de la référence, structure par structure ?
- **Mouvement résiduel :** combien de volumes restent liés à Agitation après FDR, et lesquels ?
- **Zone centrale :** le changement est-il un contraste visuel, un volume différent ou un label différent ?

Aucune amélioration globale ne sera appelée « restauration anatomique » sans cohérence entre ces lectures.""")

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name":"Python (cortical-motion)",
                                      "language":"python", "name":"python3"}
    nbf.write(nb, OUT)
    print(f"Notebook créé: {OUT}")


if __name__ == "__main__":
    build()
