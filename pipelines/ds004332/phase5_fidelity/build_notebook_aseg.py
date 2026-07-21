#!/usr/bin/env python3
"""Génère le notebook minimal d'analyse des volumes FreeSurfer aseg.stats."""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("explore_aseg_rigide.ipynb")


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))

    md("""# Fidélité des volumes FreeSurfer après traitement

Ce notebook répond à seulement deux questions :

1. **Scan presque immobile :** le traitement modifie-t-il les volumes de run-01 alors qu'il y a peu de mouvement à corriger ?
2. **Scans bougés :** JDAC ou ses variantes réduisent-ils l'erreur davantage que le preprocessing seul ?

La référence anatomique opérationnelle est toujours le brut/run-01 du même sujet. Les analyses principales utilisent une erreur régionale médiane calculée par sujet : le sujet, et non chaque structure séparée, reste l'unité statistique.""")

    code("""from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

CONDITIONS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
LABELS = {
    "brut": "brut", "preproc": "preproc", "jdac": "JDAC",
    "jdac_antiartonly": "antiart ×1", "jdac_nodenoise": "antiart ×4",
}
ASEG_DIR = Path.home() / "Documents/derivatives/ds004332/aseg_stats"
ASEG_FILES = {
    "brut": "aseg_brut.csv",
    "preproc": "aseg_preproc.csv",
    "jdac": "aseg_jdac.csv",
    "jdac_antiartonly": "aseg_antiartonly.csv",
    "jdac_nodenoise": "aseg_nodenoise.csv",
}

def load_table(condition, filename):
    wide = pd.read_csv(ASEG_DIR / filename, sep="\t")
    id_col = "Measure:volume" if "Measure:volume" in wide.columns else wide.columns[0]
    ids = wide[id_col].astype(str).str.extract(r"(sub-[^_/\\s]+)_(run-\\d+)", expand=True)
    if ids.isna().any().any():
        raise ValueError(f"Identifiants non reconnus dans {filename}")
    return (wide.drop(columns=id_col)
            .assign(subject=ids[0].values, run=ids[1].values)
            .melt(id_vars=["subject", "run"], var_name="measure", value_name="value_mm3")
            .assign(condition=condition))

missing = [name for name in ASEG_FILES.values() if not (ASEG_DIR / name).is_file()]
if missing:
    raise FileNotFoundError("Tables absentes : " + ", ".join(missing))

d = pd.concat([load_table(c, f) for c, f in ASEG_FILES.items()], ignore_index=True)
d["value_mm3"] = pd.to_numeric(d["value_mm3"], errors="coerce")
d["condition"] = pd.Categorical(d["condition"], CONDITIONS, ordered=True)

GLOBAL = [
    "CortexVol", "TotalGrayVol", "CerebralWhiteMatterVol",
    "SubCortGrayVol", "CSF", "BrainSegVolNotVent",
]
EXCLUDED = {
    "EstimatedTotalIntraCranialVol", "SegmentedTotalIntracranialVol",
    "BrainSegVol-to-eTIV", "MaskVol-to-eTIV",
    "SurfaceHoles", "lhSurfaceHoles", "rhSurfaceHoles",
    "WM-hypointensities", "Left-WM-hypointensities",
    "Right-WM-hypointensities", "non-WM-hypointensities",
    "Left-non-WM-hypointensities", "Right-non-WM-hypointensities",
}
base_median = (d[(d.condition=="brut") & (d.run=="run-01")]
               .groupby("measure", observed=True).value_mm3.median())
regional = [m for m in base_median.index
            if m not in EXCLUDED and m not in GLOBAL and base_median[m] > 100]

context = pd.DataFrame([{
    "Sujets": d.subject.nunique(),
    "Runs": d.run.nunique(),
    "Conditions": d.condition.nunique(),
    "Acquisitions disponibles": f"{d[['subject','run','condition']].drop_duplicates().shape[0]}/{d.subject.nunique()*d.run.nunique()*d.condition.nunique()}",
    "Régions analysées": len(regional),
}])
display(context)""")

    md("""## 1. Le traitement modifie-t-il le scan presque immobile ?

Pour chaque sujet et chaque condition, on calcule l'écart relatif absolu à son brut/run-01 dans chaque région, puis la médiane entre les régions :

**erreur d'identité (%) = médiane des |volume traité − volume brut| / volume brut × 100**

Une valeur faible est meilleure. La comparaison principale est JDAC contre preproc, car JDAC est appliqué après ce preprocessing.""")

    code("""ref = (d[(d.condition=="brut") & (d.run=="run-01") & d.measure.isin(regional)]
       [["subject","measure","value_mm3"]]
       .rename(columns={"value_mm3":"reference_mm3"}))

still = (d[(d.run=="run-01") & d.measure.isin(regional)]
         .merge(ref, on=["subject","measure"], how="inner"))
still["error_pct"] = 100 * (still.value_mm3-still.reference_mm3).abs() / still.reference_mm3
identity = (still.groupby(["subject","condition"], observed=True).error_pct
            .median().reset_index(name="erreur_identite_pct"))

plt.figure(figsize=(8,4.5))
sns.boxplot(data=identity[identity.condition!="brut"], x="condition",
            y="erreur_identite_pct", order=CONDITIONS[1:], color="lightsteelblue")
sns.stripplot(data=identity[identity.condition!="brut"], x="condition",
              y="erreur_identite_pct", order=CONDITIONS[1:],
              color="black", alpha=.55, size=3)
plt.xticks(range(4), [LABELS[c] for c in CONDITIONS[1:]])
plt.ylabel("Erreur régionale médiane vs brut/run-01 (%)")
plt.xlabel("")
plt.title("Scan presque immobile : erreur ajoutée par le traitement")
plt.tight_layout()
plt.show()""")

    code("""def bootstrap_ci(values, n_boot=10000, seed=20260720):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    return np.percentile(np.median(draws, axis=1), [2.5, 97.5])

def paired_contrasts(scores, specs, value_col):
    wide = scores.pivot(index="subject", columns="condition", values=value_col)
    rows = []
    for label, candidate, comparator in specs:
        pair = wide[[candidate, comparator]].dropna()
        diff = pair[candidate] - pair[comparator]
        lo, hi = bootstrap_ci(diff)
        p = stats.wilcoxon(diff).pvalue if len(diff) >= 3 and np.any(diff != 0) else 1.0
        rows.append({"comparaison":label, "n":len(diff),
                     "différence médiane (points %)":np.median(diff),
                     "IC95 bas":lo, "IC95 haut":hi, "p":p})
    out = pd.DataFrame(rows)
    out["p Holm"] = multipletests(out.p, method="holm")[1]
    return out

identity_tests = paired_contrasts(
    identity[identity.condition!="brut"],
    [
        ("JDAC − preproc", "jdac", "preproc"),
        ("antiart ×1 − preproc", "jdac_antiartonly", "preproc"),
        ("antiart ×4 − preproc", "jdac_nodenoise", "preproc"),
    ],
    "erreur_identite_pct",
)
display(identity_tests.round(3))
print("Lecture : une différence positive signifie plus d'erreur que preproc.")""")

    md("""### Où se produit le changement ?

Ce tableau secondaire montre la direction des changements dans six compartiments globaux. Il sert à expliquer l'erreur d'identité; il ne remplace pas le score régional principal.""")

    code("""gref = (d[(d.condition=="brut") & (d.run=="run-01") & d.measure.isin(GLOBAL)]
        [["subject","measure","value_mm3"]]
        .rename(columns={"value_mm3":"reference_mm3"}))
g = (d[(d.run=="run-01") & d.measure.isin(GLOBAL) & (d.condition!="brut")]
     .merge(gref, on=["subject","measure"], how="inner"))
g["signed_pct"] = 100 * (g.value_mm3-g.reference_mm3) / g.reference_mm3
global_table = (g.groupby(["condition","measure"], observed=True).signed_pct
                .median().unstack("measure").reindex(CONDITIONS[1:]))
global_table.index = [LABELS[c] for c in global_table.index]
display(global_table.round(1))""")

    md("""## 2. Sur les scans bougés, le traitement fait-il mieux que preproc ?

Pour run-02 et run-03, on calcule la même erreur régionale médiane par rapport au brut/run-01 du sujet.

La comparaison essentielle est directe :

- différence négative : le candidat produit moins d'erreur que preproc ;
- différence positive : le candidat produit plus d'erreur que preproc.""")

    code("""moved = (d[d.run.isin(["run-02","run-03"]) & d.measure.isin(regional)]
         .merge(ref, on=["subject","measure"], how="inner"))
moved["error_pct"] = 100 * (moved.value_mm3-moved.reference_mm3).abs() / moved.reference_mm3
fidelity = (moved.groupby(["subject","run","condition"], observed=True).error_pct
            .median().reset_index(name="erreur_regionale_pct"))

gplot = sns.catplot(data=fidelity, x="condition", y="erreur_regionale_pct",
                    col="run", kind="box", order=CONDITIONS,
                    color="lightsteelblue", height=4.2, aspect=1.05)
for ax in gplot.axes.flat:
    ax.set_xticks(range(5))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Erreur régionale médiane vs brut/run-01 (%)")
gplot.fig.subplots_adjust(top=.84)
gplot.fig.suptitle("Scans bougés : fidélité régionale par condition")
plt.show()""")

    code("""rows = []
for run in ["run-02","run-03"]:
    sub = fidelity[fidelity.run==run]
    tests = paired_contrasts(
        sub,
        [
            ("preproc − brut", "preproc", "brut"),
            ("JDAC − preproc", "jdac", "preproc"),
            ("antiart ×1 − preproc", "jdac_antiartonly", "preproc"),
            ("antiart ×4 − preproc", "jdac_nodenoise", "preproc"),
        ],
        "erreur_regionale_pct",
    )
    tests.insert(0, "run", run)
    rows.append(tests)
motion_tests = pd.concat(rows, ignore_index=True)
motion_tests["p Holm"] = multipletests(motion_tests.p, method="holm")[1]
display(motion_tests.drop(columns="p").round(3))
print("Lecture : une différence négative favorise la première condition nommée.")""")

    md("""## Conclusion

La décision repose uniquement sur les deux tableaux de contrastes :

1. **Identité :** JDAC ou ses variantes ajoutent-ils plus d'erreur que preproc sur run-01 ?
2. **Fidélité :** sur run-02 et run-03, réduisent-ils l'erreur par rapport à preproc ?

Une amélioration visuelle n'est pas considérée comme une restauration anatomique si l'erreur régionale n'est pas réduite.""")

    code("""def line_for(row, prefix):
    diff = row["différence médiane (points %)"]
    direction = "plus d'erreur" if diff > 0 else "moins d'erreur"
    return (f"{prefix}: {abs(diff):.2f} points de {direction} "
            f"[IC95 {row['IC95 bas']:.2f}; {row['IC95 haut']:.2f}], "
            f"p Holm={row['p Holm']:.3g}.")

id_jdac = identity_tests[identity_tests.comparaison=="JDAC − preproc"].iloc[0]
m_jdac = motion_tests[motion_tests.comparaison=="JDAC − preproc"].set_index("run")

print(line_for(id_jdac, "1. Identité, JDAC vs preproc"))
print(line_for(m_jdac.loc["run-02"], "2. Run-02, JDAC vs preproc"))
print(line_for(m_jdac.loc["run-03"], "3. Run-03, JDAC vs preproc"))

identity_worse = id_jdac["différence médiane (points %)"] > 0
improves_02 = (m_jdac.loc["run-02","différence médiane (points %)"] < 0
               and m_jdac.loc["run-02","p Holm"] < 0.05)
improves_03 = (m_jdac.loc["run-03","différence médiane (points %)"] < 0
               and m_jdac.loc["run-03","p Holm"] < 0.05)

if improves_02 and improves_03 and not identity_worse:
    conclusion = "JDAC préserve run-01 et améliore les deux runs bougés par rapport à preproc."
elif (improves_02 or improves_03) and identity_worse:
    conclusion = "JDAC améliore au moins un run bougé, mais ajoute une erreur sur run-01."
elif improves_02 or improves_03:
    conclusion = "JDAC montre un bénéfice limité à un seul niveau de mouvement."
else:
    conclusion = "JDAC ne démontre pas de bénéfice supplémentaire par rapport à preproc."
print("4. Conclusion:", conclusion)""")

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name":"Python (cortical-motion)",
        "language":"python",
        "name":"python3",
    }
    nbf.write(nb, OUT)
    print(f"Notebook créé : {OUT}")


if __name__ == "__main__":
    build()
