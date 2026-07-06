"""Génère le notebook d'analyse de l'épaisseur corticale, 5 conditions, pipeline rigide :
    explore_epaisseur_rigide.ipynb

Conditions : brut, preproc, jdac, jdac_antiartonly (anti-artefact x1), jdac_nodenoise (boucle x4).
Structure volontairement réduite (3 questions), chaque sortie en mm, chaque cellule avec sa
question et sa lecture, les données par sujet montrées avant toute statistique.

  A. Que fait la condition sur un scan immobile (rien à corriger) ? -> offset / lissage.
  B. La condition rapproche-t-elle immobile et bougé, sujet par sujet ? -> figure + décompte.
  C. Après la condition, le mouvement prédit-il encore l'épaisseur ? -> modèles M0 vs M1.

Vocabulaire : `condition` = les 5 traitements ; `consigne` = still/nodding/shaking.
Registre impersonnel, aucun chiffre de résultat écrit en dur (ils viennent des cellules).
"""
import nbformat as nbf
from pathlib import Path

OUTDIR = Path(__file__).parent
FNAME = "explore_epaisseur_rigide.ipynb"


def build():
    nb = nbf.v4.new_notebook()
    cells = []
    def md(t): cells.append(nbf.v4.new_markdown_cell(t))
    def code(t): cells.append(nbf.v4.new_code_cell(t))

    md("""# Épaisseur corticale après JDAC et ses variantes (ds004332, pipeline rigide)

**Question.** Chaque condition de traitement corrige-t-elle le mouvement (l'épaisseur mesurée ne dépend plus du mouvement), le lisse-t-elle (elle abaisse l'épaisseur même sans mouvement), ou le sur-corrige-t-elle (elle inverse le lien) ?

**Cinq conditions** (toutes en rigide) : `brut`, `preproc`, `jdac` (complet), `jdac_antiartonly` (anti-artefact ×1), `jdac_nodenoise` (anti-artefact ×4, sans débruiteur).

**Unité.** Toutes les épaisseurs sont en **mm**. Le mouvement est le score Agitation (sans unité). Trois runs par sujet : run-01 immobile (still), run-02 (nodding), run-03 bougé (shaking).

**Sources des données** (traçabilité) : brut = `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv` ; preproc/jdac = `derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv` ; variantes = `derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/…` ; Agitation = `results/ds004332/agitation/ds004332_agitation_clinica.csv` ; âge/sexe = `raw_datasets/ds004332/participants.tsv`.""")

    code('''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from IPython.display import display
import warnings; warnings.filterwarnings("ignore")

def show(df, question, lecture, fmt="{:.3f}"):
    # Affiche un tableau précédé de la question posée et suivi de la lecture.
    print("Question :", question)
    display(df.style.format(fmt, na_rep="—").set_table_styles([
        {"selector": "th", "props": "background-color:#d9e1f2;padding:5px 12px;font-size:12px;"},
        {"selector": "td", "props": "padding:5px 12px;font-size:12px;text-align:right;"},
        {"selector": "tbody tr:nth-child(odd)", "props": "background-color:#f6f6f6;"}]))
    print("Lecture :", lecture)

HOME  = Path.home()
REPO  = HOME / "Documents/jdac-motion-correction"
DERIV = HOME / "Documents/derivatives/ds004332"
CONDITIONS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
SHORT = {"brut": "brut", "preproc": "prep", "jdac": "jdac",
         "jdac_antiartonly": "aa×1", "jdac_nodenoise": "nod×4"}
CONSIGNE = {"run-01": "still", "run-02": "nodding", "run-03": "shaking"}
WIDE = {"preproc": "thickness_preproc_rigid_{h}.csv",
        "jdac": "thickness_jdac_rigid_{h}.csv",
        "jdac_antiartonly": "thickness_jdac_antiartonly_rigid/thickness_jdac_antiartonly_rigid_{h}.csv",
        "jdac_nodenoise": "thickness_jdac_nodenoise_rigid/thickness_jdac_nodenoise_rigid_{h}.csv"}

def load_brut():
    d = pd.read_csv(REPO / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv")
    d = d[d["ThickAvg"] > 0].copy()
    d["run"] = d["subject"].str.split("_").str[1]
    d["subject"] = d["subject"].str.split("_").str[0]
    return d.rename(columns={"ThickAvg": "thickness"}).assign(condition="brut")[
        ["subject", "run", "hemi", "region", "thickness", "condition"]]

def load_wide(cond):
    frames = []
    for hemi in ["lh", "rh"]:
        w = pd.read_csv(DERIV / WIDE[cond].format(h=hemi), sep="\\t")
        w = w.rename(columns={w.columns[0]: "id"})
        cols = [c for c in w.columns if c.endswith("_thickness") and "MeanThickness" not in c]
        long = w.melt(id_vars="id", value_vars=cols, value_name="thickness")
        long["subject"] = long["id"].str.split("_").str[0]
        long["run"] = long["id"].str.split("_").str[1]
        frames.append(long)
    return pd.concat(frames, ignore_index=True).assign(condition=cond)[
        ["subject", "run", "thickness", "condition"]]

# Épaisseur corticale moyenne (mm) par acquisition (moyenne des régions des 2 hémisphères)
parts = [load_brut()[["subject", "run", "thickness", "condition"]]] + \\
        [load_wide(c) for c in CONDITIONS if c != "brut"]
thick = pd.concat(parts, ignore_index=True)
thick = thick[thick["thickness"] > 0]
g = thick.groupby(["subject", "run", "condition"], observed=True)["thickness"].mean().reset_index()

agit = pd.read_csv(REPO / "results/ds004332/agitation/ds004332_agitation_clinica.csv") \\
         .rename(columns={"condition": "run", "sub": "subject", "motion": "agitation"})
demog = pd.read_csv(HOME / "Documents/raw_datasets/ds004332/participants.tsv", sep="\\t") \\
          .rename(columns={"participant_id": "subject"})
demog["sex_bin"] = (demog["sex"] == "F").astype(int)
g = g.merge(agit[["subject", "run", "agitation"]], on=["subject", "run"], how="left")
g = g.merge(demog[["subject", "age", "sex_bin"]], on="subject", how="left")
g["consigne"] = g["run"].map(CONSIGNE)
g["condition"] = pd.Categorical(g["condition"], categories=CONDITIONS, ordered=True)

print("Acquisitions par condition (une par sujet × run) :")
print(g.groupby("condition", observed=True).size().to_string())
print("\\nÉpaisseur = moyenne des régions FreeSurfer, en mm. Une ligne = un sujet × run × condition.")''')

    # ---------------------------------------------------------------- A. immobiles
    md("""## A. Effet sur un scan immobile (offset / lissage)

Sur un scan immobile (run-01), il n'y a pas de mouvement à corriger. Toute différence d'épaisseur entre conditions vient donc du traitement lui-même.""")

    code('''imm = g[g["consigne"] == "still"]
tA = imm.groupby("condition", observed=True)["thickness"].agg(["mean", "std", "count"])
tA.columns = ["épaisseur immobile (mm)", "écart-type (mm)", "n sujets"]
ref = tA.loc["brut", "épaisseur immobile (mm)"]
tA["écart au brut (mm)"] = tA["épaisseur immobile (mm)"] - ref
tA["écart au brut (%)"] = 100 * tA["écart au brut (mm)"] / ref
show(tA.reindex(CONDITIONS),
     "Sur les scans immobiles, chaque condition change-t-elle l'épaisseur par rapport au brut ?",
     "Un écart au brut négatif = la condition amincit un cerveau propre = lissage/offset. "
     "Une correction fidèle laisse l'immobile proche du brut (écart proche de 0 mm).",
     {"épaisseur immobile (mm)": "{:.3f}", "écart-type (mm)": "{:.3f}", "n sujets": "{:.0f}",
      "écart au brut (mm)": "{:+.3f}", "écart au brut (%)": "{:+.1f}"})''')

    # ---------------------------------------------------------------- B. par sujet
    md("""## B. Immobile vs bougé, sujet par sujet

Cœur de l'analyse, vu sur les données brutes. Pour chaque sujet on regarde son scan immobile (run-01) et son scan bougé (run-03). Si une condition corrige le mouvement, ces deux valeurs se rapprochent ; si elle sur-corrige, le bougé passe au-dessus de l'immobile.

La figure montre un panneau par sujet (`sub-19` en évidence). Chaque panneau : épaisseur (mm) de l'immobile et du bougé, pour les 5 conditions.""")

    code('''pv = g.pivot_table(index=["subject", "condition"], columns="consigne",
                   values="thickness", observed=True)
subjects = sorted(g["subject"].unique())
ncol = 5; nrow = int(np.ceil(len(subjects) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow), sharey=True)
xpos = np.arange(len(CONDITIONS))
for ax, subj in zip(axes.ravel(), subjects):
    nod   = [pv.loc[(subj, c), "nodding"] if (subj, c) in pv.index else np.nan for c in CONDITIONS]
    still = [pv.loc[(subj, c), "still"] if (subj, c) in pv.index else np.nan for c in CONDITIONS]
    shak  = [pv.loc[(subj, c), "shaking"] if (subj, c) in pv.index else np.nan for c in CONDITIONS]
    ax.plot(xpos, still, "-o", color="tab:green",  ms=4, lw=1.3)
    ax.plot(xpos, nod,   "-s", color="tab:orange", ms=3, lw=1.0)
    ax.plot(xpos, shak,  "-o", color="tab:red",    ms=4, lw=1.3)
    hot = (subj == "sub-19")
    ax.set_title(subj, fontsize=8, color=("crimson" if hot else "black"),
                 fontweight=("bold" if hot else "normal"))
    ax.set_xticks(xpos); ax.set_xticklabels([SHORT[c] for c in CONDITIONS], fontsize=6, rotation=45)
    ax.tick_params(labelsize=6)
for ax in axes.ravel()[len(subjects):]:
    ax.axis("off")
axes.ravel()[0].plot([], [], "-o", color="tab:green",  label="immobile (run-01)")
axes.ravel()[0].plot([], [], "-s", color="tab:orange", label="nodding (run-02)")
axes.ravel()[0].plot([], [], "-o", color="tab:red",    label="shaking (run-03)")
axes.ravel()[0].legend(fontsize=6, loc="best")
fig.suptitle("Épaisseur corticale (mm) par condition : immobile vs bougé, un panneau par sujet", y=1.005)
fig.supylabel("épaisseur (mm)")
plt.tight_layout(); plt.show()
print("Lecture : les deux points proches = mouvement sans effet ; bougé (rouge) sous immobile (vert)"
      " = mouvement qui amincit ; bougé au-dessus = sur-correction.")''')

    md("""Le décompte résume la figure. Pour chaque sujet, l'écart immobile − bougé (mm) est calculé dans le brut et dans la condition, puis comparé.""")

    code('''ec = pv.reset_index().dropna(subset=["still", "shaking"])
ec["ecart"] = ec["still"] - ec["shaking"]          # > 0 : le mouvement amincit
w = ec.pivot(index="subject", columns="condition", values="ecart")

rows = []
for c in CONDITIONS:
    if c == "brut":
        continue
    p = w[["brut", c]].dropna()
    p = p[p["brut"] > 0]                            # sujets où le mouvement amincit en brut
    rows.append({"condition": c, "n sujets": len(p),
                 "améliorés": int(((p[c] >= 0) & (p[c] < p["brut"])).sum()),
                 "sur-corrigés": int((p[c] < 0).sum()),
                 "inchangés/pires": int((p[c] >= p["brut"]).sum()),
                 "écart médian brut (mm)": p["brut"].median(),
                 "écart médian condition (mm)": p[c].median()})
tB = pd.DataFrame(rows).set_index("condition")
show(tB, "Chez combien de sujets l'écart immobile − bougé se réduit vraiment (sans s'inverser) ?",
     "améliorés = écart rapproché de 0 (bon) ; sur-corrigés = écart devenu négatif, le bougé plus "
     "épais que l'immobile (mauvais) ; inchangés/pires = pas rapproché.",
     {"n sujets": "{:.0f}", "améliorés": "{:.0f}", "sur-corrigés": "{:.0f}",
      "inchangés/pires": "{:.0f}", "écart médian brut (mm)": "{:.3f}",
      "écart médian condition (mm)": "{:.3f}"})''')

    # ---------------------------------------------------------------- C. M0 vs M1
    md("""## C. Le mouvement prédit-il encore l'épaisseur ? (modèles M0 vs M1)

Pour chaque condition, deux modèles sur l'épaisseur moyenne par acquisition, avec un effet aléatoire par sujet :
- **M0** : épaisseur expliquée par l'âge et le sexe seulement.
- **M1** : on ajoute le score de mouvement (Agitation).

On compare M0 et M1. La **p-value** est la probabilité d'observer un gain d'ajustement aussi grand si le mouvement n'avait en réalité aucun effet : petite (< 0.05) = le mouvement apporte de l'information ; grande = il n'en apporte plus. Le **coefficient d'Agitation** (mm par point de score) dit de combien l'épaisseur change quand le mouvement augmente d'un point.""")

    code('''def fit(formula, d):
    return smf.mixedlm(formula, d, groups=d["subject"]).fit(reml=False, method="powell", disp=False)

rows = []
for c in CONDITIONS:
    d = g[g["condition"] == c].dropna(subset=["age", "sex_bin", "agitation", "thickness"])
    m0 = fit("thickness ~ age + sex_bin", d)
    m1 = fit("thickness ~ age + sex_bin + agitation", d)
    lr = 2 * (m1.llf - m0.llf)
    coef = m1.params["agitation"]
    rows.append({"condition": c, "n acquisitions": d.groupby(["subject", "run"]).ngroups,
                 "coef Agitation (mm/point)": coef, "p (M1 vs M0)": stats.chi2.sf(lr, 1),
                 "sens": "amincit" if coef < 0 else "épaissit"})
tC = pd.DataFrame(rows).set_index("condition")
show(tC, "Après la condition, ajouter le score de mouvement améliore-t-il encore le modèle ?",
     "brut : p petite et coef négatif (le mouvement amincit l'épaisseur mesurée). "
     "Bonne correction : p grande (le mouvement ne prédit plus l'épaisseur). "
     "coef positif = le mouvement épaissit = sur-correction.",
     {"n acquisitions": "{:.0f}", "coef Agitation (mm/point)": "{:+.4f}", "p (M1 vs M0)": "{:.2g}",
      "sens": "{}"})''')

    md("""## D. Évaluation image façon JDAC : les contours correspondent-ils au scan propre ?

L'épaisseur seule ne distingue pas correction et lissage (les deux peuvent la faire monter). On emprunte donc l'évaluation de l'article JDAC : comparer, en pleine référence, chaque scan bougé au scan **propre** du même sujet, sur l'image **et sur les cartes de gradient** (les contours). Toutes les images sont sur la même grille rigide, donc aucun recalage. Référence propre = `preproc` run-01 (le brut adapté à l'espace rigide, non corrigé). SSIM entre 0 et 1 (1 = identique au propre), moyennée sur les sujets. Le brut natif n'a pas la même grille et n'entre pas dans cette comparaison ; `preproc` sert de baseline « avant correction ».

Les métriques sont calculées à part (`compute_image_metrics.py`) et chargées ici.""")

    code('''im = pd.read_csv(REPO / "results/ds004332/phase4_compare_3bras/image_metrics.csv")
IMGCONDS = ["preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
lab = {"run-02": "nodding (mvt modéré)", "run-03": "shaking (fort mvt)"}
agg = im.groupby(["condition", "run"])[["clean_ssim_img", "clean_ssim_grad", "intra_ssim_grad"]].mean()
for metric, titre in [("clean_ssim_img", "SSIM image vs scan propre (preproc run-01)"),
                      ("clean_ssim_grad", "SSIM gradient (contours) vs scan propre (preproc run-01)")]:
    t = agg[metric].unstack("run").reindex(IMGCONDS).rename(columns=lab)
    show(t, f"{titre} : le scan bougé ressemble-t-il au scan propre ? (moyenne sujets, 0 = différent, 1 = identique)",
         "preproc = avant correction (baseline). Au-dessus de preproc = plus proche du propre après correction. "
         "Sur le gradient, en dessous de preproc = contours qui s'éloignent du propre (lissage ou contours déplacés), "
         "même si l'image paraît nette. Réserve : comparer une sortie JDAC au propre preproc mélange correction du "
         "mouvement et changement d'intensité de la condition, d'où le tableau suivant.", "{:.3f}")

ti = agg["intra_ssim_grad"].unstack("run").reindex(IMGCONDS).rename(columns=lab)
show(ti, "SSIM gradient vs le scan immobile de la MÊME condition : lecture sans le biais d'intensité",
     "Référence traitée pareil, on isole donc le mouvement. Au-dessus de preproc = après correction, le scan bougé "
     "ressemble davantage à son propre immobile (contours du mouvement réduits) = correction réelle. "
     "≤ preproc = pas de gain de contours, la netteté éventuelle ne correspond pas à l'anatomie propre.", "{:.3f}")''')

    md("""## E. Les mesures suivent-elles ? Récupération vers la vraie épaisseur (par région, mm)

Question de fond pour un article : pour un scan bougé, l'épaisseur mesurée **par région** se rapproche-t-elle de la vraie valeur (le scan immobile du même sujet) après correction ? On évite la moyenne globale (qui peut coïncider par hasard) et on mesure la distance région par région, en mm. Trois distances :
- **mouvement restant** : écart au scan immobile de la **même** condition (immobile et bougé subissent le même offset, donc ce terme est net d'offset) ;
- **erreur à la vérité** : écart au scan immobile **brut** (la vraie valeur, la moins traitée) ;
- **offset** : distorsion appliquée par la condition à un scan propre.

Métriques calculées à part (`compute_recovery.py`).""")

    code('''rec = pd.read_csv(REPO / "results/ds004332/phase4_compare_3bras/recovery_metrics.csv")
RCONDS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
for run, cons in [("run-03", "shaking (fort mouvement)"), ("run-02", "nodding (mouvement modéré)")]:
    t = (rec[rec["run"] == run].groupby("condition")[["mae_within", "mae_truth", "offset"]]
         .mean().reindex(RCONDS))
    t.columns = ["mouvement restant (mm)", "erreur à la vérité (mm)", "offset sur le propre (mm)"]
    show(t, f"Scan {cons} : l'épaisseur régionale se rapproche-t-elle de la vraie valeur (immobile brut) ?",
         "brut (1re ligne) = niveau sans correction. Une correction ferait baisser le mouvement restant ET "
         "l'erreur à la vérité SOUS le niveau du brut. Si ces distances restent égales ou plus hautes que le brut, "
         "la correction ne rapproche pas la mesure de la vraie valeur (les mesures ne suivent pas).", "{:.3f}")''')

    md("""## Synthèse

Une condition **corrige** le mouvement si : elle laisse l'immobile proche du brut (A), elle rapproche immobile et bougé chez une majorité de sujets (B), le mouvement ne prédit plus l'épaisseur (C, p grande), les contours se rapprochent du propre (D), et surtout **l'épaisseur régionale du scan bougé se rapproche de la vraie valeur, sous le niveau brut (E)**. Elle **lisse** si elle abaisse l'immobile (A) et distord les contours (D). Elle **sur-corrige** si le coefficient d'Agitation devient positif (C) et si l'erreur régionale à la vérité **augmente** malgré une moyenne qui semble récupérée (E). La section E est le juge final pour un article : elle dit si la mesure corticale suit vraiment la correction, région par région.""")

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    nbf.write(nb, OUTDIR / FNAME)
    print("Notebook écrit :", OUTDIR / FNAME, "|", len(nb["cells"]), "cellules")
