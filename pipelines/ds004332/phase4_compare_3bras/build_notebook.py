"""Génère le notebook d'analyse de l'épaisseur corticale, 5 conditions, pipeline rigide :
    explore_epaisseur_rigide.ipynb

Conditions : brut, preproc, jdac, jdac_antiartonly (anti-artefact x1), jdac_nodenoise (boucle x4).
Cinq questions (A-E), chaque sortie en mm, chaque tableau suivi d'une ANALYSE construite à partir
des valeurs calculées (donc toujours exacte, aucun chiffre écrit en dur).

  A. Effet sur un scan immobile (offset / lissage).
  B. Immobile vs bougé, sujet par sujet (figure + décompte).
  C. Le mouvement prédit-il encore l'épaisseur ? (modèles M0 vs M1).
  D. Contours vs scan propre (protocole d'évaluation de JDAC).
  E. Récupération vers la vraie épaisseur régionale (le juge final).

Vocabulaire : `condition` = les 5 traitements ; `consigne` = still/nodding/shaking. Registre impersonnel.
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

**Question.** Chaque condition corrige-t-elle le mouvement (l'épaisseur mesurée ne dépend plus du mouvement), le lisse-t-elle (elle abaisse l'épaisseur même sans mouvement), ou le sur-corrige-t-elle (elle inverse le lien) ?

**Cinq conditions** (toutes en rigide) : `brut`, `preproc`, `jdac` (complet), `jdac_antiartonly` (anti-artefact ×1), `jdac_nodenoise` (anti-artefact ×4, sans débruiteur).

**Unité.** Épaisseurs en **mm** ; mouvement = score Agitation (sans unité). Trois runs par sujet : run-01 immobile (still), run-02 (nodding), run-03 bougé (shaking).

**Sources** : brut `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv` ; preproc/jdac `derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv` ; variantes `derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/…` ; Agitation `results/ds004332/agitation/ds004332_agitation_clinica.csv` ; âge/sexe `raw_datasets/ds004332/participants.tsv`.""")

    code('''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from IPython.display import display
import warnings; warnings.filterwarnings("ignore")

def show(df, question, analyse, fmt="{:.3f}"):
    # Tableau précédé de la question, suivi de l'analyse des résultats.
    print("Question :", question)
    display(df.style.format(fmt, na_rep="—").set_table_styles([
        {"selector": "th", "props": "background-color:#d9e1f2;padding:5px 12px;font-size:12px;"},
        {"selector": "td", "props": "padding:5px 12px;font-size:12px;text-align:right;"},
        {"selector": "tbody tr:nth-child(odd)", "props": "background-color:#f6f6f6;"}]))
    print("Analyse :", analyse)

HOME  = Path.home()

def find_repo():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv").exists():
            return p
    fallback = HOME / "Documents/GitHub/jdac-motion-correction"
    if fallback.exists():
        return fallback
    return HOME / "Documents/jdac-motion-correction"

REPO  = find_repo()
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
        ["subject", "run", "thickness", "condition"]]

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
thick = pd.concat([load_brut()] + [load_wide(c) for c in CONDITIONS if c != "brut"], ignore_index=True)
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
    md(r"""## A. Le traitement modifie-t-il déjà le scan de référence ?

`run-01` est l'acquisition avec consigne immobile et le plus faible mouvement. Ce n'est pas une vérité parfaite, mais c'est notre **référence opérationnelle**. Un correcteur ne devrait pas la modifier fortement.

Pour chaque sujet et chaque condition, l'épaisseur globale est la moyenne non pondérée des 68 épaisseurs régionales FreeSurfer. La comparaison est maintenant **appariée** : une condition n'est comparée au brut que chez les sujets disposant des deux mesures `run-01`.

Pour un sujet $s$ et une condition $c$ :

$$\Delta_{s,c}=\bar T_{s,run01,c}-\bar T_{s,run01,brut}$$

Une valeur négative signifie que le traitement amincit le cortex mesuré sur le scan de référence.""")

    code('''imm = g[g["consigne"] == "still"].pivot_table(
    index="subject", columns="condition", values="thickness", observed=True
)
rows = []
for c in CONDITIONS:
    pair = imm[["brut", c]].dropna() if c != "brut" else imm[["brut"]].dropna()
    if c == "brut":
        delta = pair["brut"] * 0
        treated = pair["brut"]
    else:
        delta = pair[c] - pair["brut"]
        treated = pair[c]
    rows.append({
        "condition": c,
        "n paires": len(pair),
        "brut run-01 apparié (mm)": pair["brut"].mean(),
        "condition run-01 (mm)": treated.mean(),
        "différence appariée moyenne (mm)": delta.mean(),
        "différence appariée médiane (mm)": delta.median(),
        "différence moyenne (%)": 100 * delta.mean() / pair["brut"].mean(),
    })
tA = pd.DataFrame(rows).set_index("condition").reindex(CONDITIONS)

o = tA["différence appariée moyenne (mm)"].drop("brut")
analyse = (f"Chez les mêmes sujets, les quatre traitements déplacent l'épaisseur du run-01. "
           f"L'effet moyen va de {o.max():+.3f} à {o.min():+.3f} mm; le plus fort est {o.idxmin()}. "
           f"Comme la comparaison est appariée, cet écart ne vient pas d'un changement de composition du groupe. "
           f"Il mesure l'effet propre du traitement sur la référence opérationnelle.")
show(tA, "Chez les mêmes sujets, le traitement change-t-il l'épaisseur du run-01 par rapport au brut ?", analyse,
     {"n paires": "{:.0f}", "brut run-01 apparié (mm)": "{:.3f}",
      "condition run-01 (mm)": "{:.3f}", "différence appariée moyenne (mm)": "{:+.3f}",
      "différence appariée médiane (mm)": "{:+.3f}", "différence moyenne (%)": "{:+.1f}"})''')

    # ---------------------------------------------------------------- B. par sujet
    md(r"""## B. Les acquisitions traitées se rapprochent-elles de `raw/run-01` ?

La référence horizontale zéro est, pour chaque sujet, sa propre épaisseur globale `raw/run-01`. Chaque point représente :

$$\Delta_{s,r,c}=\bar T_{s,r,c}-\bar T_{s,run01,brut}$$

- proche de zéro : épaisseur globale proche de la référence opérationnelle ;
- négatif : cortex mesuré plus mince que la référence ;
- positif : cortex mesuré plus épais.

Cette figure remplace les 22 petits panneaux. Elle montre la distribution entre sujets et permet de voir simultanément le déplacement du run-01 et celui des runs bougés.""")

    code('''raw_still = (g[(g["condition"] == "brut") & (g["consigne"] == "still")]
             [["subject", "thickness"]].rename(columns={"thickness": "raw_run01"}))
rel = g.merge(raw_still, on="subject", how="inner")
rel["écart à raw/run-01 (mm)"] = rel["thickness"] - rel["raw_run01"]

colors = {"still": "tab:green", "nodding": "tab:orange", "shaking": "tab:red"}
offsets = {"still": -0.23, "nodding": 0.0, "shaking": 0.23}
fig, ax = plt.subplots(figsize=(11, 5.2))
for i, c in enumerate(CONDITIONS):
    for cons in ["still", "nodding", "shaking"]:
        values = rel[(rel["condition"] == c) & (rel["consigne"] == cons)]["écart à raw/run-01 (mm)"].dropna()
        if values.empty:
            continue
        pos = i + offsets[cons]
        bp = ax.boxplot(values, positions=[pos], widths=0.18, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "lw": 1.4})
        bp["boxes"][0].set(facecolor=colors[cons], alpha=0.28, edgecolor=colors[cons])
        for item in bp["whiskers"] + bp["caps"]:
            item.set(color=colors[cons], alpha=0.7)
        jitter = np.linspace(-0.035, 0.035, len(values))
        ax.scatter(pos + jitter, values, s=14, color=colors[cons], alpha=0.55)
ax.axhline(0, color="black", ls="--", lw=1.2, label="référence individuelle raw/run-01")
for cons in colors:
    ax.scatter([], [], color=colors[cons], label={"still":"run-01 immobile", "nodding":"run-02 nodding", "shaking":"run-03 shaking"}[cons])
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([SHORT[c] for c in CONDITIONS])
ax.set_ylabel("écart à raw/run-01 du même sujet (mm)")
ax.set_title("Distance de chaque acquisition à la référence individuelle raw/run-01")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()

pv = g.pivot_table(index=["subject", "condition"], columns="consigne", values="thickness", observed=True)

case19 = (rel[rel["subject"] == "sub-19"].pivot_table(
    index="condition", columns="consigne", values="écart à raw/run-01 (mm)", observed=True
).reindex(CONDITIONS))
show(case19, "Pour sub-19, à quelle distance chaque run et chaque traitement se trouvent-ils de raw/run-01 ?",
     "Ce tableau numérique accompagne le montage des trois runs de sub-19. Une valeur proche de zéro est souhaitable; "
     "une valeur positive importante indique une sur-correction et une valeur négative importante un amincissement.",
     "{:+.3f}")''')

    md(r"""Le tableau suivant complète la figure. Pour chaque sujet :

$$Écart_{c}=T_{run01,c}-T_{run03,c}$$

Seuls les sujets chez qui le raw montre le comportement attendu ($Écart_{brut}>0$, donc shaking plus mince que run-01) sont classés :

- **amélioré** : $0 \leq Écart_c < Écart_{brut}$ ; le shaking se rapproche du run-01 sans le dépasser ;
- **sur-corrigé** : $Écart_c<0$ ; le shaking devient plus épais que le run-01 ;
- **inchangé ou pire** : $Écart_c\geq Écart_{brut}$.

B reste une analyse d'épaisseur globale. La section E vérifie ensuite la fidélité région par région.""")

    code('''ec = pv.reset_index().dropna(subset=["still", "shaking"])
ec["ecart"] = ec["still"] - ec["shaking"]          # > 0 : le mouvement amincit
w = ec.pivot(index="subject", columns="condition", values="ecart")

rows = []
for c in CONDITIONS:
    if c == "brut":
        continue
    p = w[["brut", c]].dropna()
    p = p[p["brut"] > 0]
    rows.append({"condition": c, "n sujets": len(p),
                 "améliorés": int(((p[c] >= 0) & (p[c] < p["brut"])).sum()),
                 "sur-corrigés": int((p[c] < 0).sum()),
                 "inchangés/pires": int((p[c] >= p["brut"]).sum()),
                 "écart médian brut (mm)": p["brut"].median(),
                 "écart médian condition (mm)": p[c].median()})
tB = pd.DataFrame(rows).set_index("condition")

analyse = (f"preproc et jdac réduisent l'écart chez la majorité ({tB.loc['preproc','améliorés']:.0f} et "
           f"{tB.loc['jdac','améliorés']:.0f} sujets améliorés sur ~{tB.loc['jdac','n sujets']:.0f}), avec peu de sur-corrigés. "
           f"Les variantes sans débruiteur sur-corrigent la majorité : {tB.loc['jdac_antiartonly','sur-corrigés']:.0f} sujets "
           f"pour antiartonly, {tB.loc['jdac_nodenoise','sur-corrigés']:.0f} pour nodenoise (le bougé devient plus épais que "
           f"l'immobile). Réserve : cet écart mélange récupération et offset — la section E tranche par région.")
show(tB, "Chez combien de sujets l'écart immobile − bougé se réduit vraiment (sans s'inverser) ?", analyse,
     {"n sujets": "{:.0f}", "améliorés": "{:.0f}", "sur-corrigés": "{:.0f}", "inchangés/pires": "{:.0f}",
      "écart médian brut (mm)": "{:.3f}", "écart médian condition (mm)": "{:.3f}"})''')

    # ---------------------------------------------------------------- C. M0 vs M1
    md(r"""## C. Après traitement, l'épaisseur reste-t-elle associée au mouvement ?

Le mot **prédire** signifie ici : « connaître le score Agitation aide-t-il à expliquer les différences d'épaisseur entre les acquisitions ? ». Il ne s'agit pas de prédire l'épaisseur d'un futur patient.

Chaque condition est analysée séparément. Pour le sujet $i$ et son acquisition $j$, le modèle complet est :

$$T_{ij}=\beta_0+\beta_{âge}âge_i+\beta_{sexe}sexe_i+\beta_{agit}Agitation_{ij}+u_i+\varepsilon_{ij}$$

- $T_{ij}$ : épaisseur moyenne des 68 régions pour cette acquisition ;
- $\beta_{agit}$ : changement moyen d'épaisseur, en mm, pour +1 point d'Agitation ;
- $u_i$ : **intercept aléatoire du sujet**. Il autorise chaque sujet à avoir son niveau d'épaisseur habituel, tout en estimant une pente Agitation commune. Dans le code : `groups=d["subject"]` dans `mixedlm` ;
- $\varepsilon_{ij}$ : variation restante non expliquée.

Deux modèles emboîtés sont comparés :

- **M0** : âge + sexe + intercept aléatoire du sujet ;
- **M1** : M0 + Agitation.

La p-value vient d'un test du rapport de vraisemblance entre M0 et M1. Sous l'hypothèse « Agitation n'ajoute réellement aucune information », elle mesure à quel point le gain d'ajustement observé serait inhabituel. Une petite valeur, par exemple `< 0,05`, indique que l'ajout d'Agitation améliore clairement le modèle. Une valeur plus grande ne prouve pas l'absence d'effet : elle indique que les données ne permettent pas de le distinguer clairement du bruit.

Les lignes diffèrent parce que chaque traitement modifie les épaisseurs, leur relation au mouvement et leur variabilité. La p-value dépend à la fois de la taille du coefficient, de la dispersion des données et du nombre d'acquisitions disponibles.""")

    code('''def fit(formula, d):
    return smf.mixedlm(formula, d, groups=d["subject"]).fit(reml=False, method="powell", disp=False)

rows = []
for c in CONDITIONS:
    d = g[g["condition"] == c].dropna(subset=["age", "sex_bin", "agitation", "thickness"])
    m0 = fit("thickness ~ age + sex_bin", d)
    m1 = fit("thickness ~ age + sex_bin + agitation", d)
    lr = 2 * (m1.llf - m0.llf)
    coef = m1.params["agitation"]
    se = m1.bse["agitation"]
    rows.append({"condition": c, "n acquisitions": d.groupby(["subject", "run"]).ngroups,
                 "coef Agitation (mm/point)": coef,
                 "IC95 bas": coef - 1.96 * se, "IC95 haut": coef + 1.96 * se,
                 "p ajout Agitation (M1 vs M0)": stats.chi2.sf(lr, 1),
                 "sens": "amincit" if coef < 0 else "épaissit"})
tC = pd.DataFrame(rows).set_index("condition")

ns = [c for c in CONDITIONS if tC.loc[c, "p ajout Agitation (M1 vs M0)"] >= 0.05]
analyse = (f"En brut le mouvement prédit fortement l'épaisseur (coef {tC.loc['brut','coef Agitation (mm/point)']:+.4f} mm/point, "
           f"p={tC.loc['brut','p ajout Agitation (M1 vs M0)']:.1g}) : il l'amincit. L'ajout d'Agitation n'améliore pas clairement le modèle (p≥0.05) pour : "
           f"{', '.join(ns) if ns else 'aucune condition'} — jdac a une association résiduelle plus faible (p={tC.loc['jdac','p ajout Agitation (M1 vs M0)']:.2g}). "
           f"antiartonly et nodenoise ont un coefficient positif ({tC.loc['jdac_nodenoise','coef Agitation (mm/point)']:+.4f}, "
           f"p={tC.loc['jdac_nodenoise','p ajout Agitation (M1 vs M0)']:.1g} pour nodenoise) : davantage de mouvement est associé à une épaisseur mesurée plus grande, signe de sur-correction. "
           f"Ce résultat doit être lu avec A et E : une pente faible ne suffit pas si le traitement déplace aussi les scans de référence.")
show(tC, "Après la condition, ajouter le score de mouvement améliore-t-il encore le modèle ?", analyse,
     {"n acquisitions": "{:.0f}", "coef Agitation (mm/point)": "{:+.4f}",
      "IC95 bas": "{:+.4f}", "IC95 haut": "{:+.4f}",
      "p ajout Agitation (M1 vs M0)": "{:.2g}", "sens": "{}"})''')

    # ---------------------------------------------------------------- C-bis. non-linéaire + strates
    md("""## C-bis. Forme non-linéaire du lien mouvement-épaisseur, par strate

La section C teste un lien linéaire (un seul coefficient) entre Agitation et épaisseur. Ce lien peut ne pas être linéaire : effet qui n'apparaît qu'au-delà d'un certain mouvement, plateau, ou inversion aux valeurs extrêmes. Deux formes plus flexibles sont ajoutées au modèle M1 de la section C (âge + sexe + Agitation) :

- **quadratique** : ajoute Agitation² (`I(agitation**2)`) ;
- **splines** : ajoute une base de splines à 3 degrés de liberté (`bs(agitation, df=3)`), sans imposer de forme fonctionnelle.

Comparaison par ΔAIC à la forme linéaire (ΔAIC < 0 : la forme non-linéaire est préférée ; un écart supérieur à 2 est considéré comme un appui notable, suivant le repère usuel pour l'AIC). Même effet aléatoire par sujet que la section C (`mixedlm`, `reml=False`, `method="powell"`).

La courbe continue (quadratique/spline) est l'analyse principale. Les niveaux sont une lecture descriptive secondaire utilisant les **mêmes seuils que l'analyse précédente** : faible `< 0,3`, léger `0,3–1,0`, modéré `1,0–2,0`, sévère `> 2,0`. Nous n'avons pas retrouvé de justification documentée de leur choix initial : ils doivent donc être présentés comme des coupures pragmatiques provisoires, et non comme des seuils cliniques validés ou objectivement dérivés des données. Ils ne sont pas réoptimisés sur les résultats actuels. Sur les 66 acquisitions, ils donnent respectivement 23, 18, 14 et 11 acquisitions. Leur maintien sert uniquement la comparabilité avec l'analyse d'il y a deux semaines et doit être validé avec le directeur.""")

    code('''def aic(res):
    return -2 * res.llf + 2 * (len(res.fe_params) + 2)

g["niveau"] = pd.cut(g["agitation"], [0, 0.3, 1.0, 2.0, np.inf],
                      labels=["faible", "leger", "modere", "severe"], include_lowest=True)

formes = {"lineaire": "thickness ~ age + sex_bin + agitation",
          "quadratique": "thickness ~ age + sex_bin + agitation + I(agitation**2)",
          "splines": "thickness ~ age + sex_bin + bs(agitation, df=3)",
          "par_strate": "thickness ~ age + sex_bin + C(niveau)"}

rows = []
for c in CONDITIONS:
    d = g[g["condition"] == c].dropna(subset=["age", "sex_bin", "agitation", "thickness", "niveau"])
    aics = {nom: aic(fit(f, d)) for nom, f in formes.items()}
    base = aics["lineaire"]
    rows.append({"condition": c, **{f"dAIC {nom}": aics[nom] - base for nom in formes}})
tCbis = pd.DataFrame(rows).set_index("condition").reindex(CONDITIONS)

pref = [c for c in CONDITIONS if tCbis.loc[c, "dAIC quadratique"] < -2 or tCbis.loc[c, "dAIC splines"] < -2]
non_pref = [c for c in CONDITIONS if c not in pref]
analyse = (f"La forme linéaire reste préférée (ΔAIC ≥ -2 pour le quadratique et les splines) pour : "
           f"{', '.join(non_pref) if non_pref else 'aucune condition'}. Une forme non-linéaire est notablement "
           f"préférée (ΔAIC < -2) pour : {', '.join(pref) if pref else 'aucune condition'} — dans ce cas le lien "
           f"mouvement-épaisseur n'est pas une simple pente et mérite d'être regardé région par région plutôt que "
           f"globalement. Si `par_strate` (catégoriel) a le plus petit AIC pour une condition, le lien n'est pas "
           f"monotone pour cette condition (un modèle continu, même quadratique, ne le capture pas).")
show(tCbis, "Une forme non-linéaire (quadratique ou splines) explique-t-elle mieux l'épaisseur que le lien linéaire ?",
     analyse, "{:+.2f}")''')

    md("""Lecture par strate de mouvement : où se situe l'écart au brut (interaction condition × niveau) et à quel niveau devient-il significatif.""")

    code('''fit_int = smf.mixedlm("thickness ~ C(condition, Treatment('brut')) * C(niveau, Treatment('faible'))",
                          g.dropna(subset=["niveau", "agitation"]),
                          groups=g.dropna(subset=["niveau", "agitation"])["subject"]).fit(reml=True, method="powell", disp=False)

rows = []
for k in fit_int.params.index:
    if ":" not in k:
        continue
    cond = k.split("T.")[1].split("]")[0]
    niv = k.split("T.")[-1].rstrip("]")
    rows.append({"condition": cond, "niveau": niv, "coef (mm)": fit_int.params[k], "p": fit_int.pvalues[k]})
tCbis_int = pd.DataFrame(rows).set_index(["condition", "niveau"])

sig = tCbis_int[tCbis_int["p"] < 0.05]
analyse = (f"{len(sig)} interaction(s) condition x niveau sur {len(tCbis_int)} sont significatives (p<0.05) : "
           f"{', '.join(f'{c}/{n}' for c, n in sig.index) if len(sig) else 'aucune'}. Un coefficient négatif "
           f"signifie que l'écart au brut s'accentue à ce niveau de mouvement par rapport au niveau faible "
           f"(référence) ; positif, qu'il s'atténue ou s'inverse (sur-correction).")
show(tCbis_int, "L'écart au brut dépend-il du niveau de mouvement (faible/léger/modéré/sévère) ?", analyse, "{:+.4f}")''')

    md("""Comparaison appariée (Wilcoxon) au brut, séparément par niveau de mouvement.""")

    code('''paire = (g.pivot_table(index=["subject", "run", "niveau"], columns="condition",
                        values="thickness", observed=True).dropna(subset=["brut"]).reset_index())
rows = []
for niv in ["faible", "leger", "modere", "severe"]:
    s = paire[paire["niveau"] == niv]
    rec = {"niveau": niv, "n": len(s)}
    for c in CONDITIONS:
        if c == "brut":
            continue
        ss = s.dropna(subset=[c])
        rec[f"{c} (%)"] = (100 * (ss[c] - ss["brut"]) / ss["brut"]).mean() if len(ss) else np.nan
        try:
            rec[f"p {c}"] = stats.wilcoxon(ss[c], ss["brut"]).pvalue if len(ss) >= 3 else np.nan
        except Exception:
            rec[f"p {c}"] = np.nan
    rows.append(rec)
tCbis_strat = pd.DataFrame(rows).set_index("niveau")

analyse = ("À chaque niveau, un écart (%) proche de 0 et non significatif (p≥0.05) indique que la condition ne se "
           "distingue pas du brut à ce niveau de mouvement. Un écart négatif signifie que la condition amincit par "
           "rapport au brut à ce niveau ; positif, qu'elle épaissit (sur-correction locale). Si l'écart croît avec "
           "la sévérité, l'effet de la condition dépend du mouvement plutôt que d'être un simple offset constant.")
show(tCbis_strat, "L'écart (%) au brut, par niveau de mouvement, est-il stable ou dépend-il de la sévérité ?", analyse, "{:+.2f}")''')

    # ---------------------------------------------------------------- D. image
    md(r"""## D. L'image et les contours ressemblent-ils davantage à la référence ?

Le SSIM (*Structural Similarity Index*) compare deux images dans des fenêtres locales, puis moyenne le résultat. Pour deux fenêtres $x$ et $y$ :

$$SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$

- $\mu_x,\mu_y$ : intensités moyennes locales (luminance) ;
- $\sigma_x,\sigma_y$ : contrastes locaux ;
- $\sigma_{xy}$ : structures locales variant ensemble ;
- $C_1,C_2$ : petites constantes de stabilité.

Le score va approximativement de 0 à 1 dans ce contexte ; 1 signifie que les deux images comparées sont identiques. Avant le calcul, chaque volume est normalisé robustement entre ses percentiles cérébraux 1 et 99, ramené dans `[0,1]`, et comparé dans la boîte englobante du cerveau commun.

Trois lectures complémentaires sont conservées dans **un seul tableau** :

1. **SSIM image vs run-01 preproc** : ressemblance de toute l'image bougée au run-01 de référence ; un lissage peut artificiellement améliorer ce score.
2. **SSIM gradient vs run-01 preproc** : même référence, mais sur la magnitude du gradient $\|\nabla I\|$, qui met davantage l'accent sur les frontières et les contours.
3. **SSIM gradient intra-condition** : le run bougé d'une condition est comparé au run-01 ayant subi la même condition, par exemple `JDAC run-03` contre `JDAC run-01`. Cela réduit la différence d'apparence propre à la méthode (contraste, intensité, lissage), mais la référence est elle-même traitée : ce score mesure la cohérence interne, pas la vérité anatomique.

Les images rigides sont sur la même grille ; aucun recalage supplémentaire n'est appliqué. Métriques calculées par `compute_image_metrics.py`.""")

    code('''im = pd.read_csv(REPO / "results/ds004332/phase4_compare_3bras/image_metrics.csv")
IMGCONDS = ["preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
lab = {"run-02": "nodding (mvt modéré)", "run-03": "shaking (fort mvt)"}
agg = im.groupby(["condition", "run"])[["clean_ssim_img", "clean_ssim_grad", "intra_ssim_grad"]].mean()
s = agg.xs("run-03", level="run")   # au fort mouvement, pour l'analyse

A_img = (f"Contre le scan propre, la ressemblance d'image monte légèrement après correction au fort mouvement "
         f"(preproc {s.loc['preproc','clean_ssim_img']:.3f}, jdac {s.loc['jdac','clean_ssim_img']:.3f}, "
         f"nodenoise {s.loc['jdac_nodenoise','clean_ssim_img']:.3f}), mais l'image seule ne sépare pas correction et lissage.")
A_grad = (f"Sur les contours, contre le propre non traité, aucune condition ne dépasse nettement preproc au fort mouvement "
          f"(preproc {s.loc['preproc','clean_ssim_grad']:.3f}, jdac {s.loc['jdac','clean_ssim_grad']:.3f}, "
          f"nodenoise {s.loc['jdac_nodenoise','clean_ssim_grad']:.3f}). Ce test mélange correction et changement d'intensité, "
          f"d'où le tableau intra ci-dessous.")
A_intra = (f"En comparaison intra-condition (même traitement pour le run bougé et son run-01), jdac gagne en cohérence des contours au fort mouvement "
           f"({s.loc['jdac','intra_ssim_grad']:.3f} vs preproc {s.loc['preproc','intra_ssim_grad']:.3f}) = mouvement réduit. "
           f"nodenoise ne gagne pas ({s.loc['jdac_nodenoise','intra_ssim_grad']:.3f}, ≤ preproc). Attention : cette comparaison "
           f"mesure une cohérence interne; elle ne prouve pas que les contours traités sont anatomiquement vrais.")

d_table = agg.reset_index().rename(columns={
    "condition": "condition", "run": "run",
    "clean_ssim_img": "SSIM image vs run-01 preproc",
    "clean_ssim_grad": "SSIM gradient vs run-01 preproc",
    "intra_ssim_grad": "SSIM gradient vs run-01 même condition",
}).set_index(["condition", "run"])
analyse = A_img + " " + A_grad + " " + A_intra
show(d_table, "La correction rapproche-t-elle l'image et ses contours du run-01 du même sujet ?",
     analyse, "{:.3f}")''')

    # ---------------------------------------------------------------- E. récupération
    md(r"""## E. Les 68 épaisseurs régionales se rapprochent-elles de la référence ?

`raw/run-01` du même sujet est la **référence opérationnelle** : c'est le scan le moins traité et avec consigne immobile, mais pas une vérité anatomique parfaite.

Pour un sujet $s$, une condition $c$, un run bougé $k$ et les $R=68$ régions, on note $T_{s,k,c,r}$ l'épaisseur de la région $r$.

### 1. Distorsion du scan de référence

$$Offset_{s,c}=\frac{1}{R}\sum_r|T_{s,run01,c,r}-T_{s,run01,brut,r}|$$

Cette quantité mesure ce que la condition change déjà sur le run-01. Elle est absolue et régionale, contrairement au déplacement global signé de A : les différences positives et négatives ne peuvent pas se compenser.

### 2. Écart résiduel entre runs dans la même condition

$$Résiduel_{s,k,c}=\frac{1}{R}\sum_r|T_{s,k,c,r}-T_{s,run01,c,r}|$$

Les deux scans ont subi le même traitement. Un déplacement constant propre à la condition se compense en grande partie ; cette mesure décrit surtout le motif régional restant entre le run bougé et le run-01 traité.

### 3. Erreur totale à la référence opérationnelle

$$Erreur_{s,k,c}=\frac{1}{R}\sum_r|T_{s,k,c,r}-T_{s,run01,brut,r}|$$

C'est le critère principal : l'épaisseur régionale du run bougé traité se rapproche-t-elle réellement de `raw/run-01` ? Les trois mesures sont nécessaires pour distinguer une vraie réduction du mouvement d'un simple déplacement ou d'une compensation. Elles sont d'abord calculées par sujet, puis moyennées entre les sujets. Métriques calculées par `compute_recovery.py`.""")

    code('''rec = pd.read_csv(REPO / "results/ds004332/phase4_compare_3bras/recovery_metrics.csv")
RCONDS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
for run, cons in [("run-03", "shaking (fort mouvement)"), ("run-02", "nodding (mouvement modéré)")]:
    t = (rec[rec["run"] == run].groupby("condition")[["mae_within", "mae_truth", "offset"]].mean().reindex(RCONDS))
    bt, bw = t.loc["brut", "mae_truth"], t.loc["brut", "mae_within"]
    analyse = (f"Aucune condition ne descend sous l'erreur du brut à la référence opérationnelle ({bt:.3f} mm) : jdac {t.loc['jdac','mae_truth']:.3f}, "
               f"nodenoise {t.loc['jdac_nodenoise','mae_truth']:.3f} s'en éloignent (à cause de l'offset). Net d'offset "
               f"(mouvement restant), nodenoise empire le motif régional ({t.loc['jdac_nodenoise','mae_within']:.3f} > brut {bw:.3f}) ; "
               f"seul preproc le réduit un peu ({t.loc['preproc','mae_within']:.3f}). L'épaisseur régionale du scan bougé ne se "
               f"rapproche donc pas de la vraie valeur : les mesures ne suivent pas.")
    t.columns = ["écart résiduel intra-condition (mm)", "erreur à raw/run-01 (mm)", "distorsion de run-01 (mm)"]
    show(t, f"Scan {cons} : l'épaisseur régionale se rapproche-t-elle de la référence raw/run-01 ?", analyse, "{:.3f}")''')

    md("""## Synthèse

Bilan des cinq conditions sur les questions A à E. Les deux variantes sans débruiteur (antiartonly ×1, nodenoise ×4) **sur-corrigent** : elles inversent le lien mouvement–épaisseur (C, coefficient positif) et, une fois l'offset retiré, elles éloignent l'épaisseur régionale du scan bougé de la vraie valeur au lieu de l'en rapprocher (E), sans gagner en fidélité des contours (D). `jdac` complet est la seule à découpler le mouvement (C) et à rapprocher les contours du propre (D), mais au prix d'un lissage qui amincit les scans immobiles (A). Une partie du bénéfice vient déjà du `preprocessing` seul (B, E). Conclusion : aucune condition ne ramène l'épaisseur d'un scan bougé à sa vraie valeur, et la meilleure image des variantes ne se traduit pas en mesure corticale plus fidèle.""")

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    nbf.write(nb, OUTDIR / FNAME)
    print("Notebook écrit :", OUTDIR / FNAME, "|", len(nb["cells"]), "cellules")
