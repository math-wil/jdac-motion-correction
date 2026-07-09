"""Bloc à coller dans build_notebook.py (fonction build(), juste après la section C
et avant la section D), pour générer la section C-bis dans explore_epaisseur_rigide.ipynb :
forme non-lineaire du lien mouvement-epaisseur (quadratique, splines) + lecture par strate
de mouvement, reprenant la logique de nonlineaire() et strates() dans compare_conditions.py.

Prérequis déjà présents dans le notebook au moment où ce bloc s'exécute (cellules A à C) :
- `g` : dataframe agrégé (une ligne par sujet x run x condition), colonnes subject, run,
  condition, thickness, agitation, age, sex_bin, consigne ;
- `fit(formula, d)` : wrapper mixedlm (reml=False, method="powell"), défini dans la cellule C ;
- `CONDITIONS`, `show(df, question, analyse, fmt)`, imports numpy/pandas/scipy.stats/statsmodels.
"""

    # ---------------------------------------------------------------- C-bis. non-linéaire + strates
    md("""## C-bis. Forme non-linéaire du lien mouvement-épaisseur, par strate

La section C teste un lien linéaire (un seul coefficient) entre Agitation et épaisseur. Ce lien peut ne pas être linéaire : effet qui n'apparaît qu'au-delà d'un certain mouvement, plateau, ou inversion aux valeurs extrêmes. Deux formes plus flexibles sont ajoutées au modèle M1 de la section C (âge + sexe + Agitation) :

- **quadratique** : ajoute Agitation² (`I(agitation**2)`) ;
- **splines** : ajoute une base de splines à 3 degrés de liberté (`bs(agitation, df=3)`), sans imposer de forme fonctionnelle.

Comparaison par ΔAIC à la forme linéaire (ΔAIC < 0 : la forme non-linéaire est préférée ; un écart supérieur à 2 est considéré comme un appui notable, suivant le repère usuel pour l'AIC). Même effet aléatoire par sujet que la section C (`mixedlm`, `reml=False`, `method="powell"`). Niveaux de mouvement (mêmes seuils que le score Agitation) : faible (< 0.3), léger (0.3-1.0), modéré (1.0-2.0), sévère (> 2.0).""")

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

