#!/usr/bin/env python3
"""Génère le notebook morphométrique descriptif : explore_morphometrie.ipynb

But (réunion du 24 juillet) : placer côte à côte des boxplots STRICTEMENT comparables
pour épaisseur, surface corticale, volume cortical et SubCortGrayVol, plus quelques
structures sous-corticales. Chaque panneau utilise exactement la définition du boxplot
d'épaisseur de la phase 4 : distance SIGNÉE de chaque acquisition à brut/run-01 du même
sujet (valeur(acquisition) − valeur(brut/run-01)).

Source unique : results/ds004332/phase5_fidelity/morphometry_long.csv, produite par
extract_morphometry_stats.py. Aucune analyse Agitation, aucun test, aucune p-value.
"""

from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("explore_morphometrie.ipynb")


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: cells.append(nbf.v4.new_code_cell(t))

    md("""# Morphométrie descriptive des cinq conditions

**Question.** Les effets déjà vus sur l'épaisseur corticale se retrouvent-ils, dans le même sens, sur la surface corticale, le volume cortical et les volumes sous-corticaux ?

**Définition unique, identique au boxplot d'épaisseur de la phase 4.** Pour chaque acquisition (sujet × run × condition), on calcule la **distance signée** à la référence individuelle :

distance = valeur(acquisition) − valeur(**brut/run-01 du même sujet**)

- La référence est toujours `brut/run-01` du même sujet. C'est le scan le moins traité et le plus immobile, une référence opérationnelle, pas une vérité anatomique parfaite.
- Distance **signée** : négatif = la mesure a diminué par rapport au brut immobile, positif = elle a augmenté. `brut/run-01` vaut donc 0 par construction.
- Unité statistique : une valeur par acquisition ; les boxplots montrent la distribution entre sujets.

**Quatre mesures globales.** L'agrégation entre régions diffère selon la mesure, et c'est volontaire :

- **épaisseur** : moyenne des régions corticales (mm), comme dans la phase 4 ;
- **surface corticale** : somme des régions (mm²), car une surface est additive (le total est la mesure globale naturelle, pas une moyenne par région) ;
- **volume cortical** : somme des régions (mm³), additif de même ;
- **SubCortGrayVol** : mesure globale unique de `aseg.stats` (mm³), aucune agrégation.

**Source unique** : `results/ds004332/phase5_fidelity/morphometry_long.csv`, produite par `extract_morphometry_stats.py` (une seule table pour les cinq conditions). Aucun modèle Agitation, aucun test, aucune p-value : analyse descriptive seulement.""")

    code('''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

HOME = Path.home()
def find_repo():
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "results/ds004332/phase5_fidelity").exists():
            return p
    return HOME / "Documents/jdac-motion-correction"
REPO = find_repo()
MORPH = REPO / "results/ds004332/phase5_fidelity/morphometry_long.csv"

CONDITIONS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
SHORT = {"brut":"brut", "preproc":"prep", "jdac":"jdac",
         "jdac_antiartonly":"aa×1", "jdac_nodenoise":"aa×4"}
CONSIGNE = {"run-01":"still", "run-02":"nodding", "run-03":"shaking"}
COLORS = {"still":"tab:green", "nodding":"tab:orange", "shaking":"tab:red"}
OFFSETS = {"still":-0.23, "nodding":0.0, "shaking":0.23}

# Garde : la source doit exister. Sinon on indique exactement quoi exécuter.
if not MORPH.is_file():
    print("Fichier source absent :", MORPH)
    print()
    print("Il est produit par extract_morphometry_stats.py à partir des sorties FreeSurfer")
    print("(lh/rh.aparc.stats pour surface et volume cortical, aseg.stats pour les volumes)")
    print("des CINQ conditions. Surface et volume cortical ne sont PAS disponibles en local")
    print("pour l'instant ; il faut lancer l'extraction sur Narval ou le PC labo :")
    print()
    print("  python pipelines/ds004332/phase5_fidelity/extract_morphometry_stats.py \\\\")
    print("    --root brut=<SUBJECTS_DIR_brut> \\\\")
    print("    --root preproc=<SUBJECTS_DIR_preproc> \\\\")
    print("    --root jdac=<SUBJECTS_DIR_jdac> \\\\")
    print("    --root jdac_antiartonly=<SUBJECTS_DIR_antiartonly> \\\\")
    print("    --root jdac_nodenoise=<SUBJECTS_DIR_nodenoise>")
    print()
    print("Colonnes attendues : subject, run, condition, family, hemi, region, metric, value, unit.")
    print("Relancer ce notebook une fois le CSV présent.")
    raise SystemExit("morphometry_long.csv manquant : voir le message ci-dessus.")

d = pd.read_csv(MORPH)
d = d[d["condition"].isin(CONDITIONS)].copy()
d["consigne"] = d["run"].map(CONSIGNE)
print("Source :", MORPH)
print(f"{len(d):,} lignes | {d['subject'].nunique()} sujets | "
      f"{d[['subject','run','condition']].drop_duplicates().shape[0]} acquisitions")
print("familles :", ", ".join(sorted(d["family"].unique())))''')

    md("""## Outils communs

Deux fonctions réutilisées par toutes les figures : agréger une mesure par acquisition, puis calculer la distance signée à `brut/run-01`. Le boxplot est identique en forme à celui de la phase 4 (une couleur par consigne, points par sujet, ligne de référence à 0).""")

    code('''def par_acquisition(d, family, metric, agg, region=None, positive_only=False):
    """Une valeur par (sujet, run, condition) : agg des régions d'une mesure."""
    sub = d[(d["family"] == family) & (d["metric"] == metric)].copy()
    if region is not None:
        sub = sub[sub["region"] == region]
    if positive_only:
        sub = sub[sub["value"] > 0]
    per = (sub.groupby(["subject","run","condition"], observed=True)["value"]
              .agg(agg).reset_index(name="value"))
    return per

def distance_a_reference(per):
    """Ajoute la distance signée à brut/run-01 du même sujet."""
    ref = (per[(per["condition"] == "brut") & (per["run"] == "run-01")]
           [["subject","value"]].rename(columns={"value":"ref"}))
    per = per.merge(ref, on="subject", how="inner")
    per["dist"] = per["value"] - per["ref"]
    per["consigne"] = per["run"].map(CONSIGNE)
    return per

def box_distance(ax, per, titre, unite):
    for i, c in enumerate(CONDITIONS):
        for cons in ["still","nodding","shaking"]:
            vals = per[(per["condition"] == c) & (per["consigne"] == cons)]["dist"].dropna()
            if vals.empty:
                continue
            pos = i + OFFSETS[cons]
            bp = ax.boxplot(vals, positions=[pos], widths=0.18, patch_artist=True,
                            showfliers=False, medianprops={"color":"black","lw":1.3})
            bp["boxes"][0].set(facecolor=COLORS[cons], alpha=0.28, edgecolor=COLORS[cons])
            for it in bp["whiskers"] + bp["caps"]:
                it.set(color=COLORS[cons], alpha=0.7)
            jitter = np.linspace(-0.035, 0.035, len(vals))
            ax.scatter(pos + jitter, vals, s=11, color=COLORS[cons], alpha=0.5)
    ax.axhline(0, color="black", ls="--", lw=1.1)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([SHORT[c] for c in CONDITIONS], rotation=20)
    ax.set_title(titre, fontsize=12)
    ax.set_ylabel(f"écart à brut/run-01 ({unite})")
    ax.grid(axis="y", alpha=0.18)

def legende_consignes(fig):
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c],
               label={"still":"run-01 immobile","nodding":"run-02 nodding","shaking":"run-03 shaking"}[c])
               for c in ["still","nodding","shaking"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=11)''')

    md("""## Figure principale : quatre mesures globales côte à côte

**Question posée.** Sur un scan immobile (points verts), le traitement déplace-t-il déjà la mesure par rapport au brut, et sur les scans bougés (orange, rouge) se rapproche-t-il ou s'éloigne-t-il de la référence ?

**Comment lire.** La ligne pointillée à 0 est le brut/run-01 du sujet. Une boîte proche de 0 = mesure proche de la référence. Une boîte **sous** 0 = mesure plus petite que le brut immobile (par exemple amincissement ou perte de volume) ; **au-dessus** de 0 = plus grande. La cohérence attendue est que épaisseur, surface et volume cortical bougent dans le même sens.""")

    code('''MESURES = [
    ("Épaisseur corticale (moyenne régions)", "mm",
     par_acquisition(d, "cortical_region", "thickness", "mean", positive_only=True)),
    ("Surface corticale (somme régions)", "mm²",
     par_acquisition(d, "cortical_region", "surface_area", "sum")),
    ("Volume cortical gris (somme régions)", "mm³",
     par_acquisition(d, "cortical_region", "cortical_gray_volume", "sum")),
    ("SubCortGrayVol (aseg global)", "mm³",
     par_acquisition(d, "aseg_global", "volume", "sum", region="SubCortGrayVol")),
]
manquantes = [titre for titre, _, per in MESURES if per.empty]
if manquantes:
    display(Markdown("**Mesures absentes de la source (panneau vide) :** " + ", ".join(manquantes)
                     + ". Vérifier que `lh/rh.aparc.stats` ont bien été extraits pour les cinq conditions."))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (titre, unite, per) in zip(axes.flat, MESURES):
    if per.empty:
        ax.set_title(titre + " (données absentes)", fontsize=12); ax.set_axis_off(); continue
    box_distance(ax, distance_a_reference(per), titre, unite)
legende_consignes(fig)
fig.suptitle("Distance signée à brut/run-01 du même sujet, par condition et par consigne", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()''')

    md("""## Figure complémentaire : quelques structures sous-corticales

**Question posée.** Le même déplacement se voit-il structure par structure (thalamus, hippocampe, putamen, ventricule latéral) ?

**Comment lire.** Volumes bilatéraux (gauche + droit) en mm³, même distance signée à brut/run-01. Sous 0 = structure plus petite qu'au brut immobile ; au-dessus = plus grande. Un ventricule qui grossit et une structure grise qui rétrécit sont des signes classiques de lissage ou d'érosion, à confronter avec la figure principale.""")

    code('''REGIONS = [
    ("Thalamus (G+D)", "Thalamus"),
    ("Hippocampe (G+D)", "Hippocampus"),
    ("Putamen (G+D)", "Putamen"),
    ("Ventricule latéral (G+D)", "Lateral-Ventricle"),
]
aseg_reg = d[d["family"] == "aseg_region"].copy()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (titre, motif) in zip(axes.flat, REGIONS):
    sub = aseg_reg[aseg_reg["region"].str.contains(motif, na=False)]
    if sub.empty:
        ax.set_title(titre + " (absent de aseg)", fontsize=12); ax.set_axis_off(); continue
    per = (sub.groupby(["subject","run","condition"], observed=True)["value"]
              .sum().reset_index(name="value"))
    box_distance(ax, distance_a_reference(per), titre, "mm³")
legende_consignes(fig)
fig.suptitle("Structures sous-corticales : distance signée à brut/run-01", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()''')

    md("""## Contrôle : SubCortGrayVol sujet par sujet, avant toute interprétation

La réunion a signalé que JDAC complet et anti-artefact ×1 semblaient réduire `SubCortGrayVol` de façon **similaire** sur le scan immobile, alors que les traitements diffèrent. Avant d'interpréter, il faut vérifier si cette ressemblance tient **sujet par sujet** ou seulement en médiane.

**Comment lire le nuage.** Un point = un sujet, sur `run-01`. En abscisse la distance de `aa×1` à brut/run-01, en ordonnée celle de `jdac`. Sur la diagonale `y = x`, les deux traitements déplacent ce sujet de la même quantité. Un nuage serré sur la diagonale confirme une ressemblance réelle ; un nuage dispersé indique que la ressemblance médiane est en partie une coïncidence d'agrégation.""")

    code('''scv = par_acquisition(d, "aseg_global", "volume", "sum", region="SubCortGrayVol")
if scv.empty:
    display(Markdown("**SubCortGrayVol absent de la source.** Contrôle impossible tant que `morphometry_long.csv` n'est pas produit."))
else:
    scv = distance_a_reference(scv)
    r01 = scv[scv["run"] == "run-01"].pivot_table(index="subject", columns="condition",
                                                   values="dist", observed=True)
    paire = r01[["jdac_antiartonly", "jdac"]].dropna()

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(paire["jdac_antiartonly"], paire["jdac"], s=30, alpha=0.75)
    lim = np.nanmax(np.abs(paire.values)) * 1.1 if len(paire) else 1
    ax.plot([-lim, lim], [-lim, lim], color="black", ls="--", lw=1, label="y = x (même déplacement)")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("aa×1 : écart SubCortGrayVol à brut/run-01 (mm³)")
    ax.set_ylabel("jdac : écart SubCortGrayVol à brut/run-01 (mm³)")
    ax.set_title("SubCortGrayVol sur run-01, sujet par sujet : jdac vs aa×1")
    ax.legend(frameon=False, fontsize=10); ax.set_aspect("equal", "box")
    plt.tight_layout(); plt.show()

    tab = r01[["jdac", "jdac_antiartonly"]].copy()
    tab.columns = ["jdac (mm³)", "aa×1 (mm³)"]
    tab["écart jdac − aa×1 (mm³)"] = tab["jdac (mm³)"] - tab["aa×1 (mm³)"]
    display(tab.round(0))
    display(Markdown(
        f"Médianes sur run-01 : jdac {r01['jdac'].median():+.0f} mm³, aa×1 {r01['jdac_antiartonly'].median():+.0f} mm³. "
        "Regarder si les points suivent la diagonale (ressemblance réelle) ou non, et si le signe de l'écart "
        "jdac − aa×1 est stable entre sujets, avant toute conclusion."))''')

    md("""## Constats permis par les données

Cette page est descriptive. À remplir seulement d'après ce que montrent les figures :

- **Cohérence globale :** épaisseur, surface et volume cortical bougent-ils dans le même sens sur `run-01` et sur les scans bougés ?
- **Sous-cortical :** quelles structures se déplacent, et le ventricule latéral grossit-il quand la matière grise rétrécit ?
- **Contrôle SubCortGrayVol :** la ressemblance jdac / aa×1 tient-elle sujet par sujet, ou seulement en médiane ?

Aucune conclusion forte n'est écrite automatiquement ici. La surface et le volume cortical n'apparaissent que si `morphometry_long.csv` a été produit avec les mesures `aparc` des cinq conditions.""")

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name":"Python (cortical-motion)",
                                    "language":"python", "name":"python3"}
    nbf.write(nb, OUT)
    print(f"Notebook créé : {OUT}")


if __name__ == "__main__":
    build()
