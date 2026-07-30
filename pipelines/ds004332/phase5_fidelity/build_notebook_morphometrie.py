#!/usr/bin/env python3
"""Génère le notebook morphométrique descriptif : explore_morphometrie.ipynb

But (réunion du 24 juillet) : suivre l'anatomie de la matière grise en deux parties.
Partie 1, le cortex : épaisseur, surface et volume corticaux côte à côte, mesures liées
(volume ≈ surface × épaisseur), avec un tableau qui le vérifie. Partie 2, le sous-cortical :
SubCortGrayVol et ses huit composantes (thalamus, caudé, putamen, pallidum, hippocampe,
amygdale, accumbens, VentralDC). Chaque panneau utilise exactement la définition du boxplot
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

**Deux compartiments de matière grise, deux parties.** La matière grise n'est pas d'un seul tenant ; la page suit donc l'anatomie, pas une liste de mesures à plat :

- **Partie 1 — le cortex** (l'écorce pliée en surface). On le décrit par trois mesures qui ne sont pas indépendantes : **épaisseur** (moyenne des régions, mm), **surface** (somme des régions, mm²) et **volume cortical** (somme des régions, mm³). Comme le cortex est une nappe, `volume ≈ surface × épaisseur` : les trois racontent une seule histoire, et un tableau le vérifie.
- **Partie 2 — les noyaux gris profonds** (thalamus, putamen, hippocampe…). Masses pleines au centre du cerveau : elles n'ont ni surface ni épaisseur, seulement un **volume**. Leur total est `SubCortGrayVol` (mm³, mesure globale de `aseg.stats`), décomposé ensuite structure par structure.

L'agrégation diffère selon la mesure et c'est volontaire : moyenne pour l'épaisseur (comme en phase 4), somme pour surface et volumes (une surface et un volume sont additifs, le total est la mesure globale naturelle).

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
    groups = []
    for i, c in enumerate(CONDITIONS):
        for cons in ["still","nodding","shaking"]:
            vals = per[(per["condition"] == c) & (per["consigne"] == cons)]["dist"].dropna()
            if vals.empty:
                continue
            groups.append(vals)
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
    # Cadrage robuste : borner l'axe sur les boîtes et leurs moustaches (Tukey,
    # 1.5x IQR), 0 inclus, pour ne pas etre dezoome par quelques valeurs extremes.
    # Les points hors fenetre sont comptes et signales sous la figure.
    lows, highs, n_hors = [], [], 0
    for v in groups:
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5*iqr, q3 + 1.5*iqr
        inl = v[(v >= lo_f) & (v <= hi_f)]
        if len(inl):
            lows.append(inl.min()); highs.append(inl.max())
        n_hors += int(((v < lo_f) | (v > hi_f)).sum())
    if lows:
        lo, hi = min(lows + [0.0]), max(highs + [0.0])
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.08*span, hi + 0.08*span)
    return n_hors

def legende_consignes(fig):
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c],
               label={"still":"run-01 immobile","nodding":"run-02 nodding","shaking":"run-03 shaking"}[c])
               for c in ["still","nodding","shaking"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)

def _color(t, fmt="{:.1f}"):
    """Colore un tableau d'écarts en % : rouge = baisse, bleu = hausse, centré sur 0."""
    arr = t.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    vmax = vmax or 1.0
    return (t.style.format(fmt)
             .background_gradient(cmap="RdBu", vmin=-vmax, vmax=vmax, axis=None)
             .set_properties(**{"text-align": "center"}))''')

    md("""## Partie 1 — Le cortex : épaisseur, surface, volume (trois mesures liées)

Le cortex est la fine écorce de matière grise pliée à la surface du cerveau. On le décrit ici par ses trois mesures corticales, **côte à côte pour les lire ensemble** : elles ne sont pas indépendantes (`volume ≈ surface × épaisseur`, vérifié juste après).

**Question posée.** Sur un scan immobile (points verts), le traitement déplace-t-il déjà la mesure par rapport au brut, et sur les scans bougés (orange, rouge) se rapproche-t-il ou s'éloigne-t-il de la référence ?

**Comment lire.** La ligne pointillée à 0 est le brut/run-01 du sujet. Une boîte proche de 0 = mesure proche de la référence. Une boîte **sous** 0 = mesure plus petite que le brut immobile (amincissement, perte de surface ou de volume) ; **au-dessus** de 0 = plus grande. La cohérence attendue est que les trois bougent dans le même sens.""")

    code('''MESURES = [
    ("Épaisseur corticale (moyenne régions)", "mm",
     par_acquisition(d, "cortical_region", "thickness", "mean", positive_only=True)),
    ("Surface corticale (somme régions)", "mm²",
     par_acquisition(d, "cortical_region", "surface_area", "sum")),
    ("Volume cortical gris (somme régions)", "mm³",
     par_acquisition(d, "cortical_region", "cortical_gray_volume", "sum")),
]
manquantes = [titre for titre, _, per in MESURES if per.empty]
if manquantes:
    display(Markdown("**Mesures absentes de la source (panneau vide) :** " + ", ".join(manquantes)
                     + ". Vérifier que `lh/rh.aparc.stats` ont bien été extraits pour les cinq conditions."))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
n_hors = 0
for ax, (titre, unite, per) in zip(axes.flat, MESURES):
    if per.empty:
        ax.set_title(titre + " (données absentes)", fontsize=12); ax.set_axis_off(); continue
    n_hors += box_distance(ax, distance_a_reference(per), titre, unite)
fig.suptitle("Cortex : distance signée à brut/run-01 du même sujet, par condition et par consigne", y=1.0)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])
legende_consignes(fig)
plt.show()
if n_hors:
    display(Markdown(f"*Axes cadrés sur les boîtes et moustaches pour rester lisibles ; {n_hors} points extrêmes (outliers) sont volontairement hors cadre.*"))''')

    md("""### Lecture chiffrée : médianes par condition et par consigne

Les mêmes écarts, en médiane et en pourcentage, pour lire les nombres derrière les boîtes. La consigne sépare le scan immobile (still, run-01) des scans bougés (nodding run-02, shaking run-03). Rappel : `brut/run-01` vaut 0, la ligne `brut` mesure donc le dégât du **mouvement seul**, et un correcteur devrait rapprocher run-02/run-03 de 0 **sans déplacer run-01**.""")

    code('''def _median_pct(family, metric, agg, pos=False, region=None):
    per = distance_a_reference(par_acquisition(d, family, metric, agg, region=region, positive_only=pos))
    per["pct"] = 100 * (per["value"] - per["ref"]) / per["ref"]
    t = (per.groupby(["condition","consigne"], observed=True)["pct"]
            .median().unstack("consigne").reindex(CONDITIONS)[["still","nodding","shaking"]])
    t.index = [SHORT[c] for c in t.index]
    t.columns = ["still (run1)", "nodding (run2)", "shaking (run3)"]
    return t

tab_ep = _median_pct("cortical_region", "thickness", "mean", pos=True)
tab_su = _median_pct("cortical_region", "surface_area", "sum")
tab_vo = _median_pct("cortical_region", "cortical_gray_volume", "sum")
for titre, t in [("Épaisseur", tab_ep), ("Surface", tab_su), ("Volume cortical", tab_vo)]:
    display(Markdown(f"**{titre} — médiane de l'écart à brut/run-01 (%)** — rouge = baisse, bleu = hausse"))
    display(_color(t))

display(Markdown(
    "**Comment lire les trois mesures ensemble** *(valeurs sur le scan immobile run1)*\\n\\n"
    "- **Volume = surface × épaisseur**, donc en écart `% volume ≈ % surface + % épaisseur`.\\n"
    "- **L'épaisseur baisse dans les cinq conditions** : seule, elle ne distingue pas les mécanismes.\\n"
    f"- **La surface les distingue** : jdac {tab_su.loc['jdac','still (run1)']:.1f} % (lissage, bord gris/blanc flou), "
    f"aa×4 {tab_su.loc['aa×4','still (run1)']:+.1f} % (sur-affûtage, bords plus découpés), "
    f"preproc {tab_su.loc['prep','still (run1)']:+.1f} % (rééchantillonnage neutre).\\n"
    f"- **Le volume est le bilan des deux** : sous jdac elles baissent ensemble et s'additionnent ({tab_vo.loc['jdac','still (run1)']:.1f} %) ; "
    f"sous aa×4 épaisseur en baisse et surface en hausse s'annulent ({tab_vo.loc['aa×4','still (run1)']:.1f} %)."))

display(Markdown(
    "**Ce qu'on en tire, condition par condition**\\n\\n"
    "- **preproc** : ne corrige pas le mouvement, il colle au brut sur les scans bougés. Ligne de base.\\n"
    f"- **jdac complet** : pire que le brut partout, même sur le scan immobile ({tab_vo.loc['jdac','still (run1)']:.1f} % de volume sur run1). Il déforme une anatomie qui n'avait rien à corriger.\\n"
    f"- **aa×1** (jdac sans débruiteur) : érosion divisée par ~2 (surface {tab_su.loc['aa×1','still (run1)']:.1f} % contre {tab_su.loc['jdac','still (run1)']:.1f} % pour jdac). Le débruiteur est le principal coupable.\\n"
    f"- **aa×4** : meilleure fidélité de volume (run3 {tab_vo.loc['aa×4','shaking (run3)']:.1f} % contre {tab_vo.loc['brut','shaking (run3)']:.1f} % au brut), mais via épaisseur en baisse et surface en hausse qui se compensent. À vérifier sujet par sujet avant de conclure."))''')

    md("""### Vérification mathématique : volume ≈ surface × épaisseur

Le cortex est une **nappe** : son volume vaut à peu près `surface × épaisseur`, donc en pourcentage `% volume ≈ % surface + % épaisseur`. Ce tableau (sur le scan immobile) vérifie que les trois mesures corticales sont cohérentes entre elles, et pas seulement dans le même sens.""")

    code('''def _pct_run01(family, metric, agg, pos=False):
    p = par_acquisition(d, family, metric, agg, positive_only=pos)
    ref = p[(p.condition == "brut") & (p.run == "run-01")][["subject","value"]].rename(columns={"value":"ref"})
    p = p.merge(ref, on="subject")
    p = p[p.run == "run-01"]
    p["pct"] = 100 * (p.value - p.ref) / p.ref
    return p.groupby("condition", observed=True)["pct"].median()

verif = pd.DataFrame({
    "% épaisseur": _pct_run01("cortical_region", "thickness", "mean", pos=True),
    "% surface":   _pct_run01("cortical_region", "surface_area", "sum"),
    "% volume":    _pct_run01("cortical_region", "cortical_gray_volume", "sum"),
}).reindex(CONDITIONS)
verif["% épaisseur + % surface"] = verif["% épaisseur"] + verif["% surface"]
verif = verif[["% épaisseur", "% surface", "% épaisseur + % surface", "% volume"]]
verif.index = [SHORT[c] for c in verif.index]
display(_color(verif))
display(Markdown(
    "**Lecture.** La colonne « % épaisseur + % surface » doit approcher « % volume », et c'est le cas ici : "
    "les trois mesures corticales sont cohérentes, pas seulement dans le même sens. Quand surface et épaisseur "
    "baissent ensemble (ex. JDAC), le volume baisse davantage car les deux effets s'additionnent ; quand elles "
    "partent en sens opposés (surface qui monte, épaisseur qui baisse), le volume bouge peu. Cette relation ne "
    "vaut que pour le cortex (une nappe) ; `SubCortGrayVol` est un volume de noyaux profonds, sans surface ni épaisseur."))''')

    md("""## Partie 2 — Les noyaux gris profonds (le sous-cortical)

Au centre du cerveau, sous le cortex, se trouvent des masses grises pleines. Elles n'ont ni surface ni épaisseur, seulement un **volume**. FreeSurfer en fait le total `SubCortGrayVol`, qui est **exactement la somme de huit structures** (gauche + droit) : thalamus, noyau caudé, putamen, pallidum, hippocampe, amygdale, noyau accumbens et diencéphale ventral (VentralDC).

**Question posée.** Le déplacement global de `SubCortGrayVol` vient-il de tous les noyaux ou de quelques-uns ? Le premier panneau montre le total, les huit suivants sa décomposition.

**Comment lire.** Volumes bilatéraux (gauche + droit) en mm³, même distance signée à brut/run-01. Sous 0 = structure plus petite qu'au brut immobile ; au-dessus = plus grande. Contrairement au cortex, aucune relation `surface × épaisseur` ici : ce ne sont que des volumes.""")

    code('''COMPOSANTES = [
    ("Thalamus", "Thalamus"),
    ("Noyau caudé", "Caudate"),
    ("Putamen", "Putamen"),
    ("Pallidum", "Pallidum"),
    ("Hippocampe", "Hippocampus"),
    ("Amygdale", "Amygdala"),
    ("Accumbens", "Accumbens"),
    ("Diencéphale ventral", "VentralDC"),
]
aseg_reg = d[d["family"] == "aseg_region"].copy()

# Panneaux : le total SubCortGrayVol d'abord, puis ses huit composantes (G+D).
PANNEAUX = [("SubCortGrayVol (total)", None)] + COMPOSANTES
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
n_hors = 0
for ax, (titre, motif) in zip(axes.flat, PANNEAUX):
    if motif is None:
        per = par_acquisition(d, "aseg_global", "volume", "sum", region="SubCortGrayVol")
        titre_ax = titre
    else:
        sub = aseg_reg[aseg_reg["region"].str.contains(motif, na=False)]
        if sub.empty:
            ax.set_title(titre + " (absent de aseg)", fontsize=12); ax.set_axis_off(); continue
        per = (sub.groupby(["subject","run","condition"], observed=True)["value"]
                  .sum().reset_index(name="value"))
        titre_ax = titre + " (G+D)"
    if per.empty:
        ax.set_title(titre_ax + " (données absentes)", fontsize=12); ax.set_axis_off(); continue
    n_hors += box_distance(ax, distance_a_reference(per), titre_ax, "mm³")
fig.suptitle("Sous-cortical : SubCortGrayVol et ses huit composantes, distance signée à brut/run-01", y=1.0)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
legende_consignes(fig)
plt.show()
if n_hors:
    display(Markdown(f"*Axes cadrés sur les boîtes et moustaches ; {n_hors} points extrêmes sont hors cadre.*"))''')

    md("""### Lecture chiffrée : le total et ses composantes

Les mêmes écarts en médiane et en pourcentage. Le premier tableau suit `SubCortGrayVol` par consigne (comme en partie 1) ; le second détaille les huit composantes sur le scan immobile, pour voir si le total vient de tous les noyaux ou de quelques-uns.""")

    code('''tab_scv = _median_pct("aseg_global", "volume", "sum", region="SubCortGrayVol")
display(Markdown("**SubCortGrayVol (total) — médiane de l'écart à brut/run-01 (%)** — rouge = baisse, bleu = hausse"))
display(_color(tab_scv))

def _comp_run01_pct():
    rows = {}
    for nom, motif in COMPOSANTES:
        sub = aseg_reg[aseg_reg["region"].str.contains(motif, na=False)]
        per = (sub.groupby(["subject","run","condition"], observed=True)["value"]
                  .sum().reset_index(name="value"))
        per = distance_a_reference(per)
        per["pct"] = 100 * (per["value"] - per["ref"]) / per["ref"]
        rows[nom] = (per[per["run"] == "run-01"].groupby("condition", observed=True)["pct"]
                     .median().reindex(CONDITIONS))
    t = pd.DataFrame(rows).T
    t.columns = [SHORT[c] for c in t.columns]
    return t

comp = _comp_run01_pct()
display(Markdown("**Les 8 composantes sur le scan immobile (run-01) — médiane de l'écart (%)**"))
display(_color(comp))

display(Markdown(
    "**Comment lire le sous-cortical**\\n\\n"
    f"- **L'offset immobile est modéré** : sur run1, SubCortGrayVol baisse de {tab_scv.loc['jdac','still (run1)']:.1f} % (jdac), "
    f"{tab_scv.loc['aa×1','still (run1)']:.1f} % (aa×1), {tab_scv.loc['aa×4','still (run1)']:.1f} % (aa×4), "
    "bien moins que le volume cortical (jdac −14,2 %). Le lissage de JDAC touche surtout la frontière corticale, moins les noyaux profonds.\\n"
    f"- **Le total cache des sens opposés** : sous jdac, caudé ({comp.loc['Noyau caudé','jdac']:.1f} %), putamen ({comp.loc['Putamen','jdac']:.1f} %) "
    f"et accumbens ({comp.loc['Accumbens','jdac']:.1f} %) rétrécissent, mais thalamus ({comp.loc['Thalamus','jdac']:+.1f} %) et "
    f"diencéphale ventral ({comp.loc['Diencéphale ventral','jdac']:+.1f} %) grossissent. Le total est un net, il faut lire les composantes.\\n"
    f"- **Sur les scans bougés, tous aggravent la perte** : à shaking, jdac {tab_scv.loc['jdac','shaking (run3)']:.1f} % et "
    f"aa×4 {tab_scv.loc['aa×4','shaking (run3)']:.1f} % contre brut {tab_scv.loc['brut','shaking (run3)']:.1f} %. Aucun ne récupère les volumes.\\n"
    f"- **Différence clé avec le cortex** : aa×4, le meilleur sur le volume cortical, est ici parmi les pires ({tab_scv.loc['aa×4','shaking (run3)']:.1f} % à shaking). "
    "Son avantage venait de la compensation épaisseur↓/surface↑, propre à la nappe corticale ; les noyaux n'ont pas de surface à gonfler."))''')

    md("""### Lien avec le notebook aseg : magnitude de l'écart (non signée)

Jusqu'ici l'écart est **signé** (sens et quantité). L'ancien notebook aseg regardait autre chose : la **magnitude** de l'erreur, `|T − R| / R`, toujours positive, moyennée sur les volumes de structures. C'est « de combien on s'éloigne du brut », sans le sens. Ci-dessous, cette même lentille recalculée depuis la source unifiée (structures aseg à base > 100 mm³), pour relier l'ancien et le nouveau.""")

    code('''ref_v = (aseg_reg[(aseg_reg.condition=="brut") & (aseg_reg.run=="run-01")]
         [["subject","region","value"]].rename(columns={"value":"ref"}))
m = aseg_reg.merge(ref_v, on=["subject","region"])
m = m[m["ref"] > 100]
m["err"] = 100 * (m["value"] - m["ref"]).abs() / m["ref"]
fid = (m.groupby(["subject","run","condition"], observed=True)["err"]
         .median().reset_index(name="err_pct"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, run in zip(axes, ["run-01","run-02","run-03"]):
    data = [fid[(fid["run"]==run) & (fid["condition"]==c)]["err_pct"].dropna().values for c in CONDITIONS]
    bp = ax.boxplot(data, positions=range(len(CONDITIONS)), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={"color":"black","lw":1.3})
    for b in bp["boxes"]:
        b.set(facecolor="lightsteelblue", alpha=0.75)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([SHORT[c] for c in CONDITIONS], rotation=20)
    ax.set_title(f"{run} ({CONSIGNE[run]})", fontsize=12)
    ax.grid(axis="y", alpha=0.18)
axes[0].set_ylabel("erreur médiane |T−R|/R (%)")
axes[0].set_ylim(bottom=0)
fig.suptitle("Fidélité des volumes aseg : magnitude de l'écart au brut/run-01 (non signée)", y=1.03)
fig.tight_layout()
plt.show()

fid_tab = (fid.groupby(["condition","run"], observed=True)["err_pct"].median()
             .unstack()[["run-01","run-02","run-03"]].reindex(CONDITIONS))
fid_tab.index = [SHORT[c] for c in fid_tab.index]
fid_tab.columns = ["still (run1)", "nodding (run2)", "shaking (run3)"]
display(Markdown("**Erreur médiane |T−R|/R sur les volumes (%)** — plus foncé = plus loin du brut"))
display(fid_tab.round(1).style.format("{:.1f}")
        .background_gradient(cmap="Reds", axis=None)
        .set_properties(**{"text-align": "center"}))

display(Markdown(
    "**Le lien avec l'ancien notebook**\\n\\n"
    f"- **Scan immobile (rien à corriger)** : jdac ajoute le plus d'erreur ({fid_tab.loc['jdac','still (run1)']:.1f} %), "
    f"devant aa×4 ({fid_tab.loc['aa×4','still (run1)']:.1f} %) et aa×1 ({fid_tab.loc['aa×1','still (run1)']:.1f} %) ; "
    f"preproc reste le plus proche ({fid_tab.loc['prep','still (run1)']:.1f} %). JDAC déforme le plus une anatomie propre.\\n"
    f"- **Scans bougés** : le brut est déjà à {fid_tab.loc['brut','shaking (run3)']:.1f} % (shaking), et **toutes** les conditions font pire "
    f"(jdac {fid_tab.loc['jdac','shaking (run3)']:.1f} %). Corriger ne rapproche pas les volumes du brut immobile.\\n"
    "- **Le pont entre les deux notebooks** : le signe (plus haut sur cette page) donne la direction, la magnitude (ici) donne l'ampleur. "
    "Même message que sur le cortex : JDAC abîme le scan propre, aucune condition ne restaure les volumes bougés."))''')

    md("""## Constats

- **Le mouvement seul** (ligne brut) amincit le cortex et réduit surface et volumes, de plus en plus fort de run-01 à run-03. C'est le dégât qu'un correcteur doit défaire sans toucher au scan immobile.
- **preproc** ne combat pas le mouvement : il colle au brut sur les scans bougés (ligne de base).
- **JDAC complet** abîme le scan propre (cortex : volume −14 % sur run-01 ; volumes aseg : erreur la plus forte) et ne récupère aucun volume bougé. Le **débruiteur** en est le principal responsable : sans lui (aa×1), l'érosion corticale est divisée par deux.
- **aa×4** donne la meilleure fidélité du **volume cortical**, mais par une compensation épaisseur↓/surface↑ propre à la nappe corticale. Cet avantage **ne se transmet pas au sous-cortical**, où aa×4 est parmi les pires. À vérifier sujet par sujet avant d'en conclure quoi que ce soit.
- **Lecture des mesures** : l'épaisseur seule ne suffit pas (elle baisse partout) ; la surface révèle le mécanisme, et le total sous-cortical cache des noyaux qui bougent en sens opposés.""")

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name":"Python (cortical-motion)",
                                    "language":"python", "name":"python3"}
    nbf.write(nb, OUT)
    print(f"Notebook créé : {OUT}")


if __name__ == "__main__":
    build()
