# Phase 5 — Fidélité morphométrique FreeSurfer

## Analyse morphométrique unifiée (source de référence)

Analyse descriptive côte à côte de l'épaisseur, de la surface, du volume cortical et des volumes sous-corticaux, toutes rapportées à la **distance signée à `brut/run-01` du même sujet** (même définition que le boxplot d'épaisseur de la phase 4).

- Source compilée unique, **versionnée dans le dépôt** (utilisable depuis GitHub, sans `derivatives`) : `results/ds004332/phase5_fidelity/morphometry_long.csv` (+ `morphometry_completeness.csv`). Table longue : `subject, run, condition, family, hemi, region, metric, value, unit`, familles `cortical_region` (thickness/surface_area/cortical_gray_volume, depuis `aparc.stats`), `aseg_global` (dont `SubCortGrayVol`) et `aseg_region`. La colonne `source_file` (chemins Narval) a été retirée.
- Produite par `extract_morphometry_stats.py` sur Narval (lit `aseg.stats` + `lh/rh.aparc.stats` des 5 conditions ; chemins par défaut = dossiers `*_rigid` de Narval), puis rapatriée par `rsync`.
- Notebook `explore_morphometrie.ipynb` (généré par `build_notebook_morphometrie.py`), export `explore_morphometrie.html` : figure principale (4 mesures globales), figure sous-corticale (thalamus, hippocampe, putamen, ventricule latéral) et contrôle `SubCortGrayVol` sujet par sujet. Descriptif seulement, sans Agitation ni test.

Pour régénérer et exécuter depuis le dépôt (le CSV suffit, pas besoin de `derivatives`) :

```bash
python build_notebook_morphometrie.py
jupyter nbconvert --to notebook --execute --inplace explore_morphometrie.ipynb
jupyter nbconvert --to html --no-input explore_morphometrie.ipynb
```

## Analyse aseg antérieure (volumes seuls)

Cette section analyse les tables `aseg.stats` séparément. Le notebook `aseg` est volontairement limité à deux questions et reste local (source `derivatives/.../aseg_stats/`).

## Questions principales

1. **Identité sur run-01 :** un traitement ajoute-t-il une erreur régionale sur le scan presque immobile ?
2. **Fidélité sur run-02/03 :** JDAC ou ses variantes réduisent-ils l'erreur davantage que le preprocessing seul ?

La référence opérationnelle est le `brut/run-01` du même sujet. Pour chaque sujet, l'erreur principale est la médiane de l'erreur relative absolue sur les régions anatomiques retenues. Le sujet reste ainsi l'unité statistique.

Agitation, les dizaines de modèles par structure, les classements exploratoires et la figure centrale ont été retirés du notebook principal.

## Fichiers

- `build_notebook_aseg.py` : source reproductible de l'analyse et des cellules du notebook.
- `explore_aseg_rigide.ipynb` : notebook minimal généré; il doit être exécuté sur le PC du labo.
- `inspect_central_structure.py` : contrôle visuel séparé des thalamus et ventricules à partir de `T1.mgz` et `aseg.mgz`.
- `extract_aseg_stats.py` : outil de secours permettant de reconstruire les tables si elles sont perdues.
- `extract_morphometry_stats.py` : extraction unifiée des `aseg.stats` et `lh/rh.aparc.stats` des cinq conditions. Il produit les épaisseurs, surfaces, volumes corticaux et volumes sous-corticaux avec leurs unités et leur provenance, sans effectuer d'analyse statistique.

L'export HTML doit être recréé seulement après exécution complète de cette version.

## Tables sur le PC du labo

```text
~/Documents/derivatives/ds004332/aseg_stats/
├── aseg_brut.csv
├── aseg_preproc.csv
├── aseg_jdac.csv
├── aseg_antiartonly.csv
└── aseg_nodenoise.csv
```

Les tables sont tabulées malgré leur extension `.csv` et restent hors Git.

## Sorties principales attendues

- un mini-tableau de complétude;
- une figure d'erreur d'identité sur `run-01`;
- trois contrastes appariés contre `preproc`, avec IC95 et correction de Holm;
- une figure de fidélité régionale sur `run-02` et `run-03`;
- les comparaisons directes `JDAC − preproc` et variantes `− preproc`;
- une conclusion automatique fondée sur ces contrastes.

Une différence positive dans les tableaux de contrastes signifie que la première condition nommée produit davantage d'erreur. Une différence négative signifie qu'elle produit moins d'erreur.

## Exécution au labo

Le dépôt est directement dans `~/Documents/` :

```bash
cd ~/Documents/jdac-motion-correction
git pull --ff-only origin main
~/miniconda3/envs/cortical-motion/bin/jupyter-lab
```

Ouvrir ensuite `pipelines/ds004332/phase5_fidelity/explore_aseg_rigide.ipynb`, exécuter toutes les cellules, vérifier les deux tableaux de contrastes, puis exporter le HTML.
