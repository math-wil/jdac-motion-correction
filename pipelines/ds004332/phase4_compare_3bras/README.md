# Phase 4 — Comparaison des 3 bras

Compare l'épaisseur corticale entre **brute / preproc / jdac** en fonction du score de mouvement Agitation, pour déterminer si JDAC corrige le mouvement (pente épaisseur ~ mouvement aplatie) ou s'il lisse (offset sans changement de pente).

## Analyses
- **E1** : descriptif par condition (still/nodding/shaking).
- **Par niveau de mouvement** : seuils Agitation 0.3 / 2.6 (faible/modéré/sévère, ancrés sur la distribution) + interaction bras × niveau.
- **E2** : pente intra-sujet (épaisseur ~ Agitation, par sujet), test apparié Wilcoxon.
- **E3** : modèle mixte `épaisseur ~ âge + sexe + Agitation × bras + (1|sujet)`, par région + FDR.

## Fichiers
- `build_notebook.py` : génère les deux notebooks (un par pipeline).
- `explore_3bras_natif.ipynb` / `explore_3bras_rigide.ipynb` : analyse pas à pas, lisible. Lancer dans l'env `cortical-motion` (`jupyter lab`).
- `compare_3bras.py` : script équivalent (option `--pipeline natif|rigide`).

## Entrées
- Épaisseur : `derivatives/ds004332/thickness_{preproc,jdac}[_rigid]_{lh,rh}.csv` ; brut `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv`.
- Agitation : `results/ds004332/agitation/ds004332_agitation_clinica.csv`.

## Résultat
JDAC aplatit la pente épaisseur ~ mouvement (rigide : variation +0.050, p=0.004, 18/68 régions FDR), mais par offset et lissage : il modifie surtout les scans peu bougés (interaction bras × niveau significative) et ne récupère pas les scans très bougés. Cohérent natif/rigide. Détail : `research-notes/02_Experiments/comparaison-3bras.md`.
