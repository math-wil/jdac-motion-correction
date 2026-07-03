# Phase 4 — Comparaison de l'épaisseur entre conditions de traitement

Compare l'épaisseur corticale (FreeSurfer) entre **cinq conditions de traitement**, en pipeline rigide, en fonction du score de mouvement Agitation, pour déterminer si JDAC corrige le mouvement (pente épaisseur ~ Agitation aplatie, sans baisse sur les immobiles) ou s'il lisse (offset sur les immobiles).

Vocabulaire : `condition` = les cinq traitements ; `consigne` = still / nodding / shaking (instruction donnée au sujet).

## Les 5 conditions (toutes en rigide)
- `brut` : image brute → FreeSurfer.
- `preproc` : N4 + recalage rigide MNI + SynthStrip → FreeSurfer.
- `jdac` : JDAC complet (débruiteur + anti-artefact, boucle itérative).
- `jdac_antiartonly` : réseau anti-artefact appliqué une seule fois.
- `jdac_nodenoise` : boucle de JDAC sans débruiteur, anti-artefact jusqu'à quatre fois.

## Structure de l'analyse (plan Sylvain, réunion 2026-07-02)
1. **Étape 1 — immobiles** : épaisseur des 5 conditions sur les scans immobiles ; isole le lissage / offset (rien à corriger sur un immobile).
2. **Étape 2 — pentes** épaisseur ~ Agitation par condition (globale + intra-sujet, Wilcoxon vs brut).
3. **Étape 3 — M0 vs M1** (méthode de Charles) : `épaisseur ~ âge + sexe` contre `+ Agitation`, par condition ; le score améliore-t-il le modèle (LRT, ΔAIC, coefficient).
4. **Étape 3b — non-linéarité par strate** : formes linéaire / quadratique (Agitation²) / splines / niveau catégoriel, comparées par ΔAIC.

Analyses descriptives conservées : par consigne, stratification par niveau de mouvement, interaction condition × niveau, Wilcoxon par niveau.

## Fichiers
- `build_notebook.py` : génère le notebook.
- `explore_epaisseur_rigide.ipynb` : analyse pas à pas, lisible. Lancer dans l'env `cortical-motion` (`jupyter lab`).
- `compare_conditions.py` : script équivalent (mêmes modèles), écrit les tables dans `results/ds004332/phase4_compare_3bras/`.

## Entrées
- Épaisseur : brut `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv` ; preproc/jdac `derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv` ; variantes `derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/…_{lh,rh}.csv`.
- Agitation : `results/ds004332/agitation/ds004332_agitation_clinica.csv`.

## Résultat
JDAC complet est la seule condition où le mouvement cesse de prédire l'épaisseur (M0 vs M1 : Agitation n'améliore plus le modèle, pente ~ nulle), mais au prix d'un offset négatif sur les scans immobiles (≈ −6 %) : aplatissement par lissage, pas correction ciblée. Les variantes sans débruiteur **sur-corrigent** : la pente s'inverse (positive), l'offset sur les immobiles est plus fort, et `jdac_nodenoise` (×4) rend les scans très bougés plus épais que les immobiles. Retirer le débruiteur et itérer ne récupère pas le cortex, cela introduit un biais inverse proportionnel au mouvement. Détail : `research-notes` (`STATUS.md`, réunion 2026-07-02).
