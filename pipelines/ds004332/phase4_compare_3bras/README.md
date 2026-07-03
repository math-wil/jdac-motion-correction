# Phase 4 — Comparaison de l'épaisseur entre conditions de traitement

Compare l'épaisseur corticale (FreeSurfer) entre **cinq conditions de traitement**, en pipeline rigide, en fonction du score de mouvement Agitation, pour déterminer si JDAC corrige le mouvement (pente épaisseur ~ Agitation aplatie, sans baisse sur les immobiles) ou s'il lisse (offset sur les immobiles).

Vocabulaire : `condition` = les cinq traitements ; `consigne` = still / nodding / shaking (instruction donnée au sujet).

## Les 5 conditions (toutes en rigide)
- `brut` : image brute → FreeSurfer.
- `preproc` : N4 + recalage rigide MNI + SynthStrip → FreeSurfer.
- `jdac` : JDAC complet (débruiteur + anti-artefact, boucle itérative).
- `jdac_antiartonly` : réseau anti-artefact appliqué une seule fois.
- `jdac_nodenoise` : boucle de JDAC sans débruiteur, anti-artefact jusqu'à quatre fois.

## Structure de l'analyse (notebook, épaisseurs en mm, chaque sortie expliquée)
- **A. Immobiles** : épaisseur des conditions sur les scans immobiles (run-01), écart au brut en mm ; isole le lissage / offset (rien à corriger sur un immobile).
- **B. Immobile vs bougé, par sujet** : écart `épaisseur(run-01) − épaisseur(run-03)` par sujet ; figure par sujet (les 3 runs) + décompte améliorés / sur-corrigés / inchangés par condition.
- **C. M0 vs M1** : par condition, `épaisseur ~ âge + sexe` (M0) contre `+ Agitation` (M1) ; le mouvement prédit-il encore l'épaisseur (p du LRT, coefficient en mm/point).
- **D. Évaluation image (protocole JDAC)** : SSIM sur l'image et sur les cartes de gradient du scan bougé contre le scan propre (référence propre = preproc run-01 ; référence intra-condition pour enlever le biais d'intensité). Distingue correction et lissage indépendamment de l'épaisseur.

## Fichiers
- `build_notebook.py` : génère le notebook.
- `explore_epaisseur_rigide.ipynb` : analyse pas à pas, lisible. Lancer dans l'env `cortical-motion` (`jupyter lab`).
- `compute_image_metrics.py` : calcule les métriques image (SSIM image + gradient vs scan propre), écrit `results/ds004332/phase4_compare_3bras/image_metrics.csv` (chargé par la section D). À exécuter avant le notebook.
- `compare_conditions.py` : script CLI (à réaligner sur la structure A/B/C/D du notebook).

## Entrées
- Épaisseur : brut `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv` ; preproc/jdac `derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv` ; variantes `derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/…_{lh,rh}.csv`.
- Agitation : `results/ds004332/agitation/ds004332_agitation_clinica.csv`.

## Résultat
Les deux variantes sans débruiteur **sur-corrigent** : l'écart immobile − bougé s'inverse chez presque tous les sujets (nodenoise 19/20), le coefficient d'Agitation devient positif (nodenoise +0.092 mm/point, p=7e-7), et côté image leurs contours ne se rapprochent pas du scan propre malgré leur netteté. **jdac complet** découple le mieux mouvement et épaisseur (M0 vs M1 : coef −0.020, p=0.11 ; contours plus proches du propre en référence intra), mais avec un offset de −0.16 mm sur les immobiles (lissage). Une grande part du rapprochement immobile/bougé vient déjà du **preprocessing seul** (16 sujets/19). Détail : `research-notes` (`STATUS.md`, `06_Daily-logs/2026-07-03`).
