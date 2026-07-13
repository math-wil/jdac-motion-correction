# Phase 4 — Comparaison de l'épaisseur entre conditions de traitement

Ce dossier porte un ancien nom (`phase4_compare_3bras`), mais l'analyse active compare maintenant **cinq conditions** en pipeline rigide :

- `brut`
- `preproc`
- `jdac`
- `jdac_antiartonly`
- `jdac_nodenoise`

Objectif : déterminer si JDAC corrige réellement le biais de mouvement sur l'épaisseur corticale FreeSurfer, ou s'il réduit surtout l'association mouvement-épaisseur par lissage / offset.

Vocabulaire :

- `condition` = traitement appliqué avant FreeSurfer ;
- `consigne` = instruction d'acquisition (`still`, `nodding`, `shaking`) ;
- `Agitation` = score continu de mouvement utilisé comme covariable.

## Notebook de référence

Le dernier résultat cohérent est le notebook exécuté :

```text
explore_epaisseur_rigide.ipynb
```

Il contient cinq blocs :

| Bloc | Question | Sortie actuelle |
|---|---|---|
| A | Que fait chaque condition sur un scan immobile ? | `a_immobiles_offset.csv` |
| B | L'écart immobile - bougé se réduit-il par sujet ? | `b_ecart_immobile_bouge.csv` |
| C | Agitation prédit-il encore l'épaisseur ? | `c_m0_vs_m1.csv` |
| D | Les images / contours se rapprochent-ils du scan propre ? | `d_image_metrics_summary.csv`, `image_metrics.csv` |
| E | L'épaisseur régionale du scan bougé se rapproche-t-elle de la vérité ? | `e_recovery_summary.csv`, `recovery_metrics.csv` |

Les anciens fichiers `e1_par_condition.csv`, `e2_pentes_sujet.csv`, `e3_interaction_regions.csv` et `compare_3bras.png` correspondaient à l'ancienne analyse 3 conditions. Ils sont obsolètes dans l'analyse active à 5 conditions.

## Entrées

Les fichiers lourds ne sont pas versionnés.

Épaisseur corticale :

- brut : `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv` ;
- preproc / jdac : `~/Documents/derivatives/ds004332/thickness_{preproc,jdac}_rigid_{lh,rh}.csv` ;
- variantes : `~/Documents/derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/..._{lh,rh}.csv`.

Covariables :

- Agitation : `results/ds004332/agitation/ds004332_agitation_clinica.csv` ;
- âge / sexe : `~/Documents/raw_datasets/ds004332/participants.tsv`.

Images pour les métriques D :

- `~/Documents/derivatives/ds004332/preproc_rigid/`
- `~/Documents/derivatives/ds004332/jdac_rigid/`
- `~/Documents/derivatives/ds004332/jdac_rigid_antiartonly/`
- `~/Documents/derivatives/ds004332/jdac_rigid_nodenoise/`

## Scripts

| Script | Rôle |
|---|---|
| `build_notebook.py` | Génère le notebook d'analyse. Contient la section non-linéaire C-bis (quadratique, splines, par strate) ; régénérer et exécuter le notebook pour en produire les résultats. |
| `compare_conditions.py` | Version CLI pour les tableaux A/C et analyses par strates. Nécessite les CSV d'épaisseur dérivés locaux. |
| `compute_image_metrics.py` | Calcule les métriques image du bloc D et écrit `image_metrics.csv`. Nécessite les NIfTI dérivés locaux. |
| `compute_recovery.py` | Calcule la récupération régionale du bloc E et écrit `recovery_metrics.csv`. Nécessite les CSV d'épaisseur dérivés locaux. |

Dans cette session Codex, les dossiers `~/Documents/derivatives/ds004332` et `~/Documents/raw_datasets/ds004332` ne sont pas présents ; les scripts n'ont donc pas été réexécutés. Les valeurs ci-dessous proviennent du notebook déjà exécuté et des CSV détaillés versionnés.

## Résultats actuels

### A. Scans immobiles : offset / lissage

Sur `run-01`, il n'y a pas de mouvement à corriger. Toute différence d'épaisseur vient donc du traitement.

| Condition | Épaisseur immobile (mm) | Écart au brut (mm) | Écart au brut (%) |
|---|---:|---:|---:|
| `brut` | 2.595 | +0.000 | +0.0 |
| `preproc` | 2.518 | -0.077 | -3.0 |
| `jdac` | 2.435 | -0.160 | -6.2 |
| `jdac_antiartonly` | 2.418 | -0.177 | -6.8 |
| `jdac_nodenoise` | 2.378 | -0.216 | -8.3 |

Lecture : tous les traitements amincissent un cerveau immobile. `jdac`, `antiartonly` et surtout `nodenoise` introduisent un offset important ; ce n'est pas une correction du mouvement.

### B. Immobile vs shaking, par sujet

Décompte des sujets chez qui l'écart `immobile - shaking` se réduit sans s'inverser :

| Condition | n sujets | Améliorés | Sur-corrigés | Inchangés / pires |
|---|---:|---:|---:|---:|
| `preproc` | 19 | 16 | 1 | 2 |
| `jdac` | 19 | 15 | 3 | 1 |
| `jdac_antiartonly` | 21 | 7 | 14 | 0 |
| `jdac_nodenoise` | 20 | 1 | 19 | 0 |

Lecture : `preproc` et `jdac` rapprochent souvent les scans, mais les variantes sans débruiteur inversent le signal chez la majorité des sujets.

### C. Modèles M0 vs M1

M0 = `épaisseur ~ âge + sexe`.  
M1 = `M0 + Agitation`.

| Condition | n acquisitions | Coef Agitation (mm/point) | p M1 vs M0 | Sens |
|---|---:|---:|---:|---|
| `brut` | 65 | -0.0662 | 6.6e-05 | amincit |
| `preproc` | 64 | -0.0308 | 0.028 | amincit |
| `jdac` | 64 | -0.0203 | 0.11 | amincit faiblement |
| `jdac_antiartonly` | 66 | +0.0253 | 0.081 | épaissit |
| `jdac_nodenoise` | 65 | +0.0923 | 7.1e-07 | épaissit |

Lecture : `jdac` complet découple le mieux Agitation et épaisseur (`p = 0.11`), mais il garde un offset sur les immobiles. `nodenoise` inverse fortement le lien : plus le scan bouge, plus l'épaisseur mesurée augmente.

### D. Métriques image

Au fort mouvement (`shaking`), SSIM image vs scan propre :

| Condition | SSIM image | SSIM gradient | SSIM gradient intra-condition |
|---|---:|---:|---:|
| `preproc` | 0.540 | 0.489 | 0.489 |
| `jdac` | 0.561 | 0.487 | 0.529 |
| `jdac_antiartonly` | 0.552 | 0.493 | 0.489 |
| `jdac_nodenoise` | 0.555 | 0.479 | 0.471 |

Lecture : `jdac` améliore légèrement la similarité image et améliore les contours en référence intra-condition. Les variantes sans débruiteur ne gagnent pas en fidélité des contours malgré leur netteté visuelle.

### E. Récupération vers la vraie épaisseur régionale

Vérité = épaisseur régionale du scan immobile brut du même sujet.

Au fort mouvement (`shaking`) :

| Condition | Mouvement restant (mm) | Erreur à la vérité (mm) | Offset sur le propre (mm) |
|---|---:|---:|---:|
| `brut` | 0.315 | 0.315 | 0.000 |
| `preproc` | 0.271 | 0.311 | 0.087 |
| `jdac` | 0.317 | 0.373 | 0.232 |
| `jdac_antiartonly` | 0.341 | 0.314 | 0.193 |
| `jdac_nodenoise` | 0.415 | 0.336 | 0.255 |

Lecture : aucune condition ne descend sous l'erreur du brut à la vérité. Net d'offset, seul `preproc` réduit un peu le motif de mouvement ; `nodenoise` l'empire.

## Conclusion

Le résultat cohérent actuel est :

- `jdac` complet réduit le lien Agitation-épaisseur et donne le meilleur découplage statistique ;
- mais `jdac` complet amincit aussi les scans immobiles, donc une part importante de l'effet est un offset / lissage ;
- les variantes sans débruiteur ne résolvent pas le problème : elles sur-corrigent et ne rapprochent pas l'épaisseur régionale de la vérité ;
- aucune condition ne ramène actuellement l'épaisseur d'un scan bougé à sa vraie valeur régionale.

La Phase 4 est désormais gelée : le notebook exécuté et les CSV A–E restent les sources de vérité. Les scripts historiques ne sont pas supprimés tant que la reproduction complète n'a pas été refaite avec les dérivés externes.

La suite est implémentée dans `../phase5_fidelity/`. Elle remplace l'ancien libellé « ICC test-retest » par **ICC(A,1) d'accord absolu sous mouvement induit**, ajoute la médiane de l'erreur régionale comme critère principal, le contrôle positif PMC, le QC FreeSurfer aveugle et un pilote BME-X verrouillé. Aucun réentraînement JDAC n'est lancé avant ces gates.
