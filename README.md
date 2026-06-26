# jdac-motion-correction

Évaluation de l'outil **JDAC** (correction d'artefacts de mouvement en IRM cérébrale structurelle) sur le dataset **ds004332**, via le score de mouvement **Agitation** (Bricout) et l'épaisseur corticale **FreeSurfer**. Labo Neuro-iX (ÉTS), Mathilde Wilfart.

## Question
Le mouvement pendant l'acquisition biaise l'épaisseur corticale mesurée par FreeSurfer. JDAC corrige-t-il ce biais, ou applique-t-il un lissage ? L'épaisseur est mesurée à trois stades de traitement (« bras ») des mêmes images, puis comparée en fonction du mouvement.

## Les 3 bras
- **RAW** : image brute → FreeSurfer (référence).
- **PREPROC** : N4 + recalage rigide MNI + SynthStrip → FreeSurfer (effet du prétraitement seul).
- **JDAC** : cerveau prétraité → JDAC → FreeSurfer (mouvement « corrigé »).

Le bras brut est commun. PREPROC et JDAC existent en deux versions : **natif** (sans recalage rigide, analyse préliminaire) et **rigide** (recalage rigide, version courante, pour des épaisseurs comparables).

## Pipeline (par phases)
1. `pipelines/ds004332/phase1_RAW/` : recon-all sur images brutes + GLM épaisseur ~ mouvement.
2. `pipelines/ds004332/phase2_PREPROC/` : prétraitement (N4 + rigide + SynthStrip) + recon-all.
3. `pipelines/ds004332/phase3_JDAC/` : JDAC sur cerveaux prétraités + recon-all.
4. `pipelines/ds004332/phase4_compare_3bras/` : comparaison des 3 bras (descriptif, niveaux de mouvement, pentes intra-sujet, modèle mixte). Notebooks `explore_3bras_{natif,rigide}.ipynb`.

Chaque phase a son propre README. Index : `pipelines/ds004332/README.md` (code) et `results/ds004332/README.md` (résultats).

## État courant (résumé)
- **Pipeline natif** : 3 bras complets, comparaison faite.
- **Pipeline rigide** : recon-all terminé (PREPROC 64/66, JDAC 64/66 ; quelques run-03 « shaking » irréconstructibles), comparaison faite.
- **Conclusion cohérente** : JDAC aplatit la pente épaisseur ~ mouvement, mais surtout via le prétraitement et un lissage (il modifie le plus les scans peu bougés), pas par une correction ciblée.

État détaillé et à jour : vault `research-notes` (`STATUS.md`).

## Données (hors dépôt, volumineuses)
- **Brut** : `~/Documents/raw_datasets/ds004332/` (BIDS).
- **Dérivés** (preproc, JDAC, FreeSurfer) : `~/Documents/derivatives/ds004332/`.
- **Calcul** : Narval (compte ctb-sbouix). JDAC et analyses en local (env conda `cortical-motion`).
- **Résultats légers** (CSV, figures) versionnés dans `results/`.

## Résultat clé
Le mouvement réduit l'épaisseur corticale (RAW : Agitation β ≈ −0.066 mm/mm, ~34/67 régions FDR). JDAC aplatit la pente épaisseur ~ mouvement (rigide : variation +0.050, p=0.004, 18/68 régions), mais par offset et lissage (effet maximal sur les scans peu bougés, pas de récupération des scans très bougés), pas par une correction ciblée du mouvement.

---
Anciennes expériences (OASIS-1, ds000115, ds001907, MR-ART, FastSurfer) et anciens scripts (preprocessing FLIRT, SSIM/PSNR) : dépôt archivé **motion-analysis** (rien supprimé là-bas).
