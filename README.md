# jdac-motion-correction

Évaluation de l'outil **JDAC** (correction d'artefacts de mouvement en IRM cérébrale structurelle) sur le dataset **ds004332**, via le score de mouvement **Agitation** (Bricout) et l'épaisseur corticale **FreeSurfer**. Labo Neuro-iX (ÉTS), Mathilde Wilfart.

## Question
Le mouvement pendant l'acquisition biaise l'épaisseur corticale mesurée par FreeSurfer. JDAC corrige-t-il ce biais, ou applique-t-il un lissage ? L'épaisseur est mesurée sur plusieurs conditions de traitement des mêmes images, puis comparée en fonction du mouvement.

## Les 5 conditions (pipeline rigide)
- **brut** : image brute → FreeSurfer (référence).
- **preproc** : N4 + recalage rigide MNI + SynthStrip → FreeSurfer (effet du prétraitement seul).
- **jdac** : cerveau prétraité → JDAC complet → FreeSurfer (mouvement « corrigé »).
- **jdac_antiartonly** : JDAC sans débruiteur, anti-artefact appliqué une fois.
- **jdac_nodenoise** : JDAC sans débruiteur, anti-artefact itéré (×4).

Le pipeline **natif** (sans recalage rigide) a servi d'analyse préliminaire ; il est abandonné au profit du **rigide** (épaisseurs comparables).

## Pipeline (par phases)
1. `pipelines/ds004332/phase1_RAW/` : recon-all sur images brutes + GLM épaisseur ~ mouvement.
2. `pipelines/ds004332/phase2_PREPROC/` : prétraitement (N4 + rigide + SynthStrip) + recon-all.
3. `pipelines/ds004332/phase3_JDAC/` : JDAC sur cerveaux prétraités + recon-all.
4. `pipelines/ds004332/phase4_compare_3bras/` : comparaison des 5 conditions (immobiles, pentes épaisseur ~ Agitation, modèles emboîtés M0 vs M1, non-linéarité par strate). Notebook `explore_epaisseur_rigide.ipynb`, script `compare_conditions.py`.

Chaque phase a son propre README. Index : `pipelines/ds004332/README.md` (code) et `results/ds004332/README.md` (résultats).

## État courant (résumé)
- **5 conditions rigides** : recon-all terminé (brut 65/66, preproc 64/66, jdac 64/66, jdac_antiartonly 66/66, jdac_nodenoise 65/66). Comparaison à 5 conditions faite (notebook `explore_epaisseur_rigide.ipynb`).
- **Conclusion** : JDAC complet est le seul à découpler mouvement et épaisseur (le mouvement ne prédit plus l'épaisseur), mais via un lissage (offset sur les immobiles). Les variantes sans débruiteur sur-corrigent et inversent la pente (jdac_nodenoise rend les scans bougés plus épais).

État détaillé et à jour : vault `research-notes` (`STATUS.md`).

## Données (hors dépôt, volumineuses)
- **Brut** : `~/Documents/raw_datasets/ds004332/` (BIDS).
- **Dérivés** (preproc, JDAC, FreeSurfer) : `~/Documents/derivatives/ds004332/`.
- **Calcul** : Narval (compte ctb-sbouix). JDAC et analyses en local (env conda `cortical-motion`).
- **Résultats légers** (CSV, figures) versionnés dans `results/`.

## Résultat clé
Le mouvement réduit l'épaisseur corticale mesurée (brut : pente épaisseur ~ Agitation ≈ −0.07 ; le mouvement prédit l'épaisseur). Après JDAC complet, Agitation ne prédit plus l'épaisseur (modèles emboîtés M0 vs M1), mais JDAC abaisse aussi l'épaisseur des scans immobiles (≈ −6 %) : aplatissement par lissage, pas correction ciblée. Sans le débruiteur, l'anti-artefact sur-corrige proportionnellement au mouvement et inverse la pente (jdac_nodenoise ×4 : scans très bougés plus épais que les immobiles).

---
Anciennes expériences (OASIS-1, ds000115, ds001907, MR-ART, FastSurfer) et anciens scripts (preprocessing FLIRT, SSIM/PSNR) : dépôt archivé **motion-analysis** (rien supprimé là-bas).
