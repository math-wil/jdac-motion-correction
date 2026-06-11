# jdac-motion-correction

Évaluation de l'outil **JDAC** (correction d'artefacts de mouvement en IRM cérébrale structurelle) sur le dataset **ds004332**, via le score **Agitation** et l'épaisseur corticale **FreeSurfer**. Labo Neuro-iX (ÉTS), Mathilde Wilfart.

## Idée
Le mouvement pendant l'acquisition biaise les mesures FreeSurfer (épaisseur corticale). On teste si JDAC corrige ce biais, en comparant deux pipelines sur les mêmes images.

## Les 3 phases
- **Phase 1 — FreeSurfer sur images brutes** (`pipelines/ds004332/phase1_freesurfer_raw/`) : recon-all sur le brut, puis GLM Pipeline B (`épaisseur ~ âge + sexe + Agitation + (1|sujet)`). **[terminé]**
- **Phase 2 — Preprocessing puis FreeSurfer** (`phase2_preproc_freesurfer/`) : Clinica t1-linear + SynthStrip, puis recon-all `-noskullstrip` (script 2 passes, `brainmask = T1.mgz`). Vérifie que le preprocessing seul ne change pas les mesures (comparaison Phase 1 vs Phase 2). **[en cours]**
- **Phase 3 — JDAC puis FreeSurfer** (`phase3_jdac/`) : Clinica + SynthStrip → normalisation [0,1] → JDAC → dénormalisation → recon-all. GLM Pipeline A, comparé à B : si JDAC corrige le biais, les betas âge/sexe doivent être identiques entre A et B. **[à venir]**

L'état courant détaillé est dans le vault de notes `research-notes` (STATUS.md).

## Structure
```
pipelines/ds004332/
  phase1_freesurfer_raw/      # recon-all brut + GLM Pipeline B
  phase2_preproc_freesurfer/  # Clinica+SynthStrip + recon-all + comparaison P1/P2
  phase3_jdac/                # JDAC + GLM Pipeline A (à venir)
  agitation/                  # scoring Agitation (brut + Clinica)
  qc/                         # SSIM/PSNR, visualisation
  shared/                     # scripts génériques + infra Narval
                              #   aggregate_freesurfer.py : origine repo cortical-motion (C. Bricout)
results/ds004332/             # résultats LÉGERS versionnés (CSV, figures), par phase
```

## Où sont les données (hors dépôt, trop volumineuses)
- **Brut** : `~/Documents/Datasets/ds004332/` (BIDS).
- **Dérivés / sorties** (preprocessing, JDAC, FreeSurfer) : `~/Documents/Results/ds004332/`.
- **Résultats finaux validés** : serveur **hippocampus**, `/project/hippocampus/common/mathilde/ds004332/`.
- **Calcul** : Narval (compte ctb-sbouix). Scripts SLURM dans les dossiers de phase.

## Environnement
Env conda `cortical-motion` (pandas, statsmodels, nibabel). FreeSurfer 8.0.0, Clinica, JDAC sur Narval.

## Résultat clé (Phase 1)
GLM Pipeline B : le mouvement réduit l'épaisseur corticale mesurée (Agitation β ≈ −0.066 mm/mm ; 34/67 régions significatives après correction FDR ; plus marqué en temporal/limbique).

---
Anciennes expériences (OASIS-1, ds000115, ds001907, MR-ART, FastSurfer) : voir le dépôt archivé **motion-analysis** (rien n'a été supprimé).
