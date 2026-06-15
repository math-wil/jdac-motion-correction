# jdac-motion-correction

Évaluation de l'outil **JDAC** (correction d'artefacts de mouvement en IRM cérébrale structurelle) sur le dataset **ds004332**, via le score **Agitation** et l'épaisseur corticale **FreeSurfer**. Labo Neuro-iX (ÉTS), Mathilde Wilfart.

## Idée
Le mouvement pendant l'acquisition biaise l'épaisseur corticale mesurée par FreeSurfer. On teste si JDAC corrige ce biais, en mesurant l'épaisseur à 3 stades de traitement (« bras ») des mêmes images et en comparant.

## Les 3 bras (= les 3 dossiers de phase)
- **RAW** (`phase1_RAW/`) : images brutes → FreeSurfer. Baseline. GLM Pipeline B. **[terminé]**
- **PREPROC** (`phase2_PREPROC/`) : preprocessing + SynthStrip → FreeSurfer. Effet du preprocessing seul. **[sub-01 fait]**
- **JDAC** (`phase3_JDAC/`) : preprocessing + SynthStrip → JDAC → dénormalisation → FreeSurfer. Mouvement corrigé. GLM Pipeline A vs B. **[en cours, M1]**

État courant détaillé : vault `research-notes` (`STATUS.md`, `roadmap-pipeline-AB.md`).

## Contenu du dépôt (chaque fichier)
```
pipelines/ds004332/
  phase1_RAW/
    submit_recon_all_raw.sh        # SLURM : recon-all sur les 66 images brutes
    glm_pipelineB.py               # GLM : epaisseur ~ age + sexe + Agitation + (1|sujet)
  phase2_PREPROC/
    preproc_clinica_synthstrip.sh  # SLURM : Clinica t1-linear (affine) + SynthStrip [à passer en rigide, M2]
    recon_all_preproc.sh           # SLURM : recon-all 2 passes sur cerveau prétraité
    compare_raw_vs_preproc.py      # compare l'épaisseur RAW vs PREPROC (par région/condition)
  phase3_JDAC/
    run_jdac.py                    # LE script JDAC (reproduit JDAC_Application.ipynb des auteurs)
    m1_sub01_subjects.csv          # entrées du test M1 (sub-01, 3 runs)
    README.md                      # plan du bras JDAC + entrées/sorties vérifiées
  agitation/
    run_agitation_on_clinica.py    # Agitation sur images Clinica -> CSV covariable du GLM
  utils/
    transfer_to_narval.sh          # copie des données vers le cluster Narval
    narval_README.md               # guide d'exécution sur Narval
results/ds004332/                  # résultats LÉGERS versionnés (CSV, figures)
  phase1/   ThickAvg_phase1*.csv, glm_pipeline_b_freesurfer_results.csv
  phase2/   compare_phase1_phase2_sub-01.csv, preproc_avant_apres_sub-01_run-02.png
  agitation/ ds004332_agitation_clinica.csv
```

## Où sont les données (hors dépôt, trop volumineuses)
- **Brut** : `~/Documents/raw_datasets/ds004332/` (BIDS).
- **Dérivés** (preprocessing, JDAC, FreeSurfer) : `~/Documents/derivatives/ds004332/`.
- **Résultats finaux validés** : hippocampus, `/project/hippocampus/common/mathilde/ds004332/`.
- **Calcul** : Narval (compte ctb-sbouix). Scripts SLURM dans les dossiers de phase ; JDAC tourne en local (env `cortical-motion`).

## Résultat clé (bras RAW)
GLM Pipeline B : le mouvement réduit l'épaisseur corticale mesurée (Agitation β ≈ −0.066 mm/mm ; 34/67 régions FDR ; plus marqué en temporal/limbique).

---
Anciennes expériences (OASIS-1, ds000115, ds001907, MR-ART, FastSurfer) et anciens scripts (preprocessing FLIRT, SSIM/PSNR) : dépôt archivé **motion-analysis** (rien supprimé là-bas).
