# Phase 1 — RAW (images brutes → FreeSurfer)

Bras de référence : recon-all FreeSurfer sur les images brutes, sans aucun traitement.

## Scripts
- `submit_recon_all_raw.sh` : SLURM, recon-all sur les 66 images brutes (Narval).
- `glm_pipelineB.py` : GLM `épaisseur ~ âge + sexe + Agitation + (1|sujet)`, par région + correction FDR.

## Sorties
- 66 runs, **65 utilisables** (`sub-01_run-03` : échec de reconstruction, mouvement sévère).
- `results/ds004332/phase1_RAW/` : `ThickAvg_phase1_complete.csv`, `glm_pipeline_b_freesurfer_results.csv`.

## Résultat
Agitation β ≈ −0.066 mm/mm, ~34/67 régions FDR : le mouvement réduit l'épaisseur corticale mesurée (effet plus marqué en temporal/limbique).
