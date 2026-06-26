# Phase 2 — PREPROC (prétraitement → FreeSurfer)

Bras prétraitement : N4 + recalage rigide MNI + SynthStrip, puis recon-all. Isole l'effet du prétraitement seul (sans JDAC). Le recalage **rigide** (6 DOF) préserve l'échelle, donc des épaisseurs comparables entre bras (l'affine 12 DOF de Clinica les faussait, +0.14 mm).

## Scripts
- `preproc.py` : N4 (ANTsPy) + recalage rigide MNI (grille élargie `MNI_PAD=48`) + SynthStrip. Produit `_n4`, `_rigid`, `_brain`, `_mask`.
- `recon_all_preproc_rigid.sh` : SLURM, recon-all 2 passes sur le cerveau rigide (avec `-cw256`, le FOV élargi dépasse 256 mm).
- `recon_all_preproc.sh` : version natif (sans recalage rigide), analyse préliminaire.
- `view_preproc_rigid_sample.sh`, `view_preproc_sample.sh` : QC visuel FSLeyes.
- `fig_preproc_steps.py`, `montage_preproc_mask.py`, `assemble_preproc_screens.py` : figures.

## Sorties
- Cerveaux prétraités : `derivatives/ds004332/preproc_{natif,rigid}/` (hors dépôt).
- recon-all rigide : **64/66** (manquent `sub-10_run-01` (normalisation) et `sub-11_run-03` (topologie)).
- Figures : `results/ds004332/phase2_PREPROC/`.
