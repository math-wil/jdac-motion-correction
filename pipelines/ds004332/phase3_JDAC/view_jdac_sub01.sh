#!/bin/bash
# QC visuel JDAC sur sub-01 (run-01 propre, run-03 bouge), les 3 roles du pipeline :
#   brut    : image d'origine (raw_datasets)
#   preproc : cerveau N4 + SynthStrip = ENTREE de JDAC
#   jdac    : sortie de JDAC (corrige, deja skull-strippe, intensites [0,1])
#
# Geste de JDAC = preproc -> jdac. Questions QC :
#   - run-03 : JDAC attenue-t-il le flou/ringing de mouvement ?
#   - run-01 : JDAC sur-lisse-t-il un scan deja propre ?
#
# Affiche une image a la fois (panneau "Overlay list", clic sur l'oeil) en coupe axiale.
# Usage : bash view_jdac_sub01.sh

RAW=~/Documents/raw_datasets/ds004332
PRE=~/Documents/derivatives/ds004332/preproc_natif
JD=~/Documents/derivatives/ds004332/jdac_natif   # images JDAC recalibrees (meme grille que preproc)

ARGS=()
for r in run-01 run-03; do
  id=sub-01_$r
  ARGS+=( "$RAW/sub-01/anat/sub-01_acq-mpragepmcoff_rec-wore_${r}_T1w.nii" )
  ARGS+=( "$PRE/$id/${id}_brain.nii.gz" )
  ARGS+=( "$JD/$id/${id}_T1w_jdac.nii.gz" )
done

fsleyes "${ARGS[@]}"
