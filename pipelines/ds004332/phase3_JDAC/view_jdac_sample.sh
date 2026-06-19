#!/bin/bash
# QC visuel JDAC sur le meme echantillon de 8 que le preproc (view_preproc_sample.sh),
# couvrant la plage de mouvement (faible -> tres fort) : sub-01 (3 runs, ancrage M1),
# les 4 plus forts mouvements, et 1 controle median.
#
# Pour chaque image, 3 calques (geste JDAC = preproc -> jdac) :
#   - brut (_T1w.nii)            : reference d'origine
#   - _brain.nii.gz (preproc)    : cerveau N4 + SynthStrip = ENTREE de JDAC
#   - _T1w_jdac.nii.gz (fixed)   : sortie JDAC recalibree (meme grille que preproc)
#
# Questions QC :
#   - forts mouvements : JDAC attenue-t-il le flou / le ringing ?
#   - sub-01_run-01 (immobile) : JDAC sur-lisse-t-il un scan deja propre ?
#
# Astuce : 1 image a la fois dans "Overlay list" (clic sur l'oeil), coupe axiale.
# Usage : bash view_jdac_sample.sh

RAW=~/Documents/raw_datasets/ds004332
PRE=~/Documents/derivatives/ds004332/preproc
JD=~/Documents/derivatives/ds004332/jdac_fixed   # JDAC recalibre (meme grille que preproc)

# id:motion (motion = score d'agitation, pour memoire)
SAMPLE=(
  "sub-01_run-01:0.20"
  "sub-01_run-02:0.26"
  "sub-01_run-03:3.16"
  "sub-11_run-03:3.29"
  "sub-03_run-02:3.25"
  "sub-14_run-03:3.23"
  "sub-07_run-03:3.18"
  "sub-20_run-03:0.71"
)

ARGS=()
for entry in "${SAMPLE[@]}"; do
  id="${entry%%:*}"
  s="${id%_run-*}"; r="${id#*_}"
  raw="$RAW/$s/anat/${s}_acq-mpragepmcoff_rec-wore_${r}_T1w.nii"
  brain="$PRE/$id/${id}_brain.nii.gz"
  jdac="$JD/$id/${id}_T1w_jdac.nii.gz"
  ARGS+=( "$raw" )
  ARGS+=( "$brain" )
  ARGS+=( "$jdac" )
done

fsleyes "${ARGS[@]}"
