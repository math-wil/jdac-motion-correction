#!/bin/bash
# QC visuel du preprocessing (abp_n4 + mri_synthstrip) sur un echantillon de 8 images
# couvrant la plage de mouvement (faible -> tres fort) : sub-01 (3 runs, ancrage M1),
# les 4 plus forts mouvements (skull-strip a risque), et 1 controle median.
#
# Pour chaque image, 3 calques sont charges :
#   - brut (_T1w.nii)        : reference d'origine
#   - _n4.nii.gz             : apres correction de biais (intensites homogenes ?)
#   - _mask.nii.gz           : masque skull-strip, en ROUGE semi-transparent
#                              (sur le _n4 : le cortex est-il rogne ? crane/dure-mere residuels ?)
#
# Astuce : dans le panneau "Overlay list", n'affiche qu'une image a la fois (clic sur l'oeil).
# Pour juger le skull-strip : affiche _n4 + son _mask rouge par-dessus.
#
# Usage : bash view_preproc_sample.sh

RAW=~/Documents/raw_datasets/ds004332
PRE=~/Documents/derivatives/ds004332/preproc

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
  n4="$PRE/$id/${id}_n4.nii.gz"
  mask="$PRE/$id/${id}_mask.nii.gz"
  ARGS+=( "$raw" )
  ARGS+=( "$n4" )
  ARGS+=( "$mask" -cm red -a 30 )
done

fsleyes "${ARGS[@]}"
