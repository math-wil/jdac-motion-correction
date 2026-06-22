#!/bin/bash
# QC visuel du preprocessing RIGIDE (abp_n4 + recalage rigide MNI + mri_synthstrip)
# sur le meme echantillon de 8 que la note de reunion du 19 : sub-01 (3 runs),
# les 4 plus forts mouvements, et 1 controle median.
#
# 3 calques par image :
#   - brut (_T1w.nii)        : image d'origine (espace natif)
#   - _n4.nii.gz (rigide)    : apres N4 + recalage rigide (espace MNI 1mm)
#   - _mask.nii.gz (rigide)  : masque skull-strip, en ROUGE
#
# Le _n4 rigide et son _mask sont dans le meme espace -> ils se superposent
# (c'est le QC du skull-strip : le cortex est-il rogne ? crane garde ?).
# Le brut est en espace natif (reference), il ne se superpose pas au rigide.
#
# Usage : bash view_preproc_rigid_sample.sh

RAW=~/Documents/raw_datasets/ds004332
PRE=~/Documents/derivatives/ds004332/preproc_rigid

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
