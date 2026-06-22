#!/bin/bash
# QC visuel du preprocessing RIGIDE (N4 + recalage rigide MNI + mri_synthstrip),
# sur les 8 sujets de la note du 19 : sub-01 (3 runs), 4 plus forts mouvements, 1 median.
#
# 3 calques par image :
#   - brut (_T1w.nii)        : image d'origine, espace NATIF (reference seulement,
#                              ne se superpose PAS au rigide -> normal)
#   - _rigid.nii.gz          : image apres N4 + recalage rigide (espace MNI)
#   - _mask.nii.gz           : masque skull-strip, en ROUGE
#
# IMPORTANT : _rigid et _mask sont dans le MEME espace -> le masque se superpose
# correctement a l'image rigide. C'est la-dessus qu'on juge le skull strip :
# affiche _rigid puis _mask (rouge) par-dessus. Le cortex est-il bien pris ?
# Reste-t-il du crane ou de la dure-mere ?
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
  rigid="$PRE/$id/${id}_rigid.nii.gz"
  mask="$PRE/$id/${id}_mask.nii.gz"
  ARGS+=( "$raw" )
  ARGS+=( "$rigid" )
  ARGS+=( "$mask" -cm red -a 30 )
done

fsleyes "${ARGS[@]}"
