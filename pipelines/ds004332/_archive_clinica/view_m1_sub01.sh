#!/bin/bash
# Ouvre dans FSLeyes les 9 images du test M1 (sub-01 : 3 runs x brut / pretraite / jdac),
# telles quelles (espaces d'origine, AUCUN recalage). Tu affiches une image a la fois
# (panneau "Overlay list", clic sur l'oeil) et tu fais tes captures.
#
# Usage : bash view_m1_sub01.sh

RAW=~/Documents/raw_datasets/ds004332/sub-01/anat
PRE=/project/hippocampus/common/mathilde/ds004332/phase2_preproc
JD=~/Documents/derivatives/ds004332/jdac_m1_test

fsleyes \
  "$RAW/sub-01_acq-mpragepmcoff_rec-wore_run-01_T1w.nii" \
  "$PRE/sub-01_run-01/sub-01_run-01_clinica_synthstrip_brain.nii.gz" \
  "$JD/sub-01_run-01/sub-01_run-01_T1w_jdac.nii.gz" \
  "$RAW/sub-01_acq-mpragepmcoff_rec-wore_run-02_T1w.nii" \
  "$PRE/sub-01_run-02/sub-01_run-02_clinica_synthstrip_brain.nii.gz" \
  "$JD/sub-01_run-02/sub-01_run-02_T1w_jdac.nii.gz" \
  "$RAW/sub-01_acq-mpragepmcoff_rec-wore_run-03_T1w.nii" \
  "$PRE/sub-01_run-03/sub-01_run-03_clinica_synthstrip_brain.nii.gz" \
  "$JD/sub-01_run-03/sub-01_run-03_T1w_jdac.nii.gz"
