#!/usr/bin/env bash
# Reconstitution vérifiée du pilote MRIQC sub-19 exécuté le 14/09/2026.
# Le pilote original a été lancé en commandes directes. Ce script documente la
# procédure et les options confirmées ; il ne prétend pas être le script original.
set -euo pipefail

DERIV="${DERIV:-$HOME/Documents/derivatives/ds004332}"
RAW="${RAW:-$HOME/Documents/raw_datasets/ds004332}"
SIF="${SIF:-/project/hippocampus/common/containers/mriqc_23.1.0.sif}"
PILOT="${PILOT:-$DERIV/mriqc_pilot_20260914}"
SUBJ="sub-19"

for path in "$DERIV" "$RAW" "$SIF"; do
  if [[ ! -e "$path" ]]; then
    echo "Entrée absente : $path" >&2
    exit 1
  fi
done

link_or_copy() {
  local source="$1" destination="$2"
  mkdir -p "$(dirname "$destination")"
  if ! ln -f "$source" "$destination" 2>/dev/null; then
    cp --reflink=auto -f "$source" "$destination"
  fi
}

stage_three_runs() {
  local bids_root="$1" acq="$2" extension="$3"
  shift 3
  local run source destination
  for run in 01 02 03; do
    source="$1"; shift
    destination="$bids_root/$SUBJ/anat/${SUBJ}_acq-${acq}_run-${run}_T1w.${extension}"
    link_or_copy "$source" "$destination"
  done
}

IN_BO="$PILOT/input_brainonly"
IN_RAW="$PILOT/input_raw"

stage_three_runs "$IN_BO" preproc nii.gz \
  "$DERIV/preproc_rigid/${SUBJ}_run-01/${SUBJ}_run-01_brain.nii.gz" \
  "$DERIV/preproc_rigid/${SUBJ}_run-02/${SUBJ}_run-02_brain.nii.gz" \
  "$DERIV/preproc_rigid/${SUBJ}_run-03/${SUBJ}_run-03_brain.nii.gz"
stage_three_runs "$IN_BO" jdac nii.gz \
  "$DERIV/jdac_rigid/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac.nii.gz" \
  "$DERIV/jdac_rigid/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac.nii.gz" \
  "$DERIV/jdac_rigid/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac.nii.gz"
stage_three_runs "$IN_BO" aa1 nii.gz \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac_antiartonly.nii.gz" \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac_antiartonly.nii.gz" \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac_antiartonly.nii.gz"
stage_three_runs "$IN_BO" aa4 nii.gz \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac_nodenoise.nii.gz" \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac_nodenoise.nii.gz" \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac_nodenoise.nii.gz"

stage_three_runs "$IN_RAW" raw nii \
  "$RAW/$SUBJ/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-01_T1w.nii" \
  "$RAW/$SUBJ/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-02_T1w.nii" \
  "$RAW/$SUBJ/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-03_T1w.nii"

# Les sidecars des bruts existaient dans l'entrée réellement utilisée.
for run in 01 02 03; do
  source="$RAW/$SUBJ/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-${run}_T1w.json"
  if [[ -f "$source" ]]; then
    link_or_copy "$source" "$IN_RAW/$SUBJ/anat/${SUBJ}_acq-raw_run-${run}_T1w.json"
  fi
done

for bids_root in "$IN_BO" "$IN_RAW"; do
  cat >"$bids_root/dataset_description.json" <<'JSON'
{
  "Name": "ds004332 MRIQC technical pilot",
  "BIDSVersion": "1.10.0",
  "DatasetType": "raw"
}
JSON
done

mkdir -p "$PILOT/logs"
run_mriqc() {
  local input="$1" output="$2" work="$3" log="$4"
  mkdir -p "$output" "$work"
  local command=(
    apptainer run --cleanenv
    -B "$input:/data:ro"
    -B "$output:/out"
    -B "$work:/work"
    "$SIF"
    /data /out participant
    --participant-label 19
    -m T1w
    --no-sub
    --nprocs 4
    --omp-nthreads 2
    --mem 24
    -w /work
  )
  printf 'Commande :'; printf ' %q' "${command[@]}"; printf '\n'
  "${command[@]}" 2>&1 | tee "$log"
}

run_mriqc "$IN_BO" "$PILOT/output_brainonly" "$PILOT/work_brainonly" "$PILOT/logs/brainonly.log"
run_mriqc "$IN_RAW" "$PILOT/output_raw" "$PILOT/work_raw" "$PILOT/logs/raw.log"

echo "MRIQC terminé. Les JSON et HTML sont dans $PILOT/output_* ."
