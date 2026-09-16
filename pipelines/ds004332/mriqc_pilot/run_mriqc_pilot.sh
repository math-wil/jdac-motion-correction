#!/usr/bin/env bash
# =============================================================================
# Pilote MRIQC sub-19 — procédure documentée et reproductible (14/09/2026)
#
# IMPORTANT : ce script REPRODUIT ce que l'assistant Codex a fait en commandes
# directes le 14/09. Il n'existait aucun script d'origine ; celui-ci est écrit
# après coup à partir des traces réelles :
#   - les images sources retrouvées par inode (liens durs) ;
#   - les chemins /data /out participant lus dans logs/brainonly.log ;
#   - les commandes internes visibles dans work_*/mriqc_wf/**/command.txt.
# Les options exactes de la ligne apptainer externe (nprocs, etc.) ne sont pas
# sauvegardées ; on garde ici les essentielles, vérifiées par le comportement.
#
# But du pilote : vérifier que MRIQC tourne et QUALIFIER les IQM sur nos deux
# types d'images. Un seul sujet -> aucune conclusion scientifique.
# =============================================================================
set -euo pipefail

DERIV="$HOME/Documents/derivatives/ds004332"
RAW="$HOME/Documents/raw_datasets/ds004332"
SIF="/project/hippocampus/common/containers/mriqc_23.1.0.sif"   # container, sur Hippocampus
PILOT="$DERIV/mriqc_pilot_20260914"
SUBJ="sub-19"

# --- 1) Ranger les images en BIDS (liens durs = pas de duplication) ----------
# MRIQC n'accepte qu'un dossier BIDS. On encode la condition dans "acq-" et la
# consigne (still/nodding/shaking) dans "run-01/02/03".
stage() {  # stage <dossier_bids> <acq> <fichier_source_run01> <..run02> <..run03>
  local d="$1/$SUBJ/anat"; mkdir -p "$d"
  ln -f "$3" "$d/${SUBJ}_acq-$2_run-01_T1w.${6}"
  ln -f "$4" "$d/${SUBJ}_acq-$2_run-02_T1w.${6}"
  ln -f "$5" "$d/${SUBJ}_acq-$2_run-03_T1w.${6}"
}

IN_BO="$PILOT/input_brainonly"; IN_RAW="$PILOT/input_raw"
# Sorties brain-only (12 images) : même grille 278x314x278, 1 mm, LAS
stage "$IN_BO" preproc \
  "$DERIV/preproc_rigid/${SUBJ}_run-01/${SUBJ}_run-01_brain.nii.gz" \
  "$DERIV/preproc_rigid/${SUBJ}_run-02/${SUBJ}_run-02_brain.nii.gz" \
  "$DERIV/preproc_rigid/${SUBJ}_run-03/${SUBJ}_run-03_brain.nii.gz" nii.gz
stage "$IN_BO" jdac \
  "$DERIV/jdac_rigid/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac.nii.gz" \
  "$DERIV/jdac_rigid/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac.nii.gz" \
  "$DERIV/jdac_rigid/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac.nii.gz" nii.gz
stage "$IN_BO" aa1 \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac_antiartonly.nii.gz" \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac_antiartonly.nii.gz" \
  "$DERIV/jdac_rigid_antiartonly/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac_antiartonly.nii.gz" nii.gz
stage "$IN_BO" aa4 \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-01/${SUBJ}_run-01_T1w_jdac_nodenoise.nii.gz" \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-02/${SUBJ}_run-02_T1w_jdac_nodenoise.nii.gz" \
  "$DERIV/jdac_rigid_nodenoise/${SUBJ}_run-03/${SUBJ}_run-03_T1w_jdac_nodenoise.nii.gz" nii.gz
# Entrées brut full-head (3 images) : autre grille 192x256x256, ~0.9 mm, RAS
stage "$IN_RAW" raw \
  "$RAW/${SUBJ}/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-01_T1w.nii" \
  "$RAW/${SUBJ}/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-02_T1w.nii" \
  "$RAW/${SUBJ}/anat/${SUBJ}_acq-mpragepmcoff_rec-wore_run-03_T1w.nii" nii

# dataset_description.json minimal (obligatoire pour que MRIQC lise le dossier)
for d in "$IN_BO" "$IN_RAW"; do
  printf '{\n  "Name": "ds004332 MRIQC pilot",\n  "BIDSVersion": "1.10.0",\n  "DatasetType": "raw"\n}\n' > "$d/dataset_description.json"
done

# --- 2) Lancer MRIQC, brain-only et brut SÉPARÉMENT (grilles différentes) -----
# apptainer "monte" les dossiers locaux dans le container : entrée -> /data,
# sortie -> /out, work -> /work. "participant" = une analyse par image.
# "--no-sub" = ne PAS envoyer les IQM au serveur public MRIQC.
run_mriqc() {  # run_mriqc <input_bids> <output> <work> <log>
  mkdir -p "$2" "$3"
  apptainer run --cleanenv \
    -B "$1":/data:ro -B "$2":/out -B "$3":/work \
    "$SIF" /data /out participant --no-sub -w /work \
    2>&1 | tee "$4"
}
run_mriqc "$IN_BO"  "$PILOT/output_brainonly" "$PILOT/work_brainonly" "$PILOT/logs/brainonly.log"
run_mriqc "$IN_RAW" "$PILOT/output_raw"       "$PILOT/work_raw"       "$PILOT/logs/raw.log"

# --- 3) Extraire et qualifier les IQM ----------------------------------------
python3 "$PILOT/extract_iqm.py"

# Chaque image produit : output_*/sub-19/anat/<image>.json (les 68 IQM)
#                    et : output_*/<image>.html (rapport visuel à ouvrir).
echo "Terminé. Ouvre un .html pour voir, ou iqm_table.csv pour les chiffres."
