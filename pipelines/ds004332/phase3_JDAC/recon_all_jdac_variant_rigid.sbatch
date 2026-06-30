#!/bin/bash
#SBATCH --job-name=recon_jdac_variant
#SBATCH --account=ctb-sbouix
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-65
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# recon-all sur une VARIANTE de JDAC sans debruitage (pipeline rigide).
# Identique au bras JDAC normal (2 passes, -noskullstrip, -cw256), mais la branche
# est choisie par la variable BRANCH passee a la soumission :
#   sbatch --export=ALL,BRANCH=nodenoise   recon_all_jdac_variant_rigid.sbatch
#   sbatch --export=ALL,BRANCH=antiartonly recon_all_jdac_variant_rigid.sbatch
#
# Rangement (clair, une branche = un dossier d'entree + un SUBJECTS_DIR) :
#   entree   : ~/projects/ctb-sbouix/mathw/jdac_rigid_<BRANCH>_brains/<id>/<id>_T1w_jdac_<BRANCH>.nii.gz
#   sorties  : ~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_<BRANCH>_ds004332/<id>/

set -uo pipefail

: "${BRANCH:?ERREUR : passer --export=ALL,BRANCH=nodenoise (ou antiartonly)}"
case "$BRANCH" in nodenoise|antiartonly) ;; *) echo "BRANCH invalide : $BRANCH"; exit 1;; esac

module load StdEnv/2023
module load freesurfer/8.0.0-1

export FS_LICENSE=$HOME/.licenses/freesurfer.lic
export SUBJECTS_DIR=$HOME/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_${BRANCH}_ds004332
mkdir -p "$SUBJECTS_DIR"

OMP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ -n "${FREESURFER_HOME:-}" ] || { echo "ERREUR : FREESURFER_HOME indefini"; exit 1; }

# Cerveaux de la variante (structure : <id>/<id>_T1w_jdac_<BRANCH>.nii.gz)
INPUT_ROOT=$HOME/projects/ctb-sbouix/mathw/jdac_rigid_${BRANCH}_brains

mapfile -t BRAINS < <(ls "$INPUT_ROOT"/*/*_T1w_jdac_${BRANCH}.nii.gz | sort)
INPUT="${BRAINS[$SLURM_ARRAY_TASK_ID]}"
SUBJECT_ID="$(basename "$INPUT")"; SUBJECT_ID="${SUBJECT_ID%_T1w_jdac_${BRANCH}.nii.gz}"   # ex : sub-01_run-01
SUBJ_DIR="$SUBJECTS_DIR/$SUBJECT_ID"
M="$SUBJ_DIR/mri"

echo "DEBUG BRANCH=$BRANCH  SUBJECT_ID=$SUBJECT_ID  INPUT=$INPUT"
[ -f "$INPUT" ] || { echo "ERREUR : entree introuvable : $INPUT"; exit 1; }
[ -e "$SUBJECTS_DIR/fsaverage" ] || ln -s "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/fsaverage"

if [ -f "$SUBJ_DIR/scripts/recon-all.done" ]; then echo "SKIP : $SUBJECT_ID deja fait."; exit 0; fi
rm -f "$SUBJ_DIR/scripts/IsRunning."* 2>/dev/null || true

# Passe 1 : produire T1.mgz (arret attendu a seg2cc en FS8 -> "|| true")
if [ ! -f "$M/T1.mgz" ]; then
    recon-all -s "$SUBJECT_ID" -i "$INPUT" -autorecon1 -cw256 -noskullstrip -no-isrunning \
        -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR" || true
fi
[ -f "$M/T1.mgz" ] || { echo "ERREUR : T1.mgz absent apres passe 1."; exit 1; }

# brainmask = T1.mgz
if [ ! -f "$M/brainmask.mgz" ]; then
    cp -f "$M/T1.mgz" "$M/brainmask.auto.mgz"
    cp -f "$M/T1.mgz" "$M/brainmask.mgz"
fi
rm -f "$SUBJ_DIR/scripts/IsRunning."* 2>/dev/null || true

# Passe 2 : reprise complete
set -e
recon-all -s "$SUBJECT_ID" -autorecon1 -autorecon2 -autorecon3 -cw256 -noskullstrip -no-isrunning \
    -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR"
echo "Termine : $SUBJECT_ID"
