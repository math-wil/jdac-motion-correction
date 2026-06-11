#!/bin/bash
#SBATCH --job-name=recon_ds004332
#SBATCH --account=ctb-sbouix
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000M
#SBATCH --array=0-65
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca
#SBATCH --mail-type=FAIL,END

# FreeSurfer recon-all — ds004332, 22 sujets × 3 runs = 66 jobs
# Input  : $SCRATCH/ds004332/sub-XX/anat/sub-XX_acq-mpragepmcoff_rec-wore_run-0X_T1w.nii
# Output : $HOME/projects/def-sbouix/mathw/freesurfer_ds004332/  (permanent, hors purge)
#
# Images brutes uniquement (acq-mpragepmcoff rec-wore).
# Ne pas utiliser les sorties JDAC (skull-stripped + normalisées [0,1]).

module load StdEnv/2023
module load freesurfer/8.0.0-1

export FS_LICENSE=$HOME/.licenses/freesurfer.lic
export SCRATCH=/scratch/mathw

echo "DEBUG SCRATCH=$SCRATCH"
echo "DEBUG HOME=$HOME"

export SUBJECTS_DIR=$HOME/projects/def-sbouix/mathw/freesurfer_ds004332
mkdir -p "$SUBJECTS_DIR"

# Résolution de l'index du tableau → (sujet, run)
SUBJECTS=($(seq -w 1 22 | awk '{printf "sub-%02d\n", $1}'))
RUNS=(run-01 run-02 run-03)

SUB_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

SUB="${SUBJECTS[$SUB_IDX]}"
RUN="${RUNS[$RUN_IDX]}"
SUBJECT_ID="${SUB}_${RUN}"

INPUT="/scratch/mathw/ds004332/${SUB}/anat/${SUB}_acq-mpragepmcoff_rec-wore_${RUN}_T1w.nii"

echo "DEBUG INPUT=$INPUT"
ls -la "/scratch/mathw/ds004332/${SUB}/anat/" 2>&1 | head -5

if [ ! -f "$INPUT" ]; then
    echo "ERREUR : fichier introuvable : $INPUT"
    exit 1
fi

# Skip si déjà terminé (recon-all écrit ce fichier à la fin)
if [ -f "$SUBJECTS_DIR/$SUBJECT_ID/scripts/recon-all.done" ]; then
    echo "SKIP : $SUBJECT_ID déjà terminé."
    exit 0
fi

echo "Démarrage : $SUBJECT_ID"
echo "Input     : $INPUT"
echo "SUBJECTS_DIR : $SUBJECTS_DIR"

recon-all \
    -s "$SUBJECT_ID" \
    -i "$INPUT" \
    -all \
    -parallel \
    -openmp "$SLURM_CPUS_PER_TASK" \
    -sd "$SUBJECTS_DIR"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERREUR recon-all (code $EXIT_CODE) : $SUBJECT_ID"
    exit $EXIT_CODE
fi

echo "Terminé : $SUBJECT_ID"
