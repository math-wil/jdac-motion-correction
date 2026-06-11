#!/bin/bash
#SBATCH --job-name=fs_fix
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-sbouix
#SBATCH --array=0-24
#SBATCH --output=/home/mathw/logs/fix_%A_%a.out
#SBATCH --error=/home/mathw/logs/fix_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca

mkdir -p /home/mathw/logs
module load StdEnv/2023 freesurfer/8.0.0-1

SUBJECTS_DIR=~/projects/def-sbouix/mathw/freesurfer_ds004332
SUBJECTS=(sub-02_run-02 sub-02_run-03 sub-03_run-03 sub-04_run-02 sub-05_run-01 sub-06_run-03 sub-07_run-01 sub-08_run-01 sub-08_run-02 sub-08_run-03 sub-09_run-03 sub-11_run-01 sub-11_run-02 sub-13_run-01 sub-13_run-02 sub-14_run-02 sub-16_run-03 sub-17_run-01 sub-17_run-02 sub-18_run-02 sub-18_run-03 sub-19_run-02 sub-20_run-01 sub-20_run-03 sub-21_run-01)

SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "Fix $SUBJ — $(date)"

rm -f "$SUBJECTS_DIR/$SUBJ/scripts/recon-all.done"

recon-all -s "$SUBJ" -sd "$SUBJECTS_DIR" -autorecon2 -autorecon3 -openmp 8

echo "Fin $SUBJ — $(date)"
