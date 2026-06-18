#!/bin/bash
#SBATCH --job-name=recon_m1_jdac
#SBATCH --account=ctb-sbouix
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Test M1 : recon-all sur sub-01, AVANT vs APRES JDAC (n=1, pipeline affine, preliminaire).
# 6 jobs : 3 "before" (entree normalisee comme JDAC mais SANS jdac) + 3 "jdac" (sortie JDAC).
# before et jdac ne different QUE par le reseau JDAC -> isole l'effet pur de JDAC sur l'epaisseur.
#
# Reprend EXACTEMENT le correctif valide en Phase 2 (recon_all_preproc.sh) :
#   - --mem=64G (evite l'OOM de mri_synthseg observe a 16G)
#   - 2 passes pour FS8 : autorecon1 s'arrete a seg2cc (ATTENDU, "|| true"),
#     puis brainmask = T1.mgz (entree deja skull-strippee), puis autorecon1/2/3.
#   - skip si recon-all.done, nettoyage des locks IsRunning.

set -uo pipefail   # PAS -e ici : la passe 1 s'arrete volontairement a seg2cc

module load StdEnv/2023
module load freesurfer/8.0.0-1

export FS_LICENSE=$HOME/.licenses/freesurfer.lic
export SUBJECTS_DIR=$HOME/projects/ctb-sbouix/mathw/freesurfer_m1_jdac_test
mkdir -p "$SUBJECTS_DIR"

OMP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ -n "${FREESURFER_HOME:-}" ] || { echo "ERREUR : FREESURFER_HOME indefini apres module load"; exit 1; }

# Repertoire des 6 entrees (a transferer depuis le poste local, voir README).
INPUT_ROOT=$HOME/projects/ctb-sbouix/mathw/m1_jdac_test/inputs

# index array -> (id du sujet recon-all, fichier d'entree)
LABELS=(sub-01_run-01_before sub-01_run-02_before sub-01_run-03_before \
        sub-01_run-01_jdac   sub-01_run-02_jdac   sub-01_run-03_jdac)
FILES=(sub-01_run-01_before.nii.gz sub-01_run-02_before.nii.gz sub-01_run-03_before.nii.gz \
       sub-01_run-01_jdac.nii.gz   sub-01_run-02_jdac.nii.gz   sub-01_run-03_jdac.nii.gz)

SUBJECT_ID="${LABELS[$SLURM_ARRAY_TASK_ID]}"
INPUT="$INPUT_ROOT/${FILES[$SLURM_ARRAY_TASK_ID]}"
SUBJ_DIR="$SUBJECTS_DIR/$SUBJECT_ID"
M="$SUBJ_DIR/mri"

echo "DEBUG SUBJECT_ID=$SUBJECT_ID"
echo "DEBUG INPUT=$INPUT"
echo "DEBUG OMP_THREADS=$OMP_THREADS"
recon-all --version 2>/dev/null | head -1 || true

[ -f "$INPUT" ] || { echo "ERREUR : entree introuvable : $INPUT"; exit 1; }

# Lien fsaverage (idempotent)
[ -e "$SUBJECTS_DIR/fsaverage" ] || ln -s "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/fsaverage"

# Deja termine ?
if [ -f "$SUBJ_DIR/scripts/recon-all.done" ]; then
    echo "SKIP : $SUBJECT_ID deja termine."
    exit 0
fi

rm -f "$SUBJ_DIR/scripts/IsRunning."* 2>/dev/null || true

# --- Passe 1 : produire T1.mgz (arret attendu a seg2cc en FS8) -------------
if [ ! -f "$M/T1.mgz" ]; then
    echo "Passe 1 (autorecon1 -noskullstrip ; arret attendu a seg2cc) : $SUBJECT_ID"
    recon-all -s "$SUBJECT_ID" -i "$INPUT" -autorecon1 -noskullstrip -no-isrunning \
        -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR" || true
fi
[ -f "$M/T1.mgz" ] || { echo "ERREUR : T1.mgz absent apres la passe 1. Arret."; exit 1; }

# --- brainmask = T1.mgz (entree deja strippee) -----------------------------
if [ ! -f "$M/brainmask.mgz" ]; then
    echo "Fabrication brainmask.mgz = T1.mgz : $SUBJECT_ID"
    cp -f "$M/T1.mgz" "$M/brainmask.auto.mgz"
    cp -f "$M/T1.mgz" "$M/brainmask.mgz"
fi

rm -f "$SUBJ_DIR/scripts/IsRunning."* 2>/dev/null || true

# --- Passe 2 : reprise complete --------------------------------------------
set -e
echo "Passe 2 (autorecon1 -autorecon2 -autorecon3, reprise) : $SUBJECT_ID"
recon-all -s "$SUBJECT_ID" -autorecon1 -autorecon2 -autorecon3 -noskullstrip -no-isrunning \
    -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR"

echo "Termine : $SUBJECT_ID"
