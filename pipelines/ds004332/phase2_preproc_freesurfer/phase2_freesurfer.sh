#!/bin/bash
#SBATCH --job-name=recon_phase2
#SBATCH --account=ctb-sbouix
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# FreeSurfer recon-all Phase 2 -- ds004332, cerveau deja skull-strippe par SynthStrip (FS 8.0.0-1)
#
# PROBLEME FS8 : -autorecon1 a ete etendu jusqu'aux segmentations (CC seg / seg2cc),
#   qui exigent norm.mgz, lequel depend de brainmask.mgz. -noskullstrip ne cree pas
#   brainmask.mgz -> seg2cc plante avec "cannot find norm.mgz". Et -autorecon1 meurt
#   en plein milieu, donc fabriquer brainmask APRES (version precedente) ne marche pas.
#
# CORRECTIF (semantique -noskullstrip conservee, identique pour la Phase 3 sur sortie JDAC) :
#   Passe 1 : -autorecon1 -noskullstrip produit T1.mgz puis s'arrete a seg2cc (ATTENDU, tolere).
#   brainmask = T1.mgz : l'entree etant deja strippee, T1.mgz EST notre cerveau SynthStrip
#             conforme dans l'espace 256^3 de recon-all. Pas de re-skull-strip, pas de perte.
#   Passe 2 : reprise -autorecon1 -autorecon2 -autorecon3 : seg2cc repasse (norm.mgz cree
#             grace a brainmask), puis surfaces.

set -uo pipefail   # PAS -e ici : la passe 1 s'arrete volontairement a seg2cc

module load StdEnv/2023
module load freesurfer/8.0.0-1

export FS_LICENSE=$HOME/.licenses/freesurfer.lic
export SUBJECTS_DIR=$HOME/projects/ctb-sbouix/mathw/freesurfer_phase2_ds004332
mkdir -p "$SUBJECTS_DIR"

LOGDIR=$HOME/projects/ctb-sbouix/mathw/logs_phase2
mkdir -p "$LOGDIR"

OMP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ -n "${FREESURFER_HOME:-}" ] || { echo "ERREUR : FREESURFER_HOME indefini apres module load"; exit 1; }

PREP_ROOT=$HOME/projects/ctb-sbouix/mathw/phase2_preproc
SUBJECTS=(sub-01_run-01 sub-01_run-02 sub-01_run-03)
SUBJ_RUN="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
SUBJECT_ID="${SUBJ_RUN}_phase2"
INPUT="$PREP_ROOT/$SUBJ_RUN/${SUBJ_RUN}_clinica_synthstrip_brain.nii.gz"
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

# ---------------------------------------------------------------------------
# Passe 1 : produire T1.mgz. En FS8, -autorecon1 ira jusqu'a seg2cc et s'arretera
#   (norm.mgz manquant tant qu'il n'y a pas de brainmask). C'est ATTENDU -> "|| true".
#   Sautee si T1.mgz existe deja (reprise d'un run precedent).
# ---------------------------------------------------------------------------
if [ ! -f "$M/T1.mgz" ]; then
    echo "Passe 1 (autorecon1 -noskullstrip ; arret attendu a seg2cc) : $SUBJECT_ID"
    recon-all -s "$SUBJECT_ID" -i "$INPUT" -autorecon1 -noskullstrip -no-isrunning \
        -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR" || true
fi
[ -f "$M/T1.mgz" ] || { echo "ERREUR : T1.mgz absent apres la passe 1. Arret."; exit 1; }

# ---------------------------------------------------------------------------
# brainmask = T1.mgz (notre cerveau SynthStrip, deja conforme et brain-only).
# ---------------------------------------------------------------------------
if [ ! -f "$M/brainmask.mgz" ]; then
    echo "Fabrication brainmask.mgz = T1.mgz : $SUBJECT_ID"
    cp -f "$M/T1.mgz" "$M/brainmask.auto.mgz"
    cp -f "$M/T1.mgz" "$M/brainmask.mgz"
fi

rm -f "$SUBJ_DIR/scripts/IsRunning."* 2>/dev/null || true

# ---------------------------------------------------------------------------
# Passe 2 : reprise. seg2cc repasse (brainmask present -> norm.mgz cree), puis
#   autorecon2 + autorecon3 (surfaces). set -e : echec si erreur reelle.
# ---------------------------------------------------------------------------
set -e
echo "Passe 2 (autorecon1 -autorecon2 -autorecon3, reprise) : $SUBJECT_ID"
recon-all -s "$SUBJECT_ID" -autorecon1 -autorecon2 -autorecon3 -noskullstrip -no-isrunning \
    -parallel -openmp "$OMP_THREADS" -sd "$SUBJECTS_DIR"

echo "Termine : $SUBJECT_ID"
