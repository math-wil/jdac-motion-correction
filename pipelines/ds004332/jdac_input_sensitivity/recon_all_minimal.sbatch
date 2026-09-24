#!/bin/bash
#SBATCH --job-name=jdac_minimal_fs
#SBATCH --account=ctb-sbouix
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-17%3
#SBATCH --output=/project/ctb-sbouix/mathw/jdac_minimal_pilot/logs/%x_%A_%a.out
#SBATCH --error=/project/ctb-sbouix/mathw/jdac_minimal_pilot/logs/%x_%A_%a.err

# Dix-huit reconstructions : neuf cerveaux SynthStrip seuls, puis les neuf
# sorties JDAC correspondantes. Les deux branches ont leurs SUBJECTS_DIR.
# Ne pas copier -cw256 : l'ancienne grille MNI élargie n'est pas utilisée ici.

set -euo pipefail
module load StdEnv/2023
module load freesurfer/8.0.0-1

ROOT=/project/ctb-sbouix/mathw/jdac_minimal_pilot
export FS_LICENSE="$HOME/.licenses/freesurfer.lic"
SIDS=(
    sub-17_run-01 sub-17_run-02 sub-17_run-03
    sub-13_run-01 sub-13_run-02 sub-13_run-03
    sub-14_run-01 sub-14_run-02 sub-14_run-03
)

idx=${SLURM_ARRAY_TASK_ID:?Ce script doit etre lance par sbatch --array}
if (( idx < 0 || idx >= 18 )); then
    echo "Index de job invalide : $idx" >&2
    exit 2
fi
sid=${SIDS[$((idx % 9))]}
if (( idx < 9 )); then
    branch=preproc_minimal
    input="$ROOT/$branch/$sid/${sid}_brain.nii.gz"
else
    branch=jdac_minimal
    input="$ROOT/$branch/$sid/${sid}_T1w_jdac_minimal.nii.gz"
fi

export SUBJECTS_DIR="$ROOT/freesurfer/$branch"
subject_dir="$SUBJECTS_DIR/$sid"
[[ -f "$input" ]] || { echo "Entree absente : $input" >&2; exit 1; }
[[ -f "$FS_LICENSE" ]] || { echo "Licence FreeSurfer absente" >&2; exit 1; }
[[ -n "${FREESURFER_HOME:-}" ]] || { echo "Module FreeSurfer non charge" >&2; exit 1; }
mkdir -p "$SUBJECTS_DIR"
[[ -e "$SUBJECTS_DIR/fsaverage" ]] || ln -s "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/fsaverage"

if [[ -f "$subject_dir/scripts/recon-all.done" ]] && \
   [[ -f "$subject_dir/stats/aseg.stats" ]] && \
   [[ -f "$subject_dir/stats/lh.aparc.stats" ]] && \
   [[ -f "$subject_dir/stats/rh.aparc.stats" ]]; then
    echo "Deja termine : $branch/$sid"
    exit 0
fi
if [[ -e "$subject_dir" ]] && [[ ! -f "$subject_dir/mri/T1.mgz" ]]; then
    echo "Reconstruction partielle sans T1.mgz : examiner avant reprise" >&2
    exit 1
fi

echo "Date : $(date -Is)"
echo "Branche : $branch ; acquisition : $sid"
echo "Entree : $input"
echo "FreeSurfer : $(recon-all -version 2>&1)"
echo "Dossier de sortie : $SUBJECTS_DIR"

# Même procédure à deux passes que les bras historiques brain-only FS8.
# La première passe peut s'arrêter après avoir créé T1.mgz ; cette exception
# n'est acceptée que si ce fichier existe réellement.
if [[ ! -f "$subject_dir/mri/T1.mgz" ]]; then
    recon-all -s "$sid" -i "$input" -autorecon1 -noskullstrip -no-isrunning \
        -parallel -openmp "${SLURM_CPUS_PER_TASK:-8}" -sd "$SUBJECTS_DIR" || true
fi
[[ -f "$subject_dir/mri/T1.mgz" ]] || { echo "T1.mgz absent apres passe 1" >&2; exit 1; }
cp "$subject_dir/mri/T1.mgz" "$subject_dir/mri/brainmask.auto.mgz"
cp "$subject_dir/mri/T1.mgz" "$subject_dir/mri/brainmask.mgz"
# FreeSurfer 8 peut laisser un verrou après l'arrêt attendu de la première passe.
# Chaque acquisition n'a qu'un seul job dans ce nouvel espace de résultats.
rm -f "$subject_dir/scripts/IsRunning."* 2>/dev/null || true

recon-all -s "$sid" -autorecon1 -autorecon2 -autorecon3 \
    -noskullstrip -no-isrunning -parallel -openmp "${SLURM_CPUS_PER_TASK:-8}" \
    -sd "$SUBJECTS_DIR"

[[ -f "$subject_dir/scripts/recon-all.done" ]] || { echo "recon-all.done absent" >&2; exit 1; }
for expected in aseg.stats lh.aparc.stats rh.aparc.stats; do
    [[ -f "$subject_dir/stats/$expected" ]] || { echo "$expected absent" >&2; exit 1; }
done
echo "Termine : $branch/$sid"
