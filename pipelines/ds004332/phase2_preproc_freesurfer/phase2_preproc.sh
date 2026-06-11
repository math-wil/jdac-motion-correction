#!/bin/bash
#SBATCH --job-name=phase2_preproc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --account=def-sbouix
#SBATCH --array=0-2
#SBATCH --output=/home/mathw/logs/phase2_preproc_%A_%a.out
#SBATCH --error=/home/mathw/logs/phase2_preproc_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mathilde.wilfart.1@ens.etsmtl.ca

# ==============================================================================
# Phase 2 — Preprocessing T1w ds004332
# ==============================================================================
#
# Pipeline :
#   raw T1w
#   → Clinica t1-linear
#   → SynthStrip
#
# Sorties principales :
#   *_clinica_t1linear.nii.gz
#       sortie Clinica conservée pour traçabilité/QC
#
#   *_clinica_synthstrip_brain.nii.gz
#       image brain-only avec intensités conservées
#       entrée FreeSurfer Phase 2 avec -noskullstrip
#
#   *_clinica_synthstrip_mask.nii.gz
#       masque binaire SynthStrip
#
#   *_clinica_synthstrip_brain_norm01.nii.gz
#       même image brain-only, normalisée dans [0,1]
#       entrée JDAC Phase 3
#
# Phase 2 teste l'effet du preprocessing seul :
#   raw → Clinica → SynthStrip → FreeSurfer
#
# Phase 3 testera l'effet de JDAC :
#   raw → Clinica → SynthStrip → normalisation [0,1] → JDAC → FreeSurfer
#
# ==============================================================================

# Arrête le script si :
# - une commande échoue ;
# - une variable non définie est utilisée ;
# - une commande dans un pipe échoue.
set -euo pipefail
trap 'echo "ERREUR ligne $LINENO exit=$?" >&2' ERR

# Création du dossier de logs si absent.
mkdir -p /home/mathw/logs

# Informations de début de job, utiles dans les fichiers .out.
echo "=== Phase 2 preproc — $(date) ==="
echo "Array task : ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Host       : $(hostname)"
echo

# ==============================================================================
# 0. Environnement logiciel
# ==============================================================================

# Chargement des modules nécessaires sur Narval :
# - ants : utilisé par Clinica t1-linear pour la registration affine ;
# - freesurfer/8.0.0-1 : fournit mri_synthstrip.
module load StdEnv/2023 ants freesurfer/8.0.0-1

# Initialisation de l'environnement FreeSurfer.
# Nécessaire pour que les commandes FreeSurfer soient disponibles dans le PATH.
export FS_FREESURFERENV_NO_OUTPUT=1
export PS1=""
set +euo pipefail
source "$EBROOTFREESURFER/FreeSurferEnv.sh"
set -euo pipefail

# Activation de l'environnement Python contenant Clinica.
set +euo pipefail
source ~/clinica_env/bin/activate
unset PYTHONPATH
set -euo pipefail

# Affichage des chemins et versions réellement utilisés par le job.
# Sert à vérifier dans les logs que le bon environnement est chargé.
echo "clinica          : $(which clinica || true)"
echo "clinica version  : $(clinica --version 2>/dev/null || true)"
echo "antsRegistration : $(which antsRegistration || true)"
echo "mri_synthstrip   : $(which mri_synthstrip || true)"
echo "python3          : $(which python3 || true)"
echo

# Vérification que Clinica est accessible.
if ! command -v clinica >/dev/null 2>&1; then
    echo "ERREUR : clinica introuvable."
    exit 1
fi

# Vérification que ANTs est accessible.
# Clinica t1-linear dépend d'ANTs pour le recalage vers MNI.
if ! command -v antsRegistration >/dev/null 2>&1; then
    echo "ERREUR : antsRegistration introuvable."
    exit 1
fi

# Vérification que SynthStrip est accessible.
if ! command -v mri_synthstrip >/dev/null 2>&1; then
    echo "ERREUR : mri_synthstrip introuvable."
    exit 1
fi

# Vérification que l'environnement Python contient les packages nécessaires
# à la normalisation [0,1] et au contrôle des fichiers NIfTI.
python3 - <<'PYTEST'
import nibabel
import numpy
PYTEST

# ==============================================================================
# 1. Paramètres projet
# ==============================================================================

# Dossier contenant les images brutes ds004332 sur Narval.
RAW_DIR=/scratch/mathw/ds004332

# Dossier de sortie des images prétraitées.
OUT_ROOT=/home/mathw/projects/ctb-sbouix/mathw/phase2_preproc

# Dossier CAPS produit par Clinica.
# Chaque run aura son propre CAPS temporaire/permanent pour éviter les collisions.
CAPS_ROOT=/home/mathw/projects/ctb-sbouix/mathw/phase2_clinica_caps

mkdir -p "$OUT_ROOT" "$CAPS_ROOT"

# Liste initiale des runs à traiter.
# Le job array Slurm utilise l'indice 0, 1 ou 2 pour sélectionner une entrée.
SUBJECTS=(
  sub-01_run-01
  sub-01_run-02
  sub-01_run-03
)

# Sécurité : vérifie que l'indice du job array correspond bien à un élément.
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
    echo "ERREUR : SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} hors limites."
    exit 1
fi

# Sujet/run traité par cette tâche Slurm.
SUBJ_RUN="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

# Extraction de sub-XX et run-YY depuis une chaîne de type sub-01_run-01.
SUB=$(echo "$SUBJ_RUN" | cut -d_ -f1)
RUN=$(echo "$SUBJ_RUN" | cut -d_ -f2)

# Session artificielle pour le mini-BIDS.
# Le dataset original n'a pas besoin d'une session réelle ici ; elle sert à rendre
# la structure BIDS plus stable pour Clinica.
SES=ses-01

echo "Sujet/run : $SUBJ_RUN"
echo "SUB       : $SUB"
echo "RUN       : $RUN"
echo "SES       : $SES"
echo

# ==============================================================================
# 2. Recherche de l'image brute T1w
# ==============================================================================

# Dossier anatomique du sujet dans le dataset brut.
ANAT_DIR="$RAW_DIR/$SUB/anat"

# Vérifie que le dossier anatomique existe.
if [ ! -d "$ANAT_DIR" ]; then
    echo "ERREUR : dossier anat introuvable : $ANAT_DIR"
    exit 1
fi

# Recherche du fichier T1w correspondant au run.
# Le nom exact contient plusieurs entités BIDS possibles, par exemple acq/rec.
# On ne hardcode donc pas toute la chaîne, seulement le run et le suffixe T1w.
RAW=$(find "$ANAT_DIR" \
    -maxdepth 1 \
    -type f \
    \( -name "*_${RUN}_T1w.nii" -o -name "*_${RUN}_T1w.nii.gz" \) \
    | sort \
    | head -1)

# Arrêt si aucune image n'est trouvée.
if [ -z "$RAW" ]; then
    echo "ERREUR : image brute introuvable pour $SUBJ_RUN"
    echo "Contenu de $ANAT_DIR :"
    ls -lh "$ANAT_DIR" || true
    exit 1
fi

echo "[0] Image brute : $RAW"
echo

# ==============================================================================
# 3. Construction d'un mini-BIDS temporaire
# ==============================================================================

# Clinica attend une entrée au format BIDS.
# Comme le but est de traiter chaque run séparément, le script crée un mini-BIDS
# temporaire contenant uniquement l'image du run courant.

# Dossier de travail local au job Slurm.
# SLURM_TMPDIR est plus rapide que /home pour les opérations temporaires.
WORK="${SLURM_TMPDIR:-/tmp}/phase2_${SUBJ_RUN}"

# Racine du mini-BIDS temporaire.
MINI_BIDS="$WORK/bids"

# CAPS Clinica spécifique à ce sujet/run.
CAPS_DIR="$CAPS_ROOT/$SUBJ_RUN"

# Dossier de travail utilisé par Clinica.
CLINICA_WD="$WORK/clinica_wd"

# Nettoyage/recréation du dossier de travail.
rm -rf "$WORK"
mkdir -p "$MINI_BIDS/$SUB/$SES/anat" "$CAPS_DIR" "$CLINICA_WD"

# Fichier minimal requis par BIDS.
cat > "$MINI_BIDS/dataset_description.json" <<JSON
{
  "Name": "ds004332_phase2_${SUBJ_RUN}",
  "BIDSVersion": "1.8.0",
  "DatasetType": "raw"
}
JSON

# Fichier participants minimal avec sujet et session.
printf "participant_id\tsession_id\n%s\t%s\n" "$SUB" "$SES" > "$MINI_BIDS/participants.tsv"

# Nom original du fichier brut.
RAW_BASE=$(basename "$RAW")

# Nouveau nom BIDS incluant la session artificielle.
# Exemple :
#   sub-01_acq-..._run-01_T1w.nii
# devient :
#   sub-01_ses-01_acq-..._run-01_T1w.nii
NEW_BASE="${SUB}_${SES}_${RAW_BASE#${SUB}_}"

# Symlink du fichier brut vers le mini-BIDS.
# Le fichier n'est pas copié physiquement, ce qui évite de dupliquer les données.
ln -sf "$RAW" "$MINI_BIDS/$SUB/$SES/anat/$NEW_BASE"

# Recherche du JSON sidecar correspondant au NIfTI brut.
if [[ "$RAW" == *.nii.gz ]]; then
    RAW_JSON="${RAW%.nii.gz}.json"
elif [[ "$RAW" == *.nii ]]; then
    RAW_JSON="${RAW%.nii}.json"
else
    RAW_JSON=""
fi

# Symlink du JSON sidecar si présent.
# Le JSON contient les métadonnées d'acquisition ; il peut être utile pour BIDS/Clinica.
if [ -n "$RAW_JSON" ] && [ -f "$RAW_JSON" ]; then
    RAW_JSON_BASE=$(basename "$RAW_JSON")
    NEW_JSON="${SUB}_${SES}_${RAW_JSON_BASE#${SUB}_}"
    ln -sf "$RAW_JSON" "$MINI_BIDS/$SUB/$SES/anat/$NEW_JSON"
    echo "[1] JSON sidecar : $NEW_JSON"
else
    echo "[1] JSON sidecar absent."
fi

# Affichage du mini-BIDS créé.
echo "[1] Mini-BIDS : $MINI_BIDS"
find "$MINI_BIDS" -maxdepth 5 \( -type f -o -type l \) | sort
echo

# ==============================================================================
# 4. Clinica t1-linear
# ==============================================================================

# Clinica t1-linear applique notamment :
# - correction de biais N4 ;
# - registration affine avec ANTs ;
# - recalage vers le template MNI152NLin2009cSym ;
# - production d'un CAPS.
#
# L'option --uncropped_image demande de conserver une image non croppée.
# Cette sortie est utilisée ensuite pour SynthStrip.

echo "[2] Clinica t1-linear..."

# echo "y" répond automatiquement à une éventuelle confirmation interactive.
echo "y" | clinica run t1-linear \
    "$MINI_BIDS" \
    "$CAPS_DIR" \
    -wd "$CLINICA_WD" \
    -np "$SLURM_CPUS_PER_TASK" \
    --uncropped_image

echo "[2] Clinica terminé."
echo

# Vérifie que Clinica a produit un dossier subjects dans le CAPS.
if [ ! -d "$CAPS_DIR/subjects" ]; then
    echo "ERREUR : dossier subjects absent dans $CAPS_DIR"
    find "$CAPS_DIR" -maxdepth 5 \( -type f -o -type l \) | sort || true
    exit 1
fi

# Recherche de l'image T1w produite par Clinica en espace MNI152NLin2009cSym.
# La version desc-Crop est exclue pour garder la version non croppée.
CLINICA_IMG=$(find "$CAPS_DIR/subjects" \
    -type f \
    -name "*space-MNI152NLin2009cSym*res-1x1x1*T1w.nii.gz" \
    ! -name "*desc-Crop*" \
    | sort \
    | head -1)

# Arrêt si la sortie attendue n'est pas trouvée.
if [ -z "$CLINICA_IMG" ]; then
    echo "ERREUR : sortie Clinica non croppée introuvable."
    find "$CAPS_DIR" -type f | sort
    exit 1
fi

echo "[2] Sortie Clinica : $CLINICA_IMG"
echo

# ==============================================================================
# 5. SynthStrip
# ==============================================================================

# SynthStrip retire les tissus non cérébraux.
# Il produit :
# - une image brain-only avec les intensités de la sortie Clinica conservées ;
# - un masque binaire du cerveau.

# Dossier final de sortie pour ce sujet/run.
OUT_DIR="$OUT_ROOT/$SUBJ_RUN"
mkdir -p "$OUT_DIR"

# Fichiers produits/conservés.
CLINICA_COPY="$OUT_DIR/${SUBJ_RUN}_clinica_t1linear.nii.gz"
FS_BRAIN="$OUT_DIR/${SUBJ_RUN}_clinica_synthstrip_brain.nii.gz"
MASK="$OUT_DIR/${SUBJ_RUN}_clinica_synthstrip_mask.nii.gz"
JDAC_INPUT="$OUT_DIR/${SUBJ_RUN}_clinica_synthstrip_brain_norm01.nii.gz"

# Copie de la sortie Clinica dans le dossier final pour traçabilité.
cp "$CLINICA_IMG" "$CLINICA_COPY"

echo "[3] SynthStrip..."

# Skull stripping sur la sortie Clinica.
mri_synthstrip \
    -i "$CLINICA_COPY" \
    -o "$FS_BRAIN" \
    -m "$MASK"

echo "[3] SynthStrip terminé."
echo "    FreeSurfer input : $FS_BRAIN"
echo "    SynthStrip mask  : $MASK"
echo

# ==============================================================================
# 6. Normalisation [0,1] pour JDAC
# ==============================================================================

# JDAC attend une image brain-only avec intensités dans [0,1].
# La normalisation est calculée uniquement sur les voxels du cerveau, définis
# par le masque SynthStrip.
#
# Hors cerveau, les voxels restent à 0.
#
# FS_BRAIN :
#   image brain-only avec intensités conservées, utilisée pour FreeSurfer Phase 2.
#
# JDAC_INPUT :
#   même image, mais normalisée [0,1], utilisée comme entrée JDAC Phase 3.

echo "[4] Normalisation [0,1] pour JDAC..."

python3 - << PYEOF
import nibabel as nib
import numpy as np
from pathlib import Path

brain_path = Path("$FS_BRAIN")
mask_path = Path("$MASK")
out_path = Path("$JDAC_INPUT")

brain_img = nib.load(str(brain_path))
mask_img = nib.load(str(mask_path))

brain = np.asanyarray(brain_img.dataobj).astype(np.float32)
mask = np.asanyarray(mask_img.dataobj) > 0

if brain.shape != mask.shape:
    raise RuntimeError(f"Shape mismatch: brain={brain.shape}, mask={mask.shape}")

vals = brain[mask]

if vals.size == 0:
    raise RuntimeError("Masque SynthStrip vide.")

vmin = float(np.nanmin(vals))
vmax = float(np.nanmax(vals))
vrange = vmax - vmin

if not np.isfinite(vmin) or not np.isfinite(vmax) or vrange <= 0:
    raise RuntimeError(f"Intensités invalides : min={vmin}, max={vmax}, range={vrange}")

norm = np.zeros_like(brain, dtype=np.float32)
norm[mask] = (brain[mask] - vmin) / vrange
norm = np.clip(norm, 0.0, 1.0)

out = nib.Nifti1Image(norm, brain_img.affine, brain_img.header)
out.set_data_dtype(np.float32)
nib.save(out, str(out_path))

print("Normalisation JDAC OK")
print(f"    brain shape    : {brain.shape}")
print(f"    voxels cerveau : {int(mask.sum())}")
print(f"    range input    : [{vmin:.3f}, {vmax:.3f}]")
print(f"    output JDAC    : {out_path}")
PYEOF

echo

# ==============================================================================
# 7. QC rapide des sorties
# ==============================================================================

# Liste des fichiers finaux.
echo "[5] QC sorties"
ls -lh "$OUT_DIR"
echo

# Affichage des dimensions et types de données des NIfTI produits.
# Permet de vérifier que les fichiers ont la même géométrie.
python3 - << PYEOF
import nibabel as nib
from pathlib import Path

paths = [
    Path("$CLINICA_COPY"),
    Path("$FS_BRAIN"),
    Path("$MASK"),
    Path("$JDAC_INPUT"),
]

for p in paths:
    img = nib.load(str(p))
    print(f"{p.name}: shape={img.shape}, dtype={img.get_data_dtype()}")
PYEOF

echo

# Rappel des commandes/logiques à utiliser ensuite.
echo "=== Utilisation ensuite ==="
echo
echo "FreeSurfer Phase 2 :"
echo "  recon-all -s ${SUBJ_RUN}_phase2 -i $FS_BRAIN -sd <SUBJECTS_DIR> -all -noskullstrip"
echo
echo "JDAC Phase 3 input :"
echo "  $JDAC_INPUT"
echo
echo "Après JDAC :"
echo "  recon-all -s ${SUBJ_RUN}_jdac -i <JDAC_OUTPUT.nii.gz> -sd <SUBJECTS_DIR> -all -noskullstrip"
echo
echo "=== Fin $SUBJ_RUN — $(date) ==="
