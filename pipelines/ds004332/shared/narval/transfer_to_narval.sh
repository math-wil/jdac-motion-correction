#!/bin/bash
# Transfert des images brutes ds004332 vers Narval (scratch).
#
# À exécuter depuis la machine locale (pas sur Narval).
# Prérequis : avoir une clé SSH configurée pour Narval.
#
# Usage :
#   bash transfer_to_narval.sh <narval_user>
#   ex : bash transfer_to_narval.sh av62870

set -euo pipefail

NARVAL_USER="${1:-}"
if [ -z "$NARVAL_USER" ]; then
    echo "Usage : $0 <narval_user>"
    exit 1
fi

NARVAL_HOST="narval.computecanada.ca"
# ds004332_minimal contient déjà uniquement les images acq-mpragepmcoff rec-wore
# (PMC off, sans réacquisition) — 66 NIfTI, structure sub-XX/anat/
LOCAL_DS="$HOME/Documents/derivatives/ds004332_minimal"
REMOTE_DEST="${NARVAL_USER}@${NARVAL_HOST}:scratch/ds004332"

if [ ! -d "$LOCAL_DS" ]; then
    echo "ERREUR : dossier local introuvable : $LOCAL_DS"
    exit 1
fi

echo "Transfert : $LOCAL_DS → $REMOTE_DEST"
echo "66 images (22 sujets × run-01/02/03), acq-mpragepmcoff rec-wore."

rsync -avz --progress "$LOCAL_DS/" "$REMOTE_DEST/"

echo "Transfert terminé."
