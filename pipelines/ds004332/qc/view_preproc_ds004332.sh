#!/bin/bash
# Affiche brut + preprocessé pour un sujet ds004332
# Usage interactif : ./view_preproc_ds004332.sh
# Usage direct     : ./view_preproc_ds004332.sh <1-22>

RAW_DIR=~/Documents/raw_datasets/ds004332
PREP_DIR=~/Documents/derivatives/ds004332/jdac_ready

open_subject() {
    local NUM=$1
    if [ "$NUM" -lt 1 ] || [ "$NUM" -gt 22 ] 2>/dev/null; then
        echo "Numéro invalide (1-22)."
        return
    fi
    local SUB=$(printf "sub-%02d" "$NUM")
    echo "Sujet : $SUB"

    local RAW_RUN01=${RAW_DIR}/${SUB}/anat/${SUB}_acq-mpragepmcoff_rec-wore_run-01_T1w.nii
    local RAW_RUN02=${RAW_DIR}/${SUB}/anat/${SUB}_acq-mpragepmcoff_rec-wore_run-02_T1w.nii
    local RAW_RUN03=${RAW_DIR}/${SUB}/anat/${SUB}_acq-mpragepmcoff_rec-wore_run-03_T1w.nii
    local PREP_RUN01=${PREP_DIR}/${SUB}/anat/${SUB}_run-01_brain_norm01.nii.gz
    local PREP_RUN02=${PREP_DIR}/${SUB}/anat/${SUB}_run-02_brain_norm01.nii.gz
    local PREP_RUN03=${PREP_DIR}/${SUB}/anat/${SUB}_run-03_brain_norm01.nii.gz

    local MISSING=0
    for F in "$RAW_RUN01" "$RAW_RUN02" "$RAW_RUN03" "$PREP_RUN01" "$PREP_RUN02" "$PREP_RUN03"; do
        if [ ! -f "$F" ]; then
            echo "MANQUANT : $F"
            MISSING=1
        fi
    done
    [ $MISSING -eq 1 ] && return

    echo "FSLeyes : brut run-01/02/03 | preprocessé run-01/02/03"
    fsleyes "$RAW_RUN01" "$RAW_RUN02" "$RAW_RUN03" \
            "$PREP_RUN01" "$PREP_RUN02" "$PREP_RUN03" &
}

if [ -n "$1" ]; then
    open_subject "$1"
else
    while true; do
        read -rp "Sujet (1-22, q pour quitter) : " INPUT
        [ "$INPUT" = "q" ] && break
        open_subject "$INPUT"
    done
fi
