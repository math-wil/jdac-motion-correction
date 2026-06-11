#!/bin/bash
# Inspection avant/après JDAC pour ds004332 run-02 et run-03
# Usage interactif : ./view_jdac_ds004332.sh
# Usage direct     : ./view_jdac_ds004332.sh <1-22>

PREP_DIR=~/Documents/derivatives/ds004332/jdac_ready
JDAC_DIR=~/Documents/derivatives/ds004332/jdac_outputs

open_subject() {
    local NUM=$1
    if ! [[ "$NUM" =~ ^[0-9]+$ ]] || [ "$NUM" -lt 1 ] || [ "$NUM" -gt 22 ]; then
        echo "Numéro invalide (1-22)."
        return
    fi
    local SUB=$(printf "sub-%02d" "$NUM")
    echo "Sujet : $SUB"

    local PRE_R02=${PREP_DIR}/${SUB}/anat/${SUB}_run-02_brain_norm01.nii.gz
    local PRE_R03=${PREP_DIR}/${SUB}/anat/${SUB}_run-03_brain_norm01.nii.gz
    local JAC_R02=${JDAC_DIR}/run-02/${SUB}/${SUB}_run-02_jdac.nii.gz
    local JAC_R03=${JDAC_DIR}/run-03/${SUB}/${SUB}_run-03_jdac.nii.gz

    local MISSING=0
    for F in "$PRE_R02" "$PRE_R03" "$JAC_R02" "$JAC_R03"; do
        [ ! -f "$F" ] && echo "MANQUANT : $F" && MISSING=1
    done
    [ $MISSING -eq 1 ] && return

    echo "FSLeyes : avant JDAC (run-02, run-03) | après JDAC (run-02, run-03)"
    fsleyes "$PRE_R02" "$PRE_R03" "$JAC_R02" "$JAC_R03" &
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
