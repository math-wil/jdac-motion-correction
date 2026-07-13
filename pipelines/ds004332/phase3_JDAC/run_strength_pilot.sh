#!/usr/bin/env bash
# Run the minimal JDAC anti-artifact strength pilot on the laboratory PC.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/pilot_jdac_strength.csv"
JDAC_SOURCE="${JDAC_SOURCE:-$HOME/Documents/jdac}"
OUTPUT_DIR="${JDAC_PILOT_OUT:-$HOME/Documents/derivatives/ds004332/jdac_strength_pilot}"

test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST"; exit 1; }
test -d "$JDAC_SOURCE/PretrainedModels" || {
    echo "Missing JDAC models: $JDAC_SOURCE/PretrainedModels"
    exit 1
}

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/pilot_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "JDAC strength pilot"
echo "Manifest: $MANIFEST"
echo "Outputs:  $OUTPUT_DIR"
echo "Baseline step 1.00 is reused from the existing jdac_rigid condition."

cd "$JDAC_SOURCE"
for step in 0.25 0.50; do
    echo
    echo "===== antiart step $step ====="
    conda run -n cortical-motion python "$SCRIPT_DIR/run_jdac.py" \
        --subjects "$MANIFEST" \
        --out_dir "$OUTPUT_DIR" \
        --antiart-step-lr "$step" \
        --max-iter 4
done

echo
echo "Pilot inference complete."
find "$OUTPUT_DIR" -type f -name '*_T1w_jdac_step*_iter4.nii.gz' | sort
