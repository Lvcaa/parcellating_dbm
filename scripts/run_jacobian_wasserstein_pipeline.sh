#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/run_jacobian_wasserstein_pipeline.sh <moving_image.nii.gz>" >&2
    exit 1
fi

MOVING_IMAGE="$1"
if [ ! -f "$MOVING_IMAGE" ]; then
    echo "Error: moving image does not exist: $MOVING_IMAGE" >&2
    exit 1
fi

MOVING_IMAGE_NAME="$(basename "$MOVING_IMAGE")"
if [[ "$MOVING_IMAGE_NAME" == *.nii.gz ]]; then
    SUBJECT_ID="${MOVING_IMAGE_NAME%.nii.gz}"
elif [[ "$MOVING_IMAGE_NAME" == *.nii ]]; then
    SUBJECT_ID="${MOVING_IMAGE_NAME%.nii}"
else
    SUBJECT_ID="${MOVING_IMAGE_NAME%.*}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FIXED_IMAGE="$PROJECT_ROOT/data/reference/MNI152_T1_1mm.nii.gz"
ANTS_OUTPUT_ROOT="$PROJECT_ROOT/outputs/ants_registration"
PARCEL_OUTPUT_ROOT="$PROJECT_ROOT/outputs/jacobian_parcel_vectors"

ANTS_SCRIPT="$SCRIPT_DIR/registration/run_ants_jacobian.py"
EXPORT_SCRIPT="$SCRIPT_DIR/parcellation/export_masked_jacobian_vectors.py"
GRAPH_SCRIPT="$SCRIPT_DIR/graph_building/wasserstein_distance_graph2.py"

if [ ! -f "$FIXED_IMAGE" ]; then
    echo "Error: fixed template does not exist: $FIXED_IMAGE" >&2
    exit 1
fi

LOG_JACOBIAN="$ANTS_OUTPUT_ROOT/$SUBJECT_ID/${SUBJECT_ID}_to_template_logJacobian.nii.gz"

echo
if [ -f "$LOG_JACOBIAN" ]; then
    echo "Step 1/4: found existing log-Jacobian, skipping ANTs"
    echo "  $LOG_JACOBIAN"
else
    echo "Step 1/4: running ANTs registration and log-Jacobian export"
    python "$ANTS_SCRIPT" \
        --fixed-image "$FIXED_IMAGE" \
        --moving-image "$MOVING_IMAGE" \
        --subject-id "$SUBJECT_ID"
fi

if [ -z "$LOG_JACOBIAN" ] || [ ! -f "$LOG_JACOBIAN" ]; then
    echo "Error: expected log-Jacobian image was not created: $LOG_JACOBIAN" >&2
    exit 1
fi

LOG_JACOBIAN_NAME="$(basename "$LOG_JACOBIAN")"
LOG_JACOBIAN_STEM="${LOG_JACOBIAN_NAME%.nii.gz}"
PARCEL_SUBJECT_ID="$LOG_JACOBIAN_STEM"
IFS="_" read -r -a LOG_STEM_PARTS <<< "$LOG_JACOBIAN_STEM"
for PART in "${LOG_STEM_PARTS[@]}"; do
    if [[ "$PART" == sub-* ]]; then
        PARCEL_SUBJECT_ID="$PART"
        break
    fi
done

echo
PARCEL_SUBJECT_DIR="$PARCEL_OUTPUT_ROOT/$PARCEL_SUBJECT_ID"
if [ -d "$PARCEL_SUBJECT_DIR" ] && find "$PARCEL_SUBJECT_DIR" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "Step 2/4: found existing parcel vectors, skipping extraction"
    echo "  $PARCEL_SUBJECT_DIR"
else
    echo "Step 2/4: extracting parcel vectors from $LOG_JACOBIAN"
    python "$EXPORT_SCRIPT" \
        --jacobian "$LOG_JACOBIAN" \
        --num-workers 8
fi

if [ ! -d "$PARCEL_SUBJECT_DIR" ]; then
    echo "Error: expected parcel-vector folder was not created: $PARCEL_SUBJECT_DIR" >&2
    exit 1
fi

echo
echo "Step 3/4: building Wasserstein graph with exp(-W)"
python "$GRAPH_SCRIPT" \
    --input-folder "$PARCEL_SUBJECT_DIR" \
    --sim-formula 1 \
    --num-workers 8

echo
echo "Step 4/4: building Wasserstein graph with 1/(1+W)"
python "$GRAPH_SCRIPT" \
    --input-folder "$PARCEL_SUBJECT_DIR" \
    --sim-formula 2 \
    --num-workers 8

echo
echo "Pipeline complete."
