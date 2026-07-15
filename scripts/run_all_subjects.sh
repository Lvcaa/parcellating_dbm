#!/bin/bash
# Run the full Jacobian → parcellation → Wasserstein graph pipeline
# for every subject already copied into data/healthy/ and data/unhealthy/.
#
# Usage:
#   bash scripts/run_all_subjects.sh
#   bash scripts/run_all_subjects.sh --dry-run      # print commands, do not run
#
# Graphs are built with the 1/(1+W) similarity formula.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$SCRIPT_DIR/run_jacobian_wasserstein_pipeline.sh"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run_subject() {
    local image="$1"
    local group="$2"
    echo "=========================================="
    echo "[$group] $(basename "$image")"
    echo "=========================================="
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry-run) bash $PIPELINE $image --sim-formula 2"
    else
        bash "$PIPELINE" "$image" --sim-formula 2
    fi
}

HEALTHY_DIR="$PROJECT_ROOT/data/healthy"
UNHEALTHY_DIR="$PROJECT_ROOT/data/unhealthy"

for image in "$HEALTHY_DIR"/*.nii.gz; do
    [[ -f "$image" ]] || { echo "No images found in $HEALTHY_DIR"; break; }
    run_subject "$image" "healthy"
done

for image in "$UNHEALTHY_DIR"/*.nii.gz; do
    [[ -f "$image" ]] || { echo "No images found in $UNHEALTHY_DIR"; break; }
    run_subject "$image" "unhealthy"
done

echo
echo "All subjects processed."
