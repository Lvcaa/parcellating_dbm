#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash scripts/run_jacobian_wasserstein_pipeline.sh <subject_id> --run-root PATH --rois-dir PATH [options]" >&2
    echo "Options: --sim-formula {1,2} --save-matrix {true,false} --jacobian-geometric {true,false} --num-workers N" >&2
    exit 1
}

[ "$#" -ge 1 ] || usage
SUBJECT_ID="$1"
shift

SIM_FORMULA=""
SAVE_MATRIX="false"
RUN_ROOT=""
ROIS_DIR=""
JACOBIAN_GEOMETRIC="false"
NUM_WORKERS=20

while [ "$#" -gt 0 ]; do
    case "$1" in
        --sim-formula)
            [ "$#" -ge 2 ] || usage
            case "$2" in 1|2) ;; *) usage ;; esac
            SIM_FORMULA="$2"; shift 2 ;;
        --save-matrix)
            [ "$#" -ge 2 ] || usage
            case "$2" in true|false) ;; *) usage ;; esac
            SAVE_MATRIX="$2"; shift 2 ;;
        --run-root)
            [ "$#" -ge 2 ] || usage
            RUN_ROOT="$2"; shift 2 ;;
        --rois-dir)
            [ "$#" -ge 2 ] || usage
            ROIS_DIR="$2"; shift 2 ;;
        --jacobian-geometric)
            [ "$#" -ge 2 ] || usage
            case "$2" in true|false) ;; *) usage ;; esac
            JACOBIAN_GEOMETRIC="$2"; shift 2 ;;
        --num-workers)
            [ "$#" -ge 2 ] || usage
            NUM_WORKERS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[ -n "$RUN_ROOT" ] || { echo "Error: --run-root is required" >&2; exit 1; }
[ -n "$ROIS_DIR" ] || { echo "Error: --rois-dir is required" >&2; exit 1; }
case "$NUM_WORKERS" in ''|*[!0-9]*|0) echo "Error: --num-workers must be positive" >&2; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
ROIS_DIR="$(realpath -m "$ROIS_DIR")"
ATLAS_MANIFEST="$ROIS_DIR/atlas_manifest.json"
WARP_IMAGE="$PROJECT_ROOT/data/warps/$SUBJECT_ID/Reg_/_SyN1Warp.nii.gz"
ANTS_OUTPUT_ROOT="$RUN_ROOT/ants_registration"
PARCEL_OUTPUT_ROOT="$RUN_ROOT/jacobian_parcel_vectors"

[ -f "$WARP_IMAGE" ] || { echo "Error: warp image does not exist: $WARP_IMAGE" >&2; exit 1; }
[ -f "$ATLAS_MANIFEST" ] || { echo "Error: validated atlas manifest missing: $ATLAS_MANIFEST" >&2; exit 1; }

python "$SCRIPT_DIR/registration/run_ants_jacobian.py" \
    --warp-image "$WARP_IMAGE" \
    --subject-id "$SUBJECT_ID" \
    --output-root "$ANTS_OUTPUT_ROOT" \
    --jacobian-geometric "$JACOBIAN_GEOMETRIC"

LOG_JACOBIAN="$ANTS_OUTPUT_ROOT/$SUBJECT_ID/${SUBJECT_ID}_to_template_logJacobian.nii.gz"
[ -f "$LOG_JACOBIAN" ] || { echo "Error: log-Jacobian missing: $LOG_JACOBIAN" >&2; exit 1; }

python "$SCRIPT_DIR/parcellation/export_masked_jacobian_vectors.py" \
    --jacobian "$LOG_JACOBIAN" \
    --rois-dir "$ROIS_DIR" \
    --roi-manifest "$ATLAS_MANIFEST" \
    --output-dir "$PARCEL_OUTPUT_ROOT" \
    --num-workers "$NUM_WORKERS"

PARCEL_SUBJECT_DIR="$PARCEL_OUTPUT_ROOT/$SUBJECT_ID"
GRAPH_SCRIPT="$SCRIPT_DIR/graph_building/wasserstein_distance_graph2.py"

if [ -z "$SIM_FORMULA" ] || [ "$SIM_FORMULA" = "1" ]; then
    python "$GRAPH_SCRIPT" \
        --input-folder "$PARCEL_SUBJECT_DIR" \
        --output-folder "$RUN_ROOT/wasserstein_graphs_expW/$SUBJECT_ID" \
        --sim-formula 1 \
        --save-matrix "$SAVE_MATRIX" \
        --num-workers "$NUM_WORKERS"
fi

if [ -z "$SIM_FORMULA" ] || [ "$SIM_FORMULA" = "2" ]; then
    python "$GRAPH_SCRIPT" \
        --input-folder "$PARCEL_SUBJECT_DIR" \
        --output-folder "$RUN_ROOT/wasserstein_graphs_inv1pW/$SUBJECT_ID" \
        --sim-formula 2 \
        --save-matrix "$SAVE_MATRIX" \
        --num-workers "$NUM_WORKERS"
fi

echo "Pipeline complete: $SUBJECT_ID"
