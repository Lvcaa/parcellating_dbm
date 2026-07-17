#!/bin/bash
set -uo pipefail

usage() {
    echo "Usage: bash scripts/run_batch_pipeline.sh SUBJECTS.txt --run-id ID --rois-dir PATH [options]" >&2
    echo "Options: --sim-formula {1,2} --save-matrix {true,false} --jacobian-geometric {true,false} --num-workers N" >&2
    exit 1
}

[ "$#" -ge 1 ] || usage
ID_LIST="$1"
shift
RUN_ID=""
ROIS_DIR=""
EXTRA_ARGS=()
SIM_FORMULA_CONFIG="both"
SAVE_MATRIX_CONFIG="false"
JACOBIAN_GEOMETRIC_CONFIG="false"
NUM_WORKERS_CONFIG="20"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id)
            [ "$#" -ge 2 ] || usage
            RUN_ID="$2"; shift 2 ;;
        --rois-dir)
            [ "$#" -ge 2 ] || usage
            ROIS_DIR="$2"; shift 2 ;;
        --sim-formula)
            [ "$#" -ge 2 ] || usage
            SIM_FORMULA_CONFIG="$2"
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --save-matrix)
            [ "$#" -ge 2 ] || usage
            SAVE_MATRIX_CONFIG="$2"
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --jacobian-geometric)
            [ "$#" -ge 2 ] || usage
            JACOBIAN_GEOMETRIC_CONFIG="$2"
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --num-workers)
            [ "$#" -ge 2 ] || usage
            NUM_WORKERS_CONFIG="$2"
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *) usage ;;
    esac
done

[ -f "$ID_LIST" ] || { echo "Error: subject list not found: $ID_LIST" >&2; exit 1; }
[ -n "$RUN_ID" ] || { echo "Error: --run-id is required" >&2; exit 1; }
[ -n "$ROIS_DIR" ] || { echo "Error: --rois-dir is required" >&2; exit 1; }
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "Error: invalid --run-id" >&2; exit 1 ;; esac
case "$SIM_FORMULA_CONFIG" in both|1|2) ;; *) echo "Error: --sim-formula must be 1 or 2" >&2; exit 1 ;; esac
case "$SAVE_MATRIX_CONFIG" in true|false) ;; *) echo "Error: --save-matrix must be true or false" >&2; exit 1 ;; esac
case "$JACOBIAN_GEOMETRIC_CONFIG" in true|false) ;; *) echo "Error: --jacobian-geometric must be true or false" >&2; exit 1 ;; esac
case "$NUM_WORKERS_CONFIG" in ''|*[!0-9]*|0) echo "Error: --num-workers must be positive" >&2; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ID_LIST="$(realpath "$ID_LIST")"
ROIS_DIR="$(realpath "$ROIS_DIR")"
ATLAS_MANIFEST="$ROIS_DIR/atlas_manifest.json"
RUN_ROOT="$PROJECT_ROOT/outputs/runs/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
RUN_MANIFEST="$RUN_ROOT/run_manifest.json"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_jacobian_wasserstein_pipeline.sh"
FREEZER="$SCRIPT_DIR/validation/freeze_pipeline.py"
PREFLIGHT="$SCRIPT_DIR/validation/validate_batch_inputs.py"
DATABASE="$PROJECT_ROOT/data/database_finale_labels_corrette.csv"
WARPS_ROOT="$PROJECT_ROOT/data/warps"

[ -f "$ATLAS_MANIFEST" ] || { echo "Error: atlas manifest missing: $ATLAS_MANIFEST" >&2; exit 1; }
python "$PREFLIGHT" \
    --subjects "$ID_LIST" \
    --database "$DATABASE" \
    --warps-root "$WARPS_ROOT"
mkdir -p "$LOG_DIR"

FREEZE_ARGS=(
    --manifest "$RUN_MANIFEST"
    --run-id "$RUN_ID"
    --atlas-manifest "$ATLAS_MANIFEST"
    --subject-list "$ID_LIST"
    --file "$PIPELINE_SCRIPT"
    --file "$SCRIPT_DIR/registration/run_ants_jacobian.py"
    --file "$SCRIPT_DIR/parcellation/export_masked_jacobian_vectors.py"
    --file "$SCRIPT_DIR/graph_building/wasserstein_distance_graph2.py"
    --file "$SCRIPT_DIR/pipeline_integrity.py"
    --file "$PREFLIGHT"
    --config "sim_formula=$SIM_FORMULA_CONFIG"
    --config "save_matrix=$SAVE_MATRIX_CONFIG"
    --config "jacobian_geometric=$JACOBIAN_GEOMETRIC_CONFIG"
    --config "num_workers=$NUM_WORKERS_CONFIG"
)
if [ -f "$RUN_MANIFEST" ]; then
    python "$FREEZER" check "${FREEZE_ARGS[@]}"
else
    python "$FREEZER" create "${FREEZE_ARGS[@]}"
    : > "$LOG_DIR/batch_failures.log"
fi

TOTAL=$(awk 'NF {count++} END {print count+0}' "$ID_LIST")
COUNT=0
FAILED=0
START_TIME=$(date +%s)

while IFS= read -r SUBJECT_ID; do
    [ -n "$SUBJECT_ID" ] || continue
    COUNT=$((COUNT + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "=== [$COUNT/$TOTAL] $SUBJECT_ID (elapsed ${ELAPSED}s) ==="
    if ! python "$FREEZER" check "${FREEZE_ARGS[@]}"; then
        echo "Frozen pipeline changed; aborting before $SUBJECT_ID" >&2
        exit 2
    fi
    if bash "$PIPELINE_SCRIPT" "$SUBJECT_ID" \
        --run-root "$RUN_ROOT" \
        --rois-dir "$ROIS_DIR" \
        "${EXTRA_ARGS[@]}" >> "$LOG_DIR/pipeline_${SUBJECT_ID}.log" 2>&1; then
        echo "[$COUNT/$TOTAL] $SUBJECT_ID done"
    else
        echo "[$COUNT/$TOTAL] $SUBJECT_ID FAILED (see $LOG_DIR/pipeline_${SUBJECT_ID}.log)"
        echo "$SUBJECT_ID" >> "$LOG_DIR/batch_failures.log"
        FAILED=$((FAILED + 1))
    fi
done < "$ID_LIST"

echo "Batch complete: $((COUNT - FAILED)) succeeded, $FAILED failed out of $TOTAL"
[ "$FAILED" -eq 0 ]
