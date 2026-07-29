#!/usr/bin/env bash
# Full pipeline: ROI check -> cohort preflight -> freeze -> per-subject
# (log-Jacobian -> parcel vectors -> was/kl/meiq graphs) -> cohort
# validation -> parcel-wise statistics.
#
# Registration is NOT part of this script: data/warps/ already holds
# validated forward warps, so the pipeline starts from the Jacobian
# determinant computation.
#
# Every stage below is a hard gate (set -euo pipefail): the script stops
# at the first failed assertion instead of continuing on bad input. Every
# underlying python call is already idempotent via its own completion
# marker, so re-running this script after a partial failure resumes
# instead of redoing finished work.
#
# Usage:
#   scripts/pipelines/run_full_pipeline.sh [run-id]
#
# Environment overrides (all optional):
#   SUBJECTS_FILE   default: data/test_cohort_warps_available.txt
#   DATABASE        default: data/database_finale_labels_corrette.csv
#   WARPS_ROOT      default: data/warps
#   ROIS_DIR        default: outputs/rois
#   EXPECTED_HEALTHY / EXPECTED_UNHEALTHY   default: 684 / 504
#   EXPORT_WORKERS  default: 4   (parcel-vector export thread count)
#   GRAPH_WORKERS   default: 8   (graph-building process count)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${1:-full_pipeline_$(date +%Y%m%d)}"
SUBJECTS_FILE="${SUBJECTS_FILE:-data/test_cohort_warps_available.txt}"
DATABASE="${DATABASE:-data/database_finale_labels_corrette.csv}"
WARPS_ROOT="${WARPS_ROOT:-data/warps}"
ROIS_DIR="${ROIS_DIR:-outputs/rois}"
EXPECTED_HEALTHY="${EXPECTED_HEALTHY:-684}"
EXPECTED_UNHEALTHY="${EXPECTED_UNHEALTHY:-504}"
EXPORT_WORKERS="${EXPORT_WORKERS:-4}"
GRAPH_WORKERS="${GRAPH_WORKERS:-8}"
OUTPUT_ROOT="outputs"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"

echo "=== run-id=$RUN_ID  started $(date +"%d-%m-%Y %H:%M") ==="

echo "--- [1/6] Checking ROI completeness ---"
python scripts/validation/check_roi_counts.py --rois-dir "$ROIS_DIR"

echo "--- [2/6] Preflighting subject cohort ---"
python scripts/validation/validate_batch_inputs.py \
  --subjects "$SUBJECTS_FILE" \
  --database "$DATABASE" \
  --warps-root "$WARPS_ROOT" \
  --expected-healthy "$EXPECTED_HEALTHY" \
  --expected-unhealthy "$EXPECTED_UNHEALTHY"

echo "--- [3/6] Freezing run manifest ---"
FREEZE_MODE="create"
[[ -f "$RUN_MANIFEST" ]] && FREEZE_MODE="check"
python scripts/validation/freeze_pipeline.py "$FREEZE_MODE" \
  --manifest "$RUN_MANIFEST" \
  --run-id "$RUN_ID" \
  --atlas-manifest "$ROIS_DIR/atlas_manifest.json" \
  --subject-list "$SUBJECTS_FILE" \
  --config sim_formula=both \
  --config save_matrix=false \
  --config method=was,kl,meiq

echo "--- [4/6] Per-subject: log-Jacobian -> parcel vectors -> graphs ---"
subject_count=$(grep -c . "$SUBJECTS_FILE")
i=0
while IFS= read -r subject_id; do
  [[ -z "$subject_id" ]] && continue
  i=$((i + 1))
  echo "[$i/$subject_count] $subject_id"

  python scripts/registration/compute_log_jacobian.py \
    --warp-image "$WARPS_ROOT/$subject_id/Reg_/_SyN1Warp.nii.gz" \
    --subject-id "$subject_id"

  python scripts/parcellation/export_masked_jacobian_vectors.py \
    --jacobian "outputs/ants_registration/$subject_id/${subject_id}_to_template_logJacobian.nii.gz" \
    --rois-dir "$ROIS_DIR" \
    --num-workers "$EXPORT_WORKERS"

  python scripts/graph_building/wasserstein_distance_graph2.py \
    --subject-id "$subject_id" --method was,kl,meiq --sim-formula 1 --num-workers "$GRAPH_WORKERS"
  python scripts/graph_building/wasserstein_distance_graph2.py \
    --subject-id "$subject_id" --method was,kl,meiq --sim-formula 2 --num-workers "$GRAPH_WORKERS"
done < "$SUBJECTS_FILE"

echo "--- [5/6] Validating full cohort ---"
python scripts/validation/validate_cohort_outputs.py \
  --run-root "$OUTPUT_ROOT" \
  --subjects "$SUBJECTS_FILE" \
  --database "$DATABASE" \
  --expected-healthy "$EXPECTED_HEALTHY" \
  --expected-unhealthy "$EXPECTED_UNHEALTHY"

echo "--- [6/6] Running parcel-wise statistics ---"
FEATURES=(
  weighted_degree_expW
  weighted_degree_inv1pW
  weighted_degree_kl_expW
  weighted_degree_kl_inv1pW
  weighted_degree_meiq_expW
  weighted_degree_meiq_inv1pW
  direct_mean
  direct_median
)
for feature in "${FEATURES[@]}"; do
  python scripts/analysis/parcel_statistics.py \
    --run-root "$OUTPUT_ROOT" \
    --subjects "$SUBJECTS_FILE" \
    --database "$DATABASE" \
    --feature "$feature" \
    --output-dir "$OUTPUT_ROOT/statistics" \
    --expected-healthy "$EXPECTED_HEALTHY" \
    --expected-unhealthy "$EXPECTED_UNHEALTHY"
done

echo "=== run-id=$RUN_ID  finished $(date +"%d-%m-%Y %H:%M") ==="
