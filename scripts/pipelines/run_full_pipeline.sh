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
# All configuration below is hardcoded on purpose (not read from the
# environment): a stray `export METHODS=was` left over in a shell/screen
# session from a previous provisional run silently changed the config of
# a later run and cost real debugging time. To change a value, edit the
# literal below and re-run -- do not reintroduce environment overrides.
#
# METHODS=was is a provisional escape hatch for the KL edge definition's
# numerical-instability bug (symmetrized Gaussian KL blows up to the
# thousands at ~15-voxel parcel resolution, so exp(-KL) hard-underflows
# to 0.0 and trips the similarity sanity check -- see graph-building
# stage). Cohort validation with METHODS=was is explicitly marked
# non-definitive; re-run with was,kl,meiq once the KL similarity
# transform is fixed, before drawing any conclusions.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${1:-full_pipeline_$(date +%Y%m%d)}"
SUBJECTS_FILE="data/test_cohort_warps_available.txt"
DATABASE="data/database_finale_labels_corrette.csv"
WARPS_ROOT="data/warps"
ROIS_DIR="outputs/rois"
EXPECTED_HEALTHY=684
EXPECTED_UNHEALTHY=504
EXPORT_WORKERS=8
GRAPH_WORKERS=16
METHODS="was"  # provisional: kl is broken (see note above); flip to "was,kl,meiq" once fixed
OUTPUT_ROOT="outputs"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest_${RUN_ID}.json"

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
  --config method="$METHODS"

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
    --subject-id "$subject_id" --method "$METHODS" --sim-formula 1 --num-workers "$GRAPH_WORKERS"
  python scripts/graph_building/wasserstein_distance_graph2.py \
    --subject-id "$subject_id" --method "$METHODS" --sim-formula 2 --num-workers "$GRAPH_WORKERS"
done < "$SUBJECTS_FILE"

echo "--- [5/6] Validating cohort (methods: $METHODS) ---"
python scripts/validation/validate_cohort_outputs.py \
  --run-root "$OUTPUT_ROOT" \
  --subjects "$SUBJECTS_FILE" \
  --database "$DATABASE" \
  --expected-healthy "$EXPECTED_HEALTHY" \
  --expected-unhealthy "$EXPECTED_UNHEALTHY" \
  --methods "$METHODS" \
  --manifest "$RUN_MANIFEST"

echo "--- [6/6] Running parcel-wise statistics ---"
FEATURES=(direct_mean direct_median)
if [[ ",$METHODS," == *",was,"* ]]; then FEATURES+=(weighted_degree_expW weighted_degree_inv1pW); fi
if [[ ",$METHODS," == *",kl,"* ]]; then FEATURES+=(weighted_degree_kl_expW weighted_degree_kl_inv1pW); fi
if [[ ",$METHODS," == *",meiq,"* ]]; then FEATURES+=(weighted_degree_meiq_expW weighted_degree_meiq_inv1pW); fi
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
