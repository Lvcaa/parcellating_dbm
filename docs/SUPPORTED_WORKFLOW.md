# Supported workflow for this branch

This branch provides a research workflow from existing forward subject-to-MNI
warps to parcel log-Jacobian features and Wasserstein weighted degree. It does
not provide a validated diagnostic model or a complete registration workflow.
Run commands from the repository root with the same Python environment.

## Environment and external inputs

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

ANTs `CreateJacobianDeterminantImage` must be on `PATH` for real Jacobian
creation. Supply the MNI-space segmentation, existing forward warps, a subject
manifest with one unique `sub-XXXX` identifier per line, and the cohort CSV
separately. The CSV uses `OASISID`, `HStatus` (`Healthy` or `Unhealthy`),
`age at visit`, `GENDER`, and `EDUC`. Data and computed outputs are not part of
the supported source-only example.

The commands below illustrate a new isolated run. Review the atlas, cohort,
Jacobian convention, CPU allocation, and RAM estimate before running real data.
They have not been tested end to end on a real cohort as part of this PR.

## Connected atlas

The active builder is `second_parcellation_method.py`. For the 1,000-parcel
atlas excluding CSF, use a new, empty destination:

```bash
python scripts/parcellation/separate_cases/second_parcellation_method.py \
  --segmentation data/reference/MNI152_T1_1mm_seg.nii.gz \
  --exclude-label 24 --target-parcels 1000 --workers 1 \
  --output-root outputs/atlases/atlas-1000-nocsf/rois
```

This is a substantial computation even with one writer. `--validate-only`
still constructs parcels in memory; it is not a cheap preflight. Reuse an
existing reviewed atlas instead of regenerating it unnecessarily.

The builder writes a schema-2 `atlas_manifest.json` and per-label manifests.
Connectivity, coverage, and non-overlap are checked by the builder. Keep those
manifests with their masks. Do not run the legacy `check_roi_counts.py` over a
connected-atlas directory: it checks a different count rule and overwrites the
atlas manifest. Tiny disconnected components of the source segmentation
remain separate connected parcels. Connectivity does not guarantee compactness
or equal parcel sizes. See the dated atlas build and validation records for
measurements of the earlier 1,000-parcel artifact, not guarantees about a new run.

## Freeze the run

Set paths for the reviewed atlas and a new run. The subject list and database
must refer to the same cohort. Choose the cohort scientifically before running
analysis; do not silently intersect it with whichever outputs happen to exist.

```bash
ATLAS=outputs/atlases/atlas-1000-nocsf/rois
RUN=outputs/runs/connected_wasserstein_v1
SUBJECTS=data/test_cohort_warps_available.txt
DATABASE=data/database_finale_labels_corrette.csv
WARPS=data/warps
GEOMETRIC=false

python scripts/validation/freeze_pipeline.py create \
  --manifest "$RUN/run_manifest.json" --run-id connected_wasserstein_v1 \
  --atlas-manifest "$ATLAS/atlas_manifest.json" --subject-list "$SUBJECTS" \
  --file "$DATABASE" \
  --file scripts/registration/compute_log_jacobian.py \
  --file scripts/parcellation/export_masked_jacobian_vectors.py \
  --file scripts/graph_building/wasserstein_distance_graph2.py \
  --file scripts/pipeline_integrity.py \
  --file scripts/validation/freeze_pipeline.py \
  --file scripts/validation/validate_cohort_outputs.py \
  --file scripts/analysis/parcel_statistics.py \
  --config method=was --config sim_formula=both \
  --config jacobian_geometric="$GEOMETRIC"
```

Use `check` with the same arguments before resuming an existing run instead
of `create`. Files are frozen only when supplied with `--file`; this is not
a complete environment lock. Record the installed Python packages and ANTs
version separately. `GEOMETRIC=false` matches this branch's existing default;
it is an explicit methodological choice, not a claim that the convention is
best for every analysis. `compare_jacobian_modes.py` supports a separate review
of that choice before freezing it.

## Compute each subject

Run the following in the same shell after defining the variables above:

```bash
set -euo pipefail
while IFS= read -r subject || [ -n "$subject" ]; do
  [ -z "$subject" ] && continue
  python scripts/registration/compute_log_jacobian.py \
    --warp-image "$WARPS/$subject/Reg_/_SyN1Warp.nii.gz" \
    --subject-id "$subject" --jacobian-geometric "$GEOMETRIC" --threads 1 \
    --output-root "$RUN/ants_registration"
  python scripts/parcellation/export_masked_jacobian_vectors.py \
    --jacobian "$RUN/ants_registration/$subject/${subject}_to_template_logJacobian.nii.gz" \
    --rois-dir "$ATLAS" --output-dir "$RUN/jacobian_parcel_vectors" \
    --num-workers 1
  for formula in 1 2; do
    if [ "$formula" = 1 ]; then name=expW; else name=inv1pW; fi
    python scripts/graph_building/wasserstein_distance_graph2.py \
      --input-folder "$RUN/jacobian_parcel_vectors/$subject" \
      --output-folder "$RUN/wasserstein_graphs_$name/$subject" \
      --method was --sim-formula "$formula" --num-workers 1 --block-size 100
  done
done < "$SUBJECTS"
```

Parcel export writes one concatenated float32 vector file, int64 offsets,
explicit parcel order, float64 direct mean/median features, and completion
metadata. Weighted degree is the sum of off-diagonal similarities divided by
`N - 1`; the builder retains no dense adjacency matrix. Wasserstein-1 uses
15 midpoint quantiles and is approximate for general empirical distributions.
Formula 1 is `exp(-W)` and formula 2 is `1/(1+W)`.

At 1,000 parcels a float32 block of 100 rows is about 0.4 MB. This is only one
buffer: image loading, parcel features, temporary arrays, and Python add to RAM
use. Pairwise work still grows quadratically; estimate resources again before
using a much larger atlas or additional processes.

The new `wasserstein_distance.py` is a separate reusable in-memory API. The
committed command-line pipeline still uses `wasserstein_distance_graph2.py`;
migrating callers to the reusable API is outside this change.

## Validate the cohort before statistics

Replace the illustrative group counts below with the exact counts in the
reviewed subject manifest. Both Wasserstein formulas must be present.

```bash
HEALTHY_COUNT=137
UNHEALTHY_COUNT=130
python scripts/validation/validate_cohort_outputs.py \
  --run-root "$RUN" --subjects "$SUBJECTS" --database "$DATABASE" \
  --manifest "$RUN/run_manifest.json" --methods was \
  --expected-healthy "$HEALTHY_COUNT" --expected-unhealthy "$UNHEALTHY_COUNT"

for feature in direct_mean direct_median weighted_degree_expW weighted_degree_inv1pW; do
  python scripts/analysis/parcel_statistics.py \
    --run-root "$RUN" --subjects "$SUBJECTS" --database "$DATABASE" \
    --feature "$feature" --output-dir "$RUN/statistics" \
    --expected-healthy "$HEALTHY_COUNT" --expected-unhealthy "$UNHEALTHY_COUNT"
done
```

The validator checks expected subject folders, counts, completion metadata,
atlas/order agreement, feature sizes, and finite values. Passing these checks
establishes structural consistency, not biological validity. A Wasserstein-only
run is marked `definitive: false` by the existing validator. KL and median/IQR
are implemented but are not established alternative endpoints; exponential KL
can underflow and fail the similarity checks. Do not enable methods simply to
obtain a report marked definitive.

Statistics include Welch tests, per-feature BH-FDR, confidence intervals,
Hedges' g, an age/sex/education-adjusted linear model, and leave-one-out
sensitivity. Missing covariates reduce the adjusted-model cohort. BH correction
within a feature does not correct the entire collection of methods, thresholds,
and exploratory analyses. Treat biological conclusions as exploratory pending
a prespecified analysis and appropriate sensitivity checks.

## Other workflows and checks

`scripts/pipelines/run_full_pipeline.sh` is the legacy 15-voxel workflow with
hardcoded cohort counts, `outputs/rois`, and a count-based atlas check. It is
not the driver for the connected-atlas commands above. Legacy wrappers and
containers are retained for compatibility; container recipes have path fixes
but have not been rebuilt in this validation.

The thresholded graph and NBS exporters are additional research workflows.
NBS export retains parcels with at least 15 voxels and uses available test-split
warps, so its node set and cohort can differ from the unthresholded workflow.
It saves subject order for design construction. These outputs and MATLAB NBS
inference were not validated by the synthetic Python pipeline check.

Run the small checks without source data or ANTs:

```bash
python -m unittest discover -s tests -p test_pipeline_smoke.py -v
python -m unittest discover -s tests -p test_wasserstein_distance.py -v
python -m unittest discover -s tests -p test_connected_workflow.py -v
```

Tests cover synthetic arithmetic, connectivity, extraction, reusable Wasserstein
output, and a small connected-atlas/export/freeze/graph/cohort-validation chain.
They retain temporary fixtures for inspection. They do not demonstrate clinical
performance, real-data registration quality, or general approximation accuracy.
