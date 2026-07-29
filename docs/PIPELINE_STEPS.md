# Pipeline Steps & Assertions

Ordered checklist for the full DBM pipeline: existing warps → parcel graphs →
validated statistics. Each step names its assertion gate. Status tags:

- `[have]` — script exists and is wired in.
- `[build]` — does not exist yet, needs to be written.

Registration is explicitly **not** part of this pipeline: `data/warps/`
already holds validated forward warps from a separate, trusted process, so
the pipeline starts from the log-Jacobian computation. Jacobian sanity
checking (shape/affine/plausibility) is deliberately **not** implemented
either, for the same reason — the warps are trusted.

The whole sequence below is implemented end to end in
[`scripts/pipelines/run_full_pipeline.sh`](../scripts/pipelines/run_full_pipeline.sh).

---

1. **Check ROI completeness** `[have]`
   `scripts/validation/check_roi_counts.py`
   Simple sanity check, not a cryptographic atlas freeze: confirms every
   label in `KEEP_LABELS` has a folder under `outputs/rois/` and that its
   parcel count matches `ceil(voxel_count / parcel_size)` computed straight
   from the segmentation. Writes the per-label pass/fail table as
   `outputs/rois/atlas_manifest.json` — this *is* the atlas contract now.
   Run this before anything else — step 6 refuses to run unless every label
   in this manifest is `"ok"`.

2. **Preflight the subject cohort** `[have]`
   `scripts/validation/validate_batch_inputs.py`
   Confirms every subject has a warp under `data/warps/`, group balance is
   684 Healthy / 504 Unhealthy (regenerated against current `data/warps/`
   contents via `scripts/preprocessing/build_subject_manifest.py`), and
   every subject has a database label.

3. **Freeze the run manifest** `[have]`
   `scripts/validation/freeze_pipeline.py create` (or `check` on rerun)
   Hashes the atlas manifest, subject list, and config so the batch can't
   silently drift mid-run.

4. **Compute log-Jacobian per subject** `[have]`
   `scripts/registration/compute_log_jacobian.py --warp-image ...`
   Trimmed from the old `run_ants_jacobian.py`: no registration code path at
   all now, only `CreateJacobianDeterminantImage` over the existing warp.

5. **Export parcel-level Jacobian vectors** `[have]`
   `scripts/parcellation/export_masked_jacobian_vectors.py`
   Enforces the atlas contract from step 1; writes `direct_mean`/
   `direct_median` and a completion marker per subject.

6. **Build graphs** `[have]`
   `scripts/graph_building/wasserstein_distance_graph2.py --method was,kl,meiq`
   One script, three edge definitions, selected via `--method` (comma-separated
   subset of `was`/`kl`/`meiq`). All three share the same `--sim-formula`
   distance→similarity transform and the same dense-only, `--save-matrix`
   default-false contract. `was` keeps its legacy output folder
   (`outputs/wasserstein_graphs_*`); `kl` and `meiq` write to
   `outputs/kl_graphs_*` and `outputs/median_iqr_graphs_*`.

7. **Validate full cohort outputs** `[have]`
   `scripts/validation/validate_cohort_outputs.py`
   Extended to cover all 6 graph folders (`was`/`kl`/`meiq` × 2 similarity
   formulas), checking every subject's `method`, `sim_formula`, parcel
   count/order hash, and atlas hash agree. This doubles as the cross-
   edge-type consistency check — no separate script needed, since
   per-subject internal correctness is already guaranteed by each stage's
   own completion marker; what's left to check is only cross-subject/
   cross-edge-type agreement, which is inherently a whole-cohort question.

8. **Run parcel-wise statistics per feature** `[have]`
   `scripts/analysis/parcel_statistics.py`
   Welch t-test, Hedges' g, CI, covariate-adjusted GLM, leave-one-out
   sensitivity. `FEATURE_LOCATIONS` now has all 6 weighted-degree features
   plus the 2 direct summaries. Expected group counts are a CLI arg now
   (`--expected-healthy`/`--expected-unhealthy`), not hardcoded.

9. **Cross-edge-definition agreement report** `[build]`
   Compare significant parcels across all 3 edge definitions (`was`/`kl`/
   `meiq`). Today only the 2 Wasserstein transforms are compared, ad hoc,
   inside `t_tests.ipynb`. CLAUDE.md non-negotiable: never treat one edge
   definition as canonical before this cross-definition validation exists.
   Deliberately out of the v1 orchestrated script — this is a stats-layer
   output, not a pipeline gate.

10. **Threshold/sparsification sensitivity pass** `[build]`
    Re-run steps 6–9 under 2–3 threshold variants once the dense pipeline
    is validated end to end. Deliberately last, per `ProjectDescription.md`.

## Test coverage (parallel track, add alongside the step that introduces the logic)

11. Tests for `check_roi_counts.py` / `validate_batch_inputs.py` /
    `validate_cohort_outputs.py` — add alongside steps 1 / 2 / 7. `[build]`

12. Tests for `parcel_statistics.py` math (Welch / Hedges' g / GLM /
    leave-one-out) — add alongside step 8. `[build]`

13. Permanent tests for the KL and median/IQR block-compute functions in
    `wasserstein_distance_graph2.py` — spot-checked numerically inline
    during implementation (identical parcels → similarity 1, distant
    parcels → low similarity, symmetric, bounded in `(0,1]`), but not yet
    captured as a real test alongside the existing Wasserstein test in
    `tests/test_pipeline_smoke.py`. `[build]`
