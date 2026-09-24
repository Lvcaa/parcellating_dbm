# Pipeline Issues Report

**Date:** 2026-08-21  
**Scope:** Parcel atlas, Jacobian extraction, similarity graphs, thresholded weighted degree, cohort selection, and statistical analysis.

## Executive summary

The weak subject-level separation is not caused by an upper-triangle indexing, graph mirroring, or weighted-degree arithmetic error. The core graph calculation was reproduced correctly with synthetic and real data.

The audit nevertheless identified one concrete upstream artifact defect and several important statistical and methodological limitations:

1. The current atlas contains disconnected parcels.
2. Most unadjusted discoveries disappear after covariate adjustment.
3. Subject-specific percentile thresholds deliberately remove graph-density differences.
4. Global all-to-all degree averaging is dominated by cortex and CSF and can dilute localized effects.
5. The thresholded analysis silently uses 121+121 subjects instead of the intended 150+150 test cohort.

The current biological maps should therefore be treated as exploratory rather than final.

## 1. The current parcel atlas contains disconnected parcels

**Severity:** High — confirmed artifact defect.

- `10,765 / 83,442` parcel masks are disconnected under 6-connectivity (`12.90%`).
- CSF: `5,915 / 18,940` disconnected (`31.23%`).
- Right cerebellar cortex: `1,276 / 4,724` disconnected (`27.01%`).
- Some parcels combine voxels separated by as much as 130 voxels.
- Seven nominal 15-voxel parcels consist entirely of isolated voxels.

The likely cause is the legacy parcelizer:

- Its neighbour check can never trigger: the expression is bounded by 3 but is tested against `> 3` in [`sub_parcels_equal_size.py`](../scripts/parcellation/sub_parcels_equal_size.py#L88).
- Its global centroid-slot assignment does not preserve spatial connectivity in [`sub_parcels_equal_size.py`](../scripts/parcellation/sub_parcels_equal_size.py#L130).

A corrected implementation already exists and explicitly rejects disconnected parcels in [`second_roi_test.py`](../scripts/parcellation/separate_cases/second_roi_test.py#L500), but the current ROI outputs were not generated with it.

The atlas validator did not catch this because it checks only the number of files per label, not connectivity, coverage, overlaps, or mask hashes: [`check_roi_counts.py`](../scripts/validation/check_roi_counts.py#L52).

### Impact

A single parcel can represent distant voxels, blurring anatomical localization and mixing unrelated Jacobian values. Disconnected parcels were not enriched among the significant findings, so this defect is serious but does not by itself explain all of the weak group separation.

## 2. Most unadjusted discoveries disappear after covariate adjustment

**Severity:** High — current inference is substantially confounded.

The thresholded analysis in [`t_tests.ipynb`](../scripts/analysis/t_tests.ipynb) uses unadjusted Welch tests. The included groups have the following characteristics:

| Characteristic | Healthy | Unhealthy |
|---|---:|---:|
| Mean age | 66.49 | 67.83 |
| Mean education | 16.03 | 15.09 |
| Female | 73 | 73 |
| Male | 48 | 48 |

The education difference has `p = 0.0039`.

Using the repository's existing age/sex/education-adjusted GLM from [`parcel_statistics.py`](../scripts/analysis/parcel_statistics.py#L140) changes the number of discoveries dramatically:

| Percentile | Welch/BH discoveries | Adjusted GLM/BH discoveries |
|---:|---:|---:|
| P10 | 235 | 8 |
| P20 | 232 | 9 |
| P40 | 213 | 27 |
| P80 | 204 | 33 |

Every adjusted discovery is a subset of the Welch discoveries.

### Impact

The current maps containing 204–235 significant parcels should not be interpreted as diagnosis-specific effects. The adjusted 8–33 parcels are the more defensible starting point.

## 3. Subject-specific percentiles remove density differences by design

**Severity:** Medium — consequential methodological choice, not a coding bug.

Each subject calculates its own threshold in [`thresholded_similarity_graph.py`](../scripts/graph_building/thresholded_similarity_graph.py#L190). This forces nearly identical graph densities:

| Percentile threshold | Edges retained |
|---:|---:|
| P10 | 90% |
| P20 | 80% |
| P40 | 60% |
| P80 | 20% |

The absolute thresholds differ slightly between groups:

| Percentile | Healthy threshold | Unhealthy threshold |
|---:|---:|---:|
| P10 | 0.3954 | 0.3889 |
| P20 | 0.4924 | 0.4868 |
| P40 | 0.6320 | 0.6277 |
| P80 | 0.8592 | 0.8577 |

### Impact

Potential group differences in overall graph density are normalized away. Fixed-density thresholding is appropriate if topology at equal density is the intended measurement. If overall similarity or density is biologically meaningful, use a common threshold learned from training data only, or analyse the subject-specific threshold itself as a feature.

## 4. The graph construction strongly dilutes localized effects

**Severity:** Medium — the feature may not match the scientific target.

Every parcel is compared with all other 83,441 parcels, irrespective of anatomy or spatial proximity. Atlas composition is highly uneven:

| Label group | Percentage of graph nodes |
|---|---:|
| Left cortex | 27.91% |
| Right cortex | 27.62% |
| CSF | 22.70% |

These three labels represent `78.23%` of all graph nodes. The stated grey-matter label set also includes CSF and ventricles in [`const.py`](../scripts/const.py#L68).

Consequently, weighted degree primarily measures similarity to a global cortex/CSF-dominated distribution. Small structures contribute very little to another parcel's degree.

The graph is also invariant to adding the same constant to every parcel distribution within a subject. Numerically, shifting every parcel log-Jacobian by `+3.25` changed the graph only by floating-point noise. Subject-wide absolute deformation shifts are therefore removed by construction.

Comparison with direct features:

| Feature | FDR discoveries | Between/within subject-distance ratio |
|---|---:|---:|
| Direct parcel mean | 775 | 1.0020 |
| Unthresholded graph degree | 307 | 1.0024 |

Weak global separation is also present in the direct data, so it is not created entirely by the graph. However, the graph transformation discards some absolute DBM information.

## 5. The test cohort is silently incomplete

**Severity:** Medium — reproducibility and selection risk.

[`splits.json`](../data/splits.json) specifies:

- 150 Healthy test subjects.
- 150 Unhealthy test subjects.

The thresholded run contains:

- 121 Healthy subjects.
- 121 Unhealthy subjects.

The notebook silently intersects the split with the available output folders and only requires at least two subjects. Exactly 29 subjects from each group are omitted.

The resulting 121+121 cohort remains sex-balanced, but it is not the complete intended test cohort. The analysis should use a frozen subject manifest and require the exact same subjects at every percentile.

## 6. Jacobian generation needs stronger provenance

**Severity:** Medium/low — missing provenance and QC, with no evidence that current Jacobians are wrong.

The script defaults to non-geometric Jacobians in [`compute_log_jacobian.py`](../scripts/registration/compute_log_jacobian.py#L84).

ANTs currently documents the geometric command (`... 1 1`) for morphometry and notes that older non-geometric calculations could be affected by an orientation bug:

- [ANTs Jacobian documentation](https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images%2C-pointsets-and-Jacobians)
- [ANTs 2.6.1 release notes](https://github.com/ANTsX/ANTs/releases/tag/v2.6.1)

The current outputs were created with ANTs 2.6.2 after that fix, and the warp/Jacobian grid is correctly aligned with MNI. No evidence was found that the current Jacobians are incorrect.

However:

- Completion metadata does not record the ANTs version.
- Jacobian QC checks only dimensionality and finite values.
- Expected shape, affine, deformation folding, and robust outliers are not validated.

## Confirmed-correct components

The audit found no evidence of:

- Healthy/Unhealthy label mixing.
- Train/test overlap.
- Duplicate subject outputs.
- Identical weighted-degree files between subjects.
- Parcel-order mismatch.
- Wrong voxel extraction.
- Incorrect upper-triangle slicing.
- Incorrect mirroring.
- Self-loops entering weighted degree.
- Incorrect `N−1` normalization.
- Float32 precision causing meaningful errors.

Independent synthetic calculations matched the threshold implementation exactly. Independently recomputed rows for `sub-0025` matched the stored P10/P20/P40/P80 weighted degrees with maximum absolute error `0.0`.

All 1,188 vector outputs have unique source-Jacobian hashes. Five complete subject vectors were checked voxel-for-voxel against their Jacobian images and matched exactly.

## Additional repository problem

**Severity:** Medium — regression protection is currently unavailable.

The test suite fails before running because [`test_pipeline_smoke.py`](../tests/test_pipeline_smoke.py#L21) imports the removed `export_vectors` API. The vector exporter was renamed to `extract_vectors`, but the tests were not updated.

This did not corrupt the existing outputs, but the repository currently lacks a working regression gate.

## Overall diagnosis

The weak subject-level separation is not caused by an upper-triangle or weighted-degree coding error. The current results nevertheless cannot be considered definitive because of the defective legacy atlas, covariate sensitivity, thresholding choice, graph composition, and incomplete test cohort.

## Recommended actions

1. Regenerate the atlas with the corrected connected parcelizer.
2. Cryptographically validate and freeze the actual parcel masks.
3. Run sensitivity analyses excluding CSF and ventricular labels.
4. Decide explicitly between fixed-density and common-threshold graphs.
5. Use the covariate-adjusted GLM as the primary inference.
6. Require a frozen subject manifest and account for testing four percentiles.
7. Create FDR-masked NIfTI maps separately from maps containing all effect sizes.
8. Update and restore the smoke-test suite before further pipeline changes.

