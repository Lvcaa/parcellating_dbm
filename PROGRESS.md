# Project Progress

**Status**: In progress
**Last Updated**: May 7, 2026

## Current State

The project is now in the transition phase between parcel generation and
validation-ready anomaly analysis.

We already have the core ingredients needed for the final pipeline described
in `ProjectDescription.md`:

- ROI masks split into smaller parcel masks.
- Parcel outputs grouped by anatomical label.
- Jacobian determinant values extracted voxel-wise inside each parcel.
- A first feature extraction path for summary statistics and histograms.
- A first Wasserstein graph path for parcel-to-parcel similarity analysis.

The most important recent milestone is that we can now reliably export the
full Jacobian determinant vector for each parcel. This means the project has
the correct low-level representation for distribution-based parcel features.

## What Is Working Now

### 1. Parcel generation

The original anatomical ROIs have already been subdivided into smaller
sub-parcels and saved as binary NIfTI masks.

This gives us the structural units that will act as graph nodes in later
analysis.

### 2. Jacobian value extraction per parcel

The current script `scripts/parcellation/export_masked_jacobian_vectors.py`
loads the Jacobian image once, applies each parcel mask, and saves one `.npy`
vector per parcel.

This is the key feature representation for the current workflow because:

- each parcel keeps its full voxel-wise Jacobian distribution
- we do not collapse the parcel too early into only one summary value
- the exported vectors can be reused for multiple edge definitions

### 3. Initial parcel-level feature extraction

The script `scripts/jacobian_work/extract_parcel_jacobian_features.py`
already produces:

- voxel count
- mean Jacobian
- median Jacobian
- anomaly score relative to a baseline
- normalized histogram bins
- raw voxel-value storage

This is still useful as the structured feature-table branch of the project.

### 4. Initial Wasserstein graph construction

The graph workflow can already build dense parcel-by-parcel Wasserstein
similarity matrices from exported parcel vectors.

This supports exploratory pattern analysis and graph-based validation, but it
should be interpreted carefully:

- it measures parcel-to-parcel morphometric similarity
- it does not yet directly produce a final anomaly detector
- anomaly detection will come from cross-subject validation after graph
  features are computed

### 5. Multi-label exploratory analysis

The inspection workflow now supports merging multiple labels into one combined
Wasserstein matrix.

This is useful for exploratory questions such as comparing:

- left hippocampus
- left thalamus
- left putamen

inside one shared subcortical similarity matrix to see whether they show
distinct morphometric patterns.

## What The Current Pipeline Means Scientifically

The current implementation is now correctly set up for the first half of the
final objective:

- extract warp-derived Jacobian information from parcels
- represent each parcel by a deformation distribution
- compare parcels using distribution-aware metrics

However, the current graph matrix itself is not yet the final anomaly result.

Right now:

- Jacobian extraction is the feature stage
- parcel-to-parcel similarity is the graph construction stage
- anomaly detection will be the validation stage

That final stage still requires healthy-vs-diseased comparison across
subjects.

## Active Direction

The project remains aligned with the refined plan in `ProjectDescription.md`.

The active analysis direction is:

- build subject-level dense parcel graphs
- use three edge definitions:
  - Wasserstein-based similarity
  - KL divergence using Gaussian summaries
  - robust median + IQR comparison
- compute weighted degree for every parcel
- compare parcel-wise graph metrics between healthy and diseased subjects
- identify parcels that show stable disease-related deviations

The working hypothesis has been refined after the first subject-level
comparisons.

Earlier, the expectation was that atrophic subjects would show a broad global
decrease in weighted degree, because parcels affected by atrophy should become
less morphometrically similar to the rest of the brain. The recent comparisons
suggest that this is too simple.

The updated hypothesis is:

- neurodegeneration and atrophy produce localized, label-specific disruptions
  in parcel-to-parcel morphometric similarity
- these disruptions are visible as weighted-degree drops and altered adjacency
  submatrix structure in affected anatomical regions
- the strongest evidence should come from reproducible regional patterns
  across subjects, not from a single global mean-degree decrease

This reframing fits the current observations:

- OAS30145 showed strong degree drops in subcortical regions such as the right
  accumbens, right putamen, left accumbens, and right pallidum, while some
  ventricular labels showed positive mean changes
- OAS30999 showed a smaller global mean-degree decrease than expected from its
  apparent atrophy severity, but clear drops in ventricular and caudate-related
  regions
- the top-degree-drop submatrix comparison showed that the same high-drop
  parcels form a more coherent high-similarity structure in the healthy graph
  than in the atrophy graph

The interpretation is therefore that the Wasserstein graph is not yet a simple
atrophy severity scalar. It is better understood as a way to detect how local
deformation-distribution patterns reorganize the morphometric similarity
network. Validation should focus on whether the same labels and parcel
neighborhoods show stable disruptions across multiple atrophic subjects and
remain distinct from healthy controls.

## Immediate Next Steps

### 1. Finalize the parcel feature schema

Create one stable parcel-level feature format that supports all planned edge
definitions.

The canonical per-parcel output should include:

- parcel ID
- parent label
- voxel count
- raw Jacobian vector or stable raw-values reference
- histogram representation
- mean
- median
- IQR
- Gaussian summary terms needed for KL-based comparison

This is the handoff format that should feed the graph-building stage.

### 2. Separate exploratory graph analysis from final anomaly detection

Keep the current Wasserstein graph workflow, but treat it as:

- exploratory morphometric similarity analysis
- an intermediate graph-construction step

Do not treat the heatmap alone as the final anomaly result.

### 3. Build subject-level graph outputs consistently

For each subject:

- build one dense graph for Wasserstein
- build one dense graph for KL Gaussian
- build one dense graph for median + IQR
- compute weighted degree per parcel
- save those results in a consistent subject-level format

### 4. Add the real validation step

For each parcel across subjects:

- split subjects into healthy and diseased groups
- compare weighted degree values
- run statistical testing
- compute effect sizes
- rank parcels by reproducible group difference

This is the step that turns graph features into anomaly evidence.

### 5. Run focused exploratory tests before scaling up

Before moving to the full approximately 90k-parcel setting, run targeted tests
on a smaller anatomical subset such as:

- left hippocampus
- left thalamus
- left putamen

This should be used to verify:

- parcel extraction quality
- graph construction behavior
- whether different subcortical structures show distinct similarity patterns
- whether the metrics are numerically stable

### 6. Compare similarity transforms for Wasserstein

The current graph code uses `exp(-wasserstein_distance)`.

It is worth testing at least one alternative transform as a sensitivity check,
for example:

- `1 / (1 + wasserstein_distance)`

The goal is to understand whether the chosen transform compresses differences
too strongly or affects weighted-degree interpretability.

### 7. Keep thresholding as a later sensitivity analysis

The dense graph should remain the primary first-pass analysis.

Only after the dense version is validated should we test:

- 2-3 threshold strategies
- stability of parcel-wise weighted degree results under sparsification

## Practical Priority Order

The recommended order of work from here is:

1. Lock the parcel-level feature schema.
2. Add any missing summary features needed for KL and median/IQR edges.
3. Standardize subject-level graph output for all three edge definitions.
4. Compute weighted degree systematically for every subject.
5. Run healthy-vs-diseased parcel-wise validation.
6. Repeat across multiple resolutions and later across threshold settings.

## Summary

What is complete:

- parcel masks exist
- Jacobian determinant values can be extracted for each parcel
- parcel vectors can be compared with Wasserstein distance
- merged multi-label exploratory matrices are possible

What is not complete yet:

- final validation-ready feature schema
- all three graph edge definitions in a unified pipeline
- subject-level healthy-vs-diseased validation
- final parcel-wise anomaly ranking

The project is therefore in a strong intermediate state: the representation of
warp-derived Jacobian information is now working, and the next phase is to
turn that representation into statistically validated anomaly detection.
