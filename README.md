# Parcellating DBM

Research code for parcel-level deformation-based morphometry (DBM).

The repository turns nonlinear-registration Jacobian images into subject-level
brain graphs:

1. Split anatomical labels into small, spatially contiguous parcels.
2. Register a subject T1 image to an MNI template with ANTs.
3. Extract the Jacobian values inside every parcel.
4. Compare parcel distributions with Wasserstein distance.
5. Build dense similarity graphs and compute parcel-level weighted degree.
6. Compare graph features across subjects to look for localized morphometric
   disruptions.

The project is actively evolving. The current implemented graph path focuses on
Wasserstein similarity. The planned analysis also includes Gaussian KL
divergence and robust median/IQR comparisons; see
[`ProjectDescription.md`](ProjectDescription.md) and [`PROGRESS.md`](PROGRESS.md)
for the research plan and current status.

## Repository Structure

```text
.
|-- data/
|   `-- reference/                 # MNI template, segmentation, mask, example T1 images
|-- docs/
|   |-- label_lookup.csv           # Anatomical labels and globally assigned parcel IDs
|   |-- high_memory_labels.txt     # Labels intended for a separate high-RAM workflow
|   `-- TO_DO_CHECKS.md            # Focused validation notes
|-- outputs/                       # Generated files; ignored by Git
|-- scripts/
|   |-- preprocessing/             # Label inspection and OASIS-3 subject selection
|   |-- parcellation/              # Parcel creation, aggregation, naming, vector export
|   |-- registration/              # ANTs registration and Jacobian generation
|   |-- graph_building/            # Wasserstein graphs, inspection, subject comparison
|   |-- benchmarks/                # Synthetic scalability experiments
|   `-- run_jacobian_wasserstein_pipeline.sh
|-- containers/
|   |-- local/                     # Local Docker recipe
|   `-- cluster/                   # Docker and Apptainer/Singularity recipes
|-- ProjectDescription.md          # Refined scientific objective
|-- PROGRESS.md                    # Current implementation status and next steps
`-- requirements.txt               # Base Python dependencies
```

## Main Workflow

### 1. Inspect the reference segmentation

[`scripts/preprocessing/inspect_labels.py`](scripts/preprocessing/inspect_labels.py)
prints voxel counts for the selected SynthSeg labels and creates:

```text
data/reference/MNI152_keep_labels_mask.nii.gz
```

Run it with:

```bash
python scripts/preprocessing/inspect_labels.py
```

### 2. Generate parcel masks

[`scripts/parcellation/sub_parcels_equal_size.py`](scripts/parcellation/sub_parcels_equal_size.py)
splits one anatomical ROI into approximately equal-sized parcels. It uses
spatially constrained agglomerative clustering and writes binary NIfTI masks:

```text
outputs/rois/<label>/roi_XXXX.nii.gz
```

Example:

```bash
python scripts/parcellation/sub_parcels_equal_size.py \
  --roi-label 10 \
  --parcel-size 15 \
  --skip-neighbor-check
```

The algorithm can allocate a large distance matrix for high-volume labels.
Test smaller ROIs first and use the cluster recipes for memory-heavy runs.

Useful companion scripts:

- [`aggregate_parcels.py`](scripts/parcellation/aggregate_parcels.py) adds parcel
  masks back together to check whether they cover the expected ROI.
- [`assign_unique_labels.py`](scripts/parcellation/assign_unique_labels.py)
  renames parcel masks with globally unique sequential IDs and updates
  `docs/label_lookup.csv`. Use `--dry-run` before renaming files.

### 3. Register one subject and create a log-Jacobian image

[`scripts/registration/run_ants_jacobian.py`](scripts/registration/run_ants_jacobian.py)
runs nonlinear registration with `antsRegistrationSyNQuick.sh`, then calls
`CreateJacobianDeterminantImage`.

Example:

```bash
python scripts/registration/run_ants_jacobian.py \
  --fixed-image data/reference/MNI152_T1_1mm.nii.gz \
  --moving-image data/reference/sub-0091_ses-V01_T1w.nii.gz \
  --subject-id sub-0091
```

The resulting log-Jacobian image is written under:

```text
outputs/ants_registration/<subject-id>/
```

### 4. Export parcel-level Jacobian vectors

[`scripts/parcellation/export_masked_jacobian_vectors.py`](scripts/parcellation/export_masked_jacobian_vectors.py)
loads a Jacobian image once, applies every parcel mask, and saves each parcel's
voxel values as a one-dimensional NumPy array:

```text
outputs/jacobian_parcel_vectors/<subject-id>/label_<label>/roi_XXXX.npy
```

Example:

```bash
python scripts/parcellation/export_masked_jacobian_vectors.py \
  --jacobian outputs/ants_registration/sub-0091/sub-0091_to_template_logJacobian.nii.gz \
  --num-workers 8
```

Use `--input-dir` to batch-process a directory of Jacobian images, or `--label`
and `--n-parcels` for a smaller test run.

### 5. Build dense Wasserstein graphs

[`scripts/graph_building/wasserstein_distance_graph2.py`](scripts/graph_building/wasserstein_distance_graph2.py)
is the active optimized graph builder. It projects sorted parcel vectors onto a
common quantile grid, computes a dense pairwise Wasserstein matrix in blocks,
and saves the graph as memory-mapped arrays.

Choose the distance-to-similarity transform explicitly:

```bash
python scripts/graph_building/wasserstein_distance_graph2.py \
  --subject-id sub-0091 \
  --sim-formula 1 \
  --num-workers 8

python scripts/graph_building/wasserstein_distance_graph2.py \
  --subject-id sub-0091 \
  --sim-formula 2 \
  --num-workers 8
```

Formula `1` uses `exp(-W)`. Formula `2` uses `1 / (1 + W)`.

Outputs are saved under:

```text
outputs/wasserstein_graphs_expW/<subject-id>/
outputs/wasserstein_graphs_inv1pW/<subject-id>/
```

Each graph folder contains:

```text
adjacency_matrix.dat    # Dense float32 memory-mapped similarity matrix
weighted_degree.dat     # Float64 weighted-degree vector
metadata.npy            # Number of parcels
parcel_order.txt        # Row/column order for interpreting graph arrays
```

### 6. Compare subjects

[`scripts/graph_building/compare_subject_wasserstein_graphs.py`](scripts/graph_building/compare_subject_wasserstein_graphs.py)
compares weighted-degree vectors between two subjects and produces summaries
and focused plots.

Example:

```bash
python scripts/graph_building/compare_subject_wasserstein_graphs.py \
  --healthy-subject sub-0006 \
  --atrophy-subject sub-OAS30999 \
  --sim-formula expW
```

### One-command subject pipeline

Once parcel masks exist, the main shell wrapper runs registration, vector
export, and both Wasserstein graph variants:

```bash
bash scripts/run_jacobian_wasserstein_pipeline.sh \
  data/reference/sub-0091_ses-V01_T1w.nii.gz
```

Existing log-Jacobian images and parcel vectors are reused automatically.

## Script Guide

### Preprocessing

- `inspect_labels.py`: prints segmentation-label counts and writes the retained
  label mask.
- `extract_unhealthy_images_OAS3.py`: ranks OASIS-3 sessions with a composite
  atrophy score and writes a selected session list. Its dataset CSV path is
  machine-specific and should be adjusted before use.

### Parcellation

- `sub_parcels_equal_size.py`: primary ROI splitting implementation.
- `aggregate_parcels.py`: recombines parcel masks for coverage checks.
- `assign_unique_labels.py`: assigns globally unique parcel filenames and
  creates the label lookup CSV.
- `export_masked_jacobian_vectors.py`: extracts reusable per-parcel Jacobian
  vectors.
- `separate_cases/second_roi_test.py`: experimental connected-region-growing
  implementation for difficult ROIs.
- `separate_cases/run_allowed_parcellations.py` and
  `separate_cases/run_high_ram_labels.py`: intended batch runners for standard
  and high-memory labels.

### Graph Building

- `wasserstein_distance_graph2.py`: active optimized dense Wasserstein builder.
- `compare_subject_wasserstein_graphs.py`: compares two graph outputs and
  creates label-level summaries and plots.
- `wasserstein_distance_graph.py`: earlier, slower full-vector implementation
  retained for reference.
- `inspect_wasserstein_graph.py`: exploratory graph sanity checks and
  visualization, including small merged-label experiments.
- `smoke_test.py`: small-scale Wasserstein graph prototype.
- `run_two_subjects.sh`: early helper script retained for reference.

### Benchmarks

The scripts under `scripts/benchmarks/` generate synthetic matrices or graphs
to test memory use and runtime before scaling up:

- `phase1_feasibility.py`: dense Pearson-correlation matrix followed by
  thresholding and Leiden community detection.
- `phase1_feasability_mini_k_means.py`: worst-case dense adjacency benchmark
  using MiniBatch K-Means.
- `fake_matrix_stress_test.py`: dense synthetic weighted Leiden benchmark.
- `fake_matrix_stress_test_sparse.py`: sparse synthetic weighted Leiden
  benchmark.

These benchmarks are exploratory utilities, not the current scientific
validation endpoint.

## Setup

Create a Python environment and install the base requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Different scripts also import `nibabel`, `pandas`, `scikit-learn`, `joblib`,
`matplotlib`, and `seaborn`. Install the packages needed by the workflow you
are running:

```bash
pip install nibabel pandas scikit-learn joblib matplotlib seaborn
```

Registration additionally requires ANTs binaries on `PATH`:

```text
antsRegistrationSyNQuick.sh
CreateJacobianDeterminantImage
```

## Known Caveats

- The project is in progress. Wasserstein graph construction is implemented;
  the planned KL-divergence, median/IQR, and group-level statistical validation
  stages still need to be standardized.
- Dense graph files grow quadratically with the number of parcels. Check disk
  space and memory requirements before running full-resolution analyses.
- The scripts under `scripts/parcellation/separate_cases/` and some container
  entry points still reference earlier file locations. Treat them as
  work-in-progress helpers and review their paths before launching a batch job.
- `inspect_wasserstein_graph.py` was written around earlier graph outputs.
  Prefer `compare_subject_wasserstein_graphs.py` for the optimized graph
  builder's subject-level outputs.

Generated files belong under `outputs/`, which is intentionally excluded from
version control.
