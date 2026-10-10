# 1,000-parcel atlas build log

Date: 2026-09-29

## Request and decision

The existing validated atlas has 83,442 parcels. We wanted a separate atlas
with 1,000 nodes for network-based statistics (NBS). In the reference
segmentation, CSF (label 24) occupies 1,562 separate 6-connected pieces.
Across all 26 retained labels there are 1,645 pieces, so a 1,000-parcel atlas
cannot cover every label while keeping each parcel connected and inside its
anatomical label. We chose to exclude label 24 from this atlas. The other 25
labels, including the separately named ventricle labels 4, 5, 43, and 44,
remain included. Their 967,363 voxels occupy 83 connected pieces.

## Inputs and method

- Segmentation: `data/reference/MNI152_T1_1mm_seg.nii.gz`, SHA-256 `af9725bd35d65ebaa10d8c735294bc5c4bffde57de2d75789e0ce0c1fd5aa194`.
- Validated source atlas: `outputs/atlases/atlas-8cf3dc8f/rois/atlas_manifest.json`, SHA-256 `8cf3dc8f3037fd8fab2a16751503b415a60732e4867e9600aedf5d65373fa4a0`.
- Builder: `scripts/parcellation/separate_cases/coarsen_connected_atlas.py`.
- The builder assigns an exact parcel budget across the 83 components, with at least one parcel per component. It then merges source parcels only when they share a voxel face and belong to the same anatomical label. The merge uses source parcel voxel counts as weights and a fixed random seed. The original atlas is unchanged.

Rebuild with:

```bash
python scripts/parcellation/separate_cases/coarsen_connected_atlas.py --dry-run
python scripts/parcellation/separate_cases/coarsen_connected_atlas.py --workers 8
```

The builder refuses to overwrite an existing atlas directory. Choose a new
`--output-atlases` directory or move the old output before rebuilding.

## Output

- Atlas: `outputs/atlases/atlas-1000-nocsf-830ed276/`
- Validated atlas manifest: `rois/atlas_manifest.json`, SHA-256 `91a43e48fab785396d5f9df965fd7e44b65095bed1c5e7b946d83a79e49247b7`.
- Label manifests and binary masks: `rois/<label>/label_manifest.json` and `rois/<label>/roi_XXXX.nii.gz`.
- Single labeled volume and node order: `parcel_volume_cache/parcel_volume.nii.gz` and `parcel_volume_cache/parcel_order.txt` (voxel label N corresponds to line N of the order file, using one-based NIfTI labels).
- Machine-readable build details and per-component allocation: `build_log.json`.

## Validation and quality notes

The output contains exactly 1,000 nonempty parcels across 25 labels and
967,363 voxels. Every parcel passed a 6-neighbor connectivity check before
writing. An independent pass loaded all 1,000 written masks, checked their
geometry and lack of overlap, reconstructed the integer volume, confirmed it
matches the cached volume and covers exactly the selected non-CSF voxels, and
verified every mask and label-manifest SHA-256 against the atlas manifest.
The existing parcel-vector exporter also accepted the new validated atlas
contract and produced the same 1,000-node order as the cached volume.

Parcel sizes range from 1 to 1,725 voxels (median 961). There are 46
single-voxel parcels and 56 parcels below 15 voxels because some retained
non-CSF labels also have tiny disconnected islands. These nodes may be noisy
for distribution-based similarity and should be considered explicitly in the
NBS analysis design. Excluding CSF does not exclude ventricle labels.

For NBS, use the same `parcel_order.txt` across subjects and retain the
subject-level edge weights. Weighted-degree vectors alone do not contain the
edges needed for an edgewise NBS test. The existing
`thresholded_similarity_graph.py --save-graph` writes subject-percentile
thresholded graphs; that thresholding is distinct from the NBS threshold
applied to group edge statistics.

## Later workflow change (2026-09-29)

The separate `coarsen_connected_atlas.py` builder was removed. The existing
`atlas-1000-nocsf-830ed276` artifact remains a coarsened atlas; this section
does not change its provenance. `second_parcellation_method.py` now accepts
`--exclude-label` and `--target-parcels` to build a **new, different** atlas
directly from segmentation voxels. For this image, excluding CSF leaves 83
connected components and permits exactly 1,000 parcels. Use a separate output:

```bash
python scripts/parcellation/separate_cases/second_parcellation_method.py \
  --exclude-label 24 --target-parcels 1000 \
  --output-root outputs/atlases/atlas-1000-nocsf-from-voxels/rois
```

The full 1,000-parcel voxel build has not yet been run. The exact-budget
planner, a 10-parcel thalamus validation, and a three-parcel end-to-end
manifest/exporter check passed. The previous rebuild commands above record
the historical coarsening workflow and refer to a script that is no longer
present.

A larger validation of cerebellum label 8 (84 planned parcels) was stopped
after about three minutes of CPU time while the original voxel-balancing
algorithm was still running. No output was written. The full voxel build may
take substantially longer than the previous source-parcel merging build;
`--workers` parallelizes mask writing, not parcel construction.
