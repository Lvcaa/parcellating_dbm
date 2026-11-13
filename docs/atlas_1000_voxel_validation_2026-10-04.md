# Validation of the 1,000-parcel voxel-built atlas

Date: 2026-10-04  
Atlas: `outputs/atlases/atlas-1000-nocsf-from-voxels/rois`  
Atlas manifest SHA-256: `0c4c7d2d95e4d899585fcca09b857fedfa7e6f3ac344b0720a6a0a49ebac52ff`

## Decision

**The disconnected-parcel defect in `REPORT_ISSUES.md` is absent.** All 1,000
written masks are individually 6-connected. The atlas has exact coverage of
all 25 selected non-CSF anatomical labels, with no overlaps or voxels assigned
to a different label. Its parcel counts are tightly balanced *within each
connected source component*. The atlas does contain tiny parcels inherited
from tiny disconnected source components, and some large parcels are elongated.
It should not be described as uniformly compact or uniformly 1,000 voxels per
parcel.

## Independent checks on the written NIfTI files

- Confirmed the schema-2 atlas manifest, all 25 label manifests, all 1,000
  parcel files, and all recorded SHA-256 hashes.
- Confirmed every mask has the segmentation's shape and affine, contains only
  binary values, and is nonempty.
- Counted 3D components in each mask using face adjacency (6-connectivity):
  **0 / 1,000 disconnected**.
- Confirmed no masks overlap and the union for each label equals precisely the
  corresponding label in `MNI152_T1_1mm_seg.nii.gz`. All 967,363 selected
  voxels are covered. CSF label 24 is excluded.
- Confirmed the written voxel count of every parcel matches its manifest.

The earlier defect report found 10,765 / 83,442 disconnected masks in a legacy
atlas, including parcel voxels as far as 130 voxels apart. Those counts refer
to that earlier atlas and do not apply to this one.

## Parcel sizes

| Measure | Result |
|---|---:|
| Parcels | 1,000 |
| Minimum / median / maximum | 1 / 1,040 / 1,043 voxels |
| 25th / 75th percentile | 1,038 / 1,041 voxels |
| Parcels between 900 and 1,050 voxels | 908 |
| Parcels at least 500 voxels | 940 |
| Within 1 voxel of assigned component target | 925 |
| Within 2 voxels of assigned component target | 1,000 |
| Single-voxel parcels | 46 |
| Parcels below 15 voxels | 56 |

The source segmentation itself has exactly 46 one-voxel connected components
and 56 components below 15 voxels among the selected labels. Keeping every
voxel, respecting anatomical labels, and requiring connected parcels forces
those small components to remain separate parcels.

## Spatial extent and coherence

All parcels form one continuous 3D piece, but connectivity does not imply a
compact shape. At 1 mm voxel spacing, the median parcel bounding-box diagonal
is 47.6 mm, the 95th percentile is 60.9 mm, and the largest is 90.5 mm. The
longest parcel, `3/roi_0257.nii.gz`, has a measured maximum voxel-to-voxel
Euclidean distance of **85.6 mm**. It is connected but elongated. The longest
ventricle example, `4/roi_0002.nii.gz`, measures 71.0 mm end to end.

The bounding-box diagonal is an upper bound on any pairwise voxel distance.
Only one parcel has a bound above 80 mm, and its exact diameter is 85.6 mm;
all others have bounds below 76 mm. Thus no parcel in this atlas repeats the
reported 130-voxel pairwise span. Some parcels still cover long anatomical
strips, which may dilute localized effects even though their voxels are
connected. Orthogonal projections of the longest parcels and a comparison
parcel are in [the spatial QC figure](figures/atlas_1000_voxel_spatial_qc.png).

## Interpretation

This atlas passes the report's critical connectivity, overlap, and coverage
checks. It meets an approximately equal-size criterion within each connected
component, with unavoidable tiny exceptions. The spatial compactness question
remains a design choice: if the analysis requires a maximum physical parcel
diameter, that threshold should be set explicitly and the atlas regenerated
or filtered accordingly. No atlas files were changed during this validation.
