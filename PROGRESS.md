# Project Progress

**Status**: In progress
**Last Updated**: April 5, 2026

## What We Have Done

We have completed a substantial part of the left-brain ROI parcellation workflow.

So far, the project has:

- Taken large anatomical ROI labels from the reference segmentation.
- Split those labels into smaller sub-parcels of roughly equal voxel size.
- Written the resulting parcels as individual binary NIfTI masks.
- Preserved the spatial reference of the original template image.
- Reached the point where we now have many parcel files that represent subdivisions of larger left-brain regions.

This is an important milestone because the original large ROIs are no longer the main working unit. The new working units are the parcels themselves.

## What The Current Output Represents

At this stage, the output is a collection of separate parcel masks.

This is useful for generating and inspecting parcels, but it is not yet the most convenient format for downstream analysis. In particular, it is still awkward to compute relationships among regions when the data is spread across many independent files.

## Next Step: Build Parcel-Level Jacobian Features

Now that we have:

- a first trial image with the Jacobian determinant already computed
- ROI parcel masks for the subject

the next major step is to convert these into an **analysis-ready parcel feature table**.

This means:

- For each ROI parcel, extract all voxel values from the Jacobian determinant image.
- Save the raw voxel values for that parcel.
- Compute and save a normalized histogram of the parcel Jacobian values.
- Compute and save a central anomaly value such as the mean or median.
- Record parcel size as voxel count.
- Optionally compute one or two spatial descriptors for the abnormal voxels.
- Write everything into one analysis-ready file with one row per parcel.

Suggested columns include:

- Parcel ID
- Parent ROI / label
- Raw voxel values reference or serialized values
- Histogram bin values
- Mean or median Jacobian
- Anomaly score
- Voxel count
- Optional spatial descriptors

## Why This Step Is Needed

The parcel masks define where each parcel is, but they do not yet provide the
parcel-level morphometric features needed for anomaly detection or graph
construction.

This feature-extraction step is needed because:

- Each parcel must be converted from a binary mask into a quantitative Jacobian profile.
- Distribution-based node features require the full voxel-value distribution, not only a mean.
- Downstream similarity measures such as Jensen-Shannon divergence need a normalized histogram or density estimate for every parcel.
- A single table with one row per parcel makes later graph construction and quality control much easier.

Without this step, we still only have parcel geometry, not parcel-level
morphometric information.

## After Feature Extraction

Once the parcel-level Jacobian feature table exists, the project can move to parcel-level relationship analysis.

Examples of downstream goals include:

- Building parcel-by-parcel similarity matrices.
- Comparing morphometric similarity between parcels using histogram-based methods.
- Applying anomaly weighting to emphasize strongly abnormal parcels.
- Sparsifying the subject graph and optionally running community detection.

## Summary

Current phase:

- Large ROIs have been subdivided into many smaller parcels.
- A first trial Jacobian determinant image is available.

Next phase:

- Extract Jacobian voxel distributions and summary features for each parcel, then assemble them into a single analysis-ready parcel table.
