# Project Progress

**Status**: In progress
**Last Updated**: April 15, 2026

## What We Have Done

We have completed the core ROI parcellation workflow and a first parcel-level Jacobian feature export.

So far, the project has:

- Taken large anatomical ROI labels from the reference segmentation.
- Split those labels into smaller sub-parcels of roughly equal voxel size.
- Written the resulting parcels as individual binary NIfTI masks.
- Preserved the spatial reference of the original template image.
- Aggregated parcel outputs for multiple anatomical labels.
- Generated a first Jacobian feature export in `outputs/jacobian_features/` with a parcel feature table, compressed raw voxel values, and histogram metadata.

This is an important milestone because the original large ROIs are no longer the only working unit. We now have both parcel geometry and an initial parcel-wise morphometric feature export.

## What The Current Output Represents

At this stage, the output is no longer just a collection of separate parcel masks.

The project currently includes:

- Parcel mask files for the subdivided ROIs.
- Aggregated parcellation volumes for multiple labels.
- A first parcel-level Jacobian feature table with voxel count, central Jacobian summaries, anomaly score, histogram bins, and raw-value references.

This is a strong step toward downstream analysis, but the feature export still needs one cleanup pass so the saved schema matches the current simplified implementation focus on distribution features.

## Next Step: Standardize And Validate Parcel-Level Jacobian Features

Now that we have parcel masks and a first feature export, the next major step is to make that export the stable analysis-ready input for graph construction.

This means:

- Re-run or refresh the parcel feature export so the output schema matches the current script.
- Confirm histogram range and healthy-baseline settings for the current Jacobian representation.
- Validate parcel counts, label coverage, and feature completeness across the exported table.
- Lock the feature table format that will be used for parcel-by-parcel similarity calculations.

Suggested columns include:

- Parcel ID
- Parent ROI / label
- Raw voxel values reference or serialized values
- Histogram bin values
- Mean or median Jacobian
- Anomaly score
- Voxel count

## Why This Step Is Needed

The project now has an initial parcel-level feature table, but this step is still needed because the feature export must be stabilized before it becomes the canonical input to graph analysis.

This validation and standardization step is needed because:

- Each parcel now needs a consistent quantitative Jacobian profile in a final agreed schema.
- Distribution-based node features still require the full voxel-value distribution, not only a mean.
- Downstream similarity measures such as Jensen-Shannon divergence need a normalized histogram or density estimate for every parcel.
- A stable table with one row per parcel makes later graph construction and quality control much easier.

Without this cleanup step, the pipeline has useful outputs but not yet a fully fixed handoff format for the next analysis stage.

## After Feature Extraction

Once the parcel-level Jacobian feature table has been refreshed and validated, the project can move to parcel-level relationship analysis.

Examples of downstream goals include:

- Building parcel-by-parcel similarity matrices.
- Comparing morphometric similarity between parcels using histogram-based methods.
- Applying anomaly weighting to emphasize strongly abnormal parcels.
- Sparsifying the subject graph and optionally running community detection.

## Open Methodological Question: Matrix Density

One important open issue is how dense the parcel-by-parcel matrix should be.

This matters because the matrix density will directly affect graph structure,
stability of the giant component, and interpretation of downstream community or
similarity analyses.

The current options under consideration are:

- Keep the matrix dense and analyze the fully connected representation.
- Apply thresholding before the graph breaks apart, stopping before the giant component is disrupted.
- Repeat the main analyses across multiple density thresholds and show that the result either remains stable or changes only minimally.

At the moment, the third option is especially useful from a methodological point
of view, because it can demonstrate whether the findings are robust to the
chosen density level rather than depending on a single arbitrary threshold.

## Summary

Current phase:

- Large ROIs have been subdivided into many smaller parcels.
- A first parcel-level Jacobian feature export has been generated.
- The feature-extraction code is being simplified to focus on distribution features and summary statistics.

Next phase:

- Refresh and validate the parcel Jacobian feature export so the saved schema matches the current implementation.
- Start parcel-by-parcel similarity analysis from the standardized feature table.
- Define a principled strategy for parcel-by-parcel matrix density before final graph analysis.
