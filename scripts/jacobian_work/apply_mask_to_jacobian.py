import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np


def find_roi_files(rois_dir: Path) -> list[Path]:
    # First look for nested label folders, then fall back to flat structure
    roi_paths = sorted(
        Path(path)
        for path in glob(str(rois_dir / "*" / "roi_*.nii.gz"))
    )
    if not roi_paths:
        roi_paths = sorted(Path(path) for path in glob(str(rois_dir / "roi_*.nii.gz")))

    if not roi_paths:
        raise ValueError(f"No roi_*.nii.gz files found under {rois_dir}")

    return roi_paths


def compute_histogram(values: np.ndarray, bins: int, hist_min: float, hist_max: float) -> np.ndarray:
    """
    Compute a normalized histogram of the given values within the specified range and bins.
    This will be used to capture the distribution of Jacobian values within each parcel.
    That will ultimately be used to understand how much the parcel's deformation pattern deviates from healthy baselines, 
    and to provide a richer representation of the parcel's characteristics beyond simple summary statistics.
    """
    # Use numpy histogram to compute the distribution of values within the specified range and bins.
    hist, _ = np.histogram(values, bins=bins, range=(hist_min, hist_max), density=False)
    hist = hist.astype(np.float64)

    # Normalize the histogram to sum to 1, converting it to a probability distribution.
    total = hist.sum()

    # Avoid division by zero in case of empty histogram
    if total > 0:
        hist /= total
    return hist


def spatial_descriptors(mask_indices: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute simple spatial descriptors for the given mask indices.
    A histogram of Jacobian values tells you about the values inside the parcel,
    but not much about the parcel’s geometry. 
    These two numbers are cheap little descriptors that give some spatial context
    """
    # Compute the centroid of the mask indices
    centroid = mask_indices.mean(axis=0, dtype=np.float64)

    # Compute the spatial spread as the root mean square distance of the mask indices from the centroid
    centered = mask_indices - centroid

    # The spatial spread is a measure of how dispersed the mask indices are around the centroid.
    spatial_spread = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    return centroid, spatial_spread


def process_roi_file(
    roi_path: Path,
    rois_dir: str,
    jacobian_path: str,
    bins: int,
    hist_min: float,
    hist_max: float,
    healthy_baseline: float,
) -> tuple[dict, np.ndarray]:

"""
Process a single ROI file to extract Jacobian features. 
This function will be called in parallel for each ROI file.
"""
    jac_img = nib.load(jacobian_path)
    jac_data = jac_img.get_fdata()

    roi_img = nib.load(str(roi_path))
    if roi_img.shape != jac_img.shape:
        raise ValueError(
            f"Shape mismatch for {roi_path}: expected {jac_img.shape}, got {roi_img.shape}"
        )

    if not np.allclose(roi_img.affine, jac_img.affine):
        raise ValueError(f"Affine mismatch for {roi_path}")

    # Create a boolean mask where the ROI image has values greater than 0. 
    # This identifies the voxels that belong to the parcel defined by the ROI.
    mask = roi_img.get_fdata() > 0

    # Get the voxel indices where the mask is True.
    # This will be used for spatial descriptors and to extract values from the Jacobian data.
    voxel_indices = np.argwhere(mask)
    if voxel_indices.size == 0:
        raise ValueError(f"ROI {roi_path} is empty")

    # Extract the Jacobian values at the masked voxel locations and convert to float32 for memory efficiency.
    values = jac_data[mask].astype(np.float32, copy=False)
    
    # Compute summary statistics, histogram, and spatial descriptors for the masked Jacobian values.

    # The mean and median provide central tendency measures of the Jacobian values within the parcel
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))

    # Anomaly score quantifies how much the parcel's median Jacobian
    # value deviates from a healthy baseline.
    anomaly_score = float(abs(median_value - healthy_baseline))

    # Compute the normalized histogram of Jacobian values within the parcel.
    # Captures the distribution of values, which can provide insights into the parcel deformation characteristics.
    hist = compute_histogram(values, bins=bins, hist_min=hist_min, hist_max=hist_max)

    # Compute spatial descriptors (centroid and spatial spread) for the parcel based on the voxel indices.
    centroid, spatial_spread = spatial_descriptors(voxel_indices)

    relative_roi_path = roi_path.relative_to(Path(rois_dir))
    parent_label = relative_roi_path.parts[0] if len(relative_roi_path.parts) > 1 else ""
    parcel_id = roi_path.stem.replace(".nii", "")
    raw_values_key = "__".join(relative_roi_path.parts).replace(".nii.gz", "")

    row = {
        "parcel_id": parcel_id,
        "parent_label": parent_label,
        "source_roi_path": str(roi_path),
        "voxel_count": int(values.size),
        "mean_jacobian": mean_value,
        "median_jacobian": median_value,
        "anomaly_score": anomaly_score,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "spatial_spread": spatial_spread,
        "raw_values_key": raw_values_key,
    }

    # Add histogram bins to the row with keys like "hist_bin_000", "hist_bin_001", etc.
    for idx, value in enumerate(hist):
        row[f"hist_bin_{idx:03d}"] = float(value)

    return row, values


def write_rows_csv(output_csv: Path, rows: list[dict], bins: int) -> None:

    """
    Write the list of row dictionaries to a CSV file with the appropriate fieldnames, 
    including histogram bins. 
    
    This will create a structured feature table where each row corresponds
    to a parcel, and columns include summary statistics, spatial descriptors, anomaly score, and 
    histogram values.
    """
    
    fieldnames = [
        "parcel_id",
        "parent_label",
        "source_roi_path",
        "voxel_count",
        "mean_jacobian",
        "median_jacobian",
        "anomaly_score",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "spatial_spread",
        "raw_values_key",
    ]

    # Add histogram bin fieldnames to the list, matching the keys used in the row dictionaries.
    fieldnames.extend(f"hist_bin_{idx:03d}" for idx in range(bins))

    # Write the rows to the CSV file using a DictWriter, ensuring that the header includes all fieldnames.
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_table(
    rois_dir: Path,
    jacobian_path: Path,
    output_dir: Path,
    bins: int,
    hist_min: float | None,
    hist_max: float | None,
    healthy_baseline: float,
    workers: int,
) -> None:
    roi_files = find_roi_files(rois_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jac_img = nib.load(str(jacobian_path))
    jac_data = jac_img.get_fdata()

    # To determine the histogram range, we need to find the global minimum and maximum 
    # finite Jacobian values across the entire image.
    finite_jac = jac_data[np.isfinite(jac_data)]
    if finite_jac.size == 0:
        raise ValueError(f"No finite Jacobian values found in {jacobian_path}")

    if hist_min is None:
        hist_min = float(np.min(finite_jac))
    if hist_max is None:
        hist_max = float(np.max(finite_jac))
    if hist_min >= hist_max:
        raise ValueError("Histogram range is invalid: hist_min must be smaller than hist_max")

    print(f"Found {len(roi_files)} ROI files", flush=True)
    print(f"Jacobian image: {jacobian_path}", flush=True)
    print(f"Output dir:     {output_dir}", flush=True)
    print(f"Histogram bins: {bins}", flush=True)
    print(f"Histogram min:  {hist_min}", flush=True)
    print(f"Histogram max:  {hist_max}", flush=True)
    print(f"Healthy baseline: {healthy_baseline}", flush=True)

    rows: list[dict] = []
    raw_values_dict: dict[str, np.ndarray] = {}

    if workers <= 1 or len(roi_files) == 1:
        print("Running feature extraction in serial mode", flush=True)

        # Process each ROI file sequentially, extracting features and storing raw values in a dictionary.
        for idx, roi_path in enumerate(roi_files, start=1):

            # The process_roi_file function extracts features for the given ROI file and returns 
            # a row dictionary and the raw Jacobian values for that parcel.
            row, values = process_roi_file(
                roi_path=roi_path,
                rois_dir=str(rois_dir),
                jacobian_path=str(jacobian_path),
                bins=bins,
                hist_min=hist_min,
                hist_max=hist_max,
                healthy_baseline=healthy_baseline,
            )
            rows.append(row)

            # Store the raw Jacobian values in a dictionary with a key that uniquely identifies the parcel
            raw_values_dict[row["raw_values_key"]] = values
            print(f"[progress] {idx}/{len(roi_files)} parcels done ({row['parcel_id']})", flush=True)
    else:

        # Same logic but parallelized using ProcessPoolExecutor. 
        # Each ROI file is processed in a separate worker process
        n_workers = min(workers, len(roi_files))
        print(f"Running feature extraction with {n_workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_roi_file,
                    roi_path,
                    str(rois_dir),
                    str(jacobian_path),
                    bins,
                    hist_min,
                    hist_max,
                    healthy_baseline,
                ): roi_path
                for roi_path in roi_files
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                row, values = future.result()
                rows.append(row)
                raw_values_dict[row["raw_values_key"]] = values
                print(
                    f"[progress] {completed}/{len(roi_files)} parcels done ({row['parcel_id']})",
                    flush=True,
                )

    rows.sort(key=lambda row: row["parcel_id"])

    output_csv = output_dir / "parcel_jacobian_features.csv"
    output_npz = output_dir / "parcel_raw_values.npz"
    metadata_path = output_dir / "feature_metadata.csv"

    write_rows_csv(output_csv, rows, bins)
    np.savez_compressed(output_npz, **raw_values_dict)

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["histogram_bins", "histogram_min", "histogram_max", "healthy_baseline"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "histogram_bins": bins,
                "histogram_min": hist_min,
                "histogram_max": hist_max,
                "healthy_baseline": healthy_baseline,
            }
        )

    print(f"Wrote parcel feature table to {output_csv}", flush=True)
    print(f"Wrote raw voxel values to {output_npz}", flush=True)
    print(f"Wrote histogram metadata to {metadata_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract parcel-level Jacobian features from ROI masks. "
            "For each parcel, save raw voxel values, a normalized histogram, "
            "central Jacobian summaries, anomaly score, voxel count, and "
            "simple spatial descriptors."
        )
    )
    parser.add_argument(
        "--rois_dir",
        type=Path,
        default=Path("outputs/rois"),
        help="Directory containing ROI parcel masks. Supports nested label folders.",
    )
    parser.add_argument(
        "--jacobian",
        type=Path,
        default=Path("data/subToMNI_relative_logJac_MNI.nii.gz"),
        help="Path to the Jacobian image.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/jacobian_features"),
        help="Directory where the parcel feature table and raw values will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=64,
        help="Number of histogram bins for each parcel distribution.",
    )
    parser.add_argument(
        "--hist_min",
        type=float,
        default=None,
        help="Lower bound for histogram bins. Defaults to the global min finite Jacobian value.",
    )
    parser.add_argument(
        "--hist_max",
        type=float,
        default=None,
        help="Upper bound for histogram bins. Defaults to the global max finite Jacobian value.",
    )
    parser.add_argument(
        "--healthy_baseline",
        type=float,
        default=0.0,
        help=(
            "Healthy reference value used for anomaly scoring. "
            "Use 0.0 for log-Jacobian images and 1.0 for raw Jacobian determinants."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use. One ROI file is processed per task.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.bins < 1:
        raise ValueError("--bins must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    build_feature_table(
        rois_dir=args.rois_dir,
        jacobian_path=args.jacobian,
        output_dir=args.output_dir,
        bins=args.bins,
        hist_min=args.hist_min,
        hist_max=args.hist_max,
        healthy_baseline=args.healthy_baseline,
        workers=args.workers,
    )
