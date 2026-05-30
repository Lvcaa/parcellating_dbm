"""
Run sub-parcellation and aggregation for retained, non-deferred ROI labels.

The script estimates distance-matrix RAM before invoking the sibling
parcellation and aggregation utilities as subprocesses.

Usage:
    python scripts/parcellation/separate_cases/run_allowed_parcellations.py [options]

Parameters:
    --segmentation PATH              Input segmentation image.
    --output-root PATH               ROI parcel root (default: outputs/rois).
    --aggregated-output-root PATH    Aggregated NIfTI root.
    --parcel-size INT                Target voxels per parcel (default: 15).
    --skip-neighbor-check            Skip connectivity diagnostics.
    --aggregate-workers INT          Workers passed to aggregation (default: 1).
    --max-estimated-distance-gb N    RAM preflight limit in GiB (default: 8.0).
    --allow-high-memory-labels       Continue when a preflight warning is raised.

Examples:
    python scripts/parcellation/separate_cases/run_allowed_parcellations.py
    python scripts/parcellation/separate_cases/run_allowed_parcellations.py --parcel-size 10 --aggregate-workers 4
    python scripts/parcellation/separate_cases/run_allowed_parcellations.py --max-estimated-distance-gb 12 --allow-high-memory-labels
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing.inspect_labels import SEGMENTATION_PATH, keep_labels, label_dict


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rois"
DEFAULT_AGGREGATED_ROOT = PROJECT_ROOT / "outputs" / "aggregated_parcellations"
DEFAULT_PARCEL_SIZE = 15
DEFAULT_MAX_ESTIMATED_DISTANCE_GB = 8.0
HIGH_MEMORY_LABELS_PATH = PROJECT_ROOT / "scripts" / "parcellation" / "high_memory_labels.txt"

# Excluded since we are preprocessing them separately due to high matrices volume
EXCLUDED_LABELS = {3, 42}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run sub-parcellation for the allowed labels from inspect_labels.py "
            "and then aggregate each label back into a single NIfTI volume."
        )
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=SEGMENTATION_PATH,
        help="Path to the input segmentation image.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where ROI-specific parcel folders will be written.",
    )
    parser.add_argument(
        "--aggregated-output-root",
        type=Path,
        default=DEFAULT_AGGREGATED_ROOT,
        help="Directory where aggregated per-label NIfTI files will be written.",
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        default=DEFAULT_PARCEL_SIZE,
        help="Target number of voxels per sub-parcel.",
    )
    parser.add_argument(
        "--skip-neighbor-check",
        action="store_true",
        help="Skip the local connectivity diagnostic during sub-parcellation.",
    )
    parser.add_argument(
        "--aggregate-workers",
        type=int,
        default=1,
        help="Number of workers to pass to aggregate_parcels.py.",
    )
    parser.add_argument(
        "--max-estimated-distance-gb",
        type=float,
        default=DEFAULT_MAX_ESTIMATED_DISTANCE_GB,
        help=(
            "Abort before running if an ROI is estimated to require a distance matrix "
            "larger than this many GiB."
        ),
    )
    parser.add_argument(
        "--allow-high-memory-labels",
        action="store_true",
        help="Run even if the memory preflight estimates that some labels are too large.",
    )
    return parser.parse_args()


def label_filename(label):
    label_name = label_dict[label].lower()
    label_name = re.sub(r"[^a-z0-9]+", "_", label_name).strip("_")
    return f"{label}_{label_name}.nii.gz"


def load_high_memory_labels(file_path):
    labels = set()
    if not file_path.exists():
        return labels

    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        labels.add(int(line))

    return labels


def allowed_labels(deferred_labels):
    excluded_labels = EXCLUDED_LABELS | deferred_labels
    return [label for label in keep_labels if label not in excluded_labels]


def load_label_counts(segmentation_path):
    img = nib.load(str(segmentation_path))
    data = np.asanyarray(img.dataobj)

    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        if np.allclose(data, rounded, atol=1e-6):
            data = rounded.astype(np.int64, copy=False)

    labels, counts = np.unique(data, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def estimate_distance_matrix_gb(n_voxels, parcel_size):
    n_clusters = int(np.ceil(n_voxels / parcel_size))
    n_centers = n_clusters * parcel_size
    return (n_voxels * n_centers * 8) / (1024 ** 3)


def print_memory_summary(labels, label_counts, parcel_size):
    print("Estimated distance-matrix usage per label:", flush=True)
    for label in labels:
        n_voxels = label_counts.get(label, 0)
        estimated_gb = estimate_distance_matrix_gb(n_voxels, parcel_size)
        print(
            f"  Label {label} ({label_dict[label]}): "
            f"{n_voxels} voxels, estimated matrix {estimated_gb:.3f} GiB",
            flush=True,
        )


def validate_memory_estimates(
    labels,
    label_counts,
    parcel_size,
    max_estimated_distance_gb,
    allow_high_memory_labels,
):
    oversized_labels = []

    for label in labels:
        n_voxels = label_counts.get(label, 0)
        estimated_gb = estimate_distance_matrix_gb(n_voxels, parcel_size)
        if estimated_gb > max_estimated_distance_gb:
            oversized_labels.append((label, n_voxels, estimated_gb))

    if not oversized_labels:
        return

    message_lines = [
        "Memory preflight failed.",
        (
            "The current sub-parcellation algorithm builds a distance matrix whose size "
            "grows roughly quadratically with the voxel count of each label."
        ),
        f"Threshold: {max_estimated_distance_gb:.3f} GiB",
        "Labels above threshold:",
    ]
    for label, n_voxels, estimated_gb in oversized_labels:
        message_lines.append(
            f"  {label} ({label_dict[label]}): {n_voxels} voxels, "
            f"estimated matrix {estimated_gb:.3f} GiB"
        )
    message_lines.append(
        "Use a larger --max-estimated-distance-gb, preprocess these labels separately, "
        "or pass --allow-high-memory-labels if you really want to force the run."
    )

    if allow_high_memory_labels:
        print("\n".join(["Warning:"] + message_lines), flush=True)
        return

    raise RuntimeError("\n".join(message_lines))


def run_command(command):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_sub_parcellations(labels, segmentation_path, output_root, parcel_size, skip_neighbor_check):
    script_path = PROJECT_ROOT / "scripts" / "parcellation" / "sub_parcels_equal_size.py"

    for label in labels:
        command = [
            sys.executable,
            str(script_path),
            "--roi-label",
            str(label),
            "--parcel-size",
            str(parcel_size),
            "--segmentation",
            str(segmentation_path),
            "--output-root",
            str(output_root),
        ]
        if skip_neighbor_check:
            command.append("--skip-neighbor-check")

        print(f"Processing label {label} ({label_dict[label]})", flush=True)
        run_command(command)


def run_aggregations(labels, output_root, aggregated_output_root, aggregate_workers):
    script_path = PROJECT_ROOT / "scripts" / "parcellation" / "aggregate_parcels.py"
    aggregated_output_root.mkdir(parents=True, exist_ok=True)

    for label in labels:
        parcels_dir = output_root / str(label)
        output_path = aggregated_output_root / label_filename(label)
        command = [
            sys.executable,
            str(script_path),
            str(parcels_dir),
            str(output_path),
            "--workers",
            str(aggregate_workers),
        ]

        print(f"Aggregating label {label} ({label_dict[label]})", flush=True)
        run_command(command)


def main():
    args = parse_args()

    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")
    if args.aggregate_workers < 1:
        raise ValueError("--aggregate-workers must be at least 1")
    if args.max_estimated_distance_gb <= 0:
        raise ValueError("--max-estimated-distance-gb must be positive")

    deferred_labels = load_high_memory_labels(HIGH_MEMORY_LABELS_PATH)
    labels = allowed_labels(deferred_labels)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.aggregated_output_root.mkdir(parents=True, exist_ok=True)
    label_counts = load_label_counts(args.segmentation)

    print(f"Using segmentation: {args.segmentation}", flush=True)
    print(f"Sub-parcel output root: {args.output_root}", flush=True)
    print(f"Aggregated output root: {args.aggregated_output_root}", flush=True)
    print(f"Parcel size: {args.parcel_size}", flush=True)
    print(f"Labels to process: {labels}", flush=True)
    if deferred_labels:
        print(
            f"Deferred high-memory labels from {HIGH_MEMORY_LABELS_PATH}: "
            f"{sorted(deferred_labels)}",
            flush=True,
        )
    print_memory_summary(labels, label_counts, args.parcel_size)
    validate_memory_estimates(
        labels=labels,
        label_counts=label_counts,
        parcel_size=args.parcel_size,
        max_estimated_distance_gb=args.max_estimated_distance_gb,
        allow_high_memory_labels=args.allow_high_memory_labels,
    )

    run_sub_parcellations(
        labels=labels,
        segmentation_path=args.segmentation,
        output_root=args.output_root,
        parcel_size=args.parcel_size,
        skip_neighbor_check=args.skip_neighbor_check,
    )

    run_aggregations(
        labels=labels,
        output_root=args.output_root,
        aggregated_output_root=args.aggregated_output_root,
        aggregate_workers=args.aggregate_workers,
    )

    print("Finished sub-parcellation and aggregation for all allowed labels.", flush=True)


if __name__ == "__main__":
    main()
