import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.parcellation.run_allowed_parcellations import (
    DEFAULT_AGGREGATED_ROOT,
    DEFAULT_MAX_ESTIMATED_DISTANCE_GB,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PARCEL_SIZE,
    HIGH_MEMORY_LABELS_PATH,
    load_high_memory_labels,
    load_label_counts,
    print_memory_summary,
    run_aggregations,
    run_sub_parcellations,
    validate_memory_estimates,
)
from scripts.preprocessing.inspect_labels import SEGMENTATION_PATH, keep_labels, label_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run sub-parcellation and aggregation for the deferred high-memory labels "
            "listed in the labels file."
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
        "--labels-file",
        type=Path,
        default=HIGH_MEMORY_LABELS_PATH,
        help="Path to the text file containing the high-memory labels to process.",
    )
    parser.add_argument(
        "--max-estimated-distance-gb",
        type=float,
        default=DEFAULT_MAX_ESTIMATED_DISTANCE_GB,
        help=(
            "Warn if a label is estimated to require a distance matrix larger than this many GiB."
        ),
    )
    return parser.parse_args()


def load_requested_labels(labels_file):
    labels = sorted(load_high_memory_labels(labels_file))
    if not labels:
        raise ValueError(f"No labels found in {labels_file}")

    unknown_labels = [label for label in labels if label not in label_dict]
    if unknown_labels:
        raise ValueError(f"Unknown labels in {labels_file}: {unknown_labels}")

    disallowed_labels = [label for label in labels if label not in keep_labels]
    if disallowed_labels:
        raise ValueError(
            "The high-memory runner only accepts labels that are already part of keep_labels: "
            f"{disallowed_labels}"
        )

    return labels


def main():
    args = parse_args()

    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")
    if args.aggregate_workers < 1:
        raise ValueError("--aggregate-workers must be at least 1")
    if args.max_estimated_distance_gb <= 0:
        raise ValueError("--max-estimated-distance-gb must be positive")

    labels = load_requested_labels(args.labels_file)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.aggregated_output_root.mkdir(parents=True, exist_ok=True)
    label_counts = load_label_counts(args.segmentation)

    print(f"Using segmentation: {args.segmentation}", flush=True)
    print(f"Sub-parcel output root: {args.output_root}", flush=True)
    print(f"Aggregated output root: {args.aggregated_output_root}", flush=True)
    print(f"Parcel size: {args.parcel_size}", flush=True)
    print(f"High-memory labels file: {args.labels_file}", flush=True)
    print(f"Labels to process: {labels}", flush=True)
    print_memory_summary(labels, label_counts, args.parcel_size)

    # These labels are intentionally handled in a dedicated high-RAM workflow.
    validate_memory_estimates(
        labels=labels,
        label_counts=label_counts,
        parcel_size=args.parcel_size,
        max_estimated_distance_gb=args.max_estimated_distance_gb,
        allow_high_memory_labels=True,
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

    print("Finished sub-parcellation and aggregation for all requested high-memory labels.", flush=True)


if __name__ == "__main__":
    main()
