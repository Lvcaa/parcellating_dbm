"""
Run sub-parcellation and aggregation for deferred high-memory ROI labels.

Labels are read from docs/high_memory_labels.txt, validated against
const.KEEP_LABELS, and processed after memory estimates are printed.

Usage:
    python scripts/parcellation/separate_cases/run_high_ram_labels.py [options]

Parameters:
    --segmentation PATH              Input segmentation image.
    --output-root PATH               ROI parcel root (default: const.ROIS_DIR).
    --aggregated-output-root PATH    Aggregated NIfTI root.
    --parcel-size INT                Target voxels per parcel (default: 15).
    --skip-neighbor-check            Skip connectivity diagnostics.
    --aggregate-workers INT          Workers passed to aggregation (default: 1).
    --labels-file PATH               Text file with one label per line.
    --max-estimated-distance-gb N    Warning threshold in GiB (default: 8.0).

Examples:
    python scripts/parcellation/separate_cases/run_high_ram_labels.py --skip-neighbor-check
    python scripts/parcellation/separate_cases/run_high_ram_labels.py --aggregate-workers 4 --skip-neighbor-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))        # scripts/ → for const
sys.path.insert(0, str(_here.parent))            # separate_cases/ → for sibling import

import const
from run_allowed_parcellations import (
    DEFAULT_MAX_ESTIMATED_DISTANCE_GB,
    DEFAULT_PARCEL_SIZE,
    load_high_memory_labels,
    load_label_counts,
    print_memory_summary,
    run_aggregations,
    run_sub_parcellations,
    validate_memory_estimates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sub-parcellation and aggregation for deferred high-memory labels."
    )
    parser.add_argument("--segmentation", type=Path, default=const.SEGMENTATION_PATH)
    parser.add_argument("--output-root", type=Path, default=const.ROIS_DIR)
    parser.add_argument("--aggregated-output-root", type=Path, default=const.AGGREGATED_ROIS_DIR)
    parser.add_argument("--parcel-size", type=int, default=DEFAULT_PARCEL_SIZE)
    parser.add_argument("--skip-neighbor-check", action="store_true")
    parser.add_argument("--aggregate-workers", type=int, default=1)
    parser.add_argument("--labels-file", type=Path, default=const.HIGH_MEMORY_LABELS_PATH)
    parser.add_argument(
        "--max-estimated-distance-gb", type=float, default=DEFAULT_MAX_ESTIMATED_DISTANCE_GB
    )
    return parser.parse_args()


def load_requested_labels(labels_file: Path) -> list[int]:
    labels = sorted(load_high_memory_labels(labels_file))
    if not labels:
        raise ValueError(f"No labels found in {labels_file}")
    unknown = [l for l in labels if l not in const.LABEL_DICT]
    if unknown:
        raise ValueError(f"Unknown labels in {labels_file}: {unknown}")
    disallowed = [l for l in labels if l not in const.KEEP_LABELS]
    if disallowed:
        raise ValueError(f"Labels not in KEEP_LABELS: {disallowed}")
    return labels


def main() -> None:
    args = parse_args()
    labels = load_requested_labels(args.labels_file)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.aggregated_output_root.mkdir(parents=True, exist_ok=True)
    label_counts = load_label_counts(args.segmentation)

    print(f"Labels to process: {labels}", flush=True)
    print_memory_summary(labels, label_counts, args.parcel_size)
    # allow_high_memory_labels=True: warn instead of aborting
    validate_memory_estimates(
        labels, label_counts, args.parcel_size,
        args.max_estimated_distance_gb, allow_high_memory_labels=True,
    )
    run_sub_parcellations(
        labels, args.segmentation, args.output_root,
        args.parcel_size, args.skip_neighbor_check,
    )
    run_aggregations(labels, args.output_root, args.aggregated_output_root, args.aggregate_workers)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
