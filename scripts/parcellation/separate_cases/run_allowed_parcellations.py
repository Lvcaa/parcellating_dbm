"""
Run sub-parcellation and aggregation for retained, non-deferred ROI labels.

The script estimates distance-matrix RAM before invoking the sibling
parcellation and aggregation utilities as subprocesses.

Usage:
    python scripts/parcellation/separate_cases/run_allowed_parcellations.py [options]

Parameters:
    --segmentation PATH              Input segmentation image.
    --output-root PATH               ROI parcel root (default: const.ROIS_DIR).
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

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import const

DEFAULT_PARCEL_SIZE = 15
DEFAULT_MAX_ESTIMATED_DISTANCE_GB = 8.0

EXCLUDED_LABELS: set[int] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sub-parcellation for the allowed labels from const.KEEP_LABELS "
            "and then aggregate each label back into a single NIfTI volume."
        )
    )
    parser.add_argument("--segmentation", type=Path, default=const.SEGMENTATION_PATH)
    parser.add_argument("--output-root", type=Path, default=const.ROIS_DIR)
    parser.add_argument("--aggregated-output-root", type=Path, default=const.AGGREGATED_ROIS_DIR)
    parser.add_argument("--parcel-size", type=int, default=DEFAULT_PARCEL_SIZE)
    parser.add_argument("--skip-neighbor-check", action="store_true")
    parser.add_argument("--aggregate-workers", type=int, default=1)
    parser.add_argument(
        "--max-estimated-distance-gb", type=float, default=DEFAULT_MAX_ESTIMATED_DISTANCE_GB
    )
    parser.add_argument("--allow-high-memory-labels", action="store_true")
    return parser.parse_args()


def label_filename(label: int) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", const.LABEL_DICT[label].lower()).strip("_")
    return f"{label}_{name}.nii.gz"


def load_high_memory_labels(file_path: Path) -> set[int]:
    """Read integer label IDs from a text file (one per line, # comments ignored)."""
    if not file_path.exists():
        return set()
    labels = set()
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            labels.add(int(line))
    return labels


def allowed_labels(deferred: set[int]) -> list[int]:
    return [l for l in const.KEEP_LABELS if l not in EXCLUDED_LABELS and l not in deferred]


def load_label_counts(segmentation_path: Path) -> dict[int, int]:
    data = np.asanyarray(nib.load(str(segmentation_path)).dataobj)
    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        if np.allclose(data, rounded, atol=1e-6):
            data = rounded.astype(np.int64, copy=False)
    labels, counts = np.unique(data, return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}


def estimate_distance_matrix_gb(n_voxels: int, parcel_size: int) -> float:
    n_clusters = int(np.ceil(n_voxels / parcel_size))
    return (n_voxels * n_clusters * parcel_size * 8) / (1024 ** 3)


def print_memory_summary(labels: list[int], label_counts: dict[int, int], parcel_size: int) -> None:
    print("Estimated distance-matrix usage per label:", flush=True)
    for label in labels:
        n = label_counts.get(label, 0)
        gb = estimate_distance_matrix_gb(n, parcel_size)
        print(f"  Label {label} ({const.LABEL_DICT[label]}): {n} voxels, {gb:.3f} GiB", flush=True)


def validate_memory_estimates(
    labels: list[int],
    label_counts: dict[int, int],
    parcel_size: int,
    max_estimated_distance_gb: float,
    allow_high_memory_labels: bool,
) -> None:
    oversized = [
        (l, label_counts.get(l, 0), estimate_distance_matrix_gb(label_counts.get(l, 0), parcel_size))
        for l in labels
        if estimate_distance_matrix_gb(label_counts.get(l, 0), parcel_size) > max_estimated_distance_gb
    ]
    if not oversized:
        return
    lines = [
        f"Memory preflight failed (threshold: {max_estimated_distance_gb:.1f} GiB).",
        "Labels above threshold:",
    ] + [f"  {l} ({const.LABEL_DICT[l]}): {n} voxels, {gb:.3f} GiB" for l, n, gb in oversized]
    if allow_high_memory_labels:
        print("Warning: " + "\n".join(lines), flush=True)
        return
    raise RuntimeError("\n".join(lines))


def run_command(command: list) -> None:
    print("Running:", " ".join(str(p) for p in command), flush=True)
    subprocess.run(command, check=True)


def run_sub_parcellations(
    labels: list[int],
    segmentation_path: Path,
    output_root: Path,
    parcel_size: int,
    skip_neighbor_check: bool,
) -> None:
    # Use the connected region-growing implementation for every label. The
    # older balanced-clustering implementation can produce disconnected masks.
    script = const.SCRIPTS_DIR / "parcellation" / "separate_cases" / "second_roi_test.py"
    for label in labels:
        print(f"Processing label {label} ({const.LABEL_DICT[label]})", flush=True)
        cmd = [
            sys.executable, str(script),
            "--roi-label", str(label),
            "--parcel-size", str(parcel_size),
            "--segmentation", str(segmentation_path),
            "--output-root", str(output_root),
        ]
        if skip_neighbor_check:
            cmd.append("--skip-neighbor-check")
        run_command(cmd)


def run_aggregations(
    labels: list[int],
    output_root: Path,
    aggregated_output_root: Path,
    aggregate_workers: int,
) -> None:
    script = const.SCRIPTS_DIR / "parcellation" / "aggregate_parcels.py"
    aggregated_output_root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        print(f"Aggregating label {label} ({const.LABEL_DICT[label]})", flush=True)
        run_command([
            sys.executable, str(script),
            str(output_root / str(label)),
            str(aggregated_output_root / label_filename(label)),
            "--workers", str(aggregate_workers),
        ])


def main() -> None:
    args = parse_args()
    deferred = load_high_memory_labels(const.HIGH_MEMORY_LABELS_PATH)
    labels = allowed_labels(deferred)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.aggregated_output_root.mkdir(parents=True, exist_ok=True)
    label_counts = load_label_counts(args.segmentation)

    print(f"Labels to process: {labels}", flush=True)
    if deferred:
        print(f"Deferred: {sorted(deferred)}", flush=True)
    print_memory_summary(labels, label_counts, args.parcel_size)
    validate_memory_estimates(
        labels, label_counts, args.parcel_size,
        args.max_estimated_distance_gb, args.allow_high_memory_labels,
    )
    run_sub_parcellations(labels, args.segmentation, args.output_root, args.parcel_size, args.skip_neighbor_check)
    run_aggregations(labels, args.output_root, args.aggregated_output_root, args.aggregate_workers)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
