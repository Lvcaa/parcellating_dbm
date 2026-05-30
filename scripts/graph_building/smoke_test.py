"""
Run a direct Wasserstein graph-construction smoke test from parcel vectors.

The script builds an in-memory similarity matrix, computes weighted degree,
and saves the degree memmap plus parcel order. Use this for small validation
runs before full graph construction.

Usage:
    python scripts/graph_building/smoke_test.py [options]

Parameters:
    --input-dir PATH                Directory containing parcel .npy vectors.
    --output-dir PATH               Default output directory (default: outputs/smoke_test).
    --weighted-degree-output PATH   Explicit weighted-degree memmap path.
    --parcel-order-output PATH      Explicit parcel-order text path.
    --workers INT                   Joblib workers; defaults to half of visible CPUs.

Examples:
    python scripts/graph_building/smoke_test.py
    python scripts/graph_building/smoke_test.py --input-dir outputs/jacobian_parcel_vectors/sub-0091/label_10 --output-dir outputs/smoke_test_sub-0091_label_10 --workers 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.stats import wasserstein_distance

try:
    from joblib import Parallel, delayed
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise SystemExit(
        "This script requires joblib. Install it before running the smoke test."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors" / "label_10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "smoke_test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke test."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a Wasserstein similarity matrix from parcel `.npy` vectors, "
            "compute weighted degree, and save the degree vector as a memory-mapped array."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/graph_building/smoke_test.py\n\n"
            "Custom input/output:\n"
            "  python scripts/graph_building/smoke_test.py "
            "--input-dir outputs/jacobian_parcel_vectors/label_10 "
            "--output-dir outputs/smoke_test_run_01"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing the parcel `.npy` vectors to compare. "
            f"Default: {DEFAULT_INPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where the weighted degree memmap and parcel order will be written. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--weighted-degree-output",
        type=Path,
        default=None,
        help=(
            "Optional explicit path for the weighted degree memory-mapped `.npy` file. "
            "If omitted, it will be written inside --output-dir."
        ),
    )
    parser.add_argument(
        "--parcel-order-output",
        type=Path,
        default=None,
        help=(
            "Optional explicit path for the parcel-order text file. "
            "If omitted, it will be written inside --output-dir."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of joblib workers to use. "
            "If omitted, the script uses half of the available CPU cores."
        ),
    )
    return parser.parse_args()


def resolve_worker_count(requested_workers: int | None) -> int:
    """Use half of the visible CPUs by default, but never fewer than one worker."""
    available_cpus = os.cpu_count() or 1
    default_workers = max(1, available_cpus // 2)

    if requested_workers is None:
        return default_workers
    if requested_workers < 1:
        raise ValueError("--workers must be at least 1")

    return requested_workers


def load_parcel_vectors(input_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """Load all parcel vectors from disk and keep their file stems as parcel IDs."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find all `.npy` files in the input directory and load them into memory as float32 arrays
    npy_paths = sorted(input_dir.glob("*.npy"))
    if not npy_paths:
        raise ValueError(f"No `.npy` parcel vectors found in {input_dir}")

    parcel_ids: list[str] = []
    parcel_vectors: list[np.ndarray] = []
    
    # Iterate over each `.npy` file
    for npy_path in npy_paths:

        # Load the parcel vector and ensure it is a 1D float32 array.
        vector = np.asarray(np.load(npy_path), dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError(f"Parcel vector is empty: {npy_path}")

        # Store the parcel ID (file stem) and the loaded vector for later processing
        parcel_ids.append(npy_path.stem)
        parcel_vectors.append(vector)

    return parcel_ids, parcel_vectors

def split_row_indices(n_rows: int, n_workers: int) -> list[list[int]]:
    """Split row indices into balanced chunks so each worker gets a block of rows."""
    row_chunks = np.array_split(np.arange(n_rows), n_workers)
    return [chunk.tolist() for chunk in row_chunks if len(chunk) > 0]


def compute_worker_rows(row_indices: list[int], parcel_vectors: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """
    Compute Wasserstein-based similarities for one chunk of matrix rows.

    Each worker only computes the upper triangle for its assigned rows. The main
    process later mirrors `(i, j)` into `(j, i)` while merging results.
    """
    worker_results: list[tuple[int, int, float]] = []
    n_parcels = len(parcel_vectors)

    for i in row_indices:
        vector_i = parcel_vectors[i]

        for j in range(i + 1, n_parcels):
            distance = wasserstein_distance(vector_i, parcel_vectors[j])
            similarity = float(np.exp(-distance))
            worker_results.append((i, j, similarity))

    return worker_results


def merge_worker_results(
    matrix: np.ndarray,
    all_results: list[list[tuple[int, int, float]]],
) -> None:
    """
    Merge joblib results into the symmetric similarity matrix.

    Joblib returns one result list per worker. We fill both `(i, j)` and
    `(j, i)` here so the matrix is symmetric.
    """
    for worker_results in all_results:
        for i, j, value in worker_results:
            matrix[i, j] = value
            matrix[j, i] = value


def compute_weighted_degree(matrix: np.ndarray) -> np.ndarray:
    """Compute weighted degree as row sum divided by the number of columns."""
    n_columns = matrix.shape[1]
    return matrix.sum(axis=1, dtype=np.float64) / n_columns


def save_weighted_degree_memmap(weighted_degree: np.ndarray, output_path: Path) -> None:
    """Persist the weighted degree vector as a memory-mapped `.npy` file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    degree_memmap = open_memmap(
        filename=output_path,
        mode="w+",
        dtype=np.float32,
        shape=weighted_degree.shape,
    )
    degree_memmap[:] = weighted_degree.astype(np.float32)
    degree_memmap.flush()


def save_parcel_order(parcel_ids: list[str], output_path: Path) -> None:
    """Write parcel IDs in matrix order so the saved degree vector stays traceable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parcel_ids) + "\n", encoding="utf-8")


def main() -> None:
    """Run the end-to-end smoke test."""
    args = parse_args()
    workers = resolve_worker_count(args.workers)

    weighted_degree_output = (
        args.weighted_degree_output
        if args.weighted_degree_output is not None
        else args.output_dir / "weighted_degree_memmap.npy"
    )
    parcel_order_output = (
        args.parcel_order_output
        if args.parcel_order_output is not None
        else args.output_dir / "parcel_order.txt"
    )

    parcel_ids, parcel_vectors = load_parcel_vectors(args.input_dir)
    n_parcels = len(parcel_vectors)

    print(f"Loaded {n_parcels} parcel vectors from {args.input_dir}", flush=True)
    print(f"Using {workers} joblib workers", flush=True)

    # Preallocate the full zero matrix once in the main process.
    matrix = np.zeros((n_parcels, n_parcels), dtype=np.float32)

    # Split row indices into worker-sized chunks before parallel execution.
    row_chunks = split_row_indices(n_parcels, workers)
    print(f"Split work into {len(row_chunks)} row chunks", flush=True)

    # Each worker returns a list of (i, j, similarity) tuples for its rows.
    all_results = Parallel(n_jobs=workers)(
        delayed(compute_worker_rows)(row_indices, parcel_vectors)
        for row_indices in row_chunks
    )

    # Merge all worker outputs back into the symmetric matrix.
    merge_worker_results(matrix, all_results)

    # Diagonal stays zero because we do not add self-edges in this smoke test.
    weighted_degree = compute_weighted_degree(matrix)

    save_weighted_degree_memmap(weighted_degree, weighted_degree_output)

    # Save the parcel order so the weighted degree vector can be interpreted later.
    save_parcel_order(parcel_ids, parcel_order_output)

    print(f"Matrix shape: {matrix.shape}", flush=True)
    print(f"Weighted degree shape: {weighted_degree.shape}", flush=True)
    print(f"Weighted degree memmap saved to {weighted_degree_output}", flush=True)
    print(f"Parcel order saved to {parcel_order_output}", flush=True)


if __name__ == "__main__":
    main()
