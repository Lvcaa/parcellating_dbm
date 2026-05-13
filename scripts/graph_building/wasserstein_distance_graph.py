from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "wasserstein_graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one dense Wasserstein similarity matrix from all parcel vectors "
            "stored under a single subject folder. The script reads every "
            "label_* directory for that subject and merges all parcels into one graph."
        )
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        help=(
            "Optional direct path to a subject folder under jacobian_parcel_vectors. "
            "If omitted, --subject-id is used with --input-root."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root directory containing subject parcel-vector folders (default: {DEFAULT_INPUT_ROOT}).",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject folder name under --input-root, for example sub-0091.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help=(
            "Directory where graph outputs will be saved. "
            "If omitted, outputs go to outputs/wasserstein_graphs/<subject_id>."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers to use (default: 4).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "Print a progress update every N completed matrix rows "
            "(default: 25)."
        ),
    )
    return parser.parse_args()


def resolve_subject_folder(input_folder: Path | None, input_root: Path, subject_id: str | None) -> tuple[Path, str]:
    if input_folder is not None:
        subject_folder = input_folder
        resolved_subject_id = subject_folder.name
    else:
        if not subject_id:
            raise ValueError("Provide either --input-folder or --subject-id.")
        subject_folder = input_root / subject_id
        resolved_subject_id = subject_id

    if not subject_folder.exists() or not subject_folder.is_dir():
        raise ValueError(f"Subject folder does not exist: {subject_folder}")

    return subject_folder, resolved_subject_id


def load_subject_parcels(subject_folder: Path) -> tuple[list[str], list[np.ndarray]]:
    label_dirs = sorted(path for path in subject_folder.iterdir() if path.is_dir() and path.name.startswith("label_"))
    if not label_dirs:
        raise ValueError(f"No label_* directories found under {subject_folder}")

    parcel_ids: list[str] = []
    parcel_vectors: list[np.ndarray] = []

    # Iterate through each label directory and load parcel vectors
    for label_dir in label_dirs:
        npy_paths = sorted(label_dir.glob("*.npy"))
        if not npy_paths:
            continue

        # Load each parcel vector and store its ID and vector
        for npy_path in npy_paths:
            vector = np.asarray(np.load(npy_path), dtype=np.float32).reshape(-1)
            if vector.size == 0:
                raise ValueError(f"Parcel vector is empty: {npy_path}")

            parcel_ids.append(f"{label_dir.name}/{npy_path.stem}")
            parcel_vectors.append(vector)

    if not parcel_vectors:
        raise ValueError(f"No parcel .npy files found under {subject_folder}")

    return parcel_ids, parcel_vectors


def allocate_empty_matrix(num_parcels: int) -> np.ndarray:
    matrix = np.zeros((num_parcels, num_parcels), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_row(
    i: int,
    all_parcels: list[np.ndarray],
) -> list[tuple[int, int, float]]:
    wasserstein_similarities: list[tuple[int, int, float]] = []

    # Compute similarities for the upper triangle of the matrix (j > i)
    for j in range(i + 1, len(all_parcels)):
        distance = wasserstein_distance(all_parcels[i], all_parcels[j])

        # Convert distance to similarity using an exponential decay function
        similarity = float(np.exp(-distance))
        wasserstein_similarities.append((i, j, similarity))
    return wasserstein_similarities


def save_parcel_order(parcel_ids: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(parcel_ids) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    subject_folder, subject_id = resolve_subject_folder(
        input_folder=args.input_folder,
        input_root=args.input_root,
        subject_id=args.subject_id,
    )
    parcel_ids, all_parcels = load_subject_parcels(subject_folder)
    num_parcels = len(all_parcels)
    matrix = allocate_empty_matrix(num_parcels)

    print(f"Subject: {subject_id}", flush=True)
    print(f"Input folder: {subject_folder}", flush=True)
    print(f"Loaded {num_parcels} parcel vectors across all labels", flush=True)
    print(
        f"Computing Wasserstein similarities with {args.num_workers} workers; "
        f"progress update every {args.progress_every} rows",
        flush=True,
    )
    start_time = time.time()
    results = Parallel(
        n_jobs=args.num_workers,
        prefer="processes",
        return_as="generator",
    )(
        delayed(compute_row)(i, all_parcels)
        for i in range(num_parcels)
    )

    for completed_rows, worker_results in enumerate(results, start=1):
        if completed_rows % args.progress_every == 0 or completed_rows == num_parcels:
            elapsed = time.time() - start_time
            percent = (completed_rows / num_parcels) * 100 if num_parcels else 100.0
            print(
                f"[progress] completed {completed_rows}/{num_parcels} rows "
                f"({percent:.1f}%) after {elapsed:.1f}s",
                flush=True,
            )
        for i, j, value in worker_results:
            matrix[i, j] = value
            matrix[j, i] = value

    if num_parcels == 1:
        weighted_degree = np.zeros((1,), dtype=np.float64)
    else:
        # Computed as the average similarity to all other parcels (excluding self-similarity)
        weighted_degree = (np.sum(matrix, axis=1) - 1.0) / (num_parcels - 1)

    out_dir = args.output_folder if args.output_folder is not None else DEFAULT_OUTPUT_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mm_matrix = np.memmap(
        out_dir / "adjacency_matrix.dat",
        dtype="float64",
        mode="w+",
        shape=(num_parcels, num_parcels),
    )
    mm_matrix[:] = matrix
    del mm_matrix

    mm_degree = np.memmap(
        out_dir / "weighted_degree.dat",
        dtype="float64",
        mode="w+",
        shape=(num_parcels,),
    )
    mm_degree[:] = weighted_degree
    del mm_degree

    np.save(out_dir / "metadata.npy", np.array([num_parcels], dtype=np.int64))
    save_parcel_order(parcel_ids, out_dir / "parcel_order.txt")

    print(f"Saved dense Wasserstein graph to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
