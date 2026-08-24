"""Build thresholded parcel-similarity weighted degrees for listed subjects."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, save_npz

from wasserstein_distance_graph2 import SIM_FORMULA_EXP, _compute_block, _to_quantile_grid
from pipeline_integrity import (
    atomic_write_json,
    atomic_write_raw,
    atomic_write_text,
    sha256_lines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"
DEFAULT_SUBJECTS_FILE = PROJECT_ROOT / "outputs" / "subjects.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "thresholded_weighted_degree"
PERCENTILES = [10, 20, 40, 80]
DEFAULT_N_PROCESSES = 100
BLOCK_SIZE = 500


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_N_PROCESSES)
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help=(
            "Also reconstruct and save the symmetric sparse adjacency matrix. "
            "By default only weighted degree is calculated and saved."
        ),
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    return args

def read_list_subjects(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing subjects file: {file_path}")

    subjects = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    duplicates = sorted(
        subject for subject, count in Counter(subjects).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate subjects in {file_path}: {', '.join(duplicates)}"
        )

    invalid = [subject for subject in subjects if Path(subject).name != subject]
    if invalid:
        raise ValueError(
            f"Invalid subject names in {file_path}: {', '.join(invalid)}"
        )

    return subjects

def compute_upper_rows(i_start, i_end, quantile_matrix):
    _, _, similarity_block = _compute_block(
        i_start, i_end, quantile_matrix, SIM_FORMULA_EXP
    )
    return [
        similarity_block[local_i, i + 1:].copy()
        for local_i, i in enumerate(range(i_start, i_end))
    ]


def compute_thresholded_degree(upper_rows, threshold, n_parcels, build_graph=False):
    """Calculate weighted degree, optionally reconstructing the sparse graph."""
    weighted_sum = np.zeros(n_parcels, dtype=np.float64)
    retained_edges = 0

    if build_graph:
        rows, cols, weights = [], [], []

    for i, upper_values in enumerate(upper_rows):
        keep = upper_values >= threshold
        js = np.flatnonzero(keep) + i + 1
        kept_values = upper_values[keep].astype(np.float32, copy=False)

        weighted_sum[i] += kept_values.sum(dtype=np.float64)
        weighted_sum[js] += kept_values
        retained_edges += js.size

        if build_graph:
            js = js.astype(np.int32, copy=False)
            i_values = np.full(js.size, i, dtype=np.int32)
            rows.extend((i_values, js))
            cols.extend((js, i_values))
            weights.extend((kept_values, kept_values))

    graph = None
    if build_graph:
        graph = coo_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_parcels, n_parcels),
            dtype=np.float32,
        ).tocsr()

    weighted_degree = weighted_sum / (n_parcels - 1)
    return weighted_degree, retained_edges, graph


def save_sparse_graph(path, graph):
    """Atomically save a SciPy sparse matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.npz")
    try:
        save_npz(temporary_path, graph)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    input_root = args.input_root.expanduser()
    subjects_file = args.subjects_file.expanduser()
    output_root = args.output_root.expanduser()
    subjects = read_list_subjects(subjects_file)
    if not subjects:
        raise ValueError(f"No subjects listed in: {subjects_file}")
    print(f"Subjects file: {subjects_file} ({len(subjects)} subjects)")
    print(f"Save sparse graphs: {args.save_graph}")

    # Loop over each subject in list
    for subject in subjects:
        subject_folder = input_root / subject
        if not subject_folder.is_dir():
            raise ValueError(f"Missing subject folder: {subject_folder}")

        print(f"Processing subject: {subject}")

        # Read the parcel vectors and offsets for the subject
        data = np.fromfile(subject_folder / "parcel_vectors.dat", dtype=np.float32)
        offsets = np.fromfile(subject_folder / "parcel_offsets.dat", dtype=np.int64)

        # Compute the quantile matri using offset indices to slice data correctly
        quantile_matrix = np.stack(
            [_to_quantile_grid(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:])]
        )
        n_parcels = quantile_matrix.shape[0]
        if n_parcels < 2:
            raise ValueError(f"At least two parcels are required: {subject_folder}")
        parcel_order_text = (subject_folder / "parcel_order.txt").read_text(encoding="utf-8")
        parcel_ids = parcel_order_text.splitlines()
        if len(parcel_ids) != n_parcels:
            raise ValueError(
                f"Parcel-order count {len(parcel_ids)} differs from data count "
                f"{n_parcels}: {subject_folder}"
            )
        parcel_order_hash = sha256_lines(parcel_ids)
        print("Quantile matrix shape:", quantile_matrix.shape)

        # Prepare the blocks list for parallel processing, each block contains a range of parcel indices to compute the upper rows of the similarity matrix.
        blocks = [
            (start, min(start + BLOCK_SIZE, n_parcels - 1))
            for start in range(0, n_parcels - 1, BLOCK_SIZE)
        ]
        # Store the computed upper rows in sequence of block rows in a parallel manner, where each block is processed in a separate process.
        block_rows = Parallel(n_jobs=args.num_workers, prefer="processes")(
            delayed(compute_upper_rows)(i_start, i_end, quantile_matrix)
            for i_start, i_end in blocks
        )

        # Concatenate all the upper rows from the blocks into a single array to compute the threshold based on the specified percentiles.
        upper_rows = [row for rows_in_block in block_rows for row in rows_in_block]

        all_upper_values = np.concatenate(upper_rows)

        # Calculate one weighted-degree vector for each percentile. Sparse graph
        # reconstruction is optional because it is not needed for row sums.
        for percentile in PERCENTILES:
            threshold = np.percentile(all_upper_values, percentile)
            print(f"{percentile}th-percentile similarity threshold: {threshold:.6f}")

            weighted_degree, retained_edges, graph = compute_thresholded_degree(
                upper_rows,
                threshold,
                n_parcels,
                build_graph=args.save_graph,
            )

            directed_entries = 2 * retained_edges
            retained_density = directed_entries / (n_parcels * (n_parcels - 1))
            output_dir = output_root / f"percentile_{percentile:02d}" / subject
            if graph is not None:
                save_sparse_graph(output_dir / "adjacency_matrix.npz", graph)
            atomic_write_raw(
                output_dir / "weighted_degree.dat", weighted_degree, "float64"
            )
            atomic_write_text(
                output_dir / "parcel_order.txt",
                parcel_order_text,
            )
            atomic_write_json(
                output_dir / "metadata.json",
                {
                    "subject_id": subject,
                    "method": "thresholded_wasserstein",
                    "n_parcels": n_parcels,
                    "distance": "Wasserstein-1",
                    "similarity_formula": "exp(-W)",
                    "percentile": percentile,
                    "similarity_threshold": float(threshold),
                    "retained_density": retained_density,
                    "stored_undirected_edges": retained_edges,
                    "adjacency_matrix_saved": args.save_graph,
                    "adjacency_dtype": "float32",
                    "adjacency_format": "scipy_csr_npz" if args.save_graph else None,
                    "weighted_degree_dtype": "float64",
                    "weighted_degree_normalization": (
                        "sum of retained off-diagonal similarities divided by N-1"
                    ),
                    "self_loops_in_weighted_degree": False,
                    "parcel_order_sha256": parcel_order_hash,
                },
            )

            if graph is not None:
                print(f"Saved sparse graph: {output_dir / 'adjacency_matrix.npz'}")
            print(f"Retained directed entries: {directed_entries}")
            print(f"Retained undirected edges: {retained_edges}")
            print(f"Saved weighted degree: {output_dir}")
            del graph



if __name__ == "__main__":
    main()
