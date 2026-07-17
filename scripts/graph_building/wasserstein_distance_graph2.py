"""
Build a dense Wasserstein similarity graph for one subject.

Parcel vectors are sorted and projected onto a common 15-point quantile grid.
Batched NumPy operations compute row blocks in parallel. Weighted degree is
always computed and stored as a float64 memmap. The dense (N, N) adjacency
matrix is only written to disk when --save-matrix true is passed — at scale
(up to ~90k parcels) the full matrix is far larger than the degree vector, so
it is skipped by default. Compute cost still grows quadratically with the
parcel count either way, since every pairwise similarity is computed to
derive weighted degree; only the disk-write cost is avoided.

Usage:
    python scripts/graph_building/wasserstein_distance_graph2.py (--input-folder PATH | --subject-id ID) --sim-formula {1,2} [options]

Parameters:
    --input-folder PATH      Direct subject parcel-vector folder.
    --input-root PATH        Subject-folder root (default: outputs/jacobian_parcel_vectors).
    --subject-id TEXT        Folder name under --input-root when --input-folder is omitted.
    --output-folder PATH     Custom graph output folder.
    --sim-formula {1,2}      Required transform: 1 = exp(-W), 2 = 1/(1+W).
    --save-matrix {true,false}  Write the dense adjacency matrix to disk (default: false).
    --num-workers INT        Worker processes (default: 8).
    --block-size INT         Rows per worker job (default: 500).
    --progress-every INT     Progress interval in completed blocks (default: 10).

Outputs:
    weighted_degree.dat, metadata.npy, metadata.json, parcel_order.txt, and
    adjacency_matrix.dat only when --save-matrix true.

Examples:
    python scripts/graph_building/wasserstein_distance_graph2.py --subject-id sub-0091 --sim-formula 1
    python scripts/graph_building/wasserstein_distance_graph2.py --input-folder outputs/jacobian_parcel_vectors/sub-OAS30999 --sim-formula 2 --num-workers 4 --block-size 200
    python scripts/graph_building/wasserstein_distance_graph2.py --subject-id sub-0091 --sim-formula 1 --save-matrix true
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_raw,
    atomic_write_text,
    load_completion,
    sha256_lines,
    validate_raw_vector,
    write_completion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"

QUANTILE_LEN = 15  # dominant parcel size; shorter parcels are interpolated to this
SIM_FORMULA_EXP = 1
SIM_FORMULA_INV1PW = 2
SIM_FORMULA_LABELS = {
    SIM_FORMULA_EXP: "exp(-W)",
    SIM_FORMULA_INV1PW: "1/(1+W)",
}
SIM_FORMULA_OUTPUT_NAMES = {
    SIM_FORMULA_EXP: "expW",
    SIM_FORMULA_INV1PW: "inv1pW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one dense Wasserstein similarity matrix from all parcel vectors "
            "stored under a single subject folder."
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
            "If omitted, outputs go to a formula-specific folder under outputs/."
        ),
    )
    parser.add_argument(
        "--sim-formula",
        "--sim_formula",
        dest="sim_formula",
        type=int,
        choices=(SIM_FORMULA_EXP, SIM_FORMULA_INV1PW),
        required=True,
        help=(
            "Required distance-to-similarity transform. Use 1 for exp(-W) "
            "or 2 for the script's current 1/(1+W) formula."
        ),
    )
    parser.add_argument(
        "--save-matrix",
        dest="save_matrix",
        type=str.lower,
        choices=("true", "false"),
        default="false",
        help=(
            "Write the dense (N, N) adjacency matrix to disk (default: false). "
            "The matrix is large at scale; weighted degree is always computed "
            "and saved regardless of this flag."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=500,
        help=(
            "Number of rows processed per worker job (default: 500). "
            "Peak RAM per worker ≈ 2 × block_size × N × 4 bytes."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress update every N completed blocks (default: 10).",
    )
    parser.add_argument(
        "--allow-unvalidated-input",
        action="store_true",
        help="Allow parcel folders without complete.json (diagnostic use only).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute a matching completed graph.")
    return parser.parse_args()


def resolve_subject_folder(
    input_folder: Path | None, input_root: Path, subject_id: str | None
) -> tuple[Path, str]:
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


def default_output_root(sim_formula: int) -> Path:
    formula_name = SIM_FORMULA_OUTPUT_NAMES[sim_formula]
    return PROJECT_ROOT / "outputs" / f"wasserstein_graphs_{formula_name}"


def _to_quantile_grid(v: np.ndarray) -> np.ndarray:
    """Sort v and interpolate to QUANTILE_LEN evenly-spaced quantile levels."""
    v_sorted = np.sort(v.astype(np.float32))
    L = len(v_sorted)
    if L == QUANTILE_LEN:
        return v_sorted
    src_q = (np.arange(L, dtype=np.float32) + 0.5) / L
    tgt_q = (np.arange(QUANTILE_LEN, dtype=np.float32) + 0.5) / QUANTILE_LEN
    return np.interp(tgt_q, src_q, v_sorted).astype(np.float32)


def load_subject_parcels(subject_folder: Path) -> tuple[list[str], np.ndarray]:
    """Return parcel IDs and a float32 sorted matrix of shape (N, QUANTILE_LEN)."""
    label_dirs = sorted(
        (p for p in subject_folder.iterdir() if p.is_dir() and p.name.startswith("label_")),
        key=lambda path: int(path.name.removeprefix("label_")),
    )
    if not label_dirs:
        raise ValueError(f"No label_* directories found under {subject_folder}")

    parcel_ids: list[str] = []
    rows: list[np.ndarray] = []

    for label_dir in label_dirs:
        for npy_path in sorted(
            label_dir.glob("roi_*.npy"), key=lambda path: int(path.stem.removeprefix("roi_"))
        ):
            v = np.load(npy_path).reshape(-1)
            if v.size == 0:
                raise ValueError(f"Parcel vector is empty: {npy_path}")
            if not np.isfinite(v).all():
                raise ValueError(f"Parcel vector contains non-finite values: {npy_path}")
            parcel_ids.append(f"{label_dir.name}/{npy_path.stem}")
            rows.append(_to_quantile_grid(v))

    if not rows:
        raise ValueError(f"No parcel .npy files found under {subject_folder}")
    if len(set(parcel_ids)) != len(parcel_ids):
        raise ValueError(f"Duplicate parcel IDs found under {subject_folder}")

    return parcel_ids, np.stack(rows, axis=0)  # (N, QUANTILE_LEN)


def _compute_block(
    i_start: int,
    i_end: int,
    sorted_matrix: np.ndarray,
    sim_formula: int,
) -> tuple[int, int, np.ndarray]:
    """
    Compute the similarity sub-matrix for rows [i_start, i_end).

    W1(i, j) = mean_k |sorted[i, k] - sorted[j, k]|   (exact for equal-length samples)
    similarity = exp(-W1) or 1 / (1 + W1), depending on --sim-formula

    Returns (i_start, i_end, sim_block) where sim_block has shape (i_end-i_start, N).
    """

    # Slice the full sorted matrix to get the block of rows for this job;
    block = sorted_matrix[i_start:i_end]          # (B, L)
    
    N, L = sorted_matrix.shape

    # Allocate a block of distances
    dist = np.zeros((len(block), N), dtype=np.float32)

    # Loop over each quantile level.
    for k in range(L):

        # Compute the absolute difference between the block and the full matrix at this quantile level.
        dist += np.abs(block[:, k : k + 1] - sorted_matrix[:, k])   # (B, N)
    dist /= L

    if sim_formula == SIM_FORMULA_EXP:
        sim = np.exp(-dist).astype(np.float32, copy=False)
    elif sim_formula == SIM_FORMULA_INV1PW:
        sim = 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unsupported similarity formula: {sim_formula}")

    if not np.isfinite(sim).all():
        raise ValueError(f"Non-finite similarities in row block [{i_start}, {i_end})")
    if np.any(sim <= 0.0) or np.any(sim > 1.0):
        raise ValueError(f"Similarities outside (0, 1] in row block [{i_start}, {i_end})")

    # restore exact 1.0 on the diagonal
    for local_i, global_i in enumerate(range(i_start, i_end)):
        sim[local_i, global_i] = 1.0

    return i_start, i_end, sim


def save_parcel_order(parcel_ids: list[str], output_path: Path) -> None:
    atomic_write_text(output_path, "\n".join(parcel_ids) + "\n")


def save_metadata(
    output_path: Path,
    subject_id: str,
    n_parcels: int,
    sim_formula: int,
    save_matrix: bool,
    input_completion: dict,
    parcel_order_hash: str,
) -> None:
    metadata = {
        "subject_id": subject_id,
        "n_parcels": n_parcels,
        "node_feature": "15-point quantile representation of parcel log-Jacobian values",
        "jacobian_type": "log-Jacobian determinant",
        "quantile_count": QUANTILE_LEN,
        "distance": "Wasserstein-1 approximated by mean absolute quantile difference",
        "similarity_formula": SIM_FORMULA_LABELS[sim_formula],
        "adjacency_matrix_saved": save_matrix,
        "adjacency_dtype": "float32",
        "weighted_degree_dtype": "float64",
        "weighted_degree_normalization": "sum of off-diagonal similarities divided by N-1",
        "self_loops_in_adjacency": True,
        "self_loops_in_weighted_degree": False,
        "atlas_id": input_completion.get("atlas_id"),
        "atlas_manifest_sha256": input_completion.get("atlas_manifest_sha256"),
        "parcel_order_sha256": parcel_order_hash,
    }
    atomic_write_json(output_path, metadata)


def read_input_contract(subject_folder: Path, allow_unvalidated: bool) -> tuple[dict, list[str]]:
    completion = load_completion(subject_folder)
    order_path = subject_folder / "parcel_order.txt"
    if completion is None:
        if not allow_unvalidated:
            raise ValueError(
                f"Validated parcel completion marker missing: {subject_folder / 'complete.json'}"
            )
        return {"stage": "UNVALIDATED", "atlas_id": "UNVALIDATED"}, []
    if completion.get("stage") != "parcel_vectors":
        raise ValueError(f"Unexpected input completion stage in {subject_folder}")
    if not order_path.is_file():
        raise ValueError(f"Parcel order missing: {order_path}")
    parcel_ids = order_path.read_text(encoding="utf-8").splitlines()
    if len(parcel_ids) != completion.get("parcel_count"):
        raise ValueError(f"Parcel-order count differs from completion manifest: {subject_folder}")
    if sha256_lines(parcel_ids) != completion.get("parcel_order_sha256"):
        raise ValueError(f"Parcel-order hash differs from completion manifest: {subject_folder}")
    return completion, parcel_ids


def graph_is_complete(
    out_dir: Path,
    *,
    subject_id: str,
    n_parcels: int,
    parcel_order_hash: str,
    input_completion: dict,
    sim_formula: int,
    save_matrix: bool,
) -> bool:
    completion = load_completion(out_dir)
    if completion is None or completion.get("stage") != "wasserstein_graph":
        return False
    expected = {
        "subject_id": subject_id,
        "parcel_count": n_parcels,
        "parcel_order_sha256": parcel_order_hash,
        "atlas_manifest_sha256": input_completion.get("atlas_manifest_sha256"),
        "sim_formula": sim_formula,
        "adjacency_matrix_saved": save_matrix,
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        return False
    try:
        validate_raw_vector(out_dir / "weighted_degree.dat", length=n_parcels)
    except ValueError:
        return False
    if save_matrix:
        matrix = out_dir / "adjacency_matrix.dat"
        if not matrix.is_file() or matrix.stat().st_size != n_parcels * n_parcels * 4:
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1")
    save_matrix = args.save_matrix == "true"

    subject_folder, subject_id = resolve_subject_folder(
        input_folder=args.input_folder,
        input_root=args.input_root,
        subject_id=args.subject_id,
    )

    print(f"Subject:      {subject_id}", flush=True)
    print(f"Input folder: {subject_folder}", flush=True)
    print(f"Save dense adjacency matrix: {save_matrix}", flush=True)
    print(f"Similarity formula: {args.sim_formula} ({SIM_FORMULA_LABELS[args.sim_formula]})", flush=True)

    input_completion, expected_parcel_ids = read_input_contract(
        subject_folder, args.allow_unvalidated_input
    )
    out_dir = (
        args.output_folder
        if args.output_folder is not None
        else default_output_root(args.sim_formula) / subject_id
    )
    if expected_parcel_ids:
        expected_hash = sha256_lines(expected_parcel_ids)
        if not args.force and graph_is_complete(
            out_dir,
            subject_id=subject_id,
            n_parcels=len(expected_parcel_ids),
            parcel_order_hash=expected_hash,
            input_completion=input_completion,
            sim_formula=args.sim_formula,
            save_matrix=save_matrix,
        ):
            print(f"Validated graph output already complete: {out_dir}", flush=True)
            return

    parcel_ids, sorted_matrix = load_subject_parcels(subject_folder)
    N = len(parcel_ids)
    if N < 2:
        raise ValueError("At least two parcels are required to build a graph")
    parcel_order_hash = sha256_lines(parcel_ids)
    if expected_parcel_ids and parcel_ids != expected_parcel_ids:
        raise ValueError("Loaded parcel files differ from the validated input parcel order")
    print(f"Loaded {N} parcels  →  sorted matrix shape {sorted_matrix.shape}", flush=True)

    # Pre-compute row block boundaries
    blocks = [
        (i, min(i + args.block_size, N))
        for i in range(0, N, args.block_size)
    ]
    num_blocks = len(blocks)
    print(
        f"Computing similarities: {num_blocks} blocks of ≤{args.block_size} rows, "
        f"{args.num_workers} workers",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    mm_matrix = None
    temporary_matrix_path = out_dir / f".adjacency_matrix.{os.getpid()}.tmp"
    if save_matrix:
        mm_matrix = np.memmap(
            temporary_matrix_path,
            dtype="float32",
            mode="w+",
            shape=(N, N),
        )
        mm_matrix[:] = 0.0
        np.fill_diagonal(mm_matrix, 1.0)

    weighted_degree = np.zeros(N, dtype=np.float64)

    start_time = time.time()
    results = Parallel(n_jobs=args.num_workers, prefer="processes", return_as="generator")(
        delayed(_compute_block)(i_start, i_end, sorted_matrix, args.sim_formula)
        for i_start, i_end in blocks
    )

    for completed, (i_start, i_end, sim_block) in enumerate(results, start=1):
        if mm_matrix is not None:
            mm_matrix[i_start:i_end, :] = sim_block
            mm_matrix[:, i_start:i_end] = sim_block.T
        # sim_block already spans all N columns for these rows, so weighted
        # degree for this block is complete on its own regardless of whether
        # the full matrix is persisted.
        block_degree = (sim_block.sum(axis=1, dtype=np.float64) - 1.0) / (N - 1)
        if not np.isfinite(block_degree).all():
            raise ValueError(f"Non-finite weighted degree in row block [{i_start}, {i_end})")
        weighted_degree[i_start:i_end] = block_degree

        if completed % args.progress_every == 0 or completed == num_blocks:
            elapsed = time.time() - start_time
            pct = completed / num_blocks * 100
            print(
                f"[progress] {completed}/{num_blocks} blocks ({pct:.1f}%) — {elapsed:.1f}s",
                flush=True,
            )

    if mm_matrix is not None:
        mm_matrix.flush()
        del mm_matrix
        os.replace(temporary_matrix_path, out_dir / "adjacency_matrix.dat")

    atomic_write_raw(out_dir / "weighted_degree.dat", weighted_degree, "float64")
    atomic_save_npy(out_dir / "metadata.npy", np.array([N], dtype=np.int64))
    save_metadata(
        out_dir / "metadata.json",
        subject_id,
        N,
        args.sim_formula,
        save_matrix,
        input_completion,
        parcel_order_hash,
    )
    save_parcel_order(parcel_ids, out_dir / "parcel_order.txt")
    write_completion(out_dir, {
        "schema_version": 1,
        "stage": "wasserstein_graph",
        "subject_id": subject_id,
        "parcel_count": N,
        "parcel_order_sha256": parcel_order_hash,
        "atlas_id": input_completion.get("atlas_id"),
        "atlas_manifest_sha256": input_completion.get("atlas_manifest_sha256"),
        "sim_formula": args.sim_formula,
        "similarity_formula": SIM_FORMULA_LABELS[args.sim_formula],
        "adjacency_matrix_saved": save_matrix,
        "weighted_degree_dtype": "float64",
    })

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s  →  {out_dir}", flush=True)


if __name__ == "__main__":
    main()
